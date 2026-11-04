/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "utils.h"

#include <tvm/ffi/cast.h>

#include <vector>
namespace tvm {
namespace relax {
namespace distributed {

ffi::Array<distributed::DTensorType> GetInputDTensorType(const Call& call,
                                                         const BlockBuilder& ctx) {
  Op op = call->op.as_or_throw<Op>();
  ffi::Array<Expr> args = GetCallArgs(call);
  ffi::Array<distributed::DTensorType> input_tensor_ty;
  input_tensor_ty.reserve(args.size());
  for (const Expr& arg : args) {
    const auto* ty = GetTypeAs<distributed::DTensorTypeObj>(arg);
    if (ty != nullptr) {
      input_tensor_ty.push_back(ffi::GetRef<distributed::DTensorType>(ty));
    }
  }
  return input_tensor_ty;
}

Type InferShardingSpec(const Call& call, const BlockBuilder& ctx, const Type& orig_output_ty,
                       distributed::FBuildAxisGraph f_build_graph) {
  ffi::Array<distributed::DTensorType> input_dtensor_tys = GetInputDTensorType(call, ctx);
  for (int i = 1; i < static_cast<int>(input_dtensor_tys.size()); i++) {
    TVM_FFI_ICHECK(ffi::StructuralEqual()(input_dtensor_tys[0]->device_mesh,
                                          input_dtensor_tys[i]->device_mesh));
  }
  distributed::DeviceMesh device_mesh = input_dtensor_tys[0]->device_mesh;
  Var output_var("output", orig_output_ty);
  distributed::AxisGroupGraph axis_group_graph;
  f_build_graph(output_var, call, &axis_group_graph);
  ffi::Array<Expr> args = GetCallArgs(call);
  int n_input_var = input_dtensor_tys.size();
  for (int i = 0; i < n_input_var; i++) {
    distributed::DTensorType dtensor_ty = input_dtensor_tys[i];
    Expr input_tensor = args[i];
    for (int j = 0; j < static_cast<int>(device_mesh->shape.size()); j++) {
      distributed::PlacementSpec placement_spec = dtensor_ty->placement->dim_specs[j];
      if (placement_spec->kind != distributed::PlacementSpecKind::kSharding) {
        continue;
      }
      axis_group_graph.AddSrcShardingPoint({input_tensor.get(), placement_spec->axis},
                                           {dtensor_ty->device_mesh, j});
    }
  }
  axis_group_graph.PropagateShardingSpec();
  ffi::Array<TensorType> orig_output_tensor_tys;
  if (const auto* tensor_ty = orig_output_ty.as<TensorTypeObj>()) {
    orig_output_tensor_tys.push_back(ffi::GetRef<TensorType>(tensor_ty));
  } else {
    const auto* tuple_ty = orig_output_ty.as<TupleTypeObj>();
    TVM_FFI_ICHECK(tuple_ty);
    for (const auto& ty : tuple_ty->fields) {
      orig_output_tensor_tys.push_back(ty.as_or_throw<TensorType>());
    }
  }
  ffi::Array<Type> new_output_dtensor_tys;
  for (int idx = 0; idx < static_cast<int>(orig_output_tensor_tys.size()); idx++) {
    ffi::Array<distributed::PlacementSpec> output_placement_specs(
        std::vector<distributed::PlacementSpec>(device_mesh->shape.size(),
                                                distributed::PlacementSpec::Replica()));
    for (int i = 0; i < orig_output_tensor_tys[idx]->ndim; i++) {
      std::optional<distributed::AxisShardingSpec> sharding_spec;
      bool has_sharding_spec;
      std::tie(sharding_spec, has_sharding_spec) =
          axis_group_graph.GetAxisShardingSpec({output_var.get(), i, idx});
      if (has_sharding_spec) {
        output_placement_specs.Set(sharding_spec->second, distributed::PlacementSpec::Sharding(i));
      }
    }
    new_output_dtensor_tys.push_back(DTensorType(orig_output_tensor_tys[idx], device_mesh,
                                                 distributed::Placement(output_placement_specs)));
  }

  return new_output_dtensor_tys.size() == 1 ? new_output_dtensor_tys[0]
                                            : TupleType(new_output_dtensor_tys);
}

}  // namespace distributed
}  // namespace relax
}  // namespace tvm
