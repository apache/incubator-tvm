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

/*!
 * \file tvm/relax/distributed/transform/legalize_redistribute.cc
 * \brief Pass for legalizing redistribute op to ccl op.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/distributed/axis_group_graph.h>
#include <tvm/relax/distributed/transform.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op/ccl.h>
#include <tvm/relax/op/distributed.h>
#include <tvm/tirx/stmt_functor.h>

#include "../../../s_tir/schedule/transform.h"
#include "../../op/ccl/ccl.h"
#include "../../op/distributed/distributed.h"

namespace tvm {
namespace relax {
namespace distributed {

class RedistributeLegalizer : public ExprMutator {
 public:
  static IRModule LegalizeRedistribute(IRModule mod) {
    return RedistributeLegalizer(mod).Legalize();
  }

 private:
  explicit RedistributeLegalizer(IRModule mod) : ExprMutator(mod) {}

  IRModule Legalize() {
    auto mod = builder_->GetContextIRModule();
    for (const auto& [gv, base_func] : mod->functions) {
      const auto* func_ = base_func.as<FunctionObj>();
      if (func_ == nullptr) {
        continue;
      }
      Expr new_func_body = VisitExpr(func_->body);
      auto new_func = ffi::make_object<FunctionObj>(*func_);
      new_func->body = new_func_body;
      builder_->UpdateFunction(gv, Function(new_func));
    }
    return builder_->GetContextIRModule();
  }
  using ExprMutator::VisitExpr_;
  Expr VisitExpr_(const CallObj* op) final {
    Call call = ExprMutator::VisitExpr_(op).as_or_throw<Call>();
    static Op redistribute_op = Op::Get("relax.dist.redistribute");
    if (call->op.same_as(redistribute_op)) {
      const auto* attrs = call->attrs.as<DistributionAttrs>();
      TVM_FFI_ICHECK(attrs);
      const auto* input_ty = call->args[0]->ty.as<DTensorTypeObj>();
      TVM_FFI_ICHECK(input_ty);
      // As the first step, we only support redistribute in the same device mesh,
      // and the device mesh must be 1d
      // todo: extend the ccl ops so that it can support 2d device mesh, and different sharding
      // dimension
      TVM_FFI_ICHECK(ffi::StructuralEqual()(input_ty->device_mesh, attrs->device_mesh));
      TVM_FFI_ICHECK(input_ty->device_mesh->shape.size() == 1);
      // only support "S[x]"-> "R" and "R" -> "S[x]"
      PlacementSpec input_spec = input_ty->placement->dim_specs[0];
      PlacementSpec output_spec = attrs->placement->dim_specs[0];
      if (input_spec->kind == PlacementSpecKind::kReplica &&
          output_spec->kind == PlacementSpecKind::kReplica) {
        // "R" -> "R"
        return call->args[0];
      } else if (input_spec->kind == PlacementSpecKind::kSharding &&
                 output_spec->kind == PlacementSpecKind::kSharding) {
        // "S[x]" -> "S[y]"
        if (input_spec->axis != output_spec->axis) {
          TVM_FFI_THROW(InternalError) << "AlltoAll not implemented yet";
        } else {
          return call->args[0];
        }
      } else if (input_spec->kind == PlacementSpecKind::kSharding &&
                 output_spec->kind == PlacementSpecKind::kReplica) {
        // "S[x]" -> "R"
        TVM_FFI_THROW(InternalError) << "Allgather not implemented yet";
      } else if (input_spec->kind == PlacementSpecKind::kReplica &&
                 output_spec->kind == PlacementSpecKind::kSharding) {
        // "R" -> "S[x]"
        return redistribute_replica_to_shard(call->args[0], attrs->device_mesh->shape[0],
                                             output_spec->axis);
      } else {
        TVM_FFI_THROW(InternalError) << "Unsupported redistribute op";
      }
    }
    return call;
  }
};

namespace transform {

Pass LegalizeRedistribute() {
  auto pass_func = [=](IRModule m, PassContext pc) {
    return RedistributeLegalizer::LegalizeRedistribute(m);
  };
  return CreateModulePass(pass_func, 1, "LegalizeRedistribute");
}
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.distributed.transform.LegalizeRedistribute", LegalizeRedistribute);
}
}  // namespace transform

}  // namespace distributed
}  // namespace relax
}  // namespace tvm
