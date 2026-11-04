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
namespace tvm {
namespace relax {
namespace distributed {

bool TypeCompatibleWithDistIR(ffi::Array<Type> tys) {
  bool compatible = true;
  for (const auto& ty : tys) {
    if (const auto* tuple_ty = ty.as<TupleTypeObj>()) {
      compatible &= TypeCompatibleWithDistIR(tuple_ty->fields);
    } else {
      compatible &= !ty->IsInstance<TensorTypeObj>();
    }
  }
  return compatible;
}

bool TypeCompatibleWithRelax(ffi::Array<Type> tys) {
  bool compatible = true;
  for (const auto& ty : tys) {
    if (const auto* tuple_ty = ty.as<TupleTypeObj>()) {
      compatible &= TypeCompatibleWithRelax(tuple_ty->fields);
    } else {
      compatible &= !ty->IsInstance<DTensorTypeObj>();
    }
  }
  return compatible;
}
bool IsDistIRFunc(Function func) {
  ffi::Array<Type> param_tys;
  for (const auto& param : func->params) {
    TVM_FFI_ICHECK(!param->ty.as<MissingType>().has_value());
    param_tys.push_back(param->ty.as_or_throw<Type>());
  }
  bool compatible_with_dist_ir = TypeCompatibleWithDistIR(param_tys);
  bool compatible_with_relax = TypeCompatibleWithRelax(param_tys);
  if (compatible_with_relax) {
    return false;
  } else if (compatible_with_dist_ir && !compatible_with_relax) {
    return true;
  } else {
    TVM_FFI_THROW(InternalError) << "mixed use of DTensor and Tensor in: " << func;
  }
}

bool IsShardingAnnotatedFunc(Function func) {
  bool has_annotate_sharding = false;
  PostOrderVisit(func, [&has_annotate_sharding](const Expr& e) {
    const CallObj* call = e.as<CallObj>();
    if (!call) {
      return;
    }
    static Op annotate_sharding_op = Op::Get("relax.dist.annotate_sharding");
    if (call->op.same_as(annotate_sharding_op)) {
      has_annotate_sharding = true;
    }
  });
  return has_annotate_sharding;
}
}  // namespace distributed
}  // namespace relax
}  // namespace tvm
