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
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/distributed/type.h>
#include <tvm/relax/op/op.h>
#include <tvm/relax/script/ir_builder/ir.h>
#include <tvm/relax/type.h>

#include "./utils.h"

namespace tvm {
namespace relax {
Expr MakeCallTIRDist(Expr func, Tuple args, ffi::Array<distributed::DTensorType> out_ty_list) {
  for (const distributed::DTensorType& ty : out_ty_list) {
    const auto* shape = ty->tensor_ty->shape.as<ShapeExprObj>();
    TVM_FFI_ICHECK(shape != nullptr)
        << "out_ty of call_tir should have defined ShapeExpr as shape. "
           "However, one given type information is "
        << ty;
  }

  Type out_ty = Type::Missing();
  if (out_ty_list.size() == 1) {
    out_ty = out_ty_list[0];
  } else {
    out_ty = TupleType({out_ty_list.begin(), out_ty_list.end()});
  }

  static const Op op = Op::Get("relax.call_tir");
  return Call(Type::Missing(), op, {func, args}, {}, {out_ty});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.ir_builder.relax.distributed.call_tir_dist", MakeCallTIRDist);
}

}  // namespace relax
}  // namespace tvm
