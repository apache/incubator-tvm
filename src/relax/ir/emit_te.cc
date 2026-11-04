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
 * \file relax/src/ir/emit_te.cc
 */
#include "./emit_te.h"

#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/type.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm {
namespace relax {

// Pattern A (RM): auto-default repr from reflection for RXPlaceholderOpObj.

TVM_FFI_STATIC_INIT_BLOCK() { RXPlaceholderOpObj::RegisterReflection(); }

te::Tensor TETensor(Expr value, ffi::Map<tvm::Var, PrimExpr> tir_var_map, std::string name) {
  auto n = ffi::make_object<RXPlaceholderOpObj>(value);
  n->name = name;

  // If the value is a constant, it might come as an argument of EmitTE and thus its shape and
  // checked-type might not be properly set. In this case we set the shape and dtype of the returned
  // TE tensor.
  if (const auto* constant = value.as<GenericConstObj>()) {
    n->dtype = PrimType(constant->value.cast<ffi::Tensor>().dtype());

    int ndim = constant->value.cast<ffi::Tensor>().ndim();
    ffi::Shape shape_tuple = constant->value.cast<ffi::Tensor>().shape();
    ffi::Array<PrimExpr> shape;
    shape.reserve(ndim);
    for (int i = 0; i < ndim; ++i) {
      shape.push_back(IntImm::Int64(shape_tuple[i]));
    }
    n->shape = std::move(shape);
    return te::PlaceholderOp(ffi::UnsafeInit{}, std::move(n)).output(0);
  }
  TVM_FFI_ICHECK(!value->ty.as<MissingType>().has_value())
      << "value must be normalized and contain Type";
  auto* tensor_ty = GetTypeAs<TensorTypeObj>(value);
  TVM_FFI_ICHECK(tensor_ty) << "Value must be a tensor";
  auto* shape_expr = tensor_ty->shape.as<ShapeExprObj>();
  TVM_FFI_CHECK(shape_expr, ValueError)
      << "Expression does not have an known symbolic shape, please consider use "
         "match_cast "
      << "to constrain the shape before passing into te_tensor";
  auto f_substitute =
      [&tir_var_map](const tvm::Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (auto repl = tir_var_map.Get(var)) return ffi::Any(*std::move(repl));
    return ffi::Unchanged();
  };
  n->shape = shape_expr->values.Map([&f_substitute](const PrimExpr& expr) {
    return ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(expr, f_substitute)
        .as_or_throw<PrimExpr>();
  });
  n->dtype = tensor_ty->dtype.value();
  return te::PlaceholderOp(ffi::UnsafeInit{}, std::move(n)).output(0);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.TETensor", TETensor);
}

}  // namespace relax
}  // namespace tvm
