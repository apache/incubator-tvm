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
 * \file relax/src/ir/emit_te.h
 * \brief Tensor expression extension in Relax.
 */
#ifndef TVM_RELAX_IR_EMIT_TE_H_
#define TVM_RELAX_IR_EMIT_TE_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/expr.h>
#include <tvm/te/operation.h>

#include <string>

namespace tvm {
namespace relax {

/*!
 * \brief A placeholder op that represents a relax expression.
 */
class RXPlaceholderOpObj : public te::PlaceholderOpObj {
 public:
  explicit RXPlaceholderOpObj(Expr value) : value(std::move(value)) {}
  explicit RXPlaceholderOpObj(ffi::UnsafeInit) : value(ffi::UnsafeInit{}) {}

  /*! \brief The relax expression. */
  Expr value;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<RXPlaceholderOpObj>().def_ro("value", &RXPlaceholderOpObj::value);
  }

  // FFI system configuration for structural equality and hashing
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.TEPlaceholderOp", RXPlaceholderOpObj,
                                    te::PlaceholderOpObj);
};

/*!
 * \brief Create a TE tensor from relax expression, with TIR variables in the
 * tensor shape substituted by the given mapping.
 * \param value The relax expression, which is required to have TensorType.
 * \param tir_var_map The mapping to substitute the TIR variables appeared in the
 * shape of the input Expr.
 * \param name The name of the created tensor.
 */
te::Tensor TETensor(Expr value, ffi::Map<tvm::Var, PrimExpr> tir_var_map, std::string name);

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_IR_EMIT_TE_H_
