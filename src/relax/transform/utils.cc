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

#include <tvm/relax/analysis.h>

namespace tvm {
namespace relax {

bool IsScalarTensor(const Type& ty) {
  if (!ty->IsInstance<TensorTypeObj>()) {
    return false;
  }
  TensorType tensor_ty = ty.as_or_throw<TensorType>();
  if (!tensor_ty->shape.has_value() || !tensor_ty->shape.value()->IsInstance<ShapeExprObj>()) {
    return false;
  }
  return tensor_ty->shape.as<ShapeExprObj>()->values.size() == 0;
}

bool IsScalarTensor(const Expr& expr) { return IsScalarTensor(GetType(expr)); }

bool IsNestedTensor(const Type& ty) {
  return IsNestedTensorConditioned(ty, [](const TensorType& ty) { return true; });
}

bool IsNestedTensor(const Expr& expr) { return IsNestedTensor(GetType(expr)); }

Function ComposeFunctions(Function func_a, Function func_b) {
  ffi::Array<Binding> bindings;

  Var func_a_output("func_a_output", func_a->ret_ty);

  bindings.push_back(VarBinding(func_a_output, func_a->body));

  auto func_a_outputs = [&]() -> ffi::Array<Expr> {
    if (auto func_a_output_tuple = func_a->ret_ty.as<TupleTypeObj>()) {
      ffi::Array<Expr> outputs;
      for (size_t i = 0; i < func_a_output_tuple->fields.size(); i++) {
        outputs.push_back(TupleGetItem(func_a_output, i));
      }
      return outputs;
    } else {
      return {func_a_output};
    }
  }();

  if (func_b->params.size() == 1 && func_b->params[0]->ty.as<TupleTypeObj>()) {
    // Special case where the output of the first function is a tuple
    // that should be provided as-is to the second function, and
    // should not be unpacked into individual elements.
    auto param = func_b->params[0];
    bindings.push_back(MatchCast(param, func_a_output, GetType(param)));
  } else {
    TVM_FFI_CHECK_EQ(func_a_outputs.size(), func_b->params.size(), ValueError)
        << "Cannot compose functions together.  "
        << "First function produces " << func_a_outputs.size() << " values, "
        << "but second function expects " << func_b->params.size() << " parameters as input";
    for (size_t i = 0; i < func_a_outputs.size(); i++) {
      auto param = func_b->params[i];
      bindings.push_back(MatchCast(param, func_a_outputs[i], GetType(param)));
    }
  }

  auto new_body = SeqExpr({BindingBlock(bindings)}, func_b->body);

  auto new_function = Function(func_a->params, new_body, func_b->ret_ty,
                               func_a->is_pure && func_b->is_pure, func_a->attrs);

  new_function = CopyWithNewVars(new_function);
  new_function = CanonicalizeBindings(new_function).as_or_throw<Function>();
  new_function = RemoveAllUnused(new_function).as_or_throw<Function>();

  return new_function;
}

}  // namespace relax
}  // namespace tvm
