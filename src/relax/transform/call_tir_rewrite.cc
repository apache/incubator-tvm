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
 * \file src/relax/transform/call_tir_rewrite.cc
 * \brief Perform explicit tensor allocation for call_tir,
 *        call_tir_inplace, and call_dps_packed.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op/op.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/tirx/op/memory.h>

#include <algorithm>

#include "utils.h"

namespace tvm {
namespace relax {

// ==================
// CallTIRMutator
// Perform explicit tensor allocation for call_tir, call_tir_inplace, or call_dps_packed.
// Example:
// lv0: Tensor(n, m) = rx.call_tir(func, (x), (n, m), dtype="float32")
// -->
// gv0 = rx.call("relax.builtin.alloc_tensor", [n, m], dtype="float32")
// rx.call_packed(func, x, gv0)

class CallTIRMutator : public ExprMutator {
 public:
  explicit CallTIRMutator(const IRModule& mod) : ExprMutator(mod), mod_(std::move(mod)) {}

  IRModule Run() {
    for (const auto& [gv, func] : mod_->functions) {
      if (func->IsInstance<FunctionObj>()) {
        auto updated_func = this->VisitExpr(func).as_or_throw<Function>();
        builder_->UpdateFunction(gv, updated_func);
      }
    }
    return builder_->GetContextIRModule();
  }

 private:
  using ExprMutator::VisitExpr_;
  Expr VisitExpr_(const CallObj* call) override {
    // post-order mutation
    Expr expr = VisitExprPostOrder_(call);
    call = expr.as<CallObj>();

    static const Op call_tir_op = Op::Get("relax.call_tir");
    static const Op call_tir_inplace_op = Op::Get("relax.call_tir_inplace");
    static const Op call_dps_packed_op = Op::Get("relax.call_dps_packed");
    static const Op alloc_tensor_op = Op::Get("relax.builtin.alloc_tensor");
    if (call->op.same_as(call_tir_op) || call->op.same_as(call_tir_inplace_op) ||
        call->op.same_as(call_dps_packed_op)) {
      bool is_inplace = call->op.same_as(call_tir_inplace_op);
      const auto* inplace_attrs = call->attrs.as<CallTIRInplaceAttrs>();
      ffi::Array<Expr> outs;
      ffi::Optional<TupleType> tuple_output_type;
      if (const auto& tensor_ty = MatchType<TensorType>(expr)) {
        // single output case
        const TensorType& output_ty = tensor_ty.value();
        if (!is_inplace) {
          outs.push_back(AllocateOutputTensor(output_ty, alloc_tensor_op));
        } else {
          // if there is only one output, it must be an in-place argument, but check anyway
          TVM_FFI_ICHECK(inplace_attrs->inplace_indices[0] != -1)
              << "If calling call_tir_inplace and there is one output, its in-place index must not"
                 " be -1.";
          outs.push_back(
              call->args[1].as_or_throw<Tuple>()->fields[inplace_attrs->inplace_indices[0]]);
        }
      } else if (const auto& tuple_ty = MatchType<TupleType>(expr)) {
        // multiple output case
        const TupleType& output_ty = tuple_ty.value();
        tuple_output_type = output_ty;
        bool has_nested_tuple =
            std::any_of(output_ty->fields.begin(), output_ty->fields.end(),
                        [](const Type& field) { return field->IsInstance<TupleTypeObj>(); });

        if (has_nested_tuple) {
          TVM_FFI_ICHECK(!is_inplace)
              << "call_tir_inplace does not support nested tuple output types";
          FlattenAndAllocateOutputs(output_ty, alloc_tensor_op, &outs);
        } else {
          for (size_t i = 0; i < output_ty->fields.size(); ++i) {
            const auto& field = output_ty->fields[i];

            TVM_FFI_ICHECK(field->IsInstance<TensorTypeObj>())
                << "call_tir expects Tuple of TensorType, but got " << field
                << " as an element of TupleType";
            const auto& field_tensor = field.as_or_throw<TensorType>();

            if (!is_inplace || inplace_attrs->inplace_indices[i] == -1) {
              outs.push_back(AllocateOutputTensor(field_tensor, alloc_tensor_op));
            } else {
              outs.push_back(
                  call->args[1].as_or_throw<Tuple>()->fields[inplace_attrs->inplace_indices[i]]);
            }
          }
        }
      } else {
        TVM_FFI_THROW(TypeError) << "The type of call_tir expects to be TensorType or "
                                    "TupleType, but got"
                                 << expr->ty;
      }

      ffi::Array<Expr> args;
      if (call->args[1].as<TupleObj>()) {
        args = call->args[1].as_or_throw<Tuple>()->fields;
        // for call_tir_inplace, don't reinsert in-place args, only the newly allocated ones
        if (!is_inplace) {
          args.insert(args.end(), outs.begin(), outs.end());
        } else {
          for (size_t i = 0; i < outs.size(); i++) {
            if (inplace_attrs->inplace_indices[i] == -1) {
              args.push_back(outs[i]);
            }
          }
        }
        Expr invocation = call->op.same_as(call_dps_packed_op)
                              ? Expr(Call(Type::Missing(), call->args[0], args))
                              : Expr(Call(Type::Missing(), Op::Get("relax.call_tir_packed"),
                                          {call->args[0], Tuple(args)}));
        builder_->Emit(invocation, "_");
      } else {
        if (!is_inplace) {
          args = outs;
          args.insert(args.begin(), call->args[1]);
        } else {
          args.push_back(call->args[1]);
        }
        Expr invocation = call->op.same_as(call_dps_packed_op)
                              ? Expr(Call(Type::Missing(), call->args[0], args))
                              : Expr(Call(Type::Missing(), Op::Get("relax.call_tir_packed"),
                                          {call->args[0], Tuple(args)}));
        builder_->Emit(invocation, "_");
      }

      if (tuple_output_type.has_value()) {
        size_t index = 0;
        Expr output = RebuildOutputTuple(tuple_output_type.value(), outs, &index);
        TVM_FFI_ICHECK_EQ(index, outs.size());
        return output;
      }
      if (outs.size() == 1) return outs[0];
      return std::move(Tuple(outs));
    }

    return ffi::GetRef<Expr>(call);
  }

  Expr AllocateOutputTensor(const TensorType& tensor_ty, const Op& alloc_tensor_op) {
    TVM_FFI_ICHECK(tensor_ty->shape.has_value())
        << "call_tir expects all TensorType has shape, but got " << tensor_ty;

    int dev_index = 0;
    runtime::MemoryScope scope;
    if (tensor_ty->vdevice.has_value()) {
      dev_index = GetDeviceIndex(mod_, tensor_ty->vdevice.value());
      scope = tensor_ty->vdevice.value()->memory_scope;
    } else {
      dev_index = GetDeviceIndexByScope(mod_, scope);
    }

    return builder_->Emit(Call(Type::Missing(), alloc_tensor_op,
                               {tensor_ty->shape.value().as_or_throw<ShapeExpr>(),
                                DataTypeImm(tensor_ty->dtype.value()->dtype),
                                IntImm::Int64(dev_index), StringImm(scope.name())},
                               std::nullopt, {tensor_ty}),
                          "alloc");
  }

  void FlattenAndAllocateOutputs(const Type& type, const Op& alloc_tensor_op,
                                 ffi::Array<Expr>* outs) {
    if (const auto* tensor_ty = type.as<TensorTypeObj>()) {
      outs->push_back(AllocateOutputTensor(ffi::GetRef<TensorType>(tensor_ty), alloc_tensor_op));
      return;
    }
    if (const auto* tuple_ty = type.as<TupleTypeObj>()) {
      for (const Type& field : tuple_ty->fields) {
        FlattenAndAllocateOutputs(field, alloc_tensor_op, outs);
      }
      return;
    }
    TVM_FFI_THROW(TypeError) << "call_tir expects nested tuple outputs to contain only "
                                "TensorType, but got "
                             << type;
  }

  Expr RebuildOutputTuple(const Type& type, const ffi::Array<Expr>& outs, size_t* index) {
    if (type->IsInstance<TensorTypeObj>()) {
      TVM_FFI_ICHECK_LT(*index, outs.size());
      return outs[(*index)++];
    }
    if (const auto* tuple_ty = type.as<TupleTypeObj>()) {
      ffi::Array<Expr> fields;
      fields.reserve(tuple_ty->fields.size());
      for (const Type& field : tuple_ty->fields) {
        fields.push_back(RebuildOutputTuple(field, outs, index));
      }
      return Tuple(std::move(fields));
    }
    TVM_FFI_THROW(TypeError) << "call_tir expects nested tuple outputs to contain only "
                                "TensorType, but got "
                             << type;
  }

  /*! \brief The context IRModule. */
  IRModule mod_;
};

namespace transform {

Pass CallTIRRewrite() {
  auto pass_func = [=](IRModule mod, PassContext pc) { return CallTIRMutator(mod).Run(); };
  return CreateModulePass(/*pass_function=*/pass_func,
                          /*opt_level=*/0,
                          /*pass_name=*/"CallTIRRewrite");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.CallTIRRewrite", CallTIRRewrite);
}

}  // namespace transform

}  // namespace relax
}  // namespace tvm
