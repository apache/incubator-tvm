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
 * \file unsafe_select_rewrite.cc
 * \brief Rewrite uinsafe select expression.
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/transform.h>
#include <tvm/tirx/op/attr.h>
#include <tvm/tirx/op/memory.h>

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

// For now, rewrite unsafe select expression to if_then_else
// TODO(tqchen) pattern matching to support masked load
class UnsafeExprDetector : public tvm::ExprFunctor<bool(const Expr& n)> {
 public:
  // select itself is always considered safe if condition is safe
  // Because we will issue guard to make sure it is.
  bool Dispatch_(const prim::SelectObj* op) { return Dispatch(op->condition); }
  bool Dispatch_(const CallObj* op) {
    if (op->op.same_as(prim::if_then_else_op())) {
      return Dispatch(op->args[0].as_or_throw<PrimExpr>());
    } else if (op->op.same_as(tirx::address_of_op())) {
      if (const auto* load = op->args[0].as<TensorLoadObj>()) {
        for (const auto& index : load->indices) {
          if (Dispatch(index)) {
            return true;
          }
        }
        return false;
      }
      return Dispatch(op->args[0]);
    } else if (auto opt = op->op.as<Op>()) {
      auto effect_kind = op_call_effect_[opt.value()];
      if (effect_kind == CallEffectKind::kPure || effect_kind == CallEffectKind::kExprAnnotation) {
        for (const Expr& arg : op->args) {
          if (Dispatch(arg)) return true;
        }
        return false;
      } else {
        return true;
      }
    } else {
      return true;
    }
  }
  bool Dispatch_(const TensorLoadObj* op) {
    // Load is considered unsafe.
    return true;
  }
  bool Dispatch_(const prim::AddObj* op) final { return BinaryOp(op); }
  bool Dispatch_(const prim::SubObj* op) final { return BinaryOp(op); }
  bool Dispatch_(const prim::MulObj* op) final { return BinaryOp(op); }
  bool Dispatch_(const prim::DivObj* op) final { return BinaryOp(op); }
  bool Dispatch_(const prim::ModObj* op) final { return BinaryOp(op); }
  bool Dispatch_(const prim::FloorDivObj* op) final { return BinaryOp(op); }
  bool Dispatch_(const prim::FloorModObj* op) final { return BinaryOp(op); }
  bool Dispatch_(const prim::MinObj* op) final { return BinaryOp(op); }
  bool Dispatch_(const prim::MaxObj* op) final { return BinaryOp(op); }
  bool Dispatch_(const prim::EQObj* op) final { return BinaryOp(op); }
  bool Dispatch_(const prim::NEObj* op) final { return BinaryOp(op); }
  bool Dispatch_(const prim::LTObj* op) final { return BinaryOp(op); }
  bool Dispatch_(const prim::LEObj* op) final { return BinaryOp(op); }
  bool Dispatch_(const prim::GTObj* op) final { return BinaryOp(op); }
  bool Dispatch_(const prim::GEObj* op) final { return BinaryOp(op); }
  bool Dispatch_(const prim::AndObj* op) final { return BinaryOp(op); }
  bool Dispatch_(const prim::OrObj* op) final { return BinaryOp(op); }
  bool Dispatch_(const prim::LShiftObj* op) final { return BinaryOp(op); }
  bool Dispatch_(const prim::RShiftObj* op) final { return BinaryOp(op); }
  bool Dispatch_(const prim::BitwiseAndObj* op) final { return BinaryOp(op); }
  bool Dispatch_(const prim::BitwiseOrObj* op) final { return BinaryOp(op); }
  bool Dispatch_(const prim::BitwiseXorObj* op) final { return BinaryOp(op); }
  bool Dispatch_(const prim::BitwiseNotObj* op) final { return Dispatch(op->a); }
  bool Dispatch_(const prim::NotObj* op) final { return Dispatch(op->a); }
  bool Dispatch_(const prim::LetObj* op) final { return Dispatch(op->body) || Dispatch(op->value); }
  bool Dispatch_(const prim::CastObj* op) final { return Dispatch(op->value); }
  bool Dispatch_(const prim::BroadcastObj* op) final { return Dispatch(op->value); }
  bool Dispatch_(const prim::RampObj* op) final {
    return Dispatch(op->base) && Dispatch(op->stride);
  }
  bool Dispatch_(const prim::ShuffleObj* op) final {
    for (PrimExpr e : op->vectors) {
      if (Dispatch(e)) return true;
    }
    return false;
  }
  bool Dispatch_(const VarObj* op) final { return false; }
  bool Dispatch_(const IntImmObj* op) final { return false; }
  bool Dispatch_(const FloatImmObj* op) final { return false; }
  bool Dispatch_(const StringImmObj* op) final { return false; }

 private:
  template <typename T>
  bool BinaryOp(const T* op) {
    return Dispatch(op->a) || Dispatch(op->b);
  }

  OpAttrMap<CallEffectKind> op_call_effect_ = Op::GetAttrMap(tvm::op_attr::kCallEffectKind);
};

class UnsafeSelectRewriter : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  UnchangedOr<PrimExpr> Mutate_(const prim::SelectObj* op, InplaceMode inplace_mode) {
    PrimExpr expr =
        StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<PrimExpr>(op));
    op = expr.as<prim::SelectObj>();
    UnsafeExprDetector unsafe;
    PrimType cond_ty = op->condition.ty();
    bool cond_is_scalar_bool = cond_ty.MatchesCode(DLDataTypeCode::kDLBool) && cond_ty.IsScalar();
    if ((unsafe.Dispatch(op->true_value) || unsafe.Dispatch(op->false_value)) &&
        cond_is_scalar_bool) {
      return Call(op->ty.as_or_throw<PrimType>(), prim::if_then_else_op(),
                  {op->condition, op->true_value, op->false_value})
          .as_or_throw<PrimExpr>();
    } else {
      return expr;
    }
  }
};

Stmt RewriteUnsafeSelect(Stmt stmt) {
  return ffi::make_object<UnsafeSelectRewriter>()
      ->Mutate(stmt, InplaceMode::kAllow)
      .ValueOrUnchanged(std::move(stmt));
}

namespace transform {

Pass RewriteUnsafeSelect() {
  auto pass_func = [](Function f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = ffi::make_object<UnsafeSelectRewriter>()
                  ->Mutate(n->body, InplaceMode::kAllow)
                  .ValueOrUnchanged(std::move(n->body));
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "s_tir.RewriteUnsafeSelect");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.RewriteUnsafeSelect", RewriteUnsafeSelect);
}

}  // namespace transform

}  // namespace s_tir
}  // namespace tvm
