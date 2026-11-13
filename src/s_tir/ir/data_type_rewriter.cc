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

#include "data_type_rewriter.h"

#include <tvm/ffi/cast.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/region.h>

#include <algorithm>
#include <functional>
#include <utility>

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;
using namespace tvm::prim;

UnchangedOr<Stmt> DataTypeLegalizer::Mutate_(const ForObj* op, InplaceMode inplace_mode) {
  auto result = StmtExprMutator::Mutate_(op, inplace_mode);
  if (!result.IsUnchanged()) {
    op = ffi::AnyView(result).as<ForObj>();
    TVM_FFI_ICHECK(op != nullptr) << "Expected type to be ForObj, but get "
                                  << ffi::AnyView(result).GetTypeKey();
    if (!op->unique()) inplace_mode = InplaceMode::kDisallow;
  }
  PrimExpr e = Mutate(op->var, inplace_mode).ValueOrUnchanged(op->var);
  Var var = e.as_or_throw<Var>();
  PrimType var_ty = var->ty.as_or_throw<PrimType>();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* n = const_cast<ForObj*>(op);
    n->begin = prim::cast(op->begin, var_ty);
    n->end = prim::cast(op->end, var_ty);
    n->step = prim::cast(op->step, var_ty);
    return result;
  }
  auto n = ffi::make_object<ForObj>(*op);
  n->begin = prim::cast(op->begin, var_ty);
  n->end = prim::cast(op->end, var_ty);
  n->step = prim::cast(op->step, var_ty);
  return For(n);
}

UnchangedOr<Stmt> DataTypeLegalizer::Mutate_(const RegionStmtObj* op, InplaceMode inplace_mode) {
  Stmt result = StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
  const auto* region = result.as<RegionStmtObj>();
  if (!region || !region->op.same_as(tirx::launch_thread_op())) return result;
  PrimVar var = region->body_params[0].as_or_throw<PrimVar>();
  PrimExpr extent = region->args[1].as_or_throw<PrimExpr>();
  if (var.ty() == extent.ty()) return result;
  RegionStmt updated = ffi::GetRef<RegionStmt>(region);
  updated.CopyOnWrite()->args.Set(1, prim::cast(extent, var.ty()));
  return updated;
}

UnchangedOr<PrimExpr> DataTypeLegalizer::Mutate_(const prim::LetObj* op, InplaceMode inplace_mode) {
  auto value_result = this->Mutate(op->value, inplace_mode);
  bool value_unchanged = value_result.UnchangedOrSameAs(op->value);
  PrimExpr value = std::move(value_result).ValueOrUnchanged(op->value);
  Var var = Mutate(op->var).ValueOrUnchanged(op->var).as_or_throw<Var>();
  if (value.ty() != var->ty.as_or_throw<PrimType>()) {
    if (var.same_as(op->var)) {
      var = op->var.CopyWithDType(value.ty());
      VarRemapSet(op->var, var);
    } else {
      value = prim::cast(value, var->ty.as_or_throw<PrimType>());
      value_unchanged = false;
    }
  }
  auto new_body_result = this->Mutate(op->body, inplace_mode);
  bool new_body_unchanged = new_body_result.UnchangedOrSameAs(op->body);
  PrimExpr new_body = std::move(new_body_result).ValueOrUnchanged(op->body);

  if (value_unchanged && new_body_unchanged && var.same_as(op->var)) {
    return ffi::Unchanged();
  } else {
    return prim::Let(var, value, new_body, op->loc);
  }
}

UnchangedOr<Stmt> DataTypeLegalizer::Mutate_(const BindObj* op, InplaceMode inplace_mode) {
  if (!op->value.as<PrimExpr>()) {
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }
  auto value_result = this->Mutate(op->value, inplace_mode);
  bool value_unchanged = value_result.UnchangedOrSameAs(op->value);
  Expr value = std::move(value_result).ValueOrUnchanged(op->value);
  Var var = op->var;

  if (auto prim_value = value.as<PrimExpr>()) {
    if (prim_value.value().ty() != op->var->ty.as_or_throw<PrimType>()) {
      var = op->var.CopyWithDType(prim_value.value().ty());
      VarRemapSet(op->var, var);
    }
  }

  if (value_unchanged && var.same_as(op->var)) {
    return ffi::Unchanged();
  } else {
    return Bind(var, value, op->loc);
  }
}

UnchangedOr<PrimExpr> DataTypeLegalizer::Mutate_(const prim::SelectObj* op,
                                                 InplaceMode inplace_mode) {
  auto condition_result = this->Mutate(op->condition, inplace_mode);
  bool condition_unchanged = condition_result.UnchangedOrSameAs(op->condition);
  PrimExpr condition = std::move(condition_result).ValueOrUnchanged(op->condition);
  auto true_value_result = this->Mutate(op->true_value, inplace_mode);
  bool true_value_unchanged = true_value_result.UnchangedOrSameAs(op->true_value);
  PrimExpr true_value = std::move(true_value_result).ValueOrUnchanged(op->true_value);
  auto false_value_result = this->Mutate(op->false_value, inplace_mode);
  bool false_value_unchanged = false_value_result.UnchangedOrSameAs(op->false_value);
  PrimExpr false_value = std::move(false_value_result).ValueOrUnchanged(op->false_value);
  if (condition_unchanged && true_value_unchanged && false_value_unchanged &&
      true_value.ty() == false_value.ty()) {
    return ffi::Unchanged();
  } else {
    PrimType true_dtype = true_value.ty();
    PrimType false_dtype = false_value.ty();
    int bits = std::max(true_dtype.bits(), false_dtype.bits());
    PrimType dtype = true_dtype.WithBits(bits);
    if (true_dtype != dtype) true_value = prim::cast(true_value, dtype);
    if (false_dtype != dtype) false_value = prim::cast(false_value, dtype);
    return prim::Select(condition, true_value, false_value);
  }
}

UnchangedOr<PrimExpr> DataTypeLegalizer::Mutate_(const prim::RampObj* op,
                                                 InplaceMode inplace_mode) {
  auto base_result = Mutate(op->base, inplace_mode);
  bool base_unchanged = base_result.UnchangedOrSameAs(op->base);
  PrimExpr base = std::move(base_result).ValueOrUnchanged(op->base);
  auto stride_result = Mutate(op->stride, inplace_mode);
  bool stride_unchanged = stride_result.UnchangedOrSameAs(op->stride);
  PrimExpr stride = std::move(stride_result).ValueOrUnchanged(op->stride);
  if (base_unchanged && stride_unchanged && base.ty() == stride.ty()) {
    return ffi::Unchanged();
  } else {
    PrimType base_dtype = base.ty();
    PrimType stride_dtype = stride.ty();
    TVM_FFI_ICHECK(base_dtype.MatchesCode(DLDataTypeCode::kDLInt) &&
                   stride_dtype.MatchesCode(DLDataTypeCode::kDLInt));
    int bits = std::max(base_dtype.bits(), stride_dtype.bits());
    PrimType dtype = base_dtype.WithBits(bits);
    if (base_dtype->dtype != dtype->dtype) base = prim::cast(base, dtype);
    if (stride_dtype->dtype != dtype->dtype) stride = prim::cast(stride, dtype);
    return prim::Ramp(base, stride, op->lanes);
  }
}

UnchangedOr<PrimExpr> DataTypeLegalizer::Mutate_(const prim::BroadcastObj* op,
                                                 InplaceMode inplace_mode) {
  auto value = Mutate(op->value, inplace_mode);
  auto lanes = Mutate(op->lanes, inplace_mode);
  if (value.UnchangedOrSameAs(op->value) && lanes.UnchangedOrSameAs(op->lanes)) {
    return ffi::Unchanged();
  }
  // Construction re-infers the dtype; the shared structural hook retains the old dtype.
  return prim::Broadcast(std::move(value).ValueOrUnchanged(op->value),
                         std::move(lanes).ValueOrUnchanged(op->lanes));
}

UnchangedOr<PrimExpr> DataTypeLegalizer::Mutate_(const prim::ShuffleObj* op,
                                                 InplaceMode inplace_mode) {
  auto vectors = Mutate(op->vectors, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
  auto indices = Mutate(op->indices, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
  bool unchanged = vectors.UnchangedOrSameAs(op->vectors) && indices.UnchangedOrSameAs(op->indices);
  if (unchanged && inplace_mode == InplaceMode::kDisallow) return ffi::Unchanged();
  // Arrays can change in place; reconstruct to infer the dtype from the final vectors.
  PrimExpr updated = prim::Shuffle(std::move(vectors).ValueOrUnchanged(op->vectors),
                                   std::move(indices).ValueOrUnchanged(op->indices));
  if (unchanged && updated.ty() == op->ty.as_or_throw<PrimType>()) return ffi::Unchanged();
  return updated;
}

#define TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(OP, FUNC)                                \
  UnchangedOr<PrimExpr> DataTypeLegalizer::Mutate_(const OP* op, InplaceMode inplace_mode) { \
    auto a_result = this->Mutate(op->a, inplace_mode);                                       \
    bool a_unchanged = a_result.UnchangedOrSameAs(op->a);                                    \
    PrimExpr a = std::move(a_result).ValueOrUnchanged(op->a);                                \
    auto b_result = this->Mutate(op->b, inplace_mode);                                       \
    bool b_unchanged = b_result.UnchangedOrSameAs(op->b);                                    \
    PrimExpr b = std::move(b_result).ValueOrUnchanged(op->b);                                \
    if (a_unchanged && b_unchanged && a.ty() == b.ty()) {                                    \
      return ffi::Unchanged();                                                               \
    } else {                                                                                 \
      return FUNC(a, b);                                                                     \
    }                                                                                        \
  }

TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::AddObj, operator+);
TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::SubObj, operator-);
TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::MulObj, operator*);
TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::DivObj, div);
TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::ModObj, truncmod);
TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::FloorDivObj, floordiv);
TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::FloorModObj, floormod);
TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::MinObj, min);
TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::MaxObj, max);
TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::EQObj, operator==);
TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::NEObj, operator!=);
TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::LEObj, operator<=);
TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::LTObj, operator<);  // NOLINT(*)
TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::GTObj, operator>);  // NOLINT(*)
TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::GEObj, operator>=);

TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::BitwiseAndObj, bitwise_and);
TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::BitwiseOrObj, bitwise_or);
TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::BitwiseXorObj, bitwise_xor);

#undef TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH

UnchangedOr<PrimExpr> DataTypeLegalizer::Mutate_(const prim::LShiftObj* op,
                                                 InplaceMode inplace_mode) {
  PrimType before_dtype = op->a.ty();
  // Preserve the original operands while computing the narrowed shift.
  PrimExpr lhs = Mutate(op->a, InplaceMode::kDisallow).ValueOrUnchanged(op->a);
  PrimExpr rhs = Mutate(op->b, InplaceMode::kDisallow).ValueOrUnchanged(op->b);
  PrimType after_dtype = lhs.ty();
  if (ShouldClampShiftAmounts() && before_dtype.code() == DLDataTypeCode::kDLInt &&
      after_dtype.code() == DLDataTypeCode::kDLInt && before_dtype.bits() > after_dtype.bits()) {
    // Values fit in the narrowed dtype.  Clamp lane-wise to keep dynamic and
    // vector shift amounts below its width, preserving representable results.
    rhs = min(rhs, MakeConst(rhs.ty(), after_dtype.bits() - 1, op->loc), op->loc);
  }
  if (lhs.same_as(op->a) && rhs.same_as(op->b) && lhs.ty() == rhs.ty()) {
    return ffi::Unchanged();
  }
  return left_shift(lhs, rhs, op->loc);
}

UnchangedOr<PrimExpr> DataTypeLegalizer::Mutate_(const prim::RShiftObj* op,
                                                 InplaceMode inplace_mode) {
  PrimType before_dtype = op->a.ty();
  // Preserve the original operands while computing the narrowed shift.
  PrimExpr lhs = Mutate(op->a, InplaceMode::kDisallow).ValueOrUnchanged(op->a);
  PrimExpr rhs = Mutate(op->b, InplaceMode::kDisallow).ValueOrUnchanged(op->b);
  PrimType after_dtype = lhs.ty();
  if (ShouldClampShiftAmounts() && before_dtype.code() == DLDataTypeCode::kDLInt &&
      after_dtype.code() == DLDataTypeCode::kDLInt && before_dtype.bits() > after_dtype.bits()) {
    // Values fit in the narrowed dtype.  Clamp lane-wise to keep dynamic and
    // vector shift amounts below its width, preserving representable results.
    rhs = min(rhs, MakeConst(rhs.ty(), after_dtype.bits() - 1, op->loc), op->loc);
  }
  if (lhs.same_as(op->a) && rhs.same_as(op->b) && lhs.ty() == rhs.ty()) {
    return ffi::Unchanged();
  }
  return right_shift(lhs, rhs, op->loc);
}

UnchangedOr<PrimExpr> DataTypeLegalizer::Mutate_(const prim::BitwiseNotObj* op,
                                                 InplaceMode inplace_mode) {
  auto a = Mutate(op->a, inplace_mode);
  if (a.UnchangedOrSameAs(op->a)) return ffi::Unchanged();
  return prim::BitwiseNot(std::move(a).ValueOrUnchanged(op->a), op->loc);
}

UnchangedOr<Expr> DataTypeLegalizer::Mutate_(const CallObj* op, InplaceMode inplace_mode) {
  Call before = ffi::GetRef<Call>(op);
  // Keep the original argument dtype available for clz correction below.
  Expr e =
      StmtExprMutator::Mutate_(op, InplaceMode::kDisallow).ValueOrUnchanged(ffi::GetRef<Expr>(op));
  op = e.as<CallObj>();
  TVM_FFI_ICHECK(op != nullptr) << "Expected type to be CallObj"
                                << ", but get " << e->GetTypeKey();
  if (!op->ty.as<PrimTypeObj>()) {
    return e;
  }
  PrimExpr prim_e = e.as_or_throw<PrimExpr>();
  static const Op pow_op = Op::Get("prim.pow");
  static const Op& clz_op = prim::clz_op();
  if (op->op.same_as(pow_op)) {
    return pow(op->args[0].as_or_throw<PrimExpr>(), op->args[1].as_or_throw<PrimExpr>());
  } else if (op->op.same_as(prim::if_then_else_op())) {
    return Call(op->ty.as_or_throw<PrimType>(), op->op,
                {op->args[0].as_or_throw<PrimExpr>(), op->args[1].as_or_throw<PrimExpr>(),
                 op->args[2].as_or_throw<PrimExpr>()},
                op->attrs, {}, op->loc)
        .as_or_throw<PrimExpr>();
  } else if (op->op.same_as(clz_op)) {
    PrimType before_dtype = before->args[0].as_or_throw<PrimExpr>().ty();
    PrimType after_dtype = op->args[0].as_or_throw<PrimExpr>().ty();
    TVM_FFI_ICHECK((before_dtype.code() == DLDataTypeCode::kDLInt ||
                    before_dtype.code() == DLDataTypeCode::kDLUInt) &&
                   (before_dtype.bits() == 32 || before_dtype.bits() == 64))
        << "clz only supports 32 or 64 bit integer types, but get type before legalizing: "
        << before_dtype;
    TVM_FFI_ICHECK((after_dtype.code() == DLDataTypeCode::kDLInt ||
                    after_dtype.code() == DLDataTypeCode::kDLUInt) &&
                   (after_dtype.bits() == 32 || after_dtype.bits() == 64))
        << "clz only supports 32 or 64 bit integer types, but get type after legalizing: "
        << after_dtype;
    return prim_e - after_dtype.bits() + before_dtype.bits();
  }
  return prim_e;
}

UnchangedOr<Stmt> IndexDataTypeRewriter::Mutate_(const RegionStmtObj* op,
                                                 InplaceMode inplace_mode) {
  bool was_enabled = is_enabled_;
  if (op->op.same_as(tirx::launch_thread_op())) is_enabled_ = true;
  auto result = DataTypeLegalizer::Mutate_(op, inplace_mode);
  is_enabled_ = was_enabled;
  return result;
}

UnchangedOr<ffi::Any> IndexDataTypeRewriter::Mutate(ffi::AnyView value, InplaceMode inplace_mode) {
  bool is_enabled = is_enabled_;
  if (value.as<TensorTypeObj>()) is_enabled_ = true;
  auto result = DataTypeLegalizer::Mutate(value, inplace_mode);
  is_enabled_ = is_enabled;
  return result;
}

UnchangedOr<Stmt> IndexDataTypeRewriter::Mutate_(const TensorStoreObj* op,
                                                 InplaceMode inplace_mode) {
  TensorStore store = ffi::GetRef<TensorStore>(op);

  TensorVar new_tensor = Mutate(op->dest.as_or_throw<TensorVar>(), inplace_mode)
                             .as_or_throw<UnchangedOr<TensorVar>>()
                             .ValueOrUnchanged(op->dest.as_or_throw<TensorVar>());
  auto value_result = this->Mutate(op->value, inplace_mode);
  bool value_unchanged = value_result.UnchangedOrSameAs(op->value);
  auto value = std::move(value_result).ValueOrUnchanged(op->value);
  PrimType value_dtype = value.ty();
  if (new_tensor.dtype() != value_dtype && value_dtype.IsScalar()) {
    value = prim::cast(value, new_tensor.dtype());
    value_unchanged = false;
  }
  auto indices = VisitIndices(op->indices, inplace_mode);

  if (!new_tensor.same_as(op->dest.as_or_throw<TensorVar>()) || !value_unchanged ||
      !indices.same_as(op->indices)) {
    auto writer = store.CopyOnWrite();
    writer->dest = new_tensor;
    writer->value = value;
    writer->indices = indices;
  }

  return store;
}

UnchangedOr<PrimExpr> IndexDataTypeRewriter::Mutate_(const TensorLoadObj* op,
                                                     InplaceMode inplace_mode) {
  TensorLoad load = ffi::GetRef<TensorLoad>(op);

  TensorVar new_tensor =
      Mutate(op->source, inplace_mode).ValueOrUnchanged(op->source).as_or_throw<TensorVar>();
  auto indices = VisitIndices(op->indices, inplace_mode);

  if (!new_tensor.same_as(op->source.as_or_throw<tvm::tirx::TensorVar>()) ||
      !indices.same_as(op->indices)) {
    return TensorLoad(new_tensor, indices, op->loc);
  }

  return load;
}

ffi::Array<PrimExpr> IndexDataTypeRewriter::VisitIndices(const ffi::Array<PrimExpr>& indices,
                                                         InplaceMode inplace_mode) {
  bool is_enabled = is_enabled_;
  is_enabled_ = true;
  auto result = Mutate(indices, inplace_mode)
                    .as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>()
                    .ValueOrUnchanged(indices);
  is_enabled_ = is_enabled;
  return result;
}

UnchangedOr<Stmt> IndexDataTypeRewriter::Mutate_(const IfObj* op, InplaceMode inplace_mode) {
  bool is_condition = is_condition_;
  is_condition_ = true;
  auto cond_result = Mutate(op->condition, inplace_mode);
  bool cond_unchanged = cond_result.UnchangedOrSameAs(op->condition);
  PrimExpr cond = std::move(cond_result).ValueOrUnchanged(op->condition);
  is_condition_ = is_condition;
  auto then_case_result = Mutate(op->then_case, inplace_mode);
  bool then_case_unchanged = then_case_result.UnchangedOrSameAs(op->then_case);
  SeqStmt then_case = std::move(then_case_result).ValueOrUnchanged(op->then_case);
  ffi::Optional<SeqStmt> else_case =
      op->else_case.has_value()
          ? ffi::Optional<SeqStmt>{Mutate(op->else_case.value(), inplace_mode)
                                       .ValueOrUnchanged(op->else_case.value())}
          : std::nullopt;
  if (!cond_unchanged || !then_case_unchanged || !else_case.same_as(op->else_case)) {
    If new_stmt = ffi::GetRef<If>(op);
    auto* n = new_stmt.CopyOnWrite();
    n->condition = std::move(cond);
    n->then_case = std::move(then_case);
    n->else_case = std::move(else_case);
    return new_stmt;
  }
  return ffi::Unchanged();
}

UnchangedOr<Stmt> IndexDataTypeRewriter::Mutate_(const ForObj* op, InplaceMode inplace_mode) {
  bool is_enabled = is_enabled_;
  is_enabled_ = true;
  PrimVar new_loop_var =
      Mutate(op->var, inplace_mode).ValueOrUnchanged(op->var).as_or_throw<PrimVar>();
  auto begin_result = Mutate(op->begin, inplace_mode);
  bool begin_unchanged = begin_result.UnchangedOrSameAs(op->begin);
  PrimExpr begin = std::move(begin_result).ValueOrUnchanged(op->begin);
  auto end_result = Mutate(op->end, inplace_mode);
  bool end_unchanged = end_result.UnchangedOrSameAs(op->end);
  PrimExpr end = std::move(end_result).ValueOrUnchanged(op->end);
  auto step_result = Mutate(op->step, inplace_mode);
  bool step_unchanged = step_result.UnchangedOrSameAs(op->step);
  PrimExpr step = std::move(step_result).ValueOrUnchanged(op->step);
  is_enabled_ = is_enabled;
  auto new_body_result = Mutate(op->body, inplace_mode);
  bool new_body_unchanged = new_body_result.UnchangedOrSameAs(op->body);
  Stmt new_body = std::move(new_body_result).ValueOrUnchanged(op->body);

  if (!new_loop_var.same_as(op->var) || !begin_unchanged || !end_unchanged || !step_unchanged ||
      !new_body_unchanged) {
    For new_for = ffi::GetRef<For>(op);
    auto* n = new_for.CopyOnWrite();
    n->var = new_loop_var;
    n->begin = prim::cast(begin, new_loop_var.ty());
    n->end = prim::cast(end, new_loop_var.ty());
    n->step = prim::cast(step, new_loop_var.ty());
    n->body = new_body;
    return new_for;

  } else {
    return ffi::Unchanged();
  }
}

UnchangedOr<Stmt> IndexDataTypeRewriter::Mutate_(const BindObj* op, InplaceMode inplace_mode) {
  if (!op->value.as<PrimExpr>()) {
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }
  auto mapped = VarRemapGet(op->var);
  if (mapped == nullptr || mapped.type_index() == ffi::TypeIndex::kTVMFFIUnchanged) {
    return DataTypeLegalizer::Mutate_(op, inplace_mode);
  }
  Var var = mapped.as_or_throw<Var>();
  bool is_enabled = is_enabled_;
  is_enabled_ = true;
  PrimExpr value =
      Mutate(op->value, inplace_mode).ValueOrUnchanged(op->value).as_or_throw<PrimExpr>();
  is_enabled_ = is_enabled;
  // The collected index requirement need not apply to every variable in the RHS.
  return Bind(var, prim::cast(value, var->ty.as_or_throw<PrimType>()), op->loc);
}

#define TVM_DEFINE_CMPOP_EXPR_MUTATE_WITH_TYPE_MATCH(OP, FUNC)                                   \
  UnchangedOr<PrimExpr> IndexDataTypeRewriter::Mutate_(const OP* op, InplaceMode inplace_mode) { \
    bool is_enabled = is_enabled_;                                                               \
    is_enabled_ = is_condition_ && op->a.ty().MatchesCode(DLDataTypeCode::kDLInt) &&             \
                  op->b.ty().MatchesCode(DLDataTypeCode::kDLInt);                                \
    auto result = Parent::Mutate_(op, inplace_mode);                                             \
    is_enabled_ = is_enabled;                                                                    \
    return result;                                                                               \
  }

TVM_DEFINE_CMPOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::EQObj, operator==);
TVM_DEFINE_CMPOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::NEObj, operator!=);
TVM_DEFINE_CMPOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::LEObj, operator<=);
TVM_DEFINE_CMPOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::LTObj, operator<);  // NOLINT(*)
TVM_DEFINE_CMPOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::GTObj, operator>);  // NOLINT(*)
TVM_DEFINE_CMPOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::GEObj, operator>=);

UnchangedOr<Expr> IndexDataTypeRewriter::Mutate_(const CallObj* op, InplaceMode inplace_mode) {
  // handle if_then_else condition
  if (op->op.same_as(prim::if_then_else_op())) {
    bool is_condition = is_condition_;
    is_condition_ = true;
    PrimExpr cond = Mutate(op->args[0]).ValueOrUnchanged(op->args[0]).as_or_throw<PrimExpr>();
    is_condition_ = is_condition;
    PrimExpr true_value = Mutate(op->args[1]).ValueOrUnchanged(op->args[1]).as_or_throw<PrimExpr>();
    PrimExpr false_value =
        Mutate(op->args[2]).ValueOrUnchanged(op->args[2]).as_or_throw<PrimExpr>();
    PrimType true_dtype = true_value.ty();
    PrimType false_dtype = false_value.ty();
    PrimType dtype = true_dtype.WithBits(std::max(true_dtype.bits(), false_dtype.bits()));
    if (true_dtype != dtype) true_value = prim::cast(true_value, dtype);
    if (false_dtype != dtype) false_value = prim::cast(false_value, dtype);
    return Call(dtype, op->op, {cond, true_value, false_value}, op->attrs, {}, op->loc)
        .as_or_throw<PrimExpr>();
  }
  return Parent::Mutate_(op, inplace_mode);
}

UnchangedOr<PrimExpr> IndexDataTypeRewriter::Mutate_(const prim::SelectObj* op,
                                                     InplaceMode inplace_mode) {
  bool is_condition = true;
  std::swap(is_condition_, is_condition);
  auto condition_result = this->Mutate(op->condition, inplace_mode);
  bool condition_unchanged = condition_result.UnchangedOrSameAs(op->condition);
  PrimExpr condition = std::move(condition_result).ValueOrUnchanged(op->condition);
  std::swap(is_condition_, is_condition);
  auto true_value_result = this->Mutate(op->true_value, inplace_mode);
  bool true_value_unchanged = true_value_result.UnchangedOrSameAs(op->true_value);
  PrimExpr true_value = std::move(true_value_result).ValueOrUnchanged(op->true_value);
  auto false_value_result = this->Mutate(op->false_value, inplace_mode);
  bool false_value_unchanged = false_value_result.UnchangedOrSameAs(op->false_value);
  PrimExpr false_value = std::move(false_value_result).ValueOrUnchanged(op->false_value);

  if (condition_unchanged && true_value_unchanged && false_value_unchanged &&
      true_value.ty() == false_value.ty()) {
    return ffi::Unchanged();
  } else {
    PrimType true_dtype = true_value.ty();
    PrimType false_dtype = false_value.ty();
    int bits = std::max(true_dtype.bits(), false_dtype.bits());
    PrimType dtype = true_dtype.WithBits(bits);
    if (true_dtype->dtype != dtype->dtype) true_value = prim::cast(true_value, dtype);
    if (false_dtype->dtype != dtype->dtype) false_value = prim::cast(false_value, dtype);
    return prim::Select(condition, true_value, false_value);
  }
}

#undef TVM_DEFINE_CMPOP_EXPR_MUTATE_WITH_TYPE_MATCH

Function IndexDataTypeNormalizer::Rewrite(Function func) {
  // Keep this short setup local so its collector uses S-TIR block semantics
  // without adding a dialect-specific collector hook to the TIRX normalizer.
  // Collect scalar dtype requirements without changing types.  Tensor definitions
  // are rewritten only after every scalar replacement has been seeded.
  class IndexVarCollector : public IndexDataTypeNormalizer {
   public:
    explicit IndexVarCollector(std::function<void(const VarObj*)> collect)
        : IndexDataTypeNormalizer(PrimType::Int(64)), collect_(std::move(collect)) {}
    using IndexDataTypeNormalizer::Mutate;
    using IndexDataTypeNormalizer::Mutate_;
    UnchangedOr<Expr> Mutate_(const VarObj* op, InplaceMode mode) final {
      if (def_region_kind() == kTVMFFIDefRegionKindNone && is_enabled_) collect_(op);
      return IndexDataTypeNormalizer::Mutate_(op, mode);
    }

   protected:
    bool CanRewriteDType(PrimType dtype) const final { return false; }

   private:
    std::function<void(const VarObj*)> collect_;
  };
  auto seed = [this](const VarObj* var) {
    auto dtype = var->ty.as<PrimType>();
    if (dtype && CanRewriteDType(dtype.value()) && dtype.value() != target_data_type_ &&
        VarRemapGet(ffi::AnyView(var)) == nullptr) {
      VarRemapSet(ffi::AnyView(var), ffi::GetRef<Var>(var).CopyWithDType(target_data_type_));
    }
  };
  auto collector = ffi::make_object<IndexVarCollector>(seed);
  collector->Mutate(func->body);
  for (const Var& param : func->params) {
    if (param.as<TensorVar>()) {
      collector->WithDefRegionKind(kTVMFFIDefRegionKindSimple,
                                   [&] { return collector->Mutate(param); });
    } else {
      seed(param.get());
    }
  }
  ffi::Array<Var> params = func->params.Map([this](const Var& param) {
    return WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&] {
      return Mutate(param).ValueOrUnchanged(param).as_or_throw<Var>();
    });
  });
  FunctionObj* new_func = func.CopyOnWrite();
  new_func->params = std::move(params);
  new_func->body = Mutate(new_func->body).ValueOrUnchanged(new_func->body);
  return func;
}

UnchangedOr<Stmt> IndexDataTypeNormalizer::Mutate_(const SBlockRealizeObj* op,
                                                   InplaceMode inplace_mode) {
  bool is_condition = this->is_condition_;
  this->is_condition_ = true;
  auto new_predicate_result = this->Mutate(op->predicate, inplace_mode);
  bool new_predicate_unchanged = new_predicate_result.UnchangedOrSameAs(op->predicate);
  auto new_predicate = std::move(new_predicate_result).ValueOrUnchanged(op->predicate);
  this->is_condition_ = is_condition;

  bool is_enabled = this->is_enabled_;
  this->is_enabled_ = true;
  auto new_iter_values = this->Mutate(op->iter_values, inplace_mode)
                             .as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>()
                             .ValueOrUnchanged(op->iter_values);
  this->is_enabled_ = is_enabled;
  SBlock new_body =
      this->Mutate(op->block, inplace_mode).ValueOrUnchanged(op->block).as_or_throw<SBlock>();
  if (!new_predicate_unchanged || !new_iter_values.same_as(op->iter_values) ||
      !new_body.same_as(op->block)) {
    SBlockRealize new_block_realize = ffi::GetRef<SBlockRealize>(op);
    auto* n = new_block_realize.CopyOnWrite();
    n->predicate = std::move(new_predicate);
    n->iter_values = std::move(new_iter_values);
    n->block = std::move(new_body);
    return new_block_realize;

  } else {
    return ffi::Unchanged();
  }
}

UnchangedOr<Stmt> IndexDataTypeNormalizer::Mutate_(const SBlockObj* op, InplaceMode inplace_mode) {
  auto new_alloc_tensors = this->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&] {
    return this->Mutate(op->alloc_tensors, inplace_mode)
        .as_or_throw<UnchangedOr<ffi::Array<TensorVar>>>()
        .ValueOrUnchanged(op->alloc_tensors);
  });
  auto new_match_tensors = op->match_tensors.Map([this](const MatchTensorRegion& match) {
    TensorVar tensor = this->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&] {
      return this->Mutate(match->tensor, InplaceMode::kDisallow)
          .as_or_throw<UnchangedOr<TensorVar>>()
          .ValueOrUnchanged(match->tensor);
    });
    TensorRegion source = VisitTensorRegion(match->source);
    if (tensor.same_as(match->tensor) && source.same_as(match->source)) return match;
    return MatchTensorRegion(tensor, source);
  });
  ffi::Array<TensorRegion> new_reads = op->reads.Map(
      [this](const TensorRegion& tensor_region) { return VisitTensorRegion(tensor_region); });
  ffi::Array<TensorRegion> new_writes = op->writes.Map(
      [this](const TensorRegion& tensor_region) { return VisitTensorRegion(tensor_region); });
  ffi::Array<IterVar> new_iter_vars =
      op->iter_vars.Map([this](const IterVar& iter_var) { return VisitIterVar(iter_var); });
  ffi::Optional<SeqStmt> new_init = std::nullopt;
  if (op->init.has_value()) {
    new_init = this->Mutate(op->init.value(), inplace_mode).ValueOrUnchanged(op->init.value());
  }
  ffi::Map<ffi::String, ffi::Any> new_annotations = VisitBlockAnnotations(op->annotations);
  auto new_body_result = this->Mutate(op->body, inplace_mode);
  bool new_body_unchanged = new_body_result.UnchangedOrSameAs(op->body);
  Stmt new_body = std::move(new_body_result).ValueOrUnchanged(op->body);

  if (!new_init.same_as(op->init) || !new_body_unchanged ||
      !new_alloc_tensors.same_as(op->alloc_tensors) ||
      !new_match_tensors.same_as(op->match_tensors) || !new_reads.same_as(op->reads) ||
      !new_writes.same_as(op->writes) || !new_iter_vars.same_as(op->iter_vars) ||
      !new_annotations.same_as(op->annotations)) {
    SBlock new_block = ffi::GetRef<SBlock>(op);
    SBlockObj* n = new_block.CopyOnWrite();
    n->alloc_tensors = std::move(new_alloc_tensors);
    n->match_tensors = std::move(new_match_tensors);
    n->reads = std::move(new_reads);
    n->writes = std::move(new_writes);
    n->iter_vars = std::move(new_iter_vars);
    n->init = std::move(new_init);
    n->annotations = std::move(new_annotations);
    n->body = std::move(new_body);
    return new_block;
  }
  return ffi::Unchanged();
}

ffi::Map<ffi::String, ffi::Any> IndexDataTypeNormalizer::VisitBlockAnnotations(
    const ffi::Map<ffi::String, ffi::Any>& annotations) {
  auto new_annotations = annotations;

  std::function<Any(const Any&)> f_mutate_obj = [this, &f_mutate_obj](const Any& obj) -> Any {
    if (obj == nullptr) {
      return obj;
    }
    if (auto var = obj.as<Var>(); var && var.value()->ty.as<TensorTypeObj>()) {
      TensorVar tensor = var.value().as_or_throw<TensorVar>();
      if (TensorVar new_tensor = this->Mutate(tensor, InplaceMode::kDisallow)
                                     .as_or_throw<UnchangedOr<TensorVar>>()
                                     .ValueOrUnchanged(tensor);
          !new_tensor.same_as(tensor)) {
        return new_tensor;
      }
    } else if (obj.as<ffi::ArrayObj>()) {
      return obj.as_or_throw<ffi::Array<Any>>().Map(f_mutate_obj);
    }
    return obj;
  };
  for (const auto& [key, value] : annotations) {
    if (auto opt_object_ref = value.as<ffi::ObjectRef>()) {
      auto new_value = f_mutate_obj(*opt_object_ref);
      if (!new_value.same_as(*opt_object_ref)) {
        new_annotations.Set(key, new_value);
      }
    }
  }
  return new_annotations;
}

IterVar IndexDataTypeNormalizer::VisitIterVar(const IterVar& iter_var) {
  bool is_enabled = this->is_enabled_;
  this->is_enabled_ = true;
  PrimVar new_var = this->Mutate(iter_var->var, InplaceMode::kDisallow)
                        .ValueOrUnchanged(iter_var->var)
                        .as_or_throw<PrimVar>();
  PrimExpr min = this->Mutate(iter_var->dom.value()->begin, InplaceMode::kDisallow)
                     .ValueOrUnchanged(iter_var->dom.value()->begin);
  PrimExpr extent = this->Mutate(iter_var->dom.value()->extent, InplaceMode::kDisallow)
                        .ValueOrUnchanged(iter_var->dom.value()->extent);
  this->is_enabled_ = is_enabled;
  if (!new_var.same_as(iter_var->var) || !min.same_as(iter_var->dom.value()->begin) ||
      !extent.same_as(iter_var->dom.value()->extent)) {
    IterVar new_iter_var = iter_var;
    IterVarObj* n = new_iter_var.CopyOnWrite();
    n->var = std::move(new_var);
    n->ty = n->var.ty();
    n->dom = Range::FromMinExtent(min, extent);
    return new_iter_var;
  }
  return iter_var;
}

TensorRegion IndexDataTypeNormalizer::VisitTensorRegion(const TensorRegion& tensor_region) {
  TensorVar remapped_tensor =
      this->Mutate(tensor_region->source.as_or_throw<TensorVar>(), InplaceMode::kDisallow)
          .as_or_throw<UnchangedOr<TensorVar>>()
          .ValueOrUnchanged(tensor_region->source.as_or_throw<TensorVar>());

  bool is_enabled = this->is_enabled_;
  this->is_enabled_ = true;
  auto new_region = tensor_region->region.Map([&](const Range& range) {
    return Range::FromMinExtent(
        this->Mutate(range->begin, InplaceMode::kDisallow).ValueOrUnchanged(range->begin),
        this->Mutate(range->extent, InplaceMode::kDisallow).ValueOrUnchanged(range->extent));
  });
  this->is_enabled_ = is_enabled;

  if (!remapped_tensor.same_as(tensor_region->source.as_or_throw<TensorVar>()) ||
      !new_region.same_as(tensor_region->region)) {
    return TensorRegion(remapped_tensor, new_region, TensorRegionType());
  } else {
    return tensor_region;
  }
}

bool IndexDataTypeNormalizer::CanRewriteDType(PrimType dtype) const {
  return dtype.code() == DLDataTypeCode::kDLInt && dtype.bits() >= 32;
}

UnchangedOr<PrimExpr> IndexDataTypeNormalizer::Mutate_(const IntImmObj* op,
                                                       InplaceMode inplace_mode) {
  if (is_enabled_ && CanRewriteDType(op->ty.as_or_throw<PrimType>())) {
    TVM_FFI_ICHECK_LE(op->value, max_value(target_data_type_).as_or_throw<IntImm>()->value);
    return prim::cast(ffi::GetRef<IntImm>(op), target_data_type_);
  }
  return ffi::Unchanged();
}

UnchangedOr<PrimExpr> IndexDataTypeNormalizer::Mutate_(const prim::CastObj* op,
                                                       InplaceMode inplace_mode) {
  // Unwrap the cast only when the dtype of this cast is integer dtype.
  // When the dtype of this cast is not integer dtype, it means that this cast
  // has some other purpose, and we should not unwrap the cast.
  PrimType dtype = op->ty.as_or_throw<PrimType>();
  if (is_enabled_ && CanRewriteDType(dtype)) {
    PrimExpr value = this->Mutate(op->value, inplace_mode).ValueOrUnchanged(op->value);
    return value.ty() == target_data_type_ ? value : prim::Cast(target_data_type_, value);
  }
  return IndexDataTypeRewriter::Mutate_(op, inplace_mode);
}

}  // namespace s_tir
}  // namespace tvm
