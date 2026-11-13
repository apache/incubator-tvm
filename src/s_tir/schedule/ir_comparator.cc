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
#include "./ir_comparator.h"

#include <tvm/ffi/cast.h>
#include <tvm/ir/prim/op.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm {

namespace {
// File-local helper: true if `expr` is a call to prim::vscale_op().
bool IsVScaleCall(const PrimExpr& expr) {
  if (const auto* call = expr.as<CallObj>()) {
    return call->op.same_as(prim::vscale_op());
  }
  return false;
}

// File-local helper: true if `expr` contains a call to prim::vscale_op().
bool ContainsVscaleCall(const PrimExpr& expr) {
  struct VScaleFinder : tirx::StmtExprVisitor {
    ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) final {
      if (auto expr = value.as<PrimExpr>(); expr && IsVScaleCall(*expr)) {
        return VisitInterrupt();
      }
      return StmtExprVisitor::Visit(value);
    }
  };
  return ffi::make_object<VScaleFinder>()->Visit(expr).has_value();
}
}  // namespace

namespace s_tir {
using namespace tvm::prim;
using namespace tvm::tirx;

/******** Tensorize Comparator ********/

class TensorIntrinMismatchError : public ScheduleErrorContextObj {
 public:
  explicit TensorIntrinMismatchError(IRModule lhs_mod, Stmt lhs_stmt, Stmt rhs_stmt,
                                     std::vector<std::string> error_messages)
      : lhs_mod_(std::move(lhs_mod)),
        lhs_stmt_(std::move(lhs_stmt)),
        rhs_stmt_(std::move(rhs_stmt)),
        error_messages_(std::move(error_messages)) {
    TVM_FFI_ICHECK(lhs_stmt_->IsInstance<ForObj>() || lhs_stmt_->IsInstance<SBlockObj>());
  }

  ffi::String FastErrorString() const final {
    return "ScheduleError: The stmt doesn't match the tensor intrin.";
  }

  ffi::String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The stmt {0} doesn't match the tensor intrin\nThe pattern attempting to be matched:\n"
       << lhs_stmt_ << "\nDoes not match the tensorize description:\n"
       << rhs_stmt_ << '\n';
    for (const auto& msg : error_messages_) {
      os << msg << std::endl;
    }
    return os.str();
  }

  IRModule mod() const final { return lhs_mod_; }

  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {lhs_stmt_}; }

 private:
  IRModule lhs_mod_;
  Stmt lhs_stmt_;
  Stmt rhs_stmt_;
  std::vector<std::string> error_messages_;
};

/* Override the dispatcher to make sure RHS is always valid */
bool TensorizeComparator::Dispatch(const Stmt& n, const Stmt& other) {
  bool equal = n.same_as(other) ||
               ((n->type_index() == other->type_index()) && StmtComparator::Dispatch(n, other));
  if (!equal && assert_mode_ && (n->IsInstance<ForObj>() || n->IsInstance<SBlockObj>())) {
    throw MakeScheduleError<TensorIntrinMismatchError>(lhs_mod_, n, other,
                                                       std::move(error_messages_));
  }
  return equal;
}

bool TensorizeComparator::Dispatch(const Expr& expr, const PrimExpr& other) {
  PrimExpr n = expr.as_or_throw<PrimExpr>();
  bool equal = n.same_as(other) ||
               ((n->type_index() == other->type_index()) && n.ty().code() == other.ty().code() &&
                ExprComparator::Dispatch(n, other)) ||
               (ContainsVscaleCall(n) && analyzer_->CanProveEqual(n, other));

  if (!equal && assert_mode_) {
    std::ostringstream os;
    os << "Expression mismatch: " << n << " vs " << other;
    EmitError(os.str());
  }
  return equal;
}

bool TensorizeComparator::CompareExpr(const Expr& lhs, const Expr& rhs) {
  if (lhs.same_as(rhs)) return true;
  if (auto rhs_prim = rhs.as<PrimExpr>()) {
    if (!lhs.as<PrimExpr>()) return false;
    return Dispatch(lhs, rhs_prim.value());
  }
  if (!ffi::StructuralEqual()(lhs->ty, rhs->ty)) return false;
  if (auto lhs_var = lhs.as<Var>()) {
    auto rhs_var = rhs.as<Var>();
    return rhs_var && DefEqual(lhs_var.value(), rhs_var.value());
  }
  if (auto lhs_call = lhs.as<Call>()) {
    auto rhs_call = rhs.as<Call>();
    if (!rhs_call || !lhs_call.value()->op.same_as(rhs_call.value()->op) ||
        lhs_call.value()->args.size() != rhs_call.value()->args.size()) {
      return false;
    }
    for (size_t i = 0; i < lhs_call.value()->args.size(); ++i) {
      if (!CompareExpr(lhs_call.value()->args[i], rhs_call.value()->args[i])) return false;
    }
    return true;
  }
  return ffi::StructuralEqual()(lhs, rhs);
}

bool TensorizeComparator::Dispatch_(const CallObj* op, const PrimExpr& other) {
  const auto* rhs = other.as<CallObj>();
  if (!rhs->op.same_as(op->op)) return false;
  if (op->ty.as_or_throw<PrimType>().code() != rhs->ty.as_or_throw<PrimType>().code()) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "CallObj data type codes do not match: op->dtype.code()="
         << op->ty.as_or_throw<PrimType>().code()
         << " vs rhs->dtype.code()=" << rhs->ty.as_or_throw<PrimType>().code();
      EmitError(os.str());
    }
    return false;
  }
  if (op->args.size() != rhs->args.size()) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "CallObj arg size mismatch: op->args.size()=" << op->args.size()
         << " vs rhs->args.size()=" << rhs->args.size();
      EmitError(os.str());
    }
    return false;
  }
  for (size_t i = 0; i < op->args.size(); ++i) {
    if (!CompareExpr(op->args[i], rhs->args[i])) {
      if (assert_mode_) {
        std::ostringstream os;
        os << "CallObj args do not match at index " << i << ": op->args[i]=" << op->args[i]
           << " vs rhs->args[i]=" << rhs->args[i];
        EmitError(os.str());
      }
      return false;
    }
  }
  return true;
}

bool TensorizeComparator::Dispatch_(const ForObj* op, const Stmt& other) {
  const auto* rhs = other.as<ForObj>();
  if (!DefEqual(op->var, rhs->var)) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "ForObj loop vars do not match: op->var=" << op->var << " vs rhs->var=" << rhs->var;
      EmitError(os.str());
    }
    return false;
  }
  if (!Dispatch(op->begin, rhs->begin)) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "ForObj begin values do not match: op->begin=" << op->begin
         << " vs rhs->begin=" << rhs->begin;
      EmitError(os.str());
    }
    return false;
  }
  if (!Dispatch(op->end, rhs->end)) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "ForObj end values do not match: op->end=" << op->end << " vs rhs->end=" << rhs->end;
      EmitError(os.str());
    }
    return false;
  }
  if (!Dispatch(op->step, rhs->step)) {
    if (assert_mode_) {
      EmitError("ForObj step values do not match");
    }
    return false;
  }
  if (op->kind != rhs->kind) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "ForObj kinds do not match: op->kind=" << op->kind << " vs rhs->kind=" << rhs->kind;
      EmitError(os.str());
    }
    return false;
  }
  if (!CompareAnnotationMap(op->annotations, rhs->annotations)) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "ForObj annotation maps do not match: op->annotations=" << op->annotations
         << " vs rhs->annotations=" << rhs->annotations;
      EmitError(os.str());
    }
    return false;
  }
  return Dispatch(op->body, rhs->body);
}

bool TensorizeComparator::Dispatch_(const SeqStmtObj* op, const Stmt& other) {
  const auto* rhs = other.as<SeqStmtObj>();
  return CompareArray(op->seq, rhs->seq,
                      static_cast<bool (TensorizeComparator::*)(const Stmt&, const Stmt&)>(
                          &TensorizeComparator::Dispatch));
}

bool TensorizeComparator::Dispatch_(const TensorStoreObj* op, const Stmt& other) {
  const auto* rhs = other.as<TensorStoreObj>();
  return CompareTensorAccess(op, rhs) && Dispatch(op->value, rhs->value);
}

bool TensorizeComparator::Dispatch_(const SBlockRealizeObj* op, const Stmt& other) {
  const auto* rhs = other.as<SBlockRealizeObj>();
  if (!is_scope_block) {
    if (!CompareArray(op->iter_values, rhs->iter_values,
                      static_cast<bool (TensorizeComparator::*)(const Expr&, const PrimExpr&)>(
                          &TensorizeComparator::Dispatch))) {
      if (assert_mode_) {
        std::ostringstream os;
        os << "BlockRealizeNode iter_values do not match: op->iter_values=" << op->iter_values
           << " vs rhs->iter_values=" << rhs->iter_values;
        EmitError(os.str());
      }
      return false;
    }
  }
  return Dispatch(op->predicate, rhs->predicate) && Dispatch(op->block, rhs->block);
}

bool TensorizeComparator::Dispatch_(const SBlockObj* op, const Stmt& other) {
  const auto* rhs = other.as<SBlockObj>();
  for (const IterVar& iter : op->iter_vars) {
    {
      auto context = lhs_analyzer_->GetConstraintContext();
      context->Append(sym::RangeConstraint(iter->var, iter->dom.value()));
      lhs_analyzer_->SyncConstraintContext(context);
    }
  }
  // Check block equality.
  // All iter vars and tensor regions including the order should match.
  // When checking iter vars, DefEqual is used to remap variables.
  if (!is_scope_block) {
    if (!CompareArray(op->iter_vars, rhs->iter_vars, &TensorizeComparator::CompareIterVar)) {
      if (assert_mode_) {
        std::ostringstream os;
        os << "BlockNode iter_vars do not match: op->alloc_tensors=" << op->iter_vars
           << " vs rhs->alloc_tensors=" << rhs->iter_vars;
        EmitError(os.str());
      }
      return false;
    }
    if (!CompareArray(op->alloc_tensors, rhs->alloc_tensors, &TensorizeComparator::CompareTensor)) {
      if (assert_mode_) {
        std::ostringstream os;
        os << "BlockNode alloc_tensors do not match: op->alloc_tensors=" << op->alloc_tensors
           << " vs rhs->alloc_tensors=" << rhs->alloc_tensors;
        EmitError(os.str());
      }
      return false;
    }
  }
  if (!CompareArray(op->writes, rhs->writes, &TensorizeComparator::CompareTensorRegion)) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "BlockNode write tensors do not match: op->writes=" << op->writes
         << " vs rhs->writes=" << rhs->writes;
      EmitError(os.str());
    }
    return false;
  }
  if (!CompareArray(op->reads, rhs->reads, &TensorizeComparator::CompareTensorRegion)) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "BlockNode read tensors regions do not match: op->reads=" << op->reads
         << " vs rhs->reads=" << rhs->reads;
      EmitError(os.str());
    }
    return false;
  }
  is_scope_block = false;
  return Dispatch(op->body, rhs->body);
}

// Exprs
#define TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(OpName)                           \
  bool TensorizeComparator::Dispatch_(const OpName* op, const PrimExpr& other) { \
    const auto* rhs = other.as<OpName>();                                        \
    return Dispatch(op->a, rhs->a) && Dispatch(op->b, rhs->b);                   \
  }

TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(AddObj);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(SubObj);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(MulObj);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(DivObj);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(ModObj);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(EQObj);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(NEObj);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(LTObj);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(LEObj);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(GTObj);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(GEObj);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(AndObj);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(OrObj);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(MinObj);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(MaxObj);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(FloorDivObj);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(FloorModObj);

TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(prim::LShiftObj);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(prim::RShiftObj);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(prim::BitwiseAndObj);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(prim::BitwiseOrObj);
TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(prim::BitwiseXorObj);

bool TensorizeComparator::Dispatch_(const prim::BitwiseNotObj* op, const PrimExpr& other) {
  const auto* rhs = other.as<prim::BitwiseNotObj>();
  return Dispatch(op->a, rhs->a);
}

bool TensorizeComparator::Dispatch_(const IntImmObj* op, const PrimExpr& other) {
  const auto* rhs = other.as<IntImmObj>();
  if (op->value != rhs->value) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "IntImmObj values do not match: op->value=" << op->value
         << " vs rhs->value=" << rhs->value;
      EmitError(os.str());
    }
    return false;
  }
  return true;
}

bool TensorizeComparator::Dispatch_(const FloatImmObj* op, const PrimExpr& other) {
  const auto* rhs = other.as<FloatImmObj>();
  if (op->value != rhs->value) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "FloatImmObj values do not match: op->value=" << op->value
         << " vs rhs->value=" << rhs->value;
      EmitError(os.str());
    }
    return false;
  }
  return true;
}

bool TensorizeComparator::Dispatch_(const CastObj* op, const PrimExpr& other) {
  const auto* rhs = other.as<CastObj>();
  return Dispatch(op->value, rhs->value);
}

bool TensorizeComparator::Dispatch_(const VarObj* op, const PrimExpr& other) {
  auto rhs_ref = other.as<PrimVar>();
  if (!rhs_ref.has_value()) return false;
  const auto* rhs = rhs_ref.value().get();
  auto lhs = ffi::GetRef<Var>(op);
  if (lhs.same_as(other)) return true;
  PrimType lhs_ty = op->ty.as_or_throw<PrimType>();
  PrimType rhs_ty = rhs->ty.as_or_throw<PrimType>();
  if (lhs_ty.code() != rhs_ty.code()) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "VarObj data type codes do not match: op->dtype.code()=" << lhs_ty.code()
         << " vs rhs->dtype.code()=" << rhs_ty.code();
      EmitError(os.str());
    }
    return false;
  }
  auto it = equal_map_.find(lhs);
  return it != equal_map_.end() && it->second.same_as(other);
}

bool TensorizeComparator::Dispatch_(const TensorLoadObj* op, const PrimExpr& other) {
  const auto* rhs = other.as<TensorLoadObj>();
  return CompareTensorAccess(op, rhs);
}

bool TensorizeComparator::Dispatch_(const SelectObj* op, const PrimExpr& other) {
  const auto* rhs = other.as<SelectObj>();
  return Dispatch(op->condition, rhs->condition) && Dispatch(op->true_value, rhs->true_value) &&
         Dispatch(op->false_value, rhs->false_value);
}

bool TensorizeComparator::DefEqual(const Var& lhs, const Var& rhs) {
  if (lhs.same_as(rhs)) return true;
  auto it = equal_map_.find(lhs);
  // If there is already a mapping
  if (it != equal_map_.end()) return it->second.same_as(rhs);
  auto lhs_prim_type = lhs->ty.as<PrimType>();
  auto rhs_prim_type = rhs->ty.as<PrimType>();
  if (!lhs_prim_type || !rhs_prim_type) {
    if (!ffi::StructuralEqual()(lhs->ty, rhs->ty)) return false;
    equal_map_.insert_or_assign(lhs, rhs);
    return true;
  }
  // Otherwise remap lhs to rhs
  equal_map_.insert_or_assign(lhs, rhs);
  // Cast if necessary. This allows the workload and the tensor intrin to have different dtypes in
  // the indices.
  {
    auto context = analyzer_->GetConstraintContext();
    context->Append(
        sym::BindingConstraint(lhs, cast(rhs.as_or_throw<PrimExpr>(), lhs_prim_type.value())));
    analyzer_->SyncConstraintContext(context);
  }
  return true;
}

bool TensorizeComparator::CompareAnnotation(const std::pair<ffi::String, ffi::Any>& lhs,
                                            const std::pair<ffi::String, ffi::Any>& rhs) {
  if (lhs.first != rhs.first) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "CompareAnnotation key mismatch: lhs.first=" << lhs.first
         << " vs rhs.first=" << rhs.first;
      EmitError(os.str());
    }
    return false;
  }
  // handle expr values
  if (lhs.second.as<PrimExpr>() && rhs.second.as<PrimExpr>()) {
    return Dispatch(lhs.second.as_or_throw<PrimExpr>(), rhs.second.as_or_throw<PrimExpr>());
  }
  // handle any other values via any equal
  if (!ffi::AnyEqual()(lhs.second, rhs.second)) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "CompareAnnotation value mismatch: lhs.second=" << lhs.first
         << " vs rhs.second=" << rhs.first;
      EmitError(os.str());
    }
    return false;
  }
  return true;
}

bool TensorizeComparator::CompareAnnotationMap(const ffi::Map<ffi::String, ffi::Any>& lhs,
                                               const ffi::Map<ffi::String, ffi::Any>& rhs) {
  if (lhs.same_as(rhs)) return true;
  if (lhs.size() != rhs.size()) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "CompareAnnotationMap size mismatch: lhs.size()=" << lhs.size()
         << " vs rhs.size()=" << rhs.size();
      EmitError(os.str());
    }
    return false;
  }

  auto sort_map = [](const ffi::Map<ffi::String, ffi::Any>& map)
      -> std::vector<std::pair<ffi::String, ffi::Any>> {
    std::vector<std::pair<ffi::String, ffi::Any>> ret(map.begin(), map.end());
    sort(ret.begin(), ret.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    return ret;
  };

  std::vector<std::pair<ffi::String, ffi::Any>> lhs_array = sort_map(lhs);
  std::vector<std::pair<ffi::String, ffi::Any>> rhs_array = sort_map(rhs);

  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!CompareAnnotation(lhs_array[i], rhs_array[i])) {
      if (assert_mode_) {
        std::ostringstream os;
        os << "CompareAnnotationMap annotations mismatch within AnnotationMap.";
        EmitError(os.str());
      }
      return false;
    }
  }
  return true;
}

bool TensorizeComparator::CompareTensor(const TensorVar& lhs, const TensorVar& rhs) {
  if (lhs.same_as(rhs)) return true;
  auto it = rhs_tensor_map_.find(rhs);
  bool equal;
  if (it != rhs_tensor_map_.end()) {
    equal = (*it).second.same_as(lhs);
  } else {
    // Remap the tensor variable definition without recursively comparing its
    // TensorType.  Tensorization intentionally matches a region of a larger
    // workload tensor against the intrinsic's smaller descriptor tensor.
    auto data_it = equal_map_.find(lhs);
    if (data_it != equal_map_.end()) {
      equal = data_it->second.same_as(rhs);
    } else {
      equal = lhs.dtype() == rhs.dtype() && lhs.scope() == rhs.scope();
      if (equal) {
        equal_map_.insert_or_assign(lhs, rhs);
      }
    }
    if (equal) {
      rhs_tensor_map_.insert_or_assign(rhs, lhs);
    } else {
      if (assert_mode_) {
        std::ostringstream os;
        os << "CompareTensor tensor mismatch: " << lhs << " vs " << rhs
           << ", dtypes: " << lhs.dtype() << " vs " << rhs.dtype() << ", scope(): " << lhs.scope()
           << " vs " << rhs.scope();
        EmitError(os.str());
      }
    }
  }
  return equal;
}

bool TensorizeComparator::CompareTensorRegion(const TensorRegion& lhs, const TensorRegion& rhs) {
  if (!CompareTensor(lhs->source.as_or_throw<tvm::tirx::TensorVar>(),
                     rhs->source.as_or_throw<tvm::tirx::TensorVar>())) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "CompareTensorRegion returning false due to tensor mismatch: lhs->source="
         << lhs->source.as_or_throw<tvm::tirx::TensorVar>()
         << " vs rhs->source=" << rhs->source.as_or_throw<tvm::tirx::TensorVar>();
      EmitError(os.str());
    }
    return false;
  }
  int offset = static_cast<int>(lhs->region.size()) - static_cast<int>(rhs->region.size());
  // Number of indices in RHS (desc of the tensor intrinsic) must be smaller than it in LHS
  if (offset < 0) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "CompareTensorRegion returning false because tensor region sizes do not match: "
            "lhs->region.size()="
         << lhs->region.size() << " vs rhs->region.size()=" << rhs->region.size();
      EmitError(os.str());
    }
    return false;
  }

  auto it = tensor_indices_.find(lhs->source.as_or_throw<tvm::tirx::TensorVar>());
  if (it == tensor_indices_.end()) {
    // Update base indices for the tensor, this can only happen if it is visiting the scope block.
    TVM_FFI_ICHECK(is_scope_block);
    std::vector<PrimExpr> indices_base;
    indices_base.reserve(lhs->region.size());
    for (int i = 0; i < offset; i++) {
      // High-dim region must be element-wise
      if (!IsOne(lhs->region[i]->extent)) {
        if (assert_mode_) {
          std::ostringstream os;
          os << "CompareTensorRegion returning false because tensor extent high-dim region must be "
                "element-wise. lhs region width="
             << lhs->region[i]->extent;
          EmitError(os.str());
        }
        return false;
      }
      indices_base.emplace_back(lhs->region[i]->begin);
    }
    for (size_t i = 0; i < rhs->region.size(); i++) {
      // save base index
      indices_base.emplace_back(lhs->region[i + offset]->begin);
      // check extent match
      if (!analyzer_->CanProveEqual(lhs->region[i + offset]->extent, rhs->region[i]->extent)) {
        if (assert_mode_) {
          std::ostringstream os;
          os << "CompareTensorRegion tensor extent mismatch: lhs->region[i + offset]="
             << lhs->region[i + offset] << " vs rhs->region[i]=" << rhs->region[i];
          EmitError(os.str());
        }
        return false;
      }
    }
    tensor_indices_.emplace(lhs->source.as_or_throw<tvm::tirx::TensorVar>(),
                            std::move(indices_base));
  } else {
    // Check the base indices are consistent.
    const std::vector<PrimExpr>& indices_base = it->second;
    for (int i = 0; i < offset; i++) {
      // High-dim region must be element-wise
      if (!IsOne(lhs->region[i]->extent)) {
        if (assert_mode_) {
          std::ostringstream os;
          os << "CompareTensorRegion returning false because tensor extent high-dim region must be "
                "element-wise. lhs region width="
             << lhs->region[i]->extent;
          EmitError(os.str());
        }
        return false;
      }
      if (!lhs_analyzer_->CanProveEqual(indices_base[i], lhs->region[i]->begin)) {
        if (assert_mode_) {
          std::ostringstream os;
          os << "TensorVar base index consistency check failed due to unequal index base: "
                "indices_base[i]="
             << indices_base[i] << " vs lhs->region[i]->begin=" << lhs->region[i]->begin;
          EmitError(os.str());
        }
        return false;
      }
    }
    for (size_t i = 0; i < rhs->region.size(); i++) {
      // check extent match
      if (!analyzer_->CanProveEqual(lhs->region[i + offset]->extent, rhs->region[i]->extent)) {
        if (assert_mode_) {
          std::ostringstream os;
          os << "CompareTensorRegion tensor region extent mismatch. lhs->region[i + offset]="
             << lhs->region[i + offset] << " vs rhs->region[i]=" << rhs->region[i];
          EmitError(os.str());
        }
        return false;
      }
      PrimExpr normalized_lhs_min =
          lhs_analyzer_->Simplify((lhs->region[i + offset]->begin - indices_base[i + offset]));
      if (!analyzer_->CanProveEqual(normalized_lhs_min, rhs->region[i]->begin)) {
        if (assert_mode_) {
          std::ostringstream os;
          os << "CompareTensorRegion tensor region min mismatch. lhs->region[i + offset]="
             << lhs->region[i + offset] << " vs rhs->region[i]=" << rhs->region[i];
          EmitError(os.str());
        }
        return false;
      }
    }
  }
  return true;
}

// Comparator for TensorStoreObj and TensorLoadObj
inline TensorVar GetTensorAccessTensor(const TensorStoreObj* op) {
  return op->dest.as_or_throw<TensorVar>();
}
inline TensorVar GetTensorAccessTensor(const TensorLoadObj* op) {
  return op->source.as_or_throw<tvm::tirx::TensorVar>();
}

template <typename T>
bool TensorizeComparator::CompareTensorAccess(const T* lhs, const T* rhs) {
  TensorVar lhs_tensor = GetTensorAccessTensor(lhs);
  TensorVar rhs_tensor = GetTensorAccessTensor(rhs);
  if (!CompareTensor(lhs_tensor, rhs_tensor)) return false;
  int offset = static_cast<int>(lhs->indices.size()) - static_cast<int>(rhs->indices.size());
  if (offset < 0) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "CompareTensorAccess returning false because tensor indices sizes do not match: "
            "lhs->indices.size()="
         << lhs->indices.size() << " vs rhs->indices.size()=" << rhs->indices.size();
      EmitError(os.str());
    }
    return false;
  }
  auto it = tensor_indices_.find(lhs_tensor);
  TVM_FFI_ICHECK(it != tensor_indices_.end());
  const std::vector<PrimExpr>& indices_base = (*it).second;
  TVM_FFI_ICHECK_EQ(indices_base.size(), rhs->indices.size() + offset);
  for (size_t i = 0; i < rhs->indices.size(); i++) {
    PrimExpr normalized_lhs_index = lhs->indices[i + offset] - indices_base[i + offset];
    if (!analyzer_->CanProveEqual(normalized_lhs_index, rhs->indices[i])) {
      if (assert_mode_) {
        std::ostringstream os;
        os << "CompareTensorAccess tensor indices mismatch. lhs->indices[i + offset]="
           << lhs->indices[i + offset] << " vs rhs->indices[i]=" << rhs->indices[i];
        EmitError(os.str());
      }
      return false;
    }
  }
  return true;
}

template <typename T, typename Self, typename F>
bool TensorizeComparator::CompareArray(const ffi::Array<T>& lhs, const ffi::Array<T>& rhs,
                                       F Self::* cmp) {
  if (lhs.same_as(rhs)) return true;
  if (lhs.size() != rhs.size()) {
    if (assert_mode_) {
      std::ostringstream os;
      os << "CompareArray array size mismatch. lhs.size()=" << lhs.size()
         << " vs rhs.size()=" << rhs.size();
      EmitError(os.str());
    }
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!(static_cast<Self*>(this)->*cmp)(lhs[i], rhs[i])) return false;
  }
  return true;
}

bool TensorizeComparator::CompareRange(const Range& lhs, const Range& rhs) {
  return Dispatch(lhs->begin, rhs->begin) && Dispatch(lhs->extent, rhs->extent);
}

bool TensorizeComparator::CompareIterVar(const IterVar& lhs, const IterVar& rhs) {
  return DefEqual(lhs->var, rhs->var) && lhs->iter_type == rhs->iter_type;
}

void TensorizeComparator::EmitError(const std::string& error_message) {
  error_messages_.push_back(error_message);
}

/******** AutoTensorize Extractor ********/

bool AutoTensorizeComparator::DispatchDefault_(const ffi::Object* op, const PrimExpr& other) {
  return false;
}

bool AutoTensorizeComparator::DispatchDefault_(const ffi::Object* op, const Stmt& other) {
  return false;
}

bool AutoTensorizeComparator::Dispatch_(const SBlockObj* op, const Stmt& other) {
  const auto* rhs = other.as<SBlockObj>();
  // Check block equality.
  // All iter vars and tensor regions including the order should match.
  // When checking iter vars, DefEqual is used to remap variables.
  if (!is_scope_block) {
    if (!CompareArray(op->iter_vars, rhs->iter_vars, &AutoTensorizeComparator::CompareIterVar)) {
      return false;
    }
    if (!CompareAnnotationMap(op->annotations, rhs->annotations)) {
      return false;
    }
    if (!CompareArray(op->alloc_tensors, rhs->alloc_tensors,
                      &AutoTensorizeComparator::CompareTensor)) {
      return false;
    }
    for (const IterVar& block_iter : op->iter_vars) {
      inner_iter_dom_map_.Set(block_iter->var, sym::IntSet::FromRange(block_iter->dom.value()));
    }
  } else {
    auto collect_iter = [&](const SBlockObj* op, std::vector<IterVar>& iters) -> bool {
      for (const auto& iter : op->iter_vars) {
        {
          auto context = analyzer_->GetConstraintContext();
          context->Append(sym::RangeConstraint(iter->var, iter->dom.value()));
          analyzer_->SyncConstraintContext(context);
        }
        if (iter->iter_type == IterVarType::kDataPar ||
            iter->iter_type == IterVarType::kCommReduce) {
          iters.push_back(iter);
        } else {
          return false;
        }
      }
      return true;
    };
    if (!collect_iter(op, lhs_iters_)) {
      return false;
    }
    if (!collect_iter(rhs, rhs_iters_)) {
      return false;
    }
  }
  is_scope_block = false;
  return Dispatch(op->body, rhs->body);
}

bool AutoTensorizeComparator::CompareTensor(const TensorVar& lhs, const TensorVar& rhs) {
  if (lhs.same_as(rhs)) return true;
  auto it = rhs_tensor_map_.find(rhs);
  bool equal;
  if (it != rhs_tensor_map_.end()) {
    equal = (*it).second.same_as(lhs);
  } else {
    // Remap the tensor itself, skipping tensor shape and storage scope.  Auto
    // tensorization inserts the cache stages that move workload tensors into
    // an intrinsic's required scope, while the element dtype must still agree.
    auto data_it = equal_map_.find(lhs);
    if (data_it != equal_map_.end()) {
      equal = data_it->second.same_as(rhs);
    } else {
      equal = lhs.dtype() == rhs.dtype();
      if (equal) {
        equal_map_.insert_or_assign(lhs, rhs);
      }
    }
    if (equal) {
      rhs_tensor_map_.insert_or_assign(rhs, lhs);
      lhs_tensor_map_.insert_or_assign(lhs, rhs);
    }
  }
  return equal;
}

bool AutoTensorizeComparator::Dispatch_(const TensorStoreObj* op, const Stmt& other) {
  const auto* rhs = other.as<TensorStoreObj>();
  return CompareTensorAccess(op, rhs) && Dispatch(op->value, rhs->value);
}

bool AutoTensorizeComparator::Dispatch_(const TensorLoadObj* op, const PrimExpr& other) {
  const auto* rhs = other.as<TensorLoadObj>();
  return CompareTensorAccess(op, rhs);
}

template <typename T>
bool AutoTensorizeComparator::CompareTensorAccess(const T* lhs, const T* rhs) {
  TensorVar lhs_tensor = GetTensorAccessTensor(lhs);
  TensorVar rhs_tensor = GetTensorAccessTensor(rhs);
  if (!CompareTensor(lhs_tensor, rhs_tensor)) return false;
  auto it_lhs = lhs_tensor_indices_map_.find(lhs_tensor);
  if (it_lhs == lhs_tensor_indices_map_.end()) {
    if (rhs_tensor_indices_map_.find(rhs_tensor) != rhs_tensor_indices_map_.end()) {
      return false;
    }
    std::vector<PrimExpr> lhs_indices;
    for (const PrimExpr& index : lhs->indices) {
      lhs_indices.push_back(SimplifyNonTrivialExpr(index, analyzer_.get()));
    }

    auto is_scalar_access = [](const ffi::Array<PrimExpr>& indices, PrimExpr index) {
      // Check if the indexing is of the form C[0]
      if (indices.size() > 1) return false;
      auto int_imm = index.template as<IntImmObj>();
      if (int_imm && int_imm->value == 0) return true;
      return false;
    };

    for (const auto& index : rhs->indices) {
      if (!index.template as<PrimVar>() && !is_scalar_access(rhs->indices, index)) return false;
    }
    lhs_tensor_indices_map_[lhs_tensor] = lhs_indices;
    rhs_tensor_indices_map_[rhs_tensor] = rhs->indices;
  } else {
    auto it_rhs = rhs_tensor_indices_map_.find(rhs_tensor);
    if (it_rhs == rhs_tensor_indices_map_.end()) {
      return false;
    }
    auto indices_check = [&](const ffi::Array<PrimExpr>& indices,
                             const ffi::Array<PrimExpr>& old_indices) -> bool {
      if (indices.size() != old_indices.size()) {
        return false;
      }
      for (size_t i = 0; i < indices.size(); ++i) {
        if (!analyzer_->CanProveEqual(indices[i], old_indices[i])) {
          return false;
        }
      }
      return true;
    };
    if (!indices_check(lhs->indices, it_lhs->second)) return false;
    if (!indices_check(rhs->indices, it_rhs->second)) return false;
  }
  return true;
}

}  // namespace s_tir
}  // namespace tvm
