/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file compact_tensor_region.cc
 * \brief Compact the tensor size into its exact need.
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/transform.h>
#include <tvm/support/gpu_thread_tag.h>
#include <tvm/sym/int_set.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/type.h>

#include <numeric>
#include <stack>

#include "../../support/arena.h"
#include "../analysis/conditional_bounds.h"
#include "../schedule/utils.h"
#include "../support/nd_int_set.h"
#include "../tensor_region.h"
#include "ir_utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

using support::NDIntSet;

/*! \brief a more constrained bound estimate for n-dimentional int set */
NDIntSet NDIntSetEval(ffi::Array<Range> region, PrimExpr predicate,
                      const ffi::Map<Var, sym::IntSet>& dom_map, sym::AnalyzerObj* analyzer) {
  std::unordered_map<Var, Range, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> var_dom;
  for (const auto& it : dom_map) {
    var_dom.insert_or_assign(it.first, it.second.CoverRange(Range::FromMinExtent(0, 0)).value());
  }
  sym::Analyzer analyzer_ref = ffi::GetRef<sym::Analyzer>(analyzer);
  ffi::Optional<ffi::Array<sym::IntSet>> eval_res =
      sym::EstimateRegionUpperBound(region, var_dom, predicate, analyzer_ref);

  if (eval_res.has_value()) {
    return NDIntSet(eval_res.value().begin(), eval_res.value().end());
  }
  return support::NDIntSetEval(support::NDIntSetFromRegion(region), dom_map);
}

/*!
 * \brief Collect tensor aliasing information.
 */
class Var2TensorCollector : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;
  /*! \brief Map the tensor var to all aliased tensors. */
  std::unordered_map<Var, std::unordered_set<TensorVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>>
      var2tensor_;

 private:
  ffi::Optional<VisitInterrupt> Visit_(const TensorStoreObj* op) final {
    var2tensor_[op->dest.as_or_throw<TensorVar>()].insert(op->dest.as_or_throw<TensorVar>());
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadObj* op) final {
    TensorVar tensor = op->source.as_or_throw<tvm::tirx::TensorVar>();
    var2tensor_[tensor].insert(tensor);
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const SBlockObj* op) final {
    for (const TensorVar& tensor : op->alloc_tensors) {
      var2tensor_[tensor].insert(tensor);
    }
    for (const MatchTensorRegion& region : op->match_tensors) {
      var2tensor_[region->tensor].insert(region->tensor);
      var2tensor_[region->source->source.as_or_throw<tvm::tirx::TensorVar>()].insert(
          region->source->source.as_or_throw<tvm::tirx::TensorVar>());
    }
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const BindObj* op) final {
    if (auto tensor = op->var.as<TensorVar>()) {
      var2tensor_[op->var].insert(tensor.value());
    }
    return StmtExprVisitor::Visit_(op);
  }
};

/*!
 * \brief Collect the access region of each tensor.
 * \note The param tensor regions will not be collected.
 */
class TensorAccessRegionCollector : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;
  static std::unordered_map<TensorVar, ffi::Array<Range>, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
  Collect(const Function& f, bool collect_inbound) {
    auto region_collector = ffi::make_object<TensorAccessRegionCollector>(collect_inbound);
    // collect tensor var to aliased tensor mapping
    auto var2tensor_collector = ffi::make_object<Var2TensorCollector>();
    var2tensor_collector->Visit(f->body);
    std::swap(region_collector->var2tensor_, var2tensor_collector->var2tensor_);

    // collect tensor access regions
    region_collector->Visit(f->body);
    // Compact any remaining flat AllocTensor nodes at function scope
    region_collector->CompactPendingFlatAllocTensors();
    return std::move(region_collector->tensor_access_region_);
  }

 private:
  static bool IsIndexTypedExpr(const PrimExpr& expr) {
    PrimType type = expr.ty();
    return type.IsScalar() && type.MatchesCode(DLDataTypeCode::kDLInt) &&
           (type.bits() == 32 || type.bits() == 64);
  }

  struct TensorAccessInfo {
    /*! \brief The tensor. */
    TensorVar tensor;
    /*! \brief The tensor access region, which can be updated during visiting. */
    NDIntSet accessed_region;

    explicit TensorAccessInfo(const TensorVar& tensor, const NDIntSet& region)
        : tensor(tensor), accessed_region(region) {}
  };

 public:
  explicit TensorAccessRegionCollector(bool collect_inbound) : collect_inbound_(collect_inbound) {}

 private:
  /**************** Visitor overload ****************/

  // Declared regions carry bounds, not opaque runtime accesses.
  ffi::Optional<VisitInterrupt> Visit_(const TensorRegionObj* op) final {
    if (!op->source.as<TensorVar>()) return StmtExprVisitor::Visit_(op);
    for (const Range& range : op->region) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(range->begin));
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(range->extent));
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorStoreObj* op) final {
    VisitTensorAccess(s_tir::TensorRegionFromPoint(op->dest.as_or_throw<TensorVar>(), op->indices));
    return Visit(op->value);
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadObj* op) final {
    TensorVar tensor = op->source.as_or_throw<tvm::tirx::TensorVar>();
    auto explicit_it = explicit_access_annotations_.find(tensor);
    if (explicit_it != explicit_access_annotations_.end()) {
      VisitTensorAccess(explicit_it->second);
    } else {
      VisitTensorAccess(s_tir::TensorRegionFromPoint(tensor, op->indices));
    }
    for (const auto& index : op->indices) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(index));
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const VarObj* op) final {
    if (def_region_kind() != kTVMFFIDefRegionKindNone) return StmtExprVisitor::Visit_(op);
    VisitTensorVar(ffi::GetRef<Var>(op));
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const ForObj* op) final {
    auto outer_context = dom_analyzer_->GetConstraintContext();
    Range loop_range = Range(op->begin, op->end);
    IterVar iter = tvm::s_tir::GetThreadBinding(op).has_value()
                       ? IterVar(std::nullopt, op->var, IterVarType::kThreadIndex,
                                 tvm::s_tir::GetThreadBinding(op).value())
                       : IterVar(std::nullopt, op->var, IterVarType::kDataPar);
    ancestor_iters_.push_back(iter);
    {
      auto context = dom_analyzer_->GetConstraintContext();
      context->Append(sym::RangeConstraint(op->var, loop_range));
      dom_analyzer_->SyncConstraintContext(context);
    }
    dom_map_.Set(op->var, sym::IntSet::FromRange(loop_range));
    size_t n_pending_before = pending_flat_alloc_tensors_.size();
    if (auto interrupt = StmtExprVisitor::Visit_(op)) {
      dom_analyzer_->SyncConstraintContext(outer_context);
      return interrupt;
    }
    // Compact flat AllocTensors defined inside this For scope
    CompactPendingFlatAllocTensors(n_pending_before);
    dom_analyzer_->SyncConstraintContext(outer_context);
    dom_map_.erase(op->var);
    ancestor_iters_.pop_back();
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const BindObj* op) final {
    if (const auto* call = op->value.as<CallObj>();
        call && call->op.same_as(tirx::alloc_tensor_op())) {
      return DispatchAllocTensor(op);
    }
    if (const auto* call = op->value.as<CallObj>();
        call && call->op.same_as(tirx::decl_tensor_op()))
      return StmtExprVisitor::Visit_(op);
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit(op->value));
    if (auto value = op->value.as<PrimExpr>(); value && IsIndexTypedExpr(value.value())) {
      {
        auto context = dom_analyzer_->GetConstraintContext();
        context->Append(sym::BindingConstraint(op->var, value.value()));
        dom_analyzer_->SyncConstraintContext(context);
      }
      dom_map_.Set(op->var, sym::IntSet::SinglePoint(value.value()));
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const LetObj* op) final {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit(op->value));
    auto outer_context = dom_analyzer_->GetConstraintContext();
    if (IsIndexTypedExpr(op->value)) {
      {
        auto context = dom_analyzer_->GetConstraintContext();
        context->Append(sym::BindingConstraint(op->var, op->value));
        dom_analyzer_->SyncConstraintContext(context);
      }
      dom_map_.Set(op->var, sym::IntSet::SinglePoint(op->value));
    }
    if (auto interrupt = StmtExprVisitor::Visit(op->body)) {
      dom_analyzer_->SyncConstraintContext(outer_context);
      return interrupt;
    }
    dom_analyzer_->SyncConstraintContext(outer_context);
    if (IsIndexTypedExpr(op->value)) {
      dom_map_.erase(op->var);
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const IfObj* op) final {
    // Visit condition
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit(op->condition));
    {
      // Visit then branch
      With<ConditionalBoundsContext> ctx(op->condition, &dom_map_, &hint_map_,
                                         &pending_conditions_);
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit(op->then_case));
    }
    if (op->else_case) {
      // Visit else branch
      With<ConditionalBoundsContext> ctx(!op->condition, &dom_map_, &hint_map_,
                                         &pending_conditions_);
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit(op->else_case.value()));
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const CallObj* op) final {
    if (op->op.same_as(prim::if_then_else_op())) {
      PrimExpr condition = op->args[0].as_or_throw<PrimExpr>();
      PrimExpr then_value = op->args[1].as_or_throw<PrimExpr>();
      PrimExpr else_value = op->args[2].as_or_throw<PrimExpr>();
      // Visit condition
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit(condition));
      {
        // Visit then branch
        With<ConditionalBoundsContext> ctx(condition, &dom_map_, &hint_map_, &pending_conditions_);
        TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit(then_value));
      }
      {
        // Visit else branch
        With<ConditionalBoundsContext> ctx(!condition, &dom_map_, &hint_map_, &pending_conditions_);
        TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit(else_value));
      }
      return std::nullopt;
    }
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const SBlockObj* op) final {
    // Step 0. Check there is no init part and block is opaque
    TVM_FFI_ICHECK(!op->init.has_value());
    TVM_FFI_ICHECK_EQ(op->iter_vars.size(), 0) << "CompactTensorRegion only works on opaque blocks";
    // Step 1. Record and update current read/write region annotations
    std::unordered_map<TensorVar, std::vector<TensorRegion>, ffi::ObjectPtrHash,
                       ffi::ObjectPtrEqual>
        cur_access_annotations;
    for (const TensorRegion& region : op->reads) {
      cur_access_annotations[region->source.as_or_throw<tvm::tirx::TensorVar>()].push_back(region);
    }
    for (const TensorRegion& region : op->writes) {
      cur_access_annotations[region->source.as_or_throw<tvm::tirx::TensorVar>()].push_back(region);
    }
    for (auto& p : cur_access_annotations) {
      auto& regions = access_annotations_[p.first];
      p.second.swap(regions);
    }

    // Step 2. Record explicit read/write region annotations
    auto record_explicit_region = [&](const ffi::String& attr_key, TensorIndexType index_type) {
      auto it = op->annotations.find(attr_key);
      if (it != op->annotations.end()) {
        ffi::Array<int64_t> tensor_indices = (*it).second.as_or_throw<ffi::Array<int64_t>>();
        for (int64_t index : tensor_indices) {
          int tensor_index = static_cast<int>(index);
          if (tensor_index >= 0 && tensor_index < static_cast<int>(op->reads.size())) {
            const TensorRegion& explicit_region = index_type == TensorIndexType::kRead
                                                      ? op->reads[tensor_index]
                                                      : op->writes[tensor_index];
            explicit_access_annotations_.insert_or_assign(
                explicit_region->source.as_or_throw<tvm::tirx::TensorVar>(), explicit_region);
          }
        }
      }
    };

    record_explicit_region(s_tir::attr::kExplicitReadRegion.name, TensorIndexType::kRead);
    record_explicit_region(s_tir::attr::kExplicitWriteRegion.name, TensorIndexType::kWrite);

    // Step 3. Record relax position of ancestor_loops_
    for (const TensorVar& tensor : op->alloc_tensors) {
      RecordTensorDefinition(tensor);
    }
    // Step 4. Visit match tensors
    for (const MatchTensorRegion& region : op->match_tensors) {
      VisitTensorAccess(region->source);
    }
    // Step 5. Visit block body recursively
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
    // Step 6. Recover read/write region annotations
    for (auto& p : cur_access_annotations) {
      auto& regions = access_annotations_[p.first];
      if (p.second.empty()) {
        access_annotations_.erase(p.first);
      } else {
        regions.swap(p.second);
      }
    }
    // Step 7. Clear explicit access annotations
    explicit_access_annotations_.clear();
    // Step 8. Update tensor_access_region_ from relaxed_accesses_ for inner tensors.
    for (const TensorVar& tensor : op->alloc_tensors) {
      TVM_FFI_ICHECK_EQ(var2tensor_[tensor].size(), 1)
          << "Block allocation tensor shoud not be alised";
      SimplifyAndNarrowTensorRegionFromNDIntSet(tensor);
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const SBlockRealizeObj* op) final {
    With<ConditionalBoundsContext> ctx(op->predicate, &dom_map_, &hint_map_, &pending_conditions_);
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> DispatchAllocTensor(const BindObj* op) {
    // AllocTensor is flat: register the tensor def and track for post-scope compaction.
    RecordTensorDefinition(op->var.as_or_throw<TensorVar>());
    pending_flat_alloc_tensors_.push_back(op->var.as_or_throw<TensorVar>());
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const RegionStmtObj* op) final {
    if (op->op.same_as(tirx::launch_thread_op())) {
      auto outer_context = dom_analyzer_->GetConstraintContext();
      PrimExpr extent = op->args[1].as_or_throw<PrimExpr>();
      Range dom = Range::FromMinExtent(IntImm(extent.ty(), 0), extent);
      IterVar iter(dom, op->body_params[0].as_or_throw<PrimVar>(), IterVarType::kThreadIndex,
                   op->args[0].as_or_throw<StringImm>()->value);
      ancestor_iters_.push_back(iter);
      {
        auto context = dom_analyzer_->GetConstraintContext();
        context->Append(sym::RangeConstraint(iter->var, dom));
        dom_analyzer_->SyncConstraintContext(context);
      }
      dom_map_.Set(iter->var, sym::IntSet::FromRange(dom));
      size_t n_pending_before = pending_flat_alloc_tensors_.size();
      if (auto interrupt = StmtExprVisitor::Visit_(op)) {
        dom_analyzer_->SyncConstraintContext(outer_context);
        return interrupt;
      }
      CompactPendingFlatAllocTensors(n_pending_before);
      dom_analyzer_->SyncConstraintContext(outer_context);
      dom_map_.erase(iter->var);
      ancestor_iters_.pop_back();
      return std::nullopt;
    }
    size_t n_pending_before = pending_flat_alloc_tensors_.size();
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
    CompactPendingFlatAllocTensors(n_pending_before);
    return std::nullopt;
  }

  /**************** Helper functions ****************/

  /*! \brief Record information on the tensor defining point. */
  void RecordTensorDefinition(const Var& tensor_data) {
    auto it = tensor_scope_depth_.find(tensor_data);
    TVM_FFI_ICHECK(it == tensor_scope_depth_.end()) << tensor_data << " has duplicate definitions";
    tensor_scope_depth_.insert(it, {tensor_data, ancestor_iters_.size()});
  }

  void VisitTensorAccess(const TensorRegion& tensor_region) {
    const TensorVar& tensor = tensor_region->source.as_or_throw<tvm::tirx::TensorVar>();
    auto it = tensor_scope_depth_.find(tensor);
    if (it != tensor_scope_depth_.end()) {
      size_t n_ancestor_loops = it->second;
      // Step 1. Stop ancestor loop vars out of the allocation block from
      // being relaxed unless NeedRelaxThread() is true.
      std::vector<ffi::Optional<sym::IntSet>> non_relaxed(n_ancestor_loops);
      for (size_t i = 0; i < n_ancestor_loops; ++i) {
        const IterVar& iter = ancestor_iters_[i];
        const Var& v = iter->var;
        if (NeedRelaxThread(iter, tensor.scope())) {
          continue;
        }
        auto dom_it = dom_map_.find(v);
        TVM_FFI_ICHECK(dom_it != dom_map_.end())
            << "Could not find domain for loop variable " << v->name;
        non_relaxed[i] = (*dom_it).second;
        dom_map_.erase(v);
      }
      // Step 2. Relax the access region
      auto normalize_pred = [](const PrimExpr& pred) {
        PrimType pred_ty = pred.ty();
        if (pred_ty.MatchesCode(DLDataTypeCode::kDLBool)) return pred;
        return pred != IntImm(pred.ty(), 0);
      };
      PrimExpr predicate = dom_analyzer_->Simplify(std::accumulate(
          pending_conditions_.begin(), pending_conditions_.end(), PrimExpr(IntImm::Bool(true)),
          [normalize_pred](const PrimExpr& x, const PrimExpr& y) {
            return normalize_pred(x) && normalize_pred(y);
          }));
      NDIntSet nd_int_set =
          NDIntSetEval(tensor_region->region, predicate, dom_map_, dom_analyzer_.get());

      // Step 3. Restore the non-relaxed ancestor loops domain
      for (size_t i = 0; i < n_ancestor_loops; ++i) {
        const Var& v = ancestor_iters_[i]->var;
        if (non_relaxed[i].has_value()) dom_map_.Set(v, non_relaxed[i].value());
      }
      // Step 4. Update relaxed_accesses_ dict
      auto access_it = relaxed_accesses_.find(tensor);
      if (access_it != relaxed_accesses_.end()) {
        support::NDIntSetUnionWith(&access_it->second, nd_int_set);
      } else {
        relaxed_accesses_.insert(access_it, {tensor, nd_int_set});
      }
    }
  }

  void VisitTensorVar(const Var& var) {
    auto it = var2tensor_.find(var);
    if (it == var2tensor_.end()) {
      return;
    }
    for (const TensorVar& tensor : it->second) {
      auto annotation_it = access_annotations_.find(tensor);
      if (annotation_it != access_annotations_.end()) {
        // opaque tensor has explicit accessed region annotations
        for (const TensorRegion& region : annotation_it->second) {
          VisitTensorAccess(region);
        }
      } else {
        VisitTensorAccess(s_tir::FullTensorRegion(tensor));
      }
    }
  }

  /*! \brief Check whether the thread binding iter should be relaxed with given storage scope. */
  static bool NeedRelaxThread(const IterVar& iter, const runtime::MemoryScope& scope) {
    if (iter->iter_type != IterVarType::kThreadIndex) {
      return false;
    }
    // When there is warp memory
    // threadIdx.x must be set to be warp index.
    return CanRelaxStorageUnderThread(scope, ParseThreadBindingTag((iter->thread_tag)));
  }

  /*!
   * \brief simplify and narrow down the region collected by NDIntSet.
   * Update the `relaxed_accesses_` dict. If `collect_inbound_` is true,
   * the result region would never exceed the original tensor shape.
   */
  void SimplifyAndNarrowTensorRegionFromNDIntSet(const TensorVar& tensor) {
    auto it = relaxed_accesses_.find(tensor);
    TVM_FFI_ICHECK(it != relaxed_accesses_.end())
        << tensor << " is allocated but not accessed within block scope";

    const ffi::Array<PrimExpr>& original_shape = tensor.shape();
    const NDIntSet& nd_int_set = it->second;
    ffi::Array<Range>& result_region = tensor_access_region_[tensor];
    result_region.resize(nd_int_set.size());

    for (size_t i = 0; i < nd_int_set.size(); ++i) {
      const sym::IntSet& int_set = nd_int_set[i];
      Range original =
          Range(/*begin=*/IntImm(original_shape[i].ty(), 0), /*end=*/original_shape[i]);
      Range range = int_set.CoverRange(original).value();
      PrimExpr min{ffi::UnsafeInit{}};
      PrimExpr extent{ffi::UnsafeInit{}};
      if (collect_inbound_) {
        min = dom_analyzer_->Simplify(tvm::max(0, range->begin));
        extent = range->extent;
        // Apply stronger symbolic proof to help us remove symbolic min here.
        if (!dom_analyzer_->CanProveLessEqualThanSymbolicShapeValue(extent, original_shape[i])) {
          extent = tvm::min(original_shape[i], range->extent);
        }
        extent = dom_analyzer_->Simplify(extent);
      } else {
        min = dom_analyzer_->Simplify(range->begin);
        extent = dom_analyzer_->Simplify(range->extent);
      }

      // We check the tensor extent is pure and not loop dependent, since loop dependent
      // or data dependent allocation is not supported yet. Otherwise we should
      // fallback to use original tensor shape.
      if (SideEffect(extent) > CallEffectKind::kPure) {
        result_region.Set(i, original);
        continue;
      }
      auto is_loop_var = [this](const VarObj* v) {
        return std::any_of(ancestor_iters_.begin(), ancestor_iters_.end(),
                           [v](const IterVar& n) { return n->var.get() == v; });
      };
      auto walkfn = [&](const Var& var) -> ffi::Expected<ffi::WalkResult> {
        return is_loop_var(var.get()) ? ffi::WalkResult::Interrupt(ffi::VisitInterrupt(var))
                                      : ffi::WalkResult::Advance();
      };
      if (ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(extent, walkfn).has_value()) {
        // try estimate a constant upperbound on region's extent
        int64_t upperbound = dom_analyzer_->const_int_bound(extent)->max_value;
        if (upperbound != sym::ConstIntBound::kPosInf) {
          extent = IntImm(extent.ty(), upperbound);
        } else {
          result_region.Set(i, original);
          continue;
        }
      }
      result_region.Set(i, Range::FromMinExtent(min, extent));
    }
  }

  /*!
   * \brief Compact pending flat AllocTensor nodes registered since position n_before.
   * Call SimplifyAndNarrowTensorRegionFromNDIntSet for each, then remove them.
   */
  void CompactPendingFlatAllocTensors(size_t n_before = 0) {
    for (size_t i = n_before; i < pending_flat_alloc_tensors_.size(); ++i) {
      const TensorVar& tensor = pending_flat_alloc_tensors_[i];
      auto it = relaxed_accesses_.find(tensor);
      if (it != relaxed_accesses_.end()) {
        SimplifyAndNarrowTensorRegionFromNDIntSet(tensor);
      }
    }
    pending_flat_alloc_tensors_.erase(pending_flat_alloc_tensors_.begin() + n_before,
                                      pending_flat_alloc_tensors_.end());
  }

  /**************** Class members ****************/
  /*! \brief Only collect accessed region within original tensor shape bound. */
  bool collect_inbound_{true};
  /*! \brief Pending flat AllocTensor nodes to compact when leaving scope. */
  std::vector<TensorVar> pending_flat_alloc_tensors_;

  /*! \brief The iteration scopes from the current node up to the root. */
  std::vector<IterVar> ancestor_iters_;

  /*!
   * \brief Map each tensor var to the n_ancester_loop. which is the loop depth at the
   * define point. ancestor_loops_[0: n_ancester_loop] should not be relaxed when
   * we evaluate this tensor's access regions.
   */
  std::unordered_map<Var, size_t> tensor_scope_depth_;

  /*! \brief Map the tensor var to all aliased tensors. */
  std::unordered_map<Var, std::unordered_set<TensorVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>>
      var2tensor_;

  /*! \brief The map from loop vars to their iter range. */
  ffi::Map<Var, sym::IntSet> dom_map_;
  /*! \brief Extra map from free vars to their iter range hints. */
  ffi::Map<Var, sym::IntSet> hint_map_;
  /*! \brief Unresolved conditions within current scope. */
  std::vector<PrimExpr> pending_conditions_;
  /*! \brief The analyzer aware of loop domains. */
  sym::Analyzer dom_analyzer_;
  /*! \brief The map from TensorVar to it's relaxed access set. */
  std::unordered_map<TensorVar, NDIntSet, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      relaxed_accesses_;

  /*!
   * \brief The map from TensorVar to it entire access region, used for returning.
   * The entire access region should get updated on the tensor's define point
   * and we sanity check that every tensor is defined only once.
   */
  std::unordered_map<TensorVar, ffi::Array<Range>, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      tensor_access_region_;

  /*! \brief The map from TensorVar to it's access regions annotated by current block. */
  std::unordered_map<TensorVar, std::vector<TensorRegion>, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      access_annotations_;
  /*! \brief The map from TensorVar to its explicit access region annotated by the block. */
  std::unordered_map<TensorVar, TensorRegion, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      explicit_access_annotations_;
};

/*! \brief The storage alignment for a dimension */
struct DimAlignInfo {
  /*! \brief The factor of the alignment */
  int align_factor{0};
  /*! \brief The offset of the alignment */
  int align_offset{0};
};

struct TensorAllocInfo {
  /*! \brief The tensor access region. */
  ffi::Array<Range> region;
  /*! \brief The storage alignment information. */
  std::vector<DimAlignInfo> dim_aligns;
  /*!
   * \brief The reallocated tensor with minimal size.
   * \note The value if std::nullopt if the tensor do not need reallocate (e.g parameter tensor).
   */
  TensorVar new_tensor{ffi::UnsafeInit{}};
};

/*! \brief Reallocate the tensors with minimal region. */
class TensorCompactor : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  explicit TensorCompactor(std::unordered_map<Var, TensorAllocInfo> tensor_info)
      : tensor_info_(std::move(tensor_info)) {}

  UnchangedOr<Stmt> Mutate_(const TensorStoreObj* _op, InplaceMode inplace_mode) final {
    TensorVar original_tensor = _op->dest.as_or_throw<TensorVar>();
    TensorStore store = StmtExprMutator::Mutate_(_op, inplace_mode)
                            .ValueOrUnchanged(ffi::GetRef<Stmt>(_op))
                            .as_or_throw<TensorStore>();
    TensorStoreObj* op = store.CopyOnWrite();
    TensorVar tensor = op->dest.as_or_throw<TensorVar>();
    RewriteTensorAccess(original_tensor, &tensor, &op->indices);
    op->dest = std::move(tensor);
    return store;
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* _op, InplaceMode inplace_mode) final {
    TensorVar original_tensor = _op->source.as_or_throw<tvm::tirx::TensorVar>();
    TensorLoad load = StmtExprMutator::Mutate_(_op, inplace_mode)
                          .ValueOrUnchanged(ffi::GetRef<PrimExpr>(_op))
                          .as_or_throw<TensorLoad>();
    TensorVar tensor = load->source.as_or_throw<tvm::tirx::TensorVar>();
    ffi::Array<PrimExpr> indices = load->indices;
    RewriteTensorAccess(original_tensor, &tensor, &indices);
    return TensorLoad(tensor, indices, load->loc);
  }

  UnchangedOr<Stmt> Mutate_(const SBlockObj* op, InplaceMode inplace_mode) final {
    // Step 0. Check there is no Init part.
    TVM_FFI_ICHECK(!op->init.has_value());
    // Rewrite the signature while its tensor identities still match tensor_info_.
    SBlock block = ffi::GetRef<SBlock>(op);
    SBlockObj* n = block.CopyOnWrite();
    RewriteTensorRegions(&n->reads);
    RewriteTensorRegions(&n->writes);
    RewriteMatchTensors(&n->match_tensors);
    n->alloc_tensors = op->alloc_tensors.Map(
        [this](const TensorVar& tensor) { return RewriteAllocTensor(tensor); });
    // Recursively rewrite the body after installing the allocation remaps.
    return StmtExprMutator::Mutate_(block.get(),
                                    block.unique() ? inplace_mode : InplaceMode::kDisallow)
        .ValueOrUnchanged(block);
  }

  UnchangedOr<Stmt> Mutate_(const BindObj* op, InplaceMode inplace_mode) final {
    const auto* call = op->value.as<CallObj>();
    if (!call ||
        (!call->op.same_as(tirx::alloc_tensor_op()) && !call->op.same_as(tirx::decl_tensor_op()))) {
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }
    TensorVar tensor = op->var.as_or_throw<TensorVar>();
    TensorVar new_tensor = RewriteAllocTensor(tensor);
    bool is_alloc = call->op.same_as(tirx::alloc_tensor_op());
    if (new_tensor.same_as(tensor) ||
        (is_alloc && call->ty_args[0].as_or_throw<TensorType>()->dtype != new_tensor.dtype())) {
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }

    // Update the producer before generic Bind mutation propagates its result type.
    ffi::Array<Expr> args = call->args;
    auto rewritten = ffi::make_object<BindObj>(*op);
    rewritten->value =
        Call(new_tensor.ty(), call->op, args, call->attrs, {new_tensor.ty()}, call->loc);
    tvm::Bind binding(std::move(rewritten));
    return StmtExprMutator::Mutate_(binding.get(), inplace_mode).ValueOrUnchanged(binding);
  }

  TensorVar RewriteAllocTensor(const TensorVar& tensor) {
    auto it = tensor_info_.find(tensor);
    if (it != tensor_info_.end()) {
      const TensorVar& new_tensor = it->second.new_tensor;
      if (!new_tensor.same_as(tensor)) {
        VarRemapSet(tensor, new_tensor);
      }
      return new_tensor;
    }
    return tensor;
  }

  void RewriteTensorAccess(const TensorVar& original_tensor, TensorVar* tensor,
                           ffi::Array<PrimExpr>* indices) const {
    auto it = tensor_info_.find(original_tensor);
    if (it == tensor_info_.end()) {
      return;
    }
    const TensorAllocInfo& info = it->second;
    TVM_FFI_ICHECK_EQ(indices->size(), info.region.size());
    int ndim = info.region.size();
    ffi::Array<PrimExpr> new_indices;
    new_indices.reserve(ndim);
    for (int i = 0; i < ndim; ++i) {
      new_indices.push_back((*indices)[i] - info.region[i]->begin);
    }
    *tensor = info.new_tensor;
    *indices = std::move(new_indices);
  }

  void RewriteTensorRegion(TensorVar* tensor, ffi::Array<Range>* region) const {
    auto it = tensor_info_.find((*tensor));
    if (it == tensor_info_.end()) {
      // Skip if the tensor is parameter
      return;
    }
    const TensorAllocInfo& info = it->second;
    TVM_FFI_ICHECK_EQ(region->size(), info.region.size());
    ffi::Array<Range> new_region;
    new_region.reserve(info.region.size());
    for (size_t i = 0; i < info.region.size(); ++i) {
      const Range& range = (*region)[i];
      new_region.push_back(
          Range::FromMinExtent(range->begin - info.region[i]->begin, range->extent));
    }
    *tensor = info.new_tensor;
    *region = std::move(new_region);
  }

  void RewriteTensorRegions(ffi::Array<TensorRegion>* regions) const {
    ffi::Array<TensorRegion> new_regions;
    new_regions.reserve(regions->size());
    for (const auto& region : *regions) {
      TensorRegion tensor_region = region;
      TensorRegionObj* p = tensor_region.CopyOnWrite();
      TensorVar source = p->source.as_or_throw<tvm::tirx::TensorVar>();
      RewriteTensorRegion(&source, &p->region);
      p->source = source;
      new_regions.push_back(tensor_region);
    }
    *regions = std::move(new_regions);
  }

  void RewriteMatchTensors(ffi::Array<MatchTensorRegion>* match_tensors) const {
    ffi::Array<MatchTensorRegion> result;
    result.reserve(match_tensors->size());
    for (const auto& match_tensor : *match_tensors) {
      const TensorRegion& tensor_region = match_tensor->source;
      auto p = ffi::make_object<TensorRegionObj>(*tensor_region.get());
      TensorVar source = p->source.as_or_throw<tvm::tirx::TensorVar>();
      RewriteTensorRegion(&source, &p->region);
      p->source = source;
      result.push_back(MatchTensorRegion(match_tensor->tensor, TensorRegion(p)));
    }
    *match_tensors = std::move(result);
  }

  /*! \brief Map tensor var to the allocation information about each tensor. */
  std::unordered_map<Var, TensorAllocInfo> tensor_info_;
};

ffi::Array<PrimExpr> CalcStrides(const TensorAllocInfo& alloc_info,
                                 const ffi::Array<PrimExpr>& shape) {
  std::vector<PrimExpr> strides;
  if (alloc_info.dim_aligns.size()) {
    TVM_FFI_ICHECK(alloc_info.dim_aligns.size() == shape.size());
    strides.reserve(shape.size());
    PrimExpr stride = IntImm(shape[0].ty(), 1);
    for (size_t i = shape.size(); i != 0; --i) {
      size_t dim = i - 1;
      DimAlignInfo info = alloc_info.dim_aligns[dim];
      int align_factor = info.align_factor;
      int align_offset = info.align_offset;
      if (align_factor != 0) {
        PrimExpr factor = IntImm(stride.ty(), align_factor);
        PrimExpr offset = IntImm(stride.ty(), align_offset);
        stride = stride + indexmod(factor + offset - indexmod(stride, factor), factor);
      }
      strides.push_back(stride);
      stride = stride * shape[dim];
    }
  }
  std::reverse(strides.begin(), strides.end());
  return strides;
}

Stmt TensorCompactorCompact(
    const Function& f,
    const std::unordered_map<TensorVar, ffi::Array<Range>, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>&
        regions,
    const std::unordered_map<Var, StorageAlignAnnotation>& storage_align) {
  // collect tensor allocation info for no-alias tensors
  std::unordered_map<Var, TensorAllocInfo> tensor_info;
  for (const auto& kv : regions) {
    const TensorVar& tensor = kv.first;
    // set dim alignment info
    ffi::Array<Range> region = kv.second;
    TensorAllocInfo alloc_info;
    auto it = storage_align.find(tensor);
    if (it != storage_align.end()) {
      std::vector<DimAlignInfo> dim_aligns(tensor.shape().size());
      for (const StorageAlignTuple& dim_align : (*it).second) {
        int dim = dim_align.get<1>();
        int factor = dim_align.get<2>();
        int offset = dim_align.get<3>();
        dim_aligns.at(dim) = {factor, offset};
      }
      alloc_info.dim_aligns = std::move(dim_aligns);
    }

    // prepare new tensor
    ffi::Array<PrimExpr> shape = region.Map([](const Range& range) { return range->extent; });
    ffi::Array<PrimExpr> strides = CalcStrides(alloc_info, shape);
    ffi::ObjectPtr<TensorTypeObj> n = ffi::make_object<TensorTypeObj>(*tensor.ty().get());
    n->shape = std::move(shape);
    if (strides.empty()) {
      if (n->layout.has_value()) {
        auto tile = n->layout.value().as_or_throw<tvm::tirx::TileLayout>();
        n->layout = tvm::tirx::TileLayout(tvm::tirx::TileLayout::DefaultLayout(n->shape)->shard, {},
                                          tile->offset);
      }
    } else {
      static const auto memory_axis = tvm::tirx::Axis::Get("m");
      ffi::Array<tvm::tirx::Iter> shard;
      for (size_t i = 0; i < n->shape.size(); ++i) {
        shard.push_back(tvm::tirx::Iter(n->shape[i], strides[i], memory_axis));
      }
      ffi::Map<tvm::tirx::Axis, PrimExpr> offset;
      if (n->layout.has_value())
        offset = n->layout.value().as_or_throw<tvm::tirx::TileLayout>()->offset;
      n->layout = tvm::tirx::TileLayout(std::move(shard), {}, std::move(offset));
    }
    alloc_info.new_tensor = TensorVar(tensor->name, TensorType(std::move(n)), tensor->loc);
    alloc_info.region = region;
    tensor_info.emplace(tensor, std::move(alloc_info));
  }
  auto compactor = ffi::make_object<TensorCompactor>(std::move(tensor_info));
  Stmt stmt = compactor->Mutate(f->body.value()).ValueOrUnchanged(f->body.value());
  return stmt;
}

namespace transform {

Pass CompactTensorAllocation(bool is_strict) {
  auto pass_func = [=](Function f, IRModule m, PassContext ctx) {
    if (!f->body.has_value()) return f;
    FunctionObj* fptr = f.CopyOnWrite();
    auto region = TensorAccessRegionCollector::Collect(f, /*collect_inbound=*/is_strict);
    auto storage_align = CollectStorageAlignAnnotation(f->body.value());
    fptr->body = TensorCompactorCompact(f, region, storage_align);
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "s_tir.CompactTensorAllocation");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.CompactTensorAllocation", CompactTensorAllocation);
}
}  // namespace transform

}  // namespace s_tir
}  // namespace tvm
