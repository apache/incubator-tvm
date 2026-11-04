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
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/sym/int_set.h>
#include <tvm/tirx/type.h>

#include "../../transform/replace_selected_expr.h"
#include "../utils.h"
#include "cache_index_helpers.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

/******** Helper Functions/Classes ********/

/*! \brief The auxiliary info used for the insertion point and content of the cache stage. */
struct IndexInfo {
  explicit IndexInfo(StmtSRef target) : target_sblock(std::move(target)) {}
  /*! \brief The target block to perform cache_index */
  StmtSRef target_sblock;
  /*! \brief Record the common subexpr extract threshold */
  size_t cse_thresh;
  /*! \brief The cache tensor to store the precomputed index */
  std::vector<TensorVar> cache_tensor;
  /*! \brief The expr to be precomputed */
  std::vector<PrimExpr> index_exprs;
  /*! \brief The range of the loop vars relating to index computation */
  ffi::Map<Var, Range> range_map;
  /*! \brief The binding table of the block var and the loop var */
  ffi::Map<Var, PrimExpr> var_binding;
  /*! \brief The block var of the target block */
  std::vector<ffi::Array<Var>> origin_block_vars;
  /*! \brief The index to insert the cache stage. */
  size_t loc_pos;
  /*! \brief The cache stage to be inserted. */
  Stmt cache_stage{ffi::UnsafeInit{}};
  /*! \brief The map used for ScheduleStateObj::Replace. */
  ffi::Map<SBlock, SBlock> block_reuse;
};

/*!
 * \brief Determine the data type base on the integer range.
 * \param range The range of the integer.
 * \returns A data type that covers the input range.
 */
PrimType DeterminePrimType(const sym::IntSet& range) {
  sym::Analyzer ana;
  if (ana->CanProve(range.min() >= INT32_MIN && range.max() <= INT32_MAX)) {
    return PrimType::Int(32);
  } else {
    TVM_FFI_ICHECK(ana->CanProve(range.min() >= IntImm::Int64(INT64_MIN) &&
                                 range.max() <= IntImm::Int64(INT64_MAX)));
    return PrimType::Int(64);
  }
}

/*! \brief Collect the index info to be cached */
class IndexInfoCollector : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;

  /*!
   * \brief Collect the index info for cache_index and write into the IndexInfo
   * \param self The state of the schedule \param block_sref The sref of the target
   * block of the target tensor being applied cache_index \param scope_sref The sref
   * of the scope block of the target block \param info The index info.
   */
  static void Collect(const ScheduleState& self, const StmtSRef& block_sref,
                      const StmtSRef& scope_sref, IndexInfo* info) {
    auto collector =
        ffi::make_object<IndexInfoCollector>(self, block_sref, scope_sref, info->cse_thresh);
    collector->Visit(ffi::GetRef<Stmt>(scope_sref->stmt));
    info->loc_pos = collector->loc_pos_;
    info->index_exprs = collector->exprs_;
    info->range_map = collector->range_map_;
  }

  /*!
   * \brief Constructor
   * \param self The state of the schedule
   * \param block_sref The sref of the target block of the tensor being applied cache_index
   * \param scope_sref The sref of the scope block of the target block
   * \param cse_thresh The repeat threshold that determines a common subexpr
   */
  IndexInfoCollector(const ScheduleState self, const StmtSRef& block_sref,
                     const StmtSRef& scope_sref, int cse_thresh)
      : self_(self), block_sref_(block_sref), scope_sref_(scope_sref), cse_thresh_(cse_thresh) {}

 private:
  ffi::Optional<VisitInterrupt> Visit_(const SeqStmtObj* seq_stmt) final {
    for (size_t i = 0; i < seq_stmt->size(); ++i) {
      if (loc_pos_ != -1) {
        break;
      }
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(seq_stmt->seq[i]));
      // `pos` can be assigned only once when we visited `block_sref`
      if (visited_block_ && loc_pos_ == -1 && update_seq_pos_) {
        // The offset of insert position from the block
        loc_pos_ = i;
        return std::nullopt;
      }
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const SBlockObj* block) final {
    visiting_target_sblock = static_cast<bool>(block_sref_->stmt == block);
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(block));
    visiting_target_sblock = false;
    if (block == scope_sref_->stmt) {
      // The block vistied is the current parent scope
      // Handling cases when no SeqStmt in the scope
      if (visited_block_ && loc_pos_ == -1) {
        loc_pos_ = 0;
      }
    } else if (block_sref_->stmt == block) {
      visited_block_ = true;
    }
    // Update seq pos only at top scope
    if (visited_block_ && self_->stmt2ref.at(block)->parent == scope_sref_.get()) {
      update_seq_pos_ = true;
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const ForObj* loop) final {
    range_map_.Set(loop->var, Range(loop->begin, loop->end));
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(loop));
    // Update seq pos only at top scope
    if (visited_block_ && self_->stmt2ref.at(loop)->parent == scope_sref_.get()) {
      update_seq_pos_ = true;
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorStoreObj* store) final {
    // Only analyze the cache candidate for stores in target block
    if (visiting_target_sblock) {
      auto IsEligibleComputation = [](const PrimExpr& expr) {
        return (SideEffect(expr) <= CallEffectKind::kPure && CalculateExprComplexity(expr) > 1 &&
                (expr.as<RampObj>() == nullptr) && (expr.as<BroadcastObj>() == nullptr));
      };

      // Analyze sub expr candidates
      ComputationTable table_syntactic_comp_done_by_stmt =
          ComputationsDoneBy::GetComputationsDoneBy(ffi::GetRef<Stmt>(store), IsEligibleComputation,
                                                    [](const PrimExpr& expr) { return true; });
      std::vector<std::pair<PrimExpr, size_t>> semantic_comp_done_by_stmt =
          SyntacticToSemanticComputations(table_syntactic_comp_done_by_stmt, true);

      // Analyze the sub expr of a candidate whose repeat time is under cse_thresh_
      for (size_t i = 0; i < semantic_comp_done_by_stmt.size(); i++) {
        std::pair<PrimExpr, size_t>& computation_and_nb = semantic_comp_done_by_stmt[i];
        if (computation_and_nb.second < cse_thresh_) {
          std::vector<PrimExpr> direct_subexprs = DirectSubexpr::GetDirectSubexpressions(
              computation_and_nb.first, IsEligibleComputation,
              [](const PrimExpr& expr) { return true; });
          InsertVectorToSortedSemanticComputations(&semantic_comp_done_by_stmt, direct_subexprs,
                                                   true, computation_and_nb.second);
        }
      }

      // Record the final sub expr with repeat time greater than cse_thresh_
      // In order to make the result stable, sort it by post order and then by complexity
      auto walk_fn = [&semantic_comp_done_by_stmt,
                      this](const PrimExpr& this_expr) -> ffi::Expected<ffi::WalkResult> {
        for (auto& it : semantic_comp_done_by_stmt) {
          if (it.second >= this->cse_thresh_ && EquivalentTerms(this_expr, it.first, true)) {
            auto find_result =
                std::find_if(this->exprs_.begin(), this->exprs_.end(),
                             [&](PrimExpr expr) { return expr.get() == it.first.get(); });
            if (find_result == this->exprs_.end()) {
              this->exprs_.push_back(it.first);
            }
          }
        }
        return ffi::WalkResult::Advance();
      };
      ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(store->value, walk_fn);
      auto cmp = [&](const PrimExpr& lhs, const PrimExpr& rhs) -> bool {
        return CalculateExprComplexity(lhs) > CalculateExprComplexity(rhs);
      };
      std::stable_sort(exprs_.begin(), exprs_.end(), cmp);
    }
    return StmtExprVisitor::Visit_(store);
  }

  /*! \brief The schedule class */
  const ScheduleState self_;
  /*! \brief The target block that read the target tensor */
  const StmtSRef& block_sref_;
  /*! \brief The parent scope of the target block */
  const StmtSRef& scope_sref_;
  /*! \brief Record the common subexpr extract threshold */
  size_t cse_thresh_;
  /*! \brief The calculation expr to be precomputed */
  std::vector<PrimExpr> exprs_;
  /*! \brief The flag whether we have visited the target block */
  bool visited_block_{false};
  /*! \brief The flag indicating currently visiting target block */
  bool visiting_target_sblock{false};
  /*! \brief The index to insert the cache_index stage */
  int loc_pos_{-1};
  /*! \brief The flag indicating the right scope to update seq pos */
  bool update_seq_pos_{false};
  /*! \brief Record the ranges of iter vars */
  ffi::Map<Var, Range> range_map_;
};

/*!
 * \brief Create a loop nest that writes precomputed index into index tensor.
 * \param info The cache stage information, which will be updated in the function.
 * \param storage_scope The storage scope of the cached tensor (only used in naming here)
 * \returns A block indicating the body of the loop nesting.
 */
ffi::Array<SBlock> MakeIndexCacheStage(IndexInfo* info, const ffi::String& storage_scope) {
  ffi::Array<SBlock> blocks;
  ffi::Array<Stmt> bodies;
  bodies.reserve(info->index_exprs.size());
  info->cache_tensor.reserve(info->index_exprs.size());

  // For each index calculation, create a block to pre-compute.
  for (size_t expr_index = 0; expr_index < info->index_exprs.size(); expr_index++) {
    const PrimExpr& index_expr = info->index_exprs[expr_index];

    // Collect the block vars in original index computation
    info->origin_block_vars.push_back({});
    auto collect_origin_var = [&info,
                               &expr_index](const Var& var) -> ffi::Expected<ffi::WalkResult> {
      if (auto prim_var = var.as<PrimVar>()) {
        Var iter_var = prim_var.value();
        const ffi::Array<Var>& origin_block_var = info->origin_block_vars[expr_index];
        auto find_result = std::find_if(origin_block_var.begin(), origin_block_var.end(),
                                        [&](Var it) { return it.get() == iter_var.get(); });
        if (find_result == origin_block_var.end()) {
          info->origin_block_vars[expr_index].push_back(iter_var);
        }
      }
      return ffi::WalkResult::Advance();
    };
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(index_expr, collect_origin_var);

    // Collect the loop vars corresponding to collected block vars,
    // which will be used to create new loop vars
    std::vector<Var> iter_vars;
    auto collect_iter_var = [&iter_vars](const Var& var) -> ffi::Expected<ffi::WalkResult> {
      if (auto prim_var = var.as<PrimVar>()) {
        Var iter_var = prim_var.value();
        if (std::find_if(iter_vars.begin(), iter_vars.end(),
                         [&](Var it) { return it.get() == iter_var.get(); }) == iter_vars.end()) {
          iter_vars.push_back(iter_var);
        }
      }
      return ffi::WalkResult::Advance();
    };
    for (const Var& it : info->origin_block_vars[expr_index]) {
      ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(info->var_binding.at(it), collect_iter_var);
    }

    PrimType data_ty = index_expr.ty();
    ffi::String index_tensor_name = "index_var_" + std::to_string(expr_index);
    ffi::Array<PrimExpr> tensor_shape;
    for (const Var& it : info->origin_block_vars[expr_index]) {
      tensor_shape.push_back(
          sym::EvalSet(info->var_binding.at(it), sym::AsIntSet(info->range_map)).max() + 1);
    }
    info->cache_tensor.push_back(
        TensorVar(index_tensor_name,
                  TensorType(tensor_shape, data_ty, runtime::MemoryScope::Get(storage_scope), 0,
                             tvm::tirx::TileLayout::DefaultLayout(tensor_shape))));

    // Create loop vars and block vars' binding_value
    std::vector<PrimVar> loop_vars;
    ffi::Map<Var, Var> replace_table;
    for (const Var& it : iter_vars) {
      PrimType data_ty = DeterminePrimType(sym::IntSet::FromRange(info->range_map.at(it)));
      PrimVar loop_var("ax" + std::to_string(replace_table.size()), data_ty);
      loop_vars.push_back(loop_var);
      replace_table.Set(it, loop_var);
    }
    // Create iter_values from the original block.
    std::vector<PrimExpr> iter_values;
    auto f_substitute =
        [&replace_table](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
      if (auto repl = replace_table.Get(var)) return ffi::Any(*std::move(repl));
      return ffi::Unchanged();
    };
    for (const Var& it : info->origin_block_vars[expr_index]) {
      iter_values.push_back(
          ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(info->var_binding.at(it), f_substitute)
              .as_or_throw<PrimExpr>());
    }
    // block variables
    ffi::Array<IterVar> block_vars;
    // block access region for write tensors
    ffi::Array<Range> access_region;
    // indices used in block body
    ffi::Array<PrimExpr> access_indices;
    ffi::Map<Var, Var> block_var_map;
    // Create block vars, block's accessed region and accessing indices
    for (size_t i = 0; i < info->origin_block_vars[expr_index].size(); i++) {
      const Var& block_var = info->origin_block_vars[expr_index][i];
      PrimType block_var_ty = block_var->ty.as_or_throw<PrimType>();
      PrimVar var("v" + std::to_string(access_indices.size()), block_var_ty);
      Range range =
          Range::FromMinExtent(IntImm(block_var_ty, 0), info->range_map.at(iter_vars[i])->extent);
      block_vars.push_back(IterVar(/*dom=*/range,
                                   /*var=*/var,
                                   /*IterVarType=*/kDataPar));

      access_indices.push_back(var);
      access_region.push_back(Range::FromMinExtent(var, IntImm(var.ty(), 1)));
      block_var_map.Set(block_var, var);
    }

    // Create the index computing block
    auto f_substitute_block =
        [&block_var_map](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
      if (auto repl = block_var_map.Get(var)) return ffi::Any(*std::move(repl));
      return ffi::Unchanged();
    };
    PrimExpr new_expr =
        ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(index_expr, f_substitute_block)
            .as_or_throw<PrimExpr>();
    SBlock block(
        /*iter_vars=*/std::move(block_vars),
        /*reads=*/{},
        /*writes=*/
        {TensorRegion(info->cache_tensor[expr_index], access_region, TensorRegionType())},
        /*name_hint=*/"index_" + std::to_string(expr_index),
        /*body=*/
        TensorStore(info->cache_tensor[expr_index], access_indices, new_expr),
        /*init=*/std::nullopt,
        /*alloc_tensors=*/{},
        /*match_tensors=*/{},
        /*annotations=*/{});
    blocks.push_back(block);
    // Create the block realize node
    Stmt body = SBlockRealize(/*values=*/iter_values,
                              /*predicate=*/IntImm::Bool(true),
                              /*block=*/block);
    // Create surrounding loops
    for (size_t i = loop_vars.size(); i >= 1; --i) {
      body = For(loop_vars[i - 1], 0, info->range_map.at(iter_vars[i - 1])->extent,
                 IntImm(loop_vars[i - 1].ty(), 1), ForKind::kDefault, body);
    }
    bodies.push_back(body);
  }

  info->cache_stage = SeqStmt(bodies);
  return blocks;
}

/*!
 * \brief Insert the cache stages into the specific position
 * \param stmt A sequence of statements or a single statement that the new stage is inserted in
 * \param pos The position where the cache stage is inserted
 * \param stage The stage to be inserted
 * \return A SeqStmt, the result after insertion
 */
Stmt InsertIndexStage(const Stmt& stmt, int pos, const Stmt& stage) {
  if (const auto* seq_stmt = stmt.as<SeqStmtObj>()) {
    ffi::Array<Stmt> result = seq_stmt->seq;
    result.insert(result.begin() + pos, stage);
    return SeqStmt(result, seq_stmt->loc);
  }
  if (pos == 0) {
    return SeqStmt({stage, stmt});
  }
  TVM_FFI_ICHECK_EQ(pos, 1);
  return SeqStmt({stmt, stage});
}

/*! \brief Mutator for CacheIndex. */
class CacheIndexRewriter : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  /*!
   * \brief Rewrite the AST and add stages of writting precomputed index
   * \param scope_sref The parent scope of this mutation
   * \param info The index information
   * \return The new AST rooting at the original parent scope
   */
  static Stmt Rewrite(const StmtSRef& scope_sref, IndexInfo* info) {
    auto rewriter = ffi::make_object<CacheIndexRewriter>(scope_sref, info);
    return rewriter->Mutate(ffi::GetRef<Stmt>(scope_sref->stmt))
        .ValueOrUnchanged(ffi::GetRef<Stmt>(scope_sref->stmt));
  }

  explicit CacheIndexRewriter(const StmtSRef& scope_sref, IndexInfo* info)
      : scope_sref_(scope_sref), info_(info) {
    cache_indices_.reserve(info_->origin_block_vars.size());
    for (const ffi::Array<Var>& group_it : info_->origin_block_vars) {
      cache_indices_.push_back({});
      for (const Var& it : group_it) {
        cache_indices_.back().push_back(it.as_or_throw<PrimExpr>());
      }
    }
  }

 private:
  UnchangedOr<Stmt> Mutate_(const SBlockObj* block, InplaceMode inplace_mode) final {
    SBlock old_stmt = ffi::GetRef<SBlock>(block);
    // Mutate the body
    visiting_target_sblock = static_cast<bool>(block == info_->target_sblock->stmt);
    SBlock stmt = StmtExprMutator::Mutate_(block, inplace_mode)
                      .ValueOrUnchanged(ffi::GetRef<Stmt>(block))
                      .as_or_throw<SBlock>();
    visiting_target_sblock = false;

    // Check if it is the block corresponding to the parent scope
    if (block == scope_sref_->stmt) {
      // If so, put tensor allocation and insert cache stages on the parent scope
      ffi::ObjectPtr<SBlockObj> n = ffi::make_object<SBlockObj>(*stmt.as<SBlockObj>());
      n->body = InsertIndexStage(n->body, info_->loc_pos, info_->cache_stage);
      for (const TensorVar& it : info_->cache_tensor) {
        n->alloc_tensors.push_back(it);
      }
      stmt = SBlock(n);
    }
    info_->block_reuse.Set(old_stmt, stmt);
    return stmt;
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreObj* store, InplaceMode inplace_mode) final {
    Stmt ret_stmt =
        StmtExprMutator::Mutate_(store, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(store));
    // Replace common sub expr for target block, with cached tensor load
    if (visiting_target_sblock) {
      for (size_t i = 0; i < info_->index_exprs.size(); i++) {
        PrimExpr& computation = info_->index_exprs[i];
        std::function<bool(const PrimExpr&)> predicate_selector =
            [computation](const PrimExpr& current_expr) {
              return (EquivalentTerms(current_expr, computation, true));
            };
        TensorLoad load = TensorLoad(info_->cache_tensor[i], cache_indices_[i]);
        ret_stmt = ReplaceSelectedExpr::ReplaceSelectedExprInStmt(
            ret_stmt, predicate_selector, std::move(load),
            [](const PrimExpr& expr) { return true; });
      }
    }
    return ret_stmt;
  }

  /*! \brief The parent scope of the insertion */
  const StmtSRef& scope_sref_;
  /*! \brief The info for inserting cache stage */
  IndexInfo* info_;
  /*! \brief The indices for the cache tensor */
  std::vector<ffi::Array<PrimExpr>> cache_indices_;
  /*! \brief Indicating whether cache stage is inserted, only do index replacement afterwards*/
  bool visiting_target_sblock{false};
};

ffi::Array<StmtSRef> CacheIndex(ScheduleState self, const StmtSRef& block_sref,
                                const ffi::String& storage_scope, int cse_thresh) {
  /*!
   * Check:
   *   - The index is in the array of block reading region
   *
   * Mutate:
   *   - Allocate new cache tensors under the current scope.
   *   - Precompute the index and store it in cache tensors.
   */

  // Step 0. Checking index, getting the target tensor and the parent scope
  IndexInfo info(block_sref);
  TVM_FFI_ICHECK_GE(cse_thresh, 0) << "cse_thresh should not be negative number";
  info.cse_thresh = cse_thresh;
  StmtSRef scope_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/false);

  // Step 1. Collect the indexing info of target tensor.
  IndexInfoCollector::Collect(self, block_sref, scope_sref, &info);

  // Step 2. Create cache stages and rewrite the stmt.
  SBlockRealize realize = GetSBlockRealize(self, block_sref);
  info.var_binding = GetBindings(realize);
  ffi::Array<SBlock> cache_stages = MakeIndexCacheStage(&info, storage_scope);
  Stmt new_scope = CacheIndexRewriter::Rewrite(/*scope_sref=*/scope_sref, /*info=*/&info);

  bool old_stage_pipeline = self->block_info[block_sref].stage_pipeline;

  // Step 3. Replacing and updating flags.
  self->Replace(scope_sref, new_scope, info.block_reuse);
  ffi::Array<StmtSRef> result_block_srefs;
  for (const SBlock& it : cache_stages) {
    StmtSRef result_block_sref = self->stmt2ref.at(it.get());
    result_block_srefs.push_back(result_block_sref);
    SBlockInfo& block_info = self->block_info[result_block_sref];

    bool affine_binding = false;
    if (result_block_sref->parent == nullptr) {
      affine_binding = true;
    } else {
      sym::Analyzer analyzer;
      StmtSRef parent_sref = ffi::GetRef<StmtSRef>(result_block_sref->parent);
      affine_binding = IsAffineBinding(/*realize=*/GetSBlockRealize(self, result_block_sref),
                                       /*loop_var_ranges=*/LoopDomainOfSRefTreePath(parent_sref),
                                       /*analyzer=*/analyzer.get());
    }

    block_info.affine_binding = affine_binding;
    block_info.region_cover = true;
    block_info.stage_pipeline = old_stage_pipeline;
  }

  return result_block_srefs;
}

/******** InstructionKind Registration ********/

struct CacheIndexTraits : public UnpackedInstTraits<CacheIndexTraits> {
  static constexpr const char* kName = "CacheIndex";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static ffi::Array<SBlockRV> UnpackedApplyToSchedule(Schedule sch, SBlockRV block,
                                                      ffi::String storage_scope, int cse_thresh) {
    return sch->CacheIndex(block, storage_scope, cse_thresh);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block,
                                      ffi::String storage_scope, int cse_thresh) {
    PythonAPICall py("cache_index");
    py.Input("block", block);
    py.Input("storage_scope", storage_scope);
    py.Input("cse_thresh", cse_thresh);
    py.OutputList(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

TVM_FFI_STATIC_INIT_BLOCK() { RegisterInstructionKind<CacheIndexTraits>(); }

}  // namespace s_tir
}  // namespace tvm
