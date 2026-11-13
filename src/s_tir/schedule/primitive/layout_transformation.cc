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
#include <tvm/ir/analysis.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor_with_constraint_context.h>
#include <tvm/sym/analyzer.h>

#include <optional>
#include <variant>

#include "../../tensor_region.h"
#include "../utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::prim;

using namespace tvm::tirx;

/*! \brief Planning stage prior to rewriting in TransformLayoutRewriter
 *
 * There are four ways that transformation may be handled.  Each
 * updates the tensor shape and the indices used to acces the tensor
 * in TensorStore/TensorLoad nodes, but differ in how they handle the
 * `pad_value`.  In order of preference, the different strategies are
 * as follows:
 *
 * 1. NoPaddingRequired.  The transformation does not introduce
 * padding, so only local changes to update the indices of
 * TensorLoad/TensorStore nodes are required.  No blocks are added,
 * removed, or replaced.
 *
 * 2. ProloguePlan.  The transformation introduces padding, but the
 * analyzed block has no write stages for the transformed tensor.
 * This tensor is an input and the caller is responsible for ensuring
 * that the padding contains the specified `pad_value`.  The generated
 * prologue contains `prim::assume_op()` calls that will expose this
 * known value during scheduling/simplification, but will be removed
 * during lowering.
 *
 * 3. ReplacementPlan.  The transformation introduces padding, has at
 * least one write stage for the transformed tensor, and at least one
 * of those write stages writes to all pre-transformation indices
 * following a row-major traversal.  These write stage is rewritten to
 * be row-major traversals of the post-transformation indices, with a
 * `tvm::if_then_else` call to write either the specified `pad_value`
 * into padding or the computed value into non-padding.
 *
 * 4. EpiloguePlan.  The transformation introduces padding, has at
 * least one write stage for the transformed tensor, but no write
 * stage can be rewritten to use `tvm::if_then_else`.  The
 * transformation still requires the `pad_value` to be written into
 * the padding, so a new block is inserted after the last write stage
 * to explicitly fill the padding.
 *
 */
class TransformLayoutPlanner : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;

  // Statement to be inserted prior to the analyzed block
  struct ProloguePlan {
    Stmt prologue;
  };

  // Loops within the analyzed block that should be replaced
  struct ReplacementPlan {
    ffi::Map<For, Stmt> replacements;
    ffi::Map<SBlock, SBlock> new_block_to_old;
  };

  // The block to be inserted, along with the location at which it
  // should be inserted.  The location will be either a For or a
  // SBlock, and will be after all writes the transformed tensor.
  struct EpiloguePlan {
    Stmt insert_after;
    Stmt new_block;
  };

  struct NoPaddingRequired {};

  using TransformPlan =
      std::variant<ProloguePlan, ReplacementPlan, EpiloguePlan, NoPaddingRequired>;

  static TransformPlan Plan(SBlock block, TensorVar old_tensor, TensorVar new_tensor,
                            IndexMap index_map, IndexMap inverse, PrimExpr padding_predicate,
                            ffi::Optional<IndexMap> pad_value, sym::AnalyzerObj* analyzer) {
    TVM_FFI_ICHECK(!pad_value.has_value() || pad_value.value()->final_indices.size() == 1)
        << "Internal error: Should be caught by ScheduleError checks prior to this point";
    auto visitor = ffi::make_object<TransformLayoutPlanner>(old_tensor);
    visitor->Visit(block);
    return visitor->Finalize(new_tensor, index_map, inverse, padding_predicate, pad_value,
                             analyzer);
  }

 private:
  struct WriteInfo {
    // The TensorStore object
    TensorStore store;

    // The block realize that contains the store, if any.
    ffi::Optional<SBlockRealize> innermost_block_realize;

    // The nested loops whose values contribute to the indices used in
    // the store.  Not all loop variables in the loopnest need to
    // contribute, but the first and last must.
    std::vector<For> dependent_loopnest;

    // Whether the padding could be represented as a tvm::if_then_else
    // node.  This requires that the surrounding loop iterators
    // iterate over all pre-transformation tensor axes, that there are
    // no data dependencies between loop iterations, and that
    bool contains_row_major_traversal{false};
  };

 public:
  explicit TransformLayoutPlanner(TensorVar old_tensor) : old_tensor_(old_tensor) {}

 private:
  ffi::Optional<VisitInterrupt> Visit_(const ForObj* op) override {
    BindLoopVar context(this, ffi::GetRef<For>(op));
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const BindObj* op) override {
    BindVariableDefinition context(this, op->var, op->value);
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const SBlockRealizeObj* op) override {
    BindBlockRealize context(this, ffi::GetRef<SBlockRealize>(op));
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorStoreObj* op) override {
    if (!op->dest.as_or_throw<TensorVar>().same_as(old_tensor_)) {
      return std::nullopt;
    }

    std::optional<std::pair<size_t, size_t>> loop_dependency_range = std::nullopt;
    for (const auto& index : op->indices) {
      if (auto index_depth = LoopDependencyRange(index); index_depth.has_value()) {
        if (loop_dependency_range) {
          loop_dependency_range = {
              std::min(loop_dependency_range.value().first, index_depth.value().first),
              std::max(loop_dependency_range.value().second, index_depth.value().second)};
        } else {
          loop_dependency_range = index_depth;
        }
      }
    }

    WriteInfo write_info{ffi::GetRef<TensorStore>(op)};
    if (loop_dependency_range) {
      size_t i = loop_dependency_range.value().first;
      size_t j = loop_dependency_range.value().second;
      TVM_FFI_ICHECK_LT(i, active_loops_.size());
      TVM_FFI_ICHECK_LT(j, active_loops_.size());

      write_info.dependent_loopnest = {active_loops_.begin() + i, active_loops_.begin() + j + 1};
    }
    write_info.innermost_block_realize = innermost_block_realize_;

    write_info.contains_row_major_traversal = [&]() -> bool {
      const auto& loopnest = write_info.dependent_loopnest;
      if (loopnest.empty()) {
        return false;
      }

      if (loopnest.size() != old_tensor_.shape().size() || loopnest.size() != op->indices.size()) {
        return false;
      }

      auto f_substitute = [this](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
        if (auto it = active_var_bindings_.find(var.get()); it != active_var_bindings_.end()) {
          return ffi::Any(it->second);
        }
        return ffi::Unchanged();
      };
      for (size_t i = 0; i < loopnest.size(); i++) {
        const For& loop = loopnest[i];
        const PrimExpr& tensor_dim = old_tensor_.shape()[i];
        PrimExpr index = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(op->indices[i], f_substitute)
                             .as_or_throw<PrimExpr>();
        bool is_loop_over_axis =
            index.same_as(loop->var) && IsConstInt(loop->begin, 0) &&
            prim::ExprDeepEqual()(Range::GetExtentFromBeginEnd(loop->begin, loop->end),
                                  tensor_dim) &&
            loop->kind == ForKind::kDefault;
        if (!is_loop_over_axis) {
          return false;
        }
      }

      return true;
    }();

    write_info_.push_back(write_info);

    // Don't need to continue recursing, as the entire goal was to
    // find the TensorStore.
    return std::nullopt;
  }

  std::optional<std::pair<size_t, size_t>> LoopDependencyRange(const PrimExpr& expr) const {
    std::optional<std::pair<size_t, size_t>> prev = std::nullopt;
    for (const auto& var : UndefinedVars(expr)) {
      auto it = loop_depth_lookup_.find(var.get());
      if (it != loop_depth_lookup_.end()) {
        if (prev.has_value()) {
          prev = {std::min(prev.value().first, it->second.first),
                  std::max(prev.value().second, it->second.second)};
        } else {
          prev = it->second;
        }
      }
    }

    return prev;
  }

  class TensorStoreReplacer : public StmtExprMutator {
   public:
    using StmtExprMutator::Mutate;
    using StmtExprMutator::Mutate_;

    TensorStoreReplacer(const WriteInfo& info, const TensorVar& new_tensor,
                        PrimExpr padding_predicate, const IndexMap& inverse,
                        const ffi::Optional<IndexMap>& pad_value,
                        ffi::Map<SBlock, SBlock>* new_block_to_old, sym::AnalyzerObj* analyzer)
        : info(info),
          new_tensor(new_tensor),
          new_indices(
              inverse->initial_indices.Map([](PrimVar var) { return static_cast<Var>(var); })),
          padding_predicate(padding_predicate),
          inverse(inverse),
          pad_value(pad_value),
          new_block_to_old(*new_block_to_old),
          analyzer(analyzer) {
      TVM_FFI_ICHECK_EQ(info.dependent_loopnest.size(), inverse->final_indices.size());
      for (size_t i = 0; i < info.dependent_loopnest.size(); i++) {
        Var var = info.dependent_loopnest[i]->var;
        PrimExpr expr = inverse->final_indices[i];
        VarRemapSet(var, expr);
      }

      DefineBlockUpdates();
    }

    bool is_all_stores_replaced() const { return all_stores_replaced; }

   private:
    void DefineBlockUpdates() {
      if (!info.innermost_block_realize) {
        return;
      }

      SBlockRealize block_realize = info.innermost_block_realize.value();
      const auto& block = block_realize->block;
      const ffi::Array<PrimExpr>& old_indices = info.store->indices;
      const auto& old_iter_vars = block->iter_vars;

      this->new_iter_vars = old_iter_vars;
      this->new_iter_values = block_realize->iter_values;

      if (old_indices.empty()) {
        return;
      }

      // Find the block iterators that are used to access the tensor.  Must be in the same
      // order as they appear in the indices.
      if (block->iter_vars.size() < old_indices.size()) {
        return;
      }

      size_t block_index_start = 0;
      for (; block_index_start < old_iter_vars.size() - old_indices.size(); block_index_start++) {
        if (old_indices[0].same_as(old_iter_vars[block_index_start]->var)) {
          break;
        }
      }
      if (block_index_start > old_iter_vars.size() - old_indices.size()) {
        return;
      }

      for (size_t i = 0; i < old_indices.size(); i++) {
        if (!old_indices[i].same_as(old_iter_vars[block_index_start + i]->var) ||
            old_iter_vars[block_index_start + i]->iter_type != kDataPar) {
          return;
        }
      }

      // If we got to this point, all indices used to access the
      // tensor are virtual indices defined in the innermost block.
      // Therefore, generate new virtual indices for iterating over
      // the post-transform tensor.

      new_indices = inverse->initial_indices.Map([](PrimVar var) {
        std::stringstream ss;
        ss << "v_" << var->name;
        return Var(ss.str(), var.ty());
      });

      ffi::Map<Var, Var>
          loop_var_to_virtual_var;  // For updating padding_predicate in terms of the new indices
      ffi::Array<PrimExpr> new_iter_values;  // For BlockRealize
      ffi::Array<IterVar> new_iter_vars;     // For Block

      for (size_t i = 0; i < block_index_start; i++) {
        new_iter_vars.push_back(old_iter_vars[i]);
        new_iter_values.push_back(block_realize->iter_values[i]);
      }

      TVM_FFI_ICHECK_EQ(new_indices.size(), new_tensor.shape().size());
      for (size_t i = 0; i < new_indices.size(); i++) {
        Var var = inverse->initial_indices[i];
        Var virtual_var = new_indices[i];
        PrimExpr dim = new_tensor.shape()[i];
        new_iter_values.push_back(var.as_or_throw<PrimExpr>());
        new_iter_vars.push_back(IterVar(Range::FromMinExtent(IntImm(dim.ty(), 0), dim),
                                        virtual_var.as_or_throw<PrimVar>(), kDataPar));
        loop_var_to_virtual_var.Set(var, virtual_var);
      }

      for (size_t i = block_index_start + old_indices.size(); i < old_iter_vars.size(); i++) {
        new_iter_vars.push_back(old_iter_vars[i]);
        new_iter_values.push_back(block_realize->iter_values[i]);
      }

      TVM_FFI_ICHECK_EQ(inverse->final_indices.size(), old_indices.size());
      auto f_substitute =
          [&loop_var_to_virtual_var](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
        if (auto repl = loop_var_to_virtual_var.Get(var)) return ffi::Any(*std::move(repl));
        return ffi::Unchanged();
      };
      for (size_t i = 0; i < old_indices.size(); i++) {
        Var var = old_indices[i].as_or_throw<Var>();
        PrimExpr expr =
            ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(inverse->final_indices[i], f_substitute)
                .as_or_throw<PrimExpr>();
        VarRemapSet(var, expr);
      }

      padding_predicate =
          ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(padding_predicate, f_substitute)
              .as_or_throw<PrimExpr>();

      this->new_iter_vars = new_iter_vars;
      this->new_iter_values = new_iter_values;
    }

    UnchangedOr<Stmt> Mutate_(const TensorStoreObj* op, InplaceMode inplace_mode) final {
      bool can_replace = [&]() -> bool {
        if (!op->dest.as_or_throw<TensorVar>().same_as(info.store->dest)) {
          return false;
        }

        const ffi::Array<PrimExpr>& old_indices = info.store->indices;

        TVM_FFI_ICHECK_EQ(old_indices.size(), op->indices.size());
        prim::ExprDeepEqual expr_equal;
        for (size_t i = 0; i < old_indices.size(); i++) {
          if (!expr_equal(old_indices[i], op->indices[i])) {
            return false;
          }
        }
        return true;
      }();

      TensorStore store = ffi::GetRef<TensorStore>(op);
      if (can_replace) {
        ffi::Array<PrimExpr> new_index_exprs =
            new_indices.Map([](const Var& var) { return var.as_or_throw<PrimExpr>(); });
        PrimExpr pad_value_at_index =
            pad_value.value()->MapIndices(new_index_exprs, ffi::GetRef<sym::Analyzer>(analyzer))[0];
        store = TensorStore(new_tensor, new_index_exprs,
                            if_then_else(padding_predicate, pad_value_at_index, op->value));
      } else {
        all_stores_replaced = false;
      }
      return StmtExprMutator::Mutate_(store.get(),
                                      store.unique() ? inplace_mode : InplaceMode::kDisallow)
          .ValueOrUnchanged(store);
    }

    UnchangedOr<Stmt> Mutate_(const SBlockRealizeObj* op, InplaceMode inplace_mode) final {
      SBlockRealize realize = StmtExprMutator::Mutate_(op, inplace_mode)
                                  .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                                  .as_or_throw<SBlockRealize>();

      if (info.innermost_block_realize.has_value() &&
          op == info.innermost_block_realize.value().get()) {
        SBlock block = realize->block;
        if (!block->iter_vars.same_as(this->new_iter_vars)) {
          block.CopyOnWrite()->iter_vars = this->new_iter_vars;
          RecordReplacement(op->block, block);
        }

        if (!block.same_as(realize->block) ||
            !realize->iter_values.same_as(this->new_iter_values)) {
          auto write_ptr = realize.CopyOnWrite();
          write_ptr->block = block;
          write_ptr->iter_values = this->new_iter_values;
        }
      }

      return realize;
    }

    UnchangedOr<Stmt> Mutate_(const SBlockObj* op, InplaceMode inplace_mode) final {
      SBlock orig = ffi::GetRef<SBlock>(op);
      SBlock input = orig;
      if (info.innermost_block_realize.has_value() &&
          op == info.innermost_block_realize.value()->block.get()) {
        input.CopyOnWrite()->iter_vars = new_iter_vars;
      }
      SBlock mutated = StmtExprMutator::Mutate_(input.get(), InplaceMode::kDisallow)
                           .ValueOrUnchanged(input)
                           .as_or_throw<SBlock>();

      RecordReplacement(orig, mutated);
      return mutated;
    }

    void RecordReplacement(SBlock before, SBlock after) {
      if (before.same_as(after)) {
        return;
      }

      TVM_FFI_ICHECK(!new_block_to_old.count(after));

      while (true) {
        if (auto opt = new_block_to_old.Get(before)) {
          before = opt.value();
        } else {
          break;
        }
      }

      new_block_to_old.Set(after, before);
    }

    const WriteInfo& info;
    const TensorVar& new_tensor;
    ffi::Array<Var> new_indices;
    ffi::Array<IterVar> new_iter_vars;
    ffi::Array<PrimExpr> new_iter_values;
    PrimExpr padding_predicate;
    const IndexMap& inverse;
    const ffi::Optional<IndexMap>& pad_value;
    ffi::Map<SBlock, SBlock>& new_block_to_old;
    bool all_stores_replaced{true};
    sym::AnalyzerObj* analyzer;
  };

  TransformPlan Finalize(TensorVar new_tensor, IndexMap index_map, IndexMap inverse,
                         PrimExpr padding_predicate, ffi::Optional<IndexMap> pad_value,
                         sym::AnalyzerObj* analyzer) const {
    if (auto prologue_plan = FinalizeProloguePlan(new_tensor, index_map, inverse, padding_predicate,
                                                  pad_value, analyzer);
        prologue_plan.has_value()) {
      return prologue_plan.value();
    } else if (auto replacement_plan = FinalizeReplacementPlan(
                   new_tensor, index_map, inverse, padding_predicate, pad_value, analyzer);
               replacement_plan.has_value()) {
      return replacement_plan.value();
    } else if (auto epilogue_plan = FinalizeEpiloguePlan(new_tensor, index_map, inverse,
                                                         padding_predicate, pad_value, analyzer);
               epilogue_plan.has_value()) {
      return epilogue_plan.value();
    } else {
      return NoPaddingRequired();
    }
  }

  std::optional<ProloguePlan> FinalizeProloguePlan(TensorVar new_tensor, IndexMap index_map,
                                                   IndexMap inverse, PrimExpr padding_predicate,
                                                   ffi::Optional<IndexMap> pad_value,
                                                   sym::AnalyzerObj* analyzer) const {
    if (write_info_.size() || IsZero(padding_predicate) || !pad_value.has_value()) {
      return std::nullopt;
    }

    ffi::Array<IterVar> iter_vars;
    ffi::Array<PrimExpr> iter_values;
    ffi::Array<PrimExpr> indices;
    ffi::Map<Var, Var> loop_indices_to_block_indices;
    TVM_FFI_ICHECK_EQ(inverse->initial_indices.size(), new_tensor.shape().size());
    for (size_t i = 0; i < inverse->initial_indices.size(); i++) {
      const auto& loop_var = inverse->initial_indices[i];
      const auto& dim = new_tensor.shape()[i];
      Var block_var("v_" + loop_var->name, loop_var.ty());
      IterVar iter_var(Range(0, dim), block_var.as_or_throw<PrimVar>(), kDataPar);
      loop_indices_to_block_indices.Set(loop_var, block_var);
      indices.push_back(iter_var->var);
      iter_vars.push_back(iter_var);
      iter_values.push_back(loop_var);
    }
    auto f_substitute = [&loop_indices_to_block_indices](
                            const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
      if (auto repl = loop_indices_to_block_indices.Get(var)) return ffi::Any(*std::move(repl));
      return ffi::Unchanged();
    };
    padding_predicate =
        ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(std::move(padding_predicate), f_substitute)
            .as_or_throw<PrimExpr>();

    PrimExpr pad_value_at_index =
        pad_value.value()->MapIndices(indices, ffi::GetRef<sym::Analyzer>(analyzer))[0];
    PrimExpr expr = (!padding_predicate) || (TensorLoad(new_tensor, indices) == pad_value_at_index);
    Stmt stmt = Evaluate(Call(PrimType::Bool(), prim::assume_op(), {expr}).as_or_throw<PrimExpr>());

    std::stringstream block_name;
    block_name << "tensor_" << new_tensor->name << "_assumptions";
    auto read_region = s_tir::TensorRegionFromPoint(new_tensor, indices);
    stmt = SBlockRealize(iter_values, IntImm::Bool(true),
                         SBlock(iter_vars, {read_region}, {}, block_name.str(), stmt));

    for (size_t rev_i = 0; rev_i < inverse->initial_indices.size(); rev_i++) {
      size_t i = (inverse->initial_indices.size() - 1) - rev_i;
      Var loop_var = inverse->initial_indices[i];
      PrimExpr extent = new_tensor.shape()[i];
      stmt = For(loop_var.as_or_throw<PrimVar>(), 0, extent,
                 IntImm(loop_var.as_or_throw<PrimVar>().ty(), 1), ForKind::kDefault, stmt);
    }
    return ProloguePlan{stmt};
  }

  std::optional<ReplacementPlan> FinalizeReplacementPlan(TensorVar new_tensor, IndexMap index_map,
                                                         IndexMap inverse,
                                                         PrimExpr padding_predicate,
                                                         ffi::Optional<IndexMap> pad_value,
                                                         sym::AnalyzerObj* analyzer) const {
    if (write_info_.empty() || IsZero(padding_predicate) || !pad_value.has_value()) {
      return std::nullopt;
    }

    ffi::Map<SBlock, SBlock> new_block_to_old;
    auto generate_if_then_else_block = [&](const WriteInfo& info) -> ffi::Optional<Stmt> {
      if (!info.contains_row_major_traversal || !pad_value.has_value() ||
          IsZero(padding_predicate)) {
        return std::nullopt;
      }

      auto replacer = ffi::make_object<TensorStoreReplacer>(
          info, new_tensor, padding_predicate, inverse, pad_value, &new_block_to_old, analyzer);
      Stmt stmt = replacer->Mutate(info.dependent_loopnest.back()->body)
                      .ValueOrUnchanged(info.dependent_loopnest.back()->body);
      if (!replacer->is_all_stores_replaced()) {
        return std::nullopt;
      }

      TVM_FFI_ICHECK_EQ(inverse->initial_indices.size(), new_tensor.shape().size());
      for (size_t rev_i = 0; rev_i < inverse->initial_indices.size(); rev_i++) {
        size_t i = (inverse->initial_indices.size() - 1) - rev_i;
        Var loop_var = inverse->initial_indices[i];
        PrimExpr extent = new_tensor.shape()[i];
        stmt = For(loop_var.as_or_throw<PrimVar>(), 0, extent,
                   IntImm(loop_var.as_or_throw<PrimVar>().ty(), 1), ForKind::kDefault, stmt);
      }

      return stmt;
    };

    ffi::Map<For, Stmt> loop_replacements;

    for (const auto& info : write_info_) {
      if (info.dependent_loopnest.size()) {
        if (auto opt_stmt = generate_if_then_else_block(info)) {
          loop_replacements.Set(info.dependent_loopnest[0], opt_stmt.value());
        }
      }
    }

    if (loop_replacements.size()) {
      return ReplacementPlan{std::move(loop_replacements), std::move(new_block_to_old)};
    } else {
      return std::nullopt;
    }
  }

  std::optional<EpiloguePlan> FinalizeEpiloguePlan(TensorVar new_tensor, IndexMap index_map,
                                                   IndexMap inverse, PrimExpr padding_predicate,
                                                   ffi::Optional<IndexMap> pad_value,
                                                   sym::AnalyzerObj* analyzer) const {
    if (write_info_.empty() || IsZero(padding_predicate) || !pad_value.has_value()) {
      return std::nullopt;
    }

    ffi::Array<IterVar> iter_vars;
    ffi::Array<PrimExpr> iter_values;
    ffi::Array<PrimExpr> indices;
    TVM_FFI_ICHECK_EQ(inverse->initial_indices.size(), new_tensor.shape().size());
    for (size_t i = 0; i < inverse->initial_indices.size(); i++) {
      const auto& loop_var = inverse->initial_indices[i];
      const auto& dim = new_tensor.shape()[i];
      Var block_var("v_" + loop_var->name, loop_var.ty());
      IterVar iter_var(Range(0, dim), block_var.as_or_throw<PrimVar>(), kDataPar);
      indices.push_back(iter_var->var);
      iter_vars.push_back(iter_var);
      iter_values.push_back(loop_var);
    }

    PrimExpr pad_value_at_index =
        pad_value.value()->MapIndices(indices, ffi::GetRef<sym::Analyzer>(analyzer))[0];
    Stmt stmt = TensorStore(new_tensor, indices, pad_value_at_index);

    std::stringstream block_name;
    block_name << "tensor_" << new_tensor->name << "_padding";
    auto write_region = s_tir::TensorRegionFromPoint(new_tensor, indices);
    stmt = SBlockRealize(iter_values, padding_predicate,
                         SBlock(iter_vars, {}, {write_region}, block_name.str(), stmt));

    TVM_FFI_ICHECK_EQ(inverse->initial_indices.size(), new_tensor.shape().size());
    for (size_t rev_i = 0; rev_i < inverse->initial_indices.size(); rev_i++) {
      size_t i = (inverse->initial_indices.size() - 1) - rev_i;
      Var loop_var = inverse->initial_indices[i];
      PrimExpr extent = new_tensor.shape()[i];
      stmt = For(loop_var.as_or_throw<PrimVar>(), 0, extent,
                 IntImm(loop_var.as_or_throw<PrimVar>().ty(), 1), ForKind::kDefault, stmt);
    }

    const auto& info = write_info_.back();
    Stmt insert_after = [&]() -> Stmt {
      if (info.dependent_loopnest.size()) {
        return info.dependent_loopnest.front();
      } else if (info.innermost_block_realize) {
        return info.innermost_block_realize.value();
      } else {
        TVM_FFI_THROW(InternalError) << "Write occured outside of any block/loop";
      }
    }();
    return EpiloguePlan{insert_after, stmt};
  }

  struct BindLoopVar {
    BindLoopVar(TransformLayoutPlanner* self, For for_node) : self_(self), var_(for_node->var) {
      size_t loop_depth = self_->active_loops_.size();
      self_->loop_depth_lookup_[var_.get()] = {loop_depth, loop_depth};
      self_->active_loops_.push_back(std::move(for_node));
    }
    ~BindLoopVar() {
      self_->active_loops_.pop_back();
      self_->loop_depth_lookup_.erase(var_.get());
    }
    BindLoopVar(const BindLoopVar&) = delete;
    BindLoopVar& operator=(const BindLoopVar&) = delete;
    BindLoopVar(BindLoopVar&&) = delete;
    BindLoopVar& operator=(BindLoopVar&&) = delete;

    TransformLayoutPlanner* self_{nullptr};
    Var var_;
  };

  // Under SSA, each variable is bound exactly once, so the lookup maps
  // grow monotonically and cleanup is unnecessary.  Omitting cleanup also
  // ensures correctness for flat BindObj (which has no body): the
  // binding must remain visible to subsequent sibling statements.
  struct BindVariableDefinition {
    BindVariableDefinition() {}
    BindVariableDefinition(TransformLayoutPlanner* self, Var var, Expr value) {
      auto prim_value = value.as<PrimExpr>();
      if (!prim_value) return;
      if (auto loop_depth = self->LoopDependencyRange(prim_value.value()); loop_depth.has_value()) {
        self->loop_depth_lookup_[var.get()] = loop_depth.value();
        auto f_substitute = [self](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
          if (auto it = self->active_var_bindings_.find(var.get());
              it != self->active_var_bindings_.end()) {
            return ffi::Any(it->second);
          }
          return ffi::Unchanged();
        };
        self->active_var_bindings_.insert_or_assign(
            var.get(),
            ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(prim_value.value(), f_substitute)
                .as_or_throw<PrimExpr>());
      }
    }
    ~BindVariableDefinition() {}
    BindVariableDefinition(const BindVariableDefinition&) = delete;
    BindVariableDefinition& operator=(const BindVariableDefinition&) = delete;
    BindVariableDefinition(BindVariableDefinition&&) = default;
    BindVariableDefinition& operator=(BindVariableDefinition&&) = default;
  };

  struct BindBlockRealize {
    BindBlockRealize(TransformLayoutPlanner* self, SBlockRealize block_realize) : self_(self) {
      TVM_FFI_ICHECK_EQ(block_realize->iter_values.size(), block_realize->block->iter_vars.size());
      for (size_t i = 0; i < block_realize->iter_values.size(); i++) {
        bound_vars_.emplace_back(self, block_realize->block->iter_vars[i]->var,
                                 block_realize->iter_values[i]);
      }
      cache_ = std::move(block_realize);
      std::swap(self_->innermost_block_realize_, cache_);
    }
    ~BindBlockRealize() { std::swap(self_->innermost_block_realize_, cache_); }
    BindBlockRealize(const BindBlockRealize&) = delete;
    BindBlockRealize& operator=(const BindBlockRealize&) = delete;
    BindBlockRealize(BindBlockRealize&&) = delete;
    BindBlockRealize& operator=(BindBlockRealize&&) = delete;

    TransformLayoutPlanner* self_{nullptr};
    ffi::Optional<SBlockRealize> cache_;
    std::vector<BindVariableDefinition> bound_vars_;
  };

  /*! \brief Collected information about each TensorStore */
  std::vector<WriteInfo> write_info_;

  /*! \brief The loop iterators surrounding the current node
   *
   * The outermost loop iterator is `active_loops_.front()`, and the
   * innermost loop iterator is `active_loops_.back()`.
   *
   * Used to fill the `WriteInfo::dependent_loopnest` field.
   */
  std::vector<For> active_loops_;

  /*! \brief Lookup for the outer/inner loops
   *
   * Used to fill the `WriteInfo::dependent_loopnest` field.
   */
  std::unordered_map<const VarObj*, std::pair<size_t, size_t>> loop_depth_lookup_;

  /*! \brief The variable mappings that are currently in-scope
   *
   * Used to determine whether the indices of a TensorStore are a
   * row-major traversal, even if they are rebound in let/block
   * mappings.
   */
  std::unordered_map<const VarObj*, PrimExpr> active_var_bindings_;

  /*! \brief The innermost BlockRealize surrounding the current node
   *
   * Used to fill the `WriteInfo::innermost_block_realize` field..
   */
  ffi::Optional<SBlockRealize> innermost_block_realize_{std::nullopt};

  /*! \brief The tensor to be replaced */
  TensorVar old_tensor_;
};

/*!
 * \brief Collect blocks that are part of root block to be passed to ScheduleState::Replace for SRef
 * reuse
 */
class ReuseBlocksCollector : public s_tir::StmtExprVisitor {
 public:
  using s_tir::StmtExprVisitor::Visit_;

  ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
    if (value.as<ExprObj>()) return std::nullopt;
    return s_tir::StmtExprVisitor::Visit(value);
  }

  static ffi::Map<SBlock, SBlock> Collect(SBlock result,
                                          ffi::Map<SBlock, SBlock> new_block_to_old) {
    return ffi::make_object<ReuseBlocksCollector>(new_block_to_old)->Run(result);
  }

 private:
  /*! \brief Entry point */
  ffi::Map<SBlock, SBlock> Run(const SBlock result) {
    Visit(result);
    return block_sref_reuse_;
  }
  /*! \brief Constructor */
 public:
  explicit ReuseBlocksCollector(ffi::Map<SBlock, SBlock> new_block_to_old)
      : new_block_to_old_(new_block_to_old) {}

 private:
  /*! \brief Override the Stmt visiting behaviour */
  ffi::Optional<VisitInterrupt> Visit_(const s_tir::SBlockObj* block) override {
    SBlock block_ref = ffi::GetRef<SBlock>(block);
    auto it = new_block_to_old_.find(block_ref);
    if (it != new_block_to_old_.end()) {
      block_sref_reuse_.Set((*it).second, (*it).first);
    }
    return StmtExprVisitor::Visit_(block);
  }

  /*! \brief New map to be filled with just blocks from scope block */
  ffi::Map<SBlock, SBlock> block_sref_reuse_;
  /*! \brief All block replacements collected so far */
  ffi::Map<SBlock, SBlock> new_block_to_old_;
};

class TransformLayoutRewriter : public s_tir::StmtExprMutatorWithConstraintContext {
 public:
  using s_tir::StmtExprMutatorWithConstraintContext::Mutate;
  using s_tir::StmtExprMutatorWithConstraintContext::Mutate_;

  /*!
   * \brief Rewrite the access to the tensor after the transformation
   * \param scope_stmt The parent statement that contains all accesses to the target tensor
   * \param old_tensor The target tensor before transformation
   * \param new_tensor The new tensor after transformation
   * \param index_map The transformation applied to the tensor
   * \return The new AST rooting at the original parent scope and the map from the old block to the
   * new block
   */
  static std::pair<Stmt, ffi::Map<SBlock, SBlock>> Rewrite(
      const SBlock& scope_stmt, const TensorVar& old_tensor, const TensorVar& new_tensor,
      const IndexMap& index_map, const ffi::Optional<IndexMap>& opt_inverse,
      const PrimExpr& padding_predicate, const ffi::Optional<IndexMap>& pad_value) {
    sym::Analyzer analyzer;
    auto plan = pad_value.has_value()
                    ? TransformLayoutPlanner::Plan(scope_stmt, old_tensor, new_tensor, index_map,
                                                   opt_inverse.value(), padding_predicate,
                                                   pad_value, analyzer.get())
                    : TransformLayoutPlanner::NoPaddingRequired();

    auto rewriter = ffi::make_object<TransformLayoutRewriter>(old_tensor, new_tensor, index_map,
                                                              plan, analyzer);
    SBlock result = rewriter->Mutate(scope_stmt).ValueOrUnchanged(scope_stmt).as_or_throw<SBlock>();
    if (auto plan_ptr = std::get_if<TransformLayoutPlanner::ProloguePlan>(&plan)) {
      auto write_ptr = result.CopyOnWrite();
      write_ptr->body = SeqStmt({plan_ptr->prologue, write_ptr->body});
    }

    ffi::Map<SBlock, SBlock> block_sref_reuse =
        ReuseBlocksCollector::Collect(result, rewriter->new_block_to_old_);

    return {result, block_sref_reuse};
  }

  TransformLayoutRewriter(const TensorVar& old_tensor, const TensorVar& new_tensor,
                          const IndexMap& index_map,
                          const TransformLayoutPlanner::TransformPlan& plan,
                          const sym::Analyzer& analyzer)
      : StmtExprMutatorWithConstraintContext(analyzer->GetConstraintContext()),
        old_tensor_(old_tensor),
        new_tensor_(new_tensor),
        index_map_(index_map),
        plan_(plan),
        tensor_data_to_tensor_{{new_tensor, new_tensor}} {
    if (auto plan_ptr = std::get_if<TransformLayoutPlanner::ReplacementPlan>(&plan_)) {
      new_block_to_old_ = plan_ptr->new_block_to_old;
    }
  }

 private:
  void RewriteTensorAccess(TensorVar* tensor, ffi::Array<PrimExpr>* indices) {
    *tensor = new_tensor_;
    *indices = index_map_->MapIndices(*indices, index_simplifier_);
    *indices = this->IterMapSimplifyWithContext(*indices, true);
  }

  using Parent = s_tir::StmtExprMutatorWithConstraintContext;

  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) final {
    const auto* stmt = value.as<StmtObj>();
    auto plan_ptr = std::get_if<TransformLayoutPlanner::EpiloguePlan>(&plan_);
    bool insert_after = stmt && plan_ptr && plan_ptr->insert_after.get() == stmt;
    auto result = Parent::Mutate(value, inplace_mode);
    if (!insert_after) return result;
    Stmt output = std::move(result).ValueOrUnchanged(value).as_or_throw<Stmt>();
    return ffi::Any(SeqStmt({output, plan_ptr->new_block}));
  }

  UnchangedOr<Stmt> Mutate_(const ForObj* op, InplaceMode inplace_mode) final {
    // Some replacements may include the original string, such as
    // replacing `loop` with `{loop, post_proc}`.  In this case, avoid
    // infinite recursion.

    For node = ffi::GetRef<For>(op);
    if (auto plan_ptr = std::get_if<TransformLayoutPlanner::ReplacementPlan>(&plan_)) {
      auto it = plan_ptr->replacements.find(node);
      if (it != plan_ptr->replacements.end()) {
        return Mutate((*it).second).ValueOrUnchanged((*it).second);
      }
    }
    return Parent::Mutate_(op, inplace_mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* op, InplaceMode inplace_mode) final {
    TensorLoad tensor_load = Parent::Mutate_(op, inplace_mode)
                                 .ValueOrUnchanged(ffi::GetRef<PrimExpr>(op))
                                 .as_or_throw<TensorLoad>();
    if (tensor_load->source.as_or_throw<tvm::tirx::TensorVar>().same_as(old_tensor_)) {
      TensorVar tensor = tensor_load->source.as_or_throw<tvm::tirx::TensorVar>();
      ffi::Array<PrimExpr> indices = tensor_load->indices;
      RewriteTensorAccess(&tensor, &indices);
      return TensorLoad(tensor, indices, tensor_load->loc);
    }
    return tensor_load;
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreObj* op, InplaceMode inplace_mode) final {
    TensorStore tensor_store = Parent::Mutate_(op, inplace_mode)
                                   .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                                   .as_or_throw<TensorStore>();
    if (tensor_store->dest.as_or_throw<TensorVar>().same_as(old_tensor_)) {
      auto* n = tensor_store.CopyOnWrite();
      TensorVar tensor = n->dest.as_or_throw<TensorVar>();
      RewriteTensorAccess(&tensor, &n->indices);
      n->dest = std::move(tensor);
    }
    return tensor_store;
  }

  void RewriteAccessRegion(ffi::Array<TensorRegion>* old_access_regions,
                           const ffi::Array<TensorRegion>& infered_access_regions) {
    auto fmutate = [this, &infered_access_regions](const TensorRegion& tensor_region) {
      if (tensor_region->source.as_or_throw<tvm::tirx::TensorVar>().same_as(old_tensor_)) {
        TVM_FFI_ICHECK(infered_access_regions.size() == 1);
        TensorRegion result = infered_access_regions[0];
        // The inferred region may reference old_tensor_ (e.g. when resolved
        // through match_tensor source).  Ensure we use new_tensor_ instead.
        if (result->source.as_or_throw<tvm::tirx::TensorVar>().same_as(old_tensor_)) {
          auto* n = result.CopyOnWrite();
          n->source = new_tensor_;
        }
        return result;
      }
      return tensor_region;
    };
    (*old_access_regions).MutateByApply(fmutate);
  }

  UnchangedOr<Stmt> Mutate_(const SBlockObj* op, InplaceMode inplace_mode) final {
    SBlock orig = [&]() {
      SBlock block = ffi::GetRef<SBlock>(op);
      while (true) {
        if (auto it = new_block_to_old_.find(block); it != new_block_to_old_.end()) {
          block = (*it).second;
        } else {
          break;
        }
      }
      return block;
    }();

    SBlock block = Parent::Mutate_(op, inplace_mode)
                       .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                       .as_or_throw<SBlock>();

    return WithConstraintScope([&]() -> ffi::Expected<UnchangedOr<Stmt>> {
             for (const auto& iter_var : block->iter_vars) {
               AppendConstraint(sym::RangeConstraint(iter_var->var, iter_var->dom.value()));
             }
             auto infered_access_regions = GetSBlockReadWriteRegion(block, tensor_data_to_tensor_);
             auto* n = block.CopyOnWrite();
             RewriteAccessRegion(&n->reads, infered_access_regions[0]);
             RewriteAccessRegion(&n->writes, infered_access_regions[1]);
             // Update match_tensors whose source references old_tensor_
             n->match_tensors.MutateByApply([this](const MatchTensorRegion& match_tensor) {
               if (match_tensor->source->source.as_or_throw<tvm::tirx::TensorVar>().same_as(
                       old_tensor_)) {
                 auto new_source = match_tensor->source;
                 auto* source_n = new_source.CopyOnWrite();
                 source_n->source = new_tensor_;
                 auto new_match = match_tensor;
                 new_match.CopyOnWrite()->source = new_source;
                 return new_match;
               }
               return match_tensor;
             });
             n->alloc_tensors.MutateByApply([this](const TensorVar& tensor) {
               if (tensor.same_as(old_tensor_)) {
                 return new_tensor_;
               } else {
                 return tensor;
               }
             });

             RecordReplacement(orig, block);
             return block;
           })
        .value();
  }

  void RecordReplacement(SBlock before, SBlock after) {
    if (before.same_as(after)) {
      return;
    }

    TVM_FFI_ICHECK(!new_block_to_old_.count(after));

    while (true) {
      if (auto opt = new_block_to_old_.Get(before)) {
        before = opt.value();
      } else {
        break;
      }
    }

    new_block_to_old_.Set(after, before);
  }

  const TensorVar& old_tensor_;
  const TensorVar& new_tensor_;
  const IndexMap& index_map_;
  const TransformLayoutPlanner::TransformPlan& plan_;
  ffi::Map<Var, TensorVar> tensor_data_to_tensor_;
  ffi::Map<SBlock, SBlock> new_block_to_old_;
  sym::Analyzer index_simplifier_;
};

class TensorIsSubregionError : public ScheduleErrorContextObj {
 public:
  explicit TensorIsSubregionError(IRModule mod, TensorVar tensor) : mod_(mod), tensor_(tensor) {}

  ffi::String FastErrorString() const final {
    return "ScheduleError: The input tensor is defined in `match_tensor` of a block, it is expected"
           " to be a function parameter or allocated by a block";
  }

  ffi::String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "ScheduleError: The input tensor " << tensor_->name << " is defined in `match_tensor` of "
       << "a block, it is expected to be a function parameter or allocated by a block.";
    return os.str();
  }

  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {}; }
  IRModule mod() const final { return mod_; }

 private:
  IRModule mod_;
  TensorVar tensor_;
};

class TransformationPaddingIndexMapError : public ScheduleErrorContextObj {
 public:
  TransformationPaddingIndexMapError(IRModule mod, IndexMap pad_value)
      : mod_(mod), pad_value_(pad_value) {}

  ffi::String FastErrorString() const final {
    std::ostringstream ss;
    ss << "ScheduleError: The IndexMap specifying pad_value has "
       << pad_value_->final_indices.size() << " outputs, should only have one output";
    return ss.str();
  }

  ffi::String DetailRenderTemplate() const final {
    std::ostringstream ss;
    ss << "ScheduleError: Pad value is specified as " << pad_value_ << " which has "
       << pad_value_->final_indices.size() << " outputs, but should only have one output";
    return ss.str();
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {}; }

 private:
  IRModule mod_;
  IndexMap pad_value_;
};

class TransformationPaddingTypeError : public ScheduleErrorContextObj {
 public:
  TransformationPaddingTypeError(IRModule mod, TensorVar tensor, IndexMap pad_value)
      : mod_(mod), tensor_(tensor), pad_value_(pad_value) {
    TVM_FFI_ICHECK_EQ(pad_value_->final_indices.size(), 1);
    pad_value_dtype_ = pad_value_->final_indices[0].ty()->dtype;
  }

  ffi::String FastErrorString() const final {
    std::ostringstream ss;
    ss << "ScheduleError: Type mismatch " << tensor_.dtype() << " vs " << pad_value_dtype_;
    return ss.str();
  }

  ffi::String DetailRenderTemplate() const final {
    std::ostringstream ss;
    ss << "ScheduleError: Tensor " << tensor_->name << " has elements of type " << tensor_.dtype()
       << ", but the transformation fills padding with " << pad_value_ << ", which is of type "
       << pad_value_dtype_;
    return ss.str();
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {}; }

 private:
  IRModule mod_;
  TensorVar tensor_;
  IndexMap pad_value_;
  DLDataType pad_value_dtype_;
};

class TransformationPaddingExpressionError : public ScheduleErrorContextObj {
 public:
  static void Check(IRModule mod, TensorVar tensor, IndexMap pad_value) {
    auto visitor = ffi::make_object<Visitor>(tensor);
    TVM_FFI_ICHECK_EQ(pad_value->final_indices.size(), 1)
        << "Internal error: Should be caught by ScheduleError checks prior to this point";
    visitor->Visit(pad_value->final_indices[0]);
    if (visitor->illegal_load) {
      throw MakeScheduleError<TransformationPaddingExpressionError>(mod, tensor, pad_value,
                                                                    visitor->illegal_load.value());
    }
  }

 private:
  struct Visitor : StmtExprVisitor {
   public:
    using StmtExprVisitor::Visit_;

    explicit Visitor(const TensorVar& tensor) : tensor_(tensor) {}

    ffi::Optional<VisitInterrupt> Visit_(const TensorLoadObj* op) final {
      if (!op->source.as_or_throw<tvm::tirx::TensorVar>().same_as(tensor_)) {
        illegal_load = ffi::GetRef<TensorLoad>(op);
      }
      return StmtExprVisitor::Visit_(op);
    }

    const TensorVar& tensor_;
    ffi::Optional<TensorLoad> illegal_load;
  };

 public:
  TransformationPaddingExpressionError(IRModule mod, TensorVar tensor, IndexMap pad_value,
                                       TensorLoad illegal_load)
      : mod_(mod), tensor_(tensor), pad_value_(pad_value), illegal_load_(illegal_load) {}

 private:
  ffi::String FastErrorString() const final {
    std::ostringstream ss;
    ss << "ScheduleError: Pad value may not contain load from "
       << illegal_load_->source.as_or_throw<tvm::tirx::TensorVar>()->name;
    return ss.str();
  }

  ffi::String DetailRenderTemplate() const final {
    std::ostringstream ss;
    ss << "ScheduleError: Pad value may only contain TensorLoad from the transformed tensor "
       << tensor_->name << ", but pad_value " << pad_value_ << " contains expression "
       << illegal_load_;
    return ss.str();
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {}; }

  IRModule mod_;
  TensorVar tensor_;
  IndexMap pad_value_;
  TensorLoad illegal_load_;
};

class TransformationIntroducesPaddingError : public ScheduleErrorContextObj {
 public:
  TransformationIntroducesPaddingError(IRModule mod, TensorVar tensor, IndexMap index_map,
                                       PrimExpr padding_predicate)
      : mod_(std::move(mod)),
        tensor_(std::move(tensor)),
        index_map_(std::move(index_map)),
        padding_predicate_(std::move(padding_predicate)) {}

  ffi::String FastErrorString() const final {
    std::ostringstream ss;
    ss << "ScheduleError: Transformation would introduce padding at " << padding_predicate_ << ".";
    return ss.str();
  }

  ffi::String DetailRenderTemplate() const final {
    sym::Analyzer analyzer;
    auto new_shape = index_map_->MapShape(tensor_.shape(), analyzer);
    std::ostringstream os;
    os << "The transformation " << index_map_ << " applied on tensor " << tensor_->name
       << " of shape " << tensor_.shape() << " would result in shape " << new_shape
       << ".  However, this would introduce padding wherever " << padding_predicate_ << " is true.";
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {}; }

 private:
  IRModule mod_;
  TensorVar tensor_;
  IndexMap index_map_;
  PrimExpr padding_predicate_;
};

// Make the dtypes of indices in IndexMap be the same as the dtype of the tensor shape, to avoid
// dtype-mismatch issues later.
IndexMap LegalizeIndexMapDType(const IndexMap& index_map, const ffi::Array<PrimExpr>& args) {
  const auto& initial_indices_orig = index_map->initial_indices;
  TVM_FFI_ICHECK(args.size() == initial_indices_orig.size());

  ffi::Array<Var> initial_indices;
  ffi::Map<Var, PrimExpr> var_map;
  std::optional<DLDataType> index_dtype = std::nullopt;

  for (size_t i = 0; i < args.size(); ++i) {
    DLDataType arg_dtype = args[i].ty()->dtype;
    if (index_dtype.has_value()) {
      TVM_FFI_ICHECK_EQ(*index_dtype, arg_dtype)
          << "Tensor index " << args[i] << " has dtype " << arg_dtype
          << ", but previous index for the same tensor access used index type " << *index_dtype;
    } else {
      index_dtype = arg_dtype;
    }

    DLDataType initial_dtype = initial_indices_orig[i].ty()->dtype;
    if (arg_dtype != initial_dtype) {
      auto new_idx = Var(initial_indices_orig[i]->name, args[i].ty());
      initial_indices.push_back(new_idx);
      var_map.Set(initial_indices_orig[i], new_idx.as_or_throw<PrimExpr>());
    } else {
      initial_indices.push_back(initial_indices_orig[i]);
    }
  }

  if (!var_map.empty()) {
    auto final_indices = index_map->final_indices.Map([&](PrimExpr index) {
      if (auto* ptr = index.as<IntImmObj>()) {
        TVM_FFI_ICHECK(index_dtype.has_value());
        return tvm::prim::MakeConst(PrimType(*index_dtype), ptr->value);
      } else {
        return SubstituteWithDataTypeLegalization(
            index, [&](const Var& var) -> ffi::Optional<PrimExpr> { return var_map.Get(var); });
      }
    });
    ffi::Optional<IndexMap> opt_inverse_index_map = std::nullopt;
    if (index_map->inverse_index_map.has_value()) {
      opt_inverse_index_map = index_map->inverse_index_map.value().as_or_throw<IndexMap>();
    }
    if (opt_inverse_index_map.has_value()) {
      opt_inverse_index_map = LegalizeIndexMapDType(opt_inverse_index_map.value(), final_indices);
    }
    return IndexMap(initial_indices.Map([](Var var) { return var.as_or_throw<PrimVar>(); }),
                    final_indices, opt_inverse_index_map);
  }
  return index_map;
}

void TransformLayout(ScheduleState self, const StmtSRef& block_sref, int tensor_index,
                     TensorIndexType tensor_index_type, const IndexMap& index_map_orig,
                     const ffi::Optional<IndexMap>& pad_value, bool assume_injective_transform) {
  sym::Analyzer analyzer;
  AddShapeVarBounds(self, block_sref.get(), analyzer.get());
  // Step 1: Input handling and error checking
  const SBlockObj* block_ptr = TVM_SREF_TO_SBLOCK(block_sref);
  TensorVar old_tensor =
      GetNthAccessTensor(self, ffi::GetRef<SBlock>(block_ptr), tensor_index, tensor_index_type);

  auto index_map = LegalizeIndexMapDType(index_map_orig, old_tensor.shape());

  auto [defining_site_sref, is_alloc] = GetTensorDefiningSite(block_sref, old_tensor);
  if (defining_site_sref.has_value() && !is_alloc) {
    throw MakeScheduleError<TensorIsSubregionError>(self->mod, old_tensor);
  }
  if (pad_value) {
    if (pad_value.value()->final_indices.size() != 1) {
      throw MakeScheduleError<TransformationPaddingIndexMapError>(self->mod, pad_value.value());
    }
    if (pad_value.value()->final_indices[0].ty() != old_tensor.dtype()) {
      throw MakeScheduleError<TransformationPaddingTypeError>(self->mod, old_tensor,
                                                              pad_value.value());
    }

    TransformationPaddingExpressionError::Check(self->mod, old_tensor, pad_value.value());
  }

  StmtSRef scope_sref = defining_site_sref.has_value()
                            ? defining_site_sref.value()
                            : GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/false);
  const SBlockObj* scope_block = TVM_SREF_TO_SBLOCK(scope_sref);

  ffi::Optional<IndexMap> opt_inverse = std::nullopt;
  PrimExpr padding_predicate = IntImm::Bool(false);
  if (!assume_injective_transform) {
    auto inverse_and_padding = [&]() {
      ffi::Array<Range> region;
      for (const auto& dim : old_tensor.shape()) {
        region.push_back(Range::FromMinExtent(IntImm(dim.ty(), 0), dim));
      }
      return index_map.NonSurjectiveInverse(region, analyzer);
    }();
    opt_inverse = inverse_and_padding.get<0>();
    padding_predicate = inverse_and_padding.get<1>();
  }

  bool has_padding = !IsZero(padding_predicate);
  if (has_padding && !pad_value.has_value()) {
    throw MakeScheduleError<TransformationIntroducesPaddingError>(self->mod, old_tensor, index_map,
                                                                  padding_predicate);
  }

  // Step 2: Infer the shape of the new tensor
  auto new_tensor_type = ffi::make_object<TensorTypeObj>(*old_tensor.ty().get());
  new_tensor_type->shape = index_map->MapShape(old_tensor.shape(), analyzer);
  TensorVar new_tensor =
      TensorVar(old_tensor->name, TensorType(std::move(new_tensor_type)), old_tensor->loc);

  // Step 3: Rewrite TensorLoad/TensorStore access indices, block read/write regions, and block
  // alloc_tensors.
  auto [new_stmt, block_sref_reuse] =
      TransformLayoutRewriter::Rewrite(ffi::GetRef<SBlock>(scope_block), old_tensor, new_tensor,
                                       index_map, opt_inverse, padding_predicate, pad_value);
  SBlock new_scope_block = new_stmt.as_or_throw<SBlock>();

  // Step 4: Rewrite the Function tensor parameter if necessary.
  if (!defining_site_sref.has_value()) {
    GlobalVar g_var{ffi::UnsafeInit{}};
    const auto* old_func = GetRootFunction(self->mod, scope_block, &g_var);
    IRModuleObj* new_mod = self->mod.CopyOnWrite();
    ffi::MapObj* new_map = new_mod->functions.CopyOnWrite();

    ffi::Array<Var> new_params = old_func->params.Map([&](Var param) -> Var {
      if (auto tensor = param.as<TensorVar>(); tensor && tensor.value().same_as(old_tensor)) {
        return new_tensor;
      }
      return param;
    });

    Function ref_new_func(new_params, old_func->body, old_func->ret_type, old_func->attrs,
                          old_func->loc);
    new_map->at(g_var) = std::move(ref_new_func);
  }

  // Step 4: Replace the scope block with the new block
  self->Replace(scope_sref, new_scope_block, block_sref_reuse);
}

/*!
 * \brief Detect the block iter type associated with the expression
 *
 * This function collects block iters in the expression and check if the block iters have the same
 * iter type. The detected iter type is the iter type of the block iters in the expression
 * if they have the same iter type, otherwise the detected iter type will be kOpaque.
 *
 * \param expr The expression
 * \param block_iter_type_map The mapping from block iter to iter type
 * \return The detected block iter type
 */
IterVarType DetectNewBlockIterType(
    const PrimExpr& expr,
    const std::unordered_map<const VarObj*, IterVarType>& block_iter_type_map) {
  IterVarType result{kOpaque};
  bool found = false;
  auto walk_fn = [&](const Var& var) -> ffi::Expected<ffi::WalkResult> {
    if (auto prim_var = var.as<PrimVar>()) {
      auto it = block_iter_type_map.find(prim_var.value().get());
      if (it != block_iter_type_map.end()) {
        if (!found) {
          found = true;
          result = it->second;
        } else if (result != it->second) {
          result = kOpaque;
          return ffi::WalkResult::Interrupt();
        }
      }
    }
    return ffi::WalkResult::Advance();
  };
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(expr, walk_fn);
  return result;
}

class NotBijectiveAffineIndexMapError : public ScheduleErrorContextObj {
 public:
  NotBijectiveAffineIndexMapError(IRModule mod, IndexMap index_map)
      : mod_(std::move(mod)), index_map_(std::move(index_map)) {}
  ffi::String FastErrorString() const final {
    return "ScheduleError: The index map is not bijective affine.";
  }

  ffi::String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The index map " << index_map_->ToPythonString() << " is not bijective affine.";
    return os.str();
  }

  IRModule mod() const final { return mod_; }

  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {}; }

 private:
  IRModule mod_;
  IndexMap index_map_;
};

class IndexMapNotApplicableToBlockIterError : public ScheduleErrorContextObj {
 public:
  static void Check(const IRModule mod, const SBlock& block, const IndexMap& index_map) {
    if (index_map->initial_indices.size() != block->iter_vars.size()) {
      throw MakeScheduleError<IndexMapNotApplicableToBlockIterError>(mod, block, index_map);
    }
  }
  explicit IndexMapNotApplicableToBlockIterError(IRModule mod, SBlock block, IndexMap index_map)
      : mod_(std::move(mod)), block_(std::move(block)), index_map_(std::move(index_map)) {}

  ffi::String FastErrorString() const final {
    return "ScheduleError: The index map can't be applied to block iters because the number of "
           "parameters mismatch.";
  }

  ffi::String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The index map " << index_map_->ToPythonString()
       << " can't be applied to block iters of {0} because the number of parameters mismatch. "
          "Expected: "
       << index_map_->initial_indices.size() << ", actual: " << block_->iter_vars.size();
    return os.str();
  }

  IRModule mod() const final { return mod_; }

  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_}; }

 private:
  IRModule mod_;
  SBlock block_;
  IndexMap index_map_;
};

class OpaqueNewIterTypeError : public ScheduleErrorContextObj {
 public:
  explicit OpaqueNewIterTypeError(IRModule mod, SBlock block, PrimExpr iter_value)
      : mod_(std::move(mod)), block_(std::move(block)), iter_value_(std::move(iter_value)) {}

  ffi::String FastErrorString() const final {
    return "ScheduleError: Cannot detect the new block iter type because it contains more than one "
           "type of original iter vars.";
  }

  ffi::String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "Cannot detect the block iter type for new iter value " << iter_value_
       << " in {0} because it contains more than one type of original iter vars.";
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_}; }

 private:
  IRModule mod_;
  SBlock block_;
  PrimExpr iter_value_;
};

void TransformBlockLayout(ScheduleState self, const StmtSRef& block_sref,
                          const IndexMap& index_map) {
  const SBlockObj* block_ptr = TVM_SREF_TO_SBLOCK(block_sref);
  const SBlock& block = ffi::GetRef<SBlock>(block_ptr);
  sym::Analyzer analyzer;
  AddShapeVarBounds(self, block_sref.get(), analyzer.get());

  // Step 1: Collect outer loops and loop vars
  ffi::Array<StmtSRef> loops = GetLoops(block_sref);  // outer loops of the block
  std::unordered_set<const VarObj*> loop_vars;        // loop vars of the outer loops
  for (const StmtSRef& loop_sref : loops) {
    CheckLoopStartsWithZero(self, loop_sref, analyzer.get());
    loop_vars.emplace(loop_sref->StmtAs<ForObj>()->var.get());
  }

  // Step 2: Check the all outer loops have a single child and the block bindings are trivial (all
  // binding values are loop vars)
  StmtSRef scope_sref{ffi::UnsafeInit{}};  // the scope statement for replacement
  if (!loops.empty()) {
    scope_sref = loops.front();
    CheckGetSingleChildBlockRealizeOnSRefTree(self, loops.front());
  } else {
    scope_sref = block_sref;
  }

  SBlockRealize block_realize = GetSBlockRealize(self, block_sref);
  CheckBlockHasTrivialBinding(self, block_sref);

  // Step 3: Collect information of block iter vars
  ffi::Array<PrimExpr> block_vars;      // iter_var->var of each block iter
  ffi::Map<Var, Range> block_iter_dom;  // domain of block iter
  std::unordered_map<const VarObj*, IterVarType> block_iter_type;  // iter type of block iter

  ffi::Array<PrimExpr>
      block_iter_range_array;  // array of block iter extents in the same order as block iters
  for (const auto& iter_var : block->iter_vars) {
    block_vars.push_back(iter_var->var);
    block_iter_dom.Set(iter_var->var, iter_var->dom.value());
    block_iter_type[iter_var->var.get()] = iter_var->iter_type;
    TVM_FFI_ICHECK(IsZero(iter_var->dom.value()->begin));
    block_iter_range_array.push_back(iter_var->dom.value()->extent);
  }

  // Step 4: Apply the IndexMap to block iters.
  IndexMapNotApplicableToBlockIterError::Check(self->mod, block, index_map);
  ffi::Array<PrimExpr> transformed_block_iters = index_map->MapIndices(block_vars, analyzer);
  ffi::Array<PrimExpr> new_block_iter_range = index_map->MapShape(block_iter_range_array, analyzer);

  // Step 5: Create the new block after transformation.

  // Step 5.1: Create new block iters. After applying the IndexMap f to block iters ax_0, ..., ax_n,
  // create block iter each expression in f(ax_0, ..., ax_n).
  ffi::Array<IterVar> new_block_iters;  // new block iters
  ffi::Array<PrimExpr> new_block_vars;  // iter_var->var of new block iters
  for (size_t i = 0; i < transformed_block_iters.size(); ++i) {
    Var new_block_var{"v" + std::to_string(i), transformed_block_iters[i].ty()};
    new_block_vars.push_back(new_block_var.as_or_throw<PrimExpr>());
    IterVarType iter_type;
    if (IsOne(new_block_iter_range[i])) {
      iter_type = kDataPar;
    } else {
      iter_type = DetectNewBlockIterType(transformed_block_iters[i], block_iter_type);
    }
    if (iter_type == kOpaque) {
      throw MakeScheduleError<OpaqueNewIterTypeError>(self->mod, ffi::GetRef<SBlock>(block_ptr),
                                                      transformed_block_iters[i]);
    }
    PrimType dtype = new_block_var->ty.as_or_throw<PrimType>();
    new_block_iters.push_back(IterVar(
        /*dom=*/Range::FromMinExtent(IntImm(dtype, 0), cast(new_block_iter_range[i], dtype)),
        /*var=*/std::move(new_block_var).as_or_throw<PrimVar>(), /*iter_type=*/iter_type));
  }

  // Step 5.2: Update the block body. Use the inverse map f^{-1} to replace the original block iters
  // in the body.
  ffi::Map<Var, PrimExpr> inverse_subst_map;
  // Construct the inverse map
  {
    ffi::Array<Range> initial_ranges;
    for (const PrimExpr& extent : block_iter_range_array) {
      initial_ranges.push_back(Range::FromMinExtent(IntImm(extent.ty(), 0), extent));
    }
    IndexMap inverse_index_map{ffi::UnsafeInit{}};
    try {
      inverse_index_map = index_map.Inverse(initial_ranges, analyzer);
    } catch (...) {
      throw MakeScheduleError<NotBijectiveAffineIndexMapError>(self->mod, index_map);
    }
    // old block vars written in terms of new block vars
    ffi::Array<PrimExpr> inversed_new_block_vars =
        inverse_index_map->MapIndices(new_block_vars, analyzer);
    for (int i = 0, n = block_vars.size(); i < n; ++i) {
      inverse_subst_map.Set(block_vars[i].as_or_throw<Var>(), inversed_new_block_vars[i]);
    }
  }
  auto f_substitute = [&inverse_subst_map](
                          const Var& var,
                          TVMFFIDefRegionKind kind) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (kind != kTVMFFIDefRegionKindNone) {
      return ffi::Unchanged();
    }
    if (auto repl = inverse_subst_map.Get(var)) return ffi::Any(*std::move(repl));
    return ffi::Unchanged();
  };
  SBlock new_block =
      ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(ffi::GetRef<SBlock>(block_ptr), f_substitute)
          .as_or_throw<SBlock>();
  new_block.CopyOnWrite()->iter_vars = new_block_iters;
  new_block = BlockTensorAccessSimplifier::Simplify(new_block, analyzer).as_or_throw<SBlock>();

  // Step 5.3: Create outer loops for each new block iter.

  // Make new loop vars
  ffi::Array<PrimExpr> new_loop_vars;
  for (int i = 0; i < static_cast<int>(new_block_iters.size()); ++i) {
    new_loop_vars.push_back(
        Var("ax" + std::to_string(i), new_block_iters[i]->var.ty()).as_or_throw<PrimExpr>());
  }

  // Make new block realize
  SBlockRealizeObj* new_block_realize = block_realize.CopyOnWrite();
  new_block_realize->iter_values = new_loop_vars;
  new_block_realize->block = new_block;

  // Generate outer loops
  Stmt body = ffi::GetRef<Stmt>(new_block_realize);
  for (int i = static_cast<int>(new_loop_vars.size()) - 1; i >= 0; --i) {
    body = For(new_loop_vars[i].as_or_throw<PrimVar>(), 0, new_block_iter_range[i],
               IntImm(new_loop_vars[i].as_or_throw<PrimVar>().ty(), 1), ForKind::kDefault,
               std::move(body));
  }

  // Step 6: Do the actual replacement
  if (scope_sref->StmtAs<SBlockObj>()) {
    TVM_FFI_ICHECK(new_loop_vars.empty())
        << "Invalid block to loop replacement due to layout transform " << index_map;
  }
  self->Replace(scope_sref, body, {{block, new_block}});
}

/******** InstructionKind Registration ********/

struct TransformLayoutTraits : public UnpackedInstTraits<TransformLayoutTraits> {
  static constexpr const char* kName = "TransformLayout";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 4;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, SBlockRV block_rv, IndexMap index_map,
                                      int tensor_index, int tensor_index_type,
                                      ffi::Optional<IndexMap> pad_value,
                                      bool assume_injective_transform) {
    return sch->TransformLayout(block_rv, tensor_index,
                                static_cast<TensorIndexType>(tensor_index_type), index_map,
                                pad_value, assume_injective_transform);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block_rv,
                                      IndexMap index_map, int tensor_index, int tensor_index_type,
                                      ffi::Optional<IndexMap> pad_value,
                                      bool assume_injective_transform) {
    PythonAPICall py("transform_layout");
    py.Input("block", block_rv);

    std::ostringstream os;
    os << "(\"" << TensorIndexType2Str(static_cast<TensorIndexType>(tensor_index_type)) << "\", "
       << tensor_index << ")";
    py.Input("tensor", os.str());
    py.Input("index_map", index_map->ToPythonString());
    py.Input("pad_value", pad_value ? pad_value.value()->ToPythonString() : "None");
    py.Input("assume_injective_transform", assume_injective_transform);

    return py.Str();
  }

 public:
  static ffi::ObjectRef AttrsAsJSON(const ffi::Array<Any>& attrs) {
    ffi::Array<Any> attrs_record;
    attrs_record.reserve(kNumAttrs);
    attrs_record.push_back(attrs[0]);
    attrs_record.push_back(attrs[1]);
    if (attrs[2] != nullptr) {
      attrs_record.push_back(ffi::String(ffi::json::Stringify(
          ffi::ToJSONGraph(attrs[2], ffi::json::Object{{"tvm_version", TVM_VERSION}}),
          /*indent=*/2)));
    } else {
      attrs_record.push_back(attrs[2]);
    }
    attrs_record.push_back(attrs[3]);
    return attrs_record;
  }

  static ffi::Array<Any> AttrsFromJSON(const ffi::ObjectRef& attrs_record_) {
    ffi::Array<Any> attrs_record = attrs_record_.as_or_throw<ffi::Array<Any>>();
    ffi::Array<Any> attrs;
    attrs.push_back(attrs_record[0]);
    attrs.push_back(attrs_record[1]);
    if (attrs_record[2] != nullptr) {
      attrs.push_back(
          ffi::FromJSONGraph(ffi::json::Parse(attrs_record[2].as_or_throw<ffi::String>())));
    } else {
      attrs.push_back(attrs_record[2]);
    }
    attrs.push_back(attrs_record[3]);
    return attrs;
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

struct TransformBlockLayoutTraits : public UnpackedInstTraits<TransformBlockLayoutTraits> {
  static constexpr const char* kName = "TransformBlockLayout";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, SBlockRV block_rv, IndexMap index_map) {
    return sch->TransformBlockLayout(block_rv, index_map);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block_rv,
                                      IndexMap index_map) {
    PythonAPICall py("transform_block_layout");
    py.Input("block", block_rv);
    py.Input("index_map", index_map->ToPythonString());
    return py.Str();
  }

 public:
  static ffi::ObjectRef AttrsAsJSON(const ffi::Array<Any>& attrs) {
    ffi::Array<Any> attrs_record;
    attrs_record.reserve(kNumAttrs);
    attrs_record.push_back(ffi::String(ffi::json::Stringify(
        ffi::ToJSONGraph(attrs[0], ffi::json::Object{{"tvm_version", TVM_VERSION}}),
        /*indent=*/2)));
    return attrs_record;
  }

  static ffi::Array<Any> AttrsFromJSON(const ffi::ObjectRef& attrs_record_) {
    ffi::Array<Any> attrs_record = attrs_record_.as_or_throw<ffi::Array<Any>>();
    ffi::Array<Any> attrs;
    attrs.push_back(
        ffi::FromJSONGraph(ffi::json::Parse(attrs_record[0].as_or_throw<ffi::String>())));
    return attrs;
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

TVM_FFI_STATIC_INIT_BLOCK() {
  RegisterInstructionKind<TransformLayoutTraits>();
  RegisterInstructionKind<TransformBlockLayoutTraits>();
}

}  // namespace s_tir
}  // namespace tvm
