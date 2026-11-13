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

#include "../utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::prim;
using namespace tvm::tirx;

/*! \brief Append a new predicate to the each child of type BlockRealize (not recursively) */
class BlockPredicateAppender : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) override {
    if (value.as<ExprObj>()) return ffi::Unchanged();
    return StmtExprMutator::Mutate(value, inplace_mode);
  }

  /*!
   * \brief Constructor
   * \param to_append The predicate to be appended to BlockRealizeNode
   */
  explicit BlockPredicateAppender(const PrimExpr& to_append) : to_append_(to_append) {}

 private:
  // For each direct child of type BlockRealizeNode, append the predicate
  UnchangedOr<Stmt> Mutate_(const SBlockRealizeObj* realize, InplaceMode inplace_mode) final {
    // We do not recursively do this
    PrimExpr predicate = realize->predicate && to_append_;
    if (inplace_mode == InplaceMode::kAllow) {
      const_cast<SBlockRealizeObj*>(realize)->predicate = std::move(predicate);
      return ffi::Unchanged();
    } else {
      auto copy = ffi::make_object<SBlockRealizeObj>(*realize);
      copy->predicate = std::move(predicate);
      return SBlockRealize(std::move(copy));
    }
  }

  /*! \brief The predicate to be appended */
  const PrimExpr& to_append_;
};

/*! \brief Substitute vars and collect the reuse mapping of opaque blocks */
class SubstituteVarAndCollectOpaqueBlock : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  explicit SubstituteVarAndCollectOpaqueBlock(const ffi::Map<Var, PrimExpr>& substitutions,
                                              ffi::Map<SBlock, SBlock>* opaque_blocks)
      : opaque_blocks_(opaque_blocks) {
    for (const auto& [var, replacement] : substitutions) {
      VarRemapSet(var, tvm::prim::cast(replacement, var->ty.as_or_throw<PrimType>()));
    }
  }

 private:
  UnchangedOr<Stmt> Mutate_(const SBlockRealizeObj* op, InplaceMode inplace_mode) final {
    SBlockRealize realize = StmtExprMutator::Mutate_(op, inplace_mode)
                                .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                                .as_or_throw<SBlockRealize>();
    if (realize->block->iter_vars.empty()) {
      opaque_blocks_->Set(op->block, realize->block);
    }
    return realize;
  }

  /*! \brief The reuse mapping of opaque blocks */
  ffi::Map<SBlock, SBlock>* opaque_blocks_;
};

/*! \brief Simplify the binding of block realize and update the opaque block reuse mapping */
class IterMapSimplifyBlockBinding : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  explicit IterMapSimplifyBlockBinding(ffi::MapObj* opaque_blocks,
                                       ffi::Map<PrimVar, Range> loop_var2extent,
                                       bool preserve_unit_iters)
      : opaque_blocks_(opaque_blocks),
        loop_var2extent_(loop_var2extent),
        preserve_unit_iters_(preserve_unit_iters) {}

  static For SimplifyBindings(Stmt stmt, const ffi::Array<StmtSRef>& loop_srefs,
                              ffi::MapObj* opaque_blocks, bool preserve_unit_iters) {
    ffi::Map<PrimVar, Range> loop_var2extent;
    for (const StmtSRef& sref : loop_srefs) {
      const ForObj* loop = TVM_SREF_TO_FOR(sref);
      loop_var2extent.Set(loop->var, Range(loop->begin, loop->end));
    }
    return ffi::make_object<IterMapSimplifyBlockBinding>(opaque_blocks, std::move(loop_var2extent),
                                                         preserve_unit_iters)
        ->Mutate(stmt, InplaceMode::kAllow)
        .ValueOrUnchanged(std::move(stmt))
        .as_or_throw<For>();
  }

 private:
  UnchangedOr<Stmt> Mutate_(const ForObj* op, InplaceMode inplace_mode) final {
    loop_var2extent_.Set(op->var, Range(op->begin, op->end));
    Stmt res = StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    loop_var2extent_.erase(op->var);
    return res;
  }

  UnchangedOr<Stmt> Mutate_(const SBlockRealizeObj* op, InplaceMode inplace_mode) final {
    // skip opaque block and update mapping
    if (op->iter_values.empty()) {
      SBlock block = op->block;
      SBlockRealize realize = StmtExprMutator::Mutate_(op, inplace_mode)
                                  .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                                  .as_or_throw<SBlockRealize>();
      for (const auto& entry : *opaque_blocks_) {
        if (entry.second.same_as(block)) {
          opaque_blocks_->at(entry.first) = realize->block;
          break;
        }
      }
      return realize;
    }
    ffi::Array<PrimExpr> v =
        sym::IterMapSimplify(/*indices=*/op->iter_values,
                             /*input_iters=*/loop_var2extent_,
                             /*input_pred=*/op->predicate,
                             /*check_level=*/sym::IterMapLevel::Surjective,
                             /*analyzer=*/analzyer_,
                             /*simplify_trivial_iterators=*/!preserve_unit_iters_);
    if (v.same_as(op->iter_values)) {
      return ffi::Unchanged();
    } else {
      if (inplace_mode == InplaceMode::kAllow) {
        const_cast<SBlockRealizeObj*>(op)->iter_values = std::move(v);
        return ffi::Unchanged();
      } else {
        auto copy = ffi::make_object<SBlockRealizeObj>(*op);
        copy->iter_values = std::move(v);
        return SBlockRealize(std::move(copy));
      }
    }
  }

  /*! \brief The reuse mapping */
  ffi::MapObj* opaque_blocks_;
  /*! \brief The range of loops */
  ffi::Map<PrimVar, Range> loop_var2extent_;
  /*! \brief Internal analyzer */
  sym::Analyzer analzyer_;
  /*! \brief Whether or not to simplify unit iterators */
  bool preserve_unit_iters_;
};

class BlockPropertyError : public ScheduleErrorContextObj {
 public:
  /*!
   * \brief Check that all the blocks under the specific stmt have affine bindings
   *     wrt top loop sref and only have data-parallel or reduction block iters
   * \param self The state of the schedule
   * \param sref The sref to the specific stmt
   */
  static void CheckBlockIterTypeAndAffineBinding(const ScheduleState& self, const StmtSRefObj* top,
                                                 const StmtSRefObj* sref) {
    class BlockIterTypeAndAffineBindingChecker : public StmtExprVisitor {
     public:
      using StmtExprVisitor::Visit_;

      ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
        if (value.as<ExprObj>()) return std::nullopt;
        return StmtExprVisitor::Visit(value);
      }

      explicit BlockIterTypeAndAffineBindingChecker(const ScheduleState& state,
                                                    const StmtSRefObj* top)
          : state_(state), top_(top) {}

     private:
      ffi::Optional<VisitInterrupt> Visit_(const SBlockObj* op) final {
        for (const IterVar& iter_var : op->iter_vars) {
          if (iter_var->iter_type != kDataPar && iter_var->iter_type != kCommReduce) {
            throw MakeScheduleError<BlockPropertyError>(state_->mod, ffi::GetRef<SBlock>(op));
          }
          ffi::Optional<StmtSRef> high_exclusive = top_->parent
                                                       ? ffi::GetRef<StmtSRef>(top_->parent)
                                                       : ffi::Optional<StmtSRef>(std::nullopt);
          CheckPartialAffineBinding(state_, ffi::GetRef<SBlock>(op), high_exclusive);
        }
        return std::nullopt;
      }
      const ScheduleState& state_;
      const StmtSRefObj* top_;
    };

    auto checker = ffi::make_object<BlockIterTypeAndAffineBindingChecker>(self, top);
    checker->Visit(ffi::GetRef<Stmt>(sref->stmt));
  }

  explicit BlockPropertyError(IRModule mod, SBlock block) : mod_(mod), block_(std::move(block)) {}

  ffi::String FastErrorString() const final {
    return "ScheduleError: The block under the loops to be reordered have block iter type other "
           "than data-parallel or reduction";
  }

  ffi::String DetailRenderTemplate() const final {
    return "The block {0} under the loops to be reordered have block iter type other than "
           "data-parallel or reduction";
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_}; }

  IRModule mod_;
  SBlock block_;
};

class HasAnnotationOrThreadBindingError : public ScheduleErrorContextObj {
 public:
  explicit HasAnnotationOrThreadBindingError(IRModule mod, For loop)
      : mod_(mod), loop_(std::move(loop)) {}

  ffi::String FastErrorString() const final {
    return "ScheduleError: The primitive can't be applied because the loop has annotation or "
           "thread binding";
  }

  ffi::String DetailRenderTemplate() const final {
    return "The primitive can't be applied because the loop {0} has annotation or thread binding";
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {loop_}; }

  IRModule mod_;
  For loop_;
};

class OuterNotInnerParent : public ScheduleErrorContextObj {
 public:
  explicit OuterNotInnerParent(IRModule mod, For outer, For inner)
      : mod_(mod), outer_(std::move(outer)), inner_(std::move(inner)) {}

  ffi::String FastErrorString() const final {
    return "ScheduleError: The outer loop is not the parent of the inner loop";
  }

  ffi::String DetailRenderTemplate() const final {
    return "The loops can't be fused because the outer loop {0} is not the parent of the inner "
           "loop {1}";
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {outer_, inner_}; }

  IRModule mod_;
  For outer_;
  For inner_;
};

class NotOnlyChildError : public ScheduleErrorContextObj {
 public:
  explicit NotOnlyChildError(IRModule mod, For outer, For inner)
      : mod_(mod), outer_(std::move(outer)), inner_(std::move(inner)) {}

  ffi::String FastErrorString() const final {
    return "ScheduleError: The inner loop is not the only child of outer loop";
  }

  ffi::String DetailRenderTemplate() const final {
    return "The loops can't be fused because the inner loop {1} is not the only child of outer "
           "loop {0}.";
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {outer_, inner_}; }

  IRModule mod_;
  For outer_;
  For inner_;
};

class NotSingleInferFactorError : public ScheduleErrorContextObj {
 public:
  explicit NotSingleInferFactorError(IRModule mod) : mod_(mod) {}

  ffi::String FastErrorString() const final {
    return "ScheduleError: only one factor can be specified as -1 or none";
  }

  ffi::String DetailRenderTemplate() const final {
    return "Only one factor can be specified as -1 or none";
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {}; }

  IRModule mod_;
};

class WrongFactorProductError : public ScheduleErrorContextObj {
 public:
  explicit WrongFactorProductError(IRModule mod, For loop) : mod_(mod), loop_(std::move(loop)) {}

  ffi::String FastErrorString() const final {
    return "ScheduleError: The product of factors is not larger than or equal to the extent of "
           "loop";
  }

  ffi::String DetailRenderTemplate() const final {
    return "The product of factors is not larger than or equal to the extent of loop {0}";
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {loop_}; }

  IRModule mod_;
  For loop_;
};

class LoopMultiAppearanceError : public ScheduleErrorContextObj {
 public:
  explicit LoopMultiAppearanceError(IRModule mod, For loop) : mod_(mod), loop_(std::move(loop)) {}

  ffi::String FastErrorString() const final {
    return "ScheduleError: Some loop appears in the input array for multiple times.";
  }

  ffi::String DetailRenderTemplate() const final {
    return "Loop {0} appears in the input array for multiple times.";
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {loop_}; }

  IRModule mod_;
  For loop_;
};

class LoopsNotAChainError : public ScheduleErrorContextObj {
 public:
  enum class ProblemKind { kNotUnderAScope, kHaveNonSingleBranchStmt };

  explicit LoopsNotAChainError(IRModule mod, ffi::Optional<Stmt> problematic_loop, ProblemKind kind)
      : mod_(mod), problematic_loop_(std::move(problematic_loop)), kind_(kind) {}

  ffi::String FastErrorString() const final {
    return "ScheduleError: the loops are not in a chain";
  }

  ffi::String DetailRenderTemplate() const final {
    std::stringstream ss;
    ss << "The loops are not in a chain because";
    if (kind_ == ProblemKind::kNotUnderAScope) {
      ss << " they are not under the same scope.";
    } else {
      ss << " there is a non-single-branch stmt in between. Problematic stmt: {0}";
    }
    return ss.str();
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final {
    if (kind_ == ProblemKind::kNotUnderAScope) {
      return {};
    } else {
      TVM_FFI_ICHECK(problematic_loop_.has_value());
      return {problematic_loop_.value()};
    }
  }

  IRModule mod_;
  ffi::Optional<Stmt> problematic_loop_;
  ProblemKind kind_;
};

class DependentLoopError : public ScheduleErrorContextObj {
 public:
  enum class PrimitiveKind { kFuse, kReorder };
  explicit DependentLoopError(IRModule mod, For loop, ffi::String inner_var, PrimitiveKind kind)
      : mod_(mod), loop_(std::move(loop)), inner_var_(std::move(inner_var)), kind_(kind) {}

  ffi::String FastErrorString() const final {
    if (kind_ == PrimitiveKind::kReorder) {
      return "ScheduleError: An outer loop's `min` or `extent` is dependent on an inner loop "
             "in the new order";
    } else {
      return "ScheduleError: A loop's `extent` is dependent on another loop";
    }
  }

  ffi::String DetailRenderTemplate() const final {
    if (kind_ == PrimitiveKind::kReorder) {
      return "Outer Loop {0}'s `min` or `extent` is dependent on an inner loop " + inner_var_ +
             " in the new order";
    } else {
      return "A loop {0}'s `extent` is dependent on another loop " + inner_var_;
    }
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {loop_}; }

  IRModule mod_;
  For loop_;
  ffi::String inner_var_;
  PrimitiveKind kind_;
};

ffi::Array<StmtSRef> Split(ScheduleState self, const StmtSRef& loop_sref,
                           const ffi::Array<PrimExpr>& factors, bool preserve_unit_iters,
                           bool disable_predication) {
  // Invariance
  // - The total repeat number has not changed for each direct child block with updating predicate.
  // - The execution order has not changed. (The block executes with the same args and the same
  // order with before.
  // Step 1. Check correctness
  const ForObj* loop = TVM_SREF_TO_FOR(loop_sref);
  if (!loop->annotations.empty()) {
    throw MakeScheduleError<HasAnnotationOrThreadBindingError>(self->mod, ffi::GetRef<For>(loop));
  }
  // Currently, loops not starting with 0 are not supported
  sym::Analyzer analyzer;
  CheckLoopStartsWithZero(self, loop_sref, analyzer.get());

  // Find the most common dtype
  PrimType dtype = PrimType::Int(32);
  {
    int bits = loop->var.ty().bits();
    for (const PrimExpr& factor : factors) {
      bits = std::max(bits, factor.ty().bits());
    }
    dtype = PrimType::Int(bits);
  }
  int n = factors.size();
  PrimExpr substitute_value = IntImm(dtype, 0);
  std::vector<Var> new_loop_vars;
  new_loop_vars.reserve(n);
  for (int i = 0; i < n; i++) {
    const PrimExpr& factor = factors[i];
    Var var = loop->var.CopyWithSuffix("_" + std::to_string(i)).CopyWithDType(dtype);
    substitute_value = substitute_value * factor + var.as_or_throw<PrimExpr>();
    {
      auto context = analyzer->GetConstraintContext();
      context->Append(sym::RangeConstraint(
          var, Range::FromMinExtent(IntImm(dtype, 0), tvm::prim::cast(factor, dtype))));
      analyzer->SyncConstraintContext(context);
    }
    new_loop_vars.emplace_back(std::move(var));
  }
  ffi::Map<SBlock, SBlock> opaque_block_reuse;
  Stmt new_stmt = loop->body;
  new_stmt = ffi::make_object<SubstituteVarAndCollectOpaqueBlock>(
                 ffi::Map<Var, PrimExpr>{{loop->var, substitute_value}}, &opaque_block_reuse)
                 ->Mutate(new_stmt, InplaceMode::kAllow)
                 .ValueOrUnchanged(std::move(new_stmt));
  // Step 3. Update predicate to guard the loop
  PrimExpr predicate = substitute_value < Range::GetExtentFromBeginEnd(loop->begin, loop->end);
  if (!disable_predication && !analyzer->CanProve(predicate, sym::ProofStrength::kSymbolicBound)) {
    new_stmt = ffi::make_object<BlockPredicateAppender>(/*predicate=*/predicate)
                   ->Mutate(new_stmt, InplaceMode::kAllow)
                   .ValueOrUnchanged(std::move(new_stmt));
  }
  // Step 4. Generate nested loops to replace the original loop and simplify the binding
  for (int i = n - 1; i >= 0; i--) {
    new_stmt =
        For(new_loop_vars[i].as_or_throw<PrimVar>(), 0, factors[i],
            IntImm(new_loop_vars[i].as_or_throw<PrimVar>().ty(), 1), ForKind::kDefault, new_stmt);
  }
  new_stmt = IterMapSimplifyBlockBinding::SimplifyBindings(std::move(new_stmt), GetLoops(loop_sref),
                                                           opaque_block_reuse.CopyOnWrite(),
                                                           preserve_unit_iters);
  self->Replace(loop_sref, new_stmt, opaque_block_reuse);
  ffi::Array<StmtSRef> result_srefs;
  result_srefs.reserve(n);
  for (int i = 0; i < n; i++) {
    result_srefs.push_back(self->stmt2ref.at(new_stmt.get()));
    if (i + 1 < n) {
      const ForObj* outer_loop = TVM_TYPE_AS(new_stmt, ForObj);
      TVM_FFI_ICHECK_EQ(outer_loop->body->size(), 1);
      new_stmt = outer_loop->body->seq[0];
    }
  }
  return result_srefs;
}

class TensorIndicesMapExtractor : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;

  explicit TensorIndicesMapExtractor(Var loop_var) : loop_var_(loop_var) {}

  static ffi::Map<TensorVar, ffi::Array<Var>> Extract(Var loop_var, SBlock& block) {
    auto extractor = ffi::make_object<TensorIndicesMapExtractor>(loop_var);
    extractor->Visit(std::move(block->body));
    return extractor->tensor_indices_map;
  }

 private:
  ffi::Optional<VisitInterrupt> Visit_(const TensorStoreObj* store) final {
    ffi::Array<Var> indices;
    bool check_ = false;
    for (size_t i = 0; i < store->indices.size(); i++) {
      auto var = store->indices[i].as<PrimVar>();
      if (!var.has_value()) {
        check_ = true;
        break;
      }
      indices.push_back(var.value());
    }
    if (tensor_indices_map.find(store->dest.as_or_throw<TensorVar>()) == tensor_indices_map.end() &&
        !check_)
      tensor_indices_map.Set(store->dest.as_or_throw<TensorVar>(), indices);
    return StmtExprVisitor::Visit_(store);
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadObj* load) final {
    ffi::Array<Var> indices;
    bool check_ = false;
    for (size_t i = 0; i < load->indices.size(); i++) {
      auto var = load->indices[i].as<PrimVar>();
      if (!var.has_value()) {
        check_ = true;
        break;
      }
      indices.push_back(var.value());
    }
    TensorVar tensor = load->source.as_or_throw<tvm::tirx::TensorVar>();
    if (tensor_indices_map.find(tensor) == tensor_indices_map.end() && !check_) {
      tensor_indices_map.Set(tensor, indices);
    }
    return StmtExprVisitor::Visit_(load);
  }

  Var loop_var_;
  ffi::Map<TensorVar, ffi::Array<Var>> tensor_indices_map;
};

ffi::Array<TensorRegion> MutateTensorRegion(ffi::Map<TensorVar, ffi::Array<Var>> tensor_indices_map,
                                            ffi::Map<Var, Range> index_range_map,
                                            ffi::Array<TensorRegion> region_arr) {
  // Update the region with new Ranges and return new TensorRegion
  ffi::Array<TensorRegion> new_region_arr =
      region_arr.Map([&tensor_indices_map, &index_range_map](const TensorRegion& region) {
        TensorRegion new_region = region;
        auto it = tensor_indices_map.find(new_region->source.as_or_throw<TensorVar>());
        if (it == tensor_indices_map.end()) return new_region;

        ffi::Array<Var> old_indices = (*it).second;
        ffi::Array<Range> new_ranges;
        for (size_t i = 0; i < old_indices.size(); i++) {
          new_ranges.push_back(index_range_map[old_indices[i]]);
        }
        new_region.CopyOnWrite()->region = std::move(new_ranges);
        return new_region;
      });
  return new_region_arr;
}

class BlockMutator : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  explicit BlockMutator(Var new_loop_var, PrimExpr min, PrimExpr extent)
      : new_loop_var_(new_loop_var), min_(min), extent_(extent) {}

 private:
  UnchangedOr<Stmt> Mutate_(const SBlockObj* _op, InplaceMode inplace_mode) final {
    SBlock new_block = StmtExprMutator::Mutate_(_op, inplace_mode)
                           .ValueOrUnchanged(ffi::GetRef<Stmt>(_op))
                           .as_or_throw<SBlock>();

    // If iter_vars.size() is 0, then the block most probably be an Opaque block
    if (new_block->iter_vars.size() == 0 || inner_iter_var_index == -1) {
      new_block.CopyOnWrite()->name_hint =
          new_block.CopyOnWrite()->name_hint + "_" + new_loop_var_->name;
      return new_block;
    }

    Var iter_var_ = new_block->iter_vars[inner_iter_var_index]->var;
    inner_iter_var_index = -1;
    // As we are working on cloned block, we need to create new instances of iter_var
    ffi::Array<IterVar> new_iter_vars =
        new_block->iter_vars.Map([this, &iter_var_](const IterVar& iter) {
          auto dtype = iter->var.ty();
          // Create new Var instance for each IterVar
          Var new_var = Var(iter->var->name, iter->var.ty());
          IterVar new_iter = iter;
          new_iter.CopyOnWrite()->var = new_var.as_or_throw<PrimVar>();
          // Change the domain of IterVar corresponding to partitioned loop_var
          if (iter_var_.same_as(iter->var)) {
            new_iter.CopyOnWrite()->dom =
                Range(tvm::prim::cast(min_, dtype), tvm::prim::cast(extent_, dtype));
          }
          return new_iter;
        });

    // Update the IterVars of new_block
    if (!new_block->iter_vars.same_as(new_iter_vars)) {
      new_block.CopyOnWrite()->iter_vars = std::move(new_iter_vars);
      new_block.CopyOnWrite()->name_hint =
          new_block.CopyOnWrite()->name_hint + "_" + new_loop_var_->name;
    }

    // Get the (iter_var, new Range) map
    ffi::Map<Var, Range> index_range_map;
    for (size_t i = 0; i < new_block->iter_vars.size(); i++) {
      IterVar iter = new_block->iter_vars[i];
      // The body still uses the original Vars until substitution below. Their
      // identities distinguish axes even when producer names are empty or equal.
      index_range_map.Set(_op->iter_vars[i]->var, iter->dom.value());
    }

    // Get the (TensorVar, indices) map
    ffi::Map<TensorVar, ffi::Array<Var>> tensor_indices_map =
        TensorIndicesMapExtractor::Extract(new_loop_var_, new_block);
    ffi::Array<TensorRegion> new_writes =
        MutateTensorRegion(tensor_indices_map, index_range_map, new_block->writes);
    if (!new_block->writes.same_as(new_writes)) {
      // Update the writes with new_writes
      new_block.CopyOnWrite()->writes = std::move(new_writes);
    }
    ffi::Array<TensorRegion> new_reads =
        MutateTensorRegion(tensor_indices_map, index_range_map, new_block->reads);
    if (!new_block->reads.same_as(new_reads)) {
      // Update the reads with new_reads
      new_block.CopyOnWrite()->reads = std::move(new_reads);
    }

    ffi::Map<Var, Var> var_map;
    for (size_t i = 0; i < new_block->iter_vars.size(); i++) {
      var_map.Set(_op->iter_vars[i]->var, new_block->iter_vars[i]->var);
    }

    // Update all instances of old iter_vars in the block with new iter_vars
    auto f_substitute = [&var_map](
                            const Var& var,
                            TVMFFIDefRegionKind kind) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
      if (kind != kTVMFFIDefRegionKindNone) return ffi::Unchanged();
      if (auto repl = var_map.Get(var)) return ffi::Any(*std::move(repl));
      return ffi::Unchanged();
    };
    auto block_stmt = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(new_block, f_substitute)
                          .as_or_throw<SBlock>();
    return block_stmt;
  }

  UnchangedOr<Stmt> Mutate_(const SBlockRealizeObj* realize, InplaceMode inplace_mode) final {
    ffi::Array<PrimExpr> iter_values = realize->iter_values;
    for (size_t i = 0; i < iter_values.size(); i++) {
      if (new_loop_var_.same_as(iter_values[i])) {
        // Get the iter_var index corresponding to loop_var iter_value index
        inner_iter_var_index = i;
        break;
      }
    }
    SBlockRealize stmt = StmtExprMutator::Mutate_(realize, inplace_mode)
                             .ValueOrUnchanged(ffi::GetRef<Stmt>(realize))
                             .as_or_throw<SBlockRealize>();
    return stmt;
  }

  UnchangedOr<Stmt> Mutate_(const ForObj* op, InplaceMode inplace_mode) final {
    For res = StmtExprMutator::Mutate_(op, inplace_mode)
                  .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                  .as_or_throw<For>();
    Var new_var = Var(op->var->name, op->var.ty());

    if (!op->var.same_as(new_var)) {
      // If the partioned loop contains nested for loop, then create new iteration variable instance
      auto f_substitute = [old_var = op->var, &new_var](const Var& var, TVMFFIDefRegionKind kind)
          -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
        if (kind != kTVMFFIDefRegionKindNone) return ffi::Unchanged();
        if (var.same_as(old_var)) return ffi::Any(new_var);
        return ffi::Unchanged();
      };
      res.CopyOnWrite()->body =
          ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(res->body, f_substitute)
              .as_or_throw<Stmt>();
      res.CopyOnWrite()->var = new_var.as_or_throw<PrimVar>();
    }
    return res;
  }

  Var new_loop_var_;
  PrimExpr min_, extent_;
  int inner_iter_var_index = -1;
};

const ffi::String get_sblock_name(SeqStmt loop_body) {
  TVM_FFI_ICHECK_EQ(loop_body->size(), 1);
  const Stmt& child = loop_body->seq[0];
  if (const auto* block_realize = child.as<SBlockRealizeObj>()) {
    return block_realize->block->name_hint;
  }
  return get_sblock_name(child.as_or_throw<For>()->body);
}

ffi::Array<StmtSRef> LoopPartition(ScheduleState self, const StmtSRef& loop_sref,
                                   const ffi::Array<PrimExpr>& factors, bool preserve_unit_iters) {
  const ForObj* loop = TVM_SREF_TO_FOR(loop_sref);
  if (!loop->annotations.empty()) {
    throw MakeScheduleError<HasAnnotationOrThreadBindingError>(self->mod, ffi::GetRef<For>(loop));
  }

  sym::Analyzer analyzer;
  // Find the most common dtype
  PrimType dtype = PrimType::Int(32);
  {
    int bits = loop->var.ty().bits();
    for (const PrimExpr& factor : factors) {
      bits = std::max(bits, factor.ty().bits());
    }
    dtype = PrimType::Int(bits);
  }

  ffi::String block_name = get_sblock_name(loop->body) + "_" + loop->var->name;
  int n = factors.size();
  PrimExpr min_value = loop->begin;

  ffi::Array<Stmt> block_partitions;
  block_partitions.reserve(n);

  // Iterate over each pair of factors and create partition
  for (int i = 0; i < n; i++) {
    PrimExpr extent_value = analyzer->Simplify(factors[i]);
    Var new_loop_var = loop->var.CopyWithSuffix(std::to_string(i)).CopyWithDType(dtype);
    // The widened dtype is intentional: partition factors determine the common loop dtype.
    auto f_substitute = [old_var = loop->var, &new_loop_var](
                            const Var& var,
                            TVMFFIDefRegionKind kind) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
      if (kind != kTVMFFIDefRegionKindNone) return ffi::Unchanged();
      if (var.same_as(old_var)) return ffi::Any(new_loop_var);
      return ffi::Unchanged();
    };
    Stmt loop_body =
        ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(loop->body, f_substitute).as_or_throw<Stmt>();

    // Create new block with new reference to each variable/stmt/expr in the existing block
    loop_body = ffi::make_object<BlockMutator>(new_loop_var, min_value, extent_value)
                    ->Mutate(loop_body, InplaceMode::kAllow)
                    .ValueOrUnchanged(std::move(loop_body));
    // Create new for loop with appropriate range
    auto for_node =
        For(new_loop_var.as_or_throw<PrimVar>(), min_value, extent_value,
            IntImm(new_loop_var.as_or_throw<PrimVar>().ty(), 1), ForKind::kDefault, loop_body);

    const auto& partition_block_name = block_name + std::to_string(i) + "_partition";
    // Create partition_block for the partitioned for loop
    SBlockRealize partition_block({}, extent_value > 0,
                                  SBlock({}, {}, {}, partition_block_name, for_node));
    block_partitions.push_back(partition_block);

    min_value = extent_value;
  }

  // Create common block with all the partitioned blocks as its children blocks
  SBlockRealize common({}, IntImm::Bool(true),
                       SBlock({}, {}, {}, block_name + "_common", SeqStmt(block_partitions)));

  // Replace existing loop with the newly created common block
  self->Replace(loop_sref, common, {});
  StmtSRef scope_sref = self->stmt2ref.at(common->block.get());
  StmtSRef scope_root = GetScopeRoot(self, scope_sref, /*require_stage_pipeline=*/false);
  bool scope_block_affine_binding = self->IsAffineBlockBinding(scope_root);
  // Update the SRefTree for the newly created common block
  self->UpdateScopeSBlockInfo(GetSBlockRealize(self, scope_root));
  self->block_info[scope_root].affine_binding = scope_block_affine_binding;

  // Collect the SRef for each partitioned loop and return
  ffi::Array<StmtSRef> partition_srefs;
  partition_srefs.reserve(n);
  for (int i = 0; i < n; i++) {
    const SeqStmt& body = block_partitions[i].as<SBlockRealizeObj>()->block->body;
    TVM_FFI_ICHECK_EQ(body->size(), 1);
    StmtSRef partition_loop_sref = self->stmt2ref.at(body->seq[0].get());
    partition_srefs.push_back(partition_loop_sref);
  }
  return partition_srefs;
}

class LoopReconstructor : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) override {
    if (value.as<ExprObj>()) return ffi::Unchanged();
    return StmtExprMutator::Mutate(value, inplace_mode);
  }

  explicit LoopReconstructor(SBlock scope_root, const std::vector<std::vector<For>>& loops)
      : scope_root_(scope_root), loops_(loops) {}

  /*!
   * \brief Create the new nest loops induced by the given loops
   */
  void MakeNewLoop() {
    ffi::Array<Var> new_loop_vars;
    ffi::Array<PrimExpr> new_loop_extents;
    ffi::Array<Stmt> new_stmts;
    for (size_t i = 0; i < loops_.size(); i++) {
      ffi::Map<Var, PrimExpr> var_map;
      for (size_t j = 0; j < loops_[i].size(); j++) {
        if (i == 0) {
          Var merged_var = loops_[i][j]->var.CopyWithSuffix("_m");
          new_loop_vars.push_back(merged_var);
          new_loop_extents.push_back(
              Range::GetExtentFromBeginEnd(loops_[i][j]->begin, loops_[i][j]->end));
        }
        var_map.Set(loops_[i][j]->var, new_loop_vars[j].as_or_throw<PrimExpr>());
      }
      auto f_substitute =
          [&var_map](const Var& var,
                     TVMFFIDefRegionKind kind) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
        if (kind != kTVMFFIDefRegionKindNone) return ffi::Unchanged();
        if (auto repl = var_map.Get(var)) return ffi::Any(*std::move(repl));
        return ffi::Unchanged();
      };
      auto new_stmt =
          ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(loops_[i][0]->body, f_substitute)
              .as_or_throw<Stmt>();
      new_stmts.push_back(new_stmt);
      this->need_remove_loop_.push_back(loops_[i].back());
    }
    auto new_loop =
        For(new_loop_vars[0].as_or_throw<PrimVar>(), IntImm::Int32(0), new_loop_extents[0],
            IntImm(new_loop_vars[0].as_or_throw<PrimVar>().ty(), 1), ForKind::kDefault,
            SeqStmt(std::move(new_stmts)));
    this->new_inner_loop_ = new_loop;
    for (size_t i = 1; i < new_loop_vars.size(); ++i) {
      const Var& loop_var = new_loop_vars[i];
      const PrimExpr& loop_extent = new_loop_extents[i];
      new_loop = For(loop_var.as_or_throw<PrimVar>(), IntImm::Int32(0), loop_extent,
                     IntImm(loop_var.as_or_throw<PrimVar>().ty(), 1), ForKind::kDefault, new_loop);
    }
    this->new_outer_loop_ = new_loop;
  }

 private:
  UnchangedOr<Stmt> Mutate_(const SBlockObj* block, InplaceMode inplace_mode) final {
    if (block != scope_root_.get()) {
      return ffi::Unchanged();
    }
    return StmtExprMutator::Mutate_(block, inplace_mode);
  }

  UnchangedOr<Stmt> Mutate_(const ForObj* loop, InplaceMode inplace_mode) final {
    if (ffi::GetRef<For>(loop) == need_remove_loop_.back()) {
      return new_outer_loop_.value();
    } else if (std::count(need_remove_loop_.begin(), need_remove_loop_.end(),
                          ffi::GetRef<For>(loop))) {
      return Evaluate(0);
    }
    return StmtExprMutator::Mutate_(loop, inplace_mode);
  }

 public:
  /*! \brief The root block of the block scope */
  SBlock scope_root_;
  /*! \brief The given loops to be merge */
  const std::vector<std::vector<For>>& loops_;
  /*! \brief The outermost new loop to replace the original loop */
  ffi::Optional<For> new_outer_loop_;
  /*! \brief The innermost new loop to replace the original loop */
  ffi::Optional<For> new_inner_loop_;
  /*! \brief The loops to be removed */
  std::vector<For> need_remove_loop_;
};

StmtSRef Merge(ScheduleState self, const ffi::Array<StmtSRef>& loop_srefs) {
  // Invariance
  // - The total repeat number has not changed for each direct child block.
  // - The execution order has not changed. (The block executes with the same
  //   args and the same order with before.)
  sym::Analyzer analyzer;
  ffi::Optional<StmtSRef> scope_root_sref;
  StmtSRef lca = GetSRefLowestCommonAncestor(loop_srefs);
  std::vector<std::vector<For>> lca_nest_loops;
  // Step 1. check correctness
  std::vector<For> nest_loop_loops;
  std::vector<PrimExpr> nest_loop_extents;
  for (size_t i = 0; i < loop_srefs.size(); i++) {
    const StmtSRef& sref = loop_srefs[i];
    auto scope_root_sref_ = GetScopeRoot(self, sref, /*require_stage_pipeline=*/false);
    std::vector<PrimExpr> nest_loop_i_extents;
    std::vector<For> nest_loop_i_loops;
    for (auto p = sref.get(); p != lca.get(); p = p->parent) {
      if (auto loop = p->StmtAs<ForObj>()) {
        if (!loop->annotations.empty()) {
          throw MakeScheduleError<HasAnnotationOrThreadBindingError>(self->mod,
                                                                     ffi::GetRef<For>(loop));
        }
        CheckLoopStartsWithZero(self, ffi::GetRef<StmtSRef>(p), analyzer.get());
        nest_loop_i_loops.push_back(ffi::GetRef<For>(loop));
        nest_loop_i_extents.push_back(Range::GetExtentFromBeginEnd(loop->begin, loop->end));
      }
    }
    lca_nest_loops.push_back(nest_loop_i_loops);
    const ForObj* outer_loop = nullptr;
    for (auto iter = nest_loop_i_loops.rbegin(); iter != nest_loop_i_loops.rend(); ++iter) {
      if (outer_loop &&
          (outer_loop->body->size() != 1 || !outer_loop->body->seq[0].same_as(*iter))) {
        throw MakeScheduleError<NotOnlyChildError>(self->mod, ffi::GetRef<For>(outer_loop), *iter);
      }
      outer_loop = (*iter).get();
    }
    if (i == 0) {
      scope_root_sref = scope_root_sref_;
      nest_loop_loops = nest_loop_i_loops;
      nest_loop_extents = nest_loop_i_extents;
    } else {
      if (scope_root_sref_.get() != scope_root_sref.value().get()) {
        TVM_FFI_THROW(ScheduleError) << "Expected the loops to be under the same block scope.";
        throw;
      }
      if (nest_loop_i_extents.size() != nest_loop_extents.size()) {
        TVM_FFI_THROW(ScheduleError) << "Merge loop's nesting depth must be same, but not.";
        throw;
      } else {
        for (size_t j = 0; j < nest_loop_i_extents.size(); j++) {
          if (!analyzer->CanProveEqual(nest_loop_i_extents[j], nest_loop_extents[j])) {
            TVM_FFI_THROW(ScheduleError) << "Merge loop's `extent` must be same, but not."
                                         << " extent=[" << j << "," << nest_loop_extents[j] << ","
                                         << nest_loop_i_extents[j] << "]";
            throw;
          }
        }
      }
    }
  }
  // Step 2. Create merged loops and replace the original loops
  SBlock scope_root = ffi::GetRef<SBlock>(scope_root_sref.value()->StmtAs<SBlockObj>());
  auto reconstructor = ffi::make_object<LoopReconstructor>(scope_root, lca_nest_loops);
  reconstructor->MakeNewLoop();
  SBlock new_scope_root =
      reconstructor->Mutate(scope_root).ValueOrUnchanged(scope_root).as_or_throw<SBlock>();
  // Step 3. Do the actual replacement
  self->Replace(scope_root_sref.value(), new_scope_root, {{scope_root, new_scope_root}});
  return self->stmt2ref.at(reconstructor->new_inner_loop_.value().get());
}

StmtSRef Fuse(ScheduleState self, const ffi::Array<StmtSRef>& loop_srefs,
              bool preserve_unit_iters) {
  // Invariance
  // - The total repeat number has not changed for each direct child block.
  // - The execution order has not changed. (The block executes with the same
  //   args and the same order with before.)
  std::vector<const ForObj*> loops;
  loops.reserve(loop_srefs.size());
  ffi::Optional<StmtSRef> outer_loop_sref;
  const ForObj* outer_loop = nullptr;
  sym::Analyzer analyzer;
  std::unordered_set<const VarObj*> outer_loop_vars;
  // Step 1. check correctness
  for (const StmtSRef& sref : loop_srefs) {
    const ForObj* loop = TVM_SREF_TO_FOR(sref);
    if (!loop->annotations.empty()) {
      throw MakeScheduleError<HasAnnotationOrThreadBindingError>(self->mod, ffi::GetRef<For>(loop));
    }
    if (outer_loop_sref.has_value()) {
      if (sref->parent != outer_loop_sref.value().get()) {
        throw MakeScheduleError<OuterNotInnerParent>(self->mod, ffi::GetRef<For>(outer_loop),
                                                     ffi::GetRef<For>(loop));
      }
      if (outer_loop->body->size() != 1 ||
          !outer_loop->body->seq[0].same_as(ffi::GetRef<For>(loop))) {
        throw MakeScheduleError<NotOnlyChildError>(self->mod, ffi::GetRef<For>(outer_loop),
                                                   ffi::GetRef<For>(loop));
      }
    }
    outer_loop_sref = sref;
    outer_loop = loop;
    CheckLoopStartsWithZero(self, sref, analyzer.get());
    auto walkfn = [&outer_loop_vars](const Var& var) -> ffi::Expected<ffi::WalkResult> {
      return outer_loop_vars.count(var.get()) ? ffi::WalkResult::Interrupt(ffi::VisitInterrupt(var))
                                              : ffi::WalkResult::Advance();
    };
    auto result = ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
        Range::GetExtentFromBeginEnd(loop->begin, loop->end), walkfn);
    if (result.has_value()) {
      Var used_var = result.value()->value.cast<Var>();
      throw MakeScheduleError<DependentLoopError>(self->mod, ffi::GetRef<For>(loop), used_var->name,
                                                  DependentLoopError::PrimitiveKind::kFuse);
    }
    outer_loop_vars.insert(loop->var.get());
    loops.push_back(loop);
  }
  // Step 2. Create fused loop var and replace the original loop vars
  std::string suffix;
  int n = loops.size();
  int bits = loops[0]->var.ty().bits();
  for (int i = 1; i < n; i++) {
    suffix += "_" + loops[i]->var->name;
    bits = std::max(bits, loops[i]->var.ty().bits());
  }
  suffix += "_fused";

  Var fused_var = loops[0]->var.CopyWithSuffix(suffix).CopyWithDType(PrimType::Int(bits));
  ffi::Array<PrimExpr> substitute_value;
  substitute_value.resize(loops.size());
  PrimExpr lower = 1;
  for (int i = static_cast<int>(loops.size()) - 1; i > 0; i--) {
    PrimExpr next_lower = analyzer->canonical_simplify(
        Range::GetExtentFromBeginEnd(loops[i]->begin, loops[i]->end) * lower);
    substitute_value.Set(
        i, IsOne(Range::GetExtentFromBeginEnd(loops[i]->begin, loops[i]->end))
               ? PrimExpr(0)
               : floordiv(floormod(fused_var.as_or_throw<PrimExpr>(), next_lower), lower));
    lower = next_lower;
  }
  substitute_value.Set(0, IsOne(Range::GetExtentFromBeginEnd(loops[0]->begin, loops[0]->end))
                              ? PrimExpr(0)
                              : floordiv(fused_var.as_or_throw<PrimExpr>(), lower));
  Stmt new_stmt = loops.back()->body;
  ffi::Map<SBlock, SBlock> opaque_block_reuse;
  ffi::Map<Var, PrimExpr> substitutions;
  for (int i = 0; i < n; ++i) substitutions.Set(loops[i]->var, substitute_value[i]);
  new_stmt =
      ffi::make_object<SubstituteVarAndCollectOpaqueBlock>(substitutions, &opaque_block_reuse)
          ->Mutate(new_stmt, InplaceMode::kAllow)
          .ValueOrUnchanged(std::move(new_stmt));
  // Step 3. Generate a loop to replace the original loops
  PrimExpr fused_extent = 1;
  for (int i = 0; i < n; i++) {
    fused_extent *= Range::GetExtentFromBeginEnd(loops[i]->begin, loops[i]->end);
  }
  fused_extent = analyzer->Simplify(fused_extent);
  new_stmt = For(fused_var.as_or_throw<PrimVar>(), 0, fused_extent,
                 IntImm(fused_var.as_or_throw<PrimVar>().ty(), 1), ForKind::kDefault, new_stmt);
  new_stmt = IterMapSimplifyBlockBinding::SimplifyBindings(
      std::move(new_stmt), GetLoops(loop_srefs[0]), opaque_block_reuse.CopyOnWrite(),
      preserve_unit_iters);
  self->Replace(loop_srefs[0], new_stmt, opaque_block_reuse);
  return self->stmt2ref.at(new_stmt.get());
}

/*!
 * \brief Collect an array of loop srefs into a set
 * \param self The schedule state
 * \param ordered_loop_srefs The array of loop srefs
 * \return A set containing all loops in the array
 * \throws ScheduleError If there are duplicate loops in the array
 */
std::unordered_set<const StmtSRefObj*> CollectLoopsIntoSet(
    const ScheduleState& self, const ffi::Array<StmtSRef>& ordered_loop_srefs) {
  std::unordered_set<const StmtSRefObj*> loop_srefs;
  loop_srefs.reserve(ordered_loop_srefs.size());
  for (const StmtSRef& loop_sref : ordered_loop_srefs) {
    auto inserted = loop_srefs.insert(loop_sref.get());
    if (!inserted.second) {
      const ForObj* loop = TVM_SREF_TO_FOR(loop_sref);
      throw MakeScheduleError<LoopMultiAppearanceError>(self->mod, ffi::GetRef<For>(loop));
    }
  }
  return loop_srefs;
}

/*!
 * \brief Get the top and bottom boundary of reorder range (which should be a chain)
 * \param self The schedule state
 * \param loop_srefs The set containing the srefs to the loops to be reordered
 * \return A pair containing the top and bottom boundary of the reorder range
 * \throws ScheduleError If the loops to be reordered is not in a chain
 */
std::pair<const StmtSRefObj*, const StmtSRefObj*> GetBoundaryOfReorderRange(
    const ScheduleState& self, const std::unordered_set<const StmtSRefObj*>& loop_srefs) {
  const StmtSRefObj* top = nullptr;
  const StmtSRefObj* bottom = *loop_srefs.begin();
  std::unordered_set<const StmtSRefObj*> visited;
  bool scope_block_visited = false;
  bool first_traversal = true;
  for (const StmtSRefObj* loop_sref : loop_srefs) {
    if (visited.count(loop_sref)) {
      continue;
    }
    for (const StmtSRefObj* v = loop_sref;; v = v->parent) {
      // Case 1. If `v` corresponds to a block, stop traversal.
      if (v->stmt->IsInstance<SBlockObj>()) {
        if (scope_block_visited) {
          throw MakeScheduleError<LoopsNotAChainError>(
              self->mod, std::nullopt, LoopsNotAChainError::ProblemKind::kNotUnderAScope);
        }
        scope_block_visited = true;
        break;
      }
      // Case 2. If `v` corresponds to a previously-visited loop, stop traversal and update
      // `bottom`.
      if (visited.count(v)) {
        if (v != bottom) {
          throw MakeScheduleError<LoopsNotAChainError>(
              self->mod, ffi::GetRef<Stmt>(v->stmt),
              LoopsNotAChainError::ProblemKind::kHaveNonSingleBranchStmt);
        }
        bottom = loop_sref;
        break;
      }
      // Case 3. Add `v` into `visited`
      visited.insert(v);
      // If it's the first traversal and the loop corresponding to `v` is in the input array,
      // update `top`.
      if (first_traversal && loop_srefs.count(v)) {
        top = v;
      }
    }
    first_traversal = false;
  }
  return std::make_pair(top, bottom);
}

/*!
 * \brief Get all the loops in the reorder range
 * \param self The schedule state
 * \param top The top boundary of the reorder range
 * \param bottom The bottom boundary of the reorder range
 * \return An array containing all the loops in the reorder range
 * \throws ScheduleError If some loop in the reorder range is not single-branch
 */
std::vector<const StmtSRefObj*> GetLoopsInReorderRange(const ScheduleState& self,
                                                       const StmtSRefObj* top,
                                                       const StmtSRefObj* bottom) {
  std::vector<const StmtSRefObj*> chain;
  for (const StmtSRefObj* loop_sref = bottom; loop_sref != top;) {
    const StmtSRefObj* parent_loop_sref = loop_sref->parent;
    const ForObj* outer = parent_loop_sref->StmtAs<ForObj>();
    const ForObj* inner = loop_sref->StmtAs<ForObj>();
    TVM_FFI_ICHECK(outer != nullptr && inner != nullptr);
    if (outer->body->size() != 1 || outer->body->seq[0].get() != inner) {
      throw MakeScheduleError<LoopsNotAChainError>(
          self->mod, ffi::GetRef<For>(outer),
          LoopsNotAChainError::ProblemKind::kHaveNonSingleBranchStmt);
    }
    chain.push_back(loop_sref);
    loop_sref = parent_loop_sref;
  }
  chain.push_back(top);
  return chain;
}

/*!
 * \brief Construct a loop chain in the new order
 * \param self The schedule state
 * \param chain The loops in the reorder range
 * \param ordered_loop_srefs The loop srefs to be reordered
 * \param loop_srefs The set containing loop srefs to be reordered
 * \return The new loop chain
 * \throws ScheduleError If the domain of an outer loop depends on any of the inner loops after
 * reordering
 */
For ConstructNewLoopChain(const ScheduleState& self, std::vector<const StmtSRefObj*> chain,
                          const ffi::Array<StmtSRef>& ordered_loop_srefs,
                          const std::unordered_set<const StmtSRefObj*>& loop_srefs) {
  std::unordered_set<const VarObj*> inner_vars;
  inner_vars.reserve(chain.size());
  ffi::Optional<For> new_loop;
  int index = static_cast<int>(ordered_loop_srefs.size()) - 1;
  for (const StmtSRefObj* loop_sref : chain) {
    const ForObj* copy = nullptr;
    if (loop_srefs.count(loop_sref)) {
      copy = ordered_loop_srefs[index]->StmtAs<ForObj>();
      --index;
    } else {
      copy = loop_sref->StmtAs<ForObj>();
    }
    TVM_FFI_ICHECK(copy != nullptr);
    ffi::ObjectPtr<ForObj> n = ffi::make_object<ForObj>(*copy);
    if (new_loop.has_value()) {
      n->body = new_loop.value();
    } else {
      n->body = loop_sref->StmtAs<ForObj>()->body;
    }
    auto walkfn = [&inner_vars](const Var& var) -> ffi::Expected<ffi::WalkResult> {
      return inner_vars.count(var.get()) ? ffi::WalkResult::Interrupt(ffi::VisitInterrupt(var))
                                         : ffi::WalkResult::Advance();
    };
    auto result = ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(copy->begin, walkfn);
    if (!result.has_value()) {
      result = ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
          Range::GetExtentFromBeginEnd(copy->begin, copy->end), walkfn);
    }
    if (result.has_value()) {
      Var used_var = result.value()->value.cast<Var>();
      throw MakeScheduleError<DependentLoopError>(self->mod, ffi::GetRef<For>(copy), used_var->name,
                                                  DependentLoopError::PrimitiveKind::kReorder);
    }
    inner_vars.insert(copy->var.get());
    new_loop = For(std::move(n));
  }
  return new_loop.value();
}

void Reorder(ScheduleState self, const ffi::Array<StmtSRef>& ordered_loop_srefs) {
  if (ordered_loop_srefs.size() <= 1) {
    return;
  }
  // Step 1. Check uniqueness and collect the input loop srefs into a set
  std::unordered_set<const StmtSRefObj*> loop_srefs = CollectLoopsIntoSet(self, ordered_loop_srefs);
  // Step 2. Gather loops to be reordered
  // For each loop sref in the input sref array, traverse upwards along its parent pointer in the
  // sref tree, and stop on either a block, or a previously-visited loop
  // - the top of the reorder range is the last loop visited in the first traversal which exists in
  //   the input array
  // - the bottom of the reorder range is the last loop in the input array which is not visited in
  // the previous traversals
  auto [top, bottom] = GetBoundaryOfReorderRange(self, loop_srefs);
  // Step 3. Collect all loops in the chain and check the loops are single-branch
  std::vector<const StmtSRefObj*> chain = GetLoopsInReorderRange(self, top, bottom);
  // Step 4. Check the block below has all its block_var to be data-parallel or reduction,
  // and the block has an affine binding wrt top of the loop range.
  BlockPropertyError::CheckBlockIterTypeAndAffineBinding(self, top, bottom);
  // Step 5. Replace the original loops with the reordered loops and check that outer loop is
  // not dependent on inner loop
  For new_loop = ConstructNewLoopChain(self, std::move(chain), ordered_loop_srefs, loop_srefs);
  self->Replace(ffi::GetRef<StmtSRef>(top), new_loop, {});
}

StmtSRef AddUnitLoop(ScheduleState self, StmtSRef sref) {
  if (sref->stmt->IsInstance<ForObj>()) {
    For new_loop = For(PrimVar("u", PrimType::Int(32)), 0, 1, IntImm(PrimType::Int(32), 1),
                       ForKind::kDefault, ffi::GetRef<Stmt>(sref->stmt));
    self->Replace(sref, new_loop, {});
    return self->stmt2ref.at(new_loop.get());
  }
  class NewLoopCreator : public StmtExprMutator {
   public:
    using StmtExprMutator::Mutate;
    using StmtExprMutator::Mutate_;
    UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) override {
      if (value.as<ExprObj>()) return ffi::Unchanged();
      return StmtExprMutator::Mutate(value, inplace_mode);
    }

    explicit NewLoopCreator(const StmtObj* src_block) : src_block_(src_block) {}

    UnchangedOr<Stmt> Mutate_(const SBlockRealizeObj* realize, InplaceMode inplace_mode) final {
      if (realize->block.get() == src_block_) {
        new_loop_ = For(PrimVar("u", PrimType::Int(32)), 0, 1, IntImm(PrimType::Int(32), 1),
                        ForKind::kDefault, ffi::GetRef<SBlockRealize>(realize));
        return new_loop_.value();
      }
      return StmtExprMutator::Mutate_(realize, inplace_mode);
    }

    const StmtObj* src_block_;
    ffi::Optional<For> new_loop_;
  };

  TVM_FFI_CHECK(sref->parent != nullptr, ValueError) << "Cannot add loops on top of the root block";
  StmtSRef parent_sref = ffi::GetRef<StmtSRef>(sref->parent);
  auto creator = ffi::make_object<NewLoopCreator>(sref->stmt);
  Stmt new_stmt = creator->Mutate(ffi::GetRef<Stmt>(parent_sref->stmt))
                      .ValueOrUnchanged(ffi::GetRef<Stmt>(parent_sref->stmt));
  if (new_stmt->IsInstance<ForObj>()) {
    self->Replace(parent_sref, std::move(new_stmt), {});
  } else {
    SBlock old_parent_block = ffi::GetRef<SBlock>(parent_sref->StmtAs<SBlockObj>());
    SBlock new_parent_block = new_stmt.as_or_throw<SBlock>();
    self->Replace(parent_sref, new_stmt, {{old_parent_block, new_parent_block}});
  }
  return self->stmt2ref.at(creator->new_loop_.value().get());
}

/******** InstructionKind Registration ********/

struct SplitTraits : public UnpackedInstTraits<SplitTraits> {
  static constexpr const char* kName = "Split";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  template <size_t delta>
  TVM_FFI_INLINE static void _SetInputs(AnyView* packed_args, const ffi::Array<Any>& inputs) {
    thread_local Any loop_rv{nullptr};
    thread_local ffi::Array<Any> factors{nullptr};
    loop_rv = inputs[0];
    factors = ffi::Array<Any>{inputs.begin() + 1, inputs.end()};
    packed_args[delta] = loop_rv;
    packed_args[delta + 1] = factors;
  }

  static ffi::Array<LoopRV> UnpackedApplyToSchedule(Schedule sch, LoopRV loop_rv,
                                                    ffi::Array<ffi::Optional<ExprRV>> factors,
                                                    bool preserve_unit_iters,
                                                    bool disable_predication) {
    return sch->Split(loop_rv, factors, preserve_unit_iters, disable_predication);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String loop_rv,
                                      ffi::Array<Any> factors, bool preserve_unit_iters,
                                      bool disable_predication) {
    PythonAPICall py("split");
    py.Input("loop", loop_rv);
    py.Input("factors", factors);
    py.Input("preserve_unit_iters", preserve_unit_iters);
    py.Input("disable_predication", disable_predication);
    py.OutputList(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

struct LoopPartitionTraits : public UnpackedInstTraits<LoopPartitionTraits> {
  static constexpr const char* kName = "LoopPartition";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  template <size_t delta>
  TVM_FFI_INLINE static void _SetInputs(AnyView* packed_args, const ffi::Array<Any>& inputs) {
    thread_local Any loop_rv{nullptr};
    thread_local ffi::Array<Any> factors{nullptr};
    loop_rv = inputs[0];
    factors = ffi::Array<Any>{inputs.begin() + 1, inputs.end()};
    packed_args[delta] = loop_rv;
    packed_args[delta + 1] = factors;
  }

  static ffi::Array<LoopRV> UnpackedApplyToSchedule(Schedule sch, LoopRV loop_rv,
                                                    ffi::Array<ffi::Optional<ExprRV>> factors,
                                                    bool preserve_unit_iters) {
    return sch->LoopPartition(loop_rv, factors, preserve_unit_iters);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String loop_rv,
                                      ffi::Array<Any> factors, bool preserve_unit_iters) {
    PythonAPICall py("loop_partition");
    py.Input("loop", loop_rv);
    py.Input("factors", factors);
    py.Input("preserve_unit_iters", preserve_unit_iters);
    py.OutputList(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

struct MergeTraits : public UnpackedInstTraits<MergeTraits> {
  static constexpr const char* kName = "Merge";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  template <size_t delta>
  TVM_FFI_INLINE static void _SetInputs(AnyView* packed_args, const ffi::Array<Any>& inputs) {
    packed_args[delta] = inputs;
  }

  static LoopRV UnpackedApplyToSchedule(Schedule sch, ffi::Array<LoopRV> loop_rvs) {
    return sch->Merge(loop_rvs);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs,
                                      ffi::Array<ffi::String> loop_rvs) {
    PythonAPICall py("merge");
    for (const ffi::String& loop_rv : loop_rvs) {
      py.Input("", loop_rv);
    }
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

struct FuseTraits : public UnpackedInstTraits<FuseTraits> {
  static constexpr const char* kName = "Fuse";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  template <size_t delta>
  TVM_FFI_INLINE static void _SetInputs(AnyView* packed_args, const ffi::Array<Any>& inputs) {
    packed_args[delta] = inputs;
  }

  static LoopRV UnpackedApplyToSchedule(Schedule sch, ffi::Array<LoopRV> loop_rvs,
                                        bool preserve_unit_iters) {
    return sch->Fuse(loop_rvs, preserve_unit_iters);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs,
                                      ffi::Array<ffi::String> loop_rvs, bool preserve_unit_iters) {
    PythonAPICall py("fuse");
    for (const ffi::String& loop_rv : loop_rvs) {
      py.Input("", loop_rv);
    }
    py.Input("preserve_unit_iters", preserve_unit_iters);
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

struct ReorderTraits : public UnpackedInstTraits<ReorderTraits> {
  static constexpr const char* kName = "Reorder";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  template <size_t delta>
  TVM_FFI_INLINE static void _SetInputs(AnyView* packed_args, const ffi::Array<Any>& inputs) {
    packed_args[delta] = inputs;
  }

  static void UnpackedApplyToSchedule(Schedule sch, ffi::Array<LoopRV> loop_rvs) {
    return sch->Reorder(loop_rvs);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs,
                                      ffi::Array<ffi::String> loop_rvs) {
    PythonAPICall py("reorder");
    for (const ffi::String& loop_rv : loop_rvs) {
      py.Input("", loop_rv);
    }
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

struct AddUnitLoopTraits : public UnpackedInstTraits<AddUnitLoopTraits> {
  static constexpr const char* kName = "AddUnitLoop";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static LoopRV UnpackedApplyToSchedule(Schedule sch, ffi::ObjectRef rv) {
    if (auto block = rv.as<SBlockRV>()) {
      return sch->AddUnitLoop(block.value());
    } else if (auto loop = rv.as<LoopRV>()) {
      return sch->AddUnitLoop(loop.value());
    } else {
      TVM_FFI_THROW(TypeError) << "AddUnitLoop expects a loop or block";
      throw;
    }
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String rv) {
    PythonAPICall py("add_unit_loop");
    py.Input("block_or_loop", rv);
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

TVM_FFI_STATIC_INIT_BLOCK() {
  RegisterInstructionKind<SplitTraits>();
  RegisterInstructionKind<LoopPartitionTraits>();
  RegisterInstructionKind<MergeTraits>();
  RegisterInstructionKind<FuseTraits>();
  RegisterInstructionKind<ReorderTraits>();
  RegisterInstructionKind<AddUnitLoopTraits>();
}

}  // namespace s_tir
}  // namespace tvm
