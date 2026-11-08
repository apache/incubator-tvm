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
#include <tvm/ir/analysis.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/s_tir/stmt.h>

#include "../utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::prim;
using namespace tvm::tirx;

using support::NDIntSet;

/******** Error Classes ********/

/*!
 * \brief An error raised when not all required blocks are under the given loop.
 * \tparam is_consumer Indicates if all the required blocks are consumers or producers
 */
template <bool is_consumer>
class NotAllRequiredBlocksAreVisitedError : public ScheduleErrorContextObj {
 public:
  explicit NotAllRequiredBlocksAreVisitedError(IRModule mod, int num_not_visited,
                                               const ffi::Array<StmtSRef>& required)
      : mod_(mod), num_not_visited_(num_not_visited) {
    required_.reserve(required.size());
    for (const StmtSRef& block_sref : required) {
      const SBlockObj* block = TVM_SREF_TO_SBLOCK(block_sref);
      required_.push_back(ffi::GetRef<SBlock>(block));
    }
  }

  ffi::String FastErrorString() const final {
    return "ScheduleError: Not all required blocks are under the loop scope";
  }

  ffi::String DetailRenderTemplate() const final {
    ffi::String relation = is_consumer ? "consumer(s)" : "producer(s)";
    std::ostringstream os;
    os << "The primitive requires all the " << relation
       << " of the given block to be present under the target loop. However, there are "
       << num_not_visited_ << " " << relation << " not satisfying the constraint. List of the "
       << relation << ":";
    for (int i = 0, n = required_.size(); i < n; ++i) {
      os << "{" << i << "}";
    }
    return os.str();
  }

  IRModule mod() const final { return mod_; }

  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final {
    return {required_.begin(), required_.end()};
  }

 private:
  IRModule mod_;
  int num_not_visited_;
  ffi::Array<SBlock> required_;
};

/*!
 * \brief An error raised when the given block is not in the same block scope as the given loop,
 * or the given loop is the ancestor of the given block.
 */
class NotInSameScopeError : public ScheduleErrorContextObj {
 public:
  static void CheckAndBindLoopDomain(const ScheduleState& self, const StmtSRef& block_sref,
                                     const StmtSRef& loop_sref, const StmtSRef& scope_root_sref,
                                     sym::AnalyzerObj* analyzer) {
    for (const StmtSRefObj* p = loop_sref.get();; p = p->parent) {
      if (const ForObj* loop = p->StmtAs<ForObj>()) {
        {
          auto context = analyzer->GetConstraintContext();
          context->Append(sym::RangeConstraint(loop->var, Range(loop->begin, loop->end)));
          analyzer->SyncConstraintContext(context);
        }
      } else if (p != scope_root_sref.get()) {
        throw MakeScheduleError<NotInSameScopeError>(self->mod, block_sref, loop_sref);
      } else {
        break;
      }
    }
    for (const StmtSRefObj* p = block_sref->parent; p != scope_root_sref.get(); p = p->parent) {
      if (p == loop_sref.get()) {
        throw MakeScheduleError<NotInSameScopeError>(self->mod, block_sref, loop_sref);
      }
    }
  }

  ffi::String FastErrorString() const final {
    return "ScheduleError: Expected the block and loop to be under the same block scope, and loop "
           "not to be the ancestor of block";
  }
  ffi::String DetailRenderTemplate() const final {
    return "ScheduleError: Expected the block {0} and loop {1} to be under the same block scope, "
           "and loop not to be the ancestor of block";
  }
  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_, loop_}; }

  explicit NotInSameScopeError(IRModule mod, const StmtSRef& block_sref, const StmtSRef& loop_sref)
      : mod_(mod),
        block_(ffi::GetRef<SBlock>(block_sref->StmtAs<SBlockObj>())),
        loop_(ffi::GetRef<For>(loop_sref->StmtAs<ForObj>())) {}

 private:
  IRModule mod_;
  SBlock block_;
  For loop_;
};

/******** Helper Functions/Classes ********/

/*!
 * \brief Find a point where the block can be inserted under the loop
 * \tparam require_all_producers_visited Requires all producer blocks to be present under the loop
 * \tparam require_all_consumers_visited Requires all consumer blocks to be present under the loop
 * \param self The schedule state
 * \param subtrees The subtrees under the loop, among which the insertion points are sought
 * \param producer_srefs The producer blocks
 * \param consumer_srefs The consumer blocks
 * \param block2realize A cache that maps a block to its realize
 * \param index The block index of the loop body subtree blocks:
 * - `index = -1` means inserted into the last possible insertion point;
 * - `index = -2` means inserted into the first possible insertion point;
 * - Otherwise, `index` is a nonnegative number that indicates the insertion point
 * \return The possible position the new block can be inserted into, and the
 * producer-consumer-relationship is still satisfied.
 * \throws ScheduleError if there is no such insertion point found
 */
template <bool require_all_producers_visited, bool require_all_consumers_visited>
int FindInsertionPoint(const ScheduleState& self, const ffi::Array<Stmt>& subtrees,
                       const ffi::Array<StmtSRef>& producer_srefs,
                       const ffi::Array<StmtSRef>& consumer_srefs,
                       std::unordered_map<const SBlockObj*, const SBlockRealizeObj*>* block2realize,
                       int index) {
  ProducerConsumerSplit split =
      ProducerConsumerSplit::Find(self, subtrees, producer_srefs, consumer_srefs, block2realize);
  // Step 1. Check if all the producers are visited in the subtrees, if required to
  if (require_all_producers_visited) {
    int num_producers = producer_srefs.size();
    if (split.n_producers_visited < num_producers) {
      throw MakeScheduleError<NotAllRequiredBlocksAreVisitedError<false>>(
          self->mod, num_producers - split.n_producers_visited, producer_srefs);
    }
  }
  // Step 2. Check if all the consumers are visited in the subtrees, if required to
  if (require_all_consumers_visited) {
    int num_consumers = consumer_srefs.size();
    if (split.n_consumers_visited < num_consumers) {
      throw MakeScheduleError<NotAllRequiredBlocksAreVisitedError<true>>(
          self->mod, num_consumers - split.n_consumers_visited, consumer_srefs);
    }
  }
  // Step 3. Check if there is at least one index of the position can be inserted into
  // The valid indices are: (last_producer_position, first_consumer_position]
  TVM_FFI_ICHECK(split.last_producer_position < split.first_consumer_position);
  // Step 4. Return the possible insertion point according to index
  int insert_position;
  if (index == -1) {
    insert_position = split.first_consumer_position;
  } else if (index == -2) {
    insert_position = split.last_producer_position + 1;
  } else if (index >= 0 && index >= split.last_producer_position + 1 &&
             index <= split.first_consumer_position) {
    insert_position = index;
  } else {
    TVM_FFI_THROW(InternalError) << "Valid index:(-1, -2, [" << split.last_producer_position + 1
                                 << ", " << split.first_consumer_position << "]), "
                                 << "current index=" << index;
    throw;
  }
  return insert_position;
}

/*!
 * \brief Represent the iteration domain to fully cover the required region of Intersect(dom, bound)
 * The bound region may not get directly intersected with dom region, instead we try to generate
 * extra predicates for non-trivial bound. The domain info class can also union with each other.
 */
struct BlockVarDomainInfo {
  sym::IntSet dom{sym::IntSet::Nothing()};  // dom is ensured to be bounded
  sym::IntSet bound{sym::IntSet::Nothing()};

  /*! \brief Relaxed union operation */
  void Union(const BlockVarDomainInfo& other) {
    // just relax (d0 ^ b0) v (d1 ^ b1) to (d0 v d1) ^ (b0 v b1)
    dom = sym::Union({dom, other.dom});
    bound = sym::Union({bound, other.bound});
  }

  /*! \brief Simplify domain info */
  void Simplify(sym::AnalyzerObj* analyzer) {
    auto to_simplified = [analyzer](const sym::IntSet& set) {
      PrimExpr min = set.HasLowerBound() ? analyzer->Simplify(set.min()) : set.min();
      PrimExpr max = set.HasUpperBound() ? analyzer->Simplify(set.max()) : set.max();
      return sym::IntSet::Interval(min, max);
    };
    // if no dom specified, try use bound as dom
    if (dom.IsNothing()) {
      if (bound.HasLowerBound() && bound.HasUpperBound()) {
        bound = to_simplified(bound);
        std::swap(dom, bound);
      }
      return;
    }
    // simplify intset
    dom = to_simplified(dom);
    bound = to_simplified(bound);
    // if can proof the dom is within bound, remove bound
    auto intersect = to_simplified(sym::Intersect({dom, bound}));
    if (analyzer->CanProveEqual(dom.min(), intersect.min()) &&
        analyzer->CanProveEqual(dom.max(), intersect.max())) {
      bound = sym::IntSet::Nothing();
    } else if (analyzer->CanProveEqual(bound.min(), intersect.min()) &&
               analyzer->CanProveEqual(bound.max(), intersect.max())) {
      dom = bound;
      bound = sym::IntSet::Nothing();
    } else if (IsConstInt(intersect.min()) && IsConstInt(intersect.max())) {
      // if the bound induce constant iter range, merge bound to loop domain
      dom = intersect;
      bound = sym::IntSet::Nothing();
    }
  }
};

/*!
 * \brief Rewrite a block predicate from the loop nest being removed into the regenerated nest.
 *
 * `split` of a non-divisible loop stores the tail guard on the block realize (`k0 * factor + k1 <
 * extent`). `rfactor` keeps that guard, but the guard is not part of the tensor region
 * `compute_at` uses to rebuild loops, so the rebuilt realize would otherwise read out of bounds.
 * The result is the original predicate with removed loop vars replaced by the new iter values, or
 * nullopt when a removed loop var cannot be eliminated.
 */
ffi::Optional<PrimExpr> CarryBlockPredicate(const SBlockRealize& old_realize,
                                            const ffi::Array<PrimExpr>& new_iter_values,
                                            const ffi::Map<Var, Range>& old_loop_domains,
                                            const ffi::Map<Var, Range>& kept_loop_domains,
                                            sym::AnalyzerObj* analyzer) {
  PrimExpr predicate = old_realize->predicate;
  if (IsOne(predicate)) {
    return IntImm::Bool(true);
  }
  const SBlock& block = old_realize->block;
  int n = static_cast<int>(block->iter_vars.size());
  TVM_FFI_ICHECK_EQ(n, static_cast<int>(old_realize->iter_values.size()));
  TVM_FFI_ICHECK_EQ(n, static_cast<int>(new_iter_values.size()));

  auto dropped_loop = [&](const Var& var) -> bool {
    return old_loop_domains.find(var) != old_loop_domains.end() &&
           kept_loop_domains.find(var) == kept_loop_domains.end();
  };
  auto apply_subst = [&](PrimExpr expr, const ffi::Map<Var, PrimExpr>& subst) -> PrimExpr {
    auto f_substitute = [&subst](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
      if (auto repl = subst.Get(var)) return ffi::Any(*std::move(repl));
      return ffi::Unchanged();
    };
    PrimExpr rewritten =
        ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(std::move(expr), f_substitute)
            .as_or_throw<PrimExpr>();
    return analyzer->Simplify(std::move(rewritten));
  };
  auto free_of_dropped_loops = [&](const PrimExpr& expr) -> bool {
    for (const Var& var : UndefinedVars(expr)) {
      if (dropped_loop(var)) {
        return false;
      }
    }
    return true;
  };

  // Trivial bindings are loop vars. This is the form `rfactor` gives the factored block.
  {
    ffi::Map<Var, PrimExpr> trivial;
    bool trivial_ok = true;
    for (int i = 0; i < n; ++i) {
      const auto* loop_var = old_realize->iter_values[i].as<VarNode>();
      if (loop_var == nullptr) {
        trivial_ok = false;
        break;
      }
      Var key = ffi::GetRef<Var>(loop_var);
      const PrimExpr& replacement = new_iter_values[i];
      if (ffi::Optional<PrimExpr> previous = trivial.Get(key)) {
        if (!analyzer->CanProveEqual(previous.value(), replacement)) {
          trivial_ok = false;
          break;
        }
      } else {
        trivial.Set(key, replacement);
      }
    }
    if (trivial_ok) {
      PrimExpr rewritten = apply_subst(predicate, trivial);
      if (free_of_dropped_loops(rewritten)) {
        return rewritten;
      }
    }
  }

  // Non-trivial affine bindings: express the removed loops in the block iters, then in the new
  // iter values. Ignore the predicate here so a tail guard does not make the binding look
  // non-bijective.
  ffi::Map<PrimVar, Range> input_iters;
  for (const auto& kv : old_loop_domains) {
    input_iters.Set(kv.first.as_or_throw<PrimVar>(), kv.second);
  }
  sym::Analyzer analyzer_ref = ffi::GetRef<sym::Analyzer>(analyzer);
  sym::IterMapResult detected = sym::DetectIterMap(
      old_realize->iter_values, input_iters, IntImm::Bool(true), sym::IterMapLevel::Bijective,
      analyzer_ref, /*simplify_trivial_iterators=*/false);
  if (detected->indices.size() != static_cast<size_t>(n)) {
    return std::nullopt;
  }
  ffi::Array<PrimExpr> block_vars;
  block_vars.reserve(n);
  for (const IterVar& iter_var : block->iter_vars) {
    block_vars.push_back(iter_var->var);
  }
  ffi::Map<Var, PrimExpr> loop_to_block_var =
      sym::InverseAffineIterMap(detected->indices, block_vars);
  PrimExpr in_block_vars = apply_subst(predicate, loop_to_block_var);
  ffi::Map<Var, PrimExpr> block_var_to_new;
  for (int i = 0; i < n; ++i) {
    block_var_to_new.Set(block->iter_vars[i]->var, new_iter_values[i]);
  }
  PrimExpr rewritten = apply_subst(std::move(in_block_vars), block_var_to_new);
  if (!free_of_dropped_loops(rewritten)) {
    return std::nullopt;
  }
  return rewritten;
}

/*!
 * \brief A helper to reconstruct the block scope where the given block is moved under the given
 * loop, and the given block's induced loop nest is regenerated to satisfy the required region.
 */
class ScopeReconstructor : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) override {
    if (value.as<ExprObj>()) return ffi::Unchanged();
    return StmtExprMutator::Mutate(value, inplace_mode);
  }

  explicit ScopeReconstructor(SBlock scope_root, SBlock block, For loop)
      : scope_root_(scope_root), block_(block), loop_(loop) {}

  /*!
   * \brief Create the loop nest on top of the block, induced by the given block var's domain
   * \param insert_position The position among the subtrees where the block and its induced loop
   * nest is inserted
   * \param iter_doms The domain of each block var
   * \param analyzer The arithmetic analyzer
   * \param preserve_unit_loops Whether to generate unit loops where the loop extent is 1
   * \param old_realize The realize being moved, whose predicate is carried onto the new nest
   * \param old_loop_domains Loop domains surrounding `old_realize` before the move
   * \param kept_loop_domains Loop domains that remain outside the regenerated nest
   */
  void MakeNewLoop(int insert_position, std::vector<BlockVarDomainInfo> iter_doms,
                   sym::AnalyzerObj* analyzer, bool preserve_unit_loops,
                   const SBlockRealize& old_realize, const ffi::Map<Var, Range>& old_loop_domains,
                   const ffi::Map<Var, Range>& kept_loop_domains) {
    int n_iters = iter_doms.size();
    ffi::Array<Var> loop_vars;
    ffi::Array<PrimExpr> loop_extents;
    ffi::Array<PrimExpr> iter_values;
    loop_vars.reserve(n_iters);
    loop_extents.reserve(n_iters);
    iter_values.reserve(n_iters);
    PrimExpr predicate = IntImm::Bool(true);
    for (int i = 0; i < n_iters; ++i) {
      Range iter_dom = iter_doms[i].dom.CoverRange(block_->iter_vars[i]->dom.value()).value();
      if (preserve_unit_loops || !IsOne(iter_dom->extent)) {
        int bits = std::max(iter_dom->begin.ty().bits(), iter_dom->extent.ty().bits());
        Var var("ax" + std::to_string(loop_vars.size()), PrimType::Int(bits));
        loop_vars.push_back(var);
        loop_extents.push_back(analyzer->Simplify(iter_dom->extent));
        iter_values.push_back(iter_dom->begin + var.as_or_throw<PrimExpr>());
        {
          auto context = analyzer->GetConstraintContext();
          context->Append(sym::RangeConstraint(
              var,
              Range::FromMinExtent(IntImm(var->ty.as_or_throw<PrimType>(), 0), iter_dom->extent)));
          analyzer->SyncConstraintContext(context);
        }
      } else {
        iter_values.push_back(iter_dom->begin);
      }
      const sym::IntSet& pred_bound = iter_doms[i].bound;
      if (!pred_bound.IsNothing()) {
        // NOTE: Apply strong analyzer proofs to get rid of symbolic bound
        if (pred_bound.HasLowerBound()) {
          PrimExpr lower_bound = iter_values[i] >= pred_bound.min();
          if (!analyzer->CanProve(lower_bound, sym::ProofStrength::kSymbolicBound)) {
            predicate = predicate && lower_bound;
          }
        }
        if (pred_bound.HasUpperBound()) {
          PrimExpr upper_bound = iter_values[i] < pred_bound.max() + 1;
          if (!analyzer->CanProve(upper_bound, sym::ProofStrength::kSymbolicBound)) {
            predicate = predicate && upper_bound;
          }
        }
      }
    }
    // Keep a tail guard (or any other realize predicate) that the region-derived loops do not
    // imply. `reverse_compute_at` shares this path.
    if (ffi::Optional<PrimExpr> carried = CarryBlockPredicate(
            old_realize, iter_values, old_loop_domains, kept_loop_domains, analyzer)) {
      if (!analyzer->CanProve(carried.value())) {
        predicate = predicate && carried.value();
      }
    }
    this->new_block_realize_ =
        SBlockRealize(std::move(iter_values), analyzer->Simplify(predicate), std::move(block_));
    Stmt new_subtree = this->new_block_realize_.value();
    for (int i = static_cast<int>(loop_vars.size()) - 1; i >= 0; --i) {
      const Var& loop_var = loop_vars[i];
      const PrimExpr& loop_extent = loop_extents[i];
      new_subtree = For(loop_var.as_or_throw<PrimVar>(), IntImm::Int32(0), loop_extent,
                        IntImm(loop_var.as_or_throw<PrimVar>().ty(), 1), ForKind::kDefault,
                        std::move(new_subtree));
    }
    ffi::Array<Stmt> subtrees = AsArray(loop_->body);
    subtrees.insert(subtrees.begin() + insert_position, std::move(new_subtree));
    ffi::ObjectPtr<ForObj> new_loop = ffi::make_object<ForObj>(*loop_.get());
    new_loop->body = SeqStmt(std::move(subtrees));
    this->new_loop_ = For(std::move(new_loop));
  }

 private:
  UnchangedOr<Stmt> Mutate_(const SBlockObj* block, InplaceMode inplace_mode) final {
    if (block != scope_root_.get()) {
      return ffi::GetRef<SBlock>(block);
    }
    if (block == rm_src_stmt_.get()) {
      block = TVM_TYPE_AS(rm_tgt_stmt_, SBlockObj);
    }
    return StmtExprMutator::Mutate_(block, block->unique() ? inplace_mode : InplaceMode::kDisallow)
        .ValueOrUnchanged(ffi::GetRef<Stmt>(block));
  }

  UnchangedOr<Stmt> Mutate_(const ForObj* loop, InplaceMode inplace_mode) final {
    if (loop == rm_src_stmt_.get()) {
      loop = TVM_TYPE_AS(rm_tgt_stmt_, ForObj);
    }
    if (loop == loop_.get()) {
      return new_loop_.value();
    }
    return StmtExprMutator::Mutate_(loop, loop->unique() ? inplace_mode : InplaceMode::kDisallow)
        .ValueOrUnchanged(ffi::GetRef<Stmt>(loop));
  }

 public:
  /*! \brief The root block of the block scope */
  SBlock scope_root_;
  /*! \brief The given block to be moved */
  SBlock block_;
  /*! \brief The given loop the block and its loop nest to be put under */
  For loop_;
  /*! \brief The new loop to replace the original loop */
  ffi::Optional<For> new_loop_;
  /*! \brief The new block realize to the moved block */
  ffi::Optional<SBlockRealize> new_block_realize_;
  /*! \brief The plan to remove the given block by replacing this loop/block in the AST */
  // LeafBlockRemovalPlan initializes both removal statements before this mutator is used.
  Stmt rm_src_stmt_{ffi::UnsafeInit{}};
  /*! \brief The plan to remove the given block by replacing to this loop/block in the AST */
  Stmt rm_tgt_stmt_{ffi::UnsafeInit{}};
};

/*!
 * \brief Calculate a list of accessed tensor regions under a path of loops
 * \tparam relax_storage_scope Whether to relax beyond the path according to the storage and
 * execution scope
 * \param binding The block binding, used to unbind the tensor regions
 * \param tensor_regions The tensor regions to be calculated
 * \param relax_path_low_inclusive The lowest point in the loop path, inclusive
 * \param relax_path_high_exclusive The highest point in the loop path, exclusive
 * \param relaxed Where the calculation result is stored
 */
template <bool relax_storage_scope>
void RelaxTensorRegions(const ffi::Map<Var, PrimExpr>& binding,
                        const ffi::Array<TensorRegion>& tensor_regions,
                        const StmtSRef& relax_path_low_inclusive,
                        const StmtSRef& relax_path_high_exclusive,
                        std::unordered_map<const VarObj*, std::vector<NDIntSet>>* relaxed) {
  runtime::MemoryScope global_scope = runtime::MemoryScope::Global();
  // We cache the variable domains
  runtime::MemoryScope previous_scope = global_scope;
  ffi::Optional<ffi::Map<Var, sym::IntSet>> var_dom = std::nullopt;
  auto f_substitute = [&binding](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (auto repl = binding.Get(var)) return ffi::Any(*std::move(repl));
    return ffi::Unchanged();
  };
  // Enumerate every tensor region
  for (const TensorRegion& tensor_region : tensor_regions) {
    const TensorVar& tensor = tensor_region->source.as_or_throw<tvm::tirx::TensorVar>();
    const ffi::Array<Range>& region = tensor_region->region;
    // Skip the tensor regions we are not interested in
    auto it = relaxed->find(tensor.get());
    if (it == relaxed->end()) {
      continue;
    }
    std::vector<NDIntSet>& relaxed_regions = it->second;
    // Check and update the cached `var_dom`
    runtime::MemoryScope scope = relax_storage_scope ? tensor.scope() : global_scope;
    if (scope != previous_scope || !var_dom.has_value()) {
      previous_scope = scope;
      var_dom = sym::AsIntSet(LoopDomainOfSRefTreePath(
          /*low_inclusive=*/relax_path_low_inclusive,
          /*high_exclusive=*/relax_path_high_exclusive,
          /*extra_relax_scope=*/scope));
    }
    // Relax the region
    ffi::Array<Range> mapped_region = region.Map([&f_substitute](const Range& range) {
      PrimExpr min = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(range->begin, f_substitute)
                         .template as_or_throw<PrimExpr>();
      PrimExpr extent = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(range->extent, f_substitute)
                            .template as_or_throw<PrimExpr>();
      return Range::FromMinExtent(min, extent);
    });
    ffi::Array<sym::IntSet> relaxed_region = sym::EvalSet(mapped_region, var_dom.value());
    relaxed_regions.push_back({relaxed_region.begin(), relaxed_region.end()});
  }
}

/*!
 * \brief Calculate the iteration domain of a provided integer set to fully cover the required
 * domain
 * \param provided The provided integer set to cover the required domain
 * \param required The required domain to be covered
 * \param dim_max The maximum index bound by the tensor shape
 * \param analyzer The arithmetic analyzer
 */
std::pair<Var, BlockVarDomainInfo> SolveBlockVarDomain(const sym::IntSet& provided,
                                                       const sym::IntSet& required,
                                                       PrimExpr dim_max,
                                                       sym::AnalyzerObj* analyzer) {
  PrimExpr provided_min = analyzer->Simplify(provided.min());
  PrimExpr provided_max = analyzer->Simplify(provided.max());
  PrimExpr required_min = analyzer->Simplify(required.min());
  PrimExpr required_max = analyzer->Simplify(required.max());
  sym::IntSet var_dom{ffi::UnsafeInit{}}, var_bound{ffi::UnsafeInit{}};
  ffi::Optional<Var> var;
  ffi::Optional<PrimExpr> factor;
  if (const auto* mul = provided_min.as<prim::MulObj>()) {
    if (auto lhs = mul->a.as<Var>()) {
      var = lhs;
      factor = mul->b;
    } else if (auto rhs = mul->b.as<Var>()) {
      var = rhs;
      factor = mul->a;
    }
  }
  if (factor.has_value()) {
    PrimExpr e = factor.value();
    var_dom = sym::IntSet::Interval(floordiv(required_min, e), floordiv(required_max, e));
    var_bound = sym::IntSet::Interval(0, floordiv(dim_max, e));
  } else if (analyzer->CanProveEqual(provided_min, provided_max)) {
    if (auto single_var = provided_min.as<Var>()) {
      var = single_var;
      var_dom = sym::IntSet::Interval(required_min, required_max);
      var_bound = sym::IntSet::Interval(0, dim_max);
    } else {
      if (const auto* div = provided_min.as<prim::FloorDivObj>()) {
        PrimExpr var_expr = div->a;
        PrimExpr fac = div->b;
        if (analyzer->CanProveGreaterEqual(fac, 1)) {
          if (var_expr.as<PrimVar>()) {
            // a <= (x // factor) <= b, fac > 0 ==> (a * fac) <= x <= (b * fac + fac - 1)
            var = var_expr.as_or_throw<Var>();
            var_dom = sym::IntSet::Interval(required_min * fac,
                                            analyzer->Simplify(required_max * fac + fac - 1));
            var_bound = sym::IntSet::Interval(0, analyzer->Simplify(dim_max * fac + fac - 1));
          } else {
            const sym::IntSet new_provided = sym::IntSet::SinglePoint(div->a);
            const sym::IntSet new_required = sym::IntSet::Interval(
                required_min * fac, analyzer->Simplify(required_max * fac + fac - 1));
            return SolveBlockVarDomain(new_provided, new_required, dim_max, analyzer);
          }
        }
      } else if (const auto* mod = provided_min.as<prim::FloorModObj>()) {
        PrimExpr var_expr = mod->a;
        if (var_expr.as<PrimVar>()) {
          // generally domain of (x % fac) enforce no constraints to domain of x
          Var var_mod = var_expr.as_or_throw<Var>();
          return {var_mod, BlockVarDomainInfo()};
        } else {
          PrimExpr mod_1 = mod->a;
          PrimExpr mod_2 = mod->b;
          if (analyzer->CanProveGreaterEqual(mod_1, 1) &&
              analyzer->CanProveGreaterEqual(mod_2, 1)) {
            const sym::IntSet new_provided = sym::IntSet::SinglePoint(mod->a);
            if (analyzer->CanProveGreaterEqual(required_min, 0)) {
              const sym::IntSet new_required =
                  sym::IntSet::Interval(required_min, sym::IntSet::Everything().max());
              return SolveBlockVarDomain(new_provided, new_required, dim_max, analyzer);
            }
          }
        }
      }
    }
  }
  TVM_FFI_CHECK(var.has_value(), ValueError)
      << "TensorRegion pattern match failed: " << provided_min;
  return {var.value(), BlockVarDomainInfo{var_dom, var_bound}};
}

/*!
 * \brief Calculate and update the iteration domain info to fully cover the required domain in
 * dimension-wise fashion. The region relation on each tensor dimension is independently estimated.
 * \param tensor The accessed tensor
 * \param provided_region The provided NDIntSet to cover the required domain
 * \param required_region The required NDIntSet domain to be covered
 * \param analyzer The arithmetic analyzer
 * \param iter_doms The result iteration domains to be updated
 */
void UpdateBlockVarDomainDimwise(const VarObj* tensor, const NDIntSet& provided_region,
                                 const NDIntSet& required_region, sym::AnalyzerObj* analyzer,
                                 std::unordered_map<const VarObj*, BlockVarDomainInfo>* iter_doms) {
  size_t ndim = ffi::GetRef<Var>(tensor).as_or_throw<TensorVar>().shape().size();
  for (size_t i = 0; i < ndim; ++i) {
    sym::IntSet provided = provided_region[i];
    sym::IntSet required = required_region[i];
    PrimExpr dim_max = max(ffi::GetRef<Var>(tensor).as_or_throw<TensorVar>().shape()[i] - 1, 0);
    sym::Analyzer analyzer_ref = ffi::GetRef<sym::Analyzer>(analyzer);

    if (provided.CanProveSinglePoint(analyzer_ref) && IsConstInt(provided.min())) {
      TVM_FFI_ICHECK(required.CanProveSinglePoint(analyzer_ref) &&
                     analyzer->CanProveEqual(provided.min(), required.min()));
      continue;
    }

    auto [var, dom_info] = SolveBlockVarDomain(provided, required, dim_max, analyzer);
    auto it = iter_doms->find(var.get());
    if (it != iter_doms->end()) {
      it->second.Union(dom_info);
    } else {
      TVM_FFI_ICHECK(analyzer->CanProveEqual(provided.min(), required.min()));
      TVM_FFI_ICHECK(analyzer->CanProveEqual(provided.max(), required.max()));
    }
  }
}

/*! \brief Helper function to implement intset version of `InverseAffineIterMap`. */
ffi::Map<Var, sym::IntSet> InverseAffineIterMap(const ffi::Array<sym::IterSumExpr>& iter_map,
                                                const NDIntSet& outputs,
                                                sym::AnalyzerObj* analyzer) {
  ffi::Array<PrimExpr> min_point, max_point;
  min_point.reserve(outputs.size());
  max_point.reserve(outputs.size());
  for (const auto& intset : outputs) {
    TVM_FFI_ICHECK(intset.HasLowerBound() && intset.HasUpperBound());
    min_point.push_back(intset.min());
    max_point.push_back(intset.max());
  }
  auto rev_min = InverseAffineIterMap(iter_map, min_point);
  auto rev_max = InverseAffineIterMap(iter_map, max_point);
  ffi::Map<Var, sym::IntSet> dom_map;
  for (const auto& kv : rev_min) {
    const Var& var = kv.first;
    auto it = rev_max.find(var);
    TVM_FFI_ICHECK(it != rev_max.end());  // InverseAffineIterMap's result vars are assumed stable
    const PrimExpr& rev_min_point = kv.second;
    const PrimExpr& rev_max_point = (*it).second;
    dom_map.Set(var, sym::IntSet::Interval(analyzer->Simplify(min(rev_min_point, rev_max_point)),
                                           analyzer->Simplify(max(rev_min_point, rev_max_point))));
  }
  return dom_map;
}

/*!
 * \brief Calculate and update the iteration domain info to fully cover the required domain
 * with affine analysis. It requires bijective mapping of block var to provided region points.
 * \param tensor The accessed tensor
 * \param iter_vars The list of block vars to cover the required region
 * \param provided_region The provided NDIntSet to cover the required domain
 * \param required_region The required NDIntSet domain to be covered
 * \param analyzer The arithmetic analyzer
 * \param iter_doms The result iteration domains to be updated
 * \returns bool. Denotes whether update success
 */
bool UpdateBlockVarDomainAffine(const VarObj* tensor, const ffi::Array<IterVar>& iter_vars,
                                const NDIntSet& provided_region, const NDIntSet& required_region,
                                sym::AnalyzerObj* analyzer,
                                std::unordered_map<const VarObj*, BlockVarDomainInfo>* iter_doms) {
  // we only support single point provided region now, which could cover most cases
  sym::Analyzer analyzer_ref = ffi::GetRef<sym::Analyzer>(analyzer);
  for (const auto& intset : provided_region) {
    if (!intset.CanProveSinglePoint(analyzer_ref)) return false;
  }
  // calculate forward mapping (block vars -> provided region point)
  ffi::Map<PrimVar, Range> dom_map;
  for (const IterVar& iter_var : iter_vars) {
    dom_map.Set(iter_var->var, iter_var->dom.value());
  }
  size_t ndim = ffi::GetRef<Var>(tensor).as_or_throw<TensorVar>().shape().size();
  ffi::Array<PrimExpr> provide_indices;
  provide_indices.reserve(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    provide_indices.push_back(provided_region[i].min());
  }
  auto res = sym::DetectIterMap(provide_indices, dom_map, IntImm::Bool(true),
                                sym::IterMapLevel::Bijective, analyzer_ref, false);
  if (res->indices.empty()) {
    return false;
  }
  // calculate backward mapping (required region point -> block vars)
  NDIntSet required_bound;
  for (size_t i = 0; i < ndim; ++i) {
    required_bound.push_back(sym::IntSet::Interval(
        IntImm(ffi::GetRef<Var>(tensor).as_or_throw<TensorVar>().shape()[i].ty(), 0),
        max(ffi::GetRef<Var>(tensor).as_or_throw<TensorVar>().shape()[i] - 1, 0)));
  }
  ffi::Map<Var, sym::IntSet> var_dom =
      InverseAffineIterMap(res->indices, required_region, analyzer);
  ffi::Map<Var, sym::IntSet> var_bound =
      InverseAffineIterMap(res->indices, required_bound, analyzer);
  for (const auto& kv : var_dom) {
    const Var& var = kv.first;
    auto it = var_bound.find(var);
    TVM_FFI_ICHECK(it != var_bound.end());  // InverseAffineIterMap's result vars are assumed stable
    (*iter_doms)[var.get()].Union(BlockVarDomainInfo{kv.second, (*it).second});
  }
  return true;
}

/*!
 * \brief Calculate the domain of block vars to cover the required region
 * \param iter_vars The list of block vars to cover the required region
 * \param provided_regions The region provided by one iteration instance of the block vars
 * \param required_regions The region required to be covered
 * \param analyzer The arithmetic analyzer
 * \return A list of iteration domain info corresponding to the given list of block vars
 */
std::vector<BlockVarDomainInfo> CalculateBlockVarDomain(
    const ffi::Array<IterVar>& iter_vars,
    std::unordered_map<const VarObj*, std::vector<NDIntSet>> provided_regions,
    std::unordered_map<const VarObj*, std::vector<NDIntSet>> required_regions,
    sym::AnalyzerObj* analyzer) {
  int n_iters = iter_vars.size();
  // Step 1. Construct the mapping from block var to their iteration domain (initialized to empty)
  std::unordered_map<const VarObj*, BlockVarDomainInfo> iter_doms;
  iter_doms.reserve(n_iters);
  for (const IterVar& iter_var : iter_vars) {
    iter_doms[iter_var->var.get()] = BlockVarDomainInfo();
  }
  // Step 2. For each tensor, update the domain according to the provided and required regions
  for (const auto& kv : provided_regions) {
    const VarObj* tensor = kv.first;
    const std::vector<NDIntSet>& many_provided_regions = kv.second;
    // Calculate `provided_region` and `required_region`
    auto it = required_regions.find(tensor);
    if (it == required_regions.end() || it->second.empty()) {
      continue;
    }
    NDIntSet required_region = support::NDIntSetUnion(it->second);
    NDIntSet provided_region = support::NDIntSetUnion(many_provided_regions);
    TVM_FFI_ICHECK_EQ(provided_region.size(),
                      ffi::GetRef<Var>(tensor).as_or_throw<TensorVar>().shape().size());
    TVM_FFI_ICHECK_EQ(required_region.size(),
                      ffi::GetRef<Var>(tensor).as_or_throw<TensorVar>().shape().size());
    // Try update iter var domains with current required and provided region pair.
    if (!UpdateBlockVarDomainAffine(tensor, iter_vars, provided_region, required_region, analyzer,
                                    &iter_doms)) {
      UpdateBlockVarDomainDimwise(tensor, provided_region, required_region, analyzer, &iter_doms);
    }
  }
  // Union the iter var domains, put them in the same order of block vars, and return
  std::vector<BlockVarDomainInfo> result;
  result.reserve(n_iters);
  for (const IterVar& iter_var : iter_vars) {
    BlockVarDomainInfo& info = iter_doms.at(iter_var->var.get());
    if (info.bound.IsNothing()) {
      info.bound = sym::IntSet::FromRange(iter_var->dom.value());
    } else {
      info.bound = sym::Intersect({info.bound, sym::IntSet::FromRange(iter_var->dom.value())});
    }
    info.Simplify(analyzer);
    TVM_FFI_ICHECK(!info.dom.IsNothing());
    result.push_back(info);
  }
  return result;
}

/*!
 * \brief Calculate the provided region of the given block by one single of its execution instance,
 * as well as the required tensor regions relaxed to the given loop
 * \tparam is_compute_at Indicates if the operation is compute-at or reverse-compute-at
 * \param block The given block that provides tensor regions
 * \param loop_sref The given loop under which the block is going to be moved to
 * \param block2realize Maps a block to its corresponding BlockRealize
 * \param producer_srefs The producers of the given block
 * \param consumer_srefs The consumers of the given block
 * \param provided_regions The calculated regions provided by the block
 * \param required_regions The calculated regions required by its consumers (in compute-at) or
 * producers (in reverse-compute-at)
 */
template <bool is_compute_at>
void CalculateProvidedRequiredRegions(
    const SBlockObj* block, const StmtSRef& loop_sref,
    std::unordered_map<const SBlockObj*, const SBlockRealizeObj*> block2realize,
    ffi::Array<StmtSRef> producer_srefs, ffi::Array<StmtSRef> consumer_srefs,
    std::unordered_map<const VarObj*, std::vector<NDIntSet>>* provided_regions,
    std::unordered_map<const VarObj*, std::vector<NDIntSet>>* required_regions) {
  // Step 1. Calculate the region provided by a single execution instance of `block`
  const ffi::Array<TensorRegion>& provided_tensors = is_compute_at ? block->writes : block->reads;
  provided_regions->reserve(provided_tensors.size());
  required_regions->reserve(provided_tensors.size());
  for (const TensorRegion& provided_tensor_region : provided_tensors) {
    const VarObj* tensor = provided_tensor_region->source.as_or_throw<tvm::tirx::TensorVar>().get();
    const ffi::Array<Range>& region = provided_tensor_region->region;
    (*provided_regions)[tensor].push_back(support::NDIntSetFromRegion(region));
    (*required_regions)[tensor].clear();
  }
  // Step 2. Calculate the region required by dependent blocks under `loop`
  for (const StmtSRef& required_block_sref : is_compute_at ? consumer_srefs : producer_srefs) {
    const SBlockObj* required_block = TVM_SREF_TO_SBLOCK(required_block_sref);
    TVM_FFI_ICHECK(block2realize.count(required_block));
    RelaxTensorRegions</*relax_storage_scope=*/is_compute_at>(
        /*binding=*/GetBindings(ffi::GetRef<SBlockRealize>(block2realize.at(required_block))),
        /*tensor_regions=*/is_compute_at ? required_block->reads : required_block->writes,
        /*relax_path_low_inclusive=*/ffi::GetRef<StmtSRef>(required_block_sref->parent),
        /*relax_path_high_exclusive=*/loop_sref, /*relaxed=*/required_regions);
  }
}

/******** Main Implementation ********/

template <bool is_compute_at>
void ComputeAtOrReverseComputeAtImpl(ScheduleState self, const StmtSRef& block_sref,
                                     const StmtSRef& loop_sref, bool preserve_unit_loops,
                                     sym::AnalyzerObj* analyzer, bool check_only = false,
                                     int index = -1) {
  const SBlockObj* block = TVM_SREF_TO_SBLOCK(block_sref);
  const ForObj* loop = TVM_SREF_TO_FOR(loop_sref);
  // Step 1. Bunch of checks
  // Check condition 1) : scope stage pipeline
  StmtSRef scope_root_sref = GetScopeRoot(self, block_sref,
                                          /*require_stage_pipeline=*/true);
  SBlock scope_root = ffi::GetRef<SBlock>(scope_root_sref->StmtAs<SBlockObj>());
  AddShapeVarBounds(self, scope_root_sref.get(), analyzer);
  SBlockScope scope = self->GetSBlockScope(scope_root_sref);
  ffi::Array<StmtSRef> producer_srefs = GetProducers(block_sref, scope);
  ffi::Array<StmtSRef> consumer_srefs = GetConsumers(block_sref, scope);
  // Check condition 2) : `block` is a complete or reduction block
  CheckCompleteOrReductionBlock(self, block_sref, scope_root_sref);
  // Check condition 3): `block` and `loop` are under the same scope,
  // and `loop` is not the ancestor of `block`
  NotInSameScopeError::CheckAndBindLoopDomain(self, block_sref, loop_sref, scope_root_sref,
                                              analyzer);
  // Check condition 4): `block` is not an output block
  if (is_compute_at) {
    CheckNotOutputBlock(self, block_sref, scope_root_sref);
  }
  // Step 2. Plan for the removal of `block`
  auto reconstructor = ffi::make_object<ScopeReconstructor>(scope_root, ffi::GetRef<SBlock>(block),
                                                            ffi::GetRef<For>(loop));
  LeafBlockRemovalPlan(self, block_sref, &reconstructor->rm_src_stmt_,
                       &reconstructor->rm_tgt_stmt_);
  // Step 3. Find the insertion point under `loop`
  // Check condition 5): all the required block are under the given loop
  std::unordered_map<const SBlockObj*, const SBlockRealizeObj*> block2realize;
  block2realize.reserve(self->block_info.size());
  int insert_position = FindInsertionPoint<!is_compute_at, is_compute_at>(
      /*self=*/self,
      /*subtrees=*/AsArray(loop->body),
      /*producer_srefs=*/producer_srefs,
      /*consumer_srefs=*/consumer_srefs, /*block2realize=*/&block2realize,
      /*index=*/index);
  // Step 4. Calculate the region provided by a single execution instance of `block`,
  // as well as the region required by dependent blocks under `loop`.
  // Here is the definition of `provide` and `require`:
  // - In compute-at, `provide` means `produce`, and `require` means `consume`
  // - In reverse-compute-at, `provide` means `consume`, and `require` means `produce`
  std::unordered_map<const VarObj*, std::vector<NDIntSet>> provided_regions;
  std::unordered_map<const VarObj*, std::vector<NDIntSet>> required_regions;
  CalculateProvidedRequiredRegions<is_compute_at>(
      /*block=*/block, /*loop_sref=*/loop_sref, /*block2realize=*/std::move(block2realize),
      /*producer_srefs=*/std::move(producer_srefs),
      /*consumer_srefs=*/std::move(consumer_srefs),
      /*provided_regions=*/&provided_regions, /*required_regions=*/&required_regions);
  // Step 5. Calculate the iteration domain for each block var
  std::vector<BlockVarDomainInfo> iter_doms =
      CalculateBlockVarDomain(/*iter_vars=*/block->iter_vars,
                              /*provided_regions=*/std::move(provided_regions),
                              /*required_regions=*/std::move(required_regions),
                              /*analyzer=*/analyzer);
  // Step 6. Create the new scope according to the iteration domain
  SBlockRealize old_realize = GetSBlockRealize(self, block_sref);
  ffi::Map<Var, Range> old_loop_domains =
      LoopDomainOfSRefTreePath(ffi::GetRef<StmtSRef>(block_sref->parent));
  ffi::Map<Var, Range> kept_loop_domains = LoopDomainOfSRefTreePath(loop_sref);
  reconstructor->MakeNewLoop(/*insert_position=*/insert_position,
                             /*iter_doms=*/std::move(iter_doms),
                             /*analyzer=*/analyzer, /*preserve_unit_loops=*/preserve_unit_loops,
                             /*old_realize=*/old_realize, /*old_loop_domains=*/old_loop_domains,
                             /*kept_loop_domains=*/kept_loop_domains);
  SBlock new_scope_root =
      reconstructor->Mutate(scope_root).ValueOrUnchanged(scope_root).as_or_throw<SBlock>();

  // Step 7. Do the actual replacement
  if (check_only) {
    return;
  }
  self->Replace(scope_root_sref, new_scope_root, {{scope_root, new_scope_root}});
  // Step 8. Update the cached flags
  SBlockInfo& block_info = self->block_info[block_sref];
  block_info.affine_binding = IsAffineBinding(
      /*realize=*/reconstructor->new_block_realize_.value(),
      /*loop_var_ranges=*/LoopDomainOfSRefTreePath(ffi::GetRef<StmtSRef>(block_sref->parent)),
      /*analyzer=*/analyzer);
}

void ComputeAt(ScheduleState self, const StmtSRef& block_sref, const StmtSRef& loop_sref,
               bool preserve_unit_loops, int index) {
  sym::Analyzer analyzer;
  ComputeAtOrReverseComputeAtImpl<true>(self, block_sref, loop_sref, preserve_unit_loops,
                                        analyzer.get(), false, index);
}

void ReverseComputeAt(ScheduleState self, const StmtSRef& block_sref, const StmtSRef& loop_sref,
                      bool preserve_unit_loops, int index) {
  sym::Analyzer analyzer;
  ComputeAtOrReverseComputeAtImpl<false>(self, block_sref, loop_sref, preserve_unit_loops,
                                         analyzer.get(), false, index);
}

bool CanComputeAt(const ScheduleState& self, const StmtSRef& block_sref, const StmtSRef& loop_sref,
                  bool preserve_unit_loops) {
  sym::Analyzer analyzer;
  try {
    ComputeAtOrReverseComputeAtImpl<true>(self, block_sref, loop_sref, preserve_unit_loops,
                                          analyzer.get(), true);
  } catch (const tvm::ffi::Error& e) {
    return false;
  }
  return true;
}

bool CanReverseComputeAt(const ScheduleState& self, const StmtSRef& block_sref,
                         const StmtSRef& loop_sref, bool preserve_unit_loops) {
  sym::Analyzer analyzer;
  try {
    ComputeAtOrReverseComputeAtImpl<false>(self, block_sref, loop_sref, preserve_unit_loops,
                                           analyzer.get(), true);
  } catch (const tvm::ffi::Error& e) {
    return false;
  }
  return true;
}

/******** InstructionKind Registration ********/

struct ComputeAtTraits : public UnpackedInstTraits<ComputeAtTraits> {
  static constexpr const char* kName = "ComputeAt";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, SBlockRV block_rv, LoopRV loop_rv,
                                      bool preserve_unit_loops, int index) {
    return sch->ComputeAt(block_rv, loop_rv, preserve_unit_loops, index);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block_rv,
                                      ffi::String loop_rv, bool preserve_unit_loops, int index) {
    PythonAPICall py("compute_at");
    py.Input("block", block_rv);
    py.Input("loop", loop_rv);
    py.Input("preserve_unit_loops", preserve_unit_loops);
    py.Input("index", index);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

struct ReverseComputeAtTraits : public UnpackedInstTraits<ReverseComputeAtTraits> {
  static constexpr const char* kName = "ReverseComputeAt";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, SBlockRV block_rv, LoopRV loop_rv,
                                      bool preserve_unit_loops, int index) {
    return sch->ReverseComputeAt(block_rv, loop_rv, preserve_unit_loops, index);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block_rv,
                                      ffi::String loop_rv, bool preserve_unit_loops, int index) {
    PythonAPICall py("reverse_compute_at");
    py.Input("block", block_rv);
    py.Input("loop", loop_rv);
    py.Input("preserve_unit_loops", preserve_unit_loops);
    py.Input("index", index);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

TVM_FFI_STATIC_INIT_BLOCK() {
  RegisterInstructionKind<ComputeAtTraits>();
  RegisterInstructionKind<ReverseComputeAtTraits>();
}

}  // namespace s_tir
}  // namespace tvm
