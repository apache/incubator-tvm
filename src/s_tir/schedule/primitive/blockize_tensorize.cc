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
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/tensor_intrin.h>

#include <functional>

#include "../../ir/data_type_rewriter.h"
#include "../../transform/stmt_simplify.h"
#include "../ir_comparator.h"
#include "../utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::prim;
using namespace tvm::tirx;

Range RangeFromExtent(const PrimExpr& extent) {
  return Range::FromMinExtent(IntImm(extent.ty(), 0), extent);
}

template <class T>
T DeepCopy(const T& stmt) {
  return ffi::FromJSONGraph(ffi::ToJSONGraph(stmt)).template as_or_throw<T>();
}

/*!
 * \brief ScheduleError that the bindings of the inner block are not divisible by the subspace
 * represented by the outer loops.
 */
class SubspaceNotDivisibleError : public ScheduleErrorContextObj {
 public:
  explicit SubspaceNotDivisibleError(IRModule mod, For scope_loop, SBlock inner_block)
      : mod_(std::move(mod)),
        scope_loop_(std::move(scope_loop)),
        inner_block_(std::move(inner_block)) {}

  ffi::String FastErrorString() const final {
    return "ScheduleError: The bindings of the inner block can not be blockized.";
  }

  ffi::String DetailRenderTemplate() const final {
    return "ScheduleError: The bindings of the inner block {0} can not be blockized by the loops "
           "starting at {1}.";
  }

  IRModule mod() const final { return mod_; }

  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final {
    return {inner_block_, scope_loop_};
  }

 private:
  IRModule mod_;
  For scope_loop_;
  SBlock inner_block_;
};

/*!
 * \brief Detect if bindings are a trivial case of the subspace division where we can divide the
 * block iter bindings into two categories:
 *   1. The binding covers no inner loop vars.
 *   2. The binding covers only inner loop vars.
 *
 * The bindings are not required to be quasi-affine. Trivial block iters are always preserved.
 *
 * \param iter_vars The input iterators
 * \param bindings The values of iter_vars
 * \param predicate The predicate constraint on the input iterators.
 * \param outer_iters The iters of the outer space
 * \param inner_iters The iters of the inner space
 * \return The result of the subspace division.
 */
ffi::Array<ffi::Array<sym::IterMark>> TrivialSubspaceDivision(const ffi::Array<IterVar>& iter_vars,
                                                              const ffi::Array<PrimExpr>& bindings,
                                                              const PrimExpr& predicate,
                                                              const ffi::Array<Var>& outer_iters,
                                                              const ffi::Array<Var>& inner_iters) {
  if (!IsOne(predicate)) return {};
  ffi::Array<ffi::Array<sym::IterMark>> res;
  std::unordered_set<const VarObj*> outer_loop_vars;
  std::unordered_set<const VarObj*> inner_loop_vars;

  auto make_uses_var =
      [](const ffi::Array<Var>& vars) -> std::function<bool(const PrimExpr& expr)> {
    std::unordered_set<const VarObj*> var_set;
    var_set.reserve(vars.size());
    for (const Var& var : vars) {
      var_set.insert(var.get());
    }
    return [var_set = std::move(var_set)](const PrimExpr& expr) -> bool {
      auto walkfn = [&var_set](const Var& var) -> ffi::Expected<ffi::WalkResult> {
        return var_set.count(var.get()) ? ffi::WalkResult::Interrupt(ffi::VisitInterrupt(var))
                                        : ffi::WalkResult::Advance();
      };
      return ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(expr, walkfn).has_value();
    };
  };
  auto use_outer_loop_vars = make_uses_var(outer_iters);
  auto use_inner_loop_vars = make_uses_var(inner_iters);
  sym::IterMark unit_iter_mark(sym::IterSumExpr({}, 0), 1);

  for (int i = 0, n = bindings.size(); i < n; ++i) {
    bool outer = use_outer_loop_vars(bindings[i]);
    bool inner = use_inner_loop_vars(bindings[i]);
    sym::IterMark iter_mark{ffi::UnsafeInit{}};
    if (bindings[i].as<PrimVar>()) {
      iter_mark = sym::IterMark(
          sym::IterSplitExpr(sym::IterMark(bindings[i], iter_vars[i]->dom.value()->extent)),
          iter_vars[i]->dom.value()->extent);
    } else {
      iter_mark =
          sym::IterMark(sym::IterSumExpr({}, bindings[i]), iter_vars[i]->dom.value()->extent);
    }
    if (outer && !inner) {
      res.push_back({/*outer_iter=*/iter_mark, /*inner_iter=*/unit_iter_mark});
    } else if (inner && !outer) {
      res.push_back({/*outer_iter=*/unit_iter_mark, /*inner_iter=*/iter_mark});
    } else if (!outer && !inner) {
      res.push_back({/*outer_iter=*/unit_iter_mark, /*inner_iter=*/unit_iter_mark});
    } else {
      return {};
    }
  }
  res.push_back({sym::IterMark(sym::IterSumExpr({}, 0), IntImm::Bool(true)),
                 sym::IterMark(sym::IterSumExpr({}, 0), IntImm::Bool(true))});
  return res;
}

/*!
 * \brief Subspace division. The space is divided into two subspaces:
 * If loop_sref_as_outer is false:
 *  1. The subspace represented by the outer loops above `loop_sref` (exclusive).
 *  2. The subspace represented by the inner loops below `loop_sref` (inclusive).
 * else:
 *  1. The subspace represented by the outer loops above `loop_sref` (inclusive).
 *  2. The subspace represented by the inner loops below `loop_sref` (exclusive).
 * \param realize The inner block
 * \param block_sref The sref to the inner block
 * \param loop_sref The loop that is the root of the second subspace.
 * \param loops The loops that represents the second part of the subspace.
 * \param analyzer The arithmetic analyzer to use.
 * \param preserve_unit_iters Whether or not to preserve unit iterators in block bindings
 * \param loop_sref_as_outer Whether loop_sref is divided into outer or inner
 */
ffi::Array<ffi::Array<sym::IterMark>> SubspaceDivide(const SBlockRealize& realize,
                                                     const StmtSRef& block_sref,  //
                                                     const StmtSRef& loop_sref,   //
                                                     std::vector<const ForObj*>* loops,
                                                     sym::AnalyzerObj* analyzer,
                                                     bool preserve_unit_iters,
                                                     bool loop_sref_as_outer = false) {
  ffi::Array<Var> inner_vars;
  ffi::Array<Var> outer_vars;
  ffi::Array<PrimVar> primitive_inner_vars;
  ffi::Map<PrimVar, Range> primitive_loop_var_domain;
  bool inner = true;
  for (StmtSRefObj* sref = block_sref->parent;    //
       sref && sref->stmt->IsInstance<ForObj>();  //
       sref = sref->parent) {
    const ForObj* loop = static_cast<const ForObj*>(sref->stmt);
    if (inner) {
      loops->push_back(loop);
      inner_vars.push_back(loop->var);
      primitive_inner_vars.push_back(loop->var);
    } else {
      outer_vars.push_back(loop->var);
    }
    primitive_loop_var_domain.Set(loop->var, Range(loop->begin, loop->end));
    if ((loop_sref_as_outer && sref->parent == loop_sref.get()) || sref == loop_sref.get()) {
      inner = false;
    }
  }
  ffi::Array<ffi::Array<sym::IterMark>> result = sym::SubspaceDivide(
      realize->iter_values, primitive_loop_var_domain, primitive_inner_vars, realize->predicate,
      sym::IterMapLevel::Surjective, ffi::GetRef<sym::Analyzer>(analyzer),
      /*simplify_trivial_iterators=*/!preserve_unit_iters);
  if (!result.empty()) {
    return result;
  }
  return TrivialSubspaceDivision(realize->block->iter_vars,
                                 realize->iter_values,  //
                                 realize->predicate,    //
                                 outer_vars, inner_vars);
}

/*!
 * \brief Derive the block bindings for both inner and outer block
 * \param iter_vars The original block iterators to the inner block
 * \param division The subspace division.
 * \param outer_iter_vars The outer block iterators.
 * \param outer_bindings The outer block bindings.
 * \param inner_iter_vars The inner block iterators.
 * \param inner_bindings The inner block bindings.
 * \param preserve_unit_iters Whether or not to preserve unit iterators in block bindings
 * \return A substitution plan to the iterators in the original inner block.
 */
ffi::Map<Var, PrimExpr> DeriveBlockBinding(
    const ffi::Array<IterVar>& iter_vars,                   //
    const ffi::Array<ffi::Array<sym::IterMark>>& division,  //
    ffi::Array<IterVar>* outer_iter_vars,                   //
    ffi::Array<PrimExpr>* outer_bindings,                   //
    ffi::Array<IterVar>* inner_iter_vars,                   //
    ffi::Array<PrimExpr>* inner_bindings,                   //
    bool preserve_unit_iters, bool reuse_outer = false) {
  using sym::IterMapExpr;
  using sym::IterMapExprObj;
  using sym::NormalizeIterMapToExpr;
  ffi::Map<Var, PrimExpr> block_var_subst;
  TVM_FFI_ICHECK_EQ(iter_vars.size() + 1, division.size());
  sym::Analyzer ana;
  for (int i = 0, n = iter_vars.size(); i < n; ++i) {
    const IterVar& iter_var = iter_vars[i];
    sym::IterMark outer_mark = division[i][0];
    sym::IterMark inner_mark = division[i][1];
    IterMapExpr outer_binding = outer_mark->source.as_or_throw<IterMapExpr>();
    IterMapExpr inner_binding = inner_mark->source.as_or_throw<IterMapExpr>();
    // After computing the subspace division, bindings[i] can be written as
    // outer_binding * inner_binding->extent + inner_binding
    // The outer block will have binding: iter_outer -> outer_binding
    // The inner block will have binding: iter_inner -> inner_binding
    // The iter in the original block will be substituted with base + iter_inner where
    // base == iter_outer * iter_inner_extent
    // create iter var for the outer block
    IterVar outer_iter(ffi::UnsafeInit{});
    if (reuse_outer) {
      outer_iter = outer_iter_vars->operator[](i);
      TVM_FFI_ICHECK(ana->CanProveEqual(outer_iter->dom.value()->extent, outer_mark->extent));
      TVM_FFI_ICHECK(
          ana->CanProveEqual(outer_bindings->operator[](i), NormalizeIterMapToExpr(outer_binding)));
    } else {
      outer_iter = IterVar(/*dom=*/RangeFromExtent(outer_mark->extent),
                           /*var=*/iter_var->var.CopyWithSuffix("_o"),
                           /*iter_type=*/iter_var->iter_type);
      outer_bindings->push_back(NormalizeIterMapToExpr(outer_binding));
      outer_iter_vars->push_back(outer_iter);
    }
    PrimExpr sub = [&]() -> PrimExpr {
      if (IsOne(inner_mark->extent)) {
        // Skip inner var when extent is 1
        // substitution
        if (IsOne(outer_mark->extent) && !preserve_unit_iters) {
          // Simplify outer if not preserve_unit_iters
          return IntImm(outer_mark->extent.ty(), 0);
        } else {
          return outer_iter->var;
        }
      } else {
        // create iter var for the inner block
        IterVar inner_iter(/*dom=*/RangeFromExtent(inner_mark->extent),
                           /*var=*/iter_var->var.CopyWithSuffix("_i"),
                           /*iter_type=*/iter_var->iter_type);
        inner_bindings->push_back(NormalizeIterMapToExpr(inner_binding));
        inner_iter_vars->push_back(inner_iter);
        // substitution
        if (IsOne(outer_mark->extent)) {
          return inner_iter->var;
        } else {
          return outer_iter->var * inner_mark->extent + inner_iter->var;
        }
      }
    }();
    block_var_subst.Set(iter_var->var, sub);
  }
  return block_var_subst;
}

/*!
 * \brief Generate the inner block for blockization
 * \param is_write_reduction Whether the write regions of the inner block are actually reduction.
 * \param iter_vars IterVars used in the inner block.
 * \param iter_values IterVar bindings used in the inner block.
 * \param predicate The predicate of the inner block.
 * \param block The inner block as a template to be created from. This method will modify its
 * `iter_vars`, `init` and `reads` fields.
 * \return The inner block created.
 */
SBlockRealize GenerateInner(bool is_write_reduction,
                            const ffi::Array<IterVar>& iter_vars,     //
                            const ffi::Array<PrimExpr>& iter_values,  //
                            const PrimExpr& predicate,                //
                            SBlock block) {
  SBlockObj* n = block.CopyOnWrite();
  n->iter_vars = iter_vars;
  n->init = std::nullopt;
  if (is_write_reduction) {
    ffi::Array<TensorRegion> reads;
    reads.reserve(block->writes.size() + block->reads.size());
    reads.insert(reads.end(), block->writes.begin(), block->writes.end());
    reads.insert(reads.end(), block->reads.begin(), block->reads.end());
    n->reads = std::move(reads);
  }
  return SBlockRealize(/*iter_values=*/iter_values, /*predicate=*/predicate,
                       /*block=*/block);
}

/*!
 * \brief Generate the init stmt for the outer block
 * \param block The original block with init.
 * \param inner_realize The block realize of the inner block after blockize.
 * \param loops The inner loops after blockize.
 * \return The subtree of the init block and its outer loops.
 */
Stmt GenerateOuterInit(const Stmt& block_init, const SBlockRealize& inner_realize,
                       const std::vector<const ForObj*>& loops, ffi::String block_name) {
  const SBlock& inner_block = inner_realize->block;
  ffi::Map<Var, Var> subst_map;
  // Step 1: Create new block vars for the block inside the init stmt of outer block
  // A iter is used in the block if
  // 1) It is data parallel
  // 2) It is used in the original init block
  ffi::Array<IterVar> iter_vars;
  ffi::Array<PrimExpr> iter_values;
  TVM_FFI_ICHECK_EQ(inner_block->iter_vars.size(), inner_realize->iter_values.size());
  int n = inner_block->iter_vars.size();
  iter_vars.reserve(n);
  iter_values.reserve(n);
  for (int i = 0; i < n; ++i) {
    const IterVar& old_iter_var = inner_block->iter_vars[i];
    const PrimExpr& iter_value = inner_realize->iter_values[i];
    auto walkfn = [target =
                       old_iter_var->var.get()](const Var& var) -> ffi::Expected<ffi::WalkResult> {
      return var.get() == target ? ffi::WalkResult::Interrupt(ffi::VisitInterrupt(var))
                                 : ffi::WalkResult::Advance();
    };
    if (old_iter_var->iter_type == IterVarType::kDataPar &&
        ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(block_init, walkfn).has_value()) {
      ffi::ObjectPtr<IterVarObj> new_iter_var = ffi::make_object<IterVarObj>(*old_iter_var.get());
      new_iter_var->var = new_iter_var->var.CopyWithSuffix("_init");
      subst_map.Set(old_iter_var->var, new_iter_var->var);
      iter_vars.push_back(IterVar(new_iter_var));
      iter_values.push_back(iter_value);
    }
  }
  // Step 2: Generate the block inside init stmt of outer block
  Stmt stmt = SBlockRealize(
      /*iter_values=*/iter_values,
      /*predicate=*/inner_realize->predicate,
      /*block=*/
      SBlock(/*iter_vars=*/iter_vars,
             /*reads=*/{},
             /*writes=*/inner_block->writes,
             /*name_hint=*/block_name,
             /*body=*/block_init,
             /*init=*/std::nullopt));
  // Step 3. Create the loop nest on top of the block
  for (const ForObj* loop : loops) {
    bool is_init_loop = false;
    auto walkfn = [target = loop->var.get()](const Var& var) -> ffi::Expected<ffi::WalkResult> {
      return var.get() == target ? ffi::WalkResult::Interrupt(ffi::VisitInterrupt(var))
                                 : ffi::WalkResult::Advance();
    };
    for (const PrimExpr& init_binding : iter_values) {
      if (ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(init_binding, walkfn).has_value()) {
        is_init_loop = true;
        break;
      }
    }
    if (is_init_loop) {
      ffi::ObjectPtr<ForObj> new_loop = ffi::make_object<ForObj>(*loop);
      new_loop->var = loop->var.CopyWithSuffix("");
      new_loop->body = std::move(stmt);
      subst_map.Set(loop->var, new_loop->var);
      stmt = For(new_loop);
    }
  }
  // Step 4: Substitute the iter vars and loop vars
  auto f_substitute = [&subst_map](
                          const Var& var,
                          TVMFFIDefRegionKind kind) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (kind != kTVMFFIDefRegionKindNone) return ffi::Unchanged();
    if (auto repl = subst_map.Get(var)) return ffi::Any(*std::move(repl));
    return ffi::Unchanged();
  };
  return ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(stmt, f_substitute).as_or_throw<Stmt>();
}

/*!
 * \brief Replace variables in the stmt, do simplification and track block replacement
 * \param stmt The stmt to be substituted.
 * \param sub The substitution map.
 * \param block_sref_reuse The block substitution happens during the substitution.
 * \param analyzer The analyzer for arithmetic simplification.
 * \return The substituted stmt.
 */
Stmt ReplaceAndSimplify(const Stmt& stmt, const ffi::Map<Var, PrimExpr>& sub,
                        ffi::Map<SBlock, SBlock>* block_sref_reuse, sym::AnalyzerObj* analyzer) {
  struct Replacer : public StmtExprMutator {
   public:
    using StmtExprMutator::Mutate;
    using StmtExprMutator::Mutate_;

    explicit Replacer(const ffi::Map<Var, PrimExpr>& sub,
                      ffi::Map<SBlock, SBlock>* block_sref_reuse, sym::AnalyzerObj* analyzer)
        : block_sref_reuse_(block_sref_reuse), analyzer_(analyzer) {
      for (const auto& [var, replacement] : sub) VarRemapSet(var, replacement);
    }

    UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) final {
      auto result = StmtExprMutator::Mutate(value, inplace_mode);
      if (!value.as<ExprObj>() || result.UnchangedOrSameAs(value)) return result;
      auto replacement = std::move(result).ValueOrUnchanged(value);
      if (auto prim_result = replacement.as<PrimExpr>()) {
        return ffi::Any(analyzer_->Simplify(prim_result.value()));
      }
      return replacement;
    }

    UnchangedOr<Stmt> Mutate_(const SBlockObj* op, InplaceMode inplace_mode) final {
      SBlock src = ffi::GetRef<SBlock>(op);
      SBlock tgt = StmtExprMutator::Mutate_(op, inplace_mode)
                       .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                       .as_or_throw<SBlock>();
      if (!src.same_as(tgt)) {
        block_sref_reuse_->Set(src, tgt);
      }
      return tgt;
    }

    ffi::Map<SBlock, SBlock>* block_sref_reuse_;
    sym::AnalyzerObj* analyzer_;
  };
  return ffi::make_object<Replacer>(sub, block_sref_reuse, analyzer)
      ->Mutate(stmt)
      .ValueOrUnchanged(stmt);
}

/*!
 * \brief Relax the variables for the given regions
 * \param regions The regions to be relaxed.
 * \param dom_map The variables to be relaxed
 * \return The relaxed regions
 */
ffi::Array<TensorRegion> EvalSetRegions(const ffi::Array<TensorRegion>& regions,
                                        const ffi::Map<Var, sym::IntSet>& dom_map) {
  ffi::Array<TensorRegion> results;
  results.reserve(regions.size());
  for (const TensorRegion& tensor_region : regions) {
    const TensorVar& tensor = tensor_region->source.as_or_throw<tvm::tirx::TensorVar>();
    ffi::Array<sym::IntSet> relaxed = sym::EvalSet(tensor_region->region, dom_map);
    TVM_FFI_ICHECK_EQ(relaxed.size(), tensor.shape().size());
    int ndim = tensor.shape().size();
    ffi::Array<Range> new_region;
    new_region.reserve(ndim);
    for (int i = 0; i < ndim; ++i) {
      new_region.push_back(relaxed[i].CoverRange(RangeFromExtent(tensor.shape()[i])).value());
    }
    results.push_back(TensorRegion(tensor, new_region, TensorRegionType()));
  }
  return results;
}

/*!
 * \brief Get the union of the given regions
 * \param regions The input regions for the union.
 * \return The union regions
 */
ffi::Array<TensorRegion> UnionRegions(const ffi::Array<TensorRegion>& regions) {
  typedef std::vector<ffi::Array<sym::IntSet>> ranges_t;
  std::unordered_map<TensorVar, ranges_t, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> intset_map;
  for (const TensorRegion& tensor_region : regions) {
    const TensorVar& tensor = tensor_region->source.as_or_throw<tvm::tirx::TensorVar>();
    if (intset_map.find(tensor) == intset_map.end()) {
      intset_map[tensor] = {tensor.shape().size(), ffi::Array<sym::IntSet>()};
    }
    std::vector<ffi::Array<sym::IntSet>> dim_range(tensor.shape().size(),
                                                   ffi::Array<sym::IntSet>());
    for (size_t dim = 0; dim < tensor.shape().size(); ++dim) {
      intset_map[tensor][dim].push_back(sym::IntSet::FromRange(tensor_region->region[dim]));
    }
  }
  ffi::Array<TensorRegion> results;
  for (const auto& it : intset_map) {
    const TensorVar& tensor = it.first;
    ffi::Array<Range> regions;
    for (size_t dim = 0; dim < tensor.shape().size(); ++dim) {
      const sym::IntSet intset = sym::Union(it.second[dim]);
      regions.push_back({intset.min(), intset.max() + 1});
    }
    results.push_back(TensorRegion(tensor, regions, TensorRegionType()));
  }
  return results;
}

/*!
 * \brief Create the loop nest on top of the given stmt.
 * \param stmt The stmt to be wrapped.
 * \param loops The loop nests
 * \return The wrapped stmt.
 */
Stmt MakeLoopNest(Stmt stmt, const std::vector<const ForObj*>& loops) {
  for (const ForObj* loop : loops) {
    ffi::ObjectPtr<ForObj> new_loop = ffi::make_object<ForObj>(*loop);
    new_loop->body = std::move(stmt);
    stmt = For(new_loop);
  }
  return stmt;
}

SBlockRealize BlockizeImpl(const ScheduleState& self, const StmtSRef& loop_sref,
                           ffi::Map<SBlock, SBlock>* block_sref_reuse, sym::AnalyzerObj* analyzer,
                           bool preserve_unit_iters) {
  TVM_SREF_TO_FOR(loop_sref);
  // Step 1: Check and get the only block under `loop`.
  SBlockRealize block_realize = CheckGetSingleChildBlockRealizeOnSRefTree(self, loop_sref);
  SBlock block = block_realize->block;
  StmtSRef block_sref = self->stmt2ref.at(block.get());
  // Step 2: Derive subspace division
  std::vector<const ForObj*> loops;
  ffi::Array<ffi::Array<sym::IterMark>> division =
      SubspaceDivide(block_realize, block_sref, loop_sref, &loops, analyzer, preserve_unit_iters);
  if (division.empty()) {
    throw MakeScheduleError<SubspaceNotDivisibleError>(self->mod, ffi::GetRef<For>(loops.back()),
                                                       block);
  }
  PrimExpr outer_predicate = division.back()[0]->extent;
  PrimExpr inner_predicate = division.back()[1]->extent;
  // Step 3. Derive block bindings for both outer and inner block.
  ffi::Array<IterVar> outer_iter_vars;
  ffi::Array<IterVar> inner_iter_vars;
  ffi::Array<PrimExpr> outer_bindings;
  ffi::Array<PrimExpr> inner_bindings;
  ffi::Map<Var, PrimExpr> block_var_subst =                  //
      DeriveBlockBinding(block->iter_vars, division,         //
                         &outer_iter_vars, &outer_bindings,  //
                         &inner_iter_vars, &inner_bindings,  //
                         preserve_unit_iters);
  // Step 4: Do var substitution to adjust to the new block bindings
  ffi::Map<Var, sym::IntSet> inner_iter_dom;
  for (const IterVar& iter : inner_iter_vars) {
    inner_iter_dom.Set(iter->var, sym::IntSet::FromRange(iter->dom.value()));
    {
      auto context = analyzer->GetConstraintContext();
      context->Append(sym::RangeConstraint(iter->var, iter->dom.value()));
      analyzer->SyncConstraintContext(context);
    }
  }
  SBlock block_subst =
      ReplaceAndSimplify(block, block_var_subst, block_sref_reuse, analyzer).as_or_throw<SBlock>();
  // Step 5: Generate the inner block. The write regions of the inner blocks will be reduction if
  // 1. The original block has init stmt.
  // 2. There are outer reduction iter vars.
  bool has_outer_reduction = false;
  if (block_subst->init.has_value()) {
    for (const IterVar& iter_var : outer_iter_vars) {
      if (iter_var->iter_type == kCommReduce) {
        has_outer_reduction = true;
        break;
      }
    }
  }
  SBlockRealize inner_realize = GenerateInner(/*is_write_reduction=*/has_outer_reduction,
                                              /*iter_vars=*/inner_iter_vars,
                                              /*iter_values*/ inner_bindings,
                                              /*predicate=*/inner_predicate,
                                              /*block=*/block_subst);
  block_sref_reuse->Set(block, inner_realize->block);
  // Step 6: Generate the outer block.
  return SBlockRealize(
      /*iter_values=*/std::move(outer_bindings),
      /*predicate=*/std::move(outer_predicate),
      /*block=*/
      SBlock(/*iter_vars=*/std::move(outer_iter_vars),
             /*reads=*/EvalSetRegions(block_subst->reads, inner_iter_dom),
             /*writes=*/EvalSetRegions(block_subst->writes, inner_iter_dom),
             /*name_hint=*/block_subst->name_hint + "_o",
             /*body=*/MakeLoopNest(inner_realize, loops),
             /*init=*/
             block_subst->init.has_value()  //
                 ? ffi::Optional<SeqStmt>(
                       SeqStmt(GenerateOuterInit(block_subst->init.value(), inner_realize, loops,
                                                 block_subst->name_hint + "_init")))
                 : ffi::Optional<SeqStmt>(std::nullopt)));
}

StmtSRef Blockize(ScheduleState self, const StmtSRef& loop_sref, bool preserve_unit_iters) {
  sym::Analyzer analyzer;
  ffi::Map<SBlock, SBlock> block_sref_reuse;
  SBlockRealize blockized =
      BlockizeImpl(self, loop_sref, &block_sref_reuse, analyzer.get(), preserve_unit_iters);
  self->Replace(loop_sref, blockized, block_sref_reuse);
  StmtSRef result = self->stmt2ref.at(blockized->block.get());
  StmtSRef scope_root = GetScopeRoot(self, result, /*require_stage_pipeline=*/false);
  bool scope_block_affine_binding = self->IsAffineBlockBinding(scope_root);
  self->UpdateScopeSBlockInfo(GetSBlockRealize(self, scope_root));
  self->block_info[scope_root].affine_binding = scope_block_affine_binding;
  return result;
}

SBlockRealize BlockizeBlocks(const ScheduleState& self, const ffi::Array<StmtSRef>& block_srefs,
                             const StmtSRef& lca, ffi::Map<SBlock, SBlock>* block_sref_reuse,
                             bool preserve_unit_iters) {
  ffi::Array<Stmt> seq_body;
  ffi::Optional<PrimExpr> outer_predicate;
  ffi::Array<IterVar> outer_iter_vars;
  ffi::Array<PrimExpr> outer_bindings;
  ffi::Array<TensorRegion> read_regions;
  ffi::Array<TensorRegion> write_regions;
  std::string outer_block_name = "outer_";
  ffi::Map<Var, Var> loop_var_subst;
  sym::Analyzer analyzer;
  for (const auto& block_sref : block_srefs) {
    auto block_realize = GetSBlockRealize(self, block_sref);
    auto block = block_realize->block;
    // Step 1: Derive subspace division
    std::vector<const ForObj*> loops;
    ffi::Array<ffi::Array<sym::IterMark>> division = SubspaceDivide(
        block_realize, block_sref, lca, &loops, analyzer.get(), preserve_unit_iters, true);
    if (division.empty()) {
      throw MakeScheduleError<SubspaceNotDivisibleError>(self->mod, ffi::GetRef<For>(loops.back()),
                                                         block);
    }
    outer_predicate = division.back()[0]->extent;
    PrimExpr inner_predicate = division.back()[1]->extent;
    // Step 2. Derive block bindings for both outer and inner block.
    ffi::Array<IterVar> inner_iter_vars;
    ffi::Array<PrimExpr> inner_bindings;
    ffi::Map<Var, PrimExpr> block_var_subst =                  //
        DeriveBlockBinding(block->iter_vars, division,         //
                           &outer_iter_vars, &outer_bindings,  //
                           &inner_iter_vars, &inner_bindings,  //
                           preserve_unit_iters, !outer_iter_vars.empty());
    // Step 3: Do var substitution to adjust to the new block bindings
    for (size_t i = 0; i < outer_iter_vars.size(); ++i) {
      if (outer_bindings[i].as<Var>()) {
        loop_var_subst.Set(outer_bindings[i].as_or_throw<Var>(), outer_iter_vars[i]->var);
      }
    }
    auto f_substitute =
        [&loop_var_subst](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
      if (auto repl = loop_var_subst.Get(var)) return ffi::Any(*std::move(repl));
      return ffi::Unchanged();
    };
    ffi::Map<Var, sym::IntSet> inner_iter_dom;
    for (const IterVar& iter : inner_iter_vars) {
      PrimExpr min =
          ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(iter->dom.value()->begin, f_substitute)
              .as_or_throw<PrimExpr>();
      PrimExpr extent =
          ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(iter->dom.value()->extent, f_substitute)
              .as_or_throw<PrimExpr>();
      Range dom = Range::FromMinExtent(min, extent);
      inner_iter_dom.Set(iter->var, sym::IntSet::FromRange(dom));
      {
        auto context = analyzer->GetConstraintContext();
        context->Append(sym::RangeConstraint(iter->var, dom));
        analyzer->SyncConstraintContext(context);
      }
    }
    SBlock block_subst =
        ReplaceAndSimplify(block, block_var_subst, block_sref_reuse, analyzer.get())
            .as_or_throw<SBlock>();
    auto reads = EvalSetRegions(block_subst->reads, inner_iter_dom);
    auto writes = EvalSetRegions(block_subst->writes, inner_iter_dom);
    read_regions.insert(read_regions.end(), reads.begin(), reads.end());
    write_regions.insert(write_regions.end(), writes.begin(), writes.end());
    outer_block_name += block_subst->name_hint + "_";
    // Step 4: Generate the inner block. No reduction iter vars allowed for the outer loops.
    bool has_outer_reduction = false;
    if (block_subst->init.has_value()) {
      for (const IterVar& iter_var : outer_iter_vars) {
        if (iter_var->iter_type == kCommReduce) {
          has_outer_reduction = true;
          break;
        }
      }
    }
    TVM_FFI_ICHECK(has_outer_reduction == false)
        << "No reduction iter vars allowed for the outer loops when blockize multiple blocks";
    SBlockRealize inner_realize = GenerateInner(/*is_write_reduction=*/has_outer_reduction,
                                                /*iter_vars=*/inner_iter_vars,
                                                /*iter_values*/ inner_bindings,
                                                /*predicate=*/inner_predicate,
                                                /*block=*/block_subst);
    block_sref_reuse->Set(block, inner_realize->block);
    Stmt stmt = inner_realize;
    for (const ForObj* loop : loops) {
      ffi::ObjectPtr<ForObj> new_loop = ffi::make_object<ForObj>(*loop);
      new_loop->body = std::move(stmt);
      new_loop->end =
          new_loop->begin +
          ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(
              Range::GetExtentFromBeginEnd(new_loop->begin, new_loop->end), f_substitute)
              .as_or_throw<PrimExpr>();
      stmt = For(new_loop);
    }
    seq_body.push_back(stmt);
  }
  // Step 5: Generate the outer block.
  return SBlockRealize(
      /*iter_values=*/std::move(outer_bindings),
      /*predicate=*/std::move(outer_predicate).value(),
      /*block=*/
      SBlock(/*iter_vars=*/std::move(outer_iter_vars),
             /*reads=*/UnionRegions(read_regions),
             /*writes=*/UnionRegions(write_regions),
             /*name_hint=*/outer_block_name,
             /*body=*/SeqStmt(seq_body),
             /*init=*/std::nullopt));
}

class BlockizeRewriter : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) override {
    if (value.as<ExprObj>()) return ffi::Unchanged();
    return StmtExprMutator::Mutate(value, inplace_mode);
  }

  static Stmt Rewrite(const StmtSRef& lca, const ffi::Array<StmtSRef>& blocks,
                      const SBlockRealize& blockized) {
    auto rewriter = ffi::make_object<BlockizeRewriter>(lca, blocks, blockized);
    return rewriter->Mutate(ffi::GetRef<Stmt>(lca->stmt))
        .ValueOrUnchanged(ffi::GetRef<Stmt>(lca->stmt));
  }

  explicit BlockizeRewriter(const StmtSRef& lca, const ffi::Array<StmtSRef>& blocks,
                            const SBlockRealize& blockized)
      : lca_(lca), blocks_(blocks), blockized_(blockized) {}

 private:
  Stmt RewriteSeq(const Stmt& stmt) {
    const SeqStmtObj* seq = stmt.as<SeqStmtObj>();
    TVM_FFI_ICHECK(seq) << "Target blocks must not be nested with each other!";
    int idx_start = -1;
    int last_found_idx = -1;
    size_t cur_idx = 0;
    ffi::Array<Stmt> new_seq;
    for (const Stmt& it : seq->seq) {
      target_in_ = false;
      Stmt stmt = StmtExprMutator::Mutate(ffi::AnyView(it), InplaceMode::kDisallow)
                      .ValueOrUnchanged(it)
                      .as_or_throw<Stmt>();
      if (target_in_) {
        if (idx_start == -1) {
          idx_start = cur_idx;
          new_seq.push_back(blockized_);
        } else {
          TVM_FFI_ICHECK_EQ(last_found_idx, cur_idx - 1) << "Target blocks must be consecutive!";
        }
        last_found_idx = cur_idx;
      } else {
        new_seq.push_back(it);
      }
      ++cur_idx;
    }
    return SeqStmt(new_seq, seq->loc);
  }

  UnchangedOr<Stmt> Mutate_(const ForObj* loop, InplaceMode inplace_mode) final {
    if (loop == lca_->stmt) {
      return For(loop->var, loop->begin, loop->end, loop->step, loop->kind, RewriteSeq(loop->body),
                 loop->annotations, loop->loc);
    }
    return StmtExprMutator::Mutate_(loop, inplace_mode);
  }

  UnchangedOr<Stmt> Mutate_(const SBlockObj* block, InplaceMode inplace_mode) final {
    if (block == lca_->stmt) {
      return SBlock(block->iter_vars, block->reads, block->writes, block->name_hint,
                    RewriteSeq(block->body), block->init, block->alloc_tensors,
                    block->match_tensors, block->annotations, block->loc);
    }
    for (const StmtSRef& block_sref : blocks_) {
      if (block_sref->stmt == block) {
        target_in_ = true;
        break;
      }
    }
    return ffi::Unchanged();
  }

  StmtSRef lca_;
  ffi::Array<StmtSRef> blocks_;
  SBlockRealize blockized_;
  bool target_in_ = false;
};

StmtSRef Blockize(ScheduleState self, const ffi::Array<StmtSRef>& blocks,
                  bool preserve_unit_iters) {
  ffi::Map<SBlock, SBlock> block_sref_reuse;
  auto lca = GetSRefLowestCommonAncestor(blocks);
  SBlockRealize blockized =
      BlockizeBlocks(self, blocks, lca, &block_sref_reuse, preserve_unit_iters);
  auto new_root = BlockizeRewriter::Rewrite(lca, blocks, blockized);
  self->Replace(lca, new_root, block_sref_reuse);
  StmtSRef result = self->stmt2ref.at(blockized->block.get());
  StmtSRef scope_root = GetScopeRoot(self, result, /*require_stage_pipeline=*/false);
  self->UpdateScopeSBlockInfo(GetSBlockRealize(self, scope_root));
  return result;
}

void Tensorize(ScheduleState self, const StmtSRef& sref, const TensorIntrin& intrin,
               bool preserve_unit_iters) {
  // Step 1: Blockize the subtree rooted at the given loop if needed
  ffi::Optional<SBlock> old_block = std::nullopt;
  SBlockRealize block_realize = [&]() -> SBlockRealize {
    if (sref->stmt->IsInstance<SBlockObj>()) {
      SBlockRealize result = GetSBlockRealize(self, sref);
      old_block = result->block;
      return result;
    } else if (sref->stmt->IsInstance<ForObj>()) {
      sym::Analyzer analyzer;
      ffi::Map<SBlock, SBlock> block_sref_reuse;
      return BlockizeImpl(self, sref, &block_sref_reuse, analyzer.get(), preserve_unit_iters);
    } else {
      TVM_FFI_THROW(TypeError) << "Tensorize only support For or SBlock, but gets: "
                               << ffi::GetRef<Stmt>(sref->stmt);
      throw;
    }
  }();

  sym::Analyzer analyzer;
  Function intrin_desc = s_tir::StmtSimplify(intrin->desc, analyzer);
  Function intrin_impl = DeepCopy(intrin->impl);

  int index_dtype_bits = -1;
  auto f_update_max_dtype_bits_from_region = [&](const ffi::Array<TensorRegion>& tensor_regions) {
    for (const TensorRegion& tensor_region : tensor_regions) {
      for (const auto& range : tensor_region->region) {
        index_dtype_bits = std::max(index_dtype_bits, range->begin.ty().bits());
      }
    }
  };
  f_update_max_dtype_bits_from_region(block_realize->block->reads);
  f_update_max_dtype_bits_from_region(block_realize->block->writes);
  TVM_FFI_ICHECK(index_dtype_bits > 0);
  intrin_impl = ffi::make_object<IndexDataTypeNormalizer>(PrimType::Int(index_dtype_bits))
                    ->Rewrite(intrin_impl);
  // Step 2: Structural pattern matching
  TensorizeComparator comparator(self->mod, /*assert_mode=*/true);
  TVM_FFI_CHECK(intrin_desc->body.has_value(), ValueError)
      << "A tensor intrinsic description must have a body";
  TVM_FFI_CHECK_EQ(intrin_desc->body.value()->size(), 1, ValueError)
      << "A tensor intrinsic description must contain a single root block";
  comparator.Dispatch(block_realize, intrin_desc->body.value()->seq[0]);
  // Step 3: Prepare necessary mapping
  // 1) TensorVar mapping from intrin impl tensors to intrin desc tensors.
  // 2) TensorVar mapping from intrin impl tensors to tensors in the current AST.
  // 3) Mapping impl tensors to their accessed regions.
  std::unordered_map<TensorVar, TensorVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> impl2desc;
  TVM_FFI_ICHECK_EQ(intrin_desc->params.size(), intrin_impl->params.size());
  for (int i = 0, n = intrin_desc->params.size(); i < n; ++i) {
    TensorVar desc = intrin_desc->params[i].as_or_throw<tvm::tirx::TensorVar>();
    TensorVar impl = intrin_impl->params[i].as_or_throw<tvm::tirx::TensorVar>();
    impl2desc.insert_or_assign(impl, desc);
  }
  std::unordered_map<TensorVar, TensorVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> impl2cur;
  for (const auto& pair : impl2desc) {
    const TensorVar& impl = pair.first;
    const TensorVar& desc = pair.second;
    TVM_FFI_ICHECK(comparator.rhs_tensor_map_.count(desc));
    impl2cur.insert_or_assign(impl, comparator.rhs_tensor_map_.at(desc));
  }
  std::unordered_map<TensorVar, ffi::Array<Range>, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      impl2region;
  TVM_FFI_ICHECK(intrin_impl->body.has_value());
  TVM_FFI_ICHECK_EQ(intrin_impl->body.value()->size(), 1);
  SBlock impl_block = intrin_impl->body.value()->seq[0].as_or_throw<SBlockRealize>()->block;
  for (const TensorRegion& read : impl_block->reads) {
    impl2region.emplace(read->source.as_or_throw<tvm::tirx::TensorVar>(), read->region);
  }
  for (const TensorRegion& write : impl_block->writes) {
    impl2region.emplace(write->source.as_or_throw<tvm::tirx::TensorVar>(), write->region);
  }
  // Step 4: Create MatchTensorRegion for the params of the impl function of the tensor
  // intrin to make them subregions of the tensor in the original IR.
  ffi::Array<MatchTensorRegion> match_tensor_regions;
  match_tensor_regions.reserve(intrin_impl->params.size());
  for (int i = 0, n = intrin_impl->params.size(); i < n; ++i) {
    TensorVar impl = intrin_impl->params[i].as_or_throw<tvm::tirx::TensorVar>();
    const TensorVar& cur = impl2cur.at(impl);
    const ffi::Array<Range>& old_region = impl2region.at(impl);
    const std::vector<PrimExpr>& indices_base = comparator.tensor_indices_.at(cur);
    int offset = static_cast<int>(indices_base.size()) - static_cast<int>(old_region.size());
    TVM_FFI_ICHECK(offset >= 0);
    ffi::Array<Range> new_region;
    new_region.reserve(cur.shape().size());
    for (int i = 0; i < offset; i++) {
      PrimExpr min = indices_base[i];
      PrimExpr extent = IntImm(min.ty(), 1);
      new_region.push_back(Range::FromMinExtent(min, extent));
    }
    for (int i = 0; i < static_cast<int>(old_region.size()); i++) {
      PrimExpr min = indices_base[i + offset];
      PrimExpr extent = cast(old_region[i]->extent, min.ty());
      new_region.push_back(Range::FromMinExtent(min, extent));
    }
    match_tensor_regions.push_back(
        MatchTensorRegion(impl, TensorRegion(cur, new_region, TensorRegionType())));
  }
  // Step 5: Replace the subtree in the original IR with the tensor intrin impl.
  {
    SBlockObj* block = block_realize.CopyOnWrite()->block.CopyOnWrite();
    block->body = impl_block->body;
    block->match_tensors = std::move(match_tensor_regions);
    for (const auto& [key, val] : impl_block->annotations) {
      if (block->annotations.count(key) && !ffi::AnyEqual()(block->annotations[key], val)) {
        LOG(WARNING) << "Conflict of annotation \"" << key << "\". Tensor intrinsic and schedule "
                     << "has different values : " << block->annotations[key] << " vs " << val << " "
                     << "The value from tensor intrinsic is skipped.";
        continue;
      }
      block->annotations.Set(key, val);
    }
  }
  if (old_block.has_value()) {
    self->Replace(sref, block_realize->block, {{old_block.value(), block_realize->block}});
  } else {
    self->Replace(sref, block_realize, {});
  }
  // Step 6: Update the cached flags.
  StmtSRef result = self->stmt2ref.at(block_realize->block.get());
  StmtSRef scope_root = GetScopeRoot(self, result, /*require_stage_pipeline=*/false);
  self->UpdateScopeSBlockInfo(scope_root->StmtAs<SBlockObj>()->body);
}

/******** InstructionKind Registration ********/

struct BlockizeTraits : public UnpackedInstTraits<BlockizeTraits> {
  static constexpr const char* kName = "Blockize";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  static SBlockRV UnpackedApplyToSchedule(Schedule sch, ffi::ObjectRef target,
                                          bool preserve_unit_iters) {
    if (auto loop = target.as<LoopRV>()) {
      return sch->Blockize(loop.value(), preserve_unit_iters);
    } else if (auto blocks = target.as<ffi::Array<SBlockRV>>()) {
      return sch->Blockize(blocks.value(), preserve_unit_iters);
    }
    TVM_FFI_THROW(TypeError) << "expect Loop or list of SBlocks, but gets:" << target->GetTypeKey();
    TVM_FFI_UNREACHABLE();
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::ObjectRef target,
                                      bool preserve_unit_iters) {
    PythonAPICall py("blockize");
    py.Input("target", target);
    py.Input("preserve_unit_iters", preserve_unit_iters);
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

struct TensorizeTraits : public UnpackedInstTraits<TensorizeTraits> {
  static constexpr const char* kName = "Tensorize";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, ffi::ObjectRef block_or_loop_rv,
                                      ffi::String intrin, bool preserve_unit_iters) {
    if (auto block = block_or_loop_rv.as<SBlockRV>()) {
      sch->Tensorize(block.value(), intrin, preserve_unit_iters);
    } else if (auto loop = block_or_loop_rv.as<LoopRV>()) {
      sch->Tensorize(loop.value(), intrin, preserve_unit_iters);
    } else {
      TVM_FFI_THROW(TypeError) << "Expected SBlock or Loop, but gets: "
                               << block_or_loop_rv->GetTypeKey();
    }
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block_or_loop_rv,
                                      ffi::String intrin, bool preserve_unit_iters) {
    PythonAPICall py("tensorize");
    py.Input("block_or_loop", block_or_loop_rv);
    py.Input("tensor_intrin", intrin);
    py.Input("preserve_unit_iters", preserve_unit_iters);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

TVM_FFI_STATIC_INIT_BLOCK() {
  RegisterInstructionKind<BlockizeTraits>();
  RegisterInstructionKind<TensorizeTraits>();
}

}  // namespace s_tir
}  // namespace tvm
