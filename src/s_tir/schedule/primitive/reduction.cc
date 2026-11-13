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
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/te/operation.h>

#include "../utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::prim;
using namespace tvm::tirx;

/*!
 * \brief A helper class to create a new scope that contains decomposed init body
 * and replaced old reduction block.
 */
class DecomposeReductionBlockReplacer : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) override {
    if (value.as<ExprObj>()) return ffi::Unchanged();
    return StmtExprMutator::Mutate(value, inplace_mode);
  }

  /*!
   * \brief The open interface to users to call the helper class
   * \param old_scope_root The original block scope before decomposition
   * \param target_loop The loop we insert the decomposed init body before
   * \param decompose_body The decomposed init body
   * \param old_reduction_block The reduction block we want to decompose
   * \return The new block scope and the updated reduction block
   */
  static std::pair<SBlock, SBlock> Replace(SBlock old_scope_root, For target_loop,
                                           Stmt decomposed_body, SBlock old_reduction_block) {
    auto replacer = ffi::make_object<DecomposeReductionBlockReplacer>(
        std::move(target_loop), std::move(decomposed_body), std::move(old_reduction_block));
    SBlock new_scope_root = replacer->Mutate(old_scope_root, InplaceMode::kAllow)
                                .ValueOrUnchanged(std::move(old_scope_root))
                                .as_or_throw<SBlock>();
    return std::make_pair(std::move(new_scope_root), replacer->new_reduction_block_.value());
  }

  explicit DecomposeReductionBlockReplacer(For target_loop, Stmt decomposed_body,
                                           SBlock old_reduction_block)
      : target_loop_(std::move(target_loop)),
        decomposed_body_(std::move(decomposed_body)),
        old_reduction_block_(std::move(old_reduction_block)) {}

 private:
  UnchangedOr<Stmt> Mutate_(const ForObj* loop, InplaceMode inplace_mode) final {
    Stmt mutated_stmt =
        StmtExprMutator::Mutate_(loop, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(loop));
    if (loop == target_loop_.get()) {
      return SeqStmt({decomposed_body_, mutated_stmt});
    } else {
      return mutated_stmt;
    }
  }

  UnchangedOr<Stmt> Mutate_(const SBlockObj* block, InplaceMode inplace_mode) final {
    if (block == old_reduction_block_.get()) {
      auto p_new_block = ffi::make_object<SBlockObj>(*block);
      p_new_block->name_hint = p_new_block->name_hint + "_update";
      p_new_block->init = std::nullopt;
      // Add write regions back to read regions in update block.
      ffi::Array<TensorRegion> new_reads;
      std::unordered_set<const VarObj*> read_tensors;
      for (const TensorRegion& read_access : block->reads) {
        read_tensors.insert(read_access->source.as_or_throw<tvm::tirx::TensorVar>().get());
      }
      for (const TensorRegion& write_access : block->writes) {
        if (read_tensors.find(write_access->source.as_or_throw<tvm::tirx::TensorVar>().get()) ==
            read_tensors.end()) {
          new_reads.push_back(write_access);
        }
      }
      for (const TensorRegion& read_access : block->reads) {
        new_reads.push_back(read_access);
      }
      p_new_block->reads = new_reads;
      new_reduction_block_ = SBlock(p_new_block);
      return new_reduction_block_.value();
    } else {
      return StmtExprMutator::Mutate_(block, inplace_mode);
    }
  }

  For target_loop_;
  Stmt decomposed_body_;
  SBlock old_reduction_block_;
  ffi::Optional<SBlock> new_reduction_block_;
};

class LoopHeightError : public ScheduleErrorContextObj {
 public:
  static void CheckLoopHigherThanReduceLoops(const IRModule& mod, const SBlockObj* block,
                                             const SBlockRealizeObj* realize,
                                             const ffi::Array<StmtSRef>& loops,
                                             const StmtSRef& loop_sref) {
    for (int i = 0, n = block->iter_vars.size(); i < n; ++i) {
      // For each block var of type kCommReduce, check its binding
      const IterVar& iter_var = block->iter_vars[i];
      const PrimExpr& binding = realize->iter_values[i];
      if (iter_var->iter_type != IterVarType::kCommReduce) {
        continue;
      }
      for (const StmtSRef& higher_loop : loops) {
        // Only check loops not lower than the target loop
        if (higher_loop.same_as(loop_sref)) {
          break;
        }
        // loop_var of a higher loop shouldn't contain loop var
        const Var& loop_var = higher_loop->StmtAs<ForObj>()->var;
        auto walkfn = [v = loop_var.get()](const Var& var) -> ffi::Expected<ffi::WalkResult> {
          return var.get() == v ? ffi::WalkResult::Interrupt(ffi::VisitInterrupt(var))
                                : ffi::WalkResult::Advance();
        };
        if (ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(binding, walkfn).has_value()) {
          const ForObj* loop = TVM_SREF_TO_FOR(loop_sref);
          throw MakeScheduleError<LoopHeightError>(mod, ffi::GetRef<For>(loop),
                                                   ffi::GetRef<SBlock>(block));
        }
      }
    }
  }

  explicit LoopHeightError(IRModule mod, For loop, SBlock block)
      : mod_(std::move(mod)), loop_(std::move(loop)), block_(std::move(block)) {}

  ffi::String FastErrorString() const final {
    return "ScheduleError: decompose_reduction expect the loop to be higher than all the loops "
           "related to reduce block var";
  }

  ffi::String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "ScheduleError: decompose_reduction expect the loop {0} to be higher than all the loops "
          "related to reduce block var of block {1}";
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {loop_, block_}; }

  IRModule mod_;
  For loop_;
  SBlock block_;
};

PrimExpr RewriteInitPredicate(PrimExpr pred,
                              const std::unordered_set<const VarObj*>& discarded_loops) {
  if (IsOne(pred)) return IntImm::Bool(true);
  if (const auto* and_node = pred.as<AndObj>()) {
    return RewriteInitPredicate(and_node->a, discarded_loops) &&
           RewriteInitPredicate(and_node->b, discarded_loops);
  }
  auto uses_discarded_loop = [&discarded_loops](const VarObj* var) {
    return discarded_loops.count(var);
  };
  auto walkfn = [&](const Var& var) -> ffi::Expected<ffi::WalkResult> {
    return uses_discarded_loop(var.get()) ? ffi::WalkResult::Interrupt(ffi::VisitInterrupt(var))
                                          : ffi::WalkResult::Advance();
  };
  return ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(pred, walkfn).has_value()
             ? IntImm::Bool(true)
             : pred;
}

StmtSRef DecomposeReduction(ScheduleState self, const StmtSRef& block_sref,
                            const StmtSRef& loop_sref) {
  /*!
   *  Check
   *    - block is a reduction block
   *    - loop is not lower than all the loops related to reduce block var
   *  Mutate
   *    - generate loops related to data par block vars
   *    - generate corresponding init block and update block
   */
  // Condition Checks and Information Collection
  const SBlockObj* block = TVM_SREF_TO_SBLOCK(block_sref);
  const ForObj* loop = TVM_SREF_TO_FOR(loop_sref);
  // Get the outer loops from high to low
  ffi::Array<StmtSRef> loops = GetLoops(block_sref);
  const SBlockRealizeObj* realize = GetSBlockRealize(self, block_sref).get();
  StmtSRef scope_root_sref = GetScopeRoot(self, block_sref,
                                          /*require_stage_pipeline=*/false);
  if (self->enable_check) {
    // Cond 0. Check loop_sref is an ancestor of block_sref
    if (std::find(loops.begin(), loops.end(), loop_sref) == loops.end()) {
      throw MakeScheduleError<LoopPositionError>(self->mod, ffi::GetRef<For>(loop),
                                                 ffi::GetRef<SBlock>(block), "decompose_reduction");
    }
    // Cond 1. Check block is reduction
    CheckReductionBlock(self, block_sref, scope_root_sref);
    // Cond 2. Check 'loop' is higher than all the loops related to block var of type reduction
    LoopHeightError::CheckLoopHigherThanReduceLoops(self->mod, block, realize, loops, loop_sref);
  }
  // IR Manipulation
  ffi::Array<IterVar> init_iter_vars;
  ffi::Array<PrimExpr> init_iter_values;
  // Step 1. Create new block vars and their bindings
  // Maps an old block var to the new corresponding block var
  std::unordered_map<Var, Var> block_var_map;
  block_var_map.reserve(block->iter_vars.size());
  for (int i = 0, n = block->iter_vars.size(); i < n; ++i) {
    const IterVar& iter_var = block->iter_vars[i];
    const PrimExpr& binding = realize->iter_values[i];
    // Only process data parallel block vars
    if (iter_var->iter_type != IterVarType::kDataPar) {
      continue;
    }
    // Create a new block var
    IterVar new_iter_var(/*dom=*/iter_var->dom.value(),
                         /*var=*/iter_var->var.CopyWithSuffix(""),
                         /*iter_type=*/iter_var->iter_type,
                         /*thread_tag=*/iter_var->thread_tag);
    // Add a block var and its binding
    init_iter_vars.push_back(new_iter_var);
    init_iter_values.push_back(binding);
    // Add a mapping from old block vars to new block vars
    block_var_map.insert_or_assign(iter_var->var, new_iter_var->var);
  }
  // Step 2. After copying block vars, substitute them in init block
  auto map_block_var = [&block_var_map](
                           const Var& var,
                           TVMFFIDefRegionKind kind) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (kind != kTVMFFIDefRegionKindNone) return ffi::Unchanged();
    if (auto it = block_var_map.find(var); it != block_var_map.end()) {
      return ffi::Any(it->second);
    }
    return ffi::Unchanged();
  };
  auto init_block = ffi::make_object<SBlockObj>(
      ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(block->init.value(), map_block_var)
          .as_or_throw<Stmt>());
  init_block->name_hint = block->name_hint + "_init";
  init_block->annotations = block->annotations;
  init_block->iter_vars = std::move(init_iter_vars);
  for (const TensorRegion& write : block->writes) {
    ffi::Array<Range> mapped_region = write->region.Map([&map_block_var](const Range& range) {
      PrimExpr min = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(range->begin, map_block_var)
                         .as_or_throw<PrimExpr>();
      PrimExpr extent = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(range->extent, map_block_var)
                            .as_or_throw<PrimExpr>();
      return Range::FromMinExtent(min, extent);
    });
    init_block->writes.push_back(TensorRegion(write->source.as_or_throw<tvm::tirx::TensorVar>(),
                                              mapped_region, TensorRegionType()));
  }
  // Step 3. Scan loops not higher than the specified loop above the reduction block.
  //         If the loop is used in the init block binding, then it is chosen.
  //         Otherwise, it is discarded.
  std::unordered_set<const VarObj*> discarded_loops;
  std::vector<int> chosen_loops;
  for (int i = static_cast<int>(loops.size()) - 1; i >= 0; --i) {
    const VarObj* loop_var = loops[i]->StmtAs<ForObj>()->var.get();
    bool discarded = true;
    auto walkfn = [v = loop_var](const Var& var) -> ffi::Expected<ffi::WalkResult> {
      return var.get() == v ? ffi::WalkResult::Interrupt(ffi::VisitInterrupt(var))
                            : ffi::WalkResult::Advance();
    };
    for (const PrimExpr& expr : init_iter_values) {
      if (!ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(expr, walkfn).has_value()) {
        continue;
      }
      // The loop is related to init block bindings;
      chosen_loops.push_back(i);
      discarded = false;
      break;
    }
    if (discarded) {
      discarded_loops.insert(loop_var);
    }
    // Only scan loops not higher than the given loop
    if (loops[i].same_as(loop_sref)) {
      break;
    }
  }
  // Step 4. Derive the predicate for the init block realize.  Omit conjunction clauses that
  //         depend on discarded loops.
  auto init_realize = ffi::make_object<SBlockRealizeObj>(
      RewriteInitPredicate(realize->predicate, discarded_loops), SBlock(init_block));
  init_realize->iter_values = std::move(init_iter_values);
  // Step 5. Create new loops above init block
  std::unordered_map<Var, Var> loop_var_map;
  Stmt body = SBlockRealize(init_realize);
  for (int i : chosen_loops) {
    For old_loop = ffi::GetRef<For>(TVM_SREF_TO_FOR(loops[i]));
    // Create a new equivalent to the chosen loop
    Var old_loop_var = old_loop->var;
    PrimVar new_loop_var = old_loop->var.CopyWithSuffix("_init");
    loop_var_map.insert_or_assign(old_loop_var, new_loop_var);
    auto new_loop = old_loop.CopyOnWrite();
    new_loop->var = new_loop_var;
    new_loop->body = body;
    body = ffi::GetRef<For>(new_loop);
  }
  auto f_substitute = [&loop_var_map](
                          const Var& var,
                          TVMFFIDefRegionKind kind) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (kind != kTVMFFIDefRegionKindNone) return ffi::Unchanged();
    if (auto it = loop_var_map.find(var); it != loop_var_map.end()) {
      return ffi::Any(it->second);
    }
    return ffi::Unchanged();
  };
  body = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(body, f_substitute).as_or_throw<Stmt>();
  // Step 6. Mutate IR
  const SBlockObj* old_scope_root = TVM_SREF_TO_SBLOCK(scope_root_sref);
  auto [new_scope_root, new_reduction_block] = DecomposeReductionBlockReplacer::Replace(
      ffi::GetRef<SBlock>(old_scope_root), ffi::GetRef<For>(loop), body,
      ffi::GetRef<SBlock>(block));
  self->Replace(scope_root_sref, new_scope_root,
                {{ffi::GetRef<SBlock>(old_scope_root), new_scope_root},
                 {ffi::GetRef<SBlock>(block), new_reduction_block}});
  self->UpdateScopeSBlockInfo(new_scope_root);
  return self->stmt2ref.at(init_block.get());
}

/******** Commutative Reducer ********/

/*!
 * \brief A structure used for registering new commutative reducers, and store all the registered
 * reducers. The reducers are preserved in a list, in the form of "reducer-getter function". When
 * invoking a reducer-getter function with a specific datatype, the reducer-getter will return the
 * te::CommReducer of the corresponding reduction pattern and the specific datatype
 */
struct ReducerRegistry {
  ReducerRegistry()
      : reducer_getters{
            CreateReducerGetter(
                /*n_tensors=*/1,
                [](const ffi::Array<Var>& x, const ffi::Array<Var>& y) {
                  return ffi::Array<PrimExpr>{x[0].as_or_throw<PrimExpr>() +
                                              y[0].as_or_throw<PrimExpr>()};
                },
                [](const ffi::Array<PrimExpr>& values) {
                  return ffi::Array<PrimExpr>{prim::MakeConst(values[0].ty(), 0)};
                }),
            CreateReducerGetter(
                /*n_tensors=*/1,
                [](const ffi::Array<Var>& x, const ffi::Array<Var>& y) {
                  return ffi::Array<PrimExpr>{x[0].as_or_throw<PrimExpr>() *
                                              y[0].as_or_throw<PrimExpr>()};
                },
                [](const ffi::Array<PrimExpr>& values) {
                  return ffi::Array<PrimExpr>{prim::MakeConst(values[0].ty(), 1)};
                }),
            CreateReducerGetter(
                /*n_tensors=*/1,
                [](const ffi::Array<Var>& x, const ffi::Array<Var>& y) {
                  return ffi::Array<PrimExpr>{
                      min(x[0].as_or_throw<PrimExpr>(), y[0].as_or_throw<PrimExpr>())};
                },
                [](const ffi::Array<PrimExpr>& values) {
                  return ffi::Array<PrimExpr>{max_value(values[0].ty())};
                }),
            CreateReducerGetter(
                /*n_tensors=*/1,
                [](const ffi::Array<Var>& x, const ffi::Array<Var>& y) {
                  return ffi::Array<PrimExpr>{
                      max(x[0].as_or_throw<PrimExpr>(), y[0].as_or_throw<PrimExpr>())};
                },
                [](const ffi::Array<PrimExpr>& values) {
                  return ffi::Array<PrimExpr>{min_value(values[0].ty())};
                }),
            CreateReducerGetter(
                /*n_tensors=*/2,
                [](const ffi::Array<Var>& x, const ffi::Array<Var>& y) {
                  return ffi::Array<PrimExpr>{
                      x[0].as_or_throw<PrimExpr>() + y[0].as_or_throw<PrimExpr>(),
                      x[1].as_or_throw<PrimExpr>() + y[1].as_or_throw<PrimExpr>()};
                },
                [](const ffi::Array<PrimExpr>& values) {
                  return ffi::Array<PrimExpr>{prim::MakeConst(values[0].ty(), 0),
                                              prim::MakeConst(values[1].ty(), 0)};
                }),
            CreateReducerGetter(
                /*n_tensors=*/2,
                [](const ffi::Array<Var>& x, const ffi::Array<Var>& y) {
                  PrimExpr idx =
                      Select(x[1].as_or_throw<PrimExpr>() >= y[1].as_or_throw<PrimExpr>(),
                             x[0].as_or_throw<PrimExpr>(), y[0].as_or_throw<PrimExpr>());
                  PrimExpr val =
                      Select(x[1].as_or_throw<PrimExpr>() >= y[1].as_or_throw<PrimExpr>(),
                             x[1].as_or_throw<PrimExpr>(), y[1].as_or_throw<PrimExpr>());
                  return ffi::Array<PrimExpr>{idx, val};
                },
                [](const ffi::Array<PrimExpr>& values) {
                  return ffi::Array<PrimExpr>{prim::MakeConst(values[0].ty(), -1),
                                              min_value(values[1].ty())};
                }),
            CreateReducerGetter(
                /*n_tensors=*/2,
                [](const ffi::Array<Var>& x, const ffi::Array<Var>& y) {
                  PrimExpr idx = Select(
                      Or(greater(x[1].as_or_throw<PrimExpr>(), y[1].as_or_throw<PrimExpr>()),
                         And(equal(x[1].as_or_throw<PrimExpr>(), y[1].as_or_throw<PrimExpr>()),
                             less(x[0].as_or_throw<PrimExpr>(), y[0].as_or_throw<PrimExpr>()))),
                      x[0].as_or_throw<PrimExpr>(), y[0].as_or_throw<PrimExpr>());
                  PrimExpr val =
                      Select(greater(x[1].as_or_throw<PrimExpr>(), y[1].as_or_throw<PrimExpr>()),
                             x[1].as_or_throw<PrimExpr>(), y[1].as_or_throw<PrimExpr>());
                  return ffi::Array<PrimExpr>{idx, val};
                },
                [](const ffi::Array<PrimExpr>& values) {
                  return ffi::Array<PrimExpr>{prim::MakeConst(values[0].ty(), -1),
                                              min_value(values[1].ty())};
                }),
            CreateReducerGetter(
                /*n_tensors=*/2,
                [](const ffi::Array<Var>& x, const ffi::Array<Var>& y) {
                  PrimExpr idx =
                      Select(x[1].as_or_throw<PrimExpr>() <= y[1].as_or_throw<PrimExpr>(),
                             x[0].as_or_throw<PrimExpr>(), y[0].as_or_throw<PrimExpr>());
                  PrimExpr val =
                      Select(x[1].as_or_throw<PrimExpr>() <= y[1].as_or_throw<PrimExpr>(),
                             x[1].as_or_throw<PrimExpr>(), y[1].as_or_throw<PrimExpr>());
                  return ffi::Array<PrimExpr>{idx, val};
                },
                [](const ffi::Array<PrimExpr>& values) {
                  return ffi::Array<PrimExpr>{prim::MakeConst(values[0].ty(), -1),
                                              max_value(values[1].ty())};
                }),
            CreateReducerGetter(
                /*n_tensors=*/2,
                [](const ffi::Array<Var>& x, const ffi::Array<Var>& y) {
                  PrimExpr idx = Select(
                      Or(less(x[1].as_or_throw<PrimExpr>(), y[1].as_or_throw<PrimExpr>()),
                         And(equal(x[1].as_or_throw<PrimExpr>(), y[1].as_or_throw<PrimExpr>()),
                             less(x[0].as_or_throw<PrimExpr>(), y[0].as_or_throw<PrimExpr>()))),
                      x[0].as_or_throw<PrimExpr>(), y[0].as_or_throw<PrimExpr>());
                  PrimExpr val =
                      Select(less(x[1].as_or_throw<PrimExpr>(), y[1].as_or_throw<PrimExpr>()),
                             x[1].as_or_throw<PrimExpr>(), y[1].as_or_throw<PrimExpr>());
                  return ffi::Array<PrimExpr>{idx, val};
                },
                [](const ffi::Array<PrimExpr>& values) {
                  return ffi::Array<PrimExpr>{prim::MakeConst(values[0].ty(), -1),
                                              max_value(values[1].ty())};
                }),
            // argmax with `lhs_val > rhs_val` and tie-break `lhs_idx > rhs_idx`, which corresponds
            // to topi.argmax with `select_last_index=True` (preferring the last occurrence).
            CreateReducerGetter(
                /*n_tensors=*/2,
                [](const ffi::Array<Var>& x, const ffi::Array<Var>& y) {
                  PrimExpr idx = Select(
                      Or(greater(x[1].as_or_throw<PrimExpr>(), y[1].as_or_throw<PrimExpr>()),
                         And(equal(x[1].as_or_throw<PrimExpr>(), y[1].as_or_throw<PrimExpr>()),
                             greater(x[0].as_or_throw<PrimExpr>(), y[0].as_or_throw<PrimExpr>()))),
                      x[0].as_or_throw<PrimExpr>(), y[0].as_or_throw<PrimExpr>());
                  PrimExpr val =
                      Select(greater(x[1].as_or_throw<PrimExpr>(), y[1].as_or_throw<PrimExpr>()),
                             x[1].as_or_throw<PrimExpr>(), y[1].as_or_throw<PrimExpr>());
                  return ffi::Array<PrimExpr>{idx, val};
                },
                [](const ffi::Array<PrimExpr>& values) {
                  return ffi::Array<PrimExpr>{prim::MakeConst(values[0].ty(), -1),
                                              min_value(values[1].ty())};
                }),
            // argmin with `lhs_val < rhs_val` and tie-break `lhs_idx > rhs_idx`, which corresponds
            // to topi.argmin with `select_last_index=True` (preferring the last occurrence).
            CreateReducerGetter(
                /*n_tensors=*/2,
                [](const ffi::Array<Var>& x, const ffi::Array<Var>& y) {
                  PrimExpr idx = Select(
                      Or(less(x[1].as_or_throw<PrimExpr>(), y[1].as_or_throw<PrimExpr>()),
                         And(equal(x[1].as_or_throw<PrimExpr>(), y[1].as_or_throw<PrimExpr>()),
                             greater(x[0].as_or_throw<PrimExpr>(), y[0].as_or_throw<PrimExpr>()))),
                      x[0].as_or_throw<PrimExpr>(), y[0].as_or_throw<PrimExpr>());
                  PrimExpr val =
                      Select(less(x[1].as_or_throw<PrimExpr>(), y[1].as_or_throw<PrimExpr>()),
                             x[1].as_or_throw<PrimExpr>(), y[1].as_or_throw<PrimExpr>());
                  return ffi::Array<PrimExpr>{idx, val};
                },
                [](const ffi::Array<PrimExpr>& values) {
                  return ffi::Array<PrimExpr>{prim::MakeConst(values[0].ty(), -1),
                                              max_value(values[1].ty())};
                })} {}

  static void RegisterReducer(
      int n_tensors,
      ffi::TypedFunction<ffi::Array<PrimExpr>(ffi::Array<Var>, ffi::Array<Var>)> combiner_getter,
      ffi::TypedFunction<ffi::Array<PrimExpr>(ffi::Array<PrimExpr>)> identity_getter) {
    ReducerRegistry::Global()->reducer_getters.push_back(ReducerRegistry::CreateReducerGetter(
        n_tensors, std::move(combiner_getter), std::move(identity_getter)));
  }

  static ffi::TypedFunction<ffi::Optional<te::CommReducer>(ffi::Array<PrimExpr>)>
  CreateReducerGetter(
      int n_tensors,
      ffi::TypedFunction<ffi::Array<PrimExpr>(ffi::Array<Var>, ffi::Array<Var>)> combiner_getter,
      ffi::TypedFunction<ffi::Array<PrimExpr>(ffi::Array<PrimExpr>)> identity_getter) {
    return [n_tensors,                                     //
            combiner_getter = std::move(combiner_getter),  //
            identity_getter = std::move(identity_getter)   //
    ](ffi::Array<PrimExpr> values) -> ffi::Optional<te::CommReducer> {
      if (static_cast<int>(values.size()) != n_tensors) {
        return std::nullopt;
      }
      ffi::Array<PrimVar> lhs;
      ffi::Array<PrimVar> rhs;
      ffi::Array<Var> callback_lhs;
      ffi::Array<Var> callback_rhs;
      for (int i = 0; i < n_tensors; ++i) {
        PrimVar lhs_var("x" + std::to_string(i), values[i].ty());
        PrimVar rhs_var("y" + std::to_string(i), values[i].ty());
        lhs.push_back(lhs_var);
        rhs.push_back(rhs_var);
        callback_lhs.push_back(lhs_var);
        callback_rhs.push_back(rhs_var);
      }
      return te::CommReducer(lhs, rhs, combiner_getter(callback_lhs, callback_rhs),
                             identity_getter(values));
    };
  }

  static ReducerRegistry* Global() {
    static ReducerRegistry instance;
    return &instance;
  }

  std::vector<ffi::TypedFunction<ffi::Optional<te::CommReducer>(ffi::Array<PrimExpr>)>>
      reducer_getters;
};

std::vector<ffi::TypedFunction<ffi::Optional<te::CommReducer>(ffi::Array<PrimExpr>)>>
GetReducerGetters() {
  return ReducerRegistry::Global()->reducer_getters;
}

class NotSerialLoopKindError : public ScheduleErrorContextObj {
 public:
  explicit NotSerialLoopKindError(IRModule mod, For loop)
      : mod_(std::move(mod)), loop_(std::move(loop)) {}

  ffi::String FastErrorString() const final {
    return "ScheduleError: The input loop of rfactor is required to be `kSerial`";
  }

  ffi::String DetailRenderTemplate() const final {
    ffi::String str_kind = ForKind2String(loop_->kind);
    std::ostringstream os;
    os << "ScheduleError: The input loop {0} of rfactor is required to be `Serial`. However, the "
          "kind of {0} is `"
       << str_kind << "`";
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {loop_}; }

  IRModule mod_;
  For loop_;
};

class FactorAxisOutOfRangeError : public ScheduleErrorContextObj {
 public:
  explicit FactorAxisOutOfRangeError(IRModule mod, TensorVar tensor, int factor_axis)
      : mod_(std::move(mod)), tensor_(std::move(tensor)), factor_axis_(factor_axis) {}

  ffi::String FastErrorString() const final {
    return "ScheduleError: The input `factor_axis` is out of range. It is required to be in range "
           "[-(ndim + 1), ndim] where `ndim` is the number of dimensions of the write tensor";
  }

  ffi::String DetailRenderTemplate() const final {
    std::ostringstream os;
    int ndim = static_cast<int>(tensor_.shape().size());
    os << "The write tensor " << tensor_->name << " has " << ndim
       << " dimension(s), so `factor_axis` is required to be in [" << -(ndim + 1) << ", " << ndim
       << "] for rfactor. However, the input `factor_axis` is " << factor_axis_
       << ", which is out of the expected range";
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {}; }

  static int CheckAndUpdate(const IRModule& mod, const TensorVar& tensor, int factor_axis) {
    int ndim = static_cast<int>(tensor.shape().size());
    if (factor_axis < -(ndim + 1) || factor_axis > ndim) {
      throw MakeScheduleError<FactorAxisOutOfRangeError>(mod, tensor, factor_axis);
    }
    // If factor_axis is negative, convert it to a non-negative one.
    if (factor_axis < 0) {
      factor_axis += ndim + 1;
    }
    return factor_axis;
  }

  IRModule mod_;
  TensorVar tensor_;
  int factor_axis_;
};

class LoopPropertyError : public ScheduleErrorContextObj {
 public:
  enum ErrorType {
    kDataParIterTouchRFactorLoop = 0,
    kLoopTouchedByBothKindsOfBlockIters = 1,
    kNotFirstChildBlockOfOutermostLoop = 2,
    kUnboundLoopUnderReductionLoop = 3
  };

  explicit LoopPropertyError(IRModule mod, For loop, ErrorType error_type)
      : mod_(std::move(mod)), loop_(std::move(loop)), error_type_(error_type) {}

  ffi::String FastErrorString() const final {
    switch (error_type_) {
      case kDataParIterTouchRFactorLoop:
        return "ScheduleError: The loop to be applied rfactor is required not to be touched by any "
               "data parallel block iter of the block";
      case kLoopTouchedByBothKindsOfBlockIters:
        return "ScheduleError: The loops outside of the reduction block are required not to be "
               "touched by both data parallel block iters and reduction block iters";
      case kNotFirstChildBlockOfOutermostLoop:
        return "ScheduleError: The reduction block should be the first child block of the "
               "outermost loop outside of it";
      case kUnboundLoopUnderReductionLoop:
        return "ScheduleError: A loop who has extent greater than one and is not bound to any "
               "block iter should not appear under a reduction loop";
    }
    TVM_FFI_ICHECK(false) << "Unreachable";
    throw;
  }

  ffi::String DetailRenderTemplate() const final {
    switch (error_type_) {
      case kDataParIterTouchRFactorLoop:
        return "The loop to be applied rfactor is {0}, which is required not to be touched by any "
               "data parallel block iter of the block below. However, some of the block's data "
               "parallel block iters touch this loop";
      case kLoopTouchedByBothKindsOfBlockIters:
        return "It is not allowed that the loop {0} is touched by both some data parallel block "
               "iters and some reduction block iters";
      case kNotFirstChildBlockOfOutermostLoop:
        return "The first child block of the outermost loop {0} is not the reduction block.";
      case kUnboundLoopUnderReductionLoop:
        return "The loop {0} has extent greater than one, and is not bound to any block iter. "
               "Therefore it shouldn't appear under a reduction loop";
    }
    TVM_FFI_ICHECK(false) << "Unreachable";
    throw;
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {loop_}; }

  static void CheckLoopProperty(const ScheduleState& self, const ffi::Array<For>& loops,
                                const ForObj* rf_loop, const SBlock& block,
                                const std::unordered_set<const VarObj*>& data_par_loop_vars,
                                const std::unordered_set<const VarObj*>& reduce_loop_vars) {
    ffi::Array<SBlockRealize> children_of_outermost_loop =
        GetChildBlockRealizeOnSRefTree(self->stmt2ref.at(loops[0].get()));
    if (!children_of_outermost_loop[0]->block.same_as(block)) {
      throw MakeScheduleError<LoopPropertyError>(self->mod, loops[0],
                                                 kNotFirstChildBlockOfOutermostLoop);
    }

    bool meet_reduction_loop = false;
    for (const For& loop : loops) {
      bool data_par_touched = data_par_loop_vars.count(loop->var.get());
      bool reduction_touched = reduce_loop_vars.count(loop->var.get());

      if (data_par_touched && reduction_touched) {
        throw MakeScheduleError<LoopPropertyError>(self->mod, loop,
                                                   kLoopTouchedByBothKindsOfBlockIters);
      } else if (data_par_touched) {
        if (loop.get() == rf_loop) {
          throw MakeScheduleError<LoopPropertyError>(self->mod, loop, kDataParIterTouchRFactorLoop);
        }
        continue;
      } else if (reduction_touched) {
        if (!meet_reduction_loop) {
          CheckGetSingleChildBlockRealizeOnSRefTree(self, self->stmt2ref.at(loop.get()));
          meet_reduction_loop = true;
        }
        continue;
      } else if (meet_reduction_loop &&
                 !IsOne(Range::GetExtentFromBeginEnd(loop->begin, loop->end))) {
        throw MakeScheduleError<LoopPropertyError>(self->mod, loop, kUnboundLoopUnderReductionLoop);
      }
    }
  }

  IRModule mod_;
  For loop_;
  ErrorType error_type_;
};

/*!
 * \brief For each loop in the given array of loop, associate its loop var with the loop itself
 * using a mapping
 * \param loops The loops to be analyzed
 * \return A mapping from loops to their corresponding loop vars
 */
std::unordered_map<const VarObj*, For> GetLoopVar2LoopMap(const ffi::Array<For>& loops) {
  std::unordered_map<const VarObj*, For> loop_vars2loop;
  loop_vars2loop.reserve(loops.size());
  for (const For& loop : loops) {
    loop_vars2loop.insert_or_assign(loop->var.get(), loop);
  }
  return loop_vars2loop;
}

/*!
 * \brief Create the intermediate rfactor tensors, which the rfactor block writes to and the
 * write-back block reads from
 * \param tensor_stores The TensorStores of the original block, where the rfactor tensors will be
 * created from
 * \param factor_axis The `factor_axis` parameter of rfactor
 * \param rf_loop The rfactor loop
 * \return The new created intermediate rfactor tensor
 */
ffi::Array<TensorVar> CreateRFactorTensors(const ffi::Array<TensorStore>& tensor_stores,
                                           int factor_axis, const ForObj* rf_loop) {
  ffi::Array<TensorVar> rf_tensors;
  rf_tensors.reserve(tensor_stores.size());
  for (const TensorStore& tensor_store : tensor_stores) {
    TensorVar tensor = tensor_store->dest.as_or_throw<TensorVar>();
    ffi::Array<PrimExpr> rf_shape = tensor.shape();
    rf_shape.insert(rf_shape.begin() + factor_axis,
                    Range::GetExtentFromBeginEnd(rf_loop->begin, rf_loop->end));

    ffi::ObjectPtr<TensorTypeObj> n = ffi::make_object<TensorTypeObj>(*tensor.ty().get());
    n->shape = rf_shape;
    rf_tensors.push_back(TensorVar(tensor->name + ".rf", TensorType(std::move(n)), tensor->loc));
  }
  return rf_tensors;
}

/*!
 * \brief The base class of the rfactor/write-back block creator, which creates the blocks in four
 * steps:
 * 1) Create the new block iters and the their iter bindings
 * 2) Create the body and init of the new block
 * 3) Create the read/write regions of the new block
 * 4) Create the new block and the new block-realize
 */
class BaseBlockCreator {
 public:
  explicit BaseBlockCreator(SBlockRealize old_block_realize, For rf_loop,
                            ffi::Array<TensorStore> old_reduction_updates, te::CommReducer reducer,
                            ffi::Array<TensorVar> rf_tensors, bool is_rf_block)
      : old_block_realize_(std::move(old_block_realize)),
        rf_loop_(std::move(rf_loop)),
        old_reduction_updates_(std::move(old_reduction_updates)),
        reducer_(std::move(reducer)),
        rf_tensors_(std::move(rf_tensors)),
        n_tensors_(static_cast<int>(rf_tensors_.size())),
        is_rf_block_(is_rf_block) {
    n_block_iters_ = static_cast<int>(old_block_realize_->iter_values.size());
    update_tensors_.reserve(n_tensors_);
    update_indices_.reserve(n_tensors_);
    update_lhs_.reserve(n_tensors_);
    update_rhs_.reserve(n_tensors_);
  }

  void CreateBlock() {
    CreateAdditionalIter();
    for (int i = 0; i < n_block_iters_; ++i) {
      CreateNormalIters(i);
    }
    bool has_reduce_iter = false;
    for (const IterVar& iter_var : iter_vars_) {
      if (iter_var->iter_type == IterVarType::kCommReduce) {
        has_reduce_iter = true;
        break;
      }
    }

    // The pre-processing finds out the tensors written in the block, the indices of the tensor
    // accesses, and the reduction LHS and RHS of the stored values.
    PreProcess();
    auto map_block_var = [this](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
      if (auto repl = var_map_.Get(var)) return ffi::Any(*std::move(repl));
      return ffi::Unchanged();
    };
    Stmt block_body = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(
                          CreateBlockBody(has_reduce_iter), map_block_var)
                          .as_or_throw<Stmt>();
    ffi::Optional<SeqStmt> block_init = CreateBlockInit(has_reduce_iter);
    if (block_init.has_value()) {
      block_init =
          SeqStmt(ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(block_init.value(), map_block_var)
                      .as_or_throw<Stmt>());
    }
    CreateReadWriteRegions();

    ffi::String new_block_name = old_block_realize_->block->name_hint;
    PrimExpr predicate = IntImm::Bool(true);
    if (is_rf_block_) {
      new_block_name = new_block_name + "_rf";
      predicate = old_block_realize_->predicate;
    }
    new_block_ = SBlock(
        /*iter_vars=*/iter_vars_,
        /*reads=*/read_regions_,
        /*writes=*/write_regions_,
        /*name_hint=*/new_block_name,
        /*body=*/std::move(block_body),
        /*init=*/std::move(block_init),
        /*alloc_tensors=*/{},
        /*match_tensors=*/{},
        /*annotations=*/old_block_realize_->block->annotations);
    new_block_realize_ = SBlockRealize(iter_values_, predicate, new_block_);
  }

 private:
  virtual void CreateAdditionalIter() = 0;
  virtual void CreateNormalIters(int idx) = 0;
  virtual void PreProcess() = 0;
  virtual void CreateReadWriteRegions() = 0;

  SeqStmt CreateBlockBody(bool has_reduce_iter) {
    ffi::Array<Stmt> tensor_stores;
    tensor_stores.reserve(n_tensors_);

    // Case 1. If the block has no reduction iterator, we just store the RHS values into the
    // tensors.
    if (!has_reduce_iter) {
      for (int i = 0; i < n_tensors_; ++i) {
        tensor_stores.push_back(
            TensorStore(update_tensors_[i], update_indices_[i], update_rhs_[i]));
      }
      return SeqStmt(tensor_stores);
    }

    // Case 2. If the reduction is for single tensor, the block body is a single TensorStore.
    ffi::Array<PrimExpr> stored_values = (*reducer_.get())(update_lhs_, update_rhs_);
    if (n_tensors_ == 1) {
      return TensorStore(update_tensors_[0], update_indices_[0], stored_values[0]);
    }

    // Case 3. In case the reduction is for multiple tensors, we should create the reduction with
    // Bind nodes so that the reduction execution generates correct results.
    ffi::Array<Var> let_vars;
    let_vars.reserve(n_tensors_);
    for (int i = 0; i < n_tensors_; ++i) {
      Var var("v_" + update_tensors_[i]->name, stored_values[i].ty());
      let_vars.push_back(var);
      tensor_stores.push_back(
          TensorStore(update_tensors_[i], update_indices_[i], var.as_or_throw<PrimExpr>()));
    }
    ffi::Array<Stmt> stmts;
    for (int i = 0; i < n_tensors_; ++i) {
      stmts.push_back(tvm::Bind(let_vars[i], stored_values[i]));
    }
    for (const auto& store : tensor_stores) {
      stmts.push_back(store);
    }
    return SeqStmt(stmts);
  }

  ffi::Optional<SeqStmt> CreateBlockInit(bool has_reduce_iter) {
    if (!has_reduce_iter) {
      return std::nullopt;
    }

    ffi::Array<Stmt> inits;
    inits.reserve(n_tensors_);
    for (int i = 0; i < n_tensors_; ++i) {
      inits.push_back(
          TensorStore(update_tensors_[i], update_indices_[i], reducer_->identity_element[i]));
    }
    return SeqStmt(inits);
  }

 public:
  /*! \brief The new created block */
  SBlock new_block_{ffi::UnsafeInit{}};
  /*! \brief The new created block-realize */
  SBlockRealize new_block_realize_{ffi::UnsafeInit{}};
  /*! \brief The indices used to access the intermediate rfactor tensor */
  ffi::Array<PrimExpr> rf_tensor_access_indices_;

 protected:
  /*! \brief The old block-realize */
  SBlockRealize old_block_realize_;
  /*! \brief The number of block iters in the old block */
  int n_block_iters_;
  /*! \brief The rfactor loop */
  For rf_loop_;
  /*! \brief The update TensorStores of the old block */
  ffi::Array<TensorStore> old_reduction_updates_;
  /*! \brief The matched commutative reducer */
  te::CommReducer reducer_;
  /*! \brief The intermediate rfactor tensors */
  ffi::Array<TensorVar> rf_tensors_;
  /*! \brief The number of rfactor tensors. */
  const int n_tensors_;
  /*!
   * \brief A mapping which maps old block iters to new expressions. The old iters will be replaced
   * by the expressions in future substitution for the two blocks
   */
  ffi::Map<Var, PrimExpr> var_map_;

  /*! \brief Whether we are creating the rfactor block or the write-back block */
  bool is_rf_block_;
  /*! \brief The new block iters of the new created block */
  std::vector<IterVar> iter_vars_;
  /*! \brief The new block iter bindings of the new created block-realize */
  std::vector<PrimExpr> iter_values_;
  /*! \brief The tensors updated in this block */
  ffi::Array<TensorVar> update_tensors_;
  /*! \brief The indices of the tensors updated in this block, respectively */
  ffi::Array<ffi::Array<PrimExpr>> update_indices_;
  /*! \brief The LHS values of the reduction in this block */
  ffi::Array<PrimExpr> update_lhs_;
  /*! \brief THe RHS values of the reduction in this block */
  ffi::Array<PrimExpr> update_rhs_;
  /*! \brief The read regions of the new created block */
  ffi::Array<TensorRegion> read_regions_;
  /*! \brief The write regions of the new created block */
  ffi::Array<TensorRegion> write_regions_;
};

/*!
 * \brief The derived class of the rfactor block creator, which implements all virtual methods in
 * the base creator
 * \details Start constructing the rfactor block. The main difficulty to construct the rfactor block
 * is to create its block iters. So here we introduce the algorithm to create the block iters.
 *  1. Create a block iter for the rfactor loop. The block binding of this iter is the loop var, and
 *     the block iter is data parallel.
 *  2. For all the old block's block iters, there are two cases:
 *    (a) If it is data parallel block iter, or a reduction block iter which doesn't touch the
 *        rfactor loop, we keep it and its block binding in the rfactor block.
 *    (b) Otherwise it is a reduction block iter which touches the rfactor loop. In this case, we
 *        "split" the block iter into one or more new block iters and do not keep the old block
 *        var. More specifically, we create a new reduction block iter for each loop var that
 *        appears in the reduction block iter's binding (except for the rfactor loop), and the
 *        binding of the new block iter is exactly the loop var. (Note that for each loop var, we
 *        create at most one block iter, even if there are multiple old block iters which touch
 *        both this loop and the rfactor loop).
 *        Then we substitute the appearances of the old block iter with the new created block
 *        iters by recording two mappings: one maps loops vars to new created block iters which
 *        is used for binding substitution, and another maps old block iters to new expressions
 *        which is used for substitutions of the old block iters.
 */
class RFactorBlockCreator : public BaseBlockCreator {
 public:
  explicit RFactorBlockCreator(SBlockRealize old_block_realize, For rf_loop,
                               ffi::Array<TensorStore> old_reduction_updates,
                               te::CommReducer reducer, ffi::Array<TensorVar> rf_tensors,
                               std::unordered_map<const VarObj*, For> loop_vars2loop,
                               int factor_axis, ffi::Array<PrimExpr> combiner_rhs)
      : BaseBlockCreator(std::move(old_block_realize), std::move(rf_loop),
                         std::move(old_reduction_updates), std::move(reducer),
                         std::move(rf_tensors), true),
        loop_vars2loop_(std::move(loop_vars2loop)),
        factor_axis_(factor_axis),
        combiner_rhs_(std::move(combiner_rhs)) {}

 private:
  void CreateAdditionalIter() final {
    // Create a new data parallel block iter for the rfactor loop.
    additional_iter_ = IterVarFromLoop(rf_loop_, "v" + rf_loop_->var->name, IterVarType::kDataPar);
    loop_var2block_binding_.insert_or_assign(rf_loop_->var.get(), additional_iter_->var);
    iter_vars_.push_back(additional_iter_);
    iter_values_.push_back(rf_loop_->var);
  }

  void CreateNormalIters(int idx) final {
    IterVar old_iter = old_block_realize_->block->iter_vars[idx];
    PrimExpr old_binding = old_block_realize_->iter_values[idx];
    auto walkfn = [v = rf_loop_->var.get()](const Var& var) -> ffi::Expected<ffi::WalkResult> {
      return var.get() == v ? ffi::WalkResult::Interrupt(ffi::VisitInterrupt(var))
                            : ffi::WalkResult::Advance();
    };
    if (old_iter->iter_type == IterVarType::kDataPar ||
        !ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(old_binding, walkfn).has_value()) {
      // The old block iter is either a data parallel block iter, or a reduction block iter that
      // doesn't touch the rfactor loop. In this case reuse the old reduction block iter and its
      // corresponding binding.
      iter_vars_.push_back(old_iter);
      iter_values_.push_back(old_binding);
      return;
    }
    TVM_FFI_ICHECK(old_iter->iter_type == kCommReduce);
    // This block iter is a reduction block iter that touches the rfactor loop. So next we try to
    // create a new block iter for all loop vars that appear in the old binding.
    ffi::Array<Var> vars_in_old_binding = UndefinedVars(old_binding);
    for (const Var& var : vars_in_old_binding) {
      auto it = loop_vars2loop_.find(var.get());
      if (it == loop_vars2loop_.end()) {
        // `var` is not a loop var. So skip.
        continue;
      }
      const For& loop = it->second;
      if (loop_var2block_binding_.find(var.get()) == loop_var2block_binding_.end()) {
        // We haven't created the new block iter for `var`. So here we create it, append it
        // and its binding to `rf_block_iter_vars` and `rf_block_iter_values` respectively.
        IterVar new_iter_var =
            IterVarFromLoop(loop, "v" + loop->var->name, IterVarType::kCommReduce);
        loop_var2block_binding_.insert_or_assign(var.get(), new_iter_var->var);
        iter_vars_.push_back(new_iter_var);
        iter_values_.push_back(var.as_or_throw<PrimExpr>());
      }
    }
    // Substitute the original binding with new block iters. Store the result expression
    // in `rf_var_map` for future substitution.
    auto f_substitute = [this](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
      if (auto it = loop_var2block_binding_.find(var.get()); it != loop_var2block_binding_.end()) {
        return ffi::Any(it->second);
      }
      return ffi::Unchanged();
    };
    var_map_.Set(old_iter->var,
                 ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(old_binding, f_substitute)
                     .as_or_throw<PrimExpr>());
  }

  void PreProcess() final {
    // The accessed indices for all reduction tensors are the same.
    rf_tensor_access_indices_ = old_reduction_updates_[0]->indices;
    rf_tensor_access_indices_.insert(rf_tensor_access_indices_.begin() + factor_axis_,
                                     additional_iter_->var);
    for (int i = 0; i < n_tensors_; ++i) {
      update_tensors_.push_back(rf_tensors_[i]);
      update_indices_.push_back(rf_tensor_access_indices_);
      update_lhs_.push_back(TensorLoad(update_tensors_[i], rf_tensor_access_indices_));
      update_rhs_.push_back(combiner_rhs_[i]);
    }
  }

  void CreateReadWriteRegions() final {
    ffi::Map<TensorVar, TensorVar> tensor_map;
    for (int i = 0; i < n_tensors_; ++i) {
      tensor_map.Set(old_reduction_updates_[i]->dest.as_or_throw<TensorVar>(), rf_tensors_[i]);
    }
    const SBlock& old_block = old_block_realize_->block;
    auto map_block_var = [this](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
      if (auto repl = var_map_.Get(var)) return ffi::Any(*std::move(repl));
      return ffi::Unchanged();
    };
    read_regions_.reserve(old_block->reads.size());
    for (const TensorRegion& read_region : old_block->reads) {
      ffi::Array<Range> region = read_region->region.Map([&map_block_var](const Range& range) {
        PrimExpr min = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(range->begin, map_block_var)
                           .as_or_throw<PrimExpr>();
        PrimExpr extent =
            ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(range->extent, map_block_var)
                .as_or_throw<PrimExpr>();
        return Range::FromMinExtent(min, extent);
      });
      read_regions_.push_back(TensorRegion(read_region->source.as_or_throw<tvm::tirx::TensorVar>(),
                                           region, TensorRegionType()));
    }
    write_regions_.reserve(old_block->writes.size());
    for (const TensorRegion& write_region : old_block->writes) {
      ffi::Array<Range> region = write_region->region;
      region.insert(
          region.begin() + factor_axis_,
          Range::FromMinExtent(additional_iter_->var, IntImm(additional_iter_->var.ty(), 1)));
      ffi::Optional<TensorVar> rf_tensor =
          tensor_map.Get(write_region->source.as_or_throw<tvm::tirx::TensorVar>());
      TVM_FFI_ICHECK(rf_tensor.has_value());
      region.MutateByApply([&map_block_var](const Range& range) {
        PrimExpr min = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(range->begin, map_block_var)
                           .as_or_throw<PrimExpr>();
        PrimExpr extent =
            ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(range->extent, map_block_var)
                .as_or_throw<PrimExpr>();
        return Range::FromMinExtent(min, extent);
      });
      write_regions_.push_back(TensorRegion(rf_tensor.value(), region, TensorRegionType()));
    }
  }

 public:
  /*! \brief The generated additional block iter in rfactor block for the rfactor loop */
  IterVar additional_iter_{ffi::UnsafeInit{}};

 private:
  /*!
   * \brief A mapping which maps a loop var to its corresponding For loop for all the reduction
   * block's outer loops
   */
  std::unordered_map<const VarObj*, For> loop_vars2loop_;
  /*! \brief The factor_axis specified for rfactor */
  int factor_axis_;
  /*! \brief The RHS values of the reduction in the old block */
  ffi::Array<PrimExpr> combiner_rhs_;
  /*!
   * \brief A mapping which maps loop vars to new created block iters. This map is used to
   * substitute the loop vars which appear in the bindings of some old block iters with the new
   * created block iters
   */
  std::unordered_map<const VarObj*, Var> loop_var2block_binding_;
};

/*!
 * \brief The derived class of the write-back block creator, which implements all virtual methods in
 * the base creator
 */
class WriteBackBlockCreator : public BaseBlockCreator {
 public:
  explicit WriteBackBlockCreator(SBlockRealize old_block_realize, For rf_loop,
                                 ffi::Array<TensorStore> old_reduction_updates,
                                 te::CommReducer reducer, ffi::Array<TensorVar> rf_tensors,
                                 IterVar rf_additional_iter, ffi::Array<PrimExpr> combiner_lhs,
                                 ffi::Array<PrimExpr> rf_tensor_access_indices)
      : BaseBlockCreator(std::move(old_block_realize), std::move(rf_loop),
                         std::move(old_reduction_updates), std::move(reducer),
                         std::move(rf_tensors), false),
        rf_additional_iter_(std::move(rf_additional_iter)),
        combiner_lhs_(std::move(combiner_lhs)) {
    iter_vars_.reserve(n_block_iters_);
    iter_values_.reserve(n_block_iters_);
    rf_tensor_access_indices_ = std::move(rf_tensor_access_indices);
  }

 private:
  void CreateAdditionalIter() final {
    // Create a new reduction block iter for the rfactor loop.
    IterVar wb_new_block_iter = IterVarFromLoop(rf_loop_, "v" + rf_loop_->var->name, kCommReduce);
    iter_vars_.push_back(wb_new_block_iter);
    iter_values_.push_back(rf_loop_->var);
    var_map_.Set(rf_additional_iter_->var, wb_new_block_iter->var);
  }

  void CreateNormalIters(int idx) final {
    IterVar old_block_iter = old_block_realize_->block->iter_vars[idx];
    if (old_block_iter->iter_type == IterVarType::kDataPar) {
      iter_vars_.emplace_back(old_block_iter->dom.value(), old_block_iter->var.CopyWithSuffix(""),
                              kDataPar);
      iter_values_.push_back(old_block_realize_->iter_values[idx]);
      var_map_.Set(old_block_iter->var, iter_vars_.back()->var);
    }
  }

  void PreProcess() final {
    auto map_block_var = [this](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
      if (auto repl = var_map_.Get(var)) return ffi::Any(*std::move(repl));
      return ffi::Unchanged();
    };
    for (int i = 0; i < n_tensors_; ++i) {
      PrimExpr rhs = TensorLoad(rf_tensors_[i], rf_tensor_access_indices_);
      update_tensors_.push_back(old_reduction_updates_[i]->dest.as_or_throw<TensorVar>());
      update_indices_.push_back(old_reduction_updates_[i]->indices);
      update_lhs_.push_back(
          ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(combiner_lhs_[i], map_block_var)
              .as_or_throw<PrimExpr>());
      update_rhs_.push_back(
          ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(std::move(rhs), map_block_var)
              .as_or_throw<PrimExpr>());
    }
  }

  void CreateReadWriteRegions() final {
    CreateRegion(update_rhs_, true);
    CreateRegion(update_lhs_, false);
  }

  void CreateRegion(const ffi::Array<PrimExpr>& tensor_loads, bool is_read) {
    ffi::Array<TensorRegion>& tensor_regions = is_read ? read_regions_ : write_regions_;
    for (const PrimExpr& expr : tensor_loads) {
      const auto* tensor_load = expr.as<TensorLoadObj>();
      TVM_FFI_ICHECK(tensor_load != nullptr);
      ffi::Array<Range> region;
      region.reserve(tensor_load->indices.size());
      for (const PrimExpr& index : tensor_load->indices) {
        region.push_back(Range::FromMinExtent(index, prim::MakeConst(index.ty(), 1)));
      }
      tensor_regions.push_back(TensorRegion(tensor_load->source.as_or_throw<tvm::tirx::TensorVar>(),
                                            std::move(region), TensorRegionType()));
    }
  }

 private:
  /*! \brief The new created additional block iter of the rfactor block */
  IterVar rf_additional_iter_;
  /*! \brief The LHS values of the reduction in the old block */
  ffi::Array<PrimExpr> combiner_lhs_;
};

/*!
 * \brief Create new outer loops for the rfactor block, meanwhile update the rfactor block's iter
 * bindings to use the new created loop vars
 * \param rf_block_realize The BlockRealize of the rfactor block
 * \param loops The loops to be wrapped over the rfactor block
 * \return A Stmt which is the wrapping result
 */
Stmt CreateLoopOutsideRfactorBlock(SBlockRealize rf_block_realize, const ffi::Array<For>& loops) {
  int n_loops = static_cast<int>(loops.size());

  // Step 1. Create new loop vars.
  ffi::Array<For> new_loops;
  std::unordered_map<const VarObj*, Var> new_loop_var_map;
  new_loops.reserve(n_loops);
  new_loop_var_map.reserve(n_loops);
  for (const For& old_loop : loops) {
    Var new_loop_var = old_loop->var.CopyWithSuffix("");
    new_loop_var_map.insert_or_assign(old_loop->var.get(), new_loop_var);
  }

  // Step 2. Update the iter bindings and predicate of the rfactor block.
  auto map_loop_var =
      [&new_loop_var_map](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (auto it = new_loop_var_map.find(var.get()); it != new_loop_var_map.end()) {
      return ffi::Any(it->second);
    }
    return ffi::Unchanged();
  };
  ffi::Array<PrimExpr> new_bindings;
  new_bindings.reserve(rf_block_realize->iter_values.size());
  for (const PrimExpr& old_binding : rf_block_realize->iter_values) {
    new_bindings.push_back(ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(old_binding, map_loop_var)
                               .as_or_throw<PrimExpr>());
  }
  {
    SBlockRealizeObj* p_rf_block_realize = rf_block_realize.CopyOnWrite();
    p_rf_block_realize->iter_values = new_bindings;
    p_rf_block_realize->predicate =
        ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(rf_block_realize->predicate, map_loop_var)
            .as_or_throw<PrimExpr>();
  }

  // Step 3. Wrap `rf_block_realize` with outer loops.
  Stmt rf_body = rf_block_realize;
  for (int i = n_loops - 1; i >= 0; --i) {
    ffi::ObjectPtr<ForObj> p_loop = ffi::make_object<ForObj>(*loops[i].get());
    p_loop->var = new_loop_var_map.at(loops[i]->var.get()).as_or_throw<PrimVar>();
    p_loop->body = rf_body;
    rf_body = For(std::move(p_loop));
  }

  return rf_body;
}

class BlockReplacer : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) override {
    if (value.as<ExprObj>()) return ffi::Unchanged();
    return StmtExprMutator::Mutate(value, inplace_mode);
  }

  /*!
   * \brief The replace takes the old scope root block as input, and does four things:
   *  1) replace the reduction block with the write-back block,
   *  2) remove loops outside the write-back block that are touched by reduction block iters, except
   *  for the rfactor loop
   *  3) combine the rfactor block (wrapped with outer loops) and the transformed outermost loop
   *  into a SeqStmt, and
   *  4) insert the rfactor tensor into the scope root block's `alloc_tensors`
   * After transformation, the function returns the new scope root block
   * \param scope_root_block The old scope root block
   * \param rf_body The rfactor block, which is already wrapped with outer loops
   * \param outermost_loop The loop that is outermost among all loops outside the reduction block
   * \param wb_block_realize The new created BlockRealize of the write-back block
   * \param old_block_realize The BlockRealize of the reduction block
   * \param rf_loop The rfactor loop, which should be kept outside the write-back block
   * \param reduce_loop_vars The loops that are touched by reduction block iters, used to remove
   * loops outside the write-back block
   * \param loop_vars2loop The mapping from loop vars to loops that are outside the reduction block,
   * which is used to reduce redundant recursive visits
   * \param rf_tensor The rfactor tensor to be added into the scope root's `alloc_tensors`
   * \return The transformed new scope root block
   */
  static SBlock Replace(SBlock scope_root_block, Stmt rf_body, For outermost_loop,
                        SBlockRealize wb_block_realize, SBlockRealize old_block_realize,
                        For rf_loop, std::unordered_set<const VarObj*> reduce_loop_vars,
                        std::unordered_map<const VarObj*, For> loop_vars2loop,
                        const ffi::Array<TensorVar>& rf_tensors) {
    auto replacer = ffi::make_object<BlockReplacer>(
        std::move(rf_body), std::move(outermost_loop), std::move(wb_block_realize),
        std::move(old_block_realize), std::move(rf_loop), std::move(reduce_loop_vars),
        std::move(loop_vars2loop));
    SBlock new_scope_root = replacer->Mutate(scope_root_block, InplaceMode::kAllow)
                                .ValueOrUnchanged(std::move(scope_root_block))
                                .as_or_throw<SBlock>();
    SBlockObj* p = new_scope_root.CopyOnWrite();
    for (const TensorVar& rf_tensor : rf_tensors) {
      p->alloc_tensors.push_back(rf_tensor);
    }
    return new_scope_root;
  }

  explicit BlockReplacer(Stmt rf_body, For outermost_loop, SBlockRealize wb_block_realize,
                         SBlockRealize old_block_realize, For rf_loop,
                         std::unordered_set<const VarObj*> reduce_loop_vars,
                         std::unordered_map<const VarObj*, For> loop_vars2loop)
      : rf_body_(std::move(rf_body)),
        outermost_loop_(std::move(outermost_loop)),
        wb_block_realize_(std::move(wb_block_realize)),
        old_block_realize_(std::move(old_block_realize)),
        rf_loop_(std::move(rf_loop)),
        reduce_loop_vars_(std::move(reduce_loop_vars)),
        loop_vars2loop_(std::move(loop_vars2loop)) {}

 private:
  UnchangedOr<Stmt> Mutate_(const ForObj* loop, InplaceMode inplace_mode) final {
    // Step 1. Check whether this loop is outside the reduction block. Given that we've made sure
    // that the scope root block has stage-pipeline property, if this loop is not outside the
    // reduction block, there's no need to recursively mutate.
    if (!loop_vars2loop_.count(loop->var.get())) {
      return ffi::Unchanged();
    }

    // Step 2. Recursively mutate.
    Stmt body = StmtExprMutator::Mutate(ffi::AnyView(loop->body), inplace_mode)
                    .ValueOrUnchanged(loop->body)
                    .as_or_throw<Stmt>();

    // Step 3. If this loop is the rfactor loop and isn't touched by any reduction block iter, it
    // should be kept outside the write-back block. Otherwise it shouldn't.
    if (loop == rf_loop_.get() || !reduce_loop_vars_.count(loop->var.get())) {
      if (inplace_mode == InplaceMode::kAllow) {
        const_cast<ForObj*>(loop)->body = std::move(body);
        body = ffi::GetRef<For>(loop);
      } else {
        auto copy = ffi::make_object<ForObj>(*loop);
        copy->body = std::move(body);
        body = For(std::move(copy));
      }
    }

    // Step 4. If this loop is the outermost loop of the reduction block, return the combination of
    // `rf_body_` and the mutation result `body`. Otherwise return the mutation result.
    return loop == outermost_loop_.get() ? SeqStmt({rf_body_, body}) : body;
  }

  UnchangedOr<Stmt> Mutate_(const SBlockRealizeObj* block_realize, InplaceMode inplace_mode) final {
    // Due to the visitor's behavior on ForObj, this block-realize must be the reduction block's
    // block-realize. And we directly return the new `wb_block_realize`.
    TVM_FFI_ICHECK_EQ(block_realize, old_block_realize_.get());
    return wb_block_realize_;
  }

  Stmt rf_body_;
  For outermost_loop_;
  SBlockRealize wb_block_realize_;
  SBlockRealize old_block_realize_;
  For rf_loop_;
  std::unordered_set<const VarObj*> reduce_loop_vars_;
  std::unordered_map<const VarObj*, For> loop_vars2loop_;
};

StmtSRef RFactor(ScheduleState self, const StmtSRef& rf_loop_sref, int factor_axis) {
  // *****************************************************
  // *    Condition Checks and Information Collection    *
  // *****************************************************

  // Step 1. Check some basic conditions for rfactor. Get the block and block-realize.
  SBlockRealize block_realize = CheckGetSingleChildBlockRealizeOnSRefTree(self, rf_loop_sref);
  const StmtSRef& block_sref = self->stmt2ref.at(block_realize->block.get());
  const SBlock& block = block_realize->block;
  StmtSRef scope_root = GetScopeRoot(self, block_sref,  //
                                     /*require_stage_pipeline=*/true);
  if (self->enable_check) {
    CheckReductionBlock(self, block_sref, scope_root);
  }
  const ForObj* rf_loop = TVM_SREF_TO_FOR(rf_loop_sref);
  if (rf_loop->kind != ForKind::kDefault) {
    throw MakeScheduleError<NotSerialLoopKindError>(self->mod, ffi::GetRef<For>(rf_loop));
  }

  // Step 2. Collect loop vars that are touched by data parallel block iters and reduction block
  // iters, respectively.
  std::unordered_set<const VarObj*> data_par_loop_vars;
  std::unordered_set<const VarObj*> reduce_loop_vars;
  GetVarsTouchedByBlockIters(block_realize, &data_par_loop_vars, &reduce_loop_vars);

  // Step 3. Collect the loops of the reduction block. Construct a mapping from loops to
  // corresponding loop vars.
  ffi::Array<For> loops = LoopSRefs2Loops(GetLoops(block_sref));
  std::unordered_map<const VarObj*, For> loop_vars2loop = GetLoopVar2LoopMap(loops);

  // Step 4. Check four properties that the loops should have:
  // - the rfactor loop cannot be touched by any data parallel block iter;
  // - all the loops cannot be touched by both data parallel block iters and reduction block iters;
  // - the outermost loop should have the reduction block as its first child block;
  // - the outermost loop that is touched by some reduction block iters can only have one child
  // block.
  if (self->enable_check) {
    LoopPropertyError::CheckLoopProperty(self, loops, rf_loop, block, data_par_loop_vars,
                                         reduce_loop_vars);
  }

  // Step 5. Get the `init` identity and the `update` combiner of the reduction. Extract the
  // commutative reducer, combiner lhs and combiner rhs from the reduction identity and the
  // reduction combiner. The lhs will be used when constructing the write-back block, and the rhs
  // will be used when constructing the rfactor block.
  ffi::Array<PrimExpr> init_values;
  ffi::Array<TensorStore> updates;
  te::CommReducer reducer{ffi::UnsafeInit{}};
  ffi::Array<PrimExpr> combiner_lhs;
  ffi::Array<PrimExpr> combiner_rhs;
  std::tie(init_values, updates) = GetInitValuesAndUpdatesFromReductionBlock(self, block);
  std::tie(reducer, combiner_lhs, combiner_rhs) =
      GetReducerAndCombinerLhsRhs(self, init_values, updates);

  // Step 6. Check whether `factor_axis` is in a correct range, and convert it to non-negative if it
  // is negative.
  factor_axis = FactorAxisOutOfRangeError::CheckAndUpdate(
      self->mod, updates[0]->dest.as_or_throw<TensorVar>(), factor_axis);

  // *****************************************************
  // *                 IR Manipulation                   *
  // *****************************************************
  // Since rfactor splits the reduction block into two, we call the first one "rfactor block", and
  // the latter one "write-back block", and the intermediate tensor is called "rfactor tensor".

  // Step 1. Create the intermediate tensor (a.k.a. rfactor tensor), which has an additional
  // dimension that specified by `factor_axis` and `rf_loop`.
  ffi::Array<TensorVar> rf_tensors = CreateRFactorTensors(updates, factor_axis, rf_loop);

  // Step 2. Create the rfactor block.
  RFactorBlockCreator rf_block_creator(block_realize, ffi::GetRef<For>(rf_loop), updates, reducer,
                                       rf_tensors, loop_vars2loop, factor_axis,
                                       std::move(combiner_rhs));
  rf_block_creator.CreateBlock();

  // Step 3. Create the write-back block.
  WriteBackBlockCreator wb_block_creator(block_realize, ffi::GetRef<For>(rf_loop), updates, reducer,
                                         rf_tensors, std::move(rf_block_creator.additional_iter_),
                                         std::move(combiner_lhs),
                                         std::move(rf_block_creator.rf_tensor_access_indices_));
  wb_block_creator.CreateBlock();

  // Step 4. Wrap the rfactor block with loops.
  Stmt rf_body = CreateLoopOutsideRfactorBlock(rf_block_creator.new_block_realize_, loops);

  // *****************************************************
  // *           Schedule Replacement & Update           *
  // *****************************************************

  // Step 1. Substitute the old scope root block with the new scope root block.
  SBlock old_scope_root_block = ffi::GetRef<SBlock>(scope_root->StmtAs<SBlockObj>());
  SBlock new_scope_root_block = BlockReplacer::Replace(
      old_scope_root_block, rf_body, loops[0], wb_block_creator.new_block_realize_, block_realize,
      ffi::GetRef<For>(rf_loop), reduce_loop_vars, loop_vars2loop, rf_tensors);
  self->Replace(
      scope_root, new_scope_root_block,
      {{old_scope_root_block, new_scope_root_block}, {block, wb_block_creator.new_block_}});

  // Step 2. Update scope information.
  std::vector<StmtSRef> new_block_srefs{self->stmt2ref.at(rf_block_creator.new_block_.get()),
                                        self->stmt2ref.at(wb_block_creator.new_block_.get())};
  for (const StmtSRef& new_block_sref : new_block_srefs) {
    SBlockInfo& info = self->block_info[new_block_sref];
    info.affine_binding = true;
    info.region_cover = true;
    info.stage_pipeline = true;
  }
  return new_block_srefs[0];
}

/******** InstructionKind Registration ********/

struct DecomposeReductionTraits : public UnpackedInstTraits<DecomposeReductionTraits> {
  static constexpr const char* kName = "DecomposeReduction";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static SBlockRV UnpackedApplyToSchedule(Schedule sch, SBlockRV block_rv, LoopRV loop_rv) {
    return sch->DecomposeReduction(block_rv, loop_rv);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block_rv,
                                      ffi::String loop_rv) {
    PythonAPICall py("decompose_reduction");
    py.Input("block", block_rv);
    py.Input("loop", loop_rv);
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

struct RFactorTraits : public UnpackedInstTraits<RFactorTraits> {
  static constexpr const char* kName = "RFactor";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  static SBlockRV UnpackedApplyToSchedule(Schedule sch, LoopRV loop_rv, int factor_axis) {
    return sch->RFactor(loop_rv, factor_axis);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String loop_rv,
                                      int factor_axis) {
    PythonAPICall py("rfactor");
    py.Input("loop", loop_rv);
    py.Input("factor_axis", factor_axis);
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

TVM_FFI_STATIC_INIT_BLOCK() {
  RegisterInstructionKind<RFactorTraits>();
  RegisterInstructionKind<DecomposeReductionTraits>();
}

/******** FFI ********/

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "s_tir.schedule.RegisterReducer",
      [](int n_tensors, ffi::Function combiner_getter, ffi::Function identity_getter) {
        ReducerRegistry::RegisterReducer(n_tensors, std::move(combiner_getter),
                                         std::move(identity_getter));
      });
}

}  // namespace s_tir
}  // namespace tvm
