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
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/stmt.h>

#include "../utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::prim;
using namespace tvm::tirx;

/*! \brief Information used to create new padding block */
struct PaddingSBlockInfo {
  /*! \brief In-bound block iter regions, wrt loop vars. */
  ffi::Array<Range> in_bound_region;
  /*! \brief In-bound value, wrt block iter vars. */
  PrimExpr in_bound_value;
  /*! \brief Condition of in-bound write, wrt loop vars. */
  PrimExpr in_bound_predicate;
  /*! \brief Padding value, should be a constant. */
  PrimExpr pad_value;
};

class PaddingPatternMatchError : public ScheduleErrorContextObj {
 public:
  PaddingPatternMatchError(IRModule mod, SBlock block, const std::string& error_msg)
      : mod_(std::move(mod)), block_(std::move(block)), error_msg_(error_msg) {}

  ffi::String FastErrorString() const final {
    return "ScheduleError: decompose_padding expect the block to match padding pattern\n  " +
           error_msg_;
  }

  ffi::String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "ScheduleError: decompose_padding expect the block {0} to match padding pattern\n  "
       << error_msg_;
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_}; }

  IRModule mod_;
  SBlock block_;
  std::string error_msg_;
};

/*!
 * \brief Helper class to analyze and check the padding pattern of the block,
 * then return the padding information.
 */
class PaddingInfoAnalyzer {
 public:
  static PaddingSBlockInfo CheckAndGetPaddingInfo(IRModule mod, const SBlockRealizeObj* realize,
                                                  const ffi::Map<PrimVar, Range>& dom_map,
                                                  sym::AnalyzerObj* analyzer) {
    PaddingInfoAnalyzer padding_analyzer(analyzer);
    if (!padding_analyzer.MatchPadding(realize, dom_map)) {
      throw MakeScheduleError<PaddingPatternMatchError>(mod, realize->block,
                                                        padding_analyzer.error_msg_);
    }
    return padding_analyzer.info_.value();
  }

 private:
  explicit PaddingInfoAnalyzer(sym::AnalyzerObj* analyzer) : analyzer_(analyzer) {}

  /*! \brief Detect padding pattern and update result. */
  bool MatchPadding(const SBlockRealizeObj* realize, const ffi::Map<PrimVar, Range>& dom_map) {
    // Step 1. Check match padding computation pattern.
    // A[...] = T.if_then_else(predicate, B[...], imm)
    SBlock block = realize->block;
    std::unordered_map<const VarObj*, PrimExpr> iter_values;
    for (size_t i = 0; i < realize->iter_values.size(); ++i) {
      Var block_var = block->iter_vars[i]->var;
      iter_values.insert_or_assign(block_var.get(), realize->iter_values[i]);
    }
    const TensorStoreObj* store =
        block->body->size() == 1 ? block->body->seq[0].as<TensorStoreObj>() : nullptr;
    if (!store) {
      SetError("Block body expect a TensorStore to the write tensor");
      return false;
    }
    const CallObj* if_then_else = store->value.as<CallObj>();
    if (!if_then_else || !if_then_else->op.same_as(prim::if_then_else_op())) {
      SetError("Value of TensorStore expect to be constrained by a padding predicate");
      return false;
    }
    auto f_substitute =
        [&iter_values](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
      if (auto it = iter_values.find(var.get()); it != iter_values.end()) {
        return ffi::Any(it->second);
      }
      return ffi::Unchanged();
    };
    PrimExpr pad_predicate = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(
                                 if_then_else->args[0].as_or_throw<PrimExpr>(), f_substitute)
                                 .as_or_throw<PrimExpr>();
    PrimExpr in_bound_value = if_then_else->args[1].as_or_throw<PrimExpr>();
    PrimExpr pad_value = if_then_else->args[2].as_or_throw<PrimExpr>();
    if (!IsConstNumber(pad_value)) {
      SetError("Pad value should be constant");
      return false;
    }

    // Step 2. Check in-bound computation to be effectiveless.
    if (SideEffect(if_then_else->args[1].as_or_throw<PrimExpr>()) > CallEffectKind::kReadState) {
      SetError("Inbound computation should not have side-effect");
      return false;
    }

    // Step 3. Analyze in-bound write region.
    PrimExpr in_bound_predicate = RewritePredicate(pad_predicate && realize->predicate);
    if (analyzer_->CanProveEqual(in_bound_predicate, 1)) {
      SetError("The in-bound predicate is trivial");
      return false;
    }
    ffi::Array<Range> in_bound_region = this->EstimateInBoundRegion(
        /*iter_values=*/realize->iter_values, /*dom_map=*/dom_map,
        /*in_bound_predicate=*/in_bound_predicate);
    if (in_bound_region.empty()) {
      return false;
    }

    // Step 4. Update result information.
    info_ = PaddingSBlockInfo{in_bound_region, if_then_else->args[1].as_or_throw<PrimExpr>(),
                              in_bound_predicate, pad_value};
    return true;
  }

  /*! \brief Rewrite predicate to left recursive conjunction, drop likely annotation. */
  PrimExpr RewritePredicate(const PrimExpr& predicate) {
    PrimExpr res = IntImm::Bool(true);
    std::function<void(PrimExpr)> update = [&res, &update](PrimExpr e) {
      if (const auto* and_node = e.as<prim::AndObj>()) {
        update(and_node->a);
        update(and_node->b);
      } else {
        if (const CallObj* call = e.as<CallObj>()) {
          if (call->op.same_as(prim::likely_op())) {
            e = call->args[0].as_or_throw<PrimExpr>();
          }
        }
        res = res && e;
      }
    };
    update(predicate);
    return analyzer_->Simplify(res);
  }

  /*! \brief Return iteration region of block vars where the padding predicate evals to true. */
  ffi::Array<Range> EstimateInBoundRegion(const ffi::Array<PrimExpr>& iter_values,
                                          const ffi::Map<PrimVar, Range>& dom_map,
                                          const PrimExpr& in_bound_predicate) {
    ffi::Array<Range> region;

    sym::Analyzer analyzer_ref = ffi::GetRef<sym::Analyzer>(analyzer_);
    auto res = sym::DetectIterMap(iter_values, dom_map, in_bound_predicate,
                                  sym::IterMapLevel::Surjective, analyzer_ref);
    if (res->indices.empty()) {
      SetError("Block iters are not independent wrt padding condition");
      return {};
    }
    for (const sym::IterSumExpr& sum : res->indices) {
      if (sum->args.empty()) {
        region.push_back(Range::FromMinExtent(sum->base, IntImm(sum->base.ty(), /* value */ 1)));
      } else {
        TVM_FFI_ICHECK_EQ(sum->args.size(), 1U);
        if (!analyzer_->CanProveEqual(sum->args[0]->scale, 1)) {
          SetError("Strided iteration is not supported");
          return {};
        }
        region.push_back(Range::FromMinExtent(sum->base, sum->args[0]->extent));
      }
    }
    return region;
  }

  void SetError(const std::string& msg) { error_msg_ = msg; }

  /*! \brief padding info analyse result. */
  std::optional<PaddingSBlockInfo> info_;
  /*! \brief current error message. */
  std::string error_msg_;
  /*! \brief arithmetic analyzer. */
  sym::AnalyzerObj* analyzer_;
};

/*! \brief Create block to fill constant pad values into full region */
static std::pair<Stmt, SBlockRealize> CreateConstBlock(const SBlockRealizeObj* realize,
                                                       const PaddingSBlockInfo& info,
                                                       const ffi::Array<For>& loops,
                                                       const Stmt& highest_pos_inclusive,
                                                       sym::AnalyzerObj* analyzer) {
  const SBlock& block = realize->block;
  ffi::Array<IterVar> new_iter_vars;
  ffi::Map<Var, PrimExpr> repl_dict;

  // create new block itervars
  for (size_t i = 0; i < block->iter_vars.size(); ++i) {
    const IterVar& origin_iter = block->iter_vars[i];
    PrimVar new_var = origin_iter->var.CopyWithSuffix("");
    new_iter_vars.push_back(IterVar(origin_iter->dom, new_var, IterVarType::kDataPar));
    repl_dict.Set(origin_iter->var, new_var);
  }

  auto f_substitute = [&repl_dict](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (auto repl = repl_dict.Get(var)) return ffi::Any(*std::move(repl));
    return ffi::Unchanged();
  };
  // rewrite expr helper
  auto rewrite_expr = [&f_substitute, analyzer](const PrimExpr& e) {
    // The replacement map contains only fresh variables, so pre-order is safe here.
    return analyzer->Simplify(
        ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(e, f_substitute).as_or_throw<PrimExpr>());
  };

  // create new write region
  TVM_FFI_ICHECK_EQ(block->writes.size(), 1U);
  TensorRegion write_region =
      TensorRegion(block->writes[0]->source.as_or_throw<tvm::tirx::TensorVar>(),
                   block->writes[0]->region.Map([rewrite_expr](const Range& r) {
                     return Range::FromMinExtent(rewrite_expr(r->begin), rewrite_expr(r->extent));
                   }),
                   TensorRegionType());

  // create block to fill const pad values
  TVM_FFI_ICHECK_EQ(block->body->size(), 1);
  TensorStore store = block->body->seq[0].as_or_throw<TensorStore>();
  store.CopyOnWrite()->value = info.pad_value;
  store.CopyOnWrite()->indices = store->indices.Map(rewrite_expr);
  SBlock new_block(/*iter_vars=*/new_iter_vars, /*reads=*/{}, /*writes=*/{write_region},
                   /*name_hint=*/block->name_hint + "_pad_const", /*body=*/std::move(store));

  // create new loop vars
  ffi::Array<Var> new_loop_vars;
  for (const For& loop : loops) {
    Var new_var = loop->var.CopyWithSuffix("");
    new_loop_vars.push_back(new_var);
    repl_dict.Set(loop->var, new_var.as_or_throw<PrimExpr>());
    if (loop.same_as(highest_pos_inclusive)) {
      break;
    }
  }

  // create new block realize node
  ffi::Array<PrimExpr> new_iter_values;
  for (size_t i = 0; i < realize->iter_values.size(); ++i) {
    new_iter_values.push_back(rewrite_expr(realize->iter_values[i]));
  }
  SBlockRealize new_realize(/*iter_values=*/new_iter_values,
                            /*predicate=*/rewrite_expr(realize->predicate),
                            /*block=*/new_block);

  // create new loops
  Stmt nest_stmt_root = new_realize;
  for (size_t i = 0; i < new_loop_vars.size(); ++i) {
    For loop = loops[i];
    nest_stmt_root = For(new_loop_vars[i].as_or_throw<PrimVar>(), loop->begin, loop->end,
                         IntImm(new_loop_vars[i].as_or_throw<PrimVar>().ty(), 1), ForKind::kDefault,
                         nest_stmt_root);
  }

  return {nest_stmt_root, new_realize};
}

/*! \brief Create block to fill in-bound region values. */
static std::pair<Stmt, SBlockRealize> CreateInBoundBlock(const SBlockRealizeObj* realize,
                                                         const PaddingSBlockInfo& info,

                                                         const ffi::Array<For>& loops,
                                                         const Stmt& highest_pos_inclusive,
                                                         sym::AnalyzerObj* analyzer) {
  const SBlock& block = realize->block;
  ffi::Array<IterVar> new_iter_vars;
  ffi::Map<Var, PrimExpr> repl_dict;

  // record loop ranges to be mutated
  ffi::Map<Var, Range> new_loop_ranges;
  for (const For& loop : loops) {
    new_loop_ranges.Set(loop->var, Range(loop->begin, loop->end));
    if (loop.same_as(highest_pos_inclusive)) {
      break;
    }
  }

  // create new block iter vars and iter bindings
  ffi::Array<PrimExpr> new_iter_binding;
  for (size_t i = 0; i < info.in_bound_region.size(); ++i) {
    // add new block itervar
    const IterVar& origin_itervar = block->iter_vars[i];
    PrimVar new_var = origin_itervar->var.CopyWithSuffix("");
    Range new_range =
        Range::FromMinExtent(IntImm(new_var.ty(), 0), info.in_bound_region[i]->extent);
    new_iter_vars.push_back(IterVar(new_range, new_var, IterVarType::kDataPar));
    repl_dict.Set(origin_itervar->var, new_var + info.in_bound_region[i]->begin);

    // update new loop range
    if (auto opt = realize->iter_values[i].as<Var>(); opt && new_loop_ranges.count(opt.value())) {
      // if the block binding is the loop var with single child, mutate loop range
      // instead of insert extra block predicate
      auto loop_var = opt.value();
      new_loop_ranges.Set(loop_var, new_range);
      new_iter_binding.push_back(realize->iter_values[i]);
      repl_dict.Set(loop_var, loop_var.as_or_throw<PrimExpr>() + info.in_bound_region[i]->begin);
      {
        sym::ConstraintContext context;
        analyzer->GetConstraintContext()->ForEach([&](const sym::Constraint& premise) {
          if (auto binding = premise.as<sym::BindingConstraintObj>();
              binding && binding->var.same_as(loop_var))
            return;
          if (auto range = premise.as<sym::RangeConstraintObj>();
              range && range->var.same_as(loop_var))
            return;
          context->Append(premise);
        });
        context->Append(sym::RangeConstraint(loop_var, new_range));
        analyzer->SyncConstraintContext(context);
      }
    } else {
      new_iter_binding.push_back(
          analyzer->Simplify(realize->iter_values[i] - info.in_bound_region[i]->begin));
    }
  }

  auto f_substitute = [&repl_dict](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (auto repl = repl_dict.Get(var)) return ffi::Any(*std::move(repl));
    return ffi::Unchanged();
  };
  // rewrite helpers
  auto rewrite_expr = [&f_substitute, analyzer](const PrimExpr& e) {
    // The map is self-referential, so post-order must not revisit replacement expressions.
    return analyzer->Simplify(
        ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(e, f_substitute).as_or_throw<PrimExpr>());
  };
  auto rewrite_region = [rewrite_expr](const ffi::Array<Range>& region) {
    return region.Map([rewrite_expr](const Range& r) {
      return Range::FromMinExtent(rewrite_expr(r->begin), rewrite_expr(r->extent));
    });
  };

  // create new read/write region for in-bound accesses
  ffi::Array<TensorRegion> reads, writes;
  for (const TensorRegion& read : block->reads) {
    reads.push_back(TensorRegion(read->source.as_or_throw<tvm::tirx::TensorVar>(),
                                 rewrite_region(read->region), TensorRegionType()));
  }
  for (const TensorRegion& write : block->writes) {
    writes.push_back(TensorRegion(write->source.as_or_throw<tvm::tirx::TensorVar>(),
                                  rewrite_region(write->region), TensorRegionType()));
  }

  // create new block realize node
  TVM_FFI_ICHECK_EQ(block->body->size(), 1);
  TensorStore store = block->body->seq[0].as_or_throw<TensorStore>();
  store.CopyOnWrite()->value = rewrite_expr(info.in_bound_value);
  store.CopyOnWrite()->indices = store->indices.Map(rewrite_expr);
  SBlock new_block(/*iter_vars=*/new_iter_vars, /*reads=*/reads, /*writes=*/writes,
                   /*name_hint=*/block->name_hint, /*body=*/std::move(store));
  PrimExpr new_predicate = rewrite_expr(info.in_bound_predicate);
  SBlockRealize new_realize(/*iter_values=*/new_iter_binding, /*predicate=*/new_predicate,
                            /*block=*/new_block);

  // create new loops
  Stmt nest_stmt_root = new_realize;
  for (const For& loop : loops) {
    auto it = new_loop_ranges.find(loop->var);
    PrimExpr min = it == new_loop_ranges.end() ? loop->begin : (*it).second->begin;
    PrimExpr extent = it == new_loop_ranges.end()
                          ? Range::GetExtentFromBeginEnd(loop->begin, loop->end)
                          : (*it).second->extent;
    nest_stmt_root = For(loop->var, min, min + extent, loop->step, loop->kind, nest_stmt_root,
                         loop->annotations, loop->loc);
    if (loop.same_as(highest_pos_inclusive)) {
      break;
    }
  }
  return {nest_stmt_root, new_realize};
}

/*!
 * \brief A helper class to create a new scope that contains decomposed padding blocks.
 */
class DecomposePaddingBlockReplacer : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) override {
    if (value.as<ExprObj>()) return ffi::Unchanged();
    return StmtExprMutator::Mutate(value, inplace_mode);
  }

  /*! \brief Replacement information */
  struct ReplaceDesc {
    /*! \brief loop above which to insert const pad value filling code. */
    For const_filling_pos;
    /*! \brief loop under which to insert in bound value filling code. */
    For in_bound_filling_pos;
    /*! \brief const pad value filling loop. */
    Stmt const_filling_loop;
    /*! \brief highest in bound value filling loop with single child. */
    Stmt in_bound_filling_loop;
    /*! \brief const pad value filling block. */
    SBlockRealize const_filling_block;
    /*! \brief in bound value filling block. */
    SBlockRealize in_bound_filling_block;
  };

  static SBlock Replace(SBlock scope_root, const ReplaceDesc& desc) {
    auto replacer = ffi::make_object<DecomposePaddingBlockReplacer>(desc);
    return replacer->Mutate(scope_root, InplaceMode::kAllow)
        .ValueOrUnchanged(std::move(scope_root))
        .as_or_throw<SBlock>();
  }

  explicit DecomposePaddingBlockReplacer(const ReplaceDesc& desc) : desc_(desc) {}

 private:
  UnchangedOr<Stmt> Mutate_(const ForObj* op, InplaceMode inplace_mode) final {
    Stmt new_loop{ffi::UnsafeInit{}};
    if (op == desc_.in_bound_filling_pos.get()) {
      // position to rewrite inbound filling code
      new_loop = desc_.in_bound_filling_loop;
    } else {
      new_loop = StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    }
    if (op == desc_.const_filling_pos.get()) {
      // position to insert pad value filling code
      return std::move(SeqStmt({desc_.const_filling_loop, new_loop}));
    }
    return new_loop;
  }

  const ReplaceDesc& desc_;
};

StmtSRef DecomposePaddingImpl(ScheduleState self, const StmtSRef& block_sref,
                              const StmtSRef& loop_sref, bool check_only) {
  /*!
   *  Check
   *    - the block is a compact block
   *    - the loop is an ancester of the block
   *    - the block match padding pattern
   *  Mutate
   *    - generate new block to fill padding values
   *    - trim original block to write non-padding part only
   */
  // Condition Checks and Information Collection
  const SBlockObj* block = TVM_SREF_TO_SBLOCK(block_sref);
  const SBlockRealizeObj* realize = GetSBlockRealize(self, block_sref).get();
  ffi::Map<PrimVar, Range> dom_map;
  sym::Analyzer analyzer;

  // Check 1. check the block is complete.
  StmtSRef scope_root_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/false);
  CheckCompleteBlock(self, block_sref, scope_root_sref);

  // Check 2. Check loop_sref is an ancestor of block_sref. Also collect
  //   - the highest loop position (inclusive) to insert const pad value filling code above.
  //   - the highest loop position (inclusive) to replace with in-bound value filling code.
  ffi::Array<StmtSRef> loop_srefs = GetLoops(block_sref);
  ffi::Array<For> loops;
  bool found_const_filling_pos = false;
  bool found_in_bound_filling_pos = false;
  For const_filling_pos = ffi::GetRef<For>(loop_sref->StmtAs<ForObj>());
  ffi::Optional<For> in_bound_filling_pos;
  for (auto it = loop_srefs.rbegin(); it != loop_srefs.rend(); ++it) {
    For cur_loop = ffi::GetRef<For>((*it)->StmtAs<ForObj>());
    Range range = Range(cur_loop->begin, cur_loop->end);
    dom_map.Set(cur_loop->var, range);
    {
      auto context = analyzer->GetConstraintContext();
      context->Append(sym::RangeConstraint(cur_loop->var, range));
      analyzer->SyncConstraintContext(context);
    }
    loops.push_back(cur_loop);

    if (cur_loop.same_as(const_filling_pos)) {
      TVM_FFI_ICHECK(!found_const_filling_pos);
      found_const_filling_pos = true;
      if (!found_in_bound_filling_pos) {
        found_in_bound_filling_pos = true;
        in_bound_filling_pos = cur_loop;
      }
    } else if (!found_in_bound_filling_pos) {
      if (cur_loop->body->size() != 1 ||
          (!cur_loop->body->seq[0]->IsInstance<ForObj>() &&
           !cur_loop->body->seq[0]->IsInstance<SBlockRealizeObj>())) {
        found_in_bound_filling_pos = true;
      } else {
        in_bound_filling_pos = cur_loop;
      }
    }
  }
  TVM_FFI_ICHECK(in_bound_filling_pos.has_value());
  if (!found_const_filling_pos) {
    throw MakeScheduleError<LoopPositionError>(self->mod, const_filling_pos,
                                               ffi::GetRef<SBlock>(block), "decompose_padding");
  }

  // Check 3. match padding pattern and return padding operation info.
  PaddingSBlockInfo info =
      PaddingInfoAnalyzer::CheckAndGetPaddingInfo(self->mod, realize, dom_map, analyzer.get());

  // IR Manipulation
  // Step 1. Create const pad value filling part and in-bound value filling part.
  auto [const_loop, const_block] =
      CreateConstBlock(realize, info, loops, const_filling_pos, analyzer.get());
  auto [in_bound_loop, in_bound_block] =
      CreateInBoundBlock(realize, info, loops, in_bound_filling_pos.value(), analyzer.get());
  DecomposePaddingBlockReplacer::ReplaceDesc replace_desc{
      const_filling_pos, in_bound_filling_pos.value(), const_loop, in_bound_loop, const_block,
      in_bound_block};

  // Step 2. Execute IR replacement.
  SBlock old_scope_root_block = ffi::GetRef<SBlock>(scope_root_sref->StmtAs<SBlockObj>());
  SBlock new_scope_root =
      DecomposePaddingBlockReplacer::Replace(old_scope_root_block, replace_desc);
  if (check_only) {
    return block_sref;
  }

  // Step 3. Update schedule states.
  self->Replace(scope_root_sref, new_scope_root,
                {{old_scope_root_block, new_scope_root},
                 {ffi::GetRef<SBlock>(block), replace_desc.in_bound_filling_block->block}});
  auto new_block_sref = self->stmt2ref.at(replace_desc.const_filling_block->block.get());

  // Set block info of created const pad value filling block
  SBlockInfo& block_info = self->block_info[new_block_sref];
  block_info.affine_binding = true;
  block_info.region_cover = true;
  block_info.stage_pipeline = true;

  // If the const pad value filling block is lifted out of the original subtree,
  // set the region_cover flag as false since region_cover is the property under the subtree.
  bool preserve_stage_pipeline = true;
  for (const StmtSRef& consumer_sref : GetConsumers(self, block_sref)) {
    StmtSRef lca = GetSRefLowestCommonAncestor({consumer_sref, block_sref});
    const StmtSRefObj* parent = new_block_sref->parent;
    bool is_under_lca = false;
    while (parent) {
      if (parent == lca.get()) {
        is_under_lca = true;
        break;
      }
      parent = parent->parent;
    }
    if (!is_under_lca) {
      preserve_stage_pipeline = false;
      self->block_info[consumer_sref].region_cover = false;
    }
  }
  if (!preserve_stage_pipeline) {
    self->block_info[scope_root_sref].stage_pipeline = false;
  }
  return new_block_sref;
}

StmtSRef DecomposePadding(ScheduleState self, const StmtSRef& block_sref,
                          const StmtSRef& loop_sref) {
  return DecomposePaddingImpl(self, block_sref, loop_sref, false);
}

bool CanDecomposePadding(ScheduleState self, const StmtSRef& block_sref,
                         const StmtSRef& loop_sref) {
  try {
    DecomposePaddingImpl(self, block_sref, loop_sref, true);
  } catch (const tvm::ffi::Error& e) {
    return false;
  }
  return true;
}

/******** FFI ********/

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "s_tir.schedule.CanDecomposePadding", [](Schedule self, SBlockRV block_rv, LoopRV loop_rv) {
        return CanDecomposePadding(self->state(), self->GetSRef(block_rv), self->GetSRef(loop_rv));
      });
}

/******** InstructionKind Registration ********/

struct DecomposPaddingTraits : public UnpackedInstTraits<DecomposPaddingTraits> {
  static constexpr const char* kName = "DecomposePadding";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static SBlockRV UnpackedApplyToSchedule(Schedule sch, SBlockRV block_rv, LoopRV loop_rv) {
    return sch->DecomposePadding(block_rv, loop_rv);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block_rv,
                                      LoopRV loop_rv) {
    PythonAPICall py("decompose_padding");
    py.Input("block", block_rv);
    py.Input("loop", loop_rv);
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

TVM_FFI_STATIC_INIT_BLOCK() { RegisterInstructionKind<DecomposPaddingTraits>(); }

}  // namespace s_tir
}  // namespace tvm
