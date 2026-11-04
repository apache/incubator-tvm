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

/*!
 * \file loop_partition.cc
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/transform.h>
#include <tvm/support/gpu_thread_tag.h>
#include <tvm/sym/analyzer.h>
#include <tvm/sym/int_set.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op/annotation.h>
#include <tvm/tirx/op/gpu.h>
#include <tvm/tirx/op/region.h>

#include <optional>
#include <unordered_map>
#include <unordered_set>

#include "ir_utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::prim;
using namespace tvm::tirx;

struct LoopPartitionConfigObj : public ffi::Object {
  bool partition_const_loop;
  bool no_unroll_loop_with_extent_one;
  bool unroll_loop_with_partition_hint_no_interval;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<LoopPartitionConfigObj>()
        .def_ro("partition_const_loop", &LoopPartitionConfigObj::partition_const_loop,
                "Split constant loop", refl::DefaultValue(false))
        .def_ro("no_unroll_loop_with_extent_one",
                &LoopPartitionConfigObj::no_unroll_loop_with_extent_one,
                "Don't unroll loops with extent 1", refl::DefaultValue(false))
        .def_ro("unroll_loop_with_partition_hint_no_interval",
                &LoopPartitionConfigObj::unroll_loop_with_partition_hint_no_interval,
                "Unroll loops with loop_partition_hint and no interval", refl::DefaultValue(false));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.transform.LoopPartitionConfig", LoopPartitionConfigObj,
                                    ffi::Object);
};

TVM_FFI_STATIC_INIT_BLOCK() { LoopPartitionConfigObj::RegisterReflection(); }

class LoopPartitionConfig : public ffi::ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(LoopPartitionConfig, ffi::ObjectRef,
                                                LoopPartitionConfigObj);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  ::tvm::transform::PassContext::RegisterConfigOption<LoopPartitionConfig>("s_tir.LoopPartition");
}

using sym::DeduceBound;
using sym::Intersect;
using sym::IntSet;

using PartitionKey = std::pair<PrimExpr, bool>;
struct PartitionKeyHash {
  std::size_t operator()(PartitionKey const& k) const noexcept {
    std::size_t h1 = ffi::ObjectPtrHash{}(k.first);  // NOLINT(whitespace/braces)
    std::size_t h2 = std::hash<bool>{}(k.second);
    return h1 ^ h2;
  }
};

struct PartitionKeyEqual {
  bool operator()(const PartitionKey& k1, const PartitionKey& k2) const {
    // NOLINTNEXTLINE(whitespace/braces)
    return k1.second == k2.second && ffi::ObjectPtrEqual{}(k1.first, k2.first);
  }
};

// Each mapping (cond, cond_value) -> interval represents the fact that
// condition cond is proven to have value cond_value (true or false) in interval.
using Partition = std::unordered_map<PartitionKey, IntSet, PartitionKeyHash, PartitionKeyEqual>;

using ExpressionSet = std::unordered_set<PrimExpr, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

// Virtual threads are lowered separately and are not hardware partition scopes.
static bool IsVirtualThread(const ForObj* op) {
  if (!tvm::s_tir::GetThreadBinding(op).has_value()) return false;
  const auto tag = tvm::s_tir::GetThreadBinding(op).value();
  return tag.starts_with("vthread");
}

static bool HasPartitionHint(const ForObj* op) {
  auto it = op->annotations.find(tvm::s_tir::attr::kLoopPartitionHint.name);
  return it != op->annotations.end() && IsOne((*it).second.as_or_throw<PrimExpr>());
}

// Resolve expression-valued hints before partitioning clones or renames loops.
// Like the original hint selection, this proof does not assume loop bounds.
class NormalizePartitionHints : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  UnchangedOr<Stmt> Mutate_(const ForObj* op, InplaceMode inplace_mode) final {
    For loop = StmtExprMutator::Mutate_(op, inplace_mode)
                   .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                   .as_or_throw<For>();
    auto it = loop->annotations.find(tvm::s_tir::attr::kLoopPartitionHint.name);
    if (it != loop->annotations.end()) {
      auto value = (*it).second;
      if (value == nullptr) {
        loop.CopyOnWrite()->annotations.erase(tvm::s_tir::attr::kLoopPartitionHint.name);
      } else {
        bool enabled = analyzer_->CanProve(value.cast<PrimExpr>());
        loop.CopyOnWrite()->annotations.Set(tvm::s_tir::attr::kLoopPartitionHint.name,
                                            IntImm::Bool(enabled));
      }
    }
    return loop;
  }

 private:
  sym::Analyzer analyzer_;
};

// Select potential candidate IRs that can be partitioned.
// Rule:
//   - the range should not be const
//   - there exist a condition expression in the scope that use the var
class CandidateSelector final : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;
  using VarIsUsed = bool;
  explicit CandidateSelector(bool partition_const_loop)
      : partition_const_loop_(partition_const_loop) {}

  ffi::Optional<VisitInterrupt> Visit_(const ForObj* op) final {
    if (IsVirtualThread(op)) return StmtExprVisitor::Visit_(op);
    const VarObj* var = op->var.get();
    if (HasPartitionHint(op)) {
      candidates.insert(ffi::GetRef<Stmt>(op));
      return StmtExprVisitor::Visit_(op);
    }
    if (tvm::s_tir::GetThreadBinding(op).has_value() &&
        support::GPUThreadTag::Parse(tvm::s_tir::GetThreadBinding(op).value()).rank != 0) {
      return StmtExprVisitor::Visit_(op);
    }
    // partition const loop when sets partition_const_loop_
    if (!IsConstInt(op->begin) || !IsConstInt(Range::GetExtentFromBeginEnd(op->begin, op->end)) ||
        partition_const_loop_) {
      record_.insert({var, false});
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
      if (record_.at(var) && !no_split_) {
        candidates.insert(ffi::GetRef<Stmt>(op));
      }
      record_.erase(var);
    } else {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const RegionStmtObj* op) final {
    if (op->op.same_as(tirx::launch_thread_op()) &&
        !op->args[0].as_or_throw<StringImm>()->value.starts_with("vthread")) {
      Var var = op->body_params[0].as_or_throw<PrimVar>();
      support::GPUThreadTag scope =
          support::GPUThreadTag::Parse(op->args[0].as_or_throw<StringImm>()->value);
      PrimExpr extent = op->args[1].as_or_throw<PrimExpr>();
      if ((scope.rank == 0) && (!IsConstInt(extent) || partition_const_loop_)) {
        record_.insert({var.get(), false});
        TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
        if (record_.at(var.get()) && !no_split_) {
          candidates.insert(ffi::GetRef<Stmt>(op));
        }
        record_.erase(var.get());
        return std::nullopt;
      }
    }
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const SeqStmtObj* op) final {
    bool init_no_split = no_split_;
    for (Stmt stmt : op->seq) {
      // erase the no split state of before visiting the next one.
      bool temp = init_no_split;
      std::swap(temp, no_split_);
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(stmt));
      // restore the no split flag.
      no_split_ = no_split_ || temp;
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const CallObj* op) final {
    if (op->op.same_as(prim::likely_op())) {
      in_likely_ = true;
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
      in_likely_ = false;
    } else if (op->op.same_as(tirx::gpu_thread_allreduce_op())) {
      // no split if the body contains allreduce.
      no_split_ = true;
      return std::nullopt;
    } else {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const VarObj* op) final {
    if (in_likely_ && record_.count(op)) {
      record_.at(op) = true;
    }
    return StmtExprVisitor::Visit_(op);
  }

  std::unordered_set<Stmt, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> candidates;

 private:
  bool in_likely_{false};
  bool no_split_{false};
  bool partition_const_loop_{false};
  std::unordered_map<const VarObj*, VarIsUsed> record_;
};

// Finder try best to find partitions for hinted vars
#define DEFINE_PARTITION_FINDER_VISIT_CMP_OP(OpNodeT)             \
  ffi::Optional<VisitInterrupt> Visit_(const OpNodeT* op) final { \
    if (has_partition_hint_) {                                    \
      DeduceCondition(ffi::GetRef<PrimExpr>(op));                 \
      return std::nullopt;                                        \
    }                                                             \
    return StmtExprVisitor::Visit_(op);                           \
  }

// Populate partitions data structure, i.e., for a specific variable,
// find an interval in which each condition has fixed true or false value
class PartitionFinder : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;
  explicit PartitionFinder(Var current_var, ffi::Map<Var, IntSet>& hint_map,
                           ffi::Map<Var, IntSet>& relax_map, bool has_partition_hint)
      : current_var_(current_var),
        has_partition_hint_(has_partition_hint),
        hint_map_(hint_map),
        relax_map_(relax_map) {
    for (const auto& kv : hint_map) {
      out_vars_.insert(kv.first.get());
    }
    for (const auto& kv : relax_map) {
      out_vars_.insert(kv.first.get());
    }
  }

  ffi::Optional<VisitInterrupt> Visit_(const ForObj* op) final {
    if (IsVirtualThread(op)) return StmtExprVisitor::Visit_(op);
    auto f_vset_contains = [this](const VarObj* var) { return out_vars_.count(var); };
    auto walkfn = [&](const Var& var) -> ffi::Expected<ffi::WalkResult> {
      return f_vset_contains(var.get()) ? ffi::WalkResult::Interrupt(ffi::VisitInterrupt(var))
                                        : ffi::WalkResult::Advance();
    };
    if (ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(op->begin, walkfn).has_value() ||
        ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
            Range::GetExtentFromBeginEnd(op->begin, op->end), walkfn)
            .has_value()) {
      return std::nullopt;
    }

    const Var& var = op->var;
    hint_map_.Set(var, IntSet::Interval(op->begin, op->end - 1));
    relax_map_.Set(var, IntSet::Interval(op->begin, op->end - 1));
    auto result = StmtExprVisitor::Visit_(op);
    relax_map_.erase(var);
    hint_map_.erase(var);
    return result;
  }

  ffi::Optional<VisitInterrupt> Visit_(const RegionStmtObj* op) final {
    // handle thread_axis
    if (op->op.same_as(tirx::launch_thread_op()) &&
        !op->args[0].as_or_throw<StringImm>()->value.starts_with("vthread")) {
      Var var = op->body_params[0].as_or_throw<PrimVar>();
      PrimExpr extent = op->args[1].as_or_throw<PrimExpr>();
      IntSet dom = IntSet::FromRange(Range(IntImm(extent.ty(), 0), extent));
      hint_map_.Set(var, dom);
      relax_map_.Set(var, dom);
      auto result = StmtExprVisitor::Visit_(op);
      relax_map_.erase(var);
      hint_map_.erase(var);
      return result;
    } else {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const CallObj* op) final {
    if (op->op.same_as(prim::likely_op())) {
      DeduceCondition(op->args[0].as_or_throw<PrimExpr>());
    } else if (op->op.same_as(tirx::ignore_loop_partition_op())) {
      return std::nullopt;
    } else {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
    }
    return std::nullopt;
  }

  DEFINE_PARTITION_FINDER_VISIT_CMP_OP(GEObj);
  DEFINE_PARTITION_FINDER_VISIT_CMP_OP(GTObj);
  DEFINE_PARTITION_FINDER_VISIT_CMP_OP(LEObj);
  DEFINE_PARTITION_FINDER_VISIT_CMP_OP(LTObj);
  DEFINE_PARTITION_FINDER_VISIT_CMP_OP(EQObj);
  DEFINE_PARTITION_FINDER_VISIT_CMP_OP(NEObj);

  Partition partitions;

 private:
  void DeduceCondition(const PrimExpr& cond) {
    // For cond, find out the interval, if exists, in which we can prove that cond is
    // true. Also find the interval, if exists, in which we can prove that cond is
    // false.
    auto walkfn = [this](const Var& var) -> ffi::Expected<ffi::WalkResult> {
      return var.get() == current_var_.get() ? ffi::WalkResult::Interrupt(ffi::VisitInterrupt(var))
                                             : ffi::WalkResult::Advance();
    };
    if (ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(cond, walkfn).has_value()) {
      IntSet interval =
          DeduceBound(current_var_.as_or_throw<PrimExpr>(), cond, hint_map_, relax_map_);
      if (!interval.IsNothing()) {
        // cond is true within interval
        partitions.insert_or_assign({cond, true}, interval);
      }

      if (interval.IsNothing()) {
        // `DeduceBound` do not support NE now, thus when
        // deduce l==r failed, just only try (l<=r && l>=r)
        if (const EQObj* op = cond.as<EQObj>()) {
          IntSet part1 = DeduceBound(current_var_.as_or_throw<PrimExpr>(), GE(op->a, op->b),
                                     hint_map_, relax_map_);
          IntSet part2 = DeduceBound(current_var_.as_or_throw<PrimExpr>(), LE(op->a, op->b),
                                     hint_map_, relax_map_);
          interval = sym::Intersect({part1, part2});
          if (!interval.IsNothing()) {
            // cond is true within interval
            partitions.insert_or_assign({cond, true}, interval);
            return;
          }
        }
      }

      auto inverse_cond = InverseCond(cond);
      if (inverse_cond.has_value()) {
        IntSet interval = DeduceBound(current_var_.as_or_throw<PrimExpr>(), inverse_cond.value(),
                                      hint_map_, relax_map_);
        if (!interval.IsNothing()) {
          // cond is false within interval
          partitions.insert_or_assign({cond, false}, interval);
        }
      }
    }
  }

  ffi::Optional<PrimExpr> InverseCond(const PrimExpr& cond) {
    ffi::Optional<PrimExpr> inverse_cond;
    if (const LTObj* op = cond.as<LTObj>()) {
      // a < b -> a >= b
      inverse_cond = GE(op->a, op->b);
    } else if (const GTObj* op = cond.as<GTObj>()) {
      // a > b -> a <= b
      inverse_cond = LE(op->a, op->b);
    } else if (const LEObj* op = cond.as<LEObj>()) {
      // a <= b -> a > b
      inverse_cond = GT(op->a, op->b);
    } else if (const GEObj* op = cond.as<GEObj>()) {
      // a >= b -> a < b
      inverse_cond = LT(op->a, op->b);
    } else if (const EQObj* op = cond.as<EQObj>()) {
      // a == b -> a != b
      inverse_cond = NE(op->a, op->b);
      // a != b -> a == b
    } else if (const NEObj* op = cond.as<NEObj>()) {
      inverse_cond = EQ(op->a, op->b);
    }
    return inverse_cond;
  }

  Var current_var_;
  bool has_partition_hint_;
  std::unordered_set<const VarObj*> out_vars_;
  // Borrow the caller's maps; each visit restores its scoped bindings.
  ffi::Map<Var, IntSet>& hint_map_;
  ffi::Map<Var, IntSet>& relax_map_;
};

// Replace the set of conditions given by ps with cond_value (true or false)
class ConditionEliminator : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  explicit ConditionEliminator(const ExpressionSet& ps, bool cond_value = true)
      : ps_(ps), cond_value_(cond_value) {}

  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) final {
    if (auto expr = value.as<PrimExpr>()) {
      if (ps_.count(*expr)) {
        return ffi::Any(IntImm::Bool(cond_value_));
      }
    }
    return StmtExprMutator::Mutate(value, inplace_mode);
  }

 private:
  ExpressionSet ps_;
  bool cond_value_;
};

// Insert the partition branch at the innermost thread scope
class ThreadPartitionInserter : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) override {
    if (value.as<ExprObj>()) return ffi::Unchanged();
    return StmtExprMutator::Mutate(value, inplace_mode);
  }

  explicit ThreadPartitionInserter(const ExpressionSet& ps, PrimExpr cond)
      : ps_(ps), cond_(cond), innermost_thread_scope_(false) {}

  UnchangedOr<Stmt> Mutate_(const ForObj* op, InplaceMode inplace_mode) final {
    if (!tvm::s_tir::GetThreadBinding(op).has_value() || IsVirtualThread(op)) {
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }
    innermost_thread_scope_ = true;
    For loop = StmtExprMutator::Mutate_(op, inplace_mode)
                   .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                   .as_or_throw<For>();
    if (innermost_thread_scope_) {
      Stmt body = loop->body;
      Stmt simplified_body = ffi::make_object<ConditionEliminator>(ps_)
                                 ->Mutate(body, InplaceMode::kDisallow)
                                 .ValueOrUnchanged(body);
      loop.CopyOnWrite()->body = If(cond_, simplified_body, SeqStmt(body));
    }
    innermost_thread_scope_ = false;
    return loop;
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj* op, InplaceMode inplace_mode) final {
    if (op->op.same_as(tirx::launch_thread_op()) &&
        !op->args[0].as_or_throw<StringImm>()->value.starts_with("vthread")) {
      innermost_thread_scope_ = true;
      Stmt stmt =
          StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
      // add branch code inside the innermost thread scope
      if (innermost_thread_scope_) {
        Stmt simplified_body =
            ffi::make_object<ConditionEliminator>(ps_)->Mutate(op->body).ValueOrUnchanged(op->body);
        Stmt body = If(cond_, simplified_body, op->body);
        auto region = stmt.as_or_throw<RegionStmt>();
        region.CopyOnWrite()->body = body;
        stmt = region;
      }
      innermost_thread_scope_ = false;
      return stmt;
    } else {
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }
  }

 private:
  const ExpressionSet& ps_;
  PrimExpr cond_;
  bool innermost_thread_scope_;
};

// Try to partition range of iteration variables in order to remove (some)
// likely conditions
class LoopPartitioner : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) override {
    if (value.as<ExprObj>()) return ffi::Unchanged();
    return StmtExprMutator::Mutate(value, inplace_mode);
  }

  explicit LoopPartitioner(bool partition_const_loop, bool no_unroll_loop_with_extent_one,
                           bool unroll_loop_with_partition_hint_no_interval)
      : selector(ffi::make_object<CandidateSelector>(partition_const_loop)),
        no_unroll_loop_with_extent_one_(no_unroll_loop_with_extent_one),
        unroll_loop_with_partition_hint_no_interval_(unroll_loop_with_partition_hint_no_interval) {}

  Stmt VisitAndMutate(Stmt stmt) {
    selector->Visit(stmt);
    return Mutate(stmt, InplaceMode::kAllow).ValueOrUnchanged(std::move(stmt));
  }

  UnchangedOr<Stmt> Mutate_(const ForObj* op, InplaceMode inplace_mode) final {
    if (IsVirtualThread(op)) return StmtExprMutator::Mutate_(op, inplace_mode);
    bool thread_scope = tvm::s_tir::GetThreadBinding(op).has_value();
    {
      sym::ConstraintContext context;
      analyzer_->GetConstraintContext()->ForEach([&](const sym::Constraint& premise) {
        if (auto binding = premise.as<sym::BindingConstraintObj>();
            binding && binding->var.same_as(op->var))
          return;
        if (auto range = premise.as<sym::RangeConstraintObj>();
            range && range->var.same_as(op->var))
          return;
        context->Append(premise);
      });
      context->Append(sym::RangeConstraint(op->var, Range(op->begin, op->end)));
      analyzer_->SyncConstraintContext(context);
    }
    auto fs = ffi::GetRef<Stmt>(op);
    if (selector->candidates.count(fs)) {
      auto s = TryPartition(fs, op->var, op->begin, op->end - 1, op->body, thread_scope);
      if (s.has_value()) return s.value();
    }

    // Relax threadIdx ranges to avoid introducing divergent partition branches.
    bool relax = thread_scope &&
                 support::GPUThreadTag::Parse(tvm::s_tir::GetThreadBinding(op).value()).rank == 1;
    auto& ranges = relax ? relax_map_ : hint_map_;
    ranges.Set(op->var, IntSet::Interval(op->begin, op->end - 1));
    Stmt res = StmtExprMutator::Mutate_(op, InplaceMode::kDisallow)
                   .ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    ranges.erase(op->var);
    return res;
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj* op, InplaceMode inplace_mode) final {
    if (!op->op.same_as(tirx::launch_thread_op()) ||
        op->args[0].as_or_throw<StringImm>()->value.starts_with("vthread")) {
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }

    Var var = op->body_params[0].as_or_throw<PrimVar>();
    PrimExpr extent = op->args[1].as_or_throw<PrimExpr>();
    auto as = ffi::GetRef<Stmt>(op);
    if (selector->candidates.count(as)) {
      auto s = TryPartition(as, var, 0, extent - 1, op->body, true);
      if (s.has_value()) return s.value();
    }

    // normal path when loop parittion fails.
    support::GPUThreadTag scope =
        support::GPUThreadTag::Parse(op->args[0].as_or_throw<StringImm>()->value);
    Stmt res{ffi::UnsafeInit{}};
    if (scope.rank == 1) {
      // threadIdx should be put into relax map, in case of divergence.
      relax_map_.Set(var, IntSet::Interval(IntImm(var->ty.as_or_throw<PrimType>(), 0), extent - 1));
      res = StmtExprMutator::Mutate_(op, InplaceMode::kDisallow)
                .ValueOrUnchanged(ffi::GetRef<Stmt>(op));
      relax_map_.erase(var);
    } else {
      hint_map_.Set(var, IntSet::Interval(IntImm(var->ty.as_or_throw<PrimType>(), 0), extent - 1));
      res = StmtExprMutator::Mutate_(op, InplaceMode::kDisallow)
                .ValueOrUnchanged(ffi::GetRef<Stmt>(op));
      hint_map_.erase(var);
    }
    return res;
  }

 private:
  ffi::Optional<Stmt> TryPartition(const Stmt& stmt, Var var, PrimExpr min, PrimExpr max, Stmt body,
                                   bool partition_thread_scope);

  std::pair<IntSet, ExpressionSet> GetIntervalAndCondset(const Partition& partitions,
                                                         const IntSet& for_interval,
                                                         bool cond_value, bool has_partition_hint);

  inline Stmt MakeFor(const ffi::Object* op, PrimExpr extent, Stmt body);

  /* Candidate IRs that may be partitioned potentially */
  ffi::Map<Var, IntSet> hint_map_;
  ffi::Map<Var, IntSet> relax_map_;
  sym::Analyzer analyzer_;
  ffi::ObjectPtr<CandidateSelector> selector;
  bool no_unroll_loop_with_extent_one_;
  bool unroll_loop_with_partition_hint_no_interval_;
};

// Returns an interval (in the first component) in which all the conditions
// given in the second component provably have value given by cond_value
std::pair<IntSet, ExpressionSet> LoopPartitioner::GetIntervalAndCondset(const Partition& partitions,
                                                                        const IntSet& for_interval,
                                                                        bool cond_value,
                                                                        bool has_partition_hint) {
  ffi::Array<IntSet> sets;
  ExpressionSet cond_set;

  for (const auto& kv : partitions) {
    if (kv.first.second == cond_value) {
      const IntSet& interval = kv.second;
      IntSet intersection = sym::Intersect(interval, for_interval, analyzer_);

      if (!intersection.IsNothing()) {
        sets.push_back(kv.second);
        cond_set.insert(kv.first.first);
      }
    }
  }
  IntSet interval = sets.empty() ? IntSet::Nothing() : Intersect(sets);

  // Try to find the intersection of the cond_intervals until the intersection
  // is nothing when has_partition_hint is true.
  if (interval.IsNothing() && has_partition_hint) {
    IntSet cond_intersection = IntSet::Everything();
    cond_set.clear();

    for (const auto& kv : partitions) {
      if (kv.first.second == cond_value) {
        const IntSet& cond_interval = kv.second;
        IntSet intersection = sym::Intersect(cond_interval, for_interval, analyzer_);
        if (!intersection.IsNothing()) {
          cond_intersection = sym::Intersect(cond_intersection, cond_interval, analyzer_);
          // Return the latest interval and cond_set if the cond_intersection is nothing.
          if (!cond_intersection.IsNothing()) {
            cond_set.insert(kv.first.first);
            interval = IntSet::Interval(analyzer_->Simplify(cond_intersection.min()),
                                        analyzer_->Simplify(cond_intersection.max()));
          } else {
            break;
          }
        }
      }
    }
  }

  return std::make_pair(interval, cond_set);
}

/*
 * Tries to recursively partition the range of the variable (given by var) of
 * the for loop (given by node and stmt) into a
 * number of disjoint ranges such that in some ranges one or more predicates
 * in the loopnest are provably true or false in each range. For example, given the
 * following loop to partition:
 * for (i = 0; i < 4; i++)
 *    for (j = 0; j < 10; j++)
 *        if (likely(i*10 + j < 36))
 *            A[10*i+j] = B[10*i+j]
 *
 * We first partition range of i, i.e., [0,3] into subranges [0,2] and [3,3] because the
 * likely condition is always true for the first subrange but not always true for the
 * second subrange. Therefore, we'll have
 * for (i = 0; i < 3; i++)
 *    for (j = 0; j < 10; j++)
 *        if (likely(1))
 *           A[10*i+j] = B[10*i+j]
 * for (i = 0; i < 1; i++)
 *    for (j = 0; j < 10; j++)
 *        if (likely((i+3)*10 + j < 36))
 *            A[10*(i+3)+j] = B[10*(i+3)+j]
 * Which is simplified as:
 * for (i = 0; i < 3; i++)
 *    for (j = 0; j < 10; j++)
 *        A[10*i+j] = B[10*i+j]
 * for (j = 0; j < 10; j++) // loopnest 1
 *    if (likely(j < 6))
 *            A[30+j] = B[30+j]
 * Now, we recursively partition j in loopnest 1 into subranges [0,5] and [6,9] where the
 * condition is true for the first subrange and now always true for the second subrange.
 * for (j = 0; j < 6; j++)
 *    if (likely(1))
 *         A[30+j] = B[30+j]
 * for (j = 0; j < 4; j++) // loop 2
 *    if (likely(j < 0))
 *        A[36+j] = B[36+j]
 * Finally we recursively partition loop 2 above into subrange [0,3] where the
 * condition is false and empty interval where the condition is not false,
 * therefore we generate
 * for (j = 0; j < 4; j++)
 *    if (likely(0))
 *        A[36+j] = B[36+j]
 * which will eventually be simplified to empty code. And because only one loop was generated
 * from loop 2 we stop recursing.
 */
ffi::Optional<Stmt> LoopPartitioner::TryPartition(const Stmt& stmt, Var var, PrimExpr min,
                                                  PrimExpr max, Stmt body,
                                                  bool partition_thread_scope) {
  using namespace sym;
  // include hint of var.
  hint_map_.Set(var, IntSet::Interval(min, max));

  const auto* loop = stmt.as<ForObj>();
  bool has_partition_hint_ = loop && HasPartitionHint(loop);
  auto finder = ffi::make_object<PartitionFinder>(var, hint_map_, relax_map_, has_partition_hint_);
  finder->Visit(body);

  hint_map_.erase(var);
  if (finder->partitions.empty()) return std::nullopt;

  IntSet for_interval = IntSet::Interval(min, max);

  auto [middle_interval, cond_set, opt_cond_value] =
      [&]() -> std::tuple<ffi::Optional<IntSet>, ExpressionSet, std::optional<bool>> {
    {
      // find an interval in which all conditions on var are true
      auto [middle_interval, cond_set] =
          GetIntervalAndCondset(finder->partitions, for_interval, true, has_partition_hint_);
      if (!middle_interval.IsNothing()) {
        return {middle_interval, cond_set, true};
      }
    }

    {
      // if such interval doesn't exist, find an interval in which all
      // conditions on var are false
      auto [middle_interval, cond_set] =
          GetIntervalAndCondset(finder->partitions, for_interval, false, has_partition_hint_);

      if (!middle_interval.IsNothing()) {
        return {middle_interval, cond_set, false};
      }
    }

    bool all_singlepoints_outside = true;

    // Check all partitions to see if they are single points and outside `for_interval`
    for (const auto& partition : finder->partitions) {
      const auto& intset = partition.second;
      // Only proceed if the interval set is a single point
      if (intset.IsSinglePoint()) {
        auto single_point = intset.PointValue();
        // Check if the single point is outside the `for_interval`
        bool is_inside = analyzer_->CanProve(single_point >= for_interval.min()) &&
                         analyzer_->CanProve(single_point <= for_interval.max());
        if (is_inside) {
          // If any single point is inside, this is an error condition
          LOG(ERROR) << "unexpected case happened.";
          all_singlepoints_outside = false;
          break;
        }
      } else {
        // If there is any intset that is not a single point, follow default logic
        // For now, we set all_singlepoints_outside to false to indicate default logic was used
        all_singlepoints_outside = false;
        break;
      }
    }

    if (all_singlepoints_outside) {
      // If all single points are outside `for_interval`, return a nothing interval and false
      return {IntSet::Nothing(), ExpressionSet(), false};
    }

    // we couldn't find an interval in which the conditions are
    // provably true or false.  Therefore, we can't partition the loop
    // based on those conds
    return {{}, {}, std::nullopt};
  }();

  if (opt_cond_value == false && middle_interval.value().IsNothing()) {
    return std::nullopt;
  }

  if (!opt_cond_value.has_value()) {
    if (!partition_thread_scope && has_partition_hint_ &&
        unroll_loop_with_partition_hint_no_interval_ && analyzer_->CanProve(max - min > 0)) {
      auto new_body = VisitAndMutate(body);
      auto unrolled = ffi::make_object<ForObj>(*loop);
      unrolled->kind = ForKind::kUnrolled;
      unrolled->body = std::move(new_body);
      return For(unrolled);
    }
    return std::nullopt;
  }
  bool cond_value = opt_cond_value.value();

  // middle_interval is the subrange of the loop variable range for which a
  // set of conditions are true (or false resp.)
  // The part of the loop variable range that is before (after resp.) that
  // subrange is prefixed with pre- (post- resp.)

  // Calculating pre-subrange and generating code for it.
  // pre-subrange = [min, body_begin)
  PrimExpr body_begin = min;
  ffi::Optional<Stmt> pre_stmt;
  bool pre_stmt_recurse = true;
  if (middle_interval.value().HasLowerBound()) {
    body_begin = analyzer_->Simplify(middle_interval.value().min());
    if (!analyzer_->CanProve(body_begin == min)) {
      PrimExpr extent = analyzer_->Simplify(body_begin - min);
      if (!analyzer_->CanProve(extent > 0)) {
        body_begin = tvm::max(body_begin, min);
        // stop recursing on this interval if we can't prove it has non-negative length
        pre_stmt_recurse = false;
      }
      if (!analyzer_->CanProve(extent <= 0)) {
        if (!partition_thread_scope) {
          auto f_substitute =
              [&var, &min](const Var& candidate) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
            if (candidate.same_as(var)) {
              return ffi::Any(var.as_or_throw<PrimExpr>() + min);
            }
            return ffi::Unchanged();
          };
          Stmt pre_body = ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(body, f_substitute)
                              .as_or_throw<Stmt>();
          pre_stmt = MakeFor(stmt.get(), body_begin - min, pre_body);
        }
      }
    }
  } else {
    body_begin = min;
  }

  // Calculating post-subrange and generating code for it.
  // post-subrange = [post_doubt_begin, max+1)
  PrimExpr post_doubt_begin = max + 1;
  ffi::Optional<Stmt> post_stmt;
  bool post_stmt_recurse = true;
  if (middle_interval.value().HasUpperBound()) {
    post_doubt_begin = analyzer_->Simplify(middle_interval.value().max() + 1);
    if (!analyzer_->CanProve(middle_interval.value().max() == max)) {
      // require the extent to be non-negative
      PrimExpr extent = analyzer_->Simplify(max - post_doubt_begin + 1);
      if (!analyzer_->CanProve(extent > 0)) {
        post_doubt_begin = tvm::min(post_doubt_begin, max + 1);
        // stop recursing on this interval if we can't prove it has non-negative length
        post_stmt_recurse = false;
      }
      if (!analyzer_->CanProve(extent <= 0)) {
        if (!partition_thread_scope) {
          auto f_substitute =
              [&var, &post_doubt_begin](
                  const Var& candidate) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
            if (candidate.same_as(var)) {
              return ffi::Any(var.as_or_throw<PrimExpr>() + post_doubt_begin);
            }
            return ffi::Unchanged();
          };
          Stmt post_body = ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(body, f_substitute)
                               .as_or_throw<Stmt>();
          post_stmt = MakeFor(stmt.get(), extent, post_body);
        }
      }
    }
  } else {
    post_doubt_begin = max + 1;
  }

  Stmt s{ffi::UnsafeInit{}};

  // Generating code for middle subrange
  if (!partition_thread_scope) {
    ffi::Optional<Stmt> mid_stmt;
    if (!analyzer_->CanProve(body_begin >= post_doubt_begin)) {
      // [body_begin, post_doubt_begin)
      Stmt simplified_body = ffi::make_object<ConditionEliminator>(cond_set, cond_value)
                                 ->Mutate(body)
                                 .ValueOrUnchanged(body);
      auto f_substitute =
          [&var, &body_begin](const Var& candidate) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
        if (candidate.same_as(var)) {
          return ffi::Any(var.as_or_throw<PrimExpr>() + body_begin);
        }
        return ffi::Unchanged();
      };
      Stmt new_body = ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(simplified_body, f_substitute)
                          .as_or_throw<Stmt>();
      mid_stmt = MakeFor(stmt.get(), post_doubt_begin - body_begin, new_body);
      // Recurse until partitions is empty
      mid_stmt = VisitAndMutate(mid_stmt.value());
      // Recurse for each non-empty subrange only if there are at least
      // two non-empty subranges
      if (pre_stmt.has_value() || post_stmt.has_value()) {
        if (pre_stmt.has_value() && pre_stmt_recurse) {
          pre_stmt = VisitAndMutate(pre_stmt.value());
        }
        if (post_stmt.has_value() && post_stmt_recurse) {
          post_stmt = VisitAndMutate(post_stmt.value());
        }
      }
    }
    ffi::Array<Stmt> partitions;
    if (pre_stmt.has_value()) partitions.push_back(pre_stmt.value());
    if (mid_stmt.has_value()) partitions.push_back(mid_stmt.value());
    if (post_stmt.has_value()) partitions.push_back(post_stmt.value());
    s = SeqStmt(partitions);
  } else {
    PrimExpr cond = IntImm::Bool(true);
    if (!analyzer_->CanProve(body_begin == min)) {
      cond = cond && (var.as_or_throw<PrimExpr>() >= body_begin);
    }
    if (!analyzer_->CanProve(post_doubt_begin == (max + 1)))
      cond = cond && (var.as_or_throw<PrimExpr>() < post_doubt_begin);
    s = ffi::make_object<ThreadPartitionInserter>(cond_set, cond)
            ->Mutate(stmt)
            .ValueOrUnchanged(stmt);
  }
  s = s_tir::ConvertSSA(s);
  return s;
}

inline Stmt LoopPartitioner::MakeFor(const ffi::Object* node, PrimExpr extent, Stmt body) {
  const ForObj* for_node = static_cast<const ForObj*>(node);
  TVM_FFI_ICHECK(for_node);

  if (analyzer_->CanProve(extent == IntImm::Int32(1)) && !no_unroll_loop_with_extent_one_ &&
      for_node->annotations.size() ==
          for_node->annotations.count(tvm::s_tir::attr::kLoopPartitionHint.name)) {
    // If the loop extent is 1, do not create the loop anymore
    auto f_substitute =
        [loop_var = for_node->var](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
      if (var.same_as(loop_var)) return ffi::Any(IntImm::Int32(0));
      return ffi::Unchanged();
    };
    return ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(body, f_substitute).as_or_throw<Stmt>();
  } else {
    TVM_FFI_ICHECK(!tvm::s_tir::GetThreadBinding(for_node).has_value());
    auto new_loop = ffi::make_object<ForObj>(*for_node);
    new_loop->begin = IntImm(for_node->begin.ty(), 0);
    new_loop->end = extent;
    new_loop->body = body;
    return For(new_loop);
  }
}

class RemoveLikelyTagsAndHints : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  UnchangedOr<Expr> Mutate_(const CallObj* op, InplaceMode inplace_mode) final {
    if (op->op.same_as(prim::likely_op())) {
      TVM_FFI_ICHECK_EQ(op->args.size(), 1);
      return StmtExprMutator::Mutate(op->args[0]).ValueOrUnchanged(op->args[0]).as_or_throw<Expr>();
    } else if (op->op.same_as(tirx::ignore_loop_partition_op())) {
      TVM_FFI_ICHECK_EQ(op->args.size(), 1);
      return StmtExprMutator::Mutate(op->args[0]).ValueOrUnchanged(op->args[0]).as_or_throw<Expr>();
    } else {
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }
  }

  UnchangedOr<Stmt> Mutate_(const ForObj* op, InplaceMode inplace_mode) final {
    For loop = StmtExprMutator::Mutate_(op, inplace_mode)
                   .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                   .as_or_throw<For>();
    if (loop->annotations.count(tvm::s_tir::attr::kLoopPartitionHint.name)) {
      loop.CopyOnWrite()->annotations.erase(tvm::s_tir::attr::kLoopPartitionHint.name);
    }
    return loop;
  }
};

Stmt LoopPartition(Stmt stmt, bool partition_const_loop, bool no_unroll_loop_with_extent_one,
                   bool unroll_loop_with_partition_hint_no_interval) {
  stmt = ffi::make_object<NormalizePartitionHints>()
             ->Mutate(stmt, InplaceMode::kAllow)
             .ValueOrUnchanged(std::move(stmt));
  stmt = ffi::make_object<LoopPartitioner>(partition_const_loop, no_unroll_loop_with_extent_one,
                                           unroll_loop_with_partition_hint_no_interval)
             ->VisitAndMutate(std::move(stmt));
  stmt = ffi::make_object<RemoveLikelyTagsAndHints>()
             ->Mutate(stmt, InplaceMode::kAllow)
             .ValueOrUnchanged(std::move(stmt));
  return stmt;
}

namespace transform {

Pass LoopPartition() {
  auto pass_func = [=](Function f, IRModule m, PassContext ctx) {
    if (!f->body.has_value()) return f;
    auto* n = f.CopyOnWrite();
    auto cfg = ctx->GetConfigOrDefault<LoopPartitionConfig>("s_tir.LoopPartition");
    n->body = s_tir::LoopPartition(std::move(n->body).value(), cfg->partition_const_loop,
                                   cfg->no_unroll_loop_with_extent_one,
                                   cfg->unroll_loop_with_partition_hint_no_interval);
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "s_tir.LoopPartition");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.LoopPartition", LoopPartition);
}

}  // namespace transform

}  // namespace s_tir
}  // namespace tvm
