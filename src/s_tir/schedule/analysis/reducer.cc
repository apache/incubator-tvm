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
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/te/operation.h>
#include <tvm/tirx/op/memory.h>

#include "../utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

/******** Pattern Matcher ********/

/*!
 * \brief PrimExpr pattern matcher.
 *
 * It is different from the pattern matcher in tvm/ir/prim/pattern_match.h, which is dedicated
 * for compile-time constant patterns. This pattern matcher can work on dynamic user-specific
 * patterns.
 *
 * The code below shows how to use the pattern matcher.
 *
 * \code
 *
 * Var x("x"), y("y");
 * // use PrimExpr to declare patterns, x, y are holes that can be filled with
 * PatternMatcher pattern_matcher(x + y);
 * // expr = C[i, j] + A[i, k] * B[k, j], which is the expr we want to match
 * pattern_matcher.Match(expr);
 *
 * if (pattern_matcher.Success()) {
 *   pattern_matcher.Eval(x) // C[i, j]
 *   pattern_matcher.Eval(y) // A[i, k] * B[k, j]
 * }
 *
 * \endcode
 */
class PatternMatcher : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;

  explicit PatternMatcher(ffi::Array<PrimExpr> pattern) : pattern_(std::move(pattern)) {}

  ffi::Optional<VisitInterrupt> Visit_(const VarObj* op) final {
    auto it = filled_map_.find(op);
    if (it == filled_map_.end()) {
      filled_map_.insert_or_assign(op, expr_to_match_);
    } else {
      if (it->second.same_as(expr_to_match_) ||
          ffi::StructuralEqual()(it->second, expr_to_match_)) {
        return std::nullopt;
      }
      match_success_ = false;
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const LetObj* op) final {
    const auto* ptr = expr_to_match_.as<LetObj>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      Expr tmp = expr_to_match_;
      expr_to_match_ = ptr->var;
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->var));
      expr_to_match_ = ptr->value;
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->value));
      expr_to_match_ = ptr->body;
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->body));
      std::swap(expr_to_match_, tmp);
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const CallObj* op) final {
    const auto* ptr = expr_to_match_.as<CallObj>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      if (!op->op.same_as(ptr->op) || op->args.size() != ptr->args.size()) {
        match_success_ = false;
      } else {
        Expr tmp = expr_to_match_;
        for (size_t i = 0; i < op->args.size(); ++i) {
          expr_to_match_ = ptr->args[i];
          TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->args[i]));
        }
        std::swap(expr_to_match_, tmp);
      }
    }
    return std::nullopt;
  }

#define TVM_DECLARE_PATTERN_MATCHER_BIN_OP(OpName)         \
  ffi::Optional<VisitInterrupt> Visit_(const OpName* op) { \
    const auto* ptr = expr_to_match_.as<OpName>();         \
    if (ptr == nullptr) {                                  \
      match_success_ = false;                              \
    } else {                                               \
      Expr current = expr_to_match_;                       \
      expr_to_match_ = ptr->a;                             \
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->a));    \
      expr_to_match_ = ptr->b;                             \
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->b));    \
      std::swap(expr_to_match_, current);                  \
    }                                                      \
    return std::nullopt;                                   \
  }

  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(AddObj);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(SubObj);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(MulObj);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(DivObj);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(ModObj);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(FloorDivObj);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(FloorModObj);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(MinObj);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(MaxObj);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(EQObj);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(NEObj);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(LTObj);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(LEObj);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(GTObj);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(GEObj);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(AndObj);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(OrObj);

  ffi::Optional<VisitInterrupt> Visit_(const CastObj* op) final {
    const auto* ptr = expr_to_match_.as<CastObj>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      if (op->ty.as_or_throw<PrimType>() != ptr->ty.as_or_throw<PrimType>()) {
        match_success_ = false;
      } else {
        Expr tmp = expr_to_match_;
        expr_to_match_ = ptr->value;
        TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->value));
        std::swap(expr_to_match_, tmp);
      }
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const NotObj* op) final {
    const auto* ptr = expr_to_match_.as<NotObj>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      Expr tmp = expr_to_match_;
      expr_to_match_ = ptr->a;
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->a));
      std::swap(expr_to_match_, tmp);
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const SelectObj* op) final {
    const auto* ptr = expr_to_match_.as<SelectObj>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      Expr tmp = expr_to_match_;
      expr_to_match_ = ptr->condition;
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->condition));
      expr_to_match_ = ptr->true_value;
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->true_value));
      expr_to_match_ = ptr->false_value;
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->false_value));
      std::swap(expr_to_match_, tmp);
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const RampObj* op) final {
    const auto* ptr = expr_to_match_.as<RampObj>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      Expr tmp = expr_to_match_;
      expr_to_match_ = ptr->base;
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->base));
      expr_to_match_ = ptr->stride;
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->stride));
      expr_to_match_ = ptr->lanes;
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->lanes));
      std::swap(expr_to_match_, tmp);
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const BroadcastObj* op) final {
    const auto* ptr = expr_to_match_.as<BroadcastObj>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      Expr tmp = expr_to_match_;
      expr_to_match_ = ptr->value;
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->value));
      expr_to_match_ = ptr->lanes;
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->lanes));
      std::swap(expr_to_match_, tmp);
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const ShuffleObj* op) final {
    const auto* ptr = expr_to_match_.as<ShuffleObj>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      if (op->vectors.size() != ptr->vectors.size() || op->indices.size() != ptr->indices.size()) {
        match_success_ = false;
      } else {
        Expr tmp = expr_to_match_;
        for (size_t i = 0; i < op->indices.size(); ++i) {
          expr_to_match_ = ptr->indices[i];
          TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->indices[i]));
        }
        for (size_t i = 0; i < op->vectors.size(); ++i) {
          expr_to_match_ = ptr->vectors[i];
          TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->vectors[i]));
        }
        std::swap(expr_to_match_, tmp);
      }
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const IntImmObj* op) final {
    const auto* ptr = expr_to_match_.as<IntImmObj>();
    match_success_ = ptr != nullptr && op->value == ptr->value;
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const FloatImmObj* op) final {
    const auto* ptr = expr_to_match_.as<FloatImmObj>();
    match_success_ = ptr != nullptr && op->value == ptr->value;
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const StringImmObj* op) final {
    const auto* ptr = expr_to_match_.as<StringImmObj>();
    match_success_ = ptr != nullptr && op->value == ptr->value;
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadObj* op) final {
    const auto* ptr = expr_to_match_.as<TensorLoadObj>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      if (!op->source.as_or_throw<tvm::tirx::TensorVar>().same_as(
              ptr->source.as_or_throw<tvm::tirx::TensorVar>()) ||
          op->indices.size() != ptr->indices.size()) {
        match_success_ = false;
      } else {
        Expr tmp = expr_to_match_;
        for (size_t i = 0; i < op->indices.size(); ++i) {
          expr_to_match_ = ptr->indices[i];
          TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->indices[i]));
        }
        std::swap(expr_to_match_, tmp);
      }
    }
    return std::nullopt;
  }

  void Match(const ffi::Array<PrimExpr>& exprs_to_match) {
    this->match_success_ = true;
    this->filled_map_.clear();

    TVM_FFI_ICHECK_EQ(pattern_.size(), exprs_to_match.size());
    int n_tensors = pattern_.size();
    for (int i = 0; i < n_tensors; ++i) {
      this->expr_to_match_ = exprs_to_match[i];
      this->Visit(pattern_[i]);
    }
  }

  PrimExpr Eval(const Var& var) {
    auto it = filled_map_.find(var.operator->());
    TVM_FFI_ICHECK(it != filled_map_.end()) << "Unknown pattern variable";
    TVM_FFI_ICHECK(match_success_) << "Match failed";
    return it->second.as_or_throw<PrimExpr>();
  }

  bool Success() const { return match_success_; }

 private:
  bool match_success_{true};
  ffi::Array<PrimExpr> pattern_;
  Expr expr_to_match_{ffi::UnsafeInit{}};
  std::unordered_map<const VarObj*, Expr> filled_map_;
};

/******** Reduction SBlock Related ********/

static const char* kRFactorCrossThreadReductionApplicableBlockDef =
    R"(Definition of a reduction block that is applicable by RFactor and Cross-Thread Reduction:
1) The block init should be a single TensorStore or a SeqStmt of TensorStores
2) The tensors initialized in the block init should be all different
3) The number of consecutive Binds in the block body (if any) should equal the number of TensorStores in the block init
4) The variables of the Binds in the block body should be all different
5) The statement after the innermost Bind should be a single TensorStore or a SeqStmt of TensorStores
6) The number of TensorStores under the block body should equal the number of TensorStores in the block init, and thereby equal the number of Binds above
7) The variables bound by the Binds in the block body must all directly serve as values of the TensorStores inside, and the stored values of the TensorStores can only be those variables
8) The variables stored by the TensorStores in the block body should be all different
9) The tensors written by the TensorStores in the block body should be all different
10) The tensors initialized in the block init and written in the block body should match
11) The tensors written by the block should have same shape
12) The indices of all TensorStores in the reduction block should be the same)";

void ErrorRFactorCrossThreadReductionNotApplicable(const ffi::Optional<ScheduleState>& self,
                                                   SBlock block, int violated_cond) {
  class RFactorNotApplicableError : public ScheduleErrorContextObj {
   public:
    explicit RFactorNotApplicableError(IRModule mod, SBlock block, int violated_cond)
        : mod_(std::move(mod)), block_(std::move(block)), violated_cond_(violated_cond) {}

    ffi::String FastErrorString() const final {
      return "ScheduleError: RFactor cannot be applied to the block since the block does not meet "
             "the requirements";
    }

    ffi::String DetailRenderTemplate() const final {
      std::ostringstream os;
      os << "RFactor cannot be applied to block {0}, because the block violates condition #"
         << violated_cond_ << ".\n"
         << kRFactorCrossThreadReductionApplicableBlockDef;
      return os.str();
    }

    IRModule mod() const final { return mod_; }
    ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_}; }

    IRModule mod_;
    SBlock block_;
    int violated_cond_;
  };

  if (self.has_value()) {
    throw MakeScheduleError<RFactorNotApplicableError>(self.value()->mod, std::move(block),
                                                       violated_cond);
  } else {
    TVM_FFI_THROW(ValueError) << "Cross-thread reduction cannot be applied to the block "
                              << block->name_hint << " because the block violates the condition #"
                              << violated_cond << ".\n"
                              << kRFactorCrossThreadReductionApplicableBlockDef;
  }
}

/*!
 * \brief Extract the TensorStores, which serve as the reduction updates, from the given Bind nodes
 * and the TensorStores inside. And meanwhile set the tensor order of the reduction
 * \param self The schedule state, used for error reporting
 * \param block The reduction block, used for error reporting
 * \param let The Bind nodes from which the reduction updates are extracted
 * \param n_tensors The number of tensors participating in the reduction
 * \param updates The extracted reduction updates
 * \param tensor2index A mapping from reduction tensors to their indices of the reduction order
 * \throw ScheduleError If rfactor or cross-thread reduction cannot be applied to the block
 */
void ExtractReductionUpdates(const ffi::Optional<ScheduleState>& self, SBlock block,
                             const ffi::Array<Stmt>& stmts, int n_tensors,
                             ffi::Array<TensorStore>* updates,
                             std::unordered_map<const VarObj*, int>* tensor2index) {
  std::unordered_map<const VarObj*, int> var2index;
  ffi::Array<PrimExpr> let_values;
  let_values.reserve(n_tensors);
  updates->resize(n_tensors);

  // Step 1.
  // - Extract the Bind values from the sequence.
  // - Construct the mapping from bind variables to the index.
  // The first n_tensors stmts should be Bind nodes.
  for (int i = 0; i < n_tensors; ++i) {
    if (i >= static_cast<int>(stmts.size())) {
      ErrorRFactorCrossThreadReductionNotApplicable(self, std::move(block), /*violated_cond=*/3);
    }
    const auto* bind = stmts[i].as<BindObj>();
    if (bind == nullptr || !bind->value.as<PrimExpr>()) {
      ErrorRFactorCrossThreadReductionNotApplicable(self, std::move(block), /*violated_cond=*/3);
    }
    let_values.push_back(bind->value.as_or_throw<PrimExpr>());
    auto insert_result = var2index.insert(std::make_pair(bind->var.get(), i));
    if (!insert_result.second) {
      ErrorRFactorCrossThreadReductionNotApplicable(self, std::move(block), /*violated_cond=*/4);
    }
  }

  // There should be no more Bind after the first n_tensors.
  if (n_tensors < static_cast<int>(stmts.size()) && stmts[n_tensors]->IsInstance<BindObj>()) {
    ErrorRFactorCrossThreadReductionNotApplicable(self, std::move(block), /*violated_cond=*/3);
  }

  // The remaining stmts after the Bind nodes should be TensorStores.
  // Collect them into a sequence.
  ffi::Array<Stmt> seq;
  for (int i = n_tensors; i < static_cast<int>(stmts.size()); ++i) {
    seq.push_back(stmts[i]);
  }
  if (static_cast<int>(seq.size()) != n_tensors) {
    ErrorRFactorCrossThreadReductionNotApplicable(self, std::move(block), /*violated_cond=*/6);
  }

  // Step 2.
  // - Create TensorStores according to the variables being stored.
  // - Construct the mapping from reduction tensors to the index.
  for (const Stmt& stmt : seq) {
    const auto* tensor_store = stmt.as<TensorStoreObj>();
    if (tensor_store == nullptr) {
      ErrorRFactorCrossThreadReductionNotApplicable(self, std::move(block), /*violated_cond=*/5);
    }
    auto var_ref = tensor_store->value.as<PrimVar>();
    if (!var_ref.has_value()) {
      ErrorRFactorCrossThreadReductionNotApplicable(self, std::move(block), /*violated_cond=*/7);
    }
    const VarObj* var = var_ref.value().get();
    auto it = var2index.find(var);
    if (it == var2index.end()) {
      ErrorRFactorCrossThreadReductionNotApplicable(self, std::move(block), /*violated_cond=*/7);
    }
    int idx = it->second;
    if ((*updates)[idx].defined()) {
      ErrorRFactorCrossThreadReductionNotApplicable(self, std::move(block), /*violated_cond=*/8);
    }
    updates->Set(idx, TensorStore(tensor_store->dest, tensor_store->indices, let_values[idx]));
    auto insert_result = tensor2index->insert(
        std::make_pair(tensor_store->dest.as_or_throw<TensorVar>().get(), idx));
    if (!insert_result.second) {
      ErrorRFactorCrossThreadReductionNotApplicable(self, std::move(block), /*violated_cond=*/9);
    }
  }
  for (int i = 0; i < n_tensors; ++i) {
    TVM_FFI_ICHECK((*updates)[i].defined());
  }
}

std::pair<ffi::Array<PrimExpr>, ffi::Array<TensorStore>> GetInitValuesAndUpdatesFromReductionBlock(
    const ffi::Optional<ScheduleState>& self, SBlock block) {
  ffi::Array<TensorStore> inits;
  ffi::Array<TensorStore> updates;

  // Step 1. Extract the TensorStores serving as block inits.
  if (block->init.has_value() && !block->init.value()->seq.empty()) {
    const SeqStmt& seq_init = block->init.value();
    std::unordered_set<const VarObj*> init_tensors;
    for (const Stmt& stmt : seq_init->seq) {
      auto init = stmt.as<TensorStore>();
      if (!init) {
        ErrorRFactorCrossThreadReductionNotApplicable(self, std::move(block), /*violated_cond=*/1);
      }
      auto insert_result = init_tensors.insert(init.value()->dest.as_or_throw<TensorVar>().get());
      if (!insert_result.second) {
        ErrorRFactorCrossThreadReductionNotApplicable(self, std::move(block), /*violated_cond=*/2);
      }
      inits.push_back(init.value());
    }
  } else {
    ErrorRFactorCrossThreadReductionNotApplicable(self, std::move(block), /*violated_cond=*/1);
  }

  // Step 2. Extract the block updates, in the form of TensorStores.
  int n_tensors = inits.size();
  std::unordered_map<const VarObj*, int> tensor2index;
  if (const auto* update =
          block->body->size() == 1 ? block->body->seq[0].as<TensorStoreObj>() : nullptr) {
    updates.push_back(ffi::GetRef<TensorStore>(update));
    tensor2index[update->dest.as_or_throw<TensorVar>().get()] = 0;
  } else {
    ExtractReductionUpdates(self, block, block->body->seq, n_tensors, &updates, &tensor2index);
  }
  TVM_FFI_ICHECK_EQ(updates.size(), n_tensors);

  // Step 3. Set the init values according to the tensor order in `updates`, with the help of the
  // mapping `tensor2index`.
  ffi::Array<PrimExpr> init_values;
  init_values.resize(n_tensors);

  // - Check all tensors have the same shape
  // - Check all indices of the TensorStores are the same
  // - Check tensors written in the block init and the block body can match
  // - Check tensors do not duplicate
  const ffi::Array<PrimExpr>& expected_shape = updates[0]->dest.as_or_throw<TensorVar>().shape();
  const ffi::Array<PrimExpr>& expected_indices = updates[0]->indices;
  TVM_FFI_ICHECK_EQ(expected_shape.size(), expected_indices.size());
  int n_dim = expected_indices.size();
  sym::Analyzer ana;
  for (int i = 0; i < n_tensors; ++i) {
    if (static_cast<int>(updates[i]->dest.as_or_throw<TensorVar>().shape().size()) != n_dim) {
      ErrorRFactorCrossThreadReductionNotApplicable(self, std::move(block), /*violated_cond=*/11);
    }
    if (static_cast<int>(inits[i]->indices.size()) != n_dim ||
        static_cast<int>(updates[i]->indices.size()) != n_dim) {
      ErrorRFactorCrossThreadReductionNotApplicable(self, std::move(block), /*violated_cond=*/12);
    }
    for (int d = 0; d < n_dim; ++d) {
      if (!ana->CanProveEqual(updates[i]->dest.as_or_throw<TensorVar>().shape()[d],
                              expected_shape[d])) {
        ErrorRFactorCrossThreadReductionNotApplicable(self, std::move(block), /*violated_cond=*/11);
      }
      if (!ana->CanProveEqual(inits[i]->indices[d], expected_indices[d]) ||
          !ana->CanProveEqual(updates[i]->indices[d], expected_indices[d])) {
        ErrorRFactorCrossThreadReductionNotApplicable(self, std::move(block), /*violated_cond=*/12);
      }
    }

    auto it = tensor2index.find(inits[i]->dest.as_or_throw<TensorVar>().get());
    if (it == tensor2index.end()) {
      ErrorRFactorCrossThreadReductionNotApplicable(self, std::move(block), /*violated_cond=*/10);
    }
    int idx = it->second;
    TVM_FFI_ICHECK(updates[idx]->dest.as_or_throw<TensorVar>().same_as(
        inits[i]->dest.as_or_throw<TensorVar>()));
    TVM_FFI_ICHECK(!init_values[idx].defined());
    init_values.Set(idx, inits[i]->value);
  }
  for (int i = 0; i < n_tensors; ++i) {
    TVM_FFI_ICHECK(init_values[i].defined());
  }

  return std::make_pair(init_values, updates);
}

bool ContainsOnlyDataParAndReductionBlockIter(const ffi::Array<IterVar>& iters) {
  for (const IterVar& iter_var : iters) {
    if (iter_var->iter_type != kDataPar && iter_var->iter_type != kCommReduce) {
      return false;
    }
  }
  return true;
}

bool ReductionIterNotIndexOutputTensor(const SBlock& block) {
  // Step 1. Collect the reduction block iters.
  std::unordered_set<const VarObj*> reduction_block_iters;
  reduction_block_iters.reserve(block->iter_vars.size());
  for (const IterVar& iter_var : block->iter_vars) {
    if (iter_var->iter_type == kCommReduce) {
      reduction_block_iters.insert(iter_var->var.get());
    }
  }
  // Step 2. Check if the reduction block iters are used to index the output tensor.
  std::unordered_set<const VarObj*> tensor_written;
  tensor_written.reserve(block->writes.size());
  for (const TensorRegion& write_region : block->writes) {
    tensor_written.insert(write_region->source.as_or_throw<tvm::tirx::TensorVar>().get());
  }

  std::unordered_set<const VarObj*> tensor_allocated;
  tensor_allocated.reserve(block->alloc_tensors.size());
  for (const TensorVar& tensor : block->alloc_tensors) {
    tensor_allocated.insert(tensor.get());
  }

  auto walkfn = [&](const Var& var) -> ffi::Expected<ffi::WalkResult> {
    return reduction_block_iters.count(var.get())
               ? ffi::WalkResult::Interrupt(ffi::VisitInterrupt(var))
               : ffi::WalkResult::Advance();
  };
  auto f_uses_reduction_block_var = [&](const PrimExpr& expr) -> bool {
    return ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(expr, walkfn).has_value();
  };

  std::unordered_map<const VarObj*, const VarObj*> match_tensor_sources;
  for (const MatchTensorRegion& region : block->match_tensors) {
    match_tensor_sources[region->tensor.get()] =
        region->source->source.as_or_throw<tvm::tirx::TensorVar>().get();
  }
  auto visit_block = [&](const SBlock& nested_block) -> ffi::Expected<ffi::WalkResult> {
    for (const MatchTensorRegion& region : nested_block->match_tensors) {
      match_tensor_sources[region->tensor.get()] =
          region->source->source.as_or_throw<tvm::tirx::TensorVar>().get();
    }
    return ffi::WalkResult::Advance();
  };
  auto visit_alloc = [&](const tvm::Bind& alloc) -> ffi::Expected<ffi::WalkResult> {
    // Inline AllocTensor statements (e.g. `T.local_scalar(...)` expansions)
    // declare tensor-local scratch storage inside the block body; treat them
    // the same as block->alloc_tensors entries for the "write-without-signature"
    // check below.
    if (const auto* call = alloc->value.as<CallObj>();
        call && call->op.same_as(tirx::alloc_tensor_op())) {
      tensor_allocated.insert(alloc->var.get());
    }
    return ffi::WalkResult::Advance();
  };
  auto visit_store = [&](const TensorStore& store) -> ffi::Expected<ffi::WalkResult> {
    bool write_is_covered_by_match_tensor =
        match_tensor_sources.count(store->dest.as_or_throw<TensorVar>().get()) &&
        tensor_written.count(
            match_tensor_sources.find(store->dest.as_or_throw<TensorVar>().get())->second);
    TVM_FFI_CHECK(tensor_written.count(store->dest.as_or_throw<TensorVar>().get()) ||
                      write_is_covered_by_match_tensor ||
                      tensor_allocated.count(store->dest.as_or_throw<TensorVar>().get()),
                  ValueError)
        << "The tensor \"" << store->dest.as_or_throw<TensorVar>()
        << "\" is written in the block but is not in the block's signature nor is it covered "
           "by a match_tensor";
    for (const PrimExpr& index : store->indices) {
      if (f_uses_reduction_block_var(index)) {
        return ffi::WalkResult::Interrupt(ffi::VisitInterrupt(false));
      }
    }
    return ffi::WalkResult::Skip();
  };
  auto result = ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(block->body, visit_block,
                                                               visit_alloc, visit_store);
  return result.has_value() ? result.value()->value.cast<bool>() : true;
}

class NoMatchedReducerError : public ScheduleErrorContextObj {
 public:
  explicit NoMatchedReducerError(IRModule mod, ffi::Array<PrimExpr> identities,
                                 ffi::Array<TensorStore> combiners)
      : mod_(std::move(mod)),
        identities_(std::move(identities)),
        combiners_(std::move(combiners)) {}

  ffi::String FastErrorString() const final {
    return "ScheduleError: No matched reducer for the identity and the combiner of this reduction "
           "block. So rfactor and cross-thread reduction cannot be applied.";
  }

  ffi::String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "No matched reducer for identity " << identities_ << " and combiner " << combiners_
       << "In this case rfactor cannot be applied. You can check tvm::tirx::ReducerRegistry for "
          "default reducers or registering new reducers.";
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {}; }

  IRModule mod_;
  ffi::Array<PrimExpr> identities_;
  ffi::Array<TensorStore> combiners_;
};

std::tuple<te::CommReducer, ffi::Array<PrimExpr>, ffi::Array<PrimExpr>> GetReducerAndCombinerLhsRhs(
    const ffi::Optional<ScheduleState>& self, const ffi::Array<PrimExpr>& identities,
    const ffi::Array<TensorStore>& combiners) {
  te::CommReducer reducer{ffi::UnsafeInit{}};
  ffi::Array<PrimExpr> combiner_lhs, combiner_rhs;
  bool matched =
      FromIdentityCombiner(identities, combiners, &reducer, &combiner_lhs, &combiner_rhs);
  if (!matched) {
    if (self.has_value()) {
      throw MakeScheduleError<NoMatchedReducerError>(self.value()->mod, identities, combiners);
    } else {
      TVM_FFI_THROW(ValueError)
          << "No matched reducer for the identity and the combiner of the "
             "reduction block. So rfactor and cross-thread reduction cannot be applied.";
    }
  }
  return std::make_tuple(std::move(reducer), std::move(combiner_lhs), std::move(combiner_rhs));
}

/******** Commutative Reducer ********/

bool MatchReducer(const te::CommReducer& reducer, const ffi::Array<PrimExpr>& identities,
                  const ffi::Array<PrimExpr>& combined_values,
                  const ffi::Array<TensorLoad>& tensor_loads, ffi::Array<PrimExpr>* lhs,
                  ffi::Array<PrimExpr>* rhs) {
  prim::ExprDeepEqual equal;
  TVM_FFI_ICHECK_EQ(identities.size(), combined_values.size());
  int n_tensors = identities.size();
  for (int i = 0; i < n_tensors; ++i) {
    if (!equal(reducer->identity_element[i], identities[i])) {
      return false;
    }
  }

  auto pattern_matcher = ffi::make_object<PatternMatcher>(reducer->result);
  pattern_matcher->Match(combined_values);
  ffi::Array<PrimExpr> lhs_tmp, rhs_tmp;
  lhs_tmp.reserve(n_tensors);
  rhs_tmp.reserve(n_tensors);
  if (!pattern_matcher->Success()) {
    return false;
  }

  for (int i = 0; i < n_tensors; ++i) {
    PrimExpr l = pattern_matcher->Eval(reducer->lhs[i]);
    PrimExpr r = pattern_matcher->Eval(reducer->rhs[i]);
    if (!equal(tensor_loads[i], l)) {
      return false;
    }
    lhs_tmp.push_back(l);
    rhs_tmp.push_back(r);
  }
  *lhs = std::move(lhs_tmp);
  *rhs = std::move(rhs_tmp);
  return true;
}

bool FromIdentityCombiner(const ffi::Array<PrimExpr>& identities,
                          const ffi::Array<TensorStore>& combiners, te::CommReducer* result_reducer,
                          ffi::Array<PrimExpr>* lhs, ffi::Array<PrimExpr>* rhs) {
  int n = identities.size();
  ffi::Array<TensorLoad> tensor_loads;
  ffi::Array<PrimExpr> stored_values;
  tensor_loads.reserve(n);
  stored_values.reserve(n);

  for (int i = 0; i < n; ++i) {
    tensor_loads.push_back(
        TensorLoad(combiners[i]->dest.as_or_throw<TensorVar>(), combiners[i]->indices));
    stored_values.push_back(combiners[i]->value);
  }

  // Check reduction patterns.
  for (const ffi::TypedFunction<ffi::Optional<te::CommReducer>(ffi::Array<PrimExpr>)>&
           reducer_getter : GetReducerGetters()) {
    ffi::Optional<te::CommReducer> reducer = reducer_getter(identities);
    if (!reducer.has_value()) {
      continue;
    }
    if (MatchReducer(reducer.value(), identities, stored_values, tensor_loads, lhs, rhs)) {
      *result_reducer = reducer.value();
      return true;
    }
  }
  return false;
}

}  // namespace s_tir
}  // namespace tvm
