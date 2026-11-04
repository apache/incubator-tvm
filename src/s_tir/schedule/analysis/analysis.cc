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
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/ir/op.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/support/gpu_thread_tag.h>

#include "../ir_comparator.h"
#include "../utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::prim;
using namespace tvm::tirx;

TVM_FFI_STATIC_INIT_BLOCK() {
  TensorizeInfoObj::RegisterReflection();
  AutoTensorizeMappingInfoObj::RegisterReflection();
}

/******** IR Module ********/

const FunctionObj* GetRootFunction(const IRModule& mod, const StmtObj* root_block,
                                   GlobalVar* result_g_var) {
  for (const auto& kv : mod->functions) {
    const GlobalVar& g_var = kv.first;
    const BaseFunc& base_func = kv.second;
    if (const auto* func = base_func.as<FunctionObj>()) {
      if (const auto* realize = func->body.has_value() && func->body.value()->size() == 1
                                    ? func->body.value()->seq[0].as<SBlockRealizeObj>()
                                    : nullptr) {
        if (realize->block.get() == root_block) {
          if (result_g_var != nullptr) {
            *result_g_var = g_var;
          }
          return func;
        }
      }
    }
  }
  TVM_FFI_THROW(IndexError)
      << "Could not get the corresponding function in the schedule state of the "
         "statement:\n"
      << ffi::GetRef<Stmt>(root_block);
  throw;
}

/******** Scope ********/

StmtSRef GetScopeRoot(const ScheduleState& self, const StmtSRef& sref,
                      bool require_stage_pipeline) {
  class RootBlockError : public ScheduleErrorContextObj {
   public:
    explicit RootBlockError(IRModule mod) : mod_(mod) {}
    IRModule mod() const final { return mod_; }
    ffi::String FastErrorString() const final {
      return "ScheduleError: The primitive does not operate on the root block";
    }
    ffi::String DetailRenderTemplate() const final {
      return "The primitive does not operate on the root block";
    }
    ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {}; }
    IRModule mod_;
  };

  class NotStagePipelineError : public ScheduleErrorContextObj {
   public:
    explicit NotStagePipelineError(IRModule mod, SBlock block) : mod_(mod), block_(block) {}
    IRModule mod() const final { return mod_; }
    ffi::String FastErrorString() const final {
      return "ScheduleError: The scope root is not a stage pipeline";
    }
    ffi::String DetailRenderTemplate() const final {
      return R"(The scope {0} is not a stage pipeline.
Definition of a scope that is a stage pipeline:
- The region cover property holds for every of its child blocks
- No write-after-read dependency or opaque dependency,
- only read-after-write and write-after-write are allowed
- All the statements in the scope are schedulable statements, i.e. SBlock and For
)";
    }
    ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_}; }
    IRModule mod_;
    SBlock block_;
  };

  StmtSRef scope_root_sref{ffi::UnsafeInit{}};
  StmtSRef scope_root_subtree{ffi::UnsafeInit{}};
  // Step 1. Find the scope root and the subtree that the given sref is in
  {
    const StmtSRefObj* p = sref->parent;
    const StmtSRefObj* subtree = sref.get();
    for (; p != nullptr; subtree = p, p = p->parent) {
      if (p->stmt->IsInstance<SBlockObj>()) {
        scope_root_sref = ffi::GetRef<StmtSRef>(p);
        scope_root_subtree = ffi::GetRef<StmtSRef>(subtree);
        break;
      }
    }
    if (p == nullptr) {
      throw MakeScheduleError<RootBlockError>(self->mod);
    }
  }
  // Step 2. Handle `require_stage_pipeline`
  if (require_stage_pipeline && self->enable_check) {
    bool stage_pipeline = self->GetSBlockInfo(scope_root_sref).stage_pipeline;
    if (stage_pipeline == false) {
      const SBlockObj* block = TVM_SREF_TO_SBLOCK(scope_root_sref);
      throw MakeScheduleError<NotStagePipelineError>(self->mod, ffi::GetRef<SBlock>(block));
    }
  }
  return scope_root_sref;
}

ScopeBlockLoopInfo GetScopeBlockLoopInfo(const SBlock& scope_block) {
  struct Collector : public StmtExprVisitor {
    using StmtExprVisitor::Visit_;

    ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
      if (value.as<ExprObj>()) return std::nullopt;
      return StmtExprVisitor::Visit(value);
    }

    ffi::Optional<VisitInterrupt> Visit_(const SBlockRealizeObj* realize) final {
      result.realizes.push_back(ffi::GetRef<SBlockRealize>(realize));
      const ffi::Array<IterVar>& iter_vars = realize->block->iter_vars;
      const ffi::Array<PrimExpr>& iter_values = realize->iter_values;
      TVM_FFI_ICHECK_EQ(iter_vars.size(), iter_values.size());
      int n = realize->iter_values.size();
      for (int i = 0; i < n; ++i) {
        const IterVar& iter_var = iter_vars[i];
        const PrimExpr& iter_value = iter_values[i];
        std::unordered_set<const VarObj*>* vars = nullptr;
        if (iter_var->iter_type == IterVarType::kDataPar) {
          vars = &result.spatial_vars;
        } else {
          vars = &result.non_spatial_vars;
        }
        auto walk_fn = [vars](const Var& var) -> ffi::Expected<ffi::WalkResult> {
          if (auto prim_var = var.as<PrimVar>()) {
            vars->insert(prim_var.value().get());
          }
          return ffi::WalkResult::Advance();
        };
        ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(iter_value, walk_fn);
      }
      return std::nullopt;
    }

    ScopeBlockLoopInfo result;
  };
  auto visitor = ffi::make_object<Collector>();
  visitor->Visit(scope_block->body);
  return std::move(visitor->result);
}

/*!
 * \brief Check whether the given sref_a is higher than or equal to sref_b.
 */
void CheckSRefHigherOrEqual(const StmtSRef& sref_a, const StmtSRef& sref_b) {
  const StmtSRefObj* p = sref_b.get();
  for (; p != nullptr; p = p->parent) {
    if (p == sref_a.get()) {
      return;
    }
  }
  TVM_FFI_ICHECK(false) << "Expect StmtSRef " << sref_a << "to be higher than or equal to "
                        << sref_b;
}

/*!
 * \brief Check the dominant property of a block:
 * the block is the only writer of its output, dominating the reader of its output tensors under the
 * given root scope.
 * \param self The schedule state.
 * \param scope_root_sref The StmtSRef corresponding to the root scope.
 * \param block_sref The block whose dominant property is to be checked.
 * \return A boolean indicating if the block is a dominant block.
 */
bool IsDominantBlock(const ScheduleState& self, const StmtSRef& scope_root_sref,
                     const StmtSRef& block_sref) {
  std::unordered_map<TensorVar, ffi::Array<StmtSRef>, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      tensor_writers;
  CheckSRefHigherOrEqual(scope_root_sref, block_sref);
  const SBlockObj* maybe_root_block = scope_root_sref->StmtAs<SBlockObj>();
  if (maybe_root_block) {
    SBlockScope scope = self->GetSBlockScope(scope_root_sref);
    tensor_writers = scope->tensor_writers;
  } else {
    // Collect all child blocks of root sub-tree, and merge their tensor writers.
    ffi::Array<StmtSRef> child_block_srefs = GetChildBlockSRefOnSRefTree(self, scope_root_sref);
    for (const StmtSRef& child_block_sref : child_block_srefs) {
      SBlockScope child_scope = self->GetSBlockScope(child_block_sref);
      for (const auto& it : child_scope->tensor_writers) {
        tensor_writers.insert(it);
      }
    }
  }
  // Check whether the input block is the only writer of its outputs
  const SBlockObj* block = TVM_SREF_TO_SBLOCK(block_sref);
  for (const TensorRegion& write_region : block->writes) {
    if (tensor_writers.count(write_region->source.as_or_throw<tvm::tirx::TensorVar>())) {
      if (tensor_writers.at(write_region->source.as_or_throw<tvm::tirx::TensorVar>()).size() != 1) {
        return false;
      }
    }
  }
  return true;
}

/*!
 * \brief A helper function that checks whether a given block is a complete block under the scope,
 * or return the condition it violates if it is not a complete block
 * \param self The schedule state
 * \param block_sref The block to be checked
 * \param scope_root_sref The sref to the root block of the scope that `block_sref` is in
 * \return 0 if the block is a complete block, or a positive integer indicating which condition is
 * first violated
 */
int CheckCompleteBlockErrorCode(const ScheduleState& self, const StmtSRef& block_sref,
                                const StmtSRef& scope_root_sref) {
  // Cond 1. All block vars are data parallel
  const SBlockObj* block = TVM_SREF_TO_SBLOCK(block_sref);
  for (const IterVar& iter_var : block->iter_vars) {
    if (iter_var->iter_type != kDataPar) {
      return 1;
    }
  }
  // Cond 2. Dominant: the block is the only writer of its output,
  // dominating the reader of its output tensors
  if (!IsDominantBlock(self, scope_root_sref, block_sref)) {
    return 2;
  }
  // Cond 3. No overlap between the tensors the block reads and writes
  std::unordered_set<const VarObj*> written_tensors;
  written_tensors.reserve(block->writes.size());
  for (const TensorRegion& write : block->writes) {
    written_tensors.insert(write->source.as_or_throw<tvm::tirx::TensorVar>().get());
  }
  for (const TensorRegion& read : block->reads) {
    if (written_tensors.count(read->source.as_or_throw<tvm::tirx::TensorVar>().get())) {
      return 3;
    }
  }
  return 0;
}

static const char* kCompleteBlockDefinition = R"(Definition of a complete block:
1) All block vars are data parallel
2) Dominant: the block is the only writer of its output, dominating the reader of its output tensors
3) No overlap between the tensors the block reads and writes)";

static const char* kReductionBlockDefinition = R"(Definition of a reduction block:
1) The block has the `init` statement
2) All the block bindings are quasi-affine expressions
3) All block vars are either data parallel block vars or reduction block vars
4) Dominant: the block is the only writer of its output, dominating the reader of its output tensors
5) The reduction block vars are not used to index the output tensors)";

static const char* kLocalCompleteBlockDefinition = R"(Definition of a local complete block:
1) All block vars are data parallel
2) Local Dominant: the block is the only writer of its output, dominating the reader of its output tensors under a given subtree
3) No overlap between the tensors the block reads and writes)";

static const char* kLocalReductionBlockDefinition = R"(Definition of a reduction block:
1) The block has the `init` statement
2) All the block bindings are quasi-affine expressions
3) All block vars are either data parallel block vars or reduction block vars
4) Local Dominant: the block is the only writer of its output, dominating the reader of its output tensors under a given subtree
5) The reduction block vars are not used to index the output tensors)";

bool IsCompleteBlock(const ScheduleState& self, const StmtSRef& block_sref,
                     const StmtSRef& scope_root_sref) {
  return CheckCompleteBlockErrorCode(self, block_sref, scope_root_sref) == 0;
}

void CheckCompleteBlock(const ScheduleState& self, const StmtSRef& block_sref,
                        const StmtSRef& scope_root_sref) {
  class IncompleteBlockError : public ScheduleErrorContextObj {
   public:
    explicit IncompleteBlockError(IRModule mod, SBlock block, int violated_cond)
        : mod_(std::move(mod)), block_(std::move(block)), violated_cond_(violated_cond) {}
    ffi::String FastErrorString() const final { return "ScheduleError: Incomplete block"; }
    ffi::String DetailRenderTemplate() const final {
      std::ostringstream os;
      os << "The block {0} is not a complete block - it violates condition #" << violated_cond_;
      os << ".\n" << kCompleteBlockDefinition;
      return os.str();
    }
    IRModule mod() const final { return mod_; }
    ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_}; }
    IRModule mod_;
    SBlock block_;
    int violated_cond_;
  };

  int error_code = CheckCompleteBlockErrorCode(self, block_sref, scope_root_sref);
  if (error_code != 0) {
    const SBlockObj* block = TVM_SREF_TO_SBLOCK(block_sref);
    throw MakeScheduleError<IncompleteBlockError>(self->mod, ffi::GetRef<SBlock>(block),
                                                  error_code);
  }
}

/*!
 * \brief A helper function that checks whether a given block is a reduction block under the scope,
 * or return the condition it violates if it is not a reduction block
 * \param self The schedule state
 * \param block_sref The block to be checked
 * \param scope_root_sref The sref to the root block of the scope that `block_sref` is in
 * \return 0 if the block is a reduction block, or a positive integer indicating which condition is
 * first violated
 */
int CheckReductionBlockErrorCode(const ScheduleState& self, const StmtSRef& block_sref,
                                 const StmtSRef& scope_root_sref) {
  const SBlockObj* block = TVM_SREF_TO_SBLOCK(block_sref);
  // Cond 1. The block has the `init` statement.
  if (!block->init.has_value()) {
    return 1;
  }
  // Cond 2. All the block bindings are quasi-affine expressions.
  if (!self->IsAffineBlockBinding(block_sref)) {
    return 2;
  }
  // Cond 3. All block vars are either data parallel block vars or reduction block vars. Meanwhile,
  // we collect all the reduction block vars.
  if (!ContainsOnlyDataParAndReductionBlockIter(block->iter_vars)) {
    return 3;
  }
  // Cond 4. Dominant: the block is the only writer of its output, dominating the reader of its
  // output tensors.
  if (!IsDominantBlock(self, scope_root_sref, block_sref)) {
    return 4;
  }
  // Cond 5. The reduction block vars are not used to index the output tensors.
  return ReductionIterNotIndexOutputTensor(ffi::GetRef<SBlock>(block)) ? 0 : 5;
}

bool IsReductionBlock(const ScheduleState& self, const StmtSRef& block_sref,
                      const StmtSRef& scope_root_sref) {
  return CheckReductionBlockErrorCode(self, block_sref, scope_root_sref) == 0;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.schedule.IsReductionBlock", [](Schedule sch, SBlockRV block_rv,
                                                              SBlockRV scope_block_rv) {
    return IsReductionBlock(sch->state(), sch->GetSRef(block_rv), sch->GetSRef(scope_block_rv));
  });
}

void CheckReductionBlock(const ScheduleState& self, const StmtSRef& block_sref,
                         const StmtSRef& scope_root_sref) {
  class NotReductionBlockError : public ScheduleErrorContextObj {
   public:
    explicit NotReductionBlockError(IRModule mod, SBlock block, int violated_cond)
        : mod_(std::move(mod)), block_(std::move(block)), violated_cond_(violated_cond) {}
    ffi::String FastErrorString() const final { return "ScheduleError: Not a reduction block"; }
    ffi::String DetailRenderTemplate() const final {
      std::ostringstream os;
      os << "The block {0} is not a reduction block - it violates condition #" << violated_cond_;
      os << ".\n" << kReductionBlockDefinition;
      return os.str();
    }
    IRModule mod() const final { return mod_; }
    ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_}; }
    IRModule mod_;
    SBlock block_;
    int violated_cond_;
  };

  int error_code = CheckReductionBlockErrorCode(self, block_sref, scope_root_sref);
  if (error_code != 0) {
    const SBlockObj* block = TVM_SREF_TO_SBLOCK(block_sref);
    throw MakeScheduleError<NotReductionBlockError>(self->mod, ffi::GetRef<SBlock>(block),
                                                    error_code);
  }
}

void CheckCompleteOrReductionBlock(const ScheduleState& self, const StmtSRef& block_sref,
                                   const StmtSRef& scope_root_sref) {
  class NotCompleteOrReductionBlockError : public ScheduleErrorContextObj {
   public:
    explicit NotCompleteOrReductionBlockError(IRModule mod, SBlock block,
                                              int complete_block_error_code,
                                              int reduction_block_error_code)
        : mod_(mod),
          block_(block),
          complete_block_error_code_(complete_block_error_code),
          reduction_block_error_code_(reduction_block_error_code) {}

    ffi::String FastErrorString() const final {
      return "ScheduleError: Not a complete or reduction block";
    }
    ffi::String DetailRenderTemplate() const final {
      std::ostringstream os;
      os << "The block {0} is not a complete block - it violates condition #"
         << complete_block_error_code_;
      os << ".\n" << kCompleteBlockDefinition;
      os << "\nThe block is not a reduction block either - it violates condition #"
         << reduction_block_error_code_;
      os << ".\n" << kReductionBlockDefinition;
      return os.str();
    }
    IRModule mod() const final { return mod_; }
    ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_}; }

    IRModule mod_;
    SBlock block_;
    int complete_block_error_code_;
    int reduction_block_error_code_;
  };

  int complete_block_error_code = CheckCompleteBlockErrorCode(self, block_sref, scope_root_sref);
  if (complete_block_error_code == 0) {
    return;
  }
  int reduction_block_error_code = CheckReductionBlockErrorCode(self, block_sref, scope_root_sref);
  if (reduction_block_error_code == 0) {
    return;
  }
  const SBlockObj* block = TVM_SREF_TO_SBLOCK(block_sref);
  throw MakeScheduleError<NotCompleteOrReductionBlockError>(
      self->mod, ffi::GetRef<SBlock>(block), complete_block_error_code, reduction_block_error_code);
}

void CheckSubtreeCompactDataflow(const ScheduleState& self, const StmtSRef& subtree_root) {
  class NotCompactDataFlowError : public ScheduleErrorContextObj {
   public:
    explicit NotCompactDataFlowError(IRModule mod, Stmt subtree_root, SBlock violate_block,
                                     int local_complete_block_code, int local_reduction_block_code)
        : mod_(std::move(mod)),
          subtree_root_(std::move(subtree_root)),
          violate_block_(std::move(violate_block)),
          local_complete_block_code_(local_complete_block_code),
          local_reduction_block_code_(local_reduction_block_code) {
      TVM_FFI_ICHECK(subtree_root_->IsInstance<SBlockObj>() || subtree_root_->IsInstance<ForObj>());
    }
    ffi::String FastErrorString() const final {
      return "ScheduleError: The queried subtree root in SRef tree does not have compact dataflow, "
             "because some of its child block on SRef tree is neither a local complete block nor a "
             "local reduction block.";
    }
    ffi::String DetailRenderTemplate() const final {
      std::ostringstream os;
      os << "The queried subtree root {0} in SRef tree does not have compact dataflow, because "
            "its child block {1} on SRef tree is neither a local complete block nor a local "
            "reduction block.\n";
      os << "It violates condition #" << local_complete_block_code_
         << " as a local complete block.\n";
      os << kLocalCompleteBlockDefinition << "\n";
      os << "It violates condition #" << local_reduction_block_code_
         << " as a local reduction block.\n";
      os << kLocalReductionBlockDefinition << "\n";
      return os.str();
    }
    IRModule mod() const final { return mod_; }
    ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final {
      return {subtree_root_, violate_block_};
    }

    IRModule mod_;
    Stmt subtree_root_;
    SBlock violate_block_;
    int local_complete_block_code_;
    int local_reduction_block_code_;
  };

  ffi::Array<StmtSRef> child_block_srefs = GetChildBlockSRefOnSRefTree(self, subtree_root);
  for (const StmtSRef& block_sref : child_block_srefs) {
    int local_complete_block_code = CheckCompleteBlockErrorCode(self, block_sref, subtree_root),
        local_reduction_block_code = CheckReductionBlockErrorCode(self, block_sref, subtree_root);
    if (local_complete_block_code != 0 && local_reduction_block_code != 0) {
      const SBlockObj* block = TVM_SREF_TO_SBLOCK(block_sref);
      throw MakeScheduleError<NotCompactDataFlowError>(
          self->mod, ffi::GetRef<Stmt>(subtree_root->stmt), ffi::GetRef<SBlock>(block),
          local_complete_block_code, local_reduction_block_code);
    }
  }
}

bool IsOutputBlock(const ScheduleState& self, const StmtSRef& block_sref,
                   const StmtSRef& scope_root_sref) {
  const SBlockObj* scope_root = TVM_SREF_TO_SBLOCK(scope_root_sref);
  const SBlockObj* block = TVM_SREF_TO_SBLOCK(block_sref);
  std::unordered_set<const VarObj*> scope_allocated;
  scope_allocated.reserve(scope_root->alloc_tensors.size());
  for (const TensorVar& tensor : scope_root->alloc_tensors) {
    scope_allocated.insert(tensor.get());
  }
  for (const TensorRegion& tensor_region : block->writes) {
    if (!scope_allocated.count(tensor_region->source.as_or_throw<tvm::tirx::TensorVar>().get())) {
      return true;
    }
  }
  return false;
}

void CheckNotOutputBlock(const ScheduleState& self, const StmtSRef& block_sref,
                         const StmtSRef& scope_root_sref) {
  class OutputBlockError : public ScheduleErrorContextObj {
   public:
    explicit OutputBlockError(IRModule mod, SBlock block) : mod_(mod), block_(block) {}
    ffi::String FastErrorString() const final {
      return "ScheduleError: Cannot operate on an output block";
    }
    ffi::String DetailRenderTemplate() const final { return "The block {0} is an output block"; }
    IRModule mod() const final { return mod_; }
    ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_}; }

    IRModule mod_;
    SBlock block_;
  };
  if (IsOutputBlock(self, block_sref, scope_root_sref)) {
    const SBlockObj* block = TVM_SREF_TO_SBLOCK(block_sref);
    throw MakeScheduleError<OutputBlockError>(self->mod, ffi::GetRef<SBlock>(block));
  }
}

std::vector<IterVarType> GetSBlockVarTypes(const SBlockObj* block) {
  std::vector<IterVarType> results;
  results.reserve(block->iter_vars.size());
  for (const IterVar& iter_var : block->iter_vars) {
    results.push_back(iter_var->iter_type);
  }
  return results;
}

std::vector<IterVarType> GetSBlockVarTypes(const StmtSRef& block_sref) {
  const SBlockObj* block = TVM_SREF_TO_SBLOCK(block_sref);
  return GetSBlockVarTypes(block);
}

bool IsWriteCache(const StmtSRef& block_sref) {
  const SBlockObj* block = TVM_SREF_TO_SBLOCK(block_sref);
  if (block->writes.size() != 1) {
    return false;
  }
  const TensorRegion& write_region = block->writes[0];
  for (const TensorRegion& read_region : block->reads) {
    auto [exists, surjective, injective, ordered, no_const_read, no_shift_read] =
        AnalyzeReadWritePattern(read_region, write_region);
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=81767
    (void)exists;
    (void)surjective;
    (void)no_const_read;
    (void)no_shift_read;
    if (!(injective && ordered)) {
      return false;
    }
  }
  return true;
}

/******** Binding ********/

bool IsAffineBinding(const SBlockRealize& realize, const ffi::Map<Var, Range>& loop_var_ranges,
                     sym::AnalyzerObj* analyzer) {
  if (loop_var_ranges.empty()) {
    return true;
  }
  ffi::Map<PrimVar, Range> primitive_loop_var_ranges;
  for (const auto& [var, range] : loop_var_ranges) {
    primitive_loop_var_ranges.Set(var.as_or_throw<PrimVar>(), range);
  }
  auto res = sym::DetectIterMap(
      /*indices=*/realize->iter_values,
      /*input_iters=*/primitive_loop_var_ranges,
      /*predicate=*/realize->predicate,
      /*check_level=*/sym::IterMapLevel::Surjective,
      /*analyzer=*/ffi::GetRef<sym::Analyzer>(analyzer),
      /*simplify_trivial_iterators=*/false);
  if (res->indices.empty()) {
    return false;
  }
  for (const sym::IterSumExpr& sum_expr : res->indices) {
    const ffi::Array<sym::IterSplitExpr>& args = sum_expr->args;
    if (!args.empty() && !IsOne(args[0]->scale)) {
      return false;
    }
  }
  return true;
}

void CheckPartialAffineBinding(const ScheduleState& self, SBlock block,
                               const ffi::Optional<StmtSRef>& high_exclusive) {
  class NotAffineBindingError : public ScheduleErrorContextObj {
   public:
    explicit NotAffineBindingError(IRModule mod, SBlock block,
                                   ffi::Optional<StmtSRef> high_exclusive)
        : mod_(std::move(mod)), block_(std::move(block)) {
      if (high_exclusive.has_value()) {
        high_exclusive_loop_ = high_exclusive.value()->StmtAs<ForObj>();
      }
    }
    ffi::String FastErrorString() const final {
      std::ostringstream ss;
      if (high_exclusive_loop_) {
        ss << "ScheduleError: The block is required to have an partial affine binding under "
           << high_exclusive_loop_->var;
      } else {
        ss << "ScheduleError: The block is required to have an affine binding";
      }
      return ss.str();
    }
    ffi::String DetailRenderTemplate() const final {
      std::ostringstream ss;
      if (high_exclusive_loop_) {
        ss << "The block {0} is required to have an partial affine binding under "
           << high_exclusive_loop_->var;
      } else {
        ss << "The block {0} is required to have an affine binding";
      }
      return ss.str();
    }
    IRModule mod() const final { return mod_; }
    ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_}; }
    IRModule mod_;
    SBlock block_;
    const ForObj* high_exclusive_loop_{nullptr};
  };

  StmtSRef block_sref = self->stmt2ref.at(block.get());
  if (self->IsAffineBlockBinding(block_sref)) {
    // check block cached state for global affineness
    return;
  }
  if (block_sref->parent && high_exclusive.has_value()) {
    // if it is not of global affine binding, check affineness under high_exclusive,
    sym::Analyzer analyzer;
    ffi::Map<Var, Range> dom_map =
        LoopDomainOfSRefTreePath(ffi::GetRef<StmtSRef>(block_sref->parent), high_exclusive);
    if (IsAffineBinding(GetSBlockRealize(self, block_sref), dom_map, analyzer.get())) {
      return;
    }
  }
  throw MakeScheduleError<NotAffineBindingError>(self->mod, std::move(block), high_exclusive);
}

void CheckAffineBinding(const ScheduleState& self, SBlock block) {
  CheckPartialAffineBinding(self, std::move(block), std::nullopt);
}

void CheckBlockHasTrivialBinding(const ScheduleState& self, const StmtSRef& block_sref) {
  class NotTrivialBindingError : public ScheduleErrorContextObj {
   public:
    explicit NotTrivialBindingError(IRModule mod, SBlock block)
        : mod_(std::move(mod)), block_(std::move(block)) {}

    ffi::String FastErrorString() const final {
      return "ScheduleError: The binding values of the block are not variables of outer loops.";
    }

    ffi::String DetailRenderTemplate() const final {
      std::ostringstream os;
      os << "The binding values of the {0} are not variables of outer loops.";
      return os.str();
    }

    IRModule mod() const final { return mod_; }
    ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_}; }

   private:
    IRModule mod_;
    SBlock block_;
  };

  if (!IsTrivialBinding(self, block_sref)) {
    throw MakeScheduleError<NotTrivialBindingError>(
        self->mod, ffi::GetRef<SBlock>(block_sref->StmtAs<SBlockObj>()));
  }
}

ffi::Map<Var, Range> LoopDomainOfSRefTreePath(const StmtSRef& low_inclusive,
                                              const ffi::Optional<StmtSRef>& high_exclusive,
                                              const runtime::MemoryScope& extra_relax_scope) {
  ffi::Map<Var, Range> result;
  const StmtSRefObj* p = low_inclusive.get();
  const StmtSRefObj* limit = high_exclusive.has_value() ? high_exclusive.value().get() : nullptr;
  for (; p != limit; p = p->parent) {
    const ForObj* loop = p->StmtAs<ForObj>();
    if (loop == nullptr) {
      break;
    }
    result.Set(loop->var, Range(loop->begin, loop->end));
  }
  if (!IsGlobalMemoryScope(extra_relax_scope)) {
    for (; p; p = p->parent) {
      if (const ForObj* loop = p->StmtAs<ForObj>()) {
        if (tvm::s_tir::GetThreadBinding(loop).has_value()) {
          const ffi::String thread_tag = tvm::s_tir::GetThreadBinding(loop).value();
          if (CanRelaxStorageUnderThread(extra_relax_scope, ParseThreadBindingTag(thread_tag))) {
            result.Set(loop->var, Range(loop->begin, loop->end));
          }
        }
      }
    }
  }
  return result;
}

ffi::Map<Var, PrimExpr> GetBindings(const SBlockRealize& realize) {
  const SBlockObj* block = realize->block.get();
  const ffi::Array<IterVar>& all_lhs = block->iter_vars;
  const ffi::Array<PrimExpr>& all_rhs = realize->iter_values;
  TVM_FFI_ICHECK_EQ(all_lhs.size(), all_rhs.size());
  ffi::Map<Var, PrimExpr> result;
  for (int i = 0, n = all_lhs.size(); i < n; ++i) {
    const IterVar& lhs = all_lhs[i];
    const PrimExpr& rhs = all_rhs[i];
    result.Set(lhs->var, rhs);
  }
  return result;
}

bool GetVarsTouchedByBlockIters(const SBlockRealize& block_realize,
                                std::unordered_set<const VarObj*>* data_par_vars,
                                std::unordered_set<const VarObj*>* reduce_vars) {
  SBlock block = block_realize->block;
  TVM_FFI_CHECK(block_realize->block.same_as(block), ValueError)
      << "The input `block_realize` is required to be the exact BlockRealize of the "
         "input block";

  bool has_block_vars_of_other_types = false;
  TVM_FFI_ICHECK_EQ(block->iter_vars.size(), block_realize->iter_values.size());
  int n = static_cast<int>(block->iter_vars.size());
  for (int i = 0; i < n; ++i) {
    const IterVar& iter_var = block->iter_vars[i];
    const PrimExpr& iter_value = block_realize->iter_values[i];
    std::unordered_set<const VarObj*>* set = nullptr;
    if (iter_var->iter_type == IterVarType::kDataPar) {
      set = data_par_vars;
    } else if (iter_var->iter_type == IterVarType::kCommReduce) {
      set = reduce_vars;
    } else {
      has_block_vars_of_other_types = true;
    }
    if (set == nullptr) {
      continue;
    }
    ffi::Array<Var> vars_in_binding = UndefinedVars(iter_value);
    for (const Var& var : vars_in_binding) {
      set->insert(var.get());
    }
  }

  return has_block_vars_of_other_types;
}

/******** Loop properties ********/

void CheckLoopStartsWithZero(const ScheduleState& self, const StmtSRef& loop_sref,
                             sym::AnalyzerObj* analyzer) {
  class LoopNotStartWithZeroError : public ScheduleErrorContextObj {
   public:
    explicit LoopNotStartWithZeroError(IRModule mod, For loop)
        : mod_(mod), loop_(std::move(loop)) {}

    ffi::String FastErrorString() const final {
      return "ScheduleError: The primitive only supports loop starting with 0";
    }

    ffi::String DetailRenderTemplate() const final {
      return "The loop {0} does not start with 0, which is not supported";
    }

    IRModule mod() const final { return mod_; }
    ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {loop_}; }

    IRModule mod_;
    For loop_;
  };
  const ForObj* loop = TVM_SREF_TO_FOR(loop_sref);
  if (!analyzer->CanProve(loop->begin == 0)) {
    throw MakeScheduleError<LoopNotStartWithZeroError>(self->mod, ffi::GetRef<For>(loop));
  }
}

/******** Block-loop relation ********/

ffi::Array<StmtSRef> GetChildBlockSRefOnSRefTree(const ScheduleState& self,
                                                 const StmtSRef& parent_sref) {
  ffi::Array<SBlockRealize> child_block_realize = GetChildBlockRealizeOnSRefTree(parent_sref);
  ffi::Array<StmtSRef> child_block_srefs;
  child_block_srefs.reserve(child_block_realize.size());

  for (SBlockRealize realize : child_block_realize) {
    child_block_srefs.push_back(self->stmt2ref.at(realize->block.get()));
  }
  return child_block_srefs;
}

ffi::Array<SBlockRealize> GetChildBlockRealizeOnSRefTree(const StmtSRef& parent_sref) {
  struct Collector : public StmtExprVisitor {
    using StmtExprVisitor::Visit_;

    ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
      if (value.as<ExprObj>()) return std::nullopt;
      return StmtExprVisitor::Visit(value);
    }

    static ffi::Array<SBlockRealize> Collect(const Stmt& stmt) {
      auto collector = ffi::make_object<Collector>();
      collector->Visit(stmt);
      return std::move(collector->result_);
    }

    ffi::Optional<VisitInterrupt> Visit_(const SBlockRealizeObj* block_realize) final {
      result_.push_back(ffi::GetRef<SBlockRealize>(block_realize));
      return std::nullopt;
    }

    ffi::Array<SBlockRealize> result_;
  };

  if (parent_sref->stmt->IsInstance<ForObj>()) {
    const auto* loop = static_cast<const ForObj*>(parent_sref->stmt);
    return Collector::Collect(loop->body);
  } else if (parent_sref->stmt->IsInstance<SBlockObj>()) {
    const auto* block = static_cast<const SBlockObj*>(parent_sref->stmt);
    return Collector::Collect(block->body);
  }
  TVM_FFI_ICHECK(false) << "Unreachable";
  throw;
}

SBlockRealize CheckGetSingleChildBlockRealizeOnSRefTree(const ScheduleState& self,
                                                        const StmtSRef& parent_sref) {
  class NonSingleChildBlockError : public ScheduleErrorContextObj {
   public:
    explicit NonSingleChildBlockError(IRModule mod, const StmtSRef& sref)
        : mod_(std::move(mod)), stmt_(ffi::GetRef<Stmt>(sref->stmt)) {
      sref_type_ = stmt_.as<SBlockObj>() != nullptr ? "block" : "loop";
    }

    ffi::String FastErrorString() const final {
      std::ostringstream os;
      os << "ScheduleError: The " << sref_type_ << " is required to have only one child block";
      return os.str();
    }

    ffi::String DetailRenderTemplate() const final {
      std::ostringstream os;
      os << "The " << sref_type_ << " {0} is required to have only one child block";
      return os.str();
    }

    IRModule mod() const final { return mod_; }
    ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {stmt_}; }

    IRModule mod_;
    Stmt stmt_;
    ffi::String sref_type_;
  };

  ffi::Array<SBlockRealize> child_block_realize = GetChildBlockRealizeOnSRefTree(parent_sref);
  if (child_block_realize.size() != 1) {
    throw MakeScheduleError<NonSingleChildBlockError>(self->mod, parent_sref);
  }
  return child_block_realize[0];
}

SBlockRealize GetSBlockRealize(const ScheduleState& self, const StmtSRef& block_sref) {
  struct BlockRealizeFinder : public StmtExprVisitor {
    using StmtExprVisitor::Visit_;

    explicit BlockRealizeFinder(const SBlockObj* target_sblock)
        : target_sblock(target_sblock), result(nullptr) {}

    ffi::Optional<VisitInterrupt> Visit(ffi::AnyView stmt) final {
      if (stmt.as<ExprObj>()) return std::nullopt;
      if (result != nullptr) {
        return std::nullopt;
      }
      return StmtExprVisitor::Visit(stmt);
    }

    ffi::Optional<VisitInterrupt> Visit_(const SBlockRealizeObj* block_realize) final {
      if (block_realize->block.get() == target_sblock) {
        result = block_realize;
      }
      // No need to visit recursively, since the deeper BlockRealizes must not be the result.
      return std::nullopt;
    }

    const SBlockObj* target_sblock;
    const SBlockRealizeObj* result;
  };

  const SBlockObj* block = TVM_SREF_TO_SBLOCK(block_sref);
  if (block_sref->parent == nullptr) {
    const FunctionObj* func = GetRootFunction(self->mod, block, nullptr);
    return func->body.value()->seq[0].as_or_throw<SBlockRealize>();
  } else {
    auto finder = ffi::make_object<BlockRealizeFinder>(block);
    finder->Visit(ffi::GetRef<Stmt>(block_sref->parent->stmt));
    TVM_FFI_CHECK(finder->result != nullptr, InternalError)
        << "Cannot find the BlockRealize of block " << ffi::GetRef<SBlock>(block);
    return ffi::GetRef<SBlockRealize>(finder->result);
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.schedule.GetSBlockRealize", [](Schedule sch, SBlockRV block_rv) {
    return GetSBlockRealize(sch->state(), sch->GetSRef(block_rv));
  });
}

IterVarType GetLoopIterType(const StmtSRef& loop_sref) {
  const ForObj* loop = TVM_SREF_TO_FOR(loop_sref);
  const Var& loop_var = loop->var;
  int n_spatial = 0;
  int n_reduce = 0;
  int n_other = 0;
  auto f_visit = [&loop_var, &n_spatial, &n_reduce,
                  &n_other](const SBlockRealize& realize) -> ffi::Expected<ffi::WalkResult> {
    const SBlockObj* block = realize->block.get();
    // Number of block vars and their bindings
    TVM_FFI_ICHECK_EQ(realize->iter_values.size(), block->iter_vars.size());
    size_t n = realize->iter_values.size();
    for (size_t i = 0; i < n; ++i) {
      const IterVar& iter_var = block->iter_vars[i];
      const PrimExpr& binding = realize->iter_values[i];
      // Categorize the current block var
      int* ref = nullptr;
      if (iter_var->iter_type == IterVarType::kDataPar) {
        ref = &n_spatial;
      } else if (iter_var->iter_type == IterVarType::kCommReduce) {
        ref = &n_reduce;
      } else {
        ref = &n_other;
      }
      // Visit the binding to see if `loop_var` appears
      auto walk_fn = [&ref, &loop_var](const Var& var) -> ffi::Expected<ffi::WalkResult> {
        if (var.same_as(loop_var)) {
          (*ref) += 1;
        }
        return ffi::WalkResult::Advance();
      };
      ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(binding, walk_fn);
    }
    return ffi::WalkResult::Skip();
  };
  ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(loop->body, f_visit);
  if (n_other) {
    return IterVarType::kOpaque;
  } else if (n_spatial && n_reduce) {
    return IterVarType::kOpaque;
  } else if (n_reduce) {
    return IterVarType::kCommReduce;
  } else {
    return IterVarType::kDataPar;
  }
}

StmtSRef GetSRefLowestCommonAncestor(const ffi::Array<StmtSRef>& srefs) {
  TVM_FFI_CHECK(!srefs.empty(), ValueError)
      << "The input array is required to have at least one sref";

  std::unordered_map<const StmtSRefObj*, size_t> sref_visited_cnt;
  for (const StmtSRef& sref : srefs) {
    const StmtSRefObj* p = sref.get();
    while (p != nullptr) {
      ++sref_visited_cnt[p];
      p = p->parent;
    }
  }
  size_t n_sref = srefs.size();
  const StmtSRefObj* p = srefs[0].get();
  while (p != nullptr && sref_visited_cnt[p] != n_sref) {
    p = p->parent;
  }
  TVM_FFI_ICHECK(p != nullptr);
  return ffi::GetRef<StmtSRef>(p);
}

bool HasBeenMultiLevelTiled(const StmtSRef& block_sref) {
  return GetAnn<ffi::String>(block_sref, tvm::s_tir::attr::kMetaScheduleTilingStructure.name)
      .has_value();
}

std::pair<ffi::Array<StmtSRef>, std::vector<int>> CollectComputeLocation(
    const ScheduleState& self, const StmtSRef& block_sref) {
  ffi::Array<StmtSRef> location_srefs;
  std::vector<int> location_indices;

  // Step 1. Add the "compute-root" candidate. Add the "compute-inline" candidate if the block can
  // be inlined.
  if (CanComputeInline(self, block_sref)) {
    location_srefs.push_back(StmtSRef::InlineMark());
    location_indices.push_back(-2);
  }
  location_srefs.push_back(StmtSRef::RootMark());
  location_indices.push_back(-1);

  // Step 2. If the block has no consumer, there is no more candidate.
  ffi::Array<StmtSRef> consumers = GetConsumers(self, block_sref);
  if (consumers.empty()) {
    return std::make_pair(location_srefs, location_indices);
  }

  // Step 3. Get the deepest loop that the input block can be computed at (namely "boundary"). If
  // such a loop cannot be found, there is no more candidate and we just return.
  StmtSRef loop_boundary = consumers.size() > 1 ? GetSRefLowestCommonAncestor(consumers)
                                                : ffi::GetRef<StmtSRef>(consumers[0]->parent);
  if (loop_boundary->StmtAs<ForObj>() == nullptr) {
    return std::make_pair(location_srefs, location_indices);
  }

  // Step 4. Collect the loops outside the first consumer and locate the boundary loop. The position
  // of the boundary loop reveals the number of possible additional candidates.
  ffi::Array<StmtSRef> loop_srefs = GetLoops(consumers[0]);
  size_t lca_pos =
      std::find(loop_srefs.begin(), loop_srefs.end(), loop_boundary) - loop_srefs.begin();
  TVM_FFI_ICHECK_LT(lca_pos, loop_srefs.size());
  size_t n_candidate = lca_pos + 1;

  // Step 5. Find the position of the deepest data-parallel loop among the candidate loops. This
  // position is used for removing the unwanted candidates from the perspective of performance.
  std::vector<IterVarType> loop_iter_types;
  loop_iter_types.reserve(n_candidate);
  int i_last_datapar = -1;
  for (size_t i = 0; i < n_candidate; ++i) {
    // TODO(siyuan): improve the performance
    IterVarType iter_type = GetLoopIterType(loop_srefs[i]);
    loop_iter_types.push_back(iter_type);
    if (iter_type == IterVarType::kDataPar) {
      i_last_datapar = i;
    }
  }
  // Step 6. Check and add the candidates in turn according to the following rules:
  //  - skip the unit loops (loops with extent 1);
  //  - do not consider the data-parallel loops after a not-data-parallel loop;
  //  - do not consider the trailing not-data-parallel loops.
  location_srefs.reserve(n_candidate + 2);
  location_indices.reserve(n_candidate + 2);
  bool visited_reduce = false;
  for (size_t i = 0; i < n_candidate; ++i) {
    if (IsOne(Range::GetExtentFromBeginEnd(TVM_SREF_TO_FOR(loop_srefs[i])->begin,
                                           TVM_SREF_TO_FOR(loop_srefs[i])->end))) {
      continue;
    }

    if (loop_iter_types[i] == IterVarType::kDataPar) {
      if (visited_reduce) {
        break;
      }
    } else {
      visited_reduce = true;
      if (static_cast<int>(i) > i_last_datapar) {
        break;
      }
    }
    if (CanComputeAt(self, block_sref, loop_srefs[i], true)) {
      location_srefs.push_back(loop_srefs[i]);
      location_indices.push_back(i);
    }
  }

  return std::make_pair(location_srefs, location_indices);
}

/******** Producer-consumer relation ********/

ffi::Array<StmtSRef> GetProducers(const StmtSRef& block_sref, const SBlockScope& scope) {
  ffi::Array<Dependency> edges = scope->GetDepsByDst(block_sref);
  ffi::Array<StmtSRef> results;
  std::unordered_set<StmtSRef, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> result_set;
  results.reserve(edges.size());
  for (const Dependency& edge : edges) {
    if ((edge->kind == DepKind::kRAW || edge->kind == DepKind::kWAW) &&
        !result_set.count(edge->src)) {
      results.push_back(edge->src);
      result_set.emplace(edge->src);
    }
  }
  return results;
}

ffi::Array<StmtSRef> GetConsumers(const StmtSRef& block_sref, const SBlockScope& scope) {
  ffi::Array<Dependency> edges = scope->GetDepsBySrc(block_sref);
  ffi::Array<StmtSRef> results;
  std::unordered_set<StmtSRef, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> result_set;
  results.reserve(edges.size());
  for (const Dependency& edge : edges) {
    if ((edge->kind == DepKind::kRAW || edge->kind == DepKind::kWAW) &&
        !result_set.count(edge->dst)) {
      results.push_back(edge->dst);
      result_set.emplace(edge->dst);
    }
  }
  return results;
}

ffi::Array<StmtSRef> GetOutputBlocks(const ScheduleState& self, const SBlockObj* scope_block) {
  struct OutputSBlockCollector : public StmtExprVisitor {
    using StmtExprVisitor::Visit_;

    ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
      if (value.as<ExprObj>()) return std::nullopt;
      return StmtExprVisitor::Visit(value);
    }

    explicit OutputSBlockCollector(const ScheduleState& self) : self_(self) {}

    ffi::Optional<VisitInterrupt> Visit_(const SBlockObj* block) override {
      auto it = self_->stmt2ref.find(block);
      TVM_FFI_ICHECK(it != self_->stmt2ref.end());
      auto block_sref = it->second;
      if (block_sref->parent != nullptr) {
        StmtSRef scope_root_sref =
            GetScopeRoot(self_, block_sref, /*require_stage_pipeline=*/false);
        if (IsOutputBlock(self_, block_sref, scope_root_sref)) {
          results_.push_back(block_sref);
        }
      }
      return StmtExprVisitor::Visit_(block);
    }

    const ScheduleState& self_;
    ffi::Array<StmtSRef> results_;
  };
  auto collector = ffi::make_object<OutputSBlockCollector>(self);
  collector->Visit(scope_block->body);
  auto results = collector->results_;
  return results;
}

ProducerConsumerSplit ProducerConsumerSplit::Find(
    const ScheduleState& self, const ffi::Array<Stmt>& subtrees,
    const ffi::Array<StmtSRef>& producer_block_srefs,
    const ffi::Array<StmtSRef>& consumer_block_srefs,
    std::unordered_map<const SBlockObj*, const SBlockRealizeObj*>* block2realize) {
  class InsertionPointNotFoundError : public ScheduleErrorContextObj {
   public:
    explicit InsertionPointNotFoundError(IRModule mod, int last_producer_position,
                                         int first_consumer_position)
        : mod_(mod),
          last_producer_position_(last_producer_position),
          first_consumer_position_(first_consumer_position) {}

    ffi::String FastErrorString() const final {
      return "ScheduleError: Cannot find the insertion point that satisfies the producer-consumer "
             "constraint";
    }

    ffi::String DetailRenderTemplate() const final {
      return "Cannot find the insertion point that satisfies the producer-consumer constraint. In "
             "0-based indexing, the last producer appears in subtree " +
             std::to_string(last_producer_position_) +
             ", and the first consumer appears in subtree " +
             std::to_string(first_consumer_position_);
    }

    IRModule mod() const final { return mod_; }

    ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {}; }

   private:
    IRModule mod_;
    int last_producer_position_;
    int first_consumer_position_;
  };

  class Finder : public StmtExprVisitor {
   public:
    using StmtExprVisitor::Visit_;

    ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
      if (value.as<ExprObj>()) return std::nullopt;
      return StmtExprVisitor::Visit(value);
    }

    ffi::Optional<VisitInterrupt> Visit_(const SBlockRealizeObj* realize) final {
      const SBlockObj* block = realize->block.get();
      if (block2realize_) {
        block2realize_->emplace(block, realize);
      }
      if (producer_blocks_.count(block)) {
        ++this->n_producers_visited_;
      }
      if (consumer_blocks_.count(block)) {
        ++this->n_consumers_visited_;
      }
      return std::nullopt;
    }

    std::unordered_map<const SBlockObj*, const SBlockRealizeObj*>* block2realize_;
    std::unordered_set<const StmtObj*> producer_blocks_;
    std::unordered_set<const StmtObj*> consumer_blocks_;
    int n_producers_visited_ = 0;
    int n_consumers_visited_ = 0;
  };

  auto finder = ffi::make_object<Finder>();
  finder->block2realize_ = block2realize;
  // Set up the lookup table for producers
  finder->producer_blocks_.reserve(producer_block_srefs.size());
  for (const StmtSRef& block_sref : producer_block_srefs) {
    finder->producer_blocks_.insert(block_sref->stmt);
  }
  // Set up the lookup table for consumers
  finder->consumer_blocks_.reserve(consumer_block_srefs.size());
  for (const StmtSRef& block_sref : consumer_block_srefs) {
    finder->consumer_blocks_.insert(block_sref->stmt);
  }
  // Visit the subtrees
  int n = subtrees.size();
  int last_producer_position = -1;
  int first_consumer_position = n;
  for (int i = 0; i < n; ++i) {
    int n_producers_visited_before = finder->n_producers_visited_;
    int n_consumers_visited_before = finder->n_consumers_visited_;
    finder->Visit(subtrees[i]);
    // Check if the subtree contains at least a producer
    if (finder->n_producers_visited_ != n_producers_visited_before) {
      last_producer_position = i;
    }
    // Check if the subtree contains at least a consumer
    if (finder->n_consumers_visited_ != n_consumers_visited_before) {
      if (first_consumer_position == n) {
        first_consumer_position = i;
      }
    }
  }
  if (last_producer_position >= first_consumer_position) {
    throw MakeScheduleError<InsertionPointNotFoundError>(self->mod, last_producer_position,
                                                         first_consumer_position);
  }
  return ProducerConsumerSplit{last_producer_position,        //
                               first_consumer_position,       //
                               finder->n_producers_visited_,  //
                               finder->n_consumers_visited_};
}

/******** Block-tensor relation ********/

TensorRegion GetNthAccessTensorRegion(const ScheduleState& self, const SBlock& block, int n,
                                      TensorIndexType index_type) {
  class TensorIndexOutOfRangeError : public ScheduleErrorContextObj {
   public:
    explicit TensorIndexOutOfRangeError(IRModule mod, SBlock block, int tensor_index,
                                        TensorIndexType index_type)
        : mod_(std::move(mod)),
          block_(std::move(block)),
          tensor_index_(tensor_index),
          index_type_(index_type) {}

    ffi::String FastErrorString() const final {
      if (index_type_ == TensorIndexType::kWrite) {
        return "ScheduleError: The input `tensor_index` is out of range. It is required to be in "
               "range "
               "[0, num_write_regions) where `num_write_regions` is the number of tensor regions "
               "written by the block.";
      } else {
        return "ScheduleError: The input `tensor_index` is out of range. It is required to be in "
               "range "
               "[0, num_read_regions) where `num_read_regions` is the number of tensor regions "
               "read by the block.";
      }
    }

    ffi::String DetailRenderTemplate() const final {
      std::ostringstream os;
      size_t num =
          index_type_ == TensorIndexType::kWrite ? block_->writes.size() : block_->reads.size();
      os << "The block {0} has " << num << " " << TensorIndexType2Str(index_type_)
         << " regions, so `tensor_index` is required to be in [0, " << num
         << "). However, the input `tensor_index` is " << tensor_index_
         << ", which is out of the expected range.";
      return os.str();
    }

    IRModule mod() const final { return mod_; }
    ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_}; }

   private:
    IRModule mod_;
    SBlock block_;
    int tensor_index_;
    TensorIndexType index_type_;
  };

  const ffi::Array<TensorRegion>& access_region =
      index_type == TensorIndexType::kWrite ? block->writes : block->reads;

  if (n < 0 || static_cast<int>(access_region.size()) <= n) {
    throw MakeScheduleError<TensorIndexOutOfRangeError>(self->mod, block, n, index_type);
  }
  return access_region[n];
}

TensorVar GetNthAccessTensor(const ScheduleState& self, const SBlock& block, int n,
                             TensorIndexType index_type) {
  return GetNthAccessTensorRegion(self, block, n, index_type)
      ->source.as_or_throw<tvm::tirx::TensorVar>();
}

std::pair<ffi::Optional<StmtSRef>, bool> GetTensorDefiningSite(const StmtSRef& block_sref,
                                                               const TensorVar& tensor) {
  // Climb up along the sref tree, and find the block where `tensor` is in alloc_tensors or
  // match_tensors.
  const StmtSRefObj* defining_site_sref = block_sref.get();
  while (defining_site_sref != nullptr) {
    const auto* block = defining_site_sref->StmtAs<SBlockObj>();
    // If this sref is not a block sref, skip it.
    if (block == nullptr) {
      defining_site_sref = defining_site_sref->parent;
      continue;
    }
    // Try to find the tensor in `allloc_tensors`
    for (const TensorVar& alloc_tensor : block->alloc_tensors) {
      if (tensor.same_as(alloc_tensor)) {
        return {ffi::GetRef<StmtSRef>(defining_site_sref), true};
      }
    }
    // We do not allow the tensor being defined in `match_tensor`.
    for (const MatchTensorRegion match_tensor : block->match_tensors) {
      if (tensor.same_as(match_tensor)) {
        return {ffi::GetRef<StmtSRef>(defining_site_sref), false};
      }
    }
    defining_site_sref = defining_site_sref->parent;
  }
  // If we cannot find the defining site block, it means that the tensor must be in the function's
  // tensor_map, which isn't an intermediate tensor.
  return {std::nullopt, false};
}

/******** SRef Tree Related ********/

StmtSRef GetSRefTreeRoot(const StmtSRef& sref) {
  const StmtSRefObj* p = sref.get();
  for (; p->parent != nullptr; p = p->parent) {
  }
  return ffi::GetRef<StmtSRef>(p);
}

void AddShapeVarBounds(const ScheduleState& state, const StmtSRefObj* sref,
                       sym::AnalyzerObj* analyzer) {
  while (sref->parent != nullptr) {
    sref = sref->parent;
  }
  const FunctionObj* f = GetRootFunction(state->mod, sref->stmt, nullptr);
  auto context = analyzer->GetConstraintContext();
  for (const Var& param : f->params) {
    if (auto tensor = param.as<TensorVar>()) {
      for (const PrimExpr& e : tensor.value().shape()) {
        context->Append(sym::PredicateConstraint(e >= 0, /*assume_always=*/true));
      }
    }
  }
  analyzer->SyncConstraintContext(context);
}

/******** Misc ********/

bool HasOp(const Stmt& stmt, const ffi::Array<Op>& ops) {
  std::unordered_set<const ffi::Object*> op_set;
  op_set.reserve(ops.size());
  for (const Op& op : ops) {
    op_set.insert(op.operator->());
  }
  auto walk_fn = [&op_set](const Call& call) -> ffi::Expected<ffi::WalkResult> {
    if (op_set.count(call->op.operator->())) {
      return ffi::WalkResult::Interrupt(ffi::VisitInterrupt(true));
    }
    return ffi::WalkResult::Advance();
  };
  auto result = ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(stmt, walk_fn);
  return result.has_value() ? result.value()->value.cast<bool>() : false;
}

bool HasIfThenElse(const Stmt& stmt) {
  auto visit_realize = [](const SBlockRealize& realize) -> ffi::Expected<ffi::WalkResult> {
    if (!IsOne(realize->predicate)) {
      return ffi::WalkResult::Interrupt(ffi::VisitInterrupt(true));
    }
    return ffi::WalkResult::Advance();
  };
  auto visit_branch = [](const If&) -> ffi::Expected<ffi::WalkResult> {
    return ffi::WalkResult::Interrupt(ffi::VisitInterrupt(true));
  };
  auto visit_select = [](const Select&) -> ffi::Expected<ffi::WalkResult> {
    return ffi::WalkResult::Interrupt(ffi::VisitInterrupt(true));
  };
  auto visit_call = [](const Call& call) -> ffi::Expected<ffi::WalkResult> {
    static const Op if_then_else_op = Op::Get("prim.if_then_else");
    if (call->op.same_as(if_then_else_op)) {
      return ffi::WalkResult::Interrupt(ffi::VisitInterrupt(true));
    }
    return ffi::WalkResult::Advance();
  };
  auto result = ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(stmt, visit_realize, visit_branch,
                                                               visit_select, visit_call);
  return result.has_value() ? result.value()->value.cast<bool>() : false;
}

std::tuple</*exists=*/bool,
           /*surjective=*/bool,
           /*injective=*/bool,
           /*ordered=*/bool,
           /*no_const_read=*/bool,
           /*no_shift_read=*/bool>
AnalyzeReadWritePattern(const TensorRegion& read_region, const TensorRegion& write_region) {
  static constexpr const std::tuple<bool, bool, bool, bool, bool, bool> kNotExist =
      std::make_tuple(false, false, false, false, false, false);
  // Step 1. Extract the write indices
  int w_dim = write_region->source.as_or_throw<tvm::tirx::TensorVar>().shape().size();
  std::unordered_map<const VarObj*, int> var2idx;
  var2idx.reserve(w_dim);
  for (int i = 0; i < w_dim; ++i) {
    const Range& dom = write_region->region[i];
    if (!dom->extent.as<IntImmObj>()) {
      return kNotExist;
    }
    if (auto var = dom->begin.as<PrimVar>()) {
      var2idx.emplace(var.value().get(), i);
    } else {
      return kNotExist;
    }
  }
  // Step 2. Map each read index to a write index
  bool no_const_read = true;
  bool no_shift_read = true;
  int r_dim = read_region->source.as_or_throw<tvm::tirx::TensorVar>().shape().size();
  std::vector<int> mapped(r_dim, -1);
  for (int i = 0; i < r_dim; ++i) {
    const Range& dom = read_region->region[i];
    if (!dom->extent.as<IntImmObj>()) {
      return kNotExist;
    }
    // Case 1. Read index is a constant
    if (dom->begin.as<IntImmObj>()) {
      no_const_read = false;
      continue;
    }
    // Case 2. Read index cannot be recognized as `var +/- const`
    // where `var` is a write index and `const` is an optional constant shift
    ffi::Optional<IntImm> opt_const = std::nullopt;
    ffi::Optional<Var> opt_var = AnalyzeVarWithShift(dom->begin, &opt_const);
    const VarObj* var = opt_var.has_value() ? opt_var.value().get() : nullptr;
    if (var == nullptr || !var2idx.count(var)) {
      return kNotExist;
    }
    // Case 3. Read index is `var +/- const`
    mapped[i] = var2idx.at(var);
    if (opt_const.has_value()) {
      no_shift_read = false;
    }
  }
  // Step 3. Check if the mapping is ordered, and count how many times each var is mapped
  std::vector<int> mapped_counter(w_dim, 0);
  bool ordered = true;
  int last_mapped = -1;
  for (int i : mapped) {
    if (i != -1) {
      ++mapped_counter[i];
      if (last_mapped != -1 && last_mapped > i) {
        ordered = false;
      }
      last_mapped = i;
    }
  }
  // Step 4. Check if the mapping is surjective or injective
  // Surjective: each write index is mapped at least once
  // Injective: each write index is mapped at most once
  bool surjective = true;
  bool injective = true;
  for (int cnt : mapped_counter) {
    if (cnt == 0) {
      surjective = false;
    } else if (cnt >= 2) {
      injective = false;
    }
  }
  return std::make_tuple(/*exist=*/true, surjective, injective, ordered, no_const_read,
                         no_shift_read);
}

/******** Storage Scope ********/

void CheckStorageScope(const ScheduleState& self, ffi::String storage_scope) {
  class InvalidStorageScopeError : public ScheduleErrorContextObj {
   public:
    explicit InvalidStorageScopeError(IRModule mod, ffi::String storage_scope)
        : mod_(std::move(mod)), storage_scope_(std::move(storage_scope)) {}

    ffi::String FastErrorString() const final {
      return "ScheduleError: The input storage scope is invalid";
    }

    ffi::String DetailRenderTemplate() const final {
      return "The input storage scope \"" + storage_scope_ + "\" is invalid.";
    }

    ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {}; }
    IRModule mod() const final { return mod_; }

   private:
    IRModule mod_;
    ffi::String storage_scope_;
  };

  try {
    runtime::MemoryScope::Get(std::string(storage_scope));
  } catch (...) {
    throw MakeScheduleError<InvalidStorageScopeError>(self->mod, std::move(storage_scope));
  }
}

bool IsSpatial(const StmtSRef& block_sref) {
  const SBlockObj* block = TVM_SREF_TO_SBLOCK(block_sref);
  for (const IterVar& iter_var : block->iter_vars) {
    if (iter_var->iter_type != IterVarType::kDataPar) {
      return false;
    }
  }
  return true;
}

bool IsTrivialBinding(const ScheduleState& self, const StmtSRef& block_sref) {
  TVM_SREF_TO_SBLOCK(block_sref);
  ffi::Array<StmtSRef> loops = GetLoops(block_sref);
  ffi::Array<PrimExpr> binds = GetSBlockRealize(self, block_sref)->iter_values;
  if (loops.size() != binds.size()) {
    return false;
  }
  for (int i = 0, n = loops.size(); i < n; ++i) {
    const ForObj* loop = TVM_SREF_TO_FOR(loops[i]);
    if (binds[i].get() != loop->var.get()) {
      return false;
    }
  }
  return true;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.schedule.IsTrivialBinding", [](Schedule sch, SBlockRV block_rv) {
    return IsTrivialBinding(sch->state(), sch->GetSRef(block_rv));
  });
}

bool NeedsMultiLevelTiling(const ScheduleState& self, const StmtSRef& block_sref) {
  if (HasBeenMultiLevelTiled(block_sref)) {
    return false;
  }
  const SBlockObj* block = TVM_SREF_TO_SBLOCK(block_sref);
  if (block->writes.size() != 1 || block->reads.empty() || IsSpatial(block_sref) ||
      !IsTrivialBinding(self, block_sref)) {
    return false;
  }
  const VarObj* write_tensor = block->writes[0]->source.as_or_throw<tvm::tirx::TensorVar>().get();
  // Step 1. Sort out spatial block variables. Skip the block iters of domain [0, 1), since such
  // block iters distracts the following check of the unused block iters.
  std::vector<const VarObj*> spatial_block_vars;
  spatial_block_vars.reserve(block->iter_vars.size());
  for (const IterVar& block_var : block->iter_vars) {
    bool has_trivial_dom =
        IsZero(block_var->dom.value()->begin) && IsOne(block_var->dom.value()->extent);
    if (block_var->iter_type == IterVarType::kDataPar && !has_trivial_dom) {
      spatial_block_vars.push_back(block_var->var.get());
    }
  }
  // Step 2. Enumerate each read region, check the number of block vars that are not used
  // to index the read region
  int total_unused_block_vars = 0;
  std::unordered_set<const VarObj*> read_tensors;
  read_tensors.reserve(block->reads.size());
  for (const TensorRegion& tensor_region : block->reads) {
    const VarObj* tensor = tensor_region->source.as_or_throw<tvm::tirx::TensorVar>().get();
    const ffi::Array<Range>& regions = tensor_region->region;
    // Step 2.1. Duplication of read tensors are not allowed
    if (read_tensors.insert(tensor).second == false) {
      return false;
    }
    // Step 2.2. Skip the reduction tensor
    if (tensor == write_tensor) {
      continue;
    }
    // Step 2.3. Collect the block vars that are used to index the read region
    std::unordered_set<const VarObj*> vars;
    for (const Range& range : regions) {
      if (!range->extent.as<IntImmObj>()) {
        return false;
      }
      for (const Var& var : UndefinedVars(range->begin)) {
        vars.insert(var.get());
      }
    }
    // Step 2.4. Check if the block vars are not used to index the read region
    int n_unused_block_vars = 0;
    for (const VarObj* block_var : spatial_block_vars) {
      if (vars.count(block_var) == 0) {
        ++n_unused_block_vars;
      }
    }
    total_unused_block_vars += n_unused_block_vars;
  }
  return total_unused_block_vars >= 1;
}

bool IsSpatialFunction(const Function& func) {
  auto walk_fn = [](const SBlock& block) -> ffi::Expected<ffi::WalkResult> {
    for (const IterVar& iter_var : block->iter_vars) {
      if (iter_var->iter_type != IterVarType::kDataPar) {
        return ffi::WalkResult::Interrupt(ffi::VisitInterrupt(false));
      }
    }
    return ffi::WalkResult::Advance();
  };
  auto result = ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(func->body, walk_fn);
  return result.has_value() ? result.value()->value.cast<bool>() : true;
}

std::pair<int64_t, int64_t> GetCumulativeSpaceAndReductionLength(const s_tir::ScheduleState& self,
                                                                 const tirx::StmtSRef& block_sref) {
  ffi::Array<tirx::StmtSRef> loops = GetLoops(block_sref);
  int64_t cum_space_len = 1, cum_reduce_len = 1;
  /*
   * Return (-1, -1) if
   *   1. there is some loop with type other than kDataPar and kCommReduce;
   *   2. there is some loop which is dynamic.
   */
  for (const tirx::StmtSRef& loop_sref : loops) {
    s_tir::IterVarType type = GetLoopIterType(loop_sref);
    if (type == s_tir::kDataPar) {
      PrimExpr extent_imm_expr = Range::GetExtentFromBeginEnd(TVM_SREF_TO_FOR(loop_sref)->begin,
                                                              TVM_SREF_TO_FOR(loop_sref)->end);
      const auto* extent_imm = extent_imm_expr.as<IntImmObj>();
      auto extent = extent_imm ? extent_imm->value.as<int64_t>() : std::nullopt;
      if (extent.has_value() && *extent != -1) {
        cum_space_len *= *extent;
      } else {
        return std::make_pair(-1, -1);
      }
    } else if (type == s_tir::kCommReduce) {
      PrimExpr extent_imm_expr = Range::GetExtentFromBeginEnd(TVM_SREF_TO_FOR(loop_sref)->begin,
                                                              TVM_SREF_TO_FOR(loop_sref)->end);
      const auto* extent_imm = extent_imm_expr.as<IntImmObj>();
      auto extent = extent_imm ? extent_imm->value.as<int64_t>() : std::nullopt;
      if (extent.has_value() && *extent != -1) {
        cum_reduce_len *= *extent;
      } else {
        return std::make_pair(-1, -1);
      }
    } else {
      return std::make_pair(-1, -1);
    }
  }
  return std::make_pair(cum_space_len, cum_reduce_len);
}

bool NeedsRFactorOrCrossThreadReduction(const s_tir::ScheduleState& self,  //
                                        const tirx::StmtSRef& block_sref,  //
                                        int64_t max_parallel_extent,       //
                                        int64_t max_parallel_basic) {
  const SBlockObj* block = TVM_SREF_TO_SBLOCK(block_sref);
  ffi::Array<tirx::StmtSRef> loops = GetLoops(block_sref);

  // Cond 1. The block must have at lease one write tensor
  if (block->writes.size() == 0) {
    return false;
  }

  // Cond 2. The block is a reduction block and has trivial binding.
  const StmtSRef& scope_sref = GetScopeRoot(self, block_sref,
                                            /*require_stage_pipeline=*/false);
  if (!IsReductionBlock(self, block_sref, scope_sref)  //
      || !IsTrivialBinding(self, block_sref)           //
      || HasBeenMultiLevelTiled(block_sref)) {
    return false;
  }

  // Cond 3. Every the loop axis must be either spatial axis or reduction axis.
  for (const tirx::StmtSRef& loop_sref : loops) {
    const s_tir::IterVarType& type = GetLoopIterType(loop_sref);
    if (type != s_tir::kDataPar && type != s_tir::kCommReduce) {
      return false;
    }
  }

  // Cond 4. Whether there is at least one reduction loop.
  // Cond 5. The loops are continuous, and the body of the innermost loop is exactly the block.
  bool has_reduction_loop = false;
  for (size_t i = 0; i < loops.size(); ++i) {
    // Cond 4.
    if (GetLoopIterType(loops[i]) == s_tir::kCommReduce) {
      has_reduction_loop = true;
    }

    // Cond 5.
    const ForObj* loop_i = TVM_SREF_TO_FOR(loops[i]);
    if (i < loops.size() - 1) {
      const ForObj* loop_i1 = TVM_SREF_TO_FOR(loops[i + 1]);
      if (loop_i->body->size() != 1 || loop_i->body->seq[0].get() != loop_i1) {
        return false;
      }
    } else {
      const auto* block_realize =
          loop_i->body->size() == 1 ? loop_i->body->seq[0].as<s_tir::SBlockRealizeObj>() : nullptr;
      if (!block_realize || block_realize->block.get() != block) {
        return false;
      }
    }
  }
  if (!has_reduction_loop) {
    return false;
  }

  // Cond 6. Can successfully calculating the cumulative loop length.
  auto [cum_space_len, cum_reduce_len] = GetCumulativeSpaceAndReductionLength(self, block_sref);
  if (cum_space_len == -1 || cum_reduce_len == -1) {
    return false;
  }

  // Cond 7.
  if (NeedsMultiLevelTiling(self, block_sref)) {
    // Do not use rfactor/cross-thread-reduction if we have enough parallelism on spatial loops.
    return !(cum_space_len >= cum_reduce_len || cum_space_len > max_parallel_extent);
  } else {
    // Always try rfactor/cross-thread-reduction for other reduction blocks.
    return cum_reduce_len > 1;
  }
}

PrimExpr SimplifyNonTrivialExpr(const PrimExpr& expr, sym::AnalyzerObj* analyzer) {
  auto simplified = analyzer->Simplify(expr);
  if (simplified->IsInstance<IntImmObj>()) {
    return expr;
  } else {
    return simplified;
  }
}

/*! \brief Auxiliary data structure of information extracted from tensor intrin description */
struct TensorIntrinDescInfo {
  /*! \brief The block of the description function, which is the (unique) direct child of the root
   *         block.
   */
  const SBlockRealizeObj* desc_block = nullptr;
  /*! \brief The loops of the description function, in the order from outer loops to inner ones. */
  std::vector<const ForObj*> desc_loops;
  /*! \brief The loop variables. */
  std::unordered_set<const tvm::VarObj*> desc_loop_vars;
};

/*!
 * \brief Extract auxilary information from the tensor intrin description.
 * \param analyze The arithmetic analyzer
 * \param desc_func The description Function
 * \return The auxilary information
 */
TensorIntrinDescInfo ExtractTensorIntrinDescInfo(sym::AnalyzerObj* analyzer,
                                                 const Function& desc_func) {
  TensorIntrinDescInfo info;
  const auto* desc_scope_realize =
      desc_func->body.has_value() && desc_func->body.value()->size() == 1
          ? desc_func->body.value()->seq[0].as<SBlockRealizeObj>()
          : nullptr;
  TVM_FFI_ICHECK(desc_scope_realize);
  {
    auto visit_block = [&](const SBlockRealize& block) -> ffi::Expected<ffi::WalkResult> {
      info.desc_block = block.get();
      return ffi::WalkResult::Advance();
    };
    auto visit_loop = [&](const For& loop) -> ffi::Expected<ffi::WalkResult> {
      info.desc_loops.push_back(loop.get());
      info.desc_loop_vars.insert(loop->var.get());
      if (!analyzer->CanProve(loop->begin == 0)) {
        return ffi::WalkResult::Advance();
      }
      return ffi::WalkResult::Advance();
    };
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(desc_scope_realize->block->body, visit_block,
                                                    visit_loop);
    std::reverse(info.desc_loops.begin(), info.desc_loops.end());
    TVM_FFI_ICHECK(info.desc_block);
  }
  return info;
}

ffi::Optional<TensorizeInfo> GetTensorizeLoopMapping(const s_tir::ScheduleState& self,
                                                     const tirx::StmtSRef& block_sref,
                                                     const tirx::Function& desc_func,
                                                     bool allow_padding) {
  sym::Analyzer analyzer;
  const s_tir::SBlockRealize& block = GetSBlockRealize(self, block_sref);
  // Step 1. Analyze desc_func, extract its block, loops and loop vars
  TensorIntrinDescInfo desc_info = ExtractTensorIntrinDescInfo(analyzer.get(), desc_func);
  // Step 2. Collect loops from block_sref
  const tirx::StmtSRef& scope_sref = GetScopeRoot(self, block_sref, false);
  TVM_SREF_TO_SBLOCK(scope_sref);
  std::vector<const ForObj*> block_loops;
  std::unordered_set<const tvm::VarObj*> block_loop_vars;
  {
    for (const tirx::StmtSRefObj* loop_sref = block_sref->parent;; loop_sref = loop_sref->parent) {
      const auto* loop = loop_sref->StmtAs<ForObj>();
      if (loop == nullptr || loop->body->size() != 1) {
        break;
      }
      block_loops.push_back(loop);
      block_loop_vars.insert(loop->var.get());
      if (!analyzer->CanProve(loop->begin == 0)) {
        return std::nullopt;
      }
    }
    std::reverse(block_loops.begin(), block_loops.end());
  }
  // Step 3. Map from block loops to desc block loops
  const std::vector<const ForObj*>& desc_loops = desc_info.desc_loops;
  const std::unordered_set<const VarObj*>& desc_loop_vars = desc_info.desc_loop_vars;
  const SBlockRealizeObj* desc_block = desc_info.desc_block;
  ffi::ObjectPtr<TensorizeInfoObj> ret = ffi::make_object<TensorizeInfoObj>();
  const int n_block_vars = block->iter_values.size();
  const int n_desc_vars = desc_block->iter_values.size();
  const int offset = n_block_vars - n_desc_vars;

  std::unordered_map<int, int> block_index_to_padding;  // padding of each block iter if necessary

  if (offset < 0) {
    return std::nullopt;
  }

  const std::vector<IterVarType> iter_types_block = GetSBlockVarTypes(block_sref);
  const std::vector<IterVarType> iter_types_desc = GetSBlockVarTypes(desc_block->block.get());

  TVM_FFI_ICHECK(desc_loops.size() == static_cast<size_t>(n_desc_vars));
  TVM_FFI_ICHECK(block_loops.size() == iter_types_block.size());

  // We assume that the orders of iter_vars in the target and the desc block are consistent.
  // Based on that assumption, the following logic supports arbitrary permutations of a loop order,
  // such as

  // for k:
  //   for i:
  //     for j:
  //       C[i, j] += A[i, k] * B[k, j]

  // or

  // for i:
  //   for j:
  //     for k:
  //       C[i, j] += A[i, k] * B[k, j]

  int next_block_ind = block_loops.size() - 1;
  auto desc_walkfn = [&desc_loop_vars](const Var& var) -> ffi::Expected<ffi::WalkResult> {
    return desc_loop_vars.count(var.get()) ? ffi::WalkResult::Interrupt(ffi::VisitInterrupt(var))
                                           : ffi::WalkResult::Advance();
  };
  auto block_walkfn = [&block_loop_vars](const Var& var) -> ffi::Expected<ffi::WalkResult> {
    return block_loop_vars.count(var.get()) ? ffi::WalkResult::Interrupt(ffi::VisitInterrupt(var))
                                            : ffi::WalkResult::Advance();
  };
  for (int i_desc = n_desc_vars - 1; i_desc >= 0; --i_desc) {
    // Step 3.1. Find the corresponding loop of the i_desc-th block var of desc
    const PrimExpr& desc_bind = desc_block->iter_values[i_desc];
    const ForObj* desc_loop = nullptr;
    IterVarType iter_type_desc = iter_types_desc[i_desc];
    for (int i = 0, n = desc_loops.size(); i < n; ++i) {
      // Check if desc_bind = loops[i]->var + stuff-irrelevant-of-loop-vars
      PrimExpr residual = analyzer->Simplify(desc_bind - desc_loops[i]->var);
      if (!ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(residual, desc_walkfn).has_value()) {
        desc_loop = desc_loops[i];
        iter_type_desc = iter_types_desc[i];
        break;
      }
    }
    if (desc_loop == nullptr ||
        Range::GetExtentFromBeginEnd(desc_loop->begin, desc_loop->end).as<IntImmObj>() == nullptr) {
      return std::nullopt;
    }

    PrimExpr int_desc_extent_expr = Range::GetExtentFromBeginEnd(desc_loop->begin, desc_loop->end);
    const IntImmObj* int_desc_extent = int_desc_extent_expr.as<IntImmObj>();

    // Step 3.2. Find the corresponding iter_value of the target block with a matching iterator type
    ffi::Optional<PrimExpr> block_bind;
    int current_block_ind = next_block_ind;
    for (; current_block_ind >= 0; --current_block_ind) {
      if (iter_types_block[current_block_ind] == iter_type_desc) {
        next_block_ind = current_block_ind - 1;
        block_bind = block->iter_values[current_block_ind];
        break;
      }
    }

    if (!block_bind.has_value()) return std::nullopt;

    // Step 3.3. Find the corresponding loop of the target block
    for (int i = 0, n = block_loops.size(); i < n; ++i) {
      // Check if block_bind = block_loops[i]->var + stuff-irrelevant-of-loop-vars
      const ForObj* block_loop = block_loops[i];
      const tirx::StmtSRef& block_loop_sref = self->stmt2ref.at(block_loop);
      // Skip i-th loop if it has already been mapped
      if (ret->loop_map.find(block_loop_sref) != ret->loop_map.end()) continue;

      PrimExpr residual = analyzer->Simplify(block_bind.value() - block_loops[i]->var);
      if (ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(residual, block_walkfn).has_value()) {
        continue;
      }
      // padding is allowed only when the block has trivial bindings
      if (allow_padding && !IsZero(residual)) {
        allow_padding = false;
      }

      PrimExpr int_block_extent_expr =
          Range::GetExtentFromBeginEnd(block_loops[i]->begin, block_loops[i]->end);
      const IntImmObj* int_block_extent = int_block_extent_expr.as<IntImmObj>();

      // Check divisibility
      if (!int_block_extent) {
        return std::nullopt;
      }
      ffi::BigInt remainder = int_block_extent->value % int_desc_extent->value;
      if (remainder != 0) {
        if (allow_padding) {
          // If the block loop is not divisible by the desc loop, we pad the block loop to make it
          // divisible if padding is allowed.
          block_index_to_padding[current_block_ind] = static_cast<int64_t>(int_desc_extent->value);
        } else {
          return std::nullopt;
        }
      }

      ret->loop_map.Set(block_loop_sref, ffi::GetRef<For>(desc_loop));
      break;
    }
  }

  for (int i = 0, n = desc_loops.size(); i < n; ++i) {
    ret->desc_loop_indexer.Set(ffi::GetRef<For>(desc_loops[i]), static_cast<int64_t>(i));
  }
  if (!block_index_to_padding.empty()) {
    if (!allow_padding) {
      return std::nullopt;
    }
    ffi::Array<int64_t> paddings;
    for (int i = 0, n = block->block->iter_vars.size(); i < n; ++i) {
      if (auto it = block_index_to_padding.find(i); it != block_index_to_padding.end()) {
        paddings.push_back(static_cast<int64_t>(it->second));
      } else {
        paddings.push_back(1);
      }
    }
    ret->block_iter_paddings = std::move(paddings);
  }

  return TensorizeInfo(ret);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("s_tir.schedule.IsSpatialFunction", IsSpatialFunction)
      .def("s_tir.schedule.GetTensorizeLoopMapping", [](Schedule sch, SBlockRV block,
                                                        Function desc_func, bool allow_padding) {
        return GetTensorizeLoopMapping(sch->state(), sch->GetSRef(block), desc_func, allow_padding);
      });
}

/******** Auto Tensorization ********/

/*! \brief IndexMap proposer for layout transformation in auto tensorization. */
class AutoTensorizeMappingProposer {
 public:
  static ffi::Array<IndexMap> ProposeMappings(const AutoTensorizeComparator* extractor,
                                              sym::AnalyzerObj* analyzer) {
    AutoTensorizeMappingProposer proposer(extractor, analyzer);
    proposer.CollectFeasibleSet();
    return proposer.ProposeAllFuseMapping();
  }

 private:
  explicit AutoTensorizeMappingProposer(const AutoTensorizeComparator* extractor,
                                        sym::AnalyzerObj* analyzer)
      : extractor_(extractor), analyzer_(analyzer) {}

  using VarSet = std::unordered_set<Var>;

  void CollectFeasibleSet() {
    // Collect the set of potential iter var mapping between the workload and the tensor intrin.
    // We analyze the appearance of each variable in the tensor indices of each tensor on LHS and
    // RHS. The appearance of a variable in the tensor indices is encoded as bit-masks (TensorMask).
    // Variables on the LHS and the RHS with the same bit-mask and the same iter type are potential
    // mappings.
    //
    // For example, consider the conv2d case. We will try to match the workload
    // conv2d[n, h, w, c] = sum_{rh, rw, rc} X[n, h + rh, w + rw, c + rc] * W[rh, rw, rc, c]
    // against a matmul tensor intrin
    // C[m, n] = sum_{k} A[m, k] * B[k, n]
    // First we extract the correspondence of the tensors: conv2d <=> C, A <=> X, B <=> W.
    // Then for each variable, we extract the tensors where it is used for indexing.
    // Take the variable m on the RHS as an example. m is used to index tensor A and C. On the LHS,
    // we will find the variables used to index only the exact corresponding tensors conv2d and X
    // (the variable is not allowed to index other tensors). In this case, n, h, w is used to index
    // both tensor conv2d and W, and not in other tensors. Therefore, {n, h, w} <=> m is a potential
    // mapping.

    // Note: the mapping is not unique when multiple variables on RHS has the same bit-mask.
    // This is currently not supported.

    using TensorMask = std::vector<bool>;

    // Step 1: Assign an index to each tensor in LHS and RHS
    std::unordered_map<TensorVar, int, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> rhs_tensor_index;
    std::unordered_map<TensorVar, int, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> lhs_tensor_index;
    {
      int i = 0;
      for (const auto& kv : extractor_->rhs_tensor_map_) {
        const TensorVar& rhs_tensor = kv.first;
        const TensorVar& lhs_tensor = kv.second;
        rhs_tensor_index[rhs_tensor] = i;
        lhs_tensor_index[lhs_tensor] = i;
        ++i;
      }
    }

    // Step 2: Compute the tensor mask
    TVM_FFI_ICHECK_EQ(rhs_tensor_index.size(), lhs_tensor_index.size());
    int num_tensors = rhs_tensor_index.size();
    std::unordered_map<const VarObj*, std::vector<bool>> rhs_tensor_masks, lhs_tensor_masks;
    // helper function to initialize or update the tensor mask
    auto update_mask = [&](const VarObj* var,
                           std::unordered_map<const VarObj*, std::vector<bool>>* masks, int i) {
      if (!masks->count(var)) {
        (*masks)[var].resize(num_tensors);
      }
      (*masks)[var][i] = true;
    };

    for (const auto& it : extractor_->rhs_tensor_indices_map_) {
      const TensorVar& rhs_tensor = it.first;
      for (const PrimExpr& rhs_index : it.second) {
        if (auto var = rhs_index.as<PrimVar>()) {
          update_mask(var.value().get(), &rhs_tensor_masks, rhs_tensor_index.at(rhs_tensor));
        } else {
          TVM_FFI_THROW(ValueError)
              << "Tensor index " << rhs_index
              << " other that variables in tensor intrinsics is not supported.";
        }
      }

      auto lhs_tensor_it = extractor_->rhs_tensor_map_.find(rhs_tensor);
      TVM_FFI_ICHECK(lhs_tensor_it != extractor_->rhs_tensor_map_.end());
      const TensorVar& lhs_tensor = lhs_tensor_it->second;
      auto walk_fn = [&](const Var& var) -> ffi::Expected<ffi::WalkResult> {
        if (auto prim_var = var.as<PrimVar>()) {
          update_mask(prim_var.value().get(), &lhs_tensor_masks, lhs_tensor_index.at(lhs_tensor));
        }
        return ffi::WalkResult::Advance();
      };
      for (const PrimExpr& index : extractor_->lhs_tensor_indices_map_.at(lhs_tensor)) {
        ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(index, walk_fn);
      }
    }

    // Step 3: Find variables on LHS and RHS with the same tensor mask. Ensure LHS and RHS vars
    // have the same iter type.
    std::unordered_map<TensorMask, VarSet> mask_to_rhs_vars;
    for (const auto& kv : rhs_tensor_masks) {
      const VarObj* rhs_var = kv.first;
      const TensorMask& mask = kv.second;
      mask_to_rhs_vars[mask].insert(ffi::GetRef<Var>(rhs_var));
    }
    std::unordered_map<const VarObj*, IterVarType> rhs_var_iter_type;
    for (const auto& iter : extractor_->rhs_iters_) {
      rhs_var_iter_type.emplace(iter->var.get(), iter->iter_type);
    }
    for (const auto& iter : extractor_->lhs_iters_) {
      auto& potential_mappings = lhs_feasible_vars_[iter->var];
      VarSet rhs_candidates = mask_to_rhs_vars[lhs_tensor_masks[iter->var.get()]];
      std::copy_if(
          rhs_candidates.begin(), rhs_candidates.end(),
          std::inserter(potential_mappings, potential_mappings.begin()),
          [&](const Var& var) { return rhs_var_iter_type.at(var.get()) == iter->iter_type; });
    }
  }

  ffi::Array<IndexMap> ProposeAllFuseMapping() {
    // Now we have calcuated potential mapping for each iter var on LHS. For iters on LHS mapped to
    // the same iter on RHS, they will be fused in the original order in LHS block iters. We will
    // generate IndexMap to represent such fusion on LHS. For example, if n, h, w on LHS are mapped
    // to the same iter var on RHS, we will produce index map `lambda n, h, w: fuse(n, h, w)`, where
    // fuse(v0, .., vn) = ((v0 * v1_extent + v1) + ... ) * vn_extent + vn

    // the parameters of the result index map, each parameter corresponds to a LHS iter
    ffi::Array<Var> index_map_src;
    // the outputs of the result index map
    ffi::Array<PrimExpr> index_map_tgt;

    // Step 1: Collect extents of LHS iters and prepare the initial indices of the IndexMap
    ffi::Map<Var, PrimExpr> lhs_iter_extents;
    for (const auto& iter : extractor_->lhs_iters_) {
      lhs_iter_extents.Set(iter->var, iter->dom.value()->extent);
      index_map_src.push_back(iter->var.CopyWithSuffix(""));
    }

    // Step 2: Each iter on RHS has a group of corresponding iters on LHS. Initialize the fusion
    // result for each group of iters on LHS.
    ffi::Map<Var, PrimExpr> fused_lhs_iters;
    for (const auto& iter : extractor_->rhs_iters_) {
      fused_lhs_iters.Set(iter->var, 0);
    }

    // Step 3: Fuse LHS iters mapped to the same RHS iter
    std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> used_rhs_vars;
    for (size_t i = 0; i < extractor_->lhs_iters_.size(); ++i) {
      const Var& lhs_iter_var = extractor_->lhs_iters_[i]->var;
      const VarSet& rhs_candidates = lhs_feasible_vars_[lhs_iter_var];
      if (rhs_candidates.empty()) {
        // put unmapped iters at the beginning
        index_map_tgt.push_back(index_map_src[i].as_or_throw<PrimExpr>());
      } else if (rhs_candidates.size() == 1) {
        Var rhs_var = *rhs_candidates.begin();
        PrimExpr fused_lhs = fused_lhs_iters.at(rhs_var);
        PrimExpr updated_fused_lhs = fused_lhs * lhs_iter_extents.at(lhs_iter_var) +
                                     index_map_src[i].as_or_throw<PrimExpr>();
        fused_lhs_iters.Set(rhs_var, updated_fused_lhs);
        used_rhs_vars.insert(rhs_var);
      } else {
        // non-unique mapping is not supported
        return {};
      }
    }
    for (const auto& iter : extractor_->rhs_iters_) {
      if (!used_rhs_vars.count(iter->var)) {
        return {};
      }
      index_map_tgt.push_back(analyzer_->Simplify(fused_lhs_iters[iter->var]));
    }
    // At most one mapping is supported.
    return {IndexMap(index_map_src.Map([](Var var) { return var.as_or_throw<PrimVar>(); }),
                     index_map_tgt)};
  }

 private:
  // The extractor that has extracted information for auto tensorization from the workload and the
  // tensor intrin.
  const AutoTensorizeComparator* extractor_;
  // The arithmetic analyzer.
  sym::AnalyzerObj* analyzer_;
  /*! \brief Potential mappings on RHS for each variable on LHS */
  std::unordered_map<Var, VarSet> lhs_feasible_vars_;
};

bool CheckAutoTensorizeApplicable(const ScheduleState& state, const tirx::StmtSRef& block_sref,
                                  const tirx::Function& desc_func,
                                  AutoTensorizeComparator* extractor) {
  // Step 1. Analyze desc_func, extract its block, loops and loop vars
  // Step 2. Check if `desc_block` matches `block`
  // Ignore the scope of tensors when comparing, since we can do cache_read/write
  const SBlockRealize& block = GetSBlockRealize(state, block_sref);
  sym::Analyzer analyzer;
  auto desc_info = ExtractTensorIntrinDescInfo(analyzer.get(), desc_func);

  return extractor->Dispatch(block->block, desc_info.desc_block->block);
}

bool CheckAutoTensorizeApplicable(const s_tir::Schedule& sch, const s_tir::SBlockRV& block_rv,
                                  const tirx::Function& desc_func) {
  AutoTensorizeComparator extractor(sch->state()->mod);
  return CheckAutoTensorizeApplicable(sch->state(), sch->GetSRef(block_rv), desc_func, &extractor);
}

ffi::Optional<AutoTensorizeMappingInfo> GetAutoTensorizeMappingInfo(
    const s_tir::ScheduleState& self, const tirx::StmtSRef& block_sref,
    const tirx::Function& desc_func) {
  AutoTensorizeComparator extractor(self->mod);
  if (!CheckAutoTensorizeApplicable(self, block_sref, desc_func, &extractor)) {
    return std::nullopt;
  }
  sym::Analyzer analyzer;
  ffi::Array<IndexMap> mappings =
      AutoTensorizeMappingProposer::ProposeMappings(&extractor, analyzer.get());
  if (mappings.empty()) {
    return std::nullopt;
  }
  ffi::ObjectPtr<AutoTensorizeMappingInfoObj> ret = ffi::make_object<AutoTensorizeMappingInfoObj>();
  ret->mappings = std::move(mappings);
  ret->lhs_tensor_map = std::move(extractor.lhs_tensor_map_);
  ret->rhs_tensor_indices = std::move(extractor.rhs_tensor_indices_map_);
  ret->lhs_iters = std::move(extractor.lhs_iters_);
  ret->rhs_iters = std::move(extractor.rhs_iters_);
  return AutoTensorizeMappingInfo(ret);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("s_tir.schedule.GetAutoTensorizeMappingInfo",
           [](Schedule sch, SBlockRV block, Function desc_func) {
             return GetAutoTensorizeMappingInfo(sch->state(), sch->GetSRef(block), desc_func);
           })
      .def("s_tir.schedule.HasBlock", HasBlock)
      .def("s_tir.schedule.IsOutputBlock",
           [](Schedule sch, SBlockRV block) {
             auto state = sch->state();
             auto block_sref = sch->GetSRef(block);
             return IsOutputBlock(state, block_sref, GetScopeRoot(state, block_sref, false));
           })
      .def("s_tir.schedule.GetLoopIterType",
           [](Schedule sch, LoopRV loop) -> ffi::String {
             IterVarType kind = GetLoopIterType(sch->GetSRef(loop));
             if (kind == kDataPar) {
               return "S";
             } else if (kind == kCommReduce) {
               return "R";
             } else {
               return "O";
             }
           })
      .def("s_tir.schedule.HasIfThenElse",
           [](const Stmt& stmt) -> bool { return HasIfThenElse(stmt); });
}

}  // namespace s_tir
}  // namespace tvm
