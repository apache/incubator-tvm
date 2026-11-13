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
#ifndef TVM_S_TIR_SCHEDULE_UTILS_H_
#define TVM_S_TIR_SCHEDULE_UTILS_H_

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/json.h>
#include <tvm/ffi/extra/serialization.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/schedule/instruction.h>
#include <tvm/s_tir/schedule/schedule.h>
#include <tvm/s_tir/schedule/state.h>
#include <tvm/s_tir/schedule/trace.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/utils.h>
#include <tvm/support/gpu_thread_tag.h>
#include <tvm/sym/analyzer.h>
#include <tvm/sym/int_set.h>
#include <tvm/sym/iter_affine_map.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/function.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "../memory_scope.h"
#include "../support/array_utils.h"
#include "../support/nd_int_set.h"
#include "./analysis.h"
#include "./error.h"
#include "./instruction_traits.h"
#include "./primitive.h"
#include "./transform.h"

namespace tvm {
namespace s_tir {
using namespace tvm::prim;
using namespace tvm::tirx;
/*!
 * \brief Convert an array of loop StmtSRefs to an array of loops
 * \param loop_srefs The loop StmtSRefs to be converted
 * \return The conversion result loops
 */
inline ffi::Array<For> LoopSRefs2Loops(const ffi::Array<StmtSRef>& loop_srefs) {
  ffi::Array<For> loops;
  loops.reserve(loop_srefs.size());
  for (StmtSRef loop_sref : loop_srefs) {
    const ForObj* loop = TVM_SREF_TO_FOR(loop_sref);
    loops.push_back(ffi::GetRef<For>(loop));
  }
  return loops;
}

/*!
 * \brief Convert an array of block rvs to an array of block StmtSRefs
 * \param sch The schedule used to evaluate the random variables
 * \param block_rvs The random variables to be converted
 * \return The conversion result srefs
 */
inline ffi::Array<StmtSRef> SBlockRVs2StmtSRefs(const Schedule& sch,
                                                const ffi::Array<SBlockRV>& block_rvs) {
  ffi::Array<StmtSRef> block_srefs;
  block_srefs.reserve(block_rvs.size());
  for (const SBlockRV& block_rv : block_rvs) {
    block_srefs.push_back(sch->GetSRef(block_rv));
  }
  return block_srefs;
}

/******** Storage scope ********/

/*!
 * \brief Determine if iterators of a storage scope should be relaxed
 * under a specific thread scope
 * \param storage_scope The storage scope that the iterators are on
 * \param thread_scope The thread scope to be relaxed
 * \return A boolean indicating the result
 */
inline bool CanRelaxStorageUnderThread(const runtime::MemoryScope& storage_scope,
                                       const support::GPUThreadTag& thread_scope) {
  static const auto scope_warp = runtime::MemoryScope::Get("warp");
  static const auto scope_shared = runtime::MemoryScope::Get("shared");
  static const auto scope_shared_dyn = runtime::MemoryScope::Get("shared.dyn");

  if (storage_scope == scope_warp) {
    // for warp memory, we only relax threadIdx.x
    return thread_scope.rank == 1 && thread_scope.dim_index == 0;
  }
  // Global spaces are visible across blocks; shared spaces only across threads.
  if (IsGlobalMemoryScope(storage_scope)) {
    return thread_scope.rank >= 0;
  }
  return (storage_scope == scope_shared || storage_scope == scope_shared_dyn) &&
         thread_scope.rank == 1;
}

/******** SeqStmt ********/

/*!
 * \brief Remove a specific Stmt from a SeqStmt. If a SeqStmt contains a BlockRealize,
 * whose block is the Stmt to be removed, then remove that BlockRealize too.
 * \param seq The SeqStmt to be removed from
 * \param to_remove The Stmt to be removed
 * \return The removal result
 */
inline SeqStmt RemoveFromSeqStmt(const SeqStmt& seq, const Stmt& to_remove) {
  ffi::Array<Stmt> new_stmts;
  new_stmts.reserve(seq->size());
  for (const Stmt& stmt : seq->seq) {
    if (to_remove.same_as(stmt)) {
      continue;
    }
    if (const auto* realize = stmt.as<SBlockRealizeObj>()) {
      if (to_remove.same_as(realize->block)) {
        continue;
      }
    }
    new_stmts.push_back(stmt);
  }
  return SeqStmt(new_stmts, seq->loc);
}

/*!
 * \brief Get the statements of a scope body.
 * \param stmt The scope body.
 * \return The statements in the body.
 */
inline ffi::Array<Stmt> AsArray(const SeqStmt& stmt) { return stmt->seq; }

/*!
 * \brief Check whether a scope body contains exactly one statement.
 * \param stmt The statement to be checked
 * \return A boolean indicating the result
 */
inline bool IsSingleStmt(const SeqStmt& stmt) { return stmt->size() == 1; }

/******** IterVar ********/

/*!
 * \brief Create a new IterVar for the input For loop, with specified name and type
 * \param loop The loop to be created from
 * \param name The name of the new IterVar
 * \param iter_var_type The type of the new IterVar
 * \return The newly created IterVar
 */
inline IterVar IterVarFromLoop(const For& loop, ffi::String name, IterVarType iter_var_type) {
  return IterVar(Range(loop->begin, loop->end), PrimVar(std::move(name), loop->var.ty()),
                 iter_var_type);
}

/*! \brief Interpret physical and virtual thread bindings for S-TIR scheduling. */
inline support::GPUThreadTag ParseThreadBindingTag(const ffi::String& tag) {
  if (tag.starts_with("vthread")) {
    return {1, -1};
  }
  return support::GPUThreadTag::Parse(tag);
}

/*!
 * \brief Get the thread scope bound to the specific loop
 * \param loop The loop to be inspected
 * \return The thread scope bound to the loop
 */
inline support::GPUThreadTag GetThreadBindingTag(const ForObj* loop) {
  if (tvm::s_tir::GetThreadBinding(loop).has_value()) {
    return ParseThreadBindingTag(tvm::s_tir::GetThreadBinding(loop).value());
  }
  return support::GPUThreadTag{-1, -1};
}

/*!
 * \brief Check if the thread scope is blockIdx
 * \param thread_scope The thread scope to be checked
 * \return True if the thread scope is blockIdx
 */
inline bool IsBlockIdx(const support::GPUThreadTag& thread_scope) {
  return thread_scope.rank == 0;  // The rank of blockIdx is 0
}

/*!
 * \brief Check if the thread scope is threadIdx
 * \param thread_scope The thread scope to be checked
 * \return True if the thread scope is threadIdx
 */
inline bool IsThreadIdx(const support::GPUThreadTag& thread_scope) {
  return thread_scope.rank == 1 && thread_scope.dim_index >= 0;
}

/*!
 * \brief Check if an expression consists of a single variable,
 * or a variable plus/minus an constant integer shift
 * \param expr The expression to be checked
 * \return The single variable in the expression, or std::nullopt if the expression is neither a
 * variable or a constant shift from a variable
 */
inline ffi::Optional<Var> AnalyzeVarWithShift(const PrimExpr& expr,
                                              ffi::Optional<IntImm>* constant) {
  if (auto var = expr.as<PrimVar>()) {
    *constant = std::nullopt;
    return static_cast<Var>(var.value());
  }
  // match: "var + shift"
  if (const auto* add = expr.as<prim::AddObj>()) {
    if (auto var = add->a.as<Var>()) {
      if (auto shift = add->b.as<IntImm>()) {
        *constant = shift;
        return var;
      }
    }
    if (auto shift = add->a.as<IntImm>()) {
      if (auto var = add->b.as<Var>()) {
        *constant = shift;
        return var;
      }
    }
  }
  // match: "var - shift"
  if (const auto* sub = expr.as<prim::SubObj>()) {
    if (auto var = sub->a.as<Var>()) {
      if (auto shift = sub->b.as<IntImm>()) {
        *constant = IntImm(shift.value().ty(), -shift.value()->value);
        return var;
      }
    }
  }
  return std::nullopt;
}

/******** Annotation ********/

/*!
 * \brief Get the annotation on a Block/For
 * \tparam TObjectRef The type of the annotation value
 * \param sref The sref to the block or the for loop
 * \param ann_key The annotation key to be looked up
 * \return std::nullopt if not found; otherwise the annotation value
 */
template <class TObjectRef, class TStmtNode>
inline ffi::Optional<TObjectRef> GetAnn(const TStmtNode* stmt, const ffi::String& ann_key) {
  const ffi::Map<ffi::String, ffi::Any>* annotations = &stmt->annotations;
  for (const auto& ann : *annotations) {
    if (ann.first == ann_key) {
      return ann.second.cast<TObjectRef>();
    }
  }
  return std::nullopt;
}

/*!
 * \brief Get the annotation on a Block/For
 * \tparam TObjectRef The type of the annotation value
 * \param sref The sref to the block or the for loop
 * \param ann_key The annotation key to be looked up
 * \return std::nullopt if not found; otherwise the annotation value
 */
template <class TObjectRef>
inline ffi::Optional<TObjectRef> GetAnn(const StmtSRef& sref, const ffi::String& ann_key) {
  if (const auto* loop = sref->StmtAs<ForObj>()) {
    return GetAnn<TObjectRef, ForObj>(loop, ann_key);
  } else if (const auto* block = sref->StmtAs<SBlockObj>()) {
    return GetAnn<TObjectRef, SBlockObj>(block, ann_key);
  } else {
    TVM_FFI_THROW(TypeError) << "Unknown type of sref: " << sref->stmt->GetTypeKey();
    throw;
  }
}

/*!
 * \brief Check if a Block/For has a specific pair of annotation key and values
 * \param sref The sref to the block or the for loop
 * \param ann_key The annotation key to be checked
 * \param ann_val The annotation value to be checked
 * \return Whether a Block/For has a specific pair of annotation key and values
 */
inline bool HasAnn(const StmtSRef& sref, const ffi::String& ann_key, const ffi::String& ann_val) {
  ffi::Optional<ffi::String> result = GetAnn<ffi::String>(sref, ann_key);
  return result.has_value() && result.value() == ann_val;
}

/*!
 * \brief Check if a Block/For has a specific pair of annotation key and values
 * \param sref The sref to the block or the for loop
 * \param ann_key The annotation key to be checked
 * \param ann_val The boolean annotation value to be checked
 * \return Whether a Block/For has a specific pair of annotation key and values
 */
inline bool HasAnn(const StmtSRef& sref, const ffi::String& ann_key, bool ann_val) {
  ffi::Optional<bool> result = GetAnn<bool>(sref, ann_key);
  return result.has_value() && result.value() == ann_val;
}

/********** Helper Functions for RuleAddRFactor and RuleCrossThreadReduction **********/

/*!
 * \brief Reorder the reduction loops to innermost positions if needed.
 * \param sch The schedule
 * \param block_rv The block where to apply the reorder
 * \param fused_reduce_loop The fusion-generated loop to return.
 * \param num_spatial_loops The number of spatial loops to return.
 * \note Before invoking this helper function, make sure that the block has only spatial and
 *       reduction loop axes.
 */
inline void ReorderAndFuseReductionLoops(const s_tir::Schedule& sch,
                                         const s_tir::SBlockRV& block_rv,
                                         s_tir::LoopRV* fused_reduce_loop,
                                         size_t* num_spatial_loops) {
  ffi::Array<s_tir::LoopRV> loops = sch->GetLoops(block_rv);
  ffi::Array<tirx::StmtSRef> loop_srefs;
  for (const s_tir::LoopRV& loop_rv : loops) {
    loop_srefs.push_back(sch->GetSRef(loop_rv));
  }

  ffi::Array<s_tir::LoopRV> new_order;
  // Step 1. Add spatial loops.
  *num_spatial_loops = 0;
  for (size_t i = 0; i < loops.size(); ++i) {
    if (GetLoopIterType(loop_srefs[i]) == s_tir::kDataPar) {
      new_order.push_back(loops[i]);
      (*num_spatial_loops)++;
    }
  }
  // Step 2. Add reduction loops.
  ffi::Array<s_tir::LoopRV> reduction_loops;
  for (size_t i = 0; i < loops.size(); ++i) {
    if (GetLoopIterType(loop_srefs[i]) == s_tir::kCommReduce) {
      new_order.push_back(loops[i]);
      reduction_loops.push_back(loops[i]);
    }
  }
  // Step 3. Apply reordering if new_order differs from the original order.
  TVM_FFI_ICHECK_EQ(new_order.size(), loops.size());
  for (size_t i = 0; i < loops.size(); ++i) {
    if (!new_order[i].same_as(loops[i])) {
      sch->Reorder(new_order);
      break;
    }
  }
  // Step 4. Fuse all the reduction loops if there are multiple reduction loops.
  TVM_FFI_CHECK(!reduction_loops.empty(), ValueError)
      << "There should be at least one reduction loop";
  if (reduction_loops.size() > 1) {
    *fused_reduce_loop = sch->Fuse(reduction_loops);
  } else {
    *fused_reduce_loop = reduction_loops[0];
  }
}

/******** Helper functions for enum conversion ********/

/*!
 * \brief Convert TensorIndexType to String
 * \param tensor_index_type The TensorIndexType value to convert
 * \return The string representation of TensorIndexType
 */
inline ffi::String TensorIndexType2Str(TensorIndexType tensor_index_type) {
  if (tensor_index_type == TensorIndexType::kRead) {
    return "read";
  } else {
    TVM_FFI_ICHECK(tensor_index_type == TensorIndexType::kWrite);
    return "write";
  }
}

/******** Utilities for retrieving information about blocks ********/

/*! \brief Returns the names of the blocks in the provided module. */
inline std::unordered_set<std::string> GetSBlockNames(const IRModule& mod) {
  struct BlockNameCollector : public s_tir::StmtExprVisitor {
    using s_tir::StmtExprVisitor::Visit_;

    ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
      if (value.as<ExprObj>()) return std::nullopt;
      return s_tir::StmtExprVisitor::Visit(value);
    }

    ffi::Optional<VisitInterrupt> Visit_(const s_tir::SBlockObj* block) override {
      block_names.insert(block->name_hint);
      return StmtExprVisitor::Visit(block->body);
    }
    std::unordered_set<std::string> block_names;
  };

  if (auto function = tirx::FindEntryFunc(mod, nullptr)) {
    auto collector = ffi::make_object<BlockNameCollector>();
    collector->Visit(function->body);
    return collector->block_names;
  }
  return {};
}

/*! \brief Query if the given block name exists in the module associated with the schedule */
inline bool HasBlock(const Schedule& sch, const std::string& block_name) {
  auto block_names = GetSBlockNames(sch->mod());
  return block_names.count(block_name);
}

/******** Utilites for trace application ********/

/*!
 * \brief Translate the input objects using the provided substitution map.
 * \param inputs The input objects.
 * \param rv_map The substitution map for variables.
 * \return The transformed objects.
 */
ffi::Array<Any> TranslateInputRVs(
    const ffi::Array<Any>& inputs,
    const std::unordered_map<const ffi::Object*, const ffi::Object*>& rv_map);

/*!
 * \brief Update the variable substitution map according to the new outputs.
 * \param old_outputs The previous outputs of a schedule instruction.
 * \param new_outputs The new outputs of the same schedule instruction.
 * \param rv_map The substitution map for variables.
 */
void TranslateAddOutputRVs(const ffi::Array<Any>& old_outputs, const ffi::Array<Any>& new_outputs,
                           std::unordered_map<const ffi::Object*, const ffi::Object*>* rv_map);

/*!
 * \brief Counts the number of trace instructions.
 * \param insts The instructions representing a trace.
 * \param remove_postproc If postprocessing instructions are removed.
 * \return Number of instructions.
 */
int GetNumValidInstructions(const ffi::Array<Instruction>& insts, bool remove_postproc);

}  // namespace s_tir
}  // namespace tvm

#endif  // TVM_S_TIR_SCHEDULE_UTILS_H_
