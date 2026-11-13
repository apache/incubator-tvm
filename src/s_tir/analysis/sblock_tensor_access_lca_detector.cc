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
 * \file s_tir/analysis/sblock_tensor_access_lca_detector.cc
 * \brief Detect the lowest common ancestor(LCA) of tensor access
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/support/gpu_thread_tag.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op/region.h>

#include "../../support/arena.h"
#include "../memory_scope.h"

namespace tvm {
namespace tirx {

/*!
 * \brief Detect the lowest common ancestor(LCA) position of tensor access.
 * \note
 * - Consider block, loop, and region bodies to be LCA scopes.
 * - In the LCA locator, we are aware of the tensor scope and CUDA hierarchy so that any tensor in
 * global memory will have its tensor access LCA outside all launch sites of `blockIdx`, in order to
 * prevent conflicts between tensor memory scopes and CUDA hierarchy.
 */
class LCADetector : public s_tir::StmtExprVisitor {
 public:
  using s_tir::StmtExprVisitor::Visit_;

  static ffi::Map<TensorVar, ffi::Optional<Stmt>> Detect(const Function& func) {
    auto detector = ffi::make_object<LCADetector>();
    for (const Var& param : func->params) {
      if (auto tensor = param.as<TensorVar>()) {
        detector->tensor_var_map_.emplace(tensor.value().get(), tensor.value().get());
      }
    }

    // The root node must be explicitly present in the list of
    // ancestor_scopes_.  We cannot use nullptr to represent the root
    // node, as that is also used to represent a scope that hasn't
    // been observed before.
    ScopeInfo root(nullptr, nullptr, 0);
    detector->ancestor_scopes_.push_back(&root);

    detector->Visit(func->body);
    detector->UpdateWithBlockidx();

    // Prepare the return
    ffi::Map<TensorVar, ffi::Optional<Stmt>> tensor_lca;
    for (const auto& kv : detector->tensor_lca_) {
      TensorVar tensor = ffi::GetRef<Var>(kv.first).as_or_throw<TensorVar>();
      const ffi::Optional<Stmt> stmt = kv.second && kv.second->stmt
                                           ? ffi::Optional<Stmt>(ffi::GetRef<Stmt>(kv.second->stmt))
                                           : std::nullopt;
      tensor_lca.Set(tensor, stmt);
    }
    return tensor_lca;
  }

 private:
  /*!
   * \brief The AST node information for querying LCA.
   * \note Block, loop, and region bodies provide lexical allocation scopes.
   */
  struct ScopeInfo {
    // The parent scope info
    const ScopeInfo* parent_scope_info;
    // The parent scope stmt node
    const StmtObj* stmt;
    // The scope depth in the AST
    int depth;
    ScopeInfo(const ScopeInfo* parent_info, const StmtObj* stmt, int depth)
        : parent_scope_info(parent_info), stmt(stmt), depth(depth) {}
  };

  ffi::Optional<VisitInterrupt> Visit_(const ForObj* op) final {
    int n = ancestor_scopes_.size();
    const ScopeInfo* parent_scope = ancestor_scopes_.back();
    auto* current_scope = arena_.make<ScopeInfo>(parent_scope, op, n);

    if (tvm::s_tir::GetThreadBinding(op).has_value()) {
      const ffi::String tag = tvm::s_tir::GetThreadBinding(op).value();
      if (!tag.starts_with("vthread") && support::GPUThreadTag::Parse(tag).rank == 0) {
        blockidx_scopes_.push_back(current_scope);
      }
    }

    ancestor_scopes_.push_back(current_scope);
    loop_scope_map_.insert({op->var.get(), current_scope});
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(s_tir::StmtExprVisitor::Visit_(op));
    ancestor_scopes_.pop_back();
    loop_scope_map_.erase(op->var.get());
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const s_tir::SBlockRealizeObj* op) final {
    const s_tir::SBlockObj* block = op->block.get();
    int n = ancestor_scopes_.size();
    for (const TensorVar& tensor : block->alloc_tensors) {
      tensor_var_map_.emplace(tensor.get(), tensor.get());
    }

    const ScopeInfo* parent_scope = ancestor_scopes_.back();
    auto* current_scope = arena_.make<ScopeInfo>(parent_scope, block, n);

    ancestor_scopes_.push_back(current_scope);

    // For each accessed tensor of the block
    // If it accesses the opaque block iter vars, update the tensor's lca to
    // the lowest inclusive stmt position, which should dominate all loops
    // related to the accessed opaque block iter vars.
    // If it is the reduction block write tensor, update the tensor's lca to
    // dominate all reduction iter var related loops.
    UpdateDominateScopeOfNonDataParIter(op);

    // Update match_tensors
    for (const s_tir::MatchTensorRegion& match_tensor : block->match_tensors) {
      UpdateTensorLCA(match_tensor->source->source.as_or_throw<tvm::tirx::TensorVar>().get(),
                      ancestor_scopes_.back());
      match_tensors_.insert(match_tensor->tensor.get());
    }

    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(s_tir::StmtExprVisitor::Visit_(op));
    ancestor_scopes_.pop_back();
    return std::nullopt;
  }

  void UpdateDominateScopeOfNonDataParIter(const s_tir::SBlockRealizeObj* block_realize) {
    // map iter var to the scope which dominate all loop carried dependencies.
    std::unordered_map<const VarObj*, const ScopeInfo*> opaque_var_scope;
    // maintain highest scope which dominate all reduce loop iters. null denotes non-reduce block.
    const ScopeInfo* highest_reduce_scope = nullptr;

    // function to collect `itervar_to_dom_scope`, the result scope for each block
    // iter var should be above all loop scopes the opaque iter var binding relates to.
    auto do_collect_itervar_scope = [this](const s_tir::IterVar& itervar,
                                           const PrimExpr& binding) -> const ScopeInfo* {
      const ScopeInfo* highest_scope = nullptr;
      auto walk_fn = [this, &highest_scope](const Var& var) -> ffi::Expected<ffi::WalkResult> {
        if (auto prim_var = var.as<PrimVar>()) {
          const VarObj* loop_var = prim_var.value().get();
          auto it = loop_scope_map_.find(loop_var);
          if (it == loop_scope_map_.end()) {
            return ffi::WalkResult::Advance();
          }
          const ScopeInfo* scope = it->second->parent_scope_info;
          if (highest_scope == nullptr) {
            highest_scope = scope;
          } else if (scope->depth < highest_scope->depth) {
            highest_scope = scope;
          }
        }
        return ffi::WalkResult::Advance();
      };
      ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(binding, walk_fn);
      return highest_scope;
    };

    // collect non-data-parallel block iteration's dominate scope.
    // for reduction iter type, we maintain the highest dominate scope for all reduce iters.
    // for other iter type, we maintain the dict for each individual iter.
    const s_tir::SBlock& block = block_realize->block;
    bool is_reduce_block = false;
    for (size_t i = 0; i < block_realize->iter_values.size(); ++i) {
      const s_tir::IterVar& iter_var = block->iter_vars[i];
      if (iter_var->iter_type != s_tir::IterVarType::kDataPar) {
        const auto* scope = do_collect_itervar_scope(iter_var, block_realize->iter_values[i]);
        if (scope == nullptr) continue;
        if (iter_var->iter_type == s_tir::IterVarType::kCommReduce) {
          is_reduce_block = true;
          if (highest_reduce_scope == nullptr || scope->depth < highest_reduce_scope->depth) {
            highest_reduce_scope = scope;
          }
        } else {
          opaque_var_scope[iter_var->var.get()] = scope;
          for (const auto& write : block->writes) {
            UpdateTensorLCA(write->source.as_or_throw<tvm::tirx::TensorVar>().get(), scope);
          }
        }
      }
    }

    // function to update lca scope of the tensor with loop carried dependent tensor accesses.
    // the result scope should be above all loop scopes the accessed opaque block iter vars
    // relate to, which is record in `itervar_to_dom_scope`.
    auto do_update = [this, &opaque_var_scope, highest_reduce_scope](const TensorRegion& region,
                                                                     bool is_reduce_write = false) {
      const TensorVar& tensor = region->source.as_or_throw<tvm::tirx::TensorVar>();
      const ScopeInfo* scope = ancestor_scopes_.back();

      auto handle_itervar = [&opaque_var_scope,
                             &scope](const Var& var) -> ffi::Expected<ffi::WalkResult> {
        if (auto prim_var = var.as<PrimVar>()) {
          const VarObj* iter_var = prim_var.value().get();
          auto dom_scope_it = opaque_var_scope.find(iter_var);
          if (dom_scope_it == opaque_var_scope.end()) {
            return ffi::WalkResult::Advance();
          }
          // find the highest loop scope the accessed tensor index has
          // loop carried dependencies to (via opaque iter var binding).
          if (dom_scope_it->second->depth < scope->depth) {
            scope = dom_scope_it->second;
          }
        }
        return ffi::WalkResult::Advance();
      };

      // visit region min and max to find the lowest legal lca scope
      for (const Range& range : region->region) {
        ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(range->begin, handle_itervar);
        ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(range->begin + range->extent - 1,
                                                        handle_itervar);
      }

      // the scope should be above `highest_reduce_scope` for reduce output tensor.
      if (is_reduce_write && highest_reduce_scope != nullptr &&
          scope->depth > highest_reduce_scope->depth) {
        scope = highest_reduce_scope;
      }
      UpdateTensorLCA(tensor.get(), scope);
    };

    if (!opaque_var_scope.empty()) {
      for (const auto& read : block->reads) {
        do_update(read);
      }
      for (const auto& write : block->writes) {
        do_update(write, /*is_reduce_write=*/is_reduce_block);
      }
    } else if (is_reduce_block && highest_reduce_scope != nullptr) {
      for (const auto& write : block->writes) {
        do_update(write, /*is_reduce_write=*/true);
      }
    }
  }

  ffi::Optional<VisitInterrupt> Visit_(const RegionStmtObj* op) final {
    // Operands belong to the enclosing scope, before the local definitions.
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->args));
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->attrs));
    const ScopeInfo* parent_scope = ancestor_scopes_.back();
    auto* current_scope = arena_.make<ScopeInfo>(parent_scope, op, ancestor_scopes_.size());
    if (op->op.same_as(tirx::launch_thread_op()) &&
        !op->args[0].as_or_throw<StringImm>()->value.starts_with("vthread") &&
        support::GPUThreadTag::Parse(op->args[0].as_or_throw<StringImm>()->value).rank == 0) {
      blockidx_scopes_.push_back(parent_scope);
    }
    ancestor_scopes_.push_back(current_scope);
    for (const Var& var : op->body_params) loop_scope_map_.emplace(var.get(), current_scope);
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->body));
    for (const Var& var : op->body_params) loop_scope_map_.erase(var.get());
    ancestor_scopes_.pop_back();
    return std::nullopt;
  }

  // Declared regions carry bounds, not opaque runtime accesses.
  ffi::Optional<VisitInterrupt> Visit_(const TensorRegionObj* op) final {
    if (!op->source.as<TensorVar>()) return s_tir::StmtExprVisitor::Visit_(op);
    for (const Range& range : op->region) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(range->begin));
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(range->extent));
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadObj* op) final {
    UpdateTensorLCA(op->source.as_or_throw<tvm::tirx::TensorVar>().get(), ancestor_scopes_.back());
    for (const auto& index : op->indices) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(index));
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorStoreObj* op) final {
    UpdateTensorLCA(op->dest.as_or_throw<TensorVar>().get(), ancestor_scopes_.back());
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->value));
    for (const auto& index : op->indices) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(index));
    }
    return std::nullopt;
  }

  // Works for Load/Store and opaque access.
  ffi::Optional<VisitInterrupt> Visit_(const VarObj* op) final {
    if (def_region_kind() != kTVMFFIDefRegionKindNone) return StmtExprVisitor::Visit_(op);
    VisitTensorVar(op);
    return std::nullopt;
  }

  void VisitTensorVar(const VarObj* op) {
    auto it = tensor_var_map_.find(op);
    if (it != tensor_var_map_.end()) {
      UpdateTensorLCA(it->second, ancestor_scopes_.back());
    }
  }

  void UpdateTensorLCA(const VarObj* tensor, const ScopeInfo* scope) {
    tensor_var_map_.emplace(tensor, tensor);
    if (match_tensors_.find(tensor) == match_tensors_.end()) {
      // Ingore tensor created by block match_tensor
      const ScopeInfo*& lca = tensor_lca_[tensor];
      lca = LowestCommonAncestor(lca, scope);
    }
  }

  void UpdateWithBlockidx() {
    for (const auto& it : tensor_lca_) {
      const runtime::MemoryScope& scope =
          ffi::GetRef<Var>(it.first).as_or_throw<TensorVar>().scope();
      if (s_tir::IsGlobalMemoryScope(scope)) {
        const ScopeInfo*& lca = tensor_lca_[it.first];
        for (const ScopeInfo* blockidx_scope : blockidx_scopes_) {
          lca = LowestCommonAncestor(lca, blockidx_scope);
        }
      }
    }
  }

  static const ScopeInfo* LowestCommonAncestor(const ScopeInfo* lhs, const ScopeInfo* rhs) {
    if (lhs == nullptr) return rhs;
    if (rhs == nullptr) return lhs;
    while (lhs->parent_scope_info != nullptr &&  //
           rhs->parent_scope_info != nullptr &&  //
           lhs != rhs) {
      if (lhs->depth == rhs->depth) {
        lhs = lhs->parent_scope_info;
        rhs = rhs->parent_scope_info;
      } else if (lhs->depth < rhs->depth) {
        rhs = rhs->parent_scope_info;
      } else {
        lhs = lhs->parent_scope_info;
      }
    }
    if (lhs->parent_scope_info == nullptr) {
      return lhs;
    }
    if (rhs->parent_scope_info == nullptr) {
      return rhs;
    }
    TVM_FFI_ICHECK(lhs == rhs);
    return lhs;
  }

  /*! \brief The ancestor scope stacks info (s_tir::SBlock and For).  The
   *  first element is initialized in LCADetector::Detect to represent
   *  the root scope.
   */
  std::vector<const ScopeInfo*> ancestor_scopes_ = {};
  /*! \brief The map from TensorVar to its LCA ForObj/BlockNode. */
  std::unordered_map<const VarObj*, const ScopeInfo*> tensor_lca_ = {};
  /*! \brief The map from TensorVar data to the TensorVar. */
  std::unordered_map<const VarObj*, const VarObj*> tensor_var_map_ = {};
  /*! \brief The match tensors inside blocks. */
  std::unordered_set<const VarObj*> match_tensors_ = {};
  /*! \brief The ForNodes/BlockNodes which contain immediate `blockIdx` launch. */
  std::vector<const ScopeInfo*> blockidx_scopes_ = {};
  /*! \brief The map from loop var to the corresponding scope. */
  std::unordered_map<const VarObj*, const ScopeInfo*> loop_scope_map_ = {};
  /*! \brief Internal arena. */
  support::Arena arena_;
};

ffi::Map<TensorVar, ffi::Optional<Stmt>> DetectTensorAccessLCA(const Function& func) {
  return LCADetector::Detect(func);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.analysis.detect_tensor_access_lca", DetectTensorAccessLCA);
}
}  // namespace tirx
}  // namespace tvm
