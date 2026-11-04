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
 * \file merge_shared_memory_allocations.cc
 * \brief Each GPU kernel is allowed to have only one dynamic or static shared memory allocation.
 * This pass merges multiple TIR-level dynamic or static shared memory allocations into one
 * allocation.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/transform.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/type.h>

#include <list>
#include <map>
#include <unordered_map>
#include <unordered_set>

#include "../../support/arena.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

using runtime::MemoryScope;

namespace {

ffi::Array<PrimExpr> GetTensorAllocationShape(const TensorVar& tensor) {
  ffi::Array<PrimExpr> alloc_shape = tensor.shape();
  if (tvm::tirx::MatchShapeStridePattern(tensor.ty()).value().get<1>().size()) {
    TVM_FFI_ICHECK_EQ(tensor.shape().size(),
                      tvm::tirx::MatchShapeStridePattern(tensor.ty()).value().get<1>().size());
    for (size_t i = tvm::tirx::MatchShapeStridePattern(tensor.ty()).value().get<1>().size() - 1;
         i > 0; --i) {
      TVM_FFI_ICHECK(sym::Analyzer()->CanProveEqual(
          floormod(tvm::tirx::MatchShapeStridePattern(tensor.ty()).value().get<1>()[i - 1],
                   tvm::tirx::MatchShapeStridePattern(tensor.ty()).value().get<1>()[i]),
          0));
      alloc_shape.Set(i, tvm::tirx::MatchShapeStridePattern(tensor.ty()).value().get<1>()[i - 1] /
                             tvm::tirx::MatchShapeStridePattern(tensor.ty()).value().get<1>()[i]);
    }
  }
  return alloc_shape;
}

ffi::Optional<Var> GetBufferDataVar(const ffi::Any& data) {
  if (auto var = data.as<Var>()) {
    return var;
  }
  if (const auto* call = data.as<CallObj>();
      call && call->op.same_as(tirx::tensor_data_ptr_op()) && call->args.size() == 1) {
    return call->args[0].as<Var>();
  }
  return std::nullopt;
}

}  // namespace

bool IsDynamicSharedMemory(Var buffer_var) {
  static const auto scope_shared_dyn = runtime::MemoryScope::Get("shared.dyn");

  MemoryScope storage_scope =
      (buffer_var->ty.as<TensorTypeObj>() ? buffer_var->ty.as<TensorTypeObj>()->scope
                                          : buffer_var->ty.as_or_throw<PtrType>()->scope);
  return storage_scope == scope_shared_dyn;
}

bool IsStaticSharedMemory(Var buffer_var) {
  static const auto scope_shared = runtime::MemoryScope::Get("shared");

  MemoryScope storage_scope =
      (buffer_var->ty.as<TensorTypeObj>() ? buffer_var->ty.as<TensorTypeObj>()->scope
                                          : buffer_var->ty.as_or_throw<PtrType>()->scope);
  return storage_scope == scope_shared;
}

bool IsDynamicSharedMemory(const TensorVar& tensor) {
  static const auto scope_shared_dyn = runtime::MemoryScope::Get("shared.dyn");

  MemoryScope storage_scope = tensor.scope();
  return storage_scope == scope_shared_dyn;
}

bool IsStaticSharedMemory(const TensorVar& tensor) {
  static const auto scope_shared = runtime::MemoryScope::Get("shared");

  MemoryScope storage_scope = tensor.scope();
  return storage_scope == scope_shared;
}

/*!
 * \brief Compute constant allocation size from buffer's allocation shape.
 * \return Product of extents if all constant, 0 otherwise.
 */
static int64_t ConstantAllocationSize(const ffi::Array<PrimExpr>& extents) {
  int64_t result = 1;
  for (size_t i = 0; i < extents.size(); ++i) {
    if (const IntImmObj* int_size = extents[i].as<IntImmObj>()) {
      auto product = (result * int_size->value).as<int64_t>();
      if (!product.has_value()) return 0;
      result = *product;
    } else {
      return 0;
    }
  }
  return result;
}

/*!
 * \brief collect the mapping from the buffer var to its TensorVar within a subtree
 */
class AllocateCollector : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;
  explicit AllocateCollector(bool is_dynamic) : is_dynamic_(is_dynamic) {}

  ffi::Optional<VisitInterrupt> Visit_(const BindObj* op) final {
    if (const auto* call = op->value.as<CallObj>();
        call && call->op.same_as(tirx::alloc_tensor_op())) {
      return DispatchAllocTensor(op, call);
    }
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> DispatchAllocTensor(const BindObj* op, const CallObj* call) {
    static const auto scope_shared_dyn = runtime::MemoryScope::Get("shared.dyn");
    static const auto scope_shared = runtime::MemoryScope::Get("shared");

    runtime::MemoryScope scope = call->ty_args[0].as_or_throw<TensorType>()->scope;
    MemoryScope storage_scope = scope;
    if (is_dynamic_ && storage_scope == scope_shared_dyn) {
      shmem_allocs_.insert_or_assign(op->var.as_or_throw<TensorVar>().get(),
                                     op->var.as_or_throw<TensorVar>());
    } else if (!is_dynamic_ && storage_scope == scope_shared) {
      shmem_allocs_.insert_or_assign(op->var.as_or_throw<TensorVar>().get(),
                                     op->var.as_or_throw<TensorVar>());
    }
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const RegionStmtObj* op) final {
    if (!op->op.same_as(tirx::launch_thread_op())) has_opaque_region_ = true;
    return StmtExprVisitor::Visit_(op);
  }

  bool has_opaque_region_{false};
  // The mapping from the original buffer var to its TensorVar
  std::unordered_map<const VarObj*, TensorVar> shmem_allocs_;

 private:
  bool is_dynamic_;
};

// Find a linear pattern of storage access
// Used for liveness analysis.
// "linear" means fitting a complex access pattern into an array of StmtEntry
//
// Define "scope" as the body of For/thread_launch/If
// Composite scopes(loop/thread_launch/IfThen) is represented by three StmtEntry:
// before_scope -> scope_body -> after_scope
//
// This pass tries to detect last point that we need to keep memory
// alive under the same scope as AllocTensor.
// The storage need to be kept alive between AllocTensor and last access.
// The free point is only inserted at the same scope of AllocTensor.
//
class SharedMemLinearAccessPatternFinder final : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;
  explicit SharedMemLinearAccessPatternFinder(bool is_dynamic = true) : is_dynamic_(is_dynamic) {}
  /*! \brief record the touch list of statement. */
  struct StmtEntry {
    // The statement
    const ffi::Object* stmt;
    // The index in the linear_seq_ to point to end of the nested scope.
    // This is only set to non-zero if stmt is a nested scope.
    // if offset > 0, means this is the begin, the end entry is current_index + offset
    // if offset < 0, means this is the end, the begin entry is current_index + offset
    int64_t scope_pair_offset{0};
    // The buffer variables this statement touched.
    std::vector<const VarObj*> touched;
  };
  // The scope of each allocation
  struct AllocEntry {
    // the level in the scope stack
    size_t level{0};
    // The buffer object
    TensorVar tensor;
  };

  ffi::Optional<VisitInterrupt> DispatchAllocTensor(const BindObj* op) {
    size_t level = scope_.size();
    const VarObj* buf = op->var.as_or_throw<TensorVar>().get();
    alloc_info_.insert_or_assign(buf, AllocEntry{level, op->var.as_or_throw<TensorVar>()});
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorStoreObj* op) final {
    scope_.push_back(StmtEntry());
    // visit subexpr
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->value));
    for (const auto& index : op->indices) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(index));
    }
    // Add write access.
    const VarObj* buf = ResolveAlias(op->dest.as_or_throw<TensorVar>().get());
    auto it = alloc_info_.find(buf);
    if (it != alloc_info_.end() && it->second.tensor.defined()) {
      TVM_FFI_ICHECK_LT(it->second.level, scope_.size());
      if (IsAppropriateSharedMemory(ffi::GetRef<Var>(buf))) {
        scope_[it->second.level].touched.push_back(buf);
      }
    }
    StmtEntry e = scope_.back();
    scope_.pop_back();
    if (e.touched.size() != 0) {
      e.stmt = op;
      linear_seq_.push_back(e);
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const EvaluateObj* op) final {
    scope_.push_back(StmtEntry());
    // visit subexpr
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
    StmtEntry e = scope_.back();
    scope_.pop_back();
    if (e.touched.size() != 0) {
      e.stmt = op;
      linear_seq_.push_back(e);
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const BindObj* op) final {
    if (const auto* call = op->value.as<CallObj>();
        call && call->op.same_as(tirx::alloc_tensor_op())) {
      return DispatchAllocTensor(op);
    }
    if (const auto* call = op->value.as<CallObj>();
        call && call->op.same_as(tirx::decl_tensor_op())) {
      return DispatchDeclTensor(op, call);
    }
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> DispatchDeclTensor(const BindObj* op, const CallObj* call) {
    if (auto source = GetBufferDataVar(call->args[0])) {
      const VarObj* allocation = ResolveAlias(source.value().get());
      if (alloc_info_.count(allocation)) {
        buffer_alias_sources_[op->var.as_or_throw<TensorVar>().get()] = allocation;
        return std::nullopt;
      }
    }
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadObj* op) final {
    // Add read access.
    for (const auto& index : op->indices) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(index));
    }
    const VarObj* buf = ResolveAlias(op->source.as_or_throw<tvm::tirx::TensorVar>().get());
    auto it = alloc_info_.find(buf);
    if (it != alloc_info_.end() && it->second.tensor.defined()) {
      TVM_FFI_ICHECK_LT(it->second.level, scope_.size())
          << "Load memory in places other than store.";
      if (IsAppropriateSharedMemory(ffi::GetRef<Var>(buf))) {
        scope_[it->second.level].touched.push_back(buf);
      }
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const CallObj* op) final {
    if (op->op.same_as(tirx::address_of_op())) {
      if (const auto* load = op->args[0].as<TensorLoadObj>()) {
        TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(load->source));
        for (const auto& index : load->indices) {
          TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(index));
        }
      } else {
        TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->args[0]));
      }
    } else {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const VarObj* buf) final {
    if (def_region_kind() != kTVMFFIDefRegionKindNone) return StmtExprVisitor::Visit_(buf);
    // Directly reference to the variable count as a read.
    buf = ResolveAlias(buf);
    auto it = alloc_info_.find(buf);
    if (it != alloc_info_.end() && it->second.tensor.defined()) {
      TVM_FFI_ICHECK_LT(it->second.level, scope_.size());
      if (IsAppropriateSharedMemory(ffi::GetRef<Var>(buf))) {
        scope_[it->second.level].touched.push_back(buf);
      }
    }
    return std::nullopt;
  }

  template <typename T>
  ffi::Optional<VisitInterrupt> VisitNewScope(const T* op) {
    scope_.push_back(StmtEntry());
    StmtEntry e;
    e.stmt = op;
    int64_t begin_index = static_cast<int64_t>(linear_seq_.size());
    // before scope.
    linear_seq_.push_back(e);
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
    // after scope.
    e.touched = std::move(scope_.back().touched);
    scope_.pop_back();
    int64_t end_index = static_cast<int64_t>(linear_seq_.size());
    TVM_FFI_ICHECK_GT(end_index, begin_index);
    e.scope_pair_offset = begin_index - end_index;
    linear_seq_.push_back(e);
    // record the pointer to end index.
    TVM_FFI_ICHECK_NE(end_index, 0U);
    linear_seq_[begin_index].scope_pair_offset = end_index - begin_index;
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const RegionStmtObj* op) final {
    // Only record the outer most thread extent.
    if (op->op.same_as(tirx::launch_thread_op()) &&
        std::string(op->args[0].as_or_throw<StringImm>()->value).rfind("vthread", 0) != 0 &&
        !in_thread_env_) {
      in_thread_env_ = true;
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(VisitNewScope(op));
      in_thread_env_ = false;
    } else if (!op->op.same_as(tirx::launch_thread_op()) ||
               std::string(op->args[0].as_or_throw<StringImm>()->value).rfind("vthread", 0) == 0) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(VisitNewScope(op));
    } else {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const IfObj* op) final { return VisitNewScope(op); }

  ffi::Optional<VisitInterrupt> Visit_(const ForObj* op) final { return VisitNewScope(op); }

  ffi::Optional<VisitInterrupt> Visit_(const WhileObj* op) final { return VisitNewScope(op); }

  ffi::Optional<VisitInterrupt> Visit_(const AssertStmtObj* op) final { return VisitNewScope(op); }

  // linearized access sequence.
  std::vector<StmtEntry> linear_seq_;
  // The storage scope of each buffer
  std::unordered_map<const VarObj*, AllocEntry> alloc_info_;
  // Typed views of an allocated shared-memory buffer.
  std::unordered_map<const VarObj*, const VarObj*> buffer_alias_sources_;

 private:
  const VarObj* ResolveAlias(const VarObj* buffer) const {
    while (true) {
      auto it = buffer_alias_sources_.find(buffer);
      if (it == buffer_alias_sources_.end()) break;
      buffer = it->second;
    }
    return buffer;
  }

  // Wrapper function to determine if the shared memory allocation for a variable is appropriate.
  bool IsAppropriateSharedMemory(const Var& var) {
    return is_dynamic_ ? IsDynamicSharedMemory(var) : IsStaticSharedMemory(var);
  }
  // Whether do dynamic analysis.
  bool is_dynamic_{true};
  // Whether already in thread env.
  bool in_thread_env_{false};
  // The scope stack.
  std::vector<StmtEntry> scope_;
};

/*!
 * \brief merge the buffers whose live range has no intersection and rewrite the body
 *
 * Uses a scope-stack design: each launch region (kernel launch) gets its
 * own KernelScope that owns the merged buffer var and all per-launch bookkeeping.
 * This correctly handles Functions with multiple sibling launch regions.
 */
class SharedMemoryRewriter : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  explicit SharedMemoryRewriter(bool is_dynamic = true) : is_dynamic_{is_dynamic} {}

 private:
  using StmtEntry = SharedMemLinearAccessPatternFinder::StmtEntry;

  struct StorageEntry {
    // The constant size of the buffer in bits, only used if it is constant
    uint64_t const_nbits{0};
    // Allocs that shares this entry.
    // The inner vector means a "layer"
    // For example, it we need to allocate C in the memory of A and B:
    // |  A: 4096 bytes |  B: 4096 bytes |
    // |            C: 8192 bytes        |
    // Then the allocs = {{A, B}, {C}}
    std::vector<std::vector<const VarObj*>> allocs;
  };

  // Event entry in liveness analysis
  struct EventEntry {
    // variables we generate
    std::vector<const VarObj*> gen;
    // variables we kill
    std::vector<const VarObj*> kill;
  };

  /*!
   * \brief Per-kernel-launch scope holding all state for one launch region.
   */
  struct KernelScope {
    // The merged buffer var for THIS kernel launch.
    ffi::Optional<TensorVar> merged_tensor;
    // Total byte size of THIS kernel's merged buffer.
    PrimExpr merged_alloc_size{0};
    // Allocations from THIS kernel's subtree.
    std::unordered_map<const VarObj*, TensorVar> shmem_allocs;
    // Per-buffer byte offset into merged_buf_var.
    std::unordered_map<const VarObj*, PrimExpr> buffer_byte_offsets;
    // TensorVar-object remap: original TensorVar -> merged-data-var TensorVar.
    std::unordered_map<const VarObj*, TensorVar> tensor_remap;
    // Typed views whose physical source is one of shmem_allocs.
    std::unordered_map<const VarObj*, const VarObj*> buffer_alias_sources;
    // Remapped buffers in first-use order, for deterministic alias emission.
    std::vector<TensorVar> tensor_remap_order;
    // Has any original alloc in this scope been marked volatile?
    bool has_volatile_alloc{false};
    // Liveness data (event_map, alloc_map, const_free_map, sym_free_list) — all per-scope.
    std::unordered_map<const ffi::Object*, EventEntry> event_map;
    std::multimap<uint64_t, StorageEntry*> const_free_map;
    std::list<StorageEntry*> sym_free_list;
    std::unordered_map<const VarObj*, StorageEntry*> alloc_map;
  };

  /*!
   * \brief Create a fresh merged buffer Var for a new kernel scope.
   *        Same name string is fine — Var identity is by pointer, not name.
   */
  TensorVar MakeMergedTensor(PrimExpr size) {
    static const auto scope_shared = runtime::MemoryScope::Get("shared");
    static const auto scope_shared_dyn = runtime::MemoryScope::Get("shared.dyn");
    return TensorVar(is_dynamic_ ? "buf_dyn_shmem" : "buf_shmem",
                     TensorType({std::move(size)}, PrimType::UInt(8),
                                is_dynamic_ ? scope_shared_dyn : scope_shared));
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj* op, InplaceMode inplace_mode) final {
    if (op->op.same_as(tirx::launch_thread_op()) &&
        std::string(op->args[0].as_or_throw<StringImm>()->value).rfind("vthread", 0) != 0 &&
        !in_thread_env_) {
      in_thread_env_ = true;

      // 1. Push a fresh scope.
      scope_stack_.emplace_back();
      KernelScope& scope = scope_stack_.back();
      // 2. Collect shmem allocs that belong to THIS subtree.
      auto collector = ffi::make_object<AllocateCollector>(is_dynamic_);
      collector->Visit(op->body);
      scope.shmem_allocs = std::move(collector->shmem_allocs_);

      // Per-scope early bail-out: if this launch region has ≤1 shmem
      // allocation, there is nothing to merge. Unknown region operations also
      // prevent moving allocations across their boundaries.
      if (scope.shmem_allocs.size() <= 1 || collector->has_opaque_region_) {
        scope_stack_.pop_back();
        in_thread_env_ = false;
        return StmtExprMutator::Mutate_(op, inplace_mode);
      }

      // 3. Liveness + reuse plan over this subtree only.
      // Run the finder on the full RegionStmt (not just op->body) so that
      // VisitNewScope creates the proper scope pair entry for the launch.
      auto finder = ffi::make_object<SharedMemLinearAccessPatternFinder>(is_dynamic_);
      finder->Visit(ffi::GetRef<Stmt>(op));
      this->LivenessAnalysis(finder->linear_seq_, scope);
      this->PlanMemory(finder->linear_seq_, scope);

      // 4. Compute byte offsets / merged_alloc_size.
      this->ComputeOffsets(scope);
      scope.merged_tensor = MakeMergedTensor(scope.merged_alloc_size);

      // 5. Recursively mutate the body — reads scope_stack_.back() for all rewrites.
      Stmt visited_body = StmtExprMutator::Mutate(ffi::AnyView(op->body), inplace_mode)
                              .ValueOrUnchanged(op->body)
                              .as_or_throw<Stmt>();
      for (const TensorVar& remapped : scope.tensor_remap_order) {
        // The uint8 merged allocation intentionally supplies storage for
        // typed views; target codegen emits the required pointer cast.
        visited_body = SeqStmt(
            {Bind(remapped, Call(remapped.ty(), tirx::decl_tensor_op(),
                                 {scope.merged_tensor.value().data()}, {}, {remapped.ty()})),
             visited_body});
      }

      in_thread_env_ = false;
      // 6. If this scope has no shmem allocs, skip the wrapper.
      if (scope.shmem_allocs.empty()) {
        scope_stack_.pop_back();
        return RegionStmt(op->op, op->args, op->body_params, op->attrs, visited_body,
                          op->result_vars, op->loc);
      }

      // 7. Wrap with the merged-buffer AllocTensor.
      ffi::Map<ffi::String, ffi::Any> annotations;
      if (scope.has_volatile_alloc) {
        annotations.Set(tvm::tirx::attr::kVolatile.name, true);
      }
      Stmt alloc_stmt = Bind(scope.merged_tensor.value(),
                             Call(scope.merged_tensor.value().ty(), tirx::alloc_tensor_op(), {},
                                  DictAttrs(annotations), {scope.merged_tensor.value().ty()}));
      Stmt new_body = SeqStmt({alloc_stmt, visited_body});

      // 8. Pop the scope.
      scope_stack_.pop_back();

      return RegionStmt(op->op, op->args, op->body_params, op->attrs, new_body, op->result_vars,
                        op->loc);
    }
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Stmt> Mutate_AllocTensor(const BindObj* op, const CallObj* call,
                                       InplaceMode inplace_mode) {
    static const auto scope_shared_dyn = runtime::MemoryScope::Get("shared.dyn");
    static const auto scope_shared = runtime::MemoryScope::Get("shared");

    runtime::MemoryScope scope = call->ty_args[0].as_or_throw<TensorType>()->scope;
    MemoryScope storage_scope = scope;
    if (storage_scope == (is_dynamic_ ? scope_shared_dyn : scope_shared)) {
      if (!scope_stack_.empty()) {
        KernelScope& scope = scope_stack_.back();
        if (scope.shmem_allocs.count(op->var.as_or_throw<TensorVar>().get())) {
          if (call->attrs.as<DictAttrsObj>()->dict.count(tvm::tirx::attr::kVolatile.name)) {
            scope.has_volatile_alloc = true;
          }
          return Evaluate(0);
        }
      }
      // Outside any launch scope — leave as-is.
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Stmt> Mutate_(const BindObj* op, InplaceMode inplace_mode) final {
    if (const auto* call = op->value.as<CallObj>();
        call && call->op.same_as(tirx::alloc_tensor_op())) {
      return Mutate_AllocTensor(op, call, inplace_mode);
    }
    if (const auto* call = op->value.as<CallObj>();
        call && call->op.same_as(tirx::decl_tensor_op())) {
      return Mutate_DeclTensor(op, call, inplace_mode);
    }
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Stmt> Mutate_DeclTensor(const BindObj* op, const CallObj* call,
                                      InplaceMode inplace_mode) {
    if (!scope_stack_.empty()) {
      if (auto source = GetBufferDataVar(call->args[0])) {
        KernelScope& scope = scope_stack_.back();
        if (const VarObj* allocation = ResolveAllocation(source.value().get(), scope)) {
          scope.buffer_alias_sources[op->var.as_or_throw<TensorVar>().get()] = allocation;
          return Evaluate(0);
        }
      }
    }

    auto node = StmtExprMutator::Mutate_(op, inplace_mode)
                    .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                    .as_or_throw<Bind>();
    if (auto new_buf = GetUpdatedTensor(node->var.as_or_throw<TensorVar>());
        !new_buf.same_as(node->var)) {
      const auto* new_call = node->value.as<CallObj>();
      return Bind(new_buf,
                  Call(new_buf.ty(), tirx::decl_tensor_op(), {new_call->args[0]}, new_call->attrs,
                       {new_buf.ty()}, new_call->loc),
                  node->loc);
    }
    return node;
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* op, InplaceMode inplace_mode) final {
    auto node = StmtExprMutator::Mutate_(op, inplace_mode)
                    .ValueOrUnchanged(ffi::GetRef<PrimExpr>(op))
                    .as_or_throw<TensorLoad>();
    return VisitTensorAccess(std::move(node));
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreObj* op, InplaceMode inplace_mode) final {
    auto node = StmtExprMutator::Mutate_(op, inplace_mode)
                    .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                    .as_or_throw<TensorStore>();
    return VisitTensorAccess(std::move(node));
  }

  TensorStore VisitTensorAccess(TensorStore node) {
    TensorVar tensor = node->dest.as_or_throw<TensorVar>();
    if (IsAppropriateSharedMemory(tensor) && !scope_stack_.empty() &&
        ResolveAllocation(tensor.get(), scope_stack_.back())) {
      TVM_FFI_ICHECK_EQ(node->indices.size(), 1)
          << "MergeSharedMemoryAllocations expects flat memory tensors, "
          << "and is to be run after "
          << "FlattenTensor";
      ffi::Array<PrimExpr> indices = {node->indices[0] +
                                      this->GetBufferOffset(tensor, tensor.dtype()->dtype)};

      auto writer = node.CopyOnWrite();
      writer->dest = GetUpdatedTensor(tensor);
      writer->indices = indices;
    }

    return node;
  }

  TensorLoad VisitTensorAccess(TensorLoad node) {
    TensorVar tensor = node->source.as_or_throw<tvm::tirx::TensorVar>();
    if (!IsAppropriateSharedMemory(tensor) || scope_stack_.empty() ||
        !ResolveAllocation(tensor.get(), scope_stack_.back())) {
      return node;
    }
    TVM_FFI_ICHECK_EQ(node->indices.size(), 1)
        << "MergeSharedMemoryAllocations expects flat memory tensors, and is to be run after "
           "FlattenTensor";
    ffi::Array<PrimExpr> indices = {node->indices[0] +
                                    this->GetBufferOffset(tensor, tensor.dtype()->dtype)};
    return TensorLoad(GetUpdatedTensor(tensor), indices, node->loc);
  }

  TensorVar GetUpdatedTensor(TensorVar tensor) {
    if (scope_stack_.empty()) return tensor;
    KernelScope& scope = scope_stack_.back();
    if (!ResolveAllocation(tensor.get(), scope)) return tensor;

    auto key = tensor.get();
    auto it = scope.tensor_remap.find(key);
    if (it != scope.tensor_remap.end()) {
      return it->second;
    }

    if (IsAppropriateSharedMemory(tensor)) {
      TVM_FFI_ICHECK_EQ(tensor.shape().size(), 1)
          << "Tensor " << tensor << " has shape " << tensor.shape() << ".  "
          << "MergeSharedMemoryAllocations expects flat memory tensors, "
          << "and is to be run after "
          << "FlattenTensor";
      tensor =
          TensorVar(tensor->name, TensorType(ffi::make_object<TensorTypeObj>(*tensor.ty().get())),
                    tensor->loc);
    }

    scope.tensor_remap.insert_or_assign(key, tensor);
    scope.tensor_remap_order.push_back(tensor);
    return tensor;
  }

  UnchangedOr<Expr> Mutate_(const CallObj* op, InplaceMode inplace_mode) final {
    static const Op ptx_cp_async_op = Op::Get("tirx.s_tir.cp_async_raw");
    if (op->op.same_as(tirx::tensor_data_ptr_op())) {
      TensorVar tensor = op->args[0].as_or_throw<TensorVar>();
      if (!IsAppropriateSharedMemory(tensor) || scope_stack_.empty() ||
          !ResolveAllocation(tensor.get(), scope_stack_.back())) {
        return StmtExprMutator::Mutate_(op, inplace_mode);
      }
      // All typed views use the merged allocation base; physical pointers need
      // the same byte displacement that logical loads/stores receive.
      PrimExpr offset = GetBufferOffset(tensor, PrimType::UInt(8)->dtype);
      return Call(op->ty, tirx::ptr_byte_offset_op(), {GetUpdatedTensor(tensor).data(), offset}, {},
                  {}, op->loc);
    } else if (op->op.same_as(ptx_cp_async_op)) {
      TVM_FFI_ICHECK((op->args.size() == 5U) || (op->args.size() == 6U));
      auto buffer_opt = GetBufferDataVar(op->args[0]);
      if (!buffer_opt.has_value()) {
        return StmtExprMutator::Mutate_(op, inplace_mode);
      }
      Var buffer = buffer_opt.value();
      DLDataType dtype;
      bool is_shared;
      if (buffer->ty.as<TensorTypeObj>()) {
        TensorVar typed_tensor = buffer.as_or_throw<TensorVar>();
        dtype = typed_tensor.dtype()->dtype;
        is_shared = IsAppropriateSharedMemory(typed_tensor);
      } else {
        const auto* ptr_type = buffer->ty.as<PtrTypeObj>();
        TVM_FFI_ICHECK(ptr_type) << "The buffer should be a pointer type.";
        const auto* prim_type = ptr_type->element_type.as<PrimTypeObj>();
        TVM_FFI_ICHECK(prim_type) << "The buffer should be a pointer to a primitive type.";
        dtype = prim_type->dtype;
        is_shared = IsAppropriateSharedMemory(buffer);
      }
      if (!is_shared || scope_stack_.empty() ||
          !ResolveAllocation(buffer.get(), scope_stack_.back())) {
        return StmtExprMutator::Mutate_(op, inplace_mode);
      }
      PrimExpr extra_offset = GetBufferOffset(buffer, dtype);
      PrimExpr offset = Mutate(op->args[1]).ValueOrUnchanged(op->args[1]).as_or_throw<PrimExpr>();
      if (buffer->ty.as<TensorTypeObj>()) {
        Expr merged_data = GetUpdatedTensor(buffer.as_or_throw<TensorVar>()).data();
        ffi::Array<Expr> args = op->args;
        args.Set(0, merged_data);
        args.Set(1, extra_offset + offset);
        return Call(op->ty, op->op, args, op->attrs, {}, op->loc);
      }
      // the dst shared memory is a byte buffer generated by merging shared memory.
      // we need to multiply the offset index by the byte size of the original value dtype, to get
      // the correct offset of merged shared buffer.
      int index_factor = (static_cast<int>(dtype.bits) * static_cast<int>(dtype.lanes) + 7) / 8;
      if (op->args.size() == 5)
        return Call(op->ty.as_or_throw<PrimType>(), op->op,
                    {scope_stack_.back().merged_tensor.value().data(),
                     mul(extra_offset + offset, PrimExpr(index_factor)), op->args[2],
                     op->args[3].as_or_throw<PrimExpr>(), op->args[4].as_or_throw<PrimExpr>()})
            .as_or_throw<PrimExpr>();
      else
        return Call(op->ty.as_or_throw<PrimType>(), op->op,
                    {scope_stack_.back().merged_tensor.value().data(),
                     mul(extra_offset + offset, PrimExpr(index_factor)), op->args[2],
                     op->args[3].as_or_throw<PrimExpr>(), op->args[4].as_or_throw<PrimExpr>(),
                     op->args[5].as_or_throw<PrimExpr>()})
            .as_or_throw<PrimExpr>();
    } else {
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }
  }

  PrimExpr GetBufferOffset(Var buffer_var, DLDataType dtype) {
    TVM_FFI_ICHECK(!scope_stack_.empty());
    KernelScope& scope = scope_stack_.back();
    const VarObj* allocation = ResolveAllocation(buffer_var.get(), scope);
    TVM_FFI_ICHECK(allocation);
    auto it = scope.buffer_byte_offsets.find(allocation);
    TVM_FFI_ICHECK(it != scope.buffer_byte_offsets.end());
    int elem_bytes = (static_cast<int>(dtype.bits) * static_cast<int>(dtype.lanes) + 7) / 8;
    return indexdiv(it->second, elem_bytes);
  }

  const VarObj* ResolveAllocation(const VarObj* buffer, const KernelScope& scope) const {
    while (true) {
      auto it = scope.buffer_alias_sources.find(buffer);
      if (it == scope.buffer_alias_sources.end()) break;
      buffer = it->second;
    }
    return scope.shmem_allocs.count(buffer) ? buffer : nullptr;
  }

  // Wrapper function to determine if the shared memory allocation for a variable is appropriate.
  bool IsAppropriateSharedMemory(const Var& var) {
    return is_dynamic_ ? IsDynamicSharedMemory(var) : IsStaticSharedMemory(var);
  }

  bool IsAppropriateSharedMemory(const TensorVar& tensor) {
    return is_dynamic_ ? IsDynamicSharedMemory(tensor) : IsStaticSharedMemory(tensor);
  }

  /*!
   * \brief Liveness analysis to find gen and kill point of each variable.
   * \param seq the linear pattern of storage access
   * \param scope the kernel scope to write results into
   */
  void LivenessAnalysis(const std::vector<StmtEntry>& seq, KernelScope& scope) {
    // find kill point, do a reverse linear scan.
    std::unordered_set<const VarObj*> touched;
    for (size_t i = seq.size(); i != 0; --i) {
      const StmtEntry& s = seq[i - 1];
      for (const VarObj* buffer : s.touched) {
        if (!touched.count(buffer)) {
          touched.insert(buffer);
          scope.event_map[s.stmt].kill.push_back(buffer);
        }
      }
    }
    // find gen point, do forward scan
    touched.clear();
    for (size_t i = 0; i < seq.size(); ++i) {
      int64_t offset = seq[i].scope_pair_offset;
      if (offset < 0) continue;
      const StmtEntry& s = seq[i + offset];
      for (const VarObj* buffer : s.touched) {
        if (!touched.count(buffer)) {
          touched.insert(buffer);
          scope.event_map[s.stmt].gen.push_back(buffer);
        }
      }
    }
  }

  /*!
   * \brief Memory plan algorithm
   * \param seq the linear pattern of storage access
   * \param scope the kernel scope to write results into
   */
  void PlanMemory(const std::vector<StmtEntry>& seq, KernelScope& scope) {
    std::unordered_set<const VarObj*> inplace_flag;

    for (size_t i = 0; i < seq.size(); ++i) {
      auto it = scope.event_map.find(seq[i].stmt);
      // scope_pair_offset <= 0 means it is either
      // - leaf stmt(offset = 0)
      // - end of scope(offset < 0)
      // In both cases, we need to handle the kill event correctly
      auto is_leaf_alloc = [&](const VarObj* var) {
        return seq[i].scope_pair_offset == 0 &&
               std::find(it->second.gen.begin(), it->second.gen.end(), var) != it->second.gen.end();
      };
      if (it != scope.event_map.end() && seq[i].scope_pair_offset <= 0) {
        for (const VarObj* var : it->second.kill) {
          if (!is_leaf_alloc(var)) this->Free(var, scope);
        }
      }
      // scope_pair_offset >= 0 means it is either
      // - leaf stmt(offset = 0)
      // - beginning of scope(offset < 0)
      // In both cases, we need to handle the gen event correctly
      if (it != scope.event_map.end() && seq[i].scope_pair_offset >= 0) {
        for (const VarObj* var : it->second.gen) {
          TVM_FFI_ICHECK(scope.shmem_allocs.count(var));
          const TensorVar& buf = scope.shmem_allocs.at(var);
          StorageEntry* dst_entry = FindAlloc(buf, scope);
          scope.alloc_map[var] = dst_entry;
        }
      }
      if (it != scope.event_map.end() && seq[i].scope_pair_offset <= 0) {
        for (const VarObj* var : it->second.kill) {
          if (is_leaf_alloc(var)) this->Free(var, scope);
        }
      }
    }
  }

  /*!
   * \brief Compute byte offsets for all entries in the scope after PlanMemory.
   * \param scope the kernel scope whose offset map to fill
   */
  void ComputeOffsets(KernelScope& scope) {
    int max_layer_num = 0;
    std::vector<const StorageEntry*> all_entry;
    for (const auto& e : scope.const_free_map) {
      all_entry.push_back(e.second);
    }
    for (const StorageEntry* e : scope.sym_free_list) {
      all_entry.push_back(e);
    }
    for (const StorageEntry* e : all_entry) {
      max_layer_num = std::max(max_layer_num, static_cast<int>(e->allocs.size()));
    }
    // calculate align for each layer of each storage entry.
    std::vector<int> align(max_layer_num, 0);
    for (const StorageEntry* e : all_entry) {
      for (int i = 0; i < static_cast<int>(e->allocs.size()); i++) {
        for (const VarObj* buffer : e->allocs[i]) {
          const TensorVar& buf = scope.shmem_allocs.at(buffer);
          int elem_bytes = static_cast<int>(buf.dtype().StorageBytes());
          align[i] = std::max(align[i], elem_bytes);
        }
      }
    }
    // calculate offset for each buffer based on the align of each layer
    for (const StorageEntry* e : all_entry) {
      PrimExpr max_inner_offset = 0;
      for (int i = 0; i < static_cast<int>(e->allocs.size()); i++) {
        PrimExpr inner_offset = 0;
        for (const VarObj* buffer : e->allocs[i]) {
          const TensorVar& buf = scope.shmem_allocs.at(buffer);
          ffi::Array<PrimExpr> alloc_shape = GetTensorAllocationShape(buf);
          int elem_bytes = static_cast<int>(buf.dtype().StorageBytes());
          int align_bytes = std::max(align[i], elem_bytes);
          if (buf.ty()->data_alignment > 0) {
            TVM_FFI_ICHECK(buf.ty()->data_alignment % align_bytes == 0)
                << "The alignment of the buffer is not a multiple of the data type size.";
            align_bytes = buf.ty()->data_alignment;
          }
          PrimExpr buffer_bytes = alloc_shape[0] * elem_bytes;
          inner_offset +=
              indexmod(align_bytes - indexmod(scope.merged_alloc_size + inner_offset, align_bytes),
                       align_bytes);
          scope.buffer_byte_offsets.insert_or_assign(buffer,
                                                     scope.merged_alloc_size + inner_offset);
          inner_offset += buffer_bytes;
        }
        max_inner_offset = max(max_inner_offset, inner_offset);
      }
      scope.merged_alloc_size = scope.merged_alloc_size + max_inner_offset;
    }
  }

  /*!
   * \brief Allocate new storage entry.
   * \param buf the buffer object
   * \param const_nbits the size of the allocation in bits
   * \return the new storage entry
   */
  StorageEntry* NewAlloc(const TensorVar& buf, size_t const_nbits) {
    // Re-use not successful, allocate a new buffer.
    StorageEntry* entry = arena_.make<StorageEntry>();
    entry->allocs.push_back({buf.get()});
    entry->const_nbits = const_nbits;
    return entry;
  }

  /*!
   * \brief find the storage entry in the free list for the buffer
   * \param buf the buffer object
   * \param scope the kernel scope whose free lists to search
   * \return the storage entry
   */
  StorageEntry* FindAlloc(const TensorVar& buf, KernelScope& scope) {
    // skip plan for local variable,
    // compiler can do a better job with register allocation.
    const uint64_t match_range = 16;
    ffi::Array<PrimExpr> alloc_shape = GetTensorAllocationShape(buf);
    DLDataType dtype = buf.dtype()->dtype;
    uint64_t op_elem_bits = static_cast<uint64_t>(dtype.bits) * dtype.lanes;
    uint64_t const_nbits =
        static_cast<uint64_t>(ConstantAllocationSize(alloc_shape) * op_elem_bits);
    // disable reuse of small arrays, they will be lowered to registers in LLVM
    // This rules only apply if we are using non special memory
    if (const_nbits > 0 && const_nbits <= 32) {
      return NewAlloc(buf, const_nbits);
    }

    if (const_nbits != 0) {
      // constant allocation.
      auto begin = scope.const_free_map.lower_bound(0);
      auto mid = scope.const_free_map.lower_bound(const_nbits);
      auto end = scope.const_free_map.upper_bound(const_nbits * match_range);
      // Start looking at the buffer that is bigger than the required size first.
      // If we find one, directly allocate the buffer in its location and remove its entry in the
      // free list
      for (auto it = mid; it != end; ++it) {
        StorageEntry* e = it->second;
        e->const_nbits = std::max(const_nbits, e->const_nbits);
        scope.const_free_map.erase(it);
        e->allocs.push_back({buf.get()});
        return e;
      }
      // Then start looking at smaller buffers.
      // Keep collecting the buffer until the sum of their size exceeds the buffer to allocate
      // and finally free all these entry in the free list
      std::vector<std::multimap<uint64_t, StorageEntry*>::iterator> delete_it;
      // the alloc list for the new entry
      std::vector<std::vector<const VarObj*>> reuse_allocs;
      uint64_t mem_ct = 0;
      for (auto it = mid; it != begin;) {
        --it;
        delete_it.push_back(it);
        mem_ct += it->second->const_nbits;
        int n = it->second->allocs.size();
        if (n > static_cast<int>(reuse_allocs.size())) {
          reuse_allocs.resize(n, {});
        }
        for (int i = 0; i < n; i++) {
          for (const VarObj* alloc : it->second->allocs[i]) {
            reuse_allocs[i].push_back(alloc);
          }
        }
        if (mem_ct >= const_nbits) {
          break;
        }
      }
      reuse_allocs.push_back({buf.get()});
      if (mem_ct != 0) {
        StorageEntry* e = arena_.make<StorageEntry>();
        e->const_nbits = std::max(const_nbits, mem_ct);
        e->allocs = reuse_allocs;
        for (auto it : delete_it) {
          scope.const_free_map.erase(it);
        }
        return e;
      }
    } else {
      // if its symbolic allocation, just arbitrarily choose one entry to fit in because we don't
      // know its actual size
      for (auto it = scope.sym_free_list.begin(); it != scope.sym_free_list.end(); ++it) {
        StorageEntry* e = *it;
        scope.sym_free_list.erase(it);
        return e;
      }
    }
    return NewAlloc(buf, const_nbits);
  }

  /*!
   * \brief add the storage entry to the buffer var into the free list.
   * \param var the buffer var
   * \param scope the kernel scope whose free lists to update
   */
  void Free(const VarObj* var, KernelScope& scope) {
    auto it = scope.alloc_map.find(var);
    TVM_FFI_ICHECK(it != scope.alloc_map.end());
    StorageEntry* e = it->second;
    TVM_FFI_ICHECK_NE(e->allocs.size(), 0U);

    // disable reuse of small arrays
    if (e->const_nbits > 0 && e->const_nbits <= 32) return;

    // normal free.
    if (e->const_nbits != 0) {
      scope.const_free_map.insert({e->const_nbits, e});
    } else {
      scope.sym_free_list.push_back(e);
    }
  }

  // Whether enable dynamic analysis.
  bool is_dynamic_{true};
  // Whether already inside a hardware launch (outermost only).
  bool in_thread_env_{false};
  // Stack of per-kernel-launch scopes. Pushed on launch entry, popped on exit.
  std::vector<KernelScope> scope_stack_;
  /*! \brief allocator of all the StorageEntry (shared across all scopes) */
  support::Arena arena_;
};

Stmt MergeSharedMemoryAllocations(Stmt stmt, bool merge_static_smem) {
  // Function-level early-out: skip the rewriter entirely if the Function
  // has ≤1 dynamic shared-memory allocation (nothing to merge).
  {
    auto dyn_probe = ffi::make_object<AllocateCollector>(/*is_dynamic=*/true);
    dyn_probe->Visit(stmt);
    if (dyn_probe->shmem_allocs_.size() > 1) {
      auto dyn_rewriter = ffi::make_object<SharedMemoryRewriter>(/*is_dynamic=*/true);
      stmt = dyn_rewriter->Mutate(stmt, InplaceMode::kAllow).ValueOrUnchanged(std::move(stmt));
    }
  }
  if (merge_static_smem) {
    // Similarly skip the static rewriter if there is ≤1 static shmem alloc.
    auto static_probe = ffi::make_object<AllocateCollector>(/*is_dynamic=*/false);
    static_probe->Visit(stmt);
    if (static_probe->shmem_allocs_.size() > 1) {
      auto static_rewriter = ffi::make_object<SharedMemoryRewriter>(/*is_dynamic=*/false);
      stmt = static_rewriter->Mutate(stmt, InplaceMode::kAllow).ValueOrUnchanged(std::move(stmt));
    }
  }
  return stmt;
}

namespace transform {

Pass MergeSharedMemoryAllocations() {
  auto pass_func = [](Function f, IRModule m, PassContext ctx) {
    if (!f->body.has_value()) return f;
    bool merge_static_smem = ctx->GetConfig<bool>("tirx.merge_static_smem").value_or(false);
    auto* n = f.CopyOnWrite();
    n->body = s_tir::MergeSharedMemoryAllocations(std::move(n->body).value(), merge_static_smem);
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "s_tir.MergeSharedMemoryAllocations");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.MergeSharedMemoryAllocations",
                        static_cast<Pass (*)()>(MergeSharedMemoryAllocations));
}

}  // namespace transform
}  // namespace s_tir
}  // namespace tvm
