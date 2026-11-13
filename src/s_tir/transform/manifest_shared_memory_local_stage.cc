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
 * \file manifest_shared_memroy_local_stage.cc
 * \brief Add the explicit local stage for the shared memory access on GPU.
 *
 * This pass finds the cache_read stage on the shared memory, and create another intermediate stage
 * to store the data into local memory first, and then copy the data from local memory to the shared
 * memory. This is similar to the schedule primitive cache_read, but it bypasses the limitation
 * of requiring tensor access to be contiguous in each dimension.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/transform.h>
#include <tvm/sym/analyzer.h>

#include <unordered_set>

#include "../schedule/transform.h"
#include "../tensor_region.h"
#include "tvm/tirx/stmt.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

/*! \brief Rewriter for the block storing to the target tensor. Create an intermediate cache stage
 * to store the result. Rewrite the original block to load from the intermediate tensor.
 */
class IntermediateStageRewriter {
 public:
  explicit IntermediateStageRewriter(const std::vector<Stmt>& ancestor_loop_or_blocks)
      : ancestor_loop_or_blocks_(ancestor_loop_or_blocks) {}

  std::tuple<TensorVar, TensorVar, SBlock, Stmt> Rewrite(const SBlockObj* block) {
    static const auto scope_shared = runtime::MemoryScope::Get("shared");
    static const auto scope_shared_dyn = runtime::MemoryScope::Get("shared.dyn");

    const TensorStoreObj* store =
        block->body->size() == 1 ? block->body->seq[0].as<TensorStoreObj>() : nullptr;
    TVM_FFI_CHECK(
        store != nullptr && (store->dest.as_or_throw<TensorVar>().scope() == scope_shared ||
                             store->dest.as_or_throw<TensorVar>().scope() == scope_shared_dyn),
        ValueError)
        << "Expect the body of the block to be TensorStore to shared memory.";

    const TensorVar& target_tensor = store->dest.as_or_throw<TensorVar>();

    // Step 0: Collect relaxed loops
    std::vector<const ForObj*> relaxed_loops = CollectRelaxedOuterLoops(block, target_tensor);

    // Step 1: Create tensor for the local stage
    auto [new_tensor, tensor_indices] = CreateIntermediateTensor(relaxed_loops, target_tensor);

    // Step 2: Create the local stage block
    Stmt local_stage = MakeLocalStage(block, new_tensor, tensor_indices, relaxed_loops, store);

    // Step 3: Create TensorLoad from the intermediate tensor
    TensorLoad new_tensor_load = TensorLoad(new_tensor, tensor_indices);
    TensorStore new_tensor_store = ffi::GetRef<TensorStore>(store);
    new_tensor_store.CopyOnWrite()->value = new_tensor_load;
    SBlock new_block = ffi::GetRef<SBlock>(block);
    new_block.CopyOnWrite()->body = std::move(new_tensor_store);

    return {target_tensor, new_tensor, new_block, local_stage};
  }

 private:
  /*! \brief Collect relaxed outer loops from innermost to outermost */
  std::vector<const ForObj*> CollectRelaxedOuterLoops(const SBlockObj* block,
                                                      const TensorVar& target_tensor) {
    std::vector<const ForObj*> relaxed_loops;
    for (int n = static_cast<int>(ancestor_loop_or_blocks_.size()) - 1, i = n - 1; i >= 0; --i) {
      const Stmt& ancestor = ancestor_loop_or_blocks_[i];
      if (const ForObj* ancestor_loop = ancestor.as<ForObj>()) {
        TVM_FFI_CHECK(
            ancestor_loop->kind == ForKind::kDefault || ancestor_loop->kind == ForKind::kVectorized,
            ValueError)
            << "Expect the ancestor loops to be serial or vectorized, got " << ancestor_loop->kind;
        relaxed_loops.push_back(ancestor.as<ForObj>());

        if (i < n - 1) {
          TVM_FFI_CHECK(ancestor_loop->body->size() == 1 &&
                            ancestor_loop->body->seq[0].same_as(ancestor_loop_or_blocks_[i + 1]),
                        ValueError)
              << "Expect the ancestor loops to have a single child.";
        } else {
          const SBlockRealizeObj* block_realize =
              ancestor_loop->body->size() == 1 ? ancestor_loop->body->seq[0].as<SBlockRealizeObj>()
                                               : nullptr;
          TVM_FFI_ICHECK(block_realize != nullptr);
          TVM_FFI_CHECK(block_realize != nullptr && block_realize->block.get() == block, ValueError)
              << "Expect the ancestor loops to have a single child.";
        }
      } else {
        const SBlockRealizeObj* ancestor_block_realize = ancestor.as<SBlockRealizeObj>();
        TVM_FFI_ICHECK(ancestor_block_realize != nullptr);
        const SBlockObj* ancestor_block = ancestor_block_realize->block.get();
        auto it = std::find_if(
            ancestor_block->alloc_tensors.begin(), ancestor_block->alloc_tensors.end(),
            [&target_tensor](const TensorVar& tensor) { return tensor.same_as(target_tensor); });
        TVM_FFI_CHECK(it != ancestor_block->alloc_tensors.end(), ValueError)
            << "Expect the shared memory allocation to be in the parent block.";
        break;
      }
    }
    return relaxed_loops;
  }

  /*! \brief Create the intermediate stage. */
  Stmt MakeLocalStage(const SBlockObj* block, const TensorVar& new_tensor,
                      ffi::Array<PrimExpr> local_stage_indices,
                      std::vector<const ForObj*> relaxed_loops, const TensorStoreObj* store) {
    // Step 0: Create the body of the local stage, which is TensorStore to the intermediate tensor.
    Stmt local_stage = TensorStore(new_tensor, local_stage_indices, store->value);

    // Step 1: Make block and block realize
    TensorRegion write_tensor_region =
        s_tir::TensorRegionFromPoint(new_tensor, local_stage_indices);
    local_stage =
        SBlock(/*iter_vars=*/{}, /*reads=*/block->reads, /*writes=*/{write_tensor_region}, "",
               /*body=*/std::move(local_stage));
    local_stage = SBlockRealize(
        /*iter_values=*/{},
        /*predicate=*/ancestor_loop_or_blocks_.back().as<SBlockRealizeObj>()->predicate,
        local_stage.as_or_throw<SBlock>());

    // Step 2: Add outer loops
    ffi::Map<Var, Var> subst_map;
    for (const ForObj* relaxed_loop : relaxed_loops) {
      ffi::ObjectPtr<ForObj> for_node = ffi::make_object<ForObj>(*relaxed_loop);
      for_node->var = for_node->var.CopyWithSuffix("");
      for_node->body = std::move(local_stage);
      local_stage = For(for_node);
      subst_map.Set(relaxed_loop->var, for_node->var);
    }
    auto f_substitute = [&subst_map](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
      if (auto repl = subst_map.Get(var)) return ffi::Any(*std::move(repl));
      return ffi::Unchanged();
    };
    local_stage = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(local_stage, f_substitute)
                      .as_or_throw<Stmt>();
    return local_stage;
  }

  /*! \brief Create the intermediate tensor with the extents of the relaxed outer loops. */
  std::pair<TensorVar, ffi::Array<PrimExpr>> CreateIntermediateTensor(
      const std::vector<const ForObj*> relaxed_loops, const TensorVar& tensor) const {
    ffi::Array<PrimExpr> tensor_indices;
    ffi::Array<PrimExpr> new_tensor_shape;

    // Create the intermediate tensor for the local stage. The shape of the new tensor is the
    // extents of the relaxed outer loops.

    for (auto it = relaxed_loops.rbegin(); it != relaxed_loops.rend(); ++it) {
      const ForObj* relaxed_loop = *it;
      tensor_indices.push_back(relaxed_loop->begin + relaxed_loop->var);
      new_tensor_shape.push_back(
          Range::GetExtentFromBeginEnd(relaxed_loop->begin, relaxed_loop->end));
    }
    static const auto scope_local = runtime::MemoryScope::Get("local");
    TensorVar new_tensor = WithScope(tensor, scope_local);
    ffi::ObjectPtr<TensorTypeObj> type = ffi::make_object<TensorTypeObj>(*new_tensor.ty().get());
    type->shape = new_tensor_shape;
    new_tensor = TensorVar(new_tensor->name, TensorType(std::move(type)), new_tensor->loc);
    return {new_tensor, tensor_indices};
  }

  const std::vector<Stmt>& ancestor_loop_or_blocks_;
};

class SharedMemoryLocalStageInserter : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) override {
    if (value.as<ExprObj>()) return ffi::Unchanged();
    return StmtExprMutator::Mutate(value, inplace_mode);
  }

  UnchangedOr<Stmt> Mutate_(const ForObj* op, InplaceMode inplace_mode) final {
    ancestor_loop_or_blocks_.push_back(ffi::GetRef<Stmt>(op));
    Stmt new_stmt = StmtExprMutator::Mutate_(op, InplaceMode::kDisallow)
                        .ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    ancestor_loop_or_blocks_.pop_back();
    return new_stmt;
  }

  UnchangedOr<Stmt> Mutate_(const SBlockRealizeObj* op, InplaceMode inplace_mode) final {
    ancestor_loop_or_blocks_.push_back(ffi::GetRef<Stmt>(op));
    Stmt new_stmt = StmtExprMutator::Mutate_(op, InplaceMode::kDisallow)
                        .ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    ancestor_loop_or_blocks_.pop_back();
    return new_stmt;
  }

  UnchangedOr<Stmt> Mutate_(const SBlockObj* op, InplaceMode inplace_mode) final {
    if (op->annotations.count(s_tir::attr::kManifestSharedMemoryLocalStage.name)) {
      // Rewrite the shared memory access to load from the intermediate tensor.
      // The annotated block must be a leaf block (will be checked during rewriting). No need to
      // visit its body recursively.

      IntermediateStageRewriter rewriter(ancestor_loop_or_blocks_);
      auto [target_tensor, new_tensor, new_block, local_stage] = rewriter.Rewrite(op);
      tensor_remap_.Set(target_tensor, new_tensor);

      new_block.CopyOnWrite()->annotations.erase(s_tir::attr::kManifestSharedMemoryLocalStage.name);
      tensor_local_stage_.Set(target_tensor, local_stage);
      target_tensors_.push_back(target_tensor);

      return new_block;
    }

    std::unordered_set<TensorVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> allocated_tensors(
        op->alloc_tensors.begin(), op->alloc_tensors.end());

    // Visit children and insert local stages (if any) to the proper location.
    ffi::Array<TensorVar> new_alloc_tensors;
    ffi::Array<Stmt> new_seq;

    // Helper function to check if the subtree (body of the block) contains any target tensors.
    // If so, the allocated intermediate tensor and the local stage should be lifted to the current
    // block.
    auto f_check_subtree = [&](int start, int end) {
      for (int i = start; i < end; ++i) {
        const TensorVar& tensor = target_tensors_[i];
        if (allocated_tensors.count(tensor)) {
          new_seq.push_back(tensor_local_stage_.at(tensor));
          new_alloc_tensors.push_back(tensor_remap_.at(tensor));
        }
      }
    };

    // Visit each body statement and insert its local stage immediately before it.
    bool changed = false;
    for (const Stmt& stmt : op->body->seq) {
      int subtree_start = target_tensors_.size();
      auto result = Mutate(stmt);
      bool unchanged = result.UnchangedOrSameAs(stmt);
      Stmt new_stmt = std::move(result).ValueOrUnchanged(stmt);
      int subtree_end = target_tensors_.size();
      f_check_subtree(subtree_start, subtree_end);
      new_seq.push_back(new_stmt);
      changed |= !unchanged;
    }
    if (!changed && new_alloc_tensors.empty()) {
      return ffi::Unchanged();
    }

    SBlock new_block = ffi::GetRef<SBlock>(op);
    SBlockObj* new_block_node = new_block.CopyOnWrite();
    // Add new tensor allocations if any.
    if (new_alloc_tensors.size() > 0) {
      new_block_node->alloc_tensors = Concat(new_block_node->alloc_tensors, new_alloc_tensors);
    }
    new_block_node->body = SeqStmt(new_seq, op->body->loc);
    return new_block;
  }

  std::vector<Stmt> ancestor_loop_or_blocks_;  // ancestor loops or block realize
  ffi::Map<TensorVar, TensorVar>
      tensor_remap_;  // mapping from the target tensor to the intermediate tensor
  ffi::Map<TensorVar, Stmt>
      tensor_local_stage_;                // mapping from the target tensor to the local stage
  ffi::Array<TensorVar> target_tensors_;  // the target tensors for rewriting
};

namespace transform {

Pass ManifestSharedMemoryLocalStage() {
  auto pass_func = [=](Function f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = ffi::make_object<SharedMemoryLocalStageInserter>()
                  ->Mutate(n->body, InplaceMode::kAllow)
                  .ValueOrUnchanged(std::move(n->body));
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "s_tir.ManifestSharedMemoryLocalStage");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.ManifestSharedMemoryLocalStage",
                        ManifestSharedMemoryLocalStage);
}

}  // namespace transform
}  // namespace s_tir
}  // namespace tvm
