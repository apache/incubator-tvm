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
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ir/analysis.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/type.h>

#include <unordered_set>

#include "../../tensor_region.h"
#include "../utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::prim;
using namespace tvm::tirx;

/******** Error Classes ********/

class NotSingleWriteBlock : public ScheduleErrorContextObj {
 public:
  explicit NotSingleWriteBlock(IRModule mod, TensorVar tensor, ffi::Array<StmtSRef> write_blocks)
      : mod_(std::move(mod)), tensor_(std::move(tensor)) {
    TVM_FFI_ICHECK_GT(write_blocks.size(), 1);
    write_blocks_.reserve(write_blocks.size());
    for (const StmtSRef& block_sref : write_blocks) {
      const SBlockObj* block = TVM_SREF_TO_SBLOCK(block_sref);
      write_blocks_.push_back(ffi::GetRef<SBlock>(block));
    }
  }

  ffi::String FastErrorString() const final {
    return "ScheduleError: The tensor is allowed to be written by single block.";
  }

  ffi::String DetailRenderTemplate() const final {
    size_t k = write_blocks_.size();
    return "The tensor " + tensor_->name + " is expected to be written by single block, but got " +
           std::to_string(k) + " blocks who write it.";
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final {
    return {write_blocks_.begin(), write_blocks_.end()};
  }

 private:
  IRModule mod_;
  TensorVar tensor_;
  ffi::Array<SBlock> write_blocks_;
};

/******** Helper Functions/Classes ********/

/*! \brief The auxiliary info used for the insertion point and content of the cache stage. */
struct CacheStageInfo {
  /*! \brief The tensor to be read. */
  TensorVar read_tensor{ffi::UnsafeInit{}};
  /*! \brief The tensor to be written. */
  TensorVar write_tensor{ffi::UnsafeInit{}};
  /*! \brief The tensor allocation to be inserted into the block signature. */
  ffi::Optional<TensorVar> alloc;
  /*! \brief The AST node whose body is where the cache stage should be inserted. */
  ffi::Optional<StmtSRef> loc_sref;
  /*! \brief The index to insert the cache_read/cache_write stage. */
  size_t loc_pos;
  /*! \brief The cache_read/cache_write stage to be inserted. */
  Stmt cache_stage{ffi::UnsafeInit{}};
  /*! \brief The map used for ScheduleStateObj::Replace. */
  ffi::Map<SBlock, SBlock> block_reuse;
  /*! \brief A set of blocks that will consume the new cache. */
  std::unordered_set<StmtSRef, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> consumer_blocks;
  /*! \brief cache region for the tensor to be cached */
  TensorRegion cache_region{ffi::UnsafeInit{}};
};

/*! \brief Return the tensor region related with the tensor */
ffi::Optional<TensorRegion> GetTensorRegionFromTensor(
    const ffi::Array<TensorRegion>& tensor_regions, const TensorVar& tensor) {
  ffi::Optional<TensorRegion> res = std::nullopt;
  for (const auto& region : tensor_regions) {
    if (region->source.as_or_throw<tvm::tirx::TensorVar>().same_as(tensor)) {
      TVM_FFI_ICHECK(!res.has_value());
      res = region;
    }
  }
  return res;
}

struct ReindexCacheStageInfo : CacheStageInfo {
  /* Indices used to access the allocated cache tensor. */
  ffi::Array<PrimExpr> indices;
  /* Touched loop variable related information. */
  ffi::Array<Var> loop_vars;
  ffi::Array<Range> loop_ranges;
  /* Touched block variable related information. */
  ffi::Array<IterVar> block_iter_vars;
  ffi::Array<PrimExpr> block_iter_values;
};

/* \brief The schedule error that accessed tensor region is not a single point for
 * reindex_cache_read/write. */
class NotSinglePointAccess : public ScheduleErrorContextObj {
 public:
  explicit NotSinglePointAccess(IRModule mod, SBlock block, TensorRegion cache_region,
                                bool is_cache_read)
      : mod_(std::move(mod)), block_(std::move(block)), cache_region_(cache_region) {
    primitive_name_ = is_cache_read ? "reindex_cache_read" : "reindex_cache_write";
  }

  ffi::String FastErrorString() const final {
    return "ScheduleError: The tensor region accessed inside the block is not a single point.";
  }

  ffi::String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The tensor region " << cache_region_
       << " accessed inside block {0} is not a single point, which violates"
       << " the prerequisite of " << primitive_name_ << " primitive.";
    return ffi::String(os.str());
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_}; }

 private:
  IRModule mod_;
  SBlock block_;
  TensorRegion cache_region_;
  ffi::String primitive_name_;
};

/*!
 * \brief Create a loop nest that represents reindex cache copy (reindex_cache_read /
 * reindex_cache_write) from read tensor to write tensor.
 * \param cache_region The cached copy region.
 * \param info The cache stage information, which will be updated in the function.
 * \param storage_scope The storage scope of the cached tensor (only used in naming here)
 * \returns A block indicating the body of the loop nesting.
 */
template <bool is_cache_read>
SBlock MakeReindexCacheStage(const TensorRegion& cache_region, ReindexCacheStageInfo* info,
                             const ffi::String& storage_scope) {
  // loop variables
  std::vector<PrimVar> loop_vars;
  // block variables
  ffi::Array<IterVar> block_vars;
  // bindings in block realize
  std::vector<PrimExpr> iter_values;
  // Create loop vars and block vars' binding_value
  ffi::Map<Var, Var> var_map;
  for (size_t i = 0; i < info->loop_vars.size(); ++i) {
    Var original_var = info->loop_vars[i];
    PrimVar loop_var(original_var->name, original_var->ty.as_or_throw<PrimType>());
    var_map.Set(original_var, loop_var);
    loop_vars.push_back(loop_var);
  }
  auto f_substitute = [&var_map](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (auto repl = var_map.Get(var)) return ffi::Any(*std::move(repl));
    return ffi::Unchanged();
  };
  for (size_t i = 0; i < info->block_iter_vars.size(); ++i) {
    IterVar original_block_var = info->block_iter_vars[i];
    PrimExpr original_iter_value = info->block_iter_values[i];
    IterVar block_var = IterVar(
        /*dom=*/original_block_var->dom.value(),
        /*var=*/PrimVar(original_block_var->var->name, original_block_var->var.ty()),
        /*IterVarType=*/kDataPar);
    var_map.Set(original_block_var->var, block_var->var);
    block_vars.push_back(block_var);
    iter_values.push_back(
        ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(original_iter_value, f_substitute)
            .template as_or_throw<PrimExpr>());
  }

  // block access region for read/write tensors
  ffi::Array<Range> read_access_region, write_access_region;
  ffi::Array<PrimExpr> read_access_indices, write_access_indices;
  // Compute read/write region and read/write access indices.
  ffi::Array<PrimExpr>& old_indices = (is_cache_read) ? read_access_indices : write_access_indices;
  ffi::Array<Range>& old_region = (is_cache_read) ? read_access_region : write_access_region;
  for (const Range& range : cache_region->region) {
    old_indices.push_back(ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(range->begin, f_substitute)
                              .template as_or_throw<PrimExpr>());
    old_region.push_back(Range::FromMinExtent(old_indices.back(), IntImm::Int32(1)));
  }
  ffi::Array<PrimExpr>& new_indices = (is_cache_read) ? write_access_indices : read_access_indices;
  ffi::Array<Range>& new_region = (is_cache_read) ? write_access_region : read_access_region;
  for (const PrimExpr& idx : info->indices) {
    new_indices.push_back(ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(idx, f_substitute)
                              .template as_or_throw<PrimExpr>());
    new_region.push_back(Range::FromMinExtent(new_indices.back(), IntImm::Int32(1)));
  }

  // Create New Block
  SBlock block(
      /*iter_vars*/ std::move(block_vars),
      /*reads=*/{TensorRegion(info->read_tensor, read_access_region, TensorRegionType())},
      /*writes=*/{TensorRegion(info->write_tensor, write_access_region, TensorRegionType())},
      /*name_hint*/ cache_region->source.as_or_throw<tvm::tirx::TensorVar>()->name + "_" +
          storage_scope,
      /*body=*/
      TensorStore(info->write_tensor, write_access_indices,
                  TensorLoad(info->read_tensor, read_access_indices)),
      /*init=*/std::nullopt,
      /*alloc_tensors=*/{},
      /*match_tensors=*/{},
      /*tensor_doms=*/{});
  // Create SBlock Realize node
  Stmt body = SBlockRealize(/*values=*/iter_values,
                            /*predicate=*/IntImm::Bool(true),
                            /*block=*/block);
  // Create surrounding loops
  for (size_t i = loop_vars.size(); i >= 1; --i) {
    body = For(loop_vars[i - 1], info->loop_ranges[i - 1]->begin,
               info->loop_ranges[i - 1]->begin + info->loop_ranges[i - 1]->extent,
               IntImm(loop_vars[i - 1].ty(), 1), ForKind::kDefault, body);
  }
  info->cache_stage = std::move(body);
  return block;
}

/*!
 * \brief Create a loop nest that represents cache copy (cache_read / cache_write) from read tensor
 *        to write tensor.
 * \note This function will store the stmt with loop nesting to the CacheStageInfo, but only return
 *        the inside block.
 * \param cache_region The cached copy region.
 * \param info The cache stage information, which will be updated in the function.
 * \param storage_scope The storage scope of the cached tensor (only used in naming here)
 * \param cache_full_region A boolean indicating if the cache tensor is allocated with
 *        full region or compact region.
 * \returns A block indicating the body of the loop nesting.
 */
SBlock MakeCacheStage(const TensorRegion& cache_region, CacheStageInfo* info,
                      const ffi::String& storage_scope, bool cache_full_region = true) {
  // loop variables
  std::vector<PrimVar> loop_vars;
  // bindings in block realize
  std::vector<PrimExpr> iter_values;
  // Create loop vars and block vars' binding_value
  for (const Range& axis_range : cache_region->region) {
    PrimVar loop_var("ax" + std::to_string(loop_vars.size()), axis_range->extent.ty());
    loop_vars.push_back(loop_var);
    iter_values.push_back(cache_full_region ? (axis_range->begin + loop_var) : loop_var);
  }
  // block variables
  ffi::Array<IterVar> block_vars;
  // block access region for read/write tensors
  ffi::Array<Range> read_access_region;
  ffi::Array<Range> write_access_region;
  // indices used in block body
  ffi::Array<PrimExpr> read_access_indices;
  ffi::Array<PrimExpr> write_access_indices;
  // Create block vars, block's accessed region and accessing indices
  for (int i = 0; i < static_cast<int>(
                          cache_region->source.as_or_throw<tvm::tirx::TensorVar>().shape().size());
       ++i) {
    Range axis_range = cache_region->region[i];
    PrimVar var("v" + std::to_string(read_access_indices.size()), axis_range->extent.ty());
    if (cache_full_region) {
      PrimExpr dim = cache_region->source.as_or_throw<tvm::tirx::TensorVar>().shape()[i];
      block_vars.push_back(IterVar(/*dom=*/Range::FromMinExtent(IntImm(dim.ty(), 0), dim),
                                   /*var=*/var,
                                   /*IterVarType=*/kDataPar));
      read_access_indices.push_back(var);
      write_access_indices.push_back(var);
      read_access_region.push_back(Range::FromMinExtent(var, IntImm(var.ty(), 1)));
      write_access_region.push_back(Range::FromMinExtent(var, IntImm(var.ty(), 1)));
    } else {
      block_vars.push_back(IterVar(
          /*dom=*/Range::FromMinExtent(IntImm(axis_range->extent.ty(), 0), axis_range->extent),
          /*var=*/var,
          /*IterVarType=*/kDataPar));
      if (cache_region->source.as_or_throw<tvm::tirx::TensorVar>().same_as(info->read_tensor)) {
        // cache_read
        read_access_indices.push_back(axis_range->begin + var);
        read_access_region.push_back(
            Range::FromMinExtent(axis_range->begin + var, IntImm(var.ty(), 1)));
        write_access_indices.push_back(var);
        write_access_region.push_back(Range::FromMinExtent(var, IntImm(var.ty(), 1)));
      } else {
        // cache_write
        write_access_indices.push_back(axis_range->begin + var);
        write_access_region.push_back(
            Range::FromMinExtent(axis_range->begin + var, IntImm(var.ty(), 1)));
        read_access_indices.push_back(var);
        read_access_region.push_back(Range::FromMinExtent(var, IntImm(var.ty(), 1)));
      }
    }
  }

  // Create the body block:
  //   reads = [read_tensor[access_region]]
  //   writes = [write_tensor[access_region]]
  //     write_tensor[access_indices] = read_tensor[access_indices]
  SBlock block(
      /*iter_vars=*/std::move(block_vars),
      /*reads=*/{TensorRegion(info->read_tensor, read_access_region, TensorRegionType())},
      /*writes=*/{TensorRegion(info->write_tensor, write_access_region, TensorRegionType())},
      /*name_hint=*/cache_region->source.as_or_throw<tvm::tirx::TensorVar>()->name + "_" +
          storage_scope,
      /*body=*/
      TensorStore(info->write_tensor, write_access_indices,
                  TensorLoad(info->read_tensor, read_access_indices)),
      /*init=*/std::nullopt,
      /*alloc_tensors=*/{},
      /*match_tensors=*/{},
      /*annotations=*/{});
  // Create the block realize node
  Stmt body = SBlockRealize(/*values=*/iter_values,
                            /*predicate=*/IntImm::Bool(true),
                            /*block=*/block);
  // Create surrounding loops
  for (size_t i = loop_vars.size(); i >= 1; --i) {
    body = For(loop_vars[i - 1], 0, cache_region->region[i - 1]->extent,
               IntImm(loop_vars[i - 1].ty(), 1), ForKind::kDefault, body);
  }
  info->cache_stage = std::move(body);
  return block;
}

/*!
 * \brief Create the reindex block and generate the corresponding outer loops.
 * \details The reindex block is a data copy block between the reindex tensor (the intermediate
 * tensor), and the target tensor.
    If tensor_index_type == kWrite, copy from the reindex tensor to the target tensor.
    If tensor_index_type == kRead, copy from the target tensor to the reindex tensor.
    The reindex block has the same block iters and the surrounding loops as the input block.
 However, if a block iter is not used in the indices of the target tensor being reindexed, the
 domain of the block iter, and the corresponding outer loop, will become constant value one, making
 it a trivial iter.
 * \param block The block to be reindexed
 * \param info The cache info
 * \param covered The set of block iter vars covered in the tensor access indices
 * \param original_indices The original tensor access indices
 * \param tensor_index The index of the target tensor
 * \param tensor_index_type The type of tensor index
 * \return The reindex block.
 */
SBlock MakeReIndexStage(const SBlock& block, CacheStageInfo* info,
                        const std::unordered_set<Var>& covered,
                        const ffi::Array<PrimExpr>& original_indices, int tensor_index,
                        TensorIndexType tensor_index_type) {
  // iters of the reindex block
  ffi::Array<IterVar> new_block_iters;
  // the substitution map from the original block iter to the iters of the reindex block
  std::unordered_map<Var, Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> block_var_replace_map;
  // indices to access the reindex tensor and the target tensor
  ffi::Array<PrimExpr> reindex_indices, target_indices;

  // Step 1: Create block iters, access regions of the reindex block, and accessing indices to the
  // reindex tensor.
  std::unordered_set<int> skipped_block_iters;
  for (int i = 0, n = block->iter_vars.size(); i < n; ++i) {
    const IterVar& iter = block->iter_vars[i];
    PrimVar var("v" + std::to_string(new_block_iters.size()), iter->var.ty());
    bool used = covered.count(iter->var);
    if (used) {
      new_block_iters.push_back(IterVar(/*dom=*/iter->dom,
                                        /*var=*/var,
                                        /*IterVarType=*/kDataPar));
    } else {
      skipped_block_iters.insert(i);
    }
    if (used) {
      reindex_indices.push_back(var);
    }
    block_var_replace_map.insert_or_assign(iter->var, var);
  }

  // Step 2: Replace the original block iters with the new block iters
  auto f_substitute =
      [&block_var_replace_map](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (auto it = block_var_replace_map.find(var); it != block_var_replace_map.end()) {
      return ffi::Any(it->second);
    }
    return ffi::Unchanged();
  };
  for (const PrimExpr& index : original_indices) {
    target_indices.push_back(
        ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(index, f_substitute).as_or_throw<PrimExpr>());
  }

  // Step 3: Create the reindex block

  // The src and the dst region and indices of the data copy
  ffi::Array<Range> src_region;
  ffi::Array<Range> dst_region;
  ffi::Array<PrimExpr> src_indices;
  ffi::Array<PrimExpr> dst_indices;

  if (tensor_index_type == TensorIndexType::kWrite) {
    src_indices = reindex_indices;
    dst_indices = target_indices;
  } else {
    src_indices = target_indices;
    dst_indices = reindex_indices;
  }

  // Create the body block
  SBlock new_block(
      /*iter_vars=*/new_block_iters,
      /*reads=*/{s_tir::TensorRegionFromPoint(info->read_tensor, src_indices)},
      /*writes=*/{s_tir::TensorRegionFromPoint(info->write_tensor, dst_indices)},
      /*name_hint=*/info->write_tensor->name + "_reindex",
      /*body=*/
      TensorStore(info->write_tensor, dst_indices, TensorLoad(info->read_tensor, src_indices)));

  // Step 4: Create surrounding loops

  // Create loop vars and bindings for block iters
  std::vector<PrimVar> loop_vars;     // loop variables
  std::vector<PrimExpr> iter_values;  // bindings in block realize
  for (int i = 0; i < static_cast<int>(block->iter_vars.size()); ++i) {
    if (skipped_block_iters.count(i)) {
      continue;
    }
    PrimVar loop_var("ax" + std::to_string(loop_vars.size()), block->iter_vars[i]->var.ty());
    loop_vars.push_back(loop_var);
    iter_values.push_back(loop_var);
  }

  // Create the block realize node
  Stmt body = SBlockRealize(/*values=*/iter_values,
                            /*predicate=*/IntImm::Bool(true),
                            /*block=*/new_block);

  // Create the chain of loops
  for (int i = static_cast<int>(new_block_iters.size()) - 1; i >= 0; --i) {
    body = For(loop_vars[i], new_block_iters[i]->dom.value()->begin,
               new_block_iters[i]->dom.value()->begin + new_block_iters[i]->dom.value()->extent,
               IntImm(loop_vars[i].ty(), 1), ForKind::kDefault, std::move(body));
  }
  // Update cache info, which will be used in the later rewriting.
  info->cache_stage = std::move(body);
  return new_block;
}

/*!
 * \brief Recalculate the `affine_binding` flag of a specific block
 * \param block_sref The sref to the specific block
 */
bool CalculateAffineFlag(const ScheduleState& self, const StmtSRef& block_sref) {
  if (block_sref->parent == nullptr) {
    return true;
  }
  sym::Analyzer analyzer;
  StmtSRef parent_sref = ffi::GetRef<StmtSRef>(block_sref->parent);
  return IsAffineBinding(/*realize=*/GetSBlockRealize(self, block_sref),
                         /*loop_var_ranges=*/LoopDomainOfSRefTreePath(parent_sref),
                         /*analyzer=*/analyzer.get());
}

/*!
 * \brief Insert the cache_read/cache_write stage into the specific position
 * \param stmt A sequence of statements or a single statement that the new stage is inserted in
 * \param pos The position where the cache stage is inserted
 * \param stage The stage to be inserted
 * \return A SeqStmt, the result after insertion
 */
Stmt InsertCacheStage(const Stmt& stmt, int pos, const Stmt& stage) {
  if (const auto* seq_stmt = stmt.as<SeqStmtObj>()) {
    ffi::Array<Stmt> seq = seq_stmt->seq;
    TVM_FFI_ICHECK_LE(pos, seq.size())
        << "Cannot insert at position " << pos << " into sequence of length " << seq.size();
    seq.insert(seq.begin() + pos, stage);
    return SeqStmt(seq);
  } else if (pos == 0) {
    ffi::Array<Stmt> seq;
    seq.push_back(stage);
    seq.push_back(stmt);
    return SeqStmt(seq);
  } else if (pos == 1) {
    ffi::Array<Stmt> seq;
    seq.push_back(stmt);
    seq.push_back(stage);
    return SeqStmt(seq);
  } else {
    TVM_FFI_THROW(InternalError) << "Cannot insert at position " << pos
                                 << ".  When inserting adjacent to non-SeqStmt, "
                                 << "only positions 0 and 1 are valid.";
  }
}

/*!
 * \brief Get the only writer block of the input tensor in a given scope block.
 * \param self The state of the schedule
 * \param scope_sref The scope block where the write is considered
 * \param tensor The queried tensor
 * \return The sref of the only writer of the input tensor in the given scope,
 *         or `std::nullopt` if no block writes it in the scope.
 * \throw NotSingleWriteBlock if there are more than one interested block.
 */
ffi::Optional<StmtSRef> GetOnlyWriteBlock(ScheduleState self, const StmtSRef& scope_sref,
                                          const TensorVar& tensor) {
  SBlockScope scope = self->GetSBlockScope(scope_sref);
  auto it = scope->tensor_writers.find(tensor);
  if (it == scope->tensor_writers.end()) {
    return std::nullopt;
  } else {
    const ffi::Array<StmtSRef>& block_srefs = it->second;
    TVM_FFI_ICHECK(!block_srefs.empty());
    if (block_srefs.size() > 1) {
      throw MakeScheduleError<NotSingleWriteBlock>(self->mod, tensor, block_srefs);
    }
    return block_srefs[0];
  }
}

/*!
 * \brief Check if all the consumer blocks of the given tensor in the given
 *        block scope are the children block of the given target stmt.
 * \param self The state of the schedule .
 * \param tensor The tensor whose consumer blocks are to be check.
 * \param scope_sref The scope block of the check.
 * \param stmt_sref The target stmt
 * \return A boolean indicating if all the consumer blocks of the input tensor
 *         meet the requirement.
 */
bool AllConsumersUnderStmt(ScheduleState self, TensorVar tensor, StmtSRef scope_sref,
                           StmtSRef stmt_sref) {
  // Collect all children blocks of the target stmt.
  std::unordered_set<const SBlockObj*> blocks_under_target;
  for (const StmtSRef& block_sref : GetChildBlocks(self, stmt_sref)) {
    const auto* block = block_sref->StmtAs<SBlockObj>();
    TVM_FFI_ICHECK(block != nullptr);
    blocks_under_target.insert(block);
  }

  // For each block in the scope, if it is a consumer of the
  // input tensor, check if it is also a child block of the
  // target stmt.
  for (const StmtSRef& block_sref : GetChildBlocks(self, scope_sref)) {
    const auto* block = block_sref->StmtAs<SBlockObj>();
    TVM_FFI_ICHECK(block != nullptr);
    if (GetTensorRegionFromTensor(block->reads, tensor).has_value()) {
      if (blocks_under_target.find(block) == blocks_under_target.end()) {
        return false;
      }
    }
  }
  return true;
}

/*!
 * \brief Collect OR-combined predicates from all nested BlockRealize nodes within
 * the given statement that access the specified tensor (read or write, controlled by
 * \p index_type). Each nested block's predicate is expressed in the enclosing block's
 * scope by substituting the nested block's iter var bindings. This is needed when the
 * actual access is gated by a predicate (T.where) on a nested block while the outer
 * block has a trivially-true predicate. Sibling blocks that each access the tensor under
 * different predicates are OR-ed together so the result covers the union of their access
 * regions.
 * \param body The body statement of the outer block to search within.
 * \param tensor The tensor being accessed.
 * \param index_type Whether to look for reads (kRead) or writes (kWrite).
 * \return The OR-combination of all nested block predicates found.
 */
static PrimExpr CollectNestedBlockPredicates(const Stmt& body, const TensorVar& tensor,
                                             TensorIndexType index_type) {
  struct Collector : public StmtExprVisitor {
    using StmtExprVisitor::Visit_;

    ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
      if (value.as<ExprObj>()) return std::nullopt;
      return StmtExprVisitor::Visit(value);
    }

    Collector(const TensorVar& tensor, TensorIndexType idx_type)
        : tensor_(tensor), index_type_(idx_type), result_(IntImm::Bool(false)), found_(false) {}

    ffi::Optional<VisitInterrupt> Visit_(const SBlockRealizeObj* realize) final {
      const SBlockObj* block = realize->block.get();
      const auto& regions = (index_type_ == TensorIndexType::kRead) ? block->reads : block->writes;
      bool accesses_tensor = false;
      for (const TensorRegion& region : regions) {
        if (region->source.as_or_throw<tvm::tirx::TensorVar>().same_as(tensor_)) {
          accesses_tensor = true;
          break;
        }
      }
      if (accesses_tensor) {
        // Build substitution: nested block iter vars -> their binding values
        // (which are already expressed in terms of the outer scope).
        ffi::Map<Var, PrimExpr> subst;
        for (size_t i = 0; i < block->iter_vars.size(); ++i) {
          subst.Set(block->iter_vars[i]->var, realize->iter_values[i]);
        }
        auto f_substitute = [&subst](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
          if (auto repl = subst.Get(var)) return ffi::Any(*std::move(repl));
          return ffi::Unchanged();
        };
        PrimExpr pred = subst.empty() ? realize->predicate
                                      : ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(
                                            realize->predicate, f_substitute)
                                            .as_or_throw<PrimExpr>();
        // OR the predicates across all accessing nested blocks: each such block is an
        // independent alternative access path (sibling blocks in a SeqStmt), so the
        // cache must cover the *union* of their access regions, not the intersection.
        // Using AND (the previous behaviour) underestimates the required region when
        // sibling blocks have non-overlapping predicates.
        result_ = found_ ? (result_ || pred) : pred;
        found_ = true;
      }
      // Continue recursing into deeper nested blocks.
      return StmtExprVisitor::Visit_(realize);
    }

    const TensorVar& tensor_;
    TensorIndexType index_type_;
    PrimExpr result_;
    bool found_;
  };

  auto collector = ffi::make_object<Collector>(tensor, index_type);
  collector->Visit(body);
  // If no nested block accessed the tensor, return true (no restriction — the caller
  // will fall back to the original scope-block reads / FullRegion path).
  return collector->found_ ? collector->result_ : IntImm::Bool(true);
}

/*!
 * \brief Get the tensor region under the sref tree path [dom_low_inclusive, dom_high_exclusive)
 * \param self The state of the schedule.
 * \param tensor_region The tensor region to be analyzed.
 * \param block_sref The sref of the block related to the region.
 * \param dom_low_inclusive The lowest node in the sref tree path.
 * \param dom_high_exclusive The highest node in the sref tree path.
 * \param extra_predicate An additional predicate (e.g. collected from nested blocks) to AND
 *        with the block's own predicate before relaxation. Defaults to true (no effect).
 * \return The relaxed tensor region.
 */
TensorRegion RelaxTensorRegion(ScheduleState self, const TensorRegion& tensor_region,
                               const StmtSRef& block_sref, const StmtSRef& dom_low_inclusive,
                               const ffi::Optional<StmtSRef>& dom_high_exclusive,
                               PrimExpr extra_predicate = IntImm::Bool(true)) {
  SBlockRealize realize = GetSBlockRealize(self, block_sref);
  ffi::Map<Var, PrimExpr> binding = GetBindings(realize);
  const TensorVar& tensor = tensor_region->source.as_or_throw<tvm::tirx::TensorVar>();
  sym::Analyzer analyzer;
  auto f_substitute = [&binding](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (auto repl = binding.Get(var)) return ffi::Any(*std::move(repl));
    return ffi::Unchanged();
  };
  ffi::Array<Range> mapped_region = tensor_region->region.Map([&f_substitute](const Range& range) {
    PrimExpr min = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(range->begin, f_substitute)
                       .as_or_throw<PrimExpr>();
    PrimExpr extent = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(range->extent, f_substitute)
                          .as_or_throw<PrimExpr>();
    return Range::FromMinExtent(min, extent);
  });
  TensorRegion subst_region = TensorRegion(tensor, mapped_region, TensorRegionType());
  ffi::Array<sym::IntSet> int_sets = AnalyzeRegionUpperBound(
      /*region=*/subst_region,
      /*predicate=*/
      ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(realize->predicate && extra_predicate,
                                                    f_substitute)
          .as_or_throw<PrimExpr>(),
      /*dom_low_inclusive=*/dom_low_inclusive,
      /*dom_high_exclusive=*/dom_high_exclusive,
      /*analyzer=*/analyzer.get());
  TVM_FFI_ICHECK_EQ(tensor_region->region.size(), int_sets.size());

  ffi::Array<Range> region;
  region.reserve(int_sets.size());
  for (size_t i = 0; i < int_sets.size(); ++i) {
    region.push_back(int_sets[i].CoverRange(Range::FromMinExtent(0, tensor.shape()[i])).value());
  }
  return TensorRegion(tensor, region, TensorRegionType());
}

/*! \brief Detect the insertion position of the new cache stage */
class CacheLocDetector : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;

  ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
    if (value.as<ExprObj>()) return std::nullopt;
    return StmtExprVisitor::Visit(value);
  }

  /*!
   * \brief Detect the insertion position of the cache stage, and write the position into the
   * CacheStageInfo
   * \param self The state of the schedule
   * \param block_sref The sref of the unique writer block of the tensor being applied cache_read or
   * cache_write
   * \param scope_sref The sref of the scope block of the cached block
   * \param info The cache stage info.
   */
  template <bool is_cache_read>
  static void Detect(const ScheduleState& self, const StmtSRef& block_sref,
                     const StmtSRef& scope_sref, CacheStageInfo* info) {
    std::vector<StmtSRef> related_blocks;
    // If consumer is specified, skip detecting the others
    if (is_cache_read) {
      if (info->consumer_blocks.size() > 0) {
        for (StmtSRef consumer : info->consumer_blocks) {
          related_blocks.emplace_back(consumer);
        }
      } else {
        for (const Dependency& def : self->GetSBlockScope(scope_sref)->GetDepsBySrc(block_sref)) {
          if (def->kind == DepKind::kRAW) {
            related_blocks.push_back(def->dst);
          }
        }
      }
    } else {
      for (const Dependency& def : self->GetSBlockScope(scope_sref)->GetDepsBySrc(block_sref)) {
        if (def->kind == DepKind::kRAW) {
          if (info->consumer_blocks.count(def->dst)) {
            continue;
          }
          related_blocks.push_back(def->dst);
        }
      }
    }

    if (!related_blocks.empty()) {
      auto detector =
          ffi::make_object<CacheLocDetector>(self, block_sref, scope_sref, related_blocks);
      detector->Visit(ffi::GetRef<Stmt>(scope_sref->stmt));
      info->loc_sref = detector->loc_sref_;
      info->loc_pos = detector->loc_pos_;
    } else {
      info->loc_sref = scope_sref;

      info->loc_pos = scope_sref->StmtAs<SBlockObj>()->body->size();
    }
  }

  /*!
   * \brief Constructor
   * \param self The state of the schedule
   * \param block_sref The sref of the unique writer block of the tensor being applied cache_read or
   * cache_write
   * \param scope_sref The sref of the scope block of the cached block
   * \param related_blocks Producer blocks for cache_write, or consumer blocks for cache_read
   */
  CacheLocDetector(const ScheduleState self, const StmtSRef& block_sref, const StmtSRef& scope_sref,
                   const std::vector<StmtSRef>& related_blocks)
      : self_(self),
        block_sref_(block_sref),
        scope_sref_(scope_sref),
        related_blocks_(related_blocks) {}

 private:
  ffi::Optional<VisitInterrupt> Visit_(const SeqStmtObj* seq_stmt) final {
    bool previous_visited_block = visited_block_;
    visited_block_ = false;

    for (size_t i = 0; i < seq_stmt->size(); ++i) {
      if (loc_pos_ != -1) {
        break;
      }
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(seq_stmt->seq[i]));
      // `pos` can be assigned only once when we visited `block_sref`
      if (visited_block_ && visited_related_ && loc_pos_ == -1) {
        // The offset of insert position from the block
        loc_pos_ = i;
        break;
      } else if (visited_related_) {
        // If meet the target consumer, stop searching
        break;
      }
    }
    visited_block_ = visited_block_ || previous_visited_block;
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const SBlockObj* block) final {
    // Only visit the current scope under tensor writer's parent block
    if (block == scope_sref_->stmt) {
      // The block visited is the current parent scope
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(block));
      // Handling cases when insert outside any loop or cache_read for input tensor
      if (visited_related_ && !loc_sref_.has_value()) {
        loc_sref_ = self_->stmt2ref.at(block);
        // Handling cache_read for input tensor
        if (visited_block_ == false && loc_pos_ == -1) {
          loc_pos_ = 0;
        }
      }
      return std::nullopt;
    }
    // Update `visited_block`
    if (block_sref_->stmt == block) {
      visited_block_ = true;
      return std::nullopt;
    }
    // Update `visited_related`
    for (const StmtSRef& related_block : related_blocks_) {
      if (related_block->stmt == block) {
        visited_related_ = true;
        return std::nullopt;
      }
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const ForObj* loop) final {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(loop));
    if (visited_block_ && visited_related_ && !loc_sref_.has_value() && loc_pos_ != -1) {
      loc_sref_ = self_->stmt2ref.at(loop);
    }
    return std::nullopt;
  }

  /*! \brief The schedule class */
  const ScheduleState self_;
  /*! \brief The dominate block which write the tensor */
  const StmtSRef& block_sref_;
  /*! \brief The parent scope of the dominate block */
  const StmtSRef& scope_sref_;
  /*! \brief Producer blocks for cache_write and consumer blocks for cache_read */
  const std::vector<StmtSRef>& related_blocks_;
  /*! \brief The flag whether we have visited the dominate block */
  bool visited_block_{false};
  /*! \brief The flag whether we have visited at least one related blocks */
  bool visited_related_{false};
  /*! \brief The AST node whose body is where the cache stage should be inserted */
  ffi::Optional<StmtSRef> loc_sref_;
  /*! \brief The index to insert the cache_read/cache_write stage */
  int loc_pos_{-1};
};

/*! \brief Detect the insertion position of the new cache stage */
class CacheInplaceLocDetector : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;

  ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
    if (value.as<ExprObj>()) return std::nullopt;
    return StmtExprVisitor::Visit(value);
  }

  /*!
   * \brief Detect the insertion position of the cache stage, and write the position into the
   * CacheStageInfo
   * \param self The state of the schedule
   * \param block_sref The sref of the unique block of the tensor being applied cache_inplace
   * \param scope_sref The sref of the scope block of the cached block
   * \param info The cache stage info.
   */
  static void Detect(const ScheduleState& self, const StmtSRef& block_sref,
                     const StmtSRef& scope_sref, CacheStageInfo* info) {
    auto detector = ffi::make_object<CacheInplaceLocDetector>(self, block_sref, scope_sref);
    detector->Visit(ffi::GetRef<Stmt>(scope_sref->stmt));
    info->loc_sref = detector->loc_sref_;
    info->loc_pos = detector->loc_pos_;
  }

  /*!
   * \brief Constructor
   * \param self The state of the schedule
   * \param block_sref The sref of the unique writer block of the tensor being applied cache_inplace
   * \param scope_sref The sref of the scope block of the cached block
   */
  CacheInplaceLocDetector(const ScheduleState self, const StmtSRef& block_sref,
                          const StmtSRef& scope_sref)
      : self_(self), block_sref_(block_sref), scope_sref_(scope_sref) {}

 private:
  ffi::Optional<VisitInterrupt> Visit_(const SeqStmtObj* seq_stmt) final {
    for (size_t i = 0; i < seq_stmt->size(); ++i) {
      if (loc_pos_ != -1) {
        break;
      }
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(seq_stmt->seq[i]));
      // `pos` can be assigned only once when we visited `block_sref`
      if (visited_block_ && loc_pos_ == -1) {
        // The offset of insert position from the block
        loc_pos_ = i;
        return std::nullopt;
      }
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const SBlockObj* block) final {
    // Only visit the current scope under tensor writer's parent block
    if (block == scope_sref_->stmt) {
      // The block visited is the current parent scope
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(block));
      // Handling cases when insert outside any loop
      if (visited_block_ && !loc_sref_.has_value()) {
        loc_sref_ = self_->stmt2ref.at(block);
        // Handling for input tensor
        if (loc_pos_ == -1) {
          loc_pos_ = 0;
        }
      }
    } else if (block_sref_->stmt == block) {
      visited_block_ = true;
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const ForObj* loop) final {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(loop));
    if (visited_block_ && !loc_sref_.has_value()) {
      loc_sref_ = self_->stmt2ref.at(loop);
      if (loc_pos_ == -1) {
        loc_pos_ = 0;
      }
    }
    return std::nullopt;
  }

  /*! \brief The schedule class */
  const ScheduleState self_;
  /*! \brief The dominate block which write the tensor */
  const StmtSRef& block_sref_;
  /*! \brief The parent scope of the dominate block */
  const StmtSRef& scope_sref_;
  /*! \brief The flag whether we have visited the target block */
  bool visited_block_{false};
  /*! \brief The AST node whose body is where the cache stage should be inserted */
  ffi::Optional<StmtSRef> loc_sref_;
  /*! \brief The index to insert the cache_read/cache_write stage */
  int loc_pos_{-1};
};

class ReindexCacheReadRewriter;

/*! \brief Mutator for CacheRead. */
class CacheReadRewriter : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  /*!
   * \brief Rewrite the AST and add a cache_read stage with the information provided
   * \param scope_sref The parent scope of this mutation
   * \param info The cache stage information
   * \param cache_full_region A boolean indicating if the cache tensor is allocated with
   *        full region or compact region.
   * \return The new AST rooting at the original parent scope
   */
  static Stmt Rewrite(const StmtSRef& scope_sref, CacheStageInfo* info,
                      bool cache_full_region = true) {
    auto rewriter = ffi::make_object<CacheReadRewriter>(scope_sref, info, cache_full_region);
    return rewriter->Mutate(ffi::GetRef<Stmt>(scope_sref->stmt))
        .ValueOrUnchanged(ffi::GetRef<Stmt>(scope_sref->stmt));
  }

  explicit CacheReadRewriter(const StmtSRef& scope_sref, CacheStageInfo* info,
                             bool cache_full_region = true)
      : scope_sref_(scope_sref), info_(info), cache_full_region_(cache_full_region) {
    VarRemapSet(info_->read_tensor, info_->write_tensor);
    auto update_region = [this](const ffi::Array<Range>& region,
                                const ffi::Array<Range>& offset) -> ffi::Array<Range> {
      TVM_FFI_ICHECK_EQ(region.size(), offset.size());
      std::vector<Range> ret;
      for (size_t i = 0; i < region.size(); ++i) {
        ret.push_back(Range::FromMinExtent(ana_->Simplify(region[i]->begin - offset[i]->begin),
                                           region[i]->extent));
      }
      return ret;
    };

    update_access_regions = [this, update_region](ffi::Array<TensorRegion> regions) {
      if (cache_full_region_) {
        return ReplaceTensor(std::move(regions), info_->read_tensor, info_->write_tensor);
      }

      ffi::Array<TensorRegion> ret;
      for (const TensorRegion& region : regions) {
        if (region->source.as_or_throw<tvm::tirx::TensorVar>().same_as(info_->read_tensor)) {
          ret.push_back(TensorRegion(info_->write_tensor,
                                     update_region(region->region, info_->cache_region->region),
                                     TensorRegionType()));
        } else {
          ret.push_back(region);
        }
      }
      return ret;
    };
    update_match_tensors = [this, update_region](ffi::Array<MatchTensorRegion> match_tensors) {
      if (cache_full_region_) {
        return ReplaceTensor(std::move(match_tensors), info_->read_tensor, info_->write_tensor);
      }

      ffi::Array<MatchTensorRegion> ret;
      for (const MatchTensorRegion& match_tensor : match_tensors) {
        if (match_tensor->source->source.as_or_throw<tvm::tirx::TensorVar>().same_as(
                info_->read_tensor)) {
          ret.push_back(MatchTensorRegion(
              match_tensor->tensor,
              TensorRegion(info_->write_tensor,
                           update_region(match_tensor->source->region, info_->cache_region->region),
                           TensorRegionType())));
        } else {
          ret.push_back(match_tensor);
        }
      }
      return ret;
    };
  }

 private:
  UnchangedOr<Stmt> Mutate_(const ForObj* loop, InplaceMode inplace_mode) final {
    Stmt stmt =
        StmtExprMutator::Mutate_(loop, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(loop));
    // Check the insertion point
    if (loop == info_->loc_sref.value()->stmt) {
      // Insert cache stage into the loop if it is the right place
      ffi::ObjectPtr<ForObj> n = ffi::make_object<ForObj>(*stmt.as<ForObj>());
      n->body = InsertCacheStage(n->body, info_->loc_pos, info_->cache_stage);
      stmt = Stmt(n);
    }
    return stmt;
  }

  UnchangedOr<Stmt> Mutate_(const SBlockObj* block, InplaceMode inplace_mode) override {
    SBlock old_stmt = ffi::GetRef<SBlock>(block);
    // Check if this block is one of the specified consumers.
    // If no consumer blocks are specified, all blocks should be considered consumers.
    bool is_consumer = info_->consumer_blocks.empty();
    // Otherwise check if this is one of the specified blocks.
    for (StmtSRef consumer_sref : info_->consumer_blocks) {
      const SBlockObj* consumer_node = TVM_SREF_TO_SBLOCK(consumer_sref);
      SBlock consumer_block = ffi::GetRef<SBlock>(consumer_node);
      if (old_stmt.same_as(consumer_block)) {
        is_consumer = true;
      }
    }
    // Keep track of this blocks status. We'll use this when rewriting loads.
    current_block_consumes = is_consumer;
    // We don't mutate the block which generates info->read_tensor.
    if (block != scope_sref_->stmt &&
        GetTensorRegionFromTensor(block->writes, info_->read_tensor).has_value()) {
      return old_stmt;
    }
    // Mutate the body
    // Cache accesses change storage; the original allocation still owns its source.
    auto input_node = ffi::make_object<SBlockObj>(*block);
    input_node->alloc_tensors.clear();
    SBlock input(std::move(input_node));
    SBlock stmt = StmtExprMutator::Mutate_(input.get(), InplaceMode::kDisallow)
                      .ValueOrUnchanged(input)
                      .as_or_throw<SBlock>();
    stmt.CopyOnWrite()->alloc_tensors = block->alloc_tensors;
    // Check the insertion point
    if (block == info_->loc_sref.value()->stmt) {
      // Insert cache stage into the block if it is the right place
      ffi::ObjectPtr<SBlockObj> n = ffi::make_object<SBlockObj>(*stmt.as<SBlockObj>());
      n->body = InsertCacheStage(n->body, info_->loc_pos, info_->cache_stage);
      stmt = SBlock(n);
    }
    // Check if it is the block corresponding to the parent scope
    if (block == scope_sref_->stmt) {
      // If so, put tensor allocation on the parent scope
      ffi::ObjectPtr<SBlockObj> n = ffi::make_object<SBlockObj>(*stmt.as<SBlockObj>());
      // In cache_inplace case, alloc_tensor may be already exits.
      if (info_->alloc.has_value()) {
        n->alloc_tensors.push_back(info_->alloc.value());
        stmt = SBlock(n);
      }
    } else {
      // Otherwise, update read regions and match_tensors
      // Only make this change if the block is one of the specified consumers.
      if (is_consumer) {
        // Use the updated block stmt
        ffi::Array<TensorRegion> reads = update_access_regions(stmt->reads);
        ffi::Array<MatchTensorRegion> match_tensors = update_match_tensors(stmt->match_tensors);
        if (!reads.same_as(stmt->reads) || !match_tensors.same_as(stmt->match_tensors)) {
          ffi::ObjectPtr<SBlockObj> n = ffi::make_object<SBlockObj>(*stmt.as<SBlockObj>());
          n->reads = std::move(reads);
          n->match_tensors = std::move(match_tensors);
          stmt = SBlock(n);
        }
      }
    }
    info_->block_reuse.Set(old_stmt, stmt);
    return stmt;
  }

  ffi::Array<PrimExpr> RewriteIndices(const ffi::Array<PrimExpr>& indices) {
    std::vector<PrimExpr> ret;
    for (size_t i = 0; i < indices.size(); ++i) {
      ret.push_back(ana_->Simplify(indices[i] - info_->cache_region->region[i]->begin));
    }
    return ret;
  }

  UnchangedOr<Expr> Mutate_(const CallObj* op, InplaceMode inplace_mode) override {
    auto result = StmtExprMutator::Mutate_(op, inplace_mode);
    if (!result.IsUnchanged()) {
      op = ffi::AnyView(result).as<CallObj>();
      if (!op->unique()) inplace_mode = InplaceMode::kDisallow;
    }
    // Cache remapping can change pointer storage scope; the base Call hook preserves its type.
    if (!op->op.same_as(tirx::tensor_data_ptr_op()) || op->args.size() != 1) return result;
    PtrType type = op->args[0].as_or_throw<TensorVar>().ty()->DataPtrType();
    if (ffi::StructuralEqual()(op->ty, type)) return result;
    if (inplace_mode == InplaceMode::kAllow) {
      const_cast<CallObj*>(op)->ty = std::move(type);
      return result;
    }
    auto copy = ffi::make_object<CallObj>(*op);
    copy->ty = std::move(type);
    return Expr(std::move(copy));
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* load, InplaceMode inplace_mode) override {
    if (load->source.as_or_throw<tvm::tirx::TensorVar>().same_as(info_->read_tensor) &&
        current_block_consumes) {
      ffi::Array<PrimExpr> indices = load->indices;
      if (!cache_full_region_) {
        indices = RewriteIndices(load->indices);
      }
      TensorLoad node = ffi::GetRef<TensorLoad>(load);
      auto* n = node.CopyOnWrite();
      n->source = info_->write_tensor;
      n->indices = indices;
      return node;
    }
    auto indices = Mutate(load->indices).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    TensorLoad node = ffi::GetRef<TensorLoad>(load);
    if (!indices.UnchangedOrSameAs(load->indices)) {
      node.CopyOnWrite()->indices = std::move(indices).ValueUnchecked();
    }
    return node;
  }

  UnchangedOr<Expr> Mutate_(const TensorRegionObj* op, InplaceMode inplace_mode) final {
    if (!op->source.as<TensorVar>()) {
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }
    auto region = Mutate(op->region).as_or_throw<UnchangedOr<ffi::Array<Range>>>();
    if (region.UnchangedOrSameAs(op->region)) return ffi::Unchanged();
    TensorRegion node = ffi::GetRef<TensorRegion>(op);
    node.CopyOnWrite()->region = std::move(region).ValueUnchecked();
    return node;
  }

  /*! \brief The parent scope of the insertion */
  const StmtSRef& scope_sref_;
  /*! \brief The info for inserting cache stage */
  CacheStageInfo* info_;
  /*! \brief Whether the most recently visited block is a specified consumer. */
  bool current_block_consumes;
  /*! \brief function to update read/write region of block being cache read.*/
  std::function<ffi::Array<TensorRegion>(ffi::Array<TensorRegion>)> update_access_regions;
  /*! \brief function to update match tensors of block being cache read.*/
  std::function<ffi::Array<MatchTensorRegion>(ffi::Array<MatchTensorRegion>)> update_match_tensors;
  /*!
   * \brief A boolean indicating if the cache tensor is allocated with
   *        full region or compact region.
   */
  bool cache_full_region_;
  /*! \brief Arithmetic analyzer. */
  sym::Analyzer ana_;

  friend ReindexCacheReadRewriter;
};

/*! \brief Mutator for ReindexCacheRead. */
class ReindexCacheReadRewriter : public CacheReadRewriter {
 public:
  using CacheReadRewriter::Mutate;
  using CacheReadRewriter::Mutate_;

  /*!
   * \brief Rewrite the AST and add a cache_read stage with the information provided.
   * \param scope_sref The parent scope of this mutation.
   * \param info The cache stage information.
   * \return The new AST rooting at the original parent scope.
   */
  static Stmt Rewrite(const StmtSRef& scope_sref, ReindexCacheStageInfo* info) {
    auto rewriter = ffi::make_object<ReindexCacheReadRewriter>(scope_sref, info);
    return rewriter->Mutate(ffi::GetRef<Stmt>(scope_sref->stmt))
        .ValueOrUnchanged(ffi::GetRef<Stmt>(scope_sref->stmt));
  }

  explicit ReindexCacheReadRewriter(const StmtSRef& scope_sref, ReindexCacheStageInfo* info)
      : CacheReadRewriter(scope_sref, info) {
    new_indices_ = info->indices;
    update_access_regions = [&](ffi::Array<TensorRegion> reads) {
      ffi::Array<TensorRegion> new_reads;
      for (const TensorRegion& tensor_region : reads) {
        if (tensor_region->source.as_or_throw<tvm::tirx::TensorVar>().same_as(info_->read_tensor)) {
          ffi::Array<Range> region;
          for (const PrimExpr index : new_indices_) {
            region.push_back(Range::FromMinExtent(index, IntImm::Int32(1)));
          }
          new_reads.push_back(TensorRegion(info_->write_tensor, region, TensorRegionType()));
        } else {
          new_reads.push_back(tensor_region);
        }
      }
      return new_reads;
    };
    update_match_tensors = [&](const ffi::Array<MatchTensorRegion> match_tensors) {
      ffi::Array<MatchTensorRegion> new_match_tensors;
      for (const MatchTensorRegion& match_tensor_region : match_tensors) {
        TensorRegion source = match_tensor_region->source;
        if (source->source.as_or_throw<tvm::tirx::TensorVar>().same_as(info_->read_tensor)) {
          ffi::Array<Range> region;
          for (const PrimExpr index : new_indices_) {
            region.push_back(Range::FromMinExtent(index, IntImm::Int32(1)));
          }
          new_match_tensors.push_back(
              MatchTensorRegion(match_tensor_region->tensor,
                                TensorRegion(info_->write_tensor, region, TensorRegionType())));
        } else {
          new_match_tensors.push_back(match_tensor_region);
        }
      }
      return new_match_tensors;
    };
  }

 private:
  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* load, InplaceMode inplace_mode) final {
    if (load->source.as_or_throw<tvm::tirx::TensorVar>().same_as(info_->read_tensor) &&
        current_block_consumes) {
      TensorLoad node = ffi::GetRef<TensorLoad>(load);
      auto* n = node.CopyOnWrite();
      n->source = info_->write_tensor;
      n->indices = new_indices_;
      return node;
    }
    return CacheReadRewriter::Mutate_(load, inplace_mode);
  }

  /*! \brief The indices to use for new tensor. */
  ffi::Array<PrimExpr> new_indices_;
};

class ReindexCacheWriteRewriter;

/*! \brief Mutator for CacheWrite */
class CacheWriteRewriter : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  /*!
   * \brief Rewrite the AST and add a cache_write stage with the information provided.
   * \param scope_sref The parent scope of this mutation.
   * \param writer_block_sref The only writer block in the scope.
   * \param info The cache stage information.
   * \param cache_full_region A boolean indicating if the cache tensor is allocated with
   *        full region or compact region.
   * \return The new AST rooting at the original parent scope.
   */
  static Stmt Rewrite(const StmtSRef& scope_sref, const StmtSRef& writer_block_sref,
                      CacheStageInfo* info, bool cache_full_region = true) {
    auto rewriter = ffi::make_object<CacheWriteRewriter>(scope_sref, writer_block_sref, info,
                                                         cache_full_region);
    return rewriter->Mutate(ffi::GetRef<Stmt>(scope_sref->stmt))
        .ValueOrUnchanged(ffi::GetRef<Stmt>(scope_sref->stmt));
  }

  explicit CacheWriteRewriter(const StmtSRef& scope_sref, const StmtSRef& writer_block_sref,
                              CacheStageInfo* info, bool cache_full_region = true)
      : scope_sref_(scope_sref),
        writer_block_sref_(writer_block_sref),
        info_(info),
        cache_full_region_(cache_full_region) {
    VarRemapSet(info_->write_tensor, info_->read_tensor);
    auto update_region = [this](const ffi::Array<Range>& region,
                                const ffi::Array<Range>& offset) -> ffi::Array<Range> {
      TVM_FFI_ICHECK_EQ(region.size(), offset.size());
      std::vector<Range> ret;
      for (size_t i = 0; i < region.size(); ++i) {
        ret.push_back(Range::FromMinExtent(ana_->Simplify(region[i]->begin - offset[i]->begin),
                                           region[i]->extent));
      }
      return ret;
    };

    update_access_regions = [this, update_region](ffi::Array<TensorRegion> regions) {
      if (cache_full_region_) {
        return ReplaceTensor(regions, info_->write_tensor, info_->read_tensor);
      }

      ffi::Array<TensorRegion> ret;
      for (const TensorRegion& region : regions) {
        if (region->source.as_or_throw<tvm::tirx::TensorVar>().same_as(info_->write_tensor)) {
          ret.push_back(TensorRegion(info_->read_tensor,
                                     update_region(region->region, info_->cache_region->region),
                                     TensorRegionType()));
        } else {
          ret.push_back(region);
        }
      }
      return ret;
    };
    update_match_tensors = [this, update_region](ffi::Array<MatchTensorRegion> match_tensors) {
      if (cache_full_region_) {
        return ReplaceTensor(match_tensors, info_->write_tensor, info_->read_tensor);
      }

      ffi::Array<MatchTensorRegion> ret;
      for (const MatchTensorRegion& match_tensor : match_tensors) {
        if (match_tensor->source->source.as_or_throw<tvm::tirx::TensorVar>().same_as(
                info_->write_tensor)) {
          ret.push_back(MatchTensorRegion(
              match_tensor->tensor,
              TensorRegion(info_->read_tensor,
                           update_region(match_tensor->source->region, info_->cache_region->region),
                           TensorRegionType())));
        } else {
          ret.push_back(match_tensor);
        }
      }
      return ret;
    };
  }

 private:
  UnchangedOr<Stmt> Mutate_(const ForObj* loop, InplaceMode inplace_mode) final {
    Stmt stmt =
        StmtExprMutator::Mutate_(loop, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(loop));
    // Check the insertion point
    if (loop == info_->loc_sref.value()->stmt) {
      // Insert cache stage into the loop if it is the right place
      ffi::ObjectPtr<ForObj> n = ffi::make_object<ForObj>(*stmt.as<ForObj>());
      n->body = InsertCacheStage(n->body, info_->loc_pos, info_->cache_stage);
      stmt = Stmt(n);
    }
    return stmt;
  }

  UnchangedOr<Stmt> Mutate_(const SBlockObj* block, InplaceMode inplace_mode) override {
    SBlock old_stmt = ffi::GetRef<SBlock>(block);

    // Check if this block is one of the specified cache consumers.
    // update the read tensor to the cache.
    for (StmtSRef consumer_sref : info_->consumer_blocks) {
      const SBlockObj* consumer_node = TVM_SREF_TO_SBLOCK(consumer_sref);
      SBlock consumer_block = ffi::GetRef<SBlock>(consumer_node);
      if (old_stmt.same_as(consumer_block)) {
        ffi::Array<TensorRegion> writes = update_access_regions(block->writes);
        ffi::Array<TensorRegion> reads = update_access_regions(block->reads);
        ffi::Array<MatchTensorRegion> match_tensors = update_match_tensors(block->match_tensors);
        if (!writes.same_as(block->writes) || !reads.same_as(block->reads) ||
            !match_tensors.same_as(block->match_tensors)) {
          SBlock new_consumer = old_stmt;
          SBlockObj* n = new_consumer.CopyOnWrite();
          n->writes = std::move(writes);
          n->reads = std::move(reads);
          n->match_tensors = std::move(match_tensors);
          n->body = Mutate(block->body, inplace_mode).ValueOrUnchanged(block->body);
          info_->block_reuse.Set(old_stmt, new_consumer);
          return new_consumer;
        }
        return old_stmt;
      }
    }

    // We only mutate the block which generates info->write_tensor
    if (block != writer_block_sref_->stmt && block != scope_sref_->stmt && !under_writer_block_) {
      return old_stmt;
    }

    // Mutate the body
    bool under_scope = under_writer_block_ || block == writer_block_sref_->stmt;
    std::swap(under_scope, under_writer_block_);
    // Cache accesses change storage; the original allocation still owns its source.
    auto input_node = ffi::make_object<SBlockObj>(*block);
    input_node->alloc_tensors.clear();
    SBlock input(std::move(input_node));
    SBlock stmt = StmtExprMutator::Mutate_(input.get(), InplaceMode::kDisallow)
                      .ValueOrUnchanged(input)
                      .as_or_throw<SBlock>();
    stmt.CopyOnWrite()->alloc_tensors = block->alloc_tensors;
    std::swap(under_scope, under_writer_block_);

    // Find the insertion point
    if (block == info_->loc_sref.value()->stmt) {
      ffi::ObjectPtr<SBlockObj> n = ffi::make_object<SBlockObj>(*stmt.as<SBlockObj>());
      n->body = InsertCacheStage(n->body, info_->loc_pos, info_->cache_stage);
      stmt = SBlock(n);
    }
    // Put tensor allocation on the parent scope
    if (block == scope_sref_->stmt) {
      ffi::ObjectPtr<SBlockObj> n = ffi::make_object<SBlockObj>(*stmt.as<SBlockObj>());
      // In cache_inplace case, alloc_tensor may be already exits.
      if (info_->alloc.has_value()) {
        n->alloc_tensors.push_back(info_->alloc.value());
        stmt = SBlock(n);
      }
    } else {
      // Since cache_write changes the block, we need to update the tensor it writes
      auto writes = update_access_regions(block->writes);
      auto reads = update_access_regions(block->reads);
      auto match_tensors = update_match_tensors(block->match_tensors);
      if (!writes.same_as(block->writes) || !reads.same_as(block->reads) ||
          !match_tensors.same_as(block->match_tensors)) {
        ffi::ObjectPtr<SBlockObj> n = ffi::make_object<SBlockObj>(*stmt.as<SBlockObj>());
        n->writes = std::move(writes);
        n->reads = std::move(reads);
        n->match_tensors = std::move(match_tensors);
        stmt = SBlock(n);
      }
    }
    info_->block_reuse.Set(old_stmt, stmt);
    return stmt;
  }

  ffi::Array<PrimExpr> RewriteIndices(const ffi::Array<PrimExpr>& indices) {
    std::vector<PrimExpr> ret;
    for (size_t i = 0; i < indices.size(); ++i) {
      ret.push_back(ana_->Simplify(indices[i] - info_->cache_region->region[i]->begin));
    }
    return ret;
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreObj* store, InplaceMode inplace_mode) override {
    bool rewrite_tensor = store->dest.as_or_throw<TensorVar>().same_as(info_->write_tensor);
    auto value = Mutate(store->value);
    auto indices = Mutate(store->indices).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    TensorStore stmt = ffi::GetRef<TensorStore>(store);
    if (!value.UnchangedOrSameAs(store->value) || !indices.UnchangedOrSameAs(store->indices)) {
      auto* n = stmt.CopyOnWrite();
      n->value = std::move(value).ValueOrUnchanged(store->value);
      n->indices = std::move(indices).ValueOrUnchanged(store->indices);
    }
    if (rewrite_tensor) {
      TensorStoreObj* n = stmt.CopyOnWrite();
      n->dest = info_->read_tensor;
      if (!cache_full_region_) {
        n->indices = RewriteIndices(n->indices);
      }
      return stmt;
    } else {
      return stmt;
    }
  }

  UnchangedOr<Expr> Mutate_(const CallObj* op, InplaceMode inplace_mode) override {
    auto result = StmtExprMutator::Mutate_(op, inplace_mode);
    if (!result.IsUnchanged()) {
      op = ffi::AnyView(result).as<CallObj>();
      if (!op->unique()) inplace_mode = InplaceMode::kDisallow;
    }
    // Cache remapping can change pointer storage scope; the base Call hook preserves its type.
    if (!op->op.same_as(tirx::tensor_data_ptr_op()) || op->args.size() != 1) return result;
    PtrType type = op->args[0].as_or_throw<TensorVar>().ty()->DataPtrType();
    if (ffi::StructuralEqual()(op->ty, type)) return result;
    if (inplace_mode == InplaceMode::kAllow) {
      const_cast<CallObj*>(op)->ty = std::move(type);
      return result;
    }
    auto copy = ffi::make_object<CallObj>(*op);
    copy->ty = std::move(type);
    return Expr(std::move(copy));
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* load, InplaceMode inplace_mode) override {
    if (load->source.as_or_throw<tvm::tirx::TensorVar>().same_as(info_->write_tensor)) {
      ffi::Array<PrimExpr> indices = load->indices;
      if (!cache_full_region_) {
        indices = RewriteIndices(indices);
      }
      TensorLoad node = ffi::GetRef<TensorLoad>(load);
      auto* n = node.CopyOnWrite();
      n->source = info_->read_tensor;
      n->indices = indices;
      return node;
    }
    auto indices = Mutate(load->indices).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    TensorLoad node = ffi::GetRef<TensorLoad>(load);
    if (!indices.UnchangedOrSameAs(load->indices)) {
      node.CopyOnWrite()->indices = std::move(indices).ValueUnchecked();
    }
    return node;
  }

  UnchangedOr<Expr> Mutate_(const TensorRegionObj* op, InplaceMode inplace_mode) final {
    if (!op->source.as<TensorVar>()) {
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }
    auto region = Mutate(op->region).as_or_throw<UnchangedOr<ffi::Array<Range>>>();
    if (region.UnchangedOrSameAs(op->region)) return ffi::Unchanged();
    TensorRegion node = ffi::GetRef<TensorRegion>(op);
    node.CopyOnWrite()->region = std::move(region).ValueUnchecked();
    return node;
  }

  /*! \brief The parent scope of the insertion. */
  const StmtSRef& scope_sref_;
  /*! \brief The parent scope of the insertion. */
  const StmtSRef& writer_block_sref_;
  /*! \brief The info for inserting cache stage. */
  CacheStageInfo* info_;
  /*! \brief Whether the current node is under the given block. */
  bool under_writer_block_{false};
  /*! \brief function to update read/write region of block being cache write.*/
  std::function<ffi::Array<TensorRegion>(ffi::Array<TensorRegion>)> update_access_regions;
  /*! \brief function to update match tensors of block being cache write.*/
  std::function<ffi::Array<MatchTensorRegion>(ffi::Array<MatchTensorRegion>)> update_match_tensors;
  /*!
   * \brief A boolean indicating if the cache tensor is allocated with
   *        full region or compact region.
   */
  bool cache_full_region_;
  /*! \brief Arithmetic analyzer. */
  sym::Analyzer ana_;

  friend ReindexCacheWriteRewriter;
};

/*! \brief Mutator for ReindexCacheWrite. */
class ReindexCacheWriteRewriter : public CacheWriteRewriter {
 public:
  using CacheWriteRewriter::Mutate;
  using CacheWriteRewriter::Mutate_;

  /*!
   * \brief Rewrite the AST and add a cache_write stage with the information provided.
   * \param scope_sref The parent scope of this mutation.
   * \param writer_block_sref The only writer block in the scope.
   * \param info The cache stage information.
   * \return The new AST rooting at the original parent scope.
   */
  static Stmt Rewrite(const StmtSRef& scope_sref, const StmtSRef& writer_block_sref,
                      ReindexCacheStageInfo* info) {
    auto rewriter =
        ffi::make_object<ReindexCacheWriteRewriter>(scope_sref, writer_block_sref, info);
    return rewriter->Mutate(ffi::GetRef<Stmt>(scope_sref->stmt))
        .ValueOrUnchanged(ffi::GetRef<Stmt>(scope_sref->stmt));
  }

  explicit ReindexCacheWriteRewriter(const StmtSRef& scope_sref, const StmtSRef& writer_block_sref,
                                     ReindexCacheStageInfo* info)
      : CacheWriteRewriter(scope_sref, writer_block_sref, info) {
    new_indices_ = info->indices;
    update_access_regions = [&](ffi::Array<TensorRegion> reads) {
      ffi::Array<TensorRegion> new_reads;
      for (const TensorRegion& tensor_region : reads) {
        if (tensor_region->source.as_or_throw<tvm::tirx::TensorVar>().same_as(
                info_->write_tensor)) {
          ffi::Array<Range> region;
          for (const PrimExpr index : new_indices_) {
            region.push_back(Range::FromMinExtent(index, IntImm::Int32(1)));
          }
          new_reads.push_back(TensorRegion(info_->read_tensor, region, TensorRegionType()));
        } else {
          new_reads.push_back(tensor_region);
        }
      }
      return new_reads;
    };
    update_match_tensors = [&](const ffi::Array<MatchTensorRegion> match_tensors) {
      ffi::Array<MatchTensorRegion> new_match_tensors;
      for (const MatchTensorRegion& match_tensor_region : match_tensors) {
        TensorRegion source = match_tensor_region->source;
        if (source->source.as_or_throw<tvm::tirx::TensorVar>().same_as(info_->write_tensor)) {
          ffi::Array<Range> region;
          for (const PrimExpr index : new_indices_) {
            region.push_back(Range::FromMinExtent(index, IntImm::Int32(1)));
          }
          new_match_tensors.push_back(
              MatchTensorRegion(match_tensor_region->tensor,
                                TensorRegion(info_->read_tensor, region, TensorRegionType())));
        } else {
          new_match_tensors.push_back(match_tensor_region);
        }
      }
      return new_match_tensors;
    };
  }

 private:
  UnchangedOr<Stmt> Mutate_(const TensorStoreObj* store, InplaceMode inplace_mode) final {
    bool rewrite_tensor = store->dest.as_or_throw<TensorVar>().same_as(info_->write_tensor);
    auto value = Mutate(store->value);
    auto indices = Mutate(store->indices).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    TensorStore stmt = ffi::GetRef<TensorStore>(store);
    if (!value.UnchangedOrSameAs(store->value) || !indices.UnchangedOrSameAs(store->indices)) {
      auto* n = stmt.CopyOnWrite();
      n->value = std::move(value).ValueOrUnchanged(store->value);
      n->indices = std::move(indices).ValueOrUnchanged(store->indices);
    }
    if (rewrite_tensor) {
      TensorStoreObj* n = stmt.CopyOnWrite();
      n->dest = info_->read_tensor;
      n->indices = new_indices_;
      return stmt;
    } else {
      return stmt;
    }
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* load, InplaceMode inplace_mode) final {
    if (load->source.as_or_throw<tvm::tirx::TensorVar>().same_as(info_->write_tensor)) {
      TensorLoad node = ffi::GetRef<TensorLoad>(load);
      auto* n = node.CopyOnWrite();
      n->source = info_->read_tensor;
      n->indices = new_indices_;
      return node;
    }
    auto indices = Mutate(load->indices).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    TensorLoad node = ffi::GetRef<TensorLoad>(load);
    if (!indices.UnchangedOrSameAs(load->indices)) {
      node.CopyOnWrite()->indices = std::move(indices).ValueUnchecked();
    }
    return node;
  }

  /*! \brief The indices to use for new tensor. */
  ffi::Array<PrimExpr> new_indices_;
};

/*!
 * \brief Create a new tensor by change the shape with block iters to be used as the reindex tensor
 * \param tensor The given tensor.
 * \param block_iters The block iters.
 * \param covered Set of block iter vars covered by the tensor access indices
 * \return The new tensor with target shape.
 */
TensorVar CreateReindexTensor(const TensorVar& tensor, const ffi::Array<IterVar>& block_iters,
                              const std::unordered_set<Var>& covered) {
  ffi::ObjectPtr<TensorTypeObj> new_tensor = ffi::make_object<TensorTypeObj>(*tensor.ty().get());
  std::vector<PrimExpr> new_shape;
  for (const auto& iter : block_iters) {
    if (covered.count(iter->var)) {
      new_shape.push_back(iter->dom.value()->begin + iter->dom.value()->extent);
    }
  }
  new_tensor->shape = new_shape;
  if (new_tensor->layout.has_value()) {
    auto tile = new_tensor->layout.value().as_or_throw<tirx::TileLayout>();
    new_tensor->layout =
        tirx::TileLayout(tirx::TileLayout::DefaultLayout(new_shape)->shard, {}, tile->offset);
  }
  return TensorVar(tensor->name + "_reindex", TensorType(std::move(new_tensor)), tensor->loc);
}

/*!
 * \brief The schedule error that the target is not a leaf block.
 */
class NotLeafBlockError : public ScheduleErrorContextObj {
 public:
  NotLeafBlockError(IRModule mod, SBlock block) : mod_(std::move(mod)), block_(std::move(block)) {}
  ffi::String FastErrorString() const final {
    return "ScheduleError: The target block is not a leaf block.";
  }

  ffi::String DetailRenderTemplate() const final {
    return "The target block {0} is not a leaf block.";
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_}; }
  IRModule mod_;
  SBlock block_;
};

/*! \brief The schedule error that the tensor access is invalid for reindex. */
class InvalidTensorAccessError : public ScheduleErrorContextObj {
 public:
  enum class ErrorKind {
    kNoAccess,         // tensor access not found
    kNonUniqueAccess,  // multiple tensor accesses with different indices
    kOpaqueAccess,     // opaque access to the tensor
  };

  InvalidTensorAccessError(IRModule mod, TensorVar tensor, SBlock block, ErrorKind kind)
      : mod_(std::move(mod)), tensor_(std::move(tensor)), block_(std::move(block)), kind_(kind) {}
  ffi::String FastErrorString() const final {
    return "ScheduleError: The target tensor should be accessed via TensorLoad or TensorStore. The "
           "indices should be the same if there are multiple accesses to the target tensor.";
  }

  ffi::String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The target tensor " << tensor_->name
       << " should be accessed in the leaf block {0} via TensorLoad or TensorStore. The indices "
          "should be the same if there are multiple accesses to the target tensor. ";
    if (kind_ == ErrorKind::kNoAccess) {
      os << "No tensor accesses found.";
    } else if (kind_ == ErrorKind::kNonUniqueAccess) {
      os << "Multiple tensor accesses have non-unique indices.";
    } else if (kind_ == ErrorKind::kOpaqueAccess) {
      os << "Opaque tensor accesses found.";
    }
    return os.str();
  }
  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_}; }

 private:
  IRModule mod_;
  TensorVar tensor_;
  SBlock block_;
  ErrorKind kind_;
};

/*! \brief Collect the related Load/Store to reindex */
class ReIndexCollector : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;

  static ffi::Array<PrimExpr> Collect(const IRModule& mod, const TensorVar& tensor,
                                      const SBlock& block) {
    auto collector = ffi::make_object<ReIndexCollector>(mod, tensor, block);
    collector->Visit(block->body);
    if (!collector->tensor_access_indices_.has_value()) {
      throw MakeScheduleError<InvalidTensorAccessError>(
          mod, tensor, block, InvalidTensorAccessError::ErrorKind::kNoAccess);
    }
    return collector->tensor_access_indices_.value();
  }

  explicit ReIndexCollector(const IRModule& mod, const TensorVar& tensor, const SBlock& block)
      : mod_(mod), tensor_(tensor), block_(block) {}

 private:
  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadObj* load) final {
    for (const PrimExpr& index : load->indices) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(index));
    }
    if (load->source.as_or_throw<tvm::tirx::TensorVar>().same_as(tensor_)) {
      CheckAndUpdateTensorAccessIndices(load->indices);
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const SBlockObj* block) final {
    // no sub-blocks under this block
    throw MakeScheduleError<NotLeafBlockError>(mod_, block_);
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorStoreObj* store) final {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(store->value));
    for (const PrimExpr& index : store->indices) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(index));
    }
    if (store->dest.as_or_throw<TensorVar>().same_as(tensor_)) {
      CheckAndUpdateTensorAccessIndices(store->indices);
    }
    return std::nullopt;
  }

  void CheckAndUpdateTensorAccessIndices(const ffi::Array<PrimExpr> indices) {
    if (!tensor_access_indices_.has_value()) {
      tensor_access_indices_ = indices;
      return;
    } else if (!std::equal(tensor_access_indices_.value().begin(),
                           tensor_access_indices_.value().end(), indices.begin(), indices.end(),
                           prim::ExprDeepEqual())) {
      throw MakeScheduleError<InvalidTensorAccessError>(
          mod_, tensor_, block_, InvalidTensorAccessError::ErrorKind::kNonUniqueAccess);
    }
  }

  ffi::Optional<VisitInterrupt> Visit_(const VarObj* var) final {
    if (def_region_kind() != kTVMFFIDefRegionKindNone) return StmtExprVisitor::Visit_(var);
    if (var == tensor_.get()) {
      throw MakeScheduleError<InvalidTensorAccessError>(
          mod_, tensor_, block_, InvalidTensorAccessError::ErrorKind::kOpaqueAccess);
    }
    return std::nullopt;
  }
  /*! \brief The IR module */
  IRModule mod_;
  /*! \brief The tensor to rewrite */
  TensorVar tensor_;
  /*! \brief The block to visit */
  SBlock block_;
  /*! \brief The indices of tensor acess to rewrite */
  ffi::Optional<ffi::Array<PrimExpr>> tensor_access_indices_;
};

/*! \brief Mutator of ReIndex */
class ReIndexRewriter : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  static Stmt Rewrite(const StmtSRef& scope_sref, const StmtSRef& block_sref, CacheStageInfo* info,
                      const std::unordered_set<Var>& covered) {
    auto rewriter = ffi::make_object<ReIndexRewriter>(block_sref, info, covered);
    return rewriter->Mutate(ffi::GetRef<Stmt>(scope_sref->stmt))
        .ValueOrUnchanged(ffi::GetRef<Stmt>(scope_sref->stmt));
  }

  explicit ReIndexRewriter(const StmtSRef& block_sref, CacheStageInfo* info,
                           const std::unordered_set<Var>& covered)
      : block_sref_(block_sref),
        info_(info),
        covered_(covered),
        old_tensor_(info->read_tensor.same_as(info->alloc.value()) ? info->write_tensor
                                                                   : info->read_tensor),
        new_tensor_(info->alloc.value()) {}

 private:
  UnchangedOr<Stmt> Mutate_(const SBlockObj* block, InplaceMode inplace_mode) final {
    SBlock old_stmt = ffi::GetRef<SBlock>(block);
    if (is_scope_) {
      is_scope_ = false;
      SBlock stmt = StmtExprMutator::Mutate_(block, inplace_mode)
                        .ValueOrUnchanged(ffi::GetRef<Stmt>(block))
                        .as_or_throw<SBlock>();
      // Insert cache stage into the loop
      ffi::ObjectPtr<SBlockObj> n = ffi::make_object<SBlockObj>(*stmt.as<SBlockObj>());
      n->body = InsertCacheStage(n->body, info_->loc_pos, info_->cache_stage);
      n->alloc_tensors.push_back(info_->alloc.value());
      stmt = SBlock(n);
      info_->block_reuse.Set(old_stmt, stmt);
      return stmt;
    }

    // Visiting the blokc being reindexed
    if (block == block_sref_->stmt) {
      // Collect the updated indices and regions
      for (const IterVar& iter : block->iter_vars) {
        if (covered_.count(iter->var)) {
          indices_.push_back(iter->var);
          region_.push_back(Range::FromMinExtent(iter->var, IntImm(iter->var.ty(), 1)));
        }
      }
      SBlock stmt = StmtExprMutator::Mutate_(block, inplace_mode)
                        .ValueOrUnchanged(ffi::GetRef<Stmt>(block))
                        .as_or_throw<SBlock>();
      // Update block reads/writes to use the intermediate reindex tensor
      auto writes = ReplaceTensorRegion(block->writes, old_tensor_,
                                        TensorRegion(new_tensor_, region_, TensorRegionType()));
      auto reads = ReplaceTensorRegion(block->reads, old_tensor_,
                                       TensorRegion(new_tensor_, region_, TensorRegionType()));
      auto match_tensors =
          ReplaceTensorRegion(block->match_tensors, old_tensor_,
                              TensorRegion(new_tensor_, region_, TensorRegionType()));
      if (!writes.same_as(block->writes) || !reads.same_as(block->reads) ||
          !match_tensors.same_as(block->match_tensors)) {
        ffi::ObjectPtr<SBlockObj> n = ffi::make_object<SBlockObj>(*stmt.as<SBlockObj>());
        n->writes = std::move(writes);
        n->reads = std::move(reads);
        n->match_tensors = std::move(match_tensors);
        stmt = SBlock(n);
      }
      info_->block_reuse.Set(old_stmt, stmt);
      return stmt;
    }
    return old_stmt;
  }

  TensorStore VisitTensorAccess(TensorStore node) {
    if (node->dest.same_as(old_tensor_)) {
      auto* n = node.CopyOnWrite();
      n->dest = new_tensor_;
      n->indices = indices_;
    }
    return node;
  }
  TensorLoad VisitTensorAccess(TensorLoad node) {
    if (node->source.as_or_throw<tvm::tirx::TensorVar>().same_as(old_tensor_)) {
      auto* n = node.CopyOnWrite();
      n->source = new_tensor_;
      n->indices = indices_;
    }
    return node;
  }
  UnchangedOr<Stmt> Mutate_(const TensorStoreObj* op, InplaceMode inplace_mode) final {
    auto value = Mutate(op->value);
    auto indices = Mutate(op->indices).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    TensorStore tensor_store = ffi::GetRef<TensorStore>(op);
    if (!value.UnchangedOrSameAs(op->value) || !indices.UnchangedOrSameAs(op->indices)) {
      auto* n = tensor_store.CopyOnWrite();
      n->value = std::move(value).ValueOrUnchanged(op->value);
      n->indices = std::move(indices).ValueOrUnchanged(op->indices);
    }
    return VisitTensorAccess(std::move(tensor_store));
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* op, InplaceMode inplace_mode) final {
    auto indices = Mutate(op->indices).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    TensorLoad tensor_load = ffi::GetRef<TensorLoad>(op);
    if (!indices.UnchangedOrSameAs(op->indices)) {
      tensor_load.CopyOnWrite()->indices = std::move(indices).ValueUnchecked();
    }
    return VisitTensorAccess(std::move(tensor_load));
  }

  /*! \brief The parent scope of the insertion. */
  const StmtSRef& block_sref_;
  /*! \brief The info for inserting reindex stage. */
  CacheStageInfo* info_;
  /*! \brief Whether old block var is covered in the indices */
  const std::unordered_set<Var>& covered_;
  /*! \brief Whether the current block is scope block */
  bool is_scope_{true};
  /*! \brief The  tensor to be replaced */
  TensorVar old_tensor_;
  /*! \brief The reindex tensor */
  TensorVar new_tensor_;
  /*! \brief The new indices */
  ffi::Array<PrimExpr> indices_;
  /*! \brief The new region */
  ffi::Array<Range> region_;
};

void CheckRegionCover(const ScheduleState& self, StmtSRef scope_root, TensorVar read_tensor) {
  class NotRegionCoverError : public ScheduleErrorContextObj {
   public:
    explicit NotRegionCoverError(IRModule mod, SBlock block) : mod_(mod), block_(block) {}
    IRModule mod() const final { return mod_; }
    ffi::String FastErrorString() const final {
      return "ScheduleError: The scope root's region cover is not complete.";
    }
    ffi::String DetailRenderTemplate() const final {
      return R"(The scope {0} 's region cover is not complete.
The region cover property require to hold for every of its child blocks
)";
    }
    ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_}; }
    IRModule mod_;
    SBlock block_;
  };

  for (const auto& child_block_sref : GetChildBlocks(self, scope_root)) {
    const SBlockObj* child_block = TVM_SREF_TO_SBLOCK(child_block_sref);
    for (const TensorRegion& region : child_block->reads) {
      if (region->source.as_or_throw<tvm::tirx::TensorVar>().same_as(read_tensor)) {
        if (!self->block_info.at(child_block_sref).region_cover) {
          const SBlockObj* block = TVM_SREF_TO_SBLOCK(scope_root);
          throw MakeScheduleError<NotRegionCoverError>(self->mod, ffi::GetRef<SBlock>(block));
        }
      }
    }
  }
}

/******** Implementation ********/

StmtSRef CacheRead(ScheduleState self, const StmtSRef& block_sref, int read_tensor_index,
                   const ffi::String& storage_scope, const ffi::Array<StmtSRef> consumer_blocks) {
  /*!
   * Check:
   *   - The index is in the array of block reading region
   *   - There is at most one block who write the tensor in the scope
   *
   * Mutate:
   *   - Allocate new cache tensor under the current scope.
   *   - Find the lowest ancestor of the block and ANY ONE of the consumers blocks.
   *   - Copy the tensor with the consumed region.
   */

  // Step 0. Check the input storage scope.
  CheckStorageScope(self, storage_scope);

  // Step 1. Check index, getting the target tensor and the parent scope
  const SBlockObj* block = TVM_SREF_TO_SBLOCK(block_sref);
  TensorVar read_tensor = GetNthAccessTensor(self, ffi::GetRef<SBlock>(block), read_tensor_index,
                                             TensorIndexType::kRead);
  StmtSRef scope_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/false);
  // Check required region cover for cache_read
  CheckRegionCover(self, scope_sref, read_tensor);
  const SBlockObj* scope_block = TVM_SREF_TO_SBLOCK(scope_sref);

  // Step 2. Create CacheStageInfo
  CacheStageInfo info;
  info.read_tensor = read_tensor;

  // info.consumer_blocks indicates which tensors should consume the cache.
  for (auto consumer : consumer_blocks) {
    info.consumer_blocks.insert(consumer);
    for (auto child : GetChildBlocks(self, consumer)) {
      info.consumer_blocks.insert(child);
    }
  }

  // Step 3. Update cache stage info.
  TensorRegion cache_region = [&]() -> TensorRegion {
    if (ffi::Optional<StmtSRef> _write_block_sref =
            GetOnlyWriteBlock(self, scope_sref, read_tensor)) {
      // Case 1. The tensor is written inside the block.
      StmtSRef write_block_sref = _write_block_sref.value();
      const SBlockObj* write_block = TVM_SREF_TO_SBLOCK(write_block_sref);
      // Find the producing region
      TensorRegion region = GetTensorRegionFromTensor(write_block->writes, read_tensor).value();
      StmtSRef parent_sref = ffi::GetRef<StmtSRef>(write_block_sref->parent);

      // Detect insert position
      CacheLocDetector::Detect</*is_cache_read=*/true>(self, write_block_sref, scope_sref, &info);
      return RelaxTensorRegion(self, region, write_block_sref, parent_sref, info.loc_sref);
    } else {
      // Case 2. The tensor is the input block for the scope.
      info.loc_sref = scope_sref;
      info.loc_pos = 0;
      // When a nested block gates the actual read with T.where, the consumer block's own
      // predicate is trivially true, so the scope-block read annotation covers the full loop
      // range. Collect nested-read predicates and, if any are non-trivial, relax the consumer
      // block's read region under that predicate to get a tighter cache allocation.
      // Without a nested predicate we fall back to scope_block->reads (which preserves the
      // original tensor's dtype in its extents, e.g. int64 shapes).
      ffi::Optional<TensorRegion> read_region_opt =
          GetTensorRegionFromTensor(block->reads, read_tensor);
      PrimExpr nested_pred = read_region_opt ? CollectNestedBlockPredicates(
                                                   block->body, read_tensor, TensorIndexType::kRead)
                                             : IntImm::Bool(true);
      if (read_region_opt && !IsOne(nested_pred) && block_sref->parent != nullptr) {
        StmtSRef parent_sref = ffi::GetRef<StmtSRef>(block_sref->parent);
        return RelaxTensorRegion(self, read_region_opt.value(), block_sref, parent_sref, scope_sref,
                                 nested_pred);
      } else if (ffi::Optional<TensorRegion> scope_region =
                     GetTensorRegionFromTensor(scope_block->reads, read_tensor)) {
        return scope_region.value();
      } else {
        return s_tir::FullTensorRegion(read_tensor);
      }
    }
  }();

  // Step 4. Making new cache stage block and rewrite readers.
  bool cache_full_region =
      info.loc_sref.value()->StmtAs<SBlockObj>() == nullptr ||
      !AllConsumersUnderStmt(self, read_tensor, scope_sref, info.loc_sref.value());
  info.cache_region = cache_region;
  info.write_tensor = WithScope(read_tensor, runtime::MemoryScope::Get(storage_scope));
  if (!cache_full_region) {
    auto write_tensor = ffi::make_object<TensorTypeObj>(*info.write_tensor.ty().get());
    std::vector<PrimExpr> shape;
    for (auto cache_range : info.cache_region->region) {
      shape.push_back(cache_range->extent);
    }
    write_tensor->shape = std::move(shape);
    info.write_tensor = TensorVar(info.write_tensor->name, TensorType(std::move(write_tensor)),
                                  info.write_tensor->loc);
  }
  info.alloc = info.write_tensor;

  SBlock cache_read_stage =
      MakeCacheStage(/*cache_region=*/cache_region, /*info=*/&info,
                     /*storage_scope=*/storage_scope, /*cache_full_region=*/cache_full_region);
  Stmt new_scope = CacheReadRewriter::Rewrite(/*scope_sref=*/scope_sref, /*info=*/&info,
                                              /*cache_full_region=*/cache_full_region);

  // Step 5. Replacing and updating flags.
  self->Replace(scope_sref, new_scope, info.block_reuse);
  StmtSRef result_block_sref = self->stmt2ref.at(cache_read_stage.get());
  SBlockInfo& block_info = self->block_info[result_block_sref];
  block_info.affine_binding = CalculateAffineFlag(self, result_block_sref);
  block_info.region_cover = true;
  block_info.stage_pipeline = true;
  return result_block_sref;
}

StmtSRef CacheWrite(ScheduleState self, const StmtSRef& block_sref, int write_tensor_index,
                    const ffi::String& storage_scope, const ffi::Array<StmtSRef> consumer_blocks) {
  /*!
   * Check:
   *   - The index is in the array of block reading region
   *   - There is only one block who write the tensor in the scope
   *
   * Mutate:
   *   - Allocate new cache tensor under the current scope.
   *   - Find the lowest ancestor of the block and ANY ONE of the producer blocks.
   *   - Copy the tensor with the consumed region.
   */

  // Step 0. Check the input storage scope.
  CheckStorageScope(self, storage_scope);

  // Step 1. Checking index, getting the target tensor and the parent scope
  const SBlockObj* block = TVM_SREF_TO_SBLOCK(block_sref);
  TensorVar write_tensor = GetNthAccessTensor(self, ffi::GetRef<SBlock>(block), write_tensor_index,
                                              TensorIndexType::kWrite);
  StmtSRef scope_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/false);

  // Step 2. Creating CacheStageInfo
  CacheStageInfo info;
  // Create the corresponding tensor to be written, i.e. result of cache_write
  info.write_tensor = write_tensor;

  // info.consumer_blocks indicates which tensors should consume the cache.
  for (auto consumer : consumer_blocks) {
    info.consumer_blocks.insert(consumer);
    for (auto child : GetChildBlocks(self, consumer)) {
      info.consumer_blocks.insert(child);
    }
  }

  // Step 3. Check the only writer block.
  ffi::Optional<StmtSRef> only_write_block = GetOnlyWriteBlock(self, scope_sref, write_tensor);
  TVM_FFI_ICHECK(only_write_block.has_value());
  TVM_FFI_ICHECK_EQ(block_sref.get(), only_write_block.value().get());

  // Step 4. Find the producing region and insert position
  TensorRegion region = GetTensorRegionFromTensor(block->writes, write_tensor).value();
  // Detect insert position
  CacheLocDetector::Detect</*is_cache_read=*/false>(self, block_sref, scope_sref, &info);
  // Collect predicates from any nested blocks that gate the actual write (e.g. T.where on an
  // inner block). The outer block's own predicate may be trivially true even though the write
  // is restricted by a nested predicate, so we OR them together for a tighter region estimate.
  PrimExpr nested_write_pred =
      CollectNestedBlockPredicates(block->body, write_tensor, TensorIndexType::kWrite);
  TensorRegion cache_region{ffi::UnsafeInit{}};
  if (block_sref->parent != nullptr) {
    StmtSRef parent_sref = ffi::GetRef<StmtSRef>(block_sref->parent);
    cache_region =
        RelaxTensorRegion(self, region, block_sref, parent_sref, info.loc_sref, nested_write_pred);
  } else {
    // Root block: no enclosing loops to relax over, use the write region directly.
    cache_region = region;
  }

  bool cache_full_region =
      info.loc_sref.value()->StmtAs<SBlockObj>() == nullptr ||
      !AllConsumersUnderStmt(self, write_tensor, scope_sref, info.loc_sref.value());
  info.cache_region = cache_region;
  info.read_tensor = WithScope(write_tensor, runtime::MemoryScope::Get(storage_scope));
  if (!cache_full_region) {
    auto read_tensor_type = ffi::make_object<TensorTypeObj>(*info.read_tensor.ty().get());
    std::vector<PrimExpr> shape;
    for (auto cache_range : info.cache_region->region) {
      shape.push_back(cache_range->extent);
    }
    read_tensor_type->shape = std::move(shape);
    info.read_tensor = TensorVar(info.read_tensor->name, TensorType(std::move(read_tensor_type)),
                                 info.read_tensor->loc);
  }
  info.alloc = info.read_tensor;

  // Step 5. Making new cache stage block and rewrite readers.
  SBlock cache_write_stage =
      MakeCacheStage(/*cache_region=*/cache_region, /*info=*/&info,
                     /*storage_scope=*/storage_scope, /*cache_full_region=*/cache_full_region);
  Stmt new_scope = CacheWriteRewriter::Rewrite(/*scope_sref=*/scope_sref,
                                               /*writer_block_sref=*/block_sref, /*info=*/&info,
                                               /*cache_full_region=*/cache_full_region);

  // Step 6. Replacing and updating flags.
  self->Replace(scope_sref, new_scope, info.block_reuse);
  StmtSRef result_block_sref = self->stmt2ref.at(cache_write_stage.get());
  SBlockInfo& block_info = self->block_info[result_block_sref];
  block_info.affine_binding = CalculateAffineFlag(self, result_block_sref);
  block_info.region_cover = true;
  block_info.stage_pipeline = true;
  return result_block_sref;
}

ffi::Array<StmtSRef> GetLoopsUnderScope(const StmtSRef& block_sref,
                                        const ffi::Optional<StmtSRef>& top_sref) {
  std::vector<StmtSRef> result;
  for (StmtSRefObj* parent = block_sref->parent; parent && parent->stmt->IsInstance<ForObj>();
       parent = parent->parent) {
    if (top_sref.has_value() && parent == top_sref.value().get()) break;
    result.push_back(ffi::GetRef<StmtSRef>(parent));
  }
  return {result.rbegin(), result.rend()};
}

/*!
 * \brief The schedule error that block iter vars appears in old tensor and new
 * allocated cache tensor does not match.
 */
class ReindexCacheReadWriteNotMatchError : public ScheduleErrorContextObj {
 public:
  ReindexCacheReadWriteNotMatchError(IRModule mod, SBlock block, Var var,
                                     ffi::Array<PrimExpr> old_indices,
                                     ffi::Array<PrimExpr> new_indices, bool is_cache_read,
                                     bool appears_in_old)
      : mod_(std::move(mod)), block_(std::move(block)), var_(std::move(var)) {
    primitive_name_ = is_cache_read ? "reindex_cache_read" : "reindex_cache_write";
    if (appears_in_old) {
      appears_indices_ = std::move(old_indices);
      other_indices_ = std::move(new_indices);
    } else {
      appears_indices_ = std::move(new_indices);
      other_indices_ = std::move(old_indices);
    }
  }
  ffi::String FastErrorString() const final {
    return "ScheduleError: the block itervars appeared in lhs and rhs of reindex cache stage do "
           "not match.";
  }

  ffi::String DetailRenderTemplate() const final {
    std::stringstream s;
    s << "Error when applying " << primitive_name_ << " on block {0}, the block itervar " << var_
      << " appears in " << appears_indices_ << ", but not in " << other_indices_ << ".";
    return ffi::String(s.str());
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_}; }
  IRModule mod_;
  ffi::String primitive_name_;
  SBlock block_;
  Var var_;
  ffi::Array<PrimExpr> appears_indices_;
  ffi::Array<PrimExpr> other_indices_;
};

/*!
 * \brief Update ReindexCacheStageInfo and create new cache tensor, used in
 * both ReindexCacheRead and ReindexCacheWrite.
 * \param info Pointer to ReindexCacheStageInfo
 * \param mod The IRModule.
 * \param block_sref The StmtSRef to the block we are working on.
 * \param storage_scope The storage scope of cache tensor (e.g. "shared"/"local").
 * \param index_map The user defined indices.
 * \param blok The block we are working on.
 * \param realize The BlockRealize this block belongs to.
 * \param old_tensor The tensor whose tensor access need to be rewriten.
 * \param cache_region The old tensor access region.
 */
template <bool is_cache_read>
void CollectReindexCacheStageInfoAndCreateTensor(
    ReindexCacheStageInfo* info, const IRModule& mod, const StmtSRef& block_sref,
    const ffi::String& storage_scope, const IndexMap& index_map, const SBlock& block,
    const SBlockRealize& realize, const TensorVar& old_tensor, const TensorRegion& cache_region) {
  sym::Analyzer analyzer;
  ffi::Array<PrimExpr> block_iter_vars, block_shape;
  for (const IterVar& iter_var : block->iter_vars) {
    block_iter_vars.push_back(iter_var->var);
    block_shape.push_back(iter_var->dom.value()->extent);
  }
  ffi::Array<PrimExpr> new_indices = index_map->MapIndices(block_iter_vars, analyzer);
  ffi::Array<PrimExpr> new_shape = index_map->MapShape(block_shape, analyzer);
  info->indices = new_indices;

  // Step 5. Update CacheTouchedInfo
  auto collect_vars = [](const ffi::Array<PrimExpr>& values) {
    ffi::Array<Var> vars = UndefinedVars(values);
    return std::unordered_set<Var>(vars.begin(), vars.end());
  };
  ffi::Array<PrimExpr> old_indices;
  for (const Range& range : cache_region->region) {
    old_indices.push_back(range->begin);
  }
  auto old_vars = collect_vars(old_indices);
  auto new_vars = collect_vars(new_indices);
  for (size_t i = 0; i < block->iter_vars.size(); ++i) {
    const IterVar& block_iter_var = block->iter_vars[i];
    const PrimExpr& block_iter_value = realize->iter_values[i];
    bool appears_in_new = new_vars.count(block_iter_var->var);
    bool appears_in_old = old_vars.count(block_iter_var->var);
    if (appears_in_new != appears_in_old) {
      throw MakeScheduleError<ReindexCacheReadWriteNotMatchError>(
          mod, block, block_iter_var->var, old_indices, new_indices, is_cache_read, appears_in_old);
    }
    if (appears_in_new) {
      info->block_iter_vars.push_back(block_iter_var);
      info->block_iter_values.push_back(block_iter_value);
    }
  }

  auto iter_value_vars = collect_vars(info->block_iter_values);
  for (const StmtSRef& loop_sref : GetLoopsUnderScope(block_sref, info->loc_sref)) {
    const ForObj* loop = TVM_SREF_TO_FOR(loop_sref);
    if (iter_value_vars.count(loop->var)) {
      info->loop_vars.push_back(loop->var);
      info->loop_ranges.push_back(Range(loop->begin, loop->end));
    }
  }

  // Create new tensor
  ffi::ObjectPtr<TensorTypeObj> new_tensor =
      ffi::make_object<TensorTypeObj>(*old_tensor.ty().get());
  new_tensor->scope = runtime::MemoryScope::Get(storage_scope);
  new_tensor->shape = new_shape;
  TensorVar rebuilt = TensorVar(old_tensor->name + "_" + storage_scope,
                                TensorType(std::move(new_tensor)), old_tensor->loc);

  if (is_cache_read) {
    info->write_tensor = rebuilt;
    info->alloc = info->write_tensor;
  } else {
    info->read_tensor = rebuilt;
    info->alloc = info->read_tensor;
  }
}

/*! \brief Check whether given cache_region is a single point access. */
template <bool is_cache_read>
void CheckSinglePoint(ScheduleState self, const SBlock& block, const TensorRegion& cache_region) {
  bool single_point = true;
  for (const Range& range : cache_region->region) {
    PrimExpr ext_int_expr = range->extent;
    const auto* ext_int = ext_int_expr.as<IntImmObj>();
    if (!ext_int || ext_int->value != 1) {
      single_point = false;
    }
  }
  if (!single_point) {
    throw MakeScheduleError<NotSinglePointAccess>(self->mod, block, cache_region, is_cache_read);
  }
}

StmtSRef ReindexCacheRead(ScheduleState self, const StmtSRef& block_sref, int read_tensor_index,
                          const ffi::String& storage_scope, const IndexMap& index_map) {
  /*!
   * Check:
   *   - The index is in the array of block reading region
   *   - There is at most one block who write the tensor in the scope
   *
   * Mutate:
   *   - Allocate new cache tensor under the current scope.
   *   - Find the lowest ancestor of the block and ANY ONE of the consumers blocks.
   *   - Copy the tensor with the consumed region.
   */

  // Step 0. Check the input storage scope.
  CheckStorageScope(self, storage_scope);

  // Step 1. Check index, getting the target tensor and the parent scope
  SBlock block = ffi::GetRef<SBlock>(TVM_SREF_TO_SBLOCK(block_sref));
  SBlockRealize realize = GetSBlockRealize(self, block_sref);
  TensorVar read_tensor =
      GetNthAccessTensor(self, block, read_tensor_index, TensorIndexType::kRead);
  StmtSRef scope_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/true);

  // Step 2. Create CacheStageInfo
  ReindexCacheStageInfo info;
  info.read_tensor = read_tensor;
  info.consumer_blocks.insert(block_sref);

  // Step 3. Update cache stage info.
  ffi::Optional<TensorRegion> maybe_region = GetTensorRegionFromTensor(block->reads, read_tensor);
  TVM_FFI_ICHECK(maybe_region.has_value())
      << read_tensor << " should appear in the block's read region: " << block->reads;
  TensorRegion cache_region = maybe_region.value();
  if (ffi::Optional<StmtSRef> _write_block_sref =
          GetOnlyWriteBlock(self, scope_sref, read_tensor)) {
    // Case 1. The tensor is written inside the block.
    StmtSRef write_block_sref = _write_block_sref.value();
    // Find the producing region
    StmtSRef parent_sref = ffi::GetRef<StmtSRef>(write_block_sref->parent);
    // Detect insert position
    CacheLocDetector::Detect</*is_cache_read=*/true>(self, write_block_sref, scope_sref, &info);
  } else {
    // Case 2. The tensor is the input block for the scope.
    info.loc_sref = scope_sref;
    info.loc_pos = 0;
  }

  // Step 4. Check whether cache region is a single point.
  CheckSinglePoint</*is_cache_read=*/true>(self, block, cache_region);

  // Step 5. Collect ReindexCacheStageInfo and create new tensor.
  CollectReindexCacheStageInfoAndCreateTensor</*is_cache_read=*/true>(
      &info, self->mod, block_sref, storage_scope, index_map, block, realize, read_tensor,
      cache_region);

  // Step 6. Making new cache stage block and rewrite readers.
  SBlock cache_read_stage =
      MakeReindexCacheStage</*is_cache_read=*/true>(/*cache_region=*/cache_region,
                                                    /*info=*/&info,
                                                    /*storage_scope=*/storage_scope);
  Stmt new_scope = ReindexCacheReadRewriter::Rewrite(/*scope_sref=*/scope_sref, /*info=*/&info);

  // Step 7. Replacing and updating flags.
  self->Replace(scope_sref, new_scope, info.block_reuse);
  StmtSRef result_block_sref = self->stmt2ref.at(cache_read_stage.get());
  SBlockInfo& block_info = self->block_info[result_block_sref];
  block_info.affine_binding = CalculateAffineFlag(self, result_block_sref);
  block_info.region_cover = true;
  block_info.stage_pipeline = true;
  return result_block_sref;
}

StmtSRef ReindexCacheWrite(ScheduleState self, const StmtSRef& block_sref, int write_tensor_index,
                           const ffi::String& storage_scope, const IndexMap& index_map) {
  /*!
   * Check:
   *   - The index is in the array of block reading region
   *   - There is only one block who write the tensor in the scope
   *
   * Mutate:
   *   - Allocate new cache tensor under the current scope.
   *   - Find the lowest ancestor of the block and ANY ONE of the producer blocks.
   *   - Copy the tensor with the consumed region.
   */

  // Step 0. Check the input storage scope.
  CheckStorageScope(self, storage_scope);

  // Step 1. Checking index, getting the target tensor and the parent scope
  SBlock block = ffi::GetRef<SBlock>(TVM_SREF_TO_SBLOCK(block_sref));
  SBlockRealize realize = GetSBlockRealize(self, block_sref);
  TensorVar write_tensor =
      GetNthAccessTensor(self, block, write_tensor_index, TensorIndexType::kWrite);
  StmtSRef scope_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/true);

  // Step 2. Creating CacheStageInfo
  ReindexCacheStageInfo info;
  info.write_tensor = write_tensor;

  // Step 3. Check the only writer block.
  ffi::Optional<StmtSRef> only_write_block = GetOnlyWriteBlock(self, scope_sref, write_tensor);
  TVM_FFI_ICHECK(only_write_block.has_value());
  TVM_FFI_ICHECK_EQ(block_sref.get(), only_write_block.value().get());

  // Step 4. Find the producing region and insert position
  ffi::Optional<TensorRegion> maybe_region = GetTensorRegionFromTensor(block->writes, write_tensor);
  TVM_FFI_ICHECK(maybe_region.has_value())
      << write_tensor << " should appear in the block's write region";
  StmtSRef parent_sref = ffi::GetRef<StmtSRef>(block_sref->parent);
  // Detect insert position
  CacheLocDetector::Detect</*is_cache_read=*/false>(self, block_sref, scope_sref, &info);
  TensorRegion cache_region = maybe_region.value();

  CollectReindexCacheStageInfoAndCreateTensor</*is_cache_read=*/false>(
      &info, self->mod, block_sref, storage_scope, index_map, block, realize, write_tensor,
      cache_region);

  // Step 5. Check whether cache region is a single point.
  CheckSinglePoint</*is_cache_read=*/false>(self, block, cache_region);

  // Step 6. Making new cache stage block and rewrite readers.
  SBlock cache_write_stage =
      MakeReindexCacheStage</*is_cache_read=*/false>(/*cache_region=*/cache_region,
                                                     /*info=*/&info,
                                                     /*storage_scope=*/storage_scope);
  Stmt new_scope = ReindexCacheWriteRewriter::Rewrite(
      /*scope_sref=*/scope_sref,
      /*writer_block_sref=*/block_sref, /*info=*/&info);

  // Step 7. Replacing and updating flags.
  self->Replace(scope_sref, new_scope, info.block_reuse);
  StmtSRef result_block_sref = self->stmt2ref.at(cache_write_stage.get());
  SBlockInfo& block_info = self->block_info[result_block_sref];
  block_info.affine_binding = CalculateAffineFlag(self, result_block_sref);
  block_info.region_cover = true;
  block_info.stage_pipeline = true;
  return result_block_sref;
}

/*! \brief The schedule error that the target block doesn't both read&write target tensor. */
class NotReadWriteError : public ScheduleErrorContextObj {
 public:
  NotReadWriteError(IRModule mod, SBlock block, TensorVar tensor)
      : mod_(std::move(mod)), block_(std::move(block)), tensor_(std::move(tensor)) {}
  ffi::String FastErrorString() const final {
    return "ScheduleError: The target block does not both read & write target tensor.";
  }

  ffi::String DetailRenderTemplate() const final {
    return "The target block {0} does not both read & write target tensor {1}.";
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_, tensor_}; }
  IRModule mod_;
  SBlock block_;
  TensorVar tensor_;
};

ffi::Array<StmtSRef> CacheInplace(ScheduleState self, const StmtSRef& block_sref,
                                  int read_tensor_index, const ffi::String& storage_scope) {
  /*!
   * Do cache read then cache write
   */

  // Check 0. Check the input storage scope.
  CheckStorageScope(self, storage_scope);

  // Check 1. Check index, get the target tensor and the parent scope
  const SBlockObj* block = TVM_SREF_TO_SBLOCK(block_sref);
  TensorVar tensor = GetNthAccessTensor(self, ffi::GetRef<SBlock>(block), read_tensor_index,
                                        TensorIndexType::kRead);
  StmtSRef scope_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/false);

  // Check 3. Check required region cover for cache_read
  CheckRegionCover(self, scope_sref, tensor);

  // Check 4. Check if target block both read & write target tensor.
  const SBlockObj* rw_block = TVM_SREF_TO_SBLOCK(block_sref);
  ffi::Optional<TensorRegion> read_region = GetTensorRegionFromTensor(rw_block->reads, tensor);
  ffi::Optional<TensorRegion> write_region = GetTensorRegionFromTensor(rw_block->writes, tensor);
  if (!read_region.has_value() || !write_region.has_value()) {
    throw MakeScheduleError<NotReadWriteError>(self->mod, ffi::GetRef<SBlock>(rw_block), tensor);
  }

  ffi::Array<StmtSRef> results_block_sref;
  TensorVar new_tensor = WithScope(tensor, runtime::MemoryScope::Get(storage_scope));

  // Do cache read
  // Cache read step 0. Create CacheStageInfo
  CacheStageInfo info;
  info.read_tensor = tensor;
  // Create the corresponding tensor to be written for cache_read
  info.write_tensor = new_tensor;
  // Create the corresponding tensor allocation
  info.alloc = info.write_tensor;
  // Indicate which tensors should consume the cache.
  info.consumer_blocks.insert(block_sref);

  // Cache read step 1. Detect insert position
  CacheInplaceLocDetector::Detect(self, block_sref, scope_sref, &info);

  // Cache read step 2. Making new cache stage block and rewrite readers.
  SBlock cache_read_stage = MakeCacheStage(/*cache_region=*/read_region.value(), /*info=*/&info,
                                           /*storage_scope=*/storage_scope);
  Stmt new_scope = CacheReadRewriter::Rewrite(/*scope_sref=*/scope_sref, /*info=*/&info);

  // Cache read step 3. Replacing and updating flags for cache read.
  self->Replace(scope_sref, new_scope, info.block_reuse);
  StmtSRef result_block_sref = self->stmt2ref.at(cache_read_stage.get());
  SBlockInfo& block_info_read = self->block_info[result_block_sref];
  block_info_read.affine_binding = CalculateAffineFlag(self, result_block_sref);
  block_info_read.region_cover = true;
  block_info_read.stage_pipeline = false;
  results_block_sref.push_back(result_block_sref);

  // Do cache write
  // Cache write step 0. Update cache stage info for cache_read.
  info.read_tensor = new_tensor;
  // Create the corresponding tensor to be written, i.e. result of cache_write
  info.write_tensor = tensor;
  // Create the corresponding tensor allocation
  info.alloc = std::nullopt;
  info.consumer_blocks.clear();

  // Cache write step 1. Detect insert position
  CacheInplaceLocDetector::Detect(self, block_sref, scope_sref, &info);
  // insert after target block for cache write
  info.loc_pos += 1;

  // Cache write step 2. Making new cache stage block and rewrite readers.
  SBlock cache_write_stage = MakeCacheStage(/*cache_region=*/write_region.value(), /*info=*/&info,
                                            /*storage_scope=*/storage_scope);
  new_scope = CacheWriteRewriter::Rewrite(/*scope_sref=*/scope_sref,
                                          /*writer_block_sref=*/block_sref, /*info=*/&info);

  // Cache write step 4. Replacing and updating flags for cache write.
  self->Replace(scope_sref, new_scope, info.block_reuse);
  result_block_sref = self->stmt2ref.at(cache_write_stage.get());
  SBlockInfo& block_info_write = self->block_info[result_block_sref];
  block_info_write.affine_binding = CalculateAffineFlag(self, result_block_sref);
  block_info_write.region_cover = true;
  block_info_write.stage_pipeline = false;
  results_block_sref.push_back(result_block_sref);

  return results_block_sref;
}

StmtSRef ReIndex(ScheduleState self, const StmtSRef& block_sref, int tensor_index,
                 TensorIndexType tensor_index_type) {
  const SBlockObj* block_ptr = TVM_SREF_TO_SBLOCK(block_sref);
  SBlock block = ffi::GetRef<SBlock>(block_ptr);
  TensorVar tensor = GetNthAccessTensor(self, block, tensor_index, tensor_index_type);
  StmtSRef scope_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/true);
  sym::Analyzer analyzer;

  // Step 1. Collect the original indices and check there's only single pattern of related
  // Load/Store and the tensor is not accessed opaquely
  ffi::Array<PrimExpr> original_indices = ReIndexCollector::Collect(self->mod, tensor, block);
  // Simplify the indices if possible
  for (const IterVar& iter : block->iter_vars) {
    {
      auto context = analyzer->GetConstraintContext();
      context->Append(sym::RangeConstraint(iter->var, iter->dom.value()));
      analyzer->SyncConstraintContext(context);
    }
  }
  original_indices.MutateByApply(
      [&analyzer](const PrimExpr& expr) { return SimplifyNonTrivialExpr(expr, analyzer.get()); });

  // Collect block iters appearing in the original_indices
  std::unordered_set<Var> covered;
  auto walk_fn = [&covered](const Var& var) -> ffi::Expected<ffi::WalkResult> {
    covered.insert(var);
    return ffi::WalkResult::Advance();
  };
  for (const PrimExpr& index : original_indices) {
    ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(index, walk_fn);
  }

  // Step 2. Creating CacheStageInfo
  CacheStageInfo info;
  // Create the corresponding tensor to be read(write), i.e. the result of reindex read(write)
  if (tensor_index_type == TensorIndexType::kWrite) {
    info.read_tensor = CreateReindexTensor(tensor, block->iter_vars, covered);
    info.write_tensor = tensor;
    info.alloc = info.read_tensor;
  } else {
    info.read_tensor = tensor;
    info.write_tensor = CreateReindexTensor(tensor, block->iter_vars, covered);
    info.alloc = info.write_tensor;
  }

  // Step 3. Check the block belongs to a chain loop nesting under the scope,
  //         and get the insert location
  const StmtSRefObj* loop;
  for (loop = block_sref->parent; loop->parent != scope_sref.get();) {
    const ForObj* outer = loop->parent->StmtAs<ForObj>();
    const ForObj* inner = loop->StmtAs<ForObj>();
    TVM_FFI_ICHECK(outer != nullptr && inner != nullptr);
    TVM_FFI_ICHECK(outer->body->size() == 1 && outer->body->seq[0].get() == inner);
    loop = loop->parent;
  }

  info.loc_pos = loop->seq_index == -1 ? 0 : loop->seq_index;
  if (tensor_index_type == TensorIndexType::kWrite) {
    info.loc_pos++;
  }

  // Step 4. Making new reindex stage block and rewrite
  SBlock reindex_stage =
      MakeReIndexStage(block, &info, covered, original_indices, tensor_index, tensor_index_type);
  Stmt new_scope = ReIndexRewriter::Rewrite(scope_sref, block_sref, &info, covered);

  // Step 5. Replacing and updating flags
  self->Replace(scope_sref, new_scope, info.block_reuse);
  StmtSRef result_block_sref = self->stmt2ref.at(reindex_stage.get());
  SBlockInfo& block_info = self->block_info[result_block_sref];
  block_info.affine_binding = CalculateAffineFlag(self, result_block_sref);
  block_info.region_cover = true;
  block_info.stage_pipeline = true;
  return result_block_sref;
}

/******** Instruction Registration ********/

struct CacheReadTraits : public UnpackedInstTraits<CacheReadTraits> {
  static constexpr const char* kName = "CacheRead";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static SBlockRV UnpackedApplyToSchedule(Schedule sch, SBlockRV block,
                                          ffi::Array<SBlockRV> consumer_blocks,
                                          int read_tensor_index, ffi::String storage_scope) {
    return sch->CacheRead(block, read_tensor_index, storage_scope, consumer_blocks);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block,
                                      ffi::Array<ffi::String> consumer_blocks,
                                      int read_tensor_index, ffi::String storage_scope) {
    PythonAPICall py("cache_read");
    py.Input("block", block);
    py.Input("read_tensor_index", read_tensor_index);
    py.Input("storage_scope", storage_scope);
    // Only write out consumer blocks if provided.
    if (!consumer_blocks.empty()) {
      py.Input("consumer_blocks", consumer_blocks);
    }
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

struct CacheWriteTraits : public UnpackedInstTraits<CacheWriteTraits> {
  static constexpr const char* kName = "CacheWrite";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static SBlockRV UnpackedApplyToSchedule(Schedule sch, SBlockRV block,
                                          ffi::Array<SBlockRV> consumer_blocks,
                                          int write_tensor_index, ffi::String storage_scope) {
    return sch->CacheWrite(block, write_tensor_index, storage_scope, consumer_blocks);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block,
                                      ffi::Array<ffi::String> consumer_blocks,
                                      int write_tensor_index, ffi::String storage_scope) {
    PythonAPICall py("cache_write");
    py.Input("block", block);
    py.Input("write_tensor_index", write_tensor_index);
    py.Input("storage_scope", storage_scope);
    // Only write out consumer blocks if provided.
    if (!consumer_blocks.empty()) {
      py.Input("consumer_blocks", consumer_blocks);
    }
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

struct CacheInplaceTraits : public UnpackedInstTraits<CacheInplaceTraits> {
  static constexpr const char* kName = "CacheInplace";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static ffi::Array<SBlockRV> UnpackedApplyToSchedule(Schedule sch, SBlockRV block,
                                                      int read_tensor_index,
                                                      ffi::String storage_scope) {
    return sch->CacheInplace(block, read_tensor_index, storage_scope);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block,
                                      int read_tensor_index, ffi::String storage_scope) {
    PythonAPICall py("cache_inplace");
    py.Input("block", block);
    py.Input("read_tensor_index", read_tensor_index);
    py.Input("storage_scope", storage_scope);
    py.OutputList(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

struct ReIndexTraits : public UnpackedInstTraits<ReIndexTraits> {
  static constexpr const char* kName = "ReIndex";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static SBlockRV UnpackedApplyToSchedule(Schedule sch, SBlockRV block, int tensor_index,
                                          int tensor_index_type) {
    return sch->ReIndex(block, tensor_index, static_cast<TensorIndexType>(tensor_index_type));
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block,
                                      int tensor_index, int tensor_index_type) {
    PythonAPICall py("reindex");
    py.Input("block", block);
    std::ostringstream os;
    os << "(\"" << TensorIndexType2Str(static_cast<TensorIndexType>(tensor_index_type)) << "\", "
       << tensor_index << ")";
    py.Input("tensor", ffi::String(os.str()));
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

struct ReindexCacheReadTraits : public UnpackedInstTraits<ReindexCacheReadTraits> {
  static constexpr const char* kName = "ReindexCacheRead";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static SBlockRV UnpackedApplyToSchedule(Schedule sch, SBlockRV block, IndexMap index_map,
                                          int read_tensor_index, ffi::String storage_scope) {
    return sch->ReindexCacheRead(block, read_tensor_index, storage_scope, index_map);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block,
                                      IndexMap index_map, int read_tensor_index,
                                      ffi::String storage_scope) {
    PythonAPICall py("reindex_cache_read");
    py.Input("block", block);
    py.Input("read_tensor_index", read_tensor_index);
    py.Input("storage_scope", storage_scope);
    py.Input("index_map", index_map->ToPythonString());
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

struct ReindexCacheWriteTraits : public UnpackedInstTraits<ReindexCacheWriteTraits> {
  static constexpr const char* kName = "ReindexCacheWrite";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static SBlockRV UnpackedApplyToSchedule(Schedule sch, SBlockRV block, IndexMap index_map,
                                          int write_tensor_index, ffi::String storage_scope) {
    return sch->ReindexCacheWrite(block, write_tensor_index, storage_scope, index_map);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block,
                                      IndexMap index_map, int write_tensor_index,
                                      ffi::String storage_scope) {
    PythonAPICall py("reindex_cache_write");
    py.Input("block", block);
    py.Input("write_tensor_index", write_tensor_index);
    py.Input("storage_scope", storage_scope);
    py.Input("index_map", index_map->ToPythonString());
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

TVM_FFI_STATIC_INIT_BLOCK() {
  RegisterInstructionKind<CacheReadTraits>();
  RegisterInstructionKind<CacheWriteTraits>();
  RegisterInstructionKind<CacheInplaceTraits>();
  RegisterInstructionKind<ReIndexTraits>();
  RegisterInstructionKind<ReindexCacheReadTraits>();
  RegisterInstructionKind<ReindexCacheWriteTraits>();
}

}  // namespace s_tir
}  // namespace tvm
