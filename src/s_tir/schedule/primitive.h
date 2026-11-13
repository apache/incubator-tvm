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
#ifndef TVM_S_TIR_SCHEDULE_PRIMITIVE_H_
#define TVM_S_TIR_SCHEDULE_PRIMITIVE_H_

#include <tvm/base.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/s_tir/random_engine.h>
#include <tvm/s_tir/schedule/state.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/tensor_intrin.h>

#include <vector>

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

/******** Schedule: Sampling ********/
/*!
 * \brief Sample a random integer from a given range.
 * \param rand_state The pointer to schedule's random state.
 * \param min_inclusive The minimum value of the range, inclusive.
 * \param max_exclusive The maximum value of the range, exclusive.
 * \return The random integer sampled in the given range.
 */
TVM_DLL int32_t SampleInt(LinearCongruentialEngine::TRandState* rand_state, int32_t min_inclusive,
                          int32_t max_exclusive);
/*!
 * \brief Sample k random integers from given range without replacement, i.e, no duplication.
 * \param rand_state The pointer to schedule's random state
 * \param n The range is defined as 0 to n-1.
 * \param k The total number of samples.
 * \return The randomly selected samples from the n candidates.
 */
std::vector<int32_t> SampleWithoutReplacement(LinearCongruentialEngine::TRandState* rand_state,
                                              int32_t n, int32_t k);
/*!
 * \brief Sample once category from candidates according to the probability weights.
 * \param rand_state The pointer to schedule's random state
 * \param candidates The candidates
 * \param probs The probability distribution of the candidates
 * \param decision The sampling decision, if any
 * \return The random variable sampled from candidates
 */
TVM_DLL int64_t SampleCategorical(LinearCongruentialEngine::TRandState* rand_state,
                                  const ffi::Array<int64_t>& candidates,
                                  const ffi::Array<FloatImm>& probs,
                                  ffi::Optional<int64_t>* decision);
/*!
 * \brief Create a sampling function that does multinomial sampling.
 * \param rand_state The random state.
 * \param weights The weights for multinomial sampling.
 * \return The multinomial sampling function.
 */
TVM_DLL std::function<int32_t()> MakeMultinomialSampler(
    LinearCongruentialEngine::TRandState* rand_state, const std::vector<double>& weights);
/*!
 * \brief Sample the factors to perfect tile a specific loop
 * \param rand_state The random state
 * \param extent The loop extent to be tiled
 * \param n_split The number of tiles to be sampled
 * \return A list of length `n`, the random perfect tile sizes sampled
 */
TVM_DLL std::vector<int64_t> SamplePerfectTile(LinearCongruentialEngine::TRandState* rand_state,  //
                                               int32_t extent, int32_t n_splits);
/*!
 * \brief Sample the factors to perfect tile a specific loop
 * \param rand_state The random state
 * \param extent The loop extent to be tiled
 * \param n_split The number of tiles to be sampled
 * \param max_innermost_factor The maximum tile size allowed to be sampled in the innermost loop
 * \return A list of length `n`, the random perfect tile sizes sampled
 */
TVM_DLL std::vector<int64_t> SamplePerfectTile(LinearCongruentialEngine::TRandState* rand_state,  //
                                               int32_t extent, int32_t n_split,
                                               int32_t max_innermost_factor);
/*!
 * \brief Sample the factors to perfect tile a specific loop
 * \param rand_state The random state
 * \param loop_sref The loop to be tiled
 * \param n_split The number of tiles to be sampled
 * \param max_innermost_factor The maximum tile size allowed to be sampled in the innermost loop
 * \param decision The sampling decision
 * \return A list of length `n`, the random perfect tile sizes sampled
 */
TVM_DLL std::vector<int64_t> SamplePerfectTile(LinearCongruentialEngine::TRandState* rand_state,  //
                                               const tirx::StmtSRef& loop_sref, int32_t n_split,
                                               int32_t max_innermost_factor,
                                               ffi::Optional<ffi::Array<int64_t>>* decision);
/*!
 * \brief Sample the factors to a partitioned tile for a specific loop
 *
 *  The sampled tile size will be partitioned into two parts. The second part has a guarantee
 *  that their extent's product have a factor of `innerpart_factor`. The first part is loops at
 *  [0, partition_pos); the second part is loops at [partition_pos, n) and we will have
 *  `innerpart_factor` | prod_{l=partition_pos}^{n-1} l.extent
 *
 * \param rand_state The random state
 * \param extent The loop extent to be tiled
 * \param n_split The number of tiles to be sampled
 * \param partition_pos The position to partition tiles to two parts
 * \param innerpart_factor The factor of the second part\
 * \return A list of length `n`, the random partitioned tile sizes sampled
 */
TVM_DLL std::vector<int64_t> SamplePartitionedTile(
    LinearCongruentialEngine::TRandState* rand_state,  //
    int32_t extent, int32_t n_split, int32_t partition_pos, int32_t innerpart_factor);
/*!
 * \brief Sample the factors to a partitioned tile for a specific loop
 *
 *  The sampled tile size will be partitioned into two parts. The second part has a guarantee
 *  that their extent's product have a factor of `innerpart_factor`. The first part is loops at
 *  [0, partition_pos); the second part is loops at [partition_pos, n) and we will have
 *  `innerpart_factor` | prod_{l=partition_pos}^{n-1} l.extent
 *
 * \param rand_state The random state
 * \param loop_sref The loop to be tiled
 * \param n_split The number of tiles to be sampled
 * \param partition_pos The position to partition tiles to two parts
 * \param innerpart_factor The factor of the second part
 * \param decision The sampling decision
 * \return A list of length `n`, the random partitioned tile sizes sampled
 */
TVM_DLL std::vector<int64_t> SamplePartitionedTile(
    LinearCongruentialEngine::TRandState* rand_state,  //
    const tirx::StmtSRef& loop_sref, int32_t n_split, int32_t partition_pos,
    int32_t innerpart_factor, ffi::Optional<ffi::Array<int64_t>>* decision);
/*!
 * \brief Sample a compute-at location of the given block
 * \param self The schedule state
 * \param rand_state The random state
 * \param block_sref The sref of the block whose compute-at location is to be sampled
 * \param decision The sampling decision
 * \return The sampled loop where the input block is to be computed at
 */
TVM_DLL tirx::StmtSRef SampleComputeLocation(s_tir::ScheduleState self,
                                             LinearCongruentialEngine::TRandState* rand_state,
                                             const tirx::StmtSRef& block_sref,
                                             ffi::Optional<int64_t>* decision);

/******** Schedule: Get blocks & loops ********/
/*!
 * \brief Retrieves blocks in a specific function with its name
 * \param self The schedule state
 * \param name The name of the blocks to be retrieved
 * \param gvar The function to be retrieved
 * \return A list of blocks with the specific name
 */
ffi::Array<StmtSRef> GetSBlocks(const ScheduleState& self, const ffi::String& name,
                                const GlobalVar& gv);
/*!
 * \brief Gets the parent loops of the block in its scope, from outer to inner
 * \param self The schedule state
 * \param block_sref The query block
 * \return A list of loops above the given block in its scope, from outer to inner
 */
ffi::Array<StmtSRef> GetLoops(const StmtSRef& block_sref);
/*!
 * \brief Get the leaf blocks of a specific block/loop
 * \param self The schedule state
 * \param parent_sref The query block/loop
 * \return A list of leaf blocks inside a specific block/loop
 */
ffi::Array<StmtSRef> GetChildBlocks(const ScheduleState& self, const StmtSRef& parent_sref);
/*!
 * \brief Get the producers of a specific block
 * \param self The schedule state
 * \param block_sref The block in the query
 * \return A list of blocks, the producers of the given block
 */
ffi::Array<StmtSRef> GetProducers(const ScheduleState& self, const StmtSRef& block_sref);
/*!
 * \brief Get the consumers of a specific block
 * \param self The schedule state
 * \param block_rv The block in the query
 * \return A list of blocks, the consumers of the given block
 */
ffi::Array<StmtSRef> GetConsumers(const ScheduleState& self, const StmtSRef& block_sref);
/*!
 * \brief Get the list of output blocks within the given scope
 * An output block is a block which has atleast one tensor being written
 * to, but is not allocated within the Function
 * \param scope_block_rv The scope block from which output blocks are collected
 * \return A list of all blocks that write to some output tensor
 * block
 */
ffi::Array<StmtSRef> GetOutputBlocks(const ScheduleState& self, const StmtSRef& scope_sref);
/******** Schedule: Transform loops ********/
/*!
 * Split a loop into a list of consecutive loops. It requires:
 * 1) The loop can't have annotation or thread binding.
 * 2) The loop must start with 0.
 * \param self The state of the schedule
 * \param loop_sref The sref to the loop being split
 * \param factors The splitting factors
 * \param preserve_unit_iters Whether or not to preserve unit iterators in block bindings
 * \param disable_predication If enabled, don't create a predicate for guarding the
 *      loop. This can be useful when splitting with scalable factors that the schedule writer
 *      knows are divisible by the loop bound.
 *      Warning: enabling this feature may result in incorrect code generation if not used
 * carefully. \return An array of srefs to the loops after splitting
 */
TVM_DLL ffi::Array<StmtSRef> Split(ScheduleState self, const StmtSRef& loop_sref,
                                   const ffi::Array<PrimExpr>& factors, bool preserve_unit_iters,
                                   bool disable_predication);

/*!
 * Partition a loop into a list of consecutive loops. It requires:
 * 1) The loop can't have annotation or thread binding.
 * \param self The state of the schedule
 * \param loop_sref The sref to the loop being partition
 * \param factors The partitioning factors
 * \param preserve_unit_iters Whether or not to preserve unit iterators in block bindings
 * \return An array of srefs to the loops after partitioning
 */
TVM_DLL ffi::Array<StmtSRef> LoopPartition(ScheduleState self, const StmtSRef& loop_sref,
                                           const ffi::Array<PrimExpr>& factors,
                                           bool preserve_unit_iters);

/*!
 * \brief Merge a list of loops into one. The loops under their LCA requires:
 * 1) Under the same scope
 * 2) Can't have annotations or thread bindings
 * 3) Start with 0 and have same extent and same nesting depth
 * 4) From target loop to their LCA, the inner loop must be the only child of the outer loop
 * \param self The state of the schedule
 * \param loop_srefs An array of srefs to the loops to be merged
 * \return The new loop after merge
 */
TVM_DLL StmtSRef Merge(ScheduleState self, const ffi::Array<StmtSRef>& loop_srefs);

/*!
 * \brief Fuse a list of consecutive loops into one. It requires:
 * 1) The loops can't have annotations or thread bindings.
 * 2) The inner loop must be the only child of the outer loop.
 * 3) All loops must start with 0.
 * 4) The domain of a loop to be fused cannot depend on another loop to be fused.
 * \param self The state of the schedule
 * \param loop_srefs An array of srefs to the loops to be fused
 * \param preserve_unit_iters Whether or not to preserve unit iterators in block bindings
 * \return The sref to the fused loop
 */
TVM_DLL StmtSRef Fuse(ScheduleState self, const ffi::Array<StmtSRef>& loop_srefs,
                      bool preserve_unit_loops);
/*!
 * \brief Reorder a list of loops. It doesn't require the loops to be consecutive.
 * It requires:
 * 1) The loops are in the same chain. That means: the loops can be ordered to [l_1, l_2, ... ,
 *     l_n] where l_i is an ancestor of l_{i+1} and there are only single-branch loops between
 *     l_1 and l_n (which also indicates they are under the same scope).
 * 2) After reordering, the domain of an outer loop cannot depend on any of the inner loops.
 * 3) For every block under the loop nests, its block binding must be affine, and the block
 *    variables must be either data parallel or reduction.
 * 4) No duplicated loops are allowed in the arguments.
 * \param self The state of the schedule
 * \param ordered_loop_srefs An array of srefs which indicates the new order of loops
 */
TVM_DLL void Reorder(ScheduleState self, const ffi::Array<StmtSRef>& ordered_loop_srefs);

/*!
 * \brief Reorder itervars inside a block.
 * \param self The state of the schedule.
 * \param block_sref The sref of block to be transformed.
 * \param new_order The new itervar order.
 */
TVM_DLL void ReorderBlockIterVar(ScheduleState self, const StmtSRef& block_sref,
                                 const ffi::Array<int64_t>& new_order);

/*!
 * \brief Create a new unit loop on top of the specific block or loop.
 * \param sref The block/loop above which the new thread_binding loop is created
 * \param extent The extent of the new thread_binding loop
 * \param thread_axis The thread axis of the new thread_binding loop
 * \param attrs Extra loop attributes
 * \return The new thread_binding loop
 */
TVM_DLL StmtSRef AddUnitLoop(ScheduleState self, StmtSRef sref);

/******** Schedule: Manipulate ForKind ********/
/*!
 * \brief Parallelize the input loop. It requires:
 * 1) The scope block that the loop is in should have stage-pipeline property
 * 2) All the blocks under the loop are complete blocks or reduction blocks, and have affine
 * bindings
 * 3) For each block under the loop, the loop can only be contained in data-parallel block iters'
 * bindings
 * \param self The state of the schedule
 * \param loop_sref The sref of the loop to be parallelized
 */
TVM_DLL void Parallel(ScheduleState self, const StmtSRef& loop_sref);
/*!
 * \brief Vectorize the input loop. It requires:
 * 1) The scope block that the loop is in should have stage-pipeline property
 * 2) All the blocks under the loop are complete blocks or reduction blocks, and have affine
 * bindings
 * 3) For each block under the loop, the loop can only be contained in data-parallel block iters'
 * bindings
 * \param self The state of the schedule
 * \param loop_sref The sref of the loop to be vectorized
 */
TVM_DLL void Vectorize(ScheduleState self, const StmtSRef& loop_sref);
/*!
 * \brief Bind the input loop to the given thread axis. It requires:
 * 1) The scope block that the loop is in should have stage-pipeline property
 * 2) All the blocks under the loop are complete blocks or reduction blocks, and have affine
 * bindings
 * 3) For each block under the loop, if the thread axis starts with "threadIdx`, the loop can only
 * be contained in data-parallel block iter and reduction block iters' bindings. Otherwise the
 * loop can only be contained in data-parallel block iters' bindings
 * \param self The state of the schedule
 * \param loop_sref The sref of the loop to be bound to the thread axis
 * \param thread_axis The thread axis to be bound to the loop
 */
TVM_DLL void Bind(ScheduleState self, const StmtSRef& loop_sref, const ffi::String& thread_axis);
/*!
 * \brief Unroll the input loop. It requires nothing
 * \param self The state of the schedule
 * \param loop_sref The loop to be unrolled
 */
TVM_DLL void Unroll(ScheduleState self, const StmtSRef& loop_sref);
/******** Schedule: Insert cache stages ********/
/*!
 * \brief Create a block that reads a tensor region into a read cache. It requires:
 * 1) There is at most one block who writes the tensor in the scope.
 * 2) The scope block have stage-pipeline property.
 * \param self The state of the schedule
 * \param block_sref The consumer block of the target tensor.
 * \param read_tensor_index The index of the tensor in block's read region.
 * \param storage_scope The target storage scope.
 * \param consumer_blocks Array of blocks that consume the cache.
 * \return The cache stage block.
 */
TVM_DLL StmtSRef CacheRead(ScheduleState self, const StmtSRef& block_sref, int read_tensor_index,
                           const ffi::String& storage_scope,
                           const ffi::Array<StmtSRef> consumer_blocks = {});
/*!
 * \brief Create a block that writes a tensor region into a write cache. It requires:
 * 1) There is only one block that writes the target tensor.
 * 2) The scope block have stage-pipeline property.
 * \param self The state of the schedule
 * \param block_sref The producer of the tensor
 * \param write_tensor_index The index of the tensor in block's write region
 * \param storage_scope The target storage scope
 * \param consumer_blocks Array of blocks that consume the cache.
 * \return The cache stage block.
 */
TVM_DLL StmtSRef CacheWrite(ScheduleState self, const StmtSRef& block_sref, int write_tensor_index,
                            const ffi::String& storage_scope,
                            const ffi::Array<StmtSRef> consumer_blocks = {});
/*!
 * \brief Create a block that reads a tensor region into a read cache. It requires:
 * 1) There is at most one block who writes the tensor in the scope.
 * 2) The scope block have stage-pipeline property.
 * Compared to cache read, the indices to access allocated cache tensor is customized by user.
 * \param self The state of the schedule
 * \param block_sref The consumer block of the target tensor.
 * \param read_tensor_index The index of the tensor in block's read region.
 * \param storage_scope The target storage scope.
 * \param index_map User defined indices to access allocated cache tensor, maps from block iter
 * vars.
 * \return The cache stage block.
 */
TVM_DLL StmtSRef ReindexCacheRead(ScheduleState self, const StmtSRef& block_sref,
                                  int read_tensor_index, const ffi::String& storage_scope,
                                  const IndexMap& index_map);
/*!
 * \brief Create a block that writes a tensor region into a write cache. It requires:
 * 1) There is only one block that writes the target tensor.
 * 2) The scope block have stage-pipeline property.
 * Compared to cache write, the indices to access allocated cache tensor is customized by user.
 * \param self The state of the schedule
 * \param block_sref The producer of the tensor
 * \param write_tensor_index The index of the tensor in block's write region
 * \param storage_scope The target storage scope
 * \param index_map User defined indices to access allocated cache tensor, maps from block iter
 * vars.
 * \return The cache stage block.
 */
TVM_DLL StmtSRef ReindexCacheWrite(ScheduleState self, const StmtSRef& block_sref,
                                   int write_tensor_index, const ffi::String& storage_scope,
                                   const IndexMap& index_map);

/*!
 *!
 * \brief Create 2 blocks that read&write a tensor region into a read/write cache.
 * It requires the target block both read & write the target tensor.
 * \param self The state of the schedule
 * \param block_sref The target block operates on the target tensor.
 * \param read_tensor_index The index of the tensor in block's read region.
 * \param storage_scope The target storage scope
 * \return The cache stage blocks, cache read block together with cache write block.
 */
TVM_DLL ffi::Array<StmtSRef> CacheInplace(ScheduleState self, const StmtSRef& block_sref,
                                          int read_tensor_index, const ffi::String& storage_scope);
/*!
 * \brief Create a block to cache precomputed index for later use.
 * if there is no index computation, keep unchanged.
 * \param block_sref The target block
 * \param storage_scope The storage scope of cached block
 * \param cse_thresh The repeat threshold that determines a common sub expr
 * \return The cache stage block.
 */
TVM_DLL ffi::Array<StmtSRef> CacheIndex(ScheduleState self, const StmtSRef& block_sref,
                                        const ffi::String& storage_scope, int cse_thresh);
/*!
 *!
 * \brief Create a block that read/write a tensor region into a read/write cache with reindexing.
 * The layout of the cache will be the same as by the iterators of the block that reads/writes the
 * tensor. It requires:
 * 1) There is only one block who reads/writes the target tensor
 * 2) There is only one tensor load/store of this tensor in the block
 * \param self The state of the schedule
 * \param block_sref The block operates on the target tensor.
 * \param tensor_index The index of the tensor in block's read or write region.
 * \param tensor_index_type The type of the tensor index, kRead or kWrite.
 * \return The reindex stage block.
 */
TVM_DLL StmtSRef ReIndex(ScheduleState self, const StmtSRef& block_sref, int tensor_index,
                         TensorIndexType tensor_index_type);

/******** Schedule: Data movement ********/

TVM_DLL StmtSRef ReadAt(ScheduleState self, const StmtSRef& loop_sref, const StmtSRef& block_sref,
                        int read_tensor_index, const ffi::String& storage_scope);

TVM_DLL StmtSRef WriteAt(ScheduleState self, const StmtSRef& loop_sref, const StmtSRef& block_sref,
                         int write_tensor_index, const ffi::String& storage_scope);

/******** Schedule: Compute location ********/
/*!
 * \brief Move a producer block under the specific loop, and regenerate the
 * loops induced by the block so that the tensor region produced by the producer block could
 * cover those regions consumed by its consumer blocks under the given loop. It requires:
 * 1) `block` and `loop` are under the same scope, `loop` is not the ancestor of `block`
 * 2) The scope block has stage-pipeline property
 * 3) The subtree of the scope block, where the given block is in, satisfies the compact dataflow
 * condition. i.e. all the blocks in the scope block's subtree must be either complete block or
 * reduction block
 * 4) The block is not an output block with regard to the scope block, i.e. the tensors written by
 * the block are allocated under the scope block
 * 5) All the consumers of the block are under the given loop
 *
 * \param self The schedule state
 * \param block_sref The block to be moved
 * \param loop_sref The loop where the block to be moved to
 * \param index The block index of the loop body subtree blocks:
 * - `index = -1` means inserted into the last possible insertion point;
 * - `index = -2` means inserted into the first possible insertion point;
 * - Otherwise, `index` is a nonnegative number that indicates the insertion point
 */
TVM_DLL void ComputeAt(ScheduleState self, const StmtSRef& block_sref, const StmtSRef& loop_sref,
                       bool preserve_unit_loops, int index = -1);
/*!
 * \brief Move a consumer block under the specific loop, and regenerate the
 * loops induced by the block so that the tensor region consumed by the consumer block could
 * cover those regions produced by its producer blocks under the given loop. It requires:
 * 1) `block` and `loop` are under the same scope, `loop` is not the ancestor of `block`
 * 2) The scope block has stage-pipeline property
 * 3) The subtree of the scope block, where the given block is in, satisfies the compact dataflow
 * condition. i.e. all the blocks in the scope block's subtree must be either complete block or
 * reduction block
 * 4) All the producers of the block are under the given loop
 *
 * \param self The schedule state
 * \param block_sref The block to be moved
 * \param loop_sref The loop where the block to be moved to
 * \param preserve_unit_loops Whether to keep the trivial loops whose extents are 1
 * \param index The block index of the loop body subtree blocks:
 * - `index = -1` means inserted into the last possible insertion point;
 * - `index = -2` means inserted into the first possible insertion point;
 * - Otherwise, `index` is a nonnegative number that indicates the insertion point
 */
TVM_DLL void ReverseComputeAt(ScheduleState self, const StmtSRef& block_sref,
                              const StmtSRef& loop_sref, bool preserve_unit_loops, int index = -1);
/*!
 * \brief Inline a block into its consumer(s). It requires:
 * 1) The block is a complete non-root block, which only produces one tensor
 * 2) The block must not be the only leaf in the scope.
 * 3) The body of the block must be a TensorStore statement in the form of,
 *    A[i, j, k, ...] = ...
 * where the indices of the LHS are all distinct atomic variables,
 * and no variables other than those indexing variables are allowed in the statement.
 * \param self The state of the schedule
 * \param block_sref The sref to the block to be inlined to its consumer(s)
 */
TVM_DLL void ComputeInline(ScheduleState self, const StmtSRef& block_sref);
/*!
 * \brief Inline a block into its only producer. It requires:
 * 1) The block is a complete non-root block, which only produces and consumers one tensor
 * 2) The block must not be the only leaf in the scope.
 * 3) The only producer of the block is a read-after-write producer and a complete non-root block
 * 4) The body of the block must be a TensorStore statement in the form of,
 *    B[f(i, j, k, ...)] = g(i, j, k, A[i, j, k, ...] ...)
 * where the indices of each `TensorLoad` on the RHS are all distinct atomic variables,
 * and no variables other than those indexing variables are allowed in the statement.
 * \param self The state of the schedule
 * \param block_sref The sref to the block to be inlined to its producer
 */
TVM_DLL void ReverseComputeInline(ScheduleState self, const StmtSRef& block_sref);
/*!
 * \brief Fuse an epilogue block into a reduction block
 * \param self The state of the schedule
 * \param reduction_block_sref The sref to the reduction block
 * \param epilogue_block_sref The sref to the epilogue block to be fused
 */
TVM_DLL void FuseReductionEpilogue(ScheduleState self, const StmtSRef& reduction_block_sref,
                                   const StmtSRef& epilogue_block_sref);
/******** Schedule: Reduction ********/
/*!
 * \brief Decompose a reduction block into two separate blocks.
 * a) The init block, which is translated from the init statement of the reduction block;
 * b) The update block, which is the original block without init statement.
 *
 * The init block is inserted right before the given loop.
 *
 * The schedule primitive requires:
 * 1) The input block is a reduction block.
 * 2) The input loop is the ancestor of the block.
 * 3) The input loop is not lower than all the loops related to reduce block var.
 * \param block_rv The reduction block to be decomposed
 * \param loop_rv The loop above which the init block is inserted before.
 * \return The init block
 */
TVM_DLL StmtSRef DecomposeReduction(ScheduleState self, const StmtSRef& block_sref,
                                    const StmtSRef& loop_sref);
/*!
 * \brief Factor a reduction block by the specified loop
 * \details See python/tvm/s_tir/schedule/schedule.py
 * \param self The state of the schedule
 * \param loop_sref The loop outside block for which we want to do rfactor
 * \param factor_axis The position where the new dimension is placed in the new introduced rfactor
 *                    tensor. Suppose the original reduction block writes to tensor `B` with
 *                    ndim(B) dimensions, then `factor_axis` should be in range `[-ndim(B) - 1,
 *                    ndim(B)]`, and the negative index will be normalized to a non-negative one
 * \return The sref of the rfactor block
 */
TVM_DLL StmtSRef RFactor(ScheduleState self, const StmtSRef& loop_sref, int factor_axis);
/******** Schedule: SBlock annotation ********/

/*!
 * \brief Set alignment requirement for specific dimension such that
 *        stride[axis] == k * factor + offset for some k. This is useful to set memory layout for
 *        more friendly memory access pattern. For example, we can set alignment to be factor=2,
 *        offset=1 to avoid bank conflict for thread access on higher dimension in GPU shared
 *        memory.
 * \param self The state of the schedule
 * \param block_sref The producer block of the tensor
 * \param tensor_index The index of the tensor in block's write region
 * \param axis The dimension to be specified for alignment
 * \param factor The factor multiple of alignment
 * \param offset The required offset factor
 */
TVM_DLL void StorageAlign(ScheduleState self, const StmtSRef& block_sref, int tensor_index,
                          int axis, int factor, int offset);
/*!
 * \brief Set the storage scope of a tensor, where the tensor is specified by the a block and a
 * write-index
 * \param self The state of the schedule
 * \param block_sref The sref of the producer block of the tensor
 * \param tensor_index The index of the tensor in block's write region
 * \param storage_scope The storage scope to be set
 */
TVM_DLL void SetScope(ScheduleState self, const StmtSRef& block_sref, int tensor_index,
                      const ffi::String& storage_scope);
/*!
 * \brief Set the data type of a tensor, where the tensor is specified by a block and a
 * write-index
 * \note This schedule primitive is unsafe and may change correctness of program because of
 *   type conversion, please use with caution.
 * \param self The state of the schedule
 * \param block_sref The sref of the producer block of the tensor
 * \param tensor_index The index of the tensor in block's write region
 * \param dtype The data type to be set
 */
TVM_DLL void UnsafeSetDType(ScheduleState self, const StmtSRef& block_sref, int tensor_index,
                            const ffi::String& dtype);
/******** Schedule: Blockize & Tensorize ********/

/*!
 * \brief Convert the subtree rooted at a specific loop into a block.
 * \param self The state of the schedule
 * \param loop_sref The root of the subtree
 * \param preserve_unit_iters Whether or not to preserve unit iterators in block bindings
 * \return The new block
 */
TVM_DLL StmtSRef Blockize(ScheduleState self, const StmtSRef& loop_sref, bool preserve_unit_iters);

/*!
 * \brief Convert specific blocks into a nested block.
 * \param self The state of the schedule
 * \param blocks The target blocks to construct the new block
 * \param preserve_unit_iters Whether or not to preserve unit iterators in block bindings
 * \return The new block
 */
TVM_DLL StmtSRef Blockize(ScheduleState self, const ffi::Array<StmtSRef>& blocks,
                          bool preserve_unit_iters);

/*!
 * \brief Tensorize the computation enclosed by loop with the tensor intrinsic.
 * \param self The state of the schedule
 * \param block_or_loop_sref The block or loop to be tensorized.
 * \param intrin The tensor intrinsic.
 * \param preserve_unit_iters Whether or not to preserve unit iterators in block bindings
 */
TVM_DLL void Tensorize(ScheduleState self, const StmtSRef& block_or_loop_sref,
                       const TensorIntrin& intrin, bool preserve_unit_iters);

/******** Schedule: Annotation ********/
/*!
 * \brief Annotate a block/loop with a key value pair
 * \param self The state of the schedule
 * \param sref The block/loop sref to be annotated
 * \param ann_key The annotation key
 * \param ann_val The annotation value
 */
TVM_DLL void Annotate(ScheduleState self, const StmtSRef& sref, const ffi::String& ann_key,
                      const Any& ann_val);
/*!
 * \brief Unannotate a block/loop's annotation with key ann_key
 * \param self The state of the schedule
 * \param sref The block/loop to be unannotated
 * \param ann_key The annotation key
 */
TVM_DLL void Unannotate(ScheduleState self, const StmtSRef& sref, const ffi::String& ann_key);

/******** Schedule: Layout transformation ********/
/*!
 * \brief Apply a transformation represented by IndexMap to tensor
 * \details The indices and the access region to the target tensor is transformed by the given
 * index_map. The index_map is also used to infer the new shape of the tensor. Tensor must be
 * one of the parameter of the function, or allocated in some blocks (it cannot be a tensor
 * subregion created via match_tensor).
 * \param self The state of the schedule
 * \param block_sref The block sref that accesses the target tensor.
 * \param tensor_index The index of the tensor in block's read or write region.
 * \param tensor_index_type The type of the tensor index, kRead or kWrite.
 * \param index_map The transformation to apply.
 * \param pad_value The value to write into padding introduced by the transformation.
 * \param assume_injective_transform If set to true, the schedule primitive will assume the
 * index_map is injective and skip checking overlapping of the mapped indices. This can be useful
 * for complicated index_map that the analysis does not cover. It is the callers' responsibility
 * to ensure the index map is injective, otherwise, the correctness of the schedule is not
 * guaranteed.
 */
TVM_DLL void TransformLayout(ScheduleState self, const StmtSRef& block_sref, int tensor_index,
                             TensorIndexType tensor_index_type, const IndexMap& index_map,
                             const ffi::Optional<IndexMap>& pad_value,
                             bool assume_injective_transform);

/*!
 * \brief Apply a transformation represented by IndexMap to block
 * \details The block iters and the block body are transformed by the given index_map.
 * Outer loops corresponding to each new block iter are regenerated.
 * The index_map is required to be bijective affine since we need its inverse mapping.
 * \param self The state of the schedule
 * \param block_sref The block sref that refers to the block to be transformed
 * \param index_map The transformation to apply.
 */
TVM_DLL void TransformBlockLayout(ScheduleState self, const StmtSRef& block_sref,
                                  const IndexMap& index_map);

/******** Schedule: Padding ********/
/*!
 * \brief Decompose a padding block into a block filling const pad values and a block
 * writing in-bound values.
 * \param block_sref The block sref that match the padding pattern.
 * \param loop_sref The loop above which the const filling block is inserted before.
 * \return The padding value filling block sref.
 */
TVM_DLL StmtSRef DecomposePadding(ScheduleState self, const StmtSRef& block_sref,
                                  const StmtSRef& loop_sref);

/*!
 * \brief Pad the computation of Einsum.
 * \param self The state of the schedule
 * \param block_sref The block sref that matches the Einsum pattern.
 * \param padding The padding for each block iter.
 */
TVM_DLL void PadEinsum(ScheduleState self, const StmtSRef& block_sref,
                       const ffi::Array<int64_t>& padding);
/******** Schedule: Tensor transformation ********/
/*!
 * \brief Compute the target tensor via rolling buffering.
 * \details This primitive selects the outermost rollable axis with a positive bound overlap that
 * appears in the block's ancestor loops as `rolling axis`, fold and circularize the tensor along
 * the rolling dimension, append block predicate to avoid recomputing overlapping elements.
 * It requires:
 * 1) The tensor to be an intermediate tensor defined via `alloc_tensor`.
 * 2) The LCA of the producer and consumer of the tensor is a for loop, typically,
 *    the producer and consumer of the tensor are cascaded through compute_at.
 * 3) The access region of the tensor has at least one dimension that contains
 *    a positive bound overlap.
 * \param block_rv The producer block of the tensor.
 * \param write_tensor_index The index of the tensor in block's write region.
 */
TVM_DLL void RollingBuffer(ScheduleState self, const StmtSRef& block_sref, int write_tensor_index);
/******** Schedule: Misc ********/

/*!
 * \brief Hide some tensor access in the given block.
 * \param self The state of the schedule.
 * \param block_sref The sref of the block we hide access.
 * \param tensor_type The tensor type: read/write
 * \param tensor_index_array The array of tensor indices we hide access.
 */
TVM_DLL void UnsafeHideTensorAccess(ScheduleState self, const StmtSRef& block_sref,
                                    const ffi::String& tensor_type,
                                    const ffi::Array<int64_t>& tensor_index_array);

/*!
 * \brief Annotate the read or write region of a specific tensor in a block
 * \param self The state of the schedule
 * \param block_sref The sref of the block to be annotated
 * \param tensor_index The index of the tensor in block's read or write region
 * \param tensor_index_type The type of the tensor index, kRead or kWrite
 * \param index_map The IndexMap that defines the new read or write region for the tensor
 */
TVM_DLL void AnnotateTensorAccess(ScheduleState self, const StmtSRef& block_sref, int tensor_index,
                                  TensorIndexType tensor_index_type, const IndexMap& index_map);
}  // namespace s_tir
}  // namespace tvm

#endif  // TVM_S_TIR_SCHEDULE_PRIMITIVE_H_
