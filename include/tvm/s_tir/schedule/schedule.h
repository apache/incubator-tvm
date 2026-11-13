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
#ifndef TVM_S_TIR_SCHEDULE_SCHEDULE_H_
#define TVM_S_TIR_SCHEDULE_SCHEDULE_H_

#include <tvm/base.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/s_tir/random_engine.h>
#include <tvm/s_tir/schedule/state.h>
#include <tvm/s_tir/schedule/trace.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/index_map.h>

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

/*! \brief The level of detailed error message rendering */
enum class ScheduleErrorRenderLevel : int32_t {
  /*! \brief Render a detailed error message */
  kDetail = 0,
  /*! \brief Render the error in fast mode */
  kFast = 1,
  /*! \brief No error message at all */
  kNone = 2,
};

/*! \brief Type of tensor index */
enum class TensorIndexType : int32_t {
  /*! \brief Index of a read tensor */
  kRead = 0,
  /*! \brief Index of a written tensor */
  kWrite = 1,
};

/**************** Random variable: SBlockRV ****************/

/*! \brief A random variable that evaluates to a TensorIR block */
class SBlockRVObj : public ffi::Object {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SBlockRVObj>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.SBlockRV", SBlockRVObj, ffi::Object);
};

/*!
 * \brief Managed reference to SBlockRVObj
 * \sa SBlockRVObj
 */
class SBlockRV : public ffi::ObjectRef {
 public:
  /*! \brief Constructor */
  TVM_DLL SBlockRV();
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SBlockRV, ffi::ObjectRef, SBlockRVObj);
};

/**************** Random variable: LoopRV ****************/

/*! \brief A random variable that evaluates to a TensorIR for loop */
class LoopRVObj : public ffi::Object {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<LoopRVObj>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.LoopRV", LoopRVObj, ffi::Object);
};

/*!
 * \brief Managed reference to LoopRVObj
 * \sa LoopRVObj
 */
class LoopRV : public ffi::ObjectRef {
 public:
  /*! \brief Constructor */
  TVM_DLL LoopRV();
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(LoopRV, ffi::ObjectRef, LoopRVObj);
};

/**************** Random variable: ExprRV ****************/

/*! \brief An expr random variable */
using ExprRV = PrimExpr;

using ExprRVObj = ExprObj;

/**************** The Schedule class ****************/

class Schedule;

/*! \brief The user-facing schedule class */
class ScheduleObj : public ffi::Object {
  friend class Schedule;

 public:
  virtual ~ScheduleObj() = default;

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.Schedule", ScheduleObj, ffi::Object);

 public:
  /*! \brief Get the IRModule associated with this schedule. */
  virtual IRModule mod() const { return state()->mod; }
  /*! \return The internal state of scheduling */
  virtual ScheduleState state() const = 0;
  /*! \return The internally maintained trace of scheduling program execution */
  virtual ffi::Optional<Trace> trace() const = 0;
  /*! \return The GlobalVar of the func that the schedule is currently working on */
  virtual ffi::Optional<GlobalVar> func_working_on() const = 0;
  /*!
   * \brief Instruct the schedule to work on a function in the IRModule.
   *
   * By default, the schedule works on the function with the name "main", or the only function in
   * the IRModule if there is only one. If there is multiple functions in the IRModule, and none of
   * their names are "main", users will have to call this method to explicitly specify which
   * function to work on.
   *
   * This sugar function will guide the `GetSBlock` method if its `func_name` is not specified.
   *
   * \param func_name The name of the function to be working on
   *
   * \sa GetSBlock
   */
  virtual void WorkOn(const ffi::String& func_name) = 0;
  /*!
   * \brief Returns a copy of the schedule, including both its state and its symbol table,
   * guaranteeing that
   * 1) SRef tree is completely reconstructed;
   * 2) The IRModule being scheduled is not modified;
   * 3) All the random variables are valid in the copy, pointing to the corresponding sref
   * reconstructed
   */
  virtual Schedule Copy() = 0;
  /*!
   * \brief Seed the randomness
   * \param seed The new random seed, -1 if use device random, otherwise non-negative
   */
  virtual void Seed(LinearCongruentialEngine::TRandState seed) = 0;
  /*! \brief Fork the random state */
  virtual LinearCongruentialEngine::TRandState ForkSeed() = 0;

 public:
  /******** Lookup/Remove random variables ********/
  /*!
   * \brief Get the block corresponding to the specific SBlockRV
   * \param block_rv The SBlockRV to be looked up
   * \return The corresponding block
   */
  virtual SBlock Get(const SBlockRV& block_rv) const = 0;
  /*!
   * \brief Get the for loop corresponding to the specific LoopRV
   * \param loop_rv The LoopRV to be looked up
   * \return The corresponding for loop
   */
  virtual For Get(const LoopRV& loop_rv) const = 0;
  /*!
   * \brief Get the expr corresponding to the specific random variable
   * \param expr_rv The random variable to be looked up
   * \return The corresponding expr
   */
  virtual PrimExpr Get(const ExprRV& expr_rv) const = 0;
  /*!
   * \brief Get the block sref corresponding to the specific SBlockRV
   * \param block_rv The SBlockRV to be looked up
   * \return The corresponding block sref
   */
  virtual StmtSRef GetSRef(const SBlockRV& block_rv) const = 0;
  /*!
   * \brief Get the loop sref corresponding to the specific LoopRV
   * \param loop_rv The LoopRV to be looked up
   * \return The corresponding loop sref
   */
  virtual StmtSRef GetSRef(const LoopRV& loop_rv) const = 0;
  /*!
   * \brief Check the existance of a specific SBlockRV
   * \param block_rv The SBlockRV to be looked up
   * \return Whether the corresponding block exists
   */
  virtual bool HasBlock(const SBlockRV& block_rv) const = 0;
  /*!
   * \brief Get the block/loop sref corresponding to the specific statement
   * \param stmt The statement to be looked up
   * \return The corresponding block/loop sref
   */
  virtual StmtSRef GetSRef(const StmtObj* stmt) const;
  /*!
   * \brief Get the block/loop sref corresponding to the specific statement
   * \param stmt The statement to be looked up
   * \return The corresponding block/loop sref
   */
  StmtSRef GetSRef(const Stmt& stmt) const { return this->GetSRef(stmt.get()); }
  /*!
   * \brief Remove a block random variable from the symbol table
   * \param block_rv The random variable to be removed
   */
  virtual void RemoveRV(const SBlockRV& block_rv) = 0;
  /*!
   * \brief Remove a loop random variable from the symbol table
   * \param loop_rv The random variable to be removed
   */
  virtual void RemoveRV(const LoopRV& loop_rv) = 0;
  /*!
   * \brief Remove an integer random variable from the symbol table
   * \param expr_rv The random variable to be removed
   */
  virtual void RemoveRV(const ExprRV& expr_rv) = 0;

 public:
  /******** Schedule: Sampling ********/
  /*!
   * \brief Sample an integer given the probability distribution
   * \param candidates The candidates
   * \param probs The probability distribution of the candidates
   * \param decision The sampling decision
   * \return The random variable sampled from candidates
   */
  virtual ExprRV SampleCategorical(const ffi::Array<int64_t>& candidates,
                                   const ffi::Array<FloatImm>& probs,
                                   ffi::Optional<int64_t> decision = std::nullopt) = 0;
  /*!
   * \brief Sample the factors to perfect tile a specific loop
   * \param loop_rv The loop to be tiled
   * \param n The number of tiles to be sampled
   * \param max_innermost_factor The maximum tile size allowed to be sampled in the innermost loop
   * \param decision The sampling decision
   * \return A list of length `n`, with absent factors representing sizes inferred by Split.
   */
  virtual ffi::Array<ffi::Optional<ExprRV>> SamplePerfectTile(
      const LoopRV& loop_rv, int n, int max_innermost_factor,
      ffi::Optional<ffi::Array<int64_t>> decision = std::nullopt) = 0;
  /*!
   * \brief Sample the factors to a partitioned tile for a specific loop
   *
   *  The sampled tile size will be partitioned into two parts. The second part has a guarantee
   *  that their extent's product have a factor of `innerpart_factor`. The first part is loops at
   *  [0, partition_pos); the second part is loops at [partition_pos, n) and we will have
   *  `innerpart_factor` | (l[partition_pos].extent * ... * l[n-1].extent)
   *
   * \param loop_rv The loop to be tiled
   * \param n The number of tiles to be sampled
   * \param partition_pos The position to partition tiles to two parts
   * \param innerpart_factor The factor of the second part
   * \param decision The sampling decision
   * \return A list of length `n`, the random partitioned tile sizes sampled
   */
  virtual ffi::Array<ExprRV> SamplePartitionedTile(
      const LoopRV& loop_rv, int n, int partition_pos, int innerpart_factor,
      ffi::Optional<ffi::Array<int64_t>> decision = std::nullopt) = 0;
  /*!
   * \brief Sample a compute-at location of the given block
   * \param block_rv The block whose compute-at location is to be sampled
   * \param decision The sampling decision
   * \return The sampled loop where the input block is to be computed at
   */
  virtual LoopRV SampleComputeLocation(const SBlockRV& block_rv,
                                       ffi::Optional<int64_t> decision = std::nullopt) = 0;

  /******** Schedule: Get blocks & loops ********/
  /*!
   * \brief Retrieve a block in a specific function with its name
   *
   * By default, if `func_name` is not specified, the schedule will search for the block in the
   * function that is currently being "worked on". To switch the function to be worked on, use
   * `WorkOn` before calling this method.
   *
   * \param name The name of the block to be retrieved
   * \param func_name The name of the function
   * \return The block retrieved
   * \note Indexing error is raised if 0 or multiple blocks exist with the specific name
   *
   * \sa WorkOn
   */
  virtual SBlockRV GetSBlock(const ffi::String& name,
                             const ffi::Optional<ffi::String>& func_name = std::nullopt) = 0;
  /*!
   * \brief Get the parent loops of the block in its scope, from outer to inner
   * \param block_rv The query block
   * \return A list of loops above the given block in its scope, from outer to inner
   */
  virtual ffi::Array<LoopRV> GetLoops(const SBlockRV& block_rv) = 0;
  /*!
   * \brief Get the leaf blocks of a specific scope
   * \param block_rv The block where the scope is rooted
   * \return A list of child blocks
   */
  virtual ffi::Array<SBlockRV> GetChildBlocks(const SBlockRV& block_rv) = 0;
  /*!
   * \brief Get the leaf blocks of under a specific loop
   * \param loop_rv The loop under which collecting is conducted
   * \return A list of child blocks
   */
  virtual ffi::Array<SBlockRV> GetChildBlocks(const LoopRV& loop_rv) = 0;
  /*!
   * \brief Get the producer of a specific block, under the same block scope
   * \param block_rv The block in the query
   * \return A list of blocks, the producers of the given block under the same scope of the given
   * block
   */
  virtual ffi::Array<SBlockRV> GetProducers(const SBlockRV& block_rv) = 0;
  /*!
   * \brief Get the consumers of a specific block, under the same block scope
   * \param block_rv The block to be queried
   * \return A list of blocks, the consumers of the given block under the same scope of the given
   * block
   */
  virtual ffi::Array<SBlockRV> GetConsumers(const SBlockRV& block_rv) = 0;
  /*!
   * \brief Get the list of output blocks within the given scope
   * An output block is a block which has atleast one tensor being written
   * to, but is not allocated within the Function
   * \param scope_block_rv The scope block from which output blocks are collected
   * \return A list of all blocks that write to some output tensor
   * block
   */
  virtual ffi::Array<SBlockRV> GetOutputBlocks(const SBlockRV& scope_block_rv) = 0;
  /******** Schedule: Transform loops ********/
  /*!
   * \brief Merge a list of loops into one. The loops under their LCA requires:
   * 1) Under the same scope
   * 2) Can't have annotations or thread bindings
   * 3) Start with 0 and have same extent and same nesting depth
   * 4) From target loop to their LCA, the inner loop must be the only child of the outer loop
   * \param loop_rvs The loops to be merged
   * \return The new loop after merge
   */
  virtual LoopRV Merge(const ffi::Array<LoopRV>& loop_rvs) = 0;
  /*!
   * \brief Fuse a list of consecutive loops into one. It requires:
   * 1) The loops can't have annotations or thread bindings.
   * 2) The (i+1)-th loop must be the only child of the i-th loop.
   * 3) All loops must start with 0.
   * 4) The domain of a loop to be fused cannot depend on another loop to be fused.
   * \param loop_rvs The loops to be fused
   * \param preserve_unit_iters Whether or not to preserve unit iterators in block bindings
   * \return The new loop after fusion
   */
  virtual LoopRV Fuse(const ffi::Array<LoopRV>& loop_rvs, bool preserve_unit_iters = true) = 0;
  /*!
   * \brief Split a loop into a list of consecutive loops. It requires:
   * 1) The loop can't have annotation or thread binding.
   * 2) The loop must start with 0.
   * \param loop_rv The loop to be split
   * \param factors The positive tiling factors, and at most one of which is `std::nullopt`, which
   * means that factor is inferred. \param preserve_unit_iters Whether or not to preserve unit
   * iterators in block bindings. \param disable_predication If enabled, don't create a predicate
   * for guarding the loop. This can be useful when splitting with scalable factors that the
   * schedule writer knows are divisible by the loop bound. Warning: enabling this feature may
   * result in incorrect code generation if not used carefully. \return The new loops after split.
   */
  virtual ffi::Array<LoopRV> Split(const LoopRV& loop_rv,
                                   const ffi::Array<ffi::Optional<ExprRV>>& factors,
                                   bool preserve_unit_iters = true,
                                   bool disable_predication = false) = 0;
  /*!
   * \brief Partition the loops into sequence of multiple loops
   * 1) The loop can't have annotation or thread binding.
   * \param loop_rv The loop to be partition
   * \param factors The positive integers, and at most one of which is `std::nullopt`, which means
   * that factor is inferred.
   * \param preserve_unit_iters Whether or not to preserve unit iterators in block bindings
   * \return The new loops after partition
   */
  virtual ffi::Array<LoopRV> LoopPartition(const LoopRV& loop_rv,
                                           const ffi::Array<ffi::Optional<ExprRV>>& factors,
                                           bool preserve_unit_iters = true) = 0;
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
   * \param ordered_loop_rvs The loops in the new order
   */
  virtual void Reorder(const ffi::Array<LoopRV>& ordered_loop_rvs) = 0;
  /*!
   * \brief Reorder the itervars inside a block.
   * \param block_rv The block to be transformed.
   * \param new_order The new itervar order.
   */
  virtual void ReorderBlockIterVar(const SBlockRV& block_rv,
                                   const ffi::Array<int64_t> new_order) = 0;
  /*!
   * \brief Create a new unit loop on top of the specific block.
   * \param block_rv The block above which the new loop is created
   * \return The new loop created
   */
  virtual LoopRV AddUnitLoop(const SBlockRV& block_rv) = 0;
  /*!
   * \brief Create a new unit loop on top of the specific loop.
   * \param loop_rv The loop above which the new loop is created
   * \return The new loop created
   */
  virtual LoopRV AddUnitLoop(const LoopRV& loop_rv) = 0;
  /******** Schedule: Manipulate ForKind ********/
  /*!
   * \brief Parallelize the input loop. It requires:
   * 1) The scope block that the loop is in should have stage-pipeline property
   * 2) All the blocks under the loop are complete blocks or reduction blocks, and have affine
   * bindings
   * 3) For each block under the loop, the loop can only be contained in data-parallel block iters'
   * bindings
   * \param loop_rv The loop to be parallelized
   */
  virtual void Parallel(const LoopRV& loop_rv) = 0;
  /*!
   * \brief Vectorize the input loop. It requires:
   * 1) The scope block that the loop is in should have stage-pipeline property
   * 2) All the blocks under the loop are complete blocks or reduction blocks, and have affine
   * bindings
   * 3) For each block under the loop, the loop can only be contained in data-parallel block iters'
   * bindings
   * \param loop_rv The loop to be vectorized
   */
  virtual void Vectorize(const LoopRV& loop_rv) = 0;
  /*!
   * \brief Bind the input loop to the given thread axis. It requires:
   * 1) The scope block that the loop is in should have stage-pipeline property
   * 2) All the blocks under the loop are complete blocks or reduction blocks, and have affine
   * bindings
   * 3) For each block under the loop, if the thread axis starts with "threadIdx`, the loop can only
   * be contained in data-parallel block iter and reduction block iters' bindings. Otherwise the
   * loop can only be contained in data-parallel block iters' bindings
   * \param loop_rv The loop to be bound to the thread axis
   * \param thread_axis The thread axis to be bound to the loop
   */
  virtual void Bind(const LoopRV& loop_rv, const ffi::String& thread_axis) = 0;
  /*!
   * \brief Unroll the input loop. It requires nothing
   * \param loop_rv The loop to be unrolled
   */
  virtual void Unroll(const LoopRV& loop_rv) = 0;
  /******** Schedule: Insert cache stages ********/
  /*!
   * \brief Create a block that reads a tensor region into a read cache. It requires:
   * 1) There is at most one block who writes the tensor in the scope.
   * 2) The scope block have stage-pipeline property.
   * \param block_rv The consumer block of the target tensor.
   * \param read_tensor_index The index of the tensor in block's read region.
   * \param storage_scope The target storage scope.
   * \param consumer_blocks An optional list of consumers of the cache to rewrite.
   * \return The cache stage block.
   */
  virtual SBlockRV CacheRead(const SBlockRV& block_rv, int read_tensor_index,
                             const ffi::String& storage_scope,
                             const ffi::Array<SBlockRV> consumer_blocks = {}) = 0;
  /*!
   * \brief Create a block that writes a tensor region into a write cache. It requires:
   * 1) There is only one block who writes the target tensor.
   * 2) The scope block have stage-pipeline property.
   * \param block_rv The producer of the tensor
   * \param write_tensor_index The index of the tensor in block's write region
   * \param storage_scope The target storage scope
   * \param consumer_blocks An optional list of consumers to read from cache directly.
   * \return The cache stage block.
   */
  virtual SBlockRV CacheWrite(const SBlockRV& block_rv, int write_tensor_index,
                              const ffi::String& storage_scope,
                              const ffi::Array<SBlockRV> consumer_blocks = {}) = 0;
  /*!
   * \brief Create a block that reads a tensor region into a read cache. It requires:
   * 1) There is at most one block who writes the tensor in the scope.
   * 2) The scope block have stage-pipeline property.
   * Compared to cache read, the indices to access allocated cache tensor is customized by user.
   * \param block_rv The consumer block of the target tensor.
   * \param read_tensor_index The index of the tensor in block's read region.
   * \param storage_scope The target storage scope.
   * \param index_map User defined indices to access allocated cache tensor, maps from block iter
   * vars.
   * \return The cache stage block.
   */
  virtual SBlockRV ReindexCacheRead(const SBlockRV& block_rv, int read_tensor_index,
                                    const ffi::String& storage_scope,
                                    const IndexMap& index_map) = 0;
  /*!
   * \brief Create a block that writes a tensor region into a write cache. It requires:
   * 1) There is only one block who writes the target tensor.
   * 2) The scope block have stage-pipeline property.
   * Compared to cache write, the indices to access allocated cache tensor is customized by user.
   * \param block_rv The producer of the tensor
   * \param write_tensor_index The index of the tensor in block's write region
   * \param storage_scope The target storage scope
   * \param index_map User defined indices to access allocated cache tensor, maps from block iter
   * vars.
   * \return The cache stage block.
   */
  virtual SBlockRV ReindexCacheWrite(const SBlockRV& block_rv, int write_tensor_index,
                                     const ffi::String& storage_scope,
                                     const IndexMap& index_map) = 0;
  /*!
   * \brief Create 2 blocks that read&write a tensor region into a read/write cache.
   * It requires the target block both read & write the target tensor.
   * \param block_rv The target block operates on the target tensor.
   * \param read_tensor_index The index of the tensor in block's read region.
   * \param storage_scope The target storage scope
   * \return The cache stage blocks, cache read block together with cache write block.
   */
  virtual ffi::Array<SBlockRV> CacheInplace(const SBlockRV& block_rv, int read_tensor_index,
                                            const ffi::String& storage_scope) = 0;
  /*!
   * \brief Create a block to cache precomputed index for later use.
   * if there is no index computation, keep unchanged.
   * \param block_rv The target block
   * \param storage_scope The storage scope of cached block
   * \param cse_thresh The repeat threshold that determines a common sub expr
   * \return The cache stage blocks.
   */
  virtual ffi::Array<SBlockRV> CacheIndex(const SBlockRV& block_rv,
                                          const ffi::String& storage_scope, int cse_thresh) = 0;
  /*!
   * \brief Create a block that read/write a tensor region into a read/write cache with reindexing.
   * The layout of the cache will be the same as by the iterators of the block that reads/writes the
   * tensor. It requires:
   * 1) There is only one block who reads/writes the target tensor
   * 2) There is only one tensor load/store of this tensor in the block
   * \param block_rv The block operates on the target tensor.
   * \param tensor_index The index of the tensor in block's read or write region.
   * \param tensor_index_type The type of the tensor index, kRead or kWrite.
   * \return The reindex stage block.
   */
  virtual SBlockRV ReIndex(const SBlockRV& block_rv, int tensor_index,
                           TensorIndexType tensor_index_type) = 0;
  /******** Schedule: Data movement ********/
  virtual SBlockRV ReadAt(const LoopRV& loop_rv, const SBlockRV& block_rv, int read_tensor_index,
                          const ffi::String& storage_scope) = 0;
  virtual SBlockRV WriteAt(const LoopRV& loop_rv, const SBlockRV& block_rv, int write_tensor_index,
                           const ffi::String& storage_scope) = 0;
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
   * \param block_rv The block to be moved
   * \param loop_rv The loop where the block to be moved under
   * \param preserve_unit_loops Whether to keep the trivial loops whose extents are 1
   * \param index The block index of the loop body subtree blocks:
   * - `index = -1` means inserted into the last possible insertion point;
   * - `index = -2` means inserted into the first possible insertion point;
   * - Otherwise, `index` is a nonnegative number that indicates the insertion point
   */
  virtual void ComputeAt(const SBlockRV& block_rv, const LoopRV& loop_rv, bool preserve_unit_loops,
                         int index = -1) = 0;
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
   * \param block_rv The block to be moved
   * \param loop_rv The loop where the block to be moved under
   * \param preserve_unit_loops Whether to keep the trivial loops whose extents are 1
   * \param index The block index of the loop body subtree blocks:
   * - `index = -1` means inserted into the last possible insertion point;
   * - `index = -2` means inserted into the first possible insertion point;
   * - Otherwise, `index` is a nonnegative number that indicates the insertion point
   */
  virtual void ReverseComputeAt(const SBlockRV& block_rv, const LoopRV& loop_rv,
                                bool preserve_unit_loops, int index = -1) = 0;
  /*!
   * \brief Inline a block into its consumer(s). It requires:
   * 1) The block is a complete non-root block, which only produces one tensor
   * 2) The block must not be the only leaf in the scope.
   * 3) The body of the block must be a TensorStore statement in the form of,
   *    A[i, j, k, ...] = ...
   * where the indices of the LHS are all distinct atomic variables,
   * and no variables other than those indexing variables are allowed in the statement.
   * \param block The block to be inlined to its consumer(s)
   */
  virtual void ComputeInline(const SBlockRV& block) = 0;
  /*!
   * \brief Inline a block into its only producer. It requires:
   * 1) The block is a complete non-root block, which only produces and consumers one tensor
   * 2) The block must not be the only leaf in the scope.
   * 3) The only producer of the block is a read-after-write producer and a complete non-root block
   * 4) The body of the block must be a TensorStore statement in the form of,
   *    B[f(i, j, k, ...)] = g(i, j, k, A[i, j, k, ...] ...)
   * where the indices of each `TensorLoad` on the RHS are all distinct atomic variables,
   * and no variables other than those indexing variables are allowed in the statement.
   * \param block The block to be inlined to its producer
   */
  virtual void ReverseComputeInline(const SBlockRV& block) = 0;
  /*!
   * \brief Fuse an epilogue block into a reduction block
   * \param reduction_block The reduction block (e.g., matmul)
   * \param epilogue_block The epilogue block to be fused (e.g., bias add)
   */
  virtual void FuseReductionEpilogue(const SBlockRV& reduction_block,
                                     const SBlockRV& epilogue_block) = 0;
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
  virtual SBlockRV DecomposeReduction(const SBlockRV& block_rv, const LoopRV& loop_rv) = 0;
  /*!
   * \brief Factorize an associative reduction block by the specified loop.
   * \details An associative reduction cannot be parallelized directly,
   * because it leads to potential race condition during accumulation.
   * Alternatively, the reduction could be factorized on a loop with the following steps:
   * - Step 1: evenly slice the reduction into `n` separate chunks, where `n` is the loop extent
   * - Step 2: compute the chunks separately and write the result into `n` intermediate tensors;
   * - Step 3: accumulate the `n` separate tensor into the result tensor.
   * Note that the Step 2 above introduces opportunities for parallelization.
   * RFactor is a schedule primitive that implements the transformation described above.
   * \param loop_rv The loop outside block we want to do rfactor
   * \param factor_axis The position where the new dimension is placed in the new introduced rfactor
   *                    tensor. Suppose the original reduction block writes to tensor `B` with
   *                    ndim(B) dimensions, then `factor_axis` should be in range `[-ndim(B) - 1,
   *                    ndim(B)]`, and the negative index will be normalized to a non-negative one
   * \return The rfactor block
   */
  virtual SBlockRV RFactor(const LoopRV& loop_rv, int factor_axis) = 0;
  /******** Schedule: SBlock annotation ********/
  /*!
   * \brief Set alignment requirement for specific dimension such that
   *        stride[axis] == k * factor + offset for some k. This is useful to set memory layout for
   *        more friendly memory access pattern. For example, we can set alignment to be factor=2,
   *        offset=1 to avoid bank conflict for thread access on higher dimension in GPU shared
   *        memory.
   * \param block_rv The producer block of the tensor
   * \param tensor_index The index of the tensor in block's write region
   * \param axis The dimension to be specified for alignment
   * \param factor The factor multiple of alignment
   * \param offset The required offset factor
   */
  virtual void StorageAlign(const SBlockRV& block_rv, int tensor_index, int axis, int factor,
                            int offset) = 0;
  /*!
   * \brief Set the storage scope of a tensor, where the tensor is specified by a block and a
   * write-index
   * \param block_rv The producer block of the tensor
   * \param tensor_index The index of the tensor in block's write region
   * \param storage_scope The storage scope to be set
   */
  virtual void SetScope(const SBlockRV& block_rv, int tensor_index,
                        const ffi::String& storage_scope) = 0;
  /*!
   * \brief Set the data type of a tensor, where the tensor is specified by a block and a
   * write-index
   * \note This schedule primitive is unsafe and may change correctness of program because of
   *   type conversion, please use with caution.
   * \param block_rv The producer block of the tensor
   * \param tensor_index the index of the tensor in block's write region
   * \param dtype The data type to be set
   */
  virtual void UnsafeSetDType(const SBlockRV& block_rv, int tensor_index,
                              const ffi::String& dtype) = 0;
  /******** Schedule: Blockize & Tensorize ********/
  /*!
   * \brief Convert the subtree rooted at a specific loop into a block.
   * \param loop_rv the root of the subtree
   * \param preserve_unit_iters Whether or not to preserve unit iterators in block bindings
   * \return the new block
   */
  virtual SBlockRV Blockize(const LoopRV& loop_rv, bool preserve_unit_iters = true) = 0;
  /*!
   * \brief Convert specified blocks into a nested block.
   * \param blocks the specified block to construct the new block
   * \param preserve_unit_iters Whether or not to preserve unit iterators in block bindings
   * \return the new block
   */
  virtual SBlockRV Blockize(const ffi::Array<SBlockRV>& blocks,
                            bool preserve_unit_iters = true) = 0;
  /*!
   * \brief Tensorize the computation enclosed by loop with the tensor intrin.
   * \param loop_rv The loop to be tensorized
   * \param intrin Name of the tensor intrinsic
   * \param preserve_unit_iters Whether or not to preserve unit iterators in block bindings
   */
  virtual void Tensorize(const LoopRV& loop_rv, const ffi::String& intrin,
                         bool preserve_unit_iters = true) = 0;
  /*!
   * \brief Tensorize the computation enclosed by loop with the tensor intrin.
   * \param block_rv The block to be tensorized
   * \param intrin Name of the tensor intrinsic
   * \param preserve_unit_iters Whether or not to preserve unit iterators in block bindings
   */
  virtual void Tensorize(const SBlockRV& block_rv, const ffi::String& intrin,
                         bool preserve_unit_iters = true) = 0;

  /******** Schedule: Annotation ********/
  /*!
   * \brief Annotate a loop with a key value pair
   * \param loop_rv The loop to be annotated
   * \param ann_key The annotation key
   * \param ann_val The annotation value, a string or a ExprRV
   */
  virtual void Annotate(const LoopRV& loop_rv, const ffi::String& ann_key, const Any& ann_val) = 0;
  /*!
   * \brief Annotate a block with a key value pair
   * \param block_rv The block to be annotated
   * \param ann_key The annotation key
   * \param ann_val The annotation value, a string or a ExprRV
   */
  virtual void Annotate(const SBlockRV& block_rv, const ffi::String& ann_key,
                        const Any& ann_val) = 0;
  /*!
   * \brief Unannotate a loop's annotation with key ann_key
   * \param loop_rv The loop to be unannotated
   * \param ann_key The annotation key
   */
  virtual void Unannotate(const LoopRV& loop_rv, const ffi::String& ann_key) = 0;
  /*!
   * \brief Unannotate a block's annotation with key ann_key
   * \param block_rv The block to be unannotated
   * \param ann_key The annotation key
   */
  virtual void Unannotate(const SBlockRV& block_rv, const ffi::String& ann_key) = 0;

  /******** Schedule: Layout transformation ********/
  /*!
   * \brief Apply a transformation represented by IndexMap to tensor
   * \details The indices and the access region to the target tensor is transformed by the given
   * index_map. The index_map is used to infer the new shape of the tensor. Tensor must be either
   * a function parameter, or allocated in a block (it cannot be a tensor subregion created via
   * 'match_tensor').
   * \param block_rv The block that accesses the target tensor.
   * \param tensor_index The index of the tensor in block's read or write region.
   * \param tensor_index_type The type of the tensor index, kRead or kWrite.
   * \param index_map The transformation to apply.
   *
   * \param pad_value The value to write into padding introduced by
   *    the transformation.  If the schedule contains a producer block
   *    for the specified tensor, the pad value will be written as
   *    part of the producer block if possible, or after the producer
   *    block otherwise.  Otherwise, if the tensor is an input, will
   *    insert an annotation block to state that the padding contains
   *    the known value.
   *
   *    Note: If applied to an input tensor, the calling scope is
   *    responsible for ensuring that the pad_value is present.
   *    Algebraic symplifications, branch elimination, and other
   *    optimizations may assume that this precondition is met, and
   *    may result in incorrect results being returned.
   *
   * \param assume_injective_transform If set to true, the schedule primitive will assume the
   * index_map is injective and skip checking overlapping of the mapped indices. This can be useful
   * for complicated index_map that the analysis does not cover. It is the callers' responsibility
   * to ensure the index map is injective, otherwise, the correctness of the schedule is not
   * guaranteed.
   */
  virtual void TransformLayout(const SBlockRV& block_rv, int tensor_index,
                               TensorIndexType tensor_index_type, const IndexMap& index_map,
                               const ffi::Optional<IndexMap>& pad_value = std::nullopt,
                               bool assume_injective_transform = false) = 0;

  /*!
   * \brief Apply a transformation represented by IndexMap to block
   * \details The block iters and the block body are transformed by the given index_map.
   * Outer loops corresponding to each new block iter are regenerated.
   * The index_map is required to be bijective affine since we need its inverse mapping.
   * \param block_rv The block to be transformed
   * \param index_map The transformation to apply.
   */
  virtual void TransformBlockLayout(const SBlockRV& block_rv, const IndexMap& index_map) = 0;

  /******** Schedule: Padding ********/
  /*!
   * \brief Decompose a padding block into a block filling const pad values and a block
   * writing in-bound values.
   * \param block_rv The block that match the padding pattern.
   * \param loop_rv The loop above which the const filling block is inserted before.
   * \return The const pad value filling block.
   */
  virtual SBlockRV DecomposePadding(const SBlockRV& block_rv, const LoopRV& loop_rv) = 0;

  /*!
   * \brief Pad the computation of Einsum.
   * \param block_rv The block that matches the Einsum pattern.
   * \param padding The padding for each block iter.
   * \details This schedule primitives identifies the Einsum pattern in the block body, and find its
   * producer blocks. It then pads the computation of the Einsum pattern and its producer blocks.
   * The output tensor and the producer tensor is resized according to the padding size. It requires
   * the output tensor and the producer tensor to be allocated inside the Function.
   *
   * The padding is a list of non-negative integers, each element corresponds to the padding for
   * each block iter in the order of block iters. The block and its producer blocks should have
   * trivial bindings, i.e. each block iter is bound to a single loop variable. After padding, the
   * block iter extent and the corresponding outer loop is extended by the padding size.
   *
   * The size of the producer tensors are infered from the padding size of the Einsum computation.
   * The producer tensors are padded by the initial value of the corresponding reduction.
   */
  virtual void PadEinsum(const SBlockRV& block_rv, const ffi::Array<int64_t>& padding) = 0;

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
  virtual void RollingBuffer(const SBlockRV& block_rv, int write_tensor_index) = 0;

  /*!
   * \brief Annotate the tensor access of a block
   * \param block_rv The block to be annotated
   * \param tensor_index The index of the tensor in block's read or write region
   * \param tensor_index_type The type of the tensor index, kRead or kWrite.
   * \param index_map The index map that defines the new read or write region
   */
  virtual void AnnotateTensorAccess(const SBlockRV& block_rv, int tensor_index,
                                    TensorIndexType tensor_index_type,
                                    const IndexMap& index_map) = 0;

  /******** Schedule: Misc ********/
  /*! \brief A no-op that marks the start of postprocessing phase of scheduling */
  virtual void EnterPostproc() = 0;

  /*!
   * \brief Hide some tensor access in the given block.
   * \param block_rv The block where we hide tensor access.
   * \param tensor_type The tensor type: read/write
   * \param tensor_index_array The array of tensor indices we hide access.
   */
  virtual void UnsafeHideTensorAccess(const SBlockRV& block_rv, const ffi::String& tensor_type,
                                      const ffi::Array<int64_t>& tensor_index_array) = 0;
};

/*!
 * \brief Managed reference to ScheduleObj
 *
 * A schedule is a set of transformations that change the order of computation but
 * preserve the semantics of computation. Some example of schedules:
 * 1) Split a loop into two;
 * 2) Reorder two loops;
 * 3) Inline the computation of a specific tensor into its consumer
 *
 * The schedule class stores auxiliary information to schedule correctly and efficiently.
 *
 * Link to tutorial: https://tvm.apache.org/docs/tutorials/language/schedule_primitives.html
 *
 * \sa ScheduleObj
 */
class Schedule : public ffi::ObjectRef {
 public:
  explicit Schedule(ffi::UnsafeInit tag, ffi::ObjectPtr<ScheduleObj> data) : ffi::ObjectRef(tag) {
    data_ = std::move(data);
  }

  /*!
   * \brief Construct a concrete TensorIR schedule from an IRModule
   * \param mod The IRModule to be scheduled
   * \param seed The seed value for schedule's random state
   * \param debug_mask Do extra correctness checking after the class creation
   * and each time after calling the Replace method.
   * \param error_render_level The level of error rendering
   * \param enable_check Whether to enable some prequisite checks for schedule primitives, it's
   *   user's duty to guarantee the schedule correctness if we disable the checks.
   * \return The concrete schedule created
   * \sa ScheduleDebugMask
   * \note The checks performed includes: 1) VerifySRefTree 2) VerifyCachedFlags
   */
  TVM_DLL static Schedule Concrete(IRModule mod, LinearCongruentialEngine::TRandState seed,
                                   int debug_mask, ScheduleErrorRenderLevel error_render_level,
                                   bool enable_check = true);
  /*!
   * \brief Construct a traced concrete TensorIR schedule from an IRModule
   * \param mod The IRModule to be scheduled
   * \param seed The seed value for schedule's random state
   * \param debug_mask Do extra correctness checking after the class creation
   * and each time after calling the Replace method.
   * \param error_render_level The level of error rendering
   * \param enable_check Whether to enable prequisite checks for schedule primitives.
   * \return The concrete schedule created
   * \sa ScheduleDebugMask
   * \note The checks performed include:
   * 1) VerifySRefTree
   * 2) VerifyCachedFlags
   */
  TVM_DLL static Schedule Traced(IRModule mod, LinearCongruentialEngine::TRandState seed,
                                 int debug_mask, ScheduleErrorRenderLevel error_render_level,
                                 bool enable_check = true);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Schedule, ffi::ObjectRef, ScheduleObj);
};

}  // namespace s_tir
}  // namespace tvm

#endif  // TVM_S_TIR_SCHEDULE_SCHEDULE_H_
