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
 * \file tvm/s_tir/transform.h
 * \brief S-TIR specific transformation passes.
 */
#ifndef TVM_S_TIR_TRANSFORM_H_
#define TVM_S_TIR_TRANSFORM_H_

#include <tvm/base.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/transform.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/target/target.h>
#include <tvm/tirx/transform.h>

#include <string>
#include <vector>

namespace tvm {
namespace s_tir {
namespace transform {

using tirx::transform::CreateFunctionPass;
using tvm::transform::Pass;
using tvm::transform::PassContext;

/*! \brief De-duplicate definitions, including schedulable block iterators, across Functions. */
TVM_DLL Pass ConvertSSA();

/*! \brief Simplify schedulable TIR using block iteration constraints and shared simplifier options.
 */
TVM_DLL Pass StmtSimplify();

/*!
 * \brief Canonicalize loop to start from zero .
 * \return The pass.
 */
TVM_DLL Pass CanonicalizeLoop();

/*!
 * \brief Lower cross-thread reduction from thread
 * bindings to intrinsic function calls.
 * \return The pass.
 */
TVM_DLL Pass LowerCrossThreadReduction();

/*!
 * \brief Lower block init stmt into If stmts
 * \return The pass.
 */
TVM_DLL Pass LowerInitBlock();

/*!
 * \brief Locate the tensor allocation to the exact position (usually is
 *        the lca of tensor access). This pass will inject opaque block
 *        with alloc_tensors at the allocation site.
 * \return The pass.
 */
TVM_DLL Pass PlanAndUpdateTensorAllocationLocation();

/*!
 * \brief Substitute all the block vars with the PrimExprs they are bound to, indicated by the
 *        corresponding iter_values in BlockRealize, for opaque blocks by removing all
 *.        the iter_values in BlockRealize and iter_vars in Block.
 * \return The pass.
 */
TVM_DLL Pass ConvertBlocksToOpaque();

/*!
 * \brief Lift the same thread bindings to their LCA loops
 * \return The pass.
 */
TVM_DLL Pass LiftThreadBinding();

/*!
 * \brief Compact the tensor access region by removing the tensor regions that are not accessed,
 *        i.e. narrowing the tensor shape and adjust the access region if necessary.
 *
 * Before narrowing, `B` is a `[16, 16]` tensor, but only a skinny vector `B[i, 0:16]` is accessed.
 *
 *  \code
 *
 *  for i in range(0, 16):
 *      with T.sblock():
 *          B = T.alloc_tensor(ty_args=[T.Tensor((16, 16), "float32")])
 *          for j in range(0, 16):
 *              B[i, j] = A[i, j] + 1
 *          for j in range(0, 16):
 *              C[i, j] = B[i, j] + 1
 *
 *  \endcode
 *
 * This pass narrows the tensor shape and adjust its accessed region accordingly.
 * In this particular case, because only a `1 * 16` vector of `B` is accessed,
 * the pass narrows `B` to shape `[1, 16]`, and changes the access to `B[i, j]` to `B[0, j]`.
 *
 *  \code
 *
 *  for i in range(0, 16):
 *      with T.sblock():
 *          B = T.alloc_tensor(ty_args=[T.Tensor((1, 16), "float32")])
 *          for j in range(0, 16):
 *              B[0, j] = A[i, j] + 1
 *          for j in range(0, 16):
 *              C[i, j] = B[0, j] + 1
 *
 *  \endcode
 *
 * \param is_strict ensure the compacted shape always smaller than the original shape.
 *   otherwise it allows to grow the shape to match actual accessed tensor regions.
 * \return The pass.
 */
TVM_DLL Pass CompactTensorAllocation(bool is_strict = true);

/*!
 * \brief Remove match tensors inside the block. Also, it will validate the binding.
 * \return The pass.
 */
TVM_DLL Pass LowerMatchTensor();

/*!
 * \brief Inject permuted layout for shared memory.
 * \return The pass.
 */
TVM_DLL Pass InjectPermutedLayout();

/*!
 * \brief Transform Mma scope (mma.matrixA/B/C) to local scope with layout transformation.
 * \return The pass.
 */
TVM_DLL Pass TransformMmaTensorLayout();

/*!
 * \brief Remove the block to ensure that the TIR can not be scheduled again.
 * Run LoopPartition first when thread-binding loops carry loop_partition_hint.
 * \return The pass.
 */
TVM_DLL Pass LowerOpaqueBlock();

/*!
 * \brief Lower thread-binding loops to fresh lexical launch regions.
 * Run after LoopPartition and LowerOpaqueBlock, while retaining loops for scheduling
 * and feature extraction before this boundary.
 * \return The pass.
 */
TVM_DLL Pass LowerThreadBinding();

/*!
 * \brief This pass transforms annotated loops into pipelined ones where producers and consumers
 * are overlapped with the information provided in loop annotations, which enables optimization
 * techniques like prefetching and pipeline parallelism.
 *
 * The pipeline scope consists of the direct children of the annotated loop (ignoring SBlockRealize,
 * SBlock, SeqStmt), and the number of children is denoted by `n` in the documentation.
 *
 * The following annotations are used to guide the loop transformation:
 *
 * 1) Loop annotation `software_pipeline_stage` defines the pipeline stage.
 * An array of `n` integers, and each element should be in range [0, max_stage],
 * where max_stage is the maximum (inclusive) stage.
 * 2) Loop annotation `software_pipeline_order` defines the pipeline order.
 * An array of `n` integers, a permutation of [0, 1, ..., num_components - 1];
 * 3) SBlock annotation `double_buffer_scope` controls certain tensor sizes to allow decoupling of
 * read/write dependency. It's an integer index of the write regions of the block.
 *
 * Every annotated loop is transformed into a loop with three blocks as its direct children:
 *
 * 1) Prologue block, where components whose stage is less than `max_stage` is executed;
 *
 * 2) Body block, where all the components are executed;
 *
 * 3) Epilogue block, where only components whose stage is greater than 0 will be executed.
 * The execution order is controlled by the annotation `software_pipeline_order`,
 * and thus could be different than the original order.
 *
 * Note: For nested software pipelines, the inner software pipeline will be generated first,
 * which may affect the number of the direct children of the outer loop.
 * In this case, the annotations for the outer software
 * pipeline should include the result of the inner software pipeline,
 * which is the three blocks as discussed above.
 *
 * \return The IR transform pass.
 */
TVM_DLL Pass InjectSoftwarePipeline();

/*!
 * \brief Automatically do memory optimizations for auto copy blocks
 * \return The pass.
 */
TVM_DLL Pass LowerAutoCopy();

/*!
 * \brief Add the explicit local stage for the shared memory access on GPU.
 * \return The pass.
 */
TVM_DLL Pass ManifestSharedMemoryLocalStage();

/*! \brief Annotate irregular loop mark. */
TVM_DLL Pass AnnotateIrregularLoop();

/*!
 * \brief partition loops in the stmt.
 *
 * \return The pass.
 */
TVM_DLL Pass LoopPartition();

/*!
 * \brief Inject virtual thread loops.
 *
 * \return The pass.
 */
TVM_DLL Pass InjectVirtualThread();

/*!
 * \brief Detect and rewrite unsafe select that contains memory access.
 * \return The pass.
 */
TVM_DLL Pass RewriteUnsafeSelect();

/*!
 * \brief Lower VTCM allocations.
 * \return The pass.
 */
TVM_DLL Pass LowerVtcmAlloc();

/*!
 * \brief Insert sync between parallel read/write of shared buffers.
 * \param storage_scope The storage scope considered.
 * \return The pass.
 */
TVM_DLL Pass ThreadSync(runtime::MemoryScope storage_scope);

/*!
 * \brief Lower asynchronous queue operations and remove synchronization regions.
 * Run after ThreadSync and optional InjectPTXAsyncCopy.
 * \return The pass.
 */
TVM_DLL Pass LowerSynchronization();

/*!
 * \brief Infer the TensorCore fragment information using tensor intrinsics.
 * \return The pass.
 */
TVM_DLL Pass InferFragment();

/*!
 * \brief Rewrite global to shared memory copy on CUDA with asynchronous copy.
 * \return The pass.
 */
TVM_DLL Pass InjectPTXAsyncCopy();

/*!
 * \brief Merge multiple TIR-level shared memory allocations into one.
 * \return The pass.
 */
TVM_DLL Pass MergeSharedMemoryAllocations();

/*!
 * \brief Set default thread bindings for GPU Functions.
 * \return The pass.
 */
TVM_DLL Pass DefaultGPUSchedule(ffi::Optional<Target> target = std::nullopt);

/*!
 * \brief Remove weight layout rewrite block before benchmark.
 * \param skip_tensor_rewrite Whether to skip tensor rewrite.
 * \return The pass.
 */
TVM_DLL Pass RemoveWeightLayoutRewriteBlock(bool skip_tensor_rewrite = false);

/*!
 * \brief Remove stores of tirx::undef_op.
 * \return The pass.
 */
TVM_DLL Pass RemoveStoreUndef();

}  // namespace transform
}  // namespace s_tir
}  // namespace tvm

#endif  // TVM_S_TIR_TRANSFORM_H_
