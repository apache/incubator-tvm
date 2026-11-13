# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""S-TIR specific transformations."""
# pylint: disable=invalid-name, unsupported-binary-operation

import tvm_ffi as _ffi

from tvm_tirx.runtime import MemoryScope

from . import _ffi_api


def ConvertSSA():
    """De-duplicate definitions, including schedulable block iterators, across Functions."""
    return _ffi_api.ConvertSSA()


def StmtSimplify():
    """Simplify schedulable TIR with block constraints."""
    return _ffi_api.StmtSimplify()


def CanonicalizeLoop():
    """Canonicalize the loop to start from zero and use trivial step

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass
    """
    return _ffi_api.CanonicalizeLoop()  # type: ignore


def LowerCrossThreadReduction():
    """Lower cross-thread reduction from thread bindings to
    intrinsic function calls.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass
    """
    return _ffi_api.LowerCrossThreadReduction()  # type: ignore


def LowerInitBlock():
    """Lower block init stmt into If statements.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass
    """
    return _ffi_api.LowerInitBlock()  # type: ignore


def PlanAndUpdateTensorAllocationLocation():
    """Locate the tensor allocation to the exact position (usually is
    the lca of tensor access). This pass will inject opaque block
    with alloc_tensors at the allocation site.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass
    """
    return _ffi_api.PlanAndUpdateTensorAllocationLocation()  # type: ignore


def ConvertBlocksToOpaque():
    """Substitute all the block vars with the PrimExprs they are bound to, indicated by
    the corresponding iter_values in BlockRealize, and then convert the blocks into
    opaque ones by removing all the iter_values in BlockRealize and iter_vars in Block.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass
    """
    return _ffi_api.ConvertBlocksToOpaque()  # type: ignore


def LiftThreadBinding():
    """Lift the same thread bindings to their LCA loops.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass
    """
    return _ffi_api.LiftThreadBinding()  # type: ignore


def CompactTensorAllocation(is_strict: bool = True):
    """Compact the tensor access region by removing the tensor regions
    that are not accessed, i.e. narrowing the tensor shape and adjust
    the access region if necessary.

    Parameters
    ----------
    is_strict : bool
        Ensure the compacted shape to be always smaller than the original shape.
        Otherwise it allows to grow the shape to match actual accessed tensor regions.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass
    """
    return _ffi_api.CompactTensorAllocation(is_strict)  # type: ignore


def LowerMatchTensor():
    """Remove match tensors inside the block. Also, it will validate the binding.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass
    """
    return _ffi_api.LowerMatchTensor()  # type: ignore


def LowerOpaqueBlock():
    """Remove the block to ensure that the TIR can not be scheduled again.

    Run LoopPartition first when thread-binding loops carry loop_partition_hint.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass
    """
    return _ffi_api.LowerOpaqueBlock()  # type: ignore


def TransformMmaTensorLayout():
    """Transform mma tensor layout

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass
    """
    return _ffi_api.TransformMmaTensorLayout()  # type: ignore


def InjectPermutedLayout():
    """Inject permuted layout in mma

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass
    """
    return _ffi_api.InjectPermutedLayout()  # type: ignore


def LowerThreadBinding():
    """Lower thread-binding loops to fresh lexical launch regions.

    Run after LoopPartition and LowerOpaqueBlock. Each loop keeps its own
    lexical binding, including loops that use the same hardware axis.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass
    """
    return _ffi_api.LowerThreadBinding()  # type: ignore


def InjectSoftwarePipeline():
    """Transform annotated loops into pipelined one that parallelize producers and consumers

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass
    """
    return _ffi_api.InjectSoftwarePipeline()  # type: ignore


def LowerAutoCopy():
    """Automatically do memory optimizations for auto copy blocks

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass
    """
    return _ffi_api.LowerAutoCopy()  # type: ignore


def ManifestSharedMemoryLocalStage():
    """Add the explicit local stage for the shared memory access on GPU.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass
    """
    return _ffi_api.ManifestSharedMemoryLocalStage()  # type: ignore


def AnnotateIrregularLoop():
    """Annotate irregular loop mark.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass
    """
    return _ffi_api.AnnotateIrregularLoop()  # type: ignore


@_ffi.register_object("s_tir.transform.LoopPartitionConfig")
class LoopPartitionConfig(_ffi.Object):
    """Config for loop partition pass"""


def LoopPartition():
    """Partition loops in the stmt.

    Consumes the owning For's ``loop_partition_hint`` annotation. A provably
    true value enables partitioning even without likely tags. Run this pass
    before opaque lowering when hinting thread-binding loops.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass
    """
    return _ffi_api.LoopPartition()  # type: ignore


def InjectVirtualThread():
    """Inject virtual thread loops.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass
    """
    return _ffi_api.InjectVirtualThread()  # type: ignore


def RewriteUnsafeSelect():
    """Detect and rewrite unsafe select that contains memory access.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass
    """
    return _ffi_api.RewriteUnsafeSelect()  # type: ignore


def VerifyVTCMLimit(default_target=None):
    """Verify if the size of the allocated vtcm memory satisfies the limit.

    The limit is determined from the "vtcm-capacity" attribute of the target.

    Parameters
    ----------
    default_target : Optional[tvm_tirx.target.Target]
        The default target to use if a Function does not have a target attribute.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass
    """
    return _ffi_api.VerifyVTCMLimit(default_target)  # type: ignore


def LowerVtcmAlloc():
    """Lower vtcm allocation.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass
    """
    return _ffi_api.LowerVtcmAlloc()  # type: ignore


def ThreadSync(storage_scope: str | MemoryScope):
    """Insert sync between parallel read/write of shared tensors.

    Parameters
    ----------
    storage_scope: str or MemoryScope
        The target storage scope.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass
    """
    return _ffi_api.ThreadSync(storage_scope)  # type: ignore


def LowerSynchronization():
    """Lower async queue operations and erase regions after synchronization planning."""
    return _ffi_api.LowerSynchronization()  # type: ignore


def InferFragment():
    """Infer the TensorCore fragment information using tensor intrinsics.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass
    """
    return _ffi_api.InferFragment()  # type: ignore


def InjectPTXAsyncCopy():
    """Rewrite global to shared memory copy on CUDA with asynchronous copy.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass
    """
    return _ffi_api.InjectPTXAsyncCopy()  # type: ignore


def MergeSharedMemoryAllocations():
    """This pass merges multiple TIR-level shared memory allocations
    into one allocation.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass
    """
    return _ffi_api.MergeSharedMemoryAllocations()  # type: ignore


def DefaultGPUSchedule(*, target=None):
    """Set default thread bindings for GPU Functions.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass
    """
    from tvm_tirx.target import Target

    return _ffi_api.DefaultGPUSchedule(Target(target) if target is not None else None)  # type: ignore


def RemoveWeightLayoutRewriteBlock(skip_tensor_rewrite=False):
    """Remove weight layout rewrite block before benchmarking during tuning stage.

    Parameters
    ----------
    skip_tensor_rewrite : bool
        If True, exact rewrite of Tensor, according to the given index map, will be skipped.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass
    """
    return _ffi_api.RemoveWeightLayoutRewriteBlock(skip_tensor_rewrite)  # type: ignore


def RemoveStoreUndef():
    """Remove stores of undefined values from the Stmt.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass
    """
    return _ffi_api.RemoveStoreUndef()  # type: ignore
