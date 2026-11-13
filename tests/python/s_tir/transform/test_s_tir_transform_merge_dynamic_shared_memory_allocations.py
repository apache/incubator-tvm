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
# ruff: noqa: F401, F841
import numpy as np

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import s_tir
from tvm.s_tir import script as Ts
from tvm.topi.math import cast
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def test_matmul_t_tensor():
    """Shared allocations should be merged, preserving DeclTensor if present

    This test uses a matmul Function adapted from
    test_matmul_dyn_shared, using `T.Tensor` (Allocate without
    DeclTensor) for the replaced allocations.
    """
    transform = tvm.s_tir.transform.MergeSharedMemoryAllocations()
    tensor_func = T.Tensor

    @I.ir_module
    class Before:
        @Ts.function
        def main(
            A: T.Tensor((1024, 1024), "float16"),
            B: T.Tensor((1024, 1024), "float16"),
            matmul: T.Tensor((1024, 1024), "float32"),
        ):
            A_flat = T.decl_tensor(A.data, ty_args=[T.Tensor(1048576, "float16", data=A.data)])
            B_flat = T.decl_tensor(B.data, ty_args=[T.Tensor(1048576, "float16", data=B.data)])
            matmul_flat = T.decl_tensor(
                matmul.data, ty_args=[T.Tensor(1048576, "float32", data=matmul.data)]
            )

            threadIdx_x = T.launch_thread("threadIdx.x", 16)
            C_local = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
            A_sh = T.alloc_tensor(ty_args=[T.Tensor((256,), "float16", scope="shared.dyn")])
            B_sh = T.alloc_tensor(ty_args=[T.Tensor((256,), "float16", scope="shared.dyn")])
            C_sh = T.alloc_tensor(ty_args=[T.Tensor((256,), "float32", scope="shared.dyn")])
            threadIdx_y = T.launch_thread("threadIdx.y", 16)
            blockIdx_x = T.launch_thread("blockIdx.x", 64)
            blockIdx_y = T.launch_thread("blockIdx.y", 64)

            C_local[0] = T.float32(0)
            for i in range(64):
                A_sh[threadIdx_y * 16 + threadIdx_x] = A_flat[
                    blockIdx_y * 16384 + threadIdx_y * 1024 + i * 16 + threadIdx_x
                ]

                B_sh[threadIdx_y * 16 + threadIdx_x] = B_flat[
                    i * 16384 + threadIdx_y * 1024 + blockIdx_x * 16 + threadIdx_x
                ]
                T.gpu_storage_sync("shared")
                for k in range(16):
                    C_local[0] = C_local[0] + T.Cast(
                        "float32",
                        A_sh[threadIdx_y * 16 + k] * B_sh[k * 16 + threadIdx_x],
                    )
                T.gpu_storage_sync("shared")

            C_sh[threadIdx_y * 16 + threadIdx_x] = C_local[0]
            T.gpu_storage_sync("shared.dyn")

            matmul_flat[blockIdx_y * 16384 + threadIdx_y * 1024 + blockIdx_x * 16 + threadIdx_x] = (
                C_sh[threadIdx_y * 16 + threadIdx_x]
            )

    @I.ir_module
    class Expected:
        @Ts.function
        def main(
            A: T.Tensor((1024, 1024), "float16"),
            B: T.Tensor((1024, 1024), "float16"),
            matmul: T.Tensor((1024, 1024), "float32"),
        ):
            A_flat = T.decl_tensor(A.data, ty_args=[T.Tensor(1048576, "float16", data=A.data)])
            B_flat = T.decl_tensor(B.data, ty_args=[T.Tensor(1048576, "float16", data=B.data)])
            matmul_flat = T.decl_tensor(
                matmul.data, ty_args=[T.Tensor(1048576, "float32", data=matmul.data)]
            )

            threadIdx_x = T.launch_thread("threadIdx.x", 16)

            buf_dyn_shmem = T.alloc_tensor(ty_args=[T.Tensor((1024,), T.uint8, scope="shared.dyn")])

            C_local = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
            A_sh = T.decl_tensor(
                buf_dyn_shmem.data, ty_args=[T.Tensor(256, "float16", scope="shared.dyn")]
            )
            B_sh = T.decl_tensor(
                buf_dyn_shmem.data, ty_args=[T.Tensor(256, "float16", scope="shared.dyn")]
            )
            C_sh = T.decl_tensor(
                buf_dyn_shmem.data, ty_args=[T.Tensor(256, "float32", scope="shared.dyn")]
            )

            threadIdx_y = T.launch_thread("threadIdx.y", 16)
            blockIdx_x = T.launch_thread("blockIdx.x", 64)
            blockIdx_y = T.launch_thread("blockIdx.y", 64)

            C_local[0] = T.float32(0)
            for i in range(64):
                A_sh[threadIdx_y * 16 + threadIdx_x + 256] = A_flat[
                    blockIdx_y * 16384 + threadIdx_y * 1024 + i * 16 + threadIdx_x
                ]
                B_sh[threadIdx_y * 16 + threadIdx_x] = B_flat[
                    i * 16384 + threadIdx_y * 1024 + blockIdx_x * 16 + threadIdx_x
                ]
                T.gpu_storage_sync("shared")
                for k in range(16):
                    C_local[0] = C_local[0] + T.Cast(
                        "float32",
                        A_sh[threadIdx_y * 16 + k + 256] * B_sh[k * 16 + threadIdx_x],
                    )
                T.gpu_storage_sync("shared")

            C_sh[threadIdx_y * 16 + threadIdx_x] = C_local[0]
            T.gpu_storage_sync("shared.dyn")

    After = transform(Before)
    script = After["main"].script()
    # Verify merged allocation: one shared.dyn tensor of 1024 bytes (256*2 float16 + 256 float32)
    assert "alloc_tensor(ty_args=[T.Tensor((1024,)" in script
    assert "T.uint8" in script
    assert '"shared.dyn"' in script
    # Verify storage sync calls preserved
    assert "gpu_storage_sync" in script
    # Verify offset indexing (shared memory merged)
    assert "+ 256" in script


def test_matmul_decl_tensor():
    """Shared allocations should be merged, preserving DeclTensor if present

    This test uses a matmul Function adapted from
    test_matmul_dyn_shared, using `T.decl_tensor` (Allocate followed by DeclTensor)
    for the replaced allocations.
    """
    transform = tvm.s_tir.transform.MergeSharedMemoryAllocations()
    tensor_func = T.decl_tensor

    @I.ir_module
    class Before:
        @Ts.function
        def main(
            A: T.Tensor((1024, 1024), "float16"),
            B: T.Tensor((1024, 1024), "float16"),
            matmul: T.Tensor((1024, 1024), "float32"),
        ):
            A_flat = T.decl_tensor(A.data, ty_args=[T.Tensor(1048576, "float16", data=A.data)])
            B_flat = T.decl_tensor(B.data, ty_args=[T.Tensor(1048576, "float16", data=B.data)])
            matmul_flat = T.decl_tensor(
                matmul.data, ty_args=[T.Tensor(1048576, "float32", data=matmul.data)]
            )

            threadIdx_x = T.launch_thread("threadIdx.x", 16)
            C_local = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
            A_sh = T.alloc_tensor(ty_args=[T.Tensor((256,), "float16", scope="shared.dyn")])
            B_sh = T.alloc_tensor(ty_args=[T.Tensor((256,), "float16", scope="shared.dyn")])
            C_sh = T.alloc_tensor(ty_args=[T.Tensor((256,), "float32", scope="shared.dyn")])
            threadIdx_y = T.launch_thread("threadIdx.y", 16)
            blockIdx_x = T.launch_thread("blockIdx.x", 64)
            blockIdx_y = T.launch_thread("blockIdx.y", 64)

            C_local[0] = T.float32(0)
            for i in range(64):
                A_sh[threadIdx_y * 16 + threadIdx_x] = A_flat[
                    blockIdx_y * 16384 + threadIdx_y * 1024 + i * 16 + threadIdx_x
                ]

                B_sh[threadIdx_y * 16 + threadIdx_x] = B_flat[
                    i * 16384 + threadIdx_y * 1024 + blockIdx_x * 16 + threadIdx_x
                ]
                T.gpu_storage_sync("shared")
                for k in range(16):
                    C_local[0] = C_local[0] + T.Cast(
                        "float32",
                        A_sh[threadIdx_y * 16 + k] * B_sh[k * 16 + threadIdx_x],
                    )
                T.gpu_storage_sync("shared")

            C_sh[threadIdx_y * 16 + threadIdx_x] = C_local[0]
            T.gpu_storage_sync("shared.dyn")

            matmul_flat[blockIdx_y * 16384 + threadIdx_y * 1024 + blockIdx_x * 16 + threadIdx_x] = (
                C_sh[threadIdx_y * 16 + threadIdx_x]
            )

    After = transform(Before)
    script = After["main"].script()
    # Verify merged allocation: one shared.dyn tensor of 1024 bytes
    assert "alloc_tensor(ty_args=[T.Tensor((1024,)" in script
    assert "T.uint8" in script
    assert '"shared.dyn"' in script
    assert "gpu_storage_sync" in script
    assert "+ 256" in script


def test_simple_alloc_no_reuse():
    """Test alloc and free within the same scope."""
    transform = tvm.s_tir.transform.MergeSharedMemoryAllocations()

    @I.ir_module
    class Before:
        @Ts.function
        def main():
            threadIdx_x = T.launch_thread("threadIdx.x", 128)
            A_sh = T.alloc_tensor(ty_args=[T.Tensor((128,), T.float32, scope="shared.dyn")])
            B_sh = T.alloc_tensor(ty_args=[T.Tensor((128,), T.float32, scope="shared.dyn")])
            B_sh[threadIdx_x] = A_sh[threadIdx_x]

    After = transform(Before)
    script = After["main"].script()
    # Verify merged allocation: 1024 bytes (128*4 + 128*4)
    assert "alloc_tensor(ty_args=[T.Tensor((1024,)" in script
    assert "T.uint8" in script
    assert '"shared.dyn"' in script
    # Verify offset indexing
    assert "+ 128" in script


def test_simple_alloc_reuse():
    """Test alloc and free within the same scope with a reuse chance."""
    transform = tvm.s_tir.transform.MergeSharedMemoryAllocations()

    @I.ir_module
    class Before:
        @Ts.function
        def main():
            threadIdx_x = T.launch_thread("threadIdx.x", 128)
            A_sh = T.alloc_tensor(ty_args=[T.Tensor((128,), T.float32, scope="shared.dyn")])
            B_sh = T.alloc_tensor(ty_args=[T.Tensor((128,), T.float32, scope="shared.dyn")])
            A_sh[threadIdx_x] = 0
            B_sh[threadIdx_x] = 0

    After = transform(Before)
    script = After["main"].script()
    # Verify merged allocation: 512 bytes (128*4, reusable)
    assert "alloc_tensor(ty_args=[T.Tensor((512,)" in script
    assert "T.uint8" in script
    assert '"shared.dyn"' in script


def test_async_copy():
    """Test async copy in shared memory."""
    transform = tvm.s_tir.transform.MergeSharedMemoryAllocations()

    @I.ir_module
    class Before:
        @Ts.function
        def main(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
            threadIdx_x = T.launch_thread("threadIdx.x", 128)
            A_sh = T.alloc_tensor(ty_args=[T.Tensor((128,), T.float32, scope="shared.dyn")])
            B_sh = T.alloc_tensor(ty_args=[T.Tensor((128,), T.float32, scope="shared.dyn")])
            T.s_tir.cp_async_raw(A_sh.data, threadIdx_x, A.data, threadIdx_x, 512, ty="float32")
            T.s_tir.cp_async_raw(B_sh.data, threadIdx_x, B.data, threadIdx_x, 512, ty="float32")

    After = transform(Before)
    # The pass merges shared.dyn allocations. A_sh and B_sh are accessed
    # sequentially inside the launch_thread with non-overlapping lifetimes,
    # so the liveness analysis allows reuse — both fit in 512 bytes
    # (= 128 elements * 4 bytes).
    script = After["main"].script()
    # Verify merged allocation (512 bytes - A_sh and B_sh can be reused)
    assert "T.uint8" in script and '"shared.dyn"' in script and "(512,)" in script
    # Verify cp_async uses typed views of the merged byte allocation.  Its
    # offsets remain in float32 elements and are scaled by the intrinsic
    # lowering, rather than being pre-scaled as byte offsets here.
    assert (
        "A_sh = T.decl_tensor(buf_dyn_shmem.data, ty_args=[T.Tensor((128,), "
        'T.float32, scope="shared.dyn")])' in script
    )
    assert (
        "B_sh = T.decl_tensor(buf_dyn_shmem.data, ty_args=[T.Tensor((128,), "
        'T.float32, scope="shared.dyn")])' in script
    )
    assert (
        "T.s_tir.cp_async_raw(A_sh.data, threadIdx_x, "
        "A.data, threadIdx_x, 512, ty=T.float32)" in script
    )
    assert (
        "T.s_tir.cp_async_raw(B_sh.data, threadIdx_x, "
        "B.data, threadIdx_x, 512, ty=T.float32)" in script
    )


def test_decl_tensor_alias_extends_allocation_lifetime():
    """Access through a typed view keeps its source allocation live."""
    transform = tvm.s_tir.transform.MergeSharedMemoryAllocations()

    @I.ir_module
    class Before:
        @Ts.function
        def main(C: T.Tensor((128,), "float32")):
            threadIdx_x = T.launch_thread("threadIdx.x", 128)
            A_sh = T.alloc_tensor(ty_args=[T.Tensor((128,), T.float32, scope="shared.dyn")])
            B_sh = T.alloc_tensor(ty_args=[T.Tensor((128,), T.float32, scope="shared.dyn")])
            A_view = T.decl_tensor(
                A_sh.data, ty_args=[T.Tensor((128,), T.float32, scope="shared.dyn")]
            )
            B_view = T.decl_tensor(
                B_sh.data, ty_args=[T.Tensor((128,), T.float32, scope="shared.dyn")]
            )
            A_view[threadIdx_x] = T.float32(1)
            B_view[threadIdx_x] = T.float32(2)
            C[threadIdx_x] = A_view[threadIdx_x] + B_view[threadIdx_x]

    After = transform(Before)
    script = After["main"].script()
    assert 'alloc_tensor(ty_args=[T.Tensor((1024,), T.uint8, scope="shared.dyn")])' in script
    assert "B_view[threadIdx_x + 128]" in script
    assert "A_view[threadIdx_x]" in script


def test_multi_thread_extent_blocks():
    """Each launch_thread block must get its own merged tensor.

    Reproduces the scoping bug from PR #19605: a single Function
    with two sibling launch_thread regions, each containing its
    own shared.dyn allocations. The merged tensor must be allocated
    inside each kernel body — not just the first.
    """
    transform = tvm.s_tir.transform.MergeSharedMemoryAllocations()

    @I.ir_module(check_well_formed=False)
    class Before:
        @Ts.function(check_well_formed=False)
        def main(
            X: T.Tensor((128,), "float32"),
            Y: T.Tensor((128,), "float32"),
        ):
            X_flat = T.decl_tensor(X.data, ty_args=[T.Tensor(128, "float32", data=X.data)])
            Y_flat = T.decl_tensor(Y.data, ty_args=[T.Tensor(128, "float32", data=Y.data)])

            # First kernel launch
            with T.launch_thread("threadIdx.x", 128) as tx0:
                A_sh = T.alloc_tensor(ty_args=[T.Tensor((128,), T.float32, scope="shared.dyn")])
                B_sh = T.alloc_tensor(ty_args=[T.Tensor((128,), T.float32, scope="shared.dyn")])
                A_sh[tx0] = X_flat[tx0]
                B_sh[tx0] = A_sh[tx0]
                X_flat[tx0] = B_sh[tx0]

            # Second kernel launch — must NOT see kernel #0's merged tensor.
            with T.launch_thread("threadIdx.x", 128) as tx1:
                C_sh = T.alloc_tensor(ty_args=[T.Tensor((128,), T.float32, scope="shared.dyn")])
                D_sh = T.alloc_tensor(ty_args=[T.Tensor((128,), T.float32, scope="shared.dyn")])
                C_sh[tx1] = Y_flat[tx1]
                D_sh[tx1] = C_sh[tx1]
                Y_flat[tx1] = D_sh[tx1]

    After = transform(Before)
    script = After["main"].script()

    # Two merged allocations — one per launch_thread body.
    # Each of the four original 128-float32 tensors (A_sh, B_sh, C_sh, D_sh)
    # gets merged within its own kernel scope.
    assert script.count("shared.dyn") >= 2, (
        "Expected at least two shared.dyn allocations (one per kernel)"
    )
    assert script.count("alloc_tensor") >= 2, (
        "Expected at least two alloc_tensor nodes (one merged buf per kernel)"
    )

    # Both launch_thread blocks must contain their own merged tensor —
    # they must NOT share the same buf_dyn_shmem variable.
    # Structurally verify that the first kernel's body accesses are
    # not rewritten to the second kernel's buf_dyn_shmem (and vice versa).
    first_block, second_block = script.split('with T.launch_thread("threadIdx.x", 128) as tx1:')
    assert "buf_dyn_shmem" in first_block, "Kernel 1 must have a merged tensor"
    assert "buf_dyn_shmem" in second_block, "Kernel 2 must have a merged tensor"

    # End-to-end: post-merge IR must remain well-formed through
    # the host/device split — this is the exact ordering from
    # PR #19605 that triggers the scoping bug.
    target = tvm_tirx.target.Target("llvm")
    mod_with_target = tvm_tirx.IRModule({"main": After["main"].with_attr({"target": target})})
    split = tvm_tirx.tirx.transform.SplitHostDevice()
    # If kernel #1 referenced an undefined buf_dyn_shmem, this
    # would raise during well-formedness checking inside SplitHostDevice.
    split(mod_with_target)


if __name__ == "__main__":
    tvm_tirx.testing.main()
