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
# pylint: disable=invalid-name, too-many-nested-blocks
"""Backend kernels for cumsum operator."""

import math

from tvm.s_tir import script as Ts
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx import Function


def _is_power_of_two(n: int):
    """Check if n is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def _get_total_rounds(n, log_block_n: int, index_bits: int):
    """Count the hierarchy levels without introducing integer division."""
    total_rounds = T.int64(0)
    for round_index in range(1, (index_bits - 1) // log_block_n + 1):
        threshold = T.int64(1 << (round_index * log_block_n - 1))
        total_rounds += T.Cast("int64", n > threshold)
    return total_rounds


def gpu_2d_continuous_cumsum(
    ty_len: int = 4,
    tx_len: int = 32,
    thread_elem: int = 4,
    in_dtype: str = "int32",
    out_dtype: str | None = None,
    index_bits: int = 64,
) -> Function:
    """Generate GPU kernel for 2D continuous cumsum, i.e. The cumsum axis is -1

    Parameters
    ----------
    ty_len : int
        The length of `threadIdx.y`

    tx_len : int
        The length of `threadIdx.x`

    thread_elem : int
        The number of elements processed by single thread

    in_dtype : str
        The input data type

    out_dtype : Optional[str]
        The output data type, if None, it will be the same as in_dtype

    index_bits : int
        The number of bits available for signed index expressions

    Returns
    -------
    cumsum : Function
        The generated cumsum kernel
    """

    out_dtype = out_dtype or in_dtype

    # Configuration for GPU kernel
    TX = T.int64(tx_len)  # threadIdx.x
    TY = T.int64(ty_len)  # threadIdx.y
    N = T.int64(thread_elem)  # number of elements in single thread

    if not _is_power_of_two(TX) or not _is_power_of_two(TY) or not _is_power_of_two(N):
        raise ValueError("Configuration of TX, TY, N must be power of 2")
    if index_bits not in (32, 64):
        raise ValueError("index_bits must be either 32 or 64")

    # number of elements to be processed by single warp
    warp_elem = T.int64(tx_len * thread_elem)
    # number of elements to be processed by single block(SM)
    block_elem = T.int64(tx_len * ty_len * thread_elem)

    LOG_TX = T.int64(int(math.log2(tx_len)))
    log_block_n = int(math.log2(tx_len * ty_len * thread_elem))
    LOG_BLOCK_N = T.int64(log_block_n)
    MAX_INDEX_SHIFT = T.int64(index_bits - 1)

    @T.macro
    def block_inclusive_inside_block(
        batch: T.int64,
        cur_len: T.int64,
        num_blocks: T.int64,
        source: T.Tensor,
        output: T.Tensor,
        tmp_buf: T.Tensor,
        src_offset: T.int64,
        tmp_offset: T.int64,
    ):
        for by in T.parallel(batch, annotations={"thread_binding": "blockIdx.y"}):
            for bx in T.parallel(num_blocks, annotations={"thread_binding": "blockIdx.x"}):
                with Ts.sblock():
                    local_buf = Ts.sblock_alloc_tensor((thread_elem,), out_dtype, scope="local")
                    shared_buf = Ts.sblock_alloc_tensor((block_elem,), out_dtype, scope="shared")
                    for ty in T.parallel(TY, annotations={"thread_binding": "threadIdx.y"}):
                        for tx in T.parallel(TX, annotations={"thread_binding": "threadIdx.x"}):
                            tx_idx: T.let[T.int64] = (
                                bx * block_elem + ty * warp_elem + tx * thread_elem
                            )
                            # Load data from global memory
                            for i in T.vectorized(N):
                                local_buf[i] = T.if_then_else(
                                    tx_idx + i < cur_len,
                                    T.Cast(out_dtype, source[by, src_offset + tx_idx + i]),
                                    T.Cast(out_dtype, 0),
                                )
                            # Inclusive scan inside thread
                            for i in T.unroll(1, N):
                                local_buf[i] += local_buf[i - 1]
                            # Store data to shared memory
                            for i in T.vectorized(N):
                                shared_buf[ty * warp_elem + tx * thread_elem + i] = local_buf[i]
                            # Inclusive scan inside warp
                            for i in T.unroll(LOG_TX):
                                for j in T.vectorized(N):
                                    idx: T.let[T.int64] = ty * warp_elem + tx * thread_elem
                                    if tx >= (1 << i):
                                        shared_buf[idx + j] += shared_buf[
                                            idx - (1 << i) * thread_elem + N - 1
                                        ]
                            # Inclusive scan inside block
                            for i in T.unroll(1, TY):
                                for j in T.vectorized(N):
                                    if ty == 0:
                                        idx: T.let[T.int64] = i * warp_elem + tx * thread_elem
                                        shared_buf[idx + j] += shared_buf[i * warp_elem - 1]
                            # Write sum of block to global memory
                            for i in T.vectorized(N):
                                idx: T.let[T.int64] = ty * warp_elem + tx * thread_elem + i
                                if bx * block_elem + idx < cur_len:
                                    output[by, src_offset + bx * block_elem + idx] = shared_buf[idx]
                            if tx == 0 and ty == 0:
                                for i in T.vectorized(N):
                                    tmp_buf[by, tmp_offset + bx] = shared_buf[block_elem - 1]

    @T.macro
    def update_cross_block(
        batch: T.int64,
        cur_len: T.int64,
        num_blocks: T.int64,
        source: T.Tensor,
        output: T.Tensor,
        src_offset: T.int64,
        out_offset: T.int64,
    ):
        for by in T.parallel(batch, annotations={"thread_binding": "blockIdx.y"}):
            for bx in T.parallel(num_blocks, annotations={"thread_binding": "blockIdx.x"}):
                for ty in T.parallel(TY, annotations={"thread_binding": "threadIdx.y"}):
                    for tx in T.parallel(TX, annotations={"thread_binding": "threadIdx.x"}):
                        for i in T.serial(N):
                            idx: T.let[T.int64] = bx * block_elem + ty * warp_elem + i * TX + tx
                            if idx < cur_len:
                                output[by, out_offset + idx] += T.if_then_else(
                                    bx > 0, source[by, src_offset + bx - 1], 0
                                )

    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @Ts.function(private=True)
    def cumsum(A: T.Tensor([m, n], dtype=in_dtype), Out: T.Tensor([m, n], dtype=out_dtype)):
        T.func_attr({"tirx.is_scheduled": True})  # prevent further scheduling

        Tmp = T.alloc_tensor(ty_args=[T.Tensor([m, n], out_dtype)])
        # LowerIntrin may implement signed FloorDiv using a sign-bit shift.  Keep
        # hierarchy counting division-free so WebGPU can narrow indices to int32.
        total_rounds: T.let[T.int64] = _get_total_rounds(n, log_block_n, index_bits)

        block_inclusive_inside_block(
            m,
            n,
            T.ceildiv(n, block_elem),
            A,
            Out,
            Tmp,
            src_offset=T.int64(0),
            tmp_offset=T.int64(0),
        )
        for i in range(total_rounds):
            shift: T.let[T.int64] = T.min(T.max(LOG_BLOCK_N * (i + 1), T.int64(0)), MAX_INDEX_SHIFT)
            block_shift: T.let[T.int64] = T.min(shift + LOG_BLOCK_N, MAX_INDEX_SHIFT)
            # n is non-negative and WebGPU indices are narrowed to signed int32.
            # Spell out positive ceildiv by a power of two so lowering does not
            # introduce an int64 sign-bit test (`remainder >> 63`).
            cur_len: T.let[T.int64] = ((n - 1) >> shift) + 1
            num_blocks: T.let[T.int64] = ((n - 1) >> block_shift) + 1
            block_inclusive_inside_block(
                m,
                cur_len,
                num_blocks,
                Tmp,
                Tmp,
                Tmp,
                src_offset=i * T.ceildiv(n, block_elem),
                tmp_offset=(i + 1) * T.ceildiv(n, block_elem),
            )
        reverse_rounds: T.let[T.int64] = T.max(total_rounds - 1, 0)
        for i in range(reverse_rounds):
            real_idx: T.let[T.int64] = reverse_rounds - 1 - i
            shift: T.let[T.int64] = T.min(
                T.max(LOG_BLOCK_N * (real_idx + 1), T.int64(0)), MAX_INDEX_SHIFT
            )
            block_shift: T.let[T.int64] = T.min(shift + LOG_BLOCK_N, MAX_INDEX_SHIFT)
            cur_len: T.let[T.int64] = ((n - 1) >> shift) + 1
            num_blocks: T.let[T.int64] = ((n - 1) >> block_shift) + 1
            update_cross_block(
                m,
                cur_len,
                num_blocks,
                Tmp,
                Tmp,
                src_offset=(real_idx + 1) * T.ceildiv(n, block_elem),
                out_offset=real_idx * T.ceildiv(n, block_elem),
            )
        update_cross_block(m, n, T.ceildiv(n, block_elem), Tmp, Out, src_offset=0, out_offset=0)

    return cumsum


def gpu_3d_axis_1_cumsum(
    tx_len: int = 128,
    in_dtype: str = "int32",
    out_dtype: str | None = None,
) -> Function:
    """Generate a correctness fallback that scans axis 1 of a contiguous 3D tensor.

    Each thread handles one pair of outer and inner indices and scans the
    middle axis sequentially.  The dispatcher collapses arbitrary-rank inputs
    around the scan axis into this 3D representation.  This fallback avoids a
    transposed scan on targets where that lowering is unavailable; it is not a
    parallel scan optimization for rank-3 tensors.
    """

    out_dtype = out_dtype or in_dtype
    TX = T.int64(tx_len)

    outer = T.dynamic("outer", "int64")
    scan = T.dynamic("scan", "int64")
    inner = T.dynamic("inner", "int64")

    @Ts.function(private=True)
    def cumsum(
        A: T.Tensor([outer, scan, inner], dtype=in_dtype),
        Out: T.Tensor([outer, scan, inner], dtype=out_dtype),
    ):
        T.func_attr({"tirx.is_scheduled": True})

        for bx in T.parallel(
            T.ceildiv(outer * inner, TX), annotations={"thread_binding": "blockIdx.x"}
        ):
            for tx in T.parallel(TX, annotations={"thread_binding": "threadIdx.x"}):
                row: T.let[T.int64] = bx * TX + tx
                with Ts.sblock():
                    accumulator = Ts.sblock_alloc_tensor((), out_dtype, scope="local")
                    if row < outer * inner:
                        outer_idx: T.let[T.int64] = row // inner
                        inner_idx: T.let[T.int64] = row % inner
                        accumulator[()] = T.Cast(out_dtype, 0)
                        for k in T.serial(scan):
                            accumulator[()] += T.Cast(out_dtype, A[outer_idx, k, inner_idx])
                            Out[outer_idx, k, inner_idx] = accumulator[()]

    return cumsum
