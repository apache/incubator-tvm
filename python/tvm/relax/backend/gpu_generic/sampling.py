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
"""Backend kernels for sampling operator."""

import math
from collections.abc import Callable

from tvm.s_tir import script as Ts
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx import Function


def _is_power_of_two(n: int):
    """Check if n is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def gpu_multinomial_from_uniform(
    prob_dtype: str = "float32",
    sample_dtype: str = "float32",
    sample_indices_dtype: str = "int64",
    dtype: str = "int64",
    ty_len: int = 4,
    tx_len: int = 32,
    thread_elem: int = 4,
    eps: float = 1e-6,
    *,
    target=None,
) -> Function:
    """Generate GPU kernel for multinomial_from_uniform operator.

    Parameters
    ----------
    ty_len : int
        The length of `threadIdx.y`

    tx_len : int
        The length of `threadIdx.x`

    thread_elem : int
        The number of elements processed by single thread

    prob_dtype : str
        The probability data type

    sample_dtype : str
        The sample data type

    sample_indices_dtype : str
        The sample indices data type

    dtype : str
        The output data type

    Returns
    -------
    func : Function
        The generated function
    """

    target_dtype = "int32" if "webgpu" in str(target) else "int64"

    TX = T.int64(tx_len)  # threadIdx.x
    TY = T.int64(ty_len)  # threadIdx.y

    # number of elements to be processed by single thread
    thread_elem = T.int64(thread_elem)
    # number of elements to be processed by single warp
    warp_elem = T.int64(tx_len * thread_elem)
    # number of elements to be processed by single block(SM)
    block_elem = T.int64(tx_len * ty_len * thread_elem)

    LOG_TX = T.int64(int(math.log2(tx_len)))
    LOG_TY = T.int64(int(math.log2(ty_len)))

    if (
        not _is_power_of_two(tx_len)
        or not _is_power_of_two(ty_len)
        or not _is_power_of_two(thread_elem)
    ):
        raise ValueError(
            "Configuration of tx_len, ty_len, thread_elem must be power of 2,"
            f"but got {tx_len}, {ty_len}, {thread_elem}"
        )

    @T.macro
    def block_cumsum(
        ty: T.int64,
        tx: T.int64,
        source_local: T.Tensor,
        output_shared: T.Tensor,
    ):
        """cumsum inside block (SM)"""
        # Inclusive scan inside thread
        for i in T.unroll(1, thread_elem):
            source_local[i] += source_local[i - 1]
        # Store data to shared memory
        for i in T.vectorized(thread_elem):
            output_shared[ty * warp_elem + tx * thread_elem + i] = source_local[i]
        # Inclusive scan inside warp
        for i in T.unroll(LOG_TX):
            for j in T.vectorized(thread_elem):
                idx: T.let[T.int64] = ty * warp_elem + tx * thread_elem
                if tx >= (1 << i):
                    output_shared[idx + j] += output_shared[
                        idx - (1 << i) * thread_elem + thread_elem - 1
                    ]
        # Inclusive scan inside block
        for i in T.unroll(1, TY):
            for j in T.vectorized(thread_elem):
                if ty == 0:
                    idx: T.let[T.int64] = i * warp_elem + tx * thread_elem
                    output_shared[idx + j] += output_shared[i * warp_elem - 1]

    def compare_bool_not_equal(a: T.bool, b: T.bool) -> T.bool:
        # Vulkan does not support compare two bool value direct
        # return a != b
        return T.Cast("int8", a) != T.Cast("int8", b)

    @T.macro
    def block_adjacent_difference_left(
        ty: T.int64,
        tx: T.int64,
        source_local: T.Tensor,
        output_local: T.Tensor,
    ):
        with Ts.sblock():
            shared_buf = Ts.sblock_alloc_tensor((TX * TY,), "bool", scope="shared")
            tx_idx: T.let[T.int64] = ty * TX + tx
            shared_buf[tx_idx] = source_local[thread_elem - 1]
            output_local[0] = T.if_then_else(
                tx_idx != 0,
                compare_bool_not_equal(source_local[0], shared_buf[tx_idx - 1]),
                source_local[0],
            )
            for i in T.unroll(1, thread_elem):
                output_local[i] = compare_bool_not_equal(source_local[i], source_local[i - 1])

    def op_reduce_min(a, b):
        return T.min(a, b)

    def op_reduce_sum(a, b):
        return a + b

    @T.macro
    def block_reduce_with_mask(
        ty: T.int64,
        tx: T.int64,
        init_value,
        data_local: T.Tensor,
        output_local: T.Tensor,
        dtype: str,
        reduce_op: Callable,  # T.macro
        mask_local: T.Tensor | None = None,
    ):
        with Ts.sblock():
            local_sum = Ts.sblock_alloc_tensor((), dtype, scope="local")
            shared_buf = Ts.sblock_alloc_tensor((TX * TY,), dtype, scope="shared")
            idx: T.let[T.int64] = ty * TX + tx

            local_sum[()] = T.Cast(dtype, init_value)
            for i in T.unroll(thread_elem):
                if T.constexpr(mask_local is not None):
                    if mask_local[i]:
                        local_sum[()] = reduce_op(local_sum[()], data_local[i])
                else:
                    local_sum[()] = reduce_op(local_sum[()], data_local[i])
            shared_buf[idx] = local_sum[()]

            for i in T.unroll(LOG_TX + LOG_TY):
                if idx % (1 << (i + 1)) == 0:
                    shared_buf[idx] = reduce_op(shared_buf[idx], shared_buf[idx + (1 << i)])
            output_local[()] = shared_buf[0]

    @T.macro
    def single_batch_sampling(
        prob,
        row_idx,
        vocab_size,
        ty,
        tx,
        step_iter,
        threshold,
        aggregate,
        uniform_sample,
        sample_id_local,
    ):
        with Ts.sblock():
            prob_gt_threshold = Ts.sblock_alloc_tensor((thread_elem,), prob_dtype, scope="local")
            cumsum = Ts.sblock_alloc_tensor((block_elem,), prob_dtype, scope="shared")
            greater_than_u = Ts.sblock_alloc_tensor((thread_elem,), "bool", scope="local")
            mask = Ts.sblock_alloc_tensor((thread_elem,), "bool", scope="local")
            valid = Ts.sblock_alloc_tensor((thread_elem,), "bool", scope="local")
            indices = Ts.sblock_alloc_tensor((thread_elem), dtype, scope="local")
            step_aggregate = Ts.sblock_alloc_tensor((), prob_dtype, scope="local")
            # Load prob data from global memory to local memory
            for v in T.unroll(thread_elem):
                idx: T.let[T.int64] = step_iter * block_elem + ty * warp_elem + tx * thread_elem + v
                prob_local: T.let = T.if_then_else(
                    idx < vocab_size,
                    prob[row_idx, idx],
                    T.Cast(prob_dtype, 0),
                )
                prob_gt_threshold[v] = T.if_then_else(
                    prob_local > threshold, prob_local, T.Cast(prob_dtype, 0)
                )
                valid[v] = prob_local > threshold and idx < vocab_size

            block_reduce_with_mask(
                ty,
                tx,
                init_value=0,
                data_local=prob_gt_threshold,
                output_local=step_aggregate,
                dtype=prob_dtype,
                reduce_op=op_reduce_sum,
                mask_local=None,
            )
            if T.gpu_thread_invariant(aggregate[()] + step_aggregate[()] >= uniform_sample - eps):
                block_cumsum(ty, tx, prob_gt_threshold, cumsum)
                # Note: it should be `T.vectorized` instead of `T.unroll`
                # However, it will cause vulkan codegen error
                for v in T.unroll(thread_elem):
                    greater_than_u[v] = (
                        cumsum[ty * warp_elem + tx * thread_elem + v] + aggregate[()]
                        >= uniform_sample - eps
                    )

                block_adjacent_difference_left(ty, tx, greater_than_u, mask)
                # Same as above, it should be `T.vectorized`
                for v in T.unroll(thread_elem):
                    mask[v] = mask[v] and valid[v]
                    indices[v] = step_iter * block_elem + ty * warp_elem + tx * thread_elem + v
                block_reduce_with_mask(
                    ty,
                    tx,
                    init_value=vocab_size - 1,
                    data_local=indices,
                    output_local=sample_id_local,
                    dtype=dtype,
                    reduce_op=op_reduce_min,
                    mask_local=mask,
                )

            aggregate[()] += step_aggregate[()]

    n = T.dynamic("n", "int64")
    vocab_size = T.dynamic("vocab_size", "int64")
    batch_size = T.dynamic("batch_size", "int64")

    @Ts.function
    def parallel_sampling_from_prob(
        prob: T.Tensor((n, vocab_size), prob_dtype),
        uniform_samples: T.Tensor((batch_size, 1), sample_dtype),
        row_indices: T.Tensor((batch_size, 1), sample_indices_dtype),
        token_ids: T.Tensor((batch_size, 1), dtype),
    ):
        T.func_attr({"tirx.is_scheduled": True})
        # match buffers

        # local buffers
        aggregate = Ts.sblock_alloc_tensor((), prob_dtype, scope="local")
        sample_id_local = Ts.sblock_alloc_tensor((), dtype, scope="local")
        step_iter = Ts.sblock_alloc_tensor((), "int32", scope="local")

        for bx in T.parallel(batch_size, annotations={"thread_binding": "blockIdx.x"}):
            row_idx: T.let[T.int64] = T.Cast("int64", row_indices[bx, 0])
            for ty in T.parallel(TY, annotations={"thread_binding": "threadIdx.y"}):
                for tx in T.parallel(TX, annotations={"thread_binding": "threadIdx.x"}):
                    u: T.let[T.float32] = uniform_samples[bx, 0]
                    aggregate[()] = T.Cast(prob_dtype, 0)
                    step_iter[()] = T.int32(0)
                    # at least one iteration
                    while T.gpu_thread_invariant(
                        (step_iter[()] == 0 or aggregate[()] < u - eps)
                        and T.Cast(target_dtype, step_iter[()])
                        < T.Cast(target_dtype, T.ceildiv(vocab_size, block_elem))
                    ):
                        single_batch_sampling(
                            prob,
                            row_idx,
                            vocab_size,
                            ty,
                            tx,
                            T.Cast(target_dtype, step_iter[()]),
                            0.0,
                            aggregate,
                            u,
                            sample_id_local,
                        )
                        step_iter[()] += 1
                    if tx == 0 and ty == 0:
                        token_ids[bx, 0] = sample_id_local[()]

    return parallel_sampling_from_prob


def generic_get_sample_index(
    prob_dtype: str = "float32",
    sample_dtype: str = "float32",
    sample_indices_dtype: str = "int64",
    dtype: str = "int64",
):
    """Generate a generic get_sample_index kernel."""

    batch = T.dynamic("batch", "int64")
    vocab_size = T.dynamic("vocab_size", "int64")
    out_batch = T.dynamic("out_batch", "int64")

    @Ts.function(private=True)
    def _get_sample_index(
        prob: T.Tensor((batch, vocab_size), prob_dtype),
        usample: T.Tensor((out_batch, 1), sample_dtype),
        sample_indices: T.Tensor((out_batch, 1), sample_indices_dtype),
        output_index: T.Tensor((out_batch, 1), dtype),
    ):
        for ax0, ax1 in T.grid(out_batch, vocab_size):
            with Ts.sblock("T_get_sample_index"):
                v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                Ts.writes(output_index[v_ax0, 0])
                if (
                    usample[v_ax0, T.int64(0)] < prob[sample_indices[v_ax0, T.int64(0)], v_ax1]
                    or v_ax1 + 1 == vocab_size
                ):
                    if v_ax1 == 0:
                        output_index[v_ax0, 0] = 0
                    elif (
                        usample[v_ax0, T.int64(0)]
                        >= prob[sample_indices[v_ax0, T.int64(0)], v_ax1 - 1]
                    ):
                        output_index[v_ax0, 0] = v_ax1

    return _get_sample_index
