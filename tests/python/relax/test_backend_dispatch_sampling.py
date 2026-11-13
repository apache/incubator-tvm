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
# pylint: disable=missing-docstring
# ruff: noqa: E501

import tvm_tirx
import tvm_tirx.script
import tvm_tirx.testing
from tvm.relax import script as R
from tvm.relax.backend import DispatchSampling
from tvm.s_tir import script as Ts
from tvm_tirx.ir.base import assert_structural_equal
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


@I.ir_module
class MultiFromUniformModule:
    @R.function
    def foo(
        prob: R.Tensor((3, 5), "float32"),
        uniform_sample: R.Tensor((6, 1), "float32"),
        sample_indices: R.Tensor((6, 1), "int64"),
    ):
        with R.dataflow():
            gv = R.multinomial_from_uniform(prob, uniform_sample, sample_indices, dtype="int64")
            R.output(gv)
        return gv


def test_dispatch_multinomial_from_uniform_generic():
    # fmt: off
    batch = T.dynamic("batch", "int64")
    vocab_size = T.dynamic("vocab_size", "int64")
    out_batch = T.dynamic("out_batch", "int64")

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def get_sample_index(prob: T.Tensor((batch, vocab_size), "float32"), usample: T.Tensor((out_batch, 1), "float32"), sample_indices: T.Tensor((out_batch, 1), 'int64'), output_index: T.Tensor((out_batch, 1), 'int64')):

            # with Ts.sblock("root"):
            for ax0, ax1 in T.grid(out_batch, vocab_size):
                with Ts.sblock("T_get_sample_index"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    if usample[v_ax0, T.int64(0)] < prob[sample_indices[v_ax0, T.int64(0)], v_ax1] or v_ax1 + T.int64(1) == vocab_size:
                        if v_ax1 == T.int64(0):
                            output_index[v_ax0, 0] = T.int64(0)
                        else:
                            if usample[v_ax0, T.int64(0)] >= prob[sample_indices[v_ax0, T.int64(0)], v_ax1 - T.int64(1)]:
                                output_index[v_ax0, 0] = v_ax1

        @R.function
        def foo(prob: R.Tensor((3, 5), dtype="float32"), uniform_sample: R.Tensor((6, 1), dtype="float32"), sample_indices: R.Tensor((6, 1), dtype="int64")) -> R.Tensor((6, 1), dtype="int64"):
            cls = Expected
            with R.dataflow():
                lv: R.Tensor((3, 5), dtype="float32") = R.cumsum(prob, axis=1, dtype="float32", exclusive=0)
                gv = R.call_tir(cls.get_sample_index, (lv, uniform_sample, sample_indices), ty_args=[R.Tensor((6, 1), dtype="int64")])
                R.output(gv)
            return gv
    # fmt: on

    mod = DispatchSampling(target=tvm_tirx.target.Target("llvm"))(MultiFromUniformModule)

    assert_structural_equal(mod, Expected)


def test_dispatch_multinomial_from_uniform_gpu():
    # fmt: off
    n = T.dynamic("n", "int64")
    vocab_size = T.dynamic("vocab_size", "int64")
    batch_size = T.dynamic("batch_size", "int64")

    @I.ir_module
    class Expected:
        @Ts.function
        def parallel_sampling_from_prob(prob: T.Tensor((n, vocab_size), "float32"), uniform_samples: T.Tensor((batch_size, 1), "float32"), row_indices: T.Tensor((batch_size, 1), 'int64'), token_ids: T.Tensor((batch_size, 1), 'int64')):
            T.func_attr({"tirx.is_scheduled": True})

            # with Ts.sblock("root"):
            aggregate = Ts.sblock_alloc_tensor((), scope="local")
            sample_id_local = Ts.sblock_alloc_tensor((), "int64", scope="local")
            step_iter = Ts.sblock_alloc_tensor((), "int32", scope="local")
            for bx in T.parallel(batch_size, annotations={"thread_binding": "blockIdx.x"}):
                row_idx: T.let[T.int64] = T.Cast("int64", row_indices[bx, 0])
                for ty in T.parallel(T.int64(4), annotations={"thread_binding": "threadIdx.y"}):
                    for tx in T.parallel(T.int64(32), annotations={"thread_binding": "threadIdx.x"}):
                        u: T.let[T.float32] = uniform_samples[bx, 0]
                        aggregate[()] = T.Cast("float32", 0)
                        step_iter[()] = 0
                        while T.gpu_thread_invariant((step_iter[()] == 0 or aggregate[()] < u - T.float32(9.9999999999999995e-07)) and T.Cast("int64", step_iter[()]) < T.Cast("int64", (vocab_size + T.int64(512) - T.int64(1)) // T.int64(512))):
                            with Ts.sblock(""):
                                Ts.reads(step_iter[()], prob[row_idx, T.Cast("int64", step_iter[()]) * T.int64(512) + ty * T.int64(128) + tx * T.int64(4):T.Cast("int64", step_iter[()]) * T.int64(512) + ty * T.int64(128) + tx * T.int64(4) + T.int64(4)], aggregate[()])
                                Ts.writes(sample_id_local[()], aggregate[()])
                                prob_gt_threshold = Ts.sblock_alloc_tensor((T.int64(4),), scope="local")
                                cumsum = Ts.sblock_alloc_tensor((T.int64(512),), scope="shared")
                                greater_than_u = Ts.sblock_alloc_tensor((T.int64(4),), "bool", scope="local")
                                mask = Ts.sblock_alloc_tensor((T.int64(4),), "bool", scope="local")
                                valid = Ts.sblock_alloc_tensor((T.int64(4),), "bool", scope="local")
                                indices = Ts.sblock_alloc_tensor((T.int64(4),), "int64", scope="local")
                                step_aggregate = Ts.sblock_alloc_tensor((), scope="local")
                                for v in T.unroll(T.int64(4)):
                                    idx: T.let[T.int64] = T.Cast("int64", step_iter[()]) * T.int64(512) + ty * T.int64(128) + tx * T.int64(4) + v
                                    prob_local: T.let[T.float32] = T.if_then_else(idx < vocab_size, prob[row_idx, idx], T.Cast("float32", 0))
                                    prob_gt_threshold[v] = T.if_then_else(prob_local > T.float32(0), prob_local, T.Cast("float32", 0))
                                    valid[v] = prob_local > T.float32(0) and idx < vocab_size
                                with Ts.sblock(""):
                                    Ts.reads(prob_gt_threshold[T.int64(0):T.int64(4)])
                                    Ts.writes(step_aggregate[()])
                                    local_sum = Ts.sblock_alloc_tensor((), scope="local")
                                    shared_buf = Ts.sblock_alloc_tensor((T.int64(128),), scope="shared")
                                    idx: T.let[T.int64] = ty * T.int64(32) + tx
                                    local_sum[()] = T.Cast("float32", 0)
                                    for i in T.unroll(T.int64(4)):
                                        local_sum[()] = local_sum[()] + prob_gt_threshold[i]
                                    shared_buf[idx] = local_sum[()]
                                    for i in T.unroll(T.int64(7)):
                                        if idx % T.shift_left(T.int64(1), i + T.int64(1)) == T.int64(0):
                                            shared_buf[idx] = shared_buf[idx] + shared_buf[idx + T.shift_left(T.int64(1), i)]
                                    step_aggregate[()] = shared_buf[0]
                                if T.gpu_thread_invariant(aggregate[()] + step_aggregate[()] >= u - T.float32(9.9999999999999995e-07)):
                                    for i in T.unroll(T.int64(1), T.int64(4)):
                                        prob_gt_threshold[i] = prob_gt_threshold[i] + prob_gt_threshold[i - T.int64(1)]
                                    for i in T.vectorized(T.int64(4)):
                                        cumsum[ty * T.int64(128) + tx * T.int64(4) + i] = prob_gt_threshold[i]
                                    for i in T.unroll(T.int64(5)):
                                        for j in T.vectorized(T.int64(4)):
                                            idx: T.let[T.int64] = ty * T.int64(128) + tx * T.int64(4)
                                            if tx >= T.shift_left(T.int64(1), i):
                                                cumsum[idx + j] = cumsum[idx + j] + cumsum[idx - T.shift_left(T.int64(1), i) * T.int64(4) + T.int64(4) - T.int64(1)]
                                    for i in T.unroll(T.int64(1), T.int64(4)):
                                        for j in T.vectorized(T.int64(4)):
                                            if ty == T.int64(0):
                                                idx: T.let[T.int64] = i * T.int64(128) + tx * T.int64(4)
                                                cumsum[idx + j] = cumsum[idx + j] + cumsum[i * T.int64(128) - T.int64(1)]
                                    for v in T.unroll(T.int64(4)):
                                        greater_than_u[v] = cumsum[ty * T.int64(128) + tx * T.int64(4) + v] + aggregate[()] >= u - T.float32(9.9999999999999995e-07)
                                    with Ts.sblock(""):
                                        Ts.reads(greater_than_u[T.int64(0):T.int64(4)])
                                        Ts.writes(mask[T.int64(0):T.int64(4)])
                                        shared_buf = Ts.sblock_alloc_tensor((T.int64(128),), "bool", scope="shared")
                                        tx_idx: T.let[T.int64] = ty * T.int64(32) + tx
                                        shared_buf[tx_idx] = greater_than_u[T.int64(3)]
                                        mask[0] = T.if_then_else(tx_idx != T.int64(0), T.Cast("int8", greater_than_u[0]) != T.Cast("int8", shared_buf[tx_idx - T.int64(1)]), greater_than_u[0])
                                        for i in T.unroll(T.int64(1), T.int64(4)):
                                            mask[i] = T.Cast("int8", greater_than_u[i]) != T.Cast("int8", greater_than_u[i - T.int64(1)])
                                    for v in T.unroll(T.int64(4)):
                                        mask[v] = mask[v] and valid[v]
                                        indices[v] = T.Cast("int64", step_iter[()]) * T.int64(512) + ty * T.int64(128) + tx * T.int64(4) + v
                                    with Ts.sblock(""):
                                        Ts.reads(mask[T.int64(0):T.int64(4)], indices[T.int64(0):T.int64(4)])
                                        Ts.writes(sample_id_local[()])
                                        local_sum = Ts.sblock_alloc_tensor((), "int64", scope="local")
                                        shared_buf = Ts.sblock_alloc_tensor((T.int64(128),), "int64", scope="shared")
                                        idx: T.let[T.int64] = ty * T.int64(32) + tx
                                        local_sum[()] = T.Cast("int64", vocab_size - T.int64(1))
                                        for i in T.unroll(T.int64(4)):
                                            if mask[i]:
                                                local_sum[()] = T.min(local_sum[()], indices[i])
                                        shared_buf[idx] = local_sum[()]
                                        for i in T.unroll(T.int64(7)):
                                            if idx % T.shift_left(T.int64(1), i + T.int64(1)) == T.int64(0):
                                                shared_buf[idx] = T.min(shared_buf[idx], shared_buf[idx + T.shift_left(T.int64(1), i)])
                                        sample_id_local[()] = shared_buf[0]
                                aggregate[()] = aggregate[()] + step_aggregate[()]
                            step_iter[()] = step_iter[()] + 1
                        if tx == T.int64(0) and ty == T.int64(0):
                            token_ids[bx, 0] = sample_id_local[()]

        @R.function
        def foo(prob: R.Tensor((3, 5), dtype="float32"), uniform_sample: R.Tensor((6, 1), dtype="float32"), sample_indices: R.Tensor((6, 1), dtype="int64")) -> R.Tensor((6, 1), dtype="int64"):
            cls = Expected
            with R.dataflow():
                gv = R.call_tir(cls.parallel_sampling_from_prob, (prob, uniform_sample, sample_indices), ty_args=[R.Tensor((6, 1), dtype="int64")])
                R.output(gv)
            return gv
    # fmt: on

    mod = DispatchSampling(target=tvm_tirx.target.Target("cuda"))(MultiFromUniformModule)

    assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
