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
# ruff: noqa: E501, F841

import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import dlight as dl
from tvm.s_tir import script as Ts
from tvm_tirx.ir import IRModule, assert_structural_equal
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target


def _check(mod_before: IRModule, mod_after: IRModule):
    target = Target("nvidia/geforce-rtx-3090-ti")
    mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
        dl.gpu.GeneralReduction(), target=target
    )(mod_before)
    assert_structural_equal(mod, mod_after)


def _make_scalar_argmin(length):
    @I.ir_module
    class Before:
        @Ts.function
        def main(x: T.Tensor((T.int64(length),), "float32"), x_red: T.Tensor((), "int64")):
            T.func_attr({"tirx.noalias": True})
            x_red_temp_v0 = Ts.sblock_alloc_tensor((), "int64")
            x_red_temp_v1 = Ts.sblock_alloc_tensor(())
            for k in range(T.int64(length)):
                with Ts.sblock("x_red_temp"):
                    v_k = Ts.axis.reduce(T.int64(length), k)
                    Ts.reads(x[v_k])
                    Ts.writes(x_red_temp_v0[()], x_red_temp_v1[()])
                    with Ts.init():
                        x_red_temp_v0[()] = T.int64(-1)
                        x_red_temp_v1[()] = T.max_value(ty="float32")
                    value_is_smaller = x_red_temp_v1[()] < x[v_k]
                    value_is_equal = x_red_temp_v1[()] == x[v_k]
                    index_is_smaller = x_red_temp_v0[()] < v_k
                    new_index: T.int64 = T.Select(
                        value_is_smaller or (value_is_equal and index_is_smaller),
                        x_red_temp_v0[()],
                        v_k,
                    )
                    new_value: T.float32 = T.Select(
                        value_is_smaller,
                        x_red_temp_v1[()],
                        x[v_k],
                    )
                    x_red_temp_v0[()] = new_index
                    x_red_temp_v1[()] = new_value
            with Ts.sblock("x_red"):
                vi = Ts.axis.spatial(1, T.int64(0))
                Ts.reads(x_red_temp_v0[()])
                Ts.writes(x_red[()])
                x_red[()] = x_red_temp_v0[()]

    return Before


def test_scalar_argmin_reduction_value_scope():
    for length, expected_scope in ((1, "local"), (2, "global"), (3, "global")):
        target = Target("nvidia/geforce-rtx-3090-ti")
        mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
            dl.gpu.GeneralReduction(), target=target
        )(_make_scalar_argmin(length))

        index_temp, value_temp = mod["main"].body[0].block.alloc_tensors
        assert index_temp.scope() == tvm_tirx.runtime.MemoryScope("global")
        assert value_temp.scope() == tvm_tirx.runtime.MemoryScope(expected_scope)


def test_softmax_1():
    # fmt: off
    n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int64")

    @I.ir_module
    class Before:
        @Ts.function
        def main(lv44: T.Tensor((T.int64(1), T.int64(32), n, m), "float32"), var_compute_intermediate: T.Tensor((T.int64(1), T.int64(32), n, m), 'float16')):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            T_softmax_maxelem = Ts.sblock_alloc_tensor((T.int64(1), T.int64(32), n))
            T_softmax_exp = Ts.sblock_alloc_tensor((T.int64(1), T.int64(32), n, m))
            T_softmax_expsum = Ts.sblock_alloc_tensor((T.int64(1), T.int64(32), n))
            var_T_softmax_norm_intermediate = Ts.sblock_alloc_tensor((T.int64(1), T.int64(32), n, m))
            for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), n, m):
                with Ts.sblock("T_softmax_maxelem"):
                    v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                    Ts.reads(lv44[v_i0, v_i1, v_i2, v_k])
                    Ts.writes(T_softmax_maxelem[v_i0, v_i1, v_i2])
                    with Ts.init():
                        T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float32(-3.4028234663852886e+38)
                    T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(T_softmax_maxelem[v_i0, v_i1, v_i2], lv44[v_i0, v_i1, v_i2, v_k])
            for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), n, m):
                with Ts.sblock("T_softmax_exp"):
                    v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(lv44[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem[v_i0, v_i1, v_i2])
                    Ts.writes(T_softmax_exp[v_i0, v_i1, v_i2, v_i3])
                    T_softmax_exp[v_i0, v_i1, v_i2, v_i3] = T.exp(lv44[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2])
            for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), n, m):
                with Ts.sblock("T_softmax_expsum"):
                    v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                    Ts.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_k])
                    Ts.writes(T_softmax_expsum[v_i0, v_i1, v_i2])
                    with Ts.init():
                        T_softmax_expsum[v_i0, v_i1, v_i2] = T.float32(0)
                    T_softmax_expsum[v_i0, v_i1, v_i2] = T_softmax_expsum[v_i0, v_i1, v_i2] + T_softmax_exp[v_i0, v_i1, v_i2, v_k]
            for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), n, m):
                with Ts.sblock("T_softmax_norm"):
                    v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_i3], T_softmax_expsum[v_i0, v_i1, v_i2])
                    Ts.writes(var_T_softmax_norm_intermediate[v_i0, v_i1, v_i2, v_i3])
                    Ts.sblock_attr({"axis": 3})
                    var_T_softmax_norm_intermediate[v_i0, v_i1, v_i2, v_i3] = T_softmax_exp[v_i0, v_i1, v_i2, v_i3] / T_softmax_expsum[v_i0, v_i1, v_i2]
            for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), n, m):
                with Ts.sblock("compute"):
                    v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(var_T_softmax_norm_intermediate[v_i0, v_i1, v_i2, v_i3])
                    Ts.writes(var_compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                    var_compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.Cast("float16", var_T_softmax_norm_intermediate[v_i0, v_i1, v_i2, v_i3])

    n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int64")

    @I.ir_module
    class After:
        @Ts.function
        def main(lv44: T.Tensor((T.int64(1), T.int64(32), n, m), "float32"), var_compute_intermediate: T.Tensor((T.int64(1), T.int64(32), n, m), 'float16')):
            T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})

            # with Ts.sblock("root"):
            T_softmax_maxelem_shared = Ts.sblock_alloc_tensor((T.int64(1), T.int64(32), n), scope="shared")
            T_softmax_expsum_shared = Ts.sblock_alloc_tensor((T.int64(1), T.int64(32), n), scope="shared")
            for ax0_ax1_fused in T.parallel(n * T.int64(32), annotations={"thread_binding": "blockIdx.x"}):
                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                    for ax2_fused_1 in T.parallel(T.int64(256), annotations={"thread_binding": "threadIdx.x"}):
                        for ax2_fused_0 in T.serial((m + T.int64(255)) // T.int64(256), annotations={"tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                            with Ts.sblock("T_softmax_maxelem"):
                                v0 = Ts.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                                v1 = Ts.axis.spatial(n, ax0_ax1_fused % n + ax1)
                                v2 = Ts.axis.reduce(m, ax2_fused_0 * T.int64(256) + ax2_fused_1)
                                Ts.where(ax2_fused_0 * T.int64(256) + ax2_fused_1 < m)
                                Ts.reads(lv44[T.int64(0), v0, v1, v2])
                                Ts.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                                with Ts.init():
                                    T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv44[T.int64(0), v0, v1, v2])
                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                    for ax2_fused_1 in T.parallel(T.int64(256), annotations={"thread_binding": "threadIdx.x"}):
                        for ax2_fused_0 in T.serial((m + T.int64(255)) // T.int64(256), annotations={"tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                            with Ts.sblock("T_softmax_expsum"):
                                v0 = Ts.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                                v1 = Ts.axis.spatial(n, ax0_ax1_fused % n + ax1)
                                v2 = Ts.axis.reduce(m, ax2_fused_0 * T.int64(256) + ax2_fused_1)
                                Ts.where(ax2_fused_0 * T.int64(256) + ax2_fused_1 < m)
                                Ts.reads(lv44[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                                Ts.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                                with Ts.init():
                                    T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv44[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
                for ax2_1 in T.parallel(T.int64(256), annotations={"thread_binding": "threadIdx.x"}):
                    for ax2_0 in T.serial((m + T.int64(255)) // T.int64(256), annotations={"tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                        with Ts.sblock("compute"):
                            v0 = Ts.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                            v1 = Ts.axis.spatial(n, ax0_ax1_fused % n)
                            v2 = Ts.axis.spatial(m, ax2_0 * T.int64(256) + ax2_1)
                            Ts.where(ax2_0 * T.int64(256) + ax2_1 < m)
                            Ts.reads(lv44[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                            Ts.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                            var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv44[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
    # fmt: on
    _check(Before, After)


def test_softmax_2():
    # fmt: off
    @I.ir_module
    class Before:
        @Ts.function
        def main(A: T.Tensor((T.int64(1), T.int64(1), T.int64(32000)), "float32"), T_softmax_norm: T.Tensor((T.int64(1), T.int64(1), T.int64(32000)), "float32")):
            # with Ts.sblock("root"):
            T_softmax_maxelem = Ts.sblock_alloc_tensor((T.int64(1), T.int64(1)))
            T_softmax_exp = Ts.sblock_alloc_tensor((T.int64(1), T.int64(1), T.int64(32000)))
            T_softmax_expsum = Ts.sblock_alloc_tensor((T.int64(1), T.int64(1)))
            for i0, i1, k in T.grid(T.int64(1), T.int64(1), T.int64(32000)):
                with Ts.sblock("T_softmax_maxelem"):
                    v_i0, v_i1, v_k = Ts.axis.remap("SSR", [i0, i1, k])
                    Ts.reads(A[v_i0, v_i1, v_k])
                    Ts.writes(T_softmax_maxelem[v_i0, v_i1])
                    with Ts.init():
                        T_softmax_maxelem[v_i0, v_i1] = T.float32(-3.4028234663852886e+38)
                    T_softmax_maxelem[v_i0, v_i1] = T.max(T_softmax_maxelem[v_i0, v_i1], A[v_i0, v_i1, v_k])
            for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(32000)):
                with Ts.sblock("T_softmax_exp"):
                    v_i0, v_i1, v_i2 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(A[v_i0, v_i1, v_i2], T_softmax_maxelem[v_i0, v_i1])
                    Ts.writes(T_softmax_exp[v_i0, v_i1, v_i2])
                    T_softmax_exp[v_i0, v_i1, v_i2] = T.exp(A[v_i0, v_i1, v_i2] - T_softmax_maxelem[v_i0, v_i1])
            for i0, i1, k in T.grid(T.int64(1), T.int64(1), T.int64(32000)):
                with Ts.sblock("T_softmax_expsum"):
                    v_i0, v_i1, v_k = Ts.axis.remap("SSR", [i0, i1, k])
                    Ts.reads(T_softmax_exp[v_i0, v_i1, v_k])
                    Ts.writes(T_softmax_expsum[v_i0, v_i1])
                    with Ts.init():
                        T_softmax_expsum[v_i0, v_i1] = T.float32(0)
                    T_softmax_expsum[v_i0, v_i1] = T_softmax_expsum[v_i0, v_i1] + T_softmax_exp[v_i0, v_i1, v_k]
            for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(32000)):
                with Ts.sblock("T_softmax_norm"):
                    v_i0, v_i1, v_i2 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(T_softmax_exp[v_i0, v_i1, v_i2], T_softmax_expsum[v_i0, v_i1])
                    Ts.writes(T_softmax_norm[v_i0, v_i1, v_i2])
                    Ts.sblock_attr({"axis": 2})
                    T_softmax_norm[v_i0, v_i1, v_i2] = T_softmax_exp[v_i0, v_i1, v_i2] / T_softmax_expsum[v_i0, v_i1]

    @I.ir_module
    class After:
        @Ts.function
        def main(A: T.Tensor((T.int64(1), T.int64(1), T.int64(32000)), "float32"), T_softmax_norm: T.Tensor((T.int64(1), T.int64(1), T.int64(32000)), "float32")):
            T.func_attr({"tirx.is_scheduled": True})
            # with Ts.sblock("root"):
            T_softmax_maxelem_shared = Ts.sblock_alloc_tensor((T.int64(1), T.int64(1)), scope="shared")
            T_softmax_expsum_shared = Ts.sblock_alloc_tensor((T.int64(1), T.int64(1)), scope="shared")
            for ax0_fused in T.parallel(T.int64(1), annotations={"thread_binding": "blockIdx.x"}):
                for ax0 in T.serial(T.int64(0), T.int64(1)):
                    for ax1_fused_1 in T.parallel(T.int64(256), annotations={"thread_binding": "threadIdx.x"}):
                        for ax1_fused_0 in T.serial(T.int64(125), annotations={"tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                            with Ts.sblock("T_softmax_maxelem"):
                                v0 = Ts.axis.spatial(T.int64(1), ax0)
                                v1 = Ts.axis.reduce(T.int64(32000), ax1_fused_0 * T.int64(256) + ax1_fused_1)
                                Ts.reads(A[T.int64(0), T.int64(0), v1])
                                Ts.writes(T_softmax_maxelem_shared[T.int64(0), T.int64(0)])
                                with Ts.init():
                                    T_softmax_maxelem_shared[T.int64(0), T.int64(0)] = T.float32(-3.4028234663852886e+38)
                                T_softmax_maxelem_shared[T.int64(0), T.int64(0)] = T.max(T_softmax_maxelem_shared[T.int64(0), T.int64(0)], A[T.int64(0), T.int64(0), v1])
                for ax0 in T.serial(T.int64(0), T.int64(1)):
                    for ax1_fused_1 in T.parallel(T.int64(256), annotations={"thread_binding": "threadIdx.x"}):
                        for ax1_fused_0 in T.serial(T.int64(125), annotations={"tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                            with Ts.sblock("T_softmax_expsum"):
                                v0 = Ts.axis.spatial(T.int64(1), ax0)
                                v1 = Ts.axis.reduce(T.int64(32000), ax1_fused_0 * T.int64(256) + ax1_fused_1)
                                Ts.reads(A[T.int64(0), T.int64(0), v1], T_softmax_maxelem_shared[T.int64(0), T.int64(0)])
                                Ts.writes(T_softmax_expsum_shared[T.int64(0), T.int64(0)])
                                with Ts.init():
                                    T_softmax_expsum_shared[T.int64(0), T.int64(0)] = T.float32(0)
                                T_softmax_expsum_shared[T.int64(0), T.int64(0)] = T_softmax_expsum_shared[T.int64(0), T.int64(0)] + T.exp(A[T.int64(0), T.int64(0), v1] - T_softmax_maxelem_shared[T.int64(0), T.int64(0)])
                for ax1_1 in T.parallel(T.int64(256), annotations={"thread_binding": "threadIdx.x"}):
                    for ax1_0 in T.serial(T.int64(125), annotations={"tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                        with Ts.sblock("T_softmax_norm"):
                            v0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                            v1 = Ts.axis.spatial(T.int64(32000), ax1_0 * T.int64(256) + ax1_1)
                            Ts.reads(A[T.int64(0), T.int64(0), v1], T_softmax_maxelem_shared[T.int64(0), T.int64(0)], T_softmax_expsum_shared[T.int64(0), T.int64(0)])
                            Ts.writes(T_softmax_norm[T.int64(0), T.int64(0), v1])
                            Ts.sblock_attr({"axis": 2})
                            T_softmax_norm[T.int64(0), T.int64(0), v1] = T.exp(A[T.int64(0), T.int64(0), v1] - T_softmax_maxelem_shared[T.int64(0), T.int64(0)]) / T_softmax_expsum_shared[T.int64(0), T.int64(0)]

    # fmt: on
    _check(Before, After)


def test_softmax_3():
    # fmt: off
    @I.ir_module
    class Before:
        @Ts.function
        def main(input: T.Tensor((T.int64(1), T.int64(4), T.int64(32), T.int64(8192)), "float32"), T_softmax_norm: T.Tensor((T.int64(1), T.int64(4), T.int64(32), T.int64(8192)), "float32")):
            # with Ts.sblock("root"):
            T_softmax_maxelem = Ts.sblock_alloc_tensor((T.int64(1), T.int64(4), T.int64(8192)))
            T_softmax_exp = Ts.sblock_alloc_tensor((T.int64(1), T.int64(4), T.int64(32), T.int64(8192)))
            T_softmax_expsum = Ts.sblock_alloc_tensor((T.int64(1), T.int64(4), T.int64(8192)))
            for i0, i1, i2, k in T.grid(T.int64(1), T.int64(4), T.int64(8192), T.int64(32)):
                with Ts.sblock("T_softmax_maxelem"):
                    v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                    Ts.reads(input[v_i0, v_i1, v_k, v_i2])
                    Ts.writes(T_softmax_maxelem[v_i0, v_i1, v_i2])
                    with Ts.init():
                        T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float32(-340282346638528859811704183484516925440.0)
                    T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(T_softmax_maxelem[v_i0, v_i1, v_i2], input[v_i0, v_i1, v_k, v_i2])
            for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(4), T.int64(32), T.int64(8192)):
                with Ts.sblock("T_softmax_exp"):
                    v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(input[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem[v_i0, v_i1, v_i3])
                    Ts.writes(T_softmax_exp[v_i0, v_i1, v_i2, v_i3])
                    T_softmax_exp[v_i0, v_i1, v_i2, v_i3] = T.exp(input[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i3])
            for i0, i1, i2, k in T.grid(T.int64(1), T.int64(4), T.int64(8192), T.int64(32)):
                with Ts.sblock("T_softmax_expsum"):
                    v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                    Ts.reads(T_softmax_exp[v_i0, v_i1, v_k, v_i2])
                    Ts.writes(T_softmax_expsum[v_i0, v_i1, v_i2])
                    with Ts.init():
                        T_softmax_expsum[v_i0, v_i1, v_i2] = T.float32(0.0)
                    T_softmax_expsum[v_i0, v_i1, v_i2] = T_softmax_expsum[v_i0, v_i1, v_i2] + T_softmax_exp[v_i0, v_i1, v_k, v_i2]
            for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(4), T.int64(32), T.int64(8192)):
                with Ts.sblock("T_softmax_norm"):
                    v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_i3], T_softmax_expsum[v_i0, v_i1, v_i3])
                    Ts.writes(T_softmax_norm[v_i0, v_i1, v_i2, v_i3])
                    Ts.sblock_attr({"axis": 2})
                    T_softmax_norm[v_i0, v_i1, v_i2, v_i3] = T_softmax_exp[v_i0, v_i1, v_i2, v_i3] / T_softmax_expsum[v_i0, v_i1, v_i3]

    @I.ir_module
    class After:
        @Ts.function
        def main(input: T.Tensor((T.int64(1), T.int64(4), T.int64(32), T.int64(8192)), "float32"), T_softmax_norm: T.Tensor((T.int64(1), T.int64(4), T.int64(32), T.int64(8192)), "float32")):
            T.func_attr({"tirx.is_scheduled": True})
            # with Ts.sblock("root"):
            T_softmax_maxelem_shared = Ts.sblock_alloc_tensor((T.int64(1), T.int64(4), T.int64(8192)), scope="shared")
            T_softmax_expsum_shared = Ts.sblock_alloc_tensor((T.int64(1), T.int64(4), T.int64(8192)), scope="shared")
            for ax0_ax2_fused in T.parallel(T.int64(32768), annotations={"thread_binding": "blockIdx.x"}):
                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                    for ax2_fused_1 in T.parallel(T.int64(256), annotations={"thread_binding": "threadIdx.x"}):
                        for ax2_fused_0 in T.serial(T.int64(1), annotations={"tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                            with Ts.sblock("T_softmax_maxelem"):
                                v0 = Ts.axis.spatial(T.int64(4), ax0_ax2_fused // T.int64(8192) + ax0)
                                v1 = Ts.axis.spatial(T.int64(8192), ax0_ax2_fused % T.int64(8192) + ax1)
                                v2 = Ts.axis.reduce(T.int64(32), ax2_fused_0 * T.int64(256) + ax2_fused_1)
                                Ts.where(ax2_fused_0 * T.int64(256) + ax2_fused_1 < T.int64(32))
                                Ts.reads(input[T.int64(0), v0, v2, v1])
                                Ts.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                                with Ts.init():
                                    T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-340282346638528859811704183484516925440.0)
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], input[T.int64(0), v0, v2, v1])
                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                    for ax2_fused_1 in T.parallel(T.int64(256), annotations={"thread_binding": "threadIdx.x"}):
                        for ax2_fused_0 in T.serial(T.int64(1), annotations={"tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                            with Ts.sblock("T_softmax_expsum"):
                                v0 = Ts.axis.spatial(T.int64(4), ax0_ax2_fused // T.int64(8192) + ax0)
                                v1 = Ts.axis.spatial(T.int64(8192), ax0_ax2_fused % T.int64(8192) + ax1)
                                v2 = Ts.axis.reduce(T.int64(32), ax2_fused_0 * T.int64(256) + ax2_fused_1)
                                Ts.where(ax2_fused_0 * T.int64(256) + ax2_fused_1 < T.int64(32))
                                Ts.reads(input[T.int64(0), v0, v2, v1], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                                Ts.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                                with Ts.init():
                                    T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0.0)
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(input[T.int64(0), v0, v2, v1] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
                for ax1_1 in T.parallel(T.int64(256), annotations={"thread_binding": "threadIdx.x"}):
                    for ax1_0 in T.serial(T.int64(1), annotations={"tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                        with Ts.sblock("T_softmax_norm"):
                            v0 = Ts.axis.spatial(T.int64(4), ax0_ax2_fused // T.int64(8192))
                            v1 = Ts.axis.spatial(T.int64(32), ax1_0 * T.int64(256) + ax1_1)
                            v2 = Ts.axis.spatial(T.int64(8192), ax0_ax2_fused % T.int64(8192))
                            Ts.where(ax1_0 * T.int64(256) + ax1_1 < T.int64(32))
                            Ts.reads(input[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v2], T_softmax_expsum_shared[T.int64(0), v0, v2])
                            Ts.writes(T_softmax_norm[T.int64(0), v0, v1, v2])
                            Ts.sblock_attr({"axis": 2})
                            T_softmax_norm[T.int64(0), v0, v1, v2] = T.exp(input[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v2]) / T_softmax_expsum_shared[T.int64(0), v0, v2]
    # fmt: on
    _check(Before, After)


def test_layer_norm():
    # fmt: off
    n = T.dynamic("n", "int64")

    @I.ir_module
    class Before:
        @Ts.function
        def main(lv6: T.Tensor((T.int64(1), n, T.int64(2560)), "float32"), weight1: T.Tensor((T.int64(2560),), "float32"), bias: T.Tensor((T.int64(2560),), "float32"), var_compute_intermediate: T.Tensor((T.int64(1), n, T.int64(2560)), 'float16')):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            A_red_temp_v0 = Ts.sblock_alloc_tensor((T.int64(1), n))
            A_red_temp_v1 = Ts.sblock_alloc_tensor((T.int64(1), n))
            var_T_layer_norm_intermediate = Ts.sblock_alloc_tensor((T.int64(1), n, T.int64(2560)))
            for ax0, ax1, k2 in T.grid(T.int64(1), n, T.int64(2560)):
                with Ts.sblock("A_red_temp"):
                    v_ax0, v_ax1, v_k2 = Ts.axis.remap("SSR", [ax0, ax1, k2])
                    Ts.reads(lv6[v_ax0, v_ax1, v_k2])
                    Ts.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
                    with Ts.init():
                        A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                        A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
                    v_A_red_temp_v0: T.let[T.float32] = A_red_temp_v0[v_ax0, v_ax1] + lv6[v_ax0, v_ax1, v_k2]
                    v_A_red_temp_v1: T.let[T.float32] = A_red_temp_v1[v_ax0, v_ax1] + lv6[v_ax0, v_ax1, v_k2] * lv6[v_ax0, v_ax1, v_k2]
                    A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
                    A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
            for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
                with Ts.sblock("T_layer_norm"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(lv6[v_ax0, v_ax1, v_ax2], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], weight1[v_ax2], bias[v_ax2])
                    Ts.writes(var_T_layer_norm_intermediate[v_ax0, v_ax1, v_ax2])
                    var_T_layer_norm_intermediate[v_ax0, v_ax1, v_ax2] = (lv6[v_ax0, v_ax1, v_ax2] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(0.00039062500000000002) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00039062500000000002) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * weight1[v_ax2] + bias[v_ax2]
            for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(2560)):
                with Ts.sblock("compute"):
                    v_i0, v_i1, v_i2 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(var_T_layer_norm_intermediate[v_i0, v_i1, v_i2])
                    Ts.writes(var_compute_intermediate[v_i0, v_i1, v_i2])
                    var_compute_intermediate[v_i0, v_i1, v_i2] = T.Cast("float16", var_T_layer_norm_intermediate[v_i0, v_i1, v_i2])

    n = T.dynamic("n", "int64")

    @I.ir_module
    class After:
        @Ts.function
        def main(lv6: T.Tensor((T.int64(1), n, T.int64(2560)), "float32"), weight1: T.Tensor((T.int64(2560),), "float32"), bias: T.Tensor((T.int64(2560),), "float32"), var_compute_intermediate: T.Tensor((T.int64(1), n, T.int64(2560)), 'float16')):
            T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})

            # with Ts.sblock("root"):
            A_red_temp_v0_shared = Ts.sblock_alloc_tensor((T.int64(1), n), scope="shared")
            A_red_temp_v1_shared = Ts.sblock_alloc_tensor((T.int64(1), n), scope="shared")
            for ax0_fused in T.parallel(n, annotations={"thread_binding": "blockIdx.x"}):
                for ax0 in T.serial(T.int64(0), T.int64(1)):
                    for ax1_fused_1 in T.parallel(T.int64(256), annotations={"thread_binding": "threadIdx.x"}):
                        for ax1_fused_0 in T.serial(T.int64(10), annotations={"tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                            with Ts.sblock("A_red_temp"):
                                v0 = Ts.axis.spatial(n, ax0_fused + ax0)
                                v1 = Ts.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(256) + ax1_fused_1)
                                Ts.reads(lv6[T.int64(0), v0, v1])
                                Ts.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                                with Ts.init():
                                    A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                    A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                                v_A_red_temp_v0: T.let[T.float32] = A_red_temp_v0_shared[T.int64(0), v0] + lv6[T.int64(0), v0, v1]
                                v_A_red_temp_v1: T.let[T.float32] = A_red_temp_v1_shared[T.int64(0), v0] + lv6[T.int64(0), v0, v1] * lv6[T.int64(0), v0, v1]
                                A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                                A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
                for ax1_1 in T.parallel(T.int64(256), annotations={"thread_binding": "threadIdx.x"}):
                    for ax1_0 in T.serial(T.int64(10), annotations={"tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                        with Ts.sblock("compute"):
                            v0 = Ts.axis.spatial(n, ax0_fused)
                            v1 = Ts.axis.spatial(T.int64(2560), ax1_0 * T.int64(256) + ax1_1)
                            Ts.reads(lv6[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], weight1[v1], bias[v1])
                            Ts.writes(var_compute_intermediate[T.int64(0), v0, v1])
                            var_compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * weight1[v1] + bias[v1])
    # fmt: on
    _check(Before, After)


def test_rms_norm():
    # fmt: off
    n = T.dynamic("n", "int64")

    @I.ir_module
    class Before:
        @Ts.function
        def main(A: T.Tensor((T.int64(1), n, T.int64(4096)), 'float16'), B: T.Tensor((T.int64(4096),), "float16"), rms_norm_1: T.Tensor((T.int64(1), n, T.int64(4096)), 'float16')):
            T.func_attr({"op_pattern": 4, "tirx.noalias": True})

            # with Ts.sblock("root"):
            Ared_temp = Ts.sblock_alloc_tensor((T.int64(1), n))
            for bsz, i, k in T.grid(T.int64(1), n, T.int64(4096)):
                with Ts.sblock("Ared_temp"):
                    v_bsz, v_i, v_k = Ts.axis.remap("SSR", [bsz, i, k])
                    Ts.reads(A[v_bsz, v_i, v_k])
                    Ts.writes(Ared_temp[v_bsz, v_i])
                    with Ts.init():
                        Ared_temp[v_bsz, v_i] = T.float32(0)
                    Ared_temp[v_bsz, v_i] = Ared_temp[v_bsz, v_i] + T.Cast("float32", A[v_bsz, v_i, v_k]) * T.Cast("float32", A[v_bsz, v_i, v_k])
            for bsz, i, k in T.grid(T.int64(1), n, T.int64(4096)):
                with Ts.sblock("rms_norm"):
                    v_bsz, v_i, v_k = Ts.axis.remap("SSS", [bsz, i, k])
                    Ts.reads(B[v_k], A[v_bsz, v_i, v_k], Ared_temp[v_bsz, v_i])
                    Ts.writes(rms_norm_1[v_bsz, v_i, v_k])
                    rms_norm_1[v_bsz, v_i, v_k] = T.Cast("float16", T.Cast("float32", B[v_k]) * (T.Cast("float32", A[v_bsz, v_i, v_k]) / T.sqrt(Ared_temp[v_bsz, v_i] * T.float32(0.000244140625) + T.float32(9.9999999999999995e-07))))

    n = T.dynamic("n", "int64")

    @I.ir_module
    class After:
        @Ts.function
        def main(A: T.Tensor((T.int64(1), n, T.int64(4096)), 'float16'), B: T.Tensor((T.int64(4096),), "float16"), rms_norm_1: T.Tensor((T.int64(1), n, T.int64(4096)), 'float16')):
            T.func_attr({"op_pattern": 4, "tirx.is_scheduled": True, "tirx.noalias": True})

            # with Ts.sblock("root"):
            Ared_temp_shared = Ts.sblock_alloc_tensor((T.int64(1), n), scope="shared")
            for ax0_fused in T.parallel(n, annotations={"thread_binding": "blockIdx.x"}):
                for ax0 in T.serial(T.int64(0), T.int64(1)):
                    for ax1_fused_1 in T.parallel(T.int64(256), annotations={"thread_binding": "threadIdx.x"}):
                        for ax1_fused_0 in T.serial(T.int64(16), annotations={"tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                            with Ts.sblock("Ared_temp"):
                                v0 = Ts.axis.spatial(n, ax0_fused + ax0)
                                v1 = Ts.axis.reduce(T.int64(4096), ax1_fused_0 * T.int64(256) + ax1_fused_1)
                                Ts.reads(A[T.int64(0), v0, v1])
                                Ts.writes(Ared_temp_shared[T.int64(0), v0])
                                with Ts.init():
                                    Ared_temp_shared[T.int64(0), v0] = T.float32(0)
                                Ared_temp_shared[T.int64(0), v0] = Ared_temp_shared[T.int64(0), v0] + T.Cast("float32", A[T.int64(0), v0, v1]) * T.Cast("float32", A[T.int64(0), v0, v1])
                for ax1_1 in T.parallel(T.int64(256), annotations={"thread_binding": "threadIdx.x"}):
                    for ax1_0 in T.serial(T.int64(16), annotations={"tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                        with Ts.sblock("rms_norm"):
                            v0 = Ts.axis.spatial(n, ax0_fused)
                            v1 = Ts.axis.spatial(T.int64(4096), ax1_0 * T.int64(256) + ax1_1)
                            Ts.reads(B[v1], A[T.int64(0), v0, v1], Ared_temp_shared[T.int64(0), v0])
                            Ts.writes(rms_norm_1[T.int64(0), v0, v1])
                            rms_norm_1[T.int64(0), v0, v1] = T.Cast("float16", T.Cast("float32", B[v1]) * (T.Cast("float32", A[T.int64(0), v0, v1]) / T.sqrt(Ared_temp_shared[T.int64(0), v0] * T.float32(0.000244140625) + T.float32(9.9999999999999995e-07))))
    # fmt: on
    _check(Before, After)


def test_group_norm():
    # fmt: off
    @I.ir_module
    class Before:
        @Ts.function
        def main(A: T.Tensor((1, 2048), "float32"), B: T.Tensor((2048,), "float32"), C: T.Tensor((2048,), "float32"), T_reshape: T.Tensor((1, 2048), "float32")):
            T.func_attr({"tirx.noalias": True})
            T_reshape_1 = Ts.sblock_alloc_tensor((1, 32, 64))
            A_red_temp_v0 = Ts.sblock_alloc_tensor((1, 32))
            A_red_temp_v1 = Ts.sblock_alloc_tensor((1, 32))
            T_reshape_2 = Ts.sblock_alloc_tensor((32, 64))
            T_reshape_3 = Ts.sblock_alloc_tensor((32, 64))
            T_group_norm = Ts.sblock_alloc_tensor((1, 32, 64))
            for ax0, ax1, ax2 in T.grid(1, 32, 64):
                with Ts.sblock("T_reshape"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(A[0, (v_ax1 * 64 + v_ax2) % 2048])
                    Ts.writes(T_reshape_1[v_ax0, v_ax1, v_ax2])
                    T_reshape_1[v_ax0, v_ax1, v_ax2] = A[0, (v_ax1 * 64 + v_ax2) % 2048]
            for ax0, ax1, k2 in T.grid(1, 32, 64):
                with Ts.sblock("A_red_temp"):
                    v_ax0, v_ax1, v_k2 = Ts.axis.remap("SSR", [ax0, ax1, k2])
                    Ts.reads(T_reshape_1[v_ax0, v_ax1, v_k2])
                    Ts.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
                    with Ts.init():
                        A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                        A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
                    v_A_red_temp_v0: T.let[T.float32] = A_red_temp_v0[v_ax0, v_ax1] + T_reshape_1[v_ax0, v_ax1, v_k2]
                    v_A_red_temp_v1: T.let[T.float32] = A_red_temp_v1[v_ax0, v_ax1] + T_reshape_1[v_ax0, v_ax1, v_k2] * T_reshape_1[v_ax0, v_ax1, v_k2]
                    A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
                    A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
            for ax0, ax1 in T.grid(32, 64):
                with Ts.sblock("T_reshape_1"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(B[(v_ax0 * 64 + v_ax1) % 2048])
                    Ts.writes(T_reshape_2[v_ax0, v_ax1])
                    T_reshape_2[v_ax0, v_ax1] = B[(v_ax0 * 64 + v_ax1) % 2048]
            for ax0, ax1 in T.grid(32, 64):
                with Ts.sblock("T_reshape_2"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(C[(v_ax0 * 64 + v_ax1) % 2048])
                    Ts.writes(T_reshape_3[v_ax0, v_ax1])
                    T_reshape_3[v_ax0, v_ax1] = C[(v_ax0 * 64 + v_ax1) % 2048]
            for ax0, ax1, ax2 in T.grid(1, 32, 64):
                with Ts.sblock("T_group_norm"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(T_reshape_1[v_ax0, v_ax1, v_ax2], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], T_reshape_2[v_ax1, v_ax2], T_reshape_3[v_ax1, v_ax2])
                    Ts.writes(T_group_norm[v_ax0, v_ax1, v_ax2])
                    T_group_norm[v_ax0, v_ax1, v_ax2] = (T_reshape_1[v_ax0, v_ax1, v_ax2] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.015625)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(0.015625) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.015625) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.015625)) + T.float32(1.0000000000000001e-05)) * T_reshape_2[v_ax1, v_ax2] + T_reshape_3[v_ax1, v_ax2]
            for ax0, ax1 in T.grid(1, 2048):
                with Ts.sblock("T_reshape_3"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(T_group_norm[0, v_ax1 % 2048 // 64, v_ax1 % 64])
                    Ts.writes(T_reshape[v_ax0, v_ax1])
                    T_reshape[v_ax0, v_ax1] = T_group_norm[0, v_ax1 % 2048 // 64, v_ax1 % 64]

    @I.ir_module
    class After:
        @Ts.function
        def main(A: T.Tensor((1, 2048), "float32"), B: T.Tensor((2048,), "float32"), C: T.Tensor((2048,), "float32"), T_reshape: T.Tensor((1, 2048), "float32")):
            T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
            # with Ts.sblock("root"):
            A_red_temp_v0_shared = Ts.sblock_alloc_tensor((1, 32), scope="shared")
            A_red_temp_v1_shared = Ts.sblock_alloc_tensor((1, 32), scope="shared")
            for ax0_fused in T.parallel(T.int64(1), annotations={"thread_binding": "blockIdx.x"}):
                for ax0 in range(32):
                    for ax1_fused_1 in T.parallel(256, annotations={"thread_binding": "threadIdx.x"}):
                        for ax1_fused_0 in T.serial(1, annotations={"tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                            with Ts.sblock("A_red_temp"):
                                v0 = Ts.axis.spatial(32, ax0)
                                v1 = Ts.axis.reduce(64, ax1_fused_0 * 256 + ax1_fused_1)
                                Ts.where(ax1_fused_0 * 256 + ax1_fused_1 < 64)
                                Ts.reads(A[0, v0 * 64 + v1])
                                Ts.writes(A_red_temp_v0_shared[0, v0], A_red_temp_v1_shared[0, v0])
                                with Ts.init():
                                    A_red_temp_v0_shared[0, v0] = T.float32(0)
                                    A_red_temp_v1_shared[0, v0] = T.float32(0)
                                v_A_red_temp_v0: T.let[T.float32] = A_red_temp_v0_shared[0, v0] + A[0, v0 * 64 + v1]
                                v_A_red_temp_v1: T.let[T.float32] = A_red_temp_v1_shared[0, v0] + A[0, v0 * 64 + v1] * A[0, v0 * 64 + v1]
                                A_red_temp_v0_shared[0, v0] = v_A_red_temp_v0
                                A_red_temp_v1_shared[0, v0] = v_A_red_temp_v1
                for ax1_1 in T.parallel(256, annotations={"thread_binding": "threadIdx.x"}):
                    for ax1_0 in T.serial(8, annotations={"tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                        with Ts.sblock("T_reshape_3"):
                            v0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                            v1 = Ts.axis.spatial(2048, ax1_0 * 256 + ax1_1)
                            Ts.reads(A[0, v1], A_red_temp_v0_shared[0, v1 // 64], A_red_temp_v1_shared[0, v1 // 64], B[v1], C[v1])
                            Ts.writes(T_reshape[0, v1])
                            T_reshape[0, v1] = (A[0, v1] - A_red_temp_v0_shared[0, v1 // 64] * T.float32(0.015625)) * T.rsqrt(A_red_temp_v1_shared[0, v1 // 64] * T.float32(0.015625) - A_red_temp_v0_shared[0, v1 // 64] * T.float32(0.015625) * (A_red_temp_v0_shared[0, v1 // 64] * T.float32(0.015625)) + T.float32(1.0000000000000001e-05)) * B[v1] + C[v1]
    _check(Before, After)


def test_logsumexp():
    batch_size = T.dynamic("batch_size", "int64")
    vocab_size = T.dynamic("vocab_size", "int64")
    num_chunks = T.dynamic("num_chunks", "int64")

    @I.ir_module
    class Before:
        @Ts.function
        def compute_lse(
            A: T.Tensor((batch_size, vocab_size), dtype="float32"),
            blocked_lse: T.Tensor((batch_size, num_chunks), dtype="float32"),
        ):
            T.func_attr({"tirx.noalias": True})

            A_pad = Ts.sblock_alloc_tensor((batch_size, num_chunks, T.int64(4096)), dtype="float32")
            temp_max = Ts.sblock_alloc_tensor((batch_size, num_chunks), dtype="float32")
            temp_sum = Ts.sblock_alloc_tensor((batch_size, num_chunks), dtype="float32")

            for l0, l1, l2 in T.grid(batch_size, num_chunks, T.int64(4096)):
                with Ts.sblock("pad"):
                    v0, v1, v2 = Ts.axis.remap("SSS", [l0, l1, l2])
                    A_pad[v0, v1, v2] = T.if_then_else(
                        v1 * T.int64(4096) + v2 < vocab_size,
                        A[v0, v1 * T.int64(4096) + v2],
                        T.min_value(ty="float32"),
                    )

            for l0, l1, l2 in T.grid(batch_size, num_chunks, T.int64(4096)):
                with Ts.sblock("max"):
                    v0, v1, v2 = Ts.axis.remap("SSR", [l0, l1, l2])
                    with Ts.init():
                        temp_max[v0, v1] = T.min_value(ty="float32")
                    temp_max[v0, v1] = T.max(temp_max[v0, v1], A_pad[v0, v1, v2])

            for l0, l1, l2 in T.grid(batch_size, num_chunks, T.int64(4096)):
                with Ts.sblock("sum_exp"):
                    v0, v1, v2 = Ts.axis.remap("SSR", [l0, l1, l2])
                    with Ts.init():
                        temp_sum[v0, v1] = T.float32(0)
                    temp_sum[v0, v1] += T.if_then_else(
                        v1 * T.int64(4096) + v2 < vocab_size,
                        T.exp(A_pad[v0, v1, v2] - temp_max[v0, v1]),
                        T.float32(0),
                    )

            for l0, l1, l2 in T.grid(batch_size, num_chunks, T.int64(1)):
                with Ts.sblock("log"):
                    v0, v1, v2 = Ts.axis.remap("SSS", [l0, l1, l2])
                    blocked_lse[v0, v1] = T.log(temp_sum[v0, v1]) + temp_max[v0, v1]

    batch_size = T.dynamic("batch_size", "int64")
    vocab_size = T.dynamic("vocab_size", "int64")
    num_chunks = T.dynamic("num_chunks", "int64")

    @I.ir_module
    class After:
        @Ts.function
        def compute_lse(
            A: T.Tensor((batch_size, vocab_size), "float32"),
            blocked_lse: T.Tensor((batch_size, num_chunks), "float32"),
        ):
            T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})

            temp_max_shared = Ts.sblock_alloc_tensor((batch_size, num_chunks), scope="shared")
            temp_sum_shared = Ts.sblock_alloc_tensor((batch_size, num_chunks), scope="shared")
            for ax0_ax1_fused in T.parallel(
                batch_size * num_chunks, annotations={"thread_binding": "blockIdx.x"}
            ):
                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                    for ax2_fused_1 in T.parallel(
                        T.int64(256), annotations={"thread_binding": "threadIdx.x"}
                    ):
                        for ax2_fused_0 in T.serial(
                            T.int64(16),
                            annotations={
                                "tirx.auto_unroll_max_step": 256,
                                "tirx.unroll_explicit": 1,
                            },
                        ):
                            with Ts.sblock("max"):
                                v0 = Ts.axis.spatial(
                                    batch_size,
                                    ax0_ax1_fused // num_chunks + ax0,
                                )
                                v1 = Ts.axis.spatial(num_chunks, ax0_ax1_fused % num_chunks + ax1)
                                v2 = Ts.axis.reduce(
                                    T.int64(4096), ax2_fused_0 * T.int64(256) + ax2_fused_1
                                )
                                Ts.reads(A[v0, v1 * T.int64(4096) + v2])
                                Ts.writes(temp_max_shared[v0, v1])
                                with Ts.init():
                                    temp_max_shared[v0, v1] = T.min_value(ty="float32")
                                temp_max_shared[v0, v1] = T.max(
                                    temp_max_shared[v0, v1],
                                    T.if_then_else(
                                        v1 * T.int64(4096) + v2 < vocab_size,
                                        A[v0, v1 * T.int64(4096) + v2],
                                        T.min_value(ty="float32"),
                                    ),
                                )
                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                    for ax2_fused_1 in T.parallel(
                        T.int64(256), annotations={"thread_binding": "threadIdx.x"}
                    ):
                        for ax2_fused_0 in T.serial(
                            T.int64(16),
                            annotations={
                                "tirx.auto_unroll_max_step": 256,
                                "tirx.unroll_explicit": 1,
                            },
                        ):
                            with Ts.sblock("sum_exp"):
                                v0 = Ts.axis.spatial(
                                    batch_size,
                                    ax0_ax1_fused // num_chunks + ax0,
                                )
                                v1 = Ts.axis.spatial(num_chunks, ax0_ax1_fused % num_chunks + ax1)
                                v2 = Ts.axis.reduce(
                                    T.int64(4096), ax2_fused_0 * T.int64(256) + ax2_fused_1
                                )
                                Ts.reads(A[v0, v1 * T.int64(4096) + v2], temp_max_shared[v0, v1])
                                Ts.writes(temp_sum_shared[v0, v1])
                                with Ts.init():
                                    temp_sum_shared[v0, v1] = T.float32(0)
                                temp_sum_shared[v0, v1] = temp_sum_shared[v0, v1] + T.if_then_else(
                                    v1 * T.int64(4096) + v2 < vocab_size,
                                    T.exp(
                                        T.if_then_else(
                                            v1 * T.int64(4096) + v2 < vocab_size,
                                            A[v0, v1 * T.int64(4096) + v2],
                                            T.min_value(ty="float32"),
                                        )
                                        - temp_max_shared[v0, v1]
                                    ),
                                    T.float32(0),
                                )
                for ax2_1 in T.parallel(
                    T.int64(256), annotations={"thread_binding": "threadIdx.x"}
                ):
                    for ax2_0 in T.serial(
                        T.int64(1),
                        annotations={
                            "tirx.auto_unroll_max_step": 256,
                            "tirx.unroll_explicit": 1,
                        },
                    ):
                        with Ts.sblock("log"):
                            v0 = Ts.axis.spatial(batch_size, ax0_ax1_fused // num_chunks)
                            v1 = Ts.axis.spatial(num_chunks, ax0_ax1_fused % num_chunks)
                            v2 = Ts.axis.spatial(T.int64(1), ax2_0 * T.int64(256) + ax2_1)
                            Ts.where(ax2_0 * T.int64(256) + ax2_1 < T.int64(1))
                            Ts.reads(temp_sum_shared[v0, v1], temp_max_shared[v0, v1])
                            Ts.writes(blocked_lse[v0, v1])
                            blocked_lse[v0, v1] = (
                                T.log(temp_sum_shared[v0, v1]) + temp_max_shared[v0, v1]
                            )

    _check(Before, After)


if __name__ == "__main__":
    tvm_tirx.testing.main()
