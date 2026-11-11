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
        dl.gpu.RMSNorm(), target=target
    )(mod_before)
    assert_structural_equal(mod, mod_after)


def test_rms_norm_with_casting():
    # fmt: off
    n = T.dynamic("n", "int32")

    @I.ir_module
    class Before:
        @Ts.function
        def main(data: T.Tensor((1, n, 4096), 'float16'), weight: T.Tensor((4096,), "float16"), T_cast: T.Tensor((1, n, 4096), 'float16')):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            T_cast_1 = Ts.sblock_alloc_tensor((1, n, 4096))
            T_multiply = Ts.sblock_alloc_tensor((1, n, 4096))
            T_multiply_red = Ts.sblock_alloc_tensor((1, n))
            rsqrt = Ts.sblock_alloc_tensor((1, n))
            T_cast_2 = Ts.sblock_alloc_tensor((4096,))
            T_rms_norm = Ts.sblock_alloc_tensor((1, n, 4096))
            for ax0, ax1, ax2 in T.grid(1, n, 4096):
                with Ts.sblock("T_cast"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(data[v_ax0, v_ax1, v_ax2])
                    Ts.writes(T_cast_1[v_ax0, v_ax1, v_ax2])
                    T_cast_1[v_ax0, v_ax1, v_ax2] = T.Cast("float32", data[v_ax0, v_ax1, v_ax2])
            for ax0, ax1, ax2 in T.grid(1, n, 4096):
                with Ts.sblock("T_multiply"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(T_cast_1[v_ax0, v_ax1, v_ax2])
                    Ts.writes(T_multiply[v_ax0, v_ax1, v_ax2])
                    T_multiply[v_ax0, v_ax1, v_ax2] = T_cast_1[v_ax0, v_ax1, v_ax2] * T_cast_1[v_ax0, v_ax1, v_ax2]
            for ax0, ax1, k2 in T.grid(1, n, 4096):
                with Ts.sblock("T_multiply_red"):
                    v_ax0, v_ax1, v_k2 = Ts.axis.remap("SSR", [ax0, ax1, k2])
                    Ts.reads(T_multiply[v_ax0, v_ax1, v_k2])
                    Ts.writes(T_multiply_red[v_ax0, v_ax1])
                    with Ts.init():
                        T_multiply_red[v_ax0, v_ax1] = T.float32(0)
                    T_multiply_red[v_ax0, v_ax1] = T_multiply_red[v_ax0, v_ax1] + T_multiply[v_ax0, v_ax1, v_k2]
            for ax0, ax1 in T.grid(1, n):
                with Ts.sblock("rsqrt"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(T_multiply_red[v_ax0, v_ax1])
                    Ts.writes(rsqrt[v_ax0, v_ax1])
                    rsqrt[v_ax0, v_ax1] = T.rsqrt(T_multiply_red[v_ax0, v_ax1] * T.float32(0.000244140625) + T.float32(9.9999999999999995e-07))
            for ax0 in range(4096):
                with Ts.sblock("T_cast_1"):
                    v_ax0 = Ts.axis.spatial(4096, ax0)
                    Ts.reads(weight[v_ax0])
                    Ts.writes(T_cast_2[v_ax0])
                    T_cast_2[v_ax0] = T.Cast("float32", weight[v_ax0])
            for ax0, ax1, ax2 in T.grid(1, n, 4096):
                with Ts.sblock("T_rms_norm"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(rsqrt[v_ax0, v_ax1], T_cast_1[v_ax0, v_ax1, v_ax2], T_cast_2[v_ax2])
                    Ts.writes(T_rms_norm[v_ax0, v_ax1, v_ax2])
                    T_rms_norm[v_ax0, v_ax1, v_ax2] = rsqrt[v_ax0, v_ax1] * T_cast_1[v_ax0, v_ax1, v_ax2] * T_cast_2[v_ax2]
            for ax0, ax1, ax2 in T.grid(1, n, 4096):
                with Ts.sblock("T_cast_2"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(T_rms_norm[v_ax0, v_ax1, v_ax2])
                    Ts.writes(T_cast[v_ax0, v_ax1, v_ax2])
                    T_cast[v_ax0, v_ax1, v_ax2] = T.Cast("float16", T_rms_norm[v_ax0, v_ax1, v_ax2])

    n = T.dynamic("n", "int32")

    @I.ir_module
    class After:
        @Ts.function
        def main(data: T.Tensor((1, n, 4096), 'float16'), weight: T.Tensor((4096,), "float16"), T_cast: T.Tensor((1, n, 4096), 'float16')):
            T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})

            # with Ts.sblock("root"):
            T_multiply_local = Ts.sblock_alloc_tensor((1, n, 4096), scope="local")
            T_multiply_red_local = Ts.sblock_alloc_tensor((1, n), scope="local")
            rsqrt_shared = Ts.sblock_alloc_tensor((1, n), scope="shared")
            T_rms_norm_local = Ts.sblock_alloc_tensor((1, n, 4096), scope="local")
            data_local = Ts.sblock_alloc_tensor((1, n, 4096), "float16", scope="local")
            for ax0_ax1_fused in T.parallel(n, annotations={"thread_binding": "blockIdx.x"}):
                for ax2_0 in T.parallel(512, annotations={"thread_binding": "threadIdx.x"}):
                    for ax2_1 in range(1):
                        for ax2_2 in T.vectorized(8):
                            with Ts.sblock("data_local"):
                                v0 = Ts.axis.spatial(1, 0)
                                v1 = Ts.axis.spatial(n, ax0_ax1_fused)
                                v2 = Ts.axis.spatial(4096, ax2_0 * 8 + ax2_1 * 8 + ax2_2)
                                Ts.reads(data[v0, v1, v2])
                                Ts.writes(data_local[v0, v1, v2])
                                data_local[v0, v1, v2] = data[v0, v1, v2]
                    for ax0 in range(8):
                        with Ts.sblock("T_multiply"):
                            v_ax0 = Ts.axis.spatial(1, 0)
                            v_ax1 = Ts.axis.spatial(n, ax0_ax1_fused)
                            v_ax2 = Ts.axis.spatial(4096, ax2_0 * 8 + ax0)
                            Ts.reads(data_local[v_ax0, v_ax1, v_ax2])
                            Ts.writes(T_multiply_local[v_ax0, v_ax1, v_ax2])
                            T_multiply_local[v_ax0, v_ax1, v_ax2] = T.Cast("float32", data_local[v_ax0, v_ax1, v_ax2]) * T.Cast("float32", data_local[v_ax0, v_ax1, v_ax2])
                    for ax0 in range(8):
                        with Ts.sblock("T_multiply_red"):
                            v_ax0 = Ts.axis.spatial(1, 0)
                            v_ax1 = Ts.axis.spatial(n, ax0_ax1_fused)
                            v_k2 = Ts.axis.reduce(4096, ax2_0 * 8 + ax0)
                            Ts.reads(T_multiply_local[v_ax0, v_ax1, v_k2])
                            Ts.writes(T_multiply_red_local[v_ax0, v_ax1])
                            with Ts.init():
                                T_multiply_red_local[v_ax0, v_ax1] = T.float32(0)
                            T_multiply_red_local[v_ax0, v_ax1] = T_multiply_red_local[v_ax0, v_ax1] + T_multiply_local[v_ax0, v_ax1, v_k2]
                with Ts.sblock("rsqrt"):
                    v_ax0 = Ts.axis.spatial(1, 0)
                    v_ax1 = Ts.axis.spatial(n, ax0_ax1_fused)
                    Ts.reads(T_multiply_red_local[v_ax0, v_ax1])
                    Ts.writes(rsqrt_shared[v_ax0, v_ax1])
                    rsqrt_shared[v_ax0, v_ax1] = T.rsqrt(T_multiply_red_local[v_ax0, v_ax1] * T.float32(0.000244140625) + T.float32(9.9999999999999995e-07))
                for ax0_0 in T.parallel(512, annotations={"thread_binding": "threadIdx.x"}):
                    for ax0_1, ax0_2 in T.grid(1, 8):
                        with Ts.sblock("T_rms_norm"):
                            v_ax0 = Ts.axis.spatial(1, 0)
                            v_ax1 = Ts.axis.spatial(n, ax0_ax1_fused)
                            v_ax2 = Ts.axis.spatial(4096, ax0_0 * 8 + ax0_1 * 8 + ax0_2)
                            Ts.reads(rsqrt_shared[v_ax0, v_ax1], data_local[v_ax0, v_ax1, v_ax2], weight[v_ax2])
                            Ts.writes(T_rms_norm_local[v_ax0, v_ax1, v_ax2])
                            T_rms_norm_local[v_ax0, v_ax1, v_ax2] = rsqrt_shared[v_ax0, v_ax1] * T.Cast("float32", data_local[v_ax0, v_ax1, v_ax2]) * T.Cast("float32", weight[v_ax2])
                    for ax0 in T.vectorized(8):
                        with Ts.sblock("T_cast_local"):
                            v0 = Ts.axis.spatial(1, 0)
                            v1 = Ts.axis.spatial(n, ax0_ax1_fused)
                            v2 = Ts.axis.spatial(4096, ax0_0 * 8 + ax0)
                            Ts.reads(T_rms_norm_local[v0, v1, v2])
                            Ts.writes(T_cast[v0, v1, v2])
                            T_cast[v0, v1, v2] = T.Cast("float16", T_rms_norm_local[v0, v1, v2])
    # fmt: on
    _check(Before, After)


def test_rms_norm_without_casting():
    # fmt: off
    n = T.dynamic("n", "int32")

    @I.ir_module
    class Before:
        @Ts.function
        def main(data: T.Tensor((1, n, 4096), "float32"), weight: T.Tensor((4096,), "float32"), T_cast: T.Tensor((1, n, 4096), "float32")):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            T_multiply = Ts.sblock_alloc_tensor((1, n, 4096))
            T_multiply_red = Ts.sblock_alloc_tensor((1, n))
            rsqrt = Ts.sblock_alloc_tensor((1, n))
            T_rms_norm = Ts.sblock_alloc_tensor((1, n, 4096))
            for ax0, ax1, ax2 in T.grid(1, n, 4096):
                with Ts.sblock("T_multiply"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(data[v_ax0, v_ax1, v_ax2])
                    Ts.writes(T_multiply[v_ax0, v_ax1, v_ax2])
                    T_multiply[v_ax0, v_ax1, v_ax2] = data[v_ax0, v_ax1, v_ax2] * data[v_ax0, v_ax1, v_ax2]
            for ax0, ax1, k2 in T.grid(1, n, 4096):
                with Ts.sblock("T_multiply_red"):
                    v_ax0, v_ax1, v_k2 = Ts.axis.remap("SSR", [ax0, ax1, k2])
                    Ts.reads(T_multiply[v_ax0, v_ax1, v_k2])
                    Ts.writes(T_multiply_red[v_ax0, v_ax1])
                    with Ts.init():
                        T_multiply_red[v_ax0, v_ax1] = T.float32(0)
                    T_multiply_red[v_ax0, v_ax1] = T_multiply_red[v_ax0, v_ax1] + T_multiply[v_ax0, v_ax1, v_k2]
            for ax0, ax1 in T.grid(1, n):
                with Ts.sblock("rsqrt"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(T_multiply_red[v_ax0, v_ax1])
                    Ts.writes(rsqrt[v_ax0, v_ax1])
                    rsqrt[v_ax0, v_ax1] = T.rsqrt(T_multiply_red[v_ax0, v_ax1] * T.float32(0.000244140625) + T.float32(9.9999999999999995e-07))
            for ax0, ax1, ax2 in T.grid(1, n, 4096):
                with Ts.sblock("T_rms_norm"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(rsqrt[v_ax0, v_ax1], data[v_ax0, v_ax1, v_ax2], weight[v_ax2])
                    Ts.writes(T_rms_norm[v_ax0, v_ax1, v_ax2])
                    T_rms_norm[v_ax0, v_ax1, v_ax2] = rsqrt[v_ax0, v_ax1] * data[v_ax0, v_ax1, v_ax2] * weight[v_ax2]
            for ax0, ax1, ax2 in T.grid(1, n, 4096):
                with Ts.sblock("T_cast_2"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(T_rms_norm[v_ax0, v_ax1, v_ax2])
                    Ts.writes(T_cast[v_ax0, v_ax1, v_ax2])
                    T_cast[v_ax0, v_ax1, v_ax2] = T_rms_norm[v_ax0, v_ax1, v_ax2]

    n = T.dynamic("n", "int32")

    @I.ir_module
    class After:
        @Ts.function
        def main(data: T.Tensor((1, n, 4096), "float32"), weight: T.Tensor((4096,), "float32"), T_cast: T.Tensor((1, n, 4096), "float32")):
            T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})

            # with Ts.sblock("root"):
            T_multiply_local = Ts.sblock_alloc_tensor((1, n, 4096), scope="local")
            T_multiply_red_local = Ts.sblock_alloc_tensor((1, n), scope="local")
            rsqrt_shared = Ts.sblock_alloc_tensor((1, n), scope="shared")
            T_rms_norm_local = Ts.sblock_alloc_tensor((1, n, 4096), scope="local")
            data_local = Ts.sblock_alloc_tensor((1, n, 4096), scope="local")
            for ax0_ax1_fused in T.parallel(n, annotations={"thread_binding": "blockIdx.x"}):
                for ax2_0 in T.parallel(512, annotations={"thread_binding": "threadIdx.x"}):
                    for ax2_1 in range(1):
                        for ax2_2 in T.vectorized(8):
                            with Ts.sblock("data_local"):
                                v0 = Ts.axis.spatial(1, 0)
                                v1 = Ts.axis.spatial(n, ax0_ax1_fused)
                                v2 = Ts.axis.spatial(4096, ax2_0 * 8 + ax2_1 * 8 + ax2_2)
                                Ts.reads(data[v0, v1, v2])
                                Ts.writes(data_local[v0, v1, v2])
                                data_local[v0, v1, v2] = data[v0, v1, v2]
                    for ax0 in range(8):
                        with Ts.sblock("T_multiply"):
                            v_ax0 = Ts.axis.spatial(1, 0)
                            v_ax1 = Ts.axis.spatial(n, ax0_ax1_fused)
                            v_ax2 = Ts.axis.spatial(4096, ax2_0 * 8 + ax0)
                            Ts.reads(data_local[v_ax0, v_ax1, v_ax2])
                            Ts.writes(T_multiply_local[v_ax0, v_ax1, v_ax2])
                            T_multiply_local[v_ax0, v_ax1, v_ax2] = data_local[v_ax0, v_ax1, v_ax2] * data_local[v_ax0, v_ax1, v_ax2]
                    for ax0 in range(8):
                        with Ts.sblock("T_multiply_red"):
                            v_ax0 = Ts.axis.spatial(1, 0)
                            v_ax1 = Ts.axis.spatial(n, ax0_ax1_fused)
                            v_k2 = Ts.axis.reduce(4096, ax2_0 * 8 + ax0)
                            Ts.reads(T_multiply_local[v_ax0, v_ax1, v_k2])
                            Ts.writes(T_multiply_red_local[v_ax0, v_ax1])
                            with Ts.init():
                                T_multiply_red_local[v_ax0, v_ax1] = T.float32(0)
                            T_multiply_red_local[v_ax0, v_ax1] = T_multiply_red_local[v_ax0, v_ax1] + T_multiply_local[v_ax0, v_ax1, v_k2]
                with Ts.sblock("rsqrt"):
                    v_ax0 = Ts.axis.spatial(1, 0)
                    v_ax1 = Ts.axis.spatial(n, ax0_ax1_fused)
                    Ts.reads(T_multiply_red_local[v_ax0, v_ax1])
                    Ts.writes(rsqrt_shared[v_ax0, v_ax1])
                    rsqrt_shared[v_ax0, v_ax1] = T.rsqrt(T_multiply_red_local[v_ax0, v_ax1] * T.float32(0.000244140625) + T.float32(9.9999999999999995e-07))
                for ax0_0 in T.parallel(512, annotations={"thread_binding": "threadIdx.x"}):
                    for ax0_1, ax0_2 in T.grid(1, 8):
                        with Ts.sblock("T_rms_norm"):
                            v_ax0 = Ts.axis.spatial(1, 0)
                            v_ax1 = Ts.axis.spatial(n, ax0_ax1_fused)
                            v_ax2 = Ts.axis.spatial(4096, ax0_0 * 8 + ax0_1 * 8 + ax0_2)
                            Ts.reads(rsqrt_shared[v_ax0, v_ax1], data_local[v_ax0, v_ax1, v_ax2], weight[v_ax2])
                            Ts.writes(T_rms_norm_local[v_ax0, v_ax1, v_ax2])
                            T_rms_norm_local[v_ax0, v_ax1, v_ax2] = rsqrt_shared[v_ax0, v_ax1] * data_local[v_ax0, v_ax1, v_ax2] * weight[v_ax2]
                    for ax0 in T.vectorized(8):
                        with Ts.sblock("T_cast_local"):
                            v0 = Ts.axis.spatial(1, 0)
                            v1 = Ts.axis.spatial(n, ax0_ax1_fused)
                            v2 = Ts.axis.spatial(4096, ax0_0 * 8 + ax0)
                            Ts.reads(T_rms_norm_local[v0, v1, v2])
                            Ts.writes(T_cast[v0, v1, v2])
                            T_cast[v0, v1, v2] = T_rms_norm_local[v0, v1, v2]
    # fmt: on
    _check(Before, After)


if __name__ == "__main__":
    tvm_tirx.testing.main()
