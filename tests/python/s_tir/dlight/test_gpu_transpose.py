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
# ruff: noqa: E501, F401
import tvm
import tvm_tirx
from tvm.s_tir import dlight as dl
from tvm.s_tir import script as Ts
from tvm_tirx.ir import IRModule, assert_structural_equal
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target


def _check(mod_before: IRModule, mod_after: IRModule):
    target = Target("nvidia/geforce-rtx-3090-ti")
    mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
        dl.gpu.Transpose(), target=target
    )(mod_before)
    assert_structural_equal(mod, mod_after)


def test_transpose():
    # fmt: off
    @I.ir_module
    class Before:
        @Ts.function
        def main(rxplaceholder: T.Tensor((T.int64(512), T.int64(4096)), "float32"), T_transpose: T.Tensor((T.int64(4096), T.int64(512)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for ax0, ax1 in T.grid(T.int64(4096), T.int64(512)):
                with Ts.sblock("T_transpose"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    T_transpose[v_ax0, v_ax1] = rxplaceholder[v_ax1, v_ax0]

    @I.ir_module
    class After:
        @Ts.function
        def main(rxplaceholder: T.Tensor((T.int64(512), T.int64(4096)), "float32"), T_transpose: T.Tensor((T.int64(4096), T.int64(512)), "float32")):
            T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
            # with Ts.sblock("root"):
            rxplaceholder_shared = Ts.sblock_alloc_tensor((T.int64(512), T.int64(4096)), scope="shared")
            for ax0_0_0 in T.parallel(T.int64(512), annotations={"thread_binding": "blockIdx.y", "tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                for ax1_0 in T.parallel(T.int64(32), annotations={"thread_binding": "blockIdx.x"}):
                    for ax0_ax1_fused_0 in range(T.int64(1)):
                        for ax0_ax1_fused_1 in T.parallel(T.int64(8), annotations={"thread_binding": "threadIdx.y"}):
                            for ax0_ax1_fused_2 in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.x"}):
                                for ax0_ax1_fused_3 in T.unroll(T.int64(1)):
                                    with Ts.sblock("rxplaceholder_shared"):
                                        v0 = Ts.axis.spatial(T.int64(512), ax1_0 * T.int64(16) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(16) + ax0_ax1_fused_2 + ax0_ax1_fused_3) // T.int64(8))
                                        v1 = Ts.axis.spatial(T.int64(4096), ax0_0_0 * T.int64(8) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(16) + ax0_ax1_fused_2 + ax0_ax1_fused_3) % T.int64(8))
                                        Ts.reads(rxplaceholder[v0, v1])
                                        Ts.writes(rxplaceholder_shared[v0, v1])
                                        Ts.sblock_attr({"tensor_dim_align": [[0, 0, 32, 1]]})
                                        rxplaceholder_shared[v0, v1] = rxplaceholder[v0, v1]
                    for ax0_0_1 in T.parallel(T.int64(8), annotations={"thread_binding": "threadIdx.y"}):
                        for ax1_1 in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.x"}):
                            for ax0_1_0 in range(T.int64(1)):
                                for ax0_1_1 in range(T.int64(1)):
                                    with Ts.sblock("T_transpose"):
                                        v0 = Ts.axis.spatial(T.int64(4096), ax0_0_0 * T.int64(8) + ax0_0_1 + ax0_1_0 + ax0_1_1)
                                        v1 = Ts.axis.spatial(T.int64(512), ax1_0 * T.int64(16) + ax1_1)
                                        Ts.reads(rxplaceholder_shared[v1, v0])
                                        Ts.writes(T_transpose[v0, v1])
                                        T_transpose[v0, v1] = rxplaceholder_shared[v1, v0]
    # fmt: on
    _check(Before, After)


def test_decode_transpose():
    # fmt: off
    @I.ir_module
    class Before:
        @Ts.function
        def main(rxplaceholder: T.Tensor((T.int64(512), T.int64(4096)), "uint32"), rxplaceholder_1: T.Tensor((T.int64(128), T.int64(4096)), "uint32"), T_transpose: T.Tensor((T.int64(4096), T.int64(4096)), "float32")):
            T.func_attr({"tirx.noalias": True})
            decode = Ts.sblock_alloc_tensor((T.int64(4096), T.int64(4096)))
            for i, j in T.grid(T.int64(4096), T.int64(4096)):
                with Ts.sblock("decode"):
                    v_i, v_j = Ts.axis.remap("SS", [i, j])
                    Ts.reads(rxplaceholder[v_i // T.int64(8), v_j], rxplaceholder_1[v_i // T.int64(32), v_j])
                    Ts.writes(decode[v_i, v_j])
                    decode[v_i, v_j] = T.Cast("float32", T.bitwise_and(T.shift_right(rxplaceholder[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8) * T.int64(4))), T.uint32(15))) * T.reinterpret( T.shift_left(T.bitwise_and(rxplaceholder_1[v_i // T.int64(32), v_j], T.uint32(65535)), T.uint32(16)), ty="float32") + T.reinterpret( T.shift_left(T.bitwise_and(T.shift_right(rxplaceholder_1[v_i // T.int64(32), v_j], T.uint32(16)), T.uint32(65535)), T.uint32(16)), ty="float32")
            for ax0, ax1 in T.grid(T.int64(4096), T.int64(4096)):
                with Ts.sblock("T_transpose"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(decode[v_ax1, v_ax0])
                    Ts.writes(T_transpose[v_ax0, v_ax1])
                    T_transpose[v_ax0, v_ax1] = decode[v_ax1, v_ax0]

    @I.ir_module
    class After:
        @Ts.function
        def main(rxplaceholder: T.Tensor((T.int64(512), T.int64(4096)), "uint32"), rxplaceholder_1: T.Tensor((T.int64(128), T.int64(4096)), "uint32"), T_transpose: T.Tensor((T.int64(4096), T.int64(4096)), "float32")):
            T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
            decode_shared = Ts.sblock_alloc_tensor((T.int64(4096), T.int64(4096)), scope="shared")
            for ax0_0_0 in T.parallel(T.int64(64), annotations={"thread_binding": "blockIdx.y", "tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                for ax1_0 in T.parallel(T.int64(256), annotations={"thread_binding": "blockIdx.x"}):
                    for ax0_ax1_fused_0 in range(T.int64(1)):
                        for ax0_ax1_fused_1 in T.parallel(T.int64(8), annotations={"thread_binding": "threadIdx.y"}):
                            for ax0_ax1_fused_2 in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.x"}):
                                for ax0_ax1_fused_3 in T.unroll(T.int64(8)):
                                    with Ts.sblock("decode_shared"):
                                        v0 = Ts.axis.spatial(T.int64(4096), ax1_0 * T.int64(16) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(64))
                                        v1 = Ts.axis.spatial(T.int64(4096), ax0_0_0 * T.int64(64) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(64))
                                        Ts.reads(rxplaceholder[v0 // T.int64(8), v1], rxplaceholder_1[v0 // T.int64(32), v1])
                                        Ts.writes(decode_shared[v0, v1])
                                        Ts.sblock_attr({"tensor_dim_align": [[0, 0, 32, 1]]})
                                        decode_shared[v0, v1] = T.Cast("float32", T.bitwise_and(T.shift_right(rxplaceholder[v0 // T.int64(8), v1], T.Cast("uint32", v0 % T.int64(8) * T.int64(4))), T.uint32(15))) * T.reinterpret( T.shift_left(T.bitwise_and(rxplaceholder_1[v0 // T.int64(32), v1], T.uint32(65535)), T.uint32(16)), ty="float32") + T.reinterpret( T.shift_left(T.bitwise_and(T.shift_right(rxplaceholder_1[v0 // T.int64(32), v1], T.uint32(16)), T.uint32(65535)), T.uint32(16)), ty="float32")
                    for ax0_0_1 in T.parallel(T.int64(8), annotations={"thread_binding": "threadIdx.y"}):
                        for ax1_1 in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.x"}):
                            for ax0_1_0 in range(T.int64(2)):
                                for ax0_1_1 in T.vectorized(T.int64(4)):
                                    with Ts.sblock("T_transpose"):
                                        v0 = Ts.axis.spatial(T.int64(4096), ax0_0_0 * T.int64(64) + ax0_0_1 * T.int64(8) + ax0_1_0 * T.int64(4) + ax0_1_1)
                                        v1 = Ts.axis.spatial(T.int64(4096), ax1_0 * T.int64(16) + ax1_1)
                                        Ts.reads(decode_shared[v1, v0])
                                        Ts.writes(T_transpose[v0, v1])
                                        T_transpose[v0, v1] = decode_shared[v1, v0]
    # fmt: on
    _check(Before, After)


def test_decode_int3_transpose():
    # fmt: off
    @I.ir_module
    class Before:
        @Ts.function
        def main(A: T.Tensor((T.int64(412), T.int64(4096)), "uint32"), B: T.Tensor((T.int64(103), T.int64(4096)), "float16"), T_transpose: T.Tensor((T.int64(4096), T.int64(4096)), "float16")):
            T.func_attr({"tirx.noalias": True})
            decode_1 = Ts.sblock_alloc_tensor((T.int64(4096), T.int64(4096)), "float16")
            for i, j in T.grid(T.int64(4096), T.int64(4096)):
                with Ts.sblock("decode"):
                    v_i, v_j = Ts.axis.remap("SS", [i, j])
                    Ts.reads(A[v_i // T.int64(10), v_j], B[v_i // T.int64(40), v_j])
                    Ts.writes(decode_1[v_i, v_j])
                    decode_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(A[v_i // T.int64(10), v_j], T.Cast("uint32", v_i % T.int64(10)) * T.uint32(3)), T.uint32(7))) - T.float16(3)) * B[v_i // T.int64(40), v_j]
            for ax0, ax1 in T.grid(T.int64(4096), T.int64(4096)):
                with Ts.sblock("T_transpose"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(decode_1[v_ax1, v_ax0])
                    Ts.writes(T_transpose[v_ax0, v_ax1])
                    T_transpose[v_ax0, v_ax1] = decode_1[v_ax1, v_ax0]

    @I.ir_module
    class After:
        @Ts.function
        def main(A: T.Tensor((T.int64(412), T.int64(4096)), "uint32"), B: T.Tensor((T.int64(103), T.int64(4096)), "float16"), T_transpose: T.Tensor((T.int64(4096), T.int64(4096)), "float16")):
            T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
            # with Ts.sblock("root"):
            decode_1_shared = Ts.sblock_alloc_tensor((T.int64(4096), T.int64(4096)), "float16", scope="shared")
            for ax0_0_0 in T.parallel(T.int64(52), annotations={"thread_binding": "blockIdx.y", "tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                for ax1_0 in T.parallel(T.int64(256), annotations={"thread_binding": "blockIdx.x"}):
                    for ax0_ax1_fused_0 in range(T.int64(2)):
                        for ax0_ax1_fused_1 in T.parallel(T.int64(8), annotations={"thread_binding": "threadIdx.y"}):
                            for ax0_ax1_fused_2 in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.x"}):
                                for ax0_ax1_fused_3 in T.unroll(T.int64(10)):
                                    with Ts.sblock("decode_1_shared"):
                                        v0 = Ts.axis.spatial(T.int64(4096), ax1_0 * T.int64(16) + (ax0_ax1_fused_0 * T.int64(1280) + ax0_ax1_fused_1 * T.int64(160) + ax0_ax1_fused_2 * T.int64(10) + ax0_ax1_fused_3) // T.int64(82))
                                        v1 = Ts.axis.spatial(T.int64(4096), ax0_0_0 * T.int64(80) + (ax0_ax1_fused_0 * T.int64(1280) + ax0_ax1_fused_1 * T.int64(160) + ax0_ax1_fused_2 * T.int64(10) + ax0_ax1_fused_3) % T.int64(82))
                                        Ts.where(ax0_0_0 * T.int64(80) + (((ax0_ax1_fused_0 * T.int64(8) + ax0_ax1_fused_1) * T.int64(16) + ax0_ax1_fused_2) * T.int64(10) + ax0_ax1_fused_3) % T.int64(82) < T.int64(4096) and ((ax0_ax1_fused_0 * T.int64(8) + ax0_ax1_fused_1) * T.int64(16) + ax0_ax1_fused_2) * T.int64(10) + ax0_ax1_fused_3 < T.int64(1312))
                                        Ts.reads(A[v0 // T.int64(10), v1], B[v0 // T.int64(40), v1])
                                        Ts.writes(decode_1_shared[v0, v1])
                                        Ts.sblock_attr({"tensor_dim_align": [[0, 0, 32, 1]]})
                                        decode_1_shared[v0, v1] = (T.Cast("float16", T.bitwise_and(T.shift_right(A[v0 // T.int64(10), v1], T.Cast("uint32", v0 % T.int64(10)) * T.uint32(3)), T.uint32(7))) - T.float16(3)) * B[v0 // T.int64(40), v1]
                    for ax0_0_1 in T.parallel(T.int64(8), annotations={"thread_binding": "threadIdx.y"}):
                        for ax1_1 in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.x"}):
                            for ax0_1_0 in range(T.int64(3)):
                                for ax0_1_1 in T.vectorized(T.int64(4)):
                                    with Ts.sblock("T_transpose"):
                                        v0 = Ts.axis.spatial(T.int64(4096), (ax0_0_0 * T.int64(8) + ax0_0_1) * T.int64(10) + (ax0_1_0 * T.int64(4) + ax0_1_1))
                                        v1 = Ts.axis.spatial(T.int64(4096), ax1_0 * T.int64(16) + ax1_1)
                                        Ts.where((ax0_0_0 * T.int64(8) + ax0_0_1) * T.int64(10) + (ax0_1_0 * T.int64(4) + ax0_1_1) < T.int64(4096) and ax0_0_0 * T.int64(8) + ax0_0_1 < T.int64(410) and ax0_1_0 * T.int64(4) + ax0_1_1 < T.int64(10))
                                        Ts.reads(decode_1_shared[v1, v0])
                                        Ts.writes(T_transpose[v0, v1])
                                        T_transpose[v0, v1] = decode_1_shared[v1, v0]
    # fmt: on
    _check(Before, After)
