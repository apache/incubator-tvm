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
import tvm_tirx.testing
from tvm.s_tir import dlight as dl
from tvm.s_tir import script as Ts
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target


def test_conv3d():
    # fmt: off
    @Ts.function(private=True)
    def before(
        A: T.Tensor((14308, 3, 2, 14, 14), "float16"),
        W: T.Tensor((1280, 3, 2, 14, 14), "float16"),
        C: T.Tensor((14308, 1280, 1, 1, 1), "float16"),
    ):
        pad_A = Ts.sblock_alloc_tensor((14308, 3, 2, 14, 14), "float16")
        for i0, i1, i2, i3, i4 in T.grid(14308, 3, 2, 14, 14):
            with Ts.sblock("pad_A"):
                v_i0, v_i1, v_i2, v_i3, v_i4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                pad_A[v_i0, v_i1, v_i2, v_i3, v_i4] = A[v_i0, v_i1, v_i2, v_i3, v_i4]
        for nn, ff, yy, xx, zz, rc, ry, rx, rz in T.grid(14308, 1280, 1, 1, 1, 3, 2, 14, 14):
            with Ts.sblock("C"):
                v_nn, v_ff, v_yy, v_xx, v_zz, v_rc, v_ry, v_rx, v_rz = Ts.axis.remap("SSSSSRRRR", [nn, ff, yy, xx, zz, rc, ry, rx, rz])
                with Ts.init():
                    C[v_nn, v_ff, v_yy, v_xx, v_zz] = T.float16(0.0)
                C[v_nn, v_ff, v_yy, v_xx, v_zz] += pad_A[v_nn, v_rc, v_yy * 2 + v_ry, v_xx * 14 + v_rx, v_zz * 14 + v_rz]* W[v_ff, v_rc, v_ry, v_rx, v_rz]

    @Ts.function(private=True)
    def expected(A: T.Tensor((14308, 3, 2, 14, 14), "float16"), W: T.Tensor((1280, 3, 2, 14, 14), "float16"), C: T.Tensor((14308, 1280, 1, 1, 1), "float16")):
        T.func_attr({"tirx.is_scheduled": True})
        # with Ts.sblock("root"):
        C_reindex_pad_local = Ts.sblock_alloc_tensor((1, 14336, 1280), "float16", scope="local")
        pad_A_reindex_pad_shared = Ts.sblock_alloc_tensor((1, 14336, 1184), "float16", scope="shared")
        W_reindex_pad_shared = Ts.sblock_alloc_tensor((1, 1280, 1184), "float16", scope="shared")
        for ax0_ax2_0_fused in T.parallel(20, annotations={"thread_binding": "blockIdx.y"}):
            for ax1_0 in T.parallel(448, annotations={"thread_binding": "blockIdx.x"}):
                for ax2_1 in T.parallel(1, annotations={"thread_binding": "vthread.y"}):
                    for ax1_1 in T.parallel(1, annotations={"thread_binding": "vthread.x"}):
                        for ax2_2 in T.parallel(16, annotations={"thread_binding": "threadIdx.y"}):
                            for ax1_2 in T.parallel(8, annotations={"thread_binding": "threadIdx.x", "tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(4, 2):
                                    for ax2_3_1_init in T.vectorized(2):
                                        with Ts.sblock("C_init"):
                                            v0 = Ts.axis.spatial(1, 0)
                                            v1 = Ts.axis.spatial(14336, ax1_0 * 32 + ax1_1 * 32 + ax1_2 * 4 + ax1_3_init)
                                            v2 = Ts.axis.spatial(1280, ax0_ax2_0_fused * 64 + ax2_1 * 64 + ax2_2 * 4 + ax2_3_0_init * 2 + ax2_3_1_init)
                                            C_reindex_pad_local[0, v1, v2] = T.float16(0.0)
                                for ax3_0 in range(74):
                                    for ax0_ax1_ax2_fused_0 in T.parallel(16, annotations={"thread_binding": "threadIdx.y"}):
                                        for ax0_ax1_ax2_fused_1 in T.parallel(8, annotations={"thread_binding": "threadIdx.x"}):
                                            for ax0_ax1_ax2_fused_2 in range(2):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(2):
                                                    with Ts.sblock("pad_A_reindex_pad_shared"):
                                                        v0 = Ts.axis.spatial(1, 0)
                                                        v1 = Ts.axis.spatial(14336, ax1_0 * 32 + (ax0_ax1_ax2_fused_0 * 32 + ax0_ax1_ax2_fused_1 * 4 + ax0_ax1_ax2_fused_2 * 2 + ax0_ax1_ax2_fused_3) // 16)
                                                        v2 = Ts.axis.spatial(1184, ax3_0 * 16 + (ax0_ax1_ax2_fused_0 * 32 + ax0_ax1_ax2_fused_1 * 4 + ax0_ax1_ax2_fused_2 * 2 + ax0_ax1_ax2_fused_3) % 16)
                                                        Ts.sblock_attr({"tensor_dim_align": [[0, 1, 8, 2]]})
                                                        pad_A_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v1 < 14308 and v2 < 1176, A[v1, v2 // 392, v2 // 196 % 2, v2 // 14 % 14, v2 % 14], T.float16(0.0))
                                    for ax0_ax1_ax2_fused_0 in T.parallel(16, annotations={"thread_binding": "threadIdx.y"}):
                                        for ax0_ax1_ax2_fused_1 in T.parallel(8, annotations={"thread_binding": "threadIdx.x"}):
                                            for ax0_ax1_ax2_fused_2 in range(4):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(2):
                                                    with Ts.sblock("W_reindex_pad_shared"):
                                                        v0 = Ts.axis.spatial(1, 0)
                                                        v1 = Ts.axis.spatial(1280, ax0_ax2_0_fused * 64 + (ax0_ax1_ax2_fused_0 * 64 + ax0_ax1_ax2_fused_1 * 8 + ax0_ax1_ax2_fused_2 * 2 + ax0_ax1_ax2_fused_3) // 16)
                                                        v2 = Ts.axis.spatial(1184, ax3_0 * 16 + (ax0_ax1_ax2_fused_0 * 64 + ax0_ax1_ax2_fused_1 * 8 + ax0_ax1_ax2_fused_2 * 2 + ax0_ax1_ax2_fused_3) % 16)
                                                        Ts.sblock_attr({"tensor_dim_align": [[0, 1, 8, 2]]})
                                                        W_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v2 < 1176, W[v1, v2 // 392, v2 // 196 % 2, v2 // 14 % 14, v2 % 14], T.float16(0.0))
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(16, 4, 2):
                                        for ax2_3_1 in T.vectorized(2):
                                            with Ts.sblock("C_update"):
                                                v0 = Ts.axis.spatial(1, 0)
                                                v1 = Ts.axis.spatial(14336, ax1_0 * 32 + ax1_1 * 32 + ax1_2 * 4 + ax1_3)
                                                v2 = Ts.axis.spatial(1280, ax0_ax2_0_fused * 64 + ax2_1 * 64 + ax2_2 * 4 + ax2_3_0 * 2 + ax2_3_1)
                                                v3 = Ts.axis.reduce(1184, ax3_0 * 16 + ax3_1)
                                                C_reindex_pad_local[0, v1, v2] = C_reindex_pad_local[0, v1, v2] + pad_A_reindex_pad_shared[0, v1, v3] * W_reindex_pad_shared[0, v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(1, 4, 2):
                                    for ax2_1_1 in T.vectorized(2):
                                        with Ts.sblock("C_reindex_pad_local"):
                                            v0 = Ts.axis.spatial(1, ax0)
                                            v1 = Ts.axis.spatial(14336, ax1_0 * 32 + ax1_2 * 4 + ax1)
                                            v2 = Ts.axis.spatial(1280, ax0_ax2_0_fused * 64 + ax2_2 * 4 + ax2_0 * 2 + ax2_1_1)
                                            Ts.where(ax1_0 * 32 + ax1_2 * 4 + ax1 < 14308)
                                            C[v1, v2, 0, 0, 0] = C_reindex_pad_local[v0, v1, v2]
    # fmt: on

    mod = tvm_tirx.IRModule({"main": before})
    mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul(), target=Target("nvidia/geforce-gtx-1080-ti"))(mod)
    tvm_tirx.ir.assert_structural_equal(mod["main"], expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
