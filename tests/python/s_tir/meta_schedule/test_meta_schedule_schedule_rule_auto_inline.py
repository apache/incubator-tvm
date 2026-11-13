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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring
# ruff: noqa: E501, E741, F841
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import Schedule
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir import script as Ts
from tvm.s_tir.meta_schedule.testing.space_generation import generate_design_space
from tvm_tirx.ir.base import assert_structural_equal
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,no-self-argument,line-too-long,chained-comparison,not-callable,too-many-nested-blocks

@tvm_tirx.script.ir_module
class Conv2DBiasBnReLU:
    @Ts.function
    def main(X: T.Tensor([1, 512, 56, 56], dtype='float32'), W: T.Tensor([512, 512, 3, 3], dtype='float32'), B: T.Tensor([512, 1, 1], dtype='float32'), bn_scale: T.Tensor([512, 1, 1], dtype='float32'), bn_offset: T.Tensor([512, 1, 1], dtype='float32'), compute: T.Tensor([1, 512, 56, 56], dtype='float32')) -> None:

        pad_temp = Ts.sblock_alloc_tensor([1, 512, 58, 58], dtype="float32")
        compute_1 = Ts.sblock_alloc_tensor([1, 512, 56, 56], dtype="float32")
        bias_add = Ts.sblock_alloc_tensor([1, 512, 56, 56], dtype="float32")
        bn_mul = Ts.sblock_alloc_tensor([1, 512, 56, 56], dtype="float32")
        bn_add = Ts.sblock_alloc_tensor([1, 512, 56, 56], dtype="float32")
        for i0, i1, i2, i3 in T.grid(1, 512, 58, 58):
            with Ts.sblock("pad_temp"):
                i0_1, i1_1, i2_1, i3_1 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                pad_temp[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(i2_1 >= 1 and i2_1 < 57 and i3_1 >= 1 and i3_1 < 57, X[i0_1, i1_1, i2_1 - 1, i3_1 - 1], T.float32(0))
        for i0, i1, i2, i3, i4, i5, i6 in T.grid(1, 512, 56, 56, 512, 3, 3):
            with Ts.sblock("compute"):
                nn, ff, yy, xx, rc, ry, rx = Ts.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
                with Ts.init():
                    compute_1[nn, ff, yy, xx] = T.float32(0)
                compute_1[nn, ff, yy, xx] = compute_1[nn, ff, yy, xx] + pad_temp[nn, rc, yy + ry, xx + rx] * W[ff, rc, ry, rx]
        for i0, i1, i2, i3 in T.grid(1, 512, 56, 56):
            with Ts.sblock("bias_add"):
                i, j, k, l = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                bias_add[i, j, k, l] = compute_1[i, j, k, l] + B[j, 0, 0]
        for i0, i1, i2, i3 in T.grid(1, 512, 56, 56):
            with Ts.sblock("bn_mul"):
                i, j, k, l = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                bn_mul[i, j, k, l] = bias_add[i, j, k, l] * bn_scale[j, 0, 0]
        for i0, i1, i2, i3 in T.grid(1, 512, 56, 56):
            with Ts.sblock("bn_add"):
                i, j, k, l = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                bn_add[i, j, k, l] = bn_mul[i, j, k, l] + bn_offset[j, 0, 0]
        for i0, i1, i2, i3 in T.grid(1, 512, 56, 56):
            with Ts.sblock("compute_1"):
                i0_2, i1_2, i2_2, i3_2 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                compute[i0_2, i1_2, i2_2, i3_2] = T.max(bn_add[i0_2, i1_2, i2_2, i3_2], T.float32(0))

@tvm_tirx.script.ir_module
class Conv2DBiasBnReLUInlined:
    @Ts.function
    def main(X: T.Tensor([1, 512, 56, 56], dtype='float32'), W: T.Tensor([512, 512, 3, 3], dtype='float32'), B: T.Tensor([512, 1, 1], dtype='float32'), bn_scale: T.Tensor([512, 1, 1], dtype='float32'), bn_offset: T.Tensor([512, 1, 1], dtype='float32'), compute: T.Tensor([1, 512, 56, 56], dtype='float32')) -> None:

        pad_temp = Ts.sblock_alloc_tensor([1, 512, 58, 58], dtype="float32")
        compute_1 = Ts.sblock_alloc_tensor([1, 512, 56, 56], dtype="float32")
        for i0, i1, i2, i3 in T.grid(1, 512, 58, 58):
            with Ts.sblock("pad_temp"):
                i0_1, i1_1, i2_1, i3_1 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                pad_temp[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(i2_1 >= 1 and i2_1 < 57 and i3_1 >= 1 and i3_1 < 57, X[i0_1, i1_1, i2_1 - 1, i3_1 - 1], T.float32(0))
        for i0, i1, i2, i3, i4, i5, i6 in T.grid(1, 512, 56, 56, 512, 3, 3):
            with Ts.sblock("compute"):
                nn, ff, yy, xx, rc, ry, rx = Ts.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
                with Ts.init():
                    compute_1[nn, ff, yy, xx] = T.float32(0)
                compute_1[nn, ff, yy, xx] = compute_1[nn, ff, yy, xx] + pad_temp[nn, rc, yy + ry, xx + rx] * W[ff, rc, ry, rx]
        for i0, i1, i2, i3 in T.grid(1, 512, 56, 56):
            with Ts.sblock("compute_1"):
                i0_2, i1_2, i2_2, i3_2 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                compute[i0_2, i1_2, i2_2, i3_2] = T.max((compute_1[i0_2, i1_2, i2_2, i3_2] + B[i1_2, 0, 0]) * bn_scale[i1_2, 0, 0] + bn_offset[i1_2, 0, 0], T.float32(0))

@tvm_tirx.script.ir_module
class MultiLevelTiledConv2D:
    @Ts.function
    def main(X: T.Tensor([1, 512, 56, 56], dtype='float32'), W: T.Tensor([512, 512, 3, 3], dtype='float32'), B: T.Tensor([512, 1, 1], dtype='float32'), bn_scale: T.Tensor([512, 1, 1], dtype='float32'), bn_offset: T.Tensor([512, 1, 1], dtype='float32'), compute: T.Tensor([1, 512, 56, 56], dtype='float32')) -> None:

        pad_temp = Ts.sblock_alloc_tensor([1, 512, 58, 58], dtype="float32")
        compute_1 = Ts.sblock_alloc_tensor([1, 512, 56, 56], dtype="float32")
        compute_local = Ts.sblock_alloc_tensor([1, 512, 56, 56], dtype="float32", scope="local")
        pad_temp_shared = Ts.sblock_alloc_tensor([1, 512, 58, 58], dtype="float32", scope="shared")
        W_shared = Ts.sblock_alloc_tensor([512, 512, 3, 3], dtype="float32", scope="shared")
        for i0, i1, i2, i3 in T.grid(1, 512, 58, 58):
            with Ts.sblock("pad_temp"):
                i0_1, i1_1, i2_1, i3_1 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                pad_temp[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(i2_1 >= 1 and i2_1 < 57 and i3_1 >= 1 and i3_1 < 57, X[i0_1, i1_1, i2_1 - 1, i3_1 - 1], T.float32(0))
        for i0_0_i1_0_i2_0_i3_0_fused in T.parallel(0, 224, annotations={"thread_binding": "blockIdx.x"}):
            for i0_1_i1_1_i2_1_i3_1_fused in T.parallel(0, 2, annotations={"thread_binding": "vthread.x"}):
                for i0_2_i1_2_i2_2_i3_2_fused in T.parallel(0, 8, annotations={"thread_binding": "threadIdx.x"}):
                    for i4_0, i5_0, i6_0 in T.grid(1, 3, 1):
                        for ax0_ax1_ax2_ax3_fused_0 in T.serial(0, 40960, annotations={"meta_schedule.cooperative_fetch":1}):
                            for ax0_ax1_ax2_ax3_fused_1 in T.vectorized(0, 3):
                                with Ts.sblock("pad_temp_shared"):
                                    v0 = Ts.axis.spatial(1, 0)
                                    v1 = Ts.axis.spatial(512, (ax0_ax1_ax2_ax3_fused_0 * 3 + ax0_ax1_ax2_ax3_fused_1) // 30 // 8 % 512)
                                    v2 = Ts.axis.spatial(58, i0_0_i1_0_i2_0_i3_0_fused % 14 // 2 * 8 + i5_0 + (ax0_ax1_ax2_ax3_fused_0 * 3 + ax0_ax1_ax2_ax3_fused_1) // 30 % 8)
                                    v3 = Ts.axis.spatial(58, i0_0_i1_0_i2_0_i3_0_fused % 2 * 28 + (ax0_ax1_ax2_ax3_fused_0 * 3 + ax0_ax1_ax2_ax3_fused_1) % 30)
                                    pad_temp_shared[v0, v1, v2, v3] = pad_temp[v0, v1, v2, v3]
                        for ax0_ax1_ax2_ax3_fused_0 in T.serial(0, 12288, annotations={"meta_schedule.cooperative_fetch":1}):
                            for ax0_ax1_ax2_ax3_fused_1 in T.vectorized(0, 4):
                                with Ts.sblock("W_shared"):
                                    v0 = Ts.axis.spatial(512, i0_0_i1_0_i2_0_i3_0_fused // 14 * 32 + (ax0_ax1_ax2_ax3_fused_0 * 4 + ax0_ax1_ax2_ax3_fused_1) // 1536)
                                    v1 = Ts.axis.spatial(512, (ax0_ax1_ax2_ax3_fused_0 * 4 + ax0_ax1_ax2_ax3_fused_1) // 3 % 512)
                                    v2 = Ts.axis.spatial(3, i5_0)
                                    v3 = Ts.axis.spatial(3, (ax0_ax1_ax2_ax3_fused_0 * 4 + ax0_ax1_ax2_ax3_fused_1) % 3)
                                    W_shared[v0, v1, v2, v3] = W[v0, v1, v2, v3]
                        for i4_1, i5_1, i6_1, i0_3, i1_3, i2_3, i3_3, i4_2, i5_2, i6_2, i0_4, i1_4, i2_4, i3_4 in T.grid(32, 1, 1, 1, 1, 1, 1, 16, 1, 3, 1, 8, 2, 28):
                            with Ts.sblock("compute"):
                                nn = Ts.axis.spatial(1, 0)
                                ff = Ts.axis.spatial(512, i0_0_i1_0_i2_0_i3_0_fused // 14 * 32 + i0_2_i1_2_i2_2_i3_2_fused // 2 * 8 + i1_4)
                                yy = Ts.axis.spatial(56, i0_0_i1_0_i2_0_i3_0_fused // 2 % 7 * 8 + i0_1_i1_1_i2_1_i3_1_fused * 4 + i0_2_i1_2_i2_2_i3_2_fused % 2 * 2 + i2_4)
                                xx = Ts.axis.spatial(56, i0_0_i1_0_i2_0_i3_0_fused % 2 * 28 + i3_4)
                                rc = Ts.axis.reduce(512, i4_1 * 16 + i4_2)
                                ry, rx = Ts.axis.remap("RR", [i5_0, i6_2])
                                with Ts.init():
                                    compute_local[nn, ff, yy, xx] = T.float32(0)
                                compute_local[nn, ff, yy, xx] = compute_local[nn, ff, yy, xx] + pad_temp_shared[nn, rc, yy + ry, xx + rx] * W_shared[ff, rc, ry, rx]
                    for ax0, ax1, ax2, ax3 in T.grid(1, 8, 2, 28):
                        with Ts.sblock("compute_local"):
                            v0 = Ts.axis.spatial(1, ax0)
                            v1 = Ts.axis.spatial(512, i0_0_i1_0_i2_0_i3_0_fused // 14 * 32 + i0_2_i1_2_i2_2_i3_2_fused // 2 * 8 + ax1)
                            v2 = Ts.axis.spatial(56, i0_0_i1_0_i2_0_i3_0_fused % 14 // 2 * 8 + i0_1_i1_1_i2_1_i3_1_fused * 4 + i0_2_i1_2_i2_2_i3_2_fused % 2 * 2 + ax2)
                            v3 = Ts.axis.spatial(56, i0_0_i1_0_i2_0_i3_0_fused % 2 * 28 + ax3)
                            compute_1[v0, v1, v2, v3] = compute_local[v0, v1, v2, v3]
        for i0, i1, i2, i3 in T.grid(1, 512, 56, 56):
            with Ts.sblock("compute_1"):
                i0_2, i1_2, i2_2, i3_2 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                compute[i0_2, i1_2, i2_2, i3_2] = T.max((compute_1[i0_2, i1_2, i2_2, i3_2] + B[i1_2, 0, 0]) * bn_scale[i1_2, 0, 0] + bn_offset[i1_2, 0, 0], T.float32(0))

@tvm_tirx.script.ir_module
class MultiLevelTiledConv2DAfterInline:
    @Ts.function
    def main(X: T.Tensor((1, 512, 56, 56), "float32"), W: T.Tensor((512, 512, 3, 3), "float32"), B: T.Tensor((512, 1, 1), "float32"), bn_scale: T.Tensor((512, 1, 1), "float32"), bn_offset: T.Tensor((512, 1, 1), "float32"), compute: T.Tensor((1, 512, 56, 56), "float32")) -> None:
        compute_local = Ts.sblock_alloc_tensor([1, 512, 56, 56], dtype="float32", scope="local")
        for i0_0_i1_0_i2_0_i3_0_fused in T.parallel(224, annotations={"thread_binding": "blockIdx.x"}):
            for i0_1_i1_1_i2_1_i3_1_fused in T.parallel(2, annotations={"thread_binding": "vthread.x"}):
                for i0_2_i1_2_i2_2_i3_2_fused in T.parallel(8, annotations={"thread_binding": "threadIdx.x"}):
                    for i4_0, i5_0, i6_0, i4_1, i5_1, i6_1, i0_3, i1_3, i2_3, i3_3, i4_2, i5_2, i6_2, i0_4, i1_4, i2_4, i3_4 in T.grid(1, 3, 1, 32, 1, 1, 1, 1, 1, 1, 16, 1, 3, 1, 8, 2, 28):
                        with Ts.sblock("compute"):
                            nn = Ts.axis.spatial(1, 0)
                            ff = Ts.axis.spatial(512, i0_0_i1_0_i2_0_i3_0_fused // 14 * 32 + i0_2_i1_2_i2_2_i3_2_fused // 2 * 8 + i1_4)
                            yy = Ts.axis.spatial(56, i0_0_i1_0_i2_0_i3_0_fused // 2 % 7 * 8 + i0_1_i1_1_i2_1_i3_1_fused * 4 + i0_2_i1_2_i2_2_i3_2_fused % 2 * 2 + i2_4)
                            xx = Ts.axis.spatial(56, i0_0_i1_0_i2_0_i3_0_fused % 2 * 28 + i3_4)
                            rc = Ts.axis.reduce(512, i4_1 * 16 + i4_2)
                            ry, rx = Ts.axis.remap("RR", [i5_0, i6_2])
                            with Ts.init():
                                compute_local[nn, ff, yy, xx] = T.float32(0)
                            compute_local[nn, ff, yy, xx] = compute_local[nn, ff, yy, xx] + T.if_then_else(yy + ry >= 1 and yy + ry < 57 and xx + rx >= 1 and xx + rx < 57, X[nn, rc, yy + ry - 1, xx + rx - 1], T.float32(0)) * W[ff, rc, ry, rx]
                    for ax0, ax1, ax2, ax3 in T.grid(1, 8, 2, 28):
                        with Ts.sblock("compute_local"):
                            v0 = Ts.axis.spatial(1, ax0)
                            v1 = Ts.axis.spatial(512, i0_0_i1_0_i2_0_i3_0_fused // 14 * 32 + i0_2_i1_2_i2_2_i3_2_fused // 2 * 8 + ax1)
                            v2 = Ts.axis.spatial(56, i0_0_i1_0_i2_0_i3_0_fused % 14 // 2 * 8 + i0_1_i1_1_i2_1_i3_1_fused * 4 + i0_2_i1_2_i2_2_i3_2_fused % 2 * 2 + ax2)
                            v3 = Ts.axis.spatial(56, i0_0_i1_0_i2_0_i3_0_fused % 2 * 28 + ax3)
                            compute[v0, v1, v2, v3] = T.max((compute_local[v0, v1, v2, v3] + B[v1, 0, 0]) * bn_scale[v1, 0, 0] + bn_offset[v1, 0, 0], T.float32(0))

@tvm_tirx.script.ir_module
class SoftmaxBeforeInline:
    @Ts.function
    def main(A: T.Tensor((256, 256), "float32"), T_softmax_norm: T.Tensor((256, 256), "float32")) -> None:
        T_softmax_maxelem = Ts.sblock_alloc_tensor([256], dtype="float32")
        T_softmax_exp = Ts.sblock_alloc_tensor([256, 256], dtype="float32")
        T_softmax_expsum = Ts.sblock_alloc_tensor([256], dtype="float32")
        for i0, i1 in T.grid(256, 256):
            with Ts.sblock("T_softmax_maxelem"):
                i0_1, k = Ts.axis.remap("SR", [i0, i1])
                with Ts.init():
                    T_softmax_maxelem[i0_1] = T.min_value(ty="float32")
                T_softmax_maxelem[i0_1] = T.max(T_softmax_maxelem[i0_1], A[i0_1, k])
        for i0, i1 in T.grid(256, 256):
            with Ts.sblock("T_softmax_exp"):
                i0_2, i1_1 = Ts.axis.remap("SS", [i0, i1])
                T_softmax_exp[i0_2, i1_1] = T.exp(A[i0_2, i1_1] - T_softmax_maxelem[i0_2])
        for i0_3, i1 in T.grid(256, 256):
            with Ts.sblock("T_softmax_expsum"):
                i0_4, k = Ts.axis.remap("SR", [i0_3, i1])
                with Ts.init():
                    T_softmax_expsum[i0_4] = T.float32(0)
                T_softmax_expsum[i0_4] = T_softmax_expsum[i0_4] + T_softmax_exp[i0_4, k]
        for i0_5, i1 in T.grid(256, 256):
            with Ts.sblock("T_softmax_norm"):
                i0_6, i1_2 = Ts.axis.remap("SS", [i0_5, i1])
                T_softmax_norm[i0_6, i1_2] = T_softmax_exp[i0_6, i1_2] / T_softmax_expsum[i0_6]

@tvm_tirx.script.ir_module
class SoftmaxAfterInline:
    @Ts.function
    def main(A: T.Tensor((256, 256), "float32"), T_softmax_norm: T.Tensor((256, 256), "float32")) -> None:
        T_softmax_maxelem = Ts.sblock_alloc_tensor([256], dtype="float32")
        T_softmax_expsum = Ts.sblock_alloc_tensor([256], dtype="float32")
        for i0, i1 in T.grid(256, 256):
            with Ts.sblock("T_softmax_maxelem"):
                i0_1, k = Ts.axis.remap("SR", [i0, i1])
                with Ts.init():
                    T_softmax_maxelem[i0_1] = T.min_value(ty="float32")
                T_softmax_maxelem[i0_1] = T.max(T_softmax_maxelem[i0_1], A[i0_1, k])
        for i0, i1 in T.grid(256, 256):
            with Ts.sblock("T_softmax_expsum"):
                i0_2, k = Ts.axis.remap("SR", [i0, i1])
                with Ts.init():
                    T_softmax_expsum[i0_2] = T.float32(0)
                T_softmax_expsum[i0_2] = T_softmax_expsum[i0_2] + T.exp(A[i0_2, k] - T_softmax_maxelem[i0_2])
        for i0_3, i1 in T.grid(256, 256):
            with Ts.sblock("T_softmax_norm"):
                i0_4, i1_1 = Ts.axis.remap("SS", [i0_3, i1])
                T_softmax_norm[i0_4, i1_1] = T.exp(A[i0_4, i1_1] - T_softmax_maxelem[i0_4]) / T_softmax_expsum[i0_4]

@tvm_tirx.script.ir_module
class BeforePureSpatial:
    @Ts.function
    def main(
        placeholder: T.Tensor((1, 384), "int64"),
        placeholder_1: T.Tensor((30522, 768), "float32"),
        placeholder_2: T.Tensor((1, 384, 768), "float32"),
        T_add: T.Tensor((1, 384, 768), "float32"),
    ) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        compile_engine_const = Ts.sblock_alloc_tensor([], dtype="int64")
        T_less = Ts.sblock_alloc_tensor([1, 384], dtype="bool")
        compile_engine_const_1 = Ts.sblock_alloc_tensor([], dtype="int64")
        T_add_1 = Ts.sblock_alloc_tensor([1, 384], dtype="int64")
        T_where = Ts.sblock_alloc_tensor([1, 384], dtype="int64")
        T_take = Ts.sblock_alloc_tensor([1, 384, 768], dtype="float32")
        with Ts.sblock("compile_engine_const"):
            vi = Ts.axis.spatial(1, 0)
            Ts.reads()
            Ts.writes(compile_engine_const[()])
            compile_engine_const[()] = T.int64(0)
        for i0, i1 in T.grid(1, 384):
            with Ts.sblock("T_less"):
                ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                Ts.reads(placeholder[ax0, ax1], compile_engine_const[()])
                Ts.writes(T_less[ax0, ax1])
                T_less[ax0, ax1] = placeholder[ax0, ax1] < compile_engine_const[()]
        with Ts.sblock("compile_engine_const_1"):
            vi = Ts.axis.spatial(1, 0)
            Ts.reads()
            Ts.writes(compile_engine_const_1[()])
            compile_engine_const_1[()] = T.int64(30522)
        for i0, i1 in T.grid(1, 384):
            with Ts.sblock("T_add"):
                ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                Ts.reads(placeholder[ax0, ax1], compile_engine_const_1[()])
                Ts.writes(T_add_1[ax0, ax1])
                T_add_1[ax0, ax1] = placeholder[ax0, ax1] + compile_engine_const_1[()]
        for i0, i1 in T.grid(1, 384):
            with Ts.sblock("T_where"):
                ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                Ts.reads(T_less[ax0, ax1], T_add_1[ax0, ax1], placeholder[ax0, ax1])
                Ts.writes(T_where[ax0, ax1])
                T_where[ax0, ax1] = T.Select(
                    T.cast(T_less[ax0, ax1], ty="int32") != 0, T_add_1[ax0, ax1], placeholder[ax0, ax1]
                )
        for i0, i1, i2 in T.grid(1, 384, 768):
            with Ts.sblock("T_take"):
                ax0, ax1, ax2 = Ts.axis.remap("SSS", [i0, i1, i2])
                Ts.reads(
                    placeholder_1[T.min(T.max(T.int64(0), T_where[ax0, ax1]), T.int64(30521)), ax2],
                    T_where[ax0, ax1],
                )
                Ts.writes(T_take[ax0, ax1, ax2])
                T_take[ax0, ax1, ax2] = placeholder_1[
                    T.min(T.max(T.int64(0), T_where[ax0, ax1]), T.int64(30521)), ax2
                ]
        for i0, i1, i2 in T.grid(1, 384, 768):
            with Ts.sblock("T_add_1"):
                ax0, ax1, ax2 = Ts.axis.remap("SSS", [i0, i1, i2])
                Ts.reads(T_take[ax0, ax1, ax2], placeholder_2[ax0, ax1, ax2])
                Ts.writes(T_add[ax0, ax1, ax2])
                T_add[ax0, ax1, ax2] = T_take[ax0, ax1, ax2] + placeholder_2[ax0, ax1, ax2]

@tvm_tirx.script.ir_module
class AfterPureSpatial:
    @Ts.function
    def main(placeholder: T.Tensor((1, 384), "int64"), placeholder_1: T.Tensor((30522, 768), "float32"), placeholder_2: T.Tensor((1, 384, 768), "float32"), T_add: T.Tensor((1, 384, 768), "float32")) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        # body
        # with Ts.sblock("root")
        for i0, i1, i2 in T.grid(1, 384, 768):
            with Ts.sblock("T_add_1"):
                ax0, ax1, ax2 = Ts.axis.remap("SSS", [i0, i1, i2])
                Ts.reads(placeholder[ax0, ax1], placeholder_1[T.min(T.max(T.int64(0), placeholder[ax0, ax1]), T.int64(30521)) : T.min(T.max(T.int64(0), placeholder[ax0, ax1] + T.int64(30522)), T.int64(30521)) + T.int64(1), ax2], placeholder_2[ax0, ax1, ax2])
                Ts.writes(T_add[ax0, ax1, ax2])
                T_add[ax0, ax1, ax2] = placeholder_1[T.min(T.max(T.int64(0), T.Select(T.cast(placeholder[ax0, ax1] < T.int64(0), ty="int32") != 0, placeholder[ax0, ax1] + T.int64(30522), placeholder[ax0, ax1])), T.int64(30521)), ax2] + placeholder_2[ax0, ax1, ax2]

@tvm_tirx.script.ir_module
class ConstConsumer:
    @Ts.function
    def main(T_full: T.Tensor((1, 12, 4096), "int64")) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        # body
        # with Ts.sblock("root")
        for i0, i1, i2 in T.grid(1, 12, 4096):
            with Ts.sblock("T_full"):
                ax0, ax1, ax2 = Ts.axis.remap("SSS", [i0, i1, i2])
                Ts.reads()
                Ts.writes(T_full[ax0, ax1, ax2])
                T_full[ax0, ax1, ax2] = T.int64(0)

@tvm_tirx.script.ir_module
class Conv2dInt8:
    @Ts.function
    def main(p0: T.Tensor((16, 14, 14, 256), "int8"), p1: T.Tensor((1024, 1, 1, 256), "int8"), p2: T.Tensor((1, 1, 1, 1024), "int32"), p3: T.Tensor((1, 1, 1, 1024), "int32"), p4: T.Tensor(1024, "int32"), p5: T.Tensor(1024, "int32"), p6: T.Tensor(1024, "int32"), p7: T.Tensor(1, "int32"), p8: T.Tensor((16, 14, 14, 1024), "int32"), compute: T.Tensor((16, 14, 14, 1024), "int32")) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        # body
        # with Ts.sblock("root")
        compile_engine_const = Ts.sblock_alloc_tensor([], dtype="int32")
        pad_temp = Ts.sblock_alloc_tensor([16, 14, 14, 256], dtype="int8")
        conv2d_nhwc = Ts.sblock_alloc_tensor([16, 14, 14, 1024], dtype="int32")
        T_subtract = Ts.sblock_alloc_tensor([16, 14, 14, 1024], dtype="int32")
        T_add = Ts.sblock_alloc_tensor([16, 14, 14, 1024], dtype="int32")
        compute_1 = Ts.sblock_alloc_tensor([16, 14, 14, 1024], dtype="int32")
        T_add_1 = Ts.sblock_alloc_tensor([16, 14, 14, 1024], dtype="int32")
        compute_2 = Ts.sblock_alloc_tensor([16, 14, 14, 1024], dtype="int32")
        T_subtract_1 = Ts.sblock_alloc_tensor([16, 14, 14, 1024], dtype="int32")
        compute_3 = Ts.sblock_alloc_tensor([16, 14, 14, 1024], dtype="int32")
        T_add_2 = Ts.sblock_alloc_tensor([16, 14, 14, 1024], dtype="int32")
        with Ts.sblock("compile_engine_const"):
            vi = Ts.axis.spatial(1, 0)
            Ts.reads()
            Ts.writes(compile_engine_const[()])
            compile_engine_const[()] = 59
        for i0, i1, i2, i3 in T.grid(16, 14, 14, 256):
            with Ts.sblock("pad_temp"):
                i0_1, i1_1, i2_1, i3_1 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(p0[i0_1, i1_1, i2_1, i3_1])
                Ts.writes(pad_temp[i0_1, i1_1, i2_1, i3_1])
                pad_temp[i0_1, i1_1, i2_1, i3_1] = p0[i0_1, i1_1, i2_1, i3_1]
        for i0, i1, i2, i3, i4, i5, i6 in T.grid(16, 14, 14, 1024, 1, 1, 256):
            with Ts.sblock("conv2d_nhwc"):
                nn, yy, xx, ff, ry, rx, rc = Ts.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
                Ts.reads(pad_temp[nn, yy + ry, xx + rx, rc], p1[ff, ry, rx, rc])
                Ts.writes(conv2d_nhwc[nn, yy, xx, ff])
                with Ts.init():
                    conv2d_nhwc[nn, yy, xx, ff] = 0
                conv2d_nhwc[nn, yy, xx, ff] = conv2d_nhwc[nn, yy, xx, ff] + T.cast(pad_temp[nn, yy + ry, xx + rx, rc], ty="int32") * T.cast(p1[ff, ry, rx, rc], ty="int32")
        for i0, i1, i2, i3 in T.grid(16, 14, 14, 1024):
            with Ts.sblock("T_subtract"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(conv2d_nhwc[ax0, ax1, ax2, ax3], p2[0, 0, 0, ax3])
                Ts.writes(T_subtract[ax0, ax1, ax2, ax3])
                T_subtract[ax0, ax1, ax2, ax3] = conv2d_nhwc[ax0, ax1, ax2, ax3] - p2[0, 0, 0, ax3]
        for i0, i1, i2, i3 in T.grid(16, 14, 14, 1024):
            with Ts.sblock("T_add"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(T_subtract[ax0, ax1, ax2, ax3], p3[0, 0, 0, ax3])
                Ts.writes(T_add[ax0, ax1, ax2, ax3])
                T_add[ax0, ax1, ax2, ax3] = T_subtract[ax0, ax1, ax2, ax3] + p3[0, 0, 0, ax3]
        for i0, i1, i2, i3 in T.grid(16, 14, 14, 1024):
            with Ts.sblock("compute"):
                i0_2, i1_2, i2_2, i3_2 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(T_add[i0_2, i1_2, i2_2, i3_2], p4[i3_2], p5[i3_2], p6[i3_2])
                Ts.writes(compute_1[i0_2, i1_2, i2_2, i3_2])
                compute_1[i0_2, i1_2, i2_2, i3_2] = T.Cast("int32", T.shift_right(T.shift_left(T.Cast("int64", T_add[i0_2, i1_2, i2_2, i3_2]) * T.Cast("int64", p4[i3_2]), p5[i3_2]), (31) + (p6[i3_2])))
        for i0_3, i1_3, i2_3, i3_3 in T.grid(16, 14, 14, 1024):
            with Ts.sblock("T_add_1"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0_3, i1_3, i2_3, i3_3])
                Ts.reads(compile_engine_const[()], compute_1[ax0, ax1, ax2, ax3])
                Ts.writes(T_add_1[ax0, ax1, ax2, ax3])
                T_add_1[ax0, ax1, ax2, ax3] = compile_engine_const[()] + compute_1[ax0, ax1, ax2, ax3]
        for i0_4, i1_4, i2_4, i3_4 in T.grid(16, 14, 14, 1024):
            with Ts.sblock("compute_1"):
                i0_5, i1_5, i2_5, i3_5 = Ts.axis.remap("SSSS", [i0_4, i1_4, i2_4, i3_4])
                Ts.reads(T_add_1[i0_5, i1_5, i2_5, i3_5])
                Ts.writes(compute_2[i0_5, i1_5, i2_5, i3_5])
                compute_2[i0_5, i1_5, i2_5, i3_5] = T.max(T.min(T_add_1[i0_5, i1_5, i2_5, i3_5], 255), 0)
        for i0_6, i1_6, i2_6, i3_6 in T.grid(16, 14, 14, 1024):
            with Ts.sblock("T_subtract_1"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0_6, i1_6, i2_6, i3_6])
                Ts.reads(compute_2[ax0, ax1, ax2, ax3], p7[0])
                Ts.writes(T_subtract_1[ax0, ax1, ax2, ax3])
                T_subtract_1[ax0, ax1, ax2, ax3] = compute_2[ax0, ax1, ax2, ax3] - p7[0]
        for i0_7, i1_7, i2_7, i3_7 in T.grid(16, 14, 14, 1024):
            with Ts.sblock("compute_2"):
                i0_8, i1_8, i2_8, i3_8 = Ts.axis.remap("SSSS", [i0_7, i1_7, i2_7, i3_7])
                Ts.reads(T_subtract_1[i0_8, i1_8, i2_8, i3_8])
                Ts.writes(compute_3[i0_8, i1_8, i2_8, i3_8])
                compute_3[i0_8, i1_8, i2_8, i3_8] = T.Cast("int32", T.shift_right(T.Cast("int64", T_subtract_1[i0_8, i1_8, i2_8, i3_8]) * T.Cast("int64", 1408572815), (31) - (1)))
        for i0_9, i1_9, i2_9, i3_9 in T.grid(16, 14, 14, 1024):
            with Ts.sblock("T_add_2"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0_9, i1_9, i2_9, i3_9])
                Ts.reads(compute_3[ax0, ax1, ax2, ax3], p8[ax0, ax1, ax2, ax3])
                Ts.writes(T_add_2[ax0, ax1, ax2, ax3])
                T_add_2[ax0, ax1, ax2, ax3] = compute_3[ax0, ax1, ax2, ax3] + p8[ax0, ax1, ax2, ax3]
        for i0_10, i1_10, i2_10, i3_10 in T.grid(16, 14, 14, 1024):
            with Ts.sblock("compute_3"):
                i0_11, i1_11, i2_11, i3_11 = Ts.axis.remap("SSSS", [i0_10, i1_10, i2_10, i3_10])
                Ts.reads(T_add_2[i0_11, i1_11, i2_11, i3_11])
                Ts.writes(compute[i0_11, i1_11, i2_11, i3_11])
                compute[i0_11, i1_11, i2_11, i3_11] = T.max(T.min(T_add_2[i0_11, i1_11, i2_11, i3_11], 255), 0)

# pylint: enable=no-member,invalid-name,unused-variable,no-self-argument,line-too-long,chained-comparison,not-callable,too-many-nested-blocks
# fmt: on


def test_inline_consumer_chain():
    mod = Conv2DBiasBnReLU
    target = Target("llvm")
    (space,) = generate_design_space(
        kind="llvm",
        mod=mod,
        target=target,
        types=ms.schedule_rule.AutoInline,
    )
    tvm_tirx.ir.assert_structural_equal(lhs=space.mod, rhs=Conv2DBiasBnReLUInlined)


def test_inline_into_cache():
    mod = MultiLevelTiledConv2D
    target = Target("cuda", host="llvm")
    (space,) = generate_design_space(
        kind="cuda",
        mod=mod,
        target=target,
        types=ms.schedule_rule.AutoInline,
    )
    tvm_tirx.ir.assert_structural_equal(lhs=space.mod, rhs=MultiLevelTiledConv2DAfterInline)


def test_inline_into_multiple_consumers():
    mod = SoftmaxBeforeInline
    target = Target("cuda", host="llvm")
    (space,) = generate_design_space(
        kind="cuda",
        mod=mod,
        target=target,
        types=ms.schedule_rule.AutoInline,
    )
    tvm_tirx.ir.assert_structural_equal(lhs=space.mod, rhs=SoftmaxAfterInline)


def test_inline_pure_spatial():
    mod = BeforePureSpatial
    target = Target("llvm")
    (space,) = generate_design_space(
        kind="llvm",
        mod=mod,
        target=target,
        types=ms.schedule_rule.AutoInline,
    )
    tvm_tirx.ir.assert_structural_equal(lhs=space.mod, rhs=AfterPureSpatial)


def test_inline_constant_tensor():
    mod = ConstConsumer
    target = Target("cuda", host="llvm")
    (space,) = generate_design_space(
        kind="cuda",
        mod=mod,
        target=target,
        types=ms.schedule_rule.AutoInline,
    )
    tvm_tirx.ir.assert_structural_equal(lhs=space.mod, rhs=ConstConsumer)


def test_conv2d_int8_inline_constant_scalars():
    sch = Schedule(Conv2dInt8)

    conv2d = sch.get_sblock("conv2d_nhwc")
    sch.cache_write(conv2d, 0, "shared")

    with pytest.raises(tvm.s_tir.ScheduleError) as e:
        sch.reverse_compute_inline(sch.get_sblock("T_add_1"))

    err_msg = "The block is only allowed to read a single tensor region, but it reads 2 region(s)"
    assert err_msg in str(e)

    ms.schedule_rule.InlineConstantScalars().apply(sch, sch.get_sblock("compile_engine_const"))
    sch.reverse_compute_inline(sch.get_sblock("T_add_1"))


def test_inline_constant_scalars_skip_output_block():
    # If the constant scalar block is an output block, it should not be inlined

    @tvm_tirx.script.ir_module
    class Full:
        @Ts.function
        def main(T_full: T.Tensor((), "float32")):
            with Ts.sblock("T_full"):
                vi = Ts.axis.spatial(1, 0)
                Ts.reads()
                Ts.writes(T_full[()])
                T_full[()] = T.float32(1)

    sch = Schedule(Full)
    sch = ms.schedule_rule.InlineConstantScalars().apply(sch, sch.get_sblock("T_full"))[0]
    assert_structural_equal(sch.mod, Full)


def test_no_inline_root_block():
    @tvm_tirx.script.ir_module
    class MaxReduction:
        @Ts.function
        def main(
            data: T.Tensor((8, 8), "float32"),
            data_red: T.Tensor((), "float32"),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            with Ts.sblock("data_red"):
                Ts.reads(data[0:8, 0:8])
                Ts.writes(data_red[()])
                with Ts.init():
                    data_red[()] = T.float32(-3.4e38)
                for i, j in T.grid(8, 8):
                    with Ts.sblock("update"):
                        Ts.reads(data_red[()], data[i, j])
                        Ts.writes(data_red[()])
                        data_red[()] = T.max(data_red[()], data[i, j])

    target = Target("llvm")
    (space,) = generate_design_space(
        kind="llvm",
        mod=MaxReduction,
        target=target,
        types=ms.schedule_rule.AutoInline,
    )
    tvm_tirx.ir.assert_structural_equal(lhs=space.mod, rhs=MaxReduction)


if __name__ == "__main__":
    tvm_tirx.testing.main()
