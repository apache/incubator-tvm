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
# ruff: noqa: E501, F821, F841

import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm.relax import script as R
from tvm.relax.transform import LegalizeOps
from tvm.s_tir import script as Ts
from tvm_tirx.ir import TupleType as _TupleType
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T

##################### Neural network #####################


def test_conv1d():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Conv1d:
        @R.function
        def main(x: R.Tensor((2, 128, 28), "float32"), w: R.Tensor((64, 16, 3), "float32")) -> R.Tensor((2, 64, 13), "float32"):
            gv: R.Tensor((2, 64, 13), "float32") = R.nn.conv1d(x, w, strides=(2,), padding=(1,), dilation=(2,), groups=8)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 128, 28), dtype="float32"), w: R.Tensor((64, 16, 3), dtype="float32")) -> R.Tensor((2, 64, 13), dtype="float32"):
            gv = R.call_tir(Expected.conv1d, (x, w), ty_args=[R.Tensor((2, 64, 13), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def conv1d(A: T.Tensor((T.int64(2), T.int64(128), T.int64(28)), "float32"), B: T.Tensor((T.int64(64), T.int64(16), T.int64(3)), "float32"), group_conv1d_ncw: T.Tensor((T.int64(2), T.int64(64), T.int64(13)), "float32")):
            T.func_attr({"tirx.noalias": True})
            pad_temp = Ts.sblock_alloc_tensor((T.int64(2), T.int64(128), T.int64(30)))
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(128), T.int64(30)):
                with Ts.sblock("pad_temp"):
                    v_i0, v_i1, v_i2 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(A[v_i0, v_i1, v_i2 - T.int64(1)])
                    Ts.writes(pad_temp[v_i0, v_i1, v_i2])
                    pad_temp[v_i0, v_i1, v_i2] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(29), A[v_i0, v_i1, v_i2 - T.int64(1)], T.float32(0))
            for nn, ff, yy, rc, ry in T.grid(T.int64(2), T.int64(64), T.int64(13), T.int64(16), T.int64(3)):
                with Ts.sblock("group_conv1d_ncw"):
                    v_nn, v_ff, v_yy, v_rc, v_ry = Ts.axis.remap("SSSRR", [nn, ff, yy, rc, ry])
                    Ts.reads(pad_temp[v_nn, v_ff // T.int64(8) * T.int64(16) + v_rc, v_yy * T.int64(2) + v_ry * T.int64(2)], B[v_ff, v_rc, v_ry])
                    Ts.writes(group_conv1d_ncw[v_nn, v_ff, v_yy])
                    with Ts.init():
                        group_conv1d_ncw[v_nn, v_ff, v_yy] = T.float32(0)
                    group_conv1d_ncw[v_nn, v_ff, v_yy] = group_conv1d_ncw[v_nn, v_ff, v_yy] + pad_temp[v_nn, v_ff // T.int64(8) * T.int64(16) + v_rc, v_yy * T.int64(2) + v_ry * T.int64(2)] * B[v_ff, v_rc, v_ry]
    # fmt: on

    mod = LegalizeOps()(Conv1d)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_conv1d_with_out_dtype():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Conv1d:
        @R.function
        def main(x: R.Tensor((2, 3, 28), "float32"), w: R.Tensor((4, 3, 3), "float32")) -> R.Tensor((2, 4, 26), "float16"):
            gv: R.Tensor((2, 4, 26), "float16") = R.nn.conv1d(x, w, out_dtype="float16")
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 28), dtype="float32"), w: R.Tensor((4, 3, 3), dtype="float32")) -> R.Tensor((2, 4, 26), dtype="float16"):
            gv = R.call_tir(Expected.conv1d, (x, w), ty_args=[R.Tensor((2, 4, 26), dtype="float16")])
            return gv

        @Ts.function(private=True)
        def conv1d(rxplaceholder: T.Tensor((T.int64(2), T.int64(3), T.int64(28)), "float32"), rxplaceholder_1: T.Tensor((T.int64(4), T.int64(3), T.int64(3)), "float32"), conv1d_ncw: T.Tensor((T.int64(2), T.int64(4), T.int64(26)), "float16")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            pad_temp = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3), T.int64(28)))
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(3), T.int64(28)):
                with Ts.sblock("pad_temp"):
                    v_i0, v_i1, v_i2 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(rxplaceholder[v_i0, v_i1, v_i2])
                    Ts.writes(pad_temp[v_i0, v_i1, v_i2])
                    pad_temp[v_i0, v_i1, v_i2] = rxplaceholder[v_i0, v_i1, v_i2]
            for nn, ff, yy, rc, ry in T.grid(T.int64(2), T.int64(4), T.int64(26), T.int64(3), T.int64(3)):
                with Ts.sblock("conv1d_ncw"):
                    v_nn, v_ff, v_yy, v_rc, v_ry = Ts.axis.remap("SSSRR", [nn, ff, yy, rc, ry])
                    Ts.reads(pad_temp[v_nn, v_rc, v_yy + v_ry], rxplaceholder_1[v_ff, v_rc, v_ry])
                    Ts.writes(conv1d_ncw[v_nn, v_ff, v_yy])
                    with Ts.init():
                        conv1d_ncw[v_nn, v_ff, v_yy] = T.float16(0)
                    conv1d_ncw[v_nn, v_ff, v_yy] = conv1d_ncw[v_nn, v_ff, v_yy] + T.Cast("float16", pad_temp[v_nn, v_rc, v_yy + v_ry]) * T.Cast("float16", rxplaceholder_1[v_ff, v_rc, v_ry])
    # fmt: on

    mod = LegalizeOps()(Conv1d)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_conv1d_nwc():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Conv1d:
        @R.function
        def main(x: R.Tensor((2, 28, 128), "float32"), w: R.Tensor((64, 128, 3), "float32")) -> R.Tensor((2, 26, 64), "float32"):
            gv: R.Tensor((2, 26, 64), "float32") = R.nn.conv1d(x, w, data_layout="NWC")
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 28, 128), dtype="float32"), w: R.Tensor((64, 128, 3), dtype="float32")) -> R.Tensor((2, 26, 64), dtype="float32"):
            gv = R.call_tir(Expected.conv1d, (x, w), ty_args=[R.Tensor((2, 26, 64), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def conv1d(rxplaceholder: T.Tensor((T.int64(2), T.int64(28), T.int64(128)), "float32"), rxplaceholder_1: T.Tensor((T.int64(64), T.int64(128), T.int64(3)), "float32"), conv1d_nwc: T.Tensor((T.int64(2), T.int64(26), T.int64(64)), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            pad_temp = Ts.sblock_alloc_tensor((T.int64(2), T.int64(28), T.int64(128)))
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(28), T.int64(128)):
                with Ts.sblock("pad_temp"):
                    v_i0, v_i1, v_i2 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(rxplaceholder[v_i0, v_i1, v_i2])
                    Ts.writes(pad_temp[v_i0, v_i1, v_i2])
                    pad_temp[v_i0, v_i1, v_i2] = rxplaceholder[v_i0, v_i1, v_i2]
            for nn, yy, ff, ry, rc in T.grid(T.int64(2), T.int64(26), T.int64(64), T.int64(3), T.int64(128)):
                with Ts.sblock("conv1d_nwc"):
                    v_nn, v_yy, v_ff, v_ry, v_rc = Ts.axis.remap("SSSRR", [nn, yy, ff, ry, rc])
                    Ts.reads(pad_temp[v_nn, v_yy + v_ry, v_rc], rxplaceholder_1[v_ff, v_rc, v_ry])
                    Ts.writes(conv1d_nwc[v_nn, v_yy, v_ff])
                    with Ts.init():
                        conv1d_nwc[v_nn, v_yy, v_ff] = T.float32(0)
                    conv1d_nwc[v_nn, v_yy, v_ff] = conv1d_nwc[v_nn, v_yy, v_ff] + pad_temp[v_nn, v_yy + v_ry, v_rc] * rxplaceholder_1[v_ff, v_rc, v_ry]
    # fmt: on

    mod = LegalizeOps()(Conv1d)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_conv1d_symbolic():
    # fmt: off
    n = T.dynamic("n", "int64")
    w = T.dynamic("w", "int64")
    f = T.dynamic("f", "int64")
    kw = T.dynamic("kw", "int64")
    c = T.dynamic("c", "int64")

    @tvm_tirx.script.ir_module
    class Conv1d:
        @R.function
        def main(x: R.Tensor((n, c, w), "float32"), kernel: R.Tensor((f, c, kw), "float32")) -> R.Tensor((n, f, w - kw + 1), "float32"):
            gv: R.Tensor((n, f, w - kw + 1), "float32") = R.nn.conv1d(x, kernel)
            return gv

    n_main = T.dynamic("n", "int64")
    f_main = T.dynamic("f", "int64")
    w_main = T.dynamic("w", "int64")
    kw_main = T.dynamic("kw", "int64")
    c_main = T.dynamic("c", "int64")
    n_conv1d = T.dynamic("n", "int64")
    c_conv1d = T.dynamic("c", "int64")
    w_conv1d = T.dynamic("w", "int64")
    f_conv1d = T.dynamic("f", "int64")
    kw_conv1d = T.dynamic("kw", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((n_main, c_main, w_main), dtype="float32"), kernel: R.Tensor((f_main, c_main, kw_main), dtype="float32")) -> R.Tensor((n_main, f_main, w_main - kw_main + 1), dtype="float32"):
            gv = R.call_tir(Expected.conv1d, (x, kernel), ty_args=[R.Tensor((n_main, f_main, w_main + 1 - kw_main), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def conv1d(rxplaceholder: T.Tensor((n_conv1d, c_conv1d, w_conv1d), "float32"), rxplaceholder_1: T.Tensor((f_conv1d, c_conv1d, kw_conv1d), "float32"), conv1d_ncw: T.Tensor((n_conv1d, f_conv1d, w_conv1d + T.int64(1) - kw_conv1d), "float32")):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            pad_temp = Ts.sblock_alloc_tensor((n_conv1d, c_conv1d, w_conv1d))
            for i0, i1, i2 in T.grid(n_conv1d, c_conv1d, w_conv1d):
                with Ts.sblock("pad_temp"):
                    v_i0, v_i1, v_i2 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(rxplaceholder[v_i0, v_i1, v_i2])
                    Ts.writes(pad_temp[v_i0, v_i1, v_i2])
                    pad_temp[v_i0, v_i1, v_i2] = rxplaceholder[v_i0, v_i1, v_i2]
            for nn, ff, yy, rc, ry in T.grid(n_conv1d, f_conv1d, w_conv1d + T.int64(1) - kw_conv1d, c_conv1d, kw_conv1d):
                with Ts.sblock("conv1d_ncw"):
                    v_nn, v_ff, v_yy, v_rc, v_ry = Ts.axis.remap("SSSRR", [nn, ff, yy, rc, ry])
                    Ts.reads(pad_temp[v_nn, v_rc, v_yy + v_ry], rxplaceholder_1[v_ff, v_rc, v_ry])
                    Ts.writes(conv1d_ncw[v_nn, v_ff, v_yy])
                    with Ts.init():
                        conv1d_ncw[v_nn, v_ff, v_yy] = T.float32(0)
                    conv1d_ncw[v_nn, v_ff, v_yy] = conv1d_ncw[v_nn, v_ff, v_yy] + pad_temp[v_nn, v_rc, v_yy + v_ry] * rxplaceholder_1[v_ff, v_rc, v_ry]
    # fmt: on

    mod = LegalizeOps()(Conv1d)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_conv1d_transpose():
    # fmt: off
    @I.ir_module
    class Conv1dTranspose:
        @R.function
        def main(x: R.Tensor((2, 128, 28), "float32"), w: R.Tensor((128, 16, 3), "float32")):
            gv = R.nn.conv1d_transpose(x, w, strides=2, padding=1, dilation=1, output_padding=1, groups=8)
            return gv

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def conv1d_transpose(x: T.Tensor((T.int64(2), T.int64(128), T.int64(28)), "float32"), w: T.Tensor((T.int64(128), T.int64(16), T.int64(3)), "float32"), compute: T.Tensor((T.int64(2), T.int64(128), T.int64(56)), "float32")):
            T.func_attr({"tirx.noalias": True})
            data_dilate = Ts.sblock_alloc_tensor((T.int64(2), T.int64(128), T.int64(55)))
            data_pad = Ts.sblock_alloc_tensor((T.int64(2), T.int64(128), T.int64(58)))
            kernel = Ts.sblock_alloc_tensor((T.int64(16), T.int64(128), T.int64(3)))
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(128), T.int64(55)):
                with Ts.sblock("data_dilate"):
                    v_i0, v_i1, v_i2 = Ts.axis.remap("SSS", [i0, i1, i2])
                    data_dilate[v_i0, v_i1, v_i2] = T.if_then_else(v_i2 % T.int64(2) == T.int64(0), x[v_i0, v_i1, v_i2 // T.int64(2)], T.float32(0.0))
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(128), T.int64(58)):
                with Ts.sblock("data_pad"):
                    v_i0, v_i1, v_i2 = Ts.axis.remap("SSS", [i0, i1, i2])
                    data_pad[v_i0, v_i1, v_i2] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(56), data_dilate[v_i0, v_i1, v_i2 - T.int64(1)], T.float32(0.0))
            for o, i, w_1 in T.grid(T.int64(16), T.int64(128), T.int64(3)):
                with Ts.sblock("kernel"):
                    v_o, v_i, v_w = Ts.axis.remap("SSS", [o, i, w_1])
                    kernel[v_o, v_i, v_w] = w[v_i, v_o, T.int64(2) - v_w]
            for b_index, c_index, w_1, dc, dw in T.grid(T.int64(2), T.int64(128), T.int64(56), T.int64(16), T.int64(3)):
                with Ts.sblock("compute"):
                    v_b, v_c, v_w, v_dc, v_dw = Ts.axis.remap("SSSRR", [b_index, c_index, w_1, dc, dw])
                    with Ts.init():
                        compute[v_b, v_c, v_w] = T.float32(0.0)
                    compute[v_b, v_c, v_w] = compute[v_b, v_c, v_w] + data_pad[v_b, v_c // T.int64(16) * T.int64(16) + v_dc, v_w + v_dw] * kernel[v_c % T.int64(16), v_c // T.int64(16) * T.int64(16) + v_dc, v_dw]

        @R.function
        def main(x: R.Tensor((2, 128, 28), dtype="float32"), w: R.Tensor((128, 16, 3), dtype="float32")) -> R.Tensor((2, 128, 56), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.conv1d_transpose, (x, w), ty_args=[R.Tensor((2, 128, 56), dtype="float32")])
            return gv
    # fmt: on

    mod = LegalizeOps()(Conv1dTranspose)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_conv2d():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Conv2d:
        @R.function
        def main(x: R.Tensor((2, 128, 28, 28), "float32"), w: R.Tensor((64, 16, 3, 3), "float32")) -> R.Tensor((2, 64, 13, 13), "float32"):
            gv: R.Tensor((2, 64, 13, 13), "float32") = R.nn.conv2d(x, w, strides=(2, 2), padding=(1, 1), dilation=(2, 2), groups=8)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 128, 28, 28), "float32"), w: R.Tensor((64, 16, 3, 3), "float32")) -> R.Tensor((2, 64, 13, 13), "float32"):
            gv = R.call_tir(Expected.conv2d, (x, w), ty_args=[R.Tensor((2, 64, 13, 13), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def conv2d(rxplaceholder: T.Tensor((T.int64(2), T.int64(128), T.int64(28), T.int64(28)), "float32"), rxplaceholder_1: T.Tensor((T.int64(64), T.int64(16), T.int64(3), T.int64(3)), "float32"), group_conv2d_nchw: T.Tensor((T.int64(2), T.int64(64), T.int64(13), T.int64(13)), "float32")):
            T.func_attr({"tirx.noalias": True})
            pad_temp = Ts.sblock_alloc_tensor([T.int64(2), T.int64(128), T.int64(30), T.int64(30)], dtype="float32")
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(128), T.int64(30), T.int64(30)):
                with Ts.sblock("pad_temp"):
                    i0_1, i1_1, i2_1, i3_1 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[i0_1, i1_1, i2_1 - T.int64(1), i3_1 - T.int64(1)])
                    Ts.writes(pad_temp[i0_1, i1_1, i2_1, i3_1])
                    pad_temp[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(T.int64(1) <= i2_1 and i2_1 < T.int64(29) and T.int64(1) <= i3_1 and i3_1 < T.int64(29), rxplaceholder[i0_1, i1_1, i2_1 - T.int64(1), i3_1 - T.int64(1)], T.float32(0))
            for i0, i1, i2, i3, i4, i5, i6 in T.grid(T.int64(2), T.int64(64), T.int64(13), T.int64(13), T.int64(16), T.int64(3), T.int64(3)):
                with Ts.sblock("group_conv2d_nchw"):
                    nn, ff, yy, xx, rc, ry, rx = Ts.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
                    Ts.reads(pad_temp[nn, ff // T.int64(8) * T.int64(16) + rc, yy * T.int64(2) + ry * T.int64(2), xx * T.int64(2) + rx * T.int64(2)], rxplaceholder_1[ff, rc, ry, rx])
                    Ts.writes(group_conv2d_nchw[nn, ff, yy, xx])
                    with Ts.init():
                        group_conv2d_nchw[nn, ff, yy, xx] = T.float32(0)
                    group_conv2d_nchw[nn, ff, yy, xx] = group_conv2d_nchw[nn, ff, yy, xx] + pad_temp[nn, ff // T.int64(8) * T.int64(16) + rc, yy * T.int64(2) + ry * T.int64(2), xx * T.int64(2) + rx * T.int64(2)] * rxplaceholder_1[ff, rc, ry, rx]
    # fmt: on

    mod = LegalizeOps()(Conv2d)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_conv2d_with_out_dtype():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Conv2d:
        @R.function
        def main(x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")) -> R.Tensor((2, 4, 26, 26), "float16"):
            gv: R.Tensor((2, 4, 26, 26), "float16") = R.nn.conv2d(x, w, out_dtype="float16")
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")) -> R.Tensor((2, 4, 26, 26), "float16"):
            gv = R.call_tir(Expected.conv2d, (x, w), ty_args=[R.Tensor((2, 4, 26, 26), dtype="float16")])
            return gv

        @Ts.function(private=True)
        def conv2d(rxplaceholder: T.Tensor((T.int64(2), T.int64(3), T.int64(28), T.int64(28)), "float32"), rxplaceholder_1: T.Tensor((T.int64(4), T.int64(3), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Tensor((T.int64(2), T.int64(4), T.int64(26), T.int64(26)), "float16")):
            T.func_attr({"tirx.noalias": True})
            pad_temp = Ts.sblock_alloc_tensor([T.int64(2), T.int64(3), T.int64(28), T.int64(28)], dtype="float32")
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(28), T.int64(28)):
                with Ts.sblock("pad_temp"):
                    i0_1, i1_1, i2_1, i3_1 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[i0_1, i1_1, i2_1, i3_1])
                    Ts.writes(pad_temp[i0_1, i1_1, i2_1, i3_1])
                    pad_temp[i0_1, i1_1, i2_1, i3_1] = rxplaceholder[i0_1, i1_1, i2_1, i3_1]
            for i0, i1, i2, i3, i4, i5, i6 in T.grid(T.int64(2), T.int64(4), T.int64(26), T.int64(26), T.int64(3), T.int64(3), T.int64(3)):
                with Ts.sblock("conv2d_nchw"):
                    nn, ff, yy, xx, rc, ry, rx = Ts.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
                    Ts.reads(pad_temp[nn, rc, yy + ry, xx + rx], rxplaceholder_1[ff, rc, ry, rx])
                    Ts.writes(conv2d_nchw[nn, ff, yy, xx])
                    with Ts.init():
                        conv2d_nchw[nn, ff, yy, xx] = T.float16(0)
                    conv2d_nchw[nn, ff, yy, xx] = conv2d_nchw[nn, ff, yy, xx] + T.Cast("float16", pad_temp[nn, rc, yy + ry, xx + rx]) * T.Cast("float16", rxplaceholder_1[ff, rc, ry, rx])
    # fmt: on

    mod = LegalizeOps()(Conv2d)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_conv2d_nhwc():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Conv2d:
        @R.function
        def main(x: R.Tensor((2, 28, 28, 128), "float32"), w: R.Tensor((64, 128, 3, 3), "float32")) -> R.Tensor((2, 26, 26, 64), "float32"):
            gv: R.Tensor((2, 26, 26, 64), "float32") = R.nn.conv2d(x, w, data_layout="NHWC")
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 28, 28, 128), "float32"), w: R.Tensor((64, 128, 3, 3), "float32")) -> R.Tensor((2, 26, 26, 64), "float32"):
            gv = R.call_tir(Expected.conv2d, (x, w), ty_args=[R.Tensor((2, 26, 26, 64), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def conv2d(rxplaceholder: T.Tensor((T.int64(2), T.int64(28), T.int64(28), T.int64(128)), "float32"), rxplaceholder_1: T.Tensor((T.int64(64), T.int64(128), T.int64(3), T.int64(3)), "float32"), conv2d_nhwc: T.Tensor((T.int64(2), T.int64(26), T.int64(26), T.int64(64)), "float32")):
            T.func_attr({"tirx.noalias": True})
            pad_temp = Ts.sblock_alloc_tensor([T.int64(2), T.int64(28), T.int64(28), T.int64(128)], dtype="float32")
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(28), T.int64(28), T.int64(128)):
                with Ts.sblock("pad_temp"):
                    i0_1, i1_1, i2_1, i3_1 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[i0_1, i1_1, i2_1, i3_1])
                    Ts.writes(pad_temp[i0_1, i1_1, i2_1, i3_1])
                    pad_temp[i0_1, i1_1, i2_1, i3_1] = rxplaceholder[i0_1, i1_1, i2_1, i3_1]
            for i0, i1, i2, i3, i4, i5, i6 in T.grid(T.int64(2), T.int64(26), T.int64(26), T.int64(64), T.int64(3), T.int64(3), T.int64(128)):
                with Ts.sblock("conv2d_nhwc"):
                    nn, yy, xx, ff, ry, rx, rc = Ts.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
                    Ts.reads(pad_temp[nn, yy + ry, xx + rx, rc], rxplaceholder_1[ff, rc, ry, rx])
                    Ts.writes(conv2d_nhwc[nn, yy, xx, ff])
                    with Ts.init():
                        conv2d_nhwc[nn, yy, xx, ff] = T.float32(0)
                    conv2d_nhwc[nn, yy, xx, ff] = conv2d_nhwc[nn, yy, xx, ff] + pad_temp[nn, yy + ry, xx + rx, rc] * rxplaceholder_1[ff, rc, ry, rx]
    # fmt: on

    mod = LegalizeOps()(Conv2d)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_conv2d_symbolic():
    # fmt: off
    n = T.dynamic("n", "int64")
    h = T.dynamic("h", "int64")
    w = T.dynamic("w", "int64")
    f = T.dynamic("f", "int64")
    kh = T.dynamic("kh", "int64")
    kw = T.dynamic("kw", "int64")
    c = T.dynamic("c", "int64")

    @tvm_tirx.script.ir_module
    class Conv2d:
        @R.function
        def main(x: R.Tensor((n, c, h, w), "float32"), kernel: R.Tensor((f, c, kh, kw), "float32")) -> R.Tensor((n, f, h - kh + 1, w - kw + 1), "float32"):
            gv: R.Tensor((n, f, h - kh + 1, w - kw + 1), "float32") = R.nn.conv2d(x, kernel)
            return gv

    n_main = T.dynamic("n", "int64")
    f_main = T.dynamic("f", "int64")
    h_main = T.dynamic("h", "int64")
    kh_main = T.dynamic("kh", "int64")
    w_main = T.dynamic("w", "int64")
    kw_main = T.dynamic("kw", "int64")
    c_main = T.dynamic("c", "int64")
    c_conv2d = T.dynamic("c", "int64")
    f_conv2d = T.dynamic("f", "int64")
    h_conv2d = T.dynamic("h", "int64")
    kh_conv2d = T.dynamic("kh", "int64")
    kw_conv2d = T.dynamic("kw", "int64")
    n_conv2d = T.dynamic("n", "int64")
    w_conv2d = T.dynamic("w", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((n_main, c_main, h_main, w_main), "float32"), kernel: R.Tensor((f_main, c_main, kh_main, kw_main), "float32")) -> R.Tensor((n_main, f_main, h_main - kh_main + 1, w_main - kw_main + 1), "float32"):
            gv = R.call_tir(Expected.conv2d, (x, kernel), ty_args=[R.Tensor((n_main, f_main, h_main + 1 - kh_main, w_main + 1 - kw_main), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def conv2d(rxplaceholder: T.Tensor([n_conv2d, c_conv2d, h_conv2d, w_conv2d], dtype='float32'), rxplaceholder_1: T.Tensor([f_conv2d, c_conv2d, kh_conv2d, kw_conv2d], dtype='float32'), conv2d_nchw: T.Tensor([n_conv2d, f_conv2d, h_conv2d + T.int64(1) - kh_conv2d, w_conv2d + T.int64(1) - kw_conv2d], dtype='float32')):
            T.func_attr({"tirx.noalias": True})

            pad_temp = Ts.sblock_alloc_tensor([n_conv2d, c_conv2d, h_conv2d, w_conv2d], dtype="float32")
            for i0, i1, i2, i3 in T.grid(n_conv2d, c_conv2d, h_conv2d, w_conv2d):
                with Ts.sblock("pad_temp"):
                    i0_1, i1_1, i2_1, i3_1 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[i0_1, i1_1, i2_1, i3_1])
                    Ts.writes(pad_temp[i0_1, i1_1, i2_1, i3_1])
                    pad_temp[i0_1, i1_1, i2_1, i3_1] = rxplaceholder[i0_1, i1_1, i2_1, i3_1]
            for i0, i1, i2, i3, i4, i5, i6 in T.grid(n_conv2d, f_conv2d, h_conv2d + T.int64(1) - kh_conv2d, w_conv2d + T.int64(1) - kw_conv2d, c_conv2d, kh_conv2d, kw_conv2d):
                with Ts.sblock("conv2d_nchw"):
                    nn, ff, yy, xx, rc, ry, rx = Ts.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
                    Ts.reads(pad_temp[nn, rc, yy + ry, xx + rx], rxplaceholder_1[ff, rc, ry, rx])
                    Ts.writes(conv2d_nchw[nn, ff, yy, xx])
                    with Ts.init():
                        conv2d_nchw[nn, ff, yy, xx] = T.float32(0)
                    conv2d_nchw[nn, ff, yy, xx] = conv2d_nchw[nn, ff, yy, xx] + pad_temp[nn, rc, yy + ry, xx + rx] * rxplaceholder_1[ff, rc, ry, rx]
    # fmt: on

    mod = LegalizeOps()(Conv2d)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_conv2d_symbolic_group():
    # fmt: off
    n = T.dynamic("n", "int64")
    f = T.dynamic("f", "int64")
    c = T.dynamic("c", "int64")
    c_div_8 = T.dynamic("c_div_8", "int64")

    @tvm_tirx.script.ir_module
    class Conv2d:
        @R.function
        def main(x: R.Tensor((n, c, 28, 28), "float32"), w: R.Tensor((f, c_div_8, 3, 3), "float32")) -> R.Tensor((n, f, 26, 26), "float32"):
            gv: R.Tensor((n, f, 26, 26), "float32") = R.nn.conv2d(x, w, groups=8)
            return gv

    n_main = T.dynamic("n", "int64")
    f_main = T.dynamic("f", "int64")
    c_main = T.dynamic("c", "int64")
    c_div_8_main = T.dynamic("c_div_8", "int64")
    n_conv2d = T.dynamic("n", "int64")
    c_conv2d = T.dynamic("c", "int64")
    f_conv2d = T.dynamic("f", "int64")
    c_div_8_conv2d = T.dynamic("c_div_8", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((n_main, c_main, 28, 28), dtype="float32"), w: R.Tensor((f_main, c_div_8_main, 3, 3), dtype="float32")) -> R.Tensor((n_main, f_main, 26, 26), dtype="float32"):
            gv = R.call_tir(Expected.conv2d, (x, w), ty_args=[R.Tensor((n_main, f_main, 26, 26), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def conv2d(x: T.Tensor((n_conv2d, c_conv2d, T.int64(28), T.int64(28)), "float32"), w: T.Tensor((f_conv2d, c_div_8_conv2d, T.int64(3), T.int64(3)), "float32"), group_conv2d_nchw: T.Tensor((n_conv2d, f_conv2d, T.int64(26), T.int64(26)), "float32")):
            T.func_attr({"tirx.noalias": True})

            pad_temp = Ts.sblock_alloc_tensor((n_conv2d, c_conv2d, T.int64(28), T.int64(28)))
            for i0, i1, i2, i3 in T.grid(n_conv2d, c_conv2d, T.int64(28), T.int64(28)):
                with Ts.sblock("pad_temp"):
                    v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(x[v_i0, v_i1, v_i2, v_i3])
                    Ts.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                    pad_temp[v_i0, v_i1, v_i2, v_i3] = x[v_i0, v_i1, v_i2, v_i3]
            for nn, ff, yy, xx, rc, ry, rx in T.grid(n_conv2d, f_conv2d, T.int64(26), T.int64(26), c_conv2d // T.int64(8), T.int64(3), T.int64(3)):
                with Ts.sblock("group_conv2d_nchw"):
                    v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = Ts.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
                    Ts.reads(pad_temp[v_nn, v_ff // (f_conv2d // T.int64(8)) * (c_conv2d // T.int64(8)) + v_rc, v_yy + v_ry, v_xx + v_rx], w[v_ff, v_rc, v_ry, v_rx])
                    Ts.writes(group_conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                    with Ts.init():
                        group_conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0.0)
                    group_conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = group_conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_ff // (f_conv2d // T.int64(8)) * (c_conv2d // T.int64(8)) + v_rc, v_yy + v_ry, v_xx + v_rx] * w[v_ff, v_rc, v_ry, v_rx]
    # fmt: on

    mod = LegalizeOps()(Conv2d)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_conv2d_transpose():
    # fmt: off
    @I.ir_module
    class Conv2dTranspose:
        @R.function
        def main(x: R.Tensor((2, 128, 28, 28), "float32"), w: R.Tensor((128, 16, 3, 3), "float32")):
            gv = R.nn.conv2d_transpose(x, w, strides=(2, 3), padding=(1, 1), dilation=(1, 1), output_padding=(1, 2), groups=8)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 128, 28, 28), dtype="float32"), w: R.Tensor((128, 16, 3, 3), dtype="float32")) -> R.Tensor((2, 128, 56, 84), dtype="float32"):
            gv = R.call_tir(Expected.conv2d_transpose, (x, w), ty_args=[R.Tensor((2, 128, 56, 84), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def conv2d_transpose(rxplaceholder: T.Tensor((T.int64(2), T.int64(128), T.int64(28), T.int64(28)), "float32"), rxplaceholder_1: T.Tensor((T.int64(128), T.int64(16), T.int64(3), T.int64(3)), "float32"), compute: T.Tensor((T.int64(2), T.int64(128), T.int64(56), T.int64(84)), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            data_dilate = Ts.sblock_alloc_tensor((T.int64(2), T.int64(128), T.int64(55), T.int64(82)))
            data_pad = Ts.sblock_alloc_tensor((T.int64(2), T.int64(128), T.int64(58), T.int64(86)))
            kernel_transform = Ts.sblock_alloc_tensor((T.int64(16), T.int64(128), T.int64(3), T.int64(3)))
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(128), T.int64(55), T.int64(82)):
                with Ts.sblock("data_dilate"):
                    v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[v_i0, v_i1, v_i2 // T.int64(2), v_i3 // T.int64(3)])
                    Ts.writes(data_dilate[v_i0, v_i1, v_i2, v_i3])
                    data_dilate[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(v_i2 % T.int64(2) == T.int64(0) and v_i3 % T.int64(3) == T.int64(0), rxplaceholder[v_i0, v_i1, v_i2 // T.int64(2), v_i3 // T.int64(3)], T.float32(0))
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(128), T.int64(58), T.int64(86)):
                with Ts.sblock("data_pad"):
                    v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(data_dilate[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                    Ts.writes(data_pad[v_i0, v_i1, v_i2, v_i3])
                    data_pad[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(56) and T.int64(1) <= v_i3 and v_i3 < T.int64(83), data_dilate[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
            for i, o, h_index, w_index in T.grid(T.int64(16), T.int64(128), T.int64(3), T.int64(3)):
                with Ts.sblock("kernel_transform"):
                    v_i, v_o, v_h, v_w = Ts.axis.remap("SSSS", [i, o, h_index, w_index])
                    Ts.reads(rxplaceholder_1[v_o, v_i, T.int64(2) - v_h, T.int64(2) - v_w])
                    Ts.writes(kernel_transform[v_i, v_o, v_h, v_w])
                    kernel_transform[v_i, v_o, v_h, v_w] = rxplaceholder_1[v_o, v_i, T.int64(2) - v_h, T.int64(2) - v_w]
            for b_index, c_index, h_index, w_index, dc, dh, dw in T.grid(T.int64(2), T.int64(128), T.int64(56), T.int64(84), T.int64(16), T.int64(3), T.int64(3)):
                with Ts.sblock("compute"):
                    v_b, v_c, v_h, v_w, v_dc, v_dh, v_dw = Ts.axis.remap("SSSSRRR", [b_index, c_index, h_index, w_index, dc, dh, dw])
                    Ts.reads(data_pad[v_b, v_c // T.int64(16) * T.int64(16) + v_dc, v_h + v_dh, v_w + v_dw], kernel_transform[v_c % T.int64(16), v_c // T.int64(16) * T.int64(16) + v_dc, v_dh, v_dw])
                    Ts.writes(compute[v_b, v_c, v_h, v_w])
                    with Ts.init():
                        compute[v_b, v_c, v_h, v_w] = T.float32(0)
                    compute[v_b, v_c, v_h, v_w] = compute[v_b, v_c, v_h, v_w] + data_pad[v_b, v_c // T.int64(16) * T.int64(16) + v_dc, v_h + v_dh, v_w + v_dw] * kernel_transform[v_c % T.int64(16), v_c // T.int64(16) * T.int64(16) + v_dc, v_dh, v_dw]
    # fmt: on

    mod = LegalizeOps()(Conv2dTranspose)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_conv3d_transpose():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Conv3dTranspose:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 4, 4), "float32"), w: R.Tensor((3, 4, 3, 3, 3), "float32")):
            gv = R.nn.conv3d_transpose(x, w)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 4, 4), dtype="float32"), w: R.Tensor((3, 4, 3, 3, 3), dtype="float32")) -> R.Tensor((2, 4, 6, 6, 6), dtype="float32"):
            gv = R.call_tir(Expected.conv3d_transpose, (x, w), ty_args=[R.Tensor((2, 4, 6, 6, 6), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def conv3d_transpose(x: T.Tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(4), T.int64(4)), "float32"), w: T.Tensor((T.int64(3), T.int64(4), T.int64(3), T.int64(3), T.int64(3)), "float32"), compute: T.Tensor((T.int64(2), T.int64(4), T.int64(6), T.int64(6), T.int64(6)), "float32")):
            T.func_attr({"tirx.noalias": True})
            data_dilate = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(4), T.int64(4)))
            data_pad = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3), T.int64(8), T.int64(8), T.int64(8)))
            kernel_transform = Ts.sblock_alloc_tensor((T.int64(4), T.int64(3), T.int64(3), T.int64(3), T.int64(3)))
            for i0, i1, i2, i3, i4 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(4), T.int64(4)):
                with Ts.sblock("data_dilate"):
                    v_i0, v_i1, v_i2, v_i3, v_i4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                    Ts.reads(x[v_i0, v_i1, v_i2, v_i3, v_i4])
                    Ts.writes(data_dilate[v_i0, v_i1, v_i2, v_i3, v_i4])
                    data_dilate[v_i0, v_i1, v_i2, v_i3, v_i4] = x[v_i0, v_i1, v_i2, v_i3, v_i4]
            for i0, i1, i2, i3, i4 in T.grid(T.int64(2), T.int64(3), T.int64(8), T.int64(8), T.int64(8)):
                with Ts.sblock("data_pad"):
                    v_i0, v_i1, v_i2, v_i3, v_i4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                    Ts.reads(data_dilate[v_i0, v_i1, v_i2 - T.int64(2), v_i3 - T.int64(2), v_i4 - T.int64(2)])
                    Ts.writes(data_pad[v_i0, v_i1, v_i2, v_i3, v_i4])
                    data_pad[v_i0, v_i1, v_i2, v_i3, v_i4] = T.if_then_else(T.int64(2) <= v_i2 and v_i2 < T.int64(6) and T.int64(2) <= v_i3 and v_i3 < T.int64(6) and T.int64(2) <= v_i4 and v_i4 < T.int64(6), data_dilate[v_i0, v_i1, v_i2 - T.int64(2), v_i3 - T.int64(2), v_i4 - T.int64(2)], T.float32(0.0))
            for o, i, d, h_index, w_1 in T.grid(T.int64(4), T.int64(3), T.int64(3), T.int64(3), T.int64(3)):
                with Ts.sblock("kernel_transform"):
                    v_o, v_i, v_d, v_h, v_w = Ts.axis.remap("SSSSS", [o, i, d, h_index, w_1])
                    Ts.reads(w[v_i, v_o, T.int64(2) - v_d, T.int64(2) - v_h, T.int64(2) - v_w])
                    Ts.writes(kernel_transform[v_o, v_i, v_d, v_h, v_w])
                    kernel_transform[v_o, v_i, v_d, v_h, v_w] = w[v_i, v_o, T.int64(2) - v_d, T.int64(2) - v_h, T.int64(2) - v_w]
            for b_index, c_index, d, h_index, w_1, dc, dd, dh, dw in T.grid(T.int64(2), T.int64(4), T.int64(6), T.int64(6), T.int64(6), T.int64(3), T.int64(3), T.int64(3), T.int64(3)):
                with Ts.sblock("compute"):
                    v_b, v_c, v_d, v_h, v_w, v_dc, v_dd, v_dh, v_dw = Ts.axis.remap("SSSSSRRRR", [b_index, c_index, d, h_index, w_1, dc, dd, dh, dw])
                    Ts.reads(data_pad[v_b, v_dc, v_d + v_dd, v_h + v_dh, v_w + v_dw], kernel_transform[v_c, v_dc, v_dd, v_dh, v_dw])
                    Ts.writes(compute[v_b, v_c, v_d, v_h, v_w])
                    with Ts.init():
                        compute[v_b, v_c, v_d, v_h, v_w] = T.float32(0.0)
                    compute[v_b, v_c, v_d, v_h, v_w] = compute[v_b, v_c, v_d, v_h, v_w] + data_pad[v_b, v_dc, v_d + v_dd, v_h + v_dh, v_w + v_dw] * kernel_transform[v_c, v_dc, v_dd, v_dh, v_dw]
    # fmt: on

    mod = LegalizeOps()(Conv3dTranspose)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_conv3d_transpose_with_out_dtype():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Conv3dTranspose:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 4, 4), "float32"), w: R.Tensor((3, 4, 3, 3, 3), "float32")):
            gv = R.nn.conv3d_transpose(x, w, out_dtype="float16")
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 4, 4), dtype="float32"), w: R.Tensor((3, 4, 3, 3, 3), dtype="float32")) -> R.Tensor((2, 4, 6, 6, 6), dtype="float16"):
            gv = R.call_tir(Expected.conv3d_transpose, (x, w), ty_args=[R.Tensor((2, 4, 6, 6, 6), dtype="float16")])
            return gv

        @Ts.function(private=True)
        def conv3d_transpose(x: T.Tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(4), T.int64(4)), "float32"), w: T.Tensor((T.int64(3), T.int64(4), T.int64(3), T.int64(3), T.int64(3)), "float32"), compute: T.Tensor((T.int64(2), T.int64(4), T.int64(6), T.int64(6), T.int64(6)), "float16")):
            T.func_attr({"tirx.noalias": True})
            data_dilate = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(4), T.int64(4)))
            data_pad = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3), T.int64(8), T.int64(8), T.int64(8)))
            kernel_transform = Ts.sblock_alloc_tensor((T.int64(4), T.int64(3), T.int64(3), T.int64(3), T.int64(3)))
            for i0, i1, i2, i3, i4 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(4), T.int64(4)):
                with Ts.sblock("data_dilate"):
                    v_i0, v_i1, v_i2, v_i3, v_i4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                    Ts.reads(x[v_i0, v_i1, v_i2, v_i3, v_i4])
                    Ts.writes(data_dilate[v_i0, v_i1, v_i2, v_i3, v_i4])
                    data_dilate[v_i0, v_i1, v_i2, v_i3, v_i4] = x[v_i0, v_i1, v_i2, v_i3, v_i4]
            for i0, i1, i2, i3, i4 in T.grid(T.int64(2), T.int64(3), T.int64(8), T.int64(8), T.int64(8)):
                with Ts.sblock("data_pad"):
                    v_i0, v_i1, v_i2, v_i3, v_i4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                    Ts.reads(data_dilate[v_i0, v_i1, v_i2 - T.int64(2), v_i3 - T.int64(2), v_i4 - T.int64(2)])
                    Ts.writes(data_pad[v_i0, v_i1, v_i2, v_i3, v_i4])
                    data_pad[v_i0, v_i1, v_i2, v_i3, v_i4] = T.if_then_else(T.int64(2) <= v_i2 and v_i2 < T.int64(6) and T.int64(2) <= v_i3 and v_i3 < T.int64(6) and T.int64(2) <= v_i4 and v_i4 < T.int64(6), data_dilate[v_i0, v_i1, v_i2 - T.int64(2), v_i3 - T.int64(2), v_i4 - T.int64(2)], T.float32(0.0))
            for o, i, d, h_index, w_1 in T.grid(T.int64(4), T.int64(3), T.int64(3), T.int64(3), T.int64(3)):
                with Ts.sblock("kernel_transform"):
                    v_o, v_i, v_d, v_h, v_w = Ts.axis.remap("SSSSS", [o, i, d, h_index, w_1])
                    Ts.reads(w[v_i, v_o, T.int64(2) - v_d, T.int64(2) - v_h, T.int64(2) - v_w])
                    Ts.writes(kernel_transform[v_o, v_i, v_d, v_h, v_w])
                    kernel_transform[v_o, v_i, v_d, v_h, v_w] = w[v_i, v_o, T.int64(2) - v_d, T.int64(2) - v_h, T.int64(2) - v_w]
            for b_index, c_index, d, h_index, w_1, dc, dd, dh, dw in T.grid(T.int64(2), T.int64(4), T.int64(6), T.int64(6), T.int64(6), T.int64(3), T.int64(3), T.int64(3), T.int64(3)):
                with Ts.sblock("compute"):
                    v_b, v_c, v_d, v_h, v_w, v_dc, v_dd, v_dh, v_dw = Ts.axis.remap("SSSSSRRRR", [b_index, c_index, d, h_index, w_1, dc, dd, dh, dw])
                    Ts.reads(data_pad[v_b, v_dc, v_d + v_dd, v_h + v_dh, v_w + v_dw], kernel_transform[v_c, v_dc, v_dd, v_dh, v_dw])
                    Ts.writes(compute[v_b, v_c, v_d, v_h, v_w])
                    with Ts.init():
                        compute[v_b, v_c, v_d, v_h, v_w] = T.float16(0.0)
                    compute[v_b, v_c, v_d, v_h, v_w] = compute[v_b, v_c, v_d, v_h, v_w] + T.Cast("float16", data_pad[v_b, v_dc, v_d + v_dd, v_h + v_dh, v_w + v_dw]) * T.Cast("float16", kernel_transform[v_c, v_dc, v_dd, v_dh, v_dw])
    # fmt: on

    mod = LegalizeOps()(Conv3dTranspose)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_conv2d_transpose_with_out_dtype():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Conv2dTranspose:
        @R.function
        def main(x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((3, 4, 3, 3), "float32")):
            gv = R.nn.conv2d_transpose(x, w, out_dtype="float16")
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((3, 4, 3, 3), dtype="float32")) -> R.Tensor((2, 4, 30, 30), dtype="float16"):
            gv = R.call_tir(Expected.conv2d_transpose, (x, w), ty_args=[R.Tensor((2, 4, 30, 30), dtype="float16")])
            return gv

        @Ts.function(private=True)
        def conv2d_transpose(rxplaceholder: T.Tensor((T.int64(2), T.int64(3), T.int64(28), T.int64(28)), "float32"), rxplaceholder_1: T.Tensor((T.int64(3), T.int64(4), T.int64(3), T.int64(3)), "float32"), compute: T.Tensor((T.int64(2), T.int64(4), T.int64(30), T.int64(30)), "float16")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            data_dilate = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3), T.int64(28), T.int64(28)))
            data_pad = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3), T.int64(32), T.int64(32)))
            kernel_transform = Ts.sblock_alloc_tensor((T.int64(4), T.int64(3), T.int64(3), T.int64(3)))
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(28), T.int64(28)):
                with Ts.sblock("data_dilate"):
                    v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[v_i0, v_i1, v_i2, v_i3])
                    Ts.writes(data_dilate[v_i0, v_i1, v_i2, v_i3])
                    data_dilate[v_i0, v_i1, v_i2, v_i3] = rxplaceholder[v_i0, v_i1, v_i2, v_i3]
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(32), T.int64(32)):
                with Ts.sblock("data_pad"):
                    v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(data_dilate[v_i0, v_i1, v_i2 - T.int64(2), v_i3 - T.int64(2)])
                    Ts.writes(data_pad[v_i0, v_i1, v_i2, v_i3])
                    data_pad[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(2) <= v_i2 and v_i2 < T.int64(30) and T.int64(2) <= v_i3 and v_i3 < T.int64(30), data_dilate[v_i0, v_i1, v_i2 - T.int64(2), v_i3 - T.int64(2)], T.float32(0))
            for o, i, h_index, w_index in T.grid(T.int64(4), T.int64(3), T.int64(3), T.int64(3)):
                with Ts.sblock("kernel_transform"):
                    v_o, v_i, v_h, v_w = Ts.axis.remap("SSSS", [o, i, h_index, w_index])
                    Ts.reads(rxplaceholder_1[v_i, v_o, T.int64(2) - v_h, T.int64(2) - v_w])
                    Ts.writes(kernel_transform[v_o, v_i, v_h, v_w])
                    kernel_transform[v_o, v_i, v_h, v_w] = rxplaceholder_1[v_i, v_o, T.int64(2) - v_h, T.int64(2) - v_w]
            for b_index, c_index, h_index, w_index, dc, dh, dw in T.grid(T.int64(2), T.int64(4), T.int64(30), T.int64(30), T.int64(3), T.int64(3), T.int64(3)):
                with Ts.sblock("compute"):
                    v_b, v_c, v_h, v_w, v_dc, v_dh, v_dw = Ts.axis.remap("SSSSRRR", [b_index, c_index, h_index, w_index, dc, dh, dw])
                    Ts.reads(data_pad[v_b, v_dc, v_h + v_dh, v_w + v_dw], kernel_transform[v_c, v_dc, v_dh, v_dw])
                    Ts.writes(compute[v_b, v_c, v_h, v_w])
                    with Ts.init():
                        compute[v_b, v_c, v_h, v_w] = T.float16(0)
                    compute[v_b, v_c, v_h, v_w] = compute[v_b, v_c, v_h, v_w] + T.Cast("float16", data_pad[v_b, v_dc, v_h + v_dh, v_w + v_dw]) * T.Cast("float16", kernel_transform[v_c, v_dc, v_dh, v_dw])
    # fmt: on

    mod = LegalizeOps()(Conv2dTranspose)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_conv2d_transpose_symbolic():
    # fmt: off
    n = T.dynamic("n", "int64")
    c = T.dynamic("c", "int64")
    h = T.dynamic("h", "int64")
    w = T.dynamic("w", "int64")
    f = T.dynamic("f", "int64")
    kh = T.dynamic("kh", "int64")
    kw = T.dynamic("kw", "int64")

    @tvm_tirx.script.ir_module
    class Conv2dTranspose:
        @R.function
        def main(x: R.Tensor((n, c, h, w), "float32"), kernel: R.Tensor((f, c, kh, kw), "float32")):
            gv = R.nn.conv2d_transpose(x, kernel, strides=(3, 3))
            return gv

    n_main = T.dynamic("n", "int64")
    c_main = T.dynamic("c", "int64")
    h_main = T.dynamic("h", "int64")
    kh_main = T.dynamic("kh", "int64")
    w_main = T.dynamic("w", "int64")
    kw_main = T.dynamic("kw", "int64")
    f_main = T.dynamic("f", "int64")
    n_conv2d_transpose = T.dynamic("n", "int64")
    c_conv2d_transpose = T.dynamic("c", "int64")
    h_conv2d_transpose = T.dynamic("h", "int64")
    w_conv2d_transpose = T.dynamic("w", "int64")
    f_conv2d_transpose = T.dynamic("f", "int64")
    kh_conv2d_transpose = T.dynamic("kh", "int64")
    kw_conv2d_transpose = T.dynamic("kw", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((n_main, c_main, h_main, w_main), dtype="float32"), kernel: R.Tensor((f_main, c_main, kh_main, kw_main), dtype="float32")) -> R.Tensor((n_main, c_main, h_main * 3 + kh_main - 3, w_main * 3 + kw_main - 3), dtype="float32"):
            gv = R.call_tir(Expected.conv2d_transpose, (x, kernel), ty_args=[R.Tensor((n_main, c_main, h_main * 3 + kh_main - 3, w_main * 3 + kw_main - 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def conv2d_transpose(rxplaceholder: T.Tensor((n_conv2d_transpose, c_conv2d_transpose, h_conv2d_transpose, w_conv2d_transpose), "float32"), rxplaceholder_1: T.Tensor((f_conv2d_transpose, c_conv2d_transpose, kh_conv2d_transpose, kw_conv2d_transpose), "float32"), compute: T.Tensor((n_conv2d_transpose, c_conv2d_transpose, h_conv2d_transpose * T.int64(3) + kh_conv2d_transpose - T.int64(3), w_conv2d_transpose * T.int64(3) + kw_conv2d_transpose - T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            data_dilate = Ts.sblock_alloc_tensor((n_conv2d_transpose, c_conv2d_transpose, h_conv2d_transpose * T.int64(3) - T.int64(2), w_conv2d_transpose * T.int64(3) - T.int64(2)))
            data_pad = Ts.sblock_alloc_tensor((n_conv2d_transpose, c_conv2d_transpose, h_conv2d_transpose * T.int64(3) + kh_conv2d_transpose * T.int64(2) - T.int64(4), w_conv2d_transpose * T.int64(3) + kw_conv2d_transpose * T.int64(2) - T.int64(4)))
            kernel_transform = Ts.sblock_alloc_tensor((c_conv2d_transpose, c_conv2d_transpose, kh_conv2d_transpose, kw_conv2d_transpose))
            for i0, i1, i2, i3 in T.grid(n_conv2d_transpose, c_conv2d_transpose, h_conv2d_transpose * T.int64(3) - T.int64(2), w_conv2d_transpose * T.int64(3) - T.int64(2)):
                with Ts.sblock("data_dilate"):
                    v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[v_i0, v_i1, v_i2 // T.int64(3), v_i3 // T.int64(3)])
                    Ts.writes(data_dilate[v_i0, v_i1, v_i2, v_i3])
                    data_dilate[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(v_i2 % T.int64(3) == T.int64(0) and v_i3 % T.int64(3) == T.int64(0), rxplaceholder[v_i0, v_i1, v_i2 // T.int64(3), v_i3 // T.int64(3)], T.float32(0))
            for i0, i1, i2, i3 in T.grid(n_conv2d_transpose, c_conv2d_transpose, h_conv2d_transpose * T.int64(3) + kh_conv2d_transpose * T.int64(2) - T.int64(4), w_conv2d_transpose * T.int64(3) + kw_conv2d_transpose * T.int64(2) - T.int64(4)):
                with Ts.sblock("data_pad"):
                    v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(data_dilate[v_i0, v_i1, v_i2 + T.int64(1) - kh_conv2d_transpose, v_i3 + T.int64(1) - kw_conv2d_transpose])
                    Ts.writes(data_pad[v_i0, v_i1, v_i2, v_i3])
                    data_pad[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(kh_conv2d_transpose <= v_i2 + T.int64(1) and v_i2 + T.int64(3)< h_conv2d_transpose * T.int64(3) + kh_conv2d_transpose and kw_conv2d_transpose <= v_i3 + T.int64(1) and v_i3 + T.int64(3) < w_conv2d_transpose * T.int64(3) + kw_conv2d_transpose , data_dilate[v_i0, v_i1, v_i2 + T.int64(1) - kh_conv2d_transpose, v_i3 + T.int64(1) - kw_conv2d_transpose], T.float32(0))
            for o, i, h_1, w_1 in T.grid(c_conv2d_transpose, c_conv2d_transpose, kh_conv2d_transpose, kw_conv2d_transpose):
                with Ts.sblock("kernel_transform"):
                    v_o, v_i, v_h, v_w = Ts.axis.remap("SSSS", [o, i, h_1, w_1])
                    Ts.reads(rxplaceholder_1[v_i, v_o, kh_conv2d_transpose - v_h - T.int64(1), kw_conv2d_transpose - v_w - T.int64(1)])
                    Ts.writes(kernel_transform[v_o, v_i, v_h, v_w])
                    kernel_transform[v_o, v_i, v_h, v_w] = rxplaceholder_1[v_i, v_o, kh_conv2d_transpose - v_h - T.int64(1), kw_conv2d_transpose - v_w - T.int64(1)]
            for b_index, c_1, h_1, w_1, dc, dh, dw in T.grid(n_conv2d_transpose, c_conv2d_transpose, h_conv2d_transpose * T.int64(3) + kh_conv2d_transpose - T.int64(3), w_conv2d_transpose * T.int64(3) + kw_conv2d_transpose - T.int64(3), c_conv2d_transpose, kh_conv2d_transpose, kw_conv2d_transpose):
                with Ts.sblock("compute"):
                    v_b, v_c, v_h, v_w, v_dc, v_dh, v_dw = Ts.axis.remap("SSSSRRR", [b_index, c_1, h_1, w_1, dc, dh, dw])
                    Ts.reads(data_pad[v_b, v_dc, v_h + v_dh, v_w + v_dw], kernel_transform[v_c, v_dc, v_dh, v_dw])
                    Ts.writes(compute[v_b, v_c, v_h, v_w])
                    with Ts.init():
                        compute[v_b, v_c, v_h, v_w] = T.float32(0)
                    compute[v_b, v_c, v_h, v_w] = compute[v_b, v_c, v_h, v_w] + data_pad[v_b, v_dc, v_h + v_dh, v_w + v_dw] * kernel_transform[v_c, v_dc, v_dh, v_dw]
    # fmt: on

    mod = LegalizeOps()(Conv2dTranspose)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_conv2d_transpose_dilation():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Conv2dTranspose:
        @R.function
        def main(x: R.Tensor((1, 1, 3, 3), "float32"), w: R.Tensor((1, 1, 2, 2), "float32")):
            gv = R.nn.conv2d_transpose(x, w, dilation=(2, 2))
            return gv

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def conv2d_transpose(x: T.Tensor((T.int64(1), T.int64(1), T.int64(3), T.int64(3)), "float32"), w: T.Tensor((T.int64(1), T.int64(1), T.int64(2), T.int64(2)), "float32"), compute: T.Tensor((T.int64(1), T.int64(1), T.int64(5), T.int64(5)), "float32")):
            T.func_attr({"tirx.noalias": True})
            data_dilate = Ts.sblock_alloc_tensor((T.int64(1), T.int64(1), T.int64(3), T.int64(3)))
            data_pad = Ts.sblock_alloc_tensor((T.int64(1), T.int64(1), T.int64(7), T.int64(7)))
            kernel_dilate = Ts.sblock_alloc_tensor((T.int64(1), T.int64(1), T.int64(3), T.int64(3)))
            kernel_transform = Ts.sblock_alloc_tensor((T.int64(1), T.int64(1), T.int64(3), T.int64(3)))
            for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(1), T.int64(3), T.int64(3)):
                with Ts.sblock("data_dilate"):
                    v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    data_dilate[v_i0, v_i1, v_i2, v_i3] = x[v_i0, v_i1, v_i2, v_i3]
            for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(1), T.int64(7), T.int64(7)):
                with Ts.sblock("data_pad"):
                    v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    data_pad[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(2) <= v_i2 and v_i2 < T.int64(5) and T.int64(2) <= v_i3 and v_i3 < T.int64(5), data_dilate[v_i0, v_i1, v_i2 - T.int64(2), v_i3 - T.int64(2)], T.float32(0.0))
            for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(1), T.int64(3), T.int64(3)):
                with Ts.sblock("kernel_dilate"):
                    v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    kernel_dilate[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(v_i2 % T.int64(2) == T.int64(0) and v_i3 % T.int64(2) == T.int64(0), w[v_i0, v_i1, v_i2 // T.int64(2), v_i3 // T.int64(2)], T.float32(0.0))
            for o, i, h_index, w_1 in T.grid(T.int64(1), T.int64(1), T.int64(3), T.int64(3)):
                with Ts.sblock("kernel_transform"):
                    v_o, v_i, v_h, v_w = Ts.axis.remap("SSSS", [o, i, h_index, w_1])
                    kernel_transform[v_o, v_i, v_h, v_w] = kernel_dilate[v_i, v_o, T.int64(2) - v_h, T.int64(2) - v_w]
            for b_index, c_index, h_index, w_1, dc, dh, dw in T.grid(T.int64(1), T.int64(1), T.int64(5), T.int64(5), T.int64(1), T.int64(3), T.int64(3)):
                with Ts.sblock("compute"):
                    v_b, v_c, v_h, v_w, v_dc, v_dh, v_dw = Ts.axis.remap("SSSSRRR", [b_index, c_index, h_index, w_1, dc, dh, dw])
                    with Ts.init():
                        compute[v_b, v_c, v_h, v_w] = T.float32(0.0)
                    compute[v_b, v_c, v_h, v_w] = compute[v_b, v_c, v_h, v_w] + data_pad[v_b, v_dc, v_h + v_dh, v_w + v_dw] * kernel_transform[v_c, v_dc, v_dh, v_dw]

        @R.function
        def main(x: R.Tensor((1, 1, 3, 3), dtype="float32"), w: R.Tensor((1, 1, 2, 2), dtype="float32")) -> R.Tensor((1, 1, 5, 5), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.conv2d_transpose, (x, w), ty_args=[R.Tensor((1, 1, 5, 5), dtype="float32")])
            return gv
    # fmt: on

    mod = LegalizeOps()(Conv2dTranspose)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_max_pool2d():
    # fmt: off
    @tvm_tirx.script.ir_module
    class MaxPool2D:
        @R.function
        def main(x: R.Tensor((4, 112, 112, 6), "float32")) -> R.Tensor((4, 56, 56, 6), "float32"):
            gv: R.Tensor((4, 56, 56, 6), "float32") = R.nn.max_pool2d(x, pool_size=[3, 3], strides=[2, 2], dilation=[1, 1], padding=[1, 1, 1, 1], layout="NHWC")
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((4, 112, 112, 6), "float32")) -> R.Tensor((4, 56, 56, 6), "float32"):
            gv = R.call_tir(Expected.max_pool2d, (x,), ty_args=[R.Tensor((4, 56, 56, 6), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def max_pool2d(rxplaceholder: T.Tensor((T.int64(4), T.int64(112), T.int64(112), T.int64(6)), "float32"), pool_max: T.Tensor((T.int64(4), T.int64(56), T.int64(56), T.int64(6)), "float32")):
            T.func_attr({"tirx.noalias": True})
            pad_temp = Ts.sblock_alloc_tensor([T.int64(4), T.int64(114), T.int64(114), T.int64(6)], dtype="float32")
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(114), T.int64(114), T.int64(6)):
                with Ts.sblock("pad_temp"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[ax0, ax1 - T.int64(1), ax2 - T.int64(1), ax3])
                    Ts.writes(pad_temp[ax0, ax1, ax2, ax3])
                    pad_temp[ax0, ax1, ax2, ax3] = T.if_then_else(T.int64(1) <= ax1 and ax1 < T.int64(113) and T.int64(1) <= ax2 and ax2 < T.int64(113), rxplaceholder[ax0, ax1 - T.int64(1), ax2 - T.int64(1), ax3], T.float32(-3.4028234663852886e+38))
            for i0, i1, i2, i3, i4, i5 in T.grid(T.int64(4), T.int64(56), T.int64(56), T.int64(6), T.int64(3), T.int64(3)):
                with Ts.sblock("pool_max"):
                    ax0, ax1, ax2, ax3, rv0, rv1 = Ts.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                    Ts.reads(pad_temp[ax0, ax1 * T.int64(2) + rv0, ax2 * T.int64(2) + rv1, ax3])
                    Ts.writes(pool_max[ax0, ax1, ax2, ax3])
                    Ts.sblock_attr({"schedule_rule":"meta_schedule.pool_max"})
                    with Ts.init():
                        pool_max[ax0, ax1, ax2, ax3] = T.float32(-3.4028234663852886e+38)
                    pool_max[ax0, ax1, ax2, ax3] = T.max(pool_max[ax0, ax1, ax2, ax3], pad_temp[ax0, ax1 * T.int64(2) + rv0, ax2 * T.int64(2) + rv1, ax3])
    # fmt: on

    mod = LegalizeOps()(MaxPool2D)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_max_pool2d_NCHW16c():
    # fmt: off
    @tvm_tirx.script.ir_module
    class MaxPool2D:
        @R.function
        def main(x: R.Tensor((4, 4, 112, 112, 16), "float32")) -> R.Tensor((4, 4, 110, 110, 16), "float32"):
            gv: R.Tensor((4, 4, 110, 110, 16), "float32") = R.nn.max_pool2d(x, pool_size=[3, 3], layout="NCHW16c")
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((4, 4, 112, 112, 16), "float32")) -> R.Tensor((4, 4, 110, 110, 16), "float32"):
            gv = R.call_tir(Expected.max_pool2d, (x,), ty_args=[R.Tensor((4, 4, 110, 110, 16), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def max_pool2d(rxplaceholder: T.Tensor((T.int64(4), T.int64(4), T.int64(112), T.int64(112), T.int64(16)), "float32"), pool_max: T.Tensor((T.int64(4), T.int64(4), T.int64(110), T.int64(110), T.int64(16)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2, i3, i4, i5, i6 in T.grid(T.int64(4), T.int64(4), T.int64(110), T.int64(110), T.int64(16), T.int64(3), T.int64(3)):
                with Ts.sblock("pool_max"):
                    ax0, ax1, ax2, ax3, ax4, rv0, rv1 = Ts.axis.remap("SSSSSRR", [i0, i1, i2, i3, i4, i5, i6])
                    Ts.reads(rxplaceholder[ax0, ax1, ax2 + rv0, ax3 + rv1, ax4])
                    Ts.writes(pool_max[ax0, ax1, ax2, ax3, ax4])
                    Ts.sblock_attr({"schedule_rule":"meta_schedule.pool_max"})
                    with Ts.init():
                        pool_max[ax0, ax1, ax2, ax3, ax4] = T.float32(-3.4028234663852886e+38)
                    pool_max[ax0, ax1, ax2, ax3, ax4] = T.max(pool_max[ax0, ax1, ax2, ax3, ax4], rxplaceholder[ax0, ax1, ax2 + rv0, ax3 + rv1, ax4])
    # fmt: on

    mod = LegalizeOps()(MaxPool2D)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_max_pool2d_ceil_mode():
    # fmt: off
    @tvm_tirx.script.ir_module
    class MaxPool2D:
        @R.function
        def main(x: R.Tensor((4, 6, 112, 112), "float32")) -> R.Tensor((4, 6, 38, 38), "float32"):
            gv: R.Tensor((4, 6, 38, 38), "float32") = R.nn.max_pool2d(x, pool_size=[3, 3], strides=[3, 3], dilation=[1, 1], padding=[1, 1, 1, 1], ceil_mode=True)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((4, 6, 112, 112), dtype="float32")) -> R.Tensor((4, 6, 38, 38), dtype="float32"):
            gv = R.call_tir(Expected.max_pool2d, (x,), ty_args=[R.Tensor((4, 6, 38, 38), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def max_pool2d(rxplaceholder: T.Tensor((T.int64(4), T.int64(6), T.int64(112), T.int64(112)), "float32"), pool_max: T.Tensor((T.int64(4), T.int64(6), T.int64(38), T.int64(38)), "float32")):
            T.func_attr({"tirx.noalias": True})
            pad_temp = Ts.sblock_alloc_tensor([T.int64(4), T.int64(6), T.int64(116), T.int64(116)], dtype="float32")
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(6), T.int64(116), T.int64(116)):
                with Ts.sblock("pad_temp"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[ax0, ax1, ax2 - T.int64(1), ax3 - T.int64(1)])
                    Ts.writes(pad_temp[ax0, ax1, ax2, ax3])
                    pad_temp[ax0, ax1, ax2, ax3] = T.if_then_else(T.int64(1) <= ax2 and ax2 < T.int64(113) and T.int64(1) <= ax3 and ax3 < T.int64(113), rxplaceholder[ax0, ax1, ax2 - T.int64(1), ax3 - T.int64(1)], T.float32(-3.4028234663852886e+38))
            for i0, i1, i2, i3, i4, i5 in T.grid(T.int64(4), T.int64(6), T.int64(38), T.int64(38), T.int64(3), T.int64(3)):
                with Ts.sblock("pool_max"):
                    ax0, ax1, ax2, ax3, rv0, rv1 = Ts.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                    Ts.reads(pad_temp[ax0, ax1, ax2 * T.int64(3) + rv0, ax3 * T.int64(3) + rv1])
                    Ts.writes(pool_max[ax0, ax1, ax2, ax3])
                    Ts.sblock_attr({"schedule_rule":"meta_schedule.pool_max"})
                    with Ts.init():
                        pool_max[ax0, ax1, ax2, ax3] = T.float32(-3.4028234663852886e+38)
                    pool_max[ax0, ax1, ax2, ax3] = T.max(pool_max[ax0, ax1, ax2, ax3], pad_temp[ax0, ax1, ax2 * T.int64(3) + rv0, ax3 * T.int64(3) + rv1])
    # fmt: on

    mod = LegalizeOps()(MaxPool2D)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


@pytest.mark.skip("TOPI pooling casts every shape value to i32.")
def test_max_pool2d_symbolic():
    # fmt: off
    n = T.dynamic("n", "int64")
    c = T.dynamic("c", "int64")
    h = T.dynamic("h", "int64")
    w = T.dynamic("w", "int64")
    kh = T.dynamic("kh", "int64")
    kw = T.dynamic("kw", "int64")

    @tvm_tirx.script.ir_module
    class MaxPool2D:
        @R.function
        def main(dumb_param: R.Tensor((kh, kw)), x: R.Tensor((n, c, h, w), "float32")) -> R.Tensor((n, c, h - kh + 1, w - kw + 1), "float32"):
            gv: R.Tensor((n, c, h - kh + 1, w - kw + 1), "float32") = R.nn.max_pool2d(x, pool_size=[kh, kw])
            return gv

    # fmt: on

    mod = LegalizeOps()(MaxPool2D)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_avg_pool2d():
    # fmt: off
    @tvm_tirx.script.ir_module
    class AvgPool2D:
        @R.function
        def main(x: R.Tensor((4, 112, 112, 6), "float32")) -> R.Tensor((4, 56, 56, 6), "float32"):
            gv: R.Tensor((4, 56, 56, 6), "float32") = R.nn.avg_pool2d(x, pool_size=[3, 3], strides=[2, 2], dilation=[1, 1], padding=[1, 1, 1, 1], layout="NHWC")
            return gv

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def avg_pool2d(rxplaceholder: T.Tensor((T.int64(4), T.int64(112), T.int64(112), T.int64(6)), "float32"), pool_avg: T.Tensor((T.int64(4), T.int64(56), T.int64(56), T.int64(6)), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            pad_temp = Ts.sblock_alloc_tensor((T.int64(4), T.int64(114), T.int64(114), T.int64(6)))
            pool_sum = Ts.sblock_alloc_tensor((T.int64(4), T.int64(56), T.int64(56), T.int64(6)))
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(4), T.int64(114), T.int64(114), T.int64(6)):
                with Ts.sblock("pad_temp"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(rxplaceholder[v_ax0, v_ax1 - T.int64(1), v_ax2 - T.int64(1), v_ax3])
                    Ts.writes(pad_temp[v_ax0, v_ax1, v_ax2, v_ax3])
                    pad_temp[v_ax0, v_ax1, v_ax2, v_ax3] = T.if_then_else(T.int64(1) <= v_ax1 and v_ax1 < T.int64(113) and T.int64(1) <= v_ax2 and v_ax2 < T.int64(113), rxplaceholder[v_ax0, v_ax1 - T.int64(1), v_ax2 - T.int64(1), v_ax3], T.float32(0))
            for ax0, ax1, ax2, ax3, rv0, rv1 in T.grid(T.int64(4), T.int64(56), T.int64(56), T.int64(6), T.int64(3), T.int64(3)):
                with Ts.sblock("pool_sum"):
                    v_ax0, v_ax1, v_ax2, v_ax3, v_rv0, v_rv1 = Ts.axis.remap("SSSSRR", [ax0, ax1, ax2, ax3, rv0, rv1])
                    Ts.reads(pad_temp[v_ax0, v_ax1 * T.int64(2) + v_rv0, v_ax2 * T.int64(2) + v_rv1, v_ax3])
                    Ts.writes(pool_sum[v_ax0, v_ax1, v_ax2, v_ax3])
                    with Ts.init():
                        pool_sum[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0)
                    pool_sum[v_ax0, v_ax1, v_ax2, v_ax3] = pool_sum[v_ax0, v_ax1, v_ax2, v_ax3] + pad_temp[v_ax0, v_ax1 * T.int64(2) + v_rv0, v_ax2 * T.int64(2) + v_rv1, v_ax3]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(4), T.int64(56), T.int64(56), T.int64(6)):
                with Ts.sblock("pool_avg"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(pool_sum[v_ax0, v_ax1, v_ax2, v_ax3])
                    Ts.writes(pool_avg[v_ax0, v_ax1, v_ax2, v_ax3])
                    Ts.sblock_attr({"schedule_rule": "meta_schedule.pool_avg"})
                    pool_avg[v_ax0, v_ax1, v_ax2, v_ax3] = pool_sum[v_ax0, v_ax1, v_ax2, v_ax3] / T.Cast("float32", T.max((T.min(v_ax1 * T.int64(2) + T.int64(1), T.int64(111)) + T.int64(2) - T.max(T.int64(1) - v_ax1 * T.int64(2), T.int64(0)) - v_ax1 * T.int64(2)) * (T.min(v_ax2 * T.int64(2) + T.int64(1), T.int64(111)) + T.int64(2) - T.max(T.int64(1) - v_ax2 * T.int64(2), T.int64(0)) - v_ax2 * T.int64(2)), T.int64(1)))

        @R.function
        def main(x: R.Tensor((4, 112, 112, 6), dtype="float32")) -> R.Tensor((4, 56, 56, 6), dtype="float32"):
            gv = R.call_tir(Expected.avg_pool2d, (x,), ty_args=[R.Tensor((4, 56, 56, 6), dtype="float32")])
            return gv
    # fmt: on

    mod = LegalizeOps()(AvgPool2D)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_avg_pool2d_NCHW16c():
    # fmt: off
    @tvm_tirx.script.ir_module
    class AvgPool2D:
        @R.function
        def main(x: R.Tensor((4, 4, 112, 112, 16), "float32")) -> R.Tensor((4, 4, 110, 110, 16), "float32"):
            gv: R.Tensor((4, 4, 110, 110, 16), "float32") = R.nn.avg_pool2d(x, pool_size=[3, 3], layout="NCHW16c")
            return gv

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def avg_pool2d(rxplaceholder: T.Tensor((T.int64(4), T.int64(4), T.int64(112), T.int64(112), T.int64(16)), "float32"), pool_avg: T.Tensor((T.int64(4), T.int64(4), T.int64(110), T.int64(110), T.int64(16)), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            pool_sum = Ts.sblock_alloc_tensor((T.int64(4), T.int64(4), T.int64(110), T.int64(110), T.int64(16)))
            for ax0, ax1, ax2, ax3, ax4, rv0, rv1 in T.grid(T.int64(4), T.int64(4), T.int64(110), T.int64(110), T.int64(16), T.int64(3), T.int64(3)):
                with Ts.sblock("pool_sum"):
                    v_ax0, v_ax1, v_ax2, v_ax3, v_ax4, v_rv0, v_rv1 = Ts.axis.remap("SSSSSRR", [ax0, ax1, ax2, ax3, ax4, rv0, rv1])
                    Ts.reads(rxplaceholder[v_ax0, v_ax1, v_ax2 + v_rv0, v_ax3 + v_rv1, v_ax4])
                    Ts.writes(pool_sum[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
                    with Ts.init():
                        pool_sum[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = T.float32(0)
                    pool_sum[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = pool_sum[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] + rxplaceholder[v_ax0, v_ax1, v_ax2 + v_rv0, v_ax3 + v_rv1, v_ax4]
            for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(4), T.int64(4), T.int64(110), T.int64(110), T.int64(16)):
                with Ts.sblock("pool_avg"):
                    v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = Ts.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
                    Ts.reads(pool_sum[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
                    Ts.writes(pool_avg[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
                    Ts.sblock_attr({"schedule_rule": "meta_schedule.pool_avg"})
                    pool_avg[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = pool_sum[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] / T.Cast("float32", T.max((T.min(T.int64(2), T.int64(111) - v_ax2) + T.int64(1) - T.max(T.int64(0) - v_ax2, T.int64(0))) * (T.min(T.int64(2), T.int64(111) - v_ax3) + T.int64(1) - T.max(T.int64(0) - v_ax3, T.int64(0))), T.int64(1)))
        @R.function
        def main(x: R.Tensor((4, 4, 112, 112, 16), dtype="float32")) -> R.Tensor((4, 4, 110, 110, 16), dtype="float32"):
            gv = R.call_tir(Expected.avg_pool2d, (x,), ty_args=[R.Tensor((4, 4, 110, 110, 16), dtype="float32")])
            return gv
    # fmt: on

    mod = LegalizeOps()(AvgPool2D)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_avg_pool2d_ceil_mode():
    # fmt: off
    @tvm_tirx.script.ir_module
    class AvgPool2D:
        @R.function
        def main(x: R.Tensor((4, 6, 112, 112), "float32")) -> R.Tensor((4, 6, 38, 38), "float32"):
            gv: R.Tensor((4, 6, 38, 38), "float32") = R.nn.avg_pool2d(x, pool_size=[3, 3], strides=[3, 3], dilation=[1, 1], padding=[1, 1, 1, 1], ceil_mode=True)
            return gv

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def avg_pool2d(rxplaceholder: T.Tensor((T.int64(4), T.int64(6), T.int64(112), T.int64(112)), "float32"), pool_avg: T.Tensor((T.int64(4), T.int64(6), T.int64(38), T.int64(38)), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            pad_temp = Ts.sblock_alloc_tensor((T.int64(4), T.int64(6), T.int64(116), T.int64(116)))
            pool_sum = Ts.sblock_alloc_tensor((T.int64(4), T.int64(6), T.int64(38), T.int64(38)))
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(4), T.int64(6), T.int64(116), T.int64(116)):
                with Ts.sblock("pad_temp"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(rxplaceholder[v_ax0, v_ax1, v_ax2 - T.int64(1), v_ax3 - T.int64(1)])
                    Ts.writes(pad_temp[v_ax0, v_ax1, v_ax2, v_ax3])
                    pad_temp[v_ax0, v_ax1, v_ax2, v_ax3] = T.if_then_else(T.int64(1) <= v_ax2 and v_ax2 < T.int64(113) and T.int64(1) <= v_ax3 and v_ax3 < T.int64(113), rxplaceholder[v_ax0, v_ax1, v_ax2 - T.int64(1), v_ax3 - T.int64(1)], T.float32(0))
            for ax0, ax1, ax2, ax3, rv0, rv1 in T.grid(T.int64(4), T.int64(6), T.int64(38), T.int64(38), T.int64(3), T.int64(3)):
                with Ts.sblock("pool_sum"):
                    v_ax0, v_ax1, v_ax2, v_ax3, v_rv0, v_rv1 = Ts.axis.remap("SSSSRR", [ax0, ax1, ax2, ax3, rv0, rv1])
                    Ts.reads(pad_temp[v_ax0, v_ax1, v_ax2 * T.int64(3) + v_rv0, v_ax3 * T.int64(3) + v_rv1])
                    Ts.writes(pool_sum[v_ax0, v_ax1, v_ax2, v_ax3])
                    with Ts.init():
                        pool_sum[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0)
                    pool_sum[v_ax0, v_ax1, v_ax2, v_ax3] = pool_sum[v_ax0, v_ax1, v_ax2, v_ax3] + pad_temp[v_ax0, v_ax1, v_ax2 * T.int64(3) + v_rv0, v_ax3 * T.int64(3) + v_rv1]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(4), T.int64(6), T.int64(38), T.int64(38)):
                with Ts.sblock("pool_avg"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(pool_sum[v_ax0, v_ax1, v_ax2, v_ax3])
                    Ts.writes(pool_avg[v_ax0, v_ax1, v_ax2, v_ax3])
                    Ts.sblock_attr({"schedule_rule": "meta_schedule.pool_avg"})
                    pool_avg[v_ax0, v_ax1, v_ax2, v_ax3] = pool_sum[v_ax0, v_ax1, v_ax2, v_ax3] / T.Cast("float32", T.max((T.min(v_ax2 * T.int64(3) + T.int64(1), T.int64(111)) + T.int64(2) - T.max(T.int64(1) - v_ax2 * T.int64(3), T.int64(0)) - v_ax2 * T.int64(3)) * (T.min(v_ax3 * T.int64(3) + T.int64(1), T.int64(111)) + T.int64(2) - T.max(T.int64(1) - v_ax3 * T.int64(3), T.int64(0)) - v_ax3 * T.int64(3)), T.int64(1)))

        @R.function
        def main(x: R.Tensor((4, 6, 112, 112), dtype="float32")) -> R.Tensor((4, 6, 38, 38), dtype="float32"):
            gv = R.call_tir(Expected.avg_pool2d, (x,), ty_args=[R.Tensor((4, 6, 38, 38), dtype="float32")])
            return gv

    # fmt: on

    mod = LegalizeOps()(AvgPool2D)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


@pytest.mark.skip("TOPI pooling casts every shape value to i32.")
def test_avg_pool2d_symbolic():
    # fmt: off
    n = T.dynamic("n", "int64")
    c = T.dynamic("c", "int64")
    h = T.dynamic("h", "int64")
    w = T.dynamic("w", "int64")
    kh = T.dynamic("kh", "int64")
    kw = T.dynamic("kw", "int64")

    @tvm_tirx.script.ir_module
    class AvgPool2D:
        @R.function
        def main(dumb_param: R.Tensor((kh, kw)), x: R.Tensor((n, c, h, w), "float32")) -> R.Tensor((n, c, h - kh + 1, w - kw + 1), "float32"):
            gv: R.Tensor((n, c, h - kh + 1, w - kw + 1), "float32") = R.nn.avg_pool2d(x, pool_size=[kh, kw])
            return gv

    # fmt: on

    mod = LegalizeOps()(AvgPool2D)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_adaptive_avg_pool2d():
    # fmt: off
    @tvm_tirx.script.ir_module
    class AdaptiveAvgPool2D:
        @R.function
        def main(x: R.Tensor((2, 4, 7, 7, 16), "float32")) -> R.Tensor((2, 4, 1, 1, 16), "float32"):
            gv: R.Tensor((2, 4, 1, 1, 16), "float32") = R.nn.adaptive_avg_pool2d(x, output_size=[1, 1], layout="NCHW16c")
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 4, 7, 7, 16), "float32")) -> R.Tensor((2, 4, 1, 1, 16), "float32"):
            gv = R.call_tir(Expected.adaptive_avg_pool2d, (x,), ty_args=[R.Tensor((2, 4, 1, 1, 16), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def adaptive_avg_pool2d(rxplaceholder: T.Tensor((T.int64(2), T.int64(4), T.int64(7), T.int64(7), T.int64(16)), "float32"), adaptive_pool_avg: T.Tensor((T.int64(2), T.int64(4), T.int64(1), T.int64(1), T.int64(16)), "float32")):
            T.func_attr({"tirx.noalias": True})
            adaptive_pool_sum = Ts.sblock_alloc_tensor([T.int64(2), T.int64(4), T.int64(1), T.int64(1), T.int64(16)], dtype="float32")
            for i0, i1, i2, i3, i4, i5, i6 in T.grid(T.int64(2), T.int64(4), T.int64(1), T.int64(1), T.int64(16), T.int64(7), T.int64(7)):
                with Ts.sblock("adaptive_pool_sum"):
                    ax0, ax1, ax2, ax3, ax4, rv0, rv1 = Ts.axis.remap("SSSSSRR", [i0, i1, i2, i3, i4, i5, i6])
                    Ts.reads(rxplaceholder[ax0, ax1, ax2 * T.int64(7) + rv0, ax3 * T.int64(7) + rv1, ax4])
                    Ts.writes(adaptive_pool_sum[ax0, ax1, ax2, ax3, ax4])
                    with Ts.init():
                        adaptive_pool_sum[ax0, ax1, ax2, ax3, ax4] = T.float32(0)
                    adaptive_pool_sum[ax0, ax1, ax2, ax3, ax4] = adaptive_pool_sum[ax0, ax1, ax2, ax3, ax4] + rxplaceholder[ax0, ax1, ax2 * T.int64(7) + rv0, ax3 * T.int64(7) + rv1, ax4]
            for i0, i1, i2, i3, i4 in T.grid(T.int64(2), T.int64(4), T.int64(1), T.int64(1), T.int64(16)):
                with Ts.sblock("adaptive_pool_avg"):
                    ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                    Ts.reads(adaptive_pool_sum[ax0, ax1, ax2, ax3, ax4])
                    Ts.writes(adaptive_pool_avg[ax0, ax1, ax2, ax3, ax4])
                    Ts.sblock_attr({"schedule_rule":"meta_schedule.adaptive_pool_avg"})
                    adaptive_pool_avg[ax0, ax1, ax2, ax3, ax4] = adaptive_pool_sum[ax0, ax1, ax2, ax3, ax4] / T.float32(49.0)
    # fmt: on

    mod = LegalizeOps()(AdaptiveAvgPool2D)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_adaptive_avg_pool2d_without_output_size():
    # fmt: off
    @tvm_tirx.script.ir_module
    class AdaptiveAvgPool2D:
        @R.function
        def main(x: R.Tensor((2, 16, 7, 7), "float32")) -> R.Tensor((2, 16, 7, 7), "float32"):
            gv: R.Tensor((2, 16, 7, 7), "float32") = R.nn.adaptive_avg_pool2d(x)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 16, 7, 7), "float32")) -> R.Tensor((2, 16, 7, 7), "float32"):
            gv = R.call_tir(Expected.adaptive_avg_pool2d, (x,), ty_args=[R.Tensor((2, 16, 7, 7), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def adaptive_avg_pool2d(rxplaceholder: T.Tensor((T.int64(2), T.int64(16), T.int64(7), T.int64(7)), "float32"), adaptive_pool_avg: T.Tensor((T.int64(2), T.int64(16), T.int64(7), T.int64(7)), "float32")):
            T.func_attr({"tirx.noalias": True})
            adaptive_pool_sum = Ts.sblock_alloc_tensor([T.int64(2), T.int64(16), T.int64(7), T.int64(7)], dtype="float32")
            for i0, i1, i2, i3, i4, i5 in T.grid(T.int64(2), T.int64(16), T.int64(7), T.int64(7), T.int64(1), T.int64(1)):
                with Ts.sblock("adaptive_pool_sum"):
                    ax0, ax1, ax2, ax3, rv0, rv1 = Ts.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                    Ts.reads(rxplaceholder[ax0, ax1, ax2 + rv0, ax3 + rv1])
                    Ts.writes(adaptive_pool_sum[ax0, ax1, ax2, ax3])
                    with Ts.init():
                        adaptive_pool_sum[ax0, ax1, ax2, ax3] = T.float32(0)
                    adaptive_pool_sum[ax0, ax1, ax2, ax3] = adaptive_pool_sum[ax0, ax1, ax2, ax3] + rxplaceholder[ax0, ax1, ax2 + rv0, ax3 + rv1]
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(16), T.int64(7), T.int64(7)):
                with Ts.sblock("adaptive_pool_avg"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(adaptive_pool_sum[ax0, ax1, ax2, ax3])
                    Ts.writes(adaptive_pool_avg[ax0, ax1, ax2, ax3])
                    Ts.sblock_attr({"schedule_rule":"meta_schedule.adaptive_pool_avg"})
                    adaptive_pool_avg[ax0, ax1, ax2, ax3] = adaptive_pool_sum[ax0, ax1, ax2, ax3]
    # fmt: on

    mod = LegalizeOps()(AdaptiveAvgPool2D)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


@pytest.mark.skip("TOPI pooling casts every shape value to i32.")
def test_adaptive_avg_pool2d_symbolic():
    # fmt: off
    n = T.dynamic("n", "int64")
    c = T.dynamic("c", "int64")
    oh = T.dynamic("oh", "int64")
    ow = T.dynamic("ow", "int64")
    h = T.dynamic("h", "int64")
    w = T.dynamic("w", "int64")

    @tvm_tirx.script.ir_module
    class AdaptiveAvgPool2D:
        @R.function
        def main(dumb_param: R.Tensor((oh, ow)), x: R.Tensor((n, c, h, w), "float32")) -> R.Tensor((n, c, oh, ow), "float32"):
            gv: R.Tensor((n, c, oh, ow), "float32") = R.nn.adaptive_avg_pool2d(x, (oh, ow))
            return gv
    # fmt: on

    mod = LegalizeOps()(AdaptiveAvgPool2D)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_relu():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Relu:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), "float32") = R.nn.relu(x)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(Expected.relu, (x,), ty_args=[R.Tensor((2, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def relu(rxplaceholder: T.Tensor((T.int64(2), T.int64(3)), "float32"), compute: T.Tensor((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("compute"):
                    i0_1, i1_1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[i0_1, i1_1])
                    Ts.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.max(rxplaceholder[i0_1, i1_1], T.float32(0))
    # fmt: on

    mod = LegalizeOps()(Relu)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_relu_symbolic():
    # fmt: off
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Relu:
        @R.function
        def main(x: R.Tensor((m, n), "float32")) -> R.Tensor((m, n), "float32"):
            gv: R.Tensor((m, n), "float32") = R.nn.relu(x)
            return gv

    m_main = T.dynamic("m", "int64")
    n_main = T.dynamic("n", "int64")
    m_relu = T.dynamic("m", "int64")
    n_relu = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((m_main, n_main), "float32")) -> R.Tensor((m_main, n_main), "float32"):
            gv = R.call_tir(Expected.relu, (x,), ty_args=[R.Tensor((m_main, n_main), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def relu(rxplaceholder: T.Tensor([m_relu, n_relu], dtype='float32'), compute: T.Tensor([m_relu, n_relu], dtype='float32')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1 in T.grid(m_relu, n_relu):
                with Ts.sblock("compute"):
                    i0_1, i1_1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[i0_1, i1_1])
                    Ts.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.max(rxplaceholder[i0_1, i1_1], T.float32(0))
    # fmt: on

    mod = LegalizeOps()(Relu)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_leakyrelu():
    # fmt: off
    @tvm_tirx.script.ir_module
    class LeakyRelu:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), "float32") = R.nn.leakyrelu(x, 0.02)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(Expected.leaky_relu, (x,), ty_args=[R.Tensor((2, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def leaky_relu(x: T.Tensor((T.int64(2), T.int64(3)), "float32"), compute: T.Tensor((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("compute"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(x[v_i0, v_i1])
                    Ts.writes(compute[v_i0, v_i1])
                    compute[v_i0, v_i1] = T.Select(T.float32(0.0) < x[v_i0, v_i1], x[v_i0, v_i1], x[v_i0, v_i1] * T.float32(0.02))
    # fmt: on

    mod = LegalizeOps()(LeakyRelu)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_leakyrelu_symbolic():
    # fmt: off
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class LeakyRelu:
        @R.function
        def main(x: R.Tensor((m, n), "float32")) -> R.Tensor((m, n), "float32"):
            gv: R.Tensor((m, n), "float32") = R.nn.leakyrelu(x, 0.03)
            return gv

    m_main = T.dynamic("m", "int64")
    n_main = T.dynamic("n", "int64")
    m_leaky_relu = T.dynamic("m", "int64")
    n_leaky_relu = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((m_main, n_main), "float32")) -> R.Tensor((m_main, n_main), "float32"):
            gv = R.call_tir(Expected.leaky_relu, (x, ), ty_args=[R.Tensor((m_main, n_main), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def leaky_relu(x: T.Tensor((m_leaky_relu, n_leaky_relu), "float32"), compute: T.Tensor((m_leaky_relu, n_leaky_relu), "float32")):
            T.func_attr({"tirx.noalias": True})

            for i0, i1 in T.grid(m_leaky_relu, n_leaky_relu):
                with Ts.sblock("compute"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(x[v_i0, v_i1])
                    Ts.writes(compute[v_i0, v_i1])
                    compute[v_i0, v_i1] = T.Select(T.float32(0.0) < x[v_i0, v_i1], x[v_i0, v_i1], x[v_i0, v_i1] * T.float32(0.029999999999999999))
    # fmt: on

    mod = LegalizeOps()(LeakyRelu)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_prelu():
    # fmt: off
    @tvm_tirx.script.ir_module
    class PRelu:
        @R.function
        def main(x: R.Tensor((2, 3), "float32"), y: R.Tensor((1,), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), "float32") = R.nn.prelu(x, y)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((1,), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            gv = R.call_tir(Expected.prelu, (x, y), ty_args=[R.Tensor((2, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def prelu(x: T.Tensor((T.int64(2), T.int64(3)), "float32"), y: T.Tensor((T.int64(1),), "float32"), compute: T.Tensor((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            slope_broadcasted = Ts.sblock_alloc_tensor((T.int64(3),))
            for c_index in range(T.int64(3)):
                with Ts.sblock("slope_broadcasted"):
                    v_c = Ts.axis.spatial(T.int64(3), c_index)
                    Ts.reads(y[T.int64(0)])
                    Ts.writes(slope_broadcasted[v_c])
                    slope_broadcasted[v_c] = y[T.int64(0)]
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("compute"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(x[v_i0, v_i1], slope_broadcasted[v_i1])
                    Ts.writes(compute[v_i0, v_i1])
                    compute[v_i0, v_i1] = T.Select(T.float32(0.0) < x[v_i0, v_i1], x[v_i0, v_i1], x[v_i0, v_i1] * slope_broadcasted[v_i1])
    # fmt: on

    mod = LegalizeOps()(PRelu)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_prelu_symbolic():
    # fmt: off
    m = T.dynamic("m", "int64")

    @tvm_tirx.script.ir_module
    class PRelu:
        @R.function
        def main(x: R.Tensor((m, 7), "float32"), y: R.Tensor((1,), "float32")) -> R.Tensor((m, 7), "float32"):
            gv: R.Tensor((m, 7), "float32") = R.nn.prelu(x, y)
            return gv

    m_main = T.dynamic("m", "int64")
    m_prelu = T.dynamic("m", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((m_main, 7), dtype="float32"), y: R.Tensor((1,), dtype="float32")) -> R.Tensor((m_main, 7), dtype="float32"):
            gv = R.call_tir(Expected.prelu, (x, y), ty_args=[R.Tensor((m_main, 7), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def prelu(x: T.Tensor((m_prelu, T.int64(7)), "float32"), y: T.Tensor((T.int64(1),), "float32"), compute: T.Tensor((m_prelu, T.int64(7)), "float32")):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            slope_broadcasted = Ts.sblock_alloc_tensor((T.int64(7),))
            for c_index in range(T.int64(7)):
                with Ts.sblock("slope_broadcasted"):
                    v_c = Ts.axis.spatial(T.int64(7), c_index)
                    Ts.reads(y[T.int64(0)])
                    Ts.writes(slope_broadcasted[v_c])
                    slope_broadcasted[v_c] = y[T.int64(0)]
            for i0, i1 in T.grid(m_prelu, T.int64(7)):
                with Ts.sblock("compute"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(x[v_i0, v_i1], slope_broadcasted[v_i1])
                    Ts.writes(compute[v_i0, v_i1])
                    compute[v_i0, v_i1] = T.Select(T.float32(0.0) < x[v_i0, v_i1], x[v_i0, v_i1], x[v_i0, v_i1] * slope_broadcasted[v_i1])
    # fmt: on

    mod = LegalizeOps()(PRelu)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_gelu():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Gelu:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), "float32") = R.nn.gelu(x)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(Expected.gelu, (x,), ty_args=[R.Tensor((2, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def gelu(x: T.Tensor((T.int64(2), T.int64(3)), "float32"), T_multiply: T.Tensor((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            T_multiply_1 = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3)))
            compute = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3)))
            T_multiply_2 = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3)))
            T_add = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3)))
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_multiply"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(x[v_ax0, v_ax1])
                    Ts.writes(T_multiply_1[v_ax0, v_ax1])
                    T_multiply_1[v_ax0, v_ax1] = x[v_ax0, v_ax1] * T.float32(0.70710678118654757)
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("compute"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(T_multiply_1[v_i0, v_i1])
                    Ts.writes(compute[v_i0, v_i1])
                    compute[v_i0, v_i1] = T.erf(T_multiply_1[v_i0, v_i1])
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_multiply_1"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(compute[v_ax0, v_ax1])
                    Ts.writes(T_multiply_2[v_ax0, v_ax1])
                    T_multiply_2[v_ax0, v_ax1] = compute[v_ax0, v_ax1] * T.float32(0.5)
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_add"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(T_multiply_2[v_ax0, v_ax1])
                    Ts.writes(T_add[v_ax0, v_ax1])
                    T_add[v_ax0, v_ax1] = T.float32(0.5) + T_multiply_2[v_ax0, v_ax1]
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_multiply_2"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(x[v_ax0, v_ax1], T_add[v_ax0, v_ax1])
                    Ts.writes(T_multiply[v_ax0, v_ax1])
                    T_multiply[v_ax0, v_ax1] = x[v_ax0, v_ax1] * T_add[v_ax0, v_ax1]
    # fmt: on

    mod = LegalizeOps()(Gelu)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_gelu_symbolic():
    # fmt: off
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Gelu:
        @R.function
        def main(x: R.Tensor((m, n), "float32")) -> R.Tensor((m, n), "float32"):
            gv: R.Tensor((m, n), "float32") = R.nn.gelu(x)
            return gv

    m_main = T.dynamic("m", "int64")
    n_main = T.dynamic("n", "int64")
    m_gelu = T.dynamic("m", "int64")
    n_gelu = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((m_main, n_main), "float32")) -> R.Tensor((m_main, n_main), "float32"):
            gv = R.call_tir(Expected.gelu, (x,), ty_args=[R.Tensor((m_main, n_main), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def gelu(x: T.Tensor((m_gelu, n_gelu), "float32"), T_multiply: T.Tensor((m_gelu, n_gelu), "float32")):
            T.func_attr({"tirx.noalias": True})

            T_multiply_1 = Ts.sblock_alloc_tensor((m_gelu, n_gelu))
            compute = Ts.sblock_alloc_tensor((m_gelu, n_gelu))
            T_multiply_2 = Ts.sblock_alloc_tensor((m_gelu, n_gelu))
            T_add = Ts.sblock_alloc_tensor((m_gelu, n_gelu))
            for ax0, ax1 in T.grid(m_gelu, n_gelu):
                with Ts.sblock("T_multiply"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(x[v_ax0, v_ax1])
                    Ts.writes(T_multiply_1[v_ax0, v_ax1])
                    T_multiply_1[v_ax0, v_ax1] = x[v_ax0, v_ax1] * T.float32(0.70710678118654757)
            for i0, i1 in T.grid(m_gelu, n_gelu):
                with Ts.sblock("compute"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(T_multiply_1[v_i0, v_i1])
                    Ts.writes(compute[v_i0, v_i1])
                    compute[v_i0, v_i1] = T.erf(T_multiply_1[v_i0, v_i1])
            for ax0, ax1 in T.grid(m_gelu, n_gelu):
                with Ts.sblock("T_multiply_1"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(compute[v_ax0, v_ax1])
                    Ts.writes(T_multiply_2[v_ax0, v_ax1])
                    T_multiply_2[v_ax0, v_ax1] = compute[v_ax0, v_ax1] * T.float32(0.5)
            for ax0, ax1 in T.grid(m_gelu, n_gelu):
                with Ts.sblock("T_add"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(T_multiply_2[v_ax0, v_ax1])
                    Ts.writes(T_add[v_ax0, v_ax1])
                    T_add[v_ax0, v_ax1] = T.float32(0.5) + T_multiply_2[v_ax0, v_ax1]
            for ax0, ax1 in T.grid(m_gelu, n_gelu):
                with Ts.sblock("T_multiply_2"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(x[v_ax0, v_ax1], T_add[v_ax0, v_ax1])
                    Ts.writes(T_multiply[v_ax0, v_ax1])
                    T_multiply[v_ax0, v_ax1] = x[v_ax0, v_ax1] * T_add[v_ax0, v_ax1]
    # fmt: on

    mod = LegalizeOps()(Gelu)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_gelu_tanh():
    # fmt: off
    @tvm_tirx.script.ir_module
    class GeluTanh:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), "float32") = R.nn.gelu_tanh(x)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            gv = R.call_tir(Expected.gelu_tanh, (x,), ty_args=[R.Tensor((2, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def gelu_tanh(A: T.Tensor((T.int64(2), T.int64(3)), "float32"), T_multiply: T.Tensor((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            T_multiply_1 = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3)))
            T_multiply_2 = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3)))
            T_multiply_3 = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3)))
            T_multiply_4 = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3)))
            T_add = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3)))
            T_multiply_5 = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3)))
            compute = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3)))
            T_add_1 = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3)))
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_multiply"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(A[v_ax0, v_ax1])
                    Ts.writes(T_multiply_1[v_ax0, v_ax1])
                    T_multiply_1[v_ax0, v_ax1] = T.float32(0.5) * A[v_ax0, v_ax1]
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_multiply_1"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(A[v_ax0, v_ax1])
                    Ts.writes(T_multiply_2[v_ax0, v_ax1])
                    T_multiply_2[v_ax0, v_ax1] = T.float32(0.79788456080286541) * A[v_ax0, v_ax1]
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_multiply_2"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(A[v_ax0, v_ax1])
                    Ts.writes(T_multiply_3[v_ax0, v_ax1])
                    T_multiply_3[v_ax0, v_ax1] = T.float32(0.044714999999999998) * A[v_ax0, v_ax1]
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_multiply_3"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(T_multiply_3[v_ax0, v_ax1], A[v_ax0, v_ax1])
                    Ts.writes(T_multiply_4[v_ax0, v_ax1])
                    T_multiply_4[v_ax0, v_ax1] = T_multiply_3[v_ax0, v_ax1] * A[v_ax0, v_ax1]
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_add"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(T_multiply_4[v_ax0, v_ax1])
                    Ts.writes(T_add[v_ax0, v_ax1])
                    T_add[v_ax0, v_ax1] = T.float32(1) + T_multiply_4[v_ax0, v_ax1]
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_multiply_4"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(T_multiply_2[v_ax0, v_ax1], T_add[v_ax0, v_ax1])
                    Ts.writes(T_multiply_5[v_ax0, v_ax1])
                    T_multiply_5[v_ax0, v_ax1] = T_multiply_2[v_ax0, v_ax1] * T_add[v_ax0, v_ax1]
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("compute"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(T_multiply_5[v_i0, v_i1])
                    Ts.writes(compute[v_i0, v_i1])
                    compute[v_i0, v_i1] = T.tanh(T_multiply_5[v_i0, v_i1])
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_add_1"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(compute[v_ax0, v_ax1])
                    Ts.writes(T_add_1[v_ax0, v_ax1])
                    T_add_1[v_ax0, v_ax1] = T.float32(1) + compute[v_ax0, v_ax1]
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_multiply_5"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(T_multiply_1[v_ax0, v_ax1], T_add_1[v_ax0, v_ax1])
                    Ts.writes(T_multiply[v_ax0, v_ax1])
                    T_multiply[v_ax0, v_ax1] = T_multiply_1[v_ax0, v_ax1] * T_add_1[v_ax0, v_ax1]

    mod = LegalizeOps()(GeluTanh)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_gelu_tanh_symbolic():
    # fmt: off
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class GeluTanh:
        @R.function
        def main(x: R.Tensor((m, n), "float32")) -> R.Tensor((m, n), "float32"):
            gv: R.Tensor((m, n), "float32") = R.nn.gelu_tanh(x)
            return gv

    m_main = T.dynamic("m", "int64")
    n_main = T.dynamic("n", "int64")
    m_gelu_tanh = T.dynamic("m", "int64")
    n_gelu_tanh = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((m_main, n_main), dtype="float32")) -> R.Tensor((m_main, n_main), dtype="float32"):
            gv = R.call_tir(Expected.gelu_tanh, (x,), ty_args=[R.Tensor((m_main, n_main), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def gelu_tanh(A: T.Tensor((m_gelu_tanh, n_gelu_tanh), "float32"), T_multiply: T.Tensor((m_gelu_tanh, n_gelu_tanh), "float32")):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            T_multiply_1 = Ts.sblock_alloc_tensor((m_gelu_tanh, n_gelu_tanh))
            T_multiply_2 = Ts.sblock_alloc_tensor((m_gelu_tanh, n_gelu_tanh))
            T_multiply_3 = Ts.sblock_alloc_tensor((m_gelu_tanh, n_gelu_tanh))
            T_multiply_4 = Ts.sblock_alloc_tensor((m_gelu_tanh, n_gelu_tanh))
            T_add = Ts.sblock_alloc_tensor((m_gelu_tanh, n_gelu_tanh))
            T_multiply_5 = Ts.sblock_alloc_tensor((m_gelu_tanh, n_gelu_tanh))
            compute = Ts.sblock_alloc_tensor((m_gelu_tanh, n_gelu_tanh))
            T_add_1 = Ts.sblock_alloc_tensor((m_gelu_tanh, n_gelu_tanh))
            for ax0, ax1 in T.grid(m_gelu_tanh, n_gelu_tanh):
                with Ts.sblock("T_multiply"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(A[v_ax0, v_ax1])
                    Ts.writes(T_multiply_1[v_ax0, v_ax1])
                    T_multiply_1[v_ax0, v_ax1] = T.float32(0.5) * A[v_ax0, v_ax1]
            for ax0, ax1 in T.grid(m_gelu_tanh, n_gelu_tanh):
                with Ts.sblock("T_multiply_1"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(A[v_ax0, v_ax1])
                    Ts.writes(T_multiply_2[v_ax0, v_ax1])
                    T_multiply_2[v_ax0, v_ax1] = T.float32(0.79788456080286541) * A[v_ax0, v_ax1]
            for ax0, ax1 in T.grid(m_gelu_tanh, n_gelu_tanh):
                with Ts.sblock("T_multiply_2"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(A[v_ax0, v_ax1])
                    Ts.writes(T_multiply_3[v_ax0, v_ax1])
                    T_multiply_3[v_ax0, v_ax1] = T.float32(0.044714999999999998) * A[v_ax0, v_ax1]
            for ax0, ax1 in T.grid(m_gelu_tanh, n_gelu_tanh):
                with Ts.sblock("T_multiply_3"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(T_multiply_3[v_ax0, v_ax1], A[v_ax0, v_ax1])
                    Ts.writes(T_multiply_4[v_ax0, v_ax1])
                    T_multiply_4[v_ax0, v_ax1] = T_multiply_3[v_ax0, v_ax1] * A[v_ax0, v_ax1]
            for ax0, ax1 in T.grid(m_gelu_tanh, n_gelu_tanh):
                with Ts.sblock("T_add"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(T_multiply_4[v_ax0, v_ax1])
                    Ts.writes(T_add[v_ax0, v_ax1])
                    T_add[v_ax0, v_ax1] = T.float32(1) + T_multiply_4[v_ax0, v_ax1]
            for ax0, ax1 in T.grid(m_gelu_tanh, n_gelu_tanh):
                with Ts.sblock("T_multiply_4"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(T_multiply_2[v_ax0, v_ax1], T_add[v_ax0, v_ax1])
                    Ts.writes(T_multiply_5[v_ax0, v_ax1])
                    T_multiply_5[v_ax0, v_ax1] = T_multiply_2[v_ax0, v_ax1] * T_add[v_ax0, v_ax1]
            for i0, i1 in T.grid(m_gelu_tanh, n_gelu_tanh):
                with Ts.sblock("compute"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(T_multiply_5[v_i0, v_i1])
                    Ts.writes(compute[v_i0, v_i1])
                    compute[v_i0, v_i1] = T.tanh(T_multiply_5[v_i0, v_i1])
            for ax0, ax1 in T.grid(m_gelu_tanh, n_gelu_tanh):
                with Ts.sblock("T_add_1"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(compute[v_ax0, v_ax1])
                    Ts.writes(T_add_1[v_ax0, v_ax1])
                    T_add_1[v_ax0, v_ax1] = T.float32(1) + compute[v_ax0, v_ax1]
            for ax0, ax1 in T.grid(m_gelu_tanh, n_gelu_tanh):
                with Ts.sblock("T_multiply_5"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(T_multiply_1[v_ax0, v_ax1], T_add_1[v_ax0, v_ax1])
                    Ts.writes(T_multiply[v_ax0, v_ax1])
                    T_multiply[v_ax0, v_ax1] = T_multiply_1[v_ax0, v_ax1] * T_add_1[v_ax0, v_ax1]

    mod = LegalizeOps()(GeluTanh)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_silu():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Silu:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), "float32") = R.nn.silu(x)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(Expected.silu, (x,), ty_args=[R.Tensor((2, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def silu(rxplaceholder: T.Tensor((T.int64(2), T.int64(3)), "float32"), T_multiply: T.Tensor((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            compute = Ts.sblock_alloc_tensor([T.int64(2), T.int64(3)], dtype="float32")
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("compute"):
                    i0_1, i1_1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[i0_1, i1_1])
                    Ts.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.sigmoid(rxplaceholder[i0_1, i1_1])
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_multiply"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[ax0, ax1], compute[ax0, ax1])
                    Ts.writes(T_multiply[ax0, ax1])
                    T_multiply[ax0, ax1] = rxplaceholder[ax0, ax1] * compute[ax0, ax1]
    # fmt: on

    mod = LegalizeOps()(Silu)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_silu_symbolic():
    # fmt: off
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Silu:
        @R.function
        def main(x: R.Tensor((m, n), "float32")) -> R.Tensor((m, n), "float32"):
            gv: R.Tensor((m, n), "float32") = R.nn.silu(x)
            return gv

    m_main = T.dynamic("m", "int64")
    n_main = T.dynamic("n", "int64")
    m_silu = T.dynamic("m", "int64")
    n_silu = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((m_main, n_main), "float32")) -> R.Tensor((m_main, n_main), "float32"):
            gv = R.call_tir(Expected.silu, (x,), ty_args=[R.Tensor((m_main, n_main), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def silu(rxplaceholder: T.Tensor([m_silu, n_silu], dtype='float32'), T_multiply: T.Tensor([m_silu, n_silu], dtype='float32')):
            T.func_attr({"tirx.noalias": True})

            compute = Ts.sblock_alloc_tensor([m_silu, n_silu], dtype="float32")
            for i0, i1 in T.grid(m_silu, n_silu):
                with Ts.sblock("compute"):
                    i0_1, i1_1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[i0_1, i1_1])
                    Ts.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.sigmoid(rxplaceholder[i0_1, i1_1])
            for i0, i1 in T.grid(m_silu, n_silu):
                with Ts.sblock("T_multiply"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[ax0, ax1], compute[ax0, ax1])
                    Ts.writes(T_multiply[ax0, ax1])
                    T_multiply[ax0, ax1] = rxplaceholder[ax0, ax1] * compute[ax0, ax1]
    # fmt: on

    mod = LegalizeOps()(Silu)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_softmax():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Softmax:
        @R.function
        def main(x: R.Tensor((2, 3, 16, 32), "float32")) -> R.Tensor((2, 3, 16, 32), "float32"):
            gv: R.Tensor((2, 3, 16, 32), "float32") = R.nn.softmax(x, axis=-2)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 16, 32), "float32")) -> R.Tensor((2, 3, 16, 32), "float32"):
            gv = R.call_tir(Expected.softmax, (x,), ty_args=[R.Tensor((2, 3, 16, 32), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def softmax(rxplaceholder: T.Tensor((T.int64(2), T.int64(3), T.int64(16), T.int64(32)), "float32"), T_softmax_norm: T.Tensor((T.int64(2), T.int64(3), T.int64(16), T.int64(32)), "float32")):
            T.func_attr({"tirx.noalias": True})
            T_softmax_maxelem = Ts.sblock_alloc_tensor([T.int64(2), T.int64(3), T.int64(32)], dtype="float32")
            T_softmax_exp = Ts.sblock_alloc_tensor([T.int64(2), T.int64(3), T.int64(16), T.int64(32)], dtype="float32")
            T_softmax_expsum = Ts.sblock_alloc_tensor([T.int64(2), T.int64(3), T.int64(32)], dtype="float32")
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(32), T.int64(16)):
                with Ts.sblock("T_softmax_maxelem"):
                    i0_1, i1_1, i2_1, k = Ts.axis.remap("SSSR", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[i0_1, i1_1, k, i2_1])
                    Ts.writes(T_softmax_maxelem[i0_1, i1_1, i2_1])
                    with Ts.init():
                        T_softmax_maxelem[i0_1, i1_1, i2_1] = T.float32(-3.4028234663852886e+38)
                    T_softmax_maxelem[i0_1, i1_1, i2_1] = T.max(T_softmax_maxelem[i0_1, i1_1, i2_1], rxplaceholder[i0_1, i1_1, k, i2_1])
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(16), T.int64(32)):
                with Ts.sblock("T_softmax_exp"):
                    i0_2, i1_2, i2_2, i3_1 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[i0_2, i1_2, i2_2, i3_1], T_softmax_maxelem[i0_2, i1_2, i3_1])
                    Ts.writes(T_softmax_exp[i0_2, i1_2, i2_2, i3_1])
                    T_softmax_exp[i0_2, i1_2, i2_2, i3_1] = T.exp(rxplaceholder[i0_2, i1_2, i2_2, i3_1] - T_softmax_maxelem[i0_2, i1_2, i3_1])
            for i0_3, i1_3, i2_3, i3 in T.grid(T.int64(2), T.int64(3), T.int64(32), T.int64(16)):
                with Ts.sblock("T_softmax_expsum"):
                    i0_4, i1_4, i2_4, k = Ts.axis.remap("SSSR", [i0_3, i1_3, i2_3, i3])
                    Ts.reads(T_softmax_exp[i0_4, i1_4, k, i2_4])
                    Ts.writes(T_softmax_expsum[i0_4, i1_4, i2_4])
                    with Ts.init():
                        T_softmax_expsum[i0_4, i1_4, i2_4] = T.float32(0)
                    T_softmax_expsum[i0_4, i1_4, i2_4] = T_softmax_expsum[i0_4, i1_4, i2_4] + T_softmax_exp[i0_4, i1_4, k, i2_4]
            for i0_5, i1_5, i2_5, i3 in T.grid(T.int64(2), T.int64(3), T.int64(16), T.int64(32)):
                with Ts.sblock("T_softmax_norm"):
                    i0_6, i1_6, i2_6, i3_2 = Ts.axis.remap("SSSS", [i0_5, i1_5, i2_5, i3])
                    Ts.reads(T_softmax_exp[i0_6, i1_6, i2_6, i3_2], T_softmax_expsum[i0_6, i1_6, i3_2])
                    Ts.writes(T_softmax_norm[i0_6, i1_6, i2_6, i3_2])
                    Ts.sblock_attr({"axis":2})
                    T_softmax_norm[i0_6, i1_6, i2_6, i3_2] = T_softmax_exp[i0_6, i1_6, i2_6, i3_2] / T_softmax_expsum[i0_6, i1_6, i3_2]
    # fmt: on

    mod = LegalizeOps()(Softmax)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_softmax_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")
    c = T.dynamic("c", "int64")

    @tvm_tirx.script.ir_module
    class Softmax:
        @R.function
        def main(x: R.Tensor((a, b, c), "float32")) -> R.Tensor((a, b, c), "float32"):
            gv: R.Tensor((a, b, c), "float32") = R.nn.softmax(x)
            return gv

    a_main = T.dynamic("a", "int64")
    b_main = T.dynamic("b", "int64")
    c_main = T.dynamic("c", "int64")
    a_softmax = T.dynamic("a", "int64")
    b_softmax = T.dynamic("b", "int64")
    c_softmax = T.dynamic("c", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((a_main, b_main, c_main), "float32")) -> R.Tensor((a_main, b_main, c_main), "float32"):
            gv = R.call_tir(Expected.softmax, (x,), ty_args=[R.Tensor((a_main, b_main, c_main), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def softmax(rxplaceholder: T.Tensor([a_softmax, b_softmax, c_softmax], dtype='float32'), T_softmax_norm: T.Tensor([a_softmax, b_softmax, c_softmax], dtype='float32')):
            T.func_attr({"tirx.noalias": True})

            T_softmax_maxelem = Ts.sblock_alloc_tensor([a_softmax, b_softmax], dtype="float32")
            T_softmax_exp = Ts.sblock_alloc_tensor([a_softmax, b_softmax, c_softmax], dtype="float32")
            T_softmax_expsum = Ts.sblock_alloc_tensor([a_softmax, b_softmax], dtype="float32")
            for i0, i1, i2 in T.grid(a_softmax, b_softmax, c_softmax):
                with Ts.sblock("T_softmax_maxelem"):
                    i0_1, i1_1, k = Ts.axis.remap("SSR", [i0, i1, i2])
                    Ts.reads(rxplaceholder[i0_1, i1_1, k])
                    Ts.writes(T_softmax_maxelem[i0_1, i1_1])
                    with Ts.init():
                        T_softmax_maxelem[i0_1, i1_1] = T.float32(-3.4028234663852886e+38)
                    T_softmax_maxelem[i0_1, i1_1] = T.max(T_softmax_maxelem[i0_1, i1_1], rxplaceholder[i0_1, i1_1, k])
            for i0, i1, i2 in T.grid(a_softmax, b_softmax, c_softmax):
                with Ts.sblock("T_softmax_exp"):
                    i0_2, i1_2, i2_1 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(rxplaceholder[i0_2, i1_2, i2_1], T_softmax_maxelem[i0_2, i1_2])
                    Ts.writes(T_softmax_exp[i0_2, i1_2, i2_1])
                    T_softmax_exp[i0_2, i1_2, i2_1] = T.exp(rxplaceholder[i0_2, i1_2, i2_1] - T_softmax_maxelem[i0_2, i1_2])
            for i0_3, i1_3, i2 in T.grid(a_softmax, b_softmax, c_softmax):
                with Ts.sblock("T_softmax_expsum"):
                    i0_4, i1_4, k = Ts.axis.remap("SSR", [i0_3, i1_3, i2])
                    Ts.reads(T_softmax_exp[i0_4, i1_4, k])
                    Ts.writes(T_softmax_expsum[i0_4, i1_4])
                    with Ts.init():
                        T_softmax_expsum[i0_4, i1_4] = T.float32(0)
                    T_softmax_expsum[i0_4, i1_4] = T_softmax_expsum[i0_4, i1_4] + T_softmax_exp[i0_4, i1_4, k]
            for i0_5, i1_5, i2 in T.grid(a_softmax, b_softmax, c_softmax):
                with Ts.sblock("T_softmax_norm"):
                    i0_6, i1_6, i2_2 = Ts.axis.remap("SSS", [i0_5, i1_5, i2])
                    Ts.reads(T_softmax_exp[i0_6, i1_6, i2_2], T_softmax_expsum[i0_6, i1_6])
                    Ts.writes(T_softmax_norm[i0_6, i1_6, i2_2])
                    Ts.sblock_attr({"axis":2})
                    T_softmax_norm[i0_6, i1_6, i2_2] = T_softmax_exp[i0_6, i1_6, i2_2] / T_softmax_expsum[i0_6, i1_6]
    # fmt: on

    mod = LegalizeOps()(Softmax)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_log_softmax():
    # fmt: off
    @tvm_tirx.script.ir_module
    class LogSoftmax:
        @R.function
        def main(x: R.Tensor((2, 3, 16, 32), "float32")) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((2, 3, 16, 32), "float32") = R.nn.log_softmax(x, axis=-2)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 16, 32), dtype="float32")) -> R.Tensor((2, 3, 16, 32), dtype="float32"):
            gv = R.call_tir(Expected.log_softmax, (x,), ty_args=[R.Tensor((2, 3, 16, 32), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def log_softmax(rxplaceholder: T.Tensor((T.int64(2), T.int64(3), T.int64(16), T.int64(32)), "float32"), compute: T.Tensor((T.int64(2), T.int64(3), T.int64(16), T.int64(32)), "float32"),):
            T.func_attr({"tirx.noalias": True})
            T_softmax_maxelem = Ts.sblock_alloc_tensor([T.int64(2), T.int64(3), T.int64(32)], dtype="float32")
            compute_1 = Ts.sblock_alloc_tensor([T.int64(2), T.int64(3), T.int64(32)], dtype="float32")
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(32), T.int64(16)):
                with Ts.sblock("T_softmax_maxelem"):
                    i0_1, i1_1, i2_1, k = Ts.axis.remap("SSSR", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[i0_1, i1_1, k, i2_1])
                    Ts.writes(T_softmax_maxelem[i0_1, i1_1, i2_1])
                    with Ts.init():
                        T_softmax_maxelem[i0_1, i1_1, i2_1] = T.float32(-3.4028234663852886e38)
                    T_softmax_maxelem[i0_1, i1_1, i2_1] = T.max(T_softmax_maxelem[i0_1, i1_1, i2_1], rxplaceholder[i0_1, i1_1, k, i2_1])
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(32), T.int64(16)):
                with Ts.sblock("compute"):
                    i0_2, i1_2, i2_2, k = Ts.axis.remap("SSSR", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[i0_2, i1_2, k, i2_2], T_softmax_maxelem[i0_2, i1_2, i2_2])
                    Ts.writes(compute_1[i0_2, i1_2, i2_2])
                    with Ts.init():
                        compute_1[i0_2, i1_2, i2_2] = T.float32(0)
                    compute_1[i0_2, i1_2, i2_2] = compute_1[i0_2, i1_2, i2_2] + T.exp(rxplaceholder[i0_2, i1_2, k, i2_2] - T_softmax_maxelem[i0_2, i1_2, i2_2])
            for i0_3, i1_3, i2_3, i3 in T.grid(T.int64(2), T.int64(3), T.int64(16), T.int64(32)):
                with Ts.sblock("compute_1"):
                    i0_4, i1_4, i2_4, i3_1 = Ts.axis.remap("SSSS", [i0_3, i1_3, i2_3, i3])
                    Ts.reads(rxplaceholder[i0_4, i1_4, i2_4, i3_1], T_softmax_maxelem[i0_4, i1_4, i3_1], compute_1[i0_4, i1_4, i3_1])
                    Ts.writes(compute[i0_4, i1_4, i2_4, i3_1])
                    Ts.sblock_attr({"axis": 2})
                    compute[i0_4, i1_4, i2_4, i3_1] = (rxplaceholder[i0_4, i1_4, i2_4, i3_1] - T_softmax_maxelem[i0_4, i1_4, i3_1] - T.log(compute_1[i0_4, i1_4, i3_1]))
    # fmt: on

    mod = LegalizeOps()(LogSoftmax)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_log_softmax_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")
    c = T.dynamic("c", "int64")

    @tvm_tirx.script.ir_module
    class LogSoftmax:
        @R.function
        def main(x: R.Tensor((a, b, c), "float32")) -> R.Tensor((a, b, c), "float32"):
            gv: R.Tensor((a, b, c), "float32") = R.nn.log_softmax(x)
            return gv

    a_main = T.dynamic("a", "int64")
    b_main = T.dynamic("b", "int64")
    c_main = T.dynamic("c", "int64")
    a_log_softmax = T.dynamic("a", "int64")
    b_log_softmax = T.dynamic("b", "int64")
    c_log_softmax = T.dynamic("c", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((a_main, b_main, c_main), dtype="float32")) -> R.Tensor((a_main, b_main, c_main), dtype="float32"):
            # block 0
            gv = R.call_tir(Expected.log_softmax, (x,), ty_args=[R.Tensor((a_main, b_main, c_main), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def log_softmax(rxplaceholder: T.Tensor([a_log_softmax, b_log_softmax, c_log_softmax], dtype='float32'), compute: T.Tensor([a_log_softmax, b_log_softmax, c_log_softmax], dtype='float32')):
            T.func_attr({"tirx.noalias": True})

            T_softmax_maxelem = Ts.sblock_alloc_tensor([a_log_softmax, b_log_softmax], dtype="float32")
            compute_1 = Ts.sblock_alloc_tensor([a_log_softmax, b_log_softmax], dtype="float32")
            for i0, i1, k in T.grid(a_log_softmax, b_log_softmax, c_log_softmax):
                with Ts.sblock("T_softmax_maxelem"):
                    v_i0, v_i1, v_k = Ts.axis.remap("SSR", [i0, i1, k])
                    Ts.reads(rxplaceholder[v_i0, v_i1, v_k])
                    Ts.writes(T_softmax_maxelem[v_i0, v_i1])
                    with Ts.init():
                        T_softmax_maxelem[v_i0, v_i1] = T.float32(-3.4028234663852886e38)
                    T_softmax_maxelem[v_i0, v_i1] = T.max(T_softmax_maxelem[v_i0, v_i1], rxplaceholder[v_i0, v_i1, v_k])
            for i0, i1, k in T.grid(a_log_softmax, b_log_softmax, c_log_softmax):
                with Ts.sblock("compute"):
                    v_i0, v_i1, v_k = Ts.axis.remap("SSR", [i0, i1, k])
                    Ts.reads(rxplaceholder[v_i0, v_i1, v_k], T_softmax_maxelem[v_i0, v_i1])
                    Ts.writes(compute_1[v_i0, v_i1])
                    with Ts.init():
                        compute_1[v_i0, v_i1] = T.float32(0)
                    compute_1[v_i0, v_i1] = compute_1[v_i0, v_i1] + T.exp(rxplaceholder[v_i0, v_i1, v_k] - T_softmax_maxelem[v_i0, v_i1])
            for i0, i1, i2 in T.grid(a_log_softmax, b_log_softmax, c_log_softmax):
                with Ts.sblock("compute_1"):
                    v_i0, v_i1, v_i2 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(rxplaceholder[v_i0, v_i1, v_i2], T_softmax_maxelem[v_i0, v_i1], compute_1[v_i0, v_i1],)
                    Ts.writes(compute[v_i0, v_i1, v_i2])
                    Ts.sblock_attr({"axis": 2})
                    compute[v_i0, v_i1, v_i2] = (rxplaceholder[v_i0, v_i1, v_i2] - T_softmax_maxelem[v_i0, v_i1] - T.log(compute_1[v_i0, v_i1]))
    # fmt: on

    mod = LegalizeOps()(LogSoftmax)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_cross_entropy_with_logits():
    # fmt: off
    @tvm_tirx.script.ir_module
    class CrossEntropyWithLogits:
        @R.function
        def main(x: R.Tensor((3,), "float32"), y: R.Tensor((3,), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((), "float32") = R.nn.cross_entropy_with_logits(x, y)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((3,), dtype="float32"), y: R.Tensor((3,), dtype="float32")):
            gv = R.call_tir(Expected.cross_entropy_with_logits, (x, y), ty_args=[R.Tensor((), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def cross_entropy_with_logits(x: T.Tensor((T.int64(3),), "float32"), y: T.Tensor((T.int64(3),), "float32"), T_multiply: T.Tensor((), "float32")):
            T.func_attr({"tirx.noalias": True})
            T_multiply_1 = Ts.sblock_alloc_tensor((T.int64(3),))
            T_multiply_red = Ts.sblock_alloc_tensor(())
            for ax0 in range(T.int64(3)):
                with Ts.sblock("T_multiply"):
                    v_ax0 = Ts.axis.spatial(T.int64(3), ax0)
                    Ts.reads(x[v_ax0], y[v_ax0])
                    Ts.writes(T_multiply_1[v_ax0])
                    T_multiply_1[v_ax0] = x[v_ax0] * y[v_ax0]
            for k0 in range(T.int64(3)):
                with Ts.sblock("T_multiply_red"):
                    v_k0 = Ts.axis.reduce(T.int64(3), k0)
                    Ts.reads(T_multiply_1[v_k0])
                    Ts.writes(T_multiply_red[()])
                    with Ts.init():
                        T_multiply_red[()] = T.float32(0.0)
                    T_multiply_red[()] = T_multiply_red[()] + T_multiply_1[v_k0]
            with Ts.sblock("T_multiply_1"):
                vi = Ts.axis.spatial(T.int64(1), T.int64(0))
                Ts.reads(T_multiply_red[()])
                Ts.writes(T_multiply[()])
                T_multiply[()] = T_multiply_red[()] * T.float32(-1.0)
    # fmt: on

    mod = LegalizeOps()(CrossEntropyWithLogits)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_cross_entropy_with_logits_batch():
    # fmt: off
    @tvm_tirx.script.ir_module
    class CrossEntropyWithLogits:
        @R.function
        def main(x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 3), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((), "float32") = R.nn.cross_entropy_with_logits(x, y)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")):
            gv = R.call_tir(Expected.cross_entropy_with_logits, (x, y), ty_args=[R.Tensor((), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def cross_entropy_with_logits(x: T.Tensor((T.int64(2), T.int64(3)), "float32"), y: T.Tensor((T.int64(2), T.int64(3)), "float32"), T_divide: T.Tensor((), "float32")):
            T.func_attr({"tirx.noalias": True})
            T_multiply = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3)))
            T_multiply_red = Ts.sblock_alloc_tensor(())
            T_multiply_1 = Ts.sblock_alloc_tensor(())
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_multiply"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(x[v_ax0, v_ax1], y[v_ax0, v_ax1])
                    Ts.writes(T_multiply[v_ax0, v_ax1])
                    T_multiply[v_ax0, v_ax1] = x[v_ax0, v_ax1] * y[v_ax0, v_ax1]
            for k0, k1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_multiply_red"):
                    v_k0, v_k1 = Ts.axis.remap("RR", [k0, k1])
                    Ts.reads(T_multiply[v_k0, v_k1])
                    Ts.writes(T_multiply_red[()])
                    with Ts.init():
                        T_multiply_red[()] = T.float32(0.0)
                    T_multiply_red[()] = T_multiply_red[()] + T_multiply[v_k0, v_k1]
            with Ts.sblock("T_multiply_1"):
                vi = Ts.axis.spatial(T.int64(1), T.int64(0))
                Ts.reads(T_multiply_red[()])
                Ts.writes(T_multiply_1[()])
                T_multiply_1[()] = T_multiply_red[()] * T.float32(-1.0)
            with Ts.sblock("T_divide"):
                vi = Ts.axis.spatial(T.int64(1), T.int64(0))
                Ts.reads(T_multiply_1[()])
                Ts.writes(T_divide[()])
                T_divide[()] = T_multiply_1[()] / T.float32(2)
    # fmt: on

    mod = LegalizeOps()(CrossEntropyWithLogits)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_cross_entropy_with_logits_batch_symbolic():
    # fmt: off
    n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int64")

    @tvm_tirx.script.ir_module
    class CrossEntropyWithLogits:
        @R.function
        def main(x: R.Tensor((n, m), "float32"), y: R.Tensor((n, m), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((), "float32") = R.nn.cross_entropy_with_logits(x, y)
            return gv

    n_main = T.dynamic("n", "int64")
    m_main = T.dynamic("m", "int64")
    m_cross_entropy_with_logits = T.dynamic("m", "int64")
    n_cross_entropy_with_logits = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((n_main, m_main), dtype="float32"), y: R.Tensor((n_main, m_main), dtype="float32")):
            gv = R.call_tir(Expected.cross_entropy_with_logits, (x, y), ty_args=[R.Tensor((), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def cross_entropy_with_logits(x: T.Tensor((n_cross_entropy_with_logits, m_cross_entropy_with_logits), "float32"), y: T.Tensor((n_cross_entropy_with_logits, m_cross_entropy_with_logits), "float32"), T_divide: T.Tensor((), "float32")):
            T.func_attr({"tirx.noalias": True})

            T_multiply = Ts.sblock_alloc_tensor((n_cross_entropy_with_logits, m_cross_entropy_with_logits))
            T_multiply_red = Ts.sblock_alloc_tensor(())
            T_multiply_1 = Ts.sblock_alloc_tensor(())
            for ax0, ax1 in T.grid(n_cross_entropy_with_logits, m_cross_entropy_with_logits):
                with Ts.sblock("T_multiply"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(x[v_ax0, v_ax1], y[v_ax0, v_ax1])
                    Ts.writes(T_multiply[v_ax0, v_ax1])
                    T_multiply[v_ax0, v_ax1] = x[v_ax0, v_ax1] * y[v_ax0, v_ax1]
            for k0, k1 in T.grid(n_cross_entropy_with_logits, m_cross_entropy_with_logits):
                with Ts.sblock("T_multiply_red"):
                    v_k0, v_k1 = Ts.axis.remap("RR", [k0, k1])
                    Ts.reads(T_multiply[v_k0, v_k1])
                    Ts.writes(T_multiply_red[()])
                    with Ts.init():
                        T_multiply_red[()] = T.float32(0.0)
                    T_multiply_red[()] = T_multiply_red[()] + T_multiply[v_k0, v_k1]
            with Ts.sblock("T_multiply_1"):
                vi = Ts.axis.spatial(T.int64(1), T.int64(0))
                Ts.reads(T_multiply_red[()])
                Ts.writes(T_multiply_1[()])
                T_multiply_1[()] = T_multiply_red[()] * T.float32(-1.0)
            with Ts.sblock("T_divide"):
                vi = Ts.axis.spatial(T.int64(1), T.int64(0))
                Ts.reads(T_multiply_1[()])
                Ts.writes(T_divide[()])
                T_divide[()] = T_multiply_1[()] / T.Cast("float32", n_cross_entropy_with_logits)
    # fmt: on

    mod = LegalizeOps()(CrossEntropyWithLogits)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_batch_norm():
    # fmt: off
    @tvm_tirx.script.ir_module
    class BatchNorm:
        @R.function
        def main(x: R.Tensor((2, 3, 28, 28), "float32"), gamma: R.Tensor((3,), "float32"), beta: R.Tensor((3,), "float32"), moving_mean: R.Tensor((3,), "float32"), moving_var: R.Tensor((3,), "float32")) -> R.Tuple(R.Tensor((2, 3, 28, 28), "float32"), R.Tensor((3,), "float32"), R.Tensor((3,), "float32")):
            gv: R.Tuple(R.Tensor((2, 3, 28, 28), "float32"), R.Tensor((3,), "float32"), R.Tensor((3,), "float32")) = R.nn.batch_norm(x, gamma, beta, moving_mean, moving_var, axis=1)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function(private=True)
        def batch_norm(x: T.Tensor((T.int64(2), T.int64(3), T.int64(28), T.int64(28)), "float32"), gamma: T.Tensor((T.int64(3),), "float32"), beta: T.Tensor((T.int64(3),), "float32"), moving_mean: T.Tensor((T.int64(3),), "float32"), moving_var: T.Tensor((T.int64(3),), "float32"), T_add: T.Tensor((T.int64(2), T.int64(3), T.int64(28), T.int64(28)), "float32"), T_add_1: T.Tensor((T.int64(3),), "float32"), T_add_2: T.Tensor((T.int64(3),), "float32")):
            T.func_attr({"tirx.noalias": True})

            with Ts.sblock("root"):
                Ts.reads()
                Ts.writes()
                x_red = Ts.sblock_alloc_tensor((T.int64(3),))
                T_divide = Ts.sblock_alloc_tensor((T.int64(3),))
                T_reshape = Ts.sblock_alloc_tensor((T.int64(1), T.int64(3), T.int64(1), T.int64(1)))
                T_subtract = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3), T.int64(28), T.int64(28)))
                T_subtract_1 = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3), T.int64(28), T.int64(28)))
                T_subtract_2 = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3), T.int64(28), T.int64(28)))
                T_multiply = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3), T.int64(28), T.int64(28)))
                T_multiply_red = Ts.sblock_alloc_tensor((T.int64(3),))
                T_divide_1 = Ts.sblock_alloc_tensor((T.int64(3),))
                T_reshape_1 = Ts.sblock_alloc_tensor((T.int64(1), T.int64(3), T.int64(1), T.int64(1)))
                T_add_3 = Ts.sblock_alloc_tensor((T.int64(1), T.int64(3), T.int64(1), T.int64(1)))
                compute = Ts.sblock_alloc_tensor((T.int64(1), T.int64(3), T.int64(1), T.int64(1)))
                T_divide_2 = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3), T.int64(28), T.int64(28)))
                T_reshape_2 = Ts.sblock_alloc_tensor((T.int64(1), T.int64(3), T.int64(1), T.int64(1)))
                T_multiply_1 = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3), T.int64(28), T.int64(28)))
                T_reshape_3 = Ts.sblock_alloc_tensor((T.int64(1), T.int64(3), T.int64(1), T.int64(1)))
                T_multiply_2 = Ts.sblock_alloc_tensor((T.int64(3),))
                T_multiply_3 = Ts.sblock_alloc_tensor((T.int64(3),))
                T_multiply_4 = Ts.sblock_alloc_tensor((T.int64(3),))
                T_multiply_5 = Ts.sblock_alloc_tensor((T.int64(3),))
                for ax0 in range(T.int64(3)):
                    for k0 in range(T.int64(2)):
                        for k2 in range(T.int64(28)):
                            for k3 in range(T.int64(28)):
                                with Ts.sblock("x_red"):
                                    v_ax0 = Ts.axis.spatial(T.int64(3), ax0)
                                    v_k0 = Ts.axis.reduce(T.int64(2), k0)
                                    v_k2 = Ts.axis.reduce(T.int64(28), k2)
                                    v_k3 = Ts.axis.reduce(T.int64(28), k3)
                                    Ts.reads(x[v_k0, v_ax0, v_k2, v_k3])
                                    Ts.writes(x_red[v_ax0])
                                    with Ts.init():
                                        x_red[v_ax0] = T.float32(0.0)
                                    x_red[v_ax0] = x_red[v_ax0] + x[v_k0, v_ax0, v_k2, v_k3]
                for ax0 in range(T.int64(3)):
                    with Ts.sblock("T_divide"):
                        v_ax0 = Ts.axis.spatial(T.int64(3), ax0)
                        Ts.reads(x_red[v_ax0])
                        Ts.writes(T_divide[v_ax0])
                        T_divide[v_ax0] = x_red[v_ax0] / T.float32(1568.0)
                for ax0 in range(T.int64(1)):
                    for ax1 in range(T.int64(3)):
                        for ax2 in range(T.int64(1)):
                            for ax3 in range(T.int64(1)):
                                with Ts.sblock("T_reshape"):
                                    v_ax0 = Ts.axis.spatial(T.int64(1), ax0)
                                    v_ax1 = Ts.axis.spatial(T.int64(3), ax1)
                                    v_ax2 = Ts.axis.spatial(T.int64(1), ax2)
                                    v_ax3 = Ts.axis.spatial(T.int64(1), ax3)
                                    Ts.reads(T_divide[(v_ax1 + v_ax2 + v_ax3) % T.int64(3)])
                                    Ts.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = T_divide[(v_ax1 + v_ax2 + v_ax3) % T.int64(3)]
                for ax0 in range(T.int64(2)):
                    for ax1 in range(T.int64(3)):
                        for ax2 in range(T.int64(28)):
                            for ax3 in range(T.int64(28)):
                                with Ts.sblock("T_subtract"):
                                    v_ax0 = Ts.axis.spatial(T.int64(2), ax0)
                                    v_ax1 = Ts.axis.spatial(T.int64(3), ax1)
                                    v_ax2 = Ts.axis.spatial(T.int64(28), ax2)
                                    v_ax3 = Ts.axis.spatial(T.int64(28), ax3)
                                    Ts.reads(x[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                                    Ts.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = x[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
                for ax0 in range(T.int64(2)):
                    for ax1 in range(T.int64(3)):
                        for ax2 in range(T.int64(28)):
                            for ax3 in range(T.int64(28)):
                                with Ts.sblock("T_subtract_1"):
                                    v_ax0 = Ts.axis.spatial(T.int64(2), ax0)
                                    v_ax1 = Ts.axis.spatial(T.int64(3), ax1)
                                    v_ax2 = Ts.axis.spatial(T.int64(28), ax2)
                                    v_ax3 = Ts.axis.spatial(T.int64(28), ax3)
                                    Ts.reads(x[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                                    Ts.writes(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] = x[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
                for ax0 in range(T.int64(2)):
                    for ax1 in range(T.int64(3)):
                        for ax2 in range(T.int64(28)):
                            for ax3 in range(T.int64(28)):
                                with Ts.sblock("T_subtract_2"):
                                    v_ax0 = Ts.axis.spatial(T.int64(2), ax0)
                                    v_ax1 = Ts.axis.spatial(T.int64(3), ax1)
                                    v_ax2 = Ts.axis.spatial(T.int64(28), ax2)
                                    v_ax3 = Ts.axis.spatial(T.int64(28), ax3)
                                    Ts.reads(x[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                                    Ts.writes(T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3] = x[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
                for ax0 in range(T.int64(2)):
                    for ax1 in range(T.int64(3)):
                        for ax2 in range(T.int64(28)):
                            for ax3 in range(T.int64(28)):
                                with Ts.sblock("T_multiply"):
                                    v_ax0 = Ts.axis.spatial(T.int64(2), ax0)
                                    v_ax1 = Ts.axis.spatial(T.int64(3), ax1)
                                    v_ax2 = Ts.axis.spatial(T.int64(28), ax2)
                                    v_ax3 = Ts.axis.spatial(T.int64(28), ax3)
                                    Ts.reads(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3], T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                                    Ts.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] * T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3]
                for ax0 in range(T.int64(3)):
                    for k0 in range(T.int64(2)):
                        for k2 in range(T.int64(28)):
                            for k3 in range(T.int64(28)):
                                with Ts.sblock("T_multiply_red"):
                                    v_ax0 = Ts.axis.spatial(T.int64(3), ax0)
                                    v_k0 = Ts.axis.reduce(T.int64(2), k0)
                                    v_k2 = Ts.axis.reduce(T.int64(28), k2)
                                    v_k3 = Ts.axis.reduce(T.int64(28), k3)
                                    Ts.reads(T_multiply[v_k0, v_ax0, v_k2, v_k3])
                                    Ts.writes(T_multiply_red[v_ax0])
                                    with Ts.init():
                                        T_multiply_red[v_ax0] = T.float32(0.0)
                                    T_multiply_red[v_ax0] = T_multiply_red[v_ax0] + T_multiply[v_k0, v_ax0, v_k2, v_k3]
                for ax0 in range(T.int64(3)):
                    with Ts.sblock("T_divide_1"):
                        v_ax0 = Ts.axis.spatial(T.int64(3), ax0)
                        Ts.reads(T_multiply_red[v_ax0])
                        Ts.writes(T_divide_1[v_ax0])
                        T_divide_1[v_ax0] = T_multiply_red[v_ax0] / T.float32(1568.0)
                for ax0 in range(T.int64(1)):
                    for ax1 in range(T.int64(3)):
                        for ax2 in range(T.int64(1)):
                            for ax3 in range(T.int64(1)):
                                with Ts.sblock("T_reshape_1"):
                                    v_ax0 = Ts.axis.spatial(T.int64(1), ax0)
                                    v_ax1 = Ts.axis.spatial(T.int64(3), ax1)
                                    v_ax2 = Ts.axis.spatial(T.int64(1), ax2)
                                    v_ax3 = Ts.axis.spatial(T.int64(1), ax3)
                                    Ts.reads(T_divide_1[(v_ax1 + v_ax2 + v_ax3) % T.int64(3)])
                                    Ts.writes(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3] = T_divide_1[(v_ax1 + v_ax2 + v_ax3) % T.int64(3)]
                for ax0 in range(T.int64(1)):
                    for ax1 in range(T.int64(3)):
                        for ax2 in range(T.int64(1)):
                            for ax3 in range(T.int64(1)):
                                with Ts.sblock("T_add"):
                                    v_ax0 = Ts.axis.spatial(T.int64(1), ax0)
                                    v_ax1 = Ts.axis.spatial(T.int64(3), ax1)
                                    v_ax2 = Ts.axis.spatial(T.int64(1), ax2)
                                    v_ax3 = Ts.axis.spatial(T.int64(1), ax3)
                                    Ts.reads(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3])
                                    Ts.writes(T_add_3[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_add_3[v_ax0, v_ax1, v_ax2, v_ax3] = T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3] + T.float32(1.0000000000000001e-05)
                for i0 in range(T.int64(1)):
                    for i1 in range(T.int64(3)):
                        for i2 in range(T.int64(1)):
                            for i3 in range(T.int64(1)):
                                with Ts.sblock("compute"):
                                    v_i0 = Ts.axis.spatial(T.int64(1), i0)
                                    v_i1 = Ts.axis.spatial(T.int64(3), i1)
                                    v_i2 = Ts.axis.spatial(T.int64(1), i2)
                                    v_i3 = Ts.axis.spatial(T.int64(1), i3)
                                    Ts.reads(T_add_3[v_i0, v_i1, v_i2, v_i3])
                                    Ts.writes(compute[v_i0, v_i1, v_i2, v_i3])
                                    compute[v_i0, v_i1, v_i2, v_i3] = T.sqrt(T_add_3[v_i0, v_i1, v_i2, v_i3])
                for ax0 in range(T.int64(2)):
                    for ax1 in range(T.int64(3)):
                        for ax2 in range(T.int64(28)):
                            for ax3 in range(T.int64(28)):
                                with Ts.sblock("T_divide_2"):
                                    v_ax0 = Ts.axis.spatial(T.int64(2), ax0)
                                    v_ax1 = Ts.axis.spatial(T.int64(3), ax1)
                                    v_ax2 = Ts.axis.spatial(T.int64(28), ax2)
                                    v_ax3 = Ts.axis.spatial(T.int64(28), ax3)
                                    Ts.reads(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3], compute[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                                    Ts.writes(T_divide_2[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_divide_2[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] / compute[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
                for ax0 in range(T.int64(1)):
                    for ax1 in range(T.int64(3)):
                        for ax2 in range(T.int64(1)):
                            for ax3 in range(T.int64(1)):
                                with Ts.sblock("T_reshape_2"):
                                    v_ax0 = Ts.axis.spatial(T.int64(1), ax0)
                                    v_ax1 = Ts.axis.spatial(T.int64(3), ax1)
                                    v_ax2 = Ts.axis.spatial(T.int64(1), ax2)
                                    v_ax3 = Ts.axis.spatial(T.int64(1), ax3)
                                    Ts.reads(gamma[(v_ax1 + v_ax2 + v_ax3) % T.int64(3)])
                                    Ts.writes(T_reshape_2[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_reshape_2[v_ax0, v_ax1, v_ax2, v_ax3] = gamma[(v_ax1 + v_ax2 + v_ax3) % T.int64(3)]
                for ax0 in range(T.int64(2)):
                    for ax1 in range(T.int64(3)):
                        for ax2 in range(T.int64(28)):
                            for ax3 in range(T.int64(28)):
                                with Ts.sblock("T_multiply_1"):
                                    v_ax0 = Ts.axis.spatial(T.int64(2), ax0)
                                    v_ax1 = Ts.axis.spatial(T.int64(3), ax1)
                                    v_ax2 = Ts.axis.spatial(T.int64(28), ax2)
                                    v_ax3 = Ts.axis.spatial(T.int64(28), ax3)
                                    Ts.reads(T_divide_2[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_2[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                                    Ts.writes(T_multiply_1[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_multiply_1[v_ax0, v_ax1, v_ax2, v_ax3] = T_divide_2[v_ax0, v_ax1, v_ax2, v_ax3] * T_reshape_2[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
                for ax0 in range(T.int64(1)):
                    for ax1 in range(T.int64(3)):
                        for ax2 in range(T.int64(1)):
                            for ax3 in range(T.int64(1)):
                                with Ts.sblock("T_reshape_3"):
                                    v_ax0 = Ts.axis.spatial(T.int64(1), ax0)
                                    v_ax1 = Ts.axis.spatial(T.int64(3), ax1)
                                    v_ax2 = Ts.axis.spatial(T.int64(1), ax2)
                                    v_ax3 = Ts.axis.spatial(T.int64(1), ax3)
                                    Ts.reads(beta[(v_ax1 + v_ax2 + v_ax3) % T.int64(3)])
                                    Ts.writes(T_reshape_3[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_reshape_3[v_ax0, v_ax1, v_ax2, v_ax3] = beta[(v_ax1 + v_ax2 + v_ax3) % T.int64(3)]
                for ax0 in range(T.int64(2)):
                    for ax1 in range(T.int64(3)):
                        for ax2 in range(T.int64(28)):
                            for ax3 in range(T.int64(28)):
                                with Ts.sblock("T_add_1"):
                                    v_ax0 = Ts.axis.spatial(T.int64(2), ax0)
                                    v_ax1 = Ts.axis.spatial(T.int64(3), ax1)
                                    v_ax2 = Ts.axis.spatial(T.int64(28), ax2)
                                    v_ax3 = Ts.axis.spatial(T.int64(28), ax3)
                                    Ts.reads(T_multiply_1[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_3[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                                    Ts.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_add[v_ax0, v_ax1, v_ax2, v_ax3] = T_multiply_1[v_ax0, v_ax1, v_ax2, v_ax3] + T_reshape_3[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
                for ax0 in range(T.int64(3)):
                    with Ts.sblock("T_multiply_2"):
                        v_ax0 = Ts.axis.spatial(T.int64(3), ax0)
                        Ts.reads(moving_mean[v_ax0])
                        Ts.writes(T_multiply_2[v_ax0])
                        T_multiply_2[v_ax0] = T.float32(0.90000000000000002) * moving_mean[v_ax0]
                for ax0 in range(T.int64(3)):
                    with Ts.sblock("T_multiply_3"):
                        v_ax0 = Ts.axis.spatial(T.int64(3), ax0)
                        Ts.reads(T_divide[v_ax0])
                        Ts.writes(T_multiply_3[v_ax0])
                        T_multiply_3[v_ax0] = T.float32(0.10000000000000001) * T_divide[v_ax0]
                for ax0 in range(T.int64(3)):
                    with Ts.sblock("T_add_2"):
                        v_ax0 = Ts.axis.spatial(T.int64(3), ax0)
                        Ts.reads(T_multiply_2[v_ax0], T_multiply_3[v_ax0])
                        Ts.writes(T_add_1[v_ax0])
                        T_add_1[v_ax0] = T_multiply_2[v_ax0] + T_multiply_3[v_ax0]
                for ax0 in range(T.int64(3)):
                    with Ts.sblock("T_multiply_4"):
                        v_ax0 = Ts.axis.spatial(T.int64(3), ax0)
                        Ts.reads(moving_var[v_ax0])
                        Ts.writes(T_multiply_4[v_ax0])
                        T_multiply_4[v_ax0] = T.float32(0.90000000000000002) * moving_var[v_ax0]
                for ax0 in range(T.int64(3)):
                    with Ts.sblock("T_multiply_5"):
                        v_ax0 = Ts.axis.spatial(T.int64(3), ax0)
                        Ts.reads(T_divide_1[v_ax0])
                        Ts.writes(T_multiply_5[v_ax0])
                        T_multiply_5[v_ax0] = T.float32(0.10000000000000001) * T_divide_1[v_ax0]
                for ax0 in range(T.int64(3)):
                    with Ts.sblock("T_add_3"):
                        v_ax0 = Ts.axis.spatial(T.int64(3), ax0)
                        Ts.reads(T_multiply_4[v_ax0], T_multiply_5[v_ax0])
                        Ts.writes(T_add_2[v_ax0])
                        T_add_2[v_ax0] = T_multiply_4[v_ax0] + T_multiply_5[v_ax0]

        @R.function
        def main(x: R.Tensor((2, 3, 28, 28), dtype="float32"), gamma: R.Tensor((3,), dtype="float32"), beta: R.Tensor((3,), dtype="float32"), moving_mean: R.Tensor((3,), dtype="float32"), moving_var: R.Tensor((3,), dtype="float32")) -> R.Tuple(R.Tensor((2, 3, 28, 28), dtype="float32"), R.Tensor((3,), dtype="float32"), R.Tensor((3,), dtype="float32")):
            cls = Expected
            gv = R.call_tir(cls.batch_norm, (x, gamma, beta, moving_mean, moving_var), ty_args=[_TupleType([R.Tensor((2, 3, 28, 28), dtype="float32"), R.Tensor((3,), dtype="float32"), R.Tensor((3,), dtype="float32")])])
            return gv
    # fmt: on

    mod = LegalizeOps()(BatchNorm)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_batch_norm_symbolic():
    # fmt: off
    n = T.dynamic("n", "int64")
    h = T.dynamic("h", "int64")
    w = T.dynamic("w", "int64")
    c = T.dynamic("c", "int64")

    @tvm_tirx.script.ir_module
    class BatchNorm:
        @R.function
        def main(x: R.Tensor((n, h, w, c), "float32"), gamma: R.Tensor((c,), "float32"), beta: R.Tensor((c,), "float32"), moving_mean: R.Tensor((c,), "float32"), moving_var: R.Tensor((c,), "float32")) -> R.Tuple(R.Tensor((n, h, w, c), "float32"), R.Tensor((c,), "float32"), R.Tensor((c,), "float32")):
            gv: R.Tuple(R.Tensor((n, h, w, c), "float32"), R.Tensor((c,), "float32"), R.Tensor((c,), "float32")) = R.nn.batch_norm(x, gamma, beta, moving_mean, moving_var, axis=1)
            return gv

    n_batch_norm = T.dynamic("n", "int64")
    h_batch_norm = T.dynamic("h", "int64")
    w_batch_norm = T.dynamic("w", "int64")
    c_batch_norm = T.dynamic("c", "int64")
    n_main = T.dynamic("n", "int64")
    h_main = T.dynamic("h", "int64")
    w_main = T.dynamic("w", "int64")
    c_main = T.dynamic("c", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function(private=True)
        def batch_norm(x: T.Tensor((n_batch_norm, h_batch_norm, w_batch_norm, c_batch_norm), "float32"), gamma: T.Tensor((c_batch_norm,), "float32"), beta: T.Tensor((c_batch_norm,), "float32"), moving_mean: T.Tensor((c_batch_norm,), "float32"), moving_var: T.Tensor((c_batch_norm,), "float32"), T_add: T.Tensor((n_batch_norm, h_batch_norm, w_batch_norm, c_batch_norm), "float32"), T_add_1: T.Tensor((T.max(c_batch_norm, h_batch_norm),), "float32"), T_add_2: T.Tensor((T.max(c_batch_norm, h_batch_norm),), "float32")):
            T.func_attr({"tirx.noalias": True})

            with Ts.sblock("root"):
                Ts.reads()
                Ts.writes()
                x_red = Ts.sblock_alloc_tensor((h_batch_norm,))
                T_divide = Ts.sblock_alloc_tensor((h_batch_norm,))
                T_reshape = Ts.sblock_alloc_tensor((T.int64(1), h_batch_norm, T.int64(1), T.int64(1)))
                T_subtract = Ts.sblock_alloc_tensor((n_batch_norm, h_batch_norm, w_batch_norm, c_batch_norm))
                T_subtract_1 = Ts.sblock_alloc_tensor((n_batch_norm, h_batch_norm, w_batch_norm, c_batch_norm))
                T_subtract_2 = Ts.sblock_alloc_tensor((n_batch_norm, h_batch_norm, w_batch_norm, c_batch_norm))
                T_multiply = Ts.sblock_alloc_tensor((n_batch_norm, h_batch_norm, w_batch_norm, c_batch_norm))
                T_multiply_red = Ts.sblock_alloc_tensor((h_batch_norm,))
                T_divide_1 = Ts.sblock_alloc_tensor((h_batch_norm,))
                T_reshape_1 = Ts.sblock_alloc_tensor((T.int64(1), h_batch_norm, T.int64(1), T.int64(1)))
                T_add_3 = Ts.sblock_alloc_tensor((T.int64(1), h_batch_norm, T.int64(1), T.int64(1)))
                compute = Ts.sblock_alloc_tensor((T.int64(1), h_batch_norm, T.int64(1), T.int64(1)))
                T_divide_2 = Ts.sblock_alloc_tensor((n_batch_norm, h_batch_norm, w_batch_norm, c_batch_norm))
                T_reshape_2 = Ts.sblock_alloc_tensor((T.int64(1), h_batch_norm, T.int64(1), T.int64(1)))
                T_multiply_1 = Ts.sblock_alloc_tensor((n_batch_norm, h_batch_norm, w_batch_norm, c_batch_norm))
                T_reshape_3 = Ts.sblock_alloc_tensor((T.int64(1), h_batch_norm, T.int64(1), T.int64(1)))
                T_multiply_2 = Ts.sblock_alloc_tensor((c_batch_norm,))
                T_multiply_3 = Ts.sblock_alloc_tensor((h_batch_norm,))
                T_multiply_4 = Ts.sblock_alloc_tensor((c_batch_norm,))
                T_multiply_5 = Ts.sblock_alloc_tensor((h_batch_norm,))
                for ax0 in range(h_batch_norm):
                    for k0 in range(n_batch_norm):
                        for k2 in range(w_batch_norm):
                            for k3 in range(c_batch_norm):
                                with Ts.sblock("x_red"):
                                    v_ax0 = Ts.axis.spatial(h_batch_norm, ax0)
                                    v_k0 = Ts.axis.reduce(n_batch_norm, k0)
                                    v_k2 = Ts.axis.reduce(w_batch_norm, k2)
                                    v_k3 = Ts.axis.reduce(c_batch_norm, k3)
                                    Ts.reads(x[v_k0, v_ax0, v_k2, v_k3])
                                    Ts.writes(x_red[v_ax0])
                                    with Ts.init():
                                        x_red[v_ax0] = T.float32(0.0)
                                    x_red[v_ax0] = x_red[v_ax0] + x[v_k0, v_ax0, v_k2, v_k3]
                for ax0 in range(h_batch_norm):
                    with Ts.sblock("T_divide"):
                        v_ax0 = Ts.axis.spatial(h_batch_norm, ax0)
                        Ts.reads(x_red[v_ax0])
                        Ts.writes(T_divide[v_ax0])
                        T_divide[v_ax0] = x_red[v_ax0] / T.Cast("float32", n_batch_norm * w_batch_norm * c_batch_norm)
                for ax0 in range(T.int64(1)):
                    for ax1 in range(h_batch_norm):
                        for ax2 in range(T.int64(1)):
                            for ax3 in range(T.int64(1)):
                                with Ts.sblock("T_reshape"):
                                    v_ax0 = Ts.axis.spatial(T.int64(1), ax0)
                                    v_ax1 = Ts.axis.spatial(h_batch_norm, ax1)
                                    v_ax2 = Ts.axis.spatial(T.int64(1), ax2)
                                    v_ax3 = Ts.axis.spatial(T.int64(1), ax3)
                                    Ts.reads(T_divide[(v_ax0 * h_batch_norm + v_ax1 + v_ax2 + v_ax3) % h_batch_norm])
                                    Ts.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = T_divide[(v_ax0 * h_batch_norm + v_ax1 + v_ax2 + v_ax3) % h_batch_norm]
                for ax0 in range(n_batch_norm):
                    for ax1 in range(h_batch_norm):
                        for ax2 in range(w_batch_norm):
                            for ax3 in range(c_batch_norm):
                                with Ts.sblock("T_subtract"):
                                    v_ax0 = Ts.axis.spatial(n_batch_norm, ax0)
                                    v_ax1 = Ts.axis.spatial(h_batch_norm, ax1)
                                    v_ax2 = Ts.axis.spatial(w_batch_norm, ax2)
                                    v_ax3 = Ts.axis.spatial(c_batch_norm, ax3)
                                    Ts.reads(x[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                                    Ts.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = x[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
                for ax0 in range(n_batch_norm):
                    for ax1 in range(h_batch_norm):
                        for ax2 in range(w_batch_norm):
                            for ax3 in range(c_batch_norm):
                                with Ts.sblock("T_subtract_1"):
                                    v_ax0 = Ts.axis.spatial(n_batch_norm, ax0)
                                    v_ax1 = Ts.axis.spatial(h_batch_norm, ax1)
                                    v_ax2 = Ts.axis.spatial(w_batch_norm, ax2)
                                    v_ax3 = Ts.axis.spatial(c_batch_norm, ax3)
                                    Ts.reads(x[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                                    Ts.writes(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] = x[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
                for ax0 in range(n_batch_norm):
                    for ax1 in range(h_batch_norm):
                        for ax2 in range(w_batch_norm):
                            for ax3 in range(c_batch_norm):
                                with Ts.sblock("T_subtract_2"):
                                    v_ax0 = Ts.axis.spatial(n_batch_norm, ax0)
                                    v_ax1 = Ts.axis.spatial(h_batch_norm, ax1)
                                    v_ax2 = Ts.axis.spatial(w_batch_norm, ax2)
                                    v_ax3 = Ts.axis.spatial(c_batch_norm, ax3)
                                    Ts.reads(x[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                                    Ts.writes(T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3] = x[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
                for ax0 in range(n_batch_norm):
                    for ax1 in range(h_batch_norm):
                        for ax2 in range(w_batch_norm):
                            for ax3 in range(c_batch_norm):
                                with Ts.sblock("T_multiply"):
                                    v_ax0 = Ts.axis.spatial(n_batch_norm, ax0)
                                    v_ax1 = Ts.axis.spatial(h_batch_norm, ax1)
                                    v_ax2 = Ts.axis.spatial(w_batch_norm, ax2)
                                    v_ax3 = Ts.axis.spatial(c_batch_norm, ax3)
                                    Ts.reads(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3], T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                                    Ts.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] * T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3]
                for ax0 in range(h_batch_norm):
                    for k0 in range(n_batch_norm):
                        for k2 in range(w_batch_norm):
                            for k3 in range(c_batch_norm):
                                with Ts.sblock("T_multiply_red"):
                                    v_ax0 = Ts.axis.spatial(h_batch_norm, ax0)
                                    v_k0 = Ts.axis.reduce(n_batch_norm, k0)
                                    v_k2 = Ts.axis.reduce(w_batch_norm, k2)
                                    v_k3 = Ts.axis.reduce(c_batch_norm, k3)
                                    Ts.reads(T_multiply[v_k0, v_ax0, v_k2, v_k3])
                                    Ts.writes(T_multiply_red[v_ax0])
                                    with Ts.init():
                                        T_multiply_red[v_ax0] = T.float32(0.0)
                                    T_multiply_red[v_ax0] = T_multiply_red[v_ax0] + T_multiply[v_k0, v_ax0, v_k2, v_k3]
                for ax0 in range(h_batch_norm):
                    with Ts.sblock("T_divide_1"):
                        v_ax0 = Ts.axis.spatial(h_batch_norm, ax0)
                        Ts.reads(T_multiply_red[v_ax0])
                        Ts.writes(T_divide_1[v_ax0])
                        T_divide_1[v_ax0] = T_multiply_red[v_ax0] / T.Cast("float32", n_batch_norm * w_batch_norm * c_batch_norm)
                for ax0 in range(T.int64(1)):
                    for ax1 in range(h_batch_norm):
                        for ax2 in range(T.int64(1)):
                            for ax3 in range(T.int64(1)):
                                with Ts.sblock("T_reshape_1"):
                                    v_ax0 = Ts.axis.spatial(T.int64(1), ax0)
                                    v_ax1 = Ts.axis.spatial(h_batch_norm, ax1)
                                    v_ax2 = Ts.axis.spatial(T.int64(1), ax2)
                                    v_ax3 = Ts.axis.spatial(T.int64(1), ax3)
                                    Ts.reads(T_divide_1[(v_ax0 * h_batch_norm + v_ax1 + v_ax2 + v_ax3) % h_batch_norm])
                                    Ts.writes(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3] = T_divide_1[(v_ax0 * h_batch_norm + v_ax1 + v_ax2 + v_ax3) % h_batch_norm]
                for ax0 in range(T.int64(1)):
                    for ax1 in range(h_batch_norm):
                        for ax2 in range(T.int64(1)):
                            for ax3 in range(T.int64(1)):
                                with Ts.sblock("T_add"):
                                    v_ax0 = Ts.axis.spatial(T.int64(1), ax0)
                                    v_ax1 = Ts.axis.spatial(h_batch_norm, ax1)
                                    v_ax2 = Ts.axis.spatial(T.int64(1), ax2)
                                    v_ax3 = Ts.axis.spatial(T.int64(1), ax3)
                                    Ts.reads(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3])
                                    Ts.writes(T_add_3[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_add_3[v_ax0, v_ax1, v_ax2, v_ax3] = T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3] + T.float32(1.0000000000000001e-05)
                for i0 in range(T.int64(1)):
                    for i1 in range(h_batch_norm):
                        for i2 in range(T.int64(1)):
                            for i3 in range(T.int64(1)):
                                with Ts.sblock("compute"):
                                    v_i0 = Ts.axis.spatial(T.int64(1), i0)
                                    v_i1 = Ts.axis.spatial(h_batch_norm, i1)
                                    v_i2 = Ts.axis.spatial(T.int64(1), i2)
                                    v_i3 = Ts.axis.spatial(T.int64(1), i3)
                                    Ts.reads(T_add_3[v_i0, v_i1, v_i2, v_i3])
                                    Ts.writes(compute[v_i0, v_i1, v_i2, v_i3])
                                    compute[v_i0, v_i1, v_i2, v_i3] = T.sqrt(T_add_3[v_i0, v_i1, v_i2, v_i3])
                for ax0 in range(n_batch_norm):
                    for ax1 in range(h_batch_norm):
                        for ax2 in range(w_batch_norm):
                            for ax3 in range(c_batch_norm):
                                with Ts.sblock("T_divide_2"):
                                    v_ax0 = Ts.axis.spatial(n_batch_norm, ax0)
                                    v_ax1 = Ts.axis.spatial(h_batch_norm, ax1)
                                    v_ax2 = Ts.axis.spatial(w_batch_norm, ax2)
                                    v_ax3 = Ts.axis.spatial(c_batch_norm, ax3)
                                    Ts.reads(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3], compute[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                                    Ts.writes(T_divide_2[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_divide_2[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] / compute[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
                for ax0 in range(T.int64(1)):
                    for ax1 in range(h_batch_norm):
                        for ax2 in range(T.int64(1)):
                            for ax3 in range(T.int64(1)):
                                with Ts.sblock("T_reshape_2"):
                                    v_ax0 = Ts.axis.spatial(T.int64(1), ax0)
                                    v_ax1 = Ts.axis.spatial(h_batch_norm, ax1)
                                    v_ax2 = Ts.axis.spatial(T.int64(1), ax2)
                                    v_ax3 = Ts.axis.spatial(T.int64(1), ax3)
                                    Ts.reads(gamma[(v_ax0 * h_batch_norm + v_ax1 + v_ax2 + v_ax3) % c_batch_norm])
                                    Ts.writes(T_reshape_2[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_reshape_2[v_ax0, v_ax1, v_ax2, v_ax3] = gamma[(v_ax0 * h_batch_norm + v_ax1 + v_ax2 + v_ax3) % c_batch_norm]
                for ax0 in range(n_batch_norm):
                    for ax1 in range(h_batch_norm):
                        for ax2 in range(w_batch_norm):
                            for ax3 in range(c_batch_norm):
                                with Ts.sblock("T_multiply_1"):
                                    v_ax0 = Ts.axis.spatial(n_batch_norm, ax0)
                                    v_ax1 = Ts.axis.spatial(h_batch_norm, ax1)
                                    v_ax2 = Ts.axis.spatial(w_batch_norm, ax2)
                                    v_ax3 = Ts.axis.spatial(c_batch_norm, ax3)
                                    Ts.reads(T_divide_2[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_2[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                                    Ts.writes(T_multiply_1[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_multiply_1[v_ax0, v_ax1, v_ax2, v_ax3] = T_divide_2[v_ax0, v_ax1, v_ax2, v_ax3] * T_reshape_2[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
                for ax0 in range(T.int64(1)):
                    for ax1 in range(h_batch_norm):
                        for ax2 in range(T.int64(1)):
                            for ax3 in range(T.int64(1)):
                                with Ts.sblock("T_reshape_3"):
                                    v_ax0 = Ts.axis.spatial(T.int64(1), ax0)
                                    v_ax1 = Ts.axis.spatial(h_batch_norm, ax1)
                                    v_ax2 = Ts.axis.spatial(T.int64(1), ax2)
                                    v_ax3 = Ts.axis.spatial(T.int64(1), ax3)
                                    Ts.reads(beta[(v_ax0 * h_batch_norm + v_ax1 + v_ax2 + v_ax3) % c_batch_norm])
                                    Ts.writes(T_reshape_3[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_reshape_3[v_ax0, v_ax1, v_ax2, v_ax3] = beta[(v_ax0 * h_batch_norm + v_ax1 + v_ax2 + v_ax3) % c_batch_norm]
                for ax0 in range(n_batch_norm):
                    for ax1 in range(h_batch_norm):
                        for ax2 in range(w_batch_norm):
                            for ax3 in range(c_batch_norm):
                                with Ts.sblock("T_add_1"):
                                    v_ax0 = Ts.axis.spatial(n_batch_norm, ax0)
                                    v_ax1 = Ts.axis.spatial(h_batch_norm, ax1)
                                    v_ax2 = Ts.axis.spatial(w_batch_norm, ax2)
                                    v_ax3 = Ts.axis.spatial(c_batch_norm, ax3)
                                    Ts.reads(T_multiply_1[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_3[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                                    Ts.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_add[v_ax0, v_ax1, v_ax2, v_ax3] = T_multiply_1[v_ax0, v_ax1, v_ax2, v_ax3] + T_reshape_3[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
                for ax0 in range(c_batch_norm):
                    with Ts.sblock("T_multiply_2"):
                        v_ax0 = Ts.axis.spatial(c_batch_norm, ax0)
                        Ts.reads(moving_mean[v_ax0])
                        Ts.writes(T_multiply_2[v_ax0])
                        T_multiply_2[v_ax0] = T.float32(0.90000000000000002) * moving_mean[v_ax0]
                for ax0 in range(h_batch_norm):
                    with Ts.sblock("T_multiply_3"):
                        v_ax0 = Ts.axis.spatial(h_batch_norm, ax0)
                        Ts.reads(T_divide[v_ax0])
                        Ts.writes(T_multiply_3[v_ax0])
                        T_multiply_3[v_ax0] = T.float32(0.10000000000000001) * T_divide[v_ax0]
                for ax0 in range(T.max(c_batch_norm, h_batch_norm)):
                    with Ts.sblock("T_add_2"):
                        v_ax0 = Ts.axis.spatial(T.max(c_batch_norm, h_batch_norm), ax0)
                        Ts.reads(T_multiply_2[v_ax0], T_multiply_3[v_ax0])
                        Ts.writes(T_add_1[v_ax0])
                        T_add_1[v_ax0] = T_multiply_2[v_ax0] + T_multiply_3[v_ax0]
                for ax0 in range(c_batch_norm):
                    with Ts.sblock("T_multiply_4"):
                        v_ax0 = Ts.axis.spatial(c_batch_norm, ax0)
                        Ts.reads(moving_var[v_ax0])
                        Ts.writes(T_multiply_4[v_ax0])
                        T_multiply_4[v_ax0] = T.float32(0.90000000000000002) * moving_var[v_ax0]
                for ax0 in range(h_batch_norm):
                    with Ts.sblock("T_multiply_5"):
                        v_ax0 = Ts.axis.spatial(h_batch_norm, ax0)
                        Ts.reads(T_divide_1[v_ax0])
                        Ts.writes(T_multiply_5[v_ax0])
                        T_multiply_5[v_ax0] = T.float32(0.10000000000000001) * T_divide_1[v_ax0]
                for ax0 in range(T.max(c_batch_norm, h_batch_norm)):
                    with Ts.sblock("T_add_3"):
                        v_ax0 = Ts.axis.spatial(T.max(c_batch_norm, h_batch_norm), ax0)
                        Ts.reads(T_multiply_4[v_ax0], T_multiply_5[v_ax0])
                        Ts.writes(T_add_2[v_ax0])
                        T_add_2[v_ax0] = T_multiply_4[v_ax0] + T_multiply_5[v_ax0]

        @R.function
        def main(x: R.Tensor((n_main, h_main, w_main, c_main), dtype="float32"), gamma: R.Tensor((c_main,), dtype="float32"), beta: R.Tensor((c_main,), dtype="float32"), moving_mean: R.Tensor((c_main,), dtype="float32"), moving_var: R.Tensor((c_main,), dtype="float32")) -> R.Tuple(R.Tensor((n_main, h_main, w_main, c_main), dtype="float32"), R.Tensor((T.max(c_main, h_main),), dtype="float32"), R.Tensor((T.max(c_main, h_main),), dtype="float32")):
            cls = Expected
            gv = R.call_tir(cls.batch_norm, (x, gamma, beta, moving_mean, moving_var), ty_args=[_TupleType([R.Tensor((n_main, h_main, w_main, c_main), dtype="float32"), R.Tensor((T.max(c_main, h_main),), dtype="float32"), R.Tensor((T.max(c_main, h_main),), dtype="float32")])])
            return gv

    mod = LegalizeOps()(BatchNorm)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_layer_norm():
    # fmt: off
    @tvm_tirx.script.ir_module
    class LayerNorm:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32"), gamma: R.Tensor((4, 5), "float32"), beta: R.Tensor((4, 5), "float32")) -> R.Tensor((2, 3, 4, 5), "float32"):
            gv: R.Tensor((2, 3, 4, 5), "float32") = R.nn.layer_norm(x, gamma, beta, axes=[-2, -1])
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32"), gamma: R.Tensor((4, 5), "float32"), beta: R.Tensor((4, 5), "float32")) -> R.Tensor((2, 3, 4, 5), "float32"):
            gv = R.call_tir(Expected.layer_norm, (x, gamma, beta), ty_args=[R.Tensor((2, 3, 4, 5), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def layer_norm(x: T.Tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"), gamma: T.Tensor((T.int64(4), T.int64(5)), "float32"), beta: T.Tensor((T.int64(4), T.int64(5)), "float32"), T_layer_norm: T.Tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            x_sum = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3)))
            x_mean = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3)))
            x_var_sum = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3)))
            for ax0, ax1, k2, k3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("x_sum"):
                    v_ax0, v_ax1, v_k2, v_k3 = Ts.axis.remap("SSRR", [ax0, ax1, k2, k3])
                    Ts.reads(x[v_ax0, v_ax1, v_k2, v_k3])
                    Ts.writes(x_sum[v_ax0, v_ax1])
                    with Ts.init():
                        x_sum[v_ax0, v_ax1] = T.float32(0.0)
                    x_sum[v_ax0, v_ax1] = x_sum[v_ax0, v_ax1] + x[v_ax0, v_ax1, v_k2, v_k3]
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("x_mean"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(x_sum[v_ax0, v_ax1])
                    Ts.writes(x_mean[v_ax0, v_ax1])
                    x_mean[v_ax0, v_ax1] = x_sum[v_ax0, v_ax1] / T.float32(20.0)
            for ax0, ax1, k2, k3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("x_var_sum"):
                    v_ax0, v_ax1, v_k2, v_k3 = Ts.axis.remap("SSRR", [ax0, ax1, k2, k3])
                    Ts.reads(x[v_ax0, v_ax1, v_k2, v_k3], x_mean[v_ax0, v_ax1])
                    Ts.writes(x_var_sum[v_ax0, v_ax1])
                    with Ts.init():
                        x_var_sum[v_ax0, v_ax1] = T.float32(0.0)
                    x_var_sum[v_ax0, v_ax1] = x_var_sum[v_ax0, v_ax1] + (x[v_ax0, v_ax1, v_k2, v_k3] - x_mean[v_ax0, v_ax1]) * (x[v_ax0, v_ax1, v_k2, v_k3] - x_mean[v_ax0, v_ax1])
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("T_layer_norm"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(x[v_ax0, v_ax1, v_ax2, v_ax3], x_mean[v_ax0, v_ax1], x_var_sum[v_ax0, v_ax1], gamma[v_ax2, v_ax3], beta[v_ax2, v_ax3])
                    Ts.writes(T_layer_norm[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_layer_norm[v_ax0, v_ax1, v_ax2, v_ax3] = (x[v_ax0, v_ax1, v_ax2, v_ax3] - x_mean[v_ax0, v_ax1]) * T.rsqrt(x_var_sum[v_ax0, v_ax1] / T.float32(20.0) + T.float32(1.0000000000000001e-05)) * gamma[v_ax2, v_ax3] + beta[v_ax2, v_ax3]
    # fmt: on
    mod = LegalizeOps()(LayerNorm)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_layer_norm_1d():
    # fmt: off
    @I.ir_module
    class LayerNorm_1D:
        @R.function
        def forward(x: R.Tensor((3,), dtype="float32"), layer_norm_weight: R.Tensor((3,), dtype="float32"), layer_norm_bias: R.Tensor((3,), dtype="float32")) -> R.Tensor((3,), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                layer_norm: R.Tensor((3,), dtype="float32") = R.nn.layer_norm(x, layer_norm_weight, layer_norm_bias, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
                gv: R.Tensor((3,), dtype="float32") = layer_norm
                R.output(gv)
            return gv

    @I.ir_module
    class LayerNorm_1D_Expected:
        @Ts.function(private=True)
        def layer_norm(x: T.Tensor((T.int64(3),), "float32"), layer_norm_weight: T.Tensor((T.int64(3),), "float32"), layer_norm_bias: T.Tensor((T.int64(3),), "float32"), T_layer_norm: T.Tensor((T.int64(3),), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            x_sum = Ts.sblock_alloc_tensor(())
            x_mean = Ts.sblock_alloc_tensor(())
            x_var_sum = Ts.sblock_alloc_tensor(())
            for k0 in range(T.int64(3)):
                with Ts.sblock("x_sum"):
                    v_k0 = Ts.axis.reduce(T.int64(3), k0)
                    Ts.reads(x[v_k0])
                    Ts.writes(x_sum[()])
                    with Ts.init():
                        x_sum[()] = T.float32(0.0)
                    x_sum[()] = x_sum[()] + x[v_k0]
            with Ts.sblock("x_mean"):
                vi = Ts.axis.spatial(T.int64(1), T.int64(0))
                Ts.reads(x_sum[()])
                Ts.writes(x_mean[()])
                x_mean[()] = x_sum[()] / T.float32(3.0)
            for k0 in range(T.int64(3)):
                with Ts.sblock("x_var_sum"):
                    v_k0 = Ts.axis.reduce(T.int64(3), k0)
                    Ts.reads(x[v_k0], x_mean[()])
                    Ts.writes(x_var_sum[()])
                    with Ts.init():
                        x_var_sum[()] = T.float32(0.0)
                    x_var_sum[()] = x_var_sum[()] + (x[v_k0] - x_mean[()]) * (x[v_k0] - x_mean[()])
            for ax0 in range(T.int64(3)):
                with Ts.sblock("T_layer_norm"):
                    v_ax0 = Ts.axis.spatial(T.int64(3), ax0)
                    Ts.reads(x[v_ax0], x_mean[()], x_var_sum[()], layer_norm_weight[v_ax0], layer_norm_bias[v_ax0])
                    Ts.writes(T_layer_norm[v_ax0])
                    T_layer_norm[v_ax0] = (x[v_ax0] - x_mean[()]) * T.rsqrt(x_var_sum[()] / T.float32(3.0) + T.float32(1.0000000000000001e-05)) * layer_norm_weight[v_ax0] + layer_norm_bias[v_ax0]

        @R.function
        def forward(x: R.Tensor((3,), dtype="float32"), layer_norm_weight: R.Tensor((3,), dtype="float32"), layer_norm_bias: R.Tensor((3,), dtype="float32")) -> R.Tensor((3,), dtype="float32"):
            R.func_attr({"num_input": 1})
            cls = LayerNorm_1D_Expected
            with R.dataflow():
                layer_norm = R.call_tir(cls.layer_norm, (x, layer_norm_weight, layer_norm_bias), ty_args=[R.Tensor((3,), dtype="float32")])
                gv: R.Tensor((3,), dtype="float32") = layer_norm
                R.output(gv)
            return gv
    # fmt: on
    mod = LegalizeOps()(LayerNorm_1D)
    tvm_tirx.ir.assert_structural_equal(mod, LayerNorm_1D_Expected)


def test_layer_norm_fp16():
    # fmt: off
    @tvm_tirx.script.ir_module
    class LayerNorm:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float16"), gamma: R.Tensor((4, 5), "float16"), beta: R.Tensor((4, 5), "float16")) -> R.Tensor((2, 3, 4, 5), "float16"):
            gv: R.Tensor((2, 3, 4, 5), "float16") = R.nn.layer_norm(x, gamma, beta, axes=[-2, -1])
            return gv

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def layer_norm(
            x: T.Tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float16"),
            gamma: T.Tensor((T.int64(4), T.int64(5)), "float16"),
            beta: T.Tensor((T.int64(4), T.int64(5)), "float16"),
            T_layer_norm: T.Tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            x_sum = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3)))
            x_mean = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3)))
            x_var_sum = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3)))
            for ax0, ax1, k2, k3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("x_sum"):
                    v_ax0, v_ax1, v_k2, v_k3 = Ts.axis.remap("SSRR", [ax0, ax1, k2, k3])
                    Ts.reads(x[v_ax0, v_ax1, v_k2, v_k3])
                    Ts.writes(x_sum[v_ax0, v_ax1])
                    with Ts.init():
                        x_sum[v_ax0, v_ax1] = T.float32(0.0)
                    x_sum[v_ax0, v_ax1] = x_sum[v_ax0, v_ax1] + T.Cast("float32", x[v_ax0, v_ax1, v_k2, v_k3])
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("x_mean"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(x_sum[v_ax0, v_ax1])
                    Ts.writes(x_mean[v_ax0, v_ax1])
                    x_mean[v_ax0, v_ax1] = x_sum[v_ax0, v_ax1] / T.float32(20.0)
            for ax0, ax1, k2, k3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("x_var_sum"):
                    v_ax0, v_ax1, v_k2, v_k3 = Ts.axis.remap("SSRR", [ax0, ax1, k2, k3])
                    Ts.reads(x[v_ax0, v_ax1, v_k2, v_k3], x_mean[v_ax0, v_ax1])
                    Ts.writes(x_var_sum[v_ax0, v_ax1])
                    with Ts.init():
                        x_var_sum[v_ax0, v_ax1] = T.float32(0.0)
                    x_var_sum[v_ax0, v_ax1] = x_var_sum[v_ax0, v_ax1] + (T.Cast("float32", x[v_ax0, v_ax1, v_k2, v_k3]) - x_mean[v_ax0, v_ax1]) * (T.Cast("float32", x[v_ax0, v_ax1, v_k2, v_k3]) - x_mean[v_ax0, v_ax1])
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("T_layer_norm"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(x[v_ax0, v_ax1, v_ax2, v_ax3], x_mean[v_ax0, v_ax1], x_var_sum[v_ax0, v_ax1], gamma[v_ax2, v_ax3], beta[v_ax2, v_ax3])
                    Ts.writes(T_layer_norm[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_layer_norm[v_ax0, v_ax1, v_ax2, v_ax3] = T.Cast("float16", (T.Cast("float32", x[v_ax0, v_ax1, v_ax2, v_ax3]) - x_mean[v_ax0, v_ax1]) * T.rsqrt(x_var_sum[v_ax0, v_ax1] / T.float32(20.0) + T.float32(1.0000000000000001e-05))) * gamma[v_ax2, v_ax3] + beta[v_ax2, v_ax3]

        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), dtype="float16"), gamma: R.Tensor((4, 5), dtype="float16"), beta: R.Tensor((4, 5), dtype="float16")) -> R.Tensor((2, 3, 4, 5), dtype="float16"):
            gv = R.call_tir(Expected.layer_norm, (x, gamma, beta), ty_args=[R.Tensor((2, 3, 4, 5), dtype="float16")])
            return gv
    # fmt: on
    mod = LegalizeOps()(LayerNorm)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_layer_norm_symbolic():
    # fmt: off
    n = T.dynamic("n", "int64")
    s = T.dynamic("s", "int64")
    f = T.dynamic("f", "int64")

    @tvm_tirx.script.ir_module
    class LayerNorm:
        @R.function
        def main(x: R.Tensor((n, s, f), "float32"), gamma: R.Tensor((s, f), "float32"), beta: R.Tensor((s, f), "float32")) -> R.Tensor((n, s, f), "float32"):
            gv: R.Tensor((n, s, f), "float32") = R.nn.layer_norm(x, gamma, beta, axes=[1, 2])
            return gv

    n_main = T.dynamic("n", "int64")
    s_main = T.dynamic("s", "int64")
    f_main = T.dynamic("f", "int64")
    n_layer_norm = T.dynamic("n", "int64")
    s_layer_norm = T.dynamic("s", "int64")
    f_layer_norm = T.dynamic("f", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((n_main, s_main, f_main), "float32"), gamma: R.Tensor((s_main, f_main), "float32"), beta: R.Tensor((s_main, f_main), "float32")) -> R.Tensor((n_main, s_main, f_main), "float32"):
            gv = R.call_tir(Expected.layer_norm, (x, gamma, beta), ty_args=[R.Tensor((n_main, s_main, f_main), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def layer_norm(x: T.Tensor((n_layer_norm, s_layer_norm, f_layer_norm), "float32"), gamma: T.Tensor((s_layer_norm, f_layer_norm), "float32"), beta: T.Tensor((s_layer_norm, f_layer_norm), "float32"), T_layer_norm: T.Tensor((n_layer_norm, s_layer_norm, f_layer_norm), "float32")):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            x_sum = Ts.sblock_alloc_tensor((n_layer_norm,))
            x_mean = Ts.sblock_alloc_tensor((n_layer_norm,))
            x_var_sum = Ts.sblock_alloc_tensor((n_layer_norm,))
            for ax0, k1, k2 in T.grid(n_layer_norm, s_layer_norm, f_layer_norm):
                with Ts.sblock("x_sum"):
                    v_ax0, v_k1, v_k2 = Ts.axis.remap("SRR", [ax0, k1, k2])
                    Ts.reads(x[v_ax0, v_k1, v_k2])
                    Ts.writes(x_sum[v_ax0])
                    with Ts.init():
                        x_sum[v_ax0] = T.float32(0.0)
                    x_sum[v_ax0] = x_sum[v_ax0] + x[v_ax0, v_k1, v_k2]
            for ax0 in range(n_layer_norm):
                with Ts.sblock("x_mean"):
                    v_ax0 = Ts.axis.spatial(n_layer_norm, ax0)
                    Ts.reads(x_sum[v_ax0])
                    Ts.writes(x_mean[v_ax0])
                    x_mean[v_ax0] = x_sum[v_ax0] / (T.Cast("float32", s_layer_norm) * T.Cast("float32", f_layer_norm))
            for ax0, k1, k2 in T.grid(n_layer_norm, s_layer_norm, f_layer_norm):
                with Ts.sblock("x_var_sum"):
                    v_ax0, v_k1, v_k2 = Ts.axis.remap("SRR", [ax0, k1, k2])
                    Ts.reads(x[v_ax0, v_k1, v_k2], x_mean[v_ax0])
                    Ts.writes(x_var_sum[v_ax0])
                    with Ts.init():
                        x_var_sum[v_ax0] = T.float32(0.0)
                    x_var_sum[v_ax0] = x_var_sum[v_ax0] + (x[v_ax0, v_k1, v_k2] - x_mean[v_ax0]) * (x[v_ax0, v_k1, v_k2] - x_mean[v_ax0])
            for ax0, ax1, ax2 in T.grid(n_layer_norm, s_layer_norm, f_layer_norm):
                with Ts.sblock("T_layer_norm"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(x[v_ax0, v_ax1, v_ax2], x_mean[v_ax0], x_var_sum[v_ax0], gamma[v_ax1, v_ax2], beta[v_ax1, v_ax2])
                    Ts.writes(T_layer_norm[v_ax0, v_ax1, v_ax2])
                    T_layer_norm[v_ax0, v_ax1, v_ax2] = (x[v_ax0, v_ax1, v_ax2] - x_mean[v_ax0]) * T.rsqrt(x_var_sum[v_ax0] / (T.Cast("float32", s_layer_norm) * T.Cast("float32", f_layer_norm)) + T.float32(1.0000000000000001e-05)) * gamma[v_ax1, v_ax2] + beta[v_ax1, v_ax2]
    # fmt: on
    mod = LegalizeOps()(LayerNorm)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_group_norm():
    # fmt: off
    @tvm_tirx.script.ir_module
    class GroupNorm:
        @R.function
        def main(x: R.Tensor((2, 4, 4, 5), "float32"), gamma: R.Tensor((4,), "float32"), beta: R.Tensor((4,), "float32")) -> R.Tensor((2, 4, 4, 5), "float32"):
            gv: R.Tensor((2, 4, 4, 5), "float32") = R.nn.group_norm(x, gamma, beta, num_groups=2, channel_axis=1, axes=[2, 3])
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function(private=True)
        def group_norm(rxplaceholder: T.Tensor((T.int64(2), T.int64(4), T.int64(4), T.int64(5)), "float32"), rxplaceholder_1: T.Tensor((T.int64(4),), "float32"), rxplaceholder_2: T.Tensor((T.int64(4),), "float32"), T_reshape: T.Tensor((T.int64(2), T.int64(4), T.int64(4), T.int64(5)), "float32")):
            T.func_attr({"tirx.noalias": True})
            T_reshape_1 = Ts.sblock_alloc_tensor((T.int64(2), T.int64(2), T.int64(2), T.int64(4), T.int64(5)))
            rxplaceholder_red_temp_v0 = Ts.sblock_alloc_tensor((T.int64(2), T.int64(2)))
            rxplaceholder_red_temp_v1 = Ts.sblock_alloc_tensor((T.int64(2), T.int64(2)))
            T_reshape_2 = Ts.sblock_alloc_tensor((T.int64(2), T.int64(2)))
            T_reshape_3 = Ts.sblock_alloc_tensor((T.int64(2), T.int64(2)))
            T_group_norm = Ts.sblock_alloc_tensor((T.int64(2), T.int64(2), T.int64(2), T.int64(4), T.int64(5)))
            for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(2), T.int64(2), T.int64(4), T.int64(5)):
                with Ts.sblock("T_reshape"):
                    v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = Ts.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
                    Ts.reads(rxplaceholder[((v_ax1 * T.int64(2) + (v_ax4 // T.int64(5) + v_ax3) // T.int64(4) + v_ax2) // T.int64(4) + v_ax0) % T.int64(2), (v_ax1 * T.int64(2) + (v_ax4 // T.int64(5) + v_ax3) // T.int64(4) + v_ax2) % T.int64(4), (v_ax4 // T.int64(5) + v_ax3) % T.int64(4), v_ax4 % T.int64(5)])
                    Ts.writes(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
                    T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = rxplaceholder[((v_ax1 * T.int64(2) + (v_ax4 // T.int64(5) + v_ax3) // T.int64(4) + v_ax2) // T.int64(4) + v_ax0) % T.int64(2), (v_ax1 * T.int64(2) + (v_ax4 // T.int64(5) + v_ax3) // T.int64(4) + v_ax2) % T.int64(4), (v_ax4 // T.int64(5) + v_ax3) % T.int64(4), v_ax4 % T.int64(5)]
            for ax0, ax1, k2, k3, k4 in T.grid(T.int64(2), T.int64(2), T.int64(2), T.int64(4), T.int64(5)):
                with Ts.sblock("rxplaceholder_red_temp"):
                    v_ax0, v_ax1, v_k2, v_k3, v_k4 = Ts.axis.remap("SSRRR", [ax0, ax1, k2, k3, k4])
                    Ts.reads(T_reshape_1[v_ax0, v_ax1, v_k2, v_k3, v_k4])
                    Ts.writes(rxplaceholder_red_temp_v0[v_ax0, v_ax1], rxplaceholder_red_temp_v1[v_ax0, v_ax1])
                    with Ts.init():
                        rxplaceholder_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                        rxplaceholder_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
                    v_rxplaceholder_red_temp_v0: T.let[T.float32] = rxplaceholder_red_temp_v0[v_ax0, v_ax1] + T_reshape_1[v_ax0, v_ax1, v_k2, v_k3, v_k4]
                    v_rxplaceholder_red_temp_v1: T.let[T.float32] = rxplaceholder_red_temp_v1[v_ax0, v_ax1] + T_reshape_1[v_ax0, v_ax1, v_k2, v_k3, v_k4] * T_reshape_1[v_ax0, v_ax1, v_k2, v_k3, v_k4]
                    rxplaceholder_red_temp_v0[v_ax0, v_ax1] = v_rxplaceholder_red_temp_v0
                    rxplaceholder_red_temp_v1[v_ax0, v_ax1] = v_rxplaceholder_red_temp_v1
            for ax0, ax1 in T.grid(T.int64(2), T.int64(2)):
                with Ts.sblock("T_reshape_1"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(rxplaceholder_1[(v_ax0 * T.int64(2) + v_ax1) % T.int64(4)])
                    Ts.writes(T_reshape_2[v_ax0, v_ax1])
                    T_reshape_2[v_ax0, v_ax1] = rxplaceholder_1[(v_ax0 * T.int64(2) + v_ax1) % T.int64(4)]
            for ax0, ax1 in T.grid(T.int64(2), T.int64(2)):
                with Ts.sblock("T_reshape_2"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(rxplaceholder_2[(v_ax0 * T.int64(2) + v_ax1) % T.int64(4)])
                    Ts.writes(T_reshape_3[v_ax0, v_ax1])
                    T_reshape_3[v_ax0, v_ax1] = rxplaceholder_2[(v_ax0 * T.int64(2) + v_ax1) % T.int64(4)]
            for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(2), T.int64(2), T.int64(4), T.int64(5)):
                with Ts.sblock("T_group_norm"):
                    v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = Ts.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
                    Ts.reads(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4], rxplaceholder_red_temp_v0[v_ax0, v_ax1], rxplaceholder_red_temp_v1[v_ax0, v_ax1], T_reshape_2[v_ax1, v_ax2], T_reshape_3[v_ax1, v_ax2])
                    Ts.writes(T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
                    T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = (T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] - rxplaceholder_red_temp_v0[v_ax0, v_ax1] / T.float32(40)) * T.rsqrt(rxplaceholder_red_temp_v1[v_ax0, v_ax1] / T.float32(40) - rxplaceholder_red_temp_v0[v_ax0, v_ax1] / T.float32(40) * (rxplaceholder_red_temp_v0[v_ax0, v_ax1] / T.float32(40)) + T.float32(1.0000000000000001e-05)) * T_reshape_2[v_ax1, v_ax2] + T_reshape_3[v_ax1, v_ax2]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(4), T.int64(4), T.int64(5)):
                with Ts.sblock("T_reshape_3"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(T_group_norm[(((v_ax3 // T.int64(5) + v_ax2) // T.int64(4) + v_ax1) // T.int64(4) + v_ax0) % T.int64(2), ((v_ax3 // T.int64(5) + v_ax2) // T.int64(4) + v_ax1) % T.int64(4) // T.int64(2), ((v_ax3 // T.int64(5) + v_ax2) // T.int64(4) + v_ax1) % T.int64(2), (v_ax3 // T.int64(5) + v_ax2) % T.int64(4), v_ax3 % T.int64(5)])
                    Ts.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = T_group_norm[(((v_ax3 // T.int64(5) + v_ax2) // T.int64(4) + v_ax1) // T.int64(4) + v_ax0) % T.int64(2), ((v_ax3 // T.int64(5) + v_ax2) // T.int64(4) + v_ax1) % T.int64(4) // T.int64(2), ((v_ax3 // T.int64(5) + v_ax2) // T.int64(4) + v_ax1) % T.int64(2), (v_ax3 // T.int64(5) + v_ax2) % T.int64(4), v_ax3 % T.int64(5)]

        @R.function
        def main(x: R.Tensor((2, 4, 4, 5), dtype="float32"), gamma: R.Tensor((4,), dtype="float32"), beta: R.Tensor((4,), dtype="float32")) -> R.Tensor((2, 4, 4, 5), dtype="float32"):
            gv = R.call_tir(Expected.group_norm, (x, gamma, beta), ty_args=[R.Tensor((2, 4, 4, 5), dtype="float32")])
            return gv
    # fmt: on
    mod = LegalizeOps()(GroupNorm)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_group_norm_fp16():
    # fmt: off
    @tvm_tirx.script.ir_module
    class GroupNorm:
        @R.function
        def main(x: R.Tensor((2, 4, 4, 5), "float16"), gamma: R.Tensor((4,), "float16"), beta: R.Tensor((4,), "float16")) -> R.Tensor((2, 4, 4, 5), "float16"):
            gv: R.Tensor((2, 4, 4, 5), "float16") = R.nn.group_norm(x, gamma, beta, num_groups=2, channel_axis=1, axes=[2, 3])
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 4, 4, 5), dtype="float16"), gamma: R.Tensor((4,), dtype="float16"), beta: R.Tensor((4,), dtype="float16")) -> R.Tensor((2, 4, 4, 5), dtype="float16"):
            gv = R.call_tir(Expected.group_norm, (x, gamma, beta), ty_args=[R.Tensor((2, 4, 4, 5), dtype="float16")])
            return gv

        @Ts.function(private=True)
        def group_norm(rxplaceholder: T.Tensor((T.int64(2), T.int64(4), T.int64(4), T.int64(5)), "float16"), rxplaceholder_1: T.Tensor((T.int64(4),), "float16"), rxplaceholder_2: T.Tensor((T.int64(4),), "float16"), T_reshape: T.Tensor((T.int64(2), T.int64(4), T.int64(4), T.int64(5)), "float16")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            T_reshape_1 = Ts.sblock_alloc_tensor((T.int64(2), T.int64(2), T.int64(2), T.int64(4), T.int64(5)), "float16")
            T_cast = Ts.sblock_alloc_tensor((T.int64(2), T.int64(2), T.int64(2), T.int64(4), T.int64(5)))
            rxplaceholder_red_temp_v0 = Ts.sblock_alloc_tensor((T.int64(2), T.int64(2)))
            rxplaceholder_red_temp_v1 = Ts.sblock_alloc_tensor((T.int64(2), T.int64(2)))
            T_reshape_2 = Ts.sblock_alloc_tensor((T.int64(2), T.int64(2)), "float16")
            T_reshape_3 = Ts.sblock_alloc_tensor((T.int64(2), T.int64(2)), "float16")
            T_group_norm = Ts.sblock_alloc_tensor((T.int64(2), T.int64(2), T.int64(2), T.int64(4), T.int64(5)), "float16")
            for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(2), T.int64(2), T.int64(4), T.int64(5)):
                with Ts.sblock("T_reshape"):
                    v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = Ts.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
                    Ts.reads(rxplaceholder[((v_ax1 * T.int64(2) + (v_ax4 // T.int64(5) + v_ax3) // T.int64(4) + v_ax2) // T.int64(4) + v_ax0) % T.int64(2), (v_ax1 * T.int64(2) + (v_ax4 // T.int64(5) + v_ax3) // T.int64(4) + v_ax2) % T.int64(4), (v_ax4 // T.int64(5) + v_ax3) % T.int64(4), v_ax4 % T.int64(5)])
                    Ts.writes(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
                    T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = rxplaceholder[((v_ax1 * T.int64(2) + (v_ax4 // T.int64(5) + v_ax3) // T.int64(4) + v_ax2) // T.int64(4) + v_ax0) % T.int64(2), (v_ax1 * T.int64(2) + (v_ax4 // T.int64(5) + v_ax3) // T.int64(4) + v_ax2) % T.int64(4), (v_ax4 // T.int64(5) + v_ax3) % T.int64(4), v_ax4 % T.int64(5)]
            for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(2), T.int64(2), T.int64(4), T.int64(5)):
                with Ts.sblock("T_cast"):
                    v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = Ts.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
                    Ts.reads(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
                    Ts.writes(T_cast[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
                    T_cast[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = T.Cast("float32", T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            for ax0, ax1, k2, k3, k4 in T.grid(T.int64(2), T.int64(2), T.int64(2), T.int64(4), T.int64(5)):
                with Ts.sblock("rxplaceholder_red_temp"):
                    v_ax0, v_ax1, v_k2, v_k3, v_k4 = Ts.axis.remap("SSRRR", [ax0, ax1, k2, k3, k4])
                    Ts.reads(T_cast[v_ax0, v_ax1, v_k2, v_k3, v_k4])
                    Ts.writes(rxplaceholder_red_temp_v0[v_ax0, v_ax1], rxplaceholder_red_temp_v1[v_ax0, v_ax1])
                    with Ts.init():
                        rxplaceholder_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                        rxplaceholder_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
                    v_rxplaceholder_red_temp_v0: T.let[T.float32] = rxplaceholder_red_temp_v0[v_ax0, v_ax1] + T_cast[v_ax0, v_ax1, v_k2, v_k3, v_k4]
                    v_rxplaceholder_red_temp_v1: T.let[T.float32] = rxplaceholder_red_temp_v1[v_ax0, v_ax1] + T_cast[v_ax0, v_ax1, v_k2, v_k3, v_k4] * T_cast[v_ax0, v_ax1, v_k2, v_k3, v_k4]
                    rxplaceholder_red_temp_v0[v_ax0, v_ax1] = v_rxplaceholder_red_temp_v0
                    rxplaceholder_red_temp_v1[v_ax0, v_ax1] = v_rxplaceholder_red_temp_v1
            for ax0, ax1 in T.grid(T.int64(2), T.int64(2)):
                with Ts.sblock("T_reshape_1"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(rxplaceholder_1[(v_ax0 * T.int64(2) + v_ax1) % T.int64(4)])
                    Ts.writes(T_reshape_2[v_ax0, v_ax1])
                    T_reshape_2[v_ax0, v_ax1] = rxplaceholder_1[(v_ax0 * T.int64(2) + v_ax1) % T.int64(4)]
            for ax0, ax1 in T.grid(T.int64(2), T.int64(2)):
                with Ts.sblock("T_reshape_2"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(rxplaceholder_2[(v_ax0 * T.int64(2) + v_ax1) % T.int64(4)])
                    Ts.writes(T_reshape_3[v_ax0, v_ax1])
                    T_reshape_3[v_ax0, v_ax1] = rxplaceholder_2[(v_ax0 * T.int64(2) + v_ax1) % T.int64(4)]
            for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(2), T.int64(2), T.int64(4), T.int64(5)):
                with Ts.sblock("T_group_norm"):
                    v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = Ts.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
                    Ts.reads(T_cast[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4], rxplaceholder_red_temp_v0[v_ax0, v_ax1], rxplaceholder_red_temp_v1[v_ax0, v_ax1], T_reshape_2[v_ax1, v_ax2], T_reshape_3[v_ax1, v_ax2])
                    Ts.writes(T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
                    T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = T.Cast("float16", (T_cast[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] - rxplaceholder_red_temp_v0[v_ax0, v_ax1] / T.float32(40)) * T.rsqrt(rxplaceholder_red_temp_v1[v_ax0, v_ax1] / T.float32(40) - rxplaceholder_red_temp_v0[v_ax0, v_ax1] / T.float32(40) * (rxplaceholder_red_temp_v0[v_ax0, v_ax1] / T.float32(40)) + T.float32(1.0000000000000001e-05))) * T_reshape_2[v_ax1, v_ax2] + T_reshape_3[v_ax1, v_ax2]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(4), T.int64(4), T.int64(5)):
                with Ts.sblock("T_reshape_3"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(T_group_norm[(((v_ax3 // T.int64(5) + v_ax2) // T.int64(4) + v_ax1) // T.int64(4) + v_ax0) % T.int64(2), ((v_ax3 // T.int64(5) + v_ax2) // T.int64(4) + v_ax1) % T.int64(4) // T.int64(2), ((v_ax3 // T.int64(5) + v_ax2) // T.int64(4) + v_ax1) % T.int64(2), (v_ax3 // T.int64(5) + v_ax2) % T.int64(4), v_ax3 % T.int64(5)])
                    Ts.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = T_group_norm[(((v_ax3 // T.int64(5) + v_ax2) // T.int64(4) + v_ax1) // T.int64(4) + v_ax0) % T.int64(2), ((v_ax3 // T.int64(5) + v_ax2) // T.int64(4) + v_ax1) % T.int64(4) // T.int64(2), ((v_ax3 // T.int64(5) + v_ax2) // T.int64(4) + v_ax1) % T.int64(2), (v_ax3 // T.int64(5) + v_ax2) % T.int64(4), v_ax3 % T.int64(5)]
    # fmt: on

    mod = LegalizeOps()(GroupNorm)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_group_norm_symbolic():
    # fmt: off
    n = T.dynamic("n", "int64")
    c = T.dynamic("c", "int64")
    h = T.dynamic("h", "int64")
    w = T.dynamic("w", "int64")

    @tvm_tirx.script.ir_module
    class GroupNorm:
        @R.function
        def main(s: R.Shape([c]), x: R.Tensor((n, 4 * c, h, w), "float32"), gamma: R.Tensor((4 * c,), "float32"), beta: R.Tensor((4 * c,), "float32")) -> R.Tensor((n, 4 * c, h, w), "float32"):
            gv: R.Tensor((n, 4 * c, h, w), "float32") = R.nn.group_norm(x, gamma, beta, num_groups=4, channel_axis=1, axes=[2, 3])
            return gv

    n_group_norm = T.dynamic("n", "int64")
    h_group_norm = T.dynamic("h", "int64")
    w_group_norm = T.dynamic("w", "int64")
    n_main = T.dynamic("n", "int64")
    c = T.dynamic("c", "int64")
    h_main = T.dynamic("h", "int64")
    w_main = T.dynamic("w", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        group_norm_c = T.int64()

        @Ts.function(private=True)
        def group_norm(rxplaceholder: T.Tensor((n_group_norm, T.int64(4) * group_norm_c, h_group_norm, w_group_norm), "float32"), rxplaceholder_1: T.Tensor((T.int64(4) * group_norm_c,), "float32"), rxplaceholder_2: T.Tensor((T.int64(4) * group_norm_c,), "float32"), c: group_norm_c, T_reshape: T.Tensor((n_group_norm, T.int64(4) * group_norm_c, h_group_norm, w_group_norm), "float32")):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            T_reshape_1 = Ts.sblock_alloc_tensor((n_group_norm, T.int64(4), T.int64(4) * c // T.int64(4), h_group_norm, w_group_norm))
            rxplaceholder_red_temp_v0 = Ts.sblock_alloc_tensor((n_group_norm, T.int64(4)))
            rxplaceholder_red_temp_v1 = Ts.sblock_alloc_tensor((n_group_norm, T.int64(4)))
            T_reshape_2 = Ts.sblock_alloc_tensor((T.int64(4), T.int64(4) * c // T.int64(4)))
            T_reshape_3 = Ts.sblock_alloc_tensor((T.int64(4), T.int64(4) * c // T.int64(4)))
            T_group_norm = Ts.sblock_alloc_tensor((n_group_norm, T.int64(4), T.int64(4) * c // T.int64(4), h_group_norm, w_group_norm))
            for ax0, ax1, ax2, ax3, ax4 in T.grid(n_group_norm, T.int64(4), c, h_group_norm, w_group_norm):
                with Ts.sblock("T_reshape"):
                    v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = Ts.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
                    Ts.reads(rxplaceholder[((((v_ax0 * T.int64(4) + v_ax1) * c + v_ax2) * h_group_norm + v_ax3) * w_group_norm + v_ax4) // w_group_norm // h_group_norm // (c * T.int64(4)) % n_group_norm, ((((v_ax0 * T.int64(4) + v_ax1) * c + v_ax2) * h_group_norm + v_ax3) * w_group_norm + v_ax4) // w_group_norm // h_group_norm % (c * T.int64(4)), ((((v_ax0 * T.int64(4) + v_ax1) * c + v_ax2) * h_group_norm + v_ax3) * w_group_norm + v_ax4) // w_group_norm % h_group_norm, ((((v_ax0 * T.int64(4) + v_ax1) * c + v_ax2) * h_group_norm + v_ax3) * w_group_norm + v_ax4) % w_group_norm])
                    Ts.writes(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
                    T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = rxplaceholder[((((v_ax0 * T.int64(4) + v_ax1) * c + v_ax2) * h_group_norm + v_ax3) * w_group_norm + v_ax4) // w_group_norm // h_group_norm // (c * T.int64(4)) % n_group_norm, ((((v_ax0 * T.int64(4) + v_ax1) * c + v_ax2) * h_group_norm + v_ax3) * w_group_norm + v_ax4) // w_group_norm // h_group_norm % (c * T.int64(4)), ((((v_ax0 * T.int64(4) + v_ax1) * c + v_ax2) * h_group_norm + v_ax3) * w_group_norm + v_ax4) // w_group_norm % h_group_norm, ((((v_ax0 * T.int64(4) + v_ax1) * c + v_ax2) * h_group_norm + v_ax3) * w_group_norm + v_ax4) % w_group_norm]
            for ax0, ax1, k2, k3, k4 in T.grid(n_group_norm, T.int64(4), c, h_group_norm, w_group_norm):
                with Ts.sblock("rxplaceholder_red_temp"):
                    v_ax0, v_ax1, v_k2, v_k3, v_k4 = Ts.axis.remap("SSRRR", [ax0, ax1, k2, k3, k4])
                    Ts.reads(T_reshape_1[v_ax0, v_ax1, v_k2, v_k3, v_k4])
                    Ts.writes(rxplaceholder_red_temp_v0[v_ax0, v_ax1], rxplaceholder_red_temp_v1[v_ax0, v_ax1])
                    with Ts.init():
                        rxplaceholder_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                        rxplaceholder_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
                    v_rxplaceholder_red_temp_v0: T.let[T.float32] = rxplaceholder_red_temp_v0[v_ax0, v_ax1] + T_reshape_1[v_ax0, v_ax1, v_k2, v_k3, v_k4]
                    v_rxplaceholder_red_temp_v1: T.let[T.float32] = rxplaceholder_red_temp_v1[v_ax0, v_ax1] + T_reshape_1[v_ax0, v_ax1, v_k2, v_k3, v_k4] * T_reshape_1[v_ax0, v_ax1, v_k2, v_k3, v_k4]
                    rxplaceholder_red_temp_v0[v_ax0, v_ax1] = v_rxplaceholder_red_temp_v0
                    rxplaceholder_red_temp_v1[v_ax0, v_ax1] = v_rxplaceholder_red_temp_v1
            for ax0, ax1 in T.grid(T.int64(4), c):
                with Ts.sblock("T_reshape_1"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(rxplaceholder_1[(v_ax0 * c + v_ax1) % (c * T.int64(4))])
                    Ts.writes(T_reshape_2[v_ax0, v_ax1])
                    T_reshape_2[v_ax0, v_ax1] = rxplaceholder_1[(v_ax0 * c + v_ax1) % (c * T.int64(4))]
            for ax0, ax1 in T.grid(T.int64(4), c):
                with Ts.sblock("T_reshape_2"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(rxplaceholder_2[(v_ax0 * c + v_ax1) % (c * T.int64(4))])
                    Ts.writes(T_reshape_3[v_ax0, v_ax1])
                    T_reshape_3[v_ax0, v_ax1] = rxplaceholder_2[(v_ax0 * c + v_ax1) % (c * T.int64(4))]
            for ax0, ax1, ax2, ax3, ax4 in T.grid(n_group_norm, T.int64(4), c, h_group_norm, w_group_norm):
                with Ts.sblock("T_group_norm"):
                    v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = Ts.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
                    Ts.reads(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4], rxplaceholder_red_temp_v0[v_ax0, v_ax1], rxplaceholder_red_temp_v1[v_ax0, v_ax1], T_reshape_2[v_ax1, v_ax2], T_reshape_3[v_ax1, v_ax2])
                    Ts.writes(T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
                    T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = (T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] - rxplaceholder_red_temp_v0[v_ax0, v_ax1] / (T.Cast("float32", c) * T.Cast("float32", h_group_norm) * T.Cast("float32", w_group_norm))) * T.rsqrt(rxplaceholder_red_temp_v1[v_ax0, v_ax1] / (T.Cast("float32", c) * T.Cast("float32", h_group_norm) * T.Cast("float32", w_group_norm)) - rxplaceholder_red_temp_v0[v_ax0, v_ax1] / (T.Cast("float32", c) * T.Cast("float32", h_group_norm) * T.Cast("float32", w_group_norm)) * (rxplaceholder_red_temp_v0[v_ax0, v_ax1] / (T.Cast("float32", c) * T.Cast("float32", h_group_norm) * T.Cast("float32", w_group_norm))) + T.float32(1.0000000000000001e-05)) * T_reshape_2[v_ax1, v_ax2] + T_reshape_3[v_ax1, v_ax2]
            for ax0, ax1, ax2, ax3 in T.grid(n_group_norm, c * T.int64(4), h_group_norm, w_group_norm):
                with Ts.sblock("T_reshape_3"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(T_group_norm[(((v_ax0 * c * T.int64(4) + v_ax1) * h_group_norm + v_ax2) * w_group_norm + v_ax3) // w_group_norm // h_group_norm // c // T.int64(4) % n_group_norm, (((v_ax0 * c * T.int64(4) + v_ax1) * h_group_norm + v_ax2) * w_group_norm + v_ax3) // w_group_norm // h_group_norm // c % T.int64(4), (((v_ax0 * c * T.int64(4) + v_ax1) * h_group_norm + v_ax2) * w_group_norm + v_ax3) // w_group_norm // h_group_norm % c, (((v_ax0 * c * T.int64(4) + v_ax1) * h_group_norm + v_ax2) * w_group_norm + v_ax3) // w_group_norm % h_group_norm, (((v_ax0 * c * T.int64(4) + v_ax1) * h_group_norm + v_ax2) * w_group_norm + v_ax3) % w_group_norm])
                    Ts.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = T_group_norm[(((v_ax0 * c * T.int64(4) + v_ax1) * h_group_norm + v_ax2) * w_group_norm + v_ax3) // w_group_norm // h_group_norm // c // T.int64(4) % n_group_norm, (((v_ax0 * c * T.int64(4) + v_ax1) * h_group_norm + v_ax2) * w_group_norm + v_ax3) // w_group_norm // h_group_norm // c % T.int64(4), (((v_ax0 * c * T.int64(4) + v_ax1) * h_group_norm + v_ax2) * w_group_norm + v_ax3) // w_group_norm // h_group_norm % c, (((v_ax0 * c * T.int64(4) + v_ax1) * h_group_norm + v_ax2) * w_group_norm + v_ax3) // w_group_norm % h_group_norm, (((v_ax0 * c * T.int64(4) + v_ax1) * h_group_norm + v_ax2) * w_group_norm + v_ax3) % w_group_norm]

        @R.function
        def main(s: R.Shape([c]), x: R.Tensor((n_main, 4 * c, h_main, w_main), dtype="float32"), gamma: R.Tensor((4 * c,), dtype="float32"), beta: R.Tensor((4 * c,), dtype="float32")) -> R.Tensor((n_main, 4 * c, h_main, w_main), dtype="float32"):
            gv = R.call_tir(Expected.group_norm, (x, gamma, beta, c), ty_args=[R.Tensor((n_main, 4 * c, h_main, w_main), dtype="float32")])
            return gv
    # fmt: on
    mod = LegalizeOps()(GroupNorm)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_rms_norm():
    # fmt: off
    @tvm_tirx.script.ir_module
    class RMSNorm:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32"), weight: R.Tensor((4, 5), "float32")) -> R.Tensor((2, 3, 4, 5), "float32"):
            gv: R.Tensor((2, 3, 4, 5), "float32") = R.nn.rms_norm(x, weight, axes=[-2, -1])
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function(private=True)
        def rms_norm(A: T.Tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"), B: T.Tensor((T.int64(4), T.int64(5)), "float32"), T_cast: T.Tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            T_cast_1 = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)))
            T_multiply = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)))
            T_multiply_red = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3)))
            rsqrt = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3)))
            T_cast_2 = Ts.sblock_alloc_tensor((T.int64(4), T.int64(5)))
            T_rms_norm = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)))
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("T_cast"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                    Ts.writes(T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("T_multiply"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    Ts.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3] * T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3]
            for ax0, ax1, k2, k3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("T_multiply_red"):
                    v_ax0, v_ax1, v_k2, v_k3 = Ts.axis.remap("SSRR", [ax0, ax1, k2, k3])
                    Ts.reads(T_multiply[v_ax0, v_ax1, v_k2, v_k3])
                    Ts.writes(T_multiply_red[v_ax0, v_ax1])
                    with Ts.init():
                        T_multiply_red[v_ax0, v_ax1] = T.float32(0)
                    T_multiply_red[v_ax0, v_ax1] = T_multiply_red[v_ax0, v_ax1] + T_multiply[v_ax0, v_ax1, v_k2, v_k3]
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("rsqrt"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(T_multiply_red[v_ax0, v_ax1])
                    Ts.writes(rsqrt[v_ax0, v_ax1])
                    rsqrt[v_ax0, v_ax1] = T.rsqrt(T_multiply_red[v_ax0, v_ax1] / T.float32(20) + T.float32(1.0000000000000001e-05))
            for ax0, ax1 in T.grid(T.int64(4), T.int64(5)):
                with Ts.sblock("T_cast_1"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(B[v_ax0, v_ax1])
                    Ts.writes(T_cast_2[v_ax0, v_ax1])
                    T_cast_2[v_ax0, v_ax1] = B[v_ax0, v_ax1]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("T_rms_norm"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(rsqrt[v_ax0, v_ax1], T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3], T_cast_2[v_ax2, v_ax3])
                    Ts.writes(T_rms_norm[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_rms_norm[v_ax0, v_ax1, v_ax2, v_ax3] = rsqrt[v_ax0, v_ax1] * T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3] * T_cast_2[v_ax2, v_ax3]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("T_cast_2"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(T_rms_norm[v_ax0, v_ax1, v_ax2, v_ax3])
                    Ts.writes(T_cast[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_cast[v_ax0, v_ax1, v_ax2, v_ax3] = T_rms_norm[v_ax0, v_ax1, v_ax2, v_ax3]

        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), dtype="float32"), weight: R.Tensor((4, 5), dtype="float32")) -> R.Tensor((2, 3, 4, 5), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.rms_norm, (x, weight), ty_args=[R.Tensor((2, 3, 4, 5), dtype="float32")])
            return gv
    # fmt: on
    mod = LegalizeOps()(RMSNorm)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_rms_norm_fp16():
    # fmt: off
    @tvm_tirx.script.ir_module
    class RMSNorm:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float16"), weight: R.Tensor((4, 5), "float16")) -> R.Tensor((2, 3, 4, 5), "float16"):
            gv: R.Tensor((2, 3, 4, 5), "float16") = R.nn.rms_norm(x, weight, axes=[-2, -1])
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function(private=True)
        def rms_norm(A: T.Tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float16"), B: T.Tensor((T.int64(4), T.int64(5)), "float16"), T_cast: T.Tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float16")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            T_cast_1 = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)))
            T_multiply = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)))
            T_multiply_red = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3)))
            rsqrt = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3)))
            T_cast_2 = Ts.sblock_alloc_tensor((T.int64(4), T.int64(5)))
            T_rms_norm = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)))
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("T_cast"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                    Ts.writes(T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3] = T.Cast("float32", A[v_ax0, v_ax1, v_ax2, v_ax3])
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("T_multiply"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    Ts.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3] * T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3]
            for ax0, ax1, k2, k3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("T_multiply_red"):
                    v_ax0, v_ax1, v_k2, v_k3 = Ts.axis.remap("SSRR", [ax0, ax1, k2, k3])
                    Ts.reads(T_multiply[v_ax0, v_ax1, v_k2, v_k3])
                    Ts.writes(T_multiply_red[v_ax0, v_ax1])
                    with Ts.init():
                        T_multiply_red[v_ax0, v_ax1] = T.float32(0)
                    T_multiply_red[v_ax0, v_ax1] = T_multiply_red[v_ax0, v_ax1] + T_multiply[v_ax0, v_ax1, v_k2, v_k3]
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("rsqrt"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(T_multiply_red[v_ax0, v_ax1])
                    Ts.writes(rsqrt[v_ax0, v_ax1])
                    rsqrt[v_ax0, v_ax1] = T.rsqrt(T_multiply_red[v_ax0, v_ax1] / T.float32(20) + T.float32(1.0000000000000001e-05))
            for ax0, ax1 in T.grid(T.int64(4), T.int64(5)):
                with Ts.sblock("T_cast_1"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(B[v_ax0, v_ax1])
                    Ts.writes(T_cast_2[v_ax0, v_ax1])
                    T_cast_2[v_ax0, v_ax1] = T.Cast("float32", B[v_ax0, v_ax1])
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("T_rms_norm"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(rsqrt[v_ax0, v_ax1], T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3], T_cast_2[v_ax2, v_ax3])
                    Ts.writes(T_rms_norm[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_rms_norm[v_ax0, v_ax1, v_ax2, v_ax3] = rsqrt[v_ax0, v_ax1] * T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3] * T_cast_2[v_ax2, v_ax3]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("T_cast_2"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(T_rms_norm[v_ax0, v_ax1, v_ax2, v_ax3])
                    Ts.writes(T_cast[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_cast[v_ax0, v_ax1, v_ax2, v_ax3] = T.Cast("float16", T_rms_norm[v_ax0, v_ax1, v_ax2, v_ax3])

        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), dtype="float16"), weight: R.Tensor((4, 5), dtype="float16")) -> R.Tensor((2, 3, 4, 5), dtype="float16"):
            cls = Expected
            gv = R.call_tir(cls.rms_norm, (x, weight), ty_args=[R.Tensor((2, 3, 4, 5), dtype="float16")])
            return gv
    # fmt: on
    mod = LegalizeOps()(RMSNorm)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_rms_norm_symbolic():
    # fmt: off
    n = T.dynamic("n", "int64")
    s = T.dynamic("s", "int64")
    f = T.dynamic("f", "int64")

    @tvm_tirx.script.ir_module
    class RMSNorm:
        @R.function
        def main(x: R.Tensor((n, s, f), "float32"), weight: R.Tensor((s, f), "float32")) -> R.Tensor((n, s, f), "float32"):
            gv: R.Tensor((n, s, f), "float32") = R.nn.rms_norm(x, weight, axes=[1, 2])
            return gv

    n_rms_norm = T.dynamic("n", "int64")
    s_rms_norm = T.dynamic("s", "int64")
    f_rms_norm = T.dynamic("f", "int64")
    n_main = T.dynamic("n", "int64")
    s_main = T.dynamic("s", "int64")
    f_main = T.dynamic("f", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function(private=True)
        def rms_norm(A: T.Tensor((n_rms_norm, s_rms_norm, f_rms_norm), "float32"), B: T.Tensor((s_rms_norm, f_rms_norm), "float32"), T_cast: T.Tensor((n_rms_norm, s_rms_norm, f_rms_norm), "float32")):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            T_cast_1 = Ts.sblock_alloc_tensor((n_rms_norm, s_rms_norm, f_rms_norm))
            T_multiply = Ts.sblock_alloc_tensor((n_rms_norm, s_rms_norm, f_rms_norm))
            T_multiply_red = Ts.sblock_alloc_tensor((n_rms_norm,))
            rsqrt = Ts.sblock_alloc_tensor((n_rms_norm,))
            T_cast_2 = Ts.sblock_alloc_tensor((s_rms_norm, f_rms_norm))
            T_rms_norm = Ts.sblock_alloc_tensor((n_rms_norm, s_rms_norm, f_rms_norm))
            for ax0, ax1, ax2 in T.grid(n_rms_norm, s_rms_norm, f_rms_norm):
                with Ts.sblock("T_cast"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(A[v_ax0, v_ax1, v_ax2])
                    Ts.writes(T_cast_1[v_ax0, v_ax1, v_ax2])
                    T_cast_1[v_ax0, v_ax1, v_ax2] = A[v_ax0, v_ax1, v_ax2]
            for ax0, ax1, ax2 in T.grid(n_rms_norm, s_rms_norm, f_rms_norm):
                with Ts.sblock("T_multiply"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(T_cast_1[v_ax0, v_ax1, v_ax2])
                    Ts.writes(T_multiply[v_ax0, v_ax1, v_ax2])
                    T_multiply[v_ax0, v_ax1, v_ax2] = T_cast_1[v_ax0, v_ax1, v_ax2] * T_cast_1[v_ax0, v_ax1, v_ax2]
            for ax0, k1, k2 in T.grid(n_rms_norm, s_rms_norm, f_rms_norm):
                with Ts.sblock("T_multiply_red"):
                    v_ax0, v_k1, v_k2 = Ts.axis.remap("SRR", [ax0, k1, k2])
                    Ts.reads(T_multiply[v_ax0, v_k1, v_k2])
                    Ts.writes(T_multiply_red[v_ax0])
                    with Ts.init():
                        T_multiply_red[v_ax0] = T.float32(0)
                    T_multiply_red[v_ax0] = T_multiply_red[v_ax0] + T_multiply[v_ax0, v_k1, v_k2]
            for ax0 in range(n_rms_norm):
                with Ts.sblock("rsqrt"):
                    v_ax0 = Ts.axis.spatial(n_rms_norm, ax0)
                    Ts.reads(T_multiply_red[v_ax0])
                    Ts.writes(rsqrt[v_ax0])
                    rsqrt[v_ax0] = T.rsqrt(T_multiply_red[v_ax0] / (T.Cast("float32", s_rms_norm) * T.Cast("float32", f_rms_norm)) + T.float32(1.0000000000000001e-05))
            for ax0, ax1 in T.grid(s_rms_norm, f_rms_norm):
                with Ts.sblock("T_cast_1"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(B[v_ax0, v_ax1])
                    Ts.writes(T_cast_2[v_ax0, v_ax1])
                    T_cast_2[v_ax0, v_ax1] = B[v_ax0, v_ax1]
            for ax0, ax1, ax2 in T.grid(n_rms_norm, s_rms_norm, f_rms_norm):
                with Ts.sblock("T_rms_norm"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(rsqrt[v_ax0], T_cast_1[v_ax0, v_ax1, v_ax2], T_cast_2[v_ax1, v_ax2])
                    Ts.writes(T_rms_norm[v_ax0, v_ax1, v_ax2])
                    T_rms_norm[v_ax0, v_ax1, v_ax2] = rsqrt[v_ax0] * T_cast_1[v_ax0, v_ax1, v_ax2] * T_cast_2[v_ax1, v_ax2]
            for ax0, ax1, ax2 in T.grid(n_rms_norm, s_rms_norm, f_rms_norm):
                with Ts.sblock("T_cast_2"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(T_rms_norm[v_ax0, v_ax1, v_ax2])
                    Ts.writes(T_cast[v_ax0, v_ax1, v_ax2])
                    T_cast[v_ax0, v_ax1, v_ax2] = T_rms_norm[v_ax0, v_ax1, v_ax2]

        @R.function
        def main(x: R.Tensor((n_main, s_main, f_main), dtype="float32"), weight: R.Tensor((s_main, f_main), dtype="float32")) -> R.Tensor((n_main, s_main, f_main), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.rms_norm, (x, weight), ty_args=[R.Tensor((n_main, s_main, f_main), dtype="float32")])
            return gv
    # fmt: on
    mod = LegalizeOps()(RMSNorm)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_rms_norm_no_bias():
    # fmt: off
    @tvm_tirx.script.ir_module
    class RMSNorm:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32"), weight: R.Tensor((4, 5), "float32")) -> R.Tensor((2, 3, 4, 5), "float32"):
            gv: R.Tensor((2, 3, 4, 5), "float32") = R.nn.rms_norm(x, weight, axes=[-2, -1])
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function(private=True)
        def rms_norm(A: T.Tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"), B: T.Tensor((T.int64(4), T.int64(5)), "float32"), T_cast: T.Tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            T_cast_1 = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)))
            T_multiply = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)))
            T_multiply_red = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3)))
            rsqrt = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3)))
            T_cast_2 = Ts.sblock_alloc_tensor((T.int64(4), T.int64(5)))
            T_rms_norm = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)))
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("T_cast"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                    Ts.writes(T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("T_multiply"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    Ts.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3] * T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3]
            for ax0, ax1, k2, k3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("T_multiply_red"):
                    v_ax0, v_ax1, v_k2, v_k3 = Ts.axis.remap("SSRR", [ax0, ax1, k2, k3])
                    Ts.reads(T_multiply[v_ax0, v_ax1, v_k2, v_k3])
                    Ts.writes(T_multiply_red[v_ax0, v_ax1])
                    with Ts.init():
                        T_multiply_red[v_ax0, v_ax1] = T.float32(0)
                    T_multiply_red[v_ax0, v_ax1] = T_multiply_red[v_ax0, v_ax1] + T_multiply[v_ax0, v_ax1, v_k2, v_k3]
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("rsqrt"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(T_multiply_red[v_ax0, v_ax1])
                    Ts.writes(rsqrt[v_ax0, v_ax1])
                    rsqrt[v_ax0, v_ax1] = T.rsqrt(T_multiply_red[v_ax0, v_ax1] / T.float32(20) + T.float32(1.0000000000000001e-05))
            for ax0, ax1 in T.grid(T.int64(4), T.int64(5)):
                with Ts.sblock("T_cast_1"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(B[v_ax0, v_ax1])
                    Ts.writes(T_cast_2[v_ax0, v_ax1])
                    T_cast_2[v_ax0, v_ax1] = B[v_ax0, v_ax1]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("T_rms_norm"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(rsqrt[v_ax0, v_ax1], T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3], T_cast_2[v_ax2, v_ax3])
                    Ts.writes(T_rms_norm[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_rms_norm[v_ax0, v_ax1, v_ax2, v_ax3] = rsqrt[v_ax0, v_ax1] * T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3] * T_cast_2[v_ax2, v_ax3]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("T_cast_2"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(T_rms_norm[v_ax0, v_ax1, v_ax2, v_ax3])
                    Ts.writes(T_cast[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_cast[v_ax0, v_ax1, v_ax2, v_ax3] = T_rms_norm[v_ax0, v_ax1, v_ax2, v_ax3]

        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), dtype="float32"), weight: R.Tensor((4, 5), dtype="float32")) -> R.Tensor((2, 3, 4, 5), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.rms_norm, (x, weight), ty_args=[R.Tensor((2, 3, 4, 5), dtype="float32")])
            return gv
    # fmt: on
    mod = LegalizeOps()(RMSNorm)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_attention():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Attention:
        @R.function
        def main(q: R.Tensor((4, 16, 32, 8), "float32"), k: R.Tensor((4, 8, 32, 8), "float32"), v: R.Tensor((4, 8, 32, 16), "float32"), bias: R.Tensor((4, 32, 16, 8), "float32")):
            gv: R.Tensor((4, 16, 32, 16), "float32") = R.nn.attention(q, k, v, bias, scale=T.FloatImm("float32", 0.1), causal_mask="TopLeft")
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function(private=True)
        def attention_bias(q: T.Tensor((T.int64(4), T.int64(16), T.int64(32), T.int64(8)), "float32"), k: T.Tensor((T.int64(4), T.int64(8), T.int64(32), T.int64(8)), "float32"), v: T.Tensor((T.int64(4), T.int64(8), T.int64(32), T.int64(16)), "float32"), bias: T.Tensor((T.int64(4), T.int64(32), T.int64(16), T.int64(8)), "float32"), T_transpose: T.Tensor((T.int64(4), T.int64(16), T.int64(32), T.int64(16)), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            T_transpose_1 = Ts.sblock_alloc_tensor((T.int64(4), T.int64(32), T.int64(16), T.int64(8)))
            T_reshape = Ts.sblock_alloc_tensor((T.int64(128), T.int64(16), T.int64(8)))
            T_transpose_2 = Ts.sblock_alloc_tensor((T.int64(4), T.int64(32), T.int64(8), T.int64(8)))
            T_reshape_1 = Ts.sblock_alloc_tensor((T.int64(128), T.int64(8), T.int64(8)))
            T_batch_matmul_NT = Ts.sblock_alloc_tensor((T.int64(128), T.int64(16), T.int64(8)))
            T_multiply = Ts.sblock_alloc_tensor((T.int64(128), T.int64(16), T.int64(8)))
            T_reshape_2 = Ts.sblock_alloc_tensor((T.int64(4), T.int64(32), T.int64(16), T.int64(8)))
            T_add = Ts.sblock_alloc_tensor((T.int64(4), T.int64(32), T.int64(16), T.int64(8)))
            T_reshape_3 = Ts.sblock_alloc_tensor((T.int64(128), T.int64(16), T.int64(8)))
            trilu = Ts.sblock_alloc_tensor((T.int64(128), T.int64(16), T.int64(8)))
            trilu_red = Ts.sblock_alloc_tensor((T.int64(128), T.int64(16), T.int64(1)))
            T_subtract = Ts.sblock_alloc_tensor((T.int64(128), T.int64(16), T.int64(8)))
            compute = Ts.sblock_alloc_tensor((T.int64(128), T.int64(16), T.int64(8)))
            trilu_1 = Ts.sblock_alloc_tensor((T.int64(128), T.int64(16), T.int64(8)))
            trilu_red_1 = Ts.sblock_alloc_tensor((T.int64(128), T.int64(16), T.int64(1)))
            T_divide = Ts.sblock_alloc_tensor((T.int64(128), T.int64(16), T.int64(8)))
            T_transpose_3 = Ts.sblock_alloc_tensor((T.int64(4), T.int64(32), T.int64(8), T.int64(16)))
            T_reshape_4 = Ts.sblock_alloc_tensor((T.int64(128), T.int64(8), T.int64(16)))
            T_batch_matmul_NN = Ts.sblock_alloc_tensor((T.int64(128), T.int64(16), T.int64(16)))
            T_reshape_5 = Ts.sblock_alloc_tensor((T.int64(4), T.int64(32), T.int64(16), T.int64(16)))
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(4), T.int64(32), T.int64(16), T.int64(8)):
                with Ts.sblock("T_transpose"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(q[v_ax0, v_ax2, v_ax1, v_ax3])
                    Ts.writes(T_transpose_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_transpose_1[v_ax0, v_ax1, v_ax2, v_ax3] = q[v_ax0, v_ax2, v_ax1, v_ax3]
            for ax0, ax1, ax2 in T.grid(T.int64(128), T.int64(16), T.int64(8)):
                with Ts.sblock("T_reshape"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(T_transpose_1[((v_ax2 // T.int64(8) + v_ax1) // T.int64(16) + v_ax0) % T.int64(128) // T.int64(32), ((v_ax2 // T.int64(8) + v_ax1) // T.int64(16) + v_ax0) % T.int64(32), (v_ax2 // T.int64(8) + v_ax1) % T.int64(16), v_ax2 % T.int64(8)])
                    Ts.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                    T_reshape[v_ax0, v_ax1, v_ax2] = T_transpose_1[((v_ax2 // T.int64(8) + v_ax1) // T.int64(16) + v_ax0) % T.int64(128) // T.int64(32), ((v_ax2 // T.int64(8) + v_ax1) // T.int64(16) + v_ax0) % T.int64(32), (v_ax2 // T.int64(8) + v_ax1) % T.int64(16), v_ax2 % T.int64(8)]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(4), T.int64(32), T.int64(8), T.int64(8)):
                with Ts.sblock("T_transpose_1"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(k[v_ax0, v_ax2, v_ax1, v_ax3])
                    Ts.writes(T_transpose_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_transpose_2[v_ax0, v_ax1, v_ax2, v_ax3] = k[v_ax0, v_ax2, v_ax1, v_ax3]
            for ax0, ax1, ax2 in T.grid(T.int64(128), T.int64(8), T.int64(8)):
                with Ts.sblock("T_reshape_1"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(T_transpose_2[((v_ax2 // T.int64(8) + v_ax1) // T.int64(8) + v_ax0) % T.int64(128) // T.int64(32), ((v_ax2 // T.int64(8) + v_ax1) // T.int64(8) + v_ax0) % T.int64(32), (v_ax2 // T.int64(8) + v_ax1) % T.int64(8), v_ax2 % T.int64(8)])
                    Ts.writes(T_reshape_1[v_ax0, v_ax1, v_ax2])
                    T_reshape_1[v_ax0, v_ax1, v_ax2] = T_transpose_2[((v_ax2 // T.int64(8) + v_ax1) // T.int64(8) + v_ax0) % T.int64(128) // T.int64(32), ((v_ax2 // T.int64(8) + v_ax1) // T.int64(8) + v_ax0) % T.int64(32), (v_ax2 // T.int64(8) + v_ax1) % T.int64(8), v_ax2 % T.int64(8)]
            for b_index, i, j, k_1 in T.grid(T.int64(128), T.int64(16), T.int64(8), T.int64(8)):
                with Ts.sblock("T_batch_matmul_NT"):
                    v_b, v_i, v_j, v_k = Ts.axis.remap("SSSR", [b_index, i, j, k_1])
                    Ts.reads(T_reshape[v_b, v_i, v_k], T_reshape_1[v_b, v_j, v_k])
                    Ts.writes(T_batch_matmul_NT[v_b, v_i, v_j])
                    Ts.sblock_attr({"layout_free_placeholders": [T_reshape_1]})
                    with Ts.init():
                        T_batch_matmul_NT[v_b, v_i, v_j] = T.float32(0.0)
                    T_batch_matmul_NT[v_b, v_i, v_j] = T_batch_matmul_NT[v_b, v_i, v_j] + T_reshape[v_b, v_i, v_k] * T_reshape_1[v_b, v_j, v_k]
            for ax0, ax1, ax2 in T.grid(T.int64(128), T.int64(16), T.int64(8)):
                with Ts.sblock("T_multiply"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(T_batch_matmul_NT[v_ax0, v_ax1, v_ax2])
                    Ts.writes(T_multiply[v_ax0, v_ax1, v_ax2])
                    T_multiply[v_ax0, v_ax1, v_ax2] = T_batch_matmul_NT[v_ax0, v_ax1, v_ax2] * T.float32(0.10000000000000001)
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(4), T.int64(32), T.int64(16), T.int64(8)):
                with Ts.sblock("T_reshape_2"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(T_multiply[(v_ax0 * T.int64(32) + (v_ax3 // T.int64(8) + v_ax2) // T.int64(16) + v_ax1) % T.int64(128), (v_ax3 // T.int64(8) + v_ax2) % T.int64(16), v_ax3 % T.int64(8)])
                    Ts.writes(T_reshape_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_2[v_ax0, v_ax1, v_ax2, v_ax3] = T_multiply[(v_ax0 * T.int64(32) + (v_ax3 // T.int64(8) + v_ax2) // T.int64(16) + v_ax1) % T.int64(128), (v_ax3 // T.int64(8) + v_ax2) % T.int64(16), v_ax3 % T.int64(8)]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(4), T.int64(32), T.int64(16), T.int64(8)):
                with Ts.sblock("T_add"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(T_reshape_2[v_ax0, v_ax1, v_ax2, v_ax3], bias[v_ax0, v_ax1, v_ax2, v_ax3])
                    Ts.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_add[v_ax0, v_ax1, v_ax2, v_ax3] = T_reshape_2[v_ax0, v_ax1, v_ax2, v_ax3] + bias[v_ax0, v_ax1, v_ax2, v_ax3]
            for ax0, ax1, ax2 in T.grid(T.int64(128), T.int64(16), T.int64(8)):
                with Ts.sblock("T_reshape_3"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(T_add[((v_ax2 // T.int64(8) + v_ax1) // T.int64(16) + v_ax0) % T.int64(128) // T.int64(32), ((v_ax2 // T.int64(8) + v_ax1) // T.int64(16) + v_ax0) % T.int64(32), (v_ax2 // T.int64(8) + v_ax1) % T.int64(16), v_ax2 % T.int64(8)])
                    Ts.writes(T_reshape_3[v_ax0, v_ax1, v_ax2])
                    T_reshape_3[v_ax0, v_ax1, v_ax2] = T_add[((v_ax2 // T.int64(8) + v_ax1) // T.int64(16) + v_ax0) % T.int64(128) // T.int64(32), ((v_ax2 // T.int64(8) + v_ax1) // T.int64(16) + v_ax0) % T.int64(32), (v_ax2 // T.int64(8) + v_ax1) % T.int64(16), v_ax2 % T.int64(8)]
            for i0, i1, i2 in T.grid(T.int64(128), T.int64(16), T.int64(8)):
                with Ts.sblock("trilu"):
                    v_i0, v_i1, v_i2 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(T_reshape_3[v_i0, v_i1, v_i2])
                    Ts.writes(trilu[v_i0, v_i1, v_i2])
                    trilu[v_i0, v_i1, v_i2] = T.Select(v_i2 <= v_i1, T_reshape_3[v_i0, v_i1, v_i2], T.float32(0.0))
            for ax0, ax1, ax2, k2 in T.grid(T.int64(128), T.int64(16), T.int64(1), T.int64(8)):
                with Ts.sblock("trilu_red"):
                    v_ax0, v_ax1, v_ax2, v_k2 = Ts.axis.remap("SSSR", [ax0, ax1, ax2, k2])
                    Ts.reads(trilu[v_ax0, v_ax1, v_k2])
                    Ts.writes(trilu_red[v_ax0, v_ax1, v_ax2])
                    with Ts.init():
                        trilu_red[v_ax0, v_ax1, v_ax2] = T.float32(-340282346638528859811704183484516925440.0)
                    trilu_red[v_ax0, v_ax1, v_ax2] = T.max(trilu_red[v_ax0, v_ax1, v_ax2], trilu[v_ax0, v_ax1, v_k2])
            for ax0, ax1, ax2 in T.grid(T.int64(128), T.int64(16), T.int64(8)):
                with Ts.sblock("T_subtract"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(trilu[v_ax0, v_ax1, v_ax2], trilu_red[v_ax0, v_ax1, T.int64(0)])
                    Ts.writes(T_subtract[v_ax0, v_ax1, v_ax2])
                    T_subtract[v_ax0, v_ax1, v_ax2] = trilu[v_ax0, v_ax1, v_ax2] - trilu_red[v_ax0, v_ax1, T.int64(0)]
            for i0, i1, i2 in T.grid(T.int64(128), T.int64(16), T.int64(8)):
                with Ts.sblock("compute"):
                    v_i0, v_i1, v_i2 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(T_subtract[v_i0, v_i1, v_i2])
                    Ts.writes(compute[v_i0, v_i1, v_i2])
                    compute[v_i0, v_i1, v_i2] = T.exp(T_subtract[v_i0, v_i1, v_i2])
            for i0, i1, i2 in T.grid(T.int64(128), T.int64(16), T.int64(8)):
                with Ts.sblock("trilu_1"):
                    v_i0, v_i1, v_i2 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(compute[v_i0, v_i1, v_i2])
                    Ts.writes(trilu_1[v_i0, v_i1, v_i2])
                    trilu_1[v_i0, v_i1, v_i2] = T.Select(v_i2 <= v_i1, compute[v_i0, v_i1, v_i2], T.float32(0.0))
            for ax0, ax1, ax2, k2 in T.grid(T.int64(128), T.int64(16), T.int64(1), T.int64(8)):
                with Ts.sblock("trilu_red_1"):
                    v_ax0, v_ax1, v_ax2, v_k2 = Ts.axis.remap("SSSR", [ax0, ax1, ax2, k2])
                    Ts.reads(trilu_1[v_ax0, v_ax1, v_k2])
                    Ts.writes(trilu_red_1[v_ax0, v_ax1, v_ax2])
                    with Ts.init():
                        trilu_red_1[v_ax0, v_ax1, v_ax2] = T.float32(0.0)
                    trilu_red_1[v_ax0, v_ax1, v_ax2] = trilu_red_1[v_ax0, v_ax1, v_ax2] + trilu_1[v_ax0, v_ax1, v_k2]
            for ax0, ax1, ax2 in T.grid(T.int64(128), T.int64(16), T.int64(8)):
                with Ts.sblock("T_divide"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(trilu_1[v_ax0, v_ax1, v_ax2], trilu_red_1[v_ax0, v_ax1, T.int64(0)])
                    Ts.writes(T_divide[v_ax0, v_ax1, v_ax2])
                    T_divide[v_ax0, v_ax1, v_ax2] = trilu_1[v_ax0, v_ax1, v_ax2] / trilu_red_1[v_ax0, v_ax1, T.int64(0)]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(4), T.int64(32), T.int64(8), T.int64(16)):
                with Ts.sblock("T_transpose_2"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(v[v_ax0, v_ax2, v_ax1, v_ax3])
                    Ts.writes(T_transpose_3[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_transpose_3[v_ax0, v_ax1, v_ax2, v_ax3] = v[v_ax0, v_ax2, v_ax1, v_ax3]
            for ax0, ax1, ax2 in T.grid(T.int64(128), T.int64(8), T.int64(16)):
                with Ts.sblock("T_reshape_4"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(T_transpose_3[((v_ax2 // T.int64(16) + v_ax1) // T.int64(8) + v_ax0) % T.int64(128) // T.int64(32), ((v_ax2 // T.int64(16) + v_ax1) // T.int64(8) + v_ax0) % T.int64(32), (v_ax2 // T.int64(16) + v_ax1) % T.int64(8), v_ax2 % T.int64(16)])
                    Ts.writes(T_reshape_4[v_ax0, v_ax1, v_ax2])
                    T_reshape_4[v_ax0, v_ax1, v_ax2] = T_transpose_3[((v_ax2 // T.int64(16) + v_ax1) // T.int64(8) + v_ax0) % T.int64(128) // T.int64(32), ((v_ax2 // T.int64(16) + v_ax1) // T.int64(8) + v_ax0) % T.int64(32), (v_ax2 // T.int64(16) + v_ax1) % T.int64(8), v_ax2 % T.int64(16)]
            for b_index, i, j, k_1 in T.grid(T.int64(128), T.int64(16), T.int64(16), T.int64(8)):
                with Ts.sblock("T_batch_matmul_NN"):
                    v_b, v_i, v_j, v_k = Ts.axis.remap("SSSR", [b_index, i, j, k_1])
                    Ts.reads(T_divide[v_b, v_i, v_k], T_reshape_4[v_b, v_k, v_j])
                    Ts.writes(T_batch_matmul_NN[v_b, v_i, v_j])
                    Ts.sblock_attr({"layout_free_placeholders": [T_reshape_4]})
                    with Ts.init():
                        T_batch_matmul_NN[v_b, v_i, v_j] = T.float32(0.0)
                    T_batch_matmul_NN[v_b, v_i, v_j] = T_batch_matmul_NN[v_b, v_i, v_j] + T_divide[v_b, v_i, v_k] * T_reshape_4[v_b, v_k, v_j]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(4), T.int64(32), T.int64(16), T.int64(16)):
                with Ts.sblock("T_reshape_5"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(T_batch_matmul_NN[(v_ax0 * T.int64(32) + (v_ax3 // T.int64(16) + v_ax2) // T.int64(16) + v_ax1) % T.int64(128), (v_ax3 // T.int64(16) + v_ax2) % T.int64(16), v_ax3 % T.int64(16)])
                    Ts.writes(T_reshape_5[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_5[v_ax0, v_ax1, v_ax2, v_ax3] = T_batch_matmul_NN[(v_ax0 * T.int64(32) + (v_ax3 // T.int64(16) + v_ax2) // T.int64(16) + v_ax1) % T.int64(128), (v_ax3 // T.int64(16) + v_ax2) % T.int64(16), v_ax3 % T.int64(16)]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(4), T.int64(16), T.int64(32), T.int64(16)):
                with Ts.sblock("T_transpose_3"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(T_reshape_5[v_ax0, v_ax2, v_ax1, v_ax3])
                    Ts.writes(T_transpose[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_transpose[v_ax0, v_ax1, v_ax2, v_ax3] = T_reshape_5[v_ax0, v_ax2, v_ax1, v_ax3]

        @R.function
        def main(q: R.Tensor((4, 16, 32, 8), dtype="float32"), k: R.Tensor((4, 8, 32, 8), dtype="float32"), v: R.Tensor((4, 8, 32, 16), dtype="float32"), bias: R.Tensor((4, 32, 16, 8), dtype="float32")) -> R.Tensor((4, 16, 32, 16), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.attention_bias, (q, k, v, bias), ty_args=[R.Tensor((4, 16, 32, 16), dtype="float32")])
            return gv
    # fmt: on
    mod = LegalizeOps()(Attention)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_dynamic_attention():
    """The sequence lengths may be dynamic

    In previous implementations, the `seq_len` and `seq_len_kv` were
    assumed to be static integers, and produced an exception during
    legalization.
    """

    seq_len = T.dynamic("seq_len", "int64")
    seq_len_kv = T.dynamic("seq_len_kv", "int64")

    @tvm_tirx.script.ir_module
    class Attention:
        @R.function
        def main(
            q: R.Tensor((4, seq_len, 32, 8), "float32"),
            k: R.Tensor((4, seq_len_kv, 32, 8), "float32"),
            v: R.Tensor((4, seq_len_kv, 32, 16), "float32"),
            bias: R.Tensor((4, 32, seq_len, seq_len_kv), "float32"),
        ):
            gv = R.nn.attention(
                q, k, v, bias, scale=T.FloatImm("float32", 0.1), causal_mask="BottomRight"
            )
            return gv

    LegalizeOps()(Attention)


def test_dynamic_batch_attention():
    """The batch dimension may be dynamic (symbolic).

    fix https://github.com/apache/tvm/issues/19696
    """

    batch_size = T.dynamic("batch_size", "int64")

    @tvm_tirx.script.ir_module
    class Attention:
        @R.function
        def main(
            q: R.Tensor((batch_size, 16, 32, 8), "float32"),
            k: R.Tensor((batch_size, 8, 32, 8), "float32"),
            v: R.Tensor((batch_size, 8, 32, 16), "float32"),
        ):
            gv = R.nn.attention(q, k, v)
            return gv

    LegalizeOps()(Attention)

    batch_size = T.dynamic("batch_size", "int64")

    @tvm_tirx.script.ir_module
    class AttentionBias:
        @R.function
        def main(
            q: R.Tensor((batch_size, 16, 32, 8), "float32"),
            k: R.Tensor((batch_size, 8, 32, 8), "float32"),
            v: R.Tensor((batch_size, 8, 32, 16), "float32"),
            bias: R.Tensor((batch_size, 32, 16, 8), "float32"),
        ):
            gv = R.nn.attention(
                q, k, v, bias, scale=T.FloatImm("float32", 0.1), causal_mask="BottomRight"
            )
            return gv

    LegalizeOps()(AttentionBias)


def test_nll_loss():
    # fmt: off
    @tvm_tirx.script.ir_module
    class NLLLoss:
        @R.function
        def main(
                predictions: R.Tensor((2, 3, 4, 5), "float32"),
                targets: R.Tensor((2, 4, 5), "int64"),
                weights: R.Tensor((3,), "float32"),
        ) -> R.Tensor((), "float32"):
            gv = R.nn.nll_loss(predictions, targets, weights, reduction="mean", ignore_index=-1)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
                predictions: R.Tensor((2, 3, 4, 5), dtype="float32"),
                targets: R.Tensor((2, 4, 5), dtype="int64"),
                weights: R.Tensor((3,), dtype="float32"),
        ) -> R.Tensor((), dtype="float32"):
            gv = R.call_tir(Expected.nll_loss, (predictions, targets, weights), ty_args=[R.Tensor((), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def nll_loss(
                predictions: T.Tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"),
                targets: T.Tensor((T.int64(2), T.int64(4), T.int64(5)), "int64"),
                weights: T.Tensor(T.int64(3), "float32"),
                output: T.Tensor((), "float32"),
        ):
            # function attr dict
            T.func_attr({"tirx.noalias": True})
            # body
            # with Ts.sblock("root")
            nll_loss = Ts.sblock_alloc_tensor([T.int64(2), T.int64(4), T.int64(5)], dtype="float32")
            nll_loss_red = Ts.sblock_alloc_tensor([], dtype="float32")
            nll_loss_1 = Ts.sblock_alloc_tensor([T.int64(2), T.int64(4), T.int64(5)], dtype="float32")
            nll_loss_red_1 = Ts.sblock_alloc_tensor([], dtype="float32")
            for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4), T.int64(5)):
                with Ts.sblock("nll_loss"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(targets[v_ax0, v_ax1, v_ax2], predictions[v_ax0, targets[v_ax0, v_ax1, v_ax2], v_ax1, v_ax2], weights[targets[v_ax0, v_ax1, v_ax2]])
                    Ts.writes(nll_loss[v_ax0, v_ax1, v_ax2])
                    nll_loss[v_ax0, v_ax1, v_ax2] = T.Select(targets[v_ax0, v_ax1, v_ax2] != T.int64(-1), (T.float32(0) - predictions[v_ax0, targets[v_ax0, v_ax1, v_ax2], v_ax1, v_ax2]) * weights[targets[v_ax0, v_ax1, v_ax2]], T.float32(0))
            for k0, k1, k2 in T.grid(T.int64(2), T.int64(4), T.int64(5)):
                with Ts.sblock("nll_loss_red"):
                    v_k0, v_k1, v_k2 = Ts.axis.remap("RRR", [k0, k1, k2])
                    Ts.reads(nll_loss[v_k0, v_k1, v_k2])
                    Ts.writes(nll_loss_red[()])
                    with Ts.init():
                        nll_loss_red[()] = T.float32(0)
                    nll_loss_red[()] = nll_loss_red[()] + nll_loss[v_k0, v_k1, v_k2]
            for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4), T.int64(5)):
                with Ts.sblock("nll_loss_1"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(targets[v_ax0, v_ax1, v_ax2], weights[targets[v_ax0, v_ax1, v_ax2]])
                    Ts.writes(nll_loss_1[v_ax0, v_ax1, v_ax2])
                    nll_loss_1[v_ax0, v_ax1, v_ax2] = T.Select(targets[v_ax0, v_ax1, v_ax2] != T.int64(-1), weights[targets[v_ax0, v_ax1, v_ax2]], T.float32(0))
            for k0, k1, k2 in T.grid(T.int64(2), T.int64(4), T.int64(5)):
                with Ts.sblock("nll_loss_red_1"):
                    v_k0, v_k1, v_k2 = Ts.axis.remap("RRR", [k0, k1, k2])
                    Ts.reads(nll_loss_1[v_k0, v_k1, v_k2])
                    Ts.writes(nll_loss_red_1[()])
                    with Ts.init():
                        nll_loss_red_1[()] = T.float32(0)
                    nll_loss_red_1[()] = nll_loss_red_1[()] + nll_loss_1[v_k0, v_k1, v_k2]
            with Ts.sblock("T_divide"):
                vi = Ts.axis.spatial(T.int64(1), T.int64(0))
                Ts.reads(nll_loss_red[()], nll_loss_red_1[()])
                Ts.writes(output[()])
                output[()] = nll_loss_red[()] / nll_loss_red_1[()]
    # fmt: on
    mod = LegalizeOps()(NLLLoss)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_nll_no_weight():
    # fmt: off
    @tvm_tirx.script.ir_module
    class NLLLoss:
        @R.function
        def main(predictions: R.Tensor((2, 3, 4, 5), "float32"), targets: R.Tensor((2, 4, 5), "int64")) -> R.Tensor((), "float32"):
            gv: R.Tensor((), "float32") = R.nn.nll_loss(predictions, targets, reduction="mean", ignore_index=-1)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(predictions: R.Tensor((2, 3, 4, 5), dtype="float32"), targets: R.Tensor((2, 4, 5), dtype="int64"),) -> R.Tensor((), dtype="float32"):
            # block 0
            gv = R.call_tir(Expected.nll_loss_without_weight, (predictions, targets), ty_args=[R.Tensor((), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def nll_loss_without_weight(rxplaceholder: T.Tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"), rxplaceholder_1: T.Tensor((T.int64(2), T.int64(4), T.int64(5)), "int64"), T_divide: T.Tensor((), "float32"),):
            # function attr dict
            T.func_attr({"tirx.noalias": True})
            # body
            # with Ts.sblock("root")
            T_full = Ts.sblock_alloc_tensor([T.int64(3)], dtype="float32")
            nll_loss = Ts.sblock_alloc_tensor([T.int64(2), T.int64(4), T.int64(5)], dtype="float32")
            nll_loss_red = Ts.sblock_alloc_tensor([], dtype="float32")
            nll_loss_1 = Ts.sblock_alloc_tensor([T.int64(2), T.int64(4), T.int64(5)], dtype="float32")
            nll_loss_red_1 = Ts.sblock_alloc_tensor([], dtype="float32")
            for ax0 in T.serial(T.int64(3)):
                with Ts.sblock("T_full"):
                    v_ax0 = Ts.axis.spatial(T.int64(3), ax0)
                    Ts.reads()
                    Ts.writes(T_full[v_ax0])
                    T_full[v_ax0] = T.float32(1)
            for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4), T.int64(5)):
                with Ts.sblock("nll_loss"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(rxplaceholder_1[v_ax0, v_ax1, v_ax2], rxplaceholder[v_ax0, rxplaceholder_1[v_ax0, v_ax1, v_ax2], v_ax1, v_ax2], T_full[rxplaceholder_1[v_ax0, v_ax1, v_ax2]])
                    Ts.writes(nll_loss[v_ax0, v_ax1, v_ax2])
                    nll_loss[v_ax0, v_ax1, v_ax2] = T.Select(rxplaceholder_1[v_ax0, v_ax1, v_ax2] != T.int64(-1), (T.float32(0) - rxplaceholder[v_ax0, rxplaceholder_1[v_ax0, v_ax1, v_ax2], v_ax1, v_ax2]) * T_full[rxplaceholder_1[v_ax0, v_ax1, v_ax2]], T.float32(0))
            for k0, k1, k2 in T.grid(T.int64(2), T.int64(4), T.int64(5)):
                with Ts.sblock("nll_loss_red"):
                    v_k0, v_k1, v_k2 = Ts.axis.remap("RRR", [k0, k1, k2])
                    Ts.reads(nll_loss[v_k0, v_k1, v_k2])
                    Ts.writes(nll_loss_red[()])
                    with Ts.init():
                        nll_loss_red[()] = T.float32(0)
                    nll_loss_red[()] = nll_loss_red[()] + nll_loss[v_k0, v_k1, v_k2]
            for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4), T.int64(5)):
                with Ts.sblock("nll_loss_1"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(rxplaceholder_1[v_ax0, v_ax1, v_ax2], T_full[rxplaceholder_1[v_ax0, v_ax1, v_ax2]])
                    Ts.writes(nll_loss_1[v_ax0, v_ax1, v_ax2])
                    nll_loss_1[v_ax0, v_ax1, v_ax2] = T.Select(rxplaceholder_1[v_ax0, v_ax1, v_ax2] != T.int64(-1), T_full[rxplaceholder_1[v_ax0, v_ax1, v_ax2]], T.float32(0))
            for k0, k1, k2 in T.grid(T.int64(2), T.int64(4), T.int64(5)):
                with Ts.sblock("nll_loss_red_1"):
                    v_k0, v_k1, v_k2 = Ts.axis.remap("RRR", [k0, k1, k2])
                    Ts.reads(nll_loss_1[v_k0, v_k1, v_k2])
                    Ts.writes(nll_loss_red_1[()])
                    with Ts.init():
                        nll_loss_red_1[()] = T.float32(0)
                    nll_loss_red_1[()] = nll_loss_red_1[()] + nll_loss_1[v_k0, v_k1, v_k2]
            with Ts.sblock("T_divide"):
                vi = Ts.axis.spatial(T.int64(1), T.int64(0))
                Ts.reads(nll_loss_red[()], nll_loss_red_1[()])
                Ts.writes(T_divide[()])
                T_divide[()] = nll_loss_red[()] / nll_loss_red_1[()]
    # fmt: on

    mod = LegalizeOps()(NLLLoss)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_nll_no_batch():
    # fmt: off
    C = T.dynamic("C", "int64")

    @tvm_tirx.script.ir_module
    class NLLLoss:
        @R.function
        def main(predictions: R.Tensor((C,), "float32"), targets: R.Tensor((), "int64"), weights: R.Tensor((C,), "float32")) -> R.Tensor((), "float32"):
            gv = R.nn.nll_loss(predictions, targets, weights, reduction="mean", ignore_index=1)
            return gv

    C_main = T.dynamic("C", "int64")
    C_nll_loss = T.dynamic("C", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(predictions: R.Tensor((C_main,), dtype="float32"), targets: R.Tensor((), dtype="int64"), weights: R.Tensor((C_main,), dtype="float32")) -> R.Tensor((), dtype="float32"):
            gv = R.call_tir(Expected.nll_loss, (predictions, targets, weights), ty_args=[R.Tensor((), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def nll_loss(rxplaceholder_1: T.Tensor((C_nll_loss,), "float32"), rxplaceholder: T.Tensor((), "int64"), rxplaceholder_2: T.Tensor((C_nll_loss,), "float32"), T_divide: T.Tensor((), "float32")):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            nll_loss = Ts.sblock_alloc_tensor(())
            nll_loss_1 = Ts.sblock_alloc_tensor(())
            with Ts.sblock("nll_loss"):
                vi = Ts.axis.spatial(T.int64(1), T.int64(0))
                Ts.reads(rxplaceholder[()], rxplaceholder_1[rxplaceholder[()]], rxplaceholder_2[rxplaceholder[()]])
                Ts.writes(nll_loss[()])
                nll_loss[()] = T.Select(rxplaceholder[()] != T.int64(1), (T.float32(0) - rxplaceholder_1[rxplaceholder[()]]) * rxplaceholder_2[rxplaceholder[()]], T.float32(0))
            with Ts.sblock("nll_loss_1"):
                vi = Ts.axis.spatial(T.int64(1), T.int64(0))
                Ts.reads(rxplaceholder[()], rxplaceholder_2[rxplaceholder[()]])
                Ts.writes(nll_loss_1[()])
                nll_loss_1[()] = T.Select(rxplaceholder[()] != T.int64(1), rxplaceholder_2[rxplaceholder[()]], T.float32(0))
            with Ts.sblock("T_divide"):
                vi = Ts.axis.spatial(T.int64(1), T.int64(0))
                Ts.reads(nll_loss[()], nll_loss_1[()])
                Ts.writes(T_divide[()])
                T_divide[()] = nll_loss[()] / nll_loss_1[()]
    # fmt: on

    mod = LegalizeOps()(NLLLoss)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_nll_loss_symbolic():
    # fmt: off
    N = T.dynamic("N", "int64")
    C = T.dynamic("C", "int64")
    d1 = T.dynamic("d1", "int64")
    d2 = T.dynamic("d2", "int64")

    @tvm_tirx.script.ir_module
    class NLLLoss:
        @R.function
        def main(predictions: R.Tensor((N, C, d1, d2), "float32"), targets: R.Tensor((N, d1, d2), "int64"), weights: R.Tensor((C,), "float32")) -> R.Tensor((), "float32"):
            gv: R.Tensor((), "float32") = R.nn.nll_loss(predictions, targets, weights, reduction="mean", ignore_index=-1)
            return gv

    N_main = T.dynamic("N", "int64")
    C_main = T.dynamic("C", "int64")
    d1_main = T.dynamic("d1", "int64")
    d2_main = T.dynamic("d2", "int64")
    C_nll_loss = T.dynamic("C", "int64")
    N_nll_loss = T.dynamic("N", "int64")
    d1_nll_loss = T.dynamic("d1", "int64")
    d2_nll_loss = T.dynamic("d2", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(predictions: R.Tensor((N_main, C_main, d1_main, d2_main), dtype="float32"), targets: R.Tensor((N_main, d1_main, d2_main), dtype="int64"), weights: R.Tensor((C_main,), dtype="float32")) -> R.Tensor((), dtype="float32"):
            # block 0
            gv = R.call_tir(Expected.nll_loss, (predictions, targets, weights), ty_args=[R.Tensor((), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def nll_loss(rxplaceholder: T.Tensor([N_nll_loss, C_nll_loss, d1_nll_loss, d2_nll_loss], dtype='float32'), rxplaceholder_1: T.Tensor([N_nll_loss, d1_nll_loss, d2_nll_loss], dtype='int64'), rxplaceholder_2: T.Tensor([C_nll_loss], dtype='float32'), T_divide: T.Tensor((), "float32"),):
            # function attr dict
            T.func_attr({"tirx.noalias": True})

            # body
            # with Ts.sblock("root")
            nll_loss = Ts.sblock_alloc_tensor([N_nll_loss, d1_nll_loss, d2_nll_loss], dtype="float32")
            nll_loss_red = Ts.sblock_alloc_tensor([], dtype="float32")
            nll_loss_1 = Ts.sblock_alloc_tensor([N_nll_loss, d1_nll_loss, d2_nll_loss], dtype="float32")
            nll_loss_red_1 = Ts.sblock_alloc_tensor([], dtype="float32")
            for ax0, ax1, ax2 in T.grid(N_nll_loss, d1_nll_loss, d2_nll_loss):
                with Ts.sblock("nll_loss"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(rxplaceholder_1[v_ax0, v_ax1, v_ax2], rxplaceholder[v_ax0, rxplaceholder_1[v_ax0, v_ax1, v_ax2], v_ax1, v_ax2],rxplaceholder_2[rxplaceholder_1[v_ax0, v_ax1, v_ax2]],)
                    Ts.writes(nll_loss[v_ax0, v_ax1, v_ax2])
                    nll_loss[v_ax0, v_ax1, v_ax2] = T.Select(rxplaceholder_1[v_ax0, v_ax1, v_ax2] != T.int64(-1), (T.float32(0) - rxplaceholder[v_ax0, rxplaceholder_1[v_ax0, v_ax1, v_ax2], v_ax1, v_ax2]) * rxplaceholder_2[rxplaceholder_1[v_ax0, v_ax1, v_ax2]], T.float32(0),)
            for k0, k1, k2 in T.grid(N_nll_loss, d1_nll_loss, d2_nll_loss):
                with Ts.sblock("nll_loss_red"):
                    v_k0, v_k1, v_k2 = Ts.axis.remap("RRR", [k0, k1, k2])
                    Ts.reads(nll_loss[v_k0, v_k1, v_k2])
                    Ts.writes(nll_loss_red[()])
                    with Ts.init():
                        nll_loss_red[()] = T.float32(0)
                    nll_loss_red[()] = nll_loss_red[()] + nll_loss[v_k0, v_k1, v_k2]
            for ax0, ax1, ax2 in T.grid(N_nll_loss, d1_nll_loss, d2_nll_loss):
                with Ts.sblock("nll_loss_1"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(rxplaceholder_1[v_ax0, v_ax1, v_ax2], rxplaceholder_2[rxplaceholder_1[v_ax0, v_ax1, v_ax2]],)
                    Ts.writes(nll_loss_1[v_ax0, v_ax1, v_ax2])
                    nll_loss_1[v_ax0, v_ax1, v_ax2] = T.Select(rxplaceholder_1[v_ax0, v_ax1, v_ax2] != T.int64(-1), rxplaceholder_2[rxplaceholder_1[v_ax0, v_ax1, v_ax2]], T.float32(0),)
            for k0, k1, k2 in T.grid(N_nll_loss, d1_nll_loss, d2_nll_loss):
                with Ts.sblock("nll_loss_red_1"):
                    v_k0, v_k1, v_k2 = Ts.axis.remap("RRR", [k0, k1, k2])
                    Ts.reads(nll_loss_1[v_k0, v_k1, v_k2])
                    Ts.writes(nll_loss_red_1[()])
                    with Ts.init():
                        nll_loss_red_1[()] = T.float32(0)
                    nll_loss_red_1[()] = nll_loss_red_1[()] + nll_loss_1[v_k0, v_k1, v_k2]
            with Ts.sblock("T_divide"):
                vi = Ts.axis.spatial(T.int64(1), T.int64(0))
                Ts.reads(nll_loss_red[()], nll_loss_red_1[()])
                Ts.writes(T_divide[()])
                T_divide[()] = nll_loss_red[()] / nll_loss_red_1[()]
    # fmt: on
    mod = LegalizeOps()(NLLLoss)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_pad():
    @tvm_tirx.script.ir_module
    class Pad:
        @R.function
        def main(x: R.Tensor((2, 128, 28), "float32")) -> R.Tensor((2, 130, 30), "float32"):
            gv: R.Tensor((2, 130, 30), "float32") = R.nn.pad(x, (0, 0, 1, 1, 1, 1))
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 128, 28), dtype="float32"),
        ) -> R.Tensor((2, 130, 30), dtype="float32"):
            gv = R.call_tir(Expected.pad, (x), ty_args=[R.Tensor((2, 130, 30), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def pad(
            A: T.Tensor((T.int64(2), T.int64(128), T.int64(28)), "float32"),
            PadInput: T.Tensor((T.int64(2), T.int64(130), T.int64(30)), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(130), T.int64(30)):
                with Ts.sblock("PadInput"):
                    v_i0, v_i1, v_i2 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(A[v_i0, v_i1 - T.int64(1), v_i2 - T.int64(1)])
                    Ts.writes(PadInput[v_i0, v_i1, v_i2])
                    PadInput[v_i0, v_i1, v_i2] = T.if_then_else(
                        T.int64(1) <= v_i1
                        and v_i1 < T.int64(129)
                        and T.int64(1) <= v_i2
                        and v_i2 < T.int64(29),
                        A[v_i0, v_i1 - T.int64(1), v_i2 - T.int64(1)],
                        T.float32(0),
                    )

    mod = LegalizeOps()(Pad)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_batch_flatten():
    # fmt: off
    @tvm_tirx.script.ir_module
    class BatchFlatten:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32")) -> R.Tensor((2, 60), "float32"):
            gv: R.Tensor((2, 60), "float32") = R.nn.batch_flatten(x)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), dtype="float32")) -> R.Tensor((2, 60), dtype="float32"):
            gv = R.call_tir(Expected.reshape, (x,), ty_args=[R.Tensor((2, 60), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def reshape(x: T.Tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"), T_reshape: T.Tensor((T.int64(2), T.int64(60)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for ax0, ax1 in T.grid(T.int64(2), T.int64(60)):
                with Ts.sblock("T_reshape"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(x[(v_ax1 // T.int64(60) + v_ax0) % T.int64(2), v_ax1 % T.int64(60) // T.int64(20), v_ax1 % T.int64(20) // T.int64(5), v_ax1 % T.int64(5)])
                    Ts.writes(T_reshape[v_ax0, v_ax1])
                    T_reshape[v_ax0, v_ax1] = x[(v_ax1 // T.int64(60) + v_ax0) % T.int64(2), v_ax1 % T.int64(60) // T.int64(20), v_ax1 % T.int64(20) // T.int64(5), v_ax1 % T.int64(5)]
    # fmt: on

    mod = LegalizeOps()(BatchFlatten)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_batch_flatten_undefined_shape():
    @tvm_tirx.script.ir_module
    class BatchFlattenUndefinedShape:
        @R.function
        def main(x: R.Tensor(ndim=4, dtype="float32")) -> R.Tensor(ndim=2, dtype="float32"):
            gv: R.Tensor(ndim=2, dtype="float32") = R.nn.batch_flatten(x)
            return gv

    mod = LegalizeOps()(BatchFlattenUndefinedShape)
    tvm_tirx.ir.assert_structural_equal(mod, BatchFlattenUndefinedShape)


def test_dropout():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Dropout:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tuple(R.Tensor((2, 3), "float32"), R.Tensor((2, 3), "float32")):
            gv: R.Tuple(R.Tensor((2, 3), "float32"), R.Tensor((2, 3), "float32")) = R.nn.dropout(x, rate=0.5)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function(private=True)
        def dropout(x: T.Tensor((T.int64(2), T.int64(3)), "float32"), compute: T.Tensor((T.int64(2), T.int64(3)), "float32"), T_full_like: T.Tensor((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("compute"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(x[v_i0, v_i1])
                    Ts.writes(compute[v_i0, v_i1])
                    compute[v_i0, v_i1] = x[v_i0, v_i1]
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_full_like"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads()
                    Ts.writes(T_full_like[v_ax0, v_ax1])
                    T_full_like[v_ax0, v_ax1] = T.float32(1.0)

        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tuple(R.Tensor((2, 3), dtype="float32"), R.Tensor((2, 3), dtype="float32")):
            cls = Expected
            gv = R.call_tir(cls.dropout, (x,), ty_args=[_TupleType([R.Tensor((2, 3), dtype="float32"), R.Tensor((2, 3), dtype="float32")])])
            return gv
    # fmt: on

    mod = LegalizeOps()(Dropout)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
