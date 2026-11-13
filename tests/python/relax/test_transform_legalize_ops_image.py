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
# ruff: noqa: E501

import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm.relax.transform import LegalizeOps
from tvm.s_tir import script as Ts
from tvm_tirx.script import tirx as T


def test_image_resize2d():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Resize2D:
        @R.function
        def main(x: R.Tensor((2, 8, 8, 3), "float32")) -> R.Tensor((2, 16, 16, 3), "float32"):
            gv: R.Tensor((2, 16, 16, 3), "float32") = R.image.resize2d(x, size=(16, 16), layout="NHWC", method="nearest_neighbor", coordinate_transformation_mode="asymmetric")
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 8, 8, 3), "float32")) -> R.Tensor((2, 16, 16, 3), "float32"):
            gv = R.call_tir(Expected.resize2d, (x,), ty_args=[R.Tensor((2, 16, 16, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def resize2d(rxplaceholder: T.Tensor((T.int64(2), T.int64(8), T.int64(8), T.int64(3)), "float32"), resize: T.Tensor((T.int64(2), T.int64(16), T.int64(16), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(16), T.int64(16), T.int64(3)):
                with Ts.sblock("resize"):
                    i0_1, i1_1, i2_1, i3_1 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[i0_1, T.int64(0):T.int64(8), T.int64(0):T.int64(8), i3_1])
                    Ts.writes(resize[i0_1, i1_1, i2_1, i3_1])
                    resize[i0_1, i1_1, i2_1, i3_1] = rxplaceholder[i0_1, T.max(T.min(T.Cast("int64", T.round(T.float32(0.5) * T.Cast("float32", i1_1))), T.int64(7)), T.int64(0)), T.max(T.min(T.Cast("int64", T.round(T.float32(0.5) * T.Cast("float32", i2_1))), T.int64(7)), T.int64(0)), i3_1]
    # fmt: on

    mod = LegalizeOps()(Resize2D)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_image_resize2d_symbolic():
    # fmt: off
    n = T.dynamic("n", "int64")
    c = T.dynamic("c", "int64")
    oh = T.dynamic("oh", "int64")
    ow = T.dynamic("ow", "int64")
    h = T.dynamic("h", "int64")
    w = T.dynamic("w", "int64")

    @tvm_tirx.script.ir_module
    class Resize2D:
        @R.function
        def main(dumb_param: R.Tensor((oh, ow)), x: R.Tensor((n, c, h, w, 16), "float32")) -> R.Tensor((n, c, oh, ow, 16), "float32"):
            gv: R.Tensor((n, c, oh, ow, 16), "float32") = R.image.resize2d(x, size=(oh, ow), layout="NCHW16c", method="nearest_neighbor", coordinate_transformation_mode="asymmetric")
            return gv

    n_main = T.dynamic("n", "int64")
    c_main = T.dynamic("c", "int64")
    oh_main = T.dynamic("oh", "int64")
    ow_main = T.dynamic("ow", "int64")
    h_main = T.dynamic("h", "int64")
    w_main = T.dynamic("w", "int64")
    c_resize2d = T.dynamic("c", "int64")
    h_resize2d = T.dynamic("h", "int64")
    n_resize2d = T.dynamic("n", "int64")
    oh_resize2d = T.dynamic("oh", "int64")
    ow_resize2d = T.dynamic("ow", "int64")
    w_resize2d = T.dynamic("w", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(dumb_param: R.Tensor((oh_main, ow_main)), x: R.Tensor((n_main, c_main, h_main, w_main, 16), "float32")) -> R.Tensor((n_main, c_main, oh_main, ow_main, 16), "float32"):
            gv = R.call_tir(Expected.resize2d, (x,), ty_args=[R.Tensor((n_main, c_main, oh_main, ow_main, 16), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def resize2d(rxplaceholder: T.Tensor([n_resize2d, c_resize2d, h_resize2d, w_resize2d, T.int64(16)], dtype='float32'), resize: T.Tensor([n_resize2d, c_resize2d, oh_resize2d, ow_resize2d, T.int64(16)], dtype='float32')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1, i2, i3, i4 in T.grid(n_resize2d, c_resize2d, oh_resize2d, ow_resize2d, T.int64(16)):
                with Ts.sblock("resize"):
                    i0_1, i1_1, i2_1, i3_1, i4_1 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                    Ts.reads(rxplaceholder[i0_1, i1_1, T.int64(0) : T.max(h_resize2d, T.int64(1)), T.int64(0) : T.max(w_resize2d, T.int64(1)), i4_1])
                    Ts.writes(resize[i0_1, i1_1, i2_1, i3_1, i4_1])
                    resize[i0_1, i1_1, i2_1, i3_1, i4_1] = rxplaceholder[i0_1, i1_1, T.max(T.min(T.Cast("int64", T.round(T.Cast("float32", h_resize2d) / T.Cast("float32", oh_resize2d) * T.Cast("float32", i2_1))), h_resize2d - T.int64(1)), T.int64(0)), T.max(T.min(T.Cast("int64", T.round(T.Cast("float32", w_resize2d) / T.Cast("float32", ow_resize2d) * T.Cast("float32", i3_1))), w_resize2d - T.int64(1)), T.int64(0)), i4_1]
    # fmt: on

    mod = LegalizeOps()(Resize2D)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_image_affine_grid():
    # fmt: off
    @tvm_tirx.script.ir_module
    class AffineGrid:
        @R.function
        def main(theta: R.Tensor((2, 2, 3), "float32")) -> R.Tensor((2, 2, 16, 16), "float32"):
            gv: R.Tensor((2, 2, 16, 16), "float32") = R.image.affine_grid(theta, size=(16, 16))
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(theta: R.Tensor((2, 2, 3), "float32")) -> R.Tensor((2, 2, 16, 16), "float32"):
            gv = R.call_tir(Expected.affine_grid, (theta,), ty_args=[R.Tensor((2, 2, 16, 16), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def affine_grid(theta: T.Tensor((T.int64(2), T.int64(2), T.int64(3)), "float32"), compute: T.Tensor((T.int64(2), T.int64(2), T.int64(16), T.int64(16)), "float32")):
            T.func_attr({"tirx.noalias": True})

            with Ts.sblock("root"):
                Ts.reads()
                Ts.writes()
                for n_index, dim, i0, i1 in T.grid(T.int64(2), T.int64(2), T.int64(16), T.int64(16)):
                    with Ts.sblock("compute"):
                        v_n, v_dim, v_i0, v_i1 = Ts.axis.remap("SSSS", [n_index, dim, i0, i1])
                        Ts.reads(theta[v_n, v_dim, T.int64(0):T.int64(3)])
                        Ts.writes(compute[v_n, v_dim, v_i0, v_i1])
                        compute[v_n, v_dim, v_i0, v_i1] = theta[v_n, v_dim, T.int64(2)] + theta[v_n, v_dim, T.int64(1)] * (T.float32(-1.0) + T.Cast("float32", v_i0) * T.float32(0.13333332666666667)) + theta[v_n, v_dim, T.int64(0)] * (T.float32(-1.0) + T.Cast("float32", v_i1) * T.float32(0.13333332666666667))
    # fmt: on

    mod = LegalizeOps()(AffineGrid)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_image_resize3d():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Resize3D:
        @R.function
        def main(x: R.Tensor((2, 3, 8, 8, 8), "float32")) -> R.Tensor((2, 3, 4, 6, 7), "float32"):
            gv: R.Tensor((2, 3, 4, 6, 7), "float32") = R.image.resize3d(
                x,
                size=(4, 6, 7),
                layout="NCDHW",
                method="nearest_neighbor",
                coordinate_transformation_mode="asymmetric",
                rounding_method="floor",
            )
            return gv
    # fmt: on

    mod = LegalizeOps()(Resize3D)

    seen_call_tir = False
    seen_resize3d_relax_op = False

    def _visit(expr):
        nonlocal seen_call_tir, seen_resize3d_relax_op
        if isinstance(expr, relax.Call):
            if isinstance(expr.op, tvm_tirx.ir.Op):
                if expr.op.name == "relax.call_tir":
                    seen_call_tir = True
                if expr.op.name == "relax.image.resize3d":
                    seen_resize3d_relax_op = True

    relax.analysis.post_order_visit(mod["main"].body, _visit)
    assert seen_call_tir
    assert not seen_resize3d_relax_op


if __name__ == "__main__":
    tvm_tirx.testing.main()
