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
# ruff: noqa: E501, F841

from __future__ import annotations

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax, topi
from tvm.relax import ExternFunc as _ExternFunc
from tvm.relax import script as R
from tvm.s_tir import script as Ts
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def _check(mod_actual, mod_expected):
    mod_actual = relax.transform.AnnotateTIROpPattern()(mod_actual)
    mod_actual = relax.transform.FuseOps()(mod_actual)
    mod_expected = relax.transform.AnnotateTIROpPattern()(mod_expected)
    tvm_tirx.ir.assert_structural_equal(mod_actual, mod_expected)


def test_fuse_simple():
    """Simple testcase."""

    def before():
        bb = relax.BlockBuilder()
        x = relax.Var("x", R.Tensor([10, 20], "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.add, x, relax.const(1, "float32"))
                lv1 = bb.emit_te(topi.exp, lv0)
                gv = bb.emit_output(bb.call_te(topi.squeeze, lv1))
            bb.emit_func_output(gv)

        return bb.get()

    def expected():
        bb = relax.BlockBuilder()
        x = relax.Var("x", R.Tensor([10, 20], "float32"))
        p0 = relax.Var("p0", R.Tensor((), "float32"))

        with bb.function("fused_add_exp_squeeze", [x, p0], attrs={"Primitive": True}, private=True):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.add, x, p0)
                lv1 = bb.emit_te(topi.exp, lv0)
                gv = bb.emit_output(bb.call_te(topi.squeeze, lv1))
            bb.emit_func_output(gv)
        fused_add_exp_squeeze = bb.get().get_global_var("fused_add_exp_squeeze")

        x = relax.Var("x", R.Tensor([10, 20], "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                gv = bb.emit_output(
                    relax.Call(fused_add_exp_squeeze, [x, relax.const(1, "float32")])
                )
            bb.emit_func_output(gv)

        return bb.get()

    _check(before(), expected())


def test_conv2d_fuse():
    """Test fusion case of conv2d"""

    def before(dtype):
        bb = relax.BlockBuilder()
        x = relax.Var("x", R.Tensor((1, 16, 64, 64), dtype))
        w1 = relax.Var("w1", R.Tensor((16, 16, 3, 3), dtype))
        w2 = relax.Var("w2", R.Tensor((16, 16, 1, 1), dtype))
        w3 = relax.Var("w3", R.Tensor((16, 16, 3, 3), dtype))
        with bb.function("main", [x, w1, w2, w3]):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.add, x, relax.const(1, dtype))
                lv1 = bb.emit_te(topi.nn.conv2d, lv0, w1, strides=1, padding=1, dilation=1)
                # this is the next dominator.
                lv2 = bb.emit_te(topi.add, relax.const(1, dtype), lv1)
                lv3 = bb.emit_te(topi.add, lv1, lv2)
                # second path
                lv4 = bb.emit_te(topi.nn.conv2d, lv3, w2, strides=1, padding=0, dilation=1)
                lv5 = bb.emit_te(topi.nn.conv2d, lv3, w3, strides=1, padding=1, dilation=1)
                gv = bb.emit_output(bb.call_te(topi.add, lv4, lv5))
            bb.emit_func_output(gv)

        return bb.get()

    def expected(dtype):
        bb = relax.BlockBuilder()

        # Grouped function 1
        x = relax.Var("x", R.Tensor((1, 16, 64, 64), dtype))
        w = relax.Var("w", R.Tensor((16, 16, 3, 3), dtype))
        p0 = relax.Var("p0", R.Tensor((), dtype))
        with bb.function(
            "fused_conv2d_add1_add2", [x, w, p0], attrs={"Primitive": True}, private=True
        ):
            with bb.dataflow():
                lv0 = bb.emit_te(
                    topi.nn.conv2d,
                    x,
                    w,
                    strides=1,
                    padding=1,
                    dilation=1,
                    function_name_hint="conv2d",
                )
                lv1 = bb.emit_te(topi.add, p0, lv0, function_name_hint="add1")
                gv = bb.emit_output(bb.call_te(topi.add, lv0, lv1, function_name_hint="add2"))
            bb.emit_func_output(gv)

        # Grouped function 2
        x = relax.Var("x", R.Tensor((1, 16, 64, 64), dtype))
        w = relax.Var("w", R.Tensor((16, 16, 1, 1), dtype))
        y = relax.Var("y", R.Tensor((1, 16, 64, 64), dtype))
        with bb.function("fused_conv2d1_add2", [x, w, y], attrs={"Primitive": True}, private=True):
            with bb.dataflow():
                lv0 = bb.emit_te(
                    topi.nn.conv2d,
                    x,
                    w,
                    strides=1,
                    padding=0,
                    dilation=1,
                    function_name_hint="conv2d1",
                )
                gv = bb.emit_output(bb.call_te(topi.add, lv0, y, function_name_hint="add2"))
            bb.emit_func_output(gv)

        # Get the global variables of the grouped functions
        mod = bb.get()
        fused_conv2d_add1_add2 = mod.get_global_var("fused_conv2d_add1_add2")
        fused_conv2d1_add2 = mod.get_global_var("fused_conv2d1_add2")

        # Main function
        x = relax.Var("x", R.Tensor((1, 16, 64, 64), dtype))
        w1 = relax.Var("w1", R.Tensor((16, 16, 3, 3), dtype))
        w2 = relax.Var("w2", R.Tensor((16, 16, 1, 1), dtype))
        w3 = relax.Var("w3", R.Tensor((16, 16, 3, 3), dtype))
        with bb.function("main", [x, w1, w2, w3]):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.add, x, relax.const(1, dtype))
                lv1 = bb.emit(relax.Call(fused_conv2d_add1_add2, [lv0, w1, relax.const(1, dtype)]))
                lv2 = bb.emit_te(
                    topi.nn.conv2d,
                    lv1,
                    w3,
                    strides=1,
                    padding=1,
                    dilation=1,
                )
                gv = bb.emit_output(relax.Call(fused_conv2d1_add2, [lv1, w2, lv2]))
            bb.emit_func_output(gv)

        return bb.get()

    _check(before("float32"), expected("float32"))
    _check(before("float16"), expected("float16"))
    _check(before("int8"), expected("int8"))


def test_concatenate():
    """Test fusion case involving concat op and Tuple node"""

    def before():
        bb = relax.BlockBuilder()
        x = relax.Var("x", R.Tensor((1, 16, 64, 64), "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv0 = bb.emit_te(
                    topi.nn.pool2d,
                    x,
                    kernel=(2, 2),
                    stride=(2, 2),
                    dilation=(1, 1),
                    padding=(0, 0, 0, 0),
                    pool_type="max",
                )
                lv1 = bb.emit_te(topi.nn.upsampling, lv0, scale_h=2.0, scale_w=2.0)
                lv2 = bb.emit_te(topi.concatenate, (lv1, x), axis=1)
                gv = bb.emit_output(bb.call_te(topi.add, lv2, relax.const(1, "float32")))
            bb.emit_func_output(gv)

        return bb.get()

    def expected():
        bb = relax.BlockBuilder()

        # Grouped function
        x = relax.Var("x", R.Tensor((1, 16, 64, 64), "float32"))
        w = relax.Var("w", R.Tensor((1, 16, 32, 32), "float32"))
        p0 = relax.Var("p0", R.Tensor((), "float32"))
        with bb.function(
            "fused_upsampling_concatenate_add", [w, x, p0], attrs={"Primitive": True}, private=True
        ):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.nn.upsampling, w, scale_h=2.0, scale_w=2.0)
                lv1 = bb.emit_te(topi.concatenate, (lv0, x), axis=1)
                gv = bb.emit_output(bb.call_te(topi.add, lv1, p0))
            bb.emit_func_output(gv)

        # Get the global variables of the grouped functions
        fused_upsampling_concatenate_add = bb.get().get_global_var(
            "fused_upsampling_concatenate_add"
        )

        # Main function
        x = relax.Var("x", R.Tensor((1, 16, 64, 64), "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv0 = bb.emit_te(
                    topi.nn.pool2d,
                    x,
                    kernel=(2, 2),
                    stride=(2, 2),
                    dilation=(1, 1),
                    padding=(0, 0, 0, 0),
                    pool_type="max",
                )
                gv = bb.emit_output(
                    relax.Call(
                        fused_upsampling_concatenate_add, (lv0, x, relax.const(1, "float32"))
                    )
                )
            bb.emit_func_output(gv)

        return bb.get()

    _check(before(), expected())


def test_tuple_root():
    """Test fusion case where Tuple node is the root in its group"""

    def before():
        bb = relax.BlockBuilder()
        x = relax.Var("x", R.Tensor((1, 16, 64, 64), "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv0 = bb.emit_te(
                    topi.nn.pool2d,
                    x,
                    kernel=(2, 2),
                    stride=(2, 2),
                    dilation=(1, 1),
                    padding=(0, 0, 0, 0),
                    pool_type="max",
                )
                lv1 = bb.emit_te(topi.nn.upsampling, lv0, scale_h=2.0, scale_w=2.0)
                gv = bb.emit_output((lv1, x))
            bb.emit_func_output(gv)

        return bb.get()

    # The fusion is supposed to make no change.
    _check(before(), before())


def test_fuse_tuple_get_elemwise():
    def before(dim: int):
        bb = relax.BlockBuilder()
        x = relax.Var("x", R.Tensor((1, dim), "float32"))
        w = relax.Var("w", R.Tensor((3 * dim, dim), "float32"))
        with bb.function("main", [x, w]):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.nn.dense, x, w)
                lv1 = bb.emit_te(topi.split, lv0, indices_or_sections=3, axis=1)
                lv2 = bb.emit(relax.TupleGetItem(lv1, 0))
                lv3 = bb.emit_te(topi.sigmoid, lv2)
                lv4 = bb.emit(relax.TupleGetItem(lv1, 1))
                lv5 = bb.emit_te(topi.tanh, lv4)
                lv6 = bb.emit(relax.TupleGetItem(lv1, 2))
                lv7 = bb.emit_te(topi.exp, lv6)
                lv8 = bb.emit_te(topi.multiply, lv5, lv7)
                gv = bb.emit_output(bb.call_te(topi.add, lv3, lv8))
            bb.emit_func_output(gv)

        return bb.get()

    def expected(dim: int):
        bb = relax.BlockBuilder()

        # Grouped function
        dense = relax.Var("dense", R.Tensor((1, 3 * dim), "float32"))
        with bb.function(
            "fused_split_sigmoid_tanh_exp_multiply_add",
            [dense],
            attrs={"Primitive": True},
            private=True,
        ):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.split, dense, indices_or_sections=3, axis=1)
                lv1 = bb.emit(relax.TupleGetItem(lv0, 0))
                lv2 = bb.emit_te(topi.sigmoid, lv1)
                lv3 = bb.emit(relax.TupleGetItem(lv0, 1))
                lv4 = bb.emit_te(topi.tanh, lv3)
                lv5 = bb.emit(relax.TupleGetItem(lv0, 2))
                lv6 = bb.emit_te(topi.exp, lv5)
                lv7 = bb.emit_te(topi.multiply, lv4, lv6)
                gv = bb.emit_output(bb.call_te(topi.add, lv2, lv7))
            bb.emit_func_output(gv)

        # Get the global variables of the grouped functions
        fused_split_sigmoid_tanh_exp_multiply_add = bb.get().get_global_var(
            "fused_split_sigmoid_tanh_exp_multiply_add"
        )

        # Main function
        x = relax.Var("x", R.Tensor((1, dim), "float32"))
        w = relax.Var("w", R.Tensor((3 * dim, dim), "float32"))
        with bb.function("main", [x, w]):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.nn.dense, x, w)
                gv = bb.emit_output(relax.Call(fused_split_sigmoid_tanh_exp_multiply_add, (lv0,)))
            bb.emit_func_output(gv)

        return bb.get()

    dim = 10
    _check(before(dim), expected(dim))


def test_tuple_get_root():
    def before(dim: int):
        bb = relax.BlockBuilder()
        x = relax.Var("x", R.Tensor((1, 3 * dim), "float32"))
        w = relax.Var("w", R.Tensor((dim, dim), "float32"))
        with bb.function("main", [x, w]):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.split, x, indices_or_sections=3, axis=1)
                lv1 = bb.emit(relax.TupleGetItem(lv0, 0))
                gv = bb.emit_output(bb.call_te(topi.nn.dense, lv1, w))
            bb.emit_func_output(gv)

        return bb.get()

    def expected(dim: int):
        bb = relax.BlockBuilder()

        # Grouped function
        x = relax.Var("x", R.Tensor((1, 3 * dim), "float32"))
        with bb.function("fused_split", [x], attrs={"Primitive": True}, private=True):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.split, x, indices_or_sections=3, axis=1)
                gv = bb.emit_output(relax.TupleGetItem(lv0, 0))
            bb.emit_func_output(gv)

        # Get the global variables of the grouped functions
        fused_split = bb.get().get_global_var("fused_split")

        # Main function
        x = relax.Var("x", R.Tensor((1, 3 * dim), "float32"))
        w = relax.Var("w", R.Tensor((dim, dim), "float32"))
        with bb.function("main", [x, w]):
            with bb.dataflow():
                lv0 = bb.emit(relax.Call(fused_split, (x,)))
                gv = bb.emit_output(bb.call_te(topi.nn.dense, lv0, w))
            bb.emit_func_output(gv)

        return bb.get()

    dim = 10
    _check(before(dim), expected(dim))


def test_tuple_get_item_chain_without_call():
    """Chained TupleGetItem without any call has nothing to fuse, so it stays in main.

    Fusing it used to create a function without any call_tir, which FuseTIR rejects.
    """

    @I.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            with R.dataflow():
                inner = (x, x)
                outer = (inner, x)
                a = outer[0]
                b = a[0]
                R.output(b)
            return b

    _check(Module, Module)


def test_tuple_intermediate():
    def before():
        bb = relax.BlockBuilder()

        x = relax.Var("x", R.Tensor((1, 16, 64, 64), "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.squeeze, x)
                lv1 = bb.emit_te(topi.add, lv0, relax.const(1, "float32"))
                lv2 = bb.emit_te(topi.squeeze, lv0)
                lv3 = bb.emit_te(topi.add, lv2, relax.const(1, "float32"))
                lv4 = bb.emit_te(topi.add, lv3, relax.const(1, "float32"))
                lv5 = bb.emit_te(topi.add, lv0, relax.const(1, "float32"))
                lv6 = bb.emit_te(topi.concatenate, (lv1, lv4, lv5), axis=1)
                lv7 = bb.emit_te(topi.squeeze, lv6)
                gv = bb.emit_output(bb.call_te(topi.add, lv7, relax.const(1, "float32")))
            bb.emit_func_output(gv)

        return bb.get()

    def expected():
        bb = relax.BlockBuilder()

        # Grouped function
        x = relax.Var("x", R.Tensor((1, 16, 64, 64), "float32"))
        p0 = relax.Var("p0", R.Tensor((), "float32"))
        p1 = relax.Var("p1", R.Tensor((), "float32"))
        p2 = relax.Var("p2", R.Tensor((), "float32"))
        p3 = relax.Var("p3", R.Tensor((), "float32"))
        p4 = relax.Var("p4", R.Tensor((), "float32"))
        with bb.function(
            "fused_squeeze_add_squeeze1_add_add_add_concatenate_squeeze2_add1",
            [x, p0, p1, p2, p3, p4],
            attrs={"Primitive": True},
            private=True,
        ):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.squeeze, x)
                lv1 = bb.emit_te(topi.add, lv0, p0)
                lv2 = bb.emit_te(topi.squeeze, lv0)
                lv3 = bb.emit_te(topi.add, lv2, p1)
                lv4 = bb.emit_te(topi.add, lv3, p2)
                lv5 = bb.emit_te(topi.add, lv0, p3)
                lv6 = bb.emit_te(topi.concatenate, (lv1, lv4, lv5), axis=1)
                lv7 = bb.emit_te(topi.squeeze, lv6)
                gv = bb.emit_output(bb.call_te(topi.add, lv7, p4))
            bb.emit_func_output(gv)

        # Get the global variables of the grouped functions
        fused_func = bb.get().get_global_var(
            "fused_squeeze_add_squeeze1_add_add_add_concatenate_squeeze2_add1"
        )

        # Main func
        x = relax.Var("x", R.Tensor((1, 16, 64, 64), "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                gv = bb.emit_output(
                    relax.Call(
                        fused_func,
                        (
                            x,
                            relax.const(1, "float32"),
                            relax.const(1, "float32"),
                            relax.const(1, "float32"),
                            relax.const(1, "float32"),
                            relax.const(1, "float32"),
                        ),
                    )
                )
            bb.emit_func_output(gv)

        return bb.get()

    _check(before(), expected())


def test_tuple_consecutive():
    def before():
        bb = relax.BlockBuilder()

        x = relax.Var("x", R.Tensor((1, 16, 64, 64), "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.add, x, relax.const(1, "float32"))
                lv1 = bb.emit_te(topi.add, x, relax.const(1, "float32"))
                lv2 = bb.emit_te(topi.add, x, relax.const(1, "float32"))
                lv3 = bb.emit_te(topi.concatenate, (lv0, lv1, lv2), axis=1)
                lv4 = bb.emit_te(topi.add, lv3, relax.const(1, "float32"))
                lv5 = bb.emit_te(topi.add, x, relax.const(1, "float32"))
                lv6 = bb.emit_te(topi.add, x, relax.const(1, "float32"))
                lv7 = bb.emit_te(topi.add, x, relax.const(1, "float32"))
                lv8 = bb.emit_te(topi.concatenate, (lv5, lv6, lv7), axis=1)
                lv9 = bb.emit_te(topi.add, lv8, relax.const(1, "float32"))
                lv10 = bb.emit_te(topi.add, x, relax.const(1, "float32"))
                lv11 = bb.emit_te(topi.add, x, relax.const(1, "float32"))
                lv12 = bb.emit_te(topi.add, x, relax.const(1, "float32"))
                lv13 = bb.emit_te(topi.concatenate, (lv10, lv11, lv12), axis=1)
                lv14 = bb.emit_te(topi.add, lv13, relax.const(1, "float32"))
                lv15 = bb.emit_te(topi.concatenate, (lv4, lv9, lv14), axis=1)
                lv16 = bb.emit_te(
                    topi.nn.pool2d,
                    lv15,
                    kernel=(2, 2),
                    stride=(2, 2),
                    dilation=(1, 1),
                    padding=(0, 0, 0, 0),
                    pool_type="max",
                )
                lv17 = bb.emit_te(topi.add, lv16, relax.const(1, "float32"))
                lv18 = bb.emit_te(topi.add, lv17, relax.const(1, "float32"))
                gv = bb.emit_output((lv17, lv18))
            bb.emit_func_output(gv)

        return bb.get()

    def expected():
        bb = relax.BlockBuilder()

        # Grouped function 1
        x = relax.Var("x", R.Tensor((1, 16, 64, 64), "float32"))
        p0 = relax.Var("p0", R.Tensor((), "float32"))
        p1 = relax.Var("p1", R.Tensor((), "float32"))
        p2 = relax.Var("p2", R.Tensor((), "float32"))
        p3 = relax.Var("p3", R.Tensor((), "float32"))
        p4 = relax.Var("p4", R.Tensor((), "float32"))
        p5 = relax.Var("p5", R.Tensor((), "float32"))
        p6 = relax.Var("p6", R.Tensor((), "float32"))
        p7 = relax.Var("p7", R.Tensor((), "float32"))
        p8 = relax.Var("p8", R.Tensor((), "float32"))
        p9 = relax.Var("p9", R.Tensor((), "float32"))
        p10 = relax.Var("p10", R.Tensor((), "float32"))
        p11 = relax.Var("p11", R.Tensor((), "float32"))
        with bb.function(
            "fused_add_add_add_concatenate_add1_add_add_add_concatenate_add1_add_add_add_concatenate_add1_concatenate1",
            [x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11],
            attrs={"Primitive": True},
            private=True,
        ):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.add, x, p0)
                lv1 = bb.emit_te(topi.add, x, p1)
                lv2 = bb.emit_te(topi.add, x, p2)
                lv3 = bb.emit_te(topi.concatenate, (lv0, lv1, lv2), axis=1)
                lv4 = bb.emit_te(topi.add, lv3, p3)
                lv5 = bb.emit_te(topi.add, x, p4)
                lv6 = bb.emit_te(topi.add, x, p5)
                lv7 = bb.emit_te(topi.add, x, p6)
                lv8 = bb.emit_te(topi.concatenate, (lv5, lv6, lv7), axis=1)
                lv9 = bb.emit_te(topi.add, lv8, p7)
                lv10 = bb.emit_te(topi.add, x, p8)
                lv11 = bb.emit_te(topi.add, x, p9)
                lv12 = bb.emit_te(topi.add, x, p10)
                lv13 = bb.emit_te(topi.concatenate, (lv10, lv11, lv12), axis=1)
                lv14 = bb.emit_te(topi.add, lv13, p11)
                gv = bb.emit_output(bb.call_te(topi.concatenate, (lv4, lv9, lv14), axis=1))
            bb.emit_func_output(gv)

        # Grouped function 2
        concat = relax.Var("concat", R.Tensor((1, 144, 64, 64), "float32"))
        p0 = relax.Var("p0", R.Tensor((), "float32"))
        with bb.function(
            "fused_pool2d_add2", [concat, p0], attrs={"Primitive": True}, private=True
        ):
            with bb.dataflow():
                lv0 = bb.emit_te(
                    topi.nn.pool2d,
                    concat,
                    kernel=(2, 2),
                    stride=(2, 2),
                    dilation=(1, 1),
                    padding=(0, 0, 0, 0),
                    pool_type="max",
                )
                gv = bb.emit_output(bb.call_te(topi.add, lv0, p0))
            bb.emit_func_output(gv)

        # Get the global variables of the grouped functions
        mod = bb.get()
        fused_func1 = mod.get_global_var(
            "fused_add_add_add_concatenate_add1_add_add_add_concatenate_add1_add_add_add_concatenate_add1_concatenate1"
        )
        fused_func2 = mod.get_global_var("fused_pool2d_add2")

        # Main function
        x = relax.Var("x", R.Tensor((1, 16, 64, 64), "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv0 = bb.emit(
                    relax.Call(
                        fused_func1,
                        (
                            x,
                            relax.const(1, "float32"),
                            relax.const(1, "float32"),
                            relax.const(1, "float32"),
                            relax.const(1, "float32"),
                            relax.const(1, "float32"),
                            relax.const(1, "float32"),
                            relax.const(1, "float32"),
                            relax.const(1, "float32"),
                            relax.const(1, "float32"),
                            relax.const(1, "float32"),
                            relax.const(1, "float32"),
                            relax.const(1, "float32"),
                        ),
                    )
                )
                lv1 = bb.emit(relax.Call(fused_func2, (lv0, relax.const(1, "float32"))))
                lv2 = bb.emit_te(topi.add, lv1, relax.const(1, "float32"))
                gv = bb.emit_output((lv1, lv2))
            bb.emit_func_output(gv)

        return bb.get()

    _check(before(), expected())


def test_inception_like():
    def before():
        bb = relax.BlockBuilder()

        x = relax.Var("x", R.Tensor((1, 16, 64, 64), "float32"))
        w0 = relax.Var("w0", R.Tensor((16, 16, 3, 3), "float32"))
        w1 = relax.Var("w1", R.Tensor((16, 16, 3, 3), "float32"))
        w2 = relax.Var("w2", R.Tensor((16, 32, 3, 3), "float32"))
        w3 = relax.Var("w3", R.Tensor((16, 32, 3, 3), "float32"))
        with bb.function("main", [x, w0, w1, w2, w3]):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.nn.conv2d, x, w0, strides=1, padding=1, dilation=1)
                lv1 = bb.emit_te(topi.nn.relu, lv0)
                lv2 = bb.emit_te(topi.nn.conv2d, x, w1, strides=1, padding=1, dilation=1)
                lv3 = bb.emit_te(topi.nn.relu, lv2)
                lv4 = bb.emit_te(topi.concatenate, (lv1, lv3), axis=1)
                lv5 = bb.emit_te(topi.nn.conv2d, lv4, w2, strides=1, padding=1, dilation=1)
                lv6 = bb.emit_te(topi.nn.relu, lv5)
                lv7 = bb.emit_te(topi.nn.conv2d, lv4, w3, strides=1, padding=1, dilation=1)
                lv8 = bb.emit_te(topi.nn.relu, lv7)
                gv = bb.emit_output(bb.call_te(topi.concatenate, (lv6, lv8), axis=1))
            bb.emit_func_output(gv)

        return bb.get()

    def expected():
        bb = relax.BlockBuilder()

        # Grouped function 1
        x = relax.Var("x", R.Tensor((1, 16, 64, 64), "float32"))
        w = relax.Var("w", R.Tensor((16, 16, 3, 3), "float32"))
        with bb.function("fused_conv2d_relu", [x, w], attrs={"Primitive": True}, private=True):
            with bb.dataflow():
                lv0 = bb.emit_te(
                    topi.nn.conv2d,
                    x,
                    w,
                    strides=1,
                    padding=1,
                    dilation=1,
                    function_name_hint="conv2d",
                )
                gv = bb.emit_output(bb.call_te(topi.nn.relu, lv0))
            bb.emit_func_output(gv)

        # Grouped function 2
        x = relax.Var("x", R.Tensor((1, 32, 64, 64), "float32"))
        w = relax.Var("w", R.Tensor((16, 32, 3, 3), "float32"))
        with bb.function("fused_conv2d1_relu", [x, w], attrs={"Primitive": True}, private=True):
            with bb.dataflow():
                lv0 = bb.emit_te(
                    topi.nn.conv2d,
                    x,
                    w,
                    strides=1,
                    padding=1,
                    dilation=1,
                    function_name_hint="conv2d1",
                )
                gv = bb.emit_output(bb.call_te(topi.nn.relu, lv0))
            bb.emit_func_output(gv)

        # Get the global variables of the grouped functions
        mod = bb.get()
        fused_conv2d_relu1 = mod.get_global_var("fused_conv2d_relu")
        fused_conv2d_relu2 = mod.get_global_var("fused_conv2d1_relu")

        # Main function
        x = relax.Var("x", R.Tensor((1, 16, 64, 64), "float32"))
        w0 = relax.Var("w0", R.Tensor((16, 16, 3, 3), "float32"))
        w1 = relax.Var("w1", R.Tensor((16, 16, 3, 3), "float32"))
        w2 = relax.Var("w2", R.Tensor((16, 32, 3, 3), "float32"))
        w3 = relax.Var("w3", R.Tensor((16, 32, 3, 3), "float32"))
        with bb.function("main", [x, w0, w1, w2, w3]):
            with bb.dataflow():
                lv0 = bb.emit(relax.Call(fused_conv2d_relu1, (x, w0)))
                lv1 = bb.emit(relax.Call(fused_conv2d_relu1, (x, w1)))
                lv2 = bb.emit_te(topi.concatenate, (lv0, lv1), axis=1)
                lv3 = bb.emit(relax.Call(fused_conv2d_relu2, (lv2, w2)))
                lv4 = bb.emit(relax.Call(fused_conv2d_relu2, (lv2, w3)))
                gv = bb.emit_output(bb.call_te(topi.concatenate, (lv3, lv4), axis=1))
            bb.emit_func_output(gv)

        return bb.get()

    _check(before(), expected())


def test_fuse_parallel_injective():
    def before():
        bb = relax.BlockBuilder()

        x = relax.Var("x", R.Tensor((10, 20), "int32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.add, x, relax.const(1, "int32"))
                lv1 = bb.emit_te(topi.squeeze, lv0)
                lv2 = bb.emit_te(topi.transpose, lv0, axes=[1, 0])
                lv3 = bb.emit_te(topi.transpose, lv2, axes=[1, 0])
                gv = bb.emit_output(bb.call_te(topi.left_shift, lv1, lv3))
            bb.emit_func_output(gv)

        return bb.get()

    def expected():
        bb = relax.BlockBuilder()

        # Grouped function
        x = relax.Var("x", R.Tensor((10, 20), "int32"))
        p0 = relax.Var("p0", R.Tensor((), "int32"))
        with bb.function(
            "fused_add_squeeze_transpose_transpose1_left_shift",
            [x, p0],
            attrs={"Primitive": True},
            private=True,
        ):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.add, x, p0)
                lv1 = bb.emit_te(topi.squeeze, lv0)
                lv2 = bb.emit_te(topi.transpose, lv0, axes=[1, 0])
                lv3 = bb.emit_te(topi.transpose, lv2, axes=[1, 0], function_name_hint="transpose1")
                gv = bb.emit_output(bb.call_te(topi.left_shift, lv1, lv3))
            bb.emit_func_output(gv)

        # Get the global variables of the grouped functions
        fused_func = bb.get().get_global_var("fused_add_squeeze_transpose_transpose1_left_shift")

        # Main function
        x = relax.Var("x", R.Tensor((10, 20), "int32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                gv = bb.emit_output(relax.Call(fused_func, (x, relax.const(1, "int32"))))
            bb.emit_func_output(gv)

        return bb.get()

    _check(before(), expected())


def test_softmax():
    """Test if softmax can be fused with following ops."""

    def before():
        bb = relax.BlockBuilder()

        x = relax.Var("x", R.Tensor((16, 16), "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.nn.softmax, x)
                gv = bb.emit_output(bb.call_te(topi.cast, lv0, dtype="float16"))
            bb.emit_func_output(gv)

        return bb.get()

    def expected():
        bb = relax.BlockBuilder()

        # Grouped function
        x = relax.Var("x", R.Tensor((16, 16), "float32"))
        with bb.function("fused_softmax_cast", [x], attrs={"Primitive": True}, private=True):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.nn.softmax, x)
                gv = bb.emit_output(bb.call_te(topi.cast, lv0, dtype="float16"))
            bb.emit_func_output(gv)

        # Get the global variables of the grouped functions
        fused_func = bb.get().get_global_var("fused_softmax_cast")

        # Main function
        x = relax.Var("x", R.Tensor((16, 16), "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                gv = bb.emit_output(relax.Call(fused_func, (x,)))
            bb.emit_func_output(gv)

        return bb.get()

    _check(before(), expected())


def test_multiple_relax_functions():
    def before():
        bb = relax.BlockBuilder()
        x = relax.Var("x", R.Tensor([10, 20], "float32"))
        with bb.function("func1", [x]):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.add, x, relax.const(1, "float32"))
                lv1 = bb.emit_te(topi.exp, lv0)
                gv = bb.emit_output(bb.call_te(topi.squeeze, lv1))
            bb.emit_func_output(gv)

        x = relax.Var("x", R.Tensor([20, 10], "float32"))
        with bb.function("func2", [x]):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.add, x, relax.const(1, "float32"))
                lv1 = bb.emit_te(topi.exp, lv0)
                gv = bb.emit_output(bb.call_te(topi.squeeze, lv1))
            bb.emit_func_output(gv)

        return bb.get()

    def expected():
        bb = relax.BlockBuilder()

        x = relax.Var("x", R.Tensor([10, 20], "float32"))
        p0 = relax.Var("p0", R.Tensor((), "float32"))
        with bb.function("fused_add_exp_squeeze", [x, p0], attrs={"Primitive": True}, private=True):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.add, x, p0)
                lv1 = bb.emit_te(topi.exp, lv0)
                gv = bb.emit_output(bb.call_te(topi.squeeze, lv1))
            bb.emit_func_output(gv)
        fused_add_exp_squeeze = bb.get().get_global_var("fused_add_exp_squeeze")

        x = relax.Var("x", R.Tensor([20, 10], "float32"))
        p0 = relax.Var("p0", R.Tensor((), "float32"))
        with bb.function(
            "fused_add1_exp1_squeeze1", [x, p0], attrs={"Primitive": True}, private=True
        ):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.add, x, p0)
                lv1 = bb.emit_te(topi.exp, lv0)
                gv = bb.emit_output(bb.call_te(topi.squeeze, lv1))
            bb.emit_func_output(gv)
        fused_add1_exp1_squeeze1 = bb.get().get_global_var("fused_add1_exp1_squeeze1")

        x = relax.Var("x", R.Tensor([10, 20], "float32"))
        with bb.function("func1", [x]):
            with bb.dataflow():
                gv = bb.emit_output(
                    relax.Call(fused_add_exp_squeeze, [x, relax.const(1, "float32")])
                )
            bb.emit_func_output(gv)

        x = relax.Var("x", R.Tensor([20, 10], "float32"))
        with bb.function("func2", [x]):
            with bb.dataflow():
                gv = bb.emit_output(
                    relax.Call(fused_add1_exp1_squeeze1, [x, relax.const(1, "float32")])
                )
            bb.emit_func_output(gv)

        return bb.get()

    _check(before(), expected())


def test_skip_call_dps_packed():
    @I.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")):
            with R.dataflow():
                y = R.call_dps_packed(
                    _ExternFunc("func_packed_dps"), x, ty_args=[R.Tensor((2, 3), "float32")]
                )
                R.output(y)
            return y

    # FuseOps should does no change to it.
    _check(Module, Module)


def test_edge_with_call_dps_packed():
    @I.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")):
            cls = Module
            with R.dataflow():
                a = R.call_tir(cls.exp, (x,), ty_args=[R.Tensor((2, 3), "float32")])
                b = R.call_tir(cls.exp, (a,), ty_args=[R.Tensor((2, 3), "float32")])
                c = R.call_dps_packed(
                    _ExternFunc("packed_dps"), (a,), ty_args=[R.Tensor((2, 3), "float32")]
                )
                R.output(b, c)
            return (b, c)

        @Ts.function(private=True)
        def exp(A: T.Tensor((2, 3), "float32"), B: T.Tensor((2, 3), "float32")):
            T.evaluate(0)

    # FuseOps should does no change to it.
    _check(Module, Module)


def test_layer_norm_silu():
    # fmt: off
    @I.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor((1, 512, 64, 64), "float32"), mean: R.Tensor((64, 64), "float32"), var: R.Tensor((64, 64), "float32")):
            cls = Module
            with R.dataflow():
                gv0 = R.call_tir(cls.layer_norm, (x, mean, var), ty_args=[R.Tensor((1, 512, 64, 64), 'float32')])
                gv1 = R.call_tir(cls.relu, gv0, ty_args=[R.Tensor((1, 512, 64, 64), "float32")])
                R.output(gv1)
            return gv1

        @Ts.function(private=True)
        def layer_norm(A: T.Tensor((T.int64(1), T.int64(512), T.int64(64), T.int64(64)), "float32"), gamma: T.Tensor((T.int64(64), T.int64(64)), "float32"), beta: T.Tensor((T.int64(64), T.int64(64)), "float32"), T_layer_norm: T.Tensor((T.int64(1), T.int64(512), T.int64(64), T.int64(64)), "float32")):
            rxplaceholder_red_temp_v0 = Ts.sblock_alloc_tensor([T.int64(64), T.int64(64)], dtype="float32")
            rxplaceholder_red_temp_v1 = Ts.sblock_alloc_tensor([T.int64(64), T.int64(64)], dtype="float32")
            for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(512), T.int64(64), T.int64(64)):
                with Ts.sblock("rxplaceholder_red_temp"):
                    ax0, ax1, k2, k3 = Ts.axis.remap("SSRR", [i0, i1, i2, i3])
                    Ts.reads(A[ax0, ax1, k2, k3])
                    Ts.writes(rxplaceholder_red_temp_v0[ax0, ax1], rxplaceholder_red_temp_v1[ax0, ax1])
                    with Ts.init():
                        rxplaceholder_red_temp_v0[ax0, ax1] = T.float32(0)
                        rxplaceholder_red_temp_v1[ax0, ax1] = T.float32(0)
                    v_rxplaceholder_red_temp_v0: T.let[T.float32] = rxplaceholder_red_temp_v0[ax0, ax1] + A[ax0, ax1, k2, k3]
                    v_rxplaceholder_red_temp_v1: T.let[T.float32] = rxplaceholder_red_temp_v1[ax0, ax1] + A[ax0, ax1, k2, k3] * A[ax0, ax1, k2, k3]
                    rxplaceholder_red_temp_v0[ax0, ax1] = v_rxplaceholder_red_temp_v0
                    rxplaceholder_red_temp_v1[ax0, ax1] = v_rxplaceholder_red_temp_v1
            for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(512), T.int64(64), T.int64(64)):
                with Ts.sblock("T_layer_norm"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(A[ax0, ax1, ax2, ax3], rxplaceholder_red_temp_v0[ax0, ax1], rxplaceholder_red_temp_v1[ax0, ax1], gamma[ax2, ax3], beta[ax2, ax3])
                    Ts.writes(T_layer_norm[ax0, ax1, ax2, ax3])
                    T_layer_norm[ax0, ax1, ax2, ax3] = (A[ax0, ax1, ax2, ax3] - rxplaceholder_red_temp_v0[ax0, ax1] * T.float32(0.05)) * T.rsqrt(rxplaceholder_red_temp_v1[ax0, ax1] * T.float32(0.05) - rxplaceholder_red_temp_v0[ax0, ax1] * T.float32(0.05) * (rxplaceholder_red_temp_v0[ax0, ax1] * T.float32(0.05)) + T.float32(1e-05)) * gamma[ax2, ax3] + beta[ax2, ax3]

        @Ts.function(private=True)
        def relu(A: T.Tensor((T.int64(1), T.int64(512), T.int64(64), T.int64(64)), "float32"), B: T.Tensor((T.int64(1), T.int64(512), T.int64(64), T.int64(64)), "float32")):
            for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(512), T.int64(64), T.int64(64)):
                with Ts.sblock("relu"):
                    v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(A[v_i0, v_i1, v_i2, v_i3])
                    Ts.writes(B[v_i0, v_i1, v_i2, v_i3])
                    B[v_i0, v_i1, v_i2, v_i3] = T.max(A[v_i0, v_i1, v_i2, v_i3], T.float32(0))

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def layer_norm(A: T.Tensor((T.int64(1), T.int64(512), T.int64(64), T.int64(64)), "float32"), gamma: T.Tensor((T.int64(64), T.int64(64)), "float32"), beta: T.Tensor((T.int64(64), T.int64(64)), "float32"), T_layer_norm: T.Tensor((T.int64(1), T.int64(512), T.int64(64), T.int64(64)), "float32")):
            T.func_attr({"op_pattern": 4})
            # with Ts.sblock("root"):
            rxplaceholder_red_temp_v0 = Ts.sblock_alloc_tensor((T.int64(64), T.int64(64)))
            rxplaceholder_red_temp_v1 = Ts.sblock_alloc_tensor((T.int64(64), T.int64(64)))
            for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(512), T.int64(64), T.int64(64)):
                with Ts.sblock("rxplaceholder_red_temp"):
                    ax0, ax1, k2, k3 = Ts.axis.remap("SSRR", [i0, i1, i2, i3])
                    Ts.reads(A[ax0, ax1, k2, k3])
                    Ts.writes(rxplaceholder_red_temp_v0[ax0, ax1], rxplaceholder_red_temp_v1[ax0, ax1])
                    with Ts.init():
                        rxplaceholder_red_temp_v0[ax0, ax1] = T.float32(0)
                        rxplaceholder_red_temp_v1[ax0, ax1] = T.float32(0)
                    v_rxplaceholder_red_temp_v0: T.let[T.float32] = rxplaceholder_red_temp_v0[ax0, ax1] + A[ax0, ax1, k2, k3]
                    v_rxplaceholder_red_temp_v1: T.let[T.float32] = rxplaceholder_red_temp_v1[ax0, ax1] + A[ax0, ax1, k2, k3] * A[ax0, ax1, k2, k3]
                    rxplaceholder_red_temp_v0[ax0, ax1] = v_rxplaceholder_red_temp_v0
                    rxplaceholder_red_temp_v1[ax0, ax1] = v_rxplaceholder_red_temp_v1
            for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(512), T.int64(64), T.int64(64)):
                with Ts.sblock("T_layer_norm"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(A[ax0, ax1, ax2, ax3], rxplaceholder_red_temp_v0[ax0, ax1], rxplaceholder_red_temp_v1[ax0, ax1], gamma[ax2, ax3], beta[ax2, ax3])
                    Ts.writes(T_layer_norm[ax0, ax1, ax2, ax3])
                    T_layer_norm[ax0, ax1, ax2, ax3] = (A[ax0, ax1, ax2, ax3] - rxplaceholder_red_temp_v0[ax0, ax1] * T.float32(0.050000000000000003)) * T.rsqrt(rxplaceholder_red_temp_v1[ax0, ax1] * T.float32(0.050000000000000003) - rxplaceholder_red_temp_v0[ax0, ax1] * T.float32(0.050000000000000003) * (rxplaceholder_red_temp_v0[ax0, ax1] * T.float32(0.050000000000000003)) + T.float32(1.0000000000000001e-05)) * gamma[ax2, ax3] + beta[ax2, ax3]

        @Ts.function(private=True)
        def relu(A: T.Tensor((T.int64(1), T.int64(512), T.int64(64), T.int64(64)), "float32"), B: T.Tensor((T.int64(1), T.int64(512), T.int64(64), T.int64(64)), "float32")):
            T.func_attr({"op_pattern": 0})
            # with Ts.sblock("root"):
            for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(512), T.int64(64), T.int64(64)):
                with Ts.sblock("relu"):
                    v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(A[v_i0, v_i1, v_i2, v_i3])
                    Ts.writes(B[v_i0, v_i1, v_i2, v_i3])
                    B[v_i0, v_i1, v_i2, v_i3] = T.max(A[v_i0, v_i1, v_i2, v_i3], T.float32(0))

        @R.function(private=True)
        def fused_layer_norm_relu(x: R.Tensor((1, 512, 64, 64), dtype="float32"), mean: R.Tensor((64, 64), dtype="float32"), var: R.Tensor((64, 64), dtype="float32")) -> R.Tensor((1, 512, 64, 64), dtype="float32"):
            R.func_attr({"Primitive": True})
            cls = Expected
            with R.dataflow():
                gv0 = R.call_tir(cls.layer_norm, (x, mean, var), ty_args=[R.Tensor((1, 512, 64, 64), 'float32')])
                gv = R.call_tir(cls.relu, (gv0,), ty_args=[R.Tensor((1, 512, 64, 64), dtype="float32")])
                R.output(gv)
            return gv

        @R.function
        def main(x: R.Tensor((1, 512, 64, 64), dtype="float32"), mean: R.Tensor((64, 64), dtype="float32"), var: R.Tensor((64, 64), dtype="float32")) -> R.Tensor((1, 512, 64, 64), dtype="float32"):
            cls = Expected
            with R.dataflow():
                gv: R.Tensor((1, 512, 64, 64), dtype="float32") = cls.fused_layer_norm_relu(x, mean, var)
                R.output(gv)
            return gv
    # fmt: on

    _check(Module, Expected)


def test_multiple_paths():
    # fmt: off
    @I.ir_module
    class Module:
        @R.function
        def main(
            inp_0: R.Tensor((2, 320, 64, 64), dtype="float32"),
            inp_1: R.Tensor((2, 1280), dtype="float32"),
            w1: R.Tensor((320, 320, 3, 3), dtype="float32"),
            b1: R.Tensor((320,), "float32"),
            w2: R.Tensor((320, 1280), "float32"),
            b2: R.Tensor((320,), "float32"),
        ):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv27: R.Tensor((2, 320, 64, 64), dtype="float32") = R.nn.conv2d(inp_0, w1, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
                lv28: R.Tensor((1, 320, 1, 1), dtype="float32") = R.reshape(b1, R.shape([1, 320, 1, 1]))  ##
                lv29: R.Tensor((2, 320, 64, 64), dtype="float32") = R.add(lv27, lv28)
                lv31: R.Tensor((1280, 320), dtype="float32") = R.permute_dims(w2, axes=None)  ##
                lv32: R.Tensor((2, 320), dtype="float32") = R.matmul(inp_1, lv31, out_dtype="float32")
                lv33: R.Tensor((2, 320), dtype="float32") = R.add(lv32, b2)
                lv35: R.Tensor((2, 320, 1, 1), dtype="float32") = R.reshape(lv33, R.shape([2, 320, 1, 1]))
                lv36: R.Tensor((2, 320, 64, 64), dtype="float32") = R.add(lv29, lv35)
                gv = lv36
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def add(rxplaceholder: T.Tensor((T.int64(2), T.int64(320), T.int64(64), T.int64(64)), "float32"), rxplaceholder_1: T.Tensor((T.int64(1), T.int64(320), T.int64(1), T.int64(1)), "float32"), T_add: T.Tensor((T.int64(2), T.int64(320), T.int64(64), T.int64(64)), "float32")):
            T.func_attr({"op_pattern": 0, "tirx.noalias": True})
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(320), T.int64(64), T.int64(64)):
                with Ts.sblock("T_add"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3], rxplaceholder_1[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                    Ts.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_add[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3] + rxplaceholder_1[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

        @Ts.function(private=True)
        def add1(rxplaceholder: T.Tensor((T.int64(2), T.int64(320)), "float32"), rxplaceholder_1: T.Tensor((T.int64(320),), "float32"), T_add: T.Tensor((T.int64(2), T.int64(320)), "float32")):
            T.func_attr({"op_pattern": 0, "tirx.noalias": True})
            for ax0, ax1 in T.grid(T.int64(2), T.int64(320)):
                with Ts.sblock("T_add"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(rxplaceholder[v_ax0, v_ax1], rxplaceholder_1[v_ax1])
                    Ts.writes(T_add[v_ax0, v_ax1])
                    T_add[v_ax0, v_ax1] = rxplaceholder[v_ax0, v_ax1] + rxplaceholder_1[v_ax1]

        @Ts.function(private=True)
        def add2(rxplaceholder: T.Tensor((T.int64(2), T.int64(320), T.int64(64), T.int64(64)), "float32"), rxplaceholder_1: T.Tensor((T.int64(2), T.int64(320), T.int64(1), T.int64(1)), "float32"), T_add: T.Tensor((T.int64(2), T.int64(320), T.int64(64), T.int64(64)), "float32")):
            T.func_attr({"op_pattern": 0, "tirx.noalias": True})
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(320), T.int64(64), T.int64(64)):
                with Ts.sblock("T_add"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3], rxplaceholder_1[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    Ts.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_add[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3] + rxplaceholder_1[v_ax0, v_ax1, T.int64(0), T.int64(0)]

        @Ts.function(private=True)
        def conv2d(rxplaceholder: T.Tensor((T.int64(2), T.int64(320), T.int64(64), T.int64(64)), "float32"), rxplaceholder_1: T.Tensor((T.int64(320), T.int64(320), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Tensor((T.int64(2), T.int64(320), T.int64(64), T.int64(64)), "float32")):
            T.func_attr({"op_pattern": 4, "tirx.noalias": True})
            pad_temp = Ts.sblock_alloc_tensor((T.int64(2), T.int64(320), T.int64(66), T.int64(66)))
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(320), T.int64(66), T.int64(66)):
                with Ts.sblock("pad_temp"):
                    v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                    Ts.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                    pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(65) and T.int64(1) <= v_i3 and v_i3 < T.int64(65), rxplaceholder[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
            for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(2), T.int64(320), T.int64(64), T.int64(64), T.int64(320), T.int64(3), T.int64(3)):
                with Ts.sblock("conv2d_nchw"):
                    v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = Ts.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
                    Ts.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], rxplaceholder_1[v_ff, v_rc, v_ry, v_rx])
                    Ts.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                    with Ts.init():
                        conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
                    conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * rxplaceholder_1[v_ff, v_rc, v_ry, v_rx]

        @Ts.function(private=True)
        def matmul(rxplaceholder: T.Tensor((T.int64(2), T.int64(1280)), "float32"), rxplaceholder_1: T.Tensor((T.int64(1280), T.int64(320)), "float32"), matmul: T.Tensor((T.int64(2), T.int64(320)), "float32")):
            T.func_attr({"op_pattern": 4, "tirx.noalias": True})
            for i0, i1, k in T.grid(T.int64(2), T.int64(320), T.int64(1280)):
                with Ts.sblock("matmul"):
                    v_i0, v_i1, v_k = Ts.axis.remap("SSR", [i0, i1, k])
                    Ts.reads(rxplaceholder[v_i0, v_k], rxplaceholder_1[v_k, v_i1])
                    Ts.writes(matmul[v_i0, v_i1])
                    with Ts.init():
                        matmul[v_i0, v_i1] = T.float32(0)
                    matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + rxplaceholder[v_i0, v_k] * rxplaceholder_1[v_k, v_i1]

        @Ts.function(private=True)
        def reshape(rxplaceholder: T.Tensor((T.int64(320),), "float32"), T_reshape: T.Tensor((T.int64(1), T.int64(320), T.int64(1), T.int64(1)), "float32")):
            T.func_attr({"op_pattern": 2, "tirx.noalias": True})
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(320), T.int64(1), T.int64(1)):
                with Ts.sblock("T_reshape"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(rxplaceholder[(v_ax1 + v_ax2 + v_ax3) % T.int64(320)])
                    Ts.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[(v_ax1 + v_ax2 + v_ax3) % T.int64(320)]

        @Ts.function(private=True)
        def reshape1(rxplaceholder: T.Tensor((T.int64(2), T.int64(320)), "float32"), T_reshape: T.Tensor((T.int64(2), T.int64(320), T.int64(1), T.int64(1)), "float32")):
            T.func_attr({"op_pattern": 2, "tirx.noalias": True})
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(320), T.int64(1), T.int64(1)):
                with Ts.sblock("T_reshape"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(rxplaceholder[((v_ax1 + v_ax2 + v_ax3) // T.int64(320) + v_ax0) % T.int64(2), (v_ax1 + v_ax2 + v_ax3) % T.int64(320)])
                    Ts.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[((v_ax1 + v_ax2 + v_ax3) // T.int64(320) + v_ax0) % T.int64(2), (v_ax1 + v_ax2 + v_ax3) % T.int64(320)]

        @Ts.function(private=True)
        def transpose(rxplaceholder: T.Tensor((T.int64(320), T.int64(1280)), "float32"), T_transpose: T.Tensor((T.int64(1280), T.int64(320)), "float32")):
            T.func_attr({"op_pattern": 2, "tirx.noalias": True})
            for ax0, ax1 in T.grid(T.int64(1280), T.int64(320)):
                with Ts.sblock("T_transpose"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(rxplaceholder[v_ax1, v_ax0])
                    Ts.writes(T_transpose[v_ax0, v_ax1])
                    T_transpose[v_ax0, v_ax1] = rxplaceholder[v_ax1, v_ax0]

        @R.function(private=True)
        def fused_conv2d_add_add2(inp_0: R.Tensor((2, 320, 64, 64), dtype="float32"), w1: R.Tensor((320, 320, 3, 3), dtype="float32"), lv28: R.Tensor((1, 320, 1, 1), dtype="float32"), lv35: R.Tensor((2, 320, 1, 1), dtype="float32")) -> R.Tensor((2, 320, 64, 64), dtype="float32"):
            R.func_attr({"Primitive": True})
            cls = Expected
            with R.dataflow():
                lv27 = R.call_tir(cls.conv2d, (inp_0, w1), ty_args=[R.Tensor((2, 320, 64, 64), dtype="float32")])
                lv29 = R.call_tir(cls.add, (lv27, lv28), ty_args=[R.Tensor((2, 320, 64, 64), dtype="float32")])
                gv = R.call_tir(cls.add2, (lv29, lv35), ty_args=[R.Tensor((2, 320, 64, 64), dtype="float32")])
                R.output(gv)
            return gv

        @R.function(private=True)
        def fused_matmul_add1(inp_1: R.Tensor((2, 1280), dtype="float32"), lv31: R.Tensor((1280, 320), dtype="float32"), b2: R.Tensor((320,), dtype="float32")) -> R.Tensor((2, 320), dtype="float32"):
            cls = Expected
            R.func_attr({"Primitive": True})
            with R.dataflow():
                lv32 = R.call_tir(cls.matmul, (inp_1, lv31), ty_args=[R.Tensor((2, 320), dtype="float32")])
                gv = R.call_tir(cls.add1, (lv32, b2), ty_args=[R.Tensor((2, 320), dtype="float32")])
                R.output(gv)
            return gv

        @R.function
        def main(inp_0: R.Tensor((2, 320, 64, 64), dtype="float32"), inp_1: R.Tensor((2, 1280), dtype="float32"), w1: R.Tensor((320, 320, 3, 3), dtype="float32"), b1: R.Tensor((320,), dtype="float32"), w2: R.Tensor((320, 1280), dtype="float32"), b2: R.Tensor((320,), dtype="float32")) -> R.Tensor((2, 320, 64, 64), dtype="float32"):
            R.func_attr({"num_input": 2})
            cls = Expected
            with R.dataflow():
                lv28 = R.call_tir(cls.reshape, (b1,), ty_args=[R.Tensor((1, 320, 1, 1), dtype="float32")])
                lv31 = R.call_tir(cls.transpose, (w2,), ty_args=[R.Tensor((1280, 320), dtype="float32")])
                lv: R.Tensor((2, 320), dtype="float32") = cls.fused_matmul_add1(inp_1, lv31, b2)
                lv35 = R.call_tir(cls.reshape1, (lv,), ty_args=[R.Tensor((2, 320, 1, 1), dtype="float32")])
                lv1: R.Tensor((2, 320, 64, 64), dtype="float32") = cls.fused_conv2d_add_add2(inp_0, w1, lv28, lv35)
                gv: R.Tensor((2, 320, 64, 64), dtype="float32") = lv1
                R.output(gv)
            return gv
    # fmt: on

    mod = relax.transform.LegalizeOps()(Module)
    mod = relax.transform.AnnotateTIROpPattern()(mod)
    mod = relax.transform.FuseOps()(mod)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_dead_group():
    # fmt: off

    @I.ir_module
    class Module:
        @R.function
        def main(inp_0: R.Tensor((1, 784), dtype="float32"), inp_1: R.Tensor((1, 128), dtype="float32"), linear1_bias: R.Tensor((128,), dtype="float32"), linear1_weight: R.Tensor((128, 784), dtype="float32"), linear2_bias: R.Tensor((10,), dtype="float32"), linear2_weight: R.Tensor((10, 128), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((784, 128), dtype="float32") = R.permute_dims(linear1_weight, axes=None)
                lv1: R.Tensor((1, 128), dtype="float32") = R.matmul(inp_0, lv, out_dtype="float32")
                lv2: R.Tensor((1, 128), dtype="float32") = R.add(lv1, linear1_bias)
                lv3: R.Tensor((1, 128), dtype="float32") = R.nn.relu(lv2)
                lv4: R.Tensor((128, 10), dtype="float32") = R.permute_dims(linear2_weight, axes=None)
                lv5: R.Tensor((1, 10), dtype="float32") = R.matmul(inp_1, lv4, out_dtype="float32")
                lv6: R.Tensor((1, 10), dtype="float32") = R.add(lv5, linear2_bias)
                gv: R.Tensor((1, 10), dtype="float32") = lv6
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def add(rxplaceholder: T.Tensor((T.int64(1), T.int64(128)), "float32"), rxplaceholder_1: T.Tensor((T.int64(128),), "float32"), T_add: T.Tensor((T.int64(1), T.int64(128)), "float32")):
            T.func_attr({"op_pattern": 0, "tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(128)):
                with Ts.sblock("T_add"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(rxplaceholder[v_ax0, v_ax1], rxplaceholder_1[v_ax1])
                    Ts.writes(T_add[v_ax0, v_ax1])
                    T_add[v_ax0, v_ax1] = rxplaceholder[v_ax0, v_ax1] + rxplaceholder_1[v_ax1]

        @Ts.function(private=True)
        def add1(rxplaceholder: T.Tensor((T.int64(1), T.int64(10)), "float32"), rxplaceholder_1: T.Tensor((T.int64(10),), "float32"), T_add: T.Tensor((T.int64(1), T.int64(10)), "float32")):
            T.func_attr({"op_pattern": 0, "tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(10)):
                with Ts.sblock("T_add"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(rxplaceholder[v_ax0, v_ax1], rxplaceholder_1[v_ax1])
                    Ts.writes(T_add[v_ax0, v_ax1])
                    T_add[v_ax0, v_ax1] = rxplaceholder[v_ax0, v_ax1] + rxplaceholder_1[v_ax1]

        @Ts.function(private=True)
        def matmul(rxplaceholder: T.Tensor((T.int64(1), T.int64(784)), "float32"), rxplaceholder_1: T.Tensor((T.int64(784), T.int64(128)), "float32"), matmul_1: T.Tensor((T.int64(1), T.int64(128)), "float32")):
            T.func_attr({"op_pattern": 4, "tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0, i1, k in T.grid(T.int64(1), T.int64(128), T.int64(784)):
                with Ts.sblock("matmul"):
                    v_i0, v_i1, v_k = Ts.axis.remap("SSR", [i0, i1, k])
                    Ts.reads(rxplaceholder[v_i0, v_k], rxplaceholder_1[v_k, v_i1])
                    Ts.writes(matmul_1[v_i0, v_i1])
                    with Ts.init():
                        matmul_1[v_i0, v_i1] = T.float32(0)
                    matmul_1[v_i0, v_i1] = matmul_1[v_i0, v_i1] + rxplaceholder[v_i0, v_k] * rxplaceholder_1[v_k, v_i1]

        @Ts.function(private=True)
        def matmul1(rxplaceholder: T.Tensor((T.int64(1), T.int64(128)), "float32"), rxplaceholder_1: T.Tensor((T.int64(128), T.int64(10)), "float32"), matmul: T.Tensor((T.int64(1), T.int64(10)), "float32")):
            T.func_attr({"op_pattern": 4, "tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0, i1, k in T.grid(T.int64(1), T.int64(10), T.int64(128)):
                with Ts.sblock("matmul"):
                    v_i0, v_i1, v_k = Ts.axis.remap("SSR", [i0, i1, k])
                    Ts.reads(rxplaceholder[v_i0, v_k], rxplaceholder_1[v_k, v_i1])
                    Ts.writes(matmul[v_i0, v_i1])
                    with Ts.init():
                        matmul[v_i0, v_i1] = T.float32(0)
                    matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + rxplaceholder[v_i0, v_k] * rxplaceholder_1[v_k, v_i1]

        @Ts.function(private=True)
        def relu(rxplaceholder: T.Tensor((T.int64(1), T.int64(128)), "float32"), compute: T.Tensor((T.int64(1), T.int64(128)), "float32")):
            T.func_attr({"op_pattern": 0, "tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0, i1 in T.grid(T.int64(1), T.int64(128)):
                with Ts.sblock("compute"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[v_i0, v_i1])
                    Ts.writes(compute[v_i0, v_i1])
                    compute[v_i0, v_i1] = T.max(rxplaceholder[v_i0, v_i1], T.float32(0))

        @Ts.function(private=True)
        def transpose(rxplaceholder: T.Tensor((T.int64(128), T.int64(784)), "float32"), T_transpose: T.Tensor((T.int64(784), T.int64(128)), "float32")):
            T.func_attr({"op_pattern": 2, "tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1 in T.grid(T.int64(784), T.int64(128)):
                with Ts.sblock("T_transpose"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(rxplaceholder[v_ax1, v_ax0])
                    Ts.writes(T_transpose[v_ax0, v_ax1])
                    T_transpose[v_ax0, v_ax1] = rxplaceholder[v_ax1, v_ax0]

        @Ts.function(private=True)
        def transpose1(rxplaceholder: T.Tensor((T.int64(10), T.int64(128)), "float32"), T_transpose: T.Tensor((T.int64(128), T.int64(10)), "float32")):
            T.func_attr({"op_pattern": 2, "tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1 in T.grid(T.int64(128), T.int64(10)):
                with Ts.sblock("T_transpose"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(rxplaceholder[v_ax1, v_ax0])
                    Ts.writes(T_transpose[v_ax0, v_ax1])
                    T_transpose[v_ax0, v_ax1] = rxplaceholder[v_ax1, v_ax0]

        @R.function(private=True)
        def fused_matmul1_add1(inp_1: R.Tensor((1, 128), dtype="float32"), lv4: R.Tensor((128, 10), dtype="float32"), linear2_bias: R.Tensor((10,), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
            R.func_attr({"Primitive": True})
            cls = Expected
            with R.dataflow():
                lv5 = R.call_tir(cls.matmul1, (inp_1, lv4), ty_args=[R.Tensor((1, 10), dtype="float32")])
                gv = R.call_tir(cls.add1, (lv5, linear2_bias), ty_args=[R.Tensor((1, 10), dtype="float32")])
                R.output(gv)
            return gv

        @R.function
        def main(inp_0: R.Tensor((1, 784), dtype="float32"), inp_1: R.Tensor((1, 128), dtype="float32"), linear1_bias: R.Tensor((128,), dtype="float32"), linear1_weight: R.Tensor((128, 784), dtype="float32"), linear2_bias: R.Tensor((10,), dtype="float32"), linear2_weight: R.Tensor((10, 128), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
            R.func_attr({"num_input": 1})
            cls = Expected
            with R.dataflow():
                lv = R.call_tir(cls.transpose, (linear1_weight,), ty_args=[R.Tensor((784, 128), dtype="float32")])
                lv4 = R.call_tir(cls.transpose1, (linear2_weight,), ty_args=[R.Tensor((128, 10), dtype="float32")])
                lv_1: R.Tensor((1, 10), dtype="float32") = cls.fused_matmul1_add1(inp_1, lv4, linear2_bias)
                gv: R.Tensor((1, 10), dtype="float32") = lv_1
                R.output(gv)
            return gv

    # fmt: on

    mod = relax.transform.LegalizeOps()(Module)
    _check(mod, Expected)


def test_symbolic_shape_aware_fuse():
    n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int64")

    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor([n, m], "float32")):
            with R.dataflow():
                lv0 = R.emit_te(topi.add, x, R.const(1, "float32"))
                lv1 = R.emit_te(topi.exp, lv0)
                gv = R.emit_te(topi.squeeze, lv1)
                R.output(gv)
            return gv

    n_fused_add_exp_squeeze = T.dynamic("n", "int64")
    m_fused_add_exp_squeeze = T.dynamic("m", "int64")
    n_main = T.dynamic("n", "int64")
    m_main = T.dynamic("m", "int64")

    @I.ir_module
    class Expected:
        @R.function(private=True)
        def fused_add_exp_squeeze(
            x: R.Tensor([n_fused_add_exp_squeeze, m_fused_add_exp_squeeze], "float32"),
            p0: R.Tensor([], "float32"),
        ) -> R.Tensor([n_fused_add_exp_squeeze, m_fused_add_exp_squeeze], dtype="float32"):
            R.func_attr({"Primitive": True})
            with R.dataflow():
                lv0 = R.emit_te(topi.add, x, p0)
                lv1 = R.emit_te(topi.exp, lv0)
                gv = R.emit_te(topi.squeeze, lv1)
                R.output(gv)
            return gv

        @R.function
        def main(x: R.Tensor([n_main, m_main], "float32")) -> R.Tensor(
            [n_main, m_main], dtype="float32"
        ):
            cls = Expected
            with R.dataflow():
                gv = cls.fused_add_exp_squeeze(x, R.const(1, "float32"))
                R.output(gv)
            return gv

    _check(Before, Expected)


def test_symbolic_shape_aware_fuse_2():
    n = T.dynamic("n", "int64")

    @I.ir_module
    class Before:
        @R.function
        def main(s: R.Shape([n])):
            with R.dataflow():
                lv0 = R.emit_te(topi.full, [n, n], "float32", 0)
                lv1 = R.emit_te(topi.trilu, lv0, tvm_tirx.tirx.const(1, "int32"), upper=True)
                gv = R.emit_te(topi.broadcast_to, lv1, [1, 1, n, n])
                R.output(gv)
            return gv

    n_fused_full_trilu_broadcast_to = T.dynamic("n", "int64")
    n_main = T.dynamic("n", "int64")

    @I.ir_module
    class Expected:
        @R.function(private=True)
        def fused_full_trilu_broadcast_to(
            s: R.Shape([n_fused_full_trilu_broadcast_to]),
        ) -> R.Tensor(
            [1, 1, n_fused_full_trilu_broadcast_to, n_fused_full_trilu_broadcast_to], "float32"
        ):
            R.func_attr({"Primitive": True})
            with R.dataflow():
                lv0 = R.emit_te(
                    topi.full,
                    [n_fused_full_trilu_broadcast_to, n_fused_full_trilu_broadcast_to],
                    "float32",
                    0,
                )
                lv1 = R.emit_te(topi.trilu, lv0, tvm_tirx.tirx.const(1, "int32"), upper=True)
                gv = R.emit_te(
                    topi.broadcast_to,
                    lv1,
                    [1, 1, n_fused_full_trilu_broadcast_to, n_fused_full_trilu_broadcast_to],
                )
                R.output(gv)
            return gv

        @R.function
        def main(s: R.Shape([n_main])) -> R.Tensor((1, 1, n_main, n_main), dtype="float32"):
            cls = Expected
            with R.dataflow():
                gv: R.Tensor([1, 1, n_main, n_main], "float32") = cls.fused_full_trilu_broadcast_to(
                    R.shape([n_main])
                )
                R.output(gv)
            return gv

    _check(Before, Expected)


def test_symbolic_prim_arg_after_tensor_arg():
    n = T.dynamic("n", "int64")

    @I.ir_module
    class Before:
        add_one_n = T.int64()

        @Ts.function(private=True)
        def add_one(
            x: T.Tensor((T.int64(1), add_one_n), "float32"),
            n: add_one_n,
            out: T.Tensor((T.int64(1), add_one_n), "float32"),
        ):
            T.func_attr({"op_pattern": 0, "tirx.noalias": True})

            for i in range(n):
                with Ts.sblock("add_one"):
                    vi = Ts.axis.spatial(n, i)
                    out[0, vi] = x[0, vi] + T.float32(1)

        exp_n = T.int64()

        @Ts.function(private=True)
        def exp(
            x: T.Tensor((T.int64(1), exp_n), "float32"),
            n: exp_n,
            out: T.Tensor((T.int64(1), exp_n), "float32"),
        ):
            T.func_attr({"op_pattern": 0, "tirx.noalias": True})

            for i in range(n):
                with Ts.sblock("exp"):
                    vi = Ts.axis.spatial(n, i)
                    out[0, vi] = T.exp(x[0, vi])

        @R.function
        def main(
            x: R.Tensor((1, n), dtype="float32"),
        ) -> R.Tensor((1, n), dtype="float32"):
            cls = Before
            with R.dataflow():
                lv = R.call_tir(
                    cls.add_one,
                    (x, n),
                    ty_args=[R.Tensor((1, n), dtype="float32")],
                )
                gv = R.call_tir(
                    cls.exp,
                    (lv, n),
                    ty_args=[R.Tensor((1, n), dtype="float32")],
                )
                R.output(gv)
            return gv

    mod = relax.transform.AnnotateTIROpPattern()(Before)
    mod = relax.transform.FuseOps()(mod)
    relax.analysis.check_well_formed(mod)

    fused = next(
        mod[global_var]
        for global_var in mod.get_global_vars()
        if global_var.name_hint.startswith("fused_")
    )
    assert len(fused.params) == 1
    assert not isinstance(fused.params[0].ty, tvm_tirx.ir.PrimType)
    assert fused.ret_ty.shape is not None

    mod = relax.transform.FuseTIR()(mod)
    relax.analysis.check_well_formed(mod)
    fused_tir = next(
        mod[global_var]
        for global_var in mod.get_global_vars()
        if global_var.name_hint.startswith("fused_")
    )
    tvm.s_tir.analysis.check_well_formed(fused_tir)


def test_symbolic_prim_arg_before_tensor_arg():
    n = T.dynamic("n", "int64")

    @I.ir_module
    class Before:
        add_one_n = T.int64()

        @Ts.function(private=True)
        def add_one(
            n: add_one_n,
            x: T.Tensor((T.int64(1), add_one_n), "float32"),
            out: T.Tensor((T.int64(1), add_one_n), "float32"),
        ):
            T.func_attr({"op_pattern": 0, "tirx.noalias": True})

            for i in range(n):
                with Ts.sblock("add_one"):
                    vi = Ts.axis.spatial(n, i)
                    out[0, vi] = x[0, vi] + T.float32(1)

        exp_n = T.int64()

        @Ts.function(private=True)
        def exp(
            n: exp_n,
            x: T.Tensor((T.int64(1), exp_n), "float32"),
            out: T.Tensor((T.int64(1), exp_n), "float32"),
        ):
            T.func_attr({"op_pattern": 0, "tirx.noalias": True})

            for i in range(n):
                with Ts.sblock("exp"):
                    vi = Ts.axis.spatial(n, i)
                    out[0, vi] = T.exp(x[0, vi])

        @R.function
        def main(
            x: R.Tensor((1, n), dtype="float32"),
        ) -> R.Tensor((1, n), dtype="float32"):
            cls = Before
            with R.dataflow():
                lv = R.call_tir(
                    cls.add_one,
                    (n, x),
                    ty_args=[R.Tensor((1, n), dtype="float32")],
                )
                gv = R.call_tir(
                    cls.exp,
                    (n, lv),
                    ty_args=[R.Tensor((1, n), dtype="float32")],
                )
                R.output(gv)
            return gv

    mod = relax.transform.AnnotateTIROpPattern()(Before)
    mod = relax.transform.FuseOps()(mod)
    relax.analysis.check_well_formed(mod)

    fused = next(
        mod[global_var]
        for global_var in mod.get_global_vars()
        if global_var.name_hint.startswith("fused_")
    )
    assert len(fused.params) == 1
    assert not isinstance(fused.params[0].ty, tvm_tirx.ir.PrimType)
    assert fused.ret_ty.shape is not None

    mod = relax.transform.FuseTIR()(mod)
    relax.analysis.check_well_formed(mod)
    fused_tir = next(
        mod[global_var]
        for global_var in mod.get_global_vars()
        if global_var.name_hint.startswith("fused_")
    )
    tvm.s_tir.analysis.check_well_formed(fused_tir)


def test_symbolic_prim_arg_reused_from_derived_tensor_shape():
    n = T.dynamic("n", "int64")

    @I.ir_module
    class Before:
        add_one_n = T.int64()

        @Ts.function(private=True)
        def add_one(
            x: T.Tensor(
                (T.int64(1), (add_one_n - T.int64(1)) // T.int64(4) + T.int64(1)), "float32"
            ),
            n: add_one_n,
            out: T.Tensor(
                (T.int64(1), (add_one_n - T.int64(1)) // T.int64(4) + T.int64(1)), "float32"
            ),
        ):
            T.func_attr({"op_pattern": 0, "tirx.noalias": True})

            for i in range((n - T.int64(1)) // T.int64(4) + T.int64(1)):
                with Ts.sblock("add_one"):
                    vi = Ts.axis.spatial((n - T.int64(1)) // T.int64(4) + T.int64(1), i)
                    out[0, vi] = x[0, vi] + T.float32(1)

        exp_n = T.int64()

        @Ts.function(private=True)
        def exp(
            x: T.Tensor((T.int64(1), (exp_n - T.int64(1)) // T.int64(4) + T.int64(1)), "float32"),
            n: exp_n,
            out: T.Tensor((T.int64(1), (exp_n - T.int64(1)) // T.int64(4) + T.int64(1)), "float32"),
        ):
            T.func_attr({"op_pattern": 0, "tirx.noalias": True})

            for i in range((n - T.int64(1)) // T.int64(4) + T.int64(1)):
                with Ts.sblock("exp"):
                    vi = Ts.axis.spatial((n - T.int64(1)) // T.int64(4) + T.int64(1), i)
                    out[0, vi] = T.exp(x[0, vi])

        @R.function
        def main(
            source: R.Tensor((n,), dtype="float32"),
            x: R.Tensor((1, (n - 1) // 4 + 1), dtype="float32"),
        ) -> R.Tensor((1, (n - 1) // 4 + 1), dtype="float32"):
            cls = Before
            with R.dataflow():
                lv = R.call_tir(
                    cls.add_one,
                    (x, n),
                    ty_args=[R.Tensor((1, (n - 1) // 4 + 1), dtype="float32")],
                )
                gv = R.call_tir(
                    cls.exp,
                    (lv, n),
                    ty_args=[R.Tensor((1, (n - 1) // 4 + 1), dtype="float32")],
                )
                R.output(gv)
            return gv

    mod = relax.transform.AnnotateTIROpPattern()(Before)
    mod = relax.transform.FuseOps()(mod)

    fused = next(
        mod[global_var]
        for global_var in mod.get_global_vars()
        if global_var.name_hint.startswith("fused_")
    )
    assert len(fused.params) == 2
    assert isinstance(fused.params[1].ty, relax.ShapeType)
    assert all(not isinstance(param.ty, tvm_tirx.ir.PrimType) for param in fused.params)

    mod = relax.transform.FuseTIR()(mod)
    fused_tir = next(
        mod[global_var]
        for global_var in mod.get_global_vars()
        if global_var.name_hint.startswith("fused_")
    )
    tvm.s_tir.analysis.check_well_formed(fused_tir)


def test_symbolic_prim_arg_not_bound_by_derived_tensor_shape():
    n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int64")

    @I.ir_module
    class Before:
        add_one_n = T.int64()

        @Ts.function(private=True)
        def add_one(
            x: T.Tensor((T.int64(1), add_one_n + T.int64(1)), "float32"),
            n: add_one_n,
            m: T.int64,
            out: T.Tensor((T.int64(1), add_one_n + T.int64(1)), "float32"),
        ):
            T.func_attr({"op_pattern": 0, "tirx.noalias": True})

            for i in range(n + T.int64(1)):
                with Ts.sblock("add_one"):
                    vi = Ts.axis.spatial(n + T.int64(1), i)
                    out[0, vi] = x[0, vi] + T.float32(1)

        exp_n = T.int64()

        @Ts.function(private=True)
        def exp(
            x: T.Tensor((T.int64(1), exp_n + T.int64(1)), "float32"),
            n: exp_n,
            m: T.int64,
            out: T.Tensor((T.int64(1), exp_n + T.int64(1)), "float32"),
        ):
            T.func_attr({"op_pattern": 0, "tirx.noalias": True})

            for i in range(n + T.int64(1)):
                with Ts.sblock("exp"):
                    vi = Ts.axis.spatial(n + T.int64(1), i)
                    out[0, vi] = T.exp(x[0, vi])

        @R.function
        def main(
            shape: R.Shape([n, m]),
            x: R.Tensor((1, n + 1), dtype="float32"),
        ) -> R.Tensor((1, n + 1), dtype="float32"):
            cls = Before
            with R.dataflow():
                lv = R.call_tir(
                    cls.add_one,
                    (x, n, m),
                    ty_args=[R.Tensor((1, n + 1), dtype="float32")],
                )
                gv = R.call_tir(
                    cls.exp,
                    (lv, n, m),
                    ty_args=[R.Tensor((1, n + 1), dtype="float32")],
                )
                R.output(gv)
            return gv

    mod = relax.transform.AnnotateTIROpPattern()(Before)
    mod = relax.transform.FuseOps()(mod)

    fused = next(
        mod[global_var]
        for global_var in mod.get_global_vars()
        if global_var.name_hint.startswith("fused_")
    )
    assert len(fused.params) == 3
    assert sum(isinstance(param.ty, relax.ShapeType) for param in fused.params) == 1
    assert sum(isinstance(param.ty, tvm_tirx.ir.PrimType) for param in fused.params) == 1


def test_primitive_call_arg_not_inlined():
    @I.ir_module
    class Before:
        @Ts.function(private=True)
        def add_scalar(
            x: T.Tensor((T.int64(4),), "int64"),
            value: T.int64,
            out: T.Tensor((T.int64(4),), "int64"),
        ):
            T.func_attr({"op_pattern": 0, "tirx.noalias": True})

            for i in range(4):
                with Ts.sblock("add_scalar"):
                    vi = Ts.axis.spatial(4, i)
                    out[vi] = x[vi] + value

        @Ts.function(private=True)
        def double(x: T.Tensor((T.int64(4),), "int64"), out: T.Tensor((T.int64(4),), "int64")):
            T.func_attr({"op_pattern": 0, "tirx.noalias": True})

            for i in range(4):
                with Ts.sblock("double"):
                    vi = Ts.axis.spatial(4, i)
                    out[vi] = x[vi] * T.int64(2)

        @R.function
        def main(x: R.Tensor((4,), dtype="int64")):
            cls = Before
            with R.dataflow():
                value: T.int64 = R.call_pure_packed(
                    "get_scalar", ty_args=tvm_tirx.ir.PrimType("int64")
                )
                lv = R.call_tir(
                    cls.add_scalar,
                    (x, value),
                    ty_args=[R.Tensor((4,), dtype="int64")],
                )
                gv = R.call_tir(
                    cls.double,
                    (lv,),
                    ty_args=[R.Tensor((4,), dtype="int64")],
                )
                R.output(gv, value)
            return gv, value

    mod = relax.transform.AnnotateTIROpPattern()(Before)
    mod = relax.transform.FuseOps()(mod)
    relax.analysis.check_well_formed(mod)

    fused = next(
        mod[global_var]
        for global_var in mod.get_global_vars()
        if global_var.name_hint.startswith("fused_")
    )
    assert len(fused.params) == 2
    assert sum(isinstance(param.ty, tvm_tirx.ir.PrimType) for param in fused.params) == 1

    mod = relax.transform.FuseTIR()(mod)
    relax.analysis.check_well_formed(mod)
    fused_tir = next(
        mod[global_var]
        for global_var in mod.get_global_vars()
        if global_var.name_hint.startswith("fused_")
    )
    tvm.s_tir.analysis.check_well_formed(fused_tir)


def test_primitive_call_arg_used_by_output_shape_not_inlined():
    @I.ir_module
    class Before:
        @Ts.function(private=True)
        def make(n: T.int64, out: T.Tensor((n,), "float32")):  # noqa: F821
            T.func_attr({"op_pattern": 0, "tirx.noalias": True})

            for i in range(n):
                with Ts.sblock("make"):
                    vi = Ts.axis.spatial(n, i)
                    out[vi] = T.float32(1)

        @Ts.function(private=True)
        def double(x: T.Tensor((n,), "float32"), n: T.int64, out: T.Tensor((n,), "float32")):  # noqa: F821
            T.func_attr({"op_pattern": 0, "tirx.noalias": True})

            for i in range(n):
                with Ts.sblock("double"):
                    vi = Ts.axis.spatial(n, i)
                    out[vi] = x[vi] * T.float32(2)

        @R.function(pure=False)
        def main():
            cls = Before
            n: T.int64 = R.call_packed("get_extent", ty_args=tvm_tirx.ir.PrimType("int64"))
            with R.dataflow():
                lv = R.call_tir(
                    cls.make,
                    (n,),
                    ty_args=[R.Tensor((n,), dtype="float32")],
                )
                gv = R.call_tir(
                    cls.double,
                    (lv, n),
                    ty_args=[R.Tensor((n,), dtype="float32")],
                )
                R.output(gv)
            return gv

    mod = relax.transform.AnnotateTIROpPattern()(Before)
    mod = relax.transform.FuseOps()(mod)
    relax.analysis.check_well_formed(mod)

    fused = next(
        mod[global_var]
        for global_var in mod.get_global_vars()
        if global_var.name_hint.startswith("fused_")
    )
    assert len(fused.params) == 1
    assert isinstance(fused.params[0].ty, tvm_tirx.ir.PrimType)

    packed_calls = []

    def collect_packed_calls(expr):
        if (
            isinstance(expr, relax.Call)
            and isinstance(expr.op, relax.ExternFunc)
            and expr.op.global_symbol == "get_extent"
        ):
            packed_calls.append(expr)

    relax.analysis.post_order_visit(mod["main"], collect_packed_calls)
    assert len(packed_calls) == 1

    packed_calls.clear()
    relax.analysis.post_order_visit(
        fused,
        collect_packed_calls,
    )
    assert not packed_calls


def test_symbolic_prim_arg_used_only_by_output_shape():
    n = T.dynamic("n", "int64")

    @I.ir_module
    class Before:
        make_n = T.int64()

        @Ts.function(private=True)
        def make(n: make_n, out: T.Tensor((make_n,), "float32")):
            T.func_attr({"op_pattern": 0, "tirx.noalias": True})

            for i in range(n):
                with Ts.sblock("make"):
                    vi = Ts.axis.spatial(n, i)
                    out[vi] = T.float32(1)

        double_n = T.int64()

        @Ts.function(private=True)
        def double(
            x: T.Tensor((double_n,), "float32"), n: double_n, out: T.Tensor((double_n,), "float32")
        ):
            T.func_attr({"op_pattern": 0, "tirx.noalias": True})

            for i in range(n):
                with Ts.sblock("double"):
                    vi = Ts.axis.spatial(n, i)
                    out[vi] = x[vi] * T.float32(2)

        @R.function
        def main(
            source: R.Tensor((n,), dtype="float32"),
        ) -> R.Tensor((n,), dtype="float32"):
            cls = Before
            with R.dataflow():
                lv = R.call_tir(
                    cls.make,
                    (n,),
                    ty_args=[R.Tensor((n,), dtype="float32")],
                )
                gv = R.call_tir(
                    cls.double,
                    (lv, n),
                    ty_args=[R.Tensor((n,), dtype="float32")],
                )
                R.output(gv)
            return gv

    mod = relax.transform.AnnotateTIROpPattern()(Before)
    mod = relax.transform.FuseOps()(mod)
    relax.analysis.check_well_formed(mod)

    fused = next(
        mod[global_var]
        for global_var in mod.get_global_vars()
        if global_var.name_hint.startswith("fused_")
    )
    assert len(fused.params) == 1
    assert isinstance(fused.params[0].ty, relax.ShapeType)
    assert fused.ret_ty.shape is not None

    mod = relax.transform.FuseTIR()(mod)
    relax.analysis.check_well_formed(mod)
    fused_tir = next(
        mod[global_var]
        for global_var in mod.get_global_vars()
        if global_var.name_hint.startswith("fused_")
    )
    tvm.s_tir.analysis.check_well_formed(fused_tir)


def test_shape_expr_arg():
    n = T.dynamic("n", "int64")

    @I.ir_module
    class Before:
        @R.function
        def main(s: R.Shape([n]), kv_cache: R.Any):
            with R.dataflow():
                lv0 = R.emit_te(topi.full, [n, n], "float32", 0)
                lv1 = R.emit_te(topi.trilu, lv0, tvm_tirx.tirx.const(1, "int32"), upper=True)
                lv2 = R.emit_te(topi.broadcast_to, lv1, [1, 1, n, n])
                gv = R.call_pure_packed(
                    "vm.builtin.attention_kv_cache_view",
                    kv_cache,
                    R.shape([1 + n, 32, 128]),
                    ty_args=(R.Tensor((1 + n, 32, 128), dtype="float32"),),
                )
                R.output(gv, lv2)
            return gv, lv2

    n_fused_full_trilu_broadcast_to = T.dynamic("n", "int64")
    n_main = T.dynamic("n", "int64")

    @I.ir_module
    class Expected:
        @R.function(private=True)
        def fused_full_trilu_broadcast_to(
            s: R.Shape([n_fused_full_trilu_broadcast_to]),
        ) -> R.Tensor(
            [1, 1, n_fused_full_trilu_broadcast_to, n_fused_full_trilu_broadcast_to], "float32"
        ):
            R.func_attr({"Primitive": True})
            with R.dataflow():
                lv0 = R.emit_te(
                    topi.full,
                    [n_fused_full_trilu_broadcast_to, n_fused_full_trilu_broadcast_to],
                    "float32",
                    0,
                )
                lv1 = R.emit_te(topi.trilu, lv0, tvm_tirx.tirx.const(1, "int32"), upper=True)
                gv = R.emit_te(
                    topi.broadcast_to,
                    lv1,
                    [1, 1, n_fused_full_trilu_broadcast_to, n_fused_full_trilu_broadcast_to],
                )
                R.output(gv)
            return gv

        @R.function
        def main(s: R.Shape([n_main]), kv_cache: R.Any):
            cls = Expected
            with R.dataflow():
                lv: R.Tensor([1, 1, n_main, n_main], "float32") = cls.fused_full_trilu_broadcast_to(
                    R.shape([n_main])
                )
                gv = R.call_pure_packed(
                    "vm.builtin.attention_kv_cache_view",
                    kv_cache,
                    R.shape([1 + n_main, 32, 128]),
                    ty_args=(R.Tensor((1 + n_main, 32, 128), dtype="float32"),),
                )
                R.output(gv, lv)
            return gv, lv

    _check(Before, Expected)


def test_skipping_match_cast():
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor((10, 20), dtype="float32")) -> R.Tensor(dtype="float32", ndim=2):
            with R.dataflow():
                lv: R.Tensor((m, n), dtype="float32") = R.match_cast(
                    A, R.Tensor((m, n), dtype="float32")
                )
                gv: R.Tensor((m, n), dtype="float32") = lv
                R.output(gv)
            return gv

    _check(Module, Module)


def test_skipping_primvalue():
    @I.ir_module
    class Module:
        @R.function
        def main(inp: R.Tensor((2, 2), dtype="float32")) -> R.Tensor((2, 2), dtype="float32"):
            with R.dataflow():
                lv = R.call_pure_packed(
                    "my_func1", inp, R.prim_value(0), ty_args=[R.Tensor((2, 2), dtype="float32")]
                )
                lv1 = R.call_pure_packed(
                    "my_func2", lv, R.str("str"), ty_args=[R.Tensor((2, 2), dtype="float32")]
                )
                gv = R.call_pure_packed(
                    "my_func3",
                    lv1,
                    R.dtype("float32"),
                    ty_args=[R.Tensor((2, 2), dtype="float32")],
                )
                R.output(gv)
            return gv

    _check(Module, Module)


def test_partially_used_tuple_param():
    @I.ir_module
    class Module:
        @R.function
        def main(
            x: R.Tuple(
                R.Tensor((2,), "float32"),
                R.Tensor((2,), "float32"),
                R.Tensor((2,), "float32"),
                R.Tensor((2,), "float32"),
                R.Tensor((2,), "float32"),
                R.Tensor((2,), "float32"),
            ),
        ):
            with R.dataflow():
                x0 = x[0]
                y0 = R.emit_te(topi.add, x0, R.const(1, "float32"))
                y1 = R.emit_te(topi.divide, y0, R.const(1, "float32"))
                gv = y1
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function(private=True)
        def fused_add_divide(
            x_0: R.Tensor((2,), dtype="float32"),
            param_0: R.Tensor((), dtype="float32"),
            param_1: R.Tensor((), dtype="float32"),
        ) -> R.Tensor((2,), dtype="float32"):
            R.func_attr({"Primitive": True})
            with R.dataflow():
                y0 = R.emit_te(topi.add, x_0, param_0)
                gv = R.emit_te(topi.divide, y0, param_1)
                R.output(gv)
            return gv

        @R.function
        def main(
            x: R.Tuple(
                R.Tensor((2,), dtype="float32"),
                R.Tensor((2,), dtype="float32"),
                R.Tensor((2,), dtype="float32"),
                R.Tensor((2,), dtype="float32"),
                R.Tensor((2,), dtype="float32"),
                R.Tensor((2,), dtype="float32"),
            ),
        ) -> R.Tensor((2,), dtype="float32"):
            cls = Expected
            with R.dataflow():
                lv: R.Tensor((2,), dtype="float32") = x[0]
                lv1: R.Tensor((2,), dtype="float32") = cls.fused_add_divide(
                    lv, R.const(1, "float32"), R.const(1, "float32")
                )
                gv: R.Tensor((2,), dtype="float32") = lv1
                R.output(gv)
            return gv

    _check(Module, Expected)


def test_call_tir_inplace():
    @I.ir_module
    class Module:
        @Ts.function(private=True)
        def add(
            A: T.Tensor((T.int64(10), T.int64(20)), "float32"),
            B: T.Tensor((), "float32"),
            Out: T.Tensor((T.int64(10), T.int64(20)), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for ax0, ax1 in T.grid(T.int64(10), T.int64(20)):
                with Ts.sblock("T_add"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(A[v_ax0, v_ax1], B[()])
                    Ts.writes(Out[v_ax0, v_ax1])
                    Out[v_ax0, v_ax1] = A[v_ax0, v_ax1] + B[()]

        @Ts.function(private=True)
        def exp_inplace(A: T.Tensor((T.int64(10), T.int64(20)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(10), T.int64(20)):
                with Ts.sblock("compute"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(A[v_i0, v_i1])
                    Ts.writes(A[v_i0, v_i1])
                    A[v_i0, v_i1] = T.exp(A[v_i0, v_i1])

        @Ts.function(private=True)
        def squeeze_inplace(A: T.Tensor((T.int64(10), T.int64(20)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for ax0, ax1 in T.grid(T.int64(10), T.int64(20)):
                with Ts.sblock("T_squeeze"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(A[v_ax0, v_ax1])
                    Ts.writes(A[v_ax0, v_ax1])
                    A[v_ax0, v_ax1] = A[v_ax0, v_ax1]

        @R.function
        def main(
            x: R.Tensor((10, 20), dtype="float32"), p0: R.Tensor((), dtype="float32")
        ) -> R.Tensor((10, 20), dtype="float32"):
            cls = Module
            with R.dataflow():
                lv = R.call_tir(
                    cls.add,
                    (x, p0),
                    ty_args=[R.Tensor((10, 20), dtype="float32")],
                )
                lv1 = R.call_tir_inplace(
                    cls.exp_inplace,
                    (lv,),
                    inplace_indices=[0],
                    ty_args=[R.Tensor((10, 20), dtype="float32")],
                )
                gv = R.call_tir_inplace(
                    cls.squeeze_inplace,
                    (lv1,),
                    inplace_indices=[0],
                    ty_args=[R.Tensor((10, 20), dtype="float32")],
                )
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def add(
            A: T.Tensor((T.int64(10), T.int64(20)), "float32"),
            B: T.Tensor((), "float32"),
            Out: T.Tensor((T.int64(10), T.int64(20)), "float32"),
        ):
            T.func_attr({"tirx.noalias": True, "op_pattern": 0})
            for ax0, ax1 in T.grid(T.int64(10), T.int64(20)):
                with Ts.sblock("T_add"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(A[v_ax0, v_ax1], B[()])
                    Ts.writes(Out[v_ax0, v_ax1])
                    Out[v_ax0, v_ax1] = A[v_ax0, v_ax1] + B[()]

        @Ts.function(private=True)
        def exp_inplace(A: T.Tensor((T.int64(10), T.int64(20)), "float32")):
            T.func_attr({"tirx.noalias": True, "op_pattern": 0})
            for i0, i1 in T.grid(T.int64(10), T.int64(20)):
                with Ts.sblock("compute"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(A[v_i0, v_i1])
                    Ts.writes(A[v_i0, v_i1])
                    A[v_i0, v_i1] = T.exp(A[v_i0, v_i1])

        @Ts.function(private=True)
        def squeeze_inplace(A: T.Tensor((T.int64(10), T.int64(20)), "float32")):
            T.func_attr({"tirx.noalias": True, "op_pattern": 0})
            for ax0, ax1 in T.grid(T.int64(10), T.int64(20)):
                with Ts.sblock("T_squeeze"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(A[v_ax0, v_ax1])
                    Ts.writes(A[v_ax0, v_ax1])
                    A[v_ax0, v_ax1] = A[v_ax0, v_ax1]

        @R.function(private=True)
        def fused_add_exp_inplace_squeeze_inplace(
            x: R.Tensor((10, 20), dtype="float32"), p0: R.Tensor((), dtype="float32")
        ) -> R.Tensor((10, 20), dtype="float32"):
            R.func_attr({"Primitive": True})
            cls = Expected
            with R.dataflow():
                lv = R.call_tir(
                    cls.add,
                    (x, p0),
                    ty_args=[R.Tensor((10, 20), dtype="float32")],
                )
                lv1 = R.call_tir_inplace(
                    cls.exp_inplace,
                    (lv,),
                    inplace_indices=[0],
                    ty_args=[R.Tensor((10, 20), dtype="float32")],
                )
                gv = R.call_tir_inplace(
                    cls.squeeze_inplace,
                    (lv1,),
                    inplace_indices=[0],
                    ty_args=[R.Tensor((10, 20), dtype="float32")],
                )
                R.output(gv)
            return gv

        @R.function
        def main(
            x: R.Tensor((10, 20), dtype="float32"), p0: R.Tensor((), dtype="float32")
        ) -> R.Tensor((10, 20), dtype="float32"):
            cls = Expected
            with R.dataflow():
                gv1: R.Tensor((10, 20), dtype="float32") = (
                    cls.fused_add_exp_inplace_squeeze_inplace(x, p0)
                )
                R.output(gv1)
            return gv1

    _check(Module, Expected)


def test_packed_params():
    # fmt: off
    @I.ir_module
    class Before:
        @Ts.function(private=True)
        def cast(lv: T.Tensor((T.int64(16), T.int64(16)), "float16"), compute: T.Tensor((T.int64(16), T.int64(16)), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0, i1 in T.grid(T.int64(16), T.int64(16)):
                with Ts.sblock("compute"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(lv[v_i0, v_i1])
                    Ts.writes(compute[v_i0, v_i1])
                    compute[v_i0, v_i1] = T.Cast("float32", lv[v_i0, v_i1])

        @Ts.function(private=True)
        def matmul(x: T.Tensor((T.int64(16), T.int64(16)), "float32"), lv2: T.Tensor((T.int64(16), T.int64(16)), "float32"), T_matmul: T.Tensor((T.int64(16), T.int64(16)), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1, k in T.grid(T.int64(16), T.int64(16), T.int64(16)):
                with Ts.sblock("T_matmul"):
                    v_ax0, v_ax1, v_k = Ts.axis.remap("SSR", [ax0, ax1, k])
                    Ts.reads(x[v_ax0, v_k], lv2[v_k, v_ax1])
                    Ts.writes(T_matmul[v_ax0, v_ax1])
                    with Ts.init():
                        T_matmul[v_ax0, v_ax1] = T.float32(0)
                    T_matmul[v_ax0, v_ax1] = T_matmul[v_ax0, v_ax1] + x[v_ax0, v_k] * lv2[v_k, v_ax1]

        @R.function
        def main(x: R.Tensor((16, 16), dtype="float32"), packed_params: R.Tuple(R.Tensor((16, 16), dtype="float16"), R.Tensor((16, 16), dtype="float16"))) -> R.Tensor((16, 16), dtype="float32"):
            R.func_attr({"num_input": 1})
            cls = Before
            with R.dataflow():
                lv: R.Tensor((16, 16), dtype="float16") = packed_params[0]
                lv1: R.Tensor((16, 16), dtype="float16") = packed_params[1]
                lv2 = R.call_tir(cls.cast, (lv,), ty_args=[R.Tensor((16, 16), dtype="float32")])
                lv3 = R.call_tir(cls.matmul, (x, lv2), ty_args=[R.Tensor((16, 16), dtype="float32")])
                lv4 = R.call_tir(cls.cast, (lv1,), ty_args=[R.Tensor((16, 16), dtype="float32")])
                lv5 = R.call_tir(cls.matmul, (lv3, lv4), ty_args=[R.Tensor((16, 16), dtype="float32")])
                gv: R.Tensor((16, 16), dtype="float32") = lv5
                R.output(gv)
            return gv
    # fmt: on

    Expected = Before
    _check(Before, Expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
