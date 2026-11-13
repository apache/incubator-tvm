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
# ruff: noqa: E501, E731, F841
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm.relax.transform import LegalizeOps
from tvm.s_tir import script as Ts
from tvm.testing import device_enabled
from tvm_tirx.ir import TupleType as _TupleType
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T

##################### Manipulation #####################


def test_broadcast_to():
    # fmt: off
    @tvm_tirx.script.ir_module
    class BroadcastTo:
        @R.function
        def main(x: R.Tensor((2, 1, 3), "float32")) -> R.Tensor((4, 2, 5, 3), "float32"):
            gv: R.Tensor((4, 2, 5, 3), "float32") = R.broadcast_to(x, (4, 2, 5, 3))
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 1, 3), "float32")) -> R.Tensor((4, 2, 5, 3), "float32"):
            gv = R.call_tir(Expected.broadcast_to, (x,), ty_args=[R.Tensor((4, 2, 5, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def broadcast_to(rxplaceholder: T.Tensor((T.int64(2), T.int64(1), T.int64(3)), "float32"), T_broadcast_to: T.Tensor((T.int64(4), T.int64(2), T.int64(5), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(2), T.int64(5), T.int64(3)):
                with Ts.sblock("T_broadcast_to"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[ax1, T.int64(0), ax3])
                    Ts.writes(T_broadcast_to[ax0, ax1, ax2, ax3])
                    T_broadcast_to[ax0, ax1, ax2, ax3] = rxplaceholder[ax1, T.int64(0), ax3]
    # fmt: on

    mod = LegalizeOps()(BroadcastTo)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_broadcast_to_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")
    c = T.dynamic("c", "int64")
    d = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class BroadcastTo:
        @R.function
        def main(dumb_param: R.Tensor((a, c)), x: R.Tensor((b, 1, d), "float32")) -> R.Tensor((a, b, c, d), "float32"):
            gv: R.Tensor((a, b, c, d), "float32") = R.broadcast_to(x, (a, b, c, d))
            return gv

    a_main = T.dynamic("a", "int64")
    b_main = T.dynamic("b", "int64")
    c_main = T.dynamic("c", "int64")
    d_main = T.dynamic("d", "int64")
    a_broadcast_to = T.dynamic("a", "int64")
    b_broadcast_to = T.dynamic("b", "int64")
    c_broadcast_to = T.dynamic("c", "int64")
    d_broadcast_to = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(dumb_param: R.Tensor((a_main, c_main)), x: R.Tensor((b_main, 1, d_main), "float32")) -> R.Tensor((a_main, b_main, c_main, d_main), "float32"):
            gv = R.call_tir(Expected.broadcast_to, (x,), ty_args=[R.Tensor((a_main, b_main, c_main, d_main), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def broadcast_to(rxplaceholder: T.Tensor([b_broadcast_to, T.int64(1), d_broadcast_to], dtype='float32'), T_broadcast_to: T.Tensor([a_broadcast_to, b_broadcast_to, c_broadcast_to, d_broadcast_to], dtype='float32')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1, i2, i3 in T.grid(a_broadcast_to, b_broadcast_to, c_broadcast_to, d_broadcast_to):
                with Ts.sblock("T_broadcast_to"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[ax1, T.int64(0), ax3])
                    Ts.writes(T_broadcast_to[ax0, ax1, ax2, ax3])
                    T_broadcast_to[ax0, ax1, ax2, ax3] = rxplaceholder[ax1, T.int64(0), ax3]
    # fmt: on

    mod = LegalizeOps()(BroadcastTo)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_concat():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Concat:
        @R.function
        def main(x1: R.Tensor((1, 2, 3), "float32"), x2: R.Tensor((1, 3, 3), "float32"), x3: R.Tensor((1, 4, 3), "float32")) -> R.Tensor((1, 9, 3), "float32"):
            gv: R.Tensor((1, 9, 3), "float32") = R.concat((x1, x2, x3), axis=1)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x1: R.Tensor((1, 2, 3), "float32"), x2: R.Tensor((1, 3, 3), "float32"), x3: R.Tensor((1, 4, 3), "float32")) -> R.Tensor((1, 9, 3), "float32"):
            gv = R.call_tir(Expected.concatenate, (x1, x2, x3), ty_args=[R.Tensor((1, 9, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def concatenate(rxplaceholder: T.Tensor((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Tensor((T.int64(1), T.int64(3), T.int64(3)), "float32"), rxplaceholder_2: T.Tensor((T.int64(1), T.int64(4), T.int64(3)), "float32"), T_concat: T.Tensor((T.int64(1), T.int64(9), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2 in T.grid(T.int64(1), T.int64(9), T.int64(3)):
                with Ts.sblock("T_concat"):
                    ax0, ax1, ax2 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(rxplaceholder_2[ax0, ax1 - T.int64(5), ax2], rxplaceholder_1[ax0, ax1 - T.int64(2), ax2], rxplaceholder[ax0, ax1, ax2])
                    Ts.writes(T_concat[ax0, ax1, ax2])
                    T_concat[ax0, ax1, ax2] = T.if_then_else(T.int64(5) <= ax1, rxplaceholder_2[ax0, ax1 - T.int64(5), ax2], T.if_then_else(T.int64(2) <= ax1, rxplaceholder_1[ax0, ax1 - T.int64(2), ax2], rxplaceholder[ax0, ax1, ax2]))
    # fmt: on

    mod = LegalizeOps()(Concat)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_concat_input_tuple_var():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Concat:
        @R.function
        def main(t: R.Tuple(R.Tensor((3, 4), "float32"), R.Tensor((3, 5), "float32"))) -> R.Tensor((3, 9), "float32"):
            gv: R.Tensor((3, 9), "float32") = R.concat(t, axis=1)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(t: R.Tuple(R.Tensor((3, 4), "float32"), R.Tensor((3, 5), "float32"))) -> R.Tensor((3, 9), "float32"):
            gv: R.Tensor((3, 4), dtype="float32") = t[0]
            gv1: R.Tensor((3, 5), dtype="float32") = t[1]
            gv2 = R.call_tir(Expected.concatenate, (gv, gv1), ty_args=[R.Tensor((3, 9), dtype="float32")])
            return gv2

        @Ts.function(private=True)
        def concatenate(rxplaceholder: T.Tensor((T.int64(3), T.int64(4)), "float32"), rxplaceholder_1: T.Tensor((T.int64(3), T.int64(5)), "float32"), T_concat: T.Tensor((T.int64(3), T.int64(9)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(3), T.int64(9)):
                with Ts.sblock("T_concat"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder_1[ax0, ax1 - T.int64(4)], rxplaceholder[ax0, ax1])
                    Ts.writes(T_concat[ax0, ax1])
                    T_concat[ax0, ax1] = T.if_then_else(T.int64(4) <= ax1, rxplaceholder_1[ax0, ax1 - T.int64(4)], rxplaceholder[ax0, ax1])
    # fmt: on

    mod = LegalizeOps()(Concat)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_concat_input_tuple_var_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    b0 = T.dynamic("b0", "int64")
    b1 = T.dynamic("b1", "int64")
    b2 = T.dynamic("b2", "int64")

    @tvm_tirx.script.ir_module
    class Concat:
        @R.function
        def main(t: R.Tuple(R.Tensor((a, b0), "float32"), R.Tensor((a, b1), "float32"), R.Tensor((a, b2), "float32"))) -> R.Tensor((a, b0 + b1 + b2), "float32"):
            gv: R.Tensor((a, b0 + b1 + b2), "float32") = R.concat(t, axis=1)
            return gv

    a_main = T.dynamic("a", "int64")
    b0_main = T.dynamic("b0", "int64")
    b1_main = T.dynamic("b1", "int64")
    b2_main = T.dynamic("b2", "int64")
    a_concatenate = T.dynamic("a", "int64")
    b0_concatenate = T.dynamic("b0", "int64")
    b1_concatenate = T.dynamic("b1", "int64")
    b2_concatenate = T.dynamic("b2", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(t: R.Tuple(R.Tensor((a_main, b0_main), "float32"), R.Tensor((a_main, b1_main), "float32"), R.Tensor((a_main, b2_main), "float32"))) -> R.Tensor((a_main, b0_main + b1_main + b2_main), "float32"):
            gv: R.Tensor((a_main, b0_main), dtype="float32") = t[0]
            gv1: R.Tensor((a_main, b1_main), dtype="float32") = t[1]
            gv2: R.Tensor((a_main, b2_main), dtype="float32") = t[2]
            gv3 = R.call_tir(Expected.concatenate, (gv, gv1, gv2), ty_args=[R.Tensor((a_main, ((b0_main + b1_main) + b2_main)), dtype="float32")])
            return gv3

        @Ts.function(private=True)
        def concatenate(rxplaceholder: T.Tensor([a_concatenate, b0_concatenate], dtype='float32'), rxplaceholder_1: T.Tensor([a_concatenate, b1_concatenate], dtype='float32'), rxplaceholder_2: T.Tensor([a_concatenate, b2_concatenate], dtype='float32'), T_concat: T.Tensor([a_concatenate, b0_concatenate + b1_concatenate + b2_concatenate], dtype='float32')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1 in T.grid(a_concatenate, b0_concatenate + b1_concatenate + b2_concatenate):
                with Ts.sblock("T_concat"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder_2[ax0, ax1 - b0_concatenate - b1_concatenate], rxplaceholder_1[ax0, ax1 - b0_concatenate], rxplaceholder[ax0, ax1])
                    Ts.writes(T_concat[ax0, ax1])
                    T_concat[ax0, ax1] = T.if_then_else(T.int64(0) <= ax1 - b0_concatenate - b1_concatenate, rxplaceholder_2[ax0, ax1 - b0_concatenate - b1_concatenate], T.if_then_else(T.int64(0) <= ax1 - b0_concatenate, rxplaceholder_1[ax0, ax1 - b0_concatenate], rxplaceholder[ax0, ax1]))
    # fmt: on

    mod = LegalizeOps()(Concat)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_expand_dims():
    # fmt: off
    @tvm_tirx.script.ir_module
    class ExpandDims:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor((2, 1, 1, 1, 3, 1, 4, 1), "float32"):
            gv: R.Tensor((2, 1, 1, 1, 3, 1, 4, 1), "float32") = R.expand_dims(x, axis=[-1, 1, -6, 3, 5])
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor((2, 1, 1, 1, 3, 1, 4, 1), "float32"):
            gv = R.call_tir(Expected.expand_dims, (x,), ty_args=[R.Tensor((2, 1, 1, 1, 3, 1, 4, 1), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def expand_dims(rxplaceholder: T.Tensor((T.int64(2), T.int64(3), T.int64(4)), "float32"), expand_dims: T.Tensor((T.int64(2), T.int64(1), T.int64(1), T.int64(1), T.int64(3), T.int64(1), T.int64(4), T.int64(1)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2, i3, i4, i5, i6, i7 in T.grid(T.int64(2), T.int64(1), T.int64(1), T.int64(1), T.int64(3), T.int64(1), T.int64(4), T.int64(1)):
                with Ts.sblock("expand_dims"):
                    i0_1, i1_1, i2_1, i3_1, i4_1, i5_1, i6_1, i7_1 = Ts.axis.remap("SSSSSSSS", [i0, i1, i2, i3, i4, i5, i6, i7])
                    Ts.reads(rxplaceholder[i0_1, i4_1, i6_1])
                    Ts.writes(expand_dims[i0_1, i1_1, i2_1, i3_1, i4_1, i5_1, i6_1, i7_1])
                    expand_dims[i0_1, i1_1, i2_1, i3_1, i4_1, i5_1, i6_1, i7_1] = rxplaceholder[i0_1, i4_1, i6_1]
    # fmt: on

    mod = LegalizeOps()(ExpandDims)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_expand_dims_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")
    c = T.dynamic("c", "int64")

    @tvm_tirx.script.ir_module
    class ExpandDims:
        @R.function
        def main(x: R.Tensor((a, b, c), "float32")) -> R.Tensor((a, 1, b, 1, c, 1), "float32"):
            gv: R.Tensor((a, 1, b, 1, c, 1), "float32") = R.expand_dims(x, axis=[1, 3, 5])
            return gv

    a_main = T.dynamic("a", "int64")
    b_main = T.dynamic("b", "int64")
    c_main = T.dynamic("c", "int64")
    a_expand_dims = T.dynamic("a", "int64")
    b_expand_dims = T.dynamic("b", "int64")
    c_expand_dims = T.dynamic("c", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((a_main, b_main, c_main), "float32")) -> R.Tensor((a_main, 1, b_main, 1, c_main, 1), "float32"):
            gv = R.call_tir(Expected.expand_dims, (x,), ty_args=[R.Tensor((a_main, 1, b_main, 1, c_main, 1), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def expand_dims(rxplaceholder: T.Tensor([a_expand_dims, b_expand_dims, c_expand_dims], dtype='float32'), expand_dims: T.Tensor([a_expand_dims, T.int64(1), b_expand_dims, T.int64(1), c_expand_dims, T.int64(1)], dtype='float32')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1, i2, i3, i4, i5 in T.grid(a_expand_dims, T.int64(1), b_expand_dims, T.int64(1), c_expand_dims, T.int64(1)):
                with Ts.sblock("expand_dims"):
                    i0_1, i1_1, i2_1, i3_1, i4_1, i5_1 = Ts.axis.remap("SSSSSS", [i0, i1, i2, i3, i4, i5])
                    Ts.reads(rxplaceholder[i0_1, i2_1, i4_1])
                    Ts.writes(expand_dims[i0_1, i1_1, i2_1, i3_1, i4_1, i5_1])
                    expand_dims[i0_1, i1_1, i2_1, i3_1, i4_1, i5_1] = rxplaceholder[i0_1, i2_1, i4_1]
    # fmt: on

    mod = LegalizeOps()(ExpandDims)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_flatten():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Flatten:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor((24,), "float32"):
            gv: R.Tensor((24,), "float32") = R.flatten(x)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor((24,), "float32"):
            gv = R.call_tir(Expected.reshape, (x,), ty_args=[R.Tensor((24,), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def reshape(rxplaceholder: T.Tensor((T.int64(2), T.int64(3), T.int64(4)), "float32"), T_reshape: T.Tensor(T.int64(24), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0 in T.serial(T.int64(24)):
                with Ts.sblock("T_reshape"):
                    ax0 = Ts.axis.spatial(T.int64(24), i0)
                    Ts.reads(rxplaceholder[ax0 % T.int64(24) // T.int64(12), ax0 % T.int64(12) // T.int64(4), ax0 % T.int64(4)])
                    Ts.writes(T_reshape[ax0])
                    T_reshape[ax0] = rxplaceholder[ax0 % T.int64(24) // T.int64(12), ax0 % T.int64(12) // T.int64(4), ax0 % T.int64(4)]
    # fmt: on

    mod = LegalizeOps()(Flatten)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_flatten_zero_rank():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Flatten:
        @R.function
        def main(x: R.Tensor((), "float32")) -> R.Tensor((1,), "float32"):
            gv: R.Tensor((1,), "float32") = R.flatten(x)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((), "float32")) -> R.Tensor((1,), "float32"):
            gv = R.call_tir(Expected.reshape, (x,), ty_args=[R.Tensor((1,), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def reshape(rxplaceholder: T.Tensor((), "float32"), T_reshape: T.Tensor(T.int64(1), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0 in T.serial(T.int64(1)):
                with Ts.sblock("T_reshape"):
                    ax0 = Ts.axis.spatial(T.int64(1), i0)
                    Ts.reads(rxplaceholder[()])
                    Ts.writes(T_reshape[ax0])
                    T_reshape[ax0] = rxplaceholder[()]
    # fmt: on

    mod = LegalizeOps()(Flatten)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_flatten_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")
    c = T.dynamic("c", "int64")

    @tvm_tirx.script.ir_module
    class Flatten:
        @R.function
        def main(x: R.Tensor((a, b, c), "float32")) -> R.Tensor((a * b * c,), "float32"):
            gv: R.Tensor((a * b * c,), "float32") = R.flatten(x)
            return gv

    a_main = T.dynamic("a", "int64")
    b_main = T.dynamic("b", "int64")
    c_main = T.dynamic("c", "int64")
    a_reshape = T.dynamic("a", "int64")
    b_reshape = T.dynamic("b", "int64")
    c_reshape = T.dynamic("c", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((a_main, b_main, c_main), "float32")) -> R.Tensor((a_main * b_main * c_main,), "float32"):
            gv = R.call_tir(Expected.reshape, (x,), ty_args=[R.Tensor((((a_main * b_main) * c_main),), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def reshape(rxplaceholder: T.Tensor([a_reshape, b_reshape, c_reshape], dtype='float32'), T_reshape: T.Tensor([a_reshape * b_reshape * c_reshape], dtype='float32')):
            T.func_attr({"tirx.noalias": True})

            for i0 in T.serial(a_reshape * b_reshape * c_reshape):
                with Ts.sblock("T_reshape"):
                    ax0 = Ts.axis.spatial(a_reshape * b_reshape * c_reshape, i0)
                    Ts.reads(rxplaceholder[ax0 // c_reshape // b_reshape % a_reshape, ax0 // c_reshape % b_reshape, ax0 % c_reshape])
                    Ts.writes(T_reshape[ax0])
                    T_reshape[ax0] = rxplaceholder[ax0 // c_reshape // b_reshape % a_reshape, ax0 // c_reshape % b_reshape, ax0 % c_reshape]
    # fmt: on

    mod = LegalizeOps()(Flatten)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_permute_dims():
    # fmt: off
    @tvm_tirx.script.ir_module
    class PermuteDims:
        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor((2, 4, 3, 1), "float32"):
            gv: R.Tensor((2, 4, 3, 1), "float32") = R.permute_dims(x, axes=[1, -1, 2, -4])
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor((2, 4, 3, 1), "float32"):
            gv = R.call_tir(Expected.transpose, (x,), ty_args=[R.Tensor((2, 4, 3, 1), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def transpose(rxplaceholder: T.Tensor((T.int64(1), T.int64(2), T.int64(3), T.int64(4)), "float32"), T_transpose: T.Tensor((T.int64(2), T.int64(4), T.int64(3), T.int64(1)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(4), T.int64(3), T.int64(1)):
                with Ts.sblock("T_transpose"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[ax3, ax0, ax2, ax1])
                    Ts.writes(T_transpose[ax0, ax1, ax2, ax3])
                    T_transpose[ax0, ax1, ax2, ax3] = rxplaceholder[ax3, ax0, ax2, ax1]
    # fmt: on

    mod = LegalizeOps()(PermuteDims)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_permute_dims_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")
    c = T.dynamic("c", "int64")
    d = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class PermuteDims:
        @R.function
        def main(x: R.Tensor((a, b, c, d), "float32")) -> R.Tensor((b, d, c, a), "float32"):
            gv: R.Tensor((b, d, c, a), "float32") = R.permute_dims(x, axes=[1, -1, 2, -4])
            return gv

    b_main = T.dynamic("b", "int64")
    d_main = T.dynamic("d", "int64")
    c_main = T.dynamic("c", "int64")
    a_main = T.dynamic("a", "int64")
    a_transpose = T.dynamic("a", "int64")
    b_transpose = T.dynamic("b", "int64")
    c_transpose = T.dynamic("c", "int64")
    d_transpose = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((a_main, b_main, c_main, d_main), dtype="float32")) -> R.Tensor((b_main, d_main, c_main, a_main), dtype="float32"):
            gv = R.call_tir(Expected.transpose, (x,), ty_args=[R.Tensor((b_main, d_main, c_main, a_main), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def transpose(rxplaceholder: T.Tensor([a_transpose, b_transpose, c_transpose, d_transpose], dtype='float32'), T_transpose: T.Tensor([b_transpose, d_transpose, c_transpose, a_transpose], dtype='float32')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1, i2, i3 in T.grid(b_transpose, d_transpose, c_transpose, a_transpose):
                with Ts.sblock("T_transpose"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[ax3, ax0, ax2, ax1])
                    Ts.writes(T_transpose[ax0, ax1, ax2, ax3])
                    T_transpose[ax0, ax1, ax2, ax3] = rxplaceholder[ax3, ax0, ax2, ax1]
    # fmt: on

    mod = LegalizeOps()(PermuteDims)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_reshape():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Reshape:
        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor((8, 3), "float32"):
            gv: R.Tensor((8, 3), "float32") = R.reshape(x, (8, 3))
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor((8, 3), "float32"):
            gv = R.call_tir(Expected.reshape, (x,), ty_args=[R.Tensor((8, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def reshape(rxplaceholder: T.Tensor((T.int64(1), T.int64(2), T.int64(3), T.int64(4)), "float32"), T_reshape: T.Tensor((T.int64(8), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(8), T.int64(3)):
                with Ts.sblock("T_reshape"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[T.int64(0), (ax0 * T.int64(3) + ax1) % T.int64(24) // T.int64(12), (ax0 * T.int64(3) + ax1) % T.int64(12) // T.int64(4), (ax0 * T.int64(3) + ax1) % T.int64(4)])
                    Ts.writes(T_reshape[ax0, ax1])
                    T_reshape[ax0, ax1] = rxplaceholder[T.int64(0), (ax0 * T.int64(3) + ax1) % T.int64(24) // T.int64(12), (ax0 * T.int64(3) + ax1) % T.int64(12) // T.int64(4), (ax0 * T.int64(3) + ax1) % T.int64(4)]
    # fmt: on

    mod = LegalizeOps()(Reshape)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)

    # fmt: off
    # ShapeExpr might be produced by shape computation
    @tvm_tirx.script.ir_module
    class Reshape2:
        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor((8, 3), "float32"):
            lv: R.Shape((8, 3)) = R.shape((8, 3))
            gv: R.Tensor((8, 3), "float32") = R.reshape(x, lv)
            return gv

    # After lowering, redundant var might be removed by later dead code elimination
    @tvm_tirx.script.ir_module
    class Expected2:
        @Ts.function(private=True)
        def reshape(
            rxplaceholder: T.Tensor((T.int64(1), T.int64(2), T.int64(3), T.int64(4)), "float32"),
            T_reshape: T.Tensor((T.int64(8), T.int64(3)), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1 in T.grid(T.int64(8), T.int64(3)):
                with Ts.sblock("T_reshape"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(
                        rxplaceholder[
                            T.int64(0),
                            (v_ax0 * T.int64(3) + v_ax1) % T.int64(24) // T.int64(12),
                            (v_ax0 * T.int64(3) + v_ax1) % T.int64(12) // T.int64(4),
                            (v_ax0 * T.int64(3) + v_ax1) % T.int64(4),
                        ]
                    )
                    Ts.writes(T_reshape[v_ax0, v_ax1])
                    T_reshape[v_ax0, v_ax1] = rxplaceholder[
                        T.int64(0),
                        (v_ax0 * T.int64(3) + v_ax1) % T.int64(24) // T.int64(12),
                        (v_ax0 * T.int64(3) + v_ax1) % T.int64(12) // T.int64(4),
                        (v_ax0 * T.int64(3) + v_ax1) % T.int64(4),
                    ]

        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tensor((8, 3), dtype="float32"):
            lv: R.Shape((8, 3)) = R.shape((8, 3))
            gv = R.call_tir(Expected2.reshape, (x,), ty_args=[R.Tensor((8, 3), dtype="float32")])
            return gv
    # fmt: on

    mod2 = LegalizeOps()(Reshape2)
    tvm_tirx.ir.assert_structural_equal(mod2, Expected2)


def test_reshape_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")

    @tvm_tirx.script.ir_module
    class Reshape:
        @R.function
        def main(x: R.Tensor((a, b), "float32")) -> R.Tensor((a // 2, b * 2), "float32"):
            gv: R.Tensor((a // 2, b * 2), "float32") = R.reshape(x, (a // 2, b * 2))
            return gv

    a_main = T.dynamic("a", "int64")
    b_main = T.dynamic("b", "int64")
    a_reshape = T.dynamic("a", "int64")
    b_reshape = T.dynamic("b", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((a_main, b_main), "float32")) -> R.Tensor((a_main // 2, b_main * 2), "float32"):
            gv = R.call_tir(Expected.reshape, (x,), ty_args=[R.Tensor(((a_main // 2), (b_main * 2)), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def reshape(rxplaceholder: T.Tensor([a_reshape, b_reshape], dtype='float32'), T_reshape: T.Tensor([a_reshape // T.int64(2), b_reshape * T.int64(2)], dtype='float32')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1 in T.grid(a_reshape // T.int64(2), b_reshape * T.int64(2)):
                with Ts.sblock("T_reshape"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[(ax0 * b_reshape * T.int64(2) + ax1) // b_reshape % a_reshape, (ax0 * b_reshape * T.int64(2) + ax1) % b_reshape])
                    Ts.writes(T_reshape[ax0, ax1])
                    T_reshape[ax0, ax1] = rxplaceholder[(ax0 * b_reshape * T.int64(2) + ax1) // b_reshape % a_reshape, (ax0 * b_reshape * T.int64(2) + ax1) % b_reshape]
    # fmt: on

    mod = LegalizeOps()(Reshape)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)

    # ShapeExpr might be produced by shape computation
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")

    @tvm_tirx.script.ir_module
    class Reshape2:
        @R.function
        def main(x: R.Tensor((a, b), "float32")) -> R.Tensor((a // 2, b * 2), "float32"):
            lv: R.Shape((a // 2, b * 2)) = R.shape((a // 2, b * 2))
            gv: R.Tensor((a // 2, b * 2), "float32") = R.reshape(x, lv)
            return gv

    # After lowering, redundant var might be removed by later dead code elimination
    a_main = T.dynamic("a", "int64")
    b_main = T.dynamic("b", "int64")
    a_reshape = T.dynamic("a", "int64")
    b_reshape = T.dynamic("b", "int64")

    @tvm_tirx.script.ir_module
    class Expected2:
        @R.function
        def main(x: R.Tensor((a_main, b_main), "float32")) -> R.Tensor(
            (a_main // 2, b_main * 2), "float32"
        ):
            lv: R.Shape((a_main // 2, b_main * 2)) = R.shape((a_main // 2, b_main * 2))
            gv = R.call_tir(
                Expected2.reshape,
                (x,),
                ty_args=[R.Tensor(((a_main // 2), (b_main * 2)), dtype="float32")],
            )
            return gv

        @Ts.function(private=True)
        def reshape(
            rxplaceholder: T.Tensor([a_reshape, b_reshape], dtype="float32"),
            T_reshape: T.Tensor([a_reshape // T.int64(2), b_reshape * T.int64(2)], dtype="float32"),
        ):
            T.func_attr({"tirx.noalias": True})

            for i0, i1 in T.grid(a_reshape // T.int64(2), b_reshape * T.int64(2)):
                with Ts.sblock("T_reshape"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(
                        rxplaceholder[
                            (ax0 * b_reshape * T.int64(2) + ax1) // b_reshape % a_reshape,
                            (ax0 * b_reshape * T.int64(2) + ax1) % b_reshape,
                        ]
                    )
                    Ts.writes(T_reshape[ax0, ax1])
                    T_reshape[ax0, ax1] = rxplaceholder[
                        (ax0 * b_reshape * T.int64(2) + ax1) // b_reshape % a_reshape,
                        (ax0 * b_reshape * T.int64(2) + ax1) % b_reshape,
                    ]

    mod2 = LegalizeOps()(Reshape2)
    tvm_tirx.ir.assert_structural_equal(mod2, Expected2)

    # ShapeExpr might be produced by shape computation
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")

    @I.ir_module
    class Reshape3:
        @R.function
        def main(x: R.Tensor((10, b), "float32")) -> R.Tensor((5, b * 2), "float32"):
            lv: R.Shape((5, b * 2)) = R.shape((5, b * 2))
            gv: R.Tensor((5, b * 2), "float32") = R.reshape(x, lv)
            return gv

    # After lowering, redundant var might be removed by later dead code elimination
    b_reshape = T.dynamic("b", "int64")
    b_main = T.dynamic("b", "int64")

    @I.ir_module
    class Expected3:
        @Ts.function(private=True)
        def reshape(
            rxplaceholder: T.Tensor((T.int64(10), b_reshape), "float32"),
            T_reshape: T.Tensor((T.int64(5), b_reshape * T.int64(2)), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            for ax0, ax1 in T.grid(T.int64(5), b_reshape * T.int64(2)):
                with Ts.sblock("T_reshape"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(
                        rxplaceholder[
                            (v_ax0 * b_reshape * T.int64(2) + v_ax1) // b_reshape % T.int64(10),
                            (v_ax0 * b_reshape * T.int64(2) + v_ax1) % b_reshape,
                        ]
                    )
                    Ts.writes(T_reshape[v_ax0, v_ax1])
                    T_reshape[v_ax0, v_ax1] = rxplaceholder[
                        (v_ax0 * b_reshape * T.int64(2) + v_ax1) // b_reshape % T.int64(10),
                        (v_ax0 * b_reshape * T.int64(2) + v_ax1) % b_reshape,
                    ]

        @R.function
        def main(
            x: R.Tensor((10, b_main), dtype="float32"),
        ) -> R.Tensor((5, b_main * 2), dtype="float32"):
            lv: R.Shape([5, b_main * 2]) = R.shape([5, b_main * 2])
            gv = R.call_tir(
                Expected3.reshape, (x,), ty_args=[R.Tensor((5, b_main * 2), dtype="float32")]
            )
            return gv

    mod3 = LegalizeOps()(Reshape3)
    tvm_tirx.ir.assert_structural_equal(mod3, Expected3)


def test_data_dependent_reshape():
    # fmt: off
    @tvm_tirx.script.ir_module
    class DDReshape:
        @R.function
        def main(
            x: R.Tensor([2], dtype="int64"),
            y: R.Tensor([16],dtype='float32'),
        ):
            lv: R.Shape(ndim=2) = R.tensor_to_shape(x)
            gv = R.reshape(y, lv)
            return gv
    # fmt: on

    relax.analysis.check_well_formed(DDReshape)
    mod = relax.transform.DecomposeOpsForInference()(DDReshape)
    out_mod = relax.transform.LegalizeOps()(mod)

    # fmt: off
    M_main = T.dynamic("M", "int64")
    N_main = T.dynamic("N", "int64")
    M_reshape = T.dynamic("M", "int64")
    N_reshape = T.dynamic("N", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(
                x: R.Tensor([2], dtype="int64"),
                y: R.Tensor([16],dtype="float32"),
        ) -> R.Tensor(ndim=2, dtype="float32"):
            gv = R.call_pure_packed("vm.builtin.tensor_to_shape", x, ty_args=(R.Shape(ndim=2),))
            _ = R.match_cast(gv, R.Shape([M_main,N_main]))
            _ = R.shape([M_main,N_main])
            gv_1 = R.call_tir(Expected.reshape, (y,), ty_args=[R.Tensor([M_main,N_main], dtype="float32")])
            return gv_1

        @Ts.function(private=True)
        def reshape(
            rxplaceholder: T.Tensor(T.int64(16), "float32"),
            T_reshape: T.Tensor([M_reshape, N_reshape], 'float32'),
        ):
            T.func_attr({"tirx.noalias": True})

            for i,j in T.grid(M_reshape,N_reshape):
                with Ts.sblock("T_reshape"):
                    vi,vj = Ts.axis.remap('SS',[i,j])
                    Ts.reads(rxplaceholder[(vi*N_reshape + vj) % 16])
                    Ts.writes(T_reshape[vi,vj])
                    T_reshape[vi,vj] = rxplaceholder[(vi*N_reshape + vj) % 16]

    # fmt: on
    tvm_tirx.ir.assert_structural_equal(out_mod, Expected)


def test_split_by_indices():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Split:
        @R.function
        def main(x: R.Tensor((2, 10, 4), "float32")) -> R.Tuple([R.Tensor((2, 3, 4), "float32"), R.Tensor((2, 4, 4), "float32"), R.Tensor((2, 3, 4), "float32")]):
            gv: R.Tuple([R.Tensor((2, 3, 4), "float32"), R.Tensor((2, 4, 4), "float32"), R.Tensor((2, 3, 4), "float32")]) = R.split(x, [3, 7], axis=1)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 10, 4), "float32")) -> R.Tuple([R.Tensor((2, 3, 4), "float32"), R.Tensor((2, 4, 4), "float32"), R.Tensor((2, 3, 4), "float32")]):
            gv = R.call_tir(Expected.split, (x,), ty_args=[_TupleType([R.Tensor((2, 3, 4), "float32"), R.Tensor((2, 4, 4), "float32"), R.Tensor((2, 3, 4), "float32")])])
            return gv

        @Ts.function(private=True)
        def split(rxplaceholder: T.Tensor((T.int64(2), T.int64(10), T.int64(4)), "float32"), T_split: T.Tensor((T.int64(2), T.int64(3), T.int64(4)), "float32"), T_split_1: T.Tensor((T.int64(2), T.int64(4), T.int64(4)), "float32"), T_split_2: T.Tensor((T.int64(2), T.int64(3), T.int64(4)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(3), T.int64(4)):
                with Ts.sblock("T_split"):
                    ax0, ax1, ax2 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(rxplaceholder[ax0, ax1, ax2])
                    Ts.writes(T_split[ax0, ax1, ax2])
                    T_split[ax0, ax1, ax2] = rxplaceholder[ax0, ax1, ax2]
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(4), T.int64(4)):
                with Ts.sblock("T_split_1"):
                    ax0, ax1, ax2 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(rxplaceholder[ax0, ax1 + T.int64(3), ax2])
                    Ts.writes(T_split_1[ax0, ax1, ax2])
                    T_split_1[ax0, ax1, ax2] = rxplaceholder[ax0, ax1 + T.int64(3), ax2]
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(3), T.int64(4)):
                with Ts.sblock("T_split_2"):
                    ax0, ax1, ax2 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(rxplaceholder[ax0, ax1 + T.int64(7), ax2])
                    Ts.writes(T_split_2[ax0, ax1, ax2])
                    T_split_2[ax0, ax1, ax2] = rxplaceholder[ax0, ax1 + T.int64(7), ax2]
    # fmt: on

    mod = LegalizeOps()(Split)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_split_by_indices_n_section_indivisible():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Split:
        @R.function
        def main(x: R.Tensor((2, 10, 4), "float32")) -> R.Tuple([R.Tensor((2, 4, 4), "float32"), R.Tensor((2, 4, 4), "float32"), R.Tensor((2, 2, 4), "float32")]):
            gv: R.Tuple([R.Tensor((2, 4, 4), "float32"), R.Tensor((2, 4, 4), "float32"), R.Tensor((2, 2, 4), "float32")]) = R.split(x, indices_or_sections=3, axis=1)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 10, 4), "float32")) -> R.Tuple([R.Tensor((2, 4, 4), "float32"), R.Tensor((2, 4, 4), "float32"), R.Tensor((2, 2, 4), "float32")]):
            gv = R.call_tir(Expected.split, (x,), ty_args=[_TupleType([R.Tensor((2, 4, 4), "float32"), R.Tensor((2, 4, 4), "float32"), R.Tensor((2, 2, 4), "float32")])])
            return gv

        @Ts.function(private=True)
        def split(rxplaceholder: T.Tensor((T.int64(2), T.int64(10), T.int64(4)), "float32"), T_split_sections: T.Tensor((T.int64(2), T.int64(4), T.int64(4)), "float32"), T_split_sections_1: T.Tensor((T.int64(2), T.int64(4), T.int64(4)), "float32"), T_split_sections_2: T.Tensor((T.int64(2), T.int64(2), T.int64(4)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(4), T.int64(4)):
                with Ts.sblock("T_split_sections"):
                    ax0, ax1, ax2 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(rxplaceholder[ax0, ax1, ax2])
                    Ts.writes(T_split_sections[ax0, ax1, ax2])
                    T_split_sections[ax0, ax1, ax2] = rxplaceholder[ax0, ax1, ax2]
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(4), T.int64(4)):
                with Ts.sblock("T_split_sections_1"):
                    ax0, ax1, ax2 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(rxplaceholder[ax0, ax1 + T.int64(4), ax2])
                    Ts.writes(T_split_sections_1[ax0, ax1, ax2])
                    T_split_sections_1[ax0, ax1, ax2] = rxplaceholder[ax0, ax1 + T.int64(4), ax2]
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(2), T.int64(4)):
                with Ts.sblock("T_split_sections_2"):
                    ax0, ax1, ax2 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(rxplaceholder[ax0, ax1 + T.int64(8), ax2])
                    Ts.writes(T_split_sections_2[ax0, ax1, ax2])
                    T_split_sections_2[ax0, ax1, ax2] = rxplaceholder[ax0, ax1 + T.int64(8), ax2]

    # fmt: on

    mod = LegalizeOps()(Split)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_split_by_indices_n_section_divisible():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Split:
        @R.function
        def main(x: R.Tensor((2, 10, 4), "float32")) -> R.Tuple([R.Tensor((2, 5, 4), "float32"), R.Tensor((2, 5, 4), "float32")]):
            gv: R.Tuple([R.Tensor((2, 5, 4), "float32"), R.Tensor((2, 5, 4), "float32")]) = R.split(x, 2, axis=1)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 10, 4), "float32")) -> R.Tuple([R.Tensor((2, 5, 4), "float32"), R.Tensor((2, 5, 4), "float32")]):
            gv = R.call_tir(Expected.split, (x,), ty_args=[_TupleType([R.Tensor((2, 5, 4), "float32"), R.Tensor((2, 5, 4), "float32")])])
            return gv

        @Ts.function(private=True)
        def split(rxplaceholder: T.Tensor((T.int64(2), T.int64(10), T.int64(4)), "float32"), T_split_sections: T.Tensor((T.int64(2), T.int64(5), T.int64(4)), "float32"), T_split_sections_1: T.Tensor((T.int64(2), T.int64(5), T.int64(4)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(5), T.int64(4)):
                with Ts.sblock("T_split_sections"):
                    ax0, ax1, ax2 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(rxplaceholder[ax0, ax1, ax2])
                    Ts.writes(T_split_sections[ax0, ax1, ax2])
                    T_split_sections[ax0, ax1, ax2] = rxplaceholder[ax0, ax1, ax2]
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(5), T.int64(4)):
                with Ts.sblock("T_split_sections_1"):
                    ax0, ax1, ax2 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(rxplaceholder[ax0, ax1 + T.int64(5), ax2])
                    Ts.writes(T_split_sections_1[ax0, ax1, ax2])
                    T_split_sections_1[ax0, ax1, ax2] = rxplaceholder[ax0, ax1 + T.int64(5), ax2]
    # fmt: on

    mod = LegalizeOps()(Split)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_split_by_indices_n_section_divisible_symbolic():
    # fmt: off
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Split:
        @R.function
        def main(dumb_param: R.Tensor((n,)), x: R.Tensor((m, n * 3), "float32")) -> R.Tuple([R.Tensor((m, n), "float32"), R.Tensor((m, n), "float32"), R.Tensor((m, n), "float32")]):
            gv: R.Tuple([R.Tensor((m, n), "float32"), R.Tensor((m, n), "float32"), R.Tensor((m, n), "float32")]) = R.split(x, 3, axis=1)
            return gv

    m_main = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")
    m_split = T.dynamic("m", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(dumb_param: R.Tensor((n,)), x: R.Tensor((m_main, n * 3), "float32")) -> R.Tuple(R.Tensor((m_main, n * 3 // 3), "float32"), R.Tensor((m_main, n * 3 // 3 * 2 - n * 3 // 3), "float32"), R.Tensor((m_main, n * 3 - n * 3 // 3 * 2), "float32")):
            gv = R.call_tir(Expected.split, (x, n), ty_args=[_TupleType([R.Tensor((m_main, ((n * 3 + 3 - 1) // 3)), "float32"), R.Tensor((m_main, ((((n * 3 + 3 - 1) // 3) * 2) - ((n * 3 + 3 - 1) // 3))), "float32"), R.Tensor((m_main, ((n * 3) - (((n * 3 + 3 - 1) // 3) * 2))), "float32")])])
            return gv

        split_n = T.dynamic("split_n", "int64")

        @Ts.function(private=True)
        def split(rxplaceholder: T.Tensor([m_split, split_n * T.int64(3)], dtype='float32'), n: split_n, T_split_sections: T.Tensor([m_split, (split_n * T.int64(3) + T.int64(3) - T.int64(1)) // T.int64(3)], dtype='float32'), T_split_sections_1: T.Tensor([m_split, (split_n * T.int64(3) + T.int64(3) - T.int64(1)) // T.int64(3) * T.int64(2) - (split_n * T.int64(3) + T.int64(3) - T.int64(1)) // T.int64(3)], dtype='float32'), T_split_sections_2: T.Tensor([m_split, split_n * T.int64(3) - (split_n * T.int64(3) + T.int64(3) - T.int64(1)) // T.int64(3) * T.int64(2)], dtype='float32')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1 in T.grid(m_split, n):
                with Ts.sblock("T_split_sections"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[ax0, ax1])
                    Ts.writes(T_split_sections[ax0, ax1])
                    T_split_sections[ax0, ax1] = rxplaceholder[ax0, ax1]
            for i0, i1 in T.grid(m_split, n):
                with Ts.sblock("T_split_sections_1"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[ax0, ax1 + n])
                    Ts.writes(T_split_sections_1[ax0, ax1])
                    T_split_sections_1[ax0, ax1] = rxplaceholder[ax0, ax1 + n]
            for i0, i1 in T.grid(m_split, n):
                with Ts.sblock("T_split_sections_2"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[ax0, n * T.int64(2) + ax1])
                    Ts.writes(T_split_sections_2[ax0, ax1])
                    T_split_sections_2[ax0, ax1] = rxplaceholder[ax0, n * T.int64(2) + ax1]
    # fmt: on

    mod = LegalizeOps()(Split)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_squeeze():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Squeeze:
        @R.function
        def main(x: R.Tensor((2, 1, 3, 1, 1, 4), "float32")) -> R.Tensor((2, 3, 1, 4), "float32"):
            gv: R.Tensor((2, 3, 1, 4), "float32") = R.squeeze(x, [1, 4])
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 1, 3, 1, 1, 4), "float32")) -> R.Tensor((2, 3, 1, 4), "float32"):
            gv = R.call_tir(Expected.squeeze, (x,), ty_args=[R.Tensor((2, 3, 1, 4), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def squeeze(rxplaceholder: T.Tensor((T.int64(2), T.int64(1), T.int64(3), T.int64(1), T.int64(1), T.int64(4)), "float32"), T_squeeze: T.Tensor((T.int64(2), T.int64(3), T.int64(1), T.int64(4)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(1), T.int64(4)):
                with Ts.sblock("T_squeeze"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[ax0, T.int64(0), ax1, ax2, T.int64(0), ax3])
                    Ts.writes(T_squeeze[ax0, ax1, ax2, ax3])
                    T_squeeze[ax0, ax1, ax2, ax3] = rxplaceholder[ax0, T.int64(0), ax1, ax2, T.int64(0), ax3]
    # fmt: on

    mod = LegalizeOps()(Squeeze)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_squeeze_no_axis():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Squeeze:
        @R.function
        def main(x: R.Tensor((2, 1, 3, 1, 1, 4), "float32")) :
            gv: R.Tensor((2, 3, 4), "float32") = R.squeeze(x)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 1, 3, 1, 1, 4), "float32")) :
            gv = R.call_tir(Expected.squeeze, (x,), ty_args=[R.Tensor((2, 3, 4), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def squeeze(rxplaceholder: T.Tensor((T.int64(2), T.int64(1), T.int64(3), T.int64(1), T.int64(1), T.int64(4)), "float32"), T_squeeze: T.Tensor((T.int64(2), T.int64(3), T.int64(4)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(3), T.int64(4)):
                with Ts.sblock("T_squeeze"):
                    ax0, ax1, ax2 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(rxplaceholder[ax0, T.int64(0), ax1, T.int64(0), T.int64(0), ax2])
                    Ts.writes(T_squeeze[ax0, ax1, ax2])
                    T_squeeze[ax0, ax1, ax2] = rxplaceholder[ax0, T.int64(0), ax1, T.int64(0), T.int64(0), ax2]
    # fmt: on

    mod = LegalizeOps()(Squeeze)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_squeeze_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")

    @tvm_tirx.script.ir_module
    class Squeeze:
        @R.function
        def main(x: R.Tensor((a, 1, b, 1), "float32")) -> R.Tensor((a, b, 1), "float32"):
            gv: R.Tensor((a, b, 1), "float32") = R.squeeze(x, [1])
            return gv

    a_main = T.dynamic("a", "int64")
    b_main = T.dynamic("b", "int64")
    a_squeeze = T.dynamic("a", "int64")
    b_squeeze = T.dynamic("b", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((a_main, 1, b_main, 1), "float32")) -> R.Tensor((a_main, b_main, 1), "float32"):
            gv = R.call_tir(Expected.squeeze, (x,), ty_args=[R.Tensor((a_main, b_main, 1), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def squeeze(rxplaceholder: T.Tensor([a_squeeze, T.int64(1), b_squeeze, T.int64(1)], dtype='float32'), T_squeeze: T.Tensor([a_squeeze, b_squeeze, T.int64(1)], dtype='float32')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1, i2 in T.grid(a_squeeze, b_squeeze, T.int64(1)):
                with Ts.sblock("T_squeeze"):
                    ax0, ax1, ax2 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(rxplaceholder[ax0, T.int64(0), ax1, ax2])
                    Ts.writes(T_squeeze[ax0, ax1, ax2])
                    T_squeeze[ax0, ax1, ax2] = rxplaceholder[ax0, T.int64(0), ax1, ax2]
    # fmt: on

    mod = LegalizeOps()(Squeeze)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_collapse_sum_like():
    # fmt: off
    @tvm_tirx.script.ir_module
    class CollapseSumLike:
        @R.function
        def main(x: R.Tensor((2, 3), "float32"), y: R.Tensor((1, 3), "float32")) -> R.Tensor((1, 3), "float32"):
            gv: R.Tensor((1, 3), "float32") = R.collapse_sum_like(x, y)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32"), y: R.Tensor((1, 3), "float32")) -> R.Tensor((1, 3), "float32"):
            gv = R.call_tir(Expected.collapse_sum, (x,), ty_args=[R.Tensor((1, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def collapse_sum(rxplaceholder: T.Tensor((T.int64(2), T.int64(3)), "float32"), rxplaceholder_red: T.Tensor((T.int64(1), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2 in T.grid(T.int64(1), T.int64(3), T.int64(2)):
                with Ts.sblock("rxplaceholder_red"):
                    ax0, ax1, k0 = Ts.axis.remap("SSR", [i0, i1, i2])
                    Ts.reads(rxplaceholder[k0, ax1])
                    Ts.writes(rxplaceholder_red[ax0, ax1])
                    with Ts.init():
                        rxplaceholder_red[ax0, ax1] = T.float32(0)
                    rxplaceholder_red[ax0, ax1] = rxplaceholder_red[ax0, ax1] + rxplaceholder[k0, ax1]
    # fmt: on

    mod = LegalizeOps()(CollapseSumLike)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_collapse_sum_to():
    # fmt: off
    @tvm_tirx.script.ir_module
    class CollapseSumTo:
        @R.function
        def main(x: R.Tensor((3, 2, 3), "float32")) -> R.Tensor((2, 1), "float32"):
            gv: R.Tensor((2, 1), "float32") = R.collapse_sum_to(x, (2, 1))
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((3, 2, 3), dtype="float32")
        ) -> R.Tensor((2, 1), dtype="float32"):
            # block 0
            gv = R.call_tir(Expected.collapse_sum, (x,), ty_args=[R.Tensor((2, 1), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def collapse_sum(rxplaceholder: T.Tensor((T.int64(3), T.int64(2), T.int64(3)), "float32"), rxplaceholder_red: T.Tensor((T.int64(2), T.int64(1)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for ax0, ax1, k0, k2 in T.grid(T.int64(2), T.int64(1), T.int64(3), T.int64(3)):
                with Ts.sblock("rxplaceholder_red"):
                    v_ax0, v_ax1, v_k0, v_k2 = Ts.axis.remap("SSRR", [ax0, ax1, k0, k2])
                    Ts.reads(rxplaceholder[v_k0, v_ax0, v_k2])
                    Ts.writes(rxplaceholder_red[v_ax0, v_ax1])
                    with Ts.init():
                        rxplaceholder_red[v_ax0, v_ax1] = T.float32(0)
                    rxplaceholder_red[v_ax0, v_ax1] = (rxplaceholder_red[v_ax0, v_ax1] + rxplaceholder[v_k0, v_ax0, v_k2])
    # fmt: on

    mod = LegalizeOps()(CollapseSumTo)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_repeat():
    # fmt: off
    @I.ir_module
    class Repeat:
        @R.function
        def main(x: R.Tensor((3, 2, 3), "float32")):
            gv = R.repeat(x, 2, 0)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((3, 2, 3), dtype="float32")) -> R.Tensor((6, 2, 3), dtype="float32"):
            gv = R.call_tir(Expected.repeat, (x,), ty_args=[R.Tensor((6, 2, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def repeat(rxplaceholder: T.Tensor((T.int64(3), T.int64(2), T.int64(3)), "float32"), T_repeat: T.Tensor((T.int64(6), T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1, ax2 in T.grid(T.int64(6), T.int64(2), T.int64(3)):
                with Ts.sblock("T_repeat"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(rxplaceholder[v_ax0 // T.int64(2), v_ax1, v_ax2])
                    Ts.writes(T_repeat[v_ax0, v_ax1, v_ax2])
                    T_repeat[v_ax0, v_ax1, v_ax2] = rxplaceholder[v_ax0 // T.int64(2), v_ax1, v_ax2]
    # fmt: on

    mod = LegalizeOps()(Repeat)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_repeat_no_axis():
    # fmt: off
    @I.ir_module
    class Repeat:
        @R.function
        def main(x: R.Tensor((3, 2, 3), "float32")):
            gv = R.repeat(x, 2)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((3, 2, 3), dtype="float32")
        ) -> R.Tensor((36,), dtype="float32"):
            gv = R.call_tir(Expected.repeat, (x,), ty_args=[R.Tensor((36,), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def repeat(
            rxplaceholder: T.Tensor((T.int64(3), T.int64(2), T.int64(3)), "float32"),
            T_repeat: T.Tensor((T.int64(36),), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            T_reshape = Ts.sblock_alloc_tensor((T.int64(18),))
            for ax0 in range(T.int64(18)):
                with Ts.sblock("T_reshape"):
                    v_ax0 = Ts.axis.spatial(T.int64(18), ax0)
                    Ts.reads(
                        rxplaceholder[
                            v_ax0 % T.int64(18) // T.int64(6),
                            v_ax0 % T.int64(6) // T.int64(3),
                            v_ax0 % T.int64(3),
                        ]
                    )
                    Ts.writes(T_reshape[v_ax0])
                    T_reshape[v_ax0] = rxplaceholder[
                        v_ax0 % T.int64(18) // T.int64(6),
                        v_ax0 % T.int64(6) // T.int64(3),
                        v_ax0 % T.int64(3),
                    ]
            for ax0 in range(T.int64(36)):
                with Ts.sblock("T_repeat"):
                    v_ax0 = Ts.axis.spatial(T.int64(36), ax0)
                    Ts.reads(T_reshape[v_ax0 // T.int64(2)])
                    Ts.writes(T_repeat[v_ax0])
                    T_repeat[v_ax0] = T_reshape[v_ax0 // T.int64(2)]
    # fmt: on

    mod = LegalizeOps()(Repeat)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_repeat_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")
    c = T.dynamic("c", "int64")

    @I.ir_module
    class Repeat:
        @R.function
        def main(x: R.Tensor((a, b, c), "float32")):
            gv = R.repeat(x, 2, 0)
            return gv

    a_repeat = T.dynamic("a", "int64")
    b_repeat = T.dynamic("b", "int64")
    c_repeat = T.dynamic("c", "int64")
    a_main = T.dynamic("a", "int64")
    b_main = T.dynamic("b", "int64")
    c_main = T.dynamic("c", "int64")

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def repeat(rxplaceholder: T.Tensor((a_repeat, b_repeat, c_repeat), "float32"), T_repeat: T.Tensor((T.int64(2) * a_repeat, b_repeat, c_repeat), "float32")):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            for ax0, ax1, ax2 in T.grid(a_repeat * T.int64(2), b_repeat, c_repeat):
                with Ts.sblock("T_repeat"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(rxplaceholder[v_ax0 // T.int64(2), v_ax1, v_ax2])
                    Ts.writes(T_repeat[v_ax0, v_ax1, v_ax2])
                    T_repeat[v_ax0, v_ax1, v_ax2] = rxplaceholder[v_ax0 // T.int64(2), v_ax1, v_ax2]

        @R.function
        def main(x: R.Tensor((a_main, b_main, c_main), dtype="float32")) -> R.Tensor((2 * a_main, b_main, c_main), dtype="float32"):
            gv = R.call_tir(Expected.repeat, (x,), ty_args=[R.Tensor((2 * a_main, b_main, c_main), dtype="float32")])
            return gv
    # fmt: on

    mod = LegalizeOps()(Repeat)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_tile():
    # fmt: off
    @I.ir_module
    class Tile:
        @R.function
        def main(x: R.Tensor((3, 2, 3), "float32")):
            gv = R.tile(x, (2, 1, 2, 3))
            return gv

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def tile(rxplaceholder: T.Tensor((T.int64(3), T.int64(2), T.int64(3)), "float32"), T_tile: T.Tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(9)), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(9)):
                with Ts.sblock("T_tile"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(rxplaceholder[v_ax1 % T.int64(3), v_ax2 % T.int64(2), v_ax3 % T.int64(3)])
                    Ts.writes(T_tile[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_tile[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[v_ax1 % T.int64(3), v_ax2 % T.int64(2), v_ax3 % T.int64(3)]

        @R.function
        def main(x: R.Tensor((3, 2, 3), dtype="float32")) -> R.Tensor((2, 3, 4, 9), dtype="float32"):
            gv = R.call_tir(Expected.tile, (x,), ty_args=[R.Tensor((2, 3, 4, 9), dtype="float32")])
            return gv
    # fmt: on

    mod = LegalizeOps()(Tile)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_tile_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")
    c = T.dynamic("c", "int64")

    @I.ir_module
    class Tile:
        @R.function
        def main(x: R.Tensor((a, b, c), "float32")):
            gv = R.tile(x, (2, 1, 2, 3))
            return gv

    a_tile = T.dynamic("a", "int64")
    b_tile = T.dynamic("b", "int64")
    c_tile = T.dynamic("c", "int64")
    a_main = T.dynamic("a", "int64")
    b_main = T.dynamic("b", "int64")
    c_main = T.dynamic("c", "int64")

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def tile(rxplaceholder: T.Tensor((a_tile, b_tile, c_tile), "float32"), T_tile: T.Tensor((T.int64(2), a_tile, b_tile * T.int64(2), c_tile * T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), a_tile, b_tile * T.int64(2), c_tile * T.int64(3)):
                with Ts.sblock("T_tile"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(rxplaceholder[v_ax1 % a_tile, v_ax2 % b_tile, v_ax3 % c_tile])
                    Ts.writes(T_tile[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_tile[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[v_ax1 % a_tile, v_ax2 % b_tile, v_ax3 % c_tile]

        @R.function
        def main(x: R.Tensor((a_main, b_main, c_main), dtype="float32")) -> R.Tensor((2, a_main, b_main * 2, c_main * 3), dtype="float32"):
            gv = R.call_tir(Expected.tile, (x,), ty_args=[R.Tensor((2, a_main, b_main * 2, c_main * 3), dtype="float32")])
            return gv
    # fmt: on
    mod = LegalizeOps()(Tile)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_flip():
    # fmt: off
    @I.ir_module
    class Flip:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")):
            gv = R.flip(x, axis=0)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.flip, (x,), ty_args=[R.Tensor((2, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def flip(
            rxplaceholder: T.Tensor((T.int64(2), T.int64(3)), "float32"),
            T_reverse_sequence: T.Tensor((T.int64(2), T.int64(3)), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_reverse_sequence"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(rxplaceholder[T.int64(1) - v_ax0, v_ax1])
                    Ts.writes(T_reverse_sequence[v_ax0, v_ax1])
                    T_reverse_sequence[v_ax0, v_ax1] = rxplaceholder[
                        T.int64(1) - v_ax0, v_ax1
                    ]

    # fmt: on

    mod = LegalizeOps()(Flip)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_flip_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")

    @I.ir_module
    class Flip:
        @R.function
        def main(x: R.Tensor((a, b), "float32")):
            gv = R.flip(x, axis=1)
            return gv

    a_main = T.dynamic("a", "int64")
    b_main = T.dynamic("b", "int64")
    a_flip = T.dynamic("a", "int64")
    b_flip = T.dynamic("b", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((a_main, b_main), dtype="float32")
        ) -> R.Tensor((a_main, b_main), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.flip, (x,), ty_args=[R.Tensor((a_main, b_main), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def flip(rxplaceholder: T.Tensor((a_flip, b_flip), "float32"), T_reverse_sequence: T.Tensor((a_flip, b_flip), "float32")):
            T.func_attr({"tirx.noalias": True})

            for ax0, ax1 in T.grid(a_flip, b_flip):
                with Ts.sblock("T_reverse_sequence"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(rxplaceholder[v_ax0, b_flip - v_ax1 - T.int64(1)])
                    Ts.writes(T_reverse_sequence[v_ax0, v_ax1])
                    T_reverse_sequence[v_ax0, v_ax1] = rxplaceholder[
                        v_ax0, b_flip - v_ax1 - T.int64(1)
                    ]

    # fmt: on

    mod = LegalizeOps()(Flip)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_reverse_sequence():
    # fmt: off
    @I.ir_module
    class ReverseSequence:
        @R.function
        def main(x: R.Tensor((4, 2, 3), "float32"), seq_lengths: R.Tensor((2,), "int64")):
            gv = R.reverse_sequence(x, seq_lengths, seq_axis=0, batch_axis=1)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((4, 2, 3), dtype="float32"),
            seq_lengths: R.Tensor((2,), dtype="int64"),
        ) -> R.Tensor((4, 2, 3), dtype="float32"):
            cls = Expected
            gv = R.call_tir(
                cls.reverse_sequence,
                (x, seq_lengths),
                ty_args=[R.Tensor((4, 2, 3), dtype="float32")],
            )
            return gv

        @Ts.function(private=True)
        def reverse_sequence(
            rxplaceholder: T.Tensor((T.int64(4), T.int64(2), T.int64(3)), "float32"),
            seq_lengths: T.Tensor((T.int64(2),), "int64"),
            T_reverse_sequence: T.Tensor((T.int64(4), T.int64(2), T.int64(3)), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for ax0, ax1, ax2 in T.grid(T.int64(4), T.int64(2), T.int64(3)):
                with Ts.sblock("T_reverse_sequence"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(rxplaceholder[T.int64(0):T.int64(4), v_ax1, v_ax2], seq_lengths[v_ax1])
                    Ts.writes(T_reverse_sequence[v_ax0, v_ax1, v_ax2])
                    T_reverse_sequence[v_ax0, v_ax1, v_ax2] = rxplaceholder[
                        T.if_then_else(
                            seq_lengths[v_ax1] <= T.int64(1) or seq_lengths[v_ax1] <= v_ax0,
                            v_ax0,
                            T.if_then_else(
                                T.int64(4) < seq_lengths[v_ax1],
                                T.int64(3) - v_ax0,
                                seq_lengths[v_ax1] - v_ax0 - T.int64(1),
                            ),
                        ),
                        v_ax1,
                        v_ax2,
                    ]

    # fmt: on

    mod = LegalizeOps()(ReverseSequence)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_scatter_elements():
    # fmt: off
    @I.ir_module
    class ScatterElements:
        @R.function
        def main(x: R.Tensor((4,4), "float32"), indices: R.Tensor((2,2), "int64"), updates: R.Tensor((2,2), "float32")):
            gv = R.scatter_elements(x, indices, updates, axis=1)
            return gv
    expected_rxplaceholder_offset = T.dynamic("expected_rxplaceholder_offset", "int64")
    expected_rxplaceholder_1_offset = T.dynamic("expected_rxplaceholder_1_offset", "int64")
    expected_rxplaceholder_2_offset = T.dynamic("expected_rxplaceholder_2_offset", "int64")

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def scatter_elements(
            rxplaceholder: T.Tensor((T.int64(4), T.int64(4)), "float32",  layout=T.TileLayout(T.S[(T.int64(4), T.int64(4))] + (expected_rxplaceholder_offset))),
            rxplaceholder_1: T.Tensor((T.int64(2), T.int64(2)), 'int64',  layout=T.TileLayout(T.S[(T.int64(2), T.int64(2))] + (expected_rxplaceholder_1_offset))),
            rxplaceholder_2: T.Tensor((T.int64(2), T.int64(2)), "float32",  layout=T.TileLayout(T.S[(T.int64(2), T.int64(2))] + (expected_rxplaceholder_2_offset))),
            out_buf: T.Tensor((T.int64(4), T.int64(4)), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})

            with Ts.sblock("scatter_elements_generic"):
                for i in T.parallel(T.int64(16)):
                    out_buf[i // T.int64(4), i % T.int64(4)] = rxplaceholder[
                        i // T.int64(4), i % T.int64(4)
                    ]
                for fused in T.parallel(T.int64(2)):
                    for k in range(T.int64(2)):
                        out_buf[
                            (
                                fused * T.int64(4)
                                + (
                                    rxplaceholder_1[
                                        (fused * T.int64(2) + k) // T.int64(2),
                                        (fused * T.int64(2) + k) % T.int64(2),
                                    ]
                                    + T.Cast(
                                        "int64",
                                        rxplaceholder_1[
                                            (fused * T.int64(2) + k) // T.int64(2),
                                            (fused * T.int64(2) + k) % T.int64(2),
                                        ]
                                        < T.int64(0),
                                    )
                                    * T.int64(4)
                                )
                            )
                            // T.int64(4),
                            (
                                fused * T.int64(4)
                                + (
                                    rxplaceholder_1[
                                        (fused * T.int64(2) + k) // T.int64(2),
                                        (fused * T.int64(2) + k) % T.int64(2),
                                    ]
                                    + T.Cast(
                                        "int64",
                                        rxplaceholder_1[
                                            (fused * T.int64(2) + k) // T.int64(2),
                                            (fused * T.int64(2) + k) % T.int64(2),
                                        ]
                                        < T.int64(0),
                                    )
                                    * T.int64(4)
                                )
                            )
                            % T.int64(4),
                        ] = rxplaceholder_2[
                            (fused * T.int64(2) + k) // T.int64(2),
                            (fused * T.int64(2) + k) % T.int64(2),
                        ]

        @R.function
        def main(
            x: R.Tensor((4, 4), dtype="float32"),
            indices: R.Tensor((2, 2), dtype="int64"),
            updates: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tensor((4, 4), dtype="float32"):
            gv = R.call_tir(
                Expected.scatter_elements,
                (x, indices, updates),
                ty_args=[R.Tensor((4, 4), dtype="float32")],
            )
            return gv

    # fmt: on
    mod = LegalizeOps()(ScatterElements)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_scatter_elements_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @I.ir_module
    class ScatterElements:
        @R.function
        def main(x: R.Tensor((a, b), "float32"), indices:R.Tensor((m, n), "int64"), updates:R.Tensor((m,n), "float32")):
            gv = R.scatter_elements(x, indices, updates, axis=1)
            return gv
    a_scatter_elements = T.dynamic("a", "int64")
    b_scatter_elements = T.dynamic("b", "int64")
    m_scatter_elements = T.dynamic("m", "int64")
    n_scatter_elements = T.dynamic("n", "int64")
    a_main = T.dynamic("a", "int64")
    b_main = T.dynamic("b", "int64")
    m_main = T.dynamic("m", "int64")
    n_main = T.dynamic("n", "int64")

    expected_rxplaceholder_offset_1 = T.dynamic("expected_rxplaceholder_offset_1", "int64")
    expected_rxplaceholder_1_offset_1 = T.dynamic("expected_rxplaceholder_1_offset_1", "int64")
    expected_rxplaceholder_2_offset_1 = T.dynamic("expected_rxplaceholder_2_offset_1", "int64")

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def scatter_elements(
            rxplaceholder: T.Tensor((a_scatter_elements, b_scatter_elements), "float32",  layout=T.TileLayout(T.S[(a_scatter_elements, b_scatter_elements)] + (expected_rxplaceholder_offset_1))),
            rxplaceholder_1: T.Tensor((m_scatter_elements, n_scatter_elements), 'int64',  layout=T.TileLayout(T.S[(m_scatter_elements, n_scatter_elements)] + (expected_rxplaceholder_1_offset_1))),
            rxplaceholder_2: T.Tensor((m_scatter_elements, n_scatter_elements), "float32",  layout=T.TileLayout(T.S[(m_scatter_elements, n_scatter_elements)] + (expected_rxplaceholder_2_offset_1))),
            out_buf: T.Tensor((a_scatter_elements, b_scatter_elements), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})

            with Ts.sblock("scatter_elements_generic"):
                for i in T.parallel(a_scatter_elements * b_scatter_elements):
                    out_buf[i // b_scatter_elements, i % b_scatter_elements] = rxplaceholder[i // b_scatter_elements, i % b_scatter_elements]
                for fused in T.parallel(m_scatter_elements):
                    for k in range(n_scatter_elements):
                        out_buf[
                            (
                                fused * b_scatter_elements
                                + (
                                    rxplaceholder_1[
                                        (fused * n_scatter_elements + k) // n_scatter_elements, (fused * n_scatter_elements + k) % n_scatter_elements
                                    ]
                                    + T.Cast(
                                        "int64",
                                        rxplaceholder_1[
                                            (fused * n_scatter_elements + k) // n_scatter_elements, (fused * n_scatter_elements + k) % n_scatter_elements
                                        ]
                                        < T.int64(0),
                                    )
                                    * b_scatter_elements
                                )
                            )
                            // b_scatter_elements,
                            (
                                fused * b_scatter_elements
                                + (
                                    rxplaceholder_1[
                                        (fused * n_scatter_elements + k) // n_scatter_elements, (fused * n_scatter_elements + k) % n_scatter_elements
                                    ]
                                    + T.Cast(
                                        "int64",
                                        rxplaceholder_1[
                                            (fused * n_scatter_elements + k) // n_scatter_elements, (fused * n_scatter_elements + k) % n_scatter_elements
                                        ]
                                        < T.int64(0),
                                    )
                                    * b_scatter_elements
                                )
                            )
                            % b_scatter_elements,
                        ] = rxplaceholder_2[(fused * n_scatter_elements + k) // n_scatter_elements, (fused * n_scatter_elements + k) % n_scatter_elements]

        @R.function
        def main(
            x: R.Tensor((a_main, b_main), dtype="float32"),
            indices: R.Tensor((m_main, n_main), dtype="int64"),
            updates: R.Tensor((m_main, n_main), dtype="float32"),
        ) -> R.Tensor((a_main, b_main), dtype="float32"):
            gv = R.call_tir(
                Expected.scatter_elements,
                (x, indices, updates),
                ty_args=[R.Tensor((a_main, b_main), dtype="float32")],
            )
            return gv
    # fmt: on

    mod = LegalizeOps()(ScatterElements)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_scatter_elements_gpu():
    """scatter_elements lowered for GPU must build"""
    target = "cuda"

    @I.ir_module
    class Mod:
        @R.function
        def main(
            x: R.Tensor((4, 8), "float32"),
            indices: R.Tensor((2, 8), "int64"),
            updates: R.Tensor((2, 8), "float32"),
        ):
            with R.dataflow():
                lv = R.scatter_elements(x, indices, updates, axis=0)
                gv = lv
                R.output(gv)
            return gv

    mod = LegalizeOps(target=tvm_tirx.target.Target(target))(Mod)
    relax.build(mod, target=target)


def test_layout_transform():
    transformation = lambda a, b, c: (a, c, b // 3, b % 3)
    pad_value = 2

    # fmt: off
    @I.ir_module
    class LayoutTransform:
        @R.function
        def main(x: R.Tensor((10, 21, 30), "float32")):
            gv = R.layout_transform(
                x, index_map=transformation, pad_value=pad_value
            )
            return gv

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def te_layout_transform(A: T.Tensor((T.int64(10), T.int64(21), T.int64(30)), "float32"), te_layout_transform_1: T.Tensor((T.int64(10), T.int64(30), T.int64(7), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0, i1, i2 in T.grid(T.int64(10), T.int64(21), T.int64(30)):
                with Ts.sblock("te_layout_transform"):
                    v_i0, v_i1, v_i2 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(A[v_i0, v_i1, v_i2])
                    Ts.writes(te_layout_transform_1[v_i0, v_i2, v_i1 // T.int64(3), v_i1 % T.int64(3)])
                    te_layout_transform_1[v_i0, v_i2, v_i1 // T.int64(3), v_i1 % T.int64(3)] = A[v_i0, v_i1, v_i2]

        @R.function
        def main(x: R.Tensor((10, 21, 30), dtype="float32")) -> R.Tensor((10, 30, 7, 3), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.te_layout_transform, (x,), ty_args=[R.Tensor((10, 30, 7, 3), dtype="float32")])
            return gv
    # fmt: on

    mod = LegalizeOps()(LayoutTransform)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_layout_transform_with_pad():
    transformation = lambda a, b, c: (a, c, b // 3, b % 3)
    pad_value = 2

    # fmt: off
    @I.ir_module
    class LayoutTransform:
        @R.function
        def main(x: R.Tensor((10, 20, 30), "float32")):
            gv = R.layout_transform(
                x, index_map=transformation, pad_value=pad_value
            )
            return gv

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def te_layout_transform_with_pad(A: T.Tensor((T.int64(10), T.int64(20), T.int64(30)), "float32"), te_layout_transform_with_pad_1: T.Tensor((T.int64(10), T.int64(30), T.int64(7), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for axis0, axis1, axis2, axis3 in T.grid(T.int64(10), T.int64(30), T.int64(7), T.int64(3)):
                with Ts.sblock("te_layout_transform_with_pad"):
                    v_axis0, v_axis1, v_axis2, v_axis3 = Ts.axis.remap("SSSS", [axis0, axis1, axis2, axis3])
                    Ts.reads(A[v_axis0, v_axis2 * T.int64(3) + v_axis3, v_axis1])
                    Ts.writes(te_layout_transform_with_pad_1[v_axis0, v_axis1, v_axis2, v_axis3])
                    te_layout_transform_with_pad_1[v_axis0, v_axis1, v_axis2, v_axis3] = T.if_then_else(v_axis2 == T.int64(6) and v_axis3 == T.int64(2), T.float32(2), A[v_axis0, v_axis2 * T.int64(3) + v_axis3, v_axis1])

        @R.function
        def main(x: R.Tensor((10, 20, 30), dtype="float32")) -> R.Tensor((10, 30, 7, 3), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.te_layout_transform_with_pad, (x,), ty_args=[R.Tensor((10, 30, 7, 3), dtype="float32")])
            return gv
    # fmt: on

    mod = LegalizeOps()(LayoutTransform)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_layout_transform_symbolic():
    transformation = lambda a, b, c: (a, c, b // 3, b % 3)
    pad_value = 2

    # fmt: off
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")
    c = T.dynamic("c", "int64")

    @I.ir_module
    class LayoutTransform:
        @R.function
        def main(x: R.Tensor((a, b, c), "float32")):
            gv = R.layout_transform(
                x, index_map=transformation, pad_value=pad_value
            )
            return gv

    a_te_layout_transform_with_pad = T.dynamic("a", "int64")
    b_te_layout_transform_with_pad = T.dynamic("b", "int64")
    c_te_layout_transform_with_pad = T.dynamic("c", "int64")
    a_main = T.dynamic("a", "int64")
    c_main = T.dynamic("c", "int64")
    b_main = T.dynamic("b", "int64")

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def te_layout_transform_with_pad(A: T.Tensor((a_te_layout_transform_with_pad, b_te_layout_transform_with_pad, c_te_layout_transform_with_pad), "float32"), te_layout_transform_with_pad_1: T.Tensor((a_te_layout_transform_with_pad, c_te_layout_transform_with_pad, (b_te_layout_transform_with_pad - b_te_layout_transform_with_pad % T.int64(-3)) // T.int64(3), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            for axis0, axis1, axis2, axis3 in T.grid(a_te_layout_transform_with_pad, c_te_layout_transform_with_pad, (b_te_layout_transform_with_pad - b_te_layout_transform_with_pad % T.int64(-3)) // T.int64(3), T.int64(3)):
                with Ts.sblock("te_layout_transform_with_pad_with_pad"):
                    v_axis0, v_axis1, v_axis2, v_axis3 = Ts.axis.remap("SSSS", [axis0, axis1, axis2, axis3])
                    Ts.reads(A[v_axis0, v_axis2 * T.int64(3) + v_axis3, v_axis1])
                    Ts.writes(te_layout_transform_with_pad_1[v_axis0, v_axis1, v_axis2, v_axis3])
                    te_layout_transform_with_pad_1[v_axis0, v_axis1, v_axis2, v_axis3] = T.if_then_else(b_te_layout_transform_with_pad % T.int64(-3) < T.int64(0) and v_axis2 == b_te_layout_transform_with_pad // T.int64(3) and b_te_layout_transform_with_pad % T.int64(3) <= v_axis3, T.float32(2), A[v_axis0, v_axis2 * T.int64(3) + v_axis3, v_axis1])

        @R.function
        def main(x: R.Tensor((a_main, b_main, c_main), dtype="float32")) -> R.Tensor((a_main, c_main, (b_main - b_main % -3) // 3, 3), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.te_layout_transform_with_pad, (x,), ty_args=[R.Tensor((a_main, c_main, (b_main - b_main % -3) // 3, 3), dtype="float32")])
            return gv
    # fmt: on

    mod = LegalizeOps()(LayoutTransform)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_func_ty_of_legalized_layout_transform():
    """Function shape information must be correct

    This is a regression test.  Previously, the legalization of
    `R.layout_transform` produced a Function with `FuncType`
    different than its actual signature.  This resulted in errors
    when later passes attempted to infer the Type.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((16,), dtype="float32"), y: R.Tensor((16,), dtype="float32")
        ) -> R.Tensor((16,), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            with R.dataflow():
                lv: R.Tensor((4, 4), dtype="float32") = R.layout_transform(
                    x, index_map=lambda i: (i // 4, i % 4), pad_value=None
                )
                gv: R.Tensor((4, 4), dtype="float32") = lv
                R.output(gv)
            return gv

    After = tvm_tirx.ir.transform.Sequential(
        [
            relax.transform.LegalizeOps(),
            relax.transform.ToNonDataflow(),
            relax.transform.RemovePurityChecking(),
            relax.transform.CallTIRRewrite(),
        ]
    )(Before)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((16,), dtype="float32"),
            y: R.Tensor((16,), dtype="float32"),
        ):
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            alloc: R.Tensor((4, 4), dtype="float32") = R.emit_with_type(
                "relax.builtin.alloc_tensor",
                (R.shape([4, 4]), R.dtype("float32"), R.prim_value(0), R.str("global")),
                (R.Tensor((4, 4), dtype="float32"),),
            )
            R.call_tir_packed(cls.te_layout_transform, (x, alloc))
            lv = alloc
            gv = lv
            return gv

        @Ts.function(private=True)
        def te_layout_transform(
            A: T.Tensor((T.int64(16),), "float32"),
            te_layout_transform: T.Tensor((T.int64(4), T.int64(4)), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i in range(T.int64(16)):
                with Ts.sblock("te_layout_transform"):
                    vi = Ts.axis.spatial(T.int64(16), i)
                    te_layout_transform[vi // T.int64(4), vi % T.int64(4)] = A[vi]

    tvm_tirx.ir.assert_structural_equal(Expected, After)


def test_scatter_nd():
    # fmt: off
    @I.ir_module
    class Before:
        @R.function
        def main(
            data: R.Tensor((8,), "float32"),
            indices: R.Tensor((4, 1), "int64"),
            updates: R.Tensor((4,), "float32"),
        ) -> R.Tensor((8,), "float32"):
            gv: R.Tensor((8,), "float32") = R.scatter_nd(data, indices, updates, reduction="update")
            return gv

    After = relax.transform.LegalizeOps()(Before)

    expected_data_offset = T.dynamic("expected_data_offset", "int64")
    expected_updates_offset = T.dynamic("expected_updates_offset", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data: R.Tensor((8,), "float32"),
            indices: R.Tensor((4, 1), "int64"),
            updates: R.Tensor((4,), "float32"),
        ) -> R.Tensor((8,), "float32"):
            gv = R.call_tir(
                Expected.scatter_nd, (data, indices, updates), ty_args=[R.Tensor((8,), dtype="float32")]
            )
            return gv

        @Ts.function(private=True)
        def scatter_nd(data: T.Tensor((T.int64(8),), "float32",  layout=T.TileLayout(T.S[(T.int64(8),)] + (expected_data_offset))), indices: T.Tensor((T.int64(4), T.int64(1)), 'int64'), updates: T.Tensor((T.int64(4),), "float32",  layout=T.TileLayout(T.S[(T.int64(4),)] + (expected_updates_offset))), out_buf: T.Tensor((T.int64(8),), "float32")):
            T.func_attr({"tirx.noalias": True})

            with Ts.sblock("root"):
                Ts.reads()
                Ts.writes()
                T_transpose = Ts.sblock_alloc_tensor((T.int64(1), T.int64(4)), "int64")
                for ax0 in range(T.int64(1)):
                    for ax1 in range(T.int64(4)):
                        with Ts.sblock("T_transpose"):
                            v_ax0 = Ts.axis.spatial(T.int64(1), ax0)
                            v_ax1 = Ts.axis.spatial(T.int64(4), ax1)
                            Ts.reads(indices[v_ax1, v_ax0])
                            Ts.writes(T_transpose[v_ax0, v_ax1])
                            T_transpose[v_ax0, v_ax1] = indices[v_ax1, v_ax0]
                with Ts.sblock("scatter_nd_generic"):
                    Ts.reads()
                    Ts.writes()
                    for i in range(T.int64(8)):
                        out_buf[i] = data[i]
                    for j in range(T.int64(4)):
                        for k in T.parallel(T.int64(1)):
                            out_buf[k + T_transpose[j // T.int64(4), j % T.int64(4)]] = updates[j + k]

    # fmt: on
    tvm_tirx.ir.assert_structural_equal(After, Expected)


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_scatter_nd_gpu():
    """scatter_nd lowered for GPU must build"""
    target = "cuda"

    @I.ir_module
    class Mod:
        @R.function
        def main(
            data: R.Tensor((4, 8), "float32"),
            indices: R.Tensor((3, 2), "int64"),
            updates: R.Tensor((3,), "float32"),
        ):
            with R.dataflow():
                lv = R.scatter_nd(data, indices, updates)
                gv = lv
                R.output(gv)
            return gv

    mod = LegalizeOps(target=tvm_tirx.target.Target(target))(Mod)
    relax.build(mod, target=target)


if __name__ == "__main__":
    tvm_tirx.testing.main()
