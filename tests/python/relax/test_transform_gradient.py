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
import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.s_tir import script as Ts
from tvm_tirx.ir.base import assert_structural_equal
from tvm_tirx.script.parser import ir as I
from tvm_tirx.script.parser import relax as R
from tvm_tirx.script.parser import tirx as T


def test_simple():
    # fmt: off
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                gv = R.sum(x)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main_adjoint(x: R.Tensor((3, 3), dtype="float32")) -> R.Tuple(R.Tensor((), dtype="float32"), R.Tuple(R.Tensor((3, 3), dtype="float32"))):
            with R.dataflow():
                gv: R.Tensor((), dtype="float32") = R.sum(x, axis=None, keepdims=False)
                gv_adjoint: R.Tensor((), dtype="float32") = R.ones(R.shape([]), dtype="float32")
                x_adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(gv_adjoint, R.shape([3, 3]))
                x_adjoint_out: R.Tensor((3, 3), dtype="float32") = x_adjoint
                R.output(gv, x_adjoint_out)
            return (gv, (x_adjoint_out,))

        @R.function
        def main(x: R.Tensor((3, 3), dtype="float32")) -> R.Tensor((), dtype="float32"):
            with R.dataflow():
                gv: R.Tensor((), dtype="float32") = R.sum(x, axis=None, keepdims=False)
                R.output(gv)
            return gv
    # fmt: on

    After = relax.transform.Gradient("main")(Before)
    assert_structural_equal(After, Expected)


def test_assign_binding():
    # fmt: off
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                lv1 = x
                lv2 = lv1
                gv = R.sum(lv2)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main_adjoint(x: R.Tensor((3, 3), dtype="float32")) -> R.Tuple(R.Tensor((), dtype="float32"), R.Tuple(R.Tensor((3, 3), dtype="float32"))):
            with R.dataflow():
                lv1: R.Tensor((3, 3), dtype="float32") = x
                lv2: R.Tensor((3, 3), dtype="float32") = lv1
                gv: R.Tensor((), dtype="float32") = R.sum(lv2, axis=None, keepdims=False)
                gv_adjoint: R.Tensor((), dtype="float32") = R.ones(R.shape([]), dtype="float32")
                lv2_adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(gv_adjoint, R.shape([3, 3]))
                lv1_adjoint: R.Tensor((3, 3), dtype="float32") = lv2_adjoint
                x_adjoint: R.Tensor((3, 3), dtype="float32") = lv1_adjoint
                x_adjoint_out: R.Tensor((3, 3), dtype="float32") = x_adjoint
                R.output(gv, x_adjoint_out)
            return (gv, (x_adjoint_out,))

        @R.function
        def main(x: R.Tensor((3, 3), dtype="float32")) -> R.Tensor((), dtype="float32"):
            with R.dataflow():
                lv1: R.Tensor((3, 3), dtype="float32") = x
                lv2: R.Tensor((3, 3), dtype="float32") = lv1
                gv: R.Tensor((), dtype="float32") = R.sum(lv2, axis=None, keepdims=False)
                R.output(gv)
            return gv
    # fmt: on

    After = relax.transform.Gradient("main")(Before)
    assert_structural_equal(After, Expected)


def test_multiple_uses():
    # fmt: off
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                lv1 = R.add(x, x)
                lv2 = R.add(lv1, x)
                gv = R.sum(lv2)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main_adjoint(x: R.Tensor((3, 3), dtype="float32")) -> R.Tuple(R.Tensor((), dtype="float32"), R.Tuple(R.Tensor((3, 3), dtype="float32"))):
            with R.dataflow():
                lv1: R.Tensor((3, 3), dtype="float32") = R.add(x, x)
                lv2: R.Tensor((3, 3), dtype="float32") = R.add(lv1, x)
                gv: R.Tensor((), dtype="float32") = R.sum(lv2, axis=None, keepdims=False)
                gv_adjoint: R.Tensor((), dtype="float32") = R.ones(R.shape([]), dtype="float32")
                lv2_adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(gv_adjoint, R.shape([3, 3]))
                lv1_adjoint: R.Tensor((3, 3), dtype="float32") = lv2_adjoint
                x_adjoint: R.Tensor((3, 3), dtype="float32") = lv2_adjoint
                x_adjoint1: R.Tensor((3, 3), dtype="float32") = R.add(x_adjoint, lv1_adjoint)
                x_adjoint2: R.Tensor((3, 3), dtype="float32") = R.add(x_adjoint1, lv1_adjoint)
                x_adjoint_out: R.Tensor((3, 3), dtype="float32") = x_adjoint2
                R.output(gv, x_adjoint_out)
            return (gv, (x_adjoint_out,))

        @R.function
        def main(x: R.Tensor((3, 3), dtype="float32")) -> R.Tensor((), dtype="float32"):
            with R.dataflow():
                lv1: R.Tensor((3, 3), dtype="float32") = R.add(x, x)
                lv2: R.Tensor((3, 3), dtype="float32") = R.add(lv1, x)
                gv: R.Tensor((), dtype="float32") = R.sum(lv2, axis=None, keepdims=False)
                R.output(gv)
            return gv
    # fmt: on

    After = relax.transform.Gradient("main")(Before)
    assert_structural_equal(After, Expected)


def test_unused():
    # fmt: off
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((3, 3), "float32"), y: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                lv1 = R.add(x, x)
                lv2 = R.add(lv1, x)
                gv = R.sum(x)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main_adjoint(x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")) -> R.Tuple(R.Tensor((), dtype="float32"), R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32"))):
            with R.dataflow():
                gv: R.Tensor((), dtype="float32") = R.sum(x, axis=None, keepdims=False)
                gv_adjoint: R.Tensor((), dtype="float32") = R.ones(R.shape([]), dtype="float32")
                x_adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(gv_adjoint, R.shape([3, 3]))
                x_adjoint_out: R.Tensor((3, 3), dtype="float32") = x_adjoint
                y_adjoint: R.Tensor((3, 3), dtype="float32") = R.zeros(R.shape([3, 3]), dtype="float32")
                y_adjoint_out: R.Tensor((3, 3), dtype="float32") = y_adjoint
                R.output(gv, x_adjoint_out, y_adjoint_out)
            return (gv, (x_adjoint_out, y_adjoint_out))

        @R.function
        def main(x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")) -> R.Tensor((), dtype="float32"):
            with R.dataflow():
                lv1: R.Tensor((3, 3), dtype="float32") = R.add(x, x)
                lv2: R.Tensor((3, 3), dtype="float32") = R.add(lv1, x)
                gv: R.Tensor((), dtype="float32") = R.sum(x, axis=None, keepdims=False)
                R.output(gv)
            return gv
    # fmt: on

    After = relax.transform.Gradient("main")(Before)
    assert_structural_equal(After, Expected)


def test_default_require_grads():
    # fmt: off
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((3, 3), "float32"), y: R.Tensor((3, 3), "float32"), z: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                lv1 = R.add(x, y)
                lv2 = R.add(lv1, z)
                gv = R.sum(lv2)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected1:
        @R.function
        def main_adjoint(x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32"), z: R.Tensor((3, 3), dtype="float32")) -> R.Tuple(R.Tensor((), dtype="float32"), R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32"))):
            with R.dataflow():
                lv1: R.Tensor((3, 3), dtype="float32") = R.add(x, y)
                lv2: R.Tensor((3, 3), dtype="float32") = R.add(lv1, z)
                gv: R.Tensor((), dtype="float32") = R.sum(lv2, axis=None, keepdims=False)
                gv_adjoint: R.Tensor((), dtype="float32") = R.ones(R.shape([]), dtype="float32")
                lv2_adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(gv_adjoint, R.shape([3, 3]))
                lv1_adjoint: R.Tensor((3, 3), dtype="float32") = lv2_adjoint
                z_adjoint: R.Tensor((3, 3), dtype="float32") = lv2_adjoint
                x_adjoint: R.Tensor((3, 3), dtype="float32") = lv1_adjoint
                y_adjoint: R.Tensor((3, 3), dtype="float32") = lv1_adjoint
                x_adjoint_out: R.Tensor((3, 3), dtype="float32") = x_adjoint
                y_adjoint_out: R.Tensor((3, 3), dtype="float32") = y_adjoint
                z_adjoint_out: R.Tensor((3, 3), dtype="float32") = z_adjoint
                R.output(gv, x_adjoint_out, y_adjoint_out, z_adjoint_out)
            return (gv, (x_adjoint_out, y_adjoint_out, z_adjoint_out))

        @R.function
        def main(x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32"), z: R.Tensor((3, 3), dtype="float32")) -> R.Tensor((), dtype="float32"):
            with R.dataflow():
                lv1: R.Tensor((3, 3), dtype="float32") = R.add(x, y)
                lv2: R.Tensor((3, 3), dtype="float32") = R.add(lv1, z)
                gv: R.Tensor((), dtype="float32") = R.sum(lv2, axis=None, keepdims=False)
                R.output(gv)
            return gv
    # fmt: on

    After1 = relax.transform.Gradient("main")(Before)
    assert_structural_equal(After1, Expected1)

    # fmt: off
    @I.ir_module
    class Expected2:
        @R.function
        def main_adjoint(x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32"), z: R.Tensor((3, 3), dtype="float32")) -> R.Tuple(R.Tensor((), dtype="float32"), R.Tuple(R.Tensor((3, 3), dtype="float32"))):
            with R.dataflow():
                lv1: R.Tensor((3, 3), dtype="float32") = R.add(x, y)
                lv2: R.Tensor((3, 3), dtype="float32") = R.add(lv1, z)
                gv: R.Tensor((), dtype="float32") = R.sum(lv2, axis=None, keepdims=False)
                gv_adjoint: R.Tensor((), dtype="float32") = R.ones(R.shape([]), dtype="float32")
                lv2_adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(gv_adjoint, R.shape([3, 3]))
                lv1_adjoint: R.Tensor((3, 3), dtype="float32") = lv2_adjoint
                x_adjoint: R.Tensor((3, 3), dtype="float32") = lv1_adjoint
                x_adjoint_out: R.Tensor((3, 3), dtype="float32") = x_adjoint
                R.output(gv, x_adjoint_out)
            return (gv, (x_adjoint_out,))

        @R.function
        def main(x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32"), z: R.Tensor((3, 3), dtype="float32")) -> R.Tensor((), dtype="float32"):
            with R.dataflow():
                lv1: R.Tensor((3, 3), dtype="float32") = R.add(x, y)
                lv2: R.Tensor((3, 3), dtype="float32") = R.add(lv1, z)
                gv: R.Tensor((), dtype="float32") = R.sum(lv2, axis=None, keepdims=False)
                R.output(gv)
            return gv
    # fmt: on

    After2 = relax.transform.Gradient("main", require_grads=Before["main"].params[0])(Before)
    assert_structural_equal(After2, Expected2)


def test_target_index():
    # fmt: off
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((3, 3), "float32"), y: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                lv1 = x
                lv2 = R.sum(x)
                lv3 = R.sum(y)
                R.output(lv1, lv2, lv3)
            return (lv1, lv2, lv3)

    @I.ir_module
    class Expected:
        @R.function
        def main_adjoint(x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")) -> R.Tuple(R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((), dtype="float32"), R.Tensor((), dtype="float32")), R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32"))):
            with R.dataflow():
                lv1: R.Tensor((3, 3), dtype="float32") = x
                lv2: R.Tensor((), dtype="float32") = R.sum(x, axis=None, keepdims=False)
                lv3: R.Tensor((), dtype="float32") = R.sum(y, axis=None, keepdims=False)
                lv3_adjoint: R.Tensor((), dtype="float32") = R.ones(R.shape([]), dtype="float32")
                y_adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(lv3_adjoint, R.shape([3, 3]))
                x_adjoint: R.Tensor((3, 3), dtype="float32") = R.zeros(R.shape([3, 3]), dtype="float32")
                x_adjoint_out: R.Tensor((3, 3), dtype="float32") = x_adjoint
                y_adjoint_out: R.Tensor((3, 3), dtype="float32") = y_adjoint
                R.output(lv1, lv2, lv3, x_adjoint_out, y_adjoint_out)
            return ((lv1, lv2, lv3), (x_adjoint_out, y_adjoint_out))

        @R.function
        def main(x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")) -> R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((), dtype="float32"), R.Tensor((), dtype="float32")):
            with R.dataflow():
                lv1: R.Tensor((3, 3), dtype="float32") = x
                lv2: R.Tensor((), dtype="float32") = R.sum(x, axis=None, keepdims=False)
                lv3: R.Tensor((), dtype="float32") = R.sum(y, axis=None, keepdims=False)
                R.output(lv1, lv2, lv3)
            return (lv1, lv2, lv3)
    # fmt: on

    After = relax.transform.Gradient("main", target_index=2)(Before)
    assert_structural_equal(After, Expected)


def test_intermediate_var_require_grads():
    x = relax.Var("x", R.Tensor((3, 3), "float32"))
    y = relax.Var("y", R.Tensor((3, 3), "float32"))

    bb = relax.BlockBuilder()
    with bb.function("main", [x, y]):
        with bb.dataflow():
            lv0 = bb.emit(x * x)
            lv1 = bb.emit(lv0 * y)
            lv2 = bb.emit(lv1 * y)
            gv0 = bb.emit_output(relax.op.sum(lv2))
        bb.emit_func_output(gv0)

    Before = bb.get()

    # fmt: off
    @I.ir_module
    class Expected:
        @R.function
        def main_adjoint(x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")) -> R.Tuple(R.Tensor((), dtype="float32"), R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32"), R.Tensor((), dtype="float32"))):
            with R.dataflow():
                lv: R.Tensor((3, 3), dtype="float32") = R.multiply(x, x)
                lv1: R.Tensor((3, 3), dtype="float32") = R.multiply(lv, y)
                lv2: R.Tensor((3, 3), dtype="float32") = R.multiply(lv1, y)
                gv: R.Tensor((), dtype="float32") = R.sum(lv2, axis=None, keepdims=False)
                gv_adjoint: R.Tensor((), dtype="float32") = R.ones(R.shape([]), dtype="float32")
                lv2_adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(gv_adjoint, R.shape([3, 3]))
                lv1_adjoint: R.Tensor((3, 3), dtype="float32") = R.multiply(lv2_adjoint, y)
                lv_adjoint: R.Tensor((3, 3), dtype="float32") = R.multiply(lv1_adjoint, y)
                x_adjoint: R.Tensor((3, 3), dtype="float32") = R.multiply(lv_adjoint, x)
                lv1_1: R.Tensor((3, 3), dtype="float32") = R.multiply(lv_adjoint, x)
                x_adjoint1: R.Tensor((3, 3), dtype="float32") = R.add(x_adjoint, lv1_1)
                x_adjoint_out: R.Tensor((3, 3), dtype="float32") = x_adjoint1
                lv1_adjoint_out: R.Tensor((3, 3), dtype="float32") = lv1_adjoint
                gv_adjoint_out: R.Tensor((), dtype="float32") = gv_adjoint
                R.output(gv, x_adjoint_out, lv1_adjoint_out, gv_adjoint_out)
            return (gv, (x_adjoint_out, lv1_adjoint_out, gv_adjoint_out))

        @R.function
        def main(x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")) -> R.Tensor((), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((3, 3), dtype="float32") = R.multiply(x, x)
                lv1: R.Tensor((3, 3), dtype="float32") = R.multiply(lv, y)
                lv2: R.Tensor((3, 3), dtype="float32") = R.multiply(lv1, y)
                gv: R.Tensor((), dtype="float32") = R.sum(lv2, axis=None, keepdims=False)
                R.output(gv)
            return gv
    # fmt: on

    After = relax.transform.Gradient("main", [x, lv1, gv0])(Before)
    assert_structural_equal(After, Expected)

    # z does not occur in function
    z = relax.Var("z", R.Tensor((3, 3), "float32"))
    with pytest.raises(RuntimeError):
        relax.transform.Gradient("main", [x, lv1, z])(Before)


def test_tuple():
    # fmt: off
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3, 3), "float32")),
            y: R.Tensor((3, 3), "float32"),
            z: R.Tensor((3, 3), "float32"),
        ):
            with R.dataflow():
                lv1 = (y, z)
                lv2 = x[0]
                lv3 = lv1[0]
                lv4 = R.add(lv2, lv3)
                gv = R.sum(lv4)
                R.output(gv)
            return gv


    @I.ir_module
    class Expected:
        @R.function
        def main_adjoint(x: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")), y: R.Tensor((3, 3), dtype="float32"), z: R.Tensor((3, 3), dtype="float32")) -> R.Tuple(R.Tensor((), dtype="float32"), R.Tuple(R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")), R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32"))):
            with R.dataflow():
                lv1: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = (y, z)
                lv2: R.Tensor((3, 3), dtype="float32") = x[0]
                lv3: R.Tensor((3, 3), dtype="float32") = lv1[0]
                lv4: R.Tensor((3, 3), dtype="float32") = R.add(lv2, lv3)
                gv: R.Tensor((), dtype="float32") = R.sum(lv4, axis=None, keepdims=False)
                gv_adjoint: R.Tensor((), dtype="float32") = R.ones(R.shape([]), dtype="float32")
                lv4_adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(gv_adjoint, R.shape([3, 3]))
                lv2_adjoint: R.Tensor((3, 3), dtype="float32") = lv4_adjoint
                lv3_adjoint: R.Tensor((3, 3), dtype="float32") = lv4_adjoint
                lv: R.Tensor((3, 3), dtype="float32") = R.zeros(R.shape([3, 3]), dtype="float32")
                lv1_adjoint: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = (lv3_adjoint, lv)
                lv1_1: R.Tensor((3, 3), dtype="float32") = R.zeros(R.shape([3, 3]), dtype="float32")
                x_adjoint: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = (lv2_adjoint, lv1_1)
                y_adjoint: R.Tensor((3, 3), dtype="float32") = lv1_adjoint[0]
                z_adjoint: R.Tensor((3, 3), dtype="float32") = lv1_adjoint[1]
                x_adjoint_out: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = x_adjoint
                y_adjoint_out: R.Tensor((3, 3), dtype="float32") = y_adjoint
                z_adjoint_out: R.Tensor((3, 3), dtype="float32") = z_adjoint
                R.output(gv, x_adjoint_out, y_adjoint_out, z_adjoint_out)
            return (gv, (x_adjoint_out, y_adjoint_out, z_adjoint_out))

        @R.function
        def main(x: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")), y: R.Tensor((3, 3), dtype="float32"), z: R.Tensor((3, 3), dtype="float32")) -> R.Tensor((), dtype="float32"):
            with R.dataflow():
                lv1: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = (y, z)
                lv2: R.Tensor((3, 3), dtype="float32") = x[0]
                lv3: R.Tensor((3, 3), dtype="float32") = lv1[0]
                lv4: R.Tensor((3, 3), dtype="float32") = R.add(lv2, lv3)
                gv: R.Tensor((), dtype="float32") = R.sum(lv4, axis=None, keepdims=False)
                R.output(gv)
            return gv
    # fmt: on

    After = relax.transform.Gradient("main")(Before)
    assert_structural_equal(After, Expected)


def test_tuple_assignment():
    # fmt: off
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((3, 3), "float32"), y: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                lv1 = (x, y)
                lv2 = lv1[0]
                lv3 = R.add(lv2, x)
                lv4 = lv1
                lv5 = lv4[0]
                lv6 = R.add(lv5, lv3)
                gv = R.sum(lv6)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main_adjoint(x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")) -> R.Tuple(R.Tensor((), dtype="float32"), R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32"))):
            with R.dataflow():
                lv1: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = (x, y)
                lv2: R.Tensor((3, 3), dtype="float32") = lv1[0]
                lv3: R.Tensor((3, 3), dtype="float32") = R.add(lv2, x)
                lv4: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = lv1
                lv5: R.Tensor((3, 3), dtype="float32") = lv4[0]
                lv6: R.Tensor((3, 3), dtype="float32") = R.add(lv5, lv3)
                gv: R.Tensor((), dtype="float32") = R.sum(lv6, axis=None, keepdims=False)
                gv_adjoint: R.Tensor((), dtype="float32") = R.ones(R.shape([]), dtype="float32")
                lv6_adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(gv_adjoint, R.shape([3, 3]))
                lv5_adjoint: R.Tensor((3, 3), dtype="float32") = lv6_adjoint
                lv3_adjoint: R.Tensor((3, 3), dtype="float32") = lv6_adjoint
                lv: R.Tensor((3, 3), dtype="float32") = R.zeros(R.shape([3, 3]), dtype="float32")
                lv4_adjoint: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = (lv5_adjoint, lv)
                lv1_adjoint: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = lv4_adjoint
                lv2_adjoint: R.Tensor((3, 3), dtype="float32") = lv3_adjoint
                x_adjoint: R.Tensor((3, 3), dtype="float32") = lv3_adjoint
                lv1_1: R.Tensor((3, 3), dtype="float32") = lv1_adjoint[0]
                lv2_1: R.Tensor((3, 3), dtype="float32") = R.add(lv1_1, lv2_adjoint)
                lv3_1: R.Tensor((3, 3), dtype="float32") = lv1_adjoint[1]
                lv1_adjoint1: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = (lv2_1, lv3_1)
                lv4_1: R.Tensor((3, 3), dtype="float32") = lv1_adjoint1[0]
                x_adjoint1: R.Tensor((3, 3), dtype="float32") = R.add(x_adjoint, lv4_1)
                y_adjoint: R.Tensor((3, 3), dtype="float32") = lv1_adjoint1[1]
                x_adjoint_out: R.Tensor((3, 3), dtype="float32") = x_adjoint1
                y_adjoint_out: R.Tensor((3, 3), dtype="float32") = y_adjoint
                R.output(gv, x_adjoint_out, y_adjoint_out)
            return (gv, (x_adjoint_out, y_adjoint_out))

        @R.function
        def main(x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")) -> R.Tensor((), dtype="float32"):
            with R.dataflow():
                lv1: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = (x, y)
                lv2: R.Tensor((3, 3), dtype="float32") = lv1[0]
                lv3: R.Tensor((3, 3), dtype="float32") = R.add(lv2, x)
                lv4: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = lv1
                lv5: R.Tensor((3, 3), dtype="float32") = lv4[0]
                lv6: R.Tensor((3, 3), dtype="float32") = R.add(lv5, lv3)
                gv: R.Tensor((), dtype="float32") = R.sum(lv6, axis=None, keepdims=False)
                R.output(gv)
            return gv
    # fmt: on

    After = relax.transform.Gradient("main")(Before)
    assert_structural_equal(After, Expected)


def test_tuple_nested():
    # fmt: off
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tuple(R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3, 3), "float32")), R.Tensor((3, 3), "float32")),
            y: R.Tensor((3, 3), "float32"),
            z: R.Tensor((3, 3), "float32"),
            u: R.Tensor((3, 3), "float32"),
     ):
            with R.dataflow():
                lv1 = ((y, z), u)
                lv2 = x[0]
                lv3 = lv2[0]
                lv4 = lv1[0]
                lv5 = lv4[1]
                lv6 = R.add(lv3, lv5)
                lv7 = x[1]
                lv8 = R.add(lv6, lv7)
                gv = R.sum(lv8)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main_adjoint(x: R.Tuple(R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")), R.Tensor((3, 3), dtype="float32")), y: R.Tensor((3, 3), dtype="float32"), z: R.Tensor((3, 3), dtype="float32"), u: R.Tensor((3, 3), dtype="float32")) -> R.Tuple(R.Tensor((), dtype="float32"), R.Tuple(R.Tuple(R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")), R.Tensor((3, 3), dtype="float32")), R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32"))):
            with R.dataflow():
                lv1: R.Tuple(R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")), R.Tensor((3, 3), dtype="float32")) = ((y, z), u)
                lv2: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = x[0]
                lv3: R.Tensor((3, 3), dtype="float32") = lv2[0]
                lv4: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = lv1[0]
                lv5: R.Tensor((3, 3), dtype="float32") = lv4[1]
                lv6: R.Tensor((3, 3), dtype="float32") = R.add(lv3, lv5)
                lv7: R.Tensor((3, 3), dtype="float32") = x[1]
                lv8: R.Tensor((3, 3), dtype="float32") = R.add(lv6, lv7)
                gv: R.Tensor((), dtype="float32") = R.sum(lv8, axis=None, keepdims=False)
                gv_adjoint: R.Tensor((), dtype="float32") = R.ones(R.shape([]), dtype="float32")
                lv8_adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(gv_adjoint, R.shape([3, 3]))
                lv6_adjoint: R.Tensor((3, 3), dtype="float32") = lv8_adjoint
                lv7_adjoint: R.Tensor((3, 3), dtype="float32") = lv8_adjoint
                lv: R.Tensor((3, 3), dtype="float32") = R.zeros(R.shape([3, 3]), dtype="float32")
                lv1_1: R.Tensor((3, 3), dtype="float32") = R.zeros(R.shape([3, 3]), dtype="float32")
                x_adjoint: R.Tuple(R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")), R.Tensor((3, 3), dtype="float32")) = ((lv, lv1_1), lv7_adjoint)
                lv3_adjoint: R.Tensor((3, 3), dtype="float32") = lv6_adjoint
                lv5_adjoint: R.Tensor((3, 3), dtype="float32") = lv6_adjoint
                lv2_1: R.Tensor((3, 3), dtype="float32") = R.zeros(R.shape([3, 3]), dtype="float32")
                lv4_adjoint: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = (lv2_1, lv5_adjoint)
                lv3_1: R.Tensor((3, 3), dtype="float32") = R.zeros(R.shape([3, 3]), dtype="float32")
                lv1_adjoint: R.Tuple(R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")), R.Tensor((3, 3), dtype="float32")) = (lv4_adjoint, lv3_1)
                lv4_1: R.Tensor((3, 3), dtype="float32") = R.zeros(R.shape([3, 3]), dtype="float32")
                lv2_adjoint: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = (lv3_adjoint, lv4_1)
                lv5_1: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = x_adjoint[0]
                lv6_1: R.Tensor((3, 3), dtype="float32") = lv5_1[0]
                lv7_1: R.Tensor((3, 3), dtype="float32") = lv2_adjoint[0]
                lv8_1: R.Tensor((3, 3), dtype="float32") = R.add(lv6_1, lv7_1)
                lv9: R.Tensor((3, 3), dtype="float32") = lv5_1[1]
                lv10: R.Tensor((3, 3), dtype="float32") = lv2_adjoint[1]
                lv11: R.Tensor((3, 3), dtype="float32") = R.add(lv9, lv10)
                lv12: R.Tensor((3, 3), dtype="float32") = x_adjoint[1]
                x_adjoint1: R.Tuple(R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")), R.Tensor((3, 3), dtype="float32")) = ((lv8_1, lv11), lv12)
                lv13: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = lv1_adjoint[0]
                y_adjoint: R.Tensor((3, 3), dtype="float32") = lv13[0]
                z_adjoint: R.Tensor((3, 3), dtype="float32") = lv13[1]
                u_adjoint: R.Tensor((3, 3), dtype="float32") = lv1_adjoint[1]
                x_adjoint_out: R.Tuple(R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")), R.Tensor((3, 3), dtype="float32")) = x_adjoint1
                y_adjoint_out: R.Tensor((3, 3), dtype="float32") = y_adjoint
                z_adjoint_out: R.Tensor((3, 3), dtype="float32") = z_adjoint
                u_adjoint_out: R.Tensor((3, 3), dtype="float32") = u_adjoint
                R.output(gv, x_adjoint_out, y_adjoint_out, z_adjoint_out, u_adjoint_out)
            return (gv, (x_adjoint_out, y_adjoint_out, z_adjoint_out, u_adjoint_out))

        @R.function
        def main(x: R.Tuple(R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")), R.Tensor((3, 3), dtype="float32")), y: R.Tensor((3, 3), dtype="float32"), z: R.Tensor((3, 3), dtype="float32"), u: R.Tensor((3, 3), dtype="float32")) -> R.Tensor((), dtype="float32"):
            with R.dataflow():
                lv1: R.Tuple(R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")), R.Tensor((3, 3), dtype="float32")) = ((y, z), u)
                lv2: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = x[0]
                lv3: R.Tensor((3, 3), dtype="float32") = lv2[0]
                lv4: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = lv1[0]
                lv5: R.Tensor((3, 3), dtype="float32") = lv4[1]
                lv6: R.Tensor((3, 3), dtype="float32") = R.add(lv3, lv5)
                lv7: R.Tensor((3, 3), dtype="float32") = x[1]
                lv8: R.Tensor((3, 3), dtype="float32") = R.add(lv6, lv7)
                gv: R.Tensor((), dtype="float32") = R.sum(lv8, axis=None, keepdims=False)
                R.output(gv)
            return gv
    # fmt: on

    After = relax.transform.Gradient("main")(Before)
    assert_structural_equal(After, Expected)


def test_tuple_update():
    """One tensor `x` is used in and out of tuple many times."""

    # fmt: off
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((3, 3), "float32"), y: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                lv0 = (x, y)
                lv1 = R.add(x, y)
                lv2 = lv0[0]
                lv3 = R.add(lv2, y)
                lv4 = R.add(lv1, lv3)
                lv5 = (x, y)
                lv6 = lv5[0]
                lv7 = lv0[0]
                lv8 = R.add(lv4, lv6)
                lv9 = R.add(lv8, lv7)
                gv = R.sum(lv9)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main_adjoint(x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")) -> R.Tuple(R.Tensor((), dtype="float32"), R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32"))):
            with R.dataflow():
                lv0: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = (x, y)
                lv1: R.Tensor((3, 3), dtype="float32") = R.add(x, y)
                lv2: R.Tensor((3, 3), dtype="float32") = lv0[0]
                lv3: R.Tensor((3, 3), dtype="float32") = R.add(lv2, y)
                lv4: R.Tensor((3, 3), dtype="float32") = R.add(lv1, lv3)
                lv5: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = (x, y)
                lv6: R.Tensor((3, 3), dtype="float32") = lv5[0]
                lv7: R.Tensor((3, 3), dtype="float32") = lv0[0]
                lv8: R.Tensor((3, 3), dtype="float32") = R.add(lv4, lv6)
                lv9: R.Tensor((3, 3), dtype="float32") = R.add(lv8, lv7)
                gv: R.Tensor((), dtype="float32") = R.sum(lv9, axis=None, keepdims=False)
                gv_adjoint: R.Tensor((), dtype="float32") = R.ones(R.shape([]), dtype="float32")
                lv9_adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(gv_adjoint, R.shape([3, 3]))
                lv8_adjoint: R.Tensor((3, 3), dtype="float32") = lv9_adjoint
                lv7_adjoint: R.Tensor((3, 3), dtype="float32") = lv9_adjoint
                lv4_adjoint: R.Tensor((3, 3), dtype="float32") = lv8_adjoint
                lv6_adjoint: R.Tensor((3, 3), dtype="float32") = lv8_adjoint
                lv: R.Tensor((3, 3), dtype="float32") = R.zeros(R.shape([3, 3]), dtype="float32")
                lv0_adjoint: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = (lv7_adjoint, lv)
                lv1_1: R.Tensor((3, 3), dtype="float32") = R.zeros(R.shape([3, 3]), dtype="float32")
                lv5_adjoint: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = (lv6_adjoint, lv1_1)
                x_adjoint: R.Tensor((3, 3), dtype="float32") = lv5_adjoint[0]
                y_adjoint: R.Tensor((3, 3), dtype="float32") = lv5_adjoint[1]
                lv1_adjoint: R.Tensor((3, 3), dtype="float32") = lv4_adjoint
                lv3_adjoint: R.Tensor((3, 3), dtype="float32") = lv4_adjoint
                lv2_adjoint: R.Tensor((3, 3), dtype="float32") = lv3_adjoint
                y_adjoint1: R.Tensor((3, 3), dtype="float32") = R.add(y_adjoint, lv3_adjoint)
                lv2_1: R.Tensor((3, 3), dtype="float32") = lv0_adjoint[0]
                lv3_1: R.Tensor((3, 3), dtype="float32") = R.add(lv2_1, lv2_adjoint)
                lv4_1: R.Tensor((3, 3), dtype="float32") = lv0_adjoint[1]
                lv0_adjoint1: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = (lv3_1, lv4_1)
                x_adjoint1: R.Tensor((3, 3), dtype="float32") = R.add(x_adjoint, lv1_adjoint)
                y_adjoint2: R.Tensor((3, 3), dtype="float32") = R.add(y_adjoint1, lv1_adjoint)
                lv5_1: R.Tensor((3, 3), dtype="float32") = lv0_adjoint1[0]
                x_adjoint2: R.Tensor((3, 3), dtype="float32") = R.add(x_adjoint1, lv5_1)
                lv6_1: R.Tensor((3, 3), dtype="float32") = lv0_adjoint1[1]
                y_adjoint3: R.Tensor((3, 3), dtype="float32") = R.add(y_adjoint2, lv6_1)
                x_adjoint_out: R.Tensor((3, 3), dtype="float32") = x_adjoint2
                y_adjoint_out: R.Tensor((3, 3), dtype="float32") = y_adjoint3
                R.output(gv, x_adjoint_out, y_adjoint_out)
            return (gv, (x_adjoint_out, y_adjoint_out))

        @R.function
        def main(x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")) -> R.Tensor((), dtype="float32"):
            with R.dataflow():
                lv0: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = (x, y)
                lv1: R.Tensor((3, 3), dtype="float32") = R.add(x, y)
                lv2: R.Tensor((3, 3), dtype="float32") = lv0[0]
                lv3: R.Tensor((3, 3), dtype="float32") = R.add(lv2, y)
                lv4: R.Tensor((3, 3), dtype="float32") = R.add(lv1, lv3)
                lv5: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = (x, y)
                lv6: R.Tensor((3, 3), dtype="float32") = lv5[0]
                lv7: R.Tensor((3, 3), dtype="float32") = lv0[0]
                lv8: R.Tensor((3, 3), dtype="float32") = R.add(lv4, lv6)
                lv9: R.Tensor((3, 3), dtype="float32") = R.add(lv8, lv7)
                gv: R.Tensor((), dtype="float32") = R.sum(lv9, axis=None, keepdims=False)
                R.output(gv)
            return gv
    # fmt: on

    After = relax.transform.Gradient("main")(Before)
    assert_structural_equal(After, Expected)


def test_tuple_op_simple():
    # fmt: off
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((6,), "float32")):
            with R.dataflow():
                lv1 = R.split(x, 2)
                lv2 = R.concat(lv1)
                gv = R.sum(lv2)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main_adjoint(x: R.Tensor((6,), dtype="float32")) -> R.Tuple(R.Tensor((), dtype="float32"), R.Tuple(R.Tensor((6,), dtype="float32"))):
            with R.dataflow():
                lv1: R.Tuple(R.Tensor((3,), dtype="float32"), R.Tensor((3,), dtype="float32")) = R.split(x, indices_or_sections=2, axis=0)
                lv2: R.Tensor((6,), dtype="float32") = R.concat(lv1, axis=0)
                gv: R.Tensor((), dtype="float32") = R.sum(lv2, axis=None, keepdims=False)
                gv_adjoint: R.Tensor((), dtype="float32") = R.ones(R.shape([]), dtype="float32")
                lv2_adjoint: R.Tensor((6,), dtype="float32") = R.broadcast_to(gv_adjoint, R.shape([6]))
                lv1_adjoint: R.Tuple(R.Tensor((3,), dtype="float32"), R.Tensor((3,), dtype="float32")) = R.split(lv2_adjoint, indices_or_sections=[3], axis=0)
                x_adjoint: R.Tensor((6,), dtype="float32") = R.concat(lv1_adjoint, axis=0)
                x_adjoint_out: R.Tensor((6,), dtype="float32") = x_adjoint
                R.output(gv, x_adjoint_out)
            return (gv, (x_adjoint_out,))

        @R.function
        def main(x: R.Tensor((6,), dtype="float32")) -> R.Tensor((), dtype="float32"):
            with R.dataflow():
                lv1: R.Tuple(R.Tensor((3,), dtype="float32"), R.Tensor((3,), dtype="float32")) = R.split(x, indices_or_sections=2, axis=0)
                lv2: R.Tensor((6,), dtype="float32") = R.concat(lv1, axis=0)
                gv: R.Tensor((), dtype="float32") = R.sum(lv2, axis=None, keepdims=False)
                R.output(gv)
            return gv
    # fmt: on

    After = relax.transform.Gradient("main")(Before)
    assert_structural_equal(After, Expected)


def test_tuple_op_construct():
    # fmt: off
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((3,), "float32"), y: R.Tuple(R.Tensor((3, ), "float32"), R.Tensor((3, ), "float32")),):
            with R.dataflow():
                lv1 = (x, x)
                lv2 = R.concat(lv1)
                lv3 = R.concat((x, x))
                lv4 = R.concat(y)
                lv5 = R.add(lv2, lv3)
                lv6 = R.add(lv5, lv4)
                gv = R.sum(lv6)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main_adjoint(x: R.Tensor((3,), dtype="float32"), y: R.Tuple(R.Tensor((3,), dtype="float32"), R.Tensor((3,), dtype="float32"))) -> R.Tuple(R.Tensor((), dtype="float32"), R.Tuple(R.Tensor((3,), dtype="float32"), R.Tuple(R.Tensor((3,), dtype="float32"), R.Tensor((3,), dtype="float32")))):
            with R.dataflow():
                lv1: R.Tuple(R.Tensor((3,), dtype="float32"), R.Tensor((3,), dtype="float32")) = (x, x)
                lv2: R.Tensor((6,), dtype="float32") = R.concat(lv1, axis=0)
                lv3: R.Tensor((6,), dtype="float32") = R.concat((x, x), axis=0)
                lv4: R.Tensor((6,), dtype="float32") = R.concat(y, axis=0)
                lv5: R.Tensor((6,), dtype="float32") = R.add(lv2, lv3)
                lv6: R.Tensor((6,), dtype="float32") = R.add(lv5, lv4)
                gv: R.Tensor((), dtype="float32") = R.sum(lv6, axis=None, keepdims=False)
                gv_adjoint: R.Tensor((), dtype="float32") = R.ones(R.shape([]), dtype="float32")
                lv6_adjoint: R.Tensor((6,), dtype="float32") = R.broadcast_to(gv_adjoint, R.shape([6]))
                lv5_adjoint: R.Tensor((6,), dtype="float32") = lv6_adjoint
                lv4_adjoint: R.Tensor((6,), dtype="float32") = lv6_adjoint
                lv2_adjoint: R.Tensor((6,), dtype="float32") = lv5_adjoint
                lv3_adjoint: R.Tensor((6,), dtype="float32") = lv5_adjoint
                y_adjoint: R.Tuple(R.Tensor((3,), dtype="float32"), R.Tensor((3,), dtype="float32")) = R.split(lv4_adjoint, indices_or_sections=[3], axis=0)
                lv: R.Tuple(R.Tensor((3,), dtype="float32"), R.Tensor((3,), dtype="float32")) = R.split(lv3_adjoint, indices_or_sections=[3], axis=0)
                x_adjoint: R.Tensor((3,), dtype="float32") = lv[0]
                lv1_1: R.Tensor((3,), dtype="float32") = lv[1]
                x_adjoint1: R.Tensor((3,), dtype="float32") = R.add(x_adjoint, lv1_1)
                lv1_adjoint: R.Tuple(R.Tensor((3,), dtype="float32"), R.Tensor((3,), dtype="float32")) = R.split(lv2_adjoint, indices_or_sections=[3], axis=0)
                lv2_1: R.Tensor((3,), dtype="float32") = lv1_adjoint[0]
                x_adjoint2: R.Tensor((3,), dtype="float32") = R.add(x_adjoint1, lv2_1)
                lv3_1: R.Tensor((3,), dtype="float32") = lv1_adjoint[1]
                x_adjoint3: R.Tensor((3,), dtype="float32") = R.add(x_adjoint2, lv3_1)
                x_adjoint_out: R.Tensor((3,), dtype="float32") = x_adjoint3
                y_adjoint_out: R.Tuple(R.Tensor((3,), dtype="float32"), R.Tensor((3,), dtype="float32")) = y_adjoint
                R.output(gv, x_adjoint_out, y_adjoint_out)
            return (gv, (x_adjoint_out, y_adjoint_out))

        @R.function
        def main(x: R.Tensor((3,), dtype="float32"), y: R.Tuple(R.Tensor((3,), dtype="float32"), R.Tensor((3,), dtype="float32"))) -> R.Tensor((), dtype="float32"):
            with R.dataflow():
                lv1: R.Tuple(R.Tensor((3,), dtype="float32"), R.Tensor((3,), dtype="float32")) = (x, x)
                lv2: R.Tensor((6,), dtype="float32") = R.concat(lv1, axis=0)
                lv3: R.Tensor((6,), dtype="float32") = R.concat((x, x), axis=0)
                lv4: R.Tensor((6,), dtype="float32") = R.concat(y, axis=0)
                lv5: R.Tensor((6,), dtype="float32") = R.add(lv2, lv3)
                lv6: R.Tensor((6,), dtype="float32") = R.add(lv5, lv4)
                gv: R.Tensor((), dtype="float32") = R.sum(lv6, axis=None, keepdims=False)
                R.output(gv)
            return gv
    # fmt: on

    After = relax.transform.Gradient("main")(Before)
    assert_structural_equal(After, Expected)


def test_tuple_op_const():
    c1 = R.const(np.zeros(3).astype(np.float32))
    c2 = R.const(np.zeros(3).astype(np.float32))
    c3 = R.const(np.zeros(3).astype(np.float32))

    # fmt: off
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((3,), "float32")):
            with R.dataflow():
                lv1 = R.concat((c1, c2))
                lv2 = R.concat((c3, x))
                lv3 = R.concat((x, x))
                lv4 = R.add(lv1, lv2)
                lv5 = R.add(lv4, lv3)
                gv = R.sum(lv5)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main_adjoint(x: R.Tensor((3,), dtype="float32")) -> R.Tuple(R.Tensor((), dtype="float32"), R.Tuple(R.Tensor((3,), dtype="float32"))):
            with R.dataflow():
                lv1: R.Tensor((6,), dtype="float32") = R.concat((c1, c2), axis=0)
                lv2: R.Tensor((6,), dtype="float32") = R.concat((c3, x), axis=0)
                lv3: R.Tensor((6,), dtype="float32") = R.concat((x, x), axis=0)
                lv4: R.Tensor((6,), dtype="float32") = R.add(lv1, lv2)
                lv5: R.Tensor((6,), dtype="float32") = R.add(lv4, lv3)
                gv: R.Tensor((), dtype="float32") = R.sum(lv5, axis=None, keepdims=False)
                gv_adjoint: R.Tensor((), dtype="float32") = R.ones(R.shape([]), dtype="float32")
                lv5_adjoint: R.Tensor((6,), dtype="float32") = R.broadcast_to(gv_adjoint, R.shape([6]))
                lv4_adjoint: R.Tensor((6,), dtype="float32") = lv5_adjoint
                lv3_adjoint: R.Tensor((6,), dtype="float32") = lv5_adjoint
                lv2_adjoint: R.Tensor((6,), dtype="float32") = lv4_adjoint
                lv: R.Tuple(R.Tensor((3,), dtype="float32"), R.Tensor((3,), dtype="float32")) = R.split(lv3_adjoint, indices_or_sections=[3], axis=0)
                x_adjoint: R.Tensor((3,), dtype="float32") = lv[0]
                lv1_1: R.Tensor((3,), dtype="float32") = lv[1]
                x_adjoint1: R.Tensor((3,), dtype="float32") = R.add(x_adjoint, lv1_1)
                lv2_1: R.Tuple(R.Tensor((3,), dtype="float32"), R.Tensor((3,), dtype="float32")) = R.split(lv2_adjoint, indices_or_sections=[3], axis=0)
                lv3_1: R.Tensor((3,), dtype="float32") = lv2_1[1]
                x_adjoint2: R.Tensor((3,), dtype="float32") = R.add(x_adjoint1, lv3_1)
                x_adjoint_out: R.Tensor((3,), dtype="float32") = x_adjoint2
                R.output(gv, x_adjoint_out)
            return (gv, (x_adjoint_out,))

        @R.function
        def main(x: R.Tensor((3,), dtype="float32")) -> R.Tensor((), dtype="float32"):
            with R.dataflow():
                lv1: R.Tensor((6,), dtype="float32") = R.concat((c1, c2), axis=0)
                lv2: R.Tensor((6,), dtype="float32") = R.concat((c3, x), axis=0)
                lv3: R.Tensor((6,), dtype="float32") = R.concat((x, x), axis=0)
                lv4: R.Tensor((6,), dtype="float32") = R.add(lv1, lv2)
                lv5: R.Tensor((6,), dtype="float32") = R.add(lv4, lv3)
                gv: R.Tensor((), dtype="float32") = R.sum(lv5, axis=None, keepdims=False)
                R.output(gv)
            return gv
    # fmt: on

    After = relax.transform.Gradient("main")(Before)
    assert_structural_equal(After["main_adjoint"], Expected["main_adjoint"])


def test_const():
    """const could be used in variable assignment, call argument, and as a part of tuple"""
    cst = relax.const(np.ones((3, 3)), "float32")

    # fmt: off
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((3, 3), "float32"), y: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                lv1 = R.add(x, cst)
                lv2 = cst
                lv3 = (cst, (cst, lv1))
                lv4 = lv3[1]
                lv5 = lv4[1]
                lv6 = R.subtract(lv5, lv2)
                gv = R.sum(lv6)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main_adjoint(x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")) -> R.Tuple(R.Tensor((), dtype="float32"), R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32"))):
            with R.dataflow():
                lv1: R.Tensor((3, 3), dtype="float32") = R.add(x, cst)
                lv2: R.Tensor((3, 3), dtype="float32") = cst
                lv3: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32"))) = (cst, (cst, lv1))
                lv4: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = lv3[1]
                lv5: R.Tensor((3, 3), dtype="float32") = lv4[1]
                lv6: R.Tensor((3, 3), dtype="float32") = R.subtract(lv5, lv2)
                gv: R.Tensor((), dtype="float32") = R.sum(lv6, axis=None, keepdims=False)
                gv_adjoint: R.Tensor((), dtype="float32") = R.ones(R.shape([]), dtype="float32")
                lv6_adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(gv_adjoint, R.shape([3, 3]))
                lv5_adjoint: R.Tensor((3, 3), dtype="float32") = lv6_adjoint
                lv: R.Tensor((3, 3), dtype="float32") = R.zeros(R.shape([3, 3]), dtype="float32")
                lv4_adjoint: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = (lv, lv5_adjoint)
                lv1_1: R.Tensor((3, 3), dtype="float32") = R.zeros(R.shape([3, 3]), dtype="float32")
                lv3_adjoint: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32"))) = (lv1_1, lv4_adjoint)
                lv2_1: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = lv3_adjoint[1]
                lv1_adjoint: R.Tensor((3, 3), dtype="float32") = lv2_1[1]
                x_adjoint: R.Tensor((3, 3), dtype="float32") = lv1_adjoint
                x_adjoint_out: R.Tensor((3, 3), dtype="float32") = x_adjoint
                y_adjoint: R.Tensor((3, 3), dtype="float32") = R.zeros(R.shape([3, 3]), dtype="float32")
                y_adjoint_out: R.Tensor((3, 3), dtype="float32") = y_adjoint
                R.output(gv, x_adjoint_out, y_adjoint_out)
            return (gv, (x_adjoint_out, y_adjoint_out))

        @R.function
        def main(x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")) -> R.Tensor((), dtype="float32"):
            with R.dataflow():
                lv1: R.Tensor((3, 3), dtype="float32") = R.add(x, cst)
                lv2: R.Tensor((3, 3), dtype="float32") = cst
                lv3: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32"))) = (cst, (cst, lv1))
                lv4: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = lv3[1]
                lv5: R.Tensor((3, 3), dtype="float32") = lv4[1]
                lv6: R.Tensor((3, 3), dtype="float32") = R.subtract(lv5, lv2)
                gv: R.Tensor((), dtype="float32") = R.sum(lv6, axis=None, keepdims=False)
                R.output(gv)
            return gv
    # fmt: on

    After = relax.transform.Gradient("main")(Before)
    assert_structural_equal(After, Expected)


def test_simplify_matmul_pattern():
    # fmt: off
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((3, 3), "float32"), y: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                lv1 = R.permute_dims(x)
                lv2 = R.permute_dims(y)
                lv3 = R.matmul(lv1, lv2, out_dtype="float32")
                gv = R.sum(lv3)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main_adjoint(x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")) -> R.Tuple(R.Tensor((), dtype="float32"), R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32"))):
            with R.dataflow():
                lv1: R.Tensor((3, 3), dtype="float32") = R.permute_dims(x, axes=None)
                lv2: R.Tensor((3, 3), dtype="float32") = R.permute_dims(y, axes=None)
                lv3: R.Tensor((3, 3), dtype="float32") = R.matmul(lv1, lv2, out_dtype="float32")
                gv: R.Tensor((), dtype="float32") = R.sum(lv3, axis=None, keepdims=False)
                gv_adjoint: R.Tensor((), dtype="float32") = R.ones(R.shape([]), dtype="float32")
                lv3_adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(gv_adjoint, R.shape([3, 3]))
                lv: R.Tensor((3, 3), dtype="float32") = R.permute_dims(lv3_adjoint, axes=[1, 0])
                lv1_1: R.Tensor((3, 3), dtype="float32") = R.permute_dims(x, axes=[1, 0])
                y_adjoint: R.Tensor((3, 3), dtype="float32") = R.matmul(lv, lv1_1)
                lv2_1: R.Tensor((3, 3), dtype="float32") = R.permute_dims(y, axes=[1, 0])
                lv3_1: R.Tensor((3, 3), dtype="float32") = R.permute_dims(lv3_adjoint, axes=[1, 0])
                x_adjoint: R.Tensor((3, 3), dtype="float32") = R.matmul(lv2_1, lv3_1)
                x_adjoint_out: R.Tensor((3, 3), dtype="float32") = x_adjoint
                y_adjoint_out: R.Tensor((3, 3), dtype="float32") = y_adjoint
                R.output(gv, x_adjoint_out, y_adjoint_out)
            return (gv, (x_adjoint_out, y_adjoint_out))
        @R.function
        def main(x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")) -> R.Tensor((), dtype="float32"):
            with R.dataflow():
                lv1: R.Tensor((3, 3), dtype="float32") = R.permute_dims(x, axes=None)
                lv2: R.Tensor((3, 3), dtype="float32") = R.permute_dims(y, axes=None)
                lv3: R.Tensor((3, 3), dtype="float32") = R.matmul(lv1, lv2, out_dtype="float32")
                gv: R.Tensor((), dtype="float32") = R.sum(lv3, axis=None, keepdims=False)
                R.output(gv)
            return gv

    # fmt: on

    After = relax.transform.Gradient("main")(Before)
    assert_structural_equal(After, Expected)


def test_shape_expr():
    # fmt: off
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((3, 4), "float32")):
            with R.dataflow():
                s = R.shape([3, 2, 2])
                lv = R.reshape(x, s)
                gv = R.sum(lv)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main_adjoint(x: R.Tensor((3, 4), dtype="float32")) -> R.Tuple(R.Tensor((), dtype="float32"), R.Tuple(R.Tensor((3, 4), dtype="float32"))):
            with R.dataflow():
                s: R.Shape([3, 2, 2]) = R.shape([3, 2, 2])
                lv: R.Tensor((3, 2, 2), dtype="float32") = R.reshape(x, s)
                gv: R.Tensor((), dtype="float32") = R.sum(lv, axis=None, keepdims=False)
                gv_adjoint: R.Tensor((), dtype="float32") = R.ones(R.shape([]), dtype="float32")
                lv_adjoint: R.Tensor((3, 2, 2), dtype="float32") = R.broadcast_to(gv_adjoint, R.shape([3, 2, 2]))
                x_adjoint: R.Tensor((3, 4), dtype="float32") = R.reshape(lv_adjoint, R.shape([3, 4]))
                x_adjoint_out: R.Tensor((3, 4), dtype="float32") = x_adjoint
                R.output(gv, x_adjoint_out)
            return (gv, (x_adjoint_out,))

        @R.function
        def main(x: R.Tensor((3, 4), dtype="float32")) -> R.Tensor((), dtype="float32"):
            with R.dataflow():
                s: R.Shape([3, 2, 2]) = R.shape([3, 2, 2])
                lv: R.Tensor((3, 2, 2), dtype="float32") = R.reshape(x, s)
                gv: R.Tensor((), dtype="float32") = R.sum(lv, axis=None, keepdims=False)
                R.output(gv)
            return gv
    # fmt: on

    After = relax.transform.Gradient("main")(Before)
    assert_structural_equal(After, Expected)


def test_params_copy():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x0: R.Tensor((3, 3), "float32"),
            x1: R.Tensor((3, 3), "float32"),
            x2: R.Tensor((3, 3), "float32"),
            x3: R.Tensor((3, 3), "float32"),
        ):
            with R.dataflow():
                lv0 = R.add(x0, x1)
                lv1 = R.add(x2, x3)
                lv2 = R.add(lv0, lv1)
                gv = R.sum(lv2)
                R.output(gv)
            return gv

    After = relax.transform.Gradient("main")(Before)
    assert len(Before["main"].params) == len(After["main"].params)
    assert len(Before["main"].params) == len(After["main_adjoint"].params)
    for i in range(len(After["main"].params)):
        assert Before["main"].params[i] == After["main"].params[i]
        assert Before["main"].params[i] != After["main_adjoint"].params[i]


def test_function_copy():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x0: R.Tensor((3, 3), "float32"),
            x1: R.Tensor((3, 3), "float32"),
            x2: R.Tensor((3, 3), "float32"),
            x3: R.Tensor((3, 3), "float32"),
        ):
            with R.dataflow():
                lv0 = R.add(x0, x1)
                lv1 = R.add(x2, x3)
                lv2 = R.add(lv0, lv1)
                gv = R.sum(lv2)
                R.output(gv)
            return gv

    After = relax.transform.Gradient("main")(Before)

    # After should have the same "main" function as Before
    assert_structural_equal(Before["main"], After["main"])

    # the first bindings of After["main_adjoint"] should be the same as Before["main"]
    old_bindings = Before["main"].body.blocks[0].bindings
    old_bindings_len = len(old_bindings)
    new_bindings = After["main_adjoint"].body.blocks[0].bindings[:old_bindings_len]
    assert_structural_equal(old_bindings, new_bindings, True)
    relax.analysis.check_well_formed(After)


def test_tir_copy():
    n = T.dynamic("n", "int64")

    @I.ir_module
    class Before:
        @R.function
        def main(
            x0: R.Tensor((n, n), "float32"),
            x1: R.Tensor((n, n), "float32"),
            x2: R.Tensor((n, n), "float32"),
            x3: R.Tensor((n, n), "float32"),
        ):
            with R.dataflow():
                lv0 = R.add(x0, x1)
                lv1 = R.add(x2, x3)
                lv2 = R.add(lv0, lv1)
                gv = R.sum(lv2)
                R.output(gv)
            return gv

    After = relax.transform.Gradient("main")(Before)
    relax.analysis.check_well_formed(After)


def test_report_error():
    @I.ir_module
    class TargetNotTensor:
        @R.function
        def main(x: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                lv1 = R.sum(x)
                gv = (lv1, lv1)
                R.output(gv)
            return gv

    with pytest.raises(RuntimeError):
        relax.transform.Gradient("main")(TargetNotTensor)

    @I.ir_module
    class TargetNotScalar:
        @R.function
        def main(x0: R.Tensor((3, 3), "float32"), x1: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                gv = R.add(x0, x1)
                R.output(gv)
            return gv

    with pytest.raises(RuntimeError):
        relax.transform.Gradient("main")(TargetNotScalar)

    @I.ir_module
    class TargetNotFloat:
        @R.function
        def main(x: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                gv = R.const(1)
                R.output(gv)
            return gv

    with pytest.raises(RuntimeError):
        relax.transform.Gradient("main")(TargetNotFloat)

    @I.ir_module
    class ReturnScalarAndWrongTargetIndex:
        @R.function
        def main(x: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                gv = R.sum(x)
                R.output(gv)
            return gv

    with pytest.raises(RuntimeError):
        relax.transform.Gradient("main", target_index=1)(ReturnScalarAndWrongTargetIndex)

    @I.ir_module
    class ReturnTupleAndWrongTargetIndex:
        @R.function
        def main(x: R.Tensor((3, 3), "float32"), y: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                gv1 = R.sum(x)
                gv2 = R.sum(y)
                R.output(gv1, gv2)
            return gv1, gv2

    with pytest.raises(RuntimeError):
        relax.transform.Gradient("main", target_index=2)(ReturnTupleAndWrongTargetIndex)

    @I.ir_module
    class IndexedTargetNotVar:
        @R.function
        def main(x: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                gv = R.sum(x)
                R.output(gv)
            return gv, (gv, gv)

    with pytest.raises(RuntimeError):
        relax.transform.Gradient("main", target_index=1)(IndexedTargetNotVar)

    @I.ir_module
    class NoDataflow:
        @R.function
        def main(x0: R.Tensor((3, 3), "float32")):
            gv = R.sum(x0)
            return gv

    with pytest.raises(RuntimeError):
        relax.transform.Gradient("main")(NoDataflow)

    @I.ir_module
    class MultiBlocks:
        @R.function
        def main(x0: R.Tensor((3, 3), "float32"), x1: R.Tensor((3, 3), "float32")):
            # block 0
            with R.dataflow():
                gv = R.add(x0, x1)
                R.output(gv)
            # block 1
            gv1 = R.sum(x0)
            return gv1

    with pytest.raises(RuntimeError):
        relax.transform.Gradient("main")(MultiBlocks)

    @I.ir_module
    class NormalModule:
        @R.function
        def main(x0: R.Tensor((3, 3), "float32"), x1: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                gv = R.sum(x0)
                R.output(gv)
            return gv

        @Ts.function
        def sum(
            rxplaceholder: T.Tensor((T.int64(3), T.int64(3)), "float32"),
            rxplaceholder_red: T.Tensor((), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for k0, k1 in T.grid(T.int64(3), T.int64(3)):
                with Ts.sblock("rxplaceholder_red"):
                    v_k0, v_k1 = Ts.axis.remap("RR", [k0, k1])
                    Ts.reads(rxplaceholder[v_k0, v_k1])
                    Ts.writes(rxplaceholder_red[()])
                    with Ts.init():
                        rxplaceholder_red[()] = T.float32(0)
                    rxplaceholder_red[()] = rxplaceholder_red[()] + rxplaceholder[v_k0, v_k1]

    # no such function
    with pytest.raises(ValueError):
        relax.transform.Gradient("main1")(NormalModule)
    # wrong function type
    with pytest.raises(RuntimeError):
        relax.transform.Gradient("sum")(NormalModule)
    # no such var
    with pytest.raises(RuntimeError):
        relax.transform.Gradient("main", require_grads=MultiBlocks["main"].params[0])(NormalModule)

    @I.ir_module
    class IntDtype:
        @R.function
        def main(x: R.Tensor((3, 3), "int64")):
            with R.dataflow():
                lv1 = R.add(x, x)
                gv = R.sum(lv1)
                R.output(gv)
            return gv

    with pytest.raises(RuntimeError):
        relax.transform.Gradient("main")(IntDtype)

    @I.ir_module
    class IntDtypeTuple:
        @R.function
        def main(x: R.Tuple(R.Tensor((3, 3), "int64"), R.Tensor((3, 3), "int64"))):
            with R.dataflow():
                lv1 = x[0]
                lv2 = x[1]
                lv3 = R.add(lv1, lv2)
                gv = R.sum(lv3)
                R.output(gv)
            return gv

    with pytest.raises(RuntimeError):
        relax.transform.Gradient("main")(IntDtypeTuple)


def test_mlp_script():
    """
    An example of single layer multi-layer perceptron. You can add extra layers if you want.

    For n-layer perceptron, see test_transform_gradient_numeric.py.
    """

    # fmt: off
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((3, 10), "float32"),
            w0: R.Tensor((10, 5), "float32"),
            b0: R.Tensor((5,), "float32"),
            label: R.Tensor((3, 5), "float32"),
        ):
            with R.dataflow():
                lv0 = R.matmul(x, w0)
                out = R.add(lv0, b0)
                logits = R.nn.log_softmax(out)
                loss = R.nn.cross_entropy_with_logits(logits, label)
                R.output(loss)
            return loss

    @I.ir_module
    class Expected:
        @R.function
        def main_adjoint(x: R.Tensor((3, 10), dtype="float32"), w0: R.Tensor((10, 5), dtype="float32"), b0: R.Tensor((5,), dtype="float32"), label: R.Tensor((3, 5), dtype="float32")) -> R.Tuple(R.Tensor((), dtype="float32"), R.Tuple(R.Tensor((10, 5), dtype="float32"), R.Tensor((5,), dtype="float32"))):
            with R.dataflow():
                lv0: R.Tensor((3, 5), dtype="float32") = R.matmul(x, w0)
                out: R.Tensor((3, 5), dtype="float32") = R.add(lv0, b0)
                logits: R.Tensor((3, 5), dtype="float32") = R.nn.log_softmax(out, axis=-1)
                loss: R.Tensor((), dtype="float32") = R.nn.cross_entropy_with_logits(logits, label)
                loss_adjoint: R.Tensor((), dtype="float32") = R.ones(R.shape([]), dtype="float32")
                lv: R.Tensor((), dtype="float32") = R.divide(loss_adjoint, R.const(3, "float32"))
                lv1: R.Tensor((), dtype="float32") = R.negative(lv)
                logits_adjoint: R.Tensor((3, 5), dtype="float32") = R.multiply(lv1, label)
                lv3: R.Tensor((3, 1), dtype="float32") = R.sum(logits_adjoint, axis=[-1], keepdims=True)
                lv4: R.Tensor((3, 5), dtype="float32") = R.exp(logits)
                lv5: R.Tensor((3, 5), dtype="float32") = R.multiply(lv3, lv4)
                out_adjoint: R.Tensor((3, 5), dtype="float32") = R.subtract(logits_adjoint, lv5)
                lv0_adjoint: R.Tensor((3, 5), dtype="float32") = out_adjoint
                b0_adjoint: R.Tensor((5,), dtype="float32") = R.collapse_sum_to(out_adjoint, R.shape([5]))
                lv7: R.Tensor((10, 3), dtype="float32") = R.permute_dims(x, axes=[1, 0])
                w0_adjoint: R.Tensor((10, 5), dtype="float32") = R.matmul(lv7, lv0_adjoint)
                w0_adjoint_out: R.Tensor((10, 5), dtype="float32") = w0_adjoint
                b0_adjoint_out: R.Tensor((5,), dtype="float32") = b0_adjoint
                R.output(loss, w0_adjoint_out, b0_adjoint_out)
            return (loss, (w0_adjoint_out, b0_adjoint_out))

        @R.function
        def main(x: R.Tensor((3, 10), dtype="float32"), w0: R.Tensor((10, 5), dtype="float32"), b0: R.Tensor((5,), dtype="float32"), label: R.Tensor((3, 5), dtype="float32")) -> R.Tensor((), dtype="float32"):
            with R.dataflow():
                lv0: R.Tensor((3, 5), dtype="float32") = R.matmul(x, w0)
                out: R.Tensor((3, 5), dtype="float32") = R.add(lv0, b0)
                logits: R.Tensor((3, 5), dtype="float32") = R.nn.log_softmax(out, axis=-1)
                loss: R.Tensor((), dtype="float32") = R.nn.cross_entropy_with_logits(logits, label)
                R.output(loss)
            return loss
    # fmt: on

    After = relax.transform.Gradient("main", require_grads=Before["main"].params[1:3])(Before)
    assert_structural_equal(After, Expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
