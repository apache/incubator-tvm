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
# ruff: noqa: E501, F811
"""Unit tests for gradient with checkpointing."""

import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax.block_builder import BlockBuilder
from tvm.relax.testing import nn
from tvm_tirx.ir.base import assert_structural_equal
from tvm_tirx.script.parser import ir as I
from tvm_tirx.script.parser import relax as R


def test_sequential():
    """Comp. graph is a sequence"""

    # fmt: off
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                x_scp = R.grad.start_checkpoint(x)
                lv1 = R.power(x_scp, R.const(3, "float32"))
                lv1_ecp = R.grad.end_checkpoint(lv1)
                lv2 = R.power(lv1_ecp, R.const(3, "float32"))
                lv2_scp = R.grad.start_checkpoint(lv2)
                lv3 = R.power(lv2_scp, R.const(3, "float32"))
                lv4 = R.power(lv3, R.const(3, "float32"))
                gv = R.sum(lv4)
                gv_ecp = R.grad.end_checkpoint(gv)
                R.output(gv_ecp)
            return gv_ecp

    @I.ir_module
    class Expected:
        @R.function
        def main_adjoint(x: R.Tensor((3, 3), "float32")) -> R.Tuple(R.Tensor((), "float32"), R.Tuple(R.Tensor((3, 3), "float32"))):
            with R.dataflow():
                lv1: R.Tensor((3, 3), "float32") = R.power(x, R.const(3, "float32"))
                lv2: R.Tensor((3, 3), "float32") = R.power(lv1, R.const(3, "float32"))
                lv3: R.Tensor((3, 3), "float32") = R.power(lv2, R.const(3, "float32"))
                lv4: R.Tensor((3, 3), "float32") = R.power(lv3, R.const(3, "float32"))
                gv: R.Tensor((), "float32") = R.sum(lv4, axis=None, keepdims=False)
                gv_1: R.Tensor((), "float32") = gv
                gv_adjoint: R.Tensor((), "float32") = R.ones(R.shape([]), "float32")
                gv_adjoint1: R.Tensor((), "float32") = gv_adjoint
                lv3_cp: R.Tensor((3, 3), "float32") = R.power(lv2, R.const(3, "float32"))
                lv4_adjoint: R.Tensor((3, 3), "float32") = R.broadcast_to(gv_adjoint1, R.shape([3, 3]))
                lv: R.Tensor((3, 3), "float32") = R.multiply(lv4_adjoint, R.const(3, "float32"))
                lv1_1: R.Tensor((), "float32") = R.subtract(R.const(3, "float32"), R.const(1, "float32"))
                lv2_1: R.Tensor((3, 3), "float32") = R.power(lv3_cp, lv1_1)
                lv3_adjoint: R.Tensor((3, 3), "float32") = R.multiply(lv, lv2_1)
                lv6: R.Tensor((3, 3), "float32") = R.multiply(lv3_adjoint, R.const(3, "float32"))
                lv7: R.Tensor((), "float32") = R.subtract(R.const(3, "float32"), R.const(1, "float32"))
                lv8: R.Tensor((3, 3), "float32") = R.power(lv2, lv7)
                lv2_adjoint: R.Tensor((3, 3), "float32") = R.multiply(lv6, lv8)
                lv1_cp: R.Tensor((3, 3), "float32") = R.power(x, R.const(3, "float32"))
                lv12: R.Tensor((3, 3), "float32") = R.multiply(lv2_adjoint, R.const(3, "float32"))
                lv13: R.Tensor((), "float32") = R.subtract(R.const(3, "float32"), R.const(1, "float32"))
                lv14: R.Tensor((3, 3), "float32") = R.power(lv1_cp, lv13)
                lv1_adjoint: R.Tensor((3, 3), "float32") = R.multiply(lv12, lv14)
                lv18: R.Tensor((3, 3), "float32") = R.multiply(lv1_adjoint, R.const(3, "float32"))
                lv19: R.Tensor((), "float32") = R.subtract(R.const(3, "float32"), R.const(1, "float32"))
                lv20: R.Tensor((3, 3), "float32") = R.power(x, lv19)
                x_adjoint: R.Tensor((3, 3), "float32") = R.multiply(lv18, lv20)
                x_adjoint_out: R.Tensor((3, 3), "float32") = x_adjoint
                R.output(gv_1, x_adjoint_out)
            return (gv_1, (x_adjoint_out,))

        @R.function
        def main(x: R.Tensor((3, 3), "float32")) -> R.Tensor((), "float32"):
            with R.dataflow():
                x_scp: R.Tensor((3, 3), "float32") = R.grad.start_checkpoint(x)
                lv1: R.Tensor((3, 3), "float32") = R.power(x_scp, R.const(3, "float32"))
                lv1_ecp: R.Tensor((3, 3), "float32") = R.grad.end_checkpoint(lv1)
                lv2: R.Tensor((3, 3), "float32") = R.power(lv1_ecp, R.const(3, "float32"))
                lv2_scp: R.Tensor((3, 3), "float32") = R.grad.start_checkpoint(lv2)
                lv3: R.Tensor((3, 3), "float32") = R.power(lv2_scp, R.const(3, "float32"))
                lv4: R.Tensor((3, 3), "float32") = R.power(lv3, R.const(3, "float32"))
                gv: R.Tensor((), "float32") = R.sum(lv4, axis=None, keepdims=False)
                gv_ecp: R.Tensor((), "float32") = R.grad.end_checkpoint(gv)
                R.output(gv_ecp)
            return gv_ecp
    # fmt: on

    After = relax.transform.Gradient("main")(Before)
    assert_structural_equal(After, Expected)


def test_sequential_consecutive():
    """Comp. graph is a sequence"""

    # fmt: off
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                x_scp = R.grad.start_checkpoint(x)
                lv1 = R.power(x_scp, R.const(3, "float32"))
                lv2 = R.power(lv1, R.const(3, "float32"))
                lv2_ecp = R.grad.end_checkpoint(lv2)
                lv2_scp = R.grad.start_checkpoint(lv2_ecp)
                lv3 = R.power(lv2_scp, R.const(3, "float32"))
                lv4 = R.power(lv3, R.const(3, "float32"))
                lv4_ecp = R.grad.end_checkpoint(lv4)
                gv = R.sum(lv4_ecp)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main_adjoint(x: R.Tensor((3, 3), "float32")) -> R.Tuple(R.Tensor((), "float32"), R.Tuple(R.Tensor((3, 3), "float32"))):
            with R.dataflow():
                lv1: R.Tensor((3, 3), "float32") = R.power(x, R.const(3, "float32"))
                lv2: R.Tensor((3, 3), "float32") = R.power(lv1, R.const(3, "float32"))
                lv3: R.Tensor((3, 3), "float32") = R.power(lv2, R.const(3, "float32"))
                lv4: R.Tensor((3, 3), "float32") = R.power(lv3, R.const(3, "float32"))
                gv: R.Tensor((), "float32") = R.sum(lv4, axis=None, keepdims=False)
                gv_adjoint: R.Tensor((), "float32") = R.ones(R.shape([]), "float32")
                lv3_cp: R.Tensor((3, 3), "float32") = R.power(lv2, R.const(3, "float32"))
                lv4_adjoint: R.Tensor((3, 3), "float32") = R.broadcast_to(gv_adjoint, R.shape([3, 3]))
                lv: R.Tensor((3, 3), "float32") = R.multiply(lv4_adjoint, R.const(3, "float32"))
                lv1_1: R.Tensor((), "float32") = R.subtract(R.const(3, "float32"), R.const(1, "float32"))
                lv2_1: R.Tensor((3, 3), "float32") = R.power(lv3_cp, lv1_1)
                lv3_adjoint: R.Tensor((3, 3), "float32") = R.multiply(lv, lv2_1)
                lv6: R.Tensor((3, 3), "float32") = R.multiply(lv3_adjoint, R.const(3, "float32"))
                lv7: R.Tensor((), "float32") = R.subtract(R.const(3, "float32"), R.const(1, "float32"))
                lv8: R.Tensor((3, 3), "float32") = R.power(lv2, lv7)
                lv2_adjoint: R.Tensor((3, 3), "float32") = R.multiply(lv6, lv8)
                lv1_cp: R.Tensor((3, 3), "float32") = R.power(x, R.const(3, "float32"))
                lv12: R.Tensor((3, 3), "float32") = R.multiply(lv2_adjoint, R.const(3, "float32"))
                lv13: R.Tensor((), "float32") = R.subtract(R.const(3, "float32"), R.const(1, "float32"))
                lv14: R.Tensor((3, 3), "float32") = R.power(lv1_cp, lv13)
                lv1_adjoint: R.Tensor((3, 3), "float32") = R.multiply(lv12, lv14)
                lv18: R.Tensor((3, 3), "float32") = R.multiply(lv1_adjoint, R.const(3, "float32"))
                lv19: R.Tensor((), "float32") = R.subtract(R.const(3, "float32"), R.const(1, "float32"))
                lv20: R.Tensor((3, 3), "float32") = R.power(x, lv19)
                x_adjoint: R.Tensor((3, 3), "float32") = R.multiply(lv18, lv20)
                x_adjoint_out: R.Tensor((3, 3), "float32") = x_adjoint
                R.output(gv, x_adjoint_out)
            return (gv, (x_adjoint_out,))

        @R.function
        def main(x: R.Tensor((3, 3), "float32")) -> R.Tensor((), "float32"):
            with R.dataflow():
                x_scp: R.Tensor((3, 3), "float32") = R.grad.start_checkpoint(x)
                lv1: R.Tensor((3, 3), "float32") = R.power(x_scp, R.const(3, "float32"))
                lv2: R.Tensor((3, 3), "float32") = R.power(lv1, R.const(3, "float32"))
                lv2_ecp: R.Tensor((3, 3), "float32") = R.grad.end_checkpoint(lv2)
                lv2_scp: R.Tensor((3, 3), "float32") = R.grad.start_checkpoint(lv2_ecp)
                lv3: R.Tensor((3, 3), "float32") = R.power(lv2_scp, R.const(3, "float32"))
                lv4: R.Tensor((3, 3), "float32") = R.power(lv3, R.const(3, "float32"))
                lv4_ecp: R.Tensor((3, 3), "float32") = R.grad.end_checkpoint(lv4)
                gv: R.Tensor((), "float32") = R.sum(lv4_ecp, axis=None, keepdims=False)
                R.output(gv)
            return gv

    # fmt: on

    After = relax.transform.Gradient("main")(Before)
    assert_structural_equal(After, Expected)


def test_tuple():
    """Comp. graph is a sequence"""

    # fmt: off
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                x_scp = R.grad.start_checkpoint(x)
                lv1 = R.power(x_scp, R.const(3, "float32"))
                lv2 = (x, lv1)
                lv3 = lv2
                lv4 = R.power(lv3[0], R.const(3, "float32"))
                lv4_ecp = R.grad.end_checkpoint(lv4)
                gv = R.sum(lv4_ecp)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main_adjoint(x: R.Tensor((3, 3), "float32")) -> R.Tuple(R.Tensor((), "float32"), R.Tuple(R.Tensor((3, 3), "float32"))):
            with R.dataflow():
                lv1: R.Tensor((3, 3), "float32") = R.power(x, R.const(3, "float32"))
                lv2: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3, 3), "float32")) = x, lv1
                lv3: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3, 3), "float32")) = lv2
                lv4: R.Tensor((3, 3), "float32") = lv3[0]
                lv4_1: R.Tensor((3, 3), "float32") = R.power(lv4, R.const(3, "float32"))
                gv: R.Tensor((), "float32") = R.sum(lv4_1, axis=None, keepdims=False)
                gv_adjoint: R.Tensor((), "float32") = R.ones(R.shape([]), "float32")
                lv1_cp: R.Tensor((3, 3), "float32") = R.power(x, R.const(3, "float32"))
                lv2_cp: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3, 3), "float32")) = x, lv1_cp
                lv3_cp: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3, 3), "float32")) = lv2_cp
                lv4_cp: R.Tensor((3, 3), "float32") = lv3_cp[0]
                lv4_adjoint: R.Tensor((3, 3), "float32") = R.broadcast_to(gv_adjoint, R.shape([3, 3]))
                lv: R.Tensor((3, 3), "float32") = R.multiply(lv4_adjoint, R.const(3, "float32"))
                lv1_1: R.Tensor((), "float32") = R.subtract(R.const(3, "float32"), R.const(1, "float32"))
                lv2_1: R.Tensor((3, 3), "float32") = R.power(lv4_cp, lv1_1)
                lv4_adjoint1: R.Tensor((3, 3), "float32") = R.multiply(lv, lv2_1)
                lv6: R.Tensor((3, 3), "float32") = R.zeros(R.shape([3, 3]), "float32")
                lv3_adjoint: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3, 3), "float32")) = lv4_adjoint1, lv6
                lv2_adjoint: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3, 3), "float32")) = lv3_adjoint
                x_adjoint: R.Tensor((3, 3), "float32") = lv2_adjoint[0]
                lv1_adjoint: R.Tensor((3, 3), "float32") = lv2_adjoint[1]
                lv7: R.Tensor((3, 3), "float32") = R.multiply(lv1_adjoint, R.const(3, "float32"))
                lv8: R.Tensor((), "float32") = R.subtract(R.const(3, "float32"), R.const(1, "float32"))
                lv9: R.Tensor((3, 3), "float32") = R.power(x, lv8)
                lv12: R.Tensor((3, 3), "float32") = R.multiply(lv7, lv9)
                x_adjoint1: R.Tensor((3, 3), "float32") = R.add(x_adjoint, lv12)
                x_adjoint_out: R.Tensor((3, 3), "float32") = x_adjoint1
                R.output(gv, x_adjoint_out)
            return (gv, (x_adjoint_out,))

        @R.function
        def main(x: R.Tensor((3, 3), "float32")) -> R.Tensor((), "float32"):
            with R.dataflow():
                x_scp: R.Tensor((3, 3), "float32") = R.grad.start_checkpoint(x)
                lv1: R.Tensor((3, 3), "float32") = R.power(x_scp, R.const(3, "float32"))
                lv2: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3, 3), "float32")) = x, lv1
                lv3: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3, 3), "float32")) = lv2
                lv4: R.Tensor((3, 3), "float32") = lv3[0]
                lv4_1: R.Tensor((3, 3), "float32") = R.power(lv4, R.const(3, "float32"))
                lv4_ecp: R.Tensor((3, 3), "float32") = R.grad.end_checkpoint(lv4_1)
                gv: R.Tensor((), "float32") = R.sum(lv4_ecp, axis=None, keepdims=False)
                R.output(gv)
            return gv
    # fmt: on

    After = relax.transform.Gradient("main")(Before)
    assert_structural_equal(After, Expected)


def test_tree():
    """Comp. graph is a output-directed tree"""

    # fmt: off
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((3, 3), "float32"), y: R.Tensor((3, 3), "float32"), z: R.Tensor((3, 3), "float32"), u: R.Tensor((3, 3), "float32"), v: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                lv1 = x * y
                lv1_scp = R.grad.start_checkpoint(lv1)
                z_scp = R.grad.start_checkpoint(z)
                lv2 = lv1_scp * z_scp
                lv2_ecp = R.grad.end_checkpoint(lv2)
                u_scp = R.grad.start_checkpoint(u)
                v_scp = R.grad.start_checkpoint(v)
                lv3 = u_scp * v_scp
                lv3_ecp = R.grad.end_checkpoint(lv3)
                lv4 = lv2_ecp * lv3_ecp
                gv = R.sum(lv4)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected1:
        @R.function
        def main_adjoint(x: R.Tensor((3, 3), "float32"), y: R.Tensor((3, 3), "float32"), z: R.Tensor((3, 3), "float32"), u: R.Tensor((3, 3), "float32"), v: R.Tensor((3, 3), "float32")) -> R.Tuple(R.Tensor((), "float32"), R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3, 3), "float32"), R.Tensor((3, 3), "float32"), R.Tensor((3, 3), "float32"), R.Tensor((3, 3), "float32"))):
            with R.dataflow():
                lv1: R.Tensor((3, 3), "float32") = R.multiply(x, y)
                lv2: R.Tensor((3, 3), "float32") = R.multiply(lv1, z)
                lv3: R.Tensor((3, 3), "float32") = R.multiply(u, v)
                lv4: R.Tensor((3, 3), "float32") = R.multiply(lv2, lv3)
                gv: R.Tensor((), "float32") = R.sum(lv4, axis=None, keepdims=False)
                gv_adjoint: R.Tensor((), "float32") = R.ones(R.shape([]), "float32")
                lv4_adjoint: R.Tensor((3, 3), "float32") = R.broadcast_to(gv_adjoint, R.shape([3, 3]))
                lv2_cp: R.Tensor((3, 3), "float32") = R.multiply(lv1, z)
                lv3_cp: R.Tensor((3, 3), "float32") = R.multiply(u, v)
                lv2_adjoint: R.Tensor((3, 3), "float32") = R.multiply(lv4_adjoint, lv3_cp)
                lv3_adjoint: R.Tensor((3, 3), "float32") = R.multiply(lv4_adjoint, lv2_cp)
                u_adjoint: R.Tensor((3, 3), "float32") = R.multiply(lv3_adjoint, v)
                v_adjoint: R.Tensor((3, 3), "float32") = R.multiply(lv3_adjoint, u)
                lv1_adjoint: R.Tensor((3, 3), "float32") = R.multiply(lv2_adjoint, z)
                z_adjoint: R.Tensor((3, 3), "float32") = R.multiply(lv2_adjoint, lv1)
                x_adjoint: R.Tensor((3, 3), "float32") = R.multiply(lv1_adjoint, y)
                y_adjoint: R.Tensor((3, 3), "float32") = R.multiply(lv1_adjoint, x)
                x_adjoint_out: R.Tensor((3, 3), "float32") = x_adjoint
                y_adjoint_out: R.Tensor((3, 3), "float32") = y_adjoint
                z_adjoint_out: R.Tensor((3, 3), "float32") = z_adjoint
                u_adjoint_out: R.Tensor((3, 3), "float32") = u_adjoint
                v_adjoint_out: R.Tensor((3, 3), "float32") = v_adjoint
                R.output(gv, x_adjoint_out, y_adjoint_out, z_adjoint_out, u_adjoint_out, v_adjoint_out)
            return (gv, (x_adjoint_out, y_adjoint_out, z_adjoint_out, u_adjoint_out, v_adjoint_out))

        @R.function
        def main(x: R.Tensor((3, 3), "float32"), y: R.Tensor((3, 3), "float32"), z: R.Tensor((3, 3), "float32"), u: R.Tensor((3, 3), "float32"), v: R.Tensor((3, 3), "float32")) -> R.Tensor((), "float32"):
            with R.dataflow():
                lv1 = x * y
                lv1_scp = R.grad.start_checkpoint(lv1)
                z_scp = R.grad.start_checkpoint(z)
                lv2 = lv1_scp * z_scp
                lv2_ecp = R.grad.end_checkpoint(lv2)
                u_scp = R.grad.start_checkpoint(u)
                v_scp = R.grad.start_checkpoint(v)
                lv3 = u_scp * v_scp
                lv3_ecp = R.grad.end_checkpoint(lv3)
                lv4 = lv2_ecp * lv3_ecp
                gv = R.sum(lv4)
                R.output(gv)
            return gv
    # fmt: on

    After1 = relax.transform.Gradient("main")(Before)
    assert_structural_equal(After1, Expected1)

    # fmt: off
    @I.ir_module
    class Expected2:
        @R.function
        def main_adjoint(x: R.Tensor((3, 3), "float32"), y: R.Tensor((3, 3), "float32"), z: R.Tensor((3, 3), "float32"), u: R.Tensor((3, 3), "float32"), v: R.Tensor((3, 3), "float32")) -> R.Tuple(R.Tensor((), "float32"), R.Tuple(R.Tensor((3, 3), "float32"))):
            with R.dataflow():
                lv1: R.Tensor((3, 3), "float32") = R.multiply(x, y)
                lv2: R.Tensor((3, 3), "float32") = R.multiply(lv1, z)
                lv3: R.Tensor((3, 3), "float32") = R.multiply(u, v)
                lv4: R.Tensor((3, 3), "float32") = R.multiply(lv2, lv3)
                gv: R.Tensor((), "float32") = R.sum(lv4, axis=None, keepdims=False)
                gv_adjoint: R.Tensor((), "float32") = R.ones(R.shape([]), "float32")
                lv4_adjoint: R.Tensor((3, 3), "float32") = R.broadcast_to(gv_adjoint, R.shape([3, 3]))
                lv3_cp: R.Tensor((3, 3), "float32") = R.multiply(u, v)
                lv2_adjoint: R.Tensor((3, 3), "float32") = R.multiply(lv4_adjoint, lv3_cp)
                z_adjoint: R.Tensor((3, 3), "float32") = R.multiply(lv2_adjoint, lv1)
                z_adjoint_out: R.Tensor((3, 3), "float32") = z_adjoint
                R.output(gv, z_adjoint_out)
            return (gv, (z_adjoint_out,))

        @R.function
        def main(x: R.Tensor((3, 3), "float32"), y: R.Tensor((3, 3), "float32"), z: R.Tensor((3, 3), "float32"), u: R.Tensor((3, 3), "float32"), v: R.Tensor((3, 3), "float32")) -> R.Tensor((), "float32"):
            with R.dataflow():
                lv1 = x * y
                lv1_scp = R.grad.start_checkpoint(lv1)
                z_scp = R.grad.start_checkpoint(z)
                lv2 = lv1_scp * z_scp
                lv2_ecp = R.grad.end_checkpoint(lv2)
                u_scp = R.grad.start_checkpoint(u)
                v_scp = R.grad.start_checkpoint(v)
                lv3 = u_scp * v_scp
                lv3_ecp = R.grad.end_checkpoint(lv3)
                lv4 = lv2_ecp * lv3_ecp
                gv = R.sum(lv4)
                R.output(gv)
            return gv
    # fmt: on

    After2 = relax.transform.Gradient("main", require_grads=Before["main"].params[2])(Before)
    assert_structural_equal(After2, Expected2)


def test_dag():
    """Comp. graph is a DAG with only one output. Here we only test the simple case: comp. graph
    is a sequence of sub-graphs, and the checkpoints are the intersections of connected
    subgraphs."""

    # fmt: off
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                lv = R.grad.start_checkpoint(x)
                lv1 = R.multiply(lv, R.const(2, "float32"))
                lv2 = R.multiply(lv1, R.const(2, "float32"))
                lv3 = R.grad.end_checkpoint(lv2)
                lv4 = R.multiply(x, lv3)
                lv5 = R.grad.start_checkpoint(lv4)
                lv6 = R.multiply(lv5, R.const(2, "float32"))
                lv7 = R.multiply(lv6, R.const(2, "float32"))
                lv8 = R.grad.end_checkpoint(lv7)
                lv9 = R.multiply(lv4, lv8)
                lv10 = R.grad.start_checkpoint(lv9)
                lv11 = R.multiply(lv10, R.const(2, "float32"))
                lv12 = R.multiply(lv11, R.const(2, "float32"))
                lv13 = R.grad.end_checkpoint(lv12)
                lv14 = R.multiply(lv9, lv13)
                gv: R.Tensor((), "float32") = R.sum(lv14, axis=None, keepdims=False)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main_adjoint(x: R.Tensor((3, 3), "float32")) -> R.Tuple(R.Tensor((), "float32"), R.Tuple(R.Tensor((3, 3), "float32"))):
            with R.dataflow():
                lv1: R.Tensor((3, 3), "float32") = R.multiply(x, R.const(2, "float32"))
                lv2: R.Tensor((3, 3), "float32") = R.multiply(lv1, R.const(2, "float32"))
                lv3: R.Tensor((3, 3), "float32") = R.multiply(x, lv2)
                lv4: R.Tensor((3, 3), "float32") = R.multiply(lv3, R.const(2, "float32"))
                lv5: R.Tensor((3, 3), "float32") = R.multiply(lv4, R.const(2, "float32"))
                lv6: R.Tensor((3, 3), "float32") = R.multiply(lv3, lv5)
                lv7: R.Tensor((3, 3), "float32") = R.multiply(lv6, R.const(2, "float32"))
                lv8: R.Tensor((3, 3), "float32") = R.multiply(lv7, R.const(2, "float32"))
                lv9: R.Tensor((3, 3), "float32") = R.multiply(lv6, lv8)
                gv: R.Tensor((), "float32") = R.sum(lv9, axis=None, keepdims=False)
                gv_adjoint: R.Tensor((), "float32") = R.ones(R.shape([]), "float32")
                lv9_adjoint: R.Tensor((3, 3), "float32") = R.broadcast_to(gv_adjoint, R.shape([3, 3]))
                lv7_cp: R.Tensor((3, 3), "float32") = R.multiply(lv6, R.const(2, "float32"))
                lv8_cp: R.Tensor((3, 3), "float32") = R.multiply(lv7_cp, R.const(2, "float32"))
                lv6_adjoint: R.Tensor((3, 3), "float32") = R.multiply(lv9_adjoint, lv8_cp)
                lv8_adjoint: R.Tensor((3, 3), "float32") = R.multiply(lv9_adjoint, lv6)
                lv7_adjoint: R.Tensor((3, 3), "float32") = R.multiply(lv8_adjoint, R.const(2, "float32"))
                lv1_1: R.Tensor((3, 3), "float32") = R.multiply(lv7_adjoint, R.const(2, "float32"))
                lv6_adjoint1: R.Tensor((3, 3), "float32") = R.add(lv6_adjoint, lv1_1)
                lv4_cp: R.Tensor((3, 3), "float32") = R.multiply(lv3, R.const(2, "float32"))
                lv5_cp: R.Tensor((3, 3), "float32") = R.multiply(lv4_cp, R.const(2, "float32"))
                lv3_adjoint: R.Tensor((3, 3), "float32") = R.multiply(lv6_adjoint1, lv5_cp)
                lv5_adjoint: R.Tensor((3, 3), "float32") = R.multiply(lv6_adjoint1, lv3)
                lv4_adjoint: R.Tensor((3, 3), "float32") = R.multiply(lv5_adjoint, R.const(2, "float32"))
                lv4_1: R.Tensor((3, 3), "float32") = R.multiply(lv4_adjoint, R.const(2, "float32"))
                lv3_adjoint1: R.Tensor((3, 3), "float32") = R.add(lv3_adjoint, lv4_1)
                lv1_cp: R.Tensor((3, 3), "float32") = R.multiply(x, R.const(2, "float32"))
                lv2_cp: R.Tensor((3, 3), "float32") = R.multiply(lv1_cp, R.const(2, "float32"))
                x_adjoint: R.Tensor((3, 3), "float32") = R.multiply(lv3_adjoint1, lv2_cp)
                lv2_adjoint: R.Tensor((3, 3), "float32") = R.multiply(lv3_adjoint1, x)
                lv1_adjoint: R.Tensor((3, 3), "float32") = R.multiply(lv2_adjoint, R.const(2, "float32"))
                lv7_1: R.Tensor((3, 3), "float32") = R.multiply(lv1_adjoint, R.const(2, "float32"))
                x_adjoint1: R.Tensor((3, 3), "float32") = R.add(x_adjoint, lv7_1)
                x_adjoint_out: R.Tensor((3, 3), "float32") = x_adjoint1
                R.output(gv, x_adjoint_out)
            return (gv, (x_adjoint_out,))

        @R.function
        def main(x: R.Tensor((3, 3), "float32")) -> R.Tensor((), "float32"):
            with R.dataflow():
                lv = R.grad.start_checkpoint(x)
                lv1 = R.multiply(lv, R.const(2, "float32"))
                lv2 = R.multiply(lv1, R.const(2, "float32"))
                lv3 = R.grad.end_checkpoint(lv2)
                lv4 = R.multiply(x, lv3)
                lv5 = R.grad.start_checkpoint(lv4)
                lv6 = R.multiply(lv5, R.const(2, "float32"))
                lv7 = R.multiply(lv6, R.const(2, "float32"))
                lv8 = R.grad.end_checkpoint(lv7)
                lv9 = R.multiply(lv4, lv8)
                lv10 = R.grad.start_checkpoint(lv9)
                lv11 = R.multiply(lv10, R.const(2, "float32"))
                lv12 = R.multiply(lv11, R.const(2, "float32"))
                lv13 = R.grad.end_checkpoint(lv12)
                lv14 = R.multiply(lv9, lv13)
                gv: R.Tensor((), "float32") = R.sum(lv14, axis=None, keepdims=False)
                R.output(gv)
            return gv
    # fmt: on

    After = relax.transform.Gradient("main")(Before)
    assert_structural_equal(After, Expected)


def test_checkpoint_api():
    """Test on tvm.relax.testing.nn.checkpoint API"""

    def func1(x):
        return relax.op.power(x, relax.const(3, "float32"))

    def func2(x):
        y = relax.op.power(relax.op.power(x, relax.const(3, "float32")), relax.const(3, "float32"))
        return relax.op.sum(y)

    bb = BlockBuilder()
    x = relax.Var("x", relax.TensorType((3, 3), "float32"))
    with bb.function("main", [x]):
        with bb.dataflow():
            lv1 = bb.emit(nn.checkpoint(func1, x))
            lv2 = bb.emit(relax.op.power(lv1, relax.const(3, "float32")))
            lv3 = bb.emit_output(nn.checkpoint(func2, lv2))
        bb.emit_func_output(lv3)

    # fmt: off
    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                x_scp = R.grad.start_checkpoint(x)
                lv1 = R.power(x_scp, R.const(3, "float32"))
                lv1_ecp = R.grad.end_checkpoint(lv1)
                lv2 = R.power(lv1_ecp, R.const(3, "float32"))
                lv2_scp = R.grad.start_checkpoint(lv2)
                lv3 = R.power(lv2_scp, R.const(3, "float32"))
                lv4 = R.power(lv3, R.const(3, "float32"))
                gv = R.sum(lv4)
                gv_ecp = R.grad.end_checkpoint(gv)
                R.output(gv_ecp)
            return gv_ecp
    # fmt: on

    assert_structural_equal(bb.get(), Expected)


def test_checkpoint_tree():
    """Comp. graph is a output-directed tree"""

    def func(x, y, z, w):
        return x * y, z * w

    bb = BlockBuilder()
    x = relax.Var("x", relax.TensorType((3, 3), "float32"))
    y = relax.Var("y", relax.TensorType((3, 3), "float32"))
    z = relax.Var("z", relax.TensorType((3, 3), "float32"))
    u = relax.Var("u", relax.TensorType((3, 3), "float32"))
    v = relax.Var("v", relax.TensorType((3, 3), "float32"))
    with bb.function("main", [x, y, z, u, v]):
        with bb.dataflow():
            lv1 = bb.emit(x * y)
            cp = nn.checkpoint(func, lv1, z, u, v)
            lv2 = bb.emit(cp[0])
            lv3 = bb.emit(cp[1])
            lv4 = bb.emit(lv2 * lv3)
            gv = bb.emit_output(relax.op.sum(lv4))
        bb.emit_func_output(gv)

    # fmt: off
    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((3, 3), "float32"), y: R.Tensor((3, 3), "float32"), z: R.Tensor((3, 3), "float32"), u: R.Tensor((3, 3), "float32"), v: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                lv1 = x * y
                lv1_scp = R.grad.start_checkpoint(lv1)
                z_scp = R.grad.start_checkpoint(z)
                lv2 = lv1_scp * z_scp
                lv2_ecp = R.grad.end_checkpoint(lv2)
                u_scp = R.grad.start_checkpoint(u)
                v_scp = R.grad.start_checkpoint(v)
                lv3 = u_scp * v_scp
                lv3_ecp = R.grad.end_checkpoint(lv3)
                lv4 = lv2_ecp * lv3_ecp
                gv = R.sum(lv4)
                R.output(gv)
            return gv
    # fmt: on

    assert_structural_equal(bb.get(), Expected)


def test_checkpoint_dag():
    """Comp. graph is a DAG with only one output. Here we only test the simple case: comp. graph
    is a sequence of sub-graphs, and the checkpoints are the intersections of connected
    subgraphs."""

    def func(x):
        return x * relax.const(2, "float32") * relax.const(2, "float32")

    bb = BlockBuilder()
    x = relax.Var("x", relax.TensorType((3, 3), "float32"))
    with bb.function("main", [x]):
        with bb.dataflow():
            lv1 = bb.emit(nn.checkpoint(func, x))
            lv2 = bb.emit(x * lv1)
            lv3 = bb.emit(nn.checkpoint(func, lv2))
            lv4 = bb.emit(lv2 * lv3)
            lv5 = bb.emit(nn.checkpoint(func, lv4))
            lv6 = bb.emit(lv4 * lv5)
            gv = bb.emit_output(relax.op.sum(lv6))
        bb.emit_func_output(gv)

    # fmt: off
    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((3, 3), "float32")) -> R.Tensor((), "float32"):
            with R.dataflow():
                lv = R.grad.start_checkpoint(x)
                lv1 = R.multiply(lv, R.const(2, "float32"))
                lv2 = R.multiply(lv1, R.const(2, "float32"))
                lv3 = R.grad.end_checkpoint(lv2)
                lv4 = R.multiply(x, lv3)
                lv5 = R.grad.start_checkpoint(lv4)
                lv6 = R.multiply(lv5, R.const(2, "float32"))
                lv7 = R.multiply(lv6, R.const(2, "float32"))
                lv8 = R.grad.end_checkpoint(lv7)
                lv9 = R.multiply(lv4, lv8)
                lv10 = R.grad.start_checkpoint(lv9)
                lv11 = R.multiply(lv10, R.const(2, "float32"))
                lv12 = R.multiply(lv11, R.const(2, "float32"))
                lv13 = R.grad.end_checkpoint(lv12)
                lv14 = R.multiply(lv9, lv13)
                gv: R.Tensor((), "float32") = R.sum(lv14, axis=None, keepdims=False)
                R.output(gv)
            return gv
    # fmt: on

    assert_structural_equal(bb.get(), Expected)


def test_checkpoint_sequential():
    def func(x):
        return x + x

    bb = BlockBuilder()
    x = relax.Var("x", relax.TensorType((3, 3), "float32"))
    with bb.function("main", [x]):
        with bb.dataflow():
            lv1 = nn.emit_checkpoint_sequential([func] * 5, 2, x)
            lv2 = nn.emit_checkpoint_sequential([func] * 4, 2, lv1)
            gv = bb.emit_output(lv2)
        bb.emit_func_output(gv)

    # fmt: off
    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((3, 3), "float32")) -> R.Tensor((3, 3), "float32"):
            with R.dataflow():
                x_scp: R.Tensor((3, 3), "float32") = R.grad.start_checkpoint(x)
                lv: R.Tensor((3, 3), "float32") = R.add(x_scp, x_scp)
                lv1: R.Tensor((3, 3), "float32") = R.add(lv, lv)
                lv1_ecp: R.Tensor((3, 3), "float32") = R.grad.end_checkpoint(lv1)
                lv1_ecp_scp: R.Tensor((3, 3), "float32") = R.grad.start_checkpoint(lv1_ecp)
                lv2: R.Tensor((3, 3), "float32") = R.add(lv1_ecp_scp, lv1_ecp_scp)
                lv3: R.Tensor((3, 3), "float32") = R.add(lv2, lv2)
                lv3_ecp: R.Tensor((3, 3), "float32") = R.grad.end_checkpoint(lv3)
                lv4: R.Tensor((3, 3), "float32") = R.add(lv3_ecp, lv3_ecp)
                lv4_scp: R.Tensor((3, 3), "float32") = R.grad.start_checkpoint(lv4)
                lv5: R.Tensor((3, 3), "float32") = R.add(lv4_scp, lv4_scp)
                lv6: R.Tensor((3, 3), "float32") = R.add(lv5, lv5)
                lv6_ecp: R.Tensor((3, 3), "float32") = R.grad.end_checkpoint(lv6)
                lv7: R.Tensor((3, 3), "float32") = R.add(lv6_ecp, lv6_ecp)
                lv8: R.Tensor((3, 3), "float32") = R.add(lv7, lv7)
                gv: R.Tensor((3, 3), "float32") = lv8
                R.output(gv)
            return gv
    # fmt: on

    assert_structural_equal(bb.get(), Expected)


def test_checkpoint_sequential_checkpoint_last():
    def func(x):
        return x + x

    bb = BlockBuilder()
    x = relax.Var("x", relax.TensorType((3, 3), "float32"))
    with bb.function("main", [x]):
        with bb.dataflow():
            lv1 = nn.emit_checkpoint_sequential([func] * 5, 2, x, checkpoint_last=True)
            lv2 = nn.emit_checkpoint_sequential([func] * 4, 2, lv1, checkpoint_last=True)
            gv = bb.emit_output(lv2)
        bb.emit_func_output(gv)

    # fmt: off
    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((3, 3), "float32")) -> R.Tensor((3, 3), "float32"):
            with R.dataflow():
                x_scp: R.Tensor((3, 3), "float32") = R.grad.start_checkpoint(x)
                lv: R.Tensor((3, 3), "float32") = R.add(x_scp, x_scp)
                lv1: R.Tensor((3, 3), "float32") = R.add(lv, lv)
                lv1_ecp: R.Tensor((3, 3), "float32") = R.grad.end_checkpoint(lv1)
                lv1_ecp_scp: R.Tensor((3, 3), "float32") = R.grad.start_checkpoint(lv1_ecp)
                lv2: R.Tensor((3, 3), "float32") = R.add(lv1_ecp_scp, lv1_ecp_scp)
                lv3: R.Tensor((3, 3), "float32") = R.add(lv2, lv2)
                lv3_ecp: R.Tensor((3, 3), "float32") = R.grad.end_checkpoint(lv3)
                lv3_ecp_scp: R.Tensor((3, 3), "float32") = R.grad.start_checkpoint(lv3_ecp)
                lv4: R.Tensor((3, 3), "float32") = R.add(lv3_ecp_scp, lv3_ecp_scp)
                lv4_ecp: R.Tensor((3, 3), "float32") = R.grad.end_checkpoint(lv4)
                lv4_ecp_scp: R.Tensor((3, 3), "float32") = R.grad.start_checkpoint(lv4_ecp)
                lv5: R.Tensor((3, 3), "float32") = R.add(lv4_ecp_scp, lv4_ecp_scp)
                lv6: R.Tensor((3, 3), "float32") = R.add(lv5, lv5)
                lv6_ecp: R.Tensor((3, 3), "float32") = R.grad.end_checkpoint(lv6)
                lv6_ecp_scp: R.Tensor((3, 3), "float32") = R.grad.start_checkpoint(lv6_ecp)
                lv7: R.Tensor((3, 3), "float32") = R.add(lv6_ecp_scp, lv6_ecp_scp)
                lv8: R.Tensor((3, 3), "float32") = R.add(lv7, lv7)
                lv8_ecp: R.Tensor((3, 3), "float32") = R.grad.end_checkpoint(lv8)
                gv: R.Tensor((3, 3), "float32") = lv8_ecp
                R.output(gv)
            return gv
    # fmt: on

    assert_structural_equal(bb.get(), Expected)


def test_checkpoint_dag():
    """Comp. graph is a DAG with only one output. Here we only test the simple case: comp. graph
    is a sequence of sub-graphs, and the checkpoints are the intersections of connected
    subgraphs."""

    def func(x):
        return x * relax.const(2, "float32") * relax.const(2, "float32")

    bb = BlockBuilder()
    x = relax.Var("x", relax.TensorType((3, 3), "float32"))
    with bb.function("main", [x]):
        with bb.dataflow():
            lv1 = bb.emit(nn.checkpoint(func, x))
            lv2 = bb.emit(x * lv1)
            lv3 = bb.emit(nn.checkpoint(func, lv2))
            lv4 = bb.emit(lv2 * lv3)
            lv5 = bb.emit(nn.checkpoint(func, lv4))
            lv6 = bb.emit(lv4 * lv5)
            gv = bb.emit_output(relax.op.sum(lv6))
        bb.emit_func_output(gv)


def test_checkpoint_with_intermediate_require_grads():
    def func(x):
        return x * x * x

    bb = BlockBuilder()
    x = relax.Var("x", relax.TensorType((3, 3), "float32"))
    with bb.function("main", [x]):
        with bb.dataflow():
            lv1 = nn.emit_checkpoint(func, x)
            gv = bb.emit_output(relax.op.sum(lv1))
        bb.emit_func_output(gv)

    # fmt: off
    @I.ir_module
    class Expected:
        @R.function
        def main_adjoint(x: R.Tensor((3, 3), "float32")) -> R.Tuple(R.Tensor((), "float32"), R.Tuple(R.Tensor((3, 3), "float32"))):
            with R.dataflow():
                lv: R.Tensor((3, 3), "float32") = R.multiply(x, x)
                lv1: R.Tensor((3, 3), "float32") = R.multiply(lv, x)
                gv: R.Tensor((), "float32") = R.sum(lv1, axis=None, keepdims=False)
                gv_adjoint: R.Tensor((), "float32") = R.ones(R.shape([]), "float32")
                lv1_adjoint: R.Tensor((3, 3), "float32") = R.broadcast_to(gv_adjoint, R.shape([3, 3]))
                lv1_adjoint_out: R.Tensor((3, 3), "float32") = lv1_adjoint
                R.output(gv, lv1_adjoint_out)
            return (gv, (lv1_adjoint_out,))

        @R.function
        def main(x: R.Tensor((3, 3), "float32")) -> R.Tensor((), "float32"):
            with R.dataflow():
                x_scp: R.Tensor((3, 3), "float32") = R.grad.start_checkpoint(x)
                lv: R.Tensor((3, 3), "float32") = R.multiply(x_scp, x_scp)
                lv1: R.Tensor((3, 3), "float32") = R.multiply(lv, x_scp)
                lv1_ecp: R.Tensor((3, 3), "float32") = R.grad.end_checkpoint(lv1)
                gv: R.Tensor((), "float32") = R.sum(lv1_ecp, axis=None, keepdims=False)
                R.output(gv)
            return gv
    # fmt: on

    After = relax.transform.Gradient("main", lv1)(bb.get())
    assert_structural_equal(After, Expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
