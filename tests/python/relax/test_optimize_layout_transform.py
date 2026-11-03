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
"""Tests to validate relax optimize layout tranform pass."""

import numpy as np
import pytest

import tvm
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm.relax.transform import DeadCodeElimination, FuseTIR, OptimizeLayoutTransform
from tvm.s_tir import script as Ts
from tvm_tirx.ir.base import assert_structural_equal
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def _run_pass_compare_output(Before, Expected):
    After = tvm_tirx.ir.transform.Sequential(
        [
            OptimizeLayoutTransform(),
            DeadCodeElimination(),
            FuseTIR(),
        ]
    )(Before)

    tvm_tirx.ir.assert_structural_equal(Expected, After)


def test_optimize_transform_layout_pass_one_arg():
    @I.ir_module
    class Before:
        @Ts.function(private=True)
        def relax_add_replacement(
            arg0: T.Tensor((4, 4), "float32"),
            arg1: T.Tensor((4, 4), "float32"),
            output: T.Tensor((4, 4), "float32"),
        ):
            T.func_attr({"operator_name": "relax.add"})
            # with Ts.sblock("root"):
            for ax0, ax1 in T.grid(4, 4):
                with Ts.sblock("T_add"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(arg0[v_ax0, v_ax1], arg1[v_ax0, v_ax1])
                    Ts.writes(output[v_ax0, v_ax1])
                    output[v_ax0, v_ax1] = arg0[v_ax0, v_ax1] + arg1[v_ax0, v_ax1]

        @R.function
        def main(
            x: R.Tensor((16,), dtype="float32"), y: R.Tensor((16,), dtype="float32")
        ) -> R.Tensor((16,), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((4, 4), dtype="float32") = R.layout_transform(
                    x, index_map=lambda i: (i // 4, i % 4), pad_value=None
                )
                lv1: R.Tensor((4, 4), dtype="float32") = R.layout_transform(
                    y, index_map=lambda i: (i // 4, i % 4), pad_value=None
                )
                lv2 = R.call_tir(
                    Before.relax_add_replacement,
                    (lv, lv1),
                    ty_args=[R.Tensor((4, 4), dtype="float32")],
                )
                lv0: R.Tensor((16,), dtype="float32") = R.layout_transform(
                    lv2, index_map=lambda axis0, axis1: (axis0 * 4 + axis1,), pad_value=None
                )
                lv3: R.Tensor((4, 4), dtype="float32") = R.layout_transform(
                    lv0, index_map=lambda i: (i // 4, i % 4), pad_value=None
                )
                lv4: R.Tensor((4, 4), dtype="float32") = R.layout_transform(
                    y, index_map=lambda i: (i // 4, i % 4), pad_value=None
                )
                lv5 = R.call_tir(
                    Before.relax_add_replacement,
                    (lv4, lv3),
                    ty_args=[R.Tensor((4, 4), dtype="float32")],
                )
                lv2_1: R.Tensor((16,), dtype="float32") = R.layout_transform(
                    lv5, index_map=lambda axis0, axis1: (axis0 * 4 + axis1,), pad_value=None
                )
                gv: R.Tensor((16,), dtype="float32") = lv2_1
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def relax_add_replacement(
            arg0: T.Tensor((4, 4), "float32"),
            arg1: T.Tensor((4, 4), "float32"),
            output: T.Tensor((4, 4), "float32"),
        ):
            T.func_attr({"operator_name": "relax.add"})
            # with Ts.sblock("root"):
            for ax0, ax1 in T.grid(4, 4):
                with Ts.sblock("T_add"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(arg0[v_ax0, v_ax1], arg1[v_ax0, v_ax1])
                    Ts.writes(output[v_ax0, v_ax1])
                    output[v_ax0, v_ax1] = arg0[v_ax0, v_ax1] + arg1[v_ax0, v_ax1]

        @R.function
        def main(
            x: R.Tensor((16,), dtype="float32"), y: R.Tensor((16,), dtype="float32")
        ) -> R.Tensor((16,), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((4, 4), dtype="float32") = R.layout_transform(
                    x, index_map=lambda i: (i // 4, i % 4), pad_value=None
                )
                lv1: R.Tensor((4, 4), dtype="float32") = R.layout_transform(
                    y, index_map=lambda i: (i // 4, i % 4), pad_value=None
                )
                lv2 = R.call_tir(
                    Expected.relax_add_replacement,
                    (lv, lv1),
                    ty_args=[R.Tensor((4, 4), dtype="float32")],
                )
                lv5 = R.call_tir(
                    Expected.relax_add_replacement,
                    (lv1, lv2),
                    ty_args=[R.Tensor((4, 4), dtype="float32")],
                )
                gv: R.Tensor((16,), dtype="float32") = R.layout_transform(
                    lv5, index_map=lambda axis0, axis1: (axis0 * 4 + axis1,), pad_value=None
                )
                R.output(gv)
            return gv

    _run_pass_compare_output(Before, Expected)


def test_optimize_transform_layout_pass_two_args():
    @I.ir_module
    class Before:
        @Ts.function(private=True)
        def relax_add_replacement(
            arg0: T.Tensor((4, 4), "float32"),
            arg1: T.Tensor((4, 4), "float32"),
            output: T.Tensor((4, 4), "float32"),
        ):
            T.func_attr({"operator_name": "relax.add"})
            # with Ts.sblock("root"):
            for ax0, ax1 in T.grid(4, 4):
                with Ts.sblock("T_add"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(arg0[v_ax0, v_ax1], arg1[v_ax0, v_ax1])
                    Ts.writes(output[v_ax0, v_ax1])
                    output[v_ax0, v_ax1] = arg0[v_ax0, v_ax1] + arg1[v_ax0, v_ax1]

        @R.function
        def main(
            x: R.Tensor((16,), dtype="float32"),
            y: R.Tensor((16,), dtype="float32"),
            z: R.Tensor((16,), dtype="float32"),
        ) -> R.Tensor((16,), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((4, 4), dtype="float32") = R.layout_transform(
                    x, index_map=lambda i: (i // 4, i % 4), pad_value=None
                )
                lv1: R.Tensor((4, 4), dtype="float32") = R.layout_transform(
                    y, index_map=lambda i: (i // 4, i % 4), pad_value=None
                )
                lv2: R.Tensor((4, 4), dtype="float32") = R.layout_transform(
                    z, index_map=lambda i: (i // 4, i % 4), pad_value=None
                )
                lv3 = R.call_tir(
                    Before.relax_add_replacement,
                    (lv, lv1),
                    ty_args=[R.Tensor((4, 4), dtype="float32")],
                )
                lv4 = R.call_tir(
                    Before.relax_add_replacement,
                    (lv, lv2),
                    ty_args=[R.Tensor((4, 4), dtype="float32")],
                )
                lv5: R.Tensor((16,), dtype="float32") = R.layout_transform(
                    lv3, index_map=lambda axis0, axis1: (axis0 * 4 + axis1,), pad_value=None
                )
                lv6: R.Tensor((16,), dtype="float32") = R.layout_transform(
                    lv4, index_map=lambda axis0, axis1: (axis0 * 4 + axis1,), pad_value=None
                )
                lv7: R.Tensor((4, 4), dtype="float32") = R.layout_transform(
                    lv5, index_map=lambda i: (i // 4, i % 4), pad_value=None
                )
                lv8: R.Tensor((4, 4), dtype="float32") = R.layout_transform(
                    lv6, index_map=lambda i: (i // 4, i % 4), pad_value=None
                )
                lv9 = R.call_tir(
                    Before.relax_add_replacement,
                    (lv7, lv8),
                    ty_args=[R.Tensor((4, 4), dtype="float32")],
                )
                lv10: R.Tensor((16,), dtype="float32") = R.layout_transform(
                    lv9, index_map=lambda axis0, axis1: (axis0 * 4 + axis1,), pad_value=None
                )
                gv: R.Tensor((16,), dtype="float32") = lv10
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def relax_add_replacement(
            arg0: T.Tensor((4, 4), "float32"),
            arg1: T.Tensor((4, 4), "float32"),
            output: T.Tensor((4, 4), "float32"),
        ):
            T.func_attr({"operator_name": "relax.add"})
            # with Ts.sblock("root"):
            for ax0, ax1 in T.grid(4, 4):
                with Ts.sblock("T_add"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(arg0[v_ax0, v_ax1], arg1[v_ax0, v_ax1])
                    Ts.writes(output[v_ax0, v_ax1])
                    output[v_ax0, v_ax1] = arg0[v_ax0, v_ax1] + arg1[v_ax0, v_ax1]

        @R.function
        def main(
            x: R.Tensor((16,), dtype="float32"),
            y: R.Tensor((16,), dtype="float32"),
            z: R.Tensor((16,), dtype="float32"),
        ) -> R.Tensor((16,), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((4, 4), dtype="float32") = R.layout_transform(
                    x, index_map=lambda i: (i // 4, i % 4), pad_value=None
                )
                lv1: R.Tensor((4, 4), dtype="float32") = R.layout_transform(
                    y, index_map=lambda i: (i // 4, i % 4), pad_value=None
                )
                lv2: R.Tensor((4, 4), dtype="float32") = R.layout_transform(
                    z, index_map=lambda i: (i // 4, i % 4), pad_value=None
                )
                lv3 = R.call_tir(
                    Expected.relax_add_replacement,
                    (lv, lv1),
                    ty_args=[R.Tensor((4, 4), dtype="float32")],
                )
                lv4 = R.call_tir(
                    Expected.relax_add_replacement,
                    (lv, lv2),
                    ty_args=[R.Tensor((4, 4), dtype="float32")],
                )
                lv5 = R.call_tir(
                    Expected.relax_add_replacement,
                    (lv3, lv4),
                    ty_args=[R.Tensor((4, 4), dtype="float32")],
                )
                gv: R.Tensor((16,), dtype="float32") = R.layout_transform(
                    lv5, index_map=lambda axis0, axis1: (axis0 * 4 + axis1,), pad_value=None
                )
                R.output(gv)
            return gv

    _run_pass_compare_output(Before, Expected)


def test_tranform_layout_tir_remove_pad_transform_layout():
    p0 = T.dynamic("p0", "int64")
    i0 = T.dynamic("i0", "int64")

    @I.ir_module
    class Before:
        @Ts.function(private=True)
        def relax_relu_replacement(
            arg0: T.Tensor((16,), "float32"), output: T.Tensor((16,), "float32")
        ):
            T.func_attr({"operator_name": "relax.relu"})
            # with Ts.sblock("root"):
            for ax0 in range(16):
                with Ts.sblock("T_add"):
                    v_ax0 = Ts.axis.spatial(16, ax0)
                    Ts.reads(arg0[v_ax0])
                    Ts.writes(output[v_ax0])
                    output[v_ax0] = T.max(arg0[v_ax0], T.float32(0))

        @Ts.function(private=True)
        def remove_pad(input: T.Tensor((p0,), "float32"), output: T.Tensor((i0,), "float32")):
            T.func_attr({"operator_name": "remove_pad", "tirx.noalias": True})

            # with Ts.sblock("root"):
            for ax0 in range(i0):
                with Ts.sblock("output"):
                    v_ax0 = Ts.axis.spatial(i0, ax0)
                    Ts.reads(input[v_ax0])
                    Ts.writes(output[v_ax0])
                    output[v_ax0] = input[v_ax0]

        @R.function
        def main(x: R.Tensor((14,), dtype="float32")) -> R.Tensor((14,), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((16,), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(lambda i: (i % 16,)),
                    pad_value=None,
                )
                lv1 = R.call_tir(
                    Before.relax_relu_replacement,
                    (lv,),
                    ty_args=[R.Tensor((16,), dtype="float32")],
                )
                lv2: R.Tensor((16,), dtype="float32") = R.layout_transform(
                    lv1,
                    index_map=T.index_map(lambda axis0: (axis0,)),
                    pad_value=None,
                )
                lv_1 = R.call_tir(
                    Before.remove_pad, (lv2,), ty_args=[R.Tensor((14,), dtype="float32")]
                )
                lv3: R.Tensor((16,), dtype="float32") = R.layout_transform(
                    lv_1,
                    index_map=T.index_map(lambda i: (i % 16,)),
                    pad_value=None,
                )
                lv4 = R.call_tir(
                    Before.relax_relu_replacement,
                    (lv3,),
                    ty_args=[R.Tensor((16,), dtype="float32")],
                )
                lv5: R.Tensor((16,), dtype="float32") = R.layout_transform(
                    lv4,
                    index_map=T.index_map(lambda axis0: (axis0,)),
                    pad_value=None,
                )
                lv_2 = R.call_tir(
                    Before.remove_pad, (lv5,), ty_args=[R.Tensor((14,), dtype="float32")]
                )
                gv: R.Tensor((14,), dtype="float32") = lv_2
                R.output(gv)
            return gv

    p0 = T.dynamic("p0", "int64")
    i0 = T.dynamic("i0", "int64")

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def relax_relu_replacement(
            arg0: T.Tensor((16,), "float32"), output: T.Tensor((16,), "float32")
        ):
            T.func_attr({"operator_name": "relax.relu"})
            # with Ts.sblock("root"):
            for ax0 in range(16):
                with Ts.sblock("T_add"):
                    v_ax0 = Ts.axis.spatial(16, ax0)
                    Ts.reads(arg0[v_ax0])
                    Ts.writes(output[v_ax0])
                    output[v_ax0] = T.max(arg0[v_ax0], T.float32(0))

        @Ts.function(private=True)
        def remove_pad(input: T.Tensor((p0,), "float32"), output: T.Tensor((i0,), "float32")):
            T.func_attr({"operator_name": "remove_pad", "tirx.noalias": True})

            # with Ts.sblock("root"):
            for ax0 in range(i0):
                with Ts.sblock("output"):
                    v_ax0 = Ts.axis.spatial(i0, ax0)
                    Ts.reads(input[v_ax0])
                    Ts.writes(output[v_ax0])
                    output[v_ax0] = input[v_ax0]

        @R.function
        def main(x: R.Tensor((14,), dtype="float32")) -> R.Tensor((14,), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((16,), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(lambda i: (i % 16,)),
                    pad_value=None,
                )
                lv1 = R.call_tir(
                    Expected.relax_relu_replacement,
                    (lv,),
                    ty_args=[R.Tensor((16,), dtype="float32")],
                )
                lv4 = R.call_tir(
                    Expected.relax_relu_replacement,
                    (lv1,),
                    ty_args=[R.Tensor((16,), dtype="float32")],
                )
                lv5: R.Tensor((16,), dtype="float32") = R.layout_transform(
                    lv4,
                    index_map=T.index_map(lambda axis0: (axis0,)),
                    pad_value=None,
                )
                gv = R.call_tir(
                    Expected.remove_pad, (lv5,), ty_args=[R.Tensor((14,), dtype="float32")]
                )
                R.output(gv)
            return gv

    _run_pass_compare_output(Before, Expected)


@pytest.mark.parametrize("symbolic", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("layout", ["transpose", "split"])
def test_index_map_composition(layout, inverse, symbolic):
    extent = T.dynamic("n", "int64") if symbolic else 3
    if layout == "transpose":
        shape = (extent, extent)

        def inner_map(i, j):
            return (j, i)

        def outer_map(i, j):
            return (j, i) if inverse else (i, j)

        values = np.arange(9, dtype="float32").reshape(3, 3)
        expected = values if inverse else values.T
    else:
        shape = (extent, 4)

        def inner_map(i, j):
            return (4 * i + j,)

        def outer_map(i):
            return (i // 4, i % 4) if inverse else (i % extent, i // extent)

        values = np.arange(12, dtype="float32").reshape(3, 4)
        expected = values if inverse else values.reshape(4, 3).T

    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorType(shape, "float32"))
    with bb.function("main", [x]):
        intermediate = bb.emit(relax.op.layout_transform(x, index_map=inner_map))
        output = bb.emit(relax.op.layout_transform(intermediate, index_map=outer_map))
        bb.emit_func_output(output)
    Before = bb.get()
    After = DeadCodeElimination()(OptimizeLayoutTransform()(Before))
    relax.analysis.check_well_formed(After, check_ty=True)

    for mod in [Before, After]:
        executable = tvm.compile(mod, target="c")
        vm = relax.VirtualMachine(executable, tvm_tirx.cpu())
        output = vm["main"](tvm_tirx.runtime.tensor(values)).numpy()
        np.testing.assert_array_equal(output, expected)
        assert output.dtype == expected.dtype

    if inverse:
        expected_bb = relax.BlockBuilder()
        with expected_bb.function("main", [x]):
            expected_bb.emit_func_output(x)
        assert_structural_equal(After, expected_bb.get())
    else:
        assert_structural_equal(After, Before)


if __name__ == "__main__":
    tvm_tirx.testing.main()
