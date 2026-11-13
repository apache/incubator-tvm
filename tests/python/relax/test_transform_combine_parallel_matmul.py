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
# ruff: noqa: E731, F401
import pytest

import tvm
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm.relax.script import ir_builder as relax_builder
from tvm.relax.transform import CombineParallelMatmul
from tvm_tirx import tirx
from tvm_tirx.script import tirx as T
from tvm_tirx.script.ir_builder import IRBuilder


def get_parallel_matmul(
    num_branches,
    lhs_shape=(640, 640),
    rhs_shape=(640, 640),
    with_bias=None,
    activation=None,
):
    dtype = "float32"

    activation_map = {"relu": R.nn.relu, "gelu": R.nn.gelu}

    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name_("main")
            x = R.arg_("x", R.Tensor(lhs_shape, dtype))

            rhs = []
            bias = []

            for i in range(num_branches):
                rhs.append(R.arg_("y", R.Tensor(rhs_shape, dtype)))

                if with_bias and with_bias[i]:
                    bias.append(R.arg_("bias", R.Tensor((rhs_shape[1],), dtype)))
                else:
                    bias.append(None)

            with R.dataflow() as frame:
                branches = []

                for i, r in enumerate(rhs):
                    result = R.emit(R.matmul(x, r, out_dtype=dtype))
                    if bias[i] is not None:
                        result = R.emit(result + bias[i])
                    if activation and activation[i]:
                        result = R.emit(activation_map[activation[i]](result))

                    branches.append(result)

                R.output(R.emit(R.concat(branches, axis=1)))

            R.func_ret_value(frame.output_vars[0])

    func = builder.get()
    return tvm_tirx.IRModule({"main": func})


def test_simple():
    mod_orig = get_parallel_matmul(1)
    mod = CombineParallelMatmul()(mod_orig)

    tvm_tirx.ir.assert_structural_equal(mod, mod_orig)

    mod = get_parallel_matmul(3)
    mod = CombineParallelMatmul()(mod)

    @R.function
    def expected1(
        x: R.Tensor((640, 640), dtype="float32"),
        y: R.Tensor((640, 640), dtype="float32"),
        y_1: R.Tensor((640, 640), dtype="float32"),
        y_2: R.Tensor((640, 640), dtype="float32"),
    ) -> R.Tensor((640, 1920), dtype="float32"):
        with R.dataflow():
            lv = R.concat((y, y_1, y_2), axis=1)
            lv1 = R.matmul(x, lv, out_dtype="float32")
            lv2 = R.split(lv1, indices_or_sections=[640, 1280], axis=1)
            lv_1 = lv2[0]
            lv1_1 = lv2[1]
            lv2_1 = lv2[2]
            lv3 = R.concat((lv_1, lv1_1, lv2_1), axis=1)
            R.output(lv3)
        return lv3

    tvm_tirx.ir.assert_structural_equal(mod["main"], expected1.with_attr("global_symbol", "main"))

    # Test a batched LHS case, slicing is done on the axis 2
    mod = get_parallel_matmul(3, lhs_shape=(2, 1024, 640))
    mod = CombineParallelMatmul()(mod)

    @R.function
    def expected2(
        x: R.Tensor((2, 1024, 640), dtype="float32"),
        y: R.Tensor((640, 640), dtype="float32"),
        y_1: R.Tensor((640, 640), dtype="float32"),
        y_2: R.Tensor((640, 640), dtype="float32"),
    ) -> R.Tensor((2, 3072, 640), dtype="float32"):
        with R.dataflow():
            lv = R.concat((y, y_1, y_2), axis=1)
            lv1 = R.matmul(x, lv, out_dtype="float32")
            lv2 = R.split(lv1, indices_or_sections=[640, 1280], axis=2)
            lv_1 = lv2[0]
            lv1_1 = lv2[1]
            lv2_1 = lv2[2]
            lv3 = R.concat((lv_1, lv1_1, lv2_1), axis=1)
            R.output(lv3)
        return lv3

    tvm_tirx.ir.assert_structural_equal(mod["main"], expected2.with_attr("global_symbol", "main"))


def test_bias():
    mod = get_parallel_matmul(3, with_bias=[True, True, True])
    mod = CombineParallelMatmul()(mod)

    @R.function
    def expected1(
        x: R.Tensor((640, 640), dtype="float32"),
        y: R.Tensor((640, 640), dtype="float32"),
        bias: R.Tensor((640,), dtype="float32"),
        y_1: R.Tensor((640, 640), dtype="float32"),
        bias_1: R.Tensor((640,), dtype="float32"),
        y_2: R.Tensor((640, 640), dtype="float32"),
        bias_2: R.Tensor((640,), dtype="float32"),
    ) -> R.Tensor((640, 1920), dtype="float32"):
        with R.dataflow():
            lv = R.concat((y, y_1, y_2), axis=1)
            lv1 = R.matmul(x, lv, out_dtype="float32")
            lv2 = R.concat((bias, bias_1, bias_2), axis=0)
            lv3 = R.add(lv1, lv2)
            lv4 = R.split(lv3, indices_or_sections=[640, 1280], axis=1)
            lv1_1 = lv4[0]
            lv3_1 = lv4[1]
            lv5 = lv4[2]
            lv6 = R.concat((lv1_1, lv3_1, lv5), axis=1)
            R.output(lv6)
        return lv6

    tvm_tirx.ir.assert_structural_equal(mod["main"], expected1.with_attr("global_symbol", "main"))

    mod = get_parallel_matmul(3, with_bias=[True, False, True])
    mod = CombineParallelMatmul()(mod)

    @R.function
    def expected2(
        x: R.Tensor((640, 640), dtype="float32"),
        y: R.Tensor((640, 640), dtype="float32"),
        bias: R.Tensor((640,), dtype="float32"),
        y_1: R.Tensor((640, 640), dtype="float32"),
        y_2: R.Tensor((640, 640), dtype="float32"),
        bias_1: R.Tensor((640,), dtype="float32"),
    ) -> R.Tensor((640, 1920), dtype="float32"):
        with R.dataflow():
            lv = R.concat((y, y_1, y_2), axis=1)
            lv1 = R.matmul(x, lv, out_dtype="float32")
            lv2 = R.split(lv1, indices_or_sections=[640, 1280], axis=1)
            lv_1 = lv2[0]
            lv1_1 = R.add(lv_1, bias)
            lv2_1 = lv2[1]
            lv3 = lv2[2]
            lv4 = R.add(lv3, bias_1)
            lv5 = R.concat((lv1_1, lv2_1, lv4), axis=1)
            R.output(lv5)
        return lv5

    tvm_tirx.ir.assert_structural_equal(mod["main"], expected2.with_attr("global_symbol", "main"))


def test_activation():
    mod = get_parallel_matmul(3, activation=["relu", "relu", "relu"])
    mod = CombineParallelMatmul()(mod)

    @R.function
    def expected1(
        x: R.Tensor((640, 640), dtype="float32"),
        y: R.Tensor((640, 640), dtype="float32"),
        y_1: R.Tensor((640, 640), dtype="float32"),
        y_2: R.Tensor((640, 640), dtype="float32"),
    ) -> R.Tensor((640, 1920), dtype="float32"):
        with R.dataflow():
            lv = R.concat((y, y_1, y_2), axis=1)
            lv1 = R.matmul(x, lv, out_dtype="float32")
            lv2 = R.nn.relu(lv1)
            lv3 = R.split(lv2, indices_or_sections=[640, 1280], axis=1)
            lv1_1 = lv3[0]
            lv3_1 = lv3[1]
            lv5 = lv3[2]
            lv6 = R.concat((lv1_1, lv3_1, lv5), axis=1)
            R.output(lv6)
        return lv6

    tvm_tirx.ir.assert_structural_equal(mod["main"], expected1.with_attr("global_symbol", "main"))

    mod = get_parallel_matmul(3, activation=["gelu", "relu", "relu"])
    mod = CombineParallelMatmul()(mod)

    @R.function
    def expected2(
        x: R.Tensor((640, 640), dtype="float32"),
        y: R.Tensor((640, 640), dtype="float32"),
        y_1: R.Tensor((640, 640), dtype="float32"),
        y_2: R.Tensor((640, 640), dtype="float32"),
    ) -> R.Tensor((640, 1920), dtype="float32"):
        with R.dataflow():
            lv = R.concat((y, y_1, y_2), axis=1)
            lv1 = R.matmul(x, lv, out_dtype="float32")
            lv2 = R.split(lv1, indices_or_sections=[640, 1280], axis=1)
            lv_1 = lv2[0]
            lv1_1 = R.nn.gelu(lv_1)
            lv2_1 = lv2[1]
            lv3 = R.nn.relu(lv2_1)
            lv4 = lv2[2]
            lv5 = R.nn.relu(lv4)
            lv6 = R.concat((lv1_1, lv3, lv5), axis=1)
            R.output(lv6)
        return lv6

    tvm_tirx.ir.assert_structural_equal(mod["main"], expected2.with_attr("global_symbol", "main"))

    mod = get_parallel_matmul(3, activation=["relu", None, None])
    mod = CombineParallelMatmul()(mod)

    @R.function
    def expected3(
        x: R.Tensor((640, 640), dtype="float32"),
        y: R.Tensor((640, 640), dtype="float32"),
        y_1: R.Tensor((640, 640), dtype="float32"),
        y_2: R.Tensor((640, 640), dtype="float32"),
    ) -> R.Tensor((640, 1920), dtype="float32"):
        with R.dataflow():
            lv = R.concat((y, y_1, y_2), axis=1)
            lv1 = R.matmul(x, lv, out_dtype="float32")
            lv2 = R.split(lv1, indices_or_sections=[640, 1280], axis=1)

            lv_1 = lv2[0]
            lv1_1 = R.nn.relu(lv_1)
            lv2_1 = lv2[1]
            lv3 = lv2[2]
            lv4 = R.concat((lv1_1, lv2_1, lv3), axis=1)
            R.output(lv4)
        return lv4

    tvm_tirx.ir.assert_structural_equal(mod["main"], expected3.with_attr("global_symbol", "main"))


def test_bias_activation():
    mod = get_parallel_matmul(3, with_bias=[True, True, True], activation=["relu", "relu", "relu"])
    mod = CombineParallelMatmul()(mod)

    @R.function
    def expected1(
        x: R.Tensor((640, 640), dtype="float32"),
        y: R.Tensor((640, 640), dtype="float32"),
        bias: R.Tensor((640,), dtype="float32"),
        y_1: R.Tensor((640, 640), dtype="float32"),
        bias_1: R.Tensor((640,), dtype="float32"),
        y_2: R.Tensor((640, 640), dtype="float32"),
        bias_2: R.Tensor((640,), dtype="float32"),
    ) -> R.Tensor((640, 1920), dtype="float32"):
        with R.dataflow():
            lv = R.concat((y, y_1, y_2), axis=1)
            lv1 = R.matmul(x, lv, out_dtype="float32")
            lv2 = R.concat((bias, bias_1, bias_2), axis=0)
            lv3 = R.add(lv1, lv2)
            lv4 = R.nn.relu(lv3)
            lv5 = R.split(lv4, indices_or_sections=[640, 1280], axis=1)
            lv2_1 = lv5[0]
            lv5_1 = lv5[1]
            lv8 = lv5[2]
            lv9 = R.concat((lv2_1, lv5_1, lv8), axis=1)
            R.output(lv9)
        return lv9

    tvm_tirx.ir.assert_structural_equal(mod["main"], expected1.with_attr("global_symbol", "main"))

    mod = get_parallel_matmul(3, with_bias=[True, True, True], activation=["relu", None, "relu"])
    mod = CombineParallelMatmul()(mod)

    @R.function
    def expected2(
        x: R.Tensor((640, 640), dtype="float32"),
        y: R.Tensor((640, 640), dtype="float32"),
        bias: R.Tensor((640,), dtype="float32"),
        y_1: R.Tensor((640, 640), dtype="float32"),
        bias_1: R.Tensor((640,), dtype="float32"),
        y_2: R.Tensor((640, 640), dtype="float32"),
        bias_2: R.Tensor((640,), dtype="float32"),
    ) -> R.Tensor((640, 1920), dtype="float32"):
        with R.dataflow():
            lv = R.concat((y, y_1, y_2), axis=1)
            lv1 = R.matmul(x, lv, out_dtype="float32")
            lv2 = R.concat((bias, bias_1, bias_2), axis=0)
            lv3 = R.add(lv1, lv2)
            lv4 = R.split(lv3, indices_or_sections=[640, 1280], axis=1)
            lv1_1 = lv4[0]
            lv2_1 = R.nn.relu(lv1_1)
            lv4_1 = lv4[1]
            lv6 = lv4[2]
            lv7 = R.nn.relu(lv6)
            lv8 = R.concat((lv2_1, lv4_1, lv7), axis=1)
            R.output(lv8)
        return lv8

    tvm_tirx.ir.assert_structural_equal(mod["main"], expected2.with_attr("global_symbol", "main"))

    mod = get_parallel_matmul(3, with_bias=[True, False, True], activation=["relu", None, "relu"])
    mod = CombineParallelMatmul()(mod)

    @R.function
    def expected3(
        x: R.Tensor((640, 640), dtype="float32"),
        y: R.Tensor((640, 640), dtype="float32"),
        bias: R.Tensor((640,), dtype="float32"),
        y_1: R.Tensor((640, 640), dtype="float32"),
        y_2: R.Tensor((640, 640), dtype="float32"),
        bias_1: R.Tensor((640,), dtype="float32"),
    ) -> R.Tensor((640, 1920), dtype="float32"):
        with R.dataflow():
            lv = R.concat((y, y_1, y_2), axis=1)
            lv1 = R.matmul(x, lv, out_dtype="float32")
            lv2 = R.split(lv1, indices_or_sections=[640, 1280], axis=1)
            lv_1 = lv2[0]
            lv1_1 = R.add(lv_1, bias)
            lv2_1 = R.nn.relu(lv1_1)
            lv3 = lv2[1]
            lv4 = lv2[2]
            lv5 = R.add(lv4, bias_1)
            lv6 = R.nn.relu(lv5)
            lv7 = R.concat((lv2_1, lv3, lv6), axis=1)
            R.output(lv7)
        return lv7

    tvm_tirx.ir.assert_structural_equal(mod["main"], expected3.with_attr("global_symbol", "main"))


def test_rhs_batched():
    @R.function(private=True)
    def before(
        x: R.Tensor((1024, 640), "float32"),
        w0: R.Tensor((2, 640, 640), "float32"),
        w1: R.Tensor((640, 640), "float32"),
        w2: R.Tensor((2, 640, 640), "float32"),
        w3: R.Tensor((3, 4, 640, 640), "float32"),
    ):
        with R.dataflow():
            lv0 = R.matmul(x, w0)
            lv1 = R.matmul(x, w1)
            lv2 = R.matmul(x, w2)
            lv3 = R.matmul(x, w3)
            out = (lv0, lv1, lv2, lv3)
            R.output(out)
        return out

    after = CombineParallelMatmul()(tvm_tirx.IRModule.from_expr(before))["main"]

    @R.function(private=True)
    def expected(
        x: R.Tensor((1024, 640), dtype="float32"),
        w0: R.Tensor((2, 640, 640), dtype="float32"),
        w1: R.Tensor((640, 640), dtype="float32"),
        w2: R.Tensor((2, 640, 640), dtype="float32"),
        w3: R.Tensor((3, 4, 640, 640), dtype="float32"),
    ):
        with R.dataflow():
            lv = R.concat((w0, w2), axis=2)
            lv1 = R.matmul(x, lv, out_dtype="float32")
            lv2 = R.split(lv1, indices_or_sections=[640], axis=2)
            lv0 = lv2[0]
            lv1_1 = R.matmul(x, w1, out_dtype=None)
            lv2_1 = lv2[1]
            lv3 = R.matmul(x, w3, out_dtype=None)
            out = lv0, lv1_1, lv2_1, lv3
            R.output(out)
        return out

    tvm_tirx.ir.assert_structural_equal(after, expected)

    @tvm_tirx.script.ir_module
    class four_matmul_incompatible_batches:
        @R.function
        def main(
            x: R.Tensor((1024, 640), "float32"),
            w0: R.Tensor((2, 640, 640), "float32"),
            w1: R.Tensor((3, 640, 640), "float32"),
            w2: R.Tensor((2, 640, 640), "float32"),
            w3: R.Tensor((2, 640, 640), "float32"),
        ):
            with R.dataflow():
                lv0 = R.matmul(x, w0)
                lv1 = R.matmul(x, w1)
                lv2 = R.matmul(x, w2)
                lv3 = R.matmul(x, w3)
                out = (lv0, lv1, lv2, lv3)
                R.output(out)
            return out

    mod = CombineParallelMatmul()(four_matmul_incompatible_batches)
    # For now, when rhs matrices have the same rank but different batch sizes, we don't
    # combine any of them.
    tvm_tirx.ir.assert_structural_equal(mod, four_matmul_incompatible_batches)


def test_multiple_combine():
    @R.function(private=True)
    def before(
        x1: R.Tensor((2, 1024, 640), "float32"),
        x2: R.Tensor((2, 1024, 640), "float32"),
        w0: R.Tensor((640, 640), "float32"),
        w1: R.Tensor((640, 640), "float32"),
        w2: R.Tensor((640, 640), "float32"),
        w3: R.Tensor((640, 640), "float32"),
        w4: R.Tensor((640, 640), "float32"),
        b0: R.Tensor((640,), "float32"),
        b1: R.Tensor((640,), "float32"),
    ):
        with R.dataflow():
            lv0 = R.matmul(x1, w0)
            lv3 = R.matmul(x2, w3)
            lv1 = R.matmul(x1, w1)
            lv5 = R.add(lv3, b0)
            lv2 = R.matmul(x1, w2)
            lv4 = R.matmul(x2, w4)
            lv6 = R.add(lv4, b1)
            out = (lv0, lv1, lv2, lv5, lv6)
            R.output(out)
        return out

    after = CombineParallelMatmul()(tvm_tirx.IRModule.from_expr(before))["main"]

    @R.function(private=True)
    def expected(
        x1: R.Tensor((2, 1024, 640), dtype="float32"),
        x2: R.Tensor((2, 1024, 640), dtype="float32"),
        w0: R.Tensor((640, 640), dtype="float32"),
        w1: R.Tensor((640, 640), dtype="float32"),
        w2: R.Tensor((640, 640), dtype="float32"),
        w3: R.Tensor((640, 640), dtype="float32"),
        w4: R.Tensor((640, 640), dtype="float32"),
        b0: R.Tensor((640,), dtype="float32"),
        b1: R.Tensor((640,), dtype="float32"),
    ):
        with R.dataflow():
            lv = R.concat((w0, w1, w2), axis=1)
            lv1 = R.matmul(x1, lv, out_dtype="float32")
            lv2 = R.split(lv1, indices_or_sections=[640, 1280], axis=2)
            lv0 = lv2[0]
            lv1_1 = lv2[1]
            lv_1 = R.concat((w3, w4), axis=1)
            lv1_2 = R.matmul(x2, lv_1, out_dtype="float32")
            lv2_1 = R.concat((b0, b1), axis=0)
            lv3 = R.add(lv1_2, lv2_1)
            lv4 = R.split(lv3, indices_or_sections=[640], axis=2)
            lv5 = lv4[0]
            lv2_2 = lv2[2]
            lv6 = lv4[1]
            out = lv0, lv1_1, lv2_2, lv5, lv6
            R.output(out)
        return out

    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_check():
    @R.function(private=True)
    def before(
        x1: R.Tensor((2, 1024, 640), "float32"),
        x2: R.Tensor((2, 1024, 640), "float32"),
        w0: R.Tensor((640, 640), "float32"),
        w1: R.Tensor((640, 640), "float32"),
        w2: R.Tensor((640, 640), "float32"),
        w3: R.Tensor((640, 640), "float32"),
        w4: R.Tensor((640, 640), "float32"),
    ):
        with R.dataflow():
            lv0 = R.matmul(x1, w0)
            lv1 = R.matmul(x1, w1)
            lv2 = R.matmul(x1, w2)
            lv3 = R.matmul(x2, w3)
            lv4 = R.matmul(x2, w4)
            out = (lv0, lv1, lv2, lv3, lv4)
            R.output(out)
        return out

    check = lambda *inp: len(inp[1]) > 2  # Ignore branches with two matmuls
    after = CombineParallelMatmul(check)(tvm_tirx.IRModule.from_expr(before))["main"]

    @R.function(private=True)
    def expected(
        x1: R.Tensor((2, 1024, 640), dtype="float32"),
        x2: R.Tensor((2, 1024, 640), dtype="float32"),
        w0: R.Tensor((640, 640), dtype="float32"),
        w1: R.Tensor((640, 640), dtype="float32"),
        w2: R.Tensor((640, 640), dtype="float32"),
        w3: R.Tensor((640, 640), dtype="float32"),
        w4: R.Tensor((640, 640), dtype="float32"),
    ):
        with R.dataflow():
            lv = R.concat((w0, w1, w2), axis=1)
            lv1 = R.matmul(x1, lv, out_dtype="float32")
            lv2 = R.split(lv1, indices_or_sections=[640, 1280], axis=2)
            lv0 = lv2[0]
            lv1_1 = lv2[1]
            lv2_1 = lv2[2]
            lv3 = R.matmul(x2, w3, out_dtype=None)
            lv4 = R.matmul(x2, w4, out_dtype=None)
            out = (lv0, lv1_1, lv2_1, lv3, lv4)
            R.output(out)
        return out

    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_combine_matmul_of_static_and_dynamic_shapes():
    """Combine two matmuls, one with dynamic shape

    The `R.split` operator must have a static list of integer indices
    at which to split the matmul output, because these integer indices
    are stored as operator attributes.  However, the last output can
    still have a dynamic shape.

    """

    M = T.dynamic("M", "int64")

    @R.function(private=True)
    def before(
        x: R.Tensor((2, 1024, 640), "float32"),
        w0: R.Tensor((640, 640), "float32"),
        w1: R.Tensor((640, M), "float32"),
    ):
        with R.dataflow():
            lv0 = R.matmul(x, w0)
            lv1 = R.matmul(x, w1)
            out = (lv0, lv1)
            R.output(out)
        return out

    M = T.dynamic("M", "int64")

    @R.function(private=True)
    def expected(
        x: R.Tensor((2, 1024, 640), dtype="float32"),
        w0: R.Tensor((640, 640), dtype="float32"),
        w1: R.Tensor((640, M), dtype="float32"),
    ) -> R.Tuple(
        R.Tensor((2, 1024, 640), dtype="float32"), R.Tensor((2, 1024, M), dtype="float32")
    ):
        with R.dataflow():
            lv: R.Tensor((640, 640 + M), dtype="float32") = R.concat((w0, w1), axis=1)
            lv1: R.Tensor((2, 1024, 640 + M), dtype="float32") = R.matmul(
                x, lv, out_dtype="float32"
            )
            lv2: R.Tuple(
                R.Tensor((2, 1024, 640), dtype="float32"),
                R.Tensor((2, 1024, M), dtype="float32"),
            ) = R.split(lv1, indices_or_sections=[640], axis=2)
            lv0: R.Tensor((2, 1024, 640), dtype="float32") = lv2[0]
            lv1_1: R.Tensor((2, 1024, M), dtype="float32") = lv2[1]
            out: R.Tuple(
                R.Tensor((2, 1024, 640), dtype="float32"),
                R.Tensor((2, 1024, M), dtype="float32"),
            ) = (lv0, lv1_1)
            R.output(out)
        return out

    after = CombineParallelMatmul()(tvm_tirx.IRModule.from_expr(before))["main"]

    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_combine_matmul_of_dynamic_and_static_shapes():
    """Combine two matmuls, one with dynamic shape

    Like `test_combine_matmul_of_static_and_dynamic_shapes`, but the
    dynamic-shaped matmul is encountered first.  Due to the
    requirements imposed by `R.split` storing the split indices as
    static integers, the static-shaped weights must occur first in the
    concatenated weights.
    """

    M = T.dynamic("M", "int64")

    @R.function(private=True)
    def before(
        x: R.Tensor((2, 1024, 640), "float32"),
        w0: R.Tensor((640, M), "float32"),
        w1: R.Tensor((640, 640), "float32"),
    ):
        with R.dataflow():
            lv0 = R.matmul(x, w0)
            lv1 = R.matmul(x, w1)
            out = (lv0, lv1)
            R.output(out)
        return out

    M = T.dynamic("M", "int64")

    @R.function(private=True)
    def expected(
        x: R.Tensor((2, 1024, 640), dtype="float32"),
        w0: R.Tensor((640, M), dtype="float32"),
        w1: R.Tensor((640, 640), dtype="float32"),
    ) -> R.Tuple(
        R.Tensor((2, 1024, M), dtype="float32"), R.Tensor((2, 1024, 640), dtype="float32")
    ):
        with R.dataflow():
            lv: R.Tensor((640, 640 + M), dtype="float32") = R.concat((w1, w0), axis=1)
            lv1: R.Tensor((2, 1024, 640 + M), dtype="float32") = R.matmul(
                x, lv, out_dtype="float32"
            )
            lv2: R.Tuple(
                R.Tensor((2, 1024, 640), dtype="float32"),
                R.Tensor((2, 1024, M), dtype="float32"),
            ) = R.split(lv1, indices_or_sections=[640], axis=2)
            lv0: R.Tensor((2, 1024, M), dtype="float32") = lv2[1]
            lv1_1: R.Tensor((2, 1024, 640), dtype="float32") = lv2[0]
            out: R.Tuple(
                R.Tensor((2, 1024, M), dtype="float32"),
                R.Tensor((2, 1024, 640), dtype="float32"),
            ) = (lv0, lv1_1)
            R.output(out)
        return out

    after = CombineParallelMatmul()(tvm_tirx.IRModule.from_expr(before))["main"]

    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_limit_one_dynamic_shape_in_combined_matmul():
    """Combine two matmuls, one with dynamic shape

    Like `test_combine_matmul_of_static_and_dynamic_shapes`, but with
    two dynamic weights that could, in principle, be merged together.
    Because `R.split` must have integer indices at which to split,
    only one of the dynamic outputs can be part of the combined
    matmul.
    """

    M = T.dynamic("M", "int64")
    N = T.dynamic("N", "int64")

    @R.function(private=True)
    def before(
        x: R.Tensor((2, 1024, 640), "float32"),
        w0: R.Tensor((640, M), "float32"),
        w1: R.Tensor((640, 640), "float32"),
        w2: R.Tensor((640, N), "float32"),
    ):
        with R.dataflow():
            lv0 = R.matmul(x, w0)
            lv1 = R.matmul(x, w1)
            lv2 = R.matmul(x, w2)
            out = (lv0, lv1, lv2)
            R.output(out)
        return out

    M = T.dynamic("M", "int64")
    N = T.dynamic("N", "int64")

    @R.function(private=True)
    def expected(
        x: R.Tensor((2, 1024, 640), dtype="float32"),
        w0: R.Tensor((640, M), dtype="float32"),
        w1: R.Tensor((640, 640), dtype="float32"),
        w2: R.Tensor((640, N), "float32"),
    ) -> R.Tuple(
        R.Tensor((2, 1024, M), dtype="float32"),
        R.Tensor((2, 1024, 640), dtype="float32"),
        R.Tensor((2, 1024, N), dtype="float32"),
    ):
        with R.dataflow():
            concat_weights = R.concat((w1, w0), axis=1)
            concat_output = R.matmul(x, concat_weights, out_dtype="float32")
            split_output: R.Tuple(
                [R.Tensor([2, 1024, 640], dtype="float32"), R.Tensor([2, 1024, M], dtype="float32")]
            ) = R.split(concat_output, indices_or_sections=[640], axis=2)
            lv0 = split_output[1]
            lv1 = split_output[0]
            lv2 = R.matmul(x, w2)
            out = (lv0, lv1, lv2)
            R.output(out)
        return out

    after = CombineParallelMatmul()(tvm_tirx.IRModule.from_expr(before))["main"]

    tvm_tirx.ir.assert_structural_equal(after, expected)


@pytest.mark.parametrize("float32_branch", [0, 1])
def test_skip_matmuls_with_different_output_dtypes(float32_branch):
    if float32_branch == 0:

        @R.function(private=True)
        def before(
            x: R.Tensor((3, 4), "float16"),
            w0: R.Tensor((4, 5), "float16"),
            w1: R.Tensor((4, 6), "float16"),
        ):
            with R.dataflow():
                y0 = R.matmul(x, w0, out_dtype="float32")
                y1 = R.matmul(x, w1)
                out = (y0, y1)
                R.output(out)
            return out

    else:

        @R.function(private=True)
        def before(
            x: R.Tensor((3, 4), "float16"),
            w0: R.Tensor((4, 5), "float16"),
            w1: R.Tensor((4, 6), "float16"),
        ):
            with R.dataflow():
                y0 = R.matmul(x, w0)
                y1 = R.matmul(x, w1, out_dtype="float32")
                out = (y0, y1)
                R.output(out)
            return out

    after = CombineParallelMatmul()(tvm_tirx.IRModule.from_expr(before))["main"]

    tvm_tirx.ir.assert_structural_equal(after, before)


if __name__ == "__main__":
    tvm_tirx.testing.main()
