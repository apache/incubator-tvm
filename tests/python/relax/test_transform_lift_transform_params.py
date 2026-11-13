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
# ruff: noqa: F401, F841

from __future__ import annotations

import numpy as np
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing

pytest.importorskip("scipy")  # tvm.topi.testing imports scipy

import tvm.topi.testing
from tvm import relax
from tvm.relax import script as R
from tvm.s_tir import script as Ts
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


@pytest.mark.parametrize("consume_params", [True, False])
def test_basic(consume_params):
    @tvm_tirx.script.ir_module
    class Before:
        @Ts.function
        def transform_layout_IOHW_to_OIHW(
            w1: T.Tensor((3, 16, 3, 3), "float32"), out: T.Tensor((16, 3, 3, 3), "float32")
        ) -> None:
            for ax0, ax1, ax2, ax3 in T.grid(16, 3, 3, 3):
                with Ts.sblock("layout_transform"):
                    o, i, h, w = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    out[o, i, h, w] = w1[i, o, h, w]

        @R.function
        def main(
            x: R.Tensor((1, 3, 224, 224), "float32"),
            w1: R.Tensor((3, 16, 3, 3), "float32"),
            w2: R.Tensor((16, 16, 3, 3), "float32"),
        ) -> R.Tensor((1, 16, 224, 224), "float32"):
            R.func_attr({"num_input": 1})
            cls = Before
            with R.dataflow():
                w1_transformed = R.call_tir(
                    cls.transform_layout_IOHW_to_OIHW,
                    w1,
                    ty_args=[R.Tensor((16, 3, 3, 3), "float32")],
                )
                conv1 = R.nn.conv2d(
                    x, w1_transformed, padding=(1, 1), data_layout="NCHW", kernel_layout="OIHW"
                )
                conv2 = R.nn.conv2d(
                    conv1, w2, padding=(1, 1), data_layout="NCHW", kernel_layout="OIHW"
                )
                R.output(conv2)
            return conv2

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((1, 3, 224, 224), dtype="float32"),
            w2: R.Tensor((16, 16, 3, 3), dtype="float32"),
            w1_transformed: R.Tensor((16, 3, 3, 3), dtype="float32"),
        ) -> R.Tensor((1, 16, 224, 224), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                conv1: R.Tensor((1, 16, 224, 224), dtype="float32") = R.nn.conv2d(
                    x,
                    w1_transformed,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                )
                conv2: R.Tensor((1, 16, 224, 224), dtype="float32") = R.nn.conv2d(
                    conv1,
                    w2,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                )
                R.output(conv2)
            return conv2

        @Ts.function
        def transform_layout_IOHW_to_OIHW(
            w1: T.Tensor((3, 16, 3, 3), "float32"), out: T.Tensor((16, 3, 3, 3), "float32")
        ):
            for ax0, ax1, ax2, ax3 in T.grid(16, 3, 3, 3):
                with Ts.sblock("layout_transform"):
                    o, i, h, w = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(w1[i, o, h, w])
                    Ts.writes(out[o, i, h, w])
                    out[o, i, h, w] = w1[i, o, h, w]

        @R.function
        def main_transform_params(
            params: R.Tuple(
                R.Tensor((3, 16, 3, 3), dtype="float32"), R.Tensor((16, 16, 3, 3), dtype="float32")
            ),
        ) -> R.Tuple(
            R.Tensor((16, 16, 3, 3), dtype="float32"), R.Tensor((16, 3, 3, 3), dtype="float32")
        ):
            R.func_attr({"num_input": 0})
            cls = Expected
            with R.dataflow():
                lv1: R.Tensor((3, 16, 3, 3), dtype="float32") = params[0]
                lv2 = R.call_tir(
                    cls.transform_layout_IOHW_to_OIHW,
                    (lv1,),
                    ty_args=[R.Tensor((16, 3, 3, 3), dtype="float32")],
                )
                lv: R.Tensor((16, 16, 3, 3), dtype="float32") = params[1]
                gv: R.Tuple(
                    R.Tensor((16, 16, 3, 3), dtype="float32"),
                    R.Tensor((16, 3, 3, 3), dtype="float32"),
                ) = (lv, lv2)
                R.output(gv)
            return gv

    @tvm_tirx.script.ir_module
    class ExpectedConsumeParams:
        @R.function
        def main(
            x: R.Tensor((1, 3, 224, 224), dtype="float32"),
            w2: R.Tensor((16, 16, 3, 3), dtype="float32"),
            w1_transformed: R.Tensor((16, 3, 3, 3), dtype="float32"),
        ) -> R.Tensor((1, 16, 224, 224), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                conv1: R.Tensor((1, 16, 224, 224), dtype="float32") = R.nn.conv2d(
                    x,
                    w1_transformed,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                )
                conv2: R.Tensor((1, 16, 224, 224), dtype="float32") = R.nn.conv2d(
                    conv1,
                    w2,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                )
                R.output(conv2)
            return conv2

        @Ts.function
        def transform_layout_IOHW_to_OIHW(
            w1: T.Tensor((3, 16, 3, 3), "float32"), out: T.Tensor((16, 3, 3, 3), "float32")
        ):
            for ax0, ax1, ax2, ax3 in T.grid(16, 3, 3, 3):
                with Ts.sblock("layout_transform"):
                    o, i, h, w = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(w1[i, o, h, w])
                    Ts.writes(out[o, i, h, w])
                    out[o, i, h, w] = w1[i, o, h, w]

        @R.function
        def main_transform_params(
            params: R.Tuple(
                R.Tensor((3, 16, 3, 3), dtype="float32"), R.Tensor((16, 16, 3, 3), dtype="float32")
            ),
        ) -> R.Tuple(
            R.Tensor((16, 16, 3, 3), dtype="float32"), R.Tensor((16, 3, 3, 3), dtype="float32")
        ):
            R.func_attr({"num_input": 0})
            cls = ExpectedConsumeParams
            with R.dataflow():
                lv1: R.Tensor((3, 16, 3, 3), dtype="float32") = params[0]
                _1: R.Tuple = R.call_pure_packed(
                    "vm.builtin.tuple_reset_item",
                    params,
                    R.prim_value(T.int32(0)),
                    ty_args=(R.Tuple,),
                )
                lv2 = R.call_tir(
                    cls.transform_layout_IOHW_to_OIHW,
                    (lv1,),
                    ty_args=[R.Tensor((16, 3, 3, 3), dtype="float32")],
                )
                lv: R.Tensor((16, 16, 3, 3), dtype="float32") = params[1]
                _2: R.Tuple = R.call_pure_packed(
                    "vm.builtin.tuple_reset_item",
                    params,
                    R.prim_value(T.int32(1)),
                    ty_args=(R.Tuple,),
                )
                gv: R.Tuple(
                    R.Tensor((16, 16, 3, 3), dtype="float32"),
                    R.Tensor((16, 3, 3, 3), dtype="float32"),
                ) = (lv, lv2)
                R.output(gv)
            return gv

    mod = Before
    expected = Expected if not consume_params else ExpectedConsumeParams
    with tvm_tirx.ir.transform.PassContext(
        config={"relax.lift_transform_params.consume_params": consume_params}
    ):
        after = relax.transform.LiftTransformParams()(mod)
    tvm_tirx.ir.assert_structural_equal(after, expected)

    names_after = [param.name for param in after["main"].params]
    names_expected = [param.name for param in expected["main"].params]
    assert names_after == names_expected


def test_tuple():
    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((1, 16, 224, 224), "float32"), w1: R.Tensor((16, 16, 3, 3), "float32")
        ) -> R.Tensor((1, 16, 224, 224), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                l0 = (w1,)
                l1 = (l0,)
                l2 = l1[0]
                l3 = l2[0]
                conv1 = R.nn.conv2d(x, l3, padding=(1, 1), data_layout="NCHW", kernel_layout="OIHW")
                conv2 = R.nn.conv2d(
                    conv1, w1, padding=(1, 1), data_layout="NCHW", kernel_layout="OIHW"
                )
                R.output(conv2)
            return conv2

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((1, 16, 224, 224), dtype="float32"),
            param0: R.Tensor((16, 16, 3, 3), dtype="float32"),
            param1: R.Tensor((16, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((1, 16, 224, 224), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                conv1: R.Tensor((1, 16, 224, 224), dtype="float32") = R.nn.conv2d(
                    x,
                    param1,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                )
                conv2: R.Tensor((1, 16, 224, 224), dtype="float32") = R.nn.conv2d(
                    conv1,
                    param0,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                )
                R.output(conv2)
            return conv2

        @R.function
        def main_transform_params(
            params: R.Tuple(R.Tensor((16, 16, 3, 3), dtype="float32")),
        ) -> R.Tuple(
            R.Tensor((16, 16, 3, 3), dtype="float32"), R.Tensor((16, 16, 3, 3), dtype="float32")
        ):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                l3 = params[0]
                w1 = params[0]
                gv = (w1, l3)
                R.output(gv)
            return gv

    mod = Before
    after = relax.transform.LiftTransformParams()(mod)
    tvm_tirx.ir.assert_structural_equal(after, Expected)


def test_condition():
    """Test case that the conditional statement can't be lifted"""

    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((1, 16, 224, 224), "float32"),
            w1: R.Tensor((16, 16, 3, 3), "float32"),
            w2: R.Tensor((16, 16, 3, 3), "float32"),
            cond: R.Tensor((), "bool"),
        ) -> R.Tensor((1, 16, 224, 224), "float32"):
            R.func_attr({"num_input": 1})
            if cond:
                w = w1
            else:
                w = w2
            with R.dataflow():
                conv1 = R.nn.conv2d(x, w, padding=(1, 1), data_layout="NCHW", kernel_layout="OIHW")
                R.output(conv1)
            return conv1

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main_transform_params(
            params: R.Tuple(
                R.Tensor((16, 16, 3, 3), dtype="float32"),
                R.Tensor((16, 16, 3, 3), dtype="float32"),
                R.Tensor((), dtype="bool"),
            ),
        ) -> R.Tuple(
            R.Tensor((16, 16, 3, 3), dtype="float32"),
            R.Tensor((16, 16, 3, 3), dtype="float32"),
            R.Tensor((), dtype="bool"),
        ):
            R.func_attr({"num_input": 0})
            return params

        @R.function
        def main(
            x: R.Tensor((1, 16, 224, 224), "float32"),
            param0: R.Tensor((16, 16, 3, 3), dtype="float32"),
            param1: R.Tensor((16, 16, 3, 3), dtype="float32"),
            param2: R.Tensor((), dtype="bool"),
        ) -> R.Tensor((1, 16, 224, 224), "float32"):
            R.func_attr({"num_input": 1})
            if param2:
                w: R.Tensor((16, 16, 3, 3), dtype="float32") = param0
            else:
                w: R.Tensor((16, 16, 3, 3), dtype="float32") = param1
            with R.dataflow():
                conv1 = R.nn.conv2d(x, w, padding=(1, 1), data_layout="NCHW", kernel_layout="OIHW")
                R.output(conv1)
            return conv1

    mod = Before
    after = relax.transform.LiftTransformParams()(mod)
    tvm_tirx.ir.assert_structural_equal(after, Expected)


def test_multiple_functions():
    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def func1(
            x: R.Tensor((256, 256), "float32"),
            w1: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                w1_t = R.permute_dims(w1, [1, 0])
                y = R.matmul(x, w1_t)
                R.output(y)
            return y

        @R.function
        def func2(
            x: R.Tensor((256, 256), "float32"),
            w1: R.Tensor((128, 256), "float32"),
        ) -> R.Tensor((256, 128), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                w1_t = R.permute_dims(w1, [1, 0])
                y = R.matmul(x, w1_t)
                R.output(y)
            return y

        @R.function
        def func3(
            x: R.Tensor((256, 256), "float32"),
            w1: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            with R.dataflow():
                w1_t = R.permute_dims(w1, [1, 0])
                y = R.matmul(x, w1_t)
                R.output(y)
            return y

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def func1(
            x: R.Tensor((256, 256), dtype="float32"),
            param0: R.Tensor((256, 256), dtype="float32"),
        ) -> R.Tensor((256, 256), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                y: R.Tensor((256, 256), dtype="float32") = R.matmul(x, param0)
                R.output(y)
            return y

        @R.function
        def func1_transform_params(
            params: R.Tuple(R.Tensor((256, 256), dtype="float32")),
        ) -> R.Tuple(R.Tensor((256, 256), dtype="float32")):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                lv: R.Tensor((256, 256), dtype="float32") = params[0]
                lv1: R.Tensor((256, 256), dtype="float32") = R.permute_dims(lv, axes=[1, 0])
                gv: R.Tuple(R.Tensor((256, 256), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

        @R.function
        def func2(
            x: R.Tensor((256, 256), dtype="float32"),
            param0: R.Tensor((256, 128), dtype="float32"),
        ) -> R.Tensor((256, 128), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                y: R.Tensor((256, 128), dtype="float32") = R.matmul(x, param0)
                R.output(y)
            return y

        @R.function
        def func2_transform_params(
            params: R.Tuple(R.Tensor((128, 256), dtype="float32")),
        ) -> R.Tuple(R.Tensor((256, 128), dtype="float32")):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                lv: R.Tensor((128, 256), dtype="float32") = params[0]
                lv1: R.Tensor((256, 128), dtype="float32") = R.permute_dims(lv, axes=[1, 0])
                gv: R.Tuple(R.Tensor((256, 128), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

        @R.function
        def func3(
            x: R.Tensor((256, 256), dtype="float32"), w1: R.Tensor((256, 256), dtype="float32")
        ) -> R.Tensor((256, 256), dtype="float32"):
            with R.dataflow():
                w1_t: R.Tensor((256, 256), dtype="float32") = R.permute_dims(w1, axes=[1, 0])
                y: R.Tensor((256, 256), dtype="float32") = R.matmul(x, w1_t)
                R.output(y)
            return y

    mod = Before
    after = relax.transform.LiftTransformParams()(mod)
    tvm_tirx.ir.assert_structural_equal(after, Expected)


def test_share_identical_transform_across_multiple_functions():
    """Like test_multiple_functions, but producing a single transform_params

    `func1` and `func2` contain the same values `w1_t` and `w2_t`.
    When `shared_transform=True`, all eligible publicly-exposed
    functions must be usable with the same shared transform.
    """

    @I.ir_module
    class Before:
        @R.function
        def func1(
            x: R.Tensor((256, 256), "float32"),
            w1: R.Tensor((256, 256), "float32"),
            w2: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                w1_t = R.permute_dims(w1)
                y1 = R.matmul(x, w1_t)
                w2_t = R.permute_dims(w2)
                y2 = R.matmul(x, w2_t)
                output = R.add(y1, y2)
                R.output(output)
            return output

        @R.function
        def func2(
            x: R.Tensor((256, 256), "float32"),
            w1: R.Tensor((256, 256), "float32"),
            w2: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                w1_t = R.permute_dims(w1)
                y1 = R.matmul(x, w1_t)
                w2_t = R.permute_dims(w2)
                y2 = R.matmul(x, w2_t)
                output = R.multiply(y1, y2)
                R.output(output)
            return output

    @I.ir_module
    class Expected:
        @R.function
        def transform_params(
            params: R.Tuple(
                R.Tensor((256, 256), dtype="float32"),
                R.Tensor((256, 256), dtype="float32"),
            ),
        ):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                w1 = params[0]
                w1_t = R.permute_dims(w1)
                w2 = params[1]
                w2_t = R.permute_dims(w2)
                output = (w1_t, w2_t)
                R.output(output)
            return output

        @R.function
        def func1(
            x: R.Tensor((256, 256), "float32"),
            w1_t: R.Tensor((256, 256), "float32"),
            w2_t: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                y1 = R.matmul(x, w1_t)
                y2 = R.matmul(x, w2_t)
                output = R.add(y1, y2)
                R.output(output)
            return output

        @R.function
        def func2(
            x: R.Tensor((256, 256), "float32"),
            w1_t: R.Tensor((256, 256), "float32"),
            w2_t: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                y1 = R.matmul(x, w1_t)
                y2 = R.matmul(x, w2_t)
                output = R.multiply(y1, y2)
                R.output(output)
            return output

    after = relax.transform.LiftTransformParams(shared_transform=True)(Before)
    tvm_tirx.ir.assert_structural_equal(after, Expected)


def test_incompatible_weights_in_shared_transform_raises_error():
    """Model weights must have matched shape for shared_transform

    Here, `func1` accepts one model weight, but `func2` accepts two.
    """

    @I.ir_module
    class Before:
        @R.function
        def func1(
            x: R.Tensor((256, 256), "float32"),
            w1: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                w1_t = R.permute_dims(w1)
                y1 = R.matmul(x, w1_t)
                output = y1
                R.output(output)
            return output

        @R.function
        def func2(
            x: R.Tensor((256, 256), "float32"),
            w1: R.Tensor((256, 256), "float32"),
            w2: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                w1_t = R.permute_dims(w1)
                y1 = R.matmul(x, w1_t)
                w2_t = R.permute_dims(w2)
                y2 = R.matmul(x, w2_t)
                output = R.multiply(y1, y2)
                R.output(output)
            return output

    with pytest.raises(RuntimeError):
        relax.transform.LiftTransformParams(shared_transform=True)(Before)


def test_incompatible_shape_in_shared_transform_raises_error():
    """Model weights must have matched shape for shared_transform

    Here, `func1` accepts `w1` and `w2` with shape `[256,256]`, but `func2`
    requires shape `[128, 256]`.
    """

    @I.ir_module
    class Before:
        @R.function
        def func1(
            x: R.Tensor((256, 256), "float32"),
            w1: R.Tensor((256, 256), "float32"),
            w2: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                w1_t = R.permute_dims(w1)
                y1 = R.matmul(x, w1_t)
                w2_t = R.permute_dims(w2)
                y2 = R.matmul(x, w2_t)
                output = R.add(y1, y2)
                R.output(output)
            return output

        @R.function
        def func2(
            x: R.Tensor((256, 256), "float32"),
            w1: R.Tensor((128, 256), "float32"),
            w2: R.Tensor((128, 256), "float32"),
        ) -> R.Tensor((256, 128), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                w1_t = R.permute_dims(w1)
                y1 = R.matmul(x, w1_t)
                w2_t = R.permute_dims(w2)
                y2 = R.matmul(x, w2_t)
                output = R.multiply(y1, y2)
                R.output(output)
            return output

    with pytest.raises(RuntimeError):
        relax.transform.LiftTransformParams(shared_transform=True)(Before)


def test_incompatible_dtype_in_shared_transform_raises_error():
    """Model weights must have matched dtype for shared_transform

    Here, `func1` accepts `w1` and `w2` with "float32" dtype, but
    `func2` requires "float16".
    """

    @I.ir_module
    class Before:
        @R.function
        def func1(
            x: R.Tensor((256, 256), "float32"),
            w1: R.Tensor((256, 256), "float32"),
            w2: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                w1_t = R.permute_dims(w1)
                y1 = R.matmul(x, w1_t)
                w2_t = R.permute_dims(w2)
                y2 = R.matmul(x, w2_t)
                output = R.add(y1, y2)
                R.output(output)
            return output

        @R.function
        def func2(
            x: R.Tensor((256, 256), "float16"),
            w1: R.Tensor((128, 256), "float16"),
            w2: R.Tensor((128, 256), "float16"),
        ) -> R.Tensor((256, 128), "float16"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                w1_t = R.permute_dims(w1)
                y1 = R.matmul(x, w1_t)
                w2_t = R.permute_dims(w2)
                y2 = R.matmul(x, w2_t)
                output = R.multiply(y1, y2)
                R.output(output)
            return output

    with pytest.raises(RuntimeError):
        relax.transform.LiftTransformParams(shared_transform=True)(Before)


def test_share_transform_across_multiple_functions_has_intersection_of_transforms():
    """Like test_multiple_functions, but producing a single transform_params

    In `func1`, both `w1_t` and `w2_t` could be lifted out.  In
    `func2`, only `w1_t` could be lifted out of the function.
    Therefore, the shared `transform_params` can pre-compute `w1_t`,
    but must preserve `w2`.

    When `shared_transform=True`, all eligible publicly-exposed
    functions must be usable with the same shared transform.
    """

    @I.ir_module
    class Before:
        @R.function
        def func1(
            x: R.Tensor((256, 256), "float32"),
            w1: R.Tensor((256, 256), "float32"),
            w2: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                w1_t = R.permute_dims(w1)
                y1 = R.matmul(x, w1_t)
                w2_t = R.permute_dims(w2)
                y2 = R.matmul(x, w2_t)
                output = R.add(y1, y2)
                R.output(output)
            return output

        @R.function
        def func2(
            x: R.Tensor((256, 256), "float32"),
            w1: R.Tensor((256, 256), "float32"),
            w2: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                w1_t = R.permute_dims(w1)
                y1 = R.matmul(x, w1_t)
                y2 = Before.fused_permute_dims_matmul(x, w2)
                output = R.add(y1, y2)
                R.output(output)
            return output

        @R.function(private=True)
        def fused_permute_dims_matmul(
            x: R.Tensor((256, 256), "float32"),
            weight: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            with R.dataflow():
                weight_t = R.permute_dims(weight)
                y = R.matmul(x, weight_t)
                R.output(y)
            return y

    @I.ir_module
    class Expected:
        @R.function
        def transform_params(
            params: R.Tuple(
                R.Tensor((256, 256), dtype="float32"),
                R.Tensor((256, 256), dtype="float32"),
            ),
        ):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                w1 = params[0]
                w1_t = R.permute_dims(w1)
                w2 = params[1]
                output = (w2, w1_t)
                R.output(output)
            return output

        @R.function
        def func1(
            x: R.Tensor((256, 256), "float32"),
            w2: R.Tensor((256, 256), "float32"),
            w1_t: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                y1 = R.matmul(x, w1_t)
                w2_t = R.permute_dims(w2)
                y2 = R.matmul(x, w2_t)
                output = R.add(y1, y2)
                R.output(output)
            return output

        @R.function
        def func2(
            x: R.Tensor((256, 256), "float32"),
            w2: R.Tensor((256, 256), "float32"),
            w1_t: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                y1 = R.matmul(x, w1_t)
                y2 = Expected.fused_permute_dims_matmul(x, w2)
                output = R.add(y1, y2)
                R.output(output)
            return output

        @R.function(private=True)
        def fused_permute_dims_matmul(
            x: R.Tensor((256, 256), "float32"),
            weight: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            with R.dataflow():
                weight_t = R.permute_dims(weight)
                y = R.matmul(x, weight_t)
                R.output(y)
            return y

    after = relax.transform.LiftTransformParams(shared_transform=True)(Before)
    tvm_tirx.ir.assert_structural_equal(after, Expected)


def test_share_transforms_with_different_binding_order():
    """Like test_share_transform_across_multiple_functions, but the
    lifted bindings are in different order for each function.

    Both `func1` and `func2` compute the same value for `w1_t` and
    `w2_t`.  However, the bindings occur in different orders.  The
    shared `transform_params` can pre-compute both `w1_t` and `w2_t`,
    even though they occur in different orders.

    For consistency in testing and pre-computing weights, the order of
    `transform_params` should be deterministic.  When lifting from a
    single function, the bindings in `transform_params` may be
    determined from the order in that function.  When lifting from
    multiple functions, the order should be deterministic.  Since
    `IRModule::functions` has unspecified order, the order in this
    test assumes that public functions are visited in alphabetical
    order by name.
    """

    @I.ir_module
    class Before:
        @R.function
        def func1(
            x: R.Tensor((256, 256), "float32"),
            w1: R.Tensor((256, 256), "float32"),
            w2: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                w2_t = R.permute_dims(w2)
                w1_t = R.permute_dims(w1)
                y1 = R.matmul(x, w1_t)
                y2 = R.matmul(x, w2_t)
                output = R.add(y1, y2)
                R.output(output)
            return output

        @R.function
        def func2(
            x: R.Tensor((256, 256), "float32"),
            w1: R.Tensor((256, 256), "float32"),
            w2: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                w1_t = R.permute_dims(w1)
                y1 = R.matmul(x, w1_t)
                w2_t = R.permute_dims(w2)
                y2 = R.matmul(x, w2_t)
                output = R.multiply(y1, y2)
                R.output(output)
            return output

    @I.ir_module
    class Expected:
        @R.function
        def transform_params(
            params: R.Tuple(
                R.Tensor((256, 256), dtype="float32"),
                R.Tensor((256, 256), dtype="float32"),
            ),
        ):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                w2 = params[1]
                w2_t = R.permute_dims(w2)
                w1 = params[0]
                w1_t = R.permute_dims(w1)

                output = (w2_t, w1_t)
                R.output(output)
            return output

        @R.function
        def func1(
            x: R.Tensor((256, 256), "float32"),
            w2_t: R.Tensor((256, 256), "float32"),
            w1_t: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                y1 = R.matmul(x, w1_t)
                y2 = R.matmul(x, w2_t)
                output = R.add(y1, y2)
                R.output(output)
            return output

        @R.function
        def func2(
            x: R.Tensor((256, 256), "float32"),
            w2_t: R.Tensor((256, 256), "float32"),
            w1_t: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                y1 = R.matmul(x, w1_t)
                y2 = R.matmul(x, w2_t)
                output = R.multiply(y1, y2)
                R.output(output)
            return output

    after = relax.transform.LiftTransformParams(shared_transform=True)(Before)
    tvm_tirx.ir.assert_structural_equal(after, Expected)


def test_share_transforms_resulting_in_identical_functions():
    """Functions in the public interface must be preserved

    When lifting functions, the resulting functions may be identical.
    Even though the `relax.BlockBuilder` de-duplicates identical
    functions, functions that are part of the IRModule's public
    interface must be preserved.
    """

    @I.ir_module
    class Before:
        @R.function
        def func1(
            x: R.Tensor((256, 256), "float32"),
            w1: R.Tensor((256, 256), "float32"),
            w2: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                w2_t = R.permute_dims(w2)
                w1_t = R.permute_dims(w1)
                y1 = R.matmul(x, w1_t)
                y2 = R.matmul(x, w2_t)
                output = R.add(y1, y2)
                R.output(output)
            return output

        @R.function
        def func2(
            x: R.Tensor((256, 256), "float32"),
            w1: R.Tensor((256, 256), "float32"),
            w2: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                w1_t = R.permute_dims(w1)
                y1 = R.matmul(x, w1_t)
                w2_t = R.permute_dims(w2)
                y2 = R.matmul(x, w2_t)
                output = R.add(y1, y2)
                R.output(output)
            return output

    @I.ir_module
    class Expected:
        @R.function
        def transform_params(
            params: R.Tuple(
                R.Tensor((256, 256), dtype="float32"),
                R.Tensor((256, 256), dtype="float32"),
            ),
        ):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                w2 = params[1]
                w2_t = R.permute_dims(w2)
                w1 = params[0]
                w1_t = R.permute_dims(w1)
                output = (w2_t, w1_t)
                R.output(output)
            return output

        @R.function
        def func1(
            x: R.Tensor((256, 256), "float32"),
            w2_t: R.Tensor((256, 256), "float32"),
            w1_t: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                y1 = R.matmul(x, w1_t)
                y2 = R.matmul(x, w2_t)
                output = R.add(y1, y2)
                R.output(output)
            return output

        @R.function
        def func2(
            x: R.Tensor((256, 256), "float32"),
            w2_t: R.Tensor((256, 256), "float32"),
            w1_t: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                y1 = R.matmul(x, w1_t)
                y2 = R.matmul(x, w2_t)
                output = R.add(y1, y2)
                R.output(output)
            return output

    after = relax.transform.LiftTransformParams(shared_transform=True)(Before)
    tvm_tirx.ir.assert_structural_equal(after, Expected)


def test_share_transform_across_specified_functions():
    """Like test_multiple_functions, but producing a single transform_params

    In `func1`, both `w1_t` and `w2_t` could be lifted out.  In
    `func2`, only `w1_t` could be lifted out of the function.
    Therefore, the shared `transform_params` can pre-compute `w1_t`,
    but must preserve `w2`.

    If `func3` were included in the `transform_params`, the same logic
    would prevent `w1_t` from being computed in the shared
    `transform_params`.  However, the
    `shared_transform=['func1','func2']` argument means that `func3`
    does not have any parameter transformations lifted out.
    """

    @I.ir_module
    class Before:
        @R.function
        def func1(
            x: R.Tensor((256, 256), "float32"),
            w1: R.Tensor((256, 256), "float32"),
            w2: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                w1_t = R.permute_dims(w1)
                y1 = R.matmul(x, w1_t)
                w2_t = R.permute_dims(w2)
                y2 = R.matmul(x, w2_t)
                output = R.add(y1, y2)
                R.output(output)
            return output

        @R.function
        def func2(
            x: R.Tensor((256, 256), "float32"),
            w1: R.Tensor((256, 256), "float32"),
            w2: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                w1_t = R.permute_dims(w1)
                y1 = R.matmul(x, w1_t)
                y2 = Before.fused_permute_dims_matmul(x, w2)
                output = R.add(y1, y2)
                R.output(output)
            return output

        @R.function
        def func3(
            x: R.Tensor((256, 256), "float32"),
            w1: R.Tensor((256, 256), "float32"),
            w2: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                y1 = Before.fused_permute_dims_matmul(x, w1)
                y2 = Before.fused_permute_dims_matmul(x, w2)
                output = R.add(y1, y2)
                R.output(output)
            return output

        @R.function(private=True)
        def fused_permute_dims_matmul(
            x: R.Tensor((256, 256), "float32"),
            weight: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            with R.dataflow():
                weight_t = R.permute_dims(weight)
                y = R.matmul(x, weight_t)
                R.output(y)
            return y

    @I.ir_module
    class Expected:
        @R.function
        def transform_params(
            params: R.Tuple(
                R.Tensor((256, 256), dtype="float32"),
                R.Tensor((256, 256), dtype="float32"),
            ),
        ):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                w1 = params[0]
                w1_t = R.permute_dims(w1)
                w2 = params[1]
                output = (w2, w1_t)
                R.output(output)
            return output

        @R.function
        def func1(
            x: R.Tensor((256, 256), "float32"),
            w2: R.Tensor((256, 256), "float32"),
            w1_t: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                y1 = R.matmul(x, w1_t)
                w2_t = R.permute_dims(w2)
                y2 = R.matmul(x, w2_t)
                output = R.add(y1, y2)
                R.output(output)
            return output

        @R.function
        def func2(
            x: R.Tensor((256, 256), "float32"),
            w2: R.Tensor((256, 256), "float32"),
            w1_t: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                y1 = R.matmul(x, w1_t)
                y2 = Expected.fused_permute_dims_matmul(x, w2)
                output = R.add(y1, y2)
                R.output(output)
            return output

        @R.function
        def func3(
            x: R.Tensor((256, 256), "float32"),
            w1: R.Tensor((256, 256), "float32"),
            w2: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                y1 = Expected.fused_permute_dims_matmul(x, w1)
                y2 = Expected.fused_permute_dims_matmul(x, w2)
                output = R.add(y1, y2)
                R.output(output)
            return output

        @R.function(private=True)
        def fused_permute_dims_matmul(
            x: R.Tensor((256, 256), "float32"),
            weight: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            with R.dataflow():
                weight_t = R.permute_dims(weight)
                y = R.matmul(x, weight_t)
                R.output(y)
            return y

    after = relax.transform.LiftTransformParams(shared_transform=["func1", "func2"])(Before)
    tvm_tirx.ir.assert_structural_equal(after, Expected)


def test_share_transform_with_unused_parameter():
    """Like test_share_transform_across_specified_functions, but not
    all functions use every model weight.

    In `func1`, both `w1_t` and `w2_t` could be lifted out.  In
    `func2`, only `w1_t` could be lifted out of the function.
    Normally, the `w2` parameter would need to be preserved, as `w2_t`
    is only generated in one of the functions.  However, `func2`
    doesn't use `w2` at all, and so `w2_t` can still be pre-computed.

    For example, a `embed_vocab` function would only use the embedding
    weights.  It could accept the full set of model weights for
    consistency, but any transformations performed on unused weights
    in other functions can still be lifted out.
    """

    @I.ir_module
    class Before:
        @R.function
        def func1(
            x: R.Tensor((256, 256), "float32"),
            w1: R.Tensor((256, 256), "float32"),
            w2: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                w1_t = R.permute_dims(w1)
                y1 = R.matmul(x, w1_t)
                w2_t = R.permute_dims(w2)
                y2 = R.matmul(x, w2_t)
                output = R.add(y1, y2)
                R.output(output)
            return output

        @R.function
        def func2(
            x: R.Tensor((256, 256), "float32"),
            w1: R.Tensor((256, 256), "float32"),
            w2: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                w1_t = R.permute_dims(w1)
                y1 = R.matmul(x, w1_t)
                R.output(y1)
            return y1

    @I.ir_module
    class Expected:
        @R.function
        def transform_params(
            params: R.Tuple(
                R.Tensor((256, 256), dtype="float32"),
                R.Tensor((256, 256), dtype="float32"),
            ),
        ):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                w1 = params[0]
                w1_t = R.permute_dims(w1)
                w2 = params[1]
                output = (w2, w1_t)
                R.output(output)
            return output

        @R.function
        def func1(
            x: R.Tensor((256, 256), "float32"),
            w2: R.Tensor((256, 256), "float32"),
            w1_t: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                y1 = R.matmul(x, w1_t)
                w2_t = R.permute_dims(w2)
                y2 = R.matmul(x, w2_t)
                output = R.add(y1, y2)
                R.output(output)
            return output

        @R.function
        def func2(
            x: R.Tensor((256, 256), "float32"),
            w2: R.Tensor((256, 256), "float32"),
            w1_t: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                y1 = R.matmul(x, w1_t)
                R.output(y1)
            return y1

    after = relax.transform.LiftTransformParams(shared_transform=True)(Before)
    tvm_tirx.ir.assert_structural_equal(after, Expected)


@pytest.mark.xfail
def test_share_transform_with_no_shared_preprocessing():
    """Like test_share_transform_with_unused_parameter, but each
    function uses a single model weight.

    In `func1`, `w2_t` can be lifted out and `w1` is unused.  In
    `func2`, `w1_t` can be lifted out, and `w2` is unused.  In their
    shared `transform_params`, both `w1_t` and `w2_t` can be computed.

    For consistency in testing and pre-computing weights, the order of
    `transform_params` should be deterministic.  When lifting from a
    single function, the bindings in `transform_params` may be
    determined from the order in that function.  When lifting from
    multiple functions, the order should be deterministic.  Since
    `IRModule::functions` has unspecified order, the order in this
    test assumes that public functions are visited in alphabetical
    order by name.
    """

    @I.ir_module
    class Before:
        @R.function
        def func1(
            x: R.Tensor((256, 256), "float32"),
            w1: R.Tensor((256, 256), "float32"),
            w2: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                w2_t = R.permute_dims(w2)
                y2 = R.matmul(x, w2_t)
                R.output(y2)
            return y2

        @R.function
        def func2(
            x: R.Tensor((256, 256), "float32"),
            w1: R.Tensor((256, 256), "float32"),
            w2: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                w1_t = R.permute_dims(w1)
                y1 = R.matmul(x, w1_t)
                R.output(y1)
            return y1

    @I.ir_module
    class Expected:
        @R.function
        def transform_params(
            params: R.Tuple(
                R.Tensor((256, 256), dtype="float32"),
                R.Tensor((256, 256), dtype="float32"),
            ),
        ):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                w1 = params[0]
                w1_t = R.permute_dims(w1)
                w2 = params[1]
                w2_t = R.permute_dims(w2)
                output = (w2_t, w1_t)
                R.output(output)
            return output

        @R.function
        def func1(
            x: R.Tensor((256, 256), "float32"),
            w2_t: R.Tensor((256, 256), "float32"),
            w1_t: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                y2 = R.matmul(x, w2_t)
                R.output(y2)
            return y2

        @R.function
        def func2(
            x: R.Tensor((256, 256), "float32"),
            w2_t: R.Tensor((256, 256), "float32"),
            w1_t: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                y1 = R.matmul(x, w1_t)
                R.output(y1)
            return y1

    after = relax.transform.LiftTransformParams(shared_transform=True)(Before)
    tvm_tirx.ir.assert_structural_equal(after, Expected)


def test_stop_lifting():
    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def func1(
            x: R.Tensor((256, 256), "float32"),
            w1: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                w1_t = R.permute_dims(w1, [1, 0])
                w1_t1 = R.builtin.stop_lift_params(w1_t)
                w1_add = R.add(w1_t1, R.const(1, "float32"))
                y = R.matmul(x, w1_add)
                R.output(y)
            return y

    @I.ir_module
    class Expected:
        @R.function
        def func1(
            x: R.Tensor((256, 256), dtype="float32"),
            param0: R.Tensor((256, 256), dtype="float32"),
        ) -> R.Tensor((256, 256), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                w1_add: R.Tensor((256, 256), dtype="float32") = R.add(param0, R.const(1, "float32"))
                y: R.Tensor((256, 256), dtype="float32") = R.matmul(x, w1_add)
                R.output(y)
            return y

        @R.function
        def func1_transform_params(
            params: R.Tuple(R.Tensor((256, 256), dtype="float32")),
        ) -> R.Tuple(R.Tensor((256, 256), dtype="float32")):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                lv: R.Tensor((256, 256), dtype="float32") = params[0]
                lv1: R.Tensor((256, 256), dtype="float32") = R.permute_dims(lv, axes=[1, 0])
                gv: R.Tuple(R.Tensor((256, 256), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    mod = Before
    after = relax.transform.LiftTransformParams()(mod)
    tvm_tirx.ir.assert_structural_equal(after, Expected)


def test_symbolic_var_1():
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main(shape: R.Shape([n])):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                zeros = R.zeros((n, n), "float32")
            return shape

    n = T.dynamic("n", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main_transform_params(params: R.Tuple) -> R.Tuple:
            R.func_attr({"num_input": 0})
            # All instance of the empty tuple are normalized to be
            # in-line.
            return ()

        @R.function
        def main(shape: R.Shape([n])) -> R.Shape([n]):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                zeros: R.Tensor((n, n), dtype="float32") = R.zeros(R.shape([n, n]), dtype="float32")
                R.output()
            return shape

    mod = Before
    after = relax.transform.LiftTransformParams()(mod)
    tvm_tirx.ir.assert_structural_equal(after, Expected)


def test_symbolic_var_2():
    n_zeros = T.dynamic("n", "int64")
    n_main = T.dynamic("n", "int64")

    @I.ir_module
    class Before:
        @Ts.function
        def zeros(T_full: T.Tensor((n_zeros, n_zeros), "float32")):
            T.func_attr({"tirx.noalias": True})

            for ax0, ax1 in T.grid(n_zeros, n_zeros):
                with Ts.sblock("T_full"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads()
                    Ts.writes(T_full[v_ax0, v_ax1])
                    T_full[v_ax0, v_ax1] = T.float32(0)

        @R.function
        def main(shape: R.Shape([n_main])) -> R.Shape([n_main]):
            R.func_attr({"num_input": 1})
            cls = Before
            with R.dataflow():
                zeros = R.call_tir(
                    cls.zeros, R.tuple(), ty_args=[R.Tensor((n_main, n_main), dtype="float32")]
                )
                R.output()
            return shape

    n_zeros = T.dynamic("n", "int64")
    n_main = T.dynamic("n", "int64")

    @I.ir_module
    class Expected:
        @Ts.function
        def zeros(T_full: T.Tensor((n_zeros, n_zeros), "float32")):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            for ax0, ax1 in T.grid(n_zeros, n_zeros):
                with Ts.sblock("T_full"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads()
                    Ts.writes(T_full[v_ax0, v_ax1])
                    T_full[v_ax0, v_ax1] = T.float32(0)

        @R.function
        def main_transform_params(params: R.Tuple) -> R.Tuple:
            R.func_attr({"num_input": 0})
            return ()

        @R.function
        def main(shape: R.Shape([n_main])) -> R.Shape([n_main]):
            R.func_attr({"num_input": 1})
            cls = Expected
            with R.dataflow():
                zeros = R.call_tir(
                    cls.zeros, R.tuple(), ty_args=[R.Tensor((n_main, n_main), dtype="float32")]
                )
                R.output()
            return shape

    mod = Before
    after = relax.transform.LiftTransformParams()(mod)
    tvm_tirx.ir.assert_structural_equal(after, Expected)


def test_symbolic_var_from_shape():
    slice_index = T.dynamic("slice_index", "int64")

    @I.ir_module
    class Before:
        @R.function
        def main(
            A: R.Tensor([16, 16], "int32"),
            B: R.Tensor([16, 16], "int32"),
            shape: R.Shape([slice_index]),
        ) -> R.Tensor([16], "int32"):
            R.func_attr({"num_input": 1})
            cls = Before
            with R.dataflow():
                B_slice = R.call_tir(
                    cls.slice,
                    [B, slice_index],
                    ty_args=[R.Tensor([16], dtype="int32")],
                )
                A_slice = R.call_tir(
                    cls.slice,
                    [A, slice_index],
                    ty_args=[R.Tensor([16], dtype="int32")],
                )
                A_scale = R.multiply(A_slice, B_slice)
                R.output(A_scale)
            return A_scale

        @Ts.function(private=True)
        def slice(
            Input_2d: T.Tensor(shape=[16, 16], dtype="int32"),
            slice_index: T.int64,
            Output_Slice: T.Tensor(shape=[16], dtype="int32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for j in range(16):
                with Ts.sblock("T_full"):
                    vj = Ts.axis.remap("S", [j])
                    Output_Slice[vj] = Input_2d[slice_index, vj]

    slice_index_main = T.dynamic("slice_index", "int64")
    slice_index_main_transform_params = T.dynamic("slice_index", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            A: R.Tensor([16, 16], "int32"),
            shape: R.Shape([slice_index_main]),
            B_slice: R.Tensor([16], "int32"),
        ) -> R.Tensor([16], "int32"):
            R.func_attr({"num_input": 1})
            cls = Expected
            with R.dataflow():
                A_slice = R.call_tir(
                    cls.slice,
                    [A, slice_index_main],
                    ty_args=[R.Tensor([16], dtype="int32")],
                )
                A_scale = R.multiply(A_slice, B_slice)
                R.output(A_scale)
            return A_scale

        @R.function
        def main_transform_params(
            params: R.Tuple(
                R.Tensor([16, 16], "int32"), R.Shape([slice_index_main_transform_params])
            ),
        ):
            R.func_attr({"num_input": 0})
            cls = Expected
            with R.dataflow():
                B = params[0]
                # extra_symbolic_vars = params[1]
                B_slice = R.call_tir(
                    cls.slice,
                    [B, slice_index_main_transform_params],
                    ty_args=[R.Tensor([16], dtype="int32")],
                )
                output = (R.ShapeExpr([slice_index_main_transform_params]), B_slice)
                R.output(output)
            return output

        @Ts.function(private=True)
        def slice(
            Input_2d: T.Tensor(shape=[16, 16], dtype="int32"),
            slice_index: T.int64,
            Output_Slice: T.Tensor(shape=[16], dtype="int32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for j in range(16):
                with Ts.sblock("T_full"):
                    vj = Ts.axis.remap("S", [j])
                    Output_Slice[vj] = Input_2d[slice_index, vj]

    mod = Before
    after = relax.transform.LiftTransformParams()(mod)
    tvm_tirx.ir.assert_structural_equal(Expected, after)


def test_symbolic_var_in_param_shape():
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((1, 16, 224, n), "float32"),
            w1: R.Tensor((16, m, 3, 3), "float32"),
            w2: R.Tensor((16, m, 3, 3), "float32"),
        ) -> R.Tensor((1, 16, 224, n), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                zeros = R.zeros((n, n), "float32")
                w1 = R.add(w1, R.const(1, "float32"))
                conv1 = R.nn.conv2d(x, w1, padding=(1, 1), data_layout="NCHW", kernel_layout="OIHW")
                conv2 = R.nn.conv2d(
                    conv1, w2, padding=(1, 1), data_layout="NCHW", kernel_layout="OIHW"
                )
                R.output(conv2)
            return conv2

    m_main_transform_params = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")
    m_main = T.dynamic("m", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main_transform_params(
            params: R.Tuple(
                R.Tensor((16, m_main_transform_params, 3, 3), dtype="float32"),
                R.Tensor((16, m_main_transform_params, 3, 3), dtype="float32"),
            ),
        ) -> R.Tuple(
            R.Tensor((16, m_main_transform_params, 3, 3), dtype="float32"),
            R.Tensor((16, m_main_transform_params, 3, 3), dtype="float32"),
        ):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                lv1: R.Tensor((16, m_main_transform_params, 3, 3), dtype="float32") = params[0]
                lv2: R.Tensor((16, m_main_transform_params, 3, 3), dtype="float32") = R.add(
                    lv1, R.const(1, "float32")
                )
                lv: R.Tensor((16, m_main_transform_params, 3, 3), dtype="float32") = params[1]
                gv: R.Tuple(
                    R.Tensor((16, m_main_transform_params, 3, 3), dtype="float32"),
                    R.Tensor((16, m_main_transform_params, 3, 3), dtype="float32"),
                ) = (lv, lv2)
                R.output(gv)
            return gv

        @R.function
        def main(
            x: R.Tensor((1, 16, 224, n), dtype="float32"),
            transformed_param_0: R.Tensor((16, m_main, 3, 3), dtype="float32"),
            transformed_param_1: R.Tensor((16, m_main, 3, 3), dtype="float32"),
        ) -> R.Tensor((1, 16, 224, n), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                zeros: R.Tensor((n, n), dtype="float32") = R.zeros(R.shape([n, n]), dtype="float32")
                conv1: R.Tensor((1, 16, 224, n), dtype="float32") = R.nn.conv2d(
                    x,
                    transformed_param_1,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                )
                conv2: R.Tensor((1, 16, 224, n), dtype="float32") = R.nn.conv2d(
                    conv1,
                    transformed_param_0,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                )
                R.output(conv2)
            return conv2

    mod = Before
    after = relax.transform.LiftTransformParams()(mod)
    tvm_tirx.ir.assert_structural_equal(after, Expected)


# not supported yet
@pytest.mark.xfail
def test_symbolic_var_defined_in_params_but_used_in_weights():
    """A symbolic variable's occurrence in the weights may not define it

    In order to be a source of definition, a symbolic variable in the
    parameters must occur as a distinct parameter, as a tensor shape
    `R.Tensor(["var"])` or an explicit `R.Shape(["var"])`.  A variable
    that is part of a larger
    expression, such as `R.Tensor(["m * n"])`, are variable usages,
    not variable definitions.
    """

    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([m, n], "float32"),
            weight: R.Tensor([m * n], "float32"),
        ) -> R.Tensor([m, n], "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                weight = R.add(weight, R.const(1, "float32"))
                weight = R.reshape(weight, [m, n])
                output = R.multiply(x, weight)
                R.output(output)
            return output

    k = T.dynamic("k", "int64")
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main_transform_params(params: R.Tuple(R.Tensor((k,), dtype="float32"))) -> R.Tuple(
            R.Tensor(dtype="float32", ndim=1)
        ):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                lv: R.Tensor((k,), dtype="float32") = params[0]
                gv: R.Tuple(R.Tensor((k,), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

        @R.function
        def main(
            x: R.Tensor((m, n), dtype="float32"),
            transformed_param_0: R.Tensor(dtype="float32", ndim=1),
        ) -> R.Tensor((m, n), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor(dtype="float32", ndim=1) = transformed_param_0
                weight: R.Tensor(dtype="float32", ndim=1) = R.add(lv, R.const(1, "float32"))
                weight_1: R.Tensor((m, n), dtype="float32") = R.reshape(weight, R.shape([m, n]))
                output: R.Tensor((m, n), dtype="float32") = R.multiply(x, weight_1)
                R.output(output)
            return output

    After = relax.transform.LiftTransformParams()(Before)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


def test_only_lift_when_variable_uses_constants():
    """A variable that has no inputs should not be lifted

    For example, `R.zeros`, or the result of allocation function
    calls.
    """

    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main(
            A: R.Tensor([16], "int32"),
            B: R.Tensor([16], "int32"),
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                offset = R.ones([16], "int32")
                A_offset = R.add(A, offset)
                B_offset = R.add(B, offset)
                output = R.multiply(A_offset, B_offset)
                R.output(output)
            return output

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            A: R.Tensor([16], "int32"),
            B_offset: R.Tensor([16], "int32"),
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                offset = R.ones([16], "int32")
                A_offset = R.add(A, offset)
                output = R.multiply(A_offset, B_offset)
                R.output(output)
            return output

        @R.function
        def main_transform_params(params: R.Tuple([R.Tensor([16], "int32")])):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                offset = R.ones([16], "int32")
                B = params[0]
                B_offset = R.add(B, offset)
                output = (B_offset,)
                R.output(output)
            return output

    mod = Before
    after = relax.transform.LiftTransformParams()(mod)
    tvm_tirx.ir.assert_structural_equal(after, Expected)


@pytest.mark.parametrize("shared_transform", [True, False])
def test_lift_transform_is_idempotent(shared_transform):
    """Multiple applicates of LiftTransformParams are allowed"""

    batch_size = T.dynamic("batch_size", "int64")
    lora_rank = T.dynamic("lora_rank", "int64")

    @I.ir_module
    class Module:
        @R.function
        def main(
            state: R.Tensor([batch_size, 4096], "float16"),
            base_weights: R.Tensor([4096, 4096], "float16"),
            lora_A: R.Tensor([4096, lora_rank], "float16"),
            lora_B: R.Tensor([lora_rank, 4096], "float16"),
        ):
            R.func_attr({"num_input": 1})
            folded_weights = base_weights + R.matmul(lora_A, lora_B)
            output = R.matmul(state, folded_weights)
            return output

    transform = relax.transform.LiftTransformParams(shared_transform=shared_transform)

    AfterOneRound = transform(Module)
    assert len(AfterOneRound.functions) == 2

    AfterTwoRounds = transform(AfterOneRound)
    assert len(AfterTwoRounds.functions) == 2

    tvm_tirx.ir.assert_structural_equal(AfterOneRound, AfterTwoRounds)


def test_lift_transform_when_one_already_exists():
    """If the module already contains `transform_params`, the
    functions are composed together"""

    batch_size = T.dynamic("batch_size", "int64")
    lora_rank_main = T.dynamic("lora_rank", "int64")
    lora_rank_main_transform_params = T.dynamic("lora_rank", "int64")

    @I.ir_module
    class Module:
        @R.function
        def main(
            state: R.Tensor([batch_size, 4096], "float16"),
            base_weights: R.Tensor([4096, 4096], "float16"),
            lora_A: R.Tensor([4096, lora_rank_main], "float16"),
            lora_B: R.Tensor([lora_rank_main, 4096], "float16"),
        ):
            R.func_attr({"num_input": 1})
            folded_weights = base_weights + R.matmul(lora_A, lora_B)
            output = R.matmul(state, folded_weights)
            return output

        @R.function
        def main_transform_params(
            model_params: R.Tuple(
                R.Tensor([4096, 4096], "float16"),
                R.Tensor([4096, lora_rank_main_transform_params], "float16"),
                R.Tensor([lora_rank_main_transform_params, 4096], "float16"),
            ),
        ):
            R.func_attr({"num_input": 0})
            return model_params

    transform = relax.transform.LiftTransformParams(shared_transform=False)
    after_lift_with_previous_identity_function = transform(Module)

    del Module["main_transform_params"]
    after_lift_without_previous_identity_function = transform(Module)

    tvm_tirx.ir.assert_structural_equal(
        after_lift_without_previous_identity_function,
        after_lift_with_previous_identity_function,
    )


def test_lift_transform_with_primitive_param_used_by_model_tensor():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor(dtype="float32", ndim=1),
            extent: T.int64,
            weight: R.Tensor([extent], "float32"),
        ):
            R.func_attr({"num_input": 1})
            transformed = R.multiply(weight, weight)
            out = R.add(x, transformed)
            return out

    old_extent = Before["main"].params[1]
    after = relax.transform.LiftTransformParams(shared_transform=False)(Before)
    relax.analysis.check_well_formed(after)

    transform = after["main_transform_params"]
    bundled_ty = transform.params[0].ty
    assert isinstance(bundled_ty, relax.TupleType)
    assert isinstance(bundled_ty.fields[0], tvm_tirx.ir.PrimType)
    assert isinstance(bundled_ty.fields[1], relax.TensorType)
    assert bundled_ty.fields[1].shape is None
    assert bundled_ty.fields[1].ndim == 1

    bindings = [binding for block in transform.body.blocks for binding in block.bindings]
    extent_index = next(
        i
        for i, binding in enumerate(bindings)
        if isinstance(binding, relax.VarBinding) and tvm_tirx.ir.is_prim_var(binding.var)
    )
    match_cast_index = next(
        i for i, binding in enumerate(bindings) if isinstance(binding, relax.MatchCast)
    )
    extent = bindings[extent_index].var
    weight = bindings[match_cast_index]
    assert extent_index < match_cast_index
    assert weight.ty.shape[0].same_as(extent)

    signature_types = [param.ty for param in transform.params] + [transform.ret_ty]
    assert all(
        all(not var.same_as(old_extent) for var in relax.analysis.tir_vars_in_type(ty))
        for ty in signature_types
    )


if __name__ == "__main__":
    tvm_tirx.testing.main()
