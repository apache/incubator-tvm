# ruff: noqa: E402
import pytest

from tvm.relax import ExternFunc as _ExternFunc

pytest.importorskip("tensorflow", reason="tensorflow not available")

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
# pylint: disable=import-self, invalid-name, unused-argument, too-many-lines, len-as-condition, broad-except
# pylint: disable=import-outside-toplevel, redefined-builtin
"""TFLite to Relax converter tests"""

import os

import flatbuffers
import numpy as np
import pytest
import tensorflow as tf
import tflite.Model
from tensorflow.keras import applications as keras_app

import tvm
import tvm.relax.frontend.tflite.tflite_frontend as tflite_frontend
import tvm_tirx
from tvm import relax
from tvm.relax.frontend.tflite import from_tflite
from tvm_tirx.script.parser import ir as I
from tvm_tirx.script.parser import relax as R
from tvm_tirx.script.parser import tirx as T


def _get_mod_from_cfunc(cfunc):
    converter = tf.lite.TFLiteConverter.from_concrete_functions([cfunc])
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    tflite_model_buf = converter.convert()
    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    mod = from_tflite(tflite_model)
    mod["main"] = mod["main"].without_attr("params")
    return mod


def verify(TestClass, expected=None):
    if isinstance(TestClass, type):
        cf = TestClass().func.get_concrete_function()
    else:
        cf = TestClass
    mod = _get_mod_from_cfunc(cf)

    if expected:
        tvm_tirx.ir.assert_structural_equal(mod, expected)

    # Run E2E test only on nightly
    if "CI_ENV_NIGHTLY" not in os.environ:
        return

    # Inputs
    tf_inputs = []
    tvm_inputs = []
    for arg in mod["main"].params:
        shape = tuple(shape_val.value for shape_val in arg.ty.shape.values)
        data = np.random.uniform(0, 1, size=shape).astype(arg.ty.dtype)
        tvm_inputs.append(data)
        tf_inputs.append(tf.constant(data))

    # TF Run
    tf_output = cf(*tf_inputs)

    # TVM Run
    tgt = tvm_tirx.target.Target("c")
    ex = tvm.compile(mod, tgt)
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    vm.set_input("main", *tvm_inputs)
    vm.invoke_stateful("main")
    tvm_output = vm.get_outputs("main")

    if isinstance(tf_output, tuple):
        for tf_out, tvm_out in zip(tf_output, tvm_output):
            np.testing.assert_allclose(tf_out.numpy(), tvm_out.numpy(), rtol=1e-5, atol=1e-5)
    else:
        np.testing.assert_allclose(tf_output.numpy(), tvm_output.numpy(), rtol=1e-5, atol=1e-5)


def _verify_random_with_inputs(cfunc, inputs):
    """E2E verify random ops by shape/dtype and TVM seeded self-consistency."""
    if "CI_ENV_NIGHTLY" not in os.environ:
        return

    mod = _get_mod_from_cfunc(cfunc)
    tvm_inputs = [np.asarray(data) for data in inputs]
    tf_inputs = [tf.constant(data) for data in tvm_inputs]

    tf_output = cfunc(*tf_inputs)

    tgt = tvm_tirx.target.Target("c")
    ex = tvm.compile(mod, tgt)
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())

    def run_tvm():
        vm.set_input("main", *tvm_inputs)
        vm.invoke_stateful("main")
        return vm.get_outputs("main")

    tvm_output = run_tvm()
    tvm_output_again = run_tvm()

    if not isinstance(tf_output, tuple):
        tf_output = (tf_output,)
        tvm_output = (tvm_output,)
        tvm_output_again = (tvm_output_again,)

    for tf_out, tvm_out, tvm_out_again in zip(tf_output, tvm_output, tvm_output_again):
        tf_np = tf_out.numpy()
        tvm_np = tvm_out.numpy()
        assert tvm_np.shape == tf_np.shape
        assert tvm_np.dtype == tf_np.dtype
        np.testing.assert_equal(tvm_np, tvm_out_again.numpy())


def test_add_one_2d():
    class AddOne2D(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(2, 2), dtype=tf.float32)])
        def func(self, x):
            return x + 1

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 2), dtype="float32")) -> R.Tensor((2, 2), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="float32") = R.add(x, R.const(1.0, "float32"))
                R.output(gv)
            return gv

    verify(AddOne2D, Expected)


def test_add_n():
    class AddN(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(2, 2), dtype=tf.float32),
                tf.TensorSpec(shape=(2, 2), dtype=tf.float32),
                tf.TensorSpec(shape=(2, 2), dtype=tf.float32),
            ]
        )
        def func(self, x, y, z):
            return tf.add_n([x, y, z])

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 2), dtype="float32"),
            y: R.Tensor((2, 2), dtype="float32"),
            z: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tensor((2, 2), dtype="float32"):
            R.func_attr({"num_input": 3})
            with R.dataflow():
                lv: R.Tensor((2, 2), dtype="float32") = R.add(x, y)
                gv: R.Tensor((2, 2), dtype="float32") = R.add(lv, z)
                R.output(gv)
            return gv

    verify(AddN, Expected)


def test_cumsum():
    class Cumsum(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(3, 4), dtype=tf.float32),
                tf.TensorSpec(shape=(5, 6), dtype=tf.int32),
            ]
        )
        def func(self, x, y):
            out1 = tf.math.cumsum(x, axis=0)
            out2 = tf.math.cumsum(y, axis=1, exclusive=True)
            return out1, out2

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((3, 4), dtype="float32"),
            y: R.Tensor((5, 6), dtype="int32"),
        ) -> R.Tuple(R.Tensor((3, 4), dtype="float32"), R.Tensor((5, 6), dtype="int32")):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                gv1: R.Tensor((3, 4), dtype="float32") = R.cumsum(
                    x, axis=0, dtype="float32", exclusive=False
                )
                gv2: R.Tensor((5, 6), dtype="int32") = R.cumsum(
                    y, axis=1, dtype="int32", exclusive=True
                )
                gv = (gv1, gv2)
                R.output(gv)
            return gv

    verify(Cumsum, Expected)


def test_split_v_dynamic():
    """SPLIT_V with runtime split sizes imports shape-aware Relax IR."""

    class TfSplitVDynamic(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(10,), dtype=tf.float32),
                tf.TensorSpec(shape=(3,), dtype=tf.int32),
            ]
        )
        def func(self, x, size_splits):
            return tf.split(x, size_splits, axis=0)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((10,), dtype="float32"),
            size_splits: R.Tensor((3,), dtype="int32"),
        ) -> R.Tuple(
            R.Tensor(dtype="float32", ndim=1),
            R.Tensor(dtype="float32", ndim=1),
            R.Tensor(dtype="float32", ndim=1),
        ):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((3,), dtype="int64") = R.cumsum(
                    size_splits, axis=0, dtype="int64", exclusive=False
                )
                lv1: R.Tensor((4,), dtype="int64") = R.concat((R.const([0], "int64"), lv), axis=0)
                lv2: R.Tensor((1,), dtype="int64") = R.strided_slice(
                    lv1,
                    (R.prim_value(0),),
                    (R.prim_value(0),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )
                lv3: R.Tensor((1,), dtype="int64") = R.scatter_elements(
                    R.const([0], "int64"),
                    R.const([0], "int64"),
                    lv2,
                    axis=0,
                    reduction="update",
                )
                lv4: R.Shape([10]) = R.shape_of(x)
                lv5: R.Tensor((1,), dtype="int64") = R.shape_to_tensor(lv4)
                lv6: R.Tensor((1,), dtype="int64") = R.strided_slice(
                    lv1,
                    (R.prim_value(0),),
                    (R.prim_value(1),),
                    (R.prim_value(2),),
                    assume_inbound=False,
                )
                lv7: R.Tensor((1,), dtype="int64") = R.scatter_elements(
                    lv5, R.const([0], "int64"), lv6, axis=0, reduction="update"
                )
                lv8: R.Tensor(dtype="float32", ndim=1) = R.dynamic_strided_slice(
                    x, lv3, lv7, R.const([1], "int64")
                )
                lv9: R.Tensor((1,), dtype="int64") = R.strided_slice(
                    lv1,
                    (R.prim_value(0),),
                    (R.prim_value(1),),
                    (R.prim_value(2),),
                    assume_inbound=False,
                )
                lv10: R.Tensor((1,), dtype="int64") = R.scatter_elements(
                    R.const([0], "int64"),
                    R.const([0], "int64"),
                    lv9,
                    axis=0,
                    reduction="update",
                )
                lv11: R.Tensor((1,), dtype="int64") = R.strided_slice(
                    lv1,
                    (R.prim_value(0),),
                    (R.prim_value(2),),
                    (R.prim_value(3),),
                    assume_inbound=False,
                )
                lv12: R.Tensor((1,), dtype="int64") = R.scatter_elements(
                    lv5, R.const([0], "int64"), lv11, axis=0, reduction="update"
                )
                lv13: R.Tensor(dtype="float32", ndim=1) = R.dynamic_strided_slice(
                    x, lv10, lv12, R.const([1], "int64")
                )
                lv14: R.Tensor((1,), dtype="int64") = R.strided_slice(
                    lv1,
                    (R.prim_value(0),),
                    (R.prim_value(2),),
                    (R.prim_value(3),),
                    assume_inbound=False,
                )
                lv15: R.Tensor((1,), dtype="int64") = R.scatter_elements(
                    R.const([0], "int64"),
                    R.const([0], "int64"),
                    lv14,
                    axis=0,
                    reduction="update",
                )
                lv16: R.Tensor((1,), dtype="int64") = R.strided_slice(
                    lv1,
                    (R.prim_value(0),),
                    (R.prim_value(3),),
                    (R.prim_value(4),),
                    assume_inbound=False,
                )
                lv17: R.Tensor((1,), dtype="int64") = R.scatter_elements(
                    lv5, R.const([0], "int64"), lv16, axis=0, reduction="update"
                )
                lv18: R.Tensor(dtype="float32", ndim=1) = R.dynamic_strided_slice(
                    x, lv15, lv17, R.const([1], "int64")
                )
                gv: R.Tuple(
                    R.Tensor(dtype="float32", ndim=1),
                    R.Tensor(dtype="float32", ndim=1),
                    R.Tensor(dtype="float32", ndim=1),
                ) = (lv8, lv13, lv18)
                R.output(gv)
            return gv

    verify(TfSplitVDynamic, Expected)


def test_split_v_static():
    """SPLIT_V with static unequal size_splits lowers to Relax split."""

    class SplitVUnequal(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(2, 10, 4), dtype=tf.float32)])
        def func(self, x):
            return tf.split(x, [2, 3, 5], axis=1)

    @I.ir_module
    class ExpectedUnequal:
        @R.function
        def main(x: R.Tensor((2, 10, 4), dtype="float32")) -> R.Tuple(
            R.Tensor((2, 2, 4), dtype="float32"),
            R.Tensor((2, 3, 4), dtype="float32"),
            R.Tensor((2, 5, 4), dtype="float32"),
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((2, 2, 4), dtype="float32"),
                    R.Tensor((2, 3, 4), dtype="float32"),
                    R.Tensor((2, 5, 4), dtype="float32"),
                ) = R.split(x, indices_or_sections=[2, 5], axis=1)
                lv1: R.Tensor((2, 2, 4), dtype="float32") = lv[0]
                lv2: R.Tensor((2, 3, 4), dtype="float32") = lv[1]
                lv3: R.Tensor((2, 5, 4), dtype="float32") = lv[2]
                gv: R.Tuple(
                    R.Tensor((2, 2, 4), dtype="float32"),
                    R.Tensor((2, 3, 4), dtype="float32"),
                    R.Tensor((2, 5, 4), dtype="float32"),
                ) = lv1, lv2, lv3
                R.output(gv)
            return gv

    verify(SplitVUnequal, ExpectedUnequal)


def test_pack():
    class Pack(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(2, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(2, 3), dtype=tf.float32),
            ]
        )
        def func(self, x, y):
            return tf.raw_ops.Pack(values=[x, y], axis=0)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="float32"),
            y: R.Tensor((2, 3), dtype="float32"),
        ) -> R.Tensor((2, 2, 3), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((1, 2, 3), dtype="float32") = R.expand_dims(x, axis=[0])
                lv1: R.Tensor((1, 2, 3), dtype="float32") = R.expand_dims(y, axis=[0])
                gv: R.Tensor((2, 2, 3), dtype="float32") = R.concat((lv, lv1), axis=0)
                R.output(gv)
            return gv

    verify(Pack, Expected)


def test_cast():
    class Cast(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            return tf.cast(x, tf.int32)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 30), dtype="float32")) -> R.Tensor((1, 30), dtype="int32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 30), dtype="int32") = R.astype(x, dtype="int32")
                R.output(gv)
            return gv

    verify(Cast, Expected)


def test_bitcast_float32_to_int32():
    """BITCAST same-width: float32 -> int32, shape preserved."""

    class BitcastF32ToI32(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            return tf.bitcast(x, tf.int32)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 30), dtype="float32")) -> R.Tensor((1, 30), dtype="int32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 30), dtype="int32") = R.memory.view(
                    x, R.shape([1, 30]), R.dtype("int32")
                )
                R.output(gv)
            return gv

    verify(BitcastF32ToI32, Expected)


def test_bitcast_uint8_to_int8():
    """BITCAST same-width 8-bit: uint8 -> int8."""

    class BitcastU8ToI8(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(4,), dtype=tf.uint8)])
        def func(self, x):
            return tf.bitcast(x, tf.int8)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((4,), dtype="uint8")) -> R.Tensor((4,), dtype="int8"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((4,), dtype="int8") = R.memory.view(x, R.shape([4]), R.dtype("int8"))
                R.output(gv)
            return gv

    verify(BitcastU8ToI8, Expected)


def test_bitcast_int32_to_int16_widens_shape():
    """BITCAST width-changing (smaller): int32[3] -> int16[3, 2]."""

    class BitcastI32ToI16(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(3,), dtype=tf.int32)])
        def func(self, x):
            return tf.bitcast(x, tf.int16)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((3,), dtype="int32")) -> R.Tensor((3, 2), dtype="int16"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((3, 2), dtype="int16") = R.memory.view(
                    x, R.shape([3, 2]), R.dtype("int16")
                )
                R.output(gv)
            return gv

    verify(BitcastI32ToI16, Expected)


def test_bitcast_int16_to_int32_collapses_shape():
    """BITCAST width-changing (larger): int16[5, 2] -> int32[5]."""

    class BitcastI16ToI32(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(5, 2), dtype=tf.int16)])
        def func(self, x):
            return tf.bitcast(x, tf.int32)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((5, 2), dtype="int16")) -> R.Tensor((5,), dtype="int32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((5,), dtype="int32") = R.memory.view(x, R.shape([5]), R.dtype("int32"))
                R.output(gv)
            return gv

    verify(BitcastI16ToI32, Expected)


def test_bitwise_xor():
    """BITWISE_XOR lowers to relax.op.bitwise_xor."""

    class BitwiseXor(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(2, 3), dtype=tf.int32),
                tf.TensorSpec(shape=(2, 3), dtype=tf.int32),
            ]
        )
        def func(self, x, y):
            return tf.bitwise.bitwise_xor(x, y)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="int32"),
            y: R.Tensor((2, 3), dtype="int32"),
        ) -> R.Tensor((2, 3), dtype="int32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                gv: R.Tensor((2, 3), dtype="int32") = R.bitwise_xor(x, y)
                R.output(gv)
            return gv

    verify(BitwiseXor, Expected)


def test_right_shift():
    """RIGHT_SHIFT lowers to relax.op.right_shift."""

    class RightShift(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(2, 3), dtype=tf.int32),
                tf.TensorSpec(shape=(2, 3), dtype=tf.int32),
            ]
        )
        def func(self, x, y):
            return tf.bitwise.right_shift(x, y)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="int32"),
            y: R.Tensor((2, 3), dtype="int32"),
        ) -> R.Tensor((2, 3), dtype="int32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                gv: R.Tensor((2, 3), dtype="int32") = R.right_shift(x, y)
                R.output(gv)
            return gv

    verify(RightShift, Expected)


def test_sign():
    """SIGN lowers to relax.op.sign."""

    class Sign(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(2, 3), dtype=tf.float32)])
        def func(self, x):
            return tf.sign(x)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((2, 3), dtype="float32") = R.sign(x)
                R.output(gv)
            return gv

    verify(Sign, Expected)


def test_unique():
    """UNIQUE returns values and inverse indices."""

    class Unique(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(6,), dtype=tf.int32)])
        def func(self, x):
            return tf.raw_ops.Unique(x=x, out_idx=tf.int64)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((6,), dtype="int32"),
        ) -> R.Tuple(R.Tensor(dtype="int32", ndim=1), R.Tensor(dtype="int64", ndim=1)):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tuple(R.Tensor(dtype="int32", ndim=1), R.Tensor(dtype="int64", ndim=1)) = (
                    R.unique(
                        x,
                        R.prim_value(False),
                        R.prim_value(False),
                        R.prim_value(True),
                        R.prim_value(False),
                    )
                )
                lv1: R.Tensor(dtype="int32", ndim=1) = lv[0]
                lv2: R.Tensor(dtype="int64", ndim=1) = lv[1]
                gv: R.Tuple(R.Tensor(dtype="int32", ndim=1), R.Tensor(dtype="int64", ndim=1)) = (
                    lv1,
                    lv2,
                )
                R.output(gv)
            return gv

    mod = _get_mod_from_cfunc(Unique().func.get_concrete_function())
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_reshape():
    class Reshape(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            return tf.reshape(x, (1, 2, 15))

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 30), dtype="float32")) -> R.Tensor((1, 2, 15), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 2, 15), dtype="float32") = R.reshape(x, R.shape([1, 2, 15]))
                R.output(gv)
            return gv

    verify(Reshape, Expected)


@pytest.mark.parametrize(
    "input_shape, out_type",
    [
        ((2, 3, 4), tf.int32),
        ((5,), tf.int64),
        ((1, 1, 1, 1), tf.int32),
        ((), tf.int32),
        ((0, 3), tf.int64),
    ],
)
def test_shape(input_shape, out_type):
    """SHAPE conversion for static-rank non-quantized tensors."""

    class Shape(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32)])
        def func(self, x):
            return tf.shape(x, out_type=out_type)

    out_dtype = "int32" if out_type == tf.int32 else "int64"

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor(input_shape, dtype="float32"),
        ) -> R.Tensor((len(input_shape),), dtype=out_dtype):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((len(input_shape),), dtype=out_dtype) = R.const(
                    list(input_shape), out_dtype
                )
                R.output(gv)
            return gv

    verify(Shape, Expected)


def test_shape_dynamic_dim():
    """SHAPE conversion with a dynamic input dimension."""

    class ShapeDynamic(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(None, 3), dtype=tf.float32)])
        def func(self, x):
            return tf.shape(x, out_type=tf.int32)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 3), dtype="float32")) -> R.Tensor((2,), dtype="int32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Shape([1, 3]) = R.shape_of(x)
                lv1: R.Tensor((2,), dtype="int64") = R.shape_to_tensor(lv)
                gv: R.Tensor((2,), dtype="int32") = R.astype(lv1, dtype="int32")
                R.output(gv)
            return gv

    verify(ShapeDynamic, Expected)


def _build_rank_model():
    """Build a minimal TFLite RANK model."""
    builder = flatbuffers.Builder(1024)
    builtin_op = _get_builtin_operator("RANK")
    op_code = _build_operator_code(builder, builtin_op)
    options = _build_empty_builtin_options(builder, "RankOptions")

    tensors = [
        _build_tensor(builder, 0, [2, 3, 4]),
        _build_tensor(builder, 1, [], tensor_type=_tfl_tensor_type.INT32),
    ]
    op = _build_operator(
        builder,
        0,
        [0],
        [1],
        builtin_options_type=_get_builtin_options_type("RankOptions"),
        builtin_options=options,
    )
    subgraph = _build_subgraph(builder, tensors=tensors, operators=[op], inputs=[0], outputs=[1])
    return _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=[op_code],
        buffers=[_build_buffer(builder), _build_buffer(builder)],
    )


def test_rank():
    """RANK emits a static rank constant."""
    mod = _load_model_from_buffer(_build_rank_model())

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4), dtype="float32")) -> R.Tensor((), dtype="int32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((), dtype="int32") = R.const(3, "int32")
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_bucketize():
    """BUCKETIZE lowers to relax.op.bucketize."""

    class Bucketize(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(2, 3), dtype=tf.float32)])
        def func(self, x):
            return tf.raw_ops.Bucketize(input=x, boundaries=[0.0, 1.0, 3.0])

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="int32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((2, 3), dtype="int32") = R.bucketize(
                    x, R.const([0.0, 1.0, 3.0], "float32"), out_int32=True, right=False
                )
                R.output(gv)
            return gv

    verify(Bucketize, Expected)


@pytest.mark.parametrize(
    "start, limit, delta, dtype",
    [
        (0, 8, 2, tf.int32),
        (1, 9, 2, tf.int64),
        (0.0, 1.0, 0.2, tf.float32),
        (8, 0, -2, tf.int32),
        (0, 0, 1, tf.int32),
        (0, 7, 2, tf.int32),
        (0.0, -1.0, -0.25, tf.float32),
    ],
)
def test_range(start, limit, delta, dtype):
    """RANGE conversion with non-quantized constant scalar bounds."""

    class Range(tf.Module):
        @tf.function(input_signature=[])
        def func(self):
            return tf.range(start, limit, delta, dtype=dtype)

    np_dtype = np.float32 if dtype == tf.float32 else np.int64 if dtype == tf.int64 else np.int32
    expected_range = np.arange(start, limit, delta, dtype=np_dtype)
    out_dtype = np.dtype(np_dtype).name

    @I.ir_module
    class Expected:
        @R.function
        def main() -> R.Tensor((len(expected_range),), dtype=out_dtype):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                gv: R.Tensor((len(expected_range),), dtype=out_dtype) = R.const(
                    expected_range, out_dtype
                )
                R.output(gv)
            return gv

    verify(Range, Expected)


@pytest.mark.parametrize(
    "start, limit, delta, dtype",
    [
        (2, 13, 3, tf.int32),
        (8, 0, -2, tf.int32),
        (0.0, 1.0, 0.25, tf.float32),
        (1.0, -1.0, -0.5, tf.float32),
    ],
)
def test_range_dynamic_scalar_inputs(start, limit, delta, dtype):
    """RANGE lowers dynamic (runtime) scalar bounds for both int and float dtypes."""

    class RangeDynamic(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(), dtype=dtype),
                tf.TensorSpec(shape=(), dtype=dtype),
                tf.TensorSpec(shape=(), dtype=dtype),
            ]
        )
        def func(self, start, limit, delta):
            return tf.range(start, limit, delta)

    cf = RangeDynamic().func.get_concrete_function()
    mod = _get_mod_from_cfunc(cf)

    np_dtype = np.int32 if dtype == tf.int32 else np.float32
    inputs = [
        np.array(start, np_dtype),
        np.array(limit, np_dtype),
        np.array(delta, np_dtype),
    ]

    ex = tvm.compile(mod, tvm_tirx.target.Target("llvm"))
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    vm.set_input("main", *inputs)
    vm.invoke_stateful("main")
    tvm_out = vm.get_outputs("main").numpy()

    expected = np.arange(start, limit, delta, dtype=np_dtype)
    np.testing.assert_allclose(tvm_out, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "input_shape, multiples, dtype",
    [
        ((2, 3), [2, 1], tf.float32),
        ((1, 4, 2), [3, 1, 2], tf.float32),
        ((2, 1, 3, 1), [1, 2, 1, 4], tf.float32),
        ((3,), [2], tf.float32),
        ((2, 3), [4, 2], tf.float32),
        ((2, 2), [1, 3], tf.int32),
    ],
)
def test_tile(input_shape, multiples, dtype):
    """TILE conversion for non-quantized input and repeat factors."""

    class Tile(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=input_shape, dtype=dtype)])
        def func(self, x):
            return tf.tile(x, multiples)

    if input_shape == (2, 3) and multiples == [2, 1]:

        @I.ir_module
        class ExpectedTile2x3Repeat2x1:
            @R.function
            def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((4, 3), dtype="float32"):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    gv: R.Tensor((4, 3), dtype="float32") = R.tile(x, repeats=[2, 1])
                    R.output(gv)
                return gv

        expected = ExpectedTile2x3Repeat2x1

    elif input_shape == (1, 4, 2):

        @I.ir_module
        class ExpectedTile1x4x2:
            @R.function
            def main(x: R.Tensor((1, 4, 2), dtype="float32")) -> R.Tensor(
                (3, 4, 4), dtype="float32"
            ):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    gv: R.Tensor((3, 4, 4), dtype="float32") = R.tile(x, repeats=[3, 1, 2])
                    R.output(gv)
                return gv

        expected = ExpectedTile1x4x2

    elif input_shape == (2, 1, 3, 1):

        @I.ir_module
        class ExpectedTile2x1x3x1:
            @R.function
            def main(x: R.Tensor((2, 1, 3, 1), dtype="float32")) -> R.Tensor(
                (2, 2, 3, 4), dtype="float32"
            ):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    gv: R.Tensor((2, 2, 3, 4), dtype="float32") = R.tile(x, repeats=[1, 2, 1, 4])
                    R.output(gv)
                return gv

        expected = ExpectedTile2x1x3x1

    elif input_shape == (3,):

        @I.ir_module
        class ExpectedTile3:
            @R.function
            def main(x: R.Tensor((3,), dtype="float32")) -> R.Tensor((6,), dtype="float32"):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    gv: R.Tensor((6,), dtype="float32") = R.tile(x, repeats=[2])
                    R.output(gv)
                return gv

        expected = ExpectedTile3

    elif input_shape == (2, 3) and multiples == [4, 2]:

        @I.ir_module
        class ExpectedTile2x3Repeat4x2:
            @R.function
            def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((8, 6), dtype="float32"):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    gv: R.Tensor((8, 6), dtype="float32") = R.tile(x, repeats=[4, 2])
                    R.output(gv)
                return gv

        expected = ExpectedTile2x3Repeat4x2

    else:

        @I.ir_module
        class ExpectedTileInt32:
            @R.function
            def main(x: R.Tensor((2, 2), dtype="int32")) -> R.Tensor((2, 6), dtype="int32"):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    gv: R.Tensor((2, 6), dtype="int32") = R.tile(x, repeats=[1, 3])
                    R.output(gv)
                return gv

        expected = ExpectedTileInt32

    verify(Tile, expected)


def test_concat_v2():
    class ConcatV2(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            a, b, c = tf.split(x, 3, axis=1)
            axis = tf.add(tf.constant(1, dtype="int32"), tf.constant(0, dtype="int32"))
            return tf.raw_ops.ConcatV2(values=[a, b, c], axis=axis)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 30), dtype="float32")) -> R.Tensor((1, 30), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((1, 10), dtype="float32"),
                    R.Tensor((1, 10), dtype="float32"),
                    R.Tensor((1, 10), dtype="float32"),
                ) = R.split(x, indices_or_sections=3, axis=1)
                lv1: R.Tensor((1, 10), dtype="float32") = lv[0]
                lv2: R.Tensor((1, 10), dtype="float32") = lv[1]
                lv3: R.Tensor((1, 10), dtype="float32") = lv[2]
                gv: R.Tensor((1, 30), dtype="float32") = R.concat((lv1, lv2, lv3), axis=1)
                R.output(gv)
            return gv

    verify(ConcatV2, Expected)


def test_multi_output():
    class MultiOutput(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(2, 2), dtype=tf.float32)])
        def func(self, x):
            y = 2 * x
            return x, y

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 2), dtype="float32"), R.Tensor((2, 2), dtype="float32")):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 2), dtype="float32") = R.multiply(x, R.const(2.0, "float32"))
                gv: R.Tuple(
                    R.Tensor((2, 2), dtype="float32"), R.Tensor((2, 2), dtype="float32")
                ) = (x, lv)
                R.output(gv)
            return gv

    verify(MultiOutput, Expected)


def test_elu():
    class TfInput(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            return tf.nn.elu(x)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 30), dtype="float32")) -> R.Tensor((1, 30), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, 30), dtype="float32") = R.exp(x)
                lv1: R.Tensor((1, 30), dtype="float32") = R.subtract(R.const(1.0, "float32"), lv)
                lv2: R.Tensor((1, 30), dtype="float32") = R.nn.relu(lv1)
                lv3: R.Tensor((1, 30), dtype="float32") = R.multiply(R.const(-1.0, "float32"), lv2)
                lv4: R.Tensor((1, 30), dtype="float32") = R.nn.relu(x)
                gv: R.Tensor((1, 30), dtype="float32") = R.add(lv3, lv4)
                R.output(gv)
            return gv

    verify(TfInput, Expected)


def test_gelu():
    class TfInput(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            return tf.nn.gelu(x)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 30), dtype="float32")) -> R.Tensor((1, 30), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, 30), dtype="float32") = R.multiply(
                    x, R.const(0.70710676908493042, "float32")
                )
                lv1: R.Tensor((1, 30), dtype="float32") = R.erf(lv)
                lv2: R.Tensor((1, 30), dtype="float32") = R.multiply(lv1, R.const(0.5, "float32"))
                lv3: R.Tensor((1, 30), dtype="float32") = R.add(R.const(0.5, "float32"), lv2)
                gv: R.Tensor((1, 30), dtype="float32") = R.multiply(x, lv3)
                R.output(gv)
            return gv

    verify(TfInput, Expected)


def test_fill_dynamic_dims():
    """FILL with runtime dims legalizes and compiles."""

    class TfFillDynamic(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(2,), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
            ]
        )
        def func(self, dims, value):
            return tf.fill(dims, value)

    fill_dim_0 = T.dynamic("fill_dim_0", "int64")
    fill_dim_1 = T.dynamic("fill_dim_1", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            dims: R.Tensor((2,), dtype="int32"), value: R.Tensor((), dtype="float32")
        ) -> R.Tensor(dtype="float32", ndim=2):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((2,), dtype="int32") = R.match_cast(
                    dims, R.Tensor((2,), dtype="int32")
                )
                lv1: R.Tensor((2,), dtype="int64") = R.astype(lv, dtype="int64")
                lv2: R.Shape(ndim=2) = R.tensor_to_shape(lv1)
                _: R.Shape([fill_dim_0, fill_dim_1]) = R.match_cast(
                    lv2, R.Shape([fill_dim_0, fill_dim_1])
                )
                gv: R.Tensor((fill_dim_0, fill_dim_1), dtype="float32") = R.full(
                    R.shape([fill_dim_0, fill_dim_1]), value
                )
                R.output(gv)
            return gv

    verify(TfFillDynamic, Expected)


def test_random_uniform_dynamic_shape():
    """RANDOM_UNIFORM imports dynamic shape and validates random output metadata."""

    class TfRandomUniform(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(2,), dtype=tf.int32)])
        def func(self, shape):
            return tf.raw_ops.RandomUniform(shape=shape, dtype=tf.float32, seed=7, seed2=11)

    random_uniform_dim_0 = T.dynamic("random_uniform_dim_0", "int64")
    random_uniform_dim_1 = T.dynamic("random_uniform_dim_1", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(shape: R.Tensor((2,), dtype="int32")) -> R.Tensor(dtype="float32", ndim=2):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2,), dtype="int32") = R.match_cast(
                    shape, R.Tensor((2,), dtype="int32")
                )
                lv1: R.Tensor((2,), dtype="int64") = R.astype(lv, dtype="int64")
                lv2: R.Shape(ndim=2) = R.tensor_to_shape(lv1)
                _: R.Shape([random_uniform_dim_0, random_uniform_dim_1]) = R.match_cast(
                    lv2, R.Shape([random_uniform_dim_0, random_uniform_dim_1])
                )
                gv = R.call_dps_packed(
                    _ExternFunc("tvm.contrib.random.uniform"),
                    (
                        R.prim_value(7),
                        R.prim_value(11),
                        R.prim_value(T.float64(0.0)),
                        R.prim_value(T.float64(1.0)),
                    ),
                    ty_args=[
                        R.Tensor((random_uniform_dim_0, random_uniform_dim_1), dtype="float32")
                    ],
                )
                R.output(gv)
            return gv

    cf = TfRandomUniform().func.get_concrete_function()
    mod = _get_mod_from_cfunc(cf)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)

    _verify_random_with_inputs(cf, [np.array([2, 3], dtype="int32")])


def test_random_standard_normal_dynamic_shape():
    """RANDOM_STANDARD_NORMAL imports dynamic shape and validates random output metadata."""

    class TfRandomStandardNormal(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(2,), dtype=tf.int32)])
        def func(self, shape):
            return tf.raw_ops.RandomStandardNormal(shape=shape, dtype=tf.float32, seed=3, seed2=5)

    random_standard_normal_dim_0 = T.dynamic("random_standard_normal_dim_0", "int64")
    random_standard_normal_dim_1 = T.dynamic("random_standard_normal_dim_1", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(shape: R.Tensor((2,), dtype="int32")) -> R.Tensor(dtype="float32", ndim=2):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2,), dtype="int32") = R.match_cast(
                    shape, R.Tensor((2,), dtype="int32")
                )
                lv1: R.Tensor((2,), dtype="int64") = R.astype(lv, dtype="int64")
                lv2: R.Shape(ndim=2) = R.tensor_to_shape(lv1)
                _: R.Shape([random_standard_normal_dim_0, random_standard_normal_dim_1]) = (
                    R.match_cast(
                        lv2, R.Shape([random_standard_normal_dim_0, random_standard_normal_dim_1])
                    )
                )
                gv = R.call_dps_packed(
                    _ExternFunc("tvm.contrib.random.normal"),
                    (
                        R.prim_value(3),
                        R.prim_value(5),
                        R.prim_value(T.float64(0.0)),
                        R.prim_value(T.float64(1.0)),
                    ),
                    ty_args=[
                        R.Tensor(
                            (random_standard_normal_dim_0, random_standard_normal_dim_1),
                            dtype="float32",
                        )
                    ],
                )
                R.output(gv)
            return gv

    cf = TfRandomStandardNormal().func.get_concrete_function()
    mod = _get_mod_from_cfunc(cf)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)

    _verify_random_with_inputs(cf, [np.array([2, 4], dtype="int32")])


def test_multinomial_dynamic_num_samples():
    """MULTINOMIAL lowers through seeded uniform sampling with dynamic num_samples."""

    class TfMultinomial(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(2, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ]
        )
        def func(self, logits, num_samples):
            return tf.raw_ops.Multinomial(
                logits=logits,
                num_samples=num_samples,
                output_dtype=tf.int64,
                seed=13,
                seed2=17,
            )

    multinomial_num_samples = T.dynamic("multinomial_num_samples", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            logits: R.Tensor((2, 3), dtype="float32"),
            num_samples: R.Tensor((), dtype="int32"),
        ) -> R.Tensor(dtype="int64", ndim=2):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((), dtype="int32") = R.match_cast(
                    num_samples, R.Tensor((), dtype="int32")
                )
                lv1: R.Tensor((), dtype="int64") = R.astype(lv, dtype="int64")
                lv2: R.Tensor((1,), dtype="int64") = R.reshape(lv1, R.shape([1]))
                lv3: R.Shape(ndim=1) = R.tensor_to_shape(lv2)
                _: R.Shape([multinomial_num_samples]) = R.match_cast(
                    lv3, R.Shape([multinomial_num_samples])
                )
                lv5: R.Tensor((2, 3), dtype="float32") = R.nn.softmax(logits, axis=-1)
                lv6 = R.call_dps_packed(
                    _ExternFunc("tvm.contrib.random.uniform"),
                    (
                        R.prim_value(13),
                        R.prim_value(17),
                        R.prim_value(T.float64(0.0)),
                        R.prim_value(T.float64(1.0)),
                    ),
                    ty_args=[R.Tensor((2 * multinomial_num_samples, 1), dtype="float32")],
                )
                lv7: R.Tensor((2,), dtype="int64") = R.arange(
                    R.prim_value(0), R.prim_value(2), R.prim_value(1), dtype="int64"
                )
                lv8: R.Tensor((2, 1), dtype="int64") = R.expand_dims(lv7, axis=[1])
                lv9: R.Tensor((2, multinomial_num_samples), dtype="int64") = R.broadcast_to(
                    lv8, R.shape([2, multinomial_num_samples])
                )
                lv10: R.Tensor((2 * multinomial_num_samples, 1), dtype="int64") = R.reshape(
                    lv9, R.shape([2 * multinomial_num_samples, 1])
                )
                lv11: R.Tensor((2 * multinomial_num_samples, 1), dtype="int64") = (
                    R.multinomial_from_uniform(lv5, lv6, lv10, dtype="int64")
                )
                gv: R.Tensor((2, multinomial_num_samples), dtype="int64") = R.reshape(
                    lv11, R.shape([2, multinomial_num_samples])
                )
                R.output(gv)
            return gv

    cf = TfMultinomial().func.get_concrete_function()
    mod = _get_mod_from_cfunc(cf)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)

    _verify_random_with_inputs(
        cf,
        [
            np.array([[2.0, 1.0, 0.5], [0.1, 0.2, 3.0]], dtype="float32"),
            np.array(4, dtype="int32"),
        ],
    )


@pytest.mark.parametrize(
    "tf_op, relax_op",
    [
        (tf.add, R.add),
        (tf.subtract, R.subtract),
        (tf.multiply, R.multiply),
        (tf.divide, R.divide),
        (tf.math.floormod, R.floor_mod),
        (tf.math.floordiv, R.floor_divide),
        (tf.math.atan2, R.atan2),
    ],
)
def test_binary(tf_op, relax_op):
    class Binary(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(2, 2), dtype=tf.float32),
                tf.TensorSpec(shape=(2, 2), dtype=tf.float32),
            ]
        )
        def func(self, x, y):
            return tf_op(x, y)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 2), dtype="float32"), y: R.Tensor((2, 2), dtype="float32")
        ) -> R.Tensor((2, 2), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="float32") = relax_op(x, y)
                R.output(gv)
            return gv

    verify(Binary, Expected)


def test_static_broadcast_lowered_to_multiply():
    """Static TensorFlow broadcasts become TFLite MUL with a constant operand."""

    class RankExpansion(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(2, 2), dtype=tf.float32)])
        def func(self, x):
            return tf.broadcast_to(x, [3, 2, 2])

    @I.ir_module
    class ExpectedRankExpansion:
        @R.function
        def main(x: R.Tensor((2, 2), dtype="float32")) -> R.Tensor((3, 2, 2), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((3, 2, 2), dtype="float32") = R.multiply(
                    x, R.const(np.ones((3, 2, 2), dtype="float32"))
                )
                R.output(gv)
            return gv

    verify(RankExpansion, ExpectedRankExpansion)

    class ScalarInt(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.int32)])
        def func(self, x):
            return tf.broadcast_to(x, [4, 4])

    @I.ir_module
    class ExpectedScalarInt:
        @R.function
        def main(x: R.Tensor((), dtype="int32")) -> R.Tensor((4, 4), dtype="int32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((4, 4), dtype="int32") = R.multiply(
                    x, R.const(np.ones((4, 4), dtype="int32"))
                )
                R.output(gv)
            return gv

    verify(ScalarInt, ExpectedScalarInt)


def test_pow():
    class TfInput(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            return tf.math.pow(x, 4)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 30), dtype="float32")) -> R.Tensor((1, 30), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 30), dtype="float32") = R.power(x, R.const(4.0, "float32"))
                R.output(gv)
            return gv

    verify(TfInput, Expected)


def test_square():
    class TfInput(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            return tf.math.square(x)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 30), dtype="float32")) -> R.Tensor((1, 30), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 30), dtype="float32") = R.power(x, R.const(2.0, "float32"))
                R.output(gv)
            return gv

    verify(TfInput, Expected)


def test_broadcast_args():
    class TfInput(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(3,), dtype=tf.int32),
                tf.TensorSpec(shape=(3,), dtype=tf.int32),
            ]
        )
        def func(self, s0, s1):
            return tf.broadcast_dynamic_shape(s0, s1)

    @I.ir_module
    class Expected:
        @R.function
        def main(s0: R.Tensor((3,), dtype="int32"), s1: R.Tensor((3,), dtype="int32")) -> R.Tensor(
            (3,), dtype="int32"
        ):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((0,), dtype="int32") = R.full(
                    R.shape([0]), R.const(1, "int32"), dtype="int32"
                )
                lv1: R.Tensor((3,), dtype="int32") = R.concat((lv, s0), axis=0)
                lv2: R.Tensor((3,), dtype="bool") = R.equal(lv1, R.const(1, "int32"))
                lv3: R.Tensor((0,), dtype="int32") = R.full(
                    R.shape([0]), R.const(1, "int32"), dtype="int32"
                )
                lv4: R.Tensor((3,), dtype="int32") = R.concat((lv3, s1), axis=0)
                lv5: R.Tensor((3,), dtype="bool") = R.equal(lv4, R.const(1, "int32"))
                lv6: R.Tensor((3,), dtype="int32") = R.maximum(lv1, lv4)
                lv7: R.Tensor((3,), dtype="int32") = R.where(lv5, lv1, lv6)
                gv: R.Tensor((3,), dtype="int32") = R.where(lv2, lv4, lv7)
                R.output(gv)
            return gv

    verify(TfInput, Expected)


def test_broadcast_args_diff_length():
    """BROADCAST_ARGS with shape inputs of different lengths."""

    class TfInput(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(1,), dtype=tf.int32),
                tf.TensorSpec(shape=(3,), dtype=tf.int32),
            ]
        )
        def func(self, s0, s1):
            return tf.broadcast_dynamic_shape(s0, s1)

    @I.ir_module
    class Expected:
        @R.function
        def main(s0: R.Tensor((1,), dtype="int32"), s1: R.Tensor((3,), dtype="int32")) -> R.Tensor(
            (3,), dtype="int32"
        ):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((2,), dtype="int32") = R.full(
                    R.shape([2]), R.const(1, "int32"), dtype="int32"
                )
                lv1: R.Tensor((3,), dtype="int32") = R.concat((lv, s0), axis=0)
                lv2: R.Tensor((3,), dtype="bool") = R.equal(lv1, R.const(1, "int32"))
                lv3: R.Tensor((0,), dtype="int32") = R.full(
                    R.shape([0]), R.const(1, "int32"), dtype="int32"
                )
                lv4: R.Tensor((3,), dtype="int32") = R.concat((lv3, s1), axis=0)
                lv5: R.Tensor((3,), dtype="bool") = R.equal(lv4, R.const(1, "int32"))
                lv6: R.Tensor((3,), dtype="int32") = R.maximum(lv1, lv4)
                lv7: R.Tensor((3,), dtype="int32") = R.where(lv5, lv1, lv6)
                gv: R.Tensor((3,), dtype="int32") = R.where(lv2, lv4, lv7)
                R.output(gv)
            return gv

    verify(TfInput, Expected)


@pytest.mark.parametrize(
    "tf_op, relax_op",
    [
        (tf.nn.relu, R.nn.relu),
        (tf.nn.relu6, R.nn.relu6),
        (tf.math.floor, R.floor),
        (tf.math.ceil, R.ceil),
        (tf.math.tanh, R.tanh),
        (tf.math.sigmoid, R.sigmoid),
        (tf.math.abs, R.abs),
        (tf.math.cos, R.cos),
        (tf.math.sin, R.sin),
        (tf.math.exp, R.exp),
        (tf.math.log, R.log),
        (tf.math.negative, R.negative),
        (tf.round, R.round),
        (tf.math.rsqrt, R.rsqrt),
        (tf.nn.softmax, R.nn.softmax),
        (tf.math.sqrt, R.sqrt),
        (tf.nn.log_softmax, R.nn.log_softmax),
    ],
)
def test_element_wise(tf_op, relax_op):
    class TfInput(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            return tf_op(x)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 30), dtype="float32")) -> R.Tensor((1, 30), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 30), dtype="float32") = relax_op(x)
                R.output(gv)
            return gv

    verify(TfInput, Expected)


@pytest.mark.parametrize(
    "tf_op, relax_op",
    [
        (tf.math.less, R.less),
        (tf.math.less_equal, R.less_equal),
        (tf.math.greater, R.greater),
        (tf.math.greater_equal, R.greater_equal),
        (tf.math.equal, R.equal),
        (tf.math.not_equal, R.not_equal),
    ],
)
def test_split_compare(tf_op, relax_op):
    class Compare(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            a, b = tf.split(x, 2, axis=1)
            return tf_op(a, b, name=None)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 30), dtype="float32")) -> R.Tensor((1, 15), dtype="bool"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((1, 15), dtype="float32"),
                    R.Tensor((1, 15), dtype="float32"),
                ) = R.split(x, indices_or_sections=2, axis=1)
                lv1: R.Tensor((1, 15), dtype="float32") = lv[0]
                lv2: R.Tensor((1, 15), dtype="float32") = lv[1]
                gv: R.Tensor((1, 15), dtype="bool") = relax_op(lv1, lv2)
                R.output(gv)
            return gv

    verify(Compare, Expected)


@pytest.mark.parametrize(
    "tf_op, relax_op",
    [
        (tf.math.logical_not, R.logical_not),
    ],
)
def test_logical_unary(tf_op, relax_op):
    class Logical(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(2, 2), dtype=tf.bool),
            ]
        )
        def func(self, x):
            return tf_op(x)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 2), dtype="bool"),
        ) -> R.Tensor((2, 2), dtype="bool"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="bool") = relax_op(x)
                R.output(gv)
            return gv

    verify(Logical, Expected)


@pytest.mark.parametrize(
    "tf_op, relax_op",
    [
        (tf.math.logical_or, R.logical_or),
        (tf.math.logical_and, R.logical_and),
    ],
)
def test_logical(tf_op, relax_op):
    class Logical(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(2, 2), dtype=tf.bool),
                tf.TensorSpec(shape=(2, 2), dtype=tf.bool),
            ]
        )
        def func(self, x, y):
            return tf_op(x, y)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 2), dtype="bool"), y: R.Tensor((2, 2), dtype="bool")) -> R.Tensor(
            (2, 2), dtype="bool"
        ):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="bool") = relax_op(x, y)
                R.output(gv)
            return gv

    verify(Logical, Expected)


@pytest.mark.parametrize(
    "tf_op, relax_op",
    [
        (tf.add, R.add),
        (tf.subtract, R.subtract),
        (tf.multiply, R.multiply),
        (tf.divide, R.divide),
        (tf.math.floormod, R.floor_mod),
        (tf.math.maximum, R.maximum),
        (tf.math.minimum, R.minimum),
    ],
)
def test_split_binary(tf_op, relax_op):
    class Binary(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            a, b = tf.split(x, 2, axis=1)
            return tf_op(a, b, name=None)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 30), dtype="float32")) -> R.Tensor((1, 15), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((1, 15), dtype="float32"),
                    R.Tensor((1, 15), dtype="float32"),
                ) = R.split(x, indices_or_sections=2, axis=1)
                lv1: R.Tensor((1, 15), dtype="float32") = lv[0]
                lv2: R.Tensor((1, 15), dtype="float32") = lv[1]
                gv: R.Tensor((1, 15), dtype="float32") = relax_op(lv1, lv2)
                R.output(gv)
            return gv

    verify(Binary, Expected)


def test_squared_difference():
    class SquaredDifference(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(2, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(2, 3), dtype=tf.float32),
            ]
        )
        def func(self, x, y):
            return tf.math.squared_difference(x, y)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")
        ) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="float32") = R.subtract(x, y)
                gv: R.Tensor((2, 3), dtype="float32") = R.power(lv, R.const(2.0, "float32"))
                R.output(gv)
            return gv

    verify(SquaredDifference, Expected)


@pytest.mark.parametrize(
    "tf_op, relax_op, axis, out_shape",
    [
        (tf.math.argmax, R.argmax, 0, (30,)),
        (tf.math.argmin, R.argmin, 1, (5,)),
    ],
)
def test_reduce(tf_op, relax_op, axis, out_shape):
    class TfInput(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(5, 30), dtype=tf.float32)])
        def func(self, x):
            return tf_op(x, axis=axis)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((5, 30), dtype="float32")) -> R.Tensor(out_shape, dtype="int64"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor(out_shape, dtype="int64") = relax_op(x, axis=axis, keepdims=False)
                R.output(gv)
            return gv

    verify(TfInput, Expected)


def test_fully_connected():
    class FullyConnected(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 8), dtype=tf.float32)])
        def func(self, x):
            weight = tf.constant(np.arange(24, dtype=np.float32).reshape((3, 8)))
            bias = tf.constant(np.array([0.5, 1.0, -1.0], dtype=np.float32))
            out = tf.matmul(x, weight, transpose_b=True)
            return tf.nn.bias_add(out, bias)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 8), dtype="float32")) -> R.Tensor((1, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((8, 3), dtype="float32") = R.permute_dims(
                    R.const(np.arange(24, dtype=np.float32).reshape((3, 8))), axes=[1, 0]
                )
                lv1: R.Tensor((1, 3), dtype="float32") = R.matmul(x, lv)
                gv: R.Tensor((1, 3), dtype="float32") = R.add(
                    lv1, R.const(np.array([0.5, 1.0, -1.0], dtype=np.float32))
                )
                R.output(gv)
            return gv

    verify(FullyConnected, Expected)


def test_depthwise_conv2d():
    class DepthwiseConv2D(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(1, 8, 8, 2), dtype=tf.float32),
                tf.TensorSpec(shape=(3, 3, 2, 1), dtype=tf.float32),
            ]
        )
        def func(self, data, kernel):
            return tf.nn.depthwise_conv2d(
                input=data,
                filter=kernel,
                strides=[1, 1, 1, 1],
                padding="SAME",
            )

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data: R.Tensor((1, 8, 8, 2), dtype="float32"),
            kernel: R.Tensor((3, 3, 2, 1), dtype="float32"),
        ) -> R.Tensor((1, 8, 8, 2), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((1, 3, 3, 2), dtype="float32") = R.reshape(
                    kernel, R.shape([1, 3, 3, 2])
                )
                lv1: R.Tensor((3, 3, 2, 1), dtype="float32") = R.reshape(lv, R.shape([3, 3, 2, 1]))
                lv2: R.Tensor((1, 8, 8, 2), dtype="float32") = R.nn.conv2d(
                    data,
                    lv1,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=2,
                    data_layout="NHWC",
                    kernel_layout="HWOI",
                    out_layout="NHWC",
                )
                gv: R.Tensor((1, 8, 8, 2), dtype="float32") = R.add(
                    lv2, R.const(np.zeros((2,), dtype="float32"))
                )
                R.output(gv)
            return gv

    verify(DepthwiseConv2D, Expected)


def test_transpose_conv():
    class TransposeConv(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(1, 8, 8, 2), dtype=tf.float32),
                tf.TensorSpec(shape=(3, 3, 3, 2), dtype=tf.float32),
            ]
        )
        def func(self, data, kernel):
            output_shape = tf.constant([1, 8, 8, 3], dtype=tf.int32)
            return tf.nn.conv2d_transpose(
                input=data,
                filters=kernel,
                output_shape=output_shape,
                strides=[1, 1, 1, 1],
                padding="SAME",
            )

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data: R.Tensor((1, 8, 8, 2), dtype="float32"),
            kernel: R.Tensor((3, 3, 3, 2), dtype="float32"),
        ) -> R.Tensor((1, 8, 8, 3), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((3, 3, 3, 2), dtype="float32") = R.permute_dims(
                    kernel, axes=[2, 0, 1, 3]
                )
                lv1: R.Tensor((2, 3, 3, 3), dtype="float32") = R.permute_dims(lv, axes=[3, 0, 1, 2])
                gv: R.Tensor((1, 8, 8, 3), dtype="float32") = R.nn.conv2d_transpose(
                    data,
                    lv1,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    output_padding=[0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="IOHW",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                R.output(gv)
            return gv

    verify(TransposeConv, Expected)


def test_l2_pool2d():
    class L2Pool2D(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 8, 8, 2), dtype=tf.float32)])
        def func(self, data):
            squared = tf.math.square(data)
            pooled = tf.nn.avg_pool2d(squared, ksize=[2, 2], strides=[1, 1], padding="SAME")
            return tf.math.sqrt(pooled)

    @I.ir_module
    class Expected:
        @R.function
        def main(data: R.Tensor((1, 8, 8, 2), dtype="float32")) -> R.Tensor(
            (1, 8, 8, 2), dtype="float32"
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                squared = R.power(data, R.const(2.0, "float32"))
                pooled = R.nn.avg_pool2d(
                    squared,
                    pool_size=[2, 2],
                    strides=[1, 1],
                    padding=[0, 0, 1, 1],
                    layout="NHWC",
                )
                gv = R.sqrt(pooled)
                R.output(gv)
            return gv

    verify(L2Pool2D, Expected)


def test_l2_normalization():
    class L2Normalization(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(2, 4), dtype=tf.float32)])
        def func(self, x):
            return tf.nn.l2_normalize(x, axis=-1)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 4), dtype="float32")) -> R.Tensor((2, 4), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 4), dtype="float32") = R.square(x)
                lv1: R.Tensor((2, 1), dtype="float32") = R.sum(lv, axis=[1], keepdims=True)
                lv2: R.Tensor((2, 1), dtype="float32") = R.add(
                    lv1, R.const(9.999999960041972e-13, "float32")
                )
                lv3: R.Tensor((2, 1), dtype="float32") = R.sqrt(lv2)
                gv: R.Tensor((2, 4), dtype="float32") = R.divide(x, lv3)
                R.output(gv)
            return gv

    verify(L2Normalization, Expected)


def test_local_response_normalization():
    class LocalResponseNormalization(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 8, 8, 4), dtype=tf.float32)])
        def func(self, x):
            return tf.nn.local_response_normalization(
                x,
                depth_radius=2,
                bias=1.0,
                alpha=1e-4,
                beta=0.75,
            )

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 8, 8, 4), dtype="float32")) -> R.Tensor(
            (1, 8, 8, 4), dtype="float32"
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, 8, 8, 4), dtype="float32") = R.square(x)
                lv1: R.Tensor((64, 4, 1, 1), dtype="float32") = R.reshape(
                    lv, R.shape([64, 4, 1, 1])
                )
                lv2: R.Tensor((64, 4, 1, 1), dtype="float32") = R.nn.avg_pool2d(
                    lv1,
                    pool_size=[5, 1],
                    strides=[1, 1],
                    dilation=[1, 1],
                    padding=[2, 0, 2, 0],
                    ceil_mode=False,
                    count_include_pad=True,
                    layout="NHWC",
                    out_layout="NHWC",
                )
                lv3: R.Tensor((1, 8, 8, 4), dtype="float32") = R.reshape(lv2, R.shape([1, 8, 8, 4]))
                lv4: R.Tensor((1, 8, 8, 4), dtype="float32") = R.multiply(
                    R.const(0.00049999996554106474, "float32"), lv3
                )
                lv5: R.Tensor((1, 8, 8, 4), dtype="float32") = R.add(R.const(1.0, "float32"), lv4)
                lv6: R.Tensor((1, 8, 8, 4), dtype="float32") = R.power(
                    lv5, R.const(0.75, "float32")
                )
                gv: R.Tensor((1, 8, 8, 4), dtype="float32") = R.divide(x, lv6)
                R.output(gv)
            return gv

    verify(LocalResponseNormalization, Expected)


def test_slice():
    class Slice(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(3, 4), dtype=tf.float32)])
        def func(self, x):
            return tf.slice(x, begin=[1, 1], size=[2, 2])

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((3, 4), dtype="float32")) -> R.Tensor((2, 2), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="float32") = R.strided_slice(
                    x, axes=[0, 1], begin=[1, 1], end=[3, 3]
                )
                R.output(gv)
            return gv

    verify(Slice, Expected)


def test_strided_slice_stride():
    class StridedSliceStride(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(4, 6), dtype=tf.float32)])
        def func(self, x):
            return x[0:2, 1:5:2]

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((4, 6), dtype="float32")) -> R.Tensor((2, 2), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 2), dtype="float32") = R.strided_slice(
                    x,
                    axes=[0, 1],
                    begin=[0, 1],
                    end=[2, 5],
                    strides=[1, 2],
                    assume_inbound=False,
                )
                gv: R.Tensor((2, 2), dtype="float32") = R.reshape(lv, R.shape([2, 2]))
                R.output(gv)
            return gv

    verify(StridedSliceStride, Expected)


def test_strided_slice_negative_stride():
    class StridedSliceNegativeStride(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(4,), dtype=tf.float32)])
        def func(self, x):
            return x[::-1]

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((4,), dtype="float32")) -> R.Tensor((4,), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((4,), dtype="float32") = R.strided_slice(
                    x, axes=[0], begin=[4], end=[-5], strides=[-1], assume_inbound=False
                )
                gv: R.Tensor((4,), dtype="float32") = R.reshape(lv, R.shape([4]))
                R.output(gv)
            return gv

    verify(StridedSliceNegativeStride, Expected)


def test_reverse_v2():
    class ReverseV2(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(2, 3), dtype=tf.float32)])
        def func(self, x):
            return tf.reverse(x, axis=[1])

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((2, 3), dtype="float32") = R.flip(x, axis=1)
                R.output(gv)
            return gv

    verify(ReverseV2, Expected)


def test_reverse_sequence():
    mod = _load_model_from_buffer(_build_tflite_reverse_sequence_model())

    @I.ir_module
    class Expected:
        @R.function
        def main(
            tvmgen_tensor_0: R.Tensor((2, 4, 3), dtype="float32"),
            tvmgen_tensor_1: R.Tensor((2,), dtype="int32"),
        ) -> R.Tensor((2, 4, 3), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                gv: R.Tensor((2, 4, 3), dtype="float32") = R.reverse_sequence(
                    tvmgen_tensor_0, tvmgen_tensor_1, seq_axis=1, batch_axis=0
                )
                R.output(gv)
            return gv

    data = np.arange(24, dtype="float32").reshape((2, 4, 3))
    seq_lengths = np.array([1, 3], dtype="int32")
    expected = data.copy()
    expected[1, :3, :] = expected[1, :3, :][::-1]

    ex = tvm.compile(mod, tvm_tirx.target.Target("c"))
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    vm.set_input("main", data, seq_lengths)
    vm.invoke_stateful("main")
    output = vm.get_outputs("main")
    np.testing.assert_allclose(output.numpy(), expected, rtol=1e-5, atol=1e-5)


def test_gather():
    class Gather(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(2, 3, 4), dtype=tf.float32),
                tf.TensorSpec(shape=(2,), dtype=tf.int64),
            ]
        )
        def func(self, x, indices):
            return tf.gather(x, indices, axis=1)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 4), dtype="float32"),
            indices: R.Tensor((2,), dtype="int64"),
        ) -> R.Tensor((2, 2, 4), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((2,), dtype="int32") = R.astype(indices, dtype="int32")
                gv: R.Tensor((2, 2, 4), dtype="float32") = R.take(x, lv, axis=1, mode="fast")
                R.output(gv)
            return gv

    verify(Gather, Expected)

    # TensorFlow lowers embedding lookup to TFLite GATHER.  Keep a case with
    # constant params and multidimensional int32 indices, since the case above
    # uses runtime params and one-dimensional int64 indices along axis 1.
    class GatherConstantParams(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(2, 3), dtype=tf.int32)])
        def func(self, indices):
            params = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=tf.float32)
            return tf.gather(params, indices, axis=0)

    @I.ir_module
    class ExpectedConstantParams:
        @R.function
        def main(indices: R.Tensor((2, 3), dtype="int32")) -> R.Tensor((2, 3, 2), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="int32") = R.astype(indices, dtype="int32")
                gv: R.Tensor((2, 3, 2), dtype="float32") = R.take(
                    R.const(np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)),
                    lv,
                    axis=0,
                    mode="fast",
                )
                R.output(gv)
            return gv

    verify(GatherConstantParams, ExpectedConstantParams)


def test_gather_nd():
    class GatherND(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(2, 3, 4), dtype=tf.float32),
                tf.TensorSpec(shape=(2, 2), dtype=tf.int32),
            ]
        )
        def func(self, x, indices):
            return tf.gather_nd(x, indices)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 4), dtype="float32"),
            indices: R.Tensor((2, 2), dtype="int32"),
        ) -> R.Tensor((2, 4), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((2, 2), dtype="int32") = R.permute_dims(indices, axes=[-1, 0])
                lv1: R.Tensor((2, 2), dtype="int64") = R.astype(lv, dtype="int64")
                gv: R.Tensor((2, 4), dtype="float32") = R.gather_nd(x, lv1, batch_dims=0)
                R.output(gv)
            return gv

    verify(GatherND, Expected)


def test_squeeze():
    mod = _load_model_from_buffer(_build_tflite_squeeze_model())

    @I.ir_module
    class Expected:
        @R.function
        def main(tvmgen_tensor_0: R.Tensor((1, 2, 1, 3), dtype="float32")) -> R.Tensor(
            (2, 3), dtype="float32"
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((2, 3), dtype="float32") = R.squeeze(tvmgen_tensor_0, axis=[0, 2])
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_unpack():
    mod = _load_model_from_buffer(_build_tflite_unpack_model())

    @I.ir_module
    class Expected:
        @R.function
        def main(tvmgen_tensor_0: R.Tensor((2, 3, 4), dtype="float32")) -> R.Tuple(
            R.Tensor((2, 4), dtype="float32"),
            R.Tensor((2, 4), dtype="float32"),
            R.Tensor((2, 4), dtype="float32"),
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((2, 1, 4), dtype="float32"),
                    R.Tensor((2, 1, 4), dtype="float32"),
                    R.Tensor((2, 1, 4), dtype="float32"),
                ) = R.split(tvmgen_tensor_0, indices_or_sections=3, axis=1)
                lv1: R.Tensor((2, 1, 4), dtype="float32") = lv[0]
                lv2: R.Tensor((2, 4), dtype="float32") = R.squeeze(lv1, axis=[1])
                lv3: R.Tensor((2, 1, 4), dtype="float32") = lv[1]
                lv4: R.Tensor((2, 4), dtype="float32") = R.squeeze(lv3, axis=[1])
                lv5: R.Tensor((2, 1, 4), dtype="float32") = lv[2]
                lv6: R.Tensor((2, 4), dtype="float32") = R.squeeze(lv5, axis=[1])
                gv = (lv2, lv4, lv6)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_zeros_like():
    mod = _load_model_from_buffer(_build_tflite_zeros_like_model())

    @I.ir_module
    class Expected:
        @R.function
        def main(tvmgen_tensor_0: R.Tensor((2, 3), dtype="float32")) -> R.Tensor(
            (2, 3), dtype="float32"
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((2, 3), dtype="float32") = R.zeros_like(tvmgen_tensor_0)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def _make_conv2d_module(data_shape, kernel_shape, data_format, strides, padding):
    class Conv2DModule(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=data_shape, dtype=tf.float32),
                tf.TensorSpec(shape=kernel_shape, dtype=tf.float32),
            ]
        )
        def func(self, data, kernel):
            return tf.nn.conv2d(
                input=data,
                filters=kernel,
                data_format=data_format,
                strides=strides,
                padding=padding,
            )

    return Conv2DModule


def test_conv2d_same():
    Conv2DModule = _make_conv2d_module(
        (1, 128, 128, 32), (3, 3, 32, 32), "NHWC", (1, 1, 1, 1), "SAME"
    )

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data: R.Tensor((1, 128, 128, 32), dtype="float32"),
            kernel: R.Tensor((3, 3, 32, 32), dtype="float32"),
        ) -> R.Tensor((1, 128, 128, 32), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((32, 3, 3, 32), dtype="float32") = R.permute_dims(
                    kernel, axes=[3, 0, 1, 2]
                )
                lv1: R.Tensor((3, 3, 32, 32), dtype="float32") = R.permute_dims(
                    lv, axes=[1, 2, 3, 0]
                )
                lv2: R.Tensor((1, 128, 128, 32), dtype="float32") = R.nn.conv2d(
                    data,
                    lv1,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="HWIO",
                    out_layout="NHWC",
                )
                gv: R.Tensor((1, 128, 128, 32), dtype="float32") = R.add(
                    lv2, R.const(np.zeros((32,), dtype="float32"))
                )
                R.output(gv)
            return gv

    verify(Conv2DModule, Expected)


def test_conv2d_valid():
    Conv2DModule = _make_conv2d_module(
        (1, 128, 128, 32), (3, 3, 32, 32), "NHWC", (1, 1, 1, 1), "VALID"
    )

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data: R.Tensor((1, 128, 128, 32), dtype="float32"),
            kernel: R.Tensor((3, 3, 32, 32), dtype="float32"),
        ) -> R.Tensor((1, 126, 126, 32), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((32, 3, 3, 32), dtype="float32") = R.permute_dims(
                    kernel, axes=[3, 0, 1, 2]
                )
                lv1: R.Tensor((3, 3, 32, 32), dtype="float32") = R.permute_dims(
                    lv, axes=[1, 2, 3, 0]
                )
                lv2: R.Tensor((1, 126, 126, 32), dtype="float32") = R.nn.conv2d(
                    data,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="HWIO",
                    out_layout="NHWC",
                )
                gv: R.Tensor((1, 126, 126, 32), dtype="float32") = R.add(
                    lv2, R.const(np.zeros((32,), dtype="float32"))
                )
                R.output(gv)
            return gv

    verify(Conv2DModule, Expected)


def _make_conv3d_module(data_shape, kernel_shape, strides, padding):
    class Conv3DModule(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=data_shape, dtype=tf.float32),
                tf.TensorSpec(shape=kernel_shape, dtype=tf.float32),
            ]
        )
        def func(self, data, kernel):
            return tf.nn.conv3d(
                input=data,
                filters=kernel,
                strides=strides,
                padding=padding,
            )

    return Conv3DModule


def test_conv3d_valid():
    Conv3DModule = _make_conv3d_module((1, 8, 8, 8, 3), (3, 3, 3, 3, 16), (1, 1, 1, 1, 1), "VALID")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data: R.Tensor((1, 8, 8, 8, 3), dtype="float32"),
            kernel: R.Tensor((3, 3, 3, 3, 16), dtype="float32"),
        ) -> R.Tensor((1, 6, 6, 6, 16), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                gv: R.Tensor((1, 6, 6, 6, 16), dtype="float32") = R.nn.conv3d(
                    data,
                    kernel,
                    strides=[1, 1, 1],
                    padding=[0, 0, 0, 0, 0, 0],
                    dilation=[1, 1, 1],
                    groups=1,
                    data_layout="NDHWC",
                    kernel_layout="DHWIO",
                    out_layout="NDHWC",
                )
                R.output(gv)
            return gv

    verify(Conv3DModule, Expected)


def test_conv3d_same():
    Conv3DModule = _make_conv3d_module((1, 8, 8, 8, 3), (3, 3, 3, 3, 16), (1, 1, 1, 1, 1), "SAME")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data: R.Tensor((1, 8, 8, 8, 3), dtype="float32"),
            kernel: R.Tensor((3, 3, 3, 3, 16), dtype="float32"),
        ) -> R.Tensor((1, 8, 8, 8, 16), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                gv: R.Tensor((1, 8, 8, 8, 16), dtype="float32") = R.nn.conv3d(
                    data,
                    kernel,
                    strides=[1, 1, 1],
                    padding=[1, 1, 1, 1, 1, 1],
                    dilation=[1, 1, 1],
                    groups=1,
                    data_layout="NDHWC",
                    kernel_layout="DHWIO",
                    out_layout="NDHWC",
                )
                R.output(gv)
            return gv

    verify(Conv3DModule, Expected)


def _make_conv3d_transpose_module(data_shape, kernel_shape, strides, padding):
    # Compute the expected output_shape for tf.nn.conv3d_transpose.
    # data_shape: (N, D, H, W, C_in), kernel_shape: (KD, KH, KW, C_out, C_in)
    # strides: (1, sD, sH, sW, 1)
    batch = data_shape[0]
    out_channels = kernel_shape[3]
    out_spatial = []
    for i in range(3):  # D, H, W
        in_size = data_shape[1 + i]
        k_size = kernel_shape[i]
        s = strides[1 + i]
        if padding == "VALID":
            out_spatial.append((in_size - 1) * s + k_size)
        else:  # SAME
            out_spatial.append(in_size * s)
    computed_output_shape = [batch, *out_spatial, out_channels]

    class Conv3DTransposeModule(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=data_shape, dtype=tf.float32),
                tf.TensorSpec(shape=kernel_shape, dtype=tf.float32),
            ]
        )
        def func(self, data, kernel):
            return tf.nn.conv3d_transpose(
                input=data,
                filters=kernel,
                output_shape=computed_output_shape,
                strides=strides,
                padding=padding,
            )

    return Conv3DTransposeModule


def test_conv3d_transpose_valid():
    Conv3DTransposeModule = _make_conv3d_transpose_module(
        (1, 8, 8, 8, 3), (3, 3, 3, 8, 3), (1, 1, 1, 1, 1), "VALID"
    )

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data: R.Tensor((1, 8, 8, 8, 3), dtype="float32"),
            kernel: R.Tensor((3, 3, 3, 8, 3), dtype="float32"),
        ) -> R.Tensor((1, 10, 10, 10, 8), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                gv: R.Tensor((1, 10, 10, 10, 8), dtype="float32") = R.nn.conv3d_transpose(
                    data,
                    kernel,
                    strides=[1, 1, 1],
                    padding=[0, 0, 0, 0, 0, 0],
                    output_padding=[0, 0, 0],
                    dilation=[1, 1, 1],
                    groups=1,
                    data_layout="NDHWC",
                    kernel_layout="DHWOI",
                    out_layout="NDHWC",
                )
                R.output(gv)
            return gv

    verify(Conv3DTransposeModule, Expected)


def test_conv3d_transpose_same():
    Conv3DTransposeModule = _make_conv3d_transpose_module(
        (1, 8, 8, 8, 3), (3, 3, 3, 8, 3), (1, 1, 1, 1, 1), "SAME"
    )

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data: R.Tensor((1, 8, 8, 8, 3), dtype="float32"),
            kernel: R.Tensor((3, 3, 3, 8, 3), dtype="float32"),
        ) -> R.Tensor((1, 8, 8, 8, 8), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                gv: R.Tensor((1, 8, 8, 8, 8), dtype="float32") = R.nn.conv3d_transpose(
                    data,
                    kernel,
                    strides=[1, 1, 1],
                    padding=[1, 1, 1, 1, 1, 1],
                    output_padding=[0, 0, 0],
                    dilation=[1, 1, 1],
                    groups=1,
                    data_layout="NDHWC",
                    kernel_layout="DHWOI",
                    out_layout="NDHWC",
                )
                R.output(gv)
            return gv

    verify(Conv3DTransposeModule, Expected)


def _make_pool2d_module(pool, data_shape, ksize, data_format, strides, padding):
    class Pool2DModule(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=data_shape, dtype=tf.float32),
            ]
        )
        def func(self, data):
            return pool(
                input=data,
                ksize=ksize,
                data_format=data_format,
                strides=strides,
                padding=padding,
            )

    return Pool2DModule


def test_avg_pool2d_same():
    Pool2DModule = _make_pool2d_module(
        tf.nn.avg_pool2d, (1, 128, 128, 32), (2, 2), "NHWC", (1, 1, 1, 1), "SAME"
    )

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data: R.Tensor((1, 128, 128, 32), dtype="float32"),
        ) -> R.Tensor((1, 128, 128, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 128, 128, 32), dtype="float32") = R.nn.avg_pool2d(
                    data,
                    pool_size=[2, 2],
                    strides=[1, 1],
                    dilation=[1, 1],
                    padding=[0, 0, 1, 1],
                    ceil_mode=False,
                    count_include_pad=False,
                    layout="NHWC",
                    out_layout="NHWC",
                )
                R.output(gv)
            return gv

    verify(Pool2DModule, Expected)


def test_avg_pool2d_valid():
    Pool2DModule = _make_pool2d_module(
        tf.nn.avg_pool2d, (1, 128, 128, 32), (2, 2), "NHWC", (1, 1, 1, 1), "VALID"
    )

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data: R.Tensor((1, 128, 128, 32), dtype="float32"),
        ) -> R.Tensor((1, 127, 127, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 127, 127, 32), dtype="float32") = R.nn.avg_pool2d(
                    data,
                    pool_size=[2, 2],
                    strides=[1, 1],
                    dilation=[1, 1],
                    padding=[0, 0, 0, 0],
                    ceil_mode=False,
                    count_include_pad=False,
                    layout="NHWC",
                    out_layout="NHWC",
                )
                R.output(gv)
            return gv

    verify(Pool2DModule, Expected)


def test_max_pool2d_same():
    Pool2DModule = _make_pool2d_module(
        tf.nn.max_pool2d, (1, 128, 128, 32), (2, 2), "NHWC", (1, 1, 1, 1), "SAME"
    )

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data: R.Tensor((1, 128, 128, 32), dtype="float32"),
        ) -> R.Tensor((1, 128, 128, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 128, 128, 32), dtype="float32") = R.nn.max_pool2d(
                    data,
                    pool_size=[2, 2],
                    strides=[1, 1],
                    dilation=[1, 1],
                    padding=[0, 0, 1, 1],
                    ceil_mode=False,
                    layout="NHWC",
                    out_layout="NHWC",
                )
                R.output(gv)
            return gv

    verify(Pool2DModule, Expected)


def test_max_pool2d_valid():
    Pool2DModule = _make_pool2d_module(
        tf.nn.max_pool2d, (1, 128, 128, 32), (2, 2), "NHWC", (1, 1, 1, 1), "VALID"
    )

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data: R.Tensor((1, 128, 128, 32), dtype="float32"),
        ) -> R.Tensor((1, 127, 127, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 127, 127, 32), dtype="float32") = R.nn.max_pool2d(
                    data,
                    pool_size=[2, 2],
                    strides=[1, 1],
                    dilation=[1, 1],
                    padding=[0, 0, 0, 0],
                    ceil_mode=False,
                    count_include_pad=False,
                    layout="NHWC",
                    out_layout="NHWC",
                )
                R.output(gv)
            return gv

    verify(Pool2DModule, Expected)


@pytest.mark.parametrize(
    "net, shape",
    [
        # Limiting the tests for CI
        (keras_app.Xception, (1, 299, 299, 3)),
        # (keras_app.VGG16, (1, 224, 224, 3)),
        # (keras_app.VGG19, (1, 224, 224, 3)),
        (keras_app.ResNet50, (1, 224, 224, 3)),
        # (keras_app.ResNet50V2, (1, 224, 224, 3)),
        # (keras_app.ResNet101, (1, 224, 224, 3)),
        # (keras_app.ResNet101V2, (1, 224, 224, 3)),
        # (keras_app.ResNet152, (1, 224, 224, 3)),
        # (keras_app.ResNet152V2, (1, 224, 224, 3)),
        (keras_app.InceptionResNetV2, (1, 299, 299, 3)),
        # (keras_app.MobileNet, (1, 224, 224, 3)),
        (keras_app.MobileNetV2, (1, 224, 224, 3)),
        (keras_app.DenseNet121, (1, 224, 224, 3)),
        # (keras_app.DenseNet169, (1, 224, 224, 3)),
        # (keras_app.DenseNet201, (1, 224, 224, 3)),
        (keras_app.NASNetMobile, (1, 224, 224, 3)),
        # (keras_app.NASNetLarge, (1, 331, 331, 3)),
        (keras_app.EfficientNetB0, (1, 224, 224, 3)),
        # (keras_app.EfficientNetB1, (1, 240, 240, 3)),
        # (keras_app.EfficientNetB2, (1, 260, 260, 3)),
        # (keras_app.EfficientNetB3, (1, 300, 300, 3)),
        # (keras_app.EfficientNetB4, (1, 380, 380, 3)),
        # (keras_app.EfficientNetB5, (1, 456, 456, 3)),
        # (keras_app.EfficientNetB6, (1, 528, 528, 3)),
        # (keras_app.EfficientNetB7, (1, 600, 600, 3)),
        (keras_app.EfficientNetV2B0, (1, 224, 224, 3)),
        # (keras_app.EfficientNetV2B1, (1, 240, 240, 3)),
        # (keras_app.EfficientNetV2B2, (1, 260, 260, 3)),
        # (keras_app.EfficientNetV2B3, (1, 300, 300, 3)),
        # (keras_app.EfficientNetV2S, (1, 384, 384, 3)),
        # (keras_app.EfficientNetV2M, (1, 480, 480, 3)),
        # (keras_app.EfficientNetV2L, (1, 480, 480, 3)),
        # (keras_app.ConvNeXtTiny, (1, 224, 224, 3)),
        # (keras_app.ConvNeXtSmall, (1, 224, 224, 3)),
        # (keras_app.ConvNeXtBase, (1, 224, 224, 3)),
        # (keras_app.ConvNeXtLarge, (1, 224, 224, 3)),
        # (keras_app.ConvNeXtXLarge, (1, 224, 224, 3)),
    ],
)
def test_networks(net, shape):
    # Run network tests only in nightly builds
    if "CI_ENV_NIGHTLY" not in os.environ:
        return

    class NetworkModule(tf.Module):
        def __init__(self):
            self.model = net(weights=None, include_top=True)

        @tf.function
        def func(self, data):
            return self.model(data, training=False)

    model = NetworkModule()
    concrete_func = model.func.get_concrete_function(tf.TensorSpec(shape=shape, dtype=tf.float32))

    mod = _get_mod_from_cfunc(concrete_func)
    tvm_tirx.ir.assert_structural_equal(mod["main"].ret_ty, relax.TensorType((1, 1000), "float32"))
    verify(concrete_func)


def test_select_v2():
    class Model(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(2, 2), dtype=tf.bool),
                tf.TensorSpec(shape=(2, 2), dtype=tf.float32),
                tf.TensorSpec(shape=(2, 2), dtype=tf.float32),
            ]
        )
        def func(self, condition, x, y):
            return tf.where(condition, x, y)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            condition: R.Tensor((2, 2), dtype="bool"),
            x: R.Tensor((2, 2), dtype="float32"),
            y: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tensor((2, 2), dtype="float32"):
            R.func_attr({"num_input": 3})
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="float32") = R.where(condition, x, y)
                R.output(gv)
            return gv

    verify(Model, Expected)

    class ModelBroadcasting(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(2, 1), dtype=tf.bool),
                tf.TensorSpec(shape=(2, 2), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
            ]
        )
        def func(self, condition, x, y):
            return tf.where(condition, x, y)

    @I.ir_module
    class ExpectedBroadcasting:
        @R.function
        def main(
            condition: R.Tensor((2, 1), dtype="bool"),
            x: R.Tensor((2, 2), dtype="float32"),
            y: R.Tensor((), dtype="float32"),
        ) -> R.Tensor((2, 2), dtype="float32"):
            R.func_attr({"num_input": 3})
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="float32") = R.where(condition, x, y)
                R.output(gv)
            return gv

    verify(ModelBroadcasting, ExpectedBroadcasting)


def test_scatter_nd():
    class Model(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(4, 1), dtype=tf.int32),
                tf.TensorSpec(shape=(4,), dtype=tf.float32),
                tf.TensorSpec(shape=(1,), dtype=tf.int32),
            ]
        )
        def func(self, indices, updates, shape):
            return tf.scatter_nd(indices, updates, shape)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            indices: R.Tensor((4, 1), dtype="int32"),
            updates: R.Tensor((4,), dtype="float32"),
            shape: R.Tensor((1,), dtype="int32"),
        ) -> R.Tensor(dtype="float32", ndim=1):
            R.func_attr({"num_input": 3})
            with R.dataflow():
                lv: R.Tensor((1,), dtype="int64") = R.astype(shape, dtype="int64")
                lv1: R.Shape(ndim=1) = R.tensor_to_shape(lv)
                lv2: R.Tensor(lv1, dtype="float32") = R.zeros(lv1, dtype="float32")
                lv3: R.Tensor((1, 4), dtype="int32") = R.permute_dims(indices, axes=[-1, 0])
                gv: R.Tensor(dtype="float32", ndim=1) = R.scatter_nd(
                    lv2, lv3, updates, reduction="update"
                )
                R.output(lv1, gv)
            return gv

    verify(Model, Expected)


def test_segment_sum():
    """SEGMENT_SUM lowers to scatter_nd with add reduction."""

    class Model(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(4, 2), dtype=tf.float32)])
        def func(self, data):
            return tf.raw_ops.SegmentSum(
                data=data, segment_ids=tf.constant([0, 0, 1, 2], dtype=tf.int32)
            )

    @I.ir_module
    class Expected:
        @R.function
        def main(data: R.Tensor((4, 2), dtype="float32")) -> R.Tensor((3, 2), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((3, 2), dtype="float32") = R.zeros(R.shape([3, 2]), dtype="float32")
                lv1: R.Tensor((4, 1), dtype="int32") = R.expand_dims(
                    R.const([0, 0, 1, 2], "int32"), axis=[1]
                )
                gv: R.Tensor((3, 2), dtype="float32") = R.scatter_nd(lv, lv1, data, reduction="add")
                R.output(gv)
            return gv

    verify(Model, Expected)


def test_unsorted_segment_min():
    """UNSORTED_SEGMENT_MIN lowers to scatter_nd with min reduction."""

    class Model(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(4, 2), dtype=tf.float32)])
        def func(self, data):
            return tf.raw_ops.UnsortedSegmentMin(
                data=data,
                segment_ids=tf.constant([2, 0, 2, 1], dtype=tf.int32),
                num_segments=tf.constant(3, dtype=tf.int32),
            )

    @I.ir_module
    class Expected:
        @R.function
        def main(data: R.Tensor((4, 2), dtype="float32")) -> R.Tensor((3, 2), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((3, 2), dtype="float32") = R.full(
                    R.shape([3, 2]), R.const(np.finfo(np.float32).max, "float32"), dtype="float32"
                )
                lv1: R.Tensor((4, 1), dtype="int32") = R.expand_dims(
                    R.const([2, 0, 2, 1], "int32"), axis=[1]
                )
                gv: R.Tensor((3, 2), dtype="float32") = R.scatter_nd(lv, lv1, data, reduction="min")
                R.output(gv)
            return gv

    verify(Model, Expected)


def test_unsorted_segment_sum():
    """UNSORTED_SEGMENT_SUM lowers to scatter_nd with add reduction."""

    class Model(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(4, 2), dtype=tf.float32)])
        def func(self, data):
            return tf.raw_ops.UnsortedSegmentSum(
                data=data,
                segment_ids=tf.constant([0, 2, 1, 2], dtype=tf.int32),
                num_segments=tf.constant(3, dtype=tf.int32),
            )

    @I.ir_module
    class Expected:
        @R.function
        def main(data: R.Tensor((4, 2), dtype="float32")) -> R.Tensor((3, 2), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((3, 2), dtype="float32") = R.zeros(R.shape([3, 2]), dtype="float32")
                lv1: R.Tensor((4, 1), dtype="int32") = R.expand_dims(
                    R.const([0, 2, 1, 2], "int32"), axis=[1]
                )
                gv: R.Tensor((3, 2), dtype="float32") = R.scatter_nd(lv, lv1, data, reduction="add")
                R.output(gv)
            return gv

    verify(Model, Expected)


def test_unsorted_segment_max():
    """UNSORTED_SEGMENT_MAX lowers to scatter_nd with max reduction."""

    class Model(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(4, 2), dtype=tf.float32)])
        def func(self, data):
            return tf.raw_ops.UnsortedSegmentMax(
                data=data,
                segment_ids=tf.constant([0, 2, 1, 2], dtype=tf.int32),
                num_segments=tf.constant(3, dtype=tf.int32),
            )

    @I.ir_module
    class Expected:
        @R.function
        def main(data: R.Tensor((4, 2), dtype="float32")) -> R.Tensor((3, 2), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((3, 2), dtype="float32") = R.full(
                    R.shape([3, 2]), R.const(np.finfo(np.float32).min, "float32"), dtype="float32"
                )
                lv1: R.Tensor((4, 1), dtype="int32") = R.expand_dims(
                    R.const([0, 2, 1, 2], "int32"), axis=[1]
                )
                gv: R.Tensor((3, 2), dtype="float32") = R.scatter_nd(lv, lv1, data, reduction="max")
                R.output(gv)
            return gv

    verify(Model, Expected)


def test_unsorted_segment_prod():
    """UNSORTED_SEGMENT_PROD lowers to scatter_nd with mul reduction."""

    class Model(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(4, 2), dtype=tf.float32)])
        def func(self, data):
            return tf.raw_ops.UnsortedSegmentProd(
                data=data,
                segment_ids=tf.constant([1, 0, 1, 2], dtype=tf.int32),
                num_segments=tf.constant(3, dtype=tf.int32),
            )

    @I.ir_module
    class Expected:
        @R.function
        def main(data: R.Tensor((4, 2), dtype="float32")) -> R.Tensor((3, 2), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((3, 2), dtype="float32") = R.full(
                    R.shape([3, 2]), R.const(1, "float32"), dtype="float32"
                )
                lv1: R.Tensor((4, 1), dtype="int32") = R.expand_dims(
                    R.const([1, 0, 1, 2], "int32"), axis=[1]
                )
                gv: R.Tensor((3, 2), dtype="float32") = R.scatter_nd(lv, lv1, data, reduction="mul")
                R.output(gv)
            return gv

    verify(Model, Expected)


def test_batch_matmul():
    class BatchMatMul(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(2, 3, 4), dtype=tf.float32),
                tf.TensorSpec(shape=(2, 4, 5), dtype=tf.float32),
            ]
        )
        def func(self, x, y):
            return tf.matmul(x, y)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 4), dtype="float32"),
            y: R.Tensor((2, 4, 5), dtype="float32"),
        ) -> R.Tensor((2, 3, 5), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((2, 3, 5), dtype="float32") = R.matmul(x, y)
                gv: R.Tensor((2, 3, 5), dtype="float32") = R.reshape(lv, R.shape([2, 3, 5]))
                R.output(gv)
            return gv

    verify(BatchMatMul, Expected)


def test_batch_matmul_adj():
    class BatchMatMulAdj(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(2, 4, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(2, 5, 4), dtype=tf.float32),
            ]
        )
        def func(self, x, y):
            return tf.matmul(x, y, transpose_a=True, transpose_b=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 4, 3), dtype="float32"),
            y: R.Tensor((2, 5, 4), dtype="float32"),
        ) -> R.Tensor((2, 3, 5), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((2, 3, 4), dtype="float32") = R.permute_dims(x, axes=[0, 2, 1])
                lv1: R.Tensor((2, 4, 5), dtype="float32") = R.permute_dims(y, axes=[0, 2, 1])
                lv2: R.Tensor((2, 3, 5), dtype="float32") = R.matmul(lv, lv1)
                gv: R.Tensor((2, 3, 5), dtype="float32") = R.reshape(lv2, R.shape([2, 3, 5]))
                R.output(gv)
            return gv

    verify(BatchMatMulAdj, Expected)


def _verify_nms_v4(mod, tf_func, boxes_np, scores_np):
    """E2E verify for NMS V4: only run on nightly, compare valid outputs only."""
    if "CI_ENV_NIGHTLY" not in os.environ:
        return

    tf_indices, tf_valid = tf_func(tf.constant(boxes_np), tf.constant(scores_np))
    n_valid = int(tf_valid.numpy())

    tgt = tvm_tirx.target.Target("llvm")
    ex = tvm.compile(mod, tgt)
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    vm.set_input("main", boxes_np, scores_np)
    vm.invoke_stateful("main")
    tvm_indices, tvm_valid = vm.get_outputs("main")

    assert int(tvm_valid.numpy()) == n_valid
    np.testing.assert_array_equal(
        tf_indices.numpy()[:n_valid],
        tvm_indices.numpy()[:n_valid],
    )


def _build_nms_v4_mod(num_boxes, max_output_size, iou_threshold, score_threshold):
    """Convert a NonMaxSuppressionV4 TFLite model to a Relax module.

    Scalar params must be Python literals (not tf.constant) so TFLite can
    statically infer output shapes during conversion.
    """

    class NMSv4Module(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(num_boxes, 4), dtype=tf.float32),
                tf.TensorSpec(shape=(num_boxes,), dtype=tf.float32),
            ]
        )
        def func(self, boxes, scores):
            indices, valid = tf.raw_ops.NonMaxSuppressionV4(
                boxes=boxes,
                scores=scores,
                max_output_size=max_output_size,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold,
                pad_to_max_output_size=True,
            )
            return indices, valid

    instance = NMSv4Module()
    cf = instance.func.get_concrete_function()
    mod = _get_mod_from_cfunc(cf)
    return mod, instance.func


def _verify_nms_v5(mod, tf_func, boxes_np, scores_np, soft_nms_sigma=0.0):
    """E2E verify for NMS: only run on nightly, compare valid outputs only."""
    if "CI_ENV_NIGHTLY" not in os.environ:
        return

    tf_indices, tf_scores, tf_valid = tf_func(tf.constant(boxes_np), tf.constant(scores_np))
    n_valid = int(tf_valid.numpy())

    tgt = tvm_tirx.target.Target("llvm")
    ex = tvm.compile(mod, tgt)
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    vm.set_input("main", boxes_np, scores_np)
    vm.invoke_stateful("main")
    tvm_indices, tvm_scores, tvm_valid = vm.get_outputs("main")

    assert int(tvm_valid.numpy()) == n_valid
    np.testing.assert_array_equal(
        tf_indices.numpy()[:n_valid],
        tvm_indices.numpy()[:n_valid],
    )
    np.testing.assert_allclose(
        tf_scores.numpy()[:n_valid],
        tvm_scores.numpy()[:n_valid],
        rtol=1e-5,
        atol=1e-5,
    )
    if soft_nms_sigma > 0.0:
        np.testing.assert_allclose(
            tf_scores.numpy(),
            tvm_scores.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_array_less(-1e-6, tvm_scores.numpy()[n_valid:])


def _build_nms_v5_mod(
    num_boxes, max_output_size, iou_threshold, score_threshold, soft_nms_sigma=0.0
):
    """Convert a NonMaxSuppressionV5 TFLite model to a Relax module.

    Scalar params must be Python literals (not tf.constant) so TFLite can
    statically infer output shapes during conversion.
    """

    class NMSv5Module(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(num_boxes, 4), dtype=tf.float32),
                tf.TensorSpec(shape=(num_boxes,), dtype=tf.float32),
            ]
        )
        def func(self, boxes, scores):
            indices, out_scores, valid = tf.raw_ops.NonMaxSuppressionV5(
                boxes=boxes,
                scores=scores,
                max_output_size=max_output_size,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold,
                soft_nms_sigma=soft_nms_sigma,
                pad_to_max_output_size=True,
            )
            return indices, out_scores, valid

    instance = NMSv5Module()
    cf = instance.func.get_concrete_function()
    mod = _get_mod_from_cfunc(cf)
    return mod, instance.func


class _StubDetectionPostprocessTensor:
    def __init__(self, shape, name):
        self._shape = list(shape)
        self._name = name

    def Shape(self, index):
        return self._shape[index]

    def Name(self):
        return self._name

    def Type(self):
        return 0


class _StubDetectionPostprocessOp:
    def __init__(self, custom_options):
        self._custom_options = _encode_detection_postprocess_custom_options(custom_options)

    def CustomOptionsAsNumpy(self):
        return np.frombuffer(self._custom_options, dtype="uint8")


_DETECTION_POSTPROCESS_ANCHORS = np.array(
    [
        [0.5, 0.5, 1.0, 1.0],
        [0.5, 0.2, 1.0, 1.0],
        [0.1, 0.1, 0.5, 0.5],
        [0.8, 0.8, 0.2, 0.2],
    ],
    dtype="float32",
)


def _encode_detection_postprocess_custom_options(custom_options):
    from flatbuffers import flexbuffers

    builder = flexbuffers.Builder()
    with builder.Map():
        for key, value in custom_options.items():
            if isinstance(value, bool):
                builder.Bool(key, value)
            elif isinstance(value, int):
                builder.Int(key, value)
            else:
                builder.Float(key, float(value))
    return bytes(builder.Finish())


def _make_detection_postprocess_tensor_wrapper(tensor_idx, shape, name):
    return tflite_frontend.TensorWrapper(
        tensor_idx,
        _StubDetectionPostprocessTensor(shape, name),
        None,
    )


def _build_detection_postprocess_mod(
    *,
    num_classes=1,
    max_detections=4,
    detections_per_class=4,
    use_regular_nms=False,
    nms_iou_threshold=0.5,
    nms_score_threshold=0.3,
    x_scale=10.0,
    y_scale=10.0,
    w_scale=5.0,
    h_scale=5.0,
    batch_size=2,
    num_anchors=4,
    input_num_classes=None,
):
    custom_options = {
        "num_classes": num_classes,
        "max_detections": max_detections,
        "detections_per_class": detections_per_class,
        "nms_iou_threshold": nms_iou_threshold,
        "nms_score_threshold": nms_score_threshold,
        "x_scale": x_scale,
        "y_scale": y_scale,
        "w_scale": w_scale,
        "h_scale": h_scale,
        "use_regular_nms": use_regular_nms,
    }
    return _convert_detection_postprocess_with_options(
        custom_options,
        batch_size=batch_size,
        num_anchors=num_anchors,
        num_classes=num_classes,
        input_num_classes=input_num_classes,
    )


def _convert_detection_postprocess_with_options(
    custom_options,
    *,
    batch_size=2,
    num_anchors=4,
    num_classes=1,
    input_num_classes=None,
    build_module=True,
):
    input_num_classes = num_classes if input_num_classes is None else input_num_classes
    loc = relax.Var("loc", relax.TensorType((batch_size, num_anchors, 4), "float32"))
    cls = relax.Var(
        "cls", relax.TensorType((batch_size, num_anchors, input_num_classes), "float32")
    )
    inputs = [
        _make_detection_postprocess_tensor_wrapper(0, (batch_size, num_anchors, 4), "loc"),
        _make_detection_postprocess_tensor_wrapper(
            1, (batch_size, num_anchors, input_num_classes), "cls"
        ),
        _make_detection_postprocess_tensor_wrapper(2, (num_anchors, 4), "anchors"),
    ]
    converter = tflite_frontend.OperatorConverter.__new__(tflite_frontend.OperatorConverter)
    converter.bb = relax.BlockBuilder()
    converter.exp_tab = tflite_frontend.ExprTable()
    converter.get_input_tensors = lambda op: inputs
    converter.get_expr = lambda tensor_idx: {0: loc, 1: cls}[tensor_idx]
    converter.get_tensor_value = lambda tensor: (
        _DETECTION_POSTPROCESS_ANCHORS if tensor.tensor_idx == 2 else None
    )
    converter.get_tensor_type_str = lambda tensor_type: "float32"
    op = _StubDetectionPostprocessOp(custom_options)
    if not build_module:
        return converter.convert_detection_postprocess(op)
    bb = converter.bb
    with bb.function("main", [loc, cls]):
        with bb.dataflow():
            output = converter.convert_detection_postprocess(op)
            gv = bb.emit_output(output)
        bb.emit_func_output(gv)
    return bb.get()


def _make_valid_boxes(rng, n):
    """Generate n random boxes with y1<=y2, x1<=x2 using the given RNG."""
    raw = rng.random((n, 4), dtype=np.float32)
    return np.stack(
        [
            np.minimum(raw[:, 0], raw[:, 2]),  # y1
            np.minimum(raw[:, 1], raw[:, 3]),  # x1
            np.maximum(raw[:, 0], raw[:, 2]),  # y2
            np.maximum(raw[:, 1], raw[:, 3]),  # x2
        ],
        axis=1,
    ).astype(np.float32)


_NMS_V5_CASES = [
    pytest.param(
        6,
        3,
        0.5,
        0.0,
        np.array(
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.1, 1.0, 1.1],
                [0.0, 0.0, 1.0, 0.9],
                [0.5, 0.5, 1.5, 1.5],
                [0.0, 0.0, 0.3, 0.3],
            ],
            dtype=np.float32,
        ),
        np.array([0.9, 0.75, 0.6, 0.5, 0.4, 0.3], dtype=np.float32),
        id="basic",
    ),
    pytest.param(
        8,
        4,
        0.5,
        0.4,
        _make_valid_boxes(np.random.default_rng(42), 8),
        np.random.default_rng(42).random(8, dtype=np.float32),
        id="score_threshold",
    ),
    pytest.param(
        5,
        3,
        0.5,
        0.99,
        _make_valid_boxes(np.random.default_rng(0), 5),
        np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
        id="all_suppressed",
    ),
    pytest.param(
        6,
        6,
        0.1,
        0.0,
        np.array(
            [
                [0.0, 0.0, 0.4, 0.4],
                [0.5, 0.5, 0.9, 0.9],
                [0.1, 0.1, 0.5, 0.5],
                [0.6, 0.6, 1.0, 1.0],
                [0.0, 0.5, 0.4, 0.9],
                [0.5, 0.0, 0.9, 0.4],
            ],
            dtype=np.float32,
        ),
        np.array([0.9, 0.85, 0.7, 0.65, 0.6, 0.55], dtype=np.float32),
        id="iou_threshold",
    ),
    pytest.param(
        4,
        10,
        0.5,
        0.0,
        np.array(
            [
                [0.0, 0.0, 0.3, 0.3],
                [0.5, 0.5, 0.8, 0.8],
                [0.1, 0.1, 0.4, 0.4],
                [0.6, 0.6, 0.9, 0.9],
            ],
            dtype=np.float32,
        ),
        np.array([0.9, 0.85, 0.7, 0.65], dtype=np.float32),
        id="max_output_size_larger_than_boxes",
    ),
]


_NMS_V5_SOFT_CASES = [
    pytest.param(
        6,
        6,
        0.5,
        0.0,
        0.5,
        np.array(
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.1, 1.0, 1.1],
                [0.0, 0.0, 1.0, 0.9],
                [0.5, 0.5, 1.5, 1.5],
                [0.0, 0.0, 0.3, 0.3],
            ],
            dtype=np.float32,
        ),
        np.array([0.9, 0.75, 0.6, 0.5, 0.4, 0.3], dtype=np.float32),
        id="soft_nms_basic",
    ),
    pytest.param(
        5,
        5,
        0.5,
        0.0,
        0.3,
        np.array(
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.1, 0.1, 1.1, 1.1],
                [0.2, 0.2, 1.2, 1.2],
                [0.3, 0.3, 1.3, 1.3],
                [2.0, 2.0, 3.0, 3.0],
            ],
            dtype=np.float32,
        ),
        np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32),
        id="soft_nms_tight_sigma",
    ),
    pytest.param(
        3,
        3,
        0.5,
        0.3,
        0.1,
        np.array(
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.2, 0.2, 1.2, 1.2],
                [2.0, 2.0, 3.0, 3.0],
            ],
            dtype=np.float32,
        ),
        np.array([0.9, 0.8, 0.75], dtype=np.float32),
        id="soft_nms_threshold_hole",
    ),
    pytest.param(
        3,
        3,
        0.5,
        0.0,
        0.1,
        np.array(
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.2, 0.2, 1.2, 1.2],
                [2.0, 2.0, 3.0, 3.0],
            ],
            dtype=np.float32,
        ),
        np.array([0.9, 0.85, 0.8], dtype=np.float32),
        id="soft_nms_reorder",
    ),
]


@pytest.mark.parametrize(
    "num_boxes,max_output_size,iou_threshold,score_threshold,boxes,scores",
    _NMS_V5_CASES,
)
def test_nms_v5(num_boxes, max_output_size, iou_threshold, score_threshold, boxes, scores):
    """NON_MAX_SUPPRESSION_V5: conversion smoke test + E2E correctness (nightly only)."""
    mod, tf_func = _build_nms_v5_mod(num_boxes, max_output_size, iou_threshold, score_threshold)
    _verify_nms_v5(mod, tf_func, boxes, scores)


@pytest.mark.parametrize(
    "num_boxes,max_output_size,iou_threshold,score_threshold,soft_nms_sigma,boxes,scores",
    _NMS_V5_SOFT_CASES,
)
def test_nms_v5_soft(
    num_boxes, max_output_size, iou_threshold, score_threshold, soft_nms_sigma, boxes, scores
):
    """NON_MAX_SUPPRESSION_V5 with soft_nms_sigma: conversion smoke test + E2E correctness."""
    mod, tf_func = _build_nms_v5_mod(
        num_boxes, max_output_size, iou_threshold, score_threshold, soft_nms_sigma
    )
    _verify_nms_v5(mod, tf_func, boxes, scores, soft_nms_sigma=soft_nms_sigma)


def test_nms_v5_ir():
    """Verify the emitted Relax IR has correct structure for NON_MAX_SUPPRESSION_V5."""
    num_boxes = 6
    max_output_size = 3
    mod, _ = _build_nms_v5_mod(
        num_boxes=num_boxes,
        max_output_size=max_output_size,
        iou_threshold=0.5,
        score_threshold=0.0,
    )

    tvm_tirx.ir.assert_structural_equal(
        mod["main"].ret_ty,
        relax.TupleType(
            [
                relax.TensorType((max_output_size,), "int32"),
                relax.TensorType((max_output_size,), "float32"),
                relax.TensorType((), "int32"),
            ]
        ),
    )


_NMS_V4_CASES = [
    pytest.param(
        6,
        3,
        0.5,
        0.0,
        np.array(
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.1, 1.0, 1.1],
                [0.0, 0.0, 1.0, 0.9],
                [0.5, 0.5, 1.5, 1.5],
                [0.0, 0.0, 0.3, 0.3],
            ],
            dtype=np.float32,
        ),
        np.array([0.9, 0.75, 0.6, 0.5, 0.4, 0.3], dtype=np.float32),
        id="basic",
    ),
    pytest.param(
        8,
        4,
        0.5,
        0.4,
        _make_valid_boxes(np.random.default_rng(42), 8),
        np.random.default_rng(42).random(8, dtype=np.float32),
        id="score_threshold",
    ),
    pytest.param(
        5,
        3,
        0.5,
        0.99,
        _make_valid_boxes(np.random.default_rng(0), 5),
        np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
        id="all_suppressed",
    ),
    pytest.param(
        4,
        10,
        0.5,
        0.0,
        np.array(
            [
                [0.0, 0.0, 0.3, 0.3],
                [0.5, 0.5, 0.8, 0.8],
                [0.1, 0.1, 0.4, 0.4],
                [0.6, 0.6, 0.9, 0.9],
            ],
            dtype=np.float32,
        ),
        np.array([0.9, 0.85, 0.7, 0.65], dtype=np.float32),
        id="max_output_size_larger_than_boxes",
    ),
]


@pytest.mark.parametrize(
    "num_boxes,max_output_size,iou_threshold,score_threshold,boxes,scores",
    _NMS_V4_CASES,
)
def test_nms_v4(num_boxes, max_output_size, iou_threshold, score_threshold, boxes, scores):
    """NON_MAX_SUPPRESSION_V4: conversion smoke test + E2E correctness (nightly only)."""
    mod, tf_func = _build_nms_v4_mod(num_boxes, max_output_size, iou_threshold, score_threshold)
    _verify_nms_v4(mod, tf_func, boxes, scores)


def test_nms_v4_ir():
    """Verify the emitted Relax IR has correct structure for NON_MAX_SUPPRESSION_V4."""
    num_boxes = 6
    max_output_size = 3
    mod, _ = _build_nms_v4_mod(
        num_boxes=num_boxes,
        max_output_size=max_output_size,
        iou_threshold=0.5,
        score_threshold=0.0,
    )

    tvm_tirx.ir.assert_structural_equal(
        mod["main"].ret_ty,
        relax.TupleType(
            [
                relax.TensorType((max_output_size,), "int32"),
                relax.TensorType((), "int32"),
            ]
        ),
    )


_DETECTION_POSTPROCESS_SMOKE_CASES = [
    pytest.param(
        {
            "num_classes": 2,
            "input_num_classes": 3,
            "max_detections": 2,
            "detections_per_class": 2,
            "use_regular_nms": False,
            "nms_iou_threshold": 0.5,
            "nms_score_threshold": 0.5,
            "batch_size": 1,
            "num_anchors": 4,
        },
        2,
        True,
        id="basic_fast_nms",
    ),
    pytest.param(
        {
            "num_classes": 2,
            "input_num_classes": 3,
            "max_detections": 3,
            "detections_per_class": 2,
            "use_regular_nms": True,
            "nms_iou_threshold": 0.45,
            "nms_score_threshold": 0.25,
            "batch_size": 2,
            "num_anchors": 4,
        },
        1,
        True,
        id="regular_nms_multi_batch",
    ),
]


_DETECTION_POSTPROCESS_SHAPE_CASES = [
    pytest.param(
        {
            "num_classes": 2,
            "input_num_classes": 2,
            "max_detections": 2,
            "detections_per_class": 2,
            "use_regular_nms": False,
            "nms_iou_threshold": 0.5,
            "nms_score_threshold": 0.5,
            "batch_size": 1,
            "num_anchors": 4,
        },
        id="matching_input_classes",
    ),
    pytest.param(
        {
            "num_classes": 2,
            "input_num_classes": 3,
            "max_detections": 4,
            "detections_per_class": 4,
            "use_regular_nms": False,
            "nms_iou_threshold": 0.5,
            "nms_score_threshold": 0.5,
            "batch_size": 1,
            "num_anchors": 4,
        },
        id="larger_max_detections",
    ),
]


@pytest.mark.parametrize(
    "build_kwargs,expected_topk_count,expected_keep_background",
    _DETECTION_POSTPROCESS_SMOKE_CASES,
)
def test_detection_postprocess_smoke(build_kwargs, expected_topk_count, expected_keep_background):
    mod = _build_detection_postprocess_mod(**build_kwargs)

    topk_calls = []
    multibox_calls = []

    def _visit(expr):
        if isinstance(expr, relax.Call) and expr.op == tvm_tirx.ir.Op.get("relax.topk"):
            topk_calls.append(expr)
        if isinstance(expr, relax.Call) and expr.op == tvm_tirx.ir.Op.get(
            "relax.vision.multibox_transform_loc"
        ):
            multibox_calls.append(expr)

    relax.analysis.post_order_visit(mod["main"].body, _visit)
    assert len(topk_calls) == expected_topk_count
    assert len(multibox_calls) == 1
    assert multibox_calls[0].attrs.keep_background == expected_keep_background
    assert not multibox_calls[0].attrs.apply_softmax

    expected_batch = build_kwargs["batch_size"]
    expected_max_detections = build_kwargs["max_detections"]
    tvm_tirx.ir.assert_structural_equal(
        mod["main"].ret_ty,
        relax.TupleType(
            [
                relax.TensorType((expected_batch, expected_max_detections, 4), "float32"),
                relax.TensorType((expected_batch, expected_max_detections), "float32"),
                relax.TensorType((expected_batch, expected_max_detections), "float32"),
                relax.TensorType((expected_batch,), "float32"),
            ]
        ),
    )

    legalized = relax.transform.LegalizeOps()(mod)
    tvm_tirx.ir.assert_structural_equal(legalized["main"].ret_ty, mod["main"].ret_ty)


@pytest.mark.parametrize("build_kwargs", _DETECTION_POSTPROCESS_SHAPE_CASES)
def test_detection_postprocess_shape_variations(build_kwargs):
    mod = _build_detection_postprocess_mod(**build_kwargs)
    batch_size = build_kwargs["batch_size"]
    num_anchors = build_kwargs["num_anchors"]
    input_num_classes = build_kwargs["input_num_classes"]
    max_detections = build_kwargs["max_detections"]

    tvm_tirx.ir.assert_structural_equal(
        mod["main"].params[1].ty,
        relax.TensorType((batch_size, num_anchors, input_num_classes), "float32"),
    )
    tvm_tirx.ir.assert_structural_equal(
        mod["main"].ret_ty,
        relax.TupleType(
            [
                relax.TensorType((batch_size, max_detections, 4), "float32"),
                relax.TensorType((batch_size, max_detections), "float32"),
                relax.TensorType((batch_size, max_detections), "float32"),
                relax.TensorType((batch_size,), "float32"),
            ]
        ),
    )


def test_detection_postprocess_removes_background_without_softmax():
    """TFLite scores are probabilities; remove its optional background class exactly once."""
    mod = _build_detection_postprocess_mod(
        num_classes=2,
        input_num_classes=3,
        max_detections=2,
        detections_per_class=2,
        batch_size=1,
    )
    multibox_calls = []

    def _visit(expr):
        if isinstance(expr, relax.Call) and expr.op == tvm_tirx.ir.Op.get(
            "relax.vision.multibox_transform_loc"
        ):
            multibox_calls.append(expr)

    relax.analysis.post_order_visit(mod["main"].body, _visit)
    assert len(multibox_calls) == 1
    assert not multibox_calls[0].attrs.apply_softmax
    assert multibox_calls[0].attrs.keep_background
    tvm_tirx.ir.assert_structural_equal(
        multibox_calls[0].args[0].ty,
        relax.TensorType((1, 2, 4), "float32"),
    )


def test_detection_postprocess_rejects_invalid_class_count():
    with pytest.raises(ValueError, match=r"num_classes \+ 1"):
        _build_detection_postprocess_mod(num_classes=2, input_num_classes=5)


def _make_resize_expected(
    input_shape, output_size, method, coordinate_transformation_mode, rounding_method
):
    """Build an Expected IRModule programmatically to avoid TVMScript variable scope limitations."""
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorType(input_shape, "float32"))
    with bb.function("main", [x]):
        with bb.dataflow():
            gv = bb.emit_output(
                relax.op.image.resize2d(
                    x,
                    size=relax.ShapeExpr([output_size[0], output_size[1]]),
                    roi=[0.0, 0.0, 0.0, 0.0],
                    layout="NHWC",
                    method=method,
                    coordinate_transformation_mode=coordinate_transformation_mode,
                    rounding_method=rounding_method,
                    cubic_alpha=-0.75,
                    cubic_exclude=0,
                    extrapolation_value=0.0,
                )
            )
        bb.emit_func_output(gv)
    mod = bb.get()
    mod["main"] = mod["main"].with_attr("num_input", 1)
    return mod


@pytest.mark.parametrize(
    "input_shape, output_size, tf_op, coordinate_transformation_mode",
    [
        (
            (1, 4, 4, 1),
            [8, 8],
            lambda x: tf.image.resize(x, [8, 8], method="bilinear"),
            "half_pixel",
        ),
        (
            (1, 8, 8, 3),
            [4, 4],
            lambda x: tf.image.resize(x, [4, 4], method="bilinear"),
            "half_pixel",
        ),
        (
            (1, 4, 4, 1),
            [7, 7],
            lambda x: tf.compat.v1.image.resize_bilinear(x, [7, 7], align_corners=True),
            "align_corners",
        ),
        (
            (1, 4, 4, 2),
            [8, 8],
            lambda x: tf.compat.v1.image.resize_bilinear(x, [8, 8], half_pixel_centers=True),
            "half_pixel",
        ),
        (
            (2, 6, 6, 16),
            [12, 12],
            lambda x: tf.image.resize(x, [12, 12], method="bilinear"),
            "half_pixel",
        ),
        (
            (1, 5, 5, 3),
            [5, 5],
            lambda x: tf.image.resize(x, [5, 5], method="bilinear"),
            "half_pixel",
        ),
        (
            (1, 4, 8, 1),
            [8, 16],
            lambda x: tf.image.resize(x, [8, 16], method="bilinear"),
            "half_pixel",
        ),
    ],
)
def test_resize_bilinear(input_shape, output_size, tf_op, coordinate_transformation_mode):
    class ResizeBilinear(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32)])
        def func(self, x):
            return tf_op(x)

    expected = _make_resize_expected(
        input_shape, output_size, "linear", coordinate_transformation_mode, ""
    )
    verify(ResizeBilinear, expected)


@pytest.mark.parametrize(
    "input_shape, output_size, tf_op, coordinate_transformation_mode, rounding_method",
    [
        (
            (1, 2, 2, 1),
            [4, 4],
            lambda x: tf.image.resize(x, [4, 4], method="nearest"),
            "half_pixel",
            "round_prefer_ceil",
        ),
        (
            (1, 8, 8, 3),
            [4, 4],
            lambda x: tf.image.resize(x, [4, 4], method="nearest"),
            "half_pixel",
            "round_prefer_ceil",
        ),
        (
            (1, 4, 4, 1),
            [7, 7],
            lambda x: tf.compat.v1.image.resize_nearest_neighbor(x, [7, 7], align_corners=True),
            "align_corners",
            "",
        ),
        (
            (4, 3, 3, 8),
            [6, 6],
            lambda x: tf.image.resize(x, [6, 6], method="nearest"),
            "half_pixel",
            "round_prefer_ceil",
        ),
        (
            (1, 4, 8, 1),
            [8, 16],
            lambda x: tf.image.resize(x, [8, 16], method="nearest"),
            "half_pixel",
            "round_prefer_ceil",
        ),
        (
            (1, 3, 3, 2),
            [3, 3],
            lambda x: tf.image.resize(x, [3, 3], method="nearest"),
            "half_pixel",
            "round_prefer_ceil",
        ),
    ],
)
def test_resize_nearest_neighbor(
    input_shape, output_size, tf_op, coordinate_transformation_mode, rounding_method
):
    class ResizeNearest(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32)])
        def func(self, x):
            return tf_op(x)

    expected = _make_resize_expected(
        input_shape,
        output_size,
        "nearest_neighbor",
        coordinate_transformation_mode,
        rounding_method,
    )
    verify(ResizeNearest, expected)


def _make_reduce_expected(relax_op, input_shape, axes, keepdims, dtype):
    if axes is None:
        axes = list(range(len(input_shape)))
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorType(input_shape, dtype))
    with bb.function("main", [x]):
        with bb.dataflow():
            gv = bb.emit_output(relax_op(x, axis=axes, keepdims=keepdims))
        bb.emit_func_output(gv)
    mod = bb.get()
    mod["main"] = mod["main"].with_attr("num_input", 1)
    return mod


@pytest.mark.parametrize(
    "tf_op, relax_op",
    [
        (tf.reduce_sum, relax.op.sum),
        (tf.reduce_mean, relax.op.mean),
        (tf.reduce_max, relax.op.max),
        (tf.reduce_min, relax.op.min),
        (tf.reduce_prod, relax.op.prod),
    ],
)
@pytest.mark.parametrize(
    "input_shape, axes",
    [
        ((1, 8, 8, 3), 1),
        ((1, 8, 8, 3), [1, 2]),
        ((1, 8, 8, 3), -1),
        ((1, 8, 8, 3), None),
        ((30,), 0),
        ((2, 5, 2), [0, 2]),
    ],
)
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("dtype", [tf.float32, tf.int32])
def test_reduction_ops(tf_op, relax_op, input_shape, axes, keepdims, dtype):
    class ReduceModule(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=input_shape, dtype=dtype)])
        def func(self, x):
            return tf_op(x, axis=axes, keepdims=keepdims)

    relax_dtype = "float32" if dtype == tf.float32 else "int32"
    expected = _make_reduce_expected(relax_op, input_shape, axes, keepdims, relax_dtype)
    verify(ReduceModule, expected)


def _make_reduce_bool_expected(relax_op, input_shape, axes, keepdims):
    if axes is None:
        axes = list(range(len(input_shape)))
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorType(input_shape, "bool"))
    with bb.function("main", [x]):
        with bb.dataflow():
            cast_in = bb.emit(relax.op.astype(x, "int8"))
            reduced = bb.emit(relax_op(cast_in, axis=axes, keepdims=keepdims))
            gv = bb.emit_output(relax.op.astype(reduced, "bool"))
        bb.emit_func_output(gv)
    mod = bb.get()
    mod["main"] = mod["main"].with_attr("num_input", 1)
    return mod


@pytest.mark.parametrize(
    "tf_op, relax_op",
    [
        (tf.reduce_any, relax.op.max),
        (tf.reduce_all, relax.op.min),
    ],
)
@pytest.mark.parametrize(
    "input_shape, axes",
    [
        ((1, 8, 8, 3), 1),
        ((1, 8, 8, 3), [1, 2]),
        ((1, 8, 8, 3), -1),
        ((1, 8, 8, 3), None),
        ((30,), 0),
        ((2, 5, 2), [0, 2]),
    ],
)
@pytest.mark.parametrize("keepdims", [True, False])
def test_reduction_bool_ops(tf_op, relax_op, input_shape, axes, keepdims):
    class ReduceBoolModule(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.bool)])
        def func(self, x):
            return tf_op(x, axis=axes, keepdims=keepdims)

    expected = _make_reduce_bool_expected(relax_op, input_shape, axes, keepdims)
    verify(ReduceBoolModule, expected)

    # Regression guard: compile to catch a bool max/min lowering path.
    tvm.compile(expected, tvm_tirx.target.Target("llvm"))


def test_pad():
    class Pad(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(2, 3), dtype=tf.float32)])
        def func(self, x):
            return tf.pad(x, [[1, 1], [2, 2]])

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((4, 7), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((4, 7), dtype="float32") = R.nn.pad(
                    x, pad_width=[1, 1, 2, 2], pad_value=0.0, pad_mode="constant"
                )
                R.output(gv)
            return gv

    verify(Pad, Expected)


def test_pad_v2():
    class PadV2(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(2, 3), dtype=tf.float32)])
        def func(self, x):
            return tf.pad(x, [[1, 1], [2, 2]], constant_values=5.0)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((4, 7), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((4, 7), dtype="float32") = R.nn.pad(
                    x, pad_width=[1, 1, 2, 2], pad_value=5.0, pad_mode="constant"
                )
                R.output(gv)
            return gv

    verify(PadV2, Expected)


def test_mirror_pad():
    class MirrorPad(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(3, 4), dtype=tf.float32)])
        def func(self, x):
            return tf.pad(x, [[1, 1], [2, 2]], mode="REFLECT")

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((3, 4), dtype="float32")) -> R.Tensor((5, 8), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((5, 8), dtype="float32") = R.nn.pad(
                    x, pad_width=[1, 1, 2, 2], pad_value=0.0, pad_mode="reflect"
                )
                R.output(gv)
            return gv

    verify(MirrorPad, Expected)


def test_topk_v2():
    class TopKV2(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(5,), dtype=tf.float32)])
        def func(self, x):
            return tf.math.top_k(x, k=3).values

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((5,), dtype="float32")) -> R.Tensor((3,), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tuple(R.Tensor((3,), dtype="float32"), R.Tensor((3,), dtype="int32")) = (
                    R.topk(x, k=3, axis=-1, ret_type="both", largest=True, dtype="int32")
                )
                gv: R.Tensor((3,), dtype="float32") = lv[0]
                R.output(gv)
            return gv

    verify(TopKV2, Expected)


def test_one_hot():
    class OneHot(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(3,), dtype=tf.int32)])
        def func(self, x):
            return tf.one_hot(x, depth=4)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((3,), dtype="int32")) -> R.Tensor((3, 4), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((3, 4), dtype="float32") = R.one_hot(
                    x,
                    R.prim_value(T.float32(1.0)),
                    R.prim_value(T.float32(0.0)),
                    depth=4,
                    axis=-1,
                )
                R.output(gv)
            return gv

    verify(OneHot, Expected)


def test_select():
    class Select(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(2, 3), dtype=tf.bool),
                tf.TensorSpec(shape=(2, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(2, 3), dtype=tf.float32),
            ]
        )
        def func(self, cond, x, y):
            return tf.where(cond, x, y)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            cond: R.Tensor((2, 3), dtype="bool"),
            x: R.Tensor((2, 3), dtype="float32"),
            y: R.Tensor((2, 3), dtype="float32"),
        ) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 3})
            with R.dataflow():
                gv: R.Tensor((2, 3), dtype="float32") = R.where(cond, x, y)
                R.output(gv)
            return gv

    verify(Select, Expected)


def test_depth_to_space():
    class DepthToSpace(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 2, 4, 8), dtype=tf.float32)])
        def func(self, x):
            return tf.nn.depth_to_space(x, block_size=2)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((1, 2, 4, 8), dtype="float32"),
        ) -> R.Tensor((1, 4, 8, 2), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, 2, 4, 2, 2, 2), dtype="float32") = R.reshape(
                    x, R.shape([1, 2, 4, 2, 2, 2])
                )
                lv1: R.Tensor((1, 2, 2, 4, 2, 2), dtype="float32") = R.permute_dims(
                    lv, axes=[0, 1, 3, 2, 4, 5]
                )
                gv: R.Tensor((1, 4, 8, 2), dtype="float32") = R.reshape(lv1, R.shape([1, 4, 8, 2]))
                R.output(gv)
            return gv

    verify(DepthToSpace, Expected)


def test_space_to_depth():
    class SpaceToDepth(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 4, 4, 2), dtype=tf.float32)])
        def func(self, x):
            return tf.nn.space_to_depth(x, block_size=2)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((1, 4, 4, 2), dtype="float32"),
        ) -> R.Tensor((1, 2, 2, 8), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, 2, 2, 2, 2, 2), dtype="float32") = R.reshape(
                    x, R.shape([1, 2, 2, 2, 2, 2])
                )
                lv1: R.Tensor((1, 2, 2, 2, 2, 2), dtype="float32") = R.permute_dims(
                    lv, axes=[0, 1, 3, 2, 4, 5]
                )
                gv: R.Tensor((1, 2, 2, 8), dtype="float32") = R.reshape(lv1, R.shape([1, 2, 2, 8]))
                R.output(gv)
            return gv

    verify(SpaceToDepth, Expected)


@pytest.mark.parametrize(
    "input_shape, block_shape, paddings, expected_out_shape",
    [
        ((1, 2, 2, 1), [2, 2], [[0, 0], [0, 0]], (4, 1, 1, 1)),
        ((1, 2, 3, 1), [2, 2], [[0, 0], [1, 0]], (4, 1, 2, 1)),
    ],
)
def test_space_to_batch_nd(input_shape, block_shape, paddings, expected_out_shape):
    """SPACE_TO_BATCH_ND imports to Relax and preserves expected output shape."""

    class SpaceToBatchND(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32)])
        def func(self, x):
            return tf.space_to_batch_nd(
                x,
                tf.constant(block_shape, dtype=tf.int32),
                tf.constant(paddings, dtype=tf.int32),
            )

    if expected_out_shape == (4, 1, 1, 1):

        @I.ir_module
        class ExpectedSpaceToBatchNoPadding:
            @R.function
            def main(x: R.Tensor((1, 2, 2, 1), dtype="float32")) -> R.Tensor(
                (4, 1, 1, 1), dtype="float32"
            ):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    gv = R.call_dps_packed(
                        _ExternFunc("topi.nn.space_to_batch_nd"),
                        (
                            x,
                            R.shape([2, 2]),
                            R.shape([0, 0]),
                            R.shape([0, 0]),
                            R.prim_value(T.float64(0.0)),
                        ),
                        ty_args=[R.Tensor((4, 1, 1, 1), dtype="float32")],
                    )
                    R.output(gv)
                return gv

        expected = ExpectedSpaceToBatchNoPadding

    else:

        @I.ir_module
        class ExpectedSpaceToBatchWithPadding:
            @R.function
            def main(x: R.Tensor((1, 2, 3, 1), dtype="float32")) -> R.Tensor(
                (4, 1, 2, 1), dtype="float32"
            ):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    gv = R.call_dps_packed(
                        _ExternFunc("topi.nn.space_to_batch_nd"),
                        (
                            x,
                            R.shape([2, 2]),
                            R.shape([0, 1]),
                            R.shape([0, 0]),
                            R.prim_value(T.float64(0.0)),
                        ),
                        ty_args=[R.Tensor((4, 1, 2, 1), dtype="float32")],
                    )
                    R.output(gv)
                return gv

        expected = ExpectedSpaceToBatchWithPadding

    verify(SpaceToBatchND, expected)


@pytest.mark.parametrize(
    "input_shape, block_shape, crops, expected_out_shape",
    [
        ((4, 1, 1, 1), [2, 2], [[0, 0], [0, 0]], (1, 2, 2, 1)),
        ((4, 1, 2, 1), [2, 2], [[0, 0], [1, 0]], (1, 2, 3, 1)),
    ],
)
def test_batch_to_space_nd(input_shape, block_shape, crops, expected_out_shape):
    """BATCH_TO_SPACE_ND imports to Relax and preserves expected output shape."""

    class BatchToSpaceND(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32)])
        def func(self, x):
            return tf.raw_ops.BatchToSpaceND(
                input=x,
                block_shape=tf.constant(block_shape, dtype=tf.int32),
                crops=tf.constant(crops, dtype=tf.int32),
            )

    if expected_out_shape == (1, 2, 2, 1):

        @I.ir_module
        class ExpectedBatchToSpaceNoCrop:
            @R.function
            def main(x: R.Tensor((4, 1, 1, 1), dtype="float32")) -> R.Tensor(
                (1, 2, 2, 1), dtype="float32"
            ):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    gv = R.call_dps_packed(
                        _ExternFunc("topi.nn.batch_to_space_nd"),
                        (x, R.shape([2, 2]), R.shape([0, 0]), R.shape([0, 0])),
                        ty_args=[R.Tensor((1, 2, 2, 1), dtype="float32")],
                    )
                    R.output(gv)
                return gv

        expected = ExpectedBatchToSpaceNoCrop

    else:

        @I.ir_module
        class ExpectedBatchToSpaceWithCrop:
            @R.function
            def main(x: R.Tensor((4, 1, 2, 1), dtype="float32")) -> R.Tensor(
                (1, 2, 3, 1), dtype="float32"
            ):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    gv = R.call_dps_packed(
                        _ExternFunc("topi.nn.batch_to_space_nd"),
                        (x, R.shape([2, 2]), R.shape([0, 1]), R.shape([0, 0])),
                        ty_args=[R.Tensor((1, 2, 3, 1), dtype="float32")],
                    )
                    R.output(gv)
                return gv

        expected = ExpectedBatchToSpaceWithCrop

    verify(BatchToSpaceND, expected)


def test_leaky_relu():
    class LeakyReLU(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            return tf.nn.leaky_relu(x, alpha=0.2)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 30), dtype="float32")) -> R.Tensor((1, 30), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 30), dtype="float32") = R.nn.leakyrelu(
                    x, alpha=0.20000000298023224
                )
                R.output(gv)
            return gv

    verify(LeakyReLU, Expected)


def test_hard_swish():
    class HardSwish(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            return x * tf.nn.relu6(x + 3) / 6

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 30), dtype="float32")) -> R.Tensor((1, 30), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, 30), dtype="float32") = R.add(x, R.const(3.0, dtype="float32"))
                lv1: R.Tensor((1, 30), dtype="float32") = R.clip(
                    lv, R.prim_value(T.float64(0.0)), R.prim_value(T.float64(6.0))
                )
                lv2: R.Tensor((1, 30), dtype="float32") = R.multiply(x, lv1)
                gv: R.Tensor((1, 30), dtype="float32") = R.divide(
                    lv2, R.const(6.0, dtype="float32")
                )
                R.output(gv)
            return gv

    verify(HardSwish, Expected)


def _build_relu_0_to_1_model():
    """Build a minimal TFLite RELU_0_TO_1 model."""
    builder = flatbuffers.Builder(1024)
    builtin_op = _get_builtin_operator("RELU_0_TO_1")
    op_code = _build_operator_code(builder, builtin_op)
    tensors = [
        _build_tensor(builder, 0, [2, 2]),
        _build_tensor(builder, 1, [2, 2]),
    ]
    op = _build_operator(builder, 0, [0], [1])
    subgraph = _build_subgraph(builder, tensors=tensors, operators=[op], inputs=[0], outputs=[1])
    return _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=[op_code],
        buffers=[_build_buffer(builder), _build_buffer(builder)],
    )


def test_relu_0_to_1():
    """RELU_0_TO_1 lowers to clip(0, 1)."""
    mod = _load_model_from_buffer(_build_relu_0_to_1_model())

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 2), dtype="float32")) -> R.Tensor((2, 2), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="float32") = R.clip(x, min=0, max=1)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_relu_n1_to_1():
    class ReLU_N1_to_1(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            return tf.clip_by_value(x, -1.0, 1.0)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 30), dtype="float32")) -> R.Tensor((1, 30), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 30), dtype="float32") = R.clip(x, min=-1, max=1)
                R.output(gv)
            return gv

    verify(ReLU_N1_to_1, Expected)


def _build_fake_quant_model(*, narrow_range, num_bits=8, min_value=-1.0, max_value=1.0):
    """Build a minimal TFLite FAKE_QUANT model."""
    fake_quant_options = _get_tflite_schema_module("FakeQuantOptions")
    builder = flatbuffers.Builder(1024)
    builtin_op = _get_builtin_operator("FAKE_QUANT")
    op_code = _build_operator_code(builder, builtin_op)

    fake_quant_options.FakeQuantOptionsStart(builder)
    fake_quant_options.FakeQuantOptionsAddMin(builder, min_value)
    fake_quant_options.FakeQuantOptionsAddMax(builder, max_value)
    fake_quant_options.FakeQuantOptionsAddNumBits(builder, num_bits)
    fake_quant_options.FakeQuantOptionsAddNarrowRange(builder, narrow_range)
    options = fake_quant_options.FakeQuantOptionsEnd(builder)

    tensors = [
        _build_tensor(builder, 0, [4]),
        _build_tensor(builder, 1, [4]),
    ]
    op = _build_operator(
        builder,
        0,
        [0],
        [1],
        builtin_options_type=_get_builtin_options_type("FakeQuantOptions"),
        builtin_options=options,
    )
    subgraph = _build_subgraph(builder, tensors=tensors, operators=[op], inputs=[0], outputs=[1])
    return _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=[op_code],
        buffers=[_build_buffer(builder), _build_buffer(builder)],
    )


def _fake_quant_reference(data, *, narrow_range, num_bits=8, min_value=-1.0, max_value=1.0):
    quant_min = 1 if narrow_range else 0
    quant_max = (1 << num_bits) - 1
    scale = (max_value - min_value) / (quant_max - quant_min)
    zero_point_from_min = quant_min - min_value / scale
    if zero_point_from_min <= quant_min:
        nudged_zero_point = quant_min
    elif zero_point_from_min >= quant_max:
        nudged_zero_point = quant_max
    else:
        nudged_zero_point = round(zero_point_from_min)
    nudged_min = (quant_min - nudged_zero_point) * scale
    nudged_max = (quant_max - nudged_zero_point) * scale
    clamped = np.clip(data, nudged_min, nudged_max)
    return np.floor((clamped - nudged_min) / scale + 0.5) * scale + nudged_min


def test_fake_quant_narrow_range_vector():
    """FAKE_QUANT supports narrow_range on vector inputs."""
    mod = _load_model_from_buffer(_build_fake_quant_model(narrow_range=True))
    data = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32)
    output = _run_module(mod, data)
    expected = _fake_quant_reference(data, narrow_range=True).astype(np.float32)
    np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)


def test_prelu_basic():
    alpha = np.linspace(0.1, 0.3, 30, dtype=np.float32)
    alpha_init = tf.keras.initializers.Constant(alpha)
    prelu = tf.keras.layers.PReLU(alpha_initializer=alpha_init)

    class TfInput(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            return prelu(x)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 30), dtype="float32")) -> R.Tensor((1, 30), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, 30), dtype="float32") = R.broadcast_to(
                    R.const(alpha), R.shape([1, 30])
                )
                lv1: R.Tensor((30,), dtype="float32") = R.reshape(x, R.shape([30]))
                lv2: R.Tensor((30,), dtype="float32") = R.reshape(lv, R.shape([30]))
                lv3: R.Tensor((30,), dtype="float32") = R.nn.prelu(lv1, lv2, axis=0)
                gv: R.Tensor((1, 30), dtype="float32") = R.reshape(lv3, R.shape([1, 30]))
                R.output(gv)
            return gv

    verify(TfInput, Expected)


@pytest.mark.parametrize(
    "shared_axes",
    [
        pytest.param([1, 2], id="channelwise_shared_axes"),
        pytest.param([1, 2, 3], id="scalar_shared_axes"),
        pytest.param(None, id="elementwise_no_shared_axes"),
    ],
)
def test_prelu(shared_axes):
    inputs = tf.keras.Input(shape=(4, 4, 3), batch_size=1, dtype=tf.float32)
    prelu_kwargs = {
        "alpha_initializer": tf.initializers.constant(0.25),
    }
    if shared_axes is not None:
        prelu_kwargs["shared_axes"] = shared_axes
    outputs = tf.keras.layers.PReLU(**prelu_kwargs)(inputs)
    keras_model = tf.keras.Model(inputs, outputs)

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model_buf = converter.convert()
    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)

    mod = from_tflite(tflite_model)
    mod["main"] = mod["main"].without_attr("params")

    if shared_axes == [1, 2]:
        alpha_const = np.full((1, 1, 3), 0.25, dtype=np.float32)
    elif shared_axes == [1, 2, 3]:
        alpha_const = np.full((1, 1, 1), 0.25, dtype=np.float32)
    else:
        alpha_const = np.full((4, 4, 3), 0.25, dtype=np.float32)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 4, 4, 3), dtype="float32")) -> R.Tensor(
            (1, 4, 4, 3), dtype="float32"
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, 4, 4, 3), dtype="float32") = R.broadcast_to(
                    R.const(alpha_const), R.shape([1, 4, 4, 3])
                )
                lv1: R.Tensor((48,), dtype="float32") = R.reshape(x, R.shape([48]))
                lv2: R.Tensor((48,), dtype="float32") = R.reshape(lv, R.shape([48]))
                lv3: R.Tensor((48,), dtype="float32") = R.nn.prelu(lv1, lv2, axis=0)
                gv: R.Tensor((1, 4, 4, 3), dtype="float32") = R.reshape(lv3, R.shape([1, 4, 4, 3]))
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_matrix_diag():
    """Test TFLite MATRIX_DIAG operator."""

    class MatrixDiag(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(3,), dtype=tf.float32)])
        def func(self, diagonal):
            return tf.raw_ops.MatrixDiag(diagonal=diagonal)

    @I.ir_module
    class Expected:
        @R.function
        def main(diagonal: R.Tensor((3,), dtype="float32")) -> R.Tensor((3, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((3, 3), dtype="float32") = R.zeros(R.shape([3, 3]), dtype="float32")
                gv = R.call_dps_packed(
                    _ExternFunc("topi.matrix_set_diag"),
                    (
                        lv,
                        diagonal,
                        R.const(0, "int32"),
                        R.const(0, "int32"),
                        R.const(False, "bool"),
                        R.const(False, "bool"),
                    ),
                    ty_args=[R.Tensor((3, 3), dtype="float32")],
                )
                R.output(gv)
            return gv

    verify(MatrixDiag, Expected)


def test_matrix_set_diag():
    """Test TFLite MATRIX_SET_DIAG operator."""

    class MatrixSetDiag(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(3, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(3,), dtype=tf.float32),
            ]
        )
        def func(self, input, diagonal):
            return tf.raw_ops.MatrixSetDiag(input=input, diagonal=diagonal)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            input: R.Tensor((3, 3), dtype="float32"),
            diagonal: R.Tensor((3,), dtype="float32"),
        ) -> R.Tensor((3, 3), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                gv = R.call_dps_packed(
                    _ExternFunc("topi.matrix_set_diag"),
                    (
                        input,
                        diagonal,
                        R.const(0, "int32"),
                        R.const(0, "int32"),
                        R.const(False, "bool"),
                        R.const(False, "bool"),
                    ),
                    ty_args=[R.Tensor((3, 3), dtype="float32")],
                )
                R.output(gv)
            return gv

    verify(MatrixSetDiag, Expected)


def test_sparse_to_dense():
    """Test TFLite SPARSE_TO_DENSE operator."""

    class SparseToDense(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(2,), dtype=tf.int32),
                tf.TensorSpec(shape=(2,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
            ]
        )
        def func(self, indices, values, default_value):
            # output_shape is provided as a constant, not an input
            return tf.raw_ops.SparseToDense(
                sparse_indices=indices,
                output_shape=tf.constant([3], dtype=tf.int32),
                sparse_values=values,
                default_value=default_value,
            )

    @I.ir_module
    class Expected:
        @R.function
        def main(
            indices: R.Tensor((2,), dtype="int32"),
            values: R.Tensor((2,), dtype="float32"),
            default_value: R.Tensor((), dtype="float32"),
        ) -> R.Tensor((3,), dtype="float32"):
            R.func_attr({"num_input": 3})
            with R.dataflow():
                gv = R.call_dps_packed(
                    _ExternFunc("topi.sparse_to_dense"),
                    (indices, R.const([3], "int32"), values, default_value),
                    ty_args=[R.Tensor((3,), dtype="float32")],
                )
                R.output(gv)
            return gv

    verify(SparseToDense, Expected)


# DENSIFY operator tests
# DENSIFY converts sparse weight tensors to dense at conversion time (not runtime).
# Since TensorFlow does not provide an API to create sparse TFLite models,
# we manually build them using the flatbuffers API.


# Import schema helpers explicitly. CI's generated tflite package does not
# reliably re-export these builder helpers and enums at the package top-level.
def _get_tflite_schema_module(module_name):
    return __import__(f"tflite.{module_name}", fromlist=[module_name])


def _get_tflite_schema_enum(enum_name):
    return getattr(_get_tflite_schema_module(enum_name), enum_name)


_tfl_add_options = _get_tflite_schema_module("AddOptions")
_tfl_buffer = _get_tflite_schema_module("Buffer")
_tfl_concatenation_options = _get_tflite_schema_module("ConcatenationOptions")
_tfl_conv2d_options = _get_tflite_schema_module("Conv2DOptions")
_tfl_depthwise_conv2d_options = _get_tflite_schema_module("DepthwiseConv2DOptions")
_tfl_dilate_options = _get_tflite_schema_module("DilateOptions")
_tfl_reshape_options = _get_tflite_schema_module("ReshapeOptions")
_tfl_transpose_conv_options = _get_tflite_schema_module("TransposeConvOptions")

# ── StableHLO BuiltinOptions2 schema modules ────────────────────────────
_tfl_stablehlo_concat_opts = _get_tflite_schema_module("StablehloConcatenateOptions")
_tfl_stablehlo_bcast_opts = _get_tflite_schema_module("StablehloBroadcastInDimOptions")
_tfl_stablehlo_composite_opts = _get_tflite_schema_module("StableHLOCompositeOptions")
_tfl_stablehlo_conv_opts = _get_tflite_schema_module("StablehloConvolutionOptions")
_tfl_stablehlo_custom_call_opts = _get_tflite_schema_module("StablehloCustomCallOptions")
_tfl_stablehlo_dot_opts = _get_tflite_schema_module("StablehloDotGeneralOptions")
_tfl_stablehlo_iota_opts = _get_tflite_schema_module("StablehloIotaOptions")
_tfl_stablehlo_compare_opts = _get_tflite_schema_module("StablehloCompareOptions")
_tfl_stablehlo_comp_dir = _get_tflite_schema_module("StablehloComparisonDirection")
_tfl_stablehlo_comp_type = _get_tflite_schema_module("StablehloComparisonType")
_tfl_stablehlo_pad_opts = _get_tflite_schema_module("StablehloPadOptions")
_tfl_stablehlo_dyn_slice_opts = _get_tflite_schema_module("StablehloDynamicSliceOptions")
_tfl_stablehlo_gather_opts = _get_tflite_schema_module("StablehloGatherOptions")
_tfl_stablehlo_reduce_opts = _get_tflite_schema_module("StablehloReduceOptions")
_tfl_stablehlo_reduce_window_opts = _get_tflite_schema_module("StablehloReduceWindowOptions")
_tfl_stablehlo_scatter_opts = _get_tflite_schema_module("StablehloScatterOptions")
_tfl_stablehlo_slice_opts = _get_tflite_schema_module("StablehloSliceOptions")
_tfl_stablehlo_sort_opts = _get_tflite_schema_module("StablehloSortOptions")
_tfl_stablehlo_transpose_opts = _get_tflite_schema_module("StablehloTransposeOptions")
_tfl_stablehlo_while_opts = _get_tflite_schema_module("StablehloWhileOptions")
_tfl_stablehlo_rng_opts = _get_tflite_schema_module("StablehloRngBitGeneratorOptions")
_tfl_call_options = _get_tflite_schema_module("CallOptions")
_tfl_call_once_options = _get_tflite_schema_module("CallOnceOptions")
_tfl_dimension_metadata = _get_tflite_schema_module("DimensionMetadata")
_tfl_fully_connected_options = _get_tflite_schema_module("FullyConnectedOptions")
_tfl_if_options = _get_tflite_schema_module("IfOptions")
_tfl_int32_vector = _get_tflite_schema_module("Int32Vector")
_tfl_model = _get_tflite_schema_module("Model")
_tfl_operator = _get_tflite_schema_module("Operator")
_tfl_operator_code = _get_tflite_schema_module("OperatorCode")
_tfl_pool2d_options = _get_tflite_schema_module("Pool2DOptions")
_tfl_quantization_parameters = _get_tflite_schema_module("QuantizationParameters")
_tfl_sparsity_parameters = _get_tflite_schema_module("SparsityParameters")
_tfl_subgraph = _get_tflite_schema_module("SubGraph")
_tfl_tensor = _get_tflite_schema_module("Tensor")
_tfl_reverse_sequence_options = _get_tflite_schema_module("ReverseSequenceOptions")
_tfl_squeeze_options = _get_tflite_schema_module("SqueezeOptions")
_tfl_unpack_options = _get_tflite_schema_module("UnpackOptions")
_tfl_while_options = _get_tflite_schema_module("WhileOptions")
_tfl_zeros_like_options = _get_tflite_schema_module("ZerosLikeOptions")

_tfl_builtin_operator = _get_tflite_schema_enum("BuiltinOperator")
_tfl_builtin_options = _get_tflite_schema_enum("BuiltinOptions")
_tfl_builtin_options2 = _get_tflite_schema_enum("BuiltinOptions2")
_tfl_activation_fn = _get_tflite_schema_enum("ActivationFunctionType")
_tfl_dimension_type = _get_tflite_schema_enum("DimensionType")
_tfl_fc_weights_format = _get_tflite_schema_enum("FullyConnectedOptionsWeightsFormat")
_tfl_padding = _get_tflite_schema_enum("Padding")
_tfl_sparse_index_vector = _get_tflite_schema_enum("SparseIndexVector")
_tfl_tensor_type = _get_tflite_schema_enum("TensorType")
_tfl_rng_algorithm = _get_tflite_schema_enum("RngAlgorithm")

_tfl_lstm_options = _get_tflite_schema_module("LSTMOptions")
_tfl_sequence_rnn_options = _get_tflite_schema_module("SequenceRNNOptions")
_tfl_svdf_options = _get_tflite_schema_module("SVDFOptions")
_tfl_unidirectional_sequence_lstm_options = _get_tflite_schema_module(
    "UnidirectionalSequenceLSTMOptions"
)
_tfl_bidirectional_sequence_rnn_options = _get_tflite_schema_module(
    "BidirectionalSequenceRNNOptions"
)
_tfl_bidirectional_sequence_lstm_options = _get_tflite_schema_module(
    "BidirectionalSequenceLSTMOptions"
)

_DENSIFY_TEST_VALUES = np.array([1.0, 2.0], dtype=np.float32)
_DENSIFY_TEST_DENSE = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
_DENSIFY_ROW_PTRS = [0, 1, 2]
_DENSIFY_COL_INDICES = [0, 1]
_DENSIFY_CONV_KERNEL_DENSE_HWIO = _DENSIFY_TEST_DENSE.reshape(2, 2, 1, 1)
_DENSIFY_FC_WEIGHT_VALUES = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
_DENSIFY_FC_WEIGHT_DENSE_OI = np.diag(_DENSIFY_FC_WEIGHT_VALUES).astype(np.float32)
_DENSIFY_FC_ROW_PTRS = [0, 1, 2, 3, 4]
_DENSIFY_FC_COL_INDICES = [0, 1, 2, 3]


def _tflite_int32_vector(builder, start_vector_fn, values):
    start_vector_fn(builder, len(values))
    for value in reversed(values):
        builder.PrependInt32(value)
    return builder.EndVector()


def _tflite_int64_vector(builder, start_vector_fn, values):
    start_vector_fn(builder, len(values))
    for value in reversed(values):
        builder.PrependInt64(value)
    return builder.EndVector()


def _tflite_bool_vector(builder, start_vector_fn, values):
    start_vector_fn(builder, len(values))
    for value in reversed(values):
        builder.PrependBool(value)
    return builder.EndVector()


def _tflite_float32_vector(builder, start_vector_fn, values):
    start_vector_fn(builder, len(values))
    for value in reversed(values):
        builder.PrependFloat32(value)
    return builder.EndVector()


def _tflite_offset_vector(builder, start_vector_fn, offsets):
    start_vector_fn(builder, len(offsets))
    for offset in reversed(offsets):
        builder.PrependUOffsetTRelative(offset)
    return builder.EndVector()


def _tflite_byte_vector(builder, data):
    _tfl_buffer.BufferStartDataVector(builder, len(data))
    for byte in reversed(data):
        builder.PrependByte(byte)
    return builder.EndVector()


def _tflite_int32_table(builder, values):
    # Build the values vector directly without relying on version-specific
    # helper Int32VectorStartValuesVector, which is absent in older
    # tflite package versions used in CI.
    builder.StartVector(4, len(values), 4)
    for value in reversed(values):
        builder.PrependInt32(value)
    values_vec = builder.EndVector()
    _tfl_int32_vector.Int32VectorStart(builder)
    _tfl_int32_vector.Int32VectorAddValues(builder, values_vec)
    return _tfl_int32_vector.Int32VectorEnd(builder)


def _tflite_shape(builder, shape):
    return _tflite_int32_vector(builder, _tfl_tensor.TensorStartShapeVector, shape)


def _build_tensor(builder, buffer_idx, shape, sparsity=None, tensor_type=None, quantization=None):
    """Helper to build a TFLite tensor."""
    if tensor_type is None:
        tensor_type = _tfl_tensor_type.FLOAT32
    shape_vec = _tflite_shape(builder, shape)
    _tfl_tensor.TensorStart(builder)
    _tfl_tensor.TensorAddBuffer(builder, buffer_idx)
    _tfl_tensor.TensorAddHasRank(builder, True)
    _tfl_tensor.TensorAddIsVariable(builder, False)
    _tfl_tensor.TensorAddShape(builder, shape_vec)
    if sparsity is not None:
        _tfl_tensor.TensorAddSparsity(builder, sparsity)
    if quantization is not None:
        _tfl_tensor.TensorAddQuantization(builder, quantization)
    _tfl_tensor.TensorAddType(builder, tensor_type)
    return _tfl_tensor.TensorEnd(builder)


def _build_buffer(builder, data=None):
    # Build the data vector before starting the Buffer table to avoid
    # flatbuffers IsNestedError (vectors cannot be created inside tables).
    data_offset = None
    if data is not None:
        data_offset = _tflite_byte_vector(builder, data)
    _tfl_buffer.BufferStart(builder)
    if data_offset is not None:
        _tfl_buffer.BufferAddData(builder, data_offset)
    return _tfl_buffer.BufferEnd(builder)


def _build_quantization_parameters(builder, *, scale, zero_point, quantized_dimension):
    scale_vec = _tflite_float32_vector(
        builder, _tfl_quantization_parameters.QuantizationParametersStartScaleVector, scale
    )
    zero_point_vec = _tflite_int64_vector(
        builder,
        _tfl_quantization_parameters.QuantizationParametersStartZeroPointVector,
        zero_point,
    )
    _tfl_quantization_parameters.QuantizationParametersStart(builder)
    _tfl_quantization_parameters.QuantizationParametersAddScale(builder, scale_vec)
    _tfl_quantization_parameters.QuantizationParametersAddZeroPoint(builder, zero_point_vec)
    _tfl_quantization_parameters.QuantizationParametersAddQuantizedDimension(
        builder, quantized_dimension
    )
    return _tfl_quantization_parameters.QuantizationParametersEnd(builder)


def _build_operator(
    builder,
    opcode_index,
    inputs,
    outputs,
    builtin_options_type=None,
    builtin_options=None,
    builtin_options2_type=None,
    builtin_options2=None,
):
    inputs_vec = _tflite_int32_vector(builder, _tfl_operator.OperatorStartInputsVector, inputs)
    outputs_vec = _tflite_int32_vector(builder, _tfl_operator.OperatorStartOutputsVector, outputs)
    _tfl_operator.OperatorStart(builder)
    _tfl_operator.OperatorAddOpcodeIndex(builder, opcode_index)
    _tfl_operator.OperatorAddInputs(builder, inputs_vec)
    _tfl_operator.OperatorAddOutputs(builder, outputs_vec)
    if builtin_options_type is not None:
        _tfl_operator.OperatorAddBuiltinOptionsType(builder, builtin_options_type)
    if builtin_options is not None:
        _tfl_operator.OperatorAddBuiltinOptions(builder, builtin_options)
    if builtin_options2_type is not None:
        _tfl_operator.OperatorAddBuiltinOptions2Type(builder, builtin_options2_type)
    if builtin_options2 is not None:
        _tfl_operator.OperatorAddBuiltinOptions2(builder, builtin_options2)
    return _tfl_operator.OperatorEnd(builder)


def _build_operator_code(builder, builtin_op):
    # deprecated_builtin_code is int8 (max 127). Ops past that write 127 as a
    # placeholder and use the full builtin_code field.
    deprecated_code = builtin_op if builtin_op < 127 else 127
    _tfl_operator_code.OperatorCodeStart(builder)
    _tfl_operator_code.OperatorCodeAddDeprecatedBuiltinCode(builder, deprecated_code)
    _tfl_operator_code.OperatorCodeAddBuiltinCode(builder, builtin_op)
    _tfl_operator_code.OperatorCodeAddVersion(builder, 1)
    return _tfl_operator_code.OperatorCodeEnd(builder)


def _build_subgraph(builder, *, tensors, operators, inputs, outputs):
    tensors_vec = _tflite_offset_vector(builder, _tfl_subgraph.SubGraphStartTensorsVector, tensors)
    operators_vec = _tflite_offset_vector(
        builder, _tfl_subgraph.SubGraphStartOperatorsVector, operators
    )
    inputs_vec = _tflite_int32_vector(builder, _tfl_subgraph.SubGraphStartInputsVector, inputs)
    outputs_vec = _tflite_int32_vector(builder, _tfl_subgraph.SubGraphStartOutputsVector, outputs)

    _tfl_subgraph.SubGraphStart(builder)
    _tfl_subgraph.SubGraphAddTensors(builder, tensors_vec)
    _tfl_subgraph.SubGraphAddOperators(builder, operators_vec)
    _tfl_subgraph.SubGraphAddInputs(builder, inputs_vec)
    _tfl_subgraph.SubGraphAddOutputs(builder, outputs_vec)
    return _tfl_subgraph.SubGraphEnd(builder)


def _finish_tflite_model(builder, *, subgraph, operator_codes, buffers, extra_subgraphs=None):
    all_subgraphs = [subgraph] + (extra_subgraphs or [])
    buffers_vec = _tflite_offset_vector(builder, _tfl_model.ModelStartBuffersVector, buffers)
    opcodes_vec = _tflite_offset_vector(
        builder, _tfl_model.ModelStartOperatorCodesVector, operator_codes
    )
    subgraphs_vec = _tflite_offset_vector(
        builder, _tfl_model.ModelStartSubgraphsVector, all_subgraphs
    )

    _tfl_model.ModelStart(builder)
    _tfl_model.ModelAddBuffers(builder, buffers_vec)
    _tfl_model.ModelAddSubgraphs(builder, subgraphs_vec)
    _tfl_model.ModelAddOperatorCodes(builder, opcodes_vec)
    _tfl_model.ModelAddVersion(builder, 3)
    model = _tfl_model.ModelEnd(builder)

    builder.Finish(model, b"TFL3")
    return bytes(builder.Output())


def _build_call_options(builder, subgraph_index):
    _tfl_call_options.CallOptionsStart(builder)
    _tfl_call_options.CallOptionsAddSubgraph(builder, subgraph_index)
    return _tfl_call_options.CallOptionsEnd(builder)


def _build_if_options(builder, then_subgraph_index, else_subgraph_index):
    _tfl_if_options.IfOptionsStart(builder)
    _tfl_if_options.IfOptionsAddThenSubgraphIndex(builder, then_subgraph_index)
    _tfl_if_options.IfOptionsAddElseSubgraphIndex(builder, else_subgraph_index)
    return _tfl_if_options.IfOptionsEnd(builder)


def _build_while_options(builder, cond_subgraph_index, body_subgraph_index):
    _tfl_while_options.WhileOptionsStart(builder)
    _tfl_while_options.WhileOptionsAddCondSubgraphIndex(builder, cond_subgraph_index)
    _tfl_while_options.WhileOptionsAddBodySubgraphIndex(builder, body_subgraph_index)
    return _tfl_while_options.WhileOptionsEnd(builder)


def _build_stablehlo_while_options(builder, cond_subgraph_index, body_subgraph_index):
    _tfl_stablehlo_while_opts.StablehloWhileOptionsStart(builder)
    _tfl_stablehlo_while_opts.StablehloWhileOptionsAddCondSubgraphIndex(
        builder, cond_subgraph_index
    )
    _tfl_stablehlo_while_opts.StablehloWhileOptionsAddBodySubgraphIndex(
        builder, body_subgraph_index
    )
    return _tfl_stablehlo_while_opts.StablehloWhileOptionsEnd(builder)


def _build_call_once_options(builder, init_subgraph_index):
    _tfl_call_once_options.CallOnceOptionsStart(builder)
    _tfl_call_once_options.CallOnceOptionsAddInitSubgraphIndex(builder, init_subgraph_index)
    return _tfl_call_once_options.CallOnceOptionsEnd(builder)


def _build_squeeze_options(builder, squeeze_dims):
    squeeze_dims_vec = _tflite_int32_vector(
        builder,
        _tfl_squeeze_options.SqueezeOptionsStartSqueezeDimsVector,
        squeeze_dims,
    )
    _tfl_squeeze_options.SqueezeOptionsStart(builder)
    _tfl_squeeze_options.SqueezeOptionsAddSqueezeDims(builder, squeeze_dims_vec)
    return _tfl_squeeze_options.SqueezeOptionsEnd(builder)


def _build_reverse_sequence_options(builder, seq_dim, batch_dim):
    _tfl_reverse_sequence_options.ReverseSequenceOptionsStart(builder)
    _tfl_reverse_sequence_options.ReverseSequenceOptionsAddSeqDim(builder, seq_dim)
    _tfl_reverse_sequence_options.ReverseSequenceOptionsAddBatchDim(builder, batch_dim)
    return _tfl_reverse_sequence_options.ReverseSequenceOptionsEnd(builder)


def _build_unpack_options(builder, num, axis):
    _tfl_unpack_options.UnpackOptionsStart(builder)
    _tfl_unpack_options.UnpackOptionsAddNum(builder, num)
    _tfl_unpack_options.UnpackOptionsAddAxis(builder, axis)
    return _tfl_unpack_options.UnpackOptionsEnd(builder)


def _get_builtin_options_type(options_name):
    if not hasattr(_tfl_builtin_options, options_name):
        pytest.skip(f"TFLite schema does not provide BuiltinOptions.{options_name}")
    return getattr(_tfl_builtin_options, options_name)


def _get_resource_tensor_type():
    if not hasattr(_tfl_tensor_type, "RESOURCE"):
        pytest.skip("TFLite schema does not provide TensorType.RESOURCE")
    return getattr(_tfl_tensor_type, "RESOURCE")


def _get_string_tensor_type():
    if not hasattr(_tfl_tensor_type, "STRING"):
        pytest.skip("TFLite schema does not provide TensorType.STRING")
    return getattr(_tfl_tensor_type, "STRING")


def _build_tflite_string_buffer(values):
    encoded = [value.encode("utf-8") for value in values]
    offsets = []
    cursor = 4 * (len(encoded) + 2)
    for value in encoded:
        offsets.append(cursor)
        cursor += len(value)
    offsets.append(cursor)
    header = np.array([len(encoded), *offsets], dtype=np.int32).tobytes()
    return header + b"".join(encoded)


def _build_var_handle_options(builder, shared_name="resource_var", container=""):
    try:
        var_handle_options = _get_tflite_schema_module("VarHandleOptions")
    except ModuleNotFoundError:
        pytest.skip("TFLite schema does not provide VarHandleOptions")
    container_offset = builder.CreateString(container)
    shared_name_offset = builder.CreateString(shared_name)
    var_handle_options.VarHandleOptionsStart(builder)
    var_handle_options.VarHandleOptionsAddContainer(builder, container_offset)
    var_handle_options.VarHandleOptionsAddSharedName(builder, shared_name_offset)
    return var_handle_options.VarHandleOptionsEnd(builder)


def _build_empty_builtin_options(builder, options_name):
    try:
        options_module = _get_tflite_schema_module(options_name)
    except ModuleNotFoundError:
        pytest.skip(f"TFLite schema does not provide {options_name}")
    getattr(options_module, f"{options_name}Start")(builder)
    return getattr(options_module, f"{options_name}End")(builder)


def _build_hashtable_options(
    builder,
    table_id=0,
    key_dtype=None,
    value_dtype=None,
):
    try:
        hashtable_options = _get_tflite_schema_module("HashtableOptions")
    except ModuleNotFoundError:
        pytest.skip("TFLite schema does not provide HashtableOptions")

    key_dtype = _tfl_tensor_type.INT64 if key_dtype is None else key_dtype
    value_dtype = _get_string_tensor_type() if value_dtype is None else value_dtype
    hashtable_options.HashtableOptionsStart(builder)
    hashtable_options.HashtableOptionsAddTableId(builder, table_id)
    hashtable_options.HashtableOptionsAddKeyDtype(builder, key_dtype)
    hashtable_options.HashtableOptionsAddValueDtype(builder, value_dtype)
    return hashtable_options.HashtableOptionsEnd(builder)


def _build_embedding_lookup_sparse_options(builder, combiner):
    try:
        sparse_options = _get_tflite_schema_module("EmbeddingLookupSparseOptions")
    except ModuleNotFoundError:
        pytest.skip("TFLite schema does not provide EmbeddingLookupSparseOptions")

    sparse_options.EmbeddingLookupSparseOptionsStart(builder)
    sparse_options.EmbeddingLookupSparseOptionsAddCombiner(builder, combiner)
    return sparse_options.EmbeddingLookupSparseOptionsEnd(builder)


def _load_model_from_buffer(model_bytes):
    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(model_bytes, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(model_bytes, 0)
    mod = from_tflite(tflite_model)
    mod["main"] = mod["main"].without_attr("params")
    return mod


def _get_builtin_operator(builtin_name):
    if not hasattr(_tfl_builtin_operator, builtin_name):
        pytest.skip(f"TFLite schema does not provide BuiltinOperator.{builtin_name}")
    return getattr(_tfl_builtin_operator, builtin_name)


def test_broadcast_to_dynamic_shape():
    builder = flatbuffers.Builder(1024)
    op_code = _build_operator_code(builder, _get_builtin_operator("BROADCAST_TO"))
    tensors = [
        _build_tensor(builder, 0, [2, 2], tensor_type=_tfl_tensor_type.FLOAT32),
        _build_tensor(builder, 0, [3], tensor_type=_tfl_tensor_type.INT32),
        _build_tensor(builder, 0, [3, 2, 2], tensor_type=_tfl_tensor_type.FLOAT32),
    ]
    op = _build_operator(builder, 0, [0, 1], [2])
    subgraph = _build_subgraph(
        builder,
        tensors=tensors,
        operators=[op],
        inputs=[0, 1],
        outputs=[2],
    )
    model = _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=[op_code],
        buffers=[_build_buffer(builder)],
    )
    mod = _load_model_from_buffer(model)
    relax.analysis.check_well_formed(mod)

    shape_binding, result_binding = mod["main"].body.blocks[0].bindings
    assert not isinstance(shape_binding.var, relax.DataflowVar)
    tvm_tirx.ir.assert_structural_equal(result_binding.var.ty.shape, shape_binding.var)
    assert mod["main"].body.body.same_as(result_binding.var)


def _build_tflite_operator_marker_model(builtin_name):
    """Build a minimal model containing a TFLite marker builtin."""
    builder = flatbuffers.Builder(1024)
    builtin_op = _get_builtin_operator(builtin_name)
    op_code = _build_operator_code(builder, builtin_op)
    tensors = [
        _build_tensor(builder, 0, [1], tensor_type=_tfl_tensor_type.FLOAT32),
        _build_tensor(builder, 0, [1], tensor_type=_tfl_tensor_type.FLOAT32),
    ]
    op = _build_operator(builder, 0, [0], [1])
    subgraph = _build_subgraph(builder, tensors=tensors, operators=[op], inputs=[0], outputs=[1])
    return _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=[op_code],
        buffers=[_build_buffer(builder)],
    )


@pytest.mark.parametrize("builtin_name", ["DELEGATE", "PLACEHOLDER_FOR_GREATER_OP_CODES"])
def test_operator_marker_unsupported(builtin_name):
    """TFLite marker builtins report explicit unsupported diagnostics."""
    with pytest.raises(NotImplementedError, match=f"TFLite operator marker {builtin_name}"):
        _load_model_from_buffer(_build_tflite_operator_marker_model(builtin_name))


def _build_tflite_squeeze_model():
    builder = flatbuffers.Builder(1024)

    squeeze_opts = _build_squeeze_options(builder, [0, 2])
    squeeze_op_code = _build_operator_code(builder, _tfl_builtin_operator.SQUEEZE)

    tensors = [
        _build_tensor(builder, 0, [1, 2, 1, 3]),
        _build_tensor(builder, 0, [2, 3]),
    ]
    squeeze_op = _build_operator(
        builder,
        0,
        [0],
        [1],
        builtin_options_type=_tfl_builtin_options.SqueezeOptions,
        builtin_options=squeeze_opts,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=tensors,
        operators=[squeeze_op],
        inputs=[0],
        outputs=[1],
    )
    buffers = [_build_buffer(builder)]
    return _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=[squeeze_op_code],
        buffers=buffers,
    )


def _build_tflite_reverse_sequence_model():
    builder = flatbuffers.Builder(1024)

    reverse_sequence_opts = _build_reverse_sequence_options(builder, seq_dim=1, batch_dim=0)
    reverse_sequence_op_code = _build_operator_code(builder, _tfl_builtin_operator.REVERSE_SEQUENCE)

    tensors = [
        _build_tensor(builder, 0, [2, 4, 3]),
        _build_tensor(builder, 0, [2], tensor_type=_tfl_tensor_type.INT32),
        _build_tensor(builder, 0, [2, 4, 3]),
    ]
    reverse_sequence_op = _build_operator(
        builder,
        0,
        [0, 1],
        [2],
        builtin_options_type=_tfl_builtin_options.ReverseSequenceOptions,
        builtin_options=reverse_sequence_opts,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=tensors,
        operators=[reverse_sequence_op],
        inputs=[0, 1],
        outputs=[2],
    )
    buffers = [_build_buffer(builder)]
    return _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=[reverse_sequence_op_code],
        buffers=buffers,
    )


def _build_tflite_unpack_model():
    builder = flatbuffers.Builder(1024)

    unpack_opts = _build_unpack_options(builder, num=3, axis=1)
    unpack_op_code = _build_operator_code(builder, _tfl_builtin_operator.UNPACK)

    tensors = [
        _build_tensor(builder, 0, [2, 3, 4]),
        _build_tensor(builder, 0, [2, 4]),
        _build_tensor(builder, 0, [2, 4]),
        _build_tensor(builder, 0, [2, 4]),
    ]
    unpack_op = _build_operator(
        builder,
        0,
        [0],
        [1, 2, 3],
        builtin_options_type=_tfl_builtin_options.UnpackOptions,
        builtin_options=unpack_opts,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=tensors,
        operators=[unpack_op],
        inputs=[0],
        outputs=[1, 2, 3],
    )
    buffers = [_build_buffer(builder)]
    return _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=[unpack_op_code],
        buffers=buffers,
    )


def _build_tflite_zeros_like_model():
    builder = flatbuffers.Builder(1024)

    _tfl_zeros_like_options.ZerosLikeOptionsStart(builder)
    zeros_like_opts = _tfl_zeros_like_options.ZerosLikeOptionsEnd(builder)
    zeros_like_op_code = _build_operator_code(builder, _tfl_builtin_operator.ZEROS_LIKE)

    tensors = [
        _build_tensor(builder, 0, [2, 3]),
        _build_tensor(builder, 0, [2, 3]),
    ]
    zeros_like_op = _build_operator(
        builder,
        0,
        [0],
        [1],
        builtin_options_type=_tfl_builtin_options.ZerosLikeOptions,
        builtin_options=zeros_like_opts,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=tensors,
        operators=[zeros_like_op],
        inputs=[0],
        outputs=[1],
    )
    buffers = [_build_buffer(builder)]
    return _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=[zeros_like_op_code],
        buffers=buffers,
    )


def _run_module(mod, *inputs):
    tgt = tvm_tirx.target.Target("c")
    ex = tvm.compile(mod, tgt)
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    vm.set_input("main", *inputs)
    vm.invoke_stateful("main")
    outputs = vm.get_outputs("main")
    if hasattr(outputs, "numpy"):
        return outputs.numpy()
    return tuple(output.numpy() for output in outputs)


def _run_no_input_module(mod):
    return _run_module(mod)


def _complex64_to_pair(value):
    value = np.asarray(value, dtype=np.complex64)
    return np.stack([value.real, value.imag], axis=-1).astype("float32")


def _build_tflite_rfft2d_model(*, input_shape, fft_length, output_shape):
    """Build a minimal TFLite RFFT2D model."""
    builder = flatbuffers.Builder(1024)
    builtin_op = _get_builtin_operator("RFFT2D")
    op_code = _build_operator_code(builder, builtin_op)
    tensors = [
        _build_tensor(builder, 0, input_shape, tensor_type=_tfl_tensor_type.FLOAT32),
        _build_tensor(builder, 1, [2], tensor_type=_tfl_tensor_type.INT32),
        _build_tensor(builder, 2, output_shape, tensor_type=_tfl_tensor_type.COMPLEX64),
    ]
    op = _build_operator(builder, 0, [0, 1], [2])
    subgraph = _build_subgraph(builder, tensors=tensors, operators=[op], inputs=[0], outputs=[2])
    buffers = [
        _build_buffer(builder),
        _build_buffer(builder, np.array(fft_length, dtype=np.int32).tobytes()),
        _build_buffer(builder),
    ]
    return _finish_tflite_model(
        builder, subgraph=subgraph, operator_codes=[op_code], buffers=buffers
    )


def test_rfft2d_static_pair_output():
    """TFLite RFFT2D emits a call_tir kernel with float32 real/imag pair output."""
    mod = _load_model_from_buffer(
        _build_tflite_rfft2d_model(
            input_shape=[2, 4],
            fft_length=[2, 4],
            output_shape=[2, 3],
        )
    )

    data = np.array([[1.0, -2.0, 3.0, 4.0], [5.0, 6.0, -7.0, 8.0]], dtype="float32")
    expected = np.fft.rfft2(data).astype(np.complex64)
    # atol accommodates the float32 reference kernel: numpy's rfft2 internally uses
    # float64, while the reference TIR kernel accumulates in float32 (see
    # _build_tflite_rfft2d_function docstring).
    np.testing.assert_allclose(
        _run_module(mod, data), _complex64_to_pair(expected), rtol=1e-5, atol=1e-5
    )


def test_rfft2d_static_pair_output_with_batch():
    """RFFT2D computes over the last two axes and preserves leading batch dimensions."""
    mod = _load_model_from_buffer(
        _build_tflite_rfft2d_model(
            input_shape=[2, 2, 4],
            fft_length=[2, 4],
            output_shape=[2, 2, 3],
        )
    )

    data = np.array(
        [
            [[1.0, -2.0, 3.0, 4.0], [5.0, 6.0, -7.0, 8.0]],
            [[-1.0, 2.0, 0.5, -4.0], [3.5, -6.0, 7.0, 1.0]],
        ],
        dtype="float32",
    )
    expected = np.fft.rfft2(data).astype(np.complex64)
    np.testing.assert_allclose(
        _run_module(mod, data), _complex64_to_pair(expected), rtol=1e-5, atol=1e-5
    )


def test_rfft2d_odd_width_pair_output():
    """RFFT2D handles odd width: output has width//2 + 1 bins (TFLite convention)."""
    mod = _load_model_from_buffer(
        _build_tflite_rfft2d_model(
            input_shape=[3, 5],
            fft_length=[3, 5],
            output_shape=[3, 3],  # 5 // 2 + 1 = 3
        )
    )

    data = np.array(
        [[1.0, -2.0, 3.0, 4.0, -5.0], [0.5, 6.0, -7.0, 8.0, 2.5], [-1.5, 4.0, 0.0, -3.0, 1.0]],
        dtype="float32",
    )
    expected = np.fft.rfft2(data).astype(np.complex64)
    # atol accommodates the float32 reference kernel (see
    # _build_tflite_rfft2d_function docstring).
    np.testing.assert_allclose(
        _run_module(mod, data), _complex64_to_pair(expected), rtol=1e-5, atol=1e-5
    )


def test_rfft2d_int64_fft_length():
    """RFFT2D accepts INT64 fft_length constant (TFLite schema allows either int32 or int64)."""
    builder = flatbuffers.Builder(1024)
    rfft_op_code = _build_operator_code(builder, _get_builtin_operator("RFFT2D"))
    tensors = [
        _build_tensor(builder, 0, [2, 4], tensor_type=_tfl_tensor_type.FLOAT32),
        _build_tensor(builder, 1, [2], tensor_type=_tfl_tensor_type.INT64),
        _build_tensor(builder, 2, [2, 3], tensor_type=_tfl_tensor_type.COMPLEX64),
    ]
    op = _build_operator(builder, 0, [0, 1], [2])
    subgraph = _build_subgraph(builder, tensors=tensors, operators=[op], inputs=[0], outputs=[2])
    buffers = [
        _build_buffer(builder),
        _build_buffer(builder, np.array([2, 4], dtype=np.int64).tobytes()),
        _build_buffer(builder),
    ]
    buf = _finish_tflite_model(
        builder, subgraph=subgraph, operator_codes=[rfft_op_code], buffers=buffers
    )
    mod = _load_model_from_buffer(buf)

    data = np.array([[1.0, -2.0, 3.0, 4.0], [5.0, 6.0, -7.0, 8.0]], dtype="float32")
    expected = np.fft.rfft2(data).astype(np.complex64)
    np.testing.assert_allclose(
        _run_module(mod, data), _complex64_to_pair(expected), rtol=1e-5, atol=1e-5
    )


def test_rfft2d_4d_input_pair_output():
    """RFFT2D accepts 4D input and preserves leading batch dimensions."""
    mod = _load_model_from_buffer(
        _build_tflite_rfft2d_model(
            input_shape=[2, 3, 4, 5],  # batch=6, H=4, W=5
            fft_length=[4, 5],
            output_shape=[2, 3, 4, 3],  # 5 // 2 + 1 = 3
        )
    )

    rng = np.random.RandomState(0)
    data = (rng.randn(2, 3, 4, 5) * 0.5).astype("float32")
    expected = np.fft.rfft2(data).astype(np.complex64)
    # 4D test accumulates 20 inner terms per output; use a slightly larger atol
    # than the 2D case (which accumulates 4-8 terms).
    np.testing.assert_allclose(
        _run_module(mod, data), _complex64_to_pair(expected), rtol=1e-5, atol=1e-4
    )


def test_rfft2d_minimal_1x1_pair_output():
    """RFFT2D on a [1, 1] input: the only output is the DC component (sum of inputs)."""
    mod = _load_model_from_buffer(
        _build_tflite_rfft2d_model(
            input_shape=[1, 1],
            fft_length=[1, 1],
            output_shape=[1, 1],
        )
    )

    data = np.array([[3.5]], dtype="float32")
    expected = np.fft.rfft2(data).astype(np.complex64)
    np.testing.assert_allclose(
        _run_module(mod, data), _complex64_to_pair(expected), rtol=1e-5, atol=1e-5
    )


def test_rfft2d_fft_path_8x8():
    """RFFT2D on a square 8x8 input exercises the Cooley-Tukey FFT dispatch path."""
    mod = _load_model_from_buffer(
        _build_tflite_rfft2d_model(
            input_shape=[8, 8],
            fft_length=[8, 8],
            output_shape=[8, 5],
        )
    )

    np.random.seed(0xCAFE)
    data = np.random.randn(8, 8).astype("float32")
    expected = np.fft.rfft2(data).astype(np.complex64)
    # The FFT path uses float32 twiddles (cos/sin) and float32 butterfly
    # accumulation, so the error vs. numpy's float64 reference is in the
    # 1e-4 range on these random inputs.
    np.testing.assert_allclose(
        _run_module(mod, data), _complex64_to_pair(expected), rtol=1e-4, atol=1e-4
    )


def test_rfft2d_fft_path_4x4():
    """RFFT2D on a 4x4 input: smallest case where both row and column FFTs do real work."""
    mod = _load_model_from_buffer(
        _build_tflite_rfft2d_model(
            input_shape=[4, 4],
            fft_length=[4, 4],
            output_shape=[4, 3],
        )
    )

    np.random.seed(0xFEED)
    data = np.random.randn(4, 4).astype("float32")
    expected = np.fft.rfft2(data).astype(np.complex64)
    np.testing.assert_allclose(
        _run_module(mod, data), _complex64_to_pair(expected), rtol=1e-4, atol=1e-4
    )


def test_rfft2d_fft_path_2x2x4x8():
    """RFFT2D on a 4D input with power-of-2 height/width exercises the FFT path with batch."""
    mod = _load_model_from_buffer(
        _build_tflite_rfft2d_model(
            input_shape=[2, 2, 4, 8],
            fft_length=[4, 8],
            output_shape=[2, 2, 4, 5],
        )
    )

    np.random.seed(0xBEEF)
    data = np.random.randn(2, 2, 4, 8).astype("float32")
    expected = np.fft.rfft2(data, axes=(-2, -1)).astype(np.complex64)
    np.testing.assert_allclose(
        _run_module(mod, data), _complex64_to_pair(expected), rtol=1e-4, atol=1e-4
    )


def test_rfft2d_fft_path_16x16():
    """RFFT2D on a 16x16 input: a larger FFT to check that the unrolled kernel scales."""
    mod = _load_model_from_buffer(
        _build_tflite_rfft2d_model(
            input_shape=[16, 16],
            fft_length=[16, 16],
            output_shape=[16, 9],
        )
    )

    np.random.seed(0xDEAD)
    data = np.random.randn(16, 16).astype("float32")
    expected = np.fft.rfft2(data).astype(np.complex64)
    np.testing.assert_allclose(
        _run_module(mod, data), _complex64_to_pair(expected), rtol=1e-4, atol=1e-4
    )


def test_rfft2d_mismatched_fft_length_unsupported():
    """RFFT2D padding/truncation cases are guarded until explicitly implemented."""
    buf = _build_tflite_rfft2d_model(
        input_shape=[2, 4],
        fft_length=[4, 4],
        output_shape=[4, 3],
    )
    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)

    with pytest.raises(NotImplementedError, match="fft_length"):
        from_tflite(tflite_model)


def test_rfft2d_dynamic_fft_length_unsupported():
    """RFFT2D requires fft_length to be a constant tensor."""
    builder = flatbuffers.Builder(1024)
    rfft_op_code = _build_operator_code(builder, _get_builtin_operator("RFFT2D"))
    tensors = [
        _build_tensor(builder, 0, [2, 4], tensor_type=_tfl_tensor_type.FLOAT32),
        _build_tensor(builder, 1, [2], tensor_type=_tfl_tensor_type.INT32),
        _build_tensor(builder, 2, [2, 3], tensor_type=_tfl_tensor_type.COMPLEX64),
    ]
    op = _build_operator(builder, 0, [0, 1], [2])
    subgraph = _build_subgraph(builder, tensors=tensors, operators=[op], inputs=[0, 1], outputs=[2])
    buf = _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=[rfft_op_code],
        buffers=[_build_buffer(builder), _build_buffer(builder), _build_buffer(builder)],
    )
    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)

    with pytest.raises(NotImplementedError, match="requires a constant fft_length"):
        from_tflite(tflite_model)


def _build_tflite_call_model(
    call_subgraph_index=1,
    callee_inputs=None,
    callee_outputs=None,
    callee_output_shape=None,
    callee_output_type=None,
):
    """Build a TFLite model where main CALLs a subgraph computing x + 1."""
    builder = flatbuffers.Builder(1024)

    callee_inputs = [0] if callee_inputs is None else callee_inputs
    callee_outputs = [2] if callee_outputs is None else callee_outputs
    callee_output_shape = [2, 2] if callee_output_shape is None else callee_output_shape
    callee_output_type = (
        _tfl_tensor_type.FLOAT32 if callee_output_type is None else callee_output_type
    )
    call_options = _build_call_options(builder, call_subgraph_index)
    one = np.array(1.0, dtype=np.float32)

    main_tensors = [
        _build_tensor(builder, 0, [2, 2]),
        _build_tensor(builder, 2, [2, 2]),
    ]
    main_call = _build_operator(
        builder,
        0,
        [0],
        [1],
        builtin_options_type=_tfl_builtin_options.CallOptions,
        builtin_options=call_options,
    )
    main_subgraph = _build_subgraph(
        builder,
        tensors=main_tensors,
        operators=[main_call],
        inputs=[0],
        outputs=[1],
    )

    callee_tensors = [
        _build_tensor(builder, 0, [2, 2]),
        _build_tensor(builder, 1, []),
        _build_tensor(builder, 2, callee_output_shape, tensor_type=callee_output_type),
    ]
    callee_add = _build_operator(builder, 1, [0, 1], [2])
    callee_subgraph = _build_subgraph(
        builder,
        tensors=callee_tensors,
        operators=[callee_add],
        inputs=callee_inputs,
        outputs=callee_outputs,
    )

    operator_codes = [
        _build_operator_code(builder, _get_builtin_operator("CALL")),
        _build_operator_code(builder, _get_builtin_operator("ADD")),
    ]
    buffers = [
        _build_buffer(builder),
        _build_buffer(builder, one.tobytes()),
        _build_buffer(builder),
    ]
    return _finish_tflite_model(
        builder,
        subgraph=main_subgraph,
        extra_subgraphs=[callee_subgraph],
        operator_codes=operator_codes,
        buffers=buffers,
    )


def test_call_subgraph():
    """Test TFLite CALL conversion to a private Relax function."""
    mod = _load_model_from_buffer(_build_tflite_call_model())

    @I.ir_module
    class Expected:
        @R.function(private=True)
        def tflite_call_subgraph_1(
            tvmgen_tensor_0: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tensor((2, 2), dtype="float32"):
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="float32") = R.add(
                    tvmgen_tensor_0, R.const(1.0, "float32")
                )
                R.output(gv)
            return gv

        @R.function
        def main(
            tvmgen_tensor_0: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tensor((2, 2), dtype="float32"):
            R.func_attr({"num_input": 1})
            cls = Expected
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="float32") = cls.tflite_call_subgraph_1(tvmgen_tensor_0)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def _build_tflite_multi_output_call_model():
    """Build a TFLite model where CALL returns x + 1 and x - 1."""
    builder = flatbuffers.Builder(1024)

    call_options = _build_call_options(builder, 1)
    one = np.array(1.0, dtype=np.float32)

    main_tensors = [
        _build_tensor(builder, 0, [2, 2]),
        _build_tensor(builder, 2, [2, 2]),
        _build_tensor(builder, 3, [2, 2]),
    ]
    main_call = _build_operator(
        builder,
        0,
        [0],
        [1, 2],
        builtin_options_type=_tfl_builtin_options.CallOptions,
        builtin_options=call_options,
    )
    main_subgraph = _build_subgraph(
        builder,
        tensors=main_tensors,
        operators=[main_call],
        inputs=[0],
        outputs=[1, 2],
    )

    callee_tensors = [
        _build_tensor(builder, 0, [2, 2]),
        _build_tensor(builder, 1, []),
        _build_tensor(builder, 2, [2, 2]),
        _build_tensor(builder, 3, [2, 2]),
    ]
    callee_add = _build_operator(builder, 1, [0, 1], [2])
    callee_sub = _build_operator(builder, 2, [0, 1], [3])
    callee_subgraph = _build_subgraph(
        builder,
        tensors=callee_tensors,
        operators=[callee_add, callee_sub],
        inputs=[0],
        outputs=[2, 3],
    )

    operator_codes = [
        _build_operator_code(builder, _get_builtin_operator("CALL")),
        _build_operator_code(builder, _get_builtin_operator("ADD")),
        _build_operator_code(builder, _get_builtin_operator("SUB")),
    ]
    buffers = [
        _build_buffer(builder),
        _build_buffer(builder, one.tobytes()),
        _build_buffer(builder),
        _build_buffer(builder),
    ]
    return _finish_tflite_model(
        builder,
        subgraph=main_subgraph,
        extra_subgraphs=[callee_subgraph],
        operator_codes=operator_codes,
        buffers=buffers,
    )


def test_call_subgraph_multi_output():
    """Test CALL tuple returns are split and rebound to TFLite output tensors."""
    mod = _load_model_from_buffer(_build_tflite_multi_output_call_model())

    @I.ir_module
    class Expected:
        @R.function(private=True)
        def tflite_call_subgraph_1(
            tvmgen_tensor_0: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 2), dtype="float32"), R.Tensor((2, 2), dtype="float32")):
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="float32") = R.add(
                    tvmgen_tensor_0, R.const(1.0, "float32")
                )
                gv1: R.Tensor((2, 2), dtype="float32") = R.subtract(
                    tvmgen_tensor_0, R.const(1.0, "float32")
                )
                gv2: R.Tuple(
                    R.Tensor((2, 2), dtype="float32"), R.Tensor((2, 2), dtype="float32")
                ) = (gv, gv1)
                R.output(gv2)
            return gv2

        @R.function
        def main(
            tvmgen_tensor_0: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 2), dtype="float32"), R.Tensor((2, 2), dtype="float32")):
            R.func_attr({"num_input": 1})
            cls = Expected
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((2, 2), dtype="float32"), R.Tensor((2, 2), dtype="float32")
                ) = cls.tflite_call_subgraph_1(tvmgen_tensor_0)
                lv1: R.Tensor((2, 2), dtype="float32") = lv[0]
                lv2: R.Tensor((2, 2), dtype="float32") = lv[1]
                gv: R.Tuple(
                    R.Tensor((2, 2), dtype="float32"), R.Tensor((2, 2), dtype="float32")
                ) = (lv1, lv2)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def _build_tflite_nested_call_model():
    """Build a TFLite model where main CALLs subgraph A, which CALLs subgraph B."""
    builder = flatbuffers.Builder(1024)

    main_call_options = _build_call_options(builder, 1)
    nested_call_options = _build_call_options(builder, 2)
    one = np.array(1.0, dtype=np.float32)

    main_tensors = [
        _build_tensor(builder, 0, [2, 2]),
        _build_tensor(builder, 3, [2, 2]),
    ]
    main_call = _build_operator(
        builder,
        0,
        [0],
        [1],
        builtin_options_type=_tfl_builtin_options.CallOptions,
        builtin_options=main_call_options,
    )
    main_subgraph = _build_subgraph(
        builder,
        tensors=main_tensors,
        operators=[main_call],
        inputs=[0],
        outputs=[1],
    )

    caller_tensors = [
        _build_tensor(builder, 0, [2, 2]),
        _build_tensor(builder, 3, [2, 2]),
    ]
    nested_call = _build_operator(
        builder,
        0,
        [0],
        [1],
        builtin_options_type=_tfl_builtin_options.CallOptions,
        builtin_options=nested_call_options,
    )
    caller_subgraph = _build_subgraph(
        builder,
        tensors=caller_tensors,
        operators=[nested_call],
        inputs=[0],
        outputs=[1],
    )

    callee_tensors = [
        _build_tensor(builder, 0, [2, 2]),
        _build_tensor(builder, 1, []),
        _build_tensor(builder, 3, [2, 2]),
    ]
    callee_add = _build_operator(builder, 1, [0, 1], [2])
    callee_subgraph = _build_subgraph(
        builder,
        tensors=callee_tensors,
        operators=[callee_add],
        inputs=[0],
        outputs=[2],
    )

    operator_codes = [
        _build_operator_code(builder, _get_builtin_operator("CALL")),
        _build_operator_code(builder, _get_builtin_operator("ADD")),
    ]
    buffers = [
        _build_buffer(builder),
        _build_buffer(builder, one.tobytes()),
        _build_buffer(builder),
        _build_buffer(builder),
    ]
    return _finish_tflite_model(
        builder,
        subgraph=main_subgraph,
        extra_subgraphs=[caller_subgraph, callee_subgraph],
        operator_codes=operator_codes,
        buffers=buffers,
    )


def test_call_subgraph_nested_call():
    """Test nested CALL subgraphs register all generated private functions."""
    mod = _load_model_from_buffer(_build_tflite_nested_call_model())

    @I.ir_module
    class Expected:
        @R.function(private=True)
        def tflite_call_subgraph_2(
            tvmgen_tensor_0: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tensor((2, 2), dtype="float32"):
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="float32") = R.add(
                    tvmgen_tensor_0, R.const(1.0, "float32")
                )
                R.output(gv)
            return gv

        @R.function(private=True)
        def tflite_call_subgraph_1(
            tvmgen_tensor_0: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tensor((2, 2), dtype="float32"):
            cls = Expected
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="float32") = cls.tflite_call_subgraph_2(tvmgen_tensor_0)
                R.output(gv)
            return gv

        @R.function
        def main(
            tvmgen_tensor_0: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tensor((2, 2), dtype="float32"):
            R.func_attr({"num_input": 1})
            cls = Expected
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="float32") = cls.tflite_call_subgraph_1(tvmgen_tensor_0)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_call_subgraph_invalid_index_unsupported():
    """Test CALL rejects invalid subgraph indices before lowering."""
    with pytest.raises(NotImplementedError, match="CALL requires a valid subgraph index"):
        _load_model_from_buffer(_build_tflite_call_model(call_subgraph_index=2))


def test_call_subgraph_io_mismatch_unsupported():
    """Test CALL rejects callees whose input arity does not match the call site."""
    with pytest.raises(NotImplementedError, match="CALL subgraph input count mismatch"):
        _load_model_from_buffer(_build_tflite_call_model(callee_inputs=[]))


def test_call_subgraph_output_metadata_mismatch_unsupported():
    """Test CALL rejects callees whose output metadata does not match the call site."""
    with pytest.raises(NotImplementedError, match="CALL subgraph output tensor metadata mismatch"):
        _load_model_from_buffer(_build_tflite_call_model(callee_output_shape=[2]))


def _build_tflite_if_model(
    condition_type=_tfl_tensor_type.BOOL,
    then_subgraph_index=1,
    else_subgraph_index=2,
    then_outputs=None,
    else_outputs=None,
    else_input_shape=None,
    else_input_type=None,
    else_output_shape=None,
    else_output_type=None,
):
    """Build a TFLite model where IF selects x + 1 or x - 1."""
    builder = flatbuffers.Builder(1024)

    then_outputs = [2] if then_outputs is None else then_outputs
    else_outputs = [2] if else_outputs is None else else_outputs
    else_input_shape = [2, 2] if else_input_shape is None else else_input_shape
    else_input_type = _tfl_tensor_type.FLOAT32 if else_input_type is None else else_input_type
    else_output_shape = [2, 2] if else_output_shape is None else else_output_shape
    else_output_type = _tfl_tensor_type.FLOAT32 if else_output_type is None else else_output_type
    if_options = _build_if_options(builder, then_subgraph_index, else_subgraph_index)
    one = np.array(1.0, dtype=np.float32)

    main_tensors = [
        _build_tensor(builder, 0, [], tensor_type=condition_type),
        _build_tensor(builder, 1, [2, 2]),
        _build_tensor(builder, 3, [2, 2]),
    ]
    main_if = _build_operator(
        builder,
        0,
        [0, 1],
        [2],
        builtin_options_type=_tfl_builtin_options.IfOptions,
        builtin_options=if_options,
    )
    main_subgraph = _build_subgraph(
        builder,
        tensors=main_tensors,
        operators=[main_if],
        inputs=[0, 1],
        outputs=[2],
    )

    then_tensors = [
        _build_tensor(builder, 1, [2, 2]),
        _build_tensor(builder, 2, []),
        _build_tensor(builder, 3, [2, 2]),
    ]
    then_add = _build_operator(builder, 1, [0, 1], [2])
    then_subgraph = _build_subgraph(
        builder,
        tensors=then_tensors,
        operators=[then_add],
        inputs=[0],
        outputs=then_outputs,
    )

    else_tensors = [
        _build_tensor(builder, 1, else_input_shape, tensor_type=else_input_type),
        _build_tensor(builder, 2, []),
        _build_tensor(builder, 3, else_output_shape, tensor_type=else_output_type),
    ]
    else_sub = _build_operator(builder, 2, [0, 1], [2])
    else_subgraph = _build_subgraph(
        builder,
        tensors=else_tensors,
        operators=[else_sub],
        inputs=[0],
        outputs=else_outputs,
    )

    operator_codes = [
        _build_operator_code(builder, _get_builtin_operator("IF")),
        _build_operator_code(builder, _get_builtin_operator("ADD")),
        _build_operator_code(builder, _get_builtin_operator("SUB")),
    ]
    buffers = [
        _build_buffer(builder),
        _build_buffer(builder),
        _build_buffer(builder, one.tobytes()),
        _build_buffer(builder),
    ]
    return _finish_tflite_model(
        builder,
        subgraph=main_subgraph,
        extra_subgraphs=[then_subgraph, else_subgraph],
        operator_codes=operator_codes,
        buffers=buffers,
    )


def test_if_subgraphs():
    """Test TFLite IF conversion to Relax If."""
    mod = _load_model_from_buffer(_build_tflite_if_model())

    @I.ir_module
    class Expected:
        @R.function(private=True)
        def tflite_if_then_subgraph_1(
            tvmgen_tensor_0: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tensor((2, 2), dtype="float32"):
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="float32") = R.add(
                    tvmgen_tensor_0, R.const(1.0, "float32")
                )
                R.output(gv)
            return gv

        @R.function(private=True)
        def tflite_if_else_subgraph_2(
            tvmgen_tensor_0: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tensor((2, 2), dtype="float32"):
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="float32") = R.subtract(
                    tvmgen_tensor_0, R.const(1.0, "float32")
                )
                R.output(gv)
            return gv

        @R.function(private=True)
        def tflite_if_subgraph_1_2(
            tvmgen_tensor_0: R.Tensor((), dtype="bool"),
            tvmgen_tensor_1: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tensor((2, 2), dtype="float32"):
            cls = Expected
            if tvmgen_tensor_0:
                gv: R.Tensor((2, 2), dtype="float32") = cls.tflite_if_then_subgraph_1(
                    tvmgen_tensor_1
                )
                cond_result: R.Tensor((2, 2), dtype="float32") = gv
            else:
                gv1: R.Tensor((2, 2), dtype="float32") = cls.tflite_if_else_subgraph_2(
                    tvmgen_tensor_1
                )
                cond_result: R.Tensor((2, 2), dtype="float32") = gv1
            return cond_result

        @R.function
        def main(
            tvmgen_tensor_0: R.Tensor((), dtype="bool"),
            tvmgen_tensor_1: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tensor((2, 2), dtype="float32"):
            R.func_attr({"num_input": 2})
            cls = Expected
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="float32") = cls.tflite_if_subgraph_1_2(
                    tvmgen_tensor_0, tvmgen_tensor_1
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def _build_tflite_multi_output_if_model():
    """Build a TFLite model where IF returns two tensor outputs."""
    builder = flatbuffers.Builder(1024)

    if_options = _build_if_options(builder, 1, 2)
    one = np.array(1.0, dtype=np.float32)

    main_tensors = [
        _build_tensor(builder, 0, [], tensor_type=_tfl_tensor_type.BOOL),
        _build_tensor(builder, 1, [2, 2]),
        _build_tensor(builder, 4, [2, 2]),
        _build_tensor(builder, 5, [2, 2]),
    ]
    main_if = _build_operator(
        builder,
        0,
        [0, 1],
        [2, 3],
        builtin_options_type=_tfl_builtin_options.IfOptions,
        builtin_options=if_options,
    )
    main_subgraph = _build_subgraph(
        builder,
        tensors=main_tensors,
        operators=[main_if],
        inputs=[0, 1],
        outputs=[2, 3],
    )

    then_tensors = [
        _build_tensor(builder, 1, [2, 2]),
        _build_tensor(builder, 2, []),
        _build_tensor(builder, 3, [2, 2]),
        _build_tensor(builder, 4, [2, 2]),
    ]
    then_add = _build_operator(builder, 1, [0, 1], [2])
    then_sub = _build_operator(builder, 2, [0, 1], [3])
    then_subgraph = _build_subgraph(
        builder,
        tensors=then_tensors,
        operators=[then_add, then_sub],
        inputs=[0],
        outputs=[2, 3],
    )

    else_tensors = [
        _build_tensor(builder, 1, [2, 2]),
        _build_tensor(builder, 2, []),
        _build_tensor(builder, 3, [2, 2]),
        _build_tensor(builder, 4, [2, 2]),
    ]
    else_sub = _build_operator(builder, 2, [0, 1], [2])
    else_add = _build_operator(builder, 1, [0, 1], [3])
    else_subgraph = _build_subgraph(
        builder,
        tensors=else_tensors,
        operators=[else_sub, else_add],
        inputs=[0],
        outputs=[2, 3],
    )

    operator_codes = [
        _build_operator_code(builder, _get_builtin_operator("IF")),
        _build_operator_code(builder, _get_builtin_operator("ADD")),
        _build_operator_code(builder, _get_builtin_operator("SUB")),
    ]
    buffers = [
        _build_buffer(builder),
        _build_buffer(builder),
        _build_buffer(builder, one.tobytes()),
        _build_buffer(builder),
        _build_buffer(builder),
        _build_buffer(builder),
    ]
    return _finish_tflite_model(
        builder,
        subgraph=main_subgraph,
        extra_subgraphs=[then_subgraph, else_subgraph],
        operator_codes=operator_codes,
        buffers=buffers,
    )


def test_if_subgraphs_multi_output():
    """Test IF tuple returns are preserved through the private wrapper function."""
    mod = _load_model_from_buffer(_build_tflite_multi_output_if_model())

    @I.ir_module
    class Expected:
        @R.function(private=True)
        def tflite_if_then_subgraph_1(
            tvmgen_tensor_0: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 2), dtype="float32"), R.Tensor((2, 2), dtype="float32")):
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="float32") = R.add(
                    tvmgen_tensor_0, R.const(1.0, "float32")
                )
                gv1: R.Tensor((2, 2), dtype="float32") = R.subtract(
                    tvmgen_tensor_0, R.const(1.0, "float32")
                )
                gv2: R.Tuple(
                    R.Tensor((2, 2), dtype="float32"), R.Tensor((2, 2), dtype="float32")
                ) = (gv, gv1)
                R.output(gv2)
            return gv2

        @R.function(private=True)
        def tflite_if_else_subgraph_2(
            tvmgen_tensor_0: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 2), dtype="float32"), R.Tensor((2, 2), dtype="float32")):
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="float32") = R.subtract(
                    tvmgen_tensor_0, R.const(1.0, "float32")
                )
                gv1: R.Tensor((2, 2), dtype="float32") = R.add(
                    tvmgen_tensor_0, R.const(1.0, "float32")
                )
                gv2: R.Tuple(
                    R.Tensor((2, 2), dtype="float32"), R.Tensor((2, 2), dtype="float32")
                ) = (gv, gv1)
                R.output(gv2)
            return gv2

        @R.function(private=True)
        def tflite_if_subgraph_1_2(
            tvmgen_tensor_0: R.Tensor((), dtype="bool"),
            tvmgen_tensor_1: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 2), dtype="float32"), R.Tensor((2, 2), dtype="float32")):
            cls = Expected
            if tvmgen_tensor_0:
                gv: R.Tuple(
                    R.Tensor((2, 2), dtype="float32"), R.Tensor((2, 2), dtype="float32")
                ) = cls.tflite_if_then_subgraph_1(tvmgen_tensor_1)
                cond_result: R.Tuple(
                    R.Tensor((2, 2), dtype="float32"), R.Tensor((2, 2), dtype="float32")
                ) = gv
            else:
                gv1: R.Tuple(
                    R.Tensor((2, 2), dtype="float32"), R.Tensor((2, 2), dtype="float32")
                ) = cls.tflite_if_else_subgraph_2(tvmgen_tensor_1)
                cond_result: R.Tuple(
                    R.Tensor((2, 2), dtype="float32"), R.Tensor((2, 2), dtype="float32")
                ) = gv1
            return cond_result

        @R.function
        def main(
            tvmgen_tensor_0: R.Tensor((), dtype="bool"),
            tvmgen_tensor_1: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 2), dtype="float32"), R.Tensor((2, 2), dtype="float32")):
            R.func_attr({"num_input": 2})
            cls = Expected
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((2, 2), dtype="float32"), R.Tensor((2, 2), dtype="float32")
                ) = cls.tflite_if_subgraph_1_2(tvmgen_tensor_0, tvmgen_tensor_1)
                lv1: R.Tensor((2, 2), dtype="float32") = lv[0]
                lv2: R.Tensor((2, 2), dtype="float32") = lv[1]
                gv: R.Tuple(
                    R.Tensor((2, 2), dtype="float32"), R.Tensor((2, 2), dtype="float32")
                ) = (lv1, lv2)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_if_subgraphs_non_bool_condition_unsupported():
    """Test IF rejects non-bool condition tensors."""
    with pytest.raises(NotImplementedError, match="IF requires a scalar bool condition"):
        _load_model_from_buffer(_build_tflite_if_model(condition_type=_tfl_tensor_type.INT32))


def test_if_subgraphs_invalid_index_unsupported():
    """Test IF rejects invalid branch subgraph indices before lowering."""
    with pytest.raises(NotImplementedError, match="IF requires a valid subgraph index"):
        _load_model_from_buffer(_build_tflite_if_model(then_subgraph_index=3))


def test_if_subgraphs_output_count_mismatch_unsupported():
    """Test IF rejects branches whose output arity does not match the call site."""
    with pytest.raises(NotImplementedError, match="IF subgraph output count mismatch"):
        _load_model_from_buffer(_build_tflite_if_model(else_outputs=[]))


def test_if_subgraphs_input_metadata_mismatch_unsupported():
    """Test IF rejects branches whose input metadata does not match the call site."""
    with pytest.raises(NotImplementedError, match="IF subgraph input tensor metadata mismatch"):
        _load_model_from_buffer(_build_tflite_if_model(else_input_shape=[2]))


def test_if_subgraphs_output_metadata_mismatch_unsupported():
    """Test IF rejects branches whose output metadata does not match the call site."""
    with pytest.raises(NotImplementedError, match="IF subgraph output tensor metadata mismatch"):
        _load_model_from_buffer(_build_tflite_if_model(else_output_shape=[2]))


def _build_tflite_while_model(
    cond_subgraph_index=1,
    body_subgraph_index=2,
    cond_output_type=_tfl_tensor_type.BOOL,
    cond_input_type=_tfl_tensor_type.INT32,
    body_outputs=None,
    body_input_type=_tfl_tensor_type.INT32,
    body_output_type=_tfl_tensor_type.INT32,
    main_output_type=_tfl_tensor_type.INT32,
):
    """Build a TFLite WHILE model incrementing an int32 scalar until i < 3 is false."""
    builder = flatbuffers.Builder(1024)

    body_outputs = [2] if body_outputs is None else body_outputs
    while_options = _build_while_options(builder, cond_subgraph_index, body_subgraph_index)
    one = np.array(1, dtype=np.int32)
    three = np.array(3, dtype=np.int32)

    main_tensors = [
        _build_tensor(builder, 0, [], tensor_type=_tfl_tensor_type.INT32),
        _build_tensor(builder, 3, [], tensor_type=main_output_type),
    ]
    main_while = _build_operator(
        builder,
        0,
        [0],
        [1],
        builtin_options_type=_tfl_builtin_options.WhileOptions,
        builtin_options=while_options,
    )
    main_subgraph = _build_subgraph(
        builder,
        tensors=main_tensors,
        operators=[main_while],
        inputs=[0],
        outputs=[1],
    )

    cond_tensors = [
        _build_tensor(builder, 0, [], tensor_type=cond_input_type),
        _build_tensor(builder, 1, [], tensor_type=_tfl_tensor_type.INT32),
        _build_tensor(builder, 3, [], tensor_type=cond_output_type),
    ]
    cond_less = _build_operator(builder, 1, [0, 1], [2])
    cond_subgraph = _build_subgraph(
        builder,
        tensors=cond_tensors,
        operators=[cond_less],
        inputs=[0],
        outputs=[2],
    )

    body_tensors = [
        _build_tensor(builder, 0, [], tensor_type=body_input_type),
        _build_tensor(builder, 2, [], tensor_type=_tfl_tensor_type.INT32),
        _build_tensor(builder, 3, [], tensor_type=body_output_type),
    ]
    body_add = _build_operator(builder, 2, [0, 1], [2])
    body_subgraph = _build_subgraph(
        builder,
        tensors=body_tensors,
        operators=[body_add],
        inputs=[0],
        outputs=body_outputs,
    )

    operator_codes = [
        _build_operator_code(builder, _get_builtin_operator("WHILE")),
        _build_operator_code(builder, _get_builtin_operator("LESS")),
        _build_operator_code(builder, _get_builtin_operator("ADD")),
    ]
    buffers = [
        _build_buffer(builder),
        _build_buffer(builder, three.tobytes()),
        _build_buffer(builder, one.tobytes()),
        _build_buffer(builder),
    ]
    return _finish_tflite_model(
        builder,
        subgraph=main_subgraph,
        extra_subgraphs=[cond_subgraph, body_subgraph],
        operator_codes=operator_codes,
        buffers=buffers,
    )


def _build_tflite_repeated_while_model():
    """Build a TFLite model where two WHILE ops share the same cond/body subgraphs."""
    builder = flatbuffers.Builder(1024)

    while_options = _build_while_options(builder, 1, 2)
    one = np.array(1, dtype=np.int32)
    three = np.array(3, dtype=np.int32)

    main_tensors = [
        _build_tensor(builder, 0, [], tensor_type=_tfl_tensor_type.INT32),
        _build_tensor(builder, 3, [], tensor_type=_tfl_tensor_type.INT32),
        _build_tensor(builder, 4, [], tensor_type=_tfl_tensor_type.INT32),
    ]
    main_while_0 = _build_operator(
        builder,
        0,
        [0],
        [1],
        builtin_options_type=_tfl_builtin_options.WhileOptions,
        builtin_options=while_options,
    )
    main_while_1 = _build_operator(
        builder,
        0,
        [1],
        [2],
        builtin_options_type=_tfl_builtin_options.WhileOptions,
        builtin_options=while_options,
    )
    main_subgraph = _build_subgraph(
        builder,
        tensors=main_tensors,
        operators=[main_while_0, main_while_1],
        inputs=[0],
        outputs=[2],
    )

    cond_tensors = [
        _build_tensor(builder, 0, [], tensor_type=_tfl_tensor_type.INT32),
        _build_tensor(builder, 1, [], tensor_type=_tfl_tensor_type.INT32),
        _build_tensor(builder, 3, [], tensor_type=_tfl_tensor_type.BOOL),
    ]
    cond_less = _build_operator(builder, 1, [0, 1], [2])
    cond_subgraph = _build_subgraph(
        builder,
        tensors=cond_tensors,
        operators=[cond_less],
        inputs=[0],
        outputs=[2],
    )

    body_tensors = [
        _build_tensor(builder, 0, [], tensor_type=_tfl_tensor_type.INT32),
        _build_tensor(builder, 2, [], tensor_type=_tfl_tensor_type.INT32),
        _build_tensor(builder, 3, [], tensor_type=_tfl_tensor_type.INT32),
    ]
    body_add = _build_operator(builder, 2, [0, 1], [2])
    body_subgraph = _build_subgraph(
        builder,
        tensors=body_tensors,
        operators=[body_add],
        inputs=[0],
        outputs=[2],
    )

    operator_codes = [
        _build_operator_code(builder, _get_builtin_operator("WHILE")),
        _build_operator_code(builder, _get_builtin_operator("LESS")),
        _build_operator_code(builder, _get_builtin_operator("ADD")),
    ]
    buffers = [
        _build_buffer(builder),
        _build_buffer(builder, three.tobytes()),
        _build_buffer(builder, one.tobytes()),
        _build_buffer(builder),
        _build_buffer(builder),
    ]
    return _finish_tflite_model(
        builder,
        subgraph=main_subgraph,
        extra_subgraphs=[cond_subgraph, body_subgraph],
        operator_codes=operator_codes,
        buffers=buffers,
    )


def _build_tflite_zero_var_while_model():
    """Build a TFLite WHILE model with no loop-carried tensors."""
    builder = flatbuffers.Builder(1024)

    while_options = _build_while_options(builder, 1, 2)
    main_while = _build_operator(
        builder,
        0,
        [],
        [],
        builtin_options_type=_tfl_builtin_options.WhileOptions,
        builtin_options=while_options,
    )
    main_subgraph = _build_subgraph(
        builder,
        tensors=[],
        operators=[main_while],
        inputs=[],
        outputs=[],
    )
    cond_subgraph = _build_subgraph(builder, tensors=[], operators=[], inputs=[], outputs=[])
    body_subgraph = _build_subgraph(builder, tensors=[], operators=[], inputs=[], outputs=[])

    operator_codes = [_build_operator_code(builder, _get_builtin_operator("WHILE"))]
    buffers = [_build_buffer(builder)]
    return _finish_tflite_model(
        builder,
        subgraph=main_subgraph,
        extra_subgraphs=[cond_subgraph, body_subgraph],
        operator_codes=operator_codes,
        buffers=buffers,
    )


def test_while_subgraphs():
    """Test TFLite WHILE conversion to a recursive Relax private function."""
    mod = _load_model_from_buffer(_build_tflite_while_model())

    @I.ir_module
    class Expected:
        @R.function(private=True)
        def tflite_while_cond_subgraph_1(
            tvmgen_tensor_0: R.Tensor((), dtype="int32"),
        ) -> R.Tensor((), dtype="bool"):
            with R.dataflow():
                gv: R.Tensor((), dtype="bool") = R.less(tvmgen_tensor_0, R.const(3, "int32"))
                R.output(gv)
            return gv

        @R.function(private=True)
        def tflite_while_body_subgraph_2(
            tvmgen_tensor_0: R.Tensor((), dtype="int32"),
        ) -> R.Tensor((), dtype="int32"):
            with R.dataflow():
                gv: R.Tensor((), dtype="int32") = R.add(tvmgen_tensor_0, R.const(1, "int32"))
                R.output(gv)
            return gv

        @R.function(private=True)
        def tflite_while_subgraph_1_2(
            tvmgen_tensor_0: R.Tensor((), dtype="int32"),
        ) -> R.Tensor((), dtype="int32"):
            cls = Expected
            while_cond: R.Tensor((), dtype="bool") = cls.tflite_while_cond_subgraph_1(
                tvmgen_tensor_0
            )
            if while_cond:
                gv: R.Tensor((), dtype="int32") = cls.tflite_while_body_subgraph_2(tvmgen_tensor_0)
                gv1: R.Tensor((), dtype="int32") = cls.tflite_while_subgraph_1_2(gv)
                cond_result: R.Tensor((), dtype="int32") = gv1
            else:
                cond_result: R.Tensor((), dtype="int32") = tvmgen_tensor_0
            return cond_result

        @R.function
        def main(
            tvmgen_tensor_0: R.Tensor((), dtype="int32"),
        ) -> R.Tensor((), dtype="int32"):
            R.func_attr({"num_input": 1})
            cls = Expected
            with R.dataflow():
                gv: R.Tensor((), dtype="int32") = cls.tflite_while_subgraph_1_2(tvmgen_tensor_0)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_while_subgraphs_repeated_cond_body_pair():
    """Test repeated WHILE ops reuse the same recursive private function."""
    mod = _load_model_from_buffer(_build_tflite_repeated_while_model())
    names = [gv.name_hint for gv in mod.get_global_vars()]
    assert names.count("tflite_while_subgraph_1_2") == 1
    tvm_tirx.ir.assert_structural_equal(mod["main"].ret_ty, relax.TensorType((), "int32"))
    tvm_tirx.ir.assert_structural_equal(
        mod["tflite_while_subgraph_1_2"].ret_ty,
        relax.TensorType((), "int32"),
    )


def _build_tflite_two_var_while_model():
    """Build a TFLite WHILE model with two int32 loop-carried scalar tensors."""
    builder = flatbuffers.Builder(1024)

    while_options = _build_while_options(builder, 1, 2)
    one = np.array(1, dtype=np.int32)
    three = np.array(3, dtype=np.int32)

    main_tensors = [
        _build_tensor(builder, 0, [], tensor_type=_tfl_tensor_type.INT32),
        _build_tensor(builder, 1, [], tensor_type=_tfl_tensor_type.INT32),
        _build_tensor(builder, 4, [], tensor_type=_tfl_tensor_type.INT32),
        _build_tensor(builder, 5, [], tensor_type=_tfl_tensor_type.INT32),
    ]
    main_while = _build_operator(
        builder,
        0,
        [0, 1],
        [2, 3],
        builtin_options_type=_tfl_builtin_options.WhileOptions,
        builtin_options=while_options,
    )
    main_subgraph = _build_subgraph(
        builder,
        tensors=main_tensors,
        operators=[main_while],
        inputs=[0, 1],
        outputs=[2, 3],
    )

    cond_tensors = [
        _build_tensor(builder, 0, [], tensor_type=_tfl_tensor_type.INT32),
        _build_tensor(builder, 1, [], tensor_type=_tfl_tensor_type.INT32),
        _build_tensor(builder, 2, [], tensor_type=_tfl_tensor_type.INT32),
        _build_tensor(builder, 4, [], tensor_type=_tfl_tensor_type.BOOL),
    ]
    cond_less = _build_operator(builder, 1, [0, 2], [3])
    cond_subgraph = _build_subgraph(
        builder,
        tensors=cond_tensors,
        operators=[cond_less],
        inputs=[0, 1],
        outputs=[3],
    )

    body_tensors = [
        _build_tensor(builder, 0, [], tensor_type=_tfl_tensor_type.INT32),
        _build_tensor(builder, 1, [], tensor_type=_tfl_tensor_type.INT32),
        _build_tensor(builder, 3, [], tensor_type=_tfl_tensor_type.INT32),
        _build_tensor(builder, 4, [], tensor_type=_tfl_tensor_type.INT32),
        _build_tensor(builder, 5, [], tensor_type=_tfl_tensor_type.INT32),
    ]
    body_add_i = _build_operator(builder, 2, [0, 2], [3])
    body_add_acc = _build_operator(builder, 2, [1, 0], [4])
    body_subgraph = _build_subgraph(
        builder,
        tensors=body_tensors,
        operators=[body_add_i, body_add_acc],
        inputs=[0, 1],
        outputs=[3, 4],
    )

    operator_codes = [
        _build_operator_code(builder, _get_builtin_operator("WHILE")),
        _build_operator_code(builder, _get_builtin_operator("LESS")),
        _build_operator_code(builder, _get_builtin_operator("ADD")),
    ]
    buffers = [
        _build_buffer(builder),
        _build_buffer(builder),
        _build_buffer(builder, three.tobytes()),
        _build_buffer(builder, one.tobytes()),
        _build_buffer(builder),
        _build_buffer(builder),
    ]
    return _finish_tflite_model(
        builder,
        subgraph=main_subgraph,
        extra_subgraphs=[cond_subgraph, body_subgraph],
        operator_codes=operator_codes,
        buffers=buffers,
    )


def test_while_subgraphs_two_loop_vars():
    """Test WHILE tuple loop state with two loop-carried variables."""
    mod = _load_model_from_buffer(_build_tflite_two_var_while_model())

    @I.ir_module
    class Expected:
        @R.function(private=True)
        def tflite_while_cond_subgraph_1(
            tvmgen_tensor_0: R.Tensor((), dtype="int32"),
            tvmgen_tensor_1: R.Tensor((), dtype="int32"),
        ) -> R.Tensor((), dtype="bool"):
            with R.dataflow():
                gv: R.Tensor((), dtype="bool") = R.less(tvmgen_tensor_0, R.const(3, "int32"))
                R.output(gv)
            return gv

        @R.function(private=True)
        def tflite_while_body_subgraph_2(
            tvmgen_tensor_0: R.Tensor((), dtype="int32"),
            tvmgen_tensor_1: R.Tensor((), dtype="int32"),
        ) -> R.Tuple(R.Tensor((), dtype="int32"), R.Tensor((), dtype="int32")):
            with R.dataflow():
                gv: R.Tensor((), dtype="int32") = R.add(tvmgen_tensor_0, R.const(1, "int32"))
                gv1: R.Tensor((), dtype="int32") = R.add(tvmgen_tensor_1, tvmgen_tensor_0)
                gv2: R.Tuple(R.Tensor((), dtype="int32"), R.Tensor((), dtype="int32")) = (
                    gv,
                    gv1,
                )
                R.output(gv2)
            return gv2

        @R.function(private=True)
        def tflite_while_subgraph_1_2(
            tvmgen_tensor_0: R.Tensor((), dtype="int32"),
            tvmgen_tensor_1: R.Tensor((), dtype="int32"),
        ) -> R.Tuple(R.Tensor((), dtype="int32"), R.Tensor((), dtype="int32")):
            cls = Expected
            while_cond: R.Tensor((), dtype="bool") = cls.tflite_while_cond_subgraph_1(
                tvmgen_tensor_0, tvmgen_tensor_1
            )
            if while_cond:
                gv: R.Tuple(R.Tensor((), dtype="int32"), R.Tensor((), dtype="int32")) = (
                    cls.tflite_while_body_subgraph_2(tvmgen_tensor_0, tvmgen_tensor_1)
                )
                gv1: R.Tensor((), dtype="int32") = gv[0]
                gv2: R.Tensor((), dtype="int32") = gv[1]
                gv3: R.Tuple(R.Tensor((), dtype="int32"), R.Tensor((), dtype="int32")) = (
                    cls.tflite_while_subgraph_1_2(gv1, gv2)
                )
                cond_result: R.Tuple(R.Tensor((), dtype="int32"), R.Tensor((), dtype="int32")) = gv3
            else:
                cond_result: R.Tuple(R.Tensor((), dtype="int32"), R.Tensor((), dtype="int32")) = (
                    tvmgen_tensor_0,
                    tvmgen_tensor_1,
                )
            return cond_result

        @R.function
        def main(
            tvmgen_tensor_0: R.Tensor((), dtype="int32"),
            tvmgen_tensor_1: R.Tensor((), dtype="int32"),
        ) -> R.Tuple(R.Tensor((), dtype="int32"), R.Tensor((), dtype="int32")):
            R.func_attr({"num_input": 2})
            cls = Expected
            with R.dataflow():
                lv: R.Tuple(R.Tensor((), dtype="int32"), R.Tensor((), dtype="int32")) = (
                    cls.tflite_while_subgraph_1_2(tvmgen_tensor_0, tvmgen_tensor_1)
                )
                lv1: R.Tensor((), dtype="int32") = lv[0]
                lv2: R.Tensor((), dtype="int32") = lv[1]
                gv: R.Tuple(R.Tensor((), dtype="int32"), R.Tensor((), dtype="int32")) = (
                    lv1,
                    lv2,
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_while_subgraphs_non_bool_condition_unsupported():
    """Test WHILE rejects cond subgraphs that do not return scalar bool."""
    with pytest.raises(NotImplementedError, match="WHILE requires a scalar bool condition"):
        _load_model_from_buffer(_build_tflite_while_model(cond_output_type=_tfl_tensor_type.INT32))


def test_while_subgraphs_invalid_index_unsupported():
    """Test WHILE rejects invalid cond/body subgraph indices before lowering."""
    with pytest.raises(NotImplementedError, match="WHILE requires a valid subgraph index"):
        _load_model_from_buffer(_build_tflite_while_model(cond_subgraph_index=3))


def test_while_subgraphs_zero_loop_vars_unsupported():
    """Test WHILE rejects operators without loop-carried tensors."""
    with pytest.raises(NotImplementedError, match="WHILE requires loop-carried inputs"):
        _load_model_from_buffer(_build_tflite_zero_var_while_model())


def test_while_subgraphs_loop_state_metadata_mismatch_unsupported():
    """Test WHILE rejects loop outputs whose metadata does not match loop inputs."""
    with pytest.raises(NotImplementedError, match="WHILE loop state tensor metadata mismatch"):
        _load_model_from_buffer(
            _build_tflite_while_model(main_output_type=_tfl_tensor_type.FLOAT32)
        )


def test_while_subgraphs_output_count_mismatch_unsupported():
    """Test WHILE rejects body subgraphs whose output arity does not match loop vars."""
    with pytest.raises(NotImplementedError, match="WHILE subgraph output count mismatch"):
        _load_model_from_buffer(_build_tflite_while_model(body_outputs=[]))


def test_while_subgraphs_input_metadata_mismatch_unsupported():
    """Test WHILE rejects cond subgraph inputs whose metadata does not match loop vars."""
    with pytest.raises(NotImplementedError, match="WHILE subgraph input tensor metadata mismatch"):
        _load_model_from_buffer(_build_tflite_while_model(cond_input_type=_tfl_tensor_type.FLOAT32))


def test_while_subgraphs_output_metadata_mismatch_unsupported():
    """Test WHILE rejects body outputs whose metadata does not match loop vars."""
    with pytest.raises(NotImplementedError, match="WHILE subgraph output tensor metadata mismatch"):
        _load_model_from_buffer(
            _build_tflite_while_model(body_output_type=_tfl_tensor_type.FLOAT32)
        )


def _build_tflite_call_once_model(
    init_has_op=False,
    init_subgraph_index=1,
    call_once_inputs=None,
    call_once_outputs=None,
    init_inputs=None,
    init_outputs=None,
):
    """Build a TFLite model with CALL_ONCE and one pass-through output."""
    builder = flatbuffers.Builder(1024)

    call_once_inputs = [] if call_once_inputs is None else call_once_inputs
    call_once_outputs = [] if call_once_outputs is None else call_once_outputs
    init_inputs = [] if init_inputs is None else init_inputs
    init_outputs = [] if init_outputs is None else init_outputs

    call_once_options = _build_call_once_options(builder, init_subgraph_index)
    main_tensors = [_build_tensor(builder, 0, [2, 2])]
    main_call_once = _build_operator(
        builder,
        0,
        call_once_inputs,
        call_once_outputs,
        builtin_options_type=_tfl_builtin_options.CallOnceOptions,
        builtin_options=call_once_options,
    )
    main_subgraph = _build_subgraph(
        builder,
        tensors=main_tensors,
        operators=[main_call_once],
        inputs=[0],
        outputs=[0],
    )

    if init_has_op:
        one = np.array(1.0, dtype=np.float32)
        init_tensors = [
            _build_tensor(builder, 0, [2, 2]),
            _build_tensor(builder, 1, []),
            _build_tensor(builder, 2, [2, 2]),
        ]
        init_op = _build_operator(builder, 1, [0, 1], [2])
        buffers = [
            _build_buffer(builder),
            _build_buffer(builder, one.tobytes()),
            _build_buffer(builder),
        ]
    else:
        init_tensors = (
            [_build_tensor(builder, 0, [2, 2])]
            if len(init_inputs) != 0 or len(init_outputs) != 0
            else []
        )
        init_op = None
        buffers = [_build_buffer(builder)]

    init_subgraph = _build_subgraph(
        builder,
        tensors=init_tensors,
        operators=[] if init_op is None else [init_op],
        inputs=init_inputs,
        outputs=init_outputs,
    )

    operator_codes = [_build_operator_code(builder, _get_builtin_operator("CALL_ONCE"))]
    if init_has_op:
        operator_codes.append(_build_operator_code(builder, _get_builtin_operator("ADD")))
    return _finish_tflite_model(
        builder,
        subgraph=main_subgraph,
        extra_subgraphs=[init_subgraph],
        operator_codes=operator_codes,
        buffers=buffers,
    )


def test_call_once_empty_init_subgraph():
    """Test the no-op CALL_ONCE subset."""
    mod = _load_model_from_buffer(_build_tflite_call_once_model())

    @I.ir_module
    class Expected:
        @R.function
        def main(
            tvmgen_tensor_0: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tensor((2, 2), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="float32") = tvmgen_tensor_0
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_call_once_non_empty_init_subgraph_unsupported():
    """Test CALL_ONCE rejects init subgraphs with side-effect-like bodies."""
    with pytest.raises(NotImplementedError, match="CALL_ONCE"):
        _load_model_from_buffer(_build_tflite_call_once_model(init_has_op=True))


def test_call_once_inputs_outputs_unsupported():
    """Test CALL_ONCE rejects operator inputs and outputs."""
    with pytest.raises(NotImplementedError, match="CALL_ONCE with inputs or outputs"):
        _load_model_from_buffer(
            _build_tflite_call_once_model(call_once_inputs=[0], call_once_outputs=[0])
        )


def test_call_once_init_subgraph_io_unsupported():
    """Test CALL_ONCE rejects init subgraphs with inputs or outputs."""
    with pytest.raises(NotImplementedError, match="CALL_ONCE with non-empty init subgraph I/O"):
        _load_model_from_buffer(_build_tflite_call_once_model(init_inputs=[0], init_outputs=[0]))


def test_call_once_invalid_index_unsupported():
    """Test CALL_ONCE rejects invalid init subgraph indices before lowering."""
    with pytest.raises(NotImplementedError, match="CALL_ONCE requires a valid subgraph index"):
        _load_model_from_buffer(_build_tflite_call_once_model(init_subgraph_index=2))


def _build_tflite_resource_variable_model():
    """Build a model that initializes a resource variable in CALL_ONCE and reads it."""
    builder = flatbuffers.Builder(1024)
    resource_type = _get_resource_tensor_type()
    initial_value = np.array([1.0, 2.0], dtype=np.float32)

    call_once_options = _build_call_once_options(builder, 1)
    main_var_handle_options = _build_var_handle_options(builder)
    main_read_options = _build_empty_builtin_options(builder, "ReadVariableOptions")
    init_var_handle_options = _build_var_handle_options(builder)
    init_assign_options = _build_empty_builtin_options(builder, "AssignVariableOptions")

    resource_tensor = _build_tensor(builder, 0, [], tensor_type=resource_type)
    main_output_tensor = _build_tensor(builder, 0, [2])
    main_call_once = _build_operator(
        builder,
        0,
        [],
        [],
        builtin_options_type=_get_builtin_options_type("CallOnceOptions"),
        builtin_options=call_once_options,
    )
    main_var_handle = _build_operator(
        builder,
        1,
        [],
        [0],
        builtin_options_type=_get_builtin_options_type("VarHandleOptions"),
        builtin_options=main_var_handle_options,
    )
    main_read = _build_operator(
        builder,
        2,
        [0],
        [1],
        builtin_options_type=_get_builtin_options_type("ReadVariableOptions"),
        builtin_options=main_read_options,
    )
    main_subgraph = _build_subgraph(
        builder,
        tensors=[resource_tensor, main_output_tensor],
        operators=[main_call_once, main_var_handle, main_read],
        inputs=[],
        outputs=[1],
    )

    init_resource_tensor = _build_tensor(builder, 0, [], tensor_type=resource_type)
    init_value_tensor = _build_tensor(builder, 1, [2])
    init_var_handle = _build_operator(
        builder,
        1,
        [],
        [0],
        builtin_options_type=_get_builtin_options_type("VarHandleOptions"),
        builtin_options=init_var_handle_options,
    )
    init_assign = _build_operator(
        builder,
        3,
        [0, 1],
        [],
        builtin_options_type=_get_builtin_options_type("AssignVariableOptions"),
        builtin_options=init_assign_options,
    )
    init_subgraph = _build_subgraph(
        builder,
        tensors=[init_resource_tensor, init_value_tensor],
        operators=[init_var_handle, init_assign],
        inputs=[],
        outputs=[],
    )

    operator_codes = [
        _build_operator_code(builder, _get_builtin_operator("CALL_ONCE")),
        _build_operator_code(builder, _get_builtin_operator("VAR_HANDLE")),
        _build_operator_code(builder, _get_builtin_operator("READ_VARIABLE")),
        _build_operator_code(builder, _get_builtin_operator("ASSIGN_VARIABLE")),
    ]
    buffers = [_build_buffer(builder), _build_buffer(builder, initial_value.tobytes())]
    return _finish_tflite_model(
        builder,
        subgraph=main_subgraph,
        extra_subgraphs=[init_subgraph],
        operator_codes=operator_codes,
        buffers=buffers,
    )


def _build_tflite_resource_assign_in_main_model():
    """Build a model that attempts to assign a resource variable in the main subgraph."""
    builder = flatbuffers.Builder(1024)
    resource_type = _get_resource_tensor_type()
    value = np.array([1.0, 2.0], dtype=np.float32)

    var_handle_options = _build_var_handle_options(builder)
    assign_options = _build_empty_builtin_options(builder, "AssignVariableOptions")
    resource_tensor = _build_tensor(builder, 0, [], tensor_type=resource_type)
    value_tensor = _build_tensor(builder, 1, [2])
    var_handle = _build_operator(
        builder,
        0,
        [],
        [0],
        builtin_options_type=_get_builtin_options_type("VarHandleOptions"),
        builtin_options=var_handle_options,
    )
    assign = _build_operator(
        builder,
        1,
        [0, 1],
        [],
        builtin_options_type=_get_builtin_options_type("AssignVariableOptions"),
        builtin_options=assign_options,
    )
    main_subgraph = _build_subgraph(
        builder,
        tensors=[resource_tensor, value_tensor],
        operators=[var_handle, assign],
        inputs=[],
        outputs=[1],
    )
    operator_codes = [
        _build_operator_code(builder, _get_builtin_operator("VAR_HANDLE")),
        _build_operator_code(builder, _get_builtin_operator("ASSIGN_VARIABLE")),
    ]
    buffers = [_build_buffer(builder), _build_buffer(builder, value.tobytes())]
    return _finish_tflite_model(
        builder,
        subgraph=main_subgraph,
        operator_codes=operator_codes,
        buffers=buffers,
    )


def _build_tflite_resource_read_uninitialized_model():
    """Build a model that reads a resource variable without CALL_ONCE initialization."""
    builder = flatbuffers.Builder(1024)
    resource_type = _get_resource_tensor_type()

    var_handle_options = _build_var_handle_options(builder)
    read_options = _build_empty_builtin_options(builder, "ReadVariableOptions")
    resource_tensor = _build_tensor(builder, 0, [], tensor_type=resource_type)
    output_tensor = _build_tensor(builder, 0, [2])
    var_handle = _build_operator(
        builder,
        0,
        [],
        [0],
        builtin_options_type=_get_builtin_options_type("VarHandleOptions"),
        builtin_options=var_handle_options,
    )
    read = _build_operator(
        builder,
        1,
        [0],
        [1],
        builtin_options_type=_get_builtin_options_type("ReadVariableOptions"),
        builtin_options=read_options,
    )
    main_subgraph = _build_subgraph(
        builder,
        tensors=[resource_tensor, output_tensor],
        operators=[var_handle, read],
        inputs=[],
        outputs=[1],
    )
    operator_codes = [
        _build_operator_code(builder, _get_builtin_operator("VAR_HANDLE")),
        _build_operator_code(builder, _get_builtin_operator("READ_VARIABLE")),
    ]
    return _finish_tflite_model(
        builder,
        subgraph=main_subgraph,
        operator_codes=operator_codes,
        buffers=[_build_buffer(builder)],
    )


def _build_tflite_hashtable_find_string_to_int64_model(
    query_values=None,
    query_shape=None,
    default_values=None,
    default_shape=None,
    table_keys=None,
    table_values=None,
    query_is_input=False,
):
    """Build a static string-to-int64 HASHTABLE_FIND model."""
    builder = flatbuffers.Builder(1024)
    resource_type = _get_resource_tensor_type()
    string_type = _get_string_tensor_type()
    query_values = ["alpha", "missing", "beta"] if query_values is None else query_values
    query_shape = [len(query_values)] if query_shape is None else query_shape
    default_shape = [] if default_shape is None else default_shape
    table_keys = ["alpha", "beta", "gamma"] if table_keys is None else table_keys
    table_values = (
        np.array([10, 20, 30], dtype=np.int64)
        if table_values is None
        else np.array(table_values, dtype=np.int64)
    )
    default_values = (
        np.array(-1, dtype=np.int64)
        if default_values is None
        else np.array(default_values, dtype=np.int64)
    )
    query_buffer = _build_tflite_string_buffer(query_values)
    table_key_buffer = _build_tflite_string_buffer(table_keys)

    call_once_options = _build_call_once_options(builder, 1)
    main_table_options = _build_hashtable_options(
        builder,
        table_id=0,
        key_dtype=string_type,
        value_dtype=_tfl_tensor_type.INT64,
    )
    find_options = _build_empty_builtin_options(builder, "HashtableFindOptions")
    init_table_options = _build_hashtable_options(
        builder,
        table_id=0,
        key_dtype=string_type,
        value_dtype=_tfl_tensor_type.INT64,
    )
    import_options = _build_empty_builtin_options(builder, "HashtableImportOptions")

    query_buffer_idx = 0 if query_is_input else 1
    query_tensor = _build_tensor(builder, query_buffer_idx, query_shape, tensor_type=string_type)
    table_tensor = _build_tensor(builder, 0, [1], tensor_type=resource_type)
    default_tensor = _build_tensor(builder, 2, default_shape, tensor_type=_tfl_tensor_type.INT64)
    output_tensor = _build_tensor(builder, 0, query_shape, tensor_type=_tfl_tensor_type.INT64)
    main_call_once = _build_operator(
        builder,
        0,
        [],
        [],
        builtin_options_type=_get_builtin_options_type("CallOnceOptions"),
        builtin_options=call_once_options,
    )
    main_hashtable = _build_operator(
        builder,
        1,
        [],
        [1],
        builtin_options_type=_get_builtin_options_type("HashtableOptions"),
        builtin_options=main_table_options,
    )
    main_find = _build_operator(
        builder,
        2,
        [1, 0, 2],
        [3],
        builtin_options_type=_get_builtin_options_type("HashtableFindOptions"),
        builtin_options=find_options,
    )
    main_subgraph = _build_subgraph(
        builder,
        tensors=[query_tensor, table_tensor, default_tensor, output_tensor],
        operators=[main_call_once, main_hashtable, main_find],
        inputs=[0] if query_is_input else [],
        outputs=[3],
    )

    init_table_tensor = _build_tensor(builder, 0, [1], tensor_type=resource_type)
    init_keys_tensor = _build_tensor(builder, 3, [len(table_keys)], tensor_type=string_type)
    init_values_tensor = _build_tensor(
        builder,
        4,
        [len(table_values)],
        tensor_type=_tfl_tensor_type.INT64,
    )
    init_hashtable = _build_operator(
        builder,
        1,
        [],
        [0],
        builtin_options_type=_get_builtin_options_type("HashtableOptions"),
        builtin_options=init_table_options,
    )
    init_import = _build_operator(
        builder,
        3,
        [0, 1, 2],
        [],
        builtin_options_type=_get_builtin_options_type("HashtableImportOptions"),
        builtin_options=import_options,
    )
    init_subgraph = _build_subgraph(
        builder,
        tensors=[init_table_tensor, init_keys_tensor, init_values_tensor],
        operators=[init_hashtable, init_import],
        inputs=[],
        outputs=[],
    )

    operator_codes = [
        _build_operator_code(builder, _get_builtin_operator("CALL_ONCE")),
        _build_operator_code(builder, _get_builtin_operator("HASHTABLE")),
        _build_operator_code(builder, _get_builtin_operator("HASHTABLE_FIND")),
        _build_operator_code(builder, _get_builtin_operator("HASHTABLE_IMPORT")),
    ]
    buffers = [
        _build_buffer(builder),
        _build_buffer(builder, b"" if query_is_input else query_buffer),
        _build_buffer(builder, default_values.tobytes()),
        _build_buffer(builder, table_key_buffer),
        _build_buffer(builder, table_values.tobytes()),
    ]
    return _finish_tflite_model(
        builder,
        subgraph=main_subgraph,
        extra_subgraphs=[init_subgraph],
        operator_codes=operator_codes,
        buffers=buffers,
    )


def _build_tflite_hashtable_find_int64_to_string_model():
    """Build a static int64-to-string HASHTABLE_FIND model."""
    builder = flatbuffers.Builder(1024)
    resource_type = _get_resource_tensor_type()
    string_type = _get_string_tensor_type()
    query_values = np.array([10, 30], dtype=np.int64)
    table_keys = np.array([10, 20], dtype=np.int64)
    table_values = _build_tflite_string_buffer(["ten", "twenty"])
    default_value = _build_tflite_string_buffer(["missing"])

    call_once_options = _build_call_once_options(builder, 1)
    main_table_options = _build_hashtable_options(builder, table_id=0)
    find_options = _build_empty_builtin_options(builder, "HashtableFindOptions")
    init_table_options = _build_hashtable_options(builder, table_id=0)
    import_options = _build_empty_builtin_options(builder, "HashtableImportOptions")

    query_tensor = _build_tensor(builder, 1, [2], tensor_type=_tfl_tensor_type.INT64)
    table_tensor = _build_tensor(builder, 0, [1], tensor_type=resource_type)
    default_tensor = _build_tensor(builder, 2, [], tensor_type=string_type)
    output_tensor = _build_tensor(builder, 0, [2], tensor_type=string_type)
    main_call_once = _build_operator(
        builder,
        0,
        [],
        [],
        builtin_options_type=_get_builtin_options_type("CallOnceOptions"),
        builtin_options=call_once_options,
    )
    main_hashtable = _build_operator(
        builder,
        1,
        [],
        [1],
        builtin_options_type=_get_builtin_options_type("HashtableOptions"),
        builtin_options=main_table_options,
    )
    main_find = _build_operator(
        builder,
        2,
        [1, 0, 2],
        [3],
        builtin_options_type=_get_builtin_options_type("HashtableFindOptions"),
        builtin_options=find_options,
    )
    main_subgraph = _build_subgraph(
        builder,
        tensors=[query_tensor, table_tensor, default_tensor, output_tensor],
        operators=[main_call_once, main_hashtable, main_find],
        inputs=[],
        outputs=[3],
    )

    init_table_tensor = _build_tensor(builder, 0, [1], tensor_type=resource_type)
    init_keys_tensor = _build_tensor(builder, 3, [2], tensor_type=_tfl_tensor_type.INT64)
    init_values_tensor = _build_tensor(builder, 4, [2], tensor_type=string_type)
    init_hashtable = _build_operator(
        builder,
        1,
        [],
        [0],
        builtin_options_type=_get_builtin_options_type("HashtableOptions"),
        builtin_options=init_table_options,
    )
    init_import = _build_operator(
        builder,
        3,
        [0, 1, 2],
        [],
        builtin_options_type=_get_builtin_options_type("HashtableImportOptions"),
        builtin_options=import_options,
    )
    init_subgraph = _build_subgraph(
        builder,
        tensors=[init_table_tensor, init_keys_tensor, init_values_tensor],
        operators=[init_hashtable, init_import],
        inputs=[],
        outputs=[],
    )

    operator_codes = [
        _build_operator_code(builder, _get_builtin_operator("CALL_ONCE")),
        _build_operator_code(builder, _get_builtin_operator("HASHTABLE")),
        _build_operator_code(builder, _get_builtin_operator("HASHTABLE_FIND")),
        _build_operator_code(builder, _get_builtin_operator("HASHTABLE_IMPORT")),
    ]
    buffers = [
        _build_buffer(builder),
        _build_buffer(builder, query_values.tobytes()),
        _build_buffer(builder, default_value),
        _build_buffer(builder, table_keys.tobytes()),
        _build_buffer(builder, table_values),
    ]
    return _finish_tflite_model(
        builder,
        subgraph=main_subgraph,
        extra_subgraphs=[init_subgraph],
        operator_codes=operator_codes,
        buffers=buffers,
    )


def _build_tflite_hashtable_size_model():
    """Build a model that imports a static hashtable and returns its size."""
    builder = flatbuffers.Builder(1024)
    resource_type = _get_resource_tensor_type()
    string_type = _get_string_tensor_type()
    table_keys = np.array([10, 20], dtype=np.int64)
    table_values = _build_tflite_string_buffer(["one hundred", "two hundred"])

    call_once_options = _build_call_once_options(builder, 1)
    main_table_options = _build_hashtable_options(builder, table_id=0)
    size_options = _build_empty_builtin_options(builder, "HashtableSizeOptions")
    init_table_options = _build_hashtable_options(builder, table_id=0)
    import_options = _build_empty_builtin_options(builder, "HashtableImportOptions")

    table_tensor = _build_tensor(builder, 0, [1], tensor_type=resource_type)
    size_tensor = _build_tensor(builder, 0, [1], tensor_type=_tfl_tensor_type.INT64)
    main_call_once = _build_operator(
        builder,
        0,
        [],
        [],
        builtin_options_type=_get_builtin_options_type("CallOnceOptions"),
        builtin_options=call_once_options,
    )
    main_hashtable = _build_operator(
        builder,
        1,
        [],
        [0],
        builtin_options_type=_get_builtin_options_type("HashtableOptions"),
        builtin_options=main_table_options,
    )
    main_size = _build_operator(
        builder,
        2,
        [0],
        [1],
        builtin_options_type=_get_builtin_options_type("HashtableSizeOptions"),
        builtin_options=size_options,
    )
    main_subgraph = _build_subgraph(
        builder,
        tensors=[table_tensor, size_tensor],
        operators=[main_call_once, main_hashtable, main_size],
        inputs=[],
        outputs=[1],
    )

    init_table_tensor = _build_tensor(builder, 0, [1], tensor_type=resource_type)
    init_keys_tensor = _build_tensor(builder, 1, [2], tensor_type=_tfl_tensor_type.INT64)
    init_values_tensor = _build_tensor(builder, 2, [2], tensor_type=string_type)
    init_hashtable = _build_operator(
        builder,
        1,
        [],
        [0],
        builtin_options_type=_get_builtin_options_type("HashtableOptions"),
        builtin_options=init_table_options,
    )
    init_import = _build_operator(
        builder,
        3,
        [0, 1, 2],
        [],
        builtin_options_type=_get_builtin_options_type("HashtableImportOptions"),
        builtin_options=import_options,
    )
    init_subgraph = _build_subgraph(
        builder,
        tensors=[init_table_tensor, init_keys_tensor, init_values_tensor],
        operators=[init_hashtable, init_import],
        inputs=[],
        outputs=[],
    )

    operator_codes = [
        _build_operator_code(builder, _get_builtin_operator("CALL_ONCE")),
        _build_operator_code(builder, _get_builtin_operator("HASHTABLE")),
        _build_operator_code(builder, _get_builtin_operator("HASHTABLE_SIZE")),
        _build_operator_code(builder, _get_builtin_operator("HASHTABLE_IMPORT")),
    ]
    buffers = [
        _build_buffer(builder),
        _build_buffer(builder, table_keys.tobytes()),
        _build_buffer(builder, table_values),
    ]
    return _finish_tflite_model(
        builder,
        subgraph=main_subgraph,
        extra_subgraphs=[init_subgraph],
        operator_codes=operator_codes,
        buffers=buffers,
    )


def _build_tflite_hashtable_import_in_main_model():
    """Build a model that attempts to import hashtable values in the main subgraph."""
    builder = flatbuffers.Builder(1024)
    resource_type = _get_resource_tensor_type()
    string_type = _get_string_tensor_type()
    table_keys = np.array([10, 20], dtype=np.int64)
    table_values = _build_tflite_string_buffer(["one hundred", "two hundred"])

    table_options = _build_hashtable_options(builder, table_id=0)
    import_options = _build_empty_builtin_options(builder, "HashtableImportOptions")

    table_tensor = _build_tensor(builder, 0, [1], tensor_type=resource_type)
    keys_tensor = _build_tensor(builder, 1, [2], tensor_type=_tfl_tensor_type.INT64)
    values_tensor = _build_tensor(builder, 2, [2], tensor_type=string_type)
    hashtable = _build_operator(
        builder,
        0,
        [],
        [0],
        builtin_options_type=_get_builtin_options_type("HashtableOptions"),
        builtin_options=table_options,
    )
    hashtable_import = _build_operator(
        builder,
        1,
        [0, 1, 2],
        [],
        builtin_options_type=_get_builtin_options_type("HashtableImportOptions"),
        builtin_options=import_options,
    )
    main_subgraph = _build_subgraph(
        builder,
        tensors=[table_tensor, keys_tensor, values_tensor],
        operators=[hashtable, hashtable_import],
        inputs=[],
        outputs=[2],
    )
    operator_codes = [
        _build_operator_code(builder, _get_builtin_operator("HASHTABLE")),
        _build_operator_code(builder, _get_builtin_operator("HASHTABLE_IMPORT")),
    ]
    buffers = [
        _build_buffer(builder),
        _build_buffer(builder, table_keys.tobytes()),
        _build_buffer(builder, table_values),
    ]
    return _finish_tflite_model(
        builder,
        subgraph=main_subgraph,
        operator_codes=operator_codes,
        buffers=buffers,
    )


def _build_tflite_hashtable_size_uninitialized_model():
    """Build a model that queries the size of a hashtable without importing values."""
    builder = flatbuffers.Builder(1024)
    resource_type = _get_resource_tensor_type()

    table_options = _build_hashtable_options(builder, table_id=0)
    size_options = _build_empty_builtin_options(builder, "HashtableSizeOptions")
    table_tensor = _build_tensor(builder, 0, [1], tensor_type=resource_type)
    size_tensor = _build_tensor(builder, 0, [1], tensor_type=_tfl_tensor_type.INT64)
    hashtable = _build_operator(
        builder,
        0,
        [],
        [0],
        builtin_options_type=_get_builtin_options_type("HashtableOptions"),
        builtin_options=table_options,
    )
    hashtable_size = _build_operator(
        builder,
        1,
        [0],
        [1],
        builtin_options_type=_get_builtin_options_type("HashtableSizeOptions"),
        builtin_options=size_options,
    )
    main_subgraph = _build_subgraph(
        builder,
        tensors=[table_tensor, size_tensor],
        operators=[hashtable, hashtable_size],
        inputs=[],
        outputs=[1],
    )
    operator_codes = [
        _build_operator_code(builder, _get_builtin_operator("HASHTABLE")),
        _build_operator_code(builder, _get_builtin_operator("HASHTABLE_SIZE")),
    ]
    return _finish_tflite_model(
        builder,
        subgraph=main_subgraph,
        operator_codes=operator_codes,
        buffers=[_build_buffer(builder)],
    )


def _build_tflite_embedding_lookup_sparse_model(
    combiner, indices_data, dense_shape_data, weights_data=None
):
    builder = flatbuffers.Builder(4096)

    ids_data = np.array([1, 3, 0], dtype=np.int32)
    indices_data = np.array(indices_data, dtype=np.int32)
    dense_shape_data = np.array(dense_shape_data, dtype=np.int32)
    weights_data = (
        np.array([1.0, 2.0, 4.0], dtype=np.float32)
        if weights_data is None
        else np.array(weights_data, dtype=np.float32)
    )
    params_data = np.array(
        [
            [[0.00, 0.01], [0.10, 0.11], [0.20, 0.21]],
            [[1.00, 1.01], [1.10, 1.11], [1.20, 1.21]],
            [[2.00, 2.01], [2.10, 2.11], [2.20, 2.21]],
            [[3.00, 3.01], [3.10, 3.11], [3.20, 3.21]],
        ],
        dtype=np.float32,
    )

    output_shape = dense_shape_data[:-1].tolist() + list(params_data.shape[1:])
    sparse_options = _build_embedding_lookup_sparse_options(builder, combiner)

    ids_tensor = _build_tensor(builder, 0, list(ids_data.shape), tensor_type=_tfl_tensor_type.INT32)
    indices_tensor = _build_tensor(
        builder, 1, list(indices_data.shape), tensor_type=_tfl_tensor_type.INT32
    )
    dense_shape_tensor = _build_tensor(
        builder, 2, list(dense_shape_data.shape), tensor_type=_tfl_tensor_type.INT32
    )
    weights_tensor = _build_tensor(
        builder, 3, list(weights_data.shape), tensor_type=_tfl_tensor_type.FLOAT32
    )
    params_tensor = _build_tensor(
        builder, 4, list(params_data.shape), tensor_type=_tfl_tensor_type.FLOAT32
    )
    output_tensor = _build_tensor(builder, 5, output_shape, tensor_type=_tfl_tensor_type.FLOAT32)

    sparse_op = _build_operator(
        builder,
        0,
        [0, 1, 2, 3, 4],
        [5],
        builtin_options_type=_get_builtin_options_type("EmbeddingLookupSparseOptions"),
        builtin_options=sparse_options,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=[
            ids_tensor,
            indices_tensor,
            dense_shape_tensor,
            weights_tensor,
            params_tensor,
            output_tensor,
        ],
        operators=[sparse_op],
        inputs=[],
        outputs=[5],
    )
    operator_codes = [
        _build_operator_code(builder, _get_builtin_operator("EMBEDDING_LOOKUP_SPARSE"))
    ]
    buffers = [
        _build_buffer(builder, ids_data.tobytes()),
        _build_buffer(builder, indices_data.tobytes()),
        _build_buffer(builder, dense_shape_data.tobytes()),
        _build_buffer(builder, weights_data.tobytes()),
        _build_buffer(builder, params_data.tobytes()),
        _build_buffer(builder),
    ]
    return _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=operator_codes,
        buffers=buffers,
    )


def _build_tflite_hashtable_lookup_model(*, value_shape, value_type=None):
    """Build a model containing one HASHTABLE_LOOKUP operator."""
    builder = flatbuffers.Builder(1024)

    value_type = _tfl_tensor_type.FLOAT32 if value_type is None else value_type

    lookup_tensor = _build_tensor(builder, 0, [4], tensor_type=_tfl_tensor_type.INT32)
    key_tensor = _build_tensor(builder, 1, [3], tensor_type=_tfl_tensor_type.INT32)
    value_tensor = _build_tensor(builder, 2, value_shape, tensor_type=value_type)
    output_tensor = _build_tensor(builder, 3, [4, *value_shape[1:]], tensor_type=value_type)
    hits_tensor = _build_tensor(builder, 4, [4], tensor_type=_tfl_tensor_type.UINT8)

    hashtable_lookup = _build_operator(builder, 0, [0, 1, 2], [3, 4])
    main_subgraph = _build_subgraph(
        builder,
        tensors=[lookup_tensor, key_tensor, value_tensor, output_tensor, hits_tensor],
        operators=[hashtable_lookup],
        inputs=[0, 1, 2],
        outputs=[3, 4],
    )
    operator_codes = [_build_operator_code(builder, _get_builtin_operator("HASHTABLE_LOOKUP"))]
    buffers = [_build_buffer(builder) for _ in range(5)]
    return _finish_tflite_model(
        builder,
        subgraph=main_subgraph,
        operator_codes=operator_codes,
        buffers=buffers,
    )


def test_resource_variable_call_once_init_read():
    """Test reading a resource variable initialized by a supported CALL_ONCE subgraph."""
    mod = _load_model_from_buffer(_build_tflite_resource_variable_model())

    @I.ir_module
    class Expected:
        @R.function
        def main() -> R.Tensor((2,), dtype="float32"):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                gv: R.Tensor((2,), dtype="float32") = R.const([1.0, 2.0], "float32")
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_assign_variable_main_subgraph_unsupported():
    """Test ASSIGN_VARIABLE remains unsupported outside CALL_ONCE initialization."""
    with pytest.raises(NotImplementedError, match="ASSIGN_VARIABLE outside CALL_ONCE"):
        _load_model_from_buffer(_build_tflite_resource_assign_in_main_model())


def test_read_variable_uninitialized_unsupported():
    """Test READ_VARIABLE rejects resource handles without supported initialization."""
    with pytest.raises(NotImplementedError, match="READ_VARIABLE requires a resource"):
        _load_model_from_buffer(_build_tflite_resource_read_uninitialized_model())


def test_hashtable_call_once_import_find_string_to_int64():
    """Test HASHTABLE_FIND for a static string-to-int64 table."""
    mod = _load_model_from_buffer(_build_tflite_hashtable_find_string_to_int64_model())

    @I.ir_module
    class Expected:
        @R.function
        def main() -> R.Tensor((3,), dtype="int64"):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                gv: R.Tensor((3,), dtype="int64") = R.const([10, -1, 20], "int64")
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_hashtable_call_once_import_find_string_to_int64_2d_query():
    """Test HASHTABLE_FIND preserves the static query shape."""
    mod = _load_model_from_buffer(
        _build_tflite_hashtable_find_string_to_int64_model(
            query_values=["alpha", "beta", "missing", "gamma"],
            query_shape=[2, 2],
        )
    )

    @I.ir_module
    class Expected:
        @R.function
        def main() -> R.Tensor((2, 2), dtype="int64"):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="int64") = R.const([[10, 20], [-1, 30]], "int64")
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_hashtable_call_once_import_find_int64_to_string_unsupported():
    """Test HASHTABLE_FIND rejects int64-to-string tables until string outputs exist."""
    with pytest.raises(NotImplementedError, match="string -> int64"):
        _load_model_from_buffer(_build_tflite_hashtable_find_int64_to_string_model())


def test_hashtable_call_once_import_find_runtime_query_unsupported():
    """Test HASHTABLE_FIND rejects runtime string queries."""
    with pytest.raises(NotImplementedError, match="string queries|STRING graph inputs"):
        _load_model_from_buffer(
            _build_tflite_hashtable_find_string_to_int64_model(query_is_input=True)
        )


def test_hashtable_call_once_import_duplicate_keys_unsupported():
    """Test HASHTABLE_IMPORT rejects duplicate static keys."""
    with pytest.raises(NotImplementedError, match="duplicate keys"):
        _load_model_from_buffer(
            _build_tflite_hashtable_find_string_to_int64_model(
                table_keys=["alpha", "alpha"], table_values=[10, 20]
            )
        )


def test_hashtable_call_once_import_size():
    """Test HASHTABLE_SIZE for a table initialized by a supported CALL_ONCE subgraph."""
    mod = _load_model_from_buffer(_build_tflite_hashtable_size_model())

    @I.ir_module
    class Expected:
        @R.function
        def main() -> R.Tensor((1,), dtype="int64"):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                gv: R.Tensor((1,), dtype="int64") = R.const([2], "int64")
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_hashtable_import_main_subgraph_unsupported():
    """Test HASHTABLE_IMPORT remains unsupported outside CALL_ONCE initialization."""
    with pytest.raises(NotImplementedError, match="HASHTABLE_IMPORT outside CALL_ONCE"):
        _load_model_from_buffer(_build_tflite_hashtable_import_in_main_model())


def test_hashtable_size_uninitialized_unsupported():
    """Test HASHTABLE_SIZE rejects tables without supported initialization."""
    with pytest.raises(NotImplementedError, match="HASHTABLE_SIZE requires a table"):
        _load_model_from_buffer(_build_tflite_hashtable_size_uninitialized_model())


def test_embedding_lookup_sparse_sum():
    from tflite.CombinerType import CombinerType

    mod = _load_model_from_buffer(
        _build_tflite_embedding_lookup_sparse_model(
            CombinerType.SUM,
            indices_data=[[0, 0], [2, 0], [2, 1]],
            dense_shape_data=[3, 2],
        )
    )

    out = _run_no_input_module(mod)
    expected = np.array(
        [
            [[1.00, 1.01], [1.10, 1.11], [1.20, 1.21]],
            [[0.00, 0.00], [0.00, 0.00], [0.00, 0.00]],
            [[6.00, 6.06], [6.60, 6.66], [7.20, 7.26]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_embedding_lookup_sparse_mean():
    from tflite.CombinerType import CombinerType

    mod = _load_model_from_buffer(
        _build_tflite_embedding_lookup_sparse_model(
            CombinerType.MEAN,
            indices_data=[[0, 0], [2, 0], [2, 1]],
            dense_shape_data=[3, 2],
        )
    )

    out = _run_no_input_module(mod)
    expected = np.array(
        [
            [[1.00, 1.01], [1.10, 1.11], [1.20, 1.21]],
            [[0.00, 0.00], [0.00, 0.00], [0.00, 0.00]],
            [[1.00, 1.01], [1.10, 1.11], [1.20, 1.21]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_embedding_lookup_sparse_mean_negative_weights():
    from tflite.CombinerType import CombinerType

    mod = _load_model_from_buffer(
        _build_tflite_embedding_lookup_sparse_model(
            CombinerType.MEAN,
            indices_data=[[0, 0], [0, 1], [2, 0]],
            dense_shape_data=[3, 2],
            weights_data=[1.0, -2.0, 0.0],
        )
    )

    (output,) = (_run_no_input_module(mod),)
    expected = np.array(
        [
            [[5.0, 5.01], [5.1, 5.11], [5.2, 5.21]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-5, equal_nan=True)


def test_embedding_lookup_sparse_sqrtn():
    from tflite.CombinerType import CombinerType

    mod = _load_model_from_buffer(
        _build_tflite_embedding_lookup_sparse_model(
            CombinerType.SQRTN,
            indices_data=[[0, 0], [2, 0], [2, 1]],
            dense_shape_data=[3, 2],
        )
    )

    out = _run_no_input_module(mod)
    scale = np.sqrt(20.0).astype("float32")
    expected = np.array(
        [
            [[1.00, 1.01], [1.10, 1.11], [1.20, 1.21]],
            [[0.00, 0.00], [0.00, 0.00], [0.00, 0.00]],
            [
                [6.00 / scale, 6.06 / scale],
                [6.60 / scale, 6.66 / scale],
                [7.20 / scale, 7.26 / scale],
            ],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_embedding_lookup_sparse_indices_3d():
    from tflite.CombinerType import CombinerType

    mod = _load_model_from_buffer(
        _build_tflite_embedding_lookup_sparse_model(
            CombinerType.SUM,
            indices_data=[[0, 0, 0], [2, 0, 0], [2, 0, 1]],
            dense_shape_data=[3, 2, 2],
        )
    )

    out = _run_no_input_module(mod)
    expected = np.zeros((3, 2, 3, 2), dtype=np.float32)
    expected[0, 0] = np.array([[1.00, 1.01], [1.10, 1.11], [1.20, 1.21]], dtype=np.float32)
    expected[2, 0] = np.array([[6.00, 6.06], [6.60, 6.66], [7.20, 7.26]], dtype=np.float32)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_hashtable_lookup_1d_value():
    mod = _load_model_from_buffer(_build_tflite_hashtable_lookup_model(value_shape=[3]))

    output, hits = _run_module(
        mod,
        np.array([1234, -292, -11, 0], dtype=np.int32),
        np.array([-11, 0, 1234], dtype=np.int32),
        np.array([0.0, 0.1, 0.4], dtype=np.float32),
    )

    np.testing.assert_allclose(output, np.array([0.4, 0.0, 0.0, 0.1], dtype=np.float32))
    np.testing.assert_array_equal(hits, np.array([1, 0, 1, 1], dtype=np.uint8))


def test_hashtable_lookup_2d_value():
    mod = _load_model_from_buffer(_build_tflite_hashtable_lookup_model(value_shape=[3, 2]))

    output, hits = _run_module(
        mod,
        np.array([1234, -292, -11, 0], dtype=np.int32),
        np.array([-11, 0, 1234], dtype=np.int32),
        np.array([[0.0, 0.1], [1.0, 1.1], [2.0, 2.1]], dtype=np.float32),
    )

    np.testing.assert_allclose(
        output,
        np.array(
            [
                [2.0, 2.1],
                [0.0, 0.0],
                [0.0, 0.1],
                [1.0, 1.1],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(hits, np.array([1, 0, 1, 1], dtype=np.uint8))


def test_hashtable_lookup_string_value_unsupported():
    string_type = _get_string_tensor_type()
    with pytest.raises(NotImplementedError, match="STRING graph inputs"):
        _load_model_from_buffer(
            _build_tflite_hashtable_lookup_model(value_shape=[3], value_type=string_type)
        )


def _get_stablehlo_builtin_operator(builtin_name):
    if not hasattr(_tfl_builtin_operator, builtin_name):
        pytest.skip(f"TFLite schema does not provide BuiltinOperator.{builtin_name}")
    return getattr(_tfl_builtin_operator, builtin_name)


def _build_stablehlo_model(*, builtin_name, input_count):
    """Build a minimal TFLite model containing one StableHLO builtin operator."""
    builder = flatbuffers.Builder(1024)
    shape = [2, 2]
    output_tensor_idx = input_count
    builtin_op = _get_stablehlo_builtin_operator(builtin_name)

    tensors = [_build_tensor(builder, buffer_idx, shape) for buffer_idx in range(input_count + 1)]
    stablehlo_op = _build_operator(
        builder,
        0,
        list(range(input_count)),
        [output_tensor_idx],
    )
    subgraph = _build_subgraph(
        builder,
        tensors=tensors,
        operators=[stablehlo_op],
        inputs=list(range(input_count)),
        outputs=[output_tensor_idx],
    )
    operator_codes = [_build_operator_code(builder, builtin_op)]
    buffers = [_build_buffer(builder) for _ in range(input_count + 1)]
    return _finish_tflite_model(
        builder, subgraph=subgraph, operator_codes=operator_codes, buffers=buffers
    )


def _build_stablehlo_model_with_unused_subgraph():
    """Build a StableHLO model with an unused extra subgraph."""
    builder = flatbuffers.Builder(1024)
    builtin_op = _get_stablehlo_builtin_operator("STABLEHLO_ADD")

    main_tensors = [_build_tensor(builder, buffer_idx, [2, 2]) for buffer_idx in range(3)]
    main_op = _build_operator(builder, 0, [0, 1], [2])
    main_subgraph = _build_subgraph(
        builder,
        tensors=main_tensors,
        operators=[main_op],
        inputs=[0, 1],
        outputs=[2],
    )

    # Give the unused subgraph a conflicting input tensor name and different
    # shape. from_tflite should infer the main function input shape only from
    # Subgraphs(0).
    extra_tensors = [_build_tensor(builder, buffer_idx, [4, 4]) for buffer_idx in range(3, 6)]
    extra_op = _build_operator(builder, 0, [0, 1], [2])
    extra_subgraph = _build_subgraph(
        builder,
        tensors=extra_tensors,
        operators=[extra_op],
        inputs=[0, 1],
        outputs=[2],
    )

    operator_codes = [_build_operator_code(builder, builtin_op)]
    buffers = [_build_buffer(builder) for _ in range(6)]
    return _finish_tflite_model(
        builder,
        subgraph=main_subgraph,
        extra_subgraphs=[extra_subgraph],
        operator_codes=operator_codes,
        buffers=buffers,
    )


def _build_stablehlo_reduce_model(reducer_name, init_value):
    """Build a single-input STABLEHLO_REDUCE model with a binary reducer body."""
    builder = flatbuffers.Builder(1024)

    dimensions_vec = _tflite_int64_vector(
        builder,
        _tfl_stablehlo_reduce_opts.StablehloReduceOptionsStartDimensionsVector,
        [1],
    )
    _tfl_stablehlo_reduce_opts.StablehloReduceOptionsStart(builder)
    _tfl_stablehlo_reduce_opts.StablehloReduceOptionsAddDimensions(builder, dimensions_vec)
    _tfl_stablehlo_reduce_opts.StablehloReduceOptionsAddBodySubgraphIndex(builder, 1)
    reduce_opts = _tfl_stablehlo_reduce_opts.StablehloReduceOptionsEnd(builder)

    reduce_builtin = _get_stablehlo_builtin_operator("STABLEHLO_REDUCE")
    reducer_builtin = _get_stablehlo_builtin_operator(reducer_name)
    reduce_code = _build_operator_code(builder, reduce_builtin)
    reducer_code = _build_operator_code(builder, reducer_builtin)

    main_tensors = [
        _build_tensor(builder, 0, [2, 3]),
        _build_tensor(builder, 1, []),
        _build_tensor(builder, 2, [2]),
    ]
    reduce_op = _build_operator(
        builder,
        0,
        [0, 1],
        [2],
        builtin_options2_type=_tfl_builtin_options2.StablehloReduceOptions,
        builtin_options2=reduce_opts,
    )
    main_subgraph = _build_subgraph(
        builder,
        tensors=main_tensors,
        operators=[reduce_op],
        inputs=[0],
        outputs=[2],
    )

    body_tensors = [_build_tensor(builder, buffer_idx, []) for buffer_idx in range(3, 6)]
    reducer_op = _build_operator(builder, 1, [0, 1], [2])
    body_subgraph = _build_subgraph(
        builder,
        tensors=body_tensors,
        operators=[reducer_op],
        inputs=[0, 1],
        outputs=[2],
    )

    buffers = [
        _build_buffer(builder),
        _build_buffer(builder, np.array(init_value, dtype=np.float32).tobytes()),
        _build_buffer(builder),
        _build_buffer(builder),
        _build_buffer(builder),
        _build_buffer(builder),
    ]
    return _finish_tflite_model(
        builder,
        subgraph=main_subgraph,
        extra_subgraphs=[body_subgraph],
        operator_codes=[reduce_code, reducer_code],
        buffers=buffers,
    )


def _build_stablehlo_sort_model(comparison_direction, is_stable=False):
    """Build a single-input STABLEHLO_SORT model with a compare body."""
    builder = flatbuffers.Builder(1024)

    _tfl_stablehlo_sort_opts.StablehloSortOptionsStart(builder)
    _tfl_stablehlo_sort_opts.StablehloSortOptionsAddDimension(builder, 1)
    _tfl_stablehlo_sort_opts.StablehloSortOptionsAddIsStable(builder, is_stable)
    _tfl_stablehlo_sort_opts.StablehloSortOptionsAddComparatorSubgraphIndex(builder, 1)
    sort_opts = _tfl_stablehlo_sort_opts.StablehloSortOptionsEnd(builder)

    _tfl_stablehlo_compare_opts.StablehloCompareOptionsStart(builder)
    _tfl_stablehlo_compare_opts.StablehloCompareOptionsAddComparisonDirection(
        builder, comparison_direction
    )
    compare_opts = _tfl_stablehlo_compare_opts.StablehloCompareOptionsEnd(builder)

    sort_builtin = _get_stablehlo_builtin_operator("STABLEHLO_SORT")
    compare_builtin = _get_stablehlo_builtin_operator("STABLEHLO_COMPARE")
    sort_code = _build_operator_code(builder, sort_builtin)
    compare_code = _build_operator_code(builder, compare_builtin)

    main_tensors = [
        _build_tensor(builder, 0, [2, 3]),
        _build_tensor(builder, 1, [2, 3]),
    ]
    sort_op = _build_operator(
        builder,
        0,
        [0],
        [1],
        builtin_options2_type=_tfl_builtin_options2.StablehloSortOptions,
        builtin_options2=sort_opts,
    )
    main_subgraph = _build_subgraph(
        builder,
        tensors=main_tensors,
        operators=[sort_op],
        inputs=[0],
        outputs=[1],
    )

    body_tensors = [
        _build_tensor(builder, 2, []),
        _build_tensor(builder, 3, []),
        _build_tensor(builder, 4, [], tensor_type=_tfl_tensor_type.BOOL),
    ]
    compare_op = _build_operator(
        builder,
        1,
        [0, 1],
        [2],
        builtin_options2_type=_tfl_builtin_options2.StablehloCompareOptions,
        builtin_options2=compare_opts,
    )
    body_subgraph = _build_subgraph(
        builder,
        tensors=body_tensors,
        operators=[compare_op],
        inputs=[0, 1],
        outputs=[2],
    )

    buffers = [_build_buffer(builder) for _ in range(5)]
    return _finish_tflite_model(
        builder,
        subgraph=main_subgraph,
        extra_subgraphs=[body_subgraph],
        operator_codes=[sort_code, compare_code],
        buffers=buffers,
    )


def _build_stablehlo_reduce_window_model(
    reducer_name="STABLEHLO_MAXIMUM",
    init_value=-np.inf,
    base_dilations=None,
):
    """Build an NHWC 2D STABLEHLO_REDUCE_WINDOW model."""
    builder = flatbuffers.Builder(1024)
    if base_dilations is None:
        base_dilations = [1, 1, 1, 1]

    window_dimensions_vec = _tflite_int64_vector(
        builder,
        _tfl_stablehlo_reduce_window_opts.StablehloReduceWindowOptionsStartWindowDimensionsVector,
        [1, 2, 2, 1],
    )
    window_strides_vec = _tflite_int64_vector(
        builder,
        _tfl_stablehlo_reduce_window_opts.StablehloReduceWindowOptionsStartWindowStridesVector,
        [1, 2, 2, 1],
    )
    base_dilations_vec = _tflite_int64_vector(
        builder,
        _tfl_stablehlo_reduce_window_opts.StablehloReduceWindowOptionsStartBaseDilationsVector,
        base_dilations,
    )
    window_dilations_vec = _tflite_int64_vector(
        builder,
        _tfl_stablehlo_reduce_window_opts.StablehloReduceWindowOptionsStartWindowDilationsVector,
        [1, 1, 1, 1],
    )
    padding_vec = _tflite_int64_vector(
        builder,
        _tfl_stablehlo_reduce_window_opts.StablehloReduceWindowOptionsStartPaddingVector,
        [0, 0, 0, 0, 0, 0, 0, 0],
    )

    _tfl_stablehlo_reduce_window_opts.StablehloReduceWindowOptionsStart(builder)
    _tfl_stablehlo_reduce_window_opts.StablehloReduceWindowOptionsAddWindowDimensions(
        builder, window_dimensions_vec
    )
    _tfl_stablehlo_reduce_window_opts.StablehloReduceWindowOptionsAddWindowStrides(
        builder, window_strides_vec
    )
    _tfl_stablehlo_reduce_window_opts.StablehloReduceWindowOptionsAddBaseDilations(
        builder, base_dilations_vec
    )
    _tfl_stablehlo_reduce_window_opts.StablehloReduceWindowOptionsAddWindowDilations(
        builder, window_dilations_vec
    )
    _tfl_stablehlo_reduce_window_opts.StablehloReduceWindowOptionsAddPadding(builder, padding_vec)
    _tfl_stablehlo_reduce_window_opts.StablehloReduceWindowOptionsAddBodySubgraphIndex(builder, 1)
    reduce_window_opts = _tfl_stablehlo_reduce_window_opts.StablehloReduceWindowOptionsEnd(builder)

    reduce_window_builtin = _get_stablehlo_builtin_operator("STABLEHLO_REDUCE_WINDOW")
    reducer_builtin = _get_stablehlo_builtin_operator(reducer_name)
    reduce_window_code = _build_operator_code(builder, reduce_window_builtin)
    reducer_code = _build_operator_code(builder, reducer_builtin)

    main_tensors = [
        _build_tensor(builder, 0, [1, 4, 4, 1]),
        _build_tensor(builder, 1, []),
        _build_tensor(builder, 2, [1, 2, 2, 1]),
    ]
    reduce_window_op = _build_operator(
        builder,
        0,
        [0, 1],
        [2],
        builtin_options2_type=_tfl_builtin_options2.StablehloReduceWindowOptions,
        builtin_options2=reduce_window_opts,
    )
    main_subgraph = _build_subgraph(
        builder,
        tensors=main_tensors,
        operators=[reduce_window_op],
        inputs=[0],
        outputs=[2],
    )

    body_tensors = [_build_tensor(builder, buffer_idx, []) for buffer_idx in range(3, 6)]
    reducer_op = _build_operator(builder, 1, [0, 1], [2])
    body_subgraph = _build_subgraph(
        builder,
        tensors=body_tensors,
        operators=[reducer_op],
        inputs=[0, 1],
        outputs=[2],
    )

    buffers = [
        _build_buffer(builder),
        _build_buffer(builder, np.array(init_value, dtype=np.float32).tobytes()),
        _build_buffer(builder),
        _build_buffer(builder),
        _build_buffer(builder),
        _build_buffer(builder),
    ]
    return _finish_tflite_model(
        builder,
        subgraph=main_subgraph,
        extra_subgraphs=[body_subgraph],
        operator_codes=[reduce_window_code, reducer_code],
        buffers=buffers,
    )


def _build_stablehlo_scatter_model(reducer_name="STABLEHLO_ADD", update_window_dims=None):
    """Build a canonical point-update STABLEHLO_SCATTER model."""
    builder = flatbuffers.Builder(1024)
    if update_window_dims is None:
        update_window_dims = []

    update_window_dims_vec = _tflite_int64_vector(
        builder,
        _tfl_stablehlo_scatter_opts.StablehloScatterOptionsStartUpdateWindowDimsVector,
        update_window_dims,
    )
    inserted_window_dims_vec = _tflite_int64_vector(
        builder,
        _tfl_stablehlo_scatter_opts.StablehloScatterOptionsStartInsertedWindowDimsVector,
        [0],
    )
    scatter_dims_vec = _tflite_int64_vector(
        builder,
        _tfl_stablehlo_scatter_opts.StablehloScatterOptionsStartScatterDimsToOperandDimsVector,
        [0],
    )

    _tfl_stablehlo_scatter_opts.StablehloScatterOptionsStart(builder)
    _tfl_stablehlo_scatter_opts.StablehloScatterOptionsAddUpdateWindowDims(
        builder, update_window_dims_vec
    )
    _tfl_stablehlo_scatter_opts.StablehloScatterOptionsAddInsertedWindowDims(
        builder, inserted_window_dims_vec
    )
    _tfl_stablehlo_scatter_opts.StablehloScatterOptionsAddScatterDimsToOperandDims(
        builder, scatter_dims_vec
    )
    _tfl_stablehlo_scatter_opts.StablehloScatterOptionsAddIndexVectorDim(builder, 1)
    _tfl_stablehlo_scatter_opts.StablehloScatterOptionsAddUpdateComputationSubgraphIndex(builder, 1)
    scatter_opts = _tfl_stablehlo_scatter_opts.StablehloScatterOptionsEnd(builder)

    scatter_builtin = _get_stablehlo_builtin_operator("STABLEHLO_SCATTER")
    reducer_builtin = _get_stablehlo_builtin_operator(reducer_name)
    scatter_code = _build_operator_code(builder, scatter_builtin)
    reducer_code = _build_operator_code(builder, reducer_builtin)

    main_tensors = [
        _build_tensor(builder, 0, [4]),
        _build_tensor(builder, 1, [2, 1], tensor_type=_tfl_tensor_type.INT32),
        _build_tensor(builder, 2, [2]),
        _build_tensor(builder, 3, [4]),
    ]
    scatter_op = _build_operator(
        builder,
        0,
        [0, 1, 2],
        [3],
        builtin_options2_type=_tfl_builtin_options2.StablehloScatterOptions,
        builtin_options2=scatter_opts,
    )
    main_subgraph = _build_subgraph(
        builder,
        tensors=main_tensors,
        operators=[scatter_op],
        inputs=[0, 1, 2],
        outputs=[3],
    )

    body_tensors = [_build_tensor(builder, buffer_idx, []) for buffer_idx in range(4, 7)]
    reducer_op = _build_operator(builder, 1, [0, 1], [2])
    body_subgraph = _build_subgraph(
        builder,
        tensors=body_tensors,
        operators=[reducer_op],
        inputs=[0, 1],
        outputs=[2],
    )

    buffers = [_build_buffer(builder) for _ in range(7)]
    return _finish_tflite_model(
        builder,
        subgraph=main_subgraph,
        extra_subgraphs=[body_subgraph],
        operator_codes=[scatter_code, reducer_code],
        buffers=buffers,
    )


def _build_stablehlo_custom_call_model(
    call_target_name="Sharding",
    has_side_effect=False,
    output_tensor_type=_tfl_tensor_type.FLOAT32,
    include_options=True,
):
    """Build a single-input STABLEHLO_CUSTOM_CALL model.

    When ``include_options`` is False the operator declares the
    StablehloCustomCallOptions type but omits the options table, emulating a
    malformed flatbuffer with a missing BuiltinOptions2 payload.
    """
    builder = flatbuffers.Builder(1024)

    custom_call_opts = None
    if include_options:
        call_target_name_offset = builder.CreateString(call_target_name)
        backend_config_offset = builder.CreateString("")
        _tfl_stablehlo_custom_call_opts.StablehloCustomCallOptionsStart(builder)
        _tfl_stablehlo_custom_call_opts.StablehloCustomCallOptionsAddCallTargetName(
            builder, call_target_name_offset
        )
        _tfl_stablehlo_custom_call_opts.StablehloCustomCallOptionsAddHasSideEffect(
            builder, has_side_effect
        )
        _tfl_stablehlo_custom_call_opts.StablehloCustomCallOptionsAddBackendConfig(
            builder, backend_config_offset
        )
        custom_call_opts = _tfl_stablehlo_custom_call_opts.StablehloCustomCallOptionsEnd(builder)

    custom_call_builtin = _get_stablehlo_builtin_operator("STABLEHLO_CUSTOM_CALL")
    custom_call_code = _build_operator_code(builder, custom_call_builtin)

    main_tensors = [
        _build_tensor(builder, 0, [2, 2]),
        _build_tensor(builder, 1, [2, 2], tensor_type=output_tensor_type),
    ]
    custom_call_op = _build_operator(
        builder,
        0,
        [0],
        [1],
        builtin_options2_type=_tfl_builtin_options2.StablehloCustomCallOptions,
        builtin_options2=custom_call_opts,
    )
    main_subgraph = _build_subgraph(
        builder,
        tensors=main_tensors,
        operators=[custom_call_op],
        inputs=[0],
        outputs=[1],
    )

    buffers = [_build_buffer(builder) for _ in range(2)]
    return _finish_tflite_model(
        builder,
        subgraph=main_subgraph,
        operator_codes=[custom_call_code],
        buffers=buffers,
    )


def _build_stablehlo_while_model(
    cond_subgraph_index=1,
    body_subgraph_index=2,
    cond_output_type=_tfl_tensor_type.BOOL,
    cond_input_type=_tfl_tensor_type.INT32,
    body_outputs=None,
    body_input_type=_tfl_tensor_type.INT32,
    body_output_type=_tfl_tensor_type.INT32,
    main_output_type=_tfl_tensor_type.INT32,
):
    """Build a STABLEHLO_WHILE model incrementing an int32 scalar until i < 3 is false."""
    builder = flatbuffers.Builder(1024)

    body_outputs = [2] if body_outputs is None else body_outputs
    while_options = _build_stablehlo_while_options(
        builder, cond_subgraph_index, body_subgraph_index
    )
    _tfl_stablehlo_compare_opts.StablehloCompareOptionsStart(builder)
    _tfl_stablehlo_compare_opts.StablehloCompareOptionsAddComparisonDirection(
        builder,
        _tfl_stablehlo_comp_dir.StablehloComparisonDirection.STABLEHLO_COMPARISON_DIRECTION_LT,
    )
    compare_opts = _tfl_stablehlo_compare_opts.StablehloCompareOptionsEnd(builder)
    one = np.array(1, dtype=np.int32)
    three = np.array(3, dtype=np.int32)

    main_tensors = [
        _build_tensor(builder, 0, [], tensor_type=_tfl_tensor_type.INT32),
        _build_tensor(builder, 3, [], tensor_type=main_output_type),
    ]
    main_while = _build_operator(
        builder,
        0,
        [0],
        [1],
        builtin_options2_type=_tfl_builtin_options2.StablehloWhileOptions,
        builtin_options2=while_options,
    )
    main_subgraph = _build_subgraph(
        builder,
        tensors=main_tensors,
        operators=[main_while],
        inputs=[0],
        outputs=[1],
    )

    cond_tensors = [
        _build_tensor(builder, 0, [], tensor_type=cond_input_type),
        _build_tensor(builder, 1, [], tensor_type=_tfl_tensor_type.INT32),
        _build_tensor(builder, 3, [], tensor_type=cond_output_type),
    ]
    cond_compare = _build_operator(
        builder,
        1,
        [0, 1],
        [2],
        builtin_options2_type=_tfl_builtin_options2.StablehloCompareOptions,
        builtin_options2=compare_opts,
    )
    cond_subgraph = _build_subgraph(
        builder,
        tensors=cond_tensors,
        operators=[cond_compare],
        inputs=[0],
        outputs=[2],
    )

    body_tensors = [
        _build_tensor(builder, 0, [], tensor_type=body_input_type),
        _build_tensor(builder, 2, [], tensor_type=_tfl_tensor_type.INT32),
        _build_tensor(builder, 3, [], tensor_type=body_output_type),
    ]
    body_add = _build_operator(builder, 2, [0, 1], [2])
    body_subgraph = _build_subgraph(
        builder,
        tensors=body_tensors,
        operators=[body_add],
        inputs=[0],
        outputs=body_outputs,
    )

    operator_codes = [
        _build_operator_code(builder, _get_stablehlo_builtin_operator("STABLEHLO_WHILE")),
        _build_operator_code(builder, _get_stablehlo_builtin_operator("STABLEHLO_COMPARE")),
        _build_operator_code(builder, _get_stablehlo_builtin_operator("STABLEHLO_ADD")),
    ]
    buffers = [
        _build_buffer(builder),
        _build_buffer(builder, three.tobytes()),
        _build_buffer(builder, one.tobytes()),
        _build_buffer(builder),
    ]
    return _finish_tflite_model(
        builder,
        subgraph=main_subgraph,
        extra_subgraphs=[cond_subgraph, body_subgraph],
        operator_codes=operator_codes,
        buffers=buffers,
    )


def _build_stablehlo_composite_model(with_attributes=False, use_main_input_after_composite=False):
    """Build a STABLEHLO_COMPOSITE model that decomposes to STABLEHLO_NEGATE."""
    builder = flatbuffers.Builder(1024)

    name = builder.CreateString("test.negate")
    attributes = None
    if with_attributes:
        _tfl_stablehlo_composite_opts.StableHLOCompositeOptionsStartCompositeAttributesVector(
            builder, 1
        )
        builder.PrependUint8(1)
        attributes = builder.EndVector()

    _tfl_stablehlo_composite_opts.StableHLOCompositeOptionsStart(builder)
    _tfl_stablehlo_composite_opts.StableHLOCompositeOptionsAddName(builder, name)
    _tfl_stablehlo_composite_opts.StableHLOCompositeOptionsAddVersion(builder, 1)
    _tfl_stablehlo_composite_opts.StableHLOCompositeOptionsAddDecompositionSubgraphIndex(builder, 1)
    if attributes is not None:
        _tfl_stablehlo_composite_opts.StableHLOCompositeOptionsAddCompositeAttributes(
            builder, attributes
        )
    composite_opts = _tfl_stablehlo_composite_opts.StableHLOCompositeOptionsEnd(builder)

    composite_builtin = _get_stablehlo_builtin_operator("STABLEHLO_COMPOSITE")
    negate_builtin = _get_stablehlo_builtin_operator("STABLEHLO_NEGATE")
    add_builtin = _get_stablehlo_builtin_operator("STABLEHLO_ADD")
    composite_code = _build_operator_code(builder, composite_builtin)
    negate_code = _build_operator_code(builder, negate_builtin)
    add_code = _build_operator_code(builder, add_builtin)

    main_tensors = [
        _build_tensor(builder, 0, [2, 2]),
        _build_tensor(builder, 1, [2, 2]),
        _build_tensor(builder, 2, [2, 2]),
    ]
    composite_op = _build_operator(
        builder,
        0,
        [0],
        [1],
        builtin_options2_type=_tfl_builtin_options2.StableHLOCompositeOptions,
        builtin_options2=composite_opts,
    )
    main_ops = [composite_op]
    main_outputs = [1]
    if use_main_input_after_composite:
        main_ops.append(_build_operator(builder, 2, [0, 1], [2]))
        main_outputs = [2]

    main_subgraph = _build_subgraph(
        builder,
        tensors=main_tensors,
        operators=main_ops,
        inputs=[0],
        outputs=main_outputs,
    )

    decomposition_tensors = [
        _build_tensor(builder, 2, [2, 2]),
        _build_tensor(builder, 3, [2, 2]),
    ]
    negate_op = _build_operator(builder, 1, [0], [1])
    decomposition_subgraph = _build_subgraph(
        builder,
        tensors=decomposition_tensors,
        operators=[negate_op],
        inputs=[0],
        outputs=[1],
    )

    buffers = [_build_buffer(builder) for _ in range(4)]
    return _finish_tflite_model(
        builder,
        subgraph=main_subgraph,
        extra_subgraphs=[decomposition_subgraph],
        operator_codes=[composite_code, negate_code, add_code],
        buffers=buffers,
    )


def _build_stablehlo_typed_binary_model(*, builtin_name, tensor_type):
    """Build a minimal TFLite StableHLO binary model with the requested tensor type."""
    builder = flatbuffers.Builder(1024)
    shape = [2, 2]
    output_tensor_idx = 2
    builtin_op = _get_stablehlo_builtin_operator(builtin_name)

    tensors = [
        _build_tensor(builder, buffer_idx, shape, tensor_type=tensor_type)
        for buffer_idx in range(3)
    ]
    stablehlo_op = _build_operator(builder, 0, [0, 1], [output_tensor_idx])
    subgraph = _build_subgraph(
        builder,
        tensors=tensors,
        operators=[stablehlo_op],
        inputs=[0, 1],
        outputs=[output_tensor_idx],
    )
    operator_codes = [_build_operator_code(builder, builtin_op)]
    buffers = [_build_buffer(builder) for _ in range(3)]
    return _finish_tflite_model(
        builder, subgraph=subgraph, operator_codes=operator_codes, buffers=buffers
    )


@pytest.mark.parametrize(
    "builtin_name, relax_op",
    [
        ("STABLEHLO_ABS", R.abs),
        ("STABLEHLO_COSINE", R.cos),
        ("STABLEHLO_EXPONENTIAL", R.exp),
        ("STABLEHLO_FLOOR", R.floor),
        ("STABLEHLO_LOG", R.log),
        ("STABLEHLO_LOGISTIC", R.sigmoid),
        ("STABLEHLO_NEGATE", R.negative),
        ("STABLEHLO_RSQRT", R.rsqrt),
        ("STABLEHLO_TANH", R.tanh),
    ],
)
def test_stablehlo_unary(builtin_name, relax_op):
    """TFLite StableHLO unary elementwise operators."""
    mod = _load_model_from_buffer(_build_stablehlo_model(builtin_name=builtin_name, input_count=1))

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 2), dtype="float32")) -> R.Tensor((2, 2), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="float32") = relax_op(x)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


@pytest.mark.parametrize(
    "builtin_name, relax_op",
    [
        ("STABLEHLO_ADD", R.add),
        ("STABLEHLO_DIVIDE", R.divide),
        ("STABLEHLO_MAXIMUM", R.maximum),
        ("STABLEHLO_MINIMUM", R.minimum),
        ("STABLEHLO_MULTIPLY", R.multiply),
        ("STABLEHLO_POWER", R.power),
        ("STABLEHLO_SUBTRACT", R.subtract),
    ],
)
def test_stablehlo_binary(builtin_name, relax_op):
    """TFLite StableHLO binary elementwise operators."""
    mod = _load_model_from_buffer(_build_stablehlo_model(builtin_name=builtin_name, input_count=2))

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 2), dtype="float32"),
            y: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tensor((2, 2), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="float32") = relax_op(x, y)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_stablehlo_model_with_unused_subgraph():
    """TFLite StableHLO import ignores unused non-main subgraphs."""
    mod = _load_model_from_buffer(_build_stablehlo_model_with_unused_subgraph())

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 2), dtype="float32"),
            y: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tensor((2, 2), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="float32") = R.add(x, y)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


@pytest.mark.parametrize(
    "reducer_name, init_value, relax_op",
    [
        ("STABLEHLO_ADD", 0.0, R.sum),
        ("STABLEHLO_MAXIMUM", -np.inf, R.max),
        ("STABLEHLO_MINIMUM", np.inf, R.min),
        ("STABLEHLO_MULTIPLY", 1.0, R.prod),
    ],
)
def test_stablehlo_reduce(reducer_name, init_value, relax_op):
    """TFLite StableHLO REDUCE with simple binary reducer body subgraphs."""
    mod = _load_model_from_buffer(_build_stablehlo_reduce_model(reducer_name, init_value))

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2,), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((2,), dtype="float32") = relax_op(x, axis=[1], keepdims=False)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_stablehlo_reduce_unsupported_reducer():
    """TFLite StableHLO REDUCE rejects unsupported body reducer ops."""
    buf = _build_stablehlo_reduce_model("STABLEHLO_SUBTRACT", 0.0)
    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)

    with pytest.raises(NotImplementedError, match="reducer"):
        from_tflite(tflite_model)


def test_stablehlo_reduce_non_identity_init_unsupported():
    """TFLite StableHLO REDUCE rejects init values that Relax reductions cannot express."""
    buf = _build_stablehlo_reduce_model("STABLEHLO_ADD", 1.0)
    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)

    with pytest.raises(NotImplementedError, match="init value"):
        from_tflite(tflite_model)


@pytest.mark.parametrize(
    "comparison_direction, descending",
    [
        (
            _tfl_stablehlo_comp_dir.StablehloComparisonDirection.STABLEHLO_COMPARISON_DIRECTION_LT,
            False,
        ),
        (
            _tfl_stablehlo_comp_dir.StablehloComparisonDirection.STABLEHLO_COMPARISON_DIRECTION_GT,
            True,
        ),
    ],
)
def test_stablehlo_sort(comparison_direction, descending):
    """TFLite StableHLO SORT with LT/GT scalar compare body subgraphs."""
    mod = _load_model_from_buffer(_build_stablehlo_sort_model(comparison_direction))

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((2, 3), dtype="float32") = R.sort(x, axis=1, descending=descending)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_stablehlo_sort_unsupported_comparator():
    """TFLite StableHLO SORT rejects non-ordering comparators."""
    _DIR = _tfl_stablehlo_comp_dir.StablehloComparisonDirection
    buf = _build_stablehlo_sort_model(_DIR.STABLEHLO_COMPARISON_DIRECTION_EQ)
    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)

    with pytest.raises(NotImplementedError, match="LT or GT"):
        from_tflite(tflite_model)


def test_stablehlo_sort_stable_unsupported():
    """TFLite StableHLO SORT rejects stable sort until Relax exposes that contract."""
    _DIR = _tfl_stablehlo_comp_dir.StablehloComparisonDirection
    buf = _build_stablehlo_sort_model(_DIR.STABLEHLO_COMPARISON_DIRECTION_LT, is_stable=True)
    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)

    with pytest.raises(NotImplementedError, match="stable sort"):
        from_tflite(tflite_model)


def test_stablehlo_reduce_window_max_pool2d():
    """TFLite StableHLO REDUCE_WINDOW max reducer lowers to NHWC max_pool2d."""
    mod = _load_model_from_buffer(_build_stablehlo_reduce_window_model())

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((1, 4, 4, 1), dtype="float32"),
        ) -> R.Tensor((1, 2, 2, 1), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 2, 2, 1), dtype="float32") = R.nn.max_pool2d(
                    x,
                    pool_size=[2, 2],
                    strides=[2, 2],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    ceil_mode=False,
                    layout="NHWC",
                    out_layout="NHWC",
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_stablehlo_reduce_window_unsupported_reducer():
    """TFLite StableHLO REDUCE_WINDOW rejects non-max reducers in the pool subset."""
    buf = _build_stablehlo_reduce_window_model(reducer_name="STABLEHLO_ADD", init_value=0.0)
    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)

    with pytest.raises(NotImplementedError, match="MAXIMUM"):
        from_tflite(tflite_model)


def test_stablehlo_reduce_window_base_dilation_unsupported():
    """TFLite StableHLO REDUCE_WINDOW rejects base dilation in the pool subset."""
    buf = _build_stablehlo_reduce_window_model(base_dilations=[1, 2, 1, 1])
    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)

    with pytest.raises(NotImplementedError, match="base dilation"):
        from_tflite(tflite_model)


@pytest.mark.parametrize(
    "reducer_name, reduction",
    [
        ("STABLEHLO_ADD", "add"),
        ("STABLEHLO_MAXIMUM", "max"),
        ("STABLEHLO_MINIMUM", "min"),
        ("STABLEHLO_MULTIPLY", "mul"),
    ],
)
def test_stablehlo_scatter(reducer_name, reduction):
    """TFLite StableHLO SCATTER point updates lower to Relax scatter_nd."""
    mod = _load_model_from_buffer(_build_stablehlo_scatter_model(reducer_name))

    @I.ir_module
    class Expected:
        @R.function
        def main(
            operand: R.Tensor((4,), dtype="float32"),
            indices: R.Tensor((2, 1), dtype="int32"),
            updates: R.Tensor((2,), dtype="float32"),
        ) -> R.Tensor((4,), dtype="float32"):
            R.func_attr({"num_input": 3})
            with R.dataflow():
                gv: R.Tensor((4,), dtype="float32") = R.scatter_nd(
                    operand, indices, updates, reduction=reduction
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_stablehlo_scatter_unsupported_reducer():
    """TFLite StableHLO SCATTER rejects unsupported update computation ops."""
    buf = _build_stablehlo_scatter_model(reducer_name="STABLEHLO_SUBTRACT")
    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)

    with pytest.raises(NotImplementedError, match="reducer"):
        from_tflite(tflite_model)


def test_stablehlo_scatter_update_window_unsupported():
    """TFLite StableHLO SCATTER rejects slice update windows in the point subset."""
    buf = _build_stablehlo_scatter_model(update_window_dims=[0])
    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)

    with pytest.raises(NotImplementedError, match="point updates"):
        from_tflite(tflite_model)


def test_stablehlo_custom_call_sharding():
    """TFLite StableHLO CUSTOM_CALL Sharding annotation lowers to identity."""
    mod = _load_model_from_buffer(_build_stablehlo_custom_call_model())

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 2), dtype="float32")) -> R.Tensor((2, 2), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="float32") = x
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_stablehlo_custom_call_unsupported_target():
    """TFLite StableHLO CUSTOM_CALL rejects unknown external call targets."""
    buf = _build_stablehlo_custom_call_model(call_target_name="custom_backend")
    with pytest.raises(
        NotImplementedError,
        match="STABLEHLO_CUSTOM_CALL target custom_backend is not supported",
    ):
        _load_model_from_buffer(buf)


def test_stablehlo_custom_call_sharding_side_effect_unsupported():
    """TFLite StableHLO CUSTOM_CALL rejects side-effecting Sharding calls."""
    buf = _build_stablehlo_custom_call_model(has_side_effect=True)
    with pytest.raises(NotImplementedError, match="side effects"):
        _load_model_from_buffer(buf)


def test_stablehlo_custom_call_sharding_metadata_mismatch_unsupported():
    """TFLite StableHLO CUSTOM_CALL rejects Sharding calls that change tensor metadata."""
    buf = _build_stablehlo_custom_call_model(output_tensor_type=_tfl_tensor_type.INT32)
    with pytest.raises(NotImplementedError, match="Sharding tensor metadata mismatch"):
        _load_model_from_buffer(buf)


def test_stablehlo_options_missing_payload_unsupported():
    """A StableHLO op that declares an options type but omits the payload fails cleanly."""
    buf = _build_stablehlo_custom_call_model(include_options=False)
    with pytest.raises(
        NotImplementedError,
        match="StablehloCustomCallOptions is required but missing from the operator",
    ):
        _load_model_from_buffer(buf)


def _build_stablehlo_rng_model(algorithm, state_len, out_shape, out_tensor_type, const_state=None):
    """Build a STABLEHLO_RNG_BIT_GENERATOR model.

    When ``const_state`` is provided, the uint64 initial state is embedded as a
    constant tensor (no graph input); otherwise it is a graph input.
    """
    builder = flatbuffers.Builder(1024)

    _tfl_stablehlo_rng_opts.StablehloRngBitGeneratorOptionsStart(builder)
    _tfl_stablehlo_rng_opts.StablehloRngBitGeneratorOptionsAddAlgorithm(builder, algorithm)
    rng_opts = _tfl_stablehlo_rng_opts.StablehloRngBitGeneratorOptionsEnd(builder)

    rng_builtin = _get_stablehlo_builtin_operator("STABLEHLO_RNG_BIT_GENERATOR")
    rng_code = _build_operator_code(builder, rng_builtin)

    main_tensors = [
        _build_tensor(builder, 0, [state_len], tensor_type=_tfl_tensor_type.UINT64),
        _build_tensor(builder, 1, [state_len], tensor_type=_tfl_tensor_type.UINT64),
        _build_tensor(builder, 2, list(out_shape), tensor_type=out_tensor_type),
    ]
    rng_op = _build_operator(
        builder,
        0,
        [0],
        [1, 2],
        builtin_options2_type=_tfl_builtin_options2.StablehloRngBitGeneratorOptions,
        builtin_options2=rng_opts,
    )
    main_subgraph = _build_subgraph(
        builder,
        tensors=main_tensors,
        operators=[rng_op],
        inputs=[] if const_state is not None else [0],
        outputs=[1, 2],
    )

    state_data = None
    if const_state is not None:
        state_data = np.array(const_state, dtype="uint64").tobytes()
    buffers = [
        _build_buffer(builder, data=state_data),
        _build_buffer(builder),
        _build_buffer(builder),
    ]
    return _finish_tflite_model(
        builder,
        subgraph=main_subgraph,
        operator_codes=[rng_code],
        buffers=buffers,
    )


_TFL_TENSOR_TYPE_TO_DTYPE = {
    _tfl_tensor_type.INT32: "int32",
    _tfl_tensor_type.UINT32: "uint32",
    _tfl_tensor_type.INT64: "int64",
    _tfl_tensor_type.UINT64: "uint64",
}


# Expected vectors are taken verbatim from the TFLite runtime kernel test
# (tensorflow/lite/kernels/rng_bit_generator_test.cc), guaranteeing bit-exact parity.
_RNG_THREEFRY_EXPECTED = {
    "int32": [43444564, -2144348869, -315321645, -549236733, 1672743891, -54463903],
    "uint32": [43444564, 2150618427, 3979645651, 3745730563, 1672743891, 4240503393],
    "int64": [
        -9209908263526143660,
        -2358953802017238317,
        -233920680524772397,
        2658481902456610144,
        -2022031683723149139,
        -2324041912354448873,
    ],
    "uint64": [
        9236835810183407956,
        16087790271692313299,
        18212823393184779219,
        2658481902456610144,
        16424712389986402477,
        16122702161355102743,
    ],
}
_RNG_THREEFRY_STATE = {"int32": [1, 5], "uint32": [1, 5], "int64": [1, 8], "uint64": [1, 8]}
_RNG_PHILOX_EXPECTED = {
    "int32": [-263854262, 1366700262, 495645701, -1243243882, 89414891, 1917262711],
    "uint32": [4031113034, 1366700262, 495645701, 3051723414, 89414891, 1917262711],
    "int64": [
        5869932932755744586,
        -5339691813646437371,
        8234580641674714347,
        2641225993340350124,
        1962472297844690804,
        -3580856229565614135,
    ],
    "uint64": [
        5869932932755744586,
        13107052260063114245,
        8234580641674714347,
        2641225993340350124,
        1962472297844690804,
        14865887844143937481,
    ],
}
_RNG_PHILOX_STATE = {
    "int32": [1, 4, 3],
    "uint32": [1, 4, 3],
    "int64": [1, 5, 3],
    "uint64": [1, 5, 3],
}


@pytest.mark.parametrize(
    "out_dtype,out_tensor_type",
    [
        ("int32", _tfl_tensor_type.INT32),
        ("uint32", _tfl_tensor_type.UINT32),
        ("int64", _tfl_tensor_type.INT64),
        ("uint64", _tfl_tensor_type.UINT64),
    ],
)
def test_stablehlo_rng_bit_generator_threefry(out_dtype, out_tensor_type):
    """TFLite STABLEHLO_RNG_BIT_GENERATOR THREEFRY matches the runtime kernel bit-exactly."""
    buf = _build_stablehlo_rng_model(_tfl_rng_algorithm.THREEFRY, 2, [2, 3], out_tensor_type)
    mod = _load_model_from_buffer(buf)
    ex = tvm.compile(mod, tvm_tirx.target.Target("llvm"))
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    result = vm["main"](tvm_tirx.runtime.tensor(np.array([1, 2], dtype="uint64")))
    state, output = result[0].numpy(), result[1].numpy()
    assert output.flatten().tolist() == _RNG_THREEFRY_EXPECTED[out_dtype]
    assert state.tolist() == _RNG_THREEFRY_STATE[out_dtype]


@pytest.mark.parametrize(
    "out_dtype,out_tensor_type",
    [
        ("int32", _tfl_tensor_type.INT32),
        ("uint32", _tfl_tensor_type.UINT32),
        ("int64", _tfl_tensor_type.INT64),
        ("uint64", _tfl_tensor_type.UINT64),
    ],
)
def test_stablehlo_rng_bit_generator_philox(out_dtype, out_tensor_type):
    """TFLite STABLEHLO_RNG_BIT_GENERATOR PHILOX matches the runtime kernel bit-exactly."""
    buf = _build_stablehlo_rng_model(_tfl_rng_algorithm.PHILOX, 3, [2, 3], out_tensor_type)
    mod = _load_model_from_buffer(buf)
    ex = tvm.compile(mod, tvm_tirx.target.Target("llvm"))
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    result = vm["main"](tvm_tirx.runtime.tensor(np.array([1, 2, 3], dtype="uint64")))
    state, output = result[0].numpy(), result[1].numpy()
    assert output.flatten().tolist() == _RNG_PHILOX_EXPECTED[out_dtype]
    assert state.tolist() == _RNG_PHILOX_STATE[out_dtype]


def test_stablehlo_rng_bit_generator_default_matches_philox():
    """TFLite STABLEHLO_RNG_BIT_GENERATOR DEFAULT resolves to the PHILOX algorithm."""
    buf = _build_stablehlo_rng_model(_tfl_rng_algorithm.DEFAULT, 3, [2, 3], _tfl_tensor_type.INT32)
    mod = _load_model_from_buffer(buf)
    ex = tvm.compile(mod, tvm_tirx.target.Target("llvm"))
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    result = vm["main"](tvm_tirx.runtime.tensor(np.array([1, 2, 3], dtype="uint64")))
    state, output = result[0].numpy(), result[1].numpy()
    assert output.flatten().tolist() == _RNG_PHILOX_EXPECTED["int32"]
    assert state.tolist() == _RNG_PHILOX_STATE["int32"]


def test_stablehlo_rng_bit_generator_deterministic():
    """Re-running the imported RNG kernel yields identical bit-exact output."""
    buf = _build_stablehlo_rng_model(_tfl_rng_algorithm.PHILOX, 3, [3, 3], _tfl_tensor_type.INT32)
    mod = _load_model_from_buffer(buf)
    ex = tvm.compile(mod, tvm_tirx.target.Target("llvm"))
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    init = tvm_tirx.runtime.tensor(np.array([7, 8, 9], dtype="uint64"))
    first = vm["main"](init)
    second = vm["main"](init)
    np.testing.assert_equal(first[1].numpy(), second[1].numpy())
    np.testing.assert_equal(first[0].numpy(), second[0].numpy())


def test_stablehlo_rng_bit_generator_constant_state():
    """A constant uint64 initial state imports and stays bit-exact (no graph input)."""
    buf = _build_stablehlo_rng_model(
        _tfl_rng_algorithm.THREEFRY, 2, [2, 3], _tfl_tensor_type.INT32, const_state=[1, 2]
    )
    mod = _load_model_from_buffer(buf)
    assert len(mod["main"].params) == 0
    ex = tvm.compile(mod, tvm_tirx.target.Target("llvm"))
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    result = vm["main"]()
    assert result[1].numpy().flatten().tolist() == _RNG_THREEFRY_EXPECTED["int32"]
    assert result[0].numpy().tolist() == _RNG_THREEFRY_STATE["int32"]


def test_stablehlo_rng_bit_generator_unsupported_output_dtype():
    """TFLite STABLEHLO_RNG_BIT_GENERATOR rejects non-integer output dtypes."""
    buf = _build_stablehlo_rng_model(_tfl_rng_algorithm.PHILOX, 3, [2, 3], _tfl_tensor_type.FLOAT32)
    with pytest.raises(NotImplementedError, match="output dtype float32 is not supported"):
        _load_model_from_buffer(buf)


def test_stablehlo_rng_bit_generator_threefry_invalid_state_unsupported():
    """TFLite STABLEHLO_RNG_BIT_GENERATOR rejects a u64[3] state for THREEFRY."""
    buf = _build_stablehlo_rng_model(_tfl_rng_algorithm.THREEFRY, 3, [2, 3], _tfl_tensor_type.INT32)
    with pytest.raises(NotImplementedError, match="THREEFRY requires a u64.2. state"):
        _load_model_from_buffer(buf)


def test_stablehlo_rng_bit_generator_non_uint64_state_unsupported():
    """TFLite STABLEHLO_RNG_BIT_GENERATOR rejects a non-uint64 initial state."""
    builder = flatbuffers.Builder(1024)
    _tfl_stablehlo_rng_opts.StablehloRngBitGeneratorOptionsStart(builder)
    _tfl_stablehlo_rng_opts.StablehloRngBitGeneratorOptionsAddAlgorithm(
        builder, _tfl_rng_algorithm.PHILOX
    )
    rng_opts = _tfl_stablehlo_rng_opts.StablehloRngBitGeneratorOptionsEnd(builder)
    rng_code = _build_operator_code(
        builder, _get_stablehlo_builtin_operator("STABLEHLO_RNG_BIT_GENERATOR")
    )
    tensors = [
        _build_tensor(builder, 0, [2], tensor_type=_tfl_tensor_type.INT64),
        _build_tensor(builder, 1, [2], tensor_type=_tfl_tensor_type.INT64),
        _build_tensor(builder, 2, [2, 3], tensor_type=_tfl_tensor_type.INT32),
    ]
    rng_op = _build_operator(
        builder,
        0,
        [0],
        [1, 2],
        builtin_options2_type=_tfl_builtin_options2.StablehloRngBitGeneratorOptions,
        builtin_options2=rng_opts,
    )
    subgraph = _build_subgraph(
        builder, tensors=tensors, operators=[rng_op], inputs=[0], outputs=[1, 2]
    )
    buffers = [_build_buffer(builder) for _ in range(3)]
    buf = _finish_tflite_model(
        builder, subgraph=subgraph, operator_codes=[rng_code], buffers=buffers
    )
    with pytest.raises(NotImplementedError, match="requires a uint64 initial state"):
        _load_model_from_buffer(buf)


def test_stablehlo_while():
    """TFLite STABLEHLO_WHILE lowers to a recursive Relax private function."""
    mod = _load_model_from_buffer(_build_stablehlo_while_model())

    @I.ir_module
    class Expected:
        @R.function(private=True)
        def tflite_stablehlo_while_cond_subgraph_1(
            tvmgen_tensor_0: R.Tensor((), dtype="int32"),
        ) -> R.Tensor((), dtype="bool"):
            with R.dataflow():
                gv: R.Tensor((), dtype="bool") = R.less(tvmgen_tensor_0, R.const(3, "int32"))
                R.output(gv)
            return gv

        @R.function(private=True)
        def tflite_stablehlo_while_body_subgraph_2(
            tvmgen_tensor_0: R.Tensor((), dtype="int32"),
        ) -> R.Tensor((), dtype="int32"):
            with R.dataflow():
                gv: R.Tensor((), dtype="int32") = R.add(tvmgen_tensor_0, R.const(1, "int32"))
                R.output(gv)
            return gv

        @R.function(private=True)
        def tflite_stablehlo_while_subgraph_1_2(
            tvmgen_tensor_0: R.Tensor((), dtype="int32"),
        ) -> R.Tensor((), dtype="int32"):
            cls = Expected
            while_cond: R.Tensor((), dtype="bool") = cls.tflite_stablehlo_while_cond_subgraph_1(
                tvmgen_tensor_0
            )
            if while_cond:
                gv: R.Tensor((), dtype="int32") = cls.tflite_stablehlo_while_body_subgraph_2(
                    tvmgen_tensor_0
                )
                gv1: R.Tensor((), dtype="int32") = cls.tflite_stablehlo_while_subgraph_1_2(gv)
                cond_result: R.Tensor((), dtype="int32") = gv1
            else:
                cond_result: R.Tensor((), dtype="int32") = tvmgen_tensor_0
            return cond_result

        @R.function
        def main(
            tvmgen_tensor_0: R.Tensor((), dtype="int32"),
        ) -> R.Tensor((), dtype="int32"):
            R.func_attr({"num_input": 1})
            cls = Expected
            with R.dataflow():
                gv: R.Tensor((), dtype="int32") = cls.tflite_stablehlo_while_subgraph_1_2(
                    tvmgen_tensor_0
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_stablehlo_while_non_bool_condition_unsupported():
    """STABLEHLO_WHILE rejects cond subgraphs that do not return scalar bool."""
    with pytest.raises(
        NotImplementedError, match="STABLEHLO_WHILE requires a scalar bool condition"
    ):
        _load_model_from_buffer(
            _build_stablehlo_while_model(cond_output_type=_tfl_tensor_type.INT32)
        )


def test_stablehlo_while_invalid_index_unsupported():
    """STABLEHLO_WHILE rejects invalid cond/body subgraph indices before lowering."""
    with pytest.raises(
        NotImplementedError, match="STABLEHLO_WHILE requires a valid subgraph index"
    ):
        _load_model_from_buffer(_build_stablehlo_while_model(cond_subgraph_index=3))


def test_stablehlo_while_output_count_mismatch_unsupported():
    """STABLEHLO_WHILE rejects body subgraphs whose output arity does not match loop vars."""
    with pytest.raises(NotImplementedError, match="STABLEHLO_WHILE subgraph output count mismatch"):
        _load_model_from_buffer(_build_stablehlo_while_model(body_outputs=[]))


def test_stablehlo_while_input_metadata_mismatch_unsupported():
    """STABLEHLO_WHILE rejects cond subgraph inputs whose metadata does not match loop vars."""
    with pytest.raises(
        NotImplementedError,
        match="STABLEHLO_WHILE subgraph input tensor metadata mismatch",
    ):
        _load_model_from_buffer(
            _build_stablehlo_while_model(cond_input_type=_tfl_tensor_type.FLOAT32)
        )


def test_stablehlo_while_output_metadata_mismatch_unsupported():
    """STABLEHLO_WHILE rejects body outputs whose metadata does not match loop vars."""
    with pytest.raises(
        NotImplementedError,
        match="STABLEHLO_WHILE subgraph output tensor metadata mismatch",
    ):
        _load_model_from_buffer(
            _build_stablehlo_while_model(body_output_type=_tfl_tensor_type.FLOAT32)
        )


def test_stablehlo_composite():
    """TFLite StableHLO COMPOSITE inlines a simple decomposition subgraph."""
    mod = _load_model_from_buffer(_build_stablehlo_composite_model())

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 2), dtype="float32")) -> R.Tensor((2, 2), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="float32") = R.negative(x)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_stablehlo_composite_does_not_overwrite_main_bindings():
    """TFLite StableHLO COMPOSITE decomposition tensor names are scoped locally."""
    mod = _load_model_from_buffer(
        _build_stablehlo_composite_model(use_main_input_after_composite=True)
    )

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 2), dtype="float32")) -> R.Tensor((2, 2), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 2), dtype="float32") = R.negative(x)
                gv: R.Tensor((2, 2), dtype="float32") = R.add(x, lv)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_stablehlo_composite_attributes_unsupported():
    """TFLite StableHLO COMPOSITE rejects attributes until they are parsed."""
    buf = _build_stablehlo_composite_model(with_attributes=True)
    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)

    with pytest.raises(NotImplementedError, match="composite attributes"):
        from_tflite(tflite_model)


@pytest.mark.parametrize(
    "builtin_name, relax_op, dtype, tensor_type",
    [
        ("STABLEHLO_AND", R.logical_and, "bool", _tfl_tensor_type.BOOL),
        ("STABLEHLO_OR", R.logical_or, "bool", _tfl_tensor_type.BOOL),
        ("STABLEHLO_AND", R.bitwise_and, "int32", _tfl_tensor_type.INT32),
        ("STABLEHLO_OR", R.bitwise_or, "int32", _tfl_tensor_type.INT32),
        ("STABLEHLO_SHIFT_LEFT", R.left_shift, "int32", _tfl_tensor_type.INT32),
    ],
)
def test_stablehlo_typed_binary(builtin_name, relax_op, dtype, tensor_type):
    """TFLite StableHLO binary elementwise operators with non-float dtype requirements."""
    mod = _load_model_from_buffer(
        _build_stablehlo_typed_binary_model(builtin_name=builtin_name, tensor_type=tensor_type)
    )

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 2), dtype=dtype),
            y: R.Tensor((2, 2), dtype=dtype),
        ) -> R.Tensor((2, 2), dtype=dtype):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype=dtype) = relax_op(x, y)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


@pytest.mark.parametrize(
    "builtin_name, relax_op",
    [
        ("STABLEHLO_SELECT", R.where),
    ],
)
def test_stablehlo_ternary(builtin_name, relax_op):
    """TFLite StableHLO ternary elementwise operators."""
    builder = flatbuffers.Builder(1024)
    shape = [2, 2]
    builtin_op = _get_stablehlo_builtin_operator(builtin_name)

    # First input (condition) must be bool for R.where
    tensor_0 = _build_tensor(builder, 0, shape, tensor_type=_tfl_tensor_type.BOOL)
    tensor_1 = _build_tensor(builder, 1, shape)
    tensor_2 = _build_tensor(builder, 2, shape)
    tensor_out = _build_tensor(builder, 3, shape)
    tensors = [tensor_0, tensor_1, tensor_2, tensor_out]

    stablehlo_op = _build_operator(
        builder,
        0,
        [0, 1, 2],
        [3],
    )
    subgraph = _build_subgraph(
        builder,
        tensors=tensors,
        operators=[stablehlo_op],
        inputs=[0, 1, 2],
        outputs=[3],
    )
    operator_codes = [_build_operator_code(builder, builtin_op)]
    buffers = [_build_buffer(builder) for _ in range(4)]

    mod = _load_model_from_buffer(
        _finish_tflite_model(
            builder, subgraph=subgraph, operator_codes=operator_codes, buffers=buffers
        )
    )

    @I.ir_module
    class Expected:
        @R.function
        def main(
            c: R.Tensor((2, 2), dtype="bool"),
            x: R.Tensor((2, 2), dtype="float32"),
            y: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tensor((2, 2), dtype="float32"):
            R.func_attr({"num_input": 3})
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="float32") = relax_op(c, x, y)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def _build_stablehlo_convert_model():
    """STABLEHLO_CONVERT: float32 input -> int32 output."""
    builder = flatbuffers.Builder(1024)
    shape = [2, 2]

    t_in = _build_tensor(builder, 0, shape, tensor_type=_tfl_tensor_type.FLOAT32)
    t_out = _build_tensor(builder, 1, shape, tensor_type=_tfl_tensor_type.INT32)
    tensors = [t_in, t_out]

    op_code = _build_operator_code(builder, _get_stablehlo_builtin_operator("STABLEHLO_CONVERT"))
    op = _build_operator(builder, 0, [0], [1])
    subgraph = _build_subgraph(
        builder,
        tensors=tensors,
        operators=[op],
        inputs=[0],
        outputs=[1],
    )
    buffers = [_build_buffer(builder) for _ in range(2)]
    return _finish_tflite_model(
        builder, subgraph=subgraph, operator_codes=[op_code], buffers=buffers
    )


def test_stablehlo_convert():
    """TFLite StableHLO CONVERT (astype float32 -> int32)."""
    mod = _load_model_from_buffer(_build_stablehlo_convert_model())

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 2), dtype="float32")) -> R.Tensor((2, 2), dtype="int32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="int32") = R.astype(x, dtype="int32")
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_stablehlo_clamp():
    """TFLite StableHLO CLAMP (clip with min/operand/max order)."""
    mod = _load_model_from_buffer(
        _build_stablehlo_model(builtin_name="STABLEHLO_CLAMP", input_count=3)
    )

    @I.ir_module
    class Expected:
        @R.function
        def main(
            m: R.Tensor((2, 2), dtype="float32"),
            x: R.Tensor((2, 2), dtype="float32"),
            M: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tensor((2, 2), dtype="float32"):
            R.func_attr({"num_input": 3})
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="float32") = R.minimum(R.maximum(x, m), M)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def _build_stablehlo_concat_model(dimension, num_inputs):
    """STABLEHLO_CONCATENATE with given dimension and number of inputs."""
    builder = flatbuffers.Builder(1024)
    shape = [2, 2]

    # Build concat options
    _tfl_stablehlo_concat_opts.StablehloConcatenateOptionsStart(builder)
    _tfl_stablehlo_concat_opts.StablehloConcatenateOptionsAddDimension(builder, dimension)
    concat_opts = _tfl_stablehlo_concat_opts.StablehloConcatenateOptionsEnd(builder)

    builtin_op = _get_stablehlo_builtin_operator("STABLEHLO_CONCATENATE")
    op_code = _build_operator_code(builder, builtin_op)

    if dimension == 0:
        out_shape = [num_inputs * shape[0], shape[1]]
    else:
        out_shape = [shape[0], num_inputs * shape[1]]
    tensors = [_build_tensor(builder, i, shape) for i in range(num_inputs)] + [
        _build_tensor(builder, num_inputs, out_shape)
    ]

    op = _build_operator(
        builder,
        0,
        list(range(num_inputs)),
        [num_inputs],
        builtin_options2_type=_tfl_builtin_options2.StablehloConcatenateOptions,
        builtin_options2=concat_opts,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=tensors,
        operators=[op],
        inputs=list(range(num_inputs)),
        outputs=[num_inputs],
    )
    buffers = [_build_buffer(builder) for _ in range(num_inputs + 1)]
    return _finish_tflite_model(
        builder, subgraph=subgraph, operator_codes=[op_code], buffers=buffers
    )


@pytest.mark.parametrize("dimension", [0, 1])
def test_stablehlo_concatenate(dimension):
    """TFLite StableHLO CONCATENATE with 2 inputs along given axis."""
    num_inputs = 2
    mod = _load_model_from_buffer(
        _build_stablehlo_concat_model(dimension=dimension, num_inputs=num_inputs)
    )

    out_dim = (4, 2) if dimension == 0 else (2, 4)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 2), dtype="float32"),
            y: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tensor(out_dim, dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                gv: R.Tensor(out_dim, dtype="float32") = R.concat((x, y), axis=dimension)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def _build_stablehlo_reshape_model(input_shape, output_shape, tensor_type=_tfl_tensor_type.FLOAT32):
    """STABLEHLO_RESHAPE with given input and output shapes."""
    builder = flatbuffers.Builder(1024)

    builtin_op = _get_stablehlo_builtin_operator("STABLEHLO_RESHAPE")
    op_code = _build_operator_code(builder, builtin_op)

    tensors = [
        _build_tensor(builder, 0, input_shape, tensor_type=tensor_type),
        _build_tensor(builder, 1, output_shape, tensor_type=tensor_type),
    ]
    op = _build_operator(builder, 0, [0], [1])
    subgraph = _build_subgraph(
        builder,
        tensors=tensors,
        operators=[op],
        inputs=[0],
        outputs=[1],
    )
    buffers = [_build_buffer(builder) for _ in range(2)]
    return _finish_tflite_model(
        builder, subgraph=subgraph, operator_codes=[op_code], buffers=buffers
    )


def test_stablehlo_reshape():
    """TFLite StableHLO RESHAPE lowers to Relax reshape."""
    mod = _load_model_from_buffer(
        _build_stablehlo_reshape_model(input_shape=[2, 3], output_shape=[3, 2])
    )

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((3, 2), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((3, 2), dtype="float32") = R.reshape(x, (3, 2))
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_stablehlo_reshape_scalar():
    """TFLite StableHLO RESHAPE supports a rank-0 output."""
    mod = _load_model_from_buffer(_build_stablehlo_reshape_model(input_shape=[1], output_shape=[]))

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1,), dtype="float32")) -> R.Tensor((), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((), dtype="float32") = R.reshape(x, ())
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_stablehlo_reshape_complex64():
    """TFLite StableHLO RESHAPE preserves the Relax complex pair axis."""
    mod = _load_model_from_buffer(
        _build_stablehlo_reshape_model(
            input_shape=[2, 3],
            output_shape=[3, 2],
            tensor_type=_tfl_tensor_type.COMPLEX64,
        )
    )

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 2), dtype="float32")) -> R.Tensor((3, 2, 2), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((3, 2, 2), dtype="float32") = R.reshape(x, (3, 2, 2))
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def _build_stablehlo_slice_model(input_shape, start_indices, limit_indices, strides, output_shape):
    """STABLEHLO_SLICE with static start, limit, and stride attributes."""
    builder = flatbuffers.Builder(1024)

    start_vec = _tflite_int64_vector(
        builder,
        _tfl_stablehlo_slice_opts.StablehloSliceOptionsStartStartIndicesVector,
        start_indices,
    )
    limit_vec = _tflite_int64_vector(
        builder,
        _tfl_stablehlo_slice_opts.StablehloSliceOptionsStartLimitIndicesVector,
        limit_indices,
    )
    strides_vec = _tflite_int64_vector(
        builder,
        _tfl_stablehlo_slice_opts.StablehloSliceOptionsStartStridesVector,
        strides,
    )

    _tfl_stablehlo_slice_opts.StablehloSliceOptionsStart(builder)
    _tfl_stablehlo_slice_opts.StablehloSliceOptionsAddStartIndices(builder, start_vec)
    _tfl_stablehlo_slice_opts.StablehloSliceOptionsAddLimitIndices(builder, limit_vec)
    _tfl_stablehlo_slice_opts.StablehloSliceOptionsAddStrides(builder, strides_vec)
    slice_opts = _tfl_stablehlo_slice_opts.StablehloSliceOptionsEnd(builder)

    builtin_op = _get_stablehlo_builtin_operator("STABLEHLO_SLICE")
    op_code = _build_operator_code(builder, builtin_op)

    tensors = [
        _build_tensor(builder, 0, input_shape),
        _build_tensor(builder, 1, output_shape),
    ]
    op = _build_operator(
        builder,
        0,
        [0],
        [1],
        builtin_options2_type=_tfl_builtin_options2.StablehloSliceOptions,
        builtin_options2=slice_opts,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=tensors,
        operators=[op],
        inputs=[0],
        outputs=[1],
    )
    buffers = [_build_buffer(builder) for _ in range(2)]
    return _finish_tflite_model(
        builder, subgraph=subgraph, operator_codes=[op_code], buffers=buffers
    )


def test_stablehlo_slice():
    """TFLite StableHLO SLICE lowers to Relax strided_slice."""
    mod = _load_model_from_buffer(
        _build_stablehlo_slice_model(
            input_shape=[4, 5],
            start_indices=[1, 0],
            limit_indices=[4, 4],
            strides=[2, 2],
            output_shape=[2, 2],
        )
    )

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((4, 5), dtype="float32")) -> R.Tensor((2, 2), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="float32") = R.strided_slice(
                    x, axes=[0, 1], begin=[1, 0], end=[4, 4], strides=[2, 2]
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def _build_stablehlo_transpose_model(
    input_shape, permutation, output_shape, tensor_type=_tfl_tensor_type.FLOAT32
):
    """STABLEHLO_TRANSPOSE with a static permutation."""
    builder = flatbuffers.Builder(1024)

    perm_vec = _tflite_int64_vector(
        builder,
        _tfl_stablehlo_transpose_opts.StablehloTransposeOptionsStartPermutationVector,
        permutation,
    )
    _tfl_stablehlo_transpose_opts.StablehloTransposeOptionsStart(builder)
    _tfl_stablehlo_transpose_opts.StablehloTransposeOptionsAddPermutation(builder, perm_vec)
    transpose_opts = _tfl_stablehlo_transpose_opts.StablehloTransposeOptionsEnd(builder)

    builtin_op = _get_stablehlo_builtin_operator("STABLEHLO_TRANSPOSE")
    op_code = _build_operator_code(builder, builtin_op)

    tensors = [
        _build_tensor(builder, 0, input_shape, tensor_type=tensor_type),
        _build_tensor(builder, 1, output_shape, tensor_type=tensor_type),
    ]
    op = _build_operator(
        builder,
        0,
        [0],
        [1],
        builtin_options2_type=_tfl_builtin_options2.StablehloTransposeOptions,
        builtin_options2=transpose_opts,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=tensors,
        operators=[op],
        inputs=[0],
        outputs=[1],
    )
    buffers = [_build_buffer(builder) for _ in range(2)]
    return _finish_tflite_model(
        builder, subgraph=subgraph, operator_codes=[op_code], buffers=buffers
    )


def test_stablehlo_transpose():
    """TFLite StableHLO TRANSPOSE lowers to Relax permute_dims."""
    mod = _load_model_from_buffer(
        _build_stablehlo_transpose_model(
            input_shape=[2, 3, 4], permutation=[1, 2, 0], output_shape=[3, 4, 2]
        )
    )

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4), dtype="float32")) -> R.Tensor((3, 4, 2), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((3, 4, 2), dtype="float32") = R.permute_dims(x, axes=[1, 2, 0])
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_stablehlo_transpose_complex64():
    """TFLite StableHLO TRANSPOSE leaves the Relax complex pair axis trailing."""
    mod = _load_model_from_buffer(
        _build_stablehlo_transpose_model(
            input_shape=[2, 3, 4],
            permutation=[1, 2, 0],
            output_shape=[3, 4, 2],
            tensor_type=_tfl_tensor_type.COMPLEX64,
        )
    )

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 2), dtype="float32")) -> R.Tensor(
            (3, 4, 2, 2), dtype="float32"
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((3, 4, 2, 2), dtype="float32") = R.permute_dims(x, axes=[1, 2, 0, 3])
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def _build_stablehlo_broadcast_in_dim_model(input_shape, broadcast_dims, output_shape):
    """STABLEHLO_BROADCAST_IN_DIM with given broadcast dimensions."""
    builder = flatbuffers.Builder(1024)

    # Build broadcast dimensions vector
    _tfl_stablehlo_bcast_opts.StablehloBroadcastInDimOptionsStartBroadcastDimensionsVector(
        builder, len(broadcast_dims)
    )
    for d in reversed(broadcast_dims):
        builder.PrependInt64(d)
    dims_vec = builder.EndVector()

    _tfl_stablehlo_bcast_opts.StablehloBroadcastInDimOptionsStart(builder)
    _tfl_stablehlo_bcast_opts.StablehloBroadcastInDimOptionsAddBroadcastDimensions(
        builder, dims_vec
    )
    bcast_opts = _tfl_stablehlo_bcast_opts.StablehloBroadcastInDimOptionsEnd(builder)

    builtin_op = _get_stablehlo_builtin_operator("STABLEHLO_BROADCAST_IN_DIM")
    op_code = _build_operator_code(builder, builtin_op)

    t_in = _build_tensor(builder, 0, input_shape)
    t_out = _build_tensor(builder, 1, output_shape)
    tensors = [t_in, t_out]

    op = _build_operator(
        builder,
        0,
        [0],
        [1],
        builtin_options2_type=_tfl_builtin_options2.StablehloBroadcastInDimOptions,
        builtin_options2=bcast_opts,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=tensors,
        operators=[op],
        inputs=[0],
        outputs=[1],
    )
    buffers = [_build_buffer(builder) for _ in range(2)]
    return _finish_tflite_model(
        builder, subgraph=subgraph, operator_codes=[op_code], buffers=buffers
    )


def test_stablehlo_broadcast_in_dim():
    """TFLite StableHLO BROADCAST_IN_DIM: (3,) -> (2, 3) with dims=[1]."""
    mod = _load_model_from_buffer(
        _build_stablehlo_broadcast_in_dim_model(
            input_shape=[3], broadcast_dims=[1], output_shape=[2, 3]
        )
    )

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((3,), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((2, 3), dtype="float32") = R.broadcast_to(R.reshape(x, (1, 3)), (2, 3))
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def _build_stablehlo_iota_model(iota_dimension, output_shape):
    """STABLEHLO_IOTA with given iota dimension and output shape."""
    builder = flatbuffers.Builder(1024)

    _tfl_stablehlo_iota_opts.StablehloIotaOptionsStart(builder)
    _tfl_stablehlo_iota_opts.StablehloIotaOptionsAddIotaDimension(builder, iota_dimension)
    iota_opts = _tfl_stablehlo_iota_opts.StablehloIotaOptionsEnd(builder)

    builtin_op = _get_stablehlo_builtin_operator("STABLEHLO_IOTA")
    op_code = _build_operator_code(builder, builtin_op)

    t_out = _build_tensor(builder, 0, output_shape, tensor_type=_tfl_tensor_type.INT32)
    tensors = [t_out]

    op = _build_operator(
        builder,
        0,
        [],
        [0],
        builtin_options2_type=_tfl_builtin_options2.StablehloIotaOptions,
        builtin_options2=iota_opts,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=tensors,
        operators=[op],
        inputs=[],
        outputs=[0],
    )
    buffers = [_build_buffer(builder)]
    return _finish_tflite_model(
        builder, subgraph=subgraph, operator_codes=[op_code], buffers=buffers
    )


def test_stablehlo_iota():
    """TFLite StableHLO IOTA: iota_dim=1, shape=(2, 3), dtype=int32."""
    mod = _load_model_from_buffer(
        _build_stablehlo_iota_model(iota_dimension=1, output_shape=[2, 3])
    )

    @I.ir_module
    class Expected:
        @R.function
        def main() -> R.Tensor((2, 3), dtype="int32"):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                gv: R.Tensor((2, 3), dtype="int32") = R.broadcast_to(
                    R.reshape(R.arange(0, 3, 1, dtype="int32"), (1, 3)), (2, 3)
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def _build_stablehlo_compare_model(direction):
    """STABLEHLO_COMPARE with given comparison direction."""
    builder = flatbuffers.Builder(1024)

    _tfl_stablehlo_compare_opts.StablehloCompareOptionsStart(builder)
    _tfl_stablehlo_compare_opts.StablehloCompareOptionsAddComparisonDirection(builder, direction)
    cmp_opts = _tfl_stablehlo_compare_opts.StablehloCompareOptionsEnd(builder)

    builtin_op = _get_stablehlo_builtin_operator("STABLEHLO_COMPARE")
    op_code = _build_operator_code(builder, builtin_op)

    shape = [2, 2]
    t_lhs = _build_tensor(builder, 0, shape)
    t_rhs = _build_tensor(builder, 1, shape)
    t_out = _build_tensor(builder, 2, shape, tensor_type=_tfl_tensor_type.BOOL)
    tensors = [t_lhs, t_rhs, t_out]

    op = _build_operator(
        builder,
        0,
        [0, 1],
        [2],
        builtin_options2_type=_tfl_builtin_options2.StablehloCompareOptions,
        builtin_options2=cmp_opts,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=tensors,
        operators=[op],
        inputs=[0, 1],
        outputs=[2],
    )
    buffers = [_build_buffer(builder) for _ in range(3)]
    return _finish_tflite_model(
        builder, subgraph=subgraph, operator_codes=[op_code], buffers=buffers
    )


@pytest.mark.parametrize(
    "direction_enum, relax_op",
    [
        (
            _tfl_stablehlo_comp_dir.StablehloComparisonDirection.STABLEHLO_COMPARISON_DIRECTION_EQ,
            R.equal,
        ),
        (
            _tfl_stablehlo_comp_dir.StablehloComparisonDirection.STABLEHLO_COMPARISON_DIRECTION_NE,
            R.not_equal,
        ),
        (
            _tfl_stablehlo_comp_dir.StablehloComparisonDirection.STABLEHLO_COMPARISON_DIRECTION_GE,
            R.greater_equal,
        ),
        (
            _tfl_stablehlo_comp_dir.StablehloComparisonDirection.STABLEHLO_COMPARISON_DIRECTION_GT,
            R.greater,
        ),
        (
            _tfl_stablehlo_comp_dir.StablehloComparisonDirection.STABLEHLO_COMPARISON_DIRECTION_LE,
            R.less_equal,
        ),
        (
            _tfl_stablehlo_comp_dir.StablehloComparisonDirection.STABLEHLO_COMPARISON_DIRECTION_LT,
            R.less,
        ),
    ],
)
def test_stablehlo_compare(direction_enum, relax_op):
    """TFLite StableHLO COMPARE with various comparison directions."""
    mod = _load_model_from_buffer(_build_stablehlo_compare_model(direction_enum))

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 2), dtype="float32"),
            y: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tensor((2, 2), dtype="bool"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="bool") = relax_op(x, y)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_stablehlo_compare_totalorder_unsupported():
    """STABLEHLO_COMPARE with TOTALORDER type raises NotImplementedError."""
    builder = flatbuffers.Builder(1024)

    _DIR = _tfl_stablehlo_comp_dir.StablehloComparisonDirection
    _TYPE = _tfl_stablehlo_comp_type.StablehloComparisonType

    _tfl_stablehlo_compare_opts.StablehloCompareOptionsStart(builder)
    _tfl_stablehlo_compare_opts.StablehloCompareOptionsAddComparisonDirection(
        builder, _DIR.STABLEHLO_COMPARISON_DIRECTION_EQ
    )
    _tfl_stablehlo_compare_opts.StablehloCompareOptionsAddCompareType(
        builder, _TYPE.STABLEHLO_COMPARISON_TYPE_FLOAT_TOTAL_ORDER
    )
    cmp_opts = _tfl_stablehlo_compare_opts.StablehloCompareOptionsEnd(builder)

    builtin_op = _get_stablehlo_builtin_operator("STABLEHLO_COMPARE")
    op_code = _build_operator_code(builder, builtin_op)

    shape = [2, 2]
    t_lhs = _build_tensor(builder, 0, shape)
    t_rhs = _build_tensor(builder, 1, shape)
    t_out = _build_tensor(builder, 2, shape, tensor_type=_tfl_tensor_type.BOOL)
    tensors = [t_lhs, t_rhs, t_out]

    op = _build_operator(
        builder,
        0,
        [0, 1],
        [2],
        builtin_options2_type=_tfl_builtin_options2.StablehloCompareOptions,
        builtin_options2=cmp_opts,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=tensors,
        operators=[op],
        inputs=[0, 1],
        outputs=[2],
    )
    buffers = [_build_buffer(builder) for _ in range(3)]
    buf = _finish_tflite_model(
        builder, subgraph=subgraph, operator_codes=[op_code], buffers=buffers
    )

    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)

    with pytest.raises(NotImplementedError, match="TOTALORDER"):
        from_tflite(tflite_model)


def _stablehlo_gather_i64_vector(builder, start_vector_fn, values):
    start_vector_fn(builder, len(values))
    for value in reversed(values):
        builder.PrependInt64(value)
    return builder.EndVector()


def _build_stablehlo_gather_model(
    *,
    data_shape,
    indices_shape,
    output_shape,
    offset_dims,
    collapsed_slice_dims,
    start_index_map,
    index_vector_dim,
    slice_sizes,
):
    """Build a minimal STABLEHLO_GATHER TFLite model."""
    builder = flatbuffers.Builder(1024)

    offset_dims_vec = _stablehlo_gather_i64_vector(
        builder,
        _tfl_stablehlo_gather_opts.StablehloGatherOptionsStartOffsetDimsVector,
        offset_dims,
    )
    collapsed_slice_dims_vec = _stablehlo_gather_i64_vector(
        builder,
        _tfl_stablehlo_gather_opts.StablehloGatherOptionsStartCollapsedSliceDimsVector,
        collapsed_slice_dims,
    )
    start_index_map_vec = _stablehlo_gather_i64_vector(
        builder,
        _tfl_stablehlo_gather_opts.StablehloGatherOptionsStartStartIndexMapVector,
        start_index_map,
    )
    slice_sizes_vec = _stablehlo_gather_i64_vector(
        builder,
        _tfl_stablehlo_gather_opts.StablehloGatherOptionsStartSliceSizesVector,
        slice_sizes,
    )

    _tfl_stablehlo_gather_opts.StablehloGatherOptionsStart(builder)
    _tfl_stablehlo_gather_opts.StablehloGatherOptionsAddOffsetDims(builder, offset_dims_vec)
    _tfl_stablehlo_gather_opts.StablehloGatherOptionsAddCollapsedSliceDims(
        builder, collapsed_slice_dims_vec
    )
    _tfl_stablehlo_gather_opts.StablehloGatherOptionsAddStartIndexMap(builder, start_index_map_vec)
    _tfl_stablehlo_gather_opts.StablehloGatherOptionsAddIndexVectorDim(builder, index_vector_dim)
    _tfl_stablehlo_gather_opts.StablehloGatherOptionsAddSliceSizes(builder, slice_sizes_vec)
    gather_opts = _tfl_stablehlo_gather_opts.StablehloGatherOptionsEnd(builder)

    builtin_op = _get_stablehlo_builtin_operator("STABLEHLO_GATHER")
    op_code = _build_operator_code(builder, builtin_op)

    t_data = _build_tensor(builder, 0, data_shape)
    t_indices = _build_tensor(builder, 1, indices_shape, tensor_type=_tfl_tensor_type.INT32)
    t_out = _build_tensor(builder, 2, output_shape)
    op = _build_operator(
        builder,
        0,
        [0, 1],
        [2],
        builtin_options2_type=_tfl_builtin_options2.StablehloGatherOptions,
        builtin_options2=gather_opts,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=[t_data, t_indices, t_out],
        operators=[op],
        inputs=[0, 1],
        outputs=[2],
    )
    buffers = [_build_buffer(builder) for _ in range(3)]
    return _finish_tflite_model(
        builder, subgraph=subgraph, operator_codes=[op_code], buffers=buffers
    )


@pytest.mark.parametrize(
    "axis, offset_dims, slice_sizes, output_shape",
    [
        (0, [1], [1, 4], [2, 4]),
        (1, [0], [3, 1], [3, 2]),
    ],
)
def test_stablehlo_gather_take_equivalent(axis, offset_dims, slice_sizes, output_shape):
    """TFLite StableHLO GATHER take-equivalent subset."""
    mod = _load_model_from_buffer(
        _build_stablehlo_gather_model(
            data_shape=[3, 4],
            indices_shape=[2, 1],
            output_shape=output_shape,
            offset_dims=offset_dims,
            collapsed_slice_dims=[axis],
            start_index_map=[axis],
            index_vector_dim=1,
            slice_sizes=slice_sizes,
        )
    )

    out_shape = tuple(output_shape)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data: R.Tensor((3, 4), dtype="float32"),
            indices: R.Tensor((2, 1), dtype="int32"),
        ) -> R.Tensor(out_shape, dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                reshaped: R.Tensor((2,), dtype="int32") = R.reshape(indices, (2,))
                gv: R.Tensor(out_shape, dtype="float32") = R.take(
                    data, reshaped, axis=axis, mode="fast"
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_stablehlo_gather_complex_unsupported():
    """TFLite StableHLO GATHER with multi-dimensional start_index_map is unsupported."""
    buf = _build_stablehlo_gather_model(
        data_shape=[3, 4],
        indices_shape=[2, 2],
        output_shape=[2],
        offset_dims=[],
        collapsed_slice_dims=[0, 1],
        start_index_map=[0, 1],
        index_vector_dim=1,
        slice_sizes=[1, 1],
    )
    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)

    with pytest.raises(NotImplementedError, match="start_index_map"):
        from_tflite(tflite_model)


def _pad_vector(builder, start_vector_fn, values):
    """Build a FlatBuffers int64 vector for pad options."""
    start_vector_fn(builder, len(values))
    for v in reversed(values):
        builder.PrependInt64(v)
    return builder.EndVector()


def _build_stablehlo_pad_model(edge_low, edge_high, interior):
    """STABLEHLO_PAD with given padding vectors."""
    builder = flatbuffers.Builder(1024)

    lo_vec = _pad_vector(
        builder,
        _tfl_stablehlo_pad_opts.StablehloPadOptionsStartEdgePaddingLowVector,
        edge_low,
    )
    hi_vec = _pad_vector(
        builder,
        _tfl_stablehlo_pad_opts.StablehloPadOptionsStartEdgePaddingHighVector,
        edge_high,
    )
    int_vec = _pad_vector(
        builder,
        _tfl_stablehlo_pad_opts.StablehloPadOptionsStartInteriorPaddingVector,
        interior,
    )

    _tfl_stablehlo_pad_opts.StablehloPadOptionsStart(builder)
    _tfl_stablehlo_pad_opts.StablehloPadOptionsAddEdgePaddingLow(builder, lo_vec)
    _tfl_stablehlo_pad_opts.StablehloPadOptionsAddEdgePaddingHigh(builder, hi_vec)
    _tfl_stablehlo_pad_opts.StablehloPadOptionsAddInteriorPadding(builder, int_vec)
    pad_opts = _tfl_stablehlo_pad_opts.StablehloPadOptionsEnd(builder)

    builtin_op = _get_stablehlo_builtin_operator("STABLEHLO_PAD")
    op_code = _build_operator_code(builder, builtin_op)

    t_in = _build_tensor(builder, 0, [3, 3])
    # pad_value is a scalar tensor
    t_pad_val = _build_tensor(builder, 1, [])
    t_out = _build_tensor(builder, 2, [4, 4])
    tensors = [t_in, t_pad_val, t_out]

    op = _build_operator(
        builder,
        0,
        [0, 1],
        [2],
        builtin_options2_type=_tfl_builtin_options2.StablehloPadOptions,
        builtin_options2=pad_opts,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=tensors,
        operators=[op],
        inputs=[0],
        outputs=[2],
    )
    buffers = [
        _build_buffer(builder),
        _build_buffer(builder, np.array([0.0], dtype=np.float32).tobytes()),
        _build_buffer(builder),
    ]
    return _finish_tflite_model(
        builder, subgraph=subgraph, operator_codes=[op_code], buffers=buffers
    )


def test_stablehlo_pad():
    """TFLite StableHLO PAD: edge_low=[1,0], edge_high=[0,1], interior=[0,0]."""
    mod = _load_model_from_buffer(
        _build_stablehlo_pad_model(edge_low=[1, 0], edge_high=[0, 1], interior=[0, 0])
    )

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((3, 3), dtype="float32"),
        ) -> R.Tensor((4, 4), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((4, 4), dtype="float32") = R.nn.pad(
                    x, pad_width=[1, 0, 0, 1], pad_value=0.0
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_stablehlo_pad_interior_unsupported():
    """STABLEHLO_PAD with interior padding raises NotImplementedError."""
    builder = flatbuffers.Builder(1024)

    lo_vec = _pad_vector(
        builder,
        _tfl_stablehlo_pad_opts.StablehloPadOptionsStartEdgePaddingLowVector,
        [0, 0],
    )
    hi_vec = _pad_vector(
        builder,
        _tfl_stablehlo_pad_opts.StablehloPadOptionsStartEdgePaddingHighVector,
        [0, 0],
    )
    int_vec = _pad_vector(
        builder,
        _tfl_stablehlo_pad_opts.StablehloPadOptionsStartInteriorPaddingVector,
        [1, 0],
    )

    _tfl_stablehlo_pad_opts.StablehloPadOptionsStart(builder)
    _tfl_stablehlo_pad_opts.StablehloPadOptionsAddEdgePaddingLow(builder, lo_vec)
    _tfl_stablehlo_pad_opts.StablehloPadOptionsAddEdgePaddingHigh(builder, hi_vec)
    _tfl_stablehlo_pad_opts.StablehloPadOptionsAddInteriorPadding(builder, int_vec)
    pad_opts = _tfl_stablehlo_pad_opts.StablehloPadOptionsEnd(builder)

    builtin_op = _get_stablehlo_builtin_operator("STABLEHLO_PAD")
    op_code = _build_operator_code(builder, builtin_op)

    t_in = _build_tensor(builder, 0, [3, 3])
    t_pv = _build_tensor(builder, 1, [])
    t_out = _build_tensor(builder, 2, [3, 3])
    tensors = [t_in, t_pv, t_out]

    op = _build_operator(
        builder,
        0,
        [0, 1],
        [2],
        builtin_options2_type=_tfl_builtin_options2.StablehloPadOptions,
        builtin_options2=pad_opts,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=tensors,
        operators=[op],
        inputs=[0],
        outputs=[2],
    )
    buffers = [
        _build_buffer(builder),
        _build_buffer(builder, np.array([0.0], dtype=np.float32).tobytes()),
        _build_buffer(builder),
    ]
    buf = _finish_tflite_model(
        builder, subgraph=subgraph, operator_codes=[op_code], buffers=buffers
    )
    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)
    with pytest.raises(NotImplementedError, match="interior"):
        from_tflite(tflite_model)


def test_stablehlo_pad_negative_unsupported():
    """STABLEHLO_PAD with negative edge padding raises NotImplementedError."""
    builder = flatbuffers.Builder(1024)

    lo_vec = _pad_vector(
        builder,
        _tfl_stablehlo_pad_opts.StablehloPadOptionsStartEdgePaddingLowVector,
        [-1, 0],
    )
    hi_vec = _pad_vector(
        builder,
        _tfl_stablehlo_pad_opts.StablehloPadOptionsStartEdgePaddingHighVector,
        [0, 0],
    )
    int_vec = _pad_vector(
        builder,
        _tfl_stablehlo_pad_opts.StablehloPadOptionsStartInteriorPaddingVector,
        [0, 0],
    )

    _tfl_stablehlo_pad_opts.StablehloPadOptionsStart(builder)
    _tfl_stablehlo_pad_opts.StablehloPadOptionsAddEdgePaddingLow(builder, lo_vec)
    _tfl_stablehlo_pad_opts.StablehloPadOptionsAddEdgePaddingHigh(builder, hi_vec)
    _tfl_stablehlo_pad_opts.StablehloPadOptionsAddInteriorPadding(builder, int_vec)
    pad_opts = _tfl_stablehlo_pad_opts.StablehloPadOptionsEnd(builder)

    builtin_op = _get_stablehlo_builtin_operator("STABLEHLO_PAD")
    op_code = _build_operator_code(builder, builtin_op)

    t_in = _build_tensor(builder, 0, [3, 3])
    t_pv = _build_tensor(builder, 1, [])
    t_out = _build_tensor(builder, 2, [2, 3])
    tensors = [t_in, t_pv, t_out]

    op = _build_operator(
        builder,
        0,
        [0, 1],
        [2],
        builtin_options2_type=_tfl_builtin_options2.StablehloPadOptions,
        builtin_options2=pad_opts,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=tensors,
        operators=[op],
        inputs=[0],
        outputs=[2],
    )
    buffers = [
        _build_buffer(builder),
        _build_buffer(builder, np.array([0.0], dtype=np.float32).tobytes()),
        _build_buffer(builder),
    ]
    buf = _finish_tflite_model(
        builder, subgraph=subgraph, operator_codes=[op_code], buffers=buffers
    )
    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)
    with pytest.raises(NotImplementedError, match="negative"):
        from_tflite(tflite_model)


def _build_stablehlo_dynamic_slice_model(slice_sizes, start_vals):
    """STABLEHLO_DYNAMIC_SLICE with given slice sizes and start indices."""
    builder = flatbuffers.Builder(1024)
    ndim = len(slice_sizes)

    # Build SliceSizes vector
    _tfl_stablehlo_dyn_slice_opts.StablehloDynamicSliceOptionsStartSliceSizesVector(builder, ndim)
    for v in reversed(slice_sizes):
        builder.PrependInt64(v)
    sizes_vec = builder.EndVector()

    _tfl_stablehlo_dyn_slice_opts.StablehloDynamicSliceOptionsStart(builder)
    _tfl_stablehlo_dyn_slice_opts.StablehloDynamicSliceOptionsAddSliceSizes(builder, sizes_vec)
    dyn_opts = _tfl_stablehlo_dyn_slice_opts.StablehloDynamicSliceOptionsEnd(builder)

    builtin_op = _get_stablehlo_builtin_operator("STABLEHLO_DYNAMIC_SLICE")
    op_code = _build_operator_code(builder, builtin_op)

    # operand + start indices + output
    t_in = _build_tensor(builder, 0, [3, 3])
    start_tensors = []
    start_inputs = []
    start_buffers = []
    for i, sv in enumerate(start_vals):
        bidx = 1 + i
        start_tensors.append(_build_tensor(builder, bidx, [], tensor_type=_tfl_tensor_type.INT32))
        start_inputs.append(bidx)
        start_buffers.append(_build_buffer(builder, np.array([sv], dtype=np.int32).tobytes()))
    out_idx = 1 + ndim
    t_out = _build_tensor(builder, out_idx, slice_sizes)
    tensors = [t_in, *start_tensors, t_out]
    op_inputs = [0, *start_inputs]

    op = _build_operator(
        builder,
        0,
        op_inputs,
        [out_idx],
        builtin_options2_type=_tfl_builtin_options2.StablehloDynamicSliceOptions,
        builtin_options2=dyn_opts,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=tensors,
        operators=[op],
        inputs=[0],
        outputs=[out_idx],
    )
    buffers = [_build_buffer(builder), *start_buffers, _build_buffer(builder)]
    return _finish_tflite_model(
        builder, subgraph=subgraph, operator_codes=[op_code], buffers=buffers
    )


def _build_stablehlo_dynamic_slice_with_dynamic_starts_model(slice_sizes):
    """STABLEHLO_DYNAMIC_SLICE with runtime start-index inputs."""
    builder = flatbuffers.Builder(1024)
    ndim = len(slice_sizes)

    _tfl_stablehlo_dyn_slice_opts.StablehloDynamicSliceOptionsStartSliceSizesVector(builder, ndim)
    for v in reversed(slice_sizes):
        builder.PrependInt64(v)
    sizes_vec = builder.EndVector()

    _tfl_stablehlo_dyn_slice_opts.StablehloDynamicSliceOptionsStart(builder)
    _tfl_stablehlo_dyn_slice_opts.StablehloDynamicSliceOptionsAddSliceSizes(builder, sizes_vec)
    dyn_opts = _tfl_stablehlo_dyn_slice_opts.StablehloDynamicSliceOptionsEnd(builder)

    builtin_op = _get_stablehlo_builtin_operator("STABLEHLO_DYNAMIC_SLICE")
    op_code = _build_operator_code(builder, builtin_op)

    t_in = _build_tensor(builder, 0, [3, 3])
    start_tensors = [
        _build_tensor(builder, 1 + i, [], tensor_type=_tfl_tensor_type.INT32) for i in range(ndim)
    ]
    out_idx = 1 + ndim
    t_out = _build_tensor(builder, out_idx, slice_sizes)
    start_inputs = list(range(1, 1 + ndim))
    tensors = [t_in, *start_tensors, t_out]
    op_inputs = [0, *start_inputs]

    op = _build_operator(
        builder,
        0,
        op_inputs,
        [out_idx],
        builtin_options2_type=_tfl_builtin_options2.StablehloDynamicSliceOptions,
        builtin_options2=dyn_opts,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=tensors,
        operators=[op],
        inputs=op_inputs,
        outputs=[out_idx],
    )
    buffers = [_build_buffer(builder) for _ in range(out_idx + 1)]
    return _finish_tflite_model(
        builder, subgraph=subgraph, operator_codes=[op_code], buffers=buffers
    )


def test_stablehlo_dynamic_slice():
    """TFLite StableHLO DYNAMIC_SLICE: start=[0,1], sizes=[2,2] from (3,3)."""
    mod = _load_model_from_buffer(
        _build_stablehlo_dynamic_slice_model(slice_sizes=[2, 2], start_vals=[0, 1])
    )

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((3, 3), dtype="float32"),
        ) -> R.Tensor(dtype="float32", ndim=2):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor(dtype="float32", ndim=2) = R.dynamic_strided_slice(
                    x,
                    R.const([0, 1], dtype="int64"),
                    R.const([2, 3], dtype="int64"),
                    R.const([1, 1], dtype="int64"),
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_stablehlo_dynamic_slice_dynamic_starts_unsupported():
    """TFLite StableHLO DYNAMIC_SLICE with runtime starts is not supported yet."""
    buf = _build_stablehlo_dynamic_slice_with_dynamic_starts_model(slice_sizes=[2, 2])
    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)

    with pytest.raises(NotImplementedError, match="dynamic start"):
        from_tflite(tflite_model)


def test_stablehlo_dynamic_slice_out_of_bounds_unsupported():
    """TFLite StableHLO DYNAMIC_SLICE with out-of-bounds starts is not supported."""
    buf = _build_stablehlo_dynamic_slice_model(slice_sizes=[2, 2], start_vals=[0, 2])
    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)

    with pytest.raises(NotImplementedError, match="out-of-bounds"):
        from_tflite(tflite_model)


def test_stablehlo_cbrt():
    """TFLite StableHLO CBRT uses a sign-preserving composite expression."""
    mod = _load_model_from_buffer(
        _build_stablehlo_model(builtin_name="STABLEHLO_CBRT", input_count=1)
    )

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 2), dtype="float32")) -> R.Tensor((2, 2), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 2), dtype="float32") = R.negative(x)
                lv1: R.Tensor((2, 2), dtype="float32") = R.power(lv, R.const(1.0 / 3.0, "float32"))
                lv2: R.Tensor((2, 2), dtype="bool") = R.less(x, R.const(0, "float32"))
                lv3: R.Tensor((2, 2), dtype="float32") = R.negative(lv1)
                lv4: R.Tensor((2, 2), dtype="float32") = R.power(x, R.const(1.0 / 3.0, "float32"))
                gv: R.Tensor((2, 2), dtype="float32") = R.where(lv2, lv3, lv4)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_stablehlo_remainder():
    """TFLite StableHLO REMAINDER uses truncating remainder semantics."""
    mod = _load_model_from_buffer(
        _build_stablehlo_model(builtin_name="STABLEHLO_REMAINDER", input_count=2)
    )

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 2), dtype="float32"),
            y: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tensor((2, 2), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((2, 2), dtype="float32") = R.divide(x, y)
                lv1: R.Tensor((2, 2), dtype="float32") = R.trunc(lv)
                lv2: R.Tensor((2, 2), dtype="float32") = R.multiply(y, lv1)
                gv: R.Tensor((2, 2), dtype="float32") = R.subtract(x, lv2)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def _build_stablehlo_dynamic_update_slice_model(start_vals, dynamic_starts=False):
    """Build a minimal STABLEHLO_DYNAMIC_UPDATE_SLICE model."""
    builder = flatbuffers.Builder(1024)
    builtin_op = _get_stablehlo_builtin_operator("STABLEHLO_DYNAMIC_UPDATE_SLICE")
    op_code = _build_operator_code(builder, builtin_op)

    t_operand = _build_tensor(builder, 0, [3, 4])
    t_update = _build_tensor(builder, 1, [2, 2])
    start_tensors = [
        _build_tensor(builder, 2 + i, [], tensor_type=_tfl_tensor_type.INT32)
        for i in range(len(start_vals))
    ]
    out_idx = 2 + len(start_vals)
    t_out = _build_tensor(builder, out_idx, [3, 4])
    tensors = [t_operand, t_update, *start_tensors, t_out]

    op_inputs = [0, 1, *range(2, out_idx)]
    op = _build_operator(builder, 0, op_inputs, [out_idx])
    subgraph_inputs = op_inputs if dynamic_starts else [0, 1]
    subgraph = _build_subgraph(
        builder,
        tensors=tensors,
        operators=[op],
        inputs=subgraph_inputs,
        outputs=[out_idx],
    )
    if dynamic_starts:
        buffers = [_build_buffer(builder) for _ in range(out_idx + 1)]
    else:
        start_buffers = [
            _build_buffer(builder, np.array([start], dtype=np.int32).tobytes())
            for start in start_vals
        ]
        buffers = [
            _build_buffer(builder),
            _build_buffer(builder),
            *start_buffers,
            _build_buffer(builder),
        ]

    return _finish_tflite_model(
        builder, subgraph=subgraph, operator_codes=[op_code], buffers=buffers
    )


def test_stablehlo_dynamic_update_slice():
    """TFLite StableHLO DYNAMIC_UPDATE_SLICE with static starts."""
    mod = _load_model_from_buffer(_build_stablehlo_dynamic_update_slice_model([1, 1]))

    @I.ir_module
    class Expected:
        @R.function
        def main(
            operand: R.Tensor((3, 4), dtype="float32"),
            update: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tensor((3, 4), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                gv: R.Tensor((3, 4), dtype="float32") = R.scatter_nd(
                    operand,
                    R.const([[[1, 1], [1, 2]], [[2, 1], [2, 2]]], dtype="int64"),
                    update,
                    reduction="update",
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_stablehlo_dynamic_update_slice_dynamic_starts():
    """TFLite StableHLO DYNAMIC_UPDATE_SLICE with runtime starts lowers structurally."""
    mod = _load_model_from_buffer(
        _build_stablehlo_dynamic_update_slice_model([0, 0], dynamic_starts=True)
    )

    @I.ir_module
    class Expected:
        @R.function
        def main(
            operand: R.Tensor((3, 4), dtype="float32"),
            update: R.Tensor((2, 2), dtype="float32"),
            s0: R.Tensor((), dtype="int32"),
            s1: R.Tensor((), dtype="int32"),
        ) -> R.Tensor((3, 4), dtype="float32"):
            R.func_attr({"num_input": 4})
            with R.dataflow():
                lv: R.Tensor((2,), dtype="int64") = R.arange(0, 2, 1, dtype="int64")
                lv1: R.Tensor((), dtype="int64") = R.astype(s0, dtype="int64")
                lv2: R.Tensor((), dtype="int64") = R.maximum(lv1, R.const(0, "int64"))
                lv3: R.Tensor((), dtype="int64") = R.minimum(lv2, R.const(1, "int64"))
                lv4: R.Tensor((2,), dtype="int64") = R.add(lv, lv3)
                lv5: R.Tensor((2, 1), dtype="int64") = R.reshape(lv4, (2, 1))
                lv6: R.Tensor((2, 2), dtype="int64") = R.broadcast_to(lv5, (2, 2))
                lv7: R.Tensor((2,), dtype="int64") = R.arange(0, 2, 1, dtype="int64")
                lv8: R.Tensor((), dtype="int64") = R.astype(s1, dtype="int64")
                lv9: R.Tensor((), dtype="int64") = R.maximum(lv8, R.const(0, "int64"))
                lv10: R.Tensor((), dtype="int64") = R.minimum(lv9, R.const(2, "int64"))
                lv11: R.Tensor((2,), dtype="int64") = R.add(lv7, lv10)
                lv12: R.Tensor((1, 2), dtype="int64") = R.reshape(lv11, (1, 2))
                lv13: R.Tensor((2, 2), dtype="int64") = R.broadcast_to(lv12, (2, 2))
                lv14: R.Tensor((2, 2, 1), dtype="int64") = R.expand_dims(lv6, axis=[-1])
                lv15: R.Tensor((2, 2, 1), dtype="int64") = R.expand_dims(lv13, axis=[-1])
                lv16: R.Tensor((2, 2, 2), dtype="int64") = R.concat((lv14, lv15), axis=-1)
                gv: R.Tensor((3, 4), dtype="float32") = R.scatter_nd(
                    operand, lv16, update, reduction="update"
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_stablehlo_dynamic_update_slice_out_of_bounds_unsupported():
    """TFLite StableHLO DYNAMIC_UPDATE_SLICE rejects out-of-bounds updates."""
    buf = _build_stablehlo_dynamic_update_slice_model([2, 3])
    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)

    with pytest.raises(NotImplementedError, match="out-of-bounds"):
        from_tflite(tflite_model)


def _build_stablehlo_dot_general_model(lhs_contract, rhs_contract, lhs_batch=None, rhs_batch=None):
    """Build a minimal STABLEHLO_DOT_GENERAL model."""
    builder = flatbuffers.Builder(1024)
    lhs_batch = [] if lhs_batch is None else lhs_batch
    rhs_batch = [] if rhs_batch is None else rhs_batch

    lhs_batch_vec = _tflite_int64_vector(
        builder,
        _tfl_stablehlo_dot_opts.StablehloDotGeneralOptionsStartLhsBatchingDimensionsVector,
        lhs_batch,
    )
    rhs_batch_vec = _tflite_int64_vector(
        builder,
        _tfl_stablehlo_dot_opts.StablehloDotGeneralOptionsStartRhsBatchingDimensionsVector,
        rhs_batch,
    )
    lhs_contract_vec = _tflite_int64_vector(
        builder,
        _tfl_stablehlo_dot_opts.StablehloDotGeneralOptionsStartLhsContractingDimensionsVector,
        lhs_contract,
    )
    rhs_contract_vec = _tflite_int64_vector(
        builder,
        _tfl_stablehlo_dot_opts.StablehloDotGeneralOptionsStartRhsContractingDimensionsVector,
        rhs_contract,
    )

    _tfl_stablehlo_dot_opts.StablehloDotGeneralOptionsStart(builder)
    _tfl_stablehlo_dot_opts.StablehloDotGeneralOptionsAddLhsBatchingDimensions(
        builder, lhs_batch_vec
    )
    _tfl_stablehlo_dot_opts.StablehloDotGeneralOptionsAddRhsBatchingDimensions(
        builder, rhs_batch_vec
    )
    _tfl_stablehlo_dot_opts.StablehloDotGeneralOptionsAddLhsContractingDimensions(
        builder, lhs_contract_vec
    )
    _tfl_stablehlo_dot_opts.StablehloDotGeneralOptionsAddRhsContractingDimensions(
        builder, rhs_contract_vec
    )
    dot_opts = _tfl_stablehlo_dot_opts.StablehloDotGeneralOptionsEnd(builder)

    builtin_op = _get_stablehlo_builtin_operator("STABLEHLO_DOT_GENERAL")
    op_code = _build_operator_code(builder, builtin_op)
    t_lhs = _build_tensor(builder, 0, [2, 3])
    t_rhs = _build_tensor(builder, 1, [3, 4])
    t_out = _build_tensor(builder, 2, [2, 4])
    op = _build_operator(
        builder,
        0,
        [0, 1],
        [2],
        builtin_options2_type=_tfl_builtin_options2.StablehloDotGeneralOptions,
        builtin_options2=dot_opts,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=[t_lhs, t_rhs, t_out],
        operators=[op],
        inputs=[0, 1],
        outputs=[2],
    )
    buffers = [_build_buffer(builder) for _ in range(3)]
    return _finish_tflite_model(
        builder, subgraph=subgraph, operator_codes=[op_code], buffers=buffers
    )


def test_stablehlo_dot_general():
    """TFLite StableHLO DOT_GENERAL canonical 2D matmul."""
    mod = _load_model_from_buffer(_build_stablehlo_dot_general_model([1], [0]))

    @I.ir_module
    class Expected:
        @R.function
        def main(
            lhs: R.Tensor((2, 3), dtype="float32"),
            rhs: R.Tensor((3, 4), dtype="float32"),
        ) -> R.Tensor((2, 4), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                gv: R.Tensor((2, 4), dtype="float32") = R.matmul(lhs, rhs)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_stablehlo_dot_general_noncanonical_unsupported():
    """TFLite StableHLO DOT_GENERAL rejects non-canonical contracting dims."""
    buf = _build_stablehlo_dot_general_model([0], [0])
    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)

    with pytest.raises(NotImplementedError, match="contracting"):
        from_tflite(tflite_model)


def _build_stablehlo_convolution_model(feature_group_count=1, input_batch_dimension=0):
    """Build a minimal STABLEHLO_CONVOLUTION model."""
    builder = flatbuffers.Builder(1024)

    window_strides_vec = _tflite_int64_vector(
        builder,
        _tfl_stablehlo_conv_opts.StablehloConvolutionOptionsStartWindowStridesVector,
        [1, 1],
    )
    padding_vec = _tflite_int64_vector(
        builder,
        _tfl_stablehlo_conv_opts.StablehloConvolutionOptionsStartPaddingVector,
        [0, 0, 0, 0],
    )
    lhs_dilation_vec = _tflite_int64_vector(
        builder, _tfl_stablehlo_conv_opts.StablehloConvolutionOptionsStartLhsDilationVector, [1, 1]
    )
    rhs_dilation_vec = _tflite_int64_vector(
        builder, _tfl_stablehlo_conv_opts.StablehloConvolutionOptionsStartRhsDilationVector, [1, 1]
    )
    window_reversal_vec = _tflite_bool_vector(
        builder,
        _tfl_stablehlo_conv_opts.StablehloConvolutionOptionsStartWindowReversalVector,
        [False, False],
    )
    input_spatial_vec = _tflite_int64_vector(
        builder,
        _tfl_stablehlo_conv_opts.StablehloConvolutionOptionsStartInputSpatialDimensionsVector,
        [1, 2],
    )
    kernel_spatial_vec = _tflite_int64_vector(
        builder,
        _tfl_stablehlo_conv_opts.StablehloConvolutionOptionsStartKernelSpatialDimensionsVector,
        [0, 1],
    )
    output_spatial_vec = _tflite_int64_vector(
        builder,
        _tfl_stablehlo_conv_opts.StablehloConvolutionOptionsStartOutputSpatialDimensionsVector,
        [1, 2],
    )

    _tfl_stablehlo_conv_opts.StablehloConvolutionOptionsStart(builder)
    _tfl_stablehlo_conv_opts.StablehloConvolutionOptionsAddWindowStrides(
        builder, window_strides_vec
    )
    _tfl_stablehlo_conv_opts.StablehloConvolutionOptionsAddPadding(builder, padding_vec)
    _tfl_stablehlo_conv_opts.StablehloConvolutionOptionsAddLhsDilation(builder, lhs_dilation_vec)
    _tfl_stablehlo_conv_opts.StablehloConvolutionOptionsAddRhsDilation(builder, rhs_dilation_vec)
    _tfl_stablehlo_conv_opts.StablehloConvolutionOptionsAddWindowReversal(
        builder, window_reversal_vec
    )
    _tfl_stablehlo_conv_opts.StablehloConvolutionOptionsAddInputBatchDimension(
        builder, input_batch_dimension
    )
    _tfl_stablehlo_conv_opts.StablehloConvolutionOptionsAddInputFeatureDimension(builder, 3)
    _tfl_stablehlo_conv_opts.StablehloConvolutionOptionsAddInputSpatialDimensions(
        builder, input_spatial_vec
    )
    _tfl_stablehlo_conv_opts.StablehloConvolutionOptionsAddKernelInputFeatureDimension(builder, 2)
    _tfl_stablehlo_conv_opts.StablehloConvolutionOptionsAddKernelOutputFeatureDimension(builder, 3)
    _tfl_stablehlo_conv_opts.StablehloConvolutionOptionsAddKernelSpatialDimensions(
        builder, kernel_spatial_vec
    )
    _tfl_stablehlo_conv_opts.StablehloConvolutionOptionsAddOutputBatchDimension(builder, 0)
    _tfl_stablehlo_conv_opts.StablehloConvolutionOptionsAddOutputFeatureDimension(builder, 3)
    _tfl_stablehlo_conv_opts.StablehloConvolutionOptionsAddOutputSpatialDimensions(
        builder, output_spatial_vec
    )
    _tfl_stablehlo_conv_opts.StablehloConvolutionOptionsAddFeatureGroupCount(
        builder, feature_group_count
    )
    _tfl_stablehlo_conv_opts.StablehloConvolutionOptionsAddBatchGroupCount(builder, 1)
    conv_opts = _tfl_stablehlo_conv_opts.StablehloConvolutionOptionsEnd(builder)

    builtin_op = _get_stablehlo_builtin_operator("STABLEHLO_CONVOLUTION")
    op_code = _build_operator_code(builder, builtin_op)
    t_data = _build_tensor(builder, 0, [1, 5, 5, 2])
    t_kernel = _build_tensor(builder, 1, [3, 3, 2, 4])
    t_out = _build_tensor(builder, 2, [1, 3, 3, 4])
    op = _build_operator(
        builder,
        0,
        [0, 1],
        [2],
        builtin_options2_type=_tfl_builtin_options2.StablehloConvolutionOptions,
        builtin_options2=conv_opts,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=[t_data, t_kernel, t_out],
        operators=[op],
        inputs=[0, 1],
        outputs=[2],
    )
    buffers = [_build_buffer(builder) for _ in range(3)]
    return _finish_tflite_model(
        builder, subgraph=subgraph, operator_codes=[op_code], buffers=buffers
    )


def test_stablehlo_convolution():
    """TFLite StableHLO CONVOLUTION canonical NHWC/HWIO 2D convolution."""
    mod = _load_model_from_buffer(_build_stablehlo_convolution_model())

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data: R.Tensor((1, 5, 5, 2), dtype="float32"),
            kernel: R.Tensor((3, 3, 2, 4), dtype="float32"),
        ) -> R.Tensor((1, 3, 3, 4), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                gv: R.Tensor((1, 3, 3, 4), dtype="float32") = R.nn.conv2d(
                    data,
                    kernel,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="HWIO",
                    out_layout="NHWC",
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_stablehlo_convolution_feature_group_unsupported():
    """TFLite StableHLO CONVOLUTION rejects grouped convolution in the first subset."""
    buf = _build_stablehlo_convolution_model(feature_group_count=2)
    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)

    with pytest.raises(NotImplementedError, match="feature_group_count"):
        from_tflite(tflite_model)


def test_stablehlo_convolution_dimension_numbers_unsupported():
    """TFLite StableHLO CONVOLUTION rejects non-canonical dimension numbers."""
    buf = _build_stablehlo_convolution_model(input_batch_dimension=1)
    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)

    with pytest.raises(NotImplementedError, match="dimension numbers"):
        from_tflite(tflite_model)


# Quantized TFLite QDQ tests


def test_tensor_quantization_parameters_are_parsed():
    """Tensor quantization metadata is kept without requiring quantized op support."""
    builder = flatbuffers.Builder(1024)

    per_tensor_quantization = _build_quantization_parameters(
        builder, scale=[0.5], zero_point=[3], quantized_dimension=0
    )
    per_axis_quantization = _build_quantization_parameters(
        builder, scale=[0.25, 0.75], zero_point=[0, 0], quantized_dimension=3
    )
    per_tensor = _build_tensor(
        builder,
        0,
        [1, 4],
        tensor_type=_tfl_tensor_type.UINT8,
        quantization=per_tensor_quantization,
    )
    per_axis = _build_tensor(
        builder,
        1,
        [1, 2, 3, 2],
        tensor_type=_tfl_tensor_type.INT8,
        quantization=per_axis_quantization,
    )
    subgraph = _build_subgraph(
        builder, tensors=[per_tensor, per_axis], operators=[], inputs=[0, 1], outputs=[0, 1]
    )
    buffers = [_build_buffer(builder), _build_buffer(builder)]
    buf = _finish_tflite_model(builder, subgraph=subgraph, operator_codes=[], buffers=buffers)

    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)

    converter = tflite_frontend.OperatorConverter(
        tflite_model, tflite_model.Subgraphs(0), tflite_frontend.ExprTable(), None
    )
    per_tensor_wrapper, per_axis_wrapper = converter.get_tensors([0, 1])

    np.testing.assert_allclose(per_tensor_wrapper.qnn_params["scale"].value.numpy(), 0.5)
    np.testing.assert_equal(per_tensor_wrapper.qnn_params["zero_point"].value.numpy(), 3)
    assert per_tensor_wrapper.qnn_params["axis"] == 0

    np.testing.assert_allclose(
        per_axis_wrapper.qnn_params["scale"].value.numpy(), np.array([0.25, 0.75])
    )
    np.testing.assert_equal(per_axis_wrapper.qnn_params["zero_point"].value.numpy(), 0)
    assert per_axis_wrapper.qnn_params["axis"] == 3

    mod = from_tflite(tflite_model)
    assert len(mod["main"].params) == 2


def test_quantize_op_uses_relax_quantize():
    """TFLite QUANTIZE float32 -> int8 uses R.quantize."""
    builder = flatbuffers.Builder(1024)

    input_data = np.array([1.0, 2.0], dtype=np.float32)
    output_qparams = _build_quantization_parameters(
        builder, scale=[0.5], zero_point=[3], quantized_dimension=0
    )

    input_tensor = _build_tensor(builder, 0, [2], tensor_type=_tfl_tensor_type.FLOAT32)
    output_tensor = _build_tensor(
        builder,
        1,
        [2],
        tensor_type=_tfl_tensor_type.INT8,
        quantization=output_qparams,
    )

    quantize_op = _build_operator(builder, 0, [0], [1])
    subgraph = _build_subgraph(
        builder,
        tensors=[input_tensor, output_tensor],
        operators=[quantize_op],
        inputs=[0],
        outputs=[1],
    )
    operator_codes = [_build_operator_code(builder, _tfl_builtin_operator.QUANTIZE)]
    input_buffer = _build_buffer(builder, input_data.tobytes())
    output_buffer = _build_buffer(builder)
    buf = _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=operator_codes,
        buffers=[input_buffer, output_buffer],
    )

    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)
    mod = from_tflite(tflite_model)
    mod["main"] = mod["main"].without_attr("params")

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2,), dtype="float32")) -> R.Tensor((2,), dtype="int8"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((2,), dtype="int8") = R.quantize(
                    x,
                    R.const(0.5, "float32"),
                    R.const(3, "int32"),
                    axis=0,
                    out_dtype="int8",
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_quantize_op_requantize_uses_dq_q():
    """TFLite QUANTIZE with quantized input uses DQ→Q (requantize)."""
    builder = flatbuffers.Builder(1024)

    input_data = np.array([10, 20], dtype=np.int8)
    input_qparams = _build_quantization_parameters(
        builder, scale=[0.25], zero_point=[1], quantized_dimension=0
    )
    output_qparams = _build_quantization_parameters(
        builder, scale=[0.5], zero_point=[3], quantized_dimension=0
    )

    input_tensor = _build_tensor(
        builder,
        0,
        [2],
        tensor_type=_tfl_tensor_type.INT8,
        quantization=input_qparams,
    )
    output_tensor = _build_tensor(
        builder,
        1,
        [2],
        tensor_type=_tfl_tensor_type.INT8,
        quantization=output_qparams,
    )

    quantize_op = _build_operator(
        builder,
        0,
        [0],
        [1],
    )
    subgraph = _build_subgraph(
        builder,
        tensors=[input_tensor, output_tensor],
        operators=[quantize_op],
        inputs=[0],
        outputs=[1],
    )
    operator_codes = [
        _build_operator_code(builder, _tfl_builtin_operator.QUANTIZE),
    ]
    input_buffer = _build_buffer(builder, input_data.tobytes())
    output_buffer = _build_buffer(builder)
    buf = _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=operator_codes,
        buffers=[input_buffer, output_buffer],
    )

    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)
    mod = from_tflite(tflite_model)
    mod["main"] = mod["main"].without_attr("params")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            tvmgen_tensor_0: R.Tensor((2,), dtype="int8"),
        ) -> R.Tensor((2,), dtype="int8"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2,), dtype="float32") = R.dequantize(
                    tvmgen_tensor_0,
                    R.const(0.25, "float32"),
                    R.const(1, "int32"),
                    out_dtype="float32",
                    axis=0,
                )
                gv: R.Tensor((2,), dtype="int8") = R.quantize(
                    lv,
                    R.const(0.5, "float32"),
                    R.const(3, "int32"),
                    out_dtype="int8",
                    axis=0,
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_dequantize_op_uses_relax_dequantize():
    """TFLite DEQUANTIZE int8 -> float32 uses R.dequantize."""
    builder = flatbuffers.Builder(1024)

    input_data = np.array([10, 20], dtype=np.int8)
    input_qparams = _build_quantization_parameters(
        builder, scale=[0.5], zero_point=[3], quantized_dimension=0
    )

    input_tensor = _build_tensor(
        builder,
        0,
        [2],
        tensor_type=_tfl_tensor_type.INT8,
        quantization=input_qparams,
    )
    output_tensor = _build_tensor(builder, 1, [2], tensor_type=_tfl_tensor_type.FLOAT32)

    dequantize_op = _build_operator(builder, 0, [0], [1])
    subgraph = _build_subgraph(
        builder,
        tensors=[input_tensor, output_tensor],
        operators=[dequantize_op],
        inputs=[0],
        outputs=[1],
    )
    operator_codes = [_build_operator_code(builder, _tfl_builtin_operator.DEQUANTIZE)]
    input_buffer = _build_buffer(builder, input_data.tobytes())
    output_buffer = _build_buffer(builder)
    buf = _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=operator_codes,
        buffers=[input_buffer, output_buffer],
    )

    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)
    mod = from_tflite(tflite_model)
    mod["main"] = mod["main"].without_attr("params")

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2,), dtype="int8")) -> R.Tensor((2,), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((2,), dtype="float32") = R.dequantize(
                    x,
                    R.const(0.5, "float32"),
                    R.const(3, "int32"),
                    axis=0,
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_dequantize_float16_uses_astype():
    """TFLite DEQUANTIZE float16 -> float32 uses R.astype."""
    builder = flatbuffers.Builder(1024)

    input_data = np.array([1.5, -2.0], dtype=np.float16)

    input_tensor = _build_tensor(builder, 0, [2], tensor_type=_tfl_tensor_type.FLOAT16)
    output_tensor = _build_tensor(builder, 1, [2], tensor_type=_tfl_tensor_type.FLOAT32)

    dequantize_op = _build_operator(builder, 0, [0], [1])
    subgraph = _build_subgraph(
        builder,
        tensors=[input_tensor, output_tensor],
        operators=[dequantize_op],
        inputs=[],
        outputs=[1],
    )
    operator_codes = [_build_operator_code(builder, _tfl_builtin_operator.DEQUANTIZE)]
    input_buffer = _build_buffer(builder, input_data.tobytes())
    output_buffer = _build_buffer(builder)
    buf = _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=operator_codes,
        buffers=[input_buffer, output_buffer],
    )

    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)
    mod = from_tflite(tflite_model)
    mod["main"] = mod["main"].without_attr("params")

    @I.ir_module
    class Expected:
        @R.function
        def main() -> R.Tensor((2,), dtype="float32"):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                gv: R.Tensor((2,), dtype="float32") = R.astype(
                    R.const(np.array([1.5, -2.0], dtype=np.float16)), dtype="float32"
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_quantized_avg_pool2d_uses_astype():
    """Quantized AVERAGE_POOL_2D casts through int32 with R.astype."""
    builder = flatbuffers.Builder(1024)

    qparams = _build_quantization_parameters(
        builder, scale=[0.5], zero_point=[3], quantized_dimension=0
    )
    input_tensor = _build_tensor(
        builder,
        0,
        [1, 2, 2, 1],
        tensor_type=_tfl_tensor_type.INT8,
        quantization=qparams,
    )
    output_tensor = _build_tensor(
        builder,
        1,
        [1, 1, 1, 1],
        tensor_type=_tfl_tensor_type.INT8,
        quantization=qparams,
    )

    _tfl_pool2d_options.Pool2DOptionsStart(builder)
    _tfl_pool2d_options.Pool2DOptionsAddPadding(builder, _tfl_padding.VALID)
    _tfl_pool2d_options.Pool2DOptionsAddStrideH(builder, 1)
    _tfl_pool2d_options.Pool2DOptionsAddStrideW(builder, 1)
    _tfl_pool2d_options.Pool2DOptionsAddFilterHeight(builder, 2)
    _tfl_pool2d_options.Pool2DOptionsAddFilterWidth(builder, 2)
    _tfl_pool2d_options.Pool2DOptionsAddFusedActivationFunction(builder, _tfl_activation_fn.NONE)
    pool_opts = _tfl_pool2d_options.Pool2DOptionsEnd(builder)

    avg_pool_op = _build_operator(
        builder,
        0,
        [0],
        [1],
        builtin_options_type=_tfl_builtin_options.Pool2DOptions,
        builtin_options=pool_opts,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=[input_tensor, output_tensor],
        operators=[avg_pool_op],
        inputs=[0],
        outputs=[1],
    )
    operator_codes = [_build_operator_code(builder, _tfl_builtin_operator.AVERAGE_POOL_2D)]
    buf = _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=operator_codes,
        buffers=[_build_buffer(builder), _build_buffer(builder)],
    )

    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)

    # Exercise the public entry point so quantized pool allowlist regressions
    # cannot be hidden by calling the converter method directly.
    from_tflite(tflite_model)

    subgraph = tflite_model.Subgraphs(0)
    bb = relax.BlockBuilder()
    exp_tab = tflite_frontend.ExprTable()
    input_var = relax.Var("tvmgen_tensor_0", relax.TensorType((1, 2, 2, 1), dtype="int8"))
    exp_tab.set_expr("tvmgen_tensor_0", input_var)
    converter = tflite_frontend.OperatorConverter(tflite_model, subgraph, exp_tab, bb)
    with bb.function("main", [input_var]):
        with bb.dataflow():
            output = converter.convert_pool2d(subgraph.Operators(0), "average")
            gv = bb.emit_output(output)
        bb.emit_func_output(gv)
    mod = bb.get()

    @I.ir_module
    class Expected:
        @R.function
        def main(tvmgen_tensor_0: R.Tensor((1, 2, 2, 1), dtype="int8")) -> R.Tensor(
            (1, 1, 1, 1), dtype="int8"
        ):
            with R.dataflow():
                lv: R.Tensor((1, 2, 2, 1), dtype="int32") = R.astype(tvmgen_tensor_0, dtype="int32")
                lv1: R.Tensor((1, 1, 1, 1), dtype="int32") = R.nn.avg_pool2d(
                    lv,
                    pool_size=[2, 2],
                    strides=[1, 1],
                    dilation=[1, 1],
                    padding=[0, 0, 0, 0],
                    ceil_mode=False,
                    count_include_pad=False,
                    layout="NHWC",
                    out_layout="NHWC",
                )
                gv: R.Tensor((1, 1, 1, 1), dtype="int8") = R.astype(lv1, dtype="int8")
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_quantized_conv2d_per_tensor_uses_qdq():
    """Quantized Conv2D with per-tensor quantization uses DQ -> conv2d -> Q."""
    builder = flatbuffers.Builder(2048)

    in_q = _build_quantization_parameters(
        builder, scale=[0.5], zero_point=[3], quantized_dimension=0
    )
    wt_q = _build_quantization_parameters(
        builder, scale=[0.25], zero_point=[0], quantized_dimension=0
    )
    out_q = _build_quantization_parameters(
        builder, scale=[1.0], zero_point=[0], quantized_dimension=0
    )

    input_tensor = _build_tensor(
        builder,
        0,
        [1, 4, 4, 1],
        tensor_type=_tfl_tensor_type.INT8,
        quantization=in_q,
    )
    weight_tensor = _build_tensor(
        builder,
        1,
        [2, 3, 3, 1],
        tensor_type=_tfl_tensor_type.INT8,
        quantization=wt_q,
    )
    output_tensor = _build_tensor(
        builder,
        2,
        [1, 2, 2, 2],
        tensor_type=_tfl_tensor_type.INT8,
        quantization=out_q,
    )

    _tfl_conv2d_options.Conv2DOptionsStart(builder)
    _tfl_conv2d_options.Conv2DOptionsAddStrideH(builder, 1)
    _tfl_conv2d_options.Conv2DOptionsAddStrideW(builder, 1)
    _tfl_conv2d_options.Conv2DOptionsAddPadding(builder, _tfl_padding.VALID)
    _tfl_conv2d_options.Conv2DOptionsAddFusedActivationFunction(builder, 0)
    conv_opts = _tfl_conv2d_options.Conv2DOptionsEnd(builder)

    conv_op = _build_operator(
        builder,
        0,
        [0, 1],
        [2],
        builtin_options_type=_tfl_builtin_options.Conv2DOptions,
        builtin_options=conv_opts,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=[input_tensor, weight_tensor, output_tensor],
        operators=[conv_op],
        inputs=[0, 1],
        outputs=[2],
    )
    operator_codes = [_build_operator_code(builder, _tfl_builtin_operator.CONV_2D)]
    buf = _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=operator_codes,
        buffers=[_build_buffer(builder), _build_buffer(builder), _build_buffer(builder)],
    )

    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)
    mod = from_tflite(tflite_model)
    mod["main"] = mod["main"].without_attr("params")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            tvmgen_tensor_0: R.Tensor((1, 4, 4, 1), dtype="int8"),
            tvmgen_tensor_1: R.Tensor((2, 3, 3, 1), dtype="int8"),
        ) -> R.Tensor((1, 2, 2, 2), dtype="int8"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((1, 4, 4, 1), dtype="float32") = R.dequantize(
                    tvmgen_tensor_0,
                    R.const(0.5, "float32"),
                    R.const(3, "int32"),
                    out_dtype="float32",
                    axis=0,
                )
                lv1: R.Tensor((3, 3, 1, 2), dtype="int8") = R.permute_dims(
                    tvmgen_tensor_1,
                    axes=[1, 2, 3, 0],
                )
                lv2: R.Tensor((3, 3, 1, 2), dtype="float32") = R.dequantize(
                    lv1,
                    R.const(0.25, "float32"),
                    R.const(0, "int32"),
                    out_dtype="float32",
                    axis=3,
                )
                lv3: R.Tensor((1, 2, 2, 2), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv2,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="HWIO",
                    out_layout="NHWC",
                )
                gv: R.Tensor((1, 2, 2, 2), dtype="int8") = R.quantize(
                    lv3,
                    R.const(1.0, "float32"),
                    R.const(0, "int32"),
                    out_dtype="int8",
                    axis=0,
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_quantized_conv2d_per_channel_weight_uses_remapped_axis():
    """Quantized Conv2D remaps per-channel weight axis after OHWI -> HWIO."""
    builder = flatbuffers.Builder(2048)

    in_q = _build_quantization_parameters(
        builder, scale=[0.5], zero_point=[3], quantized_dimension=0
    )
    wt_q = _build_quantization_parameters(
        builder, scale=[0.25, 0.75], zero_point=[0, 0], quantized_dimension=0
    )
    out_q = _build_quantization_parameters(
        builder, scale=[1.0], zero_point=[0], quantized_dimension=0
    )

    input_tensor = _build_tensor(
        builder,
        0,
        [1, 4, 4, 1],
        tensor_type=_tfl_tensor_type.INT8,
        quantization=in_q,
    )
    weight_tensor = _build_tensor(
        builder,
        1,
        [2, 3, 3, 1],
        tensor_type=_tfl_tensor_type.INT8,
        quantization=wt_q,
    )
    output_tensor = _build_tensor(
        builder,
        2,
        [1, 2, 2, 2],
        tensor_type=_tfl_tensor_type.INT8,
        quantization=out_q,
    )

    _tfl_conv2d_options.Conv2DOptionsStart(builder)
    _tfl_conv2d_options.Conv2DOptionsAddStrideH(builder, 1)
    _tfl_conv2d_options.Conv2DOptionsAddStrideW(builder, 1)
    _tfl_conv2d_options.Conv2DOptionsAddPadding(builder, _tfl_padding.VALID)
    _tfl_conv2d_options.Conv2DOptionsAddFusedActivationFunction(builder, 0)
    conv_opts = _tfl_conv2d_options.Conv2DOptionsEnd(builder)

    conv_op = _build_operator(
        builder,
        0,
        [0, 1],
        [2],
        builtin_options_type=_tfl_builtin_options.Conv2DOptions,
        builtin_options=conv_opts,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=[input_tensor, weight_tensor, output_tensor],
        operators=[conv_op],
        inputs=[0, 1],
        outputs=[2],
    )
    operator_codes = [_build_operator_code(builder, _tfl_builtin_operator.CONV_2D)]
    buf = _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=operator_codes,
        buffers=[_build_buffer(builder), _build_buffer(builder), _build_buffer(builder)],
    )

    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)
    mod = from_tflite(tflite_model)
    mod["main"] = mod["main"].without_attr("params")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            tvmgen_tensor_0: R.Tensor((1, 4, 4, 1), dtype="int8"),
            tvmgen_tensor_1: R.Tensor((2, 3, 3, 1), dtype="int8"),
        ) -> R.Tensor((1, 2, 2, 2), dtype="int8"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((1, 4, 4, 1), dtype="float32") = R.dequantize(
                    tvmgen_tensor_0,
                    R.const(0.5, "float32"),
                    R.const(3, "int32"),
                    out_dtype="float32",
                    axis=0,
                )
                lv1: R.Tensor((3, 3, 1, 2), dtype="int8") = R.permute_dims(
                    tvmgen_tensor_1,
                    axes=[1, 2, 3, 0],
                )
                lv2: R.Tensor((3, 3, 1, 2), dtype="float32") = R.dequantize(
                    lv1,
                    R.const([0.25, 0.75], "float32"),
                    R.const(0, "int32"),
                    out_dtype="float32",
                    axis=3,
                )
                lv3: R.Tensor((1, 2, 2, 2), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv2,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="HWIO",
                    out_layout="NHWC",
                )
                gv: R.Tensor((1, 2, 2, 2), dtype="int8") = R.quantize(
                    lv3,
                    R.const(1.0, "float32"),
                    R.const(0, "int32"),
                    out_dtype="int8",
                    axis=0,
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_quantized_concat_uses_qdq():
    """Quantized CONCATENATION uses DQ each input → concat → Q."""
    import flatbuffers
    import tflite.Model

    builder = flatbuffers.Builder(1024)

    in_q = _build_quantization_parameters(
        builder, scale=[0.5], zero_point=[3], quantized_dimension=0
    )
    out_q = _build_quantization_parameters(
        builder, scale=[0.5], zero_point=[3], quantized_dimension=0
    )

    t0 = _build_tensor(builder, 0, [1, 2], tensor_type=_tfl_tensor_type.INT8, quantization=in_q)
    t1 = _build_tensor(builder, 1, [1, 2], tensor_type=_tfl_tensor_type.INT8, quantization=in_q)
    t2 = _build_tensor(builder, 2, [1, 4], tensor_type=_tfl_tensor_type.INT8, quantization=out_q)

    _tfl_concatenation_options.ConcatenationOptionsStart(builder)
    _tfl_concatenation_options.ConcatenationOptionsAddAxis(builder, 1)
    _tfl_concatenation_options.ConcatenationOptionsAddFusedActivationFunction(builder, 0)
    concat_opts = _tfl_concatenation_options.ConcatenationOptionsEnd(builder)

    concat_op = _build_operator(
        builder,
        0,
        [0, 1],
        [2],
        builtin_options_type=_tfl_builtin_options.ConcatenationOptions,
        builtin_options=concat_opts,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=[t0, t1, t2],
        operators=[concat_op],
        inputs=[0, 1],
        outputs=[2],
    )
    operator_codes = [_build_operator_code(builder, _tfl_builtin_operator.CONCATENATION)]
    buf = _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=operator_codes,
        buffers=[_build_buffer(builder)] * 3,
    )

    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)
    mod = from_tflite(tflite_model)
    mod["main"] = mod["main"].without_attr("params")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            tvmgen_tensor_0: R.Tensor((1, 2), dtype="int8"),
            tvmgen_tensor_1: R.Tensor((1, 2), dtype="int8"),
        ) -> R.Tensor((1, 4), dtype="int8"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((1, 2), dtype="float32") = R.dequantize(
                    tvmgen_tensor_0,
                    R.const(0.5, "float32"),
                    R.const(3, "int32"),
                    out_dtype="float32",
                    axis=0,
                )
                lv1: R.Tensor((1, 2), dtype="float32") = R.dequantize(
                    tvmgen_tensor_1,
                    R.const(0.5, "float32"),
                    R.const(3, "int32"),
                    out_dtype="float32",
                    axis=0,
                )
                lv2: R.Tensor((1, 4), dtype="float32") = R.concat((lv, lv1), axis=1)
                gv: R.Tensor((1, 4), dtype="int8") = R.quantize(
                    lv2,
                    R.const(0.5, "float32"),
                    R.const(3, "int32"),
                    out_dtype="int8",
                    axis=0,
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_quantized_concat_fused_relu_uses_quantized_clip():
    """Quantized CONCATENATION fused RELU clips in the quantized domain."""
    builder = flatbuffers.Builder(1024)

    in_q = _build_quantization_parameters(
        builder, scale=[0.5], zero_point=[3], quantized_dimension=0
    )
    out_q = _build_quantization_parameters(
        builder, scale=[0.5], zero_point=[3], quantized_dimension=0
    )

    t0 = _build_tensor(builder, 0, [1, 2], tensor_type=_tfl_tensor_type.INT8, quantization=in_q)
    t1 = _build_tensor(builder, 1, [1, 2], tensor_type=_tfl_tensor_type.INT8, quantization=in_q)
    t2 = _build_tensor(builder, 2, [1, 4], tensor_type=_tfl_tensor_type.INT8, quantization=out_q)

    _tfl_concatenation_options.ConcatenationOptionsStart(builder)
    _tfl_concatenation_options.ConcatenationOptionsAddAxis(builder, 1)
    _tfl_concatenation_options.ConcatenationOptionsAddFusedActivationFunction(
        builder, _tfl_activation_fn.RELU
    )
    concat_opts = _tfl_concatenation_options.ConcatenationOptionsEnd(builder)

    concat_op = _build_operator(
        builder,
        0,
        [0, 1],
        [2],
        builtin_options_type=_tfl_builtin_options.ConcatenationOptions,
        builtin_options=concat_opts,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=[t0, t1, t2],
        operators=[concat_op],
        inputs=[0, 1],
        outputs=[2],
    )
    operator_codes = [_build_operator_code(builder, _tfl_builtin_operator.CONCATENATION)]
    buf = _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=operator_codes,
        buffers=[_build_buffer(builder)] * 3,
    )

    mod = _load_model_from_buffer(buf)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            tvmgen_tensor_0: R.Tensor((1, 2), dtype="int8"),
            tvmgen_tensor_1: R.Tensor((1, 2), dtype="int8"),
        ) -> R.Tensor((1, 4), dtype="int8"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((1, 2), dtype="float32") = R.dequantize(
                    tvmgen_tensor_0,
                    R.const(0.5, "float32"),
                    R.const(3, "int32"),
                    out_dtype="float32",
                    axis=0,
                )
                lv1: R.Tensor((1, 2), dtype="float32") = R.dequantize(
                    tvmgen_tensor_1,
                    R.const(0.5, "float32"),
                    R.const(3, "int32"),
                    out_dtype="float32",
                    axis=0,
                )
                lv2: R.Tensor((1, 4), dtype="float32") = R.concat((lv, lv1), axis=1)
                lv3: R.Tensor((1, 4), dtype="int8") = R.quantize(
                    lv2,
                    R.const(0.5, "float32"),
                    R.const(3, "int32"),
                    out_dtype="int8",
                    axis=0,
                )
                gv: R.Tensor((1, 4), dtype="int8") = R.clip(lv3, min=3.0, max=127.0)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_quantized_add_uses_qdq():
    """Quantized ADD uses DQ each input -> add -> Q."""
    builder = flatbuffers.Builder(1024)

    lhs_q = _build_quantization_parameters(
        builder, scale=[0.5], zero_point=[3], quantized_dimension=0
    )
    rhs_q = _build_quantization_parameters(
        builder, scale=[0.25], zero_point=[1], quantized_dimension=0
    )
    out_q = _build_quantization_parameters(
        builder, scale=[1.0], zero_point=[0], quantized_dimension=0
    )

    t_lhs = _build_tensor(builder, 0, [2], tensor_type=_tfl_tensor_type.INT8, quantization=lhs_q)
    t_rhs = _build_tensor(builder, 1, [2], tensor_type=_tfl_tensor_type.INT8, quantization=rhs_q)
    t_out = _build_tensor(builder, 2, [2], tensor_type=_tfl_tensor_type.INT8, quantization=out_q)

    _tfl_add_options.AddOptionsStart(builder)
    _tfl_add_options.AddOptionsAddFusedActivationFunction(builder, 0)
    add_opts = _tfl_add_options.AddOptionsEnd(builder)

    add_op = _build_operator(
        builder,
        0,
        [0, 1],
        [2],
        builtin_options_type=_tfl_builtin_options.AddOptions,
        builtin_options=add_opts,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=[t_lhs, t_rhs, t_out],
        operators=[add_op],
        inputs=[0, 1],
        outputs=[2],
    )
    operator_codes = [_build_operator_code(builder, _tfl_builtin_operator.ADD)]
    buf = _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=operator_codes,
        buffers=[_build_buffer(builder)] * 3,
    )

    mod = _load_model_from_buffer(buf)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            tvmgen_tensor_0: R.Tensor((2,), dtype="int8"),
            tvmgen_tensor_1: R.Tensor((2,), dtype="int8"),
        ) -> R.Tensor((2,), dtype="int8"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((2,), dtype="float32") = R.dequantize(
                    tvmgen_tensor_0,
                    R.const(0.5, "float32"),
                    R.const(3, "int32"),
                    out_dtype="float32",
                    axis=0,
                )
                lv1: R.Tensor((2,), dtype="float32") = R.dequantize(
                    tvmgen_tensor_1,
                    R.const(0.25, "float32"),
                    R.const(1, "int32"),
                    out_dtype="float32",
                    axis=0,
                )
                lv2: R.Tensor((2,), dtype="float32") = R.add(lv, lv1)
                gv: R.Tensor((2,), dtype="int8") = R.quantize(
                    lv2,
                    R.const(1.0, "float32"),
                    R.const(0, "int32"),
                    out_dtype="int8",
                    axis=0,
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_quantized_add_fused_relu6_uses_float_clip_before_quantize():
    """Quantized ADD fused RELU6 applies the activation before quantizing."""
    builder = flatbuffers.Builder(1024)

    lhs_q = _build_quantization_parameters(
        builder, scale=[0.5], zero_point=[3], quantized_dimension=0
    )
    rhs_q = _build_quantization_parameters(
        builder, scale=[0.25], zero_point=[1], quantized_dimension=0
    )
    out_q = _build_quantization_parameters(
        builder, scale=[1.0], zero_point=[0], quantized_dimension=0
    )

    t_lhs = _build_tensor(builder, 0, [2], tensor_type=_tfl_tensor_type.INT8, quantization=lhs_q)
    t_rhs = _build_tensor(builder, 1, [2], tensor_type=_tfl_tensor_type.INT8, quantization=rhs_q)
    t_out = _build_tensor(builder, 2, [2], tensor_type=_tfl_tensor_type.INT8, quantization=out_q)

    _tfl_add_options.AddOptionsStart(builder)
    _tfl_add_options.AddOptionsAddFusedActivationFunction(builder, _tfl_activation_fn.RELU6)
    add_opts = _tfl_add_options.AddOptionsEnd(builder)

    add_op = _build_operator(
        builder,
        0,
        [0, 1],
        [2],
        builtin_options_type=_tfl_builtin_options.AddOptions,
        builtin_options=add_opts,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=[t_lhs, t_rhs, t_out],
        operators=[add_op],
        inputs=[0, 1],
        outputs=[2],
    )
    operator_codes = [_build_operator_code(builder, _tfl_builtin_operator.ADD)]
    buf = _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=operator_codes,
        buffers=[_build_buffer(builder)] * 3,
    )

    mod = _load_model_from_buffer(buf)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            tvmgen_tensor_0: R.Tensor((2,), dtype="int8"),
            tvmgen_tensor_1: R.Tensor((2,), dtype="int8"),
        ) -> R.Tensor((2,), dtype="int8"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((2,), dtype="float32") = R.dequantize(
                    tvmgen_tensor_0,
                    R.const(0.5, "float32"),
                    R.const(3, "int32"),
                    out_dtype="float32",
                    axis=0,
                )
                lv1: R.Tensor((2,), dtype="float32") = R.dequantize(
                    tvmgen_tensor_1,
                    R.const(0.25, "float32"),
                    R.const(1, "int32"),
                    out_dtype="float32",
                    axis=0,
                )
                lv2: R.Tensor((2,), dtype="float32") = R.add(lv, lv1)
                lv3: R.Tensor((2,), dtype="float32") = R.clip(lv2, min=0, max=6)
                gv: R.Tensor((2,), dtype="int8") = R.quantize(
                    lv3,
                    R.const(1.0, "float32"),
                    R.const(0, "int32"),
                    out_dtype="int8",
                    axis=0,
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_quantized_add_without_output_qparams_invalid():
    """Quantized ADD with missing output qparams raises AttributeError."""
    builder = flatbuffers.Builder(1024)

    in_q = _build_quantization_parameters(
        builder, scale=[0.5], zero_point=[3], quantized_dimension=0
    )

    t_lhs = _build_tensor(builder, 0, [2], tensor_type=_tfl_tensor_type.INT8, quantization=in_q)
    t_rhs = _build_tensor(builder, 1, [2], tensor_type=_tfl_tensor_type.INT8, quantization=in_q)
    t_out = _build_tensor(builder, 2, [2], tensor_type=_tfl_tensor_type.INT8)

    _tfl_add_options.AddOptionsStart(builder)
    _tfl_add_options.AddOptionsAddFusedActivationFunction(builder, _tfl_activation_fn.NONE)
    add_opts = _tfl_add_options.AddOptionsEnd(builder)

    add_op = _build_operator(
        builder,
        0,
        [0, 1],
        [2],
        builtin_options_type=_tfl_builtin_options.AddOptions,
        builtin_options=add_opts,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=[t_lhs, t_rhs, t_out],
        operators=[add_op],
        inputs=[0, 1],
        outputs=[2],
    )
    operator_codes = [_build_operator_code(builder, _tfl_builtin_operator.ADD)]
    buf = _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=operator_codes,
        buffers=[_build_buffer(builder)] * 3,
    )

    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)

    with pytest.raises(AttributeError, match="output must have quantization"):
        from_tflite(tflite_model)


def test_quantized_square_unsupported():
    """Quantized SQUARE is rejected instead of applying integer power directly."""
    builder = flatbuffers.Builder(1024)

    in_q = _build_quantization_parameters(
        builder, scale=[0.5], zero_point=[3], quantized_dimension=0
    )
    out_q = _build_quantization_parameters(
        builder, scale=[1.0], zero_point=[0], quantized_dimension=0
    )

    t_in = _build_tensor(builder, 0, [2], tensor_type=_tfl_tensor_type.INT8, quantization=in_q)
    t_out = _build_tensor(builder, 1, [2], tensor_type=_tfl_tensor_type.INT8, quantization=out_q)

    square_op = _build_operator(builder, 0, [0], [1])
    subgraph = _build_subgraph(
        builder,
        tensors=[t_in, t_out],
        operators=[square_op],
        inputs=[0],
        outputs=[1],
    )
    operator_codes = [_build_operator_code(builder, _tfl_builtin_operator.SQUARE)]
    buf = _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=operator_codes,
        buffers=[_build_buffer(builder)] * 2,
    )

    with pytest.raises(NotImplementedError, match="SQUARE"):
        _load_model_from_buffer(buf)


def test_quantized_conv2d_with_int32_bias_dequantizes_bias():
    """Conv2D with INT32 bias dequantizes bias with in_scale x wt_scale."""
    import flatbuffers
    import tflite.Model

    builder = flatbuffers.Builder(2048)

    in_q = _build_quantization_parameters(
        builder, scale=[0.5], zero_point=[3], quantized_dimension=0
    )
    wt_q = _build_quantization_parameters(
        builder, scale=[0.25], zero_point=[0], quantized_dimension=0
    )
    out_q = _build_quantization_parameters(
        builder, scale=[1.0], zero_point=[0], quantized_dimension=0
    )

    t_in = _build_tensor(
        builder, 0, [1, 4, 4, 1], tensor_type=_tfl_tensor_type.INT8, quantization=in_q
    )
    t_wt = _build_tensor(
        builder, 1, [2, 3, 3, 1], tensor_type=_tfl_tensor_type.INT8, quantization=wt_q
    )
    t_bi = _build_tensor(builder, 2, [2], tensor_type=_tfl_tensor_type.INT32)
    t_ou = _build_tensor(
        builder, 3, [1, 2, 2, 2], tensor_type=_tfl_tensor_type.INT8, quantization=out_q
    )

    _tfl_conv2d_options.Conv2DOptionsStart(builder)
    _tfl_conv2d_options.Conv2DOptionsAddStrideH(builder, 1)
    _tfl_conv2d_options.Conv2DOptionsAddStrideW(builder, 1)
    _tfl_conv2d_options.Conv2DOptionsAddPadding(builder, 1)
    _tfl_conv2d_options.Conv2DOptionsAddFusedActivationFunction(builder, 0)
    conv_opts = _tfl_conv2d_options.Conv2DOptionsEnd(builder)

    conv_op = _build_operator(
        builder,
        0,
        [0, 1, 2],
        [3],
        builtin_options_type=_tfl_builtin_options.Conv2DOptions,
        builtin_options=conv_opts,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=[t_in, t_wt, t_bi, t_ou],
        operators=[conv_op],
        inputs=[0, 1, 2],
        outputs=[3],
    )
    operator_codes = [_build_operator_code(builder, _tfl_builtin_operator.CONV_2D)]
    buf = _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=operator_codes,
        buffers=[_build_buffer(builder)] * 4,
    )

    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)
    mod = from_tflite(tflite_model)
    mod["main"] = mod["main"].without_attr("params")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            tvmgen_tensor_0: R.Tensor((1, 4, 4, 1), dtype="int8"),
            tvmgen_tensor_1: R.Tensor((2, 3, 3, 1), dtype="int8"),
            tvmgen_tensor_2: R.Tensor((2,), dtype="int32"),
        ) -> R.Tensor((1, 2, 2, 2), dtype="int8"):
            R.func_attr({"num_input": 3})
            with R.dataflow():
                lv: R.Tensor((1, 4, 4, 1), dtype="float32") = R.dequantize(
                    tvmgen_tensor_0,
                    R.const(0.5, "float32"),
                    R.const(3, "int32"),
                    out_dtype="float32",
                    axis=0,
                )
                lv1: R.Tensor((3, 3, 1, 2), dtype="int8") = R.permute_dims(
                    tvmgen_tensor_1,
                    axes=[1, 2, 3, 0],
                )
                lv2: R.Tensor((3, 3, 1, 2), dtype="float32") = R.dequantize(
                    lv1,
                    R.const(0.25, "float32"),
                    R.const(0, "int32"),
                    out_dtype="float32",
                    axis=3,
                )
                lv3: R.Tensor((1, 2, 2, 2), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv2,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="HWIO",
                    out_layout="NHWC",
                )
                lv4: R.Tensor((), dtype="float32") = R.multiply(
                    R.const(0.5, "float32"),
                    R.const(0.25, "float32"),
                )
                lv5: R.Tensor((2,), dtype="float32") = R.dequantize(
                    tvmgen_tensor_2,
                    lv4,
                    R.const(0, "int32"),
                    out_dtype="float32",
                    axis=0,
                )
                lv6: R.Tensor((1, 2, 2, 2), dtype="float32") = R.add(lv3, lv5)
                gv: R.Tensor((1, 2, 2, 2), dtype="int8") = R.quantize(
                    lv6,
                    R.const(1.0, "float32"),
                    R.const(0, "int32"),
                    out_dtype="int8",
                    axis=0,
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_quantized_conv2d_per_channel_weight_with_int32_bias_dequantizes_bias():
    """Conv2D with per-channel weight quantization uses vector bias scale."""
    builder = flatbuffers.Builder(2048)

    in_q = _build_quantization_parameters(
        builder, scale=[0.5], zero_point=[3], quantized_dimension=0
    )
    wt_q = _build_quantization_parameters(
        builder, scale=[0.25, 0.75], zero_point=[0, 0], quantized_dimension=0
    )
    out_q = _build_quantization_parameters(
        builder, scale=[1.0], zero_point=[0], quantized_dimension=0
    )

    t_in = _build_tensor(
        builder, 0, [1, 4, 4, 1], tensor_type=_tfl_tensor_type.INT8, quantization=in_q
    )
    t_wt = _build_tensor(
        builder, 1, [2, 3, 3, 1], tensor_type=_tfl_tensor_type.INT8, quantization=wt_q
    )
    t_bi = _build_tensor(builder, 2, [2], tensor_type=_tfl_tensor_type.INT32)
    t_ou = _build_tensor(
        builder, 3, [1, 2, 2, 2], tensor_type=_tfl_tensor_type.INT8, quantization=out_q
    )

    _tfl_conv2d_options.Conv2DOptionsStart(builder)
    _tfl_conv2d_options.Conv2DOptionsAddStrideH(builder, 1)
    _tfl_conv2d_options.Conv2DOptionsAddStrideW(builder, 1)
    _tfl_conv2d_options.Conv2DOptionsAddPadding(builder, 1)
    _tfl_conv2d_options.Conv2DOptionsAddFusedActivationFunction(builder, 0)
    conv_opts = _tfl_conv2d_options.Conv2DOptionsEnd(builder)

    conv_op = _build_operator(
        builder,
        0,
        [0, 1, 2],
        [3],
        builtin_options_type=_tfl_builtin_options.Conv2DOptions,
        builtin_options=conv_opts,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=[t_in, t_wt, t_bi, t_ou],
        operators=[conv_op],
        inputs=[0, 1, 2],
        outputs=[3],
    )
    operator_codes = [_build_operator_code(builder, _tfl_builtin_operator.CONV_2D)]
    buf = _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=operator_codes,
        buffers=[_build_buffer(builder)] * 4,
    )

    mod = _load_model_from_buffer(buf)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            tvmgen_tensor_0: R.Tensor((1, 4, 4, 1), dtype="int8"),
            tvmgen_tensor_1: R.Tensor((2, 3, 3, 1), dtype="int8"),
            tvmgen_tensor_2: R.Tensor((2,), dtype="int32"),
        ) -> R.Tensor((1, 2, 2, 2), dtype="int8"):
            R.func_attr({"num_input": 3})
            with R.dataflow():
                lv: R.Tensor((1, 4, 4, 1), dtype="float32") = R.dequantize(
                    tvmgen_tensor_0,
                    R.const(0.5, "float32"),
                    R.const(3, "int32"),
                    out_dtype="float32",
                    axis=0,
                )
                lv1: R.Tensor((3, 3, 1, 2), dtype="int8") = R.permute_dims(
                    tvmgen_tensor_1,
                    axes=[1, 2, 3, 0],
                )
                lv2: R.Tensor((3, 3, 1, 2), dtype="float32") = R.dequantize(
                    lv1,
                    R.const([0.25, 0.75], "float32"),
                    R.const(0, "int32"),
                    out_dtype="float32",
                    axis=3,
                )
                lv3: R.Tensor((1, 2, 2, 2), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv2,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="HWIO",
                    out_layout="NHWC",
                )
                lv4: R.Tensor((2,), dtype="float32") = R.multiply(
                    R.const(0.5, "float32"),
                    R.const([0.25, 0.75], "float32"),
                )
                lv5: R.Tensor((2,), dtype="float32") = R.dequantize(
                    tvmgen_tensor_2,
                    lv4,
                    R.const(0, "int32"),
                    out_dtype="float32",
                    axis=0,
                )
                lv6: R.Tensor((1, 2, 2, 2), dtype="float32") = R.add(lv3, lv5)
                gv: R.Tensor((1, 2, 2, 2), dtype="int8") = R.quantize(
                    lv6,
                    R.const(1.0, "float32"),
                    R.const(0, "int32"),
                    out_dtype="int8",
                    axis=0,
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_per_channel_depthwise_conv_dequantizes_before_reshape():
    """Per-channel depthwise weights keep C*M intact and lower to HWOI."""
    import flatbuffers
    import tflite.Model

    builder = flatbuffers.Builder(1024)

    in_q = _build_quantization_parameters(
        builder, scale=[0.5], zero_point=[0], quantized_dimension=0
    )
    # Two input channels with depth_multiplier=2 produce four output channels.
    wt_q = _build_quantization_parameters(
        builder,
        scale=[0.25, 0.5, 0.75, 1.0],
        zero_point=[0, 0, 0, 0],
        quantized_dimension=3,
    )
    out_q = _build_quantization_parameters(
        builder, scale=[1.0], zero_point=[0], quantized_dimension=0
    )

    t_in = _build_tensor(
        builder, 0, [1, 4, 4, 2], tensor_type=_tfl_tensor_type.INT8, quantization=in_q
    )
    t_wt = _build_tensor(
        builder, 1, [1, 3, 3, 4], tensor_type=_tfl_tensor_type.INT8, quantization=wt_q
    )
    t_ou = _build_tensor(
        builder, 2, [1, 2, 2, 4], tensor_type=_tfl_tensor_type.INT8, quantization=out_q
    )

    _tfl_depthwise_conv2d_options.DepthwiseConv2DOptionsStart(builder)
    _tfl_depthwise_conv2d_options.DepthwiseConv2DOptionsAddStrideH(builder, 1)
    _tfl_depthwise_conv2d_options.DepthwiseConv2DOptionsAddStrideW(builder, 1)
    _tfl_depthwise_conv2d_options.DepthwiseConv2DOptionsAddDepthMultiplier(builder, 2)
    _tfl_depthwise_conv2d_options.DepthwiseConv2DOptionsAddPadding(builder, 1)
    _tfl_depthwise_conv2d_options.DepthwiseConv2DOptionsAddFusedActivationFunction(builder, 0)
    dw_opts = _tfl_depthwise_conv2d_options.DepthwiseConv2DOptionsEnd(builder)

    dw_op = _build_operator(
        builder,
        0,
        [0, 1],
        [2],
        builtin_options_type=_tfl_builtin_options.DepthwiseConv2DOptions,
        builtin_options=dw_opts,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=[t_in, t_wt, t_ou],
        operators=[dw_op],
        inputs=[0, 1],
        outputs=[2],
    )
    operator_codes = [_build_operator_code(builder, _tfl_builtin_operator.DEPTHWISE_CONV_2D)]
    buf = _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=operator_codes,
        buffers=[_build_buffer(builder)] * 3,
    )

    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)

    mod = from_tflite(tflite_model)
    dequantize_calls = []
    reshape_calls = []

    def _visit(expr):
        if isinstance(expr, relax.Call) and expr.op == tvm_tirx.ir.Op.get("relax.dequantize"):
            dequantize_calls.append(expr)
        if isinstance(expr, relax.Call) and expr.op == tvm_tirx.ir.Op.get("relax.reshape"):
            reshape_calls.append(expr)

    relax.analysis.post_order_visit(mod["main"].body, _visit)
    assert any(call.attrs.axis == 3 for call in dequantize_calls)
    depthwise_reshapes = [
        call
        for call in reshape_calls
        if call.ty.dtype == "float32"
        and tuple(dim.value for dim in call.ty.shape.values) == (3, 3, 4, 1)
    ]
    assert len(depthwise_reshapes) == 1


def test_uint8_reshape_requantize_uses_dq_reshape_q():
    """uint8 RESHAPE with different qparams uses DQ→reshape→Q."""
    import flatbuffers
    import numpy as np
    import tflite.Model

    builder = flatbuffers.Builder(1024)

    in_q = _build_quantization_parameters(
        builder, scale=[0.5], zero_point=[128], quantized_dimension=0
    )
    out_q = _build_quantization_parameters(
        builder, scale=[1.0], zero_point=[100], quantized_dimension=0
    )

    t_in = _build_tensor(builder, 0, [1, 4], tensor_type=_tfl_tensor_type.UINT8, quantization=in_q)
    t_ou = _build_tensor(builder, 1, [2, 2], tensor_type=_tfl_tensor_type.UINT8, quantization=out_q)

    # Use ReshapeOptions with static new_shape [2, 2]
    new_shape_np = np.array([2, 2], dtype=np.int32)
    new_shape_vec = _tflite_int32_vector(
        builder, _tfl_reshape_options.ReshapeOptionsStartNewShapeVector, new_shape_np
    )
    _tfl_reshape_options.ReshapeOptionsStart(builder)
    _tfl_reshape_options.ReshapeOptionsAddNewShape(builder, new_shape_vec)
    reshape_opts = _tfl_reshape_options.ReshapeOptionsEnd(builder)

    reshape_op = _build_operator(
        builder,
        0,
        [0],
        [1],
        builtin_options_type=_tfl_builtin_options.ReshapeOptions,
        builtin_options=reshape_opts,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=[t_in, t_ou],
        operators=[reshape_op],
        inputs=[0],
        outputs=[1],
    )
    operator_codes = [_build_operator_code(builder, _tfl_builtin_operator.RESHAPE)]
    buf = _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=operator_codes,
        buffers=[_build_buffer(builder), _build_buffer(builder)],
    )

    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)
    mod = from_tflite(tflite_model)
    mod["main"] = mod["main"].without_attr("params")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            tvmgen_tensor_0: R.Tensor((1, 4), dtype="uint8"),
        ) -> R.Tensor((2, 2), dtype="uint8"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, 4), dtype="float32") = R.dequantize(
                    tvmgen_tensor_0,
                    R.const(0.5, "float32"),
                    R.const(128, "int32"),
                    out_dtype="float32",
                    axis=0,
                )
                lv1: R.Tensor((2, 2), dtype="float32") = R.reshape(
                    lv,
                    R.shape([2, 2]),
                )
                gv: R.Tensor((2, 2), dtype="uint8") = R.quantize(
                    lv1,
                    R.const(1.0, "float32"),
                    R.const(100, "int32"),
                    out_dtype="uint8",
                    axis=0,
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_transpose_conv_with_int32_bias_dequantizes_bias():
    """TRANSPOSE_CONV with INT32 bias dequantizes bias before adding."""
    import struct

    import flatbuffers
    import tflite.Model

    builder = flatbuffers.Builder(2048)

    in_q = _build_quantization_parameters(
        builder, scale=[0.5], zero_point=[3], quantized_dimension=0
    )
    wt_q = _build_quantization_parameters(
        builder, scale=[0.25], zero_point=[0], quantized_dimension=0
    )
    out_q = _build_quantization_parameters(
        builder, scale=[1.0], zero_point=[0], quantized_dimension=0
    )

    t_in = _build_tensor(
        builder, 0, [1, 1, 1, 1], tensor_type=_tfl_tensor_type.INT8, quantization=in_q
    )
    t_wt = _build_tensor(
        builder, 1, [1, 1, 1, 1], tensor_type=_tfl_tensor_type.INT8, quantization=wt_q
    )
    t_bi = _build_tensor(builder, 2, [1], tensor_type=_tfl_tensor_type.INT32)
    t_ou = _build_tensor(
        builder, 3, [1, 1, 1, 1], tensor_type=_tfl_tensor_type.INT8, quantization=out_q
    )
    oshape_data = struct.pack("<iiii", 1, 1, 1, 1)
    t_oshape = _build_tensor(builder, 4, [4], tensor_type=_tfl_tensor_type.INT32)

    _tfl_transpose_conv_options.TransposeConvOptionsStart(builder)
    _tfl_transpose_conv_options.TransposeConvOptionsAddStrideH(builder, 1)
    _tfl_transpose_conv_options.TransposeConvOptionsAddStrideW(builder, 1)
    _tfl_transpose_conv_options.TransposeConvOptionsAddPadding(builder, 1)  # VALID
    _tfl_transpose_conv_options.TransposeConvOptionsAddFusedActivationFunction(builder, 0)
    tc_opts = _tfl_transpose_conv_options.TransposeConvOptionsEnd(builder)

    tc_op = _build_operator(
        builder,
        0,
        [4, 1, 0, 2],
        [3],
        builtin_options_type=_tfl_builtin_options.TransposeConvOptions,
        builtin_options=tc_opts,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=[t_in, t_wt, t_bi, t_ou, t_oshape],
        operators=[tc_op],
        inputs=[0, 1, 2],
        outputs=[3],
    )
    operator_codes = [_build_operator_code(builder, _tfl_builtin_operator.TRANSPOSE_CONV)]
    buf = _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=operator_codes,
        buffers=[
            _build_buffer(builder),
            _build_buffer(builder),
            _build_buffer(builder),
            _build_buffer(builder),
            _build_buffer(builder, oshape_data),
        ],
    )

    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)
    mod = from_tflite(tflite_model)
    mod["main"] = mod["main"].without_attr("params")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            tvmgen_tensor_0: R.Tensor((1, 1, 1, 1), dtype="int8"),
            tvmgen_tensor_1: R.Tensor((1, 1, 1, 1), dtype="int8"),
            tvmgen_tensor_2: R.Tensor((1,), dtype="int32"),
        ) -> R.Tensor((1, 1, 1, 1), dtype="int8"):
            R.func_attr({"num_input": 3})
            with R.dataflow():
                lv: R.Tensor((1, 1, 1, 1), dtype="float32") = R.dequantize(
                    tvmgen_tensor_0,
                    R.const(0.5, "float32"),
                    R.const(3, "int32"),
                    out_dtype="float32",
                    axis=0,
                )
                lv1: R.Tensor((1, 1, 1, 1), dtype="int8") = R.permute_dims(
                    tvmgen_tensor_1,
                    axes=[3, 0, 1, 2],
                )
                lv2: R.Tensor((1, 1, 1, 1), dtype="float32") = R.dequantize(
                    lv1,
                    R.const(0.25, "float32"),
                    R.const(0, "int32"),
                    out_dtype="float32",
                    axis=1,
                )
                lv3: R.Tensor((1, 1, 1, 1), dtype="float32") = R.nn.conv2d_transpose(
                    lv,
                    lv2,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    data_layout="NHWC",
                    kernel_layout="IOHW",
                    out_dtype="float32",
                )
                lv4: R.Tensor((), dtype="float32") = R.multiply(
                    R.const(0.5, "float32"),
                    R.const(0.25, "float32"),
                )
                lv5: R.Tensor((1,), dtype="float32") = R.dequantize(
                    tvmgen_tensor_2,
                    lv4,
                    R.const(0, "int32"),
                    out_dtype="float32",
                    axis=0,
                )
                lv6: R.Tensor((1, 1, 1, 1), dtype="float32") = R.add(lv3, lv5)
                gv: R.Tensor((1, 1, 1, 1), dtype="int8") = R.quantize(
                    lv6,
                    R.const(1.0, "float32"),
                    R.const(0, "int32"),
                    out_dtype="int8",
                    axis=0,
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_quantized_fully_connected_with_int32_bias_dequantizes_bias():
    """Quantized FullyConnected with INT32 bias dequantizes bias with in_scale x wt_scale."""
    import flatbuffers
    import tflite.Model

    builder = flatbuffers.Builder(2048)

    in_q = _build_quantization_parameters(
        builder, scale=[0.5], zero_point=[3], quantized_dimension=0
    )
    wt_q = _build_quantization_parameters(
        builder, scale=[0.25], zero_point=[0], quantized_dimension=0
    )
    out_q = _build_quantization_parameters(
        builder, scale=[1.0], zero_point=[0], quantized_dimension=0
    )

    t_in = _build_tensor(builder, 0, [1, 4], tensor_type=_tfl_tensor_type.INT8, quantization=in_q)
    t_wt = _build_tensor(builder, 1, [2, 4], tensor_type=_tfl_tensor_type.INT8, quantization=wt_q)
    t_bi = _build_tensor(builder, 2, [2], tensor_type=_tfl_tensor_type.INT32)
    t_ou = _build_tensor(builder, 3, [1, 2], tensor_type=_tfl_tensor_type.INT8, quantization=out_q)

    _tfl_fully_connected_options.FullyConnectedOptionsStart(builder)
    _tfl_fully_connected_options.FullyConnectedOptionsAddFusedActivationFunction(builder, 0)
    _tfl_fully_connected_options.FullyConnectedOptionsAddWeightsFormat(
        builder, _tfl_fc_weights_format.DEFAULT
    )
    _tfl_fully_connected_options.FullyConnectedOptionsAddKeepNumDims(builder, 0)
    fc_opts = _tfl_fully_connected_options.FullyConnectedOptionsEnd(builder)

    fc_op = _build_operator(
        builder,
        0,
        [0, 1, 2],
        [3],
        builtin_options_type=_tfl_builtin_options.FullyConnectedOptions,
        builtin_options=fc_opts,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=[t_in, t_wt, t_bi, t_ou],
        operators=[fc_op],
        inputs=[0, 1, 2],
        outputs=[3],
    )
    operator_codes = [_build_operator_code(builder, _tfl_builtin_operator.FULLY_CONNECTED)]
    buf = _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=operator_codes,
        buffers=[_build_buffer(builder)] * 4,
    )

    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)
    mod = from_tflite(tflite_model)
    mod["main"] = mod["main"].without_attr("params")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            tvmgen_tensor_0: R.Tensor((1, 4), dtype="int8"),
            tvmgen_tensor_1: R.Tensor((2, 4), dtype="int8"),
            tvmgen_tensor_2: R.Tensor((2,), dtype="int32"),
        ) -> R.Tensor((1, 2), dtype="int8"):
            R.func_attr({"num_input": 3})
            with R.dataflow():
                lv: R.Tensor((1, 4), dtype="float32") = R.dequantize(
                    tvmgen_tensor_0,
                    R.const(0.5, "float32"),
                    R.const(3, "int32"),
                    out_dtype="float32",
                    axis=0,
                )
                lv1: R.Tensor((4, 2), dtype="int8") = R.permute_dims(
                    tvmgen_tensor_1,
                    axes=[1, 0],
                )
                lv2: R.Tensor((4, 2), dtype="float32") = R.dequantize(
                    lv1,
                    R.const(0.25, "float32"),
                    R.const(0, "int32"),
                    out_dtype="float32",
                    axis=1,
                )
                lv3: R.Tensor((1, 2), dtype="float32") = R.matmul(lv, lv2)
                lv4: R.Tensor((), dtype="float32") = R.multiply(
                    R.const(0.5, "float32"),
                    R.const(0.25, "float32"),
                )
                lv5: R.Tensor((2,), dtype="float32") = R.dequantize(
                    tvmgen_tensor_2,
                    lv4,
                    R.const(0, "int32"),
                    out_dtype="float32",
                    axis=0,
                )
                lv6: R.Tensor((1, 2), dtype="float32") = R.add(lv3, lv5)
                gv: R.Tensor((1, 2), dtype="int8") = R.quantize(
                    lv6,
                    R.const(1.0, "float32"),
                    R.const(0, "int32"),
                    out_dtype="int8",
                    axis=0,
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_fully_connected_flattens_input_without_keep_num_dims():
    builder = flatbuffers.Builder(1024)

    t_in = _build_tensor(builder, 0, [2, 3, 4])
    t_wt = _build_tensor(builder, 0, [5, 4])
    t_ou = _build_tensor(builder, 0, [6, 5])

    _tfl_fully_connected_options.FullyConnectedOptionsStart(builder)
    _tfl_fully_connected_options.FullyConnectedOptionsAddKeepNumDims(builder, False)
    fc_opts = _tfl_fully_connected_options.FullyConnectedOptionsEnd(builder)

    fc_op = _build_operator(
        builder,
        0,
        [0, 1],
        [2],
        builtin_options_type=_tfl_builtin_options.FullyConnectedOptions,
        builtin_options=fc_opts,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=[t_in, t_wt, t_ou],
        operators=[fc_op],
        inputs=[0, 1],
        outputs=[2],
    )
    buf = _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=[_build_operator_code(builder, _tfl_builtin_operator.FULLY_CONNECTED)],
        buffers=[_build_buffer(builder)],
    )
    mod = _load_model_from_buffer(buf)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            tvmgen_tensor_0: R.Tensor((2, 3, 4), dtype="float32"),
            tvmgen_tensor_1: R.Tensor((5, 4), dtype="float32"),
        ) -> R.Tensor((6, 5), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((6, 4), dtype="float32") = R.reshape(tvmgen_tensor_0, R.shape([6, 4]))
                lv1: R.Tensor((4, 5), dtype="float32") = R.permute_dims(
                    tvmgen_tensor_1, axes=[1, 0]
                )
                gv: R.Tensor((6, 5), dtype="float32") = R.matmul(lv, lv1)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def _build_csr_sparsity(
    builder,
    *,
    dense_sizes,
    row_ptrs,
    col_indices,
    sparse_axis,
    traversal_order=None,
):
    row_ptrs_vec = _tflite_int32_table(builder, row_ptrs)
    col_indices_vec = _tflite_int32_table(builder, col_indices)
    dim_metadata = []

    for axis, dense_size in enumerate(dense_sizes):
        _tfl_dimension_metadata.DimensionMetadataStart(builder)
        if axis == sparse_axis:
            _tfl_dimension_metadata.DimensionMetadataAddFormat(
                builder, _tfl_dimension_type.SPARSE_CSR
            )
            _tfl_dimension_metadata.DimensionMetadataAddArraySegmentsType(
                builder, _tfl_sparse_index_vector.Int32Vector
            )
            _tfl_dimension_metadata.DimensionMetadataAddArraySegments(builder, row_ptrs_vec)
            _tfl_dimension_metadata.DimensionMetadataAddArrayIndicesType(
                builder, _tfl_sparse_index_vector.Int32Vector
            )
            _tfl_dimension_metadata.DimensionMetadataAddArrayIndices(builder, col_indices_vec)
        else:
            _tfl_dimension_metadata.DimensionMetadataAddFormat(builder, _tfl_dimension_type.DENSE)
            _tfl_dimension_metadata.DimensionMetadataAddDenseSize(builder, dense_size)
        dim_metadata.append(_tfl_dimension_metadata.DimensionMetadataEnd(builder))

    if traversal_order is None:
        traversal_order = list(range(len(dense_sizes)))

    traversal_order_vec = _tflite_int32_vector(
        builder,
        _tfl_sparsity_parameters.SparsityParametersStartTraversalOrderVector,
        traversal_order,
    )
    dim_metadata_vec = _tflite_offset_vector(
        builder, _tfl_sparsity_parameters.SparsityParametersStartDimMetadataVector, dim_metadata
    )

    _tfl_sparsity_parameters.SparsityParametersStart(builder)
    _tfl_sparsity_parameters.SparsityParametersAddTraversalOrder(builder, traversal_order_vec)
    _tfl_sparsity_parameters.SparsityParametersAddDimMetadata(builder, dim_metadata_vec)
    return _tfl_sparsity_parameters.SparsityParametersEnd(builder)


def _build_densify_only_case(builder):
    sparse_tensor_idx = 0
    dense_tensor_idx = 1
    shape = [2, 2]
    sparsity = _build_csr_sparsity(
        builder,
        dense_sizes=shape,
        row_ptrs=_DENSIFY_ROW_PTRS,
        col_indices=_DENSIFY_COL_INDICES,
        sparse_axis=1,
    )

    sparse_tensor = _build_tensor(builder, 0, shape, sparsity)
    dense_tensor = _build_tensor(builder, 1, shape)
    densify_op = _build_operator(
        builder,
        0,
        [sparse_tensor_idx],
        [dense_tensor_idx],
        _tfl_builtin_options.DensifyOptions,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=[sparse_tensor, dense_tensor],
        operators=[densify_op],
        inputs=[],
        outputs=[dense_tensor_idx],
    )
    operator_codes = [_build_operator_code(builder, _tfl_builtin_operator.DENSIFY)]
    return _DENSIFY_TEST_VALUES, subgraph, operator_codes


def _build_densify_add_case(builder):
    input_tensor_idx = 0
    sparse_tensor_idx = 1
    dense_tensor_idx = 2
    output_tensor_idx = 3
    shape = [2, 2]
    sparsity = _build_csr_sparsity(
        builder,
        dense_sizes=shape,
        row_ptrs=_DENSIFY_ROW_PTRS,
        col_indices=_DENSIFY_COL_INDICES,
        sparse_axis=1,
    )

    input_tensor = _build_tensor(builder, 1, shape)
    sparse_tensor = _build_tensor(builder, 0, shape, sparsity)
    dense_tensor = _build_tensor(builder, 1, shape)
    output_tensor = _build_tensor(builder, 1, shape)

    densify_op = _build_operator(
        builder,
        1,
        [sparse_tensor_idx],
        [dense_tensor_idx],
        _tfl_builtin_options.DensifyOptions,
    )
    _tfl_add_options.AddOptionsStart(builder)
    add_options = _tfl_add_options.AddOptionsEnd(builder)
    add_op = _build_operator(
        builder,
        0,
        [input_tensor_idx, dense_tensor_idx],
        [output_tensor_idx],
        _tfl_builtin_options.AddOptions,
        add_options,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=[input_tensor, sparse_tensor, dense_tensor, output_tensor],
        operators=[densify_op, add_op],
        inputs=[input_tensor_idx],
        outputs=[output_tensor_idx],
    )
    operator_codes = [
        _build_operator_code(builder, _tfl_builtin_operator.ADD),
        _build_operator_code(builder, _tfl_builtin_operator.DENSIFY),
    ]
    return _DENSIFY_TEST_VALUES, subgraph, operator_codes


def _build_densify_conv2d_case(builder):
    input_tensor_idx = 0
    sparse_kernel_idx = 1
    dense_kernel_idx = 2
    output_tensor_idx = 3

    sparsity = _build_csr_sparsity(
        builder,
        dense_sizes=[1, 2, 2, 1],
        row_ptrs=_DENSIFY_ROW_PTRS,
        col_indices=_DENSIFY_COL_INDICES,
        sparse_axis=2,
    )

    input_tensor = _build_tensor(builder, 1, [1, 4, 4, 1])
    sparse_kernel = _build_tensor(builder, 0, [1, 2, 2, 1], sparsity)
    dense_kernel = _build_tensor(builder, 1, [1, 2, 2, 1])
    output_tensor = _build_tensor(builder, 1, [1, 4, 4, 1])

    _tfl_conv2d_options.Conv2DOptionsStart(builder)
    _tfl_conv2d_options.Conv2DOptionsAddStrideH(builder, 1)
    _tfl_conv2d_options.Conv2DOptionsAddStrideW(builder, 1)
    _tfl_conv2d_options.Conv2DOptionsAddPadding(builder, _tfl_padding.SAME)
    _tfl_conv2d_options.Conv2DOptionsAddDilationHFactor(builder, 1)
    _tfl_conv2d_options.Conv2DOptionsAddDilationWFactor(builder, 1)
    conv2d_options = _tfl_conv2d_options.Conv2DOptionsEnd(builder)

    densify_op = _build_operator(
        builder,
        1,
        [sparse_kernel_idx],
        [dense_kernel_idx],
        _tfl_builtin_options.DensifyOptions,
    )
    conv2d_op = _build_operator(
        builder,
        0,
        [input_tensor_idx, dense_kernel_idx],
        [output_tensor_idx],
        _tfl_builtin_options.Conv2DOptions,
        conv2d_options,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=[input_tensor, sparse_kernel, dense_kernel, output_tensor],
        operators=[densify_op, conv2d_op],
        inputs=[input_tensor_idx],
        outputs=[output_tensor_idx],
    )
    operator_codes = [
        _build_operator_code(builder, _tfl_builtin_operator.CONV_2D),
        _build_operator_code(builder, _tfl_builtin_operator.DENSIFY),
    ]
    return _DENSIFY_TEST_VALUES, subgraph, operator_codes


def _build_densify_fully_connected_case(builder):
    input_tensor_idx = 0
    sparse_weight_idx = 1
    dense_weight_idx = 2
    output_tensor_idx = 3
    weight_shape = [4, 4]

    sparsity = _build_csr_sparsity(
        builder,
        dense_sizes=weight_shape,
        row_ptrs=_DENSIFY_FC_ROW_PTRS,
        col_indices=_DENSIFY_FC_COL_INDICES,
        sparse_axis=1,
    )

    input_tensor = _build_tensor(builder, 1, [1, 4])
    sparse_weight = _build_tensor(builder, 0, weight_shape, sparsity)
    dense_weight = _build_tensor(builder, 1, weight_shape)
    output_tensor = _build_tensor(builder, 1, [1, 4])

    _tfl_fully_connected_options.FullyConnectedOptionsStart(builder)
    _tfl_fully_connected_options.FullyConnectedOptionsAddWeightsFormat(
        builder, _tfl_fc_weights_format.DEFAULT
    )
    fc_options = _tfl_fully_connected_options.FullyConnectedOptionsEnd(builder)

    densify_op = _build_operator(
        builder,
        1,
        [sparse_weight_idx],
        [dense_weight_idx],
        _tfl_builtin_options.DensifyOptions,
    )
    fc_op = _build_operator(
        builder,
        0,
        [input_tensor_idx, dense_weight_idx],
        [output_tensor_idx],
        _tfl_builtin_options.FullyConnectedOptions,
        fc_options,
    )
    subgraph = _build_subgraph(
        builder,
        tensors=[input_tensor, sparse_weight, dense_weight, output_tensor],
        operators=[densify_op, fc_op],
        inputs=[input_tensor_idx],
        outputs=[output_tensor_idx],
    )
    operator_codes = [
        _build_operator_code(builder, _tfl_builtin_operator.FULLY_CONNECTED),
        _build_operator_code(builder, _tfl_builtin_operator.DENSIFY),
    ]
    return _DENSIFY_FC_WEIGHT_VALUES, subgraph, operator_codes


def _build_densify_model(*, downstream_op=None):
    """Build a sparse TFLite model with DENSIFY operator for testing."""
    scenario_builders = {
        None: _build_densify_only_case,
        "add": _build_densify_add_case,
        "conv2d": _build_densify_conv2d_case,
        "fully_connected": _build_densify_fully_connected_case,
    }
    if downstream_op not in scenario_builders:
        raise ValueError(f"Unsupported DENSIFY downstream op: {downstream_op}")

    builder = flatbuffers.Builder(4096)
    sparse_values, subgraph, operator_codes = scenario_builders[downstream_op](builder)
    sparse_buffer = _build_buffer(builder, sparse_values.tobytes())
    empty_buffer = _build_buffer(builder)
    return _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=operator_codes,
        buffers=[sparse_buffer, empty_buffer],
    )


def _load_densify_module(downstream_op=None):
    """Load a DENSIFY test model and return the converted Relax module."""
    model_bytes = _build_densify_model(downstream_op=downstream_op)
    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(model_bytes, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(model_bytes, 0)
    mod = from_tflite(tflite_model)
    mod["main"] = mod["main"].without_attr("params")
    return mod


def test_densify():
    """Test TFLite DENSIFY operator conversion."""
    mod = _load_densify_module()

    @I.ir_module
    class Expected:
        @R.function
        def main() -> R.Tensor((2, 2), dtype="float32"):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="float32") = R.const(_DENSIFY_TEST_DENSE)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_densify_with_add():
    """Test DENSIFY followed by a downstream ADD operator."""
    mod = _load_densify_module(downstream_op="add")

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 2), dtype="float32")) -> R.Tensor((2, 2), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="float32") = R.add(x, R.const(_DENSIFY_TEST_DENSE))
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_densify_with_conv2d():
    """Test DENSIFY followed by CONV2D - a real-world scenario.

    This simulates a sparse convolution where DENSIFY converts sparse weights
    before CONV2D uses them for inference.
    """
    mod = _load_densify_module(downstream_op="conv2d")

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 4, 4, 1), dtype="float32")) -> R.Tensor(
            (1, 4, 4, 1), dtype="float32"
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 4, 4, 1), dtype="float32") = R.nn.conv2d(
                    x,
                    R.const(_DENSIFY_CONV_KERNEL_DENSE_HWIO),
                    strides=[1, 1],
                    padding=[0, 0, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="HWIO",
                    out_layout="NHWC",
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_densify_with_fully_connected():
    """Test DENSIFY followed by FULLY_CONNECTED - a real-world scenario.

    This simulates a sparse fully connected layer where DENSIFY converts
    sparse weights before matrix multiplication for inference.
    """
    mod = _load_densify_module(downstream_op="fully_connected")

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 4), dtype="float32")) -> R.Tensor((1, 4), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                weight_t: R.Tensor((4, 4), dtype="float32") = R.permute_dims(
                    R.const(_DENSIFY_FC_WEIGHT_DENSE_OI), axes=[1, 0]
                )
                gv: R.Tensor((1, 4), dtype="float32") = R.matmul(x, weight_t)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def _build_dilate_only_case(
    builder, *, input_shape, dilations, dilation_value, dynamic_dilations=False
):
    input_tensor_idx = 0
    dilations_tensor_idx = 1
    padding_value_tensor_idx = 2
    output_tensor_idx = 3

    output_shape = tuple((input_shape[i] - 1) * dilations[i] + 1 for i in range(len(input_shape)))

    input_tensor = _build_tensor(builder, 1, input_shape)
    dilations_tensor = _build_tensor(
        builder, 2, [len(dilations)], tensor_type=_tfl_tensor_type.INT32
    )
    padding_value_tensor = _build_tensor(builder, 3, [])
    output_tensor = _build_tensor(builder, 4, output_shape)

    _tfl_dilate_options.DilateOptionsStart(builder)
    dilate_opts = _tfl_dilate_options.DilateOptionsEnd(builder)

    dilate_op = _build_operator(
        builder,
        0,
        [input_tensor_idx, dilations_tensor_idx, padding_value_tensor_idx],
        [output_tensor_idx],
        builtin_options2_type=_tfl_builtin_options2.DilateOptions,
        builtin_options2=dilate_opts,
    )
    sg_inputs = (
        [input_tensor_idx, dilations_tensor_idx] if dynamic_dilations else [input_tensor_idx]
    )
    subgraph = _build_subgraph(
        builder,
        tensors=[input_tensor, dilations_tensor, padding_value_tensor, output_tensor],
        operators=[dilate_op],
        inputs=sg_inputs,
        outputs=[output_tensor_idx],
    )
    operator_codes = [_build_operator_code(builder, _tfl_builtin_operator.DILATE)]
    return subgraph, operator_codes


def test_dilate():
    """TFLite DILATE with constant dilations"""
    builder = flatbuffers.Builder(1024)
    input_shape = (3, 4)
    dilations = [2, 2]
    dilation_value = 0.5

    subgraph, operator_codes = _build_dilate_only_case(
        builder,
        input_shape=input_shape,
        dilations=dilations,
        dilation_value=dilation_value,
    )

    buffers = [
        _build_buffer(builder),
        _build_buffer(builder),
        _build_buffer(builder, np.asarray(dilations, dtype=np.int32).tobytes()),
        _build_buffer(builder, np.asarray([dilation_value], dtype=np.float32).tobytes()),
        _build_buffer(builder),
    ]

    buf = _finish_tflite_model(
        builder, subgraph=subgraph, operator_codes=operator_codes, buffers=buffers
    )
    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)
    mod = from_tflite(tflite_model)
    mod["main"] = mod["main"].without_attr("params")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            tvmgen_tensor_0: R.Tensor((3, 4), dtype="float32"),
        ) -> R.Tensor((5, 7), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((3, 1, 4), dtype="float32") = R.reshape(
                    tvmgen_tensor_0, R.shape([3, 1, 4])
                )
                lv1: R.Tensor((3, 1, 4), dtype="float32") = R.full(
                    R.shape([3, 1, 4]), R.const(0.5, "float32"), dtype="float32"
                )
                lv2: R.Tensor((3, 2, 4), dtype="float32") = R.concat((lv, lv1), axis=1)
                lv3: R.Tensor((6, 4), dtype="float32") = R.reshape(lv2, R.shape([6, 4]))
                lv4: R.Tensor((5, 4), dtype="float32") = R.strided_slice(
                    lv3, [0, 1], [0, 0], [5, 4], [1, 1], assume_inbound=False
                )
                lv5: R.Tensor((5, 4, 1), dtype="float32") = R.reshape(lv4, R.shape([5, 4, 1]))
                lv6: R.Tensor((5, 4, 1), dtype="float32") = R.full(
                    R.shape([5, 4, 1]), R.const(0.5, "float32"), dtype="float32"
                )
                lv7: R.Tensor((5, 4, 2), dtype="float32") = R.concat((lv5, lv6), axis=2)
                lv8: R.Tensor((5, 8), dtype="float32") = R.reshape(lv7, R.shape([5, 8]))
                gv: R.Tensor((5, 7), dtype="float32") = R.strided_slice(
                    lv8, [0, 1], [0, 0], [5, 7], [1, 1], assume_inbound=False
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_dilate_dynamic_dilations():
    """DILATE with runtime dilations"""
    builder = flatbuffers.Builder(1024)
    input_shape = (3, 4)
    dilations_for_shape = [2, 2]
    dilation_value = 0.5

    subgraph, operator_codes = _build_dilate_only_case(
        builder,
        input_shape=input_shape,
        dilations=dilations_for_shape,
        dilation_value=dilation_value,
        dynamic_dilations=True,
    )

    buffers = [
        _build_buffer(builder),
        _build_buffer(builder),
        _build_buffer(builder),  # dilations is a runtime input so empty buffer
        _build_buffer(builder, np.asarray([dilation_value], dtype=np.float32).tobytes()),
        _build_buffer(builder),
    ]

    buf = _finish_tflite_model(
        builder, subgraph=subgraph, operator_codes=operator_codes, buffers=buffers
    )
    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(buf, 0)
    mod = from_tflite(tflite_model)
    mod["main"] = mod["main"].without_attr("params")

    dilate_stride_0 = T.dynamic("dilate_stride_0", "int64")
    dilate_stride_1 = T.dynamic("dilate_stride_1", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            tvmgen_tensor_0: R.Tensor((3, 4), dtype="float32"),
            tvmgen_tensor_1: R.Tensor((2,), dtype="int32"),
        ) -> R.Tensor(dtype="float32", ndim=2):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((2,), dtype="int32") = R.match_cast(
                    tvmgen_tensor_1, R.Tensor((2,), dtype="int32")
                )
                lv1: R.Tensor((2,), dtype="int64") = R.astype(lv, dtype="int64")
                lv2: R.Shape(ndim=2) = R.tensor_to_shape(lv1)
                _lv3: R.Shape([dilate_stride_0, dilate_stride_1]) = R.match_cast(
                    lv2, R.Shape([dilate_stride_0, dilate_stride_1])
                )
                lv4: R.Tensor((3, 1, 4), dtype="float32") = R.reshape(
                    tvmgen_tensor_0, R.shape([3, 1, 4])
                )
                lv5: R.Tensor((3, dilate_stride_0 - 1, 4), dtype="float32") = R.full(
                    R.shape([3, dilate_stride_0 - 1, 4]),
                    R.const(0.5, "float32"),
                    dtype="float32",
                )
                lv6: R.Tensor((3, 1 + (dilate_stride_0 - 1), 4), dtype="float32") = R.concat(
                    (lv4, lv5), axis=1
                )
                lv7: R.Tensor((3 * dilate_stride_0, 4), dtype="float32") = R.reshape(
                    lv6, R.shape([3 * dilate_stride_0, 4])
                )
                lv8: R.Tensor(
                    (T.min(dilate_stride_0 * 2 + 1, dilate_stride_0 * 3), 4),
                    dtype="float32",
                ) = R.strided_slice(
                    lv7,
                    [0, 1],
                    [0, 0],
                    [2 * dilate_stride_0 + 1, 4],
                    [1, 1],
                    assume_inbound=False,
                )
                lv9: R.Tensor((2 * dilate_stride_0 + 1, 4, 1), dtype="float32") = R.reshape(
                    lv8, R.shape([2 * dilate_stride_0 + 1, 4, 1])
                )
                lv10: R.Tensor(
                    (2 * dilate_stride_0 + 1, 4, dilate_stride_1 - 1), dtype="float32"
                ) = R.full(
                    R.shape([2 * dilate_stride_0 + 1, 4, dilate_stride_1 - 1]),
                    R.const(0.5, "float32"),
                    dtype="float32",
                )
                lv11: R.Tensor(
                    (2 * dilate_stride_0 + 1, 4, 1 + (dilate_stride_1 - 1)),
                    dtype="float32",
                ) = R.concat((lv9, lv10), axis=2)
                lv12: R.Tensor((2 * dilate_stride_0 + 1, 4 * dilate_stride_1), dtype="float32") = (
                    R.reshape(lv11, R.shape([2 * dilate_stride_0 + 1, 4 * dilate_stride_1]))
                )
                gv: R.Tensor(
                    (
                        dilate_stride_0 * 2 + 1,
                        T.min(dilate_stride_1 * 3 + 1, dilate_stride_1 * 4),
                    ),
                    dtype="float32",
                ) = R.strided_slice(
                    lv12,
                    [0, 1],
                    [0, 0],
                    [2 * dilate_stride_0 + 1, 3 * dilate_stride_1 + 1],
                    [1, 1],
                    assume_inbound=False,
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


# ── LSTM ──────────────────────────────────────────────────────────────────────


def _build_lstm_model(
    batch,
    input_size,
    num_units,
    input_to_forget_weights,
    input_to_cell_weights,
    input_to_output_weights,
    recurrent_to_forget_weights,
    recurrent_to_cell_weights,
    recurrent_to_output_weights,
    forget_gate_bias,
    cell_bias,
    output_gate_bias,
    activation,
    *,
    cell_clip=0.0,
    proj_clip=0.0,
    include_unsupported=False,
):
    """Build a minimal TFLite flatbuffer model with one LSTM op (coupled input-forget).

    Tensor indices:
      0  - input                       [batch, input_size]
      1  - input_to_forget_weights     [num_units, input_size]   (constant)
      2  - input_to_cell_weights       [num_units, input_size]   (constant)
      3  - input_to_output_weights     [num_units, input_size]   (constant)
      4  - recurrent_to_forget_weights [num_units, num_units]    (constant)
      5  - recurrent_to_cell_weights   [num_units, num_units]    (constant)
      6  - recurrent_to_output_weights [num_units, num_units]    (constant)
      7  - forget_gate_bias            [num_units]               (constant)
      8  - cell_bias                   [num_units]               (constant)
      9  - output_gate_bias            [num_units]               (constant)
      10 - output_state                [batch, num_units]        (input)
      11 - cell_state                  [batch, num_units]        (input)
      12 - output                      [batch, num_units]

    Operator input indices (24 entries, -1 for absent):
      [0, -1, 1, 2, 3, -1, 4, 5, 6, -1, -1, -1, -1, 7, 8, 9, -1, -1, 10, 11, -1, -1, -1, -1]
    """
    builder = flatbuffers.Builder(4096)

    _tfl_lstm_options.LSTMOptionsStart(builder)
    _tfl_lstm_options.LSTMOptionsAddFusedActivationFunction(builder, activation)
    _tfl_lstm_options.LSTMOptionsAddCellClip(builder, cell_clip)
    _tfl_lstm_options.LSTMOptionsAddProjClip(builder, proj_clip)
    lstm_opts = _tfl_lstm_options.LSTMOptionsEnd(builder)

    lstm_op_code = _build_operator_code(builder, _tfl_builtin_operator.LSTM)

    def _t(buf_idx, shape):
        shape_vec = _tflite_shape(builder, shape)
        _tfl_tensor.TensorStart(builder)
        _tfl_tensor.TensorAddBuffer(builder, buf_idx)
        _tfl_tensor.TensorAddHasRank(builder, True)
        _tfl_tensor.TensorAddIsVariable(builder, False)
        _tfl_tensor.TensorAddShape(builder, shape_vec)
        _tfl_tensor.TensorAddType(builder, _tfl_tensor_type.FLOAT32)
        return _tfl_tensor.TensorEnd(builder)

    tensors = [
        # 0: input
        _t(0, [batch, input_size]),
        # 1: input_to_forget_weights (coupled)
        _t(1, [num_units, input_size]),
        # 2: input_to_cell_weights
        _t(2, [num_units, input_size]),
        # 3: input_to_output_weights
        _t(3, [num_units, input_size]),
        # 4: recurrent_to_forget_weights (coupled)
        _t(4, [num_units, num_units]),
        # 5: recurrent_to_cell_weights
        _t(5, [num_units, num_units]),
        # 6: recurrent_to_output_weights
        _t(6, [num_units, num_units]),
        # 7: forget_gate_bias (coupled)
        _t(7, [num_units]),
        # 8: cell_bias
        _t(8, [num_units]),
        # 9: output_gate_bias
        _t(9, [num_units]),
        # 10: output_state (input)
        _t(0, [batch, num_units]),
        # 11: cell_state (input)
        _t(0, [batch, num_units]),
        # 12: output
        _t(0, [batch, num_units]),
    ]

    if include_unsupported:
        tensors.extend(
            [
                _t(0, [num_units]),
                _t(0, [num_units]),
                _t(0, [num_units]),
                _t(0, [num_units, num_units]),
                _t(0, [num_units]),
                _t(0, [num_units]),
                _t(0, [num_units]),
                _t(0, [num_units]),
                _t(0, [num_units]),
            ]
        )

    # Operator input indices: -1 for absent optional inputs
    lstm_inputs = [
        0,
        -1,
        1,
        2,
        3,
        -1,
        4,
        5,
        6,
        13 if include_unsupported else -1,
        14 if include_unsupported else -1,
        15 if include_unsupported else -1,
        -1,
        7,
        8,
        9,
        16 if include_unsupported else -1,
        17 if include_unsupported else -1,
        10,
        11,
        18 if include_unsupported else -1,
        19 if include_unsupported else -1,
        20 if include_unsupported else -1,
        21 if include_unsupported else -1,
    ]

    lstm_op = _build_operator(
        builder,
        0,
        lstm_inputs,
        [12],
        builtin_options_type=_tfl_builtin_options.LSTMOptions,
        builtin_options=lstm_opts,
    )

    subgraph = _build_subgraph(
        builder,
        tensors=tensors,
        operators=[lstm_op],
        inputs=[0, 10, 11],
        outputs=[12],
    )

    buffers = [
        _build_buffer(builder),  # 0: empty
        _build_buffer(builder, input_to_forget_weights.tobytes()),  # 1
        _build_buffer(builder, input_to_cell_weights.tobytes()),  # 2
        _build_buffer(builder, input_to_output_weights.tobytes()),  # 3
        _build_buffer(builder, recurrent_to_forget_weights.tobytes()),  # 4
        _build_buffer(builder, recurrent_to_cell_weights.tobytes()),  # 5
        _build_buffer(builder, recurrent_to_output_weights.tobytes()),  # 6
        _build_buffer(builder, forget_gate_bias.tobytes()),  # 7
        _build_buffer(builder, cell_bias.tobytes()),  # 8
        _build_buffer(builder, output_gate_bias.tobytes()),  # 9
    ]

    if include_unsupported:
        buffers.extend([_build_buffer(builder) for _ in range(9)])

    return _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=[lstm_op_code],
        buffers=buffers,
    )


def test_lstm_none_activation():
    """LSTM with NONE activation uses the cell state before the output gate multiply."""
    from tflite.ActivationFunctionType import ActivationFunctionType

    batch, input_size, num_units = 2, 2, 2
    w_f = np.eye(num_units, input_size, dtype=np.float32)
    w_c = np.eye(num_units, input_size, dtype=np.float32)
    w_o = np.eye(num_units, input_size, dtype=np.float32)
    r_f = np.eye(num_units, dtype=np.float32)
    r_c = np.eye(num_units, dtype=np.float32)
    r_o = np.eye(num_units, dtype=np.float32)
    b_f = np.zeros(num_units, dtype=np.float32)
    b_c = np.zeros(num_units, dtype=np.float32)
    b_o = np.zeros(num_units, dtype=np.float32)

    mod = _load_model_from_buffer(
        _build_lstm_model(
            batch,
            input_size,
            num_units,
            w_f,
            w_c,
            w_o,
            r_f,
            r_c,
            r_o,
            b_f,
            b_c,
            b_o,
            ActivationFunctionType.NONE,
        )
    )

    @I.ir_module
    class Expected:
        @R.function
        def main(
            tvmgen_tensor_0: R.Tensor((2, 2), dtype="float32"),
            tvmgen_tensor_10: R.Tensor((2, 2), dtype="float32"),
            tvmgen_tensor_11: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tensor((2, 2), dtype="float32"):
            R.func_attr({"num_input": 3})
            with R.dataflow():
                lv: R.Tensor((2, 2), dtype="float32") = R.permute_dims(
                    R.const(np.eye(2, dtype=np.float32)), axes=None
                )
                lv1: R.Tensor((2, 2), dtype="float32") = R.matmul(tvmgen_tensor_0, lv)
                lv2: R.Tensor((2, 2), dtype="float32") = R.permute_dims(
                    R.const(np.eye(2, dtype=np.float32)), axes=None
                )
                lv3: R.Tensor((2, 2), dtype="float32") = R.matmul(tvmgen_tensor_10, lv2)
                lv4: R.Tensor((2, 2), dtype="float32") = R.add(lv1, lv3)
                lv5: R.Tensor((2, 2), dtype="float32") = R.add(
                    lv4, R.const(np.zeros(2, dtype=np.float32))
                )
                lv6: R.Tensor((2, 2), dtype="float32") = R.sigmoid(lv5)
                lv7: R.Tensor((2, 2), dtype="float32") = R.permute_dims(
                    R.const(np.eye(2, dtype=np.float32)), axes=None
                )
                lv8: R.Tensor((2, 2), dtype="float32") = R.matmul(tvmgen_tensor_0, lv7)
                lv9: R.Tensor((2, 2), dtype="float32") = R.permute_dims(
                    R.const(np.eye(2, dtype=np.float32)), axes=None
                )
                lv10: R.Tensor((2, 2), dtype="float32") = R.matmul(tvmgen_tensor_10, lv9)
                lv11: R.Tensor((2, 2), dtype="float32") = R.add(lv8, lv10)
                lv12: R.Tensor((2, 2), dtype="float32") = R.add(
                    lv11, R.const(np.zeros(2, dtype=np.float32))
                )
                lv13: R.Tensor((2, 2), dtype="float32") = R.sigmoid(lv12)
                lv14: R.Tensor((2, 2), dtype="float32") = R.multiply(lv13, tvmgen_tensor_11)
                lv15: R.Tensor((2, 2), dtype="float32") = R.subtract(R.const(1.0, "float32"), lv13)
                lv16: R.Tensor((2, 2), dtype="float32") = R.permute_dims(
                    R.const(np.eye(2, dtype=np.float32)), axes=None
                )
                lv17: R.Tensor((2, 2), dtype="float32") = R.matmul(tvmgen_tensor_0, lv16)
                lv18: R.Tensor((2, 2), dtype="float32") = R.permute_dims(
                    R.const(np.eye(2, dtype=np.float32)), axes=None
                )
                lv19: R.Tensor((2, 2), dtype="float32") = R.matmul(tvmgen_tensor_10, lv18)
                lv20: R.Tensor((2, 2), dtype="float32") = R.add(lv17, lv19)
                lv21: R.Tensor((2, 2), dtype="float32") = R.add(
                    lv20, R.const(np.zeros(2, dtype=np.float32))
                )
                lv22: R.Tensor((2, 2), dtype="float32") = R.tanh(lv21)
                lv23: R.Tensor((2, 2), dtype="float32") = R.multiply(lv15, lv22)
                lv24: R.Tensor((2, 2), dtype="float32") = R.add(lv14, lv23)
                gv: R.Tensor((2, 2), dtype="float32") = R.multiply(lv6, lv24)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_lstm_tanh_activation():
    """LSTM with TANH activation applies tanh before the output gate multiply."""
    from tflite.ActivationFunctionType import ActivationFunctionType

    batch, input_size, num_units = 2, 2, 2
    w_f = np.eye(num_units, input_size, dtype=np.float32)
    w_c = np.eye(num_units, input_size, dtype=np.float32)
    w_o = np.eye(num_units, input_size, dtype=np.float32)
    r_f = np.eye(num_units, dtype=np.float32)
    r_c = np.eye(num_units, dtype=np.float32)
    r_o = np.eye(num_units, dtype=np.float32)
    b_f = np.zeros(num_units, dtype=np.float32)
    b_c = np.zeros(num_units, dtype=np.float32)
    b_o = np.zeros(num_units, dtype=np.float32)

    mod = _load_model_from_buffer(
        _build_lstm_model(
            batch,
            input_size,
            num_units,
            w_f,
            w_c,
            w_o,
            r_f,
            r_c,
            r_o,
            b_f,
            b_c,
            b_o,
            ActivationFunctionType.TANH,
        )
    )

    @I.ir_module
    class Expected:
        @R.function
        def main(
            tvmgen_tensor_0: R.Tensor((2, 2), dtype="float32"),
            tvmgen_tensor_10: R.Tensor((2, 2), dtype="float32"),
            tvmgen_tensor_11: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tensor((2, 2), dtype="float32"):
            R.func_attr({"num_input": 3})
            with R.dataflow():
                lv: R.Tensor((2, 2), dtype="float32") = R.permute_dims(
                    R.const(np.eye(2, dtype=np.float32)), axes=None
                )
                lv1: R.Tensor((2, 2), dtype="float32") = R.matmul(tvmgen_tensor_0, lv)
                lv2: R.Tensor((2, 2), dtype="float32") = R.permute_dims(
                    R.const(np.eye(2, dtype=np.float32)), axes=None
                )
                lv3: R.Tensor((2, 2), dtype="float32") = R.matmul(tvmgen_tensor_10, lv2)
                lv4: R.Tensor((2, 2), dtype="float32") = R.add(lv1, lv3)
                lv5: R.Tensor((2, 2), dtype="float32") = R.add(
                    lv4, R.const(np.zeros(2, dtype=np.float32))
                )
                lv6: R.Tensor((2, 2), dtype="float32") = R.sigmoid(lv5)
                lv7: R.Tensor((2, 2), dtype="float32") = R.permute_dims(
                    R.const(np.eye(2, dtype=np.float32)), axes=None
                )
                lv8: R.Tensor((2, 2), dtype="float32") = R.matmul(tvmgen_tensor_0, lv7)
                lv9: R.Tensor((2, 2), dtype="float32") = R.permute_dims(
                    R.const(np.eye(2, dtype=np.float32)), axes=None
                )
                lv10: R.Tensor((2, 2), dtype="float32") = R.matmul(tvmgen_tensor_10, lv9)
                lv11: R.Tensor((2, 2), dtype="float32") = R.add(lv8, lv10)
                lv12: R.Tensor((2, 2), dtype="float32") = R.add(
                    lv11, R.const(np.zeros(2, dtype=np.float32))
                )
                lv13: R.Tensor((2, 2), dtype="float32") = R.sigmoid(lv12)
                lv14: R.Tensor((2, 2), dtype="float32") = R.multiply(lv13, tvmgen_tensor_11)
                lv15: R.Tensor((2, 2), dtype="float32") = R.subtract(R.const(1.0, "float32"), lv13)
                lv16: R.Tensor((2, 2), dtype="float32") = R.permute_dims(
                    R.const(np.eye(2, dtype=np.float32)), axes=None
                )
                lv17: R.Tensor((2, 2), dtype="float32") = R.matmul(tvmgen_tensor_0, lv16)
                lv18: R.Tensor((2, 2), dtype="float32") = R.permute_dims(
                    R.const(np.eye(2, dtype=np.float32)), axes=None
                )
                lv19: R.Tensor((2, 2), dtype="float32") = R.matmul(tvmgen_tensor_10, lv18)
                lv20: R.Tensor((2, 2), dtype="float32") = R.add(lv17, lv19)
                lv21: R.Tensor((2, 2), dtype="float32") = R.add(
                    lv20, R.const(np.zeros(2, dtype=np.float32))
                )
                lv22: R.Tensor((2, 2), dtype="float32") = R.tanh(lv21)
                lv23: R.Tensor((2, 2), dtype="float32") = R.multiply(lv15, lv22)
                lv24: R.Tensor((2, 2), dtype="float32") = R.add(lv14, lv23)
                lv25: R.Tensor((2, 2), dtype="float32") = R.tanh(lv24)
                gv: R.Tensor((2, 2), dtype="float32") = R.multiply(lv6, lv25)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_lstm_rejects_unsupported_features():
    """LSTM with peephole/projection/layer norm tensors should be rejected."""
    from tflite.ActivationFunctionType import ActivationFunctionType

    batch, input_size, num_units = 2, 2, 2
    zeros_w = np.zeros((num_units, input_size), dtype=np.float32)
    zeros_r = np.zeros((num_units, num_units), dtype=np.float32)
    zeros_b = np.zeros(num_units, dtype=np.float32)

    with pytest.raises(NotImplementedError, match="not supported yet"):
        _load_model_from_buffer(
            _build_lstm_model(
                batch,
                input_size,
                num_units,
                zeros_w,
                zeros_w,
                zeros_w,
                zeros_r,
                zeros_r,
                zeros_r,
                zeros_b,
                zeros_b,
                zeros_b,
                ActivationFunctionType.NONE,
                include_unsupported=True,
            )
        )


# ── SVDF ──────────────────────────────────────────────────────────────────────


def _build_svdf_model(
    batch,
    input_size,
    num_units,
    rank,
    memory_size,
    num_filters,
    feat_weights,
    time_weights,
    bias,
    activation,
):
    """Build a minimal TFLite flatbuffer model containing one SVDF op.

    Tensor indices:
      0 - input           [batch, input_size]           (model input)
      1 - feature_weights [num_filters, input_size]     (constant)
      2 - time_weights    [num_filters, memory_size]    (constant)
      3 - bias            [num_units]                   (constant)
      4 - state           [batch, num_filters * memory_size]  (variable, model input)
      5 - output          [batch, num_units]
    """
    builder = flatbuffers.Builder(4096)

    _tfl_svdf_options.SVDFOptionsStart(builder)
    _tfl_svdf_options.SVDFOptionsAddRank(builder, rank)
    _tfl_svdf_options.SVDFOptionsAddFusedActivationFunction(builder, activation)
    svdf_opts = _tfl_svdf_options.SVDFOptionsEnd(builder)

    svdf_op_code = _build_operator_code(builder, _tfl_builtin_operator.SVDF)

    def _t(buf_idx, shape):
        shape_vec = _tflite_shape(builder, shape)
        _tfl_tensor.TensorStart(builder)
        _tfl_tensor.TensorAddBuffer(builder, buf_idx)
        _tfl_tensor.TensorAddHasRank(builder, True)
        _tfl_tensor.TensorAddIsVariable(builder, False)
        _tfl_tensor.TensorAddShape(builder, shape_vec)
        _tfl_tensor.TensorAddType(builder, _tfl_tensor_type.FLOAT32)
        return _tfl_tensor.TensorEnd(builder)

    tensors = [
        _t(0, [batch, input_size]),  # 0: input
        _t(1, [num_filters, input_size]),  # 1: feature_weights
        _t(2, [num_filters, memory_size]),  # 2: time_weights
        _t(3, [num_units]),  # 3: bias
        _t(0, [batch, num_filters * memory_size]),  # 4: state (variable, zero-filled)
        _t(0, [batch, num_units]),  # 5: output
    ]

    svdf_op = _build_operator(
        builder,
        0,
        [0, 1, 2, 3, 4],
        [5],
        builtin_options_type=_tfl_builtin_options.SVDFOptions,
        builtin_options=svdf_opts,
    )

    subgraph = _build_subgraph(
        builder,
        tensors=tensors,
        operators=[svdf_op],
        inputs=[0, 4],
        outputs=[5],
    )

    buffers = [
        _build_buffer(builder),  # 0: empty
        _build_buffer(builder, feat_weights.tobytes()),  # 1
        _build_buffer(builder, time_weights.tobytes()),  # 2
        _build_buffer(builder, bias.tobytes()),  # 3
    ]

    return _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=[svdf_op_code],
        buffers=buffers,
    )


def test_svdf_none_activation():
    """SVDF with NONE activation, verifying output shape and params."""
    from tflite.ActivationFunctionType import ActivationFunctionType

    batch, input_size, num_units, rank, memory_size = 2, 3, 2, 2, 3
    num_filters = num_units * rank
    np.random.seed(42)
    feat_weights = np.random.randn(num_filters, input_size).astype(np.float32)
    time_weights = np.random.randn(num_filters, memory_size).astype(np.float32)
    bias = np.zeros(num_units, dtype=np.float32)

    mod = _load_model_from_buffer(
        _build_svdf_model(
            batch,
            input_size,
            num_units,
            rank,
            memory_size,
            num_filters,
            feat_weights,
            time_weights,
            bias,
            ActivationFunctionType.NONE,
        )
    )

    fn = mod["main"]
    assert len(fn.params) == 2, f"expected 2 params (input, state), got {len(fn.params)}"
    tvm_tirx.ir.assert_structural_equal(
        fn.params[0].ty,
        relax.TensorType((batch, input_size), "float32"),
    )
    tvm_tirx.ir.assert_structural_equal(
        fn.params[1].ty,
        relax.TensorType((batch, num_filters * memory_size), "float32"),
    )
    tvm_tirx.ir.assert_structural_equal(fn.ret_ty, relax.TensorType((batch, num_units), "float32"))


def _build_two_step_shared_state_svdf_model(
    batch,
    input_size,
    num_units,
    rank,
    memory_size,
    feat_weights_0,
    time_weights_0,
    bias_0,
    feat_weights_1,
    time_weights_1,
    bias_1,
    activation,
):
    """Build two consecutive SVDF ops sharing a single state tensor."""
    builder = flatbuffers.Builder(4096)
    num_filters = num_units * rank

    _tfl_svdf_options.SVDFOptionsStart(builder)
    _tfl_svdf_options.SVDFOptionsAddRank(builder, rank)
    _tfl_svdf_options.SVDFOptionsAddFusedActivationFunction(builder, activation)
    svdf_opts = _tfl_svdf_options.SVDFOptionsEnd(builder)

    svdf_op_code = _build_operator_code(builder, _tfl_builtin_operator.SVDF)

    def _t(buf_idx, shape):
        shape_vec = _tflite_shape(builder, shape)
        _tfl_tensor.TensorStart(builder)
        _tfl_tensor.TensorAddBuffer(builder, buf_idx)
        _tfl_tensor.TensorAddHasRank(builder, True)
        _tfl_tensor.TensorAddIsVariable(builder, False)
        _tfl_tensor.TensorAddShape(builder, shape_vec)
        _tfl_tensor.TensorAddType(builder, _tfl_tensor_type.FLOAT32)
        return _tfl_tensor.TensorEnd(builder)

    tensors = [
        _t(0, [batch, input_size]),  # 0 input_0
        _t(1, [num_filters, input_size]),  # 1 feat_weights_0
        _t(2, [num_filters, memory_size]),  # 2 time_weights_0
        _t(3, [num_units]),  # 3 bias_0
        _t(0, [batch, num_filters * memory_size]),  # 4 shared state
        _t(0, [batch, num_units]),  # 5 output_0
        _t(0, [batch, input_size]),  # 6 input_1
        _t(4, [num_filters, input_size]),  # 7 feat_weights_1
        _t(5, [num_filters, memory_size]),  # 8 time_weights_1
        _t(6, [num_units]),  # 9 bias_1
        _t(0, [batch, num_units]),  # 10 output_1
    ]

    svdf_op_0 = _build_operator(
        builder,
        0,
        [0, 1, 2, 3, 4],
        [5],
        builtin_options_type=_tfl_builtin_options.SVDFOptions,
        builtin_options=svdf_opts,
    )
    svdf_op_1 = _build_operator(
        builder,
        0,
        [6, 7, 8, 9, 4],
        [10],
        builtin_options_type=_tfl_builtin_options.SVDFOptions,
        builtin_options=svdf_opts,
    )

    subgraph = _build_subgraph(
        builder,
        tensors=tensors,
        operators=[svdf_op_0, svdf_op_1],
        inputs=[0, 6, 4],
        outputs=[10],
    )

    buffers = [
        _build_buffer(builder),
        _build_buffer(builder, feat_weights_0.tobytes()),
        _build_buffer(builder, time_weights_0.tobytes()),
        _build_buffer(builder, bias_0.tobytes()),
        _build_buffer(builder, feat_weights_1.tobytes()),
        _build_buffer(builder, time_weights_1.tobytes()),
        _build_buffer(builder, bias_1.tobytes()),
    ]

    return _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=[svdf_op_code],
        buffers=buffers,
    )


def test_svdf_shared_state_updates_exp_tab():
    """Two SVDF ops sharing state should use the updated FIFO state in the second step."""
    from tflite.ActivationFunctionType import ActivationFunctionType

    batch, input_size, num_units, rank, memory_size = 1, 1, 1, 2, 3
    feat_weights_0 = np.array([[1.0], [2.0]], dtype=np.float32)
    time_weights_0 = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], dtype=np.float32)
    bias_0 = np.zeros(num_units, dtype=np.float32)

    feat_weights_1 = np.array([[7.0], [11.0]], dtype=np.float32)
    time_weights_1 = np.array([[13.0, 17.0, 19.0], [23.0, 29.0, 31.0]], dtype=np.float32)
    bias_1 = np.zeros(num_units, dtype=np.float32)

    mod = _load_model_from_buffer(
        _build_two_step_shared_state_svdf_model(
            batch,
            input_size,
            num_units,
            rank,
            memory_size,
            feat_weights_0,
            time_weights_0,
            bias_0,
            feat_weights_1,
            time_weights_1,
            bias_1,
            ActivationFunctionType.NONE,
        )
    )

    @I.ir_module
    class Expected:
        @R.function
        def main(
            tvmgen_tensor_0: R.Tensor((1, 1), dtype="float32"),
            tvmgen_tensor_6: R.Tensor((1, 1), dtype="float32"),
            tvmgen_tensor_4: R.Tensor((1, 6), dtype="float32"),
        ) -> R.Tensor((1, 1), dtype="float32"):
            R.func_attr({"num_input": 3})
            with R.dataflow():
                lv: R.Tensor((1, 2, 3), dtype="float32") = R.reshape(
                    tvmgen_tensor_4, R.shape([1, 2, 3])
                )
                lv1: R.Tensor((1, 2, 3), dtype="float32") = R.reshape(
                    R.const(np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], dtype=np.float32)),
                    R.shape([1, 2, 3]),
                )
                lv2: R.Tensor((1, 2, 3), dtype="float32") = R.multiply(lv, lv1)
                lv3: R.Tensor((1, 2), dtype="float32") = R.sum(lv2, axis=[-1], keepdims=False)
                lv4: R.Tensor((1, 1, 2), dtype="float32") = R.reshape(lv3, R.shape([1, 1, 2]))
                lv5: R.Tensor((1, 1), dtype="float32") = R.sum(  # noqa: F841
                    lv4, axis=[-1], keepdims=False
                )
                lv6: R.Tensor((1, 2, 2), dtype="float32") = R.strided_slice(
                    lv,
                    (R.prim_value(2),),
                    (R.prim_value(1),),
                    (R.prim_value(3),),
                    assume_inbound=False,
                )
                lv7: R.Tensor((1, 2), dtype="float32") = R.permute_dims(
                    R.const(np.array([[1.0], [2.0]], dtype=np.float32)), axes=None
                )
                lv8: R.Tensor((1, 2), dtype="float32") = R.matmul(
                    tvmgen_tensor_0,
                    lv7,
                )
                lv9: R.Tensor((1, 2, 1), dtype="float32") = R.expand_dims(lv8, axis=[-1])
                lv10: R.Tensor((1, 2, 3), dtype="float32") = R.concat((lv6, lv9), axis=2)
                lv11: R.Tensor((1, 6), dtype="float32") = R.reshape(lv10, R.shape([1, 6]))
                lv12: R.Tensor((1, 2, 3), dtype="float32") = R.reshape(lv11, R.shape([1, 2, 3]))
                lv13: R.Tensor((1, 2, 3), dtype="float32") = R.reshape(
                    R.const(np.array([[13.0, 17.0, 19.0], [23.0, 29.0, 31.0]], dtype=np.float32)),
                    R.shape([1, 2, 3]),
                )
                lv14: R.Tensor((1, 2, 3), dtype="float32") = R.multiply(lv12, lv13)
                lv15: R.Tensor((1, 2), dtype="float32") = R.sum(lv14, axis=[-1], keepdims=False)
                lv16: R.Tensor((1, 1, 2), dtype="float32") = R.reshape(lv15, R.shape([1, 1, 2]))
                lv17: R.Tensor((1, 1), dtype="float32") = R.sum(lv16, axis=[-1], keepdims=False)
                gv: R.Tensor((1, 1), dtype="float32") = R.add(
                    lv17, R.const(np.zeros(1, dtype=np.float32))
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


# ── UNIDIRECTIONAL_SEQUENCE_LSTM ─────────────────────────────────────────────


def _build_unidirectional_sequence_lstm_model(
    batch,
    time,
    input_size,
    num_units,
    input_to_forget_weights,
    input_to_cell_weights,
    input_to_output_weights,
    recurrent_to_forget_weights,
    recurrent_to_cell_weights,
    recurrent_to_output_weights,
    forget_gate_bias,
    cell_bias,
    output_gate_bias,
    activation,
    *,
    time_major=False,
    cell_clip=0.0,
    proj_clip=0.0,
    projection_weights=None,
):
    """Build a TFLite flatbuffer model with one UNIDIRECTIONAL_SEQUENCE_LSTM op.

    Tensor indices (same layout as single-step LSTM, but input is 3D):
      0  - input                       [batch, time, input_size]
      1  - input_to_forget_weights     [num_units, input_size]
      2  - input_to_cell_weights       [num_units, input_size]
      3  - input_to_output_weights     [num_units, input_size]
      4  - recurrent_to_forget_weights [num_units, num_units]
      5  - recurrent_to_cell_weights   [num_units, num_units]
      6  - recurrent_to_output_weights [num_units, num_units]
      7  - forget_gate_bias            [num_units]
      8  - cell_bias                   [num_units]
      9  - output_gate_bias            [num_units]
      10 - output_state                [batch, num_units]   (model input)
      11 - cell_state                  [batch, num_units]   (model input)
      12 - output                      [batch, time, num_units] or [time, batch, num_units]
    """
    builder = flatbuffers.Builder(4096)

    _tfl_unidirectional_sequence_lstm_options.UnidirectionalSequenceLSTMOptionsStart(builder)
    _tfl_unidirectional_sequence_lstm_options.UnidirectionalSequenceLSTMOptionsAddFusedActivationFunction(
        builder, activation
    )
    _tfl_unidirectional_sequence_lstm_options.UnidirectionalSequenceLSTMOptionsAddTimeMajor(
        builder, time_major
    )
    _tfl_unidirectional_sequence_lstm_options.UnidirectionalSequenceLSTMOptionsAddCellClip(
        builder, cell_clip
    )
    _tfl_unidirectional_sequence_lstm_options.UnidirectionalSequenceLSTMOptionsAddProjClip(
        builder, proj_clip
    )
    lstm_opts = _tfl_unidirectional_sequence_lstm_options.UnidirectionalSequenceLSTMOptionsEnd(
        builder
    )

    lstm_op_code = _build_operator_code(builder, _tfl_builtin_operator.UNIDIRECTIONAL_SEQUENCE_LSTM)

    def _t(buf_idx, shape):
        shape_vec = _tflite_shape(builder, shape)
        _tfl_tensor.TensorStart(builder)
        _tfl_tensor.TensorAddBuffer(builder, buf_idx)
        _tfl_tensor.TensorAddHasRank(builder, True)
        _tfl_tensor.TensorAddIsVariable(builder, False)
        _tfl_tensor.TensorAddShape(builder, shape_vec)
        _tfl_tensor.TensorAddType(builder, _tfl_tensor_type.FLOAT32)
        return _tfl_tensor.TensorEnd(builder)

    input_shape = [time, batch, input_size] if time_major else [batch, time, input_size]
    output_shape = [time, batch, num_units] if time_major else [batch, time, num_units]
    tensors = [
        _t(0, input_shape),  # 0: input
        _t(1, [num_units, input_size]),  # 1: input_to_forget_weights
        _t(2, [num_units, input_size]),  # 2: input_to_cell_weights
        _t(3, [num_units, input_size]),  # 3: input_to_output_weights
        _t(4, [num_units, num_units]),  # 4: recurrent_to_forget_weights
        _t(5, [num_units, num_units]),  # 5: recurrent_to_cell_weights
        _t(6, [num_units, num_units]),  # 6: recurrent_to_output_weights
        _t(7, [num_units]),  # 7: forget_gate_bias
        _t(8, [num_units]),  # 8: cell_bias
        _t(9, [num_units]),  # 9: output_gate_bias
        _t(0, [batch, num_units]),  # 10: output_state (model input)
        _t(0, [batch, num_units]),  # 11: cell_state (model input)
        _t(0, output_shape),  # 12: output
    ]

    # 24 operator inputs, -1 for absent.
    lstm_inputs = [
        0,
        -1,
        1,
        2,
        3,
        -1,
        4,
        5,
        6,
        -1,
        -1,
        -1,
        -1,
        7,
        8,
        9,
        -1,
        -1,
        10,
        11,
        -1,
        -1,
        -1,
        -1,
    ]
    buffers = [
        _build_buffer(builder),  # 0: empty
        _build_buffer(builder, input_to_forget_weights.tobytes()),  # 1
        _build_buffer(builder, input_to_cell_weights.tobytes()),  # 2
        _build_buffer(builder, input_to_output_weights.tobytes()),  # 3
        _build_buffer(builder, recurrent_to_forget_weights.tobytes()),  # 4
        _build_buffer(builder, recurrent_to_cell_weights.tobytes()),  # 5
        _build_buffer(builder, recurrent_to_output_weights.tobytes()),  # 6
        _build_buffer(builder, forget_gate_bias.tobytes()),  # 7
        _build_buffer(builder, cell_bias.tobytes()),  # 8
        _build_buffer(builder, output_gate_bias.tobytes()),  # 9
    ]
    if projection_weights is not None:
        tensors.append(_t(len(buffers), [num_units, num_units]))
        lstm_inputs[16] = len(tensors) - 1
        buffers.append(_build_buffer(builder, projection_weights.tobytes()))

    lstm_op = _build_operator(
        builder,
        0,
        lstm_inputs,
        [12],
        builtin_options_type=_tfl_builtin_options.UnidirectionalSequenceLSTMOptions,
        builtin_options=lstm_opts,
    )

    subgraph = _build_subgraph(
        builder,
        tensors=tensors,
        operators=[lstm_op],
        inputs=[0, 10, 11],
        outputs=[12],
    )

    return _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=[lstm_op_code],
        buffers=buffers,
    )


def test_unidirectional_sequence_lstm_none_activation():
    """UNIDIRECTIONAL_SEQUENCE_LSTM with NONE activation keeps cell activation linear."""
    from tflite.ActivationFunctionType import ActivationFunctionType

    batch, time, input_size, num_units = 2, 1, 2, 2
    w_f = np.eye(num_units, input_size, dtype=np.float32)
    w_c = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    w_o = np.array([[0.5, -0.25], [0.75, 0.5]], dtype=np.float32)
    r_f = np.eye(num_units, dtype=np.float32)
    r_c = np.array([[0.5, 0.0], [0.0, 0.25]], dtype=np.float32)
    r_o = np.array([[0.1, 0.0], [0.0, 0.2]], dtype=np.float32)
    b_f = np.zeros(num_units, dtype=np.float32)
    b_c = np.zeros(num_units, dtype=np.float32)
    b_o = np.zeros(num_units, dtype=np.float32)

    mod = _load_model_from_buffer(
        _build_unidirectional_sequence_lstm_model(
            batch,
            time,
            input_size,
            num_units,
            w_f,
            w_c,
            w_o,
            r_f,
            r_c,
            r_o,
            b_f,
            b_c,
            b_o,
            ActivationFunctionType.NONE,
        )
    )

    fn = mod["main"]
    tvm_tirx.ir.assert_structural_equal(
        fn.params[0].ty,
        relax.TensorType((batch, time, input_size), "float32"),
    )
    tvm_tirx.ir.assert_structural_equal(
        fn.ret_ty,
        relax.TensorType((batch, time, num_units), "float32"),
    )


def test_unidirectional_sequence_lstm_tanh_activation():
    """UNIDIRECTIONAL_SEQUENCE_LSTM with TANH activation applies it inside the cell."""
    from tflite.ActivationFunctionType import ActivationFunctionType

    batch, time, input_size, num_units = 2, 1, 2, 2
    w_f = np.eye(num_units, input_size, dtype=np.float32)
    w_c = np.array([[1.0, -1.0], [0.25, 0.5]], dtype=np.float32)
    w_o = np.array([[0.5, 0.5], [-0.5, 1.0]], dtype=np.float32)
    r_f = np.eye(num_units, dtype=np.float32)
    r_c = np.array([[0.0, 0.1], [0.2, 0.0]], dtype=np.float32)
    r_o = np.array([[0.3, 0.0], [0.0, 0.4]], dtype=np.float32)
    b_f = np.zeros(num_units, dtype=np.float32)
    b_c = np.zeros(num_units, dtype=np.float32)
    b_o = np.zeros(num_units, dtype=np.float32)

    mod = _load_model_from_buffer(
        _build_unidirectional_sequence_lstm_model(
            batch,
            time,
            input_size,
            num_units,
            w_f,
            w_c,
            w_o,
            r_f,
            r_c,
            r_o,
            b_f,
            b_c,
            b_o,
            ActivationFunctionType.TANH,
        )
    )

    fn = mod["main"]
    tvm_tirx.ir.assert_structural_equal(
        fn.params[0].ty,
        relax.TensorType((batch, time, input_size), "float32"),
    )
    tvm_tirx.ir.assert_structural_equal(
        fn.ret_ty,
        relax.TensorType((batch, time, num_units), "float32"),
    )


def test_unidirectional_sequence_lstm_time_major():
    """UNIDIRECTIONAL_SEQUENCE_LSTM preserves time-major output layout."""
    from tflite.ActivationFunctionType import ActivationFunctionType

    batch, time, input_size, num_units = 2, 3, 2, 2
    weights = np.eye(num_units, input_size, dtype=np.float32)
    recurrent = np.eye(num_units, dtype=np.float32)
    bias = np.zeros(num_units, dtype=np.float32)

    mod = _load_model_from_buffer(
        _build_unidirectional_sequence_lstm_model(
            batch,
            time,
            input_size,
            num_units,
            weights,
            weights,
            weights,
            recurrent,
            recurrent,
            recurrent,
            bias,
            bias,
            bias,
            ActivationFunctionType.NONE,
            time_major=True,
        )
    )

    fn = mod["main"]
    tvm_tirx.ir.assert_structural_equal(
        fn.params[0].ty,
        relax.TensorType((time, batch, input_size), "float32"),
    )
    tvm_tirx.ir.assert_structural_equal(
        fn.ret_ty,
        relax.TensorType((time, batch, num_units), "float32"),
    )


def test_unidirectional_sequence_lstm_rejects_projection():
    """UNIDIRECTIONAL_SEQUENCE_LSTM rejects unsupported projection inputs."""
    from tflite.ActivationFunctionType import ActivationFunctionType

    batch, time, input_size, num_units = 2, 2, 2, 2
    weights = np.eye(num_units, input_size, dtype=np.float32)
    recurrent = np.eye(num_units, dtype=np.float32)
    bias = np.zeros(num_units, dtype=np.float32)

    with pytest.raises(NotImplementedError, match="projection LSTM"):
        _load_model_from_buffer(
            _build_unidirectional_sequence_lstm_model(
                batch,
                time,
                input_size,
                num_units,
                weights,
                weights,
                weights,
                recurrent,
                recurrent,
                recurrent,
                bias,
                bias,
                bias,
                ActivationFunctionType.NONE,
                projection_weights=np.eye(num_units, dtype=np.float32),
            )
        )


# ── BIDIRECTIONAL_SEQUENCE_RNN ───────────────────────────────────────────────


def _build_bidirectional_sequence_rnn_model(
    batch,
    time,
    input_size,
    num_units,
    fw_weights,
    fw_recurrent_weights,
    fw_bias,
    bw_weights,
    bw_recurrent_weights,
    bw_bias,
    activation,
    *,
    time_major=False,
    merge_outputs=True,
    with_aux_input=False,
):
    """Build a TFLite flatbuffer model with one BIDIRECTIONAL_SEQUENCE_RNN op.

    Tensor indices:
      0  - input               [batch, time, input_size]
      1  - fw_weights          [num_units, input_size]
      2  - fw_recurrent_weights [num_units, num_units]
      3  - fw_bias             [num_units]
      4  - fw_hidden_state     [batch, num_units]   (model input)
      5  - bw_weights          [num_units, input_size]
      6  - bw_recurrent_weights [num_units, num_units]
      7  - bw_bias             [num_units]
      8  - bw_hidden_state     [batch, num_units]   (model input)
      9  - aux_input           (optional)
      10 - fw_aux_weights      (optional)
      11 - bw_aux_weights      (optional)
      12 - output (or fw_output if merge_outputs=False)
      13 - bw_output (only if merge_outputs=False)
    """
    builder = flatbuffers.Builder(4096)

    _tfl_bidirectional_sequence_rnn_options.BidirectionalSequenceRNNOptionsStart(builder)
    _tfl_bidirectional_sequence_rnn_options.BidirectionalSequenceRNNOptionsAddTimeMajor(
        builder, time_major
    )
    _tfl_bidirectional_sequence_rnn_options.BidirectionalSequenceRNNOptionsAddFusedActivationFunction(
        builder, activation
    )
    _tfl_bidirectional_sequence_rnn_options.BidirectionalSequenceRNNOptionsAddMergeOutputs(
        builder, merge_outputs
    )
    rnn_opts = _tfl_bidirectional_sequence_rnn_options.BidirectionalSequenceRNNOptionsEnd(builder)

    rnn_op_code = _build_operator_code(builder, _tfl_builtin_operator.BIDIRECTIONAL_SEQUENCE_RNN)

    def _t(buf_idx, shape):
        shape_vec = _tflite_shape(builder, shape)
        _tfl_tensor.TensorStart(builder)
        _tfl_tensor.TensorAddBuffer(builder, buf_idx)
        _tfl_tensor.TensorAddHasRank(builder, True)
        _tfl_tensor.TensorAddIsVariable(builder, False)
        _tfl_tensor.TensorAddShape(builder, shape_vec)
        _tfl_tensor.TensorAddType(builder, _tfl_tensor_type.FLOAT32)
        return _tfl_tensor.TensorEnd(builder)

    input_shape = [time, batch, input_size] if time_major else [batch, time, input_size]
    output_prefix = [time, batch] if time_major else [batch, time]
    output_shape = output_prefix + ([num_units * 2] if merge_outputs else [num_units])

    tensors = [
        _t(0, input_shape),  # 0: input
        _t(1, [num_units, input_size]),  # 1: fw_weights
        _t(2, [num_units, num_units]),  # 2: fw_recurrent_weights
        _t(3, [num_units]),  # 3: fw_bias
        _t(0, [batch, num_units]),  # 4: fw_hidden_state (model input)
        _t(4, [num_units, input_size]),  # 5: bw_weights
        _t(5, [num_units, num_units]),  # 6: bw_recurrent_weights
        _t(6, [num_units]),  # 7: bw_bias
        _t(0, [batch, num_units]),  # 8: bw_hidden_state (model input)
    ]
    buffers = [
        _build_buffer(builder),  # 0: empty
        _build_buffer(builder, fw_weights.tobytes()),  # 1
        _build_buffer(builder, fw_recurrent_weights.tobytes()),  # 2
        _build_buffer(builder, fw_bias.tobytes()),  # 3
        _build_buffer(builder, bw_weights.tobytes()),  # 4
        _build_buffer(builder, bw_recurrent_weights.tobytes()),  # 5
        _build_buffer(builder, bw_bias.tobytes()),  # 6
    ]
    rnn_inputs = [*list(range(9)), -1, -1, -1]
    if with_aux_input:
        tensors.extend(
            [
                _t(len(buffers), input_shape),
                _t(len(buffers) + 1, [num_units, input_size]),
                _t(len(buffers) + 2, [num_units, input_size]),
            ]
        )
        rnn_inputs[9:12] = [len(tensors) - 3, len(tensors) - 2, len(tensors) - 1]
        buffers.extend(
            [
                _build_buffer(builder, np.zeros(input_shape, dtype=np.float32).tobytes()),
                _build_buffer(
                    builder, np.zeros((num_units, input_size), dtype=np.float32).tobytes()
                ),
                _build_buffer(
                    builder, np.zeros((num_units, input_size), dtype=np.float32).tobytes()
                ),
            ]
        )

    if merge_outputs:
        tensors.append(_t(0, output_shape))
        outputs = [len(tensors) - 1]
    else:
        tensors.extend([_t(0, output_shape), _t(0, output_shape)])
        outputs = [len(tensors) - 2, len(tensors) - 1]

    rnn_op = _build_operator(
        builder,
        0,
        rnn_inputs,
        outputs,
        builtin_options_type=_tfl_builtin_options.BidirectionalSequenceRNNOptions,
        builtin_options=rnn_opts,
    )

    subgraph = _build_subgraph(
        builder,
        tensors=tensors,
        operators=[rnn_op],
        inputs=[0, 4, 8],
        outputs=outputs,
    )

    return _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=[rnn_op_code],
        buffers=buffers,
    )


def test_bidirectional_sequence_rnn_none_activation():
    """BIDIRECTIONAL_SEQUENCE_RNN with NONE activation lowers the expected equations."""
    from tflite.ActivationFunctionType import ActivationFunctionType

    batch, time, input_size, num_units = 2, 1, 2, 2
    fw_w = np.array([[1.0, 0.0], [0.5, -1.0]], dtype=np.float32)
    fw_r = np.array([[0.25, 0.0], [0.0, 0.5]], dtype=np.float32)
    fw_b = np.zeros(num_units, dtype=np.float32)
    bw_w = np.array([[0.0, 1.0], [-0.5, 0.75]], dtype=np.float32)
    bw_r = np.array([[0.1, 0.0], [0.0, 0.2]], dtype=np.float32)
    bw_b = np.zeros(num_units, dtype=np.float32)

    mod = _load_model_from_buffer(
        _build_bidirectional_sequence_rnn_model(
            batch,
            time,
            input_size,
            num_units,
            fw_w,
            fw_r,
            fw_b,
            bw_w,
            bw_r,
            bw_b,
            ActivationFunctionType.NONE,
        )
    )

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 1, 2), dtype="float32"),
            fw_h: R.Tensor((2, 2), dtype="float32"),
            bw_h: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tensor((2, 1, 4), dtype="float32"):
            R.func_attr({"num_input": 3})
            with R.dataflow():
                x_t: R.Tensor((2, 2), dtype="float32") = R.squeeze(x, axis=[1])
                fw_w_t: R.Tensor((2, 2), dtype="float32") = R.permute_dims(R.const(fw_w), axes=None)
                fw_x: R.Tensor((2, 2), dtype="float32") = R.matmul(x_t, fw_w_t)
                fw_r_t: R.Tensor((2, 2), dtype="float32") = R.permute_dims(R.const(fw_r), axes=None)
                fw_h_proj: R.Tensor((2, 2), dtype="float32") = R.matmul(fw_h, fw_r_t)
                fw_out: R.Tensor((2, 2), dtype="float32") = R.add(
                    R.add(fw_x, fw_h_proj), R.const(fw_b)
                )
                fw_stacked: R.Tensor((2, 1, 2), dtype="float32") = R.stack((fw_out,), axis=1)
                bw_w_t: R.Tensor((2, 2), dtype="float32") = R.permute_dims(R.const(bw_w), axes=None)
                bw_x: R.Tensor((2, 2), dtype="float32") = R.matmul(x_t, bw_w_t)
                bw_r_t: R.Tensor((2, 2), dtype="float32") = R.permute_dims(R.const(bw_r), axes=None)
                bw_h_proj: R.Tensor((2, 2), dtype="float32") = R.matmul(bw_h, bw_r_t)
                bw_out: R.Tensor((2, 2), dtype="float32") = R.add(
                    R.add(bw_x, bw_h_proj), R.const(bw_b)
                )
                bw_stacked: R.Tensor((2, 1, 2), dtype="float32") = R.stack((bw_out,), axis=1)
                gv: R.Tensor((2, 1, 4), dtype="float32") = R.concat(
                    (fw_stacked, bw_stacked), axis=-1
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_bidirectional_sequence_rnn_time_major():
    """BIDIRECTIONAL_SEQUENCE_RNN preserves time-major output layout."""
    from tflite.ActivationFunctionType import ActivationFunctionType

    batch, time, input_size, num_units = 2, 3, 2, 2
    weights = np.eye(num_units, input_size, dtype=np.float32)
    recurrent = np.eye(num_units, dtype=np.float32)
    bias = np.zeros(num_units, dtype=np.float32)

    mod = _load_model_from_buffer(
        _build_bidirectional_sequence_rnn_model(
            batch,
            time,
            input_size,
            num_units,
            weights,
            recurrent,
            bias,
            weights,
            recurrent,
            bias,
            ActivationFunctionType.NONE,
            time_major=True,
        )
    )

    fn = mod["main"]
    tvm_tirx.ir.assert_structural_equal(
        fn.params[0].ty,
        relax.TensorType((time, batch, input_size), "float32"),
    )
    tvm_tirx.ir.assert_structural_equal(
        fn.ret_ty,
        relax.TensorType((time, batch, num_units * 2), "float32"),
    )


def test_bidirectional_sequence_rnn_rejects_aux_input():
    """BIDIRECTIONAL_SEQUENCE_RNN rejects unsupported auxiliary input tensors."""
    from tflite.ActivationFunctionType import ActivationFunctionType

    batch, time, input_size, num_units = 2, 2, 2, 2
    weights = np.eye(num_units, input_size, dtype=np.float32)
    recurrent = np.eye(num_units, dtype=np.float32)
    bias = np.zeros(num_units, dtype=np.float32)

    with pytest.raises(NotImplementedError, match="aux input"):
        _load_model_from_buffer(
            _build_bidirectional_sequence_rnn_model(
                batch,
                time,
                input_size,
                num_units,
                weights,
                recurrent,
                bias,
                weights,
                recurrent,
                bias,
                ActivationFunctionType.NONE,
                with_aux_input=True,
            )
        )


# ── BIDIRECTIONAL_SEQUENCE_LSTM ──────────────────────────────────────────────


def _build_bidirectional_sequence_lstm_model(
    batch,
    time,
    input_size,
    num_units,
    fw_w_f,
    fw_w_c,
    fw_w_o,
    fw_r_f,
    fw_r_c,
    fw_r_o,
    fw_b_f,
    fw_b_c,
    fw_b_o,
    bw_w_f,
    bw_w_c,
    bw_w_o,
    bw_r_f,
    bw_r_c,
    bw_r_o,
    bw_b_f,
    bw_b_c,
    bw_b_o,
    activation,
    *,
    time_major=False,
    merge_outputs=True,
    cell_clip=0.0,
    proj_clip=0.0,
    with_aux_input=False,
):
    """Build a TFLite flatbuffer model with one BIDIRECTIONAL_SEQUENCE_LSTM op.

    48 operator inputs. Forward LSTM: indices 0-17, Backward LSTM: indices 18-34,
    States: indices 35-38.
    """
    builder = flatbuffers.Builder(8192)

    _tfl_bidirectional_sequence_lstm_options.BidirectionalSequenceLSTMOptionsStart(builder)
    _tfl_bidirectional_sequence_lstm_options.BidirectionalSequenceLSTMOptionsAddFusedActivationFunction(
        builder, activation
    )
    _tfl_bidirectional_sequence_lstm_options.BidirectionalSequenceLSTMOptionsAddTimeMajor(
        builder, time_major
    )
    _tfl_bidirectional_sequence_lstm_options.BidirectionalSequenceLSTMOptionsAddMergeOutputs(
        builder, merge_outputs
    )
    _tfl_bidirectional_sequence_lstm_options.BidirectionalSequenceLSTMOptionsAddCellClip(
        builder, cell_clip
    )
    _tfl_bidirectional_sequence_lstm_options.BidirectionalSequenceLSTMOptionsAddProjClip(
        builder, proj_clip
    )
    lstm_opts = _tfl_bidirectional_sequence_lstm_options.BidirectionalSequenceLSTMOptionsEnd(
        builder
    )

    lstm_op_code = _build_operator_code(builder, _tfl_builtin_operator.BIDIRECTIONAL_SEQUENCE_LSTM)

    def _t(buf_idx, shape, is_variable=False):
        shape_vec = _tflite_shape(builder, shape)
        _tfl_tensor.TensorStart(builder)
        _tfl_tensor.TensorAddBuffer(builder, buf_idx)
        _tfl_tensor.TensorAddHasRank(builder, True)
        _tfl_tensor.TensorAddIsVariable(builder, is_variable)
        _tfl_tensor.TensorAddShape(builder, shape_vec)
        _tfl_tensor.TensorAddType(builder, _tfl_tensor_type.FLOAT32)
        return _tfl_tensor.TensorEnd(builder)

    input_shape = [time, batch, input_size] if time_major else [batch, time, input_size]
    output_size = num_units * 2 if merge_outputs else num_units
    output_shape = ([time, batch] if time_major else [batch, time]) + [output_size]

    tensors = [
        _t(0, input_shape),  # 0: input
        _t(1, [num_units, input_size]),  # 1: fw_w_f
        _t(2, [num_units, input_size]),  # 2: fw_w_c
        _t(3, [num_units, input_size]),  # 3: fw_w_o
        _t(4, [num_units, num_units]),  # 4: fw_r_f
        _t(5, [num_units, num_units]),  # 5: fw_r_c
        _t(6, [num_units, num_units]),  # 6: fw_r_o
        _t(7, [num_units]),  # 7: fw_b_f
        _t(8, [num_units]),  # 8: fw_b_c
        _t(9, [num_units]),  # 9: fw_b_o
        _t(10, [num_units, input_size]),  # 10: bw_w_f
        _t(11, [num_units, input_size]),  # 11: bw_w_c
        _t(12, [num_units, input_size]),  # 12: bw_w_o
        _t(13, [num_units, num_units]),  # 13: bw_r_f
        _t(14, [num_units, num_units]),  # 14: bw_r_c
        _t(15, [num_units, num_units]),  # 15: bw_r_o
        _t(16, [num_units]),  # 16: bw_b_f
        _t(17, [num_units]),  # 17: bw_b_c
        _t(18, [num_units]),  # 18: bw_b_o
        _t(0, [batch, num_units]),  # 19: fw_activation_state (model input)
        _t(0, [batch, num_units]),  # 20: fw_cell_state (model input)
        _t(0, [batch, num_units]),  # 21: bw_activation_state (model input)
        _t(0, [batch, num_units]),  # 22: bw_cell_state (model input)
        _t(0, output_shape),  # 23: output
    ]

    # Build operator inputs: 48 total, with unsupported optional inputs set to -1.
    fw_inputs = [0, -1, 1, 2, 3, -1, 4, 5, 6, -1, -1, -1, -1, 7, 8, 9, -1, -1]
    bw_inputs = [-1, 10, 11, 12, -1, 13, 14, 15, -1, -1, -1, -1, 16, 17, 18, -1, -1]
    states = [19, 20, 21, 22]
    aux_inputs = [-1] * 9
    if with_aux_input:
        tensors.append(_t(0, input_shape))
        aux_inputs[0] = len(tensors) - 1
    lstm_inputs = fw_inputs + bw_inputs + states + aux_inputs

    lstm_op = _build_operator(
        builder,
        0,
        lstm_inputs,
        [23],
        builtin_options_type=_tfl_builtin_options.BidirectionalSequenceLSTMOptions,
        builtin_options=lstm_opts,
    )

    subgraph = _build_subgraph(
        builder,
        tensors=tensors,
        operators=[lstm_op],
        inputs=[0, 19, 20, 21, 22],
        outputs=[23],
    )

    buffers = [
        _build_buffer(builder),  # 0: empty
        _build_buffer(builder, fw_w_f.tobytes()),  # 1
        _build_buffer(builder, fw_w_c.tobytes()),  # 2
        _build_buffer(builder, fw_w_o.tobytes()),  # 3
        _build_buffer(builder, fw_r_f.tobytes()),  # 4
        _build_buffer(builder, fw_r_c.tobytes()),  # 5
        _build_buffer(builder, fw_r_o.tobytes()),  # 6
        _build_buffer(builder, fw_b_f.tobytes()),  # 7
        _build_buffer(builder, fw_b_c.tobytes()),  # 8
        _build_buffer(builder, fw_b_o.tobytes()),  # 9
        _build_buffer(builder, bw_w_f.tobytes()),  # 10
        _build_buffer(builder, bw_w_c.tobytes()),  # 11
        _build_buffer(builder, bw_w_o.tobytes()),  # 12
        _build_buffer(builder, bw_r_f.tobytes()),  # 13
        _build_buffer(builder, bw_r_c.tobytes()),  # 14
        _build_buffer(builder, bw_r_o.tobytes()),  # 15
        _build_buffer(builder, bw_b_f.tobytes()),  # 16
        _build_buffer(builder, bw_b_c.tobytes()),  # 17
        _build_buffer(builder, bw_b_o.tobytes()),  # 18
    ]

    return _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=[lstm_op_code],
        buffers=buffers,
    )


def test_bidirectional_sequence_lstm_none_activation():
    """BIDIRECTIONAL_SEQUENCE_LSTM with NONE activation keeps both cell activations linear."""
    from tflite.ActivationFunctionType import ActivationFunctionType

    batch, time, input_size, num_units = 2, 1, 2, 2

    def _eye_or_randn(m, n):
        if m == n:
            return np.eye(m, dtype=np.float32)
        return np.arange(m * n, dtype=np.float32).reshape(m, n) / 10.0

    fw_w_f = _eye_or_randn(num_units, input_size)
    fw_w_c = np.array([[1.0, -0.5], [0.25, 0.75]], dtype=np.float32)
    fw_w_o = np.array([[0.5, 0.25], [-0.25, 1.0]], dtype=np.float32)
    fw_r_f = _eye_or_randn(num_units, num_units)
    fw_r_c = np.array([[0.2, 0.0], [0.0, 0.3]], dtype=np.float32)
    fw_r_o = np.array([[0.1, 0.0], [0.0, 0.2]], dtype=np.float32)
    fw_b_f = np.zeros(num_units, dtype=np.float32)
    fw_b_c = np.zeros(num_units, dtype=np.float32)
    fw_b_o = np.zeros(num_units, dtype=np.float32)

    bw_w_f = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    bw_w_c = np.array([[0.5, 0.5], [-0.5, 1.0]], dtype=np.float32)
    bw_w_o = np.array([[0.25, -0.25], [0.75, 0.5]], dtype=np.float32)
    bw_r_f = np.array([[0.4, 0.0], [0.0, 0.6]], dtype=np.float32)
    bw_r_c = np.array([[0.3, 0.0], [0.0, 0.2]], dtype=np.float32)
    bw_r_o = np.array([[0.2, 0.0], [0.0, 0.1]], dtype=np.float32)
    bw_b_f = np.zeros(num_units, dtype=np.float32)
    bw_b_c = np.zeros(num_units, dtype=np.float32)
    bw_b_o = np.zeros(num_units, dtype=np.float32)

    mod = _load_model_from_buffer(
        _build_bidirectional_sequence_lstm_model(
            batch,
            time,
            input_size,
            num_units,
            fw_w_f,
            fw_w_c,
            fw_w_o,
            fw_r_f,
            fw_r_c,
            fw_r_o,
            fw_b_f,
            fw_b_c,
            fw_b_o,
            bw_w_f,
            bw_w_c,
            bw_w_o,
            bw_r_f,
            bw_r_c,
            bw_r_o,
            bw_b_f,
            bw_b_c,
            bw_b_o,
            ActivationFunctionType.NONE,
        )
    )

    fn = mod["main"]
    tvm_tirx.ir.assert_structural_equal(
        fn.params[0].ty,
        relax.TensorType((batch, time, input_size), "float32"),
    )
    tvm_tirx.ir.assert_structural_equal(
        fn.ret_ty,
        relax.TensorType((batch, time, num_units * 2), "float32"),
    )


def test_bidirectional_sequence_lstm_time_major():
    """BIDIRECTIONAL_SEQUENCE_LSTM preserves time-major output layout."""
    from tflite.ActivationFunctionType import ActivationFunctionType

    batch, time, input_size, num_units = 2, 3, 2, 2
    weights = np.eye(num_units, input_size, dtype=np.float32)
    recurrent = np.eye(num_units, dtype=np.float32)
    bias = np.zeros(num_units, dtype=np.float32)

    mod = _load_model_from_buffer(
        _build_bidirectional_sequence_lstm_model(
            batch,
            time,
            input_size,
            num_units,
            weights,
            weights,
            weights,
            recurrent,
            recurrent,
            recurrent,
            bias,
            bias,
            bias,
            weights,
            weights,
            weights,
            recurrent,
            recurrent,
            recurrent,
            bias,
            bias,
            bias,
            ActivationFunctionType.NONE,
            time_major=True,
        )
    )

    fn = mod["main"]
    tvm_tirx.ir.assert_structural_equal(
        fn.params[0].ty,
        relax.TensorType((time, batch, input_size), "float32"),
    )
    tvm_tirx.ir.assert_structural_equal(
        fn.ret_ty,
        relax.TensorType((time, batch, num_units * 2), "float32"),
    )


def test_bidirectional_sequence_lstm_rejects_aux_input():
    """BIDIRECTIONAL_SEQUENCE_LSTM rejects unsupported auxiliary inputs."""
    from tflite.ActivationFunctionType import ActivationFunctionType

    batch, time, input_size, num_units = 2, 2, 2, 2
    weights = np.eye(num_units, input_size, dtype=np.float32)
    recurrent = np.eye(num_units, dtype=np.float32)
    bias = np.zeros(num_units, dtype=np.float32)

    with pytest.raises(NotImplementedError, match="aux input"):
        _load_model_from_buffer(
            _build_bidirectional_sequence_lstm_model(
                batch,
                time,
                input_size,
                num_units,
                weights,
                weights,
                weights,
                recurrent,
                recurrent,
                recurrent,
                bias,
                bias,
                bias,
                weights,
                weights,
                weights,
                recurrent,
                recurrent,
                recurrent,
                bias,
                bias,
                bias,
                ActivationFunctionType.NONE,
                with_aux_input=True,
            )
        )


# ── UNIDIRECTIONAL_SEQUENCE_RNN ───────────────────────────────────────────────


def _build_unidirectional_sequence_rnn_model(
    batch,
    time,
    input_size,
    num_units,
    weights,
    recurrent_weights,
    bias,
    activation,
    *,
    time_major=False,
):
    """Build a minimal TFLite flatbuffer model containing one UNIDIRECTIONAL_SEQUENCE_RNN op.

    Tensor layout (indices 0-5):
      0 - input          [batch, time, input_size]  (or [time, batch, input_size] if time_major)
      1 - input_weights  [num_units, input_size]    (constant)
      2 - recurrent_wts  [num_units, num_units]     (constant)
      3 - bias           [num_units]                (constant)
      4 - hidden_state   [batch, num_units]         (variable, zero-initialised)
      5 - output         [batch, time, num_units]
    """
    builder = flatbuffers.Builder(4096)

    _tfl_sequence_rnn_options.SequenceRNNOptionsStart(builder)
    _tfl_sequence_rnn_options.SequenceRNNOptionsAddTimeMajor(builder, time_major)
    _tfl_sequence_rnn_options.SequenceRNNOptionsAddFusedActivationFunction(builder, activation)
    rnn_opts = _tfl_sequence_rnn_options.SequenceRNNOptionsEnd(builder)

    rnn_op_code = _build_operator_code(builder, _tfl_builtin_operator.UNIDIRECTIONAL_SEQUENCE_RNN)

    input_shape = [time, batch, input_size] if time_major else [batch, time, input_size]

    def _t(buf_idx, shape, is_variable=False):
        shape_vec = _tflite_shape(builder, shape)
        _tfl_tensor.TensorStart(builder)
        _tfl_tensor.TensorAddBuffer(builder, buf_idx)
        _tfl_tensor.TensorAddHasRank(builder, True)
        _tfl_tensor.TensorAddIsVariable(builder, is_variable)
        _tfl_tensor.TensorAddShape(builder, shape_vec)
        _tfl_tensor.TensorAddType(builder, _tfl_tensor_type.FLOAT32)
        return _tfl_tensor.TensorEnd(builder)

    tensors = [
        _t(0, input_shape),
        _t(1, [num_units, input_size]),
        _t(2, [num_units, num_units]),
        _t(3, [num_units]),
        _t(4, [batch, num_units], is_variable=True),
        _t(5, [batch, time, num_units]),
    ]

    rnn_op = _build_operator(
        builder,
        0,
        [0, 1, 2, 3, 4],
        [5],
        builtin_options_type=_tfl_builtin_options.SequenceRNNOptions,
        builtin_options=rnn_opts,
    )

    subgraph = _build_subgraph(
        builder,
        tensors=tensors,
        operators=[rnn_op],
        inputs=[0],
        outputs=[5],
    )

    buffers = [
        _build_buffer(builder),
        _build_buffer(builder, weights.tobytes()),
        _build_buffer(builder, recurrent_weights.tobytes()),
        _build_buffer(builder, bias.tobytes()),
        _build_buffer(builder),
        _build_buffer(builder),
    ]

    return _finish_tflite_model(
        builder,
        subgraph=subgraph,
        operator_codes=[rnn_op_code],
        buffers=buffers,
    )


def test_unidirectional_sequence_rnn_none_activation():
    """UNIDIRECTIONAL_SEQUENCE_RNN with NONE activation, time=1, lowers to matmul/add/stack.

    Cell equation: h_t = x_t @ W.T + h_{t-1} @ Wr.T + b  (no activation for NONE)
    """
    from tflite.ActivationFunctionType import ActivationFunctionType

    batch, time, input_size, num_units = 2, 1, 2, 2
    weights = np.eye(num_units, input_size, dtype=np.float32)
    recurrent_weights = np.eye(num_units, dtype=np.float32)
    bias = np.zeros(num_units, dtype=np.float32)

    mod = _load_model_from_buffer(
        _build_unidirectional_sequence_rnn_model(
            batch,
            time,
            input_size,
            num_units,
            weights,
            recurrent_weights,
            bias,
            ActivationFunctionType.NONE,
        )
    )

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 1, 2), dtype="float32")) -> R.Tensor((2, 1, 2), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 2), dtype="float32") = R.squeeze(x, axis=[1])
                lv1: R.Tensor((2, 2), dtype="float32") = R.permute_dims(
                    R.const(np.eye(2, dtype=np.float32)), axes=None
                )
                lv2: R.Tensor((2, 2), dtype="float32") = R.matmul(lv, lv1)
                lv3: R.Tensor((2, 2), dtype="float32") = R.zeros(R.shape([2, 2]), dtype="float32")
                lv4: R.Tensor((2, 2), dtype="float32") = R.permute_dims(
                    R.const(np.eye(2, dtype=np.float32)), axes=None
                )
                lv5: R.Tensor((2, 2), dtype="float32") = R.matmul(lv3, lv4)
                lv6: R.Tensor((2, 2), dtype="float32") = R.add(lv2, lv5)
                lv7: R.Tensor((2, 2), dtype="float32") = R.add(
                    lv6, R.const(np.zeros(2, dtype=np.float32))
                )
                gv: R.Tensor((2, 1, 2), dtype="float32") = R.stack((lv7,), axis=1)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_unidirectional_sequence_rnn_relu_activation():
    """UNIDIRECTIONAL_SEQUENCE_RNN with RELU activation and multiple time steps."""
    from tflite.ActivationFunctionType import ActivationFunctionType

    batch, time, input_size, num_units = 2, 3, 4, 8
    np.random.seed(42)
    weights = np.random.randn(num_units, input_size).astype(np.float32)
    recurrent_weights = np.random.randn(num_units, num_units).astype(np.float32)
    bias = np.random.randn(num_units).astype(np.float32)

    mod = _load_model_from_buffer(
        _build_unidirectional_sequence_rnn_model(
            batch,
            time,
            input_size,
            num_units,
            weights,
            recurrent_weights,
            bias,
            ActivationFunctionType.RELU,
        )
    )

    fn = mod["main"]
    assert len(fn.params) == 1, "only the sequence input should be a graph input"
    tvm_tirx.ir.assert_structural_equal(
        fn.params[0].ty,
        relax.TensorType((batch, time, input_size), "float32"),
    )
    tvm_tirx.ir.assert_structural_equal(
        fn.ret_ty,
        relax.TensorType((batch, time, num_units), "float32"),
    )


def test_unidirectional_sequence_rnn_time_major():
    """UNIDIRECTIONAL_SEQUENCE_RNN with time_major=True transposes before unrolling."""
    from tflite.ActivationFunctionType import ActivationFunctionType

    batch, time, input_size, num_units = 3, 4, 2, 5
    np.random.seed(7)
    weights = np.random.randn(num_units, input_size).astype(np.float32)
    recurrent_weights = np.random.randn(num_units, num_units).astype(np.float32)
    bias = np.zeros(num_units, dtype=np.float32)

    mod = _load_model_from_buffer(
        _build_unidirectional_sequence_rnn_model(
            batch,
            time,
            input_size,
            num_units,
            weights,
            recurrent_weights,
            bias,
            ActivationFunctionType.NONE,
            time_major=True,
        )
    )

    fn = mod["main"]
    # Input to the graph is the raw time-major tensor [time, batch, input_size].
    tvm_tirx.ir.assert_structural_equal(
        fn.params[0].ty,
        relax.TensorType((time, batch, input_size), "float32"),
    )
    # Output is always batch-major [batch, time, num_units].
    tvm_tirx.ir.assert_structural_equal(
        fn.ret_ty,
        relax.TensorType((batch, time, num_units), "float32"),
    )


def test_real():
    class Real(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(2, 4), dtype=tf.complex64)])
        def func(self, x):
            return tf.math.real(x)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 4, 2), dtype="float32")) -> R.Tensor((2, 4), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 4, 1), dtype="float32") = R.strided_slice(
                    x,
                    (R.prim_value(-1),),
                    (R.prim_value(0),),
                    (R.prim_value(1),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )
                gv: R.Tensor((2, 4), dtype="float32") = R.squeeze(lv, axis=[-1])
                R.output(gv)
            return gv

    verify(Real, Expected)


def test_imag():
    class Imag(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(2, 4), dtype=tf.complex64)])
        def func(self, x):
            return tf.math.imag(x)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 4, 2), dtype="float32")) -> R.Tensor((2, 4), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 4, 1), dtype="float32") = R.strided_slice(
                    x,
                    (R.prim_value(-1),),
                    (R.prim_value(1),),
                    (R.prim_value(2),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )
                gv: R.Tensor((2, 4), dtype="float32") = R.squeeze(lv, axis=[-1])
                R.output(gv)
            return gv

    verify(Imag, Expected)


def test_complex_abs():
    class ComplexAbs(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(2, 4), dtype=tf.complex64)])
        def func(self, x):
            return tf.math.abs(x)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 4, 2), dtype="float32")) -> R.Tensor((2, 4), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 4, 1), dtype="float32") = R.strided_slice(
                    x,
                    (R.prim_value(-1),),
                    (R.prim_value(0),),
                    (R.prim_value(1),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )
                lv1: R.Tensor((2, 4), dtype="float32") = R.squeeze(lv, axis=[-1])
                lv2: R.Tensor((2, 4, 1), dtype="float32") = R.strided_slice(
                    x,
                    (R.prim_value(-1),),
                    (R.prim_value(1),),
                    (R.prim_value(2),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )
                lv3: R.Tensor((2, 4), dtype="float32") = R.squeeze(lv2, axis=[-1])
                lv4: R.Tensor((2, 4), dtype="float32") = R.multiply(lv1, lv1)
                lv5: R.Tensor((2, 4), dtype="float32") = R.multiply(lv3, lv3)
                lv6: R.Tensor((2, 4), dtype="float32") = R.add(lv4, lv5)
                gv: R.Tensor((2, 4), dtype="float32") = R.sqrt(lv6)
                R.output(gv)
            return gv

    verify(ComplexAbs, Expected)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
