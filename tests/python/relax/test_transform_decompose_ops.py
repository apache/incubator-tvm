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

from typing import Union

import tvm
import tvm_tirx
import tvm_tirx.script
import tvm_tirx.testing
from tvm import relax
from tvm.relax import Function
from tvm.relax import script as R
from tvm_tirx import IRModule
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def test_batch_norm_inference():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((1, 64, 112, 112), "float32"),
            gamma: R.Tensor((64,), "float32"),
            beta: R.Tensor((64,), "float32"),
            moving_mean: R.Tensor((64,), "float32"),
            moving_var: R.Tensor((64,), "float32"),
        ):
            with R.dataflow():
                bn = R.nn.batch_norm(
                    x,
                    gamma,
                    beta,
                    moving_mean,
                    moving_var,
                    axis=1,
                    epsilon=1e-5,
                    center=True,
                    scale=True,
                )
                gv = bn[0]
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((1, 64, 112, 112), dtype="float32"),
            gamma: R.Tensor((64,), dtype="float32"),
            beta: R.Tensor((64,), dtype="float32"),
            moving_mean: R.Tensor((64,), dtype="float32"),
            moving_var: R.Tensor((64,), dtype="float32"),
        ) -> R.Tensor((1, 64, 112, 112), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 64, 1, 1), dtype="float32") = R.expand_dims(
                    moving_mean, axis=[0, 2, 3]
                )
                lv1: R.Tensor((1, 64, 112, 112), dtype="float32") = R.subtract(x, lv)
                lv2: R.Tensor((1, 64, 1, 1), dtype="float32") = R.expand_dims(
                    moving_var, axis=[0, 2, 3]
                )
                lv3: R.Tensor((1, 64, 1, 1), dtype="float32") = R.add(
                    lv2, R.const(9.9999997473787516e-06, "float32")
                )
                lv4: R.Tensor((1, 64, 1, 1), dtype="float32") = R.sqrt(lv3)
                lv5: R.Tensor((1, 64, 112, 112), dtype="float32") = R.divide(lv1, lv4)
                lv6: R.Tensor((1, 64, 1, 1), dtype="float32") = R.expand_dims(gamma, axis=[0, 2, 3])
                lv7: R.Tensor((1, 64, 112, 112), dtype="float32") = R.multiply(lv5, lv6)
                lv8: R.Tensor((1, 64, 1, 1), dtype="float32") = R.expand_dims(beta, axis=[0, 2, 3])
                lv9: R.Tensor((1, 64, 112, 112), dtype="float32") = R.add(lv7, lv8)
                bn: R.Tuple(
                    R.Tensor((1, 64, 112, 112), dtype="float32"),
                    R.Tensor((64,), dtype="float32"),
                    R.Tensor((64,), dtype="float32"),
                ) = (lv9, moving_mean, moving_var)
                gv: R.Tensor((1, 64, 112, 112), dtype="float32") = bn[0]
                R.output(gv)
            return gv

    After = relax.transform.DecomposeOpsForInference("main")(Before)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


def test_batch_norm_training():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((1, 64, 112, 112), "float32"),
            gamma: R.Tensor((64,), "float32"),
            beta: R.Tensor((64,), "float32"),
            moving_mean: R.Tensor((64,), "float32"),
            moving_var: R.Tensor((64,), "float32"),
        ):
            with R.dataflow():
                bn = R.nn.batch_norm(
                    x,
                    gamma,
                    beta,
                    moving_mean,
                    moving_var,
                    axis=1,
                    epsilon=1e-5,
                    center=True,
                    scale=True,
                    momentum=0.1,
                )
                gv0 = bn[0]
                gv1 = bn[1]
                gv2 = bn[2]
                R.output(gv0, gv1, gv2)
            return gv0, gv1, gv2

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((1, 64, 112, 112), dtype="float32"),
            gamma: R.Tensor((64,), dtype="float32"),
            beta: R.Tensor((64,), dtype="float32"),
            moving_mean: R.Tensor((64,), dtype="float32"),
            moving_var: R.Tensor((64,), dtype="float32"),
        ) -> R.Tuple(
            R.Tensor((1, 64, 112, 112), dtype="float32"),
            R.Tensor((64,), dtype="float32"),
            R.Tensor((64,), dtype="float32"),
        ):
            with R.dataflow():
                # This portion is training-specific, computing the
                # mean/variance of the dataset.
                lv = R.mean(x, axis=[0, 2, 3], keepdims=False)
                lv3 = R.variance(x, axis=[0, 2, 3], keepdims=False)

                # This portion is identical to the batch_norm run during inference
                lv1 = R.expand_dims(lv, axis=[0, 2, 3])
                lv2 = R.subtract(x, lv1)
                lv4 = R.expand_dims(lv3, axis=[0, 2, 3])
                lv5 = R.add(lv4, R.const(9.9999997473787516e-06, "float32"))
                lv6 = R.sqrt(lv5)
                lv7 = R.divide(lv2, lv6)
                lv8 = R.expand_dims(gamma, axis=[0, 2, 3])
                lv9 = R.multiply(lv7, lv8)
                lv10 = R.expand_dims(beta, axis=[0, 2, 3])
                lv11 = R.add(lv9, lv10)
                inner_tuple = (lv11, lv, lv3)
                # This is the result that would be returned from a
                # batch_norm at inference.

                # However, at training we need to update the moving
                # mean/variance, and to return those updated values.
                inner_res = inner_tuple[0]
                lv12 = R.multiply(R.const(0.89999997615814209, "float32"), moving_mean)
                lv13 = R.multiply(R.const(0.10000000149011612, "float32"), lv)
                lv14 = R.add(lv12, lv13)
                lv15 = R.multiply(R.const(0.89999997615814209, "float32"), moving_var)
                lv16 = R.multiply(R.const(0.10000000149011612, "float32"), lv3)
                lv17 = R.add(lv15, lv16)
                bn = (inner_res, lv14, lv17)
                gv0 = bn[0]
                gv1 = bn[1]
                gv2 = bn[2]
                R.output(gv0, gv1, gv2)
            return (gv0, gv1, gv2)

    After = relax.transform.DecomposeOpsForTraining("main")(Before)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


def test_batch_norm_training_negative_axis():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((2, 3, 4), "float32"),
            gamma: R.Tensor((4,), "float32"),
            beta: R.Tensor((4,), "float32"),
            moving_mean: R.Tensor((4,), "float32"),
            moving_var: R.Tensor((4,), "float32"),
        ):
            return R.nn.batch_norm(
                x,
                gamma,
                beta,
                moving_mean,
                moving_var,
                axis=-1,
                training=True,
            )

    After = relax.transform.DecomposeOpsForTraining()(Before)
    relax.analysis.check_well_formed(After, check_ty=True)


def test_batch_norm_multiple_functions():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((1, 64, 112, 112), "float32"),
            gamma: R.Tensor((64,), "float32"),
            beta: R.Tensor((64,), "float32"),
            moving_mean: R.Tensor((64,), "float32"),
            moving_var: R.Tensor((64,), "float32"),
        ):
            with R.dataflow():
                bn = R.nn.batch_norm(
                    x,
                    gamma,
                    beta,
                    moving_mean,
                    moving_var,
                    axis=1,
                    epsilon=1e-5,
                    center=True,
                    scale=True,
                )
                gv = bn[0]
                R.output(gv)
            return gv

        @R.function
        def main1(
            x: R.Tensor((1, 64, 112, 112), "float32"),
            gamma: R.Tensor((64,), "float32"),
            beta: R.Tensor((64,), "float32"),
            moving_mean: R.Tensor((64,), "float32"),
            moving_var: R.Tensor((64,), "float32"),
        ):
            with R.dataflow():
                bn = R.nn.batch_norm(
                    x,
                    gamma,
                    beta,
                    moving_mean,
                    moving_var,
                    axis=1,
                    epsilon=1e-5,
                    center=True,
                    scale=True,
                )
                gv = bn[0]
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main1(
            x: R.Tensor((1, 64, 112, 112), dtype="float32"),
            gamma: R.Tensor((64,), dtype="float32"),
            beta: R.Tensor((64,), dtype="float32"),
            moving_mean: R.Tensor((64,), dtype="float32"),
            moving_var: R.Tensor((64,), dtype="float32"),
        ) -> R.Tensor((1, 64, 112, 112), dtype="float32"):
            with R.dataflow():
                lv10: R.Tensor((1, 64, 1, 1), dtype="float32") = R.expand_dims(
                    moving_mean, axis=[0, 2, 3]
                )
                lv11: R.Tensor((1, 64, 112, 112), dtype="float32") = R.subtract(x, lv10)
                lv12: R.Tensor((1, 64, 1, 1), dtype="float32") = R.expand_dims(
                    moving_var, axis=[0, 2, 3]
                )
                lv13: R.Tensor((1, 64, 1, 1), dtype="float32") = R.add(
                    lv12, R.const(9.9999997473787516e-06, "float32")
                )
                lv14: R.Tensor((1, 64, 1, 1), dtype="float32") = R.sqrt(lv13)
                lv15: R.Tensor((1, 64, 112, 112), dtype="float32") = R.divide(lv11, lv14)
                lv16: R.Tensor((1, 64, 1, 1), dtype="float32") = R.expand_dims(
                    gamma, axis=[0, 2, 3]
                )
                lv17: R.Tensor((1, 64, 112, 112), dtype="float32") = R.multiply(lv15, lv16)
                lv18: R.Tensor((1, 64, 1, 1), dtype="float32") = R.expand_dims(beta, axis=[0, 2, 3])
                lv19: R.Tensor((1, 64, 112, 112), dtype="float32") = R.add(lv17, lv18)
                bn: R.Tuple(
                    R.Tensor((1, 64, 112, 112), dtype="float32"),
                    R.Tensor((64,), dtype="float32"),
                    R.Tensor((64,), dtype="float32"),
                ) = (lv19, moving_mean, moving_var)
                gv: R.Tensor((1, 64, 112, 112), dtype="float32") = bn[0]
                R.output(gv)
            return gv

        @R.function
        def main(
            x: R.Tensor((1, 64, 112, 112), dtype="float32"),
            gamma: R.Tensor((64,), dtype="float32"),
            beta: R.Tensor((64,), dtype="float32"),
            moving_mean: R.Tensor((64,), dtype="float32"),
            moving_var: R.Tensor((64,), dtype="float32"),
        ) -> R.Tensor((1, 64, 112, 112), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 64, 1, 1), dtype="float32") = R.expand_dims(
                    moving_mean, axis=[0, 2, 3]
                )
                lv1: R.Tensor((1, 64, 112, 112), dtype="float32") = R.subtract(x, lv)
                lv2: R.Tensor((1, 64, 1, 1), dtype="float32") = R.expand_dims(
                    moving_var, axis=[0, 2, 3]
                )
                lv3: R.Tensor((1, 64, 1, 1), dtype="float32") = R.add(
                    lv2, R.const(9.9999997473787516e-06, "float32")
                )
                lv4: R.Tensor((1, 64, 1, 1), dtype="float32") = R.sqrt(lv3)
                lv5: R.Tensor((1, 64, 112, 112), dtype="float32") = R.divide(lv1, lv4)
                lv6: R.Tensor((1, 64, 1, 1), dtype="float32") = R.expand_dims(gamma, axis=[0, 2, 3])
                lv7: R.Tensor((1, 64, 112, 112), dtype="float32") = R.multiply(lv5, lv6)
                lv8: R.Tensor((1, 64, 1, 1), dtype="float32") = R.expand_dims(beta, axis=[0, 2, 3])
                lv9: R.Tensor((1, 64, 112, 112), dtype="float32") = R.add(lv7, lv8)
                bn: R.Tuple(
                    R.Tensor((1, 64, 112, 112), dtype="float32"),
                    R.Tensor((64,), dtype="float32"),
                    R.Tensor((64,), dtype="float32"),
                ) = (lv9, moving_mean, moving_var)
                gv: R.Tensor((1, 64, 112, 112), dtype="float32") = bn[0]
                R.output(gv)
            return gv

    After = relax.transform.DecomposeOpsForInference()(Before)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


def test_layer_norm():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((4, 64, 112, 112), "float32"),
            gamma: R.Tensor((112, 112), "float32"),
            beta: R.Tensor((112, 112), "float32"),
        ):
            with R.dataflow():
                ln = R.nn.layer_norm(
                    x,
                    gamma,
                    beta,
                    axes=[-2, -1],
                    epsilon=1e-5,
                    center=True,
                    scale=True,
                )
                R.output(ln)
            return ln

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((4, 64, 112, 112), dtype="float32"),
            gamma: R.Tensor((112, 112), dtype="float32"),
            beta: R.Tensor((112, 112), dtype="float32"),
        ) -> R.Tensor((4, 64, 112, 112), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((4, 64, 1, 1), dtype="float32") = R.mean(
                    x, axis=[-2, -1], keepdims=True
                )
                lv1: R.Tensor((4, 64, 112, 112), dtype="float32") = R.subtract(x, lv)
                lv2: R.Tensor((4, 64, 1, 1), dtype="float32") = R.variance(
                    x, axis=[-2, -1], keepdims=True
                )
                lv3: R.Tensor((4, 64, 1, 1), dtype="float32") = R.add(
                    lv2, R.const(9.9999997473787516e-06, "float32")
                )
                lv4: R.Tensor((4, 64, 1, 1), dtype="float32") = R.sqrt(lv3)
                lv5: R.Tensor((4, 64, 112, 112), dtype="float32") = R.divide(lv1, lv4)
                lv6: R.Tensor((4, 64, 112, 112), dtype="float32") = R.multiply(lv5, gamma)
                ln: R.Tensor((4, 64, 112, 112), dtype="float32") = R.add(lv6, beta)
                R.output(ln)
            return ln

    After = relax.transform.DecomposeOpsForTraining()(Before)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


def test_op_tensor_to_shape():
    @I.ir_module
    class Before:
        @R.function
        def main(t: R.Tensor([3], dtype="int64")):
            gv: R.Shape(ndim=3) = R.tensor_to_shape(t)
            return gv

    x = T.dynamic("x", "int64")
    x_1 = T.dynamic("x_1", "int64")
    x_2 = T.dynamic("x_2", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(t: R.Tensor([3], dtype="int64")) -> R.Shape(ndim=3):
            gv: R.Shape(ndim=3) = R.call_pure_packed(
                "vm.builtin.tensor_to_shape", t, ty_args=(R.Shape(ndim=3),)
            )
            y: R.Shape([x, x_1, x_2]) = R.match_cast(gv, R.Shape([x, x_1, x_2]))
            gv_1: R.Shape([x, x_1, x_2]) = R.shape([x, x_1, x_2])
            return gv_1

    After = relax.transform.DecomposeOpsForInference()(Before)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


if __name__ == "__main__":
    tvm_tirx.testing.main()
