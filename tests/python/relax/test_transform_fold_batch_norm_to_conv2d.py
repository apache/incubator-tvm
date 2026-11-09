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
# ruff: noqa: F401

import numpy as np
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm.relax.expr_functor import PyExprVisitor, visitor
from tvm.relax.script import ir_builder as relax_builder
from tvm_tirx.ir.module import IRModule
from tvm_tirx.script import ir as I
from tvm_tirx.script.ir_builder import IRBuilder


def get_conv2d_batchnorm_sample():
    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name_("main")
            data = R.arg_("data", R.Tensor((1, 3, 224, 224), "float32"))
            weight = R.arg_("weight", R.Tensor((32, 3, 3, 3), "float32"))
            with R.dataflow() as frame:
                output = R.emit(
                    R.nn.conv2d(
                        data,
                        weight,
                        out_dtype="float32",
                        strides=(1, 1),
                        dilation=(1, 1),
                        padding=(1, 1),
                        data_layout="NCHW",
                        kernel_layout="OIHW",
                        groups=1,
                    )
                )
                gamma = R.arg_("gamma", R.Tensor((32,), "float32"))
                beta = R.arg_("beta", R.Tensor((32,), "float32"))
                mean = R.arg_("mean", R.Tensor((32,), "float32"))
                variance = R.arg_("variance", R.Tensor((32,), "float32"))
                output = R.emit(
                    R.nn.batch_norm(output, gamma, beta, mean, variance, axis=1, epsilon=1e-5)[0]
                )
                R.output(output)

            R.func_ret_value(frame.output_vars[0])

    func = builder.get()

    return tvm_tirx.IRModule({"main": func})


def test_fold_batchnorm_info_conv2d():
    mod = get_conv2d_batchnorm_sample()
    mod_fold = get_conv2d_batchnorm_sample()

    target = tvm_tirx.target.Target("llvm", host="llvm")
    data_in = tvm_tirx.runtime.tensor(np.random.rand(1, 3, 224, 224).astype(np.float32))

    weight_data = tvm_tirx.runtime.tensor(np.random.rand(32, 3, 3, 3).astype(np.float32))
    gamma_data = tvm_tirx.runtime.tensor(np.random.rand(32).astype(np.float32))
    beta_data = tvm_tirx.runtime.tensor(np.random.rand(32).astype(np.float32))
    mean_data = tvm_tirx.runtime.tensor(np.random.rand(32).astype(np.float32))
    variance_data = tvm_tirx.runtime.tensor(np.random.rand(32).astype(np.float32))
    params_np = {
        "weight": weight_data,
        "gamma": gamma_data,
        "beta": beta_data,
        "mean": mean_data,
        "variance": variance_data,
    }

    mod = tvm.relax.transform.BindParams("main", params_np)(mod)
    mod_fold = tvm.relax.transform.BindParams("main", params_np)(mod_fold)

    # Normal build
    mod = tvm.relax.transform.DecomposeOpsForInference()(mod)
    ex = tvm.compile(mod, target)
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    out = vm["main"](data_in)

    # Fold BN to Conv2D
    mod_fold = relax.transform.FoldBatchnormToConv2D()(mod_fold)
    mod_fold = relax.transform.FoldConstant()(mod_fold)
    ex_fold = tvm.compile(mod_fold, target)
    vm_fold = relax.VirtualMachine(ex_fold, tvm_tirx.cpu())
    out_fold = vm_fold["main"](data_in)

    tvm_tirx.testing.assert_allclose(out.numpy(), out_fold.numpy(), rtol=1e-5, atol=1e-5)


@visitor
class VerifyFolding(PyExprVisitor):  # pylint: disable=abstract-method
    def visit(self, mod: IRModule) -> None:
        """Entry point"""
        for _, func in mod.functions_items():
            if isinstance(func, relax.Function):
                self.visit_expr(func)

    def visit_call_(self, call: relax.Call) -> None:  # pylint: disable=arguments-renamed
        assert call.op.name != "relax.nn.batch_norm", (
            "Batchnorm op shouldn't be present after folding to previous conv2d"
        )


def test_fold_batchnorm_info_conv2d_transform():
    mod = get_conv2d_batchnorm_sample()
    mod = relax.transform.FoldBatchnormToConv2D()(mod)
    weight_data = tvm_tirx.runtime.tensor(np.random.rand(32, 3, 3, 3).astype(np.float32))
    gamma_data = tvm_tirx.runtime.tensor(np.random.rand(32).astype(np.float32))
    beta_data = tvm_tirx.runtime.tensor(np.random.rand(32).astype(np.float32))
    mean_data = tvm_tirx.runtime.tensor(np.random.rand(32).astype(np.float32))
    variance_data = tvm_tirx.runtime.tensor(np.random.rand(32).astype(np.float32))
    params_np = {
        "weight": weight_data,
        "gamma": gamma_data,
        "beta": beta_data,
        "mean": mean_data,
        "variance": variance_data,
    }
    mod = tvm.relax.transform.BindParams("main", params_np)(mod)
    mod = relax.transform.FoldBatchnormToConv2D()(mod)
    mod = relax.transform.FoldConstant()(mod)

    VerifyFolding().visit(mod)


@pytest.mark.parametrize(
    "axis,scale,center,dtype,should_fold",
    [
        (1, False, True, "float32", False),
        (1, True, False, "float32", False),
        (1, False, False, "float32", False),
        (2, True, True, "float32", False),
        (-2, True, True, "float32", False),
        (1, True, True, "float16", False),
        (1, True, True, "float32", True),
        (-3, True, True, "float32", True),
    ],
)
def test_fold_batchnorm_contracts(axis, scale, center, dtype, should_fold):
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorType((1, 1, 2, 2), dtype))
    weight = relax.const(np.array([1.5, -2.0], dtype=dtype).reshape(2, 1, 1, 1))
    constants = [
        relax.const(np.array(values, dtype="float32"))
        for values in ([2.0, 3.0], [5.0, 7.0], [1.0, -1.0], [4.0, 9.0])
    ]
    with bb.function("main", [x]):
        with bb.dataflow():
            conv = bb.emit(relax.op.nn.conv2d(x, weight, out_dtype="float32"))
            bn = bb.emit(
                relax.op.nn.batch_norm(
                    conv, *constants, axis=axis, scale=scale, center=center, training=False
                )
            )
            out = bb.emit_output(relax.TupleGetItem(bn, 0))
        bb.emit_func_output(out)
    before = bb.finalize()
    after = relax.transform.FoldBatchnormToConv2D()(before)
    assert relax.analysis.check_well_formed(after, check_ty=True)
    if should_fold:
        VerifyFolding().visit(after)
    else:
        tvm.ir.assert_structural_equal(after, before)


if __name__ == "__main__":
    tvm_tirx.testing.main()
