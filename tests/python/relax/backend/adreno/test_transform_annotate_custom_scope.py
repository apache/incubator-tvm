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

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax.expr_functor import PyExprMutator, PyExprVisitor, mutator, visitor
from tvm.relax.transform.legalize_ops import adreno as legalize_adreno
from tvm_tirx.ir.module import IRModule
from tvm_tirx.runtime import MemoryScope
from tvm_tirx.script.parser import ir as I
from tvm_tirx.script.parser import relax as R
from tvm_tirx.script.parser import tirx as T


@visitor
class ValidateScope(PyExprVisitor):  # pylint: disable=abstract-method
    def __init__(self, scope_info: dict) -> None:
        self.scope_info = {
            name: [[MemoryScope(scope) for scope in scopes] for scopes in signature]
            for name, signature in scope_info.items()
        }
        self.matched = True

    def visit(self, mod: IRModule) -> None:
        """Entry point"""
        for _, func in mod.functions_items():
            if isinstance(func, relax.Function):
                self.visit_expr(func)
        return self.matched

    def visit_call_(self, call: relax.Call) -> None:  # pylint: disable=arguments-renamed
        if call.op.name == "relax.call_tir":
            # if call.args[0].name_hint in self.scope_info:
            for idx, arg in enumerate(call.args[1]):
                arg_ty = arg.ty
                assert isinstance(arg_ty, relax.TensorType), (
                    f"Expected TensorType but git {type(arg_ty)}"
                )
                call_mem_scope = (
                    MemoryScope("global") if not arg_ty.vdevice else arg_ty.vdevice.memory_scope
                )
                assert call_mem_scope == self.scope_info[call.args[0].name_hint][0][idx], (
                    f"Scope mismatched for argument {idx} in {call.args[0].name_hint}"
                )
            if isinstance(call.ty_args[0], relax.TensorType):
                call_mem_scope = (
                    MemoryScope("global")
                    if not call.ty_args[0].vdevice
                    else call.ty_args[0].vdevice.memory_scope
                )
                assert call_mem_scope == self.scope_info[call.args[0].name_hint][1][0], (
                    f"Scope mismatched for return scope: {call.args[0].name_hint}"
                )
            else:
                assert isinstance(call.ty_args[0], relax.TupleType), (
                    f"Expected TupleType but git {type(call.ty_args[0])}"
                )
                for idx, ty in enumerate(call.ty_args[0].fields):
                    call_mem_scope = (
                        MemoryScope("global") if not ty.vdevice else ty.vdevice.memory_scope
                    )
                    assert call_mem_scope == self.scope_info[call.args[0].name_hint][1][idx], (
                        f"Scope mismatched for return scope for {idx} in {call.args[0].name_hint}"
                    )


def verify(mod, expected):
    tgt = tvm_tirx.target.Target({"kind": "opencl", "device": "adreno"}, host="llvm")
    skip_ops = [
        "relax.nn.conv2d",
        "relax.nn.max_pool2d",
        "relax.nn.adaptive_avg_pool2d",
        # "relax.nn.layer_norm",
    ]
    mod = tvm_tirx.tirx.transform.BindTarget(tgt)(mod)
    mod = tvm.relax.transform.DecomposeOpsForInference()(mod)
    mod = tvm.relax.transform.FoldConstant()(mod)
    desired_layouts = {"relax.nn.conv2d": ["NCHW4c", "OIHW4o", "NCHW4c"]}
    mod = tvm.relax.transform.ConvertLayout(desired_layouts)(mod)
    mod = tvm.relax.transform.Normalize()(mod)
    mod = tvm.relax.transform.FoldConstant()(mod)
    mod = tvm.relax.transform.LegalizeOps(skip_ops=skip_ops, target=tgt)(mod)
    mod = tvm.relax.transform.AnnotateTIROpPattern()(mod)
    mod = tvm.relax.backend.adreno.transform.AnnotateCustomMemoryScope(tgt)(mod)
    # There is a possibility of some skipped ops above might not use 5D layouts.
    mod = tvm.relax.transform.LegalizeOps(target=tgt)(mod)
    mod = tvm.relax.transform.LegalizeOps(
        {"relax.nn.conv2d": legalize_adreno.conv2d_NCHWc_OIHWo}, target=tgt
    )(mod)
    # Lets get pattern info for newly legalized ops
    mod = tvm.relax.transform.AnnotateTIROpPattern()(mod)
    mod = tvm.relax.transform.FoldConstant()(mod)
    mod = tvm.relax.transform.FuseOps()(mod)
    mod = tvm.relax.transform.FuseTIR()(mod)
    mod = tvm.relax.transform.DeadCodeElimination()(mod)
    mod = tvm.relax.backend.adreno.transform.FoldVDeviceScopeChange()(mod)
    mod = tvm.relax.transform.DeadCodeElimination()(mod)
    mod = tvm.relax.transform.SpecializeFunctionBasedOnCallSite()(mod)
    mod = tvm.relax.transform.Normalize()(mod)

    ValidateScope(expected).visit(mod)


def test_conv2d():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 64, 56, 56), "float32"), w: R.Tensor((32, 64, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 32, 54, 54), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                R.output(gv)
            return gv

    Expected = {
        "te_layout_transform": (["global"], ["global.texture-nhwc"]),
        "te_layout_transform1": (["global"], ["global"]),
        "conv2d_NCHWc_OIHWo_opencl": (["global.texture-nhwc", "global"], ["global"]),
        "te_layout_transform2": (["global"], ["global"]),
    }

    verify(Input, Expected)


def test_conv2d_NCHW_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(
                    x,
                    w,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_dtype="float32",
                )
                R.output(gv)
            return gv

    Expected = {
        "te_layout_transform": (["global"], ["global.texture-weight"]),
        "te_layout_transform1": (["global"], ["global"]),
        "conv2d_NCHWc_OIHWo_opencl": (
            ["global.texture-weight", "global"],
            ["global"],
        ),
        "te_layout_transform2": (["global"], ["global"]),
    }

    verify(Input, Expected)


def test_conv2d_NHWC_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 28, 28, 16), "float32"), w: R.Tensor((4, 3, 3, 16), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 26, 26, 4), "float32") = R.nn.conv2d(
                    x,
                    w,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_dtype="float32",
                )
                R.output(gv)
            return gv

    Expected = {
        "te_layout_transform": (["global"], ["global.texture-weight"]),
        "te_layout_transform1": (["global"], ["global"]),
        "conv2d_NCHWc_OIHWo_opencl": (
            ["global.texture-weight", "global"],
            ["global"],
        ),
        "te_layout_transform2": (["global"], ["global"]),
    }

    verify(Input, Expected)


def _test_conv2d_symbolic_sub_indexed():
    N, H, W = T.dynamic("N", "int64"), T.dynamic("H", "int64"), T.dynamic("W", "int64")
    Hw, Ww = T.dynamic("Hw", "int64"), T.dynamic("Ww", "int64")

    N = T.dynamic("N", "int64")
    H = T.dynamic("H", "int64")
    W = T.dynamic("W", "int64")
    Hw = T.dynamic("Hw", "int64")
    Ww = T.dynamic("Ww", "int64")

    @I.ir_module
    class Input:
        @R.function
        def main(x: R.Tensor("float32", ndim=4), w: R.Tensor("float32", ndim=4)) -> R.Tensor(
            "float32", ndim=4
        ):
            with R.dataflow():
                C = I.meta_var(T.int64(16))
                Nw = I.meta_var(T.int64(4))
                Cw = I.meta_var(T.int64(16))
                lv0 = R.match_cast(x, R.Tensor((N, C, H, W), "float32"))
                lv1 = R.match_cast(w, R.Tensor((Nw, Cw, Hw, Ww), "float32"))
                gv: R.Tensor(
                    (N, T.int64(4), H + T.int64(1) - Hw, W + T.int64(1) - Ww), "float32"
                ) = R.nn.conv2d(lv0, lv1, out_dtype="float32")
                R.output(gv)
            return gv

    Expected = {
        "te_layout_transform": (["global"], ["global.texture-weight"]),
        "te_layout_transform1": (["global"], ["global"]),
        "conv2d_NCHWc_OIHWo_opencl": (
            ["global.texture-weight", "global"],
            ["global"],
        ),
        "te_layout_transform2": (["global"], ["global"]),
    }

    verify(Input, Expected)


def test_conv2d_relu_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
                R.output(gv2)
            return gv2

    Expected = {
        "te_layout_transform": (["global"], ["global.texture-weight"]),
        "te_layout_transform1": (["global"], ["global"]),
        "fused_conv2d_NCHWc_OIHWo_opencl_relu": (
            ["global.texture-weight", "global"],
            ["global"],
        ),
        "te_layout_transform2": (["global"], ["global"]),
    }

    verify(Input, Expected)


def test_relu_conv2d_relu_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                x0: R.Tensor((2, 16, 28, 28), "float32") = R.nn.relu(x)
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x0, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
                R.output(gv2)
            return gv2

    Expected = {
        "relu": (["global"], ["global"]),
        "te_layout_transform": (["global"], ["global.texture-weight"]),
        "te_layout_transform1": (["global"], ["global"]),
        "fused_conv2d_NCHWc_OIHWo_opencl_relu1": (
            ["global.texture-weight", "global"],
            ["global"],
        ),
        "te_layout_transform2": (["global"], ["global"]),
    }

    verify(Input, Expected)


def test_conv2d_relu_tanh_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
                gv3: R.Tensor((2, 4, 26, 26), "float32") = R.tanh(gv2)
                R.output(gv3)
            return gv3

    Expected = {
        "te_layout_transform": (["global"], ["global.texture-weight"]),
        "te_layout_transform1": (["global"], ["global"]),
        "fused_conv2d_NCHWc_OIHWo_opencl_relu_tir_tanh": (
            ["global.texture-weight", "global"],
            ["global"],
        ),
        "te_layout_transform2": (["global"], ["global"]),
    }
    verify(Input, Expected)


def test_conv2d_add_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"),
            w: R.Tensor((4, 16, 3, 3), "float32"),
            bias: R.Tensor((2, 4, 26, 26), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.add(gv, bias)
                R.output(gv2)
            return gv2

    Expected = {
        "te_layout_transform": (["global"], ["global.texture-weight"]),
        "te_layout_transform1": (["global"], ["global"]),
        "te_layout_transform2": (["global"], ["global.texture-weight"]),
        "fused_conv2d_NCHWc_OIHWo_opencl_add": (
            ["global.texture-weight", "global", "global.texture-weight"],
            ["global"],
        ),
        "te_layout_transform3": (["global"], ["global"]),
    }
    verify(Input, Expected)


def test_conv2d_fma_relu_conv2d_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 4, 28, 28), "float32"),
            w: R.Tensor((4, 4, 3, 3), "float32"),
            scale: R.Tensor((2, 4, 26, 26), dtype="float32"),
            bias: R.Tensor((2, 4, 26, 26), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.ewise_fma(gv, scale, bias)
                gv3: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv2)
                gv4: R.Tensor((2, 4, 24, 24), "float32") = R.nn.conv2d(gv3, w, out_dtype="float32")
                R.output(gv4)
            return gv4

    Expected = {
        "te_layout_transform": (["global"], ["global.texture-weight"]),
        "te_layout_transform1": (["global"], ["global"]),
        "conv2d_NCHWc_OIHWo_opencl": (
            ["global.texture-weight", "global"],
            ["global"],
        ),
        "te_layout_transform2": (["global"], ["global"]),
        "relu": (["global"], ["global"]),
        "te_layout_transform3": (["global"], ["global.texture-weight"]),
        "conv2d_NCHWc_OIHWo1_opencl": (
            ["global.texture-weight", "global"],
            ["global"],
        ),
        "te_layout_transform4": (["global"], ["global"]),
    }
    verify(Input, Expected)


def test_conv2d_sum_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4), "float32") = R.sum(gv, axis=[2, 3])
                R.output(gv2)
            return gv2

    Expected = {
        "te_layout_transform": (["global"], ["global.texture-weight"]),
        "te_layout_transform1": (["global"], ["global"]),
        "conv2d_NCHWc_OIHWo_opencl": (
            ["global.texture-weight", "global"],
            ["global"],
        ),
        "sum": (["global"], ["global"]),
        "te_layout_transform2": (["global"], ["global"]),
    }
    verify(Input, Expected)


def test_conv2d_sum_keepdims_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 1, 1), "float32") = R.sum(gv, axis=[2, 3], keepdims=True)
                R.output(gv2)
            return gv2

    Expected = {
        "te_layout_transform": (["global"], ["global.texture-weight"]),
        "te_layout_transform1": (["global"], ["global"]),
        "conv2d_NCHWc_OIHWo_opencl": (
            ["global.texture-weight", "global"],
            ["global"],
        ),
        "sum": (["global"], ["global"]),
        "te_layout_transform2": (["global"], ["global"]),
    }
    verify(Input, Expected)


def test_conv2d_sum_reduce_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 26), "float32") = R.sum(gv, axis=[1, 2])
                R.output(gv2)
            return gv2

    Expected = {
        "te_layout_transform": (["global"], ["global.texture-weight"]),
        "te_layout_transform1": (["global"], ["global"]),
        "conv2d_NCHWc_OIHWo_opencl": (
            ["global.texture-weight", "global"],
            ["global"],
        ),
        "sum": (["global"], ["global"]),
        "te_layout_transform2": (["global"], ["global"]),
    }
    verify(Input, Expected)


def test_conv2d_transpose_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((26, 26, 4, 2), "float32") = R.permute_dims(gv, axes=[3, 2, 1, 0])
                R.output(gv2)
            return gv2

    Expected = {
        "te_layout_transform": (["global"], ["global.texture-weight"]),
        "te_layout_transform1": (["global"], ["global"]),
        "conv2d_NCHWc_OIHWo_opencl": (
            ["global.texture-weight", "global"],
            ["global"],
        ),
        "te_layout_transform2": (["global"], ["global"]),
        "transpose": (["global"], ["global"]),
    }
    verify(Input, Expected)


def test_conv2d_expand_dims_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=6):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 1, 4, 1, 26, 26), "float32") = R.expand_dims(gv, axis=(-3, 1))
                R.output(gv2)
            return gv2

    Expected = {
        "te_layout_transform": (["global"], ["global.texture-weight"]),
        "te_layout_transform1": (["global"], ["global"]),
        "conv2d_NCHWc_OIHWo_opencl": (
            ["global.texture-weight", "global"],
            ["global"],
        ),
        "te_layout_transform2": (["global"], ["global"]),
        "expand_dims": (["global"], ["global"]),
    }
    verify(Input, Expected)


def test_conv2d_squeeze_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((1, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=3):
            with R.dataflow():
                gv: R.Tensor((1, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((4, 26, 26), "float32") = R.squeeze(gv, axis=[0])
                R.output(gv2)
            return gv2

    Expected = {
        "te_layout_transform": (["global"], ["global.texture-weight"]),
        "te_layout_transform1": (["global"], ["global"]),
        "conv2d_NCHWc_OIHWo_opencl": (
            ["global.texture-weight", "global"],
            ["global"],
        ),
        "te_layout_transform2": (["global"], ["global"]),
        "squeeze": (["global"], ["global"]),
    }
    verify(Input, Expected)


def test_conv2d_strided_slice_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 2, 9, 7), dtype="float32") = R.strided_slice(
                    gv, begin=[0, 0, 0], end=[4, 26, 26], strides=[2, 3, 4], axes=[1, 2, 3]
                )
                R.output(gv2)
            return gv2

    Expected = {
        "te_layout_transform": (["global"], ["global.texture-weight"]),
        "te_layout_transform1": (["global"], ["global"]),
        "conv2d_NCHWc_OIHWo_opencl": (
            ["global.texture-weight", "global"],
            ["global"],
        ),
        "te_layout_transform2": (["global"], ["global"]),
        "strided_slice": (["global"], ["global"]),
    }
    verify(Input, Expected)


def test_conv2d_relu_concat_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
                gv3: R.Tensor((2, 8, 26, 26), "float32") = R.concat((gv, gv2), axis=1)
                R.output(gv3)
            return gv3

    Expected = {
        "te_layout_transform": (["global"], ["global.texture-weight"]),
        "te_layout_transform1": (["global"], ["global"]),
        "conv2d_NCHWc_OIHWo_opencl": (
            ["global.texture-weight", "global"],
            ["global.texture-weight"],
        ),
        "fused_relu_concatenate": (["global.texture-weight"], ["global"]),
        "te_layout_transform2": (["global"], ["global"]),
    }
    verify(Input, Expected)


def test_conv2d_relu_concat_split_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
                gv3: R.Tensor((2, 8, 26, 26), "float32") = R.concat((gv, gv2), axis=1)
                gv4 = R.split(gv3, indices_or_sections=2, axis=1)
                R.output(gv4)
            return gv4

    Expected = {
        "te_layout_transform": (["global"], ["global.texture-weight"]),
        "te_layout_transform1": (["global"], ["global"]),
        "conv2d_NCHWc_OIHWo_opencl": (
            ["global.texture-weight", "global"],
            ["global.texture-weight"],
        ),
        "fused_relu_concatenate_split": (["global.texture-weight"], ["global", "global"]),
        "te_layout_transform2": (["global"], ["global"]),
        "te_layout_transform3": (["global"], ["global"]),
    }
    verify(Input, Expected)


def test_conv2d_relu_concat_split_transpose_concat_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
                gv3: R.Tensor((2, 8, 26, 26), "float32") = R.concat((gv, gv2), axis=1)
                gv4 = R.split(gv3, indices_or_sections=2, axis=1)
                gv5: R.Tensor((26, 26, 4, 2), "float32") = R.permute_dims(gv4[0], axes=[3, 2, 1, 0])
                gv6: R.Tensor((26, 26, 4, 2), "float32") = R.permute_dims(gv4[1], axes=[3, 2, 1, 0])
                gv7: R.Tensor((26, 26, 8, 2), "float32") = R.concat((gv5, gv6), axis=2)
                R.output(gv7)
            return gv7

    Expected = {
        "te_layout_transform": (["global"], ["global.texture-weight"]),
        "te_layout_transform1": (["global"], ["global"]),
        "conv2d_NCHWc_OIHWo_opencl": (
            ["global.texture-weight", "global"],
            ["global.texture-weight"],
        ),
        "fused_relu_concatenate_split": (["global.texture-weight"], ["global", "global"]),
        "te_layout_transform2": (["global"], ["global"]),
        "te_layout_transform3": (["global"], ["global"]),
        "fused_transpose_transpose_concatenate1": (["global", "global"], ["global"]),
    }
    verify(Input, Expected)


def test_conv2d_maxpool2d_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2 = R.nn.max_pool2d(
                    gv,
                    pool_size=[2, 2],
                    strides=[2, 2],
                    padding=[0, 0],
                    layout="NCHW",
                    out_layout="NCHW",
                )
                R.output(gv2)
            return gv2

    Expected = {
        "te_layout_transform": (["global"], ["global.texture-weight"]),
        "te_layout_transform1": (["global"], ["global"]),
        "conv2d_NCHWc_OIHWo_opencl": (
            ["global.texture-weight", "global"],
            ["global.texture-weight"],
        ),
        "max_pool2d_opencl": (["global.texture-weight"], ["global"]),
        "te_layout_transform2": (["global"], ["global"]),
    }
    verify(Input, Expected)


def test_conv2d_avgpool2d_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2 = R.nn.adaptive_avg_pool2d(gv, output_size=[13, 13], layout="NCHW")
                R.output(gv2)
            return gv2

    Expected = {
        "te_layout_transform": (["global"], ["global.texture-weight"]),
        "te_layout_transform1": (["global"], ["global"]),
        "conv2d_NCHWc_OIHWo_opencl": (
            ["global.texture-weight", "global"],
            ["global.texture-weight"],
        ),
        "adaptive_avg_pool2d_opencl": (["global.texture-weight"], ["global"]),
        "te_layout_transform2": (["global"], ["global"]),
    }
    verify(Input, Expected)


def test_conv2d_softmax_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2 = R.nn.softmax(gv, axis=1)
                R.output(gv2)
            return gv2

    Expected = {
        "te_layout_transform": (["global"], ["global.texture-weight"]),
        "te_layout_transform1": (["global"], ["global"]),
        "conv2d_NCHWc_OIHWo_opencl": (
            ["global.texture-weight", "global"],
            ["global"],
        ),
        "te_layout_transform2": (["global"], ["global"]),
        "softmax": (["global"], ["global"]),
    }
    verify(Input, Expected)


def test_conv2d_layernorm_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"),
            w: R.Tensor((4, 16, 3, 3), "float32"),
            gamma: R.Tensor((26, 26), dtype="float32"),
            beta: R.Tensor((26, 26), dtype="float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.nn.layer_norm(
                    gv, gamma, beta, axes=[-2, -1]
                )
                R.output(gv2)
            return gv2

    Expected = {
        "te_layout_transform": (["global"], ["global.texture-weight"]),
        "te_layout_transform1": (["global"], ["global"]),
        "conv2d_NCHWc_OIHWo_opencl": (
            ["global.texture-weight", "global"],
            ["global"],
        ),
        "layer_norm": (["global", "global", "global"], ["global"]),
        "te_layout_transform2": (["global"], ["global"]),
    }
    verify(Input, Expected)


def test_binary_broadcast_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"),
            w: R.Tensor((4, 16, 3, 3), "float32"),
            bias: R.Tensor((26, 26), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.add(gv, bias)
                R.output(gv2)
            return gv2

    Expected = {
        "te_layout_transform": (["global"], ["global.texture-weight"]),
        "te_layout_transform1": (["global"], ["global"]),
        "conv2d_NCHWc_OIHWo_opencl": (
            ["global.texture-weight", "global"],
            ["global"],
        ),
        "te_layout_transform2": (["global"], ["global"]),
        "add": (["global", "global"], ["global"]),
    }
    verify(Input, Expected)


def test_binary_ewise_scalar_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.add(gv, R.const(1, "float32"))
                R.output(gv2)
            return gv2

    Expected = {
        "te_layout_transform": (["global"], ["global.texture-weight"]),
        "te_layout_transform1": (["global"], ["global"]),
        "fused_conv2d_NCHWc_OIHWo_opencl_add": (
            ["global.texture-weight", "global"],
            ["global"],
        ),
        "te_layout_transform2": (["global"], ["global"]),
    }
    verify(Input, Expected)


def test_residual_block():
    r"""
    - some kind of residual block followed by convolution to have texture after residual block
    - scalar data type verification which should be mapped to global memory scope
        layout_transform (NCHW->NCHW4c)
                  |                      <- buffer
                conv2d (1)                  <- to get textures as output
               /         \
            conv2d (2)    |
                 \       /
                    add                     <- add should be fused into conv2d (2)
                multiply to scalar          <- buffer to the input of multiply scalar value
                    relu
                     |                      <- texture in intermediate tensor
                  conv2d (3)
                   relu
                     |                      <- buffer
               layout_transform (NCHW4c->NCHW)
    """

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 32, 40, 40), "float32"),
            w1: R.Tensor((32, 32, 2, 2), "float32"),
            w2: R.Tensor((32, 32, 1, 1), "float32"),
            w3: R.Tensor((32, 32, 2, 2), "float32"),
            bias: R.Tensor((1, 32, 1, 1), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv = R.nn.conv2d(x, w1, strides=[2, 2], out_dtype="float32")
                gv1 = R.add(gv, bias)
                gv2 = R.nn.relu(gv1)
                gv3 = R.nn.conv2d(gv2, w2, strides=[1, 1], out_dtype="float32")
                bias_1 = R.multiply(bias, R.const(0.15, "float32"))
                gv4 = R.add(gv3, bias_1)
                gv5 = R.nn.relu(gv4)
                gv6 = R.nn.conv2d(gv5, w3, strides=[2, 2], out_dtype="float32")
                gv7 = R.nn.relu(gv6)
                R.output(gv7)
            return gv7

    Expected = {
        "te_layout_transform": (["global"], ["global.texture-weight"]),
        "te_layout_transform1": (["global"], ["global"]),
        "te_layout_transform2": (["global"], ["global.texture-weight"]),
        "fused_conv2d_NCHWc_OIHWo_opencl_add_relu": (
            ["global.texture-weight", "global", "global.texture-weight"],
            ["global.texture-weight"],
        ),
        "te_layout_transform3": (["global"], ["global"]),
        "multiply": (["global"], ["global"]),
        "fused_conv2d_NCHWc_OIHWo1_opencl_add_relu": (
            ["global.texture-weight", "global", "global.texture-weight"],
            ["global.texture-weight"],
        ),
        "fused_conv2d_NCHWc_OIHWo2_opencl_relu1": (
            ["global.texture-weight", "global"],
            ["global"],
        ),
        "te_layout_transform4": (["global"], ["global"]),
    }
    verify(Input, Expected)


def test_conv2d_conv2d_fallback_to_buffer_conv2d():
    r"""
        layout_transform (NCHW->NCHW4c)
                  |                      <- texture
                conv2d (1)               <- textures as output
               /         \
            conv2d (2)    conv2d (3)     <- conv2d (2) emits texture, conv2d (3) emits buffer
                 \       /               <- concat shouldn't support textures here
                concatenation
                     |                   <- buffer
               layout_transform (NCHW4c->NCHW)
    """

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 32, 40, 40), "float32"),
            w1: R.Tensor((96, 32, 2, 2), "float32"),
            w2: R.Tensor((32, 96, 2, 2), "float32"),
            w3: R.Tensor((5, 96, 2, 2), "float32"),
            bias1: R.Tensor((1, 96, 1, 1), "float32"),
            bias2: R.Tensor((1, 32, 1, 1), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv = R.nn.conv2d(x, w1, strides=[2, 2], out_dtype="float32")
                gv1 = R.add(gv, bias1)
                gv2 = R.nn.relu(gv1)
                gv3 = R.nn.conv2d(gv2, w2, strides=[2, 2], out_dtype="float32")
                gv4 = R.add(gv3, bias2)
                gv5 = R.nn.relu(gv4)
                gv6 = R.nn.conv2d(gv2, w3, strides=[2, 2], out_dtype="float32")
                gv7 = R.concat((gv3, gv6), axis=1)
                R.output(gv7)
            return gv7

    Expected = {
        "te_layout_transform": (["global"], ["global.texture-weight"]),
        "te_layout_transform1": (["global"], ["global"]),
        "te_layout_transform2": (["global"], ["global.texture-weight"]),
        "fused_conv2d_NCHWc_OIHWo_opencl_add_relu": (
            ["global.texture-weight", "global", "global.texture-weight"],
            ["global"],
        ),
        "te_layout_transform3": (["global"], ["global"]),
        "conv2d_NCHWc_OIHWo1_opencl": (
            ["global.texture-weight", "global"],
            ["global"],
        ),
        "te_layout_transform4": (["global"], ["global"]),
        "conv2d": (["global", "global"], ["global"]),
        "te_layout_transform5": (["global"], ["global"]),
        "concatenate": (["global", "global"], ["global"]),
    }
    verify(Input, Expected)


def test_conv2d_conv2d_conv2d_concat():
    r"""
        layout_transform (NCHW->NCHW4c)
                  |                      <- texture
                conv2d (1)               <- textures as output
               /         \
            conv2d (2)    conv2d (3)
                 \       /               <- concat does support textures here
                concatenation
                     |                   <- buffer
               layout_transform (NCHW4c->NCHW)
    """

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 32, 40, 40), "float32"),
            w1: R.Tensor((96, 32, 2, 2), "float32"),
            w2: R.Tensor((32, 96, 2, 2), "float32"),
            w3: R.Tensor((8, 96, 2, 2), "float32"),
            bias1: R.Tensor((1, 96, 1, 1), "float32"),
            bias2: R.Tensor((1, 32, 1, 1), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv = R.nn.conv2d(x, w1, strides=[2, 2], out_dtype="float32")
                gv1 = R.add(gv, bias1)
                gv2 = R.nn.relu(gv1)
                gv3 = R.nn.conv2d(gv2, w2, strides=[2, 2], out_dtype="float32")
                gv4 = R.add(gv3, bias2)
                gv5 = R.nn.relu(gv4)
                gv6 = R.nn.conv2d(gv2, w3, strides=[2, 2], out_dtype="float32")
                gv7 = R.concat((gv3, gv6), axis=1)
                R.output(gv7)
            return gv7

    Expected = {
        "te_layout_transform": (["global"], ["global.texture-weight"]),
        "te_layout_transform1": (["global"], ["global"]),
        "te_layout_transform2": (["global"], ["global.texture-weight"]),
        "fused_conv2d_NCHWc_OIHWo_opencl_add_relu": (
            ["global.texture-weight", "global", "global.texture-weight"],
            ["global.texture-weight"],
        ),
        "te_layout_transform3": (["global"], ["global"]),
        "conv2d_NCHWc_OIHWo1_opencl": (
            ["global.texture-weight", "global"],
            ["global.texture-weight"],
        ),
        "te_layout_transform4": (["global"], ["global"]),
        "conv2d_NCHWc_OIHWo2_opencl": (
            ["global.texture-weight", "global"],
            ["global.texture-weight"],
        ),
        "concatenate": (["global.texture-weight", "global.texture-weight"], ["global"]),
        "te_layout_transform5": (["global"], ["global"]),
    }
    verify(Input, Expected)


def test_pooling_branching_texture_params():
    r"""
    Verification of the pooling and many branches having textures
                layout_transform (NCHW->NCHW4c)
                         |                        <- texture
                      conv2d (0)                  <- to get textures
                         |                        <- textures
                     pooling
               /           \           \          <- textures
            conv2d (1)    conv2d (2)    conv2d (3)
                \             /           |
                     add                  |       <- to have  the only one output, will be fused
                      \                  /
                            add                  <- to have  the only one output, will be fused
                             |                   <- buffer
                    layout_transform (NCHW4c->NCHW)
    """

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 32, 40, 40), "float32"),
            w1: R.Tensor((32, 32, 1, 1), "float32"),
            w2: R.Tensor((32, 32, 2, 2), "float32"),
            w3: R.Tensor((32, 32, 1, 1), "float32"),
            w4: R.Tensor((32, 32, 2, 2), "float32"),
            bias1: R.Tensor((1, 32, 1, 1), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv = R.nn.conv2d(x, w1, strides=[1, 1], out_dtype="float32")
                gv1 = R.nn.max_pool2d(gv, pool_size=[2, 2], strides=[2, 2])
                gv2 = R.nn.conv2d(
                    gv1, w2, padding=[0, 0, 1, 1], strides=[1, 1], out_dtype="float32"
                )
                gv3 = R.add(gv2, bias1)
                gv4 = R.nn.relu(gv3)
                gv5 = R.nn.conv2d(
                    gv1, w3, padding=[0, 0, 0, 0], strides=[1, 1], out_dtype="float32"
                )
                gv6 = R.nn.conv2d(
                    gv1, w4, padding=[0, 1, 1, 0], strides=[1, 1], out_dtype="float32"
                )
                gv7 = R.nn.relu(gv6)
                gv8 = R.add(gv2, gv5)
                gv9 = R.add(gv8, gv6)
                R.output(gv9)
            return gv9

    Expected = {
        "te_layout_transform": (["global"], ["global.texture-weight"]),
        "te_layout_transform1": (["global"], ["global"]),
        "conv2d_NCHWc_OIHWo_opencl": (
            ["global.texture-weight", "global"],
            ["global.texture-weight"],
        ),
        "max_pool2d_opencl": (["global.texture-weight"], ["global.texture-weight"]),
        "te_layout_transform2": (["global"], ["global"]),
        "conv2d_NCHWc_OIHWo2_opencl": (
            ["global.texture-weight", "global"],
            ["global.texture-weight"],
        ),
        "fused_conv2d_NCHWc_OIHWo1_opencl_add": (
            ["global.texture-weight", "global", "global.texture-weight"],
            ["global.texture-weight"],
        ),
        "fused_conv2d_NCHWc_OIHWo3_opencl_add": (
            ["global.texture-weight", "global", "global.texture-weight"],
            ["global"],
        ),
        "te_layout_transform3": (["global"], ["global"]),
    }
    verify(Input, Expected)


def test_injective_inputs1():
    r"""
                                     Input
                               /                   \
                            /                      |
                         |                        /
                      conv2d (1)                 /
                         |                      /
                      conv2d (2)       mean    /
                  /         \                 /
                 |           |      \        /
                 |           |       (3) add
                 |           |         |
                 |             \    /
                 \                mul
                  \            /
                        add

    """

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((1, 4, 40, 40), "float32"),
            w1: R.Tensor((4, 4, 3, 3), "float32"),
            w2: R.Tensor((4, 4, 3, 3), "float32"),
            w3: R.Tensor((4, 4, 3, 3), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                mean = R.mean(x, axis=1, keepdims=True)
                conv1 = R.nn.conv2d(
                    x, w1, padding=[1, 1, 1, 1], strides=[1, 1], out_dtype="float32"
                )
                conv2 = R.nn.conv2d(
                    conv1, w2, padding=[1, 1, 1, 1], strides=[1, 1], out_dtype="float32"
                )
                ad3 = R.add(conv1, conv2)
                ad1 = R.add(mean, conv1)
                ad2 = R.multiply(ad1, conv1)
                gv = R.add(ad3, ad2)
                R.output(gv)
            return gv

    Expected = {
        "te_layout_transform": (["global"], ["global.texture-weight"]),
        "te_layout_transform1": (["global"], ["global"]),
        "conv2d_NCHWc_OIHWo_opencl": (
            ["global.texture-weight", "global"],
            ["global"],
        ),
        "te_layout_transform2": (["global"], ["global"]),
        "fused_mean_add1": (["global", "global"], ["global"]),
        "fused_conv2d_NCHWc_OIHWo_opencl_add_multiply_add": (
            [
                "global.texture-weight",
                "global",
                "global.texture-weight",
                "global.texture-weight",
                "global.texture-weight",
            ],
            ["global"],
        ),
    }
    verify(Input, Expected)


def test_injective_nwo_inputs2():
    r"""
                                     Input
                               /             \
                         |                    \
                      conv2d                   \
                         |                     /
                      conv2d           mean    /
                  /         \                 /
                add         |   \             |
                 |           |    \           |
                 |           |      \        /
                 |           |       (3) add
                 |           |         |
                 |            \       /
                 |             \    /
                 \                mul
                  \            /
                        add

    """

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((1, 4, 40, 40), "float32"),
            w1: R.Tensor((4, 4, 3, 3), "float32"),
            w2: R.Tensor((4, 4, 3, 3), "float32"),
            w3: R.Tensor((4, 4, 3, 3), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                mean = R.mean(x, axis=1, keepdims=True)
                conv1 = R.nn.conv2d(
                    x, w1, padding=[1, 1, 1, 1], strides=[1, 1], out_dtype="float32"
                )
                conv2 = R.nn.conv2d(
                    conv1, w2, padding=[1, 1, 1, 1], strides=[1, 1], out_dtype="float32"
                )
                ad3 = R.add(conv1, conv2)
                ad1 = R.add(mean, conv1)
                ad2 = R.multiply(ad1, conv2)
                gv = R.add(ad2, ad3)
                R.output(gv)
            return gv

    Expected = {
        "te_layout_transform": (["global"], ["global.texture-weight"]),
        "te_layout_transform1": (["global"], ["global"]),
        "conv2d_NCHWc_OIHWo_opencl": (
            ["global.texture-weight", "global"],
            ["global"],
        ),
        "te_layout_transform2": (["global"], ["global"]),
        "fused_mean_add1": (["global", "global"], ["global"]),
        "fused_conv2d_NCHWc_OIHWo_opencl_add_multiply_add": (
            [
                "global.texture-weight",
                "global",
                "global.texture-weight",
                "global.texture-weight",
            ],
            ["global"],
        ),
    }
    verify(Input, Expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
