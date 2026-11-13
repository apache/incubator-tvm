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
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel
"""Pattern table and codegen for CoreML"""

import os
import shutil

import tvm_ffi

import tvm
import tvm_tirx
from tvm.contrib import coreml_runtime
from tvm.relax import transform
from tvm.relax.dpl.pattern import is_op, wildcard
from tvm.relax.expr import BindingBlock, Function, SeqExpr, Var, VarBinding
from tvm.relax.transform import PatternCheckContext
from tvm.relax.type import TensorType
from tvm_tirx.ir import Call, GenericConst, PrimType
from tvm_tirx.support.xcode import compile_coreml

from ...expr_functor import PyExprVisitor, visitor
from ..pattern_registry import get_patterns_with_prefix, register_patterns
from ..patterns import make_matmul_pattern


def _check_default(context: PatternCheckContext) -> bool:
    return True


def default_binary_patterns(op_name: str):
    """
    Returns a list of binary op patterns in coreML BYOC backend.
    """

    def _make_binary_pattern():
        lhs = wildcard()
        rhs = wildcard()
        out = is_op("relax." + op_name)(lhs, rhs)
        annotations = {"lhs": lhs, "rhs": rhs, "root": out}
        return out, annotations

    def _binary_pattern(pattern_name):
        return (pattern_name, *_make_binary_pattern(), _check_default)

    return [_binary_pattern("coreml." + op_name)]


def default_unary_patterns(op_name: str):
    """
    Returns a list of unary op patterns in coreML BYOC backend.
    """

    def _make_unary_pattern():
        lhs = wildcard()
        out = is_op("relax." + op_name)(lhs)
        annotations = {"lhs": lhs, "root": out}
        return out, annotations

    def _unary_pattern(pattern_name):
        return (pattern_name, *_make_unary_pattern(), _check_default)

    return [_unary_pattern("coreml." + op_name)]


def conv2d_patterns():
    """
    Returns a list of conv2d patterns in coreML BYOC backend.
    """

    def _make_conv2d_pattern():
        lhs = wildcard()
        rhs = wildcard()
        out = is_op("relax.nn.conv2d")(lhs, rhs)
        annotations = {"lhs": lhs, "rhs": rhs, "root": out}
        return out, annotations

    def _conv2d_pattern(pattern_name):
        return (pattern_name, *_make_conv2d_pattern(), _check_default)

    return [_conv2d_pattern("coreml.nn.conv2d")]


def matmul_patterns():
    """
    Returns a list of all matmul patterns in coreML BYOC backend.
    """

    def _matmul_pattern(pattern_name):
        return (
            pattern_name,
            *make_matmul_pattern(),
            _check_default,
        )

    return [_matmul_pattern("coreml.matmul")]


def clip_patterns():
    """
    Returns a list of clip patterns in coreML BYOC backend.
    """

    def _make_clip_pattern():
        arg0 = wildcard()
        arg1 = wildcard()
        arg2 = wildcard()
        out = is_op("relax.clip")(arg0, arg1, arg2)
        annotations = {"arg0": arg0, "arg1": arg1, "arg2": arg2, "root": out}
        return out, annotations

    def _conv2d_pattern(pattern_name):
        return (pattern_name, *_make_clip_pattern(), _check_default)

    return [_conv2d_pattern("coreml.clip")]


register_patterns(
    [
        *default_binary_patterns(op_name="add"),
        *default_binary_patterns(op_name="multiply"),
        *default_unary_patterns(op_name="nn.softmax"),
        *default_unary_patterns(op_name="nn.relu"),
        *default_unary_patterns(op_name="expand_dims"),
        *default_unary_patterns(op_name="nn.avg_pool2d"),
        *default_unary_patterns(op_name="nn.batch_flatten"),
        *conv2d_patterns(),
        *clip_patterns(),
        *matmul_patterns(),
    ]
)


def partition_for_coreml(mod):
    """
    Partition the input module into coreml-supported subgraphs.

    Parameters
    ----------
    mod: tvm_tirx.IRModule
        The IRModule to be partitioned.

    Returns
    -------
    mod: tvm_tirx.IRModule
        The resulting IRModule, containing partitioned subgraphs to be
        offloaded to the coreml backend.
    """

    patterns = get_patterns_with_prefix("coreml")
    mod = transform.CanonicalizeBindings()(mod)
    mod = transform.FuseOpsByPattern(patterns, bind_constants=True, annotate_codegen=False)(mod)
    mod = transform.MergeCompositeFunctions()(mod)
    return mod


# Codegen for coreml API reference:
# https://apple.github.io/coremltools/source/coremltools.models.neural_network.html
def _convert_add(builder, name, inputs, outputs, args, attrs):
    builder.add_elementwise(name=name, input_names=inputs, output_name=outputs[0], mode="ADD")


def _convert_multiply(builder, name, inputs, outputs, args, attrs):
    builder.add_elementwise(name=name, input_names=inputs, output_name=outputs[0], mode="MULTIPLY")


def _convert_matmul(builder, name, inputs, outputs, args, attrs):
    builder.add_batched_mat_mul(
        name=name,
        input_names=inputs,
        output_name=outputs[0],
    )


def _convert_clip(builder, name, inputs, outputs, args, attrs):
    builder.add_clip(
        name=name,
        input_name=inputs[0],
        output_name=outputs[0],
        min_value=inputs[1],
        max_value=inputs[2],
    )


def _convert_batch_flatten(builder, name, inputs, outputs, args, attrs):
    builder.add_flatten_to_2d(name=name, input_name=inputs[0], output_name=outputs[0])


def _convert_expand_dims(builder, name, inputs, outputs, args, attrs):
    axes = [int(v) for v in attrs["axis"]]
    builder.add_expand_dims(name=name, input_name=inputs[0], output_name=outputs[0], axes=axes)


def _convert_relu(builder, name, inputs, outputs, args, attrs):
    builder.add_activation(
        name=name, non_linearity="RELU", input_name=inputs[0], output_name=outputs[0]
    )


def _convert_softmax(builder, name, inputs, outputs, args, attrs):
    builder.add_softmax_nd(
        name=name, input_name=inputs[0], output_name=outputs[0], axis=int(attrs["axis"])
    )


def _convert_conv2d(builder, name, inputs, outputs, args, attrs):
    weight = args[1].value.numpy()
    oc, kc, kh, kw = weight.shape

    builder.add_convolution(
        name=name,
        kernel_channels=kc,
        output_channels=oc,
        height=kh,
        width=kw,
        stride_height=int(attrs["strides"][0]),
        stride_width=int(attrs["strides"][0]),
        border_mode="valid",
        groups=int(attrs["groups"]),
        W=weight,
        b=None,
        has_bias=False,
        input_name=inputs[0],
        output_name=outputs[0],
        dilation_factors=[int(v) for v in attrs["dilation"]],
        padding_top=int(attrs["padding"][0]),
        padding_bottom=int(attrs["padding"][2]),
        padding_left=int(attrs["padding"][1]),
        padding_right=int(attrs["padding"][3]),
    )


def _convert_avg_pool2d(builder, name, inputs, outputs, args, attrs):
    builder.add_pooling(
        name=name,
        height=1,
        width=1,
        stride_height=1,
        stride_width=1,
        layer_type="AVERAGE",
        padding_type="VALID",
        input_name=inputs[0],
        output_name=outputs[0],
    )


_convert_map = {
    "add": _convert_add,
    "multiply": _convert_multiply,
    "matmul": _convert_matmul,
    "clip": _convert_clip,
    "expand_dims": _convert_expand_dims,
    "nn.relu": _convert_relu,
    "nn.batch_flatten": _convert_batch_flatten,
    "nn.softmax": _convert_softmax,
    "nn.conv2d": _convert_conv2d,
    "nn.avg_pool2d": _convert_avg_pool2d,
}


@visitor
class CallNodeInfoCollector(PyExprVisitor):
    """
    Collect Expr, GenericConst and attributes in the inner function
    """

    def __init__(self, op_name):
        self.primvals = []
        self.attrs = []
        self.consts = []
        self.op_name = op_name

    def visit_call_(self, call: Call) -> None:
        self.attrs.append(call.attrs)
        for arg in call.args:
            if tvm_tirx.ir.is_prim_expr(arg):
                self.primvals.append(arg)
            if isinstance(arg, GenericConst):
                self.consts.append(arg)

    def collect(self, expr):
        self.visit_expr(expr)
        return self.primvals, self.attrs, self.consts


@visitor
class CodegenCoreML(PyExprVisitor):
    """
    A visitor to traverse subgraphs and build Core ML models.
    """

    def __init__(self, model_name, function):
        try:
            import coremltools
            from coremltools.models.neural_network import NeuralNetworkBuilder
        except ImportError as err:
            raise ImportError(
                "coremltools is required by the CoreML backend. "
                "Install it with: pip install coremltools"
            ) from err

        self.model_name = model_name
        self.function = function
        self.out_map = {}
        self.const_map = {}  # (buffer name, object)
        self.model_inputs_ = []
        self.buf_idx_ = 0

        getter = tvm_ffi.get_global_func("relax.analysis.get_var2val")
        assert getter, "Cannot find `relax.analysis.get_var2val` function."

        self.var2val = getter(function)
        self.cur_binding_var = None

        inputs = [
            (
                "",
                coremltools.models.datatypes.Array(
                    1,
                ),
            )
            for _ in self.function.params
        ]
        outputs = [
            (
                "",
                coremltools.models.datatypes.Array(
                    1,
                ),
            )
        ]
        self.builder = NeuralNetworkBuilder(inputs, outputs, disable_rank5_shape_mapping=True)

    def visit_function_(self, op) -> None:
        for var in op.params:
            name = var.name
            ty = var.ty
            if isinstance(ty, TensorType):
                shape = [int(v) for v in list(ty.shape)]
            elif isinstance(ty, PrimType):
                shape = []
            else:
                raise Exception("Currently not supported: ", type(ty))
            dtype = ty.dtype
            self.model_inputs_.append((name, shape, dtype))

        self.visit_expr(op.body)

    def visit_var_(self, var):
        self.out_map[var] = [var.name]
        prev_binding_var = self.cur_binding_var
        self.cur_binding_var = var
        if var in self.var2val:
            self.visit_expr(self.var2val[var])
        self.cur_binding_var = prev_binding_var

    def visit_call_(self, call: Call) -> None:
        assert isinstance(call.op, Var)
        assert call.op in self.var2val
        func = self.var2val[call.op]

        assert "Composite" in func.attrs, "Only composite functions are supported."
        composite_name = func.attrs["Composite"]

        # Get the op name and remove "relax." prefix.
        op_name = composite_name[7:]

        inputs = []
        args = []
        for arg in call.args:
            args.append(arg)
            super().visit_expr(arg)
            for out in self.out_map[arg]:
                inputs.append(out)

        primvals, attrs, consts = CallNodeInfoCollector(op_name).collect(func.body)
        for arg in primvals:
            args.append(arg)
            inputs.append(arg.value.value)

        for arg in consts:
            output = "buf_" + str(self.buf_idx_)
            self.builder.add_load_constant_nd(
                name=output,
                output_name=output,
                constant_value=arg.value.numpy(),
                shape=arg.value.shape,
            )
            self.buf_idx_ = self.buf_idx_ + 1
            self.out_map[arg] = [output]
            inputs.append(output)
            args.append(arg)

        layer_name = op_name + "_" + str(self.buf_idx_)

        assert op_name in _convert_map, f"{op_name} is not supported"
        outputs = ["buf_" + str(self.buf_idx_)]
        _convert_map[op_name](self.builder, layer_name, inputs, outputs, args, attrs[0])
        self.buf_idx_ = self.buf_idx_ + 1
        self.out_map[self.cur_binding_var] = outputs

    def visit_var_binding_(self, binding: VarBinding) -> None:
        # Visit var of the last binding
        self.visit_expr(binding.var)

    def visit_binding_block_(self, block: BindingBlock) -> None:
        # We only visit the last VarBinding to retrieve
        # target composite function
        self.visit_binding(block.bindings[-1])

    def visit_seq_expr_(self, op: SeqExpr) -> None:
        for bb in op.blocks:
            self.visit_binding_block_(bb)

    def serialize(self, func: Function):
        self.visit_expr(func)

    def compile(self, out_dir):
        """
        Build a Core ML model and compile it with Xcode toolchain.
        """
        import coremltools
        from coremltools.proto.Model_pb2 import ArrayFeatureType

        FEATURE_TYPE_MAP = {
            "float32": ArrayFeatureType.FLOAT32,
            "float64": ArrayFeatureType.DOUBLE,
            "int32": ArrayFeatureType.INT32,
        }
        input_names, input_dims, input_dtypes = zip(*self.model_inputs_)
        self.builder.set_input(input_names, input_dims)

        for i, dtype in enumerate(input_dtypes):
            assert dtype in FEATURE_TYPE_MAP
            input_desc = self.builder.spec.description.input
            input_desc[i].type.multiArrayType.dataType = FEATURE_TYPE_MAP[dtype]

        output_dim = [int(n) for n in self.function.ty.ret.shape]

        last_binding_var = self.function.body.blocks[0].bindings[-1].var
        self.builder.set_output(self.out_map[last_binding_var], [output_dim])

        for i, dtype in enumerate([self.function.ty.ret.dtype]):
            assert dtype in FEATURE_TYPE_MAP
            output_desc = self.builder.spec.description.output
            output_desc[i].type.multiArrayType.dataType = FEATURE_TYPE_MAP[dtype]

        model = coremltools.models.MLModel(self.builder.spec)
        compile_coreml(model, self.model_name, out_dir)


@tvm_ffi.register_global_func("relax.ext.coreml")
def coreml_compiler(funcs, options, constant_names):
    """
    Create a CoreML runtime from a Relax module.
    """
    compiled_funcs = []
    for func in funcs:
        assert isinstance(func, tvm.relax.Function)
        model_dir = os.getcwd() + "/tmp/"
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        name = str(func.attrs.global_symbol)
        builder = CodegenCoreML(name, func)
        builder.serialize(func)

        mlmodelc_path = f"{model_dir}/{name}.mlmodelc"

        if os.path.exists(mlmodelc_path):
            shutil.rmtree(mlmodelc_path)

        builder.compile(model_dir)
        dev = tvm_tirx.cpu(0)
        compiled_funcs.append(coreml_runtime.create(name, mlmodelc_path, dev).module)
    return compiled_funcs
