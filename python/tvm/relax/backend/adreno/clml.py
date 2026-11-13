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
# pylint: disable=invalid-name, unused-argument, pointless-exception-statement
"""Pattern table for CLML backend"""

import tvm_ffi

import tvm_tirx
from tvm import relax
from tvm.relax import transform
from tvm.relax.dpl.pattern import (
    GlobalVarPattern,
    TuplePattern,
    is_const,
    is_op,
    is_tuple_get_item,
    wildcard,
)
from tvm.relax.expr import TupleGetItem, VarBinding
from tvm.relax.expr_functor import PyExprMutator, mutator
from tvm.relax.transform import PatternCheckContext
from tvm_tirx import IRModule
from tvm_tirx.ir.transform import PassContext, module_pass

from ..pattern_registry import register_patterns


def _dtype_str(dtype):
    return str(dtype.dtype) if isinstance(dtype, tvm_tirx.ir.PrimType) else str(dtype)


@mutator
class AppendReshapeToBNRewriter(PyExprMutator):
    """
    Append Reshape Operator to BatchNorm Pass Rewriter Pass

    - Automatically appends a reshape operation after BatchNorm operators
    - Resolves fusion issues for custom backends where BatchNorm output
      might explicitly access the first elment of the Tuple

    Algo:
    Identifies BatchNorm operators in the computational graph
    When BatchNorm's first output is accessed via TupleGetItem
    Automatically inserts a reshape operation to match input shape

    """

    def __init__(self, mod):
        super().__init__(mod)
        self.bn_vars = {}

    def visit_tuple_getitem_(self, op: TupleGetItem):
        tuple_value = op.tuple_value
        reshape_op = tvm_tirx.ir.Op.get("relax.reshape")

        if isinstance(tuple_value, relax.Var) and tuple_value in self.bn_vars:
            bn_call = self.bn_vars[tuple_value]
            if op.index == 0:
                bn_out = relax.TupleGetItem(bn_call, 0)
                input_shape = bn_call.args[0].ty.shape
                return relax.Call(reshape_op, [bn_out, input_shape])

        return super().visit_tuple_getitem_(op)

    def visit_var_binding_(self, binding: VarBinding):
        if isinstance(binding.value, relax.Call) and binding.value.op.name == "relax.nn.batch_norm":
            self.bn_vars[binding.var] = binding.value
        return super().visit_var_binding_(binding)


@transform.function_pass(opt_level=0, name="AppendReshapeToBN")
class AppendReshapeToBNRewriterPass:
    def transform_function(
        self, func: relax.Function, mod: IRModule, _ctx: tvm_tirx.ir.transform.PassContext
    ) -> relax.Function:
        updated_func = AppendReshapeToBNRewriter(mod).visit_expr(func)
        updated_func = relax.analysis.remove_all_unused(updated_func)
        return updated_func


def clml_sdk_version():
    """Utility function to get clml version.

    Probes the FFI registry for the OpenCLML version registered by the
    CLML backend at build time.  Returns 2 when CLML is not present.
    """
    # Registry: "relax.get_openclml_version" — returns the CLML SDK version
    # that TVM was built against; registered unconditionally in codegen.cc.
    # Grep hint: grep -rn 'relax.get_openclml_version' src/
    get_version = tvm_ffi.get_global_func("relax.get_openclml_version", allow_missing=True)
    if get_version is None:
        return 2
    return int(get_version())


def is_clml_runtime_enabled():
    """Check if the CLML graph runtime is present.

    Returns
    -------
    ret: bool
        True if present, False if not.
    """
    check_enabled = tvm_ffi.get_global_func("relax.op.is_openclml_runtime_enabled", True)
    if check_enabled:
        return check_enabled()
    return False


def _check_default(context: PatternCheckContext) -> bool:
    return True


def clml_pattern_table():
    """Get the CLML pattern table."""

    def _check_conv2d(context: PatternCheckContext) -> bool:
        if "root" in context.annotated_expr:
            root_call = context.annotated_expr["root"]
            if root_call.op.name == "relax.nn.conv2d":
                input_layout = root_call.attrs.data_layout
                weight_layout = root_call.attrs.kernel_layout
                if input_layout != "NCHW" or weight_layout != "OIHW":
                    return False
            if root_call.op.name == "relax.nn.conv2d_transpose":
                input_layout = root_call.attrs.data_layout
                weight_layout = root_call.attrs.kernel_layout
                if input_layout != "NCHW" or weight_layout != "OIHW":
                    return False

        if "data" in context.annotated_expr:
            input_expr = context.annotated_expr["data"]
            input_dtype = _dtype_str(input_expr.ty.dtype)
            if input_dtype not in ["float32", "float16"]:
                return False

        if "weight" in context.annotated_expr:
            weight_expr = context.annotated_expr["weight"]
            weight_dtype = _dtype_str(weight_expr.ty.dtype)
            if weight_dtype not in ["float32", "float16"]:
                return False

        return True

    def populate_patterns(patterns, name, op, annotations, *args):
        ret = {}
        for k, v in patterns.items():
            ret_ann = v["annotation"].copy()
            ret_ann.update(annotations)
            ret[name + "." + k] = {"pattern": op(v["pattern"], *args), "annotation": ret_ann.copy()}

        return ret

    def conv_pattern():
        """Create a convolution pattern."""
        data = wildcard()
        weight = wildcard()
        bias = is_const()
        bn_scale = is_const()
        bn_bias = is_const()
        bn_mean = is_const()
        bn_var = is_const()

        annotations = {
            "data": data,
            "weight": weight,
        }

        patterns = {}
        patterns["nn.conv2d"] = {
            "pattern": is_op("relax.nn.conv2d")(data, weight),
            "annotation": annotations.copy(),
        }

        pad_annotations = annotations.copy()
        patterns["pad.nn.conv2d"] = {
            "pattern": is_op("relax.nn.conv2d")(is_op("relax.nn.pad")(data), weight),
            "annotation": pad_annotations,
        }

        patterns["nn.conv2d_transpose"] = {
            "pattern": is_op("relax.nn.conv2d_transpose")(data, weight),
            "annotation": annotations.copy(),
        }
        patterns.update(
            populate_patterns(patterns, "bias", is_op("relax.add"), {"bias": bias}, bias)
        )
        patterns.update(
            populate_patterns(
                patterns,
                "bn",
                is_op("relax.nn.batch_norm"),
                {
                    "bn_scale": bn_scale,
                    "bn_bias": bn_bias,
                    "bn_mean": bn_mean,
                    "bn_var": bn_var,
                },
                bn_scale,
                bn_bias,
                bn_mean,
                bn_var,
            )
        )
        tuple_patterns = {}
        for k, v in patterns.items():
            tuple_annotation = v["annotation"].copy()
            tuple_patterns["tuple" + "." + k] = {
                "pattern": is_tuple_get_item(v["pattern"], 0),
                "annotation": tuple_annotation,
            }
        patterns.update(tuple_patterns)

        relu_patterns = populate_patterns(patterns, "relu", is_op("relax.nn.relu"), {})
        clip_patterns = populate_patterns(patterns, "clip", is_op("relax.clip"), {})
        patterns.update(relu_patterns)
        patterns.update(clip_patterns)

        conv_patterns = []
        for k, v in patterns.items():
            ret_annotations = v["annotation"]
            ret_annotations["root"] = v["pattern"]
            conv_patterns.append(
                ("openclml." + (k), v["pattern"], ret_annotations.copy(), _check_conv2d)
            )
        return conv_patterns[::-1]

    def _check_maxpool2d(context: PatternCheckContext) -> bool:
        root = context.annotated_expr.get("root")
        if root is None or not isinstance(root, relax.Call):
            return False

        if root.op.name != "relax.nn.max_pool2d":
            return False

        if "data" not in context.annotated_expr:
            return False

        data = context.annotated_expr["data"]
        input_shape = data.ty.shape

        if len(input_shape) != 4:
            return False

        if any(dim <= 0 for dim in input_shape):
            return False

        pool_size = root.attrs.pool_size
        if len(pool_size) != 2:
            return False
        if any(size <= 0 for size in pool_size):
            return False

        strides = root.attrs.strides
        if len(strides) != 2:
            return False
        if any(stride <= 0 for stride in strides):
            return False

        dilation = root.attrs.dilation
        if len(dilation) != 2:
            return False
        if any(d <= 0 for d in dilation):
            return False

        padding = root.attrs.padding
        if len(padding) != 4:
            return False
        if any(p < 0 for p in padding):
            return False

        return True

    def maxpool_pattern():
        """Create Pool Pattern"""
        data = wildcard()
        annotations = {
            "data": data,
        }
        patterns = {}
        patterns["nn.max_pool2d"] = {
            "pattern": is_op("relax.nn.max_pool2d")(data),
            "annotation": annotations.copy(),
        }

        pool_patterns = []
        for k, v in patterns.items():
            ret_annotations = v["annotation"]
            ret_annotations["root"] = v["pattern"]
            pool_patterns.append(
                ("openclml." + (k), v["pattern"], ret_annotations.copy(), _check_maxpool2d)
            )
        return pool_patterns

    def _check_avgpool2d(context: PatternCheckContext) -> bool:
        root = context.annotated_expr.get("root")
        if root is None or not isinstance(root, relax.Call):
            return False

        if root.op.name != "relax.nn.avg_pool2d":
            return False

        if "data" not in context.annotated_expr:
            return False

        data = context.annotated_expr["data"]
        input_shape = data.ty.shape

        if len(input_shape) != 4:
            return False

        if any(dim <= 0 for dim in input_shape):
            return False

        pool_size = root.attrs.pool_size
        if len(pool_size) != 2:
            return False
        if any(size <= 0 for size in pool_size):
            return False

        strides = root.attrs.strides
        if len(strides) != 2:
            return False
        if any(stride <= 0 for stride in strides):
            return False

        padding = root.attrs.padding
        if len(padding) != 4:
            return False
        if any(p < 0 for p in padding):
            return False

        return True

    def avgpool_pattern():
        data = wildcard()
        annotations = {
            "data": data,
        }
        patterns = {}
        patterns["nn.avg_pool2d"] = {
            "pattern": is_op("relax.nn.avg_pool2d")(data),
            "annotation": annotations.copy(),
        }

        pool_patterns = []
        for k, v in patterns.items():
            ret_annotations = v["annotation"]
            ret_annotations["root"] = v["pattern"]
            pool_patterns.append(
                ("openclml." + (k), v["pattern"], ret_annotations.copy(), _check_avgpool2d)
            )
        return pool_patterns

    def _check_global_avgpool(context: PatternCheckContext) -> bool:
        root = context.annotated_expr.get("root")
        if root is None or not isinstance(root, relax.Call):
            return False

        if root.op.name != "relax.mean":
            return False

        if "data" not in context.annotated_expr:
            return False

        data = context.annotated_expr["data"]
        input_shape = data.ty.shape

        if len(input_shape) != 4:
            return False

        if input_shape[1] <= 0 or input_shape[2] <= 0 or input_shape[3] <= 0:
            return False

        if not hasattr(root.attrs, "axis"):
            return False

        axis = root.attrs.axis
        if not (len(axis) == 2 and axis[0] == 2 and axis[1] == 3):
            return False

        return True

    def global_avgpool_pattern():
        """Create Pool Pattern"""
        data = wildcard()
        pattern = is_op("relax.mean")(data).has_attr({"axis": [2, 3]})

        annotations = {
            "data": data,
            "root": pattern,
        }

        return [
            ("openclml.nn.global_avg_pool2d", pattern, annotations, _check_global_avgpool),
        ]

    def _check_reshape(context: PatternCheckContext) -> bool:
        root = context.annotated_expr.get("root")
        if root is None or not isinstance(root, relax.Call):
            return False

        if root.op.name != "relax.reshape":
            return False

        shape_arg = root.args[1]
        if not isinstance(shape_arg, relax.Expr):
            return False

        return True

    def reshape_pattern():
        """Create Reshape Pattern"""

        pattern = is_op("relax.reshape")(wildcard(), wildcard())
        annotations = {
            "root": pattern,
        }
        return [("openclml.reshape", pattern, annotations, _check_reshape)]

    def _check_batchnorm(context: PatternCheckContext) -> bool:
        root = context.annotated_expr.get("root")
        if root is None or not isinstance(root, relax.Call):
            return False

        if root.op.name != "relax.reshape":
            return False

        required_params = ["moving_var", "gamma", "moving_mean", "beta"]
        for param in required_params:
            if param not in context.annotated_expr:
                return False

        params = {
            "moving_var": context.annotated_expr["moving_var"],
            "gamma": context.annotated_expr["gamma"],
            "moving_mean": context.annotated_expr["moving_mean"],
            "beta": context.annotated_expr["beta"],
        }

        for param in params.values():
            if not isinstance(param, tvm_tirx.ir.GenericConst):
                return False

        base_shape = None
        for param in params.values():
            shape = param.ty.shape
            dtype = _dtype_str(param.ty.dtype)

            if dtype not in {"float32"}:
                return False

            # Initialize base_shape if not set
            if base_shape is None:
                base_shape = shape
                continue

            # All parameters should have same shape
            if len(shape) != len(base_shape):
                return False
            if any(s1 != s2 for s1, s2 in zip(shape, base_shape)):
                return False

        return True

    def batch_norm_pattern():
        """Create a batch norm pattern."""
        data = wildcard()
        bn_scale = is_const()
        bn_bias = is_const()
        bn_mean = is_const()
        bn_var = is_const()

        pattern = is_op("relax.nn.batch_norm")(data, bn_scale, bn_bias, bn_mean, bn_var)
        pattern = is_tuple_get_item(pattern, 0)
        pattern = is_op("relax.reshape")(pattern, wildcard())

        annotations = {
            "gamma": bn_scale,
            "beta": bn_bias,
            "moving_mean": bn_mean,
            "moving_var": bn_var,
            "root": pattern,
        }

        return [
            ("openclml.nn.batch_norm", pattern, annotations, _check_batchnorm),
        ]

    def _check_binary_op(context: PatternCheckContext) -> bool:
        def _check_arg(input_expr):
            input_dtype = _dtype_str(input_expr.ty.dtype)
            input_shape = input_expr.ty.shape
            if len(input_shape) == 0:
                return False

            # Avoid any operators with dtype Int64
            if input_dtype == "int64":
                return False

            # No support for batch> 1
            if input_shape[0] > 1:
                return False

            return True

        def compare_shapes(lhs_shape, rhs_shape):
            if len(lhs_shape) != len(rhs_shape):
                return False
            for lhs_dim, rhs_dim in zip(lhs_shape, rhs_shape):
                if lhs_dim != rhs_dim:
                    return False
            return True

        lhs_shape = None
        rhs_shape = None
        if "lhs" in context.annotated_expr:
            lhs = context.annotated_expr["lhs"]
            lhs_shape = lhs.ty.shape
            if not _check_arg(lhs):
                return False

        if "rhs" in context.annotated_expr:
            rhs = context.annotated_expr["rhs"]
            rhs_shape = rhs.ty.shape
            if not _check_arg(rhs):
                return False

        # Checking for BinaryOps ( False for unaryOp )
        if (
            "lhs" in context.annotated_expr
            and "rhs" in context.annotated_expr
            and not compare_shapes(lhs_shape, rhs_shape)
        ):
            return False

        return True

    def binary_op_pattern():
        """Create a binary op pattern."""

        def make_pattern(op):
            lhs = wildcard()
            rhs = wildcard()
            pattern = is_op(op)(lhs, rhs)
            annotations = {"lhs": lhs, "rhs": rhs}
            return ("openclml." + op, pattern, annotations, _check_binary_op)

        binary_ops = [
            "relax.add",
            "relax.subtract",
            "relax.multiply",
            "relax.divide",
            "relax.maximum",
            "relax.minimum",
        ]

        return [make_pattern(op) for op in binary_ops]

    def unary_op_pattern():
        """Create a unary op pattern."""

        def make_pattern(op):
            lhs = wildcard()
            pattern = is_op(op)(lhs)
            annotations = {"lhs": lhs}
            return ("openclml." + op, pattern, annotations, _check_binary_op)

        unary_ops = [
            "relax.nn.softmax",
            "relax.nn.relu",
            "relax.clip",
        ]

        return [make_pattern(op) for op in unary_ops]

    return [
        *conv_pattern(),
        *batch_norm_pattern(),
        *binary_op_pattern(),
        *unary_op_pattern(),
        *maxpool_pattern(),
        *avgpool_pattern(),
        *global_avgpool_pattern(),
        *reshape_pattern(),
    ]


clml_patterns = clml_pattern_table()
register_patterns(clml_patterns)


@module_pass(opt_level=0, name="OpenCLMLOffLoad")
class OpenCLMLOffLoad:
    """The pass sequence used for CLML offload"""

    def transform_module(self, mod: IRModule, ctx: PassContext) -> IRModule:
        """The transform"""

        clml_layouts = {
            "relax.nn.conv2d": ["NCHW", "OIHW"],
            "relax.nn.conv2d_transpose": ["NCHW", "OIHW"],
        }
        seq = tvm_tirx.ir.transform.Sequential(
            [
                transform.ConvertLayout(clml_layouts),
                transform.Normalize(),
                transform.FoldBatchnormToConv2D(),
                AppendReshapeToBNRewriterPass(),
                transform.FoldConstant(),
                transform.FuseOpsByPattern(clml_pattern_table()),
                transform.MergeCompositeFunctions(),
                transform.RunCodegen(),
            ],
        )
        mod = seq(mod)
        return mod


def _check_dequantize_matmul(ctx: relax.transform.PatternCheckContext) -> bool:
    _input = ctx.annotated_expr["lhs"]
    root = ctx.annotated_expr["root"]
    wdq = ctx.annotated_expr["w_decoded"]
    w_pack = ctx.annotated_expr["w_encoded"]

    if _dtype_str(ctx.annotated_expr["lhs"].ty.dtype) != "float16":
        return False
    if not isinstance(wdq, relax.Call):
        return False
    g_var = wdq.args[0]
    if not (isinstance(g_var, relax.GlobalVar) and "dequantize" in g_var.name_hint):
        return False

    if not (
        (len(root.ty.shape) == 3)
        and isinstance(root.ty.shape[0], tvm_tirx.ir.IntImm)
        and (_dtype_str(root.ty.dtype) == "float16")
        and (root.ty.shape[0] == 1)
    ):
        return False

    if not (
        (len(wdq.ty.shape) == 2)
        and (w_pack.ty.shape[-1] == root.ty.shape[-1])
        and (wdq.ty.shape[-2] == _input.ty.shape[-1])
    ):
        return False

    return True


def dequantize_matmul_patterns():
    """Returns a list of supported decode -> matmul patterns."""

    def _dequantize_matmul_pattern(name):
        scales = wildcard()
        x = wildcard()
        w_packed = wildcard()

        w_decoded = is_op("relax.call_tir")(
            GlobalVarPattern(),
            TuplePattern([w_packed, scales]),
        )
        matmul = is_op("relax.matmul")(x, w_decoded)

        annotations = {
            "root": matmul,
            "lhs": x,
            "w_encoded": w_packed,
            "w_decoded": w_decoded,
            "scales": scales,
        }

        return name, matmul, annotations, _check_dequantize_matmul

    return [
        _dequantize_matmul_pattern("openclml.dequant_matmul"),
    ]


clml_llm_patterns = [
    *dequantize_matmul_patterns(),
]
register_patterns(clml_llm_patterns)


@tvm_tirx.ir.transform.module_pass(opt_level=0, name="OpenCLMLOffLoadForLLM")
class OpenCLMLOffLoadForLLM:
    """A compiler pass that partition the graph with dequant Matmul to CLML backend offload."""

    def __init__(self, target: tvm_tirx.target.Target) -> None:
        """Initializer.
        Parameters
        ----------
        target : tvm_tirx.target.Target
            Target device.
        """
        self.target = target

    def transform_module(
        self,
        mod: IRModule,
        _ctx: tvm_tirx.ir.transform.PassContext,
    ) -> IRModule:
        """Apply required passed to transform"""

        if "adreno" in self.target.keys and (clml_sdk_version() >= 5):
            mod = tvm_tirx.ir.transform.Sequential(
                [
                    transform.Normalize(),
                    transform.FuseOpsByPattern(clml_llm_patterns, annotate_codegen=True),
                    transform.RunCodegen(),
                ]
            )(mod)

        return mod
