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

"""Pattern table for cuDNN backend"""

import operator
from functools import partial, reduce

import tvm_tirx
from tvm import relax
from tvm.relax import PyExprMutator, expr_functor, transform
from tvm.relax.transform import PatternCheckContext

from ..pattern_registry import get_patterns_with_prefix, register_patterns
from ..patterns import make_conv2d_pattern, make_stacked_attention_pattern
from ..utils import has_leaking_intermediate_variables


def _is_supported_dtype(lhs_dtype, rhs_dtype):
    """Check if dtypes in the given workload are supported by cuDNN BYOC."""
    return (lhs_dtype == "float16" and rhs_dtype == "float16") or (
        lhs_dtype == "float32" and rhs_dtype == "float32"
    )


def _is_supported_format(data_layout, kernel_layout):
    """Check if layouts in the given workload are supported by cuDNN BYOC."""
    return (data_layout == "NHWC" and kernel_layout == "OHWI") or (
        data_layout == "NCHW" and kernel_layout == "OIHW"
    )


def _check_conv2d(context: PatternCheckContext) -> bool:
    if has_leaking_intermediate_variables(context):
        return False
    # Retrieve the annotated expression from context
    conv2d_call = context.annotated_expr["root"]
    input_expr = context.annotated_expr["input"]
    weight_expr = context.annotated_expr["weight"]

    # Check if the data types of input and weights are supported by cuDNN BYOC
    input_dtype = input_expr.ty.dtype
    weight_dtype = weight_expr.ty.dtype
    if not _is_supported_dtype(input_dtype, weight_dtype):
        return False

    input_layout = conv2d_call.attrs.data_layout
    weight_layout = conv2d_call.attrs.kernel_layout
    if not _is_supported_format(input_layout, weight_layout):
        return False

    return True


def _check_stacked_attention(context: PatternCheckContext, layout: str) -> bool:
    """Check if the given stacked attention workload can be offloaded to cuDNN."""
    if has_leaking_intermediate_variables(context):
        return False
    if layout == "BS3NH":
        if not context.annotated_expr["stacked_qkv"].ty.ndim == 3:
            return False
        if "split" in context.annotated_expr:
            split_op = context.annotated_expr["split"]
            if not split_op.attrs.axis == 2:
                return False
    elif layout == "SBN3H":
        if not context.annotated_expr["stacked_qkv"].ty.ndim == 4:
            return False
        if "split" in context.annotated_expr:
            split_op = context.annotated_expr["split"]
            if not split_op.attrs.axis == 3:
                return False
    else:
        raise NotImplementedError(f"Unsupported layout: {layout}")
    if not context.annotated_expr["stacked_qkv"].ty.dtype == "float16":
        return False
    # The cuDNN runtime builds an unmasked SDPA graph, so offloading a masked attention would
    # silently compute bidirectional attention.
    attention = context.annotated_expr["attention"]
    if attention.attrs.causal_mask is not None:
        return False
    if attention.attrs.window_size is not None:
        return False
    return True


register_patterns(
    [
        (
            "cudnn.conv2d.nhwc_ohwi",
            *make_conv2d_pattern(
                with_bias=False,
            ),
            _check_conv2d,
        ),
        (
            "cudnn.conv2d.nhwc_ohwi_bias",
            *make_conv2d_pattern(
                with_bias=True,
            ),
            _check_conv2d,
        ),
        (
            "cudnn.conv2d.nhwc_ohwi_bias_relu",
            *make_conv2d_pattern(
                with_bias=True,
                activation="relax.nn.relu",
            ),
            _check_conv2d,
        ),
        (
            "cudnn.attention.BS3NH",
            *make_stacked_attention_pattern(start_op="split", layout="BS3NH"),
            partial(_check_stacked_attention, layout="BS3NH"),
        ),
        (
            "cudnn.attention.SBN3H",
            *make_stacked_attention_pattern(start_op="split", layout="SBN3H"),
            partial(_check_stacked_attention, layout="SBN3H"),
        ),
    ]
)


def partition_for_cudnn(mod):
    """
    Partition the input module into cuDNN-supported subgraphs.

    Parameters
    ----------
    mod: tvm_tirx.IRModule
        The IRModule to be partitioned.

    Returns
    -------
    mod: tvm_tirx.IRModule
        The resulting IRModule, containing partitioned subgraphs to be
        offloaded to the cuDNN backend.
    """

    patterns = get_patterns_with_prefix("cudnn")
    return tvm_tirx.ir.transform.Sequential(
        [
            transform.FuseOpsByPattern(patterns, bind_constants=False, annotate_codegen=True),
            annotate_workspace,
            transform.AllocateWorkspace(),
        ]
    )(mod)


def _shape_1d(shape):
    return reduce(operator.mul, shape, 1)


@expr_functor.mutator
class WorkspaceAnnotator(PyExprMutator):
    """Annotate a workspace requirement for each cuDNN-offloaded function."""

    def __init__(self, mod):
        super().__init__(mod)

    def visit_function_(self, f):
        if "Composite" not in f.attrs:
            body = super().visit_expr(f.body)
            new_f = relax.Function(f.params, body, f.ret_ty, f.is_pure, f.attrs, f.loc)

            if "global_symbol" in f.attrs and "cudnn" in f.attrs["global_symbol"]:
                composite_func = body.blocks[0].bindings[0].value
                if "WorkspaceSize" in composite_func.attrs:
                    return new_f.with_attr("WorkspaceSize", composite_func.attrs["WorkspaceSize"])

            return new_f

        if "attention" in f.attrs["Composite"] and "cudnn" in f.attrs["Composite"]:
            # Workspace is needed only for larger head sizes, but for simplicity we always allocate.
            out_dtype = f.ret_ty.dtype
            out_size_1d = _shape_1d(f.ret_ty.shape)
            # This needs to be in sync with the actual value that the kernel expects.
            workspace_size_bytes = out_size_1d * {"float16": 2, "float32": 4}[out_dtype]
            if not isinstance(workspace_size_bytes, int | tvm_tirx.ir.IntImm):
                # Tempororay workaround for dynamic shape workload. Will be removed when
                # workspace for dynamic shape workload is implemented.
                workspace_size_bytes = 8
            return f.with_attr("WorkspaceSize", workspace_size_bytes)

        return f


@tvm_tirx.ir.transform.module_pass(opt_level=0)
def annotate_workspace(mod, _):
    """Pass to annotate a workspace requirement for each cuDNN-offloaded function."""
    annotator = WorkspaceAnnotator(mod)
    for name, f in mod.functions_items():
        if isinstance(f, relax.Function):
            new_f = annotator.visit_expr(f)
            mod.update_func(name, new_f)
    return mod
