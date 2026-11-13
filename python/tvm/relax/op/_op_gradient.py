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
# pylint: disable=unused-argument, redefined-builtin, invalid-name
"""Gradient definitions for Relax operators."""

import functools
import operator

from tvm import relax
from tvm.relax.type import ShapeType
from tvm_tirx.ir import Call, PrimType
from tvm_tirx.sym import Analyzer

from ..block_builder import BlockBuilder
from ..expr import Expr, ShapeExpr, Var
from .base import register_gradient
from .binary import greater_equal, less
from .create import triu
from .datatype import astype
from .grad import (
    avg_pool2d_backward,
    max_pool2d_backward,
    nll_loss_backward,
    no_grad,
    take_backward,
)
from .index import strided_slice
from .linear_algebra import matmul
from .manipulate import (
    broadcast_to,
    collapse_sum_to,
    concat,
    expand_dims,
    flatten,
    permute_dims,
    reshape,
    split,
    squeeze,
)
from .nn import conv2d, conv2d_transpose
from .search import where
from .statistical import cumsum, sum
from .unary import cos, exp, log, sigmoid, sin

# TODO(yixin, chaofan): handle symbolic shape for most of the gradients


##################### Utilities #####################


def _get_shape(expr: Expr) -> ShapeExpr:
    """Get the shape from a Tensor expr."""
    try:
        shape = expr.ty.shape
    except Exception as error:
        raise RuntimeError(
            f"Get the shape of {expr} failed. Please normalize it first and ensure it is a Tensor."
        ) from error
    return shape


def _get_dtype(expr: Expr) -> str:
    """Get the dtype from a Tensor expr."""
    try:
        dtype = expr.ty.dtype
    except Exception as error:
        raise RuntimeError(
            f"Get the dtype of {expr} failed. Please normalize it first and ensure it is a Tensor."
        ) from error
    if isinstance(dtype, PrimType):
        dtype = dtype.dtype
    return dtype


def _fit_shape(bb: BlockBuilder, input_grad: Expr, input: Expr) -> Expr:
    """When expr and target has the same shape, return expr;
    otherwise return `collapse_sum_to(expr, target.ty.shape)`.

    Will use BlockBuilder to normalize expr first.
    """
    target_shape = _get_shape(input)
    expr_ty = _get_shape(bb.normalize(input_grad)).ty
    target_ty = target_shape.ty
    assert isinstance(expr_ty, ShapeType)
    assert isinstance(target_ty, ShapeType)

    def _check_shape_equal(lhs: ShapeType, rhs: ShapeType):
        if len(lhs.values) != len(rhs.values):
            return False
        analyzer = Analyzer()
        for i, field in enumerate(lhs.values):
            if not analyzer.can_prove_equal(field, rhs.values[i]):
                return False
        return True

    return (
        input_grad
        if _check_shape_equal(expr_ty, target_ty)
        else collapse_sum_to(input_grad, target_shape)
    )


def _get_shape_prod(expr, axis):
    # Requires static shape
    shape = _get_shape(expr)
    if axis is None:
        return functools.reduce(operator.mul, (int(i) for i in shape), 1)
    return functools.reduce(operator.mul, (int(shape[int(i)]) for i in axis), 1)


##################### Binary #####################


@register_gradient("relax.add")
def add_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of add.

    Forward Form:
        `z = relax.add(x, y)`

    Backward:
        Returns `[z_output_grad, z_grad]`.
    """
    return [
        _fit_shape(ctx, output_grad, orig_call.args[0]),
        _fit_shape(ctx, output_grad, orig_call.args[1]),
    ]


@register_gradient("relax.subtract")
def subtract_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of subtract.

    Forward Form:
        `z = relax.subtract(x, y)`

    Backward:
        Returns `[z_output_grad, -z_grad]`.
    """
    return [
        _fit_shape(ctx, output_grad, orig_call.args[0]),
        _fit_shape(ctx, -output_grad, orig_call.args[1]),
    ]


@register_gradient("relax.multiply")
def multiply_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of multiply.

    Forward Form:
        `z = relax.multiply(x, y)`

    Backward:
        Returns `[z_grad * y, z_grad * x]`.
    """
    x, y = orig_call.args
    return [
        _fit_shape(ctx, output_grad * y, x),
        _fit_shape(ctx, output_grad * x, y),
    ]


@register_gradient("relax.divide")
def divide_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of divide.

    Forward Form:
        `z = relax.divide(x, y)`

    Backward:
        Returns `[z_grad / y,  -z_grad * z / y]`.
    """
    x, y = orig_call.args
    return [
        _fit_shape(ctx, output_grad / y, x),
        _fit_shape(ctx, -output_grad * orig_var / y, y),
    ]


@register_gradient("relax.power")
def power_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of power.

    Forward Form:
        `z = relax.power(x, y)`

    Backward:
        Returns `[y * x ** (y-1) * z_grad, z * ln(x) * z_grad]`.

        The gradient w.r.t. the second parameter, y, makes sense only when x > 0.
    """
    x, y = orig_call.args
    one = relax.const(1, _get_dtype(y))
    return [
        _fit_shape(ctx, output_grad * y * (x ** (y - one)), x),
        _fit_shape(ctx, output_grad * orig_var * log(x), y),
    ]


@register_gradient("relax.maximum")
def maximum_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of maximum.

    Forward Form:
        `z = relax.maximum(x, y)`

    Backward:
        Returns `[where(x < y, 0, z_grad), where(x >= y, 0, z_grad)]`.
    """
    x = orig_call.args[0]
    y = orig_call.args[1]
    zero = relax.const(0, _get_dtype(x))
    return [
        _fit_shape(ctx, where(less(x, y), zero, output_grad), x),
        _fit_shape(ctx, where(greater_equal(x, y), zero, output_grad), y),
    ]


@register_gradient("relax.minimum")
def minimum_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of minimum.

    Forward Form:
        `z = relax.minimum(x, y)`

    Backward:
        Returns `[where(x >= y, 0, z_grad), where(x < y, 0, z_grad)]`.
    """
    x = orig_call.args[0]
    y = orig_call.args[1]
    zero = relax.const(0, _get_dtype(x))
    return [
        _fit_shape(ctx, where(greater_equal(x, y), zero, output_grad), x),
        _fit_shape(ctx, where(less(x, y), zero, output_grad), y),
    ]


##################### Binary Comparison #####################

# For comparison operators, the gradients are no_grad


@register_gradient("relax.equal")
def equal_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    return [no_grad(orig_call.args[0]), no_grad(orig_call.args[1])]


@register_gradient("relax.greater")
def greater_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    return [no_grad(orig_call.args[0]), no_grad(orig_call.args[1])]


@register_gradient("relax.greater_equal")
def greater_equal_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    return [no_grad(orig_call.args[0]), no_grad(orig_call.args[1])]


@register_gradient("relax.less")
def less_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    return [no_grad(orig_call.args[0]), no_grad(orig_call.args[1])]


@register_gradient("relax.less_equal")
def less_equal_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    return [no_grad(orig_call.args[0]), no_grad(orig_call.args[1])]


@register_gradient("relax.not_equal")
def not_equal_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    return [no_grad(orig_call.args[0]), no_grad(orig_call.args[1])]


##################### Create #####################

# For zeros/ones/full operators, the gradients are no_grad.


@register_gradient("relax.zeros_like")
def zeros_like_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    return [no_grad(orig_call.args[0])]


@register_gradient("relax.ones_like")
def ones_like_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    return [no_grad(orig_call.args[0])]


@register_gradient("relax.full_like")
def full_like_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    return [no_grad(orig_call.args[0]), no_grad(orig_call.args[1])]


@register_gradient("relax.zeros")
def zeros_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    return [no_grad(orig_call.args[0])]


@register_gradient("relax.ones")
def ones_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    return [no_grad(orig_call.args[0])]


@register_gradient("relax.full")
def full_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    return [no_grad(orig_call.args[0]), no_grad(orig_call.args[1])]


# Other create gradients operators


@register_gradient("relax.triu")
def triu_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of triu.

    Forward Form:
        `y = relax.triu(x, k)`

    Backward:
        Returns `[triu(y_grad, k)]`.
    """
    k = orig_call.args[1]
    return [triu(output_grad, k)]


##################### Unary #####################


@register_gradient("relax.abs")
def abs_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of abs.

    Forward Form:
        `y = relax.abs(x)`

    Backward:
        Returns `[y_grad * where(x < 0, -1, 1)]`.
    """
    x = orig_call.args[0]
    zero = relax.const(0, _get_dtype(x))
    one = relax.const(1, _get_dtype(x))
    return [output_grad * where(less(x, zero), -one, one)]


@register_gradient("relax.cos")
def cos_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of cos.

    Forward Form:
        `y = relax.cos(x)`

    Backward:
        Returns `[-y_grad * sin(x)]`.
    """
    return [-output_grad * sin(orig_call.args[0])]


@register_gradient("relax.exp")
def exp_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of exp.

    Forward Form:
        `y = relax.exp(x)`

    Backward:
        Returns `[y_grad * y]`.
    """
    return [output_grad * orig_var]


@register_gradient("relax.log")
def log_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of log.

    Forward Form:
        `y = relax.log(x)`

    Backward:
        Returns `[y_grad / x]`.
    """
    return [output_grad / orig_call.args[0]]


@register_gradient("relax.negative")
def negative_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of negative.

    Forward Form:
        `y = relax.negative(x)`

    Backward:
        Returns `[-y_grad]`.
    """
    return [-output_grad]


@register_gradient("relax.sigmoid")
def sigmoid_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of sigmoid.

    Forward Form:
        `y = relax.sigmoid(x)`

    Backward:
        Returns `[y_grad * y * (1 - y)]`.
    """
    one = relax.const(1, _get_dtype(orig_call.args[0]))
    return [output_grad * orig_var * (one - orig_var)]


@register_gradient("relax.sin")
def sin_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of sin.

    Forward Form:
        `y = relax.sin(x)`

    Backward:
        Returns `[y_grad * cos(x)]`.
    """
    return [output_grad * cos(orig_call.args[0])]


@register_gradient("relax.sqrt")
def sqrt_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of sqrt.

    Forward Form:
        `y = relax.sqrt(x)`

    Backward:
        Returns `[0.5 * y_grad / y]`.
    """
    x = orig_call.args[0]
    cst = relax.const(0.5, _get_dtype(x))
    return [cst * output_grad / orig_var]


@register_gradient("relax.tanh")
def tanh_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of tanh.

    Forward Form:
        `y = relax.tanh(x)`

    Backward:
        Returns `[y_grad * (1 - y * y)]`.
    """
    one = relax.const(1, _get_dtype(orig_call.args[0]))
    return [output_grad * (one - orig_var * orig_var)]


##################### Statistical #####################


@register_gradient("relax.sum")
def sum_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of sum.

    Forward Form:
        `y = relax.sum(x, axis, keepdims)`

    Backward:
        Returns `[broadcast_to(y_output_grad, x.shape)]`.

        If `keepdims=False`, the summed axis will be added back.
    """
    axis = orig_call.attrs.axis
    keepdims = orig_call.attrs.keepdims
    if not keepdims and axis:
        output_grad = expand_dims(output_grad, axis)
    return [broadcast_to(output_grad, _get_shape(orig_call.args[0]))]


@register_gradient("relax.mean")
def mean_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of mean.

    Forward Form:
        `y = relax.mean(x, axis, keepdims)`

    Backward:
        Returns `[broadcast_to(y_output_grad, x.shape) / prod(x.shape[i] for i in axis)]`.

        If `keepdims=False`, the mean axis will be added back.
    """
    axis = orig_call.attrs.axis
    keepdims = orig_call.attrs.keepdims
    output_grad = output_grad / relax.const(
        _get_shape_prod(orig_call.args[0], axis), _get_dtype(output_grad)
    )
    if not keepdims and axis:
        output_grad = expand_dims(output_grad, axis)
    return [broadcast_to(output_grad, _get_shape(orig_call.args[0]))]


@register_gradient("relax.variance")
def variance_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of variance.

    Forward Form:
        `y = relax.variance(x, axis, keepdims)`

    Backward:
        Returns `[broadcast_to(y_output_grad, x.shape)]`.

        If `keepdims=False`, the summed axis will be added back.
    """
    x = orig_call.args[0]
    axis = orig_call.attrs.axis
    keepdims = orig_call.attrs.keepdims
    shape_prod = _get_shape_prod(x, axis)
    dtype = _get_dtype(x)
    grad1 = relax.const(2.0 / shape_prod, dtype) * x
    grad2 = relax.const(2.0 / shape_prod / shape_prod, dtype) * sum(x, axis, keepdims=True)
    if not keepdims and axis:
        output_grad = expand_dims(output_grad, axis)
    return [output_grad * (grad1 - grad2)]


##################### Manipulate #####################


@register_gradient("relax.permute_dims")
def permute_dims_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of permute_dims.

    Forward Form:
        `y = relax.permute_dims(x, axes)`

    Backward:
        Returns grad transposed over the **inverse permutation** of the original permute_dims axes.
    """
    axes = orig_call.attrs.axes
    if axes:
        dims = len(axes)
        new_axes = [0] * dims
        for i in range(dims):
            new_axes[int(axes[i])] = i
        return [permute_dims(output_grad, axes=new_axes)]
    return [permute_dims(output_grad)]


@register_gradient("relax.concat")
def concat_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of concat.

    Forward Form:
        `y = relax.concat((x1, x2, x3), axis)`

    Backward:
        Returns `[split(y_output_grad, [x1.shape[axis], x1.shape[axis] + x2.shape[axis]], axis)]`.
    """
    axis = orig_call.attrs.axis
    assert axis is not None
    axis = int(axis)
    split_indices: list[Expr] = []
    ty = orig_call.args[0].ty
    assert isinstance(ty, relax.TupleType)
    for i in range(len(ty.fields) - 1):
        tensor_ty = ty.fields[i]
        assert isinstance(tensor_ty, relax.TensorType)
        assert tensor_ty.shape is not None
        index = tensor_ty.shape[axis]
        if i > 0:
            index += split_indices[i - 1]
        split_indices.append(index)
    return [split(output_grad, split_indices, axis)]


@register_gradient("relax.split")
def split_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of split.

    Forward Form:
        `y = relax.split(x, indices, axis)`

    Backward:
        Returns `[concat(y_output_grad, axis)]`.
    """
    axis = orig_call.attrs.axis
    axis = int(axis)
    return [concat(output_grad, axis)]


@register_gradient("relax.expand_dims")
def expand_dims_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of expand_dims.

    Forward Form:
        `y = relax.expand_dims(x, axis)`

    Backward:
        Returns `[squeeze_dims(y_grad, axis)]`.
    """
    return [squeeze(output_grad, orig_call.attrs.axis)]


@register_gradient("relax.reshape")
def reshape_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of reshape.

    Forward Form:
        `y = relax.reshape(x, new_shape)`

    Backward:
        Returns `[reshape(y_grad, x.shape), no_grad]`.

        The second parameter, the target ShapeExpr, is not differentiable.
    """
    return [
        reshape(output_grad, _get_shape(orig_call.args[0])),
        no_grad(orig_call.args[1]),
    ]


@register_gradient("relax.cumsum")
def cumsum_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of cumsum.

    Forward Form:
        `y = relax.cumsum(x, axis)`

    Backward:
        The "reversed" cumsum along the same axis. Implemented by some tricks now.
    """

    axis = orig_call.attrs.axis
    dtype = orig_call.attrs.dtype
    x_shape = _get_shape(orig_call.args[0])

    if axis is not None:
        axis = int(axis)
        grad = sum(output_grad, axis, keepdims=True) - cumsum(output_grad, axis) + output_grad
    else:
        grad = reshape(
            sum(output_grad, keepdims=True) - cumsum(output_grad) + flatten(output_grad), x_shape
        )

    if dtype is not None:
        grad = astype(grad, _get_dtype(orig_call.args[0]))

    return [grad]


@register_gradient("relax.broadcast_to")
def broadcast_to_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of broadcast_to.

    Forward Form:
        `y = relax.broadcast_to(x, new_shape)`

    Backward:
        Returns `[collapse_sum_to(y_grad, x.shape), no_grad]`.

        The second parameter, the target ShapeExpr, is not differentiable.
    """
    return [
        collapse_sum_to(output_grad, _get_shape(orig_call.args[0])),
        no_grad(orig_call.args[1]),
    ]


##################### Index #####################


@register_gradient("relax.take")
def take_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of take.

    Forward Form:
        `y = relax.take(x, indices, axis)`

    Backward:
        Returns [x_grad, no_grad].

        The second parameter, the indices, is not differentiable.
    """

    axis = orig_call.attrs["axis"]

    return [
        take_backward(output_grad, orig_call.args[0], orig_call.args[1], axis),
        no_grad(orig_call.args[1]),
    ]


##################### Search #####################


@register_gradient("relax.where")
def where_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of where.

    Forward Form:
        `y = relax.where(cond, x1, x2)`

    Backward:
        Returns `[where(cond, y_grad, 0), where(cond, 0, y_grad)]`.

        The first parameter, the condition, is not differentiable.
    """

    cond = orig_call.args[0]
    x1_zero = relax.const(0, _get_dtype(orig_call.args[1]))
    x2_zero = relax.const(0, _get_dtype(orig_call.args[2]))

    return [
        no_grad(orig_call.args[0]),
        where(cond, output_grad, x1_zero),
        where(cond, x2_zero, output_grad),
    ]


##################### Linear Algebra #####################


@register_gradient("relax.matmul")
def matmul_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of matmul.

    Forward Form:
        `c = relax.matmul(a, b)`

    Backward:
        Generally, returns `[c_grad @ b^T, a^T @ c_grad]`.

        Here we only transpose the last two dimensions because of the definition
        of batch matmul. Note that ndim=1 should be treaded specially.
    """

    tensor_a, tensor_b = orig_call.args

    a_dim = len(_get_shape(tensor_a))
    b_dim = len(_get_shape(tensor_b))

    def _transpose_last_two_dim(tensor, ndim):
        """Helper function for reversing the last two dimensions."""
        assert ndim > 1
        return permute_dims(
            tensor, axes=[i if i < ndim - 2 else 2 * ndim - 3 - i for i in range(ndim)]
        )

    if a_dim > 1 and b_dim > 1:
        a_grad = matmul(output_grad, _transpose_last_two_dim(tensor_b, b_dim))
        b_grad = matmul(_transpose_last_two_dim(tensor_a, a_dim), output_grad)
    elif a_dim == 1 and b_dim > 1:
        a_expand = expand_dims(tensor_a, 1)
        grad_expand = expand_dims(output_grad, -2)
        a_grad = matmul(grad_expand, _transpose_last_two_dim(tensor_b, b_dim))
        b_grad = matmul(a_expand, grad_expand)
    elif b_dim == 1 and a_dim > 1:
        b_expand = expand_dims(tensor_b, 0)
        grad_expand = expand_dims(output_grad, -1)
        a_grad = matmul(grad_expand, b_expand)
        b_grad = squeeze(
            matmul(_transpose_last_two_dim(tensor_a, a_dim), grad_expand), axis=-1
        )  # squeeze last dim
    else:
        assert a_dim == 1 and b_dim == 1
        a_grad = output_grad * tensor_b
        b_grad = output_grad * tensor_a

    return [
        _fit_shape(ctx, a_grad, tensor_a),
        _fit_shape(ctx, b_grad, tensor_b),
    ]


##################### Datatype #####################


@register_gradient("relax.astype")
def astype_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of astype.

    Forward Form:
        `y = relax.astype(x, dtype_of_y)`

    Backward:
        Returns `[astype(y_grad, dtype_of_x)]`.
    """
    return [astype(output_grad, _get_dtype(orig_call.args[0]))]


##################### Neural network #####################


@register_gradient("relax.nn.relu")
def relu_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of relu.

    Forward Form:
        `y = relax.relu(x)`

    Backward:
        Returns `[where(x < 0, 0, y_grad)]`.
    """
    x = orig_call.args[0]
    zero = relax.const(0, _get_dtype(x))
    return [where(less(x, zero), zero, output_grad)]


@register_gradient("relax.nn.silu")
def silu_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of silu.

    Forward Form:
        `y = relax.silu(x)`

    Backward:
        Returns `[y_grad * (sigmoid(x) + y * (1 - sigmoid(x)))]`.
    """
    x = orig_call.args[0]
    sig = sigmoid(x)
    one = relax.const(1, _get_dtype(x))
    return [output_grad * (sig + orig_var * (one - sig))]


@register_gradient("relax.nn.softmax")
def softmax_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of softmax.

    Forward Form:
        `y = relax.softmax(x, axis)`

    Backward:
        Returns `[(y_grad - sum(y_grad * y, axis, keepdims=True)) * y]`
    """
    return [(output_grad - sum(output_grad * orig_var, orig_call.attrs.axis, True)) * orig_var]


@register_gradient("relax.nn.log_softmax")
def log_softmax_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of log_softmax.

    Forward Form:
        `y = relax.log_softmax(x, axis)`

    Backward:
        Returns `[y_grad - sum(y_grad, axis, keepdims=True) * exp(y)]`
    """
    y_exp = exp(orig_var)
    return [output_grad - sum(output_grad, orig_call.attrs.axis, True) * y_exp]


@register_gradient("relax.nn.cross_entropy_with_logits")
def cross_entropy_with_logits_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of cross_entropy_with_logits.

    Forward Form:
        `z = relax.nn.cross_entropy_with_logits(x, y)`

    Backward:
        Returns `[-z_grad * y, -z_grad * x]`.
        If it has batch_size N, the results should divide by N.
    """
    x, y = orig_call.args

    if x.ty.ndim > 1:
        batch_size = int(_get_shape(x)[0])
        output_grad = output_grad / relax.const(batch_size, _get_dtype(output_grad))

    return [-output_grad * y, -output_grad * x]


# TODO(chaofan, yixin): remove nll_loss_backward and register the gradient using existing operators
# This may require one_hot, strided_set, etc.
@register_gradient("relax.nn.nll_loss")
def nll_loss_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
):
    """Gradient of nll_loss.

    Forward Form:
        `z = relax.nn.nll_loss(predictions, targets, weights, reduction, ignore_index)`

        Suppose that `out = nll_loss(predictions, targets, weights, "none", ignore_index)`, and
        `z = reduction(out)` where reduction is in `["none", "mean", "sum"]`.

    Backward:
        First find the gradient w.r.t. `out`. Assume it is `out_grad`.

        Gererally, the gradient w.r.t. predictions is

        `predictions_grad[n, c, i_1, ..., i_k] = -o * w if c == t else 0`, where
        - `o = out_grad[n, i_1, ..., i_k]`,
        - `w = weights[n, i_1, ..., i_k]`,
        - `t = targets[n, i_1, ..., i_k]`.

        Additional checks are added if `ignore_index >= 0`, `weights=None`, or the predictions
        provided do not have batch.

        The gradient w.r.t. targets and weights are not available.
    """
    pred_grad = nll_loss_backward(
        output_grad,
        orig_call.args[0],
        orig_call.args[1],
        weights=orig_call.args[2] if len(orig_call.args) == 3 else None,
        reduction=orig_call.attrs.reduction,
        ignore_index=orig_call.attrs.ignore_index,
    )
    if len(orig_call.args) == 2:
        return [pred_grad, no_grad(orig_call.args[1])]

    return [pred_grad, no_grad(orig_call.args[1]), no_grad(orig_call.args[2])]


@register_gradient("relax.nn.conv2d")
def conv2d_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> list[Expr]:
    """Gradient of conv2d. Now only supports `NCHW` data layout and `OIHW` kernel layout.

    Forward Form:
        `y = relax.nn.conv2d(x, weight, strides, padding, dilation, groups, data_layout, \
kernel_layout, out_layout, out_dtype)`

    Backward:
        Returns `[x_grad, weight_grad]`
    """
    attrs = orig_call.attrs
    assert attrs.data_layout == "NCHW", "only support NCHW data layout"
    assert attrs.kernel_layout == "OIHW", "only support OIHW kernel layout"
    assert attrs.out_layout == "NCHW", "only support NCHW output layout"

    assert len(attrs.padding) == 4
    assert len(attrs.strides) == 2
    assert len(attrs.dilation) == 2

    # calculate output_padding
    data, weight = orig_call.args
    batch, out_channel, grad_h, grad_w = _get_shape(orig_var)
    _, in_channel, in_h, in_w = _get_shape(data)
    _, _, filter_h, filter_w = _get_shape(weight)

    pad_top, pad_left, pad_bottom, pad_right = attrs.padding
    stride_h, stride_w = attrs.strides
    dilation_h, dilation_w = attrs.dilation

    out_h = (grad_h - 1) * stride_h - pad_top - pad_bottom + filter_h
    out_w = (grad_w - 1) * stride_w - pad_left - pad_right + filter_w

    output_padding = (int(in_h - out_h), int(in_w - out_w))

    data_grad = conv2d_transpose(  # type: ignore
        output_grad,
        orig_call.args[1],
        attrs.strides,
        attrs.padding,
        output_padding,
        attrs.dilation,
        attrs.groups,
        attrs.out_layout,
        attrs.kernel_layout[1] + attrs.kernel_layout[0] + attrs.kernel_layout[2:],
        attrs.data_layout,
        attrs.out_dtype,
    )

    if attrs.groups != 1:
        data = reshape(data, (batch, attrs.groups, in_channel // attrs.groups, in_h, in_w))
        data = permute_dims(data, [1, 0, 2, 3, 4])
        data = reshape(data, (batch * attrs.groups, in_channel // attrs.groups, in_h, in_w))

    weight_grad = conv2d(
        data,
        output_grad,
        strides=attrs.dilation,
        padding=attrs.padding,
        dilation=attrs.strides,
        groups=attrs.groups,
        out_dtype=attrs.out_dtype,
        data_layout="CNHW",
        kernel_layout="IOHW",
        out_layout="CNHW",
    )

    # infer shape of weight_grad
    weight_grad_h = (in_h - (grad_h - 1) * stride_h - 1 + pad_top + pad_bottom) // dilation_h + 1
    weight_grad_w = (in_w - (grad_w - 1) * stride_w - 1 + pad_left + pad_right) // dilation_w + 1

    assert weight_grad_h >= filter_h
    assert weight_grad_w >= filter_w

    if weight_grad_h > filter_h or weight_grad_w > filter_w:
        weight_grad = strided_slice(
            weight_grad,
            axes=[0, 1, 2, 3],
            begin=[0, 0, 0, 0],
            end=[out_channel, in_channel // attrs.groups, filter_h, filter_w],
        )

    return [data_grad, weight_grad]


@register_gradient("relax.nn.max_pool2d")
def max_pool2d_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
):
    """Gradient of max_pool2d.

    Forward Form:
        `y = relax.nn.max_pool2d(x, pool_size, strides, padding, dilation, ceil_mode, layout, \
out_layout)`

    Backward:
        Returns `[x_grad]`
    """
    return [
        max_pool2d_backward(  # type: ignore
            output_grad,
            orig_call.args[0],
            orig_call.attrs.pool_size,
            orig_call.attrs.strides,
            orig_call.attrs.padding,
            orig_call.attrs.dilation,
            orig_call.attrs.ceil_mode,
            orig_call.attrs.count_include_pad,
            orig_call.attrs.layout,
            orig_call.attrs.out_layout,
        )
    ]


@register_gradient("relax.nn.avg_pool2d")
def avg_pool2d_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
):
    """Gradient of avg_pool2d.

    Forward Form:
        `y = relax.nn.avg_pool2d(x, pool_size, strides, padding, dilation, ceil_mode, layout, \
out_layout)`

    Backward:
        Returns `[x_grad]`
    """
    return [
        avg_pool2d_backward(  # type: ignore
            output_grad,
            orig_call.args[0],
            orig_call.attrs.pool_size,
            orig_call.attrs.strides,
            orig_call.attrs.padding,
            orig_call.attrs.dilation,
            orig_call.attrs.ceil_mode,
            orig_call.attrs.count_include_pad,
            orig_call.attrs.layout,
            orig_call.attrs.out_layout,
        )
    ]
