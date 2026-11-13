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
# pylint: disable=redefined-builtin
"""Operators to implement operaor gradients. Used in `_op_gradient.py`.

We are trying to keep grad operators as simple as possible, and hope they are only used for finding
gradients for forward operators. The ty inference for grad operators just returns the
ty of the input.
"""

from tvm_tirx.ir import Call as _Call
from tvm_tirx.ir.attrs import make_node as _make_attrs
from tvm_tirx.ir.location import UNKNOWN_LOC as _UNKNOWN_LOC
from tvm_tirx.ir.location import Location as _Location

from ...expr import Expr


def no_grad(input: Expr, *, ty=None, loc: _Location = _UNKNOWN_LOC) -> Expr:
    """No gradient dummy operator w.r.t. the input.

    Parameters
    ----------
    input : relax.Expr
      The corresponding input tensor.

    Returns
    -------
    result : relax.Expr
      The no-gradient representation w.r.t. input.
    """
    return _Call("relax.grad.no_grad", [input], loc=loc, ty=ty)  # type: ignore


def start_checkpoint(input: Expr, *, ty=None, loc: _Location = _UNKNOWN_LOC) -> Expr:
    """Mark the start of the checkpoint stage. The computation between start_checkpoint and
    end_checkpoint will be marked as the checkpoint stage.

    Rather than storing all intermediate activations of the entire computation graph for
    computing backward, the checkpointed stage does not save intermediate activations, and instead
    recomputes them in backward process.

    For instance,
    ```
    a = relax.Var("a", relax.TensorType((2, 2), "float32"))
    b = relax.Var("b", relax.TensorType((2, 2), "float32"))
    c = a * 2
    d = b * 2
    c_cp = start_checkpoint(c)
    d_cp = start_checkpoint(d)
    e = c_cp + d_cp
    e_out = end_checkpoint(e)
    ```
    Then `e` will be recomputed in the backward stage.

    See tvm.relax.transform.Gradient, tvm.relax.testing.nn.checkpoint,
    tvm.relax.op.grad.end_checkpoint for more information.

    Parameters
    ----------
    input : relax.Expr
      The tensor marking the input of the checkpoint stage.

    Returns
    -------
    result : relax.Expr
      The same tensor as the input.
    """
    return _Call(
        "relax.grad.start_checkpoint",
        [input],
        loc=loc,
        ty=ty,
    )  # type: ignore


def end_checkpoint(input: Expr, *, ty=None, loc: _Location = _UNKNOWN_LOC) -> Expr:
    """Mark the end of checkpoint stage. See tvm.relax.op.grad.start_checkpoint.

    Parameters
    ----------
    input : relax.Expr
      The output of the checkpoint stage.

    Returns
    -------
    result : relax.Expr
      The same tensor as the input.
    """
    return _Call(
        "relax.grad.end_checkpoint",
        [input],
        loc=loc,
        ty=ty,
    )  # type: ignore


def nll_loss_backward(
    output_grad: Expr,
    predictions: Expr,
    targets: Expr,
    weights: Expr | None = None,
    reduction: str = "mean",
    ignore_index: int = -100,
    *,
    ty=None,
    loc: _Location = _UNKNOWN_LOC,
) -> Expr:
    """Backward operator of relax.nn.nll_loss. All parameters except output_grad is the same as
    relax.nn.nll_loss. Returns the gradient w.r.t. predictions.

    Parameters
    ----------
    output_grad : relax.Expr
      The gradient w.r.t. the result of nll_loss.

    Returns
    -------
    result : relax.Expr
      The gradient w.r.t. predictions.
    """
    return _Call(
        "relax.grad.nll_loss_backward",
        [output_grad, predictions, targets, *([] if weights is None else [weights])],
        attrs=_make_attrs(
            "relax.attrs.NLLLossAttrs", reduction=reduction, ignore_index=ignore_index
        ),
        loc=loc,
        ty=ty,
    )


def max_pool2d_backward(
    output_grad: Expr,
    data: Expr,
    pool_size: tuple[int, int] = (1, 1),
    strides: tuple[int, int] = (1, 1),
    padding: tuple[int, int, int, int] = (0, 0, 0, 0),
    dilation: tuple[int, int] = (1, 1),
    ceil_mode: bool = False,
    count_include_pad: bool = False,
    layout: str = "NCHW",
    out_layout: str | None = None,
    *,
    ty=None,
    loc: _Location = _UNKNOWN_LOC,
) -> Expr:
    """Backward operator of relax.nn.max_pool2d. All parameters except output_grad is the same as
    relax.nn.max_pool2d. Returns the gradient w.r.t. data.

    Parameters
    ----------
    output_grad : relax.Expr
      The gradient w.r.t. the result of max_pool2d.

    Returns
    -------
    result : relax.Expr
      The gradient w.r.t. data.
    """
    if len(padding) == 2:
        padding = tuple(padding) + tuple(padding)
    if len(strides) == 1:
        strides = tuple(strides) * 2
    if len(dilation) == 1:
        dilation = tuple(dilation) * 2
    if len(pool_size) == 1:
        pool_size = tuple(pool_size) * 2
    return _Call(
        "relax.grad.max_pool2d_backward",
        [output_grad, data],
        attrs=_make_attrs(
            "relax.attrs.Pool2DAttrs",
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            layout=layout,
            out_layout=(layout if out_layout is None else out_layout),
        ),
        loc=loc,
        ty=ty,
    )


def avg_pool2d_backward(
    output_grad: Expr,
    data: Expr,
    pool_size: tuple[int, int] = (1, 1),
    strides: tuple[int, int] = (1, 1),
    padding: tuple[int, int, int, int] = (0, 0, 0, 0),
    dilation: tuple[int, int] = (1, 1),
    ceil_mode: bool = False,
    count_include_pad: bool = False,
    layout: str = "NCHW",
    out_layout: str | None = None,
    *,
    ty=None,
    loc: _Location = _UNKNOWN_LOC,
) -> Expr:
    """Backward operator of relax.nn.avg_pool2d. All parameters except output_grad is the same as
    relax.nn.avg_pool2d. Returns the gradient w.r.t. data.

    Parameters
    ----------
    output_grad : relax.Expr
      The gradient w.r.t. the result of avg_pool2d.

    Returns
    -------
    result : relax.Expr
      The gradient w.r.t. data.
    """
    if len(padding) == 2:
        padding = tuple(padding) + tuple(padding)
    if len(strides) == 1:
        strides = tuple(strides) * 2
    if len(dilation) == 1:
        dilation = tuple(dilation) * 2
    if len(pool_size) == 1:
        pool_size = tuple(pool_size) * 2
    return _Call(
        "relax.grad.avg_pool2d_backward",
        [output_grad, data],
        attrs=_make_attrs(
            "relax.attrs.Pool2DAttrs",
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            layout=layout,
            out_layout=(layout if out_layout is None else out_layout),
        ),
        loc=loc,
        ty=ty,
    )


def take_backward(
    output_grad: Expr,
    x: Expr,
    indices: Expr,
    axis: int | None = None,
    *,
    ty=None,
    loc: _Location = _UNKNOWN_LOC,
) -> Expr:
    """Backward operator of relax.take. All parameters except output_grad is the same as
    relax.take. Returns the gradient w.r.t. x.

    Parameters
    ----------
    output_grad : relax.Expr
      The gradient w.r.t. the result of take.

    Returns
    -------
    result : relax.Expr
      The gradient w.r.t. x.
    """
    return _Call(
        "relax.grad.take_backward",
        [output_grad, x, indices],
        attrs=_make_attrs("relax.attrs.TakeBackwardAttrs", axis=axis),
        loc=loc,
        ty=ty,
    )  # type: ignore
