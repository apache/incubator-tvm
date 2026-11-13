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
# pylint: disable=redefined-builtin, invalid-name
# ruff: noqa: RUF005
"""PyTorch-like nn.Module API for constructing workloads."""

from collections.abc import Callable
from typing import Any

import numpy as np  # type: ignore

import tvm
import tvm_tirx
from tvm import relax, topi
from tvm.relax.op.grad.grad import end_checkpoint, start_checkpoint


def emit(expr: relax.Expr, name_hint: str = "") -> relax.Var:
    return relax.BlockBuilder.current().emit(expr, name_hint=name_hint)


def emit_te(func: Callable, *args: Any, **kwargs: Any) -> relax.Var:
    return relax.BlockBuilder.current().emit_te(func, *args, **kwargs)


def checkpoint(
    func: Callable, *args: Any, **kwargs: Any
) -> relax.Var | list[relax.Var] | list[Any]:
    """Mark function(*args, **kwargs) should be computed in a checkpointed manner during backward.

    To be specific, args and kwargs will be checkpointed, and func(*args, **kwargs) will be
    recomputed in the backward stage.
    """
    args = [start_checkpoint(v) if isinstance(v, relax.Expr) else v for v in args]
    kwargs = {k: start_checkpoint(v) if isinstance(v, relax.Expr) else v for k, v in kwargs.items()}
    result = func(*args, **kwargs)
    if isinstance(result, list | tuple):
        result = [end_checkpoint(v) if isinstance(v, relax.Expr) else v for v in result]
    else:
        assert isinstance(result, relax.Expr)
        result = end_checkpoint(result)
    return result


def emit_checkpoint(
    func: Callable, *args: Any, **kwargs: Any
) -> relax.Var | list[relax.Var] | list[Any]:
    """Mark function(*args, **kwargs) should be computed in a checkpointed manner during backward.

    To be specific, args and kwargs will be checkpointed, and func(*args, **kwargs) will be
    recomputed in the backward stage.

    This interface will additionally emit the exprs marked with start_checkpoint() and
    end_checkpoint() with suffix "_scp" and "_ecp" respectively, for easily understanding the
    result tvmscript.
    """
    bb = relax.BlockBuilder.current()
    args = [
        bb.emit(start_checkpoint(v), v.name + "_scp") if isinstance(v, relax.Var) else v
        for v in args
    ]
    kwargs = {
        k: bb.emit(start_checkpoint(v), v.name + "_scp") if isinstance(v, relax.Var) else v
        for k, v in kwargs.items()
    }
    result = func(*args, **kwargs)
    if isinstance(result, list | tuple):
        result = list(result)
        for i, v in enumerate(result):
            if isinstance(v, relax.Expr):
                if not isinstance(v, relax.Var):
                    v = bb.emit(v)
                result[i] = bb.emit(end_checkpoint(v), v.name + "_ecp")
    else:
        assert isinstance(result, relax.Expr)
        result_emit = bb.emit(result)
        result = bb.emit(end_checkpoint(result_emit), result_emit.name + "_ecp")

    return result


def emit_checkpoint_sequential(
    functions: list[Callable],
    segments: int | list[int],
    input: relax.Var,
    checkpoint_last: bool = False,
) -> relax.Var | list[relax.Var] | list[Any]:
    """A helper function for checkpointing sequential models. This interface has similar purpose
    as torch.utils.checkpoint.checkpoint_sequential.

    Sequential models execute a list of modules/functions in order (sequentially). Therefore, we
    can divide such a model in various segments and checkpoint each segment. By default, we will
    checkpoint all segments except the last, meaning their inputs will be saved from the forward
    stage and they will be recomputed in the backward stage.

    Parameters
    ----------
    functions : List[Callable]
        The list of functions to be executed sequentially.

    segments : Union[int, List[int]]
        The segments. If segments is int `n`, functions will be evenly divided into `n` segments;
        if segments is a list of ints, it marks the start of every segment.

    input : relax.Var
        The input of the first function.

    checkpoint_last : bool
        Whether the last segment will be checkpointed. Default: False

    Returns
    -------
    output : Union[relax.Var, List[relax.Var], List[Any]]
        The emited output of the last function.
    """
    bb = relax.BlockBuilder.current()

    def run_function(start, end, functions):
        def forward(input):
            for j in range(start, end):
                input = functions[j](input)
            return input

        return forward

    n = len(functions)
    if not isinstance(segments, list):
        segments = list(range(0, n, n // segments)) + [n]
    if segments[-1] != n:
        segments = segments + [n]

    assert len(segments) >= 2

    for i in range(len(segments) - 1):
        if i == len(segments) - 2 and not checkpoint_last:
            input = run_function(segments[i], segments[i + 1], functions)(input)
        else:
            input = emit_checkpoint(run_function(segments[i], segments[i + 1], functions), input)

    assert isinstance(input, relax.Expr)
    if not isinstance(input, relax.Var):
        input = bb.emit(input)
    return input


def _try_unique_name(name: str):
    """Attempt to uniquify the name

    If a `relax.BlockBuilder` is active, use it to return a unique
    name.  Otherwise, return the name itself.

    Two distinct variables in Relax may have identical names.
    However, for user readability, it is convenient to have all names
    be unique within a Relax function.  If a Placeholder or Parameter
    is defined within an active `relax.BlockBuilder`, that context may
    be used to provide a unique name.  Otherwise, allow the duplicate
    names.

    Parameters
    ----------
    name: str

        The variable name

    Returns
    -------
    updated_name: str

        The updated variable name


    """
    block_builder = relax.BlockBuilder.current()
    if block_builder is None:
        return name
    else:
        return block_builder.get_unique_name(name)


class Placeholder(relax.Var):
    """A placeholder variable that can represent model input."""

    def __init__(self, shape: list[Any] | tuple[Any, ...], dtype="float32", name="data"):
        if not isinstance(shape, list | tuple):
            raise TypeError("the shape of Placeholder is expected to be a list or a tuple")
        super().__init__(_try_unique_name(name), relax.TensorType(shape, dtype))


class Parameter(relax.Var):
    """A special kind of relax Var that represents model parameter(weight)."""

    def __init__(self, shape: list[Any] | tuple[Any, ...], dtype="float32", name="param"):
        if not isinstance(shape, list | tuple):
            raise TypeError("the shape of Parameter is expected to be a list or a tuple")

        super().__init__(_try_unique_name(name), relax.TensorType(shape, dtype))


class Module(tvm.relax.frontend.nn.SubroutineMixin):
    """Base class for all model modules.

    A neural network or a layer can subclass this class.

    By default, calls into this module will generate the `relax.Expr`
    representing the output within the current function body.  Setting
    the variable "define_subrouine" to True; either at the
    `nn.Module`, subclass, or instance level; will instead produce a
    subroutine within the same module, which is then called within the
    current function body.

    Example
    -------
    .. code-block:: python

        # Define a linear layer
        class Linear(Module)
            def __init__(self, in_features, out_features, bias=True):
                self.in_features = in_features
                self.out_features = out_features
                self.weight = Parameter((in_features, out_features), name="linear_weight")
                if bias:
                    self.bias = Parameter((out_features,), name="linear_bias")
                else:
                    self.bias = None

            # All submodules should implement forward.
            # Defines the forward computation performed at every call.
            def forward(self, input: relax.Expr) -> relax.Var:
                y = emit_te(topi.matmul, input, self.weight)
                if self.bias is not None:
                    y = emit_te(topi.add, y, self.bias)
                return y
    """

    define_subroutine: bool = False

    def parameters(self) -> list[Parameter]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def forward(self, input: relax.Expr):
        """Define the computation performed at every call."""
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def _unpack_params(value: object) -> list[relax.Var]:
    if isinstance(value, Parameter):
        return [value]
    if isinstance(value, Module):
        return value.parameters()
    if isinstance(value, dict):
        params = []
        for v in value.values():
            params += _unpack_params(v)
        return params
    if isinstance(value, list | tuple):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    return []


def init_params(mod: tvm_tirx.IRModule) -> list[tvm_tirx.runtime.Tensor]:
    """Utility function to initialize model's parameters."""
    shape_dict = {v.name: v.ty.shape for v in mod["main"].params}
    params = []
    for k, v in shape_dict.items():
        if k.startswith("data"):
            continue
        if isinstance(v, relax.ShapeExpr):
            shape = []
            for i in v:
                if isinstance(i, tvm_tirx.ir.IntImm):
                    shape.append(int(i))
                else:
                    raise TypeError("cannot initialize for unknown-shape parameters.")
            params.append(tvm_tirx.runtime.tensor(np.zeros(shape).astype(np.float32)))
        else:
            raise TypeError("cannot initialize for unknown-shape parameters.")
    return params


class Sequential(Module):
    """A sequential container that concatenates modules in it.

    Example
    -------
    .. code-block:: python

        model = nn.Sequential(
                    nn.Conv2d(1, 20, 5),
                    nn.ReLU(),
                    nn.Conv2d(20, 64, 5),
                    nn.ReLU()
                )
    """

    def __init__(self, *modules: Module):
        self.modules = modules

    def forward(self, input: relax.Expr) -> relax.Var:
        for module in self.modules:
            input = module(input)
        return input


class ReLU(Module):
    """Applies the rectified linear unit activation function on the input."""

    def forward(self, input: relax.Expr) -> relax.Var:
        return emit_te(topi.nn.relu, input)


class LogSoftmax(Module):
    """Applies log softmax activation function on the input."""

    def forward(self, input: relax.Expr) -> relax.Var:
        return emit_te(topi.nn.log_softmax, input)


class Linear(Module):
    """Applies a linear transformation to the input data: :math:`y = xA + b`."""

    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter((in_features, out_features), name="linear_weight")
        if bias:
            self.bias = Parameter((out_features,), name="linear_bias")
        else:
            self.bias = None

    def forward(self, input: relax.Expr) -> relax.Var:
        y = emit_te(topi.matmul, input, self.weight)
        if self.bias is not None:
            y = emit_te(topi.add, y, self.bias)
        return y
