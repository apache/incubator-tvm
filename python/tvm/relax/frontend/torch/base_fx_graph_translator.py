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
# ruff: noqa: E731

# pylint: disable=invalid-name, inconsistent-return-statements, unidiomatic-typecheck
# pylint: disable=import-outside-toplevel
"""Base class for PyTorch FX Graph importer."""

import abc
import math
import operator
from collections.abc import Callable
from functools import reduce

import tvm_ffi

import tvm_tirx
from tvm import relax
from tvm_tirx import tirx
from tvm_tirx.ir import PrimType
from tvm_tirx.runtime import DataTypeCode


def _diagonal_einsum_subscripts(ndim: int, dim1: int, dim2: int) -> str:
    """Return explicit einsum subscripts that extract the ``dim1``/``dim2`` diagonal.

    This is the fast-path lowering for :meth:`BaseFXGraphImporter._diagonal`
    when ``offset == 0`` and both diagonal axes have the same extent. The
    non-diagonal axes keep their natural order in the output while the diagonal
    axis is appended last, matching ``aten.diagonal``. Non-diagonal axes are
    labelled ``a``, ``b``, ... and the repeated (diagonal) label is ``z``, e.g.
    an ``N x N`` input with ``dim1 == 0``, ``dim2 == 1`` gives ``"zz->z"``.
    """
    labels = [None] * ndim
    # Non-diagonal axes are labelled from ``a`` to ``y``; ``z`` is reserved for
    # the repeated (diagonal) label so the two never collide.
    letters = iter(ch for ch in "abcdefghijklmnopqrstuvwxyz" if ch != "z")
    for i in range(ndim):
        if i != dim1 and i != dim2:
            labels[i] = next(letters)
    labels[dim1] = "z"
    labels[dim2] = "z"
    leading = "".join(labels[i] for i in range(ndim) if i != dim1 and i != dim2)
    return f"{''.join(labels)}->{leading}z"


class BaseFXGraphImporter(metaclass=abc.ABCMeta):
    """Base class for FX Graph Importer."""

    import torch  # type: ignore
    from torch import fx

    def __init__(self) -> None:
        import torch  # type: ignore
        from torch import fx

        self.env: dict[fx.Node, relax.Expr] = {}
        self.params: dict[torch.Tensor, relax.Expr] = {}
        self.block_builder: relax.BlockBuilder = None
        self.convert_map: dict[torch.nn.Module | str, Callable[[fx.Node], relax.Var]] = (
            self.create_convert_map()
        )

    ########## Utilities ##########

    def update_convert_map(self, custom_convert_map: dict[str, Callable]):
        """Update self.convert_map with custom convert map

        Parameters
        ----------
        custom_convert_map : Dict[str, Callable]
            A custom op conversion map in the same format as self.convert_map
        """

        self.convert_map.update(custom_convert_map)

    @staticmethod
    def _convert_data_type(input_type: str | torch.dtype, env: dict | None = None):
        """converts the PyTorch scalar type input_type to a TVM dtype."""
        import torch  # type: ignore

        if env is not None and input_type in env:
            input_type = env[input_type]

        if isinstance(input_type, PrimType):
            input_type = input_type.dtype
        if hasattr(input_type, "type_code") and hasattr(input_type, "bits"):
            input_type = str(input_type)
        input_type = input_type.lower() if isinstance(input_type, str) else input_type
        # Float types
        if input_type in ["float16", "torch.float16", torch.float16]:
            return "float16"
        elif input_type in ["float", "float32", "torch.float32", torch.float32]:
            return "float32"
        elif input_type in ["float64", "double", "torch.float64", torch.float64]:
            return "float64"
        elif input_type in ["bfloat16", "torch.bfloat16", torch.bfloat16]:
            return "bfloat16"
        # Signed integer types
        elif input_type in ["int8", "torch.int8", torch.int8]:
            return "int8"
        elif input_type in ["int16", "torch.int16", torch.int16]:
            return "int16"
        elif input_type in ["int32", "torch.int32", torch.int32]:
            return "int32"
        elif input_type in ["int64", "torch.int64", torch.int64]:
            return "int64"
        # Unsigned integer types
        elif input_type in ["uint8", "torch.uint8", torch.uint8]:
            return "uint8"
        elif input_type in ["uint16", "torch.uint16", torch.uint16]:
            return "uint16"
        elif input_type in ["uint32", "torch.uint32", torch.uint32]:
            return "uint32"
        elif input_type in ["uint64", "torch.uint64", torch.uint64]:
            return "uint64"
        # Boolean
        elif input_type in ["bool", "torch.bool", torch.bool]:
            return "bool"
        else:
            raise NotImplementedError(f"input_type {input_type} is not handled yet")

    @staticmethod
    def _convert_torch_tensor_to_relax(tensor: torch.Tensor) -> relax.Var:
        tensor = tensor.detach().cpu()
        dtype = BaseFXGraphImporter._convert_data_type(str(tensor.data.dtype))
        return relax.const(tensor.data.numpy(), dtype)

    @staticmethod
    def shape_of(tensor):
        """Get the shape of a tensor."""
        import torch  # type: ignore

        if isinstance(tensor, relax.Expr):
            if not isinstance(tensor.ty, relax.TensorType):
                raise TypeError("The input Expr of shape_of should be a Tensor")
            return tensor.ty.shape
        elif isinstance(tensor, torch.Tensor):
            return tensor.shape
        raise ValueError(f"Unsupported type: {type(tensor)}")

    @staticmethod
    def _static_dim(value):
        """Return ``value`` as a Python int when it is a compile-time constant, else ``None``."""
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        const = getattr(value, "value", None)
        if isinstance(const, int) and not isinstance(const, bool):
            return const
        return None

    def _torch_reshape_chain(self, x, dims):
        """Return the relax reshape targets that reproduce PyTorch's ``dims``.

        PyTorch reads a literal ``0`` in a target shape as a real zero-sized dimension.
        ``relax.op.reshape`` reads it as "copy the corresponding input dimension", which is
        ONNX ``Reshape`` with ``allowzero=0``. A literal ``0`` therefore only survives at a
        position whose input dimension is itself ``0``; anywhere else it silently becomes
        that input dimension.

        A position that does not survive can be written as ``-1``, whose inference yields
        ``0`` for an empty input. Only one ``-1`` is allowed per reshape, so when several
        positions need it the rewrite is split: each step turns one of them into a real
        ``0``, which lets the next step spell that position as a literal. Targets with at
        most one such position - every case seen in practice - stay a single reshape.

        Only a literal is read as a copy, so a symbolic dimension in the target is carried
        through untouched and does not stop the literal zeros beside it from being
        rewritten. ``x.flatten(1, 2)`` on ``(batch, 2, 0, 4)`` asks for ``(batch, 0, 4)``,
        where the zero still has to survive.

        Shapes that do not need the rewrite are returned unchanged. In particular, for a
        non-empty input PyTorch rejects a zero in the target outright, and rewriting it
        would produce a shape rather than surface that error.
        """
        dims = list(dims)
        target = [self._static_dim(d) for d in dims]
        if 0 not in target or -1 in target:
            return [dims]
        shape = self.shape_of(x)
        if shape is None:
            return [dims]
        shape = list(shape)
        current = [self._static_dim(d) for d in shape]
        if 0 not in current:
            # Without a statically known zero the input is not known to be empty, and
            # PyTorch rejects a zero in the target for a non-empty input. A symbolic
            # dimension elsewhere does not change that: one known zero already fixes the
            # element count at zero whatever the symbols turn out to be.
            return [dims]

        steps = []
        while True:
            unusable = [
                i
                for i, t in enumerate(target)
                if t == 0 and not (i < len(current) and current[i] == 0)
            ]
            if not unusable:
                if not steps:
                    # Nothing needed rewriting; emit the target as given.
                    steps.append(dims)
                # Otherwise the last step already produced the target shape, since every
                # remaining zero now sits over an input dimension that is zero as well.
                return steps
            rewritten = unusable[0]
            step, resulting = [], []
            for i, t in enumerate(target):
                if i == rewritten:
                    step.append(-1)
                    resulting.append(0)
                elif t != 0:
                    step.append(dims[i])
                    resulting.append(t)
                elif i < len(current) and current[i] == 0:
                    step.append(0)
                    resulting.append(0)
                else:
                    # Hold this position as it stands -- a symbolic dimension included, since
                    # it is not a literal and so is not read as a copy -- and rewrite it in a
                    # later step, once it has become a real zero.
                    step.append(shape[i] if i < len(shape) else 1)
                    resulting.append(current[i] if i < len(current) else 1)
            steps.append(step)
            shape = [0 if i == rewritten else step[i] for i in range(len(step))]
            current = resulting

    def _emit_torch_reshape(self, x, dims):
        """Emit the reshape(s) giving ``dims`` PyTorch's meaning. See _torch_reshape_chain."""
        for step in self._torch_reshape_chain(x, dims):
            x = self.block_builder.emit(relax.op.reshape(x, step))
        return x

    @staticmethod
    def _promote_common_dtype(
        lhs_dtype: str | None, rhs_dtype: str | None, lhs_ndim: int, rhs_ndim: int
    ) -> str | None:
        """Return the promoted dtype following PyTorch rules, or None if unsupported."""
        import torch  # type: ignore

        lhs_dtype = None if lhs_dtype is None else str(lhs_dtype)
        rhs_dtype = None if rhs_dtype is None else str(rhs_dtype)
        if lhs_dtype is None or rhs_dtype is None or lhs_dtype == rhs_dtype:
            return None

        tvm_to_torch = {
            "float64": torch.float64,
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int64": torch.int64,
            "int32": torch.int32,
            "int16": torch.int16,
            "int8": torch.int8,
            "uint8": torch.uint8,
            "bool": torch.bool,
        }
        torch_to_tvm = {v: k for k, v in tvm_to_torch.items()}

        lhs_torch = tvm_to_torch.get(lhs_dtype)
        rhs_torch = tvm_to_torch.get(rhs_dtype)
        if lhs_torch is None or rhs_torch is None:
            return None

        if lhs_ndim < 0 or rhs_ndim < 0:
            promoted = torch.promote_types(lhs_torch, rhs_torch)
        else:
            promoted = torch.result_type(
                torch.empty(() if lhs_ndim == 0 else (1,), dtype=lhs_torch, device="meta"),
                torch.empty(() if rhs_ndim == 0 else (1,), dtype=rhs_torch, device="meta"),
            )
        return torch_to_tvm.get(promoted, None)

    @staticmethod
    def _is_no_bias(bias):
        """Check if bias represents 'no bias' condition.

        This handles both Python None and relax.op.null_value() expressions
        that might be used to represent missing bias parameters.
        """
        if bias is None:
            return True

        # Check if this is a null_value expression
        if isinstance(bias, relax.Call) and bias.op.name == "relax.null_value":
            return True

        return False

    def retrieve_args(self, node: fx.Node):
        return self._retrieve_args(node.args)

    def _retrieve_args(self, node):
        from torch import fx

        if isinstance(node, fx.Node):
            return self.env[node]
        elif isinstance(node, tuple):
            return tuple(self._retrieve_args(x) for x in node)
        elif isinstance(node, list):
            return [self._retrieve_args(x) for x in node]
        elif isinstance(node, dict):
            return {self._retrieve_args(k): self._retrieve_args(v) for k, v in node.items()}
        elif node is None:
            return None
        else:
            return node

    def _check_unsupported_func_type(self, nodes: list[fx.Node]):
        missing_func_types = list(
            {
                node.target.__name__
                for node in nodes
                if node.op == "call_function" and node.target.__name__ not in self.convert_map
            }
        )
        assert not missing_func_types, f"Unsupported function types {missing_func_types}"

    ########## Unary Ops ##########

    def _unary_op(self, op: Callable) -> Callable:
        from torch import fx

        def convert(node: fx.Node) -> relax.Var:
            return self.block_builder.emit(op(self.env[node.args[0]]))

        return convert

    def _celu(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        alpha = node.args[1] if len(node.args) > 1 else node.kwargs.get("alpha", 1.0)
        dtype = x.ty.dtype

        if isinstance(alpha, int | float):
            alpha = relax.const(alpha, dtype)
        else:
            if not isinstance(alpha, relax.Var):
                alpha = self.block_builder.emit(relax.const(alpha, dtype))

        zero = relax.const(0, dtype)
        # alpha * min(0, exp(x / alpha) - 1) + max(0, x)
        return self.block_builder.emit(
            relax.op.add(
                relax.op.multiply(
                    alpha,
                    relax.op.minimum(
                        zero,
                        relax.op.subtract(
                            relax.op.divide(relax.op.exp(x), alpha), relax.const(1, dtype)
                        ),
                    ),
                ),
                relax.op.nn.relu(x),
            )
        )

    def _clamp(self, node: fx.Node) -> relax.Expr:
        args = self.retrieve_args(node)
        x = args[0]
        a_min = args[1] if len(args) > 1 else node.kwargs.get("min", -math.inf)
        a_max = args[2] if len(args) > 2 else node.kwargs.get("max", math.inf)

        a_min = -math.inf if a_min is None else a_min
        a_max = math.inf if a_max is None else a_max

        # Handle the case where a_min is a tensor
        if not isinstance(a_min, int | float):
            from torch import fx

            if isinstance(a_min, fx.Node):
                # Extract relax Expr (needed for fx.tracer)
                a_min = self.env[a_min]
            assert isinstance(a_min, relax.Expr), (
                f"Unexpected argument type "
                f"passed to torch.clamp/clip: {a_min} with type {type(a_min)}"
            )
            a_min = self.block_builder.emit(relax.op.broadcast_to(a_min, self.shape_of(x)))
            x = self.block_builder.emit(relax.op.maximum(x, a_min))
            a_min = -math.inf

        # Handle the case where a_max is a tensor
        if not isinstance(a_max, int | float):
            from torch import fx

            if isinstance(a_max, fx.Node):
                # Extract relax Expr (needed for fx.tracer)
                a_max = self.env[a_max]
            assert isinstance(a_max, relax.Expr), (
                f"Unexpected argument type "
                f"passed to torch.clamp/clip: {a_max} with type {type(a_max)}"
            )
            a_max = self.block_builder.emit(relax.op.broadcast_to(a_max, self.shape_of(x)))
            x = self.block_builder.emit(relax.op.minimum(x, a_max))
            a_max = math.inf

        return self.block_builder.emit(relax.op.clip(x, a_min, a_max))

    def _clamp_min(self, node: fx.Node) -> relax.Expr:
        args = self.retrieve_args(node)
        x = args[0]
        a_min = args[1] if len(args) > 1 else node.kwargs.get("min", -math.inf)
        a_max = math.inf

        a_min = -math.inf if a_min is None else a_min

        # Handle the case where a_min is a tensor
        if not isinstance(a_min, int | float):
            from torch import fx

            if isinstance(a_min, fx.Node):
                # Extract relax Expr (needed for fx.tracer)
                a_min = self.env[a_min]
            assert isinstance(a_min, relax.Expr), (
                f"Unexpected argument type "
                f"passed to torch.clamp/clip: {a_min} with type {type(a_min)}"
            )
            a_min = self.block_builder.emit(relax.op.broadcast_to(a_min, self.shape_of(x)))
            x = self.block_builder.emit(relax.op.maximum(x, a_min))
            a_min = -math.inf

        return self.block_builder.emit(relax.op.clip(x, a_min, a_max))

    def _clamp_max(self, node: fx.Node) -> relax.Expr:
        args = self.retrieve_args(node)
        x = args[0]
        a_min = -math.inf
        a_max = args[2] if len(args) > 2 else node.kwargs.get("max", math.inf)

        a_max = math.inf if a_max is None else a_max

        # Handle the case where a_max is a tensor
        if not isinstance(a_max, int | float):
            from torch import fx

            if isinstance(a_max, fx.Node):
                # Extract relax Expr (needed for fx.tracer)
                a_max = self.env[a_max]
            assert isinstance(a_max, relax.Expr), (
                f"Unexpected argument type "
                f"passed to torch.clamp/clip: {a_max} with type {type(a_max)}"
            )
            a_max = self.block_builder.emit(relax.op.broadcast_to(a_max, self.shape_of(x)))
            x = self.block_builder.emit(relax.op.minimum(x, a_max))
            a_max = math.inf

        return self.block_builder.emit(relax.op.clip(x, a_min, a_max))

    def _elu(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        alpha = node.args[1] if len(node.args) > 1 else node.kwargs.get("alpha", 1.0)
        dtype = x.ty.dtype

        if isinstance(alpha, int | float):
            alpha = relax.const(-alpha, dtype)
        else:
            if not isinstance(alpha, relax.Var):
                alpha = self.block_builder.emit(relax.const(-alpha, dtype))

        # alpha * ReLU(1 - exp(x)) + ReLU(x)
        return self.block_builder.emit(
            relax.op.add(
                relax.op.multiply(
                    alpha,
                    relax.op.nn.relu(relax.op.subtract(relax.const(1, dtype), relax.op.exp(x))),
                ),
                relax.op.nn.relu(x),
            )
        )

    def _gelu(self, node: fx.Node) -> relax.Expr:
        approximate = node.kwargs.get("approximate", "none")
        if approximate == "none":
            return self.block_builder.emit(relax.op.nn.gelu(self.env[node.args[0]]))
        elif approximate == "tanh":
            return self.block_builder.emit(relax.op.nn.gelu_tanh(self.env[node.args[0]]))
        else:
            raise KeyError(f"Unregonized approximate algorithm for gelu: {approximate}.")

    def _hardsigmoid(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        dtype = x.ty.dtype
        x0 = relax.op.add(x, relax.const(3, dtype))
        x1 = relax.op.clip(x0, 0, 6)
        return self.block_builder.emit(relax.op.divide(x1, relax.const(6, dtype)))

    def _hardswish(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        dtype = x.ty.dtype
        x0 = relax.op.add(x, relax.const(3, dtype))
        x1 = relax.op.clip(x0, 0, 6)
        x2 = relax.op.divide(x1, relax.const(6, dtype))
        return self.block_builder.emit(relax.op.multiply(x, x2))

    def _hardtanh(self, node: fx.Node) -> relax.Expr:
        args = self.retrieve_args(node)
        x = args[0]
        min_val = node.kwargs.get("min_val", -1.0)
        max_val = node.kwargs.get("max_val", 1.0)
        return self.block_builder.emit(relax.op.clip(x, min_val, max_val))

    def _leakyrelu(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        alpha = node.args[1] if len(node.args) > 1 else node.kwargs.get("negative_slope", 0.01)
        return self.block_builder.emit(relax.op.nn.leakyrelu(x, alpha))

    def _log_softmax(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", -1)
        return self.block_builder.emit(relax.op.nn.log_softmax(x, dim))

    def _logical_and(self, node: fx.Node) -> relax.Var:
        lhs = self.env[node.args[0]]
        rhs = self.env[node.args[1]]
        # torch.logical_and accepts any dtype (treating nonzero as True) and returns bool, but
        # relax.op.logical_and requires boolean inputs, so cast non-bool inputs to bool first.
        if lhs.ty.dtype != "bool":
            lhs = self.block_builder.emit(relax.op.astype(lhs, "bool"))
        if rhs.ty.dtype != "bool":
            rhs = self.block_builder.emit(relax.op.astype(rhs, "bool"))
        return self.block_builder.emit(relax.op.logical_and(lhs, rhs))

    def _logical_not(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        # torch.logical_not accepts any dtype (treating nonzero as True) and returns bool, but
        # relax.op.logical_not requires a boolean input, so cast non-bool inputs to bool first.
        if x.ty.dtype != "bool":
            x = self.block_builder.emit(relax.op.astype(x, "bool"))
        return self.block_builder.emit(relax.op.logical_not(x))

    def _logical_or(self, node: fx.Node) -> relax.Var:
        lhs = self.env[node.args[0]]
        rhs = self.env[node.args[1]]
        # torch.logical_or accepts any dtype (treating nonzero as True) and returns bool, but
        # relax.op.logical_or requires boolean inputs, so cast non-bool inputs to bool first.
        if lhs.ty.dtype != "bool":
            lhs = self.block_builder.emit(relax.op.astype(lhs, "bool"))
        if rhs.ty.dtype != "bool":
            rhs = self.block_builder.emit(relax.op.astype(rhs, "bool"))
        return self.block_builder.emit(relax.op.logical_or(lhs, rhs))

    def _logical_xor(self, node: fx.Node) -> relax.Var:
        lhs = self.env[node.args[0]]
        rhs = self.env[node.args[1]]
        # torch.logical_xor accepts any dtype (treating nonzero as True) and returns bool, but
        # relax.op.logical_xor requires boolean inputs, so cast non-bool inputs to bool first.
        if lhs.ty.dtype != "bool":
            lhs = self.block_builder.emit(relax.op.astype(lhs, "bool"))
        if rhs.ty.dtype != "bool":
            rhs = self.block_builder.emit(relax.op.astype(rhs, "bool"))
        return self.block_builder.emit(relax.op.logical_xor(lhs, rhs))

    def _prelu(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        alpha = self.env[node.args[1]]
        axis = 0 if len(x.ty.shape) == 1 else 1
        return self.block_builder.emit(relax.op.nn.prelu(x, alpha, axis))

    def _round(self, node: fx.Node) -> relax.Expr:
        arg = self.env[node.args[0]]
        decimals = node.kwargs.get("decimals", 0)

        if decimals == 0:
            return self.block_builder.emit(relax.op.round(arg))

        # For decimals != 0, round on the exact power-of-10 scale and scale back:
        # round(x * 10^decimals) / 10^decimals. The scaling must always use an
        # integer power of 10: multiply for positive decimals, divide for negative
        # ones. Dividing for negative decimals (instead of multiplying by
        # 10**decimals, i.e. 0.1 / 0.01 / ...) avoids float precision errors such as
        # 25 * 0.1 == 2.5000000000000004 in float64, which would round up to 30
        # instead of 20 for torch.round(25, -1).
        #
        # For float16/bfloat16 inputs the scaling is done in float32 and cast
        # back, because 10**|decimals| can overflow the input range: 10**4 == 10000
        # with 25 * 10000 == 250000 overflows float16 (max 65504) to inf, and 10**5
        # already overflows float16 to inf, turning decimals=5 and -5 into NaN.
        input_dtype = arg.ty.dtype
        dtype = input_dtype
        if dtype in ("float16", "bfloat16"):
            dtype = "float32"
            arg = self.block_builder.emit(relax.op.astype(arg, dtype))

        # Build the scale 10**|decimals| directly in `dtype` instead of as a host
        # Python int. relax.const(10**n, dtype) first materializes 10**n as an
        # unbounded int, which is both wasteful for large n and, once n >= 309, dies
        # in the int-to-float conversion with "OverflowError: int too large to
        # convert to float". PyTorch accepts such decimals (e.g.
        # torch.round(x, decimals=309)) and exports a valid aten.round.decimals
        # node, so importing these programs must not crash. Computing the power as a
        # float and saturating it to inf once it leaves the finite range of `dtype`
        # is exactly what happens when PyTorch evaluates the same power in the input
        # dtype.
        scale_exp = abs(decimals)
        # Largest exponent for which 10**n is still finite in `dtype`.
        # (float16/bfloat16 are upcast to float32 above, so dtype is float32 or
        # float64 here.) Note that `dtype` may be a tvm_ffi.dtype-like object whose
        # repr is "T.float32" instead of a plain str, so select via == rather than
        # indexing a str-keyed dict.
        max_scale_exp = 308 if dtype == "float64" else 38
        if scale_exp > max_scale_exp:
            scale = relax.const(float("inf"), dtype)
        else:
            scale = relax.const(10.0**scale_exp, dtype)

        if decimals > 0:
            scaled = relax.op.multiply(arg, scale)
            rounded = relax.op.round(scaled)
            result = relax.op.divide(rounded, scale)
        else:
            scaled = relax.op.divide(arg, scale)
            rounded = relax.op.round(scaled)
            result = relax.op.multiply(rounded, scale)

        if input_dtype in ("float16", "bfloat16"):
            result = relax.op.astype(result, input_dtype)
        return self.block_builder.emit(result)

    def _softmax(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", -1)
        return self.block_builder.emit(relax.op.nn.softmax(x, dim))

    def _softplus(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        beta = node.args[1] if len(node.args) > 1 else node.kwargs.get("beta", 1.0)
        threshold = node.args[2] if len(node.args) > 2 else node.kwargs.get("threshold", 20.0)
        return self.block_builder.emit(relax.op.nn.softplus(x, beta, threshold))

    def _softsign(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        abs_x = self.block_builder.emit(relax.op.abs(x))
        denom = self.block_builder.emit(relax.op.add(abs_x, relax.const(1.0, dtype="float32")))
        return self.block_builder.emit(relax.op.divide(x, denom))

    def _softshrink(self, node: fx.Node) -> relax.Var:
        """
        Applies the Softshrink activation function in Relax.

        Softshrink(x) =
            x - λ    if x > λ
            x + λ    if x < -λ
            0        otherwise

        Args:
            node (fx.Node): The input node containing the tensor and lambda value.

        Returns:
            relax.Var: The resulting tensor after applying Softshrink.
        """
        args = self.retrieve_args(node)
        x = args[0]
        lambd = relax.const(args[1] if len(args) > 1 else 0.5, x.ty.dtype)

        # Apply Softshrink transformation with masking
        shrink_pos = relax.op.multiply(
            relax.op.subtract(x, lambd),
            relax.op.astype(relax.op.greater(x, lambd), x.ty.dtype.dtype),
        )

        shrink_neg = relax.op.multiply(
            relax.op.add(x, lambd),
            relax.op.astype(relax.op.less(x, relax.op.negative(lambd)), x.ty.dtype.dtype),
        )

        # Combine the positive and negative shrink results
        return self.block_builder.emit(relax.op.add(shrink_pos, shrink_neg))

    def _tril_triu(self, op: Callable) -> Callable:
        from torch import fx

        def convert(node: fx.Node) -> relax.Var:
            x = self.env[node.args[0]]
            k = node.args[1] if len(node.args) > 1 else node.kwargs.get("diagonal", 0)
            assert isinstance(k, int)
            return self.block_builder.emit(op(x, k))

        return convert

    ########## Binary Ops ##########

    def _binary_op(self, relax_op: Callable, intrinsic_op: Callable) -> Callable:
        from torch import fx

        def convert(node: fx.Node) -> relax.Var:
            def promote_binary_op_args(lhs, rhs):
                if isinstance(lhs, relax.Expr) and isinstance(rhs, relax.Expr):
                    lhs_si = getattr(lhs, "ty", None)
                    rhs_si = getattr(rhs, "ty", None)
                    if isinstance(lhs_si, relax.TensorType) and isinstance(
                        rhs_si, relax.TensorType
                    ):
                        target_dtype = self._promote_common_dtype(
                            lhs_si.dtype, rhs_si.dtype, lhs_si.ndim, rhs_si.ndim
                        )
                        if target_dtype is not None:
                            if lhs_si.dtype != target_dtype:
                                lhs = self.block_builder.emit(relax.op.astype(lhs, target_dtype))
                            if rhs_si.dtype != target_dtype:
                                rhs = self.block_builder.emit(relax.op.astype(rhs, target_dtype))
                    return lhs, rhs
                elif isinstance(lhs, relax.Expr):
                    assert isinstance(lhs.ty, relax.TensorType)
                    return lhs, relax.const(rhs, lhs.ty.dtype)
                elif isinstance(rhs, relax.Expr):
                    assert isinstance(rhs.ty, relax.TensorType)
                    return relax.const(lhs, rhs.ty.dtype), rhs
                else:
                    assert False

            def call_binary_op(op, lhs, rhs):
                lhs, rhs = promote_binary_op_args(lhs, rhs)
                return self.block_builder.emit(op(lhs, rhs))

            lhs, rhs = self.retrieve_args(node)

            if (isinstance(lhs, tvm_tirx.ir.Var) and isinstance(lhs.ty, relax.TensorType)) or (
                isinstance(rhs, tvm_tirx.ir.Var) and isinstance(rhs.ty, relax.TensorType)
            ):
                return call_binary_op(relax_op, lhs, rhs)
            elif isinstance(lhs, tvm_tirx.ir.GenericConst) and not isinstance(
                rhs, tvm_tirx.ir.GenericConst
            ):
                return call_binary_op(relax_op, lhs, relax.const(rhs, dtype=lhs.ty.dtype))
            elif isinstance(rhs, tvm_tirx.ir.GenericConst) and not isinstance(
                lhs, tvm_tirx.ir.GenericConst
            ):
                return call_binary_op(relax_op, relax.const(lhs, dtype=rhs.ty.dtype), rhs)
            return intrinsic_op(lhs, rhs)

        return convert

    def _pow(self, node: fx.Node) -> relax.Var:
        lhs, rhs = self.retrieve_args(node)
        # torch integer pow returns an integer tensor, but relax.op.power legalizes to
        # TOPI power which requires floating-point inputs. Decompose an integer base with
        # a constant non-negative integer exponent into repeated multiplication instead.
        if (
            isinstance(lhs, relax.Expr)
            and isinstance(lhs.ty, relax.TensorType)
            and lhs.ty.dtype.matches_code(DataTypeCode.INT, DataTypeCode.UINT)
            and isinstance(rhs, int)
            and not isinstance(rhs, bool)
            and rhs >= 0
        ):
            if rhs == 0:
                return self.block_builder.emit(relax.op.ones_like(lhs))
            result = lhs
            for _ in range(rhs - 1):
                result = self.block_builder.emit(relax.op.multiply(result, lhs))
            return result
        return self._binary_op(relax.op.power, operator.pow)(node)

    def _div(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        inp_1 = args[0]
        inp_2 = args[1]

        # Handle scalar cases
        if isinstance(inp_2, int | float):
            inp_2 = relax.const(inp_2)

        # Get rounding_mode from node kwargs
        rounding_mode = args[2] if len(node.args) > 2 else node.kwargs.get("rounding_mode", None)

        # Perform division based on rounding mode
        if rounding_mode is None:
            # True division (normal float division)
            return self.block_builder.emit(relax.op.divide(inp_1, inp_2))
        elif rounding_mode == "floor":
            # Floor division
            return self.block_builder.emit(relax.op.floor_divide(inp_1, inp_2))
        elif rounding_mode == "trunc":
            # Trunc division: perform true division then truncate
            true_div = self.block_builder.emit(relax.op.divide(inp_1, inp_2))
            return self.block_builder.emit(relax.op.trunc(true_div))
        else:
            raise ValueError(f"Unsupported rounding_mode: {rounding_mode}")

    def _fmod(self, node: fx.Node):
        args = self.retrieve_args(node)
        lhs = args[0]
        rhs = args[1]
        if isinstance(lhs, relax.Expr) and isinstance(rhs, relax.Expr):
            return self.block_builder.emit(relax.op.mod(lhs, rhs))
        elif isinstance(lhs, relax.Expr):
            rhs = relax.const(rhs, lhs.ty.dtype)
        elif isinstance(rhs, relax.Expr):
            lhs = relax.const(lhs, rhs.ty.dtype)
        else:
            assert False
        return self.block_builder.emit(relax.op.mod(lhs, rhs))

    def _rsub(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        lhs = args[0]
        rhs = args[1]

        if isinstance(rhs, int | float):
            rhs = relax.const(rhs)

        return self.block_builder.emit(relax.op.subtract(rhs, lhs))

    def _isin(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        elements = args[0]
        test_elements = args[1]

        expanded_elements = relax.op.expand_dims(elements, axis=-1)
        flattened_test_elements = relax.op.reshape(test_elements, (-1,))

        comparison = relax.op.equal(expanded_elements, flattened_test_elements)
        summed = relax.op.sum(comparison, axis=-1)
        result = relax.op.greater(summed, relax.const(0, dtype=elements.ty.dtype))

        return self.block_builder.emit(result)

    ########## Linear Algebra ##########

    def _linalg_vector_norm(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)

        data = args[0]
        # Default ord=2 if not supplied
        ord_val = args[1] if len(args) > 1 else 2.0
        dim = args[2] if len(args) > 2 else None
        keepdim = args[3] if len(args) > 3 else False

        # If ord_val is a Python float/int, wrap it in a Relax const
        # so that it matches data's dtype.
        dtype = data.ty.dtype
        ord_expr = (
            ord_val if isinstance(ord_val, relax.Expr) else relax.const(float(ord_val), dtype)
        )
        # Reciprocal
        reci_expr = (
            relax.op.divide(relax.const(1.0, dtype), ord_expr)
            if isinstance(ord_val, relax.Expr)
            else relax.const(1.0 / float(ord_val), dtype)
        )

        # abs(data)
        abs_data = self.block_builder.emit(relax.op.abs(data))
        # abs_data^ord
        abs_data_pow = self.block_builder.emit(relax.op.power(abs_data, ord_expr))
        # sum over dim
        reduced = self.block_builder.emit(relax.op.sum(abs_data_pow, dim, keepdims=keepdim))
        # (sum(...))^(1/ord)
        norm_val = self.block_builder.emit(relax.op.power(reduced, reci_expr))

        return norm_val

    ########## Neural Network ##########

    def _adaptive_avg_pool1d(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        output_size = node.args[1] if len(node.args) > 1 else node.kwargs["output_size"]
        # Expand to 3D by adding batch dim if input is 2D
        x_ndim = x.ty.ndim
        if x_ndim == 2:
            x = relax.op.expand_dims(x, axis=0)

        result = self.block_builder.emit(
            relax.op.nn.adaptive_avg_pool1d(x, output_size, layout="NCW")
        )
        # Remove added batch dim from result
        if x_ndim == 2:
            result = relax.op.squeeze(result, axis=[0])
        return result

    def _adaptive_avg_pool2d(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        output_size = node.args[1]
        # Expand to 4D by adding batch dim if input is 3D
        x_ndim = x.ty.ndim
        if x_ndim == 3:
            x = relax.op.expand_dims(x, axis=0)

        result = self.block_builder.emit(
            relax.op.nn.adaptive_avg_pool2d(x, output_size, layout="NCHW")
        )
        # Remove added batch dim from result
        if x_ndim == 3:
            result = relax.op.squeeze(result, axis=[0])
        return result

    def _adaptive_avg_pool3d(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        output_size = node.args[1]
        # Expand to 5D by adding batch dim if input is 4D
        x_ndim = x.ty.ndim
        if x_ndim == 4:
            x = relax.op.expand_dims(x, axis=0)

        result = self.block_builder.emit(
            relax.op.nn.adaptive_avg_pool3d(x, output_size, layout="NCDHW")
        )
        # Remove added batch dim from result
        if x_ndim == 4:
            result = relax.op.squeeze(result, axis=[0])
        return result

    def _addmm(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        y = self.env[node.args[1]]
        z = self.env[node.args[2]]
        alpha = node.kwargs.get("alpha", 1)
        beta = node.kwargs.get("beta", 1)

        res = None
        if alpha != 0:
            res = self.block_builder.emit(relax.op.linear_algebra.matmul(y, z, out_dtype="float32"))
            if alpha != 1:
                dtype = res.ty.dtype
                res = self.block_builder.emit(relax.op.multiply(res, relax.const(alpha, dtype)))
        if beta != 0:
            dtype = x.ty.dtype
            if beta != 1:
                bias = self.block_builder.emit(relax.op.multiply(x, relax.const(beta, dtype)))
            else:
                bias = x
            res = bias if res is None else self.block_builder.emit(relax.op.add(bias, res))
        return res

    def _avg_pool1d_impl(
        self,
        x: relax.Expr,
        kernel_size: int | tuple[int] = 1,
        stride: int | tuple[int] | None = None,
        padding: int | None = 0,
        ceil_mode: bool | None = False,
        count_include_pad: bool | None = True,
    ) -> relax.Var:
        # Expand to 3D by adding batch dim if input is 2D
        x_ndim = x.ty.ndim
        if x_ndim == 2:
            x = relax.op.expand_dims(x, axis=0)
        stride = kernel_size if stride is None or stride == [] else stride

        result = self.block_builder.emit(
            relax.op.nn.avg_pool1d(
                x,
                pool_size=kernel_size,
                strides=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
                layout="NCW",
            )
        )
        # Remove added batch dim from result
        if x_ndim == 2:
            result = relax.op.squeeze(result, axis=[0])
        return result

    def _avg_pool1d(self, node: fx.Node) -> relax.Var:
        args, kwargs = node.normalized_arguments(node)
        x = self.env[args[0]]
        kernel_size = args[1] if len(args) > 1 else kwargs["kernel_size"]
        stride = args[2] if len(args) > 2 else kwargs.get("stride", None)
        padding = args[3] if len(args) > 3 else kwargs.get("padding", 0)
        ceil_mode = args[4] if len(args) > 4 else kwargs.get("ceil_mode", False)
        count_include_pad = args[5] if len(args) > 5 else kwargs.get("count_include_pad", True)

        return self._avg_pool1d_impl(x, kernel_size, stride, padding, ceil_mode, count_include_pad)

    def _avg_pool2d_impl(
        self,
        x: relax.Expr,
        kernel_size: int | tuple[int, int] = (1, 1),
        stride: int | tuple[int, int] | None = None,
        padding: int | None = 0,
        ceil_mode: bool | None = False,
        count_include_pad: bool | None = True,
    ) -> relax.Var:
        # Expand to 4D by adding batch dim if input is 3D
        x_ndim = x.ty.ndim
        if x_ndim == 3:
            x = relax.op.expand_dims(x, axis=0)
        stride = kernel_size if stride is None or stride == [] else stride

        result = self.block_builder.emit(
            relax.op.nn.avg_pool2d(
                x,
                pool_size=kernel_size,
                strides=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
                layout="NCHW",
            )
        )
        # Remove added batch dim from result
        if x_ndim == 3:
            result = relax.op.squeeze(result, axis=[0])
        return result

    def _avg_pool2d(self, node: fx.Node) -> relax.Var:
        args, kwargs = node.normalized_arguments(node)
        x = self.env[args[0]]
        kernel_size = args[1] if len(args) > 1 else kwargs["kernel_size"]
        stride = args[2] if len(args) > 2 else kwargs.get("stride", None)
        padding = args[3] if len(args) > 3 else kwargs.get("padding", 0)
        ceil_mode = args[4] if len(args) > 4 else kwargs.get("ceil_mode", False)
        count_include_pad = args[5] if len(args) > 5 else kwargs.get("count_include_pad", True)
        return self._avg_pool2d_impl(x, kernel_size, stride, padding, ceil_mode, count_include_pad)

    def _avg_pool3d_impl(
        self,
        x: relax.Expr,
        kernel_size: int | tuple[int, int, int] = (1, 1, 1),
        stride: int | tuple[int, int, int] | None = None,
        padding: int | None = 0,
        ceil_mode: bool | None = False,
        count_include_pad: bool | None = True,
    ) -> relax.Var:
        # Expand to 5D by adding batch dim if input is 4D
        x_ndim = x.ty.ndim
        if x_ndim == 4:
            x = relax.op.expand_dims(x, axis=0)
        stride = kernel_size if stride is None or stride == [] else stride

        result = self.block_builder.emit(
            relax.op.nn.avg_pool3d(
                x,
                pool_size=kernel_size,
                strides=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
                layout="NCDHW",
            )
        )
        # Remove added batch dim from result
        if x_ndim == 4:
            result = relax.op.squeeze(result, axis=[0])
        return result

    def _avg_pool3d(self, node: fx.Node) -> relax.Var:
        args, kwargs = node.normalized_arguments(node)
        x = self.env[args[0]]
        kernel_size = args[1] if len(args) > 1 else kwargs["kernel_size"]
        stride = args[2] if len(args) > 2 else kwargs.get("stride", None)
        padding = args[3] if len(args) > 3 else kwargs.get("padding", 0)
        ceil_mode = args[4] if len(args) > 4 else kwargs.get("ceil_mode", False)
        count_include_pad = args[5] if len(args) > 5 else kwargs.get("count_include_pad", True)

        return self._avg_pool3d_impl(x, kernel_size, stride, padding, ceil_mode, count_include_pad)

    def _baddbmm(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        batch1 = self.env[node.args[1]]
        batch2 = self.env[node.args[2]]
        alpha = node.kwargs.get("alpha", 1)
        beta = node.kwargs.get("beta", 1)

        res = None
        if alpha != 0:
            res = self.block_builder.emit(relax.op.matmul(batch1, batch2))
            if alpha != 1:
                dtype = res.ty.dtype
                res = self.block_builder.emit(relax.op.multiply(res, relax.const(alpha, dtype)))
        if beta != 0:
            dtype = x.ty.dtype
            if beta != 1:
                bias = self.block_builder.emit(relax.op.multiply(x, relax.const(beta, dtype)))
            else:
                bias = x
            res = bias if res is None else self.block_builder.emit(relax.op.add(res, bias))
        return res

    def _conv_transpose1d_impl(
        self,
        x: relax.Expr,
        weight: relax.Expr,
        bias: relax.Expr | None,
        strides: tuple | None,
        padding: tuple | None,
        dilation: tuple | None,
        groups: tuple | None,
        output_padding: tuple | None,
    ) -> relax.Var:
        conv1d_transpose = self.block_builder.emit(
            relax.op.nn.conv1d_transpose(
                x,
                weight,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                output_padding=output_padding,
                data_layout="NCW",
                kernel_layout="IOW",
                out_dtype="float32",
            )
        )

        if self._is_no_bias(bias):
            return conv1d_transpose

        assert len(self.shape_of(bias)) == 1
        bias = relax.op.reshape(bias, (1, -1, 1))
        return self.block_builder.emit(relax.op.add(conv1d_transpose, bias))

    def _conv_transpose1d(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else None
        stride = args[3] if len(args) > 3 else 1
        padding = args[4] if len(args) > 4 else 0
        output_padding = args[5] if len(args) > 5 else 0
        groups = args[6] if len(args) > 6 else 1
        dilation = args[7] if len(args) > 7 else 1
        return self._conv_transpose1d_impl(
            x,
            weight,
            bias=bias,
            strides=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            output_padding=output_padding,
        )

    def _conv_transpose2d_impl(
        self,
        x: relax.Expr,
        weight: relax.Expr,
        bias: relax.Expr | None,
        strides: tuple | None,
        padding: tuple | None,
        dilation: tuple | None,
        groups: tuple | None,
        output_padding: tuple | None,
    ) -> relax.Var:
        conv2d_transpose = self.block_builder.emit(
            relax.op.nn.conv2d_transpose(
                x,
                weight,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                output_padding=output_padding,
                data_layout="NCHW",
                kernel_layout="IOHW",
                out_dtype="float32",
            )
        )

        if self._is_no_bias(bias):
            return conv2d_transpose

        assert len(self.shape_of(bias)) == 1
        bias = relax.op.reshape(bias, (1, -1, 1, 1))
        return self.block_builder.emit(relax.op.add(conv2d_transpose, bias))

    def _conv_transpose2d(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else None
        stride = args[3] if len(args) > 3 else 1
        padding = args[4] if len(args) > 4 else 0
        output_padding = args[5] if len(args) > 5 else 0
        groups = args[6] if len(args) > 6 else 1
        dilation = args[7] if len(args) > 7 else 1
        return self._conv_transpose2d_impl(
            x,
            weight,
            bias=bias,
            strides=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            output_padding=output_padding,
        )

    def _conv1d_impl(
        self,
        x: relax.Expr,
        weight: relax.Expr,
        bias: relax.Expr | None,
        strides: tuple | None,
        padding: tuple | None,
        dilation: tuple | None,
        groups: tuple | None,
    ) -> relax.Var:
        conv1d = self.block_builder.emit(
            relax.op.nn.conv1d(
                x,
                weight,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                data_layout="NCW",
                kernel_layout="OIW",
                out_dtype="float32",
            )
        )

        if self._is_no_bias(bias):
            return conv1d
        assert len(self.shape_of(bias)) == 1
        bias = relax.op.reshape(bias, (1, -1, 1))
        return self.block_builder.emit(relax.op.add(conv1d, bias))

    def _conv1d(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else None
        stride = args[3] if len(args) > 3 else 1
        padding = args[4] if len(args) > 4 else 0
        dilation = args[5] if len(args) > 5 else 1
        groups = args[6] if len(args) > 6 else 1
        return self._conv1d_impl(
            x,
            weight,
            bias=bias,
            strides=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    def _conv2d_impl(
        self,
        x: relax.Expr,
        weight: relax.Expr,
        bias: relax.Expr | None,
        strides: tuple | None,
        padding: tuple | None,
        dilation: tuple | None,
        groups: tuple | None,
    ):
        conv2d = self.block_builder.emit(
            relax.op.nn.conv2d(
                x,
                weight,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                data_layout="NCHW",
                kernel_layout="OIHW",
                out_dtype="float32",
            )
        )

        if self._is_no_bias(bias):
            return conv2d
        assert len(self.shape_of(bias)) == 1
        bias = relax.op.reshape(bias, (1, -1, 1, 1))
        return self.block_builder.emit(relax.op.add(conv2d, bias))

    def _conv2d(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else None
        stride = args[3] if len(args) > 3 else 1
        padding = args[4] if len(args) > 4 else 0
        dilation = args[5] if len(args) > 5 else 1
        groups = args[6] if len(args) > 6 else 1
        return self._conv2d_impl(
            x,
            weight,
            bias=bias,
            strides=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    def _conv3d_impl(
        self,
        x: relax.Expr,
        weight: relax.Expr,
        bias: relax.Expr | None,
        strides: tuple | None,
        padding: tuple | None,
        dilation: tuple | None,
        groups: tuple | None,
    ):
        conv3d = self.block_builder.emit(
            relax.op.nn.conv3d(
                x,
                weight,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                data_layout="NCDHW",
                kernel_layout="OIDHW",
                out_dtype="float32",
            )
        )

        if self._is_no_bias(bias):
            return conv3d
        assert len(self.shape_of(bias)) == 1
        bias = relax.op.reshape(bias, (1, -1, 1, 1, 1))
        return self.block_builder.emit(relax.op.add(conv3d, bias))

    def _conv3d(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else None
        stride = args[3] if len(args) > 3 else 1
        padding = args[4] if len(args) > 4 else 0
        dilation = args[5] if len(args) > 5 else 1
        groups = args[6] if len(args) > 6 else 1
        return self._conv3d_impl(
            x,
            weight,
            bias=bias,
            strides=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    def _convolution(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else None
        stride = args[3] if len(args) > 3 else 1
        padding = args[4] if len(args) > 4 else 0
        dilation = args[5] if len(args) > 5 else 1
        transposed = args[6] if len(args) > 6 else False
        output_padding = args[7] if len(args) > 7 else 0
        groups = args[8] if len(args) > 8 else 1

        input_shape = self.shape_of(x)
        ndim = len(input_shape)

        if transposed:
            if ndim == 3:  # 1D convolution (N, C, W)
                return self._conv_transpose1d_impl(
                    x,
                    weight,
                    bias=bias,
                    strides=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    output_padding=output_padding,
                )
            elif ndim == 4:  # 2D convolution (N, C, H, W)
                return self._conv_transpose2d_impl(
                    x,
                    weight,
                    bias=bias,
                    strides=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    output_padding=output_padding,
                )
            else:
                raise ValueError(f"Unsupported transposed convolution dimensionality: {ndim}")
        else:
            if ndim == 3:  # 1D convolution (N, C, W)
                return self._conv1d_impl(
                    x,
                    weight,
                    bias=bias,
                    strides=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
            elif ndim == 4:  # 2D convolution (N, C, H, W)
                return self._conv2d_impl(
                    x,
                    weight,
                    bias=bias,
                    strides=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
            elif ndim == 5:  # 3D convolution (N, C, D, H, W)
                return self._conv3d_impl(
                    x,
                    weight,
                    bias=bias,
                    strides=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
            else:
                raise ValueError(f"Unsupported convolution dimensionality: {ndim}")

    def _cross_entropy_loss(
        self,
        preds: relax.Expr,
        targets: relax.Expr,
        weights: relax.Expr | None,
        reduction: str,
        ignore_index: int,
    ) -> relax.Expr:
        log_probs = relax.op.nn.log_softmax(preds)
        return self.block_builder.emit(
            relax.op.nn.nll_loss(
                log_probs,
                targets,
                weights,
                reduction,
                ignore_index,
            )
        )

    def _einsum(self, node: fx.Node) -> relax.Var:
        import torch  # type: ignore

        args = self.retrieve_args(node)
        operands = args[1] if isinstance(args[1], torch.Size | tuple | list) else args[1:]
        return self.block_builder.emit(relax.op.einsum(operands, args[0]))

    def _diagonal(self, node: fx.Node) -> relax.Var:
        """Convert ``aten.diagonal`` / ``torch.diagonal`` to Relax.

        ``diagonal(input, offset=0, dim1=0, dim2=1)`` extracts the elements
        ``input[..., i, i + offset]`` along the ``dim1`` / ``dim2`` axes. It
        shows up in the exported graph through ``run_decompositions`` of
        ``torch.einsum`` with repeated subscripts (e.g. ``"ii->i"``,
        ``"ii->"``, ``"...ii->...i"``), which lower to an ``aten.diagonal``
        followed by a ``sum`` reduction.

        We lower it as: when ``offset == 0`` and both diagonal axes have equal
        extent (the common ``torch.einsum("ii->i")`` case), a single einsum
        whose subscript repeats one label over the two axes extracts the
        diagonal directly, so no full-size permute / slice intermediate is
        materialized. Otherwise we permute ``dim1`` / ``dim2`` to the trailing
        two axes, slice each trailing axis to the diagonal length (min of the
        two extents, adjusted by ``offset``), and take the diagonal with an
        einsum contraction ``...zz->...z`` (the repeated ``z`` label runs over
        both trailing axes simultaneously).
        """

        args = self.retrieve_args(node)
        x = args[0]
        offset = args[1] if len(args) > 1 else node.kwargs.get("offset", 0)
        dim1 = args[2] if len(args) > 2 else node.kwargs.get("dim1", 0)
        dim2 = args[3] if len(args) > 3 else node.kwargs.get("dim2", 1)

        shape = self.shape_of(x)
        ndim = len(shape.values)
        dim1 = dim1 if dim1 >= 0 else ndim + dim1
        dim2 = dim2 if dim2 >= 0 else ndim + dim2
        if dim1 == dim2:
            raise ValueError(f"diagonal requires dim1 != dim2, got {dim1} == {dim2}")

        offset = int(offset)

        n = shape.values[dim1]
        m = shape.values[dim2]
        # Fast path for the common ``offset == 0`` case with equal extents on the
        # diagonal axes (e.g. torch.einsum("ii->i") on an N x N input). The
        # diagonal can be read with a single einsum that repeats one subscript
        # label over the two axes (``relax.op.einsum`` lowers it to one O(N)
        # loop), avoiding the identity permute / strided-slice that would each
        # materialize a full-size O(N^2) copy. Non-diagonal axes must still fit
        # in the single-letter einsum label alphabet.
        if offset == 0 and ndim - 2 <= 25 and tvm_ffi.structural_equal(n, m):
            subscripts = _diagonal_einsum_subscripts(ndim, dim1, dim2)
            return self.block_builder.emit(relax.op.einsum([x], subscripts))

        # Move dim1, dim2 to the trailing two axes.
        perm = [i for i in range(ndim) if i != dim1 and i != dim2] + [dim1, dim2]
        permuted = self.block_builder.emit(relax.op.permute_dims(x, perm))

        if offset >= 0:
            diag_len = tirx.max(0, tirx.min(n, m - offset))
            begin1, end1 = 0, diag_len
            begin2, end2 = offset, offset + diag_len
        else:
            diag_len = tirx.max(0, tirx.min(n + offset, m))
            begin1, end1 = -offset, -offset + diag_len
            begin2, end2 = 0, diag_len

        # Crop both diagonal axes to the diagonal length so the einsum ``z``
        # label sees equal extents on both trailing axes.
        cropped = self.block_builder.emit(
            relax.op.strided_slice(
                permuted, axes=[ndim - 2], begin=[begin1], end=[end1], strides=[1]
            )
        )
        cropped = self.block_builder.emit(
            relax.op.strided_slice(
                cropped, axes=[ndim - 1], begin=[begin2], end=[end2], strides=[1]
            )
        )

        # ``...zz -> ...z``: keep every leading axis, contract the diagonal pair.
        return self.block_builder.emit(relax.op.einsum([cropped], "...zz->...z"))

    def _embedding_impl(
        self,
        x,
        weight,
    ) -> relax.Var:
        x = self.block_builder.emit(relax.op.astype(x, "int32"))

        ndim = x.ty.ndim
        if ndim == 1:
            return self.block_builder.emit(relax.op.take(weight, x, axis=0))
        else:
            x_shape = x.ty.shape.values
            emb_size = weight.ty.shape.values[-1]
            x = self.block_builder.emit(relax.op.reshape(x, shape=[-1]))
            embedding = self.block_builder.emit(relax.op.take(weight, x, axis=0))
            return self.block_builder.emit(relax.op.reshape(embedding, [*x_shape, emb_size]))

    def _layer_norm_impl(self, x, gamma, beta, eps, normalized_shape) -> relax.Var:
        import numpy as np  # type: ignore
        from torch.fx.immutable_collections import immutable_list

        if isinstance(normalized_shape, immutable_list | tuple):
            normalized_shape = tuple(normalized_shape)
        else:
            try:
                normalized_shape = self.env[normalized_shape]
            except TypeError:
                normalized_shape = tuple(normalized_shape)

        dim_num = len(normalized_shape)
        axes = list(range(-dim_num, 0))

        if gamma is None:
            shape_tuple = [int(s) for s in normalized_shape]
            gamma = relax.const(np.ones(shape_tuple), x.ty.dtype)
        if beta is None:
            shape_tuple = [int(s) for s in normalized_shape]
            beta = relax.const(np.zeros(shape_tuple), x.ty.dtype)

        return self.block_builder.emit(
            relax.op.nn.layer_norm(
                x,
                gamma,
                beta,
                axes=axes,
                epsilon=eps,
            )
        )

    def _layer_norm(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        normalized_shape = node.args[1]
        gamma = self.env[node.args[2]] if len(node.args) > 2 else None
        beta = self.env[node.args[3]] if len(node.args) > 3 else None
        eps = node.args[4] if len(node.args) > 4 else 1e-05
        return self._layer_norm_impl(x, gamma, beta, eps, normalized_shape)

    def _layer_norm_module(self, node: fx.Node) -> relax.Var:
        import torch  # type: ignore

        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        normalized_shape = module.normalized_shape
        if module.elementwise_affine:
            gamma = self.params[module.weight]
            beta = self.params[module.bias]
        else:
            gamma = relax.const(torch.ones_like(module.normalized_shape), x.ty.dtype)
            beta = relax.const(torch.zeros_like(module.normalized_shape), x.ty.dtype)
        eps = module.eps
        return self._layer_norm_impl(x, gamma, beta, eps, normalized_shape)

    def _linear(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else None
        return self.block_builder.emit(relax.op.linear(x, weight, bias, "float32"))

    def _max_pool1d_impl(
        self,
        x: relax.Expr,
        kernel_size: int | tuple[int] = 1,
        stride: int | tuple[int] | None = None,
        padding: int | None = 0,
        dilation: int | None = 1,
        ceil_mode: bool | None = False,
    ) -> relax.Var:
        # Expand to 3D by adding batch dim if input is 2D
        x_ndim = x.ty.ndim
        if x_ndim == 2:
            x = relax.op.expand_dims(x, axis=0)

        stride = kernel_size if stride is None else stride

        result = self.block_builder.emit(
            relax.op.nn.max_pool1d(
                x,
                pool_size=kernel_size,
                strides=stride,
                padding=padding,
                dilation=dilation,
                ceil_mode=ceil_mode,
                layout="NCW",
            )
        )

        # Remove added batch dim from result
        if x_ndim == 2:
            result = relax.op.squeeze(result, axis=[0])
        return result

    def _max_pool1d(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        kernel_size = args[1]
        stride = args[2] if len(args) > 2 else None
        padding = args[3] if len(args) > 3 else 0
        dilation = args[4] if len(args) > 4 else 1
        ceil_mode = args[5] if len(args) > 5 else False

        return self._max_pool1d_impl(x, kernel_size, stride, padding, dilation, ceil_mode)

    def _max_pool2d_impl(
        self,
        x: relax.Expr,
        kernel_size: int | tuple[int, int] = (1, 1),
        stride: int | tuple[int, int] | None = None,
        padding: int | None = 0,
        dilation: int | None = 1,
        ceil_mode: bool | None = False,
    ) -> relax.Var:
        # Expand to 4D by adding batch dim if input is 3D
        x_ndim = x.ty.ndim
        if x_ndim == 3:
            x = relax.op.expand_dims(x, axis=0)

        stride = kernel_size if stride is None else stride

        result = self.block_builder.emit(
            relax.op.nn.max_pool2d(
                x,
                pool_size=kernel_size,
                strides=stride,
                padding=padding,
                dilation=dilation,
                ceil_mode=ceil_mode,
                layout="NCHW",
            )
        )

        # Remove added batch dim from result
        if x_ndim == 3:
            result = relax.op.squeeze(result, axis=[0])
        return result

    def _max_pool2d(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        kernel_size = args[1]
        stride = args[2] if len(args) > 2 else None
        padding = args[3] if len(args) > 3 else 0
        dilation = args[4] if len(args) > 4 else 1
        ceil_mode = args[5] if len(args) > 5 else False

        return self._max_pool2d_impl(x, kernel_size, stride, padding, dilation, ceil_mode)

    def _max_pool3d_impl(
        self,
        x: relax.Expr,
        kernel_size: int | tuple[int, int, int] = (1, 1, 1),
        stride: int | tuple[int, int, int] | None = None,
        padding: int | None = 0,
        dilation: int | None = 1,
        ceil_mode: bool | None = False,
    ) -> relax.Var:
        # Expand to 5D by adding batch dim if input is 4D
        x_ndim = x.ty.ndim
        if x_ndim == 4:
            x = relax.op.expand_dims(x, axis=0)

        stride = kernel_size if stride is None else stride

        result = self.block_builder.emit(
            relax.op.nn.max_pool3d(
                x,
                pool_size=kernel_size,
                strides=stride,
                padding=padding,
                dilation=dilation,
                ceil_mode=ceil_mode,
                layout="NCDHW",
            )
        )

        # Remove added batch dim from result
        if x_ndim == 4:
            result = relax.op.squeeze(result, axis=[0])
        return result

    def _max_pool3d(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        kernel_size = args[1]
        stride = args[2] if len(args) > 2 else None
        padding = args[3] if len(args) > 3 else 0
        dilation = args[4] if len(args) > 4 else 1
        ceil_mode = args[5] if len(args) > 5 else False
        return self._max_pool3d_impl(x, kernel_size, stride, padding, dilation, ceil_mode)

    def _max_pool1d_with_indices(self, node: fx.Node) -> relax.Var:
        # max_pool1d_with_indices returns (output, indices)
        # We only compute the output and create a placeholder for indices
        args = self.retrieve_args(node)
        x = args[0]
        kernel_size = args[1]
        stride = args[2] if len(args) > 2 else None
        padding = args[3] if len(args) > 3 else 0
        dilation = args[4] if len(args) > 4 else 1
        ceil_mode = args[5] if len(args) > 5 else False

        output = self._max_pool1d_impl(x, kernel_size, stride, padding, dilation, ceil_mode)
        # Create a placeholder for indices (empty tensor with same shape as output)
        indices = relax.op.zeros_like(output)
        return self.block_builder.emit(relax.Tuple([output, indices]))

    def _max_pool2d_with_indices(self, node: fx.Node) -> relax.Var:
        # max_pool2d_with_indices returns (output, indices)
        # We only compute the output and create a placeholder for indices
        args = self.retrieve_args(node)
        x = args[0]
        kernel_size = args[1]
        stride = args[2] if len(args) > 2 else None
        padding = args[3] if len(args) > 3 else 0
        dilation = args[4] if len(args) > 4 else 1
        ceil_mode = args[5] if len(args) > 5 else False

        output = self._max_pool2d_impl(x, kernel_size, stride, padding, dilation, ceil_mode)
        # Create a placeholder for indices (empty tensor with same shape as output)
        indices = relax.op.zeros_like(output)
        return self.block_builder.emit(relax.Tuple([output, indices]))

    def _max_pool3d_with_indices(self, node: fx.Node) -> relax.Var:
        # max_pool3d_with_indices returns (output, indices)
        # We only compute the output and create a placeholder for indices
        args = self.retrieve_args(node)
        x = args[0]
        kernel_size = args[1]
        stride = args[2] if len(args) > 2 else None
        padding = args[3] if len(args) > 3 else 0
        dilation = args[4] if len(args) > 4 else 1
        ceil_mode = args[5] if len(args) > 5 else False

        output = self._max_pool3d_impl(x, kernel_size, stride, padding, dilation, ceil_mode)
        # Create a placeholder for indices (empty tensor with same shape as output)
        indices = relax.op.zeros_like(output)
        return self.block_builder.emit(relax.Tuple([output, indices]))

    def _pad(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        pad = node.args[1]
        mode = node.args[2] if len(node.args) > 2 else node.kwargs.get("mode", "constant")
        value = node.args[3] if len(node.args) > 3 else node.kwargs.get("value", 0.0)
        value = 0.0 if value is None else value

        # Calculate symmetric padding width for each dimension
        # and applying them in reverse order to match the input dimensions.
        input_ndim = x.ty.ndim
        pad_width = [0] * (input_ndim * 2)
        pad_pairs = [pad[i : i + 2] for i in range(0, len(pad), 2)]
        reversed_pairs = list(reversed(pad_pairs))
        flattened = [value for pair in reversed_pairs for value in pair]
        pad_width[-len(flattened) :] = flattened

        return self.block_builder.emit(relax.op.nn.pad(x, pad_width, mode, value))

    def _constant_pad_nd(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        pad = node.args[1]
        value = node.args[2] if len(node.args) > 2 else node.kwargs.get("value", 0.0)
        value = 0.0 if value is None else value

        # Calculate symmetric padding width for each dimension
        # and applying them in reverse order to match the input dimensions.
        input_ndim = x.ty.ndim
        pad_width = [0] * (input_ndim * 2)
        pad_pairs = [pad[i : i + 2] for i in range(0, len(pad), 2)]
        reversed_pairs = list(reversed(pad_pairs))
        flattened = [v for pair in reversed_pairs for v in pair]
        pad_width[-len(flattened) :] = flattened

        return self.block_builder.emit(relax.op.nn.pad(x, pad_width, "constant", value))

    def _pixel_shuffle(self, node: fx.Node) -> relax.Var:
        data = self.env[node.args[0]]
        upscale_factor = node.args[1]
        assert isinstance(upscale_factor, int), (
            "PixelShuffle only accepts an integer upscale_factor."
        )

        return self.block_builder.emit(relax.op.nn.pixel_shuffle(data, upscale_factor))

    def _scaled_dot_product_attention(self, node: fx.Node) -> relax.Var:
        query_tensor = self.env[node.args[0]]
        key_tensor = self.env[node.args[1]]
        value_tensor = self.env[node.args[2]]

        # Check the dimensionality of the input tensors
        query_ndim = len(query_tensor.ty.shape)

        # TVM's nn.attention requires 4D inputs in format (batch, num_heads, seq_len, head_dim)
        # For 2D inputs (seq_len, head_dim), we need to reshape to 4D first
        if query_ndim == 2:
            # 2D input: (seq_len, head_dim) -> expand to (1, 1, seq_len, head_dim)
            # Add batch dimension at axis 0
            query_3d = self.block_builder.emit(relax.op.expand_dims(query_tensor, axis=0))
            key_3d = self.block_builder.emit(relax.op.expand_dims(key_tensor, axis=0))
            value_3d = self.block_builder.emit(relax.op.expand_dims(value_tensor, axis=0))
            # Add num_heads dimension at axis 1
            query = self.block_builder.emit(relax.op.expand_dims(query_3d, axis=1))
            key = self.block_builder.emit(relax.op.expand_dims(key_3d, axis=1))
            value = self.block_builder.emit(relax.op.expand_dims(value_3d, axis=1))

            # No permutation needed for 2D inputs after expanding to 4D
            # After attention, squeeze back to 2D: (1, 1, seq_len, head_dim) -> (seq_len, head_dim)
            def transpose_and_reshape_back(tensor):
                # Squeeze batch and num_heads dimensions
                return self.block_builder.emit(relax.op.squeeze(tensor, axis=[0, 1]))

        elif query_ndim == 4:
            # 4D input: (batch, seq_len, num_heads, head_dim)
            # -> (batch, num_heads, seq_len, head_dim)
            transpose_S_H = lambda tensor: relax.op.permute_dims(tensor, [0, 2, 1, 3])
            query = self.block_builder.emit(transpose_S_H(query_tensor))
            key = self.block_builder.emit(transpose_S_H(key_tensor))
            value = self.block_builder.emit(transpose_S_H(value_tensor))

            # For 4D, transpose back after attention
            def transpose_and_reshape_back(tensor):
                return self.block_builder.emit(transpose_S_H(tensor))

        else:
            raise ValueError(
                f"scaled_dot_product_attention expects 2D or 4D inputs, but got {query_ndim}D input"
            )

        attn_mask = node.args[3] if len(node.args) > 3 else node.kwargs.get("attn_mask", None)
        dropout_p = node.args[4] if len(node.args) > 4 else node.kwargs.get("dropout_p", 0.0)
        assert dropout_p == 0.0, "Dropout is not supported"
        is_causal = node.args[5] if len(node.args) > 5 else node.kwargs.get("is_causal", False)
        causal_mask = "TopLeft" if is_causal else None

        if attn_mask is not None:
            attn_mask = self.env[attn_mask]
            msg = "Only a float mask is supported for the attn_mask input."
            assert attn_mask.ty.dtype.matches_code(DataTypeCode.FLOAT, DataTypeCode.BFLOAT), msg

        attention_output = self.block_builder.emit(
            relax.op.nn.attention(query, key, value, bias=attn_mask, causal_mask=causal_mask)
        )

        return transpose_and_reshape_back(attention_output)

    def _unbind(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        assert isinstance(dim, int), "Expected 2nd argument of unbind as int"
        selections = self.shape_of(x)[dim].value
        ret = []
        if selections == 1:
            ret.append(self.block_builder.emit(relax.op.squeeze(x, axis=dim)))
        else:
            split = self.block_builder.emit(relax.op.split(x, selections, dim))
            for i in range(selections):
                ret.append(self.block_builder.emit(relax.op.squeeze(split[i], axis=dim)))
        return self.block_builder.emit(relax.Tuple(ret))

    ########## Statistical ##########

    def _mean(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        dim = args[1] if len(node.args) > 1 else node.kwargs.get("dim", None)
        keepdim = args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        dtype = node.kwargs.get("dtype", None)
        if dtype is not None:
            x = self.block_builder.emit(
                relax.op.astype(x, self._convert_data_type(dtype, self.env))
            )
        return self.block_builder.emit(relax.op.mean(x, dim, keepdims=keepdim))

    def _median(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        dim = args[1] if len(node.args) > 1 else node.kwargs.get("dim", None)
        keepdim = args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        return self.block_builder.emit(relax.op.median(x, dim, keepdims=keepdim))

    def _norm(self, node: fx.Node) -> relax.Var:
        data = self.env[node.args[0]]
        dtype = data.ty.dtype
        order = node.args[1] if len(node.args) > 1 else node.kwargs.get("p", 2)
        axis = node.args[2] if len(node.args) > 2 else None
        keepdims = node.args[3] if len(node.args) > 3 else False

        if order == float("inf"):
            return self.block_builder.emit(
                relax.op.max(relax.op.abs(data), axis=axis, keepdims=keepdims)
            )
        elif order == float("-inf"):
            return self.block_builder.emit(
                relax.op.min(relax.op.abs(data), axis=axis, keepdims=keepdims)
            )
        # frobenius_norm
        elif order == "fro":
            return self.block_builder.emit(
                relax.op.sqrt(
                    relax.op.sum(relax.op.multiply(data, data), axis=axis, keepdims=keepdims)
                )
            )
        else:
            ord_expr = (
                order if isinstance(order, relax.Expr) else relax.const(float(order), dtype=dtype)
            )
            reci_order = (
                relax.op.divide(relax.const(1.0, dtype), ord_expr)
                if isinstance(order, relax.Expr)
                else relax.const(1.0 / order, dtype=dtype)
            )
            return self.block_builder.emit(
                relax.op.power(
                    relax.op.sum(
                        relax.op.power(relax.op.abs(data), ord_expr), axis=axis, keepdims=keepdims
                    ),
                    reci_order,
                )
            )

    def _prod(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        dim = args[1] if len(node.args) > 1 else node.kwargs.get("dim", None)
        keepdim = args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        return self.block_builder.emit(relax.op.prod(x, dim, keepdims=keepdim))

    def _std(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        dim = args[1] if len(node.args) > 1 else node.kwargs.get("dim", None)
        keepdim = args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        return self.block_builder.emit(relax.op.std(x, dim, keepdims=keepdim))

    def _sum(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        dim = args[1] if len(node.args) > 1 else node.kwargs.get("dim", None)
        if isinstance(dim, list | tuple) and len(dim) == 0:
            dim = None
        keepdim = args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        dtype = node.kwargs.get("dtype", None)
        if dtype is not None:
            x = self.block_builder.emit(
                relax.op.astype(x, self._convert_data_type(dtype, self.env))
            )
        else:
            # Match PyTorch type promotion: summing bool or integer tensors
            # accumulates in int64 unless an explicit dtype is given.
            input_dtype = x.ty.dtype.dtype
            if input_dtype == "bool" or (
                (input_dtype.startswith("int") or input_dtype.startswith("uint"))
                and input_dtype != "int64"
            ):
                x = self.block_builder.emit(relax.op.astype(x, "int64"))
        return self.block_builder.emit(relax.op.sum(x, dim, keepdims=keepdim))

    def _var(self, node: fx.Node) -> relax.Var:
        # `aten.var.correction` (and decomposed `aten.std.*`) carries an
        # optional `correction` kwarg whose `None` default means 1 (Bessel).
        # Legacy fx `tensor.var(...)` calls go through the original path
        # below to keep this fix narrowly scoped.
        target = node.target
        if (
            getattr(target, "_overloadname", None) == "correction"
            or getattr(target, "overload_name", None) == "correction"
        ):
            return self._var_correction(node)
        args = self.retrieve_args(node)
        x = args[0]
        dim = args[1] if len(node.args) > 1 else node.kwargs.get("dim", None)
        keepdim = args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        return self.block_builder.emit(relax.op.variance(x, dim, keepdims=keepdim))

    def _var_correction(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        dim = args[1] if len(node.args) > 1 else node.kwargs.get("dim", None)
        keepdim = node.kwargs.get("keepdim", False)
        correction = node.kwargs.get("correction", None)
        if correction is None:
            correction = 1
        var = self.block_builder.emit(relax.op.variance(x, dim, keepdims=keepdim))
        if correction == 0:
            return var
        n = self._reduction_size(x, dim)
        if n is None:
            raise NotImplementedError(
                "var/std with non-zero correction requires statically known reduction-axis sizes."
            )
        # PyTorch returns NaN (with a warning) when `n - correction <= 0`;
        # mirror that semantics rather than failing the import.
        if n - correction <= 0:
            scale = float("nan")
        else:
            scale = float(n) / float(n - correction)
        return self.block_builder.emit(relax.op.multiply(var, relax.const(scale, x.ty.dtype)))

    @staticmethod
    def _reduction_size(x: relax.Expr, dim) -> int | None:
        """Static product of reduced-axis sizes; None if any axis is dynamic."""
        shape = x.ty.shape
        if shape is None:
            return None
        rank = len(shape)
        if dim is None:
            axes = list(range(rank))
        elif isinstance(dim, int):
            axes = [dim]
        elif isinstance(dim, list | tuple) and all(isinstance(a, int) for a in dim):
            axes = list(dim)
        else:
            return None
        n = 1
        for ax in axes:
            ax = ax + rank if ax < 0 else ax
            s = shape[ax]
            if not isinstance(s, tvm_tirx.ir.IntImm):
                return None
            n *= int(s.value)
        return n

    def _any(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        dim = args[1] if len(node.args) > 1 else node.kwargs.get("dim", None)
        keepdim = args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)

        # max doesn't support boolean tensors directly, so we compute it in int8 and cast back
        if x.ty.dtype == "bool":
            x = relax.op.astype(x, "int8")
            ret = relax.op.max(x, dim, keepdims=keepdim)
            return self.block_builder.emit(relax.op.astype(ret, "bool"))

        # For boolean tensors, any is equivalent to max (checking if any element is True)
        return self.block_builder.emit(relax.op.max(x, dim, keepdims=keepdim))

    ########## Search ##########

    def _argmax_argmin(self, op: Callable) -> Callable:
        from torch import fx

        def convert(node: fx.Node):
            x = self.env[node.args[0]]
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", None)
            keepdim = node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
            return self.block_builder.emit(op(x, dim, keepdim))

        return convert

    def _where(self, node: fx.Node) -> relax.Var:
        condition = self.env[node.args[0]]
        x = self.env[node.args[1]]
        y = self.env[node.args[2]]
        return self.block_builder.emit(relax.op.where(condition, x, y))

    def _bucketize(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        input_tensor = args[0]
        boundaries = args[1]

        right = node.kwargs.get("right", False)
        out_int32 = node.kwargs.get("out_int32", False)

        return self.block_builder.emit(
            relax.op.bucketize(input_tensor, boundaries, out_int32, right)
        )

    ########## Manipulation ##########

    def _argsort(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", -1)
        descending = node.args[2] if len(node.args) > 2 else node.kwargs.get("descending", False)
        return self.block_builder.emit(relax.op.argsort(x, dim, descending, dtype="int64"))

    def _broadcast_to(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        shape = args[1] if len(args) > 1 else args[0]
        return self.block_builder.emit(relax.op.broadcast_to(x, shape))

    def _cat(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        axis = args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        return self.block_builder.emit(relax.op.concat(args[0], axis=axis))

    def _chunk(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        chunks = node.args[1]
        dim = node.args[2] if len(node.args) > 2 else node.kwargs.get("dim", 0)
        x_shape = self.shape_of(x)
        max_chunks = x_shape[dim].value
        n_sections = min(chunks, max_chunks)
        return self.block_builder.emit(
            relax.op.split(x=x, indices_or_sections=n_sections, axis=dim)
        )

    def _cumprod(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]

        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", None)
        if "dtype" in node.kwargs:
            dtype = self._convert_data_type(str(node.kwargs["dtype"]), self.env)
        else:
            dtype = None

        return self.block_builder.emit(relax.op.cumprod(x, dim, dtype))

    def _cumsum(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]

        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", None)
        if "dtype" in node.kwargs:
            dtype = self._convert_data_type(str(node.kwargs["dtype"]), self.env)
        else:
            dtype = None
        if "out" in node.kwargs:
            raise ValueError("specifying out for cumsum is not supported yet")

        return self.block_builder.emit(relax.op.cumsum(x, dim, dtype))

    def _expand(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        sizes = args[1:] if len(args) > 2 else args[1]
        x = args[0]
        broadcast_shape = []
        in_shape = self.shape_of(x)
        input_rank = len(in_shape) if in_shape is not None else None
        if input_rank is None and hasattr(node.args[0], "meta") and "val" in node.args[0].meta:
            input_rank = len(node.args[0].meta["val"].shape)
        rank_offset = len(sizes) - input_rank if input_rank is not None else 0
        for idx, i in enumerate(sizes):
            if isinstance(i, int) and i == -1:
                input_idx = idx - rank_offset
                if input_idx < 0:
                    raise ValueError(f"Cannot use -1 in expand for new leading dim {idx}")
                if in_shape is not None:
                    broadcast_shape.append(in_shape[input_idx])
                elif hasattr(node.args[0], "meta") and "val" in node.args[0].meta:
                    # Fallback: get shape from FX node metadata (FakeTensor)
                    fake_shape = node.args[0].meta["val"].shape
                    broadcast_shape.append(fake_shape[input_idx])
                else:
                    raise ValueError(
                        f"Cannot use -1 in expand for dim {idx} when input shape is unknown"
                    )
            else:
                broadcast_shape.append(i)
        return self.block_builder.emit(relax.op.broadcast_to(x, broadcast_shape))

    def _expand_as(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        # args[0] is the 'self' tensor
        # args[1] is the 'other' tensor
        data = args[0]
        other_shape = self.shape_of(args[1])  # the shape of 'other'
        return self.block_builder.emit(relax.op.broadcast_to(data, other_shape))

    def _flatten_impl(self, x, start_dim, end_dim) -> relax.Var:
        shape = self.shape_of(x)
        rank = len(shape)

        # torch.flatten() normalizes its dims against a rank of at least one, so a 0-d
        # input still accepts a start_dim/end_dim of 0 or -1.
        dim_post_expr = max(rank, 1)
        norm_start_dim = start_dim + dim_post_expr if start_dim < 0 else start_dim
        norm_end_dim = end_dim + dim_post_expr if end_dim < 0 else end_dim

        # torch rejects invalid flatten dims only when the model is executed. fx.symbolic_trace
        # does not execute it, so an invalid flatten reaches this converter as a traceable node
        # and has to be rejected here instead of failing later on an empty reduce().
        if not 0 <= norm_start_dim < dim_post_expr:
            raise ValueError(
                f"flatten start_dim {start_dim} is out of range "
                f"[-{dim_post_expr}, {dim_post_expr - 1}] for an input of rank {rank}"
            )
        if not 0 <= norm_end_dim < dim_post_expr:
            raise ValueError(
                f"flatten end_dim {end_dim} is out of range "
                f"[-{dim_post_expr}, {dim_post_expr - 1}] for an input of rank {rank}"
            )
        if norm_start_dim > norm_end_dim:
            raise ValueError("flatten() has invalid args: start_dim cannot come after end_dim")

        start_dim, end_dim = norm_start_dim, norm_end_dim

        # torch.flatten() on a 0-d input returns a 1-d tensor holding the single element.
        if rank == 0:
            return self.block_builder.emit(relax.op.reshape(x, [1]))

        flattened = reduce(lambda x, y: x * y, [shape[i] for i in range(start_dim, end_dim + 1)])
        new_shape = (
            [shape[i] for i in range(0, start_dim)]
            + [flattened]
            + [shape[i] for i in range(end_dim + 1, rank)]
        )
        return self._emit_torch_reshape(x, new_shape)

    def _flatten(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        start_dim = node.args[1] if len(node.args) >= 2 else node.kwargs.get("start_dim", 0)
        end_dim = node.args[2] if len(node.args) == 3 else node.kwargs.get("end_dim", -1)
        return self._flatten_impl(x, start_dim, end_dim)

    def _flip(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dims = node.args[1] if len(node.args) > 1 else node.kwargs.get("dims", None)
        if isinstance(dims, int):
            dims = [dims]
        elif not isinstance(dims, list | tuple):
            raise TypeError(f"flip expects an int or list of ints, but got {type(dims)}: {dims}")
        # relax.op.flip is single-axis; iterate to honor multi-axis torch.flip semantics.
        out = x
        for d in dims:
            out = self.block_builder.emit(relax.op.flip(out, d))
        return out

    def _gather(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        index = self.env[node.args[2]]
        return self.block_builder.emit(relax.op.gather_elements(x, index, axis=dim))

    def _index_put(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        tensor = args[0]
        indices = args[1] if len(args) > 1 else node.kwargs.get("indices")
        values = args[2] if len(args) > 2 else node.kwargs.get("values")
        accumulate = args[3] if len(args) > 3 else node.kwargs.get("accumulate", False)

        if indices is None or values is None:
            raise ValueError("'indices and values' arguments are required for index_put operation")

        if not isinstance(accumulate, bool):
            raise TypeError(f"'accumulate' must be a boolean value, got {type(accumulate)}")

        if isinstance(indices, list | tuple):
            # In PyTorch index_put, None means "select all elements" for that dimension
            non_none_indices = [(i, idx) for i, idx in enumerate(indices) if idx is not None]

            if len(non_none_indices) < len(indices):
                data_shape = self.shape_of(tensor)
                processed_indices = []

                max_ndim = max((idx.ty.ndim for _, idx in non_none_indices), default=1)

                for i, idx in enumerate(indices):
                    if idx is None:
                        # Replace None with arange for full dimension indexing
                        arange_idx = self.block_builder.emit(
                            relax.op.arange(
                                relax.prim_value(0),
                                data_shape[i],
                                relax.prim_value(1),
                                "int64",
                            )
                        )
                        # Reshape to [dim_size, 1, 1, ...] for broadcasting
                        # Add an extra dimension so it broadcasts with other indices
                        arange_idx = self.block_builder.emit(
                            relax.op.reshape(arange_idx, [data_shape[i]] + [1] * max_ndim)
                        )
                        processed_indices.append(arange_idx)
                    else:
                        processed_indices.append(idx)

                indices = relax.Tuple(processed_indices)
            else:
                indices = relax.Tuple(indices)

        output = self.block_builder.emit(relax.op.index_put(tensor, indices, values, accumulate))

        target_name = (
            node.target if isinstance(node.target, str) else getattr(node.target, "__name__", "")
        )
        if target_name.startswith("index_put_") and len(node.args) > 0:
            from torch import fx

            if isinstance(node.args[0], fx.Node):
                # `index_put_` is in-place. If the mutated input is an alias of another
                # FX node, later reads via either the alias node or the original node
                # must oberve the updated tensor.
                aliased_expr = tensor
                for env_node, env_expr in list(self.env.items()):
                    if env_expr is aliased_expr:
                        self.env[env_node] = output

        return output

    def _index_tensor(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        data = args[0]
        indices = args[1]

        # In PyTorch's aten.index.Tensor, None means "select all elements" for that dimension
        non_none_indices = [(i, idx) for i, idx in enumerate(indices) if idx is not None]

        # Special case: if there's only one non-None index, use take operation
        if len(non_none_indices) == 1:
            axis, index_tensor = non_none_indices[0]
            return self.block_builder.emit(relax.op.take(data, index_tensor, axis=axis))

        # If no dimension is sliced (no None entries), this is plain NumPy-style
        # advanced indexing: the index tensors broadcast together and are applied
        # jointly ("zipped"), which is exactly relax.op.index_tensor's semantics.
        # Note that the sequential-take path below is NOT equivalent: it computes
        # an outer product over the index tensors, which only matches PyTorch
        # when the index shapes are mutually orthogonal (e.g. (H, 1) and (W,)).
        if len(non_none_indices) == len(indices):
            return self.block_builder.emit(
                relax.op.index_tensor(data, [idx for _, idx in non_none_indices])
            )

        # Check if all indices can be squeezed to 1D for sequential take
        def is_squeezable(idx):
            if idx.ty.ndim == 1:
                return True
            if idx.ty.ndim == 2:
                shape = idx.ty.shape
                for d in shape:
                    if isinstance(d, int) and d == 1:
                        return True
                    # Check for ir.IntImm
                    if hasattr(d, "value") and d.value == 1:
                        return True
            return False

        all_squeezable = all(is_squeezable(idx) for _, idx in non_none_indices)
        if all_squeezable:
            result = data
            for axis, idx in reversed(non_none_indices):
                if idx.ty.ndim > 1:
                    idx = self.block_builder.emit(relax.op.squeeze(idx))
                result = self.block_builder.emit(relax.op.take(result, idx, axis=axis))
            return result

        # General case: replace None with arange, reshaped for broadcasting
        max_ndim = max((idx.ty.ndim for _, idx in non_none_indices), default=1)
        processed_indices = []
        data_shape = self.shape_of(data)

        for i, idx in enumerate(indices):
            if idx is None:
                arange_idx = self.block_builder.emit(
                    relax.op.arange(
                        relax.prim_value(0),
                        data_shape[i],
                        relax.prim_value(1),
                        "int64",
                    )
                )
                # Reshape to [dim_size, 1, 1, ...] for broadcasting
                arange_idx = self.block_builder.emit(
                    relax.op.reshape(arange_idx, [data_shape[i]] + [1] * (max_ndim - 1))
                )
                processed_indices.append(arange_idx)
            else:
                processed_indices.append(idx)

        return self.block_builder.emit(relax.op.index_tensor(data, processed_indices))

    def _meshgrid(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        indexing = args[1] if len(node.args) > 1 else node.kwargs.get("indexing", "ij")
        input_list = args[0]

        # Single input: return as-is, meshgrid not applicable.
        if len(input_list) == 1:
            return input_list
        new_inputs = []
        for i, item in enumerate(input_list):
            if item.ty.ndim == 1:
                new_inputs.append(item)
            elif item.ty.ndim == 0:  # Change scalar value into 1D
                const_tensor = relax.op.reshape(item, (1,))
                new_inputs.append(const_tensor)
            else:
                raise TypeError(f"Unsupported meshgrid input type at index {i}: {type(item)}")

        return self.block_builder.emit(relax.op.meshgrid(new_inputs, indexing=indexing))

    def _slice_scatter(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        input_tensor = args[0]
        src = args[1]
        dim = args[2] if len(args) > 2 else node.kwargs.get("dim", 0)
        start = args[3] if len(args) > 3 else node.kwargs.get("start", 0)
        end = args[4] if len(args) > 4 else node.kwargs.get("end", self.shape_of(input_tensor)[dim])
        step = args[5] if len(args) > 5 else node.kwargs.get("step", 1)

        # Normalize bounds to match PyTorch behavior (negative and open-ended slices).
        input_shape = self.shape_of(input_tensor)
        axis = dim if dim >= 0 else dim + len(input_shape)

        def _normalize_bound(bound):
            # PyTorch uses a large positive value (2^63-1) to mean "len".
            max_index_val = 9223372036854775807

            def _adjust(val):
                if isinstance(val, int | tvm_tirx.ir.IntImm):
                    int_val = int(val)
                    if int_val >= max_index_val:
                        return input_shape[axis]
                    if int_val < 0:
                        return input_shape[axis] + int_val
                    if isinstance(input_shape[axis], int | tvm_tirx.ir.IntImm) and int_val > int(
                        input_shape[axis]
                    ):
                        return input_shape[axis]
                return val

            if tvm_tirx.ir.is_prim_expr(bound):
                value = _adjust(bound)
                return relax.prim_value(value)

            bound = _adjust(bound)
            if not tvm_tirx.ir.is_prim_expr(bound):
                bound = relax.prim_value(bound)
            return bound

        start = _normalize_bound(start)
        end = _normalize_bound(end)
        if not tvm_tirx.ir.is_prim_expr(step):
            step = relax.prim_value(step)

        return self.block_builder.emit(
            relax.op.slice_scatter(input_tensor, src, start, end, step, axis=dim)
        )

    def _permute(self, node: fx.Node) -> relax.Var:
        import torch  # type: ignore

        args = self.retrieve_args(node)
        x = args[0]
        dims = args[1] if isinstance(args[1], torch.Size | tuple | list) else args[1:]
        return self.block_builder.emit(relax.op.permute_dims(x, dims))

    def _repeat(self, node: fx.Node) -> relax.Var:
        import torch  # type: ignore

        args = self.retrieve_args(node)
        x = args[0]
        dims = args[1] if isinstance(args[1], torch.Size | tuple | list) else args[1:]
        return self.block_builder.emit(relax.op.tile(x, dims))

    def _roll(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        input_tensor = args[0]
        shifts = args[1] if len(node.args) > 1 else node.kwargs.get("shifts", None)
        dims = args[2] if len(node.args) > 2 else node.kwargs.get("dims", None)

        # Get original shape
        original_shape = self.shape_of(input_tensor)

        def to_int(val):
            if isinstance(val, tvm_tirx.ir.IntImm):
                return int(val.value)
            elif isinstance(val, int):
                return val
            elif hasattr(val, "__int__"):
                return int(val)
            raise TypeError(f"Unsupported type for shift/dim: {type(val)}")

        def roll_single_dim(tensor: relax.Var, shift: int, dim: int) -> relax.Var:
            shape = self.shape_of(tensor)

            dim_size = shape.values[dim]
            shift_val = to_int(shift)
            dim_size_val = to_int(dim_size)
            shift_mod = shift_val % dim_size_val
            if shift_mod == 0:
                return tensor

            split_pos = dim_size_val - shift_mod
            part1 = self.block_builder.emit(
                relax.op.strided_slice(
                    tensor,
                    axes=[dim],
                    begin=[0],
                    end=[split_pos],
                    strides=[1],
                )
            )
            part2 = self.block_builder.emit(
                relax.op.strided_slice(
                    tensor,
                    axes=[dim],
                    begin=[split_pos],
                    end=[dim_size_val],
                    strides=[1],
                )
            )
            return self.block_builder.emit(relax.op.concat([part2, part1], axis=dim))

        # Handle dims=None (flatten -> roll -> reshape)
        if dims is None:
            flattened = self.block_builder.emit(relax.op.reshape(input_tensor, (-1,)))
            shift_scalar = to_int(shifts[0] if isinstance(shifts, list | tuple) else shifts)
            rolled = roll_single_dim(flattened, shift_scalar, 0)
            return self.block_builder.emit(relax.op.reshape(rolled, original_shape))

        # Normalize shifts and dims
        if isinstance(shifts, list | tuple):
            shifts = [to_int(s) for s in shifts]
        else:
            shifts = [to_int(shifts)]

        if isinstance(dims, list | tuple):
            dims = [to_int(d) for d in dims]
        else:
            dims = [to_int(dims)]

        if len(shifts) != len(dims):
            raise ValueError("shifts and dims must have the same length")

        result = input_tensor
        rank = len(original_shape.values)
        for shift, dim in zip(shifts, dims):
            if dim < 0:
                dim += rank
            result = roll_single_dim(result, shift, dim)

        return result

    def _reshape(self, node: fx.Node) -> relax.Var:
        import torch  # type: ignore

        args = self.retrieve_args(node)
        x = args[0]
        dims = args[1] if isinstance(args[1], torch.Size | tuple | list) else args[1:]

        # Skip identity reshape
        current_shape = self.shape_of(x)
        if current_shape is not None and list(current_shape) == list(dims):
            return x

        return self._emit_torch_reshape(x, dims)

    def _reshape_as(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        other = args[1]
        dims = self.shape_of(other)
        return self._emit_torch_reshape(x, dims)

    def _scatter(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        if len(node.args) == 1:
            dim = node.kwargs["dim"]
            index = self.env[node.kwargs["index"]]
            src = self.env[node.kwargs["src"]]
        elif len(node.args) == 4:
            dim = node.args[1]
            index = self.env[node.args[2]]
            src = self.env[node.args[3]]
        else:
            raise Exception("Unexpected args " + str(node.args))
        return self.block_builder.emit(relax.op.scatter_elements(x, index, src, axis=dim))

    def _sort(self, node: fx.Node) -> relax.Var:
        # torch.sort() returns a tuple of values and indices
        # we use argsort to get indices and gather_elements to get values
        x = self.env[node.args[0]]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", -1)
        descending = node.args[2] if len(node.args) > 2 else node.kwargs.get("descending", False)

        indices = self.block_builder.emit(relax.op.argsort(x, dim, descending, dtype="int64"))
        values = self.block_builder.emit(relax.op.gather_elements(x, indices, axis=dim))
        return self.block_builder.emit(relax.Tuple([values, indices]))

    def _split(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        split_size = node.args[1]
        dim = node.args[2] if len(node.args) > 2 else node.kwargs.get("dim", 0)
        if isinstance(split_size, list | tuple):
            n_section = []
            for s in split_size[:-1]:
                cum_sum = 0 if not n_section else n_section[-1]
                n_section.append(s + cum_sum)
        else:
            # torch.split(x, s, dim) splits dim into chunks of size s, with the
            # last chunk smaller if D % s != 0. relax.op.split's integer argument
            # is the number of *equal* sections, so passing ceil(D / s) yields
            # wrong shapes whenever ceil(D / ceil(D / s)) != s (e.g. s > D/2).
            # Convert the per-chunk size to the cumulative cut positions instead,
            # mirroring the list/tuple branch above.
            dim_size = self.shape_of(x)[dim].value
            num_chunks = (dim_size + split_size - 1) // split_size
            n_section = [split_size * i for i in range(1, num_chunks)]
        return self.block_builder.emit(relax.op.split(x, n_section, dim))

    def _squeeze(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", None)
        # Support both "dim" and "dims" parameters
        if dim is None:
            dim = node.kwargs.get("dims", None)

        # fx.symbolic_trace never executes the model, so torch's IndexError for an
        # out-of-range dim does not fire; reject the axis here instead of dropping it.
        if dim is not None:
            rank = len(self.shape_of(x))
            for d in dim if isinstance(dim, list | tuple) else [dim]:
                if isinstance(d, int) and not -rank <= d < rank:
                    raise ValueError(
                        f"squeeze dim {d} is out of range "
                        f"[-{rank}, {rank - 1}] for an input of rank {rank}"
                    )

        return self.block_builder.emit(relax.op.squeeze(x, dim))

    def _stack(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        tensor_list = args[0]
        axis = args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        return self.block_builder.emit(relax.op.stack(tensor_list, axis=axis))

    def _take(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        indices = self.env[node.args[1]]
        indices = self.block_builder.emit(relax.op.astype(indices, "int32"))
        return self.block_builder.emit(relax.op.take(x, indices))

    def _tile(self, node: fx.Node) -> relax.Var:
        import torch  # type: ignore

        args = self.retrieve_args(node)
        x = args[0]
        dims = args[1] if isinstance(args[1], torch.Size | tuple | list) else args[1:]
        return self.block_builder.emit(relax.op.tile(x, dims))

    def _topk(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        k = args[1] if len(args) > 1 else node.kwargs.get("k", 1)
        dim = args[2] if len(args) > 2 else node.kwargs.get("dim", -1)
        largest = args[3] if len(args) > 3 else node.kwargs.get("largest", True)
        _sorted = args[4] if len(args) > 4 else node.kwargs.get("_sorted", True)

        if not _sorted:
            msg = "Currently supports only sorted output for topk operator."
            raise AssertionError(msg)

        return self.block_builder.emit(
            relax.op.topk(x, k=k, axis=dim, largest=largest, ret_type="both", dtype="int64")
        )

    def _transpose(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        full_idx = list(range(len(self.shape_of(args[0]))))
        full_idx[args[1]], full_idx[args[2]] = full_idx[args[2]], full_idx[args[1]]
        return self.block_builder.emit(relax.op.permute_dims(args[0], full_idx))

    ########## Creation ##########

    def _detach(self, node: fx.Node) -> relax.Var:
        # There is no way to implement detach() such that the output shares
        # the same memory as the input. In-place operations are not supported
        # by the translator, and therefore we just return a copy of the input.
        return self.env[node.args[0]]

    def _copy_(self, node: fx.Node) -> relax.Var:
        dest = self.env[node.args[0]]
        src = self.env[node.args[1]]

        # Match PyTorch semantics: cast to destination dtype and broadcast to destination shape.
        if src.ty.dtype != dest.ty.dtype:
            src = self.block_builder.emit(relax.op.astype(src, dest.ty.dtype.dtype))

        dest_shape = self.shape_of(dest)
        src_shape = self.shape_of(src)
        if dest_shape != src_shape:
            src = self.block_builder.emit(relax.op.broadcast_to(src, dest_shape))

        # copy_ writes into the destination tensor, so update env accordingly
        self.env[node.args[0]] = src
        return src

    def _to_copy(self, node: fx.Node) -> relax.Var:
        # Returns a copy of the input tensor
        import torch  # type: ignore

        x = self.env[node.args[0]]
        if len(node.args) == 2:
            if isinstance(node.args[1], torch.dtype):
                dtype = self._convert_data_type(node.args[1], self.env)
                return self.block_builder.emit(relax.op.astype(x, dtype))
        elif "dtype" in node.kwargs:
            dtype = self._convert_data_type(node.kwargs["dtype"], self.env)
            return self.block_builder.emit(relax.op.astype(x, dtype))
        return x

    def _arange(self, node: fx.Node) -> relax.Var:
        import torch  # type: ignore

        start_end_step = [None, None, None]
        if "start" in node.kwargs:
            start_end_step[0] = node.kwargs["start"]
        if "end" in node.kwargs:
            start_end_step[1] = node.kwargs["end"]
        if "step" in node.kwargs:
            start_end_step[2] = node.kwargs["step"]

        if len(node.args) == 1:
            assert start_end_step[1] is None
            start_end_step[1] = node.args[0]
        elif len(node.args) == 2:
            assert start_end_step[0] is None
            assert start_end_step[1] is None
            start_end_step[0] = node.args[0]
            start_end_step[1] = node.args[1]
        elif len(node.args) == 3:
            assert start_end_step[0] is None
            assert start_end_step[1] is None
            assert start_end_step[2] is None
            start_end_step[0] = node.args[0]
            start_end_step[1] = node.args[1]
            start_end_step[2] = node.args[2]

        if start_end_step[0] is None:
            start_end_step[0] = 0
        if start_end_step[2] is None:
            start_end_step[2] = 1

        if "dtype" in node.kwargs:
            dtype = self._convert_data_type(str(node.kwargs["dtype"]), self.env)
        elif any([isinstance(x, float) for x in start_end_step]):
            dtype = self._convert_data_type(torch.get_default_dtype())
        else:
            dtype = "int64"
        start_end_step = [
            self.env[x] if isinstance(x, torch.fx.Node) else x for x in start_end_step
        ]
        return self.block_builder.emit(relax.op.arange(*start_end_step, dtype=dtype))

    def _empty(self, node: fx.Node) -> relax.Var:
        import torch

        dtype = self._convert_data_type(
            node.kwargs.get("dtype", torch.get_default_dtype()), self.env
        )
        return self.block_builder.emit(relax.op.zeros(node.args[0], dtype))

    def _empty_like(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        return self.block_builder.emit(relax.op.zeros_like(x))

    def _eye(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        n = args[0]
        m = args[1] if len(args) > 1 else n
        dtype = self._convert_data_type(str(node.kwargs["dtype"]), self.env)
        return self.block_builder.emit(relax.op.eye(n, m, dtype=dtype))

    def _fill(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        dtype = str(x.ty.dtype)
        value = self._convert_scalar_fill_value(args[1], dtype)
        return self.block_builder.emit(relax.op.full(x.ty.shape, value, dtype))

    def _prim_value_to_scalar_tensor(self, value: relax.Expr, dtype: str) -> relax.Var:
        """Materialize an integer or boolean primitive value as a rank-zero tensor."""
        if not value.ty.matches_code(DataTypeCode.INT, DataTypeCode.UINT, DataTypeCode.BOOL):
            raise TypeError(f"Cannot materialize primitive value of dtype {value.ty} as a tensor")
        shape_value = value if str(value.ty) == "int64" else value.astype("int64")
        value_tensor = self.block_builder.emit(
            relax.op.shape_to_tensor(relax.ShapeExpr([shape_value]))
        )
        if dtype != "int64":
            value_tensor = self.block_builder.emit(relax.op.astype(value_tensor, dtype))
        return self.block_builder.emit(relax.op.squeeze(value_tensor, axis=[0]))

    def _convert_scalar_fill_value(self, value, dtype: str) -> relax.Expr:
        """Convert a PyTorch scalar fill value to a rank-zero Relax tensor."""
        if isinstance(getattr(value, "ty", None), PrimType):
            return self._prim_value_to_scalar_tensor(value, dtype)
        if isinstance(value, relax.Expr):
            return value
        return relax.const(value, dtype)

    def _inplace_fill(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        dtype = str(x.ty.dtype)
        value = self._convert_scalar_fill_value(args[1], dtype)
        filled = self.block_builder.emit(relax.op.full(x.ty.shape, value, dtype))
        self.env[node.args[0]] = filled
        return filled

    def _full(self, node: fx.Node) -> relax.Var:
        import torch

        args = self.retrieve_args(node)
        size = relax.ShapeExpr(args[0] if isinstance(args[0], list | tuple) else (args[0],))
        torch_dtype = node.kwargs.get("dtype")
        if torch_dtype is None:
            output_meta = node.meta.get("val")
            if output_meta is None:
                output_meta = node.meta.get("tensor_meta")
            torch_dtype = getattr(output_meta, "dtype", None)
        if torch_dtype is None:
            if isinstance(args[1], bool):
                torch_dtype = "bool"
            elif isinstance(args[1], int):
                torch_dtype = "int64"
            elif isinstance(getattr(args[1], "ty", None), PrimType):
                torch_dtype = args[1].ty.dtype
            else:
                torch_dtype = torch.get_default_dtype()
        dtype = self._convert_data_type(torch_dtype, self.env)
        value = self._convert_scalar_fill_value(args[1], dtype)
        return self.block_builder.emit(
            relax.op.full(
                size,
                value,
                dtype,
            )
        )

    def _full_like(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        value = args[1]
        x_dtype = str(x.ty.dtype)
        torch_dtype = node.kwargs.get("dtype")
        dtype = self._convert_data_type(x_dtype if torch_dtype is None else torch_dtype, self.env)
        fill_dtype = dtype if dtype != x_dtype else None
        if (
            fill_dtype is None
            and isinstance(value, int | float)
            and (math.isinf(value) or math.isnan(value))
        ):
            if not ("float" in x_dtype or "bfloat16" in x_dtype):
                fill_dtype = "float32"
                dtype = fill_dtype

        if isinstance(getattr(value, "ty", None), PrimType):
            fill_value = self._prim_value_to_scalar_tensor(value, dtype)
        elif isinstance(value, relax.Expr):
            fill_value = value
        else:
            fill_value = relax.const(value, dtype)

        return self.block_builder.emit(relax.op.full_like(x, fill_value, dtype=fill_dtype))

    def _index_select(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dim = node.args[1]
        index = self.env[node.args[2]]
        return self.block_builder.emit(relax.op.take(x, index, dim))

    def _inplace_masked_fill(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x, mask, value = args[:3]
        x_dtype = str(x.ty.dtype)
        fill_dtype = None
        if isinstance(value, int | float) and (math.isinf(value) or math.isnan(value)):
            if not ("float" in x_dtype or "bfloat16" in x_dtype):
                fill_dtype = "float32"
        rx_value = self._convert_scalar_fill_value(value, fill_dtype or x_dtype)

        values = self.block_builder.emit(relax.op.full_like(x, rx_value, dtype=fill_dtype))

        # Cast x to match values dtype if necessary
        if fill_dtype is not None:
            x = self.block_builder.emit(relax.op.astype(x, fill_dtype))

        output = self.block_builder.emit(relax.op.where(mask, values, x))
        self.env[node.args[0]] = output
        return output

    def _linspace(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        start = args[0]
        stop = args[1]
        step = args[2]

        if step != 1:
            step = (stop - start) / (step - 1)
            stop = stop + (step / 2)
        else:
            stop = start + step

        if len(args) <= 3 or args[3] is None:
            import torch

            dtype = self._convert_data_type(str(torch.get_default_dtype()))
        else:
            dtype = self._convert_data_type(args[3])

        return self.block_builder.emit(
            relax.op.arange(start=start, end=stop, step=step, dtype=dtype)
        )

    def _masked_fill(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x, mask, value = args[:3]
        x_dtype = str(x.ty.dtype)
        fill_dtype = None
        if isinstance(value, int | float) and (math.isinf(value) or math.isnan(value)):
            if not ("float" in x_dtype or "bfloat16" in x_dtype):
                fill_dtype = "float32"
        rx_value = self._convert_scalar_fill_value(value, fill_dtype or x_dtype)

        values = self.block_builder.emit(relax.op.full_like(x, rx_value, dtype=fill_dtype))

        # Cast x to match values dtype if necessary
        if fill_dtype is not None:
            x = self.block_builder.emit(relax.op.astype(x, fill_dtype))

        return self.block_builder.emit(relax.op.where(mask, values, x))

    def _masked_select(self, node: fx.Node) -> relax.Var:
        data = self.env[node.args[0]]
        mask = self.env[node.args[1]]

        data_shape = self.shape_of(data)
        mask_shape = self.shape_of(mask)
        shapes_equal = tvm_ffi.structural_equal(data_shape, mask_shape)

        if not shapes_equal:
            mask = self.block_builder.emit(relax.op.broadcast_to(mask, data_shape))

        data_flat = self.block_builder.emit(relax.op.reshape(data, [-1]))
        mask_flat = self.block_builder.emit(relax.op.reshape(mask, [-1]))
        indices = self.block_builder.emit(relax.op.nonzero(mask_flat))
        tensor_meta = node.meta.get("tensor_meta")
        if tensor_meta is not None and len(tensor_meta.shape) == 1:
            num_selected = tensor_meta.shape[0]
            if not isinstance(num_selected, int):
                num_selected = tvm_tirx.ir.Var(str(num_selected), "int64")
        else:
            num_selected = tvm_tirx.ir.Var(f"{node.name}_num_selected", "int64")
        indices = self.block_builder.match_cast(
            indices, relax.TensorType([1, num_selected], "int64")
        )
        indices_1d = self.block_builder.emit(relax.op.squeeze(indices, axis=[0]))

        result = self.block_builder.emit(relax.op.take(data_flat, indices_1d, axis=0))

        return result

    def _new_ones(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        self_var = args[0]
        size = args[1] if isinstance(args[1], list | tuple) else args[1:]
        if not isinstance(size, list | tuple):
            size = (size,)
        size = relax.ShapeExpr(size)
        return self.block_builder.emit(
            relax.op.full(
                size,
                relax.const(1, self_var.ty.dtype.dtype),
                self_var.ty.dtype.dtype,
            )
        )

    def _new_zeros(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        input_tensor = args[0]
        size = (
            args[1]
            if isinstance(args[1], list | tuple)
            else (args[1],)
            if len(args[1:]) == 1
            else args[1:]
        )
        size = relax.ShapeExpr(size)
        return self.block_builder.emit(
            relax.op.full(
                size,
                relax.const(0, input_tensor.ty.dtype.dtype),
                input_tensor.ty.dtype.dtype,
            )
        )

    def _ones(self, node: fx.Node) -> relax.Var:
        import torch

        args = self.retrieve_args(node)
        size = relax.ShapeExpr(args[0] if isinstance(args[0], list | tuple) else (args[0],))
        dtype = self._convert_data_type(
            node.kwargs.get("dtype", torch.get_default_dtype()), self.env
        )
        return self.block_builder.emit(
            relax.op.full(
                size,
                relax.const(1, dtype),
                dtype,
            )
        )

    ########## DataType ##########

    def _to(self, node: fx.Node) -> relax.Var:
        import torch

        x = self.env[node.args[0]]
        if len(node.args) == 2:
            if isinstance(node.args[1], torch.dtype):
                dtype = BaseFXGraphImporter._convert_data_type(node.args[1], self.env)
                return self.block_builder.emit(relax.op.astype(x, dtype))
        elif "dtype" in node.kwargs:
            dtype = BaseFXGraphImporter._convert_data_type(node.kwargs["dtype"], self.env)
            return self.block_builder.emit(relax.op.astype(x, dtype))
        return x

    def _type_as(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        other = self.env[node.args[1]]
        dtype = other.ty.dtype.dtype
        return self.block_builder.emit(relax.op.astype(x, dtype))

    ########## Others ##########

    def _getitem(self, node: fx.Node) -> relax.Var:
        import torch

        x = self.env[node.args[0]]
        if isinstance(x, list | tuple | relax.ShapeExpr | relax.Tuple):
            return x[node.args[1]]
        elif isinstance(x, tvm_tirx.ir.Var) and isinstance(
            x.ty, relax.TupleType | relax.TensorType
        ):
            if isinstance(x.ty, relax.TupleType):
                return self.block_builder.emit(relax.TupleGetItem(x, node.args[1]))

            assert isinstance(x.ty, relax.TensorType)
            if isinstance(node.args[1], int):
                return x
            if not isinstance(node.args[1], list | tuple):
                indices = [node.args[1]]
            else:
                indices = node.args[1]
            take_indices = []
            take_axes = []
            stride_begin = []
            stride_end = []
            stride = []
            stride_axes = []
            expand_dim = []
            i = 0
            shape = self.shape_of(x)
            non_ellipsis_cnt = 0
            for index in indices:
                if isinstance(index, int | slice | torch.fx.Node):
                    non_ellipsis_cnt += 1
            for index in indices:
                if isinstance(index, int):
                    stride_begin.append(index)
                    stride_end.append(index + 1)
                    stride.append(1)
                    stride_axes.append(i)
                    i = i + 1
                elif isinstance(index, slice):
                    stride_begin.append(0 if index.start is None else index.start)
                    stride_end.append(shape[i] if index.stop is None else index.stop)
                    stride.append(1 if index.step is None else index.step)
                    stride_axes.append(i)
                    i = i + 1
                elif index is None:
                    expand_dim.append(len(stride_axes) + len(expand_dim))
                elif index is Ellipsis:
                    for _ in range(len(shape) - non_ellipsis_cnt):
                        stride_begin.append(0)
                        stride_end.append(shape[i])
                        stride.append(1)
                        stride_axes.append(i)
                        i += 1
                elif isinstance(index, torch.fx.Node):
                    node_index = self.env[index]
                    if not isinstance(node_index, relax.Expr):
                        raise ValueError(
                            "Unsupported index type for relax.op.take: " + str(type(node_index))
                        )
                    take_indices.append(node_index)
                    take_axes.append(i)
                    i = i + 1
                else:
                    raise ValueError("Unsupported index type: " + str(type(index)))
            while i < len(shape):
                stride_begin.append(0)
                stride_end.append(shape[i])
                stride.append(1)
                stride_axes.append(i)
                i += 1
            taken = x
            if len(take_indices) > 1:
                raise ValueError("Multiple tensors as index not yet supported")
            for each_index, each_axis in zip(take_indices, take_axes):
                taken = self.block_builder.emit(relax.op.take(taken, each_index, each_axis))
            sliced = self.block_builder.emit(
                relax.op.strided_slice(taken, stride_axes, stride_begin, stride_end, stride)
            )
            sliced_shape = list(self.shape_of(sliced))
            for i in expand_dim:
                sliced_shape.insert(i, 1)
            return self.block_builder.emit(relax.op.reshape(sliced, sliced_shape))
        elif isinstance(x, tvm_tirx.ir.GenericConst):
            dtype = x.ty.dtype
            return relax.const(x.value.numpy()[node.args[1]], dtype)
        else:
            assert False

    def _item(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        shape = self.shape_of(x)
        dtype = x.ty.dtype
        analyzer = tvm_tirx.sym.Analyzer()
        has_single_element = shape is not None and all(
            analyzer.can_prove_equal(dim, 1) for dim in shape
        )
        if has_single_element and dtype.matches_code(DataTypeCode.INT, DataTypeCode.UINT):
            scalar = x
            if str(dtype) != "int64":
                scalar = self.block_builder.emit(relax.op.astype(scalar, "int64"))
            scalar = self.block_builder.emit(relax.op.reshape(scalar, [1]))
            shape_value = self.block_builder.emit(relax.op.tensor_to_shape(scalar))
            dim = tirx.Var(f"{node.name}_dim", "int64")
            self.block_builder.match_cast(shape_value, relax.ShapeType([dim]))
            return dim
        if shape is not None and len(shape) == 0:
            return x
        return self.block_builder.emit(relax.op.take(x, relax.const(0, "int64"), axis=0))

    def _sym_size_int(self, node: fx.Node) -> relax.Expr:
        x = self.env[node.args[0]]
        shape = self.shape_of(x)
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)

        # Handle case where shape is unknown (None) - this can happen for operations
        # with dynamic output shapes.
        if shape is None:
            return self.block_builder.emit(relax.const(0, "int64"))

        shape_dim = shape[dim]
        if hasattr(shape_dim, "value"):
            return self.block_builder.emit(relax.const(shape_dim.value, dtype="int32"))
        return shape_dim

    def _zeros_inplace(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        output = self.block_builder.emit(relax.op.zeros_like(x))
        self.env[node.args[0]] = output
        return output

    def _zeros_like(self, node: fx.node) -> relax.Var:
        x = self.env[node.args[0]]
        return self.block_builder.emit(relax.op.zeros_like(x))

    @abc.abstractmethod
    def create_convert_map(
        self,
    ) -> dict[torch.nn.Module | str, Callable[[fx.Node], relax.Var]]:
        """Create convert map"""
