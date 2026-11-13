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
"""Tensor class for computation declaration."""

# pylint: disable=invalid-name
import tvm_ffi

from tvm_tirx.ir import Expr, OpaqueExpr, SeqStmt, const, is_prim_expr
from tvm_tirx.ir.location import UNKNOWN_LOC, Location
from tvm_tirx.tirx import expr as _expr

from . import _ffi_api, _te_tensor_overload


def _as_scalar_operand(value):
    return value.__tvm_ffi_object__() if isinstance(value, TensorSlice) else value


class TensorSlice:
    """Auxiliary data structure for enable slicing syntax from tensor."""

    def __init__(self, tensor, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        self.tensor = tensor
        self.indices = indices

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        return TensorSlice(self.tensor, self.indices + indices)

    def __tvm_ffi_object__(self) -> Expr:
        """Convert slice to object."""
        return self.tensor.__call__(*self.indices)

    @property
    def dtype(self):
        """Data content of the tensor."""
        return self.tensor.dtype

    def expr_ty(self):
        """Compile-time element type of the tensor."""
        return self.tensor.expr_ty()

    def __add__(self, other):
        result = _te_tensor_overload.__add__(self, other)
        if result is not NotImplemented:
            return result
        return _expr.ExprOp.__add__(self.__tvm_ffi_object__(), _as_scalar_operand(other))

    def __radd__(self, other):
        result = _te_tensor_overload.__radd__(self, other)
        if result is not NotImplemented:
            return result
        return _expr.ExprOp.__radd__(self.__tvm_ffi_object__(), _as_scalar_operand(other))

    def __sub__(self, other):
        result = _te_tensor_overload.__sub__(self, other)
        if result is not NotImplemented:
            return result
        return _expr.ExprOp.__sub__(self.__tvm_ffi_object__(), _as_scalar_operand(other))

    def __rsub__(self, other):
        result = _te_tensor_overload.__rsub__(self, other)
        if result is not NotImplemented:
            return result
        return _expr.ExprOp.__rsub__(self.__tvm_ffi_object__(), _as_scalar_operand(other))

    def __mul__(self, other):
        result = _te_tensor_overload.__mul__(self, other)
        if result is not NotImplemented:
            return result
        return _expr.ExprOp.__mul__(self.__tvm_ffi_object__(), _as_scalar_operand(other))

    def __rmul__(self, other):
        result = _te_tensor_overload.__rmul__(self, other)
        if result is not NotImplemented:
            return result
        return _expr.ExprOp.__rmul__(self.__tvm_ffi_object__(), _as_scalar_operand(other))

    def __div__(self, other):
        result = _te_tensor_overload.__div__(self, other)
        if result is not NotImplemented:
            return result
        return _expr.ExprOp.__div__(self.__tvm_ffi_object__(), _as_scalar_operand(other))

    def __rdiv__(self, other):
        result = _te_tensor_overload.__rdiv__(self, other)
        if result is not NotImplemented:
            return result
        return _expr.ExprOp.__rdiv__(self.__tvm_ffi_object__(), _as_scalar_operand(other))

    def __truediv__(self, other):
        result = _te_tensor_overload.__truediv__(self, other)
        if result is not NotImplemented:
            return result
        return _expr.ExprOp.__truediv__(self.__tvm_ffi_object__(), _as_scalar_operand(other))

    def __rtruediv__(self, other):
        result = _te_tensor_overload.__rtruediv__(self, other)
        if result is not NotImplemented:
            return result
        return _expr.ExprOp.__rtruediv__(self.__tvm_ffi_object__(), _as_scalar_operand(other))

    def __floordiv__(self, other):
        return _expr.ExprOp.__floordiv__(self.__tvm_ffi_object__(), _as_scalar_operand(other))

    def __rfloordiv__(self, other):
        return _expr.ExprOp.__rfloordiv__(self.__tvm_ffi_object__(), _as_scalar_operand(other))

    def __mod__(self, other):
        return _expr.ExprOp.__mod__(self.__tvm_ffi_object__(), _as_scalar_operand(other))

    def __rmod__(self, other):
        return _expr.ExprOp.__rmod__(self.__tvm_ffi_object__(), _as_scalar_operand(other))

    def __neg__(self):
        return _expr.ExprOp.__neg__(self.__tvm_ffi_object__())

    def __lshift__(self, other):
        return _expr.ExprOp.__lshift__(self.__tvm_ffi_object__(), _as_scalar_operand(other))

    def __rlshift__(self, other):
        return _expr.ExprOp.__rlshift__(self.__tvm_ffi_object__(), _as_scalar_operand(other))

    def __rshift__(self, other):
        return _expr.ExprOp.__rshift__(self.__tvm_ffi_object__(), _as_scalar_operand(other))

    def __rrshift__(self, other):
        return _expr.ExprOp.__rrshift__(self.__tvm_ffi_object__(), _as_scalar_operand(other))

    def __and__(self, other):
        return _expr.ExprOp.__and__(self.__tvm_ffi_object__(), _as_scalar_operand(other))

    def __rand__(self, other):
        return _expr.ExprOp.__rand__(self.__tvm_ffi_object__(), _as_scalar_operand(other))

    def __or__(self, other):
        return _expr.ExprOp.__or__(self.__tvm_ffi_object__(), _as_scalar_operand(other))

    def __ror__(self, other):
        return _expr.ExprOp.__ror__(self.__tvm_ffi_object__(), _as_scalar_operand(other))

    def __xor__(self, other):
        return _expr.ExprOp.__xor__(self.__tvm_ffi_object__(), _as_scalar_operand(other))

    def __rxor__(self, other):
        return _expr.ExprOp.__rxor__(self.__tvm_ffi_object__(), _as_scalar_operand(other))

    def __invert__(self):
        return _expr.ExprOp.__invert__(self.__tvm_ffi_object__())

    def __lt__(self, other):
        return _expr.ExprOp.__lt__(self.__tvm_ffi_object__(), _as_scalar_operand(other))

    def __le__(self, other):
        return _expr.ExprOp.__le__(self.__tvm_ffi_object__(), _as_scalar_operand(other))

    def __eq__(self, other):
        return _expr.ExprOp.__eq__(self.__tvm_ffi_object__(), _as_scalar_operand(other))

    def __ne__(self, other):
        return _expr.ExprOp.__ne__(self.__tvm_ffi_object__(), _as_scalar_operand(other))

    def __gt__(self, other):
        return _expr.ExprOp.__gt__(self.__tvm_ffi_object__(), _as_scalar_operand(other))

    def __ge__(self, other):
        return _expr.ExprOp.__ge__(self.__tvm_ffi_object__(), _as_scalar_operand(other))

    def __nonzero__(self):
        return _expr.ExprOp.__nonzero__(self.__tvm_ffi_object__())

    def __bool__(self):
        return self.__nonzero__()

    def equal(self, other, loc: Location = UNKNOWN_LOC):
        return _expr.ExprOp.equal(self.__tvm_ffi_object__(), _as_scalar_operand(other), loc)

    def astype(self, dtype, loc: Location = UNKNOWN_LOC):
        return _expr.ExprOp.astype(self.__tvm_ffi_object__(), dtype, loc)


class TensorOpBase:
    """Operator overloads for whole TE Tensor values."""

    def __add__(self, other):
        return _te_tensor_overload.__add__(self, other)

    def __radd__(self, other):
        return _te_tensor_overload.__radd__(self, other)

    def __sub__(self, other):
        return _te_tensor_overload.__sub__(self, other)

    def __rsub__(self, other):
        return _te_tensor_overload.__rsub__(self, other)

    def __mul__(self, other):
        return _te_tensor_overload.__mul__(self, other)

    def __rmul__(self, other):
        return _te_tensor_overload.__rmul__(self, other)

    def __div__(self, other):
        return _te_tensor_overload.__div__(self, other)

    def __rdiv__(self, other):
        return _te_tensor_overload.__rdiv__(self, other)

    def __truediv__(self, other):
        return _te_tensor_overload.__truediv__(self, other)

    def __rtruediv__(self, other):
        return _te_tensor_overload.__rtruediv__(self, other)

    def __neg__(self):
        return self.__mul__(const(-1, self.expr_ty()))

    def __nonzero__(self):
        return _expr.ExprOp.__nonzero__(self)

    def __bool__(self):
        return self.__nonzero__()

    def equal(self, other, loc: Location = UNKNOWN_LOC):
        return _expr.ExprOp.equal(self, other, loc)

    def astype(self, dtype, loc: Location = UNKNOWN_LOC):
        result = _te_tensor_overload.astype(self, dtype, loc)
        if result is NotImplemented:
            raise TypeError("TE Tensor overload astype is not registered")
        return result


@tvm_ffi.register_object("te.Tensor")
class Tensor(OpaqueExpr, TensorOpBase):
    """Tensor object, to construct, see function.Tensor"""

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op.name={self.op.name})"

    def __call__(self, *indices):
        ndim = self.ndim
        if len(indices) != ndim:
            raise ValueError(
                f"Need to provide {ndim} index in tensor but {len(indices)} was provided"
            )
        return _ffi_api.TensorLoad(self, indices)

    def __getitem__(self, indices):
        return TensorSlice(self, indices)

    def __hash__(self):
        return _ffi_api.TensorHash(self)

    def __eq__(self, other):
        if not isinstance(other, Tensor):
            if isinstance(other, _expr.ExprOp) or is_prim_expr(other):
                return _expr.EqualOp(self, other)
            return False
        if self.ndim == 0 and other.ndim == 0:
            raise ValueError(
                "Equal == comparison among rank-0 tensor is ambiguous, "
                "use Tensor.equal for content expression equvalence, "
                "use Tensor.same_as for exact reference comparison"
            )
        return _ffi_api.TensorEqual(self, other)

    @property
    def ndim(self):
        """Dimension of the tensor."""
        return len(self.shape)

    @property
    def dtype(self):
        """Data content of the tensor."""
        return _ffi_api.TensorDType(self)

    def expr_ty(self):
        """Compile-time element type of the tensor."""
        return self.dtype

    @property
    def name(self):
        op = self.op
        if op.num_outputs == 1:
            return op.name
        return f"{op.name}.v{self.value_index}"


@tvm_ffi.register_object("te.Operation")
class Operation(tvm_ffi.Object):
    """Represent an operation that generates a tensor"""

    def output(self, index):
        """Get the index-th output of the operation

        Parameters
        ----------
        index : int
            The index size.

        Returns
        -------
        out : Tensor
            The i-th output.
        """
        return _ffi_api.OpGetOutput(self, index)

    @property
    def num_outputs(self):
        """Number of outputs from this op."""
        return _ffi_api.OpNumOutputs(self)

    @property
    def input_tensors(self):
        """List of input tensors to this op."""
        return _ffi_api.OpInputTensors(self)


@tvm_ffi.register_object("te.PlaceholderOp")
class PlaceholderOp(Operation):
    """Placeholder operation."""


@tvm_ffi.register_object("te.BaseComputeOp")
class BaseComputeOp(Operation):
    """Compute operation."""


@tvm_ffi.register_object("te.ComputeOp")
class ComputeOp(BaseComputeOp):
    """Scalar operation."""


@tvm_ffi.register_object("te.ScanOp")
class ScanOp(Operation):
    """Scan operation."""


@tvm_ffi.register_object("te.ExternOp")
class ExternOp(Operation):
    """External operation."""

    body: SeqStmt
