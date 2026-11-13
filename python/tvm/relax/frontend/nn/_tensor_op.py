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
# ruff: noqa: F821
"""Adding member operators to nn.Tensor."""

import tvm_tirx


def _op():
    from tvm.relax.frontend.nn import op  # pylint: disable=import-outside-toplevel

    return op


def _convert_scalar(scalar, ref) -> "Tensor":
    from .core import Tensor  # pylint: disable=import-outside-toplevel

    if isinstance(scalar, Tensor):
        return scalar
    if isinstance(scalar, tvm_tirx.ir.FloatImm | tvm_tirx.ir.IntImm):
        return Tensor.from_scalar(scalar.value, dtype=ref.dtype)
    if isinstance(scalar, int | float):
        return Tensor.from_scalar(scalar, dtype=ref.dtype)
    return scalar


class _TensorOp:
    def __add__(self, other):
        other = _convert_scalar(other, self)
        return _op().add(self, other)

    def __radd__(self, other):
        other = _convert_scalar(other, self)
        return _op().add(self, other)

    def __sub__(self, other):
        other = _convert_scalar(other, self)
        return _op().subtract(self, other)

    def __rsub__(self, other):
        other = _convert_scalar(other, self)
        return _op().subtract(other, self)

    def __mul__(self, other):
        other = _convert_scalar(other, self)
        return _op().multiply(self, other)

    def __rmul__(self, other):
        other = _convert_scalar(other, self)
        return _op().multiply(self, other)

    def __truediv__(self, other):
        other = _convert_scalar(other, self)
        return _op().divide(self, other)

    def __lt__(self, other):
        other = _convert_scalar(other, self)
        return _op().less(self, other)

    def __le__(self, other):
        other = _convert_scalar(other, self)
        return _op().less_equal(self, other)

    def __gt__(self, other):
        other = _convert_scalar(other, self)
        return _op().greater(self, other)

    def __ge__(self, other):
        other = _convert_scalar(other, self)
        return _op().greater_equal(self, other)

    def astype(self, dtype):
        return _op().astype(self, dtype)

    def maximum(self, other):
        other = _convert_scalar(other, self)
        return _op().maximum(self, other)

    def minimum(self, other):
        other = _convert_scalar(other, self)
        return _op().minimum(self, other)

    def reshape(self, *shape):
        return _op().reshape(self, shape)

    def permute_dims(self, *axes):
        return _op().permute_dims(self, axes)

    def repeat(self, repeats: int, axis: int | None = None):
        return _op().repeat(self, repeats, axis)
