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
# pylint: disable=invalid-name
"""Scan (cumulative binary) operators"""

import operator
from collections.abc import Callable

import tvm
from tvm_tirx.script.ir_builder import IRBuilder
from tvm_tirx.tirx import TensorType, Var
from tvm_tirx.tirx.script import ir_builder as T

from ..te import extern
from . import utils
from .math import cast


def scanop(
    data: tvm.te.Tensor,
    binop: Callable[["tvm.Expr", "tvm.Expr"], "tvm.Expr"],
    identity_value: "tvm.Expr",
    op_name: str,
    axis: int | None = None,
    dtype: str | None = None,
    exclusive: bool | None = None,
) -> tvm.te.Tensor:
    """Cumulative binary operator (scan) with similar axis behavior as np.cumsum and np.cumprod.

    See cumprod and cumsum for an example of use.

    E.g. if * is your binary operator and the input tensor is [1, 2, 3, 4] the output may be
    [1, 1 * 2, 1 * 2 * 3, 1 * 2 * 3 * 4]

    Parameters
    ----------
    data : tvm.te.Tensor
        The input data to the operator.

    binop: Callable (tvm.Expr, tvm.Expr) -> tvm.Expr
        A binary operator which should be associative and commutative. E.g. if * is your
        operator then a * (b * c) = (a * b) * c and a * b = b * a

    identity_value: tvm.Expr
        A value for the binary operation which provides the identity property. E.g. if * is
        your operator and i is the identity_value then a * i = a for all a in the domain of
        your operation.

    axis : int, optional
        Axis along which the operation is computed. The default (None) is to compute
        the cumulative operation over the flattened array.

    dtype : string, optional
        Type of the returned array and of the accumulator in which the elements are computed.
        If dtype is not specified, it defaults to the dtype of data.

    exclusive : bool, optional
        If True will return exclusive cumulative operation in which the first element is not
        included. In other terms, if True, the j-th output element would be
        the cumulative operation of the first (j-1) elements. Otherwise, it would be the
        cumulative operation of the first j elements. The cumulative operation of zero elements
        is assumed to be the identity_value.

    Returns
    -------
    result : tvm.te.Tensor
        The result has the same size as data, and the same shape as data if axis is not None.
        If axis is None, the result is a 1-d array.
    """
    if dtype is None or dtype == "":
        dtype = data.dtype

    if exclusive is None:
        exclusive = False

    def maybe_cast(x):
        if dtype != data.dtype:
            return cast(x, dtype)
        return x

    axis_mul_before = 1
    axis_mul_after = 1

    if axis is None:
        axis = 0
        cumsum_axis_len = utils.prod(data.shape)
        shape = (cumsum_axis_len,)
    else:
        if not isinstance(axis, int):
            axis = utils.get_const_int(axis)

        shape = data.shape
        cumsum_axis_len = shape[axis]

        if axis < 0:
            axis = len(shape) + axis

        for i, value in enumerate(shape, 0):
            if i < axis:
                axis_mul_before *= value
            elif i > axis:
                axis_mul_after *= value

    def gen_ir(data_buf, out_buf):
        with IRBuilder() as ib:
            with T.parallel(0, axis_mul_before * axis_mul_after) as fused:
                i = fused // axis_mul_after
                j = fused % axis_mul_after
                base_idx = i * cumsum_axis_len * axis_mul_after + j
                if exclusive:
                    T.tensor_store(
                        out_buf, T.tensor_indices(out_buf, base_idx), cast(identity_value, dtype)
                    )
                else:
                    T.tensor_store(
                        out_buf,
                        T.tensor_indices(out_buf, base_idx),
                        maybe_cast(data_buf[T.tensor_indices(data_buf, base_idx)]),
                    )
                with T.serial(0, cumsum_axis_len - 1) as _k:
                    k = _k + 1
                    cur_idx = base_idx + k * axis_mul_after
                    prev_idx = base_idx + (k - 1) * axis_mul_after
                    if exclusive:
                        T.tensor_store(
                            out_buf,
                            T.tensor_indices(out_buf, cur_idx),
                            binop(
                                out_buf[T.tensor_indices(out_buf, prev_idx)],
                                maybe_cast(data_buf[T.tensor_indices(data_buf, prev_idx)]),
                            ),
                        )
                    else:
                        T.tensor_store(
                            out_buf,
                            T.tensor_indices(out_buf, cur_idx),
                            binop(
                                out_buf[T.tensor_indices(out_buf, prev_idx)],
                                maybe_cast(data_buf[T.tensor_indices(data_buf, cur_idx)]),
                            ),
                        )

            return ib.get()

    out_buf = Var("out_buf", TensorType(shape, dtype))

    return extern(
        [shape],
        [data],
        lambda ins, outs: gen_ir(ins[0], outs[0]),
        dtype=dtype,
        out_tensors=[out_buf],
        name=op_name,
        tag=op_name,
    )


def cumsum(
    data: tvm.te.Tensor,
    axis: int | None = None,
    dtype: str | None = None,
    exclusive: bool | None = None,
) -> tvm.te.Tensor:
    """Numpy style cumsum op. Return the cumulative sum of the elements along a given axis.

    Parameters
    ----------
    data : tvm.te.Tensor
        The input data to the operator.

    axis : int, optional
        Axis along which the cumulative sum is computed. The default (None) is to compute
        the cumsum over the flattened array.

    dtype : string, optional
        Type of the returned array and of the accumulator in which the elements are summed.
        If dtype is not specified, it defaults to the dtype of data.

    exclusive : bool, optional
        If True, will return exclusive sum in which the first element is not
        included. In other terms, if True, the j-th output element would be
        the sum of the first (j-1) elements. Otherwise, it would be the sum of
        the first j elements.

    Returns
    -------
    result : tvm.te.Tensor
        The result has the same size as data, and the same shape as data if axis is not None.
        If axis is None, the result is a 1-d array.
    """
    return scanop(
        data=data,
        binop=operator.add,
        identity_value=0,
        op_name="cumsum_generic",
        axis=axis,
        dtype=dtype,
        exclusive=exclusive,
    )


def cumprod(
    data: tvm.te.Tensor,
    axis: int | None = None,
    dtype: int | None = None,
    exclusive: bool | None = None,
) -> tvm.te.Tensor:
    """Numpy style cumprod op. Return the cumulative product of the elements along a given axis.

    Parameters
    ----------
    data : tvm.te.Tensor
        The input data to the operator.

    axis : int, optional
        Axis along which the cumulative product is computed. The default (None) is to compute
        the cumproduct over the flattened array.

    dtype : string, optional
        Type of the returned array and of the accumulator in which the elements are multiplied.
        If dtype is not specified, it defaults to the dtype of data.

    exclusive : bool, optional
        If True, will return exclusive product in which the first element is not
        included. In other terms, if True, the j-th output element would be
        the product of the first (j-1) elements. Otherwise, it would be the product of
        the first j elements.

    Returns
    -------
    result : tvm.te.Tensor
        The result has the same size as data, and the same shape as data if axis is not None.
        If axis is None, the result is a 1-d array.
    """
    return scanop(
        data=data,
        binop=operator.mul,
        identity_value=1,
        op_name="cumprod_generic",
        axis=axis,
        dtype=dtype,
        exclusive=exclusive,
    )
