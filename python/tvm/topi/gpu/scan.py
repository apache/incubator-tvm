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
# pylint: disable=invalid-name, too-many-locals, too-many-statements
"Scan related operators"

import operator
from collections.abc import Callable

import tvm
import tvm_tirx
from tvm import te
from tvm.contrib.thrust import can_use_rocthrust, can_use_thrust
from tvm_tirx.script.ir_builder import IRBuilder
from tvm_tirx.tirx.script import ir_builder as T

from ..math import cast, ceil_log2
from ..transform import expand_dims, reshape, squeeze, transpose
from ..utils import ceil_div, get_const_int, prod, swap

_THRUST_SUM_SCAN = "tvm.contrib.thrust.sum_scan"


def _get_thrust_func_name(tvmop):
    if tvmop is not operator.add:
        raise ValueError(f"{tvmop} not supported by thrust")
    return _THRUST_SUM_SCAN


def _can_use_scan_thrust(binop, *, target=None):
    """
    Check if scan_thrust can be utilized based on the current target and binary op.
    """
    if target is None:
        return False
    return binop is operator.add and any(
        [
            can_use_thrust(target, _THRUST_SUM_SCAN),
            can_use_rocthrust(target, _THRUST_SUM_SCAN),
        ]
    )


def exclusive_scan_ir(
    data, output, reduction=None, binop=operator.add, identity_value=0, *, target=None
):
    """Low level IR to do exclusive sum scan along rows of 2D input.

    Parameters
    ----------
    data : Buffer
        Input N-D Buffer. Scan is done over the innermost axis.

    output: Buffer
        A buffer to store the output scan, of the same shape as data

    reduction: Buffer, optional
        (N-1)-D Buffer, to store the sum of each scan axis.

    binop: function, optional
        A binary associative op to use for scan. The function takes two TIR expressions
        and produce a new TIR expression. By default it uses ``operator.add`` to compute prefix
        sum.

    identity_value: int or float
        A value for the binary operation which provides the identity property. E.g. if * is
        your operator and i is the identity_value then a * i = a for all a in the domain of
        your operation.
    """

    batch_size = cast(prod(data.shape[:-1]), "int32")
    scan_axis_size = cast(data.shape[-1], "int32")

    with IRBuilder() as ib:
        out_dtype = output.dtype

        max_threads = int(target.attrs["max_num_threads"])

        with T.if_(scan_axis_size == 0):
            with T.then_():
                with T.launch_thread("blockIdx.x", batch_size) as bx:
                    with T.if_(bx < batch_size):
                        with T.then_():
                            if reduction is not None:
                                T.tensor_store(
                                    reduction,
                                    T.tensor_indices(reduction, bx),
                                    cast(identity_value, out_dtype),
                                )
            with T.else_():
                nthread_tx = max_threads
                blocks_per_batch = ceil_div(scan_axis_size, max_threads)

                # Flatten the batch and scan-block axes into blockIdx.x.  On CUDA,
                # blockIdx.y is limited to 65535 even when blockIdx.x can be much larger.
                # Copy data to output
                with T.frame_scope(
                    [
                        T.launch_thread("threadIdx.x", nthread_tx),
                        T.launch_thread("blockIdx.x", blocks_per_batch * batch_size),
                    ]
                ) as (tx, bx):
                    batch = tvm_tirx.tirx.indexdiv(bx, blocks_per_batch)
                    tid = tvm_tirx.tirx.indexmod(bx, blocks_per_batch) * nthread_tx + tx
                    with T.if_(tid < scan_axis_size):
                        with T.then_():
                            T.tensor_store(
                                output,
                                T.tensor_indices(output, batch * scan_axis_size + tid),
                                cast(
                                    data[T.tensor_indices(data, batch * scan_axis_size + tid)],
                                    out_dtype,
                                ),
                            )

                # The following algorithm performs parallel exclusive scan
                # Up Sweep of exclusive scan
                lim = ceil_log2(scan_axis_size, target=target)

                with T.serial(0, cast(lim, "int32")) as l2_width:
                    width = 2 << l2_width

                    blocks_per_batch = cast(ceil_div(scan_axis_size, max_threads * width), "int32")
                    start_buf = T.bind_(
                        T.alloc_tensor(ty_args=[T.Tensor([1], "int32", scope="local")])
                    )
                    middle_buf = T.bind_(
                        T.alloc_tensor(ty_args=[T.Tensor([1], "int32", scope="local")])
                    )
                    end_buf = T.bind_(
                        T.alloc_tensor(ty_args=[T.Tensor([1], "int32", scope="local")])
                    )
                    with T.frame_scope(
                        [
                            T.launch_thread("threadIdx.x", nthread_tx),
                            T.launch_thread("blockIdx.x", blocks_per_batch * batch_size),
                        ]
                    ) as (tx, bx):
                        batch = tvm_tirx.tirx.indexdiv(bx, blocks_per_batch)
                        tid = tvm_tirx.tirx.indexmod(bx, blocks_per_batch) * nthread_tx + tx
                        start = start_buf
                        middle = middle_buf
                        end = end_buf
                        T.tensor_store(start, T.tensor_indices(start, 0), width * tid)
                        with T.if_(start[T.tensor_indices(start, 0)] < scan_axis_size):
                            with T.then_():
                                T.tensor_store(
                                    middle,
                                    T.tensor_indices(middle, 0),
                                    start[T.tensor_indices(start, 0)]
                                    + tvm_tirx.tirx.indexdiv(width, 2),
                                )
                                T.tensor_store(
                                    end,
                                    T.tensor_indices(end, 0),
                                    tvm.te.min(
                                        start[T.tensor_indices(start, 0)] + width, scan_axis_size
                                    ),
                                )
                                with T.if_(middle[T.tensor_indices(middle, 0)] < scan_axis_size):
                                    with T.then_():
                                        T.tensor_store(
                                            output,
                                            T.tensor_indices(
                                                output,
                                                (
                                                    batch * scan_axis_size
                                                    + end[T.tensor_indices(end, 0)]
                                                    - 1
                                                ),
                                            ),
                                            binop(
                                                output[
                                                    T.tensor_indices(
                                                        output,
                                                        (
                                                            batch * scan_axis_size
                                                            + end[T.tensor_indices(end, 0)]
                                                            - 1
                                                        ),
                                                    )
                                                ],
                                                output[
                                                    T.tensor_indices(
                                                        output,
                                                        (
                                                            batch * scan_axis_size
                                                            + middle[T.tensor_indices(middle, 0)]
                                                            - 1
                                                        ),
                                                    )
                                                ],
                                            ),
                                        )

                # Down Sweep of exclusive scan
                with T.launch_thread("blockIdx.x", batch_size) as bx:
                    with T.if_(bx < batch_size):
                        with T.then_():
                            if reduction is not None:
                                T.tensor_store(
                                    reduction,
                                    T.tensor_indices(reduction, bx),
                                    output[
                                        T.tensor_indices(output, ((bx + 1) * scan_axis_size - 1))
                                    ],
                                )
                            T.tensor_store(
                                output,
                                T.tensor_indices(output, ((bx + 1) * scan_axis_size - 1)),
                                cast(identity_value, out_dtype),
                            )

                with T.serial(0, cast(lim, "int32")) as l2_width:
                    width = 2 << (lim - l2_width - 1)

                    blocks_per_batch = cast(ceil_div(scan_axis_size, max_threads * width), "int32")
                    start_buf = T.bind_(
                        T.alloc_tensor(ty_args=[T.Tensor([1], "int32", scope="local")])
                    )
                    middle_buf = T.bind_(
                        T.alloc_tensor(ty_args=[T.Tensor([1], "int32", scope="local")])
                    )
                    end_buf = T.bind_(
                        T.alloc_tensor(ty_args=[T.Tensor([1], "int32", scope="local")])
                    )
                    tmp_buf = T.bind_(
                        T.alloc_tensor(ty_args=[T.Tensor([1], out_dtype, scope="local")])
                    )
                    with T.frame_scope(
                        [
                            T.launch_thread("threadIdx.x", nthread_tx),
                            T.launch_thread("blockIdx.x", blocks_per_batch * batch_size),
                        ]
                    ) as (tx, bx):
                        batch = tvm_tirx.tirx.indexdiv(bx, blocks_per_batch)
                        tid = tvm_tirx.tirx.indexmod(bx, blocks_per_batch) * nthread_tx + tx
                        start = start_buf
                        middle = middle_buf
                        end = end_buf
                        tmp = tmp_buf
                        T.tensor_store(start, T.tensor_indices(start, 0), width * tid)
                        with T.if_(
                            tvm_tirx.tirx.all(start[T.tensor_indices(start, 0)] < scan_axis_size)
                        ):
                            with T.then_():
                                T.tensor_store(
                                    middle,
                                    T.tensor_indices(middle, 0),
                                    start[T.tensor_indices(start, 0)]
                                    + tvm_tirx.tirx.indexdiv(width, 2),
                                )
                                T.tensor_store(
                                    end,
                                    T.tensor_indices(end, 0),
                                    tvm_tirx.tirx.min(
                                        start[T.tensor_indices(start, 0)] + width, scan_axis_size
                                    ),
                                )
                                with T.if_(middle[T.tensor_indices(middle, 0)] < scan_axis_size):
                                    with T.then_():
                                        T.tensor_store(
                                            tmp,
                                            T.tensor_indices(tmp, 0),
                                            output[
                                                T.tensor_indices(
                                                    output,
                                                    (
                                                        batch * scan_axis_size
                                                        + middle[T.tensor_indices(middle, 0)]
                                                        - 1
                                                    ),
                                                )
                                            ],
                                        )
                                        T.tensor_store(
                                            output,
                                            T.tensor_indices(
                                                output,
                                                (
                                                    batch * scan_axis_size
                                                    + middle[T.tensor_indices(middle, 0)]
                                                    - 1
                                                ),
                                            ),
                                            output[
                                                T.tensor_indices(
                                                    output,
                                                    (
                                                        batch * scan_axis_size
                                                        + end[T.tensor_indices(end, 0)]
                                                        - 1
                                                    ),
                                                )
                                            ],
                                        )
                                        T.tensor_store(
                                            output,
                                            T.tensor_indices(
                                                output,
                                                (
                                                    batch * scan_axis_size
                                                    + end[T.tensor_indices(end, 0)]
                                                    - 1
                                                ),
                                            ),
                                            binop(
                                                output[
                                                    T.tensor_indices(
                                                        output,
                                                        (
                                                            batch * scan_axis_size
                                                            + end[T.tensor_indices(end, 0)]
                                                            - 1
                                                        ),
                                                    )
                                                ],
                                                tmp[T.tensor_indices(tmp, 0)],
                                            ),
                                        )

        return ib.get()


def get_reduction_from_exclusive_scan(data, ex_scan_output, binop=operator.add, *, target=None):
    """Return the sum of the last element of data and the exclusive scan output.
    The is the reduction of data along each row (for 2-D case).

    Parameters
    ----------
    data : tvm.te.Tensor
        Input data of any shape

    ex_scan_output : tvm.te.Tensor
        The output of exclusive scan on data

    binop: function, optional
        A binary associative op to use for scan. The function takes two TIR expressions
        and produce a new TIR expression. By default it uses ``operator.add`` to compute prefix
        sum.

    Returns
    -------
    reduction : tvm.te.Tensor
        (N-1)-D tensor storing the reduction of each scan axis.
    """
    ndim = len(data.shape)
    if ndim == 1:
        data = expand_dims(data, axis=0)
        ex_scan_output = expand_dims(ex_scan_output, axis=0)

    def ir(data_buf, data_ex_scan_buf, reduction_buf):
        batch_size = cast(prod(data_buf.shape[:-1]), "int32")
        scan_axis_size = cast(data_buf.shape[-1], "int32")

        max_threads = int(target.attrs["max_num_threads"])

        with IRBuilder() as ib:
            data = data_buf
            data_ex_scan = data_ex_scan_buf
            reduction = reduction_buf

            nthread_tx = max_threads
            nthread_bx = ceil_div(batch_size, max_threads)
            with T.frame_scope(
                [
                    T.launch_thread("threadIdx.x", nthread_tx),
                    T.launch_thread("blockIdx.x", nthread_bx),
                ]
            ) as (tx, bx):
                tid = bx * max_threads + tx
                with T.if_(tid < batch_size):
                    with T.then_():
                        with T.if_(scan_axis_size > 0):
                            with T.then_():
                                T.tensor_store(
                                    reduction,
                                    T.tensor_indices(reduction, tid),
                                    binop(
                                        data_ex_scan[
                                            T.tensor_indices(
                                                data_ex_scan,
                                                (tid * scan_axis_size + scan_axis_size - 1),
                                            )
                                        ],
                                        data[
                                            T.tensor_indices(
                                                data, (tid * scan_axis_size + scan_axis_size - 1)
                                            )
                                        ],
                                    ),
                                )
                            with T.else_():
                                T.tensor_store(
                                    reduction,
                                    T.tensor_indices(reduction, tid),
                                    cast(0, reduction_buf.dtype),
                                )

            return ib.get()

    data_buf = tvm_tirx.tirx.Var(
        "valid_indices_buf",
        tvm_tirx.tirx.TensorType(data.shape, data.dtype, data_alignment=8, layout=None),
    )
    ex_scan_output_buf = tvm_tirx.tirx.Var(
        "ex_scan_output_buf",
        tvm_tirx.tirx.TensorType(
            ex_scan_output.shape, ex_scan_output.dtype, data_alignment=8, layout=None
        ),
    )

    reduction = te.extern(
        [data.shape[:-1]],
        [data, ex_scan_output],
        lambda ins, outs: ir(ins[0], ins[1], outs[0]),
        dtype=[ex_scan_output.dtype],
        in_tensors=[data_buf, ex_scan_output_buf],
        name="ex_scan_reduction",
        tag="ex_scan_reduction_gpu",
    )

    if ndim == 1:
        return squeeze(reduction, 0)

    return reduction


def scan_thrust(
    data,
    output_dtype,
    exclusive=True,
    return_reduction=False,
    binop=operator.add,
    workspace=None,
    *,
    target=None,
):
    """Do exclusive or inclusive scan on 1D or multidimensional input, using thrust.

    Parameters
    ----------
    data : tvm.te.Tensor
        Input data of any shape. The scan is done over the innermost axis.

    output_dtype: string
        The dtype of the output scan tensor.

    exclusive: bool, optional
        Whether or not do exclusive or inclusive scan.

    return_reduction: bool, optional
        Whether or not return a (N-1)-D tensor storing the reduction of each scan axis.
        Reductions are computed as part of the upsweep pass, so there is no extra cost.
        If False, reductions are ignored. It must be False when exclusive is False.

    binop: function, optional
        A binary associative op to use for scan. Since we need to lookup the corresponding
        thrust function, arbitrariy callables are not supported. Currently only
        ``operator.add`` can be passed in.

    workspace: Optional[tvm.te.Tensor]
        A buffer to store intermediate results. The size of the workspace should be sufficiently
        large, this can be obtained by overestimation or memory usage profiling. If None, it will
        fallback to use thrust internal memory allocation.

    Returns
    -------
    output : tvm.te.Tensor
        A N-D tensor of the same rank N and shape as the input data.

    reduction : tvm.te.Tensor, optional
        (N-1)-D tensor storing the reduction of each scan axis.
        Returned if return_reduction is True.
    """
    data_buf = tvm_tirx.tirx.Var(
        "data_buf", tvm_tirx.tirx.TensorType(data.shape, data.dtype, data_alignment=8, layout=None)
    )
    output_buf = tvm_tirx.tirx.Var(
        "output_buf",
        tvm_tirx.tirx.TensorType(
            data.shape,
            "float32" if output_dtype is None else output_dtype,
            data_alignment=8,
            layout=None,
        ),
    )

    workspace_buf = (
        tvm_tirx.tirx.Var(
            "workspace_buf",
            tvm_tirx.tirx.TensorType(
                workspace.shape, workspace.dtype, data_alignment=8, layout=None
            ),
        )
        if workspace is not None
        else None
    )

    def f_compute(ins, outs):
        args = [_get_thrust_func_name(binop), ins[0], outs[0], exclusive]
        if workspace is not None:
            args.append(ins[1])
        return tvm_tirx.tirx.call_packed(*args)

    output = te.extern(
        [data.shape],
        [data] if workspace is None else [data, workspace],
        f_compute,
        dtype=[output_dtype],
        in_tensors=[data_buf] if workspace is None else [data_buf, workspace_buf],
        out_tensors=[output_buf],
        name="exclusive_scan_thrust",
        tag="exclusive_scan_thrust_gpu",
    )

    if return_reduction:
        assert exclusive, "return_reduction should be False for inclusive scan"
        reduction = get_reduction_from_exclusive_scan(data, output, binop, target=target)
        return output, reduction

    return output


def exclusive_scan(
    data,
    axis=-1,
    return_reduction=False,
    output_dtype=None,
    binop=operator.add,
    identity_value=0,
    workspace=None,
    *,
    target=None,
):
    """Do exclusive scan on 1D or multidimensional input.

    Parameters
    ----------
    data : tvm.te.Tensor
        Input data of any shape.

    axis: int, optional
        The axis to do scan on. By default, scan is done on the innermost axis.

    return_reduction: bool, optional
        Whether or not return a tensor storing the reduction over each scan axis.
        If the input rank is N, this tensor is of rank N - 1.
        Reductions are computed as part of the upsweep pass, so there is no extra cost.
        If False, reductions are ignored.

    output_dtype: string, optional
        The dtype of the output scan tensor. If not provided, the dtype of the input is used.

    binop: function, optional
        A binary associative op to use for scan. The function takes two TIR expressions
        and produce a new TIR expression. By default it uses ``operator.add`` to compute prefix
        sum.

    identity_value: int or float
        A value for the binary operation which provides the identity property. E.g. if * is
        your operator and i is the identity_value then a * i = a for all a in the domain of
        your operation.

    workspace: Optional[tvm.te.Tensor]
        A buffer to store intermediate results if thrust is enabled. The size of the workspace
        should be sufficiently large, this can be obtained by overestimation or memory usage
        profiling. If None, it will fallback to use thrust internal memory allocation.

    Returns
    -------
    output : tvm.te.Tensor
        A N-D tensor of the same rank N and shape as the input data.

    reduction : tvm.te.Tensor, optional
        (N-1)-D tensor storing the reduction of each scan axis.
        Returned if return_reduction is True.
    """

    def do_scan(data, output_dtype):
        # TODO: add support for a prod_scan
        if _can_use_scan_thrust(binop, target=target):
            return scan_thrust(
                data,
                output_dtype,
                exclusive=True,
                return_reduction=return_reduction,
                binop=binop,
                workspace=workspace,
                target=target,
            )

        if ndim == 1:
            # TIR exclusive scan accepts only 2D or higher-rank inputs.
            data = expand_dims(data, axis=0)

        data_buf = tvm_tirx.tirx.Var(
            "data_buf",
            tvm_tirx.tirx.TensorType(data.shape, data.dtype, data_alignment=8, layout=None),
        )
        output_buf = tvm_tirx.tirx.Var(
            "output_buf",
            tvm_tirx.tirx.TensorType(
                data.shape,
                "float32" if output_dtype is None else output_dtype,
                data_alignment=8,
                layout=None,
            ),
        )

        if return_reduction:
            output, reduction = te.extern(
                [data.shape, data.shape[:-1]],
                [data],
                lambda ins, outs: exclusive_scan_ir(
                    ins[0],
                    outs[0],
                    outs[1],
                    binop=binop,
                    identity_value=identity_value,
                    target=target,
                ),
                dtype=[output_dtype, output_dtype],
                in_tensors=[data_buf],
                name="exclusive_scan",
                tag="exclusive_scan_gpu",
            )
        else:
            output = te.extern(
                [data.shape],
                [data],
                lambda ins, outs: exclusive_scan_ir(
                    ins[0], outs[0], binop=binop, identity_value=identity_value, target=target
                ),
                dtype=[output_dtype],
                in_tensors=[data_buf],
                out_tensors=[output_buf],
                name="exclusive_scan",
                tag="exclusive_scan_gpu",
            )
            reduction = None

        if ndim == 1:
            output = squeeze(output, 0)
            if return_reduction:
                reduction = squeeze(reduction, 0)

        if return_reduction:
            return output, reduction

        return output

    if output_dtype is None or output_dtype == "":
        output_dtype = data.dtype

    ndim = len(data.shape)
    if axis < 0:
        axis += ndim

    # If scan axis is not the innermost one, swap the scan and the innermost axes
    # Scan is always done on the innermost axis, for performance reason.
    if axis != ndim - 1:
        axes = swap(list(range(ndim)), axis)
        data = transpose(data, axes)

    if return_reduction:
        output, reduction = do_scan(data, output_dtype)
    else:
        output = do_scan(data, output_dtype)

    if axis != ndim - 1:
        axes = swap(list(range(ndim)), axis)
        output = transpose(output, axes)

    if return_reduction:
        return output, reduction

    return output


def inclusive_scan(
    data,
    axis=-1,
    output_dtype=None,
    binop=operator.add,
    identity_value=0,
    workspace=None,
    *,
    target=None,
):
    """Do inclusive scan on 1D or multidimensional input.

    Parameters
    ----------
    data : tvm.te.Tensor
        Input data of any shape.

    axis: int, optional
        The axis to do scan on. By default, scan is done on the innermost axis.

    output_dtype: string, optional
        The dtype of the output scan tensor. If not provided, the dtype of the input is used.

    binop: function, optional
        A binary associative op to use for scan. The function takes two TIR expressions
        and produce a new TIR expression. By default it uses ``operator.add`` to compute prefix
        sum.

    identity_value: int or float
        A value for the binary operation which provides the identity property. E.g. if * is
        your operator and i is the identity_value then a * i = a for all a in the domain of
        your operation.

    workspace: Optional[tvm.te.Tensor]
        A buffer to store intermediate results if thrust is enabled. The size of the workspace
        should be sufficiently large, this can be obtained by overestimation or memory usage
        profiling. If None, it will fallback to use thrust internal memory allocation.

    Returns
    -------
    output : tvm.te.Tensor
        A N-D tensor of the same rank N as the input data.
    """

    if _can_use_scan_thrust(binop, target=target):
        if output_dtype is None or output_dtype == "":
            output_dtype = data.dtype
        ndim = len(data.shape)
        if axis < 0:
            axis += ndim

        if axis != ndim - 1:
            axes = swap(list(range(ndim)), axis)
            data = transpose(data, axes)
        output = scan_thrust(
            data, output_dtype, exclusive=False, binop=binop, workspace=workspace, target=target
        )
        if axis != ndim - 1:
            axes = swap(list(range(ndim)), axis)
            output = transpose(output, axes)
        return output

    ex_scan = exclusive_scan(
        data,
        axis,
        output_dtype=output_dtype,
        binop=binop,
        identity_value=identity_value,
        workspace=workspace,
        target=target,
    )

    if output_dtype is not None and data.dtype != output_dtype and output_dtype != "":
        data = cast(data, output_dtype)

    return binop(data, ex_scan)


def scanop(
    data: tvm.te.Tensor,
    binop: Callable[["tvm.Expr", "tvm.Expr"], "tvm.Expr"],
    identity_value: float | int,
    axis: int | None = None,
    dtype: str | None = None,
    exclusive: bool | None = None,
    workspace: tvm.te.Tensor | None = None,
    *,
    target=None,
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

    identity_value: int or float
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
        If true will return exclusive cumulative operation in which the first element is not
        included. In other terms, if true, the j-th output element would be
        the cumulative operation of the first (j-1) elements. Otherwise, it would be the
        cumulative operation of the first j elements.

    workspace: Optional[tvm.te.Tensor]

    Returns
    -------
    result : tvm.te.Tensor
        The result has the same size as data, and the same shape as data if axis is not None.
        If axis is None, the result is a 1-d array.
    """
    if axis is None:
        axis = 0
        data = reshape(data, (prod(data.shape),))
    axis = get_const_int(axis)
    if exclusive is not None and exclusive:
        return exclusive_scan(
            data,
            axis,
            output_dtype=dtype,
            binop=binop,
            identity_value=identity_value,
            workspace=workspace,
            target=target,
        )
    return inclusive_scan(
        data,
        axis,
        output_dtype=dtype,
        binop=binop,
        identity_value=identity_value,
        workspace=workspace,
        target=target,
    )


def cumsum(
    data: tvm.te.Tensor,
    axis: int | None = None,
    dtype: int | None = None,
    exclusive: bool | None = None,
    workspace: tvm.te.Tensor | None = None,
    *,
    target=None,
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
        If true will return exclusive sum in which the first element is not
        included. In other terms, if true, the j-th output element would be
        the sum of the first (j-1) elements. Otherwise, it would be the sum of
        the first j elements.

    workspace: Optional[tvm.te.Tensor]
        A buffer to store intermediate results if thrust is enabled. The size of the workspace
        should be sufficiently large, this can be obtained by overestimation or memory usage
        profiling. If None, it will fallback to use thrust internal memory allocation.

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
        axis=axis,
        dtype=dtype,
        exclusive=exclusive,
        workspace=workspace,
        target=target,
    )


def cumprod(
    data: tvm.te.Tensor,
    axis: int | None = None,
    dtype: int | None = None,
    exclusive: bool | None = None,
    workspace: tvm.te.Tensor | None = None,
    *,
    target=None,
):
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

    workspace: Optional[tvm.te.Tensor]
        A buffer to store intermediate results if thrust is enabled. The size of the workspace
        should be sufficiently large, this can be obtained by overestimation or memory usage
        profiling. If None, it will fallback to use thrust internal memory allocation.

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
        axis=axis,
        dtype=dtype,
        exclusive=exclusive,
        workspace=workspace,
        target=target,
    )
