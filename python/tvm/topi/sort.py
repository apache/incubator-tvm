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
# pylint: disable=too-many-arguments
"""Argsort operator"""

import tvm
import tvm_tirx
from tvm import te

from .utils import get_const_tuple


def sort(data, axis=-1, is_ascend=1):
    """Performs sorting along the given axis and returns an array
    in sorted order.

    Parameters
    ----------
    data : tvm.te.Tensor
        The input tensor.

    axis : int, optional
        Axis along which to sort the input tensor.
        By default the flattened array is used.

    is_ascend : boolean, optional
        Whether to sort in ascending or descending order.

    dtype : string, optional
        DType of the output indices.

    Returns
    -------
    out : tvm.te.Tensor
        Sorted index tensor.

    """
    data_buf = tvm_tirx.tirx.Var(
        "data_buf", tvm_tirx.tirx.TensorType(data.shape, data.dtype, data_alignment=8, layout=None)
    )
    out_buf = tvm_tirx.tirx.Var(
        "out_buf", tvm_tirx.tirx.TensorType(data.shape, data.dtype, data_alignment=8, layout=None)
    )
    out = te.extern(
        data.shape,
        [data],
        lambda ins, outs: tvm_tirx.tirx.call_packed(
            "tvm.contrib.sort.sort", ins[0], outs[0], axis, is_ascend
        ),
        dtype=data.dtype,
        in_tensors=[data_buf],
        out_tensors=out_buf,
        name="sort_cpu",
        tag="sort_cpu",
    )
    return out


def argsort(data, valid_count=None, axis=-1, is_ascend=1, dtype="float32"):
    """Performs sorting along the given axis and returns an array
    of indices having the same shape as an input array that index
    data in sorted order.

    Parameters
    ----------
    data : tvm.te.Tensor
        The input tensor.

    valid_count : tvm.te.Tensor, optional
        1-D tensor for valid number of boxes.

    axis : int, optional
        Axis along which to sort the input tensor.
        By default the flattened array is used.

    is_ascend : boolean, optional
        Whether to sort in ascending or descending order.

    dtype : string, optional
        DType of the output indices.

    Returns
    -------
    out : tvm.te.Tensor
        Sorted index tensor.

    Example
    --------
    .. code-block:: python

        # An example to use argsort
        dshape = (1, 5, 6)
        data = te.placeholder(dshape, name="data")
        axis = 0
        is_ascend = False
        out = argsort(data, axis=axis, is_ascend=is_ascend)
        np_data = np.random.uniform(dshape)
        s = topi.generic.schedule_argsort(out)
        f = tvm.compile(s, [data, out], "llvm")
        dev = tvm_tirx.cpu()
        tvm_data = tvm_tirx.runtime.tensor(np_data, dev)
        tvm_out = tvm_tirx.runtime.tensor(np.zeros(dshape, dtype=data.dtype.dtype), dev)
        f(tvm_data, tvm_out)
    """
    data_buf = tvm_tirx.tirx.Var(
        "data_buf", tvm_tirx.tirx.TensorType(data.shape, data.dtype, data_alignment=8, layout=None)
    )
    if valid_count is not None:
        valid_count_buf = tvm_tirx.tirx.Var(
            "valid_count_buf",
            tvm_tirx.tirx.TensorType(
                valid_count.shape, valid_count.dtype, data_alignment=4, layout=None
            ),
        )
        out_buf = tvm_tirx.tirx.Var(
            "out_buf", tvm_tirx.tirx.TensorType(data.shape, "int32", data_alignment=8, layout=None)
        )
        out = te.extern(
            data.shape,
            [data, valid_count],
            lambda ins, outs: tvm_tirx.tirx.call_packed(
                "tvm.contrib.sort.argsort_nms", ins[0], ins[1], outs[0], axis, is_ascend
            ),
            dtype="int32",
            in_tensors=[data_buf, valid_count_buf],
            out_tensors=out_buf,
            name="argsort_nms_cpu",
            tag="argsort_nms_cpu",
        )
    else:
        out_buf = tvm_tirx.tirx.Var(
            "out_buf",
            tvm_tirx.tirx.TensorType(
                data.shape, "float32" if dtype is None else dtype, data_alignment=8, layout=None
            ),
        )
        out = te.extern(
            data.shape,
            [data],
            lambda ins, outs: tvm_tirx.tirx.call_packed(
                "tvm.contrib.sort.argsort", ins[0], outs[0], axis, is_ascend
            ),
            dtype=dtype,
            in_tensors=[data_buf],
            out_tensors=out_buf,
            name="argsort_cpu",
            tag="argsort_cpu",
        )
    return out


def topk(data, k=1, axis=-1, ret_type="both", is_ascend=False, dtype="int64"):
    """Get the top k elements in an input tensor along the given axis.

    Parameters
    ----------
    data : tvm.te.Tensor
        The input tensor.

    k : int or tvm.te.Tensor, optional
        Number of top elements to select. Return all elements if k < 1.

    axis : int, optional
        Axis long which to sort the input tensor.

    ret_type: str, optional
        The return type [both, values, indices].
        "both": return both top k data and indices.
        "values": return top k data only.
        "indices": return top k indices only.

    is_ascend : boolean, optional
        Whether to sort in ascending or descending order.

    dtype : string, optional
        The data type of the indices output.

    Returns
    -------
    out : tvm.te.Tensor or List[tvm.te.Tensor]
        The computed result.
    """
    assert ret_type in ["both", "values", "indices"]
    data_buf = tvm_tirx.tirx.Var(
        "data_buf", tvm_tirx.tirx.TensorType(data.shape, data.dtype, data_alignment=8, layout=None)
    )
    out_shape = list(get_const_tuple(data.shape))
    kvar = tvm.te.var("k")
    if not isinstance(k, int):
        out_shape[axis] = kvar
    elif k >= 1:
        out_shape[axis] = k
    out_bufs = []
    if ret_type in ["both", "values"]:
        out_bufs.append(
            tvm_tirx.tirx.Var(
                "value_buf",
                tvm_tirx.tirx.TensorType(out_shape, data.dtype, data_alignment=8, layout=None),
            )
        )
    if ret_type in ["both", "indices"]:
        out_bufs.append(
            tvm_tirx.tirx.Var(
                "indices_buf",
                tvm_tirx.tirx.TensorType(
                    out_shape, "float32" if dtype is None else dtype, data_alignment=8, layout=None
                ),
            )
        )
    out_shapes = [out_shape] * len(out_bufs)

    kv = kvar if not isinstance(k, int) else k
    out = te.extern(
        out_shapes,
        [data],
        lambda ins, outs: tvm_tirx.tirx.call_packed(
            "tvm.contrib.sort.topk", ins[0], *outs, kv, axis, ret_type, is_ascend
        ),
        in_tensors=[data_buf],
        out_tensors=out_bufs,
        name="topk_cpu",
        tag="topk_cpu",
    )
    return out
