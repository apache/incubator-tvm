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
"""Binary Neural Network (BNN) Operators"""

import tvm_tirx
from tvm import te

from .. import tag
from ..utils import get_const_int, simplify


def binarize_pack(data, axis=None, name="PackedInput"):
    """Binarization and bit-packing along a certain axis.

    Parameters
    ----------
    data : tvm.te.Tensor
        n-D input, can be any layout.

    axis : None or int
        The axis along which to do binarization and bit-packing,
        default is the last axis.

    name : str, optional
        The name prefix operators generate.

    Returns
    -------
    output : tvm.te.Tensor
        n-D, the same layout as input, dtype is uint32.
    """
    ishape = data.shape
    if axis is None:
        axis = len(ishape) - 1
    assert get_const_int(ishape[axis]) % 32 == 0
    n = len(ishape)
    oshape = tuple(simplify(ishape[i] // 32) if i == axis else ishape[i] for i in range(n))

    def _binarize_pack(*indices):
        start_idx = [indices[i] * 32 if i == axis else indices[i] for i in range(n)]
        packed = tvm_tirx.tirx.const(0, "uint32")
        for j in range(32):
            idx = [start_idx[i] + j if i == axis else start_idx[i] for i in range(n)]
            sign = (data(*idx) >= 0).astype("uint32")
            packed = packed | sign
            if j == 31:
                return packed
            packed = packed << 1
        raise RuntimeError("not resach")

    return te.compute(oshape, _binarize_pack, name=name, tag="binarize_pack")


def binary_dense(data, weight):
    """Binary matrix multiplication using xor and bit-count.

    Parameters
    ----------
    data : tvm.te.Tensor
        2-D with shape [batch, in_dim], dtype is uint32.

    weight : tvm.te.Tensor
        2-D with shape [out_dim, in_dim], dtype is uint32.

    Returns
    -------
    output : tvm.te.Tensor
        2-D with shape [batch, out_dim], dtype is float32.
    """
    assert data.dtype == "uint32" and weight.dtype == "uint32", (
        "dtype of data and weight should be uint32"
    )
    assert len(data.shape) == 2 and len(weight.shape) == 2, "only support 2-dim binary dense"
    batch, in_dim = data.shape
    out_dim, _ = weight.shape
    k = te.reduce_axis((0, in_dim), name="k")
    matmul = te.compute(
        (batch, out_dim),
        lambda i, j: te.sum(tvm_tirx.tirx.popcount(data[i, k] ^ weight[j, k]), axis=k),
        tag="binary_dense",
    )

    return te.compute(
        (batch, out_dim), lambda i, j: 32 * in_dim - 2.0 * matmul(i, j), tag=tag.ELEMWISE
    )
