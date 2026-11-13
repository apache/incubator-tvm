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
# pylint: disable=invalid-name, too-many-locals, too-many-arguments
"""Bitserial Dense operator."""

import tvm_tirx
from tvm import te
from tvm.topi.utils import get_const_tuple

from .bitserial_util import bitpack


def bitserial_dense(
    data, weight, data_bits, weight_bits, pack_dtype="uint32", out_dtype="int16", unipolar=True
):
    """The default implementation of bitserial dense in topi.

    Parameters
    ----------
    data : tvm.te.Tensor
        2-D with shape [batch, in_dim]
    weight : tvm.te.Tensor
        2-D with shape [out_dim, in_dim] or
        3-D with shape [out_dim, weight_bits, in_dim]
    Returns
    -------
    output : tvm.te.Tensor
        2-D with shape [batch, out_dim]
    """
    data_packed = bitpack(data, data_bits, pack_axis=1, bit_axis=1, pack_type=pack_dtype)
    if len(weight.shape) == 2:
        weight_packed = bitpack(weight, weight_bits, pack_axis=1, bit_axis=1, pack_type=pack_dtype)
    else:
        weight_packed = weight
    Y, DB, K = get_const_tuple(data_packed.shape)
    X, WB, _ = get_const_tuple(weight_packed.shape)

    oshape = (Y, X)
    k = te.reduce_axis((0, K), name="k")
    db = te.reduce_axis((0, DB), name="db")
    wb = te.reduce_axis((0, WB), name="wb")

    matmul_unipolar = te.compute(
        oshape,
        lambda i, j: te.sum(
            (
                tvm_tirx.tirx.popcount(weight_packed[j, wb, k] & data_packed[i, db, k])
                - tvm_tirx.tirx.popcount(~weight_packed[j, wb, k] & data_packed[i, db, k])
            ).astype(out_dtype)
            << (db + wb).astype(out_dtype),
            axis=[wb, db, k],
        ),
        tag="bitserial_dense_unipolar",
    )

    matmul = te.compute(
        oshape,
        lambda i, j: te.sum(
            tvm_tirx.tirx.popcount(weight_packed[j, wb, k] & data_packed[i, db, k]).astype(
                out_dtype
            )
            << (db + wb).astype(out_dtype),
            axis=[wb, db, k],
        ),
        tag="bitserial_dense",
    )

    if unipolar:
        return matmul_unipolar
    return matmul
