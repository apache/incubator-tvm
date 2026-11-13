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
"""Dilation operators"""

import tvm_tirx
from tvm import te

from .. import tag, utils


@te.tag_scope(tag=tag.INJECTIVE + ",dilate")
def dilate(data, strides, dilation_value=0.0, name="DilatedInput"):
    """Dilate data with given dilation value (0 by default).

    Parameters
    ----------
    data : tvm.te.Tensor
        n-D, can be any layout.

    strides : list / tuple of n ints
        Dilation stride on each dimension, 1 means no dilation.

    dilation_value : int/float, optional
        Value used to dilate the input.

    name : str, optional
        The name prefix operators generated

    Returns
    -------
    Output : tvm.te.Tensor
        n-D, the same layout as data.
    """
    n = len(data.shape)
    if len(strides) != n:
        raise ValueError(f"data dimension and strides size dismatch : {n} vs {len(strides)}")
    ana = tvm_tirx.sym.Analyzer()
    out_shape = tuple(ana.simplify((data.shape[i] - 1) * strides[i] + 1) for i in range(n))

    def _dilate(*indices):
        not_zero = []
        index_tuple = []
        idxdiv = tvm_tirx.tirx.indexdiv
        idxmod = tvm_tirx.tirx.indexmod
        for i in range(n):
            if not utils.equal_const_int(strides[i], 1):
                index_tuple.append(idxdiv(indices[i], strides[i]))
                not_zero.append(idxmod(indices[i], strides[i]).equal(0))
            else:
                index_tuple.append(indices[i])
        if not_zero:
            not_zero = tvm_tirx.tirx.all(*not_zero)
            return tvm_tirx.tirx.if_then_else(
                not_zero, data(*index_tuple), tvm_tirx.tirx.const(dilation_value, data.dtype)
            )
        return data(*index_tuple)

    return te.compute(out_shape, _dilate, name=name)
