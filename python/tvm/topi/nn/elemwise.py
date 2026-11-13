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
"""Elementwise operators"""

import tvm
import tvm_tirx
from tvm import te

from .. import tag
from ..utils import get_const_int


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def relu(x):
    """Take relu of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: tvm.te.max(x(*i), tvm_tirx.tirx.const(0, x.dtype)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def leaky_relu(x, alpha):
    """Take leaky relu of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    alpha : float
        The slope for the small gradient when x < 0

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """

    def _compute(*indices):
        value = x(*indices)
        calpha = tvm_tirx.tirx.const(alpha, value.ty)
        return tvm_tirx.tirx.Select(value > 0, value, value * calpha)

    return te.compute(x.shape, _compute)


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def softplus(x, beta=1.0, threshold=20.0):
    """Compute Softplus activation for input x with numerical stability.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input tensor.

    beta : float, optional
        The scaling factor β in the Softplus formula (default is 1.0).

    threshold : float, optional
        The threshold value for numerical stability (default is 20.0).

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """

    def _compute(*indices):
        value = x(*indices)
        b = tvm_tirx.tirx.const(beta, value.ty)
        t = tvm_tirx.tirx.const(threshold, value.ty)

        return tvm_tirx.tirx.Select(
            b * value > t, value, (1 / b) * tvm_tirx.tirx.log(1 + tvm_tirx.tirx.exp(b * value))
        )

    return te.compute(x.shape, _compute)


@tvm.te.tag_scope(tag=tag.BROADCAST)
def prelu(x, slope, axis=1):
    """PReLU.
    It accepts two arguments: an input ``x`` and a weight array ``W``
    and computes the output as :math:`PReLU(x) y = x > 0 ? x : W * x`,
    where :math:`*` is an elementwise multiplication for each sample in the
    batch.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    slope : tvm.te.Tensor
        Channelised slope tensor for prelu

    axis : int
        The axis where the channel data needs to be applied

    Returns
    -------
    y : tvm.te.Tensor
        The result.

    Links
    -----
    [http://arxiv.org/pdf/1502.01852v1.pdf]
    """

    assert len(slope.shape) == 1
    assert axis < len(x.shape)
    if slope.shape[0] == 1:
        slope = te.compute(
            (get_const_int(x.shape[axis]),), lambda c: slope[0], name="slope_broadcasted"
        )
    assert get_const_int(slope.shape[0]) == get_const_int(x.shape[axis])

    def _compute_channelwise(*indices):
        xval = x(*indices)
        return tvm_tirx.tirx.Select(xval > 0, xval, xval * slope(indices[axis]))

    return te.compute(x.shape, _compute_channelwise)
