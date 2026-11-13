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
"""Sampling operators."""

import tvm_ffi

from tvm_tirx.ir import Attrs
from tvm_tirx.ir import Call as _Call
from tvm_tirx.ir.attrs import make_node as _make_attrs
from tvm_tirx.ir.location import UNKNOWN_LOC, Location

from ..expr import Expr


@tvm_ffi.register_object("relax.attrs.MultinomialFromUniformAttrs")
class MultinomialFromUniformAttrs(Attrs):
    """Attributes for multinomial_from_uniform operator"""


def multinomial_from_uniform(
    prob: Expr,
    uniform_sample: Expr,
    sample_indices: Expr,
    dtype: str = "int64",
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Returns a tensor where each row contains the index sampled from the multinomial
    probability distribution located in the corresponding row of tensor prob.

    Notes
    -----
    For better cpu performance, use 'vm.builtin.multinomial_from_uniform'.
    For accurate results, ensure probabilities are between 0 and 1 and sum to 1.

    Parameters
    ----------
    prob : relax.Expr
        A 2-D tensor of shape (batch, vocab_size) representing probability distributions.
        Each row is a distribution across vocabulary for a batch, where:
        Values range from [0, 1], indicating the probability of each vocabulary item.
        The sum of values in each row is 1, forming a valid distribution.

    uniform_sample : relax.Expr
        The uniformly sampled 2-D tensor with the shape (n, 1).
        Values range from 0 to 1, indicating probabilities sampled uniformly.

    sample_indices : relax.Expr
        The 2-D tensor with the shape [n, 1], which indicates the specific
        probability distribution to sample from. The value of sample_indices[i]
        determines that the ith token should be sampled from the sample_indices[i]th
        probability distribution. For instance, if there are 3 distinct probability
        distributions and the requirement is to sample 2, 3, and 4 tokens from each,
        then sample_indices would be [0, 0, 1, 1, 1, 2, 2, 2, 2].

    dtype : str
        The data type of the output tensor.

    Returns
    -------
    result : relax.Expr
        The computed tensor with shape (n, 1).

    Examples
    --------
    .. code-block:: python

        prob = [[0.2, 0.3, 0.5], [0.3, 0.4, 0.3]]
        usample = [[0.4], [0.9]]
        sample_indices = [[0], [1]]

        multinomial_from_uniform(prob, usample)
        -> [[1], [2]]
        multinomial_from_uniform(prob, usample, sample_indices)
        -> [[1], [2]]

    """

    return _Call(
        "relax.multinomial_from_uniform",
        [prob, uniform_sample, sample_indices],
        attrs=_make_attrs("relax.attrs.MultinomialFromUniformAttrs", dtype=dtype),
        loc=loc,
        ty=ty,
    )
