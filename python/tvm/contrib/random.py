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
"""External function interface to random library."""

import tvm_ffi

import tvm_tirx
from tvm import te


def randint(low, high, size, dtype="int32"):
    """Return random integers from low (inclusive) to high (exclusive).
    Return random integers from the "discrete uniform" distribution of the
    specified dtype in the "half-open" interval [low, high).

    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution
    high : int
        One above the largest (signed) integer to be drawn from the distribution

    Returns
    -------
    out : Tensor
        A tensor with specified size and dtype
    """
    assert "int" in dtype, "the type of randint output must be int or uint"
    return te.extern(
        size,
        [],
        lambda ins, outs: tvm_tirx.tirx.call_packed(
            "tvm.contrib.random.randint", int(low), int(high), outs[0]
        ),
        dtype=dtype,
    )


def uniform(low, high, size):
    """Draw samples from a uniform distribution.

    Samples are uniformly distributed over the half-open interval [low, high)
    (includes low, but excludes high). In other words, any value within the
    given interval is equally likely to be drawn by uniform.

    Parameters
    ----------
    low : float
        Lower boundary of the output interval. All values generated will be
        greater than or equal to low.
    high : float
        Upper boundary of the output interval. All values generated will be
        less than high.
    size : tuple of ints
        Output shape. If the given shape is, e.g., (m, n, k), then m * n * k
        samples are drawn.

    Returns
    -------
    out : Tensor
        A tensor with specified size and dtype.
    """
    return te.extern(
        size,
        [],
        lambda ins, outs: tvm_tirx.tirx.call_packed(
            "tvm.contrib.random.uniform", float(low), float(high), outs[0]
        ),
        dtype="float32",
    )


def normal(loc, scale, size):
    """Draw samples from a normal distribution.

    Return random samples from a normal distribution.

    Parameters
    ----------
    loc : float
        loc of the distribution.
    scale : float
        Standard deviation of the distribution.
    size : tuple of ints
        Output shape. If the given shape is, e.g., (m, n, k), then m * n * k
        samples are drawn.

    Returns
    ------
    out : Tensor
        A tensor with specified size and dtype
    """
    return te.extern(
        size,
        [],
        lambda ins, outs: tvm_tirx.tirx.call_packed(
            "tvm.contrib.random.normal", float(loc), float(scale), outs[0]
        ),
        dtype="float32",
    )


tvm_ffi.init_ffi_api("tvm.contrib.random")
