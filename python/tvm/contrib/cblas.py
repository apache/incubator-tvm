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
"""External function interface to BLAS libraries."""

import tvm_tirx
from tvm import te


def matmul(lhs, rhs, transa=False, transb=False, **kwargs):
    """Create an extern op that compute matrix mult of A and rhs with CrhsLAS
    This function serves as an example on how to call external libraries.

    Parameters
    ----------
    lhs: Tensor
        The left matrix operand
    rhs: Tensor
        The right matrix operand
    transa: bool
        Whether transpose lhs
    transb: bool
        Whether transpose rhs

    Returns
    -------
    C: Tensor
        The result tensor.
    """
    n = lhs.shape[1] if transa else lhs.shape[0]
    m = rhs.shape[0] if transb else rhs.shape[1]
    return te.extern(
        (n, m),
        [lhs, rhs],
        lambda ins, outs: tvm_tirx.tirx.call_packed(
            "tvm.contrib.cblas.matmul", ins[0], ins[1], outs[0], transa, transb
        ),
        name="C",
        **kwargs,
    )


def batch_matmul(lhs, rhs, transa=False, transb=False, iterative=False, **kwargs):
    """Create an extern op that compute batched matrix mult of A and rhs with CBLAS
    This function serves as an example on how to call external libraries.

    Parameters
    ----------
    lhs: Tensor
        The left matrix operand
    rhs: Tensor
        The right matrix operand
    transa: bool
        Whether transpose lhs
    transb: bool
        Whether transpose rhs

    Returns
    -------
    C: Tensor
        The result tensor.
    """
    b = te.max(lhs.shape[0], rhs.shape[0])
    n = lhs.shape[2] if transa else lhs.shape[1]
    m = rhs.shape[1] if transb else rhs.shape[2]
    return te.extern(
        (b, n, m),
        [lhs, rhs],
        lambda ins, outs: tvm_tirx.tirx.call_packed(
            "tvm.contrib.cblas.batch_matmul"
            if not iterative
            else "tvm.contrib.cblas.batch_matmul_iterative",
            ins[0],
            ins[1],
            outs[0],
            transa,
            transb,
        ),
        name="C",
        **kwargs,
    )
