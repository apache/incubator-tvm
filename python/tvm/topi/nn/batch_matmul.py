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
# ruff: noqa: E731
"""Batch matrix multiplication"""

# pylint: disable=invalid-name
import logging

import tvm_tirx
from tvm import te

from ..utils import get_const_tuple

logger = logging.getLogger("topi")


def batch_matmul(
    tensor_a,
    tensor_b,
    oshape=None,
    out_dtype=None,
    transpose_a=False,
    transpose_b=True,
    auto_scheduler_rewritten_layout="",
    meta_schedule_original_shape=None,
):
    """Compute batch matrix multiplication of `tensor_a` and `tensor_b`.

    Both `tensor_a` and `tensor_b` can be transposed. For legacy reason, we use NT format
    (transpose_a=False, transpose_b=True) by default.

    Parameters
    ----------
    tensor_a : tvm.te.Tensor
        3-D with shape [batch, M, K] or [batch, K, M].

    tensor_b : tvm.te.Tensor
        3-D with shape [batch, K, N] or [batch, N, K].

    oshape : List[Optional]
        Explicit intended output shape of the computation. Can be useful in cases
        with dynamic input shapes.

    out_dtype : Optional[str]
        Specifies the output data type for mixed precision batch matmul.

    transpose_a : Optional[bool] = False
        Whether the first tensor is in transposed format.

    transpose_b : Optional[bool] = True
        Whether the second tensor is in transposed format.

    auto_scheduler_rewritten_layout: Optional[str] = ""
        The layout after auto-scheduler's layout rewrite pass.

    meta_schedule_original_shape: Optional[List[Expr]] = None
        The original shape of the tensor

    Returns
    -------
    output : tvm.te.Tensor
        3-D with shape [batch, M, N]
    """
    assert len(tensor_a.shape) == 3, "tensor_a only support 3-dim"
    if transpose_a:
        XB, XK, XI = get_const_tuple(tensor_a.shape)
    else:
        XB, XI, XK = get_const_tuple(tensor_a.shape)
    if auto_scheduler_rewritten_layout:
        raise RuntimeError("LEGACY-FLOW triggered, to be removed")
    if meta_schedule_original_shape:
        raise RuntimeError("LEGACY-FLOW triggered, to be removed")

    assert len(tensor_b.shape) == 3, "tensor_b only support 3-dim"
    if transpose_b:
        YB, YJ, YK = get_const_tuple(tensor_b.shape)
    else:
        YB, YK, YJ = get_const_tuple(tensor_b.shape)

    assert XK == YK or tvm_tirx.ir.is_prim_var(YK), "shapes of x and y are inconsistent"
    k = te.reduce_axis((0, XK), name="k")
    if oshape is None:
        assert XB == YB or XB == 1 or YB == 1, "batch dimension doesn't match"
        batch = (
            tvm_tirx.tirx.expr.Var("batch", "int32")
            if tvm_tirx.ir.is_prim_var(XB) or tvm_tirx.ir.is_prim_var(YB)
            else te.max(XB, YB)
        )
        oshape = (batch, XI, YJ)
    if out_dtype is None:
        out_dtype = tensor_a.dtype
        if tensor_a.dtype != tensor_b.dtype:
            logger.warning(
                "tensor_a has different data type with tensor_b: %s, %s",
                tensor_a.dtype,
                tensor_b.dtype,
            )

    if (transpose_a, transpose_b) == (True, True):
        compute_lambda = lambda b, i, j: te.sum(
            tensor_a[b if XB != 1 else 0, k, i].astype(out_dtype)
            * tensor_b[b if YB != 1 else 0, j, k].astype(out_dtype),
            axis=k,
        )
        compute_name = "T_batch_matmul_TT"
    elif (transpose_a, transpose_b) == (True, False):
        compute_lambda = lambda b, i, j: te.sum(
            tensor_a[b if XB != 1 else 0, k, i].astype(out_dtype)
            * tensor_b[b if YB != 1 else 0, k, j].astype(out_dtype),
            axis=k,
        )
        compute_name = "T_batch_matmul_TN"
    elif (transpose_a, transpose_b) == (False, True):
        compute_lambda = lambda b, i, j: te.sum(
            tensor_a[b if XB != 1 else 0, i, k].astype(out_dtype)
            * tensor_b[b if YB != 1 else 0, j, k].astype(out_dtype),
            axis=k,
        )
        compute_name = "T_batch_matmul_NT"
    else:  # (transpose_a, transpose_b) == (False, False):
        compute_lambda = lambda b, i, j: te.sum(
            tensor_a[b if XB != 1 else 0, i, k].astype(out_dtype)
            * tensor_b[b if YB != 1 else 0, k, j].astype(out_dtype),
            axis=k,
        )
        compute_name = "T_batch_matmul_NN"

    output = te.compute(
        oshape,
        compute_lambda,
        name=compute_name,
        tag="batch_matmul",
        attrs={"layout_free_placeholders": [tensor_b]},
    )
    if auto_scheduler_rewritten_layout:
        raise RuntimeError("LEGACY-FLOW triggered, to be removed")

    return output
