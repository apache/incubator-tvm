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
# pylint: disable=invalid-name,unused-argument
# ruff: noqa: E741, F821
"""TVM operator fully connected compute."""

import tvm_tirx
from tvm import te

from .. import add, tag


def matmul(
    tensor_a,
    tensor_b,
    bias=None,
    out_dtype=None,
    transpose_a=False,
    transpose_b=False,
    auto_scheduler_rewritten_layout="",
    meta_schedule_original_shape=None,
):
    """The default implementation of matmul in topi.

    Parameters
    ----------
    tensor_a : tvm.te.Tensor
        2-D with shape [batch, in_dim]

    tensor_b : tvm.te.Tensor
        2-D with shape [out_dim, in_dim]

    bias : Optional[tvm.te.Tensor]
        1-D with shape [out_dim]

    out_dtype : Optional[str]
        The output type. This is used for mixed precision.

    transpose_a : Optional[bool] = False
        Whether the tensor_a is in transposed format.

    transpose_b : Optional[bool] = False
        Whether the tensor_b is in transposed format.

    auto_scheduler_rewritten_layout: Optional[str] = ""
        The layout after auto-scheduler's layout rewrite pass.

    meta_schedule_original_shape: Optional[List[Expr]] = None
        The original shape of the input tensor.

    Returns
    -------
    output : tvm.te.Tensor
        2-D with shape [batch, out_dim]
    """
    # TODO(yixin): support cases for 1-dim input
    # TODO(yixin): adding support and further check for >2-dim input in autotvm template
    assert len(tensor_a.shape) >= 2 and len(tensor_b.shape) >= 2, (
        "1-dim matmul is not supported yet."
    )

    if bias is not None:
        assert len(bias.shape) == 1
    if out_dtype is None:
        out_dtype = tensor_a.dtype
    if transpose_a:
        reduce_dim_a, in_dim = tensor_a.shape[-2:]
    else:
        in_dim, reduce_dim_a = tensor_a.shape[-2:]
    batch_dims_a = tensor_a.shape[:-2]

    if auto_scheduler_rewritten_layout:
        # Infer shape for the rewritten layout
        raise RuntimeError("LEGACY-FLOW triggered, to be removed")
    if meta_schedule_original_shape:
        raise RuntimeError("LEGACY-FLOW triggered, to be removed")

    if transpose_b:
        out_dim, reduce_dim_b = tensor_b.shape[-2:]
    else:
        reduce_dim_b, out_dim = tensor_b.shape[-2:]
    batch_dims_b = tensor_b.shape[:-2]

    if not tvm_tirx.ir.is_prim_var(reduce_dim_a) and not tvm_tirx.ir.is_prim_var(reduce_dim_b):
        assert int(reduce_dim_a) == int(reduce_dim_b), (
            f"Reduction dimensions of dense do not match. {reduce_dim_a} vs {reduce_dim_b}."
        )

    result_ndim = max(len(batch_dims_a), len(batch_dims_b))
    batch_dims_a = [1] * (result_ndim - len(batch_dims_a)) + batch_dims_a
    batch_dims_b = [1] * (result_ndim - len(batch_dims_b)) + batch_dims_b

    for idx, (l, r) in enumerate(zip(batch_dims_a, batch_dims_b)):
        if (
            not tvm_tirx.ir.is_prim_var(l)
            and not tvm_tirx.ir.is_prim_var(r)
            and int(l) != 1
            and int(r) != 1
        ):
            assert int(l) == int(r), (
                "Batch dimensions of dense do not match: "
                f"{tensor_a.shape[:-2]} vs {tensor_b.shape[:-2]}."
            )
        if not tvm_tirx.ir.is_prim_var(l) and int(l) == 1:
            batch_dims_a[idx] = batch_dims_b[idx]

    k = te.reduce_axis((0, reduce_dim_a), name="k")

    def compute(*indices):
        batch_indices_a = indices[-len(tensor_a.shape) : -2]
        batch_indices_a = [
            i if tvm_tirx.ir.is_prim_var(dim) or int(dim) != 1 else 0
            for i, dim in zip(batch_indices_a, tensor_a.shape[:-2])
        ]
        batch_indices_b = indices[-len(tensor_b.shape) : -2]
        batch_indices_b = [
            i if tvm_tirx.ir.is_prim_var(dim) or int(dim) != 1 else 0
            for i, dim in zip(batch_indices_b, tensor_b.shape[:-2])
        ]
        i, j = indices[-2:]
        a_indices = (*batch_indices_a, k, i) if transpose_a else (*batch_indices_a, i, k)
        b_indices = (*batch_indices_b, j, k) if transpose_b else (*batch_indices_b, k, j)
        return te.sum(
            tensor_a[a_indices].astype(out_dtype) * tensor_b[b_indices].astype(out_dtype), axis=k
        )

    compute_name = {
        (True, True): "T_matmul_TT",
        (True, False): "T_matmul_TN",
        (False, True): "T_matmul_NT",
        (False, False): "T_matmul_NN",
    }[(transpose_a, transpose_b)]

    # TODO(jcf94): Remove `dense` when `matmul` is finally ready
    compute_tag = "dense" if (transpose_a, transpose_b) == (False, True) else "matmul"

    mat = te.compute(
        (*batch_dims_a, in_dim, out_dim),
        compute,
        name=compute_name,
        tag=compute_tag,
        attrs={"layout_free_placeholders": [tensor_b]},
    )

    if bias is not None:
        mat = add(mat, bias.astype(out_dtype))

    if auto_scheduler_rewritten_layout:
        raise RuntimeError("LEGACY-FLOW triggered, to be removed")

    return mat


def dense(
    data,
    weight,
    bias=None,
    out_dtype=None,
    auto_scheduler_rewritten_layout="",
    meta_schedule_original_shape=None,
):
    """The default implementation of dense in topi.
    This is an alias of matmul_nt operator for data tensor in non-transposed format and weight
    tensor in transposed format.

    Parameters
    ----------
    data : tvm.te.Tensor
        2-D with shape [batch, in_dim]

    weight : tvm.te.Tensor
        2-D with shape [out_dim, in_dim]

    bias : Optional[tvm.te.Tensor]
        1-D with shape [out_dim]

    out_dtype : Optional[str]
        The output type. This is used for mixed precision.

    auto_scheduler_rewritten_layout: str = ""
        The layout after auto-scheduler's layout rewrite pass.

    meta_schedule_original_shape: Optional[List[Expr]] = None
        The original shape of the input tensor.

    Returns
    -------
    output : tvm.te.Tensor
        2-D with shape [batch, out_dim]
    """

    return matmul(
        data,
        weight,
        bias,
        out_dtype,
        False,
        True,
        auto_scheduler_rewritten_layout,
        meta_schedule_original_shape,
    )


def dense_pack(data, weight, bias=None, out_dtype=None):
    """The default implementation of dense_pack in topi.

    Parameters
    ----------
    data : tvm.te.Tensor
        2-D with shape [batch, in_dim]

    weight : tvm.te.Tensor
        2-D with shape [out_dim, in_dim]

    bias : Optional[tvm.te.Tensor]
        1-D with shape [out_dim]

    out_dtype : Optional[str]
        The output type. This is used for mixed precision.

    Returns
    -------
    output : tvm.te.Tensor
        2-D with shape [batch, out_dim]
    """
    if out_dtype is None:
        out_dtype = data.dtype
    M, K = get_const_tuple(data.shape)  # batch, in_dim
    N, _, packw_bn = get_const_tuple(weight.shape)  # out_dim
    N = N * packw_bn

    idxdiv = tvm_tirx.tirx.indexdiv
    idxmod = tvm_tirx.tirx.indexmod
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda y, x: te.sum(
            data[y, k].astype(out_dtype)
            * weight[idxdiv(x, packw_bn), k, idxmod(x, packw_bn)].astype(out_dtype),
            axis=k,
        ),
        name="T_dense_pack",
        tag="dense_pack",
    )
    if bias is not None:
        C = te.compute((M, N), lambda i, j: C[i, j] + bias[j].astype(out_dtype), tag=tag.BROADCAST)
    return C
