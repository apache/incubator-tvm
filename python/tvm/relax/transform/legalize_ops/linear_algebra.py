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
"""Default legalization function for linear algebra operators."""

import tvm_tirx
from tvm import relax, te, topi
from tvm_tirx import DataTypeCode, tirx
from tvm_tirx.ir import Call

from ...block_builder import BlockBuilder
from ...expr import Expr, Tuple, TupleGetItem, Var
from .common import register_legalize


@register_legalize("relax.matmul")
def _matmul(bb: BlockBuilder, call: Call) -> Expr:
    def is_known_tensor_dtype(dtype) -> bool:
        raw_dtype = dtype.dtype
        return not (
            raw_dtype.type_code == int(DataTypeCode.HANDLE)
            and raw_dtype.bits == 0
            and raw_dtype.lanes == 0
        )

    def te_matmul(a: te.Tensor, b: te.Tensor) -> te.Tensor:
        a_shape = list(a.shape)
        b_shape = list(b.shape)
        a_prepended = False
        b_appended = False
        if len(a_shape) == 1:
            a_prepended = True
            a_shape.insert(0, 1)
        if len(b_shape) == 1:
            b_appended = True
            b_shape.append(1)

        is_a_larger = len(a_shape) > len(b_shape)
        offset = len(a_shape) - len(b_shape) if is_a_larger else len(b_shape) - len(a_shape)

        a_relax = relax.Var("a", relax.TensorType(a.shape))
        b_relax = relax.Var("b", relax.TensorType(b.shape))
        f_infer_ty = call.op.get_attr("relax.FInferTypeWithBuilder")
        output_shape = f_infer_ty(relax.op.matmul(a_relax, b_relax), bb).shape
        if isinstance(a_shape[-1], tvm_tirx.ir.IntImm) and a_shape[-1] == 0:
            return te.compute(
                output_shape,
                lambda *_: tirx.const(0, call.ty.dtype),
                name="matmul",
            )

        def matmul_compute(*idx_spatial):
            k = te.reduce_axis((0, a_shape[-1]), name="k")

            def multiply_compute(idx_reduce):
                a_indices = []
                b_indices = []

                for i in range(offset):
                    if is_a_larger:
                        a_indices.append(idx_spatial[i])
                    else:
                        b_indices.append(idx_spatial[i])
                for i in range(offset, len(output_shape) - (2 - a_prepended - b_appended)):
                    a_dim = a_shape[i if is_a_larger else i - offset]
                    b_dim = b_shape[i if not is_a_larger else i - offset]
                    dim_equal = a_dim == b_dim
                    if not isinstance(dim_equal, tvm_tirx.ir.IntImm) or dim_equal == 0:
                        a_dim_is_one = isinstance(a_dim, tvm_tirx.ir.IntImm) and a_dim == 1
                        b_dim_is_one = isinstance(b_dim, tvm_tirx.ir.IntImm) and b_dim == 1
                        a_indices.append(0 if a_dim_is_one else idx_spatial[i])
                        b_indices.append(0 if b_dim_is_one else idx_spatial[i])
                    else:
                        a_indices.append(idx_spatial[i])
                        b_indices.append(idx_spatial[i])

                if not a_prepended:
                    a_indices.append(idx_spatial[-2 + b_appended])
                a_indices.append(idx_reduce)
                b_indices.append(idx_reduce)
                if not b_appended:
                    b_indices.append(idx_spatial[-1])

                dtype = call.attrs.out_dtype
                if dtype is not None and dtype != "":
                    return a(*a_indices).astype(dtype) * b(*b_indices).astype(dtype)
                return a(*a_indices) * b(*b_indices)

            return te.sum(multiply_compute(k), axis=k)

        return te.compute(
            output_shape,
            lambda *idx: matmul_compute(*idx),  # pylint: disable=unnecessary-lambda
            name="matmul",
        )

    lhs, rhs = call.args
    lhs_ty = call.args[0].ty
    rhs_ty = call.args[1].ty
    assert (
        lhs_ty.dtype
        and rhs_ty.dtype
        and is_known_tensor_dtype(lhs_ty.dtype)
        and is_known_tensor_dtype(rhs_ty.dtype)
    ), (
        f"To legalize R.matmul into R.call_tir, the dtype of both operands must be known.  "
        f"However, the LHS {lhs} has type {lhs_ty} (dtype='{lhs_ty.dtype}') "
        f"and the RHS {rhs} has type {rhs_ty} (dtype='{rhs_ty.dtype}')."
    )
    return bb.call_te(te_matmul, call.args[0], call.args[1], function_name_hint="matmul")


@register_legalize("relax.einsum")
def _einsum(bb: BlockBuilder, call: Call) -> Expr:
    t = call.args[0]
    n_field = len(t.ty.fields)
    while isinstance(t, Var):
        binding = bb.lookup_binding(t)
        if not isinstance(binding, Tuple | Var):
            break
        t = binding

    assert isinstance(t, Tuple | Var)
    fields = (
        t.fields if isinstance(t, Tuple) else [bb.emit(TupleGetItem(t, i)) for i in range(n_field)]
    )
    return bb.call_te(topi.einsum, call.attrs.subscripts, *fields)


@register_legalize("relax.outer")
def _outer(bb: BlockBuilder, call: Call) -> Expr:
    def te_outer(a: te.Tensor, b: te.Tensor) -> te.Tensor:
        a_shape = list(a.shape)
        b_shape = list(b.shape)
        assert len(a_shape) == 1 and len(b_shape) == 1, "outer requires 1D tensors"

        n = a_shape[0]
        m = b_shape[0]

        def compute_fn(i, j):
            return a[i] * b[j]

        return te.compute((n, m), compute_fn, name="outer")

    lhs, rhs = call.args
    return bb.call_te(te_outer, lhs, rhs, function_name_hint="outer")
