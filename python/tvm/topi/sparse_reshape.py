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
# pylint: disable=invalid-name, too-many-arguments, too-many-nested-blocks
"""Sparse_Reshape operator"""

from tvm.te import div, extern, floordiv, floormod
from tvm_tirx.script.ir_builder import IRBuilder
from tvm_tirx.tirx import Cast, TensorType, Var
from tvm_tirx.tirx.script import ir_builder as T


def sparse_reshape(
    sparse_indices,
    prev_shape,
    new_shape,
    new_sparse_indices_shape,
    new_shape_shape,
):
    """
    Reshape a Sparse Tensor

    Parameters
    ----------
    sparse_indices : te.Expr
        A 2-D tensor[N, n_dim] of integers containing location of sparse values, where N is the
        number of sparse values and n_dim is the number of dimensions of the dense_shape

    prev_shape : te.Expr
        A 1-D tensor containing the previous shape of the dense tensor

    new_shape : te.Expr
        A 1-D tensor containing the new shape of the dense tensor

    Returns
    -------
    result: te.Expr
        Output tensor.

    Examples
    --------
    .. code-block:: python

        sparse_indices = [[0, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 0],
                            [1, 2, 3]]
        prev_shape = [2, 3, 4]
        new_shape = [9, -1]
        new_sparse_indices, new_shape = topi.sparse_reshape(
            sparse_indices, prev_shape, new_shape)
        new_sparse_indices = [[0, 0],
                              [0, 1],
                              [1, 2],
                              [4, 2],
                              [8, 1]]
        new_shape = [9, 4]
    """

    def gen_ir(
        sparse_indices_ptr,
        prev_shape_ptr,
        new_shape_ptr,
        new_sparse_indices_ptr,
        out_new_shape_ptr,
    ):
        with IRBuilder() as ib:
            sparse_indices = sparse_indices_ptr
            prev_shape = prev_shape_ptr

            new_shape = new_shape_ptr
            out_new_shape = out_new_shape_ptr
            new_sparse_indices = new_sparse_indices_ptr

            prev_shape_size = prev_shape_ptr.shape[0]
            new_shape_size = new_shape_ptr.shape[0]

            multipliers_buf = T.bind_(
                T.alloc_tensor(
                    ty_args=[T.Tensor([prev_shape_size], new_shape_ptr.dtype, scope="local")]
                )
            )
            multipliers = multipliers_buf
            dividers_buf = T.bind_(
                T.alloc_tensor(
                    ty_args=[T.Tensor([new_shape_size], new_shape_ptr.dtype, scope="local")]
                )
            )
            dividers = dividers_buf
            flattened_indices_buf = T.bind_(
                T.alloc_tensor(
                    ty_args=[
                        T.Tensor([sparse_indices_ptr.shape[0]], new_shape_ptr.dtype, scope="local")
                    ]
                )
            )
            flattened_indices = flattened_indices_buf
            total_ele_buf = T.bind_(
                T.alloc_tensor(ty_args=[T.Tensor([1], new_shape_ptr.dtype, scope="local")])
            )
            total_ele = total_ele_buf
            division_total_ele_buf = T.bind_(
                T.alloc_tensor(ty_args=[T.Tensor([1], new_shape_ptr.dtype, scope="local")])
            )
            division_total_ele = division_total_ele_buf
            equal_shape_buf = T.bind_(
                T.alloc_tensor(ty_args=[T.Tensor([1], "bool", scope="local")])
            )
            equal_shape = equal_shape_buf

            T.tensor_store(
                total_ele,
                T.tensor_indices(total_ele, 0),
                prev_shape[T.tensor_indices(prev_shape, 0)],
            )

            # Cumulative Reverse Exclusive Multiply
            T.tensor_store(
                multipliers,
                T.tensor_indices(multipliers, prev_shape_size - 1),
                Cast(new_shape_ptr.dtype, 1),
            )
            with T.serial(0, prev_shape_size - 1) as i_:
                i = i_ + 1
                T.tensor_store(
                    multipliers,
                    T.tensor_indices(multipliers, prev_shape_size - 1 - i),
                    prev_shape[T.tensor_indices(prev_shape, prev_shape_size - i)]
                    * multipliers[T.tensor_indices(multipliers, prev_shape_size - i)],
                )
                T.tensor_store(
                    total_ele,
                    T.tensor_indices(total_ele, 0),
                    total_ele[T.tensor_indices(total_ele, 0)]
                    * (prev_shape[T.tensor_indices(prev_shape, prev_shape_size - i)]),
                )

            T.tensor_store(
                division_total_ele,
                T.tensor_indices(division_total_ele, 0),
                Cast(new_shape_ptr.dtype, 1),
            )
            with T.serial(0, new_shape_size) as i:
                with T.if_(new_shape[T.tensor_indices(new_shape, i)] != -1):
                    with T.then_():
                        T.tensor_store(
                            division_total_ele,
                            T.tensor_indices(division_total_ele, 0),
                            division_total_ele[T.tensor_indices(division_total_ele, 0)]
                            * (new_shape[T.tensor_indices(new_shape, i)]),
                        )

            # Compute true output shape (replace negative ones)
            with T.serial(0, new_shape_size) as i:
                with T.if_(new_shape[T.tensor_indices(new_shape, i)] == -1):
                    with T.then_():
                        T.tensor_store(
                            out_new_shape,
                            T.tensor_indices(out_new_shape, i),
                            Cast(
                                new_shape_ptr.dtype,
                                div(
                                    total_ele[T.tensor_indices(total_ele, 0)],
                                    division_total_ele[T.tensor_indices(division_total_ele, 0)],
                                ),
                            ),
                        )
                    with T.else_():
                        T.tensor_store(
                            out_new_shape,
                            T.tensor_indices(out_new_shape, i),
                            new_shape[T.tensor_indices(new_shape, i)],
                        )

            # Check if prev_shape and new_shape are equal
            T.tensor_store(equal_shape, T.tensor_indices(equal_shape, 0), True)
            with T.if_(prev_shape_size == new_shape_size):
                with T.then_():
                    with T.serial(0, prev_shape_size) as i:
                        with T.if_(
                            prev_shape[T.tensor_indices(prev_shape, i)]
                            != out_new_shape[T.tensor_indices(out_new_shape, i)]
                        ):
                            with T.then_():
                                T.tensor_store(equal_shape, T.tensor_indices(equal_shape, 0), False)
                with T.else_():
                    T.tensor_store(equal_shape, T.tensor_indices(equal_shape, 0), False)

            # Return same inputs if shapes are equal
            with T.if_(equal_shape[T.tensor_indices(equal_shape, 0)]):
                with T.then_():
                    with T.parallel(0, sparse_indices_ptr.shape[0]) as i:
                        with T.serial(0, sparse_indices_ptr.shape[1]) as j:
                            T.tensor_store(
                                new_sparse_indices,
                                (i, j),
                                sparse_indices[(i, j)],
                            )

                # Else compute new_sparse_indices
                with T.else_():
                    T.tensor_store(
                        dividers,
                        T.tensor_indices(dividers, new_shape_size - 1),
                        Cast(new_shape_ptr.dtype, 1),
                    )
                    with T.serial(0, new_shape_size - 1) as i_:
                        i = i_ + 1
                        T.tensor_store(
                            dividers,
                            T.tensor_indices(dividers, new_shape_size - 1 - i),
                            dividers[T.tensor_indices(dividers, new_shape_size - i)]
                            * out_new_shape[T.tensor_indices(out_new_shape, new_shape_size - i)],
                        )

                    with T.parallel(0, sparse_indices_ptr.shape[0]) as i:
                        T.tensor_store(
                            flattened_indices,
                            T.tensor_indices(flattened_indices, i),
                            Cast(new_shape_ptr.dtype, 0),
                        )
                        with T.serial(0, sparse_indices_ptr.shape[1]) as j:
                            T.tensor_store(
                                flattened_indices,
                                T.tensor_indices(flattened_indices, i),
                                flattened_indices[T.tensor_indices(flattened_indices, i)]
                                + (
                                    sparse_indices[(i, j)]
                                    * multipliers[T.tensor_indices(multipliers, j)]
                                ),
                            )

                    with T.parallel(0, new_sparse_indices_ptr.shape[0]) as i:
                        current_element_buf = T.bind_(
                            T.alloc_tensor(
                                ty_args=[T.Tensor([1], new_shape_ptr.dtype, scope="local")]
                            )
                        )
                        current_element = current_element_buf
                        T.tensor_store(
                            current_element,
                            T.tensor_indices(current_element, 0),
                            flattened_indices[T.tensor_indices(flattened_indices, i)],
                        )

                        with T.serial(0, new_sparse_indices_ptr.shape[1]) as j:
                            T.tensor_store(
                                new_sparse_indices,
                                (i, j),
                                Cast(
                                    sparse_indices_ptr.dtype,
                                    floordiv(
                                        current_element[T.tensor_indices(current_element, 0)],
                                        dividers[T.tensor_indices(dividers, j)],
                                    ),
                                ),
                            )
                            T.tensor_store(
                                current_element,
                                T.tensor_indices(current_element, 0),
                                floormod(
                                    current_element[T.tensor_indices(current_element, 0)],
                                    dividers[T.tensor_indices(dividers, j)],
                                ),
                            )

            return ib.get()

    new_sparse_indices_buf = Var(
        "new_sparse_indices_buf", TensorType(new_sparse_indices_shape, sparse_indices.dtype)
    )
    new_shape_buf = Var("new_shape_buf", TensorType(new_shape_shape, prev_shape.dtype))

    return extern(
        [new_sparse_indices_shape, new_shape_shape],
        [sparse_indices, prev_shape, new_shape],
        lambda ins, outs: gen_ir(ins[0], ins[1], ins[2], outs[0], outs[1]),
        out_tensors=[new_sparse_indices_buf, new_shape_buf],
        name="sparse_reshape_cpu",
        tag="sparse_reshape_cpu",
    )
