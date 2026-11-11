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
# ruff: noqa: E741
"""scatter_nd related operators"""

from tvm import te  # hide redefinition of min and max
from tvm_tirx import tirx
from tvm_tirx.script.ir_builder import IRBuilder
from tvm_tirx.tirx.script import ir_builder as T

from ..math import cast
from ..scatter import _verify_scatter_nd_inputs
from ..utils import ceil_div


def scatter_nd(data, indices, updates, mode, *, target=None):
    """GPU implementation of scatter_nd with explicit thread bindings."""
    _verify_scatter_nd_inputs(data, indices, updates)

    def gen_ir(data_ptr, indices_ptr, updates_ptr, out_ptr):
        # pylint: disable=invalid-name
        data = data_ptr
        indices = indices_ptr
        updates = updates_ptr
        out = out_ptr

        # We combine all the indices dimensions but the first one into a single
        # dimension so we can iterate it in single loop instead of an arbitrary
        # number of loops. We do the same thing for all the update dimensions.
        fused_indices_dimension = 1
        for i in indices_ptr.shape[1:]:
            fused_indices_dimension *= i

        fused_updates_dimension = 1
        for i in updates_ptr.shape[len(indices_ptr.shape) - 1 :]:
            fused_updates_dimension *= i

        fused_shape = 1
        for i in data_ptr.shape:
            fused_shape *= i

        max_threads = int(target.attrs["max_num_threads"])

        with IRBuilder() as ib:
            # Init
            nthread_bx_init = cast(ceil_div(fused_shape, max_threads), "int32")
            with T.frame_scope(
                [
                    T.launch_thread("blockIdx.x", nthread_bx_init),
                    T.launch_thread("threadIdx.x", max_threads),
                ]
            ) as (bx_init, tx_init):
                tid = bx_init * max_threads + tx_init
                with T.if_(tid < fused_shape):
                    with T.then_():
                        T.tensor_store(
                            out,
                            T.tensor_indices(out, tid),
                            data[T.tensor_indices(data, tid)],
                        )

            # Scatter
            nthread_bx_scat = cast(ceil_div(fused_updates_dimension, max_threads), "int32")
            with T.frame_scope(
                [
                    T.launch_thread("blockIdx.x", nthread_bx_scat),
                    T.launch_thread("threadIdx.x", max_threads),
                ]
            ) as (bx_scat, tx_scat):
                j = bx_scat * max_threads + tx_scat
                with T.if_(j < fused_updates_dimension):
                    with T.then_():
                        with T.serial(0, fused_indices_dimension) as i:
                            offset = fused_updates_dimension
                            index = j  # x_M, .. x_{N-1} part of the index into out.
                            # Build up the indices[0, y_0, ..], ..,
                            # indices[M-1, y_0, ..] part of the index into out.
                            for l in reversed(range(indices_ptr.shape[0].value)):
                                # indices[l, y_0, ... y_{k-1}]
                                index += (
                                    offset
                                    * indices[
                                        T.tensor_indices(indices, (i + l * fused_indices_dimension))
                                    ]
                                )
                                offset *= data_ptr.shape[l]
                            if mode == "update":
                                T.tensor_store(
                                    out,
                                    T.tensor_indices(out, index),
                                    updates[
                                        T.tensor_indices(updates, (i * fused_updates_dimension + j))
                                    ],
                                )
                            elif mode == "add":
                                T.tensor_store(
                                    out,
                                    T.tensor_indices(out, index),
                                    out[T.tensor_indices(out, index)]
                                    + (
                                        updates[
                                            T.tensor_indices(
                                                updates, (i * fused_updates_dimension + j)
                                            )
                                        ]
                                    ),
                                )
                            elif mode == "mul":
                                T.tensor_store(
                                    out,
                                    T.tensor_indices(out, index),
                                    out[T.tensor_indices(out, index)]
                                    * (
                                        updates[
                                            T.tensor_indices(
                                                updates, (i * fused_updates_dimension + j)
                                            )
                                        ]
                                    ),
                                )
                            elif mode == "min":
                                T.tensor_store(
                                    out,
                                    T.tensor_indices(out, index),
                                    tirx.min(
                                        out[T.tensor_indices(out, index)],
                                        updates[
                                            T.tensor_indices(
                                                updates, (i * fused_updates_dimension + j)
                                            )
                                        ],
                                    ),
                                )
                            elif mode == "max":
                                T.tensor_store(
                                    out,
                                    T.tensor_indices(out, index),
                                    tirx.max(
                                        out[T.tensor_indices(out, index)],
                                        updates[
                                            T.tensor_indices(
                                                updates, (i * fused_updates_dimension + j)
                                            )
                                        ],
                                    ),
                                )
                            else:
                                raise NotImplementedError(
                                    "scatter_nd mode not in [update, add, mul, min, max]:",
                                    mode,
                                )

            return ib.get()

    out_buf = tirx.Var("out_buf", tirx.TensorType(data.shape, data.dtype, layout=None))
    return te.extern(
        [data.shape],
        [data, indices, updates],
        lambda ins, outs: gen_ir(ins[0], ins[1], ins[2], outs[0]),
        dtype=data.dtype,
        out_tensors=[out_buf],
        name="scatter_nd.gpu",
        tag="scatter_nd.gpu",
    )
