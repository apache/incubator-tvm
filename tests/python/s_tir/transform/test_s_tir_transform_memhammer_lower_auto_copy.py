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
# ruff: noqa: E501, F401

import sys

import pytest
import tvm_ffi

import tvm
import tvm_tirx
from tvm import s_tir
from tvm.s_tir import match_shape_stride_pattern
from tvm.s_tir import script as Ts
from tvm_tirx.script import tirx as T


def _is_tensor_binding(node, *op_names):
    return (
        isinstance(node, tvm_tirx.ir.Bind)
        and isinstance(node.value, tvm_tirx.ir.Call)
        and isinstance(node.value.op, tvm_tirx.ir.Op)
        and node.value.op.name in op_names
    )


@tvm_tirx.script.ir_module
class Transpose:
    @Ts.function
    def main(A: T.Tensor([1024, 1024], "float32"), B: T.Tensor([1024, 1024], "float32")) -> None:
        with Ts.sblock("root"):
            Ts.sblock_attr({"warp_execution": True})
            for ty in T.parallel(8, annotations={"thread_binding": "threadIdx.y"}):
                with Ts.sblock():
                    A_shared_dyn = Ts.sblock_alloc_tensor(
                        [16, 128], dtype="float32", scope="shared.dyn"
                    )
                    with Ts.sblock("A_shared"):
                        Ts.sblock_attr({"auto_copy": True})
                        for ax0, ax1 in T.grid(128, 16):
                            A_shared_dyn[ax1, ax0] = A[ax0, ax1]
                    with Ts.sblock("B"):
                        Ts.sblock_attr({"auto_copy": True})
                        for ax1, ax0 in T.grid(16, 128):
                            B[ax1, ax0] = A_shared_dyn[ax1, ax0]


@tvm_tirx.script.ir_module
class GlobalToShared:
    @Ts.function
    def main(A: T.Tensor([1024, 1024], "float32"), B: T.Tensor([1024, 1024], "float32")) -> None:
        with Ts.sblock("root"):
            Ts.sblock_attr({"warp_execution": True})
            for bx in T.parallel(8, annotations={"thread_binding": "blockIdx.x"}):
                for by in T.parallel(8, annotations={"thread_binding": "blockIdx.y"}):
                    for ty in T.parallel(8, annotations={"thread_binding": "threadIdx.y"}):
                        with Ts.sblock():
                            A_shared_dyn = Ts.sblock_alloc_tensor(
                                [128, 128], dtype="float32", scope="shared.dyn"
                            )
                            with Ts.sblock("A_shared"):
                                Ts.sblock_attr({"auto_copy": True, "vector_bytes": 16})
                                for ax0, ax1 in T.grid(128, 128):
                                    A_shared_dyn[ax0, ax1] = A[bx * 128 + ax0, by * 128 + ax1]
                            with Ts.sblock("B"):
                                for ax0, ax1 in T.grid(128, 128):
                                    B[bx * 128 + ax0, by * 128 + ax1] = A_shared_dyn[ax0, ax1]


@tvm_tirx.script.ir_module
class SharedToGlobal:
    @Ts.function
    def main(A: T.Tensor([1024, 1024], "float32"), B: T.Tensor([1024, 1024], "float32")) -> None:
        with Ts.sblock("root"):
            Ts.sblock_attr({"warp_execution": True})
            for bx in T.parallel(8, annotations={"thread_binding": "blockIdx.x"}):
                for by in T.parallel(8, annotations={"thread_binding": "blockIdx.y"}):
                    for ty in T.parallel(8, annotations={"thread_binding": "threadIdx.y"}):
                        with Ts.sblock():
                            A_shared_dyn = Ts.sblock_alloc_tensor(
                                [128, 128], dtype="float32", scope="shared.dyn"
                            )
                            with Ts.sblock("A_shared"):
                                for ax0, ax1 in T.grid(128, 128):
                                    A_shared_dyn[ax1, ax0] = A[bx * 128 + ax0, by * 128 + ax1]
                            with Ts.sblock("B"):
                                Ts.sblock_attr({"auto_copy": True, "vector_bytes": 16})
                                for ax1, ax0 in T.grid(128, 128):
                                    B[bx * 128 + ax0, by * 128 + ax1] = A_shared_dyn[ax1, ax0]


@tvm_tirx.script.ir_module
class GlobalToSharedWithLocalStage:
    @Ts.function
    def main(A: T.Tensor([1024, 1024], "float32"), B: T.Tensor([1024, 1024], "float32")) -> None:
        with Ts.sblock("root"):
            Ts.sblock_attr({"warp_execution": True})
            for bx in T.parallel(8, annotations={"thread_binding": "blockIdx.x"}):
                for by in T.parallel(8, annotations={"thread_binding": "blockIdx.y"}):
                    for ty in T.parallel(8, annotations={"thread_binding": "threadIdx.y"}):
                        with Ts.sblock():
                            A_shared_dyn = Ts.sblock_alloc_tensor(
                                [128, 128], dtype="float32", scope="shared.dyn"
                            )
                            with Ts.sblock("A_shared"):
                                Ts.sblock_attr(
                                    {"auto_copy": True, "vector_bytes": 16, "local_stage": True}
                                )
                                for ax0, ax1 in T.grid(128, 128):
                                    A_shared_dyn[ax0, ax1] = A[bx * 128 + ax0, by * 128 + ax1]
                            with Ts.sblock("B"):
                                for ax0, ax1 in T.grid(128, 128):
                                    B[bx * 128 + ax0, by * 128 + ax1] = A_shared_dyn[ax0, ax1]


@tvm_tirx.script.ir_module
class SharedToWmma:
    @Ts.function
    def main() -> None:
        with Ts.sblock("root"):
            Ts.sblock_attr({"warp_execution": True})
            for bx in T.parallel(8, annotations={"thread_binding": "blockIdx.x"}):
                for by in T.parallel(8, annotations={"thread_binding": "blockIdx.y"}):
                    for ty in T.parallel(8, annotations={"thread_binding": "threadIdx.y"}):
                        with Ts.sblock():
                            A_shared_dyn = Ts.sblock_alloc_tensor(
                                [128, 128], dtype="float16", scope="shared.dyn"
                            )
                            A_wmma = Ts.sblock_alloc_tensor(
                                [128, 128], dtype="float16", scope="wmma.matrix_a"
                            )
                            with Ts.sblock("A_wmma"):
                                Ts.sblock_attr({"auto_copy": True})
                                for ax0, ax1 in T.grid(128, 128):
                                    A_wmma[ax0, ax1] = A_shared_dyn[ax0, ax1]


@tvm_tirx.script.ir_module
class WmmaToShared:
    @Ts.function
    def main() -> None:
        with Ts.sblock("root"):
            Ts.sblock_attr({"warp_execution": True})
            for bx in T.parallel(8, annotations={"thread_binding": "blockIdx.x"}):
                for by in T.parallel(8, annotations={"thread_binding": "blockIdx.y"}):
                    for ty in T.parallel(8, annotations={"thread_binding": "threadIdx.y"}):
                        with Ts.sblock():
                            C_accum = Ts.sblock_alloc_tensor(
                                [128, 128], dtype="float32", scope="wmma.accumulator"
                            )
                            C_shared = Ts.sblock_alloc_tensor(
                                [128, 128], dtype="float32", scope="shared.dyn"
                            )
                            with Ts.sblock("C_shared"):
                                Ts.sblock_attr({"auto_copy": True})
                                for ax0, ax1 in T.grid(128, 128):
                                    C_shared[ax0, ax1] = C_accum[ax0, ax1]


@tvm_tirx.script.ir_module
class WmmaToGlobal:
    @Ts.function
    def main(C: T.Tensor([1024, 1024], "float32")) -> None:
        with Ts.sblock("root"):
            Ts.sblock_attr({"warp_execution": True})
            for bx in T.parallel(8, annotations={"thread_binding": "blockIdx.x"}):
                for by in T.parallel(8, annotations={"thread_binding": "blockIdx.y"}):
                    for ty in T.parallel(8, annotations={"thread_binding": "threadIdx.y"}):
                        with Ts.sblock():
                            C_accum = Ts.sblock_alloc_tensor(
                                [128, 128], dtype="float32", scope="wmma.accumulator"
                            )
                            with Ts.sblock("C_global"):
                                Ts.sblock_attr({"auto_copy": True, "vector_bytes": 16})
                                for ax0, ax1 in T.grid(128, 128):
                                    C[bx * 128 + ax0, by * 128 + ax1] = C_accum[ax0, ax1]


@tvm_tirx.script.ir_module
class WmmaToGlobalWithFusion:
    @Ts.function
    def main(A: T.Tensor([1024], "float32"), C: T.Tensor([1024, 1024], "float32")) -> None:
        with Ts.sblock("root"):
            Ts.sblock_attr({"warp_execution": True})
            for bx in T.parallel(8, annotations={"thread_binding": "blockIdx.x"}):
                for by in T.parallel(8, annotations={"thread_binding": "blockIdx.y"}):
                    for ty in T.parallel(8, annotations={"thread_binding": "threadIdx.y"}):
                        with Ts.sblock():
                            C_accum = Ts.sblock_alloc_tensor(
                                [128, 128], dtype="float32", scope="wmma.accumulator"
                            )
                            with Ts.sblock("C_global"):
                                Ts.sblock_attr({"auto_copy": True, "vector_bytes": 16})
                                for ax0, ax1 in T.grid(128, 128):
                                    C[bx * 128 + ax0, by * 128 + ax1] = (
                                        C_accum[ax0, ax1] + A[bx * 128 + ax0]
                                    )


@tvm_tirx.script.ir_module
class MmaToGlobal:
    @Ts.function
    def main(C: T.Tensor([1024, 1024], "float32")) -> None:
        with Ts.sblock("root"):
            Ts.sblock_attr({"warp_execution": True})
            for bx in T.parallel(8, annotations={"thread_binding": "blockIdx.x"}):
                for by in T.parallel(8, annotations={"thread_binding": "blockIdx.y"}):
                    for ty in T.parallel(8, annotations={"thread_binding": "threadIdx.y"}):
                        with Ts.sblock():
                            C_accum = Ts.sblock_alloc_tensor(
                                [128, 128], dtype="float32", scope="mma.matrixC"
                            )
                            with Ts.sblock("C_global"):
                                Ts.sblock_attr({"auto_copy": True, "vector_bytes": 16})
                                for ax0, ax1 in T.grid(128, 128):
                                    C[bx * 128 + ax0, by * 128 + ax1] = C_accum[ax0, ax1]


@tvm_tirx.script.ir_module
class TransformedGlobalToShared:
    @Ts.function
    def main(A: T.Tensor([1024, 1024], "float32"), B: T.Tensor([1024, 1024], "float32")) -> None:
        with Ts.sblock("root"):
            Ts.sblock_attr({"warp_execution": True})
            for bx in T.parallel(8, annotations={"thread_binding": "blockIdx.x"}):
                for by in T.parallel(8, annotations={"thread_binding": "blockIdx.y"}):
                    for ty in T.parallel(8, annotations={"thread_binding": "threadIdx.y"}):
                        with Ts.sblock():
                            A_shared_dyn = Ts.sblock_alloc_tensor(
                                [128, 128],
                                dtype="float32",
                                layout=T.TileLayout(T.S[[128, 128] : [128, 1]]),
                                scope="shared.dyn",
                            )
                            with Ts.sblock("A_shared"):
                                Ts.sblock_attr({"auto_copy": True, "vector_bytes": 16})
                                for outer in T.serial(16):
                                    for ty_1 in T.parallel(
                                        8, annotations={"thread_binding": "threadIdx.y"}
                                    ):
                                        for tx in T.parallel(
                                            32, annotations={"thread_binding": "threadIdx.x"}
                                        ):
                                            for vec in T.vectorized(4):
                                                A_shared_dyn[
                                                    (((outer * 8 + ty_1) * 32 + tx) * 4 + vec)
                                                    // 128
                                                    % 128,
                                                    (((outer * 8 + ty_1) * 32 + tx) * 4 + vec)
                                                    % 128,
                                                ] = A[
                                                    bx * 128
                                                    + (((outer * 8 + ty_1) * 32 + tx) * 4 + vec)
                                                    // 128
                                                    % 128,
                                                    by * 128
                                                    + (((outer * 8 + ty_1) * 32 + tx) * 4 + vec)
                                                    % 128,
                                                ]
                            with Ts.sblock("B"):
                                for ax0, ax1 in T.grid(128, 128):
                                    B[bx * 128 + ax0, by * 128 + ax1] = A_shared_dyn[ax0, ax1]


@tvm_tirx.script.ir_module
class TransformedSharedToGlobal:
    @Ts.function
    def main(A: T.Tensor([1024, 1024], "float32"), B: T.Tensor([1024, 1024], "float32")) -> None:
        with Ts.sblock("root"):
            Ts.sblock_attr({"warp_execution": True})
            for bx in T.parallel(8, annotations={"thread_binding": "blockIdx.x"}):
                for by in T.parallel(8, annotations={"thread_binding": "blockIdx.y"}):
                    for ty in T.parallel(8, annotations={"thread_binding": "threadIdx.y"}):
                        with Ts.sblock():
                            A_shared_dyn = Ts.sblock_alloc_tensor(
                                [128, 128],
                                dtype="float32",
                                layout=T.TileLayout(T.S[[128, 128] : [129, 1]]),
                                scope="shared.dyn",
                            )
                            with Ts.sblock("A_shared"):
                                Ts.reads(A[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128])
                                Ts.writes(A_shared_dyn[0:128, 0:128])
                                for ax0, ax1 in T.grid(128, 128):
                                    A_shared_dyn[ax1, ax0] = A[bx * 128 + ax0, by * 128 + ax1]
                            with Ts.sblock("B"):
                                Ts.sblock_attr({"auto_copy": True, "vector_bytes": 16})
                                for outer in T.serial(16):
                                    for ty_1 in T.parallel(
                                        8, annotations={"thread_binding": "threadIdx.y"}
                                    ):
                                        for tx in T.parallel(
                                            32, annotations={"thread_binding": "threadIdx.x"}
                                        ):
                                            for vec in T.vectorized(4):
                                                B[
                                                    bx * 128
                                                    + (((outer * 8 + ty_1) * 32 + tx) * 4 + vec)
                                                    // 128
                                                    % 128,
                                                    by * 128
                                                    + (((outer * 8 + ty_1) * 32 + tx) * 4 + vec)
                                                    % 128,
                                                ] = A_shared_dyn[
                                                    (((outer * 8 + ty_1) * 32 + tx) * 4 + vec)
                                                    % 128,
                                                    (((outer * 8 + ty_1) * 32 + tx) * 4 + vec)
                                                    // 128
                                                    % 128,
                                                ]


@tvm_tirx.script.ir_module
class TransformedGlobalToSharedWithLocalStage:
    @Ts.function
    def main(A: T.Tensor((1024, 1024), "float32"), B: T.Tensor((1024, 1024), "float32")):
        with Ts.sblock("root"):
            Ts.sblock_attr({"warp_execution": True})
            for bx in T.parallel(8, annotations={"thread_binding": "blockIdx.x"}):
                for by in T.parallel(8, annotations={"thread_binding": "blockIdx.y"}):
                    for ty in T.parallel(8, annotations={"thread_binding": "threadIdx.y"}):
                        with Ts.sblock(""):
                            Ts.reads(A[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128])
                            Ts.writes(B[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128])
                            A_shared_dyn = Ts.sblock_alloc_tensor(
                                (128, 128),
                                layout=T.TileLayout(T.S[(128, 128) : (128, 1)]),
                                scope="shared.dyn",
                            )
                            with Ts.sblock("A_shared"):
                                Ts.reads(A[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128])
                                Ts.writes(A_shared_dyn[0:128, 0:128])
                                Ts.sblock_attr(
                                    {"auto_copy": True, "local_stage": True, "vector_bytes": 16}
                                )
                                A_shared_dyn_local = Ts.sblock_alloc_tensor((16, 4), scope="local")
                                for ax0_ax1_fused_1 in T.parallel(
                                    8, annotations={"thread_binding": "threadIdx.y"}
                                ):
                                    for ax0_ax1_fused_2 in T.parallel(
                                        32, annotations={"thread_binding": "threadIdx.x"}
                                    ):
                                        for ax0_ax1_fused_0_cache in range(16):
                                            for ax0_ax1_fused_3_cache in T.vectorized(4):
                                                A_shared_dyn_local[
                                                    ax0_ax1_fused_0_cache
                                                    * 8
                                                    * 32
                                                    * 4
                                                    // 128
                                                    % 128
                                                    // 8,
                                                    ax0_ax1_fused_3_cache % 128,
                                                ] = A[
                                                    bx * 128
                                                    + (
                                                        (
                                                            (
                                                                ax0_ax1_fused_0_cache * 8
                                                                + ax0_ax1_fused_1
                                                            )
                                                            * 32
                                                            + ax0_ax1_fused_2
                                                        )
                                                        * 4
                                                        + ax0_ax1_fused_3_cache
                                                    )
                                                    // 128
                                                    % 128,
                                                    by * 128
                                                    + (
                                                        (
                                                            (
                                                                ax0_ax1_fused_0_cache * 8
                                                                + ax0_ax1_fused_1
                                                            )
                                                            * 32
                                                            + ax0_ax1_fused_2
                                                        )
                                                        * 4
                                                        + ax0_ax1_fused_3_cache
                                                    )
                                                    % 128,
                                                ]
                                        for ax0_ax1_fused_0 in range(16):
                                            for ax0_ax1_fused_3 in T.vectorized(4):
                                                A_shared_dyn[
                                                    (
                                                        (
                                                            (ax0_ax1_fused_0 * 8 + ax0_ax1_fused_1)
                                                            * 32
                                                            + ax0_ax1_fused_2
                                                        )
                                                        * 4
                                                        + ax0_ax1_fused_3
                                                    )
                                                    // 128
                                                    % 128,
                                                    (
                                                        (
                                                            (ax0_ax1_fused_0 * 8 + ax0_ax1_fused_1)
                                                            * 32
                                                            + ax0_ax1_fused_2
                                                        )
                                                        * 4
                                                        + ax0_ax1_fused_3
                                                    )
                                                    % 128,
                                                ] = A_shared_dyn_local[
                                                    ax0_ax1_fused_0 * 8 * 32 * 4 // 128 % 128 // 8,
                                                    ax0_ax1_fused_3 % 128,
                                                ]
                            with Ts.sblock("B"):
                                Ts.reads(A_shared_dyn[0:128, 0:128])
                                Ts.writes(B[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128])
                                for ax0 in range(128):
                                    for ax1 in range(128):
                                        B[bx * 128 + ax0, by * 128 + ax1] = A_shared_dyn[ax0, ax1]


s0 = T.dynamic("s0", "int32")
s1 = T.dynamic("s1", "int32")


transformedsharedtowmma_src_offset = T.dynamic("transformedsharedtowmma_src_offset", "int32")
transformedsharedtowmma_tgt_offset = T.dynamic("transformedsharedtowmma_tgt_offset", "int32")


@tvm_tirx.script.ir_module
class TransformedSharedToWmma:
    @Ts.function
    def main() -> None:
        # body
        with Ts.sblock("root"):
            Ts.sblock_attr({"warp_execution": True})
            for bx in T.parallel(8, annotations={"thread_binding": "blockIdx.x"}):
                for by in T.parallel(8, annotations={"thread_binding": "blockIdx.y"}):
                    for ty in T.parallel(8, annotations={"thread_binding": "threadIdx.y"}):
                        with Ts.sblock():
                            A_shared_dyn = Ts.sblock_alloc_tensor(
                                [128, 128],
                                dtype="float16",
                                layout=T.TileLayout(T.S[[128, 128] : [136, 1]]),
                                scope="shared.dyn",
                            )
                            A_wmma = Ts.sblock_alloc_tensor(
                                [128, 128], dtype="float16", scope="wmma.matrix_a"
                            )
                            with Ts.sblock("C_shared"):
                                Ts.reads(A_shared_dyn[0:128, 0:128])
                                Ts.writes(A_wmma[0:128, 0:128])
                                Ts.sblock_attr({"auto_copy": True})
                                for ax00, ax10 in T.grid(8, 8):
                                    with Ts.sblock("wmma_load"):
                                        Ts.reads(
                                            A_shared_dyn[
                                                ax00 * 16 : ax00 * 16 + 16,
                                                ax10 * 16 : ax10 * 16 + 16,
                                            ]
                                        )
                                        Ts.writes(
                                            A_wmma[
                                                ax00 * 16 : ax00 * 16 + 16,
                                                ax10 * 16 : ax10 * 16 + 16,
                                            ]
                                        )
                                        src = Ts.match_tensor(
                                            A_shared_dyn[
                                                ax00 * 16 : ax00 * 16 + 16,
                                                ax10 * 16 : ax10 * 16 + 16,
                                            ],
                                            [16, 16],
                                            dtype="float16",
                                            layout=T.TileLayout(
                                                T.S[[16, 16] : [s1, s0]]
                                                + ((16) * transformedsharedtowmma_src_offset)
                                            ),
                                            scope="shared.dyn",
                                        )
                                        tgt = Ts.match_tensor(
                                            A_wmma[
                                                ax00 * 16 : ax00 * 16 + 16,
                                                ax10 * 16 : ax10 * 16 + 16,
                                            ],
                                            [16, 16],
                                            dtype="float16",
                                            scope="wmma.matrix_a",
                                            layout=T.TileLayout(
                                                T.S[[16, 16]]
                                                + ((16) * transformedsharedtowmma_tgt_offset)
                                            ),
                                        )
                                        tgt_pattern = T.meta_var(match_shape_stride_pattern(tgt.ty))
                                        T.evaluate(
                                            T.gpu_load_matrix_sync(
                                                tgt.data,
                                                16,
                                                16,
                                                16,
                                                tgt_pattern.elem_offset // 256
                                                + tgt_pattern.elem_offset % 256 // 16,
                                                src.ptr_to([0] * len(src.shape)),
                                                s1,
                                                "row_major",
                                            )
                                        )


s0 = T.dynamic("s0", "int32")
s1 = T.dynamic("s1", "int32")


transformedwmmatoshared_src_offset = T.dynamic("transformedwmmatoshared_src_offset", "int32")
transformedwmmatoshared_tgt_offset = T.dynamic("transformedwmmatoshared_tgt_offset", "int32")


@tvm_tirx.script.ir_module
class TransformedWmmaToShared:
    @Ts.function
    def main() -> None:
        # body
        with Ts.sblock("root"):
            Ts.sblock_attr({"warp_execution": True})
            for bx in T.parallel(8, annotations={"thread_binding": "blockIdx.x"}):
                for by in T.parallel(8, annotations={"thread_binding": "blockIdx.y"}):
                    for ty in T.parallel(8, annotations={"thread_binding": "threadIdx.y"}):
                        with Ts.sblock():
                            C_accum = Ts.sblock_alloc_tensor(
                                [128, 128], dtype="float32", scope="wmma.accumulator"
                            )
                            C_shared = Ts.sblock_alloc_tensor(
                                [128, 128],
                                dtype="float32",
                                layout=T.TileLayout(T.S[[128, 128] : [136, 1]]),
                                scope="shared.dyn",
                            )
                            with Ts.sblock("A_wmma"):
                                Ts.reads(C_accum[0:128, 0:128])
                                Ts.writes(C_shared[0:128, 0:128])
                                Ts.sblock_attr({"auto_copy": True})
                                for ax00, ax10 in T.grid(8, 8):
                                    with Ts.sblock("wmma_store"):
                                        Ts.reads(
                                            C_accum[
                                                ax00 * 16 : ax00 * 16 + 16,
                                                ax10 * 16 : ax10 * 16 + 16,
                                            ]
                                        )
                                        Ts.writes(
                                            C_shared[
                                                ax00 * 16 : ax00 * 16 + 16,
                                                ax10 * 16 : ax10 * 16 + 16,
                                            ]
                                        )
                                        src = Ts.match_tensor(
                                            C_accum[
                                                ax00 * 16 : ax00 * 16 + 16,
                                                ax10 * 16 : ax10 * 16 + 16,
                                            ],
                                            [16, 16],
                                            dtype="float32",
                                            scope="wmma.accumulator",
                                            layout=T.TileLayout(
                                                T.S[[16, 16]]
                                                + ((16) * transformedwmmatoshared_src_offset)
                                            ),
                                        )
                                        tgt = Ts.match_tensor(
                                            C_shared[
                                                ax00 * 16 : ax00 * 16 + 16,
                                                ax10 * 16 : ax10 * 16 + 16,
                                            ],
                                            [16, 16],
                                            dtype="float32",
                                            layout=T.TileLayout(
                                                T.S[[16, 16] : [s1, s0]]
                                                + ((16) * transformedwmmatoshared_tgt_offset)
                                            ),
                                            scope="shared.dyn",
                                        )
                                        src_pattern = T.meta_var(match_shape_stride_pattern(src.ty))
                                        T.evaluate(
                                            T.gpu_store_matrix_sync(
                                                src.data,
                                                16,
                                                16,
                                                16,
                                                src_pattern.elem_offset // 256
                                                + src_pattern.elem_offset % 256 // 16,
                                                tgt.ptr_to([0] * len(tgt.shape)),
                                                s1,
                                                "row_major",
                                            )
                                        )


s1 = T.dynamic("s1", "int32")
s0 = T.dynamic("s0", "int32")


transformedwmmatoglobal_src_offset = T.dynamic("transformedwmmatoglobal_src_offset", "int32")
transformedwmmatoglobal_tgt_offset = T.dynamic("transformedwmmatoglobal_tgt_offset", "int32")


@tvm_tirx.script.ir_module
class TransformedWmmaToGlobal:
    @Ts.function
    def main(C: T.Tensor((1024, 1024), "float32")):
        with Ts.sblock("root"):
            Ts.sblock_attr({"warp_execution": True})
            for bx in T.parallel(8, annotations={"thread_binding": "blockIdx.x"}):
                for by in T.parallel(8, annotations={"thread_binding": "blockIdx.y"}):
                    for ty in T.parallel(8, annotations={"thread_binding": "threadIdx.y"}):
                        with Ts.sblock(""):
                            Ts.reads()
                            Ts.writes(C[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128])
                            C_accum = Ts.sblock_alloc_tensor((128, 128), scope="wmma.accumulator")
                            with Ts.sblock("C_global"):
                                Ts.reads(C_accum[0:128, 0:128])
                                Ts.writes(C[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128])
                                Ts.sblock_attr({"auto_copy": True, "vector_bytes": 16})
                                C_accum_shared_dyn = Ts.sblock_alloc_tensor(
                                    (8, 8, 16, 16),
                                    layout=T.TileLayout(T.S[(8, 8, 16, 16) : (2048, 256, 16, 1)]),
                                    scope="shared.dyn",
                                )
                                for ax0_0 in range(8):
                                    for ax1_0 in range(8):
                                        with Ts.sblock("wmma_store"):
                                            Ts.reads(
                                                C_accum[
                                                    ax0_0 * 16 : ax0_0 * 16 + 16,
                                                    ax1_0 * 16 : ax1_0 * 16 + 16,
                                                ]
                                            )
                                            Ts.writes(C_accum_shared_dyn[ty, ax1_0, 0:16, 0:16])
                                            src = Ts.match_tensor(
                                                C_accum[
                                                    ax0_0 * 16 : ax0_0 * 16 + 16,
                                                    ax1_0 * 16 : ax1_0 * 16 + 16,
                                                ],
                                                (16, 16),
                                                scope="wmma.accumulator",
                                                layout=T.TileLayout(
                                                    T.S[(16, 16)]
                                                    + ((16) * transformedwmmatoglobal_src_offset)
                                                ),
                                            )
                                            tgt = Ts.match_tensor(
                                                C_accum_shared_dyn[ty, ax1_0, 0:16, 0:16],
                                                (16, 16),
                                                layout=T.TileLayout(
                                                    T.S[(16, 16) : (s1, s0)]
                                                    + ((16) * transformedwmmatoglobal_tgt_offset)
                                                ),
                                                scope="shared.dyn",
                                            )
                                            src_pattern = T.meta_var(
                                                match_shape_stride_pattern(src.ty)
                                            )
                                            T.gpu_store_matrix_sync(
                                                src.data,
                                                16,
                                                16,
                                                16,
                                                src_pattern.elem_offset // 256
                                                + src_pattern.elem_offset % 256 // 16,
                                                tgt.ptr_to([0] * len(tgt.shape)),
                                                s1,
                                                "row_major",
                                            )
                                    for (
                                        ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0
                                    ) in range(16):
                                        for (
                                            ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1
                                        ) in T.parallel(
                                            8, annotations={"thread_binding": "threadIdx.y"}
                                        ):
                                            for (
                                                ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2
                                            ) in T.parallel(
                                                32, annotations={"thread_binding": "threadIdx.x"}
                                            ):
                                                for ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_3 in T.vectorized(
                                                    4
                                                ):
                                                    C[
                                                        bx * 128
                                                        + (
                                                            ax0_0 * 16
                                                            + (
                                                                (
                                                                    (
                                                                        ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0
                                                                        * 8
                                                                        + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1
                                                                    )
                                                                    * 32
                                                                    + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2
                                                                )
                                                                * 4
                                                                + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_3
                                                            )
                                                            // 16
                                                            % 16
                                                        ),
                                                        by * 128
                                                        + (
                                                            (
                                                                (
                                                                    (
                                                                        ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0
                                                                        * 8
                                                                        + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1
                                                                    )
                                                                    * 32
                                                                    + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2
                                                                )
                                                                * 4
                                                                + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_3
                                                            )
                                                            // 16
                                                            // 16
                                                            % 8
                                                            * 16
                                                            + (
                                                                (
                                                                    (
                                                                        ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0
                                                                        * 8
                                                                        + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1
                                                                    )
                                                                    * 32
                                                                    + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2
                                                                )
                                                                * 4
                                                                + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_3
                                                            )
                                                            % 16
                                                        ),
                                                    ] = C_accum_shared_dyn[
                                                        (
                                                            (
                                                                (
                                                                    ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0
                                                                    * 8
                                                                    + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1
                                                                )
                                                                * 32
                                                                + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2
                                                            )
                                                            * 4
                                                            + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_3
                                                        )
                                                        // 16
                                                        // 16
                                                        // 8
                                                        % 8,
                                                        (
                                                            (
                                                                (
                                                                    ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0
                                                                    * 8
                                                                    + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1
                                                                )
                                                                * 32
                                                                + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2
                                                            )
                                                            * 4
                                                            + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_3
                                                        )
                                                        // 16
                                                        // 16
                                                        % 8,
                                                        (
                                                            (
                                                                (
                                                                    ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0
                                                                    * 8
                                                                    + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1
                                                                )
                                                                * 32
                                                                + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2
                                                            )
                                                            * 4
                                                            + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_3
                                                        )
                                                        // 16
                                                        % 16,
                                                        (
                                                            (
                                                                (
                                                                    ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0
                                                                    * 8
                                                                    + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1
                                                                )
                                                                * 32
                                                                + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2
                                                            )
                                                            * 4
                                                            + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_3
                                                        )
                                                        % 16,
                                                    ]


s0_0 = T.dynamic("s0_0", "int32")
s1_0 = T.dynamic("s1_0", "int32")
s1_1 = T.dynamic("s1_1", "int32")
s0_1 = T.dynamic("s0_1", "int32")


transformedwmmatoglobalwithfusion_src_offset = T.dynamic(
    "transformedwmmatoglobalwithfusion_src_offset", "int32"
)
transformedwmmatoglobalwithfusion_tgt_offset = T.dynamic(
    "transformedwmmatoglobalwithfusion_tgt_offset", "int32"
)


@tvm_tirx.script.ir_module
class TransformedWmmaToGlobalWithFusion:
    @Ts.function
    def main(A: T.Tensor((1024,), "float32"), C: T.Tensor((1024, 1024), "float32")) -> None:
        # body
        with Ts.sblock("root"):
            Ts.sblock_attr({"warp_execution": True})
            for bx in T.parallel(8, annotations={"thread_binding": "blockIdx.x"}):
                for by in T.parallel(8, annotations={"thread_binding": "blockIdx.y"}):
                    for ty in T.parallel(8, annotations={"thread_binding": "threadIdx.y"}):
                        with Ts.sblock():
                            Ts.reads(A[bx * 128 : bx * 128 + 128])
                            Ts.writes(C[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128])
                            C_accum = Ts.sblock_alloc_tensor(
                                [128, 128], dtype="float32", scope="wmma.accumulator"
                            )
                            with Ts.sblock("C_global"):
                                Ts.reads(C_accum[0:128, 0:128], A[bx * 128 : bx * 128 + 128])
                                Ts.writes(C[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128])
                                Ts.sblock_attr({"auto_copy": True, "vector_bytes": 16})
                                C_accum_shared_dyn = Ts.sblock_alloc_tensor(
                                    (8, 8, 16, 16),
                                    layout=T.TileLayout(T.S[(8, 8, 16, 16) : (2048, 256, 16, 1)]),
                                    scope="shared.dyn",
                                )
                                for ax0_0 in range(8):
                                    for ax1_0 in range(8):
                                        with Ts.sblock("wmma_store"):
                                            Ts.reads(
                                                C_accum[
                                                    ax0_0 * 16 : ax0_0 * 16 + 16,
                                                    ax1_0 * 16 : ax1_0 * 16 + 16,
                                                ]
                                            )
                                            Ts.writes(C_accum_shared_dyn[ty, ax1_0, 0:16, 0:16])
                                            src = Ts.match_tensor(
                                                C_accum[
                                                    ax0_0 * 16 : ax0_0 * 16 + 16,
                                                    ax1_0 * 16 : ax1_0 * 16 + 16,
                                                ],
                                                (16, 16),
                                                scope="wmma.accumulator",
                                                layout=T.TileLayout(
                                                    T.S[(16, 16)]
                                                    + (
                                                        (16)
                                                        * transformedwmmatoglobalwithfusion_src_offset
                                                    )
                                                ),
                                            )
                                            tgt = Ts.match_tensor(
                                                C_accum_shared_dyn[ty, ax1_0, 0:16, 0:16],
                                                (16, 16),
                                                layout=T.TileLayout(
                                                    T.S[(16, 16) : (s1_1, s0_1)]
                                                    + (
                                                        (16)
                                                        * transformedwmmatoglobalwithfusion_tgt_offset
                                                    )
                                                ),
                                                scope="shared.dyn",
                                            )
                                            src_pattern = T.meta_var(
                                                match_shape_stride_pattern(src.ty)
                                            )
                                            T.gpu_store_matrix_sync(
                                                src.data,
                                                16,
                                                16,
                                                16,
                                                src_pattern.elem_offset // 256
                                                + src_pattern.elem_offset % 256 // 16,
                                                tgt.ptr_to([0] * len(tgt.shape)),
                                                s1_1,
                                                "row_major",
                                            )
                                    for (
                                        ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0
                                    ) in range(16):
                                        for (
                                            ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1
                                        ) in T.parallel(
                                            8, annotations={"thread_binding": "threadIdx.y"}
                                        ):
                                            for (
                                                ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2
                                            ) in T.parallel(
                                                32, annotations={"thread_binding": "threadIdx.x"}
                                            ):
                                                for ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_3 in T.vectorized(
                                                    4
                                                ):
                                                    C[
                                                        bx * 128
                                                        + (
                                                            ax0_0 * 16
                                                            + (
                                                                (
                                                                    (
                                                                        ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0
                                                                        * 8
                                                                        + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1
                                                                    )
                                                                    * 32
                                                                    + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2
                                                                )
                                                                * 4
                                                                + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_3
                                                            )
                                                            // 16
                                                            % 16
                                                        ),
                                                        by * 128
                                                        + (
                                                            (
                                                                (
                                                                    (
                                                                        ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0
                                                                        * 8
                                                                        + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1
                                                                    )
                                                                    * 32
                                                                    + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2
                                                                )
                                                                * 4
                                                                + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_3
                                                            )
                                                            // 16
                                                            // 16
                                                            % 8
                                                            * 16
                                                            + (
                                                                (
                                                                    (
                                                                        ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0
                                                                        * 8
                                                                        + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1
                                                                    )
                                                                    * 32
                                                                    + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2
                                                                )
                                                                * 4
                                                                + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_3
                                                            )
                                                            % 16
                                                        ),
                                                    ] = (
                                                        C_accum_shared_dyn[
                                                            (
                                                                (
                                                                    (
                                                                        ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0
                                                                        * 8
                                                                        + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1
                                                                    )
                                                                    * 32
                                                                    + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2
                                                                )
                                                                * 4
                                                                + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_3
                                                            )
                                                            // 16
                                                            // 16
                                                            // 8
                                                            % 8,
                                                            (
                                                                (
                                                                    (
                                                                        ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0
                                                                        * 8
                                                                        + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1
                                                                    )
                                                                    * 32
                                                                    + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2
                                                                )
                                                                * 4
                                                                + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_3
                                                            )
                                                            // 16
                                                            // 16
                                                            % 8,
                                                            (
                                                                (
                                                                    (
                                                                        ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0
                                                                        * 8
                                                                        + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1
                                                                    )
                                                                    * 32
                                                                    + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2
                                                                )
                                                                * 4
                                                                + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_3
                                                            )
                                                            // 16
                                                            % 16,
                                                            (
                                                                (
                                                                    (
                                                                        ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0
                                                                        * 8
                                                                        + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1
                                                                    )
                                                                    * 32
                                                                    + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2
                                                                )
                                                                * 4
                                                                + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_3
                                                            )
                                                            % 16,
                                                        ]
                                                        + A[
                                                            bx * 128
                                                            + (
                                                                ax0_0 * 16
                                                                + (
                                                                    (
                                                                        (
                                                                            ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0
                                                                            * 8
                                                                            + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1
                                                                        )
                                                                        * 32
                                                                        + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2
                                                                    )
                                                                    * 4
                                                                    + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_3
                                                                )
                                                                // 16
                                                                % 16
                                                            )
                                                        ]
                                                    )


s1 = T.dynamic("s1", "int32")
s0 = T.dynamic("s0", "int32")


transformedmmatoglobal_src_offset = T.dynamic("transformedmmatoglobal_src_offset", "int32")
transformedmmatoglobal_tgt_offset = T.dynamic("transformedmmatoglobal_tgt_offset", "int32")


@tvm_tirx.script.ir_module
class TransformedMmaToGlobal:
    @Ts.function
    def main(C: T.Tensor((1024, 1024), "float32")):
        with Ts.sblock("root"):
            Ts.sblock_attr({"warp_execution": True})
            for bx in T.parallel(8, annotations={"thread_binding": "blockIdx.x"}):
                for by in T.parallel(8, annotations={"thread_binding": "blockIdx.y"}):
                    for ty in T.parallel(8, annotations={"thread_binding": "threadIdx.y"}):
                        with Ts.sblock(""):
                            Ts.reads()
                            Ts.writes(C[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128])
                            C_accum = Ts.sblock_alloc_tensor((128, 128), scope="mma.matrixC")
                            with Ts.sblock("C_global"):
                                Ts.reads(C_accum[0:128, 0:128])
                                Ts.writes(C[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128])
                                Ts.sblock_attr({"auto_copy": True, "vector_bytes": 16})
                                C_accum_shared_dyn = Ts.sblock_alloc_tensor(
                                    (8, 16, 8, 8),
                                    layout=T.TileLayout(T.S[(8, 16, 8, 8) : (1152, 72, 8, 1)]),
                                    scope="shared.dyn",
                                )
                                for ax0_0 in range(16):
                                    for ax1_0 in range(16):
                                        with Ts.sblock("mma_store"):
                                            Ts.reads(
                                                C_accum[
                                                    ax0_0 * 8 : ax0_0 * 8 + 8,
                                                    ax1_0 * 8 : ax1_0 * 8 + 8,
                                                ]
                                            )
                                            Ts.writes(C_accum_shared_dyn[ty, ax1_0, 0:8, 0:8])
                                            src = Ts.match_tensor(
                                                C_accum[
                                                    ax0_0 * 8 : ax0_0 * 8 + 8,
                                                    ax1_0 * 8 : ax1_0 * 8 + 8,
                                                ],
                                                (8, 8),
                                                scope="mma.matrixC",
                                                layout=T.TileLayout(
                                                    T.S[(8, 8)]
                                                    + ((8) * transformedmmatoglobal_src_offset)
                                                ),
                                            )
                                            tgt = Ts.match_tensor(
                                                C_accum_shared_dyn[ty, ax1_0, 0:8, 0:8],
                                                (8, 8),
                                                layout=T.TileLayout(
                                                    T.S[(8, 8) : (s1, s0)]
                                                    + ((8) * transformedmmatoglobal_tgt_offset)
                                                ),
                                                scope="shared.dyn",
                                            )
                                            tx = T.launch_thread("threadIdx.x", 32)
                                            for vec in T.vectorized(2):
                                                tgt[tx // 4, tx % 4 * 2 + vec] = src[
                                                    tx // 4, tx % 4 * 2 + vec
                                                ]
                                    for ax1_1 in range(8):
                                        for ty_0 in T.parallel(
                                            8, annotations={"thread_binding": "threadIdx.y"}
                                        ):
                                            for tx_0 in T.parallel(
                                                32, annotations={"thread_binding": "threadIdx.x"}
                                            ):
                                                for v in T.vectorized(4):
                                                    C[
                                                        bx * 128
                                                        + (
                                                            ax0_0 * 8
                                                            + (
                                                                ((ax1_1 * 8 + ty_0) * 32 + tx_0) * 4
                                                                + v
                                                            )
                                                            // 8
                                                            % 8
                                                        ),
                                                        by * 128
                                                        + (
                                                            (
                                                                ((ax1_1 * 8 + ty_0) * 32 + tx_0) * 4
                                                                + v
                                                            )
                                                            // 8
                                                            // 8
                                                            % 16
                                                            * 8
                                                            + (
                                                                ((ax1_1 * 8 + ty_0) * 32 + tx_0) * 4
                                                                + v
                                                            )
                                                            % 8
                                                        ),
                                                    ] = C_accum_shared_dyn[
                                                        (((ax1_1 * 8 + ty_0) * 32 + tx_0) * 4 + v)
                                                        // 8
                                                        // 8
                                                        // 16
                                                        % 8,
                                                        (((ax1_1 * 8 + ty_0) * 32 + tx_0) * 4 + v)
                                                        // 8
                                                        // 8
                                                        % 16,
                                                        (((ax1_1 * 8 + ty_0) * 32 + tx_0) * 4 + v)
                                                        // 8
                                                        % 8,
                                                        (((ax1_1 * 8 + ty_0) * 32 + tx_0) * 4 + v)
                                                        % 8,
                                                    ]


def _check(original, transformed):
    mod = tvm.s_tir.transform.LowerAutoCopy()(original)
    tvm_tirx.ir.assert_structural_equal(mod, transformed, True)


def test_coalesce_vectorize():
    _check(GlobalToShared, TransformedGlobalToShared)


def test_inverse():
    _check(SharedToGlobal, TransformedSharedToGlobal)


def test_local_stage():
    _check(GlobalToSharedWithLocalStage, TransformedGlobalToSharedWithLocalStage)


def test_rewrite_shared_to_wmma():
    _check(SharedToWmma, TransformedSharedToWmma)


def test_rewrite_wmma_to_shared():
    _check(WmmaToShared, TransformedWmmaToShared)


def test_rewrite_wmma_to_global():
    _check(WmmaToGlobal, TransformedWmmaToGlobal)


def verify_single_allocation(stmt, alloc_size=None):
    num_alloc = [0]
    alloc_extents = []

    def verify(n):
        if _is_tensor_binding(
            n, "tirx.alloc_tensor"
        ) and n.var.scope() == tvm_tirx.runtime.MemoryScope("shared.dyn"):
            num_alloc[0] += 1
            alloc_extents.append(n.var.shape)

    tvm_ffi.structural_walk(stmt, verify)
    assert num_alloc[0] == 1

    if alloc_size:

        def prod(arr):
            ret = 1
            for element in arr:
                ret *= element
            return ret

        assert prod(alloc_extents[0]) == alloc_size


def test_auto_padding():
    mod = tvm.s_tir.transform.LowerAutoCopy()(Transpose)
    mod = tvm.s_tir.transform.LowerOpaqueBlock()(mod)
    mod = tvm_tirx.tirx.transform.FlattenTensor()(mod)
    verify_single_allocation(mod["main"].body, 16 * 130)


def test_rewrite_wmma_to_global_fusion():
    _check(WmmaToGlobalWithFusion, TransformedWmmaToGlobalWithFusion)


def test_rewrite_mma_to_global():
    _check(MmaToGlobal, TransformedMmaToGlobal)


if __name__ == "__main__":
    test_coalesce_vectorize()
    test_inverse()
    test_local_stage()
    test_rewrite_shared_to_wmma()
    test_rewrite_wmma_to_shared()
    test_rewrite_wmma_to_global()
    test_auto_padding()
    test_rewrite_wmma_to_global_fusion()
    test_rewrite_mma_to_global()
