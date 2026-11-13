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
# pylint: disable=invalid-name,missing-function-docstring
"""Intrinsics for AMDGPU tensorization.

Descriptors use a logical (lane, value) layout to match a whole wavefront's
matrix operation.  Implementations bind only (1, values_per_lane) in local
storage: every lane owns its own fragment, indexed at lane zero.  The hardware
lane id selects the shared/global matrix coordinates for loads and stores;
MFMA communicates the register operands across lanes.  Binding a singleton
lane dimension also lets CompactTensorAllocation remove the logical 64-lane
extent without a warp-memory lowering pass.
"""

import tvm_ffi

from tvm.s_tir import script as Ts
from tvm_tirx.ir import IntImm
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx.expr import Cast

from .. import TensorIntrin
from .dot_product_common import get_dp4a_intrin

lift = tvm_ffi.convert


sdot4_a_offset = T.dynamic("sdot4_a_offset", "int32")
sdot4_b_offset = T.dynamic("sdot4_b_offset", "int32")
sdot4_c_offset = T.dynamic("sdot4_c_offset", "int32")


@Ts.function
def sdot4(
    A: T.Tensor(
        (4,), "int8", layout=T.TileLayout(T.S[(4,)] + (sdot4_a_offset)), align=4, scope="shared"
    ),
    B: T.Tensor(
        (4,), "int8", layout=T.TileLayout(T.S[(4,)] + (sdot4_b_offset)), align=4, scope="shared"
    ),
    C: T.Tensor(
        (1,), "int32", layout=T.TileLayout(T.S[(1,)] + (sdot4_c_offset)), align=4, scope="local"
    ),
) -> None:
    with Ts.sblock("root"):
        Ts.reads(C[0], A[0:4], B[0:4])
        Ts.writes(C[0])

        C[0] += T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.amdgcn.sdot4"),
            T.reinterpret(A[T.Ramp(0, 1, 4)], ty="int32"),
            T.reinterpret(B[T.Ramp(0, 1, 4)], ty="int32"),
            T.int32(0),
            T.bool(1),
            ty="int32",
        )


AMDGPU_SDOT4_INTRIN = "sdot4"

dp4a_desc, _ = get_dp4a_intrin("int8", "int8", "int32")
TensorIntrin.register(AMDGPU_SDOT4_INTRIN, dp4a_desc, sdot4)

WARP_SIZE = 64
M_DIM = 16
N_DIM = 16


def shared_16x4_to_local_64x1_layout_A(i, j):
    thread_id = j * 16 + i
    return thread_id, IntImm("int32", 0)


def thread_id_shared_access_64x1_to_16x4_layout_A(thread_id, local_id):
    i = thread_id % 16
    j = thread_id // 16 + local_id
    return i, j


def shared_4x16_to_local_64x1_layout_B(i, j):
    thread_id = i * 16 + j
    return thread_id, IntImm("int32", 0)


def thread_id_shared_access_64x1_to_4x16_layout_B(thread_id, local_id):
    i = thread_id // 16
    j = thread_id % 16 + local_id
    return i, j


def shared_16x16_to_local_64x4_layout_C(i, j):
    thread_id = j + (i // 4) * 16
    local = i % 4
    return thread_id, local


def thread_id_shared_access_64x4_to_16x16_layout_A(thread_id, local_id):
    i = thread_id % 16
    j = (thread_id // 16) * 4 + local_id
    return i, j


def shared_16x16_to_local_64x4_layout_A(i, j):
    thread_id = i + 16 * (j // 4)
    local = j % 4
    return thread_id, local


def thread_id_shared_access_64x4_to_16x16_layout_B(thread_id, local_id):
    i = local_id + (thread_id // 16) * 4
    j = thread_id % 16
    return i, j


def shared_16x16_to_local_64x4_layout_B(i, j):
    thread_id = j + (i // 4) * 16
    local = i % 4
    return thread_id, local


def thread_id_shared_access_64x4_to_16x16_layout_C(thread_id, local_id):
    i = local_id + (thread_id // 16) * 4
    j = thread_id % 16
    return i, j


def get_mma_fill_intrin(dtype, local_size):
    zero = IntImm("int32", 0).astype(dtype)

    # Assume M = N = 16
    index_map = shared_16x16_to_local_64x4_layout_C

    @Ts.function
    def mma_fill_desc(
        C_warp: T.Tensor([WARP_SIZE, local_size], dtype=dtype, scope="local"),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes(C_warp[0:WARP_SIZE, 0:local_size])
            for i0, i1 in T.grid(M_DIM, N_DIM):
                with Ts.sblock("C_warp"):
                    i, j = Ts.axis.remap("SS", [i0, i1])
                    warp_indices = index_map(i, j)
                    Ts.reads()
                    Ts.writes(C_warp[warp_indices[0], warp_indices[1]])
                    C_warp[warp_indices[0], warp_indices[1]] = zero

    mma_fill_impl_c_warp_offset = T.dynamic("mma_fill_impl_c_warp_offset", "int32")

    @Ts.function
    def mma_fill_impl(
        C_warp: T.Tensor(
            [1, local_size],
            dtype=dtype,
            scope="local",
            layout=T.TileLayout(T.S[[1, local_size]] + (mma_fill_impl_c_warp_offset)),
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes(C_warp[0:1, 0:local_size])
            _tx = T.launch_thread("threadIdx.x", WARP_SIZE)
            for local_id in T.serial(0, local_size):
                C_warp[0, local_id] = zero

    return mma_fill_desc, mma_fill_impl


def get_mfma_load_intrin(
    k_dim=4,
    dtype="float32",
    scope="shared",
    is_b=False,
    transposed=False,
):
    local_size = (M_DIM * k_dim) // WARP_SIZE if not is_b else (N_DIM * k_dim) // WARP_SIZE
    memory_shape = (M_DIM, k_dim)
    if is_b:
        memory_shape = (N_DIM, k_dim) if transposed else (k_dim, N_DIM)

    row_dim, col_dim = memory_shape

    if k_dim == 4:
        index_map = shared_16x4_to_local_64x1_layout_A
        reverse_index_map = thread_id_shared_access_64x1_to_16x4_layout_A
        if is_b:
            index_map = (
                shared_16x4_to_local_64x1_layout_A
                if transposed
                else shared_4x16_to_local_64x1_layout_B
            )
            reverse_index_map = (
                thread_id_shared_access_64x1_to_16x4_layout_A
                if transposed
                else thread_id_shared_access_64x1_to_4x16_layout_B
            )
    elif k_dim == 16:
        index_map = shared_16x16_to_local_64x4_layout_A
        reverse_index_map = thread_id_shared_access_64x4_to_16x16_layout_A

        if is_b:
            index_map = (
                shared_16x16_to_local_64x4_layout_A
                if transposed
                else shared_16x16_to_local_64x4_layout_B
            )
            reverse_index_map = (
                thread_id_shared_access_64x4_to_16x16_layout_A
                if transposed
                else thread_id_shared_access_64x4_to_16x16_layout_B
            )
    else:
        raise ValueError("k_dim must be 4 or 16 currently")

    mfma_load_desc_reg_offset = T.dynamic("mfma_load_desc_reg_offset", "int32")
    mfma_load_desc_memory_offset = T.dynamic("mfma_load_desc_memory_offset", "int32")

    @Ts.function
    def mfma_load_desc(
        reg: T.Tensor(
            (WARP_SIZE, local_size),
            dtype,
            layout=T.TileLayout(T.S[(WARP_SIZE, local_size)] + (mfma_load_desc_reg_offset)),
            scope="local",
        ),
        memory: T.Tensor(
            memory_shape,
            dtype,
            layout=T.TileLayout(T.S[memory_shape] + (mfma_load_desc_memory_offset)),
            scope=scope,
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(memory[0:row_dim, 0:col_dim])
            Ts.writes(reg[0:WARP_SIZE, 0:local_size])

            for ax0, ax1 in T.grid(row_dim, col_dim):
                with Ts.sblock("memory_reg"):
                    v0, v1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(memory[v0, v1])

                    warp_indices = index_map(v0, v1)
                    Ts.writes(reg[warp_indices[0], warp_indices[1]])
                    reg[warp_indices[0], warp_indices[1]] = memory[v0, v1]

    s0 = T.dynamic("s0", "int32")
    s1 = T.dynamic("s1", "int32")

    mfma_load_impl_reg_offset = T.dynamic("mfma_load_impl_reg_offset", "int32")
    mfma_load_impl_memory_offset = T.dynamic("mfma_load_impl_memory_offset", "int32")

    @Ts.function
    def mfma_load_impl(
        reg: T.Tensor(
            (1, local_size),
            dtype,
            align=64,
            layout=T.TileLayout(T.S[(1, local_size)] + (mfma_load_impl_reg_offset)),
            scope="local",
        ),
        memory: T.Tensor(
            memory_shape,
            dtype,
            align=64,
            scope=scope,
            layout=T.TileLayout(T.S[memory_shape : [s0, s1]] + (mfma_load_impl_memory_offset)),
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(memory[0:row_dim, 0:col_dim])
            Ts.writes(reg[0:1, 0:local_size])
            for local_id in T.serial(0, local_size):
                tx = T.launch_thread("threadIdx.x", WARP_SIZE)
                matrix_indices = reverse_index_map(tx, local_id)
                reg[0, local_id] = memory[matrix_indices[0], matrix_indices[1]]

    return mfma_load_desc, mfma_load_impl


def get_mfma_intrin(k_dim, in_dtype="float32", out_dtype="float32", b_transposed=False):
    local_size = (M_DIM * k_dim) // WARP_SIZE
    local_size_out = (M_DIM * N_DIM) // WARP_SIZE
    if k_dim == 4:
        index_map_A = shared_16x4_to_local_64x1_layout_A
        index_map_B = shared_4x16_to_local_64x1_layout_B
        index_map_C = shared_16x16_to_local_64x4_layout_C
    elif k_dim == 16:
        index_map_A = shared_16x16_to_local_64x4_layout_A
        index_map_B = shared_16x16_to_local_64x4_layout_B
        index_map_C = shared_16x16_to_local_64x4_layout_C
    else:
        raise ValueError("k_dim must be 4 or 16 currently")

    out_dtype_abbrv = {"float16": "f16", "float32": "f32", "int8": "i8", "int32": "i32"}[out_dtype]

    in_dtype_abbrv = {"float16": "f16", "float32": "f32", "int8": "i8", "int32": "i32"}[in_dtype]

    mfma_intrin = f"llvm.amdgcn.mfma.{out_dtype_abbrv}.{M_DIM}x{N_DIM}x{k_dim}{in_dtype_abbrv}"

    def maybe_cast(v):
        if out_dtype != in_dtype:
            return Cast(out_dtype, v)
        return v

    def maybe_swap(i, j):
        if b_transposed:
            return j, i
        return i, j

    mfma_sync_desc_a_offset = T.dynamic("mfma_sync_desc_a_offset", "int32")
    mfma_sync_desc_b_offset = T.dynamic("mfma_sync_desc_b_offset", "int32")
    mfma_sync_desc_c_offset = T.dynamic("mfma_sync_desc_c_offset", "int32")

    @Ts.function
    def mfma_sync_desc(
        A: T.Tensor(
            (WARP_SIZE, local_size),
            in_dtype,
            layout=T.TileLayout(T.S[(WARP_SIZE, local_size)] + (mfma_sync_desc_a_offset)),
            scope="local",
        ),
        B: T.Tensor(
            (WARP_SIZE, local_size),
            in_dtype,
            layout=T.TileLayout(T.S[(WARP_SIZE, local_size)] + (mfma_sync_desc_b_offset)),
            scope="local",
        ),
        C: T.Tensor(
            (WARP_SIZE, local_size_out),
            out_dtype,
            layout=T.TileLayout(T.S[(WARP_SIZE, local_size_out)] + (mfma_sync_desc_c_offset)),
            scope="local",
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(
                C[0:WARP_SIZE, 0:local_size_out],
                A[0:WARP_SIZE, 0:local_size],
                B[0:WARP_SIZE, 0:local_size],
            )
            Ts.writes(C[0:WARP_SIZE, 0:local_size_out])

            for i, j, k in T.grid(M_DIM, N_DIM, k_dim):
                with Ts.sblock("C"):
                    vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                    b_indices = maybe_swap(vk, vj)

                    c_warp_indices = index_map_C(vi, vj)
                    a_warp_indices = index_map_A(vi, vk)
                    b_warp_indices = index_map_B(b_indices[0], b_indices[1])

                    Ts.reads(
                        C[c_warp_indices[0], c_warp_indices[1]],
                        A[a_warp_indices[0], a_warp_indices[1]],
                        B[b_warp_indices[0], b_warp_indices[1]],
                    )
                    Ts.writes(C[c_warp_indices[0], c_warp_indices[1]])

                    C[c_warp_indices[0], c_warp_indices[1]] += maybe_cast(
                        A[a_warp_indices[0], a_warp_indices[1]]
                    ) * maybe_cast(B[b_warp_indices[0], b_warp_indices[1]])

    mfma_sync_impl_float_a_offset = T.dynamic("mfma_sync_impl_float_a_offset", "int32")
    mfma_sync_impl_float_b_offset = T.dynamic("mfma_sync_impl_float_b_offset", "int32")
    mfma_sync_impl_float_c_offset = T.dynamic("mfma_sync_impl_float_c_offset", "int32")

    @Ts.function
    def mfma_sync_impl_float(
        A: T.Tensor(
            (1, local_size),
            in_dtype,
            layout=T.TileLayout(T.S[(1, local_size)] + (mfma_sync_impl_float_a_offset)),
            scope="local",
        ),
        B: T.Tensor(
            (1, local_size),
            in_dtype,
            layout=T.TileLayout(T.S[(1, local_size)] + (mfma_sync_impl_float_b_offset)),
            scope="local",
        ),
        C: T.Tensor(
            (1, local_size_out),
            out_dtype,
            layout=T.TileLayout(T.S[(1, local_size_out)] + (mfma_sync_impl_float_c_offset)),
            scope="local",
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(
                A[0:1, 0:local_size],
                B[0:1, 0:local_size],
                C[0:1, 0:local_size_out],
            )
            Ts.writes(C[0:1, 0:local_size_out])
            _tx = T.launch_thread("threadIdx.x", WARP_SIZE)
            C[0, T.ramp(0, 1, local_size_out)] = T.call_llvm_pure_intrin(
                T.llvm_lookup_intrinsic_id(mfma_intrin),
                A[0, T.ramp(0, 1, local_size) if T.constexpr(local_size > 1) else 0],
                B[0, T.ramp(0, 1, local_size) if T.constexpr(local_size > 1) else 0],
                C[0, T.ramp(0, 1, local_size_out)],
                T.int32(0),
                T.int32(0),
                T.int32(0),
                ty=f"{out_dtype}x4",
            )

    mfma_sync_impl_integer_a_offset = T.dynamic("mfma_sync_impl_integer_a_offset", "int32")
    mfma_sync_impl_integer_b_offset = T.dynamic("mfma_sync_impl_integer_b_offset", "int32")
    mfma_sync_impl_integer_c_offset = T.dynamic("mfma_sync_impl_integer_c_offset", "int32")

    @Ts.function
    def mfma_sync_impl_integer(
        A: T.Tensor(
            (1, local_size),
            in_dtype,
            layout=T.TileLayout(T.S[(1, local_size)] + (mfma_sync_impl_integer_a_offset)),
            scope="local",
        ),
        B: T.Tensor(
            (1, local_size),
            in_dtype,
            layout=T.TileLayout(T.S[(1, local_size)] + (mfma_sync_impl_integer_b_offset)),
            scope="local",
        ),
        C: T.Tensor(
            (1, local_size_out),
            out_dtype,
            layout=T.TileLayout(T.S[(1, local_size_out)] + (mfma_sync_impl_integer_c_offset)),
            scope="local",
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(
                A[0:1, 0:local_size],
                B[0:1, 0:local_size],
                C[0:1, 0:local_size_out],
            )
            Ts.writes(C[0:1, 0:local_size_out])
            _tx = T.launch_thread("threadIdx.x", WARP_SIZE)

            C[0, T.ramp(0, 1, local_size_out)] = T.call_llvm_pure_intrin(
                T.llvm_lookup_intrinsic_id(mfma_intrin),
                T.call_intrin(
                    "tirx.reinterpret",
                    A[0, T.ramp(0, 1, local_size) if T.constexpr(local_size > 1) else 0],
                    ty="int32",
                ),
                T.call_intrin(
                    "tirx.reinterpret",
                    B[0, T.ramp(0, 1, local_size) if T.constexpr(local_size > 1) else 0],
                    ty="int32",
                ),
                C[0, T.ramp(0, 1, local_size_out)],
                T.int32(0),
                T.int32(0),
                T.int32(0),
                ty=f"{out_dtype}x4",
            )

    return (
        (mfma_sync_desc, mfma_sync_impl_integer)
        if in_dtype == "int8"
        else (mfma_sync_desc, mfma_sync_impl_float)
    )


def get_mfma_store_intrin(local_size=4, dtype="float32", scope="global"):
    index_map = shared_16x16_to_local_64x4_layout_C

    @Ts.function
    def mfma_store_desc(
        C_warp: T.Tensor([WARP_SIZE, local_size], dtype=dtype, scope="local"),
        C: T.Tensor([M_DIM, N_DIM], dtype=dtype, scope=scope),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(C_warp[0:WARP_SIZE, 0:local_size])
            Ts.writes(C[0:M_DIM, 0:N_DIM])
            for i0, i1 in T.grid(M_DIM, N_DIM):
                with Ts.sblock("C_warp"):
                    v0, v1 = Ts.axis.remap("SS", [i0, i1])
                    warp_indices = index_map(v0, v1)
                    Ts.reads(C_warp[warp_indices[0], warp_indices[1]])
                    Ts.writes(C[v0, v1])
                    C[v0, v1] = C_warp[warp_indices[0], warp_indices[1]]

    s0 = T.dynamic("s0", "int32")
    s1 = T.dynamic("s1", "int32")

    mfma_store_impl_c_warp_offset = T.dynamic("mfma_store_impl_c_warp_offset", "int32")
    mfma_store_impl_c_offset = T.dynamic("mfma_store_impl_c_offset", "int32")

    @Ts.function
    def mfma_store_impl(
        C_warp: T.Tensor(
            [1, local_size],
            dtype=dtype,
            scope="local",
            layout=T.TileLayout(T.S[[1, local_size]] + (mfma_store_impl_c_warp_offset)),
        ),
        C: T.Tensor(
            [M_DIM, N_DIM],
            dtype=dtype,
            scope=scope,
            layout=T.TileLayout(T.S[[M_DIM, N_DIM] : [s0, s1]] + (mfma_store_impl_c_offset)),
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(C_warp[0:1, 0:local_size])
            Ts.writes(C[0:M_DIM, 0:N_DIM])
            tx = T.launch_thread("threadIdx.x", WARP_SIZE)
            for i in range(local_size):
                C[((tx // 16) * 4) + i, (tx % 16)] = C_warp[0, i]

    return mfma_store_desc, mfma_store_impl


ROCM_MFMA_fill_16x16_f32_INTRIN = "ROCM_mfma_fill_16x16_f32"
TensorIntrin.register(ROCM_MFMA_fill_16x16_f32_INTRIN, *get_mma_fill_intrin("float32", 4))

ROCM_MFMA_fill_16x16_i32_INTRIN = "ROCM_mfma_fill_16x16_i32"
TensorIntrin.register(ROCM_MFMA_fill_16x16_i32_INTRIN, *get_mma_fill_intrin("int", 4))

ROCM_MFMA_LOAD_16x16_A_SHARED_s8_INTRIN = "rocm_mfma_load_16x16_a_shared_s8"
TensorIntrin.register(
    ROCM_MFMA_LOAD_16x16_A_SHARED_s8_INTRIN, *get_mfma_load_intrin(16, "int8", "shared")
)
ROCM_MFMA_LOAD_16x16_B_SHARED_s8_INTRIN = "rocm_mfma_load_b_16x16_shared_s8"
TensorIntrin.register(
    ROCM_MFMA_LOAD_16x16_B_SHARED_s8_INTRIN, *get_mfma_load_intrin(16, "int8", "shared", is_b=True)
)

ROCM_MFMA_LOAD_16x16_A_SHARED_f16_INTRIN = "rocm_mfma_load_16x16_a_shared_f16"
TensorIntrin.register(
    ROCM_MFMA_LOAD_16x16_A_SHARED_f16_INTRIN, *get_mfma_load_intrin(16, "float16", "shared")
)
ROCM_MFMA_LOAD_16x16_B_SHARED_f16_INTRIN = "rocm_mfma_load_b_16x16_shared_f16"
TensorIntrin.register(
    ROCM_MFMA_LOAD_16x16_B_SHARED_f16_INTRIN,
    *get_mfma_load_intrin(16, "float16", "shared", is_b=True),
)

ROCM_MFMA_LOAD_16x4_A_SHARED_f32_INTRIN = "rocm_mfma_load_16x4_a_shared_f32"
TensorIntrin.register(
    ROCM_MFMA_LOAD_16x4_A_SHARED_f32_INTRIN, *get_mfma_load_intrin(4, "float32", "shared")
)
ROCM_MFMA_LOAD_16x4_B_SHARED_f32_INTRIN = "rocm_mfma_load_b_16x4_shared_f32"
TensorIntrin.register(
    ROCM_MFMA_LOAD_16x4_B_SHARED_f32_INTRIN,
    *get_mfma_load_intrin(4, "float32", "shared", is_b=True),
)

ROCM_MFMA_f32f32f32_INTRIN = "rocm_mfma_f32f32f32"
TensorIntrin.register(ROCM_MFMA_f32f32f32_INTRIN, *get_mfma_intrin(4, "float32", "float32"))

ROCM_MFMA_f16f16f32_INTRIN = "rocm_mfma_f16f16f32"
TensorIntrin.register(ROCM_MFMA_f16f16f32_INTRIN, *get_mfma_intrin(16, "float16", "float32"))

ROCM_MFMA_s8s8s32_INTRIN = "rocm_mfma_s8s8s32"
TensorIntrin.register(ROCM_MFMA_s8s8s32_INTRIN, *get_mfma_intrin(16, "int8", "int32"))

ROCM_MFMA_STORE_16x16_s32_INTRIN = "rocm_mfma_store_16x16_s32"
TensorIntrin.register(
    ROCM_MFMA_STORE_16x16_s32_INTRIN, *get_mfma_store_intrin(4, "int32", "global")
)

ROCM_MFMA_STORE_16x16_f32_INTRIN = "rocm_mfma_store_16x16_f32"
TensorIntrin.register(
    ROCM_MFMA_STORE_16x16_f32_INTRIN, *get_mfma_store_intrin(4, "float32", "global")
)
