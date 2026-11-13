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
# pylint: disable=missing-function-docstring,missing-module-docstring
# ruff: noqa: E501, E741, F401
import pytest

import tvm
import tvm.s_tir.tensor_intrin
import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import script as Ts
from tvm.s_tir.schedule.testing import (
    assert_structural_equal_ignore_global_symbol,
    verify_trace_roundtrip,
)
from tvm_tirx import tirx
from tvm_tirx.script import tirx as T

# pylint: disable=no-member,invalid-name,unused-variable


@Ts.function
def elementwise(A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")) -> None:
    B = Ts.sblock_alloc_tensor((128, 128))

    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@Ts.function
def elementwise_multi_producer_consumer(
    A: T.Tensor((128, 128), "float32"),
    C: T.Tensor((128, 128), "float32"),
    D: T.Tensor((128, 128), "float32"),
) -> None:
    B = Ts.sblock_alloc_tensor((128, 128))

    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0  # B has two consumers
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0
    for i, j in T.grid(128, 128):
        with Ts.sblock("D"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            D[vi, vj] = B[vi, vj] + 2.0 + C[vi, vj]  # D has two producers


@Ts.function
def elementwise_multi_consumer_inlined(
    A: T.Tensor((128, 128), "float32"),
    C: T.Tensor((128, 128), "float32"),
    D: T.Tensor((128, 128), "float32"),
) -> None:
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = A[vi, vj] * 2.0 + 1.0
    for i, j in T.grid(128, 128):
        with Ts.sblock("D"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            D[vi, vj] = A[vi, vj] * 2.0 + 2.0 + C[vi, vj]


@Ts.function
def elementwise_standalone(
    A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
) -> None:
    B = Ts.sblock_alloc_tensor((128, 128))

    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = A[vi, vj] + 1.0


@Ts.function
def elementwise_standalone_dce(
    A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
) -> None:
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = A[vi, vj] + 1.0


@Ts.function
def elementwise_under_loop(
    A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
) -> None:
    B = Ts.sblock_alloc_tensor((128, 128))
    for i in T.serial(0, 128):
        for j in T.serial(0, 128):
            with Ts.sblock("B"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] * 2.0
        for j in T.serial(0, 128):
            with Ts.sblock("C"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                C[vi, vj] = B[vi, vj] + 1.0


@Ts.function
def elementwise_inlined(
    A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
) -> None:
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = A[vi, vj] * 2.0 + 1.0


@Ts.function
def fail_multi_reader_writer(
    A: T.Tensor((128, 128), "float32"), D: T.Tensor((128, 128), "float32")
) -> None:
    B = Ts.sblock_alloc_tensor((128, 128))
    C = Ts.sblock_alloc_tensor((128, 128))

    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
            C[vi, vj] = A[vi, vj] + 2.0
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            D[vi, vj] = B[vi, vj] + C[vi, vj]


@Ts.function
def elementwise_multi_reverse_loads(
    A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
) -> None:
    B = Ts.sblock_alloc_tensor((128, 128))

    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = (B[vi, vj] + 1.0) * (B[vi, vj] * 2.0) + 3.0


@Ts.function
def elementwise_multi_reverse_loads_inlined(
    A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
) -> None:
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = (A[vi, vj] * 2.0 + 1.0) * (A[vi, vj] * 2.0 * 2.0) + 3.0


@Ts.function
def elementwise_reverse_affine_load(
    A: T.Tensor((128, 128), "float32"), C: T.Tensor((8, 32, 8, 8), "float32")
) -> None:
    B = Ts.sblock_alloc_tensor((128, 128))
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j, k, l in T.grid(8, 32, 8, 8):
        with Ts.sblock("C"):
            vi, vj, vk, vl = Ts.axis.remap("SSSS", [i, j, k, l])
            C[vi, vj, vk, vl] = B[
                ((((vi * 32) + vj) * 8 + vk) * 8 + vl) // 128,
                ((((vi * 32) + vj) * 8 + vk) * 8 + vl) % 128,
            ]


@Ts.function
def elementwise_reverse_affine_load_inlined(
    A: T.Tensor((128, 128), "float32"), C: T.Tensor((8, 32, 8, 8), "float32")
) -> None:
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[
                (vj + vi * 128) // 2048,
                (vj + vi * 128) // 64 % 32,
                ((vj + vi * 128) // 8) % 8,
                (vj + vi * 128) % 8,
            ] = A[vi, vj] * 2.0


@Ts.function
def elementwise_reverse_affine_load_unit_iter(
    A: T.Tensor((128, 128), "float32"),
    B: T.Tensor((8, 16, 1), "float32"),
    D: T.Tensor((1, 8, 16, 128), "float32"),
) -> None:
    C = Ts.sblock_alloc_tensor((128, 128))
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = A[vi, vj] * 2.0
    for i, j, k, l in T.grid(1, 8, 16, 128):
        with Ts.sblock("C"):
            vi, vj, vk, vl = Ts.axis.remap("SSSS", [i, j, k, l])
            D[vi, vj, vk, vl] = C[vj * 16 + vk, vl] + B[vj, vk, vi]


@Ts.function
def elementwise_reverse_affine_load_unit_iter_inlined(
    A: T.Tensor((128, 128), "float32"),
    B: T.Tensor((8, 16, 1), "float32"),
    D: T.Tensor((1, 8, 16, 128), "float32"),
) -> None:
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            D[0, vi // 16, vi % 16, vj] = A[vi, vj] * 2.0 + B[vi // 16, vi % 16, 0]


@Ts.function
def elementwise_reverse_affine_load_unit_iter_simplified(
    A: T.Tensor((128, 128), "float32"),
    B: T.Tensor((8, 16, 1), "float32"),
    D: T.Tensor((1, 8, 16, 128), "float32"),
) -> None:
    C = Ts.sblock_alloc_tensor((128, 128))
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = A[vi, vj] * 2.0
    for i, j, k in T.grid(8, 16, 128):
        with Ts.sblock("C"):
            vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
            D[0, vi, vj, vk] = C[vi * 16 + vj, vk] + B[vi, vj, 0]


@Ts.function
def elementwise_reverse_affine_load_unit_iter_simplified_inlined(
    A: T.Tensor((128, 128), "float32"),
    B: T.Tensor((8, 16, 1), "float32"),
    D: T.Tensor((1, 8, 16, 128), "float32"),
) -> None:
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            D[0, vi // 16, vi % 16, vj] = A[vi, vj] * 2.0 + B[vi // 16, vi % 16, 0]


@Ts.function
def elementwise_reverse_affine_chain(
    A: T.Tensor((128, 128), "float32"), D: T.Tensor((1, 8, 16, 128), "float32")
):
    B = Ts.sblock_alloc_tensor((128, 128))
    C = Ts.sblock_alloc_tensor((8, 16, 128))
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j, k in T.grid(8, 16, 128):
        with Ts.sblock("C"):
            vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
            C[vi, vj, vk] = B[vi * 16 + vj, vk] + 1.0
    for i, j, k, l in T.grid(1, 8, 16, 128):
        with Ts.sblock("D"):
            vi, vj, vk, vl = Ts.axis.remap("SSSS", [i, j, k, l])
            D[vi, vj, vk, vl] = C[vj, vk, vl]


@Ts.function
def elementwise_reverse_affine_chain_inlined(
    A: T.Tensor((128, 128), "float32"), D: T.Tensor((1, 8, 16, 128), "float32")
) -> None:
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            D[0, vi // 16, vi % 16, vj] = A[vi, vj] * 2.0 + 1.0


@Ts.function
def elementwise_multi_reverse_affine_load(
    A: T.Tensor((128, 128), "float32"),
    C: T.Tensor((8, 16, 128), "float32"),
) -> None:
    B = Ts.sblock_alloc_tensor((128, 128))
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j, k in T.grid(8, 16, 128):
        with Ts.sblock("C"):
            vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
            C[vi, vj, vk] = B[vi * 16 + vj, vk] + B[vi * 16 + vj, vk]


@Ts.function
def elementwise_multi_reverse_affine_load_inlined(
    A: T.Tensor((128, 128), "float32"),
    C: T.Tensor((8, 16, 128), "float32"),
) -> None:
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi // 16, vi % 16, vj] = A[vi, vj] * 2.0 + A[vi, vj] * 2.0


@Ts.function
def elementwise_reverse_non_affine_load(
    A: T.Tensor((128, 128), "float32"), C: T.Tensor((8, 16, 128), "float32")
) -> None:
    B = Ts.sblock_alloc_tensor((128, 128))
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j, k in T.grid(8, 16, 128):
        with Ts.sblock("C"):
            vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
            C[vi, vj, vk] = B[vi * 16 + vj, vi * 16 + vj]


@Ts.function
def opaque_access_load(
    A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
) -> None:
    B = Ts.sblock_alloc_tensor((128, 128))

    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads(B[0:128, 0:128])
            Ts.writes(C[0:128, 0:128])
            T.evaluate(B.ptr_to([0] * len(B.shape)))
            C[vi, vj] = B[vi, vj] + 1.0


@Ts.function
def opaque_access_store(
    A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
) -> None:
    B = Ts.sblock_alloc_tensor((128, 128))

    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads(B[0:128, 0:128])
            Ts.writes(C[0:128, 0:128])
            T.evaluate(B.ptr_to([0] * len(B.shape)))
            T.evaluate(C.ptr_to([0] * len(C.shape)))
            C[vi, vj] = B[vi, vj] + 1.0


@Ts.function
def tensor_matched(A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")) -> None:
    B = Ts.sblock_alloc_tensor((128, 128))

    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Bb = Ts.match_tensor(B[vi : vi + 1, vj], (1, 1))
            C[vi, vj] = Bb[0, 0] + 1.0


@Ts.function
def elementwise_predicate(
    A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
) -> None:
    B = Ts.sblock_alloc_tensor((128, 128))

    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.where(B[i, j] < 10.0)
            C[vi, vj] = B[vi, vj] + 1.0


@Ts.function
def elementwise_predicate_inlined(
    A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
) -> None:
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.where(A[i, j] * 2.0 < 10.0)
            C[vi, vj] = A[vi, vj] * 2.0 + 1.0


@Ts.function
def elementwise_multi_loads(
    A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
) -> None:
    B = Ts.sblock_alloc_tensor((128, 128))

    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + B[vi, vj + 1] + B[vi, vj + 2]


@Ts.function
def elementwise_multi_loads_inlined(
    A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
) -> None:
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = A[vi, vj] * 2.0 + A[vi, vj + 1] * 2.0 + A[vi, vj + 2] * 2.0


@Ts.function
def access_opaque_ptr_then_elemwise(
    A: T.Tensor([1024], "float32"), B: T.Tensor([1024], "float32")
) -> None:
    A_cache = Ts.sblock_alloc_tensor([1024])
    BB = Ts.sblock_alloc_tensor([1024])
    with Ts.sblock("opaque"):
        # annotated opaque partial access
        Ts.reads(A[0:512])
        Ts.writes(A_cache[0:512])
        T.evaluate(A.ptr_to([0] * len(A.shape)))
        T.evaluate(A_cache.ptr_to([0] * len(A_cache.shape)))
    for i in range(512):
        with Ts.sblock("BB"):
            vi = Ts.axis.remap("S", [i])
            BB[vi] = A_cache[vi] * 2.0
    for i in range(512):
        with Ts.sblock("B"):
            vi = Ts.axis.remap("S", [i])
            B[vi] = BB[vi] + 1.0


@Ts.function
def access_opaque_ptr_then_elemwise_inline(
    A: T.Tensor([1024], dtype="float32"), B: T.Tensor([1024], dtype="float32")
) -> None:
    A_cache = Ts.sblock_alloc_tensor([1024], dtype="float32")
    with Ts.sblock("opaque"):
        # annotated opaque partial access should be kept
        Ts.reads(A[0:512])
        Ts.writes([A_cache[0:512]])
        T.evaluate(A.ptr_to([0] * len(A.shape)))
        T.evaluate(A_cache.ptr_to([0] * len(A_cache.shape)))
    for i in T.serial(0, 512):
        with Ts.sblock("B"):
            vi = Ts.axis.spatial(512, i)
            Ts.reads([A_cache[vi]])
            Ts.writes([B[vi]])
            B[vi] = A_cache[vi] * 2.0 + 1.0


@Ts.function
def matmul_relu(
    A: T.Tensor([512, 512], dtype="float32"),
    B: T.Tensor([512, 512], dtype="float32"),
    compute: T.Tensor([512, 512], dtype="float32"),
) -> None:
    C = Ts.sblock_alloc_tensor([512, 512], dtype="float32")
    for i0, i1, i2 in T.grid(512, 512, 512):
        with Ts.sblock("C"):
            i, j, k = Ts.axis.remap("SSR", [i0, i1, i2])
            Ts.reads([C[i, j], A[i, k], B[k, j]])
            Ts.writes([C[i, j]])
            with Ts.init():
                C[i, j] = T.float32(0)
            C[i, j] = C[i, j] + A[i, k] * B[k, j]
    for i0, i1 in T.grid(512, 512):
        with Ts.sblock("compute"):
            i0_1, i1_1 = Ts.axis.remap("SS", [i0, i1])
            Ts.reads([C[i0_1, i1_1]])
            Ts.writes([compute[i0_1, i1_1]])
            compute[i0_1, i1_1] = T.max(C[i0_1, i1_1], T.float32(0))


@Ts.function
def elementwise_output(
    A: T.Tensor((128, 128), "float32"),
    B: T.Tensor((128, 128), "float32"),
    C: T.Tensor((128, 128), "float32"),
) -> None:
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@Ts.function
def inline_block_with_init(
    A: T.Tensor((1, 512, 7, 7), "float32"),
    B: T.Tensor((1, 512, 1, 1), "float32"),
) -> None:
    B_rf = Ts.sblock_alloc_tensor([1, 512, 1, 1, 49], dtype="float32")
    for i0, i1, i2, i3, i4, i5 in T.grid(1, 512, 1, 1, 49, 1):
        with Ts.sblock("tensor_rf"):
            vi4 = Ts.axis.spatial(49, i4)
            ax0 = Ts.axis.spatial(1, 0)
            ax1 = Ts.axis.spatial(512, i1)
            ax2 = Ts.axis.spatial(1, 0)
            ax3 = Ts.axis.spatial(1, 0)
            with Ts.init():
                B_rf[ax0, ax1, ax2, ax3, vi4] = T.float32(0)
            B_rf[ax0, ax1, ax2, ax3, vi4] = (
                B_rf[ax0, ax1, ax2, ax3, vi4]
                + A[
                    ax0,
                    ax1,
                    ax2 * 7 + vi4 // 7,
                    ax3 * 7 + vi4 % 7,
                ]
            )
    for i0, i1 in T.grid(1, 512):
        for ax0, ax1, ax2, ax3, ax4 in T.grid(49, 1, 1, 1, 1):
            with Ts.sblock("tensor"):
                vi4, ax0_1 = Ts.axis.remap("RS", [ax0, ax1])
                ax1_1 = Ts.axis.spatial(512, i1 + ax2)
                ax2_1, ax3_1 = Ts.axis.remap("SS", [ax3, ax4])
                with Ts.init():
                    B[ax0_1, ax1_1, ax2_1, ax3_1] = T.float32(0)
                B[ax0_1, ax1_1, ax2_1, ax3_1] = (
                    B[ax0_1, ax1_1, ax2_1, ax3_1] + B_rf[ax0_1, ax1_1, ax2_1, ax3_1, vi4]
                )


@Ts.function
def exp_exp_opaque_access_with_pointer(
    lookup_table: T.Tensor((1024,), "int8"),
    x: T.Tensor((16,), "float16"),
    compute: T.Tensor((16,), "float16"),
) -> None:
    compute_1 = Ts.sblock_alloc_tensor([16], dtype="float16")
    for i0 in T.serial(16):
        with Ts.sblock("compute"):
            i0_1 = Ts.axis.spatial(16, i0)
            Ts.reads(x[i0_1])
            Ts.writes(compute_1[i0_1])
            compute_1[i0_1] = T.exp(x[i0_1])
    for i0 in T.serial(16):
        with Ts.sblock("compute_1"):
            i0_2 = Ts.axis.spatial(16, i0)
            Ts.reads(lookup_table[0:1024], compute_1[i0_2])
            Ts.writes(compute[i0_2])
            T.evaluate(lookup_table.ptr_to([0] * len(lookup_table.shape)))
            compute[i0_2] = T.exp(
                compute_1[i0_2],
            )


@Ts.function
def exp_exp_opaque_access_with_pointer_inlined(
    lookup_table: T.Tensor((1024,), "int8"),
    x: T.Tensor((16,), "float16"),
    compute: T.Tensor((16,), "float16"),
) -> None:
    for i0 in T.serial(16):
        with Ts.sblock("compute_1"):
            i0_1 = Ts.axis.spatial(16, i0)
            # Pointer operands have no direction metadata, so recomputed regions
            # conservatively include the referenced storage in both sets.
            Ts.reads(x[i0_1], lookup_table[0:1024])
            Ts.writes(compute[i0_1], lookup_table[0:1024])
            T.evaluate(lookup_table.ptr_to([0] * len(lookup_table.shape)))
            compute[i0_1] = T.exp(
                T.exp(x[i0_1]),
            )


@Ts.function
def elementwise_overcomputed_producer(
    A: T.Tensor((128, 128), "float32"), C: T.Tensor((127, 127), "float32")
) -> None:
    B = Ts.sblock_alloc_tensor((128, 128))
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(127, 127):
        with Ts.sblock("C"):
            cvi, cvj = Ts.axis.remap("SS", [i, j])
            C[cvi, cvj] = B[cvi, cvj] + 1.0


@Ts.function
def elementwise_overcomputed_producer_reverse_inlined(
    A: T.Tensor((128, 128), "float32"), C: T.Tensor((127, 127), "float32")
) -> None:
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.where(i < 127 and j < 127)
            C[vi, vj] = A[vi, vj] * 2.0 + 1.0


@Ts.function
def elementwise_overcomputed_producer_simplify_predicate(
    A: T.Tensor((128, 128), "float32"), C: T.Tensor((127, 127), "float32")
) -> None:
    B = Ts.sblock_alloc_tensor((128, 128))
    for i in T.grid(16384):
        with Ts.sblock("B"):
            vi = Ts.axis.spatial(128, i // 128)
            vj = Ts.axis.spatial(128, i % 128)
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(127, 127):
        with Ts.sblock("C"):
            cvi, cvj = Ts.axis.remap("SS", [i, j])
            C[cvi, cvj] = B[cvi, cvj] + 1.0


@Ts.function
def elementwise_overcomputed_producer_simplify_predicate_reverse_inlined(
    A: T.Tensor((128, 128), "float32"), C: T.Tensor((127, 127), "float32")
) -> None:
    for i in T.grid(16384):
        with Ts.sblock("B"):
            vi = Ts.axis.spatial(128, i // 128)
            vj = Ts.axis.spatial(128, i % 128)
            Ts.where(i < 16255 and i % 128 < 127)
            C[vi, vj] = A[vi, vj] * 2.0 + 1.0


@Ts.function
def elementwise_overcomputed_producer_injective_load(
    A: T.Tensor((128, 128), "float32"), C: T.Tensor((127, 127), "float32")
) -> None:
    B = Ts.sblock_alloc_tensor((8, 8, 16, 16))
    for i0, j0, i1, j1 in T.grid(8, 8, 16, 16):
        with Ts.sblock("B"):
            vi, vj, vm, vn = Ts.axis.remap("SSSS", [i0, j0, i1, j1])
            B[vi, vj, vm, vn] = A[vi * 16 + vm, vj * 16 + vn] * 2.0
    for i, j in T.grid(127, 127):
        with Ts.sblock("C"):
            cvi, cvj = Ts.axis.remap("SS", [i, j])
            C[cvi, cvj] = B[cvi // 16, cvj // 16, cvi % 16, cvj % 16] + 1.0


@Ts.function
def elementwise_overcomputed_producer_injective_load_reverse_inlined(
    A: T.Tensor((128, 128), "float32"), C: T.Tensor((127, 127), "float32")
) -> None:
    for i0, j0, i1, j1 in T.grid(8, 8, 16, 16):
        with Ts.sblock("B"):
            vi, vj, vm, vn = Ts.axis.remap("SSSS", [i0, j0, i1, j1])
            Ts.where(i0 * 16 + i1 < 127 and j0 * 16 + j1 < 127)
            C[vm + vi * 16, vn + vj * 16] = A[vi * 16 + vm, vj * 16 + vn] * 2.0 + 1.0


@Ts.function
def elementwise_producer_not_cover_consumer(
    A: T.Tensor((128, 128), "float32"), D: T.Tensor((256, 128), "float32")
) -> None:
    B = Ts.sblock_alloc_tensor((128, 128))
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(256, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            D[vi, vj] = T.if_then_else(vi >= 128, B[vi - 128, vj], T.float32(0))


@Ts.function
def elementwise_producer_is_reduction(
    A: T.Tensor((128, 128), "float32"), D: T.Tensor((128), "float32")
) -> None:
    B = Ts.sblock_alloc_tensor(128)
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SR", [i, j])
            with Ts.init():
                B[vi] = T.float32(0)
            B[vi] = B[vi] + A[vi, vj]
    for i in T.grid(128):
        with Ts.sblock("C"):
            vi = Ts.axis.remap("S", [i])
            D[vi] = B[vi] + 1.0


@Ts.function
def elementwise_predicate_producer(
    A: T.Tensor((128, 128), "float32"), C: T.Tensor((127, 128), "float32")
) -> None:
    B = Ts.sblock_alloc_tensor((127, 128))

    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.where(i < 127)
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(127, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@Ts.function
def elementwise_predicate_producer_inlined(
    A: T.Tensor((128, 128), "float32"), C: T.Tensor((127, 128), "float32")
) -> None:
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            Ts.where(i < 127)
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads(A[vi, vj])
            Ts.writes(C[vi, vj])
            C[vi, vj] = A[vi, vj] * T.float32(2) + T.float32(1)


# fmt: off
@tvm_tirx.script.ir_module
class Conv2dInt8_TensorCore_with_predicate_before:
    @Ts.function
    def main(p0: T.Tensor((16, 56, 56, 64), "int8"), p1: T.Tensor((256, 1, 1, 64), "int8"), p2: T.Tensor((1, 1, 1, 256), "int32"), p3: T.Tensor((1, 1, 1, 256), "int32"), p4: T.Tensor(256, "int32"), p5: T.Tensor(256, "int32"), p6: T.Tensor(256, "int32"), p7: T.Tensor((), "int32"), p8: T.Tensor(1, "int32"), p9: T.Tensor((16, 56, 56, 256), "int32"), compute: T.Tensor((16, 56, 56, 256), "int32")):
        # function attr dict
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        # body
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.unroll_explicit":1024})
            compute_3 = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
            conv2d_nhwc_reindex_shared = Ts.sblock_alloc_tensor([50176, 256], dtype="int32", scope="shared")
            conv2d_nhwc_reindex_shared_wmma_accumulator = Ts.sblock_alloc_tensor([50176, 256], dtype="int32", scope="wmma.accumulator")
            pad_temp_reindex_shared = Ts.sblock_alloc_tensor([50176, 64], dtype="int8", scope="shared")
            p1_reindex_shared = Ts.sblock_alloc_tensor([1, 1, 256, 64], dtype="int8", scope="shared")
            pad_temp_reindex_shared_wmma_matrix_a = Ts.sblock_alloc_tensor([50176, 64], dtype="int8", scope="wmma.matrix_a")
            p1_reindex_shared_wmma_matrix_b = Ts.sblock_alloc_tensor([1, 1, 256, 64], dtype="int8", scope="wmma.matrix_b")
            for ax2_0_0_ax3_0_0_fused in T.parallel(32, annotations={"thread_binding": "blockIdx.y"}):
                for ax2_0_1_ax3_0_1_fused in T.parallel(196, annotations={"thread_binding": "blockIdx.x"}):
                    for ax2_0_2_ax3_0_2_fused in T.parallel(4, annotations={"thread_binding": "threadIdx.y"}):
                        for ax0_0, ax1_0, ax4_0_0 in T.grid(1, 1, 2):
                            for ax0_ax1_fused in T.serial(1024):
                                with Ts.sblock("pad_temp_reindex_shared"):
                                    v0 = Ts.axis.spatial(50176, ax2_0_0_ax3_0_0_fused // 4 * 6272 + ax2_0_1_ax3_0_1_fused * 32 + ax0_ax1_fused // 32)
                                    v1 = Ts.axis.spatial(64, ax4_0_0 * 32 + ax0_ax1_fused % 32)
                                    Ts.reads(p0[v0 // 3136, v0 % 3136 // 56, v0 % 56, v1])
                                    Ts.writes(pad_temp_reindex_shared[v0, v1])
                                    Ts.sblock_attr({"tensor_dim_align":[[0, 0, 32, 16]], "meta_schedule.cooperative_fetch":4})
                                    pad_temp_reindex_shared[v0, v1] = p0[v0 // 3136, v0 % 3136 // 56, v0 % 56, v1]
                            for ax0_ax1_ax2_ax3_fused in T.serial(2048):
                                with Ts.sblock("p1_reindex_shared"):
                                    v0 = Ts.axis.spatial(1, 0)
                                    v1 = Ts.axis.spatial(1, 0)
                                    v2 = Ts.axis.spatial(256, ax2_0_0_ax3_0_0_fused % 4 * 64 + ax0_ax1_ax2_ax3_fused // 32)
                                    v3 = Ts.axis.spatial(64, ax4_0_0 * 32 + ax0_ax1_ax2_ax3_fused % 32)
                                    Ts.reads(p1[v2, v0, v1, v3])
                                    Ts.writes(p1_reindex_shared[v0, v1, v2, v3])
                                    Ts.sblock_attr({"tensor_dim_align":[[0, 2, 32, 16]], "meta_schedule.cooperative_fetch":3})
                                    p1_reindex_shared[v0, v1, v2, v3] = p1[v2, v0, v1, v3]
                            for ax0_1, ax1_1, ax4_0_1 in T.grid(1, 1, 2):
                                for ax0_0_1, ax1_0_1 in T.grid(1, 1):
                                    with Ts.sblock("pad_temp_reindex_shared_wmma.matrix_a_o"):
                                        v0_o = Ts.axis.spatial(3136, ax2_0_0_ax3_0_0_fused // 4 * 392 + ax2_0_1_ax3_0_1_fused * 2 + ax2_0_2_ax3_0_2_fused // 2)
                                        v1_o = Ts.axis.spatial(4, ax4_0_0 * 2 + ax4_0_1)
                                        Ts.reads(pad_temp_reindex_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                        Ts.writes(pad_temp_reindex_shared_wmma_matrix_a[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                        Ts.sblock_attr({"meta_schedule.auto_tensorize":"wmma_load_16x16x16_s8_a_shared"})
                                        for ax0_1_1, ax1_1_1 in T.grid(16, 16):
                                            with Ts.sblock("pad_temp_reindex_shared_wmma.matrix_a"):
                                                v0_i, v1_i = Ts.axis.remap("SS", [ax0_1_1, ax1_1_1])
                                                Ts.reads(pad_temp_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                                Ts.writes(pad_temp_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                                pad_temp_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = pad_temp_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                                for ax0, ax1, ax2_0, ax3_0 in T.grid(1, 1, 2, 1):
                                    with Ts.sblock("p1_reindex_shared_wmma.matrix_b_o"):
                                        v0 = Ts.axis.spatial(1, 0)
                                        v1 = Ts.axis.spatial(1, 0)
                                        v2_o = Ts.axis.spatial(16, ax2_0_0_ax3_0_0_fused % 4 * 4 + ax2_0_2_ax3_0_2_fused % 2 * 2 + ax2_0)
                                        v3_o = Ts.axis.spatial(4, ax4_0_0 * 2 + ax4_0_1)
                                        Ts.reads(p1_reindex_shared[v0, v1, v2_o * 16 : v2_o * 16 + 16, v3_o * 16 : v3_o * 16 + 16])
                                        Ts.writes(p1_reindex_shared_wmma_matrix_b[v0, v1, v2_o * 16 : v2_o * 16 + 16, v3_o * 16 : v3_o * 16 + 16])
                                        Ts.sblock_attr({"meta_schedule.auto_tensorize":"wmma_load_16x16x16_s8_b_trans_shared"})
                                        for ax2_1, ax3_1 in T.grid(16, 16):
                                            with Ts.sblock("p1_reindex_shared_wmma.matrix_b"):
                                                v2_i, v3_i = Ts.axis.remap("SS", [ax2_1, ax3_1])
                                                Ts.reads(p1_reindex_shared[v0, v1, v2_o * 16 + v2_i, v3_o * 16 + v3_i])
                                                Ts.writes(p1_reindex_shared_wmma_matrix_b[v0, v1, v2_o * 16 + v2_i, v3_o * 16 + v3_i])
                                                p1_reindex_shared_wmma_matrix_b[v0, v1, v2_o * 16 + v2_i, v3_o * 16 + v3_i] = p1_reindex_shared[v0, v1, v2_o * 16 + v2_i, v3_o * 16 + v3_i]
                                for ax2_0_3, ax3_0_3, ax0_2, ax1_2, ax4_0_2, ax2_0_4, ax3_0_4 in T.grid(1, 1, 1, 1, 1, 1, 2):
                                    with Ts.sblock("conv2d_nhwc_o"):
                                        v0 = Ts.axis.reduce(1, 0)
                                        v1 = Ts.axis.reduce(1, 0)
                                        v2_o = Ts.axis.spatial(3136, ax2_0_0_ax3_0_0_fused // 4 * 392 + ax2_0_1_ax3_0_1_fused * 2 + ax2_0_2_ax3_0_2_fused // 2 + ax2_0_3 + ax2_0_4)
                                        v3_o = Ts.axis.spatial(16, ax2_0_0_ax3_0_0_fused % 4 * 4 + ax2_0_2_ax3_0_2_fused % 2 * 2 + ax3_0_3 * 2 + ax3_0_4)
                                        v4_o = Ts.axis.reduce(4, ax4_0_0 * 2 + ax4_0_1 + ax4_0_2)
                                        Ts.reads(pad_temp_reindex_shared_wmma_matrix_a[v2_o * 16 : v2_o * 16 + 16, v4_o * 16 : v4_o * 16 + 16], p1_reindex_shared_wmma_matrix_b[v0, v1, v3_o * 16 : v3_o * 16 + 16, v4_o * 16 : v4_o * 16 + 16])
                                        Ts.writes(conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 : v2_o * 16 + 16, v3_o * 16 : v3_o * 16 + 16])
                                        Ts.sblock_attr({"meta_schedule.auto_tensorize":"wmma_sync_16x16x16_s8s8s32_trans", "meta_schedule.auto_tensorize_init":"wmma_fill_16x16x16_s32", "meta_schedule.thread_extent_high_inclusive":1024, "meta_schedule.thread_extent_low_inclusive":32, "warp_execution":1})
                                        with Ts.init():
                                            for ax2_1, ax3_1 in T.grid(16, 16):
                                                with Ts.sblock("conv2d_nhwc_init"):
                                                    v2_i_init, v3_i_init = Ts.axis.remap("SS", [ax2_1, ax3_1])
                                                    Ts.reads()
                                                    Ts.writes(conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 + v2_i_init, v3_o * 16 + v3_i_init])
                                                    conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 + v2_i_init, v3_o * 16 + v3_i_init] = 0
                                        for ax2_1, ax3_1, ax4_1 in T.grid(16, 16, 16):
                                            with Ts.sblock("conv2d_nhwc"):
                                                v2_i, v3_i, v4_i = Ts.axis.remap("SSR", [ax2_1, ax3_1, ax4_1])
                                                Ts.reads(conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 + v2_i, v3_o * 16 + v3_i], pad_temp_reindex_shared_wmma_matrix_a[v2_o * 16 + v2_i, v4_o * 16 + v4_i], p1_reindex_shared_wmma_matrix_b[v0, v1, v3_o * 16 + v3_i, v4_o * 16 + v4_i])
                                                Ts.writes(conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 + v2_i, v3_o * 16 + v3_i])
                                                Ts.sblock_attr({"meta_schedule.tiling_structure":"SSSRRSRS"})
                                                conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 + v2_i, v3_o * 16 + v3_i] = conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 + v2_i, v3_o * 16 + v3_i] + T.cast(pad_temp_reindex_shared_wmma_matrix_a[v2_o * 16 + v2_i, v4_o * 16 + v4_i], ty="int32") * T.cast(p1_reindex_shared_wmma_matrix_b[v0, v1, v3_o * 16 + v3_i, v4_o * 16 + v4_i], ty="int32")
                        for ax0_0, ax1_0 in T.grid(1, 2):
                            with Ts.sblock("conv2d_nhwc_reindex_shared_wmma.accumulator_o"):
                                v0_o = Ts.axis.spatial(3136, ax2_0_0_ax3_0_0_fused // 4 * 392 + ax2_0_1_ax3_0_1_fused * 2 + ax2_0_2_ax3_0_2_fused // 2)
                                v1_o = Ts.axis.spatial(16, ax2_0_0_ax3_0_0_fused % 4 * 4 + ax2_0_2_ax3_0_2_fused % 2 * 2 + ax1_0)
                                Ts.reads(conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                Ts.writes(conv2d_nhwc_reindex_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                Ts.sblock_attr({"meta_schedule.auto_tensorize":"wmma_store_16x16x16_s32_shared"})
                                for ax0_1, ax1_1 in T.grid(16, 16):
                                    with Ts.sblock("conv2d_nhwc_reindex_shared_wmma.accumulator"):
                                        v0_i, v1_i = Ts.axis.remap("SS", [ax0_1, ax1_1])
                                        Ts.reads(conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                        Ts.writes(conv2d_nhwc_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                        conv2d_nhwc_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                    for ax0, ax1_0, ax1_1, ax1_2, ax1_3 in T.grid(32, 1, 4, 32, 2):
                        with Ts.sblock("conv2d_nhwc_reindex_shared"):
                            Ts.where(((ax1_0 * 4 + ax1_1) * 32 + ax1_2) * 2 + ax1_3 < 64)
                            v0 = Ts.axis.spatial(50176, ax2_0_0_ax3_0_0_fused // 4 * 6272 + ax2_0_1_ax3_0_1_fused * 32 + ax0)
                            v1 = Ts.axis.spatial(256, ax2_0_0_ax3_0_0_fused % 4 * 64 + (ax1_0 * 256 + ax1_1 * 64 + ax1_2 * 2 + ax1_3))
                            Ts.reads(p7[()], conv2d_nhwc_reindex_shared[v0, v1], p2[0, 0, 0, v1], p3[0, 0, 0, v1], p4[v1], p5[v1], p6[v1], p8[0])
                            Ts.writes(compute_3[v0 // 3136, v0 % 3136 // 56, v0 % 56, v1])
                            compute_3[v0 // 3136, v0 % 3136 // 56, v0 % 56, v1] = T.Cast("int32", T.shift_right(T.Cast("int64", T.max(T.min(p7[()] + T.Cast("int32", T.shift_right(T.shift_left(T.Cast("int64", conv2d_nhwc_reindex_shared[v0, v1] - p2[0, 0, 0, v1] + p3[0, 0, 0, v1]) * T.Cast("int64", p4[v1]), p5[v1]), (31) + (p6[v1]))), 255), 0) - p8[0]) * T.Cast("int64", 1457846997), (31) - (0)))
            for i0_12, i1_12, i2_12, i3_12 in T.grid(16, 56, 56, 256):
                with Ts.sblock("compute_4"):
                    i0_13, i1_13, i2_13, i3_13 = Ts.axis.remap("SSSS", [i0_12, i1_12, i2_12, i3_12])
                    Ts.reads(compute_3[i0_13, i1_13, i2_13, i3_13], p9[i0_13, i1_13, i2_13, i3_13])
                    Ts.writes(compute[i0_13, i1_13, i2_13, i3_13])
                    compute[i0_13, i1_13, i2_13, i3_13] = T.max(T.min(compute_3[i0_13, i1_13, i2_13, i3_13] + T.Cast("int32", T.shift_right(T.Cast("int64", p9[i0_13, i1_13, i2_13, i3_13]) * T.Cast("int64", 2101000910), (31) - (0))), 255), 0)

@tvm_tirx.script.ir_module
class Conv2dInt8_TensorCore_with_predicate_after:
    @Ts.function
    def main(p0: T.Tensor((16, 56, 56, 64), "int8"), p1: T.Tensor((256, 1, 1, 64), "int8"), p2: T.Tensor((1, 1, 1, 256), "int32"), p3: T.Tensor((1, 1, 1, 256), "int32"), p4: T.Tensor((256,), "int32"), p5: T.Tensor((256,), "int32"), p6: T.Tensor((256,), "int32"), p7: T.Tensor((), "int32"), p8: T.Tensor((1,), "int32"), p9: T.Tensor((16, 56, 56, 256), "int32"), compute: T.Tensor((16, 56, 56, 256), "int32")):
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.unroll_explicit": 1024})
            conv2d_nhwc_reindex_shared = Ts.sblock_alloc_tensor((50176, 256), "int32", scope="shared")
            conv2d_nhwc_reindex_shared_wmma_accumulator = Ts.sblock_alloc_tensor((50176, 256), "int32", scope="wmma.accumulator")
            pad_temp_reindex_shared = Ts.sblock_alloc_tensor((50176, 64), "int8", scope="shared")
            p1_reindex_shared = Ts.sblock_alloc_tensor((1, 1, 256, 64), "int8", scope="shared")
            pad_temp_reindex_shared_wmma_matrix_a = Ts.sblock_alloc_tensor((50176, 64), "int8", scope="wmma.matrix_a")
            p1_reindex_shared_wmma_matrix_b = Ts.sblock_alloc_tensor((1, 1, 256, 64), "int8", scope="wmma.matrix_b")
            for ax2_0_0_ax3_0_0_fused in T.parallel(32, annotations={"thread_binding": "blockIdx.y"}):
                for ax2_0_1_ax3_0_1_fused in T.parallel(196, annotations={"thread_binding": "blockIdx.x"}):
                    for ax2_0_2_ax3_0_2_fused in T.parallel(4, annotations={"thread_binding": "threadIdx.y"}):
                        for ax0_0, ax1_0, ax4_0_0 in T.grid(1, 1, 2):
                            for ax0_ax1_fused in range(1024):
                                with Ts.sblock("pad_temp_reindex_shared"):
                                    v0 = Ts.axis.spatial(50176, ax2_0_0_ax3_0_0_fused // 4 * 6272 + ax2_0_1_ax3_0_1_fused * 32 + ax0_ax1_fused // 32)
                                    v1 = Ts.axis.spatial(64, ax4_0_0 * 32 + ax0_ax1_fused % 32)
                                    Ts.reads(p0[v0 // 3136, v0 % 3136 // 56, v0 % 56, v1])
                                    Ts.writes(pad_temp_reindex_shared[v0, v1])
                                    Ts.sblock_attr({"tensor_dim_align": [[0, 0, 32, 16]], "meta_schedule.cooperative_fetch": 4})
                                    pad_temp_reindex_shared[v0, v1] = p0[v0 // 3136, v0 % 3136 // 56, v0 % 56, v1]
                            for ax0_ax1_ax2_ax3_fused in range(2048):
                                with Ts.sblock("p1_reindex_shared"):
                                    v0 = Ts.axis.spatial(1, 0)
                                    v1 = Ts.axis.spatial(1, 0)
                                    v2 = Ts.axis.spatial(256, ax2_0_0_ax3_0_0_fused % 4 * 64 + ax0_ax1_ax2_ax3_fused // 32)
                                    v3 = Ts.axis.spatial(64, ax4_0_0 * 32 + ax0_ax1_ax2_ax3_fused % 32)
                                    Ts.reads(p1[v2, v0, v1, v3])
                                    Ts.writes(p1_reindex_shared[v0, v1, v2, v3])
                                    Ts.sblock_attr({"tensor_dim_align": [[0, 2, 32, 16]], "meta_schedule.cooperative_fetch": 3})
                                    p1_reindex_shared[v0, v1, v2, v3] = p1[v2, v0, v1, v3]
                            for ax0_1, ax1_1, ax4_0_1 in T.grid(1, 1, 2):
                                for ax0_0_1, ax1_0_1 in T.grid(1, 1):
                                    with Ts.sblock("pad_temp_reindex_shared_wmma.matrix_a_o"):
                                        v0_o = Ts.axis.spatial(3136, ax2_0_0_ax3_0_0_fused // 4 * 392 + ax2_0_1_ax3_0_1_fused * 2 + ax2_0_2_ax3_0_2_fused // 2)
                                        v1_o = Ts.axis.spatial(4, ax4_0_0 * 2 + ax4_0_1)
                                        Ts.reads(pad_temp_reindex_shared[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                        Ts.writes(pad_temp_reindex_shared_wmma_matrix_a[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                        Ts.sblock_attr({"meta_schedule.auto_tensorize": "wmma_load_16x16x16_s8_a_shared"})
                                        for ax0_1_1, ax1_1_1 in T.grid(16, 16):
                                            with Ts.sblock("pad_temp_reindex_shared_wmma.matrix_a"):
                                                v0_i, v1_i = Ts.axis.remap("SS", [ax0_1_1, ax1_1_1])
                                                Ts.reads(pad_temp_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                                Ts.writes(pad_temp_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                                pad_temp_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = pad_temp_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                                for ax0, ax1, ax2_0, ax3_0 in T.grid(1, 1, 2, 1):
                                    with Ts.sblock("p1_reindex_shared_wmma.matrix_b_o"):
                                        v0 = Ts.axis.spatial(1, 0)
                                        v1 = Ts.axis.spatial(1, 0)
                                        v2_o = Ts.axis.spatial(16, ax2_0_0_ax3_0_0_fused % 4 * 4 + ax2_0_2_ax3_0_2_fused % 2 * 2 + ax2_0)
                                        v3_o = Ts.axis.spatial(4, ax4_0_0 * 2 + ax4_0_1)
                                        Ts.reads(p1_reindex_shared[v0, v1, v2_o * 16:v2_o * 16 + 16, v3_o * 16:v3_o * 16 + 16])
                                        Ts.writes(p1_reindex_shared_wmma_matrix_b[v0, v1, v2_o * 16:v2_o * 16 + 16, v3_o * 16:v3_o * 16 + 16])
                                        Ts.sblock_attr({"meta_schedule.auto_tensorize": "wmma_load_16x16x16_s8_b_trans_shared"})
                                        for ax2_1, ax3_1 in T.grid(16, 16):
                                            with Ts.sblock("p1_reindex_shared_wmma.matrix_b"):
                                                v2_i, v3_i = Ts.axis.remap("SS", [ax2_1, ax3_1])
                                                Ts.reads(p1_reindex_shared[v0, v1, v2_o * 16 + v2_i, v3_o * 16 + v3_i])
                                                Ts.writes(p1_reindex_shared_wmma_matrix_b[v0, v1, v2_o * 16 + v2_i, v3_o * 16 + v3_i])
                                                p1_reindex_shared_wmma_matrix_b[v0, v1, v2_o * 16 + v2_i, v3_o * 16 + v3_i] = p1_reindex_shared[v0, v1, v2_o * 16 + v2_i, v3_o * 16 + v3_i]
                                for ax2_0_3, ax3_0_3, ax0_2, ax1_2, ax4_0_2, ax2_0_4, ax3_0_4 in T.grid(1, 1, 1, 1, 1, 1, 2):
                                    with Ts.sblock("conv2d_nhwc_o"):
                                        v0 = Ts.axis.reduce(1, 0)
                                        v1 = Ts.axis.reduce(1, 0)
                                        v2_o = Ts.axis.spatial(3136, ax2_0_0_ax3_0_0_fused // 4 * 392 + ax2_0_1_ax3_0_1_fused * 2 + ax2_0_2_ax3_0_2_fused // 2 + ax2_0_3 + ax2_0_4)
                                        v3_o = Ts.axis.spatial(16, ax2_0_0_ax3_0_0_fused % 4 * 4 + ax2_0_2_ax3_0_2_fused % 2 * 2 + ax3_0_3 * 2 + ax3_0_4)
                                        v4_o = Ts.axis.reduce(4, ax4_0_0 * 2 + ax4_0_1 + ax4_0_2)
                                        Ts.reads(pad_temp_reindex_shared_wmma_matrix_a[v2_o * 16:v2_o * 16 + 16, v4_o * 16:v4_o * 16 + 16], p1_reindex_shared_wmma_matrix_b[v0, v1, v3_o * 16:v3_o * 16 + 16, v4_o * 16:v4_o * 16 + 16])
                                        Ts.writes(conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16:v2_o * 16 + 16, v3_o * 16:v3_o * 16 + 16])
                                        Ts.sblock_attr({"meta_schedule.auto_tensorize": "wmma_sync_16x16x16_s8s8s32_trans", "meta_schedule.auto_tensorize_init": "wmma_fill_16x16x16_s32", "meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "warp_execution": 1})
                                        with Ts.init():
                                            for ax2_1, ax3_1 in T.grid(16, 16):
                                                with Ts.sblock("conv2d_nhwc_init"):
                                                    v2_i_init, v3_i_init = Ts.axis.remap("SS", [ax2_1, ax3_1])
                                                    Ts.reads()
                                                    Ts.writes(conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 + v2_i_init, v3_o * 16 + v3_i_init])
                                                    conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 + v2_i_init, v3_o * 16 + v3_i_init] = 0
                                        for ax2_1, ax3_1, ax4_1 in T.grid(16, 16, 16):
                                            with Ts.sblock("conv2d_nhwc"):
                                                v2_i, v3_i, v4_i = Ts.axis.remap("SSR", [ax2_1, ax3_1, ax4_1])
                                                Ts.reads(conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 + v2_i, v3_o * 16 + v3_i], pad_temp_reindex_shared_wmma_matrix_a[v2_o * 16 + v2_i, v4_o * 16 + v4_i], p1_reindex_shared_wmma_matrix_b[v0, v1, v3_o * 16 + v3_i, v4_o * 16 + v4_i])
                                                Ts.writes(conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 + v2_i, v3_o * 16 + v3_i])
                                                Ts.sblock_attr({"meta_schedule.tiling_structure": "SSSRRSRS"})
                                                conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 + v2_i, v3_o * 16 + v3_i] = conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 + v2_i, v3_o * 16 + v3_i] + T.Cast("int32", pad_temp_reindex_shared_wmma_matrix_a[v2_o * 16 + v2_i, v4_o * 16 + v4_i]) * T.Cast("int32", p1_reindex_shared_wmma_matrix_b[v0, v1, v3_o * 16 + v3_i, v4_o * 16 + v4_i])
                        for ax0_0, ax1_0 in T.grid(1, 2):
                            with Ts.sblock("conv2d_nhwc_reindex_shared_wmma.accumulator_o"):
                                v0_o = Ts.axis.spatial(3136, ax2_0_0_ax3_0_0_fused // 4 * 392 + ax2_0_1_ax3_0_1_fused * 2 + ax2_0_2_ax3_0_2_fused // 2)
                                v1_o = Ts.axis.spatial(16, ax2_0_0_ax3_0_0_fused % 4 * 4 + ax2_0_2_ax3_0_2_fused % 2 * 2 + ax1_0)
                                Ts.reads(conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                Ts.writes(conv2d_nhwc_reindex_shared[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                Ts.sblock_attr({"meta_schedule.auto_tensorize": "wmma_store_16x16x16_s32_shared"})
                                for ax0_1, ax1_1 in T.grid(16, 16):
                                    with Ts.sblock("conv2d_nhwc_reindex_shared_wmma.accumulator"):
                                        v0_i, v1_i = Ts.axis.remap("SS", [ax0_1, ax1_1])
                                        Ts.reads(conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                        Ts.writes(conv2d_nhwc_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                        conv2d_nhwc_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                    for ax0, ax1_0, ax1_1, ax1_2, ax1_3 in T.grid(32, 1, 4, 32, 2):
                        with Ts.sblock("conv2d_nhwc_reindex_shared"):
                            v0 = Ts.axis.spatial(50176, ax2_0_0_ax3_0_0_fused // 4 * 6272 + ax2_0_1_ax3_0_1_fused * 32 + ax0)
                            v1 = Ts.axis.spatial(256, ax2_0_0_ax3_0_0_fused % 4 * 64 + (ax1_0 * 256 + ax1_1 * 64 + ax1_2 * 2 + ax1_3))
                            Ts.where(((ax1_0 * 4 + ax1_1) * 32 + ax1_2) * 2 + ax1_3 < 64)
                            Ts.reads(p7[()], conv2d_nhwc_reindex_shared[v0, v1], p2[0, 0, 0, v1], p3[0, 0, 0, v1], p4[v1], p5[v1], p6[v1], p8[0], p9[v0 // 3136, v0 % 3136 // 56, v0 % 56, v1])
                            Ts.writes(compute[v0 // 3136, v0 % 3136 // 56, v0 % 56, v1])
                            compute[v0 // 3136, v0 % 3136 // 56, v0 % 56, v1] = T.max(T.min(T.Cast("int32", T.shift_right(T.Cast("int64", T.max(T.min(p7[()] + T.Cast("int32", T.shift_right(T.shift_left(T.Cast("int64", conv2d_nhwc_reindex_shared[v0, v1] - p2[0, 0, 0, v1] + p3[0, 0, 0, v1]) * T.Cast("int64", p4[v1]), p5[v1]), (31) + (p6[v1]))), 255), 0) - p8[0]) * T.Cast("int64", 1457846997), (31) - (0))) + T.Cast("int32", T.shift_right(T.Cast("int64", p9[v0 // 3136, v0 % 3136 // 56, v0 % 56, v1]) * T.Cast("int64", 2101000910), (31) - (0))), 255), 0)
# fmt: on

# pylint: enable=no-member,invalid-name,unused-variable


@pytest.fixture(
    params=[pytest.param(False, id="block_obj"), pytest.param(True, id="block_name")],
    scope="session",
)
def use_block_name(request):
    return request.param


def test_compute_inline_elementwise(use_block_name):
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_sblock("B")
    block_c = sch.get_sblock("C")
    sch.compute_inline(block_b)
    assert_structural_equal_ignore_global_symbol(elementwise_inlined, sch.mod["main"])
    assert sch.get(block_c).name_hint == "C"
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_compute_inline_under_loop(use_block_name):
    sch = tvm.s_tir.Schedule(elementwise_under_loop, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_sblock("B")
    block_c = sch.get_sblock("C")
    sch.compute_inline(block_b)
    assert_structural_equal_ignore_global_symbol(elementwise_inlined, sch.mod["main"])
    assert sch.get(block_c).name_hint == "C"
    verify_trace_roundtrip(sch=sch, mod=elementwise_under_loop)


def test_compute_inline_as_dce(use_block_name):
    sch = tvm.s_tir.Schedule(elementwise_standalone, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_sblock("B")
    block_c = sch.get_sblock("C")
    sch.compute_inline(block_b)
    assert_structural_equal_ignore_global_symbol(elementwise_standalone_dce, sch.mod["main"])
    assert sch.get(block_c).name_hint == "C"
    verify_trace_roundtrip(sch=sch, mod=elementwise_standalone)


def test_compute_inline_multi_consumer(use_block_name):
    sch = tvm.s_tir.Schedule(elementwise_multi_producer_consumer, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_sblock("B")
    block_c = sch.get_sblock("C")
    block_d = sch.get_sblock("D")
    sch.compute_inline(block_b)
    assert_structural_equal_ignore_global_symbol(
        elementwise_multi_consumer_inlined, sch.mod["main"]
    )
    assert sch.get(block_c).name_hint == "C"
    assert sch.get(block_d).name_hint == "D"
    verify_trace_roundtrip(sch=sch, mod=elementwise_multi_producer_consumer)


def test_compute_inline_fail_multi_writer(use_block_name):
    sch = tvm.s_tir.Schedule(fail_multi_reader_writer, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_sblock("B")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.compute_inline(block_b)


def test_reverse_compute_inline_elementwise(use_block_name):
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_sblock("B")
    block_c = "C" if use_block_name else sch.get_sblock("C")
    sch.reverse_compute_inline(block_c)
    assert_structural_equal_ignore_global_symbol(elementwise_inlined, sch.mod["main"])
    assert sch.get(block_b).name_hint == "B"
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_reverse_compute_inline_under_loop(use_block_name):
    sch = tvm.s_tir.Schedule(elementwise_under_loop, debug_mask="all")
    block_b = sch.get_sblock("B")
    block_c = "C" if use_block_name else sch.get_sblock("C")
    sch.reverse_compute_inline(block_c)
    assert_structural_equal_ignore_global_symbol(elementwise_inlined, sch.mod["main"])
    assert sch.get(block_b).name_hint == "B"
    verify_trace_roundtrip(sch=sch, mod=elementwise_under_loop)


def test_reverse_compute_inline_fail_as_dce(use_block_name):
    sch = tvm.s_tir.Schedule(elementwise_standalone, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_sblock("B")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.reverse_compute_inline(block_b)


def test_reverse_compute_inline_fail_multi_producer(use_block_name):
    sch = tvm.s_tir.Schedule(elementwise_multi_producer_consumer, debug_mask="all")
    block_d = "D" if use_block_name else sch.get_sblock("D")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.reverse_compute_inline(block_d)


def test_reverse_compute_inline_fail_multi_reader(use_block_name):
    sch = tvm.s_tir.Schedule(fail_multi_reader_writer, debug_mask="all")
    block_c = "C" if use_block_name else sch.get_sblock("C")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.reverse_compute_inline(block_c)


def test_reverse_compute_multi_reverse_loads(use_block_name):
    sch = tvm.s_tir.Schedule(elementwise_multi_reverse_loads, debug_mask="all")
    block_c = "C" if use_block_name else sch.get_sblock("C")
    sch.reverse_compute_inline(block_c)
    assert_structural_equal_ignore_global_symbol(
        elementwise_multi_reverse_loads_inlined, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=elementwise_multi_reverse_loads)


def test_reverse_compute_inline_affine_load(use_block_name):
    sch = tvm.s_tir.Schedule(elementwise_reverse_affine_load, debug_mask="all")
    block_c = "C" if use_block_name else sch.get_sblock("C")
    sch.reverse_compute_inline(block_c)
    assert_structural_equal_ignore_global_symbol(
        elementwise_reverse_affine_load_inlined, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=elementwise_reverse_affine_load)


def test_reverse_compute_inline_multi_affine_load(use_block_name):
    sch = tvm.s_tir.Schedule(elementwise_multi_reverse_affine_load, debug_mask="all")
    block_c = "C" if use_block_name else sch.get_sblock("C")
    sch.reverse_compute_inline(block_c)
    assert_structural_equal_ignore_global_symbol(
        elementwise_multi_reverse_affine_load_inlined, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=elementwise_multi_reverse_affine_load)


def test_reverse_compute_inline_affine_load_unit_iter(use_block_name):
    sch = tvm.s_tir.Schedule(elementwise_reverse_affine_load_unit_iter, debug_mask="all")
    block_c = "C" if use_block_name else sch.get_sblock("C")
    sch.reverse_compute_inline(block_c)
    assert_structural_equal_ignore_global_symbol(
        elementwise_reverse_affine_load_unit_iter_inlined, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=elementwise_reverse_affine_load_unit_iter)


def test_reverse_compute_inline_affine_load_unit_iter_simplified(use_block_name):
    sch = tvm.s_tir.Schedule(elementwise_reverse_affine_load_unit_iter_simplified, debug_mask="all")
    block_c = "C" if use_block_name else sch.get_sblock("C")
    sch.reverse_compute_inline(block_c)
    assert_structural_equal_ignore_global_symbol(
        elementwise_reverse_affine_load_unit_iter_simplified_inlined, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=elementwise_reverse_affine_load_unit_iter_simplified)


@pytest.mark.parametrize("reverse_order", [True, False])
def test_reverse_compute_inline_affine_chain(use_block_name, reverse_order):
    sch = tvm.s_tir.Schedule(elementwise_reverse_affine_chain, debug_mask="all")
    block_c = "C" if use_block_name else sch.get_sblock("C")
    block_d = "D" if use_block_name else sch.get_sblock("D")
    if reverse_order:
        sch.reverse_compute_inline(block_d)
        sch.reverse_compute_inline(block_c)
    else:
        sch.reverse_compute_inline(block_c)
        sch.reverse_compute_inline(block_d)
    assert_structural_equal_ignore_global_symbol(
        elementwise_reverse_affine_chain_inlined, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=elementwise_reverse_affine_chain)


def test_reverse_compute_fail_non_affine_load(use_block_name):
    sch = tvm.s_tir.Schedule(elementwise_reverse_non_affine_load, debug_mask="all")
    block_c = "C" if use_block_name else sch.get_sblock("C")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.reverse_compute_inline(block_c)


def test_reverse_compute_fail_multi_reverse_loads(use_block_name):
    sch = tvm.s_tir.Schedule(elementwise_multi_loads, debug_mask="all")
    block_c = "C" if use_block_name else sch.get_sblock("C")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.reverse_compute_inline(block_c)


def test_opaque_access_load(use_block_name):
    sch = tvm.s_tir.Schedule(opaque_access_load, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_sblock("B")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.compute_inline(block_b)


def test_opaque_access_store(use_block_name):
    sch = tvm.s_tir.Schedule(opaque_access_store, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_sblock("B")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.compute_inline(block_b)


def test_tensor_matched(use_block_name):
    sch = tvm.s_tir.Schedule(tensor_matched, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_sblock("B")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.compute_inline(block_b)


def test_output_block(use_block_name):
    sch = tvm.s_tir.Schedule(matmul_relu, debug_mask="all")
    block = sch.get_sblock("compute")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.compute_inline(block)

    sch = tvm.s_tir.Schedule(elementwise_output, debug_mask="all")
    block = sch.get_sblock("B")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.compute_inline(block)

    block = sch.get_sblock("C")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.reverse_compute_inline(block)


def test_compute_inline_predicate(use_block_name):
    sch = tvm.s_tir.Schedule(elementwise_predicate, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_sblock("B")
    sch.compute_inline(block_b)
    assert_structural_equal_ignore_global_symbol(elementwise_predicate_inlined, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise_predicate)


def test_compute_inline_multi_loads(use_block_name):
    sch = tvm.s_tir.Schedule(elementwise_multi_loads, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_sblock("B")
    sch.compute_inline(block_b)
    assert_structural_equal_ignore_global_symbol(elementwise_multi_loads_inlined, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise_multi_loads)


def test_compute_inline_with_opaque_access(use_block_name):
    """Test not rewrite opaque reads/writes after irrelavant compute inline"""
    sch = tvm.s_tir.Schedule(access_opaque_ptr_then_elemwise, debug_mask="all")
    BB = "BB" if use_block_name else sch.get_sblock("BB")
    sch.compute_inline(BB)
    assert_structural_equal_ignore_global_symbol(
        access_opaque_ptr_then_elemwise_inline, sch.mod["main"]
    )


def test_inline_block_with_init():
    sch = tvm.s_tir.Schedule(inline_block_with_init, debug_mask="all")
    block = sch.get_sblock(name="tensor_rf", func_name="main")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.compute_inline(block=block)


def test_compute_inline_opaque_access_with_pointer(use_block_name):
    """Test opaque access with pointer after compute inline"""
    sch = tvm.s_tir.Schedule(exp_exp_opaque_access_with_pointer, debug_mask="all")
    compute = "compute" if use_block_name else sch.get_sblock("compute")
    sch.compute_inline(compute)
    assert_structural_equal_ignore_global_symbol(
        exp_exp_opaque_access_with_pointer_inlined, sch.mod["main"]
    )


def test_reverse_compute_inline_overcomputed_producer(use_block_name):
    """Test reverse compute inline overcomputed producer"""
    sch = tvm.s_tir.Schedule(elementwise_overcomputed_producer, debug_mask="all")
    compute = "C" if use_block_name else sch.get_sblock("C")
    sch.reverse_compute_inline(compute)
    assert_structural_equal_ignore_global_symbol(
        elementwise_overcomputed_producer_reverse_inlined, sch.mod["main"]
    )


def test_reverse_compute_inline_overcomputed_producer_simplify_predicate(use_block_name):
    """Test reverse compute inline overcomputed producer where the predicate should be simplified"""
    sch = tvm.s_tir.Schedule(elementwise_overcomputed_producer_simplify_predicate, debug_mask="all")
    compute = "C" if use_block_name else sch.get_sblock("C")
    sch.reverse_compute_inline(compute)
    assert_structural_equal_ignore_global_symbol(
        elementwise_overcomputed_producer_simplify_predicate_reverse_inlined, sch.mod["main"]
    )


def test_reverse_compute_inline_overcomputed_producer_injective_load(use_block_name):
    """Test reverse compute inline overcomputed producer with injective tensor load"""
    sch = tvm.s_tir.Schedule(elementwise_overcomputed_producer_injective_load, debug_mask="all")
    compute = "C" if use_block_name else sch.get_sblock("C")
    sch.reverse_compute_inline(compute)
    assert_structural_equal_ignore_global_symbol(
        elementwise_overcomputed_producer_injective_load_reverse_inlined, sch.mod["main"]
    )


def test_reverse_compute_inline_error_producer_not_cover_consumer(use_block_name):
    """Test reverse compute inline failure when the inlined block iter domains are not covered by
    its producer
    """
    sch = tvm.s_tir.Schedule(elementwise_producer_not_cover_consumer, debug_mask="all")
    compute = "C" if use_block_name else sch.get_sblock("C")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.reverse_compute_inline(compute)


def test_reverse_compute_inline_producer_predicate_allowed():
    """Test a case where reverse compute inline is allowed even though the producer has a
    non-trivial predicate.
    """

    sch = tvm.s_tir.Schedule(elementwise_predicate_producer, debug_mask="all")
    sch.reverse_compute_inline(sch.get_sblock("C"))
    assert_structural_equal_ignore_global_symbol(
        elementwise_predicate_producer_inlined, sch.mod["main"]
    )


def test_reverse_compute_inline_producer_predicate_disallowed():
    """Test reverse compute inline failure when the producer has a non-trivial predicate that cannot be
    implied by the synthesized predicate of the new inlined block.
    """

    sch = tvm.s_tir.Schedule(Conv2dInt8_TensorCore_with_predicate_before, debug_mask="all")
    sch.reverse_compute_inline(sch.get_sblock("compute_4"))
    assert_structural_equal_ignore_global_symbol(
        Conv2dInt8_TensorCore_with_predicate_after["main"], sch.mod["main"]
    )


def test_reverse_compute_inline_producer_is_reduction():
    """Test reverse comput inline when producer is reduction"""
    sch = tvm.s_tir.Schedule(elementwise_producer_is_reduction, debug_mask="all")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.reverse_compute_inline(sch.get_sblock("C"))


def test_compute_inline_softmax():
    # fmt: off
    n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int64")

    @Ts.function
    def before(lv44: T.Tensor((T.int64(1), T.int64(32), n, m), "float32"), var_compute_intermediate: T.Tensor((T.int64(1), T.int64(32), n, m), 'float16')):
        T.func_attr({"tirx.noalias": True})

        T_softmax_maxelem = Ts.sblock_alloc_tensor((T.int64(1), T.int64(32), n))
        T_softmax_exp = Ts.sblock_alloc_tensor((T.int64(1), T.int64(32), n, m))
        T_softmax_expsum = Ts.sblock_alloc_tensor((T.int64(1), T.int64(32), n))
        var_T_softmax_norm_intermediate = Ts.sblock_alloc_tensor((T.int64(1), T.int64(32), n, m))
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), n, m):
            with Ts.sblock("T_softmax_maxelem"):
                v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                Ts.reads(lv44[v_i0, v_i1, v_i2, v_k])
                Ts.writes(T_softmax_maxelem[v_i0, v_i1, v_i2])
                with Ts.init():
                    T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float32(-3.4028234663852886e+38)
                T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(T_softmax_maxelem[v_i0, v_i1, v_i2], lv44[v_i0, v_i1, v_i2, v_k])
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), n, m):
            with Ts.sblock("T_softmax_exp"):
                v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(lv44[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem[v_i0, v_i1, v_i2])
                Ts.writes(T_softmax_exp[v_i0, v_i1, v_i2, v_i3])
                T_softmax_exp[v_i0, v_i1, v_i2, v_i3] = T.exp(lv44[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2])
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), n, m):
            with Ts.sblock("T_softmax_expsum"):
                v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                Ts.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_k])
                Ts.writes(T_softmax_expsum[v_i0, v_i1, v_i2])
                with Ts.init():
                    T_softmax_expsum[v_i0, v_i1, v_i2] = T.float32(0)
                T_softmax_expsum[v_i0, v_i1, v_i2] = T_softmax_expsum[v_i0, v_i1, v_i2] + T_softmax_exp[v_i0, v_i1, v_i2, v_k]
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), n, m):
            with Ts.sblock("T_softmax_norm"):
                v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_i3], T_softmax_expsum[v_i0, v_i1, v_i2])
                Ts.writes(var_T_softmax_norm_intermediate[v_i0, v_i1, v_i2, v_i3])
                Ts.sblock_attr({"axis": 3})
                var_T_softmax_norm_intermediate[v_i0, v_i1, v_i2, v_i3] = T_softmax_exp[v_i0, v_i1, v_i2, v_i3] / T_softmax_expsum[v_i0, v_i1, v_i2]
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), n, m):
            with Ts.sblock("compute"):
                v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(var_T_softmax_norm_intermediate[v_i0, v_i1, v_i2, v_i3])
                Ts.writes(var_compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                var_compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.Cast("float16", var_T_softmax_norm_intermediate[v_i0, v_i1, v_i2, v_i3])

    n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int64")

    @Ts.function
    def after(lv44: T.Tensor((T.int64(1), T.int64(32), n, m), "float32"), var_compute_intermediate: T.Tensor((T.int64(1), T.int64(32), n, m), 'float16')):
        T.func_attr({"tirx.noalias": True})

        # with Ts.sblock("root"):
        T_softmax_maxelem = Ts.sblock_alloc_tensor((T.int64(1), T.int64(32), n))
        T_softmax_expsum = Ts.sblock_alloc_tensor((T.int64(1), T.int64(32), n))
        var_T_softmax_norm_intermediate = Ts.sblock_alloc_tensor((T.int64(1), T.int64(32), n, m))
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), n, m):
            with Ts.sblock("T_softmax_maxelem"):
                v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                Ts.reads(lv44[v_i0, v_i1, v_i2, v_k])
                Ts.writes(T_softmax_maxelem[v_i0, v_i1, v_i2])
                with Ts.init():
                    T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float32(-3.4028234663852886e+38)
                T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(T_softmax_maxelem[v_i0, v_i1, v_i2], lv44[v_i0, v_i1, v_i2, v_k])
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), n, m):
            with Ts.sblock("T_softmax_expsum"):
                v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                Ts.reads(lv44[v_i0, v_i1, v_i2, v_k], T_softmax_maxelem[v_i0, v_i1, v_i2])
                Ts.writes(T_softmax_expsum[v_i0, v_i1, v_i2])
                with Ts.init():
                    T_softmax_expsum[v_i0, v_i1, v_i2] = T.float32(0)
                T_softmax_expsum[v_i0, v_i1, v_i2] = T_softmax_expsum[v_i0, v_i1, v_i2] + T.exp(lv44[v_i0, v_i1, v_i2, v_k] - T_softmax_maxelem[v_i0, v_i1, v_i2])
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), n, m):
            with Ts.sblock("T_softmax_norm"):
                v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(lv44[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem[v_i0, v_i1, v_i2], T_softmax_expsum[v_i0, v_i1, v_i2])
                Ts.writes(var_T_softmax_norm_intermediate[v_i0, v_i1, v_i2, v_i3])
                Ts.sblock_attr({"axis": 3})
                var_T_softmax_norm_intermediate[v_i0, v_i1, v_i2, v_i3] = T.exp(lv44[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2]) / T_softmax_expsum[v_i0, v_i1, v_i2]
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), n, m):
            with Ts.sblock("compute"):
                v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(var_T_softmax_norm_intermediate[v_i0, v_i1, v_i2, v_i3])
                Ts.writes(var_compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                var_compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.Cast("float16", var_T_softmax_norm_intermediate[v_i0, v_i1, v_i2, v_i3])
    # fmt: on

    sch = tvm.s_tir.Schedule(before)
    sch.compute_inline(sch.get_sblock("T_softmax_exp"))
    assert_structural_equal_ignore_global_symbol(after, sch.mod["main"])


def test_reverse_compute_inline_layer_norm():
    # fmt: off
    n = T.dynamic("n", "int64")

    @Ts.function
    def before(lv6: T.Tensor((T.int64(1), n, T.int64(2560)), "float32"), weight1: T.Tensor((T.int64(2560),), "float32"), bias: T.Tensor((T.int64(2560),), "float32"), var_compute_intermediate: T.Tensor((T.int64(1), n, T.int64(2560)), 'float16')):
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})

        A_red_temp_v0_shared = Ts.sblock_alloc_tensor((T.int64(1), n), scope="shared")
        A_red_temp_v1_shared = Ts.sblock_alloc_tensor((T.int64(1), n), scope="shared")
        var_T_layer_norm_intermediate = Ts.sblock_alloc_tensor((T.int64(1), n, T.int64(2560)))
        for ax0_ax1_fused in T.parallel(n, annotations={"thread_binding": "blockIdx.x", "tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(10)):
                for ax2_1 in T.parallel(T.int64(256), annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("A_red_temp"):
                        v_ax0 = Ts.axis.spatial(T.int64(1), ax0)
                        v_ax1 = Ts.axis.spatial(n, ax0_ax1_fused + ax1)
                        v_k2 = Ts.axis.reduce(T.int64(2560), ax2_0 * T.int64(256) + ax2_1)
                        Ts.reads(lv6[v_ax0, v_ax1, v_k2])
                        Ts.writes(A_red_temp_v0_shared[v_ax0, v_ax1], A_red_temp_v1_shared[v_ax0, v_ax1])
                        with Ts.init():
                            A_red_temp_v0_shared[v_ax0, v_ax1] = T.float32(0)
                            A_red_temp_v1_shared[v_ax0, v_ax1] = T.float32(0)
                        v_A_red_temp_v0: T.let[T.float32] = A_red_temp_v0_shared[v_ax0, v_ax1] + lv6[v_ax0, v_ax1, v_k2]
                        v_A_red_temp_v1: T.let[T.float32] = A_red_temp_v1_shared[v_ax0, v_ax1] + lv6[v_ax0, v_ax1, v_k2] * lv6[v_ax0, v_ax1, v_k2]
                        A_red_temp_v0_shared[v_ax0, v_ax1] = v_A_red_temp_v0
                        A_red_temp_v1_shared[v_ax0, v_ax1] = v_A_red_temp_v1
            for ax2_0 in range(T.int64(10)):
                for ax2_1 in T.parallel(T.int64(256), annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("T_layer_norm"):
                        v_ax0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = Ts.axis.spatial(n, ax0_ax1_fused)
                        v_ax2 = Ts.axis.spatial(T.int64(2560), ax2_0 * T.int64(256) + ax2_1)
                        Ts.reads(lv6[v_ax0, v_ax1, v_ax2], A_red_temp_v0_shared[v_ax0, v_ax1], A_red_temp_v1_shared[v_ax0, v_ax1], weight1[v_ax2], bias[v_ax2])
                        Ts.writes(var_T_layer_norm_intermediate[v_ax0, v_ax1, v_ax2])
                        var_T_layer_norm_intermediate[v_ax0, v_ax1, v_ax2] = (lv6[v_ax0, v_ax1, v_ax2] - A_red_temp_v0_shared[v_ax0, v_ax1] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[v_ax0, v_ax1] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[v_ax0, v_ax1] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[v_ax0, v_ax1] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * weight1[v_ax2] + bias[v_ax2]
        for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(2560)):
            with Ts.sblock("compute"):
                v_i0, v_i1, v_i2 = Ts.axis.remap("SSS", [i0, i1, i2])
                Ts.reads(var_T_layer_norm_intermediate[v_i0, v_i1, v_i2])
                Ts.writes(var_compute_intermediate[v_i0, v_i1, v_i2])
                var_compute_intermediate[v_i0, v_i1, v_i2] = T.Cast("float16", var_T_layer_norm_intermediate[v_i0, v_i1, v_i2])

    n = T.dynamic("n", "int64")

    @Ts.function
    def after(lv6: T.Tensor((T.int64(1), n, T.int64(2560)), "float32"), weight1: T.Tensor((T.int64(2560),), "float32"), bias: T.Tensor((T.int64(2560),), "float32"), var_compute_intermediate: T.Tensor((T.int64(1), n, T.int64(2560)), 'float16')):
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})

        # with Ts.sblock("root"):
        A_red_temp_v0_shared = Ts.sblock_alloc_tensor((T.int64(1), n), scope="shared")
        A_red_temp_v1_shared = Ts.sblock_alloc_tensor((T.int64(1), n), scope="shared")
        for ax0_ax1_fused in T.parallel(n, annotations={"thread_binding": "blockIdx.x", "tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
            for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(10)):
                for ax2_1 in T.parallel(T.int64(256), annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("A_red_temp"):
                        v_ax0 = Ts.axis.spatial(T.int64(1), ax0)
                        v_ax1 = Ts.axis.spatial(n, ax0_ax1_fused + ax1)
                        v_k2 = Ts.axis.reduce(T.int64(2560), ax2_0 * T.int64(256) + ax2_1)
                        Ts.reads(lv6[v_ax0, v_ax1, v_k2])
                        Ts.writes(A_red_temp_v0_shared[v_ax0, v_ax1], A_red_temp_v1_shared[v_ax0, v_ax1])
                        with Ts.init():
                            A_red_temp_v0_shared[v_ax0, v_ax1] = T.float32(0)
                            A_red_temp_v1_shared[v_ax0, v_ax1] = T.float32(0)
                        v_A_red_temp_v0: T.let[T.float32] = A_red_temp_v0_shared[v_ax0, v_ax1] + lv6[v_ax0, v_ax1, v_k2]
                        v_A_red_temp_v1: T.let[T.float32] = A_red_temp_v1_shared[v_ax0, v_ax1] + lv6[v_ax0, v_ax1, v_k2] * lv6[v_ax0, v_ax1, v_k2]
                        A_red_temp_v0_shared[v_ax0, v_ax1] = v_A_red_temp_v0
                        A_red_temp_v1_shared[v_ax0, v_ax1] = v_A_red_temp_v1
            for ax2_0 in range(T.int64(10)):
                for ax2_1 in T.parallel(T.int64(256), annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("T_layer_norm"):
                        v_ax0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = Ts.axis.spatial(n, ax0_ax1_fused)
                        v_ax2 = Ts.axis.spatial(T.int64(2560), ax2_0 * T.int64(256) + ax2_1)
                        Ts.reads(lv6[v_ax0, v_ax1, v_ax2], A_red_temp_v0_shared[v_ax0, v_ax1], A_red_temp_v1_shared[v_ax0, v_ax1], weight1[v_ax2], bias[v_ax2])
                        Ts.writes(var_compute_intermediate[v_ax0, v_ax1, v_ax2])
                        var_compute_intermediate[v_ax0, v_ax1, v_ax2] = T.Cast("float16", (lv6[v_ax0, v_ax1, v_ax2] - A_red_temp_v0_shared[v_ax0, v_ax1] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[v_ax0, v_ax1] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[v_ax0, v_ax1] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[v_ax0, v_ax1] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * weight1[v_ax2] + bias[v_ax2])
    # fmt: on

    sch = tvm.s_tir.Schedule(before)
    sch.reverse_compute_inline(sch.get_sblock("compute"))
    assert_structural_equal_ignore_global_symbol(after, sch.mod["main"])


def test_reverse_compute_inline_slicing_then_cachewrite():
    @Ts.function
    def before(
        x: T.Tensor((1, 16, 7, 7), "float32"),
        T_strided_slice_with_axes: T.Tensor((1, 12, 7, 7), "float32"),
    ):
        T_add = Ts.sblock_alloc_tensor((1, 16, 7, 7))
        for ax0, ax1, ax2, ax3 in T.grid(1, 16, 7, 7):
            with Ts.sblock("T_add"):
                v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = x[v_ax0, v_ax1, v_ax2, v_ax3] + T.float32(1)
        for ax0, ax1, ax2, ax3 in T.grid(1, 12, 7, 7):
            with Ts.sblock("T_strided_slice_with_axes"):
                v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2, v_ax3] = T_add[
                    v_ax0, v_ax1, v_ax2, v_ax3
                ]

    @Ts.function
    def after(
        x: T.Tensor((1, 16, 7, 7), "float32"),
        T_strided_slice_with_axes: T.Tensor((1, 12, 7, 7), "float32"),
    ):
        T_strided_slice_with_axes_global = Ts.sblock_alloc_tensor((1, 12, 7, 7))
        for ax0, ax1, ax2, ax3 in T.grid(1, 16, 7, 7):
            with Ts.sblock("T_add"):
                v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                Ts.where(ax1 < 12)
                T_strided_slice_with_axes_global[v_ax0, v_ax1, v_ax2, v_ax3] = x[
                    v_ax0, v_ax1, v_ax2, v_ax3
                ] + T.float32(1)
        for ax0, ax1, ax2, ax3 in T.grid(1, 12, 7, 7):
            with Ts.sblock("T_strided_slice_with_axes_global"):
                v0, v1, v2, v3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T_strided_slice_with_axes[v0, v1, v2, v3] = T_strided_slice_with_axes_global[
                    v0, v1, v2, v3
                ]

    sch = tvm.s_tir.Schedule(before)
    sch.reverse_compute_inline(sch.get_sblock("T_strided_slice_with_axes"))
    sch.cache_write(sch.get_sblock("T_add"), 0, "global")
    assert_structural_equal_ignore_global_symbol(after, sch.mod["main"])


def test_inline_with_reduction():
    @Ts.function
    def before(
        T_softmax_norm: T.Tensor((T.int64(6), T.int64(1), T.int64(1)), "float32"),
        T_reshape_2: T.Tensor((T.int64(6), T.int64(1), T.int64(64)), "float32"),
        T_transpose: T.Tensor((T.int64(1), T.int64(1), T.int64(6), T.int64(64)), "float32"),
    ):
        T_batch_matmul_NN = Ts.sblock_alloc_tensor((T.int64(6), T.int64(1), T.int64(64)))
        for ax0, ax1 in T.grid(T.int64(6), T.int64(64)):
            with Ts.sblock("bmm"):
                v0, v1 = Ts.axis.remap("SS", [ax0, ax1])
                Ts.reads(
                    T_softmax_norm[v0, T.int64(0), T.int64(0)], T_reshape_2[v0, T.int64(0), v1]
                )
                Ts.writes(T_batch_matmul_NN[v0, T.int64(0), v1])
                with Ts.init():
                    T_batch_matmul_NN[v0, T.int64(0), v1] = T.float32(0.0)
                T_batch_matmul_NN[v0, T.int64(0), v1] = (
                    T_batch_matmul_NN[v0, T.int64(0), v1]
                    + T_softmax_norm[v0, T.int64(0), T.int64(0)] * T_reshape_2[v0, T.int64(0), v1]
                )
        for ax0, ax1 in T.grid(T.int64(6), T.int64(64)):
            with Ts.sblock("transpose"):
                v0, v1 = Ts.axis.remap("SS", [ax0, ax1])
                Ts.reads(T_batch_matmul_NN[v0, T.int64(0), v1])
                Ts.writes(T_transpose[T.int64(0), T.int64(0), v0, v1])
                T_transpose[T.int64(0), T.int64(0), v0, v1] = T_batch_matmul_NN[v0, T.int64(0), v1]

    @Ts.function
    def after(
        T_softmax_norm: T.Tensor((T.int64(6), T.int64(1), T.int64(1)), "float32"),
        T_reshape_2: T.Tensor((T.int64(6), T.int64(1), T.int64(64)), "float32"),
        T_transpose: T.Tensor((T.int64(1), T.int64(1), T.int64(6), T.int64(64)), "float32"),
    ):
        for ax0, ax1 in T.grid(T.int64(6), T.int64(64)):
            with Ts.sblock("bmm"):
                v0, v1 = Ts.axis.remap("SS", [ax0, ax1])
                Ts.reads(
                    T_softmax_norm[v0, T.int64(0), T.int64(0)], T_reshape_2[v0, T.int64(0), v1]
                )
                Ts.writes(T_transpose[T.int64(0), T.int64(0), v0, v1])
                with Ts.init():
                    T_transpose[T.int64(0), T.int64(0), v0, v1] = T.float32(0.0)
                T_transpose[T.int64(0), T.int64(0), v0, v1] = (
                    T_transpose[T.int64(0), T.int64(0), v0, v1]
                    + T_softmax_norm[v0, T.int64(0), T.int64(0)] * T_reshape_2[v0, T.int64(0), v1]
                )

    sch = tvm.s_tir.Schedule(before)
    sch.reverse_compute_inline(sch.get_sblock("transpose"))
    assert_structural_equal_ignore_global_symbol(after, sch.mod["main"])


if __name__ == "__main__":
    tvm_tirx.testing.main()
