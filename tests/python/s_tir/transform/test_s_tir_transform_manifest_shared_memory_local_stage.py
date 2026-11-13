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
import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import s_tir
from tvm.s_tir import script as Ts
from tvm_tirx.script import tirx as T

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks


@tvm_tirx.script.ir_module
class MatmulBefore:
    @Ts.function
    def main(A: T.Tensor((1024, 1024), "float32"), B: T.Tensor((1024, 1024), "float32"), C: T.Tensor((1024, 1024), "float32")) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "default_function", "tirx.noalias": True})
        # body
        # with Ts.sblock("root")
        for blockIdx_y in T.parallel(32, annotations={"thread_binding": "blockIdx.y"}):
            for blockIdx_x in T.parallel(32, annotations={"thread_binding": "blockIdx.x"}):
                for threadIdx_y in T.parallel(2, annotations={"thread_binding": "threadIdx.y"}):
                    for threadIdx_x in T.parallel(2, annotations={"thread_binding": "threadIdx.x"}):
                        for k_0 in T.serial(32):
                            with Ts.sblock():
                                Ts.reads(A[blockIdx_y * 32 : blockIdx_y * 32 + 32, k_0 * 32 : k_0 * 32 + 32], B[k_0 * 32 : k_0 * 32 + 32, blockIdx_x * 32 : blockIdx_x * 32 + 32])
                                Ts.writes(C[blockIdx_y * 32 : blockIdx_y * 32 + 32, blockIdx_x * 32 : blockIdx_x * 32 + 32])
                                A_shared = Ts.sblock_alloc_tensor([1024, 1024], dtype="float32", scope="shared")
                                B_shared = Ts.sblock_alloc_tensor([1024, 1024], dtype="float32", scope="shared")
                                for ax0_ax1_fused_0 in T.serial(64):
                                    for ax0_ax1_fused_3 in T.vectorized(4):
                                        with Ts.sblock("A_shared"):
                                            Ts.reads(A[blockIdx_y * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) // 32, k_0 * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) % 32])
                                            Ts.writes(A_shared[blockIdx_y * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) // 32, k_0 * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) % 32])
                                            Ts.sblock_attr({"tirx.manifest_shared_memory_local_stage":1})
                                            A_shared[blockIdx_y * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) // 32, k_0 * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) % 32] = A[blockIdx_y * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) // 32, k_0 * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) % 32]
                                for ax0_ax1_fused_0 in T.serial(64):
                                    for ax0_ax1_fused_3 in T.vectorized(4):
                                        with Ts.sblock("B_shared"):
                                            Ts.reads(B[k_0 * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) // 32, blockIdx_x * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) % 32])
                                            Ts.writes(B_shared[k_0 * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) // 32, blockIdx_x * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) % 32])
                                            Ts.sblock_attr({"tirx.manifest_shared_memory_local_stage":1})
                                            B_shared[k_0 * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) // 32, blockIdx_x * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) % 32] = B[k_0 * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) // 32, blockIdx_x * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) % 32]
                                for k_1, i_2, j_2, k_2 in T.grid(2, 16, 16, 16):
                                    with Ts.sblock("C"):
                                        Ts.reads(A_shared[blockIdx_y * 32 + threadIdx_y * 16 + i_2, k_0 * 32 + k_1 * 16 + k_2], B_shared[k_0 * 32 + k_1 * 16 + k_2, blockIdx_x * 32 + threadIdx_x * 16 + j_2])
                                        Ts.writes(C[blockIdx_y * 32 + threadIdx_y * 16 + i_2, blockIdx_x * 32 + threadIdx_x * 16 + j_2])
                                        if k_0 * 32 + k_1 * 16 + k_2 == 0:
                                            C[blockIdx_y * 32 + threadIdx_y * 16 + i_2, blockIdx_x * 32 + threadIdx_x * 16 + j_2] = T.float32(0)
                                        C[blockIdx_y * 32 + threadIdx_y * 16 + i_2, blockIdx_x * 32 + threadIdx_x * 16 + j_2] = C[blockIdx_y * 32 + threadIdx_y * 16 + i_2, blockIdx_x * 32 + threadIdx_x * 16 + j_2] + A_shared[blockIdx_y * 32 + threadIdx_y * 16 + i_2, k_0 * 32 + k_1 * 16 + k_2] * B_shared[k_0 * 32 + k_1 * 16 + k_2, blockIdx_x * 32 + threadIdx_x * 16 + j_2]


@tvm_tirx.script.ir_module
class MatmulAfter:
    @Ts.function
    def main(A: T.Tensor((1024, 1024), "float32"), B: T.Tensor((1024, 1024), "float32"), C: T.Tensor((1024, 1024), "float32")) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "default_function", "tirx.noalias": True})
        # body
        # with Ts.sblock("root")
        for blockIdx_y in T.parallel(32, annotations={"thread_binding": "blockIdx.y"}):
            for blockIdx_x in T.parallel(32, annotations={"thread_binding": "blockIdx.x"}):
                for threadIdx_y in T.parallel(2, annotations={"thread_binding": "threadIdx.y"}):
                    for threadIdx_x in T.parallel(2, annotations={"thread_binding": "threadIdx.x"}):
                        for k_0 in T.serial(32):
                            with Ts.sblock():
                                Ts.reads(A[blockIdx_y * 32 : blockIdx_y * 32 + 32, k_0 * 32 : k_0 * 32 + 32], B[k_0 * 32 : k_0 * 32 + 32, blockIdx_x * 32 : blockIdx_x * 32 + 32])
                                Ts.writes(C[blockIdx_y * 32 : blockIdx_y * 32 + 32, blockIdx_x * 32 : blockIdx_x * 32 + 32])
                                A_shared = Ts.sblock_alloc_tensor([1024, 1024], dtype="float32", scope="shared")
                                B_shared = Ts.sblock_alloc_tensor([1024, 1024], dtype="float32", scope="shared")
                                A_shared_local = Ts.sblock_alloc_tensor([64, 4], dtype="float32", scope="local")
                                B_shared_local = Ts.sblock_alloc_tensor([64, 4], dtype="float32", scope="local")
                                for ax0_ax1_fused_0 in T.serial(64):
                                    for ax0_ax1_fused_3 in T.vectorized(4):
                                        with Ts.sblock():
                                            Ts.reads(A[blockIdx_y * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) // 32, k_0 * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) % 32])
                                            Ts.writes(A_shared_local[ax0_ax1_fused_0, ax0_ax1_fused_3])
                                            A_shared_local[ax0_ax1_fused_0, ax0_ax1_fused_3] = A[blockIdx_y * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) // 32, k_0 * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) % 32]
                                for ax0_ax1_fused_0 in T.serial(64):
                                    for ax0_ax1_fused_3 in T.vectorized(4):
                                        with Ts.sblock("A_shared"):
                                            Ts.reads(A[blockIdx_y * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) // 32, k_0 * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) % 32])
                                            Ts.writes(A_shared[blockIdx_y * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) // 32, k_0 * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) % 32])
                                            A_shared[blockIdx_y * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) // 32, k_0 * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) % 32] = A_shared_local[ax0_ax1_fused_0, ax0_ax1_fused_3]
                                for ax0_ax1_fused_0 in T.serial(64):
                                    for ax0_ax1_fused_3 in T.vectorized(4):
                                        with Ts.sblock():
                                            Ts.reads(B[k_0 * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) // 32, blockIdx_x * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) % 32])
                                            Ts.writes(B_shared_local[ax0_ax1_fused_0, ax0_ax1_fused_3])
                                            B_shared_local[ax0_ax1_fused_0, ax0_ax1_fused_3] = B[k_0 * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) // 32, blockIdx_x * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) % 32]
                                for ax0_ax1_fused_0 in T.serial(64):
                                    for ax0_ax1_fused_3 in T.vectorized(4):
                                        with Ts.sblock("B_shared"):
                                            Ts.reads(B[k_0 * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) // 32, blockIdx_x * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) % 32])
                                            Ts.writes(B_shared[k_0 * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) // 32, blockIdx_x * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) % 32])
                                            B_shared[k_0 * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) // 32, blockIdx_x * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) % 32] = B_shared_local[ax0_ax1_fused_0, ax0_ax1_fused_3]
                                for k_1, i_2, j_2, k_2 in T.grid(2, 16, 16, 16):
                                    with Ts.sblock("C"):
                                        Ts.reads(A_shared[blockIdx_y * 32 + threadIdx_y * 16 + i_2, k_0 * 32 + k_1 * 16 + k_2], B_shared[k_0 * 32 + k_1 * 16 + k_2, blockIdx_x * 32 + threadIdx_x * 16 + j_2])
                                        Ts.writes(C[blockIdx_y * 32 + threadIdx_y * 16 + i_2, blockIdx_x * 32 + threadIdx_x * 16 + j_2])
                                        if k_0 * 32 + k_1 * 16 + k_2 == 0:
                                            C[blockIdx_y * 32 + threadIdx_y * 16 + i_2, blockIdx_x * 32 + threadIdx_x * 16 + j_2] = T.float32(0)
                                        C[blockIdx_y * 32 + threadIdx_y * 16 + i_2, blockIdx_x * 32 + threadIdx_x * 16 + j_2] = C[blockIdx_y * 32 + threadIdx_y * 16 + i_2, blockIdx_x * 32 + threadIdx_x * 16 + j_2] + A_shared[blockIdx_y * 32 + threadIdx_y * 16 + i_2, k_0 * 32 + k_1 * 16 + k_2] * B_shared[k_0 * 32 + k_1 * 16 + k_2, blockIdx_x * 32 + threadIdx_x * 16 + j_2]


# fmt: on
# pylint: enable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks


def _check(before, expected):
    after = tvm.s_tir.transform.ManifestSharedMemoryLocalStage()(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_transform_matmul():
    _check(MatmulBefore, MatmulAfter)


if __name__ == "__main__":
    tvm_tirx.testing.main()
