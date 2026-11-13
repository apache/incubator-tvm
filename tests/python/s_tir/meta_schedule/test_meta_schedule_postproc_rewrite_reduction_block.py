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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring
# ruff: noqa: E501, F401

import tvm
import tvm_tirx
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir import script as Ts
from tvm_tirx import tirx
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target


def _target() -> Target:
    return Target("cuda", host="llvm")


def _create_context(mod, target) -> ms.TuneContext:
    ctx = ms.TuneContext(
        mod=mod,
        target=target,
        space_generator=ms.space_generator.PostOrderApply(
            sch_rules=[],
            postprocs=[
                ms.postproc.RewriteReductionBlock(),
            ],
            mutator_probs={},
        ),
        task_name="test",
    )
    return ctx


# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,no-self-argument,line-too-long,chained-comparison,not-callable,too-many-nested-blocks

@tvm_tirx.script.ir_module
class Matmul_before_rewrite:
    @Ts.function
    def main(A: T.Tensor([512, 512], dtype='float32'), B: T.Tensor([512, 512], dtype='float32'), C: T.Tensor([512, 512], dtype='float32')) -> None:

        C_local = Ts.sblock_alloc_tensor([512, 512], dtype="float32", scope="local")
        A_shared = Ts.sblock_alloc_tensor([512, 512], dtype="float32", scope="shared")
        B_shared = Ts.sblock_alloc_tensor([512, 512], dtype="float32", scope="shared")
        for i0_0_i1_0_fused in T.parallel(0, 16, annotations={"thread_binding": "blockIdx.x"}):
            for i0_1_i1_1_fused in T.parallel(0, 16, annotations={"thread_binding": "vthread.x"}):
                for i0_2_i1_2_fused in T.parallel(0, 8, annotations={"thread_binding": "threadIdx.x"}):
                    for i2_0 in T.serial(0, 1):
                        for ax0_ax1_fused_0 in T.serial(0, 32768):
                            for ax0_ax1_fused_1 in T.parallel(0, 8, annotations={"thread_binding": "threadIdx.x"}):
                                with Ts.sblock("A_shared"):
                                    v0 = Ts.axis.spatial(512, (ax0_ax1_fused_0 * 8 + ax0_ax1_fused_1) // 512)
                                    v1 = Ts.axis.spatial(512, (ax0_ax1_fused_0 * 8 + ax0_ax1_fused_1) % 512)
                                    Ts.reads([A[v0, v1]])
                                    Ts.writes([A_shared[v0, v1]])
                                    Ts.sblock_attr({"meta_schedule.cooperative_fetch":1})
                                    A_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused_0 in T.serial(0, 1024):
                            for ax0_ax1_fused_1 in T.parallel(0, 8, annotations={"thread_binding": "threadIdx.x"}):
                                for ax0_ax1_fused_2 in T.vectorized(0, 2):
                                    with Ts.sblock("B_shared"):
                                        v0 = Ts.axis.spatial(512, (ax0_ax1_fused_0 * 16 + ax0_ax1_fused_1 * 2 + ax0_ax1_fused_2) // 32)
                                        v1 = Ts.axis.spatial(512, i0_0_i1_0_fused * 32 + (ax0_ax1_fused_0 * 16 + ax0_ax1_fused_1 * 2 + ax0_ax1_fused_2) % 32)
                                        Ts.reads([B[v0, v1]])
                                        Ts.writes([B_shared[v0, v1]])
                                        Ts.sblock_attr({"meta_schedule.cooperative_fetch":2})
                                        B_shared[v0, v1] = B[v0, v1]
                        for i2_1, i0_3, i1_3, i2_2, i0_4, i1_4 in T.grid(16, 2, 2, 32, 16, 2):
                            with Ts.sblock("C"):
                                i = Ts.axis.spatial(512, i0_1_i1_1_fused * 32 + i0_3 * 16 + i0_4)
                                j = Ts.axis.spatial(512, i0_0_i1_0_fused * 32 + i0_2_i1_2_fused * 4 + i1_3 * 2 + i1_4)
                                k = Ts.axis.reduce(512, i2_1 * 32 + i2_2)
                                Ts.reads([C_local[i, j], A_shared[i, k], B_shared[k, j]])
                                Ts.writes([C_local[i, j]])
                                with Ts.init():
                                    C_local[i, j] = T.float32(0)
                                C_local[i, j] = C_local[i, j] + A_shared[i, k] * B_shared[k, j]
                    for ax0, ax1 in T.grid(32, 4):
                        with Ts.sblock("C_local"):
                            v0 = Ts.axis.spatial(512, i0_1_i1_1_fused * 32 + ax0)
                            v1 = Ts.axis.spatial(512, i0_0_i1_0_fused * 32 + i0_2_i1_2_fused * 4 + ax1)
                            Ts.reads([C_local[v0, v1]])
                            Ts.writes([C[v0, v1]])
                            C[v0, v1] = C_local[v0, v1]

@tvm_tirx.script.ir_module
class Matmul_after_rewrite:
    @Ts.function
    def main(A: T.Tensor([512, 512], dtype='float32'), B: T.Tensor([512, 512], dtype='float32'), C: T.Tensor([512, 512], dtype='float32')) -> None:

        C_local = Ts.sblock_alloc_tensor([512, 512], dtype="float32", scope="local")
        A_shared = Ts.sblock_alloc_tensor([512, 512], dtype="float32", scope="shared")
        B_shared = Ts.sblock_alloc_tensor([512, 512], dtype="float32", scope="shared")
        for i0_0_i1_0_fused in T.parallel(0, 16, annotations={"thread_binding": "blockIdx.x"}):
            for i0_1_i1_1_fused in T.parallel(0, 16, annotations={"thread_binding": "vthread.x"}):
                for i0_2_i1_2_fused in T.parallel(0, 8, annotations={"thread_binding": "threadIdx.x"}):
                    for i2_0 in T.serial(0, 1):
                        for ax0_ax1_fused_0 in T.serial(0, 32768):
                            for ax0_ax1_fused_1 in T.parallel(0, 8, annotations={"thread_binding": "threadIdx.x"}):
                                with Ts.sblock("A_shared"):
                                    v0 = Ts.axis.spatial(512, (ax0_ax1_fused_0 * 8 + ax0_ax1_fused_1) // 512)
                                    v1 = Ts.axis.spatial(512, (ax0_ax1_fused_0 * 8 + ax0_ax1_fused_1) % 512)
                                    Ts.reads([A[v0, v1]])
                                    Ts.writes([A_shared[v0, v1]])
                                    Ts.sblock_attr({"meta_schedule.cooperative_fetch":1})
                                    A_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused_0 in T.serial(0, 1024):
                            for ax0_ax1_fused_1 in T.parallel(0, 8, annotations={"thread_binding": "threadIdx.x"}):
                                for ax0_ax1_fused_2 in T.vectorized(0, 2):
                                    with Ts.sblock("B_shared"):
                                        v0 = Ts.axis.spatial(512, (ax0_ax1_fused_0 * 16 + ax0_ax1_fused_1 * 2 + ax0_ax1_fused_2) // 32)
                                        v1 = Ts.axis.spatial(512, i0_0_i1_0_fused * 32 + (ax0_ax1_fused_0 * 16 + ax0_ax1_fused_1 * 2 + ax0_ax1_fused_2) % 32)
                                        Ts.reads([B[v0, v1]])
                                        Ts.writes([B_shared[v0, v1]])
                                        Ts.sblock_attr({"meta_schedule.cooperative_fetch":2})
                                        B_shared[v0, v1] = B[v0, v1]
                        for i0_3_init, i1_3_init, i0_4_init, i1_4_init in T.grid(2, 2, 16, 2):
                            with Ts.sblock("C_init"):
                                i = Ts.axis.spatial(512, i0_1_i1_1_fused * 32 + i0_3_init * 16 + i0_4_init)
                                j = Ts.axis.spatial(512, i0_0_i1_0_fused * 32 + i0_2_i1_2_fused * 4 + i1_3_init * 2 + i1_4_init)
                                Ts.reads([])
                                Ts.writes([C_local[i, j]])
                                C_local[i, j] = T.float32(0)
                        for i2_1, i0_3, i1_3, i2_2, i0_4, i1_4 in T.grid(16, 2, 2, 32, 16, 2):
                            with Ts.sblock("C_update"):
                                i = Ts.axis.spatial(512, i0_1_i1_1_fused * 32 + i0_3 * 16 + i0_4)
                                j = Ts.axis.spatial(512, i0_0_i1_0_fused * 32 + i0_2_i1_2_fused * 4 + i1_3 * 2 + i1_4)
                                k = Ts.axis.reduce(512, i2_1 * 32 + i2_2)
                                Ts.reads([C_local[i, j], A_shared[i, k], B_shared[k, j]])
                                Ts.writes([C_local[i, j]])
                                C_local[i, j] = C_local[i, j] + A_shared[i, k] * B_shared[k, j]
                    for ax0, ax1 in T.grid(32, 4):
                        with Ts.sblock("C_local"):
                            v0 = Ts.axis.spatial(512, i0_1_i1_1_fused * 32 + ax0)
                            v1 = Ts.axis.spatial(512, i0_0_i1_0_fused * 32 + i0_2_i1_2_fused * 4 + ax1)
                            Ts.reads([C_local[v0, v1]])
                            Ts.writes([C[v0, v1]])
                            C[v0, v1] = C_local[v0, v1]

@tvm_tirx.script.ir_module
class Softmax_cross_thread_reduction:
    @Ts.function
    def main(A: T.Tensor((256, 256), "float32"), T_softmax_norm: T.Tensor((256, 256), "float32")) -> None:
        T_softmax_maxelem_shared = Ts.sblock_alloc_tensor([256], dtype="float32", scope="shared")
        T_softmax_expsum_shared = Ts.sblock_alloc_tensor([256], dtype="float32", scope="shared")
        for i0 in T.serial(256):
            for ax0, ax1_0 in T.grid(1, 8):
                for ax1_1 in T.parallel(32, annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("T_softmax_maxelem"):
                        i0_1 = Ts.axis.spatial(256, i0)
                        k = Ts.axis.reduce(256, ax1_0 * 32 + ax1_1)
                        Ts.reads(T_softmax_maxelem_shared[i0_1], A[i0_1, k])
                        Ts.writes(T_softmax_maxelem_shared[i0_1])
                        with Ts.init():
                            T_softmax_maxelem_shared[i0_1] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[i0_1] = T.max(T_softmax_maxelem_shared[i0_1], A[i0_1, k])
            for ax0, ax1_0 in T.grid(1, 8):
                for ax1_1 in T.parallel(32, annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("T_softmax_expsum"):
                        i0_2 = Ts.axis.spatial(256, i0)
                        k = Ts.axis.reduce(256, ax1_0 * 32 + ax1_1)
                        Ts.reads(T_softmax_expsum_shared[i0_2], A[i0_2, k], T_softmax_maxelem_shared[i0_2])
                        Ts.writes(T_softmax_expsum_shared[i0_2])
                        with Ts.init():
                            T_softmax_expsum_shared[i0_2] = T.float32(0)
                        T_softmax_expsum_shared[i0_2] = T_softmax_expsum_shared[i0_2] + T.exp(A[i0_2, k] - T_softmax_maxelem_shared[i0_2])
            for i1_0 in T.serial(8):
                for i1_1 in T.parallel(32, annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("T_softmax_norm"):
                        i0_3 = Ts.axis.spatial(256, i0)
                        i1 = Ts.axis.spatial(256, i1_0 * 32 + i1_1)
                        Ts.reads(A[i0_3, i1], T_softmax_maxelem_shared[i0_3], T_softmax_expsum_shared[i0_3])
                        Ts.writes(T_softmax_norm[i0_3, i1])
                        Ts.sblock_attr({"axis":1})
                        T_softmax_norm[i0_3, i1] = T.exp(A[i0_3, i1] - T_softmax_maxelem_shared[i0_3]) / T_softmax_expsum_shared[i0_3]

# pylint: enable=no-member,invalid-name,unused-variable,no-self-argument,line-too-long,chained-comparison,not-callable,too-many-nested-blocks
# fmt: on


def test_rewrite_tiled_matmul():
    mod = Matmul_before_rewrite
    target = _target()
    ctx = _create_context(mod, target)
    sch = tvm.s_tir.Schedule(mod, debug_mask="all")
    sch.enter_postproc()
    assert ctx.space_generator.postprocs[0].apply(sch)
    tvm_tirx.ir.assert_structural_equal(sch.mod, Matmul_after_rewrite)


def test_rewrite_softmax():
    mod = Softmax_cross_thread_reduction
    target = _target()
    ctx = _create_context(mod, target)
    sch = tvm.s_tir.Schedule(mod, debug_mask="all")
    sch.enter_postproc()
    assert ctx.space_generator.postprocs[0].apply(sch)
    # The module should not be rewritten
    tvm_tirx.ir.assert_structural_equal(sch.mod, Softmax_cross_thread_reduction)


if __name__ == "__main__":
    test_rewrite_tiled_matmul()
    test_rewrite_softmax()
