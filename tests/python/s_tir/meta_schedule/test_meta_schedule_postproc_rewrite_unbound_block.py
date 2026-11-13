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
# ruff: noqa: F401

import tvm
import tvm_tirx
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir import script as Ts
from tvm_tirx import tirx
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target


def _target() -> Target:
    return Target({"kind": "cuda", "max_threads_per_block": 1024}, host="llvm")


def _create_context(mod, target) -> ms.TuneContext:
    ctx = ms.TuneContext(
        mod=mod,
        target=target,
        space_generator=ms.space_generator.PostOrderApply(
            sch_rules=[],
            postprocs=[ms.postproc.RewriteUnboundBlock()],
            mutator_probs={},
        ),
        task_name="test",
    )
    return ctx


# pylint: disable=no-member,invalid-name,unused-variable,no-self-argument,line-too-long,chained-comparison,not-callable,too-many-nested-blocks


@tvm_tirx.script.ir_module
class Before_cooperative_fetch:
    @Ts.function
    def main(
        A: T.Tensor([512, 512], dtype="float32"), B: T.Tensor([512, 512], dtype="float32")
    ) -> None:
        for i, j in T.grid(512, 512):
            with Ts.sblock("C"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] + 1.0


@tvm_tirx.script.ir_module
class After_cooperative_fetch:
    @Ts.function
    def main(
        A: T.Tensor([512, 512], dtype="float32"), B: T.Tensor([512, 512], dtype="float32")
    ) -> None:
        for i_j_fused_0 in T.parallel(256, annotations={"thread_binding": "blockIdx.x"}):
            for i_j_fused_1 in T.parallel(1024, annotations={"thread_binding": "threadIdx.x"}):
                with Ts.sblock("C"):
                    vi = Ts.axis.spatial(512, (i_j_fused_0 * 1024 + i_j_fused_1) // 512)
                    vj = Ts.axis.spatial(512, (i_j_fused_0 * 1024 + i_j_fused_1) % 512)
                    B[vi, vj] = A[vi, vj] + 1.0


@tvm_tirx.script.ir_module
class Before_norm_bmn:
    @Ts.function
    def main(A: T.Tensor((1, 256, 256), "float32"), D: T.Tensor((1,), "float32")) -> None:
        C = Ts.sblock_alloc_tensor([1], dtype="float32")
        for i0, i1, i2 in T.grid(1, 256, 256):
            with Ts.sblock("C"):
                b, i, j = Ts.axis.remap("SRR", [i0, i1, i2])
                with Ts.init():
                    C[b] = T.float32(0)
                C[b] = C[b] + A[b, i, j] * A[b, i, j]
        for i0 in T.serial(1):
            with Ts.sblock("D"):
                b = Ts.axis.S(1, i0)
                D[b] = T.sqrt(C[b])


@tvm_tirx.script.ir_module
class After_norm_bmn:
    @Ts.function
    def main(A: T.Tensor((1, 256, 256), "float32"), D: T.Tensor((1,), "float32")) -> None:
        C = Ts.sblock_alloc_tensor([1], dtype="float32")
        for i0_fused_0 in T.parallel(1, annotations={"thread_binding": "blockIdx.x"}):
            for i0_fused_1 in T.parallel(1, annotations={"thread_binding": "threadIdx.x"}):
                for i1, i2 in T.grid(256, 256):
                    with Ts.sblock("C"):
                        b = Ts.axis.S(1, 0)
                        i, j = Ts.axis.remap("RR", [i1, i2])
                        with Ts.init():
                            C[b] = T.float32(0)
                        C[b] = C[b] + A[b, i, j] * A[b, i, j]
        for i0_fused_0 in T.parallel(1, annotations={"thread_binding": "blockIdx.x"}):
            for i0_fused_1 in T.parallel(1, annotations={"thread_binding": "threadIdx.x"}):
                with Ts.sblock("D"):
                    b = Ts.axis.S(1, 0)
                    D[b] = T.sqrt(C[b])


@tvm_tirx.script.ir_module
class Bert_fused_reshape_transpose_reshape:
    @Ts.function
    def main(
        placeholder: T.Tensor((12, 64, 64), "float32"), T_reshape: T.Tensor((64, 768), "float32")
    ) -> None:
        for i0_i1_fused_0, i0_i1_fused_1 in T.grid(1536, 32):
            with Ts.sblock("T_reshape_1"):
                ax0 = Ts.axis.spatial(64, (i0_i1_fused_0 * 32 + i0_i1_fused_1) // 768)
                ax1 = Ts.axis.spatial(768, (i0_i1_fused_0 * 32 + i0_i1_fused_1) % 768)
                Ts.reads(placeholder[ax1 % 768 // 64, (ax1 // 768 + ax0) % 64, ax1 % 64])
                Ts.writes(T_reshape[ax0, ax1])
                T_reshape[ax0, ax1] = placeholder[
                    ((ax1 % 64 // 64 + (ax1 // 768 + ax0) % 64) // 64 + ax1 % 768 // 64) % 12,
                    (ax1 % 64 // 64 + (ax1 // 768 + ax0) % 64) % 64,
                    ax1 % 64 % 64,
                ]


@tvm_tirx.script.ir_module
class Bert_fused_reshape_transpose_reshape_large:
    @Ts.function
    def main(
        placeholder: T.Tensor((12, 64, 64), "float32"), T_reshape: T.Tensor((64, 768), "float32")
    ) -> None:
        for i0_i1_fused_0, i0_i1_fused_1 in T.grid(1536000, 32):
            with Ts.sblock("T_reshape_1"):
                ax0 = Ts.axis.spatial(64, (i0_i1_fused_0 * 32 + i0_i1_fused_1) // 768)
                ax1 = Ts.axis.spatial(768, (i0_i1_fused_0 * 32 + i0_i1_fused_1) % 768)
                Ts.reads(placeholder[ax1 % 768 // 64, (ax1 // 768 + ax0) % 64, ax1 % 64])
                Ts.writes(T_reshape[ax0, ax1])
                T_reshape[ax0, ax1] = placeholder[
                    ((ax1 % 64 // 64 + (ax1 // 768 + ax0) % 64) // 64 + ax1 % 768 // 64) % 12,
                    (ax1 % 64 // 64 + (ax1 // 768 + ax0) % 64) % 64,
                    ax1 % 64 % 64,
                ]


@tvm_tirx.script.ir_module
class Bert_fused_reshape_transpose_reshape_after_rub:
    @Ts.function
    def main(
        placeholder: T.Tensor((12, 64, 64), "float32"), T_reshape: T.Tensor((64, 768), "float32")
    ) -> None:
        for i0_i1_fused_0_i0_i1_fused_1_fused_0 in T.parallel(
            48, annotations={"thread_binding": "blockIdx.x"}
        ):
            for i0_i1_fused_0_i0_i1_fused_1_fused_1 in T.parallel(
                1024, annotations={"thread_binding": "threadIdx.x"}
            ):
                with Ts.sblock("T_reshape_1"):
                    ax0 = Ts.axis.spatial(
                        64,
                        (
                            i0_i1_fused_0_i0_i1_fused_1_fused_0 * 1024
                            + i0_i1_fused_0_i0_i1_fused_1_fused_1
                        )
                        // 768,
                    )
                    ax1 = Ts.axis.spatial(
                        768,
                        (
                            i0_i1_fused_0_i0_i1_fused_1_fused_0 * 1024
                            + i0_i1_fused_0_i0_i1_fused_1_fused_1
                        )
                        % 768,
                    )
                    Ts.reads(placeholder[ax1 % 768 // 64, (ax1 // 768 + ax0) % 64, ax1 % 64])
                    Ts.writes(T_reshape[ax0, ax1])
                    T_reshape[ax0, ax1] = placeholder[
                        ((ax1 % 64 // 64 + (ax1 // 768 + ax0) % 64) // 64 + ax1 % 768 // 64) % 12,
                        (ax1 % 64 // 64 + (ax1 // 768 + ax0) % 64) % 64,
                        ax1 % 64 % 64,
                    ]


@tvm_tirx.script.ir_module
class Bert_fused_reshape_transpose_reshape_after_rub_large:
    @Ts.function
    def main(
        placeholder: T.Tensor((12, 64, 64), "float32"), T_reshape: T.Tensor((64, 768), "float32")
    ) -> None:
        # body
        # with Ts.sblock("root")
        for i0_i1_fused_0_i0_i1_fused_1_fused_1 in T.parallel(
            256, annotations={"thread_binding": "blockIdx.x"}
        ):
            for i0_i1_fused_0_i0_i1_fused_1_fused_2 in T.parallel(
                1024, annotations={"thread_binding": "threadIdx.x"}
            ):
                for i0_i1_fused_0_i0_i1_fused_1_fused_0 in T.serial(188):
                    with Ts.sblock("T_reshape_1"):
                        ax0 = Ts.axis.spatial(
                            64,
                            (
                                i0_i1_fused_0_i0_i1_fused_1_fused_0 * 262144
                                + i0_i1_fused_0_i0_i1_fused_1_fused_1 * 1024
                                + i0_i1_fused_0_i0_i1_fused_1_fused_2
                            )
                            // 768,
                        )
                        ax1 = Ts.axis.spatial(
                            768,
                            (
                                i0_i1_fused_0_i0_i1_fused_1_fused_0 * 262144
                                + i0_i1_fused_0_i0_i1_fused_1_fused_1 * 1024
                                + i0_i1_fused_0_i0_i1_fused_1_fused_2
                            )
                            % 768,
                        )
                        Ts.where(
                            (
                                i0_i1_fused_0_i0_i1_fused_1_fused_0 * 256
                                + i0_i1_fused_0_i0_i1_fused_1_fused_1
                            )
                            * 1024
                            + i0_i1_fused_0_i0_i1_fused_1_fused_2
                            < 49152000
                        )
                        Ts.reads(placeholder[ax1 % 768 // 64, (ax1 // 768 + ax0) % 64, ax1 % 64])
                        Ts.writes(T_reshape[ax0, ax1])
                        T_reshape[ax0, ax1] = placeholder[
                            ((ax1 % 64 // 64 + (ax1 // 768 + ax0) % 64) // 64 + ax1 % 768 // 64)
                            % 12,
                            (ax1 % 64 // 64 + (ax1 // 768 + ax0) % 64) % 64,
                            ax1 % 64 % 64,
                        ]


@Ts.function
def before_unrolled_loop(
    placeholder: T.Tensor((1, 56, 56, 64), "float32"),
) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tirx.noalias": True})
    bgemm = Ts.sblock_alloc_tensor([6, 6, 196, 64], dtype="float32")
    inverse = Ts.sblock_alloc_tensor([4, 4, 196, 64], dtype="float32")
    for i2_0, i3_0, i2_1, i3_1 in T.grid(98, 4, 2, 16):
        for i0 in T.unroll(4):
            for i1 in T.unroll(4):
                for i4 in T.unroll(6):
                    for i5 in T.unroll(6):
                        with Ts.sblock("inverse"):
                            vh, vw = Ts.axis.remap("SS", [i0, i1])
                            p = Ts.axis.spatial(196, i2_0 * 2 + i2_1)
                            co = Ts.axis.spatial(64, i3_0 * 16 + i3_1)
                            r_a, r_b = Ts.axis.remap("RR", [i4, i5])
                            Ts.reads(bgemm[r_a, r_b, p, co])
                            Ts.writes(inverse[vh, vw, p, co])
                            with Ts.init():
                                inverse[vh, vw, p, co] = T.float32(0)
                            inverse[vh, vw, p, co] = inverse[vh, vw, p, co] + bgemm[r_a, r_b, p, co]


@Ts.function
def after_unrolled_loop(
    placeholder: T.Tensor((1, 56, 56, 64), "float32"),
) -> None:
    T.func_attr({"global_symbol": "main", "tirx.noalias": True})
    # body
    # with Ts.sblock("root")
    bgemm = Ts.sblock_alloc_tensor([6, 6, 196, 64], dtype="float32")
    inverse = Ts.sblock_alloc_tensor([4, 4, 196, 64], dtype="float32")
    for i2_0_i3_0_i2_1_i3_1_fused_0 in T.parallel(13, annotations={"thread_binding": "blockIdx.x"}):
        for i2_0_i3_0_i2_1_i3_1_fused_1 in T.parallel(
            1024, annotations={"thread_binding": "threadIdx.x"}
        ):
            for i0 in T.unroll(4):
                for i1 in T.unroll(4):
                    for i4 in T.unroll(6):
                        for i5 in T.unroll(6):
                            with Ts.sblock("inverse"):
                                vh, vw = Ts.axis.remap("SS", [i0, i1])
                                p = Ts.axis.spatial(
                                    196,
                                    (
                                        i2_0_i3_0_i2_1_i3_1_fused_0 * 1024
                                        + i2_0_i3_0_i2_1_i3_1_fused_1
                                    )
                                    // 128
                                    * 2
                                    + (
                                        i2_0_i3_0_i2_1_i3_1_fused_0 * 1024
                                        + i2_0_i3_0_i2_1_i3_1_fused_1
                                    )
                                    % 32
                                    // 16,
                                )
                                co = Ts.axis.spatial(
                                    64,
                                    (
                                        i2_0_i3_0_i2_1_i3_1_fused_0 * 1024
                                        + i2_0_i3_0_i2_1_i3_1_fused_1
                                    )
                                    % 128
                                    // 32
                                    * 16
                                    + (
                                        i2_0_i3_0_i2_1_i3_1_fused_0 * 1024
                                        + i2_0_i3_0_i2_1_i3_1_fused_1
                                    )
                                    % 16,
                                )
                                r_a, r_b = Ts.axis.remap("RR", [i4, i5])
                                Ts.where(
                                    i2_0_i3_0_i2_1_i3_1_fused_0 * 1024 + i2_0_i3_0_i2_1_i3_1_fused_1
                                    < 12544
                                )
                                Ts.reads(bgemm[r_a, r_b, p, co])
                                Ts.writes(inverse[vh, vw, p, co])
                                with Ts.init():
                                    inverse[vh, vw, p, co] = T.float32(0)
                                inverse[vh, vw, p, co] = (
                                    inverse[vh, vw, p, co] + bgemm[r_a, r_b, p, co]
                                )


# pylint: enable=no-member,invalid-name,unused-variable,no-self-argument,line-too-long,chained-comparison,not-callable,too-many-nested-blocks
# fmt: on


def test_rewrite_cooperative_fetch():
    mod = Before_cooperative_fetch
    target = _target()
    ctx = _create_context(mod, target)
    sch = tvm.s_tir.Schedule(mod, debug_mask="all")
    sch.enter_postproc()
    assert ctx.space_generator.postprocs[0].apply(sch)
    tvm_tirx.ir.assert_structural_equal(sch.mod, After_cooperative_fetch)


def test_rewrite_norm_bmn():
    mod = Before_norm_bmn
    target = _target()
    ctx = _create_context(mod, target)
    sch = tvm.s_tir.Schedule(mod, debug_mask="all")
    sch.enter_postproc()
    assert ctx.space_generator.postprocs[0].apply(sch)
    tvm_tirx.ir.assert_structural_equal(sch.mod, After_norm_bmn)


def test_rewrite_cuda_loop_split_no_reduction():
    mod = Bert_fused_reshape_transpose_reshape
    target = Target("nvidia/nvidia-v100", host="llvm")
    ctx = _create_context(mod, target)
    sch = tvm.s_tir.Schedule(mod, debug_mask="all")
    sch.enter_postproc()
    assert ctx.space_generator.postprocs[0].apply(sch)
    tvm_tirx.ir.assert_structural_equal(sch.mod, Bert_fused_reshape_transpose_reshape_after_rub)


def test_rewrite_cuda_loop_split_no_reduction_large():
    mod = Bert_fused_reshape_transpose_reshape_large
    target = Target("nvidia/nvidia-v100", host="llvm")
    ctx = _create_context(mod, target)
    sch = tvm.s_tir.Schedule(mod, debug_mask="all")
    sch.enter_postproc()
    assert ctx.space_generator.postprocs[0].apply(sch)
    tvm_tirx.ir.assert_structural_equal(
        sch.mod, Bert_fused_reshape_transpose_reshape_after_rub_large
    )


def test_rewrite_cuda_loop_split_for_kind():
    mod = before_unrolled_loop
    target = Target("nvidia/nvidia-v100", host="llvm")
    ctx = _create_context(mod, target)
    sch = tvm.s_tir.Schedule(mod, debug_mask="all")
    sch.enter_postproc()
    assert ctx.space_generator.postprocs[0].apply(sch)
    tvm_tirx.ir.assert_structural_equal(sch.mod["main"], after_unrolled_loop)


if __name__ == "__main__":
    test_rewrite_cooperative_fetch()
    test_rewrite_norm_bmn()
    test_rewrite_cuda_loop_split_no_reduction()
    test_rewrite_cuda_loop_split_no_reduction_large()
    test_rewrite_cuda_loop_split_for_kind()
