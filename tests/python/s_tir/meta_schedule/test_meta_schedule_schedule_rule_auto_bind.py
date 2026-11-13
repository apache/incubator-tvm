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
# ruff: noqa: F841
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir import script as Ts
from tvm.s_tir.meta_schedule.testing.space_generation import (
    check_sketches,
    generate_design_space,
)
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target


@Ts.function
def element_wise(
    A: T.Tensor([512, 512], dtype="float32"), B: T.Tensor([512, 512], dtype="float32")
) -> None:
    for i, j in T.grid(512, 512):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] + 1.0


@Ts.function
def reduction_loop_only(
    A: T.Tensor(2, "float32"),
    B: T.Tensor(2, "float32"),
    C: T.Tensor((), "float32"),
) -> None:
    for i0 in T.serial(2):
        with Ts.sblock("C"):
            k0 = Ts.axis.reduce(2, i0)
            Ts.reads(A[k0], B[k0])
            Ts.writes(C[()])
            with Ts.init():
                C[()] = T.float32(1.0)
            C[()] = T.min(C[()], A[k0] / B[k0])


@Ts.function
def zero_dim_add(
    A: T.Tensor((), "float32"),
    B: T.Tensor((), "float32"),
    C: T.Tensor((), "float32"),
) -> None:
    with Ts.sblock("C"):
        vi = Ts.axis.spatial(1, 0)
        C[()] = A[()] + B[()]


def test_cuda_element_wise():
    @Ts.function
    def elementwise_0(
        A: T.Tensor((512, 512), "float32"),
        B: T.Tensor((512, 512), "float32"),
    ) -> None:
        # body
        # with Ts.sblock("root")
        for i_j_fused_0 in T.parallel(256, annotations={"thread_binding": "blockIdx.x"}):
            for i_j_fused_1 in T.parallel(1024, annotations={"thread_binding": "threadIdx.x"}):
                with Ts.sblock("C"):
                    vi = Ts.axis.spatial(512, (i_j_fused_0 * 1024 + i_j_fused_1) // 512)
                    vj = Ts.axis.spatial(512, (i_j_fused_0 * 1024 + i_j_fused_1) % 512)
                    Ts.reads(A[vi, vj])
                    Ts.writes(B[vi, vj])
                    B[vi, vj] = A[vi, vj] + T.float32(1)

    decision_0 = [
        ("SampleCategorical", 5),
    ]
    mod = element_wise
    actual = generate_design_space(
        kind="cuda",
        mod=mod,
        target=Target("nvidia/geforce-rtx-3080", host="llvm"),
        types=ms.schedule_rule.AutoBind,
    )
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[elementwise_0],
        expected_decisions=[decision_0],
    )


def test_cuda_reduction_loop_only():
    @Ts.function
    def reduction_loop_only_0(
        A: T.Tensor(2, "float32"),
        B: T.Tensor(2, "float32"),
        C: T.Tensor((), "float32"),
    ) -> None:
        for u_fused_0 in T.parallel(1, annotations={"thread_binding": "blockIdx.x"}):
            for u_fused_1 in T.parallel(1, annotations={"thread_binding": "threadIdx.x"}):
                for i0 in T.serial(2):
                    with Ts.sblock("C"):
                        k0 = Ts.axis.reduce(2, i0)
                        Ts.reads(A[k0], B[k0])
                        Ts.writes(C[()])
                        with Ts.init():
                            C[()] = T.float32(1)
                        C[()] = T.min(C[()], A[k0] / B[k0])

    mod = reduction_loop_only
    actual = generate_design_space(
        kind="cuda",
        mod=mod,
        target=Target("nvidia/geforce-rtx-3080", host="llvm"),
        types=ms.schedule_rule.AutoBind,
    )
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[reduction_loop_only_0],
        expected_decisions=[[]],
    )


def test_cuda_zero_dim_add():
    @Ts.function
    def zero_dim_add_0(
        A: T.Tensor((), "float32"),
        B: T.Tensor((), "float32"),
        C: T.Tensor((), "float32"),
    ) -> None:
        for u_fused_0 in T.parallel(1, annotations={"thread_binding": "blockIdx.x"}):
            for u_fused_1 in T.parallel(1, annotations={"thread_binding": "threadIdx.x"}):
                with Ts.sblock("C"):
                    vi = Ts.axis.spatial(1, 0)
                    Ts.reads(A[()], B[()])
                    Ts.writes(C[()])
                    C[()] = A[()] + B[()]

    mod = zero_dim_add
    actual = generate_design_space(
        kind="cuda",
        mod=mod,
        target=Target("nvidia/geforce-rtx-3080", host="llvm"),
        types=ms.schedule_rule.AutoBind,
    )
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[zero_dim_add_0],
        expected_decisions=[[]],
    )


if __name__ == "__main__":
    test_cuda_element_wise()
    test_cuda_reduction_loop_only()
    test_cuda_zero_dim_add()
