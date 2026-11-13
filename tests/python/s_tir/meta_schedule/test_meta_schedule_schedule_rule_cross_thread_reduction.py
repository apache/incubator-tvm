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

import tvm_tirx
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir import script as Ts
from tvm.s_tir.meta_schedule.testing import te_workload
from tvm.s_tir.meta_schedule.testing.space_generation import (
    check_sketches,
    generate_design_space,
)
from tvm.te import create_function
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target


@tvm_tirx.script.ir_module
class Softmax_mn_after_inline:
    @Ts.function
    def main(
        A: T.Tensor((256, 256), "float32"), T_softmax_norm: T.Tensor((256, 256), "float32")
    ) -> None:
        T_softmax_maxelem = Ts.sblock_alloc_tensor([256], dtype="float32")
        T_softmax_expsum = Ts.sblock_alloc_tensor([256], dtype="float32")
        for i0, i1 in T.grid(256, 256):
            with Ts.sblock("T_softmax_maxelem"):
                i0_1, k = Ts.axis.remap("SR", [i0, i1])
                with Ts.init():
                    T_softmax_maxelem[i0_1] = T.min_value(ty="float32")
                T_softmax_maxelem[i0_1] = T.max(T_softmax_maxelem[i0_1], A[i0_1, k])
        for i0, i1 in T.grid(256, 256):
            with Ts.sblock("T_softmax_expsum"):
                i0_2, k = Ts.axis.remap("SR", [i0, i1])
                with Ts.init():
                    T_softmax_expsum[i0_2] = T.float32(0)
                T_softmax_expsum[i0_2] = T_softmax_expsum[i0_2] + T.exp(
                    A[i0_2, k] - T_softmax_maxelem[i0_2]
                )
        for i0_3, i1 in T.grid(256, 256):
            with Ts.sblock("T_softmax_norm"):
                i0_4, i1_1 = Ts.axis.remap("SS", [i0_3, i1])
                Ts.sblock_attr({"axis": 1})
                T_softmax_norm[i0_4, i1_1] = (
                    T.exp(A[i0_4, i1_1] - T_softmax_maxelem[i0_4]) / T_softmax_expsum[i0_4]
                )


def test_gpu_softmax_mn():
    @Ts.function
    def softmax_mn_0(
        A: T.Tensor((256, 256), "float32"),
        T_softmax_norm: T.Tensor((256, 256), "float32"),
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        # body
        # with Ts.sblock("root")
        T_softmax_maxelem = Ts.sblock_alloc_tensor([256], dtype="float32")
        T_softmax_exp = Ts.sblock_alloc_tensor([256, 256], dtype="float32")
        T_softmax_expsum = Ts.sblock_alloc_tensor([256], dtype="float32")
        for i0, i1 in T.grid(256, 256):
            with Ts.sblock("T_softmax_maxelem"):
                i0_1, k = Ts.axis.remap("SR", [i0, i1])
                Ts.reads(A[i0_1, k])
                Ts.writes(T_softmax_maxelem[i0_1])
                with Ts.init():
                    T_softmax_maxelem[i0_1] = T.float32(-3.4028234663852886e38)
                T_softmax_maxelem[i0_1] = T.max(T_softmax_maxelem[i0_1], A[i0_1, k])
        for i0, i1 in T.grid(256, 256):
            with Ts.sblock("T_softmax_exp"):
                i0_2, i1_1 = Ts.axis.remap("SS", [i0, i1])
                Ts.reads(A[i0_2, i1_1], T_softmax_maxelem[i0_2])
                Ts.writes(T_softmax_exp[i0_2, i1_1])
                T_softmax_exp[i0_2, i1_1] = T.exp(A[i0_2, i1_1] - T_softmax_maxelem[i0_2])
        for i0_3, i1 in T.grid(256, 256):
            with Ts.sblock("T_softmax_expsum"):
                i0_4, k = Ts.axis.remap("SR", [i0_3, i1])
                Ts.reads(T_softmax_exp[i0_4, k])
                Ts.writes(T_softmax_expsum[i0_4])
                with Ts.init():
                    T_softmax_expsum[i0_4] = T.float32(0)
                T_softmax_expsum[i0_4] = T_softmax_expsum[i0_4] + T_softmax_exp[i0_4, k]
        for i0_5, i1 in T.grid(256, 256):
            with Ts.sblock("T_softmax_norm"):
                i0_6, i1_2 = Ts.axis.remap("SS", [i0_5, i1])
                Ts.reads(T_softmax_exp[i0_6, i1_2], T_softmax_expsum[i0_6])
                Ts.writes(T_softmax_norm[i0_6, i1_2])
                Ts.sblock_attr({"axis": 1})
                T_softmax_norm[i0_6, i1_2] = T_softmax_exp[i0_6, i1_2] / T_softmax_expsum[i0_6]

    @Ts.function
    def softmax_mn_1(
        A: T.Tensor((256, 256), "float32"), T_softmax_norm: T.Tensor((256, 256), "float32")
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        # body
        # with Ts.sblock("root")
        T_softmax_maxelem_shared = Ts.sblock_alloc_tensor([256], dtype="float32", scope="shared")
        T_softmax_exp = Ts.sblock_alloc_tensor([256, 256], dtype="float32")
        T_softmax_expsum = Ts.sblock_alloc_tensor([256], dtype="float32")
        for i0 in T.serial(256):
            for ax0, ax1_0 in T.grid(1, 1):
                for ax1_1 in T.parallel(512, annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("T_softmax_maxelem"):
                        Ts.where(ax1_0 * 512 + ax1_1 < 256)
                        i0_1 = Ts.axis.spatial(256, i0 + ax0)
                        k = Ts.axis.reduce(256, ax1_0 * 512 + ax1_1)
                        Ts.reads(A[i0_1, k])
                        Ts.writes(T_softmax_maxelem_shared[i0_1])
                        with Ts.init():
                            T_softmax_maxelem_shared[i0_1] = T.float32(-3.4028234663852886e38)
                        T_softmax_maxelem_shared[i0_1] = T.max(
                            T_softmax_maxelem_shared[i0_1], A[i0_1, k]
                        )
            for i1_0 in T.serial(1):
                for i1_1 in T.parallel(512, annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("T_softmax_exp"):
                        Ts.where(i1_0 * 512 + i1_1 < 256)
                        i0_2 = Ts.axis.spatial(256, i0)
                        i1 = Ts.axis.spatial(256, i1_0 * 512 + i1_1)
                        Ts.reads(A[i0_2, i1], T_softmax_maxelem_shared[i0_2])
                        Ts.writes(T_softmax_exp[i0_2, i1])
                        T_softmax_exp[i0_2, i1] = T.exp(
                            A[i0_2, i1] - T_softmax_maxelem_shared[i0_2]
                        )
        for i0_3, i1 in T.grid(256, 256):
            with Ts.sblock("T_softmax_expsum"):
                i0_4, k = Ts.axis.remap("SR", [i0_3, i1])
                Ts.reads(T_softmax_exp[i0_4, k])
                Ts.writes(T_softmax_expsum[i0_4])
                with Ts.init():
                    T_softmax_expsum[i0_4] = T.float32(0)
                T_softmax_expsum[i0_4] = T_softmax_expsum[i0_4] + T_softmax_exp[i0_4, k]
        for i0_5, i1 in T.grid(256, 256):
            with Ts.sblock("T_softmax_norm"):
                i0_6, i1_2 = Ts.axis.remap("SS", [i0_5, i1])
                Ts.reads(T_softmax_exp[i0_6, i1_2], T_softmax_expsum[i0_6])
                Ts.writes(T_softmax_norm[i0_6, i1_2])
                Ts.sblock_attr({"axis": 1})
                T_softmax_norm[i0_6, i1_2] = T_softmax_exp[i0_6, i1_2] / T_softmax_expsum[i0_6]

    @Ts.function
    def softmax_mn_2(
        A: T.Tensor((256, 256), "float32"), T_softmax_norm: T.Tensor((256, 256), "float32")
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        # body
        # with Ts.sblock("root")
        T_softmax_maxelem = Ts.sblock_alloc_tensor([256], dtype="float32")
        T_softmax_exp = Ts.sblock_alloc_tensor([256, 256], dtype="float32")
        T_softmax_expsum_shared = Ts.sblock_alloc_tensor([256], dtype="float32", scope="shared")
        for i0, i1 in T.grid(256, 256):
            with Ts.sblock("T_softmax_maxelem"):
                i0_1, k = Ts.axis.remap("SR", [i0, i1])
                Ts.reads(A[i0_1, k])
                Ts.writes(T_softmax_maxelem[i0_1])
                with Ts.init():
                    T_softmax_maxelem[i0_1] = T.float32(-3.4028234663852886e38)
                T_softmax_maxelem[i0_1] = T.max(T_softmax_maxelem[i0_1], A[i0_1, k])
        for i0, i1 in T.grid(256, 256):
            with Ts.sblock("T_softmax_exp"):
                i0_2, i1_1 = Ts.axis.remap("SS", [i0, i1])
                Ts.reads(A[i0_2, i1_1], T_softmax_maxelem[i0_2])
                Ts.writes(T_softmax_exp[i0_2, i1_1])
                T_softmax_exp[i0_2, i1_1] = T.exp(A[i0_2, i1_1] - T_softmax_maxelem[i0_2])
        for i0_3 in T.serial(256):
            for ax0, ax1_0 in T.grid(1, 32):
                for ax1_1 in T.parallel(8, annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("T_softmax_expsum"):
                        i0_4 = Ts.axis.spatial(256, i0_3 + ax0)
                        k = Ts.axis.reduce(256, ax1_0 * 8 + ax1_1)
                        Ts.reads(T_softmax_exp[i0_4, k])
                        Ts.writes(T_softmax_expsum_shared[i0_4])
                        with Ts.init():
                            T_softmax_expsum_shared[i0_4] = T.float32(0)
                        T_softmax_expsum_shared[i0_4] = (
                            T_softmax_expsum_shared[i0_4] + T_softmax_exp[i0_4, k]
                        )
            for i1_0 in T.serial(32):
                for i1_1_1 in T.parallel(8, annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("T_softmax_norm"):
                        i0_5 = Ts.axis.spatial(256, i0_3)
                        i1 = Ts.axis.spatial(256, i1_0 * 8 + i1_1_1)
                        Ts.reads(T_softmax_exp[i0_5, i1], T_softmax_expsum_shared[i0_5])
                        Ts.writes(T_softmax_norm[i0_5, i1])
                        Ts.sblock_attr({"axis": 1})
                        T_softmax_norm[i0_5, i1] = (
                            T_softmax_exp[i0_5, i1] / T_softmax_expsum_shared[i0_5]
                        )

    @Ts.function
    def softmax_mn_3(
        A: T.Tensor((256, 256), "float32"), T_softmax_norm: T.Tensor((256, 256), "float32")
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        # body
        # with Ts.sblock("root")
        T_softmax_maxelem_shared = Ts.sblock_alloc_tensor([256], dtype="float32", scope="shared")
        T_softmax_exp = Ts.sblock_alloc_tensor([256, 256], dtype="float32")
        T_softmax_expsum_shared = Ts.sblock_alloc_tensor([256], dtype="float32", scope="shared")
        for i0 in T.serial(256):
            for ax0, ax1_0 in T.grid(1, 1):
                for ax1_1 in T.parallel(512, annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("T_softmax_maxelem"):
                        Ts.where(ax1_0 * 512 + ax1_1 < 256)
                        i0_1 = Ts.axis.spatial(256, i0 + ax0)
                        k = Ts.axis.reduce(256, ax1_0 * 512 + ax1_1)
                        Ts.reads(A[i0_1, k])
                        Ts.writes(T_softmax_maxelem_shared[i0_1])
                        with Ts.init():
                            T_softmax_maxelem_shared[i0_1] = T.float32(-3.4028234663852886e38)
                        T_softmax_maxelem_shared[i0_1] = T.max(
                            T_softmax_maxelem_shared[i0_1], A[i0_1, k]
                        )
            for i1_0 in T.serial(1):
                for i1_1 in T.parallel(512, annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("T_softmax_exp"):
                        Ts.where(i1_0 * 512 + i1_1 < 256)
                        i0_2 = Ts.axis.spatial(256, i0)
                        i1 = Ts.axis.spatial(256, i1_0 * 512 + i1_1)
                        Ts.reads(A[i0_2, i1], T_softmax_maxelem_shared[i0_2])
                        Ts.writes(T_softmax_exp[i0_2, i1])
                        T_softmax_exp[i0_2, i1] = T.exp(
                            A[i0_2, i1] - T_softmax_maxelem_shared[i0_2]
                        )
        for i0_3 in T.serial(256):
            for ax0, ax1_0 in T.grid(1, 32):
                for ax1_1 in T.parallel(8, annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("T_softmax_expsum"):
                        i0_4 = Ts.axis.spatial(256, i0_3 + ax0)
                        k = Ts.axis.reduce(256, ax1_0 * 8 + ax1_1)
                        Ts.reads(T_softmax_exp[i0_4, k])
                        Ts.writes(T_softmax_expsum_shared[i0_4])
                        with Ts.init():
                            T_softmax_expsum_shared[i0_4] = T.float32(0)
                        T_softmax_expsum_shared[i0_4] = (
                            T_softmax_expsum_shared[i0_4] + T_softmax_exp[i0_4, k]
                        )
            for i1_0 in T.serial(32):
                for i1_1 in T.parallel(8, annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("T_softmax_norm"):
                        i0_5 = Ts.axis.spatial(256, i0_3)
                        i1 = Ts.axis.spatial(256, i1_0 * 8 + i1_1)
                        Ts.reads(T_softmax_exp[i0_5, i1], T_softmax_expsum_shared[i0_5])
                        Ts.writes(T_softmax_norm[i0_5, i1])
                        Ts.sblock_attr({"axis": 1})
                        T_softmax_norm[i0_5, i1] = (
                            T_softmax_exp[i0_5, i1] / T_softmax_expsum_shared[i0_5]
                        )

    decision_0 = []  # type: ignore
    decision_1 = [
        ("SampleCategorical", 7),
    ]
    decision_2 = [
        ("SampleCategorical", 1),
    ]
    decision_3 = [
        ("SampleCategorical", 1),
        ("SampleCategorical", 7),
    ]
    mod = create_function(te_workload.softmax_mn(n=256, m=256))
    actual = generate_design_space(
        kind="cuda",
        mod=mod,
        target=Target("nvidia/geforce-rtx-3090", host="llvm"),
        types=ms.schedule_rule.CrossThreadReduction,
    )
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[softmax_mn_0, softmax_mn_1, softmax_mn_2, softmax_mn_3],
        expected_decisions=[decision_0, decision_1, decision_2, decision_3],
    )


def test_gpu_softmax_mn_after_inline():
    @Ts.function
    def softmax_mn_after_inline_0(
        A: T.Tensor((256, 256), "float32"), T_softmax_norm: T.Tensor((256, 256), "float32")
    ) -> None:
        T_softmax_maxelem = Ts.sblock_alloc_tensor([256], dtype="float32")
        T_softmax_expsum = Ts.sblock_alloc_tensor([256], dtype="float32")
        for i0, i1 in T.grid(256, 256):
            with Ts.sblock("T_softmax_maxelem"):
                i0_1, k = Ts.axis.remap("SR", [i0, i1])
                Ts.reads(A[i0_1, k])
                Ts.writes(T_softmax_maxelem[i0_1])
                with Ts.init():
                    T_softmax_maxelem[i0_1] = T.float32(-3.4028234663852886e38)
                T_softmax_maxelem[i0_1] = T.max(T_softmax_maxelem[i0_1], A[i0_1, k])
        for i0, i1 in T.grid(256, 256):
            with Ts.sblock("T_softmax_expsum"):
                i0_2, k = Ts.axis.remap("SR", [i0, i1])
                Ts.reads(A[i0_2, k], T_softmax_maxelem[i0_2])
                Ts.writes(T_softmax_expsum[i0_2])
                with Ts.init():
                    T_softmax_expsum[i0_2] = T.float32(0)
                T_softmax_expsum[i0_2] = T_softmax_expsum[i0_2] + T.exp(
                    A[i0_2, k] - T_softmax_maxelem[i0_2]
                )
        for i0_3, i1 in T.grid(256, 256):
            with Ts.sblock("T_softmax_norm"):
                i0_4, i1_1 = Ts.axis.remap("SS", [i0_3, i1])
                Ts.reads(A[i0_4, i1_1], T_softmax_maxelem[i0_4], T_softmax_expsum[i0_4])
                Ts.writes(T_softmax_norm[i0_4, i1_1])
                Ts.sblock_attr({"axis": 1})
                T_softmax_norm[i0_4, i1_1] = (
                    T.exp(A[i0_4, i1_1] - T_softmax_maxelem[i0_4]) / T_softmax_expsum[i0_4]
                )

    @Ts.function
    def softmax_mn_after_inline_1(
        A: T.Tensor((256, 256), "float32"), T_softmax_norm: T.Tensor((256, 256), "float32")
    ) -> None:
        T_softmax_maxelem = Ts.sblock_alloc_tensor([256], dtype="float32")
        T_softmax_expsum = Ts.sblock_alloc_tensor([256], dtype="float32")
        for i0, i1_0 in T.grid(256, 4):
            for i1_1 in T.parallel(64, annotations={"thread_binding": "threadIdx.x"}):
                with Ts.sblock("T_softmax_maxelem"):
                    i0_1 = Ts.axis.spatial(256, i0)
                    k = Ts.axis.reduce(256, i1_0 * 64 + i1_1)
                    Ts.reads(A[i0_1, k])
                    Ts.writes(T_softmax_maxelem[i0_1])
                    with Ts.init():
                        T_softmax_maxelem[i0_1] = T.float32(-3.4028234663852886e38)
                    T_softmax_maxelem[i0_1] = T.max(T_softmax_maxelem[i0_1], A[i0_1, k])
        for i0, i1 in T.grid(256, 256):
            with Ts.sblock("T_softmax_expsum"):
                i0_2, k = Ts.axis.remap("SR", [i0, i1])
                Ts.reads(A[i0_2, k], T_softmax_maxelem[i0_2])
                Ts.writes(T_softmax_expsum[i0_2])
                with Ts.init():
                    T_softmax_expsum[i0_2] = T.float32(0)
                T_softmax_expsum[i0_2] = T_softmax_expsum[i0_2] + T.exp(
                    A[i0_2, k] - T_softmax_maxelem[i0_2]
                )
        for i0_3, i1 in T.grid(256, 256):
            with Ts.sblock("T_softmax_norm"):
                i0_4, i1_1 = Ts.axis.remap("SS", [i0_3, i1])
                Ts.reads(A[i0_4, i1_1], T_softmax_maxelem[i0_4], T_softmax_expsum[i0_4])
                Ts.writes(T_softmax_norm[i0_4, i1_1])
                Ts.sblock_attr({"axis": 1})
                T_softmax_norm[i0_4, i1_1] = (
                    T.exp(A[i0_4, i1_1] - T_softmax_maxelem[i0_4]) / T_softmax_expsum[i0_4]
                )

    @Ts.function
    def softmax_mn_after_inline_2(
        A: T.Tensor((256, 256), "float32"), T_softmax_norm: T.Tensor((256, 256), "float32")
    ) -> None:
        T_softmax_maxelem = Ts.sblock_alloc_tensor([256], dtype="float32")
        T_softmax_expsum_shared = Ts.sblock_alloc_tensor([256], dtype="float32", scope="shared")
        for i0, i1 in T.grid(256, 256):
            with Ts.sblock("T_softmax_maxelem"):
                i0_1, k = Ts.axis.remap("SR", [i0, i1])
                Ts.reads(A[i0_1, k])
                Ts.writes(T_softmax_maxelem[i0_1])
                with Ts.init():
                    T_softmax_maxelem[i0_1] = T.float32(-3.4028234663852886e38)
                T_softmax_maxelem[i0_1] = T.max(T_softmax_maxelem[i0_1], A[i0_1, k])
        for i0_3 in T.serial(256):
            for ax0, ax1_0 in T.grid(1, 1):
                for ax1_1 in T.parallel(512, annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("T_softmax_expsum"):
                        Ts.where(ax1_0 * 512 + ax1_1 < 256)
                        i0_2 = Ts.axis.spatial(256, i0_3 + ax0)
                        k = Ts.axis.reduce(256, ax1_0 * 512 + ax1_1)
                        Ts.reads(A[i0_2, k], T_softmax_maxelem[i0_2])
                        Ts.writes(T_softmax_expsum_shared[i0_2])
                        with Ts.init():
                            T_softmax_expsum_shared[i0_2] = T.float32(0)
                        T_softmax_expsum_shared[i0_2] = T_softmax_expsum_shared[i0_2] + T.exp(
                            A[i0_2, k] - T_softmax_maxelem[i0_2]
                        )
            for i1_0 in T.serial(1):
                for i1_1 in T.parallel(512, annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("T_softmax_norm"):
                        Ts.where(i1_0 * 512 + i1_1 < 256)
                        i0_4 = Ts.axis.spatial(256, i0_3)
                        i1_1_1 = Ts.axis.spatial(256, i1_0 * 512 + i1_1)
                        Ts.reads(
                            A[i0_4, i1_1_1], T_softmax_maxelem[i0_4], T_softmax_expsum_shared[i0_4]
                        )
                        Ts.writes(T_softmax_norm[i0_4, i1_1_1])
                        Ts.sblock_attr({"axis": 1})
                        T_softmax_norm[i0_4, i1_1_1] = (
                            T.exp(A[i0_4, i1_1_1] - T_softmax_maxelem[i0_4])
                            / T_softmax_expsum_shared[i0_4]
                        )

    @Ts.function
    def softmax_mn_after_inline_3(
        A: T.Tensor((256, 256), "float32"), T_softmax_norm: T.Tensor((256, 256), "float32")
    ) -> None:
        T_softmax_maxelem_shared = Ts.sblock_alloc_tensor([256], dtype="float32", scope="shared")
        T_softmax_expsum_shared = Ts.sblock_alloc_tensor([256], dtype="float32", scope="shared")
        for i0_3 in T.serial(256):
            for ax0, ax1_0 in T.grid(1, 1):
                for ax1_1 in T.parallel(512, annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("T_softmax_maxelem"):
                        Ts.where(ax1_0 * 512 + ax1_1 < 256)
                        i0_1 = Ts.axis.spatial(256, i0_3 + ax0)
                        k = Ts.axis.reduce(256, ax1_0 * 512 + ax1_1)
                        Ts.reads(A[i0_1, k])
                        Ts.writes(T_softmax_maxelem_shared[i0_1])
                        with Ts.init():
                            T_softmax_maxelem_shared[i0_1] = T.float32(-3.4028234663852886e38)
                        T_softmax_maxelem_shared[i0_1] = T.max(
                            T_softmax_maxelem_shared[i0_1], A[i0_1, k]
                        )
            for ax0, ax1_0 in T.grid(1, 1):
                for ax1_1 in T.parallel(512, annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("T_softmax_expsum"):
                        Ts.where(ax1_0 * 512 + ax1_1 < 256)
                        i0_2 = Ts.axis.spatial(256, i0_3 + ax0)
                        k = Ts.axis.reduce(256, ax1_0 * 512 + ax1_1)
                        Ts.reads(A[i0_2, k], T_softmax_maxelem_shared[i0_2])
                        Ts.writes(T_softmax_expsum_shared[i0_2])
                        with Ts.init():
                            T_softmax_expsum_shared[i0_2] = T.float32(0)
                        T_softmax_expsum_shared[i0_2] = T_softmax_expsum_shared[i0_2] + T.exp(
                            A[i0_2, k] - T_softmax_maxelem_shared[i0_2]
                        )
            for i1_0 in T.serial(1):
                for i1_1 in T.parallel(512, annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("T_softmax_norm"):
                        Ts.where(i1_0 * 512 + i1_1 < 256)
                        i0_4 = Ts.axis.spatial(256, i0_3)
                        i1_1_1 = Ts.axis.spatial(256, i1_0 * 512 + i1_1)
                        Ts.reads(
                            A[i0_4, i1_1_1],
                            T_softmax_maxelem_shared[i0_4],
                            T_softmax_expsum_shared[i0_4],
                        )
                        Ts.writes(T_softmax_norm[i0_4, i1_1_1])
                        Ts.sblock_attr({"axis": 1})
                        T_softmax_norm[i0_4, i1_1_1] = (
                            T.exp(A[i0_4, i1_1_1] - T_softmax_maxelem_shared[i0_4])
                            / T_softmax_expsum_shared[i0_4]
                        )

    decision_0 = []  # type: ignore
    decision_1 = [
        ("SampleCategorical", 4),
    ]
    decision_2 = [
        ("SampleCategorical", 7),
    ]
    decision_3 = [
        ("SampleCategorical", 7),
        ("SampleCategorical", 0),
    ]

    mod = Softmax_mn_after_inline
    actual = generate_design_space(
        kind="cuda",
        mod=mod,
        target=Target("nvidia/geforce-rtx-3090", host="llvm"),
        types=ms.schedule_rule.CrossThreadReduction,
    )
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[
            softmax_mn_after_inline_0,
            softmax_mn_after_inline_1,
            softmax_mn_after_inline_2,
            softmax_mn_after_inline_3,
        ],
        expected_decisions=[decision_0, decision_1, decision_2, decision_3],
    )


def test_gpu_batch_norm_bmn():
    @Ts.function
    def batch_norm_bmn_0(A: T.Tensor((1, 512, 512), "float32"), D: T.Tensor(1, "float32")) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        # body
        # with Ts.sblock("root")
        C = Ts.sblock_alloc_tensor([1], dtype="float32")
        for i0, i1, i2 in T.grid(1, 512, 512):
            with Ts.sblock("C"):
                b, i, j = Ts.axis.remap("SRR", [i0, i1, i2])
                Ts.reads(A[b, i, j])
                Ts.writes(C[b])
                with Ts.init():
                    C[b] = T.float32(0)
                C[b] = C[b] + A[b, i, j] * A[b, i, j]
        for i0 in T.serial(1):
            with Ts.sblock("D"):
                b = Ts.axis.spatial(1, i0)
                Ts.reads(C[b])
                Ts.writes(D[b])
                D[b] = T.sqrt(C[b])

    @Ts.function
    def batch_norm_bmn_1(A: T.Tensor((1, 512, 512), "float32"), D: T.Tensor(1, "float32")) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        # body
        # with Ts.sblock("root")
        C_shared = Ts.sblock_alloc_tensor([1], dtype="float32", scope="shared")
        for i0_0 in T.serial(1):
            for ax0, ax1_ax2_fused_0 in T.grid(1, 1024):
                for ax1_ax2_fused_1 in T.parallel(
                    256, annotations={"thread_binding": "threadIdx.x"}
                ):
                    with Ts.sblock("C"):
                        b = Ts.axis.spatial(1, ax0)
                        i = Ts.axis.reduce(512, (ax1_ax2_fused_0 * 256 + ax1_ax2_fused_1) // 512)
                        j = Ts.axis.reduce(512, (ax1_ax2_fused_0 * 256 + ax1_ax2_fused_1) % 512)
                        Ts.reads(A[b, i, j])
                        Ts.writes(C_shared[b])
                        with Ts.init():
                            C_shared[b] = T.float32(0)
                        C_shared[b] = C_shared[b] + A[b, i, j] * A[b, i, j]
            for i0_1 in T.parallel(256, annotations={"thread_binding": "threadIdx.x"}):
                with Ts.sblock("D"):
                    Ts.where(i0_0 * 256 + i0_1 < 1)
                    b = Ts.axis.spatial(1, i0_0 * 256 + i0_1)
                    Ts.reads(C_shared[b])
                    Ts.writes(D[b])
                    D[b] = T.sqrt(C_shared[b])

    decision_0 = []  # type: ignore
    decision_1 = [
        ("SampleCategorical", 6),
    ]

    mod = create_function(te_workload.norm_bmn(B=1, M=512, N=512))
    actual = generate_design_space(
        kind="cuda",
        mod=mod,
        target=Target("nvidia/geforce-rtx-3090", host="llvm"),
        types=ms.schedule_rule.CrossThreadReduction,
    )
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[batch_norm_bmn_0, batch_norm_bmn_1],
        expected_decisions=[decision_0, decision_1],
    )


@Ts.function
def argmax(
    idx: T.Tensor((128, 128), "int32"),
    val: T.Tensor((128, 128), "float32"),
    argmax_v0: T.Tensor((128,), "int32"),
    argmax_v1: T.Tensor((128,), "float32"),
) -> None:
    for i0, i1 in T.grid(128, 128):
        with Ts.sblock("argmax"):
            i = Ts.axis.spatial(128, i0)
            k = Ts.axis.reduce(128, i1)
            Ts.reads(idx[i, k], val[i, k])
            Ts.writes(argmax_v0[i], argmax_v1[i])
            with Ts.init():
                argmax_v0[i] = -1
                argmax_v1[i] = T.min_value(ty="float32")
            v_argmax_v0: T.let[T.int32] = T.Select(
                argmax_v1[i] >= val[i, k], argmax_v0[i], idx[i, k]
            )
            v_argmax_v1: T.let[T.float32] = T.Select(
                argmax_v1[i] >= val[i, k], argmax_v1[i], val[i, k]
            )
            argmax_v0[i] = v_argmax_v0
            argmax_v1[i] = v_argmax_v1


@Ts.function
def argmax_32(
    idx: T.Tensor((1, 32), "int32"),
    val: T.Tensor((1, 32), "float32"),
    argmax_v0: T.Tensor((1,), "int32"),
    argmax_v1: T.Tensor((1,), "float32"),
) -> None:
    for i0, i1 in T.grid(1, 32):
        with Ts.sblock("argmax"):
            i = Ts.axis.spatial(1, i0)
            k = Ts.axis.reduce(32, i1)
            Ts.reads(idx[i, k], val[i, k])
            Ts.writes(argmax_v0[i], argmax_v1[i])
            with Ts.init():
                argmax_v0[i] = -1
                argmax_v1[i] = T.min_value(ty="float32")
            v_argmax_v0: T.let[T.int32] = T.Select(
                argmax_v1[i] >= val[i, k], argmax_v0[i], idx[i, k]
            )
            v_argmax_v1: T.let[T.float32] = T.Select(
                argmax_v1[i] >= val[i, k], argmax_v1[i], val[i, k]
            )
            argmax_v0[i] = v_argmax_v0
            argmax_v1[i] = v_argmax_v1


def test_gpu_argmax():
    @Ts.function
    def argmax_0(
        idx: T.Tensor((128, 128), "int32"),
        val: T.Tensor((128, 128), "float32"),
        argmax_v0: T.Tensor(128, "int32"),
        argmax_v1: T.Tensor(128, "float32"),
    ) -> None:
        # body
        # with Ts.sblock("root")
        for i0, i1 in T.grid(128, 128):
            with Ts.sblock("argmax"):
                i, k = Ts.axis.remap("SR", [i0, i1])
                Ts.reads(idx[i, k], val[i, k])
                Ts.writes(argmax_v0[i], argmax_v1[i])
                with Ts.init():
                    argmax_v0[i] = -1
                    argmax_v1[i] = T.float32(-3.4028234663852886e38)
                v_argmax_v0: T.let[T.int32] = T.Select(
                    argmax_v1[i] >= val[i, k], argmax_v0[i], idx[i, k]
                )
                v_argmax_v1: T.let[T.float32] = T.Select(
                    argmax_v1[i] >= val[i, k], argmax_v1[i], val[i, k]
                )
                argmax_v0[i] = v_argmax_v0
                argmax_v1[i] = v_argmax_v1

    @Ts.function
    def argmax_1(
        idx: T.Tensor((128, 128), "int32"),
        val: T.Tensor((128, 128), "float32"),
        argmax_v0: T.Tensor(128, "int32"),
        argmax_v1: T.Tensor(128, "float32"),
    ) -> None:
        # body
        # with Ts.sblock("root")
        for i0, i1_0 in T.grid(128, 2):
            for i1_1 in T.parallel(64, annotations={"thread_binding": "threadIdx.x"}):
                with Ts.sblock("argmax"):
                    i = Ts.axis.spatial(128, i0)
                    k = Ts.axis.reduce(128, i1_0 * 64 + i1_1)
                    Ts.reads(idx[i, k], val[i, k])
                    Ts.writes(argmax_v0[i], argmax_v1[i])
                    with Ts.init():
                        argmax_v0[i] = -1
                        argmax_v1[i] = T.float32(-3.4028234663852886e38)
                    v_argmax_v0: T.let[T.int32] = T.Select(
                        argmax_v1[i] >= val[i, k], argmax_v0[i], idx[i, k]
                    )
                    v_argmax_v1: T.let[T.float32] = T.Select(
                        argmax_v1[i] >= val[i, k], argmax_v1[i], val[i, k]
                    )
                    argmax_v0[i] = v_argmax_v0
                    argmax_v1[i] = v_argmax_v1

    decision_0 = []  # type: ignore
    decision_1 = [
        ("SampleCategorical", 4),
    ]

    mod = argmax
    actual = generate_design_space(
        kind="cuda",
        mod=mod,
        target=Target("nvidia/geforce-rtx-3090", host="llvm"),
        types=ms.schedule_rule.CrossThreadReduction,
    )
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[argmax_0, argmax_1],
        expected_decisions=[decision_0, decision_1],
    )


def test_gpu_argmax_32():
    @Ts.function
    def argmax_0(
        idx: T.Tensor((1, 32), "int32"),
        val: T.Tensor((1, 32), "float32"),
        argmax_v0: T.Tensor((1,), "int32"),
        argmax_v1: T.Tensor((1,), "float32"),
    ) -> None:
        # body
        # with Ts.sblock("root")
        for i0, i1 in T.grid(1, 32):
            with Ts.sblock("argmax"):
                i, k = Ts.axis.remap("SR", [i0, i1])
                Ts.reads(idx[i, k], val[i, k])
                Ts.writes(argmax_v0[i], argmax_v1[i])
                with Ts.init():
                    argmax_v0[i] = -1
                    argmax_v1[i] = T.float32(-3.4028234663852886e38)
                v_argmax_v0: T.let[T.int32] = T.Select(
                    argmax_v1[i] >= val[i, k], argmax_v0[i], idx[i, k]
                )
                v_argmax_v1: T.let[T.float32] = T.Select(
                    argmax_v1[i] >= val[i, k], argmax_v1[i], val[i, k]
                )
                argmax_v0[i] = v_argmax_v0
                argmax_v1[i] = v_argmax_v1

    @Ts.function
    def argmax_1(
        idx: T.Tensor((1, 32), "int32"),
        val: T.Tensor((1, 32), "float32"),
        argmax_v0: T.Tensor((1,), "int32"),
        argmax_v1: T.Tensor((1,), "float32"),
    ) -> None:
        # body
        # with Ts.sblock("root")
        for i0, i1_0 in T.grid(1, 1):
            for i1_1 in T.parallel(64, annotations={"thread_binding": "threadIdx.x"}):
                with Ts.sblock("argmax"):
                    i = Ts.axis.spatial(1, i0)
                    k = Ts.axis.reduce(32, i1_0 * 64 + i1_1)
                    Ts.where(i1_0 * 64 + i1_1 < 32)
                    Ts.reads(idx[i, k], val[i, k])
                    Ts.writes(argmax_v0[i], argmax_v1[i])
                    with Ts.init():
                        argmax_v0[i] = -1
                        argmax_v1[i] = T.float32(-3.4028234663852886e38)
                    v_argmax_v0: T.let[T.int32] = T.Select(
                        argmax_v1[i] >= val[i, k], argmax_v0[i], idx[i, k]
                    )
                    v_argmax_v1: T.let[T.float32] = T.Select(
                        argmax_v1[i] >= val[i, k], argmax_v1[i], val[i, k]
                    )
                    argmax_v0[i] = v_argmax_v0
                    argmax_v1[i] = v_argmax_v1

    decision_0 = []  # type: ignore
    decision_1 = [
        ("SampleCategorical", 4),
    ]

    mod = argmax_32
    actual = generate_design_space(
        kind="cuda",
        mod=mod,
        target=Target("nvidia/geforce-rtx-3090", host="llvm"),
        types=ms.schedule_rule.CrossThreadReduction,
    )
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[argmax_0, argmax_1],
        expected_decisions=[decision_0, decision_1],
    )


if __name__ == "__main__":
    test_gpu_softmax_mn()
    test_gpu_softmax_mn_after_inline()
    test_gpu_batch_norm_bmn()
    test_gpu_argmax()
    test_gpu_argmax_32()
