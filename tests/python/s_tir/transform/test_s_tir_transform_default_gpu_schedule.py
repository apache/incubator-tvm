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
# pylint: disable=invalid-name,,missing-function-docstring
# ruff: noqa: E501, F841
import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import script as Ts
from tvm.s_tir.transform import DefaultGPUSchedule
from tvm_tirx.script import tirx as T


def test_broadcast_to_symbolic():
    # pylint: disable=no-self-argument,missing-class-docstring,line-too-long
    # fmt: off
    x_0 = T.dynamic("x_0", "int64")
    x_1 = T.dynamic("x_1", "int64")

    @tvm_tirx.script.ir_module
    class Before:
        @Ts.function
        def broadcast_to(
            rxplaceholder: T.Tensor((T.int64(3), T.int64(1)), "float32"),
            T_broadcast_to: T.Tensor((x_0, x_1), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            for ax0, ax1 in T.grid(x_0, x_1):
                with Ts.sblock("T_broadcast_to"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(rxplaceholder[v_ax0, T.int64(0)])
                    Ts.writes(T_broadcast_to[v_ax0, v_ax1])
                    T_broadcast_to[v_ax0, v_ax1] = rxplaceholder[v_ax0, T.int64(0)]

    x_0 = T.dynamic("x_0", "int64")
    x_1 = T.dynamic("x_1", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function
        def broadcast_to(rxplaceholder: T.Tensor((T.int64(3), T.int64(1)), "float32"), T_broadcast_to: T.Tensor((x_0, x_1), "float32")):
            T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})

            for ax0_ax1_fused_1 in T.parallel(T.int64(256), annotations={"thread_binding": "blockIdx.x"}):
                for ax0_ax1_fused_2 in T.parallel(T.int64(1024), annotations={"thread_binding": "threadIdx.x"}):
                    for ax0_ax1_fused_0 in range((x_0 * x_1 + T.int64(262143)) // T.int64(262144)):
                        with Ts.sblock("T_broadcast_to"):
                            v_ax0 = Ts.axis.spatial(x_0, (ax0_ax1_fused_0 * T.int64(262144) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2) // x_1)
                            v_ax1 = Ts.axis.spatial(x_1, (ax0_ax1_fused_0 * T.int64(262144) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2) % x_1)
                            Ts.where((ax0_ax1_fused_0 * T.int64(256) + ax0_ax1_fused_1) * T.int64(1024) + ax0_ax1_fused_2 < x_0 * x_1)
                            T_broadcast_to[v_ax0, v_ax1] = rxplaceholder[v_ax0, T.int64(0)]
    # fmt: on
    # pylint: enable=no-self-argument,missing-class-docstring,line-too-long
    target = tvm_tirx.target.Target("nvidia/geforce-rtx-3070")
    with tvm_tirx.ir.transform.PassContext(opt_level=3):
        After = DefaultGPUSchedule(target=target)(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_matmul():
    # pylint: disable=no-self-argument,missing-class-docstring,line-too-long
    # fmt: off
    @tvm_tirx.script.ir_module
    class Before:
        @Ts.function
        def matmul(
            A: T.Tensor((32, 32), "float16"),
            B: T.Tensor((32, 32), "float16"),
            C: T.Tensor((32, 32), "float16"),
        ):
            T.func_attr({"global_symbol": "main", "tirx.noalias": True})
            # with Ts.sblock("root"):
            for i, j, k in T.grid(32, 32, 32):
                with Ts.sblock("C"):
                    v_i, v_j, v_k = Ts.axis.remap("SSR", [i, j, k])
                    Ts.reads(A[v_i, v_k], B[v_k, v_j])
                    Ts.writes(C[v_i, v_j])
                    with Ts.init():
                        C[v_i, v_j] = T.float16(0)
                    C[v_i, v_j] = C[v_i, v_j] + A[v_i, v_k] * B[v_k, v_j]

        @Ts.function
        def matmul_gpu(
            A: T.Tensor((32, 32), "float16"),
            B: T.Tensor((32, 32), "float16"),
            C: T.Tensor((32, 32), "float16"),
        ):
            T.func_attr({"global_symbol": "main",
                         "target": T.target({"arch": "sm_86",
                                             "keys": ["cuda", "gpu"],
                                             "kind": "cuda",
                                             "max_num_threads": 1024,
                                             "tag": "",
                                             "thread_warp_size": 32}),
                         "tirx.noalias": True})
            # with Ts.sblock("root"):
            for i, j, k in T.grid(32, 32, 32):
                with Ts.sblock("C"):
                    v_i, v_j, v_k = Ts.axis.remap("SSR", [i, j, k])
                    Ts.reads(A[v_i, v_k], B[v_k, v_j])
                    Ts.writes(C[v_i, v_j])
                    with Ts.init():
                        C[v_i, v_j] = T.float16(0)
                    C[v_i, v_j] = C[v_i, v_j] + A[v_i, v_k] * B[v_k, v_j]

        @Ts.function
        def matmul_cpu(
            A: T.Tensor((32, 32), "float16"),
            B: T.Tensor((32, 32), "float16"),
            C: T.Tensor((32, 32), "float16"),
        ):
            T.func_attr({"global_symbol": "main",
                         "target": T.target({"keys": ["cpu"], "kind": "llvm", "tag": ""}),
                        "tirx.noalias": True})
            # with Ts.sblock("root"):
            for i, j, k in T.grid(32, 32, 32):
                with Ts.sblock("C"):
                    v_i, v_j, v_k = Ts.axis.remap("SSR", [i, j, k])
                    Ts.reads(A[v_i, v_k], B[v_k, v_j])
                    Ts.writes(C[v_i, v_j])
                    with Ts.init():
                        C[v_i, v_j] = T.float16(0)
                    C[v_i, v_j] = C[v_i, v_j] + A[v_i, v_k] * B[v_k, v_j]

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function
        def matmul(
            A: T.Tensor((32, 32), "float16"),
            B: T.Tensor((32, 32), "float16"),
            C: T.Tensor((32, 32), "float16"),
        ):
            T.func_attr({"tirx.is_scheduled": True, "global_symbol": "main", "tirx.noalias": True})
            # with Ts.sblock("root"):
            for i_j_fused_0 in T.parallel(1, annotations={"thread_binding": "blockIdx.x"}):
                for i_j_fused_1 in T.parallel(1024, annotations={"thread_binding": "threadIdx.x"}):
                    for k in range(32):
                        with Ts.sblock("C"):
                            v_i = Ts.axis.spatial(
                                32, (i_j_fused_0 * 1024 + i_j_fused_1) // 32
                            )
                            v_j = Ts.axis.spatial(
                                32, (i_j_fused_0 * 1024 + i_j_fused_1) % 32
                            )
                            v_k = Ts.axis.reduce(32, k)
                            Ts.reads(A[v_i, v_k], B[v_k, v_j])
                            Ts.writes(C[v_i, v_j])
                            with Ts.init():
                                C[v_i, v_j] = T.float16(0)
                            C[v_i, v_j] = C[v_i, v_j] + A[v_i, v_k] * B[v_k, v_j]

        @Ts.function
        def matmul_cpu(A: T.Tensor((32, 32), "float16"), B: T.Tensor((32, 32), "float16"), C: T.Tensor((32, 32), "float16")):
            T.func_attr({"global_symbol": "main", "target": T.target({"keys": ["cpu"], "kind": "llvm", "tag": ""}), "tirx.is_scheduled": True, "tirx.noalias": True})
            # with Ts.sblock("root"):
            for i, j, k in T.grid(32, 32, 32):
                with Ts.sblock("C"):
                    v_i, v_j, v_k = Ts.axis.remap("SSR", [i, j, k])
                    Ts.reads(A[v_i, v_k], B[v_k, v_j])
                    Ts.writes(C[v_i, v_j])
                    with Ts.init():
                        C[v_i, v_j] = T.float16(0)
                    C[v_i, v_j] = C[v_i, v_j] + A[v_i, v_k] * B[v_k, v_j]

        @Ts.function
        def matmul_gpu(A: T.Tensor((32, 32), "float16"), B: T.Tensor((32, 32), "float16"), C: T.Tensor((32, 32), "float16")):
            T.func_attr({"global_symbol": "main", "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "tirx.is_scheduled": True, "tirx.noalias": True})
            # with Ts.sblock("root"):
            for i_j_fused_0 in T.parallel(1, annotations={"thread_binding": "blockIdx.x"}):
                for i_j_fused_1 in T.parallel(1024, annotations={"thread_binding": "threadIdx.x"}):
                    for k in range(32):
                        with Ts.sblock("C"):
                            v_i = Ts.axis.spatial(32, (i_j_fused_0 * 1024 + i_j_fused_1) // 32)
                            v_j = Ts.axis.spatial(32, (i_j_fused_0 * 1024 + i_j_fused_1) % 32)
                            v_k = Ts.axis.reduce(32, k)
                            Ts.reads(A[v_i, v_k], B[v_k, v_j])
                            Ts.writes(C[v_i, v_j])
                            with Ts.init():
                                C[v_i, v_j] = T.float16(0)
                            C[v_i, v_j] = C[v_i, v_j] + A[v_i, v_k] * B[v_k, v_j]
    # fmt: on
    # pylint: enable=no-self-argument,missing-class-docstring,line-too-long
    target = tvm_tirx.target.Target("nvidia/geforce-rtx-3070")
    with tvm_tirx.ir.transform.PassContext(opt_level=3):
        After = DefaultGPUSchedule(target=target)(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_add():
    # pylint: disable=no-self-argument,missing-class-docstring,line-too-long
    # fmt: off
    @tvm_tirx.script.ir_module
    class Before:
        @Ts.function
        def add(rxplaceholder: T.Tensor((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_add: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with Ts.sblock("T_add"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    Ts.writes(T_add[ax0, ax1, ax2, ax3])
                    T_add[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] + rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function
        def add(
            rxplaceholder: T.Tensor((T.int64(1), T.int64(2), T.int64(3)), "float32"),
            rxplaceholder_1: T.Tensor(
                (T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"
            ),
            T_add: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "float32"),
        ):
            T.func_attr({"tirx.is_scheduled": True,  "tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0_i1_i2_i3_fused_0 in T.parallel(T.int64(1), annotations={"thread_binding": "blockIdx.x"}):
                for i0_i1_i2_i3_fused_1 in T.parallel(T.int64(72), annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("T_add"):
                        ax0 = Ts.axis.spatial(
                            T.int64(4),
                            (i0_i1_i2_i3_fused_0 * T.int64(72) + i0_i1_i2_i3_fused_1)
                            // T.int64(18),
                        )
                        ax1 = Ts.axis.spatial(
                            T.int64(3),
                            (i0_i1_i2_i3_fused_0 * T.int64(72) + i0_i1_i2_i3_fused_1)
                            % T.int64(18)
                            // T.int64(6),
                        )
                        ax2 = Ts.axis.spatial(
                            T.int64(2),
                            (i0_i1_i2_i3_fused_0 * T.int64(72) + i0_i1_i2_i3_fused_1)
                            % T.int64(6)
                            // T.int64(3),
                        )
                        ax3 = Ts.axis.spatial(
                            T.int64(3),
                            (i0_i1_i2_i3_fused_0 * T.int64(72) + i0_i1_i2_i3_fused_1)
                            % T.int64(3),
                        )
                        Ts.reads(
                            rxplaceholder[T.int64(0), ax2, ax3],
                            rxplaceholder_1[ax0, ax1, ax2, T.int64(0)],
                        )
                        Ts.writes(T_add[ax0, ax1, ax2, ax3])
                        T_add[ax0, ax1, ax2, ax3] = (
                            rxplaceholder[T.int64(0), ax2, ax3]
                            + rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
                        )

    # fmt: on
    # pylint: enable=no-self-argument,missing-class-docstring,line-too-long
    target = tvm_tirx.target.Target("nvidia/geforce-rtx-3070")
    with tvm_tirx.ir.transform.PassContext(opt_level=3):
        After = DefaultGPUSchedule(target=target)(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_full():
    # pylint: disable=no-self-argument,missing-class-docstring,line-too-long
    # fmt: off
    @tvm_tirx.script.ir_module
    class Before:
        @Ts.function
        def full(rxplaceholder: T.Tensor((), "int32"), T_full: T.Tensor((T.int64(2), T.int64(3)), "int32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_full"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[()])
                    Ts.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = rxplaceholder[()]

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function
        def full(
            rxplaceholder: T.Tensor((), "int32"),
            T_full: T.Tensor((T.int64(2), T.int64(3)), "int32"),
        ):
            T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0_i1_fused_0 in T.parallel(T.int64(1), annotations={"thread_binding": "blockIdx.x"}):
                for i0_i1_fused_1 in T.parallel(T.int64(6), annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("T_full"):
                        ax0 = Ts.axis.spatial(
                            T.int64(2),
                            (i0_i1_fused_0 * T.int64(6) + i0_i1_fused_1) // T.int64(3),
                        )
                        ax1 = Ts.axis.spatial(
                            T.int64(3),
                            (i0_i1_fused_0 * T.int64(6) + i0_i1_fused_1) % T.int64(3),
                        )
                        Ts.reads(rxplaceholder[()])
                        Ts.writes(T_full[ax0, ax1])
                        T_full[ax0, ax1] = rxplaceholder[()]

    # fmt: on
    # pylint: enable=no-self-argument,missing-class-docstring,line-too-long
    target = tvm_tirx.target.Target("nvidia/geforce-rtx-3070")
    with tvm_tirx.ir.transform.PassContext(opt_level=3):
        After = DefaultGPUSchedule(target=target)(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_scheduled():
    # pylint: disable=no-self-argument,missing-class-docstring,line-too-long
    # fmt: off

    @tvm_tirx.script.ir_module
    class Scheduled:
        @Ts.function
        def full(
            rxplaceholder: T.Tensor((), "int32"),
            T_full: T.Tensor((T.int64(2), T.int64(3)), "int32"),
        ):
            T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0_i1_fused_0 in T.parallel(T.int64(1), annotations={"thread_binding": "blockIdx.x"}):
                for i0_i1_fused_1 in T.parallel(T.int64(6), annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("T_full"):
                        ax0 = Ts.axis.spatial(
                            T.int64(2),
                            (i0_i1_fused_0 * T.int64(6) + i0_i1_fused_1) // T.int64(3),
                        )
                        ax1 = Ts.axis.spatial(
                            T.int64(3),
                            (i0_i1_fused_0 * T.int64(6) + i0_i1_fused_1) % T.int64(3),
                        )
                        Ts.reads(rxplaceholder[()])
                        Ts.writes(T_full[ax0, ax1])
                        T_full[ax0, ax1] = rxplaceholder[()]

    # fmt: on
    # pylint: enable=no-self-argument,missing-class-docstring,line-too-long
    target = tvm_tirx.target.Target("nvidia/geforce-rtx-3070")
    with tvm_tirx.ir.transform.PassContext(opt_level=3):
        # should do nothing
        After = DefaultGPUSchedule(target=target)(Scheduled)
    tvm_tirx.ir.assert_structural_equal(After, Scheduled)


def test_multiple():
    # pylint: disable=no-self-argument,missing-class-docstring,line-too-long
    # fmt: off
    @tvm_tirx.script.ir_module
    class Before:
        @Ts.function
        def add(rxplaceholder: T.Tensor((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_add: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with Ts.sblock("T_add"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    Ts.writes(T_add[ax0, ax1, ax2, ax3])
                    T_add[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] + rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]

        @Ts.function
        def full(rxplaceholder: T.Tensor((), "int32"), T_full: T.Tensor((T.int64(2), T.int64(3)), "int32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_full"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[()])
                    Ts.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = rxplaceholder[()]

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function
        def add(
            rxplaceholder: T.Tensor((T.int64(1), T.int64(2), T.int64(3)), "float32"),
            rxplaceholder_1: T.Tensor(
                (T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"
            ),
            T_add: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "float32"),
        ):
            T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0_i1_i2_i3_fused_0 in T.parallel(T.int64(1), annotations={"thread_binding": "blockIdx.x"}):
                for i0_i1_i2_i3_fused_1 in T.parallel(T.int64(72), annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("T_add"):
                        ax0 = Ts.axis.spatial(
                            T.int64(4),
                            (i0_i1_i2_i3_fused_0 * T.int64(72) + i0_i1_i2_i3_fused_1)
                            // T.int64(18),
                        )
                        ax1 = Ts.axis.spatial(
                            T.int64(3),
                            (i0_i1_i2_i3_fused_0 * T.int64(72) + i0_i1_i2_i3_fused_1)
                            % T.int64(18)
                            // T.int64(6),
                        )
                        ax2 = Ts.axis.spatial(
                            T.int64(2),
                            (i0_i1_i2_i3_fused_0 * T.int64(72) + i0_i1_i2_i3_fused_1)
                            % T.int64(6)
                            // T.int64(3),
                        )
                        ax3 = Ts.axis.spatial(
                            T.int64(3),
                            (i0_i1_i2_i3_fused_0 * T.int64(72) + i0_i1_i2_i3_fused_1)
                            % T.int64(3),
                        )
                        Ts.reads(
                            rxplaceholder[T.int64(0), ax2, ax3],
                            rxplaceholder_1[ax0, ax1, ax2, T.int64(0)],
                        )
                        Ts.writes(T_add[ax0, ax1, ax2, ax3])
                        T_add[ax0, ax1, ax2, ax3] = (
                            rxplaceholder[T.int64(0), ax2, ax3]
                            + rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
                        )

        @Ts.function
        def full(
            rxplaceholder: T.Tensor((), "int32"),
            T_full: T.Tensor((T.int64(2), T.int64(3)), "int32"),
        ):
            T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0_i1_fused_0 in T.parallel(T.int64(1), annotations={"thread_binding": "blockIdx.x"}):
                for i0_i1_fused_1 in T.parallel(T.int64(6), annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("T_full"):
                        ax0 = Ts.axis.spatial(
                            T.int64(2),
                            (i0_i1_fused_0 * T.int64(6) + i0_i1_fused_1) // T.int64(3),
                        )
                        ax1 = Ts.axis.spatial(
                            T.int64(3),
                            (i0_i1_fused_0 * T.int64(6) + i0_i1_fused_1) % T.int64(3),
                        )
                        Ts.reads(rxplaceholder[()])
                        Ts.writes(T_full[ax0, ax1])
                        T_full[ax0, ax1] = rxplaceholder[()]
    # fmt: on
    # pylint: enable=no-self-argument,missing-class-docstring,line-too-long
    target = tvm_tirx.target.Target("nvidia/geforce-rtx-3070")
    with tvm_tirx.ir.transform.PassContext(opt_level=3):
        After = DefaultGPUSchedule(target=target)(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_add_on_metal():
    # pylint: disable=no-self-argument,missing-class-docstring,line-too-long
    # fmt: off
    @tvm_tirx.script.ir_module
    class Before:
        @Ts.function
        def add(rxplaceholder: T.Tensor((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_add: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with Ts.sblock("T_add"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    Ts.writes(T_add[ax0, ax1, ax2, ax3])
                    T_add[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] + rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function
        def add(rxplaceholder: T.Tensor((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_add: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
            for i0_i1_i2_i3_fused_0 in T.parallel(T.int64(1), annotations={"thread_binding": "blockIdx.x"}):
                for i0_i1_i2_i3_fused_1 in T.parallel(T.int64(72), annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("T_add"):
                        ax0 = Ts.axis.spatial(T.int64(4), (i0_i1_i2_i3_fused_0 * T.int64(72) + i0_i1_i2_i3_fused_1) // T.int64(18))
                        ax1 = Ts.axis.spatial(T.int64(3), (i0_i1_i2_i3_fused_0 * T.int64(72) + i0_i1_i2_i3_fused_1) % T.int64(18) // T.int64(6))
                        ax2 = Ts.axis.spatial(T.int64(2), (i0_i1_i2_i3_fused_0 * T.int64(72) + i0_i1_i2_i3_fused_1) % T.int64(6) // T.int64(3))
                        ax3 = Ts.axis.spatial(T.int64(3), (i0_i1_i2_i3_fused_0 * T.int64(72) + i0_i1_i2_i3_fused_1) % T.int64(3))
                        Ts.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                        Ts.writes(T_add[ax0, ax1, ax2, ax3])
                        T_add[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] + rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
    # fmt: on
    # pylint: enable=no-self-argument,missing-class-docstring,line-too-long
    target = tvm_tirx.target.Target("apple/m1-gpu")
    with tvm_tirx.ir.transform.PassContext(opt_level=0):
        mod = DefaultGPUSchedule(target=target)(Before)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_scalar_add():
    # pylint: disable=no-self-argument,missing-class-docstring,line-too-long
    # fmt: off
    @tvm_tirx.script.ir_module
    class Before:
        @Ts.function
        def add(rxplaceholder: T.Tensor((), "int64"), T_add: T.Tensor((), "int64")):
            T.func_attr({"tirx.noalias": True})
            with Ts.sblock("T_add"):
                vi = Ts.axis.spatial(1, T.int64(0))
                Ts.reads(rxplaceholder[()])
                Ts.writes(T_add[()])
                T_add[()] = rxplaceholder[()] + T.int64(1)

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function
        def add(rxplaceholder: T.Tensor((), "int64"), T_add: T.Tensor((), "int64")):
            T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
            # with Ts.sblock("root"):
            for u_fused_0 in T.parallel(1, annotations={"thread_binding": "blockIdx.x"}):
                for u_fused_1 in T.parallel(1, annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("T_add"):
                        vi = Ts.axis.spatial(1, T.int64(0))
                        Ts.reads(rxplaceholder[()])
                        Ts.writes(T_add[()])
                        T_add[()] = rxplaceholder[()] + T.int64(1)
    # fmt: on
    # pylint: enable=no-self-argument,missing-class-docstring,line-too-long
    target = tvm_tirx.target.Target("nvidia/geforce-rtx-3070")
    with tvm_tirx.ir.transform.PassContext(opt_level=0):
        mod = DefaultGPUSchedule(target=target)(Before)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_sum():
    # sum has two reduction axes and no spatial axis
    # pylint: disable=no-self-argument,missing-class-docstring,line-too-long
    # fmt: off
    @tvm_tirx.script.ir_module
    class Before:
        @Ts.function
        def sum(A: T.Tensor((T.int64(2), T.int64(2)), "float64"), A_red: T.Tensor((), "float64")):
            for k0, k1 in T.grid(T.int64(2), T.int64(2)):
                with Ts.sblock("A_red"):
                    v_k0, v_k1 = Ts.axis.remap("RR", [k0, k1])
                    with Ts.init():
                        A_red[()] = T.float64(0)
                    A_red[()] = A_red[()] + A[v_k0, v_k1]

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function
        def sum(A: T.Tensor((T.int64(2), T.int64(2)), "float64"), A_red: T.Tensor((), "float64")):
            T.func_attr({"tirx.is_scheduled": True})
            # with Ts.sblock("root"):
            for u_fused_0 in T.parallel(1, annotations={"thread_binding": "blockIdx.x"}):
                for u_fused_1 in T.parallel(1, annotations={"thread_binding": "threadIdx.x"}):
                    for k0, k1 in T.grid(T.int64(2), T.int64(2)):
                        with Ts.sblock("A_red"):
                            v_k0, v_k1 = Ts.axis.remap("RR", [k0, k1])
                            Ts.reads(A[v_k0, v_k1])
                            Ts.writes(A_red[()])
                            with Ts.init():
                                A_red[()] = T.float64(0)
                            A_red[()] = A_red[()] + A[v_k0, v_k1]
    # fmt: on
    # pylint: enable=no-self-argument,missing-class-docstring,line-too-long
    target = tvm_tirx.target.Target("nvidia/geforce-rtx-3070")
    with tvm_tirx.ir.transform.PassContext(opt_level=0):
        mod = DefaultGPUSchedule(target=target)(Before)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_scalar_block_no_loops():
    # A Function whose body is a bare SBlockRealize (e.g. a fully-scalar op)
    # used to crash DefaultGPUSchedule with "Cannot add loops on top of the
    # root block" because the realized block was the function's root sref.
    # pylint: disable=no-self-argument,missing-class-docstring,line-too-long
    # fmt: off
    @tvm_tirx.script.ir_module
    class Before:
        @Ts.function
        def scalar_add(a: T.Tensor((), "float32"), b: T.Tensor((), "float32"), c: T.Tensor((), "float32")):
            with Ts.sblock("scalar_add"):
                c[()] = a[()] + b[()]

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function
        def scalar_add(a: T.Tensor((), "float32"), b: T.Tensor((), "float32"), c: T.Tensor((), "float32")):
            T.func_attr({"tirx.is_scheduled": True})
            # with Ts.sblock("root"):
            for u_fused_0 in T.parallel(1, annotations={"thread_binding": "blockIdx.x"}):
                for u_fused_1 in T.parallel(1, annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("scalar_add"):
                        vu = Ts.axis.spatial(1, 0)
                        Ts.reads()
                        Ts.writes()
                        c[()] = a[()] + b[()]
    # fmt: on
    # pylint: enable=no-self-argument,missing-class-docstring,line-too-long
    target = tvm_tirx.target.Target("nvidia/geforce-rtx-3070")
    with tvm_tirx.ir.transform.PassContext(opt_level=0):
        mod = DefaultGPUSchedule(target=target)(Before)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
