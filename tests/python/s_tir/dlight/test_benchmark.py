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
# pylint: disable=missing-docstring
# ruff: noqa: E501, F841

import tempfile

import pytest

pytest.importorskip("cloudpickle")  # tvm.s_tir.dlight.benchmark imports cloudpickle

import tvm_tirx.testing
from tvm.relax import script as R
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir import script as Ts
from tvm.s_tir.dlight.benchmark import (
    benchmark,
    benchmark_function,
    benchmark_relax_func,
    extract_from_relax,
    extract_func_info_from_function,
    extract_function,
)
from tvm.s_tir.meta_schedule.testing.local_rpc import LocalRPC
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T

# The test function uses an undefined symbolic var in Relax.
# In principle, this should be attached to an argument.
# pylint: disable=no-self-argument,invalid-name,line-too-long,no-method-argument
# fmt: off
full1_n = T.dynamic("n", "int64")
full2_n = T.dynamic("n", "int64")
matmul1_n = T.dynamic("n", "int64")
_test_n = T.dynamic("n", "int64")

@I.ir_module(check_well_formed=False)
class Module:
    @Ts.function
    def full1(T_full: T.Tensor((T.int64(1), T.int64(32), T.int64(1), full1_n), 'float16')):
        T.func_attr({"op_pattern": 0, "tirx.noalias": True})

        # with Ts.sblock("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), T.int64(1), full1_n):
            with Ts.sblock("T_full"):
                v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                Ts.reads()
                Ts.writes(T_full[v_ax0, v_ax1, v_ax2, v_ax3])
                T_full[v_ax0, v_ax1, v_ax2, v_ax3] = T.float16(1.0)

    @Ts.function
    def full2(T_full: T.Tensor((T.int64(1), T.int64(32), full2_n, T.int64(128)), 'float16')):
        T.func_attr({"op_pattern": 0, "tirx.noalias": True})

        # with Ts.sblock("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), full2_n, T.int64(128)):
            with Ts.sblock("T_full"):
                v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                Ts.reads()
                Ts.writes(T_full[v_ax0, v_ax1, v_ax2, v_ax3])
                T_full[v_ax0, v_ax1, v_ax2, v_ax3] = T.float16(1.0)

    @Ts.function
    def matmul1(A: T.Tensor((T.int64(1), T.int64(32), T.int64(1), matmul1_n), 'float16'), B: T.Tensor((T.int64(1), T.int64(32), matmul1_n, T.int64(128)), 'float16'), matmul: T.Tensor((T.int64(1), T.int64(32), T.int64(1), T.int64(128)), "float16")):
        T.func_attr({"op_pattern": 4, "tirx.noalias": True})

        # with Ts.sblock("root"):
        for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), T.int64(1), T.int64(128), matmul1_n):
            with Ts.sblock("matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = Ts.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                Ts.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_k, v_i3])
                Ts.writes(matmul[v_i0, v_i1, v_i2, v_i3])
                with Ts.init():
                    matmul[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
                matmul[v_i0, v_i1, v_i2, v_i3] = matmul[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_k, v_i3]

    @R.function
    def test():
        R.func_attr({"tir_var_upper_bound": {"n": 2048}})
        cls = Module
        with R.dataflow():
            lv1 = R.call_tir(cls.full1,(), ty_args=[R.Tensor((1, 32, 1, _test_n), dtype="float16")])
            lv1_1 = R.call_tir(cls.full1,(), ty_args=[R.Tensor((1, 32, 1, _test_n), dtype="float16")])
            lv1_2 = R.call_tir(cls.full1,(), ty_args=[R.Tensor((1, 32, 1, _test_n), dtype="float16")])
            lv2 = R.call_tir(cls.full2,(), ty_args=[R.Tensor((1, 32, _test_n, 128), dtype="float16")])
            lv2_1 = R.call_tir(cls.full2,(), ty_args=[R.Tensor((1, 32, _test_n, 128), dtype="float16")])
            lv3 = R.call_tir(cls.matmul1, (lv1, lv2), ty_args=[R.Tensor((1, 32, 1, 128), dtype="float16")])
            R.output(lv3)
        return lv3

m = T.dynamic("m", "int64")

@Ts.function
def cuda_workload(inp0: T.Tensor((T.int64(1), m, T.int64(4096)), "float32"), inp1: T.Tensor((T.int64(4096), T.int64(4096)), "float32"), matmul: T.Tensor((T.int64(1), m, T.int64(4096)), "float32")):
    T.func_attr({"tirx.is_scheduled": True})

    # with Ts.sblock("root"):
    matmul_reindex_pad_local = Ts.sblock_alloc_tensor((T.int64(1), (m + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(4096)), scope="local")
    inp0_reindex_pad_shared = Ts.sblock_alloc_tensor((T.int64(1), (m + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(4096)), scope="shared")
    inp1_reindex_shared = Ts.sblock_alloc_tensor((T.int64(1), T.int64(4096), T.int64(4096)), scope="shared")
    for ax0 in T.parallel(T.int64(1), annotations={"thread_binding": "blockIdx.z"}):
        for ax1_0 in T.parallel((m + T.int64(31)) // T.int64(32), annotations={"thread_binding": "blockIdx.x"}):
            for ax2_0 in T.parallel(T.int64(64), annotations={"thread_binding": "blockIdx.y"}):
                for ax2_1 in T.parallel(T.int64(1), annotations={"thread_binding": "vthread.y"}):
                    for ax1_1 in T.parallel(T.int64(1), annotations={"thread_binding": "vthread.x"}):
                        for ax2_2 in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.y"}):
                            for ax1_2 in T.parallel(T.int64(8), annotations={"thread_binding": "threadIdx.x", "tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                                for ax2_3_init, ax1_3_init in T.grid(T.int64(4), T.int64(4)):
                                    with Ts.sblock("matmul_init"):
                                        v0 = Ts.axis.spatial(T.int64(1), ax0)
                                        v1 = Ts.axis.spatial((m + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                        v2 = Ts.axis.spatial(T.int64(4096), ax2_0 * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_init)
                                        Ts.reads()
                                        Ts.writes(matmul_reindex_pad_local[T.int64(0), v1, v2])
                                        matmul_reindex_pad_local[T.int64(0), v1, v2] = T.float32(0)
                                for ax3_0 in range(T.int64(256)):
                                    for ax0_ax1_ax2_fused_0 in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.y"}):
                                        for ax0_ax1_ax2_fused_1 in T.parallel(T.int64(8), annotations={"thread_binding": "threadIdx.x"}):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with Ts.sblock("inp0_reindex_pad_shared"):
                                                        v0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = Ts.axis.spatial((m + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = Ts.axis.spatial(T.int64(4096), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        Ts.reads(inp0[v0, v1, v2])
                                                        Ts.writes(inp0_reindex_pad_shared[v0, v1, v2])
                                                        Ts.sblock_attr({"tensor_dim_align": [[0, 1, 8, 2]]})
                                                        inp0_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v1 < m, inp0[v0, v1, v2], T.float32(0))
                                    for ax0_ax1_ax2_fused_0 in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.y"}):
                                        for ax0_ax1_ax2_fused_1 in T.parallel(T.int64(8), annotations={"thread_binding": "threadIdx.x"}):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with Ts.sblock("inp1_reindex_shared"):
                                                        v0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = Ts.axis.spatial(T.int64(4096), ax2_0 * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = Ts.axis.spatial(T.int64(4096), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        Ts.reads(inp1[v2, v1])
                                                        Ts.writes(inp1_reindex_shared[v0, v1, v2])
                                                        Ts.sblock_attr({"tensor_dim_align": [[0, 1, 8, 2]]})
                                                        inp1_reindex_shared[v0, v1, v2] = inp1[v2, v1]
                                    for ax3_1, ax2_3, ax1_3 in T.grid(T.int64(16), T.int64(4), T.int64(4)):
                                        with Ts.sblock("matmul_update"):
                                            v0 = Ts.axis.spatial(T.int64(1), ax0)
                                            v1 = Ts.axis.spatial((m + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                            v2 = Ts.axis.spatial(T.int64(4096), ax2_0 * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3)
                                            v3 = Ts.axis.reduce(T.int64(4096), ax3_0 * T.int64(16) + ax3_1)
                                            Ts.reads(matmul_reindex_pad_local[T.int64(0), v1, v2], inp0_reindex_pad_shared[T.int64(0), v1, v3], inp1_reindex_shared[T.int64(0), v2, v3])
                                            Ts.writes(matmul_reindex_pad_local[T.int64(0), v1, v2])
                                            matmul_reindex_pad_local[T.int64(0), v1, v2] = matmul_reindex_pad_local[T.int64(0), v1, v2] + inp0_reindex_pad_shared[T.int64(0), v1, v3] * inp1_reindex_shared[T.int64(0), v2, v3]
                                for ax0_1, ax1, ax2_0_1 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with Ts.sblock("matmul_reindex_pad_local"):
                                            v0 = Ts.axis.spatial(T.int64(1), ax0_1)
                                            v1 = Ts.axis.spatial((m + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = Ts.axis.spatial(T.int64(4096), ax2_0 * T.int64(64) + ax2_2 * T.int64(4) + ax2_0_1 * T.int64(2) + ax2_1_1)
                                            Ts.reads(matmul_reindex_pad_local[v0, v1, v2])
                                            Ts.writes(matmul[T.int64(0), v1, v2])
                                            if v1 < m:
                                                matmul[T.int64(0), v1, v2] = matmul_reindex_pad_local[v0, v1, v2]
# fmt: on
# pylint: enable=no-self-argument,invalid-name,line-too-long,no-method-argument


@pytest.mark.skip("requires CUDA")
def test_benchmark_function_rpc():
    with LocalRPC() as rpc:
        rpc_config = ms.runner.RPCConfig(
            tracker_host=rpc.tracker_host,
            tracker_port=rpc.tracker_port,
            tracker_key=rpc.tracker_key,
            session_priority=1,
            session_timeout_sec=100,
        )
        input_infos, _, _ = benchmark(
            cuda_workload,
            args=[
                ((1, "m", 4096), "float32"),
                ((4096, 4096), "float32"),
                ((1, "m", 4096), "float32"),
            ],
            dym_var_sample={"m": 128},
            target="nvidia/geforce-rtx-3070",
            rpc_config=rpc_config,
        )
        assert input_infos == [
            ((1, 128, 4096), "float32"),
            ((4096, 4096), "float32"),
            ((1, 128, 4096), "float32"),
        ]


@pytest.mark.skip("requires CUDA")
def test_benchmark_function_local():
    input_infos, _, _ = benchmark(
        cuda_workload,
        args=[
            ((1, "m", 4096), "float32"),
            ((4096, 4096), "float32"),
            ((1, "m", 4096), "float32"),
        ],
        dym_var_sample={"m": 128},
        target="nvidia/geforce-rtx-3070",
    )
    assert input_infos == [
        ((1, 128, 4096), "float32"),
        ((4096, 4096), "float32"),
        ((1, 128, 4096), "float32"),
    ]


@pytest.mark.skip("requires CUDA")
def test_benchmark_function_full_local():
    benchmark_function(cuda_workload, target=tvm_tirx.target.Target("nvidia/geforce-rtx-3070"))


@pytest.mark.skip("requires CUDA")
def test_benchmark_function_full_rpc():
    with LocalRPC() as rpc:
        rpc_config = ms.runner.RPCConfig(
            tracker_host=rpc.tracker_host,
            tracker_port=rpc.tracker_port,
            tracker_key=rpc.tracker_key,
            session_priority=1,
            session_timeout_sec=100,
        )
        benchmark_function(
            cuda_workload,
            target="nvidia/geforce-rtx-3070",
            rpc_config=rpc_config,
            evaluator_config=ms.runner.EvaluatorConfig(
                number=10,
                repeat=10,
                min_repeat_ms=0,
                enable_cpu_cache_flush=False,
            ),
        )


def test_benchmark_relax_func():
    benchmark_relax_func(
        Module, "test", target=tvm_tirx.target.Target({"kind": "llvm", "num-cores": 4})
    )


def test_extract_function_full1():
    print(
        extract_function(
            model_name="TEST",
            relax_func_name="test",
            function_name="full1",
            func=Module["full1"],  # type: ignore
            func_args=[((1, 32, 1, "n"), "float16")],
            dym_var_dict={"n": "int32"},
            weight=2,
            sample_number=10,
            target={"kind": "llvm", "num-cores": 4},
        )
    )


def test_extract_function_matmul1():
    print(
        extract_function(
            model_name="TEST",
            relax_func_name="test",
            function_name="matmul1",
            func=Module["matmul1"],  # type: ignore
            weight=2,
            sample_number=10,
            target={"kind": "llvm", "num-cores": 4},
        )
    )


def test_extract_from_relax():
    with tempfile.TemporaryDirectory() as filepath:
        extract_from_relax(
            Module,
            "TEST",
            file_path=filepath,
            target=tvm_tirx.target.Target({"kind": "llvm", "num-cores": 4}),
        )


def test_extract_func_info_from_function():
    assert (
        str(extract_func_info_from_function(cuda_workload))
        == "([((1, m, 4096), 'float32'), ((4096, 4096), 'float32'), ((1, m, 4096), 'float32')], {'m': 'int64'})"
    )
    assert (
        str(extract_func_info_from_function(Module["full1"]))
        == "([((1, 32, 1, n), 'float16')], {'n': 'int64'})"
    )
    assert (
        str(extract_func_info_from_function(Module["matmul1"]))
        == "([((1, 32, 1, n), 'float16'), ((1, 32, n, 128), 'float16'), ((1, 32, 1, 128), 'float16')], {'n': 'int64'})"
    )
    assert (
        str(extract_func_info_from_function(Module["full2"]))
        == "([((1, 32, n, 128), 'float16')], {'n': 'int64'})"
    )


if __name__ == "__main__":
    tvm_tirx.testing.main()
