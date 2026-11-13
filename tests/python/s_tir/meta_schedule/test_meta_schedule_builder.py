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
# ruff: noqa: F401
"""Test Meta Schedule Builder"""

import os
import sys
import time

import pytest
import tvm_ffi
from tvm_ffi import register_global_func

import tvm
import tvm_tirx.testing
from tvm.s_tir import script as Ts
from tvm.s_tir.meta_schedule.builder import (
    BuilderInput,
    BuilderResult,
    LocalBuilder,
    PyBuilder,
)
from tvm_tirx import script
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target

# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,missing-docstring


@script.ir_module
class MatmulModule:
    @Ts.function
    def matmul(
        A: T.Tensor((1024, 1024), "float32"),
        B: T.Tensor((1024, 1024), "float32"),
        C: T.Tensor((1024, 1024), "float32"),
    ) -> None:  # pylint: disable=no-self-argument
        T.func_attr({"global_symbol": "matmul", "tirx.noalias": True})

        for i, j, k in T.grid(1024, 1024, 1024):
            with Ts.sblock("matmul"):
                vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                with Ts.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@script.ir_module
class MatmulReluModule:
    @Ts.function
    def matmul_relu(  # pylint: disable=no-self-argument
        A: T.Tensor((1024, 1024), "float32"),
        B: T.Tensor((1024, 1024), "float32"),
        D: T.Tensor((1024, 1024), "float32"),
    ) -> None:
        T.func_attr({"global_symbol": "matmul_relu", "tirx.noalias": True})

        C = Ts.sblock_alloc_tensor((1024, 1024), "float32")
        for i, j, k in T.grid(1024, 1024, 1024):
            with Ts.sblock("matmul"):
                vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                with Ts.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(1024, 1024):
            with Ts.sblock("relu"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                D[vi, vj] = T.max(C[vi, vj], 0.0)


@script.ir_module
class BatchMatmulModule:
    @Ts.function
    def batch_matmul(  # pylint: disable=no-self-argument
        A: T.Tensor([16, 128, 128], "float32"),
        B: T.Tensor([16, 128, 128], "float32"),
        C: T.Tensor([16, 128, 128], "float32"),
    ) -> None:
        T.func_attr({"global_symbol": "batch_matmul", "tirx.noalias": True})

        for n, i, j, k in T.grid(16, 128, 128, 128):
            with Ts.sblock("update"):
                vn, vi, vj, vk = Ts.axis.remap("SSSR", [n, i, j, k])
                with Ts.init():
                    C[vn, vi, vj] = 0.0
                C[vn, vi, vj] = C[vn, vi, vj] + A[vn, vi, vk] * B[vn, vj, vk]


# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,missing-docstring


def _check_build_results(builder_results: list[BuilderResult]):
    """Simple check whether the build is successful"""
    for result in builder_results:
        artifact_path = result.artifact_path
        error_msg = result.error_msg
        assert artifact_path is not None
        assert error_msg is None
        os.remove(artifact_path)
        os.rmdir(os.path.dirname(artifact_path))


@pytest.mark.skip("Tuning test - launches runner")
def test_meta_schedule_single_build():
    """Test meta schedule builder for a single build"""
    mod = MatmulModule
    builder = LocalBuilder()
    builder_inputs = [BuilderInput(mod, Target("llvm"))]
    builder_results = builder.build(builder_inputs)
    assert len(builder_results) == len(builder_inputs)
    _check_build_results(builder_results)


@pytest.mark.skip("Tuning test - launches runner")
def test_meta_schedule_multiple_build():
    """Test meta schedule builder for multiple builds"""
    builder = LocalBuilder()
    builder_inputs = [
        BuilderInput(MatmulModule, Target("llvm")),
        BuilderInput(MatmulReluModule, Target("llvm")),
        BuilderInput(BatchMatmulModule, Target("llvm")),
    ]
    builder_results = builder.build(builder_inputs)
    assert len(builder_results) == len(builder_inputs)
    _check_build_results(builder_results)


def test_meta_schedule_error_handle_test_builder():
    """Test the error handing during building"""

    class TestBuilder(PyBuilder):
        def build(  # pylint: disable=no-self-use
            self,
            build_inputs: list[BuilderInput],
        ) -> list[BuilderResult]:
            return [BuilderResult(None, "error") for w in build_inputs]

    builder = TestBuilder()
    builder_inputs = [
        BuilderInput(MatmulModule, Target("llvm")),
        BuilderInput(MatmulReluModule, Target("llvm")),
        BuilderInput(BatchMatmulModule, Target("llvm")),
    ]
    builder_results = builder.build(builder_inputs)
    assert len(builder_results) == len(builder_inputs)
    for result in builder_results:
        artifact_path = result.artifact_path
        error_msg = result.error_msg
        assert artifact_path is None
        assert error_msg == "error"


@pytest.mark.skip("Tuning test - launches runner")
def test_meta_schedule_error_handle_build_func():
    """Test the error handing during building"""

    def initializer():
        @register_global_func("meta_schedule.builder.test_build")
        def test_build(mod: tvm_ffi.Module, target: Target, _) -> None:  # pylint: disable=unused-variable
            raise ValueError("Builder intended Test Error (build func).")

    builder = LocalBuilder(f_build="meta_schedule.builder.test_build", initializer=initializer)
    builder_inputs = [BuilderInput(MatmulModule, Target("llvm"))]
    builder_results = builder.build(builder_inputs)
    assert len(builder_results) == len(builder_inputs)
    for result in builder_results:
        artifact_path = result.artifact_path
        error_msg = result.error_msg
        assert artifact_path is None
        assert error_msg.startswith("LocalBuilder: An exception occurred")


@pytest.mark.skip("Tuning test - launches runner")
def test_meta_schedule_error_handle_export_func():
    """Test the error handing during building"""

    def initializer():
        @register_global_func("meta_schedule.builder.test_export")
        def test_build(mod: tvm_ffi.Module) -> str:  # pylint: disable=unused-variable
            raise ValueError("Builder intended Test Error (export func).")

    builder = LocalBuilder(f_export="meta_schedule.builder.test_export", initializer=initializer)
    builder_inputs = [BuilderInput(MatmulModule, Target("llvm"))]
    builder_results = builder.build(builder_inputs)
    assert len(builder_results) == len(builder_inputs)
    for result in builder_results:
        artifact_path = result.artifact_path
        error_msg = result.error_msg
        assert artifact_path is None
        assert error_msg.startswith("LocalBuilder: An exception occurred")


@pytest.mark.skip("Tuning test - launches runner")
def test_meta_schedule_error_handle_time_out():
    """Test the error handing time out during building"""

    def initializer():
        @register_global_func("meta_schedule.builder.test_time_out")
        def timeout_build(mod, target, _):  # pylint: disable=unused-argument, unused-variable
            time.sleep(2)

    builder = LocalBuilder(
        timeout_sec=1,
        f_build="meta_schedule.builder.test_time_out",
        initializer=initializer,
    )
    builder_inputs = [BuilderInput(MatmulModule, Target("llvm"))]
    builder_results = builder.build(builder_inputs)
    assert len(builder_results) == len(builder_inputs)
    for result in builder_results:
        artifact_path = result.artifact_path
        error_msg = result.error_msg
        assert artifact_path is None
        assert error_msg.startswith("LocalBuilder: Timeout")


def test_meta_schedule_missing_build_func():
    pytest.importorskip("cloudpickle")
    with pytest.raises(ValueError):
        LocalBuilder(f_build="wrong-name")


if __name__ == "__main__":
    tvm_tirx.testing.main()
