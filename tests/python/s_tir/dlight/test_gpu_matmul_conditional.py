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
"""Regression coverage for matmul consumer chains with conditional epilogues."""

import shutil

import numpy as np
import pytest

import tvm
import tvm_tirx
from tvm import relax, s_tir
from tvm.s_tir import dlight as dl
from tvm.s_tir import script as Ts
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target
from tvm_tirx.testing import env


def _conditional_epilogue(threshold, else_value):
    @Ts.function
    def main(
        A: T.Tensor((32, 32), "float32"),
        B: T.Tensor((32, 32), "float32"),
        E: T.Tensor((32, 32), "float32"),
    ):
        T.func_attr({"tirx.noalias": True})
        C = Ts.sblock_alloc_tensor((32, 32), "float32")
        D = Ts.sblock_alloc_tensor((32, 32), "float32")
        for i, j, k in T.grid(32, 32, 32):
            with Ts.sblock("matmul"):
                vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                with Ts.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(32, 32):
            with Ts.sblock("conditional_add"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                if vi >= threshold:
                    D[vi, vj] = C[vi, vj] + T.float32(1)
                else:
                    D[vi, vj] = C[vi, vj] + T.float32(else_value)
        for i, j in T.grid(32, 32):
            with Ts.sblock("output"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                E[vi, vj] = C[vi, vj] + D[vi, vj]

    return main


@pytest.mark.parametrize("threshold,else_value", [(0, 1.0), (16, -1.0)])
def test_matmul_declines_conditional_epilogue(threshold, else_value):
    func = _conditional_epilogue(threshold, else_value)
    s_tir.analysis.check_well_formed(func)
    target = Target({"kind": "cuda", "arch": "sm_80"})
    with target:
        assert dl.gpu.Matmul().apply(func, target, False) is None
        # A rejected, partially transformed schedule must not affect later rules.
        tvm_tirx.ir.assert_structural_equal(func, _conditional_epilogue(threshold, else_value))
        scheduled = dl.ApplyDefaultSchedule(dl.gpu.Matmul(), dl.gpu.Fallback())(
            tvm_tirx.IRModule({"main": func})
        )
        expected = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(tvm_tirx.IRModule({"main": func}))
    assert scheduled["main"].attrs["tirx.is_scheduled"]
    tvm_tirx.ir.assert_structural_equal(scheduled, expected)


@pytest.mark.skipif(not env.has_cuda() or not env.has_llvm(), reason="need CUDA and LLVM")
@pytest.mark.parametrize("threshold,else_value", [(0, 1.0), (16, -1.0)])
def test_compile_matmul_conditional_epilogue(threshold, else_value):
    if shutil.which("nvcc") is None:
        pytest.skip("need NVCC")
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorType([32, 32], "float32"))
    y = relax.Var("y", relax.TensorType([32, 32], "float32"))
    kernel = bb.add_func(
        _conditional_epilogue(threshold, else_value).with_attr("global_symbol", "kernel"),
        "kernel",
    )
    with bb.function("main", [x, y]):
        with bb.dataflow():
            result = bb.emit_output(
                relax.call_tir(kernel, (x, y), ty_args=[relax.TensorType([32, 32], "float32")])
            )
        bb.emit_func_output(result)
    mod = bb.get()
    relax.analysis.check_well_formed(mod)
    dev = tvm_tirx.cuda(0)
    target = Target(
        {
            "kind": "cuda",
            "arch": "sm_" + dev.compute_version.replace(".", ""),
            "backend_config": {"cuda": {"compiler": "nvcc"}},
        },
        host={"kind": "llvm", "mcpu": "generic"},
    )
    # Use the default compilation pipeline without simplifying the input first.
    with target:
        executable = tvm.compile(mod, target=target)
    vm = relax.VirtualMachine(executable, dev)
    offset = np.where(np.arange(32)[:, None] >= threshold, 1.0, else_value).astype("float32")
    for seed in (42, 9001, 20261102):
        rng = np.random.default_rng(seed)
        a, b = [rng.choice([-1, 1], size=(32, 32)).astype("float32") / 16 for _ in range(2)]
        actual = vm["main"](tvm_tirx.runtime.tensor(a, dev), tvm_tirx.runtime.tensor(b, dev))
        np.testing.assert_array_equal(actual.numpy(), 2 * (a @ b) + offset)


if __name__ == "__main__":
    tvm_tirx.testing.main()
