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
import tempfile

import pytest
import tvm_ffi

import tvm
import tvm_tirx
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir import script as Ts
from tvm.s_tir.meta_schedule.schedule_rule import ApplyCustomRule
from tvm_tirx.script import tirx as T


@tvm_tirx.script.ir_module
class Matmul:
    @Ts.function
    def main(
        A: T.Tensor((1024, 1024), "float32"),
        B: T.Tensor((1024, 1024), "float32"),
        C: T.Tensor((1024, 1024), "float32"),
    ) -> None:
        T.func_attr({"global_symbol": "main"})

        for i, j, k in T.grid(1024, 1024, 1024):
            with Ts.sblock("matmul"):
                Ts.sblock_attr({"schedule_rule": "test_apply_custom_rule"})
                vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                with Ts.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@tvm_ffi.register_global_func("s_tir.meta_schedule.cpu.test_apply_custom_rule")
def sch_fn(sch: tvm.s_tir.Schedule, block: tvm.s_tir.SBlock) -> list[tvm.s_tir.Schedule]:
    raise ValueError("Intended for s_tir.meta_schedule.cpu.test_apply_custom_rule")


@pytest.mark.skip("Tuning test - launches runner")
def test_custom_rule():
    with pytest.raises(ValueError) as e_info:
        with tempfile.TemporaryDirectory() as tmpdir:
            sch_rules = [ApplyCustomRule()]
            space_gen = ms.space_generator.PostOrderApply(sch_rules=sch_rules)
            ms.tune_tir(
                mod=Matmul,
                target={"kind": "llvm", "num-cores": 1},
                work_dir=tmpdir,
                max_trials_global=10,
                space=space_gen,
            )
    assert "Intended for s_tir.meta_schedule.cpu.test_apply_custom_rule" in str(e_info.value)


if __name__ == "__main__":
    test_custom_rule()
