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
# pylint: disable=missing-docstring,no-member,invalid-name,unused-variable
# ruff: noqa: F401
import logging
import tempfile

import numpy as np
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir import script as Ts
from tvm.s_tir.meta_schedule.runner.config import EvaluatorConfig
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target
from tvm_tirx.testing import env

logging.basicConfig()
logging.getLogger("tvm.s_tir.meta_schedule").setLevel(logging.DEBUG)


@Ts.function
def matmul(
    A: T.Tensor([128, 128], "float32"),
    B: T.Tensor([128, 128], "float32"),
    C: T.Tensor([128, 128], "float32"),
) -> None:
    for i, j, k in T.grid(128, 128, 128):
        with Ts.sblock("update"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            with Ts.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@pytest.mark.skip("Integration test")
@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_tune_matmul_cpu():
    with tempfile.TemporaryDirectory() as work_dir:
        target = Target({"kind": "llvm", "num-cores": 16})
        database = ms.tir_integration.tune_tir(
            mod=matmul,
            target=target,
            work_dir=work_dir,
            max_trials_global=32,
            num_trials_per_iter=16,
            post_optimization=True,
        )
        trials = 32
        database = ms.tune_tir(
            mod=matmul,
            target=target,
            max_trials_global=trials,
            num_trials_per_iter=64,
            work_dir=work_dir,
            runner=ms.runner.LocalRunner(
                evaluator_config=EvaluatorConfig(
                    number=1,
                    repeat=1,
                    min_repeat_ms=100,
                )
            ),
            cost_model=ms.cost_model.XGBModel(
                extractor=ms.feature_extractor.PerStoreFeature(),
                adaptive_training=False,
            ),
            strategy=ms.search_strategy.EvolutionarySearch(),
            post_optimization=True,  # testing post optmization
        )
        # +1 because of post optmization
        assert len(database) == trials + 1


@pytest.mark.skip("Integration test")
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_tune_matmul_cuda():
    with tempfile.TemporaryDirectory() as work_dir:
        target = Target("nvidia/geforce-rtx-3070")
        trials = 32
        database = ms.tune_tir(
            mod=matmul,
            target=target,
            max_trials_global=trials,
            num_trials_per_iter=64,
            work_dir=work_dir,
            runner=ms.runner.LocalRunner(
                evaluator_config=EvaluatorConfig(
                    number=1,
                    repeat=1,
                    min_repeat_ms=100,
                )
            ),
            cost_model=ms.cost_model.XGBModel(
                extractor=ms.feature_extractor.PerStoreFeature(),
                adaptive_training=False,
            ),
            strategy=ms.search_strategy.EvolutionarySearch(),
            post_optimization=True,  # testing post optmization
        )
        # +1 because of post optmization
        assert len(database) == trials + 1


if __name__ == """__main__""":
    test_tune_matmul_cpu()
    test_tune_matmul_cuda()
