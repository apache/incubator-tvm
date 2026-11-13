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
"""Dummy objects for testing."""

import random

from tvm.s_tir.schedule import Trace
from tvm.s_tir.utils import derived_object

from ..builder import BuilderInput, BuilderResult, PyBuilder
from ..mutator import PyMutator
from ..runner import PyRunner, PyRunnerFuture, RunnerFuture, RunnerInput, RunnerResult
from ..tune_context import TuneContext  # pylint: disable=unused-import


@derived_object
class DummyRunnerFuture(PyRunnerFuture):
    def done(self) -> bool:
        return True

    def result(self) -> RunnerResult:
        run_secs = [random.uniform(5, 30) for _ in range(random.randint(1, 10))]
        return RunnerResult(run_secs, None)


@derived_object
class DummyBuilder(PyBuilder):
    def build(self, build_inputs: list[BuilderInput]) -> list[BuilderResult]:
        return [BuilderResult("test_path", None) for _ in build_inputs]


@derived_object
class DummyRunner(PyRunner):
    def run(self, runner_inputs: list[RunnerInput]) -> list[RunnerFuture]:
        return [DummyRunnerFuture() for _ in runner_inputs]  # type: ignore


@derived_object
class DummyMutator(PyMutator):
    """Dummy Mutator for testing"""

    def _initialize_with_tune_context(self, context: "TuneContext") -> None:
        pass

    def apply(self, trace: Trace, _) -> Trace | None:
        return Trace(trace.insts, {})

    def clone(self):
        return DummyMutator()
