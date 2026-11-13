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
# ruff: noqa: F821
import argparse
import logging

import tvm_tirx
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir.meta_schedule.testing.te_workload import create_te_workload
from tvm.s_tir.testing import strtobool
from tvm.support import describe


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--workload",
        type=str,
        required=True,
    )
    args.add_argument(
        "--target",
        type=str,
        required=True,
    )
    args.add_argument(
        "--num-trials",
        type=int,
        required=True,
    )
    args.add_argument(
        "--rpc-host",
        type=str,
        required=True,
    )
    args.add_argument(
        "--rpc-port",
        type=int,
        required=True,
    )
    args.add_argument(
        "--rpc-key",
        type=str,
        required=True,
    )
    args.add_argument(
        "--work-dir",
        type=str,
        required=True,
    )
    args.add_argument(
        "--number",
        type=int,
        default=3,
    )
    args.add_argument(
        "--repeat",
        type=int,
        default=1,
    )
    args.add_argument(
        "--min-repeat-ms",
        type=int,
        default=100,
    )
    args.add_argument(
        "--adaptive-training",
        type=lambda x: bool(strtobool(x)),
        required=False,
        help="example: True / False",
        default=True,
    )
    args.add_argument(
        "--cpu-flush",
        type=lambda x: bool(strtobool(x)),
        help="example: True / False",
        required=True,
    )
    parsed = args.parse_args()
    parsed.target = tvm_tirx.target.Target(parsed.target)
    parsed.rpc_config = ms.runner.RPCConfig(
        tracker_host=parsed.rpc_host,
        tracker_port=parsed.rpc_port,
        tracker_key=parsed.rpc_key,
        session_timeout_sec=60,
    )
    return parsed


logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logging.getLogger("tvm.s_tir.meta_schedule").setLevel(logging.DEBUG)
ARGS = _parse_args()


def main():
    describe()
    print(f"Workload: {ARGS.workload}")
    with ms.Profiler() as profiler:
        sch: s_tir.Schedule | None = ms.tir_integration.tune_tir(
            mod=create_te_workload(ARGS.workload, 0),
            target=ARGS.target,
            work_dir=ARGS.work_dir,
            max_trials_global=ARGS.num_trials,
            num_trials_per_iter=64,
            runner=ms.runner.RPCRunner(  # type: ignore
                rpc_config=ARGS.rpc_config,
                evaluator_config=ms.runner.EvaluatorConfig(
                    number=ARGS.number,
                    repeat=ARGS.repeat,
                    min_repeat_ms=ARGS.min_repeat_ms,
                    enable_cpu_cache_flush=ARGS.cpu_flush,
                ),
                alloc_repeat=1,
            ),
            cost_model=ms.cost_model.XGBModel(  # type: ignore
                extractor=ms.feature_extractor.PerStoreFeature(),
                adaptive_training=ARGS.adaptive_training,
            ),
            strategy=ms.search_strategy.EvolutionarySearch(),
        )

    print("Tuning Time:")
    print(profiler.table())

    if sch is None:
        print("No valid schedule found!")
    else:
        print(sch.mod.script())
        print(sch.trace)


if __name__ == "__main__":
    main()
