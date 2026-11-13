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
"""The class of Space used to optimize the Meta parameters"""

import json
import random
from copy import deepcopy
from typing import Any

import numpy as np  # type: ignore

from tvm.s_tir import Schedule
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir.meta_schedule.database import TuningRecord, Workload
from tvm.s_tir.meta_schedule.utils import remove_build_dir
from tvm_tirx.target import Target

from .utils import write_file


class Space:
    """Space class

    Parameters
    ----------
    data: json data
        A json file template
    workload: json data
        A json file workload
    target: Target data
        Target device information
    """

    def __init__(self, data: Any, workload: Any, target: Target):
        self.cfg = deepcopy(data)
        self._id = data[0]
        self.workload = Workload.from_json(workload)
        self.target = target
        self.dev = self.get_device_type(target)
        self.total_dims = 0
        self.dims: list[int] = []
        self.start: list[int] = []
        self.config_space: dict[str, list[int]] = dict()
        self.create_space()

    def __repr__(self) -> str:
        """Print the config space"""
        out = ""
        for key in self.config_space:
            out += f"{key}: dims={self.config_space[key]}\n"
        out += f"Total dimensions: {self.total_dims}\n"
        return out

    def __str__(self) -> str:
        """Print the config space"""
        out = ""
        for key in self.config_space:
            out += f"{key}: dims={self.config_space[key]}\n"
        out += f"Total dimensions: {self.total_dims}\n"
        return out

    def get_value(self, key, pos):
        """Return the space"""
        return self.config_space[key][pos]

    def add_space(self, space_list: list, element_list: list, limit=10000) -> list[int]:
        """Return a list without repeat and with limited value"""
        new_list = element_list
        for elem in space_list:
            if elem not in new_list and elem <= limit:
                new_list.append(elem)
        return new_list

    def knob2point(self, knob):
        """Convert a array to point"""
        point = 0
        for j, k in enumerate(knob):
            point += int(np.prod(self.dims[:j])) * k
        return point

    def point2knob(self, point):
        """Convert point form (single integer) to knob (vector)"""
        knob = []
        for dim in self.dims:
            knob.append(point % dim)
            point //= dim
        return knob

    def power_of_two(self, min_value: int, max_value: int) -> list:
        """Return power of two array in interval"""
        return [1 << i for i in range(min_value, max_value + 1)]

    def get_index(self, array: list, value: int):
        """returns an index if it finds the value"""
        for i in range(len(array)):
            if array[i][0] == value:
                return i
        return -1

    def template(self, values=None, create=True):
        """Generate the template from the values"""
        idx = -1
        config = deepcopy(self.cfg[1])
        for counter, cfg in enumerate(config[0][0]):
            opt = cfg[0]
            if opt == "Annotate":
                ann_key = cfg[2]
                if ann_key == ["meta_schedule.parallel"]:
                    interval = self.power_of_two(5, 9)
                elif ann_key == ["meta_schedule.vectorize"]:
                    interval = self.power_of_two(4, 8)
                elif ann_key == ["tirx.auto_unroll_max_step"]:
                    interval = self.power_of_two(7, 11)
                elif ann_key == ["meta_schedule.thread_extent_low_inclusive"]:
                    interval = self.power_of_two(5, 6)
                elif ann_key == ["meta_schedule.thread_extent_high_inclusive"]:
                    interval = self.power_of_two(8, 12)
                else:
                    continue
                idx += 1
                key = f"ann_{idx}"
                ann_value = cfg[1][1]
                if create:
                    self.config_space[key] = self.add_space(interval, [ann_value])
                else:
                    cfg[1][1] = self.get_value(key, values[idx])
            elif opt == "SamplePerfectTile":
                tile = config[0][1]
                tile_idx = self.get_index(tile, counter)
                tile_val = tile[tile_idx][1]
                interval = self.power_of_two(1, 6)
                for i in range(len(tile_val)):
                    idx += 1
                    key = f"sp_{counter}_{idx}"
                    split = tile_val[i]
                    if create:
                        self.config_space[key] = self.add_space(interval, [split])
                    else:
                        config[0][1][tile_idx][1][i] = self.get_value(key, values[idx])
            elif opt == "TransformLayout":
                del config[0][0][counter]
        if create:
            return None
        return config

    def create_space(self):
        """Create the space using Meta's space"""
        self.template(create=True)
        # print(self.config_space)
        self.dims = []
        for key in self.config_space:
            self.dims.append(len(self.config_space[key]))
        self.total_dims = 1
        if len(self.dims) > 0:
            for dim in self.dims:
                self.total_dims *= dim

    def get_device_type(self, target: Target) -> str:
        """Get the device type string from a target.

        Parameters
        ----------
        target : Target
            The target to get the device type from.

        Returns
        -------
        device_type : str
            The device type string.
        """
        if target.kind.name == "llvm":
            return "cpu"
        elif target.kind.name == "cuda":
            return "cuda"
        else:
            raise RuntimeError(f"Unsupported target kind for device type: {target.kind.name}")

    def save_log(
        self,
        path: str,
        record: ms.database.TuningRecord,
        results: ms.runner.RunnerResult,
    ) -> None:
        """Save the log file"""
        new_json = [self._id, record.as_json()]
        new_json[1][1] = results
        write_file([new_json], path, "a")

    def run(
        self,
        json_file_list,
        final_log,
        timeout=10,
        number=2,
        repeat=3,
        min_repeat_ms=0,
        cpu_cache=False,
    ):
        """Execute a log file and save"""

        builder = ms.builder.LocalBuilder(timeout_sec=timeout)
        runner = ms.runner.LocalRunner(
            evaluator_config=ms.runner.EvaluatorConfig(
                number=number,
                repeat=repeat,
                min_repeat_ms=min_repeat_ms,
                enable_cpu_cache_flush=cpu_cache,
            ),
        )

        results = np.full(len(json_file_list), [10000], dtype=list)
        records, mods = [], []
        for i, cfg in enumerate(json_file_list):
            try:
                record = TuningRecord.from_json(json.loads(json.dumps(cfg)), self.workload)
                sch = Schedule(self.workload.mod)
                # In some layers this is a heavy impact in time cost, so
                # I applied this only 25% of the samples.
                remove_postproc = random.random() > 0.75
                record.trace.apply_to_schedule(sch, remove_postproc=remove_postproc)
                mods.append(sch.mod)
                records.append(record)
            except Exception:  # pylint: disable=broad-except, invalid-name
                continue

        builder_res = builder.build([ms.builder.BuilderInput(mod, self.target) for mod in mods])

        for i, record in enumerate(records):
            try:
                inp = ms.runner.RunnerInput(
                    builder_res[i].artifact_path,
                    device_type=self.dev,
                    args_info=ms.arg_info.TensorInfo.from_function(mods[i]["main"]),
                )
                runner_res = runner.run([inp])[0].result()
                results[i] = [v.value for v in runner_res.run_secs]  # type: ignore
            except Exception:  # pylint: disable=broad-except, invalid-name
                results[i] = [1e10]
                continue

            # save the solution in json file
            self.save_log(final_log, record, results[i])

            # clean up
            remove_build_dir(builder_res[i].artifact_path)
        return results
