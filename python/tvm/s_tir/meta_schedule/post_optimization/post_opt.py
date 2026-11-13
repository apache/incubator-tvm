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
"""Post optimization method"""

import numpy as np  # type: ignore

from tvm_tirx.target import Target

from .droplet import Droplet
from .utils import clean_file, get_time, read_cfg_file, write_file


class PostOpt:
    """PostOpt class

    Parameters
    ----------
    work_dir : str
        The working directory.
    target: Target data
        Target device information
    trials: integer value
        Max number of trials to execute the optimization
    """

    def __init__(self, work_dir: str, target: Target, trials: int = 100) -> None:
        self.work_dir = work_dir
        self.target = target
        self.trials = trials

    def run(self) -> None:
        """Execute the post optimization"""

        tuning_file = self.work_dir + "/database_tuning_record.json"
        workload_file = self.work_dir + "/database_workload.json"

        cfg = read_cfg_file(tuning_file, workload_file)

        print("id | time MS (s) | time DPMS (s) | speedup")
        for idx, layer in enumerate(cfg):
            time, data, workload = cfg[layer]
            ms_time = np.mean(time)

            temp_log = f"{self.work_dir}/opt_{idx}.log"

            # Run the exploitation by Droplet
            droplet = Droplet(data, workload, self.target, temp_log)
            droplet.tune(self.trials)

            dpms_time, dpm_sol = get_time(temp_log)
            dpms_time = np.mean(dpms_time)

            speedup = ms_time / dpms_time

            # save the best solution
            write_file([dpm_sol], tuning_file, mode="a")

            # show the perfomance
            print(f"{idx:2d} | {ms_time:.10f} | {dpms_time:.10f} | {speedup:.2f}")

            # clean the temporary files
            clean_file(temp_log)
