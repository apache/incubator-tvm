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
# pylint: disable=invalid-name

"""Android testing infrastructure"""

import os

import tvm
import tvm_tirx
from tvm.s_tir.meta_schedule.runner import EvaluatorConfig, RPCConfig, RPCRunner


def get_rpc_runner() -> tvm.s_tir.meta_schedule.runner.RPCRunner:
    if (
        "TVM_TRACKER_HOST" in os.environ
        and "TVM_TRACKER_PORT" in os.environ
        and "RPC_DEVICE_KEY" in os.environ
    ):
        rpc_host = os.environ["TVM_TRACKER_HOST"]
        rpc_port = int(os.environ["TVM_TRACKER_PORT"])
        rpc_key = os.environ["RPC_DEVICE_KEY"]
    else:
        raise Exception("Please initialize environment variables for using RPC tracker")

    rpc_config = RPCConfig(
        tracker_host=rpc_host,
        tracker_port=rpc_port,
        tracker_key=rpc_key,
        session_priority=1,
        session_timeout_sec=100,
    )
    evaluator_config = EvaluatorConfig(
        number=1,
        repeat=1,
        min_repeat_ms=0,
    )
    return RPCRunner(rpc_config, evaluator_config)


def get_android_gpu_target() -> tvm_tirx.target.Target:
    """Creates a Android GPU target"""
    target_c = "opencl"
    target_h = {"kind": "llvm", "mtriple": "arm64-linux-android"}
    return tvm_tirx.target.Target(target_c, host=target_h)
