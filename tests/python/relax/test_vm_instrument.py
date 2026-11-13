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
# ruff: noqa: RUF005
import numpy as np

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax.testing import nn
from tvm.relax.testing.lib_comparator import LibCompareVMInstrument


def get_exec(data_shape):
    builder = relax.BlockBuilder()
    weight1_np = np.random.randn(64, 64).astype("float32")
    weight2_np = np.random.randn(64, 64).astype("float32")

    with builder.function("main"):
        model = nn.Sequential(
            nn.Linear(data_shape[1], weight1_np.shape[0], bias=False),
            nn.ReLU(),
            nn.Linear(weight2_np.shape[0], weight2_np.shape[1], bias=False),
            nn.ReLU(),
        )
        data = nn.Placeholder(data_shape, name="data")
        output = model(data)
        params = [data] + model.parameters()
        builder.emit_func_output(output, params=params)

    mod = builder.get()

    params = {"linear_weight": weight1_np, "linear_weight1": weight2_np}
    mod = relax.transform.BindParams("main", params)(mod)

    target = "llvm"
    return tvm.compile(mod, target)


def get_exec_int32(data_shape):
    builder = relax.BlockBuilder()

    with builder.function("main"):
        model = nn.ReLU()
        data = nn.Placeholder(data_shape, dtype="int32", name="data")
        output = model(data)
        params = [data] + model.parameters()
        builder.emit_func_output(output, params=params)

    mod = builder.get()
    target = "llvm"
    return tvm.compile(mod, target)


def test_conv2d_cpu():
    data_np = np.random.randn(1, 64).astype("float32")
    ex = get_exec(data_np.shape)
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    hit_count = {}

    def instrument(func, name, before_run, ret_val, *args):
        if (name, before_run) not in hit_count:
            hit_count[(name, before_run)] = 0
        hit_count[(name, before_run)] += 1
        assert callable(func)
        if before_run:
            assert ret_val is None
        if name == "matmul":
            return relax.VMInstrumentReturnKind.SKIP_RUN

    vm.set_instrument(instrument)
    vm["main"](tvm_tirx.runtime.tensor(data_np))
    assert hit_count[("matmul", True)] == 2
    assert ("matmul", False) not in hit_count
    assert hit_count[("relu", True)] == 2
    assert hit_count[("relu", False)] == 2


def test_lib_comparator():
    data_np = np.random.randn(1, 64).astype("int32")
    ex = get_exec_int32(data_np.shape)
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    # compare against library module
    cmp = LibCompareVMInstrument(vm.module.imports[0], tvm_tirx.cpu(), verbose=False)
    vm.set_instrument(cmp)
    vm["main"](tvm_tirx.runtime.tensor(data_np))


if __name__ == "__main__":
    tvm_tirx.testing.main()
