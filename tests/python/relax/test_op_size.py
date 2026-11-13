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

import numpy as np

import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm_tirx.script import ir as I


def test_op_size():
    @tvm_tirx.script.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((), "int64"):
            return R.size(x)

    x_np = np.random.rand(2, 3).astype("float32")
    x = tvm_tirx.runtime.tensor(x_np)

    target = tvm_tirx.target.Target("llvm")
    ex = relax.build(Module, target)
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())

    res = vm["main"](x)
    assert res.numpy() == 6


def test_op_size_dynamic():
    m = I.dynamic("m", "int64")
    n = I.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor((m, n), "float32")) -> R.Tensor((), "int64"):
            return R.size(x)

    x_np = np.random.rand(4, 5).astype("float32")
    x = tvm_tirx.runtime.tensor(x_np)

    target = tvm_tirx.target.Target("llvm")
    ex = relax.build(Module, target)
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())

    res = vm["main"](x)
    assert res.numpy() == 20


if __name__ == "__main__":
    tvm_tirx.testing.main()
