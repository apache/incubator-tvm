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
import pytest

import tvm
import tvm_tirx
import tvm_tirx.script
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm.testing import device_enabled
from tvm_tirx.script import ir as I


def test_multinomial_from_uniform():
    @I.ir_module
    class CallSample:
        @R.function
        def foo(x: R.Tensor((3, 5), "float32"), y: R.Tensor((3, 1), "float32")):
            z = R.call_pure_packed(
                "vm.builtin.multinomial_from_uniform",
                x,
                y,
                ty_args=(R.Tensor((3, 1), dtype="int64")),
            )
            return z

    mod = CallSample
    target = tvm_tirx.target.Target("llvm", host="llvm")
    ex = tvm.compile(mod, target)
    np_rand = np.random.rand(3, 5).astype(np.float32)
    # normalize it to get the random prob
    np_prob = np_rand / np_rand.sum(axis=1, keepdims=True)
    nd_prob = tvm_tirx.runtime.tensor(np_prob)
    # special sample to get deterministic results
    nd_sample = tvm_tirx.runtime.tensor(np.array([[1.0], [0], [1]]).astype(np.float32))

    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    res = vm["foo"](nd_prob, nd_sample)
    tvm_tirx.testing.assert_allclose(res.numpy(), np.array([[4], [0], [4]]).astype(np.int64))


@pytest.mark.gpu
@pytest.mark.skipif(not device_enabled("cuda"), reason="cuda not enabled")
def test_alloc_tensor_raises_out_of_memory():
    """Out-of-memory exceptions may be raised from VM

    This is a regression test.  In previous implementations, the Relax
    VM would segfault if the built-in function
    "vm.builtin.alloc_storage" was unable to allocate the requested
    buffer.
    """
    target = "cuda"

    @I.ir_module
    class Module:
        @R.function
        def main():
            # Allocate a 1-petabyte tensor to trigger OOM.  If the CI
            # ever runs on a device with more than 1 petabyte of GPU
            # memory, this test will need to be updated.
            output = R.builtin.alloc_tensor(R.shape([1024, 1024, 1024, 1024, 1024]), "uint8", 0)
            return output

    built = tvm.compile(Module, target=target)

    def run_and_check():
        dev = tvm_tirx.cuda()
        vm = relax.VirtualMachine(built, dev)
        with pytest.raises(Exception, match="CUDA.*out of memory"):
            vm["main"]()

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm_tirx.testing.main()
