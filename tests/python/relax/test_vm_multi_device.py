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
"""Test eliminate common subexpr pass"""

import numpy as np
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm_tirx.ir.module import IRModule
from tvm_tirx.script.parser import ir as I
from tvm_tirx.script.parser import relax as R
from tvm_tirx.testing import env


def compile(mod: IRModule):
    # compile the model
    mod = relax.transform.RealizeVDevice()(mod)
    mod = relax.transform.LegalizeOps()(mod)
    mod = tvm.s_tir.transform.DefaultGPUSchedule()(mod)
    # no need to feed target argument for mult-target compilation
    return tvm.compile(mod)


def test_multi_cpu():
    @I.ir_module
    class Example:
        I.module_attrs({"attr": 10})
        I.module_global_infos(
            {
                "vdevice": [
                    R.vdevice("llvm", 0),
                    R.vdevice("llvm", 1),
                ]
            }
        )

        @R.function
        def foo(
            x: R.Tensor((2, 3), "float32"),
            y: R.Tensor((3, 4), "float32"),
            z: R.Tensor((4, 5), "float32"),
        ) -> R.Tensor((2, 5), "float32"):
            with R.dataflow():
                lv0 = R.matmul(x, y)
                lv0 = R.hint_on_device(lv0, device_type=1, index=0)
                lv1: R.Tensor((2, 4), "float32", "llvm:1") = R.to_vdevice(lv0, "llvm:1")
                gv = R.matmul(lv1, z)
                R.output(gv)
            return gv

    devices = [tvm_tirx.cpu(0), tvm_tirx.cpu(1)]
    vm = relax.VirtualMachine(compile(Example), devices)

    np_ipt0 = np.random.rand(2, 3).astype(np.float32)
    np_ipt1 = np.random.rand(3, 4).astype(np.float32)
    np_ipt2 = np.random.rand(4, 5).astype(np.float32)
    np_res = np.matmul(np.matmul(np_ipt0, np_ipt1), np_ipt2)

    ipt0 = tvm_tirx.runtime.tensor(np_ipt0, devices[0])
    ipt1 = tvm_tirx.runtime.tensor(np_ipt1, devices[0])
    ipt2 = tvm_tirx.runtime.tensor(np_ipt2, devices[1])
    res = vm["foo"](ipt0, ipt1, ipt2)
    tvm_tirx.testing.assert_allclose(res.numpy(), np_res)


@pytest.mark.skipif(not env.has_multi_gpu(), reason="need multiple gpus")
def test_multi_gpu():
    @I.ir_module
    class Example:
        I.module_attrs({"attr": 10})
        I.module_global_infos(
            {
                "vdevice": [
                    R.vdevice("cuda", 1),
                    R.vdevice("cuda", 0),
                    R.vdevice("cuda", 2),
                ]
            }
        )

        @R.function
        def foo(
            a: R.Tensor((2, 3), "float32"),
            b: R.Tensor((3, 4), "float32"),
            c: R.Tensor((4, 5), "float32"),
            d: R.Tensor((5, 6), "float32"),
        ) -> R.Tensor((2, 6), "float32"):
            with R.dataflow():
                lv0: R.Tensor((2, 4), "float32", "cuda:0") = R.matmul(a, b)
                lv1: R.Tensor((2, 4), "float32", "cuda:1") = R.to_vdevice(
                    lv0,
                    "cuda:1",
                )
                lv2: R.Tensor((2, 5), "float32", "cuda:1") = R.matmul(lv1, c)
                lv3: R.Tensor((2, 5), "float32", "cuda:2") = R.to_vdevice(
                    lv2,
                    "cuda:2",
                )
                gv: R.Tensor((2, 6), "float32", "cuda:2") = R.matmul(lv3, d)
                R.output(gv)
            return gv

    np_ipt0 = np.random.rand(2, 3).astype(np.float32)
    np_ipt1 = np.random.rand(3, 4).astype(np.float32)
    np_ipt2 = np.random.rand(4, 5).astype(np.float32)
    np_ipt3 = np.random.rand(5, 6).astype(np.float32)
    np_res = np.matmul(np.matmul(np.matmul(np_ipt0, np_ipt1), np_ipt2), np_ipt3)

    ex = compile(Example)

    def run_and_check():
        if not tvm_tirx.cuda(2).exist:
            pytest.skip("requires at least 3 visible CUDA devices")
        devices = [tvm_tirx.cuda(1), tvm_tirx.cuda(0), tvm_tirx.cuda(2)]
        vm = relax.VirtualMachine(ex, devices)
        ipt0 = tvm_tirx.runtime.tensor(np_ipt0, devices[0])
        ipt1 = tvm_tirx.runtime.tensor(np_ipt1, devices[0])
        ipt2 = tvm_tirx.runtime.tensor(np_ipt2, devices[1])
        ipt3 = tvm_tirx.runtime.tensor(np_ipt3, devices[2])
        res = vm["foo"](ipt0, ipt1, ipt2, ipt3)
        tvm_tirx.testing.assert_allclose(res.numpy(), np_res)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_gpu(), reason="need gpu")
def test_multi_device():
    @I.ir_module
    class Example:
        I.module_attrs({"attr": 10})
        I.module_global_infos(
            {
                "vdevice": [
                    R.vdevice("cuda", 0),
                    R.vdevice("llvm"),
                ]
            }
        )

        @R.function
        def foo(
            x: R.Tensor((2, 3), "float32"),
            y: R.Tensor((3, 4), "float32"),
            z: R.Tensor((4, 5), "float32"),
        ) -> R.Tensor((2, 5), "float32"):
            with R.dataflow():
                lv0: R.Tensor((2, 4), "float32", "llvm") = R.matmul(x, y)
                lv1: R.Tensor((2, 4), "float32", "cuda") = R.to_vdevice(lv0, "cuda")
                gv: R.Tensor((2, 5), "float32", "cuda") = R.matmul(lv1, z)
                R.output(gv)
            return gv

    np_ipt0 = np.random.rand(2, 3).astype(np.float32)
    np_ipt1 = np.random.rand(3, 4).astype(np.float32)
    np_ipt2 = np.random.rand(4, 5).astype(np.float32)
    np_res = np.matmul(np.matmul(np_ipt0, np_ipt1), np_ipt2)

    ex = compile(Example)

    def run_and_check():
        devices = [tvm_tirx.cuda(0), tvm_tirx.cpu(0)]
        vm = relax.VirtualMachine(ex, devices)
        ipt0 = tvm_tirx.runtime.tensor(np_ipt0, devices[1])
        ipt1 = tvm_tirx.runtime.tensor(np_ipt1, devices[1])
        ipt2 = tvm_tirx.runtime.tensor(np_ipt2, devices[0])
        res = vm["foo"](ipt0, ipt1, ipt2)
        tvm_tirx.testing.assert_allclose(res.numpy(), np_res, rtol=1e-4, atol=1e-4)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm_tirx.testing.main()
