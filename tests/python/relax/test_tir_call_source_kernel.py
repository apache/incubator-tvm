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
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm.s_tir import script as Ts
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env

add_cuda_source = """
extern "C" __global__ void add_kernel(float* x, float* y, float* output, int n_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elements) {
        output[i] = x[i] + y[i];
    }
}
"""


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_tir_call_source_kernel():
    BLOCK_SIZE = 64

    m_add = T.dynamic("m", "int64")
    m_main = T.dynamic("m", "int64")

    @I.ir_module
    class Module:
        @Ts.function
        def add(
            x: T.Tensor((m_add,), "float32"),
            y: T.Tensor((m_add,), "float32"),
            output: T.Tensor((m_add,), "float32"),
        ) -> None:
            T.func_attr({"global_symbol": "add"})

            with Ts.sblock("root"):
                Ts.reads(x[0:m_add], y[0:m_add])
                Ts.writes(output[0:m_add])
                T.call_kernel(
                    add_cuda_source,
                    ((T.ceildiv(m_add, BLOCK_SIZE),), (BLOCK_SIZE,)),
                    x.data,
                    y.data,
                    output.data,
                    m_add,
                    kernel_name="add_kernel",
                )

        @R.function
        def main(x: R.Tensor((m_main,), "float32"), y: R.Tensor((m_main,), "float32")):
            with R.dataflow():
                output = R.call_tir(
                    Module.add, [x, y], ty_args=[relax.TensorType((m_main,), "float32")]
                )
                R.output(output)
            return output

    m = T.dynamic("m", "int64")

    @I.ir_module
    class Parsed:
        @Ts.function
        def add(
            x: T.Tensor((m,), "float32"),
            y: T.Tensor((m,), "float32"),
            output: T.Tensor((m,), "float32"),
        ):
            with Ts.sblock("root"):
                Ts.reads(x[0:m], y[0:m])
                Ts.writes(output[0:m])
                T.call_packed(
                    "add_kernel",
                    x.data,
                    y.data,
                    output.data,
                    m,
                    (m + T.int64(64) - T.int64(1)) // T.int64(64),
                    64,
                )

    tvm_tirx.ir.assert_structural_equal(Module["add"], Parsed["add"])
    assert len(Module.get_attr("external_mods")) == 1

    lib = tvm.compile(Module, target=tvm_tirx.target.Target("cuda"))

    def run_and_check():
        device = tvm_tirx.cuda(0)
        x_nd = tvm_tirx.runtime.tensor(np.random.rand(256).astype(np.float32), device)
        y_nd = tvm_tirx.runtime.tensor(np.random.rand(256).astype(np.float32), device)
        output_np = x_nd.numpy() + y_nd.numpy()
        output_nd = tvm.runtime.vm.VirtualMachine(lib, device)["main"](x_nd, y_nd)
        tvm_tirx.testing.assert_allclose(output_nd.numpy(), output_np, rtol=1e-5)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)
