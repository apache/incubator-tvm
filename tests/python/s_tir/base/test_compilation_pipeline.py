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

"""S-TIR supplies its own default compilation dispatch."""

import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import script as Ts
from tvm_tirx.script import tirx as T


@pytest.mark.skipif(not tvm_tirx.runtime.enabled("llvm"), reason="LLVM is not enabled")
@pytest.mark.parametrize("mixed", [False, True])
def test_default_pipeline_selects_dialect(mixed):
    @Ts.function
    def scheduled(A: Ts.Tensor((16,), "float32"), B: Ts.Tensor((16,), "float32")):
        for i in range(16):
            with Ts.sblock("copy"):
                vi = Ts.axis.spatial(16, i)
                B[vi] = A[vi] + Ts.float32(1)

    @T.function
    def native(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
        for i in range(16):
            B[i] = A[i] + T.float32(2)

    assert "s_tir" not in scheduled.attrs
    assert scheduled.attrs["script.namespace"] == "s_tir"
    functions = {"scheduled": scheduled}
    if mixed:
        functions["native"] = native
    finalized_modules = []

    @tvm_tirx.ir.transform.pass_instrument
    class RecordFinalization:
        def run_before_pass(self, mod, info):
            if info.name == "tirx.MakePackedAPI":
                finalized_modules.append({gv.name_hint for gv in mod.functions})

    with tvm_tirx.ir.transform.PassContext(instruments=[RecordFinalization()]):
        module = tvm_tirx.tirx.compile(tvm_tirx.IRModule(functions), target="llvm").mod
    assert finalized_modules == [set(functions)]
    source = np.arange(16, dtype="float32")
    a = tvm_tirx.runtime.tensor(source)
    b = tvm_tirx.runtime.tensor(np.zeros(16, dtype="float32"))
    module["scheduled"](a, b)
    np.testing.assert_equal(b.numpy(), source + 1)
    if mixed:
        module["native"](a, b)
        np.testing.assert_equal(b.numpy(), source + 2)


if __name__ == "__main__":
    tvm_tirx.testing.main()
