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
"""Test tensor allocation with memory scope for Relax VM"""

import numpy as np
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm.s_tir import script as Ts
from tvm_tirx.runtime import MemoryScope
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


@I.ir_module
class Module:
    @Ts.function
    def add(
        arg0: T.Tensor((2, 2), "float32"),
        arg1: T.Tensor((2, 2), "float32"),
        output: T.Tensor((2, 2), "float32"),
    ):
        T.func_attr({"operator_name": "relax.add"})
        for ax0 in range(2):
            for ax1 in range(2):
                with Ts.sblock("T_add"):
                    v_ax0 = Ts.axis.spatial(2, ax0)
                    v_ax1 = Ts.axis.spatial(2, ax1)
                    Ts.reads(arg0[v_ax0, v_ax1], arg1[v_ax0, v_ax1])
                    Ts.writes(output[v_ax0, v_ax1])
                    output[v_ax0, v_ax1] = arg0[v_ax0, v_ax1] + arg1[v_ax0, v_ax1]

    @R.function(pure=False)
    def main(x: R.Tensor((2, 2), dtype="float32")):
        cls = Module
        storage = R.vm.alloc_storage(
            R.shape([2 * 2]), runtime_device_index=0, dtype="float32", storage_scope="global"
        )
        alloc = R.vm.alloc_tensor(storage, offset=0, shape=R.shape([2, 2]), dtype="float32")
        _: R.Tuple = R.call_tir_packed(cls.add, (x, x, alloc))
        out: R.Tensor((2, 2), dtype="float32") = alloc
        return out


def test_alloc_storage_with_scope_global():
    arg0 = np.random.uniform(size=(2, 2)).astype(np.float32)
    output_ref = arg0 + arg0
    mod = Module
    target = "llvm"
    with tvm_tirx.ir.transform.PassContext(opt_level=3):
        lib = tvm.relax.build(mod, target=target)

    dev = tvm_tirx.cpu()
    vm_rt = relax.VirtualMachine(lib, dev)
    x = tvm_tirx.runtime.tensor(arg0, dev)
    vm_rt.set_input("main", x)
    vm_rt.invoke_stateful("main")
    output = vm_rt.get_outputs("main").numpy()
    tvm_tirx.testing.assert_allclose(output_ref, output)


def test_relax_allocation_scope_operands():
    scope = MemoryScope("global")
    for make_call in (
        lambda value: tvm.relax.op.builtin.alloc_tensor((4,), "float32", 0, value),
        lambda value: tvm.relax.op.memory.alloc_storage((16,), 0, value, "float32"),
        lambda value: tvm.relax.op.vm.alloc_storage((16,), 0, "float32", value),
    ):
        tvm_tirx.ir.assert_structural_equal(make_call(scope), make_call(scope.name))


def test_vdevice_memory_scope():
    global_scope = MemoryScope("global")
    for scope in ("global", global_scope):
        assert tvm.relax.VDevice("llvm", memory_scope=scope).memory_scope.same_as(global_scope)
    with pytest.raises(ValueError):
        tvm.relax.VDevice("llvm", memory_scope="")


if __name__ == "__main__":
    tvm_tirx.testing.main()
