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
# ruff: noqa: E501

import tvm_tirx
import tvm_tirx.testing
from tvm.relax import script as R
from tvm.relax.analysis import estimate_memory_usage
from tvm.s_tir import script as Ts
from tvm_tirx.script import tirx as T


def test_basic():
    @tvm_tirx.script.ir_module
    class Module:
        @Ts.function
        def add(
            rxplaceholder: T.Tensor(T.int64(8), "float32"),
            rxplaceholder_1: T.Tensor((), "float32"),
            T_add: T.Tensor(T.int64(8), "float32"),
        ):
            T.evaluate(0)

        @Ts.function
        def reshape(
            rxplaceholder: T.Tensor((T.int64(2), T.int64(4)), "float32"),
            T_reshape: T.Tensor(T.int64(8), "float32"),
        ):
            T.evaluate(0)

        @Ts.function
        def relu(
            rxplaceholder: T.Tensor(T.int64(8), "float32"), compute: T.Tensor(T.int64(8), "float32")
        ):
            T.evaluate(0)

        @Ts.function
        def log(
            rxplaceholder: T.Tensor(T.int64(10), "float32"),
            compute: T.Tensor(T.int64(10), "float32"),
        ):
            T.evaluate(0)

        @Ts.function
        def exp(
            rxplaceholder: T.Tensor((T.int64(2), T.int64(4)), "float32"),
            compute: T.Tensor((T.int64(2), T.int64(4)), "float32"),
        ):
            T.evaluate(0)

        @Ts.function
        def pad(
            rxplaceholder: T.Tensor(T.int64(8), "float32"),
            PadInput: T.Tensor(T.int64(10), "float32"),
        ):
            T.evaluate(0)

        @R.function(pure=False)
        def main(x: R.Tensor((2, 4), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
            cls = Module
            storage: R.Any = R.memory.alloc_storage(
                R.shape([32]), virtual_device_index=0, storage_scope="global", dtype="float32"
            )
            alloc: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(
                storage, offset=0, shape=R.shape([2, 4]), dtype="float32"
            )
            _: R.Tuple() = R.call_tir_packed(cls.exp, (x, alloc))
            lv: R.Tensor((2, 4), dtype="float32") = alloc
            lv1: R.Tensor((8,), dtype="float32") = R.call_packed(
                "vm.builtin.reshape", lv, R.shape([8]), ty_args=[R.Tensor((8,), dtype="float32")]
            )
            storage1: R.Any = R.memory.alloc_storage(
                R.shape([40]), virtual_device_index=0, storage_scope="global", dtype="float32"
            )
            alloc1: R.Tensor((8,), dtype="float32") = R.memory.alloc_tensor(
                storage1, offset=0, shape=R.shape([8]), dtype="float32"
            )
            _1: R.Tuple() = R.call_tir_packed(cls.relu, (lv1, alloc1))
            _2: R.Tuple() = R.memory.kill_tensor(alloc)
            _3: R.Tuple() = R.memory.kill_tensor(lv1)
            lv2: R.Tensor((8,), dtype="float32") = alloc1
            alloc2: R.Tensor((8,), dtype="float32") = R.memory.alloc_tensor(
                storage, offset=0, shape=R.shape([8]), dtype="float32"
            )
            _4: R.Tuple() = R.call_tir_packed(cls.add, (lv2, R.const(1, "float32"), alloc2))
            _5: R.Tuple() = R.memory.kill_tensor(alloc1)
            lv3: R.Tensor((8,), dtype="float32") = alloc2
            alloc3: R.Tensor((10,), dtype="float32") = R.memory.alloc_tensor(
                storage1, offset=0, shape=R.shape([10]), dtype="float32"
            )
            _6: R.Tuple() = R.call_tir_packed(cls.pad, (lv3, alloc3))
            _7: R.Tuple() = R.memory.kill_tensor(alloc2)
            lv4: R.Tensor((10,), dtype="float32") = alloc3
            alloc4: R.Tensor((10,), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([10]), dtype="float32", runtime_device_index=0
            )
            _8: R.Tuple() = R.call_tir_packed(cls.log, (lv4, alloc4))
            _9: R.Tuple() = R.memory.kill_tensor(alloc3)
            gv5: R.Tensor((10,), dtype="float32") = alloc4
            _11: R.Tuple() = R.memory.kill_storage(storage)
            _10: R.Tuple() = R.memory.kill_storage(storage1)
            return gv5

    assert (
        estimate_memory_usage(Module)
        == r"""Memory usage estimation:
- Function main:
 * Without memory planning, there are 5 constant-size memory allocation(s) with total size 1.639e-07 GB.
 * With memory planning, there are 3 constant-size memory allocation(s) with total size 1.043e-07 GB.
 * Memory planning reduces constant memory size to 63.6%.
"""
    )


if __name__ == "__main__":
    tvm_tirx.testing.main()
