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

import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def test_alloc_storage():
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @I.ir_module
    class Module:
        @R.function(pure=False)
        def main(shape: R.Shape([m, n])):  # type: ignore
            storage: R.Any = R.memory.alloc_storage(
                R.shape([m, n]), R.prim_value(0), R.str("ipc_memory"), R.dtype("float16")
            )
            alloc: R.Tensor((m, n), dtype="float16") = R.memory.alloc_tensor(  # type: ignore
                storage, R.prim_value(0), R.shape([m, n]), R.dtype("float16")
            )
            return alloc

    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main(shape: R.Shape([m, n])):  # type: ignore
            storage: R.Any = R.call_packed(
                "runtime.disco.cuda_ipc.alloc_storage",
                R.shape([m, n]),
                R.dtype("float16"),
                ty_args=(R.Any,),
            )
            alloc: R.Tensor((m, n), dtype="float16") = R.memory.alloc_tensor(  # type: ignore
                storage, R.prim_value(0), R.shape([m, n]), R.dtype("float16")
            )
            return alloc

    mod = relax.transform.LowerGPUIPCAllocStorage()(Module)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_builtin_alloc_tensor():
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @I.ir_module
    class Module:
        @R.function(pure=False)
        def main(shape: R.Shape([m, n])):  # type: ignore
            tensor: R.Any = R.builtin.alloc_tensor(
                R.shape([m, n]), R.dtype("float16"), R.prim_value(0), R.str("ipc_memory")
            )
            return tensor

    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main(shape: R.Shape([m, n])):  # type: ignore
            gv: R.Any = R.call_packed(
                "runtime.disco.cuda_ipc.alloc_storage",
                R.shape([m, n]),
                R.dtype("float16"),
                ty_args=(R.Any,),
            )
            tensor: R.Tensor((m, n), dtype="float16") = R.memory.alloc_tensor(  # type: ignore
                gv, R.prim_value(0), R.shape([m, n]), R.dtype("float16")
            )
            return tensor

    mod = relax.transform.LowerGPUIPCAllocStorage()(Module)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    test_alloc_storage()
    test_builtin_alloc_tensor()
