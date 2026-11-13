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

import pytest

import tvm_tirx
import tvm_tirx.script
from tvm import relax
from tvm.relax import script as R
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def test_vm_builtin_lower_mem_alloc_storage():
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((m, n), "float32")) -> R.Tensor:
            R.func_attr({"relax.force_pure": True})

            storage = R.memory.alloc_storage(R.shape([m * n * 4]), 0, "global", "uint8")
            alloc = R.memory.alloc_tensor(storage, 0, R.shape([m, n]), "float32")
            _ = R.call_packed(
                "test.op.identity", x, alloc, ty_args=(R.Tensor(ndim=2, dtype="float32"))
            )
            gv0 = alloc
            return gv0

    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((m, n), "float32")) -> R.Tensor:
            # we expected RemovePurityChecking to have been called first
            R.func_attr({"relax.force_pure": True})

            storage = R.vm.alloc_storage(R.shape([m * n * 4]), R.prim_value(0), "uint8", "global")
            alloc = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([m, n]), "float32")

            _ = R.call_packed(
                "test.op.identity", x, alloc, ty_args=(R.Tensor(ndim=2, dtype="float32"))
            )
            gv0 = alloc
            return gv0

    After = relax.transform.LowerRuntimeBuiltin()(Before)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


def test_vm_builtin_alloc_tensor_raises_error():
    """R.builtin.alloc_tensor should be handled earlier"""

    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((m, n), "float32")) -> R.Tensor:
            R.func_attr({"relax.force_pure": True})

            alloc = R.builtin.alloc_tensor(R.shape([m, n]), runtime_device_index=0, dtype="float32")
            _ = R.call_packed(
                "test.op.identity", x, alloc, ty_args=(R.Tensor(ndim=2, dtype="float32"))
            )
            gv0 = alloc
            return gv0

    with pytest.raises(RuntimeError):
        relax.transform.LowerRuntimeBuiltin()(Before)


def test_vm_reshape_may_be_var():
    """R.reshape does not require an in-line R.shape"""

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor([16], "float32"), shape: R.Shape):
            R.func_attr({"relax.force_pure": True})
            reshape = R.reshape(A, shape)
            return reshape

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor([16], "float32"), shape: R.Shape):
            R.func_attr({"relax.force_pure": True})
            reshape = R.call_packed(
                "vm.builtin.reshape",
                A,
                shape,
                ty_args=R.Tensor(shape, dtype="float32"),
            )
            return reshape

    After = relax.transform.LowerRuntimeBuiltin()(Before)

    tvm_tirx.ir.assert_structural_equal(Expected, After)


def test_vm_reshape_using_tensor_to_shape():
    """Shape argument of R.reshape may come from tensor_to_shape"""

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor([16], "float32"), shape_tensor: R.Tensor([2], "int64")):
            R.func_attr({"relax.force_pure": True})
            shape = R.tensor_to_shape(shape_tensor)
            reshape = R.reshape(A, shape)
            return reshape

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor([16], "float32"), shape_tensor: R.Tensor([2], "int64")):
            R.func_attr({"relax.force_pure": True})

            shape = R.call_packed(
                "vm.builtin.tensor_to_shape",
                shape_tensor,
                ty_args=R.Shape(ndim=2),
            )
            reshape = R.call_packed(
                "vm.builtin.reshape",
                A,
                shape,
                ty_args=R.Tensor(shape, dtype="float32"),
            )
            return reshape

    After = relax.transform.LowerRuntimeBuiltin()(Before)

    tvm_tirx.ir.assert_structural_equal(Expected, After)


if __name__ == "__main__":
    tvm_tirx.testing.main()
