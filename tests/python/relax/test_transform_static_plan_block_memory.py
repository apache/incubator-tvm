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
# ruff: noqa: E501, F841

import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm.s_tir import script as Ts
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def test_basic():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Module:
        @Ts.function
        def add(rxplaceholder: T.Tensor(T.int64(8), "float32"), rxplaceholder_1: T.Tensor((), "float32"), T_add: T.Tensor(T.int64(8), "float32")):
            T.evaluate(0)

        @Ts.function
        def reshape(rxplaceholder: T.Tensor((T.int64(2), T.int64(4)), "float32"), T_reshape: T.Tensor(T.int64(8), "float32")):
            T.evaluate(0)

        @Ts.function
        def relu(rxplaceholder: T.Tensor(T.int64(8), "float32"), compute: T.Tensor(T.int64(8), "float32")):
            T.evaluate(0)

        @Ts.function
        def log(rxplaceholder: T.Tensor(T.int64(10), "float32"), compute: T.Tensor(T.int64(10), "float32")):
            T.evaluate(0)

        @Ts.function
        def exp(rxplaceholder: T.Tensor((T.int64(2), T.int64(4)), "float32"), compute: T.Tensor((T.int64(2), T.int64(4)), "float32")):
            T.evaluate(0)

        @Ts.function
        def pad(rxplaceholder: T.Tensor(T.int64(8), "float32"), PadInput: T.Tensor(T.int64(10), "float32")):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((2, 4), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
            # we expected RemovePurityChecking to have been invoked first
            R.func_attr({"relax.force_pure": True})
            cls = Module
            alloc: R.Tensor((2, 4), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 4]), dtype="float32", runtime_device_index=0)
            _: R.Tuple() = R.call_tir_packed(cls.exp, (x, alloc))
            lv: R.Tensor((2, 4), dtype="float32") = alloc
            lv1: R.Tensor((8,), dtype="float32") = R.reshape(lv, (8,))
            alloc1: R.Tensor((8,), dtype="float32") = R.builtin.alloc_tensor(R.shape([8]), dtype="float32", runtime_device_index=0)
            _1: R.Tuple() = R.call_tir_packed(cls.relu, (lv1, alloc1))
            lv2: R.Tensor((8,), dtype="float32") = alloc1
            alloc2: R.Tensor((8,), dtype="float32") = R.builtin.alloc_tensor(R.shape([8]), dtype="float32", runtime_device_index=0)
            _2: R.Tuple() = R.call_tir_packed(cls.add, (lv2, R.const(1, "float32"), alloc2))
            lv3: R.Tensor((8,), dtype="float32") = alloc2
            alloc3: R.Tensor((10,), dtype="float32") = R.builtin.alloc_tensor(R.shape([10]), dtype="float32", runtime_device_index=0)
            _3: R.Tuple() = R.call_tir_packed(cls.pad, (lv3, alloc3))
            lv4: R.Tensor((10,), dtype="float32") = alloc3
            alloc4: R.Tensor((10,), dtype="float32") = R.builtin.alloc_tensor(R.shape([10]), dtype="float32", runtime_device_index=0)
            _4: R.Tuple() = R.call_tir_packed(cls.log, (lv4, alloc4))
            gv: R.Tensor((10,), dtype="float32") = alloc4
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function
        def add(rxplaceholder: T.Tensor(T.int64(8), "float32"), rxplaceholder_1: T.Tensor((), "float32"), T_add: T.Tensor(T.int64(8), "float32")):
            T.evaluate(0)

        @Ts.function
        def reshape(rxplaceholder: T.Tensor((T.int64(2), T.int64(4)), "float32"), T_reshape: T.Tensor(T.int64(8), "float32")):
            T.evaluate(0)

        @Ts.function
        def relu(rxplaceholder: T.Tensor(T.int64(8), "float32"), compute: T.Tensor(T.int64(8), "float32")):
            T.evaluate(0)

        @Ts.function
        def log(rxplaceholder: T.Tensor(T.int64(10), "float32"), compute: T.Tensor(T.int64(10), "float32")):
            T.evaluate(0)

        @Ts.function
        def exp(rxplaceholder: T.Tensor((T.int64(2), T.int64(4)), "float32"), compute: T.Tensor((T.int64(2), T.int64(4)), "float32")):
            T.evaluate(0)

        @Ts.function
        def pad(rxplaceholder: T.Tensor(T.int64(8), "float32"), PadInput: T.Tensor(T.int64(10), "float32")):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((2, 4), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            storage: R.Any = R.memory.alloc_storage(R.shape([32]), virtual_device_index=0, storage_scope="global", dtype="float32")
            alloc: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage, 0, R.shape([2, 4]), dtype="float32")
            _ = R.call_tir_packed(cls.exp, (x, alloc))
            lv: R.Tensor((2, 4), dtype="float32") = alloc
            lv1: R.Tensor((8,), dtype="float32") = R.reshape(lv, (8,))
            storage1: R.Any = R.memory.alloc_storage(R.shape([40]), virtual_device_index=0, storage_scope="global", dtype="float32")
            alloc1: R.Tensor((8,), dtype="float32") = R.memory.alloc_tensor(storage1, 0, R.shape([8]), dtype="float32")
            _ = R.call_tir_packed(cls.relu, (lv1, alloc1))
            lv2: R.Tensor((8,), dtype="float32") = alloc1
            alloc2: R.Tensor((8,), dtype="float32") = R.memory.alloc_tensor(storage, 0, R.shape([8]), dtype="float32")
            _ = R.call_tir_packed(cls.add, (lv2, R.const(1, "float32"), alloc2))
            lv3: R.Tensor((8,), dtype="float32") = alloc2
            alloc3: R.Tensor((10,), dtype="float32") = R.memory.alloc_tensor(storage1, 0, R.shape([10]), dtype="float32")
            _ = R.call_tir_packed(cls.pad, (lv3, alloc3))
            lv4: R.Tensor((10,), dtype="float32") = alloc3
            alloc4: R.Tensor((10,), dtype="float32") = R.builtin.alloc_tensor(R.shape([10]), dtype="float32", runtime_device_index=0)
            _ = R.call_tir_packed(cls.log, (lv4, alloc4))
            gv5: R.Tensor((10,), dtype="float32") = alloc4
            return gv5

    @I.ir_module
    class ExpectedLowered:
        @Ts.function
        def add(rxplaceholder: T.Tensor((T.int64(8),), "float32"), rxplaceholder_1: T.Tensor((), "float32"), T_add: T.Tensor((T.int64(8),), "float32")):
            T.evaluate(0)

        @Ts.function
        def exp(rxplaceholder: T.Tensor((T.int64(2), T.int64(4)), "float32"), compute: T.Tensor((T.int64(2), T.int64(4)), "float32")):
            T.evaluate(0)

        @Ts.function
        def log(rxplaceholder: T.Tensor((T.int64(10),), "float32"), compute: T.Tensor((T.int64(10),), "float32")):
            T.evaluate(0)

        @Ts.function
        def pad(rxplaceholder: T.Tensor((T.int64(8),), "float32"), PadInput: T.Tensor((T.int64(10),), "float32")):
            T.evaluate(0)

        @Ts.function
        def relu(rxplaceholder: T.Tensor((T.int64(8),), "float32"), compute: T.Tensor((T.int64(8),), "float32")):
            T.evaluate(0)

        @Ts.function
        def reshape(rxplaceholder: T.Tensor((T.int64(2), T.int64(4)), "float32"), T_reshape: T.Tensor((T.int64(8),), "float32")):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((2, 4), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = ExpectedLowered
            storage: R.Any = R.vm.alloc_storage(R.shape([32]), R.prim_value(0), R.dtype("uint8"))
            alloc: R.Tensor((2, 4), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([2, 4]), R.dtype("float32"))
            _: R.Tuple = R.call_tir_packed(cls.exp, (x, alloc))
            lv1: R.Tensor((8,), dtype="float32") = R.call_packed("vm.builtin.reshape", alloc, R.shape([8]), ty_args=(R.Tensor((8,), dtype="float32"),))
            _ = R.vm.kill_object(alloc)
            storage1: R.Any = R.vm.alloc_storage(R.shape([40]), R.prim_value(0), R.dtype("uint8"))
            alloc1: R.Tensor((8,), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([8]), R.dtype("float32"))
            _ = R.call_tir_packed(cls.relu, (lv1, alloc1))
            _ = R.vm.kill_object(lv1)
            alloc2: R.Tensor((8,), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([8]), R.dtype("float32"))
            _ = R.vm.kill_object(storage)
            _ = R.call_tir_packed(cls.add, (alloc1, R.const(1, "float32"), alloc2))
            _ = R.vm.kill_object(alloc1)
            alloc3: R.Tensor((10,), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([10]), R.dtype("float32"))
            _ = R.vm.kill_object(storage1)
            _ = R.call_tir_packed(cls.pad, (alloc2, alloc3))
            _ = R.vm.kill_object(alloc2)
            storage2: R.Any = R.vm.alloc_storage(R.shape([40]), R.prim_value(0), R.dtype("uint8"))
            alloc4: R.Tensor((10,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([10]), R.dtype("float32"))
            _ = R.vm.kill_object(storage2)
            _ = R.call_tir_packed(cls.log, (alloc3, alloc4))
            _ = R.vm.kill_object(alloc3)
            return alloc4
    # fmt: on

    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)
    mod = relax.transform.LowerAllocTensor()(mod)
    mod = relax.transform.KillAfterLastUse()(mod)
    mod = relax.transform.LowerRuntimeBuiltin()(mod)
    tvm_tirx.ir.assert_structural_equal(mod, ExpectedLowered)


def test_different_dtype():
    @tvm_tirx.script.ir_module
    class Module:
        @Ts.function
        def add(
            A: T.Tensor((T.int64(2), T.int64(3)), "float32"),
            B: T.Tensor((T.int64(2), T.int64(3)), "float32"),
            C: T.Tensor((T.int64(2), T.int64(3)), "float32"),
        ):
            T.evaluate(0)

        @Ts.function
        def add1(
            A: T.Tensor((T.int64(2), T.int64(3)), "int32"),
            B: T.Tensor((T.int64(2), T.int64(3)), "int32"),
            C: T.Tensor((T.int64(2), T.int64(3)), "int32"),
        ):
            T.evaluate(0)

        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="int32")
        ) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Module
            alloc: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="float32", runtime_device_index=0
            )
            _: R.Tuple() = R.call_tir_packed(cls.add, (x, x, alloc))
            gv: R.Tensor((2, 3), dtype="float32") = alloc
            alloc1: R.Tensor((2, 3), dtype="int32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="int32", runtime_device_index=0
            )
            _1: R.Tuple() = R.call_tir_packed(cls.add1, (y, y, alloc1))
            gv1: R.Tensor((2, 3), dtype="int32") = alloc1
            return x

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function
        def add(
            A: T.Tensor((T.int64(2), T.int64(3)), "float32"),
            B: T.Tensor((T.int64(2), T.int64(3)), "float32"),
            C: T.Tensor((T.int64(2), T.int64(3)), "float32"),
        ):
            T.evaluate(0)

        @Ts.function
        def add1(
            A: T.Tensor((T.int64(2), T.int64(3)), "int32"),
            B: T.Tensor((T.int64(2), T.int64(3)), "int32"),
            C: T.Tensor((T.int64(2), T.int64(3)), "int32"),
        ):
            T.evaluate(0)

        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="int32")
        ) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            storage: R.Any = R.memory.alloc_storage(
                R.shape([24]), virtual_device_index=0, storage_scope="global", dtype="float32"
            )
            alloc: R.Tensor((2, 3), dtype="float32") = R.memory.alloc_tensor(
                storage, 0, R.shape([2, 3]), dtype="float32"
            )
            _: R.Tuple() = R.call_tir_packed(cls.add, (x, x, alloc))
            gv1: R.Tensor((2, 3), dtype="float32") = alloc
            storage1: R.Any = R.memory.alloc_storage(
                R.shape([24]), virtual_device_index=0, storage_scope="global", dtype="int32"
            )
            alloc1: R.Tensor((2, 3), dtype="int32") = R.memory.alloc_tensor(
                storage1, 0, R.shape([2, 3]), dtype="int32"
            )
            _2: R.Tuple() = R.call_tir_packed(cls.add1, (y, y, alloc1))
            gv12: R.Tensor((2, 3), dtype="int32") = alloc1
            return x

    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_dtype_bool():
    @tvm_tirx.script.ir_module
    class Module:
        @Ts.function
        def add1(
            A: T.Tensor((T.int64(2), T.int64(3)), "bool"),
            B: T.Tensor((T.int64(2), T.int64(3)), "bool"),
            C: T.Tensor((T.int64(2), T.int64(3)), "bool"),
        ):
            T.evaluate(0)

        @R.function
        def main(y: R.Tensor((2, 3), dtype="bool")) -> R.Tensor((2, 3), dtype="bool"):
            R.func_attr({"relax.force_pure": True})
            cls = Module
            alloc: R.Tensor((2, 3), dtype="bool") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="bool", runtime_device_index=0
            )
            _1: R.Tuple() = R.call_tir_packed(cls.add1, (y, y, alloc))
            gv1: R.Tensor((2, 3), dtype="bool") = alloc
            return y

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function
        def add1(
            A: T.Tensor((T.int64(2), T.int64(3)), "bool"),
            B: T.Tensor((T.int64(2), T.int64(3)), "bool"),
            C: T.Tensor((T.int64(2), T.int64(3)), "bool"),
        ):
            T.evaluate(0)

        @R.function
        def main(y: R.Tensor((2, 3), dtype="bool")) -> R.Tensor((2, 3), dtype="bool"):
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            storage: R.Any = R.memory.alloc_storage(
                R.shape([6]), virtual_device_index=0, storage_scope="global", dtype="bool"
            )
            alloc: R.Tensor((2, 3), dtype="bool") = R.memory.alloc_tensor(
                storage, 0, R.shape([2, 3]), dtype="bool"
            )
            _2: R.Tuple() = R.call_tir_packed(cls.add1, (y, y, alloc))
            gv12: R.Tensor((2, 3), dtype="bool") = alloc
            return y

    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_same_dtype():
    @tvm_tirx.script.ir_module
    class Module:
        @Ts.function
        def add(
            A: T.Tensor((T.int64(2), T.int64(3)), "float32"),
            B: T.Tensor((T.int64(2), T.int64(3)), "float32"),
            C: T.Tensor((T.int64(2), T.int64(3)), "float32"),
        ):
            T.evaluate(0)

        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")
        ) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Module
            alloc: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="float32", runtime_device_index=0
            )
            _: R.Tuple() = R.call_tir_packed(cls.add, (x, x, alloc))
            gv: R.Tensor((2, 3), dtype="float32") = alloc
            alloc1: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="float32", runtime_device_index=0
            )
            _1: R.Tuple() = R.call_tir_packed(cls.add, (y, y, alloc1))
            gv1: R.Tensor((2, 3), dtype="float32") = alloc1
            return x

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function
        def add(
            A: T.Tensor((T.int64(2), T.int64(3)), "float32"),
            B: T.Tensor((T.int64(2), T.int64(3)), "float32"),
            C: T.Tensor((T.int64(2), T.int64(3)), "float32"),
        ):
            T.evaluate(0)

        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")
        ) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            storage: R.Any = R.memory.alloc_storage(
                R.shape([24]), virtual_device_index=0, storage_scope="global", dtype="float32"
            )
            alloc: R.Tensor((2, 3), dtype="float32") = R.memory.alloc_tensor(
                storage, 0, R.shape([2, 3]), dtype="float32"
            )
            _: R.Tuple() = R.call_tir_packed(cls.add, (x, x, alloc))
            gv1: R.Tensor((2, 3), dtype="float32") = alloc
            alloc1: R.Tensor((2, 3), dtype="float32") = R.memory.alloc_tensor(
                storage, 0, R.shape([2, 3]), dtype="float32"
            )
            _2: R.Tuple() = R.call_tir_packed(cls.add, (y, y, alloc1))
            gv12: R.Tensor((2, 3), dtype="float32") = alloc1
            return x

    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_if_cond():
    @tvm_tirx.script.ir_module
    class Module:
        @Ts.function
        def all_less_than_zero(A: T.Tensor((2, 3), "float32"), B: T.Tensor((), "bool")):
            T.evaluate(0)

        @Ts.function
        def exp(A: T.Tensor((2, 3), "float32"), B: T.Tensor((2, 3), "float32")):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Module
            alloc: R.Tensor((), dtype="bool") = R.builtin.alloc_tensor(
                R.shape([]), dtype="bool", runtime_device_index=0
            )
            _: R.Tuple() = R.call_tir_packed(cls.all_less_than_zero, (x, alloc))
            x1: R.Tensor((), dtype="bool") = alloc
            if x1:
                y: R.Tensor((2, 3), dtype="float32") = x
            else:
                alloc1: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                    R.shape([2, 3]), dtype="float32", runtime_device_index=0
                )
                _1: R.Tuple() = R.call_tir_packed(cls.exp, (x, alloc1))
                gv3: R.Tensor((2, 3), dtype="float32") = alloc1
                y: R.Tensor((2, 3), dtype="float32") = gv3
            return x

    # The pass does no change.
    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm_tirx.ir.assert_structural_equal(mod, Module)


def test_if_then_else():
    @tvm_tirx.script.ir_module
    class Module:
        @Ts.function
        def exp(A: T.Tensor((2, 3), "float32"), B: T.Tensor((2, 3), "float32")):
            T.evaluate(0)

        @R.function
        def main(
            cond: R.Tensor((), dtype="bool"), x: R.Tensor((2, 3), dtype="float32")
        ) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Module
            alloc: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="float32", runtime_device_index=0
            )
            _: R.Tuple() = R.call_tir_packed(cls.exp, (x, alloc))
            y: R.Tensor((2, 3), dtype="float32") = alloc
            if cond:
                z: R.Tensor((2, 3), dtype="float32") = y
            else:
                z: R.Tensor((2, 3), dtype="float32") = y
            return x

    # The pass does no change.
    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm_tirx.ir.assert_structural_equal(mod, Module)


def test_cross_block_use():
    @tvm_tirx.script.ir_module
    class Module:
        @Ts.function
        def exp(A: T.Tensor((2, 3), "float32"), B: T.Tensor((2, 3), "float32")):
            T.evaluate(0)

        @R.function
        def main(
            cond: R.Tensor((), dtype="bool"), x: R.Tensor((2, 3), dtype="float32")
        ) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Module
            alloc: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="float32", runtime_device_index=0
            )
            _: R.Tuple() = R.call_tir_packed(cls.exp, (x, alloc))
            y: R.Tensor((2, 3), dtype="float32") = alloc
            if cond:
                alloc1: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                    R.shape([2, 3]), dtype="float32", runtime_device_index=0
                )
                _1: R.Tuple() = R.call_tir_packed(cls.exp, (y, alloc1))
                y2: R.Tensor((2, 3), dtype="float32") = alloc1
                z: R.Tensor((2, 3), dtype="float32") = y2
            else:
                alloc2: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                    R.shape([2, 3]), dtype="float32", runtime_device_index=0
                )
                _2: R.Tuple() = R.call_tir_packed(cls.exp, (y, alloc2))
                y2: R.Tensor((2, 3), dtype="float32") = alloc2
                z: R.Tensor((2, 3), dtype="float32") = y2
            return x

    # The pass does no change.
    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm_tirx.ir.assert_structural_equal(mod, Module)


def test_nested_tuple():
    @tvm_tirx.script.ir_module
    class Module:
        @Ts.function
        def exp(A: T.Tensor((2, 3), "float32"), B: T.Tensor((2, 3), "float32")):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            alloc: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="float32", runtime_device_index=0
            )
            cls = Module
            _: R.Tuple() = R.call_tir_packed(cls.exp, (x, alloc))
            y1: R.Tensor((2, 3), dtype="float32") = alloc
            alloc1: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="float32", runtime_device_index=0
            )
            _1: R.Tuple() = R.call_tir_packed(cls.exp, (x, alloc1))
            y2: R.Tensor((2, 3), dtype="float32") = alloc1
            alloc2: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="float32", runtime_device_index=0
            )
            _2: R.Tuple() = R.call_tir_packed(cls.exp, (x, alloc2))
            y3: R.Tensor((2, 3), dtype="float32") = alloc2
            t: R.Tuple(R.Tensor((2, 3), dtype="float32"), R.Tensor((2, 3), dtype="float32")) = (
                y1,
                y2,
            )
            nt: R.Tuple(
                R.Tuple(R.Tensor((2, 3), dtype="float32"), R.Tensor((2, 3), dtype="float32")),
                R.Tensor((2, 3), dtype="float32"),
            ) = (t, y3)
            nt0: R.Tuple(R.Tensor((2, 3), dtype="float32"), R.Tensor((2, 3), dtype="float32")) = nt[
                0
            ]
            y1_: R.Tensor((2, 3), dtype="float32") = nt0[0]
            y2_: R.Tensor((2, 3), dtype="float32") = nt0[1]
            y3_: R.Tensor((2, 3), dtype="float32") = nt[1]
            alloc3: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="float32", runtime_device_index=0
            )
            _3: R.Tuple() = R.call_tir_packed(cls.exp, (y1_, alloc3))
            z1: R.Tensor((2, 3), dtype="float32") = alloc3
            alloc4: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="float32", runtime_device_index=0
            )
            _4: R.Tuple() = R.call_tir_packed(cls.exp, (y2_, alloc4))
            z2: R.Tensor((2, 3), dtype="float32") = alloc4
            alloc5: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="float32", runtime_device_index=0
            )
            _5: R.Tuple() = R.call_tir_packed(cls.exp, (y3_, alloc5))
            z3: R.Tensor((2, 3), dtype="float32") = alloc5
            return x

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function
        def exp(A: T.Tensor((2, 3), "float32"), B: T.Tensor((2, 3), "float32")):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            storage: R.Any = R.memory.alloc_storage(
                R.shape([24]), virtual_device_index=0, storage_scope="global", dtype="float32"
            )
            alloc: R.Tensor((2, 3), dtype="float32") = R.memory.alloc_tensor(
                storage, 0, R.shape([2, 3]), dtype="float32"
            )
            _: R.Tuple() = R.call_tir_packed(cls.exp, (x, alloc))
            y1: R.Tensor((2, 3), dtype="float32") = alloc
            storage1: R.Any = R.memory.alloc_storage(
                R.shape([24]), virtual_device_index=0, storage_scope="global", dtype="float32"
            )
            alloc1: R.Tensor((2, 3), dtype="float32") = R.memory.alloc_tensor(
                storage1, 0, R.shape([2, 3]), dtype="float32"
            )
            _1: R.Tuple() = R.call_tir_packed(cls.exp, (x, alloc1))
            y2: R.Tensor((2, 3), dtype="float32") = alloc1
            storage2: R.Any = R.memory.alloc_storage(
                R.shape([24]), virtual_device_index=0, storage_scope="global", dtype="float32"
            )
            alloc2: R.Tensor((2, 3), dtype="float32") = R.memory.alloc_tensor(
                storage2, 0, R.shape([2, 3]), dtype="float32"
            )
            _2: R.Tuple() = R.call_tir_packed(cls.exp, (x, alloc2))
            y3: R.Tensor((2, 3), dtype="float32") = alloc2
            t: R.Tuple(R.Tensor((2, 3), dtype="float32"), R.Tensor((2, 3), dtype="float32")) = (
                y1,
                y2,
            )
            nt: R.Tuple(
                R.Tuple(R.Tensor((2, 3), dtype="float32"), R.Tensor((2, 3), dtype="float32")),
                R.Tensor((2, 3), dtype="float32"),
            ) = (t, y3)
            nt0: R.Tuple(R.Tensor((2, 3), dtype="float32"), R.Tensor((2, 3), dtype="float32")) = nt[
                0
            ]
            y1_: R.Tensor((2, 3), dtype="float32") = nt0[0]
            y2_: R.Tensor((2, 3), dtype="float32") = nt0[1]
            y3_: R.Tensor((2, 3), dtype="float32") = nt[1]
            storage3: R.Any = R.memory.alloc_storage(
                R.shape([24]), virtual_device_index=0, storage_scope="global", dtype="float32"
            )
            alloc3: R.Tensor((2, 3), dtype="float32") = R.memory.alloc_tensor(
                storage3, 0, R.shape([2, 3]), dtype="float32"
            )
            _3: R.Tuple() = R.call_tir_packed(cls.exp, (y1_, alloc3))
            z1: R.Tensor((2, 3), dtype="float32") = alloc3
            alloc4: R.Tensor((2, 3), dtype="float32") = R.memory.alloc_tensor(
                storage, 0, R.shape([2, 3]), dtype="float32"
            )
            _41: R.Tuple() = R.call_tir_packed(cls.exp, (y2_, alloc4))
            z2: R.Tensor((2, 3), dtype="float32") = alloc4
            alloc5: R.Tensor((2, 3), dtype="float32") = R.memory.alloc_tensor(
                storage3, 0, R.shape([2, 3]), dtype="float32"
            )
            _5: R.Tuple() = R.call_tir_packed(cls.exp, (y3_, alloc5))
            z3: R.Tensor((2, 3), dtype="float32") = alloc5
            return x

    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_call_func_other_than_function():
    @tvm_tirx.script.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")):
            R.func_attr({"relax.force_pure": True})
            alloc: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="float32", runtime_device_index=0
            )
            _ = R.add(x, alloc)
            y: R.Tensor((2, 3), dtype="float32") = alloc
            return x

    # The pass does no change.
    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm_tirx.ir.assert_structural_equal(mod, Module)


def test_call_packed_external_func():
    @I.ir_module
    class Module:
        @R.function(pure=False)
        def main(x: R.Tensor((2, 3), "float32")):
            # the extern func may or may not be pure, depends on what we're calling
            alloc: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="float32", runtime_device_index=0
            )
            _ = R.call_packed("extern_func", x, alloc, ty_args=[R.Tuple()])
            y: R.Tensor((2, 3), dtype="float32") = alloc
            alloc1: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="float32", runtime_device_index=0
            )
            _1 = R.call_packed("extern_func", y, alloc1, ty_args=[R.Tuple()])
            z: R.Tensor((2, 3), dtype="float32") = alloc1
            return z

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            storage: R.Any = R.memory.alloc_storage(
                R.shape([24]), R.prim_value(0), R.str("global"), R.dtype("float32")
            )
            alloc: R.Tensor((2, 3), dtype="float32") = R.memory.alloc_tensor(
                storage, R.prim_value(0), R.shape([2, 3]), R.dtype("float32")
            )
            _: R.Tuple = R.call_packed("extern_func", x, alloc, ty_args=(R.Tuple(),))
            y: R.Tensor((2, 3), dtype="float32") = alloc
            alloc1: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), R.dtype("float32"), R.prim_value(0)
            )
            _1: R.Tuple = R.call_packed("extern_func", y, alloc1, ty_args=(R.Tuple(),))
            z: R.Tensor((2, 3), dtype="float32") = alloc1
            return z

    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_symbolic_shape():
    m_exp = T.dynamic("m", "int64")
    n_exp = T.dynamic("n", "int64")
    m_main = T.dynamic("m", "int64")
    n_main = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Module:
        @Ts.function
        def exp(A: T.Tensor((m_exp, n_exp), "float32"), B: T.Tensor((m_exp, n_exp), "float32")):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((m_main, n_main), "float32")):
            R.func_attr({"relax.force_pure": True})
            alloc: R.Tensor((m_main, n_main), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([m_main, n_main]), dtype="float32", runtime_device_index=0
            )
            _ = R.call_tir_packed(Module.exp, (x, alloc))
            y: R.Tensor((m_main, n_main), dtype="float32") = alloc
            return x

    m_exp = T.dynamic("m", "int64")
    n_exp = T.dynamic("n", "int64")
    m_main = T.dynamic("m", "int64")
    n_main = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function
        def exp(A: T.Tensor((m_exp, n_exp), "float32"), B: T.Tensor((m_exp, n_exp), "float32")):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((m_main, n_main), dtype="float32")) -> R.Tensor(
            (m_main, n_main), dtype="float32"
        ):
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            storage: R.Any = R.memory.alloc_storage(
                R.shape([4 * (m_main * n_main)]),
                R.prim_value(0),
                R.str("global"),
                R.dtype("float32"),
            )
            alloc: R.Tensor((m_main, n_main), dtype="float32") = R.memory.alloc_tensor(
                storage, R.prim_value(0), R.shape([m_main, n_main]), R.dtype("float32")
            )
            _: R.Tuple = R.call_tir_packed(cls.exp, (x, alloc))
            y: R.Tensor((m_main, n_main), dtype="float32") = alloc
            return x

    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_zero_reference():
    @tvm_tirx.script.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")):
            R.func_attr({"relax.force_pure": True})
            alloc: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="float32", runtime_device_index=0
            )
            return x

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")):
            R.func_attr({"relax.force_pure": True})
            storage: R.Any = R.memory.alloc_storage(
                R.shape([24]), virtual_device_index=0, storage_scope="global", dtype="float32"
            )
            alloc: R.Tensor((2, 3), dtype="float32") = R.memory.alloc_tensor(
                storage, 0, R.shape([2, 3]), dtype="float32"
            )
            return x

    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_reshape_param():
    @tvm_tirx.script.ir_module
    class Module:
        @Ts.function
        def add(
            A: T.Tensor((T.int64(2), T.int64(25), T.int64(2)), "float32"),
            B: T.Tensor((T.int64(2), T.int64(25), T.int64(2)), "float32"),
            C: T.Tensor((T.int64(2), T.int64(25), T.int64(2)), "float32"),
        ):
            T.evaluate(0)

        @R.function
        def main(
            x: R.Tensor((2, 50), dtype="float32"), y: R.Tensor((100,), dtype="float32")
        ) -> R.Tensor((2, 25, 2), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            lv: R.Tensor((2, 25, 2), dtype="float32") = R.reshape(x, (2, 25, 2))
            lv1: R.Tensor((2, 25, 2), dtype="float32") = R.reshape(y, (2, 25, 2))
            alloc: R.Tensor((2, 25, 2), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 25, 2]), dtype="float32", runtime_device_index=0
            )
            _: R.Tuple() = R.call_tir_packed(Module.add, (lv, lv1, alloc))
            gv: R.Tensor((2, 25, 2), dtype="float32") = alloc
            return gv

    # The pass does no change.
    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm_tirx.ir.assert_structural_equal(mod, Module)


def test_multiple_functions():
    @tvm_tirx.script.ir_module
    class Module:
        @Ts.function
        def add(
            A: T.Tensor((T.int64(2), T.int64(3)), "float32"),
            B: T.Tensor((T.int64(2), T.int64(3)), "float32"),
            C: T.Tensor((T.int64(2), T.int64(3)), "float32"),
        ):
            T.evaluate(0)

        @Ts.function
        def add1(
            A: T.Tensor((T.int64(2), T.int64(3)), "int32"),
            B: T.Tensor((T.int64(2), T.int64(3)), "int32"),
            C: T.Tensor((T.int64(2), T.int64(3)), "int32"),
        ):
            T.evaluate(0)

        @R.function
        def func1(
            x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="int32")
        ) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Module
            alloc: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="float32", runtime_device_index=0
            )
            _: R.Tuple() = R.call_tir_packed(cls.add, (x, x, alloc))
            gv: R.Tensor((2, 3), dtype="float32") = alloc
            alloc1: R.Tensor((2, 3), dtype="int32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="int32", runtime_device_index=0
            )
            _1: R.Tuple() = R.call_tir_packed(cls.add1, (y, y, alloc1))
            gv1: R.Tensor((2, 3), dtype="int32") = alloc1
            return x

        @R.function
        def func2(
            x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")
        ) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Module
            alloc: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="float32", runtime_device_index=0
            )
            _: R.Tuple() = R.call_tir_packed(cls.add, (x, x, alloc))
            gv: R.Tensor((2, 3), dtype="float32") = alloc
            alloc1: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="float32", runtime_device_index=0
            )
            _1: R.Tuple() = R.call_tir_packed(cls.add, (y, y, alloc1))
            gv1: R.Tensor((2, 3), dtype="float32") = alloc1
            return x

    @I.ir_module
    class Expected:
        @Ts.function
        def add(
            A: T.Tensor((T.int64(2), T.int64(3)), "float32"),
            B: T.Tensor((T.int64(2), T.int64(3)), "float32"),
            C: T.Tensor((T.int64(2), T.int64(3)), "float32"),
        ):
            T.evaluate(0)

        @Ts.function
        def add1(
            A: T.Tensor((T.int64(2), T.int64(3)), "int32"),
            B: T.Tensor((T.int64(2), T.int64(3)), "int32"),
            C: T.Tensor((T.int64(2), T.int64(3)), "int32"),
        ):
            T.evaluate(0)

        @R.function
        def func1(
            x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="int32")
        ) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            storage: R.Any = R.memory.alloc_storage(
                R.shape([24]), virtual_device_index=0, storage_scope="global", dtype="float32"
            )
            alloc: R.Tensor((2, 3), dtype="float32") = R.memory.alloc_tensor(
                storage, 0, R.shape([2, 3]), dtype="float32"
            )
            _: R.Tuple() = R.call_tir_packed(cls.add, (x, x, alloc))
            gv1: R.Tensor((2, 3), dtype="float32") = alloc
            storage1: R.Any = R.memory.alloc_storage(
                R.shape([24]), virtual_device_index=0, storage_scope="global", dtype="int32"
            )
            alloc1: R.Tensor((2, 3), dtype="int32") = R.memory.alloc_tensor(
                storage1, 0, R.shape([2, 3]), dtype="int32"
            )
            _2: R.Tuple() = R.call_tir_packed(cls.add1, (y, y, alloc1))
            gv12: R.Tensor((2, 3), dtype="int32") = alloc1
            return x

        @R.function
        def func2(
            x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")
        ) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            storage: R.Any = R.memory.alloc_storage(
                R.shape([24]), virtual_device_index=0, storage_scope="global", dtype="float32"
            )
            alloc: R.Tensor((2, 3), dtype="float32") = R.memory.alloc_tensor(
                storage, 0, R.shape([2, 3]), dtype="float32"
            )
            _: R.Tuple() = R.call_tir_packed(cls.add, (x, x, alloc))
            gv1: R.Tensor((2, 3), dtype="float32") = alloc
            alloc1: R.Tensor((2, 3), dtype="float32") = R.memory.alloc_tensor(
                storage, 0, R.shape([2, 3]), dtype="float32"
            )
            _2: R.Tuple() = R.call_tir_packed(cls.add, (y, y, alloc1))
            gv12: R.Tensor((2, 3), dtype="float32") = alloc1
            return x

    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_tir_var_upper_bound():
    # fmt: off
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Module:
        @Ts.function
        def add(rxplaceholder: T.handle, rxplaceholder_1: T.handle, T_add: T.handle):
            T.evaluate(0)

        @Ts.function
        def reshape(rxplaceholder: T.handle, T_reshape: T.handle):
            T.evaluate(0)

        @Ts.function
        def relu(rxplaceholder: T.handle, compute: T.handle):
            T.evaluate(0)

        @Ts.function
        def log(rxplaceholder: T.handle, compute: T.handle):
            T.evaluate(0)

        @Ts.function
        def exp(rxplaceholder: T.handle, compute: T.handle):
            T.evaluate(0)

        @Ts.function
        def pad(rxplaceholder: T.handle, PadInput: T.handle):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((2, n), dtype="float32")) -> R.Tensor((2 * n + 2,), dtype="float32"):
            R.func_attr({"tir_var_upper_bound": {"n": 4}, "relax.force_pure": True})
            cls = Module
            alloc: R.Tensor((2, n), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, n]), dtype="float32", runtime_device_index=0)
            _: R.Tuple() = R.call_tir_packed(cls.exp, (x, alloc))
            lv: R.Tensor((2, n), dtype="float32") = alloc
            lv1: R.Tensor((2 * n,), dtype="float32") = R.reshape(lv, (2 * n,))
            alloc1: R.Tensor((2 * n,), dtype="float32") = R.builtin.alloc_tensor(R.shape([2 * n]), dtype="float32", runtime_device_index=0)
            _1: R.Tuple() = R.call_tir_packed(cls.relu, (lv1, alloc1))
            lv2: R.Tensor((2 * n,), dtype="float32") = alloc1
            alloc2: R.Tensor((2 * n,), dtype="float32") = R.builtin.alloc_tensor(R.shape([2 * n]), dtype="float32", runtime_device_index=0)
            _2: R.Tuple() = R.call_tir_packed(cls.add, (lv2, R.const(1, "float32"), alloc2))
            lv3: R.Tensor((2 * n,), dtype="float32") = alloc2
            alloc3: R.Tensor((2 * n + 2,), dtype="float32") = R.builtin.alloc_tensor(R.shape([2 * n + 2]), dtype="float32", runtime_device_index=0)
            _3: R.Tuple() = R.call_tir_packed(cls.pad, (lv3, alloc3))
            lv4: R.Tensor((2 * n + 2,), dtype="float32") = alloc3
            alloc4: R.Tensor((2 * n + 2,), dtype="float32") = R.builtin.alloc_tensor(R.shape([10]), dtype="float32", runtime_device_index=0)
            _4: R.Tuple() = R.call_tir_packed(cls.log, (lv4, alloc4))
            gv: R.Tensor((2 * n + 2,), dtype="float32") = alloc4
            return gv

    n = T.dynamic("n", "int64")

    @I.ir_module
    class Expected:
        @Ts.function
        def add(rxplaceholder: T.handle, rxplaceholder_1: T.handle, T_add: T.handle):
            T.evaluate(0)

        @Ts.function
        def exp(rxplaceholder: T.handle, compute: T.handle):
            T.evaluate(0)

        @Ts.function
        def log(rxplaceholder: T.handle, compute: T.handle):
            T.evaluate(0)

        @Ts.function
        def pad(rxplaceholder: T.handle, PadInput: T.handle):
            T.evaluate(0)

        @Ts.function
        def relu(rxplaceholder: T.handle, compute: T.handle):
            T.evaluate(0)

        @Ts.function
        def reshape(rxplaceholder: T.handle, T_reshape: T.handle):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((2, n), dtype="float32")) -> R.Tensor((2 * n + 2,), dtype="float32"):
            R.func_attr({"tir_var_upper_bound": {"n": 4}, "relax.force_pure": True})
            cls = Expected
            storage: R.Any = R.memory.alloc_storage(R.shape([32]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            alloc: R.Tensor((2, n), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([2, n]), R.dtype("float32"))
            _: R.Tuple = R.call_tir_packed(cls.exp, (x, alloc))
            lv: R.Tensor((2, n), dtype="float32") = alloc
            lv1: R.Tensor((2 * n,), dtype="float32") = R.reshape(lv, R.shape([2 * n]))
            storage1: R.Any = R.memory.alloc_storage(R.shape([40]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            alloc1: R.Tensor((2 * n,), dtype="float32") = R.memory.alloc_tensor(storage1, R.prim_value(0), R.shape([2 * n]), R.dtype("float32"))
            _1: R.Tuple = R.call_tir_packed(cls.relu, (lv1, alloc1))
            lv2: R.Tensor((2 * n,), dtype="float32") = alloc1
            alloc2: R.Tensor((2 * n,), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([2 * n]), R.dtype("float32"))
            _2: R.Tuple = R.call_tir_packed(cls.add, (lv2, R.const(1, "float32"), alloc2))
            lv3: R.Tensor((2 * n,), dtype="float32") = alloc2
            alloc3: R.Tensor((2 * n + 2,), dtype="float32") = R.memory.alloc_tensor(storage1, R.prim_value(0), R.shape([2 * n + 2]), R.dtype("float32"))
            _3: R.Tuple = R.call_tir_packed(cls.pad, (lv3, alloc3))
            lv4: R.Tensor((2 * n + 2,), dtype="float32") = alloc3
            alloc4: R.Tensor((2 * n + 2,), dtype="float32") = R.builtin.alloc_tensor(R.shape([10]), R.dtype("float32"), R.prim_value(0))
            _4: R.Tuple = R.call_tir_packed(cls.log, (lv4, alloc4))
            gv: R.Tensor((2 * n + 2,), dtype="float32") = alloc4
            return gv
    # fmt: on

    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_lower_bound_only():
    # fmt: off
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Module:
        @Ts.function
        def add(rxplaceholder: T.handle, rxplaceholder_1: T.handle, T_add: T.handle):
            T.evaluate(0)

        @Ts.function
        def reshape(rxplaceholder: T.handle, T_reshape: T.handle):
            T.evaluate(0)

        @Ts.function
        def relu(rxplaceholder: T.handle, compute: T.handle):
            T.evaluate(0)

        @Ts.function
        def log(rxplaceholder: T.handle, compute: T.handle):
            T.evaluate(0)

        @Ts.function
        def exp(rxplaceholder: T.handle, compute: T.handle):
            T.evaluate(0)

        @Ts.function
        def pad(rxplaceholder: T.handle, PadInput: T.handle):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((2, n), dtype="float32")) -> R.Tensor((2 * n + 2,), dtype="float32"):
            R.func_attr({"tir_var_lower_bound": {"n": 2}, "relax.force_pure": True})
            cls = Module
            alloc: R.Tensor((2, n), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, n]), dtype="float32", runtime_device_index=0)
            _: R.Tuple() = R.call_tir_packed(cls.exp, (x, alloc))
            lv: R.Tensor((2, n), dtype="float32") = alloc
            lv1: R.Tensor((2 * n,), dtype="float32") = R.reshape(lv, (2 * n,))
            alloc1: R.Tensor((2 * n,), dtype="float32") = R.builtin.alloc_tensor(R.shape([2 * n]), dtype="float32", runtime_device_index=0)
            _1: R.Tuple() = R.call_tir_packed(cls.relu, (lv1, alloc1))
            lv2: R.Tensor((2 * n,), dtype="float32") = alloc1
            alloc2: R.Tensor((2 * n,), dtype="float32") = R.builtin.alloc_tensor(R.shape([2 * n]), dtype="float32", runtime_device_index=0)
            _2: R.Tuple() = R.call_tir_packed(cls.add, (lv2, R.const(1, "float32"), alloc2))
            lv3: R.Tensor((2 * n,), dtype="float32") = alloc2
            alloc3: R.Tensor((2 * n + 2,), dtype="float32") = R.builtin.alloc_tensor(R.shape([2 * n + 2]), dtype="float32", runtime_device_index=0)
            _3: R.Tuple() = R.call_tir_packed(cls.pad, (lv3, alloc3))
            lv4: R.Tensor((2 * n + 2,), dtype="float32") = alloc3
            alloc4: R.Tensor((2 * n + 2,), dtype="float32") = R.builtin.alloc_tensor(R.shape([10]), dtype="float32", runtime_device_index=0)
            _4: R.Tuple() = R.call_tir_packed(cls.log, (lv4, alloc4))
            gv: R.Tensor((2 * n + 2,), dtype="float32") = alloc4
            return gv

    n = T.dynamic("n", "int64")

    @I.ir_module
    class Expected:
        @Ts.function
        def add(rxplaceholder: T.handle, rxplaceholder_1: T.handle, T_add: T.handle):
            T.evaluate(0)

        @Ts.function
        def exp(rxplaceholder: T.handle, compute: T.handle):
            T.evaluate(0)

        @Ts.function
        def log(rxplaceholder: T.handle, compute: T.handle):
            T.evaluate(0)

        @Ts.function
        def pad(rxplaceholder: T.handle, PadInput: T.handle):
            T.evaluate(0)

        @Ts.function
        def relu(rxplaceholder: T.handle, compute: T.handle):
            T.evaluate(0)

        @Ts.function
        def reshape(rxplaceholder: T.handle, T_reshape: T.handle):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((2, n), dtype="float32")) -> R.Tensor((2 * n + 2,), dtype="float32"):
            R.func_attr({"tir_var_lower_bound": {"n": 2}, "relax.force_pure": True})
            cls = Expected
            storage: R.Any = R.memory.alloc_storage(R.shape([8 * n]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            alloc: R.Tensor((2, n), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([2, n]), R.dtype("float32"), R.prim_value(0))
            _: R.Tuple = R.call_tir_packed(cls.exp, (x, alloc))
            lv: R.Tensor((2, n), dtype="float32") = alloc
            lv1: R.Tensor((2 * n,), dtype="float32") = R.reshape(lv, R.shape([2 * n]))
            storage1: R.Any = R.memory.alloc_storage(R.shape([4 * (2 * n)]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            alloc1: R.Tensor((2 * n,), dtype="float32") = R.memory.alloc_tensor(storage1, R.prim_value(0), R.shape([2 * n]), R.dtype("float32"))
            _1: R.Tuple = R.call_tir_packed(cls.relu, (lv1, alloc1))
            lv2: R.Tensor((2 * n,), dtype="float32") = alloc1
            alloc2: R.Tensor((2 * n,), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([2 * n]), R.dtype("float32"))
            _2: R.Tuple = R.call_tir_packed(cls.add, (lv2, R.const(1, "float32"), alloc2))
            lv3: R.Tensor((2 * n,), dtype="float32") = alloc2
            storage2: R.Any = R.memory.alloc_storage(R.shape([4 * (2 * n + 2)]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            alloc3: R.Tensor((2 * n + 2,), dtype="float32") = R.memory.alloc_tensor(storage2, R.prim_value(0), R.shape([2 * n + 2]), R.dtype("float32"), R.prim_value(0))
            _3: R.Tuple = R.call_tir_packed(cls.pad, (lv3, alloc3))
            lv4: R.Tensor((2 * n + 2,), dtype="float32") = alloc3
            alloc4: R.Tensor((2 * n + 2,), dtype="float32") = R.builtin.alloc_tensor(R.shape([10]), R.dtype("float32"), R.prim_value(0))
            _4: R.Tuple = R.call_tir_packed(cls.log, (lv4, alloc4))
            gv: R.Tensor((2 * n + 2,), dtype="float32") = alloc4
            return gv
    # fmt: on

    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_upper_and_lower_bounds():
    # fmt: off
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Module:
        @Ts.function
        def add(rxplaceholder: T.handle, rxplaceholder_1: T.handle, T_add: T.handle):
            T.evaluate(0)

        @Ts.function
        def reshape(rxplaceholder: T.handle, T_reshape: T.handle):
            T.evaluate(0)

        @Ts.function
        def relu(rxplaceholder: T.handle, compute: T.handle):
            T.evaluate(0)

        @Ts.function
        def log(rxplaceholder: T.handle, compute: T.handle):
            T.evaluate(0)

        @Ts.function
        def exp(rxplaceholder: T.handle, compute: T.handle):
            T.evaluate(0)

        @Ts.function
        def pad(rxplaceholder: T.handle, PadInput: T.handle):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((2, n), dtype="float32")) -> R.Tensor((2 * n + 2,), dtype="float32"):
            R.func_attr({"tir_var_upper_bound": {"n": 4}, "tir_var_lower_bound": {"n": 2}, "relax.force_pure": True})
            cls = Module
            alloc: R.Tensor((2, n), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, n]), dtype="float32", runtime_device_index=0)
            _: R.Tuple() = R.call_tir_packed(cls.exp, (x, alloc))
            lv: R.Tensor((2, n), dtype="float32") = alloc
            lv1: R.Tensor((2 * n,), dtype="float32") = R.reshape(lv, (2 * n,))
            alloc1: R.Tensor((2 * n,), dtype="float32") = R.builtin.alloc_tensor(R.shape([2 * n]), dtype="float32", runtime_device_index=0)
            _1: R.Tuple() = R.call_tir_packed(cls.relu, (lv1, alloc1))
            lv2: R.Tensor((2 * n,), dtype="float32") = alloc1
            alloc2: R.Tensor((2 * n,), dtype="float32") = R.builtin.alloc_tensor(R.shape([2 * n]), dtype="float32", runtime_device_index=0)
            _2: R.Tuple() = R.call_tir_packed(cls.add, (lv2, R.const(1, "float32"), alloc2))
            lv3: R.Tensor((2 * n,), dtype="float32") = alloc2
            alloc3: R.Tensor((2 * n + 2,), dtype="float32") = R.builtin.alloc_tensor(R.shape([2 * n + 2]), dtype="float32", runtime_device_index=0)
            _3: R.Tuple() = R.call_tir_packed(cls.pad, (lv3, alloc3))
            lv4: R.Tensor((2 * n + 2,), dtype="float32") = alloc3
            alloc4: R.Tensor((2 * n + 2,), dtype="float32") = R.builtin.alloc_tensor(R.shape([10]), dtype="float32", runtime_device_index=0)
            _4: R.Tuple() = R.call_tir_packed(cls.log, (lv4, alloc4))
            gv: R.Tensor((2 * n + 2,), dtype="float32") = alloc4
            return gv

    n = T.dynamic("n", "int64")

    @I.ir_module
    class Expected:
        @Ts.function
        def add(rxplaceholder: T.handle, rxplaceholder_1: T.handle, T_add: T.handle):
            T.evaluate(0)

        @Ts.function
        def exp(rxplaceholder: T.handle, compute: T.handle):
            T.evaluate(0)

        @Ts.function
        def log(rxplaceholder: T.handle, compute: T.handle):
            T.evaluate(0)

        @Ts.function
        def pad(rxplaceholder: T.handle, PadInput: T.handle):
            T.evaluate(0)

        @Ts.function
        def relu(rxplaceholder: T.handle, compute: T.handle):
            T.evaluate(0)

        @Ts.function
        def reshape(rxplaceholder: T.handle, T_reshape: T.handle):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((2, n), dtype="float32")) -> R.Tensor((2 * n + 2,), dtype="float32"):
            R.func_attr({"tir_var_upper_bound": {"n": 4}, "tir_var_lower_bound": {"n": 2}, "relax.force_pure": True})
            cls = Expected
            storage: R.Any = R.memory.alloc_storage(R.shape([32]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            alloc: R.Tensor((2, n), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([2, n]), R.dtype("float32"))
            _: R.Tuple = R.call_tir_packed(cls.exp, (x, alloc))
            lv: R.Tensor((2, n), dtype="float32") = alloc
            lv1: R.Tensor((2 * n,), dtype="float32") = R.reshape(lv, R.shape([2 * n]))
            storage1: R.Any = R.memory.alloc_storage(R.shape([40]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            alloc1: R.Tensor((2 * n,), dtype="float32") = R.memory.alloc_tensor(storage1, R.prim_value(0), R.shape([2 * n]), R.dtype("float32"))
            _1: R.Tuple = R.call_tir_packed(cls.relu, (lv1, alloc1))
            lv2: R.Tensor((2 * n,), dtype="float32") = alloc1
            alloc2: R.Tensor((2 * n,), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([2 * n]), R.dtype("float32"))
            _2: R.Tuple = R.call_tir_packed(cls.add, (lv2, R.const(1, "float32"), alloc2))
            lv3: R.Tensor((2 * n,), dtype="float32") = alloc2
            alloc3: R.Tensor((2 * n + 2,), dtype="float32") = R.memory.alloc_tensor(storage1, R.prim_value(0), R.shape([2 * n + 2]), R.dtype("float32"))
            _3: R.Tuple = R.call_tir_packed(cls.pad, (lv3, alloc3))
            lv4: R.Tensor((2 * n + 2,), dtype="float32") = alloc3
            alloc4: R.Tensor((2 * n + 2,), dtype="float32") = R.builtin.alloc_tensor(R.shape([10]), R.dtype("float32"), R.prim_value(0))
            _4: R.Tuple = R.call_tir_packed(cls.log, (lv4, alloc4))
            gv: R.Tensor((2 * n + 2,), dtype="float32") = alloc4
            return gv
    # fmt: on

    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_invalid_tir_var_upper_bound():
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor((2, n), dtype="float32")):
            R.func_attr({"tir_var_upper_bound": {"n": [4]}, "relax.force_pure": True})
            return x

    with pytest.raises((RuntimeError, TypeError)):
        relax.transform.StaticPlanBlockMemory()(Module)


def test_invalid_tir_var_lower_bound():
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor((2, n), dtype="float32")):
            R.func_attr({"tir_var_lower_bound": {"n": [4]}, "relax.force_pure": True})
            return x

    with pytest.raises((RuntimeError, TypeError)):
        relax.transform.StaticPlanBlockMemory()(Module)


def test_tir_var_decreasing_monotone():
    # fmt: off
    n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int64")

    @I.ir_module
    class Module:
        @Ts.function
        def tir_exp(var_rxplaceholder: T.handle, var_compute: T.handle):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((n, m, T.max(n - m, 1)), dtype="float32")) -> R.Tensor((n, m, T.max(n - m, 1)), dtype="float32"):
            R.func_attr({"tir_var_upper_bound": {"m": 5, "n": 20}, "relax.force_pure": True})
            cls = Module
            alloc: R.Tensor((n, m, T.max(n - m, 1)), dtype="float32") = R.builtin.alloc_tensor(R.shape([n, m, T.max(n - m, 1)]), R.dtype("float32"), R.prim_value(0))
            _: R.Tuple = R.call_tir_packed(cls.tir_exp, (x, alloc))
            y: R.Tensor((n, m, T.max(n - m, 1)), dtype="float32") = alloc
            alloc1: R.Tensor((n, m, T.max(n - m, 1)), dtype="float32") = R.builtin.alloc_tensor(R.shape([n, m, T.max(n - m, 1)]), R.dtype("float32"), R.prim_value(0))
            _1: R.Tuple = R.call_tir_packed(cls.tir_exp, (y, alloc1))
            z: R.Tensor((n, m, T.max(n - m, 1)), dtype="float32") = alloc1
            alloc2: R.Tensor((n, m, T.max(n - m, 1)), dtype="float32") = R.builtin.alloc_tensor(R.shape([n, m, T.max(n - m, 1)]), R.dtype("float32"), R.prim_value(0))
            _2: R.Tuple = R.call_tir_packed(cls.tir_exp, (z, alloc2))
            r: R.Tensor((n, m, T.max(n - m, 1)), dtype="float32") = alloc2
            return r

    n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int64")

    @I.ir_module
    class Expected:
        @Ts.function
        def tir_exp(var_rxplaceholder: T.handle, var_compute: T.handle):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((n, m, T.max(n - m, 1)), dtype="float32")) -> R.Tensor((n, m, T.max(n - m, 1)), dtype="float32"):
            R.func_attr({"tir_var_upper_bound": {"m": 5, "n": 20}, "relax.force_pure": True})
            cls = Expected
            storage: R.Any = R.memory.alloc_storage(R.shape([8000]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            alloc: R.Tensor((n, m, T.max(n - m, 1)), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([n, m, T.max(n - m, 1)]), R.dtype("float32"))
            _: R.Tuple = R.call_tir_packed(cls.tir_exp, (x, alloc))
            y: R.Tensor((n, m, T.max(n - m, 1)), dtype="float32") = alloc
            storage1: R.Any = R.memory.alloc_storage(R.shape([8000]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            alloc1: R.Tensor((n, m, T.max(n - m, 1)), dtype="float32") = R.memory.alloc_tensor(storage1, R.prim_value(0), R.shape([n, m, T.max(n - m, 1)]), R.dtype("float32"))
            _1: R.Tuple = R.call_tir_packed(cls.tir_exp, (y, alloc1))
            z: R.Tensor((n, m, T.max(n - m, 1)), dtype="float32") = alloc1
            alloc2: R.Tensor((n, m, T.max(n - m, 1)), dtype="float32") = R.builtin.alloc_tensor(R.shape([n, m, T.max(n - m, 1)]), R.dtype("float32"), R.prim_value(0))
            _2: R.Tuple = R.call_tir_packed(cls.tir_exp, (z, alloc2))
            r: R.Tensor((n, m, T.max(n - m, 1)), dtype="float32") = alloc2
            return r
    # fmt: on

    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_call_tir_dyn():
    # fmt: off
    n = T.dynamic("n", "int64")

    @I.ir_module
    class Module:
        @Ts.function
        def tir_full(var_full: T.handle, n: T.int64):
            T.evaluate(0)

        @Ts.function
        def tir_exp(var_rxplaceholder: T.handle, var_compute: T.handle):
            T.evaluate(0)

        @R.function
        def main(s: R.Shape([n])) -> R.Tensor((n,), dtype="float32"):
            R.func_attr({"tir_var_upper_bound": {"n": 20}, "relax.force_pure": True})
            cls = Module
            alloc: R.Tensor((n,), dtype="float32") = R.builtin.alloc_tensor(R.shape([n]), R.dtype("float32"), R.prim_value(0))
            _: R.Tuple = R.vm.call_tir_dyn(cls.tir_full, (alloc, R.shape([n])))
            full: R.Tensor((n,), dtype="float32") = alloc
            alloc1: R.Tensor((n,), dtype="float32") = R.builtin.alloc_tensor(R.shape([n]), R.dtype("float32"), R.prim_value(0))
            _1: R.Tuple = R.call_tir_packed(cls.tir_exp, (full, alloc1))
            lv2: R.Tensor((n,), dtype="float32") = alloc1
            alloc2: R.Tensor((n,), dtype="float32") = R.builtin.alloc_tensor(R.shape([n]), R.dtype("float32"), R.prim_value(0))
            _2: R.Tuple = R.call_tir_packed(cls.tir_exp, (lv2, alloc2))
            lv3: R.Tensor((n,), dtype="float32") = alloc2
            return lv3

    n = T.dynamic("n", "int64")

    @I.ir_module
    class Expected:
        @Ts.function
        def tir_exp(var_rxplaceholder: T.handle, var_compute: T.handle):
            T.evaluate(0)

        @Ts.function
        def tir_full(var_full: T.handle, n: T.int64):
            T.evaluate(0)

        @R.function
        def main(s: R.Shape([n])) -> R.Tensor((n,), dtype="float32"):
            R.func_attr({"tir_var_upper_bound": {"n": 20}, "relax.force_pure": True})
            cls = Expected
            storage: R.Any = R.memory.alloc_storage(R.shape([80]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            alloc: R.Tensor((n,), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([n]), R.dtype("float32"))
            _: R.Tuple = R.vm.call_tir_dyn(cls.tir_full, (alloc, R.shape([n])))
            full: R.Tensor((n,), dtype="float32") = alloc
            storage1: R.Any = R.memory.alloc_storage(R.shape([80]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            alloc1: R.Tensor((n,), dtype="float32") = R.memory.alloc_tensor(storage1, R.prim_value(0), R.shape([n]), R.dtype("float32"))
            _1: R.Tuple = R.call_tir_packed(cls.tir_exp, (full, alloc1))
            lv2: R.Tensor((n,), dtype="float32") = alloc1
            alloc2: R.Tensor((n,), dtype="float32") = R.builtin.alloc_tensor(R.shape([n]), R.dtype("float32"), R.prim_value(0))
            _2: R.Tuple = R.call_tir_packed(cls.tir_exp, (lv2, alloc2))
            lv3: R.Tensor((n,), dtype="float32") = alloc2
            return lv3
    # fmt: on

    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_call_tir_dyn_plan_dynamic_func_output():
    # fmt: off
    n = T.dynamic("n", "int64")

    @I.ir_module
    class Module:
        @Ts.function
        def tir_full(var_full: T.handle, n: T.int64):
            T.evaluate(0)

        @Ts.function
        def tir_exp(var_rxplaceholder: T.handle, var_compute: T.handle):
            T.evaluate(0)

        @R.function
        def main(s: R.Shape([n])) -> R.Tensor((n,), dtype="float32"):
            R.func_attr({"tir_var_upper_bound": {"n": 20}, "relax.force_pure": True, "relax.memory_plan_dynamic_func_output": True})
            cls = Module
            alloc: R.Tensor((n,), dtype="float32") = R.builtin.alloc_tensor(R.shape([n]), R.dtype("float32"), R.prim_value(0))
            _: R.Tuple = R.vm.call_tir_dyn(cls.tir_full, (alloc, R.shape([n])))
            full: R.Tensor((n,), dtype="float32") = alloc
            alloc1: R.Tensor((n,), dtype="float32") = R.builtin.alloc_tensor(R.shape([n]), R.dtype("float32"), R.prim_value(0))
            _1: R.Tuple = R.call_tir_packed(cls.tir_exp, (full, alloc1))
            lv2: R.Tensor((n,), dtype="float32") = alloc1
            alloc2: R.Tensor((n,), dtype="float32") = R.builtin.alloc_tensor(R.shape([n]), R.dtype("float32"), R.prim_value(0))
            _2: R.Tuple = R.call_tir_packed(cls.tir_exp, (lv2, alloc2))
            lv3: R.Tensor((n,), dtype="float32") = alloc2
            return lv3

    n = T.dynamic("n", "int64")

    @I.ir_module
    class Expected:
        @Ts.function
        def tir_exp(var_rxplaceholder: T.handle, var_compute: T.handle):
            T.evaluate(0)

        @Ts.function
        def tir_full(var_full: T.handle, n: T.int64):
            T.evaluate(0)

        @R.function
        def main(s: R.Shape([n])) -> R.Tensor((n,), dtype="float32"):
            R.func_attr({"tir_var_upper_bound": {"n": 20}, "relax.force_pure": True})
            cls = Expected
            storage: R.Any = R.memory.alloc_storage(R.shape([80]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            alloc: R.Tensor((n,), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([n]), R.dtype("float32"))
            _: R.Tuple = R.vm.call_tir_dyn(cls.tir_full, (alloc, R.shape([n])))
            full: R.Tensor((n,), dtype="float32") = alloc
            storage1: R.Any = R.memory.alloc_storage(R.shape([80]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            alloc1: R.Tensor((n,), dtype="float32") = R.memory.alloc_tensor(storage1, R.prim_value(0), R.shape([n]), R.dtype("float32"))
            _1: R.Tuple = R.call_tir_packed(cls.tir_exp, (full, alloc1))
            lv2: R.Tensor((n,), dtype="float32") = alloc1
            storage2: R.Any = R.memory.alloc_storage(R.shape([80]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            alloc2: R.Tensor((n,), dtype="float32") = R.memory.alloc_tensor(storage2, R.prim_value(0), R.shape([n]), R.dtype("float32"))
            _2: R.Tuple = R.call_tir_packed(cls.tir_exp, (lv2, alloc2))
            lv3: R.Tensor((n,), dtype="float32") = alloc2
            return lv3
    # fmt: on

    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_call_tir_dyn_plan_partially_dynamic():
    # fmt: off
    n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int64")

    @I.ir_module
    class Module:
        @Ts.function
        def tir_full(var_full: T.handle, n: T.int64, m: T.int64):
            T.evaluate(0)

        @Ts.function
        def tir_exp(var_rxplaceholder: T.handle, var_compute: T.handle):
            T.evaluate(0)

        @R.function
        def main(s: R.Shape([n, m])) -> R.Tensor((n, m), dtype="float32"):
            R.func_attr({"tir_var_upper_bound": {"n": 20}, "relax.force_pure": True, "relax.memory_plan_dynamic_func_output": True})
            cls = Module
            alloc: R.Tensor((n, m), dtype="float32") = R.builtin.alloc_tensor(R.shape([n, m]), R.dtype("float32"), R.prim_value(0))
            _: R.Tuple = R.vm.call_tir_dyn(cls.tir_full, (alloc, R.shape([n, m])))
            full: R.Tensor((n, m), dtype="float32") = alloc
            alloc1: R.Tensor((n, m), dtype="float32") = R.builtin.alloc_tensor(R.shape([n, m]), R.dtype("float32"), R.prim_value(0))
            _1: R.Tuple = R.call_tir_packed(cls.tir_exp, (full, alloc1))
            lv2: R.Tensor((n, m), dtype="float32") = alloc1
            alloc2: R.Tensor((n, m), dtype="float32") = R.builtin.alloc_tensor(R.shape([n, m]), R.dtype("float32"), R.prim_value(0))
            _2: R.Tuple = R.call_tir_packed(cls.tir_exp, (lv2, alloc2))
            lv3: R.Tensor((n, m), dtype="float32") = alloc2
            alloc3: R.Tensor((n, m), dtype="float32") = R.builtin.alloc_tensor(R.shape([n, m]), R.dtype("float32"), R.prim_value(0))
            _3: R.Tuple = R.call_tir_packed(cls.tir_exp, (lv3, alloc3))
            lv4: R.Tensor((n, m), dtype="float32") = alloc3
            return lv4

    n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int64")

    @I.ir_module
    class Expected:
        @Ts.function
        def tir_full(var_full: T.handle, n: T.int64, m: T.int64):
            T.evaluate(0)

        @Ts.function
        def tir_exp(var_rxplaceholder: T.handle, var_compute: T.handle):
            T.evaluate(0)

        @R.function
        def main(s: R.Shape([n, m])) -> R.Tensor((n, m), dtype="float32"):
            R.func_attr({"relax.force_pure": True, "tir_var_upper_bound": {"n": 20}})
            cls = Expected
            storage: R.Any = R.memory.alloc_storage(R.shape([80 * m]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            alloc: R.Tensor((n, m), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([n, m]), R.dtype("float32"))
            _: R.Tuple = R.vm.call_tir_dyn(cls.tir_full, (alloc, R.shape([n, m])))
            full: R.Tensor((n, m), dtype="float32") = alloc
            storage1: R.Any = R.memory.alloc_storage(R.shape([80 * m]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            alloc1: R.Tensor((n, m), dtype="float32") = R.memory.alloc_tensor(storage1, R.prim_value(0), R.shape([n, m]), R.dtype("float32"))
            _1: R.Tuple = R.call_tir_packed(cls.tir_exp, (full, alloc1))
            lv2: R.Tensor((n, m), dtype="float32") = alloc1
            alloc2: R.Tensor((n, m), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([n, m]), R.dtype("float32"))
            _2: R.Tuple = R.call_tir_packed(cls.tir_exp, (lv2, alloc2))
            lv3: R.Tensor((n, m), dtype="float32") = alloc2
            storage2: R.Any = R.memory.alloc_storage(R.shape([20 * m * 4]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            alloc3: R.Tensor((n, m), dtype="float32") = R.memory.alloc_tensor(storage2, R.prim_value(0), R.shape([n, m]), R.dtype("float32"))
            _3: R.Tuple = R.call_tir_packed(cls.tir_exp, (lv3, alloc3))
            lv4 = alloc3
            return lv4
    # fmt: on

    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_function_independence():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Module:
        @Ts.function
        def exp(A: T.handle, B: T.handle):
            T.evaluate(0)

        @R.function
        def func1(x: R.Tensor((8,), dtype="float32")) -> R.Tensor((8,), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Module
            alloc: R.Tensor((8,), dtype="float32") = R.builtin.alloc_tensor(R.shape([8,]), dtype="float32", runtime_device_index=0)
            _: R.Tuple() = R.call_tir_packed(cls.exp, (x, alloc))
            lv: R.Tensor((8,), dtype="float32") = alloc
            alloc1: R.Tensor((8,), dtype="float32") = R.builtin.alloc_tensor(R.shape([8,]), dtype="float32", runtime_device_index=0)
            _1: R.Tuple() = R.call_tir_packed(cls.exp, (lv, alloc1))
            gv: R.Tensor((8,), dtype="float32") = alloc1
            return gv

        @R.function
        def func2(x: R.Tensor((10,), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Module
            alloc: R.Tensor((10,), dtype="float32") = R.builtin.alloc_tensor(R.shape([10,]), dtype="float32", runtime_device_index=0)
            _: R.Tuple() = R.call_tir_packed(cls.exp, (x, alloc))
            lv: R.Tensor((10,), dtype="float32") = alloc
            alloc1: R.Tensor((10,), dtype="float32") = R.builtin.alloc_tensor(R.shape([10,]), dtype="float32", runtime_device_index=0)
            _1: R.Tuple() = R.call_tir_packed(cls.exp, (lv, alloc1))
            gv: R.Tensor((10,), dtype="float32") = alloc1
            return gv

    @I.ir_module
    class Expected:
        @Ts.function
        def exp(A: T.handle, B: T.handle):
            T.evaluate(0)

        @R.function
        def func1(x: R.Tensor((8,), dtype="float32")) -> R.Tensor((8,), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            storage: R.Any = R.memory.alloc_storage(R.shape([32]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            alloc: R.Tensor((8,), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([8]), R.dtype("float32"))
            _: R.Tuple = R.call_tir_packed(cls.exp, (x, alloc))
            lv: R.Tensor((8,), dtype="float32") = alloc
            alloc1: R.Tensor((8,), dtype="float32") = R.builtin.alloc_tensor(R.shape([8]), R.dtype("float32"), R.prim_value(0))
            _1: R.Tuple = R.call_tir_packed(cls.exp, (lv, alloc1))
            gv: R.Tensor((8,), dtype="float32") = alloc1
            return gv

        @R.function
        def func2(x: R.Tensor((10,), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            storage1: R.Any = R.memory.alloc_storage(R.shape([40]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            alloc: R.Tensor((10,), dtype="float32") = R.memory.alloc_tensor(storage1, R.prim_value(0), R.shape([10]), R.dtype("float32"))
            _: R.Tuple = R.call_tir_packed(cls.exp, (x, alloc))
            lv: R.Tensor((10,), dtype="float32") = alloc
            alloc1: R.Tensor((10,), dtype="float32") = R.builtin.alloc_tensor(R.shape([10]), R.dtype("float32"), R.prim_value(0))
            _1: R.Tuple = R.call_tir_packed(cls.exp, (lv, alloc1))
            gv: R.Tensor((10,), dtype="float32") = alloc1
            return gv
    # fmt: on

    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_add():
    batch_size = T.dynamic("batch_size", "int64")
    vocab_size = T.dynamic("vocab_size", "int64")

    @I.ir_module
    class Module:
        @Ts.function(private=True)
        def cumsum(var_A: T.handle, var_A_1: T.handle, var_exclusive_scan_thrust: T.handle):
            T.evaluate(0)

        @R.function
        def main(probs: R.Tensor((batch_size, vocab_size), dtype="float32")) -> R.Tensor(
            (batch_size, vocab_size), dtype="float32"
        ):
            R.func_attr(
                {
                    "relax.force_pure": True,
                    "relax.memory_plan_dynamic_func_output": 1,
                    "tir_var_upper_bound": {"batch_size": 32},
                    "tir_non_negative_var": ["vocab_size"],
                }
            )
            cls = Module
            lv1: R.Tensor(
                (2 * (batch_size * vocab_size * 4) + 4194304,),
                dtype="uint8",
            ) = R.builtin.alloc_tensor(
                R.shape([2 * (batch_size * vocab_size * 4) + 4194304]),
                R.dtype("uint8"),
                R.prim_value(0),
                R.str("global"),
            )
            alloc1: R.Tensor((batch_size, vocab_size), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([batch_size, vocab_size]),
                R.dtype("float32"),
                R.prim_value(0),
                R.str("global"),
            )
            R.call_tir_packed(cls.cumsum, (probs, lv1, alloc1))
            cumsum: R.Tensor((batch_size, vocab_size), dtype="float32") = alloc1
            lv1_1: R.Tensor((batch_size, vocab_size), dtype="float32") = R.call_packed(
                "vm.builtin.reshape",
                cumsum,
                R.shape([batch_size, vocab_size]),
                ty_args=(R.Tensor((batch_size, vocab_size), dtype="float32"),),
            )
            return lv1_1

    batch_size = T.dynamic("batch_size", "int64")
    vocab_size = T.dynamic("vocab_size", "int64")

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def cumsum(var_A: T.handle, var_A_1: T.handle, var_exclusive_scan_thrust: T.handle):
            T.evaluate(0)

        @R.function
        def main(probs: R.Tensor((batch_size, vocab_size), dtype="float32")) -> R.Tensor(
            (batch_size, vocab_size), dtype="float32"
        ):
            R.func_attr(
                {
                    "relax.force_pure": True,
                    "tir_non_negative_var": ["vocab_size"],
                    "tir_var_upper_bound": {"batch_size": 32},
                }
            )
            cls = Expected
            storage: R.Any = R.memory.alloc_storage(
                R.shape([32 * vocab_size * 4 * 2 + 4194304]),
                R.prim_value(0),
                R.str("global"),
                R.dtype("uint8"),
            )
            lv1: R.Tensor(
                (2 * (batch_size * vocab_size * 4) + 4194304,),
                dtype="uint8",
            ) = R.memory.alloc_tensor(
                storage,
                R.prim_value(0),
                R.shape([2 * (batch_size * vocab_size * 4) + 4194304]),
                R.dtype("uint8"),
            )
            storage1: R.Any = R.memory.alloc_storage(
                R.shape([32 * vocab_size * 4]),
                R.prim_value(0),
                R.str("global"),
                R.dtype("float32"),
            )
            alloc1: R.Tensor((batch_size, vocab_size), dtype="float32") = R.memory.alloc_tensor(
                storage1, R.prim_value(0), R.shape([batch_size, vocab_size]), R.dtype("float32")
            )
            R.call_tir_packed(cls.cumsum, (probs, lv1, alloc1))
            cumsum: R.Tensor((batch_size, vocab_size), dtype="float32") = alloc1
            lv1_1: R.Tensor((batch_size, vocab_size), dtype="float32") = R.call_packed(
                "vm.builtin.reshape",
                cumsum,
                R.shape([batch_size, vocab_size]),
                ty_args=(R.Tensor((batch_size, vocab_size), dtype="float32"),),
            )
            return lv1_1

    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_view():
    @I.ir_module
    class Before:
        @Ts.function
        def tir_exp(var_rxplaceholder: T.handle, var_compute: T.handle):
            T.evaluate(0)

        @R.function(pure=False)
        def main():
            cls = Before
            x = R.builtin.alloc_tensor(R.shape([16, 16]), dtype="float32", runtime_device_index=0)
            x1 = R.memory.view(x, [128], "float32", 0)
            x2 = R.memory.ensure_zero_offset(x1)
            y = R.builtin.alloc_tensor(R.shape([128]), dtype="float32", runtime_device_index=0)
            R.call_tir_packed(cls.tir_exp, (x2, y))
            z = R.builtin.alloc_tensor(R.shape([128]), dtype="float32", runtime_device_index=0)
            R.call_tir_packed(cls.tir_exp, (y, z))
            return z

    @I.ir_module
    class Expected:
        @Ts.function
        def tir_exp(var_rxplaceholder: T.handle, var_compute: T.handle):
            T.evaluate(0)

        @R.function(pure=False)
        def main() -> R.Tensor((128,), dtype="float32"):
            cls = Expected
            storage: R.Any = R.memory.alloc_storage(
                R.shape([1024]), R.prim_value(0), R.str("global"), R.dtype("float32")
            )
            x: R.Tensor((16, 16), dtype="float32") = R.memory.alloc_tensor(
                storage, R.prim_value(0), R.shape([16, 16]), R.dtype("float32")
            )
            x1: R.Tensor((128,), dtype="float32") = R.memory.view(
                x, R.shape([128]), R.dtype("float32"), R.prim_value(0)
            )
            x2: R.Tensor((128,), dtype="float32") = R.memory.ensure_zero_offset(x1)
            storage1: R.Any = R.memory.alloc_storage(
                R.shape([512]), R.prim_value(0), R.str("global"), R.dtype("float32")
            )
            y: R.Tensor((128,), dtype="float32") = R.memory.alloc_tensor(
                storage1, R.prim_value(0), R.shape([128]), R.dtype("float32")
            )
            R.call_tir_packed(cls.tir_exp, (x2, y))
            z: R.Tensor((128,), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([128]), R.dtype("float32"), R.prim_value(0), R.str("global")
            )
            R.call_tir_packed(cls.tir_exp, (y, z))
            return z

    after = relax.transform.StaticPlanBlockMemory()(Before)
    tvm_tirx.ir.assert_structural_equal(after, Expected)


def test_match_cast_preserves_storage_liveness():
    @I.ir_module
    class Before:
        @Ts.function
        def copy(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
            T.evaluate(0)

        @Ts.function
        def add(
            A: T.Tensor((16,), "float32"),
            B: T.Tensor((16,), "float32"),
            C: T.Tensor((16,), "float32"),
        ):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((16,), "float32")) -> R.Tensor((16,), "float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Before
            alloc = R.builtin.alloc_tensor(R.shape([16]), "float32", 0)
            R.call_tir_packed(cls.copy, (x, alloc))
            checked = R.match_cast(alloc, R.Tensor((16,), "float32"))
            alloc1 = R.builtin.alloc_tensor(R.shape([16]), "float32", 0)
            R.call_tir_packed(cls.copy, (x, alloc1))
            alloc2 = R.builtin.alloc_tensor(R.shape([16]), "float32", 0)
            R.call_tir_packed(cls.add, (checked, alloc1, alloc2))
            return alloc2

    after = relax.transform.StaticPlanBlockMemory()(Before)
    alloc_storage_op = tvm_tirx.ir.Op.get("relax.memory.alloc_storage")
    storage_allocations = []

    def collect_storage_allocations(expr):
        if isinstance(expr, relax.Call) and expr.op.same_as(alloc_storage_op):
            storage_allocations.append(expr)

    relax.analysis.post_order_visit(after["main"], collect_storage_allocations)
    assert len(storage_allocations) == 2


def test_builtin_reshape_preserves_storage_liveness():
    @I.ir_module
    class Before:
        @Ts.function
        def copy(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((16,), "float32")) -> R.Tensor((16,), "float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Before
            alloc = R.builtin.alloc_tensor(R.shape([16]), "float32", 0)
            R.call_tir_packed(cls.copy, (x, alloc))
            reshaped = R.call_packed(
                "vm.builtin.reshape",
                alloc,
                R.shape([16]),
                ty_args=R.Tensor((16,), "float32"),
            )
            alloc1 = R.builtin.alloc_tensor(R.shape([16]), "float32", 0)
            R.call_tir_packed(cls.copy, (reshaped, alloc1))
            alloc2 = R.builtin.alloc_tensor(R.shape([16]), "float32", 0)
            R.call_tir_packed(cls.copy, (alloc1, alloc2))
            return alloc2

    after = relax.transform.StaticPlanBlockMemory()(Before)
    alloc_storage_op = tvm_tirx.ir.Op.get("relax.memory.alloc_storage")
    storage_allocations = []

    def collect_storage_allocations(expr):
        if isinstance(expr, relax.Call) and expr.op.same_as(alloc_storage_op):
            storage_allocations.append(expr)

    relax.analysis.post_order_visit(after["main"], collect_storage_allocations)
    assert len(storage_allocations) == 2


def test_with_dataflow():
    @I.ir_module
    class Before:
        @Ts.function
        def exp(A: T.handle, B: T.handle):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((10,), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
            cls = Before
            with R.dataflow():
                alloc: R.Tensor((10,), dtype="float32") = R.builtin.alloc_tensor(
                    R.shape([10]), R.dtype("float32"), runtime_device_index=0
                )
                _: R.Tuple() = R.call_pure_packed(
                    tvm_tirx.ir.Op.get("relax.call_tir_packed"), cls.exp, (x, alloc)
                )
                gv: R.Tensor((10,), dtype="float32") = alloc
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @Ts.function
        def exp(A: T.handle, B: T.handle):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((10,), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
            cls = Expected
            with R.dataflow():
                alloc: R.Tensor((10,), dtype="float32") = R.builtin.alloc_tensor(
                    R.shape([10]), R.dtype("float32"), R.prim_value(0), R.str("global")
                )
                R.call_pure_packed(tvm_tirx.ir.Op.get("relax.call_tir_packed"), cls.exp, (x, alloc))
                gv: R.Tensor((10,), dtype="float32") = alloc
                R.output(gv)
            return gv

    after = relax.transform.StaticPlanBlockMemory()(Before)
    tvm_tirx.ir.assert_structural_equal(after, Expected)


def _count_alloc_storage(func):
    alloc_storage_op = tvm_tirx.ir.Op.get("relax.memory.alloc_storage")
    count = 0

    def visit(expr):
        nonlocal count
        if isinstance(expr, relax.Call) and expr.op.same_as(alloc_storage_op):
            count += 1

    relax.analysis.post_order_visit(func, visit)
    return count


def test_if_branches_do_not_share_storage_var():
    @I.ir_module
    class Before:
        @Ts.function
        def exp(A: T.Tensor((2, 3), "float32"), B: T.Tensor((2, 3), "float32")):
            T.evaluate(0)

        @R.function
        def main(
            cond: R.Tensor((), dtype="bool"), x: R.Tensor((2, 3), dtype="float32")
        ) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Before
            if cond:
                alloc = R.builtin.alloc_tensor(R.shape([2, 3]), "float32", 0)
                R.call_tir_packed(cls.exp, (x, alloc))
                out = R.builtin.alloc_tensor(R.shape([2, 3]), "float32", 0)
                R.call_tir_packed(cls.exp, (alloc, out))
                z = out
            else:
                alloc1 = R.builtin.alloc_tensor(R.shape([2, 3]), "float32", 0)
                R.call_tir_packed(cls.exp, (x, alloc1))
                out1 = R.builtin.alloc_tensor(R.shape([2, 3]), "float32", 0)
                R.call_tir_packed(cls.exp, (alloc1, out1))
                z = out1
            return z

    after = relax.transform.StaticPlanBlockMemory()(Before)
    relax.analysis.check_well_formed(after)
    assert _count_alloc_storage(after["main"]) == 2


def test_if_branch_storage_not_reused_after_if():
    @I.ir_module
    class Before:
        @Ts.function
        def exp(A: T.Tensor((2, 3), "float32"), B: T.Tensor((2, 3), "float32")):
            T.evaluate(0)

        @R.function
        def main(
            cond: R.Tensor((), dtype="bool"), x: R.Tensor((2, 3), dtype="float32")
        ) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Before
            if cond:
                z = x
            else:
                alloc = R.builtin.alloc_tensor(R.shape([2, 3]), "float32", 0)
                R.call_tir_packed(cls.exp, (x, alloc))
                out = R.builtin.alloc_tensor(R.shape([2, 3]), "float32", 0)
                R.call_tir_packed(cls.exp, (alloc, out))
                z = out
            alloc1 = R.builtin.alloc_tensor(R.shape([2, 3]), "float32", 0)
            R.call_tir_packed(cls.exp, (z, alloc1))
            out1 = R.builtin.alloc_tensor(R.shape([2, 3]), "float32", 0)
            R.call_tir_packed(cls.exp, (alloc1, out1))
            return out1

    after = relax.transform.StaticPlanBlockMemory()(Before)
    relax.analysis.check_well_formed(after)
    assert _count_alloc_storage(after["main"]) == 2


def test_if_branches_share_storage_allocated_before_if():
    @I.ir_module
    class Before:
        @Ts.function
        def exp(A: T.Tensor((2, 3), "float32"), B: T.Tensor((2, 3), "float32")):
            T.evaluate(0)

        @R.function
        def main(
            cond: R.Tensor((), dtype="bool"), x: R.Tensor((2, 3), dtype="float32")
        ) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Before
            alloc = R.builtin.alloc_tensor(R.shape([2, 3]), "float32", 0)
            R.call_tir_packed(cls.exp, (x, alloc))
            out = R.builtin.alloc_tensor(R.shape([2, 3]), "float32", 0)
            R.call_tir_packed(cls.exp, (alloc, out))
            if cond:
                alloc1 = R.builtin.alloc_tensor(R.shape([2, 3]), "float32", 0)
                R.call_tir_packed(cls.exp, (out, alloc1))
                z = out
            else:
                alloc2 = R.builtin.alloc_tensor(R.shape([2, 3]), "float32", 0)
                R.call_tir_packed(cls.exp, (out, alloc2))
                z = out
            return z

    after = relax.transform.StaticPlanBlockMemory()(Before)
    relax.analysis.check_well_formed(after)
    assert _count_alloc_storage(after["main"]) == 1


if __name__ == "__main__":
    tvm_tirx.testing.main()
