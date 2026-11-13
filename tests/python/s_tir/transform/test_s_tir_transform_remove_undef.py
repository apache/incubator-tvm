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

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import script as Ts
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def test_remove_store_undef():
    """Remove a store whose value is T.undef()"""

    @I.ir_module
    class Before:
        @Ts.function
        def main(A: T.Tensor(1, "int32")):
            A[0] = T.undef()

    @I.ir_module
    class Expected:
        @Ts.function
        def main(A: T.Tensor(1, "int32")):
            T.evaluate(0)

    After = tvm.s_tir.transform.RemoveStoreUndef()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_remove_store_undef_expression():
    """Expressions containing T.undef() are removed"""

    @I.ir_module
    class Before:
        @Ts.function
        def main(A: T.Tensor(1, "int32")):
            A[0] = 1 + T.undef()

    @I.ir_module
    class Expected:
        @Ts.function
        def main(A: T.Tensor(1, "int32")):
            T.evaluate(0)

    After = tvm.s_tir.transform.RemoveStoreUndef()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_keep_other_call_nodes():
    """Expressions containing other CallNodes are not removed"""

    @I.ir_module
    class Before:
        @Ts.function
        def main(A: T.Tensor(1, "int32"), n: T.int32):
            A[0] = T.shift_left(n, 1)

    Expected = Before

    After = tvm.s_tir.transform.RemoveStoreUndef()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_remove_let_undef():
    """Remove a store whose value is bound to T.undef()"""

    @I.ir_module
    class Before:
        @Ts.function
        def main(A: T.Tensor(1, "int32")):
            val: T.let[T.int32] = T.undef()
            A[0] = val

    @I.ir_module
    class Expected:
        @Ts.function
        def main(A: T.Tensor(1, "int32")):
            T.evaluate(0)

    After = tvm.s_tir.transform.RemoveStoreUndef()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_raise_error_for_undef_as_store_indices():
    """Use of T.undef() as tensor indices is an error"""

    @I.ir_module
    class Before:
        @Ts.function
        def main(A: T.Tensor(1, "int32")):
            val: T.let[T.int32] = T.undef()
            A[val] = 5

    with pytest.raises(RuntimeError):
        tvm.s_tir.transform.RemoveStoreUndef()(Before)


def test_raise_error_for_undef_as_load_indices():
    """Use of T.undef() as tensor indices is an error

    Even though this occurs as part of the TensorStore's value, the
    T.undef() may not appear in a tensor's indices.
    """

    @I.ir_module
    class Before:
        @Ts.function
        def main(A: T.Tensor(1, "int32"), B: T.Tensor(1, "int32")):
            B[0] = A[T.undef()]

    with pytest.raises(RuntimeError):
        tvm.s_tir.transform.RemoveStoreUndef()(Before)


if __name__ == "__main__":
    tvm_tirx.testing.main()
