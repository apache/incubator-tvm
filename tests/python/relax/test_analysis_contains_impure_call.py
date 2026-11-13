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
# ruff: noqa: F841

import tvm_tirx
import tvm_tirx.testing
from tvm import relax as rx
from tvm.relax import script as R
from tvm.relax.analysis import contains_impure_call


def test_simple_pure_case():
    @tvm_tirx.script.ir_module
    class PureTest:
        @R.function
        def pure_func(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            y = R.add(x, x)
            z = R.multiply(x, y)
            return R.add(z, R.const(1, "int32"))

    assert not contains_impure_call(PureTest["pure_func"])


def test_simple_impure_case():
    @tvm_tirx.script.ir_module
    class ImpureTest:
        @R.function(pure=False)
        def impure_func() -> R.Any:
            y = R.print("I am a message")
            return y

    assert contains_impure_call(ImpureTest["impure_func"])


def test_nested_function():
    @tvm_tirx.script.ir_module
    class NestedTest:
        @R.function
        def pure_with_impure_nested() -> R.Tensor((), "int32"):
            # unused
            @R.function(pure=False)
            def impure_inner() -> R.Any:
                y = R.print("Another, worse, message")
                return y

            x = R.const(0, dtype="int32")
            return R.add(x, x)

    assert not contains_impure_call(NestedTest["pure_with_impure_nested"])
    assert contains_impure_call(
        NestedTest["pure_with_impure_nested"].body.blocks[0].bindings[0].value
    )


def test_ignoring_recursive_call():
    # Ignoring a recursive call. This can be useful if some transformation
    # removes an impure operation and the compiler needs to check if the impure
    # function has become pure
    @tvm_tirx.script.ir_module
    class RecursiveTest:
        @R.function(pure=False)
        def recursive_impure() -> R.Any:
            x = R.const(1, "int32")
            y = R.add(x, x)
            z = R.print("{} {}", x, y)
            w = RecursiveTest.recursive_impure()
            return w

    assert contains_impure_call(RecursiveTest["recursive_impure"])
    # but if we remove the impure call...
    body = RecursiveTest["recursive_impure"].body
    own_name = body.blocks[0].bindings[-1].value.op
    # skipping the call to print...
    new_bindings = [
        body.blocks[0].bindings[0],
        body.blocks[0].bindings[1],
        body.blocks[0].bindings[-1],
    ]
    # Note: we construct the function in this way so that we keep the old vars
    # with their current Type. That would get fixed during normalization.
    # However, this situation is meant to correspond to an intermediate state
    # that might arise within a pass.
    new_body = rx.SeqExpr([rx.BindingBlock(new_bindings)], body.body)

    # if we didn't ignore the recursive call, the fact the var's Type
    # calls it impure would throw it off
    assert not contains_impure_call(new_body, own_name=own_name)
    assert contains_impure_call(new_body)


if __name__ == "__main__":
    tvm_tirx.testing.main()
