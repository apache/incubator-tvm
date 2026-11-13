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
from tvm import relax as rx
from tvm.relax import script as R
from tvm.relax.analysis import detect_recursion
from tvm.s_tir import script as Ts
from tvm_tirx.script import tirx as T


def assert_groups(groups: list[list[rx.GlobalVar]], expected: list[list[str]]) -> None:
    assert len(groups) == len(expected)

    # disregard order, search only by name for convenience
    expected_sets = [set(expected_group) for expected_group in expected]
    actual_sets = [set(map(lambda gv: gv.name_hint, actual_group)) for actual_group in groups]

    for expected_set in expected_sets:
        assert expected_set in actual_sets


def test_no_recursion():
    @tvm_tirx.script.ir_module
    class NoRecursion:
        @R.function
        def a(x: R.Any) -> R.Any:
            return x

        @R.function
        def b(x: R.Any) -> R.Any:
            return x

    groups = detect_recursion(NoRecursion)
    assert len(groups) == 0


def test_simple_recursion():
    @tvm_tirx.script.ir_module
    class SimpleRecursion:
        @R.function
        def c(x: R.Any) -> R.Any:
            return SimpleRecursion.c(x)

    groups = detect_recursion(SimpleRecursion)
    assert_groups(groups, ["c"])


def test_tree():
    # no cycle!
    @tvm_tirx.script.ir_module
    class Tree:
        @R.function
        def a(x: R.Any) -> R.Any:
            return Tree.b(x)

        @R.function
        def b(x: R.Any) -> R.Any:
            return Tree.c(x)

        @R.function
        def c(x: R.Any) -> R.Any:
            z: R.Any = Tree.d(x)
            return Tree.e(z)

        @R.function
        def d(x: R.Any) -> R.Any:
            return Tree.e(x)

        @R.function
        def e(x: R.Any) -> R.Any:
            return x

    groups = detect_recursion(Tree)
    assert len(groups) == 0


def test_two_function_case():
    @tvm_tirx.script.ir_module
    class TwoFunctionCase:
        @R.function
        def a(x: R.Any) -> R.Any:
            return TwoFunctionCase.b(x)

        @R.function
        def b(x: R.Any) -> R.Any:
            return TwoFunctionCase.a(x)

        # not part of the group, shouldn't be reported
        @R.function
        def c(x: R.Any) -> R.Any:
            return x

    groups = detect_recursion(TwoFunctionCase)
    assert_groups(groups, [["a", "b"]])


def test_two_groups_of_two():
    @tvm_tirx.script.ir_module
    class TwoGroupsOfTwo:
        @R.function
        def a(x: R.Any) -> R.Any:
            return TwoGroupsOfTwo.b(x)

        @R.function
        def b(x: R.Any) -> R.Any:
            return TwoGroupsOfTwo.a(x)

        @R.function
        def c(x: R.Any) -> R.Any:
            return TwoGroupsOfTwo.d(x)

        @R.function
        def d(x: R.Any) -> R.Any:
            return TwoGroupsOfTwo.c(x)

        # not part of either group, shouldn't be reported
        @R.function
        def e(x: R.Any) -> R.Any:
            return x

    groups = detect_recursion(TwoGroupsOfTwo)
    assert_groups(groups, [["a", "b"], ["c", "d"]])


def test_mutual_recursion_and_simple_recursion():
    @tvm_tirx.script.ir_module
    class MutualAndSimple:
        @R.function
        def a(x: R.Any) -> R.Any:
            return MutualAndSimple.b(x)

        @R.function
        def b(x: R.Any) -> R.Any:
            return MutualAndSimple.a(x)

        # forms its own group
        @R.function
        def c(x: R.Any) -> R.Any:
            return MutualAndSimple.c(x)

    groups = detect_recursion(MutualAndSimple)
    assert_groups(groups, [["a", "b"], ["c"]])


def test_simultaneous_mutual_and_simple_recursion():
    # even though both call themselves and each other,
    # it should still form only one group
    @tvm_tirx.script.ir_module
    class SimultaneousMutualAndSimple:
        @R.function
        def a(x: R.Any) -> R.Any:
            cls = SimultaneousMutualAndSimple
            return cls.b(cls.a(x))

        @R.function
        def b(x: R.Any) -> R.Any:
            cls = SimultaneousMutualAndSimple
            return cls.a(cls.b(x))

    groups = detect_recursion(SimultaneousMutualAndSimple)
    assert_groups(groups, [["a", "b"]])


def test_three_function_case():
    @tvm_tirx.script.ir_module
    class ThreeFunctionCase:
        @R.function
        def a(x: R.Any) -> R.Any:
            return ThreeFunctionCase.b(x)

        @R.function
        def b(x: R.Any) -> R.Any:
            return ThreeFunctionCase.c(x)

        @R.function
        def c(x: R.Any) -> R.Any:
            return ThreeFunctionCase.a(x)

    groups = detect_recursion(ThreeFunctionCase)
    assert_groups(groups, [["a", "b", "c"]])


def test_call_from_outside_of_group():
    @tvm_tirx.script.ir_module
    class CallFromOutOfGroup:
        # A calls into a group of mutually recursive functions,
        # but is not part of the cycle
        @R.function
        def a(x: R.Any) -> R.Any:
            return CallFromOutOfGroup.d(x)

        @R.function
        def b(x: R.Any) -> R.Any:
            return CallFromOutOfGroup.c(x)

        @R.function
        def c(x: R.Any) -> R.Any:
            return CallFromOutOfGroup.d(x)

        @R.function
        def d(x: R.Any) -> R.Any:
            return CallFromOutOfGroup.b(x)

        # E also calls into the cycle but isn't part of it
        @R.function
        def e(x: R.Any) -> R.Any:
            return CallFromOutOfGroup.b(x)

    groups = detect_recursion(CallFromOutOfGroup)
    assert_groups(groups, [["b", "c", "d"]])


def test_call_from_group_to_outside():
    @tvm_tirx.script.ir_module
    class CallFromGroupToOutside:
        # A calls into a group of mutually recursive functions,
        # but is not part of the cycle
        @R.function
        def a(x: R.Any) -> R.Any:
            return CallFromGroupToOutside.b(x)

        @R.function
        def b(x: R.Any) -> R.Any:
            # d is called from a member of the group but it is not part of the cycle
            z: R.Any = CallFromGroupToOutside.d(x)
            return CallFromGroupToOutside.c(z)

        @R.function
        def c(x: R.Any) -> R.Any:
            return CallFromGroupToOutside.a(x)

        @R.function
        def d(x: R.Any) -> R.Any:
            return x

    groups = detect_recursion(CallFromGroupToOutside)
    assert_groups(groups, [["a", "b", "c"]])


def test_group_with_two_cycles():
    """
    a -> b <- f
    ^    |    ^
    |    v    |
    d <- c -> e

    There are two smaller cycles in this group,
    but you can have one big cycle
    B -> C -> D -> A -> B -> C -> E -> F -> B
    """

    @tvm_tirx.script.ir_module
    class GroupWithTwoCycles:
        @R.function
        def a(x: R.Any) -> R.Any:
            return GroupWithTwoCycles.b(x)

        @R.function
        def b(x: R.Any) -> R.Any:
            return GroupWithTwoCycles.c(x)

        @R.function
        def c(x: R.Any) -> R.Any:
            y = GroupWithTwoCycles.d(x)
            return GroupWithTwoCycles.e(y)

        @R.function
        def d(x: R.Any) -> R.Any:
            return GroupWithTwoCycles.a(x)

        @R.function
        def e(x: R.Any) -> R.Any:
            return GroupWithTwoCycles.f(x)

        @R.function
        def f(x: R.Any) -> R.Any:
            return GroupWithTwoCycles.b(x)

    groups = detect_recursion(GroupWithTwoCycles)
    assert_groups(groups, [["a", "b", "c", "d", "e", "f"]])


def test_multicycle_example():
    """
    Example from the documentation
    A <-> B <-> C
    ^     |     ^
    |     v     |
    |     D     |
    |     |     |
    v     v     v
    E <-> F <-> G
    """

    @tvm_tirx.script.ir_module
    class MulticycleExample:
        @R.function
        def a(x: R.Any) -> R.Any:
            cls = MulticycleExample
            y = cls.b(x)
            return cls.e(y)

        @R.function
        def b(x: R.Any) -> R.Any:
            cls = MulticycleExample
            y = cls.a(x)
            z = cls.c(y)
            return cls.d(z)

        @R.function
        def c(x: R.Any) -> R.Any:
            cls = MulticycleExample
            y = cls.g(x)
            return cls.b(y)

        @R.function
        def d(x: R.Any) -> R.Any:
            cls = MulticycleExample
            return cls.f(x)

        @R.function
        def e(x: R.Any) -> R.Any:
            cls = MulticycleExample
            y = cls.f(x)
            return cls.a(y)

        @R.function
        def f(x: R.Any) -> R.Any:
            cls = MulticycleExample
            y = cls.g(x)
            return cls.e(y)

        @R.function
        def g(x: R.Any) -> R.Any:
            cls = MulticycleExample
            y = cls.f(x)
            return cls.c(y)

    groups = detect_recursion(MulticycleExample)
    assert_groups(groups, [["a", "b", "c", "d", "e", "f", "g"]])


def test_control_flow():
    @tvm_tirx.script.ir_module
    class ControlFlowExample:
        @R.function
        def a(x: R.Any) -> R.Any:
            cls = ControlFlowExample
            y: R.Tensor((), dtype="bool") = R.const(True, dtype="bool")
            if y:
                ret = cls.b(x)
            else:
                ret = cls.c(x)
            return ret

        @R.function
        def b(x: R.Any) -> R.Any:
            cls = ControlFlowExample
            return cls.a(x)

        @R.function
        def c(x: R.Any) -> R.Any:
            cls = ControlFlowExample
            return cls.a(x)

    groups = detect_recursion(ControlFlowExample)
    assert_groups(groups, [["a", "b", "c"]])


def test_returning_self():
    @tvm_tirx.script.ir_module
    class ReturnsSelf:
        @R.function
        def a() -> R.Any:
            # this is also a form of recursion
            return ReturnsSelf.a

    groups = detect_recursion(ReturnsSelf)
    assert_groups(groups, [["a"]])


def test_mutual_recursion_via_references():
    @tvm_tirx.script.ir_module
    class GatherReferences:
        @R.function
        def a(x: R.Any) -> R.Any:
            cls = GatherReferences
            return cls.b(x)

        @R.function
        def b(x: R.Any) -> R.Any:
            cls = GatherReferences
            return (cls.a, cls.b, cls.c)

        @R.function
        def c(x: R.Any) -> R.Any:
            cls = GatherReferences
            return cls.a(x)

    groups = detect_recursion(GatherReferences)
    assert_groups(groups, [["a", "b", "c"]])


def test_disregard_functions():
    @tvm_tirx.script.ir_module
    class CallFunction:
        # copied from test_analysis.py
        @Ts.function
        def identity_identity(A: T.Tensor((4, 4), "float32"), B: T.Tensor((4, 4), "float32")):
            C = Ts.sblock_alloc_tensor((128, 128), "float32")
            for i0, i1 in T.grid(4, 4):
                with Ts.sblock("identity"):
                    vi0, vi1 = Ts.axis.remap("SS", [i0, i1])
                    C[vi0, vi1] = A[vi0, vi1]
            for i0, i1 in T.grid(4, 4):
                with Ts.sblock("identity"):
                    vi0, vi1 = Ts.axis.remap("SS", [i0, i1])
                    B[vi0, vi1] = C[vi0, vi1]

        @R.function
        def a(x: R.Tensor((4, 4), "float32")) -> R.Any:
            cls = CallFunction
            y = R.call_tir(cls.identity_identity, x, ty_args=[R.Tensor((4, 4), "float32")])
            return cls.b(y)

        @R.function
        def b(x: R.Tensor((4, 4), "float32")) -> R.Any:
            cls = CallFunction
            y = R.call_tir(cls.identity_identity, x, ty_args=[R.Tensor((4, 4), "float32")])
            return cls.a(y)

    groups = detect_recursion(CallFunction)
    # the function should not be listed here
    assert_groups(groups, [["a", "b"]])


if __name__ == "__main__":
    tvm_tirx.testing.main()
