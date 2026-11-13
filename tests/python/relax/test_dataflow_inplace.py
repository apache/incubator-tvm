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

import numpy as np
import pytest
import torch

import tvm
import tvm_tirx
from tvm import relax
from tvm.relax.testing.transform import (
    dataflow_alias_analysis,
    dataflow_inplace_analysis,
    dataflow_liveness_analysis,
    dataflow_single_inplace_call,
)
from tvm.relax.transform import DataflowUseInplaceCalls
from tvm.s_tir import script as Ts
from tvm_tirx import testing
from tvm_tirx.ir import TupleType as _TupleType
from tvm_tirx.script.parser import ir as I
from tvm_tirx.script.parser import relax as R
from tvm_tirx.script.parser import tirx as T


def test_liveness_analysis():
    @I.ir_module
    class BasicLiveness:
        @R.function
        def main(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            with R.dataflow():
                y = R.const(1, dtype="int32")
                z = R.add(x, y)
                q = R.multiply(z, y)
                p = R.add(z, q)
                n = R.multiply(p, p)
                R.output(n, p)
            return n

    block = BasicLiveness["main"].body.blocks[0]
    live_ranges = dataflow_liveness_analysis(block)
    expected_ranges = {
        # x is live past the binding block
        "x": (-1, 5),
        "y": (0, 2),
        "z": (1, 3),
        "q": (2, 3),
        # exposed though ultimately not used
        "p": (3, 5),
        "n": (4, 5),
    }
    actual_ranges = {var.name: live_range for var, live_range in live_ranges.items()}
    assert actual_ranges == expected_ranges


def test_alias_analysis_basic():
    @I.ir_module
    class BasicAliasAnalysis:
        @R.function
        def main(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            with R.dataflow():
                y = x  # y is an alias of x
                z = R.add(y, y)  # fresh value
                n = z  # alias of z
                R.output(n)
            return n

    block = BasicAliasAnalysis["main"].body.blocks[0]
    alias_sets, tuple_map = dataflow_alias_analysis(block, BasicAliasAnalysis["main"].params)
    expected = {
        "x": {0},
        "y": {0},
        "z": {1},
        "n": {1},
    }

    for var, alias_set in alias_sets.items():
        assert alias_set == expected[var.name]
    assert tuple_map == {}


def test_alias_analysis_tuple():
    @I.ir_module
    class AliasesWithTuples:
        @R.function
        def main(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            with R.dataflow():
                y = R.const(1, dtype="int32")
                t = (x, y)
                a = t[0]
                b = t[1]
                c = t[0]
                d = t[1]
                u = t
                e = t[0]
                f = t[1]
                z = R.add(c, d)
                n = z
                R.output(n)
            return n

    block = AliasesWithTuples["main"].body.blocks[0]
    alias_sets, tuple_map = dataflow_alias_analysis(block, AliasesWithTuples["main"].params)
    expected = {
        "x": {0},
        "y": {1},
        "t": {2},
        "a": {0},
        "b": {1},
        "c": {0},
        "d": {1},
        "u": {2},
        "e": {0},
        "f": {1},
        "z": {3},
        "n": {3},
    }

    actual_alias_sets = {var.name: alias_set for var, alias_set in alias_sets.items()}
    assert expected == actual_alias_sets
    assert 2 in tuple_map
    assert tuple_map[2] == [{0}, {1}]


def test_alias_split():
    @I.ir_module
    class AliasSplit:
        @R.function
        def main(x: R.Tensor((60,), "int32")) -> R.Tensor((15,), "int32"):
            with R.dataflow():
                t = R.split(x, 4)
                y = t[0]
                z = t[1]
                q = t[2]
                p = t[3]
                n = z
                R.output(n)
            return n

    block = AliasSplit["main"].body.blocks[0]
    alias_sets, tuple_map = dataflow_alias_analysis(block, AliasSplit["main"].params)
    expected = {
        "x": {0},
        "t": {1},
        "y": {2},
        "z": {3},
        "q": {4},
        "p": {5},
        "n": {3},
    }

    actual_alias_sets = {var.name: alias_set for var, alias_set in alias_sets.items()}
    assert expected == actual_alias_sets
    assert len(tuple_map) == 1
    assert 1 in tuple_map
    assert tuple_map[1] == [{2}, {3}, {4}, {5}]


def test_alias_call_tir():
    # call TIR can yield either a single tensor or a tuple
    m_tir_id = T.dynamic("m", "int32")
    n_tir_id = T.dynamic("n", "int32")
    m_tir_id2 = T.dynamic("m", "int32")
    n_tir_id2 = T.dynamic("n", "int32")

    @I.ir_module
    class AliasCallTir:
        @Ts.function
        def tir_id(
            A: T.Tensor((m_tir_id, n_tir_id), "int32"), B: T.Tensor((m_tir_id, n_tir_id), "int32")
        ) -> None:
            T.func_attr({"global_symbol": "tir_id"})

            for i, j in T.grid(m_tir_id, n_tir_id):
                with Ts.sblock("id"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj]

        @Ts.function
        def tir_id2(
            A: T.Tensor((m_tir_id2, n_tir_id2), "int32"),
            B: T.Tensor((m_tir_id2, n_tir_id2), "int32"),
            C: T.Tensor((m_tir_id2, n_tir_id2), "int32"),
        ) -> None:
            T.func_attr({"global_symbol": "tir_id"})

            for i, j in T.grid(m_tir_id2, n_tir_id2):
                with Ts.sblock("id"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj]
                    C[vi, vj] = A[vi, vj]

        @R.function
        def main(x: R.Tensor((10, 10), "int32")) -> R.Tensor((10, 10), "int32"):
            with R.dataflow():
                cls = AliasCallTir
                y = R.call_tir(cls.tir_id, (x,), ty_args=[R.Tensor((10, 10), "int32")])
                t = R.call_tir(
                    cls.tir_id2,
                    (y,),
                    ty_args=[
                        _TupleType([R.Tensor((10, 10), "int32"), R.Tensor((10, 10), "int32")])
                    ],
                )
                z = y
                p = t[0]
                q = t[1]
                u = t
                m = u[0]
                n = u[1]
                v = n
                R.output(v)
            return v

    block = AliasCallTir["main"].body.blocks[0]
    alias_sets, tuple_map = dataflow_alias_analysis(block, AliasCallTir["main"].params)
    expected = {
        "x": {0},
        "y": {1},
        "t": {2},
        "z": {1},
        "p": {3},
        "q": {4},
        "u": {2},
        "m": {3},
        "n": {4},
        "v": {4},
    }

    actual_alias_sets = {var.name: alias_set for var, alias_set in alias_sets.items()}
    assert expected == actual_alias_sets
    assert len(tuple_map) == 1
    assert 2 in tuple_map
    assert tuple_map[2] == [{3}, {4}]


def test_mystery_calls():
    @I.ir_module
    class AliasChaosCalls:
        @R.function
        def identity(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            return x

        @R.function
        def main(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            with R.dataflow():
                cls = AliasChaosCalls
                y = cls.identity(x)
                z = cls.identity(y)
                m = R.const(1, dtype="int32")
                n = R.const(2, dtype="int32")
                t = (m, n)
                a = R.call_pure_packed(
                    "chaos", t, ty_args=R.Tuple(R.Tensor((), "int32"), R.Tensor((), "int32"))
                )
                b = a[0]
                c = a[1]
                R.output(c)
            return c

    block = AliasChaosCalls["main"].body.blocks[0]
    alias_sets, tuple_map = dataflow_alias_analysis(block, AliasChaosCalls["main"].params)
    expected = {
        "x": {0},
        "y": {0, 1},
        "z": {0, 1, 2},
        "m": {3},
        "n": {4},
        "t": {5},
        "a": {3, 4, 5, 6, 7, 8},  # either t or a fresh tuple
        "b": {3, 4, 5, 6, 7, 8},  # the tuple components can be aliased to any member...
        "c": {3, 4, 5, 6, 7, 8},  # the tuple components can be aliased to any member...
        # (in principle, we can use type information to narrow down the aliasing)
    }

    actual_alias_sets = {var.name: alias_set for var, alias_set in alias_sets.items()}
    assert expected == actual_alias_sets
    assert len(tuple_map) == 2
    assert 5 in tuple_map
    assert tuple_map[5] == [{3}, {4}]
    assert 6 in tuple_map
    assert tuple_map[6] == [{3, 4, 5, 6, 7, 8}, {3, 4, 5, 6, 7, 8}]


def test_alias_external_value():
    @I.ir_module
    class AliasExternalValue:
        @R.function
        def main(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            y = R.const(1, dtype="int32")  # not in DF block, treated as external
            t1 = (y, y)  # not in DF block, treated as external
            with R.dataflow():
                z = y  # mystery value
                a = R.const(2, dtype="int32")
                t2 = (z, a)
                b = t2[0]
                c = t1[1]  # tuple index into external value
                R.output(b)
            return b

    block = AliasExternalValue["main"].body.blocks[1]
    alias_sets, tuple_map = dataflow_alias_analysis(block, AliasExternalValue["main"].params)
    expected = {
        "x": {0},
        "z": {-1},
        "a": {1},
        "t2": {2},
        "b": {-1},
        "c": {-1},
    }

    actual_alias_sets = {var.name: alias_set for var, alias_set in alias_sets.items()}
    assert expected == actual_alias_sets
    assert len(tuple_map) == 1
    assert 2 in tuple_map
    assert tuple_map[2] == [{-1}, {1}]


def test_inplace_simple_case():
    @I.ir_module
    class InplaceBasic:
        @R.function
        def main(x: R.Tensor((2, 3), "int32"), y: R.Tensor((2, 3), "int32")) -> R.Tensor(
            (2, 3), "int32"
        ):
            with R.dataflow():
                z = R.add(x, y)  # cannot be done inplace: x and y are live later
                p = R.add(z, z)  # can be done inplace: z is not used later
                r = p  # alias of p
                m = R.multiply(p, p)  # p is not used later but r is, so can't do inplace
                n = R.add(m, r)  # can be done inplace: r is not used again
                ret = R.subtract(n, m)  # can be done inplace: neither is used again
                R.output(ret)
            return ret

    block = InplaceBasic["main"].body.blocks[0]
    size_match, exact_match = dataflow_inplace_analysis(
        block, InplaceBasic["main"].params, InplaceBasic
    )

    # order does not matter for the listing of candidates, so we have to implement as sets
    def assert_candidate_list(
        actual: list[tuple[int, set[int]]], expected: list[tuple[int, set[int]]]
    ) -> None:
        assert len(actual) == len(expected)
        for i in range(len(actual)):
            assert actual[i][0] == expected[i][0]
            assert len(expected[i][1]) == len(actual[i][1])
            for idx in actual[i][1]:
                assert idx in expected[i][1]

    assert_candidate_list(size_match, [(1, {0, 1}), (4, {1}), (5, {0, 1})])
    # TODO(@slyubomirsky): I couldn't think of an easy example where sizes don't match,
    # but broadcasting might cause it to happen
    assert_candidate_list(exact_match, [(1, {0, 1}), (4, {1}), (5, {0, 1})])


def test_inplace_single_call():
    @I.ir_module
    class TestModule:
        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")
        ) -> R.Tensor((2, 3), dtype="float32"):
            z = R.add(x, y)
            q = R.nn.silu(z)
            return q

    add_call = TestModule["main"].body.blocks[0].bindings[0].value
    new_add, new_mod = dataflow_single_inplace_call(TestModule, add_call, [0])

    @Ts.function(private=True)
    def expected_add(
        A: T.Tensor((T.int64(2), T.int64(3)), "float32"),
        B: T.Tensor((T.int64(2), T.int64(3)), "float32"),
    ):
        T.func_attr({"tirx.noalias": True})
        for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
            with Ts.sblock("T_add"):
                v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                Ts.reads(A[v_ax0, v_ax1], B[v_ax0, v_ax1])
                Ts.writes(A[v_ax0, v_ax1])
                A[v_ax0, v_ax1] = A[v_ax0, v_ax1] + B[v_ax0, v_ax1]

    tvm_tirx.ir.assert_structural_equal(new_mod["add_inplace"], expected_add)
    assert new_add.op.name == "relax.call_tir_inplace"
    assert new_add.args[0].name_hint == "add_inplace"
    for i, arg in enumerate(new_add.args[1].fields):
        arg == add_call.args[i]
    new_add.attrs.inplace_indices == [0]

    @Ts.function(private=True)
    def expected_silu(A: T.Tensor((T.int64(2), T.int64(3)), "float32")):
        T.func_attr({"tirx.noalias": True})
        compute = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3)))
        for i0, i1 in T.grid(T.int64(2), T.int64(3)):
            with Ts.sblock("compute"):
                v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                Ts.reads(A[v_i0, v_i1])
                Ts.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.sigmoid(A[v_i0, v_i1])
        for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
            with Ts.sblock("T_multiply"):
                v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                Ts.reads(A[v_ax0, v_ax1], compute[v_ax0, v_ax1])
                Ts.writes(A[v_ax0, v_ax1])
                A[v_ax0, v_ax1] = A[v_ax0, v_ax1] * compute[v_ax0, v_ax1]

    silu_call = TestModule["main"].body.blocks[0].bindings[1].value
    new_silu, new_mod = dataflow_single_inplace_call(TestModule, silu_call, [0])

    tvm_tirx.ir.assert_structural_equal(new_mod["silu_inplace"], expected_silu)
    assert new_silu.op.name == "relax.call_tir_inplace"
    assert new_silu.args[0].name_hint == "silu_inplace"
    for i, arg in enumerate(new_silu.args[1].fields):
        arg == silu_call.args[i]
    new_silu.attrs.inplace_indices == [0]


def test_insert_inplace_calls():
    @I.ir_module
    class EndToEndTest:
        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((1, 3), dtype="float32")
        ) -> R.Tensor((2, 3), dtype="float32"):
            with R.dataflow():
                z = R.add(x, y)  # broadcast happens here
                # Cannot be done in-place because x is an argument.
                a = R.add(z, y)  # this one can be done in-place
                q = R.multiply(a, y)  # broadcast again, a is eligible
                r = R.subtract(y, y)  # cannot be done in-place because y is an argument
                s = R.subtract(r, r)  # No broadcast. Can be done in-place
                m = R.multiply(q, s)  # should give us all zeros
                R.output(m)
            return m

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def add_inplace(
            A: T.Tensor((T.int64(2), T.int64(3)), "float32"),
            B: T.Tensor((T.int64(1), T.int64(3)), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_add"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(A[v_ax0, v_ax1], B[T.int64(0), v_ax1])
                    Ts.writes(A[v_ax0, v_ax1])
                    A[v_ax0, v_ax1] = A[v_ax0, v_ax1] + B[T.int64(0), v_ax1]

        @Ts.function(private=True)
        def multiply_inplace(
            A: T.Tensor((T.int64(2), T.int64(3)), "float32"),
            B: T.Tensor((T.int64(1), T.int64(3)), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_multiply"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(A[v_ax0, v_ax1], B[T.int64(0), v_ax1])
                    Ts.writes(A[v_ax0, v_ax1])
                    A[v_ax0, v_ax1] = A[v_ax0, v_ax1] * B[T.int64(0), v_ax1]

        @Ts.function(private=True)
        def subtract_inplace(
            A: T.Tensor((T.int64(1), T.int64(3)), "float32"),
            B: T.Tensor((T.int64(1), T.int64(3)), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for ax0, ax1 in T.grid(T.int64(1), T.int64(3)):
                with Ts.sblock("T_subtract"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(A[v_ax0, v_ax1], B[v_ax0, v_ax1])
                    Ts.writes(B[v_ax0, v_ax1])
                    B[v_ax0, v_ax1] = A[v_ax0, v_ax1] - B[v_ax0, v_ax1]

        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((1, 3), dtype="float32")
        ) -> R.Tensor((2, 3), dtype="float32"):
            cls = Expected
            with R.dataflow():
                z: R.Tensor((2, 3), dtype="float32") = R.add(x, y)
                a: R.Tensor((2, 3), dtype="float32") = R.call_tir_inplace(
                    cls.add_inplace,
                    (z, y),
                    inplace_indices=[0],
                    ty_args=[R.Tensor((2, 3), dtype="float32")],
                )
                q: R.Tensor((2, 3), dtype="float32") = R.call_tir_inplace(
                    cls.multiply_inplace,
                    (a, y),
                    inplace_indices=[0],
                    ty_args=[R.Tensor((2, 3), dtype="float32")],
                )
                r: R.Tensor((1, 3), dtype="float32") = R.subtract(y, y)
                s: R.Tensor((1, 3), dtype="float32") = R.call_tir_inplace(
                    cls.subtract_inplace,
                    (r, r),
                    inplace_indices=[1],
                    ty_args=[R.Tensor((1, 3), dtype="float32")],
                )
                m: R.Tensor((2, 3), dtype="float32") = R.call_tir_inplace(
                    cls.multiply_inplace,
                    (q, s),
                    inplace_indices=[0],
                    ty_args=[R.Tensor((2, 3), dtype="float32")],
                )
                R.output(m)
            return m

    transform_pass = DataflowUseInplaceCalls()
    new_mod = transform_pass(EndToEndTest)
    tvm_tirx.ir.assert_structural_equal(new_mod, Expected)

    x = tvm_tirx.runtime.tensor(np.random.rand(2, 3).astype("float32"))
    y = tvm_tirx.runtime.tensor(np.random.rand(1, 3).astype("float32"))
    expected = np.zeros((2, 3), dtype="float32")

    target = tvm_tirx.target.Target("llvm")
    ex = tvm.compile(new_mod, target)
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    res = vm["main"](x, y)
    assert (expected == res.numpy()).all()


def test_dynamic():
    a_dim = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")

    @I.ir_module
    class DynamicTestCase:
        @R.function
        def main(
            x: R.Tensor((a_dim, b), dtype="float32"), y: R.Tensor((a_dim, b), dtype="float32")
        ) -> R.Tensor((a_dim, b), dtype="float32"):
            with R.dataflow():
                z = R.add(x, y)
                # Cannot be done in-place because x and y are arguments
                a = R.add(z, y)  # this one can be done in-place
                s = R.subtract(a, a)  # No broadcast. Can be done in-place
                R.output(s)
            return s

    # the result should be all zeroes
    transform_pass = DataflowUseInplaceCalls()
    new_mod = transform_pass(DynamicTestCase)

    a_add_inplace = T.dynamic("a", "int64")
    b_add_inplace = T.dynamic("b", "int64")
    a_subtract_inplace = T.dynamic("a", "int64")
    b_subtract_inplace = T.dynamic("b", "int64")
    a_main = T.dynamic("a", "int64")
    b_main = T.dynamic("b", "int64")

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def add_inplace(
            A: T.Tensor((a_add_inplace, b_add_inplace), "float32"),
            B: T.Tensor((a_add_inplace, b_add_inplace), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})

            for ax0, ax1 in T.grid(a_add_inplace, b_add_inplace):
                with Ts.sblock("T_add"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(A[v_ax0, v_ax1], B[v_ax0, v_ax1])
                    Ts.writes(A[v_ax0, v_ax1])
                    A[v_ax0, v_ax1] = A[v_ax0, v_ax1] + B[v_ax0, v_ax1]

        @Ts.function(private=True)
        def subtract_inplace(
            A: T.Tensor((a_subtract_inplace, b_subtract_inplace), "float32"),
            B: T.Tensor((a_subtract_inplace, b_subtract_inplace), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})

            for ax0, ax1 in T.grid(a_subtract_inplace, b_subtract_inplace):
                with Ts.sblock("T_subtract"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(A[v_ax0, v_ax1], B[v_ax0, v_ax1])
                    Ts.writes(B[v_ax0, v_ax1])
                    B[v_ax0, v_ax1] = A[v_ax0, v_ax1] - B[v_ax0, v_ax1]

        @R.function
        def main(
            x: R.Tensor((a_main, b_main), dtype="float32"),
            y: R.Tensor((a_main, b_main), dtype="float32"),
        ) -> R.Tensor((a_main, b_main), dtype="float32"):
            cls = Expected
            with R.dataflow():
                z = R.add(x, y)
                a_1 = R.call_tir_inplace(
                    cls.add_inplace,
                    (z, y),
                    ty_args=[R.Tensor((a_main, b_main), dtype="float32")],
                    inplace_indices=[0],
                )
                s = R.call_tir_inplace(
                    cls.subtract_inplace,
                    (a_1, a_1),
                    ty_args=[R.Tensor((a_main, b_main), dtype="float32")],
                    inplace_indices=[1],
                )
                R.output(s)
            return s

    tvm_tirx.ir.assert_structural_equal(new_mod, Expected, map_free_vars=True)
    x = tvm_tirx.runtime.tensor(np.random.rand(2, 3).astype("float32"))
    y = tvm_tirx.runtime.tensor(np.random.rand(2, 3).astype("float32"))
    expected = np.zeros((2, 3), dtype="float32")

    target = tvm_tirx.target.Target("llvm")
    ex = tvm.compile(new_mod, target)
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    res = vm["main"](x, y)
    assert (expected == res.numpy()).all()


def test_dynamic_mismatch():
    # cannot statically prove the shapes to be equal so the module should be unchanged
    a_dim = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")
    c = T.dynamic("c", "int64")
    d = T.dynamic("d", "int64")

    @I.ir_module
    class DynamicMistmatchTestCase:
        @R.function
        def main(x: R.Tensor((a_dim, b), dtype="float32"), y: R.Tensor((c, d), dtype="float32")):
            with R.dataflow():
                z = R.add(x, y)
                # Cannot be done in-place because x and y are arguments
                a = R.add(z, y)  # cannot conclude that shapes match
                R.output(a)
            return a

    transform_pass = DataflowUseInplaceCalls()
    new_mod = transform_pass(DynamicMistmatchTestCase)
    tvm_tirx.ir.assert_structural_equal(new_mod, DynamicMistmatchTestCase)


class TestViewOpSharedStorageAndNoInplace:
    storage_ptr_x_1d = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    storage_ptr_x_2d = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    storage_ptr_x_squeeze = np.array([[[1.0], [2.0], [3.0], [4.0]]], dtype=np.float32)
    storage_ptr_x_ensure_zero_offset = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)

    @staticmethod
    def _emit_duplicate_view(op, x):
        if op == "relax.expand_dims":
            a = relax.op.expand_dims(x, axis=1)
            b = relax.op.expand_dims(x, axis=1)
        elif op == "relax.squeeze":
            a = relax.op.squeeze(x, axis=[0])
            b = relax.op.squeeze(x, axis=[0])
        elif op == "relax.reshape":
            a = relax.op.reshape(x, (4, 1))
            b = relax.op.reshape(x, (4, 1))
        elif op == "relax.permute_dims":
            a = relax.op.permute_dims(x, axes=[1, 0])
            b = relax.op.permute_dims(x, axes=[1, 0])
        elif op == "relax.memory.view":
            a = relax.op.memory.view(x, (4, 1))
            b = relax.op.memory.view(x, (4, 1))
        elif op == "relax.memory.ensure_zero_offset":
            a = relax.op.memory.ensure_zero_offset(x)
            b = relax.op.memory.ensure_zero_offset(x)
        elif op == "relax.flatten":
            a = relax.op.flatten(x)
            b = relax.op.flatten(x)
        elif op == "relax.nn.batch_flatten":
            a = relax.op.nn.batch_flatten(x)
            b = relax.op.nn.batch_flatten(x)
        else:
            raise ValueError(op)
        return a, b

    @staticmethod
    def _concat_axis_for_view_op(op):
        if op == "relax.flatten":
            return 0
        return 1

    @classmethod
    def _build_module(cls, op):
        if op == "relax.expand_dims":
            x_ty = relax.TensorType((4,), "float32")
        elif op == "relax.squeeze":
            x_ty = relax.TensorType((1, 4, 1), "float32")
        elif op == "relax.reshape":
            x_ty = relax.TensorType((4,), "float32")
        elif op == "relax.permute_dims":
            x_ty = relax.TensorType((1, 4), "float32")
        elif op == "relax.memory.view":
            x_ty = relax.TensorType((4,), "float32")
        elif op == "relax.memory.ensure_zero_offset":
            x_ty = relax.TensorType((4, 1), "float32")
        elif op in ("relax.flatten", "relax.nn.batch_flatten"):
            x_ty = relax.TensorType((1, 4), "float32")
        else:
            raise ValueError(op)

        bb = relax.BlockBuilder()
        x = relax.Var("x", x_ty)
        concat_axis = cls._concat_axis_for_view_op(op)
        with bb.function("main", [x]):
            with bb.dataflow():
                a_expr, b_expr = cls._emit_duplicate_view(op, x)
                a = bb.emit(a_expr)
                b = bb.emit(b_expr)
                prod = bb.emit(relax.op.multiply(a, b))
                out = bb.emit(relax.op.concat([prod, b], axis=concat_axis))
                gv = bb.emit_output(out)
            bb.emit_func_output(gv)
        return bb.finalize()

    @classmethod
    def _input_for_view_op(cls, op):
        if op == "relax.squeeze":
            return cls.storage_ptr_x_squeeze
        if op == "relax.memory.ensure_zero_offset":
            return cls.storage_ptr_x_ensure_zero_offset
        if op in ("relax.permute_dims", "relax.flatten", "relax.nn.batch_flatten"):
            return cls.storage_ptr_x_2d
        return cls.storage_ptr_x_1d

    @staticmethod
    def _torch_duplicate_view(x, op):
        if op == "relax.expand_dims":
            return x.unsqueeze(1)
        if op == "relax.squeeze":
            return x.squeeze(0)
        if op == "relax.reshape":
            return x.reshape(4, 1)
        if op == "relax.permute_dims":
            return x.permute(1, 0)
        if op == "relax.memory.view":
            return x.reshape(4, 1)
        if op == "relax.memory.ensure_zero_offset":
            return x
        if op == "relax.flatten":
            return x.flatten()
        if op == "relax.nn.batch_flatten":
            # TVM: ndim==2 input keeps shape (1, 4).
            return x
        raise ValueError(op)

    @classmethod
    def _expected_for_view_op(cls, op):
        x = torch.from_numpy(np.asarray(cls._input_for_view_op(op), dtype=np.float32))
        a = cls._torch_duplicate_view(x, op)
        b = cls._torch_duplicate_view(x, op)
        prod = a * b
        concat_axis = cls._concat_axis_for_view_op(op)
        return torch.cat([prod, b], dim=concat_axis).numpy()

    @pytest.mark.parametrize(
        "view_op",
        (
            # Keep this list in sync with IsViewMemoryOp() in
            # src/relax/transform/dataflow_inplace.cc
            "relax.expand_dims",
            "relax.squeeze",
            "relax.reshape",
            "relax.permute_dims",
            "relax.flatten",
            "relax.nn.batch_flatten",
            "relax.memory.view",
            "relax.memory.ensure_zero_offset",
        ),
    )
    def test_no_inplace_when_view_ops_share_input(self, view_op):
        mod = self._build_module(view_op)
        func = mod["main"]
        block = func.body.blocks[0]
        params = list(func.params)

        alias_sets, _ = dataflow_alias_analysis(block, params)
        view_vars = [
            binding.var
            for binding in block.bindings
            if (
                isinstance(binding.value, relax.Call)
                and isinstance(binding.value.op, tvm_tirx.ir.Op)
                and binding.value.op.name == view_op
            )
        ]
        a_var, b_var = view_vars[:2]
        assert alias_sets[a_var] & alias_sets[b_var], (
            f"{view_op}: duplicate views should share alias sets, but got "
            f"{alias_sets[a_var]} and {alias_sets[b_var]}"
        )

        _, exact_match = dataflow_inplace_analysis(block, params, mod)
        assert exact_match == [], f"{view_op}: expected no in-place opportunities"

        x_np = self._input_for_view_op(view_op).copy()
        mod_inplace = DataflowUseInplaceCalls()(mod)
        tvm_tirx.ir.assert_structural_equal(mod_inplace, mod)

        ex = relax.build(mod_inplace, tvm_tirx.target.Target("llvm"))
        vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
        out = vm["main"](tvm_tirx.runtime.tensor(x_np, tvm_tirx.cpu()))
        np.testing.assert_allclose(out.numpy(), self._expected_for_view_op(view_op))


if __name__ == "__main__":
    testing.main()
