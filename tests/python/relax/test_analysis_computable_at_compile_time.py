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


import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.relax import script as R
from tvm_tirx.script import tirx as T


def _analyze_func(func: tvm.relax.Function) -> list[str]:
    return [var.name for var in tvm.relax.analysis.computable_at_compile_time(func)]


def test_no_num_input_attribute():
    """Without the "num_input" attribute, all params are runtime"""

    @R.function
    def func(A: R.Tensor([16], "int32"), B: R.Tensor([16], "int32")):
        C = R.add(A, B)
        return C

    assert _analyze_func(func) == []


def test_compile_time_param():
    """Parameters after "num_input" are known at compile-time"""

    @R.function
    def func(A: R.Tensor([16], "int32"), B: R.Tensor([16], "int32")):
        R.func_attr({"num_input": 1})
        return ()

    assert _analyze_func(func) == ["B"]


def test_binding_using_one_param():
    """Bindings may be computable at compile-time"""

    @R.function
    def func(A: R.Tensor([16], "int32"), B: R.Tensor([16], "int32")):
        R.func_attr({"num_input": 1})
        C = R.add(B, B)
        D = R.add(A, C)
        return D

    assert _analyze_func(func) == ["B", "C"]


def test_binding_using_multiple_params():
    """Compile-time bindings may use multiple parameters"""

    @R.function
    def func(A: R.Tensor([16], "int32"), B: R.Tensor([16], "int32"), C: R.Tensor([16], "int32")):
        R.func_attr({"num_input": 1})
        D = R.add(B, C)
        E = R.add(A, D)
        return E

    assert _analyze_func(func) == ["B", "C", "D"]


def test_compile_time_binding_after_run_time():
    """Compile-time bindings may occur after run-time

    A binding being computable at compile-time only depends on the
    arguments used for it.  A value that is computable at compile-time
    may occur after a value that is only computable at run-time.
    """

    @R.function
    def func(A: R.Tensor([16], "int32"), B: R.Tensor([16], "int32")):
        R.func_attr({"num_input": 1})
        C = R.add(A, A)
        D = R.add(B, B)
        E = R.add(C, D)
        return E

    assert _analyze_func(func) == ["B", "D"]


def test_sequential_compile_time_bindings():
    """Compile-time bindings may occur after run-time

    A compile-time value may depend on variables defined within the
    function, so long as those variables are themselves computable at
    compile-time.
    """

    @R.function
    def func(A: R.Tensor([16], "int32"), B: R.Tensor([16], "int32")):
        R.func_attr({"num_input": 1})
        C = R.add(B, B)
        D = R.add(C, C)
        E = R.add(D, D)
        F = R.add(E, A)
        return F

    assert _analyze_func(func) == ["B", "C", "D", "E"]


def test_dataflow_vars():
    """Compile-time bindings may occur in dataflow blocks"""

    @R.function
    def func(A: R.Tensor([16], "int32"), B: R.Tensor([16], "int32")):
        R.func_attr({"num_input": 1})
        with R.dataflow():
            C = R.add(B, B)
            D = R.add(C, C)
            E = R.add(D, D)
            F = R.add(E, A)
            R.output(F)
        return F

    assert _analyze_func(func) == ["B", "C", "D", "E"]


def test_compile_time_symbolic_shape():
    """Compile-time bindings may contain symbolic shapes"""

    n = T.dynamic("n", "int64")

    @R.function
    def func(A: R.Tensor([1], "int32"), B: R.Tensor([n], "int32")):
        R.func_attr({"num_input": 1})

        C: R.Tensor([n], "int32") = R.add(B, B)
        D: R.Tensor([], "int32") = R.max(C, axis=0)
        E: R.Tensor([1], "int32") = R.add(A, D)
        return E

    assert _analyze_func(func) == ["B", "C", "D"]


def test_symbolic_variables_from_match_binding():
    """Symbolic vars may be inferred from compile-time bindings"""

    n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int64")

    @R.function
    def func(A: R.Tensor(ndim=1, dtype="int32"), B: R.Tensor(ndim=1, dtype="int32")):
        R.func_attr({"num_input": 1})

        A2 = R.match_cast(A, R.Tensor([n], "int32"))
        B2 = R.match_cast(B, R.Tensor([m], "int32"))

        C = R.add(B2, B2)
        D = R.max(C, axis=0)
        E = R.max(A2, axis=0)
        F = R.add(D, E)
        return F

    assert _analyze_func(func) == ["B", "B2", "C", "D"]


def test_compile_time_expressions_may_not_use_runtime_symbolic_variables():
    """Symbolic vars may be inferred from compile-time bindings

    Here, `C` uses the symbolic variable `m`, which can be inferred
    from the shape of `B` and is known at compile-time.  However, `D`
    uses the symbolic variable `n`, which cannot be inferred without
    first knowing `A`, and is therefore unknown at compile-time.
    """

    n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int64")

    @R.function
    def func(A: R.Tensor([n], "int32"), B: R.Tensor([m], "int32")):
        R.func_attr({"num_input": 1})

        C = R.ones([m], "int32")
        D = R.ones([n], "int32")

        E = (C, D)
        return E

    assert _analyze_func(func) == ["B", "C"]


def test_compile_time_expressions_may_infer_same_variable_as_run_time():
    """Symbolic vars may be inferred from compile-time bindings

    A symbolic variable may be inferrable from multiple sources.
    Here, while `n` can be inferred from the runtime parameter `A`, it
    can also be inferred from the compile-time parameter `B`.
    """

    n = T.dynamic("n", "int64")

    @R.function
    def func(A: R.Tensor([n], "int32"), B: R.Tensor([n], "int32")):
        R.func_attr({"num_input": 1})

        C = R.ones([n], "int32")
        D = R.ones([n], "int32")

        E = (C, D)
        return E

    assert _analyze_func(func) == ["B", "C", "D", "E"]


def test_compile_time_expressions_may_use_variables_from_match_cast():
    """Symbolic vars may be inferred from compile-time bindings

    Here, `C` uses the symbolic variable `m`, which can be inferred
    from the shape of `B` and is known at compile-time.  However, `D`
    uses the symbolic variable `n`, which cannot be inferred without
    first knowing `A`, and is therefore unknown at compile-time.
    """

    n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int64")

    @R.function
    def func(A: R.Tensor([n], "int32"), B: R.Tensor(ndim=1, dtype="int32")):
        R.func_attr({"num_input": 1})

        B2 = R.match_cast(B, R.Tensor([m], "int32"))

        C = R.ones([m], "int32")
        D = R.ones([n], "int32")

        E = (C, D)
        return E

    assert _analyze_func(func) == ["B", "B2", "C"]


if __name__ == "__main__":
    tvm_tirx.testing.main()
