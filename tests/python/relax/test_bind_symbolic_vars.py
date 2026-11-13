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
import tvm_tirx.testing
from tvm.relax import ExternFunc as _ExternFunc
from tvm.relax import script as R
from tvm_tirx.script import tirx as T


@pytest.fixture(
    params=[
        pytest.param(False, id="replace-by-string"),
        pytest.param(True, id="replace-by-tirx-var"),
    ],
    scope="session",
)
def replace_by_tir_var(request):
    return request.param


def test_bind_static_value(replace_by_tir_var):
    """Symbolic vars may be replaced

    The replaced variables may be given either as strings, or as TIR variables
    """

    M = T.dynamic("M", "int64")
    K = T.dynamic("K", "int64")
    N = T.dynamic("N", "int64")

    @R.function(private=True)
    def before(A: R.Tensor((M, K)), B: R.Tensor((K, N))) -> R.Tensor((M, N)):
        return R.matmul(A, B)

    @R.function(private=True)
    def expected(A: R.Tensor((128, 64)), B: R.Tensor((64, 32))) -> R.Tensor((128, 32)):
        return R.matmul(A, B)

    if replace_by_tir_var:
        M, K = before.params[0].ty.shape
        _, N = before.params[1].ty.shape
        symbolic_var_map = {M: 128, K: 64, N: 32}
    else:
        symbolic_var_map = {"M": 128, "K": 64, "N": 32}

    after = before.bind_symbolic_vars(symbolic_var_map)
    tvm_tirx.ir.assert_structural_equal(expected, after)


def test_error_with_duplicate_var_names():
    """Duplicate variable names may not be replaced by string

    Two TIR variables may have the same name.  If two symbolic
    variables share the same name, the replacement map may not refer
    to that variable by string.
    """
    N1 = T.dynamic("N", "int64")
    N2 = T.dynamic("N", "int64")

    @R.function(private=True)
    def func(A: R.Tensor((N1, N1)), B: R.Tensor((N1, N2))) -> R.Tensor((N1, N2)):
        out: R.Tensor((N1, N2)) = R.matmul(A, B)
        return out

    with pytest.raises(RuntimeError):
        func.bind_symbolic_vars({"N": 64})


def test_string_var_when_other_var_has_duplicate_var_names():
    """Like test_error_with_duplicate_var_names, but replacing a different variable

    If two TIR variables share the same name, the restriction against
    replacing variables by name only applies to those duplicate names.
    Other variables may still be replaced by name.
    """
    N1 = T.dynamic("N", "int64")
    N2 = T.dynamic("N", "int64")
    BatchSize = T.dynamic("BatchSize", "int64")

    @R.function(private=True)
    def before(A: R.Tensor((BatchSize, N1, N1)), B: R.Tensor((N1, N2))) -> R.Tensor(
        (BatchSize, N1, N2)
    ):
        out: R.Tensor((BatchSize, N1, N2)) = R.matmul(A, B)
        return out

    @R.function(private=True)
    def expected(A: R.Tensor((16, N1, N1)), B: R.Tensor((N1, N2))) -> R.Tensor((16, N1, N2)):
        out: R.Tensor((16, N1, N2)) = R.matmul(A, B)
        return out

    after = before.bind_symbolic_vars({"BatchSize": 16})
    tvm_tirx.ir.assert_structural_equal(expected, after)


def test_error_with_nonexisting_var_name():
    """A string name of a symbolic var must be used by the function"""

    M = T.dynamic("M", "int64")
    N = T.dynamic("N", "int64")

    @R.function(private=True)
    def func(A: R.Tensor((M, N))):
        return A

    with pytest.raises(RuntimeError):
        func.bind_symbolic_vars({"non_existing_symbolic_var": 64})


def test_error_with_nonexisting_tir_var():
    """A TIR symbolic var must be a symbolic var of the function"""

    M = T.dynamic("M", "int64")
    N = T.dynamic("N", "int64")

    @R.function(private=True)
    def func(A: R.Tensor([M, N])):
        return A

    with pytest.raises(RuntimeError):
        func.bind_symbolic_vars({tvm_tirx.tirx.Var("M", "int64"): 64})


def test_error_with_multiple_definitions():
    """The string/TIR var syntaxes may not define the same variable"""

    M = T.dynamic("M", "int64")
    N = T.dynamic("N", "int64")

    @R.function(private=True)
    def func(A: R.Tensor([M, N])):
        return A

    tir_var = func.params[0].ty.shape[0]
    symbolic_var_map = {tir_var: 0, "M": 0}

    with pytest.raises(RuntimeError):
        func.bind_symbolic_vars(symbolic_var_map)


def test_error_if_output_has_undefined():
    """The replacements may not introduce undefined symbolic vars"""

    M = T.dynamic("M", "int64")
    N = T.dynamic("N", "int64")

    @R.function(private=True)
    def func(A: R.Tensor([M, N])):
        return A

    outside_var = T.dynamic("outside_var", "int64")

    with pytest.raises(RuntimeError):
        func.bind_symbolic_vars({"M": outside_var * 2})


def test_replacements_may_produce_new_symbolic_vars():
    """The output may introduce symbolic vars, but they must be bound"""

    M = T.dynamic("M", "int64")
    N = T.dynamic("N", "int64")

    @R.function(private=True)
    def before(A: R.Tensor([M, N])):
        return A

    outside_var = T.dynamic("outside_var", "int64")

    @R.function(private=True)
    def expected(A: R.Tensor([outside_var * 2, outside_var])):
        return A

    outside_var = T.dynamic("outside_var", "int64")

    after = before.bind_symbolic_vars({"M": outside_var * 2, "N": outside_var})
    tvm_tirx.ir.assert_structural_equal(expected, after)


def test_bind_symbolic_vars_in_tensor_shape():
    """The bound variable should be replaced when appearing in type"""

    M = T.dynamic("M", "int64")
    N = T.dynamic("N", "int64")

    @R.function(private=True)
    def before(A: R.Tensor([M, N])):
        B = R.call_dps_packed(_ExternFunc("dummy_func"), [A], ty_args=[R.Tensor([2 * M * N])])
        return B

    M = T.dynamic("M", "int64")

    @R.function(private=True)
    def expected(A: R.Tensor([M, 16])):
        B = R.call_dps_packed(_ExternFunc("dummy_func"), [A], ty_args=[R.Tensor([M * 32])])
        return B

    after = before.bind_symbolic_vars({"N": 16})
    tvm_tirx.ir.assert_structural_equal(expected, after)


def test_bind_symbolic_vars_in_shape_expr():
    """The bound variable should be replaced when appearing in R.Shape"""

    M = T.dynamic("M", "int64")
    N = T.dynamic("N", "int64")

    @R.function(private=True)
    def before(A: R.Tensor([M * N]), x: R.Shape([M, N])):
        B = R.call_dps_packed(_ExternFunc("dummy_func"), [A], ty_args=[R.Tensor([2 * M * N])])
        return B

    M = T.dynamic("M", "int64")

    @R.function(private=True)
    def expected(A: R.Tensor([M * 16]), x: R.Shape([M, 16])):
        B = R.call_dps_packed(_ExternFunc("dummy_func"), [A], ty_args=[R.Tensor([M * 32])])
        return B

    after = before.bind_symbolic_vars({"N": 16})
    tvm_tirx.ir.assert_structural_equal(expected, after)


def test_bind_strided_slice():
    """relax.op.strided_slice stores Expr attributes"""

    N = T.dynamic("N", "int64")
    M = T.dynamic("M", "int64")

    @R.function(private=True)
    def before(A: R.Tensor([M, N])):
        B = R.strided_slice(A, [1], [0], [N // 4])
        return B

    M = T.dynamic("M", "int64")

    @R.function(private=True)
    def expected(A: R.Tensor([M, 32])):
        # Binding substitutes runtime primitive arguments without applying
        # shape-only analyzer simplification to them.
        B = R.strided_slice(A, [1], [0], [T.FloorDiv(T.int64(32), T.int64(4))])
        return B

    after = before.bind_symbolic_vars({"N": 32})
    tvm_tirx.ir.assert_structural_equal(expected, after)


def test_bind_inside_match_cast():
    """Symbolic variables may occur within R.match_cast"""

    M = T.dynamic("M", "int64")
    N = T.dynamic("N", "int64")

    @R.function(private=True)
    def before(A: R.Tensor([M, N]), B: R.Tensor(ndim=2)):
        C = R.match_cast(B, R.Tensor([M, N]))
        D = R.add(A, C)
        return D

    M = T.dynamic("M", "int64")

    @R.function(private=True)
    def expected(A: R.Tensor([M, 32]), B: R.Tensor(ndim=2)):
        C = R.match_cast(B, R.Tensor([M, 32]))
        D = R.add(A, C)
        return D

    after = before.bind_symbolic_vars({"N": 32})
    tvm_tirx.ir.assert_structural_equal(expected, after)


if __name__ == "__main__":
    tvm_tirx.testing.main()
