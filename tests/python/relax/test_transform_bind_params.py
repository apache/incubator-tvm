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

import numpy as np
import pytest

import tvm
import tvm_tirx
import tvm_tirx.script
import tvm_tirx.testing
from tvm import relax
from tvm.relax import ExternFunc as _ExternFunc
from tvm.relax import script as R
from tvm.s_tir import script as Ts
from tvm_tirx.script import tirx as T


@pytest.fixture(params=[False, True], scope="session")
def use_np_array(request):
    return request.param


def test_bind_params(use_np_array):
    @tvm_tirx.script.ir_module
    class InputModule:
        @Ts.function
        def tir_matmul(
            A: T.Tensor((16, 16), "float32"),
            B: T.Tensor((16, 16), "float32"),
            C: T.Tensor((16, 16), "float32"),
        ) -> None:
            T.func_attr({"global_symbol": "tir_matmul"})

            for i0, j, k0, i1, k1 in T.grid(4, 16, 4, 4, 4):
                with Ts.sblock("matmul"):
                    vi = Ts.axis.S(16, i0 * 4 + i1)
                    vj = Ts.axis.S(16, j)
                    vk = Ts.axis.R(16, k0 * 4 + k1)
                    with Ts.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

        @R.function
        def main(x: R.Tensor((16, 16), "float32"), w: R.Tensor((16, 16), "float32")) -> R.Tensor(
            (16, 16), "float32"
        ):
            gv0 = R.call_tir(
                InputModule.tir_matmul, (x, w), ty_args=[R.Tensor((16, 16), dtype="float32")]
            )
            return gv0

    x_np = np.random.rand(16, 16).astype(np.float32)
    w_np = np.random.rand(16, 16).astype(np.float32)
    x_tvm = tvm_tirx.runtime.tensor(x_np)
    w_tvm = tvm_tirx.runtime.tensor(w_np)
    params_dict = {"w": w_np if use_np_array else w_tvm}
    mod = relax.transform.BindParams("main", params_dict)(InputModule)
    assert len(mod["main"].params) == 1

    target = tvm_tirx.target.Target("llvm")
    ex_after = tvm.compile(mod, target)
    vm_after = relax.VirtualMachine(ex_after, tvm_tirx.cpu())
    res_after = vm_after["main"](x_tvm)

    ex_before = tvm.compile(InputModule, target)
    vm_before = relax.VirtualMachine(ex_before, tvm_tirx.cpu())
    res_before = vm_before["main"](x_tvm, w_tvm)

    tvm_tirx.testing.assert_allclose(res_before.numpy(), res_after.numpy())


def test_bind_params_symbolic_vars():
    batch = T.dynamic("batch", "int64")
    k = T.dynamic("k", "int64")
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((batch, m), dtype="float32"),
            w0: R.Tensor((n, m), dtype="float32"),
            b0: R.Tensor((n,), dtype="float32"),
            w1: R.Tensor((k, n), dtype="float32"),
            b1: R.Tensor((k,), dtype="float32"),
        ) -> R.Tensor((batch, k), dtype="float32"):
            with R.dataflow():
                lv0 = R.call_dps_packed(
                    _ExternFunc("linear0"),
                    (x, w0, b0),
                    ty_args=[R.Tensor((batch, n), dtype="float32")],
                )
                out = R.call_dps_packed(
                    _ExternFunc("linear1"),
                    (lv0, w1, b1),
                    ty_args=[R.Tensor((batch, k), dtype="float32")],
                )
                R.output(out)
            return out

    m, n, k = 4, 6, 8
    w0_tvm = tvm_tirx.runtime.tensor(np.random.rand(n, m).astype(np.float32))
    b0_tvm = tvm_tirx.runtime.tensor(np.random.rand(n).astype(np.float32))
    w1_tvm = tvm_tirx.runtime.tensor(np.random.rand(k, n).astype(np.float32))
    b1_tvm = tvm_tirx.runtime.tensor(np.random.rand(k).astype(np.float32))
    params_dict = {"w0": w0_tvm, "b0": b0_tvm, "w1": w1_tvm, "b1": b1_tvm}
    mod = relax.transform.BindParams("main", params_dict)(Before)

    # Since it contains ConstantObj, it's hard to check with structural equality.
    func = mod["main"]
    assert len(func.params) == 1
    batch = func.params[0].ty.shape[0]
    tvm_tirx.ir.assert_structural_equal(func.params[0].ty, relax.TensorType((batch, 4), "float32"))
    tvm_tirx.ir.assert_structural_equal(func.ret_ty, relax.TensorType((batch, 8), "float32"))
    bindings = func.body.blocks[0].bindings
    tvm_tirx.ir.assert_structural_equal(bindings[0].var.ty, relax.TensorType((batch, 6), "float32"))
    tvm_tirx.ir.assert_structural_equal(bindings[1].var.ty, relax.TensorType((batch, 8), "float32"))


def test_bind_params_consistent_explicit_and_inferred_symbolic_var():
    prim_type = tvm_tirx.ir.PrimType("int64")
    n = relax.Var("n", prim_type)
    data = relax.Var("data", relax.TensorType([n], "float32"))
    before = relax.Function([n, data], n, prim_type)
    explicit_value = tvm_tirx.tirx.Add(
        tvm_tirx.tirx.const(2, "int64"), tvm_tirx.tirx.const(3, "int64")
    )

    after = before.bind_params({n: explicit_value, data: np.zeros([5], "float32")})

    assert not after.params
    assert after.body.body.same_as(explicit_value)


def test_bind_params_conflicting_explicit_and_inferred_symbolic_var():
    prim_type = tvm_tirx.ir.PrimType("int64")
    n = relax.Var("n", prim_type)
    data = relax.Var("data", relax.TensorType([n], "float32"))
    before = relax.Function([n, data], n, prim_type)

    with pytest.raises(
        ValueError,
        match=(
            "Explicit binding for symbolic variable n conflicts with the value inferred from a "
            "parameter shape"
        ),
    ):
        before.bind_params({n: 5, data: np.zeros([7], "float32")})


def test_bind_params_rejects_non_primitive_explicit_symbolic_var():
    prim_type = tvm_tirx.ir.PrimType("int64")
    n = relax.Var("n", prim_type)
    data = relax.Var("data", relax.TensorType([n], "float32"))
    before = relax.Function([n, data], n, prim_type)

    with pytest.raises(
        ValueError,
        match="Explicit binding for symbolic variable n must be a primitive expression",
    ):
        before.bind_params(
            {
                n: relax.const(np.array(5, dtype="int64")),
                data: np.zeros([5], "float32"),
            }
        )


def test_bind_params_rejects_non_primitive_standalone_symbolic_var():
    prim_type = tvm_tirx.ir.PrimType("int64")
    n = relax.Var("n", prim_type)
    before = relax.Function([n], n, prim_type)

    with pytest.raises(
        ValueError,
        match="Explicit binding for symbolic variable n must be a primitive expression",
    ):
        before.bind_params({n: relax.const(np.array(5, dtype="int64"))})


@pytest.fixture(params=["by_string", "by_var"], scope="session")
def param_specification(request):
    return request.param


def test_bind_params_by_var_obj(param_specification):
    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor([16], "float32")):
            return A

    np_data = np.arange(16).astype("float32")
    inlined_relax_const = relax.const(np_data)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main():
            return inlined_relax_const

    if param_specification == "by_string":
        var = "A"
    elif param_specification == "by_var":
        var = Before["main"].params[0]
    else:
        raise ValueError("Unknown param_specification: {param_specification}")

    After = relax.transform.BindParams("main", {var: np_data})(Before)

    tvm_tirx.ir.assert_structural_equal(Expected, After)


def test_bind_params_by_var_name():
    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor([16], "float32")):
            return A

    np_data = np.arange(16).astype("float32")
    inlined_relax_const = relax.const(np_data)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main():
            return inlined_relax_const

    After = relax.transform.BindParams("main", {"A": np_data})(Before)

    tvm_tirx.ir.assert_structural_equal(Expected, After)


if __name__ == "__main__":
    tvm_tirx.testing.main()
