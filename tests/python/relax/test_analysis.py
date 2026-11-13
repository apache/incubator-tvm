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
# ruff: noqa: F811, F841

import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm import relax as rx
from tvm.relax import ExternFunc as _ExternFunc
from tvm.relax import script as R
from tvm.relax.analysis import (
    all_global_vars,
    all_vars,
    bound_vars,
    free_vars,
    has_reshape_pattern,
    name_to_binding,
    remove_all_unused,
    udchain,
    used_vars,
)
from tvm.s_tir import script as Ts
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def var_name_set(vars: list[rx.Var | rx.GlobalVar]) -> set[str]:
    return {v.name if isinstance(v, rx.Var) else v.name_hint for v in vars}


def test_use_def():
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")
    x = rx.Var("x", R.Tensor([m, n], "float16"))
    y = rx.Var("y", R.Tensor([n], "float16"))
    ib = rx.BlockBuilder()
    with ib.function("func", [x, y]):
        with ib.dataflow():
            lv0 = ib.emit(rx.op.add(x, y))
            lv1 = ib.emit(rx.op.multiply(lv0, y))
            gv0 = ib.emit_output(lv1)
        ib.emit_func_output(gv0)
    dfb = ib.get()["func"].body.blocks[0]
    udc = udchain(dfb)
    assert set(udc[x]) == {lv0}
    assert set(udc[y]) == {lv0, lv1}
    assert set(udc[lv0]) == {lv1}
    assert set(udc[lv1]) == {gv0}
    assert set(udc[gv0]) == set()


@pytest.mark.parametrize(
    "expr_fn, expected_var_names",
    [
        (lambda x, y, z: rx.op.add(x, y), {"x", "y"}),
        (lambda x, y, z: rx.op.multiply(x, x), {"x"}),
        (lambda x, y, z: rx.Tuple([x, y, z]), {"x", "y", "z"}),
    ],
    ids=["binary_op", "self_reference", "tuple"],
)
def test_used_vars(expr_fn, expected_var_names):
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")
    x = rx.Var("x", R.Tensor([m, n], "float16"))
    y = rx.Var("y", R.Tensor([n], "float16"))
    z = rx.Var("z", R.Tensor([m], "float16"))

    expr = expr_fn(x, y, z)
    result = used_vars(expr)
    assert var_name_set(result) == expected_var_names


def test_chained_remove_all_unused():
    @tvm_tirx.script.ir_module
    class IdentityUnused:
        @R.function
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
            with R.dataflow():
                lv0 = x
                unused0 = R.call_dps_packed(
                    _ExternFunc("my_sigmoid"), (x,), ty_args=[R.Tensor((32, 32), dtype="float32")]
                )
                unused1 = R.call_dps_packed(
                    _ExternFunc("my_dps_func"),
                    (unused0,),
                    ty_args=[R.Tensor((32, 32), dtype="float32")],
                )
                R.output(lv0)
            return lv0

    optimized = remove_all_unused(IdentityUnused["main"])

    @tvm_tirx.script.ir_module
    class GroundTruth:
        @R.function
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
            with R.dataflow():
                lv0 = x
                R.output(lv0)
            return lv0

    tvm_tirx.ir.assert_structural_equal(optimized, GroundTruth["main"])


def test_binding_block_remove_all_unused():
    """Remove unused dataflow bindings

    Removal of unused bindings may not remove side effects.  Since
    bindings within a dataflow block are guaranteed not to have side
    effects, they may be removed if unused.
    """

    @tvm_tirx.script.ir_module
    class IdentityUnused:
        @R.function(pure=False)
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
            with R.dataflow():
                lv0 = x
                unused0 = R.call_dps_packed(
                    _ExternFunc("my_sigmoid"), (x,), ty_args=[R.Tensor((32, 32), dtype="float32")]
                )
                unused1 = R.call_dps_packed(
                    _ExternFunc("my_dps_func"),
                    (unused0,),
                    ty_args=[R.Tensor((32, 32), dtype="float32")],
                )
                R.output(lv0)
            z = R.call_packed("vm.builtin.copy", lv0, ty_args=(R.Tensor((32, 32), "float32")))
            return z

    optimized = remove_all_unused(IdentityUnused["main"])

    @tvm_tirx.script.ir_module
    class GroundTruth:
        @R.function(pure=False)
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
            with R.dataflow():
                lv0 = x
                R.output(lv0)
            z = R.call_packed("vm.builtin.copy", lv0, ty_args=(R.Tensor((32, 32), "float32")))
            return z

    tvm_tirx.ir.assert_structural_equal(optimized, GroundTruth["main"])


def test_binding_block_remove_unused_pure_without_dataflow():
    """Remove unused dataflow bindings

    Removal of unused bindings may not remove side effects.  Unused
    bindings whose value is a pure operation
    (e.g. `R.call_dps_packed`) may be removed, even if outside of a
    dataflow block.
    """

    @R.function(private=True)
    def before(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        lv0 = x
        unused0 = R.call_dps_packed(
            _ExternFunc("my_sigmoid"), (x,), ty_args=[R.Tensor((32, 32), dtype="float32")]
        )
        unused1 = R.call_dps_packed(
            _ExternFunc("my_dps_func"), (unused0,), ty_args=[R.Tensor((32, 32), dtype="float32")]
        )
        return x

    @R.function(private=True)
    def expected(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        return x

    after = remove_all_unused(before)
    tvm_tirx.ir.assert_structural_equal(expected, after)


def test_binding_block_keep_impure_without_dataflow():
    """Remove unused dataflow bindings

    Removal of unused bindings may not remove side effects.  Unused
    bindings whose value is an impure operation (e.g. `R.call_packed`)
    may not be removed, as outside of a dataflow block they may
    contain side effects.
    """

    @R.function(private=True, pure=False)
    def before(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        lv0 = x
        y = R.call_packed("vm.builtin.copy", lv0, ty_args=(R.Tensor((32, 32), "float32")))
        return y

    expected = before

    after = remove_all_unused(before)
    tvm_tirx.ir.assert_structural_equal(expected, after)


def test_binding_block_keep_pure_func_used_only_for_impure():
    """Keep bindings that are used for impure functions

    Removal of unused bindings may not result in use of undefined
    variables.  Unused bindings whose value is an impure operation
    (e.g. `R.call_packed`) may not be removed, nor may any of their
    inputs.

    This is a regression test to catch an earlier failure mode, in
    which tracking of unused variables only back-propagated from the
    return value of functions, and did not consider variables that
    were required to execute impure function calls.  In that failure
    mode, the binding of `y` would be removed as unused, even though
    it was required to evaluate the packed function.
    """

    @R.function(private=True, pure=False)
    def before(x: R.Tensor((32, 32), "int32")):
        y = x * R.const(2)
        z = R.call_packed(
            "function_maybe_with_side_effects", y, ty_args=(R.Tensor((32, 32), "int32"))
        )
        return ()

    expected = before

    after = remove_all_unused(before)
    tvm_tirx.ir.assert_structural_equal(expected, after)


def test_binding_block_remove_all_unused_func_without_dataflow():
    @tvm_tirx.script.ir_module
    class IdentityUnused:
        @R.function(pure=False)
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
            lv0 = x

            @R.function
            def internal_unused_func(A: R.Tensor((32, 32), "float32")) -> R.Tensor:
                return A

            z = R.call_packed("vm.builtin.copy", lv0, ty_args=(R.Tensor((32, 32), "float32")))
            return z

    optimized = remove_all_unused(IdentityUnused["main"])

    @tvm_tirx.script.ir_module
    class GroundTruth:
        @R.function(pure=False)
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
            lv0 = x
            z = R.call_packed("vm.builtin.copy", lv0, ty_args=(R.Tensor((32, 32), "float32")))
            return z

    tvm_tirx.ir.assert_structural_equal(optimized, GroundTruth["main"])


def test_binding_block_fake_unused_remove_all_unused():
    @tvm_tirx.script.ir_module
    class IdentityUnused:
        @R.function(pure=False)
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
            with R.dataflow():
                lv0 = x
                R.output(lv0)
            z = R.call_packed("vm.builtin.copy", lv0, ty_args=(R.Tensor((32, 32), "float32")))
            return lv0

    optimized = remove_all_unused(IdentityUnused["main"])

    @tvm_tirx.script.ir_module
    class GroundTruth:
        @R.function(pure=False)
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
            with R.dataflow():
                lv0 = x
                R.output(lv0)
            # This might bring side effect so cannot be removed.
            z = R.call_packed("vm.builtin.copy", lv0, ty_args=(R.Tensor((32, 32), "float32")))
            return lv0

    tvm_tirx.ir.assert_structural_equal(optimized, GroundTruth["main"])


def test_edge_binding_block_fake_unused_remove_all_unused():
    @tvm_tirx.script.ir_module
    class IdentityUnused:
        @R.function(pure=False)
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor((32, 32), "float32"):
            z = R.call_packed("vm.builtin.copy", x, ty_args=(R.Tensor((32, 32), "float32")))
            return x

    optimized = remove_all_unused(IdentityUnused["main"])
    tvm_tirx.ir.assert_structural_equal(optimized, IdentityUnused["main"])


def test_edge_binding_block_fake_unused_remove_all_unused2():
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")
    k = T.dynamic("k", "int64")

    @tvm_tirx.script.ir_module
    class IdentityUnused:
        @R.function
        def main(x: R.Tensor((3,), dtype="int64")) -> R.Tensor(dtype="int32", ndim=3):
            with R.dataflow():
                lv: R.Shape(ndim=3) = R.call_pure_packed(
                    "vm.builtin.tensor_to_shape", x, ty_args=(R.Shape(ndim=3),)
                )
                lv1: R.Shape([m, n, k]) = R.match_cast(lv, R.Shape([m, n, k]))
                gv: R.Tensor((m, n, k), dtype="int32") = R.full(
                    R.shape([m, n, k]), R.const(1, "int32"), dtype="int32"
                )
                R.output(gv)
            return gv

    optimized = remove_all_unused(IdentityUnused["main"])
    tvm_tirx.ir.assert_structural_equal(optimized, IdentityUnused["main"])


def test_remove_all_unused_from_dataflow_block():
    """Like test_chained_remove_all_unused, but on a SeqExpr"""

    @R.function
    def before(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        with R.dataflow():
            lv0 = x
            unused0 = R.call_dps_packed(
                _ExternFunc("my_sigmoid"), (x,), ty_args=[R.Tensor((32, 32), dtype="float32")]
            )
            unused1 = R.call_dps_packed(
                _ExternFunc("my_dps_func"),
                (unused0,),
                ty_args=[R.Tensor((32, 32), dtype="float32")],
            )
            R.output(lv0)
        return lv0

    @R.function
    def expected(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        with R.dataflow():
            lv0 = x
            R.output(lv0)
        return lv0

    after = remove_all_unused(before.body)
    tvm_tirx.ir.assert_structural_equal(expected.body, after, map_free_vars=True)


def test_remove_all_unused_from_binding_block():
    """Like test_chained_remove_all_unused, but on a SeqExpr"""

    @R.function
    def before(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        lv0 = x
        unused0 = R.call_dps_packed(
            _ExternFunc("my_sigmoid"), (x,), ty_args=[R.Tensor((32, 32), dtype="float32")]
        )
        unused1 = R.call_dps_packed(
            _ExternFunc("my_dps_func"), (unused0,), ty_args=[R.Tensor((32, 32), dtype="float32")]
        )
        return lv0

    @R.function
    def expected(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        lv0 = x
        return lv0

    after = remove_all_unused(before.body)
    tvm_tirx.ir.assert_structural_equal(expected.body, after, map_free_vars=True)


def test_retain_impure_calls_unused_in_binding_block():
    """An impure call may have side effects, and must be kept"""

    @R.function(pure=False)
    def before(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        lv0 = x
        unused0 = R.call_packed("my_impure_call", x, ty_args=R.Tensor((32, 32), dtype="float32"))
        unused1 = R.call_dps_packed(
            _ExternFunc("my_unused_call"), (lv0,), ty_args=[R.Tensor((32, 32), dtype="float32")]
        )
        return lv0

    @R.function(pure=False)
    def expected(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        lv0 = x
        unused0 = R.call_packed("my_impure_call", x, ty_args=R.Tensor((32, 32), dtype="float32"))
        return lv0

    after = remove_all_unused(before.body)
    tvm_tirx.ir.assert_structural_equal(expected.body, after, map_free_vars=True)


def test_retain_calls_to_impure_builtin_ops():
    n = T.dynamic("n", "int64")

    @I.ir_module
    class Module:
        @Ts.function(private=True)
        def my_tir(A: T.handle, B: T.handle, n: T.int64):
            T.evaluate(0)

        @R.function(pure=False)
        def main(x: R.Tensor((n,), "float32")):
            cls = Module
            storage = R.memory.alloc_storage((n * 4,), 0, "global", "float32")
            alloc = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([n]), "float32")
            # "call_tir_dyn" is impure which shouldn't be removed.
            R.vm.call_tir_dyn(cls.my_tir, (x, alloc, R.shape([n])))
            # "kill_tensor"/"kill_storage" are impure which shouldn't be removed.
            R.memory.kill_tensor(alloc)
            R.memory.kill_storage(storage)
            return x

    after = remove_all_unused(Module["main"])
    tvm_tirx.ir.assert_structural_equal(after, Module["main"], map_free_vars=True)


def test_name_to_binding_var_shadowing():
    @R.function
    def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        with R.dataflow():
            lv0 = x
            lv1 = lv0
            R.output(lv1)

        with R.dataflow():
            lv0 = lv1  # shadowing
            lv2 = lv0
            R.output(lv2)
        return lv2

    n2binding = name_to_binding(main)

    assert "lv0" in n2binding
    assert "lv1" in n2binding
    assert "lv2" in n2binding

    assert len(n2binding["lv0"]) == 2


@tvm_tirx.script.ir_module
class VarExample:
    @R.function
    def func(a: R.Tensor) -> R.Tensor:
        # normalized into assigning R.add(a, a) to a var and returning it
        return R.add(a, a)

    @R.function
    def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
        cls = VarExample
        z = R.add(x, y)
        # no binding here
        _ = R.match_cast(x, R.Tensor((5, 5)))
        with R.dataflow():
            q = R.add(z, z)
            p = cls.func(q)
            r = R.match_cast(p, R.Tensor((5, 5)))
            s = r
            R.output(s)
        return s


def test_all_vars():
    vars = all_vars(VarExample["func"])
    assert len(vars) == 2
    assert vars[0].name == "a"
    # the body of the seq expr in the func body is a var
    assert vars[1] == VarExample["func"].body.body

    var_names = var_name_set(all_vars(VarExample["main"]))
    assert var_names == {"_", "x", "y", "z", "p", "q", "r", "s"}


def test_all_vars_from_expr_using_dataflow():
    """all_vars() should return all Var, including DataflowVar"""
    func = VarExample["main"]
    cls_func_q = func.body.blocks[1].bindings[1].value

    var_names = var_name_set(all_vars(cls_func_q))
    assert var_names == {"q"}


def test_bound_vars():
    vars = bound_vars(VarExample["func"])
    assert len(vars) == 2
    assert vars[0].name == "a"
    # the body of the seq expr in the func body is a bound var
    assert vars[1] == VarExample["func"].body.body

    # all the vars are bound
    var_names = var_name_set(bound_vars(VarExample["main"]))
    assert var_names == {"_", "x", "y", "z", "p", "q", "r", "s"}

    # if we consider only the body, then the function arguments are not bound
    body_names = var_name_set(bound_vars(VarExample["main"].body))
    assert body_names == {"_", "z", "p", "q", "r", "s"}

    # only binding is in the (normalized) body
    simple_body_vars = bound_vars(VarExample["func"].body)
    assert len(simple_body_vars) == 1
    assert simple_body_vars[0] == VarExample["func"].body.body


def test_free_vars():
    # all the vars are bound
    assert len(free_vars(VarExample["func"])) == 0
    assert len(free_vars(VarExample["main"])) == 0

    # the arguments are free if we look only at the bodies
    func_free = var_name_set(free_vars(VarExample["func"].body))
    main_free = var_name_set(free_vars(VarExample["main"].body))
    assert len(func_free) == 1
    assert len(main_free) == 2
    assert "a" in func_free
    assert main_free == {"x", "y"}

    # function that captures vars
    x = rx.Var("x", R.Tensor(ndim=-1))
    y = rx.Var("y", R.Tensor(ndim=-1))
    z = rx.Var("z", R.Tensor(ndim=-1))
    inner = rx.Function(
        [z],
        rx.op.add(x, rx.op.add(y, z)),
        ret_ty=R.Tensor(ndim=-1),
    )
    outer = rx.Function(
        [x, y],
        rx.Call(inner, [y]),
        ret_ty=R.Tensor(ndim=-1),
    )
    assert len(free_vars(outer)) == 0
    assert var_name_set(free_vars(inner)) == {"x", "y"}


@pytest.mark.parametrize("definition_site", ["parameter", "match_cast"])
def test_free_vars_primitive_definition_sites(definition_site):
    n = T.dynamic("n", "int64")
    if definition_site == "parameter":
        y = rx.Var("y", rx.TensorType([n], "float32"))
    else:
        y = rx.Var("y", rx.TensorType(ndim=1, dtype="float32"))

    bb = rx.BlockBuilder()
    with bb.function("inner", [y]):
        if definition_site == "match_cast":
            bb.match_cast(y, rx.TensorType([n], "float32"))
        output = bb.emit(rx.op.ones((n,), "float32"))
        bb.emit_func_output(output)

    inner = bb.get()["inner"]
    rx.analysis.check_well_formed(inner)
    assert not free_vars(inner)


def test_all_global_vars():
    # there is one call to "func"
    global_vars = all_global_vars(VarExample["main"])
    assert len(global_vars) == 1
    assert global_vars[0].name_hint == "func"

    gv1 = rx.GlobalVar("gv1")
    gv2 = rx.GlobalVar("gv2")
    gv3 = rx.GlobalVar("gv3")
    call = rx.Call(gv1, [gv2, gv3])
    call_var_names = var_name_set(all_global_vars(call))
    assert call_var_names == {"gv1", "gv2", "gv3"}


def test_reshape_pattern_reshape():
    @Ts.function
    def reshape(
        rxplaceholder: T.Tensor((1, 2, 3, 4), "float32"),
        T_reshape: T.Tensor((8, 3), "float32"),
    ):
        for i0, i1 in T.grid(8, 3):
            with Ts.sblock("T_reshape"):
                ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                Ts.reads(
                    rxplaceholder[
                        (ax0 * 3 + ax1) // 24,
                        (ax0 * 3 + ax1) % 24 // 12,
                        (ax0 * 3 + ax1) % 12 // 4,
                        (ax0 * 3 + ax1) % 4,
                    ]
                )
                Ts.writes(T_reshape[ax0, ax1])
                T_reshape[ax0, ax1] = rxplaceholder[
                    (ax0 * 3 + ax1) // 24,
                    (ax0 * 3 + ax1) % 24 // 12,
                    (ax0 * 3 + ax1) % 12 // 4,
                    (ax0 * 3 + ax1) % 4,
                ]

    assert has_reshape_pattern(reshape)


def test_reshape_pattern_reshape_scheduled():
    @Ts.function
    def reshape_scheduled(
        rxplaceholder: T.Tensor((1, 2, 3, 4), "float32"),
        T_reshape: T.Tensor((8, 3), "float32"),
    ):
        for i0_i1_fused_0 in T.parallel(1, annotations={"thread_binding": "blockIdx.x"}):
            for i0_i1_fused_1 in T.parallel(24, annotations={"thread_binding": "threadIdx.x"}):
                with Ts.sblock("T_reshape"):
                    ax0 = Ts.axis.spatial(8, (i0_i1_fused_0 * 24 + i0_i1_fused_1) // 3)
                    ax1 = Ts.axis.spatial(3, (i0_i1_fused_0 * 24 + i0_i1_fused_1) % 3)
                    Ts.reads(
                        rxplaceholder[
                            (ax0 * 3 + ax1) // 24,
                            (ax0 * 3 + ax1) % 24 // 12,
                            (ax0 * 3 + ax1) % 12 // 4,
                            (ax0 * 3 + ax1) % 4,
                        ]
                    )
                    Ts.writes(T_reshape[ax0, ax1])
                    T_reshape[ax0, ax1] = rxplaceholder[
                        (ax0 * 3 + ax1) // 24,
                        (ax0 * 3 + ax1) % 24 // 12,
                        (ax0 * 3 + ax1) % 12 // 4,
                        (ax0 * 3 + ax1) % 4,
                    ]

    assert has_reshape_pattern(reshape_scheduled)


def test_reshape_pattern_zero_extent():
    @Ts.function
    def transpose_zero(
        rxplaceholder: T.Tensor((3, 0, 4), "float32"),
        T_transpose: T.Tensor((0, 3, 4), "float32"),
    ):
        for i0, i1, i2 in T.grid(0, 3, 4):
            with Ts.sblock("T_transpose"):
                ax0, ax1, ax2 = Ts.axis.remap("SSS", [i0, i1, i2])
                Ts.reads(rxplaceholder[ax1, ax0, ax2])
                Ts.writes(T_transpose[ax0, ax1, ax2])
                T_transpose[ax0, ax1, ax2] = rxplaceholder[ax1, ax0, ax2]

    assert not has_reshape_pattern(transpose_zero)


def test_reshape_pattern_expand_dims():
    @Ts.function
    def expand_dims(
        rxplaceholder: T.Tensor((2, 3, 4), "float32"),
        expand_dims: T.Tensor((2, 1, 1, 1, 3, 1, 4, 1), "float32"),
    ):
        T.func_attr({"tirx.noalias": True})
        for i0, i1, i2, i3, i4, i5, i6, i7 in T.grid(2, 1, 1, 1, 3, 1, 4, 1):
            with Ts.sblock("expand_dims"):
                i0_1, i1_1, i2_1, i3_1, i4_1, i5_1, i6_1, i7_1 = Ts.axis.remap(
                    "SSSSSSSS", [i0, i1, i2, i3, i4, i5, i6, i7]
                )
                Ts.reads(rxplaceholder[i0_1, i4_1, i6_1])
                Ts.writes(expand_dims[i0_1, i1_1, i2_1, i3_1, i4_1, i5_1, i6_1, i7_1])
                expand_dims[i0_1, i1_1, i2_1, i3_1, i4_1, i5_1, i6_1, i7_1] = rxplaceholder[
                    i0_1, i4_1, i6_1
                ]

    assert has_reshape_pattern(expand_dims)


def test_reshape_pattern_dyn_1():
    n = T.dynamic("n", "int64")

    @Ts.function
    def reshape(
        A: T.Tensor((n, T.int64(32), T.int64(128)), "float16"),
        T_reshape: T.Tensor((T.int64(1), n, T.int64(32), T.int64(128)), "float16"),
    ):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), n, T.int64(32), T.int64(128)):
            with Ts.sblock("T_reshape"):
                v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                Ts.reads(
                    A[
                        ((v_ax3 // T.int64(128) + v_ax2) // T.int64(32) + v_ax0 * n + v_ax1) % n,
                        (v_ax3 // T.int64(128) + v_ax2) % T.int64(32),
                        v_ax3 % T.int64(128),
                    ]
                )
                Ts.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = A[
                    ((v_ax3 // T.int64(128) + v_ax2) // T.int64(32) + v_ax0 * n + v_ax1) % n,
                    (v_ax3 // T.int64(128) + v_ax2) % T.int64(32),
                    v_ax3 % T.int64(128),
                ]

    assert has_reshape_pattern(reshape)


def test_reshape_pattern_dyn_2():
    n = T.dynamic("n", "int64")

    @Ts.function
    def reshape(A: T.Tensor((T.int64(1), n), "int32"), T_reshape: T.Tensor((n,), "int32")):
        for ax0 in range(n):
            with Ts.sblock("T_reshape"):
                v_ax0 = Ts.axis.spatial(n, ax0)
                Ts.reads(A[T.int64(0), v_ax0 % n])
                Ts.writes(T_reshape[v_ax0])
                T_reshape[v_ax0] = A[T.int64(0), v_ax0 % n]

    assert has_reshape_pattern(reshape)


def test_reshape_pattern_dyn_3():
    n = T.dynamic("n", "int64")

    @Ts.function
    def reshape(
        A: T.Tensor((n, T.int64(4096)), "float16"),
        T_reshape: T.Tensor((T.int64(1), n, T.int64(4096)), "float16"),
    ):
        T.func_attr({"op_pattern": 8, "tirx.noalias": True})

        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
            with Ts.sblock("T_reshape"):
                v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                Ts.reads(A[(v_ax2 // T.int64(4096) + v_ax0 * n + v_ax1) % n, v_ax2 % T.int64(4096)])
                Ts.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = A[
                    (v_ax2 // T.int64(4096) + v_ax0 * n + v_ax1) % n, v_ax2 % T.int64(4096)
                ]

    assert has_reshape_pattern(reshape)


def test_reshape_pattern_dyn_4():
    n = T.dynamic("n", "int64")

    @Ts.function
    def reshape(
        A: T.Tensor((T.int64(1), n, T.int64(4096)), "float16"),
        T_reshape: T.Tensor((T.int64(1), n, T.int64(32), T.int64(128)), "float16"),
    ):
        T.func_attr({"op_pattern": 8, "tirx.noalias": True})

        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), n, T.int64(32), T.int64(128)):
            with Ts.sblock("T_reshape"):
                v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                Ts.reads(
                    A[
                        T.int64(0),
                        ((v_ax2 * T.int64(128) + v_ax3) // T.int64(4096) + v_ax0 * n + v_ax1) % n,
                        (v_ax2 * T.int64(128) + v_ax3) % T.int64(4096),
                    ]
                )
                Ts.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = A[
                    T.int64(0),
                    ((v_ax2 * T.int64(128) + v_ax3) // T.int64(4096) + v_ax0 * n + v_ax1) % n,
                    (v_ax2 * T.int64(128) + v_ax3) % T.int64(4096),
                ]

    assert has_reshape_pattern(reshape)


def test_reshape_pattern_dyn_5():
    n = T.dynamic("n", "int64")

    @Ts.function
    def reshape(
        A: T.Tensor((T.int64(1), n, T.int64(32), T.int64(128)), "float16"),
        T_reshape: T.Tensor((T.int64(1), n, T.int64(4096)), "float16"),
    ):
        T.func_attr({"op_pattern": 8, "tirx.noalias": True})

        # with Ts.sblock("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
            with Ts.sblock("T_reshape"):
                v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                Ts.reads(
                    A[
                        T.int64(0),
                        (v_ax2 // T.int64(4096) + v_ax0 * n + v_ax1) % n,
                        v_ax2 % T.int64(4096) // T.int64(128),
                        v_ax2 % T.int64(128),
                    ]
                )
                Ts.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = A[
                    T.int64(0),
                    (v_ax2 // T.int64(4096) + v_ax0 * n + v_ax1) % n,
                    v_ax2 % T.int64(4096) // T.int64(128),
                    v_ax2 % T.int64(128),
                ]

    assert has_reshape_pattern(reshape)


def test_reshape_pattern_with_raggedness():
    @Ts.function
    def reshape_raggedness(
        A: T.Tensor((100, 768), "float32"),
        src_indptr: T.Tensor((9,), "int32"),
        B: T.Tensor((100, 12, 64), "float32"),
    ):
        for b in T.serial(8):
            with Ts.sblock("block0"):
                vb = Ts.axis.spatial(8, b)
                for i in T.serial(src_indptr[vb + 1] - src_indptr[vb]):
                    for h in T.serial(12):
                        for f in T.serial(64):
                            with Ts.sblock("block1"):
                                vi, vh, vf = Ts.axis.remap("SSS", [i, h, f])
                                B[src_indptr[vb] + vi, vh, vf] = A[
                                    src_indptr[vb] + vi, vh * 64 + vf
                                ]

    assert has_reshape_pattern(reshape_raggedness)


def test_reshape_pattern_reject_seqstmt():
    @Ts.function
    def identity_bias(A: T.Tensor((4, 4), "float32"), B: T.Tensor((4, 4), "float32")):
        C = Ts.sblock_alloc_tensor((128, 128), "float32")
        for i0, i1 in T.grid(4, 4):
            with Ts.sblock("identity"):
                vi0, vi1 = Ts.axis.remap("SS", [i0, i1])
                C[vi0, vi1] = A[vi0, vi1]
        for i0, i1 in T.grid(4, 4):
            with Ts.sblock("identity"):
                vi0, vi1 = Ts.axis.remap("SS", [i0, i1])
                B[vi0, vi1] = C[vi0, vi1] + T.float32(1)

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

    assert not has_reshape_pattern(identity_bias)
    assert not has_reshape_pattern(identity_identity)


def test_reshape_pattern_reject_reduction():
    @Ts.function
    def reduction(A: T.Tensor((4, 4), "float32"), B: T.Tensor((4,), "float32")):
        for i0, i1 in T.grid(4, 4):
            with Ts.sblock("identity"):
                vi0, vi1 = Ts.axis.remap("SR", [i0, i1])
                with Ts.init():
                    B[vi0] = T.float32(0)
                B[vi0] = B[vi0] + A[vi0, vi1]

    assert not has_reshape_pattern(reduction)


def test_reshape_pattern_reject_reduction():
    @Ts.function
    def reduction(A: T.Tensor((4, 4), "float32"), B: T.Tensor((4,), "float32")):
        for i0, i1 in T.grid(4, 4):
            with Ts.sblock("identity"):
                vi0, vi1 = Ts.axis.remap("SR", [i0, i1])
                with Ts.init():
                    B[vi0] = T.float32(0)
                B[vi0] = B[vi0] + A[vi0, vi1]

    assert not has_reshape_pattern(reduction)


if __name__ == "__main__":
    tvm_tirx.testing.main()
