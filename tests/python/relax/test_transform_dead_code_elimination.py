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

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.relax.transform import DeadCodeElimination
from tvm.s_tir import script as Ts
from tvm_tirx.script.parser import ir as I
from tvm_tirx.script.parser import relax as R
from tvm_tirx.script.parser import tirx as T


def verify(input, expected):
    tvm_tirx.ir.assert_structural_equal(DeadCodeElimination()(input), expected)


def test_simple():
    @tvm_tirx.script.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"),
            w: R.Tensor((4, 3, 3, 3), dtype="float32"),
            bias: R.Tensor((26, 26), dtype="float32"),
        ):
            # block 0
            with R.dataflow():
                gv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
                gv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                gv2: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    gv,
                    gv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                gv21: R.Tensor((2, 4, 26, 26), dtype="float32") = R.permute_dims(
                    gv2, axes=[0, 3, 1, 2]
                )
                gv22: R.Tensor((2, 4, 26, 26), dtype="float32") = R.add(gv21, bias)
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"),
            w: R.Tensor((4, 3, 3, 3), dtype="float32"),
            bias: R.Tensor((26, 26), dtype="float32"),
        ):
            with R.dataflow():
                gv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
                gv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                gv2: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    gv,
                    gv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                R.output(gv2)
            return gv2

    verify(Input, Expected)


def test_2block():
    @tvm_tirx.script.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"),
            w: R.Tensor((4, 3, 3, 3), dtype="float32"),
            bias: R.Tensor((26, 26), dtype="float32"),
        ):
            # block 0
            with R.dataflow():
                gv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
                gv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                gv2: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    gv,
                    gv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                gv21: R.Tensor((2, 4, 26, 26), dtype="float32") = R.permute_dims(
                    gv2, axes=[0, 3, 1, 2]
                )
                gv22: R.Tensor((2, 4, 26, 26), dtype="float32") = R.add(gv21, bias)
                R.output(gv2)
            gv3 = R.astype(gv2, dtype="float16")
            return gv3

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"),
            w: R.Tensor((4, 3, 3, 3), dtype="float32"),
            bias: R.Tensor((26, 26), dtype="float32"),
        ):
            with R.dataflow():
                gv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
                gv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                gv2: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    gv,
                    gv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                R.output(gv2)
            gv3: R.Tensor((2, 26, 26, 4), dtype="float16") = R.astype(gv2, dtype="float16")
            return gv3

    verify(Input, Expected)


def check_if_func_exists(mod, func_name):
    gvs = [gv.name_hint for gv in mod.get_global_vars()]
    return func_name in gvs


def test_unused_relax_func():
    @tvm_tirx.script.ir_module
    class InputModule:
        @Ts.function
        def tir_add(
            x: T.Tensor((16, 16), "float32"),
            y: T.Tensor((16, 16), "float32"),
            z: T.Tensor((16, 16), "float32"),
        ) -> None:
            for i, j in T.grid(16, 16):
                with Ts.sblock("add"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    z[vi, vj] = x[vi, vj] + y[vi, vj]

        @R.function(private=True)
        def unused_func(x: R.Tensor((16, 16), "float32"), w: R.Tensor((16, 16), "float32")):
            gv0 = R.add(x, w)
            return gv0

        @R.function
        def main(x: R.Tensor((16, 16), "float32"), w: R.Tensor((16, 16), "float32")) -> R.Tensor(
            (16, 16), "float32"
        ):
            gv0 = R.call_tir(
                InputModule.tir_add, (x, w), ty_args=[R.Tensor((16, 16), dtype="float32")]
            )
            return gv0

    mod = InputModule
    assert mod
    new_mod = DeadCodeElimination()(mod)
    assert check_if_func_exists(new_mod, "main")
    assert check_if_func_exists(new_mod, "tir_add")
    assert not check_if_func_exists(new_mod, "unused_func")


@pytest.fixture(params=[True, False], scope="session")
def provide_entry_func_name(request):
    return request.param


def test_unused_relax_func_custom_entry_func(provide_entry_func_name):
    @tvm_tirx.script.ir_module
    class InputModule:
        @Ts.function(private=True)
        def tir_add(
            x: T.Tensor((16, 16), "float32"),
            y: T.Tensor((16, 16), "float32"),
            z: T.Tensor((16, 16), "float32"),
        ) -> None:
            for i, j in T.grid(16, 16):
                with Ts.sblock("add"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    z[vi, vj] = x[vi, vj] + y[vi, vj]

        @R.function(private=True)
        def unused_func(x: R.Tensor((16, 16), "float32"), w: R.Tensor((16, 16), "float32")):
            gv0 = R.add(x, w)
            return gv0

        @R.function
        def foo(x: R.Tensor((16, 16), "float32"), w: R.Tensor((16, 16), "float32")) -> R.Tensor(
            (16, 16), "float32"
        ):
            gv0 = R.call_tir(
                InputModule.tir_add, (x, w), ty_args=[R.Tensor((16, 16), dtype="float32")]
            )
            return gv0

    mod = InputModule
    assert mod

    if provide_entry_func_name:
        entry_functions = ["foo"]
    else:
        entry_functions = None

    # Test entry function other than "main".
    new_mod = DeadCodeElimination(entry_functions=entry_functions)(mod)
    assert check_if_func_exists(new_mod, "foo")
    assert check_if_func_exists(new_mod, "tir_add")
    assert not check_if_func_exists(new_mod, "unused_func")


def test_tracking_through_externally_exposed_func(provide_entry_func_name):
    @tvm_tirx.script.ir_module
    class InputModule:
        @Ts.function(private=True)
        def tir_add(
            x: T.Tensor((16, 16), "float32"),
            y: T.Tensor((16, 16), "float32"),
            z: T.Tensor((16, 16), "float32"),
        ) -> None:
            for i, j in T.grid(16, 16):
                with Ts.sblock("add"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    z[vi, vj] = x[vi, vj] + y[vi, vj]

        @R.function(private=True)
        def unused_func(x: R.Tensor((16, 16), "float32"), w: R.Tensor((16, 16), "float32")):
            gv0 = R.add(x, w)
            return gv0

        @R.function
        def foo(x: R.Tensor((16, 16), "float32"), w: R.Tensor((16, 16), "float32")) -> R.Tensor(
            (16, 16), "float32"
        ):
            gv0 = R.call_tir(
                InputModule.tir_add, (x, w), ty_args=[R.Tensor((16, 16), dtype="float32")]
            )
            return gv0

        @R.function
        def main(x: R.Tensor((16, 16), "float32")) -> R.Tensor((16, 16), "float32"):
            return x

    mod = InputModule
    assert mod

    # Test tracking of usage through externally-exposed function
    new_mod = DeadCodeElimination(entry_functions=["main"])(mod)
    assert check_if_func_exists(new_mod, "main")
    assert check_if_func_exists(new_mod, "foo")
    assert check_if_func_exists(new_mod, "tir_add")
    assert not check_if_func_exists(new_mod, "unused_func")


def test_unused_relax_func_symbolic_shape():
    # Test with relax function w/ symbolic shape.
    m_tir_matmul = T.dynamic("m", "int64")
    n_tir_matmul = T.dynamic("n", "int64")
    k_tir_matmul = T.dynamic("k", "int64")
    m_unused_func = T.dynamic("m", "int64")
    n_unused_func = T.dynamic("n", "int64")
    k_unused_func = T.dynamic("k", "int64")
    m_main = T.dynamic("m", "int64")
    k_main = T.dynamic("k", "int64")
    n_main = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module(check_well_formed=False)
    class InputModule:
        @Ts.function
        def tir_matmul(
            x: T.Tensor((m_tir_matmul, n_tir_matmul), "float32"),
            y: T.Tensor((n_tir_matmul, k_tir_matmul), "float32"),
            z: T.Tensor((m_tir_matmul, k_tir_matmul), "float32"),
        ) -> None:
            for i, j, k_tir_matmul_index in T.grid(m_tir_matmul, k_tir_matmul, n_tir_matmul):
                with Ts.sblock("matmul"):
                    vi, vj, vk = Ts.axis.remap("SSR", [i, j, k_tir_matmul_index])
                    with Ts.init():
                        z[vi, vj] = 0.0
                    z[vi, vj] = z[vi, vj] + x[vi, vk] * y[vk, vj]

        @R.function(private=True)
        def unused_func(
            x: R.Tensor((m_unused_func, n_unused_func), "float32"),
            w: R.Tensor((n_unused_func, k_unused_func), "float32"),
        ):
            gv0 = R.add(x, w)
            return gv0

        @R.function
        def main(
            x: R.Tensor((m_main, n_main), "float32"), w: R.Tensor((n_main, k_main), "float32")
        ):
            gv0 = R.call_tir(
                InputModule.tir_matmul,
                (x, w),
                ty_args=[R.Tensor((m_main, k_main), dtype="float32")],
            )
            return gv0

    mod = InputModule
    assert mod

    new_mod = DeadCodeElimination()(mod)
    assert check_if_func_exists(new_mod, "main")
    assert check_if_func_exists(new_mod, "tir_matmul")
    assert not check_if_func_exists(new_mod, "unused_func")


def test_unused_function():
    @tvm_tirx.script.ir_module
    class InputModule:
        @Ts.function
        def unused_func(
            x: T.Tensor((16, 16), "float32"),
            y: T.Tensor((16, 16), "float32"),
            z: T.Tensor((16, 16), "float32"),
        ) -> None:
            T.func_attr({"global_symbol": "tir_unused"})
            for i, j in T.grid(16, 16):
                with Ts.sblock("add"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    z[vi, vj] = x[vi, vj] + y[vi, vj]

        @R.function
        def relax_add(x: R.Tensor((16, 16), "float32"), w: R.Tensor((16, 16), "float32")):
            gv0 = R.add(x, w)
            return gv0

        @R.function
        def main(x: R.Tensor((16, 16), "float32"), w: R.Tensor((16, 16), "float32")) -> R.Tensor(
            (16, 16), "float32"
        ):
            gv0 = InputModule.relax_add(x, w)
            return gv0

    mod = InputModule
    assert mod
    new_mod = DeadCodeElimination()(mod)
    assert check_if_func_exists(new_mod, "main")
    assert check_if_func_exists(new_mod, "relax_add")
    # RemoveUnusedFunction pass won't remove the function with global symbol for the external linkage.
    assert check_if_func_exists(new_mod, "unused_func")


def test_preserve_indirectly_used_function():
    @tvm_tirx.script.ir_module
    class InputModule:
        @R.function
        def main(x: R.Tensor((16, 16), "float32"), w: R.Tensor((16, 16), "float32")) -> R.Tensor(
            (16, 16), "float32"
        ):
            gv0 = R.call_tir(
                InputModule.tir_add_tensors,
                [x, w],
                ty_args=[R.Tensor((16, 16), "float32")],
            )
            return gv0

        @Ts.function(private=True)
        def tir_add_tensors(
            x: T.Tensor((16, 16), "float32"),
            y: T.Tensor((16, 16), "float32"),
            z: T.Tensor((16, 16), "float32"),
        ):
            for i, j in T.grid(16, 16):
                with Ts.sblock("add"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    z[vi, vj] = InputModule.tir_add_float32(x[vi, vj], y[vi, vj])

        @Ts.function(private=True)
        def tir_add_float32(x: T.float32, y: T.float32) -> T.float32:
            return x + y

    mod = InputModule
    assert mod
    new_mod = DeadCodeElimination()(mod)

    tvm_tirx.ir.assert_structural_equal(mod, new_mod)


def test_multiple_unused_funcs():
    @tvm_tirx.script.ir_module
    class InputModule:
        @Ts.function
        def unused_func1(
            x: T.Tensor((16, 16), "float32"),
            y: T.Tensor((16, 16), "float32"),
            z: T.Tensor((16, 16), "float32"),
        ) -> None:
            T.func_attr({"global_symbol": "tir_unused"})
            for i, j in T.grid(16, 16):
                with Ts.sblock("add"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    z[vi, vj] = x[vi, vj] + y[vi, vj]

        @R.function(private=True)
        def unused_func2(x: R.Tensor((16, 16), "float32"), w: R.Tensor((16, 16), "float32")):
            gv0 = R.add(x, w)
            return gv0

        @R.function
        def main(x: R.Tensor((16, 16), "float32"), w: R.Tensor((16, 16), "float32")) -> R.Tensor(
            (16, 16), "float32"
        ):
            gv0 = R.add(x, w)
            return gv0

    mod = InputModule
    assert mod

    new_mod = DeadCodeElimination()(mod)
    assert check_if_func_exists(new_mod, "main")
    # RemoveUnusedFunction pass won't remove the function with global symbol for the external linkage.
    assert check_if_func_exists(new_mod, "unused_func1")
    assert not check_if_func_exists(new_mod, "unused_func2")


def test_unused_dfb():
    # test if an unused dataflow block can be removed.
    @tvm_tirx.script.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"),
            w: R.Tensor((4, 3, 3, 3), dtype="float32"),
        ):
            # block 0
            with R.dataflow():
                lv0: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(
                    x, axes=[0, 2, 3, 1]
                )
                lv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                lv2: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv0,
                    lv1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                )
                lv3: R.Tensor((2, 4, 26, 26), dtype="float32") = R.permute_dims(
                    lv2, axes=[0, 3, 1, 2]
                )
                R.output(lv2)
            gv3 = R.astype(lv2, dtype="float16")
            # dead block
            with R.dataflow():
                lv4: R.Tensor((2, 4, 26, 26), dtype="float16") = R.permute_dims(
                    gv3, axes=[0, 3, 1, 2]
                )
                R.output(lv4)
            return gv3

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"),
            w: R.Tensor((4, 3, 3, 3), dtype="float32"),
        ):
            # block 0
            with R.dataflow():
                lv0: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(
                    x, axes=[0, 2, 3, 1]
                )
                lv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                lv2: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv0,
                    lv1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                )
                R.output(lv2)
            gv3 = R.astype(lv2, dtype="float16")
            return gv3

    verify(Input, Expected)


def test_unused_dfb2():
    # test if an unused dataflow block can be removed.
    @tvm_tirx.script.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"),
            w: R.Tensor((4, 3, 3, 3), dtype="float32"),
        ):
            # dead block
            with R.dataflow():
                lv0: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(
                    x, axes=[0, 2, 3, 1]
                )
                R.output(lv0)

            gv_x = R.astype(x, dtype="float16")
            gv_w = R.astype(w, dtype="float16")

            with R.dataflow():
                lv1: R.Tensor((2, 28, 28, 3), dtype="float16") = R.permute_dims(
                    gv_x, axes=[0, 2, 3, 1]
                )
                lv2: R.Tensor((4, 3, 3, 3), dtype="float16") = R.permute_dims(
                    gv_w, axes=[0, 2, 3, 1]
                )
                lv3: R.Tensor((2, 26, 26, 4), dtype="float16") = R.nn.conv2d(
                    lv1,
                    lv2,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                )
                # dead instruction -> usee lv1 also dead.
                lv4: R.Tensor((2, 3, 28, 28), dtype="float32") = R.permute_dims(
                    lv0, axes=[0, 3, 1, 2]
                )
                R.output(lv3)
            return lv3

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"),
            w: R.Tensor((4, 3, 3, 3), dtype="float32"),
        ):
            gv_x = R.astype(x, dtype="float16")
            gv_w = R.astype(w, dtype="float16")

            with R.dataflow():
                lv1: R.Tensor((2, 28, 28, 3), dtype="float16") = R.permute_dims(
                    gv_x, axes=[0, 2, 3, 1]
                )
                lv2: R.Tensor((4, 3, 3, 3), dtype="float16") = R.permute_dims(
                    gv_w, axes=[0, 2, 3, 1]
                )
                lv3: R.Tensor((2, 26, 26, 4), dtype="float16") = R.nn.conv2d(
                    lv1,
                    lv2,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                )
                R.output(lv3)
            return lv3

    verify(Input, Expected)


def test_extern_func():
    """DeadCodeElimination should retain the ExternFunc in the IRModule."""

    builder = tvm.relax.BlockBuilder()
    builder.add_func(tvm.relax.extern("extern_func"), "extern_func")
    before = builder.get()

    verify(before, before)


def test_recursively_defined_lambda():
    """DCE may be applied to recursively-defined functions

    While most expressions may only contain references to
    previously-defined variables, local Relax function definitions may
    contain references to themselves.

    This is a regression test.  In previous implementations, the
    recursive use of `while_loop` resulted in an error, as
    `while_loop` was not considered in-scope by the `CollectVarUsage`
    utility until after the body of `while_loop` had been visited.

    """

    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor:
            @R.function
            def while_loop(i: R.Tensor((), "int32"), s: R.Tensor((2, 3), "float32")) -> R.Tensor(
                (2, 3), "float32"
            ):
                cond = R.call_pure_packed(
                    "test.vm.less", i, R.const(10), ty_args=R.Tensor((), dtype="bool")
                )
                c = R.const(1, dtype="int32")
                if cond:
                    new_i = R.add(i, c)
                    new_s = R.add(s, x)
                    r = while_loop(new_i, new_s)
                else:
                    r = s
                return r

            gv = while_loop(R.const(0), x)
            return gv

    Expected = Before

    verify(Before, Expected)


def test_recursively_defined_closure():
    """DCE may be applied to recursively-defined closures

    This test is identical to `test_recursively_defined_lambda`,
    except that the threshold for recursion is defined in an enclosed
    variable outside of the recursive function.

    """

    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor:
            threshold = R.const(10)

            @R.function
            def while_loop(i: R.Tensor((), "int32"), s: R.Tensor((2, 3), "float32")) -> R.Tensor(
                (2, 3), "float32"
            ):
                cond = R.call_pure_packed(
                    "test.vm.less", i, threshold, ty_args=R.Tensor((), dtype="bool")
                )
                c = R.const(1, dtype="int32")
                if cond:
                    new_i = R.add(i, c)
                    new_s = R.add(s, x)
                    r = while_loop(new_i, new_s)
                else:
                    r = s
                return r

            gv = while_loop(R.const(0), x)
            return gv

    Expected = Before

    verify(Before, Expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
