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

import tvm_tirx.script
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm.relax.testing.runtime_builtin import MakeShapeCode, MatchShapeCode
from tvm.s_tir import script as Ts
from tvm_tirx.ir import assert_structural_equal
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T

# note: we expected RemovePurityChecking to be run first, so we force purity in most test cases


def test_const_shape_arg():
    MS = MatchShapeCode

    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main(x: R.Shape([1, 2]), y: R.Shape):
            R.func_attr({"relax.force_pure": True})
            return x

        @Ts.function
        def extra_func(H: T.Tensor(T.int64(4), "int64")):
            """Extra function, checks if the pass preserves it."""
            H[T.int64(1)] = H[T.int64(0)] + T.int64(1)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Shape([1, 2]), y: R.Shape):
            R.func_attr({"relax.force_pure": True})
            shape_heap = R.null_value()
            _ = R.call_packed("vm.builtin.check_shape_info", x, 2, "", ty_args=[R.Tuple()])
            _ = R.call_packed("vm.builtin.check_shape_info", y, -1, "", ty_args=[R.Tuple()])
            _ = R.call_packed(
                "vm.builtin.match_shape",
                x,
                shape_heap,
                2,
                MS.ASSERT_EQUAL_TO_IMM,
                1,
                MS.ASSERT_EQUAL_TO_IMM,
                2,
                "",
                ty_args=[R.Tuple()],
            )
            return x

        @Ts.function
        def extra_func(H: T.Tensor(T.int64(4), "int64")):
            H[T.int64(1)] = H[T.int64(0)] + T.int64(1)

    before = Before
    expected = Expected
    after = relax.transform.VMShapeLower(emit_err_ctx=False)(before)
    assert_structural_equal(after, expected)


def test_static_fn_check():
    """Check static shape and function."""
    MS = MatchShapeCode

    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main(f: R.Callable([R.Any], R.Any), y: R.Shape([1, 2])):
            R.func_attr({"relax.force_pure": True})
            return y

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(f: R.Callable([R.Any], R.Any), y: R.Shape([1, 2])):
            R.func_attr({"relax.force_pure": True})
            shape_heap = R.null_value()
            _ = R.call_packed("vm.builtin.check_func_info", f, "", ty_args=[R.Tuple()])
            _ = R.call_packed("vm.builtin.check_shape_info", y, 2, "", ty_args=[R.Tuple()])
            _ = R.call_packed(
                "vm.builtin.match_shape",
                y,
                shape_heap,
                2,
                MS.ASSERT_EQUAL_TO_IMM,
                1,
                MS.ASSERT_EQUAL_TO_IMM,
                2,
                "",
                ty_args=[R.Tuple()],
            )
            return y

    before = Before
    expected = Expected
    after = relax.transform.VMShapeLower(emit_err_ctx=False)(before)
    assert_structural_equal(after, expected)


def test_simple_symbolic_shape():
    MS = MatchShapeCode

    n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int64")

    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor([n, 2, m], "float32")):
            R.func_attr({"relax.force_pure": True})
            return x

    sindex = {
        "n": 0,
        "m": 1,
    }

    n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor([n, 2, m], "float32")):
            R.func_attr({"relax.force_pure": True})
            shape_heap = R.call_builtin_with_ctx(
                "vm.builtin.alloc_shape_heap",
                [R.prim_value(2)],
                ty_args=[R.Tensor(ndim=1, dtype="int64")],
            )
            _ = R.call_packed(
                "vm.builtin.check_tensor_info",
                x,
                3,
                R.dtype("float32"),
                "",
                ty_args=[R.Tuple()],
            )
            _ = R.call_packed(
                "vm.builtin.match_shape",
                x,
                shape_heap,
                3,
                MS.STORE_TO_HEAP,
                sindex["n"],
                MS.ASSERT_EQUAL_TO_IMM,
                2,
                MS.STORE_TO_HEAP,
                sindex["m"],
                "",
                ty_args=[R.Tuple()],
            )
            return x

    before = Before
    expected = Expected
    after = relax.transform.VMShapeLower(emit_err_ctx=False)(before)
    assert_structural_equal(after, expected)


def test_symbolic_compute():
    MS = MatchShapeCode
    MK = MakeShapeCode

    m = T.dynamic("m", "int64")
    k = T.dynamic("k", "int64")
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor([n, m], "float32"), y: R.Tensor(ndim=3, dtype=None)) -> R.Shape(
            ndim=3
        ):
            R.func_attr({"relax.force_pure": True})
            z = R.match_cast(y, R.Tensor([k, m, k + 1], dtype=None))
            return R.shape([k + 1, m, 2])

    # slot assignment:
    # 0: n, 1: m, 2:k, 3: k+1
    sindex = {"n": 0, "m": 1, "k": 2, "k+1": 3}

    m = T.dynamic("m", "int64")
    k = T.dynamic("k", "int64")
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function(private=True)
        def shape_func(H: T.Tensor(T.int64(4), "int64")):
            # generated compute function
            T.func_attr({"tirx.is_host_func": True})
            H[T.int64(sindex["k+1"])] = H[T.int64(sindex["k"])] + T.int64(1)

        @R.function
        def main(x: R.Tensor([n, m], "float32"), y: R.Tensor(ndim=3, dtype=None)) -> R.Shape(
            ndim=3
        ):
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            shape_heap = R.call_builtin_with_ctx(
                "vm.builtin.alloc_shape_heap",
                [R.prim_value(4)],
                ty_args=[R.Tensor(ndim=1, dtype="int64")],
            )
            _ = R.call_packed(
                "vm.builtin.check_tensor_info",
                x,
                2,
                R.dtype("float32"),
                "",
                ty_args=[R.Tuple()],
            )
            gv = R.null_value()
            _ = R.call_packed("vm.builtin.check_tensor_info", y, 3, gv, "", ty_args=[R.Tuple()])
            _ = R.call_packed(
                "vm.builtin.match_shape",
                x,
                shape_heap,
                2,
                MS.STORE_TO_HEAP,
                sindex["n"],
                MS.STORE_TO_HEAP,
                sindex["m"],
                "",
                ty_args=[R.Tuple()],
            )
            _ = R.call_packed(
                "vm.builtin.match_shape",
                y,
                shape_heap,
                3,
                MS.STORE_TO_HEAP,
                sindex["k"],
                MS.ASSERT_EQUAL_TO_LOAD,
                sindex["m"],
                MS.NO_OP,
                0,
                "",
                ty_args=[R.Tuple()],
            )
            _ = R.call_tir_packed(cls.shape_func, (shape_heap,))
            # extra assertion on y's shape after shape computation
            _ = R.call_packed(
                "vm.builtin.match_shape",
                y,
                shape_heap,
                3,
                MS.ASSERT_EQUAL_TO_LOAD,
                sindex["k"],
                MS.ASSERT_EQUAL_TO_LOAD,
                sindex["m"],
                MS.ASSERT_EQUAL_TO_LOAD,
                sindex["k+1"],
                "",
                ty_args=[R.Tuple()],
            )
            z = R.match_cast(y, R.Tensor([k, m, k + 1], dtype=None))
            # construct shape value for return
            s = R.call_packed(
                "vm.builtin.make_shape",
                shape_heap,
                3,
                MK.LOAD_SHAPE,
                sindex["k+1"],
                MK.LOAD_SHAPE,
                sindex["m"],
                MK.USE_IMM,
                2,
                ty_args=[R.Shape(ndim=3)],
            )
            return s

    before = Before
    expected = Expected
    after = relax.transform.VMShapeLower(emit_err_ctx=False)(before)
    assert_structural_equal(after, expected)


def test_tuple_handling():
    MS = MatchShapeCode

    n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int64")
    k = T.dynamic("k", "int64")

    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tuple(R.Tensor([n, m], "float32"), R.Tuple(R.Shape, R.Tensor([n, k], "int32"))),
        ):
            R.func_attr({"relax.force_pure": True})
            return x

    # slot assignment:
    sindex = {"n": 0, "m": 1, "k": 2}

    n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int64")
    k = T.dynamic("k", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tuple(R.Tensor([n, m], "float32"), R.Tuple(R.Shape, R.Tensor([n, k], "int32"))),
        ):
            R.func_attr({"relax.force_pure": True})
            shape_heap = R.call_builtin_with_ctx(
                "vm.builtin.alloc_shape_heap",
                [R.prim_value(3)],
                ty_args=[R.Tensor(ndim=1, dtype="int64")],
            )
            # recursively unpack tuple for static info check
            _ = R.call_packed("vm.builtin.check_tuple_info", x, 2, "", ty_args=[R.Tuple()])
            t0 = x[0]
            _ = R.call_packed(
                "vm.builtin.check_tensor_info",
                t0,
                2,
                R.dtype("float32"),
                "",
                ty_args=[R.Tuple()],
            )
            t1 = x[1]
            _ = R.call_packed("vm.builtin.check_tuple_info", t1, 2, "", ty_args=[R.Tuple()])
            t1x0 = t1[0]
            _ = R.call_packed("vm.builtin.check_shape_info", t1x0, -1, "", ty_args=[R.Tuple()])
            t1x1 = t1[1]
            _ = R.call_packed(
                "vm.builtin.check_tensor_info",
                t1x1,
                2,
                R.dtype("int32"),
                "",
                ty_args=[R.Tuple()],
            )
            # match shape checks.
            _ = R.call_packed(
                "vm.builtin.match_shape",
                t0,
                shape_heap,
                2,
                MS.STORE_TO_HEAP,
                sindex["n"],
                MS.STORE_TO_HEAP,
                sindex["m"],
                "",
                ty_args=[R.Tuple()],
            )
            _ = R.call_packed(
                "vm.builtin.match_shape",
                t1x1,
                shape_heap,
                2,
                MS.ASSERT_EQUAL_TO_LOAD,
                sindex["n"],
                MS.STORE_TO_HEAP,
                sindex["k"],
                "",
                ty_args=[R.Tuple()],
            )
            return x

    before = Before
    expected = Expected
    after = relax.transform.VMShapeLower(emit_err_ctx=False)(before)
    assert_structural_equal(after, expected)


def test_return_match_check():
    """Test when return body is not same as ret_ty, runtime match check needed."""
    MS = MatchShapeCode

    n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int64")

    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor([n, m], "float32"), y: R.Any) -> R.Tuple(R.Tensor([n, m], "float32")):
            R.func_attr({"relax.force_pure": True})
            return y

    # slot assignment:
    sindex = {
        "n": 0,
        "m": 1,
    }

    n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor([n, m], "float32"), y: R.Any) -> R.Tuple(R.Tensor([n, m], "float32")):
            R.func_attr({"relax.force_pure": True})
            shape_heap = R.call_builtin_with_ctx(
                "vm.builtin.alloc_shape_heap",
                [R.prim_value(2)],
                ty_args=[R.Tensor(ndim=1, dtype="int64")],
            )
            _ = R.call_packed(
                "vm.builtin.check_tensor_info", x, 2, R.dtype("float32"), "", ty_args=[R.Tuple()]
            )
            _ = R.call_packed(
                "vm.builtin.match_shape",
                x,
                shape_heap,
                2,
                MS.STORE_TO_HEAP,
                sindex["n"],
                MS.STORE_TO_HEAP,
                sindex["m"],
                "",
                ty_args=[R.Tuple()],
            )
            _ = R.call_packed("vm.builtin.check_tuple_info", y, 1, "", ty_args=[R.Tuple()])
            # emit runtime function call since y do not have the right type.
            y1 = R.call_packed("vm.builtin.tuple_getitem", y, 0, ty_args=[R.Any])
            # run check
            _ = R.call_packed(
                "vm.builtin.check_tensor_info",
                y1,
                2,
                R.dtype("float32"),
                "",
                ty_args=[R.Tuple()],
            )
            # shape check
            _ = R.call_packed(
                "vm.builtin.match_shape",
                y1,
                shape_heap,
                2,
                MS.ASSERT_EQUAL_TO_LOAD,
                sindex["n"],
                MS.ASSERT_EQUAL_TO_LOAD,
                sindex["m"],
                "",
                ty_args=[R.Tuple()],
            )

            return y

    before = Before
    expected = Expected
    after = relax.transform.VMShapeLower(emit_err_ctx=False)(before)
    assert_structural_equal(after, expected)


def test_return_match_check_with_new_expr():
    """Like test_return_match_check, but requires a computation

    When return body is not same as ret_ty, a runtime match
    check is required.  This match check may require a symbolic
    expression to be computed.
    """
    MS = MatchShapeCode

    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor([n, n], "float32")) -> R.Tensor([n * n], "float32"):
            R.func_attr({"relax.force_pure": True})
            out = R.call_packed("flatten_matrix", x, ty_args=R.Any)
            return out

    # slot assignment:
    sindex = {
        "n": 0,
        "n * n": 1,
    }

    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor([n, n], "float32")) -> R.Tensor([n * n], "float32"):
            R.func_attr({"relax.force_pure": True})
            shape_heap = R.call_builtin_with_ctx(
                "vm.builtin.alloc_shape_heap",
                [R.prim_value(2)],
                ty_args=[R.Tensor(ndim=1, dtype="int64")],
            )
            _ = R.call_packed(
                "vm.builtin.check_tensor_info", x, 2, R.dtype("float32"), "", ty_args=[R.Tuple()]
            )
            _ = R.call_packed(
                "vm.builtin.match_shape",
                x,
                shape_heap,
                2,
                MS.STORE_TO_HEAP,
                sindex["n"],
                MS.ASSERT_EQUAL_TO_LOAD,
                sindex["n"],
                "",
                ty_args=[R.Tuple()],
            )

            _ = R.call_tir_packed(Expected.shape_func, (shape_heap,))

            out = R.call_packed("flatten_matrix", x, ty_args=R.Any)
            _ = R.call_packed(
                "vm.builtin.check_tensor_info",
                out,
                1,
                R.dtype("float32"),
                "",
                ty_args=[R.Tuple()],
            )
            _ = R.call_packed(
                "vm.builtin.match_shape",
                out,
                shape_heap,
                1,
                MS.ASSERT_EQUAL_TO_LOAD,
                sindex["n * n"],
                "",
                ty_args=[R.Tuple()],
            )
            return out

        @Ts.function(private=True)
        def shape_func(H: T.Tensor(T.int64(2), "int64")):
            # generated compute function
            T.func_attr({"tirx.is_host_func": True})
            H[T.int64(sindex["n * n"])] = H[T.int64(sindex["n"])] * H[T.int64(sindex["n"])]

    before = Before
    expected = Expected
    after = relax.transform.VMShapeLower(emit_err_ctx=False)(before)
    assert_structural_equal(after, expected)


def test_symbolic_shape_multiple_function():
    MS = MatchShapeCode
    MK = MakeShapeCode

    m_fn1 = T.dynamic("m", "int64")
    n_fn1 = T.dynamic("n", "int64")
    n_fn2 = T.dynamic("n", "int64")
    m_fn2 = T.dynamic("m", "int64")

    @I.ir_module
    class Before:
        @R.function
        def fn1(A: R.Tensor((m_fn1, n_fn1), dtype="float32")):
            R.func_attr({"relax.force_pure": True})
            return A

        @R.function
        def fn2(A: R.Tensor((n_fn2, m_fn2), dtype="float32")):
            R.func_attr({"relax.force_pure": True})
            return A

    # slot assignment:
    sindex_fn1 = {
        "m": 0,
        "n": 1,
    }
    sindex_fn2 = {
        "n": 0,
        "m": 1,
    }

    m_fn1 = T.dynamic("m", "int64")
    n_fn1 = T.dynamic("n", "int64")
    n_fn2 = T.dynamic("n", "int64")
    m_fn2 = T.dynamic("m", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def fn1(A: R.Tensor((m_fn1, n_fn1), dtype="float32")) -> R.Tensor(
            (m_fn1, n_fn1), dtype="float32"
        ):
            R.func_attr({"relax.force_pure": True})
            shape_heap: R.Tensor(dtype="int64", ndim=1) = R.call_builtin_with_ctx(
                "vm.builtin.alloc_shape_heap",
                (R.prim_value(2),),
                ty_args=(R.Tensor(dtype="int64", ndim=1),),
            )
            _: R.Tuple = R.call_packed(
                "vm.builtin.check_tensor_info",
                A,
                R.prim_value(2),
                R.dtype("float32"),
                R.str(""),
                ty_args=(R.Tuple,),
            )
            _1: R.Tuple = R.call_packed(
                "vm.builtin.match_shape",
                A,
                shape_heap,
                R.prim_value(2),
                MS.STORE_TO_HEAP,
                sindex_fn1["m"],
                MS.STORE_TO_HEAP,
                sindex_fn1["n"],
                R.str(""),
                ty_args=(R.Tuple,),
            )
            return A

        @R.function
        def fn2(A: R.Tensor((n_fn2, m_fn2), dtype="float32")) -> R.Tensor(
            (n_fn2, m_fn2), dtype="float32"
        ):
            R.func_attr({"relax.force_pure": True})
            shape_heap: R.Tensor(dtype="int64", ndim=1) = R.call_builtin_with_ctx(
                "vm.builtin.alloc_shape_heap",
                (R.prim_value(2),),
                ty_args=(R.Tensor(dtype="int64", ndim=1),),
            )
            _2: R.Tuple = R.call_packed(
                "vm.builtin.check_tensor_info",
                A,
                R.prim_value(2),
                R.dtype("float32"),
                R.str(""),
                ty_args=(R.Tuple,),
            )
            _3: R.Tuple = R.call_packed(
                "vm.builtin.match_shape",
                A,
                shape_heap,
                R.prim_value(2),
                MS.STORE_TO_HEAP,
                sindex_fn2["n"],
                MS.STORE_TO_HEAP,
                sindex_fn2["m"],
                R.str(""),
                ty_args=(R.Tuple,),
            )
            return A

    before = Before
    expected = Expected
    after = relax.transform.VMShapeLower(emit_err_ctx=False)(before)
    assert_structural_equal(after, expected)


def test_check_lifted_weights():
    MS = MatchShapeCode

    @I.ir_module
    class Before:
        @R.function
        def main_transform_params(params: R.Tuple(R.Tensor((16, 16), dtype="float32"))) -> R.Tuple(
            R.Tensor((16, 16), dtype="float32")
        ):
            R.func_attr({"relax.force_pure": True})
            return params

        @R.function
        def main(x: R.Tensor((16, 16), "float32"), param_0: R.Tensor((16, 16), dtype="float32")):
            R.func_attr({"relax.force_pure": True, "num_input": 1})
            return (x, param_0)

    @I.ir_module
    class Expected:
        @R.function
        def main_transform_params(params: R.Tuple(R.Tensor((16, 16), dtype="float32"))) -> R.Tuple(
            R.Tensor((16, 16), dtype="float32")
        ):
            R.func_attr({"relax.force_pure": True})
            shape_heap: R.Any = R.null_value()
            _: R.Tuple = R.call_packed(
                "vm.builtin.check_tuple_info",
                params,
                R.prim_value(1),
                R.str(""),
                ty_args=(R.Tuple,),
            )
            gv: R.Tensor((16, 16), dtype="float32") = params[0]
            _1: R.Tuple = R.call_packed(
                "vm.builtin.check_tensor_info",
                gv,
                R.prim_value(2),
                R.dtype("float32"),
                R.str(""),
                ty_args=(R.Tuple,),
            )
            _2: R.Tuple = R.call_packed(
                "vm.builtin.match_shape",
                gv,
                shape_heap,
                R.prim_value(2),
                MS.ASSERT_EQUAL_TO_IMM,
                R.prim_value(16),
                MS.ASSERT_EQUAL_TO_IMM,
                R.prim_value(16),
                R.str(""),
                ty_args=(R.Tuple,),
            )
            return params

        @R.function
        def main(
            x: R.Tensor((16, 16), dtype="float32"), param_0: R.Tensor((16, 16), dtype="float32")
        ) -> R.Tuple(R.Tensor((16, 16), dtype="float32"), R.Tensor((16, 16), dtype="float32")):
            R.func_attr({"num_input": 1, "relax.force_pure": True})
            shape_heap: R.Any = R.null_value()
            _: R.Tuple = R.call_packed(
                "vm.builtin.check_tensor_info",
                x,
                R.prim_value(2),
                R.dtype("float32"),
                R.str(""),
                ty_args=(R.Tuple,),
            )
            _1: R.Tuple = R.call_packed(
                "vm.builtin.match_shape",
                x,
                shape_heap,
                R.prim_value(2),
                MS.ASSERT_EQUAL_TO_IMM,
                R.prim_value(16),
                MS.ASSERT_EQUAL_TO_IMM,
                R.prim_value(16),
                R.str(""),
                ty_args=(R.Tuple,),
            )
            return (x, param_0)

    before = Before
    after = relax.transform.VMShapeLower(emit_err_ctx=False)(before)
    expected = Expected
    assert_structural_equal(after, expected)


def test_check_weights_with_dynamic_shape():
    MS = MatchShapeCode

    n = T.dynamic("n", "int64")

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((16, 16), "float32"),
            params: R.Tuple(R.Tensor((16, 16), dtype="float32"), R.Tensor((n,), "float32")),
        ):
            R.func_attr({"relax.force_pure": True, "num_input": 1})
            param_0 = params[0]
            param_1 = params[1]
            return (x, param_0, param_1)

    n = T.dynamic("n", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((16, 16), "float32"),
            params: R.Tuple(R.Tensor((16, 16), dtype="float32"), R.Tensor((n,), "float32")),
        ):
            R.func_attr({"num_input": 1, "relax.force_pure": True})
            shape_heap: R.Tensor(dtype="int64", ndim=1) = R.call_builtin_with_ctx(
                "vm.builtin.alloc_shape_heap",
                (R.prim_value(1),),
                ty_args=(R.Tensor(dtype="int64", ndim=1),),
            )
            _: R.Tuple = R.call_packed(
                "vm.builtin.check_tensor_info",
                x,
                R.prim_value(2),
                R.dtype("float32"),
                R.str(""),
                ty_args=(R.Tuple,),
            )
            _1: R.Tuple = R.call_packed(
                "vm.builtin.check_tuple_info",
                params,
                R.prim_value(2),
                R.str(""),
                ty_args=(R.Tuple,),
            )
            _param_1: R.Tensor((n,), dtype="float32") = params[1]
            _2: R.Tuple = R.call_packed(
                "vm.builtin.check_tensor_info",
                _param_1,
                R.prim_value(1),
                R.dtype("float32"),
                R.str(""),
                ty_args=(R.Tuple,),
            )
            _3: R.Tuple = R.call_packed(
                "vm.builtin.match_shape",
                x,
                shape_heap,
                R.prim_value(2),
                MS.ASSERT_EQUAL_TO_IMM,
                R.prim_value(16),
                MS.ASSERT_EQUAL_TO_IMM,
                R.prim_value(16),
                R.str(""),
                ty_args=(R.Tuple,),
            )
            _4: R.Tuple = R.call_packed(
                "vm.builtin.match_shape",
                _param_1,
                shape_heap,
                MS.STORE_TO_HEAP,
                R.prim_value(1),
                R.prim_value(0),
                R.str(""),
                ty_args=(R.Tuple,),
            )

            param_0 = params[0]
            param_1 = params[1]
            return (x, param_0, param_1)

    before = Before
    after = relax.transform.VMShapeLower(emit_err_ctx=False)(before)
    print(after)
    expected = Expected
    assert_structural_equal(after, expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
