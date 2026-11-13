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
"""Block builder unit test"""

# The test here do not depend on tvmscript to cover most basic features
import pytest
import tvm_ffi

import tvm
import tvm.contrib.cblas
import tvm_tirx
import tvm_tirx.testing
from tvm import relax as rx
from tvm import te, topi
from tvm.relax import ExternFunc
from tvm.relax import script as R
from tvm_tirx import tirx
from tvm_tirx.ir.base import assert_structural_equal
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx.function import Function


@pytest.fixture(scope="module")
def register_nop():
    @tvm_ffi.register_global_func("test.blockbuilder.nop")
    def nop():
        pass


def test_block_builder():
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")
    x = rx.Var("x", rx.TensorType([m, n], "float16"))
    y = rx.Var("y", rx.TensorType([n], "float16"))
    bb = rx.BlockBuilder()

    bb._begin_binding_block()
    gv0 = bb.emit(rx.op.add(x, y))
    bb._begin_dataflow_block()
    lv0 = bb.emit(rx.op.multiply(gv0, y))
    gv1 = bb.emit_output(rx.op.multiply(lv0, lv0))
    b0 = bb._end_block()
    bb._begin_dataflow_block()
    lv1 = bb.emit(rx.op.multiply(gv0, y))
    gv2 = bb.emit_output(rx.op.multiply(lv1, lv1))
    b1 = bb._end_block()
    gv3 = bb.emit(rx.op.add(x, y))
    b2 = bb._end_block()

    assert isinstance(b0, rx.DataflowBlock)
    assert isinstance(b1, rx.DataflowBlock)
    assert not isinstance(b2, rx.DataflowBlock)


def test_emit_with_name():
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")
    x = rx.Var("x", rx.TensorType([m, n], "float16"))
    y = rx.Var("y", rx.TensorType([n], "float16"))
    bb = rx.BlockBuilder()

    bb._begin_dataflow_block()
    lv0 = bb.emit(rx.op.add(x, y), "add")
    gv0 = bb.emit_output(rx.op.multiply(lv0, y), "multi")
    b0 = bb._end_block()

    assert b0.bindings[0].var.name == "add"
    assert b0.bindings[1].var.name == "multi"


def test_function_single_block():
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")
    x = rx.Var("x", rx.TensorType([m, n], "float16"))
    y = rx.Var("y", rx.TensorType([n], "float16"))
    bb = rx.BlockBuilder()

    with bb.function("func", [x, y]):
        with bb.dataflow():
            lv0 = bb.emit(rx.op.add(x, y))
            assert lv0.name == "lv"
            lv1 = bb.emit(rx.op.multiply(lv0, y))
            assert lv1.name == "lv1"
            gv0 = bb.emit_output(lv1)
        assert gv0.name == "gv"
        bb.emit_func_output(gv0)

    func = bb.finalize()["func"]
    assert func.params[0] == x
    assert func.params[1] == y
    assert func.body.body == gv0
    assert_structural_equal(gv0.ty, rx.TensorType([m, n], "float16"))
    assert len(func.body.blocks) == 1
    assert len(func.body.blocks[0].bindings) == 3


def test_function_multi_blocks():
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")
    x = rx.Var("x", rx.TensorType([m, n], "float16"))
    y = rx.Var("y", rx.TensorType([n], "float16"))
    bb = rx.BlockBuilder()

    with bb.function("func", [x, y]):
        with bb.dataflow():
            lv0 = bb.emit(rx.op.add(x, y))
            assert lv0.name == "lv"
            gv0 = bb.emit_output(lv0)
        assert gv0.name == "gv"
        gv1 = bb.emit(rx.op.add(gv0, gv0))
        assert gv1.name == "gv1"
        with bb.dataflow():
            lv1 = bb.emit(rx.op.add(gv1, gv1))
            assert lv1.name == "lv1"
            gv2 = bb.emit_output(gv1)
        bb.emit_func_output(gv2)

    func = bb.finalize()["func"]

    assert_structural_equal(gv2.ty, rx.TensorType([m, n], "float16"))
    assert func.params[0] == x
    assert func.params[1] == y
    assert func.body.body == gv2
    assert len(func.body.blocks) == 3
    assert len(func.body.blocks[0].bindings) == 2
    assert len(func.body.blocks[1].bindings) == 1
    assert len(func.body.blocks[2].bindings) == 2


def test_multi_functions():
    bb = rx.BlockBuilder()

    m_1 = T.dynamic("m", "int64")
    n_1 = T.dynamic("n", "int64")
    x_1 = rx.Var("x", rx.TensorType([m_1, n_1], "float16"))
    y_1 = rx.Var("y", rx.TensorType([n_1], "float16"))

    with bb.function("func1", [x_1, y_1]):
        with bb.dataflow():
            lv0 = bb.emit(rx.op.add(x_1, y_1))
            assert lv0.name == "lv"
            gv0 = bb.emit_output(lv0)
        bb.emit_func_output(gv0)

    m_2 = tirx.Var("m", "int64")
    n_2 = tirx.Var("n", "int64")
    x_2 = rx.Var("x", rx.TensorType([m_2, n_2], "float16"))
    y_2 = rx.Var("y", rx.TensorType([n_2], "float16"))

    with bb.function("func2", [x_2, y_2]):
        with bb.dataflow():
            lv0 = bb.emit(rx.op.add(y_2, x_2))
            # TODO(@yuchen): enable block builder to reset local var unique name map
            assert lv0.name == "lv1"
            gv0 = bb.emit_output(lv0)
        bb.emit_func_output(gv0)

    mod = bb.finalize()
    func1 = mod["func1"]
    assert func1.params[0] == x_1
    assert func1.params[1] == y_1
    assert len(func1.body.blocks) == 1
    func2 = mod["func2"]
    assert func2.params[0] == x_2
    assert func2.params[1] == y_2
    assert len(func2.body.blocks) == 1


def test_binary_shape_type_deduction():
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")
    k = tirx.Var("k", "int64")
    x = rx.Var("x", rx.TensorType([m, 1], "float16"))
    y = rx.Var("y", rx.TensorType([n], "float16"))
    z = rx.Var("z", rx.TensorType([5], "float16"))
    w = rx.Var("w", rx.TensorType([k], "float16"))
    bb = rx.BlockBuilder()

    with bb.function("func", [x, y, z, w]):
        with bb.dataflow():
            lv0 = bb.emit(rx.op.add(x, y))
            assert_structural_equal(lv0.ty, rx.TensorType([m, n], "float16"))

            lv1 = bb.emit(rx.op.multiply(x, z))
            assert_structural_equal(lv1.ty, rx.TensorType([m, 5], "float16"))

            lv2 = bb.emit(rx.op.multiply(z, w))
            assert isinstance(lv2.ty, rx.TensorType)
            assert lv2.ty.ndim == 1
            assert lv2.ty.dtype == "float16"

            lv3 = bb.emit(rx.op.multiply(y, w))
            assert isinstance(lv3.ty, rx.TensorType)
            assert lv3.ty.ndim == 1
            assert lv3.ty.dtype == "float16"

            gv0 = bb.emit_output(lv3)
        bb.emit_func_output(gv0)

        assert isinstance(gv0.ty, rx.TensorType)
        assert gv0.ty.ndim == 1
        assert gv0.ty.dtype == "float16"


def test_emit_match_cast():
    m = T.dynamic("m", dtype="int64")
    n = T.dynamic("n", dtype="int64")
    x = rx.Var("tensor_value", rx.TensorType(dtype="float32", ndim=-1))
    y = rx.Var("shape_value", rx.ShapeType([16, 8]))
    bb = rx.BlockBuilder()

    with bb.function("func", [x, y]):
        with bb.dataflow():
            # lv0: Tensor((m, n), "float32") =
            #   match_cast(x: Tensor(_, "float32"], [m, n))
            lv0 = bb.match_cast(x, rx.TensorType([m, n], "float32"))
            assert isinstance(lv0, rx.DataflowVar)
            assert_structural_equal(lv0.ty, rx.TensorType([m, n], "float32"))

            # lv1: Shape = match_cast(shape, rx.ShapeType([m, n]))
            lv1 = bb.match_cast(y, rx.ShapeType([m, n]), "var_name")
            assert lv1.ty == rx.ShapeType([m, n])
            gv0 = bb.emit_output(lv1)

        bb.emit_func_output(gv0)
    func = bb.finalize()["func"]
    block = func.body.blocks[0]
    b0, b1 = block.bindings[:2]
    assert isinstance(b0, rx.MatchCast)
    assert isinstance(b1, rx.MatchCast)

    assert b0.value == x
    assert b0.ty == rx.TensorType([m, n], "float32")
    assert b0.var == lv0

    assert b1.value == y
    assert b1.ty == rx.ShapeType([m, n])
    assert b1.var == lv1
    assert b1.var.name == "var_name"


def test_emit_match_cast_binding_in_dataflow_block():
    bb = rx.BlockBuilder()

    x = rx.Var("x", rx.TensorType(dtype="float32", ndim=-1))
    m = T.dynamic("m", dtype="int64")
    gv = rx.Var("gv", rx.TensorType(dtype="float32", ndim=-1))
    match_cast = rx.MatchCast(gv, x, rx.TensorType((m,), "float32"))

    with bb.function("main", [x]):
        with bb.dataflow():
            bb.emit_normalized(match_cast)
            bb.emit_output(gv)
        bb.emit_func_output(x)

    func = bb.finalize()["main"]
    block = func.body.blocks[0]
    b0 = block.bindings[0]
    assert isinstance(b0, rx.MatchCast)

    assert b0.value == x
    assert isinstance(b0.ty, rx.TensorType)
    assert b0.ty.shape[0] == m
    assert b0.var == gv


def test_normalize():
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    x = rx.Var("x", rx.TensorType([m, n], "float16"))
    y = rx.Var("y", rx.TensorType([n], "float16"))
    bb = rx.BlockBuilder()

    # Call node
    add_call = rx.op.multiply(x, y)

    bb.normalize(add_call)
    shape = rx.get_shape_of(add_call)

    assert isinstance(shape, rx.ShapeExpr)
    assert shape[0] == m
    assert shape[1] == n

    # Tuple node
    tuple_1 = rx.Tuple([x, y])
    bb.normalize(tuple_1)
    assert isinstance(tuple_1.ty, rx.TupleType)
    assert isinstance(tuple_1.ty.fields[0], rx.TensorType)
    assert isinstance(tuple_1.ty.fields[1], rx.TensorType)

    # Nested Tuple
    tuple_2 = rx.Tuple([x, rx.Tuple([x, y])])
    bb.normalize(tuple_2)

    assert isinstance(tuple_2.ty, rx.TupleType)
    assert isinstance(tuple_2.ty.fields[0], rx.TensorType)
    assert isinstance(tuple_2.ty.fields[1], rx.TupleType)
    assert isinstance(tuple_2.ty.fields[1].fields[0], rx.TensorType)
    assert isinstance(tuple_2.ty.fields[1].fields[1], rx.TensorType)


def test_tuple_indexing():
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    shape_x = rx.TensorType([m, n], "float16")
    shape_y = rx.TensorType([n], "float16")
    relax_tuple = rx.Var("relax_tuple", rx.TupleType([shape_x, shape_y]))

    assert isinstance(relax_tuple.ty, rx.TupleType)
    assert isinstance(relax_tuple.ty.fields[0], rx.TensorType)
    assert isinstance(relax_tuple.ty.fields[1], rx.TensorType)

    # TupleGetItem will initialize type from the
    # TupleType, if present.
    x = relax_tuple[0]
    tvm_tirx.ir.assert_structural_equal(x.ty, shape_x)

    y = relax_tuple[1]
    tvm_tirx.ir.assert_structural_equal(y.ty, shape_y)

    with pytest.raises(IndexError, match="Index out of bounds"):
        relax_tuple[2]

    # Tuple unpacking produces TupleGetItem structs
    x_unpack, y_unpack = relax_tuple
    tvm_tirx.ir.assert_structural_equal(x, x_unpack)
    tvm_tirx.ir.assert_structural_equal(y, y_unpack)

    # When TupleType is available, tuple unpacking fails immediately
    # for incorrect number of arguments.
    with pytest.raises(ValueError):
        x_unpack, y_unpack, z_unpack = relax_tuple


def test_call_te():
    bb = rx.BlockBuilder()
    n, m = tirx.Var("n", "int64"), tirx.Var("m", "int64")
    x = rx.Var("x", rx.TensorType([n, m], "float32"))
    y = rx.Var("y", rx.TensorType([n, m], "float32"))
    z = rx.Var("z", rx.TensorType([n, m], "float32"))

    def te_func(args, args_dict, msg):
        A, B = args
        C = args_dict["C"]
        D = te.compute((128, 128), lambda i, j: A[i, j] + B[i, j])
        E = te.compute((128, 128), lambda i, j: D[i, j] - C[i, j])
        return E

    with bb.function("rx_func", [x, y, z]):
        with bb.dataflow():
            out = bb.emit_output(bb.call_te(te_func, [x, y], {"C": z}, msg="hello"))
        bb.emit_func_output(out)

    mod = bb.finalize()
    rx_func = mod["rx_func"]

    assert rx_func.params[0] == x
    assert rx_func.params[1] == y
    assert rx_func.params[2] == z
    assert rx_func.body.body == out
    assert len(rx_func.body.blocks) == 1
    assert len(rx_func.body.blocks[0].bindings) == 1


def test_call_te_unique_tensor_name():
    bb = rx.BlockBuilder()
    x = rx.Var("x", R.Tensor((2, 3), "float32"))
    y = rx.Var("y", R.Tensor((3, 4), "float32"))
    with bb.function("main", [x, y]):
        gv = bb.emit_te(topi.nn.matmul, x, y)
        bb.emit_func_output(gv)

    f_matmul = bb.finalize()["matmul"]
    param_A = f_matmul.params[0]
    param_B = f_matmul.params[1]
    assert param_A.name != param_B.name
    assert tvm_tirx.tirx.is_tensor_var(param_A)
    assert tvm_tirx.tirx.is_tensor_var(param_B)
    assert not param_A.same_as(param_B)


def test_call_te_with_unsupported_shape_arg():
    bb = rx.BlockBuilder()
    x = rx.Var("x", rx.TensorType((200,), "float32"))
    s = rx.Var("s", rx.ShapeType((200,)))

    with pytest.raises(AssertionError):
        with bb.function("rx_func", [x]):
            out = bb.emit(bb.call_te(topi.reshape, x, s))
            bb.emit_func_output(out)


def test_emit_te():
    bb = rx.BlockBuilder()
    n, m = tirx.Var("n", "int64"), tirx.Var("m", "int64")
    x = rx.Var("x", rx.TensorType([n, m], "float32"))
    y = rx.Var("y", rx.TensorType([n, m], "float32"))
    z = rx.Var("z", rx.TensorType([n, m], "float32"))

    def te_func(args, args_dict, msg):
        A, B = args
        C = args_dict["C"]
        D = te.compute((128, 128), lambda i, j: A[i, j] + B[i, j])
        E = te.compute((128, 128), lambda i, j: D[i, j] - C[i, j])
        return E

    with bb.function("rx_func", [x, y, z]):
        out = bb.emit_te(te_func, [x, y], {"C": z}, msg="hello")
        bb.emit_func_output(out)

    mod = bb.finalize()
    rx_func = mod["rx_func"]

    def get_tir_func():
        A = te.placeholder((n, m), dtype="float32", name="A")
        B = te.placeholder((n, m), dtype="float32", name="B")
        C = te.placeholder((n, m), dtype="float32", name="C")
        out = te_func((A, B), {"C": C}, "")
        return tvm.te.create_function([A, B, C, out], index_dtype_override="int64")

    # check TIR structure matches expected
    assert_structural_equal(mod["te_func"].body, get_tir_func().body)

    # check Relax function calls TIR function with call_tir call
    assert rx_func.params[0] == x
    assert rx_func.params[1] == y
    assert rx_func.params[2] == z
    assert rx_func.body.body == out
    assert len(rx_func.body.blocks) == 1
    assert len(rx_func.body.blocks[0].bindings) == 1

    call_node = rx_func.body.blocks[0].bindings[0].value
    assert isinstance(call_node, rx.Call)
    assert len(call_node.args) == 2
    assert call_node.args[0].name_hint == "te_func"
    assert call_node.args[1][0] == x
    assert call_node.args[1][1] == y
    assert call_node.args[1][2] == z


def test_emit_te_multiple():
    bb = rx.BlockBuilder()
    n, m = tirx.Var("n", "int64"), tirx.Var("m", "int64")
    x = rx.Var("x", rx.TensorType([n, m], "float32"))
    y = rx.Var("y", rx.TensorType([n, m], "float32"))
    z = rx.Var("z", rx.TensorType([128, m], "float32"))

    def te_func(A):
        B = te.compute((128, 128), lambda i, j: A[i, j] + 1)
        return B

    with bb.function("rx_func", [x, y, z]):
        x1 = bb.emit_te(te_func, x)
        y1 = bb.emit_te(te_func, y)
        z1 = bb.emit_te(te_func, z)
        bb.emit_func_output(z1)

    mod = bb.finalize()
    rx_func = mod["rx_func"]

    function = []
    for gv in mod.get_global_vars():
        if isinstance(mod[gv], Function):
            function.append(mod[gv])

    # only two Functions were generated since two of them are equal so got deduped
    assert len(function) == 2
    assert rx_func.body.blocks[0].bindings[0].value.args[0].name_hint == "te_func"
    assert rx_func.body.blocks[0].bindings[1].value.args[0].name_hint == "te_func"
    assert rx_func.body.blocks[0].bindings[2].value.args[0].name_hint == "te_func1"


def test_emit_te_multiple_output():
    bb = rx.BlockBuilder()
    n, m = tirx.Var("n", "int64"), tirx.Var("m", "int64")
    x = rx.Var("x", rx.TensorType([n, m], "float32"))

    def te_func(A):
        B0, B1 = te.compute((n, m), lambda i, j: (A[i, j] + 1, A[i, j] * 2), name="B")
        return (B0, B1)

    with bb.function("rx_func", [x]):
        y = bb.emit_te(te_func, x)
        z = rx.TupleGetItem(y, 0)
        bb.emit_func_output([y, z])

    rx_func = bb.finalize()["rx_func"]

    # check call tirx output shape is a Tuple of ShapeExpr
    assert rx_func.params[0] == x
    call_node = rx_func.body.blocks[0].bindings[0].value
    assert call_node.args[0].name_hint == "te_func"
    assert isinstance(call_node.ty_args[0], rx.TupleType)
    assert len(call_node.ty_args[0].fields) == 2
    assert isinstance(call_node.ty_args[0].fields[0].shape, rx.ShapeExpr)
    assert isinstance(call_node.ty_args[0].fields[1].shape, rx.ShapeExpr)


def test_emit_te_extern():
    bb = rx.BlockBuilder()
    n, m = tirx.Var("n", "int64"), tirx.Var("m", "int64")
    x = rx.Var("x", rx.TensorType([n, m], "float32"))
    y = rx.Var("y", rx.TensorType([m, n], "float32"))

    with bb.function("rx_cblas_matmul", [x, y]):
        out = bb.emit_te(tvm.contrib.cblas.matmul, x, y, transa=False, transb=False)
        bb.emit_func_output(out)

    mod = bb.finalize()
    rx_func = mod["rx_cblas_matmul"]

    # check Relax function calls TIR function with call_tir call
    assert rx_func.params[0] == x
    assert rx_func.params[1] == y
    assert len(rx_func.body.blocks) == 1
    call_node = rx_func.body.blocks[0].bindings[0].value
    assert isinstance(call_node, rx.Call)
    assert len(call_node.args) == 2
    assert call_node.args[0].name_hint == "matmul"
    assert call_node.args[1][0] == x
    assert call_node.args[1][1] == y
    assert call_node.ty_args[0].shape[0] == n
    assert call_node.ty_args[0].shape[1] == n


def test_emit_te_prim_value():
    bb = rx.BlockBuilder()
    n, m = tirx.Var("n", "int64"), tirx.Var("m", "int64")
    x = rx.Var("x", R.Tensor([n, m], "float32"))
    a_min = tvm_tirx.ir.IntImm("int64", 0)
    a_max = tvm_tirx.ir.IntImm("int64", 6)

    with bb.function("rx_clip", [x]):
        out = bb.emit_te(topi.clip, x, a_min, a_max)
        bb.emit_func_output(out)

    rx_func = bb.finalize()["rx_clip"]

    # check Relax function calls TIR function with call_tir call
    assert rx_func.params[0] == x
    assert len(rx_func.body.blocks) == 1
    call_node = rx_func.body.blocks[0].bindings[0].value
    assert isinstance(call_node, rx.Call)
    assert len(call_node.args) == 2
    assert call_node.args[1][0] == x


def test_nested_function_fail():
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")
    x = rx.Var("x", rx.TensorType([m, n], "float16"))
    y = rx.Var("y", rx.TensorType([n], "float16"))
    bb = rx.BlockBuilder()

    with pytest.raises(RuntimeError):
        with bb.function("func", [x, y]):
            gv0 = bb.emit(rx.op.add(x, x))
            with bb.function("func1", [x, y]):
                gv1 = bb.emit(rx.op.add(x, x))
            bb.emit_func_output(gv0)


def test_emit_func_output_twice_fail():
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")
    x = rx.Var("x", rx.TensorType([m, n], "float16"))
    y = rx.Var("y", rx.TensorType([n], "float16"))
    bb = rx.BlockBuilder()

    with pytest.raises(RuntimeError):
        with bb.function("func", [x, y]):
            gv0 = bb.emit(rx.op.add(x, y))
            bb.emit_func_output(gv0)
            bb.emit_func_output(gv0)


def test_func_params_twice_fail():
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")
    x = rx.Var("x", rx.TensorType([m, n], "float16"))
    y = rx.Var("y", rx.TensorType([n], "float16"))
    bb = rx.BlockBuilder()

    with pytest.raises(RuntimeError):
        with bb.function("func", [x, y]):
            gv0 = bb.emit(rx.op.add(x, y))
            bb.emit_func_output(gv0, [x])


def test_no_func_params_fail():
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")
    x = rx.Var("x", rx.TensorType([m, n], "float16"))
    y = rx.Var("y", rx.TensorType([n], "float16"))
    bb = rx.BlockBuilder()

    with pytest.raises(RuntimeError):
        with bb.function("func"):
            gv0 = bb.emit(rx.Call(ExternFunc("test.blockbuilder.nop"), []))
            bb.emit_func_output(gv0)


def test_block_builder_scope_recovery():
    bb = rx.BlockBuilder()

    n, m = tirx.Var("n", "int64"), tirx.Var("m", "int64")
    x = rx.Var("x", rx.TensorType([n, m], "float32"))
    y = rx.Var("y", rx.TensorType([m, n], "float32"))

    with pytest.raises(RuntimeError):
        # this line fails
        with bb.function("func", [x, y]):
            gv0 = bb.emit(rx.op.add(x, y))

    # current should be recovered
    assert rx.BlockBuilder.current() is None

    # second attempt to do it correctly.
    with bb.function("func", [x, y]):
        gv0 = bb.emit(rx.op.add(x, y))
        bb.emit_func_output(gv0)


@pytest.mark.parametrize("emit_nested_tuple", [True, False])
def test_emit_nested_tuple(emit_nested_tuple):
    """Convert nested tuples when emitting relax"""

    def make_function(emit_nested_tuple: bool):
        bb = rx.BlockBuilder()

        n_sym = tirx.Var("n", "int64")
        m_sym = tirx.Var("m", "int64")
        n = rx.Var("n", tvm_tirx.ir.PrimType("int64"))
        m = rx.Var("m", tvm_tirx.ir.PrimType("int64"))
        x = rx.Var("x", rx.TensorType([n_sym, m_sym], "float32"))
        y = rx.Var("y", rx.TensorType([m_sym, n_sym], "float32"))

        with bb.function("func", [n, m, x, y]):
            scalars = (n, m)
            if not emit_nested_tuple:
                scalars = bb.emit(scalars)
            output = (scalars, x, y)
            bb.emit_func_output(output)

        return bb.finalize()["func"]

    def make_expected(emit_nested_tuple: bool):
        if emit_nested_tuple:
            n = T.dynamic("n", "int64")
            m = T.dynamic("m", "int64")

            @R.function
            def func(
                n_1: T.int64,
                m_1: T.int64,
                x: R.Tensor((n, m), dtype="float32"),
                y: R.Tensor((m, n), dtype="float32"),
            ):
                return ((n_1, m_1), x, y)

        else:
            n = T.dynamic("n", "int64")
            m = T.dynamic("m", "int64")

            @R.function
            def func(
                n_1: T.int64,
                m_1: T.int64,
                x: R.Tensor((n, m), dtype="float32"),
                y: R.Tensor((m, n), dtype="float32"),
            ):
                gv = n_1, m_1
                return (gv, x, y)

        return func

    expected = make_expected(emit_nested_tuple)
    actual = make_function(emit_nested_tuple)

    tvm_tirx.ir.assert_structural_equal(expected, actual)


@pytest.mark.skip_well_formed_check_before_transform
def test_finalize_public_private_name_conflict():
    # tirx call
    bb = rx.BlockBuilder()

    def te_zero():
        return topi.full((), "int64", tvm_tirx.ir.IntImm("int64", 0))

    def te_one():
        return topi.full((), "int64", tvm_tirx.ir.IntImm("int64", 1))

    with bb.function("func", []):
        gv0 = bb.emit_te(te_zero, function_name_hint="func")
        gv1 = bb.emit_te(te_one, function_name_hint="func")
        bb.emit_func_output((gv0, gv1))

    mod = bb.get()
    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(mod)
    mod_final = bb.finalize()
    rx.analysis.check_well_formed(mod_final)

    # relax function call
    bb = rx.BlockBuilder()

    with bb.function("func", [], private=True):
        gvar = bb.emit_func_output(rx.const(0, "int64"))

    with bb.function("func", [], private=True):
        gv0 = bb.emit(rx.Call(gvar, []))
        gvar1 = bb.emit_func_output(gv0)

    with bb.function("func", []):
        gv0 = bb.emit(rx.Call(gvar1, []))
        bb.emit_func_output(gv0)

    mod = bb.get()
    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(mod)
    mod_final = bb.finalize()
    rx.analysis.check_well_formed(mod_final)


def test_emit_nested_seqexpr_in_binding_block():
    """May emit a SeqExpr inside a BindingBlock"""

    bb = rx.BlockBuilder()

    with bb.function("func", []):
        lhs = bb.emit(rx.const(1, "int64"), "a")
        rhs = bb.emit(rx.const(2, "int64"), "b")
        out = bb.emit(rx.op.add(lhs, rhs), "c")
        bb.emit_func_output(out)

    seq_expr = bb.finalize()["func"].body

    bb = rx.BlockBuilder()
    with bb.function("func", [], private=True):
        lhs = bb.emit(rx.const(3, "int64"), "d")
        rhs = bb.emit(seq_expr, "e")
        out = bb.emit(rx.op.add(lhs, rhs), "f")
        bb.emit_func_output(out)

    output = bb.finalize()["func"]

    @R.function(private=True)
    def expected():
        d = R.const(3, "int64")
        a = R.const(1, "int64")
        b = R.const(2, "int64")
        c = R.add(a, b)
        e = c
        f = R.add(d, e)
        return f

    tvm_tirx.ir.assert_structural_equal(expected, output)


def test_emit_nested_dataflow_seqexpr_in_dataflow_block():
    """May emit a SeqExpr with dataflow inside a DataflowBlock"""
    bb = rx.BlockBuilder()

    with bb.function("func", []):
        with bb.dataflow():
            lhs = bb.emit(rx.const(1, "int64"), "a")
            rhs = bb.emit(rx.const(2, "int64"), "b")
            out = bb.emit_output(rx.op.add(lhs, rhs), "c")
        bb.emit_func_output(out)

    seq_expr = bb.finalize()["func"].body

    bb = rx.BlockBuilder()
    with bb.function("func", [], private=True):
        with bb.dataflow():
            lhs = bb.emit(rx.const(3, "int64"), "d")
            rhs = bb.emit(seq_expr, "e")
            out = bb.emit_output(rx.op.add(lhs, rhs), "f")
        bb.emit_func_output(out)

    output = bb.finalize()["func"]

    @R.function(private=True)
    def expected():
        with R.dataflow():
            d = R.const(3, "int64")
            a = R.const(1, "int64")
            b = R.const(2, "int64")
            c = R.add(a, b)
            e = c
            f = R.add(d, e)
            R.output(c, f)
        return f

    tvm_tirx.ir.assert_structural_equal(expected, output)


def test_emit_ill_formed_nested_seqexpr_in_dataflow_block():
    """May emit a SeqExpr inside a DataflowBlock

    This produces ill-formed code, but cannot be caught at the
    normalizer.  See also
    test_emit_well_formed_nested_seqexpr_in_dataflow_block.

    """
    bb = rx.BlockBuilder()

    with bb.function("func", []):
        lhs = bb.emit(rx.const(1, "int64"), "a")
        rhs = bb.emit(rx.const(2, "int64"), "b")
        out = bb.emit(rx.op.add(lhs, rhs), "c")
        bb.emit_func_output(out)

    seq_expr = bb.finalize()["func"].body

    bb = rx.BlockBuilder()
    with bb.function("func", [], private=True):
        with bb.dataflow():
            lhs = bb.emit(rx.const(3, "int64"), "d")
            # This would be ill-formed, as it requires breaking up the
            # DataflowBlock with a BindingBlock.
            rhs = bb.emit(seq_expr, "e")

            # We cannot throw an error at that point, because it is
            # only the later usage of "d" that results in use of a
            # DataflowVar outside of its home DataflowBlock.
            out = bb.emit_output(rx.op.add(lhs, rhs), "f")
        bb.emit_func_output(out)

    output = bb.finalize()["func"]

    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(tvm_tirx.ir.IRModule.from_expr(output))


def test_emit_well_formed_nested_seqexpr_in_dataflow_block():
    """May emit a SeqExpr inside a DataflowBlock

    This produces well-formed code, and should not have any output
    produced by the normalizer.  See also
    test_emit_ill_formed_nested_seqexpr_in_dataflow_block.
    """
    bb = rx.BlockBuilder()

    with bb.function("func", []):
        lhs = bb.emit(rx.const(1, "int64"), "a")
        rhs = bb.emit(rx.const(2, "int64"), "b")
        out = bb.emit(rx.op.add(lhs, rhs), "c")
        bb.emit_func_output(out)

    seq_expr = bb.finalize()["func"].body

    bb = rx.BlockBuilder()
    with bb.function("func", [], private=True):
        with bb.dataflow():
            lhs = bb.emit(rx.const(3, "int64"), "d")
            # This similarly breaks up the DataflowBlock, with
            # identical steps as the previous test up until this
            # point.
            rhs = bb.emit(seq_expr, "e")

            # But the "d" variable isn't used, and so there aren't any
            # usages of DataflowVar outside of their home
            # DataflowBlock.
            out = bb.emit_output(rhs, "f")
        bb.emit_func_output(out)

    output = bb.finalize()["func"]

    rx.analysis.check_well_formed(tvm_tirx.ir.IRModule.from_expr(output))

    @R.function(private=True)
    def expected() -> R.Tensor((), dtype="int64"):
        with R.dataflow():
            d = R.const(3, "int64")
            R.output()
        a = R.const(1, "int64")
        b = R.const(2, "int64")
        c = R.add(a, b)
        with R.dataflow():
            e = c
            f = e
            R.output(f)
        return f

    tvm_tirx.ir.assert_structural_equal(expected, output)


def test_error_when_unwrapping_dataflowvar():
    """Checks for ill-formed use of DataflowVar at normalization

    We can check for some illegal unwrapping of SeqExpr, though.  If
    the inlined non-dataflow SeqExpr uses a DataflowVar, that should
    trigger an error when the SeqExpr is being unwrapped.
    """
    bb = rx.BlockBuilder()

    lhs = rx.Var("a", rx.TensorType(shape=[], dtype="int64"))

    with bb.function("func", [lhs]):
        rhs = rx.const(2, "int64")
        out = bb.emit(rx.op.add(lhs, rhs))
        bb.emit_func_output(out)

    func = bb.finalize()["func"]

    bb = rx.BlockBuilder()
    with bb.function("func", [], private=True):
        with bb.dataflow():
            local_lhs = bb.emit(rx.const(3, "int64"), "local_a")
            rhs = bb.emit(func.bind_params({lhs: local_lhs}).body, "f")
            out = bb.emit_output(rhs, "f")

        with pytest.raises(RuntimeError, match="Malformed AST"):
            bb.emit_func_output(out)


def test_deduplication_when_input_contains_duplicates():
    """De-duplication of IRModules

    A well-formed IRModule may contain duplicate function definitions.
    This is rare, as most functions can be disambiguated by the the
    function attribute `tvm::attr::kGlobalSymbol`.  However, private
    functions do not have this attribute, and a well-formed IRModule
    may contain multiple copies of the same function.

    This is a regression test.  Previous implementation de-duplicated
    using a `Dict[Function, GlobalVar]`, which has the failure mode
    shown below.  This was resolved by de-duplicating using a
    `Dict[Function, Set[GlobalVar]]` instead.

    """

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor):
            B = Module.subroutine_a(A)
            C = Module.subroutine_b(B)
            return C

        @R.function(private=True)
        def subroutine_a(arg: R.Tensor) -> R.Tensor:
            return R.add(arg, arg)

        @R.function(private=True)
        def subroutine_b(arg: R.Tensor) -> R.Tensor:
            return R.add(arg, arg)

        @R.function(private=True)
        def subroutine_c(arg: R.Tensor) -> R.Tensor:
            return R.multiply(arg, arg)

    # This test case is only valid when the two subroutines are
    # structurally equal, and therefore allowed to be de-duplicated by
    # the BlockBuilder.
    tvm_tirx.ir.assert_structural_equal(Module["subroutine_a"], Module["subroutine_b"])

    gvar_a = Module.get_global_var("subroutine_a")
    gvar_b = Module.get_global_var("subroutine_b")
    subroutine_c = Module["subroutine_c"]

    bb = rx.BlockBuilder(Module)

    # Add a function to the module.  What we add doesn't matter, as
    # this is only to initialize the de-duplication map.
    bb.add_func(subroutine_c, "_unused")
    # The deduplication table now maps `subroutine_ab` to either
    # `gvar_a` or `gvar_b`.

    # Update gvar_a.
    bb.update_func(gvar_a, subroutine_c)
    # The deduplication map no longer has an entry for
    # `subroutine_ab`.

    # Update gvar_b.  The deduplication map is present (because we
    # called `add_func`), but doesn't contain an entry for
    # `subroutine_ab` (because it was just removed).  This throws an
    # error.
    bb.update_func(gvar_b, subroutine_c)


if __name__ == "__main__":
    tvm_tirx.testing.main()
