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
"""Relax script operators create."""

import tvm
import tvm_tirx
import tvm_tirx.script
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm_tirx import IRModule


def _check(
    parsed: relax.Function | IRModule,
    expect: relax.Function | IRModule | None,
):
    test = parsed.script(show_meta=True)
    roundtrip_mod = tvm_tirx.script.from_source(
        test, extra_vars={"I": tvm_tirx.script.ir, "R": tvm.relax.script, "T": tvm_tirx.script.tirx}
    )
    tvm_tirx.ir.assert_structural_equal(parsed, roundtrip_mod)
    if expect:
        tvm_tirx.ir.assert_structural_equal(parsed, expect)


def test_full():
    @R.function
    def foo(v: R.Tensor((), "int32")) -> R.Tensor((2, 3), "float32"):
        gv: R.Tensor((2, 3), "float32") = R.full((2, 3), v, dtype="float32")
        return gv

    bb = relax.BlockBuilder()
    v = relax.Var("v", R.Tensor((), "int32"))
    with bb.function("foo", [v]):
        gv = bb.emit(relax.op.full((2, 3), v, "float32"))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_full_like():
    @R.function
    def foo(x: R.Tensor((2, 3), "float16"), v: R.Tensor((), "float32")) -> R.Tensor(
        (2, 3), "float16"
    ):
        gv: R.Tensor((2, 3), "float16") = R.full_like(x, v)
        return gv

    x = relax.Var("x", R.Tensor((2, 3), "float16"))
    v = relax.Var("y", R.Tensor((), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x, v]):
        gv = bb.emit(relax.op.full_like(x, v))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_ones():
    @R.function
    def foo(dumb_param: R.Tensor()) -> R.Tensor((2, 3), "float32"):
        gv: R.Tensor((2, 3), "float32") = R.ones((2, 3), "float32")
        return gv

    bb = relax.BlockBuilder()
    dumb_param = relax.Var("dumb_param", R.Tensor())
    with bb.function("foo", [dumb_param]):
        gv = bb.emit(relax.op.ones((2, 3), "float32"))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_ones_like():
    @R.function
    def foo(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
        gv: R.Tensor((2, 3), "float32") = R.ones_like(x)
        return gv

    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(relax.op.ones_like(x))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_zeros():
    @R.function
    def foo(dumb_param: R.Tensor()) -> R.Tensor((2, 3), "float32"):
        gv: R.Tensor((2, 3), "float32") = R.zeros((2, 3), "float32")
        return gv

    bb = relax.BlockBuilder()
    dumb_param = relax.Var("dumb_param", R.Tensor())
    with bb.function("foo", [dumb_param]):
        gv = bb.emit(relax.op.zeros((2, 3), "float32"))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_zeros_like():
    @R.function
    def foo(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
        gv: R.Tensor((2, 3), "float32") = R.zeros_like(x)
        return gv

    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(relax.op.zeros_like(x))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_arange():
    @R.function
    def foo():
        gv = R.arange(1, 10, 2)
        return gv

    bb = relax.BlockBuilder()
    with bb.function("foo", []):
        gv = bb.emit(relax.op.arange(1, 10, 2))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_tril():
    @R.function
    def foo(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor((2, 3, 4), "float32"):
        gv: R.Tensor((2, 3, 4), "float32") = R.tril(x, k=2)
        return gv

    x = relax.Var("x", R.Tensor((2, 3, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(relax.op.tril(x, k=2))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_triu():
    @R.function
    def foo(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor((2, 3, 4), "float32"):
        gv: R.Tensor((2, 3, 4), "float32") = R.triu(x)
        return gv

    x = relax.Var("x", R.Tensor((2, 3, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(relax.op.triu(x))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


if __name__ == "__main__":
    tvm_tirx.testing.main()
