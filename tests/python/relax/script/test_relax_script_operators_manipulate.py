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
"""Relax script operators manipulate."""

import tvm
import tvm_tirx
import tvm_tirx.script
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm_tirx import IRModule
from tvm_tirx.script import ir as I


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


def test_broadcast_to():
    @R.function
    def foo(x: R.Tensor((2, 1, 3), "float32")) -> R.Tensor((4, 2, 5, 3), "float32"):
        gv: R.Tensor((4, 2, 5, 3), "float32") = R.broadcast_to(x, (4, 2, 5, 3))
        return gv

    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 1, 3), "float32"))
    with bb.function("foo", [x]):
        gv = bb.emit(relax.op.broadcast_to(x, (4, 2, 5, 3)))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_concat():
    @R.function
    def foo(
        x1: R.Tensor((1, 2, 3), "float32"),
        x2: R.Tensor((1, 3, 3), "float32"),
        x3: R.Tensor((1, 4, 3), "float32"),
    ) -> R.Tensor((1, 9, 3), "float32"):
        gv: R.Tensor((1, 9, 3), "float32") = R.concat((x1, x2, x3), axis=1)
        return gv

    x1 = relax.Var("x1", R.Tensor((1, 2, 3), "float32"))
    x2 = relax.Var("x2", R.Tensor((1, 3, 3), "float32"))
    x3 = relax.Var("x3", R.Tensor((1, 4, 3), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x1, x2, x3]):
        gv = bb.emit(relax.op.concat((x1, x2, x3), axis=1))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_concat_without_specified_axis():
    @R.function
    def foo(
        x1: R.Tensor((2,), "float32"), x2: R.Tensor((3,), "float32"), x3: R.Tensor((4,), "float32")
    ) -> R.Tensor((9,), "float32"):
        gv: R.Tensor((9,), "float32") = R.concat((x1, x2, x3), axis=None)
        return gv

    x1 = relax.Var("x1", R.Tensor((2,), "float32"))
    x2 = relax.Var("x2", R.Tensor((3,), "float32"))
    x3 = relax.Var("x3", R.Tensor((4,), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x1, x2, x3]):
        gv = bb.emit(relax.op.concat((x1, x2, x3), axis=None))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_expand_dims():
    @R.function
    def foo(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor((2, 1, 1, 1, 3, 1, 4, 1), "float32"):
        gv: R.Tensor((2, 1, 1, 1, 3, 1, 4, 1), "float32") = R.expand_dims(x, axis=[-1, 1, -6, 3, 5])
        return gv

    x = relax.Var("x", R.Tensor((2, 3, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(relax.op.expand_dims(x, axis=[-1, 1, -6, 3, 5]))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_flatten():
    @R.function
    def foo(x: R.Tensor((3, 4, 5), "float32")) -> R.Tensor((60,), "float32"):
        gv: R.Tensor((60,), "float32") = R.flatten(x)
        return gv

    x = relax.Var("x", R.Tensor((3, 4, 5), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(relax.op.flatten(x))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_layout_transform():
    transformation = lambda n, c, h, w: (n, h, w, c)  # noqa: E731

    @R.function
    def foo(x: R.Tensor((2, 3, 4, 5), "float32")):
        gv: R.Tensor((2, 4, 5, 3), "float32") = R.layout_transform(x, index_map=transformation)
        return gv

    x = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(relax.op.layout_transform(x, index_map=transformation))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_layout_transform_with_padding():
    transformation = lambda n, c, h, w: (n, c // 3, h, w, c % 3)  # noqa: E731

    @R.function
    def foo(x: R.Tensor((10, 20, 2, 2), "float32")):
        gv: R.Tensor((10, 7, 2, 2, 3), "float32") = R.layout_transform(
            x, index_map=transformation, pad_value=2
        )
        return gv

    x = relax.Var("x", R.Tensor((10, 20, 2, 2), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(relax.op.layout_transform(x, index_map=transformation, pad_value=2))
        bb.emit_func_output(gv)

        _check(foo, bb.get()["foo"])


def test_permute_dims():
    @R.function
    def foo(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor((2, 4, 3, 1), "float32"):
        gv: R.Tensor((2, 4, 3, 1), "float32") = R.permute_dims(x, axes=[1, -1, 2, -4])
        return gv

    x = relax.Var("x", R.Tensor((1, 2, 3, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(relax.op.permute_dims(x, axes=[1, -1, 2, -4]))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_permute_dims_none_arg():
    @R.function
    def foo(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor((4, 3, 2, 1), "float32"):
        gv: R.Tensor((4, 3, 2, 1), "float32") = R.permute_dims(x)
        return gv

    x = relax.Var("x", R.Tensor((1, 2, 3, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(relax.op.permute_dims(x))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_reshape():
    @R.function
    def foo(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor((8, 3), "float32"):
        gv: R.Tensor((8, 3), "float32") = R.reshape(x, (8, 3))
        return gv

    x = relax.Var("x", R.Tensor((1, 2, 3, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(relax.op.reshape(x, shape=(8, 3)))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_reshape_infer_dim():
    @R.function
    def foo(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor((8, 1, 3), "float32"):
        gv: R.Tensor((8, 1, 3), "float32") = R.reshape(x, (8, -1, 3))
        return gv

    x = relax.Var("x", R.Tensor((1, 2, 3, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(relax.op.reshape(x, shape=(8, -1, 3)))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_split_by_indices():
    @R.function
    def foo(x: R.Tensor((2, 10, 4), dtype="float32")) -> R.Tuple(
        R.Tensor((2, 0, 4), dtype="float32"),
        R.Tensor((2, 2, 4), dtype="float32"),
        R.Tensor((2, 4, 4), dtype="float32"),
        R.Tensor((2, 0, 4), dtype="float32"),
        R.Tensor((2, 4, 4), dtype="float32"),
        R.Tensor((2, 2, 4), dtype="float32"),
        R.Tensor((2, 0, 4), dtype="float32"),
        R.Tensor((2, 1, 4), dtype="float32"),
    ):
        gv: R.Tuple(
            R.Tensor((2, 0, 4), dtype="float32"),
            R.Tensor((2, 2, 4), dtype="float32"),
            R.Tensor((2, 4, 4), dtype="float32"),
            R.Tensor((2, 0, 4), dtype="float32"),
            R.Tensor((2, 4, 4), dtype="float32"),
            R.Tensor((2, 2, 4), dtype="float32"),
            R.Tensor((2, 0, 4), dtype="float32"),
            R.Tensor((2, 1, 4), dtype="float32"),
        ) = R.split(x, indices_or_sections=[-2, 2, 6, 4, 8, 12, 9], axis=1)
        return gv

    x = relax.Var("x", R.Tensor((2, 10, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(relax.op.split(x, indices_or_sections=[-2, 2, 6, 4, 8, 12, 9], axis=1))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_split_by_n_section():
    @R.function
    def foo(x: R.Tensor((2, 10, 4), dtype="float32")) -> R.Tuple(
        R.Tensor((2, 2, 4), dtype="float32"),
        R.Tensor((2, 2, 4), dtype="float32"),
        R.Tensor((2, 2, 4), dtype="float32"),
        R.Tensor((2, 2, 4), dtype="float32"),
        R.Tensor((2, 2, 4), dtype="float32"),
    ):
        gv: R.Tuple(
            R.Tensor((2, 2, 4), dtype="float32"),
            R.Tensor((2, 2, 4), dtype="float32"),
            R.Tensor((2, 2, 4), dtype="float32"),
            R.Tensor((2, 2, 4), dtype="float32"),
            R.Tensor((2, 2, 4), dtype="float32"),
        ) = R.split(x, indices_or_sections=5, axis=1)
        return gv

    x = relax.Var("x", R.Tensor((2, 10, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(relax.op.split(x, indices_or_sections=5, axis=1))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_squeeze():
    @R.function
    def foo(x: R.Tensor((2, 1, 3, 1, 1, 4), "float32")) -> R.Tensor((2, 3, 4), "float32"):
        gv: R.Tensor((2, 3, 4), "float32") = R.squeeze(x)
        return gv

    x = relax.Var("x", R.Tensor((2, 1, 3, 1, 1, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(relax.op.squeeze(x))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_squeeze_with_indices():
    @R.function
    def foo(x: R.Tensor((2, 1, 3, 1, 1, 4), "float32")) -> R.Tensor((2, 3, 1, 4), "float32"):
        gv: R.Tensor((2, 3, 1, 4), "float32") = R.squeeze(x, axis=[3, -5])
        return gv

    x = relax.Var("x", R.Tensor((2, 1, 3, 1, 1, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(relax.op.squeeze(x, axis=[3, -5]))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_collapse_sum_like():
    @R.function
    def foo(x: R.Tensor((3, 4, 5), "float32"), y: R.Tensor((4, 5), "float32")) -> R.Tensor(
        (4, 5), "float32"
    ):
        gv: R.Tensor((4, 5), "float32") = R.collapse_sum_like(x, y)
        return gv

    x = relax.Var("x", R.Tensor((3, 4, 5), "float32"))
    y = relax.Var("y", R.Tensor((4, 5), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x, y]):
        gv = bb.emit(relax.op.collapse_sum_like(x, y))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_collapse_sum_to():
    @R.function
    def foo(x: R.Tensor((3, 4, 5), "float32")) -> R.Tensor((4, 5), "float32"):
        gv: R.Tensor((4, 5), "float32") = R.collapse_sum_to(x, (4, 5))
        return gv

    x = relax.Var("x", R.Tensor((3, 4, 5), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(relax.op.collapse_sum_to(x, (4, 5)))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_repeat():
    @R.function
    def foo(x: R.Tensor((2, 3, 4), "float32")):
        gv = R.repeat(x, 3, 1)
        return gv

    x = relax.Var("x", R.Tensor((2, 3, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(relax.op.repeat(x, 3, 1))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_repeat_no_axis():
    @R.function
    def foo(x: R.Tensor((2, 3, 4), "float32")):
        gv = R.repeat(x, 3)
        return gv

    x = relax.Var("x", R.Tensor((2, 3, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(relax.op.repeat(x, 3))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_tile():
    @R.function
    def foo(x: R.Tensor((2, 3, 4), "float32")):
        gv = R.tile(x, (2, 3))
        return gv

    x = relax.Var("x", R.Tensor((2, 3, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(relax.op.tile(x, (2, 3)))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_flip():
    @R.function
    def foo(x: R.Tensor((2, 3, 4), "float32")):
        gv = R.flip(x, axis=1)
        return gv

    x = relax.Var("x", R.Tensor((2, 3, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(relax.op.flip(x, axis=1))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_to_vdevice():
    @I.ir_module
    class ToVDevice:
        I.module_global_infos({"vdevice": [R.vdevice("llvm")]})

        @R.function
        def foo(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            tensor = R.to_vdevice(x, "llvm")
            return tensor

    x = relax.Var("x", R.Tensor((), "int32"))
    bb = relax.BlockBuilder()
    vdev = R.vdevice("llvm")
    with bb.function("foo", (x,)):
        tensor = bb.emit(relax.op.to_vdevice(x, vdev))
        bb.emit_func_output(tensor)
    bb.get().update_global_info("vdevice", [vdev])

    _check(ToVDevice, bb.get())


def test_hint_on_device():
    @R.function
    def foo(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
        r = R.hint_on_device(x, device_type=1, index=0)
        return r

    x = relax.Var("x", R.Tensor((), "int32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        tensor = bb.emit(relax.op.hint_on_device(x, device_type=1, index=0))
        bb.emit_func_output(tensor)

    _check(foo, bb.get()["foo"])


def test_hint_on_device_scoped():
    @R.function
    def foo(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
        r = R.hint_on_device(x, device_type=4, index=2, memory_scope="global.texture")
        return r

    x = relax.Var("x", R.Tensor((), "int32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        tensor = bb.emit(
            relax.op.hint_on_device(x, device_type=4, index=2, memory_scope="global.texture")
        )
        bb.emit_func_output(tensor)

    _check(foo, bb.get()["foo"])


if __name__ == "__main__":
    tvm_tirx.testing.main()
