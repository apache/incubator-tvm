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
# ruff: noqa: F401
import numpy as np
import pytest
import tvm_ffi

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.contrib import tvmjs
from tvm.relax.testing.runtime_builtin import MakeShapeCode, MatchShapeCode
from tvm_tirx.ir import assert_structural_equal
from tvm_tirx.support import utils


def test_make_shape():
    MK = MakeShapeCode
    make_shape = tvm_ffi.get_global_func("vm.builtin.make_shape")
    heap = tvm_tirx.runtime.tensor(np.arange(10).astype("int64"))
    s = make_shape(heap, 3, MK.USE_IMM, 10, MK.LOAD_SHAPE, 0, MK.LOAD_SHAPE, 2)

    assert s == tvm_ffi.Shape([10, 0, 2])


def test_match_shape():
    MS = MatchShapeCode
    match_shape = tvm_ffi.get_global_func("vm.builtin.match_shape")
    heap = tvm_tirx.runtime.tensor(np.zeros(10).astype("int64"))

    assert heap.numpy()[2] == 0

    s = tvm_ffi.Shape([1, 2, 3])
    x = tvm_tirx.runtime.tensor(np.zeros([1, 2, 3]))

    match_shape(s, heap, 3, MS.ASSERT_EQUAL_TO_IMM, 1, MS.STORE_TO_HEAP, 2, MS.NO_OP, 0, "")

    assert heap.numpy()[2] == 2

    match_shape(
        x,
        heap,
        3,
        MS.ASSERT_EQUAL_TO_IMM,
        1,
        MS.ASSERT_EQUAL_TO_LOAD,
        2,
        MS.ASSERT_EQUAL_TO_IMM,
        3,
        "",
    )

    with pytest.raises(RuntimeError):
        match_shape(s, heap, 2, MS.ASSERT_EQUAL_TO_IMM, 1, MS.STORE_TO_HEAP, 2, "")

    with pytest.raises(RuntimeError):
        match_shape(s, heap, 3, MS.ASSERT_EQUAL_TO_IMM, 2, MS.STORE_TO_HEAP, 2, MS.NO_OP, 0, "")


def test_check_shape_info():
    check_shape_info = tvm_ffi.get_global_func("vm.builtin.check_shape_info")
    s = tvm_ffi.Shape([1, 2, 3])

    check_shape_info(s, 3, "")
    check_shape_info(s, -1, "")

    # wrong ndim
    with pytest.raises(ValueError):
        check_shape_info(s, 2, "")

    # wrong type
    with pytest.raises(TypeError):
        check_shape_info([], 2, "")


def test_check_tensor_info():
    check_tensor_info = tvm_ffi.get_global_func("vm.builtin.check_tensor_info")
    x = tvm_tirx.runtime.tensor(np.zeros((2, 3)).astype("int32"))

    check_tensor_info(x, 2, "int32", "")
    check_tensor_info(x, -1, "int32", "")
    check_tensor_info(x, 2, None, "")
    check_tensor_info(x, -1, None, "")

    # allow not passing in dtype
    check_tensor_info(x, 2, "")
    check_tensor_info(x, -1, "")

    # ndim mismatch
    with pytest.raises(ValueError, match=r".* ndim .*"):
        check_tensor_info(x, 3, "int32", "")

    # dtype mismatch
    with pytest.raises(ValueError, match=r"myerror.* dtype .*"):
        check_tensor_info(x, 2, "float32", "myerror")

    # error with context
    with pytest.raises(ValueError, match=r".* myerror .*"):
        check_tensor_info(x, 3, "myerror")

    # wrong type
    with pytest.raises(TypeError):
        check_tensor_info([], 2, None, "")


def test_check_tuple_info():
    check_tuple_info = tvm_ffi.get_global_func("vm.builtin.check_tuple_info")
    x = tvm_tirx.runtime.tensor(np.zeros((2, 3)).astype("int32"))
    t = tvm_ffi.convert([x, x, x])

    check_tuple_info(t, 3, "")

    # size
    with pytest.raises(ValueError, match=r".*elements.*"):
        check_tuple_info(t, 2, "")

    # wrong type
    with pytest.raises(TypeError):
        check_tuple_info(x, 2, "")


def test_check_func_info():
    check_func_info = tvm_ffi.get_global_func("vm.builtin.check_func_info")
    f = tvm_ffi.convert(lambda x: x)
    x = tvm_tirx.runtime.tensor(np.zeros((2, 3)).astype("int32"))

    check_func_info(f, "")

    # wrong type
    with pytest.raises(TypeError, match=".*myerror.*"):
        check_func_info(x, "myerror")


def test_tuple_getitem():
    tuple_getitem = tvm_ffi.get_global_func("vm.builtin.tuple_getitem")
    x = tvm_tirx.runtime.tensor(np.zeros((2, 3)).astype("int32"))
    y = tvm_tirx.runtime.tensor(np.zeros((2, 3)).astype("int32"))
    t = tvm_ffi.convert([x, y])

    assert tuple_getitem(t, 0) == x
    assert tuple_getitem(t, 1) == y


def test_attention_kv_cache():
    fcreate = tvm_ffi.get_global_func("vm.builtin.attention_kv_cache_create")
    fappend = tvm_ffi.get_global_func("vm.builtin.attention_kv_cache_append")
    fview = tvm_ffi.get_global_func("vm.builtin.attention_kv_cache_view")

    cache = fcreate(tvm_tirx.runtime.empty((1, 2), dtype="int32"), tvm_ffi.Shape([2, 2]), 0)
    num_steps = 2
    for i in range(num_steps):
        cache = fappend(cache, tvm_tirx.runtime.tensor(i * np.ones((1, 2)).astype("int32")))

    res = fview(cache, tvm_ffi.Shape((num_steps, 2))).numpy()
    for i in range(num_steps):
        assert res[i][0] == i
        assert res[i][1] == i


def test_tensor_cache():
    fload = tvm_ffi.get_global_func("vm.builtin.tensor_cache.load")
    fget_params = tvm_ffi.get_global_func("vm.builtin.param_array_from_cache")

    param_dict = {
        "x_0": np.array([1, 2, 3], dtype="int32"),
        "x_1": np.random.uniform(size=[10, 20]).astype("float32"),
    }

    temp = utils.tempdir()
    tvmjs.dump_tensor_cache(param_dict, temp.path, encode_format="f32-to-bf16")
    fload(str(temp.path), tvm_tirx.cpu().dlpack_device_type(), 0)
    res = fget_params("x", -1)
    for i, v in enumerate(res):
        v_np = param_dict[f"x_{i}"]
        if v_np.dtype == "float32":
            v_np = tvmjs._convert_bf16_to_f32(tvmjs._convert_f32_to_bf16(v_np))
        tvm_tirx.testing.assert_allclose(v.numpy(), v_np, atol=1e-6, rtol=1e-6)


def test_tensor_cache_update():
    fload = tvm_ffi.get_global_func("vm.builtin.tensor_cache.load")
    fget_params = tvm_ffi.get_global_func("vm.builtin.param_array_from_cache")

    param_dict = {
        "x_0": np.array([1, 2, 3], dtype="int32"),
        "x_1": np.random.uniform(size=[10, 20]).astype("float32"),
    }

    temp = utils.tempdir()
    tvmjs.dump_tensor_cache(param_dict, temp.path, encode_format="f32-to-bf16")
    param_dict["x_1"] = np.random.uniform(size=[10, 20]).astype("float32")
    param_dict["x_2"] = np.random.uniform(size=[10]).astype("float32")
    tvmjs.dump_tensor_cache(
        param_dict, temp.path, encode_format="f32-to-bf16", update_if_exists=True
    )
    fload(str(temp.path), tvm_tirx.cpu().dlpack_device_type(), 0)
    res = fget_params("x", -1)
    for i, v in enumerate(res):
        v_np = param_dict[f"x_{i}"]
        if v_np.dtype == "float32":
            v_np = tvmjs._convert_bf16_to_f32(tvmjs._convert_f32_to_bf16(v_np))
        tvm_tirx.testing.assert_allclose(v.numpy(), v_np, atol=1e-6, rtol=1e-6)


def test_attention_kv_cache_window_override():
    fcreate = tvm_ffi.get_global_func("vm.builtin.attention_kv_cache_create")
    foverride = tvm_ffi.get_global_func("vm.builtin.attention_kv_cache_window_override")
    fview = tvm_ffi.get_global_func("vm.builtin.attention_kv_cache_view")

    current_pos = 4
    cache = fcreate(
        tvm_tirx.runtime.tensor(np.full((16, 2), -1).astype("int32")),
        tvm_ffi.Shape([16, 2]),
        current_pos,
    )
    np_all_arrays = np.zeros((0, 2)).astype("int32")

    num_steps = 10
    for i in range(1, num_steps):
        np_array = i * np.ones((i, 2)).astype("int32")
        np_all_arrays = np.concatenate((np_all_arrays, np_array), axis=0)
        cache = foverride(cache, tvm_tirx.runtime.tensor(np_array), 16)
        current_pos = (current_pos + i) % 16

    res = fview(cache, tvm_ffi.Shape((16, 2))).numpy()

    # unrotate cache and assert cache matches last 16 elements
    assert (
        np_all_arrays[np_all_arrays.shape[0] - 16 :, :]
        == np.concatenate((res[current_pos:], res[:current_pos]))
    ).all()


def test_attention_kv_cache_window_override_with_sinks():
    fcreate = tvm_ffi.get_global_func("vm.builtin.attention_kv_cache_create")
    foverride = tvm_ffi.get_global_func("vm.builtin.attention_kv_cache_window_override_with_sinks")
    fview = tvm_ffi.get_global_func("vm.builtin.attention_kv_cache_view")

    num_attention_sinks = 2
    has_sink = False
    current_pos = 0

    cache = fcreate(
        tvm_tirx.runtime.tensor(np.full((16, 2), -1).astype("int32")),
        tvm_ffi.Shape([16, 2]),
        current_pos,
    )
    np_all_arrays = np.zeros((0, 2)).astype("int32")

    num_steps = 40
    for i in range(num_steps):
        np_array = i * np.ones((1, 2)).astype("int32")
        np_all_arrays = np.concatenate((np_all_arrays, np_array), axis=0)
        cache = foverride(cache, tvm_tirx.runtime.tensor(np_array), 16, num_attention_sinks)

        if has_sink:
            current_pos = max((current_pos + 1) % 16, num_attention_sinks)
        else:
            current_pos += 1
            has_sink = current_pos >= num_attention_sinks

    res = fview(cache, tvm_ffi.Shape((16, 2))).numpy()

    # unrotate cache and assert cache matches last 16 elements
    assert (
        np.concatenate(
            (np_all_arrays[:num_attention_sinks, :], np_all_arrays[-16 + num_attention_sinks :, :])
        )
        == np.concatenate(
            (res[:num_attention_sinks], res[current_pos:], res[num_attention_sinks:current_pos])
        )
    ).all()


if __name__ == "__main__":
    tvm_tirx.testing.main()
