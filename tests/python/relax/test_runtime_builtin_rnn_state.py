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
# pylint: disable=missing-docstring,
from collections.abc import Sequence

import numpy as np
import pytest
import tvm_ffi
from tvm_ffi import Shape

import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import dlight as dl
from tvm.s_tir import script as Ts
from tvm_tirx import tirx
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env

# pylint: disable=invalid-name

np_zero = np.full((16, 16), 0.0, "float16")
np_one = np.full((32, 32), 1.0, "float32")
np_two = np.full((16, 16), 2.0, "float16")
np_three = np.full((32, 32), 3.0, "float32")

reserved_nseq = 4
max_history = 4
num_layers = 1
# Note that kernels in this test file cannot support 1-dim states.
states = [((16, 16), "float16"), ((32, 32), "float32")]

f_clear = None
f_add_sequence = None
f_remove_sequence = None
f_fork_sequence = None
f_popn = None
f_begin_forward = None
f_end_forward = None
f_get = None
f_set = None
f_debug_get = None

f_tir_gets = []
f_tir_sets = []

# pylint: enable=invalid-name


def set_global_func():
    global f_clear, f_add_sequence, f_remove_sequence, f_fork_sequence, f_popn
    global f_begin_forward, f_end_forward, f_get, f_set, f_debug_get
    global f_tir_gets, f_tir_sets

    f_clear = tvm_ffi.get_global_func("vm.builtin.kv_state_clear")
    f_add_sequence = tvm_ffi.get_global_func("vm.builtin.kv_state_add_sequence")
    f_remove_sequence = tvm_ffi.get_global_func("vm.builtin.kv_state_remove_sequence")
    f_fork_sequence = tvm_ffi.get_global_func("vm.builtin.kv_state_fork_sequence")
    f_popn = tvm_ffi.get_global_func("vm.builtin.kv_state_popn")
    f_begin_forward = tvm_ffi.get_global_func("vm.builtin.kv_state_begin_forward")
    f_end_forward = tvm_ffi.get_global_func("vm.builtin.kv_state_end_forward")
    f_get = tvm_ffi.get_global_func("vm.builtin.rnn_state_get")
    f_set = tvm_ffi.get_global_func("vm.builtin.rnn_state_set")
    f_debug_get = tvm_ffi.get_global_func("vm.builtin.rnn_state_debug_get")

    target = tvm_tirx.target.Target("cuda")

    def _build(tir_func):
        mod = tvm_tirx.IRModule({"main": tir_func})
        mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback(), target=target)(mod)  # pylint: disable=not-callable
        f = tvm_tirx.tirx.compile(mod["main"], target=target).mod
        return f.main

    _f_tir_gets, _f_tir_sets = [], []
    for state in states:
        shape, dtype = state
        _f_tir_gets.append(_build(rnn_state_get(shape, dtype)))
        _f_tir_sets.append(_build(rnn_state_set(shape, dtype)))

    f_tir_gets = _f_tir_gets
    f_tir_sets = _f_tir_sets


def create_rnn_state(device):
    f_create = tvm_ffi.get_global_func("vm.builtin.rnn_state_create")
    init_values = [
        tvm_tirx.runtime.tensor(np_zero, device=device),
        tvm_tirx.runtime.tensor(np_one, device=device),
    ]
    return f_create(num_layers, reserved_nseq, max_history, f_tir_gets, f_tir_sets, init_values)


@pytest.fixture
def rnn_state():
    set_global_func()
    return create_rnn_state


def verify_state(state, seq_ids, expected_values):
    layer_id = 0
    for seq_id in seq_ids:
        for state_id, expected_value in enumerate(expected_values[seq_id]):
            state_value = f_debug_get(state, layer_id, state_id, seq_id)
            tvm_tirx.testing.assert_allclose(state_value.numpy(), expected_value)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_rnn_state_get(rnn_state):  # pylint: disable=redefined-outer-name
    def run_and_check():
        device = tvm_tirx.cuda()
        state = rnn_state(device)
        f_clear(state)
        f_add_sequence(state, 0)
        f_begin_forward(state, Shape([0]), Shape([1]))
        tvm_nd_0 = tvm_tirx.runtime.tensor(np.empty((1, 16, 16), "float16"), device=device)
        tvm_nd_1 = tvm_tirx.runtime.tensor(np.empty((1, 32, 32), "float32"), device=device)
        f_get(state, 0, 0, tvm_nd_0)
        f_get(state, 0, 1, tvm_nd_1)
        f_end_forward(state)
        tvm_tirx.testing.assert_allclose(tvm_nd_0.numpy(), np.zeros((1, 16, 16), "float16"))
        tvm_tirx.testing.assert_allclose(tvm_nd_1.numpy(), np.ones((1, 32, 32), "float32"))

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_rnn_state_set(rnn_state):  # pylint: disable=redefined-outer-name
    def run_and_check():
        device = tvm_tirx.cuda()
        state = rnn_state(device)
        f_clear(state)
        for seq_id in range(3):
            f_add_sequence(state, seq_id)
        f_begin_forward(state, Shape([0, 2]), Shape([1, 1]))
        f_set(
            state,
            0,
            0,
            tvm_tirx.runtime.tensor(np.full((2, 16, 16), 2.0, "float16"), device=device),
        )
        f_set(
            state,
            0,
            1,
            tvm_tirx.runtime.tensor(np.full((2, 32, 32), 3.0, "float32"), device=device),
        )
        f_end_forward(state)
        expected_values = [[np_two, np_three], [np_zero, np_one], [np_two, np_three]]
        verify_state(state, [0, 1, 2], expected_values)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_rnn_state_popn(rnn_state):  # pylint: disable=redefined-outer-name
    def run_and_check():
        device = tvm_tirx.cuda()
        state = rnn_state(device)
        f_clear(state)
        f_add_sequence(state, 0)
        f_begin_forward(state, Shape([0]), Shape([1]))
        f_set(state, 0, 0, tvm_tirx.runtime.tensor(np_two.reshape(1, 16, 16), device=device))
        f_set(state, 0, 1, tvm_tirx.runtime.tensor(np_three.reshape(1, 32, 32), device=device))
        f_end_forward(state)
        verify_state(state, [0], [[np_two, np_three]])
        f_popn(state, 0, 1)
        verify_state(state, [0], [[np_zero, np_one]])
        with pytest.raises(RuntimeError):
            f_popn(state, 0, 1)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_rnn_state_fork_sequence(rnn_state):  # pylint: disable=redefined-outer-name
    def run_and_check():
        device = tvm_tirx.cuda()
        state = rnn_state(device)
        f_clear(state)
        f_add_sequence(state, 0)
        f_begin_forward(state, Shape([0]), Shape([1]))
        f_set(state, 0, 0, tvm_tirx.runtime.tensor(np_two.reshape(1, 16, 16), device=device))
        f_set(state, 0, 1, tvm_tirx.runtime.tensor(np_three.reshape(1, 32, 32), device=device))
        f_end_forward(state)
        f_fork_sequence(state, 0, 1, -1)
        verify_state(state, [0, 1], [[np_two, np_three], [np_two, np_three]])
        f_popn(state, 1, 1)
        verify_state(state, [0, 1], [[np_two, np_three], [np_zero, np_one]])

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


def rnn_state_get(
    shape: Sequence[int],
    dtype: str,
):
    # fmt: off
    batch_size = T.dynamic("batch_size", "int32")

    @Ts.function
    def _rnn_state_get(
        storage: T.Tensor((reserved_nseq, max_history, *shape), dtype),
        seq_slot_ids: T.Tensor((batch_size,), 'int32'),
        history_slot_ids: T.Tensor((batch_size,), 'int32'),
        output: T.Tensor((batch_size, *shape), dtype),
    ):

        for i in range(batch_size):
            for (*s,) in T.grid(*shape):
                with Ts.sblock("copy"):
                    vi, *vs = Ts.axis.remap("S" * (len(shape) + 1), [i, *s])
                    seq_id: T.let[T.int32] = seq_slot_ids[vi]
                    history_id: T.let[T.int32] = history_slot_ids[vi]
                    # The following line is equivalent to:
                    # `output[vi, *vs] = storage[seq_id, history_id, *vs]`
                    # However, unpacking operator in subscript requires Python 3.11 or newer
                    T.tensor_store(
                        output, [vi, *vs], T.TensorLoad(storage, [seq_id, history_id, *vs])
                    )
    # fmt: on
    return _rnn_state_get


def rnn_state_set(
    shape: Sequence[int | tirx.Var],
    dtype: str,
):
    # fmt: off
    batch_size = T.dynamic("batch_size", "int32")

    @Ts.function
    def _rnn_state_set(
        storage: T.Tensor((reserved_nseq, max_history, *shape), dtype),
        seq_slot_ids: T.Tensor((batch_size,), 'int32'),
        history_slot_ids: T.Tensor((batch_size,), 'int32'),
        data: T.Tensor((batch_size, *shape), dtype),
    ):

        for i in range(batch_size):
            for (*s,) in T.grid(*shape):
                with Ts.sblock("copy"):
                    vi, *vs = Ts.axis.remap("S" * (len(shape) + 1), [i, *s])
                    seq_id: T.let[T.int32] = seq_slot_ids[vi]
                    history_id: T.let[T.int32] = (history_slot_ids[vi] + 1) % T.cast(
                        max_history, ty = "int32"
                    )
                    # The following line is equivalent to:
                    # `storage[seq_id, history_id, *vs] = data[vi, *vs]`
                    # However, unpacking operator in subscript requires Python 3.11 or newer
                    T.tensor_store(
                        storage, [seq_id, history_id, *vs], T.TensorLoad(data, [vi, *vs])
                    )

    # fmt: on

    return _rnn_state_set


if __name__ == "__main__":
    set_global_func()
    rnn_state = create_rnn_state
    test_rnn_state_get(rnn_state)
    test_rnn_state_set(rnn_state)
    test_rnn_state_popn(rnn_state)
    test_rnn_state_fork_sequence(rnn_state)
