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
"""Tests for GPU median dispatch."""

import numpy as np
import pytest

import tvm
import tvm_tirx
from tvm import relax
from tvm.testing import device_enabled


def make_median(shape, dtype, axis, keepdims):
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorType(shape, dtype))
    with bb.function("main", [x]):
        with bb.dataflow():
            out = bb.emit(relax.op.median(x, axis=axis, keepdims=keepdims))
            out = bb.emit_output(out)
        bb.emit_func_output(out)
    return bb.get()


@pytest.mark.parametrize("axis", [None, -1, 0])
@pytest.mark.parametrize("keepdims", [False, True])
def test_dispatch_median_cuda_type(axis, keepdims):
    mod = make_median((2, 3), "float32", axis, keepdims)
    with tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_80"}):
        dispatched = relax.backend.DispatchSortScan()(mod)
    tvm_tirx.ir.assert_structural_equal(mod["main"].ret_ty, dispatched["main"].ret_ty)
    assert "R.median(" not in dispatched.script()


def test_dispatch_median_cpu_unchanged():
    mod = make_median((2, 3), "float32", 1, False)
    with tvm_tirx.target.Target("llvm"):
        dispatched = relax.backend.DispatchSortScan()(mod)
    tvm_tirx.ir.assert_structural_equal(mod, dispatched)


@pytest.mark.gpu
@pytest.mark.skipif(
    not tvm_tirx.runtime.enabled("llvm") or not device_enabled("cuda"),
    reason="CUDA and LLVM are required",
)
@pytest.mark.parametrize(
    "shape,axis,keepdims,dtype",
    [
        ((5,), None, False, "float32"),
        ((2, 3), None, True, "float32"),
        ((3, 4), -1, False, "float32"),
        ((3, 4), 0, True, "float32"),
        ((2, 3, 4), 1, False, "float16"),
        ((2, 65), -1, True, "int32"),
        ((), None, True, "float32"),
    ],
)
def test_median_cuda_values_and_indices(shape, axis, keepdims, dtype):
    rng = np.random.default_rng(20261104)
    size = int(np.prod(shape))
    data = rng.permutation(size).astype(dtype).reshape(shape)
    mod = make_median(shape, dtype, axis, keepdims)
    dev = tvm_tirx.cuda(0)
    arch = "sm_" + dev.compute_version.replace(".", "")
    target = tvm_tirx.target.Target(
        {"kind": "cuda", "arch": arch, "host": {"kind": "llvm", "mcpu": "generic"}}
    )
    with target:
        exe = tvm.compile(mod, target=target)
    vm = relax.VirtualMachine(exe, dev)
    result = vm["main"](tvm_tirx.runtime.tensor(data, dev))
    if axis is None:
        expected = np.sort(data.ravel())[(data.size - 1) // 2]
        expected = np.asarray(expected, dtype=dtype).reshape([1] * len(shape) if keepdims else ())
        actual = result.numpy()
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(actual, expected)
    else:
        sorted_indices = np.argsort(data, axis=axis, kind="stable")
        expected_index = np.take(sorted_indices, (shape[axis] - 1) // 2, axis=axis)
        if keepdims:
            expected_index = np.expand_dims(expected_index, axis=axis)
        gather_index = expected_index if keepdims else np.expand_dims(expected_index, axis=axis)
        expected_value = np.take_along_axis(data, gather_index, axis=axis)
        if not keepdims:
            expected_value = np.squeeze(expected_value, axis=axis)
        actual_value, actual_index = result[0].numpy(), result[1].numpy()
        assert actual_value.shape == expected_value.shape
        assert actual_index.shape == expected_index.shape
        np.testing.assert_array_equal(actual_value, expected_value)
        np.testing.assert_array_equal(actual_index, expected_index)
    dev.sync()


@pytest.mark.gpu
@pytest.mark.skipif(
    not tvm_tirx.runtime.enabled("llvm") or not device_enabled("cuda"),
    reason="CUDA and LLVM are required",
)
@pytest.mark.parametrize("axis", [None, 0])
def test_median_cuda_symbolic_rows(axis):
    n = tvm_tirx.tirx.Var("n", "int64")
    mod = make_median((n, 4), "float32", axis, True)
    dev = tvm_tirx.cuda(0)
    target = tvm_tirx.target.Target(
        {
            "kind": "cuda",
            "arch": "sm_" + dev.compute_version.replace(".", ""),
            "host": {"kind": "llvm", "mcpu": "generic"},
        }
    )
    with target:
        exe = tvm.compile(mod, target=target)
    vm = relax.VirtualMachine(exe, dev)
    for rows in (3, 5):
        data = np.arange(rows * 4, dtype="float32").reshape(rows, 4)[::-1].copy()
        result = vm["main"](tvm_tirx.runtime.tensor(data, dev))
        if axis is None:
            expected = np.sort(data.ravel())[(data.size - 1) // 2].reshape((1, 1))
            actual = result.numpy()
            assert actual.shape == expected.shape
            np.testing.assert_array_equal(actual, expected)
        else:
            order = np.argsort(data, axis=0, kind="stable")
            expected_index = order[(rows - 1) // 2 : (rows + 1) // 2]
            expected_value = np.take_along_axis(data, expected_index, axis=0)
            actual_value, actual_index = result[0].numpy(), result[1].numpy()
            assert actual_value.shape == expected_value.shape
            assert actual_index.shape == expected_index.shape
            np.testing.assert_array_equal(actual_value, expected_value)
            np.testing.assert_array_equal(actual_index, expected_index)
    dev.sync()


def test_dispatch_median_without_target_unchanged():
    mod = make_median((2, 3), "float32", None, True)
    dispatched = relax.backend.DispatchSortScan()(mod)
    tvm_tirx.ir.assert_structural_equal(mod, dispatched)
