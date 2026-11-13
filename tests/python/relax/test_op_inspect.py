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

import ctypes

import numpy as np
import pytest
import tvm_ffi

import tvm
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm_tirx.ir import Op
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


# Parameterization for reading dtype of DLTensor.  Chosen to have
# multiple distinct type codes, number of lanes, and widths.
@pytest.fixture(
    params=[
        "int32",
        "int64",
        "float32",
        "float32x4",
        "bfloat",
        "float8_e4m3fn",
    ],
    scope="session",
)
def dtype(request):
    return request.param


@pytest.fixture(
    params=[
        [],
        [16],
        [128, 256],
        [1] * 64,
    ],
    scope="session",
)
def shape(request):
    return request.param


@pytest.fixture(params=[0, 64, 128], scope="session")
def elem_offset(request):
    return request.param


def inspect_tensor_field(op_name, *args):
    return relax.Call(Op.get(f"relax.inspect.{op_name}"), args)


def test_tensor_dtype_code(dtype):
    @I.ir_module
    class mod:
        @R.function
        def main(A: R.Tensor):
            return inspect_tensor_field("tensor_dtype_code", A)

    built = tvm.compile(mod)
    vm = relax.VirtualMachine(built, tvm_tirx.cpu())

    arg = tvm_tirx.runtime.empty([16], dtype)
    res = vm["main"](arg)

    expected_type_code = tvm_ffi.dtype(dtype).type_code
    assert res == expected_type_code


def test_tensor_dtype_bits(dtype):
    @I.ir_module
    class mod:
        @R.function
        def main(A: R.Tensor):
            return inspect_tensor_field("tensor_dtype_bits", A)

    built = tvm.compile(mod)
    vm = relax.VirtualMachine(built, tvm_tirx.cpu())

    arg = tvm_tirx.runtime.empty([16], dtype)
    res = vm["main"](arg)

    expected_type_bits = tvm_ffi.dtype(dtype).bits
    assert res == expected_type_bits


def test_tensor_dtype_lanes(dtype):
    @I.ir_module
    class mod:
        @R.function
        def main(A: R.Tensor):
            return inspect_tensor_field("tensor_dtype_lanes", A)

    built = tvm.compile(mod)
    vm = relax.VirtualMachine(built, tvm_tirx.cpu())

    arg = tvm_tirx.runtime.empty([16], dtype)
    res = vm["main"](arg)

    expected_type_lanes = tvm_ffi.dtype(dtype).lanes
    assert res == expected_type_lanes


def test_tensor_ndim(shape):
    @I.ir_module
    class mod:
        @R.function
        def main(A: R.Tensor):
            return inspect_tensor_field("tensor_ndim", A)

    built = tvm.compile(mod)
    vm = relax.VirtualMachine(built, tvm_tirx.cpu())

    arg = tvm_tirx.runtime.empty(shape, "int32")
    res = vm["main"](arg)

    assert res == len(shape)


def test_tensor_shape(shape):
    @I.ir_module
    class mod:
        @R.function
        def main(A: R.Tensor, axis: T.int64):
            return inspect_tensor_field("tensor_shape_i", A, axis)

    built = tvm.compile(mod)
    vm = relax.VirtualMachine(built, tvm_tirx.cpu())

    arg = tvm_tirx.runtime.empty(shape, "int32")

    res = [vm["main"](arg, i) for i, _ in enumerate(shape)]

    tvm_tirx.ir.assert_structural_equal(res, shape)


def _get_compact_striding(shape):
    strides = []
    product = 1
    for dim in reversed(shape):
        strides.append(product)
        product *= dim
    return list(reversed(strides))


def test_strides_of_compact_tensor(shape):
    @I.ir_module
    class mod:
        @R.function
        def main(A: R.Tensor, axis: T.int64):
            return inspect_tensor_field("tensor_stride_i", A, axis)

    built = tvm.compile(mod)
    vm = relax.VirtualMachine(built, tvm_tirx.cpu())

    arg = tvm_tirx.runtime.empty(shape, "int32")

    res = [vm["main"](arg, i) for i, _ in enumerate(shape)]
    expected = _get_compact_striding(shape)

    tvm_tirx.ir.assert_structural_equal(res, expected)


def test_strides_of_non_compact_tensor():
    @I.ir_module
    class mod:
        @R.function
        def main(A: R.Tensor, axis: T.int64):
            return inspect_tensor_field("tensor_stride_i", A, axis)

    built = tvm.compile(mod)
    vm = relax.VirtualMachine(built, tvm_tirx.cpu())
    view_shape = [4, 4]
    expected_strides = [1, 4]
    # use transpose to make strides non-compact
    x = np.zeros([4, 4], "int32").T
    y = tvm_ffi.from_dlpack(x, require_alignment=4, require_contiguous=False)
    res = [vm["main"](y, i) for i, _ in enumerate(view_shape)]
    tvm_tirx.ir.assert_structural_equal(res, expected_strides)


def test_byte_offset(elem_offset):
    backing_shape = [64, 64]
    view_shape = [16, 16]
    byte_offset = elem_offset * 4

    @I.ir_module
    class mod:
        @R.function
        def main(A: R.Tensor):
            return inspect_tensor_field("tensor_byte_offset", A)

    built = tvm.compile(mod)
    vm = relax.VirtualMachine(built, tvm_tirx.cpu())
    dtype = "int32"
    backing_tensor = tvm_tirx.runtime.empty(backing_shape, dtype)
    view = backing_tensor._create_view(view_shape, dtype, relative_byte_offset=byte_offset)
    res = vm["main"](view)
    assert res == byte_offset


def test_elem_offset(elem_offset, dtype):
    tvm_dtype = tvm_ffi.dtype(dtype)

    backing_shape = [64, 64]
    view_shape = [16, 16]
    element_bytes = (tvm_dtype.bits * tvm_dtype.lanes) // 8
    byte_offset = elem_offset * element_bytes

    @I.ir_module
    class mod:
        @R.function
        def main(A: R.Tensor):
            return inspect_tensor_field("tensor_elem_offset", A)

    built = tvm.compile(mod)
    vm = relax.VirtualMachine(built, tvm_tirx.cpu())

    backing_tensor = tvm_tirx.runtime.empty(backing_shape, dtype)
    view = backing_tensor._create_view(view_shape, dtype, relative_byte_offset=byte_offset)
    res = vm["main"](view)

    assert res == elem_offset


if __name__ == "__main__":
    tvm_tirx.testing.main()
