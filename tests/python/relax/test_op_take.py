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

import numpy as np
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.relax import script as R
from tvm.testing import device_enabled
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


@pytest.fixture(params=[0, 1], scope="session")
def axis(request):
    return request.param


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_take_scalar_tensor_as_index(axis):
    """The index of R.take may be a scalar tensor

    Using a scalar tensor as the index reduces the dimension of the
    output.

    """
    target = "llvm"
    dev = tvm_tirx.cpu()

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([16, 16], "float16")):
            output = R.take(A, R.const(1), axis=axis)
            return output

    built = tvm.compile(Module, target=target)
    vm = tvm.relax.VirtualMachine(built, dev)

    np_input = np.random.random(size=[16, 16]).astype("float16")
    tvm_input = tvm_tirx.runtime.tensor(np_input, dev)
    tvm_output = vm["main"](tvm_input)
    np_expected = np_input.take(1, axis=axis)

    tvm_tirx.testing.assert_allclose(tvm_output.numpy(), np_expected)


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_take_1d_tensor_as_index(axis):
    """The index of R.take may be a non-scalar tensor

    In general, `R.take` outputs a tensor of dimension
    `data.ndim + indices.ndim - 1`.
    """
    target = "llvm"
    dev = tvm_tirx.cpu()

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([16, 16], "float16")):
            output = R.take(A, R.const([1]), axis=axis)
            return output

    built = tvm.compile(Module, target=target)
    vm = tvm.relax.VirtualMachine(built, dev)

    np_input = np.random.random(size=[16, 16]).astype("float16")
    tvm_input = tvm_tirx.runtime.tensor(np_input, dev)
    tvm_output = vm["main"](tvm_input)
    np_expected = np_input.take([1], axis=axis)

    tvm_tirx.testing.assert_allclose(tvm_output.numpy(), np_expected)


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_take_2d_tensor_as_index(axis):
    """The index of R.take may be a 2-d tensor"""
    target = "llvm"
    dev = tvm_tirx.cpu()

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([16, 16], "float16")):
            output = R.take(A, R.const([[1, 3], [5, 7]]), axis=axis)
            return output

    built = tvm.compile(Module, target=target)
    vm = tvm.relax.VirtualMachine(built, dev)

    np_input = np.random.random(size=[16, 16]).astype("float16")
    tvm_input = tvm_tirx.runtime.tensor(np_input, dev)
    tvm_output = vm["main"](tvm_input)
    np_expected = np_input.take([[1, 3], [5, 7]], axis=axis)

    tvm_tirx.testing.assert_allclose(tvm_output.numpy(), np_expected)


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_take_constant_prim_value_as_index(axis):
    """The index of R.take may be a R.prim_value

    The `R.prim_value` produces output equivalent to a scalar
    tensor.

    """
    target = "llvm"
    dev = tvm_tirx.cpu()

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([16, 16], "float16")):
            output = R.take(A, R.prim_value(1), axis=axis)
            return output

    built = tvm.compile(Module, target=target)
    vm = tvm.relax.VirtualMachine(built, dev)

    np_input = np.random.random(size=[16, 16]).astype("float16")
    tvm_input = tvm_tirx.runtime.tensor(np_input, dev)
    tvm_output = vm["main"](tvm_input)
    np_expected = np_input.take(1, axis=axis)

    tvm_tirx.testing.assert_allclose(tvm_output.numpy(), np_expected)


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_take_dynamic_prim_value_as_index(axis):
    """The index of R.take may be a dynamic R.prim_value

    The `R.prim_value` produces output equivalent to a scalar
    tensor.

    """
    target = "llvm"
    dev = tvm_tirx.cpu()

    n = T.dynamic("n", "int64")

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([n, n], "float16")):
            output = R.take(A, R.prim_value(n - 1), axis=axis)
            return output

    built = tvm.compile(Module, target=target)
    vm = tvm.relax.VirtualMachine(built, dev)

    np_input = np.random.random(size=[16, 16]).astype("float16")
    tvm_input = tvm_tirx.runtime.tensor(np_input, dev)
    tvm_output = vm["main"](tvm_input)
    np_expected = np_input.take(15, axis=axis)

    tvm_tirx.testing.assert_allclose(tvm_output.numpy(), np_expected)


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_take_nan_mode_OOB_indices(axis):
    """Test R.take with mode="nan" and out-of-bounds indices.
    This test checks that out-of-bounds indices produce NaN values in the output tensor.
    """
    target = "llvm"
    dev = tvm_tirx.cpu()

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([3, 3], "float16")):
            output = R.take(A, R.const([0, 1, 2, 3]), axis=axis, mode="nan")
            return output

    built = tvm.compile(Module, target=target)
    vm = tvm.relax.VirtualMachine(built, dev)

    np_input = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype="float16")
    tvm_input = tvm_tirx.runtime.tensor(np_input, dev)
    tvm_output = vm["main"](tvm_input)
    if axis == 0:
        np_expected = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [np.nan, np.nan, np.nan]],
            dtype="float16",
        )
    elif axis == 1:
        np_expected = np.array(
            [[1.0, 2.0, 3.0, np.nan], [4.0, 5.0, 6.0, np.nan], [7.0, 8.0, 9.0, np.nan]],
            dtype="float16",
        )

    tvm_tirx.testing.assert_allclose(tvm_output.numpy(), np_expected)


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_take_wrap_mode_OOB_indices(axis):
    """Test R.take with mode="wrap" and out-of-bounds indices.
    This test checks that out-of-bounds indices wrap around to the valid range.
    """
    target = "llvm"
    dev = tvm_tirx.cpu()

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([3, 3], "float16")):
            output = R.take(A, R.const([0, 1, 2, 3]), axis=axis, mode="wrap")
            return output

    built = tvm.compile(Module, target=target)
    vm = tvm.relax.VirtualMachine(built, dev)

    np_input = np.random.random(size=[3, 3]).astype("float16")
    tvm_input = tvm_tirx.runtime.tensor(np_input, dev)
    tvm_output = vm["main"](tvm_input)
    np_expected = np.take(np_input, [0, 1, 2, 3], axis=axis, mode="wrap")

    tvm_tirx.testing.assert_allclose(tvm_output.numpy(), np_expected)


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_take_clip_mode_OOB_indices(axis):
    """Test R.take with mode="clip" and out-of-bounds indices.
    This test checks that out-of-bounds indices are clipped to the valid range.
    """
    target = "llvm"
    dev = tvm_tirx.cpu()

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([3, 3], "float16")):
            output = R.take(A, R.const([0, 1, 2, 3]), axis=axis, mode="clip")
            return output

    built = tvm.compile(Module, target=target)
    vm = tvm.relax.VirtualMachine(built, dev)
    np_input = np.random.random(size=[3, 3]).astype("float16")
    tvm_input = tvm_tirx.runtime.tensor(np_input, dev)
    tvm_output = vm["main"](tvm_input)
    np_expected = np.take(np_input, [0, 1, 2, 3], axis=axis, mode="clip")

    tvm_tirx.testing.assert_allclose(tvm_output.numpy(), np_expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
