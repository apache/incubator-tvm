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
# ruff: noqa: E501, F401
import numpy as np
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing

pytest.importorskip("scipy")  # tvm.topi.testing imports scipy

import tvm.topi.testing
from tvm import relax
from tvm.relax import script as R
from tvm.relax.transform import LegalizeOps
from tvm_tirx import tirx
from tvm_tirx.script import tirx as T

# TODO(tvm-team): `tirx.transform.DefaultGPUSchedule` does not work.
target, dev = "llvm", tvm_tirx.cpu()


def build(mod):
    exe = tvm.compile(mod, target=target)
    return relax.VirtualMachine(exe, dev)


@pytest.mark.parametrize(
    "begin, end, strides",
    [
        ([0, 2, 4, 4], [5, 5, 7, 8], [1, 1, 2, 3]),
        ([0, 2, 4, 4], [5, 5, 11, 10], [1, 1, 1, 1]),
        ([0, 2, 10, 14], [0, 5, 1, 1], [1, 1, -1, -2]),
        ([-2, 0, 0, 0], [8, 9, 10, 10], [1, 1, 1, 1]),
        ([-58, 0, 0, 0], [8, 9, 10, 10], [1, 1, 1, 1]),
        ([-1, 0, 0, 0], [-9, 9, 10, 10], [-1, 1, 1, 1]),
    ],
)
def test_dynamic_strided_slice(begin, end, strides):
    # fmt: off
    @tvm_tirx.script.ir_module
    class DynamicStridedSlice:
        @R.function
        def main(x: R.Tensor((8, 9, 10, 10), "float32"), begin: R.Tensor((4,),"int64"), end: R.Tensor((4,),"int64"), strides: R.Tensor((4,),"int64")) -> R.Tensor("float32", ndim=4):
            gv: R.Tensor("float32", ndim=4) = R.dynamic_strided_slice(x, begin, end, strides)
            return gv
    # fmt: on
    vm = build(DynamicStridedSlice)

    x_np = np.random.rand(8, 9, 10, 10).astype(np.float32)
    data_nd = tvm_tirx.runtime.tensor(x_np, dev)
    begin_nd = tvm_tirx.runtime.tensor(np.array(begin).astype("int64"), dev)
    end_nd = tvm_tirx.runtime.tensor(np.array(end).astype("int64"), dev)
    strides_nd = tvm_tirx.runtime.tensor(np.array(strides).astype("int64"), dev)

    # Reference implementation
    out_npy = tvm.topi.testing.strided_slice_python(x_np, begin, end, strides)
    out_nd = vm["main"](data_nd, begin_nd, end_nd, strides_nd)
    tvm_tirx.testing.assert_allclose(out_nd.numpy(), out_npy)


@pytest.mark.parametrize(
    "begin, end, strides",
    [
        ([0, 2, 4, 4], [5, 5, 7, 8], [1, 1, 2, 3]),
        ([0, 2, 4, 4], [5, 5, 11, 10], [1, 1, 1, 1]),
        ([0, 2, 10, 14], [0, 5, 1, 1], [1, 1, -1, -2]),
        ([-2, 0, 0, 0], [8, 9, 10, 10], [1, 1, 1, 1]),
        ([-58, 0, 0, 0], [8, 9, 10, 10], [1, 1, 1, 1]),
        ([-1, 0, 0, 0], [-9, 9, 10, 10], [-1, 1, 1, 1]),
    ],
)
def test_dynamic_strided_slice_symbolic(begin, end, strides):
    # fmt: off
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class DynamicStridedSlice:
        @R.function
        def main(x: R.Tensor((m, n, 10, 10), "float32"), begin: R.Tensor((4,),"int64"), end: R.Tensor((4,),"int64"), strides: R.Tensor((4,),"int64")) -> R.Tensor("float32", ndim=4):
            gv: R.Tensor("float32", ndim=4) = R.dynamic_strided_slice(x, begin, end, strides)
            return gv
    # fmt: on
    vm = build(DynamicStridedSlice)

    x_np = np.random.rand(8, 9, 10, 10).astype(np.float32)
    data_nd = tvm_tirx.runtime.tensor(x_np, dev)
    begin_nd = tvm_tirx.runtime.tensor(np.array(begin).astype("int64"), dev)
    end_nd = tvm_tirx.runtime.tensor(np.array(end).astype("int64"), dev)
    strides_nd = tvm_tirx.runtime.tensor(np.array(strides).astype("int64"), dev)

    # Reference implementation
    out_npy = tvm.topi.testing.strided_slice_python(x_np, begin, end, strides)
    out_nd = vm["main"](data_nd, begin_nd, end_nd, strides_nd)
    tvm_tirx.testing.assert_allclose(out_nd.numpy(), out_npy)


@pytest.mark.parametrize("begin_offset", [0, 1])
def test_strided_slice_symbolic_out_of_bounds(begin_offset):
    dim = tirx.Var("m", "int64")

    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorType([dim], "float32"))

    with bb.function("main", params=[x]):
        with bb.dataflow():
            y = bb.emit(
                relax.op.strided_slice(
                    x,
                    axes=[0],
                    begin=[dim + begin_offset],
                    end=[-dim - 1],
                    strides=[-1],
                )
            )
            gv = bb.emit_output(y)
        bb.emit_func_output(gv)

    vm = build(bb.get())

    x_np = np.arange(8, dtype=np.float32)
    out = vm["main"](tvm_tirx.runtime.tensor(x_np, dev)).numpy()

    tvm_tirx.testing.assert_allclose(out, x_np[::-1])


if __name__ == "__main__":
    tvm_tirx.testing.main()
