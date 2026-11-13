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

"""TOPI imperative kernels construct and execute explicit tensor stores."""

import numpy as np
import pytest
import tvm_ffi

import tvm_tirx
import tvm_tirx.runtime
import tvm_tirx.testing
from tvm import te, topi
from tvm_tirx import tirx


def _check_stores(output):
    outputs = output if isinstance(output, tuple | list | tvm_ffi.Array) else [output]
    stores = []

    for tensor in outputs:
        tvm_ffi.structural_walk(
            tensor.op.body,
            lambda node: stores.append(node) if isinstance(node, tvm_tirx.ir.TensorStore) else None,
        )
    assert stores
    assert all(len(store.indices) == len(store.dest.shape) for store in stores)


@pytest.mark.parametrize("mode", ["update", "add", "mul", "min", "max"])
def test_imperative_scatter_nd(mode):
    _check_stores(
        topi.scatter_nd(
            te.placeholder((2, 3), "float32"),
            te.placeholder((1, 2), "int32"),
            te.placeholder((2, 3), "float32"),
            mode,
        )
    )


@pytest.mark.parametrize("mode", ["update", "add", "mul", "mean", "min", "max"])
def test_imperative_scatter_elements(mode):
    _check_stores(
        topi.scatter_elements(
            te.placeholder((2, 3), "float32"),
            te.placeholder((2, 2), "int32"),
            te.placeholder((2, 2), "float32"),
            axis=1,
            reduction=mode,
        )
    )


@pytest.mark.parametrize("accumulate", [False, True])
def test_imperative_index_put(accumulate):
    _check_stores(
        topi.index_put(
            te.placeholder((2, 3), "float32"),
            [te.placeholder((2,), "int32"), te.placeholder((2,), "int32")],
            te.placeholder((2,), "float32"),
            accumulate=accumulate,
        )
    )


def test_imperative_search_scan_signal_and_unique():
    data = te.placeholder((2, 4), "float32")
    _check_stores(topi.searchsorted(data, te.placeholder((2, 3), "float32")))
    _check_stores(topi.cumsum(data, axis=1))
    _check_stores(topi.cumprod(data, axis=None))
    _check_stores(topi.signal.dft(data, data, inverse=False))
    _check_stores(
        topi.signal.stft(
            data,
            4,
            1,
            4,
            te.placeholder((4,), "float32"),
            False,
            True,
            (2, 3, 1, 2),
        )
    )
    from tvm.topi.unique import _calc_adjacent_diff

    _check_stores(_calc_adjacent_diff(te.placeholder((4,), "int32")))


def test_imperative_sparse_reshape():
    _check_stores(
        topi.sparse_reshape(
            te.placeholder((3, 2), "int64"),
            te.placeholder((2,), "int64"),
            te.placeholder((2,), "int64"),
            (3, 2),
            (2,),
        )
    )


def test_imperative_nms():
    data = te.placeholder((2, 4, 6), "float32")
    valid_count, boxes, indices = topi.vision.get_valid_counts(data)
    _check_stores([valid_count, boxes, indices])
    _check_stores(
        topi.vision.non_max_suppression(
            data,
            te.placeholder((2,), "int32"),
            te.placeholder((2, 4), "int32"),
            max_output_size=3,
            return_indices=False,
        )
    )


@pytest.mark.parametrize(
    "kind", ["sort", "argsort", "searchsorted", "scan", "scatter_nd", "scatter_elements"]
)
def test_gpu_imperative_tensors(kind):
    data = te.placeholder((2, 4), "float32")
    if kind == "sort":
        result = topi.gpu.sort(data, target=tvm_tirx.target.Target("cuda"))
    elif kind == "argsort":
        result = topi.gpu.argsort(data, target=tvm_tirx.target.Target("cuda"))
    elif kind == "searchsorted":
        result = topi.gpu.searchsorted(
            data, te.placeholder((2, 3), "float32"), target=tvm_tirx.target.Target("cuda")
        )
    elif kind == "scan":
        from tvm.topi.gpu.scan import exclusive_scan_ir

        body = exclusive_scan_ir(
            tirx.Var("tensor", tirx.TensorType((2, 4), "float32")),
            tirx.Var("tensor", tirx.TensorType((2, 4), "float32")),
            target=tvm_tirx.target.Target("cuda"),
        )
        assert isinstance(body, tvm_tirx.ir.Stmt)
        return
    elif kind == "scatter_nd":
        result = topi.gpu.scatter_nd(
            data,
            te.placeholder((1, 2), "int32"),
            te.placeholder((2, 4)),
            "add",
            target=tvm_tirx.target.Target("cuda"),
        )
    else:
        result = topi.gpu.scatter_elements(
            data,
            te.placeholder((2, 3), "int32"),
            te.placeholder((2, 3)),
            axis=1,
            reduction="add",
            target=tvm_tirx.target.Target("cuda"),
        )
    _check_stores(result)


@pytest.mark.skipif(not tvm_tirx.runtime.enabled("llvm"), reason="LLVM is not enabled")
@pytest.mark.parametrize(
    "kind",
    [
        "searchsorted",
        "scan",
        "scatter",
        "scatter_elements",
        "index_put",
        "dft",
        "sparse_reshape",
        "unique",
    ],
)
def test_imperative_tensors_numerical(kind):
    data = np.arange(6, dtype="float32").reshape(2, 3)
    if kind == "searchsorted":
        arrays = [data, np.array([[0.5, 2, 3], [2, 4, 6]], dtype="float32")]

        def build(x, y):
            return topi.searchsorted(x, y, right=True)

        expected = [
            np.stack([np.searchsorted(row, values, side="right") for row, values in zip(*arrays)])
        ]
    elif kind == "scan":
        arrays = [data]

        def build(x):
            return topi.cumsum(x, axis=1)

        expected = [np.cumsum(data, axis=1)]
    elif kind == "scatter":
        arrays = [data, np.array([[1, 0]], dtype="int32"), np.ones((2, 3), dtype="float32")]

        def build(x, i, u):
            return topi.scatter_nd(x, i, u, "add")

        expected = [data + 1]
    elif kind == "scatter_elements":
        arrays = [data, np.array([[0, 2], [2, 0]], dtype="int32"), np.ones((2, 2), dtype="float32")]

        def build(x, i, u):
            return topi.scatter_elements(x, i, u, axis=1, reduction="add")

        result = data.copy()
        np.add.at(result, (np.arange(2)[:, None], arrays[1]), arrays[2])
        expected = [result]
    elif kind == "index_put":
        arrays = [
            data,
            np.array([0, 1], dtype="int32"),
            np.array([2, 0], dtype="int32"),
            np.array([10, 20], dtype="float32"),
        ]

        def build(x, i, j, v):
            return topi.index_put(x, [i, j], v, accumulate=True)

        result = data.copy()
        np.add.at(result, (arrays[1], arrays[2]), arrays[3])
        expected = [result]
    elif kind == "dft":
        arrays = [data, np.zeros_like(data)]

        def build(x, y):
            return topi.signal.dft(x, y, inverse=False)

        result = np.fft.fft(data, axis=-1)
        expected = [result.real, result.imag]
    elif kind == "sparse_reshape":
        arrays = [
            np.array([[0, 0], [1, 0], [1, 2]], dtype="int64"),
            np.array([2, 3], dtype="int64"),
            np.array([3, 2], dtype="int64"),
        ]

        def build(i, old, new):
            return topi.sparse_reshape(i, old, new, (3, 2), (2,))

        expected = [np.array([[0, 0], [1, 1], [2, 1]], dtype="int64"), arrays[2]]
    else:
        from tvm.topi.unique import _calc_adjacent_diff

        arrays = [np.array([3, 5, 5, 8], dtype="int32")]
        build = _calc_adjacent_diff
        expected = [np.array([0, 2, 0, 3], dtype="int32")]
    inputs = [te.placeholder(array.shape, str(array.dtype)) for array in arrays]
    outputs = build(*inputs)
    outputs = list(outputs) if isinstance(outputs, list | tuple | tvm_ffi.Array) else [outputs]
    executable = tvm_tirx.tirx.compile(te.create_function([*inputs, *outputs]), target="llvm")
    actual = [
        tvm_tirx.runtime.empty(tuple(int(dim) for dim in out.shape), out.dtype) for out in outputs
    ]
    executable(*[tvm_tirx.runtime.tensor(array) for array in arrays], *actual)
    for result, reference in zip(actual, expected):
        tvm_tirx.testing.assert_allclose(result.numpy(), reference, atol=1e-5, rtol=1e-5)


def test_topi_tan_requires_float_dtype():
    x = te.placeholder((2, 2), dtype="int32", name="x")

    with pytest.raises(TypeError, match=r"prim\.tan only supports floating-point inputs"):
        topi.tan(x)


if __name__ == "__main__":
    tvm_tirx.testing.main()
