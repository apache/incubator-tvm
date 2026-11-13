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
# ruff: noqa: E741
"""Configure pytest"""

# pylint: disable=invalid-name
import numpy as np

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import te, topi


def test_sort():
    """Tests sort function"""
    n = 2
    l = 5
    m = 3
    data = te.placeholder((n, l, m), name="data")
    sort_num = te.placeholder((n, m), name="sort_num", dtype="int32")
    axis = 1
    is_ascend = False
    out = te.extern(
        data.shape,
        [data, sort_num],
        lambda ins, outs: tvm_tirx.tirx.call_packed(
            "tvm.contrib.sort.argsort_nms", ins[0], ins[1], outs[0], axis, is_ascend
        ),
        dtype="int32",
        name="sort_tensor",
    )
    input_data = [
        [[1, 2, 3], [2, 4.5, 3.5], [1.1, 0.5, 1], [3.2, -5, 0.5], [1.5, 0, 0]],
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
    ]
    sort_num_input = [[1, 2, 3], [4, 5, 5]]
    sorted_index = [
        [[0, 1, 1], [1, 0, 0], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
        [[3, 4, 4], [2, 3, 3], [1, 2, 2], [0, 1, 1], [4, 0, 0]],
    ]

    dev = tvm_tirx.cpu(0)
    target = "llvm"
    f = tvm.compile(te.create_function([data, sort_num, out]), target=target)
    a = tvm_tirx.runtime.tensor(np.array(input_data).astype(data.dtype.dtype), dev)
    b = tvm_tirx.runtime.tensor(np.array(sort_num_input).astype(sort_num.dtype.dtype), dev)
    c = tvm_tirx.runtime.tensor(np.zeros(a.shape, dtype=out.dtype.dtype), dev)
    f(a, b, c)
    tvm_tirx.testing.assert_allclose(
        c.numpy(), np.array(sorted_index).astype(out.dtype.dtype), rtol=1e-5
    )


def test_sort_np():
    """Tests sort function using numpy"""
    dshape = (1, 2, 3, 4, 5, 6)
    axis = 4
    reduced_shape = (1, 2, 3, 4, 6)
    is_ascend = True
    data = te.placeholder(dshape, name="data")
    sort_num = te.placeholder(reduced_shape, name="sort_num", dtype="int32")
    out = te.extern(
        data.shape,
        [data, sort_num],
        lambda ins, outs: tvm_tirx.tirx.call_packed(
            "tvm.contrib.sort.argsort_nms", ins[0], ins[1], outs[0], axis, is_ascend
        ),
        dtype="int32",
        name="sort_tensor",
    )

    dev = tvm_tirx.cpu(0)
    target = "llvm"
    f = tvm.compile(te.create_function([data, sort_num, out]), target=target)

    np_data = np.random.uniform(size=dshape)
    np_out = np.argsort(np_data, axis=axis)
    sort_num_input = np.full(reduced_shape, dshape[axis])
    a = tvm_tirx.runtime.tensor(np.array(np_data).astype(data.dtype.dtype), dev)
    b = tvm_tirx.runtime.tensor(np.array(sort_num_input).astype(sort_num.dtype.dtype), dev)
    c = tvm_tirx.runtime.tensor(np.zeros(a.shape, dtype=out.dtype.dtype), dev)
    f(a, b, c)
    tvm_tirx.testing.assert_allclose(c.numpy(), np_out, rtol=1e-5)


def test_sort_none_index_dtype_defaults_to_float32():
    data = te.placeholder((8,), dtype="int32")
    indices = topi.argsort(data, dtype=None)
    top_indices = topi.topk(data, k=2, ret_type="indices", dtype=None)
    assert indices.dtype == "float32"
    assert top_indices.dtype == "float32"


if __name__ == "__main__":
    test_sort()
    test_sort_np()
