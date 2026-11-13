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
"""Regression tests for median legalization."""

import numpy as np
import pytest

import tvm
import tvm_tirx
from tvm import relax


def make_median(shape, keepdims):
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorType(shape, "float32"))
    with bb.function("main", [x]):
        with bb.dataflow():
            result = bb.emit(relax.op.median(x, axis=None, keepdims=keepdims))
            result = bb.emit_output(result)
        bb.emit_func_output(result)
    return bb.get()


@pytest.mark.parametrize("shape", [(), (5,), (2, 3), (2, 3, 4)])
@pytest.mark.parametrize("keepdims", [False, True])
def test_median_all_axes_preserves_inferred_return_type(shape, keepdims):
    original = make_median(shape, keepdims)
    lowered = relax.transform.LegalizeOps()(original)
    tvm_tirx.ir.assert_structural_equal(original["main"].ret_ty, lowered["main"].ret_ty)


@pytest.mark.skipif(not tvm_tirx.runtime.enabled("llvm"), reason="LLVM not enabled")
@pytest.mark.parametrize(
    "shape,keepdims", [((), True), ((2, 3), True), ((2, 3, 4), True), ((2, 3), False)]
)
def test_median_all_axes_runtime_shape_and_value(shape, keepdims):
    data = np.arange(int(np.prod(shape)), dtype="float32")[::-1].reshape(shape)
    target = tvm_tirx.target.Target({"kind": "llvm", "mcpu": "generic"})
    with target:
        exe = tvm.compile(make_median(shape, keepdims), target=target)
    dev = tvm_tirx.cpu(0)
    vm = relax.VirtualMachine(exe, dev)
    result = vm["main"](tvm_tirx.runtime.tensor(data, dev)).numpy()
    expected = np.asarray(np.sort(data.ravel())[(data.size - 1) // 2], dtype="float32")
    expected = expected.reshape([1] * len(shape) if keepdims else ())
    assert result.shape == expected.shape
    np.testing.assert_array_equal(result, expected)
