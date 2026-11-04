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
import tvm_ffi

import tvm
import tvm_tirx
from tvm.s_tir import script as Ts
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env


def make_copy(src_stride, dst_stride):
    @Ts.function
    def copy(A: T.Tensor((32, 16), "float32"), B: T.Tensor((32, 4), "float32")):
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 32)
        with Ts.sblock():
            shared = Ts.sblock_alloc_tensor([32, 16], "float32", scope="shared")
            Ts.reads(A[0:32, 0:16])
            Ts.writes(B[0:32, 0:4])
            with Ts.async_copy_scope():
                shared[tx, T.Ramp(0, dst_stride, 4)] = A[tx, T.Ramp(0, src_stride, 4)]
                T.evaluate(T.ptx.cp.async_.commit_group())
                T.evaluate(T.ptx.cp.async_.wait_group(0))
                for j in range(4):
                    B[tx, j] = shared[tx, j * dst_stride]

    return copy


@pytest.mark.parametrize(
    "src_stride,dst_stride,expected_count",
    [(1, 1, 1), (2, 1, 0), (1, 2, 0), (2, 2, 0)],
)
def test_async_copy_requires_contiguous_vectors(src_stride, dst_stride, expected_count):
    mod = tvm_tirx.IRModule.from_expr(make_copy(src_stride, dst_stride))
    for transform in (
        tvm.s_tir.transform.LowerOpaqueBlock(),
        tvm.s_tir.transform.LowerThreadBinding(),
        tvm_tirx.tirx.transform.FlattenTensor(),
        tvm_tirx.tirx.transform.VectorizeLoop(),
        tvm.s_tir.transform.InjectPTXAsyncCopy(),
    ):
        mod = transform(mod)

    calls = []

    def visit(node):
        if isinstance(node, tvm_tirx.ir.Call) and node.op.name == "tirx.s_tir.cp_async_raw":
            calls.append(node)

    tvm_ffi.structural_walk(mod["main"].body, visit)
    assert len(calls) == expected_count


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("src_stride,dst_stride", [(2, 1), (1, 2), (2, 2)])
def test_noncontiguous_async_copy_values(src_stride, dst_stride):
    with tvm_tirx.ir.transform.PassContext(config={"tirx.use_async_copy": 1}):
        exe = tvm.compile(
            tvm_tirx.IRModule.from_expr(make_copy(src_stride, dst_stride)), target="cuda"
        )

    data = np.arange(32 * 16, dtype="float32").reshape(32, 16)
    expected = data[:, ::src_stride][:, :4]

    def run_and_check():
        dev = tvm_tirx.cuda()
        source = tvm_tirx.runtime.tensor(data, dev)
        result = tvm_tirx.runtime.tensor(np.full((32, 4), -1, dtype="float32"), dev)
        exe(source, result)
        np.testing.assert_array_equal(result.numpy(), expected)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)
