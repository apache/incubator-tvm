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
import tvm_tirx
from tvm import s_tir
from tvm.s_tir import script as Ts
from tvm_tirx.script import tirx as T


@Ts.function
def tensor_load_store_func(
    A: T.Tensor((128, 128), "float32"), B: T.Tensor((128, 128), "float32")
) -> None:
    C = Ts.sblock_alloc_tensor((128, 128), "float32")
    D = Ts.sblock_alloc_tensor((128, 128), "float32")
    for ii, jj in T.grid(128, 128):
        with Ts.sblock():
            i, j = Ts.axis.remap("SS", [ii, jj])
            A[i, j] = T.float32(0)
    for i0, j0, k0 in T.grid(32, 32, 32):
        with Ts.sblock():
            i, j, k = Ts.axis.remap("SSR", [i0, j0, k0])
            with Ts.init():
                for ii, jj in T.grid(4, 4):
                    B[i * 4 + ii, j * 4 + jj] = A[i * 4 + ii, j * 4 + jj]
            for ii, jj in T.grid(4, 4):
                for kk in range(0, 4):
                    B[i * 4 + ii, j * 4 + jj] += C[i * 4 + ii, k * 4 + kk]
                for kk in range(0, 4):
                    B[i * 4 + ii, j * 4 + jj] += (
                        D[j * 4 + jj, k * 4 + kk] * C[i * 4 + ii, k * 4 + kk]
                    )


@Ts.function
def tensor_opaque_access(
    B: T.Tensor([16, 16], "float32"), C: T.Tensor([16, 16], "float32")
) -> None:
    with Ts.sblock():
        Ts.reads([])
        Ts.writes(B[0:16, 0:16])
        A = T.alloc_tensor(ty_args=[T.Tensor([256], "float32")])
        for i, j in T.grid(16, 16):
            A[i * 16 + j] = 1
        for i in range(0, 16):
            for j in range(0, 16):
                T.evaluate(A[i * 16 + j])
            for j in range(0, 16):
                T.evaluate(T.gpu_fill_fragment(B.data, 16, 16, 16, 0, T.float32(0)))

    for i, j in T.grid(16, 16):
        with Ts.sblock():
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj]


@Ts.function
def lca_is_func_root(A: T.Tensor([0, 0], "float32")) -> None:
    A[0, 0] = 1.0


@Ts.function
def match_tensor_func(
    A: T.Tensor((128, 128), "float32"), B: T.Tensor((128, 128), "float32")
) -> None:
    for i, j in T.grid(8, 8):
        with Ts.sblock("block"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads(B[vi * 16 + 2 : vi * 16 + 12, vj * 16 + 2 : vj * 16 + 16])
            Ts.writes(A[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
            B0 = Ts.match_tensor(B[vi * 16 + 2 : vi * 16 + 6, vj * 16 + 2 : vj * 16 + 6], (4, 4))
            B1 = Ts.match_tensor(B[vi * 16 + 8 : vi * 16 + 12, vj * 16 + 8 : vj * 16 + 16], (4, 8))
            for ii, jj in T.grid(16, 16):
                with Ts.sblock("AAA"):
                    vii, vjj = Ts.axis.remap("SS", [ii, jj])
                    AA = Ts.match_tensor(A[vii, vjj], ())
                    AA[()] = 1.0
            T.evaluate(B0.data)
            T.evaluate(B1.data)


@Ts.function
def global_tensor_with_blockidx(
    a: T.Tensor((1, 32), "int32"), b: T.Tensor((1, 32), "int32")
) -> None:
    for i0 in T.parallel(0, 1, annotations={"thread_binding": "blockIdx.x"}):
        for i1 in T.parallel(0, 32, annotations={"thread_binding": "threadIdx.x"}):
            with Ts.sblock("copy"):
                i, j = Ts.axis.remap("SS", [i0, i1])
                Ts.reads(a[i, j])
                Ts.writes(b[i, j])
                b[i, j] = a[i, j]


def test_tensor_load_store():
    func = tensor_load_store_func
    A, B = [x for x in func.params if tvm_tirx.tirx.is_tensor_var(x)]
    C, D = func.body[0].block.alloc_tensors
    lca = s_tir.analysis.detect_tensor_access_lca(func)

    # LCA of Tensor A is root
    root_block = func.body[0].block
    assert lca[A] == func.body[0].block

    # LCA of Tensor B is the loop dominate all reduction loop
    reduce_dom_loop = root_block.body[1].body[0]
    reduce_block = reduce_dom_loop.body[0].body[0].block
    assert lca[B] == reduce_dom_loop

    # LCA of Tensor C is the second loop kk
    loop_jj = reduce_block.body[0].body[0]
    assert lca[C] == loop_jj

    # LCA of Tensor D is loop jj
    loop_kk = loop_jj.body[1]
    assert lca[D] == loop_kk


def test_opaque_access():
    func = tensor_opaque_access
    B, C = [x for x in func.params if tvm_tirx.tirx.is_tensor_var(x)]
    lca = s_tir.analysis.detect_tensor_access_lca(func)

    # Cannot detect tensor A since it is define by low-level Allocate

    # LCA of Tensor B is root
    root_block = func.body[0].block
    assert lca[B] == func.body[0].block

    # LCA of Tensor C is the correspond block
    assert lca[C] == root_block.body[1].body[0].body[0].block


def test_lca_func_root():
    func = lca_is_func_root
    (A,) = [x for x in func.params if tvm_tirx.tirx.is_tensor_var(x)]
    lca = s_tir.analysis.detect_tensor_access_lca(func)
    assert lca[A] is None


def test_match_tensor():
    func = match_tensor_func
    A, B = [x for x in func.params if tvm_tirx.tirx.is_tensor_var(x)]
    lca = s_tir.analysis.detect_tensor_access_lca(func)

    root_block = func.body[0].block
    block = root_block.body[0].body[0].body[0].block
    block_inner = block.body[0].body[0].body[0].block

    # LCA of Tensor C is the inner block
    assert lca[A] == block_inner

    # LCA of Tensor C is the main block
    assert lca[B] == block


def test_global_tensor_with_blockidx():
    func = global_tensor_with_blockidx
    A, B = [x for x in func.params if tvm_tirx.tirx.is_tensor_var(x)]
    lca = s_tir.analysis.detect_tensor_access_lca(func)

    root_block = func.body[0].block
    blockidx_loop = root_block.body[0]
    # LCA of both A and B should be the loop bound to `blockIdx`
    assert lca[A] == blockidx_loop
    assert lca[B] == blockidx_loop


if __name__ == "__main__":
    test_tensor_load_store()
    test_opaque_access()
    test_lca_func_root()
    test_match_tensor()
    test_global_tensor_with_blockidx()
