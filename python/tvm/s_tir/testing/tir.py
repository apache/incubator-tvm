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
# pylint: disable=invalid-name, import-outside-toplevel, unused-variable
"""Common utility functions in TVM tirx"""


def mfma_schedule(
    workload,
    k_inner,
    in_dtype,
    b_transposed,
    i_factors,
    j_factors,
    k_factors,
    index_map_A,
    index_map_B,
    index_map_C,
    ldmatrix_a_intrin,
    ldmatrix_b_intrin,
    mfma_intrin,
    mfma_fill_intrin,
    mfma_store_intrin,
    shared_scope="shared",
):
    """Create a tensorized schedule for GEMM with MFMA intrinsics."""
    import tvm
    import tvm_tirx

    ir_module = tvm_tirx.IRModule({"main": workload})
    sch = tvm.s_tir.Schedule(ir_module)

    wmma_m = 16
    wmma_n = 16
    wmma_k = k_inner
    warp_size = 64
    block = sch.get_sblock("C")
    i, j, k = sch.get_loops(block)
    i, i_tc = sch.split(i, factors=[None, wmma_m])
    j, j_tc = sch.split(j, factors=[None, wmma_n])
    k, k_tc = sch.split(k, factors=[None, wmma_k])

    sch.reorder(i, j, k, i_tc, j_tc, k_tc)

    block_inner = sch.blockize(i_tc)
    block_outer, block_inner = block_inner, block

    num_ty = i_factors[2] * j_factors[2]

    i0, i1, i2, i3, i4 = sch.split(i, factors=i_factors)
    j0, j1, j2, j3, j4 = sch.split(j, factors=j_factors)
    k0, k1, k2 = sch.split(k, k_factors)

    sch.reorder(i0, j0, i1, j1, j2, i2, k0, k1, i3, j3, k2, i4, j4)

    block_idx = sch.fuse(i0, j0)
    block_idy = sch.fuse(i1, j1)
    thread_idy = sch.fuse(j2, i2)
    sch.bind(block_idx, "blockIdx.x")
    sch.bind(block_idy, "blockIdx.y")
    sch.bind(thread_idy, "threadIdx.y")

    def fetch_to_shared(block, idx, ndim):
        block_read = sch.cache_read(block, idx, shared_scope)
        sch.compute_at(block_read, k0)
        vector_size = 16 if in_dtype == "int8" else 8
        fused = sch.fuse(*sch.get_loops(block_read)[-ndim:])
        _, f_1, f_2, f_3 = sch.split(fused, factors=[None, num_ty, warp_size, vector_size])
        sch.bind(f_2, "threadIdx.x")
        sch.bind(f_1, "threadIdx.y")
        sch.vectorize(f_3)
        return block_read

    fetch_to_shared(block_outer, 0, 2)
    fetch_to_shared(block_outer, 1, 2)

    # Keep the logical lane axis until tensorization binds per-lane local fragments.
    A_warp = sch.cache_read(block_outer, 0, "local")
    B_warp = sch.cache_read(block_outer, 1, "local")

    sch.compute_at(A_warp, k1)
    sch.compute_at(B_warp, k1)
    C_warp = sch.cache_write(block_outer, 0, "local")
    sch.reverse_compute_at(C_warp, thread_idy)

    ii, jj = sch.get_loops(C_warp)[-2:]
    io, ii = sch.split(ii, factors=[None, 16])
    jo, ji = sch.split(jj, factors=[None, 16])
    sch.reorder(io, jo, ii, ji)

    sch.decompose_reduction(block_outer, sch.get_loops(block_outer)[3])
    block_init_c = sch.get_sblock("C_init")

    def tile_wmma_fragment(block_read, height, width):
        i, j = sch.get_loops(block_read)[-2:]
        i0, i1 = sch.split(i, factors=[None, height])
        j0, j1 = sch.split(j, factors=[None, width])
        sch.reorder(i0, j0, i1, j1)
        return i1

    loop_a = tile_wmma_fragment(A_warp, 16, k_inner)

    if b_transposed:
        loop_b = tile_wmma_fragment(B_warp, 16, k_inner)
    else:
        loop_b = tile_wmma_fragment(B_warp, k_inner, 16)

    sch.transform_layout(A_warp, ("write", 0), index_map_A)
    sch.transform_layout(B_warp, ("write", 0), index_map_B)
    sch.transform_layout(C_warp, ("read", 0), index_map_C)

    sch.tensorize(loop_a, ldmatrix_a_intrin)
    sch.tensorize(loop_b, ldmatrix_b_intrin)
    sch.tensorize(sch.get_loops(block_inner)[-3], mfma_intrin)
    sch.tensorize(sch.get_loops(block_init_c)[-2], mfma_fill_intrin)
    sch.tensorize(sch.get_loops(C_warp)[-2], mfma_store_intrin)

    return sch
