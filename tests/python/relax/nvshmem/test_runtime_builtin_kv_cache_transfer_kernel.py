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
import functools

import numpy as np
import pytest
import tvm_ffi
from tvm_ffi import Shape

import tvm_tirx
import tvm_tirx.testing
from tvm.runtime import disco as di
from tvm_tirx.runtime import Device

page_size = 4
num_layers = 4
num_kv_heads = 4
head_dim = 128
num_pages = 100
ntokens = 16


def get_comm_rank():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    return comm, rank


def _run_rank_group_with_gpu_lock(test):
    @functools.wraps(test)
    def wrapper():
        comm, rank = get_comm_rank()

        def run_and_check():
            comm.Barrier()
            try:
                return test()
            finally:
                comm.Barrier()

        if rank == 0:
            return tvm_tirx.testing.run_with_gpu_lock(run_and_check)
        return run_and_check()

    return wrapper


@pytest.mark.skip(reason="Require NVSHMEM")
@_run_rank_group_with_gpu_lock
def test_kv_transfer_without_disco():
    comm, rank = get_comm_rank()
    layer_id = 1
    dev = tvm_tirx.cuda(rank)
    if rank == 0:
        f_init_nvshmem_uid = tvm_ffi.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid")
        uid = f_init_nvshmem_uid()
    else:
        uid = None
    uid = comm.bcast(uid, root=0)
    init_func = tvm_ffi.get_global_func("runtime.disco.nvshmem.init_nvshmem")
    init_func(uid, 2, rank)
    empty_func = tvm_ffi.get_global_func("runtime.disco.nvshmem.empty")
    pages = empty_func(
        Shape((num_layers, num_pages, 2, num_kv_heads, page_size, head_dim)), "float16", dev
    )
    position_map_array = [0, 1, 2, 3, 4, 5, 10, 11, 12, 15, 16, 17, 18, 19, 25, 27]
    np.random.seed(0)
    k_np = np.random.rand(ntokens, num_kv_heads, head_dim).astype(np.float16)
    v_np = np.random.rand(ntokens, num_kv_heads, head_dim).astype(np.float16)
    if rank == 0:
        k = tvm_tirx.runtime.tensor(k_np, dev)
        v = tvm_tirx.runtime.tensor(v_np, dev)
        remote_position_map_np = np.array(position_map_array, dtype=np.int32)
        remote_position_map = tvm_tirx.runtime.tensor(remote_position_map_np, dev)
        remote_tp_group_pe_offset_np = np.array([1] * len(position_map_array), dtype=np.int32)
        remote_tp_group_pe_offset = tvm_tirx.runtime.tensor(remote_tp_group_pe_offset_np, dev)
        transfer_func = tvm_ffi.get_global_func("nvshmem.KVTransfer")
        layer_view = pages._create_view(
            [num_pages, 2, num_kv_heads, page_size, head_dim],
            "float16",
            relative_byte_offset=layer_id * num_pages * 2 * num_kv_heads * page_size * head_dim * 2,
        )
        transfer_func(layer_view, k, v, remote_position_map, remote_tp_group_pe_offset, None)
        dev.sync()
        comm.Barrier()
    else:
        comm.Barrier()
        pages_np = pages.numpy()
        for i, position in enumerate(position_map_array):
            page_id = position // page_size
            offset_in_page = position % page_size
            original_k = k_np[i]
            transferred_k = pages_np[layer_id, page_id, 0, :, offset_in_page, :]
            tvm_tirx.testing.assert_allclose(original_k, transferred_k)
            original_v = v_np[i]
            transferred_v = pages_np[layer_id, page_id, 1, :, offset_in_page, :]
            tvm_tirx.testing.assert_allclose(original_v, transferred_v)
    finalize_func = tvm_ffi.get_global_func("runtime.disco.nvshmem.finalize_nvshmem")
    finalize_func()
    comm.Barrier()


@pytest.mark.skip(reason="Require NVSHMEM")
@_run_rank_group_with_gpu_lock
def test_kv_transfer_page_to_page_without_disco():
    comm, rank = get_comm_rank()
    layer_id = 1
    dev = tvm_tirx.cuda(rank)
    if rank == 0:
        f_init_nvshmem_uid = tvm_ffi.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid")
        uid = f_init_nvshmem_uid()
    else:
        uid = None
    uid = comm.bcast(uid, root=0)
    init_func = tvm_ffi.get_global_func("runtime.disco.nvshmem.init_nvshmem")
    init_func(uid, 2, rank)
    empty_func = tvm_ffi.get_global_func("runtime.disco.nvshmem.empty")
    pages = empty_func(
        Shape((num_layers, num_pages, 2, num_kv_heads, page_size, head_dim)), "float16", dev
    )
    rank_1_position_map_array = [0, 1, 2, 3, 4, 5, 10, 11, 12, 15, 16, 17, 18, 19, 25, 27]
    rank_0_position_map_array = list(reversed(rank_1_position_map_array))
    np.random.seed(0)
    pages_np = np.random.rand(num_layers, num_pages, 2, num_kv_heads, page_size, head_dim).astype(
        np.float16
    )
    if rank == 0:
        pages.copyfrom(pages_np)
        remote_position_map_np = np.array(rank_1_position_map_array, dtype=np.int32)
        remote_position_map = tvm_tirx.runtime.tensor(remote_position_map_np, dev)
        local_position_map_np = np.array(rank_0_position_map_array, dtype=np.int32)
        local_position_map = tvm_tirx.runtime.tensor(local_position_map_np, dev)
        remote_tp_group_pe_offset_np = np.array(
            [1] * len(rank_0_position_map_array), dtype=np.int32
        )
        remote_tp_group_pe_offset = tvm_tirx.runtime.tensor(remote_tp_group_pe_offset_np, dev)
        transfer_func = tvm_ffi.get_global_func("nvshmem.KVTransferPageToPage")
        layer_view = pages._create_view(
            [num_pages, 2, num_kv_heads, page_size, head_dim],
            "float16",
            relative_byte_offset=layer_id * num_pages * 2 * num_kv_heads * page_size * head_dim * 2,
        )
        transfer_func(
            layer_view,
            layer_view,
            remote_position_map,
            local_position_map,
            remote_tp_group_pe_offset,
            None,
        )
        dev.sync()
        comm.Barrier()
    else:
        comm.Barrier()
        new_pages_np = pages.numpy()
        for i, position in enumerate(rank_1_position_map_array):
            page_id = position // page_size
            offset_in_page = position % page_size
            rank_0_position = rank_0_position_map_array[i]
            rank_0_page_id = rank_0_position // page_size
            rank_0_offset_in_page = rank_0_position % page_size
            rank_0_entry = pages_np[layer_id, rank_0_page_id, :, :, rank_0_offset_in_page, :]
            transferred_entry = new_pages_np[layer_id, page_id, :, :, offset_in_page, :]
            tvm_tirx.testing.assert_allclose(rank_0_entry, transferred_entry)
    finalize_func = tvm_ffi.get_global_func("runtime.disco.nvshmem.finalize_nvshmem")
    finalize_func()
    comm.Barrier()


@pytest.mark.skip(reason="Require NVSHMEM")
@_run_rank_group_with_gpu_lock
def test_kv_transfer_with_disco():
    comm, rank = get_comm_rank()
    layer_id = 1
    if rank == 0:
        f_init_nvshmem_uid = tvm_ffi.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid")
        uid = f_init_nvshmem_uid()
    else:
        uid = None
    uid = comm.bcast(uid, root=0)
    sess = di.ProcessSession(num_workers=2)
    init_func = sess.get_global_func("runtime.disco.nvshmem.init_nvshmem")
    init_func(uid, 4, rank * 2)
    empty_func = sess.get_global_func("runtime.disco.nvshmem.empty")
    pages = empty_func(
        Shape((num_layers, num_pages, 2, num_kv_heads, page_size, head_dim)),
        "float16",
        Device(device_type=0, device_id=0),
    )
    position_map_array = [0, 1, 2, 3, 4, 5, 10, 11, 12, 15, 16, 17, 18, 19, 25, 27]
    np.random.seed(0)
    k_np_0 = np.random.rand(ntokens, num_kv_heads, head_dim).astype(np.float16)
    v_np_0 = np.random.rand(ntokens, num_kv_heads, head_dim).astype(np.float16)
    np.random.seed(1)
    k_np_1 = np.random.rand(ntokens, num_kv_heads, head_dim).astype(np.float16)
    v_np_1 = np.random.rand(ntokens, num_kv_heads, head_dim).astype(np.float16)
    if rank == 0:
        k = sess.empty((ntokens, num_kv_heads, head_dim), "float16")
        v = sess.empty((ntokens, num_kv_heads, head_dim), "float16")
        k.debug_copy_from(0, k_np_0)
        k.debug_copy_from(1, k_np_1)
        v.debug_copy_from(0, v_np_0)
        v.debug_copy_from(1, v_np_1)
        remote_position_map_np = np.array(position_map_array, dtype=np.int32)
        remote_position_map = sess.empty((len(position_map_array),), "int32")
        remote_tp_group_pe_offset_np = np.array([2] * len(position_map_array), dtype=np.int32)
        remote_tp_group_pe_offset = sess.empty((len(remote_tp_group_pe_offset_np),), "int32")
        f_view_func = sess.get_global_func("runtime.TensorCreateView")
        layer_view = f_view_func(
            pages,
            Shape([num_pages, 2, num_kv_heads, page_size, head_dim]),
            "float16",
            layer_id * num_pages * 2 * num_kv_heads * page_size * head_dim * 2,
        )
        remote_position_map.debug_copy_from(0, remote_position_map_np)
        remote_position_map.debug_copy_from(1, remote_position_map_np)
        remote_tp_group_pe_offset.debug_copy_from(0, remote_tp_group_pe_offset_np)
        remote_tp_group_pe_offset.debug_copy_from(1, remote_tp_group_pe_offset_np)
        transfer_func = sess.get_global_func("nvshmem.KVTransfer")
        transfer_func(layer_view, k, v, remote_position_map, remote_tp_group_pe_offset, None)
        for i in range(2):
            sess._sync_worker(i)
        for i in range(2):
            tvm_tirx.cuda(i).sync()
        comm.Barrier()
    else:
        comm.Barrier()
        pages_np = pages.debug_get_from_remote(0).numpy()
        for i, position in enumerate(position_map_array):
            page_id = position // page_size
            offset_in_page = position % page_size
            original_k = k_np_0[i]
            transferred_k = pages_np[layer_id, page_id, 0, :, offset_in_page, :]
            tvm_tirx.testing.assert_allclose(original_k, transferred_k)
            original_v = v_np_0[i]
            transferred_v = pages_np[layer_id, page_id, 1, :, offset_in_page, :]
            tvm_tirx.testing.assert_allclose(original_v, transferred_v)
        pages_np = pages.debug_get_from_remote(1).numpy()
        for i, position in enumerate(position_map_array):
            page_id = position // page_size
            offset_in_page = position % page_size
            original_k = k_np_1[i]
            transferred_k = pages_np[layer_id, page_id, 0, :, offset_in_page, :]
            tvm_tirx.testing.assert_allclose(original_k, transferred_k)
            original_v = v_np_1[i]
            transferred_v = pages_np[layer_id, page_id, 1, :, offset_in_page, :]
            tvm_tirx.testing.assert_allclose(original_v, transferred_v)
    finalize_dfunc = sess.get_global_func("runtime.disco.nvshmem.finalize_nvshmem")
    finalize_dfunc()
    for i in range(2):
        sess._sync_worker(i)
    sess.shutdown()


if __name__ == "__main__":
    tvm_tirx.testing.main()
