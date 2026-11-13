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
import tvm_ffi

import tvm
import tvm.s_tir.testing
import tvm_tirx
import tvm_tirx.testing
from tvm import s_tir
from tvm.s_tir import match_shape_stride_pattern
from tvm.s_tir import script as Ts
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env


def count_cp_async(stmt):
    num_alloc = [0]

    def verify(n):
        if isinstance(n, tvm_tirx.ir.Call) and n.op.name == "tirx.s_tir.cp_async_raw":
            num_alloc[0] += 1

    tvm_ffi.structural_walk(stmt, verify)
    return num_alloc[0]


def generate_global_to_shared_vectorized_copy(dtype, vector_size):
    num_iters = 128 // vector_size
    vector_size_expr = tvm_ffi.convert(vector_size)

    @Ts.function
    def ptx_global_to_shared_copy(
        A: T.Tensor((32, 128), dtype), B: T.Tensor((32, 128), dtype)
    ) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 32)
        with Ts.sblock():
            A_shared = Ts.sblock_alloc_tensor([32, 128], dtype, scope="shared")
            Ts.reads(A[0:32, 0:128])
            Ts.writes(B[0:32, 0:128])

            with Ts.async_copy_scope():
                for i in T.serial(num_iters):
                    for j in T.vectorized(vector_size):
                        A_shared[tx, i * vector_size_expr + j] = A[tx, i * vector_size_expr + j]

                T.evaluate(T.ptx.cp.async_.commit_group())
                T.evaluate(T.ptx.cp.async_.wait_group(0))

                for i in range(128):
                    B[tx, i] = A_shared[tx, i]

    return ptx_global_to_shared_copy


@Ts.function
def ptx_global_to_shared_copy_fp32x1(
    A: T.Tensor((32, 128), "float32"), B: T.Tensor((32, 128), "float32")
) -> None:
    T.func_attr({"global_symbol": "main", "tirx.noalias": True})
    T.launch_thread("blockIdx.x", 1)
    tx = T.launch_thread("threadIdx.x", 32)
    with Ts.sblock():
        A_shared = Ts.sblock_alloc_tensor([32, 128], "float32", scope="shared")
        Ts.reads(A[0:32, 0:128])
        Ts.writes(B[0:32, 0:128])

        with Ts.async_copy_scope():
            for i in T.serial(128):
                A_shared[tx, i] = A[tx, i]

            T.evaluate(T.ptx.cp.async_.commit_group())
            T.evaluate(T.ptx.cp.async_.wait_group(0))

            for i in range(128):
                B[tx, i] = A_shared[tx, i]


@Ts.function
def ptx_global_to_shared_dyn_copy_fp16x8(
    A: T.Tensor((32, 128), "float16"),
    B: T.Tensor((32, 128), "float16"),
    C: T.Tensor((32, 128), "float16"),
) -> None:
    T.func_attr({"global_symbol": "main", "tirx.noalias": True})
    T.launch_thread("blockIdx.x", 1)
    tx = T.launch_thread("threadIdx.x", 32)
    with Ts.sblock():
        A_shared = Ts.sblock_alloc_tensor([32, 128], "float16", scope="shared.dyn")
        B_shared = Ts.sblock_alloc_tensor([32, 128], "float16", scope="shared.dyn")
        Ts.reads(A[0:32, 0:128], B[0:32, 0:128])
        Ts.writes(C[0:32, 0:128])

        with Ts.async_copy_scope():
            for i in T.serial(16):
                for j in T.vectorized(8):
                    A_shared[tx, i * 8 + j] = A[tx, i * 8 + j]
                    B_shared[tx, i * 8 + j] = B[tx, i * 8 + j]

            T.evaluate(T.ptx.cp.async_.commit_group())
            T.evaluate(T.ptx.cp.async_.wait_group(0))

            for i in range(128):
                C[tx, i] = A_shared[tx, i] + B_shared[tx, i]


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_inject_async_copy():
    for dtype, vec_size in [("float16", 8), ("float16", 4), ("float32", 4), ("float32", 1)]:
        if vec_size == 1:
            f = ptx_global_to_shared_copy_fp32x1
        else:
            f = generate_global_to_shared_vectorized_copy(dtype, vec_size)

        mod = tvm_tirx.IRModule.from_expr(f)
        mod = tvm.s_tir.transform.LowerOpaqueBlock()(mod)
        mod = tvm.s_tir.transform.LowerThreadBinding()(mod)
        mod = tvm_tirx.tirx.transform.FlattenTensor()(mod)
        if vec_size > 1:
            mod = tvm_tirx.tirx.transform.VectorizeLoop()(mod)
        mod = tvm.s_tir.transform.InjectPTXAsyncCopy()(mod)

        assert count_cp_async(mod["main"].body) == 1

        if not tvm.s_tir.testing.is_ampere_or_newer():
            continue

        with tvm_tirx.ir.transform.PassContext(config={"tirx.use_async_copy": 1}):
            mod = tvm.compile(tvm_tirx.IRModule.from_expr(f), target="cuda")

        A_np = np.random.rand(32, 128).astype(dtype)
        B_np = np.zeros((32, 128)).astype(dtype)

        def run_and_check():
            dev = tvm_tirx.cuda(0)
            A_nd = tvm_tirx.runtime.tensor(A_np, device=dev)
            B_nd = tvm_tirx.runtime.tensor(B_np, device=dev)
            mod(A_nd, B_nd)
            tvm_tirx.testing.assert_allclose(B_nd.numpy(), A_np)

        tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_inject_async_copy_shared_dyn():
    f = ptx_global_to_shared_dyn_copy_fp16x8

    mod = tvm_tirx.IRModule.from_expr(f)
    mod = tvm.s_tir.transform.LowerOpaqueBlock()(mod)
    mod = tvm.s_tir.transform.LowerThreadBinding()(mod)
    mod = tvm_tirx.tirx.transform.FlattenTensor()(mod)
    mod = tvm_tirx.tirx.transform.VectorizeLoop()(mod)
    mod = tvm.s_tir.transform.MergeSharedMemoryAllocations()(mod)
    mod = tvm.s_tir.transform.InjectPTXAsyncCopy()(mod)

    assert count_cp_async(mod["main"].body) == 2

    if not tvm.s_tir.testing.is_ampere_or_newer():
        return

    with tvm_tirx.ir.transform.PassContext(config={"tirx.use_async_copy": 1}):
        mod = tvm.compile(tvm_tirx.IRModule.from_expr(f), target="cuda")

    A_np = np.random.rand(32, 128).astype("float16")
    B_np = np.random.rand(32, 128).astype("float16")
    C_np = np.zeros((32, 128)).astype("float16")

    def run_and_check():
        dev = tvm_tirx.cuda(0)
        A_nd = tvm_tirx.runtime.tensor(A_np, device=dev)
        B_nd = tvm_tirx.runtime.tensor(B_np, device=dev)
        C_nd = tvm_tirx.runtime.tensor(C_np, device=dev)
        mod(A_nd, B_nd, C_nd)
        tvm_tirx.testing.assert_allclose(C_nd.numpy(), A_np + B_np)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


# Note: the test_inject_async_copy_barrier case (and its function helper)
# was removed — it relied on the indexed barrier API
# (`create_barriers`, `init_barrier_thread_count`, `arrive_barrier`,
# `wait_barrier`) which fork does not provide; fork uses the
# `ptx_mbarrier_*` family instead.

expected_cuda_script = r"""#include <cuda.h>
#endif

#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
#endif
__forceinline__ __device__ void tvm_builtin_ptx_cp_async_wait_group_async_wait_group_0() {
  asm volatile("cp.async.wait_group 0;" :  :  : "memory");
}
__forceinline__ __device__ void tvm_builtin_ptx_cp_async_wait_group_async_wait_group_1() {
  asm volatile("cp.async.wait_group 1;" :  :  : "memory");
}
__forceinline__ __device__ void tvm_builtin_ptx_cp_async_wait_group_async_wait_group_2() {
  asm volatile("cp.async.wait_group 2;" :  :  : "memory");
}
__forceinline__ __device__ void tvm_builtin_ptx_cp_async_wait_group_async_wait_group_5() {
  asm volatile("cp.async.wait_group 5;" :  :  : "memory");
}

__forceinline__ __device__ void ptx_cp_async_legacy_pred_ca_4_4_4(void* dst, int dst_off, void* src, int src_off, int predicate) {
  uint8_t* dst_p = (uint8_t*)dst + dst_off * 4;
  uint8_t* src_p = (uint8_t*)src + src_off * 4;
  unsigned int dst_addr = __cvta_generic_to_shared(dst_p);
  __asm__ __volatile__(
    "{\n"
    " .reg .pred p;\n"
    " setp.eq.u32 p, %3, 1;\n"
    " @p cp.async.ca.shared.global [%0], [%1], %2;\n"
    " @!p st.shared.u32 [%0], {%4};\n"
    "}\n"
    :: "r"(dst_addr), "l"(src_p), "n"(4), "r"(predicate), "r"(0)
  );
}

__forceinline__ __device__ void ptx_cp_async_legacy_ca_4_4_4(void* dst, int dst_off, void* src, int src_off) {
  uint8_t* dst_p = (uint8_t*)dst + dst_off * 4;
  uint8_t* src_p = (uint8_t*)src + src_off * 4;
  unsigned int dst_addr = __cvta_generic_to_shared(dst_p);
  asm volatile("cp.async.ca.shared.global [%0], [%1], %2;"
    :: "r"(dst_addr), "l"(src_p), "n"(4));
}
__forceinline__ __device__ void tvm_builtin_ptx_cp_async_commit_group_async_commit_group() {
  asm volatile("cp.async.commit_group;" :  : );
}
extern "C" __global__ void __launch_bounds__(16) main_kernel(float* __restrict__ A_ptr, float* __restrict__ B_ptr, float* __restrict__ C_ptr);
extern "C" __global__ void __launch_bounds__(16) main_kernel(float* __restrict__ A_ptr, float* __restrict__ B_ptr, float* __restrict__ C_ptr) {
  __shared__ alignas(64) float _ptr[64];
  __shared__ alignas(64) float _ptr_1[64];
  _ptr[((int)threadIdx.x)] = 0x0p+0f/*0.000000e+00*/;
  _ptr_1[((int)threadIdx.x)] = 0x0p+0f/*0.000000e+00*/;
  tvm_builtin_ptx_cp_async_commit_group_async_commit_group();
  ptx_cp_async_legacy_ca_4_4_4(_ptr, (((int)threadIdx.x) + 16), A_ptr, (((int)threadIdx.x) * 14));
  ptx_cp_async_legacy_ca_4_4_4(_ptr_1, (((int)threadIdx.x) + 16), B_ptr, (((int)threadIdx.x) * 14));
  tvm_builtin_ptx_cp_async_commit_group_async_commit_group();
  ptx_cp_async_legacy_ca_4_4_4(_ptr, (((int)threadIdx.x) + 32), A_ptr, ((((int)threadIdx.x) * 14) + 1));
  ptx_cp_async_legacy_ca_4_4_4(_ptr_1, (((int)threadIdx.x) + 32), B_ptr, ((((int)threadIdx.x) * 14) + 1));
  tvm_builtin_ptx_cp_async_commit_group_async_commit_group();
  int cse_v1 = (((int)threadIdx.x) * 16);
  for (int i = 0; i < 13; ++i) {
    ptx_cp_async_legacy_pred_ca_4_4_4(_ptr, ((((i + 3) & 3) * 16) + ((int)threadIdx.x)), A_ptr, (((((int)threadIdx.x) * 14) + i) + 2), (i < 12));
    tvm_builtin_ptx_cp_async_commit_group_async_commit_group();
    tvm_builtin_ptx_cp_async_wait_group_async_wait_group_5();
    __syncthreads();
    int cse_v5 = (((i & 3) * 16) + ((int)threadIdx.x));
    C_ptr[((((int)threadIdx.x) * 16) + i)] = (_ptr[(((i & 3) * 16) + ((int)threadIdx.x))] + _ptr_1[(((i & 3) * 16) + ((int)threadIdx.x))]);
    __syncthreads();
    ptx_cp_async_legacy_pred_ca_4_4_4(_ptr_1, ((((i + 3) & 3) * 16) + ((int)threadIdx.x)), B_ptr, (((((int)threadIdx.x) * 14) + i) + 2), (i < 12));
    tvm_builtin_ptx_cp_async_commit_group_async_commit_group();
  }
  tvm_builtin_ptx_cp_async_wait_group_async_wait_group_2();
  __syncthreads();
  int cse_v2 = (((int)threadIdx.x) + 16);
  C_ptr[((((int)threadIdx.x) * 16) + 13)] = (_ptr[(((int)threadIdx.x) + 16)] + _ptr_1[(((int)threadIdx.x) + 16)]);
  tvm_builtin_ptx_cp_async_wait_group_async_wait_group_1();
  __syncthreads();
  int cse_v3 = (((int)threadIdx.x) + 32);
  C_ptr[((((int)threadIdx.x) * 16) + 14)] = (_ptr[(((int)threadIdx.x) + 32)] + _ptr_1[(((int)threadIdx.x) + 32)]);
  tvm_builtin_ptx_cp_async_wait_group_async_wait_group_0();
  __syncthreads();
  int cse_v4 = (((int)threadIdx.x) + 48);
  C_ptr[((((int)threadIdx.x) * 16) + 15)] = (_ptr[(((int)threadIdx.x) + 48)] + _ptr_1[(((int)threadIdx.x) + 48)]);
}

"""


@pytest.fixture
def postproc_if_missing_async_support():
    arch = tvm_tirx.support.nvcc.get_target_compute_version()
    major, _ = tvm_tirx.support.nvcc.parse_compute_version(arch)
    support_async = major >= 8

    func_name = "tvm_callback_cuda_postproc"
    prev_postproc = tvm_ffi.get_global_func(func_name, allow_missing=True)

    # Store the generated code prior to the post-processing.  This
    # way, even though the generated code doesn't compile on platforms
    # that do not support async, the comparison against an expected
    # output can still be performed.  We cannot use
    # `mod.inspect_source()`, as that contains the source after all
    # post-processing.
    original_code = None

    def get_original_code():
        nonlocal original_code
        return original_code

    @tvm_ffi.register_global_func(func_name, override=True)
    def tvm_callback_cuda_postproc(code, _):
        nonlocal original_code
        original_code = code
        if support_async:
            return code
        else:
            ret = []
            for line in code.split("\n"):
                ret.append(line)
                ret.append("\n")
                if line.startswith('extern "C" __global__') and line.endswith("{"):
                    break
            ret.append("}")
            return "".join(ret)

    yield get_original_code

    # Restore previous postproc func to avoid impacting other tests
    if prev_postproc is None:
        tvm_ffi.registry.remove_global_func(func_name)
    else:
        tvm_ffi.register_global_func(func_name, prev_postproc, override=True)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_cp_async_in_if_then_else(postproc_if_missing_async_support):
    @Ts.function
    def simple_compute(
        A: T.Tensor((16, 14), "float32"),
        B: T.Tensor((16, 14), "float32"),
        C: T.Tensor((16, 16), "float32"),
    ):
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        for tx in T.parallel(0, 16, annotations={"thread_binding": "threadIdx.x"}):
            for i in T.serial(
                16,
                annotations={
                    "software_pipeline_stage": [0, 0, 3],
                    "software_pipeline_order": [0, 2, 1],
                    "software_pipeline_async_stages": [0],
                },
            ):
                with Ts.sblock("compute"):
                    Ts.reads(A[tx, i])
                    Ts.writes(C[tx, i])
                    A_shared = Ts.sblock_alloc_tensor((16, 1), dtype="float32", scope="shared")
                    B_shared = Ts.sblock_alloc_tensor((16, 1), dtype="float32", scope="shared")
                    with Ts.sblock():
                        Ts.reads(A[tx, i])
                        Ts.writes(A_shared[tx, 0])
                        A_shared[tx, 0] = T.if_then_else(
                            1 <= i and i < 15, A[tx, i - 1], T.float32(0)
                        )
                    with Ts.sblock():
                        Ts.reads(B[tx, i])
                        Ts.writes(B_shared[tx, 0])
                        B_shared[tx, 0] = T.if_then_else(
                            1 <= i and i < 15, B[tx, i - 1], T.float32(0)
                        )
                    with Ts.sblock():
                        Ts.reads(A_shared[tx, 0], B_shared[tx, 0])
                        Ts.writes(C[tx, i])
                        C[tx, i] = A_shared[tx, 0] + B_shared[tx, 0]

    mod = tvm_tirx.IRModule.from_expr(simple_compute)
    with tvm_tirx.ir.transform.PassContext(config={"tirx.use_async_copy": 1}):
        tvm.compile(mod, target="cuda")
    generated_code = postproc_if_missing_async_support()
    print(generated_code)
    # Fork emits an NVRTC-aware preamble (`#ifdef __CUDACC_RTC__ ... #else ...`
    # block) before the apache-style `#include <cuda.h>`; the body after that
    # block matches the expected snippet, so compare from the kernel-body
    # onwards instead of byte-for-byte from the start.
    marker = "#include <cuda.h>"
    expected_body = expected_cuda_script[expected_cuda_script.index(marker) :]
    actual_body = generated_code[generated_code.index(marker) :]
    assert actual_body == expected_body


@pytest.mark.skip(
    reason="This test fails due to an ordering issue with MergeSharedMemoryAllocations "
    "in device_driver_api.cc. However, fixing this causes failures in MLC. "
    "This bug should be addressed. See discussion in https://github.com/apache/tvm/pull/16769 "
    "and https://github.com/apache/tvm/pull/16569#issuecomment-1992720448"
)
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_vectorize_cp_async_in_if_then_else(postproc_if_missing_async_support):
    C_s0_0 = T.dynamic("C_s0_0", "int32")
    C_s1_0 = T.dynamic("C_s1_0", "int32")
    A_s0_3 = T.dynamic("A_s0_3", "int32")
    A_s1_3 = T.dynamic("A_s1_3", "int32")
    C_s0_4 = T.dynamic("C_s0_4", "int32")
    C_s1_4 = T.dynamic("C_s1_4", "int32")
    A_s0_0 = T.dynamic("A_s0_0", "int32")
    A_s1_0 = T.dynamic("A_s1_0", "int32")
    C_s0_1 = T.dynamic("C_s0_1", "int32")
    C_s1_1 = T.dynamic("C_s1_1", "int32")
    A_s0_1 = T.dynamic("A_s0_1", "int32")
    A_s1_1 = T.dynamic("A_s1_1", "int32")
    C_s0_2 = T.dynamic("C_s0_2", "int32")
    C_s1_2 = T.dynamic("C_s1_2", "int32")
    A_s0_2 = T.dynamic("A_s0_2", "int32")
    A_s1_2 = T.dynamic("A_s1_2", "int32")
    B_s0 = T.dynamic("B_s0", "int32")
    B_s1 = T.dynamic("B_s1", "int32")
    C_s0_3 = T.dynamic("C_s0_3", "int32")
    C_s1_3 = T.dynamic("C_s1_3", "int32")

    complex_compute_c_offset = T.dynamic("complex_compute_c_offset", "int32")
    complex_compute_a_1_offset = T.dynamic("complex_compute_a_1_offset", "int32")
    complex_compute_c_offset_1 = T.dynamic("complex_compute_c_offset_1", "int32")
    complex_compute_a_1_offset_1 = T.dynamic("complex_compute_a_1_offset_1", "int32")
    complex_compute_c_offset_2 = T.dynamic("complex_compute_c_offset_2", "int32")
    complex_compute_a_1_offset_2 = T.dynamic("complex_compute_a_1_offset_2", "int32")
    complex_compute_b_offset = T.dynamic("complex_compute_b_offset", "int32")
    complex_compute_c_offset_3 = T.dynamic("complex_compute_c_offset_3", "int32")
    complex_compute_a_1_offset_3 = T.dynamic("complex_compute_a_1_offset_3", "int32")
    complex_compute_c_offset_4 = T.dynamic("complex_compute_c_offset_4", "int32")

    @Ts.function
    def complex_compute(
        A: T.Tensor((2, 16, 16, 1280), "float16"),
        W: T.Tensor((1280, 3, 3, 1280), "float16"),
        Conv: T.Tensor((512, 1280), "float16"),
    ):
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        # with Ts.sblock("root"):
        data_im2col_reindex_shared_dyn = Ts.sblock_alloc_tensor(
            (512, 11520), "float16", scope="shared.dyn"
        )
        data_im2col_reindex_shared_dyn_wmma_matrix_a = Ts.sblock_alloc_tensor(
            (512, 11520), "float16", scope="wmma.matrix_a"
        )
        weight_flatten_reindex_shared_dyn = Ts.sblock_alloc_tensor(
            (1280, 11520), "float16", scope="shared.dyn"
        )
        weight_flatten_reindex_shared_dyn_wmma_matrix_b = Ts.sblock_alloc_tensor(
            (1280, 11520), "float16", scope="wmma.matrix_b"
        )
        Conv_reindex_wmma_accumulator = Ts.sblock_alloc_tensor(
            (512, 1280), "float16", scope="wmma.accumulator"
        )
        for x_0_0 in T.parallel(8, annotations={"thread_binding": "blockIdx.y"}):
            for y_0_0 in T.parallel(20, annotations={"thread_binding": "blockIdx.x"}):
                for x_0_1 in T.parallel(2, annotations={"thread_binding": "threadIdx.y"}):
                    for y_0_1 in T.parallel(2, annotations={"thread_binding": "threadIdx.z"}):
                        for x_0_2_init, y_0_2_init in T.grid(2, 2):
                            with Ts.sblock("Conv_init_o"):
                                v_x_o = Ts.axis.spatial(32, x_0_0 * 4 + x_0_1 * 2 + x_0_2_init)
                                v_y_o = Ts.axis.spatial(80, y_0_0 * 4 + y_0_1 * 2 + y_0_2_init)
                                Ts.reads()
                                Ts.writes(
                                    Conv_reindex_wmma_accumulator[
                                        v_x_o * 16 : v_x_o * 16 + 16, v_y_o * 16 : v_y_o * 16 + 16
                                    ]
                                )
                                C = Ts.match_tensor(
                                    Conv_reindex_wmma_accumulator[
                                        v_x_o * 16 : v_x_o * 16 + 16, v_y_o * 16 : v_y_o * 16 + 16
                                    ],
                                    (16, 16),
                                    "float16",
                                    layout=T.TileLayout(
                                        T.S[(16, 16) : (C_s0_0, C_s1_0)]
                                        + ((16) * complex_compute_c_offset)
                                    ),
                                    scope="wmma.accumulator",
                                )
                                C_pattern = T.meta_var(match_shape_stride_pattern(C.ty))
                                T.gpu_fill_fragment(
                                    C.data,
                                    16,
                                    16,
                                    16,
                                    C_pattern.elem_offset // C_s0_0 // 16 * (C_s0_0 // 16)
                                    + C_pattern.elem_offset % C_s0_0 // 16,
                                    T.float32(0),
                                )
                        for k_0_0 in T.serial(
                            180,
                            annotations={
                                "software_pipeline_stage": [0, 0, 1],
                                "software_pipeline_order": [0, 1, 2],
                                "software_pipeline_async_stages": [0],
                            },
                        ):
                            for ax0_ax1_0_fused_0 in range(4):
                                for ax0_ax1_0_fused_1 in T.parallel(
                                    2, annotations={"thread_binding": "threadIdx.z"}
                                ):
                                    for ax0_ax1_0_fused_2 in T.parallel(
                                        2, annotations={"thread_binding": "threadIdx.y"}
                                    ):
                                        for ax0_ax1_0_fused_3 in T.parallel(
                                            32, annotations={"thread_binding": "threadIdx.x"}
                                        ):
                                            with Ts.sblock("data_im2col_reindex_shared.dyn_o"):
                                                v0 = Ts.axis.spatial(
                                                    512,
                                                    x_0_0 * 64
                                                    + (
                                                        ax0_ax1_0_fused_0 * 128
                                                        + ax0_ax1_0_fused_1 * 64
                                                        + ax0_ax1_0_fused_2 * 32
                                                        + ax0_ax1_0_fused_3
                                                    )
                                                    // 8,
                                                )
                                                v1_o = Ts.axis.spatial(
                                                    1440,
                                                    k_0_0 * 8
                                                    + (
                                                        ax0_ax1_0_fused_0 * 128
                                                        + ax0_ax1_0_fused_1 * 64
                                                        + ax0_ax1_0_fused_2 * 32
                                                        + ax0_ax1_0_fused_3
                                                    )
                                                    % 8,
                                                )
                                                Ts.reads(
                                                    A[
                                                        v0 // 256,
                                                        v1_o // 480 + v0 % 256 // 16 - 1,
                                                        v1_o % 480 // 160 + v0 % 16 - 1,
                                                        v1_o % 160 * 8 : v1_o % 160 * 8 + 8,
                                                    ]
                                                )
                                                Ts.writes(
                                                    data_im2col_reindex_shared_dyn[
                                                        v0, v1_o * 8 : v1_o * 8 + 8
                                                    ]
                                                )
                                                for ax1_1 in T.vectorized(8):
                                                    with Ts.sblock(
                                                        "data_im2col_reindex_shared.dyn"
                                                    ):
                                                        v1_i = Ts.axis.spatial(8, ax1_1)
                                                        Ts.reads(
                                                            A[
                                                                v0 // 256,
                                                                v1_o // 480 + v0 % 256 // 16 - 1,
                                                                v1_o % 480 // 160 + v0 % 16 - 1,
                                                                v1_o % 160 * 8 + v1_i,
                                                            ]
                                                        )
                                                        Ts.writes(
                                                            data_im2col_reindex_shared_dyn[
                                                                v0, v1_o * 8 + v1_i
                                                            ]
                                                        )
                                                        Ts.sblock_attr(
                                                            {"tensor_dim_align": [[0, 0, 32, 8]]}
                                                        )
                                                        data_im2col_reindex_shared_dyn[
                                                            v0, v1_o * 8 + v1_i
                                                        ] = T.if_then_else(
                                                            1 <= v1_o // 480 + v0 % 256 // 16
                                                            and v1_o // 480 + v0 % 256 // 16 < 17
                                                            and 1 <= v1_o % 480 // 160 + v0 % 16
                                                            and v1_o % 480 // 160 + v0 % 16 < 17,
                                                            A[
                                                                v0 // 256,
                                                                v1_o // 480 + v0 % 256 // 16 - 1,
                                                                v1_o % 480 // 160 + v0 % 16 - 1,
                                                                v1_o % 160 * 8 + v1_i,
                                                            ],
                                                            T.float16(0),
                                                        )
                            for ax0_ax1_0_fused_0 in range(4):
                                for ax0_ax1_0_fused_1 in T.parallel(
                                    2, annotations={"thread_binding": "threadIdx.z"}
                                ):
                                    for ax0_ax1_0_fused_2 in T.parallel(
                                        2, annotations={"thread_binding": "threadIdx.y"}
                                    ):
                                        for ax0_ax1_0_fused_3 in T.parallel(
                                            32, annotations={"thread_binding": "threadIdx.x"}
                                        ):
                                            for ax1_1 in T.vectorized(8):
                                                with Ts.sblock("weight_flatten_reindex_shared.dyn"):
                                                    v0 = Ts.axis.spatial(
                                                        1280,
                                                        y_0_0 * 64
                                                        + (
                                                            ax0_ax1_0_fused_0 * 128
                                                            + ax0_ax1_0_fused_1 * 64
                                                            + ax0_ax1_0_fused_2 * 32
                                                            + ax0_ax1_0_fused_3
                                                        )
                                                        // 8,
                                                    )
                                                    v1 = Ts.axis.spatial(
                                                        11520,
                                                        k_0_0 * 64
                                                        + (
                                                            ax0_ax1_0_fused_0 * 128
                                                            + ax0_ax1_0_fused_1 * 64
                                                            + ax0_ax1_0_fused_2 * 32
                                                            + ax0_ax1_0_fused_3
                                                        )
                                                        % 8
                                                        * 8
                                                        + ax1_1,
                                                    )
                                                    Ts.reads(
                                                        W[
                                                            v0,
                                                            v1 // 3840,
                                                            v1 % 3840 // 1280,
                                                            v1 % 1280,
                                                        ]
                                                    )
                                                    Ts.writes(
                                                        weight_flatten_reindex_shared_dyn[v0, v1]
                                                    )
                                                    Ts.sblock_attr(
                                                        {"tensor_dim_align": [[0, 0, 32, 8]]}
                                                    )
                                                    weight_flatten_reindex_shared_dyn[v0, v1] = W[
                                                        v0,
                                                        v1 // 1280 // 3,
                                                        v1 // 1280 % 3,
                                                        v1 % 1280,
                                                    ]
                            for k_0_1 in range(4):
                                for ax0_0, ax1_0 in T.grid(2, 1):
                                    with Ts.sblock(
                                        "data_im2col_reindex_shared.dyn_wmma.matrix_a_o"
                                    ):
                                        v0_o = Ts.axis.spatial(32, x_0_0 * 4 + x_0_1 * 2 + ax0_0)
                                        v1_o = Ts.axis.spatial(720, k_0_0 * 4 + k_0_1 + ax1_0)
                                        Ts.reads(
                                            data_im2col_reindex_shared_dyn[
                                                v0_o * 16 : v0_o * 16 + 16,
                                                v1_o * 16 : v1_o * 16 + 16,
                                            ]
                                        )
                                        Ts.writes(
                                            data_im2col_reindex_shared_dyn_wmma_matrix_a[
                                                v0_o * 16 : v0_o * 16 + 16,
                                                v1_o * 16 : v1_o * 16 + 16,
                                            ]
                                        )
                                        A_1 = Ts.match_tensor(
                                            data_im2col_reindex_shared_dyn[
                                                v0_o * 16 : v0_o * 16 + 16,
                                                v1_o * 16 : v1_o * 16 + 16,
                                            ],
                                            (16, 16),
                                            "float16",
                                            layout=T.TileLayout(
                                                T.S[(16, 16) : (A_s0_0, A_s1_0)]
                                                + ((16) * complex_compute_a_1_offset)
                                            ),
                                            scope="shared.dyn",
                                        )
                                        C = Ts.match_tensor(
                                            data_im2col_reindex_shared_dyn_wmma_matrix_a[
                                                v0_o * 16 : v0_o * 16 + 16,
                                                v1_o * 16 : v1_o * 16 + 16,
                                            ],
                                            (16, 16),
                                            "float16",
                                            layout=T.TileLayout(
                                                T.S[(16, 16) : (C_s0_1, C_s1_1)]
                                                + ((16) * complex_compute_c_offset_1)
                                            ),
                                            scope="wmma.matrix_a",
                                        )
                                        C_pattern = T.meta_var(match_shape_stride_pattern(C.ty))
                                        T.gpu_load_matrix_sync(
                                            C.data,
                                            16,
                                            16,
                                            16,
                                            C_pattern.elem_offset // C_s0_1 // 16 * (C_s0_1 // 16)
                                            + C_pattern.elem_offset % C_s0_1 // 16,
                                            A_1.ptr_to([0] * len(A_1.shape)),
                                            A_s0_0,
                                            "row_major",
                                        )
                                for ax0_0, ax1_0 in T.grid(2, 1):
                                    with Ts.sblock(
                                        "weight_flatten_reindex_shared.dyn_wmma.matrix_b_o"
                                    ):
                                        v0_o = Ts.axis.spatial(80, y_0_0 * 4 + y_0_1 * 2 + ax0_0)
                                        v1_o = Ts.axis.spatial(720, k_0_0 * 4 + k_0_1 + ax1_0)
                                        Ts.reads(
                                            weight_flatten_reindex_shared_dyn[
                                                v0_o * 16 : v0_o * 16 + 16,
                                                v1_o * 16 : v1_o * 16 + 16,
                                            ]
                                        )
                                        Ts.writes(
                                            weight_flatten_reindex_shared_dyn_wmma_matrix_b[
                                                v0_o * 16 : v0_o * 16 + 16,
                                                v1_o * 16 : v1_o * 16 + 16,
                                            ]
                                        )
                                        A_1 = Ts.match_tensor(
                                            weight_flatten_reindex_shared_dyn[
                                                v0_o * 16 : v0_o * 16 + 16,
                                                v1_o * 16 : v1_o * 16 + 16,
                                            ],
                                            (16, 16),
                                            "float16",
                                            layout=T.TileLayout(
                                                T.S[(16, 16) : (A_s0_1, A_s1_1)]
                                                + ((16) * complex_compute_a_1_offset_1)
                                            ),
                                            scope="shared.dyn",
                                        )
                                        C = Ts.match_tensor(
                                            weight_flatten_reindex_shared_dyn_wmma_matrix_b[
                                                v0_o * 16 : v0_o * 16 + 16,
                                                v1_o * 16 : v1_o * 16 + 16,
                                            ],
                                            (16, 16),
                                            "float16",
                                            layout=T.TileLayout(
                                                T.S[(16, 16) : (C_s0_2, C_s1_2)]
                                                + ((16) * complex_compute_c_offset_2)
                                            ),
                                            scope="wmma.matrix_b",
                                        )
                                        C_pattern = T.meta_var(match_shape_stride_pattern(C.ty))
                                        T.gpu_load_matrix_sync(
                                            C.data,
                                            16,
                                            16,
                                            16,
                                            C_pattern.elem_offset // C_s0_2 // 16 * (C_s0_2 // 16)
                                            + C_pattern.elem_offset % C_s0_2 // 16,
                                            A_1.ptr_to([0] * len(A_1.shape)),
                                            A_s0_1,
                                            "col_major",
                                        )
                                for x_0_2, y_0_2 in T.grid(2, 2):
                                    with Ts.sblock("Conv_update_o"):
                                        v_x_o = Ts.axis.spatial(32, x_0_0 * 4 + x_0_1 * 2 + x_0_2)
                                        v_y_o = Ts.axis.spatial(80, y_0_0 * 4 + y_0_1 * 2 + y_0_2)
                                        v_k_o = Ts.axis.reduce(720, k_0_0 * 4 + k_0_1)
                                        Ts.reads(
                                            Conv_reindex_wmma_accumulator[
                                                v_x_o * 16 : v_x_o * 16 + 16,
                                                v_y_o * 16 : v_y_o * 16 + 16,
                                            ],
                                            data_im2col_reindex_shared_dyn_wmma_matrix_a[
                                                v_x_o * 16 : v_x_o * 16 + 16,
                                                v_k_o * 16 : v_k_o * 16 + 16,
                                            ],
                                            weight_flatten_reindex_shared_dyn_wmma_matrix_b[
                                                v_y_o * 16 : v_y_o * 16 + 16,
                                                v_k_o * 16 : v_k_o * 16 + 16,
                                            ],
                                        )
                                        Ts.writes(
                                            Conv_reindex_wmma_accumulator[
                                                v_x_o * 16 : v_x_o * 16 + 16,
                                                v_y_o * 16 : v_y_o * 16 + 16,
                                            ]
                                        )
                                        A_1 = Ts.match_tensor(
                                            data_im2col_reindex_shared_dyn_wmma_matrix_a[
                                                v_x_o * 16 : v_x_o * 16 + 16,
                                                v_k_o * 16 : v_k_o * 16 + 16,
                                            ],
                                            (16, 16),
                                            "float16",
                                            layout=T.TileLayout(
                                                T.S[(16, 16) : (A_s0_2, A_s1_2)]
                                                + ((16) * complex_compute_a_1_offset_2)
                                            ),
                                            scope="wmma.matrix_a",
                                        )
                                        B = Ts.match_tensor(
                                            weight_flatten_reindex_shared_dyn_wmma_matrix_b[
                                                v_y_o * 16 : v_y_o * 16 + 16,
                                                v_k_o * 16 : v_k_o * 16 + 16,
                                            ],
                                            (16, 16),
                                            "float16",
                                            layout=T.TileLayout(
                                                T.S[(16, 16) : (B_s0, B_s1)]
                                                + ((16) * complex_compute_b_offset)
                                            ),
                                            scope="wmma.matrix_b",
                                        )
                                        C = Ts.match_tensor(
                                            Conv_reindex_wmma_accumulator[
                                                v_x_o * 16 : v_x_o * 16 + 16,
                                                v_y_o * 16 : v_y_o * 16 + 16,
                                            ],
                                            (16, 16),
                                            "float16",
                                            layout=T.TileLayout(
                                                T.S[(16, 16) : (C_s0_3, C_s1_3)]
                                                + ((16) * complex_compute_c_offset_3)
                                            ),
                                            scope="wmma.accumulator",
                                        )
                                        A_1_pattern = T.meta_var(match_shape_stride_pattern(A_1.ty))
                                        B_pattern = T.meta_var(match_shape_stride_pattern(B.ty))
                                        C_pattern = T.meta_var(match_shape_stride_pattern(C.ty))
                                        T.gpu_mma_sync(
                                            C.data,
                                            C_pattern.elem_offset // C_s0_3 // 16 * (C_s0_3 // 16)
                                            + C_pattern.elem_offset % C_s0_3 // 16,
                                            A_1.data,
                                            A_1_pattern.elem_offset // A_s0_2 // 16 * (A_s0_2 // 16)
                                            + A_1_pattern.elem_offset % A_s0_2 // 16,
                                            B.data,
                                            B_pattern.elem_offset // B_s0 // 16 * (B_s0 // 16)
                                            + B_pattern.elem_offset % B_s0 // 16,
                                            C.data,
                                            C_pattern.elem_offset // C_s0_3 // 16 * (C_s0_3 // 16)
                                            + C_pattern.elem_offset % C_s0_3 // 16,
                                        )
                        for ax0_0, ax1_0 in T.grid(2, 2):
                            with Ts.sblock("Conv_reindex_wmma.accumulator_o"):
                                v0_o = Ts.axis.spatial(32, x_0_0 * 4 + x_0_1 * 2 + ax0_0)
                                v1_o = Ts.axis.spatial(80, y_0_0 * 4 + y_0_1 * 2 + ax1_0)
                                Ts.reads(
                                    Conv_reindex_wmma_accumulator[
                                        v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                    ]
                                )
                                Ts.writes(
                                    Conv[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16]
                                )
                                A_1 = Ts.match_tensor(
                                    Conv_reindex_wmma_accumulator[
                                        v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                    ],
                                    (16, 16),
                                    "float16",
                                    layout=T.TileLayout(
                                        T.S[(16, 16) : (A_s0_3, A_s1_3)]
                                        + ((16) * complex_compute_a_1_offset_3)
                                    ),
                                    scope="wmma.accumulator",
                                )
                                C = Ts.match_tensor(
                                    Conv[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16],
                                    (16, 16),
                                    "float16",
                                    layout=T.TileLayout(
                                        T.S[(16, 16) : (C_s0_4, C_s1_4)]
                                        + ((16) * complex_compute_c_offset_4)
                                    ),
                                )
                                A_1_pattern = T.meta_var(match_shape_stride_pattern(A_1.ty))
                                T.gpu_store_matrix_sync(
                                    A_1.data,
                                    16,
                                    16,
                                    16,
                                    A_1_pattern.elem_offset // A_s0_3 // 16 * (A_s0_3 // 16)
                                    + A_1_pattern.elem_offset % A_s0_3 // 16,
                                    C.ptr_to([0] * len(C.shape)),
                                    C_s0_4,
                                    "row_major",
                                )

    mod = tvm_tirx.IRModule.from_expr(complex_compute)
    with tvm_tirx.ir.transform.PassContext(config={"tirx.use_async_copy": 1}):
        tvm.compile(mod, target="cuda")
    generated_code = postproc_if_missing_async_support()
    # generated_code must contain "  setp.ne.b32 p, %0, 0;"
    assert "setp.ne.b32" in generated_code


def test_multiplication_nodes_are_inlined():
    @I.ir_module
    class Before:
        @Ts.function
        def main(A: T.Tensor((32, 128), "float16")):
            tx = T.launch_thread("threadIdx.x", T.int64(32))
            A_flattened = T.decl_tensor(A.data, ty_args=[T.Tensor((4096,), "float16", data=A.data)])
            A_shared = T.alloc_tensor(ty_args=[T.Tensor([4096], "float16", scope="shared")])

            with Ts.async_copy_scope():
                for i in range(16):
                    cse_v1: T.int64 = T.Cast("int64", i)
                    A_shared[T.Ramp(tx * T.int64(128) + cse_v1 * T.int64(8), T.int64(1), 8)] = (
                        A_flattened[T.Ramp(tx * T.int64(128) + cse_v1 * T.int64(8), T.int64(1), 8)]
                    )
                T.ptx.cp.async_.commit_group()
                T.ptx.cp.async_.wait_group(0)

    @I.ir_module
    class Expected:
        @Ts.function
        def main(A: T.Tensor((32, 128), "float16")):
            tx = T.launch_thread("threadIdx.x", T.int64(32))
            A_flattened = T.decl_tensor(A.data, ty_args=[T.Tensor((4096,), "float16", data=A.data)])
            A_shared = T.alloc_tensor(ty_args=[T.Tensor((4096,), "float16", scope="shared")])
            for i in range(16):
                cse_v1: T.int64 = T.Cast("int64", i)
                T.s_tir.cp_async_raw(
                    A_shared.data,
                    tx * T.int64(128) + cse_v1 * T.int64(8),
                    A_flattened.data,
                    tx * T.int64(128) + cse_v1 * T.int64(8),
                    16,
                    ty="float16",
                )
            T.ptx.cp.async_.commit_group()
            T.ptx.cp.async_.wait_group(0)

    After = tvm.s_tir.transform.InjectPTXAsyncCopy()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
