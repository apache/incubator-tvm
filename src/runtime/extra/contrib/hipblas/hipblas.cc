/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file Use external hipblas library call.
 */
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include "../cblas/gemm_common.h"
#include "builtin_fp16.h"
#include "hipblas_utils.h"

namespace tvm {
namespace contrib {

inline hipblasOperation_t HIPBLASBooleanToTranspose(bool item) {
  return item ? HIPBLAS_OP_T : HIPBLAS_OP_N;
}

struct HipblasHgemmOp {
  typedef hipblasHalf TDatatype;
  hipblasHandle_t handle;
  explicit HipblasHgemmOp(hipblasHandle_t hdl) : handle(hdl) {}

  void operator()(bool ta, bool tb, int M, int N, int K, hipblasHalf alpha, hipblasHalf* A, int lda,
                  hipblasHalf* B, int ldb, hipblasHalf beta, hipblasHalf* C, int ldc) {
    CHECK_HIPBLAS_ERROR(hipblasHgemm(handle, HIPBLASBooleanToTranspose(ta),
                                     HIPBLASBooleanToTranspose(tb), M, N, K, &alpha, A, lda, B, ldb,
                                     &beta, C, ldc));
  }
};

struct HipblasSgemmOp {
  typedef float TDatatype;
  hipblasHandle_t handle;
  explicit HipblasSgemmOp(hipblasHandle_t hdl) : handle(hdl) {}

  void operator()(bool ta, bool tb, int M, int N, int K, float alpha, float* A, int lda, float* B,
                  int ldb, float beta, float* C, int ldc) {
    CHECK_HIPBLAS_ERROR(hipblasSgemm(handle, HIPBLASBooleanToTranspose(ta),
                                     HIPBLASBooleanToTranspose(tb), M, N, K, &alpha, A, lda, B, ldb,
                                     &beta, C, ldc));
  }
};

struct HipblasDgemmOp {
  typedef double TDatatype;
  hipblasHandle_t handle;
  explicit HipblasDgemmOp(hipblasHandle_t hdl) : handle(hdl) {}
  void operator()(bool ta, bool tb, int M, int N, int K, double alpha, double* A, int lda,
                  double* B, int ldb, double beta, double* C, int ldc) {
    CHECK_HIPBLAS_ERROR(hipblasDgemm(handle, HIPBLASBooleanToTranspose(ta),
                                     HIPBLASBooleanToTranspose(tb), M, N, K, &alpha, A, lda, B, ldb,
                                     &beta, C, ldc));
  }
};

struct HipblasHgemmBatchOp {
  typedef hipblasHalf TDatatype;
  hipblasHandle_t handle;
  explicit HipblasHgemmBatchOp(hipblasHandle_t hdl) : handle(hdl) {}
  void operator()(int batch_size, bool ta, bool tb, int M, int N, int K, hipblasHalf alpha,
                  hipblasHalf* A, int a_stride, int lda, hipblasHalf* B, int b_stride, int ldb,
                  hipblasHalf beta, hipblasHalf* C, int c_stride, int ldc) {
    CHECK_HIPBLAS_ERROR(hipblasHgemmStridedBatched(
        handle, HIPBLASBooleanToTranspose(ta), HIPBLASBooleanToTranspose(tb), M, N, K, &alpha, A,
        lda, a_stride, B, ldb, b_stride, &beta, C, ldc, c_stride, batch_size));
  }
};

struct HipblasSgemmBatchOp {
  typedef float TDatatype;
  hipblasHandle_t handle;
  explicit HipblasSgemmBatchOp(hipblasHandle_t hdl) : handle(hdl) {}
  void operator()(int batch_size, bool ta, bool tb, int M, int N, int K, float alpha, float* A,
                  int a_stride, int lda, float* B, int b_stride, int ldb, float beta, float* C,
                  int c_stride, int ldc) {
    CHECK_HIPBLAS_ERROR(hipblasSgemmStridedBatched(
        handle, HIPBLASBooleanToTranspose(ta), HIPBLASBooleanToTranspose(tb), M, N, K, &alpha, A,
        lda, a_stride, B, ldb, b_stride, &beta, C, ldc, c_stride, batch_size));
  }
};

struct HipblasDgemmBatchOp {
  typedef double TDatatype;
  hipblasHandle_t handle;
  explicit HipblasDgemmBatchOp(hipblasHandle_t hdl) : handle(hdl) {}
  void operator()(int batch_size, bool ta, bool tb, int M, int N, int K, double alpha, double* A,
                  int a_stride, int lda, double* B, int b_stride, int ldb, double beta, double* C,
                  int c_stride, int ldc) {
    CHECK_HIPBLAS_ERROR(hipblasDgemmStridedBatched(
        handle, HIPBLASBooleanToTranspose(ta), HIPBLASBooleanToTranspose(tb), M, N, K, &alpha, A,
        lda, a_stride, B, ldb, b_stride, &beta, C, ldc, c_stride, batch_size));
  }
};

// Check supported mix-precision computation type and return computeType
bool CheckMixPrecisionType(DLDataType in_dtype, DLDataType out_dtype, bool int_support = true) {
  if (int_support && out_dtype == DLDataType{kDLInt, 32, 1}) {
    return in_dtype == DLDataType{kDLInt, 8, 1};
  } else if (out_dtype == DLDataType{kDLFloat, 32, 1}) {
    return in_dtype == DLDataType{kDLInt, 8, 1} || in_dtype == DLDataType{kDLFloat, 16, 1};
  } else {
    return false;
  }
}

void CallHipblasLt(hipblasLtHandle_t hdl, hipStream_t stream,
                   hipblasLtMatmulPreference_t matmul_pref_desc, const DLTensor* A,
                   const DLTensor* B, const DLTensor* bias, const DLTensor* C, bool transa,
                   bool transb, void* workspace_ptr, size_t workspace_size,
                   hipblasLtEpilogue_t epilogue) {
  TVM_FFI_ICHECK(A->dtype == B->dtype);
  // Reversed strides indicates an in-place transpose operation.
  transa = IsInPlaceTransposed(A) ? !transa : transa;
  transb = IsInPlaceTransposed(B) ? !transb : transb;

  auto compute_type = HIPBLAS_COMPUTE_32F;
  auto scale_type = HIP_R_32F;
  hipDataType ab_type = HIP_R_32F;
  hipDataType c_type = HIP_R_32F;
  float one_fp32 = 1.0;
  float zero_fp32 = 0.0;
  int32_t one_i32 = 1;
  int32_t zero_i32 = 0;
  void* alpha = &one_fp32;
  void* beta = &zero_fp32;

  if (A->dtype == DLDataType{kDLFloat, 16, 1}) {
    ab_type = HIP_R_16F;
  } else if (A->dtype == DLDataType{kDLInt, 8, 1}) {
    ab_type = HIP_R_8I;
  }

  if (C->dtype == DLDataType{kDLFloat, 16, 1}) {
    c_type = HIP_R_16F;
  } else if (C->dtype == DLDataType{kDLInt, 32, 1}) {
    c_type = HIP_R_32I;
    compute_type = HIPBLAS_COMPUTE_32I;
    scale_type = HIP_R_32I;
    alpha = &one_i32;
    beta = &zero_i32;
  }

  hipblasLtMatmulDesc_t op_desc;
  hipblasOperation_t op_transa = HIPBLASBooleanToTranspose(transa);
  hipblasOperation_t op_transb = HIPBLASBooleanToTranspose(transb);

  CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescCreate(&op_desc, compute_type, scale_type));
  CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(op_desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                                      &op_transb, sizeof(op_transb)));
  CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(op_desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                                      &op_transa, sizeof(op_transa)));

  if (bias != nullptr) {
    CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(op_desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                        &bias->data, sizeof(float*)));
  }

  if (epilogue != HIPBLASLT_EPILOGUE_DEFAULT) {
    CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(op_desc, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                        &epilogue, sizeof(epilogue)));
  }

  int batch_offset_A = A->ndim - 2;
  int batch_offset_B = B->ndim - 2;

  int M = ColumnCount(B, transb, batch_offset_B);
  int N = RowCount(A, transa, batch_offset_A);
  int K = ColumnCount(A, transa, batch_offset_A);
  bool use_batched_gemm = A->ndim > 2 || B->ndim > 2;

  // If A is batched but B is not, flatten all non-reduction axes of A to use the regular GEMM.
  // This trick is only applicable if batch axes and the other spatial axis (M or N) are
  // adjacent in both the input and the output matrix. In particular, if A is of shape (M, K)
  // and B matrix is of shape (Batch, N, K) with transb = true, the output shape
  // is (Batch, M, N). Since the Batch and the N axes are not adjacent in the output, we cannot
  // use the regular GEMM if only B is batched.
  if (A->ndim > 2 && B->ndim == 2 && transa == false) {
    N = 1;
    for (int i = 0; i < A->ndim - 1; ++i) {
      N *= A->shape[i];
    }
    use_batched_gemm = false;
  }

  int lda = transb ? K : M;
  int ldb = transa ? N : K;
  int ldc = M;

  hipblasLtMatrixLayout_t A_desc, B_desc, C_desc;
  CHECK_HIPBLAS_ERROR(
      hipblasLtMatrixLayoutCreate(&A_desc, ab_type, !transb ? M : K, !transb ? K : M, lda));
  CHECK_HIPBLAS_ERROR(
      hipblasLtMatrixLayoutCreate(&B_desc, ab_type, !transa ? K : N, !transa ? N : K, ldb));
  CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutCreate(&C_desc, c_type, M, N, ldc));

  if (use_batched_gemm) {
    auto get_batch_count = [](int64_t* shape, int batch_offset) {
      int64_t count = 1;
      for (int i = 0; i < batch_offset; ++i) {
        count *= shape[i];
      }
      return count;
    };
    auto set_batch = [](hipblasLtMatrixLayout_t mat_desc, int batch_count, int64_t batch_stride) {
      CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutSetAttribute(
          mat_desc, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
      CHECK_HIPBLAS_ERROR(
          hipblasLtMatrixLayoutSetAttribute(mat_desc, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                            &batch_stride, sizeof(batch_stride)));
    };

    int batch_count_A = get_batch_count(A->shape, batch_offset_A);
    int batch_count_B = get_batch_count(B->shape, batch_offset_B);
    int batch_count_C = get_batch_count(C->shape, C->ndim - 2);
    int64_t batch_stride_A = M * K;
    int64_t batch_stride_B = K * N;
    int64_t batch_stride_C = M * N;

    // hipBLASLt does not seem to support batched GEMM with one of matrices having
    // one batch (with batch_stride 0).
    TVM_FFI_ICHECK_EQ(batch_count_A, batch_count_B);

    set_batch(A_desc, batch_count_A, batch_stride_A);
    set_batch(B_desc, batch_count_B, batch_stride_B);
    set_batch(C_desc, batch_count_C, batch_stride_C);
  }

  auto A_data = static_cast<char*>(A->data) + A->byte_offset;
  auto B_data = static_cast<char*>(B->data) + B->byte_offset;
  auto C_data = static_cast<char*>(C->data) + C->byte_offset;

  hipblasLtMatmulPreferenceSetAttribute(matmul_pref_desc, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                        &workspace_size, sizeof(size_t));

  hipblasLtMatmulHeuristicResult_t heuristic_result = {};
  int returned_result = 0;
  CHECK_HIPBLAS_ERROR(hipblasLtMatmulAlgoGetHeuristic(hdl, op_desc, A_desc, B_desc, C_desc, C_desc,
                                                      matmul_pref_desc, 1, &heuristic_result,
                                                      &returned_result));
  if (returned_result == 0) {
    CHECK_HIPBLAS_ERROR(HIPBLAS_STATUS_NOT_SUPPORTED);
  }

  CHECK_HIPBLAS_ERROR(hipblasLtMatmul(hdl, op_desc, alpha, B_data, A_desc, A_data, B_desc, beta,
                                      C_data, C_desc, C_data, C_desc, &heuristic_result.algo,
                                      workspace_ptr, workspace_size, stream));

  hipblasLtMatmulDescDestroy(op_desc);
  hipblasLtMatrixLayoutDestroy(A_desc);
  hipblasLtMatrixLayoutDestroy(B_desc);
  hipblasLtMatrixLayoutDestroy(C_desc);
}

inline void CallGemmEx(ffi::PackedArgs args, ffi::Any* ret, hipblasHandle_t hdl) {
  auto A = args[0].cast<DLTensor*>();
  auto B = args[1].cast<DLTensor*>();
  auto C = args[2].cast<DLTensor*>();
  bool transa = args[3].cast<bool>();
  bool transb = args[4].cast<bool>();
  TVM_FFI_ICHECK_EQ(A->ndim, 2);
  TVM_FFI_ICHECK_EQ(B->ndim, 2);
  TVM_FFI_ICHECK_EQ(C->ndim, 2);

  TVM_FFI_ICHECK_EQ(ElementStride(A), 1);
  TVM_FFI_ICHECK_EQ(ElementStride(B), 1);
  TVM_FFI_ICHECK_EQ(ElementStride(C), 1);

  TVM_FFI_ICHECK(A->dtype == B->dtype);

  // C can never be transposed.
  TVM_FFI_ICHECK(!IsInPlaceTransposed(C));

  // Reversed strides indicates an in-place transpose operation.
  transa = IsInPlaceTransposed(A) ? !transa : transa;
  transb = IsInPlaceTransposed(B) ? !transb : transb;

  TVM_FFI_ICHECK(CheckMixPrecisionType(A->dtype, C->dtype)) << "Unsupported data type";
  TVM_FFI_ICHECK((!(A->dtype == DLDataType{kDLInt, 8, 1}) || ColumnStride(A) % 4 == 0))
      << "leading dimension must divide 4 for int8 gemm";
  TVM_FFI_ICHECK((!(B->dtype == DLDataType{kDLInt, 8, 1}) || ColumnStride(B) % 4 == 0))
      << "leading dimension must divide 4 for int8 gemm";
  double alpha = args.size() > 5 ? args[5].cast<double>() : 1.0;
  double beta = args.size() > 6 ? args[6].cast<double>() : 0.0;

  hipblasDatatype_t hip_in_type = GetHipBlasDataType(A->dtype);
  hipblasDatatype_t hip_out_type = GetHipBlasDataType(C->dtype);
  hipblasGemmAlgo_t algo = HIPBLAS_GEMM_DEFAULT;
  void *alpha_ptr = nullptr, *beta_ptr = nullptr;
  auto alpha_int = static_cast<int32_t>(alpha);
  auto beta_int = static_cast<int32_t>(beta);
  auto alpha_float = static_cast<float>(alpha);
  auto beta_float = static_cast<float>(beta);
  if (C->dtype.code == kDLInt) {
    alpha_ptr = &alpha_int;
    beta_ptr = &beta_int;
  } else if (C->dtype.code == kDLFloat) {
    alpha_ptr = &alpha_float;
    beta_ptr = &beta_float;
  }

  auto A_data = reinterpret_cast<void*>(static_cast<char*>(A->data) + A->byte_offset);
  auto B_data = reinterpret_cast<void*>(static_cast<char*>(B->data) + B->byte_offset);
  auto C_data = reinterpret_cast<void*>(static_cast<char*>(C->data) + C->byte_offset);

  CHECK_HIPBLAS_ERROR(
      hipblasGemmEx(hdl, HIPBLASBooleanToTranspose(transb), HIPBLASBooleanToTranspose(transa),
                    ColumnCount(B, transb), RowCount(A, transa), ColumnCount(A, transa), alpha_ptr,
                    B_data, hip_in_type, ColumnStride(B), A_data, hip_in_type, ColumnStride(A),
                    beta_ptr, C_data, hip_out_type, ColumnStride(C), hip_out_type, algo));
}

inline void CallBatchGemmEx(ffi::PackedArgs args, ffi::Any* ret, hipblasHandle_t hdl) {
  auto A = args[0].cast<DLTensor*>();
  auto B = args[1].cast<DLTensor*>();
  auto C = args[2].cast<DLTensor*>();
  bool transa = args[3].cast<bool>();
  bool transb = args[4].cast<bool>();
  TVM_FFI_ICHECK_EQ(A->ndim, 3);
  TVM_FFI_ICHECK_EQ(B->ndim, 3);
  TVM_FFI_ICHECK_EQ(C->ndim, 3);

  int batch_size = BatchCount3D(C);
  TVM_FFI_ICHECK_EQ(ElementStride3D(A), 1);
  TVM_FFI_ICHECK_EQ(ElementStride3D(B), 1);
  TVM_FFI_ICHECK_EQ(ElementStride3D(C), 1);

  TVM_FFI_ICHECK(A->dtype == B->dtype);

  // C can never be transposed.
  TVM_FFI_ICHECK(!IsInPlaceTransposed3D(C));

  // Reversed strides indicates an in-place transpose operation.
  transa = IsInPlaceTransposed3D(A) ? !transa : transa;
  transb = IsInPlaceTransposed3D(B) ? !transb : transb;

  TVM_FFI_ICHECK(CheckMixPrecisionType(A->dtype, C->dtype, true)) << "Unsupported data type";
  TVM_FFI_ICHECK((!(A->dtype == DLDataType{kDLInt, 8, 1}) || ColumnStride3D(A) % 4 == 0))
      << "leading dimension must divide 4 for int8 gemm";
  TVM_FFI_ICHECK((!(B->dtype == DLDataType{kDLInt, 8, 1}) || ColumnStride3D(B) % 4 == 0))
      << "leading dimension must divide 4 for int8 gemm";
  double alpha = args.size() > 5 ? args[5].cast<double>() : 1.0;
  double beta = args.size() > 6 ? args[6].cast<double>() : 0.0;

  int A_stride = A->shape[1] * A->shape[2];
  int B_stride = B->shape[1] * B->shape[2];
  int C_stride = C->shape[1] * C->shape[2];

  // Broadcast A or B by changing its stride.
  int batch_size_a = BatchCount3D(A);
  int batch_size_b = BatchCount3D(B);
  if (batch_size_a != batch_size_b) {
    if (batch_size_a == 1) {
      A_stride = 0;
    } else if (batch_size_b == 1) {
      B_stride = 0;
    }
  } else {
    TVM_FFI_ICHECK_EQ(batch_size_a, batch_size);
    TVM_FFI_ICHECK_EQ(batch_size_b, batch_size);
  }

  hipblasDatatype_t hip_in_type = GetHipBlasDataType(A->dtype);
  hipblasDatatype_t hip_out_type = GetHipBlasDataType(C->dtype);
  hipblasGemmAlgo_t algo = HIPBLAS_GEMM_DEFAULT;
  void *alpha_ptr = nullptr, *beta_ptr = nullptr;
  auto alpha_int = static_cast<int32_t>(alpha);
  auto beta_int = static_cast<int32_t>(beta);
  auto alpha_float = static_cast<float>(alpha);
  auto beta_float = static_cast<float>(beta);
  if (C->dtype.code == kDLInt) {
    alpha_ptr = &alpha_int;
    beta_ptr = &beta_int;
  } else if (C->dtype.code == kDLFloat) {
    alpha_ptr = &alpha_float;
    beta_ptr = &beta_float;
  }

  auto A_data = reinterpret_cast<void*>(static_cast<char*>(A->data) + A->byte_offset);
  auto B_data = reinterpret_cast<void*>(static_cast<char*>(B->data) + B->byte_offset);
  auto C_data = reinterpret_cast<void*>(static_cast<char*>(C->data) + C->byte_offset);
  CHECK_HIPBLAS_ERROR(hipblasGemmStridedBatchedEx(
      hdl, HIPBLASBooleanToTranspose(transb), HIPBLASBooleanToTranspose(transa),
      ColumnCount3D(B, transb), RowCount3D(A, transa), ColumnCount3D(A, transa), alpha_ptr, B_data,
      hip_in_type, ColumnStride3D(B), B_stride, A_data, hip_in_type, ColumnStride3D(A), A_stride,
      beta_ptr, C_data, hip_out_type, ColumnStride3D(C), C_stride, batch_size, hip_out_type, algo));
}

// matrix multiplication for row major
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("tvm.contrib.hipblas.matmul",
                  [](ffi::PackedArgs args, ffi::Any* ret) {
                    auto A = args[0].cast<DLTensor*>();
                    auto C = args[2].cast<DLTensor*>();

                    HipBlasThreadEntry* entry_ptr = HipBlasThreadEntry::ThreadLocal(A->device);

                    if (A->dtype == C->dtype) {
                      TVM_FFI_ICHECK((A->dtype == DLDataType{kDLFloat, 16, 1} ||
                                      A->dtype == DLDataType{kDLFloat, 32, 1} ||
                                      A->dtype == DLDataType{kDLFloat, 64, 1}));

                      if (A->dtype == DLDataType{kDLFloat, 16, 1}) {
                        CallGemm(args, ret, HipblasHgemmOp(entry_ptr->handle));
                      } else if (A->dtype == DLDataType{kDLFloat, 32, 1}) {
                        CallGemm(args, ret, HipblasSgemmOp(entry_ptr->handle));
                      } else {
                        CallGemm(args, ret, HipblasDgemmOp(entry_ptr->handle));
                      }
                    } else {
                      CallGemmEx(args, ret, entry_ptr->handle);
                    }
                  })
      .def_packed("tvm.contrib.hipblas.batch_matmul", [](ffi::PackedArgs args, ffi::Any* ret) {
        auto A = args[0].cast<DLTensor*>();
        auto C = args[2].cast<DLTensor*>();

        HipBlasThreadEntry* entry_ptr = HipBlasThreadEntry::ThreadLocal(A->device);

        if (A->dtype == C->dtype) {
          TVM_FFI_ICHECK((A->dtype == DLDataType{kDLFloat, 16, 1} ||
                          A->dtype == DLDataType{kDLFloat, 32, 1} ||
                          A->dtype == DLDataType{kDLFloat, 64, 1}));

          if (A->dtype == DLDataType{kDLFloat, 16, 1}) {
            CallBatchGemm(args, ret, HipblasHgemmBatchOp(entry_ptr->handle));
          } else if (A->dtype == DLDataType{kDLFloat, 32, 1}) {
            CallBatchGemm(args, ret, HipblasSgemmBatchOp(entry_ptr->handle));
          } else {
            CallBatchGemm(args, ret, HipblasDgemmBatchOp(entry_ptr->handle));
          }
        } else {
          CallBatchGemmEx(args, ret, entry_ptr->handle);
        }
      });
}

}  // namespace contrib
}  // namespace tvm
