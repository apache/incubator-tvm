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

#include <tvm/ffi/extra/cuda/base.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <variant>
#include <vector>

// clang-format off
#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
// clang-format on

#define CUTLASS_CHECK(status)                                      \
  {                                                                \
    cutlass::Status error = status;                                \
    TVM_FFI_ICHECK(error == cutlass::Status::kSuccess)             \
        << "Got cutlass error: " << cutlassGetStatusString(error); \
  }

using namespace cute;
using tvm::ffi::Tensor;

template <typename TileShape, typename ClusterShape, typename ElementD>
struct CutlassFP8ScaledGroupwiseGemmRunnerSM100 {
  using ElementA = cutlass::float_e4m3_t;
  using LayoutA = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = cutlass::float_e4m3_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ElementC = void;
  using LayoutC = cutlass::layout::RowMajor;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementD>::value;

  using LayoutD = LayoutC;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  // MMA type
  using ElementAccumulator = float;  // Element Accumulator will also be our scale factor type
  using ElementCompute = float;
  using ElementBlockScale = float;

  static constexpr int ScaleGranularityM = 1;
  static constexpr int ScaleGranularityN = 128;
  static constexpr int ScaleGranularityK = 128;
  using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<
      ScaleGranularityM, ScaleGranularityN, ScaleGranularityK, UMMA::Major::MN, UMMA::Major::K>;

  using LayoutSFA =
      decltype(ScaleConfig::deduce_layoutSFA());  // Layout type for SFA matrix operand
  using LayoutSFB =
      decltype(ScaleConfig::deduce_layoutSFB());  // Layout type for SFB matrix operand
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute, ElementC,
      LayoutC, AlignmentC, ElementD, LayoutC, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, ElementA,
      cute::tuple<LayoutA, LayoutSFA>, AlignmentA, ElementB, cute::tuple<LayoutB, LayoutSFB>,
      AlignmentB, ElementAccumulator, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelScheduleSm100Blockwise>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      void>;  // Default to ClusterLaunchControl (CLC) based tile scheduler

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  void run_gemm(const ElementA* a_ptr, const ElementB* b_ptr, const ElementBlockScale* scales_a_ptr,
                const ElementBlockScale* scales_b_ptr, ElementD* o_ptr, int m, int n, int k, int l,
                uint8_t* workspace, int64_t workspace_size, cudaStream_t stream) {
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    StrideA stride_a =
        cute::make_stride(static_cast<int64_t>(k), Int<1>{}, static_cast<int64_t>(m * k));
    StrideB stride_b =
        cute::make_stride(static_cast<int64_t>(k), Int<1>{}, static_cast<int64_t>(n * k));
    StrideD stride_d =
        cute::make_stride(static_cast<int64_t>(n), Int<1>{}, static_cast<int64_t>(m * n));
    auto layout_scales_a = ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, l));
    auto layout_scales_b = ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, l));

    typename Gemm::Arguments arguments = {cutlass::gemm::GemmUniversalMode::kGemm,
                                          {m, n, k, l},
                                          {a_ptr, stride_a, b_ptr, stride_b, scales_a_ptr,
                                           layout_scales_a, scales_b_ptr, layout_scales_b},
                                          {{}, o_ptr, stride_d, o_ptr, stride_d},
                                          hw_info};

    Gemm gemm_op;
    CUTLASS_CHECK(gemm_op.can_implement(arguments));
    TVM_FFI_ICHECK_GE(workspace_size, gemm_op.get_workspace_size(arguments));
    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace, stream));
    CUTLASS_CHECK(gemm_op.run(stream));
  }
};

template <typename TileShape, typename ClusterShape, typename ElementA, typename ElementB,
          typename ElementD, typename ElementBlockScale>
void cutlass_fp8_groupwise_scaled_mm_sm100(ElementA* a, ElementB* b, ElementBlockScale* scales_a,
                                           ElementBlockScale* scales_b, ElementD* out,
                                           uint8_t* workspace, int64_t workspace_size, int64_t m,
                                           int64_t n, int64_t k, int64_t l, cudaStream_t stream) {
  using Runner = CutlassFP8ScaledGroupwiseGemmRunnerSM100<TileShape, ClusterShape, ElementD>;
  Runner runner;
  runner.run_gemm(a, b, scales_a, scales_b, out, m, n, k, l, workspace, workspace_size, stream);
}
