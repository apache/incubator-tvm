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
#include "cutlass/float8.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass_extensions/gemm/collective/collective_builder.hpp"
#include "cutlass_extensions/gemm/dispatch_policy.hpp"
// clang-format on

#define CUTLASS_CHECK(status)                                      \
  {                                                                \
    cutlass::Status error = status;                                \
    TVM_FFI_ICHECK(error == cutlass::Status::kSuccess)             \
        << "Got cutlass error: " << cutlassGetStatusString(error); \
  }

using namespace cute;
using ProblemShape = Shape<int, int, int, int>;
using tvm::ffi::Tensor;

template <typename TileShape, typename ClusterShape, typename ElementD, typename SchedulerType,
          int ScaleGranularityM = 1>
struct CutlassFP8GroupwiseScaledGemmRunner {
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementBlockScale = float;

  using ElementA = cutlass::float_e4m3_t;
  using LayoutA = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = cutlass::float_e4m3_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ElementC = void;
  using LayoutC = cutlass::layout::RowMajor;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementD>::value;

  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
  using StoreEpilogueCompute =
      typename cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90AccFetch>;

  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8BlockScaledSubGroupMAccum<
          ScaleGranularityM>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, TileShape, ClusterShape, EpilogueTileType, ElementAccumulator,
      ElementCompute, ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
      EpilogueSchedule, StoreEpilogueCompute>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
      ElementAccumulator, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,  // Indicates ProblemShape
                                           CollectiveMainloop, CollectiveEpilogue, SchedulerType>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  void run_gemm(const ElementA* a_ptr, const ElementB* b_ptr, const ElementBlockScale* scales_a_ptr,
                const ElementBlockScale* scales_b_ptr, ElementD* o_ptr, ProblemShape* problem_size,
                StrideA* stride_a, StrideB* stride_b, StrideD* stride_d, uint8_t* workspace,
                int64_t workspace_size, cudaStream_t stream) {
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    typename Gemm::GemmKernel::TileSchedulerArguments scheduler;
    static constexpr bool UsesStreamKScheduler =
        cute::is_same_v<typename Gemm::GemmKernel::TileSchedulerTag,
                        cutlass::gemm::StreamKScheduler>;
    if constexpr (UsesStreamKScheduler) {
      using DecompositionMode = typename cutlass::gemm::kernel::detail::
          PersistentTileSchedulerSm90StreamKParams::DecompositionMode;
      using ReductionMode = typename cutlass::gemm::kernel::detail::
          PersistentTileSchedulerSm90StreamKParams::ReductionMode;
      scheduler.decomposition_mode = DecompositionMode::StreamK;
      scheduler.reduction_mode = ReductionMode::Nondeterministic;
    }

    typename Gemm::Arguments arguments = {
        cutlass::gemm::GemmUniversalMode::kGemm,
        *problem_size,
        {a_ptr, *stride_a, b_ptr, *stride_b, scales_a_ptr, scales_b_ptr},
        {{}, nullptr, *stride_d, o_ptr, *stride_d},
        hw_info,
        scheduler};

    Gemm gemm_op;
    CUTLASS_CHECK(gemm_op.can_implement(arguments));
    TVM_FFI_ICHECK_GE(workspace_size, gemm_op.get_workspace_size(arguments));
    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace, stream));
    CUTLASS_CHECK(gemm_op.run(stream));
  }
};

template <typename TileShape, typename ClusterShape, typename ElementA, typename ElementB,
          typename ElementD, typename ElementBlockScale>
void cutlass_fp8_groupwise_scaled_mm_sm90(ElementA* a, ElementB* b, ElementBlockScale* scales_a,
                                          ElementBlockScale* scales_b, ElementD* out,
                                          uint8_t* workspace, int64_t workspace_size, int64_t m,
                                          int64_t n, int64_t k, int64_t l, cudaStream_t stream) {
  if (k > 3 * n) {
    using SchedulerType = cutlass::gemm::StreamKScheduler;
    using Runner =
        CutlassFP8GroupwiseScaledGemmRunner<TileShape, ClusterShape, ElementD, SchedulerType>;
    using StrideA = typename Runner::StrideA;
    using StrideB = typename Runner::StrideB;
    using StrideD = typename Runner::StrideD;

    Runner runner;
    StrideA stride_a = cute::make_stride(k, Int<1>{}, m * k);
    StrideB stride_b = cute::make_stride(k, Int<1>{}, n * k);
    StrideD stride_d = cute::make_stride(n, Int<1>{}, m * n);
    ProblemShape problem_size{static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                              static_cast<int>(l)};
    runner.run_gemm(a, b, scales_a, scales_b, out, &problem_size, &stride_a, &stride_b, &stride_d,
                    workspace, workspace_size, stream);
  } else {
    using SchedulerType = cutlass::gemm::PersistentScheduler;
    using Runner =
        CutlassFP8GroupwiseScaledGemmRunner<TileShape, ClusterShape, ElementD, SchedulerType>;
    using StrideA = typename Runner::StrideA;
    using StrideB = typename Runner::StrideB;
    using StrideD = typename Runner::StrideD;

    Runner runner;
    StrideA stride_a = cute::make_stride(k, Int<1>{}, m * k);
    StrideB stride_b = cute::make_stride(k, Int<1>{}, n * k);
    StrideD stride_d = cute::make_stride(n, Int<1>{}, m * n);
    ProblemShape problem_size{static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                              static_cast<int>(l)};
    runner.run_gemm(a, b, scales_a, scales_b, out, &problem_size, &stride_a, &stride_b, &stride_d,
                    workspace, workspace_size, stream);
  }
}
