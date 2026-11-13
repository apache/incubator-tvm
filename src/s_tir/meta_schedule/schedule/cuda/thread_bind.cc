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
#include <tvm/ir/prim/op.h>
#include <tvm/s_tir/meta_schedule/schedule/cuda/thread_bind.h>
#include <tvm/s_tir/schedule/schedule.h>
#include <tvm/support/gpu_thread_tag.h>

#include <algorithm>
#include <limits>
#include <utility>

#include "../../utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::prim;
namespace meta_schedule {

using namespace tvm::tirx;
using s_tir::ExprRV;
using s_tir::GetLoopIterType;
using s_tir::GetLoops;
using s_tir::GetThreadBindingTag;
using s_tir::HasBeenMultiLevelTiled;
using s_tir::IsBlockIdx;
using s_tir::IsSingleStmt;
using s_tir::IsThreadIdx;
using s_tir::LoopRV;
using s_tir::SBlockRV;
using s_tir::Schedule;

std::function<ExprRV(int64_t)> MakeFactorSampler(Schedule sch, ffi::Array<int64_t> thread_extents) {
  return [sch = std::move(sch),
          thread_extents = std::move(thread_extents)](int64_t max_extent) -> ExprRV {
    ffi::Array<int64_t> extents;
    extents.reserve(thread_extents.size());
    for (int64_t extent : thread_extents) {
      if (extent <= max_extent) {
        extents.push_back(extent);
      }
    }
    int n = extents.size();
    if (n == 0) {
      return IntImm::Int32(max_extent);
    }
    if (n == 1) {
      return IntImm::Int32(extents[0]);
    }
    ffi::Array<FloatImm> probs(n, FloatImm(PrimType::Float(32), 1.0 / n));
    return sch->SampleCategorical(extents, probs);
  };
}

ffi::Array<LoopRV> BindSpatialLoop(Schedule sch, LoopRV loop, int64_t max_threadblocks,
                                   int64_t max_threads_per_block,
                                   std::function<ExprRV(int64_t)> get_factor) {
  int64_t extent = -1;
  PrimExpr e_imm_expr = Range::GetExtentFromBeginEnd(sch->Get(loop)->begin, sch->Get(loop)->end);
  const auto* e_imm = e_imm_expr.as<IntImmObj>();
  if (auto e = e_imm ? e_imm->value.as<int64_t>() : std::nullopt; e.has_value()) {
    extent = *e;
  } else {
    extent = std::numeric_limits<int64_t>::max();
  }
  if (extent <= max_threadblocks * max_threads_per_block) {
    if (!get_factor) {
      get_factor = MakeFactorSampler(sch, {32, 64, 128, 256, 512, 1024});
    }
    ExprRV factor = get_factor(std::min(extent, max_threads_per_block));
    ffi::Array<LoopRV> splits = sch->Split(loop, {std::nullopt, factor});
    TVM_FFI_ICHECK_EQ(splits.size(), 2);
    sch->Bind(splits[0], "blockIdx.x");
    sch->Bind(splits[1], "threadIdx.x");
    return {splits[0], splits[1]};
  } else {
    ffi::Array<LoopRV> splits = sch->Split(loop, {std::nullopt, IntImm::Int32(max_threadblocks),  //
                                                  IntImm::Int32(max_threads_per_block)});
    TVM_FFI_ICHECK_EQ(splits.size(), 3);
    sch->Reorder({splits[1], splits[2], splits[0]});
    sch->Bind(splits[1], "blockIdx.x");
    sch->Bind(splits[2], "threadIdx.x");
    return {splits[1], splits[2], splits[0]};
  }
}

void BindBlockThreadIdx(Schedule sch, SBlockRV block_rv,  //
                        int64_t max_threadblocks, int64_t max_threads_per_block,
                        std::function<ExprRV(int64_t)> get_factor) {
  StmtSRef block_sref = sch->GetSRef(block_rv);
  if (block_sref->parent == nullptr) {
    return;
  }
  if (HasBeenMultiLevelTiled(block_sref)) {
    return;
  }
  ffi::Array<StmtSRef> loops = GetLoops(block_sref);
  int n = loops.size();
  int i_block_idx = -1;
  int i_thread_idx = -1;
  int i_multi_child = -1;
  int i_spatial_loop = -1;
  for (int i = 0; i < n; ++i) {
    const StmtSRef& loop_sref = loops[i];
    const ForObj* loop = TVM_SREF_TO_FOR(loop_sref);
    support::GPUThreadTag thread_scope = GetThreadBindingTag(loop);
    if (IsBlockIdx(thread_scope)) {
      if (i_block_idx == -1) {
        i_block_idx = i;
      }
    }
    if (IsThreadIdx(thread_scope)) {
      if (i_thread_idx == -1) {
        i_thread_idx = i;
      }
    }
    if (loop->kind != ForKind::kDefault) {
      if (i_multi_child == -1) {
        i_multi_child = i;
      }
    }
    if (!IsSingleStmt(loop->body)) {
      if (i_multi_child == -1) {
        i_multi_child = i + 1;
      }
    }
    if (GetLoopIterType(loop_sref) == IterVarType::kDataPar) {
      if (i_spatial_loop == i - 1) {
        ++i_spatial_loop;
      }
    }
  }
  if (i_multi_child == -1) {
    i_multi_child = n;
  }
  if (i_block_idx != -1 && i_thread_idx != -1) {
    return;
  }
  if (i_block_idx != -1 && i_thread_idx == -1) {
    TVM_FFI_ICHECK(false) << "Unsupported case, where blockIdx is bound but threadIdx is not";
    throw;
  }
  LoopRV loop_rv{ffi::UnsafeInit()};
  {
    ffi::Array<LoopRV> loop_rvs = sch->GetLoops(block_rv);
    if (i_spatial_loop == -1) {
      LoopRV spatial_loop_rv{ffi::UnsafeInit()};
      if (loop_rvs.empty()) {
        spatial_loop_rv = sch->AddUnitLoop(block_rv);
      } else {
        spatial_loop_rv = sch->AddUnitLoop(loop_rvs[0]);
      }
      loop_rvs.insert(loop_rvs.begin(), spatial_loop_rv);
      i_spatial_loop = 0;
      if (i_block_idx != -1) {
        i_block_idx += 1;
      }
      if (i_thread_idx != -1) {
        i_thread_idx += 1;
      }
      if (i_multi_child != -1) {
        i_multi_child += 1;
      }
    }
    if (i_block_idx == -1 && i_thread_idx != -1) {
      int num_fuse = std::min(std::min(i_multi_child, i_thread_idx), i_spatial_loop + 1);
      ffi::Array<LoopRV> loop_rvs = sch->GetLoops(block_rv);
      loop_rv = sch->Fuse({loop_rvs.begin(), loop_rvs.begin() + num_fuse});
      sch->Bind(loop_rv, "blockIdx.x");
      return;
    } else {  // i_block_idx == -1 && i_thread_idx == -1
      int num_fuse = std::min(i_multi_child, i_spatial_loop + 1);
      loop_rv = sch->Fuse({loop_rvs.begin(), loop_rvs.begin() + num_fuse});
    }
  }
  BindSpatialLoop(sch, loop_rv, max_threadblocks, max_threads_per_block, get_factor);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
