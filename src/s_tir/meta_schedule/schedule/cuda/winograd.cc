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
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/meta_schedule/schedule/cuda/thread_bind.h>
#include <tvm/s_tir/meta_schedule/schedule/generic/winograd.h>

#include <vector>

#include "../../utils.h"

namespace tvm {
namespace s_tir {
namespace meta_schedule {

using namespace tvm::tirx;
using s_tir::ExprRV;
using s_tir::LoopRV;
using s_tir::SBlockRV;
using s_tir::Schedule;

static ffi::Array<s_tir::LoopRV> ScheduleDataPack(s_tir::Schedule sch, s_tir::SBlockRV block,
                                                  std::vector<int> tiled,
                                                  std::vector<int> unrolled) {
  // This method is used for NHWC layout only. Will likely be refactored into a more schedule
  using namespace tvm::tirx;
  TVM_FFI_ICHECK_EQ(tiled.size(), 2);
  TVM_FFI_ICHECK_EQ(unrolled.size(), 4);
  ffi::Array<ffi::Optional<ExprRV>> factors{ffi::UnsafeInit()};
  ffi::Array<LoopRV> loops = sch->GetLoops(block);
  TVM_FFI_ICHECK_EQ(loops.size(), 6);

  factors = sch->SamplePerfectTile(loops[tiled[0]], /*n=*/2, /*max_innermost_factor=*/64);
  ffi::Array<LoopRV> t0 = sch->Split(loops[tiled[0]], factors);
  TVM_FFI_ICHECK_EQ(t0.size(), 2);

  factors = sch->SamplePerfectTile(loops[tiled[1]], /*n=*/2, /*max_innermost_factor=*/64);
  ffi::Array<LoopRV> t1 = sch->Split(loops[tiled[1]], factors);
  TVM_FFI_ICHECK_EQ(t1.size(), 2);

  sch->Unroll(loops[unrolled[0]]);
  sch->Unroll(loops[unrolled[1]]);
  sch->Unroll(loops[unrolled[2]]);
  sch->Unroll(loops[unrolled[3]]);
  sch->Reorder({
      t0[0],
      t1[0],
      t0[1],
      t1[1],
      loops[unrolled[0]],
      loops[unrolled[1]],
      loops[unrolled[2]],
      loops[unrolled[3]],
  });
  return {t0[0], t1[0], t0[1], t1[1]};
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("s_tir.meta_schedule.cuda.conv2d_nhwc_winograd_data_pack",
           [](Schedule sch, SBlockRV data_pack) -> ffi::Array<Schedule> {
             SBlockRV input_tile = GetWinogradProducerAndInlineConst(sch, data_pack);
             SBlockRV data_pad = GetWinogradProducerAndInlineConst(sch, input_tile);
             ffi::Array<LoopRV> loops = ScheduleDataPack(sch, data_pack, {2, 3}, {0, 1, 4, 5});
             {
               SBlockRV data_pack_local = sch->CacheWrite(data_pack, 0, "local");
               sch->ReverseComputeAt(data_pack_local, loops.back(), /*preserve_unit_loops=*/true);
             }
             {
               sch->ComputeAt(input_tile, /*loop_rv=*/loops.back(), /*preserve_unit_loops=*/true);
               sch->SetScope(input_tile, /*tensor_index=*/0, /*storage_scope=*/"local");
               sch->ComputeInline(data_pad);
             }
             {
               int64_t max_threadblocks = 256;
               int64_t max_threads_per_block = 1024;
               ffi::Array<LoopRV> loops = sch->GetLoops(data_pack);
               TVM_FFI_ICHECK_EQ(loops.size(), 8);
               BindSpatialLoop(sch, sch->Fuse({loops[0], loops[1], loops[2], loops[3]}),
                               max_threadblocks, max_threads_per_block);
             }
             return {sch};
           })
      .def("s_tir.meta_schedule.cuda.conv2d_nhwc_winograd_inverse",
           [](Schedule sch, SBlockRV inverse) -> ffi::Array<Schedule> {
             GetWinogradProducerAndInlineConst(sch, inverse);
             ScheduleDataPack(sch, inverse, /*tiled=*/{2, 3}, /*unrolled=*/{0, 1, 4, 5});
             int64_t max_threadblocks = 256;
             int64_t max_threads_per_block = 1024;
             ffi::Array<LoopRV> loops = sch->GetLoops(inverse);
             TVM_FFI_ICHECK_EQ(loops.size(), 8);
             BindSpatialLoop(sch, sch->Fuse({loops[0], loops[1], loops[2], loops[3]}),
                             max_threadblocks, max_threads_per_block);
             return {sch};
           })
      .def("s_tir.meta_schedule.cuda.conv2d_nchw_winograd_data_pack",
           [](Schedule sch, SBlockRV data_pack) -> ffi::Array<Schedule> {
             int64_t max_threadblocks = 256;
             int64_t max_threads_per_block = 1024;
             SBlockRV input_tile = GetWinogradProducerAndInlineConst(sch, data_pack);
             SBlockRV data_pad = GetWinogradProducerAndInlineConst(sch, input_tile);
             LoopRV outer{ffi::UnsafeInit()};
             {
               ffi::Array<LoopRV> loops = sch->GetLoops(data_pack);
               TVM_FFI_ICHECK_EQ(loops.size(), 6);
               sch->Reorder({loops[2], loops[3], loops[0], loops[1], loops[4], loops[5]});
               sch->Unroll(loops[0]);
               sch->Unroll(loops[1]);
               sch->Unroll(loops[4]);
               sch->Unroll(loops[5]);
               outer = BindSpatialLoop(sch, sch->Fuse({loops[2], loops[3]}), max_threadblocks,
                                       max_threads_per_block, /*get_factor=*/nullptr)
                           .back();
             }
             {
               SBlockRV data_pack_local = sch->CacheWrite(data_pack, 0, "local");
               sch->ReverseComputeAt(data_pack_local, outer, /*preserve_unit_loops=*/true);
             }
             {
               sch->ComputeAt(input_tile, /*loop_rv=*/outer, /*preserve_unit_loops=*/true);
               sch->SetScope(input_tile, /*tensor_index=*/0, /*storage_scope=*/"local");
               sch->ComputeInline(data_pad);
             }
             return {sch};
           })
      .def(
          "s_tir.meta_schedule.cuda.conv2d_nchw_winograd_inverse",
          [](Schedule sch, SBlockRV inverse) -> ffi::Array<Schedule> {
            GetWinogradProducerAndInlineConst(sch, inverse);
            // loops on top of the inverse block: [CO, P, tile_size, tile_size, alpha, alpha]
            ffi::BigInt tile_size = sch->Get(inverse)
                                        ->writes[0]
                                        ->source.as_or_throw<tvm::tirx::TensorVar>()
                                        .shape()[2]
                                        .as_or_throw<IntImm>()
                                        ->value;
            LoopRV outer{ffi::UnsafeInit()};
            {
              SBlockRV output = sch->GetConsumers(inverse)[0];
              ffi::Array<LoopRV> nchw = sch->GetLoops(output);
              TVM_FFI_ICHECK_EQ(nchw.size(), 4);
              ffi::Array<LoopRV> hs = sch->Split(nchw[2], {std::nullopt, IntImm::Int32(tile_size)});
              ffi::Array<LoopRV> ws = sch->Split(nchw[3], {std::nullopt, IntImm::Int32(tile_size)});
              sch->Reorder({hs[0], ws[0], hs[1], ws[1]});
              outer = ws[0];
            }
            {
              sch->ComputeAt(inverse, /*loop_rv=*/outer, /*preserve_unit_loops=*/true);
              sch->SetScope(inverse, /*tensor_index=*/0, /*storage_scope=*/"local");
              ffi::Array<LoopRV> loops = sch->GetLoops(inverse);
              TVM_FFI_ICHECK_EQ(loops.size(), 10);
              sch->Unroll(loops[6]);
              sch->Unroll(loops[7]);
              sch->Unroll(loops[8]);
              sch->Unroll(loops[9]);
            }
            return {sch};
          });
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
