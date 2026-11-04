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
#include <tvm/s_tir/stmt.h>

#include "../../schedule/analysis.h"
#include "../../schedule/transform.h"
#include "../utils.h"
#include "multi_level_tiling.h"

namespace tvm {
namespace s_tir {
namespace meta_schedule {

using s_tir::LoopRV;
using s_tir::SBlockRV;
using s_tir::Schedule;

/*!
 * \brief Extension of MultiLevelTiling for backends with wide vectors.
 * The loop over the innermost spatial axis of the output tensor is always vectorized with the
 * maximum vector length.
 */
class MultiLevelTilingWideVectorObj : public MultiLevelTilingObj {
 public:
  size_t vector_length_in_bits;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<MultiLevelTilingWideVectorObj>();
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.MultiLevelTilingWideVector",
                                    MultiLevelTilingWideVectorObj, MultiLevelTilingObj);

 protected:
  ScheduleRule Clone() const final {
    ffi::ObjectPtr<MultiLevelTilingWideVectorObj> n =
        ffi::make_object<MultiLevelTilingWideVectorObj>(*this);
    return ScheduleRule(ffi::UnsafeInit{}, n);
  }

  std::pair<ffi::Array<ffi::Optional<s_tir::ExprRV>>, ffi::Array<s_tir::LoopRV>> SplitLoop(
      const Schedule& sch, SBlockRV block, LoopRV loop, int n_tiles) const;
};

std::pair<ffi::Array<ffi::Optional<s_tir::ExprRV>>, ffi::Array<s_tir::LoopRV>>
MultiLevelTilingWideVectorObj::SplitLoop(const Schedule& sch, SBlockRV block_rv, LoopRV loop_rv,
                                         int n_tiles) const {
  const ForObj* loop = TVM_SREF_TO_FOR(sch->GetSRef(loop_rv));
  const tirx::StmtSRef block_sref = sch->GetSRef(block_rv);
  const s_tir::SBlockObj* block_node = block_sref->StmtAs<s_tir::SBlockObj>();
  const s_tir::SBlockRealize block_realize = s_tir::GetSBlockRealize(sch->state(), block_sref);
  TVM_FFI_ICHECK(block_node && block_node->writes.size() == 1);

  const auto out_dtype = block_node->writes[0]->source.as_or_throw<tvm::tirx::TensorVar>().dtype();
  const int vec_len = vector_length_in_bits / out_dtype.bits();

  // Determine if this loop is over the innermost axis of the output tensor.
  // In the example below, we look for a loop whose loop var is bound to the axis co.

  // for (i0, 0, 1) {
  //    for (i1, 0, 56) {
  //      for (i2, 0, 56) {
  //        for (i3, 0, 64) {
  //          for (i4, 0, 3) {
  //            for (i5, 0, 3) {
  //              for (i6, 0, 64) {
  //                block conv2d_nhwc(...) {
  //                  ...
  //                  bind(co, i3)
  //                  ...
  //                  writes([conv2d_nhwc[n, h, w, co]])
  //                  ...
  //                  conv2d_nhwc[n, h, w, co] = ...
  // }
  const size_t innermost_axis = block_node->writes[0]->region.size() - 1;
  const PrimExpr innermost_iter_value = block_realize->iter_values[innermost_axis];

  if (!sym::Analyzer()->CanProve(static_cast<PrimExpr>(loop->var) == innermost_iter_value)) {
    // If this is not the innermost spatial loop, split the loop in the normal way.
    return MultiLevelTilingObj::SplitLoop(sch, block_rv, loop_rv, n_tiles);
  } else {
    // We split the innermost spatial loop in a way that always uses the maximum vector length.
    PrimExpr extent_int_imm_expr = Range::GetExtentFromBeginEnd(loop->begin, loop->end);
    const auto* extent_int_imm = extent_int_imm_expr.as<IntImmObj>();
    auto extent_int = extent_int_imm ? extent_int_imm->value.as<int64_t>() : std::nullopt;
    if (extent_int.has_value() && *extent_int > vec_len) {
      ffi::Array<s_tir::LoopRV> inner_splits =
          sch->Split(/*loop=*/loop_rv,
                     /*factors=*/{std::nullopt, PrimExpr(vec_len)});
      ffi::Array<ffi::Optional<s_tir::ExprRV>> outer_factors = sch->SamplePerfectTile(
          /*loop=*/inner_splits[0],
          /*n=*/n_tiles - 1,
          /*max_innermost_factor=*/max_innermost_factor);
      ffi::Array<s_tir::LoopRV> outer_splits = sch->Split(
          /*loop=*/inner_splits[0], /*factors=*/outer_factors);
      outer_splits.push_back(inner_splits[1]);
      outer_factors.push_back(PrimExpr(vec_len));
      return {outer_factors, outer_splits};
    } else {
      ffi::Array<ffi::Optional<s_tir::ExprRV>> factors(n_tiles - 1, PrimExpr(1));
      factors.push_back(Range::GetExtentFromBeginEnd(loop->begin, loop->end));
      ffi::Array<s_tir::LoopRV> splits = sch->Split(/*loop=*/loop_rv,
                                                    /*factors=*/factors);
      return {factors, splits};
    }
  }
}

ScheduleRule ScheduleRule::MultiLevelTilingWideVector(
    ffi::String structure, int64_t vector_length_in_bits,
    ffi::Optional<int64_t> max_innermost_factor,
    ffi::Optional<ffi::Map<ffi::String, ffi::Any>> reuse_read,
    ffi::Optional<ffi::Map<ffi::String, ffi::Any>> reuse_write) {
  auto node = MultiLevelTilingInitCommon<MultiLevelTilingWideVectorObj>(
      structure, std::nullopt, max_innermost_factor, std::nullopt, reuse_read, reuse_write);
  node->vector_length_in_bits = vector_length_in_bits;
  return ScheduleRule(ffi::UnsafeInit{}, node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  MultiLevelTilingWideVectorObj::RegisterReflection();
  refl::GlobalDef().def("s_tir.meta_schedule.ScheduleRuleMultiLevelTilingWideVector",
                        ScheduleRule::MultiLevelTilingWideVector);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
