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
#include <tvm/s_tir/tensor_intrin.h>

#include "../../schedule/analysis.h"
#include "../../schedule/transform.h"
#include "../utils.h"
#include "multi_level_tiling.h"

namespace tvm {
namespace s_tir {
namespace meta_schedule {

/*!
 * \brief Tile a subset of loops in the block according to the given tensor intrinsic, and annotate
 * the tiled block for tensorization by postproc rewrite.
 */
ffi::Optional<s_tir::SBlockRV> TileForIntrin(s_tir::Schedule sch, s_tir::SBlockRV block,
                                             const std::string& intrin_name) {
  ffi::Optional<s_tir::LoopRV> tiled_loop_rv = TileWithTensorIntrin(sch, block, intrin_name);
  if (!tiled_loop_rv) {
    return std::nullopt;
  }
  TVM_FFI_ICHECK(tiled_loop_rv.has_value());
  s_tir::SBlockRV outer_block = sch->Blockize(tiled_loop_rv.value());
  sch->Annotate(outer_block, tvm::s_tir::attr::kMetaScheduleAutoTensorize.name,
                ffi::String(intrin_name));
  return outer_block;
}

/*!
 * \brief Extension of MultiLevelTiling for auto-tensorizing with a single intrinsic.
 */
class MultiLevelTilingWithIntrinObj : public MultiLevelTilingObj {
 protected:
  ffi::Array<s_tir::Schedule> Apply(const s_tir::Schedule& sch,
                                    const s_tir::SBlockRV& block_rv) final {
    auto desc_func = TensorIntrin::Get(intrin_name).value()->desc;
    if (!CheckAutoTensorizeApplicable(sch, block_rv, desc_func)) {
      TVM_PY_LOG(INFO, logger) << "The workload cannot be tensorized.";
      return {sch};
    }

    auto res = MultiLevelTilingObj::Apply(sch->Copy(), block_rv);

    if (res.empty()) {
      TVM_PY_LOG(INFO, logger) << "The workload cannot be tensorized.";
      return {sch};
    }
    TVM_PY_LOG(INFO, logger) << "Tensorizing with " << intrin_name;
    return res;
  }

  // Inherited from ScheduleRuleObj
  ScheduleRule Clone() const final {
    ffi::ObjectPtr<MultiLevelTilingWithIntrinObj> n =
        ffi::make_object<MultiLevelTilingWithIntrinObj>(*this);
    return ScheduleRule(ffi::UnsafeInit{}, n);
  }

  // Override ApplySubRules to tile the inner loops according to the given tensor intrinsic, then
  // tile the outerloops.
  virtual std::vector<State> ApplySubRules(std::vector<State> states) {
    states = SubRule(std::move(states), [&](State state) {
      if (auto block_rv = TileForIntrin(state->sch, state->block_rv, intrin_name)) {
        state->block_rv = block_rv.value();
        return std::vector<State>(1, state);
      }
      return std::vector<State>();
    });
    return MultiLevelTilingObj::ApplySubRules(states);
  }

 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<MultiLevelTilingWithIntrinObj>();
  }

  /*! \brief The name of a tensor intrinsic. */
  ffi::String intrin_name;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.MultiLevelTilingWithIntrin",
                                    MultiLevelTilingWithIntrinObj, MultiLevelTilingObj);
};

ScheduleRule ScheduleRule::MultiLevelTilingWithIntrin(
    ffi::String intrin_name, ffi::String structure,
    ffi::Optional<ffi::Array<ffi::String>> tile_binds, ffi::Optional<int64_t> max_innermost_factor,
    ffi::Optional<ffi::Array<int64_t>> vector_load_lens,
    ffi::Optional<ffi::Map<ffi::String, ffi::Any>> reuse_read,
    ffi::Optional<ffi::Map<ffi::String, ffi::Any>> reuse_write) {
  TVM_FFI_ICHECK(TensorIntrin::Get(intrin_name).has_value())
      << "Provided tensor intrinsic " << intrin_name << " is not registered.";
  auto node = MultiLevelTilingInitCommon<MultiLevelTilingWithIntrinObj>(
      structure, tile_binds, max_innermost_factor, vector_load_lens, reuse_read, reuse_write);
  node->intrin_name = intrin_name;
  return ScheduleRule(ffi::UnsafeInit{}, node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  MultiLevelTilingWithIntrinObj::RegisterReflection();
  refl::GlobalDef().def("s_tir.meta_schedule.ScheduleRuleMultiLevelTilingWithIntrin",
                        ScheduleRule::MultiLevelTilingWithIntrin);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
