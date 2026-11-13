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

#include "../utils.h"

namespace tvm {
namespace s_tir {
namespace meta_schedule {

class RandomComputeLocationObj : public ScheduleRuleObj {
 public:
  // Inherited from ScheduleRuleObj
  void InitializeWithTuneContext(const TuneContext& context) final {}

  // Inherited from ScheduleRuleObj
  ffi::Array<s_tir::Schedule> Apply(const s_tir::Schedule& sch,
                                    const s_tir::SBlockRV& block_rv) final {
    if (!CheckConditions(sch, block_rv)) {
      return {sch};
    }

    // Step 1. If the producer of the input block needs a random compute-at location (specified by
    // the annotation), we collect the producer first, and transform the producer block later.
    // - The reason we collect the producer before transforming the input block is that, if the
    // decision of Sample-Compute-Location is "compute-inline" for the input block, we can no longer
    // access the input block. Hence we collect its producer ahead of time.
    // - Note that only single producer is allowed in this case.
    ffi::Array<s_tir::SBlockRV> producers;
    if (s_tir::HasAnn(sch->GetSRef(block_rv),
                      tvm::s_tir::attr::kMetaScheduleRandomComputeProducer.name, true)) {
      producers = sch->GetProducers(block_rv);
      sch->Unannotate(block_rv, tvm::s_tir::attr::kMetaScheduleRandomComputeProducer.name);
      TVM_FFI_ICHECK_EQ(producers.size(), 1);
    }

    // Step 2. Transform the input block.
    s_tir::Schedule res = RandomlyComputeAt(sch, block_rv);

    // Step 3. Transform the producer block if compute-location sampling is needed.
    if (!producers.empty()) {
      res = RandomlyComputeAt(res, producers[0]);
    }

    return {res};
  }

  // Inherited from ScheduleRuleObj
  ScheduleRule Clone() const final {
    ffi::ObjectPtr<RandomComputeLocationObj> n = ffi::make_object<RandomComputeLocationObj>(*this);
    return ScheduleRule(ffi::UnsafeInit{}, n);
  }

 private:
  bool CheckConditions(const s_tir::Schedule sch, const s_tir::SBlockRV& block_rv) const {
    tirx::StmtSRef block_sref = sch->GetSRef(block_rv);
    TVM_SREF_TO_SBLOCK(block_sref);

    // Cond 1. The block is not the root block.
    if (block_sref->parent == nullptr) {
      return false;
    }
    // Cond 2. The block should be the direct child block of the root block.
    if (s_tir::GetScopeRoot(sch->state(), block_sref,
                            /*require_stage_pipeline=*/false)
            ->parent != nullptr) {
      return false;
    }
    // Cond 3 & 4. The block has at least one outer loop, and the outermost loop has only one child
    // block.
    ffi::Array<tirx::StmtSRef> loop_srefs = s_tir::GetLoops(block_sref);
    if (loop_srefs.empty()) {
      return false;
    }
    if (s_tir::GetChildBlockSRefOnSRefTree(sch->state(), loop_srefs[0]).size() > 1) {
      return false;
    }
    // Cond 5. The block is not tiled. We check this condition by examine the block's annotation.
    if (s_tir::HasBeenMultiLevelTiled(block_sref)) {
      return false;
    }
    // Cond 6. The block has at lease one consumer.
    if (s_tir::GetConsumers(sch->state(), sch->GetSRef(block_rv)).empty()) {
      return false;
    }
    return true;
  }

  /*!
   * \brief Keep sampling a compute-at location for the input block until success.
   * \param sch The TIR schedule
   * \param block_rv The block whose compute-at location is to be sampled
   * \return The TIR schedule after transformation
   */
  s_tir::Schedule RandomlyComputeAt(const s_tir::Schedule& sch, const s_tir::SBlockRV& block_rv) {
    s_tir::LoopRV compute_at_loc = sch->SampleComputeLocation(block_rv);
    sch->ComputeAt(block_rv, compute_at_loc, true);
    return sch;
  }

 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<RandomComputeLocationObj>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.RandomComputeLocation",
                                    RandomComputeLocationObj, ScheduleRuleObj);
};

ScheduleRule ScheduleRule::RandomComputeLocation() {
  return ScheduleRule(ffi::UnsafeInit{}, ffi::make_object<RandomComputeLocationObj>());
}

TVM_FFI_STATIC_INIT_BLOCK() { RandomComputeLocationObj::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.meta_schedule.ScheduleRuleRandomComputeLocation",
                        ScheduleRule::RandomComputeLocation);
}
}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
