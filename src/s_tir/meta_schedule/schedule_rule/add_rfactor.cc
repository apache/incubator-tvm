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

class AddRFactorObj : public ScheduleRuleObj {
 public:
  // Inherited from ScheduleRuleObj
  void InitializeWithTuneContext(const TuneContext& context) final {
    TVM_FFI_ICHECK(context->target.has_value());
    Target target = context->target.value();
    this->max_parallel_basic_ = GetTargetNumCores(target);
    if (this->max_jobs_per_core != -1) {
      this->max_parallel_extent_ = max_parallel_basic_ * max_jobs_per_core;
    }
  }

  // Inherited from ScheduleRuleObj
  ffi::Array<s_tir::Schedule> Apply(const s_tir::Schedule& sch, const s_tir::SBlockRV& block_rv);

  // Inherited from ScheduleRuleObj
  ScheduleRule Clone() const final {
    ffi::ObjectPtr<AddRFactorObj> n = ffi::make_object<AddRFactorObj>(*this);
    return ScheduleRule(ffi::UnsafeInit{}, n);
  }

 public:
  /*!
   * \brief The maximum number of jobs to be launched per core.
   * It sets the uplimit of parallelism, i.e. `num_cores * max_jobs_per_core`.
   * Use -1 to disable parallelism.
   */
  int max_jobs_per_core;
  /*! \brief The maximum size of the innermost factor */
  int max_innermost_factor;
  /*! \brief The number of uplimit of parallelism. */
  int max_parallel_extent_;
  /*! \brief The number of cores. */
  int max_parallel_basic_;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AddRFactorObj>()
        .def_ro("max_jobs_per_core", &AddRFactorObj::max_jobs_per_core)
        .def_ro("max_innermost_factor", &AddRFactorObj::max_innermost_factor);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.AddRFactor", AddRFactorObj,
                                    ScheduleRuleObj);
};

ScheduleRule ScheduleRule::AddRFactor(int max_jobs_per_core,
                                      ffi::Optional<int64_t> max_innermost_factor) {
  ffi::ObjectPtr<AddRFactorObj> n = ffi::make_object<AddRFactorObj>();
  n->max_jobs_per_core = max_jobs_per_core;
  n->max_innermost_factor = max_innermost_factor.value_or(-1);
  n->max_parallel_extent_ = -1;
  n->max_parallel_basic_ = -1;
  return ScheduleRule(ffi::UnsafeInit{}, n);
}

ffi::Array<s_tir::Schedule> AddRFactorObj::Apply(const s_tir::Schedule& sch,
                                                 const s_tir::SBlockRV& block_rv) {
  tirx::StmtSRef block_sref = sch->GetSRef(block_rv);
  if (!NeedsRFactorOrCrossThreadReduction(sch->state(), block_sref, max_parallel_extent_,
                                          max_parallel_basic_)) {
    return {sch};
  }

  // Make a copy of the original schedule.
  s_tir::Schedule ori_sch = sch->Copy();
  ori_sch->Seed(sch->ForkSeed());

  // Reorder the loop axes if reduction loops are not innermost.
  // After the reordering, fuse all the reduction loops.
  size_t num_spatial_loops;
  s_tir::LoopRV fused_reduce_loop;
  ReorderAndFuseReductionLoops(sch, block_rv, &fused_reduce_loop, &num_spatial_loops);

  // Split the fused reduction loop.
  ffi::Array<ffi::Optional<s_tir::ExprRV>> factors =
      sch->SamplePerfectTile(fused_reduce_loop, 2, max_innermost_factor);
  ffi::Array<s_tir::LoopRV> split_loops = sch->Split(fused_reduce_loop, factors);

  ffi::Array<s_tir::Schedule> res;
  for (const s_tir::LoopRV& split_loop : split_loops) {
    s_tir::Schedule sch_tmp = sch->Copy();
    sch_tmp->Seed(sch->ForkSeed());
    try {
      const s_tir::SBlockRV& block_rf = sch_tmp->RFactor(split_loop, num_spatial_loops);
      ffi::Array<s_tir::LoopRV> axes = sch_tmp->GetLoops(block_rf);
      TVM_FFI_ICHECK_GT(axes.size(), num_spatial_loops);

      // Annotate that the rfactor block, which is now the producer of the original block, needs to
      // be considered by the rule Random-Compute-Location.
      sch_tmp->Annotate(block_rv, tvm::s_tir::attr::kMetaScheduleRandomComputeProducer.name, 1);
      res.push_back(sch_tmp);
    } catch (const tvm::ffi::Error& e) {
    }
  }

  res.push_back(ori_sch);
  return res;
}

TVM_FFI_STATIC_INIT_BLOCK() { AddRFactorObj::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.meta_schedule.ScheduleRuleAddRFactor", ScheduleRule::AddRFactor);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
