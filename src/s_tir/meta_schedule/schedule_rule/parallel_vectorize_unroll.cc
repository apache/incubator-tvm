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
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/stmt.h>

#include "../utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

bool IsRootBlock(const Schedule& sch, const SBlockRV& block_rv) {
  StmtSRef block_sref = sch->GetSRef(block_rv);
  return block_sref->parent == nullptr;
}

bool CheckSpatialFunction(const Schedule& sch, const SBlockRV& root_block_rv) {
  return IsSpatialFunction(
      ffi::GetRef<Function>(GetRootFunction(sch->mod(), sch->Get(root_block_rv).get(), nullptr)));
}

}  // namespace s_tir
}  // namespace tvm

namespace tvm {
namespace s_tir {
namespace meta_schedule {

class ParallelizeVectorizeUnrollObj : public ScheduleRuleObj {
 public:
  // Inherited from ScheduleRuleObj
  void InitializeWithTuneContext(const TuneContext& context) final {
    TVM_FFI_ICHECK(context->target.has_value());
    if (this->max_jobs_per_core != -1) {
      Target target = context->target.value();
      this->max_parallel_extent_ = GetTargetNumCores(target) * max_jobs_per_core;
    }
  }

  // Inherited from ScheduleRuleObj
  ffi::Array<s_tir::Schedule> Apply(const s_tir::Schedule& sch, const s_tir::SBlockRV& root_rv) {
    // Currently only mark the root block with annotations.
    if (!s_tir::IsRootBlock(sch, root_rv)) {
      return {sch};
    }

    // Parallelization
    if (max_jobs_per_core != -1) {
      sch->Annotate(root_rv, tvm::s_tir::attr::kMetaScheduleParallel.name,
                    this->max_parallel_extent_);
    }
    // Vectorization
    if (max_vectorize_extent != -1) {
      sch->Annotate(root_rv, tvm::s_tir::attr::kMetaScheduleVectorize.name, max_vectorize_extent);
    }
    // Unroll
    if (!unroll_max_steps.empty() && !s_tir::CheckSpatialFunction(sch, root_rv)) {
      int n = unroll_max_steps.size();
      double prob = 1.0 / n;
      ffi::Array<FloatImm> probs(n, FloatImm(PrimType::Float(32), prob));
      PrimExpr max_step = sch->SampleCategorical(unroll_max_steps, probs);
      if (unroll_explicit) {
        sch->Annotate(root_rv, tvm::s_tir::attr::kMetaScheduleUnrollExplicit.name, max_step);
      } else {
        sch->Annotate(root_rv, tvm::s_tir::attr::kMetaScheduleUnrollImplicit.name, max_step);
      }
    }
    return {sch};
  }

  // Inherited from ScheduleRuleObj
  ScheduleRule Clone() const final {
    ffi::ObjectPtr<ParallelizeVectorizeUnrollObj> n =
        ffi::make_object<ParallelizeVectorizeUnrollObj>(*this);
    return ScheduleRule(ffi::UnsafeInit{}, n);
  }

 public:
  /*!
   * \brief The maximum number of jobs to be launched per CPU core. It sets the
   * upper limit of CPU parallelism, i.e. `num_cores * max_jobs_per_core`. Use -1 to disable
   * parallelism.
   */
  int64_t max_jobs_per_core;
  /*!
   * \brief The maximum extent to be vectorized.
   * It sets the upper limit of the hardware target vectorization. Use -1 to disable vectorization.
   */
  int max_vectorize_extent;
  /*!
   * \brief The options of the maximum number of unroll steps to be done.
   * Use an empty array to disable unroll.
   */
  ffi::Array<int64_t> unroll_max_steps;
  /*! \brief Whether to explicitly unroll the loop, or just add an "unroll" pragma. */
  bool unroll_explicit;
  /*! \brief The number of maximum available jobs in CPU. */
  int64_t max_parallel_extent_;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ParallelizeVectorizeUnrollObj>()
        .def_ro("max_jobs_per_core", &ParallelizeVectorizeUnrollObj::max_jobs_per_core)
        .def_ro("max_vectorize_extent", &ParallelizeVectorizeUnrollObj::max_vectorize_extent)
        .def_ro("unroll_max_steps", &ParallelizeVectorizeUnrollObj::unroll_max_steps)
        .def_ro("tirx.unroll_explicit", &ParallelizeVectorizeUnrollObj::unroll_explicit);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.ParallelizeVectorizeUnroll",
                                    ParallelizeVectorizeUnrollObj, ScheduleRuleObj);
};

ScheduleRule ScheduleRule::ParallelizeVectorizeUnroll(int max_jobs_per_core,
                                                      int max_vectorize_extent,
                                                      ffi::Array<int64_t> unroll_max_steps,
                                                      bool unroll_explicit) {
  ffi::ObjectPtr<ParallelizeVectorizeUnrollObj> n =
      ffi::make_object<ParallelizeVectorizeUnrollObj>();
  n->max_jobs_per_core = max_jobs_per_core;
  n->max_vectorize_extent = max_vectorize_extent;
  n->unroll_max_steps = unroll_max_steps;
  n->unroll_explicit = unroll_explicit;
  n->max_parallel_extent_ = -1;
  return ScheduleRule(ffi::UnsafeInit{}, n);
}

TVM_FFI_STATIC_INIT_BLOCK() { ParallelizeVectorizeUnrollObj::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.meta_schedule.ScheduleRuleParallelizeVectorizeUnroll",
                        ScheduleRule::ParallelizeVectorizeUnroll);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
