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

#include "../utils.h"

namespace tvm {
namespace s_tir {
namespace meta_schedule {

/*! \brief The union of design space generators. */
class ScheduleFnObj : public SpaceGeneratorObj {
 public:
  /*! \brief The random state. -1 means using random number. */
  TRandState rand_state_ = -1;
  /*! \brief The schedule function. */
  ffi::Function schedule_fn_;

  static void RegisterReflection() {
    // `schedule_fn_` is not registered.
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ScheduleFnObj>();
  }

  void InitializeWithTuneContext(const TuneContext& context) final {
    SpaceGeneratorObj::InitializeWithTuneContext(context);
    this->rand_state_ = ForkSeed(&context->rand_state);
  }

  ffi::Array<s_tir::Schedule> GenerateDesignSpace(const IRModule& mod) final {
    s_tir::Schedule sch = s_tir::Schedule::Traced(
        /*mod=*/mod,
        /*rand_state=*/ForkSeed(&this->rand_state_),
        /*debug_mode=*/0,
        /*error_render_level=*/s_tir::ScheduleErrorRenderLevel::kDetail);
    ffi::Any rv;
    rv = this->schedule_fn_(sch);
    if (rv == nullptr) {
      return {sch};
    }
    ffi::ObjectRef obj = rv.cast<ffi::ObjectRef>();
    if (auto sch = obj.as<s_tir::Schedule>()) {
      return {sch.value()};
    }
    if (const auto* arr = obj.as<ffi::ArrayObj>()) {
      ffi::Array<s_tir::Schedule> result;
      result.reserve(arr->size());
      for (Any val : *arr) {
        if (auto sch = val.as<s_tir::Schedule>()) {
          result.push_back(sch.value());
        } else {
          TVM_FFI_THROW(TypeError) << "Expect return type of ScheduleFn to be None, Schedule or "
                                      "List[Schedule], but got: "
                                   << obj->GetTypeKey();
        }
      }
      return result;
    }
    TVM_FFI_THROW(TypeError) << "Expect return type of ScheduleFn to be None, Schedule or "
                                "List[Schedule], but got: "
                             << obj->GetTypeKey();
    throw;
  }

  SpaceGenerator Clone() const final {
    ffi::ObjectPtr<ScheduleFnObj> n = ffi::make_object<ScheduleFnObj>(*this);
    CloneRules(this, n.get());
    return SpaceGenerator(n);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.ScheduleFn", ScheduleFnObj,
                                    SpaceGeneratorObj);
};

SpaceGenerator SpaceGenerator::ScheduleFn(
    ffi::Function schedule_fn, ffi::Optional<ffi::Array<ScheduleRule>> sch_rules,
    ffi::Optional<ffi::Array<Postproc>> postprocs,
    ffi::Optional<ffi::Map<Mutator, FloatImm>> mutator_probs) {
  ffi::ObjectPtr<ScheduleFnObj> n = ffi::make_object<ScheduleFnObj>();
  n->sch_rules = std::move(sch_rules);
  n->postprocs = std::move(postprocs);
  n->mutator_probs = std::move(mutator_probs);
  n->schedule_fn_ = std::move(schedule_fn);
  return SpaceGenerator(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  ScheduleFnObj::RegisterReflection();
  refl::GlobalDef().def("s_tir.meta_schedule.SpaceGeneratorScheduleFn", SpaceGenerator::ScheduleFn);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
