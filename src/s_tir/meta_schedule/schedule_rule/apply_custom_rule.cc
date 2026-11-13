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
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/stmt.h>

#include "../utils.h"

namespace tvm {
namespace s_tir {
namespace meta_schedule {

class ApplyCustomRuleObj : public ScheduleRuleObj {
 public:
  // Inherited from ScheduleRuleObj
  void InitializeWithTuneContext(const TuneContext& context) final {
    TVM_FFI_CHECK(context->target.has_value(), ValueError)
        << "Target is not defined in the tune context.";
    this->target_ = context->target;
  }

  static std::string GetCustomRuleName(const std::string& name, const std::string& key) {
    return "s_tir.meta_schedule." + key + "." + name;
  }

  // Inherited from ScheduleRuleObj
  ffi::Array<s_tir::Schedule> Apply(const s_tir::Schedule& sch,
                                    const s_tir::SBlockRV& block_rv) final {
    TVM_FFI_CHECK(this->target_.has_value(), ValueError)
        << "ApplyCustomRule is not initialized with TuneContext that has a Target.";
    ffi::Array<ffi::String> keys = this->target_.value()->keys;
    if (ffi::Optional<ffi::String> ann = s_tir::GetAnn<ffi::String>(
            sch->GetSRef(block_rv), tvm::s_tir::attr::kScheduleRule.name)) {
      if (ann.value() != "None") {
        for (const ffi::String& key : keys) {
          if (const auto custom_schedule_fn =
                  tvm::ffi::Function::GetGlobal(GetCustomRuleName(ann.value(), key))) {
            ffi::Array<s_tir::Schedule> result =
                (*custom_schedule_fn)(sch, block_rv).cast<ffi::Array<s_tir::Schedule>>();
            return result;
          }
        }
        std::ostringstream os;
        os << "Unknown schedule rule \"" << ann.value() << "\" for target keys \"" << keys
           << "\". Checked ffi::Functions:";
        for (const ffi::String& key : keys) {
          os << "\n  " << GetCustomRuleName(ann.value(), key);
        }
        LOG(WARNING) << os.str();
        sch->Unannotate(block_rv, tvm::s_tir::attr::kScheduleRule.name);
      }
    }
    return {sch};
  }

  // Inherited from ScheduleRuleObj
  ScheduleRule Clone() const final {
    ffi::ObjectPtr<ApplyCustomRuleObj> n = ffi::make_object<ApplyCustomRuleObj>(*this);
    n->target_ = target_;
    return ScheduleRule(ffi::UnsafeInit{}, n);
  }

 public:
  ffi::Optional<Target> target_ = std::nullopt;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ApplyCustomRuleObj>().def_ro("target_", &ApplyCustomRuleObj::target_);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.ApplyCustomRule", ApplyCustomRuleObj,
                                    ScheduleRuleObj);
};

ScheduleRule ScheduleRule::ApplyCustomRule() {
  ffi::ObjectPtr<ApplyCustomRuleObj> n = ffi::make_object<ApplyCustomRuleObj>();
  return ScheduleRule(ffi::UnsafeInit{}, n);
}

bool ScheduleRule::IsApplyCustomRule(const ScheduleRule& rule) {
  return rule->IsInstance<ApplyCustomRuleObj>();
}

TVM_FFI_STATIC_INIT_BLOCK() { ApplyCustomRuleObj::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.meta_schedule.ScheduleRuleApplyCustomRule",
                        ScheduleRule::ApplyCustomRule);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
