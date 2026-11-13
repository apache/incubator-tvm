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
#include <tvm/s_tir/analysis.h>

#include "../utils.h"

namespace tvm {
namespace s_tir {
namespace meta_schedule {

class VerifyVTCMLimitObj : public PostprocObj {
 public:
  int64_t vtcm_capacity = 0;

  void InitializeWithTuneContext(const TuneContext& context) final {
    TVM_FFI_ICHECK(context->target.has_value());
    Target target = context->target.value();
    TVM_FFI_ICHECK(target->kind->name == "hexagon");
    // The value of 0 will disable VTCM verification.
    vtcm_capacity = target->GetAttr<int64_t>("vtcm-capacity").value_or(0);
  }

  bool Verify(const IRModule& mod) const {
    if (!s_tir::VerifyVTCMLimit(mod, vtcm_capacity)) {
      return false;
    }
    return true;
  }

  bool Apply(const s_tir::Schedule& sch) final {
    IRModule mod = sch->mod();
    auto pass_list = s_tir::GetVTCMCompactionPasses();
    tvm::transform::PassContext pass_ctx = tvm::transform::PassContext::Current();
    IRModule lowered = tvm::transform::Sequential(pass_list)(std::move(mod));
    if (!Verify(lowered)) {
      return false;
    }
    return true;
  }

  Postproc Clone() const {
    ffi::ObjectPtr<VerifyVTCMLimitObj> n = ffi::make_object<VerifyVTCMLimitObj>(*this);
    return Postproc(ffi::UnsafeInit{}, n);
  }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<VerifyVTCMLimitObj>();
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.VerifyVTCMLimit", VerifyVTCMLimitObj,
                                    PostprocObj);
};

Postproc Postproc::VerifyVTCMLimit() {
  ffi::ObjectPtr<VerifyVTCMLimitObj> n = ffi::make_object<VerifyVTCMLimitObj>();
  return Postproc(ffi::UnsafeInit{}, n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  VerifyVTCMLimitObj::RegisterReflection();
  refl::GlobalDef().def("s_tir.meta_schedule.PostprocVerifyVTCMLimit", Postproc::VerifyVTCMLimit);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
