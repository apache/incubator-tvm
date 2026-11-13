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
using namespace tvm::tirx;

/*! \brief Check if an IRModule has any dynamic loop. */
struct DynamicExtentFinder : public StmtExprVisitor {
  using StmtExprVisitor::Visit_;

  static bool Find(const IRModule& mod) {
    auto finder = ffi::make_object<DynamicExtentFinder>();
    for (const auto& kv : mod->functions) {
      const BaseFunc& func = kv.second;
      if (const auto* function = func.as<FunctionObj>()) {
        finder->Visit(function->body);
        if (finder->found_) {
          return true;
        }
      }
    }
    return false;
  }

 private:
  ffi::Optional<VisitInterrupt> Visit_(const ForObj* loop) final {
    if (!Range::GetExtentFromBeginEnd(loop->begin, loop->end)->IsInstance<IntImmObj>()) {
      found_ = true;
    } else {
      return StmtExprVisitor::Visit_(loop);
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit(ffi::AnyView stmt) final {
    if (stmt.as<ExprObj>()) return std::nullopt;
    if (!found_) {
      return StmtExprVisitor::Visit(stmt);
    }
    return std::nullopt;
  }

  bool found_ = false;
};

}  // namespace s_tir

namespace s_tir {
namespace meta_schedule {

/*! \brief Check if the IRModule has any loop with non-constant extent. */
class DisallowDynamicLoopObj : public PostprocObj {
 public:
  // Inherited from PostprocObj
  void InitializeWithTuneContext(const TuneContext& context) final {}
  // Inherited from PostprocObj
  bool Apply(const s_tir::Schedule& sch) final {
    return !s_tir::DynamicExtentFinder::Find(sch->mod());
  }
  // Inherited from PostprocObj
  Postproc Clone() const {
    ffi::ObjectPtr<DisallowDynamicLoopObj> n = ffi::make_object<DisallowDynamicLoopObj>(*this);
    return Postproc(ffi::UnsafeInit{}, n);
  }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DisallowDynamicLoopObj>();
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.DisallowDynamicLoop",
                                    DisallowDynamicLoopObj, PostprocObj);
};

Postproc Postproc::DisallowDynamicLoop() {
  ffi::ObjectPtr<DisallowDynamicLoopObj> n = ffi::make_object<DisallowDynamicLoopObj>();
  return Postproc(ffi::UnsafeInit{}, n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  DisallowDynamicLoopObj::RegisterReflection();
  refl::GlobalDef().def("s_tir.meta_schedule.PostprocDisallowDynamicLoop",
                        Postproc::DisallowDynamicLoop);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
