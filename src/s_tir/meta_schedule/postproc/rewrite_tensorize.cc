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
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/meta_schedule/postproc.h>
#include <tvm/s_tir/stmt.h>

#include <algorithm>

#include "../utils.h"

namespace tvm {
namespace s_tir {
namespace meta_schedule {

using s_tir::LoopRV;
using s_tir::SBlockRV;

void CollectTensorizationJobs(
    const s_tir::Schedule& sch, const ffi::String& func_name, const tirx::FunctionObj* func,
    bool vectorize_init_loop,
    std::vector<std::tuple<ffi::String, ffi::String, std::function<void(s_tir::SBlockRV)>>>* jobs) {
  auto walk_fn = [=, &jobs](const s_tir::SBlock& block) -> ffi::Expected<ffi::WalkResult> {
    tirx::StmtSRef block_sref = sch->GetSRef(block.get());
    std::string block_name = block_sref->StmtAs<s_tir::SBlockObj>()->name_hint;
    if (ffi::Optional<ffi::String> intrin_name = s_tir::GetAnn<ffi::String>(
            block_sref, tvm::s_tir::attr::kMetaScheduleAutoTensorize.name)) {
      if (intrin_name.value() != "") {
        jobs->emplace_back(block_name, func_name, [sch, intrin_name](s_tir::SBlockRV block) {
          try {
            sch->Tensorize(block, intrin_name.value());
          } catch (const std::exception& e) {
            LOG(WARNING) << "Tensorize failed with error " << e.what();
          }
        });
      } else if (block_name.find("init") && vectorize_init_loop) {
        jobs->emplace_back(block_name, func_name, [sch](s_tir::SBlockRV block) {
          ffi::Array<SBlockRV> child_blocks = sch->GetChildBlocks(block);
          TVM_FFI_ICHECK(child_blocks.size() == 1);
          ffi::Array<LoopRV> init_loops = sch->GetLoops(child_blocks[0]);
          TVM_FFI_ICHECK(init_loops.size() == 1);
          sch->Vectorize(init_loops[0]);
        });
      }
    }
    return ffi::WalkResult::Advance();
  };
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(func->body, walk_fn);
}

class RewriteTensorizeObj : public PostprocObj {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<RewriteTensorizeObj>();
  }

  void InitializeWithTuneContext(const TuneContext& context) final {}

  bool Apply(const s_tir::Schedule& sch) final;

  Postproc Clone() const {
    ffi::ObjectPtr<RewriteTensorizeObj> n = ffi::make_object<RewriteTensorizeObj>(*this);
    return Postproc(ffi::UnsafeInit{}, n);
  }

  bool vectorize_init_loop = false;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.RewriteTensorize", RewriteTensorizeObj,
                                    PostprocObj);
};

bool RewriteTensorizeObj::Apply(const s_tir::Schedule& sch) {
  // The rewriting jobs, 3-tuple (block_name, func_name, job_func)
  std::vector<std::tuple<ffi::String, ffi::String, std::function<void(s_tir::SBlockRV)>>> jobs;
  for (const auto& kv : sch->mod()->functions) {
    GlobalVar g_var = kv.first;
    BaseFunc base_func = kv.second;
    if (const tirx::FunctionObj* function = base_func.as<tirx::FunctionObj>()) {
      CollectTensorizationJobs(sch, g_var->name_hint, function, vectorize_init_loop, &jobs);
    }
  }
  for (const auto& job : jobs) {
    const ffi::String& block_name = std::get<0>(job);
    const ffi::String& func_name = std::get<1>(job);
    const auto& job_func = std::get<2>(job);
    SBlockRV block = sch->GetSBlock(block_name, func_name);
    sch->Unannotate(block, tvm::s_tir::attr::kMetaScheduleAutoTensorize.name);
    job_func(block);
  }
  return true;
}

Postproc Postproc::RewriteTensorize(bool vectorize_init_loop) {
  ffi::ObjectPtr<RewriteTensorizeObj> n = ffi::make_object<RewriteTensorizeObj>();
  n->vectorize_init_loop = vectorize_init_loop;
  return Postproc(ffi::UnsafeInit{}, n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  RewriteTensorizeObj::RegisterReflection();
  refl::GlobalDef().def("s_tir.meta_schedule.PostprocRewriteTensorize", Postproc::RewriteTensorize);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
