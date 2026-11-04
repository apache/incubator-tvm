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

/*!
 * \file tvm/relax/transform/meta_schedule.cc
 * \brief Pass for meta_schedule tuning
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/transform.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/function.h>
#include <tvm/s_tir/meta_schedule/database.h>
#include <tvm/tirx/transform.h>

#include "../../s_tir/meta_schedule/module_equality.h"
#include "../../s_tir/meta_schedule/trace_apply.h"

namespace tvm {
namespace relax {
namespace transform {

class MetaScheduleTuner {
 public:
  explicit MetaScheduleTuner(Target target, ffi::String work_dir, int64_t max_trials_global,
                             int64_t max_trials_per_task,
                             ffi::Optional<ffi::Array<ffi::String>> op_names,
                             ffi::Map<ffi::String, ffi::Tensor> params = {})
      : target_(target),
        work_dir_(work_dir),
        max_trials_global_(max_trials_global),
        max_trials_per_task_(max_trials_per_task),
        op_names_(op_names),
        params_(params) {
    normalize_mod_func_ =
        tvm::ffi::Function::GetGlobalRequired("tvm.s_tir.meta_schedule.normalize_mod");
  }

  IRModule TuneIRMod(IRModule mod, transform::PassContext ctx) {
    static ffi::Function tune_relax_func =
        tvm::ffi::Function::GetGlobalRequired("tvm.s_tir.meta_schedule.tune_relax");
    tune_relax_func(mod, params_, target_, work_dir_, max_trials_global_, max_trials_per_task_,
                    op_names_);
    return mod;
  }

  tirx::Function TuneTIR(tirx::Function f, transform::PassContext ctx) {
    static ffi::Function tune_tir_func =
        tvm::ffi::Function::GetGlobalRequired("tvm.s_tir.meta_schedule.tune_tir");
    tune_tir_func(normalize_mod_func_(f), target_, work_dir_, max_trials_global_);
    return f;
  }

 private:
  Target target_;
  ffi::String work_dir_;
  int64_t max_trials_global_;
  int64_t max_trials_per_task_;
  ffi::Optional<ffi::Array<ffi::String>> op_names_;
  ffi::Map<ffi::String, ffi::Tensor> params_;
  tvm::ffi::Function normalize_mod_func_;
};

Pass MetaScheduleApplyDatabase(ffi::Optional<ffi::String> work_dir, bool enable_warning = false) {
  using tvm::s_tir::meta_schedule::Database;
  Target target = Target::Current(false).value();
  const std::optional<tvm::ffi::Function> normalize_mod_func_ =
      tvm::ffi::Function::GetGlobalRequired("tvm.s_tir.meta_schedule.normalize_mod");
  TVM_FFI_ICHECK(normalize_mod_func_.has_value()) << "Normalization function is not found.";

  auto pass_func = [=](IRModule mod, PassContext ctx) {
    Database database{ffi::UnsafeInit()};
    if (Database::Current().has_value()) {
      database = Database::Current().value();
    } else {
      TVM_FFI_ICHECK(work_dir.has_value());
      std::filesystem::create_directories(work_dir.value().c_str());
      ffi::String path_workload = work_dir.value() + "/database_workload.json";
      ffi::String path_tuning_record = work_dir.value() + "/database_tuning_record.json";
      LOG(WARNING) << "Creating JSONDatabase. Workload at: " << path_workload
                   << ", Tuning records at: " << path_tuning_record;
      database =
          s_tir::meta_schedule::Database::JSONDatabase(path_workload, path_tuning_record, true);
    }

    ffi::Map<GlobalVar, BaseFunc> result;
    auto mod_eq_structural = s_tir::meta_schedule::ModuleEquality::Create("ignore-tensor");
    for (const auto& iter : mod->functions) {
      GlobalVar gv = iter.first;
      BaseFunc base_func = iter.second;
      if (const auto* function_node = base_func.as<tirx::FunctionObj>()) {
        tirx::Function function = ffi::GetRef<tirx::Function>(function_node);

        IRModule tir_mod = (*normalize_mod_func_)(function).cast<IRModule>();
        if (ffi::Optional<s_tir::meta_schedule::TuningRecord> opt_record =
                database->QueryTuningRecord(tir_mod, target, gv->name_hint)) {
          s_tir::meta_schedule::TuningRecord record = opt_record.value();
          bool structural_equal = mod_eq_structural->Equal(tir_mod, record->workload->mod);
          s_tir::Schedule sch = s_tir::Schedule::Traced(
              structural_equal ? record->workload->mod : tir_mod, /*seed=*/-1, /*debug_mask=*/0,
              /*error_render_level=*/s_tir::ScheduleErrorRenderLevel::kDetail);
          if (!structural_equal) {
            // When the database lookup succeeds while structural equality check fails,
            // it implies that the anchor block based equality has been used during tuning.
            // The trace in the record cannot directly be applied to this query module.
            s_tir::meta_schedule::ScheduleUsingAnchorTrace(sch, record->trace, target);
          } else {
            record->trace->ApplyToSchedule(sch, /*remove_postproc=*/false);
          }
          IRModule new_mod = sch->mod();
          TVM_FFI_ICHECK_EQ(new_mod->functions.size(), 1);
          BaseFunc new_base_func = (*new_mod->functions.begin()).second;
          TVM_FFI_ICHECK(new_base_func->IsInstance<tirx::FunctionObj>());
          tirx::Function tuned_function = new_base_func.as_or_throw<tirx::Function>();
          // maintain the original attributes
          tirx::Function new_function = tirx::Function(/*params=*/tuned_function->params,
                                                       /*body=*/tuned_function->body,
                                                       /*ret_type=*/tuned_function->ret_type,
                                                       /*attrs=*/function->attrs);
          new_function =
              WithAttr(std::move(new_function), tvm::s_tir::attr::kIsScheduled.name, true);
          result.Set(gv, new_function);
          continue;
        } else if (enable_warning) {
          LOG(WARNING) << "Tuning record is not found for function: " << gv->name_hint;
        }
      }
      result.Set(gv, base_func);
    }
    return IRModule(result,       // functions
                    mod->attrs);  // attrs);
  };
  return CreateModulePass(pass_func, 0, "MetaScheduleApplyDatabase");
}

Pass MetaScheduleTuneIRMod(ffi::Map<ffi::String, ffi::Tensor> params, ffi::String work_dir,
                           int64_t max_trials_global,
                           ffi::Optional<int64_t> max_trials_per_task = std::nullopt,
                           ffi::Optional<ffi::Array<ffi::String>> op_names = std::nullopt) {
  Target target = Target::Current(false).value();
  auto pass_func = [=](IRModule m, PassContext ctx) {
    auto max_trials_task = max_trials_per_task.value_or(max_trials_global);
    return MetaScheduleTuner(target, work_dir, max_trials_global, max_trials_task, op_names, params)
        .TuneIRMod(m, ctx);
  };
  return CreateModulePass(/*pass function*/ pass_func, /*opt level*/ 0,
                          /*pass name*/ "MetaScheduleTuneIRModule");
}

Pass MetaScheduleTuneTIR(ffi::String work_dir, int64_t max_trials_global) {
  Target target = Target::Current(false).value();
  ffi::TypedFunction<tirx::Function(tirx::Function, IRModule, PassContext)> pass_func =
      [=](tirx::Function f, IRModule mod, PassContext ctx) {
        return MetaScheduleTuner(target, work_dir, max_trials_global, max_trials_global,
                                 std::nullopt)
            .TuneTIR(f, ctx);
      };
  return tirx::transform::CreateFunctionPass(/*pass function*/ pass_func, /*opt level*/ 0,
                                             /*pass name*/ "MetaScheduleTuneTIR");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.transform.MetaScheduleApplyDatabase", MetaScheduleApplyDatabase)
      .def("relax.transform.MetaScheduleTuneIRMod", MetaScheduleTuneIRMod)
      .def("relax.transform.MetaScheduleTuneTIR", MetaScheduleTuneTIR);
}
}  // namespace transform
}  // namespace relax
}  // namespace tvm
