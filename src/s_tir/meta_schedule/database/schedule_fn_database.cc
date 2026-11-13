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

class ScheduleFnDatabaseObj : public DatabaseObj {
 public:
  explicit ScheduleFnDatabaseObj(ffi::String mod_eq_name = "structural")
      : DatabaseObj(mod_eq_name) {}

  ffi::TypedFunction<bool(s_tir::Schedule)> schedule_fn;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ScheduleFnDatabaseObj>().def_ro("schedule_fn",
                                                    &ScheduleFnDatabaseObj::schedule_fn);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.ScheduleFnDatabase", ScheduleFnDatabaseObj,
                                    DatabaseObj);

 public:
  ffi::Optional<TuningRecord> QueryTuningRecord(const IRModule& mod, const Target& target,
                                                const ffi::String& workload_name) final {
    if (ffi::Optional<s_tir::Schedule> sch = this->QuerySchedule(mod, target, workload_name)) {
      return TuningRecord(sch.value()->trace().value(),
                          /*workload=*/Workload(mod, 0),  //
                          /*run_secs=*/std::nullopt,      //
                          /*target=*/target,              //
                          /*arg_info=*/std::nullopt);
    }
    return std::nullopt;
  }

  ffi::Optional<s_tir::Schedule> QuerySchedule(const IRModule& mod, const Target& target,
                                               const ffi::String& workload_name) final {
    s_tir::Schedule sch = s_tir::Schedule::Traced(
        WithAttr<IRModule>(mod, tvm::s_tir::meta_schedule::attr::kTaskName.name, workload_name),
        /*rand_state=*/-1,
        /*debug_mode=*/0,
        /*error_render_level=*/s_tir::ScheduleErrorRenderLevel::kDetail);
    if (!schedule_fn(sch)) {
      return std::nullopt;
    }
    return sch;
  }

  bool HasWorkload(const IRModule& mod) final {
    TVM_FFI_THROW(NotImplementedError) << "ScheduleFnDatabase.HasWorkload";
    throw;
  }

  Workload CommitWorkload(const IRModule& mod) final {
    TVM_FFI_THROW(NotImplementedError) << "ScheduleFnDatabase.CommitWorkload";
    throw;
  }

  void CommitTuningRecord(const TuningRecord& record) final {
    TVM_FFI_THROW(NotImplementedError) << "ScheduleFnDatabase.CommitTuningRecord";
    throw;
  }

  ffi::Array<TuningRecord> GetTopK(const Workload& workload, int top_k) final {
    TVM_FFI_THROW(NotImplementedError) << "ScheduleFnDatabase.GetTopK";
    throw;
  }

  ffi::Array<TuningRecord> GetAllTuningRecords() final {
    TVM_FFI_THROW(NotImplementedError) << "ScheduleFnDatabase.GetAllTuningRecords";
    throw;
  }

  int64_t Size() final {
    TVM_FFI_THROW(NotImplementedError) << "ScheduleFnDatabase.size";
    throw;
  }
};

Database Database::ScheduleFnDatabase(ffi::TypedFunction<bool(s_tir::Schedule)> schedule_fn,
                                      ffi::String mod_eq_name) {
  ffi::ObjectPtr<ScheduleFnDatabaseObj> n = ffi::make_object<ScheduleFnDatabaseObj>(mod_eq_name);
  n->schedule_fn = std::move(schedule_fn);
  return Database(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.meta_schedule.DatabaseScheduleFnDatabase",
                        Database::ScheduleFnDatabase);
}

TVM_FFI_STATIC_INIT_BLOCK() { ScheduleFnDatabaseObj::RegisterReflection(); }

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
