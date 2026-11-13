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

#include "../module_equality.h"
#include "../utils.h"

namespace tvm {
namespace s_tir {
namespace meta_schedule {

class MemoryDatabaseObj : public DatabaseObj {
 public:
  explicit MemoryDatabaseObj(ffi::String mod_eq_name = "structural") : DatabaseObj(mod_eq_name) {}

  ffi::Array<TuningRecord> records;
  ffi::Array<Workload> workloads;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<MemoryDatabaseObj>()
        .def_ro("records", &MemoryDatabaseObj::records)
        .def_ro("workloads", &MemoryDatabaseObj::workloads);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.MemoryDatabase", MemoryDatabaseObj,
                                    DatabaseObj);

 public:
  bool HasWorkload(const IRModule& mod) final {
    for (const auto& workload : workloads) {
      if (GetModuleEquality().Equal(workload->mod, mod)) {
        return true;
      }
    }
    return false;
  }

  Workload CommitWorkload(const IRModule& mod) final {
    for (const auto& workload : workloads) {
      if (GetModuleEquality().Equal(workload->mod, mod)) {
        return workload;
      }
    }
    Workload workload(mod, GetModuleEquality().Hash(mod));
    workloads.push_back(workload);
    return workload;
  }

  void CommitTuningRecord(const TuningRecord& record) final { records.push_back(record); }

  ffi::Array<TuningRecord> GetTopK(const Workload& workload, int top_k) final {
    TVM_FFI_CHECK_GE(top_k, 0, ValueError) << "top_k must be non-negative";
    if (top_k == 0) {
      return {};
    }
    std::vector<TuningRecord> results;
    results.reserve(records.size());
    for (const TuningRecord& record : records) {
      if (!record->IsValid()) {
        continue;
      }
      if (record->workload.same_as(workload) ||
          WorkloadEqual(GetModuleEquality())(record->workload, workload)) {
        results.emplace_back(record);
      }
    }
    std::stable_sort(results.begin(), results.end(), SortTuningRecordByMeanRunSecs());
    if (results.size() > static_cast<size_t>(top_k)) {
      return {results.begin(), results.begin() + top_k};
    } else {
      return results;
    }
  }

  ffi::Array<TuningRecord> GetAllTuningRecords() final { return records; }

  int64_t Size() final { return records.size(); }
};

Database Database::MemoryDatabase(ffi::String mod_eq_name) {
  ffi::ObjectPtr<MemoryDatabaseObj> n = ffi::make_object<MemoryDatabaseObj>(mod_eq_name);
  n->records.clear();
  n->workloads.clear();
  return Database(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.meta_schedule.DatabaseMemoryDatabase", Database::MemoryDatabase);
}

TVM_FFI_STATIC_INIT_BLOCK() { MemoryDatabaseObj::RegisterReflection(); }

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
