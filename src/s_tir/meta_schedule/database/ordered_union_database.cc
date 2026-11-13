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

class OrderedUnionDatabaseObj : public DatabaseObj {
 public:
  ffi::Array<Database> databases;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<OrderedUnionDatabaseObj>().def_ro("databases",
                                                      &OrderedUnionDatabaseObj::databases);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.OrderedUnionDatabase",
                                    OrderedUnionDatabaseObj, DatabaseObj);

 public:
  ffi::Optional<TuningRecord> QueryTuningRecord(const IRModule& mod, const Target& target,
                                                const ffi::String& task_name) final {
    for (const Database& db : databases) {
      if (ffi::Optional<TuningRecord> record = db->QueryTuningRecord(mod, target, task_name)) {
        return record;
      }
    }
    return std::nullopt;
  }

  bool HasWorkload(const IRModule& mod) final {
    TVM_FFI_THROW(NotImplementedError) << "OrderedUnionDatabase.HasWorkload";
    throw;
  }

  Workload CommitWorkload(const IRModule& mod) final {
    TVM_FFI_THROW(NotImplementedError) << "OrderedUnionDatabase.CommitWorkload";
    throw;
  }

  void CommitTuningRecord(const TuningRecord& record) final {
    TVM_FFI_THROW(NotImplementedError) << "OrderedUnionDatabase.CommitTuningRecord";
    throw;
  }

  ffi::Array<TuningRecord> GetTopK(const Workload& workload, int top_k) final {
    TVM_FFI_THROW(NotImplementedError) << "OrderedUnionDatabase.GetTopK";
    throw;
  }

  ffi::Array<TuningRecord> GetAllTuningRecords() final {
    TVM_FFI_THROW(NotImplementedError) << "OrderedUnionDatabase.GetAllTuningRecords";
    throw;
  }

  int64_t Size() final {
    TVM_FFI_THROW(NotImplementedError) << "OrderedUnionDatabase.size";
    throw;
  }
};

Database Database::OrderedUnionDatabase(ffi::Array<Database> databases) {
  ffi::ObjectPtr<OrderedUnionDatabaseObj> n = ffi::make_object<OrderedUnionDatabaseObj>();
  n->databases = std::move(databases);
  return Database(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.meta_schedule.DatabaseOrderedUnionDatabase",
                        Database::OrderedUnionDatabase);
}

TVM_FFI_STATIC_INIT_BLOCK() { OrderedUnionDatabaseObj::RegisterReflection(); }

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
