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
#ifndef TVM_S_TIR_META_SCHEDULE_DATABASE_H_
#define TVM_S_TIR_META_SCHEDULE_DATABASE_H_

#include <tvm/base.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/module.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/s_tir/meta_schedule/arg_info.h>
#include <tvm/s_tir/schedule/schedule.h>
#include <tvm/s_tir/schedule/trace.h>
#include <tvm/target/target.h>

#include <filesystem>
#include <memory>

namespace tvm {
namespace s_tir {
namespace meta_schedule {

class ModuleEquality;

/*! \brief A workload, i.e. an IRModule and its structural hash. */
class WorkloadObj : public ffi::Object {
 public:
  /*! \brief The type of structural hash */
  using THashCode = size_t;
  /*! \brief The workload's IRModule. */
  IRModule mod;
  /*! \brief The workload's structural hash. */
  THashCode shash;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<WorkloadObj>().def_ro("mod", &WorkloadObj::mod);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.Workload", WorkloadObj, ffi::Object);

  /*!
   * \brief Export the workload to a JSON string.
   * \return An array containing the structural hash and the base64 json string.
   */
  ffi::ObjectRef AsJSON() const;
};

/*!
 * \brief Managed reference to WorkloadObj.
 *  \sa WorkloadObj
 */
class Workload : public ffi::ObjectRef {
 public:
  using THashCode = WorkloadObj::THashCode;
  explicit Workload(ffi::UnsafeInit tag, ffi::ObjectPtr<WorkloadObj> data) : ffi::ObjectRef(tag) {
    data_ = std::move(data);
  }
  /*!
   * \brief Constructor of Workload.
   * \param mod The workload's IRModule.
   */
  TVM_DLL explicit Workload(IRModule mod);
  /*!
   * \brief Constructor of Workload.
   * \param mod The workload's IRModule.
   * \param shash The workload's structural hash.
   */
  TVM_DLL explicit Workload(IRModule mod, THashCode shash);
  /*!
   * \brief Create a workload from a json object.
   * \param json_obj The json object.
   * \return The created workload.
   */
  TVM_DLL static Workload FromJSON(const ffi::ObjectRef& json_obj);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Workload, ffi::ObjectRef, WorkloadObj);
};

/*! \brief The hash method for Workload */
struct WorkloadHash {
  size_t operator()(const Workload& a) const { return a->shash; }
};

/*! \brief The equality check for Workload */
struct WorkloadEqual {
  explicit WorkloadEqual(const ModuleEquality& mod_eq) : mod_eq_(mod_eq) {}

  bool operator()(const Workload& a, const Workload& b) const;

 private:
  /*! \brief The module equality testing and hashing method */
  const ModuleEquality& mod_eq_;
};

/*! \brief The class of measure candidates. */
class MeasureCandidate;

/*! \brief The class of tuning records. */
class TuningRecordObj : public ffi::Object {
 public:
  /*! \brief The trace tuned. */
  s_tir::Trace trace;
  /*! \brief The workload. */
  Workload workload{ffi::UnsafeInit()};
  /*! \brief The profiling result in seconds. */
  ffi::Optional<ffi::Array<FloatImm>> run_secs;
  /*! \brief The target for tuning. */
  ffi::Optional<Target> target;
  /*! \brief The argument information. */
  ffi::Optional<ffi::Array<ArgInfo>> args_info;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TuningRecordObj>()
        .def_ro("trace", &TuningRecordObj::trace)
        .def_ro("workload", &TuningRecordObj::workload)
        .def_ro("run_secs", &TuningRecordObj::run_secs)
        .def_ro("target", &TuningRecordObj::target)
        .def_ro("args_info", &TuningRecordObj::args_info);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.TuningRecord", TuningRecordObj,
                                    ffi::Object);

  /*! \brief Construct the measure candidate given the initial IR module and trace
   * stored in the tuning record. */
  MeasureCandidate AsMeasureCandidate() const;
  /*!
   * \brief Export the tuning record to a JSON string.
   * \return An array containing the trace, running secs, serialized target, and
   * argument information.
   */
  ffi::ObjectRef AsJSON() const;
  /*!
   * \brief Check if this tuning record has valid trace instructions and successful run results.
   * \return The check result.
   */
  bool IsValid() const;
};

/*!
 * \brief The managed reference of TuningRecordObj.
 * \sa TuningRecordObj
 */
class TuningRecord : public ffi::ObjectRef {
 public:
  /*!
   \brief Constructor of a tuning record.
   \param trace The trace of the tuning record.
   \param workload The workload of the tuning record.
   \param run_secs The running time of the tuning record.
   \param target The target of the tuning record.
   \param args_info The argument information of the tuning record.
  */
  TVM_DLL explicit TuningRecord(s_tir::Trace trace, Workload workload,
                                ffi::Optional<ffi::Array<FloatImm>> run_secs,
                                ffi::Optional<Target> target,
                                ffi::Optional<ffi::Array<ArgInfo>> args_info);
  /*!
   * \brief Create a tuning record from a json object.
   * \param json_obj The json object.
   * \param workload The workload.
   * \return The tuning record created.
   */
  TVM_DLL static TuningRecord FromJSON(const ffi::ObjectRef& json_obj, const Workload& workload);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TuningRecord, ffi::ObjectRef, TuningRecordObj);
};

class Database;

/* \brief The abstract interface of database. */
class DatabaseObj : public ffi::Object {
 public:
  /*!
   * \brief Constructor
   * \param mod_eq_name A string to specify the module equality testing and hashing method.
   *  It must be one of the followings:
   *    - "structural": Use StructuralEqual/Hash
   *    - "ignore-tensor": Same as "structural", but ignore tensor raw data during
   *                        equality testing and hashing.
   *    - "anchor-block": Apply equality testing and hashing on the anchor block extracted from a
   *                      given module. The "ignore-tensor" varint is used for the extracted blocks
   *                      or in case no anchor block is found.
   *                      For the definition of the anchor block, see tvm/tirx/analysis.h.
   */
  explicit DatabaseObj(ffi::String mod_eq_name = "structural");

  /*! \brief Default destructor */
  virtual ~DatabaseObj();
  /*!
   * \brief Check if the database has the given workload.
   * \param mod The IRModule to be searched for.
   * \return Whether the database has the given workload.
   */
  virtual bool HasWorkload(const IRModule& mod) = 0;
  /*!
   * \brief Look up or add workload to the database if missing.
   * \param mod The IRModule to be searched for or added.
   * \return The workload corresponding to the given IRModule.
   */
  virtual Workload CommitWorkload(const IRModule& mod) = 0;
  /*!
   * \brief Add a tuning record to the database.
   * \param record The tuning record to be added.
   */
  virtual void CommitTuningRecord(const TuningRecord& record) = 0;
  /*!
   * \brief Get the top K valid tuning records of given workload from the database.
   * \param workload The workload to be searched for.
   * \param top_k The number of top records to be returned.
   * \return An array of top K tuning records for the given workload.
   */
  virtual ffi::Array<TuningRecord> GetTopK(const Workload& workload, int top_k) = 0;
  /*!
   * \brief Get all tuning records from the database.
   * \return An Array of all the tuning records in the database.
   */
  virtual ffi::Array<TuningRecord> GetAllTuningRecords() = 0;
  /*!
   * \brief Get the size of the database.
   * \return The size of the database.
   */
  virtual int64_t Size() = 0;
  /*!
   * \brief Query the best record of the given workload from the database.
   * \param mod The IRModule to be searched for.
   * \param target The target to be searched for.
   * \param workload_name The name of the workload to be searched for.
   * \return The best record of the given workload; std::nullopt if not found.
   */
  virtual ffi::Optional<TuningRecord> QueryTuningRecord(const IRModule& mod, const Target& target,
                                                        const ffi::String& workload_name);
  /*!
   * \brief Query the best schedule of the given workload from the database.
   * \param mod The IRModule to be searched for.
   * \param target The target to be searched for.
   * \param workload_name The name of the workload to be searched for.
   * \return The schedule in the best schedule of the given workload; std::nullopt if not found.
   */
  virtual ffi::Optional<s_tir::Schedule> QuerySchedule(const IRModule& mod, const Target& target,
                                                       const ffi::String& workload_name);
  /*!
   * \brief Query the best IRModule of the given workload from the database.
   * \param mod The IRModule to be searched for.
   * \param target The target to be searched for.
   * \param workload_name The name of the workload to be searched for.
   * \return The IRModule in the best IRModule of the given workload; std::nullopt if not found.
   */
  virtual ffi::Optional<IRModule> QueryIRModule(const IRModule& mod, const Target& target,
                                                const ffi::String& workload_name);
  /*!
   * \brief Prune the database and dump it a given database.
   * \param destination The destination database to be dumped to.
   */
  void DumpPruned(Database destination);
  /*! \brief Return a reference to the owned module equality method instance. */
  const ModuleEquality& GetModuleEquality() const {
    TVM_FFI_ICHECK(mod_eq_);
    return *mod_eq_;
  }

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("s_tir.meta_schedule.Database", DatabaseObj, ffi::Object);

 private:
  /*! \brief The module equality testing and hashing method */
  std::unique_ptr<ModuleEquality> mod_eq_;
};

/*! \brief The database with customized methods on the python-side. */
class PyDatabaseObj : public DatabaseObj {
 public:
  /*!
   * \brief Constructor
   * \param mod_eq_name A string to specify the module equality testing and hashing method.
   *  It must be one of the followings:
   *    - "structural": Use StructuralEqual/Hash
   *    - "ignore-tensor": Same as "structural", but ignore tensor raw data during
   *                        equality testing and hashing.
   *    - "anchor-block": Apply equality testing and hashing on the anchor block extracted from a
   *                      given module. The "ignore-tensor" varint is used for the extracted blocks
   *                      or in case no anchor block is found.
   *                      For the definition of the anchor block, see tvm/tirx/analysis.h.
   */
  explicit PyDatabaseObj(ffi::String mod_eq_name = "structural");

  /*!
   * \brief The function type of `HasWorkload` method.
   * \param mod The IRModule to be searched for.
   * \return Whether the database has the given workload.
   */
  using FHasWorkload = ffi::TypedFunction<bool(const IRModule&)>;
  /*!
   * \brief The function type of `CommitWorkload` method.
   * \param mod The IRModule to be searched for or added.
   * \return The workload corresponding to the given IRModule.
   */
  using FCommitWorkload = ffi::TypedFunction<Workload(const IRModule&)>;
  /*!
   * \brief The function type of `CommitTuningRecord` method.
   * \param record The tuning record to be added.
   */
  using FCommitTuningRecord = ffi::TypedFunction<void(const TuningRecord&)>;
  /*!
   * \brief The function type of `GetTopK` method.
   * \param workload The workload to be searched for.
   * \param top_k The number of top records to be returned.
   * \return An array of top K tuning records for the given workload.
   */
  using FGetTopK = ffi::TypedFunction<ffi::Array<TuningRecord>(const Workload&, int)>;
  /*!
   * \brief The function type of `GetAllTuningRecords` method.
   * \return An Array of all the tuning records in the database.
   */
  using FGetAllTuningRecords = ffi::TypedFunction<ffi::Array<TuningRecord>()>;
  /*!
   * \brief The function type of `QueryTuningRecord` method.
   * \param mod The IRModule to be searched for.
   * \param target The target to be searched for.
   * \param workload_name The name of the workload to be searched for.
   * \return The best record of the given workload; std::nullopt if not found.
   */
  using FQueryTuningRecord = ffi::TypedFunction<ffi::Optional<TuningRecord>(
      const IRModule&, const Target&, const ffi::String&)>;
  /*!
   * \brief The function type of `QuerySchedule` method.
   * \param mod The IRModule to be searched for.
   * \param target The target to be searched for.
   * \param workload_name The name of the workload to be searched for.
   * \return The schedule in the best schedule of the given workload; std::nullopt if not found.
   */
  using FQuerySchedule = ffi::TypedFunction<ffi::Optional<s_tir::Schedule>(
      const IRModule&, const Target&, const ffi::String&)>;
  /*!
   * \brief The function type of `QueryIRModule` method.
   * \param mod The IRModule to be searched for.
   * \param target The target to be searched for.
   * \param workload_name The name of the workload to be searched for.
   * \return The IRModule in the best IRModule of the given workload; std::nullopt if not found.
   */
  using FQueryIRModule = ffi::TypedFunction<ffi::Optional<IRModule>(const IRModule&, const Target&,
                                                                    const ffi::String&)>;
  /*!
   * \brief The function type of `Size` method.
   * \return The size of the database.
   */
  using FSize = ffi::TypedFunction<int64_t()>;

  /*! \brief The packed function to the `HasWorkload` function. */
  FHasWorkload f_has_workload;
  /*! \brief The packed function to the `CommitWorkload` function. */
  FCommitWorkload f_commit_workload;
  /*! \brief The packed function to the `CommitTuningRecord` function. */
  FCommitTuningRecord f_commit_tuning_record;
  /*! \brief The packed function to the `GetTopK` function. */
  FGetTopK f_get_top_k;
  /*! \brief The packed function to the `GetAllTuningRecords` function. */
  FGetAllTuningRecords f_get_all_tuning_records;
  /*! \brief The packed function to the `QueryTuningRecord` function. */
  FQueryTuningRecord f_query_tuning_record;
  /*! \brief The packed function to the `QuerySchedule` function. */
  FQuerySchedule f_query_schedule;
  /*! \brief The packed function to the `QueryIRModule` function. */
  FQueryIRModule f_query_ir_module;
  /*! \brief The packed function to the `Size` function. */
  FSize f_size;

  static void RegisterReflection() {
    // ffi::Functions are all not registered, because the reflection system doesn't take care of
    // them, so it cannot be accessible on the python side. If there is such need from the future,
    // we can then add corresponding accessor methods to help access on python.
    // `f_has_workload` is not registered
    // `f_commit_workload` is not registered
    // `f_commit_tuning_record` is not registered
    // `f_get_top_k` is not registered
    // `f_get_all_tuning_records` is not registered
    // `f_query_tuning_record` is not registered
    // `f_query_schedule` is not registered
    // `f_query_ir_module` is not registered
    // `f_size` is not registered
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PyDatabaseObj>();
  }

  bool HasWorkload(const IRModule& mod) final {
    TVM_FFI_ICHECK(f_has_workload != nullptr) << "PyDatabase's HasWorkload method not implemented!";
    return f_has_workload(mod);
  }

  Workload CommitWorkload(const IRModule& mod) final {
    TVM_FFI_ICHECK(f_commit_workload != nullptr)
        << "PyDatabase's CommitWorkload method not implemented!";
    return f_commit_workload(mod);
  }

  void CommitTuningRecord(const TuningRecord& record) final {
    TVM_FFI_ICHECK(f_commit_tuning_record != nullptr)
        << "PyDatabase's CommitTuningRecord method not implemented!";
    f_commit_tuning_record(record);
  }

  ffi::Array<TuningRecord> GetTopK(const Workload& workload, int top_k) final {
    TVM_FFI_ICHECK(f_get_top_k != nullptr) << "PyDatabase's GetTopK method not implemented!";
    return f_get_top_k(workload, top_k);
  }

  ffi::Array<TuningRecord> GetAllTuningRecords() final {
    TVM_FFI_ICHECK(f_get_all_tuning_records != nullptr)
        << "PyDatabase's GetAllTuningRecords method not implemented!";
    return f_get_all_tuning_records();
  }

  ffi::Optional<TuningRecord> QueryTuningRecord(const IRModule& mod, const Target& target,
                                                const ffi::String& workload_name) final {
    if (f_query_tuning_record == nullptr) {
      return DatabaseObj::QueryTuningRecord(mod, target, workload_name);
    } else {
      return f_query_tuning_record(mod, target, workload_name);
    }
  }

  ffi::Optional<s_tir::Schedule> QuerySchedule(const IRModule& mod, const Target& target,
                                               const ffi::String& workload_name) final {
    if (f_query_schedule == nullptr) {
      return DatabaseObj::QuerySchedule(mod, target, workload_name);
    } else {
      return f_query_schedule(mod, target, workload_name);
    }
  }

  ffi::Optional<IRModule> QueryIRModule(const IRModule& mod, const Target& target,
                                        const ffi::String& workload_name) final {
    if (f_query_ir_module == nullptr) {
      return DatabaseObj::QueryIRModule(mod, target, workload_name);
    } else {
      return f_query_ir_module(mod, target, workload_name);
    }
  }

  int64_t Size() final {
    TVM_FFI_ICHECK(f_size != nullptr) << "PyDatabase's Size method not implemented!";
    return f_size();
  }

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.PyDatabase", PyDatabaseObj, DatabaseObj);
};

/*!
 * \brief Managed reference to DatabaseObj.
 * \sa DatabaseObj
 */
class Database : public ffi::ObjectRef {
 public:
  /*!
   * \brief Constructor from ffi::ObjectPtr<DatabaseObj>.
   * \param data The object pointer.
   */
  explicit Database(ffi::ObjectPtr<DatabaseObj> data) : ffi::ObjectRef(data) {
    TVM_FFI_ICHECK(data != nullptr);
  }
  /*!
   * \brief An in-memory database.
   * \param mod_eq_name A string to specify the module equality testing and hashing method.
   */
  TVM_DLL static Database MemoryDatabase(ffi::String mod_eq_name = "structural");
  /*!
   * \brief A database for injecting handcrafted schedule functions.
   * \param schedule_fn The function to do scheduling, which takes a TIR schedule,
   * and returns a boolean indicating if the schedule is successful.
   * \param mod_eq_name A string to specify the module equality testing and hashing method.
   */
  TVM_DLL static Database ScheduleFnDatabase(ffi::TypedFunction<bool(s_tir::Schedule)> schedule_fn,
                                             ffi::String mod_eq_name = "structural");
  /*!
   * \brief Create a default database that uses JSON file for tuning records.
   * \param path_workload The path to the workload table.
   * \param path_tuning_record The path to the database table.
   * \param allow_missing Whether to create new file when the given path is not found.
   * \param mod_eq_name A string to specify the module equality testing and hashing method.
   */
  TVM_DLL static Database JSONDatabase(ffi::String path_workload, ffi::String path_tuning_record,
                                       bool allow_missing, ffi::String mod_eq_name = "structural");
  /*!
   * \brief A database composed of multiple databases, allowing users to guide IR rewriting using
   * combined knowledge of those databases. To each query, it returns the best record among all the
   * databases given.
   * \param databases The list of databases to be combined.
   * \return The combined database.
   */
  TVM_DLL static Database UnionDatabase(ffi::Array<Database, void> databases);
  /*!
   * \brief A database composed of multiple databases, allowing users to guide IR rewriting using
   * combined knowledge of those databases. To each query, it returns the record from the first
   * database that responds to the query.
   * \param databases The database to be subsetted.
   * \return The subsetted database.
   */
  TVM_DLL static Database OrderedUnionDatabase(ffi::Array<Database, void> databases);
  /*!
   * \brief Create a database with customized methods on the python-side.
   * \param f_has_workload The packed function of `HasWorkload`.
   * \param f_commit_workload The packed function of `CommitWorkload`.
   * \param f_commit_tuning_record The packed function of `CommitTuningRecord`.
   * \param f_get_top_k The packed function of `GetTopK`.
   * \param f_get_all_tuning_records The packed function of `GetAllTuningRecords`.
   * \param f_query_tuning_record The packed function of `QueryTuningRecord`.
   * \param f_query_schedule The packed function of `QuerySchedule`.
   * \param f_query_ir_module The packed function of `QueryIRModule`.
   * \param f_size The packed function of `Size`.
   * \param mod_eq_name A string to specify the module equality testing and hashing method.
   * \return The created database.
   */
  TVM_DLL static Database PyDatabase(PyDatabaseObj::FHasWorkload f_has_workload,
                                     PyDatabaseObj::FCommitWorkload f_commit_workload,
                                     PyDatabaseObj::FCommitTuningRecord f_commit_tuning_record,
                                     PyDatabaseObj::FGetTopK f_get_top_k,
                                     PyDatabaseObj::FGetAllTuningRecords f_get_all_tuning_records,
                                     PyDatabaseObj::FQueryTuningRecord f_query_tuning_record,
                                     PyDatabaseObj::FQuerySchedule f_query_schedule,
                                     PyDatabaseObj::FQueryIRModule f_query_ir_module,
                                     PyDatabaseObj::FSize f_size,
                                     ffi::String mod_eq_name = "structural");
  /*! \return The current Database in the scope. */
  static ffi::Optional<Database> Current();
  /*! \brief Entering the scope of the context manager */
  void EnterWithScope();
  /*! \brief Exiting the scope of the context manager */
  void ExitWithScope();

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Database, ffi::ObjectRef, DatabaseObj);
};

namespace attr {
inline constexpr ffi::reflection::AttrKey<ffi::String> kTaskName{"task_name"};
}  // namespace attr

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm

#endif  // TVM_S_TIR_META_SCHEDULE_DATABASE_H_
