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
#ifndef TVM_S_TIR_META_SCHEDULE_PROFILER_H_
#define TVM_S_TIR_META_SCHEDULE_PROFILER_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/module.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/target/target.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace s_tir {
namespace meta_schedule {

class ScopedTimer {
 public:
  ~ScopedTimer() {
    if (deferred_ != nullptr) {
      deferred_();
    }
  }

 private:
  friend class Profiler;

  explicit ScopedTimer(ffi::TypedFunction<void()> deferred) : deferred_(deferred) {}
  ffi::TypedFunction<void()> deferred_;
};

/*! \brief A generic profiler */
class ProfilerObj : public ffi::Object {
 public:
  /*! \brief The segments that are already profiled */
  std::unordered_map<std::string, double> stats_sec;
  /*! \brief Counter for the total time used */
  ffi::Function total_timer;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ProfilerObj>();
  }

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.Profiler", ProfilerObj, ffi::Object);

 public:
  /*! \brief Get the internal stats of the running time */
  ffi::Map<ffi::String, FloatImm> Get() const;
  /*! \brief Return a summary of profiling results as table format */
  ffi::String Table() const;
};

/*!
 * \brief Managed reference to ProfilerObj
 * \sa ProfilerObj
 */
class Profiler : public ffi::ObjectRef {
 public:
  Profiler();
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Profiler, ffi::ObjectRef, ProfilerObj);

  /*! \brief Entering the scope of the context manager */
  void EnterWithScope();
  /*! \brief Exiting the scope of the context manager */
  void ExitWithScope();
  /*! \brief Returns the current profiler */
  static ffi::Optional<Profiler> Current();
  /*!
   * \brief Profile the time usage in the given scope in the given name.
   * \param name Name for the scope.
   * \return A scope timer for time profiling.
   */
  static ScopedTimer TimedScope(ffi::String name);
};

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm

#endif  // TVM_S_TIR_META_SCHEDULE_PROFILER_H_
