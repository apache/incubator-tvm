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
#ifndef TVM_S_TIR_META_SCHEDULE_RUNNER_H_
#define TVM_S_TIR_META_SCHEDULE_RUNNER_H_

#include <tvm/base.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/s_tir/meta_schedule/arg_info.h>

namespace tvm {
namespace s_tir {
namespace meta_schedule {

/*! \brief Runner's input containing path of artifact, type of device and argument info. */
class RunnerInputObj : public ffi::Object {
 public:
  /*! \brief The path to the built artifact. */
  ffi::String artifact_path;
  /*! \brief The type of device. */
  ffi::String device_type;
  /*! \brief The argument information. */
  ffi::Array<ArgInfo> args_info;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<RunnerInputObj>()
        .def_ro("artifact_path", &RunnerInputObj::artifact_path)
        .def_ro("device_type", &RunnerInputObj::device_type)
        .def_ro("args_info", &RunnerInputObj::args_info);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.RunnerInput", RunnerInputObj, ffi::Object);
};

/*!
 * \brief Managed reference to RunnerInputObj
 * \sa RunnerInputObj
 */
class RunnerInput : public ffi::ObjectRef {
 public:
  /*!
   * \brief Constructor of RunnerInput
   * \param artifact_path The path to the built artifact.
   * \param device_type The type of device.
   * \param args_info The argument information.
   */
  TVM_DLL explicit RunnerInput(ffi::String artifact_path, ffi::String device_type,
                               ffi::Array<ArgInfo> args_info);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(RunnerInput, ffi::ObjectRef, RunnerInputObj);
};

/*! \brief Runner's output containing measurement result of MeasureCandidate or error msg if any. */
class RunnerResultObj : public ffi::Object {
 public:
  /*! \brief The run time in seconds.*/
  ffi::Optional<ffi::Array<FloatImm>> run_secs;
  /*! \brief The error message, if any. */
  ffi::Optional<ffi::String> error_msg;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<RunnerResultObj>()
        .def_ro("run_secs", &RunnerResultObj::run_secs)
        .def_ro("error_msg", &RunnerResultObj::error_msg);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.RunnerResult", RunnerResultObj,
                                    ffi::Object);
};

/*!
 * \brief Managed reference to RunnerResultObj
 * \sa RunnerResultObj
 */
class RunnerResult : public ffi::ObjectRef {
 public:
  /*!
   * \brief Constructor
   * \brief The run time in seconds.
   * \brief The error message, if any.
   */
  TVM_DLL explicit RunnerResult(ffi::Optional<ffi::Array<FloatImm>> run_secs,
                                ffi::Optional<ffi::String> error_msg);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(RunnerResult, ffi::ObjectRef, RunnerResultObj);
};

/*!
 * \brief A class to asynchronously fetch runner's output.
 * \note The API design is consistent with python's concurrent.futures.Future:
 *  https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future
 */
class RunnerFutureObj : public ffi::Object {
 public:
  /*!
   * \brief The function type to check whether the runner has finished.
   * \return Whether the runner's output is ready.
   */
  using FDone = ffi::TypedFunction<bool()>;
  /*!
   * \brief The function type to fetch runner output if it is ready.
   * \return The runner's output.
   */
  using FResult = ffi::TypedFunction<RunnerResult()>;

  /*! \brief The packed function to check whether the runner has finished. */
  FDone f_done;
  /*! \brief The packed function to fetch runner output if it is ready. */
  FResult f_result;

  static void RegisterReflection() {
    // `f_done` is not registered
    // `f_result` is not registered
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<RunnerFutureObj>();
  }

  /*!
   * \brief Check whether the runner has finished.
   * \return A boolean indicating whether the runner has finished.
   */
  bool Done() const {
    TVM_FFI_ICHECK(f_done != nullptr) << "PyRunnerFuture's Done method not implemented!";
    return f_done();
  }
  /*!
   * \brief Fetch the runner's output if it is ready.
   * \return The runner's output.
   */
  RunnerResult Result() const {
    TVM_FFI_ICHECK(f_result != nullptr) << "PyRunnerFuture's Result method not implemented!";
    return f_result();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.RunnerFuture", RunnerFutureObj,
                                    ffi::Object);
};

/*!
 * \brief Managed reference to RunnerFutureObj
 * \sa RunnerFutureObj
 */
class RunnerFuture : public ffi::ObjectRef {
 public:
  using FDone = RunnerFutureObj::FDone;
  using FResult = RunnerFutureObj::FResult;

  /*!
   * \brief Constructor of RunnerFuture
   * \param f_done The packed function to check whether the runner has finished.
   * \param f_result The packed function to fetch runner output if it is ready.
   */
  TVM_DLL explicit RunnerFuture(FDone f_done, FResult f_result);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(RunnerFuture, ffi::ObjectRef, RunnerFutureObj);
};

/*! \brief The abstract runner interface. */
class RunnerObj : public ffi::Object {
 public:
  /*!
   * \brief The function type to run the built artifacts and get runner futures.
   * \param input The runner's inputs.
   * \return The runner futures.
   * \sa RunnerFuture
   */
  using FRun = ffi::TypedFunction<ffi::Array<RunnerFuture>(ffi::Array<RunnerInput>)>;

  /*! \brief Default destructor */
  virtual ~RunnerObj() = default;

  /*!
   * \brief Run the built artifact and get runner futures.
   * \param runner_inputs The runner's inputs.
   * \return The runner futures.
   */
  virtual ffi::Array<RunnerFuture> Run(ffi::Array<RunnerInput> runner_inputs) = 0;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<RunnerObj>();
  }

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("s_tir.meta_schedule.Runner", RunnerObj, ffi::Object);
};

/*!
 * \brief Managed reference to RunnerObj
 * \sa RunnerObj
 */
class Runner : public ffi::ObjectRef {
 public:
  using FRun = RunnerObj::FRun;
  /*!
   * \brief Constructor from ffi::ObjectPtr<RunnerObj>.
   * \param data The object pointer.
   */
  explicit Runner(ffi::ObjectPtr<RunnerObj> data) : ffi::ObjectRef(data) {
    TVM_FFI_ICHECK(data != nullptr);
  }
  /*!
   * \brief Create a runner with customized build method on the python-side.
   * \param f_run The packed function to run the built artifacts and get runner futures.
   * \return The runner created.
   */
  TVM_DLL static Runner PyRunner(FRun f_run);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Runner, ffi::ObjectRef, RunnerObj);
};

/*! \brief An abstract runner with customized build method on the python-side. */
class PyRunnerObj : public RunnerObj {
 public:
  /*! \brief The packed function to run the built artifacts and get runner futures. */
  FRun f_run;

  static void RegisterReflection() {
    // `f_run` is not registered
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PyRunnerObj>();
  }

  ffi::Array<RunnerFuture> Run(ffi::Array<RunnerInput> runner_inputs) final {
    TVM_FFI_ICHECK(f_run != nullptr) << "PyRunner's Run method not implemented!";
    return f_run(runner_inputs);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.PyRunner", PyRunnerObj, RunnerObj);
};

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm

#endif  // TVM_S_TIR_META_SCHEDULE_RUNNER_H_
