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
#ifndef TVM_S_TIR_META_SCHEDULE_TASK_SCHEDULER_H_
#define TVM_S_TIR_META_SCHEDULE_TASK_SCHEDULER_H_

#include <tvm/base.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/runtime/base.h>
#include <tvm/s_tir/meta_schedule/builder.h>
#include <tvm/s_tir/meta_schedule/cost_model.h>
#include <tvm/s_tir/meta_schedule/measure_callback.h>
#include <tvm/s_tir/meta_schedule/runner.h>
#include <tvm/s_tir/meta_schedule/tune_context.h>
#include <tvm/s_tir/random_engine.h>

#include <string>
#include <vector>

namespace tvm {
namespace s_tir {
namespace meta_schedule {

class TaskRecordObj : public ffi::Object {
 public:
  /*! \brief The tune context of the task. */
  TuneContext ctx{ffi::UnsafeInit()};
  /*! \brief The weight of the task */
  double task_weight{1.0};
  /*! \brief The FLOP count of the task */
  double flop{1.0};
  /*! \brief Whether the tuning task has been stopped or finished. */
  bool is_terminated = false;
  /*! \brief Builder errors happens in the task */
  int build_error_count = 0;
  /*! \brief Runner errors happens in the task */
  int run_error_count = 0;
  /*! \brief The latency of each run, in milliseconds. */
  std::vector<double> latency_ms = {};
  /*! \brief The measure candidates. */
  ffi::Optional<ffi::Array<MeasureCandidate>> measure_candidates = std::nullopt;
  /*! \brief The building results. */
  ffi::Optional<ffi::Array<BuilderResult>> builder_results = std::nullopt;
  /*! \brief Packed functions to fetch the runner results asynchronously. */
  ffi::Optional<ffi::Array<RunnerFuture>> runner_futures = std::nullopt;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TaskRecordObj>()
        .def_ro("ctx", &TaskRecordObj::ctx)
        .def_ro("task_weight", &TaskRecordObj::task_weight)
        .def_ro("flop", &TaskRecordObj::flop)
        .def_ro("is_terminated", &TaskRecordObj::is_terminated)
        .def_ro("build_error_count", &TaskRecordObj::build_error_count)
        .def_ro("run_error_count", &TaskRecordObj::run_error_count)
        .def_ro("measure_candidates", &TaskRecordObj::measure_candidates)
        .def_ro("builder_results", &TaskRecordObj::builder_results)
        .def_ro("runner_futures", &TaskRecordObj::runner_futures);
  }

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.TaskRecord", TaskRecordObj, ffi::Object);
};

/*!
 * \brief Managed reference to TaskRecordObj.
 * \sa TaskRecordObj
 */
class TaskRecord : public ffi::ObjectRef {
 public:
  /*! \brief Constructor */
  explicit TaskRecord(TuneContext task, double task_weight);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TaskRecord, ffi::ObjectRef, TaskRecordObj);
};

/*!
 * \brief The abstract interface of task schedulers.
 * \note The relationship between SpaceGenerator and other classes are as follows:
        +--------------------------------------------------------------+
    +--+-----------------------------------------------------------+    |
  +--+------------------ Tune Context -----------------------------+ |  |
  |                +---------------------+                        |  |  |
  |                |                     |   Generate             |  |  |
  |                |   Space Generator   +--------------+         |  |  |
  |                |                     |              |         |  |  |
  |                +---------------------+              v         |  |  |
  |                                               Design Space    |  |  |
  |                +---------------------+              |         |  |  |
  |      Generate  |                     |   Pretuning  |         |  |  |
  |    +-----------+   Search Strategy   |<-------------+         |  |  |
  |    |           |                     |                        |  +--+
  |    |           +---------------------+                        +--+
  +----+----------------------------------------------------------+
      |
      |
  +----+---------------- Managed By Task Scheduler ---------------------+
  |    |                                 +-----------+                  |
  |    |                      Send to    |           |  Send to         |
  |    v                  +-------------+|  Builder  +----------+       |
  | Measure Candidate     |   Builder    |           |  Runner  |       |
  |    |                  |              +-----------+          |       |
  |    |     +------------+------------+                        |       |
  |    |     |                         |     +-----------+      |       |
  |    +---->|   Task Scheduler        |     |           |      |       |
  |          |                         |     |  Runner   |<-----+       |
  |          +-------------------------+     |           |              |
  |                   ^                      +-----+-----+              |
  |                   |                            |                    |
  |                   +----  Runner Future <-------+                    |
  +---------------------------------------------------------------------+
*/
class TaskSchedulerObj : public ffi::Object {
 public:
  /*! \brief The tuning task's logging function. */
  ffi::Function logger;
  /*! \brief Records for each task */
  ffi::Array<TaskRecord> tasks_;
  /*! \brief The list of measure callbacks of the scheduler. */
  ffi::Array<MeasureCallback> measure_callbacks_;
  /*! \brief The database used in tuning */
  ffi::Optional<Database> database_;
  /*! \brief The cost model used in tuning */
  ffi::Optional<CostModel> cost_model_;
  /*! \brief The number of remaining tasks to be tuned. */
  int remaining_tasks_;

  /*! \brief The default destructor. */
  virtual ~TaskSchedulerObj() = default;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TaskSchedulerObj>()
        .def_ro("tasks_", &TaskSchedulerObj::tasks_)
        .def_ro("measure_callbacks_", &TaskSchedulerObj::measure_callbacks_)
        .def_ro("database_", &TaskSchedulerObj::database_)
        .def_ro("cost_model_", &TaskSchedulerObj::cost_model_)
        .def_ro("remaining_tasks_", &TaskSchedulerObj::remaining_tasks_);
  }

  /*!
   * \brief Fetch the next task id.
   * \return The next task id.
   */
  virtual int NextTaskId() = 0;
  /*!
   * \brief Wait until the task is finished.
   * \param task_id The task id to be joined.
   * \return The results from the runner.
   */
  virtual ffi::Array<RunnerResult> JoinRunningTask(int task_id);
  /*!
   * \brief Jointly tune a given list of tasks.
   * \param tasks The tasks to be tuned
   * \param task_weights The weight of each task
   * \param max_trials_global The maximum number of trials to be performed globally
   * \param max_trials_per_task The maximum number of trials to be performed for each task
   * \param num_trials_per_iter The number of trials to be performed in each iteration
   * \param builder The MetaSchedule builder
   * \param runner The MetaSchedule runner
   * \param measure_callbacks The callbacks to be called after each measurement
   * \param database The database used in tuning
   * \param cost_model The cost model used in tuning
   */
  virtual void Tune(ffi::Array<TuneContext> tasks,                  //
                    ffi::Array<FloatImm> task_weights,              //
                    int max_trials_global,                          //
                    int max_trials_per_task,                        //
                    int num_trials_per_iter,                        //
                    Builder builder,                                //
                    Runner runner,                                  //
                    ffi::Array<MeasureCallback> measure_callbacks,  //
                    ffi::Optional<Database> database,               //
                    ffi::Optional<CostModel> cost_model);
  /*!
   * \brief Terminate a task
   * \param task_id The id of the task to be terminated
   */
  void TerminateTask(int task_id);
  /*!
   * \brief Touch the task and update its status
   * \param task_id The task id to be checked.
   */
  void TouchTask(int task_id);
  /*! \brief Print out a human-readable format of the tuning statistics. */
  void PrintTuningStatistics();

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("s_tir.meta_schedule.TaskScheduler", TaskSchedulerObj, ffi::Object);
};

class TaskScheduler;

/*! \brief The task scheduler with customized methods on the python-side. */
class PyTaskSchedulerObj : public TaskSchedulerObj {
 public:
  /*!
   * \brief The function type of `NextTaskId` method.
   * \return The next task id.
   */
  using FNextTaskId = ffi::TypedFunction<int()>;
  /*!
   * \brief The function type of `JoinRunningTask` method.
   * \param task_id The task id to be joined.
   */
  using FJoinRunningTask = ffi::TypedFunction<ffi::Array<RunnerResult>(int)>;
  /*! \brief The function type of `Tune` method. */
  using FTune = ffi::TypedFunction<void(ffi::Array<TuneContext> tasks,                  //
                                        ffi::Array<FloatImm> task_weights,              //
                                        int max_trials_global,                          //
                                        int max_trials_per_task,                        //
                                        int num_trials_per_iter,                        //
                                        Builder builder,                                //
                                        Runner runner,                                  //
                                        ffi::Array<MeasureCallback> measure_callbacks,  //
                                        ffi::Optional<Database> database,               //
                                        ffi::Optional<CostModel> cost_model)>;

  /*! \brief The packed function to the `NextTaskId` function. */
  FNextTaskId f_next_task_id;
  /*! \brief The packed function to the `JoinRunningTask` function. */
  FJoinRunningTask f_join_running_task;
  /*! \brief The packed function to the `Tune` function. */
  FTune f_tune;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PyTaskSchedulerObj>();
  }

  int NextTaskId() final;
  ffi::Array<RunnerResult> JoinRunningTask(int task_id) final;
  void Tune(ffi::Array<TuneContext> tasks, ffi::Array<FloatImm> task_weights, int max_trials_global,
            int max_trials_per_task, int num_trials_per_iter, Builder builder, Runner runner,
            ffi::Array<MeasureCallback> measure_callbacks, ffi::Optional<Database> database,
            ffi::Optional<CostModel> cost_model) final;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.PyTaskScheduler", PyTaskSchedulerObj,
                                    TaskSchedulerObj);
};

/*!
 * \brief Managed reference to TaskSchedulerObj.
 * \sa TaskSchedulerObj
 */
class TaskScheduler : public ffi::ObjectRef {
 public:
  explicit TaskScheduler(ffi::ObjectPtr<TaskSchedulerObj> data) : ffi::ObjectRef(data) {
    TVM_FFI_ICHECK(data != nullptr);
  }
  /*!
   * \brief Create a task scheduler that fetches tasks in a round-robin fashion.
   * \param logger The tuning task's logging function.
   * \return The task scheduler created.
   */
  TVM_DLL static TaskScheduler RoundRobin(ffi::Function logger);
  /*!
   * \brief Create a task scheduler that fetches tasks in a gradient based fashion.
   * \param logger The tuning task's logging function.
   * \param alpha The parameter alpha to control gradient computation.
   * \param window_size The parameter to control backward window size.
   * \param seed The random seed.
   * \return The task scheduler created.
   */
  TVM_DLL static TaskScheduler GradientBased(ffi::Function logger, double alpha, int window_size,
                                             LinearCongruentialEngine::TRandState seed);
  /*!
   * \brief Create a task scheduler with customized methods on the python-side.
   * \param logger The tuning task's logging function.
   * \param f_next_task_id The packed function of `NextTaskId`.
   * \param f_join_running_task The packed function of `JoinRunningTask`.
   * \param f_tune The packed function of `Tune`.
   * \return The task scheduler created.
   */
  TVM_DLL static TaskScheduler PyTaskScheduler(
      ffi::Function logger, PyTaskSchedulerObj::FNextTaskId f_next_task_id,
      PyTaskSchedulerObj::FJoinRunningTask f_join_running_task, PyTaskSchedulerObj::FTune f_tune);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TaskScheduler, ffi::ObjectRef, TaskSchedulerObj);
};

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm

#endif  // TVM_S_TIR_META_SCHEDULE_TASK_SCHEDULER_H_
