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

/*! \brief The gradient based task scheduler. */
class GradientBasedObj final : public TaskSchedulerObj {
 public:
  double alpha;
  int window_size;
  LinearCongruentialEngine::TRandState rand_state;

  int round_robin_rounds_;
  std::vector<std::vector<double>> best_latency_history_;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GradientBasedObj>()
        .def_ro("alpha", &GradientBasedObj::alpha)
        .def_ro("window_size", &GradientBasedObj::window_size);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.GradientBased", GradientBasedObj,
                                    TaskSchedulerObj);

 public:
  void Tune(ffi::Array<TuneContext> tasks, ffi::Array<FloatImm> task_weights, int max_trials_global,
            int max_trials_per_task, int num_trials_per_iter, Builder builder, Runner runner,
            ffi::Array<MeasureCallback> measure_callbacks, ffi::Optional<Database> database,
            ffi::Optional<CostModel> cost_model) final {
    int n_tasks = tasks.size();
    round_robin_rounds_ = 0;
    best_latency_history_.resize(n_tasks, std::vector<double>());
    TaskSchedulerObj::Tune(tasks, task_weights, max_trials_global, max_trials_per_task,
                           num_trials_per_iter, builder, runner, measure_callbacks, database,
                           cost_model);
  }

  int NextTaskId() final {
    int n_tasks = this->tasks_.size();
    // Step 1. Check if it's in round robin mode.
    if (round_robin_rounds_ == 0) {
      TVM_PY_LOG_CLEAR_SCREEN(this->logger);
      this->PrintTuningStatistics();
    }
    if (round_robin_rounds_ < n_tasks) {
      return round_robin_rounds_++;
    }
    if (round_robin_rounds_ == n_tasks) {
      for (int i = 0; i < n_tasks; ++i) {
        if (this->tasks_[i]->runner_futures.has_value()) {
          this->JoinRunningTask(i);
        }
      }
      ++round_robin_rounds_;
    }
    // Step 2. Collect the tasks that are not terminated yet
    std::vector<int> tasks_alive;
    {
      tasks_alive.reserve(n_tasks);
      for (int i = 0; i < n_tasks; ++i) {
        this->TouchTask(i);
        if (!this->tasks_[i]->is_terminated) {
          tasks_alive.push_back(i);
        }
      }
      if (tasks_alive.empty()) {
        return -1;
      }
    }
    // Step 3. Calculate the gradient of each task alive
    std::vector<double> grad;
    grad.reserve(n_tasks);
    for (int task_id : tasks_alive) {
      const std::vector<double>& best_latency = this->best_latency_history_.at(task_id);
      int n = best_latency.size();
      double task_weight = this->tasks_[task_id]->task_weight;
      int w = this->window_size;
      if (n > 0 && best_latency[n - 1] < 1e9) {
        double best = best_latency[n - 1];
        double g1 = (n >= 1 + w) ? (best_latency[n - 1 - w] - best) / w : 0.0;
        double g2 = best / n;
        double g = alpha * g1 + (1 - alpha) * g2;
        grad.push_back(g * task_weight);
      } else {
        // If the best time cost is unavailable, it means some task is not valid. Skip it.
        grad.push_back(-1e9);
      }
    }
    // Step 4. Select the task with the largest gradient
    auto max_grad = std::max_element(grad.begin(), grad.end());
    auto min_grad = std::min_element(grad.begin(), grad.end());
    int task_id = -1;
    if (*max_grad == *min_grad) {
      task_id = tasks_alive[s_tir::SampleInt(&this->rand_state, 0, tasks_alive.size())];
    } else {
      task_id = tasks_alive[std::distance(grad.begin(), max_grad)];
    }
    if (this->tasks_[task_id]->runner_futures.has_value()) {
      JoinRunningTask(task_id);
    }
    return task_id;
  }

  ffi::Array<RunnerResult> JoinRunningTask(int task_id) final {
    ffi::Array<RunnerResult> results = TaskSchedulerObj::JoinRunningTask(task_id);
    TaskRecordObj* task = this->tasks_[task_id].get();
    if (task->latency_ms.size() > 0) {
      this->best_latency_history_.at(task_id).push_back(
          *std::min_element(task->latency_ms.begin(),  //
                            task->latency_ms.end()));
    }
    return results;
  }
};

TaskScheduler TaskScheduler::GradientBased(ffi::Function logger, double alpha, int window_size,
                                           LinearCongruentialEngine::TRandState seed) {
  ffi::ObjectPtr<GradientBasedObj> n = ffi::make_object<GradientBasedObj>();
  n->logger = logger;
  n->alpha = alpha;
  n->window_size = window_size;
  n->rand_state = LinearCongruentialEngine::NormalizeSeed(seed);
  return TaskScheduler(n);
}

TVM_FFI_STATIC_INIT_BLOCK() { GradientBasedObj::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.meta_schedule.TaskSchedulerGradientBased",
                        TaskScheduler::GradientBased);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
