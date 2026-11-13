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

/*! \brief A search strategy that generates measure candidates using trace and random decisions. */
class ReplayTraceObj : public SearchStrategyObj {
 public:
  /*! \brief The state of the search strategy. */
  struct State {
    /*! \brief The search strategy itself */
    ReplayTraceObj* self;
    /*! \brief The design spaces. */
    ffi::Array<s_tir::Trace> design_spaces;
    /*! \brief The number of total trials. */
    int max_trials;
    /*! \brief The number of trials per iteration. */
    int num_trials_per_iter;
    /*! \brief `[st, ed)` are the indices of the next batch of candidates. */
    int st;
    /*! \brief `[st, ed)` are the indices of the next batch of candidates. */
    int ed;

    /*! \brief The module to be tuned. */
    ffi::Array<IRModule> per_thread_mod_;

    explicit State(ReplayTraceObj* self, ffi::Array<s_tir::Trace> design_spaces, int max_trials,
                   int num_trials_per_iter)
        : self(self),
          design_spaces(design_spaces),
          max_trials(max_trials),
          num_trials_per_iter(num_trials_per_iter),
          st(0),
          ed(num_trials_per_iter) {
      IRModule mod = self->mod_.value();
      this->per_thread_mod_.reserve(self->num_threads_);
      for (int i = 0; i < self->num_threads_; i++) {
        this->per_thread_mod_.push_back(DeepCopyIRModule(mod));
      }
    }

    inline ffi::Optional<ffi::Array<MeasureCandidate>> GenerateMeasureCandidates();
    inline void NotifyRunnerResults(const ffi::Array<RunnerResult>& results);
  };

  /*! \brief The max number of failures during trace replaying. */
  int max_fail_count;

  /*! \brief The random state. -1 means using random number. */
  TRandState rand_state_ = -1;
  /*! \brief The IRModule to be scheduled from TuneContext. */
  ffi::Optional<IRModule> mod_ = std::nullopt;
  /*! \brief The number of threads to be used. */
  int num_threads_ = -1;
  /*! \brief The postprocessors. */
  ffi::Array<Postproc> postprocs_ = {};
  /*! \brief The state of the search strategy. */
  std::unique_ptr<State> state_ = nullptr;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ReplayTraceObj>().def_ro("max_fail_count", &ReplayTraceObj::max_fail_count);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.ReplayTrace", ReplayTraceObj,
                                    SearchStrategyObj);

  void InitializeWithTuneContext(const TuneContext& ctx) final {
    TVM_FFI_CHECK(ctx->mod.has_value(), ValueError) << "TuneContext.mod is not defined";
    TVM_FFI_CHECK(ctx->space_generator.has_value(), ValueError)
        << "TuneContext.space_generator is not defined";
    if (!ctx->space_generator.value()->postprocs.has_value()) {
      TVM_PY_LOG(WARNING, ctx->logger)
          << "`postprocs` is not defined in " << ctx->space_generator.value()
          << ". Please explicitly set `postprocs` to an empty list if you don't want to "
             "apply any post-processing.";
    }
    this->rand_state_ = ForkSeed(&ctx->rand_state);
    this->mod_ = ctx->mod;
    this->num_threads_ = ctx->num_threads;
    this->postprocs_ = ctx->space_generator.value()->postprocs.value_or({});
    this->state_.reset();
  }

  void PreTuning(int max_trials, int num_trials_per_iter,
                 const ffi::Array<s_tir::Schedule>& design_spaces,
                 const ffi::Optional<Database>& database,
                 const ffi::Optional<CostModel>& cost_model) final {
    TVM_FFI_ICHECK(!design_spaces.empty());
    TVM_FFI_CHECK(this->state_ == nullptr, ValueError)
        << "`PreTuning` is already invoked without corresponding `PostTuning`.";
    ffi::Array<s_tir::Trace> design_space_traces;
    design_space_traces.reserve(design_spaces.size());
    for (const s_tir::Schedule& space : design_spaces) {
      design_space_traces.push_back(space->trace().value()->Simplified(true));
    }
    this->state_ =
        std::make_unique<State>(this, design_space_traces, max_trials, num_trials_per_iter);
  }

  void PostTuning() final {
    TVM_FFI_ICHECK(this->state_ != nullptr);
    this->state_.reset();
  }

  ffi::Optional<ffi::Array<MeasureCandidate>> GenerateMeasureCandidates() final {
    TVM_FFI_ICHECK(this->state_ != nullptr);
    return this->state_->GenerateMeasureCandidates();
  }

  void NotifyRunnerResults(const ffi::Array<MeasureCandidate>& measure_candidates,
                           const ffi::Array<RunnerResult>& results) final {
    TVM_FFI_ICHECK(this->state_ != nullptr);
    this->state_->NotifyRunnerResults(results);
  }

  SearchStrategy Clone() const final {
    ffi::ObjectPtr<ReplayTraceObj> n = ffi::make_object<ReplayTraceObj>();
    n->max_fail_count = this->max_fail_count;
    n->rand_state_ = this->rand_state_;
    n->state_ = nullptr;  // cleared the state
    return SearchStrategy(ffi::UnsafeInit{}, n);
  }
};

inline ffi::Optional<ffi::Array<MeasureCandidate>>
ReplayTraceObj::State::GenerateMeasureCandidates() {
  if (st >= max_trials) {
    return std::nullopt;
  }
  ed = std::min(ed, max_trials);
  TVM_FFI_ICHECK_LT(st, ed);
  std::vector<TRandState> per_thread_rand_state = ForkSeed(&self->rand_state_, self->num_threads_);
  ffi::Array<ffi::Optional<MeasureCandidate>> per_task_result(ed - st, std::nullopt);
  ThreadedTraceApply pp(self->postprocs_);
  auto f_worker = [this, &per_thread_rand_state, &per_task_result, &pp](int thread_id,
                                                                        int task_id) -> void {
    TRandState& rand_state = per_thread_rand_state[thread_id];
    IRModule mod = this->per_thread_mod_[thread_id];

    for (int fail_count = 0; fail_count < self->max_fail_count; fail_count++) {
      int design_space_index = s_tir::SampleInt(&rand_state, 0, design_spaces.size());
      s_tir::Trace trace = design_spaces[design_space_index];
      s_tir::Trace new_trace = s_tir::Trace(trace->insts, {});
      if (ffi::Optional<s_tir::Schedule> opt_sch = pp.Apply(mod, new_trace, &rand_state)) {
        s_tir::Schedule sch = opt_sch.value();
        ffi::Array<ArgInfo> args_info = ArgInfo::FromEntryFunc(sch->mod(), /*remove_preproc=*/true);
        per_task_result.Set(task_id, MeasureCandidate(sch, args_info));
        break;
      }
    }
  };
  support::parallel_for_dynamic(0, ed - st, self->num_threads_, f_worker);
  ffi::Array<MeasureCandidate> filtered;
  filtered.reserve(ed - st);
  for (ffi::Optional<MeasureCandidate> result : per_task_result)
    if (result.has_value()) {
      filtered.push_back(*std::move(result));
    }
  return filtered;
}

inline void ReplayTraceObj::State::NotifyRunnerResults(const ffi::Array<RunnerResult>& results) {
  st += num_trials_per_iter;
  ed += num_trials_per_iter;
}

SearchStrategy SearchStrategy::ReplayTrace(int max_fail_count) {
  ffi::ObjectPtr<ReplayTraceObj> n = ffi::make_object<ReplayTraceObj>();
  n->max_fail_count = max_fail_count;
  return SearchStrategy(ffi::UnsafeInit{}, n);
}

TVM_FFI_STATIC_INIT_BLOCK() { ReplayTraceObj::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.meta_schedule.SearchStrategyReplayTrace",
                        SearchStrategy::ReplayTrace);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
