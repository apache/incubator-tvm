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

/*! \brief A search strategy that generates measure candidates using space generator. */
class ReplayFuncObj : public SearchStrategyObj {
 public:
  /*! \brief The state of the search strategy. */
  struct State {
    /*! \brief The search strategy itself */
    ReplayFuncObj* self;
    /*! \brief The number of total trials. */
    int max_trials;
    /*! \brief The number of trials per iteration. */
    int num_trials_per_iter;
    /*! \brief `[st, ed)` are the indices of the next batch of candidates. */
    int st;
    /*! \brief `[st, ed)` are the indices of the next batch of candidates. */
    int ed;

    explicit State(ReplayFuncObj* self, int max_trials, int num_trials_per_iter)
        : self(self),
          max_trials(max_trials),
          num_trials_per_iter(num_trials_per_iter),
          st(0),
          ed(num_trials_per_iter) {
      TVM_FFI_CHECK(self->mod_.has_value() && self->space_generator_.has_value(), ValueError)
          << "The search strategy has not been initialized.";
    }

    inline ffi::Optional<ffi::Array<MeasureCandidate>> GenerateMeasureCandidates();
    inline void NotifyRunnerResults(const ffi::Array<RunnerResult>& results);
  };

  /*! \brief The random state. -1 means using random number. */
  TRandState rand_state_ = -1;
  /*! \brief The IRModule to be scheduled from TuneContext. */
  ffi::Optional<IRModule> mod_ = std::nullopt;
  /*! \brief The space generator from TuneContext. */
  ffi::Optional<SpaceGenerator> space_generator_ = std::nullopt;
  /*! \brief The state of the search strategy. */
  std::unique_ptr<State> state_ = nullptr;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ReplayFuncObj>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.ReplayFunc", ReplayFuncObj,
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
    this->space_generator_ = ctx->space_generator;
    this->state_.reset();
  }

  void PreTuning(int max_trials, int num_trials_per_iter,
                 const ffi::Array<s_tir::Schedule>& design_spaces,
                 const ffi::Optional<Database>& database,
                 const ffi::Optional<CostModel>& cost_model) final {
    TVM_FFI_CHECK(this->state_ == nullptr, ValueError)
        << "`PreTuning` is already invoked without corresponding `PostTuning`.";
    this->state_ = std::make_unique<State>(this, max_trials, num_trials_per_iter);
  }

  void PostTuning() final {
    TVM_FFI_CHECK(this->state_ != nullptr, ValueError)
        << "`PostTuning` is invoked without corresponding "
           "`PreTuning`, or `PostTuning` is already invoked.";
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
    ffi::ObjectPtr<ReplayFuncObj> n = ffi::make_object<ReplayFuncObj>();
    n->rand_state_ = -1;
    n->mod_ = std::nullopt;
    n->space_generator_ = std::nullopt;
    n->state_ = nullptr;
    return SearchStrategy(ffi::UnsafeInit{}, n);
  }
};

inline ffi::Optional<ffi::Array<MeasureCandidate>>
ReplayFuncObj::State::GenerateMeasureCandidates() {
  if (st >= max_trials) {
    return std::nullopt;
  }
  ed = std::min(ed, max_trials);
  ffi::Array<MeasureCandidate> result;
  IRModule mod = self->mod_.value();
  ffi::Array<Postproc> postprocs = self->space_generator_.value()->postprocs.value_or({});
  for (int i = st; i < ed; i++) {
    for (;;) {
      ffi::Array<s_tir::Schedule> schs = self->space_generator_.value()->GenerateDesignSpace(mod);
      int design_space_index = s_tir::SampleInt(&self->rand_state_, 0, schs.size());
      s_tir::Schedule sch = schs[design_space_index];
      sch->EnterPostproc();
      bool failed = false;
      for (const Postproc& proc : postprocs) {
        if (!proc->Apply(sch)) {
          failed = true;
          break;
        }
      }
      if (!failed) {
        ffi::Array<ArgInfo> args_info = ArgInfo::FromEntryFunc(sch->mod(), /*remove_preproc=*/true);
        result.push_back(MeasureCandidate(sch, args_info));
        break;
      }
    }
  }
  return result;
}

inline void ReplayFuncObj::State::NotifyRunnerResults(const ffi::Array<RunnerResult>& results) {
  st += num_trials_per_iter;
  ed += num_trials_per_iter;
}

SearchStrategy SearchStrategy::ReplayFunc() {
  ffi::ObjectPtr<ReplayFuncObj> n = ffi::make_object<ReplayFuncObj>();
  return SearchStrategy(ffi::UnsafeInit{}, n);
}

TVM_FFI_STATIC_INIT_BLOCK() { ReplayFuncObj::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.meta_schedule.SearchStrategyReplayFunc", SearchStrategy::ReplayFunc);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
