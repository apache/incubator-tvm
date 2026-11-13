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

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>

#include <mutex>

#include "../module_equality.h"
#include "../utils.h"

#define TVM_META_SCHEDULE_CHECK_PROB_RANGE(p, name)   \
  TVM_FFI_CHECK(0.0 <= (p) && (p) <= 1.0, ValueError) \
      << "name should be within [0, 1], "             \
      << "but get `" << #p << " = " << (p) << '\'';

namespace tvm {
namespace s_tir {
namespace meta_schedule {

using s_tir::Schedule;

/**************** Data Structure ****************/

/*! \brief An auxiliary data structure to help deduplicate IRModules */
class IRModuleSet {
 public:
  explicit IRModuleSet(const ModuleEquality& mod_eq)
      : tab_(/*bucket_count*/ 0, ItemHash(), ItemEqual(mod_eq)) {}

  /*! \brief Add an IRModule to the set */
  void Add(const IRModule& mod, size_t shash) { tab_.insert(Item{mod, shash}); }
  /*! \brief Check if the IRModule is in the set */
  bool Has(const IRModule& mod, size_t shash) const { return tab_.count(Item{mod, shash}); }

 private:
  struct Item {
    IRModule mod;
    size_t shash;
  };
  struct ItemHash {
    size_t operator()(const Item& hash) const { return hash.shash; }
  };
  struct ItemEqual {
    explicit ItemEqual(const ModuleEquality& mod_eq) : mod_eq_(mod_eq) {}
    ItemEqual& operator=(const ItemEqual& other) { return *this; }

    bool operator()(const Item& lhs, const Item& rhs) const {
      return lhs.shash == rhs.shash && mod_eq_.Equal(lhs.mod, rhs.mod);
    }

    const ModuleEquality& mod_eq_;
  };

  std::unordered_set<Item, ItemHash, ItemEqual> tab_;
};

/*!
 * \brief A heap with a size up-limit. If overflow happens, it evicted the worst items.
 * \note It maintains a min heap in terms of `Item::score`. Therefore, when
 * overflow happens, the element evicted is the one with the min `Item::score`.
 * As time goes, the elements in the heap are going to be larger.
 */
class SizedHeap {
 public:
  struct Item {
    Schedule sch;
    double score;
    bool operator<(const Item& other) const { return score > other.score; }
  };

  /*!
   * \brief Constructor
   * \param size_limit The up-limit of the heap size
   */
  explicit SizedHeap(int size_limit) : size_limit(size_limit) { heap.reserve(size_limit); }

  /*!
   * \brief Push the specific item to the heap if its key did not appears in the heap
   * \param item The item to be pushed
   */
  void Push(Schedule sch, double score) {
    int size = heap.size();
    if (size < size_limit) {
      // Heap is not full, just push
      heap.emplace_back(Item{sch, score});
      std::push_heap(heap.begin(), heap.end());
    } else if (score > heap.front().score) {
      // if the item is better than the worst one in the heap, we can safely kick it out
      std::pop_heap(heap.begin(), heap.end());
      heap.back() = {sch, score};
      std::push_heap(heap.begin(), heap.end());
    }
    // Otherwise, the item is worse than any other element in the heap
  }

  /*! \brief Up-limit of the heap size */
  int size_limit;
  /*! \brief The heap, the worse the topper */
  std::vector<Item> heap;
};

struct PerThreadData {
  IRModule mod{ffi::UnsafeInit()};
  TRandState rand_state{-1};
  std::function<int32_t()> trace_sampler = nullptr;
  std::function<ffi::Optional<Mutator>()> mutator_sampler = nullptr;

  /*!
   * \brief Set the value for the trace and mutator samplers per thread.
   * \param scores The predicted score for the given samples.
   * \param genetic_mutate_prob The probability of mutation.
   * \param mutator_probs The probability of each mutator as a dict.
   */
  void Set(const std::vector<double>& scores, double genetic_mutate_prob,
           const ffi::Map<Mutator, FloatImm>& mutator_probs) {
    trace_sampler = s_tir::MakeMultinomialSampler(&rand_state, scores);
    mutator_sampler = MakeMutatorSampler(genetic_mutate_prob, mutator_probs, &rand_state);
  }

 private:
  /*!
   * \brief Create a sampler function that picks mutators according to the mass function
   * \param rand_state The random state for sampling
   * \return The sampler created
   */
  static std::function<ffi::Optional<Mutator>()> MakeMutatorSampler(
      double genetic_mutate_prob,                        //
      const ffi::Map<Mutator, FloatImm>& mutator_probs,  //
      TRandState* rand_state) {
    std::vector<ffi::Optional<Mutator>> mutators;
    std::vector<double> masses;
    mutators.push_back(std::nullopt);
    masses.push_back(1.0 - genetic_mutate_prob);
    double total_mass_mutator = 0.0;
    if (genetic_mutate_prob > 0) {
      for (const auto& kv : mutator_probs) {
        Mutator mutator = kv.first;
        double mass = kv.second->value;
        total_mass_mutator += mass;
        mutators.push_back(mutator);
        masses.push_back(mass * genetic_mutate_prob);
      }
    }
    // Normalize the sum to 1.0
    if (total_mass_mutator == 0.0) {
      masses[0] = 1.0;
      for (int i = 1, n = masses.size(); i < n; ++i) {
        masses[i] = 0.0;
      }
    } else if (total_mass_mutator != 1.0) {
      for (int i = 1, n = masses.size(); i < n; ++i) {
        masses[i] /= total_mass_mutator;
      }
    }
    return [idx_sampler = s_tir::MakeMultinomialSampler(rand_state, masses),
            mutators = std::move(mutators)]() -> ffi::Optional<Mutator> {
      int i = idx_sampler();
      return mutators[i];
    };
  }
};

struct ConcurrentBitmask {
  /*! The bit width. */
  static constexpr const int kBitWidth = 64;
  /*! \brief The size of the concurrent bitmask. */
  int size;
  /*! \brief The bitmasks. */
  std::vector<uint64_t> bitmask;
  /*! \brief The mutexes, one per kBitWidth(64 here) bitmasks. */
  std::vector<std::mutex> mutexes;

  /*!
   * \brief Constructor
   * \param n The total slots managed by the concurrent bitmask.
   */
  explicit ConcurrentBitmask(int n)
      : size((n + kBitWidth - 1) / kBitWidth), bitmask(size, 0), mutexes(size) {}
  /*!
   * \brief Query and mark the given index if not visited before.
   * \param x The index to concurrently check if used. If not, mark as used.
   * \return Whether the index has been used before.
   */
  bool QueryAndMark(int x) {
    constexpr uint64_t one = 1;
    std::unique_lock<std::mutex> lock(mutexes[x / kBitWidth]);
    if (bitmask[x / kBitWidth] & (one << (x % kBitWidth))) {
      return false;
    } else {
      bitmask[x / kBitWidth] |= one << (x % kBitWidth);
      return true;
    }
  }
};

/**************** Util Functions ****************/

/*!
 * \brief Assemble measure candidates from the given candidate traces.
 * \param traces The picked candidate traces.
 * \return The assembled measure candidates.
 */
ffi::Array<MeasureCandidate> AssembleCandidates(const std::vector<Schedule>& picks) {
  ffi::Array<MeasureCandidate> measure_inputs;
  measure_inputs.reserve(picks.size());
  for (const Schedule& sch : picks) {
    measure_inputs.push_back(
        MeasureCandidate(sch, ArgInfo::FromEntryFunc(sch->mod(), /*remove_preproc=*/true)));
  }
  return measure_inputs;
}

/*!
 * \brief Predict the normalized score of each candidate.
 * \param candidates The candidates for prediction
 * \param task The search task
 * \param space The search space
 * \return The normalized score in the prediction
 */
std::vector<double> PredictNormalizedScore(const std::vector<Schedule>& candidates,
                                           const TuneContext& context,
                                           const CostModel& cost_model) {
  auto _ = Profiler::TimedScope("EvoSearch/Evolve/PredictNormalizedScore");
  TVM_FFI_ICHECK(!candidates.empty())
      << "Candidates given for score prediction can not be empty list!";
  std::vector<double> scores = cost_model->Predict(context, AssembleCandidates(candidates));
  for (double& score : scores) {
    score = std::max(0.0, score);
  }
  return scores;
}

/**************** Evolutionary Search ****************/

/*!\brief A search strategy that generates measure candidates using evolutionary search. */
class EvolutionarySearchObj : public SearchStrategyObj {
 public:
  /*! \brief The state of the search strategy. */
  struct State {
    /*! \brief The search strategy itself */
    EvolutionarySearchObj* self;
    /*! \brief The number of total trials. */
    int max_trials;
    /*! \brief The number of trials per iteration. */
    int num_trials_per_iter;
    /*! \brief `[st, ed)` are the indices of the next batch of candidates. */
    int st;
    /*! \brief `[st, ed)` are the indices of the next batch of candidates. */
    int ed;
    /*! \brief The counter of returning empty results. */
    int num_empty_iters;
    /*! \brief The design spaces. Decisions are not used so traces only. */
    ffi::Array<s_tir::Trace> design_spaces;
    /*! \brief Pre thread data including module to be tuned and random state. */
    std::vector<PerThreadData> per_thread_data_;
    /*!
     * \brief The workloads that are already measured.
     * TODO(junrushao1994): add records from the database to avoid re-measuring.
     * */
    IRModuleSet measured_workloads_;
    /*! \brief A Database for selecting useful candidates. */
    Database database_{ffi::UnsafeInit()};
    /*! \brief A cost model helping to explore the search space */
    CostModel cost_model_{ffi::UnsafeInit()};
    /*! \brief The token registered for the given workload in database. */
    Workload token_{ffi::UnsafeInit()};

    explicit State(EvolutionarySearchObj* self, int max_trials, int num_trials_per_iter,
                   ffi::Array<Schedule> design_space_schedules, Database database,
                   CostModel cost_model)
        : self(self),
          max_trials(max_trials),
          num_trials_per_iter(num_trials_per_iter),
          st(0),
          ed(num_trials_per_iter),
          num_empty_iters(0),
          measured_workloads_(database->GetModuleEquality()) {
      design_spaces.reserve(design_space_schedules.size());
      for (const Schedule& space : design_space_schedules) {
        design_spaces.push_back(space->trace().value()->Simplified(true));
      }
      const TuneContextObj* ctx = self->ctx_;
      IRModule mod = ctx->mod.value();
      this->per_thread_data_.resize(ctx->num_threads);
      for (PerThreadData& data : this->per_thread_data_) {
        data.mod = DeepCopyIRModule(mod);
        data.rand_state = ForkSeed(&self->rand_state_);
      }
      this->database_ = database;
      this->cost_model_ = cost_model;
      this->token_ = database->CommitWorkload(mod);
    }

    /*!
     * \brief Pick up best candidates from database.
     * \param num The number of traces to produce.
     * \return The picked best candidates.
     */
    inline std::vector<Schedule> PickBestFromDatabase(int num);
    /*!
     * \brief Sample the initial population from previous measured results and randomly generated
     *  traces via trace replaying.
     * \param num The number of traces to produce.
     * \return The initial population of traces sampled.
     */
    inline std::vector<Schedule> SampleInitPopulation(int num);
    /*!
     * \brief Evolve the initial population using mutators and samplers.
     * \param population The initial population of traces sampled.
     * \param num The number of traces to produce.
     * \return The evolved traces from initial population.
     */
    inline std::vector<Schedule> EvolveWithCostModel(std::vector<Schedule> population, int num);
    /*!
     * \brief Pick final candidates from the given initial population and bests of evolved ones.
     * \param inits The initial population of traces sampled.
     * \param bests The best candidates predicted from evolved traces.
     * \param num The number of traces to produce.
     * \return The final picked candidates with a ratio of both.
     */
    inline std::vector<Schedule> PickWithEpsGreedy(const std::vector<Schedule>& inits,
                                                   const std::vector<Schedule>& bests, int num);
    /*! \brief An interface method to be called by it's counterpart in EvolutionarySearchObj */
    inline ffi::Optional<ffi::Array<MeasureCandidate>> GenerateMeasureCandidates();
    /*! \brief An interface method to be called by it's counterpart in EvolutionarySearchObj */
    inline void NotifyRunnerResults(const ffi::Array<MeasureCandidate>& measure_candidates,
                                    const ffi::Array<RunnerResult>& results);
    /*!
     * \brief Compute the hash for the given module.
     * \param mod The input TIR module.
     * \return The calculated hash.
     */
    inline size_t ModuleHash(const IRModule& mod) const;
  };

  /*! \brief The tuning context of the evolutionary search strategy. */
  const TuneContextObj* ctx_{nullptr};
  /*! \brief The postprocessors */
  ffi::Array<Postproc> postprocs_;
  /*! \brief The mutators and their probability. */
  ffi::Map<Mutator, FloatImm> mutator_probs_;
  /*! \brief The random state. To be initialized with TuneContext. */
  TRandState rand_state_;
  /*! \brief The state of the search strategy. */
  std::unique_ptr<State> state_ = nullptr;

  /*** Configuration: global ***/
  /*! \brief The population size in the evolutionary search. */
  int population_size;
  /*!
   * \brief The maximum number of iterations before early stopping to confirm the search space is
   * exhausted
   */
  int num_empty_iters_before_early_stop;
  /*** Configuration: the initial population ***/
  /*! \brief The ratio of measured states used in the initial population */
  double init_measured_ratio;
  /*! \brief The minimal size of unmeasured population in the initial sampling.*/
  int init_min_unmeasured;
  /*! \brief The maximum number of failure during initial sampling. */
  int max_fail_count;
  /*** Configuration: evolution ***/
  /*! \brief The number of iterations performed by generic algorithm. */
  int genetic_num_iters;
  /*! \brief The probability to perform mutation */
  double genetic_mutate_prob;
  /*! \brief The maximum number to try evolving the given trace. */
  int genetic_max_fail_count;
  /*** Configuration: pick states for measurement ***/
  /*! \brief The ratio of measurements to use randomly sampled states. */
  double eps_greedy;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<EvolutionarySearchObj>()
        .def_ro("population_size", &EvolutionarySearchObj::population_size)
        .def_ro("num_empty_iters_before_early_stop",
                &EvolutionarySearchObj::num_empty_iters_before_early_stop)
        .def_ro("init_measured_ratio", &EvolutionarySearchObj::init_measured_ratio)
        .def_ro("init_min_unmeasured", &EvolutionarySearchObj::init_min_unmeasured)
        .def_ro("max_fail_count", &EvolutionarySearchObj::max_fail_count)
        .def_ro("genetic_num_iters", &EvolutionarySearchObj::genetic_num_iters)
        .def_ro("genetic_mutate_prob", &EvolutionarySearchObj::genetic_mutate_prob)
        .def_ro("genetic_max_fail_count", &EvolutionarySearchObj::genetic_max_fail_count)
        .def_ro("eps_greedy", &EvolutionarySearchObj::eps_greedy);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.EvolutionarySearch", EvolutionarySearchObj,
                                    SearchStrategyObj);

  void InitializeWithTuneContext(const TuneContext& ctx) final {
    TVM_FFI_CHECK(ctx->num_threads > 0, ValueError) << "`TuneContext.num_threads` must be > 0";
    TVM_FFI_CHECK(ctx->space_generator.has_value(), ValueError)
        << "`TuneContext.space_generator` must be defined";
    TVM_FFI_CHECK(ctx->space_generator.value()->postprocs.has_value(), ValueError)
        << "`TuneContext.space_generator.postprocs` must be defined";
    TVM_FFI_CHECK(ctx->space_generator.value()->mutator_probs.has_value(), ValueError)
        << "`TuneContext.space_generator.mutator_probs` must be defined";
    this->ctx_ = ctx.get();
    this->postprocs_ = ctx->space_generator.value()->postprocs.value();
    this->mutator_probs_ = ctx->space_generator.value()->mutator_probs.value();
    this->rand_state_ = ForkSeed(&ctx->rand_state);
    this->state_.reset();
  }

  void PreTuning(int max_trials, int num_trials_per_iter, const ffi::Array<Schedule>& design_spaces,
                 const ffi::Optional<Database>& database,
                 const ffi::Optional<CostModel>& cost_model) final {
    TVM_FFI_ICHECK(!design_spaces.empty());
    TVM_FFI_CHECK(this->ctx_ != nullptr, ValueError)
        << "Did you forget to initialize the TuneContext?";
    TVM_FFI_CHECK(database.has_value(), ValueError)
        << "Database is not supplied in PreTuning. Evolutionary"
           "search algorithm requires a database to be present, so that it "
           "could sample from previously-explored population. If you do not "
           "intent to store data on disk, please use "
           "`tvm.s_tir.meta_schedule.database.MemoryDatabase`";
    TVM_FFI_CHECK(cost_model.has_value(), ValueError)
        << "CostModel is not supplied in PreTuning. Evolutionary search "
           "algorithm expects a cost model to filter out potentially less efficient kernels. If "
           "you do not expect a cost model to help, please use "
           "`tvm.s_tir.meta_schedule.cost_model.RandomModel`";
    TVM_FFI_CHECK(this->state_ == nullptr, ValueError)
        << "`PreTuning` is already invoked without corresponding `PostTuning`.";
    this->state_ = std::make_unique<State>(this, max_trials, num_trials_per_iter, design_spaces,
                                           database.value(), cost_model.value());
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
    this->state_->NotifyRunnerResults(measure_candidates, results);
  }

  SearchStrategy Clone() const final {
    ffi::ObjectPtr<EvolutionarySearchObj> n = ffi::make_object<EvolutionarySearchObj>();
    n->population_size = this->population_size;
    n->num_empty_iters_before_early_stop = this->num_empty_iters_before_early_stop;
    n->init_measured_ratio = this->init_measured_ratio;
    n->init_min_unmeasured = this->init_min_unmeasured;
    n->max_fail_count = this->max_fail_count;
    n->genetic_num_iters = this->genetic_num_iters;
    n->genetic_mutate_prob = this->genetic_mutate_prob;
    n->genetic_max_fail_count = this->genetic_max_fail_count;
    n->eps_greedy = this->eps_greedy;
    n->ctx_ = this->ctx_;
    n->rand_state_ = this->rand_state_;
    n->state_ = nullptr;  // cleared the state
    return SearchStrategy(ffi::UnsafeInit{}, n);
  }
};

std::vector<Schedule> EvolutionarySearchObj::State::PickBestFromDatabase(int num) {
  auto _ = Profiler::TimedScope("EvoSearch/PickBestFromDatabase");
  std::vector<s_tir::Trace> measured_traces;
  measured_traces.reserve(num);
  ffi::Array<TuningRecord> top_records = this->database_->GetTopK(this->token_, num);
  for (TuningRecord record : top_records) {
    measured_traces.push_back(record->trace);
  }
  int actual_num = measured_traces.size();
  ThreadedTraceApply pp(self->postprocs_);
  std::vector<ffi::Optional<Schedule>> results(actual_num);
  auto f_proc_measured = [this, &measured_traces, &results, &pp](int thread_id,
                                                                 int trace_id) -> void {
    PerThreadData& data = this->per_thread_data_.at(thread_id);
    TRandState* rand_state = &data.rand_state;
    const IRModule& mod = data.mod;
    s_tir::Trace trace = measured_traces.at(trace_id);
    ffi::Optional<Schedule>& result = results.at(trace_id);
    TVM_FFI_ICHECK(!result.has_value());
    if (ffi::Optional<Schedule> sch = pp.Apply(mod, trace, rand_state)) {
      result = sch.value();
    }
  };
  support::parallel_for_dynamic(0, actual_num, self->ctx_->num_threads, f_proc_measured);
  TVM_PY_LOG(INFO, self->ctx_->logger) << "Pick-Best-From-Database summary:\n"
                                       << pp.SummarizeFailures();
  if (pp.TraceFailCount() > 0) {
    TVM_PY_LOG(WARNING, self->ctx_->logger)
        << "PickBestFromDatabase skipped " << pp.TraceFailCount()
        << " candidate(s) due to trace replay failures";
  }
  std::vector<Schedule> filtered;
  filtered.reserve(actual_num);
  for (const ffi::Optional<Schedule>& sch : results) {
    if (sch.has_value()) {
      filtered.push_back(sch.value());
    }
  }
  return filtered;
}

std::vector<Schedule> EvolutionarySearchObj::State::SampleInitPopulation(int num) {
  auto _ = Profiler::TimedScope("EvoSearch/SampleInitPopulation");
  ThreadedTraceApply pp(self->postprocs_);
  std::vector<Schedule> out_schs;
  int fail_count = 0;
  while (static_cast<int>(out_schs.size()) < self->init_min_unmeasured &&
         fail_count < self->max_fail_count) {
    std::vector<ffi::Optional<Schedule>> results(num);
    auto f_proc_unmeasured = [this, &results, &pp](int thread_id, int trace_id) -> void {
      PerThreadData& data = this->per_thread_data_.at(thread_id);
      TRandState* rand_state = &data.rand_state;
      const IRModule& mod = data.mod;
      ffi::Optional<Schedule>& result = results.at(trace_id);
      TVM_FFI_ICHECK(!result.has_value());
      int design_space_index = s_tir::SampleInt(rand_state, 0, design_spaces.size());
      s_tir::Trace trace(design_spaces[design_space_index]->insts, {});
      if (ffi::Optional<Schedule> sch = pp.Apply(mod, trace, rand_state)) {
        result = sch.value();
      }
    };
    support::parallel_for_dynamic(0, num, self->ctx_->num_threads, f_proc_unmeasured);
    bool found_new = false;
    for (int i = 0; i < num; i++) {
      if (results[i].has_value()) {
        found_new = true;
        out_schs.push_back(results[i].value());
      }
    }
    fail_count += !found_new;
    TVM_PY_LOG(INFO, self->ctx_->logger) << "Sample-Init-Population summary:\n"
                                         << pp.SummarizeFailures();
    if (pp.TraceFailCount() > 0) {
      TVM_PY_LOG(WARNING, self->ctx_->logger)
          << "SampleInitPopulation encountered " << pp.TraceFailCount()
          << " trace replay failure(s); invalid candidates were skipped";
    }
  }
  return out_schs;
}

std::vector<Schedule> EvolutionarySearchObj::State::EvolveWithCostModel(
    std::vector<Schedule> population, int num) {
  IRModuleSet exists(database_->GetModuleEquality());
  {
    auto _ = Profiler::TimedScope("EvoSearch/Evolve/Misc/CopyMeasuredWorkloads");
    TVM_FFI_ICHECK_GT(num, 0);
    // The heap to record best schedule, we do not consider schedules that are already measured
    exists = this->measured_workloads_;
  }
  SizedHeap heap(num);
  for (int iter = 0;; ++iter) {
    // Predict normalized score with the cost model,
    std::vector<double> scores =
        PredictNormalizedScore(population, ffi::GetRef<TuneContext>(self->ctx_), this->cost_model_);

    {
      auto _ = Profiler::TimedScope("EvoSearch/Evolve/Misc");
      TVM_FFI_ICHECK_EQ(scores.size(), population.size());
      for (int i = 0, n = population.size(); i < n; ++i) {
        Schedule sch = population.at(i);
        IRModule mod = sch->mod();
        size_t shash = ModuleHash(mod);
        double score = scores.at(i);
        if (!exists.Has(mod, shash)) {
          exists.Add(mod, shash);
          heap.Push(sch, score);
        }
      }
      // Discontinue once it reaches end of search
      if (iter == self->genetic_num_iters) {
        break;
      }
      // Set threaded samplers, with probability from predicated normalized throughput
      for (PerThreadData& data : this->per_thread_data_) {
        data.Set(scores, self->genetic_mutate_prob, self->mutator_probs_);
      }
    }
    {
      auto _ = Profiler::TimedScope("EvoSearch/Evolve/Mutation");
      ThreadedTraceApply pp(self->postprocs_);
      ConcurrentBitmask cbmask(self->population_size);
      std::vector<ffi::Optional<Schedule>> next_population(self->population_size);
      // The worker function
      auto f_find_candidate = [&cbmask, &population, &next_population, &pp, this](int thread_id,
                                                                                  int trace_id) {
        // Prepare samplers
        PerThreadData& data = this->per_thread_data_.at(thread_id);
        TRandState* rand_state = &data.rand_state;
        const IRModule& mod = data.mod;
        std::function<int()>& trace_sampler = data.trace_sampler;
        std::function<ffi::Optional<Mutator>()>& mutator_sampler = data.mutator_sampler;
        ffi::Optional<Schedule>& result = next_population.at(trace_id);
        int sampled_trace_id = -1;
        // Loop until success
        for (int fail_count = 0; fail_count <= self->genetic_max_fail_count; ++fail_count) {
          sampled_trace_id = trace_sampler();
          sampled_trace_id = sampled_trace_id % self->population_size;
          s_tir::Trace trace = population.at(sampled_trace_id)->trace().value();
          if (ffi::Optional<Mutator> opt_mutator = mutator_sampler()) {
            // Decision: mutate
            Mutator mutator = opt_mutator.value();
            if (ffi::Optional<s_tir::Trace> new_trace = mutator->Apply(trace, rand_state)) {
              if (ffi::Optional<Schedule> sch = pp.Apply(mod, new_trace.value(), rand_state)) {
                // note that sch's trace is different from new_trace
                // because it contains post-processing information
                result = sch.value();
                break;
              }
            }
          } else if (cbmask.QueryAndMark(sampled_trace_id)) {
            // Decision: do not mutate
            break;
          }
        }
        // if retry count exceeds the limit, reuse an old sample
        if (!result.has_value()) {
          result = population.at(sampled_trace_id);
        }
      };
      support::parallel_for_dynamic(0, self->population_size, self->ctx_->num_threads,
                                    f_find_candidate);

      population.clear();
      for (const auto& sch : next_population) {
        population.push_back(sch.value());
      }
      TVM_PY_LOG(INFO, self->ctx_->logger) << "Evolve iter #" << iter << " done. Summary:\n"
                                           << pp.SummarizeFailures();
    }
  }
  // Return the best states from the heap, sorting from higher score to lower ones
  {
    auto _ = Profiler::TimedScope("EvoSearch/Evolve/Misc");
    std::sort(heap.heap.begin(), heap.heap.end());
    std::vector<Schedule> results;
    results.reserve(num);
    for (const SizedHeap::Item& item : heap.heap) {
      results.push_back(item.sch);
    }

    constexpr int kNumScoresPerLine = 16;
    std::ostringstream os;
    int n = heap.heap.size();
    for (int st = 0; st < n; st += kNumScoresPerLine) {
      os << std::endl;
      int ed = std::min(st + kNumScoresPerLine, n);
      os << "[" << (st + 1) << " : " << ed << "]:\t";
      for (int i = st; i < ed; ++i) {
        if (i != st) {
          os << "  ";
        }
        os << std::fixed << std::setprecision(4) << heap.heap.at(i).score;
      }
    }
    TVM_PY_LOG(INFO, self->ctx_->logger)
        << "Scores of the best " << n << " candidates:" << os.str();
    return results;
  }
}

std::vector<Schedule> EvolutionarySearchObj::State::PickWithEpsGreedy(
    const std::vector<Schedule>& unmeasured, const std::vector<Schedule>& bests, int num) {
  auto _ = Profiler::TimedScope("EvoSearch/PickWithEpsGreedy");
  int num_rands = num * self->eps_greedy;
  int num_bests = num - num_rands;
  std::vector<int> rands =
      s_tir::SampleWithoutReplacement(&self->rand_state_, unmeasured.size(), unmeasured.size());
  std::vector<Schedule> results;
  results.reserve(num);
  IRModuleSet& measured_workloads = this->measured_workloads_;
  for (int i = 0, i_bests = 0, i_rands = 0; i < num; ++i) {
    bool has_best = i_bests < static_cast<int>(bests.size());
    bool has_rand = i_rands < static_cast<int>(rands.size());
    // Pick a schedule
    Schedule sch{ffi::UnsafeInit{}};
    // If needs `bests`, then prefer `bests`
    if (i < num_bests) {
      if (has_best) {
        sch = bests[i_bests++];
      } else if (has_rand) {
        sch = unmeasured[rands[i_rands++]];
      } else {
        break;
      }
    } else {
      // Else prefer `rands`
      if (has_rand) {
        sch = unmeasured[rands[i_rands++]];
      } else if (has_best) {
        sch = bests[i_bests++];
      } else {
        break;
      }
    }
    IRModule mod = sch->mod();
    size_t shash = ModuleHash(mod);
    if (!measured_workloads.Has(mod, shash)) {
      measured_workloads.Add(mod, shash);
      results.push_back(sch);
    }
  }
  return results;
}

ffi::Optional<ffi::Array<MeasureCandidate>>
EvolutionarySearchObj::State::GenerateMeasureCandidates() {
  if (st >= max_trials) {
    return std::nullopt;
  }
  int sample_num = num_trials_per_iter;
  if (ed > max_trials) {
    sample_num = max_trials - st;
    ed = max_trials;
  }
  TVM_FFI_ICHECK_LT(st, ed);
  int pop = self->population_size;
  std::vector<Schedule> inits;
  inits.reserve(pop);

  TVM_PY_LOG(INFO, self->ctx_->logger) << "Generating candidates......";
  std::vector<Schedule> measured = PickBestFromDatabase(pop * self->init_measured_ratio);
  TVM_PY_LOG(INFO, self->ctx_->logger)
      << "Picked top " << measured.size() << " candidate(s) from database";
  std::vector<Schedule> unmeasured = SampleInitPopulation(pop - measured.size());
  if (static_cast<int>(unmeasured.size()) < self->init_min_unmeasured) {
    TVM_PY_LOG(WARNING, self->ctx_->logger)
        << "Cannot sample enough initial population, evolutionary search failed.";
    return std::nullopt;
  }
  TVM_PY_LOG(INFO, self->ctx_->logger) << "Sampled " << unmeasured.size() << " candidate(s)";
  inits.insert(inits.end(), measured.begin(), measured.end());
  inits.insert(inits.end(), unmeasured.begin(), unmeasured.end());
  std::vector<Schedule> bests = EvolveWithCostModel(inits, sample_num);
  TVM_PY_LOG(INFO, self->ctx_->logger)
      << "Got " << bests.size() << " candidate(s) with evolutionary search";
  std::vector<Schedule> picks = PickWithEpsGreedy(unmeasured, bests, sample_num);
  TVM_PY_LOG(INFO, self->ctx_->logger)
      << "Sending " << picks.size() << " candidates(s) for measurement";
  if (picks.empty()) {
    ++this->num_empty_iters;
    if (this->num_empty_iters >= self->num_empty_iters_before_early_stop) {
      return std::nullopt;
    }
  }
  return AssembleCandidates(picks);
}

void EvolutionarySearchObj::State::NotifyRunnerResults(
    const ffi::Array<MeasureCandidate>& measure_candidates,
    const ffi::Array<RunnerResult>& results) {
  st += results.size();
  ed += results.size();
}

size_t EvolutionarySearchObj::State::ModuleHash(const IRModule& mod) const {
  return database_->GetModuleEquality().Hash(mod);
}

SearchStrategy SearchStrategy::EvolutionarySearch(int population_size,         //
                                                  double init_measured_ratio,  //
                                                  int init_min_unmeasured,     //
                                                  int max_fail_count,          //
                                                  int genetic_num_iters,       //
                                                  double genetic_mutate_prob,  //
                                                  int genetic_max_fail_count,  //
                                                  double eps_greedy) {
  TVM_META_SCHEDULE_CHECK_PROB_RANGE(init_measured_ratio, "Initial measured ratio");
  TVM_META_SCHEDULE_CHECK_PROB_RANGE(genetic_mutate_prob, "Mutation probability");
  TVM_META_SCHEDULE_CHECK_PROB_RANGE(eps_greedy, "Greedy pick probability");
  ffi::ObjectPtr<EvolutionarySearchObj> n = ffi::make_object<EvolutionarySearchObj>();
  n->population_size = population_size;
  n->num_empty_iters_before_early_stop = 5;
  n->init_measured_ratio = init_measured_ratio;
  n->init_min_unmeasured = init_min_unmeasured;
  n->max_fail_count = max_fail_count;
  n->genetic_num_iters = genetic_num_iters;
  n->genetic_max_fail_count = genetic_max_fail_count;
  n->genetic_mutate_prob = genetic_mutate_prob;
  n->eps_greedy = eps_greedy;
  return SearchStrategy(ffi::UnsafeInit{}, n);
}

class EvolutionarySearch : public SearchStrategy {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(EvolutionarySearch, SearchStrategy,
                                                EvolutionarySearchObj);
};

ffi::Array<Schedule> EvolutionarySearchSampleInitPopulation(EvolutionarySearch self, int num) {
  std::vector<Schedule> results = self->state_->SampleInitPopulation(num);
  return ffi::Array<Schedule>(results.begin(), results.end());
}

ffi::Array<Schedule> EvolutionarySearchEvolveWithCostModel(EvolutionarySearch self,
                                                           ffi::Array<Schedule> population,
                                                           int num) {
  ffi::Array<Schedule> result;
  std::vector<Schedule> population_vec =
      std::vector<Schedule>(population.begin(), population.end());
  std::vector<Schedule> schs = self->state_->EvolveWithCostModel(population_vec, num);
  for (Schedule sch : schs) {
    IRModule mod = sch->mod();
    size_t shash = self->state_->ModuleHash(mod);
    if (!self->state_->measured_workloads_.Has(mod, shash)) {
      self->state_->measured_workloads_.Add(mod, shash);
      result.push_back(sch);
    }
  }
  return result;
}

TVM_FFI_STATIC_INIT_BLOCK() { EvolutionarySearchObj::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("s_tir.meta_schedule.SearchStrategyEvolutionarySearch",
           SearchStrategy::EvolutionarySearch)
      .def("s_tir.meta_schedule.SearchStrategyEvolutionarySearchSampleInitPopulation",
           EvolutionarySearchSampleInitPopulation)
      .def("s_tir.meta_schedule.SearchStrategyEvolutionarySearchEvolveWithCostModel",
           EvolutionarySearchEvolveWithCostModel);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
