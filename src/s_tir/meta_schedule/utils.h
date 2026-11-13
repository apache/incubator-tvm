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
#ifndef TVM_S_TIR_META_SCHEDULE_UTILS_H_
#define TVM_S_TIR_META_SCHEDULE_UTILS_H_

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/json.h>
#include <tvm/ffi/extra/serialization.h>
#include <tvm/ffi/optional.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/meta_schedule/arg_info.h>
#include <tvm/s_tir/meta_schedule/builder.h>
#include <tvm/s_tir/meta_schedule/cost_model.h>
#include <tvm/s_tir/meta_schedule/database.h>
#include <tvm/s_tir/meta_schedule/extracted_task.h>
#include <tvm/s_tir/meta_schedule/feature_extractor.h>
#include <tvm/s_tir/meta_schedule/measure_callback.h>
#include <tvm/s_tir/meta_schedule/profiler.h>
#include <tvm/s_tir/meta_schedule/runner.h>
#include <tvm/s_tir/meta_schedule/schedule_rule.h>
#include <tvm/s_tir/meta_schedule/search_strategy.h>
#include <tvm/s_tir/meta_schedule/space_generator.h>
#include <tvm/s_tir/meta_schedule/task_scheduler.h>
#include <tvm/s_tir/meta_schedule/tune_context.h>
#include <tvm/s_tir/schedule/schedule.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/support/bytes_io.h>
#include <tvm/support/io.h>
#include <tvm/support/serializer.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../schedule/primitive.h"
#include "../schedule/utils.h"
#include "../support/array_utils.h"
#include "../support/nd_int_set.h"
#include "../support/parallel_for.h"
#include "../support/table_printer.h"

#define TVM_PY_LOG(logging_level, logger)                                       \
  ::tvm::s_tir::meta_schedule::PyLogMessage(__FILE__, __LINE__, logger,         \
                                            PyLogMessage::Level::logging_level) \
      .stream()
#define TVM_PY_LOG_CLEAR_SCREEN(logging_func) clear_logging(__FILE__, __LINE__, logging_func)

namespace tvm {
namespace s_tir {
namespace meta_schedule {

/*!
 * \brief Class to accumulate an log message on the python side. Do not use directly, instead use
 * TVM_PY_LOG(DEBUG), TVM_PY_LOG(INFO), TVM_PY_LOG(WARNING), TVM_PY_ERROR(ERROR).
 * \sa TVM_PY_LOG
 * \sa TVM_PY_LOG_CLEAR_SCREEN
 */
class PyLogMessage {
 public:
  enum class Level : int32_t {
    CLEAR = -10,
    DEBUG = 10,
    INFO = 20,
    WARNING = 30,
    ERROR = 40,
    // FATAL not included
  };

  explicit PyLogMessage(const char* filename, int lineno, ffi::Function logger, Level logging_level)
      : filename_(filename), lineno_(lineno), logger_(logger), logging_level_(logging_level) {}

  TVM_FFI_NO_INLINE ~PyLogMessage() {
    TVM_FFI_ICHECK(logging_level_ != Level::CLEAR)
        << "Cannot use CLEAR as logging level in TVM_PY_LOG, please use TVM_PY_LOG_CLEAR_SCREEN.";
    if (this->logger_ != nullptr) {
      logger_(static_cast<int>(logging_level_), std::string(filename_), lineno_, stream_.str());
    } else {
      if (logging_level_ == Level::INFO) {
        runtime::detail::LogMessage(filename_, lineno_, TVM_LOG_LEVEL_INFO).stream()
            << stream_.str();
      } else if (logging_level_ == Level::WARNING) {
        runtime::detail::LogMessage(filename_, lineno_, TVM_LOG_LEVEL_WARNING).stream()
            << stream_.str();
      } else if (logging_level_ == Level::ERROR) {
        runtime::detail::LogMessage(filename_, lineno_, TVM_LOG_LEVEL_ERROR).stream()
            << stream_.str();
      } else if (logging_level_ == Level::DEBUG) {
        runtime::detail::LogMessage(filename_, lineno_, TVM_LOG_LEVEL_DEBUG).stream()
            << stream_.str();
      } else {
        runtime::detail::LogFatal(filename_, lineno_).stream() << stream_.str();
      }
    }
  }
  std::ostringstream& stream() { return stream_; }

 private:
  const char* filename_;
  int lineno_;
  std::ostringstream stream_;
  ffi::Function logger_;
  Level logging_level_;
};

/*!
 * \brief Whether the tuning is running on ipython kernel.
 * \return A boolean indicating whether ipython kernel is used.
 */
inline bool using_ipython() {
  bool flag = false;
  const auto f_using_ipython = tvm::ffi::Function::GetGlobal("s_tir.meta_schedule.using_ipython");
  if (f_using_ipython) {
    flag = (*f_using_ipython)().cast<bool>();
  }
  return flag;
}

/*!
 * \brief Print out the performance table interactively in jupyter notebook.
 * \param str The serialized performance table.
 */
inline void print_interactive_table(const ffi::String& data) {
  const auto f_print_interactive_table =
      tvm::ffi::Function::GetGlobal("s_tir.meta_schedule.print_interactive_table");
  TVM_FFI_ICHECK(f_print_interactive_table.has_value())
      << "Cannot find print_interactive_table function in registry.";
  (*f_print_interactive_table)(data);
}

/*!
 * \brief A helper function to clear logging output for ipython kernel and console.
 * \param file The file name.
 * \param lineno The line number.
 * \param logging_func The logging function.
 */
inline void clear_logging(const char* file, int lineno, ffi::Function logging_func) {
  if (const char* env_p = std::getenv("TVM_S_TIR_META_SCHEDULE_CLEAR_SCREEN")) {
    if (std::string(env_p) == "1") {
      if (logging_func.defined() && using_ipython()) {
        logging_func(static_cast<int>(PyLogMessage::Level::CLEAR), file, lineno, "");
      } else {
        // this would clear all logging output in the console
        runtime::detail::LogMessage(file, lineno, TVM_LOG_LEVEL_INFO).stream()
            << "\033c\033[3J\033[2J\033[0m\033[H";
      }
    }
  }
}

/*! \brief The type of the random state */
using TRandState = LinearCongruentialEngine::TRandState;

/*!
 * \brief Parses a json string into a json object.
 * \param json_str The json string.
 * \return The json object
 */
Any JSONLoads(std::string json_str);

/*!
 * \brief Dumps a json object into a json string.
 * \param json_obj The json object.
 * \return The json string
 */
std::string JSONDumps(Any json_obj);

/*!
 * \brief Converts a structural hash code to string
 * \param hash_code The hash code
 * \return The string representation of the hash code
 */
inline ffi::String SHash2Str(Workload::THashCode hash_code) { return std::to_string(hash_code); }

/*!
 * \brief Converts an TVM object to the hex string representation of its structural hash.
 * \param obj The TVM object.
 * \return The hex string representation of the hash code.
 */
inline ffi::String SHash2Hex(const ffi::ObjectRef& obj) {
  std::ostringstream os;
  size_t hash_code = 0;
  if (obj.defined()) {
    hash_code = ffi::StructuralHash()(obj);
  }
  os << "0x" << std::setw(16) << std::setfill('0') << std::hex << hash_code;
  return os.str();
}

/*!
 * \brief Fork a random state into another, i.e. PRNG splitting.
 * The given random state is also mutated.
 * \param rand_state The random state to be forked
 * \return The forked random state
 */
inline LinearCongruentialEngine::TRandState ForkSeed(
    LinearCongruentialEngine::TRandState* rand_state) {
  return LinearCongruentialEngine(rand_state).ForkSeed();
}

/*!
 * \brief Fork a random state into another ones, i.e. PRNG splitting.
 *  The given random state is also mutated.
 * \param rand_state The random state to be forked
 * \param n The number of forks
 * \return The forked random states
 */
inline std::vector<LinearCongruentialEngine::TRandState> ForkSeed(
    LinearCongruentialEngine::TRandState* rand_state, int n) {
  std::vector<LinearCongruentialEngine::TRandState> results;
  results.reserve(n);
  for (int i = 0; i < n; ++i) {
    results.push_back(LinearCongruentialEngine(rand_state).ForkSeed());
  }
  return results;
}

/*!
 * \brief Get deep copy of an IRModule.
 * \param mod The IRModule to make a deep copy.
 * \return The deep copy of the IRModule.
 */
inline IRModule DeepCopyIRModule(IRModule mod) {
  return ffi::FromJSONGraph(ffi::ToJSONGraph(mod)).cast<IRModule>();
}

/*!
 * \brief Concatenate strings
 * \param strs The strings to concatenate
 * \param delim The delimiter
 * \return The concatenated string
 */
inline std::string Concat(const ffi::Array<ffi::String>& strs, const std::string& delim) {
  if (strs.empty()) {
    return "";
  }
  std::ostringstream os;
  os << strs[0];
  for (int i = 1, n = strs.size(); i < n; ++i) {
    os << delim << strs[i];
  }
  return os.str();
}

/*!
 * \brief Get the SBlockRV from a block StmtSRef
 * \param sch The schedule
 * \param block_sref The block StmtSRef
 * \param global_var_name The global variable name
 * \return The SBlockRV
 */
inline s_tir::SBlockRV GetRVFromSRef(const s_tir::Schedule& sch, const tirx::StmtSRef& block_sref,
                                     const ffi::String& global_var_name) {
  const s_tir::SBlockObj* block = TVM_SREF_TO_SBLOCK(block_sref);
  return sch->GetSBlock(block->name_hint, global_var_name);
}

/*!
 * \brief A helper data structure that replays a trace and collects failure counts
 * for each postprocessor
 */
struct ThreadedTraceApply {
  /*! \brief Constructor */
  explicit ThreadedTraceApply(const ffi::Array<Postproc>& postprocs)
      : n_(postprocs.size()), items_(new Item[n_]) {
    for (int i = 0; i < n_; ++i) {
      items_[i].postproc = postprocs[i];
      items_[i].fail_counter = 0;
    }
  }

  /*! \brief Destructor */
  ~ThreadedTraceApply() { delete[] items_; }

  /*!
   * \brief Apply the trace and postprocessors to an IRModule
   * \param mod The IRModule to be applied
   * \param trace The trace to apply to the IRModule
   * \param rand_state The random seed
   * \return The schedule created, or std::nullopt if any postprocessor fails
   */
  ffi::Optional<s_tir::Schedule> Apply(const IRModule& mod, const s_tir::Trace& trace,
                                       TRandState* rand_state) {
    s_tir::Schedule sch{ffi::UnsafeInit{}};
    try {
      sch = s_tir::Schedule::Traced(mod,
                                    /*rand_state=*/ForkSeed(rand_state),
                                    /*debug_mode=*/0,
                                    /*error_render_level=*/
                                    s_tir::ScheduleErrorRenderLevel::kNone);
      trace->ApplyToSchedule(sch, /*remove_postproc=*/true);
      sch->EnterPostproc();
    } catch (const ffi::Error& e) {
      if (s_tir::GetScheduleErrorContext(e) == nullptr) {
        throw;
      }
      TVM_PY_LOG(WARNING, nullptr) << "Trace replay failed with ScheduleError: " << e.what();
      this->trace_fail_counter_++;
      return std::nullopt;
    } catch (const std::exception& e) {
      TVM_PY_LOG(WARNING, nullptr) << "Trace replay failed with exception: " << e.what();
      this->trace_fail_counter_++;
      return std::nullopt;
    }

    for (int i = 0; i < n_; ++i) {
      Item& item = items_[i];
      bool success = true;
      try {
        if (!item.postproc->Apply(sch)) {
          success = false;
        }
      } catch (const ffi::Error& e) {
        if (s_tir::GetScheduleErrorContext(e) == nullptr) {
          throw;
        }
        DLOG(WARNING) << "Postproc #" << i << " failed with ScheduleError: " << e.what();
        success = false;
      } catch (const std::exception& e) {
        DLOG(WARNING) << "Postproc #" << i << " failed with exception: " << e.what();
        success = false;
      }
      if (!success) {
        item.fail_counter++;
        return std::nullopt;
      }
    }
    return sch;
  }

  /*! \brief Returns a string summarizing the failures on each postprocessor */
  std::string SummarizeFailures() const {
    std::ostringstream os;
    os << "Trace replay failures: " << this->trace_fail_counter_.load() << " failure(s)";
    if (n_ > 0) {
      os << "\n";
    }
    for (int i = 0; i < n_; ++i) {
      const Item& item = items_[i];
      os << "Postproc #" << i << " [" << item.postproc  //
         << "]: " << item.fail_counter.load() << " failure(s)";
      if (i != n_ - 1) {
        os << "\n";
      }
    }
    return os.str();
  }

  /*! \brief Returns the number of trace replay failures. */
  int TraceFailCount() const { return this->trace_fail_counter_.load(); }

 private:
  /*! \brief A helper data structure that stores the fail count for each postprocessor. */
  struct Item {
    /*! \brief The postprocessor. */
    Postproc postproc{ffi::UnsafeInit()};
    /*! \brief The thread-safe postprocessor failure counter. */
    std::atomic<int> fail_counter{0};
  };

  /*! \brief The number of total postprocessors. */
  int n_;
  /*! \brief The thread-safe trace replay failure counter. */
  std::atomic<int> trace_fail_counter_{0};
  /*! \brief The pointer to the list of postprocessor items. */
  Item* items_;
};

/*!
 * \brief Get the number of cores in CPU
 * \param target The target
 * \return The number of cores.
 */
inline int GetTargetNumCores(const Target& target) {
  int num_cores = target->GetAttr<int64_t>("num-cores").value_or(-1);
  if (num_cores == -1) {
    static const auto f_cpu_count = tvm::ffi::Function::GetGlobal("s_tir.meta_schedule.cpu_count");
    TVM_FFI_CHECK(f_cpu_count.has_value(), ValueError)
        << "Cannot find the packed function \"s_tir.meta_schedule._cpu_count\"";
    num_cores = (*f_cpu_count)(false).cast<int>();
    TVM_FFI_THROW(InternalError)
        << "Target does not have attribute \"num-cores\", physical core number must be "
           "defined! For example, on the local machine, the target must be \"llvm -num-cores "
        << num_cores << "\"";
  }
  return num_cores;
}

/*!
 * \brief Get the median of the running time from RunnerResult in millisecond
 * \param results The results from RunnerResult
 * \return The median of the running time in millisecond
 */
inline double GetRunMsMedian(const RunnerResult& runner_result) {
  ffi::Array<FloatImm> run_secs = runner_result->run_secs.value();
  TVM_FFI_ICHECK(!run_secs.empty());
  std::vector<double> v;
  v.reserve(run_secs.size());
  std::transform(run_secs.begin(), run_secs.end(), std::back_inserter(v),
                 [](const FloatImm& f) -> double { return f->value; });
  std::sort(v.begin(), v.end());
  int n = v.size();
  if (n % 2 == 0) {
    return (v[n / 2 - 1] + v[n / 2]) * 0.5 * 1000.0;
  } else {
    return v[n / 2] * 1000.0;
  }
}

/*!
 * \brief Convert the given object to an array of floating point numbers
 * \param obj The object to be converted
 * \return The array of floating point numbers
 */
inline ffi::Array<FloatImm> AsFloatArray(const ffi::ObjectRef& obj) {
  const ffi::ArrayObj* arr = obj.as<ffi::ArrayObj>();
  TVM_FFI_CHECK(arr, TypeError) << "Expect an array, but gets: " << obj->GetTypeKey();
  ffi::Array<FloatImm> results;
  results.reserve(arr->size());
  for (Any val : *arr) {
    auto float_value = [&]() -> FloatImm {
      if (auto opt_int_imm = val.try_cast<IntImm>()) {
        return FloatImm(PrimType::Float(32), static_cast<double>((*opt_int_imm)->value));
      } else if (auto opt_float_imm = val.try_cast<FloatImm>()) {
        return *std::move(opt_float_imm);
      } else {
        TVM_FFI_THROW(TypeError) << "Expect an array of float or int, but gets: "
                                 << val.GetTypeKey();
        TVM_FFI_UNREACHABLE();
      }
    }();

    results.push_back(float_value);
  }
  return results;
}

/*!
 * \brief Convert the given object to an array of integers
 * \param obj The object to be converted
 * \return The array of integers
 */
inline ffi::Array<int64_t> AsIntArray(const ffi::ObjectRef& obj) {
  const ffi::ArrayObj* arr = obj.as<ffi::ArrayObj>();
  TVM_FFI_CHECK(arr, TypeError) << "Expect an array, but gets: " << obj->GetTypeKey();
  ffi::Array<int64_t> results;
  results.reserve(arr->size());
  for (Any val : *arr) {
    auto int_value = [&]() -> int64_t {
      if (auto opt_int_imm = val.try_cast<IntImm>()) {
        return static_cast<int64_t>((*opt_int_imm)->value);
      } else {
        TVM_FFI_THROW(TypeError) << "Expect an array of integers, but gets: " << val.GetTypeKey();
        TVM_FFI_UNREACHABLE();
      }
    }();
    results.push_back(int_value);
  }
  return results;
}

/*! \brief The struct defining comparison function of sorting by mean run seconds. */
struct SortTuningRecordByMeanRunSecs {
  static const constexpr double kMaxMeanTime = 1e10;

  static double Mean(const ffi::Array<FloatImm>& a) {
    if (a.empty()) {
      return kMaxMeanTime;
    }
    double sum = 0.0;
    for (const FloatImm& i : a) {
      sum += i->value;
    }
    return sum / a.size();
  }

  bool operator()(const TuningRecord& a, const TuningRecord& b) const {
    double a_time = Mean(a->run_secs.value_or({}));
    double b_time = Mean(b->run_secs.value_or({}));
    return a_time < b_time;
  }
};

/*!
 * \brief The helper function to clone schedule rules, postprocessors, and mutators.
 * \param src The source space generator.
 * \param dst The destination space generator.
 */
inline void CloneRules(const SpaceGeneratorObj* src, SpaceGeneratorObj* dst) {
  if (src->sch_rules.has_value()) {
    ffi::Array<ScheduleRule> original = src->sch_rules.value();
    ffi::Array<ScheduleRule> sch_rules;
    sch_rules.reserve(original.size());
    for (const ScheduleRule& sch_rule : original) {
      sch_rules.push_back(sch_rule->Clone());
    }
    dst->sch_rules = std::move(sch_rules);
  }
  if (src->postprocs.has_value()) {
    ffi::Array<Postproc> original = src->postprocs.value();
    ffi::Array<Postproc> postprocs;
    postprocs.reserve(original.size());
    for (const Postproc& postproc : original) {
      postprocs.push_back(postproc->Clone());
    }
    dst->postprocs = std::move(postprocs);
  }
  if (src->mutator_probs.has_value()) {
    ffi::Map<Mutator, FloatImm> original = src->mutator_probs.value();
    ffi::Map<Mutator, FloatImm> mutator_probs;
    for (const auto& kv : original) {
      mutator_probs.Set(kv.first->Clone(), kv.second);
    }
    dst->mutator_probs = std::move(mutator_probs);
  }
}

/*! \brief Returns true if the given target is one of the supported gpu targets. */
inline bool IsGPUTarget(const std::string& target_name) {
  static const std::unordered_set<std::string> gpu_targets{"cuda", "rocm", "vulkan", "metal",
                                                           "opencl"};
  return gpu_targets.count(target_name);
}

/*!
 * \brief Create an AutoInline schedule rule for the given target.
 * \param target_name The name of the target ("llvm", "cuda", etc.)
 * \return The AutoInline schedule rule for the given target.
 */
inline ScheduleRule GetDefaultAutoInline(const std::string& target_name) {
  ffi::Array<ScheduleRule> rules;
  if (target_name == "llvm") {
    rules = ScheduleRule::DefaultLLVM();
  } else if (target_name == "hexagon") {
    rules = ScheduleRule::DefaultHexagon();
  } else if (IsGPUTarget(target_name)) {
    rules = ScheduleRule::DefaultCUDA();
  } else {
    TVM_FFI_THROW(ValueError) << "Unsupported target: " << target_name;
  }
  for (const ScheduleRule& rule : rules) {
    if (rule->GetTypeKey() == "s_tir.meta_schedule.AutoInline") {
      return rule;
    }
  }
  TVM_FFI_THROW(ValueError) << "AutoInline rule is not found in the default rules for target: "
                            << target_name;
  throw;
}

/*!
 * \brief Summarize the run time of the given FloatImm array.
 * \param arr The array of FloatImm.
 * \return The summary of the values in the given array.
 */
inline double Sum(const ffi::Array<FloatImm>& arr) {
  double sum = 0;
  for (const FloatImm& f : arr) {
    sum += f->value;
  }
  return sum;
}

/*! \brief Collecting all the blocks */
class SBlockCollector : public s_tir::StmtExprVisitor {
 public:
  using s_tir::StmtExprVisitor::Visit_;

  ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
    if (value.as<ExprObj>()) return std::nullopt;
    return s_tir::StmtExprVisitor::Visit(value);
  }

  static ffi::Array<s_tir::SBlockRV> Collect(const s_tir::Schedule& sch,
                                             const ffi::Function f_block_filter = nullptr) {  //
    return ffi::make_object<SBlockCollector>(sch, f_block_filter)->Run();
  }

 private:
  /*! \brief Entry point */
  ffi::Array<s_tir::SBlockRV> Run() {
    std::vector<s_tir::SBlockRV> results;
    auto f_collect = [this, &results](tirx::Function func, ffi::String func_name) {
      func_name_ = func_name;
      block_names_.clear();
      blocks_to_collect_.clear();
      Visit(func->body);
      for (const ffi::String& name : blocks_to_collect_) {
        results.push_back(sch_->GetSBlock(name, func_name_));
      }
    };

    if (sch_->func_working_on().has_value()) {
      GlobalVar gv = sch_->func_working_on().value();
      tirx::Function func = sch_->mod()->functions[gv].as_or_throw<tirx::Function>();
      f_collect(func, gv->name_hint);
    } else {
      for (const auto& [gv, base_func] : sch_->mod()->functions) {
        // `gv->name_hint` is the name of the function
        // `base_func` can be Function or relax::Function
        if (const auto* func = base_func.as<tirx::FunctionObj>()) {
          f_collect(ffi::GetRef<tirx::Function>(func), gv->name_hint);
        }
      }
    }
    return results;
  }
  /*! \brief Constructor */
 public:
  explicit SBlockCollector(const s_tir::Schedule& sch, const ffi::Function f_block_filter = nullptr)
      : sch_(sch), f_block_filter_(f_block_filter) {}

 private:
  /*! \brief Override the Stmt visiting behaviour */
  ffi::Optional<VisitInterrupt> Visit_(const s_tir::SBlockObj* block) override {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(s_tir::StmtExprVisitor::Visit_(block));
    TVM_FFI_ICHECK(block_names_.count(block->name_hint) == 0)
        << "Duplicated block name " << block->name_hint << " in function " << func_name_
        << " not supported!";
    block_names_.insert(block->name_hint);

    // If filter function is provided, use it to selectively collect blocks.
    // Otherwise collect all blocks.
    bool collect_block = true;
    if (f_block_filter_ != nullptr) {
      collect_block = f_block_filter_(ffi::GetRef<s_tir::SBlock>(block)).cast<IntImm>()->value != 0;
    }
    if (collect_block) {
      blocks_to_collect_.push_back(block->name_hint);
    }
    return std::nullopt;
  }

  /*! \brief The schedule to be collected */
  const s_tir::Schedule& sch_;
  /*! \brief An optional packed func that allows only certain blocks to be collected. */
  const ffi::Function f_block_filter_;
  /*! \brief The set of func name and block name pair */
  std::unordered_set<ffi::String> block_names_;
  /* \brief The list of blocks to collect in order */
  ffi::Array<ffi::String> blocks_to_collect_;
  /*! \brief Name of the current Function */
  ffi::String func_name_;
};

void JSONFileAppendLine(const ffi::String& path, const std::string& line);
std::vector<Any> JSONFileReadLines(const ffi::String& path, int num_threads, bool allow_missing);
}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm

#endif  // TVM_S_TIR_META_SCHEDULE_UTILS_H_
