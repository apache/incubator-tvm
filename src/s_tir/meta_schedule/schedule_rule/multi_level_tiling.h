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
#ifndef TVM_S_TIR_META_SCHEDULE_SCHEDULE_RULE_MULTI_LEVEL_TILING_H_
#define TVM_S_TIR_META_SCHEDULE_SCHEDULE_RULE_MULTI_LEVEL_TILING_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/s_tir/meta_schedule/schedule_rule.h>
#include <tvm/s_tir/schedule/schedule.h>

#include <unordered_map>
#include <utility>
#include <vector>

#include "../../support/array_utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;
/*!
 * \brief Get the tensor dimensions for all the read tensors of a block, but marks the reduction
 * tensors' dimensions as -1
 * \param block_sref The block to be processed
 * \return The tensor dimensions for all the read tensors of a block, except for reduction tensors
 * \note The method is not designed for generic analysis and relies on assumptions in the scenario
 * of multi-level tiling, so it's intentionally kept inside this file not in the analysis header
 */
std::vector<int> GetReadTensorNDims(const StmtSRef& block_sref);

}  // namespace s_tir
}  // namespace tvm

namespace tvm {
namespace s_tir {
namespace meta_schedule {

/*!
 * \brief Configuration of data reuse type:
 * 0) kNoReuse: no reuse is allowed, then no cache_read/write is performed.
 * 1) kMayReuse: reuse is allowed, but no reuse is explored.
 * 2) kMustReuse: reuse is allowed and no reuse is not explored.
 */
enum class ReuseType : int32_t {
  kNoReuse = 0,
  kMayReuse = 1,
  kMustReuse = 2,
};

/*!
 * \brief Converts a string to ReuseType.
 * \param str The string to be converted.
 * \return The converted ReuseType.
 */
inline ReuseType Str2ReuseType(const ffi::String& str) {
  if (str == "no") {
    return ReuseType::kNoReuse;
  } else if (str == "may") {
    return ReuseType::kMayReuse;
  } else if (str == "must") {
    return ReuseType::kMustReuse;
  } else {
    TVM_FFI_THROW(ValueError) << "Unknown ReuseType: " << str;
    throw;
  }
}

/*! \brief Configuration of data reuse patterns */
struct ReuseConfig {
  /*! \brief Type of data reuse: no-reuse, may-reuse or must-reuse */
  ReuseType req;
  /*! \brief Which levels are caching stage inserted at */
  std::vector<int> levels;
  /*! \brief The storage scope */
  ffi::String scope;

  /*! \brief Default constructor: no data reuse */
  ReuseConfig() : req(ReuseType::kNoReuse) {}

  /*! \brief Construct from a configuration dictionary */
  explicit ReuseConfig(const ffi::Map<ffi::String, ffi::Any>& config)
      : req(Str2ReuseType(config.at("req").as_or_throw<ffi::String>())),
        levels([&]() {
          auto arr = config.at("levels").as_or_throw<ffi::Array<int64_t>>();
          std::vector<int> r;
          r.reserve(arr.size());
          for (int64_t v : arr) r.push_back(static_cast<int>(v));
          return r;
        }()),
        scope(config.at("scope").as_or_throw<ffi::String>()) {
    TVM_FFI_ICHECK_EQ(config.size(), 3);
  }
};

// Forware declaration
class State;

/*! \brief The state of auto scheduling for the multi-level tiling rule */
class StateObj : public ffi::Object {
 public:
  explicit StateObj(s_tir::Schedule sch) : sch(std::move(sch)) {}
  /*! \brief The schedule to date */
  s_tir::Schedule sch;
  /*! \brief The block to be tiled */
  s_tir::SBlockRV block_rv;
  /*! \brief The loop tiles */
  ffi::Array<ffi::Array<s_tir::LoopRV>> tiles;
  /*! \brief The factors of the loop tiles. */
  ffi::Array<ffi::Array<ffi::Optional<s_tir::ExprRV>>> tile_factors;
  /*! \brief The mapping from tensor index to read cache block. */
  std::unordered_map<int, s_tir::SBlockRV> read_reuse;
  /*! \brief The mapping from tensor index to write cache block. */
  std::unordered_map<int, s_tir::SBlockRV> write_reuse;

  /*!
   * \brief Create a copy of the state. The underlying schedule is copied. Schedule rules that
   * produce multiple states should use this method to create new states.
   */
  virtual State Copy() const;

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("s_tir.meta_schedule.State", StateObj, ffi::Object);
};

/*! \brief Managed reference to StateObj */
class State : public ffi::ObjectRef {
 public:
  explicit State(ffi::UnsafeInit tag, ffi::ObjectPtr<StateObj> data) : ffi::ObjectRef(tag) {
    data_ = std::move(data);
  }

  /*! \brief Default constructor */
  explicit State(s_tir::Schedule sch, s_tir::SBlockRV block_rv,
                 ffi::Array<ffi::Array<s_tir::LoopRV>> tiles = {});
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(State, ffi::ObjectRef, StateObj);
};

/*!
 * \brief Helper to apply a sub-rule to a list of auto scheduling states
 * \tparam FLambda The type of the sub-rule functor
 * \param states The list of states to be applied
 * \return The list of states after applying the sub-rule
 */
template <class FLambda>
std::vector<State> SubRule(std::vector<State> states, FLambda sub_rule) {
  std::vector<State> results;
  for (auto&& state : states) {
    std::vector<State> next = sub_rule(std::move(state));
    results.insert(results.end(),                          //
                   std::make_move_iterator(next.begin()),  //
                   std::make_move_iterator(next.end()));
  }
  return results;
}

/*!
 * \brief The mega rule: multi-level tiling with data reuse
 */
class MultiLevelTilingObj : public ScheduleRuleObj {
 public:
  virtual ~MultiLevelTilingObj() = default;

  // SubRule 1. add write cache
  std::vector<State> AddWriteReuse(State state) const;
  // SubRule 2. tile the loop nest
  std::vector<State> TileLoopNest(State state, int tile_inner_most_space_loop_num = -1) const;
  // SubRule 3. add read cache
  std::vector<State> AddReadReuse(State state) const;
  // SubRule 4. add async pipeline
  std::vector<State> AddAsyncPipeline(State state) const;

  // Do nothing; Inherited from ScheduleRuleObj
  void InitializeWithTuneContext(const TuneContext& context) final;

  // Entry of the mega rule; Inherited from ScheduleRuleObj
  ffi::Array<s_tir::Schedule> Apply(const s_tir::Schedule& sch,
                                    const s_tir::SBlockRV& block_rv) override;

  // Inherited from ScheduleRuleObj
  ScheduleRule Clone() const override;

 protected:
  virtual std::vector<State> ApplySubRules(std::vector<State> states);

  virtual std::pair<ffi::Array<ffi::Optional<s_tir::ExprRV>>, ffi::Array<s_tir::LoopRV>> SplitLoop(
      const s_tir::Schedule& sch, s_tir::SBlockRV block, s_tir::LoopRV loop, int n_tiles) const;

  // Annotate a block to use cooperative fetching
  void AnnotateCooperativeFetching(s_tir::Schedule* sch, const s_tir::SBlockRV& block) const;

 public:
  /*!
   * \brief The tiling structure. Recommended:
   * - 'SSRSRS' on CPU
   * - 'SSSRRSRS' on GPU
   */
  ffi::String structure;
  /*! \brief For each level of tiles, which thread axis it is bound to */
  ffi::Array<ffi::String> tile_binds;
  /*! \brief The maximum size of the innermost factor */
  int max_innermost_factor;
  /*! \brief The length of vector lane in vectorized cooperative fetching */
  std::vector<int> vector_load_lens;
  /*! \brief Data reuse configuration for reading */
  ReuseConfig reuse_read_;
  /*! \brief Data reuse configuration for writing */
  ReuseConfig reuse_write_;
  /*! \brief The indices of spatial tiles in `structure` */
  std::vector<int> s_indices_;
  /*! \brief The indices of reduction tiles in `structure` */
  std::vector<int> r_indices_;
  /*! \brief The size of the thread warp */
  int thread_warp_size_;
  /*! \brief The maximum number of threads to be used size of a thread warp */
  int max_threads_per_block_;
  /*! \brief All available async pipeline stages. */
  std::vector<int> stages;
  /*! \brief The logging function */
  ffi::Function logger;
  /*! \brief The function to overwrite the default condition for applying MultiLevelTiling. */
  ffi::Optional<ffi::Function> filter_fn_;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<MultiLevelTilingObj>()
        .def_ro("structure", &MultiLevelTilingObj::structure)
        .def_ro("tile_binds", &MultiLevelTilingObj::tile_binds)
        .def_ro("max_innermost_factor", &MultiLevelTilingObj::max_innermost_factor);
  }
  TVM_FFI_DECLARE_OBJECT_INFO("s_tir.meta_schedule.MultiLevelTiling", MultiLevelTilingObj,
                              ScheduleRuleObj);
};

template <typename NodeType>
ffi::ObjectPtr<NodeType> MultiLevelTilingInitCommon(
    ffi::String structure, ffi::Optional<ffi::Array<ffi::String>> tile_binds,
    ffi::Optional<int64_t> max_innermost_factor,
    ffi::Optional<ffi::Array<int64_t>> vector_load_lens,
    ffi::Optional<ffi::Map<ffi::String, ffi::Any>> reuse_read,
    ffi::Optional<ffi::Map<ffi::String, ffi::Any>> reuse_write) {
  ffi::ObjectPtr<NodeType> n = ffi::make_object<NodeType>();
  n->structure = structure;
  n->tile_binds = tile_binds.value_or({});
  n->max_innermost_factor = max_innermost_factor.value_or(-1);
  n->vector_load_lens = [&]() {
    if (!vector_load_lens.has_value()) return std::vector<int>();
    auto arr = vector_load_lens.value();
    std::vector<int> r;
    r.reserve(arr.size());
    for (int64_t v : arr) r.push_back(static_cast<int>(v));
    return r;
  }();
  n->reuse_read_ = reuse_read.has_value() ? ReuseConfig(reuse_read.value()) : ReuseConfig();
  n->reuse_write_ = reuse_write.has_value() ? ReuseConfig(reuse_write.value()) : ReuseConfig();
  for (int i = 0, len = structure.size(); i < len; ++i) {
    char c = structure.data()[i];
    if (c == 'S') {
      n->s_indices_.push_back(i);
    } else if (c == 'R') {
      n->r_indices_.push_back(i);
    } else {
      TVM_FFI_THROW(ValueError) << "Invalid tiling structure: " << structure;
    }
  }
  n->thread_warp_size_ = -1;
  n->max_threads_per_block_ = -1;
  return n;
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm

#endif  // TVM_S_TIR_META_SCHEDULE_SCHEDULE_RULE_MULTI_LEVEL_TILING_H_
