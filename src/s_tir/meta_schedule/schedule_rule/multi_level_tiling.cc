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
#include "./multi_level_tiling.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/meta_schedule/schedule_rule.h>
#include <tvm/s_tir/stmt.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "../utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

std::vector<int> GetReadTensorNDims(const StmtSRef& block_sref) {
  const SBlockObj* block = TVM_SREF_TO_SBLOCK(block_sref);
  const VarObj* write_tensor = block->writes[0]->source.as_or_throw<tvm::tirx::TensorVar>().get();
  int n = block->reads.size();
  std::vector<int> results(n, -1);
  for (int i = 0; i < n; ++i) {
    const VarObj* read_tensor = block->reads[i]->source.as_or_throw<tvm::tirx::TensorVar>().get();
    if (read_tensor != write_tensor) {
      results[i] = ffi::GetRef<Var>(read_tensor).as_or_throw<TensorVar>().shape().size();
    }
  }
  return results;
}

}  // namespace s_tir
}  // namespace tvm

namespace tvm {
namespace s_tir {
namespace meta_schedule {

using s_tir::GetSBlockVarTypes;
using s_tir::IsWriteCache;
using s_tir::IterVarType;
using s_tir::LoopRV;
using s_tir::SBlockRV;
using s_tir::Schedule;

TVM_FFI_STATIC_INIT_BLOCK() { MultiLevelTilingObj::RegisterReflection(); }

State::State(s_tir::Schedule sch, s_tir::SBlockRV block_rv,
             ffi::Array<ffi::Array<s_tir::LoopRV>> tiles) {
  ffi::ObjectPtr<StateObj> node = ffi::make_object<StateObj>(std::move(sch));
  node->block_rv = std::move(block_rv);
  node->tiles = std::move(tiles);
  data_ = std::move(node);
}

State StateObj::Copy() const {
  ffi::ObjectPtr<StateObj> node = ffi::make_object<StateObj>(*this);
  node->sch = sch->Copy();
  return State(ffi::UnsafeInit{}, node);
}

// Do nothing; Inherited from ScheduleRuleObj
void MultiLevelTilingObj::InitializeWithTuneContext(const TuneContext& context) {
  if (ffi::Optional<int64_t> v =
          context->target.value()->GetAttr<int64_t>("max_threads_per_block")) {
    this->max_threads_per_block_ = v.value();
    if (ffi::Optional<int64_t> v = context->target.value()->GetAttr<int64_t>("thread_warp_size")) {
      this->thread_warp_size_ = v.value();
    } else {
      TVM_PY_LOG(INFO, context->logger) << "'thread_warp_size' is not defined in the target";
    }
  }
  if (ffi::Optional<ffi::String> opt_sm = context->target.value()->GetAttr<ffi::String>("arch")) {
    std::string sm = opt_sm.value();
    if (sm.starts_with("sm_")) {
      sm = sm.substr(3);
      try {
        // only sm_80 or higher supports async memcopy
        if (std::stoi(sm) >= 80) {
          // only stage = 4 & 5 is tested. all integer that is bigger than 2
          // is theoretically feasible, but no guarantee for great performance.
          this->stages = {4, 5};
        }
      } catch (const std::invalid_argument& e) {
        LOG(WARNING) << "ValueError: Unable to parse `target.arch`: " << sm
                     << ". Details: " << e.what();
      }
    }
  }
  logger = context->logger;
}

// Entry of the mega rule; Inherited from ScheduleRuleObj
ffi::Array<Schedule> MultiLevelTilingObj::Apply(const Schedule& sch, const SBlockRV& block_rv) {
  if ((filter_fn_ && filter_fn_.value()(sch, sch->GetSRef(block_rv)).cast<bool>()) ||
      NeedsMultiLevelTiling(sch->state(), sch->GetSRef(block_rv))) {
    sch->Annotate(block_rv, tvm::s_tir::attr::kMetaScheduleTilingStructure.name, structure);

    ffi::Array<Schedule> results;
    for (auto&& state : ApplySubRules({State(sch, block_rv)})) {
      results.push_back(std::move(state->sch));
    }
    return results;
  }
  return {sch};
}

// Inherited from ScheduleRuleObj
ScheduleRule MultiLevelTilingObj::Clone() const {
  ffi::ObjectPtr<MultiLevelTilingObj> n = ffi::make_object<MultiLevelTilingObj>(*this);
  return ScheduleRule(ffi::UnsafeInit{}, n);
}

std::vector<State> MultiLevelTilingObj::ApplySubRules(std::vector<State> states) {
  states = SubRule(std::move(states), [&](State state) { return TileLoopNest(std::move(state)); });
  states = SubRule(std::move(states), [&](State state) { return AddWriteReuse(std::move(state)); });
  states = SubRule(std::move(states), [&](State state) { return AddReadReuse(std::move(state)); });
  states =
      SubRule(std::move(states), [&](State state) { return AddAsyncPipeline(std::move(state)); });
  return states;
}

std::vector<State> MultiLevelTilingObj::AddWriteReuse(State state) const {
  const ReuseConfig& config = this->reuse_write_;
  if (config.req == ReuseType::kNoReuse) {
    return {std::move(state)};
  }
  std::vector<int> levels = config.levels;
  ReuseType req = config.req;
  if (ffi::Optional<ffi::Array<int64_t>> ann =
          s_tir::GetAnn<ffi::Array<int64_t>>(state->sch->GetSRef(state->block_rv),
                                             tvm::s_tir::attr::kMetaScheduleWriteCacheLevel.name)) {
    req = ReuseType::kMustReuse;
    levels.clear();
    std::transform(ann.value().begin(), ann.value().end(), std::back_inserter(levels),
                   [](int64_t v) { return static_cast<int>(v); });
  }
  std::vector<State> results;
  if (req == ReuseType::kMayReuse) {
    // Case 1. If the write cache is already there, we don't need to add another.
    ffi::Array<SBlockRV> consumer_rvs = state->sch->GetConsumers(state->block_rv);
    if (consumer_rvs.size() == 1 && IsWriteCache(state->sch->GetSRef(consumer_rvs[0]))) {
      for (int level : levels) {
        State new_state = state->Copy();
        const LoopRV& loop_rv = new_state->tiles[level - 1].back();
        new_state->sch->ReverseComputeAt(consumer_rvs[0], loop_rv, true);
        results.push_back(std::move(new_state));
      }
      state->write_reuse.emplace(0, consumer_rvs[0]);
      results.push_back(state);
      return results;
    } else {
      // Case 2. No write cache is added
      State new_state = state->Copy();
      results.emplace_back(std::move(new_state));
    }
  }

  // Case 3. Add one write cache
  SBlockRV write_cache =
      state->sch->CacheWrite(/*block_rv=*/state->block_rv, /*read_tensor_index=*/0,
                             /*storage_scope=*/config.scope);
  state->write_reuse.emplace(0, write_cache);
  for (int level : levels) {
    State new_state = state->Copy();
    const LoopRV& loop_rv = new_state->tiles[level - 1].back();
    new_state->sch->ReverseComputeAt(write_cache, loop_rv, true);
    results.push_back(std::move(new_state));
  }
  return results;
}

std::pair<ffi::Array<ffi::Optional<s_tir::ExprRV>>, ffi::Array<s_tir::LoopRV>>
MultiLevelTilingObj::SplitLoop(const Schedule& sch, SBlockRV block, LoopRV loop,
                               int n_tiles) const {
  ffi::Array<ffi::Optional<s_tir::ExprRV>> factors = sch->SamplePerfectTile(
      /*loop=*/loop,
      /*n=*/n_tiles,
      /*max_innermost_factor=*/max_innermost_factor);
  ffi::Array<s_tir::LoopRV> splits = sch->Split(/*loop=*/loop,
                                                /*factors=*/factors);
  return {factors, splits};
}

std::vector<State> MultiLevelTilingObj::TileLoopNest(State state,
                                                     int tile_inner_most_space_loop_num) const {
  Schedule& sch = state->sch;
  const SBlockRV& block_rv = state->block_rv;
  // Step 1. Assuming trivial binding, pair the loops and their iter-var-types
  ffi::Array<LoopRV> loops = sch->GetLoops(block_rv);
  std::vector<IterVarType> iter_types = GetSBlockVarTypes(sch->GetSRef(state->block_rv));
  TVM_FFI_ICHECK_EQ(loops.size(), iter_types.size());
  // Step 2. For each loop axis, tile it
  int64_t spatial_loop_product = 1;

  int total_spatial_loop_num = 0;
  std::for_each(iter_types.begin(), iter_types.end(), [&](const auto& iter_type) {
    if (iter_type == IterVarType::kDataPar) total_spatial_loop_num++;
  });
  TVM_FFI_ICHECK_GE(total_spatial_loop_num, tile_inner_most_space_loop_num);
  if (tile_inner_most_space_loop_num < 0) tile_inner_most_space_loop_num = total_spatial_loop_num;
  int outer_most_spatial_loop_skipped_num = total_spatial_loop_num - tile_inner_most_space_loop_num;

  ffi::Array<LoopRV> skipped_outer_spatial_loops;
  std::vector<ffi::Array<LoopRV>> tiles(s_indices_.size() + r_indices_.size());
  state->tile_factors.resize(tiles.size());
  std::vector<ffi::Array<ffi::Optional<s_tir::ExprRV>>> tile_factors;
  tile_factors.resize(tiles.size());
  for (int i = 0, n = loops.size(); i < n; ++i) {
    LoopRV loop = loops[i];
    const std::vector<int>* idx = nullptr;

    if (iter_types[i] == IterVarType::kDataPar) {
      if (outer_most_spatial_loop_skipped_num > 0) {
        skipped_outer_spatial_loops.push_back(loop);
        outer_most_spatial_loop_skipped_num--;
        continue;
      }
      idx = &s_indices_;
      if (spatial_loop_product != -1) {
        PrimExpr extent_imm_expr =
            Range::GetExtentFromBeginEnd(sch->Get(loop)->begin, sch->Get(loop)->end);
        const auto* extent_imm = extent_imm_expr.as<IntImmObj>();
        if (auto extent = extent_imm ? extent_imm->value.as<int64_t>() : std::nullopt;
            extent.has_value()) {
          spatial_loop_product *= *extent;
        } else {
          spatial_loop_product = -1;
        }
      }
    } else if (iter_types[i] == IterVarType::kCommReduce) {
      idx = &r_indices_;
    } else {
      continue;
    }

    const int n_tiles = idx->size();

    if (n_tiles == 1) {
      tiles[idx->at(0)].push_back(loop);
    } else {
      auto [factors, splits] = SplitLoop(sch, block_rv, loop, n_tiles);

      // Put every tile to its slot
      for (int j = 0; j < n_tiles; ++j) {
        tiles[idx->at(j)].push_back(splits[j]);
        tile_factors[idx->at(j)].push_back(factors[j]);
      }
    }
  }
  state->tile_factors = std::move(tile_factors);
  // Step 3. Reorder to organize the tiles
  sch->Reorder(support::ConcatArrayList<LoopRV>(tiles.begin(), tiles.end()));
  // Step 4. Bind the tiles to threads
  int n_binds = std::min(tile_binds.size(), tiles.size());
  if (skipped_outer_spatial_loops.size() && n_binds) {
    auto& the_first_tile = tiles[0];
    the_first_tile.insert(the_first_tile.begin(), skipped_outer_spatial_loops.begin(),
                          skipped_outer_spatial_loops.end());
  }
  for (int i = 0; i < n_binds; ++i) {
    LoopRV fused = sch->Fuse(tiles[i]);
    sch->Bind(fused, tile_binds[i]);
    tiles[i] = {fused};
  }
  state->tiles = ffi::Array<ffi::Array<LoopRV>>{tiles.begin(), tiles.end()};
  if (this->thread_warp_size_ != -1) {
    int64_t low_inclusive = 1;
    int64_t high_inclusive = this->max_threads_per_block_;
    if (spatial_loop_product > 2 * this->thread_warp_size_) {
      low_inclusive = this->thread_warp_size_;
    }
    sch->Annotate(block_rv, tvm::s_tir::attr::kMetaScheduleThreadExtentLowInclusive.name,
                  low_inclusive);
    sch->Annotate(block_rv, tvm::s_tir::attr::kMetaScheduleThreadExtentHighInclusive.name,
                  high_inclusive);
  }
  return {state};
}

std::vector<State> MultiLevelTilingObj::AddReadReuse(State state) const {
  const ReuseConfig& config = this->reuse_read_;
  if (config.req == ReuseType::kNoReuse) {
    return {std::move(state)};
  }
  TVM_FFI_ICHECK(config.req != ReuseType::kMayReuse);
  const SBlockRV& block_rv = state->block_rv;
  std::vector<State> results;
  results.reserve(config.levels.size());
  for (int level : config.levels) {
    State new_state = state->Copy();
    Schedule& sch = new_state->sch;
    const LoopRV& loop_rv = state->tiles[level - 1].back();
    // Enumerate all tensors that are read but not written
    std::vector<int> read_tensor_ndims = s_tir::GetReadTensorNDims(sch->GetSRef(block_rv));
    for (int i = 0, n_reads = read_tensor_ndims.size(); i < n_reads; ++i) {
      int tensor_ndim = read_tensor_ndims[i];
      if (tensor_ndim == -1) {
        continue;
      }
      // Do cache_read
      SBlockRV cache_read_block = sch->CacheRead(block_rv, i, config.scope, {block_rv});
      // Insert cache_read block to the proper place
      sch->ComputeAt(cache_read_block, loop_rv, true);
      // Fuse the iterators of the cache_read
      ffi::Array<LoopRV> tensor_loops = sch->GetLoops(cache_read_block);
      sch->Fuse(ffi::Array<LoopRV>{tensor_loops.end() - tensor_ndim,  //
                                   tensor_loops.end()});
      AnnotateCooperativeFetching(&sch, cache_read_block);
      new_state->read_reuse.emplace(i, cache_read_block);
    }
    results.push_back(std::move(new_state));
  }
  return results;
}

std::vector<State> MultiLevelTilingObj::AddAsyncPipeline(State state) const {
  // For arch that does not support async pipeline, this->stages will be an empty vector
  if (r_indices_.size() < 1 || this->stages.empty()) {
    return {state};
  }
  // Current only support default config used by ScheduleRule::DefaultCUDA
  // @see src/meta_schedule/schedule_rule/schedule_rule.cc
  // check the reduce loop contains exactly 3 for loops
  // therefore it matches the notation array size in the following code
  tirx::StmtSRef r_loop_sref = state->sch->GetSRef(state->tiles[r_indices_[0]].back());
  const ForObj* r_for_loop = TVM_SREF_TO_FOR(r_loop_sref);
  ffi::Array<Stmt> seq = r_for_loop->body->seq;
  if (seq.size() != 3) {
    return {state};
  }
  for (auto& stmt : seq) {
    if (!stmt.as<ForObj>()) {
      return {state};
    }
  }

  std::vector<State> ret;
  ret.push_back(state);
  for (int stage : this->stages) {
    State new_state = state->Copy();
    LoopRV r_loop_fused = new_state->sch->Fuse(new_state->tiles[r_indices_[0]]);
    new_state->sch->Annotate(r_loop_fused, tvm::s_tir::attr::kSoftwarePipelineStage.name,
                             ffi::Array<int64_t>{0, 0, stage - 2});
    new_state->sch->Annotate(r_loop_fused, tvm::s_tir::attr::kSoftwarePipelineOrder.name,
                             ffi::Array<int64_t>{0, 1, 2});
    new_state->sch->Annotate(r_loop_fused, tvm::s_tir::attr::kSoftwarePipelineAsyncStages.name,
                             ffi::Array<int64_t>{0});
    ret.push_back(std::move(new_state));
  }
  return ret;
}

void MultiLevelTilingObj::AnnotateCooperativeFetching(Schedule* sch,
                                                      const s_tir::SBlockRV& block) const {
  // Filter out invalid vector lanes according to the data type.
  const s_tir::SBlockObj* block_node = (*sch)->GetSRef(block)->StmtAs<s_tir::SBlockObj>();
  TVM_FFI_ICHECK_EQ(block_node->writes.size(), 1);
  const DLDataType dtype =
      block_node->writes[0]->source.as_or_throw<tvm::tirx::TensorVar>().dtype()->dtype;
  std::function<bool(int)> f_filter = nullptr;
  if (dtype == DLDataType{kDLFloat, 32, 1}) {
    f_filter = [&](int vector_len) { return vector_len <= 4; };
  } else if (dtype == DLDataType{kDLFloat, 16, 1}) {
    f_filter = [&](int vector_len) {
      return (vector_len == 1 || vector_len % 2 == 0) && vector_len <= 8;
    };
  } else if (dtype == DLDataType{kDLInt, 8, 1}) {
    f_filter = [&](int vector_len) { return vector_len <= 16; };
  }
  std::vector<int> valid_vector_lens;
  valid_vector_lens.reserve(vector_load_lens.size());
  if (f_filter != nullptr) {
    std::copy_if(vector_load_lens.begin(), vector_load_lens.end(),
                 std::back_inserter(valid_vector_lens), f_filter);
  } else {
    valid_vector_lens = vector_load_lens;
  }

  if (!valid_vector_lens.empty()) {
    int n = valid_vector_lens.size();
    double prob = 1.0 / n;
    ffi::Array<int64_t> valid_vector_lens_arr;
    valid_vector_lens_arr.reserve(valid_vector_lens.size());
    for (int v : valid_vector_lens) valid_vector_lens_arr.push_back(static_cast<int64_t>(v));
    s_tir::ExprRV vector_load_len = (*sch)->SampleCategorical(
        valid_vector_lens_arr, ffi::Array<FloatImm>(n, FloatImm(PrimType::Float(32), prob)));
    (*sch)->Annotate(block, tvm::s_tir::attr::kMetaScheduleCooperativeFetch.name, vector_load_len);
  }
}

// Constructor

ScheduleRule ScheduleRule::MultiLevelTiling(
    ffi::String structure, ffi::Optional<ffi::Array<ffi::String>> tile_binds,
    ffi::Optional<int64_t> max_innermost_factor,
    ffi::Optional<ffi::Array<int64_t>> vector_load_lens,
    ffi::Optional<ffi::Map<ffi::String, ffi::Any>> reuse_read,
    ffi::Optional<ffi::Map<ffi::String, ffi::Any>> reuse_write,
    ffi::Optional<ffi::Function> filter_fn) {
  auto node = MultiLevelTilingInitCommon<MultiLevelTilingObj>(
      structure, tile_binds, max_innermost_factor, vector_load_lens, reuse_read, reuse_write);
  node->filter_fn_ = filter_fn;
  return ScheduleRule(ffi::UnsafeInit{}, node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.meta_schedule.ScheduleRuleMultiLevelTiling",
                        ScheduleRule::MultiLevelTiling);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
