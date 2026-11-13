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
#include <tvm/s_tir/stmt.h>

#include <mutex>
#include <unordered_map>

#include "../utils.h"

namespace tvm {
namespace s_tir {
namespace meta_schedule {

using s_tir::Instruction;
using s_tir::InstructionKind;
using s_tir::Trace;

/*!
 * \brief Cast the decision of Sample-Perfect-Tile to an array of integers
 * \param decision The decision of Sample-Perfect-Tile
 * \return The converted tile sizes
 */
std::vector<int64_t> CastTilingDecision(const ffi::ObjectRef& decision) {
  Any decision_any;
  decision_any = decision;
  if (auto int_array = decision_any.try_cast<ffi::Array<int64_t>>()) {
    return std::vector<int64_t>(int_array->begin(), int_array->end());
  }
  return support::AsVector<ffi::ObjectRef, int64_t>(
      decision_any.cast<ffi::Array<ffi::ObjectRef>>());
}

/*!
 * \brief Calculate the product of elements in an array
 * \param array The array
 * \return The product of elements in the array
 */
int64_t Product(const std::vector<int64_t>& array) {
  int64_t result = 1;
  for (int64_t x : array) {
    result *= x;
  }
  return result;
}

/*! \brief A mutator that mutates the tile size */
class MutateTileSizeObj : public MutatorObj {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<MutateTileSizeObj>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.MutateTileSize", MutateTileSizeObj,
                                    MutatorObj);

 public:
  // Inherit from `MutatorObj`
  void InitializeWithTuneContext(const TuneContext& context) final {}
  // Inherit from `MutatorObj`
  ffi::Optional<Trace> Apply(const Trace& trace, TRandState* rand_state) final;
  // Inherit from `MutatorObj`
  Mutator Clone() const final {
    ffi::ObjectPtr<MutateTileSizeObj> n = ffi::make_object<MutateTileSizeObj>(*this);
    return Mutator(ffi::UnsafeInit{}, n);
  }
};

/*!
 * \brief Find a sample-perfect-tile decision in the trace
 * \param trace The trace
 * \param rand_state The random state
 * \param inst The instruction selected
 * \param decision The decision selected
 * \return Whether a decision is found
 */
void FindSamplePerfectTile(const Trace& trace, std::vector<Instruction>* inst,
                           std::vector<std::vector<int64_t>>* decision) {
  static const InstructionKind& inst_sample_perfect_tile =
      InstructionKind::Get("SamplePerfectTile");
  std::vector<Instruction>& instructions = *inst;
  std::vector<std::vector<int64_t>>& decisions = *decision;
  instructions.reserve(trace->decisions.size());
  decisions.reserve(trace->decisions.size());
  for (const auto& kv : trace->decisions) {
    const Instruction& inst = kv.first;
    if (inst->kind.same_as(inst_sample_perfect_tile)) {
      std::vector<int64_t> tiles = CastTilingDecision(kv.second.cast<ffi::ObjectRef>());
      if (tiles.size() >= 2 && Product(tiles) >= 2) {
        instructions.push_back(inst);
        decisions.push_back(tiles);
      }
    }
  }
}

void FindSampleVectorize(const Trace& trace, std::vector<Instruction>* inst,
                         std::vector<int64_t>* decision) {
  static const InstructionKind& inst_sample_categorical = InstructionKind::Get("SampleCategorical");
  static const InstructionKind& inst_annotate = InstructionKind::Get("Annotate");
  std::vector<Instruction>& instructions = *inst;
  std::vector<int64_t>& decisions = *decision;
  std::unordered_set<const ffi::Object*> annotated;
  instructions.reserve(trace->decisions.size());
  decisions.reserve(trace->decisions.size());
  annotated.reserve(trace->decisions.size());
  // Find annotation with `meta_schedule_cooperative_fetch`
  for (const Instruction& inst : trace->insts) {
    if (inst->kind.same_as(inst_annotate)) {
      TVM_FFI_ICHECK_EQ(inst->attrs.size(), 1);
      TVM_FFI_ICHECK_EQ(inst->inputs.size(), 2);
      if (inst->attrs[0].as_or_throw<ffi::String>() ==
          tvm::s_tir::attr::kMetaScheduleCooperativeFetch.name) {
        const auto* ann_val = inst->inputs[1].as<s_tir::ExprRVObj>();
        TVM_FFI_ICHECK(ann_val);
        annotated.insert(ann_val);
      }
    }
  }
  // Find sampling instruction that generates the annotation
  for (const auto& kv : trace->decisions) {
    const Instruction& inst = kv.first;
    if (inst->kind.same_as(inst_sample_categorical)) {
      TVM_FFI_ICHECK_EQ(inst->outputs.size(), 1);
      if (annotated.count(inst->outputs[0].as<ffi::Object>())) {
        TVM_FFI_ICHECK_EQ(inst->attrs.size(), 2);
        std::vector<double> probs =
            support::AsVector<FloatImm, double>(inst->attrs[1].as_or_throw<ffi::Array<FloatImm>>());
        if (probs.size() == 1) {
          // Skip mutating the sampling instructions who have only single candidate.
          continue;
        }
        // SampleCategorical decision is Optional<int64_t> after the
        // Integer phase-out, so the Any holds a bare POD int.
        instructions.push_back(inst);
        decisions.push_back(kv.second.cast<int64_t>());
      }
    }
  }
}

struct FactorMemo {
  static std::vector<int> Factorize(int n) {
    if (const std::vector<int>* result = Global()->Query(n)) {
      return *result;
    }
    std::vector<int> result;
    for (int64_t i = 1; i * i <= n; ++i) {
      if (n % i == 0) {
        result.push_back(i);
        if (i * i != n) {
          result.push_back(n / i);
        }
      }
    }
    std::sort(result.begin(), result.end());
    Global()->Add(n, result);
    return result;
  }

 private:
  const std::vector<int>* Query(int n) {
    std::unique_lock<std::mutex> lock(mutex_);
    auto it = memo_.find(n);
    if (it != memo_.end()) {
      return &it->second;
    }
    return nullptr;
  }

  void Add(int n, std::vector<int> result) {
    std::unique_lock<std::mutex> lock(mutex_);
    memo_.emplace(n, std::move(result));
  }

  static FactorMemo* Global() {
    static FactorMemo singleton;
    return &singleton;
  }

  std::unordered_map<int, std::vector<int>> memo_;
  std::mutex mutex_;
};

ffi::Optional<Trace> MutateSampleTileSize(const Trace& trace, Instruction inst,
                                          std::vector<int64_t> tiles, TRandState* rand_state) {
  int n_splits = tiles.size();
  // Step 1. Choose two loops, `x` and `y`
  int x, y;
  // select source
  while (true) {
    x = s_tir::SampleInt(rand_state, 0, n_splits);
    if (tiles[x] <= 1) {
      continue;
    }
    y = s_tir::SampleInt(rand_state, 0, n_splits - 1);
    if (y >= x) {
      ++y;
    }
    std::vector<int> factors = FactorMemo::Factorize(tiles[x]);
    // Step 2. Choose the divide factor
    int64_t divide_factor;
    if (y != n_splits - 1) {
      divide_factor = factors[s_tir::SampleInt(rand_state, 1, factors.size())];
    } else {
      ffi::BigInt limit = inst->attrs[1].cast<int64_t>();
      int max_factor_index = static_cast<int>(factors.size()) - 1;
      for (; max_factor_index >= 1; max_factor_index--) {
        if (factors[max_factor_index] * tiles[y] <= limit) {
          break;
        }
      }
      if (max_factor_index == 0) {
        if (n_splits <= 2) {
          return std::nullopt;
        }
        // Failed on this dst_idx, try next one.
        continue;
      }
      divide_factor = factors[s_tir::SampleInt(rand_state, 1, max_factor_index + 1)];
    }
    tiles[x] /= divide_factor;
    tiles[y] *= divide_factor;
    return trace->WithDecision(inst, ffi::Array<int64_t>(tiles.begin(), tiles.end()),
                               /*remove_postproc=*/true);
  }
}

ffi::Optional<Trace> MutateSampleVectorize(const Trace& trace, Instruction inst,
                                           int64_t original_decision, TRandState* rand_state) {
  TVM_FFI_ICHECK_EQ(inst->attrs.size(), 2);
  std::vector<double> probs =
      support::AsVector<FloatImm, double>(inst->attrs[1].as_or_throw<ffi::Array<FloatImm>>());
  probs.erase(probs.begin() + original_decision);
  int result = s_tir::MakeMultinomialSampler(rand_state, probs)();
  if (result >= original_decision) {
    result += 1;
  }
  return trace->WithDecision(inst, static_cast<int64_t>(result), /*remove_postproc=*/true);
}

ffi::Optional<Trace> MutateTileSizeObj::Apply(const Trace& trace, TRandState* rand_state) {
  std::vector<Instruction> sample_perfect_tile_insts;
  std::vector<Instruction> sample_vectorize_insts;
  std::vector<std::vector<int64_t>> sample_perfect_tile_tiles;
  std::vector<int64_t> sample_vectorize_decisions;
  FindSamplePerfectTile(trace, &sample_perfect_tile_insts, &sample_perfect_tile_tiles);
  FindSampleVectorize(trace, &sample_vectorize_insts, &sample_vectorize_decisions);
  int size_a = sample_perfect_tile_insts.size();
  int size_b = sample_vectorize_insts.size();
  if (size_a == 0 && size_b == 0) {
    return std::nullopt;
  }
  int n = s_tir::SampleInt(rand_state, 0, size_a + size_b);
  if (n < size_a) {
    return MutateSampleTileSize(trace, sample_perfect_tile_insts[n], sample_perfect_tile_tiles[n],
                                rand_state);
  } else {
    n -= size_a;
    return MutateSampleVectorize(trace, sample_vectorize_insts[n], sample_vectorize_decisions[n],
                                 rand_state);
  }
}

Mutator Mutator::MutateTileSize() {
  return Mutator(ffi::UnsafeInit{}, ffi::make_object<MutateTileSizeObj>());
}

TVM_FFI_STATIC_INIT_BLOCK() { MutateTileSizeObj::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.meta_schedule.MutatorMutateTileSize", Mutator::MutateTileSize);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
