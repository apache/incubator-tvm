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

using s_tir::Instruction;
using s_tir::InstructionKind;
using s_tir::Trace;

/*! \brief A mutator that mutates the compute-at location decision of SampleComputeLocation */
class MutateComputeLocationObj : public MutatorObj {
 public:
  /*! \brief JSON representation of the workload */
  std::string json_mod_;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<MutateComputeLocationObj>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.MutateComputeLocation",
                                    MutateComputeLocationObj, MutatorObj);

 public:
  // Inherit from `MutatorObj`
  void InitializeWithTuneContext(const TuneContext& context) final {
    this->json_mod_ = ffi::json::Stringify(
        ffi::ToJSONGraph(context->mod.value(), ffi::json::Object{{"tvm_version", TVM_VERSION}}),
        /*indent=*/2);
  }
  // Inherit from `MutatorObj`
  ffi::Optional<Trace> Apply(const Trace& trace, TRandState* rand_state) final;
  // Inherit from `MutatorObj`
  Mutator Clone() const final {
    ffi::ObjectPtr<MutateComputeLocationObj> n = ffi::make_object<MutateComputeLocationObj>(*this);
    return Mutator(ffi::UnsafeInit{}, n);
  }

 private:
  struct Candidate {
    /*! \brief The SampleComputeLocation instruction */
    Instruction inst;
    /*! \brief The candidate compute-at locations */
    std::vector<int> locs;

    explicit Candidate(Instruction inst, std::vector<int> locs)
        : inst(std::move(inst)), locs(std::move(locs)) {}
  };

  std::vector<Candidate> FindCandidates(const Trace& trace, TRandState* rand_state);
};

/*!
 * \brief Find all appearances of instruction `SampleComputeLocation` whose decision can be mutated
 *        to at lease one other value
 * \param trace The trace from which to find the instructions
 * \return All the candidate instructions together with the candidate compute-at locations
 */
std::vector<MutateComputeLocationObj::Candidate> MutateComputeLocationObj::FindCandidates(
    const Trace& trace, TRandState* rand_state) {
  s_tir::Schedule sch = s_tir::Schedule::Traced(                                       //
      /*mod=*/ffi::FromJSONGraph(ffi::json::Parse(this->json_mod_)).cast<IRModule>(),  //
      /*rand_state=*/ForkSeed(rand_state),                                             //
      /*debug_mode=*/0,                                                                //
      /*error_render_level=*/s_tir::ScheduleErrorRenderLevel::kNone);

  static InstructionKind inst_sample_compute_location =
      InstructionKind::Get("SampleComputeLocation");
  std::vector<MutateComputeLocationObj::Candidate> candidates;

  auto f_decision_provider = [&](const s_tir::Instruction& inst,  //
                                 const ffi::Array<Any>& inputs,   //
                                 const ffi::Array<Any>& attrs,    //
                                 const Any& decision) -> Any {
    if (inst->kind.same_as(inst_sample_compute_location)) {
      // Step 1. Extract the instruction input and the old decision.
      TVM_FFI_ICHECK_EQ(inputs.size(), 1);
      tirx::StmtSRef block_sref = sch->GetSRef(inputs[0].as_or_throw<s_tir::SBlockRV>());
      // SampleComputeLocation decision is Optional<int64_t> after the
      // Integer phase-out.
      int old_decision = decision.cast<int64_t>();

      // Step 2. Collect all the compute_at locations.
      auto [location_srefs, location_indices] = CollectComputeLocation(sch->state(), block_sref);
      // Step 3. Remove the old decision.
      auto it = std::find(location_indices.begin(), location_indices.end(), old_decision);
      if (it != location_indices.end()) {
        location_srefs.erase(location_srefs.begin() + (it - location_indices.begin()));
        location_indices.erase(it);
      }
      TVM_FFI_ICHECK_EQ(location_srefs.size(), location_indices.size());
      // Step 4. Add a new candidate if there are at least one remaining compute-at position.
      if (!location_srefs.empty()) {
        candidates.emplace_back(inst, std::move(location_indices));
      }
    }
    return decision;
  };
  trace->ApplyToSchedule(sch,                       //
                         /*remove_postproc=*/true,  //
                         /*decision_provider=*/f_decision_provider);
  return candidates;
}

ffi::Optional<Trace> MutateComputeLocationObj::Apply(const Trace& trace, TRandState* rand_state) {
  std::vector<Candidate> candidates = FindCandidates(trace, rand_state);
  if (candidates.empty()) {
    return std::nullopt;
  }
  const Candidate& candidate = candidates[s_tir::SampleInt(rand_state, 0, candidates.size())];
  int loc = candidate.locs[s_tir::SampleInt(rand_state, 0, candidate.locs.size())];
  return trace->WithDecision(candidate.inst, static_cast<int64_t>(loc), /*remove_postproc=*/true);
}

Mutator Mutator::MutateComputeLocation() {
  return Mutator(ffi::UnsafeInit{}, ffi::make_object<MutateComputeLocationObj>());
}

TVM_FFI_STATIC_INIT_BLOCK() { MutateComputeLocationObj::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.meta_schedule.MutatorMutateComputeLocation",
                        Mutator::MutateComputeLocation);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
