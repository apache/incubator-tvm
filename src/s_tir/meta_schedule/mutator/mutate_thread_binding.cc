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

#include "../utils.h"

namespace tvm {
namespace s_tir {
namespace meta_schedule {

using s_tir::Instruction;
using s_tir::InstructionKind;
using s_tir::Trace;

/*! \brief A mutator that mutates the thread binding factor decision of SampleCategorical */
class MutateThreadBindingObj : public MutatorObj {
 public:
  /*! \brief JSON representation of the workload */
  std::string json_mod_;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<MutateThreadBindingObj>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.MutateThreadBinding",
                                    MutateThreadBindingObj, MutatorObj);

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
    ffi::ObjectPtr<MutateThreadBindingObj> n = ffi::make_object<MutateThreadBindingObj>(*this);
    return Mutator(ffi::UnsafeInit{}, n);
  }

 private:
  struct Candidate {
    /*! \brief The sampling instruction to be mutated */
    Instruction inst;
    /*! \brief The probability */
    std::vector<double> probs;
    /*! \brief The decision made */
    int decision;

    explicit Candidate(Instruction inst, std::vector<double> probs, int decision)
        : inst(std::move(inst)), probs(std::move(probs)), decision(std::move(decision)) {}
  };

  std::vector<Candidate> FindCandidates(const Trace& trace, TRandState* rand_state);
};

/*!
 * \brief Find Candidate with the following pattern:
 * \code
 * v = sch.sample_categorical(...)
 * l1, l2 = sch.split(loop=l0, factors=[None, v])
 * sch.bind(loop=l2, thread_axis="threadIdx.x")
 * \endcode
 *
 * \param trace The trace from which to find the instructions
 * \return All the candidate instructions
 */
std::vector<MutateThreadBindingObj::Candidate> MutateThreadBindingObj::FindCandidates(
    const Trace& trace, TRandState* rand_state) {
  using s_tir::InstructionObj;

  static InstructionKind inst_sample_categorical = InstructionKind::Get("SampleCategorical");
  static InstructionKind inst_split = InstructionKind::Get("Split");
  static InstructionKind inst_bind = InstructionKind::Get("Bind");

  std::vector<MutateThreadBindingObj::Candidate> candidates;
  std::unordered_map<const ExprObj*, const s_tir::InstructionObj*> sample_insts;
  std::unordered_map<const s_tir::LoopRVObj*, const s_tir::InstructionObj*> sampled_split_insts;
  std::vector<const InstructionObj*> bind_insts;

  auto is_split_by_sample = [&sample_insts](const Instruction& inst) -> bool {
    if (!inst->kind.same_as(inst_split)) {
      return false;
    }
    // Only consider cases with 2 factors and the first one is None
    if (inst->inputs.size() != 3 || inst->inputs[1] != nullptr) return false;
    TVM_FFI_ICHECK(inst->inputs[2] != nullptr);

    return sample_insts.find(inst->inputs[2].as_or_throw<PrimExpr>().get()) != sample_insts.end();
  };

  auto is_thread_binding_by_sample = [&sampled_split_insts](const Instruction& inst) -> bool {
    if (!inst->kind.same_as(inst_bind)) {
      return false;
    }
    TVM_FFI_ICHECK_EQ(inst->inputs.size(), 1);
    TVM_FFI_ICHECK_EQ(inst->attrs.size(), 1);
    if (inst->attrs[0].as_or_throw<ffi::String>() != "threadIdx.x") return false;

    return sampled_split_insts.find(inst->inputs[0].as_or_throw<s_tir::LoopRV>().get()) !=
           sampled_split_insts.end();
  };

  for (const Instruction& inst : trace->insts) {
    if (inst->kind.same_as(inst_sample_categorical)) {
      TVM_FFI_ICHECK_EQ(inst->outputs.size(), 1);
      const ExprObj* var_rv = TVM_TYPE_AS(inst->outputs[0], ExprObj);
      sample_insts[var_rv] = inst.get();
    } else if (is_split_by_sample(inst)) {
      TVM_FFI_ICHECK_EQ(inst->outputs.size(), 2);
      // Only consider the inner loop, which can be bound to threadIdx.x
      const s_tir::LoopRVObj* var_rv = TVM_TYPE_AS(inst->outputs[1], s_tir::LoopRVObj);
      sampled_split_insts[var_rv] = inst.get();
    } else if (is_thread_binding_by_sample(inst)) {
      bind_insts.push_back(inst.get());
    }
  }

  for (const InstructionObj* bind_inst : bind_insts) {
    const auto* loop_rv = TVM_TYPE_AS(bind_inst->inputs[0], s_tir::LoopRVObj);
    auto split_it = sampled_split_insts.find(loop_rv);
    TVM_FFI_ICHECK(split_it != sampled_split_insts.end());
    const InstructionObj* split_inst = split_it->second;

    const auto* expr_rv = TVM_TYPE_AS(split_inst->inputs[2], ExprObj);
    auto sample_it = sample_insts.find(expr_rv);
    TVM_FFI_ICHECK(sample_it != sample_insts.end());
    const InstructionObj* sample_inst = sample_it->second;

    // SampleCategorical decision is Optional<int64_t> after the Integer phase-out.
    int decision = trace->decisions[ffi::GetRef<Instruction>(sample_inst)].cast<int64_t>();

    std::vector<double> probs = support::AsVector<FloatImm, double>(
        sample_inst->attrs[1].as_or_throw<ffi::Array<FloatImm>>());

    candidates.emplace_back(ffi::GetRef<Instruction>(sample_inst), probs, decision);
  }
  return candidates;
}

ffi::Optional<Trace> MutateThreadBindingObj::Apply(const Trace& trace, TRandState* rand_state) {
  std::vector<Candidate> candidates = FindCandidates(trace, rand_state);
  if (candidates.empty()) {
    return std::nullopt;
  }
  Candidate candidate = candidates[s_tir::SampleInt(rand_state, 0, candidates.size())];
  // Remove the current decision
  candidate.probs.erase(candidate.probs.begin() + candidate.decision);
  int result = s_tir::MakeMultinomialSampler(rand_state, candidate.probs)();
  if (result >= candidate.decision) {
    result += 1;
  }
  return trace->WithDecision(candidate.inst, static_cast<int64_t>(result),
                             /*remove_postproc=*/true);
}

Mutator Mutator::MutateThreadBinding() {
  return Mutator(ffi::UnsafeInit{}, ffi::make_object<MutateThreadBindingObj>());
}

TVM_FFI_STATIC_INIT_BLOCK() { MutateThreadBindingObj::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.meta_schedule.MutateThreadBinding", Mutator::MutateThreadBinding);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
