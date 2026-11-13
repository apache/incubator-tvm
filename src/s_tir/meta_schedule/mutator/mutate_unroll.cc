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

#include "../utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

/*!
 * \brief Check if an instruction is annotate with
 * `meta_schedule_unroll_explicit` or `meta_schedule_unroll_implicit`
 * \param inst The instruction to be checked
 * \return Whether the instruction is annotated
 */
bool IsAnnotateWithUnroll(const Instruction& inst) {
  static const InstructionKind& inst_annotate = InstructionKind::Get("Annotate");
  if (!inst->kind.same_as(inst_annotate)) {
    return false;
  }
  TVM_FFI_ICHECK_EQ(inst->attrs.size(), 1);
  ffi::String ann_key = inst->attrs[0].as_or_throw<ffi::String>();
  return ann_key == tvm::s_tir::attr::kMetaScheduleUnrollExplicit.name ||
         ann_key == tvm::s_tir::attr::kMetaScheduleUnrollImplicit.name;
}

}  // namespace s_tir
}  // namespace tvm

namespace tvm {
namespace s_tir {
namespace meta_schedule {

using s_tir::Instruction;
using s_tir::Trace;

/*! \brief Create a Mutator that mutates auto unroll step */
class MutateUnrollObj : public MutatorObj {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<MutateUnrollObj>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.MutateUnroll", MutateUnrollObj,
                                    MutatorObj);

 public:
  struct Candidate;
  // Inherit from `MutatorObj`
  void InitializeWithTuneContext(const TuneContext& context) final {}
  // Inherit from `MutatorObj`
  ffi::Optional<Trace> Apply(const Trace& trace, TRandState* rand_state) final;
  // Inherit from `MutatorObj`
  Mutator Clone() const final {
    ffi::ObjectPtr<MutateUnrollObj> n = ffi::make_object<MutateUnrollObj>(*this);
    return Mutator(ffi::UnsafeInit{}, n);
  }
};

/*! \brief A candidate to be mutated */
struct MutateUnrollObj::Candidate {
  /*! \brief The sampling instruction to be mutated */
  ffi::Optional<Instruction> inst;
  /*! \brief The probability */
  std::vector<double> probs;
  /*! \brief The decision made */
  int decision;
};

/*!
 * \brief Find the Sample-Categorical instruction to be mutated that affects the maximal unroll step
 * \param trace The trace to be mutated
 * \param rand_state The random state
 * \param candidates The mutation candidate
 * \return Whether a decision is found
 */
bool FindUnrollDecision(const Trace& trace, TRandState* rand_state,
                        MutateUnrollObj::Candidate* candidate) {
  using s_tir::InstructionKind;
  using s_tir::InstructionObj;
  static const InstructionKind& inst_sample_categorical = InstructionKind::Get("SampleCategorical");
  std::unordered_map<const ExprObj*, const InstructionObj*> sample_insts;
  std::vector<const InstructionObj*> ann_insts;
  sample_insts.reserve(trace->insts.size());
  ann_insts.reserve(trace->insts.size());
  for (const Instruction& inst : trace->insts) {
    if (inst->kind.same_as(inst_sample_categorical)) {
      TVM_FFI_ICHECK_EQ(inst->outputs.size(), 1);
      const ExprObj* var_rv = inst->outputs[0].as_or_throw<PrimExpr>().get();
      sample_insts[var_rv] = inst.get();
    } else if (IsAnnotateWithUnroll(inst)) {
      ann_insts.push_back(inst.get());
    }
  }
  int n_ann_insts = ann_insts.size();
  if (n_ann_insts == 0) {
    return false;
  }
  const InstructionObj* ann_inst = ann_insts[s_tir::SampleInt(rand_state, 0, n_ann_insts)];
  TVM_FFI_ICHECK_EQ(ann_inst->inputs.size(), 2);
  const auto* var_rv = ann_inst->inputs[1].as_or_throw<PrimExpr>().get();
  TVM_FFI_ICHECK(sample_insts.count(var_rv));
  const InstructionObj* sample_inst = sample_insts.at(var_rv);
  TVM_FFI_ICHECK_EQ(sample_inst->attrs.size(), 2);
  candidate->inst = ffi::GetRef<Instruction>(sample_inst);
  // SampleCategorical decision is Optional<int64_t> after the Integer phase-out.
  candidate->decision = trace->decisions[ffi::GetRef<Instruction>(sample_inst)].cast<int64_t>();
  candidate->probs = support::AsVector<FloatImm, double>(
      sample_inst->attrs[1].as_or_throw<ffi::Array<FloatImm>>());
  return true;
}

ffi::Optional<Trace> MutateUnrollObj::Apply(const Trace& trace, TRandState* rand_state) {
  Candidate candidate;
  if (!FindUnrollDecision(trace, rand_state, &candidate)) {
    return std::nullopt;
  }
  if (candidate.probs.size() == 0) {
    return std::nullopt;
  }
  candidate.probs.erase(candidate.probs.begin() + candidate.decision);
  int result = s_tir::MakeMultinomialSampler(rand_state, candidate.probs)();
  if (result >= candidate.decision) {
    result += 1;
  }
  return trace->WithDecision(candidate.inst.value(), static_cast<int64_t>(result),
                             /*remove_postproc=*/true);
}

Mutator Mutator::MutateUnroll() {
  return Mutator(ffi::UnsafeInit{}, ffi::make_object<MutateUnrollObj>());
}

TVM_FFI_STATIC_INIT_BLOCK() { MutateUnrollObj::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.meta_schedule.MutatorMutateUnroll", Mutator::MutateUnroll);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
