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

#include <algorithm>
#include <map>
#include <unordered_map>

#include "../utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

/*!
 * \brief Check if the instruction is annotation with `meta_schedule_parallel`
 * \param inst The instruction to be checked
 * \return Whether the instruction is annotation with `meta_schedule_parallel`
 */
bool IsAnnotateWithParallel(const Instruction& inst) {
  static const InstructionKind& inst_annotate = InstructionKind::Get("Annotate");
  if (!inst->kind.same_as(inst_annotate)) {
    return false;
  }
  TVM_FFI_ICHECK_EQ(inst->attrs.size(), 1);
  ffi::String ann_key = inst->attrs[0].as_or_throw<ffi::String>();
  return ann_key == tvm::s_tir::attr::kMetaScheduleParallel.name;
}

/*!
 * \brief Replace the annotation value
 * \param inst The instruction to be replaced
 * \param ann_val The new annotation value
 * \return The replaced instruction
 */
Instruction ReplaceAnnValue(Instruction inst, int64_t ann_val) {
  TVM_FFI_ICHECK_EQ(inst->inputs.size(), 2);
  return Instruction(/*kind=*/inst->kind,                    //
                     /*inputs=*/{inst->inputs[0], ann_val},  //
                     /*attrs=*/inst->attrs,
                     /*outputs=*/inst->outputs);
}

/*!
 * \brief Get the output of the instruction Get-Block
 * \param inst The instruction to be checked
 * \return The output of the instruction Get-Block
 */
const SBlockRVObj* GetInstGetSBlockOutput(const Instruction& inst) {
  static const InstructionKind& inst_get_sblock = InstructionKind::Get("GetSBlock");
  if (!inst->kind.same_as(inst_get_sblock)) {
    return nullptr;
  }
  TVM_FFI_ICHECK_EQ(inst->outputs.size(), 1);
  const SBlockRVObj* block = TVM_TYPE_AS(inst->outputs[0], SBlockRVObj);
  return block;
}

/*!
 * \brief Analyze the parallel structure
 * \param self The schedule state
 * \param block_name The name of the root block
 * \param func_name The name of the Function
 * \param limit The uplimit of the parallelism
 * \return The parallel structure
 */
std::vector<std::vector<int64_t>> AnalyzeParallel(const ScheduleState& self,
                                                  const ffi::String& block_name,
                                                  const ffi::String& func_name, int64_t limit) {
  ffi::Array<StmtSRef> block_srefs =
      GetSBlocks(self, block_name, self->mod->GetGlobalVar(func_name));
  TVM_FFI_ICHECK_EQ(block_srefs.size(), 1);
  const SBlockObj* block = TVM_SREF_TO_SBLOCK(block_srefs[0]);
  ScopeBlockLoopInfo info = GetScopeBlockLoopInfo(ffi::GetRef<SBlock>(block));
  std::vector<std::vector<int64_t>> results;
  results.reserve(info.realizes.size());
  for (const SBlockRealize& realize : info.realizes) {
    // Step 1. Extract static loop extents for spatial loops
    std::vector<int64_t> loop_extents;
    const ForObj* loop = nullptr;
    for (const StmtSRefObj* loop_sref = self->stmt2ref.at(realize->block.get())->parent;
         (loop = loop_sref->StmtAs<ForObj>()) != nullptr;  //
         loop_sref = loop_sref->parent) {
      int64_t loop_extent = -1;
      PrimExpr ext_imm_expr = Range::GetExtentFromBeginEnd(loop->begin, loop->end);
      const auto* ext_imm = ext_imm_expr.as<IntImmObj>();
      if (auto ext = ext_imm ? ext_imm->value.as<int64_t>() : std::nullopt; ext.has_value()) {
        if (!info.non_spatial_vars.count(loop->var.get())) {
          loop_extent = *ext;
        }
      }
      if (loop_extent != -1) {
        loop_extents.push_back(loop_extent);
      } else {
        loop_extents.clear();
      }
    }
    // Step 2. Take the prefix product of loop extents
    if (!loop_extents.empty()) {
      results.emplace_back();
      std::vector<int64_t>& result = results.back();
      result.reserve(loop_extents.size());
      int64_t prod_extent = 1;
      for (auto it = loop_extents.rbegin(); it != loop_extents.rend(); ++it) {
        result.push_back(prod_extent *= *it);
        if (prod_extent >= limit) {
          break;
        }
      }
    }
  }
  return results;
}

/*!
 * \brief Get the number of parallelizable loops for each subtree
 * \param loop_extent_prods The parallel structure for each subtree
 * \param limit The uplimit of the parallelism
 * \return The number of parallelizable loops for each subtree
 */
std::vector<int> GetNumFusedLoops(const std::vector<std::vector<int64_t>>& loop_extent_prods,
                                  int64_t limit) {
  std::vector<int> results;
  results.reserve(loop_extent_prods.size());
  for (const std::vector<int64_t>& prods : loop_extent_prods) {
    int n = prods.size();
    int i = std::upper_bound(prods.begin(), prods.end(), limit) - prods.begin();
    if (i > 0 && prods[i - 1] == limit) {
      --i;
    }
    if (i != n) {
      ++i;
    }
    results.push_back(i);
  }
  return results;
}

}  // namespace s_tir
}  // namespace tvm

namespace tvm {
namespace s_tir {
namespace meta_schedule {

using s_tir::Instruction;
using s_tir::Trace;

/*! \brief Create a Mutator that mutates the parallel extent */
class MutateParallelObj : public MutatorObj {
 public:
  /*!
   * \brief The maximum number of jobs to be launched per CPU core.
   * It sets the uplimit of CPU parallelism, i.e. `num_cores * max_jobs_per_core`.
   * Use -1 to disable parallelism.
   */
  int64_t max_jobs_per_core;
  /*! \brief The number of cores in CPU. */
  int max_parallel_extent_;
  /*! \brief JSON representation of the workload */
  std::string json_mod_;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<MutateParallelObj>().def_ro("max_jobs_per_core",
                                                &MutateParallelObj::max_jobs_per_core);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.MutateParallel", MutateParallelObj,
                                    MutatorObj);

 public:
  struct Candidate;
  // Inherit from `MutatorObj`
  void InitializeWithTuneContext(const TuneContext& context) final {
    Target target = context->target.value();
    this->max_parallel_extent_ = GetTargetNumCores(target) * this->max_jobs_per_core;
    this->json_mod_ = ffi::json::Stringify(
        ffi::ToJSONGraph(context->mod.value(), ffi::json::Object{{"tvm_version", TVM_VERSION}}),
        /*indent=*/2);
  }
  // Inherit from `MutatorObj`
  ffi::Optional<Trace> Apply(const Trace& trace, TRandState* rand_state) final;
  // Inherit from `MutatorObj`
  Mutator Clone() const final {
    ffi::ObjectPtr<MutateParallelObj> n = ffi::make_object<MutateParallelObj>(*this);
    return Mutator(ffi::UnsafeInit{}, n);
  }
};

/*! \brief The candidate to be mutated */
struct MutateParallelObj::Candidate {
  /*! \brief The annotation instruction */
  ffi::Optional<Instruction> inst;
  /*! \brief The current parallel extent */
  int64_t parallel_extent;
  /*! \brief The name of the root block */
  ffi::String block_name;
  /*! \brief The name of the Function */
  ffi::String func_name;
};

/*!
 * \brief Get an instruction that annotates the maximum parallel extent
 * \param trace The trace to be mutated
 * \param rand_state The random state
 * \param candidate The candidate to be mutated
 * \return Whether a decision is found
 */
bool FindParallelDecision(const Trace& trace, TRandState* rand_state,
                          MutateParallelObj::Candidate* candidate) {
  using s_tir::InstructionObj;
  using s_tir::SBlockRVObj;
  std::unordered_map<const SBlockRVObj*, const InstructionObj*> get_sblock_insts;
  std::vector<const InstructionObj*> ann_insts;
  get_sblock_insts.reserve(trace->insts.size());
  ann_insts.reserve(trace->insts.size());
  for (const Instruction& inst : trace->insts) {
    if (s_tir::IsAnnotateWithParallel(inst)) {
      ann_insts.push_back(inst.get());
    }
    if (const SBlockRVObj* block_rv = s_tir::GetInstGetSBlockOutput(inst)) {
      get_sblock_insts[block_rv] = inst.get();
    }
  }
  int n_ann_insts = ann_insts.size();
  if (n_ann_insts == 0) {
    return false;
  }
  const InstructionObj* ann_inst = ann_insts[s_tir::SampleInt(rand_state, 0, n_ann_insts)];
  TVM_FFI_ICHECK_EQ(ann_inst->inputs.size(), 2);
  const InstructionObj* get_sblock_inst =
      get_sblock_insts.at(ann_inst->inputs[0].as_or_throw<s_tir::SBlockRV>().get());
  TVM_FFI_ICHECK_EQ(get_sblock_inst->attrs.size(), 2);
  candidate->inst = ffi::GetRef<Instruction>(ann_inst);
  candidate->parallel_extent = ann_inst->inputs[1].cast<int64_t>();
  candidate->block_name = get_sblock_inst->attrs[0].as_or_throw<ffi::String>();
  candidate->func_name = get_sblock_inst->attrs[1].as_or_throw<ffi::String>();
  return true;
}

ffi::Optional<Trace> MutateParallelObj::Apply(const Trace& trace, TRandState* rand_state) {
  // Step 1. Find a parallel decision.
  Candidate candidate;
  if (!FindParallelDecision(trace, rand_state, &candidate)) {
    return std::nullopt;
  }
  // Step 2. Replay the instructions to recover loop extents
  s_tir::Schedule sch = s_tir::Schedule::Traced(                                       //
      /*mod=*/ffi::FromJSONGraph(ffi::json::Parse(this->json_mod_)).cast<IRModule>(),  //
      /*rand_state=*/ForkSeed(rand_state),                                             //
      /*debug_mode=*/0,
      /*error_render_level=*/s_tir::ScheduleErrorRenderLevel::kNone);
  trace->ApplyToSchedule(sch, /*remove_postproc=*/true);
  // Step 3. Find all possible parallel plans
  std::vector<std::vector<int64_t>> loop_extent_prods = s_tir::AnalyzeParallel(
      sch->state(), candidate.block_name, candidate.func_name, this->max_parallel_extent_);
  std::unordered_map<int64_t, std::vector<int>> limit2plan;
  std::map<std::vector<int>, int64_t> plan2limit;
  for (const std::vector<int64_t>& prods : loop_extent_prods) {
    for (int64_t limit : prods) {
      if (limit <= this->max_parallel_extent_ && !limit2plan.count(limit)) {
        std::vector<int> plan = s_tir::GetNumFusedLoops(loop_extent_prods, limit);
        limit2plan[limit] = plan;
        plan2limit[plan] = limit;
      }
    }
  }
  // Step 4. Remove the original plan and remove it
  std::vector<int> original_plan =
      s_tir::GetNumFusedLoops(loop_extent_prods, candidate.parallel_extent);
  auto it = plan2limit.find(original_plan);
  if (it != plan2limit.end()) {
    plan2limit.erase(it);
  }
  // Step 5. Pick a new plan
  int n_plans = plan2limit.size();
  if (n_plans == 0) {
    return std::nullopt;
  }
  it = plan2limit.begin();
  for (int i = 0, n = s_tir::SampleInt(rand_state, 0, n_plans); i < n; ++i) {
    ++it;
  }
  int64_t limit = it->second;
  // Step 6. Assemble a new trace
  ffi::Array<Instruction> insts;
  insts.reserve(trace->insts.size());
  for (const Instruction& inst : trace->insts) {
    if (inst.same_as(candidate.inst.value())) {
      insts.push_back(s_tir::ReplaceAnnValue(candidate.inst.value(), limit));
    } else if (inst->kind->IsPostproc()) {
      break;
    } else {
      insts.push_back(inst);
    }
  }
  return Trace(insts, trace->decisions);
}

Mutator Mutator::MutateParallel(int64_t max_jobs_per_core) {
  ffi::ObjectPtr<MutateParallelObj> n = ffi::make_object<MutateParallelObj>();
  n->max_jobs_per_core = max_jobs_per_core;
  return Mutator(ffi::UnsafeInit{}, n);
}

TVM_FFI_STATIC_INIT_BLOCK() { MutateParallelObj::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.meta_schedule.MutatorMutateParallel", Mutator::MutateParallel);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
