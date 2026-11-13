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
#include <tvm/s_tir/stmt.h>
#include <tvm/support/gpu_thread_tag.h>

#include "../utils.h"

namespace tvm {
namespace s_tir {
namespace meta_schedule {

class CrossThreadReductionObj : public ScheduleRuleObj {
 public:
  // Inherited from ScheduleRuleObj
  void InitializeWithTuneContext(const TuneContext& context) final {
    TVM_FFI_ICHECK(context->target.has_value());
    Target target = context->target.value();

    ffi::Optional<int64_t> opt_max_threads_per_block =
        target->GetAttr<int64_t>("max_threads_per_block");
    ffi::Optional<int64_t> opt_warp_size = target->GetAttr<int64_t>("thread_warp_size");

    if (!opt_max_threads_per_block.has_value()) {
      TVM_PY_LOG(WARNING, context->logger)
          << "Target does not have attribute \"max_threads_per_block\", therefore the "
             "rule CrossThreadReduction will not be applied";
    }
    if (!opt_warp_size.has_value()) {
      TVM_PY_LOG(WARNING, context->logger)
          << "Target does not have attribute \"thread_warp_size\", therefore the rule "
             "CrossThreadReduction will not be applied";
    }
    max_threads_per_block = opt_max_threads_per_block.value_or(-1);
    warp_size = opt_warp_size.value_or(-1);
  }

  // Inherited from ScheduleRuleObj
  ffi::Array<s_tir::Schedule> Apply(const s_tir::Schedule& sch,
                                    const s_tir::SBlockRV& block_rv) final {
    // Step 0. Check the conditions of this rule.
    if (max_threads_per_block == -1 || warp_size == -1) {
      return {sch};
    }
    const tirx::StmtSRef& block_sref = sch->GetSRef(block_rv);
    if (!NeedsRFactorOrCrossThreadReduction(sch->state(), block_sref, max_threads_per_block,
                                            warp_size)) {
      return {sch};
    }

    // Step 1. Make a copy of the original schedule. The new copy is used for scheduling.
    s_tir::Schedule tmp_sch = sch->Copy();
    tmp_sch->Seed(sch->ForkSeed());

    // Step 2. Check the opportunity for block fusion. We say "fusible", if we can compute-at the
    // block to its consumers. We want to fuse as much as possible because it results in
    // significantly faster schedule.
    // `target_loop` is the loop position where the input block will be computed at.
    // `target_sblock` is the consumer block that we want to compute-at the input block to.
    // `tgt_block_innermost_loop` is the innermost loop outside the target block.

    auto [fusible, target_loop, target_sblock, tgt_block_innermost_loop] =
        GetComputeTargetLoopAndBlock(tmp_sch, block_rv);

    // Step 3. Try block fusion.
    int n_candidate = static_cast<int>(thread_extents.size());
    ffi::Array<FloatImm> probs(n_candidate, FloatImm(PrimType::Float(32), 1.0 / n_candidate));
    s_tir::ExprRV thread_extent = tmp_sch->SampleCategorical(thread_extents, probs);
    if (fusible) {
      TVM_FFI_ICHECK(target_sblock.defined());
      TVM_FFI_ICHECK(target_loop.defined());

      // Step 3.1.
      // - If the outer loops of `target_sblock` haven't been bound to "threadIdx.x", we should
      // first
      //   bound the innermost outer loop of `target_sblock` to threadIdx. Possibly we need to split
      //   the loop before binding.
      // - Otherwise, we search for the extent of "threadIdx.x" and use it as the split factor.
      if (!InThreadBinding(tmp_sch, target_sblock)) {
        const ffi::Array<s_tir::LoopRV>& split_res =
            tmp_sch->Split(tgt_block_innermost_loop, {std::nullopt, thread_extent});
        tmp_sch->Bind(split_res[1], "threadIdx.x");
        if (tgt_block_innermost_loop.same_as(target_loop)) {
          target_loop = split_res[0];
        }
      } else {
        thread_extent = GetThreadIdxExtentFromTrace(tmp_sch->trace().value());
      }
      // Step 3.2. Do the compute-at.
      tmp_sch->ComputeAt(block_rv, target_loop, /*preserve_unit_loops=*/true);
      // Step 3.3. Set the storage scope of the output tensor to shared memory.
      tmp_sch->SetScope(block_rv, /*tensor_index=*/0, /*storage_scope=*/"shared");
    }

    // Step 4. Reorder the loop axes if reduction loops are not innermost. After the reordering,
    // fuse all the reduction loops.
    size_t num_spatial_loops;
    s_tir::LoopRV fused_reduce_loop;
    ReorderAndFuseReductionLoops(tmp_sch, block_rv, &fused_reduce_loop, &num_spatial_loops);
    // Step 5. Split the fused reduction loop and bind the inner one to threadIdx.
    const ffi::Array<s_tir::LoopRV>& split_res =
        tmp_sch->Split(fused_reduce_loop, {std::nullopt, thread_extent});
    tmp_sch->Bind(split_res[1], "threadIdx.x");

    return {tmp_sch, sch};
  }

  // Inherited from ScheduleRuleObj
  ScheduleRule Clone() const final {
    ffi::ObjectPtr<CrossThreadReductionObj> n = ffi::make_object<CrossThreadReductionObj>(*this);
    return ScheduleRule(ffi::UnsafeInit{}, n);
  }

 private:
  /*!
   * \brief Check whether the input block is in thread scope, i.e., some of its outer loop is
   * bound to threadIdx.
   * \param sch The TensorIR schedule
   * \param block The block to be checked
   * \return A boolean indicating whether the block is in thread scope.
   */
  bool InThreadBinding(const s_tir::Schedule& sch, const s_tir::SBlockRV& block) {
    const ffi::Array<s_tir::LoopRV>& axes = sch->GetLoops(block);
    for (const s_tir::LoopRV& loop_rv : axes) {
      const For& loop = sch->Get(loop_rv);
      support::GPUThreadTag thread_scope = s_tir::GetThreadBindingTag(loop.get());
      if (s_tir::IsThreadIdx(thread_scope)) {
        return true;
      }
    }
    return false;
  }

  /*!
   * \brief Get the ExprRV which used to define the extent of a given loop.
   * \param trace The trace of the schedule, where the extent is to be found
   * \param loop The loop whose extent is to be found
   * \param extent The finding result
   * \return Whether the find is successful.
   */
  bool GetLoopRVExtentSource(const s_tir::Trace& trace, const s_tir::LoopRV& loop,
                             s_tir::ExprRV* extent) {
    for (const s_tir::Instruction& inst : trace->insts) {
      if (inst->kind->name == "Split") {
        auto fcheck = [&](const Any& a) -> bool { return a.as<ffi::Object>() == loop.get(); };
        int i = std::find_if(inst->outputs.begin(), inst->outputs.end(), fcheck) -
                inst->outputs.begin();
        TVM_FFI_CHECK(inst->inputs[1 + i] != nullptr, ValueError)
            << "Extracting an extent which needs inference is not supported so far";
        *extent = inst->inputs[1 + i].as_or_throw<s_tir::ExprRV>();
        return true;
      }
    }
    return false;
  }

  /*!
   * \brief Get the ExprRV extent of "threadIdx.x" in the given schedule trace.
   * \param trace The trace of the schedule, where the extent is to be found
   * \return The extent of "threadIdx.x" in the input schedule
   */
  s_tir::ExprRV GetThreadIdxExtentFromTrace(const s_tir::Trace& trace) {
    s_tir::ExprRV extent{ffi::UnsafeInit()};
    for (const s_tir::Instruction& inst : trace->insts) {
      if (inst->kind->name == "Bind" &&
          inst->attrs[0].as_or_throw<ffi::String>() == "threadIdx.x") {
        if (GetLoopRVExtentSource(trace, inst->inputs[0].as_or_throw<s_tir::LoopRV>(), &extent)) {
          return extent;
        }
      }
    }
    TVM_FFI_CHECK(false, ValueError) << "Unable to get the extent of \"threadIdx.x\"";
    throw;
  }

  /*!
   * \brief Get the compute-at target loop and the first block under the target loop.
   * \param sch The TensorIR schedule
   * \param block_rv The block whose compute-at target loop is queried
   * \return A tuple consisting of
   * 1. a boolean indicating whether the block can be computed at some target loop (a.k.a. fusible);
   * 2. the compute-at target loop when fusible, or a null loop random variable;
   * 3. the first block under the target loop when fusible, or a null block random variable;
   * 4. the innermost loop outside the target block when fusible, or a null block random variable.
   */
  std::tuple<bool, s_tir::LoopRV, s_tir::SBlockRV, s_tir::LoopRV> GetComputeTargetLoopAndBlock(
      const s_tir::Schedule& sch, const s_tir::SBlockRV& block_rv) {
    // Step 0. Due to technical reason of some primitives (e.g., compute-at), if the block is doing
    // a tuple reduction, fusion is temporarily not supported.
    if (sch->Get(block_rv)->writes.size() != 1) {
      return std::make_tuple(false, s_tir::LoopRV{ffi::UnsafeInit()},
                             s_tir::SBlockRV{ffi::UnsafeInit()}, s_tir::LoopRV{ffi::UnsafeInit()});
    }

    // Step 1. Get all the consumers of the input block.
    ffi::Array<s_tir::SBlockRV> consumers = sch->GetConsumers(block_rv);

    // Step 2. If the block has no consumer or the first consumer needs multi-level tiling, it is
    // not fusible.
    if (consumers.empty() ||
        s_tir::NeedsMultiLevelTiling(sch->state(), sch->GetSRef(consumers[0]))) {
      return std::make_tuple(false, s_tir::LoopRV{ffi::UnsafeInit()},
                             s_tir::SBlockRV{ffi::UnsafeInit()}, s_tir::LoopRV{ffi::UnsafeInit()});
    }

    // Step 3. Calculate the lowest common ancestor of all the consumers.
    // - If the lowest common ancestor is a block:
    //   - if there is only one consumer, the target block is that consumer;
    //   - if there are multiple consumers, they must not share a common loop, and the case is not
    //     fusible;
    // - If the lowest common ancestor is a loop, the target block is also the first consumer.
    const tirx::StmtSRef& lca_sref =
        s_tir::GetSRefLowestCommonAncestor(s_tir::SBlockRVs2StmtSRefs(sch, consumers));
    if (consumers.size() > 1 && lca_sref->StmtAs<s_tir::SBlockObj>() != nullptr) {
      return std::make_tuple(false, s_tir::LoopRV{ffi::UnsafeInit()},
                             s_tir::SBlockRV{ffi::UnsafeInit()}, s_tir::LoopRV{ffi::UnsafeInit()});
    }

    // Step 4. Get the outer loops of the target block, and get the compute-at position index.
    ffi::Array<s_tir::LoopRV> tgt_block_loops = sch->GetLoops(consumers[0]);
    int pos = GetComputePosition(sch, sch->GetLoops(block_rv), tgt_block_loops, lca_sref);

    // Step 5. A negative position index means not fusible, and vice-versa.
    if (pos < 0) {
      return std::make_tuple(false, s_tir::LoopRV{ffi::UnsafeInit()},
                             s_tir::SBlockRV{ffi::UnsafeInit()}, s_tir::LoopRV{ffi::UnsafeInit()});
    } else {
      return std::make_tuple(true, tgt_block_loops[pos], consumers[0], tgt_block_loops.back());
    }
  }

  /*!
   * \brief Get the compute-at position index of the input block, according to
   * 1. the loops outside the input block;
   * 2. the loops outside the target block;
   * 3. the lowest common ancestor of all the consumers of the input block.
   * \param sch The TensorIR schedule
   * \param block_loops The loops outside the input block
   * \param tgt_block_loops The loops outside the target block
   * \param lca_sref The lowest common ancestor of all the consumers of the input block
   * \return The compute-at position index of the input block
   */
  int GetComputePosition(const s_tir::Schedule& sch, const ffi::Array<s_tir::LoopRV>& block_loops,
                         const ffi::Array<s_tir::LoopRV>& tgt_block_loops,
                         const tirx::StmtSRef& lca_sref) {
    int n_block_loop = static_cast<int>(block_loops.size());
    int n_tgt_block_loop = static_cast<int>(tgt_block_loops.size());

    for (int i = 0; i < n_block_loop && i < n_tgt_block_loop; ++i) {
      if (s_tir::GetLoopIterType(sch->GetSRef(block_loops[i])) != s_tir::IterVarType::kDataPar) {
        return i - 1;
      } else if (sch->GetSRef(tgt_block_loops[i]).same_as(lca_sref)) {
        // If the lowest common ancestor is a loop, the compute location of the input block should
        // not be deeper than the LCA loop.
        return i;
      }
    }
    return std::min(n_block_loop, n_tgt_block_loop) - 1;
  }

 public:
  /*! \brief The maximum number of threads allowed in a thread block */
  int max_threads_per_block;
  /*! \brief The number of threads per warp */
  int warp_size;
  /*! \brief Candidates of thread axis extent (values are required to be positive). */
  ffi::Array<int64_t> thread_extents;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<CrossThreadReductionObj>()
        .def_ro("max_threads_per_block", &CrossThreadReductionObj::max_threads_per_block)
        .def_ro("warp_size", &CrossThreadReductionObj::warp_size)
        .def_ro("thread_extents", &CrossThreadReductionObj::thread_extents);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.CrossThreadReduction",
                                    CrossThreadReductionObj, ScheduleRuleObj);
};

ScheduleRule ScheduleRule::CrossThreadReduction(ffi::Array<int64_t> thread_extents) {
  for (int64_t extent : thread_extents) {
    TVM_FFI_CHECK(extent > 0, ValueError) << "The candidates of thread extent must be positive";
  }
  ffi::ObjectPtr<CrossThreadReductionObj> n = ffi::make_object<CrossThreadReductionObj>();
  n->thread_extents = std::move(thread_extents);
  return ScheduleRule(ffi::UnsafeInit{}, n);
}

TVM_FFI_STATIC_INIT_BLOCK() { CrossThreadReductionObj::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.meta_schedule.ScheduleRuleCrossThreadReduction",
                        ScheduleRule::CrossThreadReduction);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
