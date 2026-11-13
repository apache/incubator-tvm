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

#include "../utils.h"

namespace tvm {
namespace s_tir {
namespace meta_schedule {

/*!
 * \brief Design Space Generator that generates design spaces by applying schedule rules to blocks
 *  in post-DFS order.
 * */
class PostOrderApplyObj : public SpaceGeneratorObj {
 public:
  /*!
   * \brief Optional block names to target. If not specified all blocks will have spaces generated.
   */
  ffi::Function f_block_filter_ = nullptr;
  /*! \brief The random state. -1 means using random number. */
  TRandState rand_state_ = -1;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PostOrderApplyObj>();
  }

  void InitializeWithTuneContext(const TuneContext& context) final {
    SpaceGeneratorObj::InitializeWithTuneContext(context);
    this->rand_state_ = ForkSeed(&context->rand_state);
  }

  ffi::Array<s_tir::Schedule> GenerateDesignSpace(const IRModule& mod) final {
    using ScheduleAndUnvisitedBlocks = std::pair<s_tir::Schedule, ffi::Array<s_tir::SBlockRV>>;
    TVM_FFI_CHECK(sch_rules.has_value(), ValueError) << "`sch_rules` is not set in PostOrderApply";
    s_tir::Schedule sch = s_tir::Schedule::Traced(
        /*mod=*/mod,
        /*rand_state=*/ForkSeed(&this->rand_state_),
        /*debug_mode=*/0,
        /*error_render_level=*/s_tir::ScheduleErrorRenderLevel::kDetail);

    std::vector<ScheduleAndUnvisitedBlocks> stack;
    ffi::Array<s_tir::Schedule> result{sch};
    ffi::Array<s_tir::SBlockRV> all_blocks = SBlockCollector::Collect(sch, f_block_filter_);

    for (ScheduleRule sch_rule : sch_rules.value()) {
      for (const s_tir::Schedule& sch : result) {
        stack.emplace_back(sch, all_blocks);
      }
      result.clear();
      while (!stack.empty()) {
        // get the stack.top()
        auto [sch, blocks] = stack.back();
        stack.pop_back();
        // if all blocks are visited
        if (blocks.empty()) {
          result.push_back(sch);
          continue;
        }
        // otherwise, get the last block that is not visited
        s_tir::SBlockRV block_rv = blocks.back();
        blocks.pop_back();
        if (!sch->HasBlock(block_rv)) {
          stack.emplace_back(sch, blocks);
          continue;
        }
        if (!ScheduleRule::IsApplyCustomRule(sch_rule)) {
          if (s_tir::GetAnn<ffi::String>(sch->GetSRef(block_rv),
                                         tvm::s_tir::attr::kScheduleRule.name)
                  .has_value()) {
            stack.emplace_back(sch, blocks);
            continue;
          }
        }
        ffi::Array<s_tir::Schedule> applied = sch_rule->Apply(sch, /*block=*/block_rv);
        for (const s_tir::Schedule& sch : applied) {
          stack.emplace_back(sch, blocks);
        }
      }
    }
    return result;
  }

  SpaceGenerator Clone() const final {
    ffi::ObjectPtr<PostOrderApplyObj> n = ffi::make_object<PostOrderApplyObj>(*this);
    CloneRules(this, n.get());
    return SpaceGenerator(n);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.PostOrderApply", PostOrderApplyObj,
                                    SpaceGeneratorObj);
};

SpaceGenerator SpaceGenerator::PostOrderApply(
    ffi::Function f_block_filter, ffi::Optional<ffi::Array<ScheduleRule>> sch_rules,
    ffi::Optional<ffi::Array<Postproc>> postprocs,
    ffi::Optional<ffi::Map<Mutator, FloatImm>> mutator_probs) {
  ffi::ObjectPtr<PostOrderApplyObj> n = ffi::make_object<PostOrderApplyObj>();
  n->sch_rules = std::move(sch_rules);
  n->postprocs = std::move(postprocs);
  n->mutator_probs = std::move(mutator_probs);
  n->f_block_filter_ = std::move(f_block_filter);
  return SpaceGenerator(n);
}

TVM_FFI_STATIC_INIT_BLOCK() { PostOrderApplyObj::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.meta_schedule.SpaceGeneratorPostOrderApply",
                        SpaceGenerator::PostOrderApply);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
