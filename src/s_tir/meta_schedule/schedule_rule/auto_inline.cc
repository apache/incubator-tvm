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
namespace meta_schedule {

using s_tir::AnalyzeReadWritePattern;
using s_tir::CanComputeInline;
using s_tir::CanReverseComputeInline;
using s_tir::GetAnn;
using s_tir::GetConsumers;
using s_tir::GetProducers;
using s_tir::GetRootFunction;
using s_tir::GetSBlockRealize;
using s_tir::HasIfThenElse;
using s_tir::HasOp;
using s_tir::IsSpatialFunction;
using s_tir::ScheduleState;

/*! \brief The type of inline to be performed on a specific block */
enum class InlineType : int32_t {
  /*! \brief No inline opportunity */
  kNoInline = 0,
  /*! \brief Inline the block into its consumer */
  kInlineIntoConsumer = 1,
  /*! \brief Inline the block into its producer */
  kInlineIntoProducer = 2,
};

bool IsInSpatialFunction(const s_tir::Schedule& sch, const tirx::StmtSRef& block_sref) {
  using namespace tvm::tirx;
  const StmtSRefObj* sref = block_sref.get();
  for (; sref->parent != nullptr; sref = sref->parent) {
  }
  TVM_FFI_ICHECK(sref->stmt != nullptr && sref->stmt->IsInstance<SBlockObj>());
  return IsSpatialFunction(ffi::GetRef<Function>(GetRootFunction(sch->mod(), sref->stmt, nullptr)));
}

/*! \brief The rule that inlines spatial blocks if it satisfies some conditions. */
class AutoInlineObj : public ScheduleRuleObj {
 public:
  /*! \brief Checks if the specific block should be inlined */
  inline InlineType CheckInline(const s_tir::Schedule& sch, const s_tir::SBlockRV& block_rv);

  // Inherited from ScheduleRuleObj
  void InitializeWithTuneContext(const TuneContext& context) final {}

  // Inherited from ScheduleRuleObj
  ffi::Array<s_tir::Schedule> Apply(const s_tir::Schedule& sch,
                                    const s_tir::SBlockRV& block_rv) final {
    InlineType inline_type = CheckInline(sch, block_rv);
    if (inline_type == InlineType::kInlineIntoConsumer) {
      sch->ComputeInline(block_rv);
    } else if (inline_type == InlineType::kInlineIntoProducer) {
      sch->ReverseComputeInline(block_rv);
    }
    return {sch};
  }

  // Inherited from ScheduleRuleObj
  ScheduleRule Clone() const final {
    ffi::ObjectPtr<AutoInlineObj> n = ffi::make_object<AutoInlineObj>(*this);
    return ScheduleRule(ffi::UnsafeInit{}, n);
  }

 public:
  /*! \brief If allows to inline a block into its producer */
  bool into_producer;
  /*! \brief If allows to inline a block into its consumer */
  bool into_consumer;
  /*! \brief Always inline constant tensors */
  bool inline_const_tensor;
  /*! \brief Always disallow if-then-else-like constructs */
  bool disallow_if_then_else;
  /*! \brief Always require the read-to-write mapping to be injective to do auto inline */
  bool require_injective;
  /*! \brief Always require the read-to-write mapping to be ordered to do auto inline */
  bool require_ordered;
  /*! \brief The operators that are disallowed in auto inline */
  ffi::Array<Op> disallow_op;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AutoInlineObj>()
        .def_ro("into_producer", &AutoInlineObj::into_producer)
        .def_ro("into_consumer", &AutoInlineObj::into_consumer)
        .def_ro("inline_const_tensor", &AutoInlineObj::inline_const_tensor)
        .def_ro("disallow_if_then_else", &AutoInlineObj::disallow_if_then_else)
        .def_ro("require_injective", &AutoInlineObj::require_injective)
        .def_ro("require_ordered", &AutoInlineObj::require_ordered)
        .def_ro("disallow_op", &AutoInlineObj::disallow_op);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.AutoInline", AutoInlineObj,
                                    ScheduleRuleObj);
};

inline InlineType AutoInlineObj::CheckInline(const s_tir::Schedule& sch,
                                             const s_tir::SBlockRV& block_rv) {
  using namespace tvm::tirx;
  StmtSRef block_sref = sch->GetSRef(block_rv);
  bool is_pure_sptial = IsInSpatialFunction(sch, block_sref);
  ScheduleState state = sch->state();
  const SBlockObj* block = TVM_SREF_TO_SBLOCK(block_sref);
  SBlockRealize realize = GetSBlockRealize(state, block_sref);
  // Cond 1. The block has only one write tensor
  if (block->writes.size() != 1) {
    return InlineType::kNoInline;
  }
  // Cond 2. For a block that generates a constant tensor, ignore all other conditions
  if (inline_const_tensor && block->reads.empty()) {
    ffi::Array<tirx::StmtSRef> consumer_srefs = GetConsumers(state, block_sref);
    if (!consumer_srefs.empty() && CanComputeInline(state, block_sref)) {
      return InlineType::kInlineIntoConsumer;
    }
  }
  // Cond 3. The block doesn't contain any disallowed operators
  if (!is_pure_sptial && !disallow_op.empty() && HasOp(realize, disallow_op)) {
    return InlineType::kNoInline;
  }
  // Cond 4. The block doesn't have any if-then-else-like constructs
  if (!is_pure_sptial && disallow_if_then_else && HasIfThenElse(realize)) {
    return InlineType::kNoInline;
  }
  // Cond 5. The mapping from read indices to write indices are injective and ordered
  if (!is_pure_sptial && (require_injective || require_ordered)) {
    const TensorRegion& write_region = block->writes[0];
    for (const TensorRegion& read_region : block->reads) {
      bool injective, ordered;
      auto _ = std::ignore;
      std::tie(/*exists=*/_, /*surjective=*/_, injective, ordered, /*no_const_read=*/_,
               /*no_shift_read=*/_) = AnalyzeReadWritePattern(read_region, write_region);
      if (require_injective && injective == false) {
        return InlineType::kNoInline;
      }
      if (require_ordered && ordered == false) {
        return InlineType::kNoInline;
      }
    }
  }
  // Cond 6. The block is disallowed for auto inline
  if (ffi::Optional<ffi::String> ann =
          s_tir::GetAnn<ffi::String>(block_sref, tvm::s_tir::attr::kMetaScheduleInlineRule.name)) {
    if (ann.value() == "disable") return InlineType::kNoInline;
  }
  // Last cond: Check inline into the consumers or the spatial producer
  // Skip if:
  // - root block (no parent scope).
  // - root block is wrapped in a For loop, then parent is non-null, but there is still no enclosing
  // SBlockObj.
  const StmtSRefObj* p = block_sref->parent;
  for (; p != nullptr; p = p->parent) {
    if (p->stmt->IsInstance<SBlockObj>()) break;
  }
  if (p == nullptr) {
    return InlineType::kNoInline;
  }
  tirx::StmtSRef scope_block = s_tir::GetScopeRoot(sch->state(), block_sref,
                                                   /*require_stage_pipeline=*/false);
  if (into_consumer) {
    ffi::Array<tirx::StmtSRef> consumer_srefs = GetConsumers(state, block_sref);
    if (!consumer_srefs.empty() && CanComputeInline(state, block_sref)) {
      return InlineType::kInlineIntoConsumer;
    }
  }
  if (into_producer) {
    ffi::Array<tirx::StmtSRef> producer_srefs = GetProducers(state, block_sref);
    if (producer_srefs.size() == 1 &&
        s_tir::IsCompleteBlock(sch->state(), producer_srefs[0], scope_block) &&
        CanReverseComputeInline(state, block_sref) &&
        !GetAnn<ffi::String>(producer_srefs[0], tvm::s_tir::attr::kMetaScheduleAutoTensorize.name)
             .has_value()) {
      return InlineType::kInlineIntoProducer;
    }
  }
  return InlineType::kNoInline;
}

ScheduleRule ScheduleRule::AutoInline(bool into_producer,          //
                                      bool into_consumer,          //
                                      bool inline_const_tensor,    //
                                      bool disallow_if_then_else,  //
                                      bool require_injective,      //
                                      bool require_ordered,        //
                                      ffi::Optional<ffi::Array<ffi::String>> disallow_op) {
  ffi::ObjectPtr<AutoInlineObj> n = ffi::make_object<AutoInlineObj>();
  n->into_producer = into_producer;
  n->into_consumer = into_consumer;
  n->inline_const_tensor = inline_const_tensor;
  n->disallow_if_then_else = disallow_if_then_else;
  n->require_injective = require_injective;
  n->require_ordered = require_ordered;
  n->disallow_op.clear();
  if (disallow_op.has_value()) {
    ffi::Array<ffi::String> op_names = disallow_op.value();
    n->disallow_op.reserve(op_names.size());
    for (const ffi::String& op_name : op_names) {
      n->disallow_op.push_back(Op::Get(op_name));
    }
  }
  return ScheduleRule(ffi::UnsafeInit{}, n);
}

TVM_FFI_STATIC_INIT_BLOCK() { AutoInlineObj::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.meta_schedule.ScheduleRuleAutoInline", ScheduleRule::AutoInline);
}

/*! \brief Inline blocks that produce a constant scalar. */
class InlineConstantScalarsObj : public ScheduleRuleObj {
 public:
  void InitializeWithTuneContext(const TuneContext& context) final {}

  ffi::Array<s_tir::Schedule> Apply(const s_tir::Schedule& sch,
                                    const s_tir::SBlockRV& block_rv) final {
    // Look for a block of the form
    // block compile_engine_const(iter_var(vi, range(min=0, ext=1))) {
    //   reads([])
    //   writes([compile_engine_const[]])
    //   compile_engine_const[] = 59
    // }
    auto block = sch->Get(block_rv);
    if (block->reads.size() == 0 && block->writes.size() == 1 &&
        block->writes[0]->source.as_or_throw<tvm::tirx::TensorVar>().shape().size() == 0) {
      auto sref = sch->GetSRef(block_rv);
      if (!s_tir::IsOutputBlock(sch->state(), sref,
                                s_tir::GetScopeRoot(sch->state(), sref, true))) {
        sch->ComputeInline(block_rv);
      }
    }
    return {sch};
  }

  ScheduleRule Clone() const final {
    ffi::ObjectPtr<InlineConstantScalarsObj> n = ffi::make_object<InlineConstantScalarsObj>(*this);
    return ScheduleRule(ffi::UnsafeInit{}, n);
  }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<InlineConstantScalarsObj>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.InlineConstantScalars",
                                    InlineConstantScalarsObj, ScheduleRuleObj);
};

ScheduleRule ScheduleRule::InlineConstantScalars() {
  ffi::ObjectPtr<InlineConstantScalarsObj> n = ffi::make_object<InlineConstantScalarsObj>();
  return ScheduleRule(ffi::UnsafeInit{}, n);
}

TVM_FFI_STATIC_INIT_BLOCK() { InlineConstantScalarsObj::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.meta_schedule.ScheduleRuleInlineConstantScalars",
                        ScheduleRule::InlineConstantScalars);
}
}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
