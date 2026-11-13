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
#include <tvm/s_tir/stmt.h>

#include "../analysis.h"
#include "../utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

ffi::Array<StmtSRef> GetSBlocks(const ScheduleState& self, const ffi::String& name,
                                const GlobalVar& gv) {
  struct Finder : public StmtExprVisitor {
    using StmtExprVisitor::Visit_;

    ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
      if (value.as<ExprObj>()) return std::nullopt;
      return StmtExprVisitor::Visit(value);
    }

    explicit Finder(const ScheduleState& self, const ffi::String& name)
        : self_(self), name_(name) {}

    ffi::Optional<VisitInterrupt> Visit_(const SBlockObj* block) override {
      if (block->name_hint == name_) {
        auto it = self_->stmt2ref.find(block);
        TVM_FFI_ICHECK(it != self_->stmt2ref.end());
        results_.push_back(it->second);
      }
      return StmtExprVisitor::Visit_(block);
    }

    const ScheduleState& self_;
    const ffi::String& name_;
    ffi::Array<StmtSRef> results_;
  };

  BaseFunc func = self->mod->Lookup(gv);
  const auto* function = TVM_TYPE_AS(func, FunctionObj);
  auto finder = ffi::make_object<Finder>(self, name);
  finder->Visit(function->body);
  return std::move(finder->results_);
}

ffi::Array<StmtSRef> GetLoops(const StmtSRef& block_sref) {
  std::vector<StmtSRef> result;
  for (StmtSRefObj* parent = block_sref->parent; parent && parent->stmt->IsInstance<ForObj>();
       parent = parent->parent) {
    result.push_back(ffi::GetRef<StmtSRef>(parent));
  }
  return {result.rbegin(), result.rend()};
}

ffi::Array<StmtSRef> GetChildBlocks(const ScheduleState& self, const StmtSRef& parent_sref) {
  struct Collector : public StmtExprVisitor {
    using StmtExprVisitor::Visit_;

    ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
      if (value.as<ExprObj>()) return std::nullopt;
      return StmtExprVisitor::Visit(value);
    }

   private:
    ffi::Optional<VisitInterrupt> Visit_(const SBlockObj* block) final {
      result.push_back(self->stmt2ref.at(block));
      return std::nullopt;
    }

   public:
    explicit Collector(const ScheduleState& self) : self(self) {}

    const ScheduleState& self;
    ffi::Array<StmtSRef> result;
  };
  auto collector = ffi::make_object<Collector>(self);
  if (parent_sref->stmt->IsInstance<ForObj>()) {
    const auto* loop = static_cast<const ForObj*>(parent_sref->stmt);
    collector->Visit(loop->body);
  } else if (parent_sref->stmt->IsInstance<SBlockObj>()) {
    const auto* block = static_cast<const SBlockObj*>(parent_sref->stmt);
    collector->Visit(block->body);
  }
  return std::move(collector->result);
}

ffi::Array<StmtSRef> GetProducers(const ScheduleState& self, const StmtSRef& block_sref) {
  StmtSRef scope_root = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/false);
  return GetProducers(block_sref, self->GetSBlockScope(scope_root));
}

ffi::Array<StmtSRef> GetConsumers(const ScheduleState& self, const StmtSRef& block_sref) {
  StmtSRef scope_root = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/false);
  return GetConsumers(block_sref, self->GetSBlockScope(scope_root));
}

ffi::Array<StmtSRef> GetOutputBlocks(const ScheduleState& self, const StmtSRef& scope_sref) {
  const auto* scope_block = TVM_SREF_TO_SBLOCK(scope_sref);
  return GetOutputBlocks(self, scope_block);
}

/******** InstructionKind Registration ********/

struct GetSBlockTraits : public UnpackedInstTraits<GetSBlockTraits> {
  static constexpr const char* kName = "GetSBlock";
  static constexpr bool kIsPure = true;

 private:
  static constexpr size_t kNumInputs = 0;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static SBlockRV UnpackedApplyToSchedule(Schedule sch, ffi::String name, ffi::String func_name) {
    return sch->GetSBlock(name, func_name);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String name,
                                      ffi::String func_name) {
    PythonAPICall py("get_sblock");
    py.Input("name", name);
    py.Input("func_name", func_name);
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

struct GetLoopsTraits : public UnpackedInstTraits<GetLoopsTraits> {
  static constexpr const char* kName = "GetLoops";
  static constexpr bool kIsPure = true;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static ffi::Array<LoopRV> UnpackedApplyToSchedule(Schedule sch, SBlockRV block_rv) {
    return sch->GetLoops(block_rv);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block_rv) {
    PythonAPICall py("get_loops");
    py.Input("block", block_rv);
    py.OutputList(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

struct GetChildBlocksTraits : public UnpackedInstTraits<GetChildBlocksTraits> {
  static constexpr const char* kName = "GetChildBlocks";
  static constexpr bool kIsPure = true;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static ffi::Array<SBlockRV> UnpackedApplyToSchedule(Schedule sch,
                                                      ffi::ObjectRef block_or_loop_rv) {
    if (auto block = block_or_loop_rv.as<SBlockRV>()) {
      return sch->GetChildBlocks(block.value());
    }
    if (auto loop = block_or_loop_rv.as<LoopRV>()) {
      return sch->GetChildBlocks(loop.value());
    }
    TVM_FFI_THROW(TypeError) << "Expected SBlock or Loop, but gets: "
                             << block_or_loop_rv->GetTypeKey();
    throw;
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs,
                                      ffi::String block_or_loop_rv) {
    PythonAPICall py("get_child_blocks");
    py.Input("", block_or_loop_rv);
    py.OutputList(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

struct GetProducersTraits : public UnpackedInstTraits<GetProducersTraits> {
  static constexpr const char* kName = "GetProducers";
  static constexpr bool kIsPure = true;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static ffi::Array<SBlockRV> UnpackedApplyToSchedule(Schedule sch, SBlockRV block_rv) {
    return sch->GetProducers(block_rv);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block_rv) {
    PythonAPICall py("get_producers");
    py.Input("block", block_rv);
    py.OutputList(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

struct GetConsumersTraits : public UnpackedInstTraits<GetConsumersTraits> {
  static constexpr const char* kName = "GetConsumers";
  static constexpr bool kIsPure = true;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static ffi::Array<SBlockRV> UnpackedApplyToSchedule(Schedule sch, SBlockRV block_rv) {
    return sch->GetConsumers(block_rv);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block_rv) {
    PythonAPICall py("get_consumers");
    py.Input("block", block_rv);
    py.OutputList(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

struct GetOutputBlocksTraits : public UnpackedInstTraits<GetOutputBlocksTraits> {
  static constexpr const char* kName = "GetOutputBlocks";
  static constexpr bool kIsPure = true;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static ffi::Array<SBlockRV> UnpackedApplyToSchedule(Schedule sch, SBlockRV block_rv) {
    return sch->GetOutputBlocks(block_rv);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block_rv) {
    PythonAPICall py("get_output_blocks");
    py.Input("block", block_rv);
    py.OutputList(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

TVM_FFI_STATIC_INIT_BLOCK() {
  RegisterInstructionKind<GetSBlockTraits>();
  RegisterInstructionKind<GetLoopsTraits>();
  RegisterInstructionKind<GetChildBlocksTraits>();
  RegisterInstructionKind<GetProducersTraits>();
  RegisterInstructionKind<GetConsumersTraits>();
  RegisterInstructionKind<GetOutputBlocksTraits>();
}

}  // namespace s_tir
}  // namespace tvm
