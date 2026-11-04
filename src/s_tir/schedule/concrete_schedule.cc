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
#include "./concrete_schedule.h"

#include <tvm/ffi/cast.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/tensor_intrin.h>

#include <random>

namespace tvm {
namespace s_tir {
using namespace tvm::prim;
using namespace tvm::tirx;

Schedule Schedule::Concrete(IRModule mod, LinearCongruentialEngine::TRandState seed, int debug_mask,
                            ScheduleErrorRenderLevel error_render_level, bool enable_check) {
  ffi::ObjectPtr<ConcreteScheduleObj> n =
      ffi::make_object<ConcreteScheduleObj>(ScheduleState(mod, debug_mask, enable_check));
  n->error_render_level_ = error_render_level;
  n->symbol_table_ = {};
  n->analyzer_ = sym::Analyzer();
  n->Seed(seed);
  GlobalVar gv{ffi::UnsafeInit{}};
  if (FindEntryFunc(mod, &gv) != nullptr) {
    n->func_working_on_ = gv;
  } else {
    n->func_working_on_ = std::nullopt;
  }
  return Schedule(ffi::UnsafeInit{}, std::move(n));
}

/******** Copy ********/

/*! \brief Helper class to perform a deep copy of the sref tree */
class ScheduleCopier {
  using TSymbolTable = ConcreteScheduleObj::TSymbolTable;
  template <class K, class V>
  using UMap = std::unordered_map<K, V>;
  template <class K, class V>
  using SMap = std::unordered_map<K, V, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

 public:
  static void Copy(const ConcreteScheduleObj* self, ScheduleState* new_state,
                   TSymbolTable* new_symbol_table) {
    const ScheduleState& src_state = self->state_;
    ScheduleCopier copier(src_state);
    ffi::ObjectPtr<ScheduleStateObj> n = ffi::make_object<ScheduleStateObj>();
    n->mod = src_state->mod;
    n->block_info = copier.Copy(src_state->block_info);
    n->stmt2ref = copier.Copy(src_state->stmt2ref);
    n->debug_mask = src_state->debug_mask;
    n->enable_check = src_state->enable_check;
    *new_state = ScheduleState(ffi::UnsafeInit{}, std::move(n));
    *new_symbol_table = copier.Copy(self->symbol_table_);
  }

 private:
  /*! \brief Create the copier and properly set up the `old2new_` table */
  explicit ScheduleCopier(const ScheduleState& state) {
    // Create SRef tree without parents
    for (const auto& kv : state->stmt2ref) {
      const StmtSRefObj* sref = kv.second.operator->();
      old2new_.emplace(sref,                          // the old StmtSRef
                       StmtSRef(/*stmt=*/sref->stmt,  // the new StmtSRef
                                /*parent=*/nullptr,   // parent is not set yet
                                /*seq_index=*/sref->seq_index));
    }
    // Fill in the parent field
    // Find out the root along the way
    for (auto& kv : old2new_) {
      const StmtSRefObj* parent = kv.first->parent;
      StmtSRef& sref = kv.second;
      sref->parent = parent ? old2new_.at(parent).get() : nullptr;
    }
  }

  /*! \brief Copy StmtSRef */
  StmtSRef Copy(const StmtSRef& sref) { return old2new_.at(sref.operator->()); }

  /*! \brief Copy StmtSRefObj */
  StmtSRef Copy(const StmtSRefObj* sref) {
    if (old2new_.count(sref)) {
      return old2new_.at(sref);
    }
    // Handle expired sref
    return old2new_.emplace(sref, StmtSRef(nullptr, nullptr, -1)).first->second;
  }

  /*! \brief Copy ffi::Array<StmtSRef> */
  ffi::Array<StmtSRef> Copy(const ffi::Array<StmtSRef>& list) {
    ffi::Array<StmtSRef> result;
    result.reserve(list.size());
    for (const StmtSRef& elem : list) {
      result.push_back(Copy(elem));
    }
    return result;
  }

  /*! \brief Copy ffi::Array<Dependency> */
  ffi::Array<Dependency> Copy(const ffi::Array<Dependency>& list) {
    ffi::Array<Dependency> result;
    result.reserve(list.size());
    for (const Dependency& elem : list) {
      result.push_back(Dependency(Copy(elem->src), Copy(elem->dst), elem->kind));
    }
    return result;
  }

  /*! \brief Copy SMap<StmtSRef, ffi::Array<Dependency>> */
  SMap<StmtSRef, ffi::Array<Dependency>> Copy(const SMap<StmtSRef, ffi::Array<Dependency>>& map) {
    SMap<StmtSRef, ffi::Array<Dependency>> result;
    result.reserve(map.size());
    for (const auto& kv : map) {
      result[Copy(kv.first)] = Copy(kv.second);
    }
    return result;
  }

  /*! \brief Copy SMap<TensorVar, ffi::Array<StmtSRef>> */
  SMap<TensorVar, ffi::Array<StmtSRef>> Copy(const SMap<TensorVar, ffi::Array<StmtSRef>>& map) {
    SMap<TensorVar, ffi::Array<StmtSRef>> result;
    result.reserve(map.size());
    for (const auto& kv : map) {
      result[kv.first] = Copy(kv.second);
    }
    return result;
  }

  /*! \brief Copy SMap<StmtSRef, Scope> */
  SMap<StmtSRef, SBlockInfo> Copy(const SMap<StmtSRef, SBlockInfo>& scopes) {
    SMap<StmtSRef, SBlockInfo> result;
    for (const auto& kv : scopes) {
      const StmtSRef& old_sref = kv.first;
      const SBlockInfo& old_info = kv.second;
      SBlockInfo new_info = old_info;
      ffi::ObjectPtr<SBlockScopeObj> scope = ffi::make_object<SBlockScopeObj>();
      scope->src2deps = Copy(old_info.scope->src2deps);
      scope->dst2deps = Copy(old_info.scope->dst2deps);
      scope->tensor_writers = Copy(old_info.scope->tensor_writers);
      new_info.scope = SBlockScope(std::move(scope));
      result[Copy(old_sref)] = std::move(new_info);
    }
    return result;
  }

  /*! \brief Copy the stmt2ref */
  UMap<const StmtObj*, StmtSRef> Copy(const UMap<const StmtObj*, StmtSRef>& stmt2ref) {
    UMap<const StmtObj*, StmtSRef> result;
    result.reserve(stmt2ref.size());
    for (const auto& kv : stmt2ref) {
      const StmtObj* stmt = kv.first;
      const StmtSRef& sref = kv.second;
      result.emplace(stmt, Copy(sref));
    }
    return result;
  }

  /*! \brief Copy the symbol table */
  TSymbolTable Copy(const TSymbolTable& tab) {
    TSymbolTable result;
    for (const auto& kv : tab) {
      ffi::ObjectRef entry = kv.second;
      if (const auto* sref = entry.as<StmtSRefObj>()) {
        entry = Copy(sref);
      }
      result.Set(kv.first, entry);
    }
    return result;
  }

 private:
  std::unordered_map<const StmtSRefObj*, StmtSRef> old2new_;
};

void ConcreteScheduleObj::WorkOn(const ffi::String& func_name) {
  this->func_working_on_ = this->state_->mod->GetGlobalVar(func_name);
}

void ConcreteScheduleObj::Copy(ScheduleState* new_state, TSymbolTable* new_symbol_table) const {
  ScheduleCopier::Copy(this, new_state, new_symbol_table);
  new_state->get()->DebugVerify();
}

Schedule ConcreteScheduleObj::Copy() {
  ffi::ObjectPtr<ConcreteScheduleObj> n = ffi::make_object<ConcreteScheduleObj>(state_);
  n->func_working_on_ = this->func_working_on_;
  n->error_render_level_ = this->error_render_level_;
  ConcreteScheduleObj::Copy(&n->state_, &n->symbol_table_);
  n->analyzer_ = sym::Analyzer();  // new analyzer needed because it is stateful
  n->rand_state_ = ForkSeed();
  return Schedule(ffi::UnsafeInit{}, std::move(n));
}

/*! \brief Macro that guards the beginning of each invocation of TensorIR schedule primitive */
#define TVM_TIR_SCHEDULE_BEGIN() try {
/*! \brief Keep the payload on rendered errors for outer scheduling error handlers. */
#define TVM_TIR_SCHEDULE_THROW(error)                                            \
  ffi::detail::ErrorBuilder(                                                     \
      "ScheduleError", TVMFFIBacktrace(__FILE__, __LINE__, TVM_FFI_FUNC_SIG, 0), \
      TVM_FFI_ALWAYS_LOG_BEFORE_THROW, std::nullopt, (error).extra_context())    \
      .stream()
/*!
 * \brief Macro that pairs with `TVM_TIR_SCHEDULE_BEGIN`, handling potential errors and error
 * message rendering. Already rendered errors retain their original diagnostic.
 * \param level An ScheduleErrorRenderLevel enum, level of error rendering
 * \sa ScheduleErrorRenderLevel
 */
#define TVM_TIR_SCHEDULE_END(primitive, level)                           \
  }                                                                      \
  catch (const ffi::Error& error) {                                      \
    const auto* context = GetScheduleErrorContext(error);                \
    if (context == nullptr || !error.message().empty()) {                \
      throw;                                                             \
    }                                                                    \
    if ((level) == ScheduleErrorRenderLevel::kDetail) {                  \
      TVM_TIR_SCHEDULE_THROW(error) << context->RenderReport(primitive); \
    } else if ((level) == ScheduleErrorRenderLevel::kFast) {             \
      TVM_TIR_SCHEDULE_THROW(error) << context->FastErrorString();       \
    } else if ((level) == ScheduleErrorRenderLevel::kNone) {             \
      TVM_TIR_SCHEDULE_THROW(error) << "(not rendered)";                 \
    }                                                                    \
  }

/******** Schedule: Schedule: Sampling ********/

void ConcreteScheduleObj::Seed(LinearCongruentialEngine::TRandState seed) {
  this->rand_state_ = LinearCongruentialEngine::NormalizeSeed(seed);
}

LinearCongruentialEngine::TRandState ConcreteScheduleObj::ForkSeed() {
  return LinearCongruentialEngine(&rand_state_).ForkSeed();
}

ExprRV ConcreteScheduleObj::SampleCategorical(const ffi::Array<int64_t>& candidates,
                                              const ffi::Array<FloatImm>& probs,
                                              ffi::Optional<int64_t> decision) {
  TVM_TIR_SCHEDULE_BEGIN();
  return CreateRV(s_tir::SampleCategorical(&this->rand_state_, candidates, probs, &decision));
  TVM_TIR_SCHEDULE_END("sample-categorical", this->error_render_level_);
  throw;
}

ffi::Array<ffi::Optional<ExprRV>> ConcreteScheduleObj::SamplePerfectTile(
    const LoopRV& loop_rv, int n, int max_innermost_factor,
    ffi::Optional<ffi::Array<int64_t>> decision) {
  TVM_TIR_SCHEDULE_BEGIN();
  // use None RV object to denotes auto-infer tile factors.
  return CreateOptionalRV(s_tir::SamplePerfectTile(&this->rand_state_, this->GetSRef(loop_rv), n,
                                                   max_innermost_factor, &decision));
  TVM_TIR_SCHEDULE_END("sample-perfect-tile", this->error_render_level_);
  throw;
}

ffi::Array<ExprRV> ConcreteScheduleObj::SamplePartitionedTile(
    const LoopRV& loop_rv, int n, int partition_pos, int innerpart_factor,
    ffi::Optional<ffi::Array<int64_t>> decision) {
  TVM_TIR_SCHEDULE_BEGIN();
  return CreateRV(s_tir::SamplePartitionedTile(&this->rand_state_, this->GetSRef(loop_rv), n,
                                               partition_pos, innerpart_factor, &decision));
  TVM_TIR_SCHEDULE_END("sample-partitioned-tile", this->error_render_level_);
  throw;
}

LoopRV ConcreteScheduleObj::SampleComputeLocation(const SBlockRV& block_rv,
                                                  ffi::Optional<int64_t> decision) {
  TVM_TIR_SCHEDULE_BEGIN();
  return CreateRV<LoopRV>(
      s_tir::SampleComputeLocation(state_, &this->rand_state_, this->GetSRef(block_rv), &decision));
  TVM_TIR_SCHEDULE_END("sample-compute-location", this->error_render_level_);
  throw;
}

/******** Schedule: Get blocks & loops ********/

SBlockRV ConcreteScheduleObj::GetSBlock(const ffi::String& name,
                                        const ffi::Optional<ffi::String>& func_name) {
  class NotSingleResult : public ScheduleErrorContextObj {
   public:
    explicit NotSingleResult(ffi::String name, IRModule mod, const ffi::Array<StmtSRef>& blocks)
        : name_(name), mod_(mod), blocks_{} {
      blocks_.reserve(blocks.size());
      for (const StmtSRef& block_sref : blocks) {
        const SBlockObj* block = TVM_SREF_TO_SBLOCK(block_sref);
        blocks_.push_back(ffi::GetRef<SBlock>(block));
      }
    }

    IRModule mod() const final { return mod_; }
    ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final {
      return {blocks_.begin(), blocks_.end()};
    }

    ffi::String DetailRenderTemplate() const final {
      if (blocks_.empty()) {
        return "Cannot find a block with the name: " + name_;
      } else {
        return "Found  " + std::to_string(blocks_.size()) + " blocks with the name: " + name_;
      }
    }

    ffi::String FastErrorString() const final {
      if (blocks_.empty()) {
        return "ScheduleError: Cannot find a block with the specified name";
      } else {
        return "ScheduleError: Found multiple blocks with the specified name";
      }
    }

    ffi::String name_;
    IRModule mod_;
    ffi::Array<SBlock> blocks_;
  };
  GlobalVar gv{ffi::UnsafeInit{}};
  if (func_name.has_value()) {
    gv = state_->mod->GetGlobalVar(func_name.value());
  } else if (func_working_on_.has_value()) {
    gv = this->func_working_on_.value();
  } else {
    TVM_FFI_THROW(ValueError)
        << "`get_sblock` does not know which function to be working on. Please "
           "specify the function name explicitly, or call `work_on` to specify the function "
           "before using `get_sblock`.";
  }
  ffi::Array<StmtSRef> blocks = s_tir::GetSBlocks(this->state_, name, gv);
  if (blocks.size() != 1) {
    TVM_TIR_SCHEDULE_BEGIN();
    throw MakeScheduleError<NotSingleResult>(name, this->state_->mod, blocks);
    TVM_TIR_SCHEDULE_END("get-block", this->error_render_level_);
  }
  return CreateRV<SBlockRV>(blocks[0]);
}

ffi::Array<LoopRV> ConcreteScheduleObj::GetLoops(const SBlockRV& block_rv) {
  return CreateRV<LoopRV>(s_tir::GetLoops(this->GetSRef(block_rv)));
}

ffi::Array<SBlockRV> ConcreteScheduleObj::GetChildBlocks(const SBlockRV& block_rv) {
  ffi::Array<SBlockRV> result;
  TVM_TIR_SCHEDULE_BEGIN();
  result = CreateRV<SBlockRV>(s_tir::GetChildBlocks(state_, this->GetSRef(block_rv)));
  TVM_TIR_SCHEDULE_END("get-child-blocks", this->error_render_level_);
  this->state_->DebugVerify();
  return result;
}

ffi::Array<SBlockRV> ConcreteScheduleObj::GetChildBlocks(const LoopRV& loop_rv) {
  ffi::Array<SBlockRV> result;
  TVM_TIR_SCHEDULE_BEGIN();
  result = CreateRV<SBlockRV>(s_tir::GetChildBlocks(state_, this->GetSRef(loop_rv)));
  TVM_TIR_SCHEDULE_END("get-child-blocks", this->error_render_level_);
  this->state_->DebugVerify();
  return result;
}

ffi::Array<SBlockRV> ConcreteScheduleObj::GetProducers(const SBlockRV& block_rv) {
  TVM_TIR_SCHEDULE_BEGIN();
  return CreateRV<SBlockRV>(s_tir::GetProducers(state_, this->GetSRef(block_rv)));
  TVM_TIR_SCHEDULE_END("get-producers", this->error_render_level_);
  throw;
}

ffi::Array<SBlockRV> ConcreteScheduleObj::GetConsumers(const SBlockRV& block_rv) {
  TVM_TIR_SCHEDULE_BEGIN();
  return CreateRV<SBlockRV>(s_tir::GetConsumers(state_, this->GetSRef(block_rv)));
  TVM_TIR_SCHEDULE_END("get-consumers", this->error_render_level_);
  throw;
}

ffi::Array<SBlockRV> ConcreteScheduleObj::GetOutputBlocks(const SBlockRV& scope_block_rv) {
  TVM_TIR_SCHEDULE_BEGIN();
  return CreateRV<SBlockRV>(s_tir::GetOutputBlocks(state_, this->GetSRef(scope_block_rv)));
  TVM_TIR_SCHEDULE_END("get-output-blocks", this->error_render_level_);
  throw;
}

/******** Schedule: Transform loops ********/

LoopRV ConcreteScheduleObj::Merge(const ffi::Array<LoopRV>& loop_rvs) {
  TVM_FFI_CHECK(loop_rvs.size() > 1, ValueError) << "'merge' requires at least 2 loop(s)";
  ffi::Array<StmtSRef> loop_srefs = this->GetSRefs(loop_rvs);
  StmtSRef result{ffi::UnsafeInit{}};
  TVM_TIR_SCHEDULE_BEGIN();
  result = s_tir::Merge(state_, loop_srefs);
  TVM_TIR_SCHEDULE_END("merge", this->error_render_level_);
  this->state_->DebugVerify();
  return CreateRV<LoopRV>(result);
}

LoopRV ConcreteScheduleObj::Fuse(const ffi::Array<LoopRV>& loop_rvs, bool preserve_unit_iters) {
  TVM_FFI_CHECK(!loop_rvs.empty(), ValueError) << "'fuse' requires at least 1 loop(s)";
  ffi::Array<StmtSRef> loop_srefs = this->GetSRefs(loop_rvs);
  StmtSRef result{ffi::UnsafeInit{}};
  TVM_TIR_SCHEDULE_BEGIN();
  result = s_tir::Fuse(state_, loop_srefs, preserve_unit_iters);
  TVM_TIR_SCHEDULE_END("fuse", this->error_render_level_);
  this->state_->DebugVerify();
  return CreateRV<LoopRV>(result);
}

class NotSingleInferFactorError : public ScheduleErrorContextObj {
 public:
  explicit NotSingleInferFactorError(IRModule mod) : mod_(mod) {}

  ffi::String FastErrorString() const final {
    return "ScheduleError: only one factor can be specified as -1 or none";
  }

  ffi::String DetailRenderTemplate() const final {
    return "Only one factor can be specified as -1 or none";
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {}; }

  IRModule mod_;
};

class WrongFactorError : public ScheduleErrorContextObj {
 public:
  explicit WrongFactorError(IRModule mod, For loop, bool product)
      : mod_(mod), loop_(std::move(loop)), product_(product) {}

  ffi::String FastErrorString() const final {
    if (product_)
      return "ScheduleError: The product of factors is not larger than or equal to the extent of "
             "loop";
    else
      return "ScheduleError: The sum of factors is larger than or equal to the extent of loop";
  }

  ffi::String DetailRenderTemplate() const final {
    if (product_)
      return "The product of factors is not larger than or equal to the extent of loop {0}";
    else
      return "The sum of factors is larger than or equal to the extent of loop {0}";
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {loop_}; }

  IRModule mod_;
  For loop_;
  bool product_;
};

class NonPositiveFactorError : public ScheduleErrorContextObj {
 public:
  explicit NonPositiveFactorError(IRModule mod, ffi::BigInt factor, size_t idx)
      : mod_(std::move(mod)), factor_(factor), idx_(idx) {}

  ffi::String FastErrorString() const final {
    return "ScheduleError: All the constant factors are required to be positive. However, some "
           "constant input factor is zero or negative.";
  }
  ffi::String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "All the constant factors are required to be positive. However, the factor at position "
       << idx_ << " is " << factor_;
    return os.str();
  }
  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {}; }

 private:
  IRModule mod_;
  ffi::BigInt factor_;
  size_t idx_;
};

ffi::Array<LoopRV> ConcreteScheduleObj::Split(const LoopRV& loop_rv,
                                              const ffi::Array<ffi::Optional<ExprRV>>& factor_rvs,
                                              bool preserve_unit_iters, bool disable_predication) {
  // Prepare for the splitting
  StmtSRef loop_sref = this->GetSRef(loop_rv);
  const ForObj* loop = TVM_SREF_TO_FOR(loop_sref);
  ffi::Array<PrimExpr> factors;
  factors.reserve(factor_rvs.size());
  int infer_index = -1;
  PrimExpr tot_length = 1;
  ffi::Array<StmtSRef> results;
  TVM_TIR_SCHEDULE_BEGIN();
  // infer factor if needed and check validity of factors
  for (size_t i = 0; i < factor_rvs.size(); i++) {
    if (!factor_rvs[i].has_value()) {
      factors.push_back(IntImm::Int32(-1));
      if (infer_index != -1) {
        throw MakeScheduleError<NotSingleInferFactorError>(state_->mod);
      }
      infer_index = i;
    } else {
      PrimExpr factor = this->Get(factor_rvs[i].value());
      if (IsConstInt(factor) && !IsPositiveConst(factor)) {
        throw MakeScheduleError<NonPositiveFactorError>(state_->mod, factor.as<IntImmObj>()->value,
                                                        i);
      }
      if (factor.ty().bits() > Range::GetExtentFromBeginEnd(loop->begin, loop->end).ty().bits()) {
        factor = cast(factor, Range::GetExtentFromBeginEnd(loop->begin, loop->end).ty());
      }
      factors.push_back(factor);
      tot_length *= factor;
    }
  }
  if (infer_index != -1) {
    factors.Set(
        infer_index,
        this->analyzer_->Simplify(floordiv(
            Range::GetExtentFromBeginEnd(loop->begin, loop->end) + tot_length - 1, tot_length)));
  } else if (!this->analyzer_->CanProve(tot_length >=
                                        Range::GetExtentFromBeginEnd(loop->begin, loop->end))) {
    throw MakeScheduleError<WrongFactorError>(state_->mod, ffi::GetRef<For>(loop), true);
  }
  results = s_tir::Split(state_, loop_sref, factors, preserve_unit_iters, disable_predication);
  TVM_TIR_SCHEDULE_END("split", this->error_render_level_);
  this->state_->DebugVerify();
  return CreateRV<LoopRV>(results);
}

ffi::Array<LoopRV> ConcreteScheduleObj::LoopPartition(
    const LoopRV& loop_rv, const ffi::Array<ffi::Optional<ExprRV>>& factor_rvs,
    bool preserve_unit_iters) {
  class SymbolicShapeError : public ScheduleErrorContextObj {
   public:
    explicit SymbolicShapeError(IRModule mod, For loop) : mod_(mod), loop_(std::move(loop)) {}

    ffi::String FastErrorString() const final {
      return "ScheduleError: The min and extent values of the loop are required to be known at "
             "compile time. However, dynamic shape has been detected.";
    }

    ffi::String DetailRenderTemplate() const final {
      return "Detected dynamic shape in either min or extent of a loop {0}";
    }

    IRModule mod() const final { return mod_; }
    ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {loop_}; }

    IRModule mod_;
    For loop_;
  };

  // Prepare for the loop_partitioning
  StmtSRef loop_sref = this->GetSRef(loop_rv);
  const ForObj* loop = TVM_SREF_TO_FOR(loop_sref);
  ffi::Array<PrimExpr> factors;
  factors.reserve(factor_rvs.size());
  int infer_index = -1;
  PrimExpr tot_length = 0;
  ffi::Array<StmtSRef> results;
  TVM_TIR_SCHEDULE_BEGIN();
  if (!IsConstNumber(loop->begin) ||
      !IsConstNumber(Range::GetExtentFromBeginEnd(loop->begin, loop->end))) {
    throw MakeScheduleError<SymbolicShapeError>(state_->mod, ffi::GetRef<For>(loop));
  }
  // infer factor if needed and check validity of factors
  for (size_t i = 0; i < factor_rvs.size(); i++) {
    if (!factor_rvs[i].has_value()) {
      factors.push_back(IntImm::Int32(-1));
      if (infer_index != -1) {
        throw MakeScheduleError<NotSingleInferFactorError>(state_->mod);
      }
      infer_index = i;
    } else {
      PrimExpr factor = this->Get(factor_rvs[i].value());
      if (IsConstInt(factor) && !IsPositiveConst(factor)) {
        throw MakeScheduleError<NonPositiveFactorError>(state_->mod, factor.as<IntImmObj>()->value,
                                                        i);
      }
      if (factor.ty().bits() > Range::GetExtentFromBeginEnd(loop->begin, loop->end).ty().bits()) {
        factor = cast(factor, Range::GetExtentFromBeginEnd(loop->begin, loop->end).ty());
      }
      factors.push_back(factor);
      tot_length += factor;
    }
  }
  if (this->analyzer_->CanProve(tot_length >=
                                Range::GetExtentFromBeginEnd(loop->begin, loop->end))) {
    throw MakeScheduleError<WrongFactorError>(state_->mod, ffi::GetRef<For>(loop), false);
  }
  if (infer_index != -1) {
    // if there is a 'None' in the factor list, 'None' becomes the difference between the extent and
    // the sum of the factors excluding 'None' specified in the partition directive.
    factors.Set(infer_index, Range::GetExtentFromBeginEnd(loop->begin, loop->end) - tot_length);
  }
  for (size_t i = 1; i < factor_rvs.size(); i++) {
    factors.Set(i, factors[i] + factors[i - 1]);
  }
  if (infer_index == -1) {
    factors.push_back(Range::GetExtentFromBeginEnd(loop->begin, loop->end));
  }
  results = s_tir::LoopPartition(state_, loop_sref, factors, preserve_unit_iters);
  TVM_TIR_SCHEDULE_END("loop_partition", this->error_render_level_);
  this->state_->DebugVerify();
  return CreateRV<LoopRV>(results);
}

void ConcreteScheduleObj::Reorder(const ffi::Array<LoopRV>& ordered_loop_rvs) {
  TVM_TIR_SCHEDULE_BEGIN();
  s_tir::Reorder(state_, GetSRefs(ordered_loop_rvs));
  TVM_TIR_SCHEDULE_END("reorder", this->error_render_level_);
  this->state_->DebugVerify();
}

void ConcreteScheduleObj::ReorderBlockIterVar(const SBlockRV& block_rv,
                                              const ffi::Array<int64_t> new_order) {
  TVM_TIR_SCHEDULE_BEGIN();
  s_tir::ReorderBlockIterVar(state_, GetSRef(block_rv), new_order);
  TVM_TIR_SCHEDULE_END("reorder_block_iter_var", this->error_render_level_);
  this->state_->DebugVerify();
}

LoopRV ConcreteScheduleObj::AddUnitLoop(const SBlockRV& block_rv) {
  LoopRV result{ffi::UnsafeInit()};
  TVM_TIR_SCHEDULE_BEGIN();
  result = CreateRV<LoopRV>(s_tir::AddUnitLoop(state_, GetSRef(block_rv)));
  TVM_TIR_SCHEDULE_END("add-unit-loop", this->error_render_level_);
  this->state_->DebugVerify();
  return result;
}

LoopRV ConcreteScheduleObj::AddUnitLoop(const LoopRV& loop_rv) {
  LoopRV result{ffi::UnsafeInit()};
  TVM_TIR_SCHEDULE_BEGIN();
  result = CreateRV<LoopRV>(s_tir::AddUnitLoop(state_, GetSRef(loop_rv)));
  TVM_TIR_SCHEDULE_END("add-unit-loop", this->error_render_level_);
  this->state_->DebugVerify();
  return result;
}

/******** Schedule: Manipulate ForKind ********/

void ConcreteScheduleObj::Parallel(const LoopRV& loop_rv) {
  TVM_TIR_SCHEDULE_BEGIN();
  s_tir::Parallel(state_, this->GetSRef(loop_rv));
  this->state_->DebugVerify();
  TVM_TIR_SCHEDULE_END("parallel", this->error_render_level_);
}

void ConcreteScheduleObj::Vectorize(const LoopRV& loop_rv) {
  TVM_TIR_SCHEDULE_BEGIN();
  s_tir::Vectorize(state_, this->GetSRef(loop_rv));
  this->state_->DebugVerify();
  TVM_TIR_SCHEDULE_END("vectorize", this->error_render_level_);
}

void ConcreteScheduleObj::Bind(const LoopRV& loop_rv, const ffi::String& thread_axis) {
  if (thread_axis == "vthread") {
    LOG(WARNING) << "`vthread` is legacy behavior and is going to be deprecated. Please use "
                    "`vthread.x`, `vthread.y` and `vthread.z` instead";
  }
  TVM_TIR_SCHEDULE_BEGIN();
  s_tir::Bind(state_, this->GetSRef(loop_rv), thread_axis);
  this->state_->DebugVerify();
  TVM_TIR_SCHEDULE_END("bind", this->error_render_level_);
}

void ConcreteScheduleObj::Unroll(const LoopRV& loop_rv) {
  TVM_TIR_SCHEDULE_BEGIN();
  s_tir::Unroll(state_, this->GetSRef(loop_rv));
  this->state_->DebugVerify();
  TVM_TIR_SCHEDULE_END("unroll", this->error_render_level_);
}

/******** Schedule: Insert cache stages ********/

SBlockRV ConcreteScheduleObj::CacheRead(const SBlockRV& block_rv, int read_tensor_index,
                                        const ffi::String& storage_scope,
                                        const ffi::Array<SBlockRV> consumer_blocks) {
  StmtSRef result{ffi::UnsafeInit{}};
  // Create a new array of SRefs from the consumer block list.
  ffi::Array<StmtSRef> consumer_block_refs = {};
  for (SBlockRV block : consumer_blocks) {
    consumer_block_refs.push_back(this->GetSRef(block));
  }
  TVM_TIR_SCHEDULE_BEGIN();
  result = s_tir::CacheRead(state_, this->GetSRef(block_rv), read_tensor_index, storage_scope,
                            consumer_block_refs);
  TVM_TIR_SCHEDULE_END("cache-read", this->error_render_level_);
  this->state_->DebugVerify();
  return CreateRV<SBlockRV>(result);
}

SBlockRV ConcreteScheduleObj::CacheWrite(const SBlockRV& block_rv, int write_tensor_index,
                                         const ffi::String& storage_scope,
                                         const ffi::Array<SBlockRV> consumer_blocks) {
  StmtSRef result{ffi::UnsafeInit{}};
  // Create a new array of SRefs from the consumer block list.
  ffi::Array<StmtSRef> consumer_block_refs = {};
  for (SBlockRV block : consumer_blocks) {
    consumer_block_refs.push_back(this->GetSRef(block));
  }
  TVM_TIR_SCHEDULE_BEGIN();
  result = s_tir::CacheWrite(state_, this->GetSRef(block_rv), write_tensor_index, storage_scope,
                             consumer_block_refs);
  TVM_TIR_SCHEDULE_END("cache-write", this->error_render_level_);
  this->state_->DebugVerify();
  return CreateRV<SBlockRV>(result);
}

SBlockRV ConcreteScheduleObj::ReindexCacheRead(const SBlockRV& block_rv, int read_tensor_index,
                                               const ffi::String& storage_scope,
                                               const IndexMap& index_map) {
  StmtSRef result{ffi::UnsafeInit{}};
  TVM_TIR_SCHEDULE_BEGIN();
  result = s_tir::ReindexCacheRead(state_, this->GetSRef(block_rv), read_tensor_index,
                                   storage_scope, index_map);
  TVM_TIR_SCHEDULE_END("reverse-cache-read", this->error_render_level_);
  this->state_->DebugVerify();
  return CreateRV<SBlockRV>(result);
}

SBlockRV ConcreteScheduleObj::ReindexCacheWrite(const SBlockRV& block_rv, int write_tensor_index,
                                                const ffi::String& storage_scope,
                                                const IndexMap& index_map) {
  StmtSRef result{ffi::UnsafeInit{}};
  TVM_TIR_SCHEDULE_BEGIN();
  result = s_tir::ReindexCacheWrite(state_, this->GetSRef(block_rv), write_tensor_index,
                                    storage_scope, index_map);
  TVM_TIR_SCHEDULE_END("reverse-cache-write", this->error_render_level_);
  this->state_->DebugVerify();
  return CreateRV<SBlockRV>(result);
}

ffi::Array<SBlockRV> ConcreteScheduleObj::CacheInplace(const SBlockRV& block_rv,
                                                       int write_tensor_index,
                                                       const ffi::String& storage_scope) {
  ffi::Array<StmtSRef> results;
  TVM_TIR_SCHEDULE_BEGIN();
  results = s_tir::CacheInplace(state_, this->GetSRef(block_rv), write_tensor_index, storage_scope);
  TVM_TIR_SCHEDULE_END("cache-tensor", this->error_render_level_);
  this->state_->DebugVerify();
  ffi::Array<SBlockRV> return_blocks;
  return_blocks.push_back(CreateRV<SBlockRV>(results[0]));
  return_blocks.push_back(CreateRV<SBlockRV>(results[1]));
  return return_blocks;
}

ffi::Array<SBlockRV> ConcreteScheduleObj::CacheIndex(const SBlockRV& block_rv,
                                                     const ffi::String& storage_scope,
                                                     int cse_thresh) {
  ffi::Array<StmtSRef> result;
  TVM_TIR_SCHEDULE_BEGIN();
  result = s_tir::CacheIndex(state_, this->GetSRef(block_rv), storage_scope, cse_thresh);
  TVM_TIR_SCHEDULE_END("cache-index", this->error_render_level_);
  this->state_->DebugVerify();
  ffi::Array<SBlockRV> return_blocks;
  for (const StmtSRef& blockrv : result) {
    return_blocks.push_back(CreateRV<SBlockRV>(blockrv));
  }
  return return_blocks;
}

SBlockRV ConcreteScheduleObj::ReIndex(const SBlockRV& block_rv, int tensor_index,
                                      TensorIndexType tensor_index_type) {
  StmtSRef result{ffi::UnsafeInit{}};
  TVM_TIR_SCHEDULE_BEGIN();
  result = s_tir::ReIndex(state_, this->GetSRef(block_rv), tensor_index, tensor_index_type);
  TVM_TIR_SCHEDULE_END("reindex", this->error_render_level_);
  this->state_->DebugVerify();
  return CreateRV<SBlockRV>(result);
}

/******** Schedule: Data movement ********/

SBlockRV ConcreteScheduleObj::ReadAt(const LoopRV& loop_rv, const SBlockRV& block_rv,
                                     int read_tensor_index, const ffi::String& storage_scope) {
  StmtSRef result{ffi::UnsafeInit{}};
  TVM_TIR_SCHEDULE_BEGIN();
  result = s_tir::ReadAt(state_, this->GetSRef(loop_rv), this->GetSRef(block_rv), read_tensor_index,
                         storage_scope);
  TVM_TIR_SCHEDULE_END("read-at", this->error_render_level_);
  this->state_->DebugVerify();
  return CreateRV<SBlockRV>(result);
}

SBlockRV ConcreteScheduleObj::WriteAt(const LoopRV& loop_rv, const SBlockRV& block_rv,
                                      int write_tensor_index, const ffi::String& storage_scope) {
  StmtSRef result{ffi::UnsafeInit{}};
  TVM_TIR_SCHEDULE_BEGIN();
  result = s_tir::WriteAt(state_, this->GetSRef(loop_rv), this->GetSRef(block_rv),
                          write_tensor_index, storage_scope);
  TVM_TIR_SCHEDULE_END("write-at", this->error_render_level_);
  this->state_->DebugVerify();
  return CreateRV<SBlockRV>(result);
}

/******** Schedule: Compute location ********/

void ConcreteScheduleObj::ComputeAt(const SBlockRV& block_rv, const LoopRV& loop_rv,
                                    bool preserve_unit_loops, int index) {
  static StmtSRef inline_mark = StmtSRef::InlineMark();
  static StmtSRef root_mark = StmtSRef::RootMark();
  StmtSRef loop_sref = this->GetSRef(loop_rv);
  if (loop_sref.same_as(root_mark)) {
    // do nothing
  } else if (loop_sref.same_as(inline_mark)) {
    TVM_TIR_SCHEDULE_BEGIN();
    s_tir::ComputeInline(state_, this->GetSRef(block_rv));
    TVM_TIR_SCHEDULE_END("compute-at", this->error_render_level_);
  } else {
    TVM_TIR_SCHEDULE_BEGIN();
    s_tir::ComputeAt(state_, this->GetSRef(block_rv), loop_sref, preserve_unit_loops, index);
    TVM_TIR_SCHEDULE_END("compute-at", this->error_render_level_);
  }
  this->state_->DebugVerify();
}

void ConcreteScheduleObj::ReverseComputeAt(const SBlockRV& block_rv, const LoopRV& loop_rv,
                                           bool preserve_unit_loops, int index) {
  static StmtSRef inline_mark = StmtSRef::InlineMark();
  static StmtSRef root_mark = StmtSRef::RootMark();
  StmtSRef loop_sref = this->GetSRef(loop_rv);
  if (loop_sref.same_as(root_mark)) {
    // do nothing
  } else if (loop_sref.same_as(inline_mark)) {
    TVM_TIR_SCHEDULE_BEGIN();
    s_tir::ReverseComputeInline(state_, this->GetSRef(block_rv));
    TVM_TIR_SCHEDULE_END("reverse-compute-at", this->error_render_level_);
  } else {
    TVM_TIR_SCHEDULE_BEGIN();
    s_tir::ReverseComputeAt(state_, this->GetSRef(block_rv), loop_sref, preserve_unit_loops, index);
    TVM_TIR_SCHEDULE_END("reverse-compute-at", this->error_render_level_);
  }
  this->state_->DebugVerify();
}

void ConcreteScheduleObj::ComputeInline(const SBlockRV& block_rv) {
  TVM_TIR_SCHEDULE_BEGIN();
  s_tir::ComputeInline(state_, this->GetSRef(block_rv));
  TVM_TIR_SCHEDULE_END("compute-inline", this->error_render_level_);
  this->state_->DebugVerify();
}

void ConcreteScheduleObj::ReverseComputeInline(const SBlockRV& block_rv) {
  TVM_TIR_SCHEDULE_BEGIN();
  s_tir::ReverseComputeInline(state_, this->GetSRef(block_rv));
  TVM_TIR_SCHEDULE_END("reverse-compute-inline", this->error_render_level_);
  this->state_->DebugVerify();
}

void ConcreteScheduleObj::FuseReductionEpilogue(const SBlockRV& reduction_block_rv,
                                                const SBlockRV& epilogue_block_rv) {
  TVM_TIR_SCHEDULE_BEGIN();
  s_tir::FuseReductionEpilogue(state_, this->GetSRef(reduction_block_rv),
                               this->GetSRef(epilogue_block_rv));
  TVM_TIR_SCHEDULE_END("fuse-reduction-epilogue", this->error_render_level_);
  this->state_->DebugVerify();
}

/******** Schedule: SBlock Annotation ********/

void ConcreteScheduleObj::StorageAlign(const SBlockRV& block_rv, int tensor_index, int axis,
                                       int factor, int offset) {
  TVM_TIR_SCHEDULE_BEGIN();
  s_tir::StorageAlign(state_, this->GetSRef(block_rv), tensor_index, axis, factor, offset);
  TVM_TIR_SCHEDULE_END("storage-align", this->error_render_level_);
  this->state_->DebugVerify();
}

void ConcreteScheduleObj::SetScope(const SBlockRV& block_rv, int tensor_index,
                                   const ffi::String& storage_scope) {
  TVM_TIR_SCHEDULE_BEGIN();
  s_tir::SetScope(state_, this->GetSRef(block_rv), tensor_index, storage_scope);
  TVM_TIR_SCHEDULE_END("set-scope", this->error_render_level_);
  this->state_->DebugVerify();
}

void ConcreteScheduleObj::UnsafeSetDType(const SBlockRV& block_rv, int tensor_index,
                                         const ffi::String& dtype) {
  TVM_TIR_SCHEDULE_BEGIN();
  s_tir::UnsafeSetDType(state_, this->GetSRef(block_rv), tensor_index, dtype);
  TVM_TIR_SCHEDULE_END("set-dtype", this->error_render_level_);
  this->state_->DebugVerify();
}

/******** Schedule: Reduction ********/

SBlockRV ConcreteScheduleObj::DecomposeReduction(const SBlockRV& block_rv, const LoopRV& loop_rv) {
  StmtSRef result{ffi::UnsafeInit{}};
  TVM_TIR_SCHEDULE_BEGIN();
  result = s_tir::DecomposeReduction(state_, this->GetSRef(block_rv), this->GetSRef(loop_rv));
  TVM_TIR_SCHEDULE_END("decompose-reduction", this->error_render_level_);
  this->state_->DebugVerify();
  return CreateRV<SBlockRV>(result);
}

SBlockRV ConcreteScheduleObj::RFactor(const LoopRV& loop_rv, int factor_axis) {
  StmtSRef result{ffi::UnsafeInit{}};
  TVM_TIR_SCHEDULE_BEGIN();
  result = s_tir::RFactor(state_, this->GetSRef(loop_rv), factor_axis);
  TVM_TIR_SCHEDULE_END("rfactor", this->error_render_level_);
  this->state_->DebugVerify();
  return CreateRV<SBlockRV>(result);
}

/******** Schedule: Blockize & Tensorize ********/
SBlockRV ConcreteScheduleObj::Blockize(const LoopRV& loop_rv, bool preserve_unit_iters) {
  StmtSRef result{ffi::UnsafeInit{}};
  TVM_TIR_SCHEDULE_BEGIN();
  result = s_tir::Blockize(state_, this->GetSRef(loop_rv), preserve_unit_iters);
  this->state_->DebugVerify();
  TVM_TIR_SCHEDULE_END("blockize", this->error_render_level_);
  return CreateRV<SBlockRV>(result);
}

SBlockRV ConcreteScheduleObj::Blockize(const ffi::Array<SBlockRV>& blocks,
                                       bool preserve_unit_iters) {
  StmtSRef result{ffi::UnsafeInit{}};
  TVM_TIR_SCHEDULE_BEGIN();
  result = s_tir::Blockize(state_, this->GetSRefs(blocks), preserve_unit_iters);
  this->state_->DebugVerify();
  TVM_TIR_SCHEDULE_END("blockize", this->error_render_level_);
  return CreateRV<SBlockRV>(result);
}

void ConcreteScheduleObj::Tensorize(const LoopRV& loop_rv, const ffi::String& intrin,
                                    bool preserve_unit_iters) {
  TVM_TIR_SCHEDULE_BEGIN();
  s_tir::Tensorize(state_, this->GetSRef(loop_rv), TensorIntrin::Get(intrin).value(),
                   preserve_unit_iters);
  this->state_->DebugVerify();
  TVM_TIR_SCHEDULE_END("tensorize", this->error_render_level_);
}

void ConcreteScheduleObj::Tensorize(const SBlockRV& block_rv, const ffi::String& intrin,
                                    bool preserve_unit_iters) {
  TVM_TIR_SCHEDULE_BEGIN();
  s_tir::Tensorize(state_, this->GetSRef(block_rv), TensorIntrin::Get(intrin).value(),
                   preserve_unit_iters);
  this->state_->DebugVerify();
  TVM_TIR_SCHEDULE_END("tensorize", this->error_render_level_);
}

/******** Schedule: Annotation ********/

Any ConcreteScheduleObj::CheckAndGetAnnotationValue(const ffi::Any& ann_val) {
  if (auto opt_str = ann_val.try_cast<ffi::String>()) {
    return *std::move(opt_str);
  }
  if (ann_val.type_index() < ffi::TypeIndex::kTVMFFISmallStr) {
    return ann_val;
  }
  // prefer to return int/float literals for annotations
  if (auto opt_intimm = ann_val.try_cast<IntImm>()) {
    return (*std::move(opt_intimm))->value;
  }
  if (auto opt_floatimm = ann_val.try_cast<FloatImm>()) {
    return (*std::move(opt_floatimm))->value;
  }

  if (auto expr = ann_val.as<PrimExpr>()) {
    TVM_FFI_CHECK(!expr.value().as<StringImmObj>(), TypeError)
        << "ffi::String is expected, but gets StringImm";
    auto res_expr = this->Get(expr.value());
    // prefer to return int/float literals for annotations
    if (auto opt_intimm = res_expr.as<IntImm>()) {
      return (*std::move(opt_intimm))->value;
    }
    if (auto opt_floatimm = res_expr.as<FloatImm>()) {
      return (*std::move(opt_floatimm))->value;
    }
    return res_expr;
  }
  if (const auto* arr = ann_val.as<ffi::ArrayObj>()) {
    ffi::Array<Any> result;
    result.reserve(arr->size());
    for (size_t i = 0; i < arr->size(); i++) {
      result.push_back(CheckAndGetAnnotationValue(arr->at(i)));
    }
    return result;
  }
  if (const auto* dict = ann_val.as<ffi::MapObj>()) {
    ffi::Map<ffi::String, ffi::Any> result;
    for (auto it = dict->begin(); it != dict->end(); ++it) {
      const auto& key = it->first;
      auto value = CheckAndGetAnnotationValue(it->second);
      if (const StringImmObj* imm = key.as<StringImmObj>()) {
        result.Set(imm->value, value);
      } else if (auto opt_str = key.try_cast<ffi::String>()) {
        result.Set(opt_str.value(), value);
      } else {
        TVM_FFI_THROW(TypeError) << "annotation dict key expect to be ffi::String or StringImm";
      }
    }
    return result;
  }
  TVM_FFI_THROW(TypeError)
      << "Only strings, integers, floats, ExprRVs and Arrays are supported for now, but "
      << "gets: " << ann_val.GetTypeKey();
  TVM_FFI_UNREACHABLE();
}

void ConcreteScheduleObj::Annotate(const LoopRV& loop_rv, const ffi::String& ann_key,
                                   const Any& ann_val) {
  TVM_TIR_SCHEDULE_BEGIN();
  s_tir::Annotate(state_, this->GetSRef(loop_rv), ann_key,
                  this->CheckAndGetAnnotationValue(ann_val));
  this->state_->DebugVerify();
  TVM_TIR_SCHEDULE_END("annotate", this->error_render_level_);
}

void ConcreteScheduleObj::Unannotate(const LoopRV& loop_rv, const ffi::String& ann_key) {
  TVM_TIR_SCHEDULE_BEGIN();
  s_tir::Unannotate(state_, this->GetSRef(loop_rv), ann_key);
  this->state_->DebugVerify();
  TVM_TIR_SCHEDULE_END("unannotate", this->error_render_level_);
}

void ConcreteScheduleObj::Annotate(const SBlockRV& block_rv, const ffi::String& ann_key,
                                   const Any& ann_val) {
  TVM_TIR_SCHEDULE_BEGIN();
  s_tir::Annotate(state_, this->GetSRef(block_rv), ann_key,
                  this->CheckAndGetAnnotationValue(ann_val));
  this->state_->DebugVerify();
  TVM_TIR_SCHEDULE_END("annotate", this->error_render_level_);
}

void ConcreteScheduleObj::Unannotate(const SBlockRV& block_rv, const ffi::String& ann_key) {
  TVM_TIR_SCHEDULE_BEGIN();
  s_tir::Unannotate(state_, this->GetSRef(block_rv), ann_key);
  this->state_->DebugVerify();
  TVM_TIR_SCHEDULE_END("unannotate", this->error_render_level_);
}

/******** Schedule: Layout transformation ********/
void ConcreteScheduleObj::TransformLayout(const SBlockRV& block_rv, int tensor_index,
                                          TensorIndexType tensor_index_type,
                                          const IndexMap& index_map,
                                          const ffi::Optional<IndexMap>& pad_value,
                                          bool assume_injective_transform) {
  TVM_TIR_SCHEDULE_BEGIN();
  auto f_substitute = [this](
                          const Var& var,
                          TVMFFIDefRegionKind kind) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (kind != kTVMFFIDefRegionKindNone) return ffi::Unchanged();
    if (auto repl = symbol_table_.Get(var)) {
      return ffi::Any((*std::move(repl)).as_or_throw<PrimExpr>());
    }
    return ffi::Unchanged();
  };
  auto new_index_map = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(index_map, f_substitute)
                           .as_or_throw<IndexMap>();
  s_tir::TransformLayout(state_, this->GetSRef(block_rv), tensor_index, tensor_index_type,
                         new_index_map, pad_value, assume_injective_transform);
  this->state_->DebugVerify();
  TVM_TIR_SCHEDULE_END("transform_layout", this->error_render_level_);
}

void ConcreteScheduleObj::TransformBlockLayout(const SBlockRV& block_rv,
                                               const IndexMap& index_map) {
  TVM_TIR_SCHEDULE_BEGIN();
  s_tir::TransformBlockLayout(state_, this->GetSRef(block_rv), index_map);
  this->state_->DebugVerify();
  TVM_TIR_SCHEDULE_END("transform_block_layout", this->error_render_level_);
}

/******** Schedule: Padding ********/

SBlockRV ConcreteScheduleObj::DecomposePadding(const SBlockRV& block_rv, const LoopRV& loop_rv) {
  StmtSRef result{ffi::UnsafeInit{}};
  TVM_TIR_SCHEDULE_BEGIN();
  result = s_tir::DecomposePadding(state_, this->GetSRef(block_rv), this->GetSRef(loop_rv));
  TVM_TIR_SCHEDULE_END("decompose-padding", this->error_render_level_);
  this->state_->DebugVerify();
  return CreateRV<SBlockRV>(result);
}

void ConcreteScheduleObj::PadEinsum(const SBlockRV& block_rv, const ffi::Array<int64_t>& padding) {
  TVM_TIR_SCHEDULE_BEGIN();
  s_tir::PadEinsum(state_, this->GetSRef(block_rv), padding);
  TVM_TIR_SCHEDULE_END("pad-einsum", this->error_render_level_);
  this->state_->DebugVerify();
}

/******** Schedule: Tensor Transformation ********/

void ConcreteScheduleObj::RollingBuffer(const SBlockRV& block_rv, int write_tensor_index) {
  TVM_TIR_SCHEDULE_BEGIN();
  s_tir::RollingBuffer(state_, this->GetSRef(block_rv), write_tensor_index);
  TVM_TIR_SCHEDULE_END("rolling-tensor", this->error_render_level_);
  this->state_->DebugVerify();
}

/******** Schedule: Misc ********/

void ConcreteScheduleObj::UnsafeHideTensorAccess(const SBlockRV& block_rv,
                                                 const ffi::String& tensor_type,
                                                 const ffi::Array<int64_t>& tensor_index_array) {
  TVM_TIR_SCHEDULE_BEGIN();
  s_tir::UnsafeHideTensorAccess(state_, this->GetSRef(block_rv), tensor_type, tensor_index_array);
  TVM_TIR_SCHEDULE_END("hide-tensor-access", this->error_render_level_);
  this->state_->DebugVerify();
}

void ConcreteScheduleObj::AnnotateTensorAccess(const SBlockRV& block_rv, int tensor_index,
                                               TensorIndexType tensor_index_type,
                                               const IndexMap& index_map) {
  TVM_TIR_SCHEDULE_BEGIN();
  s_tir::AnnotateTensorAccess(state_, this->GetSRef(block_rv), tensor_index, tensor_index_type,
                              index_map);
  TVM_TIR_SCHEDULE_END("annotate-tensor-access", this->error_render_level_);
  this->state_->DebugVerify();
}

}  // namespace s_tir
}  // namespace tvm
