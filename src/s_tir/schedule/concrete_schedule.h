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
#ifndef TVM_S_TIR_SCHEDULE_CONCRETE_SCHEDULE_H_
#define TVM_S_TIR_SCHEDULE_CONCRETE_SCHEDULE_H_

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/s_tir/stmt.h>

#include <memory>
#include <utility>
#include <vector>

#include "./utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

class ConcreteScheduleObj : public ScheduleObj {
  friend class Schedule;
  friend class ScheduleCopier;

 public:
  explicit ConcreteScheduleObj(ScheduleState state) : state_(std::move(state)) {}

  using TSymbolTable = ffi::Map<ffi::ObjectRef, ffi::ObjectRef>;

 protected:
  /*! \brief The internal state of scheduling */
  ScheduleState state_;
  /*! \brief The function to be worked on. */
  ffi::Optional<GlobalVar> func_working_on_;
  /*! \brief The level of error rendering */
  ScheduleErrorRenderLevel error_render_level_;
  /*! \brief A symbol table that maps random variables to concrete StmtSRef/Integers */
  TSymbolTable symbol_table_;
  /*! \brief A persistent stateless arithmetic analyzer. */
  sym::Analyzer analyzer_;
  /*! \brief The value of random state for sampling. */
  LinearCongruentialEngine::TRandState rand_state_;

 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ConcreteScheduleObj>();
  }

  virtual ~ConcreteScheduleObj() = default;

 public:
  ScheduleState state() const final { return state_; }
  ffi::Optional<Trace> trace() const override { return std::nullopt; }
  ffi::Optional<GlobalVar> func_working_on() const final { return func_working_on_; }
  void WorkOn(const ffi::String& func_name) final;
  Schedule Copy() override;
  void Seed(LinearCongruentialEngine::TRandState seed) final;
  LinearCongruentialEngine::TRandState ForkSeed() final;

 public:
  /******** Lookup random variables ********/
  inline SBlock Get(const SBlockRV& block_rv) const final;
  inline For Get(const LoopRV& loop_rv) const final;
  inline PrimExpr Get(const ExprRV& expr_rv) const final;
  inline StmtSRef GetSRef(const SBlockRV& block_rv) const final;
  inline StmtSRef GetSRef(const LoopRV& loop_rv) const final;
  inline bool HasBlock(const SBlockRV& block_rv) const final;
  inline ffi::Array<StmtSRef> GetSRefs(const ffi::Array<SBlockRV>& rvs) const;
  inline ffi::Array<StmtSRef> GetSRefs(const ffi::Array<LoopRV>& rvs) const;
  void RemoveRV(const SBlockRV& block_rv) final { RemoveFromSymbolTable(block_rv); }
  void RemoveRV(const LoopRV& loop_rv) final { RemoveFromSymbolTable(loop_rv); }
  void RemoveRV(const ExprRV& expr_rv) final { RemoveFromSymbolTable(expr_rv); }
  using ScheduleObj::GetSRef;

 public:
  /******** Schedule: Sampling ********/
  ExprRV SampleCategorical(const ffi::Array<int64_t>& candidates, const ffi::Array<FloatImm>& probs,
                           ffi::Optional<int64_t> decision = std::nullopt) override;
  ffi::Array<ffi::Optional<ExprRV>> SamplePerfectTile(
      const LoopRV& loop_rv, int n, int max_innermost_factor,
      ffi::Optional<ffi::Array<int64_t>> decision = std::nullopt) override;
  ffi::Array<ExprRV> SamplePartitionedTile(
      const LoopRV& loop_rv, int n, int partition_pos, int innerpart_factor,
      ffi::Optional<ffi::Array<int64_t>> decision = std::nullopt) override;
  LoopRV SampleComputeLocation(const SBlockRV& block_rv,
                               ffi::Optional<int64_t> decision = std::nullopt) override;
  /******** Schedule: Get blocks & loops ********/
  SBlockRV GetSBlock(const ffi::String& name, const ffi::Optional<ffi::String>& func_name) override;
  ffi::Array<LoopRV> GetLoops(const SBlockRV& block_rv) override;
  ffi::Array<SBlockRV> GetChildBlocks(const SBlockRV& block_rv) override;
  ffi::Array<SBlockRV> GetChildBlocks(const LoopRV& loop_rv) override;
  ffi::Array<SBlockRV> GetProducers(const SBlockRV& block_rv) override;
  ffi::Array<SBlockRV> GetConsumers(const SBlockRV& block_rv) override;
  ffi::Array<SBlockRV> GetOutputBlocks(const SBlockRV& scope_block_rv) override;
  /******** Schedule: Transform loops ********/
  LoopRV Fuse(const ffi::Array<LoopRV>& loop_rvs, bool preserve_unit_iters) override;
  LoopRV Merge(const ffi::Array<LoopRV>& loop_rvs) override;
  ffi::Array<LoopRV> Split(const LoopRV& loop_rv, const ffi::Array<ffi::Optional<ExprRV>>& factors,
                           bool preserve_unit_iters, bool disable_predication) override;
  ffi::Array<LoopRV> LoopPartition(const LoopRV& loop_rv,
                                   const ffi::Array<ffi::Optional<ExprRV>>& factors,
                                   bool preserve_unit_iters) override;
  void Reorder(const ffi::Array<LoopRV>& ordered_loop_rvs) override;
  void ReorderBlockIterVar(const SBlockRV& block_rv, const ffi::Array<int64_t> new_order) override;
  LoopRV AddUnitLoop(const SBlockRV& block_rv) override;
  LoopRV AddUnitLoop(const LoopRV& loop_rv) override;
  /******** Schedule: Manipulate ForKind ********/
  void Parallel(const LoopRV& loop_rv) override;
  void Vectorize(const LoopRV& loop_rv) override;
  void Bind(const LoopRV& loop_rv, const ffi::String& thread_axis) override;
  void Unroll(const LoopRV& loop_rv) override;
  /******** Schedule: Insert cache stages ********/
  SBlockRV CacheRead(const SBlockRV& block_rv, int read_tensor_index,
                     const ffi::String& storage_scope,
                     const ffi::Array<SBlockRV> consumer_blocks = {}) override;
  SBlockRV CacheWrite(const SBlockRV& block_rv, int write_tensor_index,
                      const ffi::String& storage_scope,
                      const ffi::Array<SBlockRV> consumer_blocks = {}) override;
  SBlockRV ReindexCacheRead(const SBlockRV& block_rv, int read_tensor_index,
                            const ffi::String& storage_scope, const IndexMap& index_map) override;
  SBlockRV ReindexCacheWrite(const SBlockRV& block_rv, int write_tensor_index,
                             const ffi::String& storage_scope, const IndexMap& index_map) override;
  ffi::Array<SBlockRV> CacheInplace(const SBlockRV& block_rv, int read_tensor_index,
                                    const ffi::String& storage_scope) override;
  ffi::Array<SBlockRV> CacheIndex(const SBlockRV& block_rv, const ffi::String& storage_scope,
                                  int cse_thresh) override;
  SBlockRV ReIndex(const SBlockRV& block_rv, int tensor_index,
                   TensorIndexType tensor_index_type) override;
  /******** Schedule: Data movement ********/
  SBlockRV ReadAt(const LoopRV& loop_rv, const SBlockRV& block_rv, int read_tensor_index,
                  const ffi::String& storage_scope) override;
  SBlockRV WriteAt(const LoopRV& loop_rv, const SBlockRV& block_rv, int write_tensor_index,
                   const ffi::String& storage_scope) override;
  /******** Schedule: Compute location ********/
  void ComputeAt(const SBlockRV& block_rv, const LoopRV& loop_rv, bool preserve_unit_loops,
                 int index = -1) override;
  void ReverseComputeAt(const SBlockRV& block_rv, const LoopRV& loop_rv, bool preserve_unit_loops,
                        int index = -1) override;
  void ComputeInline(const SBlockRV& block) override;
  void ReverseComputeInline(const SBlockRV& block) override;
  void FuseReductionEpilogue(const SBlockRV& reduction_block,
                             const SBlockRV& epilogue_block) override;
  /******** Schedule: Reduction ********/
  SBlockRV RFactor(const LoopRV& loop_rv, int factor_axis) override;
  SBlockRV DecomposeReduction(const SBlockRV& block_rv, const LoopRV& loop_rv) override;
  void PadEinsum(const SBlockRV& block_rv, const ffi::Array<int64_t>& padding) override;
  /******** Schedule: SBlock annotation ********/
  void StorageAlign(const SBlockRV& block_rv, int tensor_index, int axis, int factor,
                    int offset) override;
  void SetScope(const SBlockRV& block_rv, int tensor_index,
                const ffi::String& storage_scope) override;
  void UnsafeSetDType(const SBlockRV& block_rv, int tensor_index,
                      const ffi::String& dtype) override;
  /******** Schedule: Blockize & Tensorize ********/
  SBlockRV Blockize(const LoopRV& loop_rv, bool preserve_unit_iters) override;
  SBlockRV Blockize(const ffi::Array<SBlockRV>& blocks, bool preserve_unit_iters) override;
  void Tensorize(const SBlockRV& block_rv, const ffi::String& intrin,
                 bool preserve_unit_iters) override;
  void Tensorize(const LoopRV& loop_rv, const ffi::String& intrin,
                 bool preserve_unit_iters) override;
  /******** Schedule: Annotation ********/
  void Annotate(const LoopRV& loop_rv, const ffi::String& ann_key, const Any& ann_val) override;
  void Unannotate(const LoopRV& loop_rv, const ffi::String& ann_key) override;
  void Annotate(const SBlockRV& block_rv, const ffi::String& ann_key, const Any& ann_val) override;
  void Unannotate(const SBlockRV& block_rv, const ffi::String& ann_key) override;
  /******** Schedule: Layout transformation ********/
  void TransformLayout(const SBlockRV& block_rv, int tensor_index,
                       TensorIndexType tensor_index_type, const IndexMap& index_map,
                       const ffi::Optional<IndexMap>& pad_value,
                       bool assume_injective_transform = false) override;
  void TransformBlockLayout(const SBlockRV& block_rv, const IndexMap& index_map) override;
  /******** Schedule: Padding decomposition ********/
  SBlockRV DecomposePadding(const SBlockRV& block_rv, const LoopRV& loop_rv) override;
  /******** Schedule: Tensor transformation ********/
  void RollingBuffer(const SBlockRV& block_rv, int write_tensor_index) override;
  /******** Schedule: Misc ********/
  void EnterPostproc() override {}
  void UnsafeHideTensorAccess(const SBlockRV& block_rv, const ffi::String& tensor_type,
                              const ffi::Array<int64_t>& tensor_index_array) override;
  void AnnotateTensorAccess(const SBlockRV& block_rv, int tensor_index,
                            TensorIndexType tensor_index_type, const IndexMap& index_map) override;

 protected:
  /******** Utility functions ********/
  /*!
   * \brief Copy the schedule state, as well as the symbol table
   * \param new_state The ScheduleState copied
   * \param new_symbol_table The symbol table copied
   */
  void Copy(ScheduleState* new_state, TSymbolTable* new_symbol_table) const;
  /*!
   * \brief Add srefs as random variables into the symbol table
   * \tparam T The type of the random variables
   * \param srefs The srefs to be added to the symbol table
   * \return The new random variables created
   */
  template <class T>
  inline ffi::Array<T> CreateRV(const ffi::Array<StmtSRef>& srefs);
  /*!
   * \brief Add an sref as a random variable into the symbol table
   * \tparam T The type of the random variable
   * \param sref The sref to be added to the symbol table
   * \return The new random variable created
   */
  template <class T>
  inline T CreateRV(const StmtSRef& sref);
  /*!
   * \brief Add an integer as a random variable into the symbol table
   * \param value The integer to be added to the symbol table
   * \return The new random variable created
   */
  inline ExprRV CreateRV(int64_t value);
  /*!
   * \brief Add a list of integers as random variables into the symbol table
   * \param value The list of integers to be added to the symbol table
   * \return The new random variables created
   */
  inline ffi::Array<ExprRV> CreateRV(const std::vector<int64_t>& value);
  /*! \brief Create tile factors, representing an inferred (-1) factor as absence. */
  inline ffi::Array<ffi::Optional<ExprRV>> CreateOptionalRV(const std::vector<int64_t>& value);
  /*! \brief Remove a random variable from the symbol table */
  inline void RemoveFromSymbolTable(const ffi::ObjectRef& rv);
  /*!
   * \brief Check the annotation value is valid and look up the random variable. Raises an exception
   * if the type of the annotation value is not allowed.
   * \param The annotation value.
   * \return The annotation value with random variables substituted with their values.
   */
  Any CheckAndGetAnnotationValue(const ffi::Any& ann_val);
};

// implementations

/******** Lookup random variables ********/

inline SBlock ConcreteScheduleObj::Get(const SBlockRV& block_rv) const {
  StmtSRef sref = this->GetSRef(block_rv);
  return ffi::GetRef<SBlock>(TVM_SREF_TO_SBLOCK(sref));
}

inline For ConcreteScheduleObj::Get(const LoopRV& loop_rv) const {
  StmtSRef sref = this->GetSRef(loop_rv);
  const ForObj* loop = TVM_SREF_TO_FOR(sref);
  return ffi::GetRef<For>(loop);
}

inline PrimExpr ConcreteScheduleObj::Get(const ExprRV& expr_rv) const {
  auto f_substitute = [this](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    auto it = this->symbol_table_.find(var);
    if (it == this->symbol_table_.end()) {
      TVM_FFI_THROW(IndexError) << "Cannot find corresponding ExprRV: " << var;
    }
    const ffi::ObjectRef& obj = (*it).second;
    const auto* int_imm = TVM_TYPE_AS(obj, IntImmObj);
    return ffi::Any(IntImm::Int32(int_imm->value));
  };
  PrimExpr transformed =
      ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(expr_rv, f_substitute).as_or_throw<PrimExpr>();
  return this->analyzer_->Simplify(transformed);
}

inline bool ConcreteScheduleObj::HasBlock(const SBlockRV& block_rv) const {
  auto it = this->symbol_table_.find(block_rv);
  if (it == this->symbol_table_.end()) {
    return false;
  }
  const ffi::ObjectRef& obj = (*it).second;
  const auto* sref = obj.as<StmtSRefObj>();
  if (sref == nullptr || sref->stmt == nullptr) {
    return false;
  }
  return true;
}

inline StmtSRef ConcreteScheduleObj::GetSRef(const SBlockRV& block_rv) const {
  auto it = this->symbol_table_.find(block_rv);
  if (it == this->symbol_table_.end()) {
    TVM_FFI_THROW(IndexError) << "Cannot find corresponding SBlockRV: " << block_rv;
  }
  const ffi::ObjectRef& obj = (*it).second;
  const auto* sref = obj.as<StmtSRefObj>();
  if (sref == nullptr) {
    TVM_FFI_THROW(ValueError) << "SBlockRV's corresponding type is invalid: "
                              << (obj.defined() ? obj->GetTypeKey() : "None");
  }
  if (sref->stmt == nullptr) {
    TVM_FFI_THROW(ValueError) << "The block no longer exists in the IRModule";
  }
  return ffi::GetRef<StmtSRef>(sref);
}

inline StmtSRef ConcreteScheduleObj::GetSRef(const LoopRV& loop_rv) const {
  static StmtSRef inline_mark = StmtSRef::InlineMark();
  static StmtSRef root_mark = StmtSRef::RootMark();
  auto it = this->symbol_table_.find(loop_rv);
  if (it == this->symbol_table_.end()) {
    TVM_FFI_THROW(IndexError) << "Cannot find corresponding LoopRV: " << loop_rv;
  }
  const ffi::ObjectRef& obj = (*it).second;
  if (obj.same_as(inline_mark)) {
    return inline_mark;
  }
  if (obj.same_as(root_mark)) {
    return root_mark;
  }
  const auto* sref = obj.as<StmtSRefObj>();
  if (sref == nullptr) {
    TVM_FFI_THROW(ValueError) << "LoopRV's corresponding type is invalid: "
                              << (obj.defined() ? obj->GetTypeKey() : "None");
  }
  if (sref->stmt == nullptr) {
    TVM_FFI_THROW(ValueError) << "The loop no longer exists in the IRModule";
  }
  return ffi::GetRef<StmtSRef>(sref);
}

template <class T>
inline ffi::Array<StmtSRef> GetSRefsHelper(const ConcreteScheduleObj* sch,
                                           const ffi::Array<T>& rvs) {
  ffi::Array<StmtSRef> result;
  result.reserve(rvs.size());
  for (const T& rv : rvs) {
    result.push_back(sch->GetSRef(rv));
  }
  return result;
}

inline ffi::Array<StmtSRef> ConcreteScheduleObj::GetSRefs(const ffi::Array<SBlockRV>& rvs) const {
  return GetSRefsHelper(this, rvs);
}

inline ffi::Array<StmtSRef> ConcreteScheduleObj::GetSRefs(const ffi::Array<LoopRV>& rvs) const {
  return GetSRefsHelper(this, rvs);
}

/******** Adding/Removing elements in the symbol table ********/

template <class T>
inline ffi::Array<T> ConcreteScheduleObj::CreateRV(const ffi::Array<StmtSRef>& srefs) {
  ffi::Array<T> result;
  result.reserve(srefs.size());
  for (const StmtSRef& sref : srefs) {
    T rv;
    this->symbol_table_.Set(rv, sref);
    result.push_back(rv);
  }
  return result;
}

template <class T>
inline T ConcreteScheduleObj::CreateRV(const StmtSRef& sref) {
  T rv;
  this->symbol_table_.Set(rv, sref);
  return rv;
}

inline ExprRV ConcreteScheduleObj::CreateRV(int64_t value) {
  Var rv("v" + std::to_string(this->symbol_table_.size() + 1), PrimType::Int(32));
  this->symbol_table_.Set(rv, IntImm::Int32(static_cast<int32_t>(value)));
  return rv.as_or_throw<PrimExpr>();
}

inline ffi::Array<ExprRV> ConcreteScheduleObj::CreateRV(const std::vector<int64_t>& value) {
  ffi::Array<ExprRV> results;
  results.reserve(value.size());
  for (int64_t v : value) {
    results.push_back(CreateRV(v));
  }
  return results;
}

inline ffi::Array<ffi::Optional<ExprRV>> ConcreteScheduleObj::CreateOptionalRV(
    const std::vector<int64_t>& value) {
  ffi::Array<ffi::Optional<ExprRV>> results;
  results.reserve(value.size());
  for (int64_t v : value) {
    results.push_back(v == -1 ? ffi::Optional<ExprRV>(std::nullopt) : CreateRV(v));
  }
  return results;
}

inline void ConcreteScheduleObj::RemoveFromSymbolTable(const ffi::ObjectRef& obj) {
  auto it = this->symbol_table_.find(obj);
  if (it != this->symbol_table_.end()) {
    this->symbol_table_.erase(obj);
  } else {
    TVM_FFI_THROW(IndexError) << "Cannot find the object in the symbol table: " << obj;
    throw;
  }
}

}  // namespace s_tir
}  // namespace tvm

#endif  // TVM_S_TIR_SCHEDULE_CONCRETE_SCHEDULE_H_
