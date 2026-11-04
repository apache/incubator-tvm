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

/*!
 * \file s_tir/analysis/sblock_access_region_detector.cc
 * \brief Detect sblock read/write regions by visiting its body
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/op/memory.h>

#include <unordered_map>
#include <unordered_set>

#include "../tensor_region.h"
#include "../transform/ir_utils.h"
#include "conditional_bounds.h"

namespace tvm {
namespace tirx {

/*!
 * \brief Detect which regions of tensors in this block are read or written to. Regions are sorted
 * by order of appearance in the AST. \note This detector can only visit blocks and will not visit
 * child blocks recursively
 */
class BlockReadWriteDetector : public s_tir::StmtExprVisitor {
 public:
  using s_tir::StmtExprVisitor::Visit_;

  explicit BlockReadWriteDetector(const ffi::Map<Var, TensorVar>& tensor_var_map)
      : tensor_var_map_(tensor_var_map) {
    for (const auto& item : tensor_var_map) {
      const TensorVar& tensor = item.second;
      tensor_var_map_.Set(tensor, tensor);
    }
  }

  /*! \brief Return read regions of the block */
  ffi::Array<TensorRegion> CollectReads(
      const std::unordered_set<const VarObj*>* excluded_tensors = nullptr);
  /*! \brief Return write regions of the block */
  ffi::Array<TensorRegion> CollectWrites(
      const std::unordered_set<const VarObj*>* excluded_tensors = nullptr);
  /*!
   * \brief Return opaque tensor regions of the block
   * \note The tensor accessed by load/store or call with tensor.data will
   *       be marked as opaque.
   */
  ffi::Array<TensorRegion> CollectOpaques();
  /*! \brief overload operator() to make sure it accepts a block node */
  void operator()(const Stmt& stmt);

 private:
  /*! \brief Iteration range for loop_vars */
  ffi::Map<Var, sym::IntSet> dom_map_;
  /*! \brief Extra iteration range hint for free vars */
  ffi::Map<Var, sym::IntSet> hint_map_;
  /*! \brief Unresolved conditions within current scope. */
  std::vector<PrimExpr> pending_conditions_;
  /*! \brief The tensors that the current block reads */
  std::vector<TensorVar> read_tensors_;
  /*! \brief The tensors that the current block writes */
  std::vector<TensorVar> writes_tensors_;
  /*! \brief The opaque tensor which is access by tensor.data */
  std::vector<TensorVar> opaque_tensors_;
  /*! \brief The read regions of the current block */
  std::vector<std::vector<tvm::sym::IntSet>> read_regions_;
  /*! \brief The write regions of the current block */
  std::vector<std::vector<tvm::sym::IntSet>> write_regions_;
  /*! \brief The opaque regions of the current block */
  std::vector<std::vector<tvm::sym::IntSet>> opaque_regions_;
  /*! \brief The outside tensor data mapping to its tensor */
  ffi::Map<Var, TensorVar> tensor_var_map_;
  /*! \brief The target tensor var mapping to its matching */
  std::unordered_map<const VarObj*, s_tir::MatchTensorRegion> match_tensors_;
  /*! \brief let bindings inside the block */
  std::unordered_map<const VarObj*, PrimExpr> let_bindings_;
  /*!\ brief Internal analyzer. */
  sym::Analyzer ana_;

  /*!
   * \brief Update read/write tensors and regions with provided tensor and region
   * \param tensors The tensors should be updated
   * \param regions The access regions should be updated
   * \param tensor The provided tensor
   * \param region The provided region
   */
  void Update(std::vector<TensorVar>* tensors, std::vector<std::vector<sym::IntSet>>* regions,
              TensorVar tensor, std::vector<sym::IntSet> region);

  /*! \brief Helper function to collect access regions. */
  ffi::Array<TensorRegion> CollectRegions(
      const std::vector<TensorVar>& tensors,
      const std::vector<std::vector<tvm::sym::IntSet>>& regions,
      const std::unordered_set<const VarObj*>* excluded_tensors = nullptr);

  /*! \brief Helper function to convert matched access region to source region. */
  std::vector<sym::IntSet> ConvertMatchedRegion(const s_tir::MatchTensorRegion& match_tensor,
                                                const std::vector<sym::IntSet>& int_sets) const;

  /*! \brief Helper function to update a opaque access. */
  void UpdateOpaque(const Var& tensor_var);

  /*! \brief Helper function to relax the tensor indices */
  sym::IntSet RelaxAccessIndex(const PrimExpr& index);

  // Declared regions carry bounds, not opaque runtime accesses.
  ffi::Optional<VisitInterrupt> Visit_(const TensorRegionObj* op) final {
    if (!op->source.as<TensorVar>()) return StmtExprVisitor::Visit_(op);
    for (const Range& range : op->region) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(range->begin));
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(range->extent));
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const ForObj* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const IfObj* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const s_tir::SBlockRealizeObj* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const TensorStoreObj* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const BindObj* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadObj* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const VarObj* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const CallObj* op) override;
};

void BlockReadWriteDetector::operator()(const Stmt& stmt) {
  const auto* block = stmt.as<s_tir::SBlockObj>();
  TVM_FFI_ICHECK(block != nullptr)
      << "Only visiting Blocks is allowed, but got " << stmt->GetTypeKey();
  for (const s_tir::MatchTensorRegion& match_tensor : block->match_tensors) {
    const Var target_var = match_tensor->tensor;
    const Var source_var = match_tensor->source->source.as_or_throw<tvm::tirx::TensorVar>();
    if (tensor_var_map_.find(source_var) != tensor_var_map_.end()) {
      match_tensors_.insert_or_assign(target_var.get(), match_tensor);
      tensor_var_map_.Set(target_var, match_tensor->tensor);
    }
  }
  s_tir::StmtExprVisitor::Visit(stmt);
}

ffi::Array<TensorRegion> BlockReadWriteDetector::CollectReads(
    const std::unordered_set<const VarObj*>* excluded_tensors) {
  return CollectRegions(read_tensors_, read_regions_, excluded_tensors);
}

ffi::Array<TensorRegion> BlockReadWriteDetector::CollectWrites(
    const std::unordered_set<const VarObj*>* excluded_tensors) {
  return CollectRegions(writes_tensors_, write_regions_, excluded_tensors);
}

ffi::Array<TensorRegion> BlockReadWriteDetector::CollectOpaques() {
  return CollectRegions(opaque_tensors_, opaque_regions_);
}

ffi::Optional<VisitInterrupt> BlockReadWriteDetector::Visit_(const VarObj* op) {
  if (def_region_kind() != kTVMFFIDefRegionKindNone) return StmtExprVisitor::Visit_(op);
  UpdateOpaque(ffi::GetRef<Var>(op));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> BlockReadWriteDetector::Visit_(const TensorLoadObj* op) {
  auto f_substitute = [this](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (auto it = let_bindings_.find(var.get()); it != let_bindings_.end()) {
      return ffi::Any(it->second);
    }
    return ffi::Unchanged();
  };
  std::vector<sym::IntSet> relaxed_region;
  for (PrimExpr index : op->indices) {
    PrimExpr remapped_index =
        ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(index, f_substitute).as_or_throw<PrimExpr>();
    while (!remapped_index.same_as(index)) {
      index = remapped_index;
      remapped_index = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(index, f_substitute)
                           .as_or_throw<PrimExpr>();
    }
    relaxed_region.push_back(sym::EvalSet(sym::IntSet::Vector(remapped_index), dom_map_));
  }
  Update(&read_tensors_, &read_regions_, op->source.as_or_throw<tvm::tirx::TensorVar>(),
         relaxed_region);
  for (const auto& index : op->indices) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(index));
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> BlockReadWriteDetector::Visit_(const ForObj* op) {
  Range range = Range(op->begin, op->end);
  dom_map_.Set(op->var, sym::IntSet::FromRange(range));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(s_tir::StmtExprVisitor::Visit_(op));
  dom_map_.erase(op->var);
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> BlockReadWriteDetector::Visit_(const IfObj* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->condition));
  {
    // Visit then branch
    With<s_tir::ConditionalBoundsContext> ctx(op->condition, &dom_map_, &hint_map_,
                                              &pending_conditions_);
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(s_tir::StmtExprVisitor::Visit(op->then_case));
  }
  if (op->else_case) {
    // Visit else branch
    With<s_tir::ConditionalBoundsContext> ctx(!op->condition, &dom_map_, &hint_map_,
                                              &pending_conditions_);
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(s_tir::StmtExprVisitor::Visit(op->else_case.value()));
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> BlockReadWriteDetector::Visit_(const BindObj* op) {
  if (const auto* call = op->value.as<CallObj>();
      call && call->op.same_as(tirx::decl_tensor_op())) {
    // A DeclTensor data expression defines the alias source.  It is not an
    // opaque tensor access by the containing block.
    return WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() { return Visit(op->var); });
  }
  if (auto value = op->value.as<PrimExpr>()) {
    let_bindings_.insert_or_assign(op->var.get(), value.value());
  }
  return s_tir::StmtExprVisitor::Visit_(op);
}

ffi::Optional<VisitInterrupt> BlockReadWriteDetector::Visit_(const CallObj* op) {
  auto update_masked_access = [this](const TensorVar& tensor, const ffi::Array<PrimExpr>& indices,
                                     std::vector<TensorVar>* tensors,
                                     std::vector<std::vector<sym::IntSet>>* regions) {
    auto f_substitute = [this](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
      if (auto it = let_bindings_.find(var.get()); it != let_bindings_.end()) {
        return ffi::Any(it->second);
      }
      return ffi::Unchanged();
    };
    std::vector<sym::IntSet> relaxed_region;
    for (PrimExpr index : indices) {
      PrimExpr remapped_index = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(index, f_substitute)
                                    .as_or_throw<PrimExpr>();
      while (!remapped_index.same_as(index)) {
        index = remapped_index;
        remapped_index = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(index, f_substitute)
                             .as_or_throw<PrimExpr>();
      }
      relaxed_region.push_back(sym::EvalSet(sym::IntSet::Vector(remapped_index), dom_map_));
    }
    Update(tensors, regions, tensor, relaxed_region);
  };
  if (op->op.same_as(tirx::masked_load_op()) || op->op.same_as(tirx::masked_store_op())) {
    bool is_load = op->op.same_as(tirx::masked_load_op());
    TensorVar tensor = op->args[0].as_or_throw<TensorVar>();
    ffi::Array<PrimExpr> indices;
    for (size_t i = is_load ? 1 : 2; i + 1 < op->args.size(); ++i) {
      indices.push_back(op->args[i].as_or_throw<PrimExpr>());
    }
    update_masked_access(tensor, indices, is_load ? &read_tensors_ : &writes_tensors_,
                         is_load ? &read_regions_ : &write_regions_);
    for (size_t i = 1; i < op->args.size(); ++i) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->args[i]));
    }
    return std::nullopt;
  }
  if (op->op.same_as(tirx::address_of_op())) {
    if (const auto* load = op->args[0].as<TensorLoadObj>()) {
      // A tensor address is an opaque use, not a load of its pointed-to value.
      UpdateOpaque(load->source.as_or_throw<TensorVar>());
      for (const PrimExpr& index : load->indices) {
        TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(index));
      }
      return std::nullopt;
    }
  }
  if (op->op.same_as(prim::if_then_else_op())) {
    PrimExpr condition = op->args[0].as_or_throw<PrimExpr>();
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(condition));
    {
      // Visit then branch
      With<s_tir::ConditionalBoundsContext> ctx(condition, &dom_map_, &hint_map_,
                                                &pending_conditions_);
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(
          s_tir::StmtExprVisitor::Visit(op->args[1].as_or_throw<PrimExpr>()));
    }
    {
      // Visit else branch
      With<s_tir::ConditionalBoundsContext> ctx(!condition, &dom_map_, &hint_map_,
                                                &pending_conditions_);
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(
          s_tir::StmtExprVisitor::Visit(op->args[2].as_or_throw<PrimExpr>()));
    }
    return std::nullopt;
  }
  return s_tir::StmtExprVisitor::Visit_(op);
}

ffi::Optional<VisitInterrupt> BlockReadWriteDetector::Visit_(const TensorStoreObj* op) {
  auto f_substitute = [this](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (auto it = let_bindings_.find(var.get()); it != let_bindings_.end()) {
      return ffi::Any(it->second);
    }
    return ffi::Unchanged();
  };
  std::vector<sym::IntSet> relaxed_region;
  for (PrimExpr index : op->indices) {
    PrimExpr remapped_index =
        ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(index, f_substitute).as_or_throw<PrimExpr>();
    while (!remapped_index.same_as(index)) {
      index = remapped_index;
      remapped_index = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(index, f_substitute)
                           .as_or_throw<PrimExpr>();
    }
    relaxed_region.push_back(sym::EvalSet(sym::IntSet::Vector(remapped_index), dom_map_));
  }
  Update(&writes_tensors_, &write_regions_, op->dest.as_or_throw<TensorVar>(), relaxed_region);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->value));
  for (const auto& index : op->indices) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(index));
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> BlockReadWriteDetector::Visit_(const s_tir::SBlockRealizeObj* op) {
  /*! \note detector will not visit child block recursively, so it will stop here */
  std::unordered_map<const VarObj*, PrimExpr> vmap;
  for (size_t i = 0; i < op->block->iter_vars.size(); ++i) {
    vmap.insert_or_assign(op->block->iter_vars[i]->var.get(), op->iter_values[i]);
  }
  auto f_substitute = [&vmap](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (auto it = vmap.find(var.get()); it != vmap.end()) return ffi::Any(it->second);
    return ffi::Unchanged();
  };
  for (const auto& read : op->block->reads) {
    std::vector<sym::IntSet> relaxed_region;
    for (const auto& range : read->region) {
      PrimExpr min = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(range->begin, f_substitute)
                         .as_or_throw<PrimExpr>();
      PrimExpr extent = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(range->extent, f_substitute)
                            .as_or_throw<PrimExpr>();
      relaxed_region.push_back(
          sym::EvalSet(sym::IntSet::FromRange(Range::FromMinExtent(min, extent)), dom_map_));
    }
    Update(&read_tensors_, &read_regions_, read->source.as_or_throw<tvm::tirx::TensorVar>(),
           relaxed_region);
  }
  for (const auto& write : op->block->writes) {
    std::vector<sym::IntSet> relaxed_region;
    for (const auto& range : write->region) {
      PrimExpr min = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(range->begin, f_substitute)
                         .as_or_throw<PrimExpr>();
      PrimExpr extent = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(range->extent, f_substitute)
                            .as_or_throw<PrimExpr>();
      relaxed_region.push_back(
          sym::EvalSet(sym::IntSet::FromRange(Range::FromMinExtent(min, extent)), dom_map_));
    }
    Update(&writes_tensors_, &write_regions_, write->source.as_or_throw<tvm::tirx::TensorVar>(),
           relaxed_region);
  }
  return std::nullopt;
}

std::vector<sym::IntSet> BlockReadWriteDetector::ConvertMatchedRegion(
    const s_tir::MatchTensorRegion& match_tensor, const std::vector<sym::IntSet>& int_sets) const {
  const TensorVar& tensor = match_tensor->tensor;

  ffi::Array<Range> region;
  region.reserve(int_sets.size());
  TVM_FFI_ICHECK_EQ(tensor.shape().size(), int_sets.size());
  for (size_t i = 0; i < int_sets.size(); ++i) {
    const tvm::sym::IntSet& int_set = int_sets[i];
    region.push_back(int_set.CoverRange(Range::FromMinExtent(0, tensor.shape()[i])).value());
  }

  region = ConvertRegion(match_tensor, region);

  std::vector<sym::IntSet> result;
  result.reserve(region.size());
  for (const Range& range : region) {
    result.push_back(sym::EvalSet(range, dom_map_));
  }
  return result;
}

void BlockReadWriteDetector::Update(std::vector<TensorVar>* tensors,
                                    std::vector<std::vector<sym::IntSet>>* regions,
                                    TensorVar tensor, std::vector<sym::IntSet> region) {
  if (tensor_var_map_.find(tensor) == tensor_var_map_.end()) return;
  // Handle match_tensor remap
  auto it = match_tensors_.find(tensor.get());
  if (it != match_tensors_.end()) {
    const s_tir::MatchTensorRegion& match_tensor = it->second;
    tensor = match_tensor->source->source.as_or_throw<tvm::tirx::TensorVar>();
    region = ConvertMatchedRegion(match_tensor, std::move(region));
  }
  TVM_FFI_ICHECK_EQ(tensors->size(), regions->size())
      << " Expected the tensor and regions to have the same size ";
  for (size_t i = 0; i < regions->size(); ++i) {
    if ((*tensors)[i].same_as(tensor)) {
      TVM_FFI_ICHECK_EQ((*regions)[i].size(), region.size()) << "Inconsistent tensor dimension";
      for (size_t j = 0; j < region.size(); ++j) {
        (*regions)[i][j] = sym::Union({(*regions)[i][j], region[j]});
      }
      return;
    }
  }
  tensors->push_back(std::move(tensor));
  regions->push_back(std::move(region));
}

ffi::Array<TensorRegion> BlockReadWriteDetector::CollectRegions(
    const std::vector<TensorVar>& tensors,
    const std::vector<std::vector<tvm::sym::IntSet>>& regions,
    const std::unordered_set<const VarObj*>* excluded_tensors) {
  TVM_FFI_ICHECK_EQ(tensors.size(), regions.size());
  ffi::Array<TensorRegion> res;
  res.reserve(tensors.size());
  for (size_t i = 0; i < regions.size(); ++i) {
    if (excluded_tensors != nullptr && excluded_tensors->count(tensors[i].get())) {
      continue;
    }
    ffi::Array<Range> region;
    region.reserve(regions[i].size());
    TVM_FFI_ICHECK_EQ(tensors[i].shape().size(), regions[i].size());
    for (size_t j = 0; j < regions[i].size(); j++) {
      const tvm::sym::IntSet& range = regions[i][j];
      if (range.CanProveSinglePoint(ana_)) {
        PrimExpr min = range.min();
        region.push_back(Range::FromMinExtent(min, prim::MakeConst(min.ty(), 1)));
      } else {
        region.push_back(range.CoverRange(Range::FromMinExtent(0, tensors[i].shape()[j])).value());
      }
    }
    res.push_back(TensorRegion(tensors[i], region, TensorRegionType()));
  }
  return res;
}

void BlockReadWriteDetector::UpdateOpaque(const Var& tensor_var) {
  auto it = tensor_var_map_.find(tensor_var);
  if (it != tensor_var_map_.end()) {
    const TensorVar& tensor = (*it).second;
    const TensorRegion tensor_region = s_tir::FullTensorRegion(tensor);
    const ffi::Array<Range>& region = tensor_region->region;
    std::vector<sym::IntSet> int_set;
    int_set.reserve(region.size());
    for (const Range& range : region) {
      int_set.push_back(sym::EvalSet(range, dom_map_));
    }
    Update(&opaque_tensors_, &opaque_regions_, tensor, int_set);
  }
}

ffi::Array<ffi::Array<TensorRegion>> GetSBlockAccessRegion(
    const s_tir::SBlock& block, const ffi::Map<Var, TensorVar>& tensor_var_map) {
  auto detector = ffi::make_object<BlockReadWriteDetector>(tensor_var_map);
  detector->operator()(block);
  ffi::Array<TensorRegion> writes = detector->CollectWrites();
  std::unordered_set<const VarObj*> excluded_tensors;
  // exclude write tensors from read regions for reductions if init block is defined.
  if (block->init.has_value()) {
    for (const TensorRegion& write_access : writes) {
      excluded_tensors.insert(write_access->source.as_or_throw<tvm::tirx::TensorVar>().get());
    }
  }
  ffi::Array<TensorRegion> reads = detector->CollectReads(&excluded_tensors);
  ffi::Array<TensorRegion> opaques = detector->CollectOpaques();
  return {reads, writes, opaques};
}

ffi::Array<ffi::Array<TensorRegion>> GetSBlockReadWriteRegion(
    const s_tir::SBlock& block, const ffi::Map<Var, TensorVar>& tensor_var_map) {
  auto detector = ffi::make_object<BlockReadWriteDetector>(tensor_var_map);
  detector->operator()(block);
  ffi::Array<TensorRegion> opaques = detector->CollectOpaques();
  std::unordered_set<const VarObj*> excluded_tensors;
  for (const TensorRegion& opaque_access : opaques) {
    excluded_tensors.insert(opaque_access->source.as_or_throw<tvm::tirx::TensorVar>().get());
  }
  ffi::Array<TensorRegion> writes = detector->CollectWrites(&excluded_tensors);
  if (block->init.has_value()) {
    for (const TensorRegion& write_access : writes) {
      excluded_tensors.insert(write_access->source.as_or_throw<tvm::tirx::TensorVar>().get());
    }
  }
  ffi::Array<TensorRegion> reads = detector->CollectReads(&excluded_tensors);
  for (const TensorRegion& opaque_access : opaques) {
    reads.push_back(opaque_access);
    writes.push_back(opaque_access);
  }
  return {reads, writes};
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("s_tir.analysis.GetSBlockAccessRegion", GetSBlockAccessRegion)
      .def("s_tir.analysis.GetSBlockReadWriteRegion", GetSBlockReadWriteRegion);
}

}  // namespace tirx
}  // namespace tvm
