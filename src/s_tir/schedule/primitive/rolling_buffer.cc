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
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/s_tir/stmt.h>

#include <functional>

#include "../ir_comparator.h"
#include "../utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::prim;
using namespace tvm::tirx;

namespace {

struct RollingBufferInfo {
  TensorVar old_tensor;
  TensorVar new_tensor;
  int rolling_axis;
  PrimExpr rolling_extent;
  std::vector<int> axis_overlaps;
  std::vector<ffi::Optional<Var>> axis_iter_vars;
  /*! \brief The map used for ScheduleStateObj::Replace. */
  ffi::Map<SBlock, SBlock> block_reuse;
};

TensorRegion GetRelaxedTensorRegion(const SBlockRealize& realize, const TensorRegion& tensor_region,
                                    const ffi::Map<Var, sym::IntSet>& dom_map) {
  ffi::Map<Var, PrimExpr> bindings = GetBindings(realize);
  auto f_substitute = [&bindings](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (auto repl = bindings.Get(var)) return ffi::Any(*std::move(repl));
    return ffi::Unchanged();
  };
  ffi::Array<Range> mapped_region = tensor_region->region.Map([&f_substitute](const Range& range) {
    PrimExpr min = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(range->begin, f_substitute)
                       .as_or_throw<PrimExpr>();
    PrimExpr extent = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(range->extent, f_substitute)
                          .as_or_throw<PrimExpr>();
    return Range::FromMinExtent(min, extent);
  });
  ffi::Array<sym::IntSet> relaxed_intsets = sym::EvalSet(mapped_region, dom_map);
  ffi::Array<Range> relaxed_region;
  relaxed_region.reserve(relaxed_intsets.size());
  for (size_t i = 0; i < relaxed_intsets.size(); ++i) {
    relaxed_region.push_back(
        relaxed_intsets[i]
            .CoverRange(Range::FromMinExtent(
                0, tensor_region->source.as_or_throw<tvm::tirx::TensorVar>().shape()[i]))
            .value());
  }
  return TensorRegion(tensor_region->source.as_or_throw<tvm::tirx::TensorVar>(), relaxed_region,
                      TensorRegionType());
}

class RollingBufferDependencyError : public ScheduleErrorContextObj {
 public:
  explicit RollingBufferDependencyError(IRModule mod, SBlock block)
      : mod_(mod), block_(std::move(block)) {}

  ffi::String FastErrorString() const final {
    return "ScheduleError: The target block is required to have only RAW dependencies";
  }

  ffi::String DetailRenderTemplate() const final {
    return "The target block {0} is required to have only RAW dependencies";
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_}; }

  /*!
   * \brief Check if the block has only RAW dependencies.
   * \param self The schedule state
   * \param block_sref The sref of the block to be checked
   * \param scope_root_sref The sref of the scope root
   * \throw ScheduleError if the block has WAW or WAR dependency.
   */
  static void Check(const ScheduleState& self, const StmtSRef& block_sref,
                    const StmtSRef& scope_root_sref) {
    SBlockScope scope = self->GetSBlockScope(scope_root_sref);
    for (const Dependency& producers : scope->GetDepsByDst(block_sref)) {
      if (!(producers->kind == DepKind::kRAW)) {
        const SBlockObj* block = TVM_SREF_TO_SBLOCK(block_sref);
        throw MakeScheduleError<RollingBufferDependencyError>(self->mod,
                                                              ffi::GetRef<SBlock>(block));
      }
    }
    for (const Dependency& consumers : scope->GetDepsBySrc(block_sref)) {
      if (!(consumers->kind == DepKind::kRAW)) {
        const SBlockObj* block = TVM_SREF_TO_SBLOCK(block_sref);
        throw MakeScheduleError<RollingBufferDependencyError>(self->mod,
                                                              ffi::GetRef<SBlock>(block));
      }
    }
  }

 private:
  IRModule mod_;
  SBlock block_;
};

class RollingBufferMatchError : public ScheduleErrorContextObj {
 public:
  RollingBufferMatchError(IRModule mod, SBlock block, TensorRegion tensor_region)
      : mod_(mod), block_(block), tensor_region_(tensor_region) {}
  ffi::String FastErrorString() const final {
    return "ScheduleError: rolling_buffer expect the tensor region to have at least one dimention"
           "matching the rolling pattern such as: hh.outer * stride + hh.inner";
  }
  ffi::String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The target tensor " << tensor_region_->source.as_or_throw<tvm::tirx::TensorVar>()->name
       << " with region " << tensor_region_->region
       << " should have at least one dimension range that matches a rolling pattern "
          "such as hh.outer * stride + hh.inner. ";
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_}; }

 private:
  IRModule mod_;
  SBlock block_;
  TensorRegion tensor_region_;
};

class RollingBufferInsertionError : public ScheduleErrorContextObj {
 public:
  RollingBufferInsertionError(IRModule mod, TensorVar tensor, SBlock block)
      : mod_(mod), tensor_(std::move(tensor)), block_(block) {}
  ffi::String FastErrorString() const final {
    return "ScheduleError: rolling_buffer injection is invalid, the lca of the access "
           "location of the target tensor is not a for loop. ";
  }

  ffi::String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "rolling_buffer injection is invalid. The block {0} should be tiled so that "
       << "the lca of the access location of the target tensor " << tensor_->name
       << " is a for loop. ";
    return os.str();
  }
  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_}; }

 private:
  IRModule mod_;
  TensorVar tensor_;
  SBlock block_;
};

class RollingBufferInfoCollector {
 public:
  static RollingBufferInfo CheckAndGetRollingBufferInfo(const IRModule& mod,
                                                        const StmtSRef& block_sref,
                                                        const TensorRegion& tensor_region) {
    RollingBufferInfoCollector collector;
    if (!collector.MatchRollingBuffer(block_sref, tensor_region)) {
      const SBlockObj* block = TVM_SREF_TO_SBLOCK(block_sref);
      throw MakeScheduleError<RollingBufferMatchError>(mod, ffi::GetRef<SBlock>(block),
                                                       tensor_region);
    }
    return collector.info_.value();
  }

 private:
  bool MatchRollingBuffer(const StmtSRef& block_sref, const TensorRegion& tensor_region) {
    const TensorVar& tensor = tensor_region->source.as_or_throw<tvm::tirx::TensorVar>();
    const ffi::Array<Range>& region = tensor_region->region;

    std::vector<ffi::Optional<Var>> bound_iter_vars;
    std::vector<int> bound_overlaps;

    for (auto bound : region) {
      auto stride = 0;
      auto divisor = 1;

      ffi::Optional<Var> iter_var;
      if (const auto* div = bound->begin.as<prim::FloorDivObj>()) {
        const auto* mul = div->a.as<prim::MulObj>();
        if (!mul) return false;
        iter_var = mul->a.as<Var>();
        auto stride_imm = mul->b.as<IntImm>();
        auto divisor_imm = div->b.as<IntImm>();
        if (!iter_var.has_value() || !stride_imm.has_value() || !divisor_imm.has_value()) {
          return false;
        }
        // Handle the case of fractional strides
        // They take this form: floordiv(hh.outer * stride, divisor)
        // Strip the floordiv and keep track of the divisor
        auto divisor_value = divisor_imm.value()->value.as<int>();
        auto stride_value = stride_imm.value()->value.as<int>();
        if (!divisor_value.has_value() || !stride_value.has_value()) return false;
        divisor = *divisor_value;
        stride = std::ceil(static_cast<float>(*stride_value) / divisor);
      } else if (const auto* mul = bound->begin.as<prim::MulObj>()) {
        // The bound is the iter var multiplied by the stride
        iter_var = mul->a.as<Var>();
        auto stride_imm = mul->b.as<IntImm>();
        if (!iter_var.has_value() || !stride_imm.has_value()) return false;
        auto stride_value = stride_imm.value()->value.as<int>();
        if (!stride_value.has_value()) return false;
        stride = *stride_value;
      } else if (auto var = bound->begin.as<Var>()) {
        // If the bound is just a Var, that implies the stride is 1
        iter_var = var;
        stride = 1;
      } else if (IsConstInt(bound->begin)) {
        // If the bound is an int, we can't roll over it
        iter_var = std::nullopt;
      } else {
        // If all of the above matches fail, we're in unknown behaviour
        return false;
      }
      auto bound_overlap = 0;
      if (iter_var.has_value()) {
        auto extent = bound->extent.as_or_throw<IntImm>()->value;
        bound_overlap = (extent - stride).as<int>().value();
        // Since Pass CompactTensorAllocation will be responsible for compacting the tensor
        // allocation region, there is no need to roll over the axis where the overlap is not
        // positive, so reset iter_var to std::nullopt.
        if (bound_overlap <= 0) {
          iter_var = std::nullopt;
        }
      }
      bound_iter_vars.push_back(iter_var);
      bound_overlaps.push_back(bound_overlap);
    }

    ffi::Array<StmtSRef> loop_srefs = GetLoops(block_sref);
    // Pick the outermost iter_var that's mentioned in the bounds
    // to be the rolling axis
    ffi::Optional<Var> roll_iter_var;
    int roll_axis = 0;
    for (const tirx::StmtSRef& loop_sref : loop_srefs) {
      auto loop_var = loop_sref->StmtAs<ForObj>()->var;

      auto it{std::find_if(
          bound_iter_vars.begin(), bound_iter_vars.end(),
          [&](ffi::Optional<Var> var) { return var && (var.value().get() == loop_var.get()); })};
      if (it != bound_iter_vars.end()) {
        auto i = std::distance(bound_iter_vars.begin(), it);
        roll_iter_var = loop_var;
        roll_axis = i;
        break;
      }
    }

    if (!roll_iter_var.has_value()) {
      return false;
    }
    ffi::Array<PrimExpr> new_shape = tensor.shape();
    new_shape.Set(roll_axis, region[roll_axis]->extent);
    auto new_tensor_type = ffi::make_object<TensorTypeObj>(*tensor.ty().get());
    new_tensor_type->shape = new_shape;
    TensorVar new_tensor =
        TensorVar(tensor->name, TensorType(std::move(new_tensor_type)), tensor->loc);

    info_ = RollingBufferInfo{tensor,         new_tensor,      roll_axis, region[roll_axis]->extent,
                              bound_overlaps, bound_iter_vars, {}};

    return true;
  }

  std::optional<RollingBufferInfo> info_;
};

class RollingBufferRewriter : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  static Stmt Rewrite(const StmtSRef& scope_sref, RollingBufferInfo* info) {
    auto rewriter = ffi::make_object<RollingBufferRewriter>(scope_sref, info);
    return rewriter->Mutate(ffi::GetRef<Stmt>(scope_sref->stmt))
        .ValueOrUnchanged(ffi::GetRef<Stmt>(scope_sref->stmt));
  }

  explicit RollingBufferRewriter(const StmtSRef& scope_sref, RollingBufferInfo* info)
      : scope_sref_(scope_sref), info_(info) {}

 private:
  void RewriteAccessRegion(ffi::Array<TensorRegion>* old_access_regions,
                           const ffi::Array<TensorRegion>& infered_access_regions) {
    auto fmutate = [this, &infered_access_regions](const TensorRegion& tensor_region) {
      if (tensor_region->source.as_or_throw<tvm::tirx::TensorVar>().same_as(info_->old_tensor)) {
        TVM_FFI_ICHECK(infered_access_regions.size() == 1);
        return infered_access_regions[0];
      }
      return tensor_region;
    };
    (*old_access_regions).MutateByApply(fmutate);
  }

  void RewriteTensorAccess(TensorVar* tensor, ffi::Array<PrimExpr>* indices) const {
    ffi::Array<PrimExpr> new_indices;
    new_indices.reserve(indices->size());
    // First modify the access indices to use modulo arithmetic
    // for the rolling axis
    for (size_t i = 0; i < indices->size(); ++i) {
      if (static_cast<int>(i) == info_->rolling_axis) {
        new_indices.push_back(FloorMod((*indices)[i], info_->rolling_extent));
      } else {
        new_indices.push_back((*indices)[i]);
      }
    }
    // Replace the accessed tensor with the new tensor.
    *tensor = info_->new_tensor;
    *indices = std::move(new_indices);
  }

  UnchangedOr<Stmt> Mutate_(const SBlockObj* block, InplaceMode inplace_mode) final {
    SBlock old_stmt = ffi::GetRef<SBlock>(block);
    SBlock stmt = StmtExprMutator::Mutate_(block, inplace_mode)
                      .ValueOrUnchanged(ffi::GetRef<Stmt>(block))
                      .as_or_throw<SBlock>();
    SBlockObj* n = stmt.CopyOnWrite();
    if (block == scope_sref_->stmt) {
      ffi::Array<TensorVar> new_alloc_tensors;
      for (const TensorVar& tensor : stmt->alloc_tensors) {
        if (tensor != info_->old_tensor) {
          new_alloc_tensors.push_back(tensor);
        } else {
          new_alloc_tensors.push_back(info_->new_tensor);
        }
      }
      n->alloc_tensors = std::move(new_alloc_tensors);
    } else {
      ffi::Array<IterVar> new_iter_vars;
      for (size_t i = 0; i < stmt->iter_vars.size(); ++i) {
        auto old_iter_var = stmt->iter_vars[i];
        if (static_cast<int>(i) == info_->rolling_axis) {
          // All inner loops of the rolling axis has a loop carried dependency
          // (i.e. each iteration calculation of the rolling axis depends on
          // the calculation results of all the historical iterations of inner loops),
          // so annotate the iteration type of the rolling axis as 'opaque',
          // avoid the iterative range of its inner loop from being compressed
          // during lowering phase.
          IterVar new_iter_var =
              IterVar(old_iter_var->dom, old_iter_var->var, IterVarType::kOpaque);
          new_iter_vars.push_back(new_iter_var);
        } else {
          new_iter_vars.push_back(old_iter_var);
        }
      }
      ffi::Map<Var, TensorVar> tensor_data_to_tensor = {{info_->new_tensor, info_->new_tensor}};
      auto infered_access_regions = GetSBlockReadWriteRegion(stmt, tensor_data_to_tensor);

      n->iter_vars = std::move(new_iter_vars);
      RewriteAccessRegion(&n->reads, infered_access_regions[0]);
      RewriteAccessRegion(&n->writes, infered_access_regions[1]);
    }
    info_->block_reuse.Set(old_stmt, stmt);
    return stmt;
  }

  UnchangedOr<Stmt> Mutate_(const SBlockRealizeObj* realize, InplaceMode inplace_mode) final {
    SBlockRealize stmt = StmtExprMutator::Mutate_(realize, inplace_mode)
                             .ValueOrUnchanged(ffi::GetRef<Stmt>(realize))
                             .as_or_throw<SBlockRealize>();
    // Append block predicate to avoid recomputing elements.
    if (rewrite_block_predicate_) {
      rewrite_block_predicate_ = false;
      PrimExpr condition = stmt->predicate;
      for (size_t i = 0; i < info_->axis_iter_vars.size(); ++i) {
        auto iter_var = info_->axis_iter_vars[i];
        if (iter_var && info_->axis_overlaps[i] > 0) {
          Var var = iter_var.value();
          const ffi::Map<Var, sym::IntSet> dmap = {
              std::make_pair(var, sym::IntSet::Interval(0, 0))};
          auto iter_value = realize->iter_values[i];
          sym::Analyzer analyzer;
          auto term_2 = analyzer->int_set(iter_value, dmap).min();
          condition = analyzer->Simplify(And(condition, Or(LT(var.as_or_throw<PrimExpr>(), 1),
                                                           GE(term_2, info_->axis_overlaps[i]))));
        }
      }
      SBlockRealizeObj* n = stmt.CopyOnWrite();
      n->predicate = condition;
    }
    return stmt;
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreObj* op, InplaceMode inplace_mode) final {
    TensorStore stmt = StmtExprMutator::Mutate_(op, inplace_mode)
                           .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                           .as_or_throw<TensorStore>();
    if (stmt->dest.as_or_throw<TensorVar>().same_as(info_->old_tensor)) {
      TensorStoreObj* n = stmt.CopyOnWrite();
      TensorVar tensor = n->dest.as_or_throw<TensorVar>();
      RewriteTensorAccess(&tensor, &n->indices);
      n->dest = std::move(tensor);
      // Need to add predicate to the current block to avoid recomputing elements.
      rewrite_block_predicate_ = true;
    }
    return stmt;
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* op, InplaceMode inplace_mode) final {
    TensorLoad stmt = StmtExprMutator::Mutate_(op, inplace_mode)
                          .ValueOrUnchanged(ffi::GetRef<PrimExpr>(op))
                          .as_or_throw<TensorLoad>();
    if (stmt->source.as_or_throw<tvm::tirx::TensorVar>().same_as(info_->old_tensor)) {
      TensorVar tensor = stmt->source.as_or_throw<tvm::tirx::TensorVar>();
      ffi::Array<PrimExpr> indices = stmt->indices;
      RewriteTensorAccess(&tensor, &indices);
      return TensorLoad(tensor, indices, stmt->loc);
    }
    return stmt;
  }

  const StmtSRef& scope_sref_;
  RollingBufferInfo* info_;
  bool rewrite_block_predicate_ = false;
};

}  // namespace

void RollingBuffer(ScheduleState self, const StmtSRef& block_sref, int write_tensor_index) {
  /*!
   *  Check
   *    - The block is not an output block.
   *    - The block has only RAW dependencies.
   *    - The block is tiled and there is access overlap between adjacent tiles.
   *  Mutate
   *    - Select the outermost rollable axis appeared in the block's loop nest
   *      as the 'rolling axis', trim the target tensor from the rolling axis.
   *    - Use modulo arithmetic to modify the target tensor's read and load
   *      indices to circularize the tensor along the rolling dimension.
   *    - Append block predicate to avoid recomputing overlapping elements.
   */
  ffi::Map<Var, sym::IntSet> dom_map;
  const SBlockRealize& realize = GetSBlockRealize(self, block_sref);
  const SBlock& block = realize->block;

  // Step 1. Checking index, getting the target tensor region and the parent scope.
  const TensorRegion& tensor_region =
      GetNthAccessTensorRegion(self, block, write_tensor_index, TensorIndexType::kWrite);
  StmtSRef scope_root_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/false);
  // Step 2. Check if the target block is not an output block and has only RAW dependencies.
  CheckNotOutputBlock(self, block_sref, scope_root_sref);
  RollingBufferDependencyError::Check(self, block_sref, scope_root_sref);

  // Step 3. Find the lca of the access location of the target tensor and relax the tensor
  ffi::Array<StmtSRef> loop_srefs = GetLoops(block_sref);
  ffi::Array<StmtSRef> consumers_sref = GetConsumers(self, block_sref);
  consumers_sref.push_back(block_sref);
  StmtSRef lca = GetSRefLowestCommonAncestor(consumers_sref);
  if (!lca->StmtAs<ForObj>()) {
    throw MakeScheduleError<RollingBufferInsertionError>(
        self->mod, tensor_region->source.as_or_throw<tvm::tirx::TensorVar>(), block);
  }

  for (auto it = loop_srefs.rbegin(); it != loop_srefs.rend(); ++it) {
    auto stmt = *it;
    // Stop at the lca of all the rolling_buffer access points;
    if (stmt == lca) {
      break;
    }
    For cur_loop = ffi::GetRef<For>(stmt->StmtAs<ForObj>());
    Range range = Range(cur_loop->begin, cur_loop->end);
    dom_map.Set(cur_loop->var, sym::IntSet::FromRange(range));
  }
  TensorRegion relaxed_region = GetRelaxedTensorRegion(realize, tensor_region, dom_map);

  // Step 4. Find a valid rolling axis and collect bound overlaps on the target tensor.
  RollingBufferInfo info = RollingBufferInfoCollector::CheckAndGetRollingBufferInfo(
      self->mod, block_sref, relaxed_region);
  // Step 5. Mutate IR to apply rolling access pattern.
  Stmt new_scope_root = RollingBufferRewriter::Rewrite(scope_root_sref, &info);

  // Step 6. Update schedule states
  self->Replace(scope_root_sref, new_scope_root, info.block_reuse);
  // Step 7. Regenerate block info from the root block, because `region_cover` for the target block
  // and `stage_pipeline` for the root block are no longer satisfied after rolling tensor injection.
  self->UpdateScopeSBlockInfo(GetSBlockRealize(self, self->stmt2ref.at(new_scope_root.get())));
}

struct RollingBufferTraits : public UnpackedInstTraits<RollingBufferTraits> {
  static constexpr const char* kName = "RollingBuffer";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, SBlockRV block, int write_tensor_index) {
    return sch->RollingBuffer(block, write_tensor_index);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block,
                                      int write_tensor_index) {
    PythonAPICall py("rolling_buffer");
    py.Input("block", block);
    py.Input("write_tensor_index", write_tensor_index);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

TVM_FFI_STATIC_INIT_BLOCK() { RegisterInstructionKind<RollingBufferTraits>(); }
}  // namespace s_tir
}  // namespace tvm
