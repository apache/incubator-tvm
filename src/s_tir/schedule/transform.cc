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
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/tensor_intrin.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/type.h>

#include "./utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::prim;
using namespace tvm::tirx;

/******** Annotation ********/

SBlock WithAnnotation(const SBlockObj* block, const ffi::String& attr_key,
                      const ffi::ObjectRef& attr_value) {
  ffi::Map<ffi::String, Any> annotations = block->annotations;
  annotations.Set(attr_key, attr_value);
  ffi::ObjectPtr<SBlockObj> new_block = ffi::make_object<SBlockObj>(*block);
  new_block->annotations = std::move(annotations);
  return SBlock(new_block);
}

/******** Tensor Related ********/
TensorVar WithScope(const TensorVar& tensor, runtime::MemoryScope scope) {
  auto type = ffi::make_object<TensorTypeObj>(*tensor.ty().get());
  type->scope = scope;
  return TensorVar(tensor->name + "_" + scope.name(), TensorType(std::move(type)), tensor->loc);
}

TensorVar WithDType(const TensorVar& tensor, PrimType dtype) {
  auto type = ffi::make_object<TensorTypeObj>(*tensor.ty().get());
  type->dtype = dtype;
  return TensorVar(tensor->name, TensorType(std::move(type)), tensor->loc);
}

ffi::Array<TensorRegion> ReplaceTensor(ffi::Array<TensorRegion> regions, const TensorVar& source,
                                       const TensorVar& target) {
  regions.MutateByApply([&source, &target](TensorRegion region) -> TensorRegion {
    if (region->source.as_or_throw<tvm::tirx::TensorVar>().same_as(source)) {
      ffi::ObjectPtr<TensorRegionObj> n = ffi::make_object<TensorRegionObj>(*region.get());
      n->source = target;
      return TensorRegion(n);
    }
    return region;
  });
  return regions;
}

ffi::Array<TensorRegion> ReplaceTensor(ffi::Array<TensorRegion> regions,
                                       const ffi::Map<TensorVar, TensorVar>& tensor_map) {
  regions.MutateByApply([&tensor_map](TensorRegion region) -> TensorRegion {
    if (tensor_map.count(region->source.as_or_throw<tvm::tirx::TensorVar>())) {
      ffi::ObjectPtr<TensorRegionObj> n = ffi::make_object<TensorRegionObj>(*region.get());
      n->source = tensor_map[region->source.as_or_throw<tvm::tirx::TensorVar>()];
      return TensorRegion(n);
    }
    return region;
  });
  return regions;
}

ffi::Array<MatchTensorRegion> ReplaceTensor(ffi::Array<MatchTensorRegion> match_tensors,
                                            const TensorVar& source, const TensorVar& target) {
  match_tensors.MutateByApply(
      [&source, &target](MatchTensorRegion match_tensor) -> MatchTensorRegion {
        if (match_tensor->source->source.as_or_throw<tvm::tirx::TensorVar>().same_as(source)) {
          ffi::ObjectPtr<MatchTensorRegionObj> n =
              ffi::make_object<MatchTensorRegionObj>(*match_tensor.get());
          n->source = TensorRegion(target, n->source->region, TensorRegionType());
          return MatchTensorRegion(ffi::UnsafeInit{}, n);
        }
        return match_tensor;
      });
  return match_tensors;
}

ffi::Array<TensorRegion> ReplaceTensorRegion(ffi::Array<TensorRegion> regions,
                                             const TensorVar& source_tensor,
                                             const TensorRegion& target) {
  regions.MutateByApply([&source_tensor, &target](const TensorRegion& region) -> TensorRegion {
    if (region->source.as_or_throw<tvm::tirx::TensorVar>().same_as(source_tensor)) {
      return target;
    }
    return region;
  });
  return regions;
}

ffi::Array<MatchTensorRegion> ReplaceTensorRegion(ffi::Array<MatchTensorRegion> match_tensors,
                                                  const TensorVar& source_tensor,
                                                  const TensorRegion& target) {
  match_tensors.MutateByApply([&source_tensor, &target](
                                  const MatchTensorRegion& match_tensor) -> MatchTensorRegion {
    if (match_tensor->source->source.as_or_throw<tvm::tirx::TensorVar>().same_as(source_tensor)) {
      ffi::ObjectPtr<MatchTensorRegionObj> n =
          ffi::make_object<MatchTensorRegionObj>(*match_tensor.get());
      n->source = target;
      return MatchTensorRegion(ffi::UnsafeInit{}, n);
    }
    return match_tensor;
  });
  return match_tensors;
}

/******** ReplaceTensorMutator ********/
ReplaceTensorMutator::ReplaceTensorMutator(const TensorVar& old_tensor, TensorVar new_tensor,
                                           ffi::Map<SBlock, SBlock>* block_sref_reuse)
    : block_sref_reuse_(block_sref_reuse) {
  VarRemapSet(old_tensor, new_tensor);
}

ReplaceTensorMutator::ReplaceTensorMutator(const ffi::Map<TensorVar, TensorVar>& tensor_map,
                                           ffi::Map<SBlock, SBlock>* block_sref_reuse)
    : block_sref_reuse_(block_sref_reuse) {
  for (const auto& [old_tensor, new_tensor] : tensor_map) {
    VarRemapSet(old_tensor, new_tensor);
  }
}

UnchangedOr<Expr> ReplaceTensorMutator::Mutate_(const CallObj* op, InplaceMode inplace_mode) {
  auto result = StmtExprMutator::Mutate_(op, inplace_mode);
  if (!result.IsUnchanged()) {
    op = ffi::AnyView(result).as<CallObj>();
    if (!op->unique()) inplace_mode = InplaceMode::kDisallow;
  }
  if (!op->op.same_as(tirx::tensor_data_ptr_op()) || op->args.size() != 1) return result;
  PtrType type = op->args[0].as_or_throw<TensorVar>().ty()->DataPtrType();
  if (ffi::StructuralEqual()(op->ty, type)) return result;
  if (inplace_mode == InplaceMode::kAllow) {
    const_cast<CallObj*>(op)->ty = std::move(type);
    return result;
  }
  auto copy = ffi::make_object<CallObj>(*op);
  copy->ty = std::move(type);
  return Expr(std::move(copy));
}

MatchTensorRegion ReplaceTensorMutator::VisitMatchTensorRegion(
    const MatchTensorRegion& match_tensor) {
  if (auto replacement =
          VarRemapGet(match_tensor->source->source.as_or_throw<tvm::tirx::TensorVar>())
              .as<TensorVar>()) {
    return MatchTensorRegion(
        match_tensor->tensor,
        TensorRegion(replacement.value(), match_tensor->source->region, TensorRegionType()));
  } else {
    return match_tensor;
  }
}

UnchangedOr<Stmt> ReplaceTensorMutator::Mutate_(const SBlockObj* block, InplaceMode inplace_mode) {
  // To reduce the number of blocks in block sref reuse map, we check whether the block is really
  // mutated (i.e., the old tensor appears in the block). If so, we return the block after
  // mutation. Otherwise we just return the original block.

  auto f_mutate_match_tensor = [this](const MatchTensorRegion& match_tensor) {
    return this->VisitMatchTensorRegion(match_tensor);
  };
  auto f_mutate_read_write_region = [this](const TensorRegion& tensor_region) {
    auto region = tensor_region->region.Map([this](const Range& range) {
      auto min_result = Mutate(range->begin, InplaceMode::kDisallow);
      bool min_unchanged = min_result.UnchangedOrSameAs(range->begin);
      PrimExpr min = std::move(min_result).ValueOrUnchanged(range->begin);
      auto extent_result = Mutate(range->extent, InplaceMode::kDisallow);
      bool extent_unchanged = extent_result.UnchangedOrSameAs(range->extent);
      PrimExpr extent = std::move(extent_result).ValueOrUnchanged(range->extent);
      if (min_unchanged && extent_unchanged) {
        return range;
      } else {
        return Range::FromMinExtent(min, extent);
      }
    });

    TensorVar tensor = VarRemapGet(tensor_region->source.as_or_throw<tvm::tirx::TensorVar>())
                           .as<TensorVar>()
                           .value_or(tensor_region->source.as_or_throw<tvm::tirx::TensorVar>());

    if (tensor.same_as(tensor_region->source.as_or_throw<tvm::tirx::TensorVar>()) &&
        region.same_as(tensor_region->region)) {
      return tensor_region;
    } else {
      return TensorRegion(tensor, region, TensorRegionType());
    }
  };
  auto f_mutate_alloc_tensors = [this](const TensorVar& tensor) {
    return VarRemapGet(tensor).as<TensorVar>().value_or(tensor);
  };

  // Step 1. Mutate `match_tensors`. If an old tensor appears as a source of MatchTensorRegion,
  ffi::Array<MatchTensorRegion> match_tensors = block->match_tensors.Map(f_mutate_match_tensor);
  // Step 2. Mutate the read/write region.
  ffi::Array<TensorRegion> reads = block->reads.Map(f_mutate_read_write_region);
  ffi::Array<TensorRegion> writes = block->writes.Map(f_mutate_read_write_region);
  // Step 3. Mutate `alloc_tensors` for the old tensor allocated in this block.
  ffi::Array<TensorVar> alloc_tensors = block->alloc_tensors.Map(f_mutate_alloc_tensors);
  // Step 4. Recursively mutate the block.
  SBlock mutated_block = StmtExprMutator::Mutate_(block, inplace_mode)
                             .ValueOrUnchanged(ffi::GetRef<Stmt>(block))
                             .as_or_throw<SBlock>();

  if (mutated_block.get() == block && reads.same_as(mutated_block->reads) &&
      writes.same_as(mutated_block->writes) &&
      alloc_tensors.same_as(mutated_block->alloc_tensors) &&
      match_tensors.same_as(mutated_block->match_tensors)) {
    return ffi::Unchanged();
  } else {
    SBlockObj* n = mutated_block.CopyOnWrite();
    n->reads = std::move(reads);
    n->writes = std::move(writes);
    n->alloc_tensors = std::move(alloc_tensors);
    n->match_tensors = std::move(match_tensors);

    SBlock new_block = std::move(mutated_block);
    if (block_sref_reuse_ != nullptr) {
      block_sref_reuse_->Set(ffi::GetRef<SBlock>(block), new_block);
    }
    return new_block;
  }
}

/******** SBlock Removal ********/

void LeafBlockRemovalPlan(const ScheduleState& self, const StmtSRef& leaf_block_sref,
                          Stmt* src_stmt, Stmt* tgt_stmt) {
  class OnlyLeafError : public ScheduleErrorContextObj {
   public:
    explicit OnlyLeafError(IRModule mod, SBlock leaf_block, SBlock scope_root)
        : mod_(mod), leaf_block_(leaf_block), scope_root_(scope_root) {}

    ffi::String FastErrorString() const final {
      return "ScheduleError: Cannot remove the only leaf in the scope";
    }

    ffi::String DetailRenderTemplate() const final {
      return "Block {0} is the only leaf in the scope {1}, which cannot be removed; Otherwise the "
             "scope will be empty.";
    }

    IRModule mod() const final { return mod_; }
    ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final {
      return {leaf_block_, scope_root_};
    }

    IRModule mod_;
    SBlock leaf_block_;
    SBlock scope_root_;
  };

  // Go upwards until find an ancestor with more than one child
  const StmtObj* last_stmt = leaf_block_sref->stmt;
  StmtSRefObj* sref = leaf_block_sref->parent;
  for (;; last_stmt = sref->stmt, sref = sref->parent) {
    if (const auto* loop = sref->StmtAs<ForObj>()) {
      if (loop->body->size() > 1) {
        break;
      }
    } else {
      // Removal is not done beyond scope-level.
      // When encountering a block, i.e. the scope root, we simply stop
      break;
    }
  }
  if (const auto* block = sref->StmtAs<SBlockObj>()) {
    if (block->body->size() > 1) {
      ffi::ObjectPtr<SBlockObj> n = ffi::make_object<SBlockObj>(*block);
      auto new_seq = RemoveFromSeqStmt(block->body, ffi::GetRef<Stmt>(last_stmt));
      n->body = new_seq;
      *src_stmt = ffi::GetRef<Stmt>(block);
      *tgt_stmt = Stmt(std::move(n));
      return;
    }
  }
  if (const auto* loop = sref->StmtAs<ForObj>()) {
    if (loop->body->size() > 1) {
      ffi::ObjectPtr<ForObj> n = ffi::make_object<ForObj>(*loop);
      n->body = RemoveFromSeqStmt(loop->body, ffi::GetRef<Stmt>(last_stmt));
      *src_stmt = ffi::GetRef<Stmt>(loop);
      *tgt_stmt = Stmt(std::move(n));
      return;
    }
  }
  TVM_FFI_ICHECK(sref != nullptr && sref->stmt != nullptr);
  const auto* leaf_block = TVM_SREF_TO_SBLOCK(leaf_block_sref);
  const auto* scope_block = TVM_SREF_TO_SBLOCK(sref);
  throw MakeScheduleError<OnlyLeafError>(self->mod, ffi::GetRef<SBlock>(leaf_block),
                                         ffi::GetRef<SBlock>(scope_block));
}

ffi::Optional<LoopRV> TileWithTensorIntrin(const s_tir::Schedule& sch,
                                           const s_tir::SBlockRV& block_rv,
                                           const ffi::String& intrin_name, bool allow_padding) {
  ffi::Optional<TensorizeInfo> opt_tensorize_info =
      GetTensorizeLoopMapping(sch->state(), sch->GetSRef(block_rv),
                              TensorIntrin::Get(intrin_name).value()->desc, allow_padding);
  if (!opt_tensorize_info) return std::nullopt;
  const TensorizeInfoObj* info = opt_tensorize_info.value().get();
  if (info->block_iter_paddings.has_value()) {
    // We have to track whether each producer or consumer is padded.
    // To do so, we first record all the Block's.
    std::unordered_set<const StmtSRefObj*> original_producers, original_consumers;
    {
      for (const auto& p : GetProducers(sch->state(), sch->GetSRef(block_rv)))
        original_producers.insert(p.get());
      for (const auto& c : GetConsumers(sch->state(), sch->GetSRef(block_rv)))
        original_consumers.insert(c.get());
    }

    // Pad. Maybe we can make PadEinsum return the changes it made, to avoid bookkeeping?
    sch->PadEinsum(block_rv, info->block_iter_paddings.value());

    // Now we need to find out all the padded Block's.
    ffi::Array<SBlockRV> inlined_producers, inlined_consumers;
    for (const auto& producer : sch->GetProducers(block_rv)) {
      // PadEinsum will not modify the producer if it does not need padding.
      if (original_producers.count(sch->GetSRef(producer).get())) {
        // Producer not padded. No inlining.
        continue;
      }
      auto the_original_producers = sch->GetProducers(producer);
      if (the_original_producers.empty()) {
        // The original producer is input.
        continue;
      }
      TVM_FFI_ICHECK_EQ(the_original_producers.size(), 1u);
      auto the_original_producer = the_original_producers[0];
      TVM_FFI_ICHECK(original_producers.count(sch->GetSRef(the_original_producer).get()));
      inlined_producers.push_back(the_original_producer);
    }
    for (const auto& consumer : sch->GetConsumers(block_rv)) {
      // PadEinsum will not modify the consumer if it does not need padding.
      if (original_consumers.count(sch->GetSRef(consumer).get())) {
        // Consumer not padded. No inlining.
        continue;
      }
      auto the_original_consumers = sch->GetConsumers(consumer);
      if (the_original_consumers.empty()) {
        // The original consumer is output.
        continue;
      }
      TVM_FFI_ICHECK_EQ(the_original_consumers.size(), 1u);
      auto the_original_consumer = the_original_consumers[0];
      TVM_FFI_ICHECK(original_consumers.count(sch->GetSRef(the_original_consumer).get()));
      inlined_consumers.push_back(consumer);
    }

    // Inline the producer and consumer padding blocks
    for (const auto& the_original_producer : inlined_producers) {
      // Inline the original producer into the padding block. This ensures that the new producer
      // has the padded shape.
      sch->ComputeInline(the_original_producer);
    }
    for (const auto& consumer : inlined_consumers) {
      sch->ComputeInline(consumer);
    }
  }
  // Construct a mapping from tirx loops back to LoopRVs
  ffi::Map<tirx::StmtSRef, LoopRV> loop2rv;
  {
    ffi::Array<LoopRV> loop_rvs = sch->GetLoops(block_rv);
    for (const LoopRV& loop_rv : loop_rvs) {
      loop2rv.Set(sch->GetSRef(loop_rv), loop_rv);
    }
  }
  // Split the loops
  sym::Analyzer analyzer;
  std::unordered_set<const tirx::StmtSRefObj*> inner_loops;
  std::vector<LoopRV> reorder_suffix;
  reorder_suffix.resize(info->loop_map.size());
  for (const auto& kv : info->loop_map) {
    // Extract mapping (block_loop => desc_loop)
    const tirx::StmtSRef& block_loop_sref = kv.first;
    const ForObj* block_loop = block_loop_sref->StmtAs<ForObj>();
    const ForObj* desc_loop = kv.second.get();
    TVM_FFI_ICHECK(block_loop != nullptr && desc_loop != nullptr);
    // Extract the loop extent
    PrimExpr block_extent =
        analyzer->Simplify(Range::GetExtentFromBeginEnd(block_loop->begin, block_loop->end));
    PrimExpr desc_extent =
        analyzer->Simplify(Range::GetExtentFromBeginEnd(desc_loop->begin, desc_loop->end));
    const auto* int_block_extent = block_extent.as<IntImmObj>();
    const auto* int_desc_extent = desc_extent.as<IntImmObj>();
    TVM_FFI_ICHECK(int_block_extent != nullptr && int_desc_extent != nullptr);
    // Check divisibility
    const ffi::BigInt& total = int_block_extent->value;
    const ffi::BigInt& inner = int_desc_extent->value;
    TVM_FFI_ICHECK_EQ(total % inner, 0);
    // Do the split. Leave the outer extent as std::nullopt (unspecified) so that the split factors
    // can be used for different extents (needed during tuning).
    ffi::Array<LoopRV> split =
        sch->Split(loop2rv.at(block_loop_sref), {std::nullopt, IntImm::Int32(inner)});
    TVM_FFI_ICHECK_EQ(split.size(), 2);
    inner_loops.insert(sch->GetSRef(split[1]).operator->());
    // The inner split will be reordered to the loop domain that is tensorized
    int desc_loop_index = static_cast<int>(info->desc_loop_indexer.at(ffi::GetRef<For>(desc_loop)));
    reorder_suffix[desc_loop_index] = split[1];
  }
  // Reorder the loops
  std::vector<LoopRV> reorder_list;
  bool meet = false;
  ffi::Array<LoopRV> all_loops = sch->GetLoops(block_rv);
  for (const LoopRV& loop : all_loops) {
    if (inner_loops.count(sch->GetSRef(loop).operator->())) {
      meet = true;
    } else if (meet) {
      reorder_list.push_back(loop);
    }
  }
  reorder_list.insert(reorder_list.end(), reorder_suffix.begin(), reorder_suffix.end());
  sch->Reorder(reorder_list);
  TVM_FFI_ICHECK(!reorder_suffix.empty());
  return reorder_suffix[0];
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.schedule.TileWithTensorIntrin", TileWithTensorIntrin);
}

/******** BlockTensorAccessSimplifier ********/
void BlockTensorAccessSimplifier::SimplifyAccessRegion(
    ffi::Array<TensorRegion>* old_access_regions) {
  auto fmutate = [this](const TensorRegion& tensor_region) {
    ffi::Array<Range> new_tensor_region;
    ffi::Array<PrimExpr> simplified_min;
    for (const auto& range : tensor_region->region) {
      simplified_min.push_back(range->begin);
    }
    simplified_min = this->IterMapSimplifyWithContext(simplified_min, true);
    int n = tensor_region->region.size();
    for (int i = 0; i < n; ++i) {
      PrimExpr min = simplified_min[i];
      PrimExpr extent = GetSyncedAnalyzer()->Simplify(tensor_region->region[i]->extent);
      new_tensor_region.push_back(Range::FromMinExtent(min, extent));
    }
    return TensorRegion(tensor_region->source.as_or_throw<tvm::tirx::TensorVar>(),
                        new_tensor_region, TensorRegionType());
  };
  (*old_access_regions).MutateByApply(fmutate);
}

void BlockTensorAccessSimplifier::SimplifyTensorIndices(ffi::Array<PrimExpr>* indices) {
  *indices = this->IterMapSimplifyWithContext(*indices, true);
}

UnchangedOr<Stmt> BlockTensorAccessSimplifier::Mutate_(const SBlockObj* op,
                                                       InplaceMode inplace_mode) {
  SBlock block = s_tir::StmtExprMutatorWithConstraintContext::Mutate_(op, inplace_mode)
                     .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                     .as_or_throw<SBlock>();
  return WithConstraintScope([&]() -> ffi::Expected<UnchangedOr<Stmt>> {
           for (const auto& iter_var : block->iter_vars) {
             AppendConstraint(sym::RangeConstraint(iter_var->var, iter_var->dom.value()));
           }
           auto* n = block.CopyOnWrite();
           SimplifyAccessRegion(&n->reads);
           SimplifyAccessRegion(&n->writes);
           return block;
         })
      .value();
}

UnchangedOr<Stmt> BlockTensorAccessSimplifier::Mutate_(const TensorStoreObj* op,
                                                       InplaceMode inplace_mode) {
  TensorStore node = s_tir::StmtExprMutatorWithConstraintContext::Mutate_(op, inplace_mode)
                         .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                         .as_or_throw<TensorStore>();
  SimplifyTensorIndices(&node.CopyOnWrite()->indices);
  return node;
}

UnchangedOr<PrimExpr> BlockTensorAccessSimplifier::Mutate_(const TensorLoadObj* op,
                                                           InplaceMode inplace_mode) {
  TensorLoad node = s_tir::StmtExprMutatorWithConstraintContext::Mutate_(op, inplace_mode)
                        .ValueOrUnchanged(ffi::GetRef<PrimExpr>(op))
                        .as_or_throw<TensorLoad>();
  SimplifyTensorIndices(&node.CopyOnWrite()->indices);
  return node;
}

/******** Function-level analysis and transformation ********/

void GetLeafBlocksHelper(Schedule sch, SBlockRV cur_block_rv, ffi::Array<SBlockRV>* leaf_blocks) {
  ffi::Array<SBlockRV> blocks = sch->GetChildBlocks(cur_block_rv);
  if (blocks.empty()) {
    leaf_blocks->push_back(cur_block_rv);
  } else {
    for (const SBlockRV& block : blocks) {
      GetLeafBlocksHelper(sch, block, leaf_blocks);
    }
  }
}

ffi::Optional<ffi::ObjectRef> NormalizeFunction(Schedule sch) {
  SBlockRV root_block = sch->GetSBlock("root");
  ffi::Array<SBlockRV> leaf_blocks;
  GetLeafBlocksHelper(sch, root_block, &leaf_blocks);
  for (const SBlockRV& block : leaf_blocks) {
    StmtSRef block_sref = sch->GetSRef(block);
    ffi::Array<StmtSRef> loops = GetLoops(block_sref);
    ffi::Array<PrimExpr> binds = GetSBlockRealize(sch->state(), block_sref)->iter_values;
    if (loops.size() == 0) continue;
    if (loops.size() != binds.size()) {
      return std::nullopt;
    }
    for (int i = 0, n = loops.size(); i < n; ++i) {
      const ForObj* loop = TVM_SREF_TO_FOR(loops[i]);
      if (binds[i].get() != loop->var.get()) {
        return std::nullopt;
      }
      if (!IsZero(loop->begin)) {
        return std::nullopt;
      }
    }
  }

  ffi::Array<ffi::Array<LoopRV>> block_loops;
  ffi::Array<ffi::Array<IterVar>> block_iters;
  ffi::Array<IntImm> block_is_reduction;
  for (const SBlockRV& block : leaf_blocks) {
    ffi::Array<IterVar> iters = sch->Get(block)->iter_vars;
    bool has_spatial_iter = false;
    ffi::Array<PrimVar> index_map_inputs;
    ffi::Array<PrimExpr> index_map_outputs;
    for (const IterVar& iter : sch->Get(block)->iter_vars) {
      PrimVar var = iter->var.CopyWithSuffix("");
      index_map_inputs.push_back(var);
      if (!IsOne(iter->dom.value()->extent)) {
        index_map_outputs.push_back(var);
        if (iter->iter_type == IterVarType::kDataPar) {
          has_spatial_iter = true;
        }
      }
    }
    if (index_map_outputs.empty() || !has_spatial_iter) {
      index_map_outputs.insert(index_map_outputs.begin(), IntImm::Int64(0));
    }
    try {
      sch->TransformBlockLayout(block, IndexMap(index_map_inputs, index_map_outputs));
    } catch (tvm::ffi::Error& e) {
      // Skip layout transformation when not transformable.
    }
    block_loops.push_back(sch->GetLoops(block));
    block_iters.push_back(sch->Get(block)->iter_vars);
    bool is_reduction = IsReductionBlock(sch->state(),         //
                                         sch->GetSRef(block),  //
                                         sch->GetSRef(root_block));
    block_is_reduction.push_back(IntImm::Bool(is_reduction));
  }
  return ffi::Array<ffi::ObjectRef>{leaf_blocks, block_loops, block_iters, block_is_reduction};
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.schedule.NormalizeFunction", NormalizeFunction);
}

}  // namespace s_tir
}  // namespace tvm
