/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file inject_software_pipeline.cc
 * \brief Transform annotated loops into pipelined one that parallelize producers and consumers
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/transform.h>
#include <tvm/target/target.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/type.h>

#include <map>
#include <unordered_set>

#include "../memory_scope.h"
#include "../schedule/utils.h"
#include "ir_utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::prim;
using namespace tvm::tirx;

namespace software_pipeline {

namespace {

ffi::Optional<Var> GetBufferDataVar(const ffi::Any& data) {
  if (auto var = data.as<Var>()) {
    return var;
  }
  if (const auto* call = data.as<CallObj>();
      call && call->op.same_as(tirx::tensor_data_ptr_op()) && call->args.size() == 1) {
    return call->args[0].as<Var>();
  }
  return std::nullopt;
}

}  // namespace

/*!
 * \brief Create a block and infer the access region with the given body.
 *
 * The result is a opaque block that doesn't contain any block iter vars. In case the body is a
 * block realize without predicate, it is unnecessary to create a new block, the block of the block
 * realize will be returned.
 *
 * \param body The body of the block.
 * \param tensor_data_to_tensor The map from tensor data to tensor.
 * \return The result block.
 */
SBlock MakeSBlock(const SeqStmt& body, const ffi::Map<Var, TensorVar>& tensor_data_to_tensor) {
  if (const SBlockRealizeObj* block_realize =
          body->size() == 1 ? body->seq[0].as<SBlockRealizeObj>() : nullptr) {
    if (IsOne(block_realize->predicate)) {
      // no need to create a new block
      return block_realize->block;
    }
  }
  SBlock block(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{}, /*name_hint=*/"", /*body*/ body);
  ffi::Array<ffi::Array<TensorRegion>> access =
      GetSBlockReadWriteRegion(block, tensor_data_to_tensor);
  SBlockObj* n = block.CopyOnWrite();
  n->reads = access[0];
  n->writes = access[1];
  return block;
}

/*! Structure that represents the provided annotation per block or loop. */
struct PipelineAnnotation {
  int stage;
  int order;
  bool async;
};

using PipelineInfo =
    std::unordered_map<SBlock, PipelineAnnotation, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

struct TensorAccessInfo {
  int def = -1;  // the defining stage of the tensor
  int use = -1;  // the last using stage of the tensor
};

class PipelineOpaqueAccessRewriter {
 public:
  /*!
   * \brief Constructor
   * \param tensor_data_to_tensor The map from tensor data to tensor.
   * \param tensor_remap The map from original tensor to the tensor with updated shape for
   *        multi-versioning in the software pipeline.
   * \param pipeline_loop The original loop to be software pipelined.
   * \param fragment_info Information about tensor core fragment
   */
  PipelineOpaqueAccessRewriter(const ffi::Map<Var, TensorVar>& tensor_data_to_tensor,
                               const ffi::Map<TensorVar, TensorVar>& tensor_remap,
                               const For& pipeline_loop,
                               const std::unordered_map<const VarObj*, FragmentInfo>& fragment_info)
      : tensor_data_to_tensor_(tensor_data_to_tensor),
        tensor_remap_(tensor_remap),
        pipeline_loop_(pipeline_loop),
        fragment_info_(fragment_info) {}

  Expr Rewrite(const Call& call) {
    // Intrinsic calls should be handled explicitly here as they are opaque accesses to
    // tensor.
    static const Op load_matrix_sync = Op::Get("tirx.gpu_load_matrix_sync");
    static const Op store_matrix_sync = Op::Get("tirx.gpu_store_matrix_sync");
    static const Op mma_sync = Op::Get("tirx.gpu_mma_sync");
    if (call->op.same_as(load_matrix_sync) || call->op.same_as(store_matrix_sync)) {
      const TensorVar& tensor = tensor_data_to_tensor_.at(GetBufferDataVar(call->args[0]).value());
      auto it = tensor_remap_.find(tensor);
      if (it != tensor_remap_.end()) {
        ffi::Array<Expr> new_args = call->args;
        const TensorVar& new_tensor = (*it).second;
        new_args.Set(0, new_tensor.data());
        new_args.Set(
            4, RewriteWmmaFragmentIndex(tensor, new_tensor, call->args[4].as_or_throw<PrimExpr>()));
        return Call(call->ty, call->op, new_args, call->attrs, call->ty_args, call->loc);
      }
    } else if (call->op.same_as(mma_sync)) {
      ffi::Array<Expr> new_args = call->args;
      for (int i = 0; i < 4; i++) {
        const Var& tensor_var = GetBufferDataVar(call->args[i * 2]).value();
        PrimExpr index = call->args[i * 2 + 1].as_or_throw<PrimExpr>();
        const TensorVar& tensor = tensor_data_to_tensor_.at(tensor_var);
        auto it = tensor_remap_.find(tensor);
        if (it != tensor_remap_.end()) {
          const TensorVar& new_tensor = (*it).second;
          PrimExpr new_index = RewriteWmmaFragmentIndex(tensor, (*it).second, index);
          new_args.Set(i * 2, new_tensor.data());
          new_args.Set(i * 2 + 1, new_index);
        }
      }
      return Call(call->ty, call->op, new_args, call->attrs, call->ty_args, call->loc);
    }
    return call;
  }

 private:
  int GetWmmaFragmentSize(const TensorVar& tensor) {
    auto it = fragment_info_.find(tensor.get());
    TVM_FFI_ICHECK(it != fragment_info_.end());
    const FragmentInfo& info = (*it).second;
    return info.GetSize();
  }

  PrimExpr RewriteWmmaFragmentIndex(const TensorVar& old_tensor, const TensorVar& new_tensor,
                                    const PrimExpr& old_index) {
    PrimExpr new_tensor_offset = old_index;

    int fragment_size = GetWmmaFragmentSize(old_tensor);
    PrimExpr offset =
        floordiv(foldl([](PrimExpr a, PrimExpr b, Location loc) { return mul(a, b, loc); },
                       IntImm::Int32(1), old_tensor.shape()),
                 fragment_size);
    new_tensor_offset +=
        floormod(pipeline_loop_->var - pipeline_loop_->begin, new_tensor.shape()[0]) * offset;
    return new_tensor_offset;
  }

  const ffi::Map<Var, TensorVar>& tensor_data_to_tensor_;
  const ffi::Map<TensorVar, TensorVar>& tensor_remap_;
  const For& pipeline_loop_;
  const std::unordered_map<const VarObj*, FragmentInfo>& fragment_info_;
};

/*!
 * \brief Rewriter for the body of the software pipeline. This pass inserts `floormod` to indices
 * of the remapped tensor to select the version corresponding to the pipeline stage.
 */
class PipelineBodyRewriter : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  /*!
   * \brief Constructor of PipelineBodyRewriter.
   * \param tensor_data_to_tensor The map from tensor data to tensor.
   * \param tensor_remap The map from original tensor to the tensor with updated shape for
   *        multi-versioning in the software pipeline.
   * \param pipeline_loop The original loop to be software pipelined.
   * \param access_all_versions Whether all versions the tensors in the software pipeline are
   *        accessed. This will be used to update block access region. In the prologue and epilogue
   *        of a two-stage software pipeline, only one version of these tensors are accessed.
   * \param fragment_info Information about tensor core fragment
   */
  PipelineBodyRewriter(const ffi::Map<Var, TensorVar>& tensor_data_to_tensor,
                       const ffi::Map<TensorVar, TensorVar>& tensor_remap, For pipeline_loop,
                       bool access_all_versions,
                       const std::unordered_map<const VarObj*, FragmentInfo>& fragment_info)
      : tensor_data_to_tensor_(tensor_data_to_tensor),
        pipeline_loop_(pipeline_loop),
        access_all_versions_(access_all_versions),
        opaque_access_rewriter_(tensor_data_to_tensor_, tensor_remap, pipeline_loop_,
                                fragment_info) {
    for (const auto& [original, remapped] : tensor_remap) {
      VarRemapSet(original, remapped);
      tensor_data_to_tensor_.Set(remapped, remapped);
    }
  }

 private:
  TensorRegion RewritePipelineTensorRegion(const TensorRegion& tensor_region) {
    if (auto replacement = VarRemapGet(tensor_region->source.as_or_throw<tvm::tirx::TensorVar>())
                               .as<TensorVar>()) {
      ffi::Array<Range> new_region = tensor_region->region;
      TensorVar new_tensor = replacement.value();
      // For pipeline tensors, relax the access region of the first dimension to full extent
      // if access_all_versions == true
      Range accessed_version =
          access_all_versions_
              ? Range::FromMinExtent(0, new_tensor.shape()[0])
              : Range::FromMinExtent(
                    floormod((pipeline_loop_->var - pipeline_loop_->begin), new_tensor.shape()[0]),
                    IntImm::Int32(1));
      new_region.insert(new_region.begin(), accessed_version);
      return TensorRegion(new_tensor, new_region, TensorRegionType());
    }
    return tensor_region;
  }

  UnchangedOr<Stmt> Mutate_(const SBlockObj* op, InplaceMode inplace_mode) final {
    for (const TensorVar& alloc_tensor : op->alloc_tensors) {
      tensor_data_to_tensor_.Set(alloc_tensor, alloc_tensor);
    }
    SBlock block = StmtExprMutator::Mutate_(op, inplace_mode)
                       .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                       .as_or_throw<SBlock>();
    for (const TensorVar& alloc_tensor : op->alloc_tensors) {
      tensor_data_to_tensor_.erase(alloc_tensor);
    }
    return block;
  }

  UnchangedOr<Expr> Mutate_(const TensorRegionObj* op, InplaceMode inplace_mode) final {
    if (!op->source.as<TensorVar>()) {
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }
    TensorRegion region = RewritePipelineTensorRegion(ffi::GetRef<TensorRegion>(op));
    return StmtExprMutator::Mutate_(region.get(), InplaceMode::kDisallow).ValueOrUnchanged(region);
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreObj* op, InplaceMode inplace_mode) final {
    auto replacement = VarRemapGet(op->dest.as_or_throw<TensorVar>()).as<TensorVar>();
    TensorStore store = StmtExprMutator::Mutate_(op, inplace_mode)
                            .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                            .as_or_throw<TensorStore>();
    if (!replacement) {
      return store;
    }
    TensorVar new_tensor = replacement.value();
    auto* n = store.CopyOnWrite();
    n->dest = new_tensor;
    PrimExpr version =
        floormod((pipeline_loop_->var - pipeline_loop_->begin), new_tensor.shape()[0]);
    n->indices.insert(n->indices.begin(), version);
    return store;
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* op, InplaceMode inplace_mode) final {
    auto replacement = VarRemapGet(op->source).as<TensorVar>();
    if (!replacement) return StmtExprMutator::Mutate_(op, inplace_mode);
    auto indices = Mutate(op->indices, inplace_mode)
                       .ValueOrUnchanged(op->indices)
                       .as_or_throw<ffi::Array<PrimExpr>>();
    TensorVar new_tensor = replacement.value();
    PrimExpr version =
        floormod((pipeline_loop_->var - pipeline_loop_->begin), new_tensor.shape()[0]);
    indices.insert(indices.begin(), version);
    return TensorLoad(new_tensor, indices, op->loc);
  }

  UnchangedOr<Expr> Mutate_(const CallObj* op, InplaceMode inplace_mode) final {
    const auto scope_mma_matrixA = GetRegisteredMemoryScope("mma.matrixA");
    const auto scope_mma_matrixB = GetRegisteredMemoryScope("mma.matrixB");

    if (op->op.same_as(tirx::tensor_data_ptr_op())) {
      TensorVar tensor = op->args[0].as_or_throw<TensorVar>();
      if (auto replacement = VarRemapGet(tensor).as<TensorVar>()) {
        TensorVar new_tensor = replacement.value();
        PrimExpr stride = tvm::tirx::MatchShapeStridePattern(new_tensor.ty()).value().get<1>()[0];
        if ((scope_mma_matrixA && tensor.scope() == *scope_mma_matrixA) ||
            (scope_mma_matrixB && tensor.scope() == *scope_mma_matrixB)) {
          // These logical matrix allocations are distributed over a warp.
          TVM_FFI_ICHECK_EQ(floormod(stride, 32).as_or_throw<IntImm>()->value, 0);
          stride = floordiv(stride, 32);
        }
        PrimType dtype = tensor.dtype();
        int bytes = (dtype.bits() * dtype.lanes() + 7) / 8;
        PrimExpr version =
            floormod(pipeline_loop_->var - pipeline_loop_->begin, new_tensor.shape()[0]);
        return Call(op->ty, tirx::ptr_byte_offset_op(),
                    {new_tensor.data(), version * stride * bytes}, {}, {}, op->loc);
      }
    }
    Call call = opaque_access_rewriter_.Rewrite(ffi::GetRef<Call>(op)).as_or_throw<Call>();
    return StmtExprMutator::Mutate_(call.get(), InplaceMode::kDisallow).ValueOrUnchanged(call);
  }

  ffi::Map<Var, TensorVar> tensor_data_to_tensor_;
  For pipeline_loop_;
  bool access_all_versions_;
  PipelineOpaqueAccessRewriter opaque_access_rewriter_;
};

/*!
 * \brief Rewriter for the software pipeline that rewrite a loop into a pipelined one.
 */
class PipelineRewriter : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  static Stmt Rewrite(
      ffi::Map<Var, TensorVar> tensor_data_to_tensor,
      const std::unordered_set<TensorVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>& double_buffers,
      const ffi::Array<TensorVar> pipeline_allocs, const For& pipeline_loop,
      const PipelineInfo& pipeline_info,
      const std::unordered_map<const VarObj*, FragmentInfo>& fragment_info,
      const ffi::Map<ffi::String, ffi::Any> preserved_annotations) {
    auto rewriter = ffi::make_object<PipelineRewriter>(
        tensor_data_to_tensor, double_buffers, pipeline_allocs, pipeline_loop, pipeline_info,
        fragment_info, preserved_annotations);
    return rewriter->BuildPipeline();
  }

  PipelineRewriter(
      ffi::Map<Var, TensorVar> tensor_data_to_tensor,
      const std::unordered_set<TensorVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>& double_buffers,
      const ffi::Array<TensorVar>& pipeline_allocs, const For& pipeline_loop,
      const PipelineInfo& pipeline_info,
      const std::unordered_map<const VarObj*, FragmentInfo>& fragment_info,
      const ffi::Map<ffi::String, ffi::Any> preserved_annotations)

      : tensor_data_to_tensor_(std::move(tensor_data_to_tensor)),
        double_buffers_(double_buffers),
        pipeline_allocs_(pipeline_allocs),
        pipeline_loop_(pipeline_loop),
        pipeline_info_(pipeline_info),
        fragment_info_(fragment_info),
        preserved_annotations_(preserved_annotations) {}

 private:
  Stmt BuildPipeline() {
    // Step 1: Analyze accesses to the tensors in the pipeline and compute the number of versions
    // need to maintain for each tensor.
    std::unordered_map<TensorVar, TensorAccessInfo, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> infos =
        GetTensorAccessInfo();
    for (const TensorVar& tensor : pipeline_allocs_) {
      int num_versions = ComputeTensorVersions(tensor, infos.at(tensor));
      if (num_versions > 1) {
        tensor_remap_.Set(tensor, RewriteAllocTensor(tensor, num_versions));
      }
    }
    for (const auto& [_, remapped] : tensor_remap_) {
      tensor_data_to_tensor_.Set(remapped, remapped);
    }

    ordered_stmts_.resize(pipeline_info_.size());
    for (const auto& pair : pipeline_info_) {
      const SBlock& block = pair.first;
      int order = pair.second.order;
      ordered_stmts_.Set(order, block);
    }

    // Step 2: Emit the pipeline prologue, body and epilogue.
    Stmt prologue = EmitImpl(pipeline_loop_->begin, pipeline_loop_->begin + max_stage_, true);
    Stmt body = EmitImpl(pipeline_loop_->begin + max_stage_, pipeline_loop_->end, false);
    // introduce extra lowerbound when the loop length is smaller than num stages
    // to ensure the epilogue interval do not overlap the prologue interval.
    PrimExpr epigogue_start = pipeline_loop_->end;
    ffi::Optional<PrimExpr> extra_epilogue_lower_bound = std::nullopt;
    if (max_stage_ > 1 &&
        !analyzer_->CanProveGreaterEqual(
            Range::GetExtentFromBeginEnd(pipeline_loop_->begin, pipeline_loop_->end), max_stage_)) {
      if (IsConstInt(epigogue_start)) {
        epigogue_start = max(epigogue_start, pipeline_loop_->begin + max_stage_);
      } else {
        // for dynamic case, introduce extra lowerbound as loop predicate
        // to ensure the epilogue part unrollable.
        extra_epilogue_lower_bound = pipeline_loop_->begin + max_stage_;
      }
    }
    Stmt epilogue = EmitImpl(epigogue_start, pipeline_loop_->end + max_stage_, true,
                             extra_epilogue_lower_bound);

    SeqStmt stmt = SeqStmt({prologue, body, epilogue});

    // Step 3: Make a new block that contains new tensor allocations after pipeline rewriting.
    ffi::Array<TensorVar> alloc_tensors;
    for (const auto& alloc : pipeline_allocs_) {
      TensorVar remapped = tensor_remap_.Get(alloc).value_or(alloc);
      alloc_tensors.push_back(remapped);
      tensor_data_to_tensor_.erase(alloc);
      tensor_data_to_tensor_.erase(remapped);
    }
    SBlock block = MakeSBlock(stmt, tensor_data_to_tensor_);
    block.CopyOnWrite()->alloc_tensors = std::move(alloc_tensors);
    return SBlockRealize({}, IntImm::Bool(true), block);
  }

  /*!
   * \brief Analyze accesses to the tensors in the software pipeline.
   *
   * This method check the 'define' and 'use' stage of the tensors in the software pipeline, which
   * can be used to compute the number of versions needed to maintain after rewriting.
   */
  std::unordered_map<TensorVar, TensorAccessInfo, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
  GetTensorAccessInfo() {
    std::unordered_map<TensorVar, TensorAccessInfo, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> infos;
    for (const auto& pair : pipeline_info_) {
      const SBlock& block = pair.first;
      int stage = pair.second.stage;
      max_stage_ = std::max(max_stage_, stage);

      for (const TensorRegion& write : block->writes) {
        if (!infos.count(write->source.as_or_throw<tvm::tirx::TensorVar>())) {
          infos.emplace(write->source.as_or_throw<tvm::tirx::TensorVar>(), TensorAccessInfo{});
        }
        auto& info = infos.at(write->source.as_or_throw<tvm::tirx::TensorVar>());
        if (info.def == -1) {
          info.def = stage;
        } else {
          info.def = std::min(info.def, stage);
        }
      }

      for (const TensorRegion& read : block->reads) {
        if (!infos.count(read->source.as_or_throw<tvm::tirx::TensorVar>())) {
          infos.emplace(read->source.as_or_throw<tvm::tirx::TensorVar>(), TensorAccessInfo{});
        }
        auto& info = infos.at(read->source.as_or_throw<tvm::tirx::TensorVar>());
        info.use = std::max(info.use, stage);
      }
    }
    return infos;
  }

  /*!
   * \brief Check whether two regions have intersections.
   * \param region1 The first region.
   * \param region2 The second region.
   * \return Whether region1 and region2 have intersections.
   */
  bool MayConflict(ffi::Array<Range> region1, ffi::Array<Range> region2) {
    TVM_FFI_ICHECK(region1.size() == region2.size());
    for (size_t i = 0; i < region1.size(); i++) {
      Range dim1 = region1[i];
      Range dim2 = region2[i];
      auto int_set1 = sym::IntSet::FromRange(dim1);
      auto int_set2 = sym::IntSet::FromRange(dim2);
      if (sym::Intersect({int_set1, int_set2}).IsNothing()) {
        return false;
      }
    }
    return true;
  }

  /*!
   * \brief Compute the number of versions need to maintain for tensor accessed in the software
   * pipeline.
   *
   * This method applies liveness analysis to the target tensor to compute the number of versions
   * need to maintain during the software pipeline.
   * Annotation `attr::double_buffer_scope` is handled here which provides a way to override the
   * result of the analysis. Additional double buffering in the software pipeline can be useful
   * to eliminate synchronizations in GPU devices.
   *
   * \param tensor The target tensor
   * \param tensor_info The access information of the target tensor.
   * \return The number of versions required for the target tensor.
   */
  int ComputeTensorVersions(const TensorVar& tensor, const TensorAccessInfo& tensor_info) {
    if (tensor_info.def == -1) {
      // Keep the original number of versions as tensors defined outside the software pipeline
      // should not be mutated.
      return 1;
    }

    // `use - def + 1` is a upper bound of the needed versions
    // We optimize a few case where the number of versions can be smaller than the upper bound
    int num_versions = tensor_info.use - tensor_info.def + 1;
    if (num_versions == 2) {
      // A special case when `use - def + 1 == 2`. Double buffering is only needed in this case when
      // these exists a reader block_i and a writer block_j such that
      // order(block_i) < order(block_j) and stage(block_i) < stage(block_j) and the access regions
      // of block_i and block_j overlap.
      bool need_multi_version = false;
      for (const auto& pair1 : pipeline_info_) {
        const SBlock& writer_block = pair1.first;
        const auto& writer_info = pair1.second;

        auto it1 = std::find_if(
            writer_block->writes.begin(), writer_block->writes.end(),
            [&](const TensorRegion& tensor_region) {
              return tensor_region->source.as_or_throw<tvm::tirx::TensorVar>().same_as(tensor);
            });
        if (it1 == writer_block->writes.end()) {
          continue;
        }

        for (const auto& pair2 : pipeline_info_) {
          const SBlock& reader_block = pair2.first;
          const auto& reader_info = pair2.second;
          auto it2 = std::find_if(
              reader_block->reads.begin(), reader_block->reads.end(),
              [&](const TensorRegion& tensor_region) {
                return tensor_region->source.as_or_throw<tvm::tirx::TensorVar>().same_as(tensor);
              });
          if (it2 == reader_block->reads.end()) {
            continue;
          }
          if (writer_info.order < reader_info.order && writer_info.stage < reader_info.stage &&
              MayConflict((*it1)->region, (*it2)->region)) {
            need_multi_version = true;
            break;
          }
        }
      }
      if (!need_multi_version) {
        num_versions = 1;
      }
    }
    if (num_versions == 1 && double_buffers_.count(tensor)) {
      num_versions = 2;
    }
    return num_versions;
  }

  /*!
   * \brief Rewrite tensor allocation to keep multiple versions of original tensor for pipelined
   * accesses.
   * \param tensor The tensor to be resized.
   * \param num_versions The number of versions to keep.
   * \return The resized tensor.
   */
  TensorVar RewriteAllocTensor(const TensorVar& tensor, int num_versions) {
    ffi::ObjectPtr<TensorTypeObj> new_tensor = ffi::make_object<TensorTypeObj>(*tensor.ty().get());
    auto strides = tvm::tirx::MatchShapeStridePattern(tensor.ty()).value().get<1>();
    new_tensor->shape.insert(new_tensor->shape.begin(), PrimExpr(num_versions));
    if (new_tensor->layout.has_value()) {
      TVM_FFI_ICHECK(strides.size() + 1 == new_tensor->shape.size());
      PrimExpr stride_0 = strides[0] * new_tensor->shape[1];
      strides.insert(strides.begin(), stride_0);
      static const auto memory_axis = tvm::tirx::Axis::Get("m");
      ffi::Array<tvm::tirx::Iter> shard;
      for (size_t i = 0; i < new_tensor->shape.size(); ++i) {
        shard.push_back(tvm::tirx::Iter(new_tensor->shape[i], strides[i], memory_axis));
      }
      ffi::Map<tvm::tirx::Axis, PrimExpr> offset;
      if (new_tensor->layout.has_value())
        offset = new_tensor->layout.value().as_or_throw<tvm::tirx::TileLayout>()->offset;
      new_tensor->layout = tvm::tirx::TileLayout(std::move(shard), {}, std::move(offset));
    }
    return TensorVar(tensor->name, TensorType(std::move(new_tensor)), tensor->loc);
  }

  // Per-stage states that need to be tracked across pipeline prologue, body, and epilogue.
  struct AsyncStateGlobal {
    // Tensors that this stage asynchronously writes.
    std::unordered_set<const VarObj*> dst_tensors;
    // An imaginary index that the latest async operation associated with this stage has written
    // into. Only valid if all associated predicates are true, so that we can count the number of
    // async invocations exactly. When it is valid, it is the "sum of extents of loops that have
    // been executed" - 1, e.g. for epilogue it is prologue extent + body extent - 1. This
    // is only needed to compute wait count for epilogue without async producers.
    ffi::Optional<PrimExpr> producer_head{PrimExpr(-1)};

    bool writes(TensorVar tensor) const { return dst_tensors.count(tensor.get()) > 0; }
  };

  // Per-stage states that are local to each of pipeline prologue, body, and epilogue.
  struct AsyncStateLocal {
    struct {
      // The index into a list of blocks, where async_wait_queue should be attached at the
      // beginning.
      int insert_before;
      // in_flight_count would be a more precise name, but the implementation uses wait_count for
      // brevity.
      ffi::Optional<PrimExpr> wait_count;

      bool valid() const { return wait_count.has_value(); }
    } pending_wait;

    // Destination tensors of async operations that have been encountered so far in the loop
    //
    // for (size_t i = 0; i < new_blocks.size(); ++i) {
    //    ...
    // }
    //
    // This is for tracking which async operations have been issued at the "current" iteration, up
    // until a point where we encounter a consumer of async result tensors. This is used to decide
    // if the producer_head of each tensor points to a copy written in the current or previous
    // iteration.
    std::unordered_set<const VarObj*> seen;

    // A symbolic expression representing the index the latest async operation associated with this
    // stage has written into, at the "current" iteration.
    ffi::Optional<PrimExpr> producer_head;
    // The predicate of BlockRealize containing the async operation of this stage.
    ffi::Optional<PrimExpr> predicate;
    // Indices into a list of blocks, where async_commit_queue scope should be attached.
    // If multiple async producers are interleaved with their consumer in between, we need separate
    // async_commit_queue for each producer. Thus, we need multiple sets of indices.
    std::vector<std::vector<size_t>> commit_groups;

    // This is set to true when we reach a stage that consumes this async stage.
    bool consumed{false};
  };

  /*! Structure holding intermediate information for pipeline loop rewriting. */
  struct RewrittenSBlockInfo {
    int stage;
    PrimExpr predicate;
    SBlock block;
    PrimExpr access_index;
    bool is_async;
  };

  // Determine where to insert async_wait and the corresponding wait count.
  void PopulateWaitCounts(const std::vector<RewrittenSBlockInfo>& new_blocks,
                          sym::AnalyzerObj* ana_normalized,
                          const std::unordered_map<const VarObj*, int>& tensor_to_commit_group,
                          std::map<int, AsyncStateLocal>* async_states_local) {
    for (size_t i = 0; i < new_blocks.size(); ++i) {
      if (new_blocks[i].is_async) {
        // Record the fact that we have encountered these write tensors.
        for (auto write_region : new_blocks[i].block->writes) {
          (*async_states_local)[new_blocks[i].stage].seen.insert(
              write_region->source.as_or_throw<tvm::tirx::TensorVar>().get());
        }
      }

      int producer_stage_idx = -1;
      for (auto read_region : new_blocks[i].block->reads) {
        for (auto kv : async_states) {
          if (kv.first <= new_blocks[i].stage &&
              kv.second.writes(read_region->source.as_or_throw<tvm::tirx::TensorVar>())) {
            // Found an earlier stage where read_region->source.as_or_throw<tvm::tirx::TensorVar>()
            // was asynchronously written
            TVM_FFI_ICHECK(producer_stage_idx == -1 || producer_stage_idx == kv.first)
                << "A dependency on multiple async stages is not supported";
            producer_stage_idx = kv.first;
          }
        }
      }

      if (producer_stage_idx == -1) continue;

      // The following logic has become complicated to handle case like this:
      //
      // for i in range(13):
      //     # Stage 0
      //     with async_copy_scope():
      //        A_shared[(i + 3) % 4] = A[...]
      //     async_commit(0)
      //
      //
      //     # Stage 1
      //     async_wait(0, 5)
      //     compute(A_shared[i], B_shared[i])
      //
      //     # Stage 0
      //     with async_copy_scope():
      //        B_shared[(i + 3) % 4] = B[...]
      //     async_commit(0)
      //
      //
      // Here, multiple async producers in the same stage are interleaved with their consumer in
      // between. Since each tensor is associated with different commit groups, the wait_count
      // before the consumer should be bigger than the simpler case:
      //
      // for i in range(13):
      //     # Stage 0
      //     with async_copy_scope():
      //        A_shared[(i + 3) % 4] = A[...]
      //        B_shared[(i + 3) % 4] = B[...]
      //     async_commit(0)
      //
      //     # Stage 1
      //     async_wait(0, 3)
      //     compute(A_shared[i], B_shared[i])
      //
      // The correct wait_count can be determined by considering each commit group separately, and
      // summing "per-commit" wait_counts.
      //
      // From A_shared's perspective, it allows for (i + 3) - i async commit groups to be in
      // flight while from B_shared's perspective, the producer head at compute points to the copy
      // done by the previous iteration, so its wait_count is calculated as ((i - 1) + 3) - i. The
      // sum of the two wait_counts gives 5.

      auto& dep_local_state = (*async_states_local)[producer_stage_idx];
      const auto num_commit_group = dep_local_state.commit_groups.size();
      std::vector<ffi::Optional<PrimExpr>> producer_head_per_commit;

      if (num_commit_group == 0) {
        // Epilogue, no async producer. Since "local" producer_head is not available, use
        // "global" producer_head.
        TVM_FFI_ICHECK(!dep_local_state.producer_head);
        producer_head_per_commit.push_back(async_states[producer_stage_idx].producer_head);
      } else {
        TVM_FFI_ICHECK(dep_local_state.producer_head);
        std::vector<bool> need_wait_count(num_commit_group, true);

        for (auto read_region : new_blocks[i].block->reads) {
          if (!async_states[producer_stage_idx].writes(
                  read_region->source.as_or_throw<tvm::tirx::TensorVar>()))
            continue;
          auto commit_group_id = tensor_to_commit_group.at(
              read_region->source.as_or_throw<tvm::tirx::TensorVar>().get());
          if (!need_wait_count[commit_group_id]) continue;

          if (!dep_local_state.seen.count(
                  read_region->source.as_or_throw<tvm::tirx::TensorVar>().get())) {
            // Multiple async producers interleaved: The most recent async write is from the
            // previous iteration. This is the B_shared case above.
            producer_head_per_commit.push_back(dep_local_state.producer_head.value() - 1);
          } else {
            // Normal case
            producer_head_per_commit.push_back(dep_local_state.producer_head.value());
          }

          need_wait_count[commit_group_id] = false;
        }
      }

      auto wait_count = [=, this, &ana_normalized]() {
        auto sum = PrimExpr(0);
        for (auto producer_head : producer_head_per_commit) {
          if (producer_head && ana_normalized->CanProve(producer_head.value() >= 0)) {
            // Here, new_blocks[i].access_index corresponds to "consumer_head".
            // The difference of producer_head and consumer_head is precisely the number of
            // async commit groups that can still be in flight after this wait.
            sum += analyzer_->Simplify(producer_head.value() - new_blocks[i].access_index);
          } else {
            // The precise count cannot be determined, give up.
            return PrimExpr(0);
          }
        }
        return sum;
      }();

      auto& pending_wait = dep_local_state.pending_wait;

      if (!pending_wait.valid()) {
        pending_wait = {static_cast<int>(i), wait_count};
      } else if (analyzer_->CanProve(wait_count < pending_wait.wait_count.value())) {
        // Coalesce multiple wait_queue if the later one allows fewer in-flight ops.
        pending_wait = {pending_wait.insert_before, wait_count};
      }
    }
  }

  // Given pipelined blocks and async-related information, generate final loop statements with async
  // scopes (if any).
  ffi::Array<Stmt> CompletePipelineLoopStatements(
      const std::vector<RewrittenSBlockInfo>& blocks,
      const std::map<int, AsyncStateLocal>& async_states_local,
      sym::AnalyzerObj* ana_normalized) const {
    std::vector<RewrittenSBlockInfo> new_blocks = blocks;
    std::vector<int> commit_group_indices(new_blocks.size(), -1);
    for (const auto& [stage_id, state] : async_states_local) {
      if (!state.commit_groups.empty()) {
        for (size_t i = 0; i < state.commit_groups.size(); ++i) {
          for (size_t j = 0; j < state.commit_groups[i].size(); ++j) {
            TVM_FFI_ICHECK(state.commit_groups[i][0] + j < new_blocks.size());
            commit_group_indices[state.commit_groups[i][0] + j] = stage_id;
          }
        }
      }

      if (state.pending_wait.valid()) {
        auto attach_wait = [&new_blocks](int i, int stage_id, PrimExpr wait_count) {
          auto& block = new_blocks[i].block;
          SBlockObj* n = block.CopyOnWrite();
          n->body = SeqStmt({Evaluate(Call(PrimType::Void(), s_tir::async_wait(),
                                           {IntImm::Int32(stage_id), wait_count})),
                             n->body});
        };

        if (state.predicate && !ana_normalized->CanProve(state.predicate.value())) {
          // If the async operation that this wait_queue is waiting on is predicated, and we cannot
          // prove that the predicate is always true, the precise wait count is only valid
          // at iterations where the predicate is true;
          auto wait_count = Call(PrimType::Int(32), prim::if_then_else_op(),
                                 ffi::Array<PrimExpr>{state.predicate.value(),
                                                      state.pending_wait.wait_count.value(), 0})
                                .as_or_throw<PrimExpr>();
          attach_wait(state.pending_wait.insert_before, stage_id, wait_count);
        } else {
          attach_wait(state.pending_wait.insert_before, stage_id,
                      state.pending_wait.wait_count.value());
        }
      }
    }

    ffi::Array<Stmt> stmts;

    for (size_t i = 0; i < new_blocks.size();) {
      if (commit_group_indices[i] == -1) {
        // A synchrnous block, not part of any commit group
        stmts.push_back(SBlockRealize({}, new_blocks[i].predicate, new_blocks[i].block));
        ++i;
      } else {
        ffi::Array<Stmt> group_bodies;
        auto stage_id = commit_group_indices[i];
        auto predicate = new_blocks[i].predicate;
        for (; i < commit_group_indices.size() && commit_group_indices[i] == stage_id; ++i) {
          TVM_FFI_ICHECK(ffi::StructuralEqual()(predicate, new_blocks[i].predicate))
              << "Predicates in the same stage are expected to be identical";
          group_bodies.push_back(new_blocks[i].block->body);
        }

        group_bodies.push_back(
            Evaluate(Call(PrimType::Void(), s_tir::async_commit(), {IntImm::Int32(stage_id)})));
        auto new_block = MakeSBlock(SeqStmt(group_bodies), tensor_data_to_tensor_);
        stmts.push_back(SBlockRealize({}, predicate, new_block));
      }
    }

    return stmts;
  }

  /*!
   * \brief Emit the pipeline loop in the given range.
   * \param start The start of the range
   * \param end The end of the range
   * \param unroll_loop Whether the loop should be unrolled.
   * \param extra_loop_lower_bound Extra loop lower bound.
   * \return The result loop.
   */
  Stmt EmitImpl(PrimExpr start, PrimExpr end, bool unroll_loop,
                ffi::Optional<PrimExpr> extra_loop_lower_bound = std::nullopt) {
    PrimExpr new_loop_var{ffi::UnsafeInit{}};
    PrimExpr extent = end - start;

    auto make_nop = []() {
      return SBlockRealize({}, IntImm::Bool(true), MakeSBlock(Evaluate(0), {}));
    };

    if (analyzer_->CanProve(extent <= 0)) {
      return make_nop();
    }
    bool is_unit_loop = analyzer_->CanProveEqual(extent, 1);
    if (is_unit_loop) {
      new_loop_var = start;  // use constants as the loop var for unit loops
    } else {
      new_loop_var = pipeline_loop_->var.CopyWithSuffix("");
      {
        auto context = analyzer_->GetConstraintContext();
        context->Append(sym::RangeConstraint(new_loop_var.as_or_throw<Var>(), Range(start, end)));
        analyzer_->SyncConstraintContext(context);
      }
    }

    // In contrast to analyzer_ which is bound to [start, end), this one is bound to
    // the "normalized" range, [pipeline_loop_->begin, extent).
    sym::Analyzer ana_normalized;
    if (!is_unit_loop) {
      {
        auto context = ana_normalized->GetConstraintContext();
        context->Append(sym::RangeConstraint(new_loop_var.as_or_throw<Var>(),
                                             Range(pipeline_loop_->begin, extent)));
        ana_normalized->SyncConstraintContext(context);
      }
    }

    std::vector<RewrittenSBlockInfo> new_blocks;

    // Async related
    std::map<int, AsyncStateLocal> async_states_local;
    std::unordered_map<const VarObj*, int> tensor_to_commit_group;

    for (const SBlock& block : ordered_stmts_) {
      int stage = pipeline_info_.at(block).stage;
      PrimExpr skewed_loop_var = new_loop_var - stage;
      PrimExpr inbound = analyzer_->Simplify(pipeline_loop_->begin <= skewed_loop_var) &&
                         (skewed_loop_var < pipeline_loop_->end);
      if (extra_loop_lower_bound.has_value()) {
        inbound = analyzer_->Simplify(inbound && new_loop_var >= extra_loop_lower_bound.value());
      }
      if (analyzer_->CanProve(!inbound)) {
        continue;
      }
      SBlock new_block =
          ffi::make_object<PipelineBodyRewriter>(tensor_data_to_tensor_, tensor_remap_,
                                                 pipeline_loop_, max_stage_ != 1, fragment_info_)
              ->Mutate(block)
              .ValueOrUnchanged(block)
              .as_or_throw<SBlock>();

      PrimExpr delta = start - pipeline_loop_->begin;
      // This variable corresponds to
      // - "producer_head" if this stage is an async producer
      // - "consumer_head" if this stage reads from asynchronously written tensors.
      PrimExpr skewed_loop_prim = skewed_loop_var.as_or_throw<PrimExpr>();
      PrimExpr normalized_access_index = is_unit_loop ? skewed_loop_prim : skewed_loop_prim + delta;

      // Adjust the block predicate and the body according to the final loop bound
      //  [pipeline_loop_->begin, extent).
      if (!is_unit_loop) {
        Var loop_iter = new_loop_var.as_or_throw<Var>();
        auto f_substitute = [loop_iter,
                             delta](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
          if (var.same_as(loop_iter)) {
            return ffi::Any(loop_iter.as_or_throw<PrimExpr>() + delta);
          }
          return ffi::Unchanged();
        };
        inbound = ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(inbound, f_substitute)
                      .as_or_throw<PrimExpr>();
      }

      auto f_substitute = [this, &normalized_access_index](
                              const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
        if (var.same_as(pipeline_loop_->var)) {
          return ffi::Any(normalized_access_index);
        }
        return ffi::Unchanged();
      };
      new_block = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(new_block, f_substitute)
                      .as_or_throw<SBlock>();

      if (pipeline_info_[block].async) {
        auto& local_state = async_states_local[stage];

        int commit_group_id = -1;
        if (local_state.commit_groups.empty() || local_state.consumed) {
          // consumed == true means there is already a consumer stage waiting for an
          // eariler async operation of this stage. In such cases, we make multiple commit_queue
          // for this stage.
          commit_group_id = local_state.commit_groups.size();
          local_state.commit_groups.push_back({new_blocks.size()});
        } else {
          // One commit can group multiple async-copy regions.
          // with async_copy_scope():
          //   A_shared[...] = ...
          // with async_copy_scope():
          //   B_shared[...] = ...
          // async_commit(stage)

          commit_group_id = local_state.commit_groups.size() - 1;
          local_state.commit_groups.back().push_back(new_blocks.size());
        }

        for (auto write_region : new_block->writes) {
          async_states[stage].dst_tensors.insert(
              write_region->source.as_or_throw<tvm::tirx::TensorVar>().get());
          tensor_to_commit_group[write_region->source.as_or_throw<tvm::tirx::TensorVar>().get()] =
              commit_group_id;
        }

        local_state.producer_head = normalized_access_index;

        if (!local_state.predicate || ana_normalized->CanProve(local_state.predicate.value())) {
          local_state.predicate = inbound;
        } else if (local_state.predicate) {
          local_state.predicate = ana_normalized->Simplify(local_state.predicate.value() & inbound);
        }

        SBlockObj* n = new_block.CopyOnWrite();
        n->body = RegionStmt(s_tir::async_copy_scope(), {}, {}, DictAttrs(), n->body);
      }

      new_blocks.push_back(
          {stage, inbound, new_block, normalized_access_index, pipeline_info_[block].async});

      for (auto read_region : new_block->reads) {
        for (auto kv : async_states) {
          int producer_stage_id = kv.first;
          if (producer_stage_id <= stage &&
              kv.second.writes(read_region->source.as_or_throw<tvm::tirx::TensorVar>())) {
            async_states_local[producer_stage_id].consumed = true;
          }
        }
      }
    }

    PopulateWaitCounts(new_blocks, ana_normalized.get(), tensor_to_commit_group,
                       &async_states_local);
    auto stmts =
        CompletePipelineLoopStatements(new_blocks, async_states_local, ana_normalized.get());

    if (stmts.empty()) {
      return SeqStmt({});
    }
    Stmt new_loop = SeqStmt(stmts);

    if (!is_unit_loop) {
      new_loop =
          For(new_loop_var.as_or_throw<PrimVar>(), pipeline_loop_->begin,
              pipeline_loop_->begin + extent, IntImm(new_loop_var.as_or_throw<PrimVar>().ty(), 1),
              unroll_loop ? ForKind::kUnrolled : pipeline_loop_->kind, std::move(new_loop),
              preserved_annotations_);
    }

    // Update producer heads in the global async states.
    for (const auto& kv : async_states_local) {
      const int stage_id = kv.first;
      const AsyncStateLocal& state = kv.second;

      if (state.predicate && ana_normalized->CanProve(state.predicate.value()) &&
          async_states[stage_id].producer_head) {
        // Advance the "global" producer head if it is still valid and we know exactly how much we
        // can increment
        async_states[stage_id].producer_head =
            async_states[stage_id].producer_head.value() + extent;
      } else {
        // Otherwise, invalidate the global producer head
        async_states[stage_id].producer_head = std::nullopt;
      }
    }

    return SBlockRealize({}, IntImm::Bool(true),
                         MakeSBlock(std::move(new_loop), tensor_data_to_tensor_));
  }

  sym::Analyzer analyzer_;
  ffi::Map<Var, TensorVar> tensor_data_to_tensor_;
  const std::unordered_set<TensorVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>& double_buffers_;
  ffi::Array<TensorVar> pipeline_allocs_;
  For pipeline_loop_;
  PipelineInfo pipeline_info_;
  const std::unordered_map<const VarObj*, FragmentInfo>& fragment_info_;
  int max_stage_ = -1;
  ffi::Map<TensorVar, TensorVar> tensor_remap_;
  ffi::Array<SBlock> ordered_stmts_;
  std::map<int, AsyncStateGlobal> async_states;
  ffi::Map<ffi::String, ffi::Any> preserved_annotations_;
};

/*!
 * \brief Build the dependency graph among a array of blocks.
 * \param[in] blocks The array of blocks.
 * \param[out] dep_src2dst Optional, a map to store dependency edges from the source to the
 * destination.
 * \param[out] dep_dst2src Optional, a map to store dependency edges from the
 * destination to the source.
 */
void BuildDependencyGraph(const ffi::Array<SBlock>& blocks,
                          std::unordered_map<SBlock, ffi::Array<SBlock>, ffi::ObjectPtrHash,
                                             ffi::ObjectPtrEqual>* dep_src2dst,
                          std::unordered_map<SBlock, ffi::Array<SBlock>, ffi::ObjectPtrHash,
                                             ffi::ObjectPtrEqual>* dep_dst2src) {
  std::unordered_map<Var, ffi::Array<SBlock>> tensor_writers;

  for (const SBlock& block : blocks) {
    for (const TensorRegion& read : block->reads) {
      auto it = tensor_writers.find(read->source.as_or_throw<tvm::tirx::TensorVar>());
      if (it != tensor_writers.end()) {
        for (const SBlock& writer : it->second) {
          if (dep_src2dst != nullptr) {
            (*dep_src2dst)[writer].push_back(block);
          }
          if (dep_dst2src != nullptr) {
            (*dep_dst2src)[block].push_back(writer);
          }
        }
      }
    }
    for (const TensorRegion& write : block->writes) {
      tensor_writers[write->source.as_or_throw<tvm::tirx::TensorVar>()].push_back(block);
    }
  }
}

class PipelineInjector : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  static Stmt Inject(const Function& func) {
    auto global_symbol = func->GetAttr(tvm::attr::kGlobalSymbol);
    auto injector = ffi::make_object<PipelineInjector>(global_symbol);
    for (const Var& param : func->params) {
      if (auto tensor = param.as<TensorVar>()) {
        injector->tensor_data_to_tensor_.Set(tensor.value(), tensor.value());
      }
    }
    injector->fragment_info_ = GetTensorCoreFragmentInfo(func->body.value());
    return injector->Mutate(func->body.value()).ValueOrUnchanged(func->body.value());
  }

  explicit PipelineInjector(ffi::Optional<ffi::String> global_symbol)
      : global_symbol_(global_symbol) {}

 private:
  /*!
   * \brief Check the pipeline satisfies the following conditions:
   * 1. No conflicting order: The order of each statement should be unique.
   * 2. Reordering of statements doesn't break tensor access dependencies. Specifically, for
   * dependency (e.g. read-after-write) from statement A to statement B, it requires:
   *   case 1: stage(A) < stage(B)
   *   case 2: stage(A) == stage(B) and order(A) < order(B)
   */
  void ValidatePipelineBody(const PipelineInfo& pipeline_info,
                            const ffi::Array<SBlock>& original_order) {
    std::unordered_set<int> used_orders;
    std::unordered_map<int, int> stage_max_order;
    std::unordered_map<int, const SBlock*> order_to_block;
    std::unordered_map<const SBlock*, int> block_to_stage;
    for (const SBlock& block : original_order) {
      const auto& stmt_info = pipeline_info.at(block);
      int order = stmt_info.order;
      TVM_FFI_CHECK(!used_orders.count(order), ValueError)
          << "Two statements in the software pipeline cannot have the same order";
      used_orders.insert(order);
    }

    std::unordered_map<SBlock, ffi::Array<SBlock>, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
        dep_src2dst;
    BuildDependencyGraph(original_order, &dep_src2dst, nullptr);

    for (const auto& pair : dep_src2dst) {
      const SBlock& src = pair.first;
      const auto& src_info = pipeline_info.at(src);
      const ffi::Array<SBlock>& dsts = pair.second;
      for (const SBlock& dst : dsts) {
        const auto& dst_info = pipeline_info.at(dst);
        TVM_FFI_CHECK_LE(src_info.stage, dst_info.stage, ValueError)
            << "statement " << dst << " in stage " << dst_info.stage
            << " cannot depends on statement " << src << " in a later stage " << src_info.stage;
        if (src_info.stage == dst_info.stage) {
          TVM_FFI_CHECK_LT(src_info.order, dst_info.order, ValueError)
              << "two statements with tensor "
                 "access dependency in the same stage of the "
                 "software pipeline cannot be reordered";
        }
      }
    }
  }

  UnchangedOr<Stmt> Mutate_(const ForObj* op, InplaceMode inplace_mode) final {
    // Step 1: Recursively rewrite the children first.
    For for_node = StmtExprMutator::Mutate_(op, inplace_mode)
                       .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                       .as_or_throw<For>();
    if (!HasPipelineAnnotation(op)) {
      return for_node;
    }
    // Step 2: Find the body and tensor allocations of the pipeline. The body can be direct child of
    // the for-loop. If the for-loop has BlockRealize as its child, the pipeline body will be the
    // child of the block.
    SeqStmt pipeline_body = for_node->body;
    ffi::Array<TensorVar> pipeline_allocs;
    if (const auto* realize =
            for_node->body->size() == 1 ? for_node->body->seq[0].as<SBlockRealizeObj>() : nullptr) {
      const auto& block = realize->block;
      for (const auto& tensor : block->alloc_tensors) {
        tensor_data_to_tensor_.Set(tensor, tensor);
      }
      pipeline_body = block->body;
      pipeline_allocs = block->alloc_tensors;
    }

    const SeqStmtObj* pipeline_body_seq = pipeline_body.get();
    TVM_FFI_CHECK_GT(pipeline_body_seq->size(), 1, ValueError)
        << "The body of the software pipeline should contain multiple statements";

    // Step 3: Blockize the components of the pipeline. Each child of the pipelined loop will be
    // converted into a block.
    PipelineInfo pipeline_info;
    ffi::Array<SBlock> original_order;  // pipeline body blocks in the original order

    auto f_add_child = [&](const Stmt& child) {
      original_order.push_back(MakeSBlock(child, tensor_data_to_tensor_));
    };
    for (size_t i = 0; i < pipeline_body_seq->seq.size(); i++) {
      const auto* nested_block_realize = pipeline_body_seq->seq[i].as<SBlockRealizeObj>();
      if (nested_block_realize && IsOne(nested_block_realize->predicate) &&
          nested_block_realize->block->body->size() > 1) {
        const SBlock& nested_pipeline_block = nested_block_realize->block;
        TVM_FFI_ICHECK(
            nested_pipeline_block->match_tensors.empty());  // match_tensor should have been lowered
        for (const auto& tensor : nested_pipeline_block->alloc_tensors) {
          pipeline_allocs.push_back(tensor);
          tensor_data_to_tensor_.Set(tensor, tensor);
        }
        const auto* nested_seq = nested_pipeline_block->body.get();
        for (size_t j = 0; j < nested_seq->seq.size(); j++) {
          f_add_child(nested_seq->seq[j]);
        }
      } else {
        f_add_child(pipeline_body_seq->seq[i]);
      }
    }

    auto pipeline_stages = op->annotations.at(tvm::s_tir::attr::kSoftwarePipelineStage.name)
                               .as_or_throw<ffi::Array<int64_t>>();
    auto pipeline_orders = op->annotations.at(tvm::s_tir::attr::kSoftwarePipelineOrder.name)
                               .as_or_throw<ffi::Array<int64_t>>();
    TVM_FFI_ICHECK_EQ(pipeline_stages.size(), original_order.size())
        << "Function " << global_symbol_ << " has original order "
        << original_order.Map([](const auto& block) { return block->name_hint; })
        << ", but pipeline annotation is " << pipeline_stages << " with different size";
    TVM_FFI_ICHECK_EQ(pipeline_orders.size(), original_order.size())
        << "Function " << global_symbol_ << " has original order "
        << original_order.Map([](const auto& block) { return block->name_hint; })
        << ", but pipeline annotation is " << pipeline_orders << " with different size";

    std::unordered_set<int> pipeline_async_stages;
    if (auto annot = op->annotations.Get(tvm::s_tir::attr::kSoftwarePipelineAsyncStages.name)) {
      for (int64_t s : annot.value().as_or_throw<ffi::Array<int64_t>>()) {
        pipeline_async_stages.insert(static_cast<int>(s));
      }
    }

    ffi::Map<ffi::String, ffi::Any> preserved_annotations;
    for (const auto& kv : op->annotations) {
      const ffi::String& key = kv.first;
      if (kv.first != tvm::s_tir::attr::kSoftwarePipelineStage.name &&
          kv.first != tvm::s_tir::attr::kSoftwarePipelineOrder.name &&
          kv.first != tvm::s_tir::attr::kSoftwarePipelineAsyncStages.name) {
        preserved_annotations.Set(key, kv.second);
      }
    }

    for (size_t i = 0; i < pipeline_stages.size(); i++) {
      int stage = static_cast<int>(pipeline_stages[i]);
      bool is_async = pipeline_async_stages.find(stage) != pipeline_async_stages.end();
      PipelineAnnotation stage_order{stage,
                                     /*order=*/static_cast<int>(pipeline_orders[i]), is_async};
      pipeline_info.emplace(original_order[i], stage_order);
    }

    ValidatePipelineBody(pipeline_info, original_order);

    // Step 4: Rewrite the pipeline body.
    Stmt pipeline = PipelineRewriter::Rewrite(tensor_data_to_tensor_, double_buffers,
                                              pipeline_allocs, ffi::GetRef<For>(op), pipeline_info,
                                              fragment_info_, preserved_annotations);

    if (const auto* realize =
            op->body->size() == 1 ? op->body->seq[0].as<SBlockRealizeObj>() : nullptr) {
      const auto& block = realize->block;
      for (const auto& tensor : block->alloc_tensors) {
        tensor_data_to_tensor_.erase(tensor);
      }
    }
    return pipeline;
  }

  /*!
   * \brief Add tensor allocations to a block and update the write region of the block.
   * \param n The block pointer to which the tensor allocations are added.
   * \param alloc_tensors The tensor allocations to be added.
   */
  void AddAllocTensors(SBlockObj* n, const ffi::Array<TensorVar> alloc_tensors) {
    for (const TensorVar& alloc_tensor : alloc_tensors) {
      n->alloc_tensors.push_back(alloc_tensor);
      ffi::Array<Range> region;
      region.reserve(alloc_tensor.shape().size());
      for (const PrimExpr& dim : alloc_tensor.shape()) {
        region.push_back(Range::FromMinExtent(0, dim));
      }
      n->writes.push_back(TensorRegion(alloc_tensor, region, TensorRegionType()));
    }
  }

  UnchangedOr<Stmt> Mutate_(const SBlockObj* op, InplaceMode inplace_mode) final {
    for (const auto& tensor : op->alloc_tensors) {
      tensor_data_to_tensor_.Set(tensor, tensor);
    }

    auto it = op->annotations.find(tvm::s_tir::attr::kDoubleBufferScope.name);
    if (it != op->annotations.end()) {
      int tensor_index = (*it).second.cast<IntImm>()->value.as<int>().value();
      TVM_FFI_CHECK(tensor_index >= 0 && static_cast<size_t>(tensor_index) < op->writes.size(),
                    ValueError)
          << "Index of the tensor exceeds the size of the write regions of the block. ("
          << tensor_index << " vs. " << op->writes.size() << ")";
      double_buffers.insert(op->writes[tensor_index]->source.as_or_throw<tvm::tirx::TensorVar>());
    }
    SBlock block = StmtExprMutator::Mutate_(op, inplace_mode)
                       .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                       .as_or_throw<SBlock>();

    for (const auto& tensor : op->alloc_tensors) {
      tensor_data_to_tensor_.erase(tensor);
    }
    return block;
  }

  bool HasPipelineAnnotation(const ForObj* op) const {
    auto it1 = op->annotations.find(tvm::s_tir::attr::kSoftwarePipelineStage.name);
    auto it2 = op->annotations.find(tvm::s_tir::attr::kSoftwarePipelineOrder.name);
    bool has_stage = it1 != op->annotations.end();
    bool has_order = it2 != op->annotations.end();
    if (has_stage && has_order) {
      return true;
    }
    if (has_stage) {
      TVM_FFI_THROW(ValueError) << "Order of the software pipeline is not defined.";
    }
    if (has_order) {
      TVM_FFI_THROW(ValueError) << "Stage of the software pipeline is not defined.";
    }
    return false;
  }

  ffi::Map<Var, TensorVar> tensor_data_to_tensor_;
  std::unordered_map<const VarObj*, FragmentInfo> fragment_info_;
  std::unordered_set<TensorVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> double_buffers;
  ffi::Optional<ffi::String> global_symbol_;
};

}  // namespace software_pipeline

namespace transform {

/*!
 * \brief Transform annotated loops into pipelined one that parallelize producers and consumers.
 * \return The IR transform pass.
 */
Pass InjectSoftwarePipeline() {
  auto pass_func = [=](Function f, IRModule m, PassContext ctx) {
    if (!f->body.has_value()) return f;
    auto* fptr = f.CopyOnWrite();
    fptr->body = software_pipeline::PipelineInjector::Inject(f);
    fptr->body = s_tir::ConvertSSA(std::move(fptr->body).value());
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "s_tir.InjectSoftwarePipeline");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.InjectSoftwarePipeline", InjectSoftwarePipeline);
}

}  // namespace transform

}  // namespace s_tir
}  // namespace tvm
