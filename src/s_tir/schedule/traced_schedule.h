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
#ifndef TVM_S_TIR_SCHEDULE_TRACED_SCHEDULE_H_
#define TVM_S_TIR_SCHEDULE_TRACED_SCHEDULE_H_

#include <tvm/ir/prim/expr.h>
#include <tvm/s_tir/stmt.h>

#include "./concrete_schedule.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

class TracedScheduleObj : public ConcreteScheduleObj {
  friend class Schedule;

 protected:
  Trace trace_;

 public:
  using ConcreteScheduleObj::ConcreteScheduleObj;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TracedScheduleObj>();
  }

  ~TracedScheduleObj() = default;

 public:
  ffi::Optional<Trace> trace() const final { return trace_; }
  Schedule Copy() final;

 public:
  /******** Schedule: Sampling ********/
  ExprRV SampleCategorical(const ffi::Array<int64_t>& candidates, const ffi::Array<FloatImm>& probs,
                           ffi::Optional<int64_t> decision = std::nullopt) final;
  ffi::Array<ffi::Optional<ExprRV>> SamplePerfectTile(
      const LoopRV& loop_rv, int n, int max_innermost_factor,
      ffi::Optional<ffi::Array<int64_t>> decision = std::nullopt) final;
  ffi::Array<ExprRV> SamplePartitionedTile(
      const LoopRV& loop_rv, int n, int partition_pos, int innerpart_factor,
      ffi::Optional<ffi::Array<int64_t>> decision = std::nullopt) final;
  LoopRV SampleComputeLocation(const SBlockRV& block_rv,
                               ffi::Optional<int64_t> decision = std::nullopt) final;
  /******** Schedule: Get blocks & loops ********/
  SBlockRV GetSBlock(const ffi::String& name, const ffi::Optional<ffi::String>& func_name) final;
  ffi::Array<LoopRV> GetLoops(const SBlockRV& block_rv) final;
  ffi::Array<SBlockRV> GetChildBlocks(const SBlockRV& block_rv) final;
  ffi::Array<SBlockRV> GetChildBlocks(const LoopRV& loop_rv) final;
  ffi::Array<SBlockRV> GetProducers(const SBlockRV& block_rv) final;
  ffi::Array<SBlockRV> GetConsumers(const SBlockRV& block_rv) final;
  ffi::Array<SBlockRV> GetOutputBlocks(const SBlockRV& scope_block_rv) final;
  /******** Schedule: Transform loops ********/
  LoopRV Fuse(const ffi::Array<LoopRV>& loop_rvs, bool preserve_unit_iters) final;
  LoopRV Merge(const ffi::Array<LoopRV>& loop_rvs) final;
  ffi::Array<LoopRV> Split(const LoopRV& loop_rv,
                           const ffi::Array<ffi::Optional<ExprRV>>& factor_rvs,
                           bool preserve_unit_iters, bool disable_predication) final;
  ffi::Array<LoopRV> LoopPartition(const LoopRV& loop_rv,
                                   const ffi::Array<ffi::Optional<ExprRV>>& factor_rvs,
                                   bool preserve_unit_iters) final;
  void Reorder(const ffi::Array<LoopRV>& ordered_loop_rvs) final;
  void ReorderBlockIterVar(const SBlockRV& block_rv, const ffi::Array<int64_t> new_order) final;
  LoopRV AddUnitLoop(const SBlockRV& block_rv) final;
  LoopRV AddUnitLoop(const LoopRV& loop_rv) final;
  /******** Schedule: Manipulate ForKind ********/
  void Parallel(const LoopRV& loop_rv) final;
  void Vectorize(const LoopRV& loop_rv) final;
  void Bind(const LoopRV& loop_rv, const ffi::String& thread_axis) final;
  void Unroll(const LoopRV& loop_rv) final;
  /******** Schedule: Insert cache stages ********/
  SBlockRV CacheRead(const SBlockRV& block_rv, int read_tensor_index,
                     const ffi::String& storage_scope,
                     const ffi::Array<SBlockRV> consumer_blocks = {}) final;
  SBlockRV CacheWrite(const SBlockRV& block_rv, int write_tensor_index,
                      const ffi::String& storage_scope,
                      const ffi::Array<SBlockRV> consumer_blocks = {}) final;
  SBlockRV ReindexCacheRead(const SBlockRV& block_rv, int read_tensor_index,
                            const ffi::String& storage_scope, const IndexMap& index_map) final;
  SBlockRV ReindexCacheWrite(const SBlockRV& block_rv, int write_tensor_index,
                             const ffi::String& storage_scope, const IndexMap& index_map) final;
  ffi::Array<SBlockRV> CacheInplace(const SBlockRV& block_rv, int read_tensor_index,
                                    const ffi::String& storage_scope) final;
  SBlockRV ReIndex(const SBlockRV& block_rv, int tensor_index,
                   TensorIndexType tensor_index_type) final;
  ffi::Array<SBlockRV> CacheIndex(const SBlockRV& block_rv, const ffi::String& storage_scope,
                                  int cse_thresh) final;
  /******** Schedule: Data movement ********/
  SBlockRV ReadAt(const LoopRV& loop_rv, const SBlockRV& block_rv, int read_tensor_index,
                  const ffi::String& storage_scope) final;
  SBlockRV WriteAt(const LoopRV& loop_rv, const SBlockRV& block_rv, int write_tensor_index,
                   const ffi::String& storage_scope) final;
  /******** Schedule: Compute location ********/
  void ComputeAt(const SBlockRV& block_rv, const LoopRV& loop_rv, bool preserve_unit_loops,
                 int index = -1) final;
  void ReverseComputeAt(const SBlockRV& block_rv, const LoopRV& loop_rv, bool preserve_unit_loops,
                        int index = -1) final;
  void ComputeInline(const SBlockRV& block_rv) final;
  void ReverseComputeInline(const SBlockRV& block_rv) final;
  void FuseReductionEpilogue(const SBlockRV& reduction_block, const SBlockRV& epilogue_block) final;
  /******** Schedule: Reduction ********/
  SBlockRV DecomposeReduction(const SBlockRV& block_rv, const LoopRV& loop_rv) final;
  SBlockRV RFactor(const LoopRV& loop_rv, int factor_axis) final;
  /******** Schedule: SBlock annotation ********/
  void StorageAlign(const SBlockRV& block_rv, int tensor_index, int axis, int factor,
                    int offset) final;
  void SetScope(const SBlockRV& block_rv, int tensor_index, const ffi::String& storage_scope) final;
  void UnsafeSetDType(const SBlockRV& block_rv, int tensor_index, const ffi::String& dtype) final;
  /******** Schedule: Blockize & Tensorize ********/
  SBlockRV Blockize(const LoopRV& loop_rv, bool preserve_unit_iters) final;
  SBlockRV Blockize(const ffi::Array<SBlockRV>& blocks, bool preserve_unit_iters) final;
  void Tensorize(const SBlockRV& block_rv, const ffi::String& intrin,
                 bool preserve_unit_iters) final;
  void Tensorize(const LoopRV& loop_rv, const ffi::String& intrin, bool preserve_unit_iters) final;
  /******** Schedule: Annotation ********/
  void Annotate(const LoopRV& loop_rv, const ffi::String& ann_key, const Any& ann_val) override;
  void Unannotate(const LoopRV& loop_rv, const ffi::String& ann_key) override;
  void Annotate(const SBlockRV& block_rv, const ffi::String& ann_key, const Any& ann_val) override;
  void Unannotate(const SBlockRV& block_rv, const ffi::String& ann_key) override;
  /******** Schedule: Layout transformation ********/
  void TransformLayout(const SBlockRV& block_rv, int tensor_index,
                       TensorIndexType tensor_index_type, const IndexMap& index_map,
                       const ffi::Optional<IndexMap>& pad_value,
                       bool assume_injective_transform) override;
  void TransformBlockLayout(const SBlockRV& block_rv, const IndexMap& index_map) override;
  /******** Schedule: Padding ********/
  SBlockRV DecomposePadding(const SBlockRV& block_rv, const LoopRV& loop_rv) final;
  void PadEinsum(const SBlockRV& block_rv, const ffi::Array<int64_t>& padding) final;
  /******** Schedule: Tensor transformation ********/
  void RollingBuffer(const SBlockRV& block_rv, int write_tensor_index) final;
  /******** Schedule: Misc ********/
  void EnterPostproc() final;
  void UnsafeHideTensorAccess(const SBlockRV& block_rv, const ffi::String& tensor_type,
                              const ffi::Array<int64_t>& tensor_index_array) final;
  void AnnotateTensorAccess(const SBlockRV& block_rv, int tensor_index,
                            TensorIndexType tensor_index_type, const IndexMap& index_map) final;
};

}  // namespace s_tir
}  // namespace tvm

#endif  // TVM_S_TIR_SCHEDULE_TRACED_SCHEDULE_H_
