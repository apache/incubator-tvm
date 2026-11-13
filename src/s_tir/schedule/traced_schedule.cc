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
#include "./traced_schedule.h"

#include <tvm/s_tir/stmt.h>

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

Schedule Schedule::Traced(IRModule mod, LinearCongruentialEngine::TRandState seed, int debug_mask,
                          ScheduleErrorRenderLevel error_render_level, bool enable_check) {
  ffi::ObjectPtr<TracedScheduleObj> n =
      ffi::make_object<TracedScheduleObj>(ScheduleState(mod, debug_mask, enable_check));
  n->error_render_level_ = error_render_level;
  n->symbol_table_ = {};
  n->analyzer_ = sym::Analyzer();
  n->trace_ = Trace();
  n->Seed(seed);
  GlobalVar gv{ffi::UnsafeInit{}};
  if (FindEntryFunc(mod, &gv) != nullptr) {
    n->func_working_on_ = gv;
  } else {
    n->func_working_on_ = std::nullopt;
  }
  return Schedule(ffi::UnsafeInit{}, std::move(n));
}

Schedule TracedScheduleObj::Copy() {
  ffi::ObjectPtr<TracedScheduleObj> n = ffi::make_object<TracedScheduleObj>(state_);
  n->error_render_level_ = this->error_render_level_;
  ConcreteScheduleObj::Copy(&n->state_, &n->symbol_table_);
  n->func_working_on_ = this->func_working_on_;
  n->analyzer_ = sym::Analyzer();  // new analyzer needed because it is stateful
  n->rand_state_ = ForkSeed();
  n->trace_ = Trace(this->trace_->insts, this->trace_->decisions);
  return Schedule(ffi::UnsafeInit{}, std::move(n));
}

/******** Schedule: Sampling ********/

ExprRV TracedScheduleObj::SampleCategorical(const ffi::Array<int64_t>& candidates,
                                            const ffi::Array<FloatImm>& probs,
                                            ffi::Optional<int64_t> decision) {
  ExprRV result =
      CreateRV(::tvm::s_tir::SampleCategorical(&this->rand_state_, candidates, probs, &decision));
  static const InstructionKind& kind = InstructionKind::Get("SampleCategorical");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,  //
                                      /*inputs=*/{},
                                      /*attrs=*/{candidates, probs},
                                      /*outputs=*/{result}),
                 /*decision=*/decision);
  return result;
}

ffi::Array<ffi::Optional<ExprRV>> TracedScheduleObj::SamplePerfectTile(
    const LoopRV& loop_rv, int n, int max_innermost_factor,
    ffi::Optional<ffi::Array<int64_t>> decision) {
  // use None RV object to denotes auto-infer tile factors.
  ffi::Array<ffi::Optional<ExprRV>> results = CreateOptionalRV(::tvm::s_tir::SamplePerfectTile(
      &this->rand_state_, this->GetSRef(loop_rv), n, max_innermost_factor, &decision));
  static const InstructionKind& kind = InstructionKind::Get("SamplePerfectTile");
  trace_->Append(
      /*inst=*/Instruction(
          /*kind=*/kind,  //
          /*inputs=*/{loop_rv},
          /*attrs=*/{n, max_innermost_factor},
          /*outputs=*/results),
      /*decision=*/decision);
  return results;
}

ffi::Array<ExprRV> TracedScheduleObj::SamplePartitionedTile(
    const LoopRV& loop_rv, int n, int partition_pos, int innerpart_factor,
    ffi::Optional<ffi::Array<int64_t>> decision) {
  ffi::Array<ExprRV> results = CreateRV(::tvm::s_tir::SamplePartitionedTile(
      &this->rand_state_, this->GetSRef(loop_rv), n, partition_pos, innerpart_factor, &decision));

  static const InstructionKind& kind = InstructionKind::Get("SamplePartitionedTile");
  trace_->Append(
      /*inst=*/Instruction(
          /*kind=*/kind,  //
          /*inputs=*/{loop_rv},
          /*attrs=*/
          {n, partition_pos, innerpart_factor},
          /*outputs=*/results),
      /*decision=*/decision);
  return results;
}

LoopRV TracedScheduleObj::SampleComputeLocation(const SBlockRV& block_rv,
                                                ffi::Optional<int64_t> decision) {
  LoopRV result = CreateRV<LoopRV>(::tvm::s_tir::SampleComputeLocation(
      this->state_, &this->rand_state_, this->GetSRef(block_rv), &decision));

  static const InstructionKind& kind = InstructionKind::Get("SampleComputeLocation");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,  //
                                      /*inputs=*/{block_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{result}),
                 /*decision=*/decision);
  return result;
}

/******** Schedule: Get blocks & loops ********/

SBlockRV TracedScheduleObj::GetSBlock(const ffi::String& name,
                                      const ffi::Optional<ffi::String>& func_name) {
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
  SBlockRV result = ConcreteScheduleObj::GetSBlock(name, func_name);

  static const InstructionKind& kind = InstructionKind::Get("GetSBlock");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,  //
                                      /*inputs=*/{},
                                      /*attrs=*/{name, gv->name_hint},
                                      /*outputs=*/{result}));
  return result;
}

ffi::Array<LoopRV> TracedScheduleObj::GetLoops(const SBlockRV& block_rv) {
  ffi::Array<LoopRV> results = ConcreteScheduleObj::GetLoops(block_rv);

  static const InstructionKind& kind = InstructionKind::Get("GetLoops");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,  //
                                      /*inputs=*/{block_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/results));
  return results;
}

ffi::Array<SBlockRV> TracedScheduleObj::GetChildBlocks(const SBlockRV& block_rv) {
  ffi::Array<SBlockRV> results = ConcreteScheduleObj::GetChildBlocks(block_rv);

  static const InstructionKind& kind = InstructionKind::Get("GetChildBlocks");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,  //
                                      /*inputs=*/{block_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/results));
  return results;
}

ffi::Array<SBlockRV> TracedScheduleObj::GetChildBlocks(const LoopRV& loop_rv) {
  ffi::Array<SBlockRV> results = ConcreteScheduleObj::GetChildBlocks(loop_rv);

  static const InstructionKind& kind = InstructionKind::Get("GetChildBlocks");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,  //
                                      /*inputs=*/{loop_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/results));
  return results;
}

ffi::Array<SBlockRV> TracedScheduleObj::GetProducers(const SBlockRV& block_rv) {
  ffi::Array<SBlockRV> results = ConcreteScheduleObj::GetProducers(block_rv);

  static const InstructionKind& kind = InstructionKind::Get("GetProducers");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,  //
                                      /*inputs=*/{block_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/results));
  return results;
}

ffi::Array<SBlockRV> TracedScheduleObj::GetConsumers(const SBlockRV& block_rv) {
  ffi::Array<SBlockRV> results = ConcreteScheduleObj::GetConsumers(block_rv);

  static const InstructionKind& kind = InstructionKind::Get("GetConsumers");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,  //
                                      /*inputs=*/{block_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/results));
  return results;
}

ffi::Array<SBlockRV> TracedScheduleObj::GetOutputBlocks(const SBlockRV& scope_block_rv) {
  ffi::Array<SBlockRV> results = ConcreteScheduleObj::GetOutputBlocks(scope_block_rv);

  static const InstructionKind& kind = InstructionKind::Get("GetOutputBlocks");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,  //
                                      /*inputs=*/{scope_block_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/results));
  return results;
}

/******** Schedule: Transform loops ********/

LoopRV TracedScheduleObj::Merge(const ffi::Array<LoopRV>& loop_rvs) {
  LoopRV result = ConcreteScheduleObj::Merge(loop_rvs);
  static const InstructionKind& kind = InstructionKind::Get("Merge");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/loop_rvs,
                                      /*attrs=*/{},
                                      /*outputs=*/{result}));
  return result;
}

LoopRV TracedScheduleObj::Fuse(const ffi::Array<LoopRV>& loop_rvs, bool preserve_unit_loops) {
  LoopRV result = ConcreteScheduleObj::Fuse(loop_rvs, preserve_unit_loops);

  static const InstructionKind& kind = InstructionKind::Get("Fuse");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/loop_rvs,
                                      /*attrs=*/{preserve_unit_loops},
                                      /*outputs=*/{result}));
  return result;
}

ffi::Array<LoopRV> TracedScheduleObj::Split(const LoopRV& loop_rv,
                                            const ffi::Array<ffi::Optional<ExprRV>>& factor_rvs,
                                            bool preserve_unit_iters, bool disable_predication) {
  ffi::Array<LoopRV> results =
      ConcreteScheduleObj::Split(loop_rv, factor_rvs, preserve_unit_iters, disable_predication);

  ffi::Array<Any> inputs;
  inputs.reserve(1 + factor_rvs.size());
  inputs.push_back(loop_rv);
  for (const auto& obj : factor_rvs) {
    inputs.push_back(obj);
  }

  static const InstructionKind& kind = InstructionKind::Get("Split");
  trace_->Append(
      /*inst=*/Instruction(/*kind=*/kind,
                           /*inputs=*/inputs,
                           /*attrs=*/ffi::Array<Any>({preserve_unit_iters, disable_predication}),
                           /*outputs=*/results));
  return results;
}

ffi::Array<LoopRV> TracedScheduleObj::LoopPartition(
    const LoopRV& loop_rv, const ffi::Array<ffi::Optional<ExprRV>>& factor_rvs,
    bool preserve_unit_iters) {
  ffi::Array<LoopRV> results =
      ConcreteScheduleObj::LoopPartition(loop_rv, factor_rvs, preserve_unit_iters);

  ffi::Array<Any> inputs;
  inputs.reserve(1 + factor_rvs.size());
  inputs.push_back(loop_rv);
  for (const auto& obj : factor_rvs) {
    inputs.push_back(obj);
  }

  static const InstructionKind& kind = InstructionKind::Get("LoopPartition");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/inputs,
                                      /*attrs=*/{preserve_unit_iters},
                                      /*outputs=*/results));
  return results;
}

void TracedScheduleObj::Reorder(const ffi::Array<LoopRV>& ordered_loop_rvs) {
  ConcreteScheduleObj::Reorder(ordered_loop_rvs);

  static const InstructionKind& kind = InstructionKind::Get("Reorder");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/ordered_loop_rvs,
                                      /*attrs=*/{},
                                      /*outputs=*/{}));
}

void TracedScheduleObj::ReorderBlockIterVar(const SBlockRV& block_rv,
                                            const ffi::Array<int64_t> new_order) {
  ConcreteScheduleObj::ReorderBlockIterVar(block_rv, new_order);
  static const InstructionKind& kind = InstructionKind::Get("ReorderBlockIterVar");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{block_rv, new_order}, /*attrs=*/{},
                                      /*outputs=*/{}));
}

LoopRV TracedScheduleObj::AddUnitLoop(const SBlockRV& block_rv) {
  LoopRV result = ConcreteScheduleObj::AddUnitLoop(block_rv);

  static const InstructionKind& kind = InstructionKind::Get("AddUnitLoop");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{block_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{result}));
  return result;
}

LoopRV TracedScheduleObj::AddUnitLoop(const LoopRV& loop_rv) {
  LoopRV result = ConcreteScheduleObj::AddUnitLoop(loop_rv);

  static const InstructionKind& kind = InstructionKind::Get("AddUnitLoop");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{loop_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{result}));
  return result;
}

/******** Schedule: Manipulate ForKind ********/

void TracedScheduleObj::Parallel(const LoopRV& loop_rv) {
  ConcreteScheduleObj::Parallel(loop_rv);

  static const InstructionKind& kind = InstructionKind::Get("Parallel");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{loop_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{}));
}

void TracedScheduleObj::Vectorize(const LoopRV& loop_rv) {
  ConcreteScheduleObj::Vectorize(loop_rv);

  static const InstructionKind& kind = InstructionKind::Get("Vectorize");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{loop_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{}));
}

void TracedScheduleObj::Bind(const LoopRV& loop_rv, const ffi::String& thread_axis) {
  ConcreteScheduleObj::Bind(loop_rv, thread_axis);

  static const InstructionKind& kind = InstructionKind::Get("Bind");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{loop_rv},
                                      /*attrs=*/{thread_axis},
                                      /*outputs=*/{}));
}

void TracedScheduleObj::Unroll(const LoopRV& loop_rv) {
  ConcreteScheduleObj::Unroll(loop_rv);

  static const InstructionKind& kind = InstructionKind::Get("Unroll");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{loop_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{}));
}

/******** Schedule: Insert cache stages ********/
SBlockRV TracedScheduleObj::CacheRead(const SBlockRV& block_rv, int read_tensor_index,
                                      const ffi::String& storage_scope,
                                      const ffi::Array<SBlockRV> consumer_blocks) {
  SBlockRV result =
      ConcreteScheduleObj::CacheRead(block_rv, read_tensor_index, storage_scope, consumer_blocks);

  static const InstructionKind& kind = InstructionKind::Get("CacheRead");
  trace_->Append(
      /*inst=*/Instruction(/*kind=*/kind,
                           /*inputs=*/{block_rv, consumer_blocks},
                           /*attrs=*/{read_tensor_index, storage_scope},
                           /*outputs=*/{result}));
  return result;
}

SBlockRV TracedScheduleObj::CacheWrite(const SBlockRV& block_rv, int write_tensor_index,
                                       const ffi::String& storage_scope,
                                       const ffi::Array<SBlockRV> consumer_blocks) {
  SBlockRV result =
      ConcreteScheduleObj::CacheWrite(block_rv, write_tensor_index, storage_scope, consumer_blocks);

  static const InstructionKind& kind = InstructionKind::Get("CacheWrite");
  trace_->Append(
      /*inst=*/Instruction(/*kind=*/kind,
                           /*inputs=*/{block_rv, consumer_blocks},
                           /*attrs=*/{write_tensor_index, storage_scope},
                           /*outputs=*/{result}));
  return result;
}

SBlockRV TracedScheduleObj::ReindexCacheRead(const SBlockRV& block_rv, int read_tensor_index,
                                             const ffi::String& storage_scope,
                                             const IndexMap& index_map) {
  SBlockRV result =
      ConcreteScheduleObj::ReindexCacheRead(block_rv, read_tensor_index, storage_scope, index_map);

  static const InstructionKind& kind = InstructionKind::Get("ReindexCacheRead");
  trace_->Append(
      /*inst=*/Instruction(
          /*kind=*/kind,
          /*inputs=*/{block_rv, index_map},
          /*attrs=*/{read_tensor_index, storage_scope},
          /*outputs=*/{result}));
  return result;
}

SBlockRV TracedScheduleObj::ReindexCacheWrite(const SBlockRV& block_rv, int write_tensor_index,
                                              const ffi::String& storage_scope,
                                              const IndexMap& index_map) {
  SBlockRV result = ConcreteScheduleObj::ReindexCacheWrite(block_rv, write_tensor_index,
                                                           storage_scope, index_map);

  static const InstructionKind& kind = InstructionKind::Get("ReindexCacheWrite");
  trace_->Append(
      /*inst=*/Instruction(
          /*kind=*/kind,
          /*inputs=*/{block_rv, index_map},
          /*attrs=*/{write_tensor_index, storage_scope},
          /*outputs=*/{result}));
  return result;
}

ffi::Array<SBlockRV> TracedScheduleObj::CacheInplace(const SBlockRV& block_rv,
                                                     int read_tensor_index,
                                                     const ffi::String& storage_scope) {
  ffi::Array<SBlockRV> result =
      ConcreteScheduleObj::CacheInplace(block_rv, read_tensor_index, storage_scope);
  ffi::Array<Any> results;
  for (const SBlockRV& r : result) {
    results.push_back(r);
  }
  static const InstructionKind& kind = InstructionKind::Get("CacheInplace");
  trace_->Append(
      /*inst=*/Instruction(/*kind=*/kind,
                           /*inputs=*/{block_rv},
                           /*attrs=*/{read_tensor_index, storage_scope},
                           /*outputs=*/results));
  return result;
}

ffi::Array<SBlockRV> TracedScheduleObj::CacheIndex(const SBlockRV& block_rv,
                                                   const ffi::String& storage_scope,
                                                   int cse_thresh) {
  ffi::Array<SBlockRV> result =
      ConcreteScheduleObj::CacheIndex(block_rv, storage_scope, cse_thresh);
  ffi::Array<Any> outputs;
  for (const SBlockRV& r : result) {
    outputs.push_back(r);
  }
  static const InstructionKind& kind = InstructionKind::Get("CacheIndex");
  trace_->Append(
      /*inst=*/Instruction(/*kind=*/kind,
                           /*inputs=*/{block_rv},
                           /*attrs=*/{storage_scope, cse_thresh},
                           /*outputs=*/outputs));
  return result;
}

SBlockRV TracedScheduleObj::ReIndex(const SBlockRV& block_rv, int tensor_index,
                                    TensorIndexType tensor_index_type) {
  SBlockRV result = ConcreteScheduleObj::ReIndex(block_rv, tensor_index, tensor_index_type);

  static const InstructionKind& kind = InstructionKind::Get("ReIndex");
  trace_->Append(
      /*inst=*/Instruction(
          /*kind=*/kind,
          /*inputs=*/{block_rv},
          /*attrs=*/
          {tensor_index, static_cast<int>(tensor_index_type)},
          /*outputs=*/{result}));
  return result;
}

/******** Schedule: Data movement ********/

SBlockRV TracedScheduleObj::ReadAt(const LoopRV& loop_rv, const SBlockRV& block_rv,
                                   int read_tensor_index, const ffi::String& storage_scope) {
  SBlockRV result =
      ConcreteScheduleObj::ReadAt(loop_rv, block_rv, read_tensor_index, storage_scope);

  static const InstructionKind& kind = InstructionKind::Get("ReadAt");
  trace_->Append(
      /*inst=*/Instruction(/*kind=*/kind,
                           /*inputs=*/{loop_rv, block_rv},
                           /*attrs=*/{read_tensor_index, storage_scope},
                           /*outputs=*/{result}));
  return result;
}

SBlockRV TracedScheduleObj::WriteAt(const LoopRV& loop_rv, const SBlockRV& block_rv,
                                    int write_tensor_index, const ffi::String& storage_scope) {
  SBlockRV result =
      ConcreteScheduleObj::WriteAt(loop_rv, block_rv, write_tensor_index, storage_scope);

  static const InstructionKind& kind = InstructionKind::Get("WriteAt");
  trace_->Append(
      /*inst=*/Instruction(/*kind=*/kind,
                           /*inputs=*/{loop_rv, block_rv},
                           /*attrs=*/{write_tensor_index, storage_scope},
                           /*outputs=*/{result}));
  return result;
}

/******** Schedule: Compute location ********/

void TracedScheduleObj::ComputeAt(const SBlockRV& block_rv, const LoopRV& loop_rv,
                                  bool preserve_unit_loops, int index) {
  ConcreteScheduleObj::ComputeAt(block_rv, loop_rv, preserve_unit_loops, index);

  static const InstructionKind& kind = InstructionKind::Get("ComputeAt");
  trace_->Append(
      /*inst=*/Instruction(
          /*kind=*/kind,
          /*inputs=*/{block_rv, loop_rv},
          /*attrs=*/
          {preserve_unit_loops, index},
          /*outputs=*/{}));
}

void TracedScheduleObj::ReverseComputeAt(const SBlockRV& block_rv, const LoopRV& loop_rv,
                                         bool preserve_unit_loops, int index) {
  ConcreteScheduleObj::ReverseComputeAt(block_rv, loop_rv, preserve_unit_loops, index);

  static const InstructionKind& kind = InstructionKind::Get("ReverseComputeAt");
  trace_->Append(/*inst=*/Instruction(
      /*kind=*/kind,
      /*inputs=*/{block_rv, loop_rv},
      /*attrs=*/{preserve_unit_loops, index},
      /*outputs=*/{}));
}

void TracedScheduleObj::ComputeInline(const SBlockRV& block_rv) {
  ConcreteScheduleObj::ComputeInline(block_rv);

  static const InstructionKind& kind = InstructionKind::Get("ComputeInline");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{block_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{}));
}

void TracedScheduleObj::ReverseComputeInline(const SBlockRV& block_rv) {
  ConcreteScheduleObj::ReverseComputeInline(block_rv);

  static const InstructionKind& kind = InstructionKind::Get("ReverseComputeInline");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{block_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{}));
}

void TracedScheduleObj::FuseReductionEpilogue(const SBlockRV& reduction_block_rv,
                                              const SBlockRV& epilogue_block_rv) {
  ConcreteScheduleObj::FuseReductionEpilogue(reduction_block_rv, epilogue_block_rv);

  static const InstructionKind& kind = InstructionKind::Get("FuseReductionEpilogue");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{reduction_block_rv, epilogue_block_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{}));
}

/******** Schedule: Reduction ********/

SBlockRV TracedScheduleObj::DecomposeReduction(const SBlockRV& block_rv, const LoopRV& loop_rv) {
  SBlockRV result = ConcreteScheduleObj::DecomposeReduction(block_rv, loop_rv);
  static const InstructionKind& kind = InstructionKind::Get("DecomposeReduction");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{block_rv, loop_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{result}));
  return result;
}

SBlockRV TracedScheduleObj::RFactor(const LoopRV& loop_rv, int factor_axis) {
  SBlockRV result = ConcreteScheduleObj::RFactor(loop_rv, factor_axis);
  static const InstructionKind& kind = InstructionKind::Get("RFactor");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{loop_rv},
                                      /*attrs=*/{factor_axis},
                                      /*outputs=*/{result}));
  return result;
}

/******** Schedule: SBlock annotation ********/

void TracedScheduleObj::StorageAlign(const SBlockRV& block_rv, int tensor_index, int axis,
                                     int factor, int offset) {
  ConcreteScheduleObj::StorageAlign(block_rv, tensor_index, axis, factor, offset);
  static const InstructionKind& kind = InstructionKind::Get("StorageAlign");
  trace_->Append(/*inst=*/Instruction(
      /*kind=*/kind,
      /*inputs=*/{block_rv},
      /*attrs=*/
      {tensor_index, axis, factor, offset},
      /*outputs=*/{}));
}

void TracedScheduleObj::SetScope(const SBlockRV& block_rv, int tensor_index,
                                 const ffi::String& storage_scope) {
  ConcreteScheduleObj::SetScope(block_rv, tensor_index, storage_scope);
  static const InstructionKind& kind = InstructionKind::Get("SetScope");
  trace_->Append(/*inst=*/Instruction(
      /*kind=*/kind,
      /*inputs=*/{block_rv},
      /*attrs=*/{tensor_index, storage_scope},
      /*outputs=*/{}));
}

void TracedScheduleObj::UnsafeSetDType(const SBlockRV& block_rv, int tensor_index,
                                       const ffi::String& dtype) {
  ConcreteScheduleObj::UnsafeSetDType(block_rv, tensor_index, dtype);
  static const InstructionKind& kind = InstructionKind::Get("UnsafeSetDType");
  trace_->Append(/*inst=*/Instruction(
      /*kind=*/kind,
      /*inputs=*/{block_rv},
      /*attrs=*/{tensor_index, dtype},
      /*outputs=*/{}));
}

/******** Schedule: Blockize & Tensorize ********/

SBlockRV TracedScheduleObj::Blockize(const LoopRV& loop_rv, bool preserve_unit_iters) {
  SBlockRV new_block = ConcreteScheduleObj::Blockize(loop_rv, preserve_unit_iters);
  static const InstructionKind& kind = InstructionKind::Get("Blockize");
  trace_->Append(/*inst=*/Instruction(
      /*kind=*/kind,
      /*inputs=*/{loop_rv},
      /*attrs=*/{preserve_unit_iters},
      /*outputs=*/{new_block}));
  return new_block;
}

SBlockRV TracedScheduleObj::Blockize(const ffi::Array<SBlockRV>& blocks, bool preserve_unit_iters) {
  SBlockRV new_block = ConcreteScheduleObj::Blockize(blocks, preserve_unit_iters);
  static const InstructionKind& kind = InstructionKind::Get("Blockize");
  trace_->Append(/*inst=*/Instruction(
      /*kind=*/kind,
      /*inputs=*/{blocks},
      /*attrs=*/{preserve_unit_iters},
      /*outputs=*/{new_block}));
  return new_block;
}

void TracedScheduleObj::Tensorize(const LoopRV& loop_rv, const ffi::String& intrin,
                                  bool preserve_unit_iters) {
  ConcreteScheduleObj::Tensorize(loop_rv, intrin, preserve_unit_iters);
  static const InstructionKind& kind = InstructionKind::Get("Tensorize");
  trace_->Append(/*inst=*/Instruction(
      /*kind=*/kind,
      /*inputs=*/{loop_rv},
      /*attrs=*/{intrin, preserve_unit_iters},
      /*outputs=*/{}));
}

void TracedScheduleObj::Tensorize(const SBlockRV& block_rv, const ffi::String& intrin,
                                  bool preserve_unit_iters) {
  ConcreteScheduleObj::Tensorize(block_rv, intrin, preserve_unit_iters);
  static const InstructionKind& kind = InstructionKind::Get("Tensorize");
  trace_->Append(/*inst=*/Instruction(
      /*kind=*/kind,
      /*inputs=*/{block_rv},
      /*attrs=*/{intrin, preserve_unit_iters},
      /*outputs=*/{}));
}

/******** Schedule: Annotation ********/

void TracedScheduleObj::Annotate(const LoopRV& loop_rv, const ffi::String& ann_key,
                                 const Any& ann_val) {
  ConcreteScheduleObj::Annotate(loop_rv, ann_key, ann_val);
  static const InstructionKind& kind = InstructionKind::Get("Annotate");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{loop_rv, ann_val},
                                      /*attrs=*/{ann_key},
                                      /*outputs=*/{}));
}

void TracedScheduleObj::Annotate(const SBlockRV& block_rv, const ffi::String& ann_key,
                                 const Any& ann_val) {
  ConcreteScheduleObj::Annotate(block_rv, ann_key, ann_val);
  static const InstructionKind& kind = InstructionKind::Get("Annotate");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{block_rv, ann_val},
                                      /*attrs=*/{ann_key},
                                      /*outputs=*/{}));
}

void TracedScheduleObj::Unannotate(const LoopRV& loop_rv, const ffi::String& ann_key) {
  ConcreteScheduleObj::Unannotate(loop_rv, ann_key);
  static const InstructionKind& kind = InstructionKind::Get("Unannotate");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{loop_rv},
                                      /*attrs=*/{ann_key},
                                      /*outputs=*/{}));
}

void TracedScheduleObj::Unannotate(const SBlockRV& block_rv, const ffi::String& ann_key) {
  ConcreteScheduleObj::Unannotate(block_rv, ann_key);
  static const InstructionKind& kind = InstructionKind::Get("Unannotate");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{block_rv},
                                      /*attrs=*/{ann_key},
                                      /*outputs=*/{}));
}

/******** Schedule: Layout transformation ********/

void TracedScheduleObj::TransformLayout(const SBlockRV& block_rv, int tensor_index,
                                        TensorIndexType tensor_index_type,
                                        const IndexMap& index_map,
                                        const ffi::Optional<IndexMap>& pad_value,
                                        bool assume_injective_transform) {
  ConcreteScheduleObj::TransformLayout(block_rv, tensor_index, tensor_index_type, index_map,
                                       pad_value, assume_injective_transform);
  static const InstructionKind& kind = InstructionKind::Get("TransformLayout");
  trace_->Append(
      /*inst=*/Instruction(
          /*kind=*/kind,
          /*inputs=*/{block_rv, index_map},
          /*attrs=*/
          {tensor_index, static_cast<int>(tensor_index_type), pad_value,
           assume_injective_transform},
          /*outputs=*/{}));
}

void TracedScheduleObj::TransformBlockLayout(const SBlockRV& block_rv, const IndexMap& index_map) {
  ConcreteScheduleObj::TransformBlockLayout(block_rv, index_map);
  static const InstructionKind& kind = InstructionKind::Get("TransformBlockLayout");
  trace_->Append(
      /*inst=*/Instruction(/*kind=*/kind,
                           /*inputs=*/{block_rv},
                           /*attrs=*/{index_map},
                           /*outputs=*/{}));
}

/******** Schedule: Padding ********/
SBlockRV TracedScheduleObj::DecomposePadding(const SBlockRV& block_rv, const LoopRV& loop_rv) {
  SBlockRV new_block = ConcreteScheduleObj::DecomposePadding(block_rv, loop_rv);
  static const InstructionKind& kind = InstructionKind::Get("DecomposePadding");
  trace_->Append(/*inst=*/Instruction(
      /*kind=*/kind,
      /*inputs=*/{block_rv, loop_rv},
      /*attrs=*/{},
      /*outputs=*/{new_block}));
  return new_block;
}

void TracedScheduleObj::PadEinsum(const SBlockRV& block_rv, const ffi::Array<int64_t>& padding) {
  ConcreteScheduleObj::PadEinsum(block_rv, padding);
  static const InstructionKind& kind = InstructionKind::Get("PadEinsum");
  trace_->Append(/*inst=*/Instruction(
      /*kind=*/kind,
      /*inputs=*/{block_rv},
      /*attrs=*/{padding},
      /*outputs=*/{}));
}

/******** Schedule: Tensor transformation ********/

void TracedScheduleObj::RollingBuffer(const SBlockRV& block_rv, int write_tensor_index) {
  ConcreteScheduleObj::RollingBuffer(block_rv, write_tensor_index);
  static const InstructionKind& kind = InstructionKind::Get("RollingBuffer");
  trace_->Append(/*inst=*/Instruction(
      /*kind=*/kind,
      /*inputs=*/{block_rv},
      /*attrs=*/{write_tensor_index},
      /*outputs=*/{}));
}

/******** Schedule: Misc ********/

void TracedScheduleObj::EnterPostproc() {
  ConcreteScheduleObj::EnterPostproc();
  static const InstructionKind& kind = InstructionKind::Get("EnterPostproc");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{},
                                      /*attrs=*/{},
                                      /*outputs=*/{}));
}

void TracedScheduleObj::UnsafeHideTensorAccess(const SBlockRV& block_rv,
                                               const ffi::String& tensor_type,
                                               const ffi::Array<int64_t>& tensor_index_array) {
  ConcreteScheduleObj::UnsafeHideTensorAccess(block_rv, tensor_type, tensor_index_array);
  static const InstructionKind& kind = InstructionKind::Get("UnsafeHideTensorAccess");
  trace_->Append(/*inst=*/Instruction(
      /*kind=*/kind,
      /*inputs=*/{block_rv, tensor_type, tensor_index_array},
      /*attrs=*/{},
      /*outputs=*/{}));
}

void TracedScheduleObj::AnnotateTensorAccess(const SBlockRV& block_rv, int tensor_index,
                                             TensorIndexType tensor_index_type,
                                             const IndexMap& index_map) {
  ConcreteScheduleObj::AnnotateTensorAccess(block_rv, tensor_index, tensor_index_type, index_map);
  static const InstructionKind& kind = InstructionKind::Get("AnnotateTensorAccess");
  trace_->Append(/*inst=*/Instruction(
      /*kind=*/kind,
      /*inputs=*/
      {block_rv, tensor_index, static_cast<int>(tensor_index_type), index_map},
      /*attrs=*/{},
      /*outputs=*/{}));
}

}  // namespace s_tir
}  // namespace tvm
