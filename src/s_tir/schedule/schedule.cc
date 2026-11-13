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
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/stmt.h>

#include "./utils.h"
namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<ScheduleObj>();
  SBlockRVObj::RegisterReflection();
  LoopRVObj::RegisterReflection();
}

/**************** Constructor ****************/

SBlockRV::SBlockRV() { this->data_ = ffi::make_object<SBlockRVObj>(); }

LoopRV::LoopRV() { this->data_ = ffi::make_object<LoopRVObj>(); }

/**************** GetSRef ****************/

StmtSRef ScheduleObj::GetSRef(const StmtObj* stmt) const {
  ScheduleState state = this->state();
  auto it = state->stmt2ref.find(stmt);
  if (it == state->stmt2ref.end()) {
    TVM_FFI_THROW(IndexError) << "The stmt doesn't exist in the IR";
  }
  return it->second;
}

/**************** FFI ****************/

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("s_tir.schedule.ScheduleGetMod", &ScheduleObj::mod)
      .def_method("s_tir.schedule.ScheduleGetState", &ScheduleObj::state)
      .def_method("s_tir.schedule.ScheduleGetTrace", &ScheduleObj::trace)
      .def_method("s_tir.schedule.ScheduleGetFuncWorkingOn", &ScheduleObj::func_working_on)
      .def_method("s_tir.schedule.ScheduleCopy", &ScheduleObj::Copy)
      .def_method("s_tir.schedule.ScheduleSeed", &ScheduleObj::Seed)
      .def_method("s_tir.schedule.ScheduleForkSeed", &ScheduleObj::ForkSeed)
      .def_method("s_tir.schedule.ScheduleWorkOn", &ScheduleObj::WorkOn);
}

/**************** (FFI) Constructor ****************/

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("s_tir.schedule.SBlockRV", []() { return SBlockRV(); })
      .def("s_tir.schedule.LoopRV", []() { return LoopRV(); })
      .def("s_tir.schedule.ConcreteSchedule",
           [](IRModule mod, LinearCongruentialEngine::TRandState seed, int debug_mask,
              int error_render_level, bool enable_check) -> Schedule {
             return Schedule::Concrete(mod, debug_mask, seed,
                                       static_cast<ScheduleErrorRenderLevel>(error_render_level),
                                       enable_check);
           })
      .def("s_tir.schedule.TracedSchedule",
           [](IRModule mod, LinearCongruentialEngine::TRandState seed, int debug_mask,
              int error_render_level, bool enable_check) -> Schedule {
             return Schedule::Traced(mod, seed, debug_mask,
                                     static_cast<ScheduleErrorRenderLevel>(error_render_level),
                                     enable_check);
           });
}

/******** (FFI) Lookup random variables ********/

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("s_tir.schedule.ScheduleGet",
           [](Schedule self, ffi::ObjectRef obj) -> ffi::ObjectRef {
             if (auto loop_rv = obj.as<LoopRV>()) {
               return self->Get(loop_rv.value());
             }
             if (auto block_rv = obj.as<SBlockRV>()) {
               return self->Get(block_rv.value());
             }
             if (auto expr_rv = obj.as<ExprRV>()) {
               return self->Get(expr_rv.value());
             }
             TVM_FFI_THROW(TypeError)
                 << "Cannot evaluate the random variable of type: " << obj->GetTypeKey()
                 << ". Its value is: " << obj;
             throw;
           })
      .def("s_tir.schedule.ScheduleGetSRef",
           [](Schedule self, ffi::ObjectRef obj) -> ffi::Optional<ffi::ObjectRef> {
             if (auto loop_rv = obj.as<LoopRV>()) {
               return self->GetSRef(loop_rv.value());
             }
             if (auto block_rv = obj.as<SBlockRV>()) {
               return self->GetSRef(block_rv.value());
             }
             if (auto stmt = obj.as<Stmt>()) {
               return self->GetSRef(stmt.value());
             }
             TVM_FFI_THROW(TypeError) << "Invalid type: " << obj->GetTypeKey();
             throw;
           })
      .def("s_tir.schedule.ScheduleRemoveRV", [](Schedule self, ffi::ObjectRef obj) -> void {
        if (auto loop_rv = obj.as<LoopRV>()) {
          return self->RemoveRV(loop_rv.value());
        }
        if (auto block_rv = obj.as<SBlockRV>()) {
          return self->RemoveRV(block_rv.value());
        }
        if (auto expr_rv = obj.as<ExprRV>()) {
          return self->RemoveRV(expr_rv.value());
        }
        TVM_FFI_THROW(TypeError) << "Invalid type: " << obj->GetTypeKey();
        throw;
      });
}

/******** (FFI) Sampling ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("s_tir.schedule.ScheduleSampleCategorical", &ScheduleObj::SampleCategorical)
      .def_method("s_tir.schedule.ScheduleSamplePerfectTile", &ScheduleObj::SamplePerfectTile)
      .def_method("s_tir.schedule.ScheduleSamplePartitionedTile",
                  &ScheduleObj::SamplePartitionedTile)
      .def_method("s_tir.schedule.ScheduleSampleComputeLocation",
                  &ScheduleObj::SampleComputeLocation);
}
/******** (FFI) Get blocks & loops ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("s_tir.schedule.ScheduleGetSBlock", &ScheduleObj::GetSBlock)
      .def_method("s_tir.schedule.ScheduleGetLoops", &ScheduleObj::GetLoops)
      .def("s_tir.schedule.ScheduleGetChildBlocks",
           [](Schedule self, ffi::ObjectRef rv) {
             if (auto block_rv = rv.as<SBlockRV>()) {
               return self->GetChildBlocks(block_rv.value());
             }
             if (auto loop_rv = rv.as<LoopRV>()) {
               return self->GetChildBlocks(loop_rv.value());
             }
             TVM_FFI_THROW(TypeError)
                 << "Cannot evaluate the random variable of type: " << rv->GetTypeKey()
                 << ". Its value is: " << rv;
             throw;
           })
      .def_method("s_tir.schedule.ScheduleGetProducers", &ScheduleObj::GetProducers)
      .def_method("s_tir.schedule.ScheduleGetConsumers", &ScheduleObj::GetConsumers)
      .def_method("s_tir.schedule.ScheduleGetOutputBlocks", &ScheduleObj::GetOutputBlocks);
}
/******** (FFI) Transform loops ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("s_tir.schedule.ScheduleMerge", &ScheduleObj::Merge)
      .def_method("s_tir.schedule.ScheduleFuse", &ScheduleObj::Fuse)
      .def_method("s_tir.schedule.ScheduleSplit", &ScheduleObj::Split)
      .def_method("s_tir.schedule.ScheduleLoopPartition", &ScheduleObj::LoopPartition)
      .def_method("s_tir.schedule.ScheduleReorder", &ScheduleObj::Reorder)
      .def_method("s_tir.schedule.ScheduleReorderBlockIterVar", &ScheduleObj::ReorderBlockIterVar)
      .def("s_tir.schedule.ScheduleAddUnitLoop", [](Schedule self, ffi::ObjectRef rv) -> LoopRV {
        if (auto loop_rv = rv.as<LoopRV>()) {
          return self->AddUnitLoop(loop_rv.value());
        } else if (auto block_rv = rv.as<SBlockRV>()) {
          return self->AddUnitLoop(block_rv.value());
        } else {
          TVM_FFI_THROW(TypeError)
              << "Cannot evaluate the random variable of type: " << rv->GetTypeKey()
              << ". Its value is: " << rv;
          throw;
        }
      });
}
/******** (FFI) Manipulate ForKind ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("s_tir.schedule.ScheduleParallel", &ScheduleObj::Parallel)
      .def_method("s_tir.schedule.ScheduleVectorize", &ScheduleObj::Vectorize)
      .def_method("s_tir.schedule.ScheduleBind", &ScheduleObj::Bind)
      .def_method("s_tir.schedule.ScheduleUnroll", &ScheduleObj::Unroll);
}
/******** (FFI) Insert cache stages ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("s_tir.schedule.ScheduleCacheRead", &ScheduleObj::CacheRead)
      .def_method("s_tir.schedule.ScheduleCacheWrite", &ScheduleObj::CacheWrite)
      .def_method("s_tir.schedule.ScheduleReindexCacheRead", &ScheduleObj::ReindexCacheRead)
      .def_method("s_tir.schedule.ScheduleReindexCacheWrite", &ScheduleObj::ReindexCacheWrite)
      .def_method("s_tir.schedule.ScheduleCacheInplace", &ScheduleObj::CacheInplace)
      .def_method("s_tir.schedule.ScheduleCacheIndex", &ScheduleObj::CacheIndex)
      .def("s_tir.schedule.ScheduleReIndex",
           [](Schedule self, const SBlockRV& block_rv, int tensor_index, int tensor_index_type) {
             return self->ReIndex(block_rv, tensor_index,
                                  static_cast<TensorIndexType>(tensor_index_type));
           });
}
/******** (FFI) Data movement ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("s_tir.schedule.ScheduleReadAt", &ScheduleObj::ReadAt)
      .def_method("s_tir.schedule.ScheduleWriteAt", &ScheduleObj::WriteAt);
}
/******** (FFI) Compute location ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("s_tir.schedule.ScheduleComputeAt", &ScheduleObj::ComputeAt)
      .def_method("s_tir.schedule.ScheduleReverseComputeAt", &ScheduleObj::ReverseComputeAt)
      .def_method("s_tir.schedule.ScheduleComputeInline", &ScheduleObj::ComputeInline)
      .def_method("s_tir.schedule.ScheduleReverseComputeInline", &ScheduleObj::ReverseComputeInline)
      .def_method("s_tir.schedule.ScheduleFuseReductionEpilogue",
                  &ScheduleObj::FuseReductionEpilogue);
}
/******** (FFI) Reduction ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("s_tir.schedule.ScheduleDecomposeReduction", &ScheduleObj::DecomposeReduction)
      .def_method("s_tir.schedule.ScheduleRFactor", &ScheduleObj::RFactor);
}
/******** (FFI) SBlock annotation ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("s_tir.schedule.ScheduleStorageAlign", &ScheduleObj::StorageAlign)
      .def_method("s_tir.schedule.ScheduleSetScope", &ScheduleObj::SetScope)
      .def_method("s_tir.schedule.ScheduleUnsafeSetDType", &ScheduleObj::UnsafeSetDType);
}
/******** (FFI) Blockize & Tensorize ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("s_tir.schedule.ScheduleBlockize",
           [](Schedule self, ffi::ObjectRef target, bool preserve_unit_iters) {
             if (auto loop_rv = target.as<LoopRV>()) {
               return self->Blockize(loop_rv.value(), preserve_unit_iters);
             } else if (auto blocks = target.as<ffi::Array<SBlockRV>>()) {
               return self->Blockize(blocks.value(), preserve_unit_iters);
             }
             TVM_FFI_THROW(InternalError) << "Unsupported target type: " << target->GetTypeKey();
           })
      .def("s_tir.schedule.ScheduleTensorize",
           [](Schedule self, ffi::ObjectRef rv, ffi::String intrin, bool preserve_unit_iters) {
             if (auto block_rv = rv.as<SBlockRV>()) {
               self->Tensorize(block_rv.value(), intrin, preserve_unit_iters);
             } else if (auto loop_rv = rv.as<LoopRV>()) {
               self->Tensorize(loop_rv.value(), intrin, preserve_unit_iters);
             } else {
               TVM_FFI_THROW(TypeError)
                   << "Cannot evaluate the random variable of type: " << rv->GetTypeKey()
                   << ". Its value is: " << rv;
             }
           });
}

/******** (FFI) Annotation ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("s_tir.schedule.ScheduleAnnotate",
           [](Schedule self, ffi::ObjectRef rv, const ffi::String& ann_key, const Any& ann_val) {
             if (auto block_rv = rv.as<SBlockRV>()) {
               return self->Annotate(block_rv.value(), ann_key, ann_val);
             }
             if (auto loop_rv = rv.as<LoopRV>()) {
               return self->Annotate(loop_rv.value(), ann_key, ann_val);
             }
             TVM_FFI_THROW(TypeError)
                 << "Cannot evaluate the random variable of type: " << rv->GetTypeKey()
                 << ". Its value is: " << rv;
             throw;
           })
      .def("s_tir.schedule.ScheduleUnannotate",
           [](Schedule self, ffi::ObjectRef rv, const ffi::String& ann_key) {
             if (auto block_rv = rv.as<SBlockRV>()) {
               return self->Unannotate(block_rv.value(), ann_key);
             }
             if (auto loop_rv = rv.as<LoopRV>()) {
               return self->Unannotate(loop_rv.value(), ann_key);
             }
             TVM_FFI_THROW(TypeError)
                 << "Cannot evaluate the random variable of type: " << rv->GetTypeKey()
                 << ". Its value is: " << rv;
             throw;
           });
}

/******** (FFI) Layout transformation ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("s_tir.schedule.ScheduleTransformLayout",
           [](Schedule self, const SBlockRV& block_rv, int tensor_index, int tensor_index_type,
              const IndexMap& index_map, const ffi::Optional<IndexMap>& pad_value,
              bool assume_injective_transform) {
             return self->TransformLayout(block_rv, tensor_index,
                                          static_cast<TensorIndexType>(tensor_index_type),
                                          index_map, pad_value, assume_injective_transform);
           })
      .def_method("s_tir.schedule.ScheduleTransformBlockLayout",
                  &ScheduleObj::TransformBlockLayout);
}

/******** (FFI) Padding decomposition ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("s_tir.schedule.ScheduleDecomposePadding", &ScheduleObj::DecomposePadding)
      .def_method("s_tir.schedule.SchedulePadEinsum", &ScheduleObj::PadEinsum);
}
/******** (FFI) Tensor transformation ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_method("s_tir.schedule.ScheduleRollingBuffer", &ScheduleObj::RollingBuffer);
}
/******** (FFI) Misc ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("s_tir.schedule.ScheduleEnterPostproc", &ScheduleObj::EnterPostproc)
      .def_method("s_tir.schedule.ScheduleUnsafeHideTensorAccess",
                  &ScheduleObj::UnsafeHideTensorAccess);
}
/******** (FFI) Annotate tensor access ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.schedule.ScheduleAnnotateTensorAccess",
                        [](Schedule self, const SBlockRV& block_rv, int tensor_index,
                           int tensor_index_type, const IndexMap& index_map) {
                          return self->AnnotateTensorAccess(
                              block_rv, tensor_index,
                              static_cast<TensorIndexType>(tensor_index_type), index_map);
                        });
}

}  // namespace s_tir
}  // namespace tvm
