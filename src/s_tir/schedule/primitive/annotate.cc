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
#include <tvm/tirx/stmt.h>

#include "../utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

void Annotate(ScheduleState self, const StmtSRef& sref, const ffi::String& ann_key,
              const Any& ann_val) {
  // Extract annotation
  const ffi::Map<ffi::String, ffi::Any>* annotations = nullptr;
  if (const auto* loop = sref->StmtAs<ForObj>()) {
    TVM_FFI_CHECK(ann_key != tvm::s_tir::attr::kThreadBinding.name, ValueError)
        << "Use loop scheduling primitives to change thread binding";
    annotations = &loop->annotations;
  } else if (const auto* block = sref->StmtAs<SBlockObj>()) {
    annotations = &block->annotations;
  } else {
    TVM_FFI_THROW(TypeError) << "Unknown type of sref: " << sref->stmt->GetTypeKey();
  }
  // Check if the annotation already exists
  if (annotations->find(ann_key) != annotations->end()) {
    return;
  }
  // Add the new annotation
  ffi::Map<ffi::String, ffi::Any> new_ann(*annotations);
  new_ann.Set(ann_key, ann_val);
  // Create the new stmt
  if (const auto* loop = sref->StmtAs<ForObj>()) {
    ffi::ObjectPtr<ForObj> n = ffi::make_object<ForObj>(*loop);
    n->annotations = std::move(new_ann);
    self->Replace(sref, For(n), {});
  } else if (const auto* block = sref->StmtAs<SBlockObj>()) {
    ffi::ObjectPtr<SBlockObj> n = ffi::make_object<SBlockObj>(*block);
    n->annotations = std::move(new_ann);
    SBlock p(n);
    self->Replace(sref, p, {{ffi::GetRef<SBlock>(block), p}});
  } else {
    TVM_FFI_THROW(TypeError) << "Unknown type of sref: " << sref->stmt->GetTypeKey();
    throw;
  }
}

void Unannotate(ScheduleState self, const StmtSRef& sref, const ffi::String& ann_key) {
  // Extract annotation
  const ffi::Map<ffi::String, ffi::Any>* annotations = nullptr;
  if (const auto* loop = sref->StmtAs<ForObj>()) {
    TVM_FFI_CHECK(ann_key != tvm::s_tir::attr::kThreadBinding.name, ValueError)
        << "Use loop scheduling primitives to change thread binding";
    annotations = &loop->annotations;
  } else if (const auto* block = sref->StmtAs<SBlockObj>()) {
    annotations = &block->annotations;
  } else {
    TVM_FFI_THROW(TypeError) << "Unknown type of sref: " << sref->stmt->GetTypeKey();
  }
  // Remove the annotation
  TVM_FFI_CHECK(annotations->find(ann_key) != annotations->end(), IndexError)
      << "Cannot find annotation key: " << ann_key;
  ffi::Map<ffi::String, ffi::Any> new_ann(*annotations);
  new_ann.erase(ann_key);
  // Create the new stmt
  if (const auto* loop = sref->StmtAs<ForObj>()) {
    ffi::ObjectPtr<ForObj> n = ffi::make_object<ForObj>(*loop);
    n->annotations = std::move(new_ann);
    self->Replace(sref, For(n), {});
  } else if (const auto* block = sref->StmtAs<SBlockObj>()) {
    ffi::ObjectPtr<SBlockObj> n = ffi::make_object<SBlockObj>(*block);
    n->annotations = std::move(new_ann);
    SBlock p(n);
    self->Replace(sref, p, {{ffi::GetRef<SBlock>(block), p}});
  } else {
    TVM_FFI_THROW(TypeError) << "Unknown type of sref: " << sref->stmt->GetTypeKey();
    throw;
  }
}

struct AnnotateTraits : public UnpackedInstTraits<AnnotateTraits> {
  static constexpr const char* kName = "Annotate";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, ffi::ObjectRef block_or_loop_rv, Any ann_val,
                                      ffi::String ann_key) {
    if (auto block = block_or_loop_rv.as<SBlockRV>()) {
      return sch->Annotate(block.value(), ann_key, ann_val);
    }
    if (auto loop = block_or_loop_rv.as<LoopRV>()) {
      return sch->Annotate(loop.value(), ann_key, ann_val);
    }
    TVM_FFI_THROW(TypeError) << "Expected SBlock or Loop, but gets: "
                             << block_or_loop_rv->GetTypeKey();
    throw;
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs,
                                      ffi::ObjectRef block_or_loop_rv, Any ann_val,
                                      ffi::String ann_key) {
    PythonAPICall py("annotate");
    py.Input("block_or_loop", block_or_loop_rv);
    py.Input("ann_key", ann_key);
    py.Input("ann_val", ann_val);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

struct UnannotateTraits : public UnpackedInstTraits<UnannotateTraits> {
  static constexpr const char* kName = "Unannotate";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, ffi::ObjectRef block_or_loop_rv,
                                      ffi::String ann_key) {
    if (auto block = block_or_loop_rv.as<SBlockRV>()) {
      return sch->Unannotate(block.value(), ann_key);
    }
    if (auto loop = block_or_loop_rv.as<LoopRV>()) {
      return sch->Unannotate(loop.value(), ann_key);
    }
    TVM_FFI_THROW(TypeError) << "Expected SBlock or Loop, but gets: "
                             << block_or_loop_rv->GetTypeKey();
    throw;
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs,
                                      ffi::ObjectRef block_or_loop_rv, ffi::String ann_key) {
    PythonAPICall py("unannotate");
    py.Input("block_or_loop", block_or_loop_rv);
    py.Input("ann_key", ann_key);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

TVM_FFI_STATIC_INIT_BLOCK() {
  RegisterInstructionKind<AnnotateTraits>();
  RegisterInstructionKind<UnannotateTraits>();
}

}  // namespace s_tir
}  // namespace tvm
