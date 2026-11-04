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
#include <tvm/ffi/container/tuple.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/s_tir/stmt.h>

#include "../../transform/ir_utils.h"
#include "../utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

class StorageAlignAxisOutOfRangeError : public ScheduleErrorContextObj {
 public:
  explicit StorageAlignAxisOutOfRangeError(IRModule mod, TensorVar tensor, int axis)
      : mod_(std::move(mod)), tensor_(std::move(tensor)), axis_(axis) {}

  ffi::String FastErrorString() const final {
    return "ScheduleError: The input `axis` is out of range. It is required to be in range "
           "[-ndim, ndim) where `ndim` is the number of dimensions of the tensor to set "
           "storage alignment.";
  }

  ffi::String DetailRenderTemplate() const final {
    std::ostringstream os;
    int ndim = static_cast<int>(tensor_.shape().size());
    os << "The tensor to set storage alignment of, " << tensor_->name << ", has " << ndim
       << " dimension(s), so `axis` is required to be in [" << -(ndim) << ", " << ndim
       << ") for storage_align. However, the input `axis` is " << axis_
       << ", which is out of the expected range.";
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {}; }

  static int CheckAndUpdate(const IRModule& mod, const TensorVar& tensor, int axis) {
    int ndim = static_cast<int>(tensor.shape().size());
    if (axis < -ndim || axis >= ndim) {
      throw MakeScheduleError<StorageAlignAxisOutOfRangeError>(mod, tensor, axis);
    }
    // If axis is negative, convert it to a non-negative one.
    if (axis < 0) {
      axis += ndim;
    }
    return axis;
  }

 private:
  IRModule mod_;
  TensorVar tensor_;
  int axis_;
};

class NonAllocatedTensorError : public ScheduleErrorContextObj {
 public:
  explicit NonAllocatedTensorError(IRModule mod, TensorVar tensor) : mod_(mod), tensor_(tensor) {}

  ffi::String FastErrorString() const final {
    return "ScheduleError: The input tensor is not allocated by a block. This means the tensor is "
           " either a function parameter or defined in `match_tensor` of a block.";
  }

  ffi::String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The input tensor " << tensor_->name
       << " is not allocated by a block. This means the tensor is either a function parameter or "
          "defined in `match_tensor` of a block.";
    return os.str();
  }

  static StmtSRef CheckAndGetTensorAllocationSite(const IRModule& mod, const StmtSRef& block_sref,
                                                  const TensorVar& tensor) {
    auto [defining_site_sref, is_alloc] = GetTensorDefiningSite(block_sref, tensor);
    if (!defining_site_sref.has_value() || !is_alloc) {
      throw MakeScheduleError<NonAllocatedTensorError>(mod, tensor);
    }

    return defining_site_sref.value();
  }

  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {}; }
  IRModule mod() const final { return mod_; }

 private:
  IRModule mod_;
  TensorVar tensor_;
};

class StorageAlignInvalidFactorError : public ScheduleErrorContextObj {
 public:
  explicit StorageAlignInvalidFactorError(IRModule mod, int factor)
      : mod_(std::move(mod)), factor_(factor) {}

  ffi::String FastErrorString() const final {
    return "ScheduleError: The input `factor` of storage_align is expected to be a positive "
           "number.";
  }

  ffi::String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The input `factor` of storage_align is expected to be a positive number. However, the "
          "input `factor` is "
       << factor_ << ", which is out of the expected range.";
    return os.str();
  }

  static void Check(const IRModule& mod, int factor) {
    if (factor <= 0) {
      throw MakeScheduleError<StorageAlignInvalidFactorError>(mod, factor);
    }
  }

  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {}; }
  IRModule mod() const final { return mod_; }

 private:
  IRModule mod_;
  int factor_;
};

class StorageAlignInvalidAnnotationError : public ScheduleErrorContextObj {
 public:
  explicit StorageAlignInvalidAnnotationError(IRModule mod, SBlock block)
      : mod_(std::move(mod)), block_(std::move(block)) {}

  ffi::String FastErrorString() const final {
    return "ScheduleError: The block annotation for storage align is expected to be an array of "
           "4-integer-tuples (tensor_index, axis, factor, offset).";
  }

  ffi::String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The block annotation for storage align is expected to be an array of 4-integer-tuples "
          "(tensor_index, axis, factor, offset). However, the block annotation with key "
       << s_tir::attr::kTensorDimAlign.name << " of the block {0} is "
       << block_->annotations.at(s_tir::attr::kTensorDimAlign.name) << ", which is unexpected.";
    return os.str();
  }

  static StorageAlignAnnotation CheckAndGetAnnotation(const IRModule& mod, const SBlock& block) {
    // Get existing annotation value.
    auto it = block->annotations.find(s_tir::attr::kTensorDimAlign.name);
    if (it != block->annotations.end()) {
      if (!IsValidAnnotation(block, (*it).second)) {
        throw MakeScheduleError<StorageAlignInvalidAnnotationError>(mod, block);
      }
      return (*it).second.as_or_throw<StorageAlignAnnotation>();
    }

    // Create new annotation value
    StorageAlignAnnotation storage_align_annotation;
    return storage_align_annotation;
  }

  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_}; }
  IRModule mod() const final { return mod_; }

 private:
  static bool IsValidAnnotation(const SBlock& block, const Any& anno_value) {
    return anno_value.try_cast<ffi::Array<ffi::Tuple<int, int, int, int>>>().has_value();
  }

  IRModule mod_;
  SBlock block_;
};

/*!
 * \brief A helper mutator which recursively mutates the old tensor's storage scope and collects
 * the block sref reuse information for the following replacement.
 */
class StorageScopeMutator : public ReplaceTensorMutator {
 public:
  using ReplaceTensorMutator::Mutate;
  using ReplaceTensorMutator::Mutate_;

  /*!
   * \param allocate_site The block where `old_tensor` was allocated.
   * \param old_tensor The old tensor
   * \param storage_scope The storage scope to be set
   * \param block_sref_reuse The block sref reuse map to be updated
   * \return The new block after the mutation
   */
  static SBlock Mutate(const SBlock& allocate_site, const TensorVar& old_tensor,
                       const ffi::String& storage_scope,
                       ffi::Map<SBlock, SBlock>* block_sref_reuse) {
    TensorVar new_tensor = WithScope(old_tensor, runtime::MemoryScope::Get(storage_scope));
    auto mutator = ffi::make_object<StorageScopeMutator>(old_tensor, new_tensor, storage_scope,
                                                         block_sref_reuse);
    Stmt new_block = mutator->Mutate(allocate_site).ValueOrUnchanged(allocate_site);
    return new_block.as_or_throw<SBlock>();
  }

  StorageScopeMutator(const TensorVar& old_tensor, TensorVar new_tensor, ffi::String storage_scope,
                      ffi::Map<SBlock, SBlock>* block_sref_reuse)
      : ReplaceTensorMutator(old_tensor, std::move(new_tensor), block_sref_reuse) {}

 private:
  MatchTensorRegion VisitMatchTensorRegion(const MatchTensorRegion& match_tensor) final {
    if (auto replacement =
            VarRemapGet(match_tensor->source->source.as_or_throw<tvm::tirx::TensorVar>())
                .as<TensorVar>()) {
      TensorVar new_target_tensor = WithScope(match_tensor->tensor, replacement.value().scope());
      VarRemapSet(match_tensor->tensor, new_target_tensor);
      return MatchTensorRegion(
          new_target_tensor,
          TensorRegion(replacement.value(), match_tensor->source->region, TensorRegionType()));
    } else {
      return match_tensor;
    }
  }
};

void StorageAlign(ScheduleState self, const StmtSRef& block_sref, int tensor_index, int axis,
                  int factor, int offset) {
  const SBlockObj* block_ptr = TVM_SREF_TO_SBLOCK(block_sref);
  TensorVar tensor = GetNthAccessTensor(self, ffi::GetRef<SBlock>(block_ptr), tensor_index,
                                        TensorIndexType::kWrite);
  StorageAlignInvalidFactorError::Check(self->mod, factor);
  axis = StorageAlignAxisOutOfRangeError::CheckAndUpdate(self->mod, tensor, axis);
  NonAllocatedTensorError::CheckAndGetTensorAllocationSite(self->mod, block_sref, tensor);

  // Step 1: Get existing or create new annotation value.
  StorageAlignAnnotation storage_align_annotation =
      StorageAlignInvalidAnnotationError::CheckAndGetAnnotation(self->mod,
                                                                ffi::GetRef<SBlock>(block_ptr));

  // Step 2: Update the annotation value
  bool found = false;
  StorageAlignTuple new_storage_align_tuple{tensor_index, axis, factor, offset};
  for (size_t j = 0; j < storage_align_annotation.size(); ++j) {
    const auto& storage_align_tuple = storage_align_annotation[j];
    if (storage_align_tuple.get<0>() == tensor_index && storage_align_tuple.get<1>() == axis) {
      storage_align_annotation.Set(j, std::move(new_storage_align_tuple));
      found = true;
      break;
    }
  }
  if (!found) {
    storage_align_annotation.push_back(std::move(new_storage_align_tuple));
  }

  // Step 3: Replace the block with the new annotation
  SBlock new_block =
      WithAnnotation(block_ptr, s_tir::attr::kTensorDimAlign.name, storage_align_annotation);
  self->Replace(block_sref, new_block, {{ffi::GetRef<SBlock>(block_ptr), new_block}});
}

void SetScope(ScheduleState self, const StmtSRef& block_sref, int tensor_index,
              const ffi::String& storage_scope) {
  const SBlockObj* block = TVM_SREF_TO_SBLOCK(block_sref);
  TensorVar tensor =
      GetNthAccessTensor(self, ffi::GetRef<SBlock>(block), tensor_index, TensorIndexType::kWrite);

  // Step 1. Validate before enum lookup to preserve the scheduling error diagnostic.
  CheckStorageScope(self, storage_scope);

  // Step 2. If `storage_scope` equals the original storage scope of the tensor, just return.
  if (tensor.scope() == runtime::MemoryScope::Get(storage_scope)) {
    return;
  }

  // Step 3. Get the allocation site of the target tensor.
  StmtSRef alloc_site_sref =
      NonAllocatedTensorError::CheckAndGetTensorAllocationSite(self->mod, block_sref, tensor);
  const SBlockObj* alloc_site = TVM_SREF_TO_SBLOCK(alloc_site_sref);

  // Step 4. Recursively replace the old tensor to a new tensor, where the new tensor has the given
  // storage scope. In the meanwhile, collect the block sref reuse information.
  ffi::Map<SBlock, SBlock> block_reuse_map;
  SBlock new_block = StorageScopeMutator::Mutate(ffi::GetRef<SBlock>(alloc_site), tensor,
                                                 storage_scope, &block_reuse_map);
  self->Replace(alloc_site_sref, new_block, block_reuse_map);
}

/*!
 * \brief A helper mutator which recursively mutates the old tensor's data type, inserts data type
 * conversions, and collecte the block sref reuse information for the following replacement.
 */
class DTypeMutator : public ReplaceTensorMutator {
 public:
  using ReplaceTensorMutator::Mutate;
  using ReplaceTensorMutator::Mutate_;

  /*!
   * \param allocate_site The block where `old_tensor` was allocated.
   * \param old_tensor The old tensor
   * \param target_dtype The data type to be set
   * \param block_sref_reuse The block sref reuse map to be updated
   * \return The new block after the mutation
   */
  static SBlock Mutate(const SBlock& allocate_site, const TensorVar& old_tensor, PrimType dtype,
                       ffi::Map<SBlock, SBlock>* block_sref_reuse) {
    TensorVar new_tensor = WithDType(old_tensor, dtype);
    auto mutator = ffi::make_object<DTypeMutator>(old_tensor, new_tensor, dtype, block_sref_reuse);
    Stmt new_block = mutator->Mutate(allocate_site).ValueOrUnchanged(allocate_site);
    return new_block.as_or_throw<SBlock>();
  }

  DTypeMutator(const TensorVar& old_tensor, TensorVar new_tensor, PrimType dtype,
               ffi::Map<SBlock, SBlock>* block_sref_reuse)
      : ReplaceTensorMutator(old_tensor, std::move(new_tensor), block_sref_reuse),
        src_dtype_(old_tensor.dtype()),
        tgt_dtype_(dtype) {}

 private:
  MatchTensorRegion VisitMatchTensorRegion(const MatchTensorRegion& match_tensor) final {
    if (auto replacement =
            VarRemapGet(match_tensor->source->source.as_or_throw<tvm::tirx::TensorVar>())
                .as<TensorVar>()) {
      TensorVar new_target_tensor = WithDType(match_tensor->tensor, replacement.value().dtype());
      VarRemapSet(match_tensor->tensor, new_target_tensor);
      return MatchTensorRegion(
          new_target_tensor,
          TensorRegion(replacement.value(), match_tensor->source->region, TensorRegionType()));
    } else {
      return match_tensor;
    }
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreObj* op, InplaceMode inplace_mode) final {
    TensorVar original_tensor = op->dest.as_or_throw<TensorVar>();
    TensorStore node = StmtExprMutator::Mutate_(op, inplace_mode)
                           .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                           .as_or_throw<TensorStore>();
    if (auto replacement = VarRemapGet(original_tensor).as<TensorVar>()) {
      node.CopyOnWrite()->dest = replacement.value();
      node.CopyOnWrite()->value = Cast(tgt_dtype_, node->value);
    }
    return node;
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* op, InplaceMode inplace_mode) final {
    Expr original_tensor = op->source;
    TensorLoad node = StmtExprMutator::Mutate_(op, inplace_mode)
                          .ValueOrUnchanged(ffi::GetRef<PrimExpr>(op))
                          .as_or_throw<TensorLoad>();
    if (auto replacement = VarRemapGet(original_tensor).as<TensorVar>()) {
      return Cast(src_dtype_, TensorLoad(replacement.value(), node->indices));
    }
    return node;
  }

  PrimType src_dtype_, tgt_dtype_;
};

void UnsafeSetDType(ScheduleState self, const StmtSRef& block_sref, int tensor_index,
                    const ffi::String& dtype) {
  const SBlockObj* block = TVM_SREF_TO_SBLOCK(block_sref);
  TensorVar tensor =
      GetNthAccessTensor(self, ffi::GetRef<SBlock>(block), tensor_index, TensorIndexType::kWrite);
  PrimType target_dtype(ffi::StringToDLDataType(dtype));

  // Step 1. If `dtype` equals the original data type, just return.
  if (tensor.dtype() == target_dtype) {
    return;
  }

  // Step 2. Get the allocation site of the target tensor.
  StmtSRef alloc_site_sref =
      NonAllocatedTensorError::CheckAndGetTensorAllocationSite(self->mod, block_sref, tensor);
  const SBlockObj* alloc_site = TVM_SREF_TO_SBLOCK(alloc_site_sref);

  // Step 3. Recursively replace old tensor to a new tensor, where the new tensor has the given
  // dtype, and insert data type conversions.
  ffi::Map<SBlock, SBlock> block_reuse_map;
  SBlock new_block =
      DTypeMutator::Mutate(ffi::GetRef<SBlock>(alloc_site), tensor, target_dtype, &block_reuse_map);
  self->Replace(alloc_site_sref, new_block, block_reuse_map);
}

/******** InstructionKind Registration ********/

struct StorageAlignTraits : public UnpackedInstTraits<StorageAlignTraits> {
  static constexpr const char* kName = "StorageAlign";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 4;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, SBlockRV block_rv, int tensor_index, int axis,
                                      int factor, int offset) {
    return sch->StorageAlign(block_rv, tensor_index, axis, factor, offset);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block_rv,
                                      int tensor_index, int axis, int factor, int offset) {
    PythonAPICall py("storage_align");
    py.Input("block", block_rv);
    py.Input("tensor_index", tensor_index);
    py.Input("axis", axis);
    py.Input("factor", factor);
    py.Input("offset", offset);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

struct SetScopeTraits : public UnpackedInstTraits<SetScopeTraits> {
  static constexpr const char* kName = "SetScope";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, SBlockRV block_rv, int tensor_index,
                                      ffi::String storage_scope) {
    return sch->SetScope(block_rv, tensor_index, storage_scope);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block_rv,
                                      int tensor_index, ffi::String storage_scope) {
    PythonAPICall py("set_scope");
    py.Input("block", block_rv);
    py.Input("tensor_index", tensor_index);
    py.Input("storage_scope", storage_scope);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

struct UnsafeSetDTypeTraits : public UnpackedInstTraits<UnsafeSetDTypeTraits> {
  static constexpr const char* kName = "UnsafeSetDType";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, SBlockRV block_rv, int tensor_index,
                                      ffi::String dtype) {
    return sch->UnsafeSetDType(block_rv, tensor_index, dtype);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block_rv,
                                      int tensor_index, ffi::String dtype) {
    PythonAPICall py("unsafe_set_dtype");
    py.Input("block", block_rv);
    py.Input("tensor_index", tensor_index);
    py.Input("dtype", dtype);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

TVM_FFI_STATIC_INIT_BLOCK() {
  RegisterInstructionKind<StorageAlignTraits>();
  RegisterInstructionKind<SetScopeTraits>();
  RegisterInstructionKind<UnsafeSetDTypeTraits>();
}

}  // namespace s_tir
}  // namespace tvm
