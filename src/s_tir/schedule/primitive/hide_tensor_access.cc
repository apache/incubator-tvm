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

#include <set>

#include "../utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

/******** Error Classes ********/

namespace {
class TensorTypeError : public ScheduleErrorContextObj {
 public:
  explicit TensorTypeError(IRModule mod, const ffi::String& tensor_type)
      : mod_(std::move(mod)), tensor_type_(tensor_type) {}

  ffi::String FastErrorString() const final {
    return "ScheduleError: Invalid tensor type for hide_tensor_access schedule.";
  }

  ffi::String DetailRenderTemplate() const final {
    return "The tensor type for hide_tensor_access schedule should either be 'read'"
           " or 'write', got " +
           tensor_type_ + " instead.";
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {}; }

 private:
  IRModule mod_;
  ffi::String tensor_type_;
};

class InvalidIndexError : public ScheduleErrorContextObj {
 public:
  explicit InvalidIndexError(IRModule mod, int num_access_regions, int tensor_idx)
      : mod_(std::move(mod)), num_access_regions_(num_access_regions), tensor_idx_(tensor_idx) {}

  ffi::String FastErrorString() const final {
    return "ScheduleError: Invalid tensor index array for hide_tensor_access schedule.";
  }

  ffi::String DetailRenderTemplate() const final {
    return "The tensor index array for hide_tensor_access schedule should be a list of integers"
           " between 0 and " +
           std::to_string(num_access_regions_ - 1) + ", got " + std::to_string(tensor_idx_) +
           " instead.";
  }

  IRModule mod() const final { return mod_; }

  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {}; }

 private:
  IRModule mod_;
  int num_access_regions_;
  int tensor_idx_;
};

}  // namespace

/******** Implementation ********/

void UnsafeHideTensorAccess(ScheduleState self, const StmtSRef& block_sref,
                            const ffi::String& tensor_type,
                            const ffi::Array<int64_t>& tensor_index_array) {
  /*!
   * Check:
   *   - validity of tensor_index_array
   *   - validity of tensor_type
   */
  const SBlockObj* block = TVM_SREF_TO_SBLOCK(block_sref);
  int num_access_regions = 0;
  if (tensor_type == "read") {
    num_access_regions = block->reads.size();
  } else if (tensor_type == "write") {
    num_access_regions = block->writes.size();
  } else {
    throw MakeScheduleError<TensorTypeError>(self->mod, tensor_type);
  }

  std::set<int> tensor_indices;
  for (int64_t tensor_idx_val : tensor_index_array) {
    if (tensor_idx_val >= 0 && tensor_idx_val < num_access_regions) {
      tensor_indices.insert(tensor_idx_val);
    } else {
      throw MakeScheduleError<InvalidIndexError>(self->mod, num_access_regions, tensor_idx_val);
    }
  }

  /* Step 0: Collect new tensor access regions. */

  ffi::Array<TensorRegion> reads, writes;

  if (tensor_type == "read") {
    for (size_t i = 0; i < block->reads.size(); ++i) {
      if (!tensor_indices.count(i)) {
        reads.push_back(block->reads[i]);
      }
    }
    writes = block->writes;
  } else if (tensor_type == "write") {
    for (size_t i = 0; i < block->writes.size(); ++i) {
      if (!tensor_indices.count(i)) {
        writes.push_back(block->writes[i]);
      }
    }
    reads = block->reads;
  } else {
    TVM_FFI_ICHECK(false) << "Unrecognized tensor type " << tensor_type
                          << ", only support read/write";
  }

  /* Step 1: Replace old block with the new block */

  auto n = ffi::make_object<SBlockObj>(*block);
  n->reads = reads;
  n->writes = writes;
  SBlock new_block = SBlock(n);
  ffi::Map<SBlock, SBlock> blk_map;
  blk_map.Set(ffi::GetRef<SBlock>(block), new_block);
  self->Replace(block_sref, new_block, blk_map);
}

struct UnsafeHideTensorAccessTraits : public UnpackedInstTraits<UnsafeHideTensorAccessTraits> {
  static constexpr const char* kName = "UnsafeHideTensorAccess";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 3;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, SBlockRV block, ffi::String tensor_type,
                                      ffi::Array<int64_t> tensor_index_array) {
    sch->UnsafeHideTensorAccess(block, tensor_type, tensor_index_array);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block,
                                      ffi::String tensor_type,
                                      ffi::Array<int64_t> tensor_index_array) {
    PythonAPICall py("unsafe_hide_tensor_access");
    py.Input("block", block);
    py.Input("tensor_type", tensor_type);
    py.Input("tensor_index_array", tensor_index_array);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

TVM_FFI_STATIC_INIT_BLOCK() { RegisterInstructionKind<UnsafeHideTensorAccessTraits>(); }

}  // namespace s_tir
}  // namespace tvm
