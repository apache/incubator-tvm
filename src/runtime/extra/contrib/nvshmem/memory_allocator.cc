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
#include <nvshmem.h>
#include <nvshmemx.h>
#include <tvm/ffi/extra/cuda/base.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/enum_def.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/tensor.h>

#include "../../disco/utils.h"

namespace tvm {
namespace runtime {

/*!
 * \brief Tensor pool for NVSHMEM symmetric memory.
 * Allocations, releases and finalization retain NVSHMEM's collective ordering
 * requirements. All tensor owners must be released before finalization.
 */
class NVSHMEMPool final : public TensorMemoryPool {
 public:
  explicit NVSHMEMPool() = default;

  /*! \brief Return the global NVSHMEM singleton pool. */
  static NVSHMEMPool* Global() {
    static auto* pool = new ffi::ObjectPtr<NVSHMEMPool>(ffi::make_object<NVSHMEMPool>());
    return pool->get();
  }

  ffi::Tensor Empty(ffi::ShapeView shape, DLDataType dtype, DLDevice device,
                    ffi::Optional<MemoryScope> mem_scope = std::nullopt) final {
    TVM_FFI_ICHECK_EQ(device.device_type, DLDeviceType::kDLCUDA)
        << "nvshmem can only allocate cuda device memory space.";
    TVM_FFI_ICHECK(dtype.code == kDLInt || dtype.code == kDLUInt || dtype.code == kDLFloat ||
                   dtype.code == kDLBfloat)
        << "nvshmem can only allocate tensor with int, unsigned int, float, or bfloat data types.";
    static const auto scope = MemoryScope::Get("nvshmem");
    TVM_FFI_CHECK(!mem_scope.has_value() || mem_scope.value() == scope, ValueError)
        << "NVSHMEM memory requires the nvshmem scope";
    return TensorMemoryPool::Empty(shape, dtype, device);
  }

 private:
  ffi::Tensor AllocBacking(size_t size, size_t alignment, DLDevice device) final {
    struct Backing {
      size_t size;
      size_t alignment;
      void AllocData(DLTensor* tensor) { tensor->data = nvshmem_align(alignment, size); }
      void FreeData(DLTensor* tensor) { nvshmem_free(tensor->data); }
    };
    int64_t extent = static_cast<int64_t>(size);
    return ffi::Tensor::FromNDAlloc(Backing{size, alignment}, ffi::ShapeView(&extent, 1),
                                    DLDataType{kDLUInt, 8, 1}, device);
  }
};

ffi::Tensor NVSHMEMEmpty(ffi::Shape shape, DLDataType dtype, ffi::Optional<Device> device) {
  Device resolved_device =
      device.has_value() ? device.value() : UseDefaultDeviceIfNone(std::nullopt);
  return NVSHMEMPool::Global()->Empty(shape, dtype, resolved_device);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  MemoryScope::Global();
  refl::EnumDef<MemoryScopeObj>("nvshmem").set_attr(
      attr::kMemoryAllocProperty, static_cast<int>(MemoryAllocProperty::kSeparateAllocations));
  refl::GlobalDef().def("runtime.disco.nvshmem.empty", NVSHMEMEmpty);
}

void NVSHMEMFinalize() {
  NVSHMEMPool::Global()->Clear();
  nvshmem_finalize();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("runtime.disco.nvshmem.finalize_nvshmem", NVSHMEMFinalize);
}

}  // namespace runtime
}  // namespace tvm
