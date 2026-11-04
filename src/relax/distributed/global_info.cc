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
#include <tvm/relax/distributed/global_info.h>

namespace tvm {
namespace relax {
namespace distributed {

TVM_FFI_STATIC_INIT_BLOCK() { DeviceMeshObj::RegisterReflection(); }

DeviceMesh::DeviceMesh(ffi::Shape shape, ffi::Array<int64_t> device_ids)
    : GlobalInfo(ffi::UnsafeInit{}) {
  int prod = 1;
  for (int i = 0; i < static_cast<int>(shape.size()); i++) {
    prod *= shape[i];
  }
  ffi::ObjectPtr<DeviceMeshObj> n = ffi::make_object<DeviceMeshObj>();
  TVM_FFI_ICHECK_EQ(prod, static_cast<int>(device_ids.size()))
      << "The number of device ids must match the product of the shape";
  n->shape = std::move(shape);
  n->device_ids = std::move(device_ids);
  data_ = std::move(n);
}

DeviceMesh::DeviceMesh(ffi::Shape shape, Range device_range) : GlobalInfo(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<DeviceMeshObj> n = ffi::make_object<DeviceMeshObj>();
  ffi::Array<int64_t> device_ids;
  int range_start = device_range->begin.as<IntImmObj>()->value.as<int>().value();
  int range_extent = device_range->extent.as<IntImmObj>()->value.as<int>().value();
  for (int i = range_start; i < range_start + range_extent; i++) {
    device_ids.push_back(i);
  }
  int prod = 1;
  for (int i = 0; i < static_cast<int>(shape.size()); i++) {
    prod *= shape[i];
  }
  TVM_FFI_ICHECK_EQ(prod, static_cast<int>(device_ids.size()))
      << "The number of device ids must match the product of the shape";
  n->device_ids = std::move(device_ids);
  n->shape = std::move(shape);
  n->device_range = std::move(device_range);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "relax.distributed.DeviceMesh",
      [](ffi::Shape shape, ffi::Array<int64_t> device_ids, ffi::Optional<Range> device_range) {
        if (device_range.has_value())
          return DeviceMesh(shape, device_range.value());
        else
          return DeviceMesh(shape, device_ids);
      });
}

}  // namespace distributed
}  // namespace relax
}  // namespace tvm
