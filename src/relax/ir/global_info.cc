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

/*!
 * \file src/relax/ir/global_info.cc
 * \brief Module global info.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/global_info.h>
namespace tvm {
namespace relax {

TVM_FFI_STATIC_INIT_BLOCK() {
  VDeviceObj::RegisterReflection();
  DummyGlobalInfoObj::RegisterReflection();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  auto make_dummy = []() {
    return DummyGlobalInfo(ffi::UnsafeInit{}, ffi::make_object<DummyGlobalInfoObj>());
  };
  refl::GlobalDef().def("relax.DummyGlobalInfo", make_dummy);
}

VDevice::VDevice(ffi::Optional<Target> tgt, int dev_id, MemoryScope mem_scope)
    : GlobalInfo(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<VDeviceObj> n = ffi::make_object<VDeviceObj>();
  n->target = std::move(tgt);
  n->vdevice_id = std::move(dev_id);
  n->memory_scope = std::move(mem_scope);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  auto make_vdevice = [](ffi::Optional<Target> tgt, int dev_id, MemoryScope mem_scope) {
    return VDevice(tgt, dev_id, mem_scope);
  };
  refl::GlobalDef().def("relax.VDevice", make_vdevice);
}
}  // namespace relax
}  // namespace tvm
