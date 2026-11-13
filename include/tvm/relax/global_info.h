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
 * \file tvm/relax/global_info.h
 * \brief Global information used by Relax.
 */
#ifndef TVM_RELAX_GLOBAL_INFO_H_
#define TVM_RELAX_GLOBAL_INFO_H_

#include <tvm/base.h>
#include <tvm/ir/global_info.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/target/target.h>

namespace tvm {
namespace relax {

/*!
 * \brief Abstract label for an area of memory.
 */
using runtime::MemoryScope;

/*!
 * \brief A global info subclass for virtual devices.
 */
class VDeviceObj : public GlobalInfoObj {
 public:
  /*! \brief The \p Target describing how to compile for the virtual device. */
  ffi::Optional<Target> target;
  /*! \brief The device identifier for the virtual device. This enables us to
   * differentiate between distinct devices with same Target, such as multiple GPUs.
   */
  int vdevice_id;
  MemoryScope memory_scope;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<VDeviceObj>()
        .def_ro("target", &VDeviceObj::target)
        .def_ro("vdevice_id", &VDeviceObj::vdevice_id)
        .def_ro("memory_scope", &VDeviceObj::memory_scope);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.VDevice", VDeviceObj, GlobalInfoObj);
};

/*!
 * \brief Managed reference to VDeviceObj.
 * \sa VDeviceObj
 */
class VDevice : public GlobalInfo {
 public:
  TVM_DLL explicit VDevice(ffi::Optional<Target> tgt, int dev_id, MemoryScope mem_scope);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(VDevice, GlobalInfo, VDeviceObj);
};

/*!
 * \brief A dummy global info sub-class for testing purpose.
 */
class DummyGlobalInfoObj : public GlobalInfoObj {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DummyGlobalInfoObj>();
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.DummyGlobalInfo", DummyGlobalInfoObj, GlobalInfoObj);
};

/*!
 * \brief Managed reference to DummyGlobalInfoObj.
 * \sa DummyGlobalInfoObj
 */
class DummyGlobalInfo : public GlobalInfo {
 public:
  explicit DummyGlobalInfo(ffi::UnsafeInit tag, ffi::ObjectPtr<DummyGlobalInfoObj> n)
      : GlobalInfo(tag) {
    data_ = std::move(n);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(DummyGlobalInfo, GlobalInfo, DummyGlobalInfoObj);
};

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_GLOBAL_INFO_H_
