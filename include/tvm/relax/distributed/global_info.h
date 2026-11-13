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
 * \file tvm/relax/distributed/global_info.h
 * \brief Data structure for distributed inference
 */

#ifndef TVM_RELAX_DISTRIBUTED_GLOBAL_INFO_H_
#define TVM_RELAX_DISTRIBUTED_GLOBAL_INFO_H_

#include <tvm/base.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/module.h>
namespace tvm {
namespace relax {
namespace distributed {
/*
 * \brief Device mesh express a view of topology of devices, represented by an n-d matrix of
 * device ids
 */
class DeviceMeshObj : public GlobalInfoObj {
 public:
  /*! \brief logical shape of the mesh*/
  ffi::Shape shape;

  /*! \brief device ids in the mesh*/
  ffi::Array<int64_t> device_ids;

  /*! \brief Optionally use range to represent device_ids*/
  ffi::Optional<Range> device_range;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DeviceMeshObj>()
        .def_ro("shape", &DeviceMeshObj::shape)
        .def_ro("device_ids", &DeviceMeshObj::device_ids)
        .def_ro("device_range", &DeviceMeshObj::device_range);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.distributed.DeviceMesh", DeviceMeshObj, GlobalInfoObj);
};

/*!
 * \brief Managed reference to a DeviceMesh.
 * \sa DeviceMeshObj
 */
class DeviceMesh : public GlobalInfo {
 public:
  TVM_DLL DeviceMesh(ffi::Shape shape, ffi::Array<int64_t> device_ids);
  TVM_DLL DeviceMesh(ffi::Shape shape, Range device_range);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(DeviceMesh, GlobalInfo, DeviceMeshObj);
};

}  // namespace distributed
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_DISTRIBUTED_GLOBAL_INFO_H_
