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
 * \file tvm/relax/op/op.h
 * \brief Attributes for relax specific operators.
 */
#ifndef TVM_RELAX_OP_OP_H_
#define TVM_RELAX_OP_OP_H_

#include <tvm/relax/expr.h>
#include <tvm/relax/global_info.h>

namespace tvm {
namespace relax {

/*! \brief Attributes used in call_tir_with_grad */
struct CallTIRWithGradAttrs : public AttrsObj {
  ffi::String te_grad_name;
  ffi::Map<ffi::String, Any> te_grad_kwargs;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.CallTIRWithGradAttrs", CallTIRWithGradAttrs,
                                    AttrsObj);
};  // struct CallTIRAttrs

/*! \brief Attributes used in call_tir_inplace */
struct CallTIRInplaceAttrs : public AttrsObj {
  /*!
   * \brief Indices that describe which input corresponds to which output.
   *
   * If the `i`th member has the value `k` >= 0, then that means that input `k` should be used to
   * store the `i`th output. If an element has the value -1, that means a new tensor should be
   * allocated for that output.
   */
  ffi::Array<int64_t> inplace_indices;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.CallTIRInplaceAttrs", CallTIRInplaceAttrs,
                                    AttrsObj);
};  // struct CallTIRInplaceAttrs

/*! \brief Attributes used in call_inplace_packed */
struct CallInplacePackedAttrs : public AttrsObj {
  /*!
   * \brief Indices that describe which input corresponds to which output.
   *
   * If the `i`th member has the value `k` >= 0, then that means that input `k` should be used to
   * store the `i`th output. If an element has the value -1, that means the output will be newly
   * allocated.
   */
  ffi::Array<int64_t> inplace_indices;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.CallInplacePackedAttrs", CallInplacePackedAttrs,
                                    AttrsObj);
};  // struct CallInplacePackedAttrs

/*! \brief Attributes used in to_vdevice */
struct ToVDeviceAttrs : public AttrsObj {
  explicit ToVDeviceAttrs(VDevice value) : dst_vdevice(std::move(value)) {}
  explicit ToVDeviceAttrs(ffi::UnsafeInit) : dst_vdevice(ffi::UnsafeInit{}) {}
  VDevice dst_vdevice;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ToVDeviceAttrs", ToVDeviceAttrs, AttrsObj);
};  // struct ToVDeviceAttrs

/*! \brief Attributes used in hint_on_device */
struct HintOnDeviceAttrs : public AttrsObj {
  int32_t device_type;
  int32_t index;
  MemoryScope memory_scope;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.HintOnDeviceAttrs", HintOnDeviceAttrs, AttrsObj);
};  // struct HintOnDeviceAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_OP_H_
