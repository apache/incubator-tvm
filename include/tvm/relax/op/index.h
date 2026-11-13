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
 * \file tvm/relax/op/index.h
 * \brief Attributes for indexing operators.
 */
#ifndef TVM_RELAX_OP_INDEX_H_
#define TVM_RELAX_OP_INDEX_H_

#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {

/*! \brief Attributes used in take operator */
struct TakeAttrs : public AttrsObj {
  ffi::Optional<int64_t> axis;
  ffi::String mode;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.TakeAttrs", TakeAttrs, AttrsObj);
};  // struct TakeAttrs

/*! \brief Attributes used in take_backward operator */
struct TakeBackwardAttrs : public AttrsObj {
  ffi::Optional<int64_t> axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.TakeBackwardAttrs", TakeBackwardAttrs, AttrsObj);
};

/*! \brief Attributes used in strided_slice operator */
struct StridedSliceAttrs : public AttrsObj {
  bool assume_inbound;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.StridedSliceAttrs", StridedSliceAttrs, AttrsObj);
};  // struct StridedSliceAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_INDEX_H_
