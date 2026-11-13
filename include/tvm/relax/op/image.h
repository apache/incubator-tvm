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
 * \file tvm/relax/op/image.h
 * \brief Attributes for image operators.
 */
#ifndef TVM_RELAX_OP_IMAGE_H_
#define TVM_RELAX_OP_IMAGE_H_

#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {

/*! \brief Attributes used in image resize2d operator */
struct Resize2DAttrs : public AttrsObj {
  ffi::Array<FloatImm> roi;
  ffi::String layout;
  ffi::String method;
  ffi::String coordinate_transformation_mode;
  ffi::String rounding_method;
  double cubic_alpha;
  int cubic_exclude;
  double extrapolation_value;
  ffi::Optional<DLDataType> out_dtype;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.Resize2DAttrs", Resize2DAttrs, AttrsObj);
};  // struct Resize2dAttrs

/*! \brief Attributes used in image resize3d operator */
struct Resize3DAttrs : public AttrsObj {
  ffi::Array<FloatImm> roi;
  ffi::String layout;
  ffi::String method;
  ffi::String coordinate_transformation_mode;
  ffi::String rounding_method;
  double cubic_alpha;
  int cubic_exclude;
  double extrapolation_value;
  ffi::Optional<DLDataType> out_dtype;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.Resize3DAttrs", Resize3DAttrs, AttrsObj);
};  // struct Resize3DAttrs

/*! \brief Attributes used in image grid_sample operator */
struct GridSampleAttrs : public AttrsObj {
  ffi::String method;
  ffi::String layout;
  ffi::String padding_mode;
  bool align_corners;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.GridSampleAttrs", GridSampleAttrs, AttrsObj);
};  // struct GridSampleAttrs

/*! \brief Attributes used in image affine_grid operator */
struct AffineGridAttrs : public AttrsObj {
  bool align_corners;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.AffineGridAttrs", AffineGridAttrs, AttrsObj);
};  // struct AffineGridAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_IMAGE_H_
