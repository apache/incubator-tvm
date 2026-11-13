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
 * \file tvm/relax/op/vision.h
 * \brief Auxiliary attributes for vision operators.
 */
#ifndef TVM_RELAX_OP_VISION_H_
#define TVM_RELAX_OP_VISION_H_

#include <tvm/ffi/string.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/type.h>
#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {

/*! \brief Attributes used in AllClassNonMaximumSuppression operator */
struct AllClassNonMaximumSuppressionAttrs : public AttrsObj {
  ffi::String output_format;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.AllClassNonMaximumSuppressionAttrs",
                                    AllClassNonMaximumSuppressionAttrs, AttrsObj);
};  // struct AllClassNonMaximumSuppressionAttrs

/*! \brief Attributes used in ROIAlign operator */
struct ROIAlignAttrs : public AttrsObj {
  ffi::Array<int64_t> pooled_size;
  double spatial_scale;
  int sample_ratio;
  bool aligned;
  ffi::String layout;
  ffi::String mode;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ROIAlignAttrs", ROIAlignAttrs, AttrsObj);
};  // struct ROIAlignAttrs

/*! \brief Attributes used in ROIPool operator */
struct ROIPoolAttrs : public AttrsObj {
  ffi::Array<int64_t> pooled_size;
  double spatial_scale;
  ffi::String layout;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ROIPoolAttrs", ROIPoolAttrs, AttrsObj);
};  // struct ROIPoolAttrs

/*! \brief Attributes used in GetValidCounts operator */
struct GetValidCountsAttrs : public AttrsObj {
  double score_threshold;
  int id_index;
  int score_index;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.GetValidCountsAttrs", GetValidCountsAttrs,
                                    AttrsObj);
};  // struct GetValidCountsAttrs

/*! \brief Attributes used in NonMaximumSuppression operator */
struct NonMaximumSuppressionAttrs : public AttrsObj {
  int max_output_size;
  double iou_threshold;
  bool force_suppress;
  int top_k;
  int coord_start;
  int score_index;
  int id_index;
  bool return_indices;
  bool invalid_to_bottom;
  double soft_nms_sigma;
  double score_threshold;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.NonMaximumSuppressionAttrs",
                                    NonMaximumSuppressionAttrs, AttrsObj);
};  // struct NonMaximumSuppressionAttrs

/*! \brief Attributes for multibox_transform_loc (SSD / TFLite-style box decode). */
struct MultiboxTransformLocAttrs : public AttrsObj {
  bool clip;
  double threshold;
  ffi::Array<double> variances;
  bool keep_background;
  bool apply_softmax;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.MultiboxTransformLocAttrs",
                                    MultiboxTransformLocAttrs, AttrsObj);
};  // struct MultiboxTransformLocAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_VISION_H_
