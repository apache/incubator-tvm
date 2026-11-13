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
 * \file tvm/relax/op/nn.h
 * \brief Attributes for neural network operators.
 */
#ifndef TVM_RELAX_OP_NN_H_
#define TVM_RELAX_OP_NN_H_

#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {

/*! \brief Attributes used in Conv1d operator */
struct Conv1DAttrs : public AttrsObj {
  ffi::Array<int64_t> strides;
  ffi::Array<int64_t> padding;
  ffi::Array<int64_t> dilation;
  int groups;
  ffi::String data_layout;
  ffi::String kernel_layout;
  ffi::String out_layout;
  ffi::Optional<DLDataType> out_dtype;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.Conv1DAttrs", Conv1DAttrs, AttrsObj);
};  // struct Conv1dAttrs

/*! \brief Attributes used in Conv2d operator */
struct Conv2DAttrs : public AttrsObj {
  ffi::Array<int64_t> strides;
  ffi::Array<int64_t> padding;
  ffi::Array<int64_t> dilation;
  int groups;
  ffi::String data_layout;
  ffi::String kernel_layout;
  ffi::String out_layout;
  ffi::Optional<DLDataType> out_dtype;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.Conv2DAttrs", Conv2DAttrs, AttrsObj);
};  // struct Conv2dAttrs

/*! \brief Attributes used in Conv3d operator */
struct Conv3DAttrs : public AttrsObj {
  ffi::Array<int64_t> strides;
  ffi::Array<int64_t> padding;
  ffi::Array<int64_t> dilation;
  int groups;
  ffi::String data_layout;
  ffi::String kernel_layout;
  ffi::String out_layout;
  ffi::Optional<DLDataType> out_dtype;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.Conv3DAttrs", Conv3DAttrs, AttrsObj);
};  // struct Conv3dAttrs

/*! \brief Attributes used in Conv1DTranspose operator */
struct Conv1DTransposeAttrs : public AttrsObj {
  ffi::Array<int64_t> strides;
  ffi::Array<int64_t> padding;
  ffi::Array<int64_t> output_padding;
  ffi::Array<int64_t> dilation;
  int groups;
  ffi::String data_layout;
  ffi::String kernel_layout;
  ffi::String out_layout;
  ffi::Optional<DLDataType> out_dtype;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.Conv1DTransposeAttrs", Conv1DTransposeAttrs,
                                    AttrsObj);
};  // struct Conv1DTransposeAttrs

/*! \brief Attributes used in Conv2d operator */
struct Conv2DTransposeAttrs : public AttrsObj {
  ffi::Array<int64_t> strides;
  ffi::Array<int64_t> padding;
  ffi::Array<int64_t> output_padding;
  ffi::Array<int64_t> dilation;
  int groups;
  ffi::String data_layout;
  ffi::String kernel_layout;
  ffi::String out_layout;
  ffi::Optional<DLDataType> out_dtype;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.Conv2DTransposeAttrs", Conv2DTransposeAttrs,
                                    AttrsObj);
};  // struct Conv2DTransposeAttrs

/*! \brief Attributes used in Conv3dTranspose operator */
struct Conv3DTransposeAttrs : public AttrsObj {
  ffi::Array<int64_t> strides;
  ffi::Array<int64_t> padding;
  ffi::Array<int64_t> output_padding;
  ffi::Array<int64_t> dilation;
  int groups;
  ffi::String data_layout;
  ffi::String kernel_layout;
  ffi::String out_layout;
  ffi::Optional<DLDataType> out_dtype;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.Conv3DTransposeAttrs", Conv3DTransposeAttrs,
                                    AttrsObj);
};  // struct Conv3DTransposeAttrs

/*! \brief Attributes used in max_pool1d and avg_pool1d operator */
struct Pool1DAttrs : public AttrsObj {
  ffi::Array<int64_t> pool_size;
  ffi::Array<int64_t> strides;
  ffi::Array<int64_t> padding;
  ffi::Array<int64_t> dilation;
  bool ceil_mode;
  bool count_include_pad;
  ffi::String layout;
  ffi::String out_layout;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.Pool1DAttrs", Pool1DAttrs, AttrsObj);
};  // struct Pool1dAttrs

/*! \brief Attributes used in max_pool2d and avg_pool2d operator */
struct Pool2DAttrs : public AttrsObj {
  ffi::Array<int64_t> pool_size;
  ffi::Array<int64_t> strides;
  ffi::Array<int64_t> padding;
  ffi::Array<int64_t> dilation;
  bool ceil_mode;
  bool count_include_pad;
  ffi::String layout;
  ffi::String out_layout;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.Pool2DAttrs", Pool2DAttrs, AttrsObj);
};  // struct Pool2dAttrs

/*! \brief Attributes used in max_pool3d and avg_pool3d operator */
struct Pool3DAttrs : public AttrsObj {
  ffi::Array<int64_t> pool_size;
  ffi::Array<int64_t> strides;
  ffi::Array<int64_t> padding;
  ffi::Array<int64_t> dilation;
  bool ceil_mode;
  bool count_include_pad;
  ffi::String layout;
  ffi::String out_layout;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.Pool3DAttrs", Pool3DAttrs, AttrsObj);
};  // struct Pool3dAttrs

/*! \brief Attributes for 1d adaptive pool operator */
struct AdaptivePool1DAttrs : public AttrsObj {
  ffi::Optional<ffi::Array<int64_t>> output_size;
  ffi::String layout;
  ffi::String out_layout;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.AdaptivePool1DAttrs", AdaptivePool1DAttrs,
                                    AttrsObj);
};  // struct AdaptivePool1DAttrs

/*! \brief Attributes for 2d adaptive pool operator */
struct AdaptivePool2DAttrs : public AttrsObj {
  ffi::Optional<ffi::Array<int64_t>> output_size;
  ffi::String layout;
  ffi::String out_layout;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.AdaptivePool2DAttrs", AdaptivePool2DAttrs,
                                    AttrsObj);
};  // struct AdaptivePool2DAttrs

/*! \brief Attributes for 3d adaptive pool operator */
struct AdaptivePool3DAttrs : public AttrsObj {
  ffi::Optional<ffi::Array<int64_t>> output_size;
  ffi::String layout;
  ffi::String out_layout;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.AdaptivePool3DAttrs", AdaptivePool3DAttrs,
                                    AttrsObj);
};  // struct AdaptivePool3DAttrs

/*! \brief Attributes used in softmax operators */
struct SoftmaxAttrs : public AttrsObj {
  int axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.SoftmaxAttrs", SoftmaxAttrs, AttrsObj);
};

/*! \brief Attributes used in softmax operators */
struct LeakyReluAttrs : public AttrsObj {
  double alpha;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.LeakyReluAttrs", LeakyReluAttrs, AttrsObj);
};

/*! \brief Attributes used in softplus operators */
struct SoftplusAttrs : public AttrsObj {
  double beta;
  double threshold;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.SoftplusAttrs", SoftplusAttrs, AttrsObj);
};

/*! \brief Attributes used in PReLU operator */
struct PReluAttrs : public AttrsObj {
  int axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.PReluAttrs", PReluAttrs, AttrsObj);
};

/*! \brief Attributes used in batch_norm operator */
struct BatchNormAttrs : public AttrsObj {
  int axis;
  double epsilon;
  bool center;
  bool scale;
  double momentum;
  bool training;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.BatchNormAttrs", BatchNormAttrs, AttrsObj);
};  // struct BatchNormAttrs

/*! \brief Attributes used in layer_norm operator */
struct LayerNormAttrs : public AttrsObj {
  ffi::Array<int64_t> axes;
  double epsilon;
  bool center;
  bool scale;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.LayerNormAttrs", LayerNormAttrs, AttrsObj);
};  // struct LayerNormAttrs

/*! \brief Attributes used in group_norm operator */
struct GroupNormAttrs : public AttrsObj {
  int num_groups;
  int channel_axis;
  ffi::Array<int64_t> axes;
  double epsilon;
  bool center;
  bool scale;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.GroupNormAttrs", GroupNormAttrs, AttrsObj);
};  // struct GroupNormAttrs

/*! \brief Attributes used in instance_norm operator */
struct InstanceNormAttrs : public AttrsObj {
  int channel_axis;
  ffi::Array<int64_t> axes;
  double epsilon;
  bool center;
  bool scale;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.InstanceNormAttrs", InstanceNormAttrs, AttrsObj);
};  // struct InstanceNormAttrs

/*! \brief Attributes used in rms_norm operator */
struct RMSNormAttrs : public AttrsObj {
  ffi::Array<int64_t> axes;
  double epsilon;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.RMSNormAttrs", RMSNormAttrs, AttrsObj);
};  // struct RMSNormAttrs

/*! \brief Attributes used in nll_loss operator */
struct NLLLossAttrs : public AttrsObj {
  ffi::String reduction;
  int ignore_index;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.NLLLossAttrs", NLLLossAttrs, AttrsObj);
};  // struct NLLLossAttrs

/*! \brief Attributes used in dropout operator */
struct DropoutAttrs : public AttrsObj {
  double rate;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.DropoutAttrs", DropoutAttrs, AttrsObj);
};  // struct DropoutAttrs

/*! \brief Attributes used in Attention operator */
struct AttentionAttrs : public AttrsObj {
  ffi::Optional<FloatImm> scale;
  ffi::Optional<ffi::String> causal_mask;
  ffi::Optional<IntImm> window_size;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.AttentionAttrs", AttentionAttrs, AttrsObj);
};  // struct AttentionAttrs

/*! \brief Attributes used for the padding operator */
struct PadAttrs : public AttrsObj {
  ffi::Array<int64_t> pad_width;
  double pad_value = 0.0;
  tvm::ffi::String pad_mode;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.PadAttrs", PadAttrs, AttrsObj);
};

/*! \brief Attributes used for the pixel shuffle operator */
struct PixelShuffleAttrs : public AttrsObj {
  int upscale_factor;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.PixelShuffleAttrs", PixelShuffleAttrs, AttrsObj);
};

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_NN_H_
