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
 * \file src/relax/op/nn/convolution.cc
 * \brief Convolution operators
 */

#include "convolution.h"

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/op_attr.h>

#include <vector>

namespace tvm {
namespace relax {
using namespace tvm::prim;

void Conv1DAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<Conv1DAttrs>()
      .def_ro("strides", &Conv1DAttrs::strides, "Specifies the strides of the convolution.",
              refl::DefaultValue(ffi::Array<int64_t>{1}))
      .def_ro("padding", &Conv1DAttrs::padding,
              "If padding is non-zero, then the input is implicitly zero-padded"
              "Padding support both symmetric and asymmetric as"
              "one int : same padding used on both sides"
              "two int : padding width in the order of (left, right)",
              refl::DefaultValue(ffi::Array<int64_t>{0, 0}))
      .def_ro("dilation", &Conv1DAttrs::dilation,
              "Specifies the dilation rate to use for dilated convolution.",
              refl::DefaultValue(ffi::Array<int64_t>{1}))
      .def_ro("groups", &Conv1DAttrs::groups,
              "Number of groups to split the input into for grouped convolution. The number of "
              "input and "
              "output channels should be divisible by the number of groups.",
              refl::DefaultValue(1))
      .def_ro("data_layout", &Conv1DAttrs::data_layout,
              "Dimension ordering of input data. Can be 'NCW', 'NWC', etc."
              "'N', 'C', 'W' stands for batch, channel, width"
              "dimensions respectively. Convolution is applied on the 'W' dimensions.",
              refl::DefaultValue(ffi::String("NCW")))
      .def_ro("kernel_layout", &Conv1DAttrs::kernel_layout,
              "Dimension ordering of weight. Can be 'OIW', 'IOW', etc."
              "'O', 'I', 'W' stands for num_filter, input_channel, and width"
              "dimensions respectively.",
              refl::DefaultValue(ffi::String("OIW")))
      .def_ro("out_layout", &Conv1DAttrs::out_layout,
              "Dimension ordering of output. Can be 'NCW', 'NWC', etc."
              "'N', 'C', 'W' stands for batch, channel, and width"
              "dimensions respectively. Default to be same as input layout.")
      .def_ro("out_dtype", &Conv1DAttrs::out_dtype,
              "Output data type, set to explicit type under mixed precision setting",
              refl::DefaultValue(ffi::Optional<DLDataType>{}));
}

void Conv2DAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<Conv2DAttrs>()
      .def_ro("strides", &Conv2DAttrs::strides, "Specifies the strides of the convolution.",
              refl::DefaultValue(ffi::Array<int64_t>{1, 1}))
      .def_ro("padding", &Conv2DAttrs::padding,
              "If padding is non-zero, then the input is implicitly zero-padded"
              "Padding support both symmetric and asymmetric as"
              "one int : same padding used on all sides"
              "two int : bottom, right will use same padding as top, left"
              "four int : padding width in the order of (top, left, bottom, right)",
              refl::DefaultValue(ffi::Array<int64_t>{0, 0, 0, 0}))
      .def_ro("dilation", &Conv2DAttrs::dilation,
              "Specifies the dilation rate to use for dilated convolution.",
              refl::DefaultValue(ffi::Array<int64_t>{1, 1}))
      .def_ro("groups", &Conv2DAttrs::groups,
              "Number of groups to split the input into for grouped convolution. The number of "
              "input and "
              "output channels should be divisible by the number of groups.",
              refl::DefaultValue(1))
      .def_ro("data_layout", &Conv2DAttrs::data_layout,
              "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
              "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
              "dimensions respectively. Convolution is applied on the 'H' and"
              "'W' dimensions.",
              refl::DefaultValue(ffi::String("NCHW")))
      .def_ro("kernel_layout", &Conv2DAttrs::kernel_layout,
              "Dimension ordering of weight. Can be 'OIHW', 'OIHW16o16i', etc."
              "'O', 'I', 'H', 'W' stands for num_filter, input_channel, height, and width"
              "dimensions respectively.",
              refl::DefaultValue(ffi::String("OIHW")))
      .def_ro("out_layout", &Conv2DAttrs::out_layout,
              "Dimension ordering of output. Can be 'NCHW', 'NHWC', etc."
              "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
              "dimensions respectively. Default to be same as input layout.")
      .def_ro("out_dtype", &Conv2DAttrs::out_dtype,
              "Output data type, set to explicit type under mixed precision setting",
              refl::DefaultValue(ffi::Optional<DLDataType>{}));
}

void Conv3DAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<Conv3DAttrs>()
      .def_ro("strides", &Conv3DAttrs::strides, "Specifies the strides of the convolution.",
              refl::DefaultValue(ffi::Array<int64_t>{1, 1, 1}))
      .def_ro("padding", &Conv3DAttrs::padding,
              "If padding is non-zero, then the input is implicitly zero-padded"
              "Padding support both symmetric and asymmetric as"
              "one int : same padding used on all sides"
              "two int : bottom, right will use same padding as top, left"
              "four int : padding width in the order of (forward, back, top, left, bottom, right)",
              refl::DefaultValue(ffi::Array<int64_t>{0, 0, 0, 0, 0, 0}))
      .def_ro("dilation", &Conv3DAttrs::dilation,
              "Specifies the dilation rate to use for dilated convolution.",
              refl::DefaultValue(ffi::Array<int64_t>{1, 1, 1}))
      .def_ro("groups", &Conv3DAttrs::groups,
              "Number of groups to split the input into for grouped convolution. The number of "
              "input and "
              "output channels should be divisible by the number of groups.",
              refl::DefaultValue(1))
      .def_ro("data_layout", &Conv3DAttrs::data_layout,
              "Dimension ordering of input data. Can be 'NCDHW', 'NDHWC', etc."
              "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
              "dimensions respectively. Convolution is applied on the 'D', 'H', and"
              "'W' dimensions.",
              refl::DefaultValue(ffi::String("NCDHW")))
      .def_ro(
          "kernel_layout", &Conv3DAttrs::kernel_layout,
          "Dimension ordering of weight. Can be 'OIDHW', 'OIDHW16o16i', etc."
          "'O', 'I', 'D', 'H', 'W' stands for num_filter, input_channel, depth, height, and width"
          "dimensions respectively.",
          refl::DefaultValue(ffi::String("OIDHW")))
      .def_ro("out_layout", &Conv3DAttrs::out_layout,
              "Dimension ordering of output. Can be 'NCDHW', 'NDHWC', etc."
              "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
              "dimensions respectively. Default to be same as input layout.")
      .def_ro("out_dtype", &Conv3DAttrs::out_dtype,
              "Output data type, set to explicit type under mixed precision setting",
              refl::DefaultValue(ffi::Optional<DLDataType>{}));
}

void Conv1DTransposeAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<Conv1DTransposeAttrs>()
      .def_ro("strides", &Conv1DTransposeAttrs::strides,
              "Specifies the strides of the convolution.",
              refl::DefaultValue(ffi::Array<int64_t>{1}))
      .def_ro("padding", &Conv1DTransposeAttrs::padding,
              "If padding is non-zero, then the input is implicitly zero-padded"
              "Padding support both symmetric and asymmetric as"
              "one int : same padding used on both sides"
              "two int : padding width in the order of (left, right)",
              refl::DefaultValue(ffi::Array<int64_t>{0, 0}))
      .def_ro("output_padding", &Conv1DTransposeAttrs::output_padding,
              "Used to disambiguate the output shape.", refl::DefaultValue(ffi::Array<int64_t>{0}))
      .def_ro("dilation", &Conv1DTransposeAttrs::dilation,
              "Specifies the dilation rate to use for dilated convolution.",
              refl::DefaultValue(ffi::Array<int64_t>{1}))
      .def_ro("groups", &Conv1DTransposeAttrs::groups,
              "Number of groups to split the input into for grouped convolution. The number of "
              "input and "
              "output channels should be divisible by the number of groups.",
              refl::DefaultValue(1))
      .def_ro("data_layout", &Conv1DTransposeAttrs::data_layout,
              "Dimension ordering of input data. Can be 'NCW', 'NWC', etc."
              "'N', 'C', 'W' stands for batch, channel, width"
              "dimensions respectively. Convolution is applied on the 'W' dimensions.",
              refl::DefaultValue(ffi::String("NCW")))
      .def_ro("kernel_layout", &Conv1DTransposeAttrs::kernel_layout,
              "Dimension ordering of weight. Can be 'OIW', 'IOW', etc."
              "'O', 'I', 'W' stands for num_filter, input_channel, and width"
              "dimensions respectively.",
              refl::DefaultValue(ffi::String("IOW")))
      .def_ro("out_layout", &Conv1DTransposeAttrs::out_layout,
              "Dimension ordering of output. Can be 'NCW', 'NWC', etc."
              "'N', 'C', 'W' stands for batch, channel, and width"
              "dimensions respectively. Default to be same as input layout.")
      .def_ro("out_dtype", &Conv1DTransposeAttrs::out_dtype,
              "Output data type, set to explicit type under mixed precision setting",
              refl::DefaultValue(ffi::Optional<DLDataType>{}));
}

void Conv2DTransposeAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<Conv2DTransposeAttrs>()
      .def_ro("strides", &Conv2DTransposeAttrs::strides,
              "Specifies the strides of the convolution.",
              refl::DefaultValue(ffi::Array<int64_t>{1, 1}))
      .def_ro("padding", &Conv2DTransposeAttrs::padding,
              "If padding is non-zero, then the input is implicitly zero-padded"
              "Padding support both symmetric and asymmetric as"
              "one int : same padding used on all sides"
              "two int : bottom, right will use same padding as top, left"
              "four int : padding width in the order of (top, left, bottom, right)",
              refl::DefaultValue(ffi::Array<int64_t>{0, 0, 0, 0}))
      .def_ro("output_padding", &Conv2DTransposeAttrs::output_padding,
              "Used to disambiguate the output shape.",
              refl::DefaultValue(ffi::Array<int64_t>{0, 0}))
      .def_ro("dilation", &Conv2DTransposeAttrs::dilation,
              "Specifies the dilation rate to use for dilated convolution.",
              refl::DefaultValue(ffi::Array<int64_t>{1, 1}))
      .def_ro("groups", &Conv2DTransposeAttrs::groups,
              "Number of groups to split the input into for grouped convolution. The number of "
              "input and "
              "output channels should be divisible by the number of groups.",
              refl::DefaultValue(1))
      .def_ro("data_layout", &Conv2DTransposeAttrs::data_layout,
              "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
              "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
              "dimensions respectively. Convolution is applied on the 'H' and"
              "'W' dimensions.",
              refl::DefaultValue(ffi::String("NCHW")))
      .def_ro("kernel_layout", &Conv2DTransposeAttrs::kernel_layout,
              "Dimension ordering of weight. Can be 'OIHW', 'OIHW16o16i', etc."
              "'O', 'I', 'H', 'W' stands for num_filter, input_channel, height, and width"
              "dimensions respectively.",
              refl::DefaultValue(ffi::String("IOHW")))
      .def_ro("out_layout", &Conv2DTransposeAttrs::out_layout,
              "Dimension ordering of output. Can be 'NCHW', 'NHWC', etc."
              "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
              "dimensions respectively. Default to be same as input layout.")
      .def_ro("out_dtype", &Conv2DTransposeAttrs::out_dtype,
              "Output data type, set to explicit type under mixed precision setting",
              refl::DefaultValue(ffi::Optional<DLDataType>{}));
}

void Conv3DTransposeAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<Conv3DTransposeAttrs>()
      .def_ro("strides", &Conv3DTransposeAttrs::strides,
              "Specifies the strides of the convolution.",
              refl::DefaultValue(ffi::Array<int64_t>{1, 1, 1}))
      .def_ro("padding", &Conv3DTransposeAttrs::padding,
              "If padding is non-zero, then the input is implicitly zero-padded"
              "Padding support both symmetric and asymmetric as"
              "one int : same padding used on all sides"
              "three int : back/bottom/right will use same padding as front/top/left"
              "six int : padding width in the order of (front, top, left, back, bottom, right)",
              refl::DefaultValue(ffi::Array<int64_t>{0, 0, 0, 0, 0, 0}))
      .def_ro("output_padding", &Conv3DTransposeAttrs::output_padding,
              "Used to disambiguate the output shape.",
              refl::DefaultValue(ffi::Array<int64_t>{0, 0, 0}))
      .def_ro("dilation", &Conv3DTransposeAttrs::dilation,
              "Specifies the dilation rate to use for dilated convolution.",
              refl::DefaultValue(ffi::Array<int64_t>{1, 1, 1}))
      .def_ro("groups", &Conv3DTransposeAttrs::groups,
              "Number of groups to split the input into for grouped convolution. The number of "
              "input and "
              "output channels should be divisible by the number of groups.",
              refl::DefaultValue(1))
      .def_ro("data_layout", &Conv3DTransposeAttrs::data_layout,
              "Dimension ordering of input data. Can be 'NCDHW', 'NDHWC', etc."
              "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
              "dimensions respectively. Convolution is applied on the 'D', 'H', and"
              "'W' dimensions.",
              refl::DefaultValue(ffi::String("NCDHW")))
      .def_ro(
          "kernel_layout", &Conv3DTransposeAttrs::kernel_layout,
          "Dimension ordering of weight. Can be 'IODHW', etc."
          "'I', 'O', 'D', 'H', 'W' stands for input_channel, output_channel, depth, height, and "
          "width"
          "dimensions respectively.",
          refl::DefaultValue(ffi::String("IODHW")))
      .def_ro("out_layout", &Conv3DTransposeAttrs::out_layout,
              "Dimension ordering of output. Can be 'NCDHW', 'NDHWC', etc."
              "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
              "dimensions respectively. Default to be same as input layout.")
      .def_ro("out_dtype", &Conv3DTransposeAttrs::out_dtype,
              "Output data type, set to explicit type under mixed precision setting",
              refl::DefaultValue(ffi::Optional<DLDataType>{}));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  Conv1DAttrs::RegisterReflection();
  Conv2DAttrs::RegisterReflection();
  Conv3DAttrs::RegisterReflection();
  Conv1DTransposeAttrs::RegisterReflection();
  Conv2DTransposeAttrs::RegisterReflection();
  Conv3DTransposeAttrs::RegisterReflection();
}

/* relax.nn.conv1d */

Expr conv1d(Expr data, Expr weight, ffi::Array<int64_t> strides, ffi::Array<int64_t> padding,
            ffi::Array<int64_t> dilation, int groups, ffi::String data_layout,
            ffi::String kernel_layout, ffi::Optional<ffi::String> out_layout,
            ffi::Optional<DLDataType> out_dtype) {
  padding = GetCompletePadding1D(std::move(padding));

  TVM_FFI_ICHECK_GT(groups, 0)
      << "The number of groups in convolution is expected to be positive. However, "
         "the given number of groups is "
      << groups;
  TVM_FFI_ICHECK_EQ(strides.size(), 1)
      << "The input strides length is expected to be 1. However, the given strides is " << strides;
  TVM_FFI_ICHECK_EQ(dilation.size(), 1)
      << "The input dilation length is expected to be 1. However, the given dilation is "
      << dilation;
  return MakeConv<Conv1DAttrs>(std::move(data), std::move(weight), std::move(strides),
                               std::move(padding), std::move(dilation), groups, data_layout,
                               std::move(kernel_layout), out_layout.value_or(data_layout),
                               out_dtype,
                               /*op_name=*/"relax.nn.conv1d");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.nn.conv1d", conv1d);
}

Type InferTypeConv1d(const Call& call, const BlockBuilder& ctx) {
  ffi::Array<TensorType> input_ty = GetInputTensorType(call, ctx);
  TensorType data_ty = input_ty[0];
  TensorType weight_ty = input_ty[1];

  const auto* attrs = call->attrs.as<Conv1DAttrs>();
  auto [data_layout, data2NCW] = CheckTensorLayout(call, ctx, attrs->data_layout,  //
                                                   /*tgt_layout=*/"NCW",           //
                                                   /*tensor_name=*/"data");
  auto [weight_layout, weight2OIW] = CheckTensorLayout(call, ctx, attrs->kernel_layout,  //
                                                       /*tgt_layout=*/"OIW",             //
                                                       /*tensor_name=*/"kernel");
  auto [out_layout, out2NCW] = CheckTensorLayout(call, ctx, attrs->out_layout,  //
                                                 /*tgt_layout=*/"NCW",          //
                                                 /*tensor_name=*/"output");

  ffi::Optional<ShapeExpr> data_shape =
      CheckNdimPerLayoutAndGetShape(call, ctx, data_ty, data_layout);
  ffi::Optional<ShapeExpr> weight_shape =
      CheckNdimPerLayoutAndGetShape(call, ctx, weight_ty, weight_layout);

  ffi::Optional<PrimType> out_dtype =
      attrs->out_dtype.has_value() ? PrimType(attrs->out_dtype.value())
                                   : InferBinaryArithOpOutDtype(call, ctx, data_ty, weight_ty);
  ffi::Optional<VDevice> vdevice = InferBinaryArithOpOutVDevice(call, ctx, data_ty, weight_ty);
  if (!data_shape.has_value() || !weight_shape.has_value()) {
    return TensorType(out_dtype, out_layout.ndim(), vdevice);
  }

  ffi::Array<PrimExpr> data_NCW_shape = data2NCW.ForwardShape(data_shape.value()->values);
  ffi::Array<PrimExpr> weight_OIW_shape = weight2OIW.ForwardShape(weight_shape.value()->values);

  sym::Analyzer analyzer = ctx->GetAnalyzer();
  PrimExpr input_channel_data = data_NCW_shape[1];
  PrimExpr input_channel_kernel = weight_OIW_shape[1];
  if (analyzer->CanProve(input_channel_data != input_channel_kernel * attrs->groups)) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "The channel size of the data should equal to the product of input channel size of the "
           "weight and the number of groups. However, the data channel size is "
        << input_channel_data << " while the weight input channel size and number of groups are "
        << input_channel_kernel << " and " << attrs->groups;
  } else if (!analyzer->CanProveEqual(input_channel_data, input_channel_kernel * attrs->groups)) {
    // Todo(relax-team): Trust the input shape at this moment, and revisit
    // this condition with runtime shape check
  }
  if (analyzer->CanProve(floormod(weight_OIW_shape[0], attrs->groups) != 0)) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Conv1d expects the number of output channels to be divisible by the "
           "number of groups. However, the number of output channels is "
        << weight_OIW_shape[0] << " while the number of groups is " << attrs->groups;
  } else if (!analyzer->CanProveEqual(floormod(weight_OIW_shape[0], attrs->groups), 0)) {
    // Todo(relax-team): Trust the input shape at this moment, and revisit
    // this condition with runtime shape check
  }

  PrimExpr input_w = data_NCW_shape[2];
  PrimExpr kernel_w = weight_OIW_shape[2];
  PrimExpr padding_w = IntImm::Int32(attrs->padding[0]) + IntImm::Int32(attrs->padding[1]);

  std::vector<PrimExpr> out_NCW_shape;
  out_NCW_shape.reserve(3);
  out_NCW_shape.push_back(data_NCW_shape[0]);
  out_NCW_shape.push_back(weight_OIW_shape[0]);

  PrimExpr numerator_w =
      input_w + padding_w - IntImm::Int32(attrs->dilation[0]) * (kernel_w - 1) - 1;
  out_NCW_shape.push_back(
      analyzer->Simplify(floordiv(numerator_w, IntImm::Int32(attrs->strides[0])) + 1));

  ffi::Array<PrimExpr> out_shape = out2NCW.BackwardShape(out_NCW_shape);
  return TensorType(ShapeExpr(out_shape), out_dtype, vdevice);
}

InferLayoutOutput InferLayoutConv1d(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  const auto& it = desired_layouts.find("relax.nn.conv1d");
  const auto* attrs = call->attrs.as<Conv1DAttrs>();
  TVM_FFI_ICHECK(attrs) << "Invalid Call";

  LayoutDecision data_layout(ffi::UnsafeInit{}), weight_layout(ffi::UnsafeInit{}),
      output_layout(ffi::UnsafeInit{});
  ffi::ObjectPtr<Conv1DAttrs> new_attrs = ffi::make_object<Conv1DAttrs>(*attrs);

  if (it != desired_layouts.end()) {
    // We have a desired layout for conv1d.
    SLayout desired_data_layout = (*it).second[0];
    SLayout desired_weight_layout = (*it).second[1];
    SLayout desired_output_layout = (*it).second.size() == 3 ? (*it).second[2] : (*it).second[0];
    TVM_FFI_ICHECK_EQ(desired_data_layout.ndim(), desired_data_layout.ndim_primal())
        << "Axis swap only";
    TVM_FFI_ICHECK_EQ(desired_weight_layout.ndim(), desired_weight_layout.ndim_primal())
        << "Axis swap only";
    TVM_FFI_ICHECK_EQ(desired_output_layout.ndim(), desired_output_layout.ndim_primal())
        << "Axis swap only";
    data_layout =
        TransposeLike(InitialLayout(3), SLayout::Create(attrs->data_layout), desired_data_layout);
    weight_layout = TransposeLike(InitialLayout(3), SLayout::Create(attrs->kernel_layout),
                                  desired_weight_layout);
    output_layout =
        TransposeLike(InitialLayout(3), SLayout::Create(attrs->out_layout), desired_output_layout);
    new_attrs->data_layout = (*it).second[0];
    new_attrs->kernel_layout = (*it).second[1];
    new_attrs->out_layout = (*it).second.size() == 3 ? (*it).second[2] : (*it).second[0];
  } else {
    // We don't have a desired layout for conv1d.
    // We can just propagate the layout from the input.
    data_layout = GetLayoutDecision(var_layout_map, call->args[0]);
    weight_layout = GetLayoutDecision(var_layout_map, call->args[1]);
    output_layout = data_layout;
    new_attrs->data_layout =
        TransposeLike(SLayout::Create(attrs->data_layout), InitialLayout(3), data_layout->layout)
            .name();
    new_attrs->kernel_layout = TransposeLike(SLayout::Create(attrs->kernel_layout),
                                             InitialLayout(3), weight_layout->layout)
                                   .name();
    new_attrs->out_layout =
        TransposeLike(SLayout::Create(attrs->out_layout), InitialLayout(3), output_layout->layout)
            .name();
  }
  return InferLayoutOutput({data_layout, weight_layout}, {output_layout}, Attrs(new_attrs));
}

Call InferMixedPrecisionConv1d(const Call& call, DLDataType out_dtype) {
  const auto* conv1d_attrs = call->attrs.as<Conv1DAttrs>();
  return conv1d(call->args[0], call->args[1], conv1d_attrs->strides, conv1d_attrs->padding,
                conv1d_attrs->dilation, conv1d_attrs->groups, conv1d_attrs->data_layout,
                conv1d_attrs->kernel_layout, conv1d_attrs->out_layout, out_dtype)
      .as_or_throw<Call>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.nn.conv1d")
      .signature(
          sig::arg("data", "The input tensor."), sig::arg("weight", "The weight tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."),
          sig::call_attrs<Conv1DAttrs>())
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeConv1d)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutConv1d)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kAlways)
      .set_attr(tvm::relax::op_attr::kInferMixedPrecision, InferMixedPrecisionConv1d)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.nn.conv2d */

Expr conv2d(Expr data, Expr weight, ffi::Array<int64_t> strides, ffi::Array<int64_t> padding,
            ffi::Array<int64_t> dilation, int groups, ffi::String data_layout,
            ffi::String kernel_layout, ffi::Optional<ffi::String> out_layout,
            ffi::Optional<DLDataType> out_dtype) {
  padding = GetCompletePadding2D(std::move(padding));
  if (strides.size() == 1) {
    strides.push_back(strides[0]);
  }
  if (dilation.size() == 1) {
    dilation.push_back(dilation[0]);
  }

  TVM_FFI_ICHECK_GT(groups, 0)
      << "The number of groups in convolution is expected to be positive. However, "
         "the given number of groups is "
      << groups;
  TVM_FFI_ICHECK_EQ(strides.size(), 2)
      << "The input strides length is expected to be 2. However, the given strides is " << strides;
  TVM_FFI_ICHECK_EQ(dilation.size(), 2)
      << "The input dilation length is expected to be 2. However, the given dilation is "
      << dilation;
  return MakeConv<Conv2DAttrs>(std::move(data), std::move(weight), std::move(strides),
                               std::move(padding), std::move(dilation), groups, data_layout,
                               std::move(kernel_layout), out_layout.value_or(data_layout),
                               out_dtype,
                               /*op_name=*/"relax.nn.conv2d");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.nn.conv2d", conv2d);
}

Type InferTypeConv2d(const Call& call, const BlockBuilder& ctx) {
  ffi::Array<TensorType> input_ty = GetInputTensorType(call, ctx);
  TensorType data_ty = input_ty[0];
  TensorType weight_ty = input_ty[1];

  const auto* attrs = call->attrs.as<Conv2DAttrs>();
  auto [data_layout, data2NCHW] = CheckTensorLayout(call, ctx, attrs->data_layout,  //
                                                    /*tgt_layout=*/"NCHW",          //
                                                    /*tensor_name=*/"data");
  auto [weight_layout, weight2OIHW] = CheckTensorLayout(call, ctx, attrs->kernel_layout,  //
                                                        /*tgt_layout=*/"OIHW",            //
                                                        /*tensor_name=*/"kernel");
  auto [out_layout, out2NCHW] = CheckTensorLayout(call, ctx, attrs->out_layout,  //
                                                  /*tgt_layout=*/"NCHW",         //
                                                  /*tensor_name=*/"output");

  ffi::Optional<ShapeExpr> data_shape =
      CheckNdimPerLayoutAndGetShape(call, ctx, data_ty, data_layout);
  ffi::Optional<ShapeExpr> weight_shape =
      CheckNdimPerLayoutAndGetShape(call, ctx, weight_ty, weight_layout);

  ffi::Optional<PrimType> out_dtype =
      attrs->out_dtype.has_value() ? PrimType(attrs->out_dtype.value())
                                   : InferBinaryArithOpOutDtype(call, ctx, data_ty, weight_ty);
  ffi::Optional<VDevice> vdevice = InferBinaryArithOpOutVDevice(call, ctx, data_ty, weight_ty);
  if (!data_shape.has_value() || !weight_shape.has_value()) {
    return TensorType(out_dtype, out_layout.ndim(), vdevice);
  }

  ffi::Array<PrimExpr> data_NCHW_shape = data2NCHW.ForwardShape(data_shape.value()->values);
  ffi::Array<PrimExpr> weight_OIHW_shape = weight2OIHW.ForwardShape(weight_shape.value()->values);

  sym::Analyzer analyzer = ctx->GetAnalyzer();
  PrimExpr input_channel_data = data_NCHW_shape[1];
  PrimExpr input_channel_kernel = weight_OIHW_shape[1];
  if (analyzer->CanProve(input_channel_data != input_channel_kernel * attrs->groups)) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "The channel size of the data should equal to the product of input channel size of the "
           "weight and the number of groups. However, the data channel size is "
        << input_channel_data << " while the weight input channel size and number of groups are "
        << input_channel_kernel << " and " << attrs->groups;
  } else if (!analyzer->CanProveEqual(input_channel_data, input_channel_kernel * attrs->groups)) {
    // Todo(relax-team): Trust the input shape at this moment, and revisit
    // this condition with runtime shape check
  }
  if (analyzer->CanProve(floormod(weight_OIHW_shape[0], attrs->groups) != 0)) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Conv2d expects the number of output channels to be divisible by the "
           "number of groups. However, the number of output channels is "
        << weight_OIHW_shape[0] << " while the number of groups is " << attrs->groups;
  } else if (!analyzer->CanProveEqual(floormod(weight_OIHW_shape[0], attrs->groups), 0)) {
    // Todo(relax-team): Trust the input shape at this moment, and revisit
    // this condition with runtime shape check
  }

  PrimExpr input_h = data_NCHW_shape[2];
  PrimExpr input_w = data_NCHW_shape[3];
  PrimExpr kernel_h = weight_OIHW_shape[2];
  PrimExpr kernel_w = weight_OIHW_shape[3];
  PrimExpr padding_h = IntImm::Int32(attrs->padding[0]) + IntImm::Int32(attrs->padding[2]);
  PrimExpr padding_w = IntImm::Int32(attrs->padding[1]) + IntImm::Int32(attrs->padding[3]);

  std::vector<PrimExpr> out_NCHW_shape;
  out_NCHW_shape.reserve(4);
  out_NCHW_shape.push_back(data_NCHW_shape[0]);
  out_NCHW_shape.push_back(weight_OIHW_shape[0]);

  PrimExpr numerator_h =
      input_h + padding_h - IntImm::Int32(attrs->dilation[0]) * (kernel_h - 1) - 1;
  PrimExpr numerator_w =
      input_w + padding_w - IntImm::Int32(attrs->dilation[1]) * (kernel_w - 1) - 1;
  out_NCHW_shape.push_back(
      analyzer->Simplify(floordiv(numerator_h, IntImm::Int32(attrs->strides[0])) + 1));
  out_NCHW_shape.push_back(
      analyzer->Simplify(floordiv(numerator_w, IntImm::Int32(attrs->strides[1])) + 1));

  ffi::Array<PrimExpr> out_shape = out2NCHW.BackwardShape(out_NCHW_shape);
  return TensorType(ShapeExpr(out_shape), out_dtype, vdevice);
}

InferLayoutOutput InferLayoutConv2d(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  const auto& it = desired_layouts.find("relax.nn.conv2d");
  const auto* attrs = call->attrs.as<Conv2DAttrs>();
  TVM_FFI_ICHECK(attrs) << "Invalid Call";

  LayoutDecision data_layout(ffi::UnsafeInit{}), weight_layout(ffi::UnsafeInit{}),
      output_layout(ffi::UnsafeInit{});
  data_layout = GetLayoutDecision(var_layout_map, call->args[0]);
  weight_layout = GetLayoutDecision(var_layout_map, call->args[1]);
  ffi::ObjectPtr<Conv2DAttrs> new_attrs = ffi::make_object<Conv2DAttrs>(*attrs);

  if (it != desired_layouts.end()) {
    // We have a desired layout for conv2d.
    SLayout desired_data_layout = (*it).second[0];
    SLayout desired_weight_layout = (*it).second[1];
    SLayout desired_output_layout = (*it).second.size() == 3 ? (*it).second[2] : (*it).second[0];
    tvm::PrimType i64_ty = tvm::PrimType::Int(64);
    tirx::SLayout input_layout(attrs->data_layout, i64_ty);
    tirx::SLayout kernel_layout(attrs->kernel_layout, i64_ty);
    tirx::SLayout out_layout(attrs->out_layout, i64_ty);

    if ((desired_data_layout.ndim() == input_layout.ndim()) &&
        (desired_weight_layout.ndim() == kernel_layout.ndim()) &&
        (desired_output_layout.ndim() == out_layout.ndim())) {
      // Just a transpose
      data_layout =
          TransposeLike(InitialLayout(4), SLayout::Create(attrs->data_layout), desired_data_layout);
      weight_layout = TransposeLike(InitialLayout(4), SLayout::Create(attrs->kernel_layout),
                                    desired_weight_layout);
      output_layout = TransposeLike(InitialLayout(4), SLayout::Create(attrs->out_layout),
                                    desired_output_layout);
      new_attrs->data_layout = (*it).second[0];
      new_attrs->kernel_layout = (*it).second[1];
      new_attrs->out_layout = (*it).second.size() == 3 ? (*it).second[2] : (*it).second[0];
      return InferLayoutOutput({data_layout, weight_layout}, {output_layout}, Attrs(new_attrs));
    } else {
      // Layout Transform
      auto data_si = GetType(call->args[0]);
      auto kernel_si = GetType(call->args[1]);
      TensorType data_ty = data_si.as<TensorType>().value();
      TensorType kernel_ty = kernel_si.as<TensorType>().value();
      ffi::Optional<ShapeExpr> data_shape =
          ffi::GetRef<ShapeExpr>(data_ty->shape.as<ShapeExprObj>());
      ffi::Optional<ShapeExpr> kernel_shape =
          ffi::GetRef<ShapeExpr>(kernel_ty->shape.as<ShapeExprObj>());

      bool can_data_proved =
          CanProveLayoutTransform(input_layout, desired_data_layout, data_shape.value()->values);
      bool can_kernel_proved = CanProveLayoutTransform(kernel_layout, desired_weight_layout,
                                                       kernel_shape.value()->values);

      if (can_data_proved && can_kernel_proved) {
        data_layout = TransposeSubLayoutLike(InitialLayout(4), input_layout, desired_data_layout);
        weight_layout =
            TransposeSubLayoutLike(InitialLayout(4), kernel_layout, desired_weight_layout);
        output_layout = TransposeSubLayoutLike(InitialLayout(4), out_layout, desired_output_layout);
        new_attrs->data_layout = (*it).second[0];
        new_attrs->kernel_layout = (*it).second[1];
        new_attrs->out_layout = (*it).second.size() == 3 ? (*it).second[2] : (*it).second[0];
        return InferLayoutOutput({data_layout, weight_layout}, {output_layout}, Attrs(new_attrs));
      } else {
        data_layout = LayoutDecision(InitialLayout(4));
        weight_layout = LayoutDecision(InitialLayout(4));
      }
    }
  }

  // We don't have a desired layout for conv2d or desired layouts not compatible.
  // We can just propagate the layout from the input.

  output_layout = data_layout;
  new_attrs->data_layout =
      TransposeLike(SLayout::Create(attrs->data_layout), InitialLayout(4), data_layout->layout)
          .name();
  new_attrs->kernel_layout =
      TransposeLike(SLayout::Create(attrs->kernel_layout), InitialLayout(4), weight_layout->layout)
          .name();
  new_attrs->out_layout =
      TransposeLike(SLayout::Create(attrs->out_layout), InitialLayout(4), output_layout->layout)
          .name();
  return InferLayoutOutput({data_layout, weight_layout}, {output_layout}, Attrs(new_attrs));
}

Call InferMixedPrecisionConv2d(const Call& call, DLDataType out_dtype) {
  const auto* conv2d_attrs = call->attrs.as<Conv2DAttrs>();
  return conv2d(call->args[0], call->args[1], conv2d_attrs->strides, conv2d_attrs->padding,
                conv2d_attrs->dilation, conv2d_attrs->groups, conv2d_attrs->data_layout,
                conv2d_attrs->kernel_layout, conv2d_attrs->out_layout, out_dtype)
      .as_or_throw<Call>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.nn.conv2d")
      .signature(
          sig::arg("data", "The input tensor."), sig::arg("weight", "The weight tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."),
          sig::call_attrs<Conv2DAttrs>())
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeConv2d)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutConv2d)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kAlways)
      .set_attr(tvm::relax::op_attr::kInferMixedPrecision, InferMixedPrecisionConv2d)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.nn.conv3d */

Expr conv3d(Expr data, Expr weight, ffi::Array<int64_t> strides, ffi::Array<int64_t> padding,
            ffi::Array<int64_t> dilation, int groups, ffi::String data_layout,
            ffi::String kernel_layout, ffi::Optional<ffi::String> out_layout,
            ffi::Optional<DLDataType> out_dtype) {
  padding = GetCompletePadding3D(std::move(padding));
  if (strides.size() == 1) {
    strides.push_back(strides[0]);
    strides.push_back(strides[0]);
  }
  if (dilation.size() == 1) {
    dilation.push_back(dilation[0]);
    dilation.push_back(dilation[0]);
  }

  TVM_FFI_ICHECK_GT(groups, 0)
      << "The number of groups in convolution is expected to be positive. However, "
         "the given number of groups is "
      << groups;
  TVM_FFI_ICHECK_EQ(strides.size(), 3)
      << "The input strides length is expected to be 3. However, the given strides is " << strides;
  TVM_FFI_ICHECK_EQ(dilation.size(), 3)
      << "The input dilation length is expected to be 3. However, the given dilation is "
      << dilation;
  return MakeConv<Conv3DAttrs>(std::move(data), std::move(weight), std::move(strides),
                               std::move(padding), std::move(dilation), groups, data_layout,
                               std::move(kernel_layout), out_layout.value_or(data_layout),
                               out_dtype,
                               /*op_name=*/"relax.nn.conv3d");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.nn.conv3d", conv3d);
}

Type InferTypeConv3d(const Call& call, const BlockBuilder& ctx) {
  ffi::Array<TensorType> input_ty = GetInputTensorType(call, ctx);
  TensorType data_ty = input_ty[0];
  TensorType weight_ty = input_ty[1];

  const auto* attrs = call->attrs.as<Conv3DAttrs>();
  auto [data_layout, data2NCDHW] = CheckTensorLayout(call, ctx, attrs->data_layout,  //
                                                     /*tgt_layout=*/"NCDHW",         //
                                                     /*tensor_name=*/"data");
  auto [weight_layout, weight2OIDHW] = CheckTensorLayout(call, ctx, attrs->kernel_layout,  //
                                                         /*tgt_layout=*/"OIDHW",           //
                                                         /*tensor_name=*/"kernel");
  auto [out_layout, out2NCDHW] = CheckTensorLayout(call, ctx, attrs->out_layout,  //
                                                   /*tgt_layout=*/"NCDHW",        //
                                                   /*tensor_name=*/"output");

  ffi::Optional<ShapeExpr> data_shape =
      CheckNdimPerLayoutAndGetShape(call, ctx, data_ty, data_layout);
  ffi::Optional<ShapeExpr> weight_shape =
      CheckNdimPerLayoutAndGetShape(call, ctx, weight_ty, weight_layout);

  ffi::Optional<PrimType> out_dtype =
      attrs->out_dtype.has_value() ? PrimType(attrs->out_dtype.value())
                                   : InferBinaryArithOpOutDtype(call, ctx, data_ty, weight_ty);
  ffi::Optional<VDevice> vdevice = InferBinaryArithOpOutVDevice(call, ctx, data_ty, weight_ty);
  if (!data_shape.has_value() || !weight_shape.has_value()) {
    return TensorType(out_dtype, out_layout.ndim(), vdevice);
  }

  ffi::Array<PrimExpr> data_NCDHW_shape = data2NCDHW.ForwardShape(data_shape.value()->values);
  ffi::Array<PrimExpr> weight_OIDHW_shape = weight2OIDHW.ForwardShape(weight_shape.value()->values);

  sym::Analyzer analyzer = ctx->GetAnalyzer();
  PrimExpr input_channel_data = data_NCDHW_shape[1];
  PrimExpr input_channel_kernel = weight_OIDHW_shape[1];
  if (analyzer->CanProve(input_channel_data != input_channel_kernel * attrs->groups)) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "The channel size of the data should equal to the product of input channel size of the "
           "weight and the number of groups. However, the data channel size is "
        << input_channel_data << " while the weight input channel size and number of groups are "
        << input_channel_kernel << " and " << attrs->groups;
  } else if (!analyzer->CanProveEqual(input_channel_data, input_channel_kernel * attrs->groups)) {
    // Todo(relax-team): Trust the input shape at this moment, and revisit
    // this condition with runtime shape check
  }
  if (analyzer->CanProve(floormod(weight_OIDHW_shape[0], attrs->groups) != 0)) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Conv3d expects the number of output channels to be divisible by the "
           "number of groups. However, the number of output channels is "
        << weight_OIDHW_shape[0] << " while the number of groups is " << attrs->groups;
  } else if (!analyzer->CanProveEqual(floormod(weight_OIDHW_shape[0], attrs->groups), 0)) {
    // Todo(relax-team): Trust the input shape at this moment, and revisit
    // this condition with runtime shape check
  }

  PrimExpr input_d = data_NCDHW_shape[2];
  PrimExpr input_h = data_NCDHW_shape[3];
  PrimExpr input_w = data_NCDHW_shape[4];
  PrimExpr kernel_d = weight_OIDHW_shape[2];
  PrimExpr kernel_h = weight_OIDHW_shape[3];
  PrimExpr kernel_w = weight_OIDHW_shape[4];
  PrimExpr padding_d = IntImm::Int32(attrs->padding[0]) + IntImm::Int32(attrs->padding[3]);
  PrimExpr padding_h = IntImm::Int32(attrs->padding[1]) + IntImm::Int32(attrs->padding[4]);
  PrimExpr padding_w = IntImm::Int32(attrs->padding[2]) + IntImm::Int32(attrs->padding[5]);

  std::vector<PrimExpr> out_NCDHW_shape;
  out_NCDHW_shape.reserve(5);
  out_NCDHW_shape.push_back(data_NCDHW_shape[0]);
  out_NCDHW_shape.push_back(weight_OIDHW_shape[0]);

  PrimExpr numerator_d =
      input_d + padding_d - IntImm::Int32(attrs->dilation[0]) * (kernel_d - 1) - 1;
  PrimExpr numerator_h =
      input_h + padding_h - IntImm::Int32(attrs->dilation[1]) * (kernel_h - 1) - 1;
  PrimExpr numerator_w =
      input_w + padding_w - IntImm::Int32(attrs->dilation[2]) * (kernel_w - 1) - 1;
  out_NCDHW_shape.push_back(
      analyzer->Simplify(floordiv(numerator_d, IntImm::Int32(attrs->strides[0])) + 1));
  out_NCDHW_shape.push_back(
      analyzer->Simplify(floordiv(numerator_h, IntImm::Int32(attrs->strides[1])) + 1));
  out_NCDHW_shape.push_back(
      analyzer->Simplify(floordiv(numerator_w, IntImm::Int32(attrs->strides[2])) + 1));

  ffi::Array<PrimExpr> out_shape = out2NCDHW.BackwardShape(out_NCDHW_shape);
  return TensorType(ShapeExpr(out_shape), out_dtype, vdevice);
}

InferLayoutOutput InferLayoutConv3d(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  const auto& it = desired_layouts.find("relax.nn.conv3d");
  const auto* attrs = call->attrs.as<Conv3DAttrs>();
  TVM_FFI_ICHECK(attrs) << "Invalid Call";

  LayoutDecision data_layout(ffi::UnsafeInit{}), weight_layout(ffi::UnsafeInit{}),
      output_layout(ffi::UnsafeInit{});
  ffi::ObjectPtr<Conv3DAttrs> new_attrs = ffi::make_object<Conv3DAttrs>(*attrs);

  if (it != desired_layouts.end()) {
    // We have a desired layout for conv3d.
    SLayout desired_data_layout = (*it).second[0];
    SLayout desired_weight_layout = (*it).second[1];
    SLayout desired_output_layout = (*it).second.size() == 3 ? (*it).second[2] : (*it).second[0];
    TVM_FFI_ICHECK_EQ(desired_data_layout.ndim(), desired_data_layout.ndim_primal())
        << "Axis swap only";
    TVM_FFI_ICHECK_EQ(desired_weight_layout.ndim(), desired_weight_layout.ndim_primal())
        << "Axis swap only";
    TVM_FFI_ICHECK_EQ(desired_output_layout.ndim(), desired_output_layout.ndim_primal())
        << "Axis swap only";
    data_layout =
        TransposeLike(InitialLayout(5), SLayout::Create(attrs->data_layout), desired_data_layout);
    weight_layout = TransposeLike(InitialLayout(5), SLayout::Create(attrs->kernel_layout),
                                  desired_weight_layout);
    output_layout =
        TransposeLike(InitialLayout(5), SLayout::Create(attrs->out_layout), desired_output_layout);
    new_attrs->data_layout = (*it).second[0];
    new_attrs->kernel_layout = (*it).second[1];
    new_attrs->out_layout = (*it).second.size() == 3 ? (*it).second[2] : (*it).second[0];
  } else {
    // We don't have a desired layout for conv2d.
    // We can just propagate the layout from the input.
    data_layout = GetLayoutDecision(var_layout_map, call->args[0]);
    weight_layout = GetLayoutDecision(var_layout_map, call->args[1]);
    output_layout = data_layout;
    new_attrs->data_layout =
        TransposeLike(SLayout::Create(attrs->data_layout), InitialLayout(5), data_layout->layout)
            .name();
    new_attrs->kernel_layout = TransposeLike(SLayout::Create(attrs->kernel_layout),
                                             InitialLayout(5), weight_layout->layout)
                                   .name();
    new_attrs->out_layout =
        TransposeLike(SLayout::Create(attrs->out_layout), InitialLayout(5), output_layout->layout)
            .name();
  }
  return InferLayoutOutput({data_layout, weight_layout}, {output_layout}, Attrs(new_attrs));
}

Call InferMixedPrecisionConv3d(const Call& call, DLDataType out_dtype) {
  const auto* conv3d_attrs = call->attrs.as<Conv3DAttrs>();
  return conv3d(call->args[0], call->args[1], conv3d_attrs->strides, conv3d_attrs->padding,
                conv3d_attrs->dilation, conv3d_attrs->groups, conv3d_attrs->data_layout,
                conv3d_attrs->kernel_layout, conv3d_attrs->out_layout, out_dtype)
      .as_or_throw<Call>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.nn.conv3d")
      .signature(
          sig::arg("data", "The input tensor."), sig::arg("weight", "The weight tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."),
          sig::call_attrs<Conv3DAttrs>())
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeConv3d)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutConv3d)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kAlways)
      .set_attr(tvm::relax::op_attr::kInferMixedPrecision, InferMixedPrecisionConv3d)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

Expr conv1d_transpose(Expr data, Expr weight, ffi::Array<int64_t> strides,
                      ffi::Array<int64_t> padding, ffi::Array<int64_t> output_padding,
                      ffi::Array<int64_t> dilation, int groups, ffi::String data_layout,
                      ffi::String kernel_layout, ffi::Optional<ffi::String> out_layout,
                      ffi::Optional<DLDataType> out_dtype) {
  padding = GetCompletePadding1D(std::move(padding));

  TVM_FFI_ICHECK_GT(groups, 0)
      << "The number of groups in convolution is expected to be positive. However, "
         "the given number of groups is "
      << groups;
  TVM_FFI_ICHECK_EQ(output_padding.size(), 1)
      << "The input output_padding length is expected to be 1. "
         "However, the given output_padding is "
      << output_padding;
  TVM_FFI_ICHECK_EQ(strides.size(), 1)
      << "The input strides length is expected to be 1. However, the given strides is " << strides;
  TVM_FFI_ICHECK_EQ(dilation.size(), 1)
      << "The input dilation length is expected to be 1. However, the given dilation is "
      << dilation;

  auto attrs = ffi::make_object<Conv1DTransposeAttrs>();
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->output_padding = std::move(output_padding);
  attrs->dilation = std::move(dilation);
  attrs->groups = groups;
  attrs->data_layout = data_layout;
  attrs->kernel_layout = std::move(kernel_layout);
  attrs->out_layout = out_layout.value_or(data_layout);
  attrs->out_dtype = out_dtype;
  const Op op = Op::Get("relax.nn.conv1d_transpose");
  return Call(Type::Missing(), op, {data, weight}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.nn.conv1d_transpose", conv1d_transpose);
}

Type InferTypeConv1dTranspose(const Call& call, const BlockBuilder& ctx) {
  ffi::Array<TensorType> input_ty = GetInputTensorType(call, ctx);
  TensorType data_ty = input_ty[0];
  TensorType weight_ty = input_ty[1];

  const auto* attrs = call->attrs.as<Conv1DTransposeAttrs>();
  auto [data_layout, data2NCW] = CheckTensorLayout(call, ctx, attrs->data_layout,  //
                                                   /*tgt_layout=*/"NCW",           //
                                                   /*tensor_name=*/"data");
  auto [weight_layout, weight2IOW] = CheckTensorLayout(call, ctx, attrs->kernel_layout,  //
                                                       /*tgt_layout=*/"IOW",             //
                                                       /*tensor_name=*/"kernel");
  auto [out_layout, out2NCW] = CheckTensorLayout(call, ctx, attrs->out_layout,  //
                                                 /*tgt_layout=*/"NCW",          //
                                                 /*tensor_name=*/"output");
  ffi::Optional<ShapeExpr> data_shape =
      CheckNdimPerLayoutAndGetShape(call, ctx, data_ty, data_layout);
  ffi::Optional<ShapeExpr> weight_shape =
      CheckNdimPerLayoutAndGetShape(call, ctx, weight_ty, weight_layout);

  ffi::Optional<PrimType> out_dtype =
      attrs->out_dtype.has_value() ? PrimType(attrs->out_dtype.value())
                                   : InferBinaryArithOpOutDtype(call, ctx, data_ty, weight_ty);
  ffi::Optional<VDevice> vdevice = InferBinaryArithOpOutVDevice(call, ctx, data_ty, weight_ty);
  if (!data_shape.has_value() || !weight_shape.has_value()) {
    return TensorType(out_dtype, out_layout.ndim(), vdevice);
  }

  ffi::Array<PrimExpr> data_NCW_shape = data2NCW.ForwardShape(data_shape.value()->values);
  ffi::Array<PrimExpr> weight_IOW_shape = weight2IOW.ForwardShape(weight_shape.value()->values);

  sym::Analyzer analyzer = ctx->GetAnalyzer();
  PrimExpr input_channel_data = data_NCW_shape[1];
  PrimExpr input_channel_kernel = weight_IOW_shape[0];
  if (analyzer->CanProve(input_channel_data != input_channel_kernel)) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Conv1dTranspose expects the channel size of the data should equal to the input channel "
           "size of the weight. However, the data channel size is "
        << input_channel_data << " while the weight input channel size is " << input_channel_kernel;
  } else if (!analyzer->CanProveEqual(input_channel_data, input_channel_kernel)) {
    // Todo(relax-team): Trust the input shape at this moment, and revisit
    // this condition with runtime shape check
  }
  if (analyzer->CanProve(floormod(input_channel_kernel, attrs->groups) != 0)) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Conv1dTranspose expects the number of input channels to be divisible by "
           "the number of groups. However, the number of input channels is "
        << input_channel_kernel << " while the number of groups is " << attrs->groups;
  } else if (!analyzer->CanProveEqual(floormod(input_channel_kernel, attrs->groups), 0)) {
    // Todo(relax-team): Trust the input shape at this moment, and revisit
    // this condition with runtime shape check
  }
  if (attrs->output_padding[0] >= attrs->strides[0]) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Conv1dTranspose expects the output padding less than the strides, but the "
           "output padding is"
        << attrs->output_padding << " while the strides are" << attrs->strides;
  } else if (!(attrs->output_padding[0] < attrs->strides[0])) {
    // Todo(relax-team): Trust the input padding at this moment, and revisit
    // this condition with runtime shape check
  }

  PrimExpr input_w = data_NCW_shape[2];
  PrimExpr kernel_w = weight_IOW_shape[2];
  PrimExpr padding_w = IntImm::Int32(attrs->padding[0]) + IntImm::Int32(attrs->padding[1]);

  std::vector<PrimExpr> out_NCW_shape;
  out_NCW_shape.reserve(3);
  out_NCW_shape.push_back(data_NCW_shape[0]);
  out_NCW_shape.push_back(weight_IOW_shape[1] * attrs->groups);

  PrimExpr out_w = (input_w - 1) * IntImm::Int32(attrs->strides[0]) - padding_w +
                   IntImm::Int32(attrs->dilation[0]) * (kernel_w - 1) +
                   IntImm::Int32(attrs->output_padding[0]) + 1;
  out_NCW_shape.push_back(analyzer->Simplify(out_w));

  ffi::Array<PrimExpr> out_shape = out2NCW.BackwardShape(out_NCW_shape);
  return TensorType(ShapeExpr(out_shape), out_dtype, vdevice);
}

InferLayoutOutput InferLayoutConv1dTranspose(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  const auto* attrs = call->attrs.as<Conv1DTransposeAttrs>();
  LayoutDecision data_layout(ffi::UnsafeInit{}), weight_layout(ffi::UnsafeInit{}),
      output_layout(ffi::UnsafeInit{});
  ffi::ObjectPtr<Conv1DTransposeAttrs> new_attrs = ffi::make_object<Conv1DTransposeAttrs>(*attrs);

  auto it = desired_layouts.find("relax.nn.conv1d_transpose");
  if (it != desired_layouts.end()) {
    SLayout desired_data_layout = (*it).second[0];
    SLayout desired_weight_layout = (*it).second[1];
    SLayout desired_output_layout = (*it).second.size() == 3 ? (*it).second[2] : (*it).second[0];
    TVM_FFI_ICHECK_EQ(desired_data_layout.ndim(), desired_data_layout.ndim_primal())
        << "Axis swap only";
    TVM_FFI_ICHECK_EQ(desired_weight_layout.ndim(), desired_weight_layout.ndim_primal())
        << "Axis swap only";
    TVM_FFI_ICHECK_EQ(desired_output_layout.ndim(), desired_output_layout.ndim_primal())
        << "Axis swap only";
    data_layout =
        TransposeLike(InitialLayout(3), SLayout::Create(attrs->data_layout), desired_data_layout);
    weight_layout = TransposeLike(InitialLayout(3), SLayout::Create(attrs->kernel_layout),
                                  desired_weight_layout);
    output_layout =
        TransposeLike(InitialLayout(3), SLayout::Create(attrs->out_layout), desired_output_layout);
    new_attrs->data_layout = (*it).second[0];
    new_attrs->kernel_layout = (*it).second[1];
    new_attrs->out_layout = (*it).second.size() == 3 ? (*it).second[2] : (*it).second[0];
  } else {
    data_layout = GetLayoutDecision(var_layout_map, call->args[0]);
    weight_layout = GetLayoutDecision(var_layout_map, call->args[1]);
    output_layout = data_layout;
    new_attrs->data_layout =
        TransposeLike(SLayout::Create(attrs->data_layout), InitialLayout(3), data_layout->layout)
            .name();
    new_attrs->kernel_layout = TransposeLike(SLayout::Create(attrs->kernel_layout),
                                             InitialLayout(3), weight_layout->layout)
                                   .name();
    new_attrs->out_layout =
        TransposeLike(SLayout::Create(attrs->out_layout), InitialLayout(3), output_layout->layout)
            .name();
  }
  return InferLayoutOutput({data_layout, weight_layout}, {output_layout}, Attrs(new_attrs));
}

Call InferMixedPrecisionConv1dTranspose(const Call& call, DLDataType out_dtype) {
  const auto* conv1d_transpose_attrs = call->attrs.as<Conv1DTransposeAttrs>();
  return conv1d_transpose(call->args[0], call->args[1], conv1d_transpose_attrs->strides,
                          conv1d_transpose_attrs->padding, conv1d_transpose_attrs->output_padding,
                          conv1d_transpose_attrs->dilation, conv1d_transpose_attrs->groups,
                          conv1d_transpose_attrs->data_layout,
                          conv1d_transpose_attrs->kernel_layout, conv1d_transpose_attrs->out_layout,
                          out_dtype)
      .as_or_throw<Call>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.nn.conv1d_transpose")
      .signature(
          sig::arg("data", "The input tensor."), sig::arg("weight", "The weight tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."),
          sig::call_attrs<Conv1DTransposeAttrs>())
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeConv1dTranspose)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutConv1dTranspose)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kAlways)
      .set_attr(tvm::relax::op_attr::kInferMixedPrecision, InferMixedPrecisionConv1dTranspose)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.nn.conv2d_transpose */

Expr conv2d_transpose(Expr data, Expr weight, ffi::Array<int64_t> strides,
                      ffi::Array<int64_t> padding, ffi::Array<int64_t> output_padding,
                      ffi::Array<int64_t> dilation, int groups, ffi::String data_layout,
                      ffi::String kernel_layout, ffi::Optional<ffi::String> out_layout,
                      ffi::Optional<DLDataType> out_dtype) {
  padding = GetCompletePadding2D(std::move(padding));
  if (output_padding.size() == 1) {
    output_padding.push_back(output_padding[0]);
  }
  if (strides.size() == 1) {
    strides.push_back(strides[0]);
  }
  if (dilation.size() == 1) {
    dilation.push_back(dilation[0]);
  }

  TVM_FFI_ICHECK_GT(groups, 0)
      << "The number of groups in convolution is expected to be positive. However, "
         "the given number of groups is "
      << groups;
  TVM_FFI_ICHECK_EQ(output_padding.size(), 2)
      << "The input output_padding length is expected to be 2. "
         "However, the given output_padding is "
      << output_padding;
  TVM_FFI_ICHECK_EQ(strides.size(), 2)
      << "The input strides length is expected to be 2. However, the given strides is " << strides;
  TVM_FFI_ICHECK_EQ(dilation.size(), 2)
      << "The input dilation length is expected to be 2. However, the given dilation is "
      << dilation;

  auto attrs = ffi::make_object<Conv2DTransposeAttrs>();
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->output_padding = std::move(output_padding);
  attrs->dilation = std::move(dilation);
  attrs->groups = groups;
  attrs->data_layout = data_layout;
  attrs->kernel_layout = std::move(kernel_layout);
  attrs->out_layout = out_layout.value_or(data_layout);
  attrs->out_dtype = out_dtype;
  const Op op = Op::Get("relax.nn.conv2d_transpose");
  return Call(Type::Missing(), op, {data, weight}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.nn.conv2d_transpose", conv2d_transpose);
}

Type InferTypeConv2dTranspose(const Call& call, const BlockBuilder& ctx) {
  ffi::Array<TensorType> input_ty = GetInputTensorType(call, ctx);
  TensorType data_ty = input_ty[0];
  TensorType weight_ty = input_ty[1];

  const auto* attrs = call->attrs.as<Conv2DTransposeAttrs>();
  auto [data_layout, data2NCHW] = CheckTensorLayout(call, ctx, attrs->data_layout,  //
                                                    /*tgt_layout=*/"NCHW",          //
                                                    /*tensor_name=*/"data");
  auto [weight_layout, weight2IOHW] = CheckTensorLayout(call, ctx, attrs->kernel_layout,  //
                                                        /*tgt_layout=*/"IOHW",            //
                                                        /*tensor_name=*/"kernel");
  auto [out_layout, out2NCHW] = CheckTensorLayout(call, ctx, attrs->out_layout,  //
                                                  /*tgt_layout=*/"NCHW",         //
                                                  /*tensor_name=*/"output");

  ffi::Optional<ShapeExpr> data_shape =
      CheckNdimPerLayoutAndGetShape(call, ctx, data_ty, data_layout);
  ffi::Optional<ShapeExpr> weight_shape =
      CheckNdimPerLayoutAndGetShape(call, ctx, weight_ty, weight_layout);

  ffi::Optional<PrimType> out_dtype =
      attrs->out_dtype.has_value() ? PrimType(attrs->out_dtype.value())
                                   : InferBinaryArithOpOutDtype(call, ctx, data_ty, weight_ty);
  ffi::Optional<VDevice> vdevice = InferBinaryArithOpOutVDevice(call, ctx, data_ty, weight_ty);
  if (!data_shape.has_value() || !weight_shape.has_value()) {
    return TensorType(out_dtype, out_layout.ndim(), vdevice);
  }

  ffi::Array<PrimExpr> data_NCHW_shape = data2NCHW.ForwardShape(data_shape.value()->values);
  ffi::Array<PrimExpr> weight_IOHW_shape = weight2IOHW.ForwardShape(weight_shape.value()->values);

  sym::Analyzer analyzer = ctx->GetAnalyzer();
  PrimExpr input_channel_data = data_NCHW_shape[1];
  PrimExpr input_channel_kernel = weight_IOHW_shape[0];
  if (analyzer->CanProve(input_channel_data != input_channel_kernel)) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Conv2dTranspose expects the channel size of the data should equal to the input channel "
           "size of the weight. However, the data channel size is "
        << input_channel_data << " while the weight input channel size is " << input_channel_kernel;
  } else if (!analyzer->CanProveEqual(input_channel_data, input_channel_kernel)) {
    // Todo(relax-team): Trust the input shape at this moment, and revisit
    // this condition with runtime shape check
  }
  if (analyzer->CanProve(floormod(input_channel_kernel, attrs->groups) != 0)) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Conv2dTranspose expects the number of input channels to be divisible by "
           "the number of groups. However, the number of input channels is "
        << input_channel_kernel << " while the number of groups is " << attrs->groups;
  } else if (!analyzer->CanProveEqual(floormod(input_channel_kernel, attrs->groups), 0)) {
    // Todo(relax-team): Trust the input shape at this moment, and revisit
    // this condition with runtime shape check
  }
  if (attrs->output_padding[0] >= attrs->strides[0] ||
      attrs->output_padding[1] >= attrs->strides[1]) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Conv2dTranspose expects the output padding less than the strides, but the "
           "output padding is"
        << attrs->output_padding << " while the strides are" << attrs->strides;
  }

  PrimExpr input_h = data_NCHW_shape[2];
  PrimExpr input_w = data_NCHW_shape[3];
  PrimExpr kernel_h = weight_IOHW_shape[2];
  PrimExpr kernel_w = weight_IOHW_shape[3];
  PrimExpr padding_h = IntImm::Int32(attrs->padding[0]) + IntImm::Int32(attrs->padding[2]);
  PrimExpr padding_w = IntImm::Int32(attrs->padding[1]) + IntImm::Int32(attrs->padding[3]);

  std::vector<PrimExpr> out_NCHW_shape;
  out_NCHW_shape.reserve(4);
  out_NCHW_shape.push_back(data_NCHW_shape[0]);
  out_NCHW_shape.push_back(weight_IOHW_shape[1] * attrs->groups);

  PrimExpr out_h = (input_h - 1) * IntImm::Int32(attrs->strides[0]) - padding_h +
                   IntImm::Int32(attrs->dilation[0]) * (kernel_h - 1) +
                   IntImm::Int32(attrs->output_padding[0]) + 1;
  PrimExpr out_w = (input_w - 1) * IntImm::Int32(attrs->strides[1]) - padding_w +
                   IntImm::Int32(attrs->dilation[1]) * (kernel_w - 1) +
                   IntImm::Int32(attrs->output_padding[1]) + 1;
  out_NCHW_shape.push_back(analyzer->Simplify(out_h));
  out_NCHW_shape.push_back(analyzer->Simplify(out_w));

  ffi::Array<PrimExpr> out_shape = out2NCHW.BackwardShape(out_NCHW_shape);
  return TensorType(ShapeExpr(out_shape), out_dtype, vdevice);
}

InferLayoutOutput InferLayoutConv2dTranspose(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  const auto* attrs = call->attrs.as<Conv2DTransposeAttrs>();
  LayoutDecision data_layout = GetLayoutDecision(var_layout_map, call->args[0]);
  LayoutDecision weight_layout = GetLayoutDecision(var_layout_map, call->args[1]);
  LayoutDecision output_layout(ffi::UnsafeInit{});
  ffi::ObjectPtr<Conv2DTransposeAttrs> new_attrs = ffi::make_object<Conv2DTransposeAttrs>(*attrs);

  auto it = desired_layouts.find("relax.nn.conv2d_transpose");
  if (it != desired_layouts.end()) {
    SLayout desired_data_layout = (*it).second[0];
    SLayout desired_weight_layout = (*it).second[1];
    SLayout desired_output_layout = (*it).second.size() == 3 ? (*it).second[2] : (*it).second[0];

    SLayout input_layout = SLayout(attrs->data_layout);
    SLayout kernel_layout = SLayout(attrs->kernel_layout);
    SLayout out_layout = SLayout(attrs->out_layout);

    if (desired_data_layout.ndim_primal() == input_layout.ndim() &&
        desired_weight_layout.ndim_primal() == kernel_layout.ndim() &&
        desired_output_layout.ndim_primal() == out_layout.ndim()) {
      data_layout =
          TransposeLike(InitialLayout(4), SLayout::Create(attrs->data_layout), desired_data_layout);
      weight_layout = TransposeLike(InitialLayout(4), SLayout::Create(attrs->kernel_layout),
                                    desired_weight_layout);
      output_layout = TransposeLike(InitialLayout(4), SLayout::Create(attrs->out_layout),
                                    desired_output_layout);
      new_attrs->data_layout = (*it).second[0];
      new_attrs->kernel_layout = (*it).second[1];
      new_attrs->out_layout = (*it).second.size() == 3 ? (*it).second[2] : (*it).second[0];
      return InferLayoutOutput({data_layout, weight_layout}, {output_layout}, Attrs(new_attrs));
    } else {
      auto data_si = GetType(call->args[0]);
      auto kernel_si = GetType(call->args[1]);
      TensorType data_ty = data_si.as<TensorType>().value();
      TensorType kernel_ty = kernel_si.as<TensorType>().value();
      ffi::Optional<ShapeExpr> data_shape =
          ffi::GetRef<ShapeExpr>(data_ty->shape.as<ShapeExprObj>());
      ffi::Optional<ShapeExpr> kernel_shape =
          ffi::GetRef<ShapeExpr>(kernel_ty->shape.as<ShapeExprObj>());

      bool can_data_proved =
          CanProveLayoutTransform(input_layout, desired_data_layout, data_shape.value()->values);
      bool can_kernel_proved = CanProveLayoutTransform(kernel_layout, desired_weight_layout,
                                                       kernel_shape.value()->values);

      if (can_data_proved && can_kernel_proved) {
        data_layout = TransposeSubLayoutLike(InitialLayout(4), input_layout, desired_data_layout);
        weight_layout =
            TransposeSubLayoutLike(InitialLayout(4), kernel_layout, desired_weight_layout);
        output_layout = TransposeSubLayoutLike(InitialLayout(4), out_layout, desired_output_layout);
        new_attrs->data_layout = (*it).second[0];
        new_attrs->kernel_layout = (*it).second[1];
        new_attrs->out_layout = (*it).second.size() == 3 ? (*it).second[2] : (*it).second[0];
        return InferLayoutOutput({data_layout, weight_layout}, {output_layout}, Attrs(new_attrs));
      } else {
        data_layout = LayoutDecision(InitialLayout(4));
        weight_layout = LayoutDecision(InitialLayout(4));
      }
    }
  }

  output_layout = data_layout;
  new_attrs->data_layout =
      TransposeLike(SLayout::Create(attrs->data_layout), InitialLayout(4), data_layout->layout)
          .name();
  new_attrs->kernel_layout =
      TransposeLike(SLayout::Create(attrs->kernel_layout), InitialLayout(4), weight_layout->layout)
          .name();
  new_attrs->out_layout =
      TransposeLike(SLayout::Create(attrs->out_layout), InitialLayout(4), output_layout->layout)
          .name();
  return InferLayoutOutput({data_layout, weight_layout}, {output_layout}, Attrs(new_attrs));
}

Call InferMixedPrecisionConv2dTranspose(const Call& call, DLDataType out_dtype) {
  const auto* conv2d_transpose_attrs = call->attrs.as<Conv2DTransposeAttrs>();
  return conv2d_transpose(call->args[0], call->args[1], conv2d_transpose_attrs->strides,
                          conv2d_transpose_attrs->padding, conv2d_transpose_attrs->output_padding,
                          conv2d_transpose_attrs->dilation, conv2d_transpose_attrs->groups,
                          conv2d_transpose_attrs->data_layout,
                          conv2d_transpose_attrs->kernel_layout, conv2d_transpose_attrs->out_layout,
                          out_dtype)
      .as_or_throw<Call>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.nn.conv2d_transpose")
      .signature(
          sig::arg("data", "The input tensor."), sig::arg("weight", "The weight tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."),
          sig::call_attrs<Conv2DTransposeAttrs>())
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeConv2dTranspose)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutConv2dTranspose)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kAlways)
      .set_attr(tvm::relax::op_attr::kInferMixedPrecision, InferMixedPrecisionConv2dTranspose)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.nn.conv3d_transpose */

Expr conv3d_transpose(Expr data, Expr weight, ffi::Array<int64_t> strides,
                      ffi::Array<int64_t> padding, ffi::Array<int64_t> output_padding,
                      ffi::Array<int64_t> dilation, int groups, ffi::String data_layout,
                      ffi::String kernel_layout, ffi::Optional<ffi::String> out_layout,
                      ffi::Optional<DLDataType> out_dtype) {
  padding = GetCompletePadding3D(std::move(padding));
  if (output_padding.size() == 1) {
    output_padding.push_back(output_padding[0]);
    output_padding.push_back(output_padding[0]);
  }
  if (strides.size() == 1) {
    strides.push_back(strides[0]);
    strides.push_back(strides[0]);
  }
  if (dilation.size() == 1) {
    dilation.push_back(dilation[0]);
    dilation.push_back(dilation[0]);
  }

  TVM_FFI_ICHECK_GT(groups, 0)
      << "The number of groups in convolution is expected to be positive. However, "
         "the given number of groups is "
      << groups;
  TVM_FFI_ICHECK_EQ(output_padding.size(), 3)
      << "The input output_padding length is expected to be 3. "
         "However, the given output_padding is "
      << output_padding;
  TVM_FFI_ICHECK_EQ(strides.size(), 3)
      << "The input strides length is expected to be 3. However, the given strides is " << strides;
  TVM_FFI_ICHECK_EQ(dilation.size(), 3)
      << "The input dilation length is expected to be 3. However, the given dilation is "
      << dilation;

  auto attrs = ffi::make_object<Conv3DTransposeAttrs>();
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->output_padding = std::move(output_padding);
  attrs->dilation = std::move(dilation);
  attrs->groups = groups;
  attrs->data_layout = data_layout;
  attrs->kernel_layout = std::move(kernel_layout);
  attrs->out_layout = out_layout.value_or(data_layout);
  attrs->out_dtype = out_dtype;
  const Op op = Op::Get("relax.nn.conv3d_transpose");
  return Call(Type::Missing(), op, {data, weight}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.nn.conv3d_transpose", conv3d_transpose);
}

Type InferTypeConv3dTranspose(const Call& call, const BlockBuilder& ctx) {
  ffi::Array<TensorType> input_ty = GetInputTensorType(call, ctx);
  TensorType data_ty = input_ty[0];
  TensorType weight_ty = input_ty[1];

  const auto* attrs = call->attrs.as<Conv3DTransposeAttrs>();
  auto [data_layout, data2NCDHW] = CheckTensorLayout(call, ctx, attrs->data_layout,  //
                                                     /*tgt_layout=*/"NCDHW",         //
                                                     /*tensor_name=*/"data");
  auto [weight_layout, weight2IODHW] = CheckTensorLayout(call, ctx, attrs->kernel_layout,  //
                                                         /*tgt_layout=*/"IODHW",           //
                                                         /*tensor_name=*/"kernel");
  auto [out_layout, out2NCDHW] = CheckTensorLayout(call, ctx, attrs->out_layout,  //
                                                   /*tgt_layout=*/"NCDHW",        //
                                                   /*tensor_name=*/"output");

  ffi::Optional<ShapeExpr> data_shape =
      CheckNdimPerLayoutAndGetShape(call, ctx, data_ty, data_layout);
  ffi::Optional<ShapeExpr> weight_shape =
      CheckNdimPerLayoutAndGetShape(call, ctx, weight_ty, weight_layout);

  ffi::Optional<PrimType> out_dtype =
      attrs->out_dtype.has_value() ? PrimType(attrs->out_dtype.value())
                                   : InferBinaryArithOpOutDtype(call, ctx, data_ty, weight_ty);
  ffi::Optional<VDevice> vdevice = InferBinaryArithOpOutVDevice(call, ctx, data_ty, weight_ty);
  if (!data_shape.has_value() || !weight_shape.has_value()) {
    return TensorType(out_dtype, out_layout.ndim(), vdevice);
  }

  ffi::Array<PrimExpr> data_NCDHW_shape = data2NCDHW.ForwardShape(data_shape.value()->values);
  ffi::Array<PrimExpr> weight_IODHW_shape = weight2IODHW.ForwardShape(weight_shape.value()->values);

  sym::Analyzer analyzer = ctx->GetAnalyzer();
  PrimExpr input_channel_data = data_NCDHW_shape[1];
  PrimExpr input_channel_kernel = weight_IODHW_shape[0];
  if (analyzer->CanProve(input_channel_data != input_channel_kernel)) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Conv3dTranspose expects the channel size of the data should equal to the input channel "
           "size of the weight. However, the data channel size is "
        << input_channel_data << " while the weight input channel size is " << input_channel_kernel;
  } else if (!analyzer->CanProveEqual(input_channel_data, input_channel_kernel)) {
    // Todo(relax-team): Trust the input shape at this moment, and revisit
    // this condition with runtime shape check
  }
  if (analyzer->CanProve(floormod(input_channel_kernel, attrs->groups) != 0)) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Conv3dTranspose expects the number of input channels to be divisible by "
           "the number of groups. However, the number of input channels is "
        << input_channel_kernel << " while the number of groups is " << attrs->groups;
  } else if (!analyzer->CanProveEqual(floormod(input_channel_kernel, attrs->groups), 0)) {
    // Todo(relax-team): Trust the input shape at this moment, and revisit
    // this condition with runtime shape check
  }
  if (attrs->output_padding[0] >= attrs->strides[0] ||
      attrs->output_padding[1] >= attrs->strides[1] ||
      attrs->output_padding[2] >= attrs->strides[2]) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Conv3dTranspose expects the output padding less than the strides, but the "
           "output padding is"
        << attrs->output_padding << " while the strides are" << attrs->strides;
  }

  PrimExpr input_d = data_NCDHW_shape[2];
  PrimExpr input_h = data_NCDHW_shape[3];
  PrimExpr input_w = data_NCDHW_shape[4];
  PrimExpr kernel_d = weight_IODHW_shape[2];
  PrimExpr kernel_h = weight_IODHW_shape[3];
  PrimExpr kernel_w = weight_IODHW_shape[4];
  PrimExpr padding_d = IntImm::Int32(attrs->padding[0]) + IntImm::Int32(attrs->padding[3]);
  PrimExpr padding_h = IntImm::Int32(attrs->padding[1]) + IntImm::Int32(attrs->padding[4]);
  PrimExpr padding_w = IntImm::Int32(attrs->padding[2]) + IntImm::Int32(attrs->padding[5]);

  std::vector<PrimExpr> out_NCDHW_shape;
  out_NCDHW_shape.reserve(5);
  out_NCDHW_shape.push_back(data_NCDHW_shape[0]);
  out_NCDHW_shape.push_back(weight_IODHW_shape[1] * attrs->groups);

  PrimExpr out_d = (input_d - 1) * IntImm::Int32(attrs->strides[0]) - padding_d +
                   IntImm::Int32(attrs->dilation[0]) * (kernel_d - 1) +
                   IntImm::Int32(attrs->output_padding[0]) + 1;
  PrimExpr out_h = (input_h - 1) * IntImm::Int32(attrs->strides[1]) - padding_h +
                   IntImm::Int32(attrs->dilation[1]) * (kernel_h - 1) +
                   IntImm::Int32(attrs->output_padding[1]) + 1;
  PrimExpr out_w = (input_w - 1) * IntImm::Int32(attrs->strides[2]) - padding_w +
                   IntImm::Int32(attrs->dilation[2]) * (kernel_w - 1) +
                   IntImm::Int32(attrs->output_padding[2]) + 1;
  out_NCDHW_shape.push_back(analyzer->Simplify(out_d));
  out_NCDHW_shape.push_back(analyzer->Simplify(out_h));
  out_NCDHW_shape.push_back(analyzer->Simplify(out_w));

  ffi::Array<PrimExpr> out_shape = out2NCDHW.BackwardShape(out_NCDHW_shape);
  return TensorType(ShapeExpr(out_shape), out_dtype, vdevice);
}

InferLayoutOutput InferLayoutConv3dTranspose(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  const auto* attrs = call->attrs.as<Conv3DTransposeAttrs>();
  LayoutDecision data_layout = GetLayoutDecision(var_layout_map, call->args[0]);
  LayoutDecision weight_layout = GetLayoutDecision(var_layout_map, call->args[1]);
  LayoutDecision output_layout(ffi::UnsafeInit{});
  ffi::ObjectPtr<Conv3DTransposeAttrs> new_attrs = ffi::make_object<Conv3DTransposeAttrs>(*attrs);

  auto it = desired_layouts.find("relax.nn.conv3d_transpose");
  if (it != desired_layouts.end()) {
    SLayout desired_data_layout = (*it).second[0];
    SLayout desired_weight_layout = (*it).second[1];
    SLayout desired_output_layout = (*it).second.size() == 3 ? (*it).second[2] : (*it).second[0];

    SLayout input_layout = SLayout(attrs->data_layout);
    SLayout kernel_layout = SLayout(attrs->kernel_layout);
    SLayout out_layout = SLayout(attrs->out_layout);

    if (desired_data_layout.ndim_primal() == input_layout.ndim() &&
        desired_weight_layout.ndim_primal() == kernel_layout.ndim() &&
        desired_output_layout.ndim_primal() == out_layout.ndim()) {
      data_layout =
          TransposeLike(InitialLayout(5), SLayout::Create(attrs->data_layout), desired_data_layout);
      weight_layout = TransposeLike(InitialLayout(5), SLayout::Create(attrs->kernel_layout),
                                    desired_weight_layout);
      output_layout = TransposeLike(InitialLayout(5), SLayout::Create(attrs->out_layout),
                                    desired_output_layout);
      new_attrs->data_layout = (*it).second[0];
      new_attrs->kernel_layout = (*it).second[1];
      new_attrs->out_layout = (*it).second.size() == 3 ? (*it).second[2] : (*it).second[0];
      return InferLayoutOutput({data_layout, weight_layout}, {output_layout}, Attrs(new_attrs));
    } else {
      auto data_si = GetType(call->args[0]);
      auto kernel_si = GetType(call->args[1]);
      TensorType data_ty = data_si.as<TensorType>().value();
      TensorType kernel_ty = kernel_si.as<TensorType>().value();
      ffi::Optional<ShapeExpr> data_shape =
          ffi::GetRef<ShapeExpr>(data_ty->shape.as<ShapeExprObj>());
      ffi::Optional<ShapeExpr> kernel_shape =
          ffi::GetRef<ShapeExpr>(kernel_ty->shape.as<ShapeExprObj>());

      bool can_data_proved =
          CanProveLayoutTransform(input_layout, desired_data_layout, data_shape.value()->values);
      bool can_kernel_proved = CanProveLayoutTransform(kernel_layout, desired_weight_layout,
                                                       kernel_shape.value()->values);

      if (can_data_proved && can_kernel_proved) {
        data_layout = TransposeSubLayoutLike(InitialLayout(5), input_layout, desired_data_layout);
        weight_layout =
            TransposeSubLayoutLike(InitialLayout(5), kernel_layout, desired_weight_layout);
        output_layout = TransposeSubLayoutLike(InitialLayout(5), out_layout, desired_output_layout);
        new_attrs->data_layout = (*it).second[0];
        new_attrs->kernel_layout = (*it).second[1];
        new_attrs->out_layout = (*it).second.size() == 3 ? (*it).second[2] : (*it).second[0];
        return InferLayoutOutput({data_layout, weight_layout}, {output_layout}, Attrs(new_attrs));
      } else {
        data_layout = LayoutDecision(InitialLayout(5));
        weight_layout = LayoutDecision(InitialLayout(5));
      }
    }
  }

  output_layout = data_layout;
  new_attrs->data_layout =
      TransposeLike(SLayout::Create(attrs->data_layout), InitialLayout(5), data_layout->layout)
          .name();
  new_attrs->kernel_layout =
      TransposeLike(SLayout::Create(attrs->kernel_layout), InitialLayout(5), weight_layout->layout)
          .name();
  new_attrs->out_layout =
      TransposeLike(SLayout::Create(attrs->out_layout), InitialLayout(5), output_layout->layout)
          .name();
  return InferLayoutOutput({data_layout, weight_layout}, {output_layout}, Attrs(new_attrs));
}

Call InferMixedPrecisionConv3dTranspose(const Call& call, DLDataType out_dtype) {
  const auto* conv3d_transpose_attrs = call->attrs.as<Conv3DTransposeAttrs>();
  return conv3d_transpose(call->args[0], call->args[1], conv3d_transpose_attrs->strides,
                          conv3d_transpose_attrs->padding, conv3d_transpose_attrs->output_padding,
                          conv3d_transpose_attrs->dilation, conv3d_transpose_attrs->groups,
                          conv3d_transpose_attrs->data_layout,
                          conv3d_transpose_attrs->kernel_layout, conv3d_transpose_attrs->out_layout,
                          out_dtype)
      .as_or_throw<Call>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.nn.conv3d_transpose")
      .signature(
          sig::arg("data", "The input tensor."), sig::arg("weight", "The weight tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."),
          sig::call_attrs<Conv3DTransposeAttrs>())
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeConv3dTranspose)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutConv3dTranspose)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kAlways)
      .set_attr(tvm::relax::op_attr::kInferMixedPrecision, InferMixedPrecisionConv3dTranspose)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

}  // namespace relax
}  // namespace tvm
