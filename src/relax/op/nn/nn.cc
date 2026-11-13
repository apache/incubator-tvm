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

#include "nn.h"

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/relax/op_attr.h>

#include <utility>
#include <vector>

namespace tvm {
namespace relax {

void SoftmaxAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<SoftmaxAttrs>().def_ro("axis", &SoftmaxAttrs::axis,
                                         "The axis to sum over when computing softmax.",
                                         refl::DefaultValue(-1));
}

void LeakyReluAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<LeakyReluAttrs>().def_ro(
      "alpha", &LeakyReluAttrs::alpha, "The slope of the negative part.", refl::DefaultValue(0.01));
}

void SoftplusAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<SoftplusAttrs>()
      .def_ro("beta", &SoftplusAttrs::beta,
              "Scaling factor controlling the sharpness of the Softplus transition.",
              refl::DefaultValue(1.0))
      .def_ro("threshold", &SoftplusAttrs::threshold,
              "Value determining when to use linear approximation for numerical stability.",
              refl::DefaultValue(20.0));
}

void PReluAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<PReluAttrs>().def_ro("axis", &PReluAttrs::axis,
                                       "The axis along which the alpha values are applied.",
                                       refl::DefaultValue(1));
}

void BatchNormAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<BatchNormAttrs>()
      .def_ro("axis", &BatchNormAttrs::axis, "The axis along which the normalization is applied.")
      .def_ro("epsilon", &BatchNormAttrs::epsilon,
              "Small float added to variance to avoid dividing by zero", refl::DefaultValue(1e-05))
      .def_ro("center", &BatchNormAttrs::center,
              "Indicating if the beta offset will be added to the normalized tensor.",
              refl::DefaultValue(true))
      .def_ro("scale", &BatchNormAttrs::scale, "Indicating if the gamma scale will be multiplied.",
              refl::DefaultValue(true))
      .def_ro("momentum", &BatchNormAttrs::momentum,
              "The value used for the moving_mean and moving_var update.", refl::DefaultValue(0.1))
      .def_ro("training", &BatchNormAttrs::training,
              "Whether we are training (i.e., not in eval mode).", refl::DefaultValue(true));
}

void LayerNormAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<LayerNormAttrs>()
      .def_ro("axes", &LayerNormAttrs::axes,
              "The axes that along which the normalization is applied.")
      .def_ro("epsilon", &LayerNormAttrs::epsilon,
              "Small float added to variance to avoid dividing by zero", refl::DefaultValue(1e-05))
      .def_ro("center", &LayerNormAttrs::center,
              "Indicating if the beta offset will be added to the normalized tensor.",
              refl::DefaultValue(true))
      .def_ro("scale", &LayerNormAttrs::scale, "Indicating if the gamma scale will be multiplied.",
              refl::DefaultValue(true));
}

void GroupNormAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<GroupNormAttrs>()
      .def_ro("num_groups", &GroupNormAttrs::num_groups,
              "The number of groups to separate the channels into.")
      .def_ro("channel_axis", &GroupNormAttrs::channel_axis,
              "The axis that represents the channel.")
      .def_ro(
          "axes", &GroupNormAttrs::axes,
          "The axes that along which the normalization is applied (excluding the channel axis).")
      .def_ro("epsilon", &GroupNormAttrs::epsilon,
              "Small float added to variance to avoid dividing by zero", refl::DefaultValue(1e-05))
      .def_ro("center", &GroupNormAttrs::center,
              "Indicating if the beta offset will be added to the normalized tensor.",
              refl::DefaultValue(true))
      .def_ro("scale", &GroupNormAttrs::scale, "Indicating if the gamma scale will be multiplied.",
              refl::DefaultValue(true));
}

void InstanceNormAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<InstanceNormAttrs>()
      .def_ro("channel_axis", &InstanceNormAttrs::channel_axis,
              "The axis that represents the channel.")
      .def_ro("axes", &InstanceNormAttrs::axes,
              "The axes that along which the normalization is applied.")
      .def_ro("epsilon", &InstanceNormAttrs::epsilon,
              "Small float added to variance to avoid dividing by zero", refl::DefaultValue(1e-05))
      .def_ro("center", &InstanceNormAttrs::center,
              "Indicating if the beta offset will be added to the normalized tensor.",
              refl::DefaultValue(true))
      .def_ro("scale", &InstanceNormAttrs::scale,
              "Indicating if the gamma scale will be multiplied.", refl::DefaultValue(true));
}

void RMSNormAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<RMSNormAttrs>()
      .def_ro("axes", &RMSNormAttrs::axes,
              "The axes that along which the normalization is applied.",
              refl::DefaultValue(ffi::Array<int64_t>{-1}))
      .def_ro("epsilon", &RMSNormAttrs::epsilon,
              "Small float added to variance to avoid dividing by zero", refl::DefaultValue(1e-05));
}

void NLLLossAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<NLLLossAttrs>()
      .def_ro("reduction", &NLLLossAttrs::reduction,
              "The reduction method to apply to the output. Can be"
              "'none', 'mean' or 'sum'.",
              refl::DefaultValue("mean"))
      .def_ro("ignore_index", &NLLLossAttrs::ignore_index, "The target value to ignore.",
              refl::DefaultValue(-100));
}

void DropoutAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<DropoutAttrs>().def_ro(
      "rate", &DropoutAttrs::rate,
      "Fraction of the input that gets dropped out during training time", refl::DefaultValue(0.5));
}

void PadAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<PadAttrs>()
      .def_ro("pad_width", &PadAttrs::pad_width,
              "Number of values padded to the edges of each axis, "
              "in the format of (before_1, after_1, ..., before_N, after_N)")
      .def_ro("pad_value", &PadAttrs::pad_value, "The value to fill in padded area with",
              refl::DefaultValue(0.0))
      .def_ro("pad_mode", &PadAttrs::pad_mode,
              "Padding type to use. \"constant\" pads with constant_value, "
              "\"edge\" pads using the edge values of the input array, "
              "\"reflect\" pads by reflecting values with respect to the edges.",
              refl::DefaultValue("constant"));
}

void PixelShuffleAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<PixelShuffleAttrs>().def_ro("upscale_factor", &PixelShuffleAttrs::upscale_factor,
                                              "Scale factor for spatial upsampling.");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  SoftmaxAttrs::RegisterReflection();
  LeakyReluAttrs::RegisterReflection();
  SoftplusAttrs::RegisterReflection();
  PReluAttrs::RegisterReflection();
  BatchNormAttrs::RegisterReflection();
  LayerNormAttrs::RegisterReflection();
  GroupNormAttrs::RegisterReflection();
  InstanceNormAttrs::RegisterReflection();
  RMSNormAttrs::RegisterReflection();
  NLLLossAttrs::RegisterReflection();
  DropoutAttrs::RegisterReflection();
  PadAttrs::RegisterReflection();
  PixelShuffleAttrs::RegisterReflection();
}

/* relax.nn.relu */
Expr relu(Expr x) {
  static const Op op = Op::Get("relax.nn.relu");
  return Call(Type::Missing(), op, {std::move(x)}, std::nullopt, {});
}

Expr gelu(Expr x) {
  static const Op op = Op::Get("relax.nn.gelu");
  return Call(Type::Missing(), op, {std::move(x)}, std::nullopt, {});
}

Expr gelu_tanh(Expr x) {
  static const Op op = Op::Get("relax.nn.gelu_tanh");
  return Call(Type::Missing(), op, {std::move(x)}, std::nullopt, {});
}

Expr selu(Expr x) {
  static const Op op = Op::Get("relax.nn.selu");
  return Call(Type::Missing(), op, {std::move(x)}, std::nullopt, {});
}

Expr silu(Expr x) {
  static const Op op = Op::Get("relax.nn.silu");
  return Call(Type::Missing(), op, {std::move(x)}, std::nullopt, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::ffi::reflection::GlobalDef().def("relax.op.nn.relu", relu);

  OpDef("relax.nn.relu")
      .signature(
          sig::arg("x", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_native_fn_attr<&InferTypeUnaryArithContextFree<false>>(tvm::op_attr::kInferType);

  /* relax.nn.gelu */

  tvm::ffi::reflection::GlobalDef().def("relax.op.nn.gelu", gelu);

  OpDef("relax.nn.gelu")
      .signature(
          sig::arg("x", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_native_fn_attr<&InferTypeUnaryArithContextFree<true>>(tvm::op_attr::kInferType);

  /* relax.nn.gelu_tanh */

  tvm::ffi::reflection::GlobalDef().def("relax.op.nn.gelu_tanh", gelu_tanh);

  OpDef("relax.nn.gelu_tanh")
      .signature(
          sig::arg("x", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_native_fn_attr<&InferTypeUnaryArithContextFree<true>>(tvm::op_attr::kInferType);

  /* relax.nn.selu */

  tvm::ffi::reflection::GlobalDef().def("relax.op.nn.selu", selu);

  OpDef("relax.nn.selu")
      .signature(
          sig::arg("x", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_native_fn_attr<&InferTypeUnaryArithContextFree<true>>(tvm::op_attr::kInferType);

  /* relax.nn.silu */

  tvm::ffi::reflection::GlobalDef().def("relax.op.nn.silu", silu);

  OpDef("relax.nn.silu")
      .signature(
          sig::arg("x", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_native_fn_attr<&InferTypeUnaryArithContextFree<true>>(tvm::op_attr::kInferType);
}

/* relax.nn.leakyrelu */

Expr leakyrelu(Expr data, double alpha) {
  auto attrs = ffi::make_object<LeakyReluAttrs>();
  attrs->alpha = alpha;
  static const Op op = Op::Get("relax.nn.leakyrelu");
  return Call(Type::Missing(), op, {data}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.nn.leakyrelu", leakyrelu);

  OpDef("relax.nn.leakyrelu")
      .signature(
          sig::arg("data", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."),
          sig::call_attrs<LeakyReluAttrs>())
      .set_native_fn_attr<&InferTypeUnaryArithContextFree</*require_float_dtype=*/true>>(
          tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.nn.softplus */

Expr softplus(Expr data, double beta, double threshold) {
  auto attrs = ffi::make_object<SoftplusAttrs>();
  attrs->beta = beta;
  attrs->threshold = threshold;
  static const Op op = Op::Get("relax.nn.softplus");
  return Call(Type::Missing(), op, {data}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.nn.softplus", softplus);

  OpDef("relax.nn.softplus")
      .signature(
          sig::arg("data", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."),
          sig::call_attrs<SoftplusAttrs>())
      .set_native_fn_attr<&InferTypeUnaryArithContextFree</*require_float_dtype=*/true>>(
          tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.nn.prelu */

Expr prelu(Expr data, Expr alpha, int axis = 1) {
  auto attrs = ffi::make_object<PReluAttrs>();
  attrs->axis = axis;
  static const Op op = Op::Get("relax.nn.prelu");
  return Call(Type::Missing(), op, {data, alpha}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.nn.prelu", prelu);
}

Type InferTypePRelu(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  if (RequiresTensorInputNormalization(call)) return Type::Missing();
  TensorType data_ty = GetUnaryInputTensorType(call);
  if (data_ty->IsUnknownNdim()) {
    return data_ty;
  }
  // PRelu preserves the old float-kind check; vector lanes are irrelevant to this check.
  if (!data_ty->IsUnknownDtype() && !data_ty->dtype.value().MatchesCode(DLDataTypeCode::kDLFloat)) {
    TVM_FFI_VISIT_THROW(TypeError, call) << "Prelu requires the input tensor to have float "
                                            "dtype. However, the given input dtype is "
                                         << data_ty->dtype;
  }
  const auto* attrs = call->attrs.as<PReluAttrs>();
  NormalizeAxis(call, data_ty->ndim, attrs->axis);

  return data_ty;
}

InferLayoutOutput InferLayoutPRelu(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  TVM_FFI_ICHECK(NoDesiredLayout(call, desired_layouts));
  const auto* attrs = call->attrs.as<PReluAttrs>();
  TVM_FFI_ICHECK(attrs) << "Invalid Call";

  LayoutDecision layout = GetLayoutDecision(var_layout_map, call->args[0]);

  // TODO(Siva): We could handle if the axis is not the sub indexed one.
  if ((layout->layout.has_value() ? layout->layout.value().ndim() : 0) !=
      (layout->layout.has_value() ? layout->layout.value().ndim_primal() : 0)) {
    const auto* tensor_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
    TVM_FFI_ICHECK(tensor_ty != nullptr) << "Invalid Call";
    TVM_FFI_ICHECK(!tensor_ty->IsUnknownNdim()) << "Only support static ndim for now";
    int ndim = tensor_ty->ndim;
    layout = LayoutDecision(InitialLayout(ndim));
  }

  ffi::ObjectPtr<PReluAttrs> new_attrs = ffi::make_object<PReluAttrs>(*attrs);
  new_attrs->axis = FindAxis(layout->layout.value(), attrs->axis);

  LayoutDecision alpha_layout = GetLayoutDecision(var_layout_map, call->args[1]);
  return InferLayoutOutput({layout, alpha_layout}, {layout}, Attrs(new_attrs));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.nn.prelu")
      .signature(sig::arg("data", "The input tensor."),
                 sig::arg("alpha", "The channel-wise learnable slope."),
                 sig::call_attrs<PReluAttrs>())
      .set_native_fn_attr<&InferTypePRelu>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutPRelu)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.nn.softmax */

Expr softmax(Expr data, int axis) {
  auto attrs = ffi::make_object<SoftmaxAttrs>();
  attrs->axis = axis;
  static const Op op = Op::Get("relax.nn.softmax");
  return Call(Type::Missing(), op, {data}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.nn.softmax", softmax);
}

Type InferTypeSoftmax(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  if (RequiresTensorInputNormalization(call)) return Type::Missing();
  TensorType data_ty = GetUnaryInputTensorType(call);
  if (data_ty->IsUnknownNdim()) {
    return data_ty;
  }
  if (!data_ty->IsUnknownDtype()) {
    PrimType data_dtype = data_ty->dtype.value();
    // Softmax only requires a floating element kind; lane encoding is irrelevant to the check.
    if (!data_dtype.MatchesCode(kDLFloat, kDLBfloat)) {
      TVM_FFI_VISIT_THROW(TypeError, call) << "Softmax requires the input tensor to have float "
                                              "dtype. However, the given input dtype is "
                                           << data_ty->dtype;
    }
  }
  const auto* attrs = call->attrs.as<SoftmaxAttrs>();
  NormalizeAxis(call, data_ty->ndim, attrs->axis);

  return data_ty;
}

InferLayoutOutput InferLayoutSoftmax(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  TVM_FFI_ICHECK(NoDesiredLayout(call, desired_layouts));
  const auto* attrs = call->attrs.as<SoftmaxAttrs>();
  TVM_FFI_ICHECK(attrs) << "Invalid Call";

  LayoutDecision layout = GetLayoutDecision(var_layout_map, call->args[0]);

  // TODO(Siva): We could handle if the axis is not the sub indexed one.
  if ((layout->layout.has_value() ? layout->layout.value().ndim() : 0) !=
      (layout->layout.has_value() ? layout->layout.value().ndim_primal() : 0)) {
    const auto* tensor_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
    TVM_FFI_ICHECK(tensor_ty != nullptr) << "Invalid Call";
    TVM_FFI_ICHECK(!tensor_ty->IsUnknownNdim()) << "Only support static ndim for now";
    int ndim = tensor_ty->ndim;
    layout = LayoutDecision(InitialLayout(ndim));
  }

  ffi::ObjectPtr<SoftmaxAttrs> new_attrs = ffi::make_object<SoftmaxAttrs>(*attrs);
  new_attrs->axis = FindAxis(layout->layout.value(), attrs->axis);
  return InferLayoutOutput({layout}, {layout}, Attrs(new_attrs));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.nn.softmax")
      .signature(sig::arg("data", "The input tensor."), sig::call_attrs<SoftmaxAttrs>())
      .set_native_fn_attr<&InferTypeSoftmax>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutSoftmax)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.nn.log_softmax */
Expr log_softmax(Expr data, int axis) {
  auto attrs = ffi::make_object<SoftmaxAttrs>();
  attrs->axis = axis;
  static const Op op = Op::Get("relax.nn.log_softmax");
  return Call(Type::Missing(), op, {data}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.nn.log_softmax", log_softmax);

  OpDef("relax.nn.log_softmax")
      .signature(sig::arg("data", "The input tensor."), sig::call_attrs<SoftmaxAttrs>())
      .set_native_fn_attr<&InferTypeSoftmax>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.nn.pad */

Expr pad(Expr data, ffi::Array<int64_t> pad_width, ffi::String pad_mode, double pad_value) {
  auto attrs = ffi::make_object<PadAttrs>();
  attrs->pad_width = std::move(pad_width);
  attrs->pad_mode = std::move(pad_mode);
  attrs->pad_value = pad_value;
  static const Op op = Op::Get("relax.nn.pad");
  return Call(Type::Missing(), op, {data}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.nn.pad", pad);
}

Type InferTypePad(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  ffi::Array<TensorType> input_ty = GetInputTensorType(call);
  const auto* attrs = call->attrs.as<PadAttrs>();
  int ndim = input_ty[0]->ndim;
  ffi::Array<int64_t> pad_width = attrs->pad_width;
  TVM_FFI_ICHECK(static_cast<int>(pad_width.size()) == 2 * ndim) << "Illegal pad_width";

  ffi::Array<PrimExpr> out_shape;
  if (input_ty[0]->shape.has_value()) {
    // Compute output shape by adding corresponding pad width to each axis.
    const auto* data_shape = input_ty[0]->shape.as<ShapeExprObj>();
    for (int i = 0; i < ndim; i++) {
      // Sum pad width for this axis.
      PrimExpr added_width = IntImm::Int64(pad_width[2 * i] + pad_width[(2 * i) + 1]);
      const PrimExpr current_width = data_shape->values[i];
      out_shape.push_back(current_width + added_width);
    }
  } else {
    // Shape isnt defined, best we can do is return ndim and dtype.
    return TensorType(input_ty[0]->dtype, ndim);
  }
  return TensorType(ShapeExpr(out_shape), input_ty[0]->dtype);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.nn.pad")
      .signature(sig::arg("data", "The input tensor."), sig::call_attrs<PadAttrs>())
      .set_native_fn_attr<&InferTypePad>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.nn.pixel_shuffle */

Expr pixel_shuffle(Expr data, int upscale_factor) {
  auto attrs = ffi::make_object<PixelShuffleAttrs>();
  attrs->upscale_factor = upscale_factor;
  static const Op op = Op::Get("relax.nn.pixel_shuffle");
  return Call(Type::Missing(), op, {data}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.nn.pixel_shuffle", pixel_shuffle);
}

Type InferTypePixelShuffle(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  ffi::Array<TensorType> input_ty = GetInputTensorType(call);
  const auto* attrs = call->attrs.as<PixelShuffleAttrs>();
  int r = attrs->upscale_factor;
  TVM_FFI_ICHECK_GT(r, 0) << "Upscale factor must be positive";

  const TensorType& input = input_ty[0];
  int ndim = input->ndim;
  TVM_FFI_ICHECK_GE(ndim, 3) << "PixelShuffle requires at least 3D input tensor";

  if (!input->shape.has_value()) {
    return TensorType(input->dtype, ndim);
  }

  const auto* shape = input->shape.as<ShapeExprObj>();
  ffi::Array<PrimExpr> in_shape = shape->values;

  int channel_idx = ndim - 3;
  int h_idx = ndim - 2;
  int w_idx = ndim - 1;

  PrimExpr c_in = in_shape[channel_idx];
  PrimExpr h_in = in_shape[h_idx];
  PrimExpr w_in = in_shape[w_idx];

  PrimExpr r_expr = IntImm::Int32(r);
  PrimExpr r_squared = r_expr * r_expr;

  const auto* c_in_imm = c_in.as<IntImmObj>();
  const auto* r2_imm = r_squared.as<IntImmObj>();

  TVM_FFI_ICHECK_EQ(c_in_imm->value % r2_imm->value, 0)
      << "Number of input channels must be divisible by the square of the upscale factor";

  // Output shape:
  ffi::Array<PrimExpr> out_shape;
  for (int i = 0; i < ndim; ++i) {
    if (i == channel_idx) {
      out_shape.push_back(c_in / r_squared);
    } else if (i == h_idx) {
      out_shape.push_back(h_in * r_expr);
    } else if (i == w_idx) {
      out_shape.push_back(w_in * r_expr);
    } else {
      out_shape.push_back(in_shape[i]);
    }
  }

  return TensorType(ShapeExpr(out_shape), input->dtype);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.nn.pixel_shuffle")
      .signature(sig::arg("data", "The input tensor."), sig::call_attrs<PixelShuffleAttrs>())
      .set_native_fn_attr<&InferTypePixelShuffle>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.nn.batchnorm */
bool NormCheckDtypeAndShape(const Call& call, const BlockBuilder& ctx,
                            const ffi::Array<TensorType>& input_ty, ffi::Array<int64_t> axes) {
  Op op = call->op.as_or_throw<Op>();
  int n_input = op->args_info.size();

  TensorType data_ty = input_ty[0];

  std::vector<int> axes_non_neg;
  if (!data_ty->IsUnknownNdim()) {
    axes_non_neg = NormalizeAxes(call, ctx, data_ty->ndim, axes);
  }
  int n_axis = axes.size();
  if (!data_ty->IsUnknownDtype()) {
    PrimType data_dtype = data_ty->dtype.value();
    // Norm ops only require a floating element kind; lane encoding is irrelevant to the check.
    if (!data_dtype.MatchesCode(kDLFloat, kDLBfloat)) {
      TVM_FFI_VISIT_THROW(TypeError, call)
          << op << " requires the input data to have float dtype. However, the given data dtype is "
          << data_ty->dtype;
    }
  }
  for (int i = 1; i < n_input; ++i) {
    if (input_ty[i]->dtype != data_ty->dtype) {
      TVM_FFI_VISIT_THROW(TypeError, call)
          << op << " requires all the input tensors to have the same dtype. However, the "
          << op->args_info[i]->name << " has dtype " << input_ty[i]->dtype
          << " which is other than the input data's dtype " << data_ty->dtype;
    } else if (input_ty[i]->ndim != n_axis) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << op << " requires the input " << op->args_info[i]->name
          << " to have as many dimensions as the length of input axes. However, the "
             "given one has ndim "
          << input_ty[i]->ndim << ", which is other than the length of axes " << n_axis;
    }
  }

  std::vector<ffi::Array<PrimExpr>> axis_lengths;
  axis_lengths.reserve(n_input);
  if (const auto* data_shape = data_ty->shape.as<ShapeExprObj>()) {
    std::vector<PrimExpr> lengths;
    lengths.reserve(n_axis);
    for (int d = 0; d < n_axis; ++d) {
      lengths.push_back(data_shape->values[axes_non_neg[d]]);
    }
    axis_lengths.push_back(lengths);
  }
  for (int i = 1; i < n_input; ++i) {
    if (const auto* shape = input_ty[i]->shape.as<ShapeExprObj>()) {
      axis_lengths.push_back(shape->values);
    }
  }

  sym::Analyzer analyzer = ctx->GetAnalyzer();
  for (int i = 1; i < static_cast<int>(axis_lengths.size()); ++i) {
    for (int d = 0; d < n_axis; ++d) {
      if (analyzer->CanProve(axis_lengths[0][d] != axis_lengths[i][d])) {
        TVM_FFI_VISIT_THROW(ValueError, call)
            << op
            << " requires the input gamma, beta, etc., to have size same as the "
               "lengths of the data on the given axes. However, there exists "
            << axis_lengths[0] << " and " << axis_lengths[i] << " that are unequal.";
      } else if (!analyzer->CanProveEqual(axis_lengths[0][d], axis_lengths[i][d])) {
        return true;
      }
    }
  }
  return false;
}

/* relax.nn.batch_norm */

Expr batch_norm(Expr data, Expr gamma, Expr beta, Expr moving_mean, Expr moving_var,  //
                int axis, double epsilon, bool center, bool scale, double momentum, bool training) {
  ffi::ObjectPtr<BatchNormAttrs> attrs = ffi::make_object<BatchNormAttrs>();
  attrs->axis = axis;
  attrs->epsilon = epsilon;
  attrs->center = center;
  attrs->scale = scale;
  attrs->momentum = momentum;
  attrs->training = training;

  static const Op op = Op::Get("relax.nn.batch_norm");
  return Call(Type::Missing(), op,
              {std::move(data), std::move(gamma), std::move(beta), std::move(moving_mean),
               std::move(moving_var)},
              Attrs{attrs}, {});
}
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.nn.batch_norm", batch_norm);
}

Type InferTypeBatchNorm(const Call& call, const BlockBuilder& ctx) {
  ffi::Array<TensorType> input_ty = GetInputTensorType(call, ctx);

  const auto* attrs = call->attrs.as<BatchNormAttrs>();
  bool unknown_shape = NormCheckDtypeAndShape(call, ctx, input_ty, {attrs->axis});

  ffi::Optional<PrimType> dtype = input_ty[0]->dtype;
  if (unknown_shape) {
    auto vdev = input_ty[0]->vdevice;
    return TupleType({TensorType(dtype, input_ty[0]->ndim, vdev),
                      TensorType(dtype, /*ndim=*/1, vdev), TensorType(dtype, /*ndim=*/1, vdev)});
  } else {
    return TupleType({input_ty[0], input_ty[3], input_ty[4]});
  }
}

InferLayoutOutput InferLayoutBatchNorm(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  TVM_FFI_ICHECK(NoDesiredLayout(call, desired_layouts));
  std::vector<NLayout> initial_layouts;
  for (size_t i = 0; i < 5; ++i) {
    const auto* tensor_ty = GetTypeAs<TensorTypeObj>(call->args[i]);
    TVM_FFI_ICHECK(tensor_ty != nullptr) << "Invalid Call";
    TVM_FFI_ICHECK(!tensor_ty->IsUnknownNdim()) << "Only support known ndim";
    initial_layouts.push_back(InitialLayoutDecision(tensor_ty->ndim));
  }
  const auto* attrs = call->attrs.as<BatchNormAttrs>();
  TVM_FFI_ICHECK(attrs) << "Invalid Call";
  LayoutDecision layout = GetLayoutDecision(var_layout_map, call->args[0]);

  // While dealing with sub layouts, its adviced to deal with batchnorm
  // on other ways like decomposing or fusion methods.
  // This handling is fail safe fallback.
  const auto* input_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
  int ndim = input_ty->ndim;
  if ((layout->layout.has_value() ? layout->layout.value().ndim() : 0) !=
      (layout->layout.has_value() ? layout->layout.value().ndim_primal() : 0)) {
    layout = LayoutDecision(InitialLayout(ndim));
  }

  ffi::ObjectPtr<BatchNormAttrs> new_attrs = ffi::make_object<BatchNormAttrs>(*attrs);
  new_attrs->axis = FindAxis(layout->layout.value(), (attrs->axis + ndim) % ndim);
  return InferLayoutOutput(
      {layout, initial_layouts[1], initial_layouts[2], initial_layouts[3], initial_layouts[4]},
      {{layout, initial_layouts[3], initial_layouts[4]}}, Attrs(new_attrs));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.nn.batch_norm")
      .signature(
          sig::arg("data", "Input to which batch_norm will be applied."),
          sig::arg("gamma", "The gamma scale factor."), sig::arg("beta", "The beta offset factor."),
          sig::arg("moving_mean", "Running mean of input."),
          sig::arg("moving_var", "Running variance of input."), sig::call_attrs<BatchNormAttrs>())
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeBatchNorm)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutBatchNorm)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.nn.layer_norm */

Expr layer_norm(Expr data, Expr gamma, Expr beta, ffi::Array<int64_t> axes, double epsilon,
                bool center, bool scale) {
  ffi::ObjectPtr<LayerNormAttrs> attrs = ffi::make_object<LayerNormAttrs>();
  attrs->axes = std::move(axes);
  attrs->epsilon = epsilon;
  attrs->center = center;
  attrs->scale = scale;

  static const Op op = Op::Get("relax.nn.layer_norm");
  return Call(Type::Missing(), op, {std::move(data), std::move(gamma), std::move(beta)},
              Attrs{attrs}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.nn.layer_norm", layer_norm);
}

Type InferTypeLayerNorm(const Call& call, const BlockBuilder& ctx) {
  ffi::Array<TensorType> input_ty = GetInputTensorType(call, ctx);

  const auto* attrs = call->attrs.as<LayerNormAttrs>();
  bool unknown_shape = NormCheckDtypeAndShape(call, ctx, input_ty, attrs->axes);

  return unknown_shape ? TensorType(input_ty[0]->dtype, input_ty[0]->ndim, input_ty[0]->vdevice)
                       : input_ty[0];
}

InferLayoutOutput InferLayoutLayerNorm(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  TVM_FFI_ICHECK(NoDesiredLayout(call, desired_layouts));
  std::vector<NLayout> initial_layouts;
  for (size_t i = 0; i < 3; ++i) {
    const auto* tensor_ty = GetTypeAs<TensorTypeObj>(call->args[i]);
    TVM_FFI_ICHECK(tensor_ty != nullptr) << "Invalid Call";
    TVM_FFI_ICHECK(!tensor_ty->IsUnknownNdim()) << "Only support known ndim";
    initial_layouts.push_back(InitialLayoutDecision(tensor_ty->ndim));
  }
  const auto* attrs = call->attrs.as<LayerNormAttrs>();
  TVM_FFI_ICHECK(attrs) << "Invalid Call";

  LayoutDecision layout = GetLayoutDecision(var_layout_map, call->args[0]);
  ffi::ObjectPtr<LayerNormAttrs> new_attrs = ffi::make_object<LayerNormAttrs>(*attrs);
  const auto* input_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
  int ndim = input_ty->ndim;
  std::vector<int64_t> new_axis;
  for (int64_t axis : attrs->axes) {
    new_axis.push_back(FindAxis(layout->layout.value(), (axis + ndim) % ndim));
  }
  new_attrs->axes = ffi::Array<int64_t>(new_axis.begin(), new_axis.end());
  return InferLayoutOutput({layout, initial_layouts[1], initial_layouts[2]}, {layout},
                           Attrs(new_attrs));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.nn.layer_norm")
      .signature(sig::arg("data", "Input to which layer_norm will be applied."),
                 sig::arg("gamma", "The gamma scale factor."),
                 sig::arg("beta", "The beta offset factor."), sig::call_attrs<LayerNormAttrs>())
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeLayerNorm)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutLayerNorm)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.nn.group_norm */

Expr group_norm(Expr data, Expr gamma, Expr beta, int num_groups, int channel_axis,
                ffi::Array<int64_t> axes, double epsilon, bool center, bool scale) {
  ffi::ObjectPtr<GroupNormAttrs> attrs = ffi::make_object<GroupNormAttrs>();
  attrs->num_groups = num_groups;
  attrs->channel_axis = channel_axis;
  attrs->axes = std::move(axes);
  attrs->epsilon = epsilon;
  attrs->center = center;
  attrs->scale = scale;

  static const Op op = Op::Get("relax.nn.group_norm");
  return Call(Type::Missing(), op, {std::move(data), std::move(gamma), std::move(beta)},
              Attrs{attrs}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.nn.group_norm", group_norm);
}

Type InferTypeGroupNorm(const Call& call, const BlockBuilder& ctx) {
  Op op = call->op.as_or_throw<Op>();
  ffi::Array<TensorType> input_ty = GetInputTensorType(call, ctx);
  const auto* attrs = call->attrs.as<GroupNormAttrs>();

  TensorType data_ty = input_ty[0];
  int channel_axis = -1;
  if (!data_ty->IsUnknownNdim()) {
    channel_axis = NormalizeAxis(call, ctx, data_ty->ndim, attrs->channel_axis);
    std::vector<int> axes = NormalizeAxes(call, ctx, data_ty->ndim, attrs->axes);
    // channel_axis must be in axes.
    if (std::find(axes.begin(), axes.end(), channel_axis) != axes.end()) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << op << " expects that channel_axis must not be in axes, but got channel_axis: "
          << channel_axis << ", axes: " << attrs->axes;
    }
  }
  // GroupNorm preserves the old float-kind check; vector lanes are irrelevant to this check.
  if (!data_ty->IsUnknownDtype() && !data_ty->dtype.value().MatchesCode(DLDataTypeCode::kDLFloat)) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << op << " expects that data must be float, but got " << data_ty->dtype;
  }
  sym::Analyzer analyzer = ctx->GetAnalyzer();
  const auto* data_shape = data_ty->shape.as<ShapeExprObj>();
  if (data_shape != nullptr && channel_axis != -1 &&
      analyzer->CanProve(floormod(data_shape->values[channel_axis], attrs->num_groups) != 0)) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << op << " expects that the size of channel_axis must be divisible by " << attrs->num_groups
        << ", but got " << data_shape->values[channel_axis];
  }
  for (int i = 1; i < static_cast<int>(op->args_info.size()); ++i) {
    if (input_ty[i]->dtype != data_ty->dtype) {
      TVM_FFI_VISIT_THROW(TypeError, call)
          << op << " expects that all inputs must have the same dtype, but got "
          << input_ty[i]->dtype << " and " << data_ty->dtype;
    } else if (input_ty[i]->ndim != 1) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << op << " expects that all inputs must have ndim=1, but got " << input_ty[i]->ndim;
    } else if (channel_axis != -1) {
      const auto* shape = input_ty[i]->shape.as<ShapeExprObj>();
      if (shape != nullptr && data_shape != nullptr) {
        PrimExpr channel_size = data_shape->values[channel_axis];
        PrimExpr input_size = shape->values[0];
        if (analyzer->CanProve(channel_size != input_size)) {
          TVM_FFI_VISIT_THROW(ValueError, call)
              << op << " expects that the size of input " << i
              << " must be equal to the size of channel_axis, but got " << input_size << " and "
              << channel_size;
        }
      }
    }
  }
  return data_ty;
}

InferLayoutOutput InferLayoutGroupNorm(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  TVM_FFI_ICHECK(NoDesiredLayout(call, desired_layouts));
  std::vector<NLayout> initial_layouts;
  for (size_t i = 0; i < 3; ++i) {
    const auto* tensor_ty = GetTypeAs<TensorTypeObj>(call->args[i]);
    TVM_FFI_ICHECK(tensor_ty != nullptr) << "Invalid Call";
    TVM_FFI_ICHECK(!tensor_ty->IsUnknownNdim()) << "Only support known ndim";
    initial_layouts.push_back(InitialLayoutDecision(tensor_ty->ndim));
  }
  const auto* attrs = call->attrs.as<GroupNormAttrs>();
  TVM_FFI_ICHECK(attrs) << "Invalid Call";

  LayoutDecision layout = GetLayoutDecision(var_layout_map, call->args[0]);
  ffi::ObjectPtr<GroupNormAttrs> new_attrs = ffi::make_object<GroupNormAttrs>(*attrs);
  std::vector<int64_t> new_axes;
  for (int64_t axis : attrs->axes) {
    new_axes.push_back(FindAxis(layout->layout.value(), axis));
  }
  new_attrs->axes = ffi::Array<int64_t>(new_axes.begin(), new_axes.end());
  new_attrs->channel_axis = FindAxis(layout->layout.value(), attrs->channel_axis);
  return InferLayoutOutput({layout, initial_layouts[1], initial_layouts[2]}, {layout},
                           Attrs(new_attrs));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.nn.group_norm")
      .signature(sig::arg("data", "Input to which group_norm will be applied."),
                 sig::arg("gamma", "The gamma scale factor."),
                 sig::arg("beta", "The beta offset factor."), sig::call_attrs<GroupNormAttrs>())
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeGroupNorm)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutGroupNorm)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.nn.instance_norm */

Expr instance_norm(Expr data, Expr gamma, Expr beta, int channel_axis, ffi::Array<int64_t> axes,
                   double epsilon, bool center, bool scale) {
  ffi::ObjectPtr<InstanceNormAttrs> attrs = ffi::make_object<InstanceNormAttrs>();
  attrs->channel_axis = std::move(channel_axis);
  attrs->axes = std::move(axes);
  attrs->epsilon = epsilon;
  attrs->center = center;
  attrs->scale = scale;

  static const Op op = Op::Get("relax.nn.instance_norm");
  return Call(Type::Missing(), op, {std::move(data), std::move(gamma), std::move(beta)},
              Attrs{attrs}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.nn.instance_norm", instance_norm);
}

Type InferTypeInstanceNorm(const Call& call, const BlockBuilder& ctx) {
  Op op = call->op.as_or_throw<Op>();
  ffi::Array<TensorType> input_ty = GetInputTensorType(call, ctx);
  const auto* attrs = call->attrs.as<InstanceNormAttrs>();
  TVM_FFI_ICHECK(attrs) << "Invalid Call";
  TensorType data_ty = input_ty[0];

  int channel_axis = -1;
  if (!data_ty->IsUnknownNdim()) {
    channel_axis = NormalizeAxis(call, ctx, data_ty->ndim, attrs->channel_axis);
    std::vector<int> axes = NormalizeAxes(call, ctx, data_ty->ndim, attrs->axes);
    // channel_axis must not be in axes.
    if (std::find(axes.begin(), axes.end(), channel_axis) != axes.end()) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << op << " expects that channel_axis must not be in axes, but got channel_axis: "
          << channel_axis << ", axes: " << attrs->axes;
    }
  }
  const auto* data_shape = data_ty->shape.as<ShapeExprObj>();
  sym::Analyzer analyzer = ctx->GetAnalyzer();
  for (int i = 1; i < static_cast<int>(op->args_info.size()); ++i) {
    if (input_ty[i]->dtype != data_ty->dtype) {
      TVM_FFI_VISIT_THROW(TypeError, call)
          << op << " expects that all inputs must have the same dtype, but got "
          << input_ty[i]->dtype << " and " << data_ty->dtype;
    } else if (input_ty[i]->ndim != 1) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << op << " expects that all inputs must have ndim=1, but got " << input_ty[i]->ndim;
    }
    const auto* shape = input_ty[i]->shape.as<ShapeExprObj>();
    if (shape != nullptr && data_shape != nullptr) {
      PrimExpr channel_size = data_shape->values[channel_axis];
      PrimExpr input_size = shape->values[0];
      if (analyzer->CanProve(channel_size != input_size)) {
        TVM_FFI_VISIT_THROW(ValueError, call)
            << op << " expects that the size of input " << i
            << " must be equal to the size of channel_axis, but got " << input_size << " and "
            << channel_size;
      }
    }
  }
  return data_ty;
}

InferLayoutOutput InferLayoutInstanceNorm(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  TVM_FFI_ICHECK(NoDesiredLayout(call, desired_layouts));
  std::vector<NLayout> initial_layouts;
  for (size_t i = 0; i < 3; ++i) {
    const auto* tensor_ty = GetTypeAs<TensorTypeObj>(call->args[i]);
    TVM_FFI_ICHECK(tensor_ty != nullptr) << "Invalid Call";
    TVM_FFI_ICHECK(!tensor_ty->IsUnknownNdim()) << "Only support known ndim";
    initial_layouts.push_back(InitialLayoutDecision(tensor_ty->ndim));
  }
  const auto* attrs = call->attrs.as<InstanceNormAttrs>();
  TVM_FFI_ICHECK(attrs) << "Invalid Call";

  LayoutDecision layout = GetLayoutDecision(var_layout_map, call->args[0]);
  ffi::ObjectPtr<InstanceNormAttrs> new_attrs = ffi::make_object<InstanceNormAttrs>(*attrs);
  std::vector<int64_t> new_axes;
  for (int64_t axis : attrs->axes) {
    new_axes.push_back(FindAxis(layout->layout.value(), axis));
  }
  new_attrs->axes = ffi::Array<int64_t>(new_axes.begin(), new_axes.end());
  new_attrs->channel_axis = FindAxis(layout->layout.value(), attrs->channel_axis);
  return InferLayoutOutput({layout, initial_layouts[1], initial_layouts[2]}, {layout},
                           Attrs(new_attrs));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.nn.instance_norm")
      .signature(sig::arg("data", "Input to which instance_norm will be applied."),
                 sig::arg("gamma", "The gamma scale factor."),
                 sig::arg("beta", "The beta offset factor."), sig::call_attrs<InstanceNormAttrs>())
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeInstanceNorm)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutInstanceNorm)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}
/* relax.nn.rms_norm */

Expr rms_norm(Expr data, Expr weight, ffi::Array<int64_t> axes, double epsilon) {
  ffi::ObjectPtr<RMSNormAttrs> attrs = ffi::make_object<RMSNormAttrs>();
  attrs->axes = std::move(axes);
  attrs->epsilon = epsilon;

  static const Op op = Op::Get("relax.nn.rms_norm");
  return Call(Type::Missing(), op, {std::move(data), std::move(weight)}, Attrs{attrs}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.nn.rms_norm", rms_norm);
}

Type InferTypeRMSNorm(const Call& call, const BlockBuilder& ctx) {
  ffi::Array<TensorType> input_ty = GetInputTensorType(call, ctx);

  const auto* attrs = call->attrs.as<RMSNormAttrs>();
  bool unknown_shape = NormCheckDtypeAndShape(call, ctx, input_ty, attrs->axes);

  return unknown_shape ? TensorType(input_ty[0]->dtype, input_ty[0]->ndim, input_ty[0]->vdevice)
                       : input_ty[0];
}

InferLayoutOutput InferLayoutRMSNorm(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  TVM_FFI_ICHECK(NoDesiredLayout(call, desired_layouts));
  std::vector<NLayout> initial_layouts;
  for (size_t i = 0; i < 2; ++i) {
    const auto* tensor_ty = GetTypeAs<TensorTypeObj>(call->args[i]);
    TVM_FFI_ICHECK(tensor_ty != nullptr) << "Invalid Call";
    TVM_FFI_ICHECK(!tensor_ty->IsUnknownNdim()) << "Only support known ndim";
    initial_layouts.push_back(InitialLayoutDecision(tensor_ty->ndim));
  }
  const auto* attrs = call->attrs.as<RMSNormAttrs>();
  TVM_FFI_ICHECK(attrs) << "Invalid Call";

  LayoutDecision layout = GetLayoutDecision(var_layout_map, call->args[0]);
  ffi::ObjectPtr<RMSNormAttrs> new_attrs = ffi::make_object<RMSNormAttrs>(*attrs);
  std::vector<int64_t> new_axes;
  for (int64_t axis : attrs->axes) {
    new_axes.push_back(FindAxis(layout->layout.value(), axis));
  }
  new_attrs->axes = ffi::Array<int64_t>(new_axes.begin(), new_axes.end());
  return InferLayoutOutput({layout, initial_layouts[1]}, {layout}, Attrs(new_attrs));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.nn.rms_norm")
      .signature(sig::arg("data", "Input to which rms_norm will be applied."),
                 sig::arg("weight", "The scale factor."), sig::call_attrs<RMSNormAttrs>())
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeRMSNorm)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutRMSNorm)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.nn.dropout */

Expr dropout(Expr data, double rate) {
  ffi::ObjectPtr<DropoutAttrs> attrs = ffi::make_object<DropoutAttrs>();
  attrs->rate = rate;

  static const Op op = Op::Get("relax.nn.dropout");
  return Call(Type::Missing(), op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.nn.dropout", dropout);
}

Type InferTypeDropout(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  TensorType data_ty = GetUnaryInputTensorType(call);
  return TupleType({data_ty, data_ty});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.nn.dropout")
      .signature(sig::arg("data", "Input to which dropout will be applied."),
                 sig::call_attrs<DropoutAttrs>())
      .set_native_fn_attr<&InferTypeDropout>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.nn.cross_entropy_with_logits */
Type InferTypeCrossEntropy(const Call& call, const BlockBuilder& ctx) {
  ffi::Array<TensorType> input_ty = GetInputTensorType(call, ctx);
  TensorType pred_ty = input_ty[0];
  TensorType label_ty = input_ty[1];

  // infer dtype
  ffi::Optional<PrimType> dtype = InferBinaryArithOpOutDtype(call, ctx, pred_ty, label_ty);

  // infer vdevice
  ffi::Optional<VDevice> vdevice = InferBinaryArithOpOutVDevice(call, ctx, pred_ty, label_ty);

  // infer ndim
  if (!pred_ty->IsUnknownNdim() && !label_ty->IsUnknownNdim() && pred_ty->ndim != label_ty->ndim) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "CrossEntropy requires predictions and labels to have the same ndim. "
           "However, the ndim of predictions is "
        << pred_ty->ndim << " while the ndim of labels is " << label_ty->ndim;
  }

  ffi::Optional<ffi::Array<PrimExpr>> pred_shape_value;
  if (pred_ty->shape.has_value()) {
    pred_shape_value = GetTypeAs<ShapeTypeObj>(pred_ty->shape.value())->values;
  }

  ffi::Optional<ffi::Array<PrimExpr>> label_shape_value;
  if (label_ty->shape.has_value()) {
    label_shape_value = GetTypeAs<ShapeTypeObj>(label_ty->shape.value())->values;
  }

  if (pred_shape_value.has_value() && label_shape_value.has_value()) {
    sym::Analyzer analyzer = ctx->GetAnalyzer();
    for (size_t i = 0; i < pred_shape_value.value().size(); ++i) {
      if (analyzer->CanProve(pred_shape_value.value()[i] != label_shape_value.value()[i])) {
        TVM_FFI_VISIT_THROW(ValueError, call)
            << "CrossEntropy requires the predictions and labels to have "
               "the same shape. However, the shape of predictions at dim "
            << i << " is" << pred_shape_value.value()[i]
            << " while the shape of labels at this dim is " << label_shape_value.value()[i];
      }
    }
  }
  return TensorType(ShapeExpr(ffi::Array<PrimExpr>()), dtype, vdevice);
}

Expr cross_entropy_with_logits(Expr predictions, Expr labels) {
  static const Op op = Op::Get("relax.nn.cross_entropy_with_logits");
  return Call(Type::Missing(), op, {std::move(predictions), std::move(labels)}, {}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.nn.cross_entropy_with_logits", cross_entropy_with_logits);

  OpDef("relax.nn.cross_entropy_with_logits")
      .signature(
          sig::arg("predictions", "The predictions."), sig::arg("labels", "The labels."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeCrossEntropy)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.nn.nll_loss */

Expr nll_loss(Expr predictions, Expr targets, ffi::Optional<Expr> weights, ffi::String reduction,
              int ignore_index) {
  ffi::ObjectPtr<NLLLossAttrs> attrs = ffi::make_object<NLLLossAttrs>();

  TVM_FFI_ICHECK(reduction == "none" || reduction == "sum" || reduction == "mean")
      << "The argument reduction of NLLLoss should be one of the following "
         "values: none, mean, sum. However, the given value is "
      << reduction;

  attrs->reduction = std::move(reduction);
  attrs->ignore_index = ignore_index;

  static const Op op = Op::Get("relax.nn.nll_loss");
  if (weights.has_value()) {
    return Call(Type::Missing(), op, {std::move(predictions), std::move(targets), weights.value()},
                Attrs{attrs}, {});
  } else {
    return Call(Type::Missing(), op, {std::move(predictions), std::move(targets)}, Attrs{attrs},
                {});
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.nn.nll_loss", nll_loss);
}

Type InferTypeNLLLoss(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() < 2 || call->args.size() > 3) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "NLLLoss op should take 2 or 3 arguments";
  }

  const auto* pred_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
  const auto* tgt_ty = GetTypeAs<TensorTypeObj>(call->args[1]);
  const TensorTypeObj* wgt_ty = nullptr;
  if (call->args.size() == 3) {
    wgt_ty = GetTypeAs<TensorTypeObj>(call->args[2]);
    if (wgt_ty == nullptr) {
      TVM_FFI_VISIT_THROW(TypeError, call)
          << "NLLLoss requires the argument weights to be Tensor. However, the given one is "
          << call->args[2]->ty->GetTypeKey();
    }
  }

  if (pred_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "NLLLoss requires the argument preditions to be Tensor. However, the given one is "
        << call->args[0]->ty->GetTypeKey();
  }
  if (tgt_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "NLLLoss requires the argument targets to be Tensor. However, the given one is "
        << call->args[1]->ty->GetTypeKey();
  }

  // infer dtype, vdevice
  ffi::Optional<PrimType> output_dtype =
      wgt_ty != nullptr ? InferBinaryArithOpOutDtype(call, ctx, ffi::GetRef<TensorType>(pred_ty),
                                                     ffi::GetRef<TensorType>(wgt_ty))
                        : pred_ty->dtype;
  ffi::Optional<VDevice> vdevice =
      wgt_ty != nullptr ? InferBinaryArithOpOutVDevice(call, ctx, ffi::GetRef<TensorType>(pred_ty),
                                                       ffi::GetRef<TensorType>(wgt_ty))
                        : pred_ty->vdevice;

  // the type of targets must be int/uint.
  if (!tgt_ty->IsUnknownDtype()) {
    PrimType target_dtype = tgt_ty->dtype.value();
    // NLLLoss only needs the target element kind; vector lanes do not affect target indexing.
    if (!target_dtype.MatchesCode(DLDataTypeCode::kDLInt) &&
        !target_dtype.MatchesCode(DLDataTypeCode::kDLUInt)) {
      TVM_FFI_VISIT_THROW(TypeError, call) << "NLLLoss expects the dtype of targets to be "
                                              "int/uint. However, the dtype of targets is "
                                           << tgt_ty->dtype;
    }
  }

  // infer ndim
  int K = kUnknownNDim;  // k dim
  if (!pred_ty->IsUnknownNdim()) {
    if (pred_ty->ndim < 1) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "NLLLoss expects the ndim of predictions >= 1. However, the ndim of predictions is "
          << pred_ty->ndim;
    }
    K = pred_ty->ndim <= 2 ? 0 : pred_ty->ndim - 2;
  }
  if (!tgt_ty->IsUnknownNdim()) {
    int K_tgt = tgt_ty->ndim <= 1 ? 0 : tgt_ty->ndim - 1;
    if (K != kUnknownNDim && K != K_tgt) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "NLLLoss expects number of dimensions K inferred from different "
             "arguments to be equal. However, K from predictions is "
          << K << " while K from targets is " << K_tgt;
    }
  }
  if (wgt_ty != nullptr && !wgt_ty->IsUnknownNdim() && wgt_ty->ndim != 1) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "NLLLoss expects the ndim of weights == 1. However, the ndim of weights is "
        << wgt_ty->ndim;
  }

  sym::Analyzer analyzer = ctx->GetAnalyzer();
  ffi::Optional<PrimExpr> N;
  ffi::Optional<PrimExpr> C;
  ffi::Array<PrimExpr> output_shape;  // N, d1, d2, ..., dk

  ffi::Optional<ffi::Array<PrimExpr>> pred_shape_value;
  if (pred_ty->shape.has_value()) {
    pred_shape_value = GetTypeAs<ShapeTypeObj>(pred_ty->shape.value())->values;
  }
  if (pred_shape_value.has_value()) {
    if (pred_shape_value.value().size() == 1) {
      // (C,)
      TVM_FFI_ICHECK(pred_ty->ndim == 1);
      C = pred_shape_value.value()[0];
    } else {
      // (N, C, d1, d2, ..., dk)
      TVM_FFI_ICHECK(pred_shape_value.value().size() >= 2);
      TVM_FFI_ICHECK(pred_ty->ndim == static_cast<int>(pred_shape_value.value().size()));
      N = pred_shape_value.value()[0];
      C = pred_shape_value.value()[1];
      output_shape = ffi::Array<PrimExpr>();
      output_shape.push_back(N.value());
      for (size_t i = 2; i < pred_shape_value.value().size(); ++i) {
        output_shape.push_back(pred_shape_value.value()[i]);
      }
    }
  }

  ffi::Optional<ffi::Array<PrimExpr>> tgt_shape_value;
  if (tgt_ty->shape.has_value()) {
    tgt_shape_value = GetTypeAs<ShapeTypeObj>(tgt_ty->shape.value())->values;
  }
  if (tgt_shape_value.has_value()) {
    if (tgt_shape_value.value().empty()) {
      // ()
      TVM_FFI_ICHECK(tgt_ty->ndim == 0);
      if (N.has_value()) {
        TVM_FFI_VISIT_THROW(ValueError, call) << "Shape mismatch for NLLLoss. Predictions shape is "
                                                 "(N, C, ...) while targets is a scalar";
      }
    } else {
      // (N,) or (N, d1, d2, ..., dk)
      // check N
      const PrimExpr& N_tgt = tgt_shape_value.value()[0];
      if (N.has_value() && analyzer->CanProve(N.value() != N_tgt)) {
        TVM_FFI_VISIT_THROW(ValueError, call)
            << "NLLLoss expects minibatch size N inferred from different "
               "arguments to be equal. However, N from predictions is "
            << N << " while N from targets is " << N_tgt;
      }
      // only C case
      if (!N.has_value() && C.has_value()) {
        TVM_FFI_VISIT_THROW(ValueError, call) << "Shape mismatch for NLLLoss. Predictions shape is "
                                                 "(C,) while targets is not a scalar";
      }

      if (tgt_shape_value.value().size() == 1) {
        // (N,)
        TVM_FFI_ICHECK(tgt_ty->IsUnknownNdim() || tgt_ty->ndim == 1);
      } else {
        // (N, d1, d2, ..., dk)
        TVM_FFI_ICHECK(tgt_shape_value.value().size() >= 2);
        TVM_FFI_ICHECK(tgt_ty->IsUnknownNdim() ||
                       tgt_ty->ndim == static_cast<int>(tgt_shape_value.value().size()));

        if (pred_shape_value.has_value()) {
          // check (d1, d2, ..., dk)
          for (size_t i = 1; i < tgt_shape_value.value().size(); ++i) {
            if (analyzer->CanProve(output_shape[i] != tgt_shape_value.value()[i])) {
              TVM_FFI_VISIT_THROW(ValueError, call)
                  << "Shape mismatch for NLLLoss. The prediction shape at this dim is "
                  << output_shape[i] << " while the target shape at this dim is "
                  << tgt_shape_value.value()[i];
            }
          }
        }
      }
    }
  }

  if (wgt_ty != nullptr) {
    ffi::Optional<ffi::Array<PrimExpr>> wgt_shape_value;
    if (wgt_ty->shape.has_value()) {
      wgt_shape_value = GetTypeAs<ShapeTypeObj>(wgt_ty->shape.value())->values;
    }
    if (wgt_shape_value.has_value()) {
      TVM_FFI_ICHECK(wgt_shape_value.value().size() == 1);
      TVM_FFI_ICHECK(wgt_ty->IsUnknownNdim() || wgt_ty->ndim == 1);
      const PrimExpr& C_wgt = wgt_shape_value.value()[0];
      if (C.has_value() && analyzer->CanProve(C.value() != C_wgt)) {
        TVM_FFI_VISIT_THROW(ValueError, call)
            << "NLLLoss expects number of classes C inferred from different "
               "arguments to be equal. However, C from predictions is "
            << C << " while C from weights is " << C_wgt;
      }
    }
  }

  const auto* attrs = call->attrs.as<NLLLossAttrs>();
  ffi::String reduction = attrs->reduction;

  if (reduction == "none") {
    // () or (N,) or (N, d1, d2, ..., dk)
    if (pred_ty->shape.as<ShapeExprObj>()) {
      return TensorType(ShapeExpr(output_shape), output_dtype, vdevice);
    } else {
      int output_ndim = pred_ty->ndim == kUnknownNDim ? kUnknownNDim : pred_ty->ndim - 1;
      return TensorType(output_dtype, /*ndim=*/output_ndim, vdevice);
    }
  } else {
    // sum or mean. output is scalar
    return TensorType(/*shape=*/ShapeExpr(ffi::Array<PrimExpr>()), output_dtype, vdevice);
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.nn.nll_loss", "Optional weights: The weight of each target values.")
      .signature(
          sig::arg("predictions", "The prediction tensor."),
          sig::arg("targets", "The target tensor."), sig::var_args("args"),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."),
          sig::call_attrs<NLLLossAttrs>())
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeNLLLoss)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.nn.batch_flatten */

Expr batch_flatten(Expr data) {
  static const Op op = Op::Get("relax.nn.batch_flatten");
  return Call(Type::Missing(), op, {std::move(data)}, {}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.nn.batch_flatten", batch_flatten);
}

Type InferTypeBatchFlatten(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  TensorType data_ty = GetUnaryInputTensorType(call);

  if (data_ty->IsUnknownNdim()) {
    return TensorType(data_ty->dtype, /*ndim=*/2, data_ty->vdevice);
  }

  if (data_ty->ndim < 2) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "batch_flatten expects input tensor to have at least 2 dimensions, "
        << "but got " << data_ty->ndim;
  }

  if (data_ty->ndim == 2) {
    return data_ty;
  }

  const auto* data_shape = data_ty->shape.as<ShapeExprObj>();
  if (data_shape == nullptr) {
    return TensorType(data_ty->dtype, /*ndim=*/2, data_ty->vdevice);
  }

  PrimExpr batch_dim = data_shape->values[0];
  PrimExpr flat_dim = IntImm::Int64(1);
  for (size_t i = 1; i < data_shape->values.size(); ++i) {
    flat_dim = flat_dim * data_shape->values[i];
  }

  return TensorType(ShapeExpr({batch_dim, flat_dim}), data_ty->dtype, data_ty->vdevice);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.nn.batch_flatten")
      .signature(sig::arg("data", "The input tensor."))
      .set_native_fn_attr<&InferTypeBatchFlatten>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

}  // namespace relax
}  // namespace tvm
