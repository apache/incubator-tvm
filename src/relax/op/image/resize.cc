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
 * \file resize.cc
 * \brief Image resize operators.
 */

#include "resize.h"

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/relax/op_attr.h>

#include <utility>

namespace tvm {
namespace relax {

void Resize2DAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<Resize2DAttrs>()
      .def_ro("roi", &Resize2DAttrs::roi,
              "Region of Interest for coordinate transformation mode 'tf_crop_and_resize'")
      .def_ro("layout", &Resize2DAttrs::layout,
              "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
              "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
              "dimensions respectively. Resize is applied on the 'H' and"
              "'W' dimensions.",
              refl::DefaultValue(ffi::String("NCHW")))
      .def_ro("method", &Resize2DAttrs::method,
              "Specify the mode to use for scaling."
              "nearest_neighbor -  Nearest Neighbor"
              "linear - Bilinear Interpolation"
              "cubic - Bicubic Interpolation",
              refl::DefaultValue(ffi::String("linear")))
      .def_ro("coordinate_transformation_mode", &Resize2DAttrs::coordinate_transformation_mode,
              "Describes how to transform the coordinate in the resized tensor"
              "to the coordinate in the original tensor."
              "Refer to the ONNX Resize operator specification for details"
              "Available options are half_pixel, align_corners and asymmetric",
              refl::DefaultValue(ffi::String("half_pixel")))
      .def_ro("rounding_method", &Resize2DAttrs::rounding_method,
              "indicates how to find the \"nearest\" pixel in nearest_neighbor method"
              "Available options are round, floor, and ceil.",
              refl::DefaultValue(ffi::String("round")))
      .def_ro("cubic_alpha", &Resize2DAttrs::cubic_alpha,
              "Spline Coefficient for Bicubic Interpolation", refl::DefaultValue(-0.75))
      .def_ro("cubic_exclude", &Resize2DAttrs::cubic_exclude,
              "Flag to exclude exterior of the image during bicubic interpolation",
              refl::DefaultValue(0))
      .def_ro("extrapolation_value", &Resize2DAttrs::extrapolation_value,
              "Value to return when roi is outside of the image", refl::DefaultValue(0.0))
      .def_ro(
          "out_dtype", &Resize2DAttrs::out_dtype,
          "The dtype of the output tensor. It it is not specified, the output will have the same "
          "dtype as input if not specified.",
          refl::DefaultValue(ffi::Optional<DLDataType>{}));
}

void Resize3DAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<Resize3DAttrs>()
      .def_ro("roi", &Resize3DAttrs::roi,
              "Region of Interest for coordinate transformation mode 'tf_crop_and_resize'")
      .def_ro("layout", &Resize3DAttrs::layout,
              "Dimension ordering of input data. Can be 'NCDHW', 'NDHWC', etc."
              "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
              "dimensions respectively. Resize is applied on the 'D', 'H' and"
              "'W' dimensions.",
              refl::DefaultValue(ffi::String("NCDHW")))
      .def_ro("method", &Resize3DAttrs::method,
              "Specify the mode to use for scaling."
              "nearest_neighbor -  Nearest Neighbor"
              "linear - Trilinear Interpolation"
              "cubic - Tricubic Interpolation",
              refl::DefaultValue(ffi::String("linear")))
      .def_ro("coordinate_transformation_mode", &Resize3DAttrs::coordinate_transformation_mode,
              "Describes how to transform the coordinate in the resized tensor"
              "to the coordinate in the original tensor."
              "Refer to the ONNX Resize operator specification for details"
              "Available options are half_pixel, align_corners and asymmetric",
              refl::DefaultValue(ffi::String("half_pixel")))
      .def_ro("rounding_method", &Resize3DAttrs::rounding_method,
              "indicates how to find the \"nearest\" pixel in nearest_neighbor method"
              "Available options are round, floor, and ceil.",
              refl::DefaultValue(ffi::String("")))
      .def_ro("cubic_alpha", &Resize3DAttrs::cubic_alpha,
              "Spline Coefficient for Tricubic Interpolation", refl::DefaultValue(-0.75))
      .def_ro("cubic_exclude", &Resize3DAttrs::cubic_exclude,
              "Flag to exclude exterior of the image during tricubic interpolation",
              refl::DefaultValue(0))
      .def_ro("extrapolation_value", &Resize3DAttrs::extrapolation_value,
              "Value to return when roi is outside of the image", refl::DefaultValue(0.0))
      .def_ro(
          "out_dtype", &Resize3DAttrs::out_dtype,
          "The dtype of the output tensor. It it is not specified, the output will have the same "
          "dtype as input if not specified.",
          refl::DefaultValue(ffi::Optional<DLDataType>{}));
}

void GridSampleAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<GridSampleAttrs>()
      .def_ro("method", &GridSampleAttrs::method,
              "Interpolation method. Can be 'nearest', 'bilinear', or 'bicubic'.",
              refl::DefaultValue(ffi::String("bilinear")))
      .def_ro("layout", &GridSampleAttrs::layout,
              "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc.",
              refl::DefaultValue(ffi::String("NCHW")))
      .def_ro("padding_mode", &GridSampleAttrs::padding_mode,
              "Padding mode for outside grid values. Can be 'zeros', 'border', or 'reflection'.",
              refl::DefaultValue(ffi::String("zeros")))
      .def_ro("align_corners", &GridSampleAttrs::align_corners,
              "If True, the corner pixels of the input and output tensors are aligned.",
              refl::DefaultValue(false));
}

void AffineGridAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<AffineGridAttrs>().def_ro(
      "align_corners", &AffineGridAttrs::align_corners,
      "If True, normalized grid coordinates map to corner pixels; otherwise to pixel centers.",
      refl::DefaultValue(true));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  Resize2DAttrs::RegisterReflection();
  Resize3DAttrs::RegisterReflection();
}

/* relax.resize2d */

Expr resize2d(Expr data, Expr size, ffi::Array<FloatImm> roi, ffi::String layout,
              ffi::String method, ffi::String coordinate_transformation_mode,
              ffi::String rounding_method, double cubic_alpha, int cubic_exclude,
              double extrapolation_value, ffi::Optional<DLDataType> out_dtype) {
  ffi::ObjectPtr<Resize2DAttrs> attrs = ffi::make_object<Resize2DAttrs>();
  attrs->roi = std::move(roi);
  attrs->layout = std::move(layout);
  attrs->method = std::move(method);
  attrs->coordinate_transformation_mode = std::move(coordinate_transformation_mode);
  attrs->rounding_method = std::move(rounding_method);
  attrs->cubic_alpha = cubic_alpha;
  attrs->cubic_exclude = cubic_exclude;
  attrs->extrapolation_value = extrapolation_value;
  attrs->out_dtype = out_dtype;

  static const Op op = Op::Get("relax.image.resize2d");
  return Call(Type::Missing(), op, {std::move(data), std::move(size)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.image.resize2d", resize2d);
}

Type InferTypeResize2D(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  if (call->args.size() != 2) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Resize2D expects 2 arguments, while the given number of arguments is "
        << call->args.size();
  }

  const auto* data_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
  const auto* size_ty = GetTypeAs<ShapeTypeObj>(call->args[1]);
  const auto* size_value = call->args[1].as<ShapeExprObj>();
  if (data_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "Resize2D expects the input data to be a Tensor, while the given data is "
        << call->args[0]->GetTypeKey();
  }
  if (size_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "Resize2D expects the given output image size to be a Shape, while the given one is "
        << call->args[1]->GetTypeKey();
  }
  if (size_ty->ndim != 2) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "Resize2D expects the given output image size to "
                                             "be a 2-dim shape, while the given one has ndim "
                                          << size_ty->ndim;
  }

  const auto* attrs = call->attrs.as<Resize2DAttrs>();
  auto [data_layout, data2NCHW] = CheckTensorLayout(call, attrs->layout,    //
                                                    /*tgt_layout=*/"NCHW",  //
                                                    /*tensor_name=*/"data");

  ffi::Optional<PrimType> out_dtype =
      attrs->out_dtype.has_value() ? ffi::Optional<PrimType>(PrimType(attrs->out_dtype.value()))
                                   : data_ty->dtype;

  ffi::Optional<ShapeExpr> data_shape =
      CheckNdimPerLayoutAndGetShape(call, ffi::GetRef<TensorType>(data_ty), data_layout);
  if (!data_shape.has_value() || size_value == nullptr) {
    return TensorType(out_dtype, data_layout.ndim(), data_ty->vdevice);
  }

  ffi::Array<PrimExpr> data_NCHW_shape = data2NCHW.ForwardShape(data_shape.value()->values);
  ffi::Array<PrimExpr> out_NCHW_shape(data_NCHW_shape);
  out_NCHW_shape.Set(2, size_value->values[0]);
  out_NCHW_shape.Set(3, size_value->values[1]);

  ffi::Array<PrimExpr> out_shape = data2NCHW.BackwardShape(out_NCHW_shape);
  return TensorType(ShapeExpr(out_shape), out_dtype, data_ty->vdevice);
}

InferLayoutOutput InferLayoutResize2d(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  const auto& it = desired_layouts.find("relax.image.resize2d");
  const auto* attrs = call->attrs.as<Resize2DAttrs>();
  TVM_FFI_ICHECK(attrs) << "Invalid Call";

  LayoutDecision data_layout(ffi::UnsafeInit{});
  ffi::ObjectPtr<Resize2DAttrs> new_attrs = ffi::make_object<Resize2DAttrs>(*attrs);

  if (it != desired_layouts.end()) {
    // We have a desired layout for resize2d.
    SLayout desired_data_layout = (*it).second[0];
    TVM_FFI_ICHECK_EQ(desired_data_layout.ndim(), desired_data_layout.ndim_primal())
        << "Axis swap only";
    data_layout =
        TransposeLike(InitialLayout(4), SLayout::Create(attrs->layout), desired_data_layout);
    new_attrs->layout = (*it).second[0];
  } else {
    // We dont have a desired layout for resize2d, propagate from the input instead.
    data_layout = GetLayoutDecision(var_layout_map, call->args[0]);
    // Not handling sub indexing now.
    if ((data_layout->layout.has_value() ? data_layout->layout.value().ndim() : 0) !=
        (data_layout->layout.has_value() ? data_layout->layout.value().ndim_primal() : 0)) {
      data_layout = LayoutDecision(InitialLayout(4));
    }
    new_attrs->layout =
        TransposeLike(SLayout::Create(attrs->layout), InitialLayout(4), data_layout->layout).name();
  }
  return InferLayoutOutput({data_layout, InitialNLayout(call->args[1])}, {data_layout},
                           Attrs(new_attrs));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.image.resize2d")
      .signature(sig::arg("data", "The input tensor."), sig::arg("size", "The output image shape."),
                 sig::call_attrs<Resize2DAttrs>())
      .set_native_fn_attr<&InferTypeResize2D>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutResize2d)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.resize3d */

Expr resize3d(Expr data, Expr size, ffi::Array<FloatImm> roi, ffi::String layout,
              ffi::String method, ffi::String coordinate_transformation_mode,
              ffi::String rounding_method, double cubic_alpha, int cubic_exclude,
              double extrapolation_value, ffi::Optional<DLDataType> out_dtype) {
  ffi::ObjectPtr<Resize3DAttrs> attrs = ffi::make_object<Resize3DAttrs>();
  attrs->roi = std::move(roi);
  attrs->layout = std::move(layout);
  attrs->method = std::move(method);
  attrs->coordinate_transformation_mode = std::move(coordinate_transformation_mode);
  attrs->rounding_method = std::move(rounding_method);
  attrs->cubic_alpha = cubic_alpha;
  attrs->cubic_exclude = cubic_exclude;
  attrs->extrapolation_value = extrapolation_value;
  attrs->out_dtype = out_dtype;

  static const Op op = Op::Get("relax.image.resize3d");
  return Call(Type::Missing(), op, {std::move(data), std::move(size)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.image.resize3d", resize3d);
}

Type InferTypeResize3D(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  if (call->args.size() != 2) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Resize3D expects 2 arguments, while the given number of arguments is "
        << call->args.size();
  }

  const auto* data_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
  const auto* size_ty = GetTypeAs<ShapeTypeObj>(call->args[1]);
  const auto* size_value = call->args[1].as<ShapeExprObj>();
  if (data_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "Resize3D expects the input data to be a Tensor, while the given data is "
        << call->args[0]->GetTypeKey();
  }
  if (size_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "Resize3D expects the given output image size to be a Shape, while the given one is "
        << call->args[1]->GetTypeKey();
  }
  if (size_ty->ndim != 3) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "Resize3D expects the given output image size to "
                                             "be a 3-dim shape, while the given one has ndim "
                                          << size_ty->ndim;
  }

  const auto* attrs = call->attrs.as<Resize3DAttrs>();
  auto [data_layout, data2NCDHW] = CheckTensorLayout(call, attrs->layout,     //
                                                     /*tgt_layout=*/"NCDHW",  //
                                                     /*tensor_name=*/"data");

  ffi::Optional<PrimType> out_dtype =
      attrs->out_dtype.has_value() ? ffi::Optional<PrimType>(PrimType(attrs->out_dtype.value()))
                                   : data_ty->dtype;

  ffi::Optional<ShapeExpr> data_shape =
      CheckNdimPerLayoutAndGetShape(call, ffi::GetRef<TensorType>(data_ty), data_layout);
  if (!data_shape.has_value() || size_value == nullptr) {
    return TensorType(out_dtype, data_layout.ndim(), data_ty->vdevice);
  }

  ffi::Array<PrimExpr> data_NCDHW_shape = data2NCDHW.ForwardShape(data_shape.value()->values);
  ffi::Array<PrimExpr> out_NCDHW_shape(data_NCDHW_shape);
  out_NCDHW_shape.Set(2, size_value->values[0]);
  out_NCDHW_shape.Set(3, size_value->values[1]);
  out_NCDHW_shape.Set(4, size_value->values[2]);

  ffi::Array<PrimExpr> out_shape = data2NCDHW.BackwardShape(out_NCDHW_shape);
  return TensorType(ShapeExpr(out_shape), out_dtype, data_ty->vdevice);
}

InferLayoutOutput InferLayoutResize3d(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  const auto& it = desired_layouts.find("relax.image.resize3d");
  const auto* attrs = call->attrs.as<Resize3DAttrs>();
  TVM_FFI_ICHECK(attrs) << "Invalid Call";

  LayoutDecision data_layout(ffi::UnsafeInit{});
  ffi::ObjectPtr<Resize3DAttrs> new_attrs = ffi::make_object<Resize3DAttrs>(*attrs);

  if (it != desired_layouts.end()) {
    SLayout desired_data_layout = (*it).second[0];
    TVM_FFI_ICHECK_EQ(desired_data_layout.ndim(), desired_data_layout.ndim_primal())
        << "Axis swap only";
    data_layout =
        TransposeLike(InitialLayout(5), SLayout::Create(attrs->layout), desired_data_layout);
    new_attrs->layout = (*it).second[0];
  } else {
    data_layout = GetLayoutDecision(var_layout_map, call->args[0]);
    if ((data_layout->layout.has_value() ? data_layout->layout.value().ndim() : 0) !=
        (data_layout->layout.has_value() ? data_layout->layout.value().ndim_primal() : 0)) {
      data_layout = LayoutDecision(InitialLayout(5));
    }
    new_attrs->layout =
        TransposeLike(SLayout::Create(attrs->layout), InitialLayout(5), data_layout->layout).name();
  }
  return InferLayoutOutput({data_layout, InitialNLayout(call->args[1])}, {data_layout},
                           Attrs(new_attrs));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.image.resize3d")
      .signature(sig::arg("data", "The input tensor."), sig::arg("size", "The output image shape."),
                 sig::call_attrs<Resize3DAttrs>())
      .set_native_fn_attr<&InferTypeResize3D>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutResize3d)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);

  /* relax.grid_sample */

  GridSampleAttrs::RegisterReflection();
}

Expr grid_sample(Expr data, Expr grid, ffi::String method, ffi::String layout,
                 ffi::String padding_mode, bool align_corners) {
  ffi::ObjectPtr<GridSampleAttrs> attrs = ffi::make_object<GridSampleAttrs>();
  attrs->method = std::move(method);
  attrs->layout = std::move(layout);
  attrs->padding_mode = std::move(padding_mode);
  attrs->align_corners = align_corners;

  static const Op op = Op::Get("relax.image.grid_sample");
  return Call(Type::Missing(), op, {std::move(data), std::move(grid)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.image.grid_sample", grid_sample);
}

Type InferTypeGridSample(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  if (call->args.size() != 2) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "GridSample expects two arguments, while the given number of arguments is "
        << call->args.size();
  }

  const auto* data_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
  const auto* grid_ty = GetTypeAs<TensorTypeObj>(call->args[1]);

  if (data_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "GridSample expects the input data to be a Tensor, while the given data is "
        << call->args[0]->GetTypeKey();
  }
  if (grid_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "GridSample expects the grid to be a Tensor, while the given grid is "
        << call->args[1]->GetTypeKey();
  }

  const auto* attrs = call->attrs.as<GridSampleAttrs>();

  // grid_sample supports both 2D (NCHW) and 3D (NCDHW) sampling. The frontend
  // sets attrs->layout to "NCDHW" for the volumetric case; everything else is
  // treated as the 2D NCHW path so existing behavior is preserved.
  const bool is_ncdhw = (attrs->layout == "NCDHW");

  auto [data_layout, data2tgt] = CheckTensorLayout(call, attrs->layout,
                                                   /*tgt_layout=*/is_ncdhw ? "NCDHW" : "NCHW",
                                                   /*tensor_name=*/"data");

  ffi::Optional<PrimType> out_dtype = data_ty->dtype;

  ffi::Optional<ShapeExpr> data_shape =
      CheckNdimPerLayoutAndGetShape(call, ffi::GetRef<TensorType>(data_ty), data_layout);
  const auto* grid_shape = grid_ty->shape.as<ShapeExprObj>();

  if (!data_shape.has_value() || grid_shape == nullptr) {
    return TensorType(out_dtype, data_layout.ndim(), data_ty->vdevice);
  }

  ffi::Array<PrimExpr> data_tgt_shape = data2tgt.ForwardShape(data_shape.value()->values);
  ffi::Array<PrimExpr> out_tgt_shape(data_tgt_shape);
  if (is_ncdhw) {
    // grid (TVM layout) is [N, 3, D_out, H_out, W_out], output is
    // [N, C, D_out, H_out, W_out]; the spatial extents are grid->values[2:].
    out_tgt_shape.Set(2, grid_shape->values[2]);  // D_out
    out_tgt_shape.Set(3, grid_shape->values[3]);  // H_out
    out_tgt_shape.Set(4, grid_shape->values[4]);  // W_out
  } else {
    // grid (TVM layout) is [N, 2, H_out, W_out], output is [N, C, H_out, W_out]
    out_tgt_shape.Set(2, grid_shape->values[2]);  // H_out
    out_tgt_shape.Set(3, grid_shape->values[3]);  // W_out
  }

  ffi::Array<PrimExpr> out_shape = data2tgt.BackwardShape(out_tgt_shape);
  return TensorType(ShapeExpr(out_shape), out_dtype, data_ty->vdevice);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.image.grid_sample")
      .signature(sig::arg("data", "The input tensor."),
                 sig::arg("grid", "The grid tensor for sampling."),
                 sig::call_attrs<GridSampleAttrs>())
      .set_native_fn_attr<&InferTypeGridSample>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.image.affine_grid */

Expr affine_grid(Expr data, Expr size, bool align_corners) {
  ffi::ObjectPtr<AffineGridAttrs> attrs = ffi::make_object<AffineGridAttrs>();
  attrs->align_corners = align_corners;
  static const Op op = Op::Get("relax.image.affine_grid");
  return Call(Type::Missing(), op, {std::move(data), std::move(size)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  AffineGridAttrs::RegisterReflection();

  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.image.affine_grid", affine_grid);
}

Type InferTypeAffineGrid(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  if (call->args.size() != 2) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "AffineGrid expects two arguments, while the given number of arguments is "
        << call->args.size();
  }

  const auto* data_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
  const auto* size_ty = GetTypeAs<ShapeTypeObj>(call->args[1]);
  const auto* size_value = call->args[1].as<ShapeExprObj>();

  if (data_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "AffineGrid expects the input data to be a Tensor, while the given data is "
        << call->args[0]->GetTypeKey();
  }
  if (size_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "AffineGrid expects the target size to be a Shape, while the given one is "
        << call->args[1]->GetTypeKey();
  }
  // 2D output uses a 2-dim spatial size ([H, W]); 3D uses a 3-dim size ([D, H, W]).
  if (size_ty->ndim != 2 && size_ty->ndim != 3) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "AffineGrid expects the target size to be a 2-dim or 3-dim shape, while the given "
           "one has ndim "
        << size_ty->ndim;
  }
  const int spatial = size_ty->ndim;

  // data should be 3-D: [batch, spatial, spatial + 1] (i.e. [N, 2, 3] or [N, 3, 4]).
  if (data_ty->ndim != -1 && data_ty->ndim != 3) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "AffineGrid expects the input data to be 3-D (batch, "
                                             "spatial, spatial + 1), but got ndim "
                                          << data_ty->ndim;
  }

  const auto* data_shape = data_ty->shape.as<ShapeExprObj>();
  if (data_shape != nullptr) {
    if (data_shape->values.size() >= 2) {
      auto* dim1 = data_shape->values[1].as<IntImmObj>();
      if (dim1 != nullptr && dim1->value != spatial) {
        TVM_FFI_VISIT_THROW(ValueError, call)
            << "AffineGrid expects the second dimension of input to be " << spatial << ", but got "
            << dim1->value;
      }
    }
    if (data_shape->values.size() >= 3) {
      auto* dim2 = data_shape->values[2].as<IntImmObj>();
      if (dim2 != nullptr && dim2->value != spatial + 1) {
        TVM_FFI_VISIT_THROW(ValueError, call)
            << "AffineGrid expects the third dimension of input to be " << spatial + 1
            << ", but got " << dim2->value;
      }
    }
  }

  ffi::Optional<PrimType> out_dtype = data_ty->dtype;

  if (data_shape == nullptr || size_value == nullptr) {
    return TensorType(out_dtype, /*ndim=*/spatial + 2, data_ty->vdevice);
  }

  // Output shape: [batch, spatial, *target_spatial_dims].
  ffi::Array<PrimExpr> out_shape;
  out_shape.push_back(data_shape->values[0]);   // batch
  out_shape.push_back(IntImm::Int64(spatial));  // number of spatial coordinates
  for (int i = 0; i < spatial; ++i) {
    out_shape.push_back(size_value->values[i]);  // target spatial dim
  }

  return TensorType(ShapeExpr(out_shape), out_dtype, data_ty->vdevice);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.image.affine_grid")
      .signature(sig::arg("data", "The input affine matrix tensor."),
                 sig::arg("size", "The target output shape (H, W)."),
                 sig::call_attrs<AffineGridAttrs>())
      .set_native_fn_attr<&InferTypeAffineGrid>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

}  // namespace relax
}  // namespace tvm
