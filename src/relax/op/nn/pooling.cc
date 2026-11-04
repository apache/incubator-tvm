/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "pooling.h"

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/relax/op_attr.h>

#include <utility>
#include <vector>

namespace tvm {
namespace relax {
using namespace tvm::prim;

void Pool1DAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<Pool1DAttrs>()
      .def_ro("pool_size", &Pool1DAttrs::pool_size, "Size of the pooling windows.",
              refl::DefaultValue(ffi::Array<int64_t>{1}))
      .def_ro("strides", &Pool1DAttrs::strides, "Specifies the strides of the convolution.",
              refl::DefaultValue(ffi::Array<int64_t>{1}))
      .def_ro("dilation", &Pool1DAttrs::dilation, "Specifies the dilation of the convolution.",
              refl::DefaultValue(ffi::Array<int64_t>{1}))
      .def_ro("padding", &Pool1DAttrs::padding,
              "If padding is non-zero, then the input is implicitly zero-padded"
              "Padding support both symmetric and asymmetric as"
              "one int : same padding used on all sides"
              "two int : padding width in the order of (left, right)",
              refl::DefaultValue(ffi::Array<int64_t>{0, 0}))
      .def_ro(
          "ceil_mode", &Pool1DAttrs::ceil_mode,
          "A boolean indicating if use ceil or floor to compute the output shape. By using ceil, "
          "every element in the input tensor will be covered by a sliding window.",
          refl::DefaultValue(false))
      .def_ro("count_include_pad", &Pool1DAttrs::count_include_pad,
              "When true, will include padding to compute the average")
      .def_ro("layout", &Pool1DAttrs::layout,
              "Dimension ordering of input data. Can be 'NCW', 'NWC', etc."
              "'N', 'C', 'W' stands for batch, channel, and width"
              "dimensions respectively. Pooling is applied on the 'W' dimensions.",
              refl::DefaultValue("NCW"))
      .def_ro("out_layout", &Pool1DAttrs::out_layout,
              "Dimension ordering of output data. Can be 'NCW', 'NWC', etc."
              "'N', 'C', 'W' stands for batch, channel, and width"
              "dimensions respectively. Pooling is applied on the 'W' dimensions.");
}

void Pool2DAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<Pool2DAttrs>()
      .def_ro("pool_size", &Pool2DAttrs::pool_size, "Size of the pooling windows.",
              refl::DefaultValue(ffi::Array<int64_t>{1, 1}))
      .def_ro("strides", &Pool2DAttrs::strides, "Specifies the strides of the convolution.",
              refl::DefaultValue(ffi::Array<int64_t>{1, 1}))
      .def_ro("dilation", &Pool2DAttrs::dilation, "Specifies the dilation of the convolution.",
              refl::DefaultValue(ffi::Array<int64_t>{1, 1}))
      .def_ro("padding", &Pool2DAttrs::padding,
              "If padding is non-zero, then the input is implicitly zero-padded"
              "Padding support both symmetric and asymmetric as"
              "one int : same padding used on all sides"
              "two int : bottom, right will use same padding as top, left"
              "four int : padding width in the order of (top, left, bottom, right)",
              refl::DefaultValue(ffi::Array<int64_t>{0, 0, 0, 0}))
      .def_ro(
          "ceil_mode", &Pool2DAttrs::ceil_mode,
          "A boolean indicating if use ceil or floor to compute the output shape. By using ceil, "
          "every element in the input tensor will be covered by a sliding window.",
          refl::DefaultValue(false))
      .def_ro("count_include_pad", &Pool2DAttrs::count_include_pad,
              "When true, will include padding to compute the average", refl::DefaultValue(false))
      .def_ro("layout", &Pool2DAttrs::layout,
              "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
              "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
              "dimensions respectively. Pooling is applied on the 'H' and"
              "'W' dimensions.",
              refl::DefaultValue(ffi::String("NCHW")))
      .def_ro("out_layout", &Pool2DAttrs::out_layout,
              "Dimension ordering of output data. Can be 'NCHW', 'NHWC', etc."
              "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
              "dimensions respectively. Pooling is applied on the 'H' and"
              "'W' dimensions.");
}

void Pool3DAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<Pool3DAttrs>()
      .def_ro("pool_size", &Pool3DAttrs::pool_size, "Size of the pooling windows.",
              refl::DefaultValue(ffi::Array<int64_t>{1, 1, 1}))
      .def_ro("strides", &Pool3DAttrs::strides, "Specifies the strides of the convolution.",
              refl::DefaultValue(ffi::Array<int64_t>{1, 1, 1}))
      .def_ro("dilation", &Pool3DAttrs::dilation, "Specifies the dilation of the convolution.",
              refl::DefaultValue(ffi::Array<int64_t>{1, 1, 1}))
      .def_ro("padding", &Pool3DAttrs::padding,
              "If padding is non-zero, then the input is implicitly zero-padded"
              "Padding support both symmetric and asymmetric as"
              "one int : same padding used on all sides"
              "three int : back, bottom, right will use same padding as front, top, left"
              "four int : padding width in the order of (front, top, left, back, bottom, right)",
              refl::DefaultValue(ffi::Array<int64_t>{0, 0, 0, 0, 0, 0}))
      .def_ro(
          "ceil_mode", &Pool3DAttrs::ceil_mode,
          "A boolean indicating if use ceil or floor to compute the output shape. By using ceil, "
          "every element in the input tensor will be covered by a sliding window.",
          refl::DefaultValue(false))
      .def_ro("count_include_pad", &Pool3DAttrs::count_include_pad,
              "When true, will include padding to compute the average")
      .def_ro("layout", &Pool3DAttrs::layout,
              "Dimension ordering of input data. Can be 'NCDHW', 'NDHWC', etc."
              "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
              "dimensions respectively. Pooling is applied on the 'D', 'H' and"
              "'W' dimensions.",
              refl::DefaultValue(ffi::String("NCDHW")))
      .def_ro("out_layout", &Pool3DAttrs::out_layout,
              "Dimension ordering of output data. Can be 'NCDHW', 'NDHWC', etc."
              "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
              "dimensions respectively. Pooling is applied on the 'D', 'H' and"
              "'W' dimensions.");
}

void AdaptivePool1DAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<AdaptivePool1DAttrs>()
      .def_ro("output_size", &AdaptivePool1DAttrs::output_size, "Output width.",
              refl::DefaultValue(ffi::Optional<ffi::Array<int64_t>>{}))
      .def_ro("layout", &AdaptivePool1DAttrs::layout,
              "Dimension ordering of input data. Can be 'NCW', 'NWC', etc."
              "'N', 'C', 'W' stands for batch, channel and width"
              "dimensions respectively. Pooling is applied on the"
              "'W' dimensions.",
              refl::DefaultValue(ffi::String("NCW")))
      .def_ro("out_layout", &AdaptivePool1DAttrs::out_layout,
              "Dimension ordering of output data. Can be 'NCW', 'NWC', etc."
              "'N', 'C', 'W' stands for batch, channel and width"
              "dimensions respectively. Pooling is applied on the"
              "'W' dimensions.");
}

void AdaptivePool2DAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<AdaptivePool2DAttrs>()
      .def_ro("output_size", &AdaptivePool2DAttrs::output_size, "Output height and width.",
              refl::DefaultValue(ffi::Optional<ffi::Array<int64_t>>{}))
      .def_ro("layout", &AdaptivePool2DAttrs::layout,
              "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
              "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
              "dimensions respectively. Pooling is applied on the 'H' and"
              "'W' dimensions.",
              refl::DefaultValue(ffi::String("NCHW")))
      .def_ro("out_layout", &AdaptivePool2DAttrs::out_layout,
              "Dimension ordering of output data. Can be 'NCHW', 'NHWC', etc."
              "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
              "dimensions respectively. Pooling is applied on the 'H' and"
              "'W' dimensions.");
}

void AdaptivePool3DAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<AdaptivePool3DAttrs>()
      .def_ro("output_size", &AdaptivePool3DAttrs::output_size, "Output depth, height and width.",
              refl::DefaultValue(ffi::Optional<ffi::Array<int64_t>>{}))
      .def_ro("layout", &AdaptivePool3DAttrs::layout,
              "Dimension ordering of input data. Can be 'NCDHW', 'NDHWC', etc."
              "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
              "dimensions respectively. Pooling is applied on 'D', 'H' and"
              "'W' dimensions.",
              refl::DefaultValue(ffi::String("NCDHW")))
      .def_ro("out_layout", &AdaptivePool3DAttrs::out_layout,
              "Dimension ordering of output data. Can be 'NCDHW', 'NDHWC', etc."
              "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
              "dimensions respectively. Pooling is applied on 'D', 'H' and"
              "'W' dimensions.");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  Pool1DAttrs::RegisterReflection();
  Pool2DAttrs::RegisterReflection();
  Pool3DAttrs::RegisterReflection();
  AdaptivePool1DAttrs::RegisterReflection();
  AdaptivePool2DAttrs::RegisterReflection();
  AdaptivePool3DAttrs::RegisterReflection();
}

/* relax.nn.max_pool1d */

Expr MakePool1d(ffi::String op_name, Expr data, ffi::Array<int64_t> pool_size,
                ffi::Array<int64_t> strides, ffi::Array<int64_t> padding,
                ffi::Array<int64_t> dilation, bool ceil_mode, bool count_include_pad,
                ffi::String layout, ffi::Optional<ffi::String> out_layout) {
  padding = GetCompletePadding1D(std::move(padding));

  TVM_FFI_ICHECK_EQ(pool_size.size(), 1)
      << "The input pool_size length is expected to be 1. However, the given pool_size is "
      << pool_size;
  TVM_FFI_ICHECK_EQ(strides.size(), 1)
      << "The input strides length is expected to be 1. However, the given strides is " << strides;
  TVM_FFI_ICHECK_EQ(dilation.size(), 1)
      << "The input dilation length is expected to be 1. However, the given dilation is "
      << dilation;

  auto attrs = ffi::make_object<Pool1DAttrs>();
  attrs->pool_size = std::move(pool_size);
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->dilation = std::move(dilation);
  attrs->ceil_mode = ceil_mode;
  attrs->count_include_pad = count_include_pad;
  attrs->layout = layout;
  attrs->out_layout = out_layout.value_or(layout);
  const Op op = Op::Get(op_name);
  return Call(Type::Missing(), op, {std::move(data)}, Attrs(attrs), {});
}

Expr max_pool1d(Expr data, ffi::Array<int64_t> pool_size, ffi::Array<int64_t> strides,
                ffi::Array<int64_t> padding, ffi::Array<int64_t> dilation, bool ceil_mode,
                bool count_include_pad, ffi::String layout, ffi::Optional<ffi::String> out_layout) {
  return MakePool1d("relax.nn.max_pool1d", data, pool_size, strides, padding, dilation, ceil_mode,
                    count_include_pad, layout, out_layout);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.nn.max_pool1d", max_pool1d);
}

Type InferTypePool1D(const Call& call, const BlockBuilder& ctx) {
  TensorType data_ty = GetUnaryInputTensorType(call, ctx);

  const auto* attrs = call->attrs.as<Pool1DAttrs>();
  auto [data_layout, data2NCW] = CheckTensorLayout(call, ctx, attrs->layout,
                                                   /*tgt_layout=*/"NCW",
                                                   /*tensor_name=*/"data");
  auto [out_layout, out2NCW] = CheckTensorLayout(call, ctx, attrs->out_layout,
                                                 /*tgt_layout=*/"NCW",
                                                 /*tensor_name=*/"output");

  ffi::Optional<ShapeExpr> data_shape =
      CheckNdimPerLayoutAndGetShape(call, ctx, data_ty, data_layout);
  if (!data_shape.has_value()) {
    return TensorType(data_ty->dtype, out_layout.ndim(), data_ty->vdevice);
  }

  ffi::Array<PrimExpr> data_NCW_shape = data2NCW.ForwardShape(data_shape.value()->values);

  PrimExpr input_w = data_NCW_shape[2];
  PrimExpr kernel_w = IntImm::Int32(attrs->pool_size[0]);
  PrimExpr padding_w = IntImm::Int32(attrs->padding[0]) + IntImm::Int32(attrs->padding[1]);

  sym::Analyzer analyzer = ctx->GetAnalyzer();
  std::vector<PrimExpr> out_NCW_shape;
  out_NCW_shape.reserve(3);
  out_NCW_shape.push_back(data_NCW_shape[0]);
  out_NCW_shape.push_back(data_NCW_shape[1]);

  PrimExpr numerator_w =
      input_w + padding_w - IntImm::Int32(attrs->dilation[0]) * (kernel_w - 1) - 1;
  if (attrs->ceil_mode) {
    numerator_w += IntImm::Int32(attrs->strides[0]) - 1;
  }
  PrimExpr raw_out_w = floordiv(numerator_w, IntImm::Int32(attrs->strides[0])) + 1;
  if (attrs->ceil_mode) {
    PrimExpr invalid_last_w = (raw_out_w - 1) * IntImm::Int32(attrs->strides[0]) >=
                              input_w + IntImm::Int32(attrs->padding[0]);
    out_NCW_shape.push_back(
        analyzer->Simplify(if_then_else(invalid_last_w, raw_out_w - 1, raw_out_w)));
  } else {
    out_NCW_shape.push_back(analyzer->Simplify(raw_out_w));
  }

  ffi::Array<PrimExpr> out_shape = out2NCW.BackwardShape(out_NCW_shape);
  return TensorType(ShapeExpr(out_shape), data_ty->dtype, data_ty->vdevice);
}

InferLayoutOutput InferLayoutPool1d(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  TVM_FFI_ICHECK(NoDesiredLayout(call, desired_layouts));
  const auto* tensor_ty = GetTypeAs<TensorTypeObj>(call);
  TVM_FFI_ICHECK(tensor_ty != nullptr) << "Invalid Call";
  TVM_FFI_ICHECK_EQ(tensor_ty->ndim, 3) << "Unsupported initial layout";
  const auto* attrs = call->attrs.as<Pool1DAttrs>();
  TVM_FFI_ICHECK(attrs) << "Invalid Call";

  LayoutDecision layout = GetLayoutDecision(var_layout_map, call->args[0]);
  ffi::ObjectPtr<Pool1DAttrs> new_attrs = ffi::make_object<Pool1DAttrs>(*attrs);
  new_attrs->layout =
      TransposeLike(SLayout::Create(attrs->layout), InitialLayout(3), layout->layout).name();
  new_attrs->out_layout =
      TransposeLike(SLayout::Create(attrs->out_layout), InitialLayout(3), layout->layout).name();
  return InferLayoutOutput({layout}, {layout}, Attrs(new_attrs));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.nn.max_pool1d")
      .signature(sig::arg("data", "The input tensor"), sig::call_attrs<Pool1DAttrs>())
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypePool1D)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutPool1d)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.nn.max_pool2d */

Expr MakePool2d(ffi::String op_name, Expr data, ffi::Array<int64_t> pool_size,
                ffi::Array<int64_t> strides, ffi::Array<int64_t> padding,
                ffi::Array<int64_t> dilation, bool ceil_mode, bool count_include_pad,
                ffi::String layout, ffi::Optional<ffi::String> out_layout) {
  padding = GetCompletePadding2D(std::move(padding));
  if (pool_size.size() == 1) {
    pool_size.push_back(pool_size[0]);
  }
  if (strides.size() == 1) {
    strides.push_back(strides[0]);
  }
  if (dilation.size() == 1) {
    dilation.push_back(dilation[0]);
  }

  TVM_FFI_ICHECK_EQ(pool_size.size(), 2)
      << "The input pool_size length is expected to be 2. However, the given pool_size is "
      << pool_size;
  TVM_FFI_ICHECK_EQ(strides.size(), 2)
      << "The input strides length is expected to be 2. However, the given strides is " << strides;
  TVM_FFI_ICHECK_EQ(dilation.size(), 2)
      << "The input dilation length is expected to be 2. However, the given dilation is "
      << dilation;

  auto attrs = ffi::make_object<Pool2DAttrs>();
  attrs->pool_size = std::move(pool_size);
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->dilation = std::move(dilation);
  attrs->ceil_mode = ceil_mode;
  attrs->count_include_pad = count_include_pad;
  attrs->layout = layout;
  attrs->out_layout = out_layout.value_or(layout);
  const Op op = Op::Get(op_name);
  return Call(Type::Missing(), op, {std::move(data)}, Attrs(attrs), {});
}

Expr max_pool2d(Expr data, ffi::Array<int64_t> pool_size, ffi::Array<int64_t> strides,
                ffi::Array<int64_t> padding, ffi::Array<int64_t> dilation, bool ceil_mode,
                bool count_include_pad, ffi::String layout, ffi::Optional<ffi::String> out_layout) {
  return MakePool2d("relax.nn.max_pool2d", data, pool_size, strides, padding, dilation, ceil_mode,
                    count_include_pad, layout, out_layout);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.nn.max_pool2d", max_pool2d);
}

Type InferTypePool2D(const Call& call, const BlockBuilder& ctx) {
  TensorType data_ty = GetUnaryInputTensorType(call, ctx);

  const auto* attrs = call->attrs.as<Pool2DAttrs>();
  auto [data_layout, data2NCHW] = CheckTensorLayout(call, ctx, attrs->layout,
                                                    /*tgt_layout=*/"NCHW",
                                                    /*tensor_name=*/"data");
  auto [out_layout, out2NCHW] = CheckTensorLayout(call, ctx, attrs->out_layout,
                                                  /*tgt_layout=*/"NCHW",
                                                  /*tensor_name=*/"output");

  ffi::Optional<ShapeExpr> data_shape =
      CheckNdimPerLayoutAndGetShape(call, ctx, data_ty, data_layout);
  if (!data_shape.has_value()) {
    return TensorType(data_ty->dtype, out_layout.ndim(), data_ty->vdevice);
  }

  ffi::Array<PrimExpr> data_NCHW_shape = data2NCHW.ForwardShape(data_shape.value()->values);

  PrimExpr input_h = data_NCHW_shape[2];
  PrimExpr input_w = data_NCHW_shape[3];
  PrimExpr kernel_h = IntImm::Int32(attrs->pool_size[0]);
  PrimExpr kernel_w = IntImm::Int32(attrs->pool_size[1]);
  PrimExpr padding_h = IntImm::Int32(attrs->padding[0]) + IntImm::Int32(attrs->padding[2]);
  PrimExpr padding_w = IntImm::Int32(attrs->padding[1]) + IntImm::Int32(attrs->padding[3]);

  sym::Analyzer analyzer = ctx->GetAnalyzer();
  std::vector<PrimExpr> out_NCHW_shape;
  out_NCHW_shape.reserve(4);
  out_NCHW_shape.push_back(data_NCHW_shape[0]);
  out_NCHW_shape.push_back(data_NCHW_shape[1]);

  PrimExpr numerator_h =
      input_h + padding_h - IntImm::Int32(attrs->dilation[0]) * (kernel_h - 1) - 1;
  PrimExpr numerator_w =
      input_w + padding_w - IntImm::Int32(attrs->dilation[1]) * (kernel_w - 1) - 1;
  if (attrs->ceil_mode) {
    numerator_h += IntImm::Int32(attrs->strides[0]) - 1;
    numerator_w += IntImm::Int32(attrs->strides[1]) - 1;
  }
  PrimExpr raw_out_h = floordiv(numerator_h, IntImm::Int32(attrs->strides[0])) + 1;
  PrimExpr raw_out_w = floordiv(numerator_w, IntImm::Int32(attrs->strides[1])) + 1;
  if (attrs->ceil_mode) {
    PrimExpr invalid_last_h = (raw_out_h - 1) * IntImm::Int32(attrs->strides[0]) >=
                              input_h + IntImm::Int32(attrs->padding[0]);
    PrimExpr invalid_last_w = (raw_out_w - 1) * IntImm::Int32(attrs->strides[1]) >=
                              input_w + IntImm::Int32(attrs->padding[1]);
    out_NCHW_shape.push_back(
        analyzer->Simplify(if_then_else(invalid_last_h, raw_out_h - 1, raw_out_h)));
    out_NCHW_shape.push_back(
        analyzer->Simplify(if_then_else(invalid_last_w, raw_out_w - 1, raw_out_w)));
  } else {
    out_NCHW_shape.push_back(analyzer->Simplify(raw_out_h));
    out_NCHW_shape.push_back(analyzer->Simplify(raw_out_w));
  }

  ffi::Array<PrimExpr> out_shape = out2NCHW.BackwardShape(out_NCHW_shape);
  return TensorType(ShapeExpr(out_shape), data_ty->dtype, data_ty->vdevice);
}

InferLayoutOutput InferLayoutPool2d(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  TVM_FFI_ICHECK(NoDesiredLayout(call, desired_layouts));
  const auto* tensor_ty = GetTypeAs<TensorTypeObj>(call);
  TVM_FFI_ICHECK(tensor_ty != nullptr) << "Invalid Call";
  TVM_FFI_ICHECK_EQ(tensor_ty->ndim, 4) << "Unsupported initial layout";
  const auto* attrs = call->attrs.as<Pool2DAttrs>();
  TVM_FFI_ICHECK(attrs) << "Invalid Call";

  LayoutDecision layout = GetLayoutDecision(var_layout_map, call->args[0]);
  ffi::ObjectPtr<Pool2DAttrs> new_attrs = ffi::make_object<Pool2DAttrs>(*attrs);

  if ((layout->layout.has_value() ? layout->layout.value().ndim() : 0) !=
      (layout->layout.has_value() ? layout->layout.value().ndim_primal() : 0)) {
    tvm::PrimType i64_ty = tvm::PrimType::Int(64);
    tirx::SLayout in_layout(attrs->layout, i64_ty);
    auto desired_layout =
        TransposeSubLayoutLike(attrs->layout, InitialLayout(4), layout->layout.value());
    auto data_si = GetType(call->args[0]);
    TensorType data_ty = data_si.as<TensorType>().value();
    ffi::Optional<ShapeExpr> data_shape = ffi::GetRef<ShapeExpr>(data_ty->shape.as<ShapeExprObj>());
    if (CanProveLayoutTransform(in_layout, desired_layout, data_shape.value()->values)) {
      // Not handling out_layout being different from in_layout now. Any use case ?
      new_attrs->layout = desired_layout.value().name();
      new_attrs->out_layout = desired_layout.value().name();
      return InferLayoutOutput({layout}, {layout}, Attrs(new_attrs));
    } else {
      layout = InitialLayout(4);
    }
  }

  new_attrs->layout =
      TransposeLike(SLayout::Create(attrs->layout), InitialLayout(4), layout->layout).name();
  new_attrs->out_layout =
      TransposeLike(SLayout::Create(attrs->out_layout), InitialLayout(4), layout->layout).name();
  return InferLayoutOutput({layout}, {layout}, Attrs(new_attrs));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.nn.max_pool2d")
      .signature(
          sig::arg("data", "The input tensor"), sig::call_attrs<Pool2DAttrs>(),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypePool2D)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutPool2d)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.nn.max_pool3d */

Expr MakePool3d(ffi::String op_name, Expr data, ffi::Array<int64_t> pool_size,
                ffi::Array<int64_t> strides, ffi::Array<int64_t> padding,
                ffi::Array<int64_t> dilation, bool ceil_mode, bool count_include_pad,
                ffi::String layout, ffi::Optional<ffi::String> out_layout) {
  padding = GetCompletePadding3D(std::move(padding));
  if (pool_size.size() == 1) {
    pool_size.push_back(pool_size[0]);
    pool_size.push_back(pool_size[0]);
  }
  if (strides.size() == 1) {
    strides.push_back(strides[0]);
    strides.push_back(strides[0]);
  }
  if (dilation.size() == 1) {
    dilation.push_back(dilation[0]);
    dilation.push_back(dilation[0]);
  }

  TVM_FFI_ICHECK_EQ(pool_size.size(), 3)
      << "The input pool_size length is expected to be 3. However, the given pool_size is "
      << pool_size;
  TVM_FFI_ICHECK_EQ(strides.size(), 3)
      << "The input strides length is expected to be 3. However, the given strides is " << strides;
  TVM_FFI_ICHECK_EQ(dilation.size(), 3)
      << "The input dilation length is expected to be 3. However, the given dilation is "
      << dilation;

  auto attrs = ffi::make_object<Pool3DAttrs>();
  attrs->pool_size = std::move(pool_size);
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->dilation = std::move(dilation);
  attrs->ceil_mode = ceil_mode;
  attrs->count_include_pad = count_include_pad;
  attrs->layout = layout;
  attrs->out_layout = out_layout.value_or(layout);
  const Op op = Op::Get(op_name);
  return Call(Type::Missing(), op, {std::move(data)}, Attrs(attrs), {});
}

Expr max_pool3d(Expr data, ffi::Array<int64_t> pool_size, ffi::Array<int64_t> strides,
                ffi::Array<int64_t> padding, ffi::Array<int64_t> dilation, bool ceil_mode,
                bool count_include_pad, ffi::String layout, ffi::Optional<ffi::String> out_layout) {
  return MakePool3d("relax.nn.max_pool3d", data, pool_size, strides, padding, dilation, ceil_mode,
                    count_include_pad, layout, out_layout);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.nn.max_pool3d", max_pool3d);
}

Type InferTypePool3D(const Call& call, const BlockBuilder& ctx) {
  TensorType data_ty = GetUnaryInputTensorType(call, ctx);

  const auto* attrs = call->attrs.as<Pool3DAttrs>();
  auto [data_layout, data2NCDHW] = CheckTensorLayout(call, ctx, attrs->layout,
                                                     /*tgt_layout=*/"NCDHW",
                                                     /*tensor_name=*/"data");
  auto [out_layout, out2NCDHW] = CheckTensorLayout(call, ctx, attrs->out_layout,
                                                   /*tgt_layout=*/"NCDHW",
                                                   /*tensor_name=*/"output");

  ffi::Optional<ShapeExpr> data_shape =
      CheckNdimPerLayoutAndGetShape(call, ctx, data_ty, data_layout);
  if (!data_shape.has_value()) {
    return TensorType(data_ty->dtype, out_layout.ndim(), data_ty->vdevice);
  }

  ffi::Array<PrimExpr> data_NCDHW_shape = data2NCDHW.ForwardShape(data_shape.value()->values);

  PrimExpr input_d = data_NCDHW_shape[2];
  PrimExpr input_h = data_NCDHW_shape[3];
  PrimExpr input_w = data_NCDHW_shape[4];
  PrimExpr kernel_d = IntImm::Int32(attrs->pool_size[0]);
  PrimExpr kernel_h = IntImm::Int32(attrs->pool_size[1]);
  PrimExpr kernel_w = IntImm::Int32(attrs->pool_size[2]);
  PrimExpr padding_d = IntImm::Int32(attrs->padding[0]) + IntImm::Int32(attrs->padding[3]);
  PrimExpr padding_h = IntImm::Int32(attrs->padding[1]) + IntImm::Int32(attrs->padding[4]);
  PrimExpr padding_w = IntImm::Int32(attrs->padding[2]) + IntImm::Int32(attrs->padding[5]);

  sym::Analyzer analyzer = ctx->GetAnalyzer();
  std::vector<PrimExpr> out_NCDHW_shape;
  out_NCDHW_shape.reserve(5);
  out_NCDHW_shape.push_back(data_NCDHW_shape[0]);
  out_NCDHW_shape.push_back(data_NCDHW_shape[1]);

  PrimExpr numerator_d =
      input_d + padding_d - IntImm::Int32(attrs->dilation[0]) * (kernel_d - 1) - 1;
  PrimExpr numerator_h =
      input_h + padding_h - IntImm::Int32(attrs->dilation[1]) * (kernel_h - 1) - 1;
  PrimExpr numerator_w =
      input_w + padding_w - IntImm::Int32(attrs->dilation[2]) * (kernel_w - 1) - 1;
  if (attrs->ceil_mode) {
    numerator_d += IntImm::Int32(attrs->strides[0]) - 1;
    numerator_h += IntImm::Int32(attrs->strides[1]) - 1;
    numerator_w += IntImm::Int32(attrs->strides[2]) - 1;
  }
  PrimExpr raw_out_d = floordiv(numerator_d, IntImm::Int32(attrs->strides[0])) + 1;
  PrimExpr raw_out_h = floordiv(numerator_h, IntImm::Int32(attrs->strides[1])) + 1;
  PrimExpr raw_out_w = floordiv(numerator_w, IntImm::Int32(attrs->strides[2])) + 1;
  if (attrs->ceil_mode) {
    PrimExpr invalid_last_d = (raw_out_d - 1) * IntImm::Int32(attrs->strides[0]) >=
                              input_d + IntImm::Int32(attrs->padding[0]);
    PrimExpr invalid_last_h = (raw_out_h - 1) * IntImm::Int32(attrs->strides[1]) >=
                              input_h + IntImm::Int32(attrs->padding[1]);
    PrimExpr invalid_last_w = (raw_out_w - 1) * IntImm::Int32(attrs->strides[2]) >=
                              input_w + IntImm::Int32(attrs->padding[2]);
    out_NCDHW_shape.push_back(
        analyzer->Simplify(if_then_else(invalid_last_d, raw_out_d - 1, raw_out_d)));
    out_NCDHW_shape.push_back(
        analyzer->Simplify(if_then_else(invalid_last_h, raw_out_h - 1, raw_out_h)));
    out_NCDHW_shape.push_back(
        analyzer->Simplify(if_then_else(invalid_last_w, raw_out_w - 1, raw_out_w)));
  } else {
    out_NCDHW_shape.push_back(analyzer->Simplify(raw_out_d));
    out_NCDHW_shape.push_back(analyzer->Simplify(raw_out_h));
    out_NCDHW_shape.push_back(analyzer->Simplify(raw_out_w));
  }

  ffi::Array<PrimExpr> out_shape = out2NCDHW.BackwardShape(out_NCDHW_shape);
  return TensorType(ShapeExpr(out_shape), data_ty->dtype, data_ty->vdevice);
}

InferLayoutOutput InferLayoutPool3d(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  TVM_FFI_ICHECK(NoDesiredLayout(call, desired_layouts));
  const auto* tensor_ty = GetTypeAs<TensorTypeObj>(call);
  TVM_FFI_ICHECK(tensor_ty != nullptr) << "Invalid Call";
  TVM_FFI_ICHECK_EQ(tensor_ty->ndim, 5) << "Unsupported initial layout";
  const auto* attrs = call->attrs.as<Pool3DAttrs>();
  TVM_FFI_ICHECK(attrs) << "Invalid Call";

  LayoutDecision layout = GetLayoutDecision(var_layout_map, call->args[0]);
  ffi::ObjectPtr<Pool3DAttrs> new_attrs = ffi::make_object<Pool3DAttrs>(*attrs);
  new_attrs->layout =
      TransposeLike(SLayout::Create(attrs->layout), InitialLayout(5), layout->layout).name();
  new_attrs->out_layout =
      TransposeLike(SLayout::Create(attrs->out_layout), InitialLayout(5), layout->layout).name();
  return InferLayoutOutput({layout}, {layout}, Attrs(new_attrs));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.nn.max_pool3d")
      .signature(sig::arg("data", "The input tensor"), sig::call_attrs<Pool3DAttrs>())
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypePool3D)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutPool3d)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.nn.avg_pool1d */
Expr avg_pool1d(Expr data, ffi::Array<int64_t> pool_size, ffi::Array<int64_t> strides,
                ffi::Array<int64_t> padding, ffi::Array<int64_t> dilation, bool ceil_mode,
                bool count_include_pad, ffi::String layout, ffi::Optional<ffi::String> out_layout) {
  return MakePool1d("relax.nn.avg_pool1d", data, pool_size, strides, padding, dilation, ceil_mode,
                    count_include_pad, layout, out_layout);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.nn.avg_pool1d", avg_pool1d);

  OpDef("relax.nn.avg_pool1d")
      .signature(sig::arg("data", "The input tensor"), sig::call_attrs<Pool1DAttrs>())
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypePool1D)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutPool1d)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.nn.avg_pool2d */
Expr avg_pool2d(Expr data, ffi::Array<int64_t> pool_size, ffi::Array<int64_t> strides,
                ffi::Array<int64_t> padding, ffi::Array<int64_t> dilation, bool ceil_mode,
                bool count_include_pad, ffi::String layout, ffi::Optional<ffi::String> out_layout) {
  return MakePool2d("relax.nn.avg_pool2d", data, pool_size, strides, padding, dilation, ceil_mode,
                    count_include_pad, layout, out_layout);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.nn.avg_pool2d", avg_pool2d);

  OpDef("relax.nn.avg_pool2d")
      .signature(sig::arg("data", "The input tensor"), sig::call_attrs<Pool2DAttrs>())
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypePool2D)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutPool2d)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.nn.avg_pool3d */
Expr avg_pool3d(Expr data, ffi::Array<int64_t> pool_size, ffi::Array<int64_t> strides,
                ffi::Array<int64_t> padding, ffi::Array<int64_t> dilation, bool ceil_mode,
                bool count_include_pad, ffi::String layout, ffi::Optional<ffi::String> out_layout) {
  return MakePool3d("relax.nn.avg_pool3d", data, pool_size, strides, padding, dilation, ceil_mode,
                    count_include_pad, layout, out_layout);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.nn.avg_pool3d", avg_pool3d);

  OpDef("relax.nn.avg_pool3d")
      .signature(sig::arg("data", "The input tensor"), sig::call_attrs<Pool3DAttrs>())
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypePool3D)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutPool3d)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.nn.adaptive_avg_pool1d */

Expr adaptive_avg_pool1d(Expr data, ffi::Optional<ffi::Array<int64_t>> output_size,
                         ffi::String layout, ffi::Optional<ffi::String> out_layout) {
  ffi::ObjectPtr<AdaptivePool1DAttrs> attrs = ffi::make_object<AdaptivePool1DAttrs>();
  attrs->layout = layout;
  attrs->out_layout = out_layout.value_or(layout);
  if (output_size.has_value()) {
    ffi::Array<int64_t> _output_size = output_size.value();
    TVM_FFI_ICHECK_EQ(_output_size.size(), 1)
        << "The output_size length is expected to be 1. However, the given output_size is "
        << _output_size;
    attrs->output_size = std::move(_output_size);
  }

  static const Op op = Op::Get("relax.nn.adaptive_avg_pool1d");
  return Call(Type::Missing(), op, {std::move(data)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.nn.adaptive_avg_pool1d", adaptive_avg_pool1d);
}

Type InferTypeAdaptiveAvgPool1D(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  TensorType data_ty = GetUnaryInputTensorType(call);

  const auto* attrs = call->attrs.as<AdaptivePool1DAttrs>();
  auto [data_layout, data2NCW] = CheckTensorLayout(call, attrs->layout,
                                                   /*tgt_layout=*/"NCW",
                                                   /*tensor_name=*/"data");
  auto [out_layout, out2NCW] = CheckTensorLayout(call, attrs->out_layout,
                                                 /*tgt_layout=*/"NCW",
                                                 /*tensor_name=*/"output");

  ffi::Optional<ShapeExpr> data_shape = CheckNdimPerLayoutAndGetShape(call, data_ty, data_layout);
  if (!data_shape.has_value()) {
    if (data_ty->shape.has_value() && attrs->out_layout == attrs->layout &&
        !attrs->output_size.has_value()) {
      return data_ty;
    } else {
      return TensorType(data_ty->dtype, out_layout.ndim(), data_ty->vdevice);
    }
  }

  ffi::Array<PrimExpr> data_NCW_shape = data2NCW.ForwardShape(data_shape.value()->values);
  ffi::Array<PrimExpr> out_NCW_shape(data_NCW_shape);
  if (attrs->output_size.has_value()) {
    out_NCW_shape.Set(2, IntImm::Int32(attrs->output_size.value()[0]));
  }

  ffi::Array<PrimExpr> out_shape = out2NCW.BackwardShape(out_NCW_shape);
  return TensorType(ShapeExpr(out_shape), data_ty->dtype, data_ty->vdevice);
}

InferLayoutOutput InferLayoutAdaptiveAvgPool1D(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  TVM_FFI_ICHECK(NoDesiredLayout(call, desired_layouts));
  const auto* tensor_ty = GetTypeAs<TensorTypeObj>(call);
  TVM_FFI_ICHECK(tensor_ty != nullptr) << "Invalid Call";
  TVM_FFI_ICHECK_EQ(tensor_ty->ndim, 3) << "Unsupported initial layout";
  const auto* attrs = call->attrs.as<AdaptivePool1DAttrs>();
  TVM_FFI_ICHECK(attrs) << "Invalid Call";

  LayoutDecision layout = GetLayoutDecision(var_layout_map, call->args[0]);
  ffi::ObjectPtr<AdaptivePool1DAttrs> new_attrs = ffi::make_object<AdaptivePool1DAttrs>(*attrs);
  new_attrs->layout =
      TransposeLike(SLayout::Create(attrs->layout), InitialLayout(3), layout->layout).name();
  new_attrs->out_layout =
      TransposeLike(SLayout::Create(attrs->out_layout), InitialLayout(3), layout->layout).name();
  return InferLayoutOutput({layout}, {layout}, Attrs(new_attrs));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.nn.adaptive_avg_pool1d")
      .signature(sig::arg("data", "The input tensor"), sig::call_attrs<AdaptivePool1DAttrs>())
      .set_native_fn_attr<&InferTypeAdaptiveAvgPool1D>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutAdaptiveAvgPool1D)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.nn.adaptive_avg_pool2d */

Expr adaptive_avg_pool2d(Expr data, ffi::Optional<ffi::Array<int64_t>> output_size,
                         ffi::String layout, ffi::Optional<ffi::String> out_layout) {
  ffi::ObjectPtr<AdaptivePool2DAttrs> attrs = ffi::make_object<AdaptivePool2DAttrs>();
  attrs->layout = layout;
  attrs->out_layout = out_layout.value_or(layout);
  if (output_size.has_value()) {
    ffi::Array<int64_t> _output_size = output_size.value();
    if (_output_size.size() == 1) {
      _output_size.push_back(_output_size[0]);
    }
    TVM_FFI_ICHECK_EQ(_output_size.size(), 2)
        << "The output_size length is expected to be 2. However, the given output_size is "
        << _output_size;
    attrs->output_size = std::move(_output_size);
  }

  static const Op op = Op::Get("relax.nn.adaptive_avg_pool2d");
  return Call(Type::Missing(), op, {std::move(data)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.nn.adaptive_avg_pool2d", adaptive_avg_pool2d);
}

Type InferTypeAdaptiveAvgPool2D(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  TensorType data_ty = GetUnaryInputTensorType(call);

  const auto* attrs = call->attrs.as<AdaptivePool2DAttrs>();
  auto [data_layout, data2NCHW] = CheckTensorLayout(call, attrs->layout,
                                                    /*tgt_layout=*/"NCHW",
                                                    /*tensor_name=*/"data");
  auto [out_layout, out2NCHW] = CheckTensorLayout(call, attrs->out_layout,
                                                  /*tgt_layout=*/"NCHW",
                                                  /*tensor_name=*/"output");

  ffi::Optional<ShapeExpr> data_shape = CheckNdimPerLayoutAndGetShape(call, data_ty, data_layout);
  if (!data_shape.has_value()) {
    if (data_ty->shape.has_value() && attrs->out_layout == attrs->layout &&
        !attrs->output_size.has_value()) {
      return data_ty;
    } else {
      return TensorType(data_ty->dtype, out_layout.ndim(), data_ty->vdevice);
    }
  }

  ffi::Array<PrimExpr> data_NCHW_shape = data2NCHW.ForwardShape(data_shape.value()->values);
  ffi::Array<PrimExpr> out_NCHW_shape(data_NCHW_shape);
  if (attrs->output_size.has_value()) {
    out_NCHW_shape.Set(2, IntImm::Int32(attrs->output_size.value()[0]));
    out_NCHW_shape.Set(3, IntImm::Int32(attrs->output_size.value()[1]));
  }

  ffi::Array<PrimExpr> out_shape = out2NCHW.BackwardShape(out_NCHW_shape);
  return TensorType(ShapeExpr(out_shape), data_ty->dtype, data_ty->vdevice);
}

InferLayoutOutput InferLayoutAdaptiveAvgPool2D(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  TVM_FFI_ICHECK(NoDesiredLayout(call, desired_layouts));
  const auto* tensor_ty = GetTypeAs<TensorTypeObj>(call);
  TVM_FFI_ICHECK(tensor_ty != nullptr) << "Invalid Call";
  TVM_FFI_ICHECK_EQ(tensor_ty->ndim, 4) << "Unsupported initial layout";
  const auto* attrs = call->attrs.as<AdaptivePool2DAttrs>();
  TVM_FFI_ICHECK(attrs) << "Invalid Call";

  LayoutDecision layout = GetLayoutDecision(var_layout_map, call->args[0]);
  ffi::ObjectPtr<AdaptivePool2DAttrs> new_attrs = ffi::make_object<AdaptivePool2DAttrs>(*attrs);
  if ((layout->layout.has_value() ? layout->layout.value().ndim() : 0) !=
      (layout->layout.has_value() ? layout->layout.value().ndim_primal() : 0)) {
    tvm::PrimType i64_ty = tvm::PrimType::Int(64);
    tirx::SLayout in_layout(attrs->layout, i64_ty);
    auto desired_layout =
        TransposeSubLayoutLike(attrs->layout, InitialLayout(4), layout->layout.value());
    auto data_si = GetType(call->args[0]);
    TensorType data_ty = data_si.as<TensorType>().value();
    ffi::Optional<ShapeExpr> data_shape = ffi::GetRef<ShapeExpr>(data_ty->shape.as<ShapeExprObj>());
    if (CanProveLayoutTransform(in_layout, desired_layout, data_shape.value()->values)) {
      // Not handling out_layout being different from in_layout now. Any use case ?
      new_attrs->layout = desired_layout.value().name();
      new_attrs->out_layout = desired_layout.value().name();
      return InferLayoutOutput({layout}, {layout}, Attrs(new_attrs));
    } else {
      layout = InitialLayout(4);
    }
  }
  new_attrs->layout =
      TransposeLike(SLayout::Create(attrs->layout), InitialLayout(4), layout->layout).name();
  new_attrs->out_layout =
      TransposeLike(SLayout::Create(attrs->out_layout), InitialLayout(4), layout->layout).name();
  return InferLayoutOutput({layout}, {layout}, Attrs(new_attrs));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.nn.adaptive_avg_pool2d")
      .signature(
          sig::arg("data", "The input tensor"), sig::call_attrs<AdaptivePool2DAttrs>(),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_native_fn_attr<&InferTypeAdaptiveAvgPool2D>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutAdaptiveAvgPool2D)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.nn.adaptive_avg_pool3d */

Expr adaptive_avg_pool3d(Expr data, ffi::Optional<ffi::Array<int64_t>> output_size,
                         ffi::String layout, ffi::Optional<ffi::String> out_layout) {
  ffi::ObjectPtr<AdaptivePool3DAttrs> attrs = ffi::make_object<AdaptivePool3DAttrs>();
  attrs->layout = layout;
  attrs->out_layout = out_layout.value_or(layout);
  if (output_size.has_value()) {
    ffi::Array<int64_t> _output_size = output_size.value();
    if (_output_size.size() == 1) {
      _output_size.push_back(_output_size[0]);
    }
    TVM_FFI_ICHECK_EQ(_output_size.size(), 3)
        << "The output_size length is expected to be 3. However, the given output_size is "
        << _output_size;
    attrs->output_size = std::move(_output_size);
  }

  static const Op op = Op::Get("relax.nn.adaptive_avg_pool3d");
  return Call(Type::Missing(), op, {std::move(data)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.nn.adaptive_avg_pool3d", adaptive_avg_pool3d);
}

Type InferTypeAdaptiveAvgPool3D(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  TensorType data_ty = GetUnaryInputTensorType(call);

  const auto* attrs = call->attrs.as<AdaptivePool3DAttrs>();
  auto [data_layout, data2NCDHW] = CheckTensorLayout(call, attrs->layout,
                                                     /*tgt_layout=*/"NCDHW",
                                                     /*tensor_name=*/"data");
  auto [out_layout, out2NCDHW] = CheckTensorLayout(call, attrs->out_layout,
                                                   /*tgt_layout=*/"NCDHW",
                                                   /*tensor_name=*/"output");

  ffi::Optional<ShapeExpr> data_shape = CheckNdimPerLayoutAndGetShape(call, data_ty, data_layout);
  if (!data_shape.has_value()) {
    if (data_ty->shape.has_value() && attrs->out_layout == attrs->layout &&
        !attrs->output_size.has_value()) {
      return data_ty;
    } else {
      return TensorType(data_ty->dtype, out_layout.ndim(), data_ty->vdevice);
    }
  }

  ffi::Array<PrimExpr> data_NCDHW_shape = data2NCDHW.ForwardShape(data_shape.value()->values);
  ffi::Array<PrimExpr> out_NCDHW_shape(data_NCDHW_shape);
  if (attrs->output_size.has_value()) {
    out_NCDHW_shape.Set(2, IntImm::Int32(attrs->output_size.value()[0]));
    out_NCDHW_shape.Set(3, IntImm::Int32(attrs->output_size.value()[1]));
    out_NCDHW_shape.Set(4, IntImm::Int32(attrs->output_size.value()[2]));
  }

  ffi::Array<PrimExpr> out_shape = out2NCDHW.BackwardShape(out_NCDHW_shape);
  return TensorType(ShapeExpr(out_shape), data_ty->dtype, data_ty->vdevice);
}

InferLayoutOutput InferLayoutAdaptiveAvgPool3D(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  TVM_FFI_ICHECK(NoDesiredLayout(call, desired_layouts));
  const auto* tensor_ty = GetTypeAs<TensorTypeObj>(call);
  TVM_FFI_ICHECK(tensor_ty != nullptr) << "Invalid Call";
  TVM_FFI_ICHECK_EQ(tensor_ty->ndim, 5) << "Unsupported initial layout";
  const auto* attrs = call->attrs.as<AdaptivePool3DAttrs>();
  TVM_FFI_ICHECK(attrs) << "Invalid Call";

  LayoutDecision layout = GetLayoutDecision(var_layout_map, call->args[0]);
  ffi::ObjectPtr<AdaptivePool3DAttrs> new_attrs = ffi::make_object<AdaptivePool3DAttrs>(*attrs);
  new_attrs->layout =
      TransposeLike(SLayout::Create(attrs->layout), InitialLayout(5), layout->layout).name();
  new_attrs->out_layout =
      TransposeLike(SLayout::Create(attrs->out_layout), InitialLayout(5), layout->layout).name();
  return InferLayoutOutput({layout}, {layout}, Attrs(new_attrs));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.nn.adaptive_avg_pool3d")
      .signature(sig::arg("data", "The input tensor"), sig::call_attrs<AdaptivePool3DAttrs>())
      .set_native_fn_attr<&InferTypeAdaptiveAvgPool3D>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutAdaptiveAvgPool3D)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

}  // namespace relax
}  // namespace tvm
