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
 * \file statistical.cc
 * \brief Statistical operators.
 */

#include "statistical.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/relax/op_attr.h>

#include <string>
#include <vector>

namespace tvm {
namespace relax {

void StatisticalAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<StatisticalAttrs>()
      .def_ro("axis", &StatisticalAttrs::axis,
              "The axis or axes along which to perform the reduction.",
              refl::DefaultValue(ffi::Optional<ffi::Array<int64_t>>{}))
      .def_ro("keepdims", &StatisticalAttrs::keepdims,
              "If this is set to `True`, the reduced axes are left in the result as dimension "
              "with size "
              "one.",
              refl::DefaultValue(false));
}

void ScanopAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<ScanopAttrs>()
      .def_ro("axis", &ScanopAttrs::axis,
              "The axis along which to perform the scan computation."
              "The default (None) is to compute over the flattened array.",
              refl::DefaultValue(ffi::Optional<int64_t>{}))
      .def_ro("dtype", &ScanopAttrs::dtype,
              "The output data type."
              "If dtype is not specified, it defaults to the dtype of input data.",
              refl::DefaultValue(ffi::Optional<DLDataType>{}))
      .def_ro("exclusive", &ScanopAttrs::exclusive, "The first element is not included",
              refl::DefaultValue(false));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  StatisticalAttrs::RegisterReflection();
  ScanopAttrs::RegisterReflection();
}

Type InferTypeStatistical(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  if (RequiresTensorInputNormalization(call)) return Type::Missing();
  TensorType data_ty = GetUnaryInputTensorType(call);
  const auto* attrs = call->attrs.as<StatisticalAttrs>();

  std::vector<int> axes;
  if (!data_ty->IsUnknownNdim() && attrs->axis.has_value()) {
    axes = NormalizeAxes(call, data_ty->ndim, attrs->axis.value());
  }

  int out_ndim;
  if (attrs->keepdims) {
    out_ndim = data_ty->ndim;
  } else if (!attrs->axis.has_value()) {
    out_ndim = 0;
  } else if (data_ty->IsUnknownNdim()) {
    out_ndim = kUnknownNDim;
  } else {
    out_ndim = data_ty->ndim - axes.size();
    TVM_FFI_ICHECK_GE(out_ndim, 0);
  }

  // The inference rule for reduction operator output shapes:
  // - axes is None, keepdims is false -> return the zero-rank shape;
  // - axes is None, keepdims is true -> return the shape whose ndim is the same as input and every
  // value is 1.
  // - axes is not None, keepdims is false -> the returned shape does not contain the input axes.
  // - axes is not None, keepdims is true -> the returned shape has value 1 at the positions of the
  // input axes
  const auto* data_shape = data_ty->shape.as<ShapeExprObj>();
  if (data_shape == nullptr) {
    if (!attrs->axis.has_value() && attrs->keepdims && out_ndim != kUnknownNDim) {
      return TensorType(ShapeExpr(ffi::Array<PrimExpr>(out_ndim, IntImm::Int64(/*value=*/1))),
                        data_ty->dtype, data_ty->vdevice);
    } else {
      return out_ndim == 0
                 ? TensorType(ShapeExpr(ffi::Array<PrimExpr>()), data_ty->dtype, data_ty->vdevice)
                 : TensorType(data_ty->dtype, out_ndim, data_ty->vdevice);
    }
  }

  ffi::Array<PrimExpr> out_shape;
  out_shape.reserve(out_ndim);
  for (int i = 0; i < data_ty->ndim; ++i) {
    if (attrs->axis.has_value() && std::find(axes.begin(), axes.end(), i) == axes.end()) {
      out_shape.push_back(data_shape->values[i]);
    } else if (attrs->keepdims) {
      out_shape.push_back(IntImm::Int64(/*value=*/1));
    }
  }
  TVM_FFI_ICHECK_EQ(static_cast<int>(out_shape.size()), out_ndim);
  return TensorType(ShapeExpr(out_shape), data_ty->dtype, data_ty->vdevice);
}

InferLayoutOutput InferLayoutStatistical(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  TVM_FFI_ICHECK(NoDesiredLayout(call, desired_layouts));

  const auto* attrs = call->attrs.as<StatisticalAttrs>();
  TVM_FFI_ICHECK(attrs != nullptr) << "Invalid Call";
  const auto* tensor_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
  TVM_FFI_ICHECK(tensor_ty != nullptr) << "Invalid Call";
  TVM_FFI_ICHECK(!tensor_ty->IsUnknownNdim()) << "Only support known ndim";
  int ndim = tensor_ty->ndim;

  ffi::Array<int64_t> axis;
  if (attrs->axis.has_value()) {
    axis = attrs->axis.value();
  } else {
    axis.reserve(ndim);
    for (int i = 0; i < ndim; ++i) {
      axis.push_back(i);
    }
  }

  std::string axis_str(ndim, '0');
  for (int64_t iter : axis) {
    axis_str[(iter + ndim) % ndim] = '#';
  }
  for (int i = 0, j = 0; i < ndim; ++i) {
    if (axis_str[i] != '#') {
      axis_str[i] = 'A' + j++;
    }
  }

  LayoutDecision exisiting_layout = GetLayoutDecision(var_layout_map, call->args[0]);
  auto new_axis_str = TransposeSubLayoutStrLike(
      axis_str, InitialLayout(ndim).name(),
      (exisiting_layout->layout.has_value() ? exisiting_layout->layout.value().name()
                                            : "__undef__"));
  std::string output_layout_ref = new_axis_str;
  new_axis_str.erase(std::remove_if(new_axis_str.begin(), new_axis_str.end(),
                                    [](unsigned char c) { return std::isdigit(c); }),
                     new_axis_str.end());

  ffi::Array<int64_t> new_axis;
  for (size_t i = 0; i < new_axis_str.size(); ++i) {
    if (new_axis_str.at(i) == '#') {
      new_axis.push_back(static_cast<int64_t>(i));
    }
  }
  std::string output_layout;
  for (size_t i = 0; i < output_layout_ref.length(); ++i) {
    if ((isdigit(output_layout_ref[i]) && (output_layout_ref[i + 1] == '#')) ||
        (output_layout_ref[i] == '#'))
      continue;
    output_layout.push_back(output_layout_ref[i]);
  }

  ffi::ObjectPtr<StatisticalAttrs> new_attrs = ffi::make_object<StatisticalAttrs>(*attrs);
  new_attrs->axis = new_axis;
  return InferLayoutOutput(
      {exisiting_layout},
      {attrs->keepdims ? exisiting_layout : LayoutDecision(SLayout::Create(output_layout))},
      Attrs(new_attrs));
}

Type InferTypeScan(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  TensorType data_ty = GetUnaryInputTensorType(call);
  const auto* attrs = call->attrs.as<ScanopAttrs>();

  ffi::Optional<PrimType> out_type = attrs->dtype.has_value()
                                         ? ffi::Optional<PrimType>(PrimType(attrs->dtype.value()))
                                         : data_ty->dtype;

  if (!attrs->axis.has_value()) {
    // flattened
    const auto* data_shape = data_ty->shape.as<ShapeExprObj>();
    if (data_shape == nullptr) {
      return TensorType(out_type, data_ty->ndim, data_ty->vdevice);
    } else {
      PrimExpr flattened_d = 1;
      for (const auto v : data_shape->values) {
        flattened_d *= v;
      }
      return TensorType(ShapeExpr(ffi::Array<PrimExpr>({flattened_d})), out_type, data_ty->vdevice);
    }
  }

  if (data_ty->shape.has_value()) {
    return TensorType(data_ty->shape.value(), out_type, data_ty->vdevice);
  } else {
    return TensorType(out_type, data_ty->ndim, data_ty->vdevice);
  }
}

Type InferTypeStatisticalExtension(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  TensorType data_ty = GetUnaryInputTensorType(call);
  const auto* attrs = call->attrs.as<StatisticalAttrs>();

  std::vector<int> axes;
  if (!data_ty->IsUnknownNdim() && attrs->axis.has_value()) {
    axes = NormalizeAxes(call, data_ty->ndim, attrs->axis.value());
  }

  int out_ndim;
  if (attrs->keepdims) {
    out_ndim = data_ty->ndim;
  } else if (!attrs->axis.has_value()) {
    out_ndim = 0;
  } else if (data_ty->IsUnknownNdim()) {
    out_ndim = kUnknownNDim;
  } else {
    out_ndim = data_ty->ndim - axes.size();
    TVM_FFI_ICHECK_GE(out_ndim, 0);
  }

  // The inference rule for median operator output shapes:
  // - axes is None || len(axes) > 1, keepdims is false -> return the zero-rank shape;
  // - axes is None || len(axes) > 1, keepdims is true -> return the shape whose ndim
  // is the same as input and every value is 1.
  // - len(axes) == 1, keepdims is false -> the returned shape does not contain the input axis.
  // - len(axes) == 1, keepdims is true -> the returned shape has value 1 at the positions of the
  // input axis
  const auto* data_shape = data_ty->shape.as<ShapeExprObj>();
  if (data_shape == nullptr) {
    if (!attrs->axis.has_value() && attrs->keepdims && out_ndim != kUnknownNDim) {
      return TensorType(ShapeExpr(ffi::Array<PrimExpr>(out_ndim, IntImm::Int64(/*value=*/1))),
                        data_ty->dtype, data_ty->vdevice);
    }
    if (out_ndim == 0) {
      return TensorType(ShapeExpr(ffi::Array<PrimExpr>()), data_ty->dtype, data_ty->vdevice);
    }
    return TupleType({TensorType(data_ty->dtype, out_ndim, data_ty->vdevice),
                      TensorType(PrimType::Int(64), out_ndim, data_ty->vdevice)});
  }

  ffi::Array<PrimExpr> out_shape;
  out_shape.reserve(out_ndim);
  for (int i = 0; i < data_ty->ndim; ++i) {
    if (attrs->axis.has_value() && std::find(axes.begin(), axes.end(), i) == axes.end()) {
      out_shape.push_back(data_shape->values[i]);
    } else if (attrs->keepdims) {
      out_shape.push_back(IntImm::Int64(/*value=*/1));
    }
  }
  TVM_FFI_ICHECK_EQ(static_cast<int>(out_shape.size()), out_ndim);

  if (!attrs->axis.has_value() || axes.size() > 1)
    return TensorType(ShapeExpr(out_shape), data_ty->dtype, data_ty->vdevice);
  else
    return TupleType({TensorType(ShapeExpr(out_shape), data_ty->dtype, data_ty->vdevice),
                      TensorType(ShapeExpr(out_shape), PrimType::Int(64), data_ty->vdevice)});
}

/* relax.cumprod */
Expr cumprod(Expr data, ffi::Optional<int64_t> axis, ffi::Optional<DLDataType> dtype,
             bool exclusive) {
  auto attrs = ffi::make_object<ScanopAttrs>();
  attrs->axis = std::move(axis);
  attrs->dtype = dtype;
  attrs->exclusive = exclusive;

  static const Op op = Op::Get("relax.cumprod");
  return Call(Type::Missing(), op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.cumprod", cumprod);

  OpDef("relax.cumprod")
      .signature(sig::arg("data", "The input tensor."), sig::call_attrs<ScanopAttrs>())
      .set_native_fn_attr<&InferTypeScan>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.cumsum */
Expr cumsum(Expr data, ffi::Optional<int64_t> axis, ffi::Optional<DLDataType> dtype,
            bool exclusive) {
  auto attrs = ffi::make_object<ScanopAttrs>();
  attrs->axis = std::move(axis);
  attrs->dtype = dtype;
  attrs->exclusive = exclusive;

  static const Op op = Op::Get("relax.cumsum");
  return Call(Type::Missing(), op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.cumsum", cumsum);

  OpDef("relax.cumsum")
      .signature(sig::arg("data", "The input tensor."), sig::call_attrs<ScanopAttrs>())
      .set_native_fn_attr<&InferTypeScan>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.median */
Expr median(Expr data, ffi::Optional<ffi::Array<int64_t>> axis, bool keepdims) {
  ffi::ObjectPtr<StatisticalAttrs> attrs = ffi::make_object<StatisticalAttrs>();
  attrs->axis = std::move(axis);
  attrs->keepdims = keepdims;
  static const Op op = Op::Get("relax.median");
  return Call(Type::Missing(), op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.median", median);

  OpDef("relax.median")
      .signature(sig::arg("data", "The input tensor."), sig::call_attrs<StatisticalAttrs>())
      .set_native_fn_attr<&InferTypeStatisticalExtension>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

Expr max(Expr x, ffi::Optional<ffi::Array<int64_t>> axis, bool keepdims) {
  ffi::ObjectPtr<StatisticalAttrs> attrs = ffi::make_object<StatisticalAttrs>();
  attrs->axis = std::move(axis);
  attrs->keepdims = keepdims;
  static const Op op = Op::Get("relax.max");
  return Call(Type::Missing(), op, {std::move(x)}, Attrs{attrs}, {});
}

Expr mean(Expr x, ffi::Optional<ffi::Array<int64_t>> axis, bool keepdims) {
  ffi::ObjectPtr<StatisticalAttrs> attrs = ffi::make_object<StatisticalAttrs>();
  attrs->axis = std::move(axis);
  attrs->keepdims = keepdims;
  static const Op op = Op::Get("relax.mean");
  return Call(Type::Missing(), op, {std::move(x)}, Attrs{attrs}, {});
}

Expr min(Expr x, ffi::Optional<ffi::Array<int64_t>> axis, bool keepdims) {
  ffi::ObjectPtr<StatisticalAttrs> attrs = ffi::make_object<StatisticalAttrs>();
  attrs->axis = std::move(axis);
  attrs->keepdims = keepdims;
  static const Op op = Op::Get("relax.min");
  return Call(Type::Missing(), op, {std::move(x)}, Attrs{attrs}, {});
}

Expr prod(Expr x, ffi::Optional<ffi::Array<int64_t>> axis, bool keepdims) {
  ffi::ObjectPtr<StatisticalAttrs> attrs = ffi::make_object<StatisticalAttrs>();
  attrs->axis = std::move(axis);
  attrs->keepdims = keepdims;
  static const Op op = Op::Get("relax.prod");
  return Call(Type::Missing(), op, {std::move(x)}, Attrs{attrs}, {});
}

Expr std(Expr x, ffi::Optional<ffi::Array<int64_t>> axis, bool keepdims) {
  ffi::ObjectPtr<StatisticalAttrs> attrs = ffi::make_object<StatisticalAttrs>();
  attrs->axis = std::move(axis);
  attrs->keepdims = keepdims;
  static const Op op = Op::Get("relax.std");
  return Call(Type::Missing(), op, {std::move(x)}, Attrs{attrs}, {});
}

Expr sum(Expr x, ffi::Optional<ffi::Array<int64_t>> axis, bool keepdims) {
  ffi::ObjectPtr<StatisticalAttrs> attrs = ffi::make_object<StatisticalAttrs>();
  attrs->axis = std::move(axis);
  attrs->keepdims = keepdims;
  static const Op op = Op::Get("relax.sum");
  return Call(Type::Missing(), op, {std::move(x)}, Attrs{attrs}, {});
}

Expr variance(Expr x, ffi::Optional<ffi::Array<int64_t>> axis, bool keepdims) {
  ffi::ObjectPtr<StatisticalAttrs> attrs = ffi::make_object<StatisticalAttrs>();
  attrs->axis = std::move(axis);
  attrs->keepdims = keepdims;
  static const Op op = Op::Get("relax.variance");
  return Call(Type::Missing(), op, {std::move(x)}, Attrs{attrs}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::ffi::reflection::GlobalDef().def("relax.op.max", max);

  OpDef("relax.max")
      .signature(sig::arg("x", "The input data tensor"), sig::call_attrs<StatisticalAttrs>())
      .set_native_fn_attr<&InferTypeStatistical>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutStatistical)
      .set_attr(tvm::relax::op_attr::kPurity, true);

  tvm::ffi::reflection::GlobalDef().def("relax.op.mean", mean);

  OpDef("relax.mean")
      .signature(sig::arg("x", "The input data tensor"), sig::call_attrs<StatisticalAttrs>())
      .set_native_fn_attr<&InferTypeStatistical>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutStatistical)
      .set_attr(tvm::relax::op_attr::kPurity, true);

  tvm::ffi::reflection::GlobalDef().def("relax.op.min", min);

  OpDef("relax.min")
      .signature(sig::arg("x", "The input data tensor"), sig::call_attrs<StatisticalAttrs>())
      .set_native_fn_attr<&InferTypeStatistical>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutStatistical)
      .set_attr(tvm::relax::op_attr::kPurity, true);

  tvm::ffi::reflection::GlobalDef().def("relax.op.prod", prod);

  OpDef("relax.prod")
      .signature(sig::arg("x", "The input data tensor"), sig::call_attrs<StatisticalAttrs>())
      .set_native_fn_attr<&InferTypeStatistical>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutStatistical)
      .set_attr(tvm::relax::op_attr::kPurity, true);

  tvm::ffi::reflection::GlobalDef().def("relax.op.std", std);

  OpDef("relax.std")
      .signature(sig::arg("x", "The input data tensor"), sig::call_attrs<StatisticalAttrs>())
      .set_native_fn_attr<&InferTypeStatistical>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutStatistical)
      .set_attr(tvm::relax::op_attr::kPurity, true);

  tvm::ffi::reflection::GlobalDef().def("relax.op.sum", sum);

  OpDef("relax.sum")
      .signature(sig::arg("x", "The input data tensor"), sig::call_attrs<StatisticalAttrs>())
      .set_native_fn_attr<&InferTypeStatistical>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutStatistical)
      .set_attr(tvm::relax::op_attr::kPurity, true);

  tvm::ffi::reflection::GlobalDef().def("relax.op.variance", variance);

  OpDef("relax.variance")
      .signature(sig::arg("x", "The input data tensor"), sig::call_attrs<StatisticalAttrs>())
      .set_native_fn_attr<&InferTypeStatistical>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutStatistical)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

}  // namespace relax
}  // namespace tvm
