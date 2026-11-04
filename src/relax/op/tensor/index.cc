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
 * \file index.cc
 * \brief indexing operators.
 */

#include "index.h"

#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/op_attr.h>
#include <tvm/runtime/logging.h>
#include <tvm/topi/transform.h>

#include <algorithm>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace tvm {
namespace relax {
using namespace tvm::prim;

void TakeAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TakeAttrs>()
      .def_ro("axis", &TakeAttrs::axis, "The axis over which to select values.",
              refl::DefaultValue(ffi::Optional<int64_t>{}))
      .def_ro("mode", &TakeAttrs::mode, "The mode for handling out-of-bounds indices.",
              refl::DefaultValue("fast"));
}

void StridedSliceAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<StridedSliceAttrs>().def_ro(
      "assume_inbound", &StridedSliceAttrs::assume_inbound,
      "Whether to assume the indices are in bound. If it is set to false, "
      "out of bound indices will be clipped to the bound.",
      refl::DefaultValue(true));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  TakeAttrs::RegisterReflection();
  StridedSliceAttrs::RegisterReflection();
}

/* relax.take */

Expr take(Expr x, Expr indices, ffi::Optional<int64_t> axis, ffi::String mode) {
  ffi::ObjectPtr<TakeAttrs> attrs = ffi::make_object<TakeAttrs>();
  attrs->axis = std::move(axis);
  attrs->mode = std::move(mode);

  static const Op op = Op::Get("relax.take");
  return Call(Type::Missing(), op, {std::move(x), std::move(indices)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.take", take);
}

Type InferTypeTake(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  if (RequiresTensorInputNormalization(call)) return Type::Missing();
  CheckNumArguments(call);
  TensorType data_ty = GetInputTensorType(call, 0);

  // Type inference when the index is a PrimExpr is equivalent
  // to that of a scalar (0-d) tensor.
  TensorType indices_ty = [&]() {
    auto arg = call->args[1];
    auto ty = GetType(arg);
    if (auto tensor_ty = ty.as<TensorType>()) {
      return tensor_ty.value();
    } else if (auto prim_ty = ty.as<PrimTypeObj>()) {
      return TensorType(ShapeExpr(ffi::Array<PrimExpr>{}), ffi::GetRef<PrimType>(prim_ty));
    } else {
      TVM_FFI_VISIT_THROW(TypeError, call)
          << "Operator " << call->op << " requires the indices argument to be "
          << "either a tensor or a scalar value.  "
          << "However, argument " << arg << " has type " << ty;
      TVM_FFI_UNREACHABLE();
    }
  }();

  if (indices_ty->IsUnknownDtype()) {
    LOG(WARNING) << "Data type of indices has not been specified. Assume it has an integer type.";
  } else {
    PrimType indices_dtype = indices_ty->dtype.value();
    if (!indices_dtype.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) {
      TVM_FFI_VISIT_THROW(TypeError, call)
          << "Take op requires the input indices to have integer dtype. However, the "
             "given indices dtype is "
          << indices_ty->dtype;
    }
  }

  const auto* attrs = call->attrs.as<TakeAttrs>();
  if (!attrs->axis.has_value() && data_ty->ndim != 1) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Take op expects the input data to be 1-dimensional tensor when the axis "
           "is not specified. However, the given data tensor has ndim "
        << data_ty->ndim;
  }
  if (data_ty->IsUnknownNdim() || indices_ty->IsUnknownNdim()) {
    return TensorType(data_ty->dtype, kUnknownNDim, data_ty->vdevice);
  }

  int axis = 0;
  if (attrs->axis.has_value()) {
    axis = NormalizeAxis(call, data_ty->ndim, attrs->axis.value());
  }
  const auto* data_shape = data_ty->shape.as<ShapeExprObj>();
  const auto* indices_shape = indices_ty->shape.as<ShapeExprObj>();
  if (data_shape == nullptr || indices_shape == nullptr) {
    return TensorType(data_ty->dtype, indices_ty->ndim + data_ty->ndim - 1, data_ty->vdevice);
  }

  ffi::Array<PrimExpr> output_shape;
  for (int i = 0; i < data_ty->ndim; i++) {
    if (i == axis) {
      for (int j = 0; j < indices_ty->ndim; j++) output_shape.push_back(indices_shape->values[j]);
    } else {
      output_shape.push_back(data_shape->values[i]);
    }
  }
  return TensorType(ShapeExpr(output_shape), data_ty->dtype, data_ty->vdevice);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.take")
      .signature(sig::arg("x", "The source tensor."),
                 sig::arg("indices", "The indices of the values to extract."),
                 sig::call_attrs<TakeAttrs>())
      .set_native_fn_attr<&InferTypeTake>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.strided_slice */

Expr strided_slice(Expr x, Expr axes, Expr begin, Expr end, ffi::Optional<Expr> strides,
                   bool assume_inbound) {
  // Initial validation of the arguments.  A more complete validation
  // will be done when inferring the Type, but that requires the
  // Type of all arguments to be populated.

  std::optional<std::tuple<const char*, size_t, Expr>> known_length;
  auto check_tuple = [&known_length](const char* name, Expr expr) {
    if (const auto* tuple = expr.as<TupleObj>()) {
      size_t length = tuple->fields.size();
      if (known_length.has_value()) {
        const auto& prev = known_length.value();
        TVM_FFI_ICHECK_EQ(length, std::get<size_t>(prev))
            << "The strided_slice operator requires that "
            << "the axes, begin, end, and strides tuples are all the same length.  "
            << "However, the " << std::get<const char*>(prev) << " argument ("
            << std::get<Expr>(prev) << ") has " << std::get<size_t>(prev) << " elements, while the "
            << name << " argument (" << expr << ") has " << length << " elements.";
      } else {
        known_length = std::tuple{name, length, expr};
      }
    }
  };
  check_tuple("axes", axes);
  check_tuple("begin", begin);
  check_tuple("end", end);
  if (strides.has_value()) check_tuple("strides", strides.value());

  ffi::ObjectPtr<StridedSliceAttrs> attrs = ffi::make_object<StridedSliceAttrs>();
  attrs->assume_inbound = assume_inbound;

  ffi::Array<Expr> args = {x, axes, begin, end};
  if (strides.has_value()) {
    args.push_back(strides.value());
  }

  static const Op op = Op::Get("relax.strided_slice");
  auto call = Call(Type::Missing(), op, args, Attrs(attrs));

  return call;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.strided_slice", strided_slice);
}

/* \brief Helper function to unpack a relax::Tuple
 *
 * A `relax::Tuple` may be provided to an operator as an in-line
 * expression, as a variable bound to known tuple within the current
 * function, as a function argument, etc.  This overload validates that
 * the Type could contain a tuple of `PrimExpr` elements.  Without a
 * concrete tuple expression, the values are not statically known.
 *
 * If the Type cannot contain a tuple of the type specified,
 * this function will throw an exception.  (e.g. Attempting to extract
 * a tuple from a `TensorType`.)
 *
 * \tparam PrimType The subtype of PrimExpr to extract.  For example,
 *     extracting an `ffi::Array<int64_t>`
 *
 * \param ty The Type to inspect
 *
 * \returns An empty array for an empty tuple, if it can be extracted.
 *     Otherwise, `std::nullopt`.
 */
template <typename PrimType = PrimExpr,
          typename = std::enable_if_t<std::is_base_of_v<PrimExpr, PrimType>>>
ffi::Optional<ffi::Array<PrimType>> UnpackTupleOfPrimExpr(ffi::Optional<Type> ty) {
  if (!ty) return std::nullopt;

  // An AnyType may contain a tuple of the desired type, but
  // it isn't yet known whether it does.  Return early, as we cannot
  // provide a known `ffi::Array<PrimType>` to the caller.
  if (ty.as<AnyTypeObj>()) return std::nullopt;

  auto tuple = ty.as<TupleTypeObj>();
  TVM_FFI_CHECK(tuple, TypeError) << "The type " << ty
                                  << " cannot contain a tuple whose elements are "
                                  << PrimType::ContainerType::_type_key;

  ffi::Array<PrimType> output;
  for (size_t i = 0; i < tuple->fields.size(); i++) {
    auto field = tuple->fields[i];

    if (field.as<AnyTypeObj>()) return std::nullopt;

    auto prim_ty = field.as<PrimTypeObj>();
    TVM_FFI_CHECK(prim_ty, TypeError)
        << "The type " << ty << " cannot contain a tuple whose elements are "
        << PrimType::ContainerType::_type_key << ", because element " << i << " has type " << field;

    return std::nullopt;
  }
  return output;
}

/* \brief Helper function to unpack a relax::Tuple
 *
 * A `relax::Tuple` may be provided to an operator as an in-line
 * expression, as a variable bound to known tuple within the current
 * function, as a function argument, etc.  This utility extracts
 * `PrimExpr` contents only when the concrete tuple expression is
 * available.
 *
 * If the Type cannot contain a tuple of the type specified,
 * this function will throw an exception.  (e.g. Attempting to extract
 * a tuple from a `TensorType`.)
 *
 * \tparam PrimType The subtype of PrimExpr to extract.  For example,
 *     extracting an `ffi::Array<int64_t>`
 *
 * \param expr The `relax::Expr` to inspect
 *
 * \returns An array of the `PrimType`, if it can be extracted.
 *     Otherwise, `std::nullopt`.
 */
template <typename PrimType = PrimExpr,
          typename = std::enable_if_t<std::is_base_of_v<PrimExpr, PrimType>>>
ffi::Optional<ffi::Array<PrimType>> UnpackTupleOfPrimExpr(ffi::Optional<Expr> expr) {
  if (!expr) return std::nullopt;

  const Expr& value = expr.value();
  if (const auto* tuple = value.as<TupleObj>()) {
    ffi::Array<PrimType> output;
    for (size_t i = 0; i < tuple->fields.size(); i++) {
      const Expr& field = tuple->fields[i];
      auto prim_value = field.as<PrimExpr>();
      TVM_FFI_CHECK(prim_value, TypeError)
          << "The expression " << value << " cannot contain a tuple whose elements are "
          << PrimType::ContainerType::_type_key << ", because element " << i << " is " << field;

      PrimExpr prim_expr = prim_value.value();
      TVM_FFI_CHECK(prim_expr.template as<typename PrimType::ContainerType>(), TypeError)
          << "The expression " << value << " cannot contain a tuple whose elements are "
          << PrimType::ContainerType::_type_key << ", because element " << i << " has value "
          << prim_expr;

      output.push_back(prim_expr.template as_or_throw<PrimType>());
    }
    return output;
  }

  return UnpackTupleOfPrimExpr<PrimType>(GetType(value));
}

Type InferTypeStridedSlice(const Call& call, const BlockBuilder& ctx) {
  size_t n_args = call->args.size();
  TVM_FFI_ICHECK(4 <= n_args && n_args <= 5)
      << "Operator " << call->op << " accepts either three arguments (data, axes, begin, end) "
      << " or four arguments (data, axes, begin, end, strides), "
      << "but received " << n_args << " in expression " << call;

  Expr data = call->args[0];
  Expr axes = call->args[1];
  Expr begin = call->args[2];
  Expr end = call->args[3];
  ffi::Optional<Expr> strides = [&]() -> ffi::Optional<Expr> {
    if (n_args > 4) {
      return call->args[4];
    } else {
      return std::nullopt;
    }
  }();

  auto axes_ty = GetType(call->args[1]);
  auto begin_ty = GetType(call->args[2]);
  auto end_ty = GetType(call->args[3]);
  auto strides_ty = [&]() -> ffi::Optional<Type> {
    if (n_args > 4) {
      return GetType(call->args[4]);
    } else {
      return std::nullopt;
    }
  }();

  TVM_FFI_ICHECK(IsBaseOf(relax::TensorType(std::nullopt, kUnknownNDim), GetType(data)))
      << "Operator " << call->op << " requires the first argument to be a tensor.  "
      << "However, in expression " << call << ", the first argument " << data << " has type "
      << GetType(data);

  // TODO(Lunderberg): Implement this check using `IsBaseOf`.  Doing
  // so will require a way to represent a `relax::TupleType` of
  // unknown length, where each element has the same `Type`.
  auto is_base_of_tuple_of_int64 = [&](const Type& ty) -> bool {
    if (ty.as<AnyTypeObj>()) {
      return true;
    }

    const auto* tuple = ty.as<TupleTypeObj>();
    if (!tuple) return false;

    return std::all_of(tuple->fields.begin(), tuple->fields.end(),
                       [](const Type& field) { return IsBaseOf(tvm::PrimType::Int(64), field); });
  };
  auto check_tuple = [&](const char* name, Expr expr) {
    auto ty = GetType(expr);

    TVM_FFI_ICHECK(is_base_of_tuple_of_int64(ty))
        << "Operator " << call->op << " requires the " << name
        << " argument to be a tuple of int64 PrimExpr values.  "
        << "However, in expression " << call << ", the " << name << " argument " << expr
        << " has type " << ty;
  };
  check_tuple("axes", call->args[1]);
  check_tuple("begin", call->args[2]);
  check_tuple("end", call->args[3]);
  if (call->args.size() > 4) {
    check_tuple("strides", call->args[4]);
  }

  const auto* data_ty = data->ty.as<TensorTypeObj>();

  ffi::Optional<PrimType> dtype = std::nullopt;
  ffi::Optional<VDevice> vdevice = std::nullopt;
  int ndim = kUnknownNDim;
  if (data_ty) {
    dtype = data_ty->dtype;
    vdevice = data_ty->vdevice;
    ndim = data_ty->ndim;
  }

  ffi::Optional<Expr> shape = [&]() -> ffi::Optional<Expr> {
    if (!data_ty) return std::nullopt;
    if (!data_ty->shape) return std::nullopt;

    auto opt_axes_tuple = UnpackTupleOfPrimExpr<IntImm>(axes);
    if (!opt_axes_tuple) return std::nullopt;
    auto axes_tuple = opt_axes_tuple.value();

    auto opt_begin_tuple = UnpackTupleOfPrimExpr(begin);
    if (!opt_begin_tuple) return std::nullopt;
    auto begin_tuple = opt_begin_tuple.value();

    TVM_FFI_ICHECK_EQ(axes_tuple.size(), begin_tuple.size())
        << "For operator " << call->op << ", "
        << "the number of axes provided must match the number of 'begin' indices.  "
        << "However, there are " << axes_tuple.size() << " axes specified (" << axes_tuple
        << ") and " << begin_tuple.size() << " 'begin' indices specified (" << begin_tuple << ")";

    auto opt_end_tuple = UnpackTupleOfPrimExpr(end);
    if (!opt_end_tuple) return std::nullopt;
    auto end_tuple = opt_end_tuple.value();

    TVM_FFI_ICHECK_EQ(axes_tuple.size(), end_tuple.size())
        << "For operator " << call->op << ", "
        << "the number of axes provided must match the number of 'end' indices.  "
        << "However, there are " << axes_tuple.size() << " axes specified (" << axes_tuple
        << ") and " << end_tuple.size() << " 'end' indices specified (" << end_tuple << ")";

    ffi::Array<PrimExpr> strides_tuple;
    if (strides.has_value()) {
      auto opt_strides_tuple = UnpackTupleOfPrimExpr(strides);
      if (!opt_strides_tuple) return std::nullopt;

      strides_tuple = opt_strides_tuple.value();
    } else {
      strides_tuple = ffi::Array<PrimExpr>(axes_tuple.size(), IntImm::Int64(1));
    }

    TVM_FFI_ICHECK_EQ(axes_tuple.size(), strides_tuple.size())
        << "For operator " << call->op << ", "
        << "when the optional 'strides' argument is provided, "
        << "the number of axes provided must match the number of strides provided.  "
        << "However, there are " << axes_tuple.size() << " axes specified (" << axes_tuple
        << ") and " << strides_tuple.size() << " strides specified (" << strides_tuple << ")";

    auto opt_data_shape = data_ty->GetShape();

    if (axes_tuple.empty() && !opt_data_shape.has_value()) {
      return data_ty->shape.value();
    } else if (!opt_data_shape.has_value()) {
      return std::nullopt;
    }

    ffi::Array<int64_t> axes_tuple_i64;
    axes_tuple_i64.reserve(axes_tuple.size());
    for (const IntImm& v : axes_tuple) axes_tuple_i64.push_back(static_cast<int64_t>(v->value));
    std::vector<int> axes = NormalizeAxes(call, ctx, data_ty->ndim, axes_tuple_i64);
    auto attrs = call->attrs.as<StridedSliceAttrs>();

    ffi::Array<PrimExpr> output_shape = data_ty->GetShape().value();
    for (size_t i = 0; i < axes.size(); i++) {
      size_t axis = axes[i];
      PrimExpr input_dim = output_shape[axis];
      PrimExpr begin = begin_tuple[i];
      PrimExpr end = end_tuple[i];

      PrimExpr output_dim =
          topi::GetLength(begin, end, strides_tuple[i], input_dim, attrs->assume_inbound);

      sym::Analyzer analyzer = ctx->GetAnalyzer();
      auto saved_context = analyzer->GetConstraintContext();
      if (attrs->assume_inbound) {
        auto context = saved_context->Snapshot();
        context->Append(sym::PredicateConstraint(0 <= begin && begin <= input_dim && 0 <= end &&
                                                 end <= input_dim));
        analyzer->SyncConstraintContext(context);
      }

      output_dim = analyzer->Simplify(output_dim);
      analyzer->SyncConstraintContext(saved_context);

      output_shape.Set(axis, output_dim);
    }
    return ShapeExpr(output_shape);
  }();

  if (shape.has_value()) {
    return TensorType(shape.value(), dtype, vdevice);
  } else {
    return TensorType(dtype, ndim, vdevice);
  }
}

InferLayoutOutput InferLayoutStridedSlice(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  TVM_FFI_ICHECK(NoDesiredLayout(call, desired_layouts));

  const auto* attrs = call->attrs.as<StridedSliceAttrs>();
  TVM_FFI_ICHECK(attrs != nullptr) << "Invalid Call";

  const auto* tensor_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
  TVM_FFI_ICHECK(tensor_ty) << "Invalid Call";
  TVM_FFI_ICHECK(!tensor_ty->IsUnknownNdim())
      << "Layout inference only supports known dimensionality, "
      << "but expression " << call << " has argument " << call->args[0]
      << " of unknown dimensionality.";
  LayoutDecision existing_layout = GetLayoutDecision(var_layout_map, call->args[0]);
  // Can't handle sub indexed layouts.
  if ((existing_layout->layout.has_value() ? existing_layout->layout.value().ndim() : 0) !=
      (existing_layout->layout.has_value() ? existing_layout->layout.value().ndim_primal() : 0)) {
    existing_layout = LayoutDecision(InitialLayout(tensor_ty->ndim));
  }

  auto opt_axes_tuple = UnpackTupleOfPrimExpr<IntImm>(call->args[1]);
  TVM_FFI_ICHECK(opt_axes_tuple) << "Layout inference of " << call->op
                                 << " requires slices to be along static axes.  "
                                 << "However, expression " << call
                                 << " slices along non-static axes " << call->args[1];
  ffi::Array<IntImm> axes_tuple = opt_axes_tuple.value();

  ffi::Array<Expr> new_axes;
  for (const auto& axis : axes_tuple) {
    int new_axis = FindAxis(existing_layout->layout.value(), axis->value.as<int>().value());
    new_axes.push_back(IntImm::Int64(new_axis));
  }

  return InferLayoutOutput({existing_layout}, {existing_layout}, call->attrs,
                           {{IntImm::Int32(1), relax::Tuple(new_axes)}});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.strided_slice")
      .signature(sig::arg("x", "The source tensor to be sliced."), sig::arg("axes", "The axes."),
                 sig::arg("begin", "The start index."), sig::arg("end", "The end index."),
                 sig::var_args("args"), sig::call_attrs<StridedSliceAttrs>())
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeStridedSlice)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutStridedSlice)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.dynamic_strided_slice */
Expr dynamic_strided_slice(Expr x,      //
                           Expr begin,  //
                           Expr end,    //
                           Expr strides) {
  static const Op op = Op::Get("relax.dynamic_strided_slice");
  return Call(Type::Missing(), op,
              {std::move(x), std::move(begin), std::move(end), std::move(strides)}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.dynamic_strided_slice", dynamic_strided_slice);
}

Type InferTypeDynStridedSlice(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  if (RequiresTensorInputNormalization(call)) return Type::Missing();
  const auto* data_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
  const auto* begin_ty = GetTypeAs<TensorTypeObj>(call->args[1]);
  const auto* end_ty = GetTypeAs<TensorTypeObj>(call->args[2]);
  const auto* strides_ty = GetTypeAs<TensorTypeObj>(call->args[3]);

  TVM_FFI_ICHECK(data_ty);
  if (data_ty->IsUnknownNdim()) {
    LOG(WARNING) << "When data rank is unknown, dynamic strided slice assumes begin/end/strides "
                    "tensors are well-formed. It could produce runtime error when this assumption "
                    "turns out to be wrong.";
    return TensorType(data_ty->dtype, kUnknownNDim, data_ty->vdevice);
  }
  if (data_ty->IsUnknownDtype()) {
    LOG(WARNING) << "When data type is unknown, dynamic strided slice assumes to have a valid "
                    "dtype. It could produce runtime error when this assumption "
                    "turns out to be wrong.";
  }

  int n_axis = data_ty->ndim;
  auto diag_def = [&](const TensorTypeObj* ty, ffi::String name) {
    TVM_FFI_ICHECK(ty) << "Dynamic strided slice requires the input " << name
                       << " to be have the type. Please try normalizing the inputs.";
    TVM_FFI_ICHECK_EQ(ty->ndim, 1)
        << "Dynamic strided slice requires " << name << " to be 1d tensor (list of values).";
    const auto* shape = ty->shape.as<ShapeExprObj>();
    TVM_FFI_ICHECK(shape) << "Dynamic strided slice requires the input " << name
                          << " to have well-defined shape.";
    // NOTE(tvm-team): This strong restriction seems necessary for now until we have a generic
    // solution in converting 1d Tensor with unknown num_elem to ffi::Array<PrimExpr>.
    const auto* num_elem = shape->values[0].as<IntImmObj>();
    TVM_FFI_ICHECK(num_elem) << "Dynamic strided slice requires the input " << name
                             << " to have a known integer shape value.";
    TVM_FFI_ICHECK_EQ(num_elem->value, n_axis)
        << "Dynamic strided slice requires the number of indices in " << name
        << " to equal the number of axes.";
    if (ty->IsUnknownDtype()) {
      LOG(WARNING) << "Dynamic strided slice assumes " << name
                   << " to be int64 when it is not specified.";
    } else {
      TVM_FFI_ICHECK(ty->dtype == PrimType::Int(64))
          << "Dynamic strided_slice expects the input " << name
          << "values to be all int64. However, " << name << " has dtype " << ty->dtype << ".";
    }
  };
  diag_def(begin_ty, "begin");
  diag_def(end_ty, "end");
  diag_def(strides_ty, "strides");

  // The output shape will depend on the runtime value in begin/end/strides tensors.
  // TODO(tvm-team): Currently, it is unable to express partially-static shape. Revisit when
  // PrimExpr lands.
  return TensorType(data_ty->dtype, n_axis, data_ty->vdevice);
}

InferLayoutOutput InferLayoutDynStridedSlice(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  TVM_FFI_ICHECK(NoDesiredLayout(call, desired_layouts));

  const auto* tensor_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
  TVM_FFI_ICHECK(tensor_ty) << "Invalid Call";
  TVM_FFI_ICHECK(!tensor_ty->IsUnknownNdim())
      << "Layout inference only supports known dimensionality, "
      << "but expression " << call << " has argument " << call->args[0]
      << " of unknown dimensionality.";
  int ndim = tensor_ty->ndim;
  // Since begin/end/strides are dynamic tensors, we cannot transform
  // them at compile time. Fall back to the initial layout.
  LayoutDecision initial = LayoutDecision(InitialLayout(ndim));
  return InferLayoutOutput({initial}, {initial}, std::nullopt);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.dynamic_strided_slice")
      .signature(sig::arg("x", "The source tensor to be sliced."),
                 sig::arg("begin", "The indices to begin with in the slicing."),
                 sig::arg("end", "Indices indicating end of the slice."),
                 sig::arg("strides", "The stride values."))
      .set_native_fn_attr<&InferTypeDynStridedSlice>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutDynStridedSlice)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_attr(tvm::relax::op_attr::kDataDependent, true);
}

}  // namespace relax
}  // namespace tvm
