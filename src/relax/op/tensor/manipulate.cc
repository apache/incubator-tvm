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
 * \file manipulate.cc
 * \brief Manipulation operators.
 */

#include "manipulate.h"

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/relax/op_attr.h>
#include <tvm/runtime/logging.h>

#include <algorithm>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "tvm/ffi/dtype.h"

namespace tvm {
namespace relax {

void ConcatAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<ConcatAttrs>().def_ro("axis", &ConcatAttrs::axis,
                                        "The axis at which the input arrays are concatenated."
                                        "Should lie in range `[-ndim, ndim)`.",
                                        refl::DefaultValue(ffi::Optional<int64_t>{0}));
}

void ExpandDimsAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<ExpandDimsAttrs>().def_ro(
      "axis", &ExpandDimsAttrs::axis,
      "The axes at which the input array are expanded. "
      "All values are required to lie in range `[-data.ndim - 1, data.ndim]`, "
      "with the convention of negative indexing.");
}

void LayoutTransformAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<LayoutTransformAttrs>()
      .def_ro("index_map", &LayoutTransformAttrs::index_map, "The layout transformation to apply.")
      .def_ro(
          "pad_value", &LayoutTransformAttrs::pad_value,
          "The specific value to be used to pad if the layout transform would result in implicit "
          "padding. If not specified, the compiler is free to choose any value.",
          refl::DefaultValue(ffi::Optional<PrimExpr>{}));
}

void PermuteDimsAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<PermuteDimsAttrs>().def_ro(
      "axes", &PermuteDimsAttrs::axes, "The target axes order, reverse order if not specified.",
      refl::DefaultValue(ffi::Optional<ffi::Array<int64_t>>{}));
}

void SplitAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<SplitAttrs>()
      .def_ro("indices_or_sections", &SplitAttrs::indices_or_sections,
              "The input array of indices or the number of split sections.")
      .def_ro("axis", &SplitAttrs::axis, "The axis to be splitted", refl::DefaultValue(0));
}

void SqueezeAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<SqueezeAttrs>().def_ro("axis", &SqueezeAttrs::axis,
                                         "The axis to squeeze in the input tensor."
                                         "If `axis = None`, all axis of dimension 1 get squeezed;"
                                         "Else, the dimension in axes get squeezed."
                                         "It is an error if an axis does not has dimension 1.",
                                         refl::DefaultValue(ffi::Optional<ffi::Array<int64_t>>{}));
}

void StackAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<StackAttrs>().def_ro("axis", &StackAttrs::axis,
                                       "The axis along which to stack the input tensors. "
                                       "The axis will be inserted at this position in the output, "
                                       "so it must be in range [-ndim-1, ndim] where ndim is the "
                                       "number of dimensions of the input tensors.",
                                       refl::DefaultValue(ffi::Optional<int64_t>{0}));
}

void RepeatAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<RepeatAttrs>()
      .def_ro("repeats", &RepeatAttrs::repeats, "The number of repetitions.")
      .def_ro("axis", &RepeatAttrs::axis,
              "The axis along which to repeat values. The negative numbers are interpreted "
              "counting from the backward. By default, use the flattened input array, and "
              "return a flat output array.",
              refl::DefaultValue(ffi::Optional<int64_t>{}));
}

void TileAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TileAttrs>().def_ro("repeats", &TileAttrs::repeats,
                                      "The number of repetitions of data along each axis.");
}

void FlipAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<FlipAttrs>().def_ro("axis", &FlipAttrs::axis,
                                      "The axis along which to flip over.");
}

void ReverseSequenceAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<ReverseSequenceAttrs>()
      .def_ro("seq_axis", &ReverseSequenceAttrs::seq_axis,
              "The axis along which to reverse variable length slices.", refl::DefaultValue(1))
      .def_ro("batch_axis", &ReverseSequenceAttrs::batch_axis, "The axis that indexes the batch.",
              refl::DefaultValue(0));
}

void GatherElementsAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<GatherElementsAttrs>().def_ro(
      "axis", &GatherElementsAttrs::axis, "The axis along which to index.", refl::DefaultValue(0));
}

void GatherNDAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<GatherNDAttrs>().def_ro("batch_dims", &GatherNDAttrs::batch_dims,
                                          "The number of batch dims.", refl::DefaultValue(0));
}

void IndexPutAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<IndexPutAttrs>().def_ro("accumulate", &IndexPutAttrs::accumulate,
                                          "Whether to accumulate (add) values rather than replace. "
                                          "If true, performs tensor[indices] += values, "
                                          "otherwise performs tensor[indices] = values.",
                                          refl::DefaultValue(false));
}

void MeshgridAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<MeshgridAttrs>().def_ro("indexing", &MeshgridAttrs::indexing,
                                          "Specifies how the grid dimensions are ordered.",
                                          refl::DefaultValue(ffi::String("ij")));
}

void ScatterElementsAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<ScatterElementsAttrs>()
      .def_ro("axis", &ScatterElementsAttrs::axis, "The axis over which to select values.",
              refl::DefaultValue(0))
      .def_ro("reduction", &ScatterElementsAttrs::reduction,
              "Reduction mode of the scatter elements, "
              "either \"update\", \"add\", \"mul\", \"mean\", \"min\" or \"max\".",
              refl::DefaultValue("update"));
}

void ScatterNDAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<ScatterNDAttrs>().def_ro(
      "reduction", &ScatterNDAttrs::reduction,
      "Accumulation mode of the ScatterND, "
      "either \"update\", \"add\", \"mul\", \"min\" or \"max\".",
      refl::DefaultValue("update"));
}

void SliceScatterAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<SliceScatterAttrs>().def_ro("axis", &SliceScatterAttrs::axis,
                                              "the dimension to insert the slice into ",
                                              refl::DefaultValue(0));
}

void OneHotAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<OneHotAttrs>()
      .def_ro("depth", &OneHotAttrs::depth, "Depth of the one hot dimension.")
      .def_ro("axis", &OneHotAttrs::axis, "Axis to fill.", refl::DefaultValue(-1));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ConcatAttrs::RegisterReflection();
  ExpandDimsAttrs::RegisterReflection();
  LayoutTransformAttrs::RegisterReflection();
  PermuteDimsAttrs::RegisterReflection();
  SplitAttrs::RegisterReflection();
  SqueezeAttrs::RegisterReflection();
  StackAttrs::RegisterReflection();
  RepeatAttrs::RegisterReflection();
  TileAttrs::RegisterReflection();
  FlipAttrs::RegisterReflection();
  ReverseSequenceAttrs::RegisterReflection();
  GatherElementsAttrs::RegisterReflection();
  GatherNDAttrs::RegisterReflection();
  IndexPutAttrs::RegisterReflection();
  MeshgridAttrs::RegisterReflection();
  ScatterElementsAttrs::RegisterReflection();
  ScatterNDAttrs::RegisterReflection();
  SliceScatterAttrs::RegisterReflection();
  OneHotAttrs::RegisterReflection();
}

/* relax.broadcast_to */
Expr broadcast_to(Expr x, Expr shape) {
  static const Op op = Op::Get("relax.broadcast_to");
  return Call(Type::Missing(), op, {std::move(x), std::move(shape)}, std::nullopt, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.broadcast_to", broadcast_to);
}

Type InferTypeBroadcastTo(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 2) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "broadcast_to should take 2 arguments.";
  }
  const auto* data_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
  const auto* tgt_shape_ty = GetTypeAs<ShapeTypeObj>(call->args[1]);
  if (data_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "broadcast_to requires the input data to be Tensor. However, the given one is "
        << call->args[0]->ty->GetTypeKey();
  }
  if (tgt_shape_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "broadcast_to requires the input new shape to be Shape. However, the given one is "
        << call->args[1]->ty->GetTypeKey();
  }

  if (!data_ty->IsUnknownNdim() && !tgt_shape_ty->IsUnknownNdim() &&
      tgt_shape_ty->ndim < data_ty->ndim) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "broadcast_to expects the input shape to have the number of ndim at least "
           "as the input tensor's. However, the given tensor has ndim "
        << data_ty->ndim << " while the target shape has ndim " << tgt_shape_ty->ndim;
  }

  // Trust the input target shape when there is no possibility to do any compile-time check.
  if (!data_ty->shape.has_value()) {
    return TensorType(/*shape=*/call->args[1], data_ty->dtype, data_ty->vdevice);
  }
  ShapeType shape_ty = data_ty->shape.value()->ty.as_or_throw<ShapeType>();
  if (!shape_ty->values.has_value() || !tgt_shape_ty->values.has_value()) {
    return TensorType(/*shape=*/call->args[1], data_ty->dtype, data_ty->vdevice);
  }

  sym::Analyzer analyzer = ctx->GetAnalyzer();
  ffi::Array<PrimExpr> old_shape_value = shape_ty->values.value();
  ffi::Array<PrimExpr> tgt_shape_value = tgt_shape_ty->values.value();
  int old_ndim = old_shape_value.size();
  int tgt_ndim = tgt_shape_value.size();
  for (int i = 0; i < old_ndim; ++i) {
    PrimExpr old_len = old_shape_value[old_ndim - i - 1];
    PrimExpr tgt_len = tgt_shape_value[tgt_ndim - i - 1];
    const auto* old_len_int = old_len.as<IntImmObj>();
    if (old_len_int != nullptr && old_len_int->value == 1) {
      continue;
    } else if (analyzer->CanProve(old_len != tgt_len)) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "broadcast_to expects the input tensor shape is broadcastable to the target shape. "
             "The target shape at dim "
          << tgt_ndim - i - 1 << " is " << tgt_len << " while the input tensor shape at dim "
          << old_ndim - i - 1 << " is " << old_len << ", which are not equal.";
    }
    // Todo(relax-team): revisit here for better check on if the tensor length
    // is consistent with the length in the given shape.
  }
  return TensorType(/*shape=*/call->args[1], data_ty->dtype, data_ty->vdevice);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.broadcast_to")
      .signature(sig::arg("x", "The input tensor."), sig::arg("shape", "The target shape."))
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeBroadcastTo)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.concat */

Expr concat(Expr tensors, ffi::Optional<int64_t> axis) {
  ffi::ObjectPtr<ConcatAttrs> attrs = ffi::make_object<ConcatAttrs>();
  attrs->axis = std::move(axis);

  static const Op op = Op::Get("relax.concat");
  return Call(Type::Missing(), op, {std::move(tensors)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.concat", concat);
}

ffi::Optional<ffi::Array<PrimExpr>> CheckConcatOutputShape(
    const Call& call, const BlockBuilder& ctx,
    const std::vector<ffi::Array<PrimExpr>>& shape_values, int axis) {
  bool shape_unknown = false;
  sym::Analyzer analyzer = ctx->GetAnalyzer();
  PrimExpr concat_sum = [&]() {
    // For the specified axis, we compute the sum of shape value over each tensor.

    // Special case, if all concatenated values have the same shape
    ffi::StructuralEqual structural_equal;
    PrimExpr first_concat_dim = shape_values[0][axis];
    bool all_same = std::all_of(shape_values.begin(), shape_values.end(), [&](const auto& a) {
      return structural_equal(a[axis], first_concat_dim);
    });
    if (all_same) {
      return first_concat_dim * IntImm::Int64(shape_values.size());
    }

    // General case, add up the dimensions along the specified axis.
    PrimExpr concat_sum = IntImm::Int64(0);
    for (ffi::Array<PrimExpr> shape_value : shape_values) {
      concat_sum += shape_value[axis];
    }
    return concat_sum;
  }();

  // For other axes, we check the equality of all tensors' shape values, to ensure safety.
  for (int d = 0; d < static_cast<int>(shape_values[0].size()); ++d) {
    if (d == axis) {
      continue;
    }
    for (int i = 1; i < static_cast<int>(shape_values.size()); ++i) {
      if (analyzer->CanProve(shape_values[i][d] != shape_values[0][d])) {
        TVM_FFI_VISIT_THROW(ValueError, call)
            << "Concat expects the input tensors to have the same shape on every "
               "dimension except the one indicated by the input axis. However, the "
               "input contains tensors whose shapes on dimension "
            << d << " is " << shape_values[0][d] << " and " << shape_values[i][d];
      } else if (!analyzer->CanProveEqual(shape_values[i][d], shape_values[0][d])) {
        shape_unknown = true;
      }
    }
  }

  if (shape_unknown) {
    return std::nullopt;
  }
  ffi::Array<PrimExpr> output_shape = shape_values[0];
  output_shape.Set(axis, concat_sum);
  return output_shape;
}

Type InferTypeConcat(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 1) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "Concat op should have 1 argument";
  }
  ffi::Array<TensorType> tensor_ty = GetTensorTypeFromTuple(call, ctx, call->args[0]);
  if (tensor_ty.empty()) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Concat op expects at least one tensor in the input Tuple. However, the "
           "given input Tuple is empty.";
  }

  const auto* attrs = call->attrs.as<ConcatAttrs>();
  int output_ndim = attrs->axis.has_value() ? kUnknownNDim : 1;
  ffi::Optional<PrimType> output_dtype = std::nullopt;
  ffi::Optional<VDevice> vdev = std::nullopt;
  bool shape_unknown = false;
  bool is_void_dtype = false;
  bool vdevice_unknown = false;
  std::vector<ffi::Array<PrimExpr>> shape_values;
  shape_values.reserve(tensor_ty.size());

  for (TensorType ty : tensor_ty) {
    // Update the output dtype.
    if (ty->IsUnknownDtype()) {
      is_void_dtype = true;
    } else if (!output_dtype.has_value()) {
      output_dtype = ty->dtype;
    } else if (ty->dtype != output_dtype) {
      TVM_FFI_VISIT_THROW(TypeError, call)
          << "Concat expects all input tensors to have the same dtype. However, the "
             "input contains tensors with dtype "
          << output_dtype << " and " << ty->dtype;
    }

    // Update the output ndim.
    // Todo(relax-team): revisit here for better check on if the input tensor has
    // ndim 1 when the input axis is undefined.
    if (output_ndim == kUnknownNDim) {
      output_ndim = ty->ndim;
    } else if (ty->ndim != kUnknownNDim && ty->ndim != output_ndim) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "Concat expects all input tensors to have same ndim. However, the "
             "input contains tensors with ndim "
          << output_ndim << " and " << ty->ndim;
    }

    // Update the virtual device.
    if (!vdevice_unknown) {
      if (ty->vdevice.has_value()) {
        if (!vdev.has_value()) {
          vdev = ty->vdevice.value();
        } else if (ty->vdevice.value()->target.has_value()) {
          // mismatch
          if (ty->vdevice.value() != vdev.value()) {
            vdevice_unknown = true;
          }
        }
      }
    }

    // Update the shape values for best effort check.
    const auto* shape_expr = ty->shape.as<ShapeExprObj>();
    if (shape_expr != nullptr) {
      shape_values.push_back(shape_expr->values);
      continue;
    }
    shape_unknown = true;

    if (!ty->shape.has_value()) {
      continue;
    }
    // Keep the shape value for equality check.
    ShapeType shape_ty = ty->shape.value()->ty.as_or_throw<ShapeType>();
    if (shape_ty->values.has_value()) {
      shape_values.push_back(shape_ty->values.value());
    }
  }

  if (is_void_dtype) {
    output_dtype = std::nullopt;
  }
  if (vdevice_unknown) {
    vdev = std::nullopt;
  }

  if (output_ndim == kUnknownNDim) {
    return tensor_ty.size() == 1 ? tensor_ty[0] : TensorType(output_dtype, output_ndim, vdev);
  }

  int axis =
      attrs->axis.has_value() ? NormalizeAxis(call, ctx, output_ndim, attrs->axis.value()) : 0;
  // If there is only one input tensor, no action is needed.
  if (tensor_ty.size() == 1) {
    return tensor_ty[0];
  }
  if (shape_values.empty()) {
    if (!vdevice_unknown) {
      return TensorType(output_dtype, output_ndim, vdev);
    }
    return TensorType(output_dtype, output_ndim);
  }

  // As long as the there is known shape value, we will do the best effort check to ensure safety.
  ffi::Optional<ffi::Array<PrimExpr>> output_shape =
      CheckConcatOutputShape(call, ctx, shape_values, axis);

  if (shape_unknown || !output_shape.has_value()) {
    if (!vdevice_unknown) {
      return TensorType(output_dtype, output_ndim, vdev);
    }
    return TensorType(output_dtype, output_ndim);
  } else {
    if (!vdevice_unknown) {
      return TensorType(ShapeExpr(output_shape.value()), output_dtype, vdev);
    }
    return TensorType(ShapeExpr(output_shape.value()), output_dtype);
  }
}

InferLayoutOutput InferLayoutConcat(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  TVM_FFI_ICHECK(NoDesiredLayout(call, desired_layouts));

  const auto* attrs = call->attrs.as<ConcatAttrs>();
  TVM_FFI_ICHECK(attrs != nullptr) << "Invalid Call";

  NLayout nlayout = GetNLayout(var_layout_map, call->args[0]);
  TVM_FFI_ICHECK(nlayout.IsNested());
  TVM_FFI_ICHECK(nlayout.NestedArray()[0].IsLeaf());

  int n_tensor = nlayout.NestedArray().size();
  LayoutDecision layout = nlayout.NestedArray()[0].LeafValue();

  // We may expect mix of sub indexed and regular layouts here
  // Pick the first sub indexed layout and try to prove it for all tensors
  // On any failre select first occuring regular layout for all
  auto nlayout_array = nlayout.NestedArray();
  for (auto n_layout : nlayout_array) {
    TVM_FFI_ICHECK(n_layout.IsLeaf());
    LayoutDecision in_layout = n_layout.LeafValue();
    if ((in_layout->layout.has_value() ? in_layout->layout.value().ndim() : 0) !=
        (in_layout->layout.has_value() ? in_layout->layout.value().ndim_primal() : 0)) {
      const auto* tuple_ty = GetTypeAs<TupleTypeObj>(call->args[0]);
      TVM_FFI_ICHECK(tuple_ty != nullptr)
          << " expects the input to be a Tuple of Tensors. However, the given input is "
          << call->args[0]->ty->GetTypeKey();
      for (size_t i = 0; i < tuple_ty->fields.size(); ++i) {
        Type field_ty = tuple_ty->fields[i];
        const auto* field_tensor_ty = field_ty.as<TensorTypeObj>();
        TVM_FFI_ICHECK(field_tensor_ty != nullptr)
            << call->op
            << " expects the input to be a Tuple of Tensors. However, the given input is "
            << call->args[0]->ty;
        auto t_ty = ffi::GetRef<TensorType>(field_tensor_ty);
        ffi::Optional<ShapeExpr> t_shape = ffi::GetRef<ShapeExpr>(t_ty->shape.as<ShapeExprObj>());
        LayoutDecision curr_layout = nlayout_array[i].LeafValue();
        if (!CanProveLayoutTransform(curr_layout->layout, in_layout->layout,
                                     t_shape.value()->values)) {
          // Some tensor unhappy with sub indexed layout, lets pick first regular layout
          for (auto pick_layout : nlayout_array) {
            if ((pick_layout.LeafValue()->layout.has_value()
                     ? pick_layout.LeafValue()->layout.value().ndim()
                     : 0) == (pick_layout.LeafValue()->layout.has_value()
                                  ? pick_layout.LeafValue()->layout.value().ndim_primal()
                                  : 0)) {
              in_layout = pick_layout.LeafValue();
              break;
            }
          }
          break;
        }
      }
      layout = in_layout;
      break;
    }
  }

  ffi::Array<NLayout> input_layouts, output_layouts;
  for (int i = 0; i < n_tensor; ++i) {
    input_layouts.push_back(layout);
  }
  output_layouts.push_back(layout);
  ffi::ObjectPtr<ConcatAttrs> new_attrs = ffi::make_object<ConcatAttrs>(*attrs);
  new_attrs->axis = FindAxis(layout->layout.value(), attrs->axis.value_or(0));
  return InferLayoutOutput({NLayout(input_layouts)}, output_layouts, Attrs(new_attrs));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.concat")
      .signature(sig::arg("tensors", "The input list of tensors."), sig::call_attrs<ConcatAttrs>())
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeConcat)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutConcat)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.expand_dims */

Expr expand_dims(Expr x, ffi::Array<int64_t> axis) {
  ffi::ObjectPtr<ExpandDimsAttrs> attrs = ffi::make_object<ExpandDimsAttrs>();
  attrs->axis = std::move(axis);

  static const Op op = Op::Get("relax.expand_dims");
  return Call(Type::Missing(), op, {std::move(x)}, Attrs{attrs}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.expand_dims", expand_dims);
}

Type InferTypeExpandDims(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  if (RequiresTensorInputNormalization(call)) return Type::Missing();
  TensorType data_ty = GetUnaryInputTensorType(call);
  const auto* attrs = call->attrs.as<ExpandDimsAttrs>();
  if (attrs->axis.empty()) {
    return data_ty;
  }

  if (data_ty->IsUnknownNdim()) {
    return TensorType(data_ty->dtype, kUnknownNDim, data_ty->vdevice);
  }

  int n_new_dim = attrs->axis.size();
  int output_ndim = data_ty->ndim + n_new_dim;
  std::vector<int> axes = NormalizeAxes(call, output_ndim, attrs->axis);

  const auto* data_shape = data_ty->shape.as<ShapeExprObj>();
  if (data_shape == nullptr) {
    return TensorType(data_ty->dtype, output_ndim, data_ty->vdevice);
  }

  std::vector<PrimExpr> output_shape;
  output_shape.reserve(output_ndim);
  int i_data_shape = 0;
  for (int i = 0; i < output_ndim; ++i) {
    if (std::find(axes.begin(), axes.end(), i) != axes.end()) {
      output_shape.push_back(IntImm::Int64(1));
    } else {
      TVM_FFI_ICHECK_LT(i_data_shape, data_ty->ndim);
      output_shape.push_back(data_shape->values[i_data_shape++]);
    }
  }
  TVM_FFI_ICHECK_EQ(i_data_shape, data_ty->ndim);
  return TensorType(ShapeExpr(output_shape), data_ty->dtype, data_ty->vdevice);
}

InferLayoutOutput InferLayoutExpandDims(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  TVM_FFI_ICHECK(NoDesiredLayout(call, desired_layouts));
  const auto* attrs = call->attrs.as<ExpandDimsAttrs>();
  TVM_FFI_ICHECK(attrs != nullptr) << "Invalid Call";
  const auto* tensor_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
  TVM_FFI_ICHECK(tensor_ty != nullptr) << "Invalid Call";
  TVM_FFI_ICHECK(!tensor_ty->IsUnknownNdim()) << "Only support static ndim for now";

  LayoutDecision existing_layout = GetLayoutDecision(var_layout_map, call->args[0]);
  int ndim = tensor_ty->ndim;
  // Can't handle sub indexed layouts.
  if ((existing_layout->layout.has_value() ? existing_layout->layout.value().ndim() : 0) !=
      (existing_layout->layout.has_value() ? existing_layout->layout.value().ndim_primal() : 0)) {
    existing_layout = LayoutDecision(InitialLayout(ndim));
  }
  int n_new_dim = attrs->axis.size();
  int output_ndim = ndim + n_new_dim;
  std::vector<bool> is_new_dim(output_ndim, false);
  for (const auto& axis : attrs->axis) {
    is_new_dim[(axis + output_ndim) % output_ndim] = true;
  }
  std::string new_layout;
  for (int i = 0; i < output_ndim; ++i) {
    if (!is_new_dim[i]) {
      new_layout.push_back('A' + i);
    }
  }
  new_layout = TransposeStrLike(new_layout, InitialLayout(ndim), existing_layout->layout);
  std::string output_layout;
  for (int i = 0, j = 0; i < output_ndim; ++i) {
    if (is_new_dim[i]) {
      output_layout.push_back('A' + i);
    } else {
      output_layout.push_back(new_layout.at(j++));
    }
  }
  return InferLayoutOutput({existing_layout}, {LayoutDecision(SLayout::Create(output_layout))},
                           call->attrs);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.expand_dims")
      .signature(sig::arg("x", "The input tensor."), sig::call_attrs<ExpandDimsAttrs>())
      .set_native_fn_attr<&InferTypeExpandDims>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutExpandDims)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

// Helper function for flatten and reshape.
PrimExpr ComputeShapeProduct(const ffi::Array<PrimExpr>& shape_values) {
  PrimExpr shape_prod = IntImm::Int64(1);
  for (PrimExpr value : shape_values) {
    shape_prod *= value;
  }
  return shape_prod;
}

/* relax.flatten */
Expr flatten(Expr x) {
  static const Op op = Op::Get("relax.flatten");
  return Call(Type::Missing(), op, {std::move(x)}, {}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.flatten", flatten);
}

Type InferTypeFlatten(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  TensorType data_ty = GetUnaryInputTensorType(call);
  if (data_ty->IsUnknownNdim()) {
    return TensorType(data_ty->dtype, /*ndim=*/1, data_ty->vdevice);
  } else if (data_ty->ndim == 0) {
    return TensorType(ShapeExpr({1}), data_ty->dtype, data_ty->vdevice);
  } else if (data_ty->ndim == 1) {
    return data_ty;
  }

  const auto* data_shape = data_ty->shape.as<ShapeExprObj>();
  if (data_shape == nullptr) {
    return TensorType(data_ty->dtype, /*ndim=*/1, data_ty->vdevice);
  }
  PrimExpr shape_prod = ComputeShapeProduct(data_shape->values);
  return TensorType(ShapeExpr({std::move(shape_prod)}), data_ty->dtype, data_ty->vdevice);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.flatten")
      .signature(sig::arg("x", "The input tensor."))
      .set_native_fn_attr<&InferTypeFlatten>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.index_tensor */

Expr index_tensor(Expr first, Expr tensors) {
  static const Op op = Op::Get("relax.index_tensor");
  return Call(Type::Missing(), op, {std::move(first), std::move(tensors)}, std::nullopt, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.index_tensor", index_tensor);
}

Type InferTypeIndexTensor(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 2) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "Index.Tensor op should have 2 arguments";
  }

  TensorType data_ty = GetInputTensorType(call, 0, ctx);
  ffi::Array<TensorType> indices_ty = GetTensorTypeFromTuple(call, ctx, call->args[1]);

  if (indices_ty.empty()) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "index_tensor expects a non‑empty tuple of index tensors";
  }

  ffi::Optional<PrimType> output_dtype = data_ty->dtype;
  int n_indices = static_cast<int>(indices_ty.size());
  ffi::Optional<VDevice> vdev = data_ty->vdevice;

  // Indices must be integers
  for (int i = 0; i < n_indices; ++i) {
    const auto& s = indices_ty[i];
    // Indexing only requires integer element kind; vector lanes do not affect shape inference.
    if (!s->IsUnknownDtype() && s->dtype.value().code() != DLDataTypeCode::kDLInt) {
      TVM_FFI_VISIT_THROW(TypeError, call)
          << "index_tensor requires every index tensor to have an integer dtype; "
          << "index " << i << " has dtype " << s->dtype;
    }
  }

  // Count of indices must be less than or equal to data.ndim
  if (!data_ty->IsUnknownNdim() && n_indices > data_ty->ndim) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "index_tensor received " << n_indices << " index tensors, but data has only "
        << data_ty->ndim << " dimensions";
  }

  sym::Analyzer analyzer = ctx->GetAnalyzer();
  bool all_index_have_shape_value = true;
  std::vector<ffi::Array<PrimExpr>> index_shapes;
  int max_index_ndim = 0;

  for (const auto& s : indices_ty) {
    const auto* shp = s->shape.as<ShapeExprObj>();
    if (!shp) {
      all_index_have_shape_value = false;
    } else {
      index_shapes.push_back(shp->values);
      max_index_ndim = std::max(max_index_ndim, static_cast<int>(shp->values.size()));
    }
    if (!s->IsUnknownNdim()) {
      max_index_ndim = std::max(max_index_ndim, s->ndim);
    }
  }

  ffi::Optional<ffi::Array<PrimExpr>> broadcast_shape;
  bool shape_unknown = !all_index_have_shape_value;

  if (all_index_have_shape_value) {
    // initialise broadcast result with 1's
    ffi::Array<PrimExpr> out_shape;
    for (int i = 0; i < max_index_ndim; ++i) {
      out_shape.push_back(IntImm::Int64(1));
    }

    for (const auto& ishape : index_shapes) {
      int cur_ndim = ishape.size();
      for (int axis = 0; axis < max_index_ndim; ++axis) {
        int lhs_axis = max_index_ndim - 1 - axis;  // aligned from right
        int rhs_axis = cur_ndim - 1 - axis;
        if (rhs_axis < 0) break;  // shorter rank – done

        PrimExpr lhs_dim = out_shape[lhs_axis];
        PrimExpr rhs_dim = ishape[rhs_axis];

        const auto* lhs_int = lhs_dim.as<IntImmObj>();
        const auto* rhs_int = rhs_dim.as<IntImmObj>();

        // Case 1: current broadcast slot is 1 -> always replace
        if (lhs_int && lhs_int->value == 1) {
          out_shape.Set(lhs_axis, rhs_dim);
          continue;
        }
        // Case 2: rhs is 1 -> keep lhs_dim unchanged
        if (rhs_int && rhs_int->value == 1) {
          continue;
        }
        // Both are non‑one constants: must equal
        if (lhs_int && rhs_int && lhs_int->value != rhs_int->value) {
          TVM_FFI_VISIT_THROW(ValueError, call)
              << "index_tensor: cannot broadcast index shapes. Mismatch at axis " << lhs_axis
              << ": " << lhs_dim << " vs " << rhs_dim;
        }
        // Give up if not provablt equal
        if (!analyzer->CanProveEqual(lhs_dim, rhs_dim)) {
          shape_unknown = true;
          break;
        }
      }
      if (shape_unknown) break;
    }

    if (!shape_unknown) broadcast_shape = out_shape;
  }

  // Count of dimensions in output
  int out_ndim = kUnknownNDim;
  if (!data_ty->IsUnknownNdim()) {
    int tail_ndim = data_ty->ndim - n_indices;
    if (broadcast_shape.has_value()) {
      out_ndim = static_cast<int>(broadcast_shape.value().size()) + tail_ndim;
    } else if (!shape_unknown) {
      out_ndim = max_index_ndim + tail_ndim;
    }
  }

  // Derive output shape
  if (broadcast_shape.has_value()) {
    const auto* data_shape_expr = data_ty->shape.as<ShapeExprObj>();
    if (data_shape_expr) {
      ffi::Array<PrimExpr> result_shape = broadcast_shape.value();
      for (int i = n_indices; i < data_ty->ndim; ++i) {
        result_shape.push_back(data_shape_expr->values[i]);
      }
      return TensorType(ShapeExpr(result_shape), output_dtype, vdev);
    }
  }

  // Unknown output shape
  return TensorType(output_dtype, out_ndim, vdev);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.index_tensor")
      .signature(sig::arg("data", "The input data."),
                 sig::arg("indices", "The indices used to index."))
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeIndexTensor)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.layout_transform */

Expr layout_transform(Expr x, tirx::IndexMap index_map, ffi::Optional<PrimExpr> pad_value) {
  ffi::ObjectPtr<LayoutTransformAttrs> attrs =
      ffi::make_object<LayoutTransformAttrs>(std::move(index_map));
  attrs->pad_value = std::move(pad_value);

  static const Op op = Op::Get("relax.layout_transform");
  return Call(Type::Missing(), op, {std::move(x)}, Attrs{attrs}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.layout_transform", layout_transform);
}

Type InferTypeLayoutTransform(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  TensorType data_ty = GetUnaryInputTensorType(call);
  const auto* attrs = call->attrs.as<LayoutTransformAttrs>();
  tirx::IndexMap index_map = attrs->index_map;
  ffi::Optional<PrimExpr> optional_pad_value = attrs->pad_value;

  // Check pad_value has same dtype as input.
  if (optional_pad_value.has_value()) {
    PrimExpr padded_value = optional_pad_value.value();
    PrimType padded_dtype = padded_value.ty();
    if (!data_ty->dtype.has_value() || padded_dtype != data_ty->dtype.value()) {
      TVM_FFI_VISIT_THROW(TypeError, call)
          << "layout_transform pad_value dtype (" << padded_dtype << ") and input dtype ("
          << data_ty->dtype << ") must be the same";
    }
  }

  if (data_ty->IsUnknownNdim()) {
    // Todo(relax-team): revisit here for better check on if the input tensor has desired ndim.
    return TensorType(data_ty->dtype, /*ndim=*/index_map->final_indices.size(), data_ty->vdevice);
  }

  // If rank is known, check that it is compatible with the index_map, i.e., #dims match.
  if (index_map->initial_indices.size() != static_cast<size_t>(data_ty->ndim)) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "number of dimensions in input must match the number of source dimensions "
           "in index map, but got "
        << data_ty->ndim << " != " << index_map->initial_indices.size();
  }

  if (!data_ty->shape.has_value()) {
    return TensorType(data_ty->dtype, /*ndim=*/index_map->final_indices.size(), data_ty->vdevice);
  }

  ShapeType shape_ty = data_ty->shape.value()->ty.as_or_throw<ShapeType>();
  if (!shape_ty->values.has_value()) {
    return TensorType(data_ty->dtype, /*ndim=*/index_map->final_indices.size(), data_ty->vdevice);
  }

  sym::Analyzer analyzer;
  ffi::Array<PrimExpr> output_shape = index_map->MapShape(shape_ty->values.value(), analyzer);
  return TensorType(ShapeExpr(output_shape), data_ty->dtype, data_ty->vdevice);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.layout_transform")
      .signature(sig::arg("x", "The input tensor."), sig::call_attrs<LayoutTransformAttrs>())
      .set_native_fn_attr<&InferTypeLayoutTransform>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.permute_dims */

Expr permute_dims(Expr x, ffi::Optional<ffi::Array<int64_t>> axes) {
  ffi::ObjectPtr<PermuteDimsAttrs> attrs = ffi::make_object<PermuteDimsAttrs>();
  attrs->axes = std::move(axes);

  static const Op op = Op::Get("relax.permute_dims");
  return Call(Type::Missing(), op, {std::move(x)}, Attrs{attrs}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.permute_dims", permute_dims);
}

bool IsIdentityPermutation(const std::vector<int>& permutation) {
  for (int i = 0; i < static_cast<int>(permutation.size()); ++i) {
    if (permutation[i] != i) {
      return false;
    }
  }
  return true;
}

Type InferTypePermuteDims(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  if (RequiresTensorInputNormalization(call)) return Type::Missing();
  TensorType data_ty = GetUnaryInputTensorType(call);

  const auto* attrs = call->attrs.as<PermuteDimsAttrs>();

  // Todo(relax-team): revisit here for better check on if the input tensor has
  // ndim same as the number of input axes.
  if (!attrs->axes.has_value() && data_ty->IsUnknownNdim()) {
    return TensorType(data_ty->dtype, kUnknownNDim, data_ty->vdevice);
  }

  if (attrs->axes.has_value()) {
    int n_axis = attrs->axes.value().size();
    if (!data_ty->IsUnknownNdim() && n_axis != data_ty->ndim) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "PermuteDims expects the number of input axes to equal the ndim of the "
             "input tensor. However, the tensor ndim is "
          << data_ty->ndim << " while the given number of axes is " << n_axis;
    }
  }

  std::vector<int> axes;
  if (attrs->axes.has_value()) {
    axes = NormalizeAxes(call, data_ty->ndim, attrs->axes.value());
  } else {
    // Construct the reverse permutation via std::iota
    axes.resize(data_ty->ndim);
    std::iota(axes.rbegin(), axes.rend(), 0);
  }
  if (IsIdentityPermutation(axes)) {
    return data_ty;
  }

  const auto* data_shape = data_ty->shape.as<ShapeExprObj>();
  if (data_shape == nullptr) {
    return TensorType(data_ty->dtype, data_ty->ndim, data_ty->vdevice);
  }
  std::vector<PrimExpr> new_shape;
  new_shape.reserve(data_ty->ndim);
  for (int i = 0; i < data_ty->ndim; ++i) {
    new_shape.push_back(data_shape->values[axes[i]]);
  }
  return TensorType(ShapeExpr(new_shape), data_ty->dtype, data_ty->vdevice);
}

InferLayoutOutput InferLayoutPermuteDims(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  TVM_FFI_ICHECK(NoDesiredLayout(call, desired_layouts));

  const auto* attrs = call->attrs.as<PermuteDimsAttrs>();
  TVM_FFI_ICHECK(attrs != nullptr) << "Invalid Call";
  const auto* tensor_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
  TVM_FFI_ICHECK(tensor_ty != nullptr) << "Invalid Call";
  TVM_FFI_ICHECK(!tensor_ty->IsUnknownNdim()) << "Only support static ndim for now";
  int ndim = tensor_ty->ndim;

  LayoutDecision existing_layout = GetLayoutDecision(var_layout_map, call->args[0]);

  // permute_dims can't handle sub indexed layouts.
  if ((existing_layout->layout.has_value() ? existing_layout->layout.value().ndim() : 0) !=
      (existing_layout->layout.has_value() ? existing_layout->layout.value().ndim_primal() : 0)) {
    existing_layout = LayoutDecision(InitialLayout(ndim));
  }

  ffi::Array<int64_t> order;
  if (attrs->axes.has_value()) {
    order = attrs->axes.value();
  } else {
    order.reserve(ndim);
    for (int i = 0; i < ndim; ++i) {
      order.push_back(ndim - i - 1);
    }
  }
  std::string order_str;
  for (int64_t axis : order) {
    order_str.push_back(static_cast<char>(axis + 'A'));
  }
  ffi::String new_axes = TransposeStrLike(InitialLayout(ndim).name(), existing_layout->layout,
                                          SLayout::Create(order_str));
  ffi::Array<int64_t> new_order;
  for (size_t i = 0; i < new_axes.size(); ++i) {
    new_order.push_back(new_axes.at(i) - 'A');
  }
  ffi::ObjectPtr<PermuteDimsAttrs> new_attrs = ffi::make_object<PermuteDimsAttrs>(*attrs);
  new_attrs->axes = new_order;
  return InferLayoutOutput({existing_layout}, {InitialLayoutDecision(ndim)}, Attrs(new_attrs));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.permute_dims")
      .signature(sig::arg("x", "The input tensor."), sig::call_attrs<PermuteDimsAttrs>())
      .set_native_fn_attr<&InferTypePermuteDims>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutPermuteDims)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.reshape */
Expr ConvertNewShapeToExpr(const Expr& data,
                           const ffi::Variant<ffi::Array<PrimExpr>, Expr>& shape) {
  const ffi::ArrayObj* array;
  // Treat shape expressions as constant arrays to handle special values.
  if (const auto* e = shape.as<ShapeExprObj>()) {
    array = e->values.as<ffi::ArrayObj>();
    // Other non-shape expressions are used directly.
  } else if (const auto* e = shape.as<ExprObj>()) {
    return ffi::GetRef<Expr>(e);
    // Process special values in constants and produce an expression.
  } else {
    array = shape.as<ffi::ArrayObj>();
  }
  TVM_FFI_ICHECK(array != nullptr)
      << "Reshape only expects the input new shape to be either an Expr or an "
         "Array of PrimExprs. However, the given new shape is "
      << shape;
  int dim_to_infer = -1;
  // Keep track of which dimensions should be copied from input.
  std::vector<int> zero_dims;
  for (int i = 0; i < static_cast<int>(array->size()); ++i) {
    auto prim_len = array->at(i).as<PrimExpr>();
    TVM_FFI_ICHECK(prim_len)
        << "Reshape only expects the input new shape to be either an Expr or an "
           "Array of PrimExprs. However, the given new shape is "
        << shape;
    PrimExpr len = prim_len.value();
    TVM_FFI_CHECK(!len.as<DataflowVarObj>(), TypeError)
        << "Reshape shape dimensions must be symbolic primitive expressions, but received " << len;
    TVM_FFI_ICHECK(len.ty().code() == DLDataTypeCode::kDLInt)
        << "Reshape requires the new shape values to be all "
           "integers. However, the give new shape is "
        << shape;
    const auto* int_len = len.as<IntImmObj>();
    if (int_len != nullptr && int_len->value == 0) {
      // Note that this dimension should be copied from the original shape.
      zero_dims.push_back(i);
    } else if (int_len != nullptr && int_len->value == -1) {
      TVM_FFI_ICHECK_EQ(dim_to_infer, -1)
          << "Reshape accepts at most one \"-1\" in the new shape. However, "
             "there are multiple \"-1\" in the given new shape  "
          << shape;
      dim_to_infer = i;
    } else {
      TVM_FFI_ICHECK(int_len == nullptr || int_len->value > 0)
          << "Reshape requires all values in the new shape to be positive except a single \"-1\". "
             "However, the given new shape is "
          << shape;
    }
  }

  ffi::Array<PrimExpr> array_ref =
      ffi::GetRef<ffi::ObjectRef>(array).as_or_throw<ffi::Array<PrimExpr>>();
  // When there is no dimension to infer, just return the input array as ShapeExpr.
  if (dim_to_infer == -1 && zero_dims.empty()) {
    return ShapeExpr(array_ref);
  }

  // Otherwise, we require the input tensor to have known shape value for inference.
  const auto* data_ty = GetTypeAs<TensorTypeObj>(data);
  TVM_FFI_ICHECK(data_ty != nullptr)
      << "Reshape expects the input data to be a Tensor. However, the given input is "
      << data->ty->GetTypeKey();
  TVM_FFI_ICHECK(data_ty->shape.has_value())
      << "Reshape expects the input tensor to have known shape when there is some dimension length "
         "to infer. However, the given input has no shape.";
  const auto* shape_ty = GetTypeAs<ShapeTypeObj>(data_ty->shape.value());
  TVM_FFI_ICHECK(shape_ty != nullptr && shape_ty->values.has_value())
      << "Reshape expects the input tensor to have known shape when there is some dimension length "
         "to infer. However, the given input shape is "
      << data_ty->shape << " whose shape value is unknown.";

  // Set any 0 valued dimensions to match the corresponding input shape.
  if (!zero_dims.empty()) {
    for (int i : zero_dims) {
      array_ref.Set(i, shape_ty->values.value()[i]);
    }
  }

  // Set any -1 dimensions to complete the number of appropriate elements.
  // Start by computing the shape product of all positive indices.
  PrimExpr new_shape_prod = IntImm::Int64(1);
  for (int i = 0; i < static_cast<int>(array_ref.size()); ++i) {
    PrimExpr new_dim = array_ref[i];
    const auto* int_dim = new_dim.as<IntImmObj>();
    // We expect any symbolic not to signal the intent of -1, and therefore do no check for
    // symbolic value here.
    if (int_dim == nullptr || int_dim->value > 0) {
      new_shape_prod = new_shape_prod * new_dim;
    }
  }

  // Assign appropriate value to -1 dimension.
  if (dim_to_infer != -1) {
    sym::Analyzer analyzer;
    PrimExpr old_shape_prod = ComputeShapeProduct(shape_ty->values.value());
    array_ref.Set(dim_to_infer, analyzer->Simplify(floordiv(old_shape_prod, new_shape_prod)));
  }
  return ShapeExpr(array_ref);
}

// Prefer shape arrays to the generic Expr conversion of sequences to tuples.
Expr reshape(Expr x, ffi::Variant<ffi::Array<PrimExpr>, Expr> shape) {
  Expr shape_in_expr = ConvertNewShapeToExpr(x, shape);
  static const Op op = Op::Get("relax.reshape");
  return Call(Type::Missing(), op, {std::move(x), std::move(shape_in_expr)}, std::nullopt, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.op.reshape", reshape)
      .def("relax.op._NormalizeReshapeShape", ConvertNewShapeToExpr);
}

Type InferTypeReshape(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 2) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "Reshape op should take 2 arguments";
  }
  const auto* data_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
  const auto* new_shape_ty = GetTypeAs<ShapeTypeObj>(call->args[1]);
  if (data_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "Reshape requires the input data to be Tensor. However, the given one is "
        << call->args[0]->ty->GetTypeKey();
  }
  if (new_shape_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "Reshape requires the input new shape to be Shape. However, the given one is "
        << call->args[1]->ty->GetTypeKey();
  }

  ffi::Optional<ffi::Array<PrimExpr>> old_shape_values;
  if (data_ty->shape.has_value()) {
    const auto* old_shape_ty = GetTypeAs<ShapeTypeObj>(data_ty->shape.value());
    TVM_FFI_ICHECK_NOTNULL(old_shape_ty);
    old_shape_values = old_shape_ty->values;
  }

  if (new_shape_ty->values.has_value() && old_shape_values.has_value()) {
    PrimExpr new_shape_prod = ComputeShapeProduct(new_shape_ty->values.value());
    PrimExpr old_shape_prod = ComputeShapeProduct(old_shape_values.value());
    if (ctx->GetAnalyzer()->CanProve(old_shape_prod != new_shape_prod)) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "Reshape expects the new shape to be convertible from the old shape. "
             "However, the old shape is "
          << data_ty->shape << ", with product " << old_shape_prod << ", while the new shape is "
          << call->args[1] << ", with product " << new_shape_prod;
    }
  }
  Expr target_shape = call->args[1];
  // If shape values are defined, use them
  if (target_shape->IsInstance<VarObj>() && new_shape_ty->values.has_value()) {
    return TensorType(ShapeExpr(new_shape_ty->values.value()), data_ty->dtype, data_ty->vdevice);
  }
  return TensorType(target_shape, data_ty->dtype, data_ty->vdevice);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.reshape")
      .signature(sig::arg("x", "The input tensor."), sig::arg("shape", "The input new shape."))
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeReshape)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.split */

Expr split(Expr x, ffi::Variant<IntImm, ffi::Array<IntImm>> indices_or_sections, int axis) {
  ffi::ObjectPtr<SplitAttrs> attrs = ffi::make_object<SplitAttrs>();
  ffi::ObjectRef indices_or_sections_obj;

  if (const auto* indices = indices_or_sections.as<ffi::ArrayObj>()) {
    for (int i = 0; i < static_cast<int>(indices->size()); ++i) {
      const auto* idx = indices->at(i).as<IntImmObj>();
      TVM_FFI_ICHECK(idx != nullptr)
          << "Split op only accepts an array of integers as the indices. "
             "However, the given indices "
          << indices_or_sections << " contains some non-integer.";
    }
    indices_or_sections_obj = ConvertIntImmToInt64(
        ffi::GetRef<ffi::ObjectRef>(indices).as_or_throw<ffi::Array<IntImm>>());
  } else if (const auto* n_section = indices_or_sections.as<IntImmObj>()) {
    TVM_FFI_ICHECK_GT(n_section->value, 0)
        << "Split op expects the input number of sections to be a "
           "positive integer. However, the given number of sections is "
        << n_section->value;
    indices_or_sections_obj = IntImm::Int64(n_section->value);
  } else {
    TVM_FFI_THROW(InternalError)
        << "Split op expects the input indices_or_sections to be either an Array of "
           "PrimExpr or an integer.";
  }
  attrs->indices_or_sections = indices_or_sections_obj;
  attrs->axis = axis;

  static const Op op = Op::Get("relax.split");
  return Call(Type::Missing(), op, {std::move(x)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.split", split);
}

Type InferTypeSplit(const Call& call, const BlockBuilder& ctx) {
  TensorType data_ty = GetUnaryInputTensorType(call, ctx);
  const auto* attrs = call->attrs.as<SplitAttrs>();
  const auto* data_shape = data_ty->shape.as<ShapeExprObj>();
  int axis = data_ty->IsUnknownNdim() ? -1 : NormalizeAxis(call, ctx, data_ty->ndim, attrs->axis);

  if (auto opt_indices = attrs->indices_or_sections.as<ffi::Array<IntImm>>()) {
    auto p_indices = opt_indices.value();
    // When there is not index, return the input tensor's type.
    if (p_indices.size() == 0) {
      return data_ty;
    }
    // Fall back to unknown shape when the input tensor doesn't have ShapeExpr as shape.
    if (data_shape == nullptr) {
      return TupleType(ffi::Array<Type>(
          p_indices.size() + 1, TensorType(data_ty->dtype, data_ty->ndim, data_ty->vdevice)));
    }

    TVM_FFI_ICHECK_NE(axis, -1);

    IntImm zero(tvm::PrimType::Int(64), /*value=*/0);

    std::vector<Type> output_ty;
    for (size_t i = 0; i < p_indices.size() + 1; i++) {
      PrimExpr left = i == 0 ? zero : p_indices[i - 1];

      PrimExpr right = i < p_indices.size() ? p_indices[i] : data_shape->values[axis];

      left = tvm::min(tvm::max(left, 0), data_shape->values[axis]);
      right = tvm::min(tvm::max(right, 0), data_shape->values[axis]);

      PrimExpr split_dim = right - left;
      split_dim = tvm::max(split_dim, 0);
      split_dim = ctx->GetAnalyzer()->Simplify(split_dim);

      ffi::Array<PrimExpr> shape = data_shape->values;
      shape.Set(axis, split_dim);
      output_ty.push_back(TensorType(ShapeExpr(shape), data_ty->dtype, data_ty->vdevice));
    }
    return TupleType(output_ty);
  } else if (const auto* p_n_section = attrs->indices_or_sections.as<IntImmObj>()) {
    TVM_FFI_ICHECK_GT(p_n_section->value, 0);
    int n_section = p_n_section->value.as<int>().value();
    // When the number of section is one, return the input tensor's type.
    if (n_section == 1) {
      return data_ty;
    }
    // Fall back to unknown shape when the input tensor doesn't have ShapeExpr as shape.
    if (data_shape == nullptr) {
      return TupleType(
          ffi::Array<Type>(n_section, TensorType(data_ty->dtype, data_ty->ndim, data_ty->vdevice)));
    }
    TVM_FFI_ICHECK_NE(axis, -1);
    PrimExpr split_len = ceildiv(data_shape->values[axis], n_section);
    split_len = ctx->GetAnalyzer()->Simplify(split_len);

    // Construct type for tensors except the last one.
    ffi::Array<PrimExpr> shape = data_shape->values;
    shape.Set(axis, split_len);
    std::vector<Type> output_ty(n_section - 1,
                                TensorType(ShapeExpr(shape), data_ty->dtype, data_ty->vdevice));

    // Construct type for the last tensor.
    PrimExpr last_split_len = data_shape->values[axis] - split_len * (n_section - 1);
    last_split_len = ctx->GetAnalyzer()->Simplify(last_split_len);
    shape.Set(axis, last_split_len);
    output_ty.push_back(TensorType(ShapeExpr(shape), data_ty->dtype, data_ty->vdevice));
    return TupleType(output_ty);
  }
  TVM_FFI_ICHECK(false) << "Cannot reach here.";
  throw;
}

InferLayoutOutput InferLayoutSplit(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  TVM_FFI_ICHECK(NoDesiredLayout(call, desired_layouts));

  const auto* attrs = call->attrs.as<SplitAttrs>();
  TVM_FFI_ICHECK(attrs != nullptr) << "Invalid Call";
  const auto* tensor_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
  TVM_FFI_ICHECK(tensor_ty != nullptr) << "Invalid Call";
  TVM_FFI_ICHECK(!tensor_ty->IsUnknownNdim()) << "Only support known ndim";

  LayoutDecision existing_layout = GetLayoutDecision(var_layout_map, call->args[0]);
  Type out_ty = InferTypeSplit(call, BlockBuilder::Create(IRModule()));
  const auto* out_tuple = out_ty.as<TupleTypeObj>();

  /*
   * Fallback if the outputs can't be represented in input sub indexed layout
   * This can happen after sub indexing, if we can't split the corresponding primal axis
   */
  if ((existing_layout->layout.has_value() ? existing_layout->layout.value().ndim() : 0) !=
      (existing_layout->layout.has_value() ? existing_layout->layout.value().ndim_primal() : 0)) {
    for (const auto& si : out_tuple->fields) {
      TVM_FFI_ICHECK(si->IsInstance<TensorTypeObj>()) << "Fields of TupleType must be TensorType"
                                                         "output structinfo, but got "
                                                      << si;
      auto ty = si.as_or_throw<TensorType>();
      ffi::Optional<ShapeExpr> shape_expr = ffi::GetRef<ShapeExpr>(ty->shape.as<ShapeExprObj>());
      TVM_FFI_ICHECK(shape_expr.has_value());
      auto shape_arr = shape_expr.value();
      if (!CanProveLayoutTransform(InitialLayout(tensor_ty->ndim), existing_layout->layout,
                                   shape_arr->values)) {
        existing_layout = InitialLayout(tensor_ty->ndim);
        break;
      }
    }
  }

  ffi::ObjectPtr<SplitAttrs> new_attrs = ffi::make_object<SplitAttrs>(*attrs);
  new_attrs->axis = FindAxis(existing_layout->layout.value(), attrs->axis);
  TVM_FFI_ICHECK(out_tuple != nullptr) << "Invalid Call";
  NLayout tuple_layouts(ffi::Array<NLayout>(out_tuple->fields.size(), existing_layout));
  return InferLayoutOutput({existing_layout}, {tuple_layouts}, Attrs(new_attrs));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.split")
      .signature(sig::arg("x", "The input tensor."), sig::call_attrs<SplitAttrs>())
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeSplit)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutSplit)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.squeeze */

Expr squeeze(Expr x, ffi::Optional<ffi::Array<int64_t>> axis) {
  ffi::ObjectPtr<SqueezeAttrs> attrs = ffi::make_object<SqueezeAttrs>();
  attrs->axis = std::move(axis);

  static const Op op = Op::Get("relax.squeeze");
  return Call(Type::Missing(), op, {std::move(x)}, Attrs{attrs}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.squeeze", squeeze);
}

Type InferTypeSqueeze(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  if (RequiresTensorInputNormalization(call)) return Type::Missing();
  TensorType data_ty = GetUnaryInputTensorType(call);
  const auto* attrs = call->attrs.as<SqueezeAttrs>();
  if (attrs->axis.has_value() && attrs->axis.value().empty()) {
    return data_ty;
  }

  if (data_ty->IsUnknownNdim()) {
    return TensorType(data_ty->dtype, kUnknownNDim, data_ty->vdevice);
  }

  ffi::Optional<ffi::Array<PrimExpr>> shape_value;
  if (data_ty->shape.has_value()) {
    shape_value = data_ty->shape.value()->ty.as_or_throw<ShapeType>()->values;
  }

  std::vector<bool> axis_removal_mask;
  axis_removal_mask.resize(data_ty->ndim, /*value=*/false);

  if (attrs->axis.has_value()) {
    std::vector<int> axes = NormalizeAxes(call, data_ty->ndim, attrs->axis.value());

    if (!shape_value.has_value()) {
      return TensorType(data_ty->dtype, data_ty->ndim - axes.size(), data_ty->vdevice);
    }
    for (int i = 0; i < static_cast<int>(axes.size()); ++i) {
      // Todo(relax-team): revisit here for better check on if the axis being squeezed has length 1.
      // When `axis` is given, the dim lengths at the axes must be integer 1 when it is not symbolic
      const auto* int_len = shape_value.value()[axes[i]].as<IntImmObj>();
      // If a dimension is not 1, silently skip it (no-op), matching PyTorch behavior.
      if ((int_len != nullptr && int_len->value == 1) || int_len == nullptr) {
        axis_removal_mask[axes[i]] = true;
      }
    }
  } else {
    // When `axis` is not defined, squeeze all unit-length dimensions.
    // Note: This is a less well-defined path in Array API standard's squeeze
    // (https://data-apis.org/array-api/latest/API_specification/generated/array_api.squeeze.html).
    // Consider discourage usage later.
    if (!shape_value.has_value()) {
      return TensorType(data_ty->dtype, kUnknownNDim, data_ty->vdevice);
    }
    for (int i = 0; i < data_ty->ndim; ++i) {
      // Whenever a dimension length is symbolic, fall back to unknown ndim.
      const auto* int_len = shape_value.value()[i].as<IntImmObj>();
      if (int_len == nullptr) {
        return TensorType(data_ty->dtype, kUnknownNDim, data_ty->vdevice);
      }
      if (int_len->value == 1) {
        axis_removal_mask[i] = true;
      }
    }
  }

  std::vector<PrimExpr> output_shape;
  output_shape.reserve(data_ty->ndim - axis_removal_mask.size());
  for (int i = 0; i < data_ty->ndim; ++i) {
    if (!axis_removal_mask[i]) {
      output_shape.push_back(shape_value.value()[i]);
    }
  }

  if (data_ty->shape.value()->IsInstance<VarObj>()) {
    if (static_cast<int>(output_shape.size()) == data_ty->ndim) {
      return data_ty;
    } else if (attrs->axis.has_value()) {
      return TensorType(data_ty->dtype, output_shape.size(), data_ty->vdevice);
    } else {
      return TensorType(data_ty->dtype, kUnknownNDim, data_ty->vdevice);
    }
  } else {
    return TensorType(ShapeExpr(output_shape), data_ty->dtype, data_ty->vdevice);
  }
}

InferLayoutOutput InferLayoutSqueeze(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  TVM_FFI_ICHECK(NoDesiredLayout(call, desired_layouts));

  const auto* attrs = call->attrs.as<SqueezeAttrs>();
  TVM_FFI_ICHECK(attrs != nullptr) << "Invalid Call";
  const auto* tensor_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
  TVM_FFI_ICHECK(tensor_ty != nullptr) << "Invalid Call";
  TVM_FFI_ICHECK(!tensor_ty->IsUnknownNdim()) << "Only support static ndim for now";
  TVM_FFI_ICHECK(tensor_ty->shape.has_value()) << "Only support static shape for now";
  int ndim = tensor_ty->ndim;
  const auto* shape = tensor_ty->shape.as<ShapeExprObj>();
  TVM_FFI_ICHECK(shape != nullptr) << "Only support static shape for now";

  ffi::Array<int64_t> axis;
  if (attrs->axis.has_value()) {
    axis = attrs->axis.value();
  } else {
    axis.reserve(ndim);
    for (int i = 0; i < ndim; ++i) {
      if (tvm::prim::IsOne(shape->values[i])) {
        axis.push_back(i);
      }
    }
  }

  std::string axis_str(ndim, '0');
  for (int64_t iter : axis) {
    axis_str[iter] = '1';
  }
  for (int i = 0, j = 0; i < ndim; ++i) {
    if (axis_str[i] != '1') {
      axis_str[i] = 'A' + j++;
    }
  }

  LayoutDecision existing_layout = GetLayoutDecision(var_layout_map, call->args[0]);
  // Can't handle sub indexed layouts.
  if ((existing_layout->layout.has_value() ? existing_layout->layout.value().ndim() : 0) !=
      (existing_layout->layout.has_value() ? existing_layout->layout.value().ndim_primal() : 0)) {
    existing_layout = LayoutDecision(InitialLayout(ndim));
  }
  ffi::String new_axis_str =
      TransposeStrLike(axis_str, InitialLayout(ndim), existing_layout->layout);
  ffi::Array<int64_t> new_axis;
  for (size_t i = 0; i < new_axis_str.size(); ++i) {
    if (new_axis_str.at(i) == '1') {
      new_axis.push_back(static_cast<int64_t>(i));
    }
  }
  std::string output_layout = new_axis_str;
  output_layout.erase(std::remove(output_layout.begin(), output_layout.end(), '1'),
                      output_layout.end());

  ffi::ObjectPtr<SqueezeAttrs> new_attrs = ffi::make_object<SqueezeAttrs>(*attrs);
  new_attrs->axis = new_axis;
  return InferLayoutOutput({existing_layout}, {LayoutDecision(SLayout::Create(output_layout))},
                           Attrs(new_attrs));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.squeeze")
      .signature(sig::arg("x", "The input tensor."), sig::call_attrs<SqueezeAttrs>())
      .set_native_fn_attr<&InferTypeSqueeze>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutSqueeze)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

void CheckCollapseShape(const Call& call, const BlockBuilder& ctx,
                        const ffi::Array<PrimExpr>& data_shape,
                        const ffi::Array<PrimExpr>& target_shape) {
  sym::Analyzer analyzer = ctx->GetAnalyzer();

  int data_ndim = data_shape.size();
  int target_ndim = target_shape.size();

  int data_ax = data_ndim - 1;
  int target_ax = target_ndim - 1;
  for (; data_ax >= 0; --data_ax) {
    if (target_ax < 0) {
      continue;
    }
    const PrimExpr& dim0 = data_shape[data_ax];
    const PrimExpr& dim1 = target_shape[target_ax];
    const auto* int_dim0 = dim0.as<IntImmObj>();
    const auto* int_dim1 = dim1.as<IntImmObj>();

    if (analyzer->CanProveEqual(dim0, dim1) || (int_dim1 != nullptr && int_dim1->value == 1)) {
      --target_ax;
    } else if (int_dim0 && int_dim1 && int_dim0->value != int_dim1->value) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "In " << call->op << ", the data shape at dim " << data_ax << " is " << dim0
          << " and the target shape at dim " << target_ax << " is " << dim1
          << ", which do not match the rule of collapse sum.";
    } else {
      // Todo(relax-team): At this moment, enforcing MatchCast is fine. But we may need to revisit
      // this requirement to reduce the workload of importers and better support dynamic shapes.
      TVM_FFI_VISIT_THROW(ValueError, call)
          << call->op
          << " fails to match the axes because of unknown dim or symbolic"
             " shape. In this position the dim of data shape is "
          << dim0 << " while the dim of target shape is " << dim1
          << ". If it is symbolic, consider use MatchCast first.";
    }
  }
}

/* relax.stack */

Expr stack(Expr tensors, ffi::Optional<int64_t> axis) {
  ffi::ObjectPtr<StackAttrs> attrs = ffi::make_object<StackAttrs>();
  attrs->axis = std::move(axis);

  static const Op op = Op::Get("relax.stack");
  return Call(Type::Missing(), op, {std::move(tensors)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.stack", stack);
}

ffi::Optional<ffi::Array<PrimExpr>> CheckStackOutputShape(
    const Call& call, const BlockBuilder& ctx,
    const std::vector<ffi::Array<PrimExpr>>& shape_values, int axis) {
  bool shape_unknown = false;
  sym::Analyzer analyzer = ctx->GetAnalyzer();

  // Stack requires all input tensors to have identical shapes
  for (int d = 0; d < static_cast<int>(shape_values[0].size()); ++d) {
    for (int i = 1; i < static_cast<int>(shape_values.size()); ++i) {
      if (analyzer->CanProve(shape_values[i][d] != shape_values[0][d])) {
        TVM_FFI_VISIT_THROW(ValueError, call)
            << "Stack expects all input tensors to have identical shapes. "
            << "Dimension " << d << " differs between tensors: " << shape_values[0][d] << " vs "
            << shape_values[i][d];
      } else if (!analyzer->CanProveEqual(shape_values[i][d], shape_values[0][d])) {
        shape_unknown = true;
      }
    }
  }

  if (shape_unknown) {
    return std::nullopt;
  }

  // Insert new dimension at axis position
  ffi::Array<PrimExpr> output_shape;
  for (int i = 0; i < axis; ++i) {
    output_shape.push_back(shape_values[0][i]);
  }
  output_shape.push_back(IntImm::Int64(shape_values.size()));  // Stack dimension
  for (int i = axis; i < static_cast<int>(shape_values[0].size()); ++i) {
    output_shape.push_back(shape_values[0][i]);
  }
  return output_shape;
}

Type InferTypeStack(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 1) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "Stack op should have 1 argument";
  }

  ffi::Array<TensorType> tensor_ty = GetTensorTypeFromTuple(call, ctx, call->args[0]);
  if (tensor_ty.empty()) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Stack op expects at least one tensor in the input Tuple. "
        << "However, the given input Tuple is empty.";
  }

  const auto* attrs = call->attrs.as<StackAttrs>();
  TVM_FFI_ICHECK(attrs != nullptr) << "Stack must have StackAttrs";

  // Default axis is 0 if not specified
  int output_ndim = tensor_ty[0]->ndim + 1;  // Stack adds one dimension
  ffi::Optional<PrimType> output_dtype = std::nullopt;
  ffi::Optional<VDevice> vdev = std::nullopt;
  bool shape_unknown = false;
  bool is_void_dtype = false;
  bool vdevice_unknown = false;
  std::vector<ffi::Array<PrimExpr>> shape_values;
  shape_values.reserve(tensor_ty.size());

  for (TensorType ty : tensor_ty) {
    // Check dtype consistency
    if (ty->IsUnknownDtype()) {
      is_void_dtype = true;
    } else if (!output_dtype.has_value()) {
      output_dtype = ty->dtype;
    } else if (ty->dtype != output_dtype) {
      TVM_FFI_VISIT_THROW(TypeError, call)
          << "Stack expects all input tensors to have the same dtype. "
          << "Found " << output_dtype << " and " << ty->dtype;
    }

    // Check ndim consistency
    if (ty->ndim != kUnknownNDim && ty->ndim != tensor_ty[0]->ndim) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "Stack expects all input tensors to have same ndim. "
          << "Found " << tensor_ty[0]->ndim << " and " << ty->ndim;
    }

    // Check virtual device consistency
    if (!vdevice_unknown) {
      if (ty->vdevice.has_value()) {
        if (!vdev.has_value()) {
          vdev = ty->vdevice.value();
        } else if (ty->vdevice.value() != vdev.value()) {
          vdevice_unknown = true;
        }
      }
    }

    // Collect shape information
    const auto* shape_expr = ty->shape.as<ShapeExprObj>();
    if (shape_expr != nullptr) {
      shape_values.push_back(shape_expr->values);
      continue;
    }
    shape_unknown = true;

    if (!ty->shape.has_value()) continue;
    ShapeType shape_ty = ty->shape.value()->ty.as_or_throw<ShapeType>();
    if (shape_ty->values.has_value()) {
      shape_values.push_back(shape_ty->values.value());
    }
  }

  if (is_void_dtype) output_dtype = std::nullopt;
  if (vdevice_unknown) vdev = std::nullopt;

  // Normalize axis (default to 0 if not specified)
  int axis = attrs->axis.has_value()
                 ? NormalizeAxis(call, ctx, output_ndim, static_cast<int>(attrs->axis.value()))
                 : 0;

  // Single tensor case
  if (tensor_ty.size() == 1) {
    if (shape_values.empty()) {
      if (!vdevice_unknown) {
        return TensorType(output_dtype, output_ndim, vdev);
      }
      return TensorType(output_dtype, output_ndim);
    }
    ffi::Array<PrimExpr> output_shape;
    for (int i = 0; i < axis; ++i) {
      output_shape.push_back(shape_values[0][i]);
    }
    output_shape.push_back(1);  // Stack size 1
    for (int i = axis; i < static_cast<int>(shape_values[0].size()); ++i) {
      output_shape.push_back(shape_values[0][i]);
    }
    if (!vdevice_unknown) {
      return TensorType(ShapeExpr(output_shape), output_dtype, vdev);
    }
    return TensorType(ShapeExpr(output_shape), output_dtype);
  }

  // Multiple tensors case
  if (shape_values.empty()) {
    if (!vdevice_unknown) {
      return TensorType(output_dtype, output_ndim, vdev);
    }
    return TensorType(output_dtype, output_ndim);
  }

  ffi::Optional<ffi::Array<PrimExpr>> output_shape =
      CheckStackOutputShape(call, ctx, shape_values, axis);
  if (shape_unknown || !output_shape.has_value()) {
    if (!vdevice_unknown) {
      return TensorType(output_dtype, output_ndim, vdev);
    }
    return TensorType(output_dtype, output_ndim);
  } else {
    if (!vdevice_unknown) {
      return TensorType(ShapeExpr(output_shape.value()), output_dtype, vdev);
    }
    return TensorType(ShapeExpr(output_shape.value()), output_dtype);
  }
}

InferLayoutOutput InferLayoutStack(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  TVM_FFI_ICHECK(NoDesiredLayout(call, desired_layouts));

  const auto* attrs = call->attrs.as<StackAttrs>();
  TVM_FFI_ICHECK(attrs != nullptr) << "Invalid Call";
  NLayout nlayout = GetNLayout(var_layout_map, call->args[0]);
  TVM_FFI_ICHECK(nlayout.IsNested());
  TVM_FFI_ICHECK(nlayout.NestedArray()[0].IsLeaf());

  int n_tensor = nlayout.NestedArray().size();
  LayoutDecision layout = nlayout.NestedArray()[0].LeafValue();
  ffi::Array<NLayout> input_layouts, output_layouts;
  for (int i = 0; i < n_tensor; ++i) {
    input_layouts.push_back(layout);
  }

  // For stack, we need to adjust the output layout by inserting a new axis
  std::string layout_str =
      (layout->layout.has_value() ? layout->layout.value().name() : "__undef__");
  int axis = attrs->axis.has_value() ? static_cast<int>(attrs->axis.value()) : 0;
  layout_str.insert(static_cast<size_t>(axis), "S");  // Add stack dimension
  SLayout output_layout = SLayout(layout_str);
  output_layouts.push_back(LayoutDecision(output_layout));

  ffi::ObjectPtr<StackAttrs> new_attrs = ffi::make_object<StackAttrs>(*attrs);
  new_attrs->axis = static_cast<int64_t>(FindAxis(layout->layout.value(), axis));
  return InferLayoutOutput({NLayout(input_layouts)}, output_layouts, Attrs(new_attrs));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.stack")
      .signature(sig::arg("tensors", "The input list of tensors to stack"),
                 sig::call_attrs<StackAttrs>())
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeStack)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutStack)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.collapse_sum_like */
Expr collapse_sum_like(Expr data, Expr collapse_target) {
  static const Op op = Op::Get("relax.collapse_sum_like");
  return Call(Type::Missing(), op, {std::move(data), std::move(collapse_target)}, std::nullopt, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.collapse_sum_like", collapse_sum_like);
}

Type InferTypeCollapseSumLike(const Call& call, const BlockBuilder& ctx) {
  ffi::Array<TensorType> input_ty = GetInputTensorType(call, ctx);
  TensorType data_ty = input_ty[0];
  TensorType collapse_target_ty = input_ty[1];

  ffi::Optional<PrimType> output_dtype = data_ty->dtype;

  ffi::Optional<ffi::Array<PrimExpr>> data_shape_value;
  if (data_ty->shape.has_value()) {
    data_shape_value = GetTypeAs<ShapeTypeObj>(data_ty->shape.value())->values;
  }
  ffi::Optional<ffi::Array<PrimExpr>> collapse_target_shape_value;
  if (collapse_target_ty->shape.has_value()) {
    collapse_target_shape_value =
        GetTypeAs<ShapeTypeObj>(collapse_target_ty->shape.value())->values;
  }

  if (data_shape_value.has_value() && collapse_target_shape_value.has_value()) {
    CheckCollapseShape(call, ctx, data_shape_value.value(), collapse_target_shape_value.value());
  }

  if (collapse_target_ty->shape.has_value()) {
    return TensorType(collapse_target_ty->shape.value(), output_dtype, collapse_target_ty->vdevice);
  } else {
    return TensorType(output_dtype, collapse_target_ty->ndim, collapse_target_ty->vdevice);
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.collapse_sum_like")
      .signature(sig::arg("data", "The input tensor."),
                 sig::arg("collapse_target", "The tensor whose shape is the shape to collapse to."))
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeCollapseSumLike)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.collapse_sum_to */
Expr collapse_sum_to(Expr data, Expr shape) {
  static const Op op = Op::Get("relax.collapse_sum_to");
  return Call(Type::Missing(), op, {std::move(data), std::move(shape)}, std::nullopt, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.collapse_sum_to", collapse_sum_to);
}

Type InferTypeCollapseSumTo(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 2) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "CollapseSumTo should have 2 arguments";
  }

  const auto* data_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
  const auto* shape_ty = GetTypeAs<ShapeTypeObj>(call->args[1]);

  if (data_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "CollapseSumTo requires the input data to be a Tensor. However, the given one is "
        << call->args[0]->ty->GetTypeKey();
  }
  if (shape_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "CollapseSumTo requires the input shape to be a Shape. However, the given one is "
        << call->args[1]->ty->GetTypeKey();
  }

  ffi::Optional<PrimType> output_dtype = data_ty->dtype;

  ffi::Optional<ffi::Array<PrimExpr>> data_shape_value;
  if (data_ty->shape.has_value()) {
    data_shape_value = GetTypeAs<ShapeTypeObj>(data_ty->shape.value())->values;
  }

  if (data_shape_value.has_value() && shape_ty->values.has_value()) {
    CheckCollapseShape(call, ctx, data_shape_value.value(), shape_ty->values.value());
  }
  return TensorType(/*shape=*/call->args[1], output_dtype, data_ty->vdevice);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.collapse_sum_to")
      .signature(sig::arg("data", "The input tensor."),
                 sig::arg("shape", "The shape to collapse to."))
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeCollapseSumTo)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.repeat */

Expr repeat(Expr data, int repeats, ffi::Optional<int64_t> axis) {
  auto attrs = ffi::make_object<RepeatAttrs>();
  attrs->repeats = std::move(repeats);
  attrs->axis = std::move(axis);

  static const Op op = Op::Get("relax.repeat");
  return Call(Type::Missing(), op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.repeat", repeat);
}

Type InferTypeRepeat(const Call& call, const BlockBuilder& ctx) {
  sym::Analyzer analyzer = ctx->GetAnalyzer();
  TensorType data_ty = GetUnaryInputTensorType(call, ctx);
  const auto* attrs = call->attrs.as<RepeatAttrs>();
  const auto* data_shape = data_ty->shape.as<ShapeExprObj>();

  if (attrs->axis.has_value() && !data_ty->IsUnknownNdim()) {
    int axis = attrs->axis.value();
    int ndim = data_ty->ndim;
    if (axis < -ndim || axis >= ndim) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "Repeat requires the input axis belongs range "
             "[-data.ty.ndim, data.ty.ndim - 1]. However, the input axis is "
          << axis << ", while ndim is " << ndim;
    }
  }

  if (data_shape == nullptr) {
    if (attrs->axis.has_value()) {
      if (analyzer->CanProveEqual(attrs->repeats, 1)) {
        // the shape does not changes
        return data_ty;
      } else {
        return TensorType(data_ty->dtype, data_ty->ndim, data_ty->vdevice);
      }
    } else {
      return TensorType(data_ty->dtype, 1, data_ty->vdevice);
    }
  }

  if (!attrs->axis.has_value()) {
    PrimExpr new_shape =
        analyzer->Simplify(ComputeShapeProduct(data_shape->values) * attrs->repeats);
    return TensorType(ShapeExpr(ffi::Array<PrimExpr>({new_shape})), data_ty->dtype,
                      data_ty->vdevice);
  }

  int axis = NormalizeAxis(call, ctx, data_ty->ndim, attrs->axis.value());
  auto shape_array = data_shape->values;
  shape_array.Set(axis, analyzer->Simplify(shape_array[axis] * attrs->repeats));
  return TensorType(ShapeExpr(shape_array), data_ty->dtype, data_ty->vdevice);
}

InferLayoutOutput InferLayoutRepeat(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  TVM_FFI_ICHECK(NoDesiredLayout(call, desired_layouts));

  const auto* attrs = call->attrs.as<RepeatAttrs>();
  TVM_FFI_ICHECK(attrs != nullptr) << "Invalid Call";
  const auto* tensor_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
  TVM_FFI_ICHECK(tensor_ty != nullptr) << "Invalid Call";
  TVM_FFI_ICHECK(!tensor_ty->IsUnknownNdim()) << "Only support static ndim for now";

  LayoutDecision existing_layout = GetLayoutDecision(var_layout_map, call->args[0]);
  int ndim = tensor_ty->ndim;

  // Can't handle sub indexed layouts.
  if ((existing_layout->layout.has_value() ? existing_layout->layout.value().ndim() : 0) !=
      (existing_layout->layout.has_value() ? existing_layout->layout.value().ndim_primal() : 0)) {
    existing_layout = LayoutDecision(InitialLayout(ndim));
  }

  // When axis is not specified, the output is the flattened input. Flattening is order-sensitive,
  // so the input must be restored to its initial (logical) layout.
  if (!attrs->axis.has_value()) {
    return InferLayoutOutput({InitialLayoutDecision(ndim)}, {InitialLayoutDecision(1)},
                             call->attrs);
  }

  // Transform the axis based on the layout
  int axis = attrs->axis.value();
  if (axis < 0) {
    axis += ndim;
  }

  // Create a mapping from original layout to existing layout
  std::string axis_str(ndim, '0');
  axis_str[axis] = '1';
  for (int i = 0, j = 0; i < ndim; ++i) {
    if (axis_str[i] != '1') {
      axis_str[i] = 'A' + j++;
    }
  }

  ffi::String new_axis_str =
      TransposeStrLike(axis_str, InitialLayout(ndim), existing_layout->layout);

  int64_t new_axis = -1;
  for (size_t i = 0; i < new_axis_str.size(); ++i) {
    if (new_axis_str.at(i) == '1') {
      new_axis = i;
      break;
    }
  }
  TVM_FFI_ICHECK_GE(new_axis, 0) << "Failed to find transformed axis";

  ffi::ObjectPtr<RepeatAttrs> new_attrs = ffi::make_object<RepeatAttrs>(*attrs);
  new_attrs->axis = new_axis;

  // When axis is specified, the layout is preserved
  return InferLayoutOutput({existing_layout}, {existing_layout}, Attrs(new_attrs));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.repeat")
      .signature(sig::arg("data", "The input tensor."), sig::call_attrs<RepeatAttrs>())
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeRepeat)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutRepeat)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.tile */

Expr tile(Expr data, ffi::Array<int64_t> repeats) {
  auto attrs = ffi::make_object<TileAttrs>();
  attrs->repeats = std::move(repeats);

  static const Op op = Op::Get("relax.tile");
  return Call(Type::Missing(), op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.tile", tile);
}

Type InferTypeTile(const Call& call, const BlockBuilder& ctx) {
  sym::Analyzer analyzer = ctx->GetAnalyzer();
  TensorType data_ty = GetUnaryInputTensorType(call, ctx);
  const auto* attrs = call->attrs.as<TileAttrs>();
  const auto* data_shape = data_ty->shape.as<ShapeExprObj>();
  int l = attrs->repeats.size();
  int ndim = data_ty->ndim;

  if (data_shape == nullptr) {
    if (data_ty->IsUnknownNdim()) {
      return TensorType(data_ty->dtype, kUnknownNDim, data_ty->vdevice);
    }
    if (l > ndim) {
      return TensorType(data_ty->dtype, l, data_ty->vdevice);
    } else {
      for (int64_t i : attrs->repeats) {
        if (i != 1) {
          return TensorType(data_ty->dtype, data_ty->ndim, data_ty->vdevice);
        }
      }
      // if control reaches here, the shape should not be changed
      return data_ty;
    }
  }

  int out_ndim = std::max(l, ndim);
  int l_delta = out_ndim - l;
  int ndim_delta = out_ndim - ndim;
  ffi::Array<PrimExpr> out_shape;
  for (int i = 0; i < out_ndim; ++i) {
    if (i < l_delta) {
      out_shape.push_back(data_shape->values[i - ndim_delta]);
    } else if (i < ndim_delta) {
      out_shape.push_back(IntImm::Int64(attrs->repeats[i - l_delta]));
    } else {
      out_shape.push_back(analyzer->Simplify(data_shape->values[i - ndim_delta] *
                                             IntImm::Int64(attrs->repeats[i - l_delta])));
    }
  }

  return TensorType(ShapeExpr(out_shape), data_ty->dtype, data_ty->vdevice);
}

InferLayoutOutput InferLayoutTile(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  TVM_FFI_ICHECK(NoDesiredLayout(call, desired_layouts));

  const auto* attrs = call->attrs.as<TileAttrs>();
  TVM_FFI_ICHECK(attrs != nullptr) << "Invalid Call";
  const auto* tensor_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
  TVM_FFI_ICHECK(tensor_ty != nullptr) << "Invalid Call";
  TVM_FFI_ICHECK(!tensor_ty->IsUnknownNdim()) << "Only support static ndim for now";

  LayoutDecision existing_layout = GetLayoutDecision(var_layout_map, call->args[0]);
  int ndim = tensor_ty->ndim;
  int l = attrs->repeats.size();
  int out_ndim = std::max(l, ndim);

  // Can't handle sub indexed layouts.
  if ((existing_layout->layout.has_value() ? existing_layout->layout.value().ndim() : 0) !=
      (existing_layout->layout.has_value() ? existing_layout->layout.value().ndim_primal() : 0)) {
    existing_layout = LayoutDecision(InitialLayout(ndim));
  }

  // Tile operation repeats data along each axis.
  // When layout changes, we need to transform the repeats array to match the new layout.
  SLayout initial_layout = InitialLayout(ndim);
  SLayout existing_layout_obj = existing_layout->layout.value();

  // Transform repeats array according to layout change.
  // The repeats array semantics:
  // - If len(repeats) < ndim: repeats are right-aligned, padded with 1s at the beginning.
  //   e.g., ndim=4, repeats=[2, 1] means [1, 1, 2, 1]
  // - If len(repeats) > ndim: first (len(repeats) - ndim) elements are new dimensions,
  //   remaining elements correspond to input dimensions.
  //   e.g., ndim=4, repeats=[2, 1, 2, 1, 1] means new dims [2, 1] + input dims [2, 1, 1]
  ffi::Array<int64_t> new_repeats;

  if (out_ndim == ndim) {
    // Same dimension: reorder repeats according to layout transformation.
    // If len(repeats) < ndim, it's padded with 1s at the beginning.
    for (int i = 0; i < ndim; ++i) {
      const tirx::SLayoutAxis& axis = existing_layout_obj[i];
      int pos_in_initial = initial_layout.IndexOf(axis);
      TVM_FFI_ICHECK_NE(pos_in_initial, -1) << "Axis not found in initial layout";
      // If len(repeats) < ndim, repeats are right-aligned.
      // pos_in_initial >= (ndim - l) means it's within the repeats array range.
      if (pos_in_initial >= ndim - l) {
        new_repeats.push_back(attrs->repeats[pos_in_initial - (ndim - l)]);
      } else {
        new_repeats.push_back(1);
      }
    }
  } else {
    // Different dimension: handle dimension expansion.
    // This case only happens when l > ndim.
    TVM_FFI_ICHECK_GT(l, ndim);
    int num_new_dims = l - ndim;
    // Repeats for new dimensions are not affected by layout change.
    for (int i = 0; i < num_new_dims; ++i) {
      new_repeats.push_back(attrs->repeats[i]);
    }
    // Repeats for existing dimensions need to be permuted.
    for (int i = 0; i < ndim; ++i) {
      const tirx::SLayoutAxis& axis = existing_layout_obj[i];
      int pos_in_initial = initial_layout.IndexOf(axis);
      TVM_FFI_ICHECK_NE(pos_in_initial, -1) << "Axis not found in initial layout";
      new_repeats.push_back(attrs->repeats[pos_in_initial + num_new_dims]);
    }
  }

  ffi::ObjectPtr<TileAttrs> new_attrs = ffi::make_object<TileAttrs>(*attrs);
  new_attrs->repeats = new_repeats;

  // Layout is preserved (same as input)
  LayoutDecision output_layout =
      (out_ndim == ndim) ? existing_layout : FollowDecision(existing_layout, out_ndim);

  return InferLayoutOutput({existing_layout}, {output_layout}, Attrs(new_attrs));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.tile")
      .signature(sig::arg("data", "The input tensor."), sig::call_attrs<TileAttrs>())
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeTile)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutTile)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.flip */

Expr flip(Expr data, int64_t axis) {
  auto attrs = ffi::make_object<FlipAttrs>();
  attrs->axis = axis;
  static const Op op = Op::Get("relax.flip");
  return Call(Type::Missing(), op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.flip", flip);
}

Type InferTypeFlip(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  if (call->args.size() != 1) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "Flip op should take 1 argument";
  }
  TensorType data_ty = GetUnaryInputTensorType(call);
  const auto* attrs = call->attrs.as<FlipAttrs>();
  int axis = static_cast<int>(attrs->axis);
  if (!data_ty->IsUnknownNdim()) {
    int ndim = data_ty->ndim;
    if (axis < -ndim || axis >= ndim) {
      TVM_FFI_VISIT_THROW(ValueError, call) << "Flip requires the input axis belongs range "
                                               "[-ndim, ndim - 1]. However, the input axis is "
                                            << axis << ", while ndim is " << ndim;
    }
  }
  return data_ty;
}

InferLayoutOutput InferLayoutFlip(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  TVM_FFI_ICHECK(NoDesiredLayout(call, desired_layouts));

  const auto* attrs = call->attrs.as<FlipAttrs>();
  TVM_FFI_ICHECK(attrs != nullptr) << "Invalid Call";
  const auto* tensor_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
  TVM_FFI_ICHECK(tensor_ty != nullptr) << "Invalid Call";
  TVM_FFI_ICHECK(!tensor_ty->IsUnknownNdim()) << "Only support static ndim for now";

  LayoutDecision existing_layout = GetLayoutDecision(var_layout_map, call->args[0]);
  int ndim = tensor_ty->ndim;

  if ((existing_layout->layout.has_value() ? existing_layout->layout.value().ndim() : 0) !=
      (existing_layout->layout.has_value() ? existing_layout->layout.value().ndim_primal() : 0)) {
    existing_layout = LayoutDecision(InitialLayout(ndim));
  }

  int axis = static_cast<int>(attrs->axis);
  if (axis < 0) {
    axis += ndim;
  }

  const int new_axis = FindAxis(existing_layout->layout.value(), axis);
  TVM_FFI_ICHECK_GE(new_axis, 0) << "Failed to find transformed axis";

  ffi::ObjectPtr<FlipAttrs> new_attrs = ffi::make_object<FlipAttrs>(*attrs);
  new_attrs->axis = static_cast<int64_t>(new_axis);

  return InferLayoutOutput({existing_layout}, {existing_layout}, Attrs(new_attrs));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.flip")
      .signature(sig::arg("data", "The input tensor."), sig::call_attrs<FlipAttrs>())
      .set_native_fn_attr<&InferTypeFlip>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutFlip)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.reverse_sequence */

Expr reverse_sequence(Expr data, Expr seq_lengths, int64_t seq_axis, int64_t batch_axis) {
  auto attrs = ffi::make_object<ReverseSequenceAttrs>();
  attrs->seq_axis = seq_axis;
  attrs->batch_axis = batch_axis;
  static const Op op = Op::Get("relax.reverse_sequence");
  return Call(Type::Missing(), op, {std::move(data), std::move(seq_lengths)}, Attrs{attrs}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.reverse_sequence", reverse_sequence);
}

Type InferTypeReverseSequence(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 2) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "ReverseSequence op should take 2 arguments";
  }
  TensorType data_ty = GetInputTensorType(call, 0, ctx);
  TensorType seq_lengths_ty = GetInputTensorType(call, 1, ctx);

  if (!seq_lengths_ty->IsUnknownNdim() && seq_lengths_ty->ndim != 1) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "ReverseSequence requires seq_lengths to be 1-D. However, seq_lengths has ndim "
        << seq_lengths_ty->ndim;
  }
  ffi::Optional<PrimType> seq_lengths_dtype = seq_lengths_ty->dtype;
  if (!seq_lengths_ty->IsUnknownDtype() &&
      !seq_lengths_ty->dtype.value().MatchesCode(DLDataTypeCode::kDLInt)) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "ReverseSequence requires seq_lengths to have dtype int32 or int64. However, "
           "seq_lengths has dtype "
        << seq_lengths_ty->dtype;
  }
  if (seq_lengths_dtype.has_value() &&
      seq_lengths_dtype.value().MatchesCode(DLDataTypeCode::kDLInt) &&
      seq_lengths_dtype.value()->dtype.bits != 32 && seq_lengths_dtype.value()->dtype.bits != 64) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "ReverseSequence requires seq_lengths to have dtype int32 or int64. However, "
           "seq_lengths has dtype "
        << seq_lengths_ty->dtype;
  }

  const auto* attrs = call->attrs.as<ReverseSequenceAttrs>();
  int64_t seq_axis = attrs->seq_axis;
  int64_t batch_axis = attrs->batch_axis;
  if (!data_ty->IsUnknownNdim()) {
    int ndim = data_ty->ndim;
    auto check_axis = [&](int64_t axis, ffi::String axis_name) {
      if (axis < -ndim || axis >= ndim) {
        TVM_FFI_VISIT_THROW(ValueError, call)
            << "ReverseSequence requires " << axis_name
            << " to belong to range [-ndim, ndim). However, the axis is " << axis
            << ", while ndim is " << ndim;
      }
    };
    check_axis(seq_axis, "seq_axis");
    check_axis(batch_axis, "batch_axis");

    if (batch_axis < 0) {
      batch_axis += ndim;
    }

    if (data_ty->shape.has_value() && seq_lengths_ty->shape.has_value()) {
      const auto* data_shape_ty = GetTypeAs<ShapeTypeObj>(data_ty->shape.value());
      const auto* seq_lengths_shape_ty = GetTypeAs<ShapeTypeObj>(seq_lengths_ty->shape.value());
      if (data_shape_ty != nullptr && seq_lengths_shape_ty != nullptr &&
          data_shape_ty->values.has_value() && seq_lengths_shape_ty->values.has_value()) {
        PrimExpr batch_extent = data_shape_ty->values.value()[batch_axis];
        PrimExpr seq_lengths_extent = seq_lengths_shape_ty->values.value()[0];
        if (ctx->GetAnalyzer()->CanProve(seq_lengths_extent != batch_extent)) {
          TVM_FFI_VISIT_THROW(ValueError, call)
              << "ReverseSequence requires seq_lengths.shape[0] to equal the batch axis extent. "
                 "However, seq_lengths.shape[0] is "
              << seq_lengths_extent << ", while data.shape[" << batch_axis << "] is "
              << batch_extent;
        }
      }
    }
  }

  return data_ty;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.reverse_sequence")
      .signature(sig::arg("data", "The input tensor."),
                 sig::arg("seq_lengths", "The sequence length tensor."),
                 sig::call_attrs<ReverseSequenceAttrs>())
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeReverseSequence)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.gather_elements */

Expr gather_elements(Expr data, Expr indices, int axis) {
  auto attrs = ffi::make_object<GatherElementsAttrs>();
  attrs->axis = axis;
  static const Op op = Op::Get("relax.gather_elements");
  return Call(Type::Missing(), op, {data, indices}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.gather_elements", gather_elements);
}

Type InferTypeGatherElements(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  if (RequiresTensorInputNormalization(call)) return Type::Missing();
  const auto* data_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
  const auto* indices_ty = GetTypeAs<TensorTypeObj>(call->args[1]);
  const auto* attrs = call->attrs.as<GatherElementsAttrs>();

  if (data_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "GatherElements requires the input data to be a Tensor. However, the given one is "
        << call->args[0]->ty->GetTypeKey();
  }
  if (indices_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "GatherElements requires the input indices to be a Tensor. However, the given one is "
        << call->args[1]->ty->GetTypeKey();
  }

  // Gather indices only require integer element kind; vector lanes do not affect shape inference.
  if (!indices_ty->IsUnknownDtype() && indices_ty->dtype.value().code() != DLDataTypeCode::kDLInt) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "GatherElements requires the input indices to have int64 dtype. However, the "
        << "given indices dtype is " << indices_ty->dtype;
  }

  if (data_ty->IsUnknownNdim() || indices_ty->IsUnknownNdim()) {
    return TensorType(data_ty->dtype, kUnknownNDim, data_ty->vdevice);
  }

  int axis = static_cast<int>(attrs->axis);
  if (axis < -data_ty->ndim || axis >= data_ty->ndim) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "GatherElements requires axis to be within the input dimension range [" << -data_ty->ndim
        << ", " << data_ty->ndim - 1 << "]. However, the "
        << "given axis is " << axis;
  }

  if (data_ty->ndim != indices_ty->ndim) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "GatherElements requires data and indices to have the same rank. However, "
        << "data rank is " << data_ty->ndim << " while indices rank is " << indices_ty->ndim;
  }
  if (indices_ty->shape.has_value()) {
    return TensorType(indices_ty->shape.value(), data_ty->dtype, data_ty->vdevice);
  }
  return TensorType(data_ty->dtype, indices_ty->ndim, data_ty->vdevice);
}

InferLayoutOutput InferLayoutGatherElements(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  TVM_FFI_ICHECK(NoDesiredLayout(call, desired_layouts));
  const auto* attrs = call->attrs.as<GatherElementsAttrs>();
  TVM_FFI_ICHECK(attrs) << "Invalid Call";

  LayoutDecision data_layout = GetLayoutDecision(var_layout_map, call->args[0]);
  LayoutDecision indices_layout = GetLayoutDecision(var_layout_map, call->args[1]);

  LayoutDecision layout = data_layout;
  // If data_layout is initial and indices_layout is not, prefer indices_layout.
  bool data_is_initial =
      (data_layout->layout.has_value() ? data_layout->layout.value().name() : "__undef__") ==
      InitialLayout((data_layout->layout.has_value() ? data_layout->layout.value().ndim() : 0))
          .name();
  bool indices_is_initial =
      (indices_layout->layout.has_value() ? indices_layout->layout.value().name() : "__undef__") ==
      InitialLayout(
          (indices_layout->layout.has_value() ? indices_layout->layout.value().ndim() : 0))
          .name();
  if (data_is_initial && !indices_is_initial) {
    layout = indices_layout;
  }

  if ((layout->layout.has_value() ? layout->layout.value().ndim() : 0) !=
      (layout->layout.has_value() ? layout->layout.value().ndim_primal() : 0)) {
    const auto* tensor_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
    TVM_FFI_ICHECK(tensor_ty != nullptr) << "Invalid Call";
    TVM_FFI_ICHECK(!tensor_ty->IsUnknownNdim()) << "Only support static ndim for now";
    int ndim = tensor_ty->ndim;
    layout = LayoutDecision(InitialLayout(ndim));
  }

  ffi::ObjectPtr<GatherElementsAttrs> new_attrs = ffi::make_object<GatherElementsAttrs>(*attrs);
  new_attrs->axis = FindAxis(layout->layout.value(), attrs->axis);
  return InferLayoutOutput({layout, layout}, {layout}, Attrs(new_attrs));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.gather_elements")
      .signature(sig::arg("data", "The input tensor."), sig::arg("indices", "The indices tensor."),
                 sig::call_attrs<GatherElementsAttrs>())
      .set_native_fn_attr<&InferTypeGatherElements>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutGatherElements)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.gather_nd */

Expr gather_nd(Expr data, Expr indices, int batch_dims) {
  auto attrs = ffi::make_object<GatherNDAttrs>();
  attrs->batch_dims = batch_dims;
  static const Op op = Op::Get("relax.gather_nd");
  return Call(Type::Missing(), op, {data, indices}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.gather_nd", gather_nd);
}

Type InferTypeGatherND(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  if (RequiresTensorInputNormalization(call)) return Type::Missing();
  const auto* data_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
  const auto* indices_ty = GetTypeAs<TensorTypeObj>(call->args[1]);
  const auto* attrs = call->attrs.as<GatherNDAttrs>();

  if (data_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "GatherND requires the input data to be a Tensor. However, the given one is "
        << call->args[0]->ty->GetTypeKey();
  }
  if (indices_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "GatherND requires the input indices to be a Tensor. However, the given one is "
        << call->args[1]->ty->GetTypeKey();
  }
  TVM_FFI_ICHECK_GE(attrs->batch_dims, 0);
  int batch_dims = static_cast<int>(attrs->batch_dims);
  int input_dims = data_ty->ndim;
  if (!indices_ty->IsUnknownDtype() && indices_ty->dtype != PrimType::Int(64)) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "GatherND requires the input indices to have int64 dtype. However, the "
        << "given indices dtype is " << indices_ty->dtype;
  }

  if (data_ty->IsUnknownNdim() || indices_ty->IsUnknownNdim()) {
    return TensorType(data_ty->dtype, kUnknownNDim, data_ty->vdevice);
  }

  if (batch_dims < 0 || batch_dims > data_ty->ndim) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "GatherND batch_dims must be in range [0, data.ndim]. However, got batch_dims="
        << batch_dims << ", data.ndim=" << input_dims;
  }

  if (batch_dims > indices_ty->ndim - 1) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "GatherND batch_dims cannot exceed indices.ndim-1. However, got batch_dims="
        << batch_dims << ", indices.ndim=" << indices_ty->ndim;
  }

  // Check if indices shape is known
  const auto* indices_shape = indices_ty->shape.as<ShapeExprObj>();
  const auto* data_shape = data_ty->shape.as<ShapeExprObj>();
  if (!indices_shape || !indices_shape->values.back()->IsInstance<IntImmObj>()) {
    return TensorType(data_ty->dtype, kUnknownNDim, data_ty->vdevice);
  }
  int l = indices_shape->values.back().as<IntImmObj>()->value.as<int>().value();
  int output_ndim = indices_ty->ndim + input_dims - l - 1 - batch_dims;
  if (!data_shape) {
    return TensorType(data_ty->dtype, output_ndim, data_ty->vdevice);
  }

  // In this condition, all input shapes are known
  ffi::Array<PrimExpr> out_shape;
  if (l > input_dims - batch_dims) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "GatherND requires the last dimension of indices to be less than or "
           "equal to the rank of data minus batch_dims. However, the given shapes are "
        << "indices: " << ShapeExpr(indices_shape->values)
        << ", data: " << ShapeExpr(data_shape->values) << ", with batch_dims=" << batch_dims;
  }
  for (int i = 0; i < indices_ty->ndim - 1; ++i) {
    out_shape.push_back(indices_shape->values[i]);
  }
  for (int i = batch_dims + l; i < input_dims; ++i) {
    out_shape.push_back(data_shape->values[i]);
  }
  TVM_FFI_ICHECK_EQ(out_shape.size(), output_ndim);
  return TensorType(ShapeExpr(out_shape), data_ty->dtype, data_ty->vdevice);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.gather_nd")
      .signature(sig::arg("data", "The input tensor."), sig::arg("indices", "The indices tensor."),
                 sig::call_attrs<GatherNDAttrs>())
      .set_native_fn_attr<&InferTypeGatherND>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.index_put */

Expr index_put(Expr data, Expr indices, Expr values, bool accumulate) {
  auto attrs = ffi::make_object<IndexPutAttrs>();
  attrs->accumulate = std::move(accumulate);
  static const Op op = Op::Get("relax.index_put");
  return Call(Type::Missing(), op, {data, indices, values}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.index_put", index_put);
}

Type InferTypeIndexPut(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  const auto* data_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
  const auto* values_ty = GetTypeAs<TensorTypeObj>(call->args[2]);

  auto diag_def = [&](const TensorTypeObj* ty, ffi::String name, ffi::String type_key) {
    if (ty == nullptr) {
      TVM_FFI_VISIT_THROW(TypeError, call)
          << "IndexPut requires the input " << name << " to be a Tensor. However, the given one is "
          << type_key;
    }
  };

  diag_def(data_ty, "data", call->args[0]->ty->GetTypeKey());
  diag_def(values_ty, "values", call->args[2]->ty->GetTypeKey());

  // Handle indices: either a single tensor or a tuple of tensors
  ffi::Array<TensorType> indices_tensors;

  if (const auto* tuple_ty = GetTypeAs<TupleTypeObj>(call->args[1])) {
    // Indices is a tuple of tensors
    for (size_t i = 0; i < tuple_ty->fields.size(); ++i) {
      const auto* tensor_ty = tuple_ty->fields[i].as<TensorTypeObj>();
      if (tensor_ty == nullptr) {
        TVM_FFI_VISIT_THROW(TypeError, call)
            << "IndexPut requires each index in the indices tuple to be a Tensor. "
            << "However, element " << i << " is " << tuple_ty->fields[i]->GetTypeKey();
      }
      indices_tensors.push_back(ffi::GetRef<TensorType>(tensor_ty));
    }
  } else if (const auto* tensor_ty = GetTypeAs<TensorTypeObj>(call->args[1])) {
    // Indices is a single tensor
    indices_tensors.push_back(ffi::GetRef<TensorType>(tensor_ty));
  } else {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "IndexPut requires indices to be a Tensor or a tuple of Tensors. "
        << "However, the given one is " << call->args[1]->ty->GetTypeKey();
  }

  if (data_ty->IsUnknownNdim()) {
    return TensorType(data_ty->dtype, kUnknownNDim, data_ty->vdevice);
  }

  // Validate each index tensor
  // Index tensors can be multi-dimensional for broadcasting
  int max_index_ndim = -1;
  for (size_t i = 0; i < indices_tensors.size(); ++i) {
    const auto& tensor_ty = indices_tensors[i];
    if (!tensor_ty->IsUnknownNdim()) {
      if (tensor_ty->ndim < 1) {
        TVM_FFI_VISIT_THROW(ValueError, call)
            << "IndexPut requires each index tensor to have at least 1 dimension. "
            << "However, index tensor " << i << " has ndim=" << tensor_ty->ndim;
      }
      if (max_index_ndim < tensor_ty->ndim) {
        max_index_ndim = tensor_ty->ndim;
      }
    }
    if (tensor_ty->IsUnknownDtype()) {
      LOG(WARNING) << "Data type of index tensor " << i
                   << " has not been specified. Assume it has an integer type.";
    } else {
      PrimType index_dtype = tensor_ty->dtype.value();
      if (!index_dtype.MatchesCode(DLDataTypeCode::kDLInt) &&
          !index_dtype.MatchesCode(DLDataTypeCode::kDLUInt)) {
        TVM_FFI_VISIT_THROW(TypeError, call)
            << "IndexPut requires each index tensor to have integer dtype. "
            << "However, index tensor " << i << " has dtype=" << tensor_ty->dtype;
      }
    }
  }

  // Validate that index tensor shapes are broadcastable
  if (max_index_ndim > 1) {
    for (size_t i = 0; i < indices_tensors.size(); ++i) {
      const auto& tensor_ty = indices_tensors[i];
      if (!tensor_ty->IsUnknownNdim() && tensor_ty->ndim > 1) {
        // Check that multi-dimensional indices are broadcastable
        const auto* shape = tensor_ty->shape.as<ShapeExprObj>();
        if (shape) {
          // Verify trailing dimensions can broadcast
          // For now, we accept any multi-dimensional index and rely on runtime validation
          LOG(INFO) << "IndexPut: index tensor " << i << " has ndim=" << tensor_ty->ndim
                    << " for broadcasting";
        }
      }
    }
  }

  // Check that the number of index tensors matches data dimensions
  if (!data_ty->IsUnknownNdim() && indices_tensors.size() != static_cast<size_t>(data_ty->ndim)) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "IndexPut requires the number of index tensors (" << indices_tensors.size()
        << ") to match the data tensor dimensions (" << data_ty->ndim << ")";
  }

  // Check data and values dtype compatibility
  if (data_ty->IsUnknownDtype() || values_ty->IsUnknownDtype()) {
    auto diag_dtype = [&](const TensorTypeObj* ty, ffi::String name) {
      if (ty->IsUnknownDtype()) {
        LOG(WARNING) << "Data type of " << name
                     << " has not been specified. Assume it has an integer type.";
      }
    };
    diag_dtype(data_ty, "data");
    diag_dtype(values_ty, "values");
  } else if (data_ty->dtype != values_ty->dtype) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "IndexPut requires the input data to have the same type as values. "
        << "However, the given types are data: " << data_ty->dtype
        << ", values: " << values_ty->dtype;
  }

  // Check values shape compatibility
  const auto* values_shape = values_ty->shape.as<ShapeExprObj>();
  if (values_shape) {
    if (values_ty->ndim != 1) {
      LOG(WARNING) << "IndexPut typically expects values to be 1D, but got ndim="
                   << values_ty->ndim;
    }
  }

  const auto* data_shape = data_ty->shape.as<ShapeExprObj>();
  if (data_shape) {
    return TensorType(ShapeExpr(data_shape->values), data_ty->dtype, data_ty->vdevice);
  }
  return TensorType(data_ty->dtype, data_ty->ndim, data_ty->vdevice);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.index_put")
      .signature(sig::arg("data", "The input tensor."),
                 sig::arg("indices", "The indices tensor(s)."),
                 sig::arg("values", "The values to put."), sig::call_attrs<IndexPutAttrs>())
      .set_native_fn_attr<&InferTypeIndexPut>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.meshgrid */

Expr meshgrid(Expr tensors, ffi::Optional<ffi::String> indexing) {
  ffi::ObjectPtr<MeshgridAttrs> attrs = ffi::make_object<MeshgridAttrs>();
  attrs->indexing = indexing;
  static const Op op = Op::Get("relax.meshgrid");
  return Call(Type::Missing(), op, {std::move(tensors)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.meshgrid", meshgrid);
}

Type InferTypeMeshgrid(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  if (call->args.size() != 1) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "meshgrid op expects 1 Tuple input argument.";
  }
  ffi::Array<TensorType> input_ty = GetTensorTypeFromTuple(call, call->args[0]);

  int n_inputs = input_ty.size();

  if (n_inputs == 0) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "meshgrid expects at least one 1D tensor in the input Tuple.";
  }

  std::vector<PrimExpr> lengths;
  ffi::Optional<PrimType> common_dtype = std::nullopt;
  bool shape_unknown = false;
  ffi::Optional<VDevice> vdev = std::nullopt;
  bool vdevice_unknown = false;

  for (int i = 0; i < n_inputs; ++i) {
    const TensorType& ty = input_ty[i];

    if (ty->ndim != 1) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "meshgrid expects each input tensor to be 1D. Got ndim = " << ty->ndim << " at index "
          << i;
    }

    if (ty->IsUnknownDtype()) {
      continue;
    } else if (!common_dtype.has_value()) {
      common_dtype = ty->dtype;
    } else if (ty->dtype != common_dtype) {
      TVM_FFI_VISIT_THROW(TypeError, call)
          << "meshgrid expects all input tensors to have the same dtype. Found " << ty->dtype
          << " and " << common_dtype;
    }

    const auto* shape_expr = ty->shape.as<ShapeExprObj>();
    if (shape_expr && shape_expr->values.size() == 1) {
      lengths.push_back(shape_expr->values[0]);
    } else {
      shape_unknown = true;
    }

    if (!vdevice_unknown) {
      if (ty->vdevice.has_value()) {
        if (!vdev.has_value()) {
          vdev = ty->vdevice.value();
        } else if (ty->vdevice.value() != vdev.value()) {
          vdevice_unknown = true;
        }
      }
    }
  }

  ffi::Array<PrimExpr> out_shape;
  if (!shape_unknown && lengths.size() == static_cast<size_t>(n_inputs)) {
    for (const PrimExpr& dim : lengths) {
      out_shape.push_back(dim);
    }
  }

  ffi::Array<Type> out_fields;
  for (int i = 0; i < n_inputs; ++i) {
    if (!out_shape.empty()) {
      if (!vdevice_unknown) {
        out_fields.push_back(TensorType(ShapeExpr(out_shape), common_dtype, vdev));
      } else {
        out_fields.push_back(TensorType(ShapeExpr(out_shape), common_dtype));
      }
    } else {
      if (!vdevice_unknown) {
        out_fields.push_back(TensorType(common_dtype, n_inputs, vdev));
      } else {
        out_fields.push_back(TensorType(common_dtype, n_inputs));
      }
    }
  }

  return TupleType(out_fields);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.meshgrid")
      .signature(sig::arg("tensors", "The input list of tensors."),
                 sig::call_attrs<MeshgridAttrs>())
      .set_native_fn_attr<&InferTypeMeshgrid>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.scatter_elements */

Expr scatter_elements(Expr data, Expr indices, Expr updates, int axis, ffi::String reduction) {
  auto attrs = ffi::make_object<ScatterElementsAttrs>();
  attrs->axis = std::move(axis);
  attrs->reduction = std::move(reduction);
  static const Op op = Op::Get("relax.scatter_elements");
  return Call(Type::Missing(), op, {data, indices, updates}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.scatter_elements", scatter_elements);
}

Type InferTypeScatterElements(const Call& call, const BlockBuilder& ctx) {
  sym::Analyzer analyzer = ctx->GetAnalyzer();
  const auto* data_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
  const auto* indices_ty = GetTypeAs<TensorTypeObj>(call->args[1]);
  const auto* updates_ty = GetTypeAs<TensorTypeObj>(call->args[2]);

  auto diag_def = [&](const TensorTypeObj* ty, ffi::String name, ffi::String type_key) {
    if (ty == nullptr) {
      TVM_FFI_VISIT_THROW(TypeError, call)
          << "ScatterElements requires the input " << name
          << " to be a Tensor. However, the given one is " << type_key;
    }
  };

  diag_def(data_ty, "data", call->args[0]->ty->GetTypeKey());
  diag_def(indices_ty, "indices", call->args[1]->ty->GetTypeKey());
  diag_def(updates_ty, "updates", call->args[2]->ty->GetTypeKey());

  if (data_ty->IsUnknownNdim()) {
    // When `data` has unknown rank, assume rest of arguments are correct and proceed.
    // If the assumption turns out to be wrong, runtime error will be triggered.
    return TensorType(data_ty->dtype, kUnknownNDim, data_ty->vdevice);
  }

  if (!indices_ty->IsUnknownNdim() && !updates_ty->IsUnknownNdim()) {
    if (data_ty->ndim != indices_ty->ndim) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "ScatterElements op requires the data tensor to have the same rank with "
             "indices tensor. However, the given dimensions are "
          << "indices: " << indices_ty->ndim << ", data: " << data_ty->ndim;
    }

    if (indices_ty->ndim != updates_ty->ndim) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "ScatterElements op requires the indices tensor to have the same rank with "
             "updates tensor. However, the given dimensions are "
          << "indices: " << indices_ty->ndim << ", updates: " << updates_ty->ndim;
    }
  }

  if (data_ty->IsUnknownDtype() || updates_ty->IsUnknownDtype()) {
    auto diag_dtype = [&](const TensorTypeObj* ty, ffi::String name) {
      if (ty->IsUnknownDtype()) {
        LOG(WARNING) << "Data type of " << name
                     << " has not been specified. Assume it has an integer type.";
      }
    };
    diag_dtype(data_ty, "data");
    diag_dtype(data_ty, "updates");
  } else {
    if (data_ty->dtype != updates_ty->dtype) {
      TVM_FFI_VISIT_THROW(TypeError, call)
          << "ScatterElements op requires the input data to have same type with "
             "updates. However, the given types are "
          << "data: " << data_ty->dtype << ", updates: " << updates_ty->dtype;
    }
  }

  if (indices_ty->IsUnknownDtype()) {
    LOG(WARNING) << "Data type of indices has not been specified. Assume it has an integer type.";
  } else {
    PrimType indices_dtype = indices_ty->dtype.value();
    if (!indices_dtype.MatchesCode(DLDataTypeCode::kDLInt) &&
        !indices_dtype.MatchesCode(DLDataTypeCode::kDLUInt)) {
      TVM_FFI_VISIT_THROW(TypeError, call)
          << "ScatterElements op requires the input indices to have integer dtype. However, the "
             "given indices dtype is "
          << indices_ty->dtype;
    }
  }

  const auto* indices_shape = indices_ty->shape.as<ShapeExprObj>();
  const auto* updates_shape = updates_ty->shape.as<ShapeExprObj>();
  if (indices_shape && updates_shape) {
    for (int i = 0; i < indices_ty->ndim; i++) {
      if (analyzer->CanProve(indices_shape->values[i] != updates_shape->values[i])) {
        TVM_FFI_VISIT_THROW(ValueError, call)
            << "ScatterElements op requires the indices tensor to have the same shape with "
               "updates tensor. However, the given shapes are "
            << "indices: " << ShapeExpr(indices_shape->values)
            << ", updates: " << ShapeExpr(updates_shape->values);
      }
    }
  }
  const auto* data_shape = data_ty->shape.as<ShapeExprObj>();
  if (data_shape) {
    return TensorType(ShapeExpr(data_shape->values), data_ty->dtype, data_ty->vdevice);
  }
  return TensorType(data_ty->dtype, data_ty->ndim, data_ty->vdevice);
}

InferLayoutOutput InferLayoutScatterElements(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  TVM_FFI_ICHECK(NoDesiredLayout(call, desired_layouts));
  const auto* attrs = call->attrs.as<ScatterElementsAttrs>();
  TVM_FFI_ICHECK(attrs) << "Invalid Call";

  LayoutDecision data_layout = GetLayoutDecision(var_layout_map, call->args[0]);
  LayoutDecision indices_layout = GetLayoutDecision(var_layout_map, call->args[1]);
  LayoutDecision updates_layout = GetLayoutDecision(var_layout_map, call->args[2]);

  LayoutDecision layout = data_layout;
  if (NLayoutEqual()(indices_layout, updates_layout)) {
    layout = indices_layout;
  }

  if ((layout->layout.has_value() ? layout->layout.value().ndim() : 0) !=
      (layout->layout.has_value() ? layout->layout.value().ndim_primal() : 0)) {
    const auto* tensor_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
    TVM_FFI_ICHECK(tensor_ty != nullptr) << "Invalid Call";
    TVM_FFI_ICHECK(!tensor_ty->IsUnknownNdim()) << "Only support static ndim for now";
    int ndim = tensor_ty->ndim;
    layout = LayoutDecision(InitialLayout(ndim));
  }

  ffi::ObjectPtr<ScatterElementsAttrs> new_attrs = ffi::make_object<ScatterElementsAttrs>(*attrs);
  new_attrs->axis = FindAxis(layout->layout.value(), attrs->axis);
  return InferLayoutOutput({layout, layout, layout}, {layout}, Attrs(new_attrs));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.scatter_elements")
      .signature(sig::arg("data", "The input tensor."), sig::arg("indices", "The indices tensor."),
                 sig::arg("updates", "The input tensor of updates."),
                 sig::call_attrs<ScatterElementsAttrs>())
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeScatterElements)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutScatterElements)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.scatter_nd */

Expr scatter_nd(Expr data, Expr indices, Expr updates, ffi::String reduction) {
  auto attrs = ffi::make_object<ScatterNDAttrs>();
  attrs->reduction = std::move(reduction);
  static const Op op = Op::Get("relax.scatter_nd");
  return Call(Type::Missing(), op, {data, indices, updates}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.scatter_nd", scatter_nd);
}

Type InferTypeScatterND(const Call& call, const BlockBuilder& ctx) {
  // `call->args` contains: [data, indices, updates]
  sym::Analyzer analyzer = ctx->GetAnalyzer();
  TVM_FFI_ICHECK_EQ(call->args.size(), 3);
  const auto* data_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
  const auto* indices_ty = GetTypeAs<TensorTypeObj>(call->args[1]);
  const auto* updates_ty = GetTypeAs<TensorTypeObj>(call->args[2]);

  if (data_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "ScatterND op requires the input data to be a tensor. However, the given type is "
        << call->args[0]->GetTypeKey();
  }
  if (indices_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "ScatterND op requires the input indices to be a tensor. However, the given type is "
        << call->args[1]->GetTypeKey();
  }
  if (updates_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "ScatterND op requires the input updates to be a tensor. However, the given type is "
        << call->args[2]->GetTypeKey();
  }

  if (data_ty->IsUnknownDtype() || updates_ty->IsUnknownDtype()) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "ScatterND op requires the input data and updates to have known dtype. "
           "However, the given types are "
        << "data: " << data_ty->dtype << ", updates: " << updates_ty->dtype;
  }

  if (data_ty->dtype != updates_ty->dtype) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "ScatterND op requires the input data to have same type with updates. "
           "However, the given types are "
        << "data: " << data_ty->dtype << ", updates: " << updates_ty->dtype;
  }

  if (indices_ty->IsUnknownDtype()) {
    LOG(WARNING) << "Data type of indices has not been specified. Assume it has an integer type.";
  } else {
    PrimType indices_dtype = indices_ty->dtype.value();
    if (!indices_dtype.MatchesCode(DLDataTypeCode::kDLInt) &&
        !indices_dtype.MatchesCode(DLDataTypeCode::kDLUInt)) {
      TVM_FFI_VISIT_THROW(TypeError, call)
          << "ScatterND op requires the input indices to have integer dtype. However, "
             "the given indices dtype is "
          << indices_ty->dtype;
    }
  }

  const auto* data_shape = data_ty->shape.as<ShapeExprObj>();
  const auto* indices_shape = indices_ty->shape.as<ShapeExprObj>();
  const auto* updates_shape = updates_ty->shape.as<ShapeExprObj>();

  if (data_shape && indices_shape && updates_shape) {
    const IntImmObj* k_dim = indices_shape->values[indices_ty->ndim - 1].as<IntImmObj>();
    if (!k_dim) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "ScatterND needs a static shape for the last axis of indices, got "
          << indices_shape->values;
    }
    const size_t data_ndim = data_ty->ndim;
    const size_t indices_ndim = indices_ty->ndim;
    const size_t updates_ndim = updates_ty->ndim;
    if (data_ndim + indices_ndim - k_dim->value - 1 != updates_ndim) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "ScatterND op requires the updates tensor to have the rank of "
             "`data tensor + indices tensor - last axis of indices tensor - 1`. "
             "However, the given shapes are "
          << "data: " << ShapeExpr(data_shape->values)
          << ", indices: " << ShapeExpr(indices_shape->values)
          << ", updates: " << ShapeExpr(updates_shape->values);
    }
    if (k_dim->value > static_cast<int>(data_ndim)) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "ScatterND op requires the last axis of indices tensor to be less than "
             "or equal to the rank of data tensor. However, the given shapes are "
          << "data: " << ShapeExpr(data_shape->values)
          << ", indices: " << ShapeExpr(indices_shape->values);
    }
    ffi::Array<PrimExpr> expected_updates_shape;
    for (size_t i = 0; i < indices_ndim - 1; i++) {
      expected_updates_shape.push_back(indices_shape->values[i]);
    }
    for (size_t i = k_dim->value.as<size_t>().value(); i < data_ndim; i++) {
      expected_updates_shape.push_back(data_shape->values[i]);
    }
    auto check_shape = [&](const ffi::Array<PrimExpr>& expected,
                           const ffi::Array<PrimExpr>& actual) {
      if (expected.size() != actual.size()) {
        return false;
      }
      for (size_t i = 0; i < expected.size(); i++) {
        if (!analyzer->CanProve(expected[i] == actual[i])) {
          return false;
        }
      }
      return true;
    };
    if (!check_shape(expected_updates_shape, updates_shape->values)) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "ScatterND op requires the updates tensor to have the shape with constraint: "
          << "`updates.shape = indices.shape[:-1] + data.shape[K:]`, but got "
          << "updates.shape: " << ShapeExpr(updates_shape->values)
          << ", indices.shape: " << ShapeExpr(indices_shape->values)
          << ", data.shape: " << ShapeExpr(data_shape->values);
    }
  }
  if (data_shape) {
    return TensorType(ShapeExpr(data_shape->values), data_ty->dtype, data_ty->vdevice);
  }
  return TensorType(data_ty->dtype, data_ty->ndim, data_ty->vdevice);
}

InferLayoutOutput InferLayoutScatterND(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  TVM_FFI_ICHECK(NoDesiredLayout(call, desired_layouts));

  LayoutDecision data_layout = GetLayoutDecision(var_layout_map, call->args[0]);
  LayoutDecision indices_layout = GetLayoutDecision(var_layout_map, call->args[1]);
  LayoutDecision updates_layout = GetLayoutDecision(var_layout_map, call->args[2]);

  const auto* data_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
  const auto* updates_ty = GetTypeAs<TensorTypeObj>(call->args[2]);
  TVM_FFI_ICHECK(data_ty != nullptr) << "Invalid Call";
  TVM_FFI_ICHECK(updates_ty != nullptr) << "Invalid Call";
  TVM_FFI_ICHECK(!data_ty->IsUnknownNdim()) << "Only support static ndim for now";
  TVM_FFI_ICHECK(!updates_ty->IsUnknownNdim()) << "Only support static ndim for now";

  LayoutDecision layout = data_layout;
  LayoutDecision out_updates_layout = updates_layout;

  // Check if data has a sub-indexed layout
  bool has_sub_indexed_layout =
      (layout->layout.has_value() ? layout->layout.value().ndim() : 0) !=
      (layout->layout.has_value() ? layout->layout.value().ndim_primal() : 0);

  if (has_sub_indexed_layout) {
    // Fall back to initial layouts for both data and updates
    layout = LayoutDecision(InitialLayout(data_ty->ndim));
    out_updates_layout = LayoutDecision(InitialLayout(updates_ty->ndim));
  } else if (data_ty->ndim == updates_ty->ndim) {
    // When data and updates have the same rank, apply the same layout to both
    out_updates_layout = layout;
  } else {
    // Different ranks - fall back to initial layouts for both
    layout = LayoutDecision(InitialLayout(data_ty->ndim));
    out_updates_layout = LayoutDecision(InitialLayout(updates_ty->ndim));
  }

  return InferLayoutOutput({layout, indices_layout, out_updates_layout}, {layout}, call->attrs);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.scatter_nd")
      .signature(sig::arg("data", "The input tensor."), sig::arg("indices", "The indices tensor."),
                 sig::arg("updates", "The input tensor of updates."),
                 sig::call_attrs<ScatterNDAttrs>())
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeScatterND)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutScatterND)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.scatter_nd */

Expr slice_scatter(Expr input, Expr src, int axis, PrimExpr start, PrimExpr end, PrimExpr step) {
  auto attrs = ffi::make_object<SliceScatterAttrs>();
  attrs->axis = std::move(axis);
  static const Op op = Op::Get("relax.slice_scatter");
  return Call(Type::Missing(), op, {input, src, start, end, step}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.slice_scatter", slice_scatter);
}

Type InferTypeSliceScatter(const Call& call, const BlockBuilder& ctx) {
  sym::Analyzer analyzer = ctx->GetAnalyzer();
  const auto* data_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
  const auto* src_ty = GetTypeAs<TensorTypeObj>(call->args[1]);
  auto* attrs = call->attrs.as<SliceScatterAttrs>();

  auto diag_tensor_check = [&](const TensorTypeObj* ty, const Expr& arg_expr, ffi::String name) {
    if (ty == nullptr) {
      TVM_FFI_VISIT_THROW(TypeError, call)
          << "SliceScatter requires the input " << name
          << " to be a Tensor. However, the given one is " << arg_expr->ty->GetTypeKey();
    }
  };

  diag_tensor_check(data_ty, call->args[0], "data");
  diag_tensor_check(src_ty, call->args[1], "src");

  if (data_ty->IsUnknownNdim()) {
    return TensorType(data_ty->dtype, kUnknownNDim, data_ty->vdevice);
  }

  int ndim = data_ty->ndim;
  int raw_axis = attrs->axis;
  if (raw_axis < -ndim || raw_axis >= ndim) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "SliceScatter requires the input axis to be in the range "
        << "[" << -ndim << ", " << ndim - 1 << "]. However, the input axis is " << raw_axis
        << ", while ndim is " << ndim;
  }

  if (!data_ty->IsUnknownNdim() && !src_ty->IsUnknownNdim()) {
    if (data_ty->ndim != src_ty->ndim) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "SliceScatter op requires the data tensor to have the same rank as the "
             "src tensor. However, the given dimensions are "
          << "src: " << src_ty->ndim << ", data: " << data_ty->ndim;
    }
  }

  if (data_ty->IsUnknownDtype() || src_ty->IsUnknownDtype()) {
    auto diag_dtype_warn = [&](const TensorTypeObj* ty, ffi::String name) {
      if (ty->IsUnknownDtype()) {
        LOG(WARNING) << "SliceScatter: Data type of " << name
                     << " has not been specified for call node " << call
                     << ". Assuming it is compatible.";
      }
    };
    diag_dtype_warn(data_ty, "data");
    diag_dtype_warn(src_ty, "src");
  } else {
    if (data_ty->dtype != src_ty->dtype) {
      TVM_FFI_VISIT_THROW(TypeError, call)
          << "SliceScatter op requires the input data to have the same type as "
             "src. However, the given types are "
          << "data: " << data_ty->dtype << ", src: " << src_ty->dtype;
    }
  }

  auto get_prim_expr_from_arg = [&call](const Expr& arg_expr, std::string key) -> PrimExpr {
    auto prim_value = arg_expr.as<PrimExpr>();
    if (!prim_value) {
      TVM_FFI_VISIT_THROW(TypeError, call)
          << "SliceScatter expects the `" << key << "` argument (" << arg_expr
          << ") to be a PrimExpr, but got " << arg_expr->GetTypeKey();
    }
    PrimExpr prim_expr = prim_value.value();
    tvm::PrimType prim_ty = prim_expr.ty();
    if (prim_ty.code() != DLDataTypeCode::kDLInt && prim_ty.code() != DLDataTypeCode::kDLUInt) {
      TVM_FFI_VISIT_THROW(TypeError, call)
          << "SliceScatter expects `" << key << "` (" << prim_expr
          << ") to be an integer PrimExpr, but got dtype " << prim_ty;
    }
    return prim_expr;
  };

  PrimExpr start_val = get_prim_expr_from_arg(call->args[2], "start");
  PrimExpr stop_val = get_prim_expr_from_arg(call->args[3], "end");
  PrimExpr step_val = get_prim_expr_from_arg(call->args[4], "step");

  if (analyzer->CanProve(step_val < 1)) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "SliceScatter op requires the step (" << step_val << ") to be >= 1.";
  }

  if (analyzer->CanProve(stop_val < start_val)) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "SliceScatter op requires start (" << start_val << ") <= end (" << stop_val << ").";
  }

  int axis = NormalizeAxis(call, ctx, ndim, attrs->axis);

  const auto* data_shape_node = data_ty->shape.as<ShapeExprObj>();
  const auto* src_shape_node = src_ty->shape.as<ShapeExprObj>();

  if (data_shape_node && src_shape_node && !src_ty->IsUnknownNdim()) {
    TVM_FFI_ICHECK_EQ(data_shape_node->values.size(), static_cast<size_t>(ndim))
        << "Internal error: data_shape_node rank mismatch with data_ty->ndim for call " << call;
    TVM_FFI_ICHECK_EQ(src_shape_node->values.size(), static_cast<size_t>(src_ty->ndim))
        << "Internal error: src_shape_node rank mismatch with src_ty->ndim for call " << call;

    PrimExpr num_elem = tvm::floordiv((stop_val - start_val + step_val - PrimExpr(1)), step_val);

    for (int i = 0; i < ndim; i++) {
      if (i != axis) {
        if (analyzer->CanProve(data_shape_node->values[i] != src_shape_node->values[i])) {
          TVM_FFI_VISIT_THROW(ValueError, call)
              << "SliceScatter op requires the data tensor to have the same shape as the "
                 "src tensor except at the scatter axis ("
              << axis << "). Mismatch at dimension " << i << ". "
              << "data shape: " << data_ty->GetShape().value()
              << ", src shape: " << src_ty->GetShape().value();
        }
      }
    }

    if (analyzer->CanProve(src_shape_node->values[axis] != num_elem)) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "SliceScatter op requires the src tensor's dimension at scatter axis (" << axis
          << ") to match the number of elements in the slice. "
          << "Actual src dimension at axis " << axis << ": " << src_shape_node->values[axis]
          << ", Expected elements in slice (num_elem): " << num_elem;
    }
  }

  if (data_ty->shape.has_value()) {
    return TensorType(data_ty->shape.value(), data_ty->dtype, data_ty->vdevice);
  }
  return TensorType(data_ty->dtype, data_ty->ndim, data_ty->vdevice);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.slice_scatter")
      .signature(
          sig::arg("input", "The input tensor."), sig::arg("src", "The source tensor to scatter."),
          sig::arg<IntExpr>("start", "The starting index of the slice (inclusive)."),
          sig::arg<IntExpr>("end", "The ending index of the slice (exclusive)."),
          sig::arg<IntExpr>("step", "The step of the slice."), sig::call_attrs<SliceScatterAttrs>())
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeSliceScatter)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.one_hot */

Expr one_hot(Expr indices, PrimExpr on_value, PrimExpr off_value, int depth, int axis) {
  ffi::ObjectPtr<OneHotAttrs> attrs = ffi::make_object<OneHotAttrs>();
  attrs->depth = depth;
  attrs->axis = axis;

  // Check if on_value and off_value have the same dtype
  PrimType on_dtype = on_value.ty();
  PrimType off_dtype = off_value.ty();
  TVM_FFI_ICHECK(on_dtype == off_dtype)
      << "one_hot: on_value and off_value must have the same dtype, "
      << "but got " << on_dtype << " and " << off_dtype;

  TVM_FFI_ICHECK(depth > 0) << "one_hot: depth must be positive, but got " << depth;

  static const Op op = Op::Get("relax.one_hot");
  return Call(Type::Missing(), op, {indices, on_value, off_value}, Attrs(attrs), {});
}  // namespace relax

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.one_hot", one_hot);
}

Type InferTypeOneHot(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  TensorType indices_ty = GetInputTensorType(call, 0);
  const auto* attrs = call->attrs.as<OneHotAttrs>();
  PrimExpr on_value = call->args[1].as_or_throw<PrimExpr>();
  PrimExpr off_value = call->args[2].as_or_throw<PrimExpr>();
  // Check if on_value and off_value have the same dtype
  PrimType on_dtype = on_value.ty();
  PrimType off_dtype = off_value.ty();
  TVM_FFI_ICHECK(on_dtype == off_dtype)
      << "one_hot: on_value and off_value must have the same dtype, "
      << "but got " << on_dtype << " and " << off_dtype;
  PrimType dtype = on_dtype;

  // Check if indices has an integer dtype
  if (indices_ty->IsUnknownDtype()) {
    LOG(WARNING) << "Data type of indices has not been specified. Assume it has an integer type.";
  } else {
    PrimType indices_dtype = indices_ty->dtype.value();
    if (!indices_dtype.MatchesCode(DLDataTypeCode::kDLInt) &&
        !indices_dtype.MatchesCode(DLDataTypeCode::kDLUInt)) {
      TVM_FFI_VISIT_THROW(TypeError, call)
          << "one_hot op requires the input indices to have integer dtype. However, the "
             "given indices dtype is "
          << indices_ty->dtype;
    }
  }
  // Check if indices has unknown dimension
  if (indices_ty->IsUnknownNdim()) {
    return TensorType(dtype, kUnknownNDim, indices_ty->vdevice);
  }
  // Get the shape of indices
  const auto* indices_shape = indices_ty->shape.as<ShapeExprObj>();
  if (indices_shape == nullptr) {
    return TensorType(dtype, indices_ty->ndim + 1, indices_ty->vdevice);
  }

  ffi::Array<PrimExpr> output_shape = indices_shape->values;
  int axis = attrs->axis;
  if (axis < 0) {
    axis += output_shape.size() + 1;
  }
  TVM_FFI_ICHECK(0 <= axis && axis <= static_cast<int>(output_shape.size()))
      << "one_hot: axis must be in the range of [0, " << output_shape.size() << "], "
      << "but got " << axis;
  output_shape.insert(output_shape.begin() + axis, attrs->depth);

  return TensorType(ShapeExpr(output_shape), dtype, indices_ty->vdevice);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.one_hot")
      .signature(sig::arg("indices", "The indices tensor."),
                 sig::arg("on_value", "The value to fill at specified indices."),
                 sig::arg("off_value", "The value to fill at other indices."),
                 sig::call_attrs<OneHotAttrs>())
      .set_native_fn_attr<&InferTypeOneHot>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

}  // namespace relax
}  // namespace tvm
