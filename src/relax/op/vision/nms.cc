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
#include "nms.h"

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/op.h>
#include <tvm/relax/op/vision.h>
#include <tvm/relax/op_attr.h>
#include <tvm/relax/type.h>
#include <tvm/sym/analyzer.h>

#include <utility>
#include <vector>

namespace tvm {
namespace relax {

void AllClassNonMaximumSuppressionAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<AllClassNonMaximumSuppressionAttrs>().def_ro(
      "output_format", &AllClassNonMaximumSuppressionAttrs::output_format,
      "Output format, onnx or tensorflow. Returns outputs in a way that can be easily "
      "consumed by each frontend.",
      refl::DefaultValue(ffi::String("onnx")));
}

void GetValidCountsAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<GetValidCountsAttrs>()
      .def_ro("score_threshold", &GetValidCountsAttrs::score_threshold,
              "Lower limit of score for valid bounding boxes.", refl::DefaultValue(0))
      .def_ro("id_index", &GetValidCountsAttrs::id_index,
              "Index of the class categories, -1 to disable.", refl::DefaultValue(0))
      .def_ro("score_index", &GetValidCountsAttrs::score_index,
              "Index of the scores/confidence of boxes.", refl::DefaultValue(1));
}

void NonMaximumSuppressionAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<NonMaximumSuppressionAttrs>()
      .def_ro("max_output_size", &NonMaximumSuppressionAttrs::max_output_size,
              "Max number of output valid boxes, -1 for no limit.", refl::DefaultValue(-1))
      .def_ro("iou_threshold", &NonMaximumSuppressionAttrs::iou_threshold,
              "Non-maximum suppression IoU threshold.", refl::DefaultValue(0.5))
      .def_ro("force_suppress", &NonMaximumSuppressionAttrs::force_suppress,
              "Whether to suppress all detections regardless of class_id.",
              refl::DefaultValue(false))
      .def_ro("top_k", &NonMaximumSuppressionAttrs::top_k,
              "Keep maximum top k detections before nms, -1 for no limit.", refl::DefaultValue(-1))
      .def_ro("coord_start", &NonMaximumSuppressionAttrs::coord_start,
              "Start index of the consecutive 4 coordinates.", refl::DefaultValue(2))
      .def_ro("score_index", &NonMaximumSuppressionAttrs::score_index,
              "Index of the scores/confidence of boxes.", refl::DefaultValue(1))
      .def_ro("id_index", &NonMaximumSuppressionAttrs::id_index,
              "Index of the class categories, -1 to disable.", refl::DefaultValue(0))
      .def_ro("return_indices", &NonMaximumSuppressionAttrs::return_indices,
              "Whether to return box indices in input data.", refl::DefaultValue(true))
      .def_ro("invalid_to_bottom", &NonMaximumSuppressionAttrs::invalid_to_bottom,
              "Whether to move all valid bounding boxes to the top.", refl::DefaultValue(false))
      .def_ro("soft_nms_sigma", &NonMaximumSuppressionAttrs::soft_nms_sigma,
              "Sigma for soft-NMS; 0.0 means standard hard NMS.", refl::DefaultValue(0.0))
      .def_ro("score_threshold", &NonMaximumSuppressionAttrs::score_threshold,
              "Score threshold for soft-NMS validity check; 0.0 when unused.",
              refl::DefaultValue(0.0));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  AllClassNonMaximumSuppressionAttrs::RegisterReflection();
  GetValidCountsAttrs::RegisterReflection();
  NonMaximumSuppressionAttrs::RegisterReflection();
}

/* relax.vision.all_class_non_max_suppression */

Expr all_class_non_max_suppression(Expr boxes, Expr scores, Expr max_output_boxes_per_class,
                                   Expr iou_threshold, Expr score_threshold,
                                   ffi::String output_format) {
  auto attrs = tvm::ffi::make_object<AllClassNonMaximumSuppressionAttrs>();
  attrs->output_format = output_format;

  static const Op op = Op::Get("relax.vision.all_class_non_max_suppression");
  return Call(Type::Missing(), op,
              {std::move(boxes), std::move(scores), std::move(max_output_boxes_per_class),
               std::move(iou_threshold), std::move(score_threshold)},
              Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.vision.all_class_non_max_suppression",
                        all_class_non_max_suppression);
}

Type InferTypeAllClassNMS(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  tvm::ffi::Array<TensorType> input_ty = GetInputTensorType(call);
  const auto boxes_ty = input_ty[0];
  const auto scores_ty = input_ty[1];
  TVM_FFI_ICHECK(!boxes_ty->IsUnknownNdim()) << "Only support known ndim";
  TVM_FFI_ICHECK(!scores_ty->IsUnknownNdim()) << "Only support known ndim";
  TVM_FFI_ICHECK_EQ(boxes_ty->ndim, 3) << "AllClassNMS input boxes should be 3-D.";
  TVM_FFI_ICHECK_EQ(scores_ty->ndim, 3) << "AllClassNMS input scores count should be 3-D.";

  const auto batch = boxes_ty->shape.as<ShapeExprObj>()->values[0];
  const auto num_classes = scores_ty->shape.as<ShapeExprObj>()->values[1];
  const auto num_boxes = boxes_ty->shape.as<ShapeExprObj>()->values[1];

  auto vdev = input_ty[0]->vdevice;
  const auto* attrs = call->attrs.as<AllClassNonMaximumSuppressionAttrs>();
  if (attrs->output_format == "onnx") {
    auto vdev = input_ty[0]->vdevice;
    auto num_total_boxes = batch * num_classes * num_boxes;
    tvm::ffi::Array<PrimExpr> oshape_values = {num_total_boxes, 3};
    ShapeExpr oshape(oshape_values);
    tvm::ffi::Array<PrimExpr> counts_values = {1};
    ShapeExpr counts_shape(counts_values);
    tvm::ffi::Array<Type> fields = {TensorType(oshape, PrimType::Int(64), vdev),
                                    TensorType(counts_shape, PrimType::Int(64), vdev)};
    return TupleType(fields);
  }

  auto num_total_boxes_per_batch = num_classes * num_boxes;
  tvm::ffi::Array<PrimExpr> indices_values = {batch, num_total_boxes_per_batch, 2};
  ShapeExpr indices_shape(indices_values);
  tvm::ffi::Array<PrimExpr> scores_values = {batch, num_total_boxes_per_batch};
  ShapeExpr scores_shape(scores_values);
  tvm::ffi::Array<PrimExpr> counts_values = {batch};
  ShapeExpr counts_shape(counts_values);
  tvm::ffi::Array<Type> fields = {TensorType(indices_shape, PrimType::Int(64), vdev),
                                  TensorType(scores_shape, PrimType::Float(32), vdev),
                                  TensorType(counts_shape, PrimType::Int(64), vdev)};
  return TupleType(fields);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.vision.all_class_non_max_suppression")
      .signature(
          sig::arg("boxes", "The input boxes in the format [batch, num_boxes, 4]."),
          sig::arg("scores",
                   "Scores for each box and class in the format [batch, num_classes, num_boxes]."),
          sig::arg("max_output_boxes_per_class", "The maximum number of output boxes per class."),
          sig::arg("iou_threshold", "The IoU threshold for box the overlap test."),
          sig::arg("score_threshold", "The score threshold to filter out low score boxes early."),
          sig::call_attrs<AllClassNonMaximumSuppressionAttrs>())
      .set_native_fn_attr<&InferTypeAllClassNMS>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.vision.get_valid_counts */

Expr get_valid_counts(Expr data, double score_threshold, int id_index, int score_index) {
  auto attrs = tvm::ffi::make_object<GetValidCountsAttrs>();
  attrs->score_threshold = score_threshold;
  attrs->id_index = id_index;
  attrs->score_index = score_index;

  static const Op op = Op::Get("relax.vision.get_valid_counts");
  return Call(Type::Missing(), op, {std::move(data)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.vision.get_valid_counts", get_valid_counts);
}

Type InferTypeGetValidCounts(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  if (RequiresTensorInputNormalization(call)) return Type::Missing();
  if (call->args.size() != 1) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "get_valid_counts expects 1 argument, got " << call->args.size();
  }

  const auto* data_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
  if (data_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call) << "get_valid_counts expects input data to be a Tensor.";
  }
  if (data_ty->ndim != -1 && data_ty->ndim != 3) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "get_valid_counts expects 3-D input, got ndim " << data_ty->ndim;
  }

  const auto* attrs = call->attrs.as<GetValidCountsAttrs>();
  TVM_FFI_ICHECK(attrs != nullptr) << "Invalid get_valid_counts attrs";
  auto vdev = data_ty->vdevice;
  const auto* data_shape = data_ty->shape.as<ShapeExprObj>();
  if (data_shape == nullptr) {
    tvm::ffi::Array<Type> fields = {TensorType(PrimType::Int(32), /*ndim=*/1, vdev),
                                    TensorType(data_ty->dtype, /*ndim=*/3, vdev),
                                    TensorType(PrimType::Int(32), /*ndim=*/2, vdev)};
    return TupleType(fields);
  }

  auto batch = data_shape->values[0];
  auto num_anchors = data_shape->values[1];
  auto elem_length = data_shape->values[2];
  const auto* elem_length_imm = elem_length.as<IntImmObj>();
  if (elem_length_imm != nullptr) {
    if (attrs->score_index < 0 || attrs->score_index >= elem_length_imm->value) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "get_valid_counts expects score_index to be in range [0, " << elem_length_imm->value
          << "), but got " << attrs->score_index;
    }
    if (attrs->id_index < -1 || attrs->id_index >= elem_length_imm->value) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "get_valid_counts expects id_index to be in range [-1, " << elem_length_imm->value
          << "), but got " << attrs->id_index;
    }
  }

  tvm::ffi::Array<Type> fields = {
      TensorType(ShapeExpr({batch}), PrimType::Int(32), vdev),
      TensorType(ShapeExpr({batch, num_anchors, elem_length}), data_ty->dtype, vdev),
      TensorType(ShapeExpr({batch, num_anchors}), PrimType::Int(32), vdev)};
  return TupleType(fields);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.vision.get_valid_counts")
      .signature(sig::arg("data", "Input data, 3-D tensor [batch_size, num_anchors, elem_length]."),
                 sig::call_attrs<GetValidCountsAttrs>())
      .set_native_fn_attr<&InferTypeGetValidCounts>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.vision.non_max_suppression */

Expr non_max_suppression(Expr data, Expr valid_count, Expr indices, int max_output_size,
                         double iou_threshold, bool force_suppress, int top_k, int coord_start,
                         int score_index, int id_index, bool return_indices, bool invalid_to_bottom,
                         double soft_nms_sigma, double score_threshold) {
  auto attrs = tvm::ffi::make_object<NonMaximumSuppressionAttrs>();
  attrs->max_output_size = max_output_size;
  attrs->iou_threshold = iou_threshold;
  attrs->force_suppress = force_suppress;
  attrs->top_k = top_k;
  attrs->coord_start = coord_start;
  attrs->score_index = score_index;
  attrs->id_index = id_index;
  attrs->return_indices = return_indices;
  attrs->invalid_to_bottom = invalid_to_bottom;
  attrs->soft_nms_sigma = soft_nms_sigma;
  attrs->score_threshold = score_threshold;

  static const Op op = Op::Get("relax.vision.non_max_suppression");
  return Call(Type::Missing(), op, {std::move(data), std::move(valid_count), std::move(indices)},
              Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.vision.non_max_suppression", non_max_suppression);
}

Type InferTypeNMS(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 3) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "non_max_suppression expects 3 arguments, got " << call->args.size();
  }

  const auto* data_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
  const auto* valid_count_ty = GetTypeAs<TensorTypeObj>(call->args[1]);
  const auto* indices_ty = GetTypeAs<TensorTypeObj>(call->args[2]);
  if (data_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "non_max_suppression expects input data to be a Tensor.";
  }
  if (valid_count_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "non_max_suppression expects valid_count to be a Tensor.";
  }
  if (indices_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call) << "non_max_suppression expects indices to be a Tensor.";
  }
  if (data_ty->ndim != -1 && data_ty->ndim != 3) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "non_max_suppression expects 3-D input, got ndim " << data_ty->ndim;
  }
  if (valid_count_ty->ndim != -1 && valid_count_ty->ndim != 1) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "non_max_suppression expects valid_count to be 1-D, got ndim " << valid_count_ty->ndim;
  }
  if (indices_ty->ndim != -1 && indices_ty->ndim != 2) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "non_max_suppression expects indices to be 2-D, got ndim " << indices_ty->ndim;
  }
  if (!valid_count_ty->IsUnknownDtype() && valid_count_ty->dtype != PrimType::Int(32)) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "non_max_suppression expects valid_count to have dtype int32, got "
        << valid_count_ty->dtype;
  }
  if (!indices_ty->IsUnknownDtype() && indices_ty->dtype != PrimType::Int(32)) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "non_max_suppression expects indices to have dtype int32, got " << indices_ty->dtype;
  }

  const auto* data_shape = data_ty->shape.as<ShapeExprObj>();
  const auto* valid_count_shape = valid_count_ty->shape.as<ShapeExprObj>();
  const auto* indices_shape = indices_ty->shape.as<ShapeExprObj>();
  if (data_shape != nullptr) {
    sym::Analyzer analyzer = ctx->GetAnalyzer();
    PrimExpr batch = data_shape->values[0];
    PrimExpr num_anchors = data_shape->values[1];
    if (valid_count_shape != nullptr &&
        !analyzer->CanProveEqual(valid_count_shape->values[0], batch)) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "non_max_suppression expects valid_count to have shape [batch_size]. "
             "However, the given data tensor has batch size `"
          << batch << "` and the given valid_count tensor has shape " << valid_count_ty->shape;
    }
    if (indices_shape != nullptr) {
      if (!analyzer->CanProveEqual(indices_shape->values[0], batch) ||
          !analyzer->CanProveEqual(indices_shape->values[1], num_anchors)) {
        TVM_FFI_VISIT_THROW(ValueError, call)
            << "non_max_suppression expects indices to have shape [batch_size, num_anchors]. "
               "However, the given data tensor has shape "
            << data_ty->shape << " and the given indices tensor has shape " << indices_ty->shape;
      }
    }
  }

  const auto* attrs = call->attrs.as<NonMaximumSuppressionAttrs>();
  TVM_FFI_ICHECK(attrs != nullptr) << "Invalid non_max_suppression attrs";
  auto vdev = data_ty->vdevice;
  if (data_shape != nullptr) {
    const auto* elem_length_imm = data_shape->values[2].as<IntImmObj>();
    if (elem_length_imm != nullptr) {
      const ffi::BigInt& elem_length = elem_length_imm->value;
      if (attrs->score_index < 0 || attrs->score_index >= elem_length) {
        TVM_FFI_VISIT_THROW(ValueError, call)
            << "non_max_suppression expects score_index to be in range [0, " << elem_length
            << "), but got " << attrs->score_index;
      }
      if (attrs->coord_start < 0 || attrs->coord_start + 3 >= elem_length) {
        TVM_FFI_VISIT_THROW(ValueError, call)
            << "non_max_suppression expects coord_start to reference four "
               "consecutive box coordinates within elem_length "
            << elem_length << ", but got " << attrs->coord_start;
      }
      if (attrs->id_index < -1 || attrs->id_index >= elem_length) {
        TVM_FFI_VISIT_THROW(ValueError, call)
            << "non_max_suppression expects id_index to be in range [-1, " << elem_length
            << "), but got " << attrs->id_index;
      }
    }
  }

  if (attrs->return_indices) {
    if (attrs->soft_nms_sigma > 0.0) {
      // Soft-NMS returns (out_data[batch, num_anchors, elem_length],
      //                   box_indices[batch, num_anchors],
      //                   valid_box_count[batch, 1])
      if (data_shape == nullptr) {
        tvm::ffi::Array<Type> fields = {TensorType(data_ty->dtype, /*ndim=*/3, vdev),
                                        TensorType(PrimType::Int(32), /*ndim=*/2, vdev),
                                        TensorType(PrimType::Int(32), /*ndim=*/2, vdev)};
        return TupleType(fields);
      }
      auto batch = data_shape->values[0];
      auto num_anchors = data_shape->values[1];
      tvm::ffi::Array<Type> fields = {
          TensorType(ffi::GetRef<ShapeExpr>(data_shape), data_ty->dtype, vdev),
          TensorType(ShapeExpr({batch, num_anchors}), PrimType::Int(32), vdev),
          TensorType(ShapeExpr({batch, IntImm::Int64(1)}), PrimType::Int(32), vdev)};
      return TupleType(fields);
    }

    // Hard NMS returns (box_indices[batch, num_anchors], valid_box_count[batch, 1])
    if (data_shape == nullptr) {
      tvm::ffi::Array<Type> fields = {TensorType(PrimType::Int(32), /*ndim=*/2, vdev),
                                      TensorType(PrimType::Int(32), /*ndim=*/2, vdev)};
      return TupleType(fields);
    }
    auto batch = data_shape->values[0];
    auto num_anchors = data_shape->values[1];
    tvm::ffi::Array<Type> fields = {
        TensorType(ShapeExpr({batch, num_anchors}), PrimType::Int(32), vdev),
        TensorType(ShapeExpr({batch, IntImm::Int64(1)}), PrimType::Int(32), vdev)};
    return TupleType(fields);
  }

  // Returns modified data tensor with the same shape as input.
  if (const auto* data_shape = data_ty->shape.as<ShapeExprObj>()) {
    return TensorType(ffi::GetRef<ShapeExpr>(data_shape), data_ty->dtype, vdev);
  }
  return TensorType(data_ty->dtype, /*ndim=*/3, vdev);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.vision.non_max_suppression")
      .signature(sig::arg("data", "Input data, 3-D tensor [batch_size, num_anchors, elem_length]."),
                 sig::arg("valid_count", "1-D tensor for valid number of boxes."),
                 sig::arg("indices", "2-D tensor with shape [batch_size, num_anchors]."),
                 sig::call_attrs<NonMaximumSuppressionAttrs>())
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeNMS)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

}  // namespace relax
}  // namespace tvm
