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
 * \file multibox_transform_loc.cc
 * \brief Multibox transform (location decode) for object detection.
 */

#include "multibox_transform_loc.h"

#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/relax/op_attr.h>
#include <tvm/relax/type.h>

#include <utility>

namespace tvm {
namespace relax {

void MultiboxTransformLocAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<MultiboxTransformLocAttrs>()
      .def_ro("clip", &MultiboxTransformLocAttrs::clip,
              "Clip decoded ymin,xmin,ymax,xmax to [0,1].", refl::DefaultValue(false))
      .def_ro("threshold", &MultiboxTransformLocAttrs::threshold,
              "After softmax, zero scores strictly below this value.", refl::DefaultValue(0.0))
      .def_ro("variances", &MultiboxTransformLocAttrs::variances,
              "(x,y,w,h) scales = TFLite 1/x_scale,1/y_scale,1/w_scale,1/h_scale on "
              "encodings. Very large w/h scales can overflow exp in decode.",
              refl::DefaultValue(ffi::Array<double>{1.0, 1.0, 1.0, 1.0}))
      .def_ro("keep_background", &MultiboxTransformLocAttrs::keep_background,
              "If false, force output scores[:,0,:] to 0 (background class).",
              refl::DefaultValue(true))
      .def_ro("apply_softmax", &MultiboxTransformLocAttrs::apply_softmax,
              "Whether to apply softmax to class predictions before thresholding.",
              refl::DefaultValue(true));
}

TVM_FFI_STATIC_INIT_BLOCK() { MultiboxTransformLocAttrs::RegisterReflection(); }

Expr multibox_transform_loc(Expr cls_pred, Expr loc_pred, Expr anchor, bool clip, double threshold,
                            ffi::Array<double> variances, bool keep_background,
                            bool apply_softmax) {
  TVM_FFI_ICHECK_EQ(variances.size(), 4)
      << "multibox_transform_loc: variances must be length 4 (x,y,w,h), got " << variances.size();

  auto attrs = ffi::make_object<MultiboxTransformLocAttrs>();
  attrs->clip = clip;
  attrs->threshold = threshold;
  attrs->variances = std::move(variances);
  attrs->keep_background = keep_background;
  attrs->apply_softmax = apply_softmax;

  static const Op op = Op::Get("relax.vision.multibox_transform_loc");
  return Call(Type::Missing(), op, {std::move(cls_pred), std::move(loc_pred), std::move(anchor)},
              Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.vision.multibox_transform_loc", multibox_transform_loc);
}

/*!
 * \brief Infer type for relax.vision.multibox_transform_loc.
 *
 * \note Shape cross-checks that need the anchor count N (e.g. loc_pred.shape[1] == 4*N,
 * anchor.shape[1] == N with N = cls_pred.shape[2]) run only when cls_pred has a known
 * static shape. If cls_pred shape is unknown, inference returns generic rank-3 outputs and
 * skips those N-based relations; other checks (ndim, dtype, loc dim divisible by 4, etc.)
 * still apply when their inputs are known.
 */
Type InferTypeMultiboxTransformLoc(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  if (RequiresTensorInputNormalization(call)) return Type::Missing();
  if (call->args.size() != 3) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "multibox_transform_loc: expected 3 inputs (cls_pred, loc_pred, anchor), "
           "got "
        << call->args.size();
  }

  ffi::Array<TensorType> input_ty = GetInputTensorType(call);
  const auto cls_ty = input_ty[0];
  const auto loc_ty = input_ty[1];
  const auto anchor_ty = input_ty[2];

  if (!cls_ty->IsUnknownNdim() && cls_ty->ndim != 3) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "multibox_transform_loc: cls_pred must be 3-D [B, num_classes, N], got "
           "ndim "
        << cls_ty->ndim;
  }
  if (!loc_ty->IsUnknownNdim() && loc_ty->ndim != 2) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "multibox_transform_loc: loc_pred must be 2-D [B, 4*N], got ndim " << loc_ty->ndim;
  }
  if (!anchor_ty->IsUnknownNdim() && anchor_ty->ndim != 3) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "multibox_transform_loc: anchor must be 3-D [1, N, 4] ltrb, got ndim "
        << anchor_ty->ndim;
  }

  if (!cls_ty->IsUnknownDtype() && !loc_ty->IsUnknownDtype() && cls_ty->dtype != loc_ty->dtype) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "multibox_transform_loc: cls_pred and loc_pred dtype must match, got " << cls_ty->dtype
        << " vs " << loc_ty->dtype;
  }
  if (!cls_ty->IsUnknownDtype() && !anchor_ty->IsUnknownDtype() &&
      cls_ty->dtype != anchor_ty->dtype) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "multibox_transform_loc: cls_pred and anchor dtype must match, got " << cls_ty->dtype
        << " vs " << anchor_ty->dtype;
  }

  auto vdev = cls_ty->vdevice;
  const auto* cls_shape = cls_ty->shape.as<ShapeExprObj>();
  const auto* loc_shape = loc_ty->shape.as<ShapeExprObj>();
  const auto* anchor_shape = anchor_ty->shape.as<ShapeExprObj>();

  if (loc_shape != nullptr) {
    const auto* loc_dim1 = loc_shape->values[1].as<IntImmObj>();
    if (loc_dim1 != nullptr && loc_dim1->value % 4 != 0) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "multibox_transform_loc: loc_pred.shape[1] must be divisible by 4, got "
          << loc_dim1->value;
    }
  }

  if (cls_shape != nullptr && loc_shape != nullptr) {
    const auto* cls_b = cls_shape->values[0].as<IntImmObj>();
    const auto* loc_b = loc_shape->values[0].as<IntImmObj>();
    if (cls_b != nullptr && loc_b != nullptr && cls_b->value != loc_b->value) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "multibox_transform_loc: cls_pred.shape[0] must match loc_pred.shape[0], "
             "got B="
          << cls_b->value << " vs " << loc_b->value;
    }
  }

  if (anchor_shape != nullptr) {
    const auto* anchor_batch = anchor_shape->values[0].as<IntImmObj>();
    if (anchor_batch != nullptr && anchor_batch->value != 1) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "multibox_transform_loc: anchor.shape[0] must be 1, got " << anchor_batch->value;
    }
    const auto* anchor_last = anchor_shape->values[2].as<IntImmObj>();
    if (anchor_last != nullptr && anchor_last->value != 4) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "multibox_transform_loc: anchor.shape[2] must be 4 (ltrb), got " << anchor_last->value;
    }
  }

  if (cls_shape == nullptr) {
    ffi::Array<Type> fields = {TensorType(cls_ty->dtype, 3, vdev),
                               TensorType(cls_ty->dtype, 3, vdev)};
    return TupleType(fields);
  }

  const auto& batch = cls_shape->values[0];
  const auto& num_classes = cls_shape->values[1];
  const auto& num_anchors = cls_shape->values[2];

  if (loc_shape != nullptr) {
    const auto* num_anchors_imm = num_anchors.as<IntImmObj>();
    const auto* loc_dim1 = loc_shape->values[1].as<IntImmObj>();
    if (num_anchors_imm != nullptr && loc_dim1 != nullptr &&
        loc_dim1->value != num_anchors_imm->value * 4) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "multibox_transform_loc: loc_pred.shape[1] must equal 4*N with "
             "N=cls_pred.shape[2], got loc_dim="
          << loc_dim1->value << ", N=" << num_anchors_imm->value;
    }
  }
  if (anchor_shape != nullptr) {
    const auto* num_anchors_imm = num_anchors.as<IntImmObj>();
    const auto* anchor_num_anchors = anchor_shape->values[1].as<IntImmObj>();
    if (num_anchors_imm != nullptr && anchor_num_anchors != nullptr &&
        anchor_num_anchors->value != num_anchors_imm->value) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "multibox_transform_loc: anchor.shape[1] must equal N=cls_pred.shape[2], "
             "got anchor_N="
          << anchor_num_anchors->value << ", N=" << num_anchors_imm->value;
    }
  }

  ffi::Array<PrimExpr> boxes_shape = {batch, num_anchors, IntImm::Int32(4)};
  ffi::Array<PrimExpr> scores_shape = {batch, num_classes, num_anchors};
  ffi::Array<Type> fields = {TensorType(ShapeExpr(boxes_shape), cls_ty->dtype, vdev),
                             TensorType(ShapeExpr(scores_shape), cls_ty->dtype, vdev)};
  return TupleType(fields);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.vision.multibox_transform_loc",
        "Decode SSD/TFLite-style priors and offsets into boxes and class scores. If "
        "cls_pred shape is unknown, N-based loc/anchor shape checks are skipped in "
        "inference. Very large variances (w,h) can overflow exp in half box sizes.")
      .signature(sig::arg("cls_pred", "[B,C,N] class logits or scores."),
                 sig::arg("loc_pred",
                          "[B,4*N] box encodings (x,y,w,h); TFLite yxhw order remapped to xywh."),
                 sig::arg("anchor", "[1,N,4] priors as ltrb (left,top,right,bottom)."),
                 sig::call_attrs<MultiboxTransformLocAttrs>())
      .set_native_fn_attr<&InferTypeMultiboxTransformLoc>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

}  // namespace relax
}  // namespace tvm
