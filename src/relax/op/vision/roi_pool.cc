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
 * \file roi_pool.cc
 * \brief ROI Pool operators.
 */

#include "roi_pool.h"

#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/relax/op_attr.h>

#include <utility>

namespace tvm {
namespace relax {

void ROIPoolAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<ROIPoolAttrs>()
      .def_ro("pooled_size", &ROIPoolAttrs::pooled_size, "Output size of roi pool.")
      .def_ro("spatial_scale", &ROIPoolAttrs::spatial_scale,
              "Ratio of input feature map height (or width) to raw image height (or width).")
      .def_ro("layout", &ROIPoolAttrs::layout, "Dimension ordering of the input data.",
              refl::DefaultValue(ffi::String("NCHW")));
}

TVM_FFI_STATIC_INIT_BLOCK() { ROIPoolAttrs::RegisterReflection(); }

Expr roi_pool(Expr data, Expr rois, ffi::Array<int64_t> pooled_size, double spatial_scale,
              ffi::String layout) {
  if (pooled_size.size() == 1) {
    pooled_size.push_back(pooled_size[0]);
  }
  TVM_FFI_ICHECK_EQ(pooled_size.size(), 2)
      << "The input pooled_size length is expected to be 2. However, the given pooled_size is "
      << pooled_size;

  auto attrs = ffi::make_object<ROIPoolAttrs>();
  attrs->pooled_size = std::move(pooled_size);
  attrs->spatial_scale = spatial_scale;
  attrs->layout = layout;

  static const Op op = Op::Get("relax.vision.roi_pool");
  return Call(Type::Missing(), op, {std::move(data), std::move(rois)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.vision.roi_pool", roi_pool);
}

Type InferTypeROIPool(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  if (call->args.size() != 2) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "ROIPool expects two arguments, while the given number of arguments is "
        << call->args.size();
  }

  const auto* data_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
  const auto* rois_ty = GetTypeAs<TensorTypeObj>(call->args[1]);
  if (data_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "ROIPool expects the input data to be a Tensor, while the given data is "
        << call->args[0]->GetTypeKey();
  }
  if (rois_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "ROIPool expects the rois to be a Tensor, while the given rois is "
        << call->args[1]->GetTypeKey();
  }
  if (!data_ty->IsUnknownNdim() && data_ty->ndim != 4) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "ROIPool expects the input data to be 4-D, while the given data has ndim "
        << data_ty->ndim;
  }
  if (!rois_ty->IsUnknownNdim() && rois_ty->ndim != 2) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "ROIPool expects the rois tensor to be 2-D, while the given rois has ndim "
        << rois_ty->ndim;
  }

  const auto* attrs = call->attrs.as<ROIPoolAttrs>();
  TVM_FFI_ICHECK(attrs != nullptr) << "Invalid ROIPool attrs";
  if (attrs->layout != "NCHW") {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "ROIPool only supports NCHW layout, but got " << attrs->layout;
  }

  const auto* rois_shape = rois_ty->shape.as<ShapeExprObj>();
  if (rois_shape != nullptr) {
    const auto* last_dim = rois_shape->values[1].as<IntImmObj>();
    if (last_dim != nullptr && last_dim->value != 5) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "ROIPool expects rois to have shape (num_roi, 5), but got last "
             "dimension "
          << last_dim->value;
    }
  }

  if (data_ty->shape.as<ShapeExprObj>() == nullptr || rois_shape == nullptr) {
    return TensorType(data_ty->dtype, 4, data_ty->vdevice);
  }

  ffi::Array<PrimExpr> data_shape = data_ty->shape.as<ShapeExprObj>()->values;
  ffi::Array<PrimExpr> out_shape = {rois_shape->values[0], data_shape[1],
                                    IntImm::Int32(attrs->pooled_size[0]),
                                    IntImm::Int32(attrs->pooled_size[1])};
  return TensorType(ShapeExpr(out_shape), data_ty->dtype, data_ty->vdevice);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.vision.roi_pool")
      .signature(
          sig::arg("data", "The input tensor."),
          sig::arg("rois",
                   "The input rois with shape (num_roi, 5) in [batch_idx, x1, y1, x2, y2] format."),
          sig::call_attrs<ROIPoolAttrs>())
      .set_native_fn_attr<&InferTypeROIPool>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

}  // namespace relax
}  // namespace tvm
