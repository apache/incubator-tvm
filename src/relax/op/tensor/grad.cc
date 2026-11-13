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
 * \file grad.cc
 * \brief Operators to implement operaor gradients.
 */

#include "grad.h"

#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/relax/op_attr.h>

#include <utility>

namespace tvm {
namespace relax {

/* relax.grad.no_grad */
Expr no_grad(Expr input) {
  static const Op op = Op::Get("relax.grad.no_grad");
  return Call(Type::Missing(), op, {std::move(input)}, {}, {});
}

void TakeBackwardAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TakeBackwardAttrs>().def_ro("axis", &TakeBackwardAttrs::axis,
                                              "The axis over which values were selected.",
                                              refl::DefaultValue(ffi::Optional<int64_t>{}));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.grad.no_grad", no_grad);
}

Type InferTypeNoGrad(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  return GetType(call->args[0]);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.grad.no_grad")
      .signature(sig::arg("x", "The corresponding input tensor."))
      .set_native_fn_attr<&InferTypeNoGrad>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.grad.start_checkpoint */
Expr start_checkpoint(Expr input) {
  static const Op op = Op::Get("relax.grad.start_checkpoint");
  return Call(Type::Missing(), op, {std::move(input)}, {}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.grad.start_checkpoint", start_checkpoint);
}

Type InferTypeStartCheckpoint(const Call& call, const BlockBuilder& ctx) {
  if (!call->args[0].as<VarObj>()) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "The argument of relax.op.grad.start_checkpoint should be a Var.";
  }
  return GetType(call->args[0]);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.grad.start_checkpoint")
      .signature(sig::arg("x", "The tensor marking the input of the checkpoint stage."))
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeStartCheckpoint)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.grad.end_checkpoint */
Expr end_checkpoint(Expr input) {
  static const Op op = Op::Get("relax.grad.end_checkpoint");
  return Call(Type::Missing(), op, {std::move(input)}, {}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.grad.end_checkpoint", end_checkpoint);
}

Type InferTypeEndCheckpoint(const Call& call, const BlockBuilder& ctx) {
  if (!call->args[0].as<VarObj>()) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "The argument of relax.op.grad.end_checkpoint should be a Var.";
  }
  return GetType(call->args[0]);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.grad.end_checkpoint")
      .signature(sig::arg("x", "The output of the checkpoint stage."))
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeEndCheckpoint)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.grad.nll_loss_backward */
Expr nll_loss_backward(Expr output_grad, Expr predictions, Expr targets,
                       ffi::Optional<Expr> weights, ffi::String reduction, int ignore_index) {
  ffi::ObjectPtr<NLLLossAttrs> attrs = ffi::make_object<NLLLossAttrs>();

  attrs->reduction = reduction;
  attrs->ignore_index = ignore_index;

  static const Op op = Op::Get("relax.grad.nll_loss_backward");
  if (weights.has_value()) {
    return Call(
        Type::Missing(), op,
        {std::move(output_grad), std::move(predictions), std::move(targets), weights.value()},
        Attrs{attrs}, {});
  } else {
    return Call(Type::Missing(), op,
                {std::move(output_grad), std::move(predictions), std::move(targets)}, Attrs{attrs},
                {});
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.grad.nll_loss_backward", nll_loss_backward);
}

Type InferTypeNLLLossBackward(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  return GetType(call->args[1]);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.grad.nll_loss_backward", "Optional weights: The weight of each target values.")
      .signature(sig::arg("output_grad", "The output gradient."),
                 sig::arg("predictions", "The prediction tensor."),
                 sig::arg("targets", "The target tensor."), sig::var_args("args"),
                 sig::call_attrs<NLLLossAttrs>())
      .set_native_fn_attr<&InferTypeNLLLossBackward>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.grad.max_pool2d_backward */
Expr max_pool2d_backward(Expr output_grad, Expr data, ffi::Array<int64_t> pool_size,
                         ffi::Array<int64_t> strides, ffi::Array<int64_t> padding,
                         ffi::Array<int64_t> dilation, bool ceil_mode, bool count_include_pad,
                         ffi::String layout, ffi::Optional<ffi::String> out_layout) {
  auto attrs = ffi::make_object<Pool2DAttrs>();
  attrs->pool_size = std::move(pool_size);
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->dilation = std::move(dilation);
  attrs->ceil_mode = ceil_mode;
  attrs->count_include_pad = count_include_pad;
  attrs->layout = layout;
  attrs->out_layout = out_layout.value_or(layout);
  static const Op op = Op::Get("relax.grad.max_pool2d_backward");
  return Call(Type::Missing(), op, {std::move(output_grad), std::move(data)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.grad.max_pool2d_backward", max_pool2d_backward);
}

Type InferTypeMaxPool2DBackward(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  return GetType(call->args[1]);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.grad.max_pool2d_backward")
      .signature(sig::arg("output_grad", "The output gradient."),
                 sig::arg("data", "The input tensor"), sig::call_attrs<Pool2DAttrs>())
      .set_native_fn_attr<&InferTypeMaxPool2DBackward>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.grad.avg_pool2d_backward */
Expr avg_pool2d_backward(Expr output_grad, Expr data, ffi::Array<int64_t> pool_size,
                         ffi::Array<int64_t> strides, ffi::Array<int64_t> padding,
                         ffi::Array<int64_t> dilation, bool ceil_mode, bool count_include_pad,
                         ffi::String layout, ffi::Optional<ffi::String> out_layout) {
  auto attrs = ffi::make_object<Pool2DAttrs>();
  attrs->pool_size = std::move(pool_size);
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->dilation = std::move(dilation);
  attrs->ceil_mode = ceil_mode;
  attrs->count_include_pad = count_include_pad;
  attrs->layout = layout;
  attrs->out_layout = out_layout.value_or(layout);
  static const Op op = Op::Get("relax.grad.avg_pool2d_backward");
  return Call(Type::Missing(), op, {std::move(output_grad), std::move(data)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.grad.avg_pool2d_backward", avg_pool2d_backward);
}

Type InferTypeAvgPool2DBackward(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  return GetType(call->args[1]);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.grad.avg_pool2d_backward")
      .signature(sig::arg("output_grad", "The output gradient."),
                 sig::arg("data", "The input tensor"), sig::call_attrs<Pool2DAttrs>())
      .set_native_fn_attr<&InferTypeAvgPool2DBackward>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.grad.take_backward */

Expr take_backward(Expr output_grad, Expr x, Expr indices, ffi::Optional<int64_t> axis) {
  ffi::ObjectPtr<TakeBackwardAttrs> attrs = ffi::make_object<TakeBackwardAttrs>();
  attrs->axis = std::move(axis);

  static const Op op = Op::Get("relax.grad.take_backward");
  return Call(Type::Missing(), op, {std::move(output_grad), std::move(x), std::move(indices)},
              Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  TakeBackwardAttrs::RegisterReflection();
  refl::GlobalDef().def("relax.op.grad.take_backward", take_backward);
}

Type InferTypeTakeBackward(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  return GetType(call->args[1]);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.grad.take_backward")
      .signature(sig::arg("output_grad", "The output gradient."),
                 sig::arg("x", "The source tensor."),
                 sig::arg("indices", "The indices of the values to extract."),
                 sig::call_attrs<TakeBackwardAttrs>())
      .set_native_fn_attr<&InferTypeTakeBackward>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

}  // namespace relax
}  // namespace tvm
