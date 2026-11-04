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
 * \file tensor.cc
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/op.h>
#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>

namespace tvm {
namespace te {

namespace {

const Op& TensorLoadOp() {
  static const Op op = Op::Get("te.tensor_load");
  return op;
}

ffi::Array<PrimExpr> ValidateTensorLoad(const Call& call, Tensor* tensor_out) {
  TVM_FFI_ICHECK(call->op.same_as(TensorLoadOp())) << "Expected a te.tensor_load Call";
  TVM_FFI_ICHECK(!call->args.empty()) << "Tensor-load Call requires a Tensor argument";
  const auto* tensor_node = call->args[0].as<TensorObj>();
  TVM_FFI_ICHECK(tensor_node != nullptr) << "Tensor-load first argument must be a TE Tensor";
  Tensor tensor = ffi::GetRef<Tensor>(tensor_node);
  TVM_FFI_ICHECK_EQ(call->args.size() - 1, tensor->shape.size())
      << "Tensor-load index count must match tensor rank";
  TVM_FFI_ICHECK(call->ty.as<PrimTypeObj>() != nullptr && call->ty == tensor->dtype)
      << "Tensor-load result type must match the tensor element type";

  ffi::Array<PrimExpr> indices;
  indices.reserve(call->args.size() - 1);
  for (size_t i = 1; i < call->args.size(); ++i) {
    auto index = call->args[i].as<PrimExpr>();
    TVM_FFI_ICHECK(index.has_value()) << "Tensor-load indices must have primitive type";
    indices.push_back(index.value());
  }
  if (tensor_out != nullptr) {
    *tensor_out = std::move(tensor);
  }
  return indices;
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> TensorVisit(
    ffi::StructuralVisitorObj*, ffi::AnyView) noexcept {
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> TensorMutate(ffi::StructuralMutatorObj*,
                                                                      ffi::AnyView) noexcept {
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> TensorMaybeInplaceMutate(
    ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  return ffi::Unchanged();
}

}  // namespace

void TensorObj::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TensorObj>()
      .def_ro("shape", &TensorObj::shape)
      .def_ro("dtype", &TensorObj::dtype)
      .def_ro("op", &TensorObj::op)
      .def_ro("value_index", &TensorObj::value_index);
}

IterVar thread_axis(Range dom, std::string tag) {
  return IterVar(dom, PrimVar(tag, dom.defined() ? dom->extent.ty() : PrimType::Int(32)),
                 kThreadIndex, tag);
}

IterVar reduce_axis(Range dom, std::string name) {
  return IterVar(dom, PrimVar(name, dom->extent.ty()), kCommReduce);
}

PrimVar var(std::string name_hint, PrimType t) { return PrimVar(name_hint, t); }

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("te.tensor_load")
      .signature(sig::arg("tensor", "The input tensor."), sig::var_args("args"))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kReadState);
}

// Tensor
inline PrimExpr Tensor::IndexTensor(ffi::Array<PrimExpr> indices,
                                    bool support_negative_indices) const {
  ffi::Array<PrimExpr> shape = (*this)->shape;

  TVM_FFI_ICHECK_EQ(shape.size(), indices.size())
      << "Tensor dimension mismatch in read "
      << "ndim = " << ndim() << ", indices.size=" << indices.size();

  if (support_negative_indices) {
    for (size_t i = 0; i < shape.size(); i++) {
      PrimExpr new_index =
          prim::Select(indices[i] < IntImm(indices[i].ty(), 0), indices[i] + shape[i], indices[i]);
      indices.Set(i, new_index);
    }
  }
  ffi::Array<Expr> args;
  args.reserve(indices.size() + 1);
  args.push_back(*this);
  for (const PrimExpr& index : indices) {
    args.push_back(index);
  }
  return Call((*this)->dtype, TensorLoadOp(), args).as_or_throw<PrimExpr>();
}

PrimExpr Tensor::operator()(ffi::Array<PrimVar> indices) const {
  ffi::Array<PrimExpr> arr =
      indices.Map([](const PrimVar& var) { return static_cast<PrimExpr>(var); });
  return operator()(arr);
}

PrimExpr Tensor::operator()(ffi::Array<PrimExpr> indices) const {
  return IndexTensor(indices, false);
}

PrimExpr Tensor::IndexWithNegativeIndices(ffi::Array<PrimVar> indices) const {
  ffi::Array<PrimExpr> arr =
      indices.Map([](const PrimVar& var) { return static_cast<PrimExpr>(var); });
  return IndexWithNegativeIndices(arr);
}

PrimExpr Tensor::IndexWithNegativeIndices(ffi::Array<PrimExpr> indices) const {
  return IndexTensor(indices, true);
}

ffi::String TensorObj::GetNameHint() const {
  return op.value()->num_outputs() == 1 ? op.value()->name
                                        : (op.value()->name + ".v" + std::to_string(value_index));
}

Tensor Operation::output(size_t i) const {
  return Tensor((*this)->output_shape(i), (*this)->output_dtype(i), *this, static_cast<int>(i));
}

Tensor::Tensor(ffi::Array<PrimExpr> shape, PrimType dtype, ffi::Optional<Operation> op,
               int value_index)
    : OpaqueExpr(ffi::UnsafeInit{}) {
  auto n = ffi::make_object<TensorObj>();
  n->ExprObj::ty = OpaqueType();
  n->shape = std::move(shape);
  n->dtype = dtype;
  n->op = op;
  n->value_index = value_index;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  TensorObj::RegisterReflection();
  refl::TypeAttrDef<TensorObj>()
      .attr(ffi::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&TensorVisit>())
      .attr(ffi::type_attr::kStructuralMutate, ffi::FStructuralMutate::FromNative<&TensorMutate>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&TensorMaybeInplaceMutate>());
}

bool IsTensorLoad(const Expr& expr) {
  const auto* call = expr.as<CallObj>();
  return call != nullptr && call->op.same_as(TensorLoadOp());
}

Tensor GetTensorFromLoad(const Call& call) {
  ValidateTensorLoad(call, nullptr);
  return call->args[0].as_or_throw<Tensor>();
}

ffi::Array<PrimExpr> GetTensorLoadIndices(const Call& call) {
  return ValidateTensorLoad(call, nullptr);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("te.Tensor",
                        [](ffi::Array<PrimExpr> shape, PrimType dtype, ffi::Optional<Operation> op,
                           int value_index) { return Tensor(shape, dtype, op, value_index); });

  // Pattern A (RM): auto-default repr from reflection.

  // Other tensor ops.

  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("te.TensorEqual", &Tensor::operator==)
      .def("te.TensorDType", [](Tensor tensor) -> PrimType { return tensor->dtype; })
      .def("te.TensorLoad",
           [](Tensor tensor, ffi::Array<PrimExpr> indices) { return tensor(indices); })
      .def("te.TensorHash",
           [](Tensor tensor) -> int64_t {
             return static_cast<int64_t>(std::hash<Tensor>()(tensor));
           })
      .def("te.OpGetOutput",
           [](Operation op, int64_t output) { return op.output(static_cast<size_t>(output)); })
      .def_method("te.OpNumOutputs", &OperationObj::num_outputs)
      .def_method("te.OpInputTensors", &OperationObj::InputTensors);
}

}  // namespace te
}  // namespace tvm
