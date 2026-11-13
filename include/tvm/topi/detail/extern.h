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
 * \file detail/extern.h
 * \brief Helpers for using external functions
 */
#ifndef TVM_TOPI_DETAIL_EXTERN_H_
#define TVM_TOPI_DETAIL_EXTERN_H_

#include <tvm/ir/prim/op.h>
#include <tvm/te/operation.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/type.h>

#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace topi {
namespace detail {

using namespace tvm::te;

/*!
 * \brief A function which constructs an Expr representing the invocation of an external
 * function. The function expects two arguments: an array of Buffers holding the input
 * tensor values, and a pre-allocated array of Buffers to be filled with the outputs.
 */
using FExtern = std::function<PrimExpr(ffi::Array<TensorVar>, ffi::Array<TensorVar>)>;

/*!
 * \brief Create tensors representing the result of invoking an external function.
 * This function will create the necessary buffers to hold input and output tensor values.
 *
 * \param out_shapes An array where each element is the shape of the corresponding output tensor.
 * \param out_types An array where each element is the dtype of the corresponding output tensor.
 * \param inputs An array of input Tensors
 * \param fextern A function that constructs an Expr representing the invocation of
 * the external function given the input and output buffers.
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 * \param attrs The additional auxiliary attributes of the operation.
 *
 * \return An array of Tensors representing the outputs of the function invocation. There will
 * be one output Tensor for each element of out_shapes, with dtype equal to the corresponding
 * element of out_types.
 */
inline ffi::Array<Tensor> make_extern(const ffi::Array<ffi::Array<PrimExpr>>& out_shapes,
                                      const std::vector<PrimType>& out_types,
                                      const ffi::Array<Tensor>& inputs, FExtern fextern,
                                      std::string name, std::string tag,
                                      ::tvm::ffi::Map<ffi::String, ffi::Any> attrs) {
  TVM_FFI_ICHECK_EQ(out_shapes.size(), out_types.size())
      << "make_extern: out_shapes and out_types must have equal size";

  ffi::Array<TensorVar> input_placeholders;
  for (auto t : inputs) {
    input_placeholders.push_back(
        tvm::tirx::TensorVar(t->op.value()->name, tvm::tirx::TensorType(t->shape, t->dtype)));
  }
  ffi::Array<TensorVar> output_placeholders;
  for (size_t i = 0; i < out_shapes.size(); ++i) {
    output_placeholders.push_back(
        tvm::tirx::TensorVar(name, tvm::tirx::TensorType(out_shapes[i], out_types[i])));
  }

  auto body = fextern(input_placeholders, output_placeholders);
  auto body_stmt = tvm::Evaluate(body);

  auto op = ExternOp(name, tag, attrs, inputs, input_placeholders, output_placeholders, body_stmt);

  ffi::Array<Tensor> outputs;
  for (size_t i = 0; i < output_placeholders.size(); ++i) {
    outputs.push_back(op.output(i));
  }
  return outputs;
}

/*!
 * \brief This function is used to create a DLTensor structure on the stack to
 * be able to pass a symbolic buffer as arguments to TVM ffi::Function
 *
 * \param buf The buffer to pack
 *
 * \return An expression representing the pack operation
 */
inline Expr pack_buffer(TensorVar buf) {
  using namespace tvm::prim;
  TVM_FFI_ICHECK_GT(buf->shape.size(), 0) << "buf shape must have at least one element";
  Expr shape = Call(PtrType(PrimType::Int(64)), tvm::tirx::stack_make_shape_op(), buf->shape);
  Expr strides;
  if (tvm::tirx::MatchShapeStridePattern(buf.ty()).value().get<1>().size() > 0) {
    strides = Call(PtrType(PrimType::Int(64)), tvm::tirx::stack_make_shape_op(),
                   tvm::tirx::MatchShapeStridePattern(buf.ty()).value().get<1>());
  } else {
    strides = PrimExpr(0);
  }
  ffi::Array<Expr> pack_args{buf->data,
                             shape,
                             strides,
                             IntImm::Int32(static_cast<int64_t>(buf->shape.size())),
                             MakeConst(PrimType(buf->dtype), 0),
                             tvm::tirx::MatchShapeStridePattern(buf.ty()).value().get<2>()};
  return Call(PtrType::VoidPtrType(), tvm::tirx::stack_make_dltensor_op(), pack_args);
}

/*!
 * \brief Construct an Expr representing the invocation of a ffi::Function
 *
 * \param args An array containing the registered name of the ffi::Function followed
 * by the arguments to pass to the ffi::Function when called. The first element of the
 * array must be a constant string expression.
 *
 * \return An expression representing the invocation
 */
inline PrimExpr call_packed(ffi::Array<Expr> args) {
  return Call(PrimType::Int(32), tvm::tirx::call_packed_op(), args).as_or_throw<PrimExpr>();
}

}  // namespace detail
}  // namespace topi
}  // namespace tvm
#endif  // TVM_TOPI_DETAIL_EXTERN_H_
