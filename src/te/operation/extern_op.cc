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
 * \brief External computation rule.
 * \file extern_op.cc
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/te/operation.h>
#include <tvm/tirx/type.h>

namespace tvm {
namespace te {
using namespace tirx;

TVM_FFI_STATIC_INIT_BLOCK() { ExternOpObj::RegisterReflection(); }

// Pattern A (RM): auto-default repr from reflection.

int ExternOpObj::num_outputs() const { return static_cast<int>(output_placeholders.size()); }

PrimType ExternOpObj::output_dtype(size_t i) const { return output_placeholders[i].dtype(); }

ffi::Array<PrimExpr> ExternOpObj::output_shape(size_t i) const {
  return output_placeholders[i].shape();
}

ExternOp::ExternOp(std::string name, std::string tag, ffi::Map<ffi::String, ffi::Any> attrs,
                   ffi::Array<Tensor> inputs, ffi::Array<TensorVar> input_placeholders,
                   ffi::Array<TensorVar> output_placeholders, SeqStmt body)
    : Operation(ffi::UnsafeInit{}) {
  if (!attrs.defined()) {
    attrs = ffi::Map<ffi::String, ffi::Any>();
  }
  auto n = ffi::make_object<ExternOpObj>(std::move(body));
  n->name = std::move(name);
  n->tag = std::move(tag);
  n->attrs = std::move(attrs);
  TVM_FFI_ICHECK_EQ(inputs.size(), input_placeholders.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    TVM_FFI_ICHECK_EQ(inputs[i]->dtype, input_placeholders[i].dtype());
    TVM_FFI_ICHECK_EQ(inputs[i]->shape.size(), input_placeholders[i].shape().size());
    for (size_t dim = 0; dim < inputs[i]->shape.size(); ++dim) {
      TVM_FFI_ICHECK(inputs[i]->shape[dim].same_as(input_placeholders[i].shape()[dim]));
    }
    const auto& layout = input_placeholders[i].ty()->layout;
    if (layout.has_value()) {
      const auto* tile = layout.as<tirx::TileLayoutObj>();
      TVM_FFI_ICHECK(tile && ffi::GetRef<tirx::TileLayout>(tile).IsContiguous())
          << "External inputs require contiguous strides";
    }
  }
  n->inputs = std::move(inputs);
  n->input_placeholders = std::move(input_placeholders);
  n->output_placeholders = std::move(output_placeholders);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "te.ExternOp",
      [](std::string name, std::string tag, ffi::Optional<ffi::Map<ffi::String, ffi::Any>> attrs,
         ffi::Array<Tensor> inputs, ffi::Array<TensorVar> input_placeholders,
         ffi::Array<TensorVar> output_placeholders, SeqStmt body) {
        return ExternOp(name, tag, attrs.value_or({}), inputs, input_placeholders,
                        output_placeholders, body);
      });
}

ffi::Array<Tensor> ExternOpObj::InputTensors() const { return inputs; }

}  // namespace te
}  // namespace tvm
