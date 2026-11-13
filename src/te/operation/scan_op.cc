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
 * \brief Scan Operator.
 * \file scan_op.cc
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/te/operation.h>

namespace tvm {
namespace te {
using namespace tvm::prim;

using namespace tirx;

TVM_FFI_STATIC_INIT_BLOCK() { ScanOpObj::RegisterReflection(); }

// Pattern A (RM): auto-default repr from reflection.

int ScanOpObj::num_outputs() const { return static_cast<int>(update.size()); }

PrimType ScanOpObj::output_dtype(size_t i) const { return update[i]->GetDataType(); }

ffi::Array<PrimExpr> ScanOpObj::output_shape(size_t i) const {
  TVM_FFI_ICHECK_LT(i, state_placeholder.size());
  return state_placeholder[i]->shape;
}

ScanOp::ScanOp(std::string name, std::string tag,
               ffi::Optional<ffi::Map<ffi::String, ffi::Any>> attrs, IterVar axis,
               ffi::Array<Tensor> init, ffi::Array<Tensor> update,
               ffi::Array<Tensor> state_placeholder, ffi::Array<Tensor> inputs)
    : Operation(ffi::UnsafeInit{}) {
  if (!attrs.has_value()) {
    attrs = ffi::Map<ffi::String, ffi::Any>();
  }
  auto n = ffi::make_object<ScanOpObj>();
  TVM_FFI_ICHECK_EQ(init.size(), update.size());
  TVM_FFI_ICHECK_EQ(init.size(), state_placeholder.size());
  sym::Analyzer analyzer;
  auto prove_equal = [&](PrimExpr lhs, PrimExpr rhs) {
    return IsZero(analyzer->Simplify(lhs - rhs));
  };

  for (size_t i = 0; i < init.size(); ++i) {
    TVM_FFI_ICHECK_EQ(init[i]->dtype, state_placeholder[i]->dtype);
    TVM_FFI_ICHECK_EQ(init[i]->dtype, update[i]->dtype);
    TVM_FFI_ICHECK(prove_equal(init[i]->shape[0], axis->dom.value()->begin))
        << "init.shape[0] need to match scan_axis.dom.begin";
    TVM_FFI_ICHECK(prove_equal(state_placeholder[i]->shape[0],
                               axis->dom.value()->begin + axis->dom.value()->extent))
        << "state_placeholder.shape[0] need to match"
        << " scan_axis.dom.begin + scan_axis.dom.extent";
    TVM_FFI_ICHECK_EQ(state_placeholder[i].ndim(), init[i].ndim())
        << "The dimension of init need to match state_placeholder";
    TVM_FFI_ICHECK_EQ(update[i].ndim(), state_placeholder[i].ndim())
        << "The update.ndim need to be state_placeholder.ndim - 1";
    for (size_t k = 0; k < update[i].ndim(); ++k) {
      TVM_FFI_ICHECK(prove_equal(update[i]->shape[k], state_placeholder[i]->shape[k]));
      if (k != 0) {
        // setup spatial axis
        std::ostringstream spatial_name;
        spatial_name << name << ".out" << i << ".i" << k;
        n->spatial_axis_.push_back(IterVar(Range::FromMinExtent(0, update[i]->shape[k]),
                                           PrimVar(spatial_name.str()), kOpaque));
      }
    }

    for (size_t k = 1; k < init[i].ndim(); ++k) {
      TVM_FFI_ICHECK(prove_equal(init[i]->shape[k], state_placeholder[i]->shape[k]));
    }
  }
  n->name = std::move(name);
  n->tag = std::move(tag);
  n->attrs = attrs.value_or({});
  n->scan_axis = std::move(axis);
  n->init = std::move(init);
  n->update = std::move(update);
  n->state_placeholder = std::move(state_placeholder);
  n->inputs = std::move(inputs);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "te.ScanOp",
      [](std::string name, std::string tag, ffi::Optional<ffi::Map<ffi::String, ffi::Any>> attrs,
         IterVar axis, ffi::Array<Tensor> init, ffi::Array<Tensor> update,
         ffi::Array<Tensor> state_placeholder, ffi::Array<Tensor> inputs) {
        return ScanOp(name, tag, attrs, axis, init, update, state_placeholder, inputs);
      });
}

ffi::Array<Tensor> scan(ffi::Array<Tensor> init, ffi::Array<Tensor> update,
                        ffi::Array<Tensor> state_placeholder, ffi::Array<Tensor> inputs,
                        std::string name, std::string tag,
                        ffi::Optional<ffi::Map<ffi::String, ffi::Any>> attrs) {
  IterVar scan_axis =
      IterVar(Range::FromMinExtent(init[0]->shape[0], update[0]->shape[0] - init[0]->shape[0]),
              PrimVar(name + ".idx"), kOrdered);
  Operation op = ScanOp(name, tag, attrs, scan_axis, init, update, state_placeholder, inputs);
  ffi::Array<Tensor> res;
  for (int i = 0; i < op->num_outputs(); ++i) {
    res.push_back(op.output(i));
  }
  return res;
}

ffi::Array<Tensor> ScanOpObj::InputTensors() const {
  ffi::Array<Tensor> ret;
  for (Tensor t : init) {
    ret.push_back(t);
  }
  for (Tensor t : update) {
    ret.push_back(t);
  }
  return ret;
}

}  // namespace te
}  // namespace tvm
