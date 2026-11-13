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
 * \brief Registration of elemwise operators
 * \file elemwise.cc
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/topi/elemwise.h>

namespace tvm {
namespace topi {
using namespace tvm::prim;

using namespace tvm;
using namespace tvm::runtime;

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("topi.acos", [](ffi::PackedArgs args,
                                  ffi::Any* rv) { *rv = acos(args[0].cast<te::Tensor>()); })
      .def_packed("topi.acosh", [](ffi::PackedArgs args,
                                   ffi::Any* rv) { *rv = acosh(args[0].cast<te::Tensor>()); })
      .def_packed("topi.asin", [](ffi::PackedArgs args,
                                  ffi::Any* rv) { *rv = asin(args[0].cast<te::Tensor>()); })
      .def_packed("topi.asinh", [](ffi::PackedArgs args,
                                   ffi::Any* rv) { *rv = asinh(args[0].cast<te::Tensor>()); })
      .def_packed("topi.atanh", [](ffi::PackedArgs args,
                                   ffi::Any* rv) { *rv = atanh(args[0].cast<te::Tensor>()); })
      .def_packed("topi.exp",
                  [](ffi::PackedArgs args, ffi::Any* rv) { *rv = exp(args[0].cast<te::Tensor>()); })
      .def_packed("topi.fast_exp", [](ffi::PackedArgs args,
                                      ffi::Any* rv) { *rv = fast_exp(args[0].cast<te::Tensor>()); })
      .def_packed("topi.erf",
                  [](ffi::PackedArgs args, ffi::Any* rv) { *rv = erf(args[0].cast<te::Tensor>()); })
      .def_packed("topi.fast_erf", [](ffi::PackedArgs args,
                                      ffi::Any* rv) { *rv = fast_erf(args[0].cast<te::Tensor>()); })
      .def_packed("topi.tan",
                  [](ffi::PackedArgs args, ffi::Any* rv) { *rv = tan(args[0].cast<te::Tensor>()); })
      .def_packed("topi.cos",
                  [](ffi::PackedArgs args, ffi::Any* rv) { *rv = cos(args[0].cast<te::Tensor>()); })
      .def_packed("topi.cosh", [](ffi::PackedArgs args,
                                  ffi::Any* rv) { *rv = cosh(args[0].cast<te::Tensor>()); })
      .def_packed("topi.sin",
                  [](ffi::PackedArgs args, ffi::Any* rv) { *rv = sin(args[0].cast<te::Tensor>()); })
      .def_packed("topi.sinh", [](ffi::PackedArgs args,
                                  ffi::Any* rv) { *rv = sinh(args[0].cast<te::Tensor>()); })
      .def_packed("topi.tanh", [](ffi::PackedArgs args,
                                  ffi::Any* rv) { *rv = tanh(args[0].cast<te::Tensor>()); })
      .def_packed(
          "topi.fast_tanh",
          [](ffi::PackedArgs args, ffi::Any* rv) { *rv = fast_tanh(args[0].cast<te::Tensor>()); })
      .def_packed("topi.atan", [](ffi::PackedArgs args,
                                  ffi::Any* rv) { *rv = atan(args[0].cast<te::Tensor>()); })
      .def_packed("topi.sigmoid", [](ffi::PackedArgs args,
                                     ffi::Any* rv) { *rv = sigmoid(args[0].cast<te::Tensor>()); })
      .def_packed("topi.sqrt", [](ffi::PackedArgs args,
                                  ffi::Any* rv) { *rv = sqrt(args[0].cast<te::Tensor>()); })
      .def_packed("topi.rsqrt", [](ffi::PackedArgs args,
                                   ffi::Any* rv) { *rv = rsqrt(args[0].cast<te::Tensor>()); })
      .def_packed("topi.log",
                  [](ffi::PackedArgs args, ffi::Any* rv) { *rv = log(args[0].cast<te::Tensor>()); })
      .def_packed("topi.log2", [](ffi::PackedArgs args,
                                  ffi::Any* rv) { *rv = log2(args[0].cast<te::Tensor>()); })
      .def_packed("topi.log10", [](ffi::PackedArgs args,
                                   ffi::Any* rv) { *rv = log10(args[0].cast<te::Tensor>()); })
      .def_packed("topi.identity", [](ffi::PackedArgs args,
                                      ffi::Any* rv) { *rv = identity(args[0].cast<te::Tensor>()); })
      .def_packed("topi.negative", [](ffi::PackedArgs args,
                                      ffi::Any* rv) { *rv = negative(args[0].cast<te::Tensor>()); })
      .def_packed("topi.clip",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    auto a_min_tensor = args[1].try_cast<te::Tensor>();
                    auto a_max_tensor = args[2].try_cast<te::Tensor>();
                    PrimExpr a_min = a_min_tensor ? a_min_tensor.value()(ffi::Array<PrimExpr>{})
                                                  : args[1].cast<PrimExpr>();
                    PrimExpr a_max = a_max_tensor ? a_max_tensor.value()(ffi::Array<PrimExpr>{})
                                                  : args[2].cast<PrimExpr>();
                    *rv = clip(args[0].cast<te::Tensor>(), a_min, a_max);
                  })
      .def_packed("topi.cast",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = cast(args[0].cast<te::Tensor>(), args[1].cast<PrimType>());
                  })
      .def_packed("topi.reinterpret",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = reinterpret(args[0].cast<te::Tensor>(), args[1].cast<PrimType>());
                  })
      .def_packed("topi.elemwise_sum",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = elemwise_sum(args[0].cast<ffi::Array<te::Tensor>>());
                  })
      .def_packed("topi.sign", [](ffi::PackedArgs args,
                                  ffi::Any* rv) { *rv = sign(args[0].cast<te::Tensor>()); })
      .def_packed("topi.full",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    auto fill_value_tensor = args[2].try_cast<te::Tensor>();
                    PrimExpr fill_value = fill_value_tensor
                                              ? fill_value_tensor.value()(ffi::Array<PrimExpr>{})
                                              : args[2].cast<PrimExpr>();
                    *rv = full(args[0].cast<ffi::Array<PrimExpr>>(), args[1].cast<PrimType>(),
                               fill_value);
                  })
      .def_packed("topi.full_like",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    auto fill_value_tensor = args[1].try_cast<te::Tensor>();
                    PrimExpr fill_value = fill_value_tensor
                                              ? fill_value_tensor.value()(ffi::Array<PrimExpr>{})
                                              : args[1].cast<PrimExpr>();
                    *rv = full_like(args[0].cast<te::Tensor>(), fill_value);
                  })
      .def_packed(
          "topi.logical_not",
          [](ffi::PackedArgs args, ffi::Any* rv) { *rv = logical_not(args[0].cast<te::Tensor>()); })
      .def_packed("topi.bitwise_not", [](ffi::PackedArgs args, ffi::Any* rv) {
        *rv = bitwise_not(args[0].cast<te::Tensor>());
      });
}

}  // namespace topi
}  // namespace tvm
