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
 * \file infer_amp_utils.h
 * \brief Utility functions to be used in to_mixed_precision pass.
 */

#ifndef TVM_RELAX_TRANSFORM_INFER_AMP_UTILS_H_
#define TVM_RELAX_TRANSFORM_INFER_AMP_UTILS_H_

#include <tvm/relax/expr.h>
#include <tvm/relax/nested_msg.h>
#include <tvm/relax/op/nn.h>
#include <tvm/relax/op_attr.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace relax {

using ffi::DLDataTypeToString;
using ffi::StringToDLDataType;

enum MixedPrecisionPolicyKind : int { kAlways = 0, kFollow = 1, kNever = 2 };

// NType is the message we want to track for vars with nested tensorstructinfo
// which represents the realization decision of the var.
// The string is the name of the dtype decision.
using NType = NestedMsg<ffi::String>;

struct NTypeEqual {
  bool operator()(const NType& a, const NType& b) const {
    auto dtype_equal = [](const ffi::String& a, const ffi::String& b) { return a == b; };
    return Equal(a, b, dtype_equal);
  }
};

// Construct a NType from an Type
NType NTypeFrom(const Type& ty, DLDataType dtype = DLDataType{kDLOpaqueHandle, 0, 0});

// Construct a NType from an Expr
NType NTypeFrom(const Expr& expr, DLDataType dtype = DLDataType{kDLOpaqueHandle, 0, 0});

// Merge two messages, we keep the higher precision type for each leaf tensor
NType NTypeMerge(const NType& a, const NType& b);

// The map that notes the NType message of each var
using VarDTypeMap = std::unordered_map<Var, NType>;

// Call is a call node, out_dtype is the expected output_dtype

// Compatibility names; new callers can deduce the value type from the key.
using FInferMixedPrecision = decltype(op_attr::kInferMixedPrecision)::ValueType;
using TMixedPrecisionPolicy = decltype(op_attr::kMixedPrecisionPolicy)::ValueType;

ffi::Array<ffi::ObjectRef> InferMixedPrecisionFollow(const Call& call, DLDataType out_dtype);

ffi::Array<ffi::ObjectRef> InferMixedPrecisionNever(const Call& call, DLDataType out_dtype);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_TRANSFORM_INFER_AMP_UTILS_H_
