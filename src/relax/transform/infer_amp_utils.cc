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

#include "infer_amp_utils.h"

namespace tvm {
namespace relax {

NType NTypeFrom(const Type& ty, DLDataType dtype) {
  auto fmapleaf = [&](const Type& ty) -> NType {
    const auto* tensor = ty.as<TensorTypeObj>();
    TVM_FFI_ICHECK(tensor) << "Expected TensorType, but got " << ty;
    if (dtype == DLDataType{kDLOpaqueHandle, 0, 0}) {
      if (tensor->IsUnknownDtype()) return NType(ffi::String(""));
      return NType(DLDataTypeToString(tensor->dtype.value()->dtype));
    } else {
      return NType(DLDataTypeToString(dtype));
    }
  };
  return MapToNestedMsg<ffi::String>(ty, fmapleaf);
}

NType NTypeFrom(const Expr& expr, DLDataType dtype) { return NTypeFrom(GetType(expr), dtype); }

NType NTypeMerge(const NType& a, const NType& b) {
  auto fcombine = [&](const ffi::String& a_str, const ffi::String& b_str) -> ffi::String {
    if (a_str == "") {
      return b_str;
    } else if (b_str == "") {
      return a_str;
    }

    DLDataType a = ffi::StringToDLDataType(a_str);
    DLDataType b = ffi::StringToDLDataType(b_str);
    TVM_FFI_ICHECK_EQ(a.code, b.code);
    TVM_FFI_ICHECK_EQ(a.lanes, b.lanes);
    return a.bits > b.bits ? a_str : b_str;
  };
  return CombineNestedMsg<ffi::String>(a, b, fcombine);
}

ffi::Array<ffi::ObjectRef> InferMixedPrecisionFollow(const Call& call, DLDataType out_dtype) {
  return {IntImm::Int32(MixedPrecisionPolicyKind::kFollow), call};
}

ffi::Array<ffi::ObjectRef> InferMixedPrecisionNever(const Call& call, DLDataType out_dtype) {
  return {IntImm::Int32(MixedPrecisionPolicyKind::kNever), call};
}

}  // namespace relax
}  // namespace tvm
