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
 * \file relax/backend/contrib/utils.h
 * \brief Utils function for backend
 */
#ifndef TVM_RELAX_BACKEND_CONTRIB_UTILS_H_
#define TVM_RELAX_BACKEND_CONTRIB_UTILS_H_

#include <tvm/ffi/cast.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>

#include <string>
#include <vector>

#include "../../transform/utils.h"

namespace tvm {
namespace relax {

namespace backend {

/*!
 * \brief Extract shape from an IndexExpr array to std::vector<int64_t>
 *
 * \param shape The shape in Array
 * \return The converted shape in std::vector<int64_t>
 */

inline std::vector<int64_t> GetIntShape(const ffi::Array<PrimExpr>& shape) {
  std::vector<int64_t> ret;
  for (const auto& dim : shape) {
    const auto* imm = dim.as<IntImmObj>();
    auto pval = imm ? imm->value.as<int64_t>() : std::nullopt;
    ret.push_back(pval.has_value() ? *pval : -1);
  }
  return ret;
}

/*!
 * \brief Convert type to string
 *
 * \param typ
 * \return std::string string format of type
 */
inline std::string DType2String(DLDataType dtype) { return tvm::ffi::DLDataTypeToString(dtype); }

/*!
 * \brief Check if a call node is calling an op with the given name
 * \param call The call node whose callee we want to check
 * \param op_name The name of the op
 * \return true if the callee op matches with the op name
 */
inline bool IsOp(const CallObj* call, const std::string& op_name) {
  const auto* op_node = call->op.as<OpObj>();
  if (!op_node) return false;
  Op op = ffi::GetRef<Op>(op_node);
  return op.same_as(Op::Get(op_name));
}

/*!
 * \brief Return a call node within the function which calls an op with the given name
 * The function must contain exactly one call to such op.
 * \param f The function to look for an op.
 * \param op_name The name of the op
 * \return A call node which calls an op with the given name or nullptr if not
 */
inline const CallObj* TryGetOpInFunction(Function f, const std::string& op_name) {
  auto local_bindings = AnalyzeVar2Value(f);
  for (const auto& entry : local_bindings) {
    if (auto call = entry.second.as<CallObj>(); call && backend::IsOp(call, op_name)) {
      return call;
    }
  }
  return nullptr;
}

/*!
 * \brief Return a call node within the function which calls an op with the given name
 * The function must contain exactly one call to such op.
 * \param f The function to look for an op.
 * \param op_name The name of the op
 * \return A call node which calls an op with the given name
 */
inline const CallObj* GetOpInFunction(Function f, const std::string& op_name) {
  const CallObj* op = TryGetOpInFunction(f, op_name);
  TVM_FFI_ICHECK(op) << op_name << " not found in the function:\n" << f;
  return op;
}

/*!
 * \brief Extract indices of the argument patterns in the function parameter list.
 * Each composite function pattern can register a mapping between variable names and the
 * corresponding patterns. This function tells at which index a given parameter
 * in the function pattern, identified by its name, appears in the partitioned function parameter
 * list.
 * \param pattern_name The name the composite function pattern.
 * \param f The function partitioned according to the function pattern.
 * \return A mapping between variable pattern names and their positions in the partitioned
 * function parameter list.
 */
ffi::Map<ffi::String, IntImm> ExtractArgIdx(ffi::String pattern_name, Function f);

/*!
 * \brief Converts a numeric value to std::string.
 * \param value A numeric value to convert.
 * \return ffi::String representation of a numeric value.
 */
template <typename Type>
std::string to_str(const Type& value) {
  std::ostringstream os;
  os << std::setprecision(12) << value;
  return os.str();
}

/*!
 * \brief Utility function to find the string pattern in string str
 * \param str the main string to check the pattern
 * \param pattern the pattern to check in the main string
 * \return return true if the main string ends with pattern, false otherwise
 */
bool EndsWithPattern(const std::string& str, const std::string& pattern);

}  // namespace backend
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_BACKEND_CONTRIB_UTILS_H_
