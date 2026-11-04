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
#include "utils.h"

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/dataflow_matcher.h>
#include <tvm/relax/expr.h>

#include <optional>

#include "../pattern_registry.h"

namespace tvm {
namespace relax {
namespace backend {

ffi::Map<ffi::String, IntImm> ExtractArgIdx(ffi::String pattern_name, Function f) {
  ffi::Map<ffi::String, IntImm> arg_idx;
  auto pattern = backend::GetPattern(pattern_name);
  TVM_FFI_ICHECK(pattern) << "Unsupported op_type " << pattern_name;

  auto bindings = AnalyzeVar2Value(f);
  auto inner_body = f->body->body;
  auto matched_expr = relax::ExtractMatchedExpr(pattern.value()->pattern, inner_body, bindings);
  TVM_FFI_CHECK(matched_expr, ValueError)
      << "For named pattern \"" << pattern_name << "\", expected to find a match for "
      << pattern.value()->pattern << ".  However, the function did not include this pattern " << f;

  auto find_index = [](const ffi::Array<Var>& params, Var v) -> std::optional<size_t> {
    for (size_t i = 0; i < params.size(); ++i) {
      if (params[i].same_as(v)) {
        return i;
      }
    }
    return std::nullopt;
  };

  for (const auto& [name, pat] : pattern.value()->annotation_patterns) {
    auto exp = matched_expr.value()[pat];
    if (auto arg_var = exp.as<VarObj>()) {
      if (auto idx = find_index(f->params, ffi::GetRef<Var>(arg_var))) {
        arg_idx.Set(name, IntImm::Int64(*idx));
      }
    }
  }

  return arg_idx;
}

/*!
 * \brief Utility function to find the string pattern in string str
 * \param str the main string to check the pattern
 * \param pattern the pattern to check in the main string
 * \return return true if the main string ends with pattern, false otherwise
 */
bool EndsWithPattern(const std::string& str, const std::string& pattern) {
  if (str.length() < pattern.length()) return false;
  return str.compare(str.length() - pattern.length(), pattern.length(), pattern) == 0;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.contrib.extract_arg_idx", ExtractArgIdx);
}

}  // namespace backend
}  // namespace relax
}  // namespace tvm
