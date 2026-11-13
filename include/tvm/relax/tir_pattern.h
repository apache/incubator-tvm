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
 * \file tir_pattern.h
 * \brief Data Structure of TIR Pattern used for matching.
 */

#ifndef TVM_RELAX_TIR_PATTERN_H_
#define TVM_RELAX_TIR_PATTERN_H_

#include <tvm/base.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/function.h>

namespace tvm {
namespace relax {

using TIRPattern = tirx::Function;

/*
 * \brief The match result of a TIR pattern.
 */
class MatchResultObj : public ffi::Object {
 public:
  explicit MatchResultObj(TIRPattern pattern) : pattern(std::move(pattern)) {}
  explicit MatchResultObj(ffi::UnsafeInit) : pattern(ffi::UnsafeInit{}) {}

  /*! The matched tirx pattern*/
  TIRPattern pattern;
  /*! \brief The evaluated values of symbolic vars. */
  ffi::Array<PrimExpr> symbol_values;
  /*! \brief The matched tensors of input and output. */
  ffi::Array<tirx::TensorVar> matched_tensors;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<MatchResultObj>()
        .def_ro("pattern", &MatchResultObj::pattern)
        .def_ro("symbol_values", &MatchResultObj::symbol_values)
        .def_ro("matched_tensors", &MatchResultObj::matched_tensors);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.MatchResult", MatchResultObj, ffi::Object);
};

/*!
 * \brief Managed reference to MatchResultObj.
 */
class MatchResult : public ffi::ObjectRef {
 public:
  /*!
   * \brief Constructor
   * \param pattern The matched tirx pattern.
   * \param symbol_values The evaluated values of symbolic vars.
   * \param matched_tensors The matched tensors of input and output.
   */
  TVM_DLL explicit MatchResult(TIRPattern pattern, ffi::Array<PrimExpr> symbol_values,
                               ffi::Array<tirx::TensorVar> matched_tensors);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(MatchResult, ffi::ObjectRef, MatchResultObj);
};

using FCodegen = ffi::TypedFunction<ffi::Array<ffi::Any>(ffi::Array<MatchResult> match_results)>;
}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_TIR_PATTERN_H_
