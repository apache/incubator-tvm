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
 * \file src/relax/transform/annotate_tir_op_pattern.cc
 * \brief Annotate Op Pattern for TIR functions. It is a pass works on TIR Functions,
 *        but they are needed for relax fusion. So we put them in the relax namespace.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/transform.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace relax {

tirx::Function AnnotateOpPattern(tirx::Function f) {
  if (f->HasNonzeroAttr(tvm::relax::attr::kOpPattern.name)) {
    return f;
  } else {
    OpPatternKind kind = AnalyzeOpPatternKind(f);
    return WithAttr(std::move(f), tvm::relax::attr::kOpPattern.name, static_cast<int>(kind));
  }
}

namespace transform {

Pass AnnotateTIROpPattern() {
  auto pass_func = [=](tirx::Function f, IRModule m, PassContext ctx) {
    return AnnotateOpPattern(std::move(f));
  };
  return tirx::transform::CreateFunctionPass(pass_func, 0, "AnnotateTIROpPattern");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.AnnotateTIROpPattern", AnnotateTIROpPattern);
}

}  // namespace transform

}  // namespace relax
}  // namespace tvm
