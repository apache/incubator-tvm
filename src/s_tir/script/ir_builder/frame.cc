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
#include <tvm/ir/function.h>
#include <tvm/ir/prim/op.h>
#include <tvm/s_tir/script/ir_builder/frame.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/function.h>

#include "./script_complete.h"
#include "./utils.h"

namespace tvm {
namespace script {
namespace ir_builder {
namespace s_tir {
TVM_FFI_STATIC_INIT_BLOCK() {
  FunctionFrameObj::RegisterReflection();
  SBlockFrameObj::RegisterReflection();
  BlockInitFrameObj::RegisterReflection();
}

tvm::tirx::Function FunctionFrameObj::FinalizeFunction(tvm::tirx::Function func) {
  func = WithAttr(std::move(func), tvm::attr::kScriptNamespace.name, ffi::String("s_tir"));
  TVM_FFI_CHECK(!is_declaration || root_alloc_tensors.empty(), ValueError)
      << "A function declaration cannot allocate tensors";
  if (!is_declaration) {
    func =
        tvm::s_tir::ScriptComplete(std::move(func), root_alloc_tensors, root_allocated_addresses);
  }
  return func;
}

void FunctionFrameObj::BindTensorRegion(tvm::tirx::TensorVar tensor, tvm::TensorRegion region,
                                        ffi::Array<PrimExpr> allocated_addr) {
  if (auto function = IRBuilder::Current()->FindFrame<tirx::FunctionFrame>()) {
    TVM_FFI_CHECK(function.value().as<FunctionFrameObj>() != nullptr, ValueError)
        << "match_tensor requires Ts.function; T.function only accepts TIRx";
  }
  SBlockFrame frame = FindSBlockFrame("Ts.match_tensor");
  frame->match_tensors.push_back(tvm::s_tir::MatchTensorRegion(tensor, std::move(region)));
  if (!allocated_addr.empty()) {
    frame->allocated_addresses.push_back(
        ffi::Tuple<Var, ffi::Array<PrimExpr>>(tensor, std::move(allocated_addr)));
  }
}

void SBlockFrameObj::ExitWithScope() {
  ir::StmtFrameObj::ExitWithScope();

  // Shared operations remain usable in S-TIR and raw builder contexts, but
  // a TIRx function cannot contain an S-TIR block, even with validation disabled.
  if (auto function = IRBuilder::Current()->FindFrame<tirx::FunctionFrame>()) {
    TVM_FFI_CHECK(function.value().as<FunctionFrameObj>() != nullptr, ValueError)
        << "S-TIR blocks require Ts.function; T.function only accepts TIRx";
  }

  ffi::Array<tvm::tirx::TensorVar> tir_alloc_tensors;
  for (const tvm::tirx::TensorVar& tensor : alloc_tensors) {
    tir_alloc_tensors.push_back(tensor);
  }
  ffi::Map<ffi::String, Any> attrs = annotations.value_or({});
  if (!allocated_addresses.empty()) {
    TVM_FFI_CHECK(!attrs.count(tvm::s_tir::attr::kTensorAllocatedAddr.name), ValueError)
        << "Tensor placement must be specified on its allocation or match_tensor";
    attrs.Set(tvm::s_tir::attr::kTensorAllocatedAddr.name, allocated_addresses);
  }
  if (int detect_access = (!reads.has_value()) | (!writes.has_value() << 1)) {
    attrs.Set(tvm::s_tir::attr::kScriptParsingDetectAccess.name, detect_access);
  }
  tvm::s_tir::SBlock block(iter_vars, reads.value_or(ffi::Array<tvm::TensorRegion>()),
                           writes.value_or(ffi::Array<tvm::TensorRegion>()), name,
                           tvm::SeqStmt(stmts), init, tir_alloc_tensors, match_tensors, attrs, loc);
  if (no_realize) {
    TVM_FFI_CHECK(iter_values.empty(), ValueError)
        << "Block bindings are not allowed when `no_realize=True`";
    TVM_FFI_CHECK(!predicate.has_value(), ValueError)
        << "`Ts.where` is not allowed when `no_realize=True`";
    AddToParent(block, loc);
  } else {
    AddToParent(
        tvm::s_tir::SBlockRealize(iter_values, predicate.value_or(IntImm::Bool(true)), block, loc),
        loc);
  }
}

void BlockInitFrameObj::EnterWithScope() {
  SBlockFrame frame = FindSBlockFrame("Ts.init");
  if (frame->init.has_value()) {
    TVM_FFI_THROW(ValueError) << "Duplicate block init declaration";
  }
  ir::StmtFrameObj::EnterWithScope();
}

void BlockInitFrameObj::ExitWithScope() {
  ir::StmtFrameObj::ExitWithScope();
  SBlockFrame frame = FindSBlockFrame("Ts.init");
  frame->init = tvm::SeqStmt(stmts);
}

}  // namespace s_tir
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm
