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
#ifndef TVM_RELAX_SCRIPT_IR_BUILDER_UTILS_H_
#define TVM_RELAX_SCRIPT_IR_BUILDER_UTILS_H_

#include <tvm/relax/script/ir_builder/frame.h>
#include <tvm/relax/script/ir_builder/ir.h>
#include <tvm/relax/type_functor.h>
#include <tvm/relax/utils.h>

#include <string>

namespace tvm {
namespace script {
namespace ir_builder {
namespace relax {

inline FunctionFrame FindFunctionFrame(const ffi::String& method) {
  if (ffi::Optional<FunctionFrame> frame = IRBuilder::Current()->FindFrame<FunctionFrame>()) {
    return frame.value();
  }
  TVM_FFI_THROW(ValueError) << "Function frame not find. Please ensure '" << method
                            << "' is called under R.function()";
  throw;
}

inline IfFrame FindIfFrame(const ffi::String& method) {
  if (ffi::Optional<IfFrame> frame = IRBuilder::Current()->GetLastFrame<IfFrame>()) {
    return frame.value();
  } else {
    TVM_FFI_THROW(ValueError) << "IfThenElse frame not find. Please ensure '" << method
                              << "' is called under R.if_()";
  }
  throw;
}

inline tvm::relax::BlockBuilder GetBlockBuilder() {
  ffi::Optional<FunctionFrame> frame = IRBuilder::Current()->FindFrame<FunctionFrame>();
  TVM_FFI_CHECK(frame.has_value(), ValueError) << "Relax Function frame not find. Please ensure "
                                                  "assignment is called under R.function()";
  return frame.value()->block_builder;
}

inline BindingBlockFrame CheckBindingBlockFrameExistAndUnended() {
  // We check if the current block is "ended" - if a block is ended, it is not allowed to emit new
  // bindings into this block, and we should throw exceptions.

  ffi::Optional<BindingBlockFrame> block_frame =
      IRBuilder::Current()->GetLastFrame<BindingBlockFrame>();
  TVM_FFI_CHECK(block_frame.has_value(), ValueError) << "Block frame not find";
  TVM_FFI_CHECK(!block_frame.value()->block_ended, ValueError)
      << "New binding is not allowed after dataflow block output.";
  return block_frame.value();
}

inline tvm::relax::SeqExpr GetSeqExprForBranch(const SeqExprFrame& frame, ffi::String* var_name) {
  // Step 0. Check frame type
  std::string method;
  std::string output_var_suffix;
  if (frame->IsInstance<ThenFrameObj>()) {
    method = "R.Then";
    output_var_suffix = "_then";
  } else if (frame->IsInstance<ElseFrameObj>()) {
    method = "R.Else";
    output_var_suffix = "_else";
  } else {
    TVM_FFI_CHECK(false, TypeError) << "Unsupported frame type: " << frame->GetTypeKey();
  }

  // Step 1. Check non-empty block and last binding is non-dataflow
  TVM_FFI_ICHECK(!frame->binding_blocks.empty())
      << "Empty body is not allowed for '" << method << "' statements.";
  const tvm::relax::BindingBlock& last_block = frame->binding_blocks.back();
  TVM_FFI_ICHECK(!last_block->bindings.empty()) << "Blocks are expected to be non-empty.";

  // Step 2. Update the last binding of each branch.  While we could
  // use the last bound value of each branch as a SeqExpr body, the
  // Normalizer would pull it back out into a `gv#` binding anyways.
  // Generating a new variable in each branch provides a more readable
  // variable name.

  tvm::relax::Binding last_binding = last_block->bindings.back();
  TVM_FFI_ICHECK(!last_binding->var->IsInstance<tvm::relax::DataflowVarObj>())
      << "A non-dataflow var is expected in the last binding of '" << method << "'.";

  *var_name = last_binding->var->name;

  // Step 3. Re-collect binding blocks to replace the last binding.
  ffi::Array<tvm::relax::BindingBlock> new_blocks(frame->binding_blocks.begin(),
                                                  frame->binding_blocks.end() - 1);
  ffi::Array<tvm::relax::Binding> last_block_bindings(last_block->bindings.begin(),
                                                      last_block->bindings.end() - 1);

  tvm::Var new_var(last_binding->var->name + output_var_suffix,
                   tvm::relax::GetType(last_binding->var), last_binding->var->loc);
  tvm::relax::Expr body = new_var;

  const auto* var_binding = last_binding.as<tvm::relax::VarBindingObj>();

  if (var_binding && tvm::relax::IsLeafOrTuple(var_binding->value)) {
    body = var_binding->value;
  } else if (var_binding) {
    last_block_bindings.push_back(
        tvm::relax::VarBinding(new_var, var_binding->value, var_binding->loc));
  } else if (const auto* match_cast = last_binding.as<tvm::relax::MatchCastObj>()) {
    last_block_bindings.push_back(
        tvm::relax::MatchCast(new_var, match_cast->value, match_cast->ty, match_cast->loc));
  } else {
    TVM_FFI_CHECK(false, TypeError) << "Unsupported binding type: " << last_binding->GetTypeKey();
  }

  new_blocks.push_back(last_block->IsInstance<tvm::relax::DataflowBlockObj>()
                           ? tvm::relax::DataflowBlock(last_block_bindings, last_block->loc)
                           : tvm::relax::BindingBlock(last_block_bindings, last_block->loc));

  return tvm::relax::SeqExpr(new_blocks, body, frame->loc);
}

}  // namespace relax
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_RELAX_SCRIPT_IR_BUILDER_UTILS_H_
