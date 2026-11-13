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
#ifndef TVM_S_TIR_SCRIPT_IR_BUILDER_UTILS_H_
#define TVM_S_TIR_SCRIPT_IR_BUILDER_UTILS_H_

#include <tvm/s_tir/script/ir_builder/frame.h>
#include <tvm/script/ir_builder/frame.h>

namespace tvm {
namespace script {
namespace ir_builder {
namespace s_tir {

using ir::AddToParent;

/*!
 * \brief Check whether the top frame in IRBuilder frame stack is SBlockFrame.
 * \param method The method name to be printed when throwing exception.
 * \return The top frame of SBlockFrame.
 */
inline SBlockFrame FindSBlockFrame(const ffi::String& method) {
  if (ffi::Optional<SBlockFrame> frame = IRBuilder::Current()->GetLastFrame<SBlockFrame>()) {
    return frame.value();
  } else if (ffi::Optional<SBlockFrame> frame = IRBuilder::Current()->FindFrame<SBlockFrame>()) {
    TVM_FFI_THROW(ValueError)
        << method << " must be called at the top of a Ts.sblock().  "
        << "While " << method << " did occur within the block \"" << frame.value()->name
        << "\", other frames (e.g. if/else/let) had been introduced since the Ts.sblock(\""
        << frame.value()->name << "\") frame";
  } else {
    TVM_FFI_THROW(ValueError) << method << " must be called at the top of a Ts.sblock(), "
                              << "but " << method << " occurred outside of any Ts.sblock() frame";
  }
  throw;
}

}  // namespace s_tir
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_S_TIR_SCRIPT_IR_BUILDER_UTILS_H_
