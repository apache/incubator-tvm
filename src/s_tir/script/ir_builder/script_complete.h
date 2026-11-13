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
 * \file s_tir/script/ir_builder/script_complete.h
 * \brief Used by TVM Script parser to expand incomplete TIR input
 */
#ifndef TVM_S_TIR_SCRIPT_IR_BUILDER_SCRIPT_COMPLETE_H_
#define TVM_S_TIR_SCRIPT_IR_BUILDER_SCRIPT_COMPLETE_H_
#include <tvm/ffi/function.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm {
namespace s_tir {

tvm::tirx::Function ScriptComplete(tvm::tirx::Function func,
                                   const ffi::Array<tvm::tirx::TensorVar>& root_allocates,
                                   const TensorAllocatedAddresses& root_addresses = {});

}  // namespace s_tir
}  // namespace tvm
#endif  // TVM_S_TIR_SCRIPT_IR_BUILDER_SCRIPT_COMPLETE_H_
