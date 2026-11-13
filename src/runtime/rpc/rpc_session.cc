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
 * \file rpc_session.cc
 * \brief RPC session for remote function call.
 */
#include "rpc_session.h"

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/device_api.h>

namespace tvm {
namespace runtime {

bool RPCSession::IsAsync() const { return false; }

void RPCSession::SendException(FAsyncCallback callback, const char* msg) {
  ffi::AnyView packed_args[1] = {msg};
  callback(RPCCode::kException, ffi::PackedArgs(packed_args, 1));
}

void RPCSession::AsyncCallFunc(PackedFuncHandle func, ffi::PackedArgs packed_args,
                               FAsyncCallback callback) {
  try {
    this->CallFunc(func, packed_args,
                   [&callback](ffi::PackedArgs args) { callback(RPCCode::kReturn, args); });
  } catch (const std::exception& e) {
    this->SendException(callback, e.what());
  }
}

void RPCSession::AsyncCopyToRemote(void* local_from_bytes, DLTensor* remote_to, uint64_t nbytes,
                                   RPCSession::FAsyncCallback callback) {
  ffi::AnyView packed_args[1] = {nullptr};

  try {
    this->CopyToRemote(local_from_bytes, remote_to, nbytes);
    callback(RPCCode::kReturn, ffi::PackedArgs(packed_args, 1));
  } catch (const std::exception& e) {
    this->SendException(callback, e.what());
  }
}

void RPCSession::AsyncCopyFromRemote(DLTensor* remote_from, void* local_to_bytes, uint64_t nbytes,
                                     RPCSession::FAsyncCallback callback) {
  ffi::AnyView packed_args[1] = {nullptr};

  try {
    this->CopyFromRemote(remote_from, local_to_bytes, nbytes);
    callback(RPCCode::kReturn, ffi::PackedArgs(packed_args, 1));
  } catch (const std::exception& e) {
    this->SendException(callback, e.what());
  }
}

void RPCSession::AsyncStreamWait(Device dev, TVMStreamHandle stream,
                                 RPCSession::FAsyncCallback callback) {
  AnyView packed_args[1] = {nullptr};

  try {
    this->StreamSync(dev, stream);
    callback(RPCCode::kReturn, ffi::PackedArgs(packed_args, 1));
  } catch (const std::exception& e) {
    this->SendException(callback, e.what());
  }
}

TVM_FFI_STATIC_INIT_BLOCK() { tvm::ffi::reflection::ObjectDef<RPCObjectRefObj>(); }

}  // namespace runtime
}  // namespace tvm
