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
 * \file rpc_local_session.h
 * \brief Local session that directs all request to the local runtime API.
 */
#ifndef TVM_RUNTIME_RPC_RPC_LOCAL_SESSION_H_
#define TVM_RUNTIME_RPC_RPC_LOCAL_SESSION_H_

#include <tvm/ffi/function.h>
#include <tvm/runtime/device_api.h>

#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rpc_session.h"

namespace tvm {
namespace runtime {

/*!
 * \brief A local session that directly use the handle repr of the
 *        local tvm runtime objects on the same process.
 */
class LocalSession : public RPCSession {
 public:
  ~LocalSession() override { ReleaseStreams(); }
  // function overrides
  PackedFuncHandle GetFunction(const std::string& name) override;

  void CallFunc(PackedFuncHandle func, ffi::PackedArgs args,
                const FEncodeReturn& fencode_return) override;

  void CopyToRemote(void* from_bytes, DLTensor* to, uint64_t nbytes) override;

  void CopyFromRemote(DLTensor* from, void* to_bytes, uint64_t nbytes) override;

  void FreeHandle(void* handle) override;

  void SetDevice(Device dev) override;
  ffi::Any GetAttr(Device dev, const ffi::String& name) override;
  void* AllocTensor(Device dev, ffi::ShapeView shape, DLDataType dtype,
                    ffi::Optional<MemoryScope> mem_scope) override;
  void FreeTensor(Device dev, void* handle) override;
  void CopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream) override;
  TVMStreamHandle CreateStream(Device dev) override;
  void FreeStream(Device dev, TVMStreamHandle stream) override;
  void StreamSync(Device dev, TVMStreamHandle stream) override;
  void SetStream(Device dev, TVMStreamHandle stream) override;
  TVMStreamHandle GetCurrentStream(Device dev) override;
  void Shutdown() override;

  bool IsLocalSession() const override { return true; }

 protected:
  /*!
   * \brief internal encode return function.
   * \param rv The return value.
   * \param encode_return The encoding function.
   */
  void EncodeReturn(ffi::Any rv, const FEncodeReturn& encode_return);

 private:
  // A backend may reuse a data handle (including nullptr for an empty tensor).
  // Retain each allocation separately and match its device when releasing it.
  std::unordered_map<void*, std::vector<ffi::Tensor>> allocations_;
  std::mutex allocation_mutex_;
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_RPC_RPC_LOCAL_SESSION_H_
