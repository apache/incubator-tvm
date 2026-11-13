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
 * \file local_session.cc
 * \brief Local session that directs requests to local API.
 */
#include "rpc_local_session.h"

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/tensor.h>

#include <memory>
#include <vector>

namespace tvm {
namespace runtime {

RPCSession::PackedFuncHandle LocalSession::GetFunction(const std::string& name) {
  if (auto fp = tvm::ffi::Function::GetGlobal(name)) {
    // return raw handle because the remote need to explicitly manage it.
    Any ret = *fp;
    TVMFFIAny ret_any = ffi::detail::AnyUnsafe::MoveAnyToTVMFFIAny(std::move(ret));
    return ret_any.v_obj;
  } else {
    return nullptr;
  }
}

void LocalSession::EncodeReturn(ffi::Any rv, const FEncodeReturn& encode_return) {
  AnyView packed_args[3];
  // NOTE: this is the place that we need to handle special RPC-related
  // ABI convention for return value passing that is built on top of Any FFI.
  // first argument is always the type index.
  packed_args[0] = rv.type_index();
  if (rv == nullptr) {
    packed_args[1] = rv;
    encode_return(ffi::PackedArgs(packed_args, 2));
  } else if (rv.as<ffi::TensorObj>()) {
    // We follow a special protocol to return Tensor to client side
    // The first pack value is the Tensor handle as DLTensor
    // The second pack value is a customized deleter that deletes the Tensor.
    TVMFFIAny ret_any = ffi::detail::AnyUnsafe::MoveAnyToTVMFFIAny(std::move(rv));
    void* opaque_handle = ret_any.v_obj;
    packed_args[1] = TVMFFITensorGetDLTensorPtr(opaque_handle);
    packed_args[2] = opaque_handle;
    encode_return(ffi::PackedArgs(packed_args, 3));
  } else if (const auto opt_bytes = rv.as<ffi::Bytes>()) {
    // always pass bytes as byte array
    TVMFFIByteArray byte_arr;
    byte_arr.data = (*opt_bytes).data();
    byte_arr.size = (*opt_bytes).size();
    packed_args[1] = &byte_arr;
    encode_return(ffi::PackedArgs(packed_args, 2));
  } else if (auto opt_str = rv.as<ffi::String>()) {
    // encode string as c_str
    packed_args[1] = (*opt_str).data();
    encode_return(ffi::PackedArgs(packed_args, 2));
  } else if (rv.as<ffi::ObjectRef>()) {
    TVMFFIAny ret_any = ffi::detail::AnyUnsafe::MoveAnyToTVMFFIAny(std::move(rv));
    void* opaque_handle = ret_any.v_obj;
    packed_args[1] = opaque_handle;
    encode_return(ffi::PackedArgs(packed_args, 2));
  } else {
    packed_args[1] = rv;
    encode_return(ffi::PackedArgs(packed_args, 2));
  }
}

void LocalSession::CallFunc(RPCSession::PackedFuncHandle func, ffi::PackedArgs args,
                            const FEncodeReturn& encode_return) {
  ffi::FunctionObj* pf = static_cast<ffi::FunctionObj*>(func);

  Any rv;
  std::vector<AnyView> packed_args(args.size());

  // unwrap RPCObjectRef in case we are directly using it to call LocalSession
  for (int i = 0; i < args.size(); ++i) {
    if (auto opt_rpc_obj = args[i].as<RPCObjectRef>()) {
      packed_args[i] = static_cast<const ffi::Object*>(opt_rpc_obj.value()->object_handle());
    } else {
      packed_args[i] = args[i];
    }
  }

  pf->CallPacked(packed_args.data(), packed_args.size(), &rv);
  this->EncodeReturn(std::move(rv), encode_return);
}

void LocalSession::CopyToRemote(void* from_bytes, DLTensor* to, uint64_t nbytes) {
  TVM_FFI_ICHECK_EQ(nbytes, ffi::GetDataSize(*to));
  DLTensor from;
  from.data = from_bytes;
  from.device = {kDLCPU, 0};
  from.ndim = to->ndim;
  from.shape = to->shape;
  from.dtype = to->dtype;
  from.strides = nullptr;
  from.byte_offset = 0;
  Device dev_to = to->device;
  this->CopyDataFromTo(&from, to, nullptr);
  // Copy can happen asynchrously
  // synchronize to make sure that copy is completed
  this->StreamSync(dev_to, nullptr);
}

void LocalSession::CopyFromRemote(DLTensor* from, void* to_bytes, uint64_t nbytes) {
  TVM_FFI_ICHECK_EQ(nbytes, ffi::GetDataSize(*from));
  DLTensor to;
  to.data = to_bytes;
  to.device = {kDLCPU, 0};
  to.ndim = from->ndim;
  to.shape = from->shape;
  to.dtype = from->dtype;
  to.strides = nullptr;
  to.byte_offset = 0;

  Device dev_from = from->device;
  this->CopyDataFromTo(from, &to, nullptr);
  // Copy can happen asynchrously
  // synchronize to make sure that copy is completed
  this->StreamSync(dev_from, nullptr);
}

void LocalSession::FreeHandle(void* handle) {
  // NOTE: the type code is no longer need during free handle.
  ffi::detail::ObjectUnsafe::DecRefObjectHandle(handle);
}

void LocalSession::SetDevice(Device dev) { DeviceAPI::Get(dev)->SetDevice(dev); }

ffi::Any LocalSession::GetAttr(Device dev, const ffi::String& name) {
  auto* api = DeviceAPI::Get(dev, name == "exist");
  if (api == nullptr) return false;
  return api->GetAttr(dev, name);
}

void* LocalSession::AllocTensor(Device dev, ffi::ShapeView shape, DLDataType dtype,
                                ffi::Optional<MemoryScope> mem_scope) {
  EnsureOpen();
  ffi::Tensor tensor = DeviceAPI::Get(dev)->Empty(shape, dtype, dev, mem_scope);
  void* handle = tensor.data_ptr();
  std::lock_guard<std::mutex> lock(allocation_mutex_);
  EnsureOpen();
  allocations_[handle].push_back(std::move(tensor));
  return handle;
}

void LocalSession::FreeTensor(Device dev, void* handle) {
  std::lock_guard<std::mutex> lock(allocation_mutex_);
  auto it = allocations_.find(handle);
  TVM_FFI_CHECK(it != allocations_.end(), ValueError) << "Unknown remote allocation";
  auto& owners = it->second;
  auto owner = std::find_if(owners.begin(), owners.end(), [dev](const ffi::Tensor& tensor) {
    return tensor.device().device_type == dev.device_type &&
           tensor.device().device_id == dev.device_id;
  });
  TVM_FFI_CHECK(owner != owners.end(), ValueError) << "Remote allocation belongs to another device";
  owners.erase(owner);
  if (owners.empty()) allocations_.erase(it);
}

void LocalSession::CopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream) {
  Device dev = from->device.device_type == kDLCPU ? to->device : from->device;
  DeviceAPI::Get(dev)->CopyDataFromTo(from, to, stream);
}

TVMStreamHandle LocalSession::CreateStream(Device dev) {
  return DeviceAPI::Get(dev)->CreateStream(dev);
}

void LocalSession::FreeStream(Device dev, TVMStreamHandle stream) {
  DeviceAPI::Get(dev)->FreeStream(dev, stream);
}

void LocalSession::StreamSync(Device dev, TVMStreamHandle stream) {
  DeviceAPI::Get(dev)->StreamSync(dev, stream);
}

void LocalSession::SetStream(Device dev, TVMStreamHandle stream) {
  DeviceAPI::Get(dev)->SetStream(dev, stream);
}

TVMStreamHandle LocalSession::GetCurrentStream(Device dev) {
  return DeviceAPI::Get(dev)->GetCurrentStream(dev);
}

void LocalSession::Shutdown() {
  std::lock_guard<std::mutex> lock(allocation_mutex_);
  allocations_.clear();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("rpc.LocalSession",
                        []() { return CreateRPCSessionModule(std::make_shared<LocalSession>()); });
}

}  // namespace runtime
}  // namespace tvm
