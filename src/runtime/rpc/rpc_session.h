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
 * \file rpc_session.h
 * \brief Base RPC session interface.
 */
#ifndef TVM_RUNTIME_RPC_RPC_SESSION_H_
#define TVM_RUNTIME_RPC_RPC_SESSION_H_

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/device_api.h>

#include <algorithm>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "minrpc/rpc_reference.h"

namespace tvm {
namespace runtime {

/*!
 * \brief Static FFI type index for `runtime::RPCObjectRef`.
 *
 * Allocated within the [kTVMFFIDynObjectBegin - 16, kTVMFFIDynObjectBegin)
 * custom-static slot range. The sibling constant `kRuntimeDiscoDRef` lives in
 * `tvm/runtime/disco/session.h` and uses `... - 14`; values must remain
 * disjoint across this small reserved block.
 */
constexpr int32_t kRuntimeRPCObjectRef = TVMFFITypeIndex::kTVMFFIDynObjectBegin - 13;

/*! \brief Stream ownership and alias state, retained until release or session teardown. */
struct RPCStreamState {
  /*! \brief Ordinary endpoint device that created or selected this stream. */
  Device device;
  /*! \brief Opaque handle interpreted only by that endpoint's DeviceAPI. */
  TVMStreamHandle handle;
  /*! \brief Created by this session and freed on release; borrowed current streams are not freed.
   */
  bool owned;
  /*! \brief Shared invalidation flag observed by every wrapper alias before handle use. */
  std::atomic<bool> closed{false};
  RPCStreamState(Device device, TVMStreamHandle handle, bool owned)
      : device(device), handle(handle), owned(owned) {}
};

/*!
 * \brief The interface of all remote RPC sessions.
 *
 *  It contains all the necessary interface to implement
 *  remote call and resource management.
 *
 *  The interface is designed to allow easy proxy-chaining
 *  by forward requests to another RPCSession.
 */
class RPCSession {
 public:
  /*! \brief ffi::Function Handle in the remote. */
  using PackedFuncHandle = void*;

  /*! \brief Module handle in the remote. */
  using ModuleHandle = void*;

  /*! \brief Tensor handle in the remote. */
  using TensorHandle = void*;

  /*!
   * \brief Callback to send an encoded return values via encode_args.
   *
   * \param encode_args The arguments that we can encode the return values into.
   *
   * Encoding convention (as list of arguments):
   * - str/float/int/byte: [tcode: int, value: TVMValue] value follows ffi::Function convention.
   * - ffi::Function/Module: [tcode: int, handle: void*]
   * - Tensor: [tcode: int,  meta: DLTensor*, nd_handle: void*]
   *            DLTensor* contains the meta-data as well as handle into the remote data.
   *            nd_handle can be used for deletion.
   */
  using FEncodeReturn = std::function<void(ffi::PackedArgs encoded_args)>;

  /*!
   * \brief Callback to send an encoded return values via encode_args.
   *
   * \param status The return status, can be RPCCode::kReturn or RPCCode::kException.
   * \param encode_args The arguments that we can encode the return values into.
   */
  using FAsyncCallback = std::function<void(RPCCode status, ffi::PackedArgs encoded_args)>;

  /*! \brief Destructor.*/
  virtual ~RPCSession() {}

  /*!
   * \brief Get function in the session.
   * \param name The name of the function.
   * \return The function handle.
   */
  virtual PackedFuncHandle GetFunction(const std::string& name) = 0;

  /*!
   * \brief Call into a remote Packed function.
   *
   *  Calling convention:
   *
   *  - type_code is follows the ffi::Function convention.
   *  - int/float/string/bytes follows the ffi::Function convention, all data are local.
   *  - ffi::Function/Module and future remote objects: pass remote handle instead.
   *  - Tensor/DLTensor: pass a DLTensor pointer, the data field of DLTensor
   *                      points to a remote data handle returned by the session.
   *                      The meta-data of the DLTensor sits on local.
   *
   *  The caller populates the arguments and manages these arguments.
   *
   *  The callee can change the content of arg_values and arg_type_codes
   *  if they want to do inplace modify and forward.
   *
   *  The callee need to store the return value into ret_value.
   *  - ffi::Function/Module are stored as void*
   *  - Tensor is encoded as a transport-only DLTensor descriptor and an object
   *    handle. The caller wraps these in an explicitly remote owning object.
   *  - In short, remote handles are "moved" as return values
   *    and the callee needs to explicitly manage them by calling
   *    the deleter functions when they are no longer needed.
   *
   * \param func The function handle.
   * \param args The input packed arguments.
   * \param fencode_return The function to set the return value,
   *                       if not called, return value is null.
   */
  virtual void CallFunc(PackedFuncHandle func, ffi::PackedArgs args,
                        const FEncodeReturn& fencode_return) = 0;

  /*!
   * \brief Copy bytes into remote array content.
   * \param local_from_bytes The source host data.
   * \param remote_to The target array.
   * \param nbytes The size of the memory in bytes.
   */
  virtual void CopyToRemote(void* local_from_bytes, DLTensor* remote_to, uint64_t nbytes) = 0;
  /*!
   * \brief Copy bytes from remote array content.
   * \param remote_from The source host data.
   * \param local_to_bytes The target array.
   * \param nbytes The size of the memory in bytes.
   */
  virtual void CopyFromRemote(DLTensor* remote_from, void* local_to_bytes, uint64_t nbytes) = 0;

  /*!
   * \brief Free a remote function.
   * \param handle The remote object handle.
   * \param type_code The type code of the underlying type.
   */
  virtual void FreeHandle(void* handle) = 0;

  // These operations transport endpoint handles, never local ffi::Tensor storage.
  virtual void SetDevice(Device dev) = 0;
  virtual ffi::Any GetAttr(Device dev, const ffi::String& name) = 0;
  virtual void* AllocTensor(Device dev, ffi::ShapeView shape, DLDataType dtype,
                            ffi::Optional<MemoryScope> mem_scope) = 0;
  virtual void FreeTensor(Device dev, void* handle) = 0;
  virtual void CopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream) = 0;
  virtual TVMStreamHandle CreateStream(Device dev) = 0;
  virtual void FreeStream(Device dev, TVMStreamHandle stream) = 0;
  virtual void StreamSync(Device dev, TVMStreamHandle stream) = 0;
  virtual void SetStream(Device dev, TVMStreamHandle stream) = 0;
  virtual TVMStreamHandle GetCurrentStream(Device dev) = 0;

  /*!
   * \brief Whether the session is a local session and we can directly
   *        the data handle returned by the session and treat it as pointer
   *        to the local memory.
   *
   * This information is useful for RPC server to directly copy into the
   * local memory without creating a temporary buffer.
   *
   * \return Whether it is a local session.
   */
  virtual bool IsLocalSession() const = 0;

  // Asynchrous variant of API
  // These APIs are used by the RPC server to allow sessions that
  // have special implementations for the async functions.
  //
  // In the async APIs, an exception is returned by the passing
  // async_error=true, encode_args=[error_msg].

  /*!
   * \brief Whether the session is async.
   *
   * If the session is not async, its Aync implementations
   * simply calls into the their synchronize counterparts,
   * and the callback is guaranteed to be called before the async function finishes.
   *
   * \return the async state.
   *
   * \note We can only use async session in an Event driven RPC server.
   */
  virtual bool IsAsync() const;

  /*!
   * \brief Asynchrously call func.
   * \param func The function handle.
   * \param args The packed arguments.
   *
   * \param callback The callback to pass the return value or exception.
   */
  virtual void AsyncCallFunc(PackedFuncHandle func, ffi::PackedArgs args, FAsyncCallback callback);

  /*!
   * \brief Asynchrous version of CopyToRemote.
   *
   * \param local_from_bytes The source host data.
   * \param remote_to The target array.
   * \param nbytes The size of the memory in bytes.
   * \param on_complete The callback to signal copy complete.
   * \note All the allocated memory in local_from, and remote_to
   *       must stay alive until on_compelete is called.
   */
  virtual void AsyncCopyToRemote(void* local_from_bytes, DLTensor* remote_to, uint64_t nbytes,
                                 FAsyncCallback on_complete);

  /*!
   * \brief Asynchrous version of CopyFromRemote.
   *
   * \param remote_from The source host data.
   * \param local_to_bytes The target array.
   * \param nbytes The size of the memory in bytes.
   * \param on_complete The callback to signal copy complete.
   * \note All the allocated memory in remote_from, and local_to
   *       must stay alive until on_compelete is called.
   */
  virtual void AsyncCopyFromRemote(DLTensor* remote_from, void* local_to_bytes, uint64_t nbytes,
                                   FAsyncCallback on_complete);
  /*!
   * \brief Asynchrously wait for all events in dev, stream compeletes.
   * \param dev The device.
   * \param stream The stream to wait on.
   * \param on_complete The callback to signal copy complete.
   */
  virtual void AsyncStreamWait(Device dev, TVMStreamHandle stream, FAsyncCallback on_compelte);

  /*! \brief Reject operations after explicit session closure. */
  void EnsureOpen() const { TVM_FFI_CHECK(!closed_, RuntimeError) << "RPC session is closed"; }
  void Close() {
    if (!closed_.exchange(true)) {
      ReleaseStreams();
      Shutdown();
    }
  }
  std::shared_ptr<RPCStreamState> TrackStream(Device dev, TVMStreamHandle handle, bool owned) {
    std::lock_guard<std::mutex> lock(stream_mutex_);
    EnsureOpen();
    for (const auto& stream : streams_) {
      if (!stream->closed && stream->handle == handle &&
          stream->device.device_type == dev.device_type &&
          stream->device.device_id == dev.device_id) {
        stream->owned |= owned;
        return stream;
      }
    }
    auto stream = std::make_shared<RPCStreamState>(dev, handle, owned);
    streams_.push_back(stream);
    return stream;
  }
  void FreeStream(const std::shared_ptr<RPCStreamState>& stream) {
    std::lock_guard<std::mutex> lock(stream_mutex_);
    if (stream->closed) return;
    stream->closed = true;
    streams_.erase(std::remove(streams_.begin(), streams_.end(), stream), streams_.end());
    if (GetCurrentStream(stream->device) == stream->handle) {
      SetStream(stream->device, nullptr);
    }
    if (stream->owned) FreeStream(stream->device, stream->handle);
  }
  // Derived destructors call this while their endpoint/transport is still alive.
  void ReleaseStreams() {
    std::vector<std::shared_ptr<RPCStreamState>> streams;
    {
      std::lock_guard<std::mutex> lock(stream_mutex_);
      streams.swap(streams_);
    }
    for (const auto& stream : streams) {
      try {
        FreeStream(stream);
      } catch (const ffi::Error&) {
      }
    }
  }
  virtual void Shutdown() {}

 protected:
  /*!
   * \brief Send an exception to the callback.
   * \param msg The exception message.
   */
  void SendException(FAsyncCallback callback, const char* msg);

 private:
  std::atomic<bool> closed_{false};
  std::vector<std::shared_ptr<RPCStreamState>> streams_;
  std::mutex stream_mutex_;
};

/*!
 * \brief ffi::Object wrapper that represents a reference to a remote object
 */
class RPCObjectRefObj : public ffi::Object {
 public:
  /*!
   * \brief constructor
   * \param object_handle handle that points to the remote object
   *
   * \param sess The remote session, when session is nullptr
   * it indicate the object is a temp object during rpc transmission
   * and we don't have to free it
   */
  RPCObjectRefObj(void* object_handle, std::shared_ptr<RPCSession> sess)
      : object_handle_(object_handle), sess_(sess) {}

  ~RPCObjectRefObj() {
    if (object_handle_ != nullptr && sess_ != nullptr) {
      try {
        sess_->FreeHandle(object_handle_);
      } catch (const ffi::Error& e) {
        // fault tolerance to remote close
      }
      object_handle_ = nullptr;
    }
  }

  const std::shared_ptr<RPCSession>& sess() const { return sess_; }

  void* object_handle() const { return object_handle_; }

  static constexpr const uint32_t _type_index = kRuntimeRPCObjectRef;
  static const constexpr bool _type_final = true;
  TVM_FFI_DECLARE_OBJECT_INFO_STATIC("runtime.RPCObjectRef", RPCObjectRefObj, ffi::Object);

 private:
  // The object handle
  void* object_handle_{nullptr};
  // The local channel
  std::shared_ptr<RPCSession> sess_;
};

/*!
 * \brief Managed reference to RPCObjectRefObj.
 * \sa RPCObjectRefObj
 * \note No public constructor is provided as it is not supposed to be directly created by users.
 */
class RPCObjectRef : public ffi::ObjectRef {
 public:
  explicit RPCObjectRef(ffi::ObjectPtr<RPCObjectRefObj> data) : ffi::ObjectRef(data) {
    TVM_FFI_ICHECK(data != nullptr);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(RPCObjectRef, ffi::ObjectRef, RPCObjectRefObj);
};

/*!
 * \brief Create a Global RPC module that refers to the session.
 * \param sess The RPC session of the global module.
 * \return The created module.
 */
ffi::Module CreateRPCSessionModule(std::shared_ptr<RPCSession> sess);

/*!
 * \brief Get the session module from a RPC session Module.
 * \param mod The input module(must be an RPCModule).
 * \return The internal RPCSession.
 */
std::shared_ptr<RPCSession> RPCModuleGetSession(ffi::Module mod);

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_RPC_RPC_SESSION_H_
