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

/*! \file rpc_remote.h
 *  \brief Session-owned RPC values. Remote addresses never enter local tensor storage.
 */
#ifndef TVM_RUNTIME_RPC_RPC_REMOTE_H_
#define TVM_RUNTIME_RPC_RPC_REMOTE_H_

#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/reflection/registry.h>

#include <memory>
#include <utility>

#include "rpc_session.h"

namespace tvm {
namespace runtime {

/*! \brief Session-owned device descriptor for an RPC execution endpoint.
 *
 * The endpoint descriptor never carries session identity. Only this object
 * binds it to a session, and all device operations must use that session.
 */
class RemoteDeviceObj : public ffi::Object {
 public:
  /*! \brief Keeps the transport alive; explicit session closure still invalidates operations. */
  std::shared_ptr<RPCSession> sess;
  /*! \brief Ordinary device type/index in the remote endpoint, not the client process. */
  Device endpoint_device;
  RPCSession* session() const {
    sess->EnsureOpen();
    return sess.get();
  }
  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("rpc.RemoteDevice", RemoteDeviceObj, ffi::Object);
};
/*! \brief Owning reference to a session-bound remote device. */
class RemoteDevice : public ffi::ObjectRef {
 public:
  explicit RemoteDevice(ffi::ObjectPtr<RemoteDeviceObj> data) : ffi::ObjectRef(std::move(data)) {}
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(RemoteDevice, ffi::ObjectRef, RemoteDeviceObj);
};
inline RemoteDevice MakeRemoteDevice(std::shared_ptr<RPCSession> sess, Device dev) {
  sess->EnsureOpen();
  auto n = ffi::make_object<RemoteDeviceObj>();
  n->sess = std::move(sess);
  n->endpoint_device = dev;
  return RemoteDevice(n);
}

/*! \brief Shared ownership of a remote allocation or returned tensor object. */
struct RemoteTensorStorage {
  /*! \brief Session that allocated the storage or returned the owning tensor handle. */
  std::shared_ptr<RPCSession> sess;
  /*! \brief Endpoint device identifying the session-owned tensor allocation. */
  Device device;
  /*! \brief Opaque endpoint data address, usable only in temporary RPC wire descriptors. */
  void* data{nullptr};
  /*! \brief Non-null returned tensor handle; FreeHandle releases its remote owner. */
  void* object_handle{nullptr};
  /*! \brief True for session allocations released with FreeTensor, not returned objects. */
  bool owns_allocation{false};
  ~RemoteTensorStorage() {
    try {
      if (object_handle != nullptr) {
        sess->FreeHandle(object_handle);
      } else if (owns_allocation) {
        sess->EnsureOpen();
        sess->FreeTensor(device, data);
      }
    } catch (const ffi::Error&) {
      // A closed transport has already released its remote resources.
    }
  }
};

/*! \brief Logical contiguous tensor metadata over shared remote storage.
 *
 * Views copy metadata and retain the same storage owner. Neither this object
 * nor its descriptor may be exported as a locally usable ffi::Tensor/DLPack.
 */
class RemoteTensorObj : public ffi::Object {
 public:
  /*! \brief Session-bound endpoint device, always from the storage owner's session. */
  RemoteDevice device{ffi::UnsafeInit{}};
  /*! \brief Independently owned logical shape, never borrowed from a remote return frame. */
  ffi::Shape shape;
  /*! \brief Element type for shape/byte-size validation and wire descriptors. */
  DLDataType dtype;
  /*! \brief Byte displacement from storage->data; bounded views accumulate this offset. */
  uint64_t byte_offset{0};
  /*! \brief Shared allocation/object owner; the last view releases the remote resource. */
  std::shared_ptr<RemoteTensorStorage> storage;
  /*! \brief Borrowed descriptor valid while this tensor remains alive during an RPC call.
   * Its data is an endpoint address and shape is local metadata; neither may
   * escape into a locally usable tensor or a DLPack export.
   */
  DLTensor descriptor() const {
    device->sess->EnsureOpen();
    return {storage->data,
            device->endpoint_device,
            static_cast<int>(shape.size()),
            dtype,
            const_cast<int64_t*>(shape.data()),
            nullptr,
            byte_offset};
  }
  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("rpc.RemoteTensor", RemoteTensorObj, ffi::Object);
};
/*! \brief Owning reference to remote metadata and its shared storage lifetime. */
class RemoteTensor : public ffi::ObjectRef {
 public:
  explicit RemoteTensor(ffi::ObjectPtr<RemoteTensorObj> data) : ffi::ObjectRef(std::move(data)) {}
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(RemoteTensor, ffi::ObjectRef, RemoteTensorObj);
};

inline RemoteTensor RemoteTensorFromHandle(std::shared_ptr<RPCSession> sess, const DLTensor& tensor,
                                           void* object_handle) {
  auto n = ffi::make_object<RemoteTensorObj>();
  n->device = MakeRemoteDevice(sess, tensor.device);
  n->shape = ffi::Shape(tensor.shape, tensor.shape + tensor.ndim);
  n->dtype = tensor.dtype;
  n->byte_offset = tensor.byte_offset;
  n->storage = std::make_shared<RemoteTensorStorage>();
  n->storage->sess = std::move(sess);
  n->storage->device = tensor.device;
  n->storage->data = tensor.data;
  n->storage->object_handle = object_handle;
  TVM_FFI_CHECK(ffi::IsContiguous(tensor), ValueError)
      << "RPC requires a contiguous returned tensor";
  return RemoteTensor(n);
}

/*! \brief Session-checked stream wrapper; aliases share one release state. */
class RemoteStreamObj : public ffi::Object {
 public:
  /*! \brief Owning session and exact endpoint device accepted by stream operations. */
  RemoteDevice device{ffi::UnsafeInit{}};
  /*! \brief Shared handle/closure state also retained by the session until release. */
  std::shared_ptr<RPCStreamState> state;
  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("rpc.RemoteStream", RemoteStreamObj, ffi::Object);
};
/*! \brief Owning reference to a session-bound stream alias. */
class RemoteStream : public ffi::ObjectRef {
 public:
  explicit RemoteStream(ffi::ObjectPtr<RemoteStreamObj> data) : ffi::ObjectRef(std::move(data)) {}
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(RemoteStream, ffi::ObjectRef, RemoteStreamObj);
};

inline TVMStreamHandle CheckRemoteStream(const RemoteDevice& dev,
                                         const ffi::Optional<RemoteStream>& stream) {
  dev->sess->EnsureOpen();
  if (!stream.has_value()) return nullptr;
  const auto& s = stream.value();
  TVM_FFI_CHECK(s->device->sess == dev->sess &&
                    s->device->endpoint_device.device_type == dev->endpoint_device.device_type &&
                    s->device->endpoint_device.device_id == dev->endpoint_device.device_id,
                ValueError)
      << "Stream belongs to a different remote device or session";
  TVM_FFI_CHECK(!s->state->closed, ValueError) << "Remote stream is closed";
  return s->state->handle;
}

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_RPC_RPC_REMOTE_H_
