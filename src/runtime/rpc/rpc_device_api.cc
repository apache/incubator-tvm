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

/*! \file rpc_device_api.cc
 *  \brief Explicit device and tensor operations owned by RPC.
 */
#include <limits>

#include "rpc_remote.h"

namespace tvm {
namespace runtime {

// Check before allocation or view arithmetic; ffi::GetDataSize assumes valid dimensions.
static uint64_t RemoteTensorBytes(const ffi::Shape& shape, DLDataType dtype) {
  TVM_FFI_CHECK(dtype.bits > 0 && dtype.lanes > 0, ValueError) << "Invalid tensor dtype";
  uint64_t size = 1;
  for (int64_t dim : shape) {
    TVM_FFI_CHECK_GE(dim, 0, ValueError) << "Negative tensor dimension";
    TVM_FFI_CHECK(dim == 0 || size <= std::numeric_limits<uint64_t>::max() / dim, ValueError)
        << "Tensor size overflow";
    size *= dim;
  }
  uint64_t element_bits = static_cast<uint64_t>(dtype.bits) * dtype.lanes;
  TVM_FFI_CHECK_LE(size, (std::numeric_limits<uint64_t>::max() - 7) / element_bits, ValueError)
      << "Tensor size overflow";
  return (size * element_bits + 7) / 8;
}

static RemoteTensor RemoteEmpty(ffi::Shape shape, DLDataType dtype, RemoteDevice dev,
                                ffi::Optional<MemoryScope> scope) {
  RemoteTensorBytes(shape, dtype);
  auto n = ffi::make_object<RemoteTensorObj>();
  n->device = dev;
  n->shape = shape;
  n->dtype = dtype;
  auto storage = std::make_shared<RemoteTensorStorage>();
  storage->sess = dev->sess;
  storage->device = dev->endpoint_device;
  storage->data = dev->session()->AllocTensor(dev->endpoint_device, shape, dtype, scope);
  storage->owns_allocation = true;
  n->storage = std::move(storage);
  return RemoteTensor(n);
}

static void CopyRemote(RemoteTensor from, RemoteTensor to, ffi::Optional<RemoteStream> stream) {
  TVM_FFI_CHECK(from->device->sess == to->device->sess, ValueError)
      << "Cannot copy across different remote sessions";
  TVM_FFI_CHECK_EQ(RemoteTensorBytes(from->shape, from->dtype),
                   RemoteTensorBytes(to->shape, to->dtype), ValueError)
      << "Tensor copy size mismatch";
  auto src = from->descriptor();
  auto dst = to->descriptor();
  auto dev = src.device.device_type == kDLCPU ? to->device : from->device;
  auto handle = CheckRemoteStream(dev, stream);
  dev->session()->CopyDataFromTo(&src, &dst, handle);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<RemoteDeviceObj>().def_ro("endpoint_device", &RemoteDeviceObj::endpoint_device);
  refl::ObjectDef<RemoteTensorObj>()
      .def_ro("device", &RemoteTensorObj::device)
      .def_ro("shape", &RemoteTensorObj::shape)
      .def_ro("dtype", &RemoteTensorObj::dtype);
  refl::ObjectDef<RemoteStreamObj>();
  refl::GlobalDef()
      .def("rpc.RemoteDevice",
           [](ffi::Module mod, Device dev) {
             return MakeRemoteDevice(RPCModuleGetSession(mod), dev);
           })
      .def("rpc.DeviceGetAttr",
           [](RemoteDevice dev, ffi::String attr) {
             dev->sess->EnsureOpen();
             return dev->sess->GetAttr(dev->endpoint_device, attr);
           })
      .def("rpc.DeviceSetDevice",
           [](RemoteDevice dev) { dev->session()->SetDevice(dev->endpoint_device); })
      .def("rpc.DeviceCreateStream",
           [](RemoteDevice dev) {
             auto n = ffi::make_object<RemoteStreamObj>();
             n->device = dev;
             n->state = dev->sess->TrackStream(
                 dev->endpoint_device, dev->session()->CreateStream(dev->endpoint_device), true);
             return RemoteStream(n);
           })
      .def("rpc.DeviceCurrentStream",
           [](RemoteDevice dev) {
             auto n = ffi::make_object<RemoteStreamObj>();
             n->device = dev;
             n->state = dev->sess->TrackStream(
                 dev->endpoint_device, dev->session()->GetCurrentStream(dev->endpoint_device),
                 false);
             return RemoteStream(n);
           })
      .def("rpc.DeviceFreeStream",
           [](RemoteDevice dev, RemoteStream stream) {
             CheckRemoteStream(dev, stream);
             dev->sess->FreeStream(stream->state);
           })
      .def("rpc.DeviceSetStream",
           [](RemoteDevice dev, ffi::Optional<RemoteStream> stream) {
             auto handle = CheckRemoteStream(dev, stream);
             dev->session()->SetStream(dev->endpoint_device, handle);
           })
      .def("rpc.DeviceSync",
           [](RemoteDevice dev, ffi::Optional<RemoteStream> stream) {
             auto handle = CheckRemoteStream(dev, stream);
             dev->session()->StreamSync(dev->endpoint_device, handle);
           })
      .def("rpc.Empty", RemoteEmpty)
      .def("rpc.CopyFromTo", CopyRemote)
      .def("rpc.CopyFromLocal",
           [](ffi::TensorView from, RemoteTensor to) {
             TVM_FFI_CHECK(from.device().device_type == kDLCPU && from.IsContiguous(), ValueError)
                 << "RPC transfer requires a contiguous local CPU tensor";
             TVM_FFI_CHECK_EQ(ffi::GetDataSize(from), RemoteTensorBytes(to->shape, to->dtype),
                              ValueError)
                 << "Tensor copy size mismatch";
             auto dst = to->descriptor();
             void* bytes = static_cast<char*>(from.data_ptr()) + from.byte_offset();
             to->device->sess->CopyToRemote(bytes, &dst, ffi::GetDataSize(from));
           })
      .def("rpc.CopyToLocal",
           [](RemoteTensor from, ffi::TensorView to) {
             TVM_FFI_CHECK(to.device().device_type == kDLCPU && to.IsContiguous(), ValueError)
                 << "RPC transfer requires a contiguous local CPU tensor";
             TVM_FFI_CHECK_EQ(RemoteTensorBytes(from->shape, from->dtype), ffi::GetDataSize(to),
                              ValueError)
                 << "Tensor copy size mismatch";
             auto src = from->descriptor();
             void* bytes = static_cast<char*>(to.data_ptr()) + to.byte_offset();
             from->device->sess->CopyFromRemote(&src, bytes, ffi::GetDataSize(to));
           })
      .def("rpc.CreateView", [](RemoteTensor source, ffi::Shape shape, DLDataType dtype,
                                int64_t offset) {
        source->device->sess->EnsureOpen();
        uint64_t source_bytes = RemoteTensorBytes(source->shape, source->dtype);
        uint64_t view_bytes = RemoteTensorBytes(shape, dtype);
        TVM_FFI_CHECK(offset >= 0 && static_cast<uint64_t>(offset) <= source_bytes &&
                          view_bytes <= source_bytes - offset,
                      ValueError)
            << "Remote tensor view exceeds source bounds";
        TVM_FFI_CHECK_LE(static_cast<uint64_t>(offset),
                         std::numeric_limits<uint64_t>::max() - source->byte_offset, ValueError)
            << "Remote tensor view offset overflow";
        auto n = ffi::make_object<RemoteTensorObj>();
        n->device = source->device;
        n->shape = shape;
        n->dtype = dtype;
        n->byte_offset = source->byte_offset + offset;
        n->storage = source->storage;
        return RemoteTensor(n);
      });
}
}  // namespace runtime
}  // namespace tvm
