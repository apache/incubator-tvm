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
 * \file rpc_reference.h
 * \brief Common header defining the communication code used in the RPC protocol.
 */
#ifndef TVM_RUNTIME_MINRPC_RPC_REFERENCE_H_
#define TVM_RUNTIME_MINRPC_RPC_REFERENCE_H_

#include <tvm/ffi/container/tensor.h>

namespace tvm {
namespace ffi {
// Forward declare TVM Object to use `Object*` in RPC protocol.
class Object;
}  // namespace ffi

namespace runtime {

/*! \brief The current RPC procotol version. */
constexpr const char* kRPCProtocolVer = "0.9.0";

// When tvm.rpc.server.GetCRTMaxPacketSize global function is not registered.
const uint64_t kRPCMaxTransferSizeBytesDefault = UINT64_MAX;

/*! \brief The RPC code */
enum class RPCCode : int {
  kNone,
  kShutdown,
  kInitServer,
  kCallFunc,
  kReturn,
  kException,
  kCopyFromRemote,
  kCopyToRemote,
  kCopyAck,
  // The following are syscall code that can send over CallRemote
  kSyscallCodeStart,
  kGetGlobalFunc = kSyscallCodeStart,
  kFreeHandle,
  kDevSetDevice,
  kDevGetAttr,
  kDevAllocData,  // Reserved: byte allocation was removed in protocol 0.9.0.
  kDevFreeData,
  kDevStreamSync,
  kCopyAmongRemote,
  kDevAllocDataWithScope,
  kDevCreateStream,
  kDevFreeStream,
  kDevSetStream,
  kDevGetCurrentStream,
};

/*!
 * \brief List of potential error status during rpc communication.
 */
enum class RPCServerStatus : int {
  kSuccess = 0,
  kInvalidTypeCodeObject,
  kInvalidTypeCodeTensor,
  kInvalidDLTensorFieldStride,
  kInvalidDLTensorFieldByteOffset,
  kUnknownTypeIndex,
  kUnknownRPCCode,
  kRPCCodeNotSupported,
  kUnknownRPCSyscall,
  kCheckError,
  kReadError,
  kWriteError,
  kAllocError
};

inline const char* RPCCodeToString(RPCCode code) {
  switch (code) {
    case RPCCode::kShutdown:
      return "kShutdown";
    case RPCCode::kInitServer:
      return "kInitServer";
    case RPCCode::kCallFunc:
      return "kCallFunc";
    case RPCCode::kReturn:
      return "kReturn";
    case RPCCode::kException:
      return "kException";
    case RPCCode::kCopyFromRemote:
      return "kCopyFromRemote";
    case RPCCode::kCopyToRemote:
      return "kCopyToRemote";
    case RPCCode::kCopyAck:
      return "kCopyAck";
    // The following are syscall code that can send over CallRemote
    case RPCCode::kGetGlobalFunc:
      return "kGetGlobalFunc";
    case RPCCode::kFreeHandle:
      return "kFreeHandle";
    case RPCCode::kDevSetDevice:
      return "kDevSetDevice";
    case RPCCode::kDevGetAttr:
      return "kDevGetAttr";
    case RPCCode::kDevAllocData:
      return "kDevAllocData";
    case RPCCode::kDevFreeData:
      return "kDevFreeData";
    case RPCCode::kDevCreateStream:
      return "kDevCreateStream";
    case RPCCode::kDevFreeStream:
      return "kDevFreeStream";
    case RPCCode::kDevStreamSync:
      return "kDevStreamSync";
    case RPCCode::kDevSetStream:
      return "kDevSetStream";
    case RPCCode::kCopyAmongRemote:
      return "kCopyAmongRemote";
    case RPCCode::kDevAllocDataWithScope:
      return "kDevAllocDataWithScope";
    default:
      return "";
  }
}

/*!
 * \brief Convert RPC server status to string.
 * \param status The status.
 * \return The corresponding string.
 */
inline const char* RPCServerStatusToString(RPCServerStatus status) {
  switch (status) {
    case RPCServerStatus::kSuccess:
      return "kSuccess";
    case RPCServerStatus::kInvalidTypeCodeObject:
      return "kInvalidTypeCodeObject";
    case RPCServerStatus::kInvalidTypeCodeTensor:
      return "kInvalidTypeCodeTensor";
    case RPCServerStatus::kInvalidDLTensorFieldStride:
      return "kInvalidDLTensorFieldStride";
    case RPCServerStatus::kInvalidDLTensorFieldByteOffset: {
      return "kInvalidDLTensorFieldByteOffset";
    }
    case RPCServerStatus::kUnknownTypeIndex:
      return "kUnknownTypeIndex";
    case RPCServerStatus::kUnknownRPCCode:
      return "kUnknownRPCCode";
    case RPCServerStatus::kRPCCodeNotSupported:
      return "RPCCodeNotSupported";
    case RPCServerStatus::kUnknownRPCSyscall:
      return "kUnknownRPCSyscall";
    case RPCServerStatus::kCheckError:
      return "kCheckError";
    case RPCServerStatus::kReadError:
      return "kReadError";
    case RPCServerStatus::kWriteError:
      return "kWriteError";
    case RPCServerStatus::kAllocError:
      return "kAllocError";
    default:
      return "";
  }
}

/*!
 * \brief Reference implementation of the communication protocol.
 *
 * \note The implementation is intentionally written via template
 *       so it can be used in a dependency free setting.
 *
 * \sa src/runtime/rpc/device/min_rpc_server.h
 */
struct RPCReference {
  /*!
   * \brief Auxiliary class to get the packed sequence.
   * \tparam TChannel The channel to throw errror.
   */
  template <typename TChannel>
  struct PackedSeqNumBytesGetter {
   public:
    explicit PackedSeqNumBytesGetter(TChannel* channel) : channel_(channel) {}

    template <typename T>
    void Write(const T& value) {
      num_bytes_ += sizeof(T);
    }

    template <typename T>
    void WriteArray(const T* value, size_t num) {
      num_bytes_ += sizeof(T) * num;
    }

    void WriteFFIAny(const TVMFFIAny* obj) { num_bytes_ += channel_->GetFFIAnyProtocolBytes(obj); }

    void ThrowError(RPCServerStatus status) { channel_->ThrowError(status); }

    uint64_t num_bytes() const { return num_bytes_; }

   private:
    TChannel* channel_;
    uint64_t num_bytes_{0};
  };

  /*!
   * \return the length of the str.
   * \param str the string.
   * \return The length.
   */
  static uint64_t StrLength(const char* str) {
    uint64_t len = 0;
    while (str[len] != '\0') ++len;
    return len;
  }

  /*!
   * \brief Get the total nbytes to be sent in the packed sequence.
   *
   * \param arg_values The values to be sent over.
   * \param type_codes The type codes to be sent over.
   * \param num_args Number of argument.
   * \param client_mode Whether it is a client to server call.
   * \param channel The communication channel handler.
   * \tparam TChannel The type of the communication channel.
   * \return The total number of bytes.
   */
  template <typename TChannel>
  static uint64_t PackedSeqGetNumBytes(const TVMFFIAny* packed_args, int num_args, bool client_mode,
                                       TChannel* channel) {
    PackedSeqNumBytesGetter<TChannel> getter(channel);
    SendPackedSeq(packed_args, num_args, client_mode, &getter);
    return getter.num_bytes();
  }

  template <typename TChannelPtr>
  static void SendDLTensor(TChannelPtr channel, DLTensor* arr) {
    DLDevice dev;
    uint64_t data;
    // When we return Tensor, we directly return
    // the space and the context
    // The client will be further wrapping
    dev = arr->device;
    data = reinterpret_cast<uint64_t>(arr->data);
    channel->Write(data);
    channel->Write(dev);
    channel->Write(arr->ndim);
    channel->Write(arr->dtype);
    channel->WriteArray(arr->shape, arr->ndim);
    if (!ffi::IsContiguous(*arr)) {
      channel->ThrowError(RPCServerStatus::kInvalidDLTensorFieldStride);
    }
    channel->Write(arr->byte_offset);
    return;
  }

  template <typename TChannelPtr>
  static DLTensor* ReceiveDLTensor(TChannelPtr channel) {
    uint64_t handle;
    channel->Read(&handle);
    DLTensor* arr = channel->template ArenaAlloc<DLTensor>(1);
    DLTensor& tensor = *arr;
    tensor.data = reinterpret_cast<void*>(handle);
    channel->Read(&(tensor.device));
    channel->Read(&(tensor.ndim));
    channel->Read(&(tensor.dtype));
    tensor.shape = channel->template ArenaAlloc<int64_t>(tensor.ndim);
    channel->ReadArray(tensor.shape, tensor.ndim);
    tensor.strides = nullptr;
    channel->Read(&(tensor.byte_offset));
    return arr;
  }

  /*!
   * \brief Send packed argument sequnce to the other peer.
   *
   * This function serves as the foundational communication primitive between peers.
   *
   * TVMValue sequence encoding protocol(according to the type):
   *
   * - int/float/uint/bytes/str: Serialize all content.
   * - DLTensor: send meta-data, send data handle as opaque handle(via uint64_t)
   * - OpaqueHandle: send as uint64_t
   * - ModuleHandle, PackedFuncHandle: send as uint64_t,
   *   The support to Module/PackedFuncHandle are reserved for arguments
   *   in the CallFunc from a client to server only.
   *   Note that we cannot simply take these argument out(as the handle)
   *   refers to a value on the remote(instead of local).
   *
   * \param packed_args The values to be sent over.
   * \param num_args Number of argument.
   * \param client_mode Whether it is a client to server call.
   * \param channel The communication channel handler.
   * \tparam TChannel The type of the communication channel.
   */
  template <typename TChannel>
  static void SendPackedSeq(const TVMFFIAny* packed_args, int num_args, bool client_mode,
                            TChannel* channel) {
    channel->Write(num_args);

    // Argument packing.
    for (int i = 0; i < num_args; ++i) {
      int32_t type_index = packed_args[i].type_index;
      channel->template Write<int32_t>(type_index);
      switch (type_index) {
        case ffi::TypeIndex::kTVMFFINone: {
          break;
        }
        case ffi::TypeIndex::kTVMFFIBool:
        case ffi::TypeIndex::kTVMFFIInt:
        case ffi::TypeIndex::kTVMFFIFloat: {
          channel->template Write<int64_t>(packed_args[i].v_int64);
          break;
        }
        case ffi::TypeIndex::kTVMFFIOpaquePtr: {
          // always send handle in 64 bit.
          uint64_t handle = reinterpret_cast<uint64_t>(packed_args[i].v_ptr);
          channel->template Write<int64_t>(handle);
          break;
        }
        case ffi::TypeIndex::kTVMFFIDataType: {
          channel->Write(packed_args[i].v_dtype);
          // padding
          int32_t padding = 0;
          channel->template Write<int32_t>(padding);
          break;
        }
        case ffi::TypeIndex::kTVMFFIDevice: {
          channel->Write(packed_args[i].v_device);
          break;
        }

        case ffi::TypeIndex::kTVMFFIFunction:
        case ffi::TypeIndex::kTVMFFIModule: {
          if (!client_mode) {
            channel->ThrowError(RPCServerStatus::kInvalidTypeCodeObject);
          }
          // always send handle in 64 bit.
          uint64_t handle = reinterpret_cast<uint64_t>(packed_args[i].v_obj);
          channel->Write(handle);
          break;
        }

        case ffi::TypeIndex::kTVMFFITensor: {
          channel->ThrowError(RPCServerStatus::kInvalidTypeCodeTensor);
          break;
        }
        case ffi::TypeIndex::kTVMFFIDLTensorPtr: {
          DLTensor* arr = static_cast<DLTensor*>(packed_args[i].v_ptr);
          SendDLTensor(channel, arr);
          break;
        }
        case ffi::TypeIndex::kTVMFFIRawStr: {
          const char* s = packed_args[i].v_c_str;
          uint64_t len = StrLength(s);
          channel->Write(len);
          channel->WriteArray(s, len);
          break;
        }
        case ffi::TypeIndex::kTVMFFIByteArrayPtr: {
          TVMFFIByteArray* bytes = static_cast<TVMFFIByteArray*>(packed_args[i].v_ptr);
          uint64_t len = bytes->size;
          channel->Write(len);
          channel->WriteArray(bytes->data, len);
          break;
        }
        default: {
          channel->WriteFFIAny(&(packed_args[i]));
          break;
        }
      }
    }
  }

  /*!
   * \brief Receive packed seq from the channel.
   *
   * \param out_packed_args The values to be received.
   * \param out_num_args Number of argument.
   * \param channel The communication channel handler.
   * \tparam TChannel The type of the communication channel.
   * \note The temporary space are populated via an arena inside channel.
   */
  template <typename TChannel>
  static void RecvPackedSeq(TVMFFIAny** out_packed_args, int32_t* out_num_args, TChannel* channel) {
    // receive number of args
    int32_t num_args;
    channel->Read(&num_args);
    *out_num_args = num_args;
    if (num_args == 0) {
      *out_packed_args = nullptr;
      return;
    }

    TVMFFIAny* packed_args = channel->template ArenaAlloc<TVMFFIAny>(num_args);
    *out_packed_args = packed_args;

    // receive arguments
    for (int32_t i = 0; i < num_args; ++i) {
      int32_t type_index;
      channel->Read(&type_index);
      packed_args[i].type_index = type_index;
      packed_args[i].zero_padding = 0;
      // clear to ensure compact for 32 bit platform
      packed_args[i].v_int64 = 0;
      switch (type_index) {
        case ffi::TypeIndex::kTVMFFINone: {
          break;
        }
        case ffi::TypeIndex::kTVMFFIBool:
        case ffi::TypeIndex::kTVMFFIInt:
        case ffi::TypeIndex::kTVMFFIFloat: {
          channel->template Read<int64_t>(&(packed_args[i].v_int64));
          break;
        }
        case ffi::TypeIndex::kTVMFFIOpaquePtr: {
          uint64_t handle;
          channel->Read(&handle);
          packed_args[i].v_ptr = reinterpret_cast<void*>(handle);
          break;
        }
        case ffi::TypeIndex::kTVMFFIDataType: {
          channel->Read(&(packed_args[i].v_dtype));
          int32_t padding = 0;
          channel->template Read<int32_t>(&padding);
          break;
        }
        case ffi::TypeIndex::kTVMFFIDevice: {
          channel->Read(&(packed_args[i].v_device));
          break;
        }
        case ffi::TypeIndex::kTVMFFIFunction:
        case ffi::TypeIndex::kTVMFFIModule: {
          // always send handle in 64 bit.
          uint64_t handle;
          channel->Read(&handle);
          packed_args[i].v_obj = reinterpret_cast<TVMFFIObject*>(handle);
          break;
        }
        case ffi::TypeIndex::kTVMFFIRawStr: {
          uint64_t len;
          channel->Read(&len);
          char* str = channel->template ArenaAlloc<char>(len + 1);
          str[len] = '\0';
          channel->ReadArray(str, len);
          packed_args[i].v_c_str = str;
          break;
        }
        case ffi::TypeIndex::kTVMFFIByteArrayPtr: {
          uint64_t len;
          channel->Read(&len);
          TVMFFIByteArray* arr = channel->template ArenaAlloc<TVMFFIByteArray>(1);
          char* data = channel->template ArenaAlloc<char>(len);
          arr->size = len;
          arr->data = data;
          channel->ReadArray(data, len);
          packed_args[i].v_ptr = arr;
          break;
        }
        case ffi::TypeIndex::kTVMFFIDLTensorPtr: {
          packed_args[i].v_ptr = ReceiveDLTensor(channel);
          break;
        }
        default: {
          if (type_index >= ffi::TypeIndex::kTVMFFIStaticObjectBegin ||
              type_index == ffi::TypeIndex::kTVMFFISmallStr ||
              type_index == ffi::TypeIndex::kTVMFFISmallBytes) {
            channel->ReadFFIAny(&(packed_args[i]));
          } else {
            channel->ThrowError(RPCServerStatus::kUnknownTypeIndex);
          }
          break;
        }
      }
    }
  }

  /*!
   * \brief Return an exception packet.
   *
   * \param msg The error message.
   * \param channel The communication channel handler.
   * \tparam TChannel The type of the communication channel.
   */
  template <typename TChannel>
  static void ReturnException(const char* msg, TChannel* channel) {
    RPCCode code = RPCCode::kException;
    int32_t num_args = 1;
    int32_t type_index = ffi::TypeIndex::kTVMFFIRawStr;
    uint64_t len = StrLength(msg);

    uint64_t packet_nbytes =
        sizeof(code) + sizeof(num_args) + sizeof(type_index) + sizeof(len) + len;

    channel->MessageStart(packet_nbytes);
    channel->Write(packet_nbytes);
    channel->Write(code);
    channel->Write(num_args);
    channel->Write(type_index);
    channel->Write(len);
    channel->WriteArray(msg, len);
    channel->MessageDone();
  }

  /*!
   * \brief Return a normal packed sequence packet.
   *
   * \param msg The error message.
   * \param channel The communication channel handler.
   * \tparam TChannel The type of the communication channel.
   */
  template <typename TChannel>
  static void ReturnPackedSeq(const TVMFFIAny* packed_args, int num_args, TChannel* channel) {
    RPCCode code = RPCCode::kReturn;

    uint64_t packet_nbytes =
        sizeof(code) + PackedSeqGetNumBytes(packed_args, num_args, false, channel);

    channel->MessageStart(packet_nbytes);
    channel->Write(packet_nbytes);
    channel->Write(code);
    SendPackedSeq(packed_args, num_args, false, channel);
    channel->MessageDone();
  }

  /*!
   * \brief Return a null(void) packet.
   *
   * \param channel The communication channel handler.
   * \tparam TChannel The type of the communication channel.
   */
  template <typename TChannel>
  static void ReturnVoid(TChannel* channel) {
    int32_t num_args = 1;
    int32_t type_index = ffi::TypeIndex::kTVMFFINone;
    RPCCode code = RPCCode::kReturn;

    uint64_t packet_nbytes = sizeof(code) + sizeof(num_args) + sizeof(type_index);

    channel->MessageStart(packet_nbytes);
    channel->Write(packet_nbytes);
    channel->Write(code);
    channel->Write(num_args);
    channel->Write(type_index);
    channel->MessageDone();
  }
};

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_MINRPC_RPC_REFERENCE_H_
