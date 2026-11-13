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
 * \file minrpc_server.h
 * \brief Minimum RPC server implementation,
 *        redirects all the calls to C runtime API.
 *
 * \note This file do not depend on c++ std or c std,
 *       and only depends on TVM's C runtime API.
 */
#ifndef TVM_RUNTIME_MINRPC_MINRPC_SERVER_H_
#define TVM_RUNTIME_MINRPC_MINRPC_SERVER_H_

#include <tvm/ffi/c_api.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/base.h>

#include <cstring>
#include <memory>
#include <utility>

#include "../../../support/arena.h"
#include "rpc_reference.h"

namespace tvm {
namespace runtime {
namespace details {
template <typename TIOHandler>
class PageAllocator;
}  // namespace details
/*!
 * \brief A minimum RPC server that only depends on the tvm C runtime..
 *
 *  All the dependencies are provided by the io arguments.
 *
 * \tparam TIOHandler IO provider to provide io handling.
 *         An IOHandler needs to provide the following functions:
 *         - PosixWrite, PosixRead, Close: posix style, read, write, close API.
 *         - MessageStart(num_bytes), MessageDone(): framing APIs.
 *         - Exit: exit with status code.
 */
template <typename TIOHandler, template <typename> class Allocator = details::PageAllocator>
class MinRPCServer {
 public:
  using PageAllocator = Allocator<TIOHandler>;

  using FServerHandler = ffi::TypedFunction<int(TVMFFIByteArray*, int)>;

  explicit MinRPCServer(TIOHandler* io) : io_(io), arena_(PageAllocator(io_)) {
    auto fsend = ffi::Function::FromTyped([this](TVMFFIByteArray* bytes) {
      return io_->PosixWrite(reinterpret_cast<const uint8_t*>(bytes->data), bytes->size);
    });
    auto fcreate = tvm::ffi::Function::GetGlobalRequired("rpc.CreateEventDrivenServer");
    ffi::Any value = fcreate(fsend, "MinRPCServer", "");
    fserver_handler_ = value.cast<FServerHandler>();
  }

  /*!
   * \brief Process a single request.
   *
   * \return true when the server should continue processing requests. false when it should be
   *  shutdown.
   */
  bool ProcessOnePacket() {
    uint64_t packet_len;
    arena_.RecycleAll();
    allow_clean_shutdown_ = true;
    Read(&packet_len);
    if (packet_len == 0) return true;
    char* read_buffer = this->ArenaAlloc<char>(sizeof(uint64_t) + packet_len);
    // copy header into read buffer
    std::memcpy(read_buffer, &packet_len, sizeof(uint64_t));
    // read the rest of the packet
    ReadRawBytes(read_buffer + sizeof(uint64_t), packet_len);
    // setup write flags
    int write_flags = 3;
    TVMFFIByteArray read_bytes{read_buffer, sizeof(uint64_t) + static_cast<size_t>(packet_len)};
    int status = fserver_handler_(&read_bytes, write_flags);

    while (status == 2) {
      TVMFFIByteArray write_bytes{nullptr, 0};
      // continue call handler until it have nothing to write
      status = fserver_handler_(&write_bytes, write_flags);
      if (status == 0) {
        this->Shutdown();
        return false;
      }
    }
    return true;
  }

  void Shutdown() {
    arena_.FreeAll();
    io_->Close();
  }

  void ThrowError(RPCServerStatus code, RPCCode info = RPCCode::kNone) {
    io_->Exit(static_cast<int>(code));
  }

  template <typename T>
  T* ArenaAlloc(int count) {
    static_assert(std::is_trivial<T>::value && std::is_standard_layout<T>::value,
                  "need to be trival");
    return arena_.template allocate_<T>(count);
  }

  template <typename T>
  void Read(T* data) {
    static_assert(std::is_trivial<T>::value && std::is_standard_layout<T>::value,
                  "need to be trival");
    ReadRawBytes(data, sizeof(T));
  }

 private:
  void ReadRawBytes(void* data, size_t size) {
    uint8_t* buf = static_cast<uint8_t*>(data);
    size_t ndone = 0;
    while (ndone < size) {
      ssize_t ret = io_->PosixRead(buf, size - ndone);
      if (ret == 0) {
        if (allow_clean_shutdown_) {
          Shutdown();
          io_->Exit(0);
        } else {
          this->ThrowError(RPCServerStatus::kReadError);
        }
      }
      if (ret == -1) {
        this->ThrowError(RPCServerStatus::kReadError);
      }
      ndone += ret;
      buf += ret;
    }
  }

  /*! \brief server handler. */
  FServerHandler fserver_handler_;
  /*! \brief IO handler. */
  TIOHandler* io_;
  /*! \brief internal arena. */
  support::GenericArena<PageAllocator> arena_;
  /*! \brief Whether we are in a state that allows clean shutdown. */
  bool allow_clean_shutdown_{true};
};

namespace details {
// Internal allocator that redirects alloc to TVM's C API.
template <typename TIOHandler>
class PageAllocator {
 public:
  using ArenaPageHeader = tvm::support::ArenaPageHeader;

  explicit PageAllocator(TIOHandler* io) : io_(io) {}

  ArenaPageHeader* allocate(size_t min_size) {
    size_t npages = ((min_size + kPageSize - 1) / kPageSize);
    void* data = malloc(npages * kPageSize);

    if (data == nullptr) {
      io_->Exit(static_cast<int>(RPCServerStatus::kAllocError));
    }

    ArenaPageHeader* header = static_cast<ArenaPageHeader*>(data);
    header->size = npages * kPageSize;
    header->offset = sizeof(ArenaPageHeader);
    return header;
  }

  void deallocate(ArenaPageHeader* page) { free(page); }

  static const constexpr int kPageSize = 2 << 10;
  static const constexpr int kPageAlign = 8;

 private:
  TIOHandler* io_;
};
}  // namespace details

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_MINRPC_MINRPC_SERVER_H_
