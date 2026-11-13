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
 * \file rpc_socket_impl.cc
 * \brief Socket based RPC implementation.
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/extra_base.h>
#include <tvm/support/io.h>
#include <tvm/support/serializer.h>

#include <memory>

#include "../../support/socket.h"
#include "rpc_endpoint.h"
#include "rpc_local_session.h"
#include "rpc_session.h"

namespace tvm {
namespace runtime {

class SockChannel final : public RPCChannel {
 public:
  explicit SockChannel(support::TCPSocket sock) : sock_(sock) {}
  ~SockChannel() {
    try {
      // BadSocket can throw
      if (!sock_.BadSocket()) {
        sock_.Close();
      }
    } catch (...) {
    }
  }
  size_t Send(const void* data, size_t size) final {
    ssize_t n = sock_.Send(data, size);
    if (n == -1) {
      support::Socket::Error("SockChannel::Send");
    }
    return static_cast<size_t>(n);
  }
  size_t Recv(void* data, size_t size) final {
    ssize_t n = sock_.Recv(data, size);
    if (n == -1) {
      support::Socket::Error("SockChannel::Recv");
    }
    return static_cast<size_t>(n);
  }

 private:
  support::TCPSocket sock_;
};

std::shared_ptr<RPCEndpoint> RPCConnect(std::string url, int port, std::string key,
                                        bool enable_logging, ffi::PackedArgs init_seq) {
  support::TCPSocket sock;
  support::SockAddr addr(url.c_str(), port);
  sock.Create(addr.ss_family());
  TVM_FFI_ICHECK(sock.Connect(addr)) << "Connect to " << addr.AsString() << " failed";
  // hand shake
  std::ostringstream os;
  int code = kRPCMagic;
  int keylen = static_cast<int>(key.length());
  TVM_FFI_ICHECK_EQ(sock.SendAll(&code, sizeof(code)), sizeof(code));
  TVM_FFI_ICHECK_EQ(sock.SendAll(&keylen, sizeof(keylen)), sizeof(keylen));
  if (keylen != 0) {
    TVM_FFI_ICHECK_EQ(sock.SendAll(key.c_str(), keylen), keylen);
  }
  TVM_FFI_ICHECK_EQ(sock.RecvAll(&code, sizeof(code)), sizeof(code));
  if (code == kRPCMagic + 2) {
    sock.Close();
    TVM_FFI_THROW(InternalError) << "URL " << url << ":" << port
                                 << " cannot find server that matches key=" << key;
  } else if (code == kRPCMagic + 1) {
    sock.Close();
    TVM_FFI_THROW(InternalError) << "URL " << url << ":" << port
                                 << " server already have key=" << key;
  } else if (code != kRPCMagic) {
    sock.Close();
    TVM_FFI_THROW(InternalError) << "URL " << url << ":" << port << " is not TVM RPC server";
  }
  TVM_FFI_ICHECK_EQ(sock.RecvAll(&keylen, sizeof(keylen)), sizeof(keylen));
  std::string remote_key;
  if (keylen != 0) {
    remote_key.resize(keylen);
    TVM_FFI_ICHECK_EQ(sock.RecvAll(&remote_key[0], keylen), keylen);
  }

  std::unique_ptr<RPCChannel> channel = std::make_unique<SockChannel>(sock);
  auto endpt = RPCEndpoint::Create(std::move(channel), key, remote_key);

  endpt->InitRemoteSession(init_seq);
  return endpt;
}

ffi::Module RPCClientConnect(std::string url, int port, std::string key, bool enable_logging,
                             ffi::PackedArgs init_seq) {
  auto endpt = RPCConnect(url, port, "client:" + key, enable_logging, init_seq);
  return CreateRPCSessionModule(CreateClientSession(endpt));
}

// TVM_RUNTIME_EXTRA_DLL needed for MSVC
TVM_RUNTIME_EXTRA_DLL void RPCServerLoop(int sockfd) {
  support::TCPSocket sock(static_cast<support::TCPSocket::SockType>(sockfd));
  RPCEndpoint::Create(std::make_unique<SockChannel>(sock), "SockServerLoop", "")->ServerLoop();
}

void RPCServerLoop(ffi::Function fsend, ffi::Function frecv) {
  RPCEndpoint::Create(std::make_unique<CallbackChannel>(fsend, frecv), "SockServerLoop", "")
      ->ServerLoop();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("rpc.Connect",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    auto url = args[0].cast<std::string>();
                    int port = args[1].cast<int>();
                    auto key = args[2].cast<std::string>();
                    bool enable_logging = args[3].cast<bool>();
                    *rv = RPCClientConnect(url, port, key, enable_logging, args.Slice(4));
                  })
      .def_packed("rpc.ServerLoop", [](ffi::PackedArgs args, ffi::Any* rv) {
        if (auto opt_int = args[0].as<int64_t>()) {
          RPCServerLoop(opt_int.value());
        } else {
          RPCServerLoop(args[0].cast<tvm::ffi::Function>(), args[1].cast<tvm::ffi::Function>());
        }
      });
}

class SimpleSockHandler : public support::Stream {
  // Things that will interface with user directly.
 public:
  explicit SimpleSockHandler(int sockfd)
      : sock_(static_cast<support::TCPSocket::SockType>(sockfd)) {}
  using support::Stream::Read;
  using support::Stream::ReadArray;
  using support::Stream::Write;
  using support::Stream::WriteArray;

  // Unused here, implemented for microTVM framing layer.
  void MessageStart(uint64_t packet_nbytes) {}
  void MessageDone() {}

  // Internal supporting.
  // Override methods that inherited from support::Stream.
 private:
  size_t Read(void* data, size_t size) final { return sock_.Recv(data, size); }
  size_t Write(const void* data, size_t size) final { return sock_.Send(data, size); }

  // Things of current class.
 private:
  support::TCPSocket sock_;
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("rpc.ReturnException", [](int sockfd, ffi::String msg) {
    auto handler = SimpleSockHandler(sockfd);
    RPCReference::ReturnException(msg.c_str(), &handler);
    return;
  });
}

}  // namespace runtime
}  // namespace tvm
