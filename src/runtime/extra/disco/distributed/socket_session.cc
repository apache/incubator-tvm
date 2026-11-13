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
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>

#include <numeric>

#include "../../../../support/socket.h"
#include "../bcast_session.h"
#include "../message_queue.h"

namespace tvm {
namespace runtime {

using namespace tvm::support;

enum class DiscoSocketAction {
  kShutdown = static_cast<int>(DiscoAction::kShutDown),
  kSend,
  kReceive,
};

class DiscoSocketChannel : public DiscoChannel {
 public:
  explicit DiscoSocketChannel(const TCPSocket& socket)
      : socket_(socket), message_queue_(&socket_) {}

  DiscoSocketChannel(DiscoSocketChannel&& other) = delete;
  DiscoSocketChannel(const DiscoSocketChannel& other) = delete;
  void Send(const ffi::PackedArgs& args) { message_queue_.Send(args); }
  ffi::PackedArgs Recv() { return message_queue_.Recv(); }
  void Reply(const ffi::PackedArgs& args) { message_queue_.Send(args); }
  ffi::PackedArgs RecvReply() { return message_queue_.Recv(); }

 private:
  TCPSocket socket_;
  DiscoStreamMessageQueue message_queue_;
};

class SocketSessionObj : public BcastSessionObj {
 public:
  explicit SocketSessionObj(int num_nodes, int num_workers_per_node, int num_groups,
                            const ffi::String& host, int port)
      : num_nodes_(num_nodes), num_workers_per_node_(num_workers_per_node) {
    const auto f_create_local_session =
        tvm::ffi::Function::GetGlobal("runtime.disco.create_socket_session_local_workers");
    TVM_FFI_ICHECK(f_create_local_session.has_value())
        << "Cannot find function runtime.disco.create_socket_session_local_workers";
    local_session_ = ((*f_create_local_session)(num_workers_per_node)).cast<BcastSession>();
    DRef f_init_workers =
        local_session_->GetGlobalFunc("runtime.disco.socket_session_init_workers");
    local_session_->CallPacked(f_init_workers, num_nodes_, /*node_id=*/0, num_groups,
                               num_workers_per_node_);

    Socket::Startup();
    SockAddr addr{host.c_str(), port};
    socket_.Create(addr.ss_family());
    socket_.SetKeepAlive(true);
    socket_.Bind(addr);
    socket_.Listen();
    LOG(INFO) << "SocketSession controller listening on " << host << ":" << port;

    AnyView packed_args[4];
    packed_args[0] = num_nodes;
    packed_args[1] = num_workers_per_node;
    packed_args[2] = num_groups;

    for (int i = 0; i + 1 < num_nodes; ++i) {
      SockAddr addr;
      remote_sockets_.push_back(socket_.Accept(&addr));
      remote_channels_.emplace_back(std::make_unique<DiscoSocketChannel>(remote_sockets_.back()));
      packed_args[3] = i + 1;
      // Send metadata to each remote node:
      //  - num_nodes
      //  - num_workers_per_node
      //  - num_groups
      //  - node_id
      remote_channels_.back()->Send(ffi::PackedArgs(packed_args, 4));
      LOG(INFO) << "Remote node " << addr.AsString() << " connected";
    }
  }

  int64_t GetNumWorkers() final { return num_nodes_ * num_workers_per_node_; }

  ffi::Any DebugGetFromRemote(int64_t reg_id, int worker_id) final {
    int node_id = worker_id / num_workers_per_node_;
    if (node_id == 0) {
      return local_session_->DebugGetFromRemote(reg_id, worker_id);
    } else {
      AnyView packed_args[5];
      ffi::PackedArgs::Fill(packed_args, static_cast<int>(DiscoSocketAction::kSend), worker_id,
                            static_cast<int>(DiscoAction::kDebugGetFromRemote), reg_id, worker_id);
      remote_channels_[node_id - 1]->Send(ffi::PackedArgs(packed_args, 5));
      ffi::PackedArgs args = this->RecvReplyPacked(worker_id);
      TVM_FFI_ICHECK_EQ(args.size(), 2);
      TVM_FFI_ICHECK(static_cast<DiscoAction>(args[0].cast<int>()) ==
                     DiscoAction::kDebugGetFromRemote);
      ffi::Any result;
      result = args[1];
      return result;
    }
  }

  void DebugSetRegister(int64_t reg_id, AnyView value, int worker_id) final {
    int node_id = worker_id / num_workers_per_node_;
    if (node_id == 0) {
      local_session_->DebugSetRegister(reg_id, value, worker_id);
    } else {
      ffi::ObjectRef wrapped{nullptr};
      if (auto opt_obj = value.as<ffi::ObjectRef>()) {
        wrapped = DiscoDebugObject::Wrap(value);
        value = wrapped;
      }
      {
        AnyView packed_args[6];
        ffi::PackedArgs::Fill(packed_args, static_cast<int>(DiscoSocketAction::kSend), worker_id,
                              static_cast<int>(DiscoAction::kDebugSetRegister), reg_id, worker_id,
                              value);
        remote_channels_[node_id - 1]->Send(ffi::PackedArgs(packed_args, 6));
      }
      ffi::Any result;
      ffi::PackedArgs args = this->RecvReplyPacked(worker_id);
      TVM_FFI_ICHECK_EQ(args.size(), 1);
      TVM_FFI_ICHECK(static_cast<DiscoAction>(args[0].cast<int>()) ==
                     DiscoAction::kDebugSetRegister);
    }
  }

  void BroadcastPacked(const ffi::PackedArgs& args) final {
    local_session_->BroadcastPacked(args);
    std::vector<AnyView> packed_args(args.size() + 2);
    ffi::PackedArgs::Fill(packed_args.data(), static_cast<int>(DiscoSocketAction::kSend), -1);
    std::copy(args.data(), args.data() + args.size(), packed_args.begin() + 2);
    for (auto& channel : remote_channels_) {
      channel->Send(ffi::PackedArgs(packed_args.data(), packed_args.size()));
    }
  }

  void SendPacked(int worker_id, const ffi::PackedArgs& args) final {
    int node_id = worker_id / num_workers_per_node_;
    if (node_id == 0) {
      local_session_->SendPacked(worker_id, args);
      return;
    }
    std::vector<AnyView> packed_args(args.size() + 2);
    ffi::PackedArgs::Fill(packed_args.data(), static_cast<int>(DiscoSocketAction::kSend),
                          worker_id);
    std::copy(args.data(), args.data() + args.size(), packed_args.begin() + 2);
    remote_channels_[node_id - 1]->Send(ffi::PackedArgs(packed_args.data(), packed_args.size()));
  }

  ffi::PackedArgs RecvReplyPacked(int worker_id) final {
    int node_id = worker_id / num_workers_per_node_;
    if (node_id == 0) {
      return local_session_->RecvReplyPacked(worker_id);
    }
    AnyView packed_args[2];
    ffi::PackedArgs::Fill(packed_args, static_cast<int>(DiscoSocketAction::kReceive), worker_id);
    remote_channels_[node_id - 1]->Send(ffi::PackedArgs(packed_args, 2));
    return remote_channels_[node_id - 1]->Recv();
  }

  void AppendHostTensor(const ffi::Tensor& host_array) final {
    TVM_FFI_ICHECK(host_array.defined());
    local_session_->AppendHostTensor(host_array);
  }

  void Shutdown() final {
    // local session will be implicitly shutdown by its destructor
    std::vector<AnyView> packed_args(2);
    ffi::PackedArgs::Fill(packed_args.data(), static_cast<int>(DiscoSocketAction::kShutdown), -1);
    for (auto& channel : remote_channels_) {
      channel->Send(ffi::PackedArgs(packed_args.data(), packed_args.size()));
    }
    for (auto& socket : remote_sockets_) {
      socket.Close();
    }
    remote_sockets_.clear();
    remote_channels_.clear();
    if (!socket_.IsClosed()) {
      socket_.Close();
    }
    Socket::Finalize();
  }

  ~SocketSessionObj() { Shutdown(); }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("runtime.disco.SocketSession", SocketSessionObj,
                                    BcastSessionObj);
  int num_nodes_;
  int num_workers_per_node_;
  TCPSocket socket_;
  std::vector<TCPSocket> remote_sockets_;
  std::vector<std::unique_ptr<DiscoSocketChannel>> remote_channels_;
  BcastSession local_session_{ffi::UnsafeInit{}};
};

class RemoteSocketSession {
 public:
  explicit RemoteSocketSession(const ffi::String& server_host, int server_port,
                               int num_local_workers) {
    Socket::Startup();
    SockAddr server_addr{server_host.c_str(), server_port};
    socket_.Create(server_addr.ss_family());
    socket_.SetKeepAlive(true);
    if (!socket_.Connect(server_addr)) {
      TVM_FFI_THROW(InternalError) << "Failed to connect to server " << server_addr.AsString()
                                   << ", errno = " << Socket::GetLastErrorCode();
    }
    channel_ = std::make_unique<DiscoSocketChannel>(socket_);
    ffi::PackedArgs metadata = channel_->Recv();
    TVM_FFI_ICHECK_EQ(metadata.size(), 4);
    num_nodes_ = metadata[0].cast<int>();
    num_workers_per_node_ = metadata[1].cast<int>();
    num_groups_ = metadata[2].cast<int>();
    node_id_ = metadata[3].cast<int>();
    TVM_FFI_ICHECK_GE(num_local_workers, num_workers_per_node_);
    InitLocalSession();
  }

  void MainLoop() {
    while (true) {
      ffi::PackedArgs args = channel_->Recv();
      DiscoSocketAction action = static_cast<DiscoSocketAction>(args[0].cast<int>());
      int worker_id = args[1].cast<int>();
      int local_worker_id = worker_id - node_id_ * num_workers_per_node_;
      switch (action) {
        case DiscoSocketAction::kSend: {
          args = args.Slice(2);
          if (worker_id == -1) {
            local_session_->BroadcastPacked(args);
          } else {
            local_session_->SendPacked(local_worker_id, args);
          }
          break;
        }
        case DiscoSocketAction::kReceive: {
          args = local_session_->RecvReplyPacked(local_worker_id);
          channel_->Reply(args);
          break;
        }
        case DiscoSocketAction::kShutdown: {
          local_session_->Shutdown();
          LOG(INFO) << "Connection closed by remote controller.";
          return;
        }
        default:
          TVM_FFI_THROW(InternalError) << "Invalid action " << static_cast<int>(action);
      }
    }
  }

  ~RemoteSocketSession() {
    socket_.Close();
    Socket::Finalize();
  }

 private:
  void InitLocalSession() {
    const auto f_create_local_session =
        tvm::ffi::Function::GetGlobal("runtime.disco.create_socket_session_local_workers");
    local_session_ = ((*f_create_local_session)(num_workers_per_node_)).cast<BcastSession>();

    DRef f_init_workers =
        local_session_->GetGlobalFunc("runtime.disco.socket_session_init_workers");
    local_session_->CallPacked(f_init_workers, num_nodes_, node_id_, num_groups_,
                               num_workers_per_node_);
  }

  TCPSocket socket_;
  BcastSession local_session_{ffi::UnsafeInit{}};
  std::unique_ptr<DiscoSocketChannel> channel_;
  int num_nodes_{-1};
  int node_id_{-1};
  int num_groups_{-1};
  int num_workers_per_node_{-1};
};

void RemoteSocketSessionEntryPoint(const ffi::String& server_host, int server_port,
                                   int num_local_workers) {
  RemoteSocketSession proxy(server_host, server_port, num_local_workers);
  proxy.MainLoop();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("runtime.disco.RemoteSocketSession", RemoteSocketSessionEntryPoint);
}

Session SocketSession(int num_nodes, int num_workers_per_node, int num_groups,
                      const ffi::String& host, int port) {
  auto n =
      ffi::make_object<SocketSessionObj>(num_nodes, num_workers_per_node, num_groups, host, port);
  return Session(ffi::UnsafeInit{}, n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<SocketSessionObj>();
  refl::GlobalDef()
      .def("runtime.disco.SocketSession", SocketSession)
      .def("runtime.disco.socket_session_init_workers",
           [](int num_nodes, int node_id, int num_groups, int num_workers_per_node) {
             LOG(INFO) << "Initializing worker group with " << num_nodes << " nodes, "
                       << num_workers_per_node << " workers per node, and " << num_groups
                       << " groups.";
             DiscoWorker* worker = DiscoWorker::ThreadLocal();
             worker->num_groups = num_groups;
             worker->worker_id = worker->worker_id + node_id * num_workers_per_node;
             worker->num_workers = num_nodes * num_workers_per_node;
           });
}

}  // namespace runtime
}  // namespace tvm
