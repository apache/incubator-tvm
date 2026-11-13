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
#include <tvm/runtime/base.h>
#include <tvm/runtime/disco/disco_worker.h>

#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include "../../../support/pipe.h"
#include "../../rpc/minrpc/rpc_reference.h"
#include "./bcast_session.h"
#include "./disco_worker_thread.h"
#include "./message_queue.h"
#include "./protocol.h"

namespace tvm {
namespace runtime {

class DiscoProcessChannel final : public DiscoChannel {
 public:
  DiscoProcessChannel(int64_t controler_to_worker_fd, int64_t worker_to_controler_fd)
      : controller_to_worker_pipe_(controler_to_worker_fd),
        worker_to_controller_pipe_(worker_to_controler_fd),
        controler_to_worker_(&controller_to_worker_pipe_),
        worker_to_controler_(&worker_to_controller_pipe_) {}

  DiscoProcessChannel(DiscoProcessChannel&& other) = delete;
  DiscoProcessChannel(const DiscoProcessChannel& other) = delete;

  void Send(const ffi::PackedArgs& args) { controler_to_worker_.Send(args); }
  ffi::PackedArgs Recv() { return controler_to_worker_.Recv(); }
  void Reply(const ffi::PackedArgs& args) { worker_to_controler_.Send(args); }
  ffi::PackedArgs RecvReply() { return worker_to_controler_.Recv(); }

  support::Pipe controller_to_worker_pipe_;
  support::Pipe worker_to_controller_pipe_;
  DiscoStreamMessageQueue controler_to_worker_;
  DiscoStreamMessageQueue worker_to_controler_;
};

class ProcessSessionObj final : public BcastSessionObj {
 public:
  explicit ProcessSessionObj(int num_workers, int num_groups, ffi::Function process_pool)
      : process_pool_(process_pool),
        worker_0_(
            std::make_unique<DiscoWorkerThread>(0, num_workers, num_groups, &worker_zero_data_)) {
    std::vector<int64_t> read_fds;
    std::vector<int64_t> write_fds;
    read_fds.reserve(num_workers - 1);
    write_fds.reserve(num_workers - 1);
    for (int i = 1; i < num_workers; ++i) {
      ffi::Shape fds = process_pool(i).cast<ffi::Shape>();
      TVM_FFI_CHECK_EQ(fds.size(), 2, ValueError)
          << "process_pool(" << i << ") should return a tuple of "
          << "size 2, but got a tuple of size " << fds.size() << ".";
      read_fds.push_back(fds[0]);
      write_fds.push_back(fds[1]);
    }
    for (int i = 0; i < num_workers - 1; ++i) {
      workers_.emplace_back(std::make_unique<DiscoProcessChannel>(write_fds[i], read_fds[i]));
    }
  }

  void Kill() {
    if (this->worker_0_ != nullptr) {
      this->Shutdown();
      this->worker_0_.reset();
      this->workers_.clear();
      this->process_pool_(0);
    }
  }

  ~ProcessSessionObj() { Kill(); }

  int64_t GetNumWorkers() { return workers_.size() + 1; }

  ffi::Any DebugGetFromRemote(int64_t reg_id, int worker_id) {
    if (worker_id == 0) {
      this->SyncWorker(worker_id);
      return worker_0_->worker->register_file.at(reg_id);
    }
    {
      ffi::AnyView packed_args[3];
      ffi::PackedArgs::Fill(packed_args, static_cast<int>(DiscoAction::kDebugGetFromRemote), reg_id,
                            worker_id);
      workers_[worker_id - 1]->Send(ffi::PackedArgs(packed_args, 3));
    }
    ffi::PackedArgs args = this->RecvReplyPacked(worker_id);
    TVM_FFI_ICHECK_EQ(args.size(), 2);
    TVM_FFI_ICHECK(static_cast<DiscoAction>(args[0].cast<int>()) ==
                   DiscoAction::kDebugGetFromRemote);
    ffi::Any result;
    result = args[1];
    return result;
  }

  void DebugSetRegister(int64_t reg_id, ffi::AnyView value, int worker_id) {
    if (worker_id == 0) {
      this->SyncWorker(worker_id);
      worker_0_->worker->SetRegister(reg_id, value);
      return;
    }
    ffi::ObjectRef wrapped{nullptr};
    if (value.as<ffi::ObjectRef>()) {
      wrapped = DiscoDebugObject::Wrap(value);
      value = wrapped;
    }
    {
      ffi::AnyView packed_args[4];
      ffi::PackedArgs::Fill(packed_args, static_cast<int>(DiscoAction::kDebugSetRegister), reg_id,
                            worker_id, value);
      SendPacked(worker_id, ffi::PackedArgs(packed_args, 4));
    }
    ffi::Any result;
    ffi::PackedArgs args = this->RecvReplyPacked(worker_id);
    TVM_FFI_ICHECK_EQ(args.size(), 1);
    TVM_FFI_ICHECK(static_cast<DiscoAction>(args[0].cast<int>()) == DiscoAction::kDebugSetRegister);
  }

  void BroadcastPacked(const ffi::PackedArgs& args) final {
    worker_0_->channel->Send(args);
    for (std::unique_ptr<DiscoProcessChannel>& channel : workers_) {
      channel->Send(args);
    }
  }

  void SendPacked(int worker_id, const ffi::PackedArgs& args) final {
    if (worker_id == 0) {
      worker_0_->channel->Send(args);
    } else {
      workers_.at(worker_id - 1)->Send(args);
    }
  }

  ffi::PackedArgs RecvReplyPacked(int worker_id) final {
    if (worker_id == 0) {
      return worker_0_->channel->RecvReply();
    }
    return this->workers_.at(worker_id - 1)->RecvReply();
  }

  DiscoChannel* GetWorkerChannel(int worker_id) {
    if (worker_id == 0) {
      return worker_0_->channel.get();
    }
    return workers_.at(worker_id - 1).get();
  }

  ffi::Function process_pool_;
  std::unique_ptr<DiscoWorkerThread> worker_0_;
  std::vector<std::unique_ptr<DiscoProcessChannel>> workers_;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("runtime.disco.ProcessSession", ProcessSessionObj, SessionObj);
};

Session Session::ProcessSession(int num_workers, int num_group, ffi::String process_pool_creator,
                                ffi::String entrypoint) {
  TVM_FFI_ICHECK_EQ(num_workers % num_group, 0)
      << "The number of workers should be divisible by the number of worker group.";
  const auto pf = tvm::ffi::Function::GetGlobal(process_pool_creator);
  TVM_FFI_CHECK(pf, ValueError) << "Cannot find function " << process_pool_creator
                                << " in the registry. Please check if it is registered.";
  auto process_pool = (*pf)(num_workers, num_group, entrypoint).cast<ffi::Function>();
  auto n = ffi::make_object<ProcessSessionObj>(num_workers, num_group, process_pool);
  return Session(ffi::UnsafeInit{}, n);
}

void WorkerProcess(int worker_id, int num_workers, int num_group, int64_t read_fd,
                   int64_t write_fd) {
  TVM_FFI_ICHECK_EQ(num_workers % num_group, 0)
      << "The number of workers should be divisible by the number of worker group.";
  DiscoProcessChannel channel(read_fd, write_fd);
  DiscoWorker worker(worker_id, num_workers, num_group, nullptr, &channel);
  worker.MainLoop();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<ProcessSessionObj>();
  refl::GlobalDef()
      .def("runtime.disco.SessionProcess", Session::ProcessSession)
      .def("runtime.disco.WorkerProcess", WorkerProcess);
}

}  // namespace runtime
}  // namespace tvm
