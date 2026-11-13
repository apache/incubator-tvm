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
#include "./bcast_session.h"

#include <tvm/ffi/cast.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/disco/session.h>

#include <sstream>

namespace tvm {
namespace runtime {

struct BcastSessionObj::Internal {
  template <typename... Args>
  TVM_FFI_INLINE static void BroadcastUnpacked(BcastSessionObj* self, DiscoAction action,
                                               int64_t reg_id, Args&&... args) {
    constexpr int kNumArgs = 2 + sizeof...(Args);
    ffi::AnyView packed_args[kNumArgs];
    ffi::PackedArgs::Fill(packed_args, static_cast<int>(action), reg_id,
                          std::forward<Args>(args)...);
    self->BroadcastPacked(ffi::PackedArgs(packed_args, kNumArgs));
  }

  static DRef MakeDRef(int reg_id, Session session) {
    ffi::ObjectPtr<DRefObj> p = ffi::make_object<DRefObj>();
    p->reg_id = reg_id;
    p->session = session;
    return DRef(std::move(p));
  }
};

DRef BcastSessionObj::GetGlobalFunc(const std::string& name) {
  int reg_id = AllocateReg();
  BcastSessionObj::Internal::BroadcastUnpacked(this, DiscoAction::kGetGlobalFunc, reg_id, name);
  return BcastSessionObj::Internal::MakeDRef(reg_id, ffi::GetRef<Session>(this));
}

void BcastSessionObj::CopyFromWorker0(const ffi::Tensor& host_array, const DRef& remote_array) {
  TVM_FFI_ICHECK(host_array.defined());
  this->AppendHostTensor(host_array);
  BcastSessionObj::Internal::BroadcastUnpacked(this, DiscoAction::kCopyFromWorker0,
                                               remote_array->reg_id);
}

void BcastSessionObj::CopyToWorker0(const ffi::Tensor& host_array, const DRef& remote_array) {
  TVM_FFI_ICHECK(host_array.defined());
  this->AppendHostTensor(host_array);
  BcastSessionObj::Internal::BroadcastUnpacked(this, DiscoAction::kCopyToWorker0,
                                               remote_array->reg_id);
}

void BcastSessionObj::Shutdown() {
  BcastSessionObj::Internal::BroadcastUnpacked(this, DiscoAction::kShutDown, 0);
}

void BcastSessionObj::InitCCL(ffi::String ccl, ffi::Shape device_ids) {
  const auto pf = tvm::ffi::Function::GetGlobal("runtime.disco." + ccl + ".init_ccl");
  TVM_FFI_CHECK(pf.has_value(), ValueError)
      << "Cannot initialize CCL `" << ccl << "`, because cannot find function: runtime.disco."
      << ccl << ".init_ccl";
  (*pf)(ffi::GetRef<Session>(this), device_ids);
}

void BcastSessionObj::SyncWorker(int worker_id) {
  BcastSessionObj::Internal::BroadcastUnpacked(this, DiscoAction::kSyncWorker, worker_id);
  ffi::PackedArgs args = this->RecvReplyPacked(worker_id);
  TVM_FFI_ICHECK_EQ(args.size(), 2);
  DiscoAction action = static_cast<DiscoAction>(args[0].cast<int>());
  int ret_worker_id = args[1].cast<int>();
  TVM_FFI_ICHECK(action == DiscoAction::kSyncWorker);
  TVM_FFI_ICHECK_EQ(ret_worker_id, worker_id);
}

DRef BcastSessionObj::CallWithPacked(const ffi::PackedArgs& args) {
  // NOTE: this action is not safe unless we know args is not
  // used else where in this case it is oK
  ffi::AnyView* args_vec = const_cast<ffi::AnyView*>(args.data());
  // tranlsate args into remote calling convention
  int reg_id = AllocateReg();
  {
    DRef func = args[2].cast<DRef>();
    args_vec[0] = static_cast<int>(DiscoAction::kCallPacked);
    args_vec[1] = reg_id;
    args_vec[2] = func->reg_id;
  }
  this->BroadcastPacked(ffi::PackedArgs(args_vec, args.size()));
  return BcastSessionObj::Internal::MakeDRef(reg_id, ffi::GetRef<Session>(this));
}

void BcastSessionObj::DeallocReg(int reg_id) {
  BcastSessionObj::Internal::BroadcastUnpacked(this, DiscoAction::kKillReg, reg_id);
  this->free_regs_.push_back(reg_id);
}

int BcastSessionObj::AllocateReg() {
  if (this->free_regs_.empty()) {
    return this->reg_count_++;
  }
  int reg_id = this->free_regs_.back();
  this->free_regs_.pop_back();
  return reg_id;
}

void BcastSessionObj::AppendHostTensor(const ffi::Tensor& host_array) {
  TVM_FFI_ICHECK(host_array.defined());
  std::lock_guard<std::mutex> lock(worker_zero_data_.queue_mutex_);
  worker_zero_data_.host_arrays.push(host_array);
}

}  // namespace runtime
}  // namespace tvm
