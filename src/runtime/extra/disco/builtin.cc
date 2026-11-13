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
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/disco/builtin.h>
#include <tvm/runtime/disco/disco_worker.h>
#include <tvm/runtime/disco/session.h>
#include <tvm/runtime/extra_base.h>
#include <tvm/runtime/tensor.h>

#include <sstream>

#include "./utils.h"

namespace tvm {
namespace runtime {

class DSOLibraryCache {
 public:
  ffi::Module Open(const std::string& library_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_.find(library_path);
    if (it == cache_.end()) {
      ffi::Module lib = ffi::Module::LoadFromFile(library_path);
      cache_.emplace(library_path, lib);
      return lib;
    }
    return it->second;
  }

  std::unordered_map<std::string, ffi::Module> cache_;
  std::mutex mutex_;
};

ffi::Module LoadVMModule(std::string path, ffi::Optional<Device> device) {
  static DSOLibraryCache cache;
  ffi::Module dso_mod = cache.Open(path);
  Device dev = UseDefaultDeviceIfNone(device);
  ffi::Optional<ffi::Function> vm_load_executable = dso_mod->GetFunction("vm_load_executable");
  if (!vm_load_executable.has_value()) {
    // not built by RelaxVM, return the dso_mod directly
    return dso_mod;
  }
  auto mod = (*vm_load_executable)().cast<ffi::Module>();
  ffi::Optional<ffi::Function> vm_initialization = mod->GetFunction("vm_initialization");
  if (!vm_initialization.has_value()) {
    TVM_FFI_THROW(ValueError)
        << "File `" << path
        << "` is not built by RelaxVM, because `vm_initialization` does not exist";
  }
  ffi::Function::GetGlobalRequired("vm.builtin.initialize_default")(mod, dev);
  return mod;
}

ffi::Tensor DiscoEmptyTensor(ffi::Shape shape, DLDataType dtype, ffi::Optional<Device> device) {
  return tvm::runtime::Empty(shape, dtype, UseDefaultDeviceIfNone(device));
}

ffi::Function GetCCLFunc(const char* name) {
  std::string ccl = DiscoWorker::ThreadLocal()->ccl;
  std::string pf_name = "runtime.disco." + ccl + "." + name;
  const auto pf = tvm::ffi::Function::GetGlobal(pf_name);
  TVM_FFI_CHECK(pf.has_value(), ValueError)
      << "Cannot find the `" << name << "` function for `" << ccl << "` via `" << pf_name << "`";
  return *pf;
}

void AllReduce(ffi::Tensor send, ReduceKind reduce_kind, bool in_group, ffi::Tensor recv) {
  TVM_FFI_ICHECK(send.defined());
  TVM_FFI_ICHECK(recv.defined());
  GetCCLFunc("allreduce")(send, static_cast<int>(reduce_kind), in_group, recv);
}

void AllGather(ffi::Tensor send, bool in_group, ffi::Tensor recv) {
  TVM_FFI_ICHECK(send.defined());
  TVM_FFI_ICHECK(recv.defined());
  GetCCLFunc("allgather")(send, in_group, recv);
}

TVM_RUNTIME_EXTRA_DLL void BroadcastFromWorker0(ffi::Tensor send, bool in_group, ffi::Tensor recv) {
  TVM_FFI_ICHECK(send.defined());
  TVM_FFI_ICHECK(recv.defined());
  GetCCLFunc("broadcast_from_worker0")(send, in_group, recv);
}

TVM_RUNTIME_EXTRA_DLL void ScatterFromWorker0(ffi::Optional<ffi::Tensor> send, bool in_group,
                                              ffi::Tensor recv) {
  TVM_FFI_ICHECK(recv.defined());
  GetCCLFunc("scatter_from_worker0")(send, in_group, recv);
}

void GatherToWorker0(ffi::Tensor send, bool in_group, ffi::Optional<ffi::Tensor> recv) {
  TVM_FFI_ICHECK(send.defined());
  GetCCLFunc("gather_to_worker0")(send, in_group, recv);
}

void RecvFromWorker0(ffi::Tensor buffer) {
  TVM_FFI_ICHECK(buffer.defined());
  GetCCLFunc("recv_from_worker0")(buffer);
}

void SendToNextGroup(ffi::Tensor buffer) {
  TVM_FFI_ICHECK(buffer.defined());
  GetCCLFunc("send_to_next_group")(buffer);
}

void RecvFromPrevGroup(ffi::Tensor buffer) {
  TVM_FFI_ICHECK(buffer.defined());
  GetCCLFunc("recv_from_prev_group")(buffer);
}

void SendToWorker(ffi::Tensor buffer, int receiver_id) {
  TVM_FFI_ICHECK(buffer.defined());
  GetCCLFunc("send_to_worker")(buffer, receiver_id);
}

void RecvFromWorker(ffi::Tensor buffer, int sender_id) {
  TVM_FFI_ICHECK(buffer.defined());
  GetCCLFunc("recv_from_worker")(buffer, sender_id);
}

int WorkerId() { return DiscoWorker::ThreadLocal()->worker_id; }

void SyncWorker() {
  if (DiscoWorker::ThreadLocal()->ccl != "") {
    GetCCLFunc("sync_worker")();
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.disco.load_vm_module", LoadVMModule)
      .def("runtime.disco.empty",
           [](ffi::Shape shape, DLDataType dtype, ffi::Optional<Device> device, bool worker0_only,
              bool in_group) -> ffi::Optional<ffi::Tensor> {
             int worker_id = WorkerId();
             int group_size =
                 DiscoWorker::ThreadLocal()->num_workers / DiscoWorker::ThreadLocal()->num_groups;
             bool is_worker0 =
                 (worker_id == 0 && !in_group) || (in_group && worker_id % group_size == 0);
             if (worker0_only && !is_worker0) {
               return std::nullopt;
             } else {
               return DiscoEmptyTensor(shape, dtype, device);
             }
           })
      .def("runtime.disco.allreduce",
           [](ffi::Tensor send, ffi::Shape reduce_kind, bool in_group, ffi::Tensor recv) {
             TVM_FFI_ICHECK(send.defined());
             TVM_FFI_ICHECK(recv.defined());
             int kind = IntegerFromShape(reduce_kind);
             TVM_FFI_CHECK(0 <= kind && kind <= 4, ValueError) << "Unknown ReduceKind: " << kind;
             AllReduce(send, static_cast<ReduceKind>(kind), in_group, recv);
           })
      .def("runtime.disco.allgather", AllGather)
      .def("runtime.disco.broadcast_from_worker0", BroadcastFromWorker0)
      .def("runtime.disco.scatter_from_worker0", ScatterFromWorker0)
      .def("runtime.disco.gather_to_worker0", GatherToWorker0)
      .def("runtime.disco.recv_from_worker0", RecvFromWorker0)
      .def("runtime.disco.send_to_next_group", SendToNextGroup)
      .def("runtime.disco.recv_from_prev_group", RecvFromPrevGroup)
      .def("runtime.disco.send_to_worker", SendToWorker)
      .def("runtime.disco.recv_from_worker", RecvFromWorker)
      .def("runtime.disco.worker_id", []() -> ffi::Shape { return ffi::Shape({WorkerId()}); })
      .def("runtime.disco.worker_rank", []() -> int64_t { return WorkerId(); })
      .def("runtime.disco.world_size",
           []() -> int64_t { return DiscoWorker::ThreadLocal()->num_workers; })
      .def("runtime.disco.device",
           []() -> Device { return DiscoWorker::ThreadLocal()->default_device; })
      .def("runtime.disco.bind_worker_to_cpu_core", [](ffi::Shape cpu_ids) {
        int worker_id = WorkerId();
        TVM_FFI_ICHECK_LT(worker_id, static_cast<int>(cpu_ids.size()));
        const auto f_set_thread_affinity = tvm::ffi::Function::GetGlobalRequired(
            "tvm.runtime.threading.set_current_thread_affinity");
        f_set_thread_affinity(ffi::Shape{cpu_ids[worker_id]});
      });
}

}  // namespace runtime
}  // namespace tvm
