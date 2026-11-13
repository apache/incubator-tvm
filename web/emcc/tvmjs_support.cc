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

/*
 * \file tvmjs_support.cc
 * \brief Support functions to be linked with wasm_runtime to provide
 *        ffi::Function callbacks in tvmjs.
 *        We do not need to link this file in standalone wasm.
 */

// configurations for tvm logging
#define TVM_LOG_DEBUG 0
#define TVM_LOG_CUSTOMIZE 1
#define TVM_FFI_ALWAYS_LOG_BEFORE_THROW 1

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/device_api.h>

#include <unordered_set>

#include "../../src/runtime/rpc/rpc_local_session.h"

extern "C" {
// --- Additional C API for the Wasm runtime ---
/*!
 * \brief Allocate space aligned to 64 bit.
 * \param size The size of the space.
 * \return The allocated space.
 */
TVM_FFI_DLL void* TVMWasmAllocSpace(int size);

/*!
 * \brief Free the space allocated by TVMWasmAllocSpace.
 * \param data The data pointer.
 */
TVM_FFI_DLL void TVMWasmFreeSpace(void* data);

/*!
 * \brief Create ffi::Function from a resource handle.
 * \param resource_handle The handle to the resource.
 * \param out The output ffi::Function.
 * \sa TVMWasmPackedCFunc, TVMWasmPackedCFuncFinalizer
3A * \return 0 if success.
 */
TVM_FFI_DLL int TVMFFIWasmFunctionCreate(void* resource_handle, TVMFFIObjectHandle* out);

/*!
 * \brief Get the last error message.
 * \return The last error message.
 */
TVM_FFI_DLL const char* TVMFFIWasmGetLastError();

// --- APIs to be implemented by the frontend. ---

/*!
 * \brief Wasm frontend new ffi call function caller.
 *
 * \param self The pointer to the ffi::Function.
 * \param args The arguments
 * \param num_args Number of arguments.
 * \param result The return value handle.
 * \return 0 if success, -1 if failure happens, set error via TVMAPISetLastError.
 */
extern int TVMFFIWasmSafeCall(void* self, const TVMFFIAny* args, int32_t num_args,
                              TVMFFIAny* result);
/*!
 * \brief Delete ffi::Function.
 * \param self The pointer to the ffi::Function.
 */
extern void TVMFFIWasmFunctionDeleter(void* self);

}  // extern "C"

void* TVMWasmAllocSpace(int size) {
  int num_count = (size + 7) / 8;
  return new int64_t[num_count];
}

void TVMWasmFreeSpace(void* arr) { delete[] static_cast<int64_t*>(arr); }

int TVMFFIWasmFunctionCreate(void* self, TVMFFIObjectHandle* out) {
  return TVMFFIFunctionCreate(self, TVMFFIWasmSafeCall, TVMFFIWasmFunctionDeleter, out);
}

const char* TVMFFIWasmGetLastError() {
  static thread_local std::string last_error;
  last_error = ::tvm::ffi::detail::MoveFromSafeCallRaised().what();
  return last_error.c_str();
}

namespace tvm {
namespace runtime {

// A special local session that can interact with async
// functions in the JS runtime.
class AsyncLocalSession : public LocalSession {
 public:
  AsyncLocalSession() {}

  PackedFuncHandle GetFunction(const std::string& name) final {
    if (name == "runtime.RPCTimeEvaluator") {
      return get_time_eval_placeholder_.get();
    } else if (auto fp = tvm::ffi::Function::GetGlobal(name)) {
      TVMFFIAny val = tvm::ffi::detail::AnyUnsafe::MoveAnyToTVMFFIAny(tvm::ffi::Any(*fp));
      return val.v_obj;
    } else if (auto fp = tvm::ffi::Function::GetGlobal("__async." + name)) {
      TVMFFIAny val = tvm::ffi::detail::AnyUnsafe::MoveAnyToTVMFFIAny(tvm::ffi::Any(*fp));
      async_func_set_.insert(val.v_obj);
      return val.v_obj;
    } else {
      return nullptr;
    }
  }

  void FreeHandle(void* handle) final {
    {
      auto it = async_func_set_.find(handle);
      if (it != async_func_set_.end()) {
        async_func_set_.erase(it);
      }
    }
    if (handle != get_time_eval_placeholder_.get()) {
      LocalSession::FreeHandle(handle);
    }
  }

  void AsyncCallFunc(PackedFuncHandle func, ffi::PackedArgs args, FAsyncCallback callback) final {
    auto it = async_func_set_.find(func);
    if (it != async_func_set_.end()) {
      ffi::Function packed_callback([callback, this](ffi::PackedArgs args, ffi::Any*) {
        int code = args[0].cast<int>();
        ffi::Any rv;
        rv = args[1];
        if (code == static_cast<int>(RPCCode::kReturn)) {
          this->EncodeReturn(std::move(rv), [&](ffi::PackedArgs encoded_args) {
            callback(RPCCode::kReturn, encoded_args);
          });
        } else {
          // for exception, we can pass through as since this is just normal encoding.
          TVM_FFI_ICHECK_EQ(code, static_cast<int>(RPCCode::kException));
          callback(RPCCode::kException, args);
        }
      });

      std::vector<AnyView> packed_args(args.data(), args.data() + args.size());
      // pass the callback as the last argument.
      packed_args.emplace_back(AnyView(packed_callback));
      auto* pf = static_cast<ffi::FunctionObj*>(func);
      Any temp;
      pf->CallPacked(packed_args.data(), packed_args.size(), &temp);
    } else if (func == get_time_eval_placeholder_.get()) {
      // special handle time evaluator.
      try {
        ffi::Function retfunc = this->GetTimeEvaluator(
            args[0].cast<ffi::Optional<ffi::Module>>(), args[1].cast<std::string>(),
            args[2].cast<int>(), args[3].cast<int>(), args[4].cast<int>(), args[5].cast<int>(),
            args[6].cast<int>(), args[7].cast<int>(), args[8].cast<int>(), args[9].cast<int>());
        ffi::Any rv;
        rv = retfunc;
        this->EncodeReturn(std::move(rv), [&](ffi::PackedArgs encoded_args) {
          const void* pf = encoded_args[0].as<ffi::FunctionObj>();
          TVM_FFI_ICHECK(pf != nullptr);
          // mark as async.
          async_func_set_.insert(const_cast<void*>(pf));
          callback(RPCCode::kReturn, encoded_args);
        });
      } catch (const std::runtime_error& e) {
        this->SendException(callback, e.what());
      }
    } else {
      LocalSession::AsyncCallFunc(func, args, callback);
    }
  }

  void AsyncCopyToRemote(void* local_from_bytes, DLTensor* remote_to, uint64_t nbytes,
                         FAsyncCallback on_complete) final {
    try {
      DLTensor local_from;
      local_from.data = local_from_bytes;
      local_from.device = Device{kDLCPU, 0};
      local_from.ndim = remote_to->ndim;
      local_from.shape = remote_to->shape;
      local_from.dtype = remote_to->dtype;
      local_from.strides = nullptr;
      local_from.byte_offset = 0;
      this->CopyDataFromTo(&local_from, remote_to, nullptr);
      this->AsyncStreamWait(remote_to->device, nullptr, on_complete);
    } catch (const std::runtime_error& e) {
      this->SendException(on_complete, e.what());
    }
  }

  void AsyncCopyFromRemote(DLTensor* remote_from, void* local_to_bytes, uint64_t nbytes,
                           FAsyncCallback on_complete) final {
    try {
      DLTensor local_to;
      local_to.data = local_to_bytes;
      local_to.device = Device{kDLCPU, 0};
      local_to.ndim = remote_from->ndim;
      local_to.shape = remote_from->shape;
      local_to.dtype = remote_from->dtype;
      local_to.strides = nullptr;
      local_to.byte_offset = 0;
      this->CopyDataFromTo(remote_from, &local_to, nullptr);
      this->AsyncStreamWait(remote_from->device, nullptr, on_complete);
    } catch (const std::runtime_error& e) {
      this->SendException(on_complete, e.what());
    }
  }

  void AsyncStreamWait(Device dev, TVMStreamHandle stream, FAsyncCallback on_complete) final {
    if (dev.device_type == kDLCPU) {
      AnyView packed_args[1];
      packed_args[0] = nullptr;
      on_complete(RPCCode::kReturn, ffi::PackedArgs(packed_args, 1));
    } else {
      TVM_FFI_ICHECK(dev.device_type == static_cast<DLDeviceType>(kDLWebGPU));
      if (!async_wait_.has_value()) {
        async_wait_ = tvm::ffi::Function::GetGlobal("__async.wasm.WebGPUWaitForTasks");
      }
      TVM_FFI_ICHECK(async_wait_.has_value());
      ffi::Function packed_callback([on_complete](ffi::PackedArgs args, ffi::Any*) {
        int code = args[0].cast<int>();
        on_complete(static_cast<RPCCode>(code), args.Slice(1));
      });
      (*async_wait_)(packed_callback);
    }
  }

  bool IsAsync() const final { return true; }

 private:
  std::unordered_set<void*> async_func_set_;
  std::unique_ptr<ffi::Function> get_time_eval_placeholder_ = std::make_unique<ffi::Function>();
  std::optional<ffi::Function> async_wait_;

  // time evaluator
  ffi::Function GetTimeEvaluator(ffi::Optional<ffi::Module> opt_mod, std::string name,
                                 int device_type, int device_id, int number, int repeat,
                                 int min_repeat_ms, int limit_zero_time_iterations,
                                 int cooldown_interval_ms, int repeats_to_cooldown) {
    Device dev;
    dev.device_type = static_cast<DLDeviceType>(device_type);
    dev.device_id = device_id;

    if (opt_mod.has_value()) {
      ffi::Module m = opt_mod.value();
      std::string tkey = m->kind();
      return WrapWasmTimeEvaluator(m->GetFunction(name, false).value(), dev, number, repeat,
                                   min_repeat_ms, limit_zero_time_iterations, cooldown_interval_ms,
                                   repeats_to_cooldown);
    } else {
      auto pf = tvm::ffi::Function::GetGlobal(name);
      TVM_FFI_ICHECK(pf.has_value()) << "Cannot find " << name << " in the global function";
      return WrapWasmTimeEvaluator(*pf, dev, number, repeat, min_repeat_ms,
                                   limit_zero_time_iterations, cooldown_interval_ms,
                                   repeats_to_cooldown);
    }
  }

  // time evaluator
  ffi::Function WrapWasmTimeEvaluator(ffi::Function pf, Device dev, int number, int repeat,
                                      int min_repeat_ms, int limit_zero_time_iterations,
                                      int cooldown_interval_ms, int repeats_to_cooldown) {
    auto ftimer = [pf, dev, number, repeat, min_repeat_ms, limit_zero_time_iterations,
                   cooldown_interval_ms, repeats_to_cooldown](ffi::PackedArgs args, ffi::Any* rv) {
      // the function is a async function.
      ffi::Function on_complete = args[args.size() - 1].cast<ffi::Function>();

      std::vector<AnyView> packed_args(args.data(), args.data() + args.size() - 1);
      auto finvoke = [pf, packed_args](int n) {
        ffi::Any temp;
        ffi::PackedArgs invoke_args(packed_args.data(), packed_args.size());
        for (int i = 0; i < n; ++i) {
          pf.CallPacked(invoke_args, &temp);
        }
      };
      auto time_exec = tvm::ffi::Function::GetGlobal("__async.wasm.TimeExecution");
      TVM_FFI_ICHECK(time_exec.has_value()) << "Cannot find wasm.GetTimer in the global function";
      (*time_exec)(ffi::TypedFunction<void(int)>(finvoke), dev, number, repeat, min_repeat_ms,
                   limit_zero_time_iterations, cooldown_interval_ms, repeats_to_cooldown,
                   /*cache_flush_bytes=*/0, on_complete);
    };
    return ffi::Function(ftimer);
  }
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("wasm.LocalSession", []() {
    return CreateRPCSessionModule(std::make_shared<AsyncLocalSession>());
  });
}

}  // namespace runtime
}  // namespace tvm
