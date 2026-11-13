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
 * \file rpc_module.cc
 * \brief RPC runtime module.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/timer.h>

#include <chrono>
#include <cstring>
#include <memory>
#include <thread>

#include "rpc_endpoint.h"
#include "rpc_remote.h"

namespace tvm {
namespace runtime {
/*!
 * \brief A wrapped remote function as a ffi::Function.
 */
class RPCWrappedFunc : public ffi::Object {
 public:
  RPCWrappedFunc(void* handle, std::shared_ptr<RPCSession> sess) : handle_(handle), sess_(sess) {}

  void operator()(ffi::PackedArgs args, ffi::Any* rv) const {
    sess_->EnsureOpen();
    std::vector<AnyView> packed_args(args.size());
    std::vector<std::unique_ptr<DLTensor>> temp_dltensors;

    // scan and check whether we need rewrite these arguments
    // to their remote variant.
    for (int i = 0; i < args.size(); ++i) {
      // handle both str and small str
      if (args[i].type_index() == ffi::TypeIndex::kTVMFFIStr) {
        // pass string as c_str
        packed_args[i] = args[i].cast<ffi::String>().data();
        continue;
      } else if (args[i].type_index() == ffi::TypeIndex::kTVMFFISmallStr) {
        // we cannot cast here, since we need to make sure the space is alive
        const TVMFFIAny* any_view_ptr = reinterpret_cast<const TVMFFIAny*>(&args.data()[i]);
        TVMFFIByteArray bytes = TVMFFISmallBytesGetContentByteArray(any_view_ptr);
        packed_args[i] = bytes.data;
        continue;
      }
      packed_args[i] = args[i];
      if (auto tensor = args[i].as<RemoteTensor>()) {
        TVM_FFI_CHECK((*tensor)->device->sess == sess_, ValueError)
            << "Tensor belongs to a different remote session";
        auto dptr = std::make_unique<DLTensor>((*tensor)->descriptor());
        packed_args[i] = dptr.get();
        temp_dltensors.emplace_back(std::move(dptr));
        continue;
      }
      if (auto dev = args[i].as<RemoteDevice>()) {
        TVM_FFI_CHECK((*dev)->sess == sess_, ValueError)
            << "Device belongs to a different remote session";
        packed_args[i] = (*dev)->endpoint_device;
        continue;
      }
      if (auto obj = args[i].as<RPCObjectRef>()) {
        TVM_FFI_CHECK((*obj)->sess() == sess_, ValueError)
            << "Object belongs to a different remote session";
        continue;
      }
      switch (args[i].type_index()) {
        case ffi::TypeIndex::kTVMFFITensor:
        case ffi::TypeIndex::kTVMFFIDLTensorPtr:
          TVM_FFI_THROW(ValueError) << "Pass an RPC RemoteTensor, not a local tensor";
          break;
        case ffi::TypeIndex::kTVMFFIFunction:
        case ffi::TypeIndex::kTVMFFIModule: {
          packed_args[i] = UnwrapRemoteValueToHandle(args[i]);
          // need to force set the type index to the correct one
          TVMFFIAny temp = packed_args[i].CopyToTVMFFIAny();
          temp.type_index = args[i].type_index();
          packed_args[i] = AnyView::CopyFromTVMFFIAny(temp);
          break;
        }
        default:
          TVM_FFI_CHECK(args[i].type_index() < ffi::TypeIndex::kTVMFFIStaticObjectBegin ||
                            args[i].type_index() == ffi::TypeIndex::kTVMFFIBytes,
                        ValueError)
              << "Cannot pass a local object to a remote function: " << args[i].GetTypeKey();
      }
    }
    auto set_return = [this, rv](ffi::PackedArgs args) { this->WrapRemoteReturnToValue(args, rv); };
    sess_->CallFunc(handle_, ffi::PackedArgs(packed_args.data(), packed_args.size()), set_return);
  }

  ~RPCWrappedFunc() {
    try {
      sess_->FreeHandle(handle_);
    } catch (const ffi::Error& e) {
      // fault tolerance to remote close
    }
  }

 private:
  // remote function handle
  void* handle_{nullptr};
  // pointer to the session.
  std::shared_ptr<RPCSession> sess_;

  // unwrap a remote value to the underlying handle.
  void* UnwrapRemoteValueToHandle(const ffi::AnyView& arg) const;
  // wrap a remote return via Set
  void WrapRemoteReturnToValue(ffi::PackedArgs args, ffi::Any* rv) const;
};

// RPC that represents a remote module session.
class RPCModuleObj final : public ffi::ModuleObj {
 public:
  RPCModuleObj(void* module_handle, std::shared_ptr<RPCSession> sess)
      : module_handle_(module_handle), sess_(sess) {}

  ~RPCModuleObj() {
    if (module_handle_ != nullptr) {
      try {
        sess_->FreeHandle(module_handle_);
      } catch (const ffi::Error& e) {
        // fault tolerance to remote close
      }
      module_handle_ = nullptr;
    }
  }

  const char* kind() const final { return "rpc"; }
  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const final { return ffi::Module::ModulePropertyMask::kRunnable; }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final {
    if (name == "CloseRPCConnection") {
      return ffi::Function([sess = sess_](ffi::PackedArgs, ffi::Any*) { sess->Close(); });
    }
    sess_->EnsureOpen();

    if (module_handle_ == nullptr) {
      return WrapRemoteFunc(sess_->GetFunction(name));
    } else {
      InitRemoteFunc(&remote_mod_get_function_, "tvm.rpc.server.ModuleGetFunction");
      return remote_mod_get_function_(ffi::GetRef<ffi::Module>(this), name, true);
    }
  }

  ffi::Function GetTimeEvaluator(const std::string& name, Device dev, int number, int repeat,
                                 int min_repeat_ms, int limit_zero_time_iterations,
                                 int cooldown_interval_ms, int repeats_to_cooldown,
                                 int cache_flush_bytes, const std::string& f_preproc_name) {
    InitRemoteFunc(&remote_get_time_evaluator_, "runtime.RPCTimeEvaluator");

    if (module_handle_ != nullptr) {
      return remote_get_time_evaluator_(
          ffi::GetRef<ffi::Module>(this), name, static_cast<int>(dev.device_type), dev.device_id,
          number, repeat, min_repeat_ms, limit_zero_time_iterations, cooldown_interval_ms,
          repeats_to_cooldown, cache_flush_bytes, f_preproc_name);
    } else {
      return remote_get_time_evaluator_(
          ffi::Optional<ffi::Module>(std::nullopt), name, static_cast<int>(dev.device_type),
          dev.device_id, number, repeat, min_repeat_ms, limit_zero_time_iterations,
          cooldown_interval_ms, repeats_to_cooldown, cache_flush_bytes, f_preproc_name);
    }
  }

  ffi::Module LoadModule(std::string name) {
    InitRemoteFunc(&remote_load_module_, "tvm.rpc.server.load_module");
    return remote_load_module_(name);
  }

  void ImportModule(const ffi::Module& other) final {
    InitRemoteFunc(&remote_import_module_, "tvm.rpc.server.ImportModule");
    remote_import_module_(ffi::GetRef<ffi::Module>(this), other);
  }

  const std::shared_ptr<RPCSession>& sess() { return sess_; }

  void* module_handle() const { return module_handle_; }

 private:
  template <typename FType>
  void InitRemoteFunc(FType* func, const std::string& name) {
    if (*func != nullptr) return;
    RPCSession::PackedFuncHandle handle = sess_->GetFunction(name);
    TVM_FFI_ICHECK(handle != nullptr) << "Cannot found remote function " << name;
    *func = WrapRemoteFunc(handle);
  }

  ffi::Function WrapRemoteFunc(RPCSession::PackedFuncHandle handle) {
    if (handle == nullptr) return ffi::Function();
    auto wf = std::make_shared<RPCWrappedFunc>(handle, sess_);
    return ffi::Function(
        [wf](ffi::PackedArgs args, ffi::Any* rv) { return wf->operator()(args, rv); });
  }

  // The module handle
  void* module_handle_{nullptr};
  // The local channel
  std::shared_ptr<RPCSession> sess_;
  // remote function to get time evaluator
  ffi::TypedFunction<ffi::Function(ffi::Optional<ffi::Module>, std::string, int, int, int, int, int,
                                   int, int, int, int, std::string)>
      remote_get_time_evaluator_;
  // remote function getter for modules.
  ffi::TypedFunction<ffi::Function(ffi::Module, std::string, bool)> remote_mod_get_function_;
  // remote function getter for load module
  ffi::TypedFunction<ffi::Module(std::string)> remote_load_module_;
  // remote function getter for load module
  ffi::TypedFunction<void(ffi::Module, ffi::Module)> remote_import_module_;
};

void* RPCWrappedFunc::UnwrapRemoteValueToHandle(const AnyView& arg) const {
  // TODO(tqchen): only support Module unwrapping for now.
  if (arg.type_index() == ffi::TypeIndex::kTVMFFIModule) {
    ffi::Module mod = arg.cast<ffi::Module>();
    std::string tkey = mod->kind();
    TVM_FFI_CHECK_EQ(tkey, "rpc", ValueError) << "Cannot pass a non-RPC module to remote";
    auto* rmod = static_cast<RPCModuleObj*>(mod.operator->());
    TVM_FFI_CHECK(rmod->sess() == sess_, ValueError)
        << "Cannot pass in module into a different remote session";
    return rmod->module_handle();
  } else {
    TVM_FFI_THROW(ValueError) << "Cannot pass type " << arg.GetTypeKey()
                              << " as an argument to the remote";
    return nullptr;
  }
}

void RPCWrappedFunc::WrapRemoteReturnToValue(ffi::PackedArgs args, ffi::Any* rv) const {
  int type_index = args[0].cast<int>();
  if (type_index == ffi::TypeIndex::kTVMFFINone) {
    *rv = nullptr;
    return;
  } else if (type_index == ffi::TypeIndex::kTVMFFIDevice) {
    *rv = MakeRemoteDevice(sess_, args[1].cast<DLDevice>());
  } else if (type_index == ffi::TypeIndex::kTVMFFIFunction) {
    TVM_FFI_ICHECK_EQ(args.size(), 2);
    void* handle = args[1].cast<void*>();
    auto wf = std::make_shared<RPCWrappedFunc>(handle, sess_);
    *rv = ffi::Function(
        [wf](ffi::PackedArgs args, ffi::Any* rv) { return wf->operator()(args, rv); });
  } else if (type_index == ffi::TypeIndex::kTVMFFIModule) {
    TVM_FFI_ICHECK_EQ(args.size(), 2);
    void* handle = args[1].cast<void*>();
    auto n = ffi::make_object<RPCModuleObj>(handle, sess_);
    *rv = ffi::Module(n);
  } else if (type_index == ffi::TypeIndex::kTVMFFITensor ||
             type_index == ffi::TypeIndex::kTVMFFIDLTensorPtr) {
    TVM_FFI_ICHECK_EQ(args.size(), 3);
    auto tensor = args[1].cast<DLTensor*>();
    void* nd_handle = args[2].cast<void*>();
    *rv = RemoteTensorFromHandle(sess_, *tensor, nd_handle);
  } else if (type_index == ffi::TypeIndex::kTVMFFIBytes ||
             type_index == ffi::TypeIndex::kTVMFFIStr ||
             type_index == ffi::TypeIndex::kTVMFFISmallStr ||
             type_index == ffi::TypeIndex::kTVMFFISmallBytes) {
    TVM_FFI_ICHECK_EQ(args.size(), 2);
    *rv = args[1];
  } else if (type_index >= ffi::TypeIndex::kTVMFFIStaticObjectBegin) {
    TVM_FFI_ICHECK_EQ(args.size(), 2);
    void* handle = args[1].cast<void*>();
    auto n = ffi::make_object<RPCObjectRefObj>(handle, sess_);
    *rv = ffi::ObjectRef(n);
  } else {
    TVM_FFI_ICHECK_EQ(args.size(), 2);
    *rv = args[1];
  }
}

ffi::Module CreateRPCSessionModule(std::shared_ptr<RPCSession> sess) {
  auto n = ffi::make_object<RPCModuleObj>(nullptr, sess);
  return ffi::Module(n);
}

std::shared_ptr<RPCSession> RPCModuleGetSession(ffi::Module mod) {
  std::string tkey = mod->kind();
  TVM_FFI_CHECK_EQ(tkey, "rpc", ValueError) << "Cannot pass a non-RPC module to remote";
  auto* rmod = static_cast<RPCModuleObj*>(mod.operator->());
  return rmod->sess();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "runtime.RPCTimeEvaluator",
      [](ffi::Optional<ffi::Module> opt_mod, std::string name, int device_type, int device_id,
         int number, int repeat, int min_repeat_ms, int limit_zero_time_iterations,
         int cooldown_interval_ms, int repeats_to_cooldown, int cache_flush_bytes,
         std::string f_preproc_name) {
        Device dev{static_cast<DLDeviceType>(device_type), device_id};
        if (opt_mod.has_value() && std::string(opt_mod.value()->kind()) == "rpc") {
          return static_cast<RPCModuleObj*>(opt_mod.value().operator->())
              ->GetTimeEvaluator(name, dev, number, repeat, min_repeat_ms,
                                 limit_zero_time_iterations, cooldown_interval_ms,
                                 repeats_to_cooldown, cache_flush_bytes, f_preproc_name);
        }
        return ffi::Function::GetGlobalRequired("runtime.LocalTimeEvaluator")(
                   opt_mod, name, device_type, device_id, number, repeat, min_repeat_ms,
                   limit_zero_time_iterations, cooldown_interval_ms, repeats_to_cooldown,
                   cache_flush_bytes, f_preproc_name)
            .cast<ffi::Function>();
      });
}

// server function registration.
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tvm.rpc.server.ImportModule",
           [](ffi::Module parent, ffi::Module child) { parent->ImportModule(child); })
      .def("tvm.rpc.server.ModuleGetFunction",
           [](ffi::Module parent, std::string name, bool query_imports) {
             return parent->GetFunction(name, query_imports);
           });
}

// functions to access an RPC module.
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("rpc.LoadRemoteModule",
           [](ffi::Module sess, std::string name) {
             std::string tkey = sess->kind();
             TVM_FFI_ICHECK_EQ(tkey, "rpc");
             return static_cast<RPCModuleObj*>(sess.operator->())->LoadModule(name);
           })
      .def("rpc.ImportRemoteModule",
           [](ffi::Module parent, ffi::Module child) {
             std::string tkey = parent->kind();
             TVM_FFI_ICHECK_EQ(tkey, "rpc");
             static_cast<RPCModuleObj*>(parent.operator->())->ImportModule(child);
           })
      .def("rpc.ValidateDevice",
           [](ffi::Module mod, RemoteDevice dev) {
             auto sess = RPCModuleGetSession(mod);
             sess->EnsureOpen();
             TVM_FFI_CHECK(sess == dev->sess, ValueError)
                 << "Device belongs to a different remote session";
             return dev->endpoint_device;
           })
      .def("rpc.ModuleTimeEvaluator",
           [](ffi::Module mod, std::string name, ffi::AnyView device, int number, int repeat,
              int min_repeat_ms, int limit_zero_time_iterations, int cooldown_interval_ms,
              int repeats_to_cooldown, int cache_flush_bytes, std::string f_preproc_name) {
             auto dev = device.cast<RemoteDevice>();
             auto sess = RPCModuleGetSession(mod);
             sess->EnsureOpen();
             TVM_FFI_CHECK(sess == dev->sess, ValueError)
                 << "Device belongs to a different remote session";
             return static_cast<RPCModuleObj*>(mod.operator->())
                 ->GetTimeEvaluator(name, dev->endpoint_device, number, repeat, min_repeat_ms,
                                    limit_zero_time_iterations, cooldown_interval_ms,
                                    repeats_to_cooldown, cache_flush_bytes, f_preproc_name);
           });
}

}  // namespace runtime
}  // namespace tvm
