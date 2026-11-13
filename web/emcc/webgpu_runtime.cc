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
 * \file webgpu_runtime.cc
 * \brief WebGPU runtime based on the TVM JS.
 */

// configurations for tvm logging
#define TVM_LOG_DEBUG 0
#define TVM_LOG_CUSTOMIZE 1
#define TVM_FFI_ALWAYS_LOG_BEFORE_THROW 1

#include <tvm/ffi/extra/json.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/device_api.h>
#include <tvm/support/bytes_io.h>

#include <algorithm>
#include <iostream>
#include <mutex>
#include <string>

#include "../../3rdparty/tvm-tirx/src/runtime/metadata.h"

namespace tvm {
namespace runtime {

// All the implementations are redirectly to the JS side.
class WebGPUDeviceAPI : public DeviceAPI {
 public:
  void SetDevice(Device dev) final {}
  ffi::Any GetAttr(Device dev, const ffi::String& name) final {
    if (name == "exist") {
      return dev.device_id == 0 && ffi::Function::GetGlobal("wasm.WebGPUDeviceAPI").has_value();
    }
    return nullptr;
  }

  ffi::Tensor Empty(ffi::ShapeView shape, DLDataType dtype, DLDevice device,
                    ffi::Optional<MemoryScope> mem_scope = std::nullopt) final {
    TVM_FFI_CHECK(!mem_scope.has_value() || mem_scope.value() == MemoryScope::Global(), ValueError)
        << "Unsupported memory scope: " << mem_scope.value_or(MemoryScope::Global());
    struct Allocator {
      WebGPUDeviceAPI* api;
      void AllocData(DLTensor* tensor) {
        size_t alignment =
            std::max(static_cast<size_t>(kAllocAlignment),
                     static_cast<size_t>((tensor->dtype.bits / 8) * tensor->dtype.lanes));
        tensor->data = api->AllocDataSpace(tensor->device, ffi::GetDataSize(*tensor), alignment,
                                           tensor->dtype);
      }
      void FreeData(DLTensor* tensor) { api->FreeDataSpace(tensor->device, tensor->data); }
    };
    return ffi::Tensor::FromNDAlloc(Allocator{this}, shape, dtype, device);
  }

 private:
  void* AllocDataSpace(Device dev, size_t nbytes, size_t alignment, DLDataType type_hint) {
    Initialize();
    double ptr_number = alloc_space_(nbytes);
    return reinterpret_cast<void*>(static_cast<int64_t>(ptr_number));
  }

  void FreeDataSpace(Device dev, void* ptr) { return free_space_(ptr); }

 protected:
  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
                      Device dev_from, Device dev_to, DLDataType type_hint,
                      TVMStreamHandle stream) final {
    Initialize();
    if (static_cast<int>(dev_from.device_type) == kDLWebGPU &&
        static_cast<int>(dev_to.device_type) == kDLWebGPU) {
      TVM_FFI_ICHECK_EQ(dev_from.device_id, dev_to.device_id);
      copy_within_gpu_(const_cast<void*>(from), from_offset, to, to_offset, size);
    } else if (static_cast<int>(dev_from.device_type) == kDLWebGPU &&
               dev_to.device_type == kDLCPU) {
      void* to_ptr = static_cast<uint8_t*>(to) + to_offset;
      copy_from_gpu_(const_cast<void*>(from), from_offset, to_ptr, size);
    } else if (dev_from.device_type == kDLCPU &&
               static_cast<int>(dev_to.device_type) == kDLWebGPU) {
      void* from_ptr = static_cast<uint8_t*>(const_cast<void*>(from)) + from_offset;
      copy_to_gpu_(from_ptr, to, to_offset, size);
    } else {
      TVM_FFI_THROW(InternalError) << "expect copy from/to WebGPU or between WebGPU";
    }
  }

 public:
  TVMStreamHandle CreateStream(Device dev) final {
    TVM_FFI_THROW(InternalError) << "Not implemented";
  }

  void FreeStream(Device dev, TVMStreamHandle stream) final {
    TVM_FFI_THROW(InternalError) << "Not implemented";
  }

  void SyncStreamFromTo(Device dev, TVMStreamHandle event_src, TVMStreamHandle event_dst) {
    TVM_FFI_THROW(InternalError) << "Not implemented";
  }

  void StreamSync(Device dev, TVMStreamHandle stream) final {
    static auto func = tvm::ffi::Function::GetGlobal("__asyncify.WebGPUWaitForTasks");
    TVM_FFI_ICHECK(func.has_value()) << "Stream sync inside c++ only supported in asyncify mode";
    (*func)();
  }

  static WebGPUDeviceAPI* Global() {
    static WebGPUDeviceAPI* inst = new WebGPUDeviceAPI();
    return inst;
  }

 private:
  void Initialize() {
    std::call_once(initialized_, [this]() {
      auto fp = tvm::ffi::Function::GetGlobal("wasm.WebGPUDeviceAPI");
      TVM_FFI_ICHECK(fp.has_value()) << "Cannot find wasm.WebGPUContext in the env";
      auto getter = ffi::TypedFunction<ffi::Function(std::string)>(*fp);
      alloc_space_ = getter("deviceAllocDataSpace");
      free_space_ = getter("deviceFreeDataSpace");
      copy_to_gpu_ = getter("deviceCopyToGPU");
      copy_from_gpu_ = getter("deviceCopyFromGPU");
      copy_within_gpu_ = getter("deviceCopyWithinGPU");
    });
  }

  std::once_flag initialized_;
  // NOTE: js return number as double.
  ffi::TypedFunction<double(int64_t nbytes)> alloc_space_;
  ffi::TypedFunction<void(void* ptr)> free_space_;
  ffi::TypedFunction<void(void* from, void* to, int64_t to_offset, int64_t nbytes)> copy_to_gpu_;
  ffi::TypedFunction<void(void* from, int64_t from_offset, void* to, int64_t nbytes)>
      copy_from_gpu_;
  ffi::TypedFunction<void(void* from, int64_t from_offset, void* to, int64_t to_offset,
                          int64_t nbytes)>
      copy_within_gpu_;
};

class WebGPUModuleObj final : public ffi::ModuleObj {
 public:
  explicit WebGPUModuleObj(std::unordered_map<std::string, std::string> smap,
                           ffi::Map<ffi::String, FunctionInfo> fmap)
      : smap_(smap), fmap_(fmap) {
    auto fp = tvm::ffi::Function::GetGlobal("wasm.WebGPUCreateShader");
    TVM_FFI_ICHECK(fp.has_value());
    create_shader_ = *fp;
  }

  const char* kind() const final { return "webgpu"; }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final {
    // special function
    if (name == "webgpu.get_fmap") {
      return ffi::Function([this](ffi::PackedArgs args, ffi::Any* rv) {
        namespace json = ::tvm::ffi::json;
        json::Object obj;
        for (const auto& kv : fmap_) {
          obj.Set(kv.first, kv.second->SaveToJSON());
        }
        *rv = std::string(json::Stringify(obj));
      });
    } else if (name == "webgpu.get_shader") {
      return ffi::Function([this](ffi::PackedArgs args, ffi::Any* rv) {
        auto name = args[0].cast<std::string>();
        auto it = smap_.find(name);
        TVM_FFI_ICHECK(it != smap_.end()) << "Cannot find code " << name;
        *rv = it->second;
      });
    } else if (name == "webgpu.update_prebuild") {
      return ffi::Function([this](ffi::PackedArgs args, ffi::Any* rv) {
        auto name = args[0].cast<std::string>();
        ffi::Function func = args[1].cast<ffi::Function>();
        prebuild_[name] = func;
      });
    }
    // check prebuild cache
    auto prebuild_it = prebuild_.find(name);
    if (prebuild_it != prebuild_.end()) {
      return prebuild_it->second;
    }

    auto it = smap_.find(name);
    if (it != smap_.end()) {
      auto opt_info = fmap_.Get(name);
      TVM_FFI_ICHECK(opt_info.has_value());
      FunctionInfo orig_info = opt_info.value();
      FunctionInfo info(name, orig_info->arg_types, orig_info->launch_param_tags,
                        orig_info->arg_extra_tags);
      namespace json = ::tvm::ffi::json;
      std::string info_json = std::string(json::Stringify(info->SaveToJSON()));
      return create_shader_(info_json, it->second);
    } else {
      return std::nullopt;
    }
  }

  int GetPropertyMask() const final { return ffi::Module::kBinarySerializable; };

  ffi::Bytes SaveToBytes() const final { TVM_FFI_THROW(InternalError) << "Not implemented"; }

  ffi::String InspectSource(const ffi::String& format) const final {
    // can only return source code.
    return source_;
  }

 private:
  // code table
  std::unordered_map<std::string, std::string> smap_;
  // function information table.
  ffi::Map<ffi::String, FunctionInfo> fmap_;
  // The source
  std::string source_;
  // prebuild_ functions
  std::unordered_map<std::string, ffi::Function> prebuild_;
  // Callback to get the GPU function.
  ffi::TypedFunction<ffi::Function(std::string finfo, std::string shader)> create_shader_;
};

ffi::Module WebGPUModuleLoadFromBytes(const ffi::Bytes& bytes) {
  support::BytesInStream stream(bytes);
  std::unordered_map<std::string, std::string> smap;

  ffi::Map<ffi::String, FunctionInfo> fmap;
  TVM_FFI_ICHECK(stream.Read(&fmap));
  stream.Read(&smap);
  return ffi::Module(ffi::make_object<WebGPUModuleObj>(smap, fmap));
}

// for now webgpu is hosted via a vulkan module.
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("ffi.Module.load_from_bytes.webgpu", WebGPUModuleLoadFromBytes)
      .def_packed("device_api.webgpu", [](ffi::PackedArgs args, ffi::Any* rv) {
        DeviceAPI* ptr = WebGPUDeviceAPI::Global();
        *rv = static_cast<void*>(ptr);
      });
}

}  // namespace runtime
}  // namespace tvm
