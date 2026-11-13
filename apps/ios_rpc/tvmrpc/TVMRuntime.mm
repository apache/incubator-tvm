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
 * \file TVMRuntime.mm
 */

#import <Foundation/Foundation.h>

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include "RPCArgs.h"

// internal TVM header
#include <tvm/support/file_utils.h>

#if defined(USE_CUSTOM_DSO_LOADER) && USE_CUSTOM_DSO_LOADER == 1
// internal TVM header to achieve Library class
#include <../../../3rdparty/tvm-tirx/3rdparty/tvm-ffi/src/ffi/extra/library_module.h>
#include <custom_dlfcn.h>
#endif

namespace tvm {
namespace runtime {
namespace detail {

// Override logging mechanism
[[noreturn]] void LogFatalImpl(const std::string& file, int lineno, const std::string& message) {
  throw tvm::runtime::InternalError(file, lineno, message);
}

void LogMessageImpl(const std::string& file, int lineno, int level, const std::string& message) {
  NSLog(@"%s:%d: %s", file.c_str(), lineno, message.c_str());
}

}  // namespace detail

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("tvm.rpc.server.workpath",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    static const std::string base_ = NSTemporaryDirectory().UTF8String;
                    const auto path = args[0].cast<std::string>();
                    *rv = base_ + "/" + path;
                  })
      .def_packed("tvm.rpc.server.load_module", [](ffi::PackedArgs args, ffi::Any* rv) {
        auto name = args[0].cast<std::string>();
        std::string fmt = support::GetFileFormat(name, "");
        NSString* base;
        if (fmt == "dylib") {
          // only load dylib from frameworks.
          NSBundle* bundle = [NSBundle mainBundle];
          base = [[bundle privateFrameworksPath] stringByAppendingPathComponent:@"tvm"];

          if (tvm::ffi::Function::GetGlobal("ffi.Module.load_from_file.dylib_custom")) {
            // Custom dso loader is present. Will use it.
            base = NSTemporaryDirectory();
            fmt = "dylib_custom";
          }
        } else {
          // Load other modules in tempdir.
          base = NSTemporaryDirectory();
        }
        NSString* path =
            [base stringByAppendingPathComponent:[NSString stringWithUTF8String:name.c_str()]];
        name = [path UTF8String];
        *rv = Module::LoadFromFile(name, fmt);
        LOG(INFO) << "Load module from " << name << " ...";
      });
}

#if defined(USE_CUSTOM_DSO_LOADER) && USE_CUSTOM_DSO_LOADER == 1

// Custom dynamic library loader. Supports unsigned binary
class UnsignedDSOLoader final : public Library {
 public:
  ~UnsignedDSOLoader() {
    if (lib_handle_) {
      custom_dlclose(lib_handle_);
      lib_handle_ = nullptr;
    };
  }
  void Init(const std::string& name) {
    lib_handle_ = custom_dlopen(name.c_str(), RTLD_NOW | RTLD_LOCAL);
    TVM_FFI_ICHECK(lib_handle_ != nullptr)
        << "Failed to load dynamic shared library " << name << " " << custom_dlerror();
  }

  void* GetSymbol(const char* name) final { return custom_dlsym(lib_handle_, name); }

 private:
  // Library handle
  void* lib_handle_{nullptr};
};

// Add UnsignedDSOLoader plugin in global registry
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("ffi.Module.load_from_file.dylib_custom",
                               [](ffi::PackedArgs args, ffi::Any* rv) {
                                 auto n = ffi::make_object<UnsignedDSOLoader>();
                                 n->Init(args[0]);
                                 *rv = tvm::ffi::CreateLibraryModule(n);
                               });
}

#endif

}  // namespace runtime
}  // namespace tvm
