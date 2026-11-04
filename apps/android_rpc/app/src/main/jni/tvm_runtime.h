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
 * \file tvm_runtime.h
 * \brief Pack all tvm runtime source files
 */
#include <sys/stat.h>

#include <fstream>

/* Enable custom logging - this will cause TVM to use a custom implementation
 * of tvm::runtime::detail::LogMessage. We use this to pass TVM log messages to
 * Android logcat.
 */
#define TVM_LOG_CUSTOMIZE 1
#define TVM_FFI_USE_LIBBACKTRACE 0

// FFI registration precedes the runtime implementation.
// clang-format off
#include "../3rdparty/tvm-tirx/3rdparty/tvm-ffi/src/ffi/backtrace.cc"
#include "../3rdparty/tvm-tirx/3rdparty/tvm-ffi/src/ffi/container.cc"
#include "../3rdparty/tvm-tirx/3rdparty/tvm-ffi/src/ffi/dtype.cc"
#include "../3rdparty/tvm-tirx/3rdparty/tvm-ffi/src/ffi/error.cc"
#include "../3rdparty/tvm-tirx/3rdparty/tvm-ffi/src/ffi/extra/library_module.cc"
#include "../3rdparty/tvm-tirx/3rdparty/tvm-ffi/src/ffi/extra/library_module_dynamic_lib.cc"
#include "../3rdparty/tvm-tirx/3rdparty/tvm-ffi/src/ffi/extra/library_module_system_lib.cc"
#include "../3rdparty/tvm-tirx/3rdparty/tvm-ffi/src/ffi/extra/module.cc"
#include "../3rdparty/tvm-tirx/3rdparty/tvm-ffi/src/ffi/function.cc"
#include "../3rdparty/tvm-tirx/3rdparty/tvm-ffi/src/ffi/object.cc"
#include "../3rdparty/tvm-tirx/3rdparty/tvm-ffi/src/ffi/tensor.cc"
#include "../src/runtime/registry.cc"
#include "../src/runtime/rpc/rpc_channel.cc"
#include "../src/runtime/rpc/rpc_endpoint.cc"
#include "../src/runtime/rpc/rpc_event_impl.cc"
#include "../src/runtime/rpc/rpc_local_session.cc"
#include "../src/runtime/rpc/rpc_module.cc"
#include "../src/runtime/rpc/rpc_server_env.cc"
#include "../src/runtime/rpc/rpc_session.cc"
#include "../src/runtime/rpc/rpc_socket_impl.cc"
#include "../src/runtime/timer.cc"
#include "../src/runtime/vm/params.cc"
#include "../3rdparty/tvm-tirx/src/runtime/cpu_device_api.cc"
#include "../3rdparty/tvm-tirx/src/runtime/device_api.cc"
#include "../3rdparty/tvm-tirx/src/runtime/logging.cc"
#include "../3rdparty/tvm-tirx/src/runtime/memory_scope.cc"
#include "../3rdparty/tvm-tirx/src/runtime/metadata.cc"
#include "../3rdparty/tvm-tirx/src/runtime/tensor.cc"
#include "../3rdparty/tvm-tirx/src/runtime/thread_pool.cc"
#include "../3rdparty/tvm-tirx/src/runtime/threading_backend.cc"
// clang-format on

#ifdef TVM_OPENCL_RUNTIME
#include "../src/runtime/opencl/opencl_device_api.cc"
#include "../src/runtime/opencl/opencl_module.cc"
#include "../src/runtime/opencl/opencl_wrapper/opencl_wrapper.cc"
#include "../src/runtime/source_utils.cc"
#endif

#ifdef TVM_VULKAN_RUNTIME
#include "../src/runtime/vulkan/vulkan_amdrgp.cc"
#include "../src/runtime/vulkan/vulkan_buffer.cc"
#include "../src/runtime/vulkan/vulkan_common.cc"
#include "../src/runtime/vulkan/vulkan_device.cc"
#include "../src/runtime/vulkan/vulkan_device_api.cc"
#include "../src/runtime/vulkan/vulkan_instance.cc"
#include "../src/runtime/vulkan/vulkan_module.cc"
#include "../src/runtime/vulkan/vulkan_stream.cc"
#include "../src/runtime/vulkan/vulkan_wrapped_func.cc"
#endif

#ifdef USE_SORT
#include "../src/runtime/extra/contrib/sort/sort.cc"
#endif

#ifdef USE_RANDOM
#include "../src/runtime/extra/contrib/random/random.cc"
#endif

#include <android/log.h>

namespace tvm {
namespace runtime {
namespace detail {
// Override logging mechanism
[[noreturn]] void LogFatalImpl(const std::string& file, int lineno, const std::string& message) {
  std::string m = file + ":" + std::to_string(lineno) + ": " + message;
  __android_log_write(ANDROID_LOG_FATAL, "TVM_RUNTIME", m.c_str());
  throw tvm::ffi::Error("InternalError", message, TVMFFIBacktrace(file.c_str(), lineno, "", 0));
}
void LogMessageImpl(const std::string& file, int lineno, int level, const std::string& message) {
  std::string m = file + ":" + std::to_string(lineno) + ": " + message;
  __android_log_write(ANDROID_LOG_DEBUG + level, "TVM_RUNTIME", m.c_str());
}

}  // namespace detail
}  // namespace runtime
}  // namespace tvm
