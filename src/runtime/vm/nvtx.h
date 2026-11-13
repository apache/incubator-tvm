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
 * \file src/runtime/vm/nvtx.h
 * \brief NVTX scoped range utility (header-only).
 *
 * Provides NVTXScopedRange: a lightweight RAII wrapper over
 * nvtxRangePush/Pop.  When TVM_NVTX_ENABLED is not defined or is 0,
 * all methods are no-ops compiled away by the optimizer.
 */
#ifndef TVM_RUNTIME_VM_NVTX_H_
#define TVM_RUNTIME_VM_NVTX_H_

#include <string>

#ifndef TVM_NVTX_ENABLED
#define TVM_NVTX_ENABLED 0
#endif

#if TVM_NVTX_ENABLED
#include <nvtx3/nvToolsExt.h>
#endif  // TVM_NVTX_ENABLED

namespace tvm {
namespace support {

/*!
 * \brief A class to create a NVTX range. No-op if TVM is not built against NVTX.
 */
class NVTXScopedRange {
 public:
  /*! \brief Enter an NVTX scoped range */
#if TVM_NVTX_ENABLED
  explicit NVTXScopedRange(const char* name) { nvtxRangePush(name); }
#else
  explicit NVTXScopedRange(const char* name) {}
#endif  // TVM_NVTX_ENABLED
  /*! \brief Enter an NVTX scoped range */
  explicit NVTXScopedRange(const std::string& name) : NVTXScopedRange(name.c_str()) {}
  /*! \brief Exit an NVTX scoped range */
#if TVM_NVTX_ENABLED
  ~NVTXScopedRange() { nvtxRangePop(); }
#else
  ~NVTXScopedRange() {}
#endif  // TVM_NVTX_ENABLED
  NVTXScopedRange(const NVTXScopedRange& other) = delete;
  NVTXScopedRange(NVTXScopedRange&& other) = delete;
  NVTXScopedRange& operator=(const NVTXScopedRange& other) = delete;
  NVTXScopedRange& operator=(NVTXScopedRange&& other) = delete;
};

}  // namespace support
}  // namespace tvm

#endif  // TVM_RUNTIME_VM_NVTX_H_
