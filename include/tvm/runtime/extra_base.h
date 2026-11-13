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
 * \file tvm/runtime/extra_base.h
 * \brief Parent runtime-extra symbol visibility.
 */
#ifndef TVM_RUNTIME_EXTRA_BASE_H_
#define TVM_RUNTIME_EXTRA_BASE_H_

// TVM_RUNTIME_EXTRA_EXPORTS selects runtime-extra exports on MSVC.
// Other libraries and downstream consumers import them. Embedders may
// override TVM_RUNTIME_EXTRA_DLL independently of compiler/core runtime.
#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

// --- TVM_RUNTIME_EXTRA_DLL (libtvm_runtime_extra) ---
#ifndef TVM_RUNTIME_EXTRA_DLL
#if defined(__EMSCRIPTEN__)
#define TVM_RUNTIME_EXTRA_DLL EMSCRIPTEN_KEEPALIVE
#elif defined(_MSC_VER)
#ifdef TVM_RUNTIME_EXTRA_EXPORTS
#define TVM_RUNTIME_EXTRA_DLL __declspec(dllexport)
#else
#define TVM_RUNTIME_EXTRA_DLL __declspec(dllimport)
#endif
#else
#define TVM_RUNTIME_EXTRA_DLL __attribute__((visibility("default")))
#endif
#endif

#endif  // TVM_RUNTIME_EXTRA_BASE_H_
