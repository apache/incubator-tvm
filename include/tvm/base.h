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
 * \file tvm/base.h
 * \brief Parent compiler symbol visibility.
 */
#ifndef TVM_BASE_H_
#define TVM_BASE_H_

// TVM_EXPORTS selects compiler exports on MSVC; consumers import them.
// TVM_DLL_EXPORT always exports. Embedders may override TVM_DLL.
#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

// --- TVM_DLL family (libtvm_compiler) ---
#if !defined(TVM_DLL) && defined(__EMSCRIPTEN__)
#define TVM_DLL EMSCRIPTEN_KEEPALIVE
#define TVM_DLL_EXPORT EMSCRIPTEN_KEEPALIVE
#endif
#if !defined(TVM_DLL) && defined(_MSC_VER)
#ifdef TVM_EXPORTS
#define TVM_DLL __declspec(dllexport)
#else
#define TVM_DLL __declspec(dllimport)
#endif
#define TVM_DLL_EXPORT __declspec(dllexport)
#endif
#ifndef TVM_DLL
#define TVM_DLL __attribute__((visibility("default")))
#define TVM_DLL_EXPORT __attribute__((visibility("default")))
#endif

#endif  // TVM_BASE_H_
