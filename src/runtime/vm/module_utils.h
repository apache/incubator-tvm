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
 * \file src/runtime/vm/module_utils.h
 * \brief Internal helpers for declaring VM module vtables.
 *
 * Provides the TVM_MODULE_VTABLE_* macros and ModuleVTableEntryHelper
 * template that VM module implementations use to expose member
 * functions as ffi::Function entries via ffi::ModuleObj::GetFunction.
 *
 * This header is private to src/runtime/vm/. Module implementations
 * outside this directory should not depend on these macros.
 */
#ifndef TVM_RUNTIME_VM_MODULE_UTILS_H_
#define TVM_RUNTIME_VM_MODULE_UTILS_H_

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>

#include <utility>

namespace tvm {
namespace runtime {
namespace vm {
namespace details {

template <typename T>
struct ModuleVTableEntryHelper {};

template <typename T, typename R, typename... Args>
struct ModuleVTableEntryHelper<R (T::*)(Args...) const> {
  using MemFnType = R (T::*)(Args...) const;
  TVM_FFI_INLINE static void Call(ffi::Any* rv, T* self, MemFnType f, ffi::PackedArgs args) {
    auto wrapped = [self, f](Args... args) -> R { return (self->*f)(std::forward<Args>(args)...); };
    ffi::detail::unpack_call<R>(std::make_index_sequence<sizeof...(Args)>{}, nullptr, wrapped,
                                args.data(), args.size(), rv);
  }
};

template <typename T, typename R, typename... Args>
struct ModuleVTableEntryHelper<R (T::*)(Args...)> {
  using MemFnType = R (T::*)(Args...);
  TVM_FFI_INLINE static void Call(ffi::Any* rv, T* self, MemFnType f, ffi::PackedArgs args) {
    auto wrapped = [self, f](Args... args) -> R { return (self->*f)(std::forward<Args>(args)...); };
    ffi::detail::unpack_call<R>(std::make_index_sequence<sizeof...(Args)>{}, nullptr, wrapped,
                                args.data(), args.size(), rv);
  }
};

template <typename T, typename... Args>
struct ModuleVTableEntryHelper<void (T::*)(Args...) const> {
  using MemFnType = void (T::*)(Args...) const;
  TVM_FFI_INLINE static void Call(ffi::Any* rv, T* self, MemFnType f, ffi::PackedArgs args) {
    auto wrapped = [self, f](Args... args) -> void { (self->*f)(std::forward<Args>(args)...); };
    ffi::detail::unpack_call<void>(std::make_index_sequence<sizeof...(Args)>{}, nullptr, wrapped,
                                   args.data(), args.size(), rv);
  }
};

template <typename T, typename... Args>
struct ModuleVTableEntryHelper<void (T::*)(Args...)> {
  using MemFnType = void (T::*)(Args...);
  TVM_FFI_INLINE static void Call(ffi::Any* rv, T* self, MemFnType f, ffi::PackedArgs args) {
    auto wrapped = [self, f](Args... args) -> void { (self->*f)(std::forward<Args>(args)...); };
    ffi::detail::unpack_call<void>(std::make_index_sequence<sizeof...(Args)>{}, nullptr, wrapped,
                                   args.data(), args.size(), rv);
  }
};

}  // namespace details
}  // namespace vm
}  // namespace runtime
}  // namespace tvm

#define TVM_MODULE_VTABLE_BEGIN(TypeKey)                                                      \
  const char* kind() const final { return TypeKey; }                                          \
  ::tvm::ffi::Optional<::tvm::ffi::Function> GetFunction(const ffi::String& _name) override { \
    using SelfPtr = std::remove_cv_t<decltype(this)>;                                         \
    ::tvm::ffi::ObjectPtr<::tvm::ffi::Object> _self =                                         \
        ::tvm::ffi::GetObjectPtr<::tvm::ffi::Object>(this);
#define TVM_MODULE_VTABLE_END() \
  return std::nullopt;          \
  }
#define TVM_MODULE_VTABLE_END_WITH_DEFAULT(MemFunc) \
  {                                                 \
    auto f = (MemFunc);                             \
    return (this->*f)(_name);                       \
  }                                                 \
  }  // NOLINT(*)
#define TVM_MODULE_VTABLE_ENTRY(Name, MemFunc)                                                \
  if (_name == Name) {                                                                        \
    return ffi::Function::FromPacked([_self](ffi::PackedArgs args, ffi::Any* rv) -> void {    \
      using Helper = ::tvm::runtime::vm::details::ModuleVTableEntryHelper<decltype(MemFunc)>; \
      SelfPtr self = static_cast<SelfPtr>(_self.get());                                       \
      Helper::Call(rv, self, MemFunc, args);                                                  \
    });                                                                                       \
  }
#define TVM_MODULE_VTABLE_ENTRY_PACKED(Name, MemFunc)                          \
  if (_name == Name) {                                                         \
    return ffi::Function([_self](ffi::PackedArgs args, ffi::Any* rv) -> void { \
      (static_cast<SelfPtr>(_self.get())->*(MemFunc))(args, rv);               \
    });                                                                        \
  }

#endif  // TVM_RUNTIME_VM_MODULE_UTILS_H_
