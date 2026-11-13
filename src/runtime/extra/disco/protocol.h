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
#ifndef TVM_RUNTIME_DISCO_PROTOCOL_H_
#define TVM_RUNTIME_DISCO_PROTOCOL_H_

#include <tvm/ffi/extra/base64.h>
#include <tvm/ffi/extra/json.h>
#include <tvm/ffi/extra/serialization.h>
#include <tvm/runtime/base.h>
#include <tvm/runtime/disco/session.h>
#include <tvm/support/bytes_io.h>
#include <tvm/support/io.h>
#include <tvm/support/serializer.h>

#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "../../../support/arena.h"
#include "../../rpc/minrpc/rpc_reference.h"

namespace tvm {
namespace runtime {

/*!
 * \brief The communication protocol used by Disco message channel.
 * \tparam SubClassType The subclass type that inherits this protocol.
 */
template <class SubClassType>
struct DiscoProtocol {
 protected:
  /*! \brief Virtual destructor */
  virtual ~DiscoProtocol() = default;

  /*! \brief Recycle all the memory used in the arena */
  inline void RecycleAll() {
    this->any_arena_.clear();
    this->arena_.RecycleAll();
  }

  /*! \brief Get the length of the object being serialized. Used by RPCReference. */
  inline uint64_t GetFFIAnyProtocolBytes(const TVMFFIAny* obj);

  /*! \brief Write the object to stream. Used by RPCReference. */
  inline void WriteFFIAny(const TVMFFIAny* obj);

  /*! \brief Read the object from stream. Used by RPCReference. */
  inline void ReadFFIAny(TVMFFIAny* out);

  /*! \brief Callback method used when starting a new message. Used by RPCReference. */
  void MessageStart(uint64_t packet_nbytes) {}

  /*! \brief Callback method used when a new message is complete. Used by RPCReference. */
  void MessageDone() {}

  /*! \brief Callback method when an error occurs in (de)-serialization. Used by RPCReference. */
  void ThrowError(RPCServerStatus status) {
    TVM_FFI_THROW(InternalError) << "Unexpected error in RPC: " << RPCServerStatusToString(status);
  }

  /*!\ brief Arena used by RPCReference to allocate POD memory */
  template <typename T>
  T* ArenaAlloc(int count) {
    static_assert(std::is_standard_layout<T>::value && std::is_trivial<T>::value,
                  "need to be trivial");
    return arena_.template allocate_<T>(count);
  }

  support::Arena arena_;
  std::vector<Any> any_arena_;
  friend struct RPCReference;
};

/*!
 * \brief The debug extension of the communication protocol that allows serialization and
 * deserialization of Tensors and reflection-capable TVM objects.
 */
struct DiscoDebugObject : public ffi::Object {
 public:
  /*! \brief The data to be serialized */
  ffi::Any data;

  /*! \brief Wrap an Tensor or reflection-capable TVM object into the debug extension. */
  static ffi::ObjectRef Wrap(const ffi::Any& data) {
    ffi::ObjectPtr<DiscoDebugObject> n = ffi::make_object<DiscoDebugObject>();
    n->data = data;
    return ffi::ObjectRef(n);
  }

  /*! \brief Wrap an Tensor or reflection-capable TVM object into the debug extension. */
  static ffi::ObjectRef Wrap(const ffi::AnyView& data) {
    ffi::Any rv;
    rv = data;
    return Wrap(std::move(rv));
  }

  /*! \brief Serialize the debug object to string */
  inline std::string SaveToStr() const;
  /*! \brief Deserialize the debug object from string */
  static inline ffi::ObjectPtr<DiscoDebugObject> LoadFromStr(std::string json_str);
  /*! \brief Get the size of the debug object in bytes */
  inline uint64_t GetFFIAnyProtocolBytes() const {
    return sizeof(uint64_t) + this->SaveToStr().size();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("runtime.disco.DiscoDebugObject", DiscoDebugObject, SessionObj);
};

template <class SubClassType>
inline uint64_t DiscoProtocol<SubClassType>::GetFFIAnyProtocolBytes(const TVMFFIAny* value) {
  const AnyView* any_view_ptr = reinterpret_cast<const AnyView*>(value);
  if (any_view_ptr->as<DRefObj>()) {
    return sizeof(uint32_t) + sizeof(int64_t);
  } else if (const auto opt_str = any_view_ptr->as<ffi::String>()) {
    uint64_t size = (*opt_str).size();
    return sizeof(uint32_t) + sizeof(uint64_t) + size * sizeof(char);
  } else if (const auto opt_bytes = any_view_ptr->as<ffi::Bytes>()) {
    uint64_t size = (*opt_bytes).size();
    return sizeof(uint32_t) + sizeof(uint64_t) + size * sizeof(char);
  } else if (const auto opt_shape = any_view_ptr->as<ffi::Shape>()) {
    uint64_t ndim = (*opt_shape).size();
    return sizeof(uint32_t) + sizeof(uint64_t) + ndim * sizeof(ffi::ShapeObj::index_type);
  } else if (const auto opt_debug_obj = any_view_ptr->as<DiscoDebugObject>()) {
    return sizeof(uint32_t) + (*opt_debug_obj).GetFFIAnyProtocolBytes();
  } else {
    TVM_FFI_THROW(ValueError) << "Object type is not supported in Disco calling convention: "
                              << any_view_ptr->GetTypeKey()
                              << " (type_index = " << any_view_ptr->type_index() << ")";
    return 0;
  }
}
template <class SubClassType>
inline void DiscoProtocol<SubClassType>::WriteFFIAny(const TVMFFIAny* value) {
  SubClassType* self = static_cast<SubClassType*>(this);
  const AnyView* any_view_ptr = reinterpret_cast<const AnyView*>(value);
  if (const auto* ref = any_view_ptr->as<DRefObj>()) {
    int64_t reg_id = ref->reg_id;
    self->template Write<uint32_t>(kRuntimeDiscoDRef);
    self->template Write<int64_t>(reg_id);
  } else if (const auto opt_str = any_view_ptr->as<ffi::String>()) {
    self->template Write<uint32_t>(ffi::TypeIndex::kTVMFFIStr);
    self->template Write<uint64_t>((*opt_str).size());
    self->template WriteArray<char>((*opt_str).data(), (*opt_str).size());
  } else if (const auto opt_bytes = any_view_ptr->as<ffi::Bytes>()) {
    self->template Write<uint32_t>(ffi::TypeIndex::kTVMFFIBytes);
    self->template Write<uint64_t>((*opt_bytes).size());
    self->template WriteArray<char>((*opt_bytes).data(), (*opt_bytes).size());
  } else if (const auto opt_shape = any_view_ptr->as<ffi::Shape>()) {
    self->template Write<uint32_t>(ffi::TypeIndex::kTVMFFIShape);
    self->template Write<uint64_t>((*opt_shape).size());
    self->template WriteArray<ffi::ShapeObj::index_type>((*opt_shape).data(), (*opt_shape).size());
  } else if (const auto opt_debug_obj = any_view_ptr->as<DiscoDebugObject>()) {
    self->template Write<uint32_t>(0);
    std::string str = (*opt_debug_obj).SaveToStr();
    self->template Write<uint64_t>(str.size());
    self->template WriteArray<char>(str.data(), str.size());
  } else {
    TVM_FFI_THROW(ValueError) << "Object type is not supported in Disco calling convention: "
                              << any_view_ptr->GetTypeKey()
                              << " (type_index = " << any_view_ptr->type_index() << ")";
  }
}

template <class SubClassType>
inline void DiscoProtocol<SubClassType>::ReadFFIAny(TVMFFIAny* out) {
  SubClassType* self = static_cast<SubClassType*>(this);
  ffi::Any result{nullptr};
  uint32_t type_index;
  self->template Read<uint32_t>(&type_index);
  if (type_index == kRuntimeDiscoDRef) {
    ffi::ObjectPtr<DRefObj> dref = ffi::make_object<DRefObj>();
    self->template Read<int64_t>(&dref->reg_id);
    result = ffi::ObjectRef(std::move(dref));
  } else if (type_index == ffi::TypeIndex::kTVMFFIStr) {
    uint64_t size = 0;
    self->template Read<uint64_t>(&size);
    std::string data(size, '\0');
    self->template ReadArray<char>(data.data(), size);
    result = ffi::String(std::move(data));
  } else if (type_index == ffi::TypeIndex::kTVMFFIBytes) {
    uint64_t size = 0;
    self->template Read<uint64_t>(&size);
    std::string data(size, '\0');
    self->template ReadArray<char>(data.data(), size);
    result = ffi::Bytes(std::move(data));
  } else if (type_index == ffi::TypeIndex::kTVMFFIShape) {
    uint64_t ndim = 0;
    self->template Read<uint64_t>(&ndim);
    std::vector<ffi::ShapeObj::index_type> data(ndim);
    self->template ReadArray<ffi::ShapeObj::index_type>(data.data(), ndim);
    result = ffi::Shape(std::move(data));
  } else if (type_index == 0) {
    uint64_t size = 0;
    self->template Read<uint64_t>(&size);
    std::string data(size, '\0');
    self->template ReadArray<char>(data.data(), size);
    result = DiscoDebugObject::LoadFromStr(std::move(data))->data.cast<ffi::ObjectRef>();
  } else {
    TVM_FFI_THROW(ValueError) << "Object type is not supported in Disco calling convention: "
                              << ffi::Object::TypeIndex2Key(type_index)
                              << " (type_index = " << type_index << ")";
  }
  *reinterpret_cast<ffi::AnyView*>(out) = result;
  any_arena_.push_back(result);
}

inline std::string DiscoDebugObject::SaveToStr() const {
  if (auto opt_nd = this->data.as<ffi::Tensor>(); opt_nd && opt_nd->defined()) {
    ffi::Tensor array = opt_nd.value();
    std::string result;
    {
      support::BytesOutStream mstrm(&result);
      runtime::SaveTensor(array, &mstrm);
    }
    result = ffi::Base64Encode(ffi::Bytes(result));
    result.push_back('1');
    return result;
  } else if (auto opt_obj = this->data.as<ffi::ObjectRef>()) {
    ffi::ObjectRef obj = opt_obj.value();
    std::string result = ffi::json::Stringify(ffi::ToJSONGraph(obj));
    result.push_back('0');
    return result;
  }
  TVM_FFI_THROW(ValueError) << "Cannot serialize the following type code in non-debugging mode: "
                            << this->data.GetTypeKey();
  return "";
}

inline ffi::ObjectPtr<DiscoDebugObject> DiscoDebugObject::LoadFromStr(std::string json_str) {
  TVM_FFI_ICHECK(!json_str.empty());
  char control_bit = json_str.back();
  json_str.pop_back();
  ffi::ObjectPtr<DiscoDebugObject> result = ffi::make_object<DiscoDebugObject>();
  if (control_bit == '0') {
    result->data = ffi::FromJSONGraph(ffi::json::Parse(json_str));
  } else if (control_bit == '1') {
    ffi::Bytes bytes = ffi::Base64Decode(json_str);
    support::BytesInStream mstrm(bytes.data(), bytes.size());
    ffi::Tensor array = LoadTensor(&mstrm);
    result->data = std::move(array);
  } else {
    TVM_FFI_THROW(ValueError) << "Unsupported control bit: " << control_bit
                              << ". Full string: " << json_str;
  }
  return result;
}

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_DISCO_PROTOCOL_H_
