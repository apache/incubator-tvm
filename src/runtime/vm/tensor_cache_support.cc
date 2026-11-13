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
 * \file src/runtime/vm/tensor_cache_support.cc
 * \brief Runtime to support tensor cache file loading.
 *
 * This file provides a minimum support for tensor cache file loading.
 *
 * The main focus of this implementation is to enable loading
 * with minimum set of intermediate files while also being
 * compatible to some of the multi-shard files that are more
 * friendly in some of the environments.
 *
 * Tensor cache also provides a way to do system-wide
 * parameter sharing across multiple VMs.
 *
 * There are likely other ways to load the parameters ndarray-ache.
 * We will keep the impact minimum by puting it as a private
 * runtime builtin provide as in this file.
 */
#include <tvm/ffi/extra/json.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/extra_base.h>
#include <tvm/runtime/tensor.h>
#include <tvm/runtime/vm/tensor_cache_support.h>
#include <tvm/support/file_utils.h>

#include <string>
#include <vector>

namespace tvm {
namespace runtime {
namespace vm {

namespace json = tvm::ffi::json;

TensorCacheMetadata::FileRecord::ParamRecord JSONAsParamRecord(const json::Object& json) {
  std::vector<ffi::Shape::index_type> shape;
  {
    json::Array shape_json = json["shape"].cast<json::Array>();
    shape.reserve(shape_json.size());
    for (const ffi::Any& d : shape_json) {
      shape.push_back(d.cast<int64_t>());
    }
  }
  TensorCacheMetadata::FileRecord::ParamRecord result;
  std::string dtype = json["dtype"].cast<ffi::String>();
  result.name = json["name"].cast<ffi::String>();
  result.dtype = ffi::StringToDLDataType(dtype);
  result.format = json["format"].cast<ffi::String>();
  result.nbytes = json["nbytes"].cast<int64_t>();
  result.byte_offset = json["byteOffset"].cast<int64_t>();
  result.shape = ffi::Shape(std::move(shape));
  return result;
}

TensorCacheMetadata::FileRecord JSONAsFileRecord(const json::Object& json) {
  json::Array records = json["records"].cast<json::Array>();
  TensorCacheMetadata::FileRecord result;
  result.data_path = json["dataPath"].cast<ffi::String>();
  result.format = json["format"].cast<ffi::String>();
  result.nbytes = json["nbytes"].cast<int64_t>();
  result.records.reserve(records.size());
  for (const ffi::Any& item : records) {
    result.records.push_back(JSONAsParamRecord(item.cast<json::Object>()));
  }
  return result;
}

TensorCacheMetadata JSONAsTensorCacheMetadata(const json::Object& json) {
  json::Array records = json["records"].cast<json::Array>();
  TensorCacheMetadata result;
  result.records.reserve(records.size());
  for (const ffi::Any& item : records) {
    result.records.push_back(JSONAsFileRecord(item.cast<json::Object>()));
  }
  return result;
}

TensorCacheMetadata TensorCacheMetadata::LoadFromStr(const std::string& json_str,
                                                     const std::string& path) {
  ffi::String err;
  json::Value json_info = json::Parse(json_str, &err);
  if (!err.empty()) {
    TVM_FFI_THROW(InternalError) << "Failed to parse JSON: " << err
                                 << ". The JSON string is:" << json_str;
  }
  TVM_FFI_CHECK(json_info.as<json::Object>(), ValueError)
      << "The given string is not a JSON object: " << json_str;
  TensorCacheMetadata result = JSONAsTensorCacheMetadata(json_info.cast<json::Object>());
  result.path = path;
  return result;
}

TVM_RUNTIME_EXTRA_DLL TensorCacheMetadata TensorCacheMetadata::Load(const std::string& path) {
  std::string json_str = support::ReadBytesFromFile(path + "/tensor-cache.json");
  ffi::String err;
  json::Value json_info = json::Parse(json_str, &err);
  if (!err.empty()) {
    TVM_FFI_THROW(InternalError) << "Failed to parse JSON: " << err
                                 << ". The JSON string is:" << json_str;
  }
  TVM_FFI_CHECK(json_info.as<json::Object>(), ValueError)
      << "The given string is not a JSON object: " << json_str;
  TensorCacheMetadata result = JSONAsTensorCacheMetadata(json_info.cast<json::Object>());
  result.path = path;
  return result;
}

void CopyTensorFromBytes(ffi::Tensor param, const void* data, size_t nbytes,
                         ffi::Optional<ffi::Tensor>* staging_buffer) {
  TVM_FFI_ICHECK(param.defined());
  Device device = param.device();
  if (device.device_type != kDLOpenCL || staging_buffer == nullptr) {
    tvm::runtime::CopyFromBytes(param, data, nbytes);
    return;
  }
  // Special handle for OpenCL runtime.
  // It creates a host side memory mirror, for every cl_mem that tries to copy data from host
  // which can cause memory issue. Her we use a large staging buffer to postpone deallocation
  if (staging_buffer->has_value()) {
    size_t curr_size = ffi::GetDataSize(*(staging_buffer->value().GetDLTensorPtr()));
    if (curr_size < nbytes) {
      *staging_buffer = std::nullopt;
    }
  }
  if (!staging_buffer->has_value()) {
    *staging_buffer = tvm::runtime::Empty(param.shape(), param.dtype(), param.device());
  }
  ffi::Tensor staging_view =
      tvm::runtime::CreateView(staging_buffer->value(), param.shape(), param.dtype());
  tvm::runtime::CopyFromBytes(staging_view, data, nbytes);
  tvm::runtime::CopyFromTo(staging_view, param);
  DeviceAPI::Get(device)->StreamSync(device, nullptr);
}

ffi::Tensor TensorCacheMetadata::FileRecord::ParamRecord::Load(
    Device device, const std::string* raw_data, ffi::Optional<ffi::Tensor>* staging_buffer) const {
  ffi::Tensor arr = tvm::runtime::Empty(shape, dtype, device);
  if (dtype == DLDataType{kDLFloat, 32, 1} && format == "f32-to-bf16") {
    // decode bf16 to f32
    std::vector<uint16_t> buffer(nbytes / 2);
    std::vector<uint32_t> decoded(nbytes / 2);
    std::memcpy(buffer.data(), raw_data->data() + byte_offset, nbytes);
    for (size_t i = 0; i < buffer.size(); ++i) {
      decoded[i] = static_cast<uint32_t>(buffer[i]) << 16;
    }
    CopyTensorFromBytes(arr, decoded.data(), decoded.size() * sizeof(uint32_t), staging_buffer);
  } else {
    CopyTensorFromBytes(arr, raw_data->data() + byte_offset, nbytes, staging_buffer);
  }
  return arr;
}

TVM_RUNTIME_EXTRA_DLL ffi::Array<ffi::Tensor> TensorCacheMetadata::FileRecord::Load(
    Device device,
    const std::string& path_prefix,  //
    std::string* raw_data_buffer,    //
    ffi::Optional<ffi::Tensor>* staging_buffer) const {
  *raw_data_buffer = support::ReadBytesFromFile(path_prefix + "/" + this->data_path);
  TVM_FFI_CHECK_EQ(this->format, "raw-shard", ValueError) << "Only `raw-shard` format is supported";
  TVM_FFI_CHECK_EQ(this->nbytes, raw_data_buffer->length(), ValueError)
      << "Encountered an corrupted parameter shard. It means it is not downloaded "
         "completely or downloading is interrupted. Please try to download again.";
  ffi::Array<ffi::Tensor> result;
  result.reserve(this->records.size());
  for (const ParamRecord& nd_rec : this->records) {
    result.push_back(nd_rec.Load(device, raw_data_buffer, staging_buffer));
  }
  return result;
}

/*!
 * A Tensor cache to store pre-loaded arrays in the system.
 */
class TensorCache {
 public:
  static TensorCache* Global() {
    static TensorCache* inst = new TensorCache();
    return inst;
  }

  static void Update(ffi::String name, ffi::Tensor arr, bool override) {
    TVM_FFI_ICHECK(arr.defined());
    TensorCache* pool = Global();
    if (!override) {
      TVM_FFI_ICHECK_EQ(pool->pool_.count(name), 0)
          << "Name " << name << " already exists in the cache";
    }
    pool->pool_.Set(name, arr);
  }

  static ffi::Optional<ffi::Tensor> Get(ffi::String name) {
    TensorCache* pool = Global();
    auto it = pool->pool_.find(name);
    if (it != pool->pool_.end()) {
      return (*it).second;
    } else {
      return std::nullopt;
    }
  }

  static void Remove(ffi::String name) {
    TensorCache* pool = Global();
    pool->pool_.erase(name);
  }

  static void Clear() { Global()->pool_.clear(); }

  /*!
   * \brief Load parameters from path and append them.
   * \param cache_path The cache to path.
   * \param device_type The type of device to be loaded.
   * \param device_id The device id.
   */
  static void Load(const std::string& cache_path, int device_type, int device_id) {
    DLDevice device{static_cast<DLDeviceType>(device_type), device_id};
    TensorCacheMetadata metadata = TensorCacheMetadata::Load(cache_path);
    ffi::Optional<ffi::Tensor> staging_buffer;
    std::string raw_data;
    ffi::Array<ffi::Tensor> params;
    for (const TensorCacheMetadata::FileRecord& shard_rec : metadata.records) {
      try {
        params = shard_rec.Load(device, cache_path, &raw_data, &staging_buffer);
      } catch (const std::runtime_error& e) {
        TVM_FFI_THROW(ValueError) << "Error when loading parameters from " << shard_rec.data_path
                                  << ": " << e.what();
      }
      int num_params = params.size();
      for (int i = 0; i < num_params; ++i) {
        Update(shard_rec.records[i].name, params[i], true);
      }
    }
  }

 private:
  ffi::Map<ffi::String, ffi::Tensor> pool_;
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("vm.builtin.tensor_cache.get", TensorCache::Get)
      .def_packed("vm.builtin.tensor_cache.update",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    TVM_FFI_ICHECK(args.size() == 2 || args.size() == 3);
                    ffi::String name = args[0].cast<ffi::String>();
                    bool is_override = args.size() == 2 ? false : args[2].cast<bool>();

                    ffi::Tensor arr = [&]() {
                      if (auto opt_nd = args[1].as<ffi::Tensor>(); opt_nd && opt_nd->defined()) {
                        return opt_nd.value();
                      } else {
                        // We support converting DLTensors to Tensors as RPC references are always
                        // DLTensors
                        auto tensor = args[1].cast<DLTensor*>();
                        std::vector<int64_t> shape;
                        for (int64_t i = 0; i < tensor->ndim; i++) {
                          shape.push_back(tensor->shape[i]);
                        }
                        ffi::Tensor arr = tvm::runtime::Empty(shape, tensor->dtype, tensor->device);
                        tvm::runtime::CopyFromTo(tensor, arr);
                        DeviceAPI::Get(arr.device())->StreamSync(arr.device(), nullptr);
                        return arr;
                      }
                    }();

                    TensorCache::Update(name, arr, is_override);
                  })
      .def("vm.builtin.tensor_cache.remove", TensorCache::Remove)
      .def("vm.builtin.tensor_cache.clear", TensorCache::Clear)
      .def("vm.builtin.tensor_cache.load", TensorCache::Load);
}

// This param module node can be useful to get param dict in RPC mode
// when the remote already have loaded parameters from file.
class ParamModuleObj : public ffi::ModuleObj {
 public:
  const char* kind() const final { return "param_module"; }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final {
    if (name == "get_params") {
      auto params = params_;
      return ffi::Function([params](ffi::PackedArgs args, ffi::Any* rv) { *rv = params; });
    } else {
      return ffi::Function();
    }
  }

  static ffi::Array<ffi::Tensor> GetParams(const ffi::String& prefix, int num_params) {
    ffi::Array<ffi::Tensor> params;
    for (int i = 0; i < num_params || num_params == -1; ++i) {
      std::string name = prefix + "_" + std::to_string(i);
      auto opt = TensorCache::Get(name);
      if (opt) {
        params.push_back(opt.value());
      } else {
        if (num_params == -1) return params;
        TVM_FFI_THROW(InternalError) << "Cannot find " << name << " in cache";
      }
    }
    return params;
  }

  static ffi::Array<ffi::Tensor> GetParamByName(const ffi::Array<ffi::String>& names) {
    ffi::Array<ffi::Tensor> result;
    result.reserve(names.size());
    for (const ffi::String& name : names) {
      if (ffi::Optional<ffi::Tensor> opt = TensorCache::Get(name)) {
        result.push_back(opt.value());
      } else {
        TVM_FFI_THROW(ValueError) << "Cannot find parameter in cache: " << name;
      }
    }
    return result;
  }

  static ffi::Module Create(const std::string& prefix, int num_params) {
    auto n = ffi::make_object<ParamModuleObj>();
    n->params_ = GetParams(prefix, num_params);
    return ffi::Module(n);
  }

  static ffi::Module CreateByName(const ffi::Array<ffi::String>& names) {
    auto n = ffi::make_object<ParamModuleObj>();
    n->params_ = GetParamByName(names);
    return ffi::Module(n);
  }

 private:
  ffi::Array<ffi::Tensor> params_;
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("vm.builtin.param_module_from_cache", ParamModuleObj::Create)
      .def("vm.builtin.param_module_from_cache_by_name", ParamModuleObj::CreateByName)
      .def("vm.builtin.param_array_from_cache", ParamModuleObj::GetParams)
      .def("vm.builtin.param_array_from_cache_by_name", ParamModuleObj::GetParamByName)
      .def_packed("vm.builtin.param_array_from_cache_by_name_unpacked",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    ffi::Array<ffi::String> names;
                    names.reserve(args.size());
                    for (int i = 0; i < args.size(); ++i) {
                      if (!args[i].try_cast<ffi::String>()) {
                        TVM_FFI_THROW(ValueError) << "Expect string as input, but get "
                                                  << args[i].GetTypeKey() << " at " << i;
                      }
                      names.push_back(args[i].cast<ffi::String>());
                    }
                    *rv = ParamModuleObj::GetParamByName(names);
                  });
}

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
