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
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/extra/json.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/disco/builtin.h>
#include <tvm/runtime/vm/tensor_cache_support.h>
#include <tvm/support/file_utils.h>

#include <functional>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "./utils.h"

namespace tvm {
namespace runtime {

namespace json = tvm::ffi::json;

using vm::TensorCacheMetadata;
using FileRecord = TensorCacheMetadata::FileRecord;
using ParamRecord = TensorCacheMetadata::FileRecord::ParamRecord;

struct ShardInfo {
  struct TensorInfo {
    ffi::Shape shape;
    DLDataType dtype;
  };
  struct ShardFunc {
    std::string name;
    TensorInfo output_info;
    std::vector<int64_t> params;
  };
  std::vector<ShardFunc> funcs;
};

std::unordered_map<std::string, ShardInfo> LoadShardInfoFromStr(const std::string& json_str);

ShardInfo::TensorInfo LoadTensorInfoFromJSON(const json::Array& json_tensor_info) {
  TVM_FFI_CHECK_EQ(json_tensor_info.size(), 2, ValueError) << "Invalid tensor info JSON";
  json::Array shape_json = json_tensor_info[0].cast<json::Array>();
  int ndim = shape_json.size();
  std::vector<int64_t> shape;
  shape.reserve(ndim);
  for (int i = 0; i < ndim; ++i) {
    shape.push_back(shape_json[i].cast<int64_t>());
  }
  std::string dtype = json_tensor_info[1].cast<ffi::String>();
  return ShardInfo::TensorInfo{ffi::Shape(std::move(shape)), ffi::StringToDLDataType(dtype)};
}

ShardInfo::ShardFunc LoadShardFuncFromJSON(const json::Array& json_shard_func) {
  int n = json_shard_func.size();
  ShardInfo::ShardFunc shard_info;
  shard_info.name = json_shard_func[0].cast<ffi::String>();
  shard_info.output_info = LoadTensorInfoFromJSON(json_shard_func[1].cast<json::Array>());
  shard_info.params.reserve(n - 2);
  for (int i = 2; i < n; ++i) {
    shard_info.params.push_back(json_shard_func[i].cast<int64_t>());
  }
  return shard_info;
}

std::unordered_map<std::string, ShardInfo> LoadShardInfoFromStr(const std::string& json_str) {
  json::Value json_info = json::Parse(json_str);
  json::Object json_obj = json_info.cast<json::Object>();
  std::unordered_map<std::string, ShardInfo> result;
  for (const auto& [k, v] : json_obj) {
    std::string name = k.cast<ffi::String>();
    json::Array json_shard_funcs = v.cast<json::Array>();
    ShardInfo info;
    std::vector<ShardInfo::ShardFunc>& shard_funcs = info.funcs;
    shard_funcs.reserve(json_shard_funcs.size());
    for (const ffi::Any& json_shard_func : json_shard_funcs) {
      shard_funcs.push_back(LoadShardFuncFromJSON(json_shard_func.cast<json::Array>()));
    }
    result[name] = info;
  }
  return result;
}

/*! \brief An object that helps to load parameters in shards. */
class ShardLoaderObj : public ffi::Object {
 public:
  /*! \brief Create a shard loader. */
  static ffi::ObjectRef Create(const std::string& path_to_metadata, const std::string& metadata,
                               std::string shard_info, ffi::Optional<ffi::Module> mod);
  /*! \brief Load the i-th parameter */
  ffi::Tensor Load(int weight_index) const;

  ffi::Tensor LoadParamOnWorker0(int weight_index) const;

  /*! \brief Load all the parameters */
  ffi::Array<ffi::Tensor> LoadAll() const;

  ffi::Tensor ApplyShardFunc(const ShardInfo::ShardFunc& shard_func,
                             const ffi::Tensor& param) const;

  /*! \brief Load all the pre-sharded parameters */
  ffi::Array<ffi::Tensor> LoadAllPresharded() const;

  /*! \brief Load the i-th parameter from presharded binaries */
  ffi::Tensor LoadPresharded(int weight_index) const;

  /*! \brief Slice the given tensor at a specific dimension */
  ffi::Tensor Shard(ffi::Tensor source, int dim, int num_slices) const;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("runtime.disco.ShardLoader", ShardLoaderObj, ffi::Object);

 public:
  /*! \brief Information of how each weight is stored and sharded */
  struct ParamInfo {
    const FileRecord* file;
    const ParamRecord* param;
    ShardInfo shard_info;
  };
  /*! \brief The ffi::Functions being used during sharding */
  std::unordered_map<std::string, ffi::Function> shard_funcs_;
  /*! \brief The metadata loaded from `tensor-cache.json` */
  TensorCacheMetadata metadata_;
  /*! \brief Sharding information for each weight */
  std::vector<ParamInfo> param_info_;
  /*! \brief Maps the name of a shard to its index */
  std::unordered_map<std::string, int> param_name_to_index_;
  /*! \brief The current file opened to load weights in it */
  mutable const FileRecord* current_file_;
  /*! \brief The context of the current file to be loaded from */
  mutable std::string current_file_stream_;

 private:
  /*! \brief Load the i-th parameter without post-processing
   *
   * This function should not be called externally, as it does not
   * check for post-processing that may be required.  Instead, the
   * public function `Load` or `LoadPresharded` should be called.
   *
   * \param weight_index The index of Tensor tensor to load
   *
   * \returns The full tensor at the specified index
   */
  ffi::Tensor LoadDirect(int weight_index) const;
};

ffi::ObjectRef ShardLoaderObj::Create(const std::string& path_to_metadata,
                                      const std::string& metadata, std::string shard_info,
                                      ffi::Optional<ffi::Module> mod) {
  if (shard_info.empty() && mod.has_value()) {
    if (auto get_shard_info = (*mod)->GetFunction("get_shard_info")) {
      shard_info = (*get_shard_info)().cast<ffi::String>();
    }
  }
  ffi::ObjectPtr<ShardLoaderObj> n = ffi::make_object<ShardLoaderObj>();
  n->metadata_ = TensorCacheMetadata::LoadFromStr(metadata, path_to_metadata);
  n->current_file_ = nullptr;
  n->param_info_.clear();
  std::unordered_map<std::string, ShardInfo> shards = LoadShardInfoFromStr(shard_info);
  for (const FileRecord& file_record : n->metadata_.records) {
    for (const ParamRecord& param_record : file_record.records) {
      const std::string& name = param_record.name;
      int index = n->param_info_.size();
      n->param_name_to_index_[name] = index;
      ShardInfo& shard_info = shards[name];
      for (const ShardInfo::ShardFunc& shard_func : shard_info.funcs) {
        const std::string& name = shard_func.name;
        if (ffi::Optional<ffi::Function> f =
                mod.has_value() ? (*mod)->GetFunction(name, true) : std::nullopt) {
          n->shard_funcs_[name] = *f;
        } else if (const auto f = tvm::ffi::Function::GetGlobal(name)) {
          n->shard_funcs_[name] = *f;
        } else {
          TVM_FFI_THROW(ValueError) << "Undefined function: " << name;
        }
      }
      n->param_info_.emplace_back(ParamInfo{&file_record, &param_record, shard_info});
    }
  }
  return ffi::ObjectRef(std::move(n));
}

ffi::Tensor ShardLoaderObj::ApplyShardFunc(const ShardInfo::ShardFunc& shard_func,
                                           const ffi::Tensor& param) const {
  TVM_FFI_ICHECK(param.defined());
  Device device = param.device();
  ffi::Tensor o =
      tvm::runtime::Empty(shard_func.output_info.shape, shard_func.output_info.dtype, device);
  ffi::Function f = this->shard_funcs_.at(shard_func.name);
  int n = static_cast<int>(shard_func.params.size());
  std::vector<ffi::AnyView> packed_args(n + 2);
  const DLTensor* w_in = param.GetDLTensorPtr();
  const DLTensor* w_out = o.GetDLTensorPtr();
  packed_args[0] = const_cast<DLTensor*>(w_in);
  for (int i = 0; i < n; ++i) {
    packed_args[i + 1] = shard_func.params[i];
  }
  packed_args[n + 1] = const_cast<DLTensor*>(w_out);
  ffi::Any rv;
  f.CallPacked(ffi::PackedArgs(packed_args.data(), packed_args.size()), &rv);
  return o;
}

std::string GetSiblingPath(const std::string& path, const std::string& filename) {
  size_t found = path.find_last_of("/\\");
  if (found != std::string::npos) {
    return path.substr(0, found + 1) + filename;
  }
  TVM_FFI_THROW(ValueError) << "Cannot find the parent directory: " << path;
  return "";
}

ffi::Tensor ShardLoaderObj::LoadParamOnWorker0(int weight_index) const {
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  int worker_id = worker->worker_id;
  Device device = worker->default_device;
  int param_index = param_name_to_index_.at("param_" + std::to_string(weight_index));
  const ParamInfo& param_info = param_info_.at(param_index);
  const ParamRecord* param = param_info.param;
  const FileRecord* file = param_info.file;

  auto load = [this, param, device, file]() {
    if (file != current_file_) {
      current_file_ = file;
      std::string file_name = GetSiblingPath(this->metadata_.path, file->data_path);
      this->current_file_stream_ = support::ReadBytesFromFile(file_name);
    }
    return param->Load(device, &this->current_file_stream_);
  };

  if (worker_id == 0) {
    ffi::Tensor w = load();
    return w;
  } else {
    ffi::Tensor w = tvm::runtime::Empty(param->shape, param->dtype, device);
    return w;
  }
}

std::tuple<int, int> ParseParamShardingInfo(const ParamRecord* param) {
  // Given a name "param_shard-X-of-Y", return the integer values
  // rank=(X-1) and world_size=Y.

  std::string name = param->name;
  size_t pos1 = name.rfind("-of-");
  TVM_FFI_ICHECK(pos1 != std::string::npos)
      << "Attempt to read num_shards from unexpected param name: " << name;
  size_t pos2 = name.rfind("_shard-", pos1 - 1);
  TVM_FFI_ICHECK(pos2 != std::string::npos)
      << "Attempt to read sharded worker_id from unexpected param name: " << name;

  int num_shards = std::stoi(name.substr(pos1 + 4));
  int worker_id = std::stoi(name.substr(pos2 + 7, pos1 - pos2 - 7)) - 1;

  TVM_FFI_ICHECK_GT(num_shards, 1);
  TVM_FFI_ICHECK_GE(worker_id, 0);
  TVM_FFI_ICHECK_LT(worker_id, num_shards);

  return {num_shards, worker_id};
}

ffi::Tensor ShardLoaderObj::LoadDirect(int weight_index) const {
  const ParamInfo& param_info = param_info_.at(weight_index);
  const ParamRecord* param = param_info.param;
  const FileRecord* file = param_info.file;

  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  Device device = worker->default_device;

  if (file != current_file_) {
    current_file_ = file;
    std::string file_name = GetSiblingPath(this->metadata_.path, file->data_path);
    this->current_file_stream_ = support::ReadBytesFromFile(file_name);
  }
  return param->Load(device, &this->current_file_stream_);
}

ffi::Tensor ShardLoaderObj::Load(int weight_index) const {
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  int worker_id = worker->worker_id;
  int num_shards = worker->num_workers;
  Device device = worker->default_device;
  const ParamInfo& param_info = param_info_.at(weight_index);
  const ParamRecord* param = param_info.param;

  bool needs_sharding = !param_info.shard_info.funcs.empty();
  if (needs_sharding) {
    ffi::Shape shape = param_info.shard_info.funcs.back().output_info.shape;
    DLDataType dtype = param_info.shard_info.funcs.back().output_info.dtype;
    TVM_FFI_CHECK(shape.size() >= 1 && shape[0] == num_shards, ValueError)
        << "The first dimension of the "
        << "output shape must be equal to the "
        << "number of shards, but got: " << shape << " and num_shards = " << num_shards;
    ffi::Tensor recv =
        tvm::runtime::Empty(ffi::Shape(shape.begin() + 1, shape.end()), dtype, device);
    if (worker_id == 0) {
      ffi::Tensor w = LoadDirect(weight_index);
      for (const ShardInfo::ShardFunc& shard_func : param_info.shard_info.funcs) {
        w = this->ApplyShardFunc(shard_func, w);
      }
      ScatterFromWorker0(w, /*in_group=*/false, recv);
    } else {
      ScatterFromWorker0(std::nullopt, /*in_group=*/false, recv);
    }
    return recv;
  } else {
    if (worker_id == 0) {
      ffi::Tensor w = LoadDirect(weight_index);
      BroadcastFromWorker0(w, /*in_group=*/false, w);
      return w;
    } else {
      ffi::Tensor w = tvm::runtime::Empty(param->shape, param->dtype, device);
      BroadcastFromWorker0(w, /*in_group=*/false, w);
      return w;
    }
  }
}

ffi::Array<ffi::Tensor> ShardLoaderObj::LoadAll() const {
  int n = static_cast<int>(param_info_.size());
  ffi::Array<ffi::Tensor> shards;
  shards.reserve(n);
  for (int i = 0; i < n; ++i) {
    std::string param_name = "param_" + std::to_string(i);
    TVM_FFI_ICHECK(this->param_name_to_index_.count(param_name));
    int shard_id = this->param_name_to_index_.at(param_name);
    shards.push_back(this->Load(shard_id));
  }
  return shards;
}

ffi::Tensor ShardLoaderObj::LoadPresharded(int weight_index) const {
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  int worker_id = worker->worker_id;
  int num_shards = worker->num_workers;
  size_t num_weights = param_info_.size() / num_shards;
  size_t index = worker_id * num_weights + weight_index;
  TVM_FFI_ICHECK(index < param_info_.size())
      << "Loading param " << weight_index << " for shard " << worker_id << " at position " << index
      << " is out of bounds for the provided ndarray chace.";

  const auto& shard_info = param_info_[index];
  const ParamRecord* param = shard_info.param;
  const FileRecord* file = shard_info.file;

  auto [p_num_shards, p_worker_id] = ParseParamShardingInfo(param);
  TVM_FFI_ICHECK_EQ(num_shards, p_num_shards)
      << "Runtime number of shards (" << num_shards
      << ") does not match number of compiled shards (" << p_num_shards << "): " << param->name
      << " loaded from " << file->data_path;
  TVM_FFI_ICHECK_EQ(worker_id, p_worker_id)
      << "Runtime worker_id (" << worker_id << ") does not match worker_id of compiled shard ("
      << p_worker_id << "): " << param->name << " loaded from " << file->data_path;

  return LoadDirect(index);
}

ffi::Array<ffi::Tensor> ShardLoaderObj::LoadAllPresharded() const {
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  size_t worker_id = static_cast<size_t>(worker->worker_id);
  size_t num_workers = static_cast<size_t>(worker->num_workers);
  size_t num_params = param_info_.size() / num_workers;

  ffi::Array<ffi::Tensor> params;
  params.reserve(num_params);
  for (size_t i_param = 0; i_param < num_params; ++i_param) {
    std::string param_name = static_cast<const std::stringstream&>(
                                 std::stringstream() << "param_" << i_param << "_shard-"
                                                     << (worker_id + 1) << "-of-" << num_workers)
                                 .str();

    auto it = param_name_to_index_.find(param_name);
    TVM_FFI_ICHECK(it != param_name_to_index_.end())
        << "Parameter " << param_name << " was not found in the parameter set";
    int param_id = this->param_name_to_index_.at(param_name);
    params.push_back(this->LoadDirect(param_id));
  }
  return params;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.disco.ShardLoader", ShardLoaderObj::Create)
      .def("runtime.disco.ShardLoaderLoad",
           [](ffi::ObjectRef loader_obj, ffi::Shape weight_index) {
             const auto* loader = loader_obj.as<ShardLoaderObj>();
             TVM_FFI_CHECK(loader != nullptr, TypeError)
                 << "Expected ShardLoaderObj, but gets: " << loader_obj->GetTypeKey();
             return loader->Load(IntegerFromShape(weight_index));
           })
      .def("runtime.disco.ShardLoaderLoadPresharded",
           [](ffi::ObjectRef loader_obj, ffi::Shape weight_index) {
             const auto* loader = loader_obj.as<ShardLoaderObj>();
             TVM_FFI_CHECK(loader != nullptr, TypeError)
                 << "Expected ShardLoaderObj, but gets: " << loader_obj->GetTypeKey();
             return loader->LoadPresharded(IntegerFromShape(weight_index));
           })
      .def("runtime.disco.ShardLoaderLoadAll",
           [](ffi::ObjectRef loader_obj) {
             const auto* loader = loader_obj.as<ShardLoaderObj>();
             TVM_FFI_CHECK(loader != nullptr, TypeError)
                 << "Expected ShardLoaderObj, but gets: " << loader_obj->GetTypeKey();
             return loader->LoadAll();
           })
      .def("runtime.disco.ShardLoaderLoadAllPresharded",
           [](ffi::ObjectRef loader_obj) {
             const auto* loader = loader_obj.as<ShardLoaderObj>();
             TVM_FFI_CHECK(loader != nullptr, TypeError)
                 << "Expected ShardLoaderObj, but gets: " << loader_obj->GetTypeKey();
             return loader->LoadAllPresharded();
           })
      .def("runtime.disco.ShardLoaderLoadParamOnWorker0",
           [](ffi::ObjectRef loader_obj, int param_index) {
             const auto* loader = loader_obj.as<ShardLoaderObj>();
             TVM_FFI_CHECK(loader != nullptr, TypeError)
                 << "Expected ShardLoaderObj, but gets: " << loader_obj->GetTypeKey();
             return loader->LoadParamOnWorker0(param_index);
           });
}

}  // namespace runtime
}  // namespace tvm
