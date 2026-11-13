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
 * \file src/runtime/contrib/tensorrt/tensorrt_runtime.cc
 * \brief JSON runtime implementation for TensorRT.
 */

#include <tvm/ffi/extra/json.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/tensor.h>
#include <tvm/support/env.h>
#include <tvm/support/file_utils.h>

#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../json/json_node.h"
#include "../json/json_runtime.h"

#ifdef TVM_GRAPH_EXECUTOR_TENSORRT
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/base.h>
#include <tvm/ffi/extra/cuda/device_guard.h>

#include "NvInfer.h"
#include "tensorrt_builder.h"
#include "tensorrt_calibrator.h"
#include "tensorrt_utils.h"
#endif

namespace tvm {
namespace runtime {
namespace contrib {

struct PairHash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2>& pair) const {
    return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
  }
};

using namespace tvm::runtime::json;

class TensorRTRuntime : public JSONRuntimeBase {
 public:
  /*!
   * \brief The TensorRT runtime module. Deserialize the provided functions
   * on creation and store in the layer cache.
   *
   * \param symbol_name The name of the function.
   * \param graph_json serialized JSON representation of a sub-graph.
   * \param const_names The names of each constant in the sub-graph.
   */
  explicit TensorRTRuntime(const std::string& symbol_name, const std::string& graph_json,
                           const ffi::Array<ffi::String>& const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names),
        use_implicit_batch_(true),
        max_workspace_size_(size_t(1) << 30),
        max_batch_size_(-1),
        multi_engine_mode_(false),
        use_fp16_(false) {
    const bool use_int8 = support::GetEnv("TVM_TENSORRT_USE_INT8", false);
    multi_engine_mode_ = support::GetEnv("TVM_TENSORRT_MULTI_ENGINE", false);
    num_calibration_batches_remaining_ = support::GetEnv("TENSORRT_NUM_CALI_INT8", 0);
    if (use_int8) {
      TVM_FFI_ICHECK(num_calibration_batches_remaining_ != 0)
          << "When using INT8 mode, "
          << "environment variable TENSORRT_NUM_CALI_INT8"
          << "must also be set to specify the number of "
          << "calibration times";
      LOG(INFO) << "settiing up " << num_calibration_batches_remaining_
                << " sample data to calibrate data ... ";
      TVM_FFI_ICHECK(multi_engine_mode_ == false) << "When using int8 mode, "
                                                  << "multi-engine is not allowed";
    }
  }

  /*!
   * \brief The type key of the module.
   *
   * \return module type key.
   */
  const char* kind() const final { return "tensorrt"; }

  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const final {
    return ffi::Module::kBinarySerializable | ffi::Module::kRunnable;
  }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) override {
    if (name == "build_engine") {
      ffi::ObjectPtr<ffi::Object> self = ffi::GetObjectPtr<ffi::Object>(this);
      return ffi::Function([self, this](ffi::PackedArgs args, ffi::Any* rv) {
        TVM_FFI_ICHECK_EQ(args.size(), 1U);
        std::lock_guard<std::mutex> guard(initialize_mutex_);
        TVM_FFI_ICHECK(!initialized_) << "Build the TensorRT engine before initializing the module";
        BuildEngineAtCompileTime(args[0].cast<ffi::Array<ffi::Tensor>>());
        *rv = 0;
      });
    }
    return JSONRuntimeBase::GetFunction(name);
  }

  ffi::Bytes SaveToBytes() const override {
    auto graph_bytes = JSONRuntimeBase::SaveToBytes();
    if (serialized_engine_.empty()) return graph_bytes;

    // Keep the JSON runtime prefix unchanged. Only opt-in, prebuilt modules have this trailer.
    std::string result(graph_bytes.data(), graph_bytes.size());
    support::BytesOutStream stream(&result);
    stream.Write(kEngineMagic);
    stream.Write(kEngineVersion);
    stream.Write(serialized_engine_);
    stream.Write(serialized_inputs_);
    stream.Write(serialized_outputs_);
    stream.Write(serialized_batch_size_);
    return ffi::Bytes(std::move(result));
  }

  static ffi::Module LoadFromBytes(const ffi::Bytes& bytes) {
    support::BytesInStream stream(bytes);
    std::string symbol, graph_json;
    std::vector<std::string> consts;
    TVM_FFI_ICHECK(stream.Read(&symbol)) << "Loading symbol name failed";
    TVM_FFI_ICHECK(stream.Read(&graph_json)) << "Loading graph json failed";
    TVM_FFI_ICHECK(stream.Read(&consts)) << "Loading the const name list failed";
    ffi::Array<ffi::String> const_names;
    for (const auto& name : consts) const_names.push_back(name);
    auto n = ffi::make_object<TensorRTRuntime>(symbol, graph_json, const_names);

    uint64_t magic;
    size_t nread = stream.Read(&magic, sizeof(magic));
    if (nread == 0) return ffi::Module(n);  // Legacy JSON-only module.
    TVM_FFI_ICHECK_EQ(nread, sizeof(magic)) << "Truncated TensorRT engine trailer";
    TVM_FFI_ICHECK_EQ(magic, kEngineMagic) << "Invalid TensorRT engine trailer";
    uint32_t version;
    TVM_FFI_ICHECK(stream.Read(&version)) << "Loading TensorRT engine version failed";
    TVM_FFI_ICHECK_EQ(version, kEngineVersion) << "Unsupported TensorRT engine version";
    TVM_FFI_ICHECK(stream.Read(&n->serialized_engine_)) << "Loading TensorRT engine failed";
    TVM_FFI_ICHECK(!n->serialized_engine_.empty()) << "The embedded TensorRT engine is empty";
    TVM_FFI_ICHECK(stream.Read(&n->serialized_inputs_)) << "Loading TensorRT input names failed";
    TVM_FFI_ICHECK(stream.Read(&n->serialized_outputs_)) << "Loading TensorRT output names failed";
    TVM_FFI_ICHECK(stream.Read(&n->serialized_batch_size_)) << "Loading TensorRT batch size failed";
    TVM_FFI_ICHECK_GT(n->serialized_batch_size_, 0) << "Invalid TensorRT batch size";
    TVM_FFI_ICHECK_EQ(n->serialized_inputs_.size(), n->input_var_eid_.size());
    TVM_FFI_ICHECK_EQ(n->serialized_outputs_.size(), n->outputs_.size());
    char trailing;
    TVM_FFI_ICHECK_EQ(stream.Read(&trailing, sizeof(trailing)), 0U)
        << "Unexpected data after the TensorRT engine";
    // Defer GPU-dependent deserialization until the first call supplies the execution device.
    return ffi::Module(n);
  }

  /*!
   * \brief Initialize runtime. Create TensorRT layer from JSON
   * representation.
   *
   * \param consts The constant params from compiled model.
   */
  void Init(const ffi::Array<ffi::Tensor>& consts) override {
    TVM_FFI_ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";
    if (!serialized_engine_.empty()) {
      TVM_FFI_ICHECK(!support::GetEnv("TVM_TENSORRT_USE_INT8", false) &&
                     support::GetEnv("TENSORRT_NUM_CALI_INT8", 0) == 0 &&
                     num_calibration_batches_remaining_ == 0)
          << "A prebuilt TensorRT engine cannot perform INT8 calibration";
    }
    LoadGlobalAttributes();
    SetupConstants(consts);
    if (serialized_engine_.empty()) GetCachedEnginesFromDisk();
  }

  void LoadGlobalAttributes() {
    // These settings are global to the entire subgraph. Codegen will add them as attributes to all
    // op nodes. Read from first one.
    for (size_t i = 0; i < nodes_.size(); ++i) {
      if (nodes_[i].HasAttr("use_implicit_batch") && nodes_[i].HasAttr("max_workspace_size")) {
        use_implicit_batch_ = static_cast<int>(nodes_[i].GetAttr<int64_t>("use_implicit_batch"));
        if (use_implicit_batch_) {
          LOG(WARNING) << "use_implicit_batch=True is ignored: TensorRT 10 removed implicit-batch "
                          "mode, so the engine is always built and run in explicit-batch mode.";
        }
        // Allow max_workspace_size to be overridden at runtime.
        size_t runtime_max_workspace_size =
            support::GetEnv("TVM_TENSORRT_MAX_WORKSPACE_SIZE", size_t(0));
        if (runtime_max_workspace_size != 0) {
          max_workspace_size_ = runtime_max_workspace_size;
        } else {
          max_workspace_size_ =
              static_cast<size_t>(nodes_[i].GetAttr<int64_t>("max_workspace_size"));
        }
      }
      if (nodes_[i].HasAttr("use_fp16")) {
        use_fp16_ = static_cast<int>(nodes_[i].GetAttr<int64_t>("use_fp16"));
      }
    }
  }

#ifdef TVM_GRAPH_EXECUTOR_TENSORRT
  /*! \brief Destroy engines and contexts on the device that owns them. */
  void DestroyEngines() {
    if (embedded_device_id_ >= 0) {
      ffi::CUDADeviceGuard guard(embedded_device_id_);
      DestroyEnginesOnCurrentDevice();
      device_buffers_.reset();
      embedded_device_id_ = -1;
    } else {
      DestroyEnginesOnCurrentDevice();
    }
  }

  void DestroyEnginesOnCurrentDevice() {
    for (auto& it : trt_engine_cache_) {
      // TensorRT 10 removed obj->destroy(); release with delete. The deserialization runtime must
      // outlive the engine it produced, so delete the context, then the engine, then the runtime.
      VLOG(1) << "Destroying TensorRT context for function '" << it.first.first << "' (batch size "
              << it.first.second << ")";
      delete it.second.context;
      VLOG(1) << "Destroying TensorRT engine for function '" << it.first.first << "' (batch size "
              << it.first.second << ")";
      delete it.second.engine;
      delete it.second.runtime;
    }
    trt_engine_cache_.clear();
  }

  ~TensorRTRuntime() {
    // FFI can retain imported modules until process shutdown, after CUDA has already unloaded.
    // TensorRT destructors and CUDADeviceGuard cannot run then; the process reclaims the resources.
    int current_device;
    if ((!trt_engine_cache_.empty() || device_buffers_ || calibrator_) &&
        cudaGetDevice(&current_device) == cudaErrorCudartUnloading) {
      // CPU-input staging tensors and the calibrator also own CUDA allocations. Release their
      // containers so member destruction cannot enter CUDA after this confirmed shutdown state.
      (void)device_buffers_.release();
      (void)calibrator_.release();
      return;
    }
    VLOG(1) << "Destroying TensorRT runtime";
    DestroyEngines();
    VLOG(1) << "Destroyed TensorRT runtime";
  }

  /*! \brief Build from static input metadata and weights, without running inference. */
  void BuildEngineAtCompileTime(const ffi::Array<ffi::Tensor>& consts) {
    for (const auto& tensor : consts) {
      TVM_FFI_ICHECK(tensor.defined());
    }
    TVM_FFI_ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";
    TVM_FFI_ICHECK(!support::GetEnv("TVM_TENSORRT_USE_INT8", false) &&
                   support::GetEnv("TENSORRT_NUM_CALI_INT8", 0) == 0 &&
                   num_calibration_batches_remaining_ == 0)
        << "TensorRT build_at_compile_time does not support INT8 calibration";
    TVM_FFI_ICHECK(!input_var_eid_.empty())
        << "TensorRT build_at_compile_time requires at least one tensor input";
    TVM_FFI_ICHECK(serialized_engine_.empty()) << "The TensorRT engine has already been built";

    std::vector<std::vector<int64_t>> shapes(NumEntries());
    std::vector<DLTensor> inputs(NumEntries());
    for (auto nid : input_nodes_) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() != "input") continue;
      auto node_shapes = node.GetOpShape();
      auto dtypes = node.GetOpDataType();
      for (size_t j = 0; j < node_shapes.size(); ++j) {
        TVM_FFI_ICHECK(dtypes[j].code == kDLFloat && dtypes[j].lanes == 1 &&
                       (dtypes[j].bits == 16 || dtypes[j].bits == 32))
            << "TensorRT build_at_compile_time requires float16 or float32 inputs";
        uint32_t eid = EntryID(nid, j);
        for (int64_t dim : node_shapes[j]) {
          TVM_FFI_ICHECK_GT(dim, 0)
              << "TensorRT build_at_compile_time requires static positive input dimensions";
          TVM_FFI_ICHECK_LE(dim, std::numeric_limits<int32_t>::max())
              << "TensorRT input dimensions must fit in int32";
          shapes[eid].push_back(dim);
        }
        inputs[eid].device = DLDevice{kDLCPU, 0};
        inputs[eid].ndim = static_cast<int32_t>(shapes[eid].size());
        inputs[eid].dtype = dtypes[j];
        inputs[eid].shape = shapes[eid].data();
      }
    }

    for (const auto& constant : consts) {
      TVM_FFI_ICHECK_EQ(constant.device().device_type, kDLCPU)
          << "TensorRT build_at_compile_time requires constants on the CPU";
      const auto dtype = constant.dtype();
      TVM_FFI_ICHECK(dtype.code == kDLFloat && dtype.lanes == 1 &&
                     (dtype.bits == 16 || dtype.bits == 32))
          << "TensorRT build_at_compile_time requires float16 or float32 constants";
    }
    LoadGlobalAttributes();
    auto original_entries = data_entry_;
    try {
      SetupConstants(consts);
      for (auto eid : input_var_eid_) data_entry_[eid] = &inputs[eid];
      int batch_size = GetBatchSize();
      BuildEngineFromJson(batch_size);
      const auto& built = trt_engine_cache_.at(std::make_pair(symbol_name_, batch_size));
      std::unique_ptr<nvinfer1::IHostMemory> plan(built.engine->serialize());
      TVM_FFI_ICHECK(plan != nullptr) << "Serializing the TensorRT engine failed";
      TVM_FFI_ICHECK_GT(plan->size(), 0U) << "The serialized TensorRT engine is empty";
      serialized_engine_.assign(static_cast<const char*>(plan->data()), plan->size());
      serialized_inputs_ = built.inputs;
      serialized_outputs_ = built.outputs;
      serialized_batch_size_ = batch_size;
    } catch (...) {
      data_entry_ = std::move(original_entries);
      DestroyEngines();
      serialized_engine_.clear();
      serialized_inputs_.clear();
      serialized_outputs_.clear();
      serialized_batch_size_ = 0;
      throw;
    }
    // Neither input metadata nor the supplied constants are owned by the runtime at this point.
    data_entry_ = std::move(original_entries);
    DestroyEngines();
  }

  void LoadEmbeddedEngine(int device_id) {
    TVM_FFI_ICHECK(!support::GetEnv("TVM_TENSORRT_USE_INT8", false) &&
                   support::GetEnv("TENSORRT_NUM_CALI_INT8", 0) == 0 &&
                   num_calibration_batches_remaining_ == 0)
        << "A prebuilt TensorRT engine cannot perform INT8 calibration";
    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(GetTensorRTLogger()));
    TVM_FFI_ICHECK(runtime != nullptr) << "Creating the TensorRT deserialization runtime failed";
    std::unique_ptr<nvinfer1::ICudaEngine> engine(
        runtime->deserializeCudaEngine(serialized_engine_.data(), serialized_engine_.size()));
    TVM_FFI_ICHECK(engine != nullptr)
        << "Failed to deserialize the embedded TensorRT engine. Recompile the model for this "
           "GPU, platform, and TensorRT version; compile-time engines are not rebuilt at runtime.";
    std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
    TVM_FFI_ICHECK(context != nullptr) << "Creating the embedded TensorRT execution context failed";
    TensorRTEngineAndContext loaded;
    loaded.inputs = serialized_inputs_;
    loaded.outputs = serialized_outputs_;
    auto key = std::make_pair(symbol_name_, serialized_batch_size_);
    auto result = trt_engine_cache_.emplace(key, std::move(loaded));
    TVM_FFI_ICHECK(result.second) << "The embedded TensorRT engine has already been initialized";
    result.first->second.runtime = runtime.release();
    result.first->second.engine = engine.release();
    result.first->second.context = context.release();
    max_batch_size_ = serialized_batch_size_;
    embedded_device_id_ = device_id;
  }

  /*! \brief Run inference using a prebuilt or lazily built engine. */
  void Run() override {
    if (serialized_engine_.empty()) {
      RunWithEngine();
      return;
    }
    // Const-loader initialization has no execution device. The first invocation does, so load the
    // plan here without invoking the TensorRT builder and restore the caller's current device.
    const DLDevice& dev = data_entry_[input_var_eid_[0]]->device;
    const int device_id = dev.device_type == kDLCUDA ? dev.device_id : 0;
    ffi::CUDADeviceGuard guard(device_id);
    if (embedded_device_id_ != device_id) {
      DestroyEngines();
      LoadEmbeddedEngine(device_id);
    }
    RunWithEngine();
  }

  void RunWithEngine() {
    auto& engine_and_context = GetOrBuildEngine();
    int batch_size = GetBatchSize();
    if (batch_size == 0) return;
    auto context = engine_and_context.context;

    // TensorRT 10 uses named-tensor I/O (setInputShape/setTensorAddress/enqueueV3, no binding
    // indices). Track input device pointers and per-sample element counts for the INT8 calibrator.
    std::vector<void*> input_bindings;
    std::vector<size_t> input_binding_sizes;

    // Setup input bindings.
    for (size_t i = 0; i < input_nodes_.size(); ++i) {
      auto nid = input_nodes_[i];
      if (nodes_[nid].GetOpType() == "input") {
        for (size_t j = 0; j < nodes_[nid].GetOpShape().size(); ++j) {
          uint32_t eid = EntryID(nid, j);
          const std::string name = nodes_[nid].GetOpName() + "_" + std::to_string(j);
          std::vector<int64_t> shape(data_entry_[eid]->shape,
                                     data_entry_[eid]->shape + data_entry_[eid]->ndim);
          auto dims = VectorToTrtDims(shape);
          TVM_FFI_ICHECK(context->setInputShape(name.c_str(), dims));

          void* device_ptr = nullptr;
          if (data_entry_[eid]->device.device_type == kDLCUDA) {
            device_ptr = data_entry_[eid]->data;
          } else {
            auto device_buffer = GetOrAllocateDeviceBuffer(name, eid);
            tvm::runtime::CopyFromTo(data_entry_[eid], device_buffer);
            device_ptr = device_buffer.data_ptr();
          }
          TVM_FFI_ICHECK(context->setTensorAddress(name.c_str(), device_ptr));

          // Per-sample element count (exclude the batch dimension d[0]); the INT8 calibrator
          // multiplies by the batch size itself when copying calibration data, so including the
          // batch dim here would over-read the device buffer by a factor of batch_size.
          int num_elements = 1;
          for (int k = 1; k < dims.nbDims; ++k) num_elements *= dims.d[k];
          input_bindings.push_back(device_ptr);
          input_binding_sizes.push_back(static_cast<size_t>(num_elements));
        }
      }
    }

    // add batch data to calibrator
    if (num_calibration_batches_remaining_ > 0) {
      if (calibrator_ != nullptr) {
        LOG(INFO) << "Starting adding last " << num_calibration_batches_remaining_
                  << "-th batch data to the calibrator";
        calibrator_->AddBatchData(input_bindings, input_binding_sizes);
        num_calibration_batches_remaining_--;
      }
      return;
    }

    // Setup output bindings.
    for (size_t i = 0; i < outputs_.size(); ++i) {
      uint32_t eid = EntryID(outputs_[i]);
      const std::string& name = engine_and_context.outputs[i];
      void* device_ptr = nullptr;
      if (data_entry_[eid]->device.device_type == kDLCUDA) {
        device_ptr = data_entry_[eid]->data;
      } else {
        auto device_buffer = GetOrAllocateDeviceBuffer(name, eid);
        device_ptr = device_buffer.data_ptr();
      }
      TVM_FFI_ICHECK(context->setTensorAddress(name.c_str(), device_ptr));
    }

    // Run on TVM's current CUDA stream so the engine is ordered after the inputs produced upstream
    // (and to avoid TensorRT's default-stream synchronization warning). enqueueV3 is async-only in
    // TRT10, so synchronize afterwards to preserve Run()'s blocking semantics.
    const DLDevice& dev = data_entry_[input_var_eid_[0]]->device;
    const int device_id = dev.device_type == kDLCUDA ? dev.device_id : 0;
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, device_id));
    TVM_FFI_ICHECK(context->enqueueV3(stream)) << "Running TensorRT failed.";
    TVM_FFI_CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // Copy outputs from GPU buffers if needed.
    for (size_t i = 0; i < outputs_.size(); ++i) {
      uint32_t eid = EntryID(outputs_[i]);
      const std::string& name = engine_and_context.outputs[i];
      if (data_entry_[eid]->device.device_type != kDLCUDA) {
        auto device_buffer = GetOrAllocateDeviceBuffer(name, eid);
        tvm::runtime::CopyFromTo(device_buffer, const_cast<DLTensor*>(data_entry_[eid]));
      }
    }
  }

 private:
  /*! \brief Get batch size for engine from the runtime input shapes. */
  int GetBatchSize() {
    return data_entry_[input_var_eid_[0]]->ndim == 0 ? 1 : data_entry_[input_var_eid_[0]]->shape[0];
  }

  /*! \brief Find an engine in the cache which we can reuse depending on the mode. If no compatible
   * engine exists, return false to indicate that a new one should be built. */
  bool FindCompatibleEngine(int batch_size, int* compatible_engine_batch_size) {
    if (multi_engine_mode_) {
      // Exact match is required for multi engine mode.
      if (trt_engine_cache_.count(std::make_pair(symbol_name_, batch_size))) {
        *compatible_engine_batch_size = batch_size;
        return true;
      }
      return false;
    }
    // Single-engine mode: TensorRT 10 engines are explicit-batch and their optimization profile
    // pins the built batch size, so a cached engine can only serve that exact batch. Require an
    // exact match (otherwise a smaller batch would be rejected by setInputShape) and rebuild on any
    // change. This replaces the implicit-batch "any batch <= max" reuse that TRT10 removed.
    if (batch_size == max_batch_size_) {
      *compatible_engine_batch_size = max_batch_size_;
      return true;
    }
    return false;
  }

  /*!
   * \brief Build TensorRT engine from JSON representation and cache it. If compatible engine is
   * already built, do nothing.
   */
  TensorRTEngineAndContext& GetOrBuildEngine() {
    int batch_size = GetBatchSize();
    int compatible_engine_batch_size = -1;
    bool find_engine_flag = FindCompatibleEngine(batch_size, &compatible_engine_batch_size);
    if (!serialized_engine_.empty()) {
      TVM_FFI_ICHECK_EQ(batch_size, serialized_batch_size_)
          << "The input batch size does not match the compile-time TensorRT engine";
      TVM_FFI_ICHECK(find_engine_flag) << "The embedded TensorRT engine has not been initialized";
      return trt_engine_cache_.at(std::make_pair(symbol_name_, serialized_batch_size_));
    }
    const bool use_int8 = (support::GetEnv("TVM_TENSORRT_USE_INT8", 0) != 0);
    const bool int8_calibration_not_used_or_not_complete =
        (calibrator_ != nullptr && num_calibration_batches_remaining_ != 0);
    if (find_engine_flag &&
        (!use_int8 || calibrator_ == nullptr || int8_calibration_not_used_or_not_complete)) {
      // A compatible engine already exists.
      return trt_engine_cache_.at(std::make_pair(symbol_name_, compatible_engine_batch_size));
    }

    // For single engine mode, remove previous engine and update max_batch_size.
    if (!multi_engine_mode_) {
      DestroyEngines();
      max_batch_size_ = batch_size;
    }
    DLOG(INFO) << "Building new TensorRT engine for subgraph " << symbol_name_
               << " with batch size " << batch_size;

    // Build engine.
    if (calibrator_ != nullptr && num_calibration_batches_remaining_ == 0) {
      // Calibration complete and build int8 engine
      BuildEngineFromJson(batch_size);
      calibrator_.reset(nullptr);
    } else {
      // Build new engine
      BuildEngineFromJson(batch_size);
      TensorRTEngineAndContext& engine_and_context =
          trt_engine_cache_[std::make_pair(symbol_name_, batch_size)];
      if (use_int8) {
        this->CreateInt8Calibrator(engine_and_context);
      }
    }

    VLOG(1) << "Finished building TensorRT engine for subgraph " << symbol_name_
            << " with batch size " << batch_size;
    CacheEngineToDisk();
    return trt_engine_cache_.at(std::make_pair(symbol_name_, batch_size));
  }

  void BuildEngineFromJson(int batch_size) {
    const bool use_fp16 = support::GetEnv("TVM_TENSORRT_USE_FP16", false) || use_fp16_;
    TensorRTBuilder builder(&GetTensorRTLogger(), data_entry_, max_workspace_size_, use_fp16,
                            calibrator_.get());
    for (size_t i = 0; i < input_nodes_.size(); ++i) {
      auto nid = input_nodes_[i];
      const auto& node = nodes_[nid];
      std::string name = node.GetOpName();
      if (node.GetOpType() == "input") {
        builder.AddInput(nid, EntryID(nid, 0), node);
      } else {
        TVM_FFI_ICHECK_EQ(node.GetOpType(), "const");
        uint32_t eid = EntryID(nid, 0);
        builder.AddConstant(nid, data_entry_[eid]);
      }
    }

    // Add layers.
    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() != "kernel") continue;
      builder.AddLayer(nid, node);
    }

    // Add outputs.
    for (size_t i = 0; i < outputs_.size(); ++i) {
      builder.AddOutput(outputs_[i], EntryID(outputs_[i]));
    }

    TensorRTEngineAndContext engine_and_context = builder.BuildEngine();
    trt_engine_cache_[std::make_pair(symbol_name_, batch_size)] = engine_and_context;
  }

  /*! \brief If TVM_TENSORRT_CACHE_DIR is set, will check that directory for
   * already built TRT engines and load into trt_engine_cache_ so they don't
   * have to be built at first inference.
   */
  bool GetCachedEnginesFromDisk() {
    std::string cache_dir = support::GetEnv("TVM_TENSORRT_CACHE_DIR", std::string(""));
    if (cache_dir.empty()) return false;
    std::string key = GetSubgraphKey();
    std::string path = cache_dir + "/" + key + ".plan";
    // Check if engine is in the cache.
    std::ifstream infile(path, std::ios::binary);
    if (!infile.good()) return false;
    LOG(INFO) << "Loading cached TensorRT engine from " << path;
    infile.close();
    std::string serialized_engine = support::ReadBytesFromFile(path);
    // Deserialize engine. TensorRT 10 dropped the trailing IPluginFactory* argument and the runtime
    // must outlive the engine, so it is owned by the cached TensorRTEngineAndContext.
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(GetTensorRTLogger());
    TensorRTEngineAndContext engine_and_context;
    engine_and_context.runtime = runtime;
    engine_and_context.engine =
        runtime->deserializeCudaEngine(&serialized_engine[0], serialized_engine.size());
    if (engine_and_context.engine == nullptr) {
      // A stale or incompatible (e.g. different TensorRT version) .plan file. Drop it and rebuild.
      delete runtime;
      LOG(WARNING) << "Failed to deserialize cached TensorRT engine from " << path
                   << "; it will be rebuilt.";
      return false;
    }
    engine_and_context.context = engine_and_context.engine->createExecutionContext();
    // Load metadata
    namespace json = ::tvm::ffi::json;
    std::string meta_path = cache_dir + "/" + key + ".meta";
    std::string serialized_meta = support::ReadBytesFromFile(meta_path);
    auto meta_obj = json::Parse(serialized_meta).cast<json::Object>();
    int batch_size;
    // Read inputs
    {
      auto arr = meta_obj.at(ffi::String("inputs")).cast<json::Array>();
      engine_and_context.inputs.clear();
      for (const auto& v : arr) {
        engine_and_context.inputs.push_back(std::string(v.cast<ffi::String>()));
      }
    }
    // Read outputs
    {
      auto arr = meta_obj.at(ffi::String("outputs")).cast<json::Array>();
      engine_and_context.outputs.clear();
      for (const auto& v : arr) {
        engine_and_context.outputs.push_back(std::string(v.cast<ffi::String>()));
      }
    }
    // Read batch_size
    batch_size = static_cast<int>(meta_obj.at(ffi::String("batch_size")).cast<int64_t>());
    trt_engine_cache_[std::make_pair(symbol_name_, batch_size)] = engine_and_context;
    max_batch_size_ = batch_size;
    LOG(INFO) << "finished loading engine and context ... ";
    return true;
  }

  /*! \brief If TVM_TENSORRT_CACHE_DIR is set, will save the engine to that
   * directory so it can be loaded later.
   */
  void CacheEngineToDisk() {
    int batch_size = GetBatchSize();
    std::string cache_dir = support::GetEnv("TVM_TENSORRT_CACHE_DIR", std::string(""));
    if (cache_dir.empty()) return;
    std::string key = GetSubgraphKey();
    std::string path = cache_dir + "/" + key + ".plan";
    DLOG(INFO) << "Caching TensorRT engine to " << path;
    // Serialize engine to disk
    nvinfer1::IHostMemory* serialized_engine =
        trt_engine_cache_[std::make_pair(symbol_name_, batch_size)].engine->serialize();
    support::WriteBytesToFile(path, std::string(static_cast<const char*>(serialized_engine->data()),
                                                serialized_engine->size()));
    delete serialized_engine;
    // Serialize metadata
    namespace json = ::tvm::ffi::json;
    json::Object meta_obj;
    {
      json::Array inputs_arr;
      for (const auto& s : trt_engine_cache_[std::make_pair(symbol_name_, batch_size)].inputs) {
        inputs_arr.push_back(ffi::String(s));
      }
      meta_obj.Set(ffi::String("inputs"), std::move(inputs_arr));
    }
    {
      json::Array outputs_arr;
      for (const auto& s : trt_engine_cache_[std::make_pair(symbol_name_, batch_size)].outputs) {
        outputs_arr.push_back(ffi::String(s));
      }
      meta_obj.Set(ffi::String("outputs"), std::move(outputs_arr));
    }
    meta_obj.Set(ffi::String("batch_size"), static_cast<int64_t>(batch_size));
    std::string meta_path = cache_dir + "/" + key + ".meta";
    support::WriteBytesToFile(meta_path, std::string(json::Stringify(meta_obj)));
  }

  std::string GetSubgraphKey() {
    // Using this key will only allow a single model per TVM_TENSORRT_CACHE_DIR directory. We could
    // instead use a hash of graph_json and all weights to allow many models in the same directory,
    // but the cost of computing the hash is high.
    return symbol_name_ + (support::GetEnv("TVM_TENSORRT_USE_FP16", false) ? "_fp16" : "_fp32");
  }

  /*! \brief Retreive a GPU buffer for input or output or allocate if needed. Keyed by TensorRT IO
   * tensor name (TRT10 has no binding indices). */
  ffi::Tensor GetOrAllocateDeviceBuffer(const std::string& name, int entry_id) {
    if (!device_buffers_) {
      device_buffers_ = std::make_unique<std::unordered_map<std::string, ffi::Tensor>>();
    }
    auto& device_buffers = *device_buffers_;
    DLDevice device{kDLCUDA, embedded_device_id_ >= 0 ? embedded_device_id_ : 0};
    std::vector<int64_t> shape(data_entry_[entry_id]->shape,
                               data_entry_[entry_id]->shape + data_entry_[entry_id]->ndim);
    if (device_buffers.count(name)) {
      // Buffer is already initialized.
      if (shape[0] > device_buffers[name].shape()[0]) {
        // Buffer is too small. Need to allocate bigger buffer.
        device_buffers[name] = runtime::Empty(shape, data_entry_[entry_id]->dtype, device);
      } else if (shape[0] < device_buffers[name].shape()[0]) {
        // Buffer is too large. Create view.
        return tvm::runtime::CreateView(device_buffers[name], shape, data_entry_[entry_id]->dtype);
      }
    } else {
      // Buffer not initialized yet.
      device_buffers[name] = runtime::Empty(shape, data_entry_[entry_id]->dtype, device);
    }
    return device_buffers.at(name);
  }

  void CreateInt8Calibrator(const TensorRTEngineAndContext& engine_and_context) {
    // Get input names in binding order.
    std::vector<std::string> input_names;
    for (size_t i = 0; i < engine_and_context.inputs.size(); i++) {
      std::string ele = engine_and_context.inputs[i];
      input_names.push_back(ele);
    }
    const int batch_size = GetBatchSize();
    calibrator_.reset(new TensorRTCalibrator(batch_size, input_names));
  }

  /*! \brief Map of function name and max batch size to TRT engine if built already. */
  std::unordered_map<std::pair<std::string, int>, TensorRTEngineAndContext, PairHash>
      trt_engine_cache_;

  /*! \brief Device owning the deserialized compile-time engine, or -1 before the first call. */
  int embedded_device_id_ = -1;

  /*! \brief Calibrator for INT8 mode. */
  std::unique_ptr<TensorRTCalibrator> calibrator_;

  /*! \brief Map of inding index to GPU buffers for inputs and outputs. Only used when target device
   * is not "cuda". Since TensorRT execution can only read data from GPU, we need to copy data from
   * the runtime device to these buffers first. These will be allocated for the highest batch size
   * used by all engines. */
  std::unique_ptr<std::unordered_map<std::string, ffi::Tensor>> device_buffers_;

#else   // TVM_GRAPH_EXECUTOR_TENSORRT
  void Run() override {
    TVM_FFI_THROW(InternalError) << "TensorRT runtime is not enabled. "
                                 << "Please build with TVM_USE_TENSORRT_RUNTIME.";
  }

  void BuildEngine() {
    LOG(WARNING) << "TensorRT runtime is not enabled. "
                 << "Please build with TVM_USE_TENSORRT_RUNTIME.";
  }

  void BuildEngineAtCompileTime(const ffi::Array<ffi::Tensor>& consts) {
    for (const auto& tensor : consts) {
      TVM_FFI_ICHECK(tensor.defined());
    }
    TVM_FFI_THROW(InternalError)
        << "TensorRT build_at_compile_time requires TVM to be built with TVM_USE_TENSORRT_RUNTIME";
  }

  bool GetCachedEnginesFromDisk() { return false; }

  void CacheEngineToDisk() {}
#endif  // TVM_GRAPH_EXECUTOR_TENSORRT

  static constexpr uint64_t kEngineMagic = 0x31545254564d5445ULL;
  static constexpr uint32_t kEngineVersion = 1;

  // Retain serialized bytes independently of initialized CUDA objects, including after reloading.
  std::string serialized_engine_;
  std::vector<std::string> serialized_inputs_;
  std::vector<std::string> serialized_outputs_;
  int serialized_batch_size_ = 0;

  bool use_implicit_batch_;

  size_t max_workspace_size_;

  /*! \brief Number of calibration batches until we are done. */
  int num_calibration_batches_remaining_;

  /*! \brief Highest batch size that an engine has been built for, used in single-engine mode only
   * (multi_engine_mode=false). */
  int max_batch_size_;

  /*! \brief The strategy to use for dynamic batching. With multi_engine_mode=true, a new TensorRT
   * engine is created for each unique batch size encountered. With multi_engine_mode=false, only
   * one TensorRT engine is alive at any given time. It is replaced if a higher batch size is
   * encountered. Multi-engine mode should give better performance, at a cost of higher memory usage
   * and more time spent building engines. */
  bool multi_engine_mode_;

  /*! \brief Use auto-conversion to fp16 */
  bool use_fp16_;
};

ffi::Module TensorRTRuntimeCreate(const ffi::String& symbol_name, const ffi::String& graph_json,
                                  const ffi::Array<ffi::String>& const_names) {
  auto n = ffi::make_object<TensorRTRuntime>(symbol_name, graph_json, const_names);
  return ffi::Module(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.tensorrt_runtime_create", TensorRTRuntimeCreate)
      .def("ffi.Module.load_from_bytes.tensorrt", TensorRTRuntime::LoadFromBytes);
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
