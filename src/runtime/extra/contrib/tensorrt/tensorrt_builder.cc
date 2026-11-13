/* * Licensed to the Apache Software Foundation (ASF) under one
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
 * \file runtime/contrib/tensorrt/tensorrt_builder.cc
 * \brief The TensorRTBuilder class can be used to convert a JSONRuntime graph into a TRT engine
 * which can be used for inference.
 */

#include "tensorrt_builder.h"

#include <tvm/runtime/logging.h>
#include <tvm/runtime/tensor.h>

#include <memory>
#include <string>

#include "tensorrt_logger.h"
#include "tensorrt_ops.h"
#include "tensorrt_utils.h"

namespace tvm {
namespace runtime {
namespace contrib {

TensorRTBuilder::TensorRTBuilder(TensorRTLogger* logger,
                                 const std::vector<const DLTensor*>& data_entry,
                                 size_t max_workspace_size, bool use_fp16,
                                 nvinfer1::IInt8Calibrator* calibrator)
    : trt_logger_(logger),
      data_entry_(data_entry),
      max_workspace_size_(max_workspace_size),
      use_fp16_(use_fp16),
      use_int8_(false),
      calibrator_(calibrator) {
  // Create TRT builder and network.
  std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(*trt_logger_));
  TVM_FFI_ICHECK(builder != nullptr) << "Creating the TensorRT builder failed";

  // TensorRT 10 removed implicit-batch mode and the kEXPLICIT_BATCH creation flag; every network is
  // explicit-batch, so the batch dimension is simply dimension 0 of each binding and is varied
  // through optimization profiles rather than IBuilder::setMaxBatchSize.
  if (calibrator_ != nullptr) {
    use_int8_ = true;
  }
  std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0U));
  TVM_FFI_ICHECK(network != nullptr) << "Creating the TensorRT network failed";
  builder_ = builder.release();
  network_ = network.release();
}

TensorRTBuilder::~TensorRTBuilder() { CleanUp(); }

nvinfer1::DataType DLDataType2NVDataType(DLDataType data_type) {
  TVM_FFI_ICHECK(data_type.code == kDLFloat && (data_type.bits == 16 || data_type.bits == 32))
      << "Invalid input Tensor type. Only float16 and float32 are supported";
  return (data_type.bits == 16) ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT;
}

void TensorRTBuilder::AddInput(int nid, uint32_t entry_id, const JSONGraphNode& node) {
  auto node_name = node.GetOpName();
  auto shapes = node.GetOpShape();
  auto dtypes = node.GetOpDataType();
  TVM_FFI_ICHECK_EQ(shapes.size(), dtypes.size());
  node_output_map_[nid] = {};
  for (size_t i = 0; i < shapes.size(); ++i) {
    const std::string name = node_name + "_" + std::to_string(i);
    auto shape = shapes[i];
    // TensorRT 10 is always explicit-batch: keep the full shape including the batch dimension.
    nvinfer1::Dims dims = VectorToTrtDims(shape);
    auto input_tensor = network_->addInput(name.c_str(), DLDataType2NVDataType(dtypes[i]), dims);
    node_output_map_[nid].push_back(TensorRTOpInput(input_tensor));
    network_input_names_.push_back(name);
    entry_id_map_[name] = entry_id + i;
  }
}

void TensorRTBuilder::AddConstant(int nid, const DLTensor* data) {
  nvinfer1::Weights weight = GetDLTensorAsWeights(data, kDLCPU);
  std::vector<int> shape(data->shape, data->shape + data->ndim);
  node_output_map_[nid] = {TensorRTOpInput(weight, shape)};
}

void TensorRTBuilder::AddOutput(const JSONGraphNodeEntry& node, uint32_t entry_id) {
  auto it = node_output_map_.find(node.id_);
  TVM_FFI_ICHECK(it != node_output_map_.end()) << "Output was not found.";
  auto out_tensor = it->second[node.index_].tensor;
  std::string name = "tensorrt_output_" + std::to_string(network_output_names_.size());
  // If the network is already marked as an input or output, make a copy to avoid TRT crash.
  if (out_tensor->isNetworkOutput()) {
    LOG(WARNING) << name << " is a duplicate output.";
    out_tensor = network_->addIdentity(*out_tensor)->getOutput(0);
  } else if (out_tensor->isNetworkInput()) {
    LOG(WARNING) << name << " is both an input and an output.";
    out_tensor = network_->addIdentity(*out_tensor)->getOutput(0);
  }
  out_tensor->setName(name.c_str());
  network_->markOutput(*out_tensor);
  network_output_names_.push_back(name);
  entry_id_map_[name] = entry_id;
}

void TensorRTBuilder::AddLayer(int nid, const JSONGraphNode& node) {
  TensorRTOpConverterParams params(network_, nid, node, &trt_weights_);
  // Look up converter.
  const std::unordered_map<std::string, std::unique_ptr<TensorRTOpConverter>>& map =
      GetOpConverters();
  auto it = map.find(params.op_name);
  TVM_FFI_ICHECK(it != map.end()) << params.op_name << ": Unsupported operator";
  const TensorRTOpConverter& converter = *it->second;
  if (!converter.variable_input_count) {
    TVM_FFI_ICHECK_EQ(node.GetInputs().size(), converter.input_types.size())
        << params.op_name << ": Mismatched input sizes";
  }
  // Get inputs.
  for (size_t i = 0; i < node.GetInputs().size(); ++i) {
    auto in_node = node.GetInputs()[i];
    auto it = node_output_map_.find(in_node.id_);
    TVM_FFI_ICHECK(it != node_output_map_.end()) << params.op_name << ": Input was not found";
    auto input = it->second[in_node.index_];
    if (!converter.variable_input_count) {
      if (converter.input_types[i] == kTensor && input.type == kWeight) {
        input = TensorRTOpInput(GetInputAsTensor(input));
      } else if (converter.input_types[i] == kWeight && input.type == kTensor) {
        TVM_FFI_THROW(InternalError) << params.op_name << ": Input " << i << " must be a constant.";
      }
    }
    params.inputs.push_back(input);
  }

  // Convert op to TRT.
  converter.Convert(&params);

  // Get outputs.
  node_output_map_[nid] = {};
  auto dtype = node.GetOpDataType();
  TVM_FFI_ICHECK_EQ(params.outputs.size(), dtype.size())
      << params.op_name << ": Mismatched output sizes";
  for (size_t i = 0; i < params.outputs.size(); ++i) {
    auto out = params.outputs[i];
    out->setType(DLDataType2NVDataType(dtype[i]));
    node_output_map_[nid].push_back(TensorRTOpInput(out));
  }
}

TensorRTEngineAndContext TensorRTBuilder::BuildEngine() {
  // Build engine.
  config_ = builder_->createBuilderConfig();
  TVM_FFI_ICHECK(config_ != nullptr) << "Creating the TensorRT builder config failed";
  // TensorRT 10 replaced IBuilderConfig::setMaxWorkspaceSize with a tunable memory pool.
  config_->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, max_workspace_size_);
  // Disable TF32 (on by default on Ampere+) so FP32 layers match TVM's full-precision reference.
  config_->clearFlag(nvinfer1::BuilderFlag::kTF32);
  if (use_fp16_) {
    config_->setFlag(nvinfer1::BuilderFlag::kFP16);
  }

  if (use_int8_) {
    config_->setFlag(nvinfer1::BuilderFlag::kINT8);
    TVM_FFI_ICHECK(calibrator_);
    config_->setInt8Calibrator(calibrator_);
    LOG(INFO) << "config finishes setting up calibrator as INT8 mode ... ";
  }

  // Every network is explicit-batch in TRT10, so always add an optimization profile that pins each
  // input to its concrete shape (with a minimum batch of 1 for dynamic batch dimensions).
  // The builder owns this profile and releases it during CleanUp.
  auto profile = builder_->createOptimizationProfile();
  TVM_FFI_ICHECK(profile != nullptr) << "Creating the TensorRT optimization profile failed";
  for (int i = 0; i < network_->getNbInputs(); ++i) {
    auto name = network_->getInput(i)->getName();
    const uint32_t entry_id = entry_id_map_[name];
    std::vector<int64_t> shape(data_entry_[entry_id]->shape,
                               data_entry_[entry_id]->shape + data_entry_[entry_id]->ndim);
    auto dims = VectorToTrtDims(shape);

    profile->setDimensions(name, nvinfer1::OptProfileSelector::kOPT, dims);
    profile->setDimensions(name, nvinfer1::OptProfileSelector::kMAX, dims);
    // The network inputs are built with static shapes, so the profile must match them exactly; only
    // lower kMIN for a genuinely dynamic (-1) leading dimension.
    if (network_->getInput(i)->getDimensions().nbDims >= 1 &&
        network_->getInput(i)->getDimensions().d[0] == -1) {
      dims.d[0] = 1;
    }
    profile->setDimensions(name, nvinfer1::OptProfileSelector::kMIN, dims);
  }
  config_->addOptimizationProfile(profile);

  // TensorRT 10 removed buildEngineWithConfig; build a serialized engine and deserialize it through
  // an IRuntime that is kept alive alongside the engine (TensorRTEngineAndContext::runtime).
  std::unique_ptr<nvinfer1::IHostMemory> plan(
      builder_->buildSerializedNetwork(*network_, *config_));
  TVM_FFI_ICHECK(plan != nullptr) << "Failed to build TensorRT serialized network.";
  std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(*trt_logger_));
  TVM_FFI_ICHECK(runtime != nullptr) << "Creating the TensorRT deserialization runtime failed";
  std::unique_ptr<nvinfer1::ICudaEngine> engine(
      runtime->deserializeCudaEngine(plan->data(), plan->size()));
  TVM_FFI_ICHECK(engine != nullptr) << "Failed to deserialize the TensorRT engine.";
  TVM_FFI_ICHECK_EQ(engine->getNbIOTensors(), static_cast<int32_t>(network_input_names_.size() +
                                                                   network_output_names_.size()));
  std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
  TVM_FFI_ICHECK(context != nullptr) << "Creating the TensorRT execution context failed";
  CleanUp();

  TensorRTEngineAndContext result;
  result.inputs = network_input_names_;
  result.outputs = network_output_names_;
  result.runtime = runtime.release();
  result.engine = engine.release();
  result.context = context.release();
  return result;
}

nvinfer1::Weights TensorRTBuilder::GetDLTensorAsWeights(const DLTensor* dptr,
                                                        DLDeviceType src_device) {
  TVM_FFI_ICHECK_EQ(dptr->device.device_type, src_device);
  TVM_FFI_ICHECK((dptr->dtype.bits != 16 || dptr->dtype.bits != 32))
      << "Invalid input Tensor type. Float16 and Float32 are supported";
  const auto trt_dtype = (static_cast<int>(dptr->dtype.bits) == 16) ? nvinfer1::DataType::kHALF
                                                                    : nvinfer1::DataType::kFLOAT;

  const size_t weight_bytes = ffi::GetDataSize(*dptr);
  nvinfer1::Weights weight{trt_dtype, nullptr, 0};
  size_t count = 1;
  for (ffi::Shape::index_type i = 0; i < dptr->ndim; ++i) {
    count *= dptr->shape[i];
  }
  weight.count = count;
  std::unique_ptr<float[]> values(new float[count]);
  weight.values = values.get();
  // Keep temporary storage owned until both the copy and registration succeed.
  tvm::runtime::CopyToBytes(dptr, values.get(), weight_bytes);
  trt_weights_.push_back(weight);
  values.release();
  return weight;
}

nvinfer1::ITensor* TensorRTBuilder::GetInputAsTensor(const TensorRTOpInput& input) {
  if (input.type == kTensor) return input.tensor;
  auto shape = input.weight_shape;
  // TensorRT 10 is always explicit-batch, so the constant keeps its full shape.
  return network_->addConstant(VectorToTrtDims(shape), input.weight)->getOutput(0);
}

void TensorRTBuilder::CleanUp() {
  // TensorRT 10 removed obj->destroy(); objects are released with the delete operator.
  VLOG(1) << "Destroying TensorRT network";
  delete network_;
  network_ = nullptr;

  VLOG(1) << "Destroying TensorRT config";
  delete config_;
  config_ = nullptr;

  VLOG(1) << "Destroying TensorRT builder";
  delete builder_;
  builder_ = nullptr;

  VLOG(1) << "Destroying TensorRT weights";
  for (auto weight : trt_weights_) {
    if (weight.type == nvinfer1::DataType::kFLOAT || weight.type == nvinfer1::DataType::kHALF) {
      delete[] static_cast<const float*>(weight.values);
    } else {
      delete[] static_cast<const uint16_t*>(weight.values);
    }
  }
  trt_weights_.clear();
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
