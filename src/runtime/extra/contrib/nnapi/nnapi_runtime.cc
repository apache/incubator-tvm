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

#include <dlpack/dlpack.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/tensor.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "../json/json_node.h"
#include "../json/json_runtime.h"

#ifdef TVM_GRAPH_EXECUTOR_NNAPI
#include <android/NeuralNetworks.h>
#include <android/log.h>

#include "nnapi_builder.h"
#include "nnapi_ops.h"
#endif

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime::json;
using JSONGraphNode = tvm::runtime::json::JSONGraphNode;

class NNAPIRuntime : public JSONRuntimeBase {
 public:
  explicit NNAPIRuntime(const std::string& symbol_name, const std::string& graph_json,
                        const ffi::Array<ffi::String>& const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  const char* kind() const final { return "nnapi"; }

#ifdef TVM_GRAPH_EXECUTOR_NNAPI
  struct CompiledModel {
    CompiledModel(NNAPIModelBuilder builder, ANeuralNetworksCompilation* compilation,
                  std::vector<NNAPIOperand> model_output_operands)
        : builder(std::move(builder)),
          compilation(compilation),
          model_output_operands(model_output_operands) {}
    NNAPIModelBuilder builder;
    ANeuralNetworksCompilation* compilation;
    std::vector<NNAPIOperand> model_output_operands;
  };

  std::optional<CompiledModel> compiled_model_;

  void Init(const ffi::Array<ffi::Tensor>& consts) final {
    TVM_FFI_ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required constants.";
    SetupConstants(consts);
    CompileModel();
  }

  void CompileModel() {
    NNAPIModelBuilder builder;

    // Clear the map, otherwise the input shapes from last inference gets used.
    node_output_map_.clear();

    // Add inputs as NNAPI model operands.
    std::vector<NNAPIOperand> model_input_operands;
    for (size_t i = 0; i < input_nodes_.size(); ++i) {
      const uint32_t nid = input_nodes_[i];
      if (nodes_[nid].GetOpType() == "input") {
        for (size_t j = 0; j < nodes_[nid].GetOpShape().size(); ++j) {
          const auto input_shape_arr = nodes_[nid].GetOpShape()[j];
          const std::vector<int64_t> input_shape(input_shape_arr.begin(), input_shape_arr.end());
          const auto input_dtype = nodes_[nid].GetOpDataType()[j];
          const NNAPIOperand operand =
              builder.CreateOperand(input_shape.data(), input_shape.size(), input_dtype);
          node_output_map_.emplace(nid, operand);
          model_input_operands.push_back(operand);
        }
      }
    }

    // Add kernels as NNAPI operations.
    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() != "kernel") {
        continue;
      }
      AddOperation(builder, nid, node);
    }

    // Collect the output operands indices.
    std::vector<NNAPIOperand> model_output_operands;
    for (size_t i = 0; i < outputs_.size(); ++i) {
      const auto& node = outputs_[i];
      auto it = node_output_map_.find(node.id_);
      TVM_FFI_ICHECK(it != node_output_map_.end()) << "Missing model output.";
      const auto& operand = it->second;
      model_output_operands.push_back(operand);
    }

    // Finish and compile the model.
    builder.Finish(model_input_operands, model_output_operands);
    ANeuralNetworksCompilation* compilation = builder.Compile();

    // Store the compilation
    compiled_model_.emplace(std::move(builder), compilation, model_output_operands);
  }

  void ExecuteModel(ANeuralNetworksCompilation* compilation,
                    const std::vector<NNAPIOperand>& model_output_operands) {
    // Execute the model.
    ANeuralNetworksExecution* execution;
    TVM_FFI_ICHECK_EQ(ANeuralNetworksExecution_create(compilation, &execution),
                      ANEURALNETWORKS_NO_ERROR);

    for (size_t i = 0; i < input_nodes_.size(); ++i) {
      const uint32_t nid = input_nodes_[i];
      if (nodes_[nid].GetOpType() == "input") {
        for (size_t j = 0; j < nodes_[nid].GetOpShape().size(); ++j) {
          auto it = node_output_map_.find(nid);
          TVM_FFI_ICHECK(it != node_output_map_.end()) << "Missing model input.";
          const auto& operand = it->second;

          const uint32_t eid = EntryID(nid, j);
          const auto entry = data_entry_[eid];

          const auto operand_data_size = ffi::GetDataSize(*entry);
          TVM_FFI_ICHECK_EQ(
              ANeuralNetworksExecution_setInput(execution, i, operand.GetOperandType().Get(),
                                                entry->data, operand_data_size),
              ANEURALNETWORKS_NO_ERROR);
        }
      }
    }

    for (size_t i = 0; i < outputs_.size(); ++i) {
      const auto& operand = model_output_operands[i];
      const auto& node = outputs_[i];

      const auto eid = EntryID(node);
      const auto entry = data_entry_[eid];

      const auto operand_data_size = ffi::GetDataSize(*entry);
      TVM_FFI_ICHECK_EQ(
          ANeuralNetworksExecution_setOutput(execution, i, operand.GetOperandType().Get(),
                                             entry->data, operand_data_size),
          ANEURALNETWORKS_NO_ERROR);
    }

    ANeuralNetworksEvent* compute_event;
    TVM_FFI_ICHECK_EQ(ANeuralNetworksExecution_startCompute(execution, &compute_event),
                      ANEURALNETWORKS_NO_ERROR);
    TVM_FFI_ICHECK_EQ(ANeuralNetworksEvent_wait(compute_event), ANEURALNETWORKS_NO_ERROR);
    ANeuralNetworksEvent_free(compute_event);

    ANeuralNetworksExecution_free(execution);
  }

  void Run() final {
    TVM_FFI_ICHECK(compiled_model_.has_value());
    CompiledModel& compiled_model = compiled_model_.value();
    ExecuteModel(compiled_model.compilation, compiled_model.model_output_operands);
  }

  void AddOperation(NNAPIModelBuilder& builder, uint32_t nid,  // NOLINT(*)
                    const JSONGraphNode& node) {
    std::vector<NNAPIOperand> inputs;
    std::vector<NNAPIOperand> outputs;

    // Map the op name to its converter.
    const auto& converter_map = GetOpConverters();
    auto it = converter_map.find(node.GetOpName());
    TVM_FFI_ICHECK(it != converter_map.end()) << node.GetOpName() << ": Unsupported operation name";
    const NNAPIOpConverter& converter = *it->second;

    // Add input operands to params.
    for (size_t i = 0; i < node.GetInputs().size(); ++i) {
      auto in_node = node.GetInputs()[i];
      auto it = node_output_map_.find(in_node.id_);
      TVM_FFI_ICHECK(it != node_output_map_.end()) << node.GetOpName() << ": Missing input";
      auto& operand = it->second;
      inputs.push_back(operand);
    }

    // Create and add output operands to params.
    const auto output_shapes = node.GetOpShape();
    const auto output_dtypes = node.GetOpDataType();
    TVM_FFI_ICHECK(output_shapes.size() == output_dtypes.size())
        << "The number of output shapes must match the number of output dtypes";
    TVM_FFI_ICHECK(output_shapes.size() == 1)
        << "NNAPI runtime currently does not support more than one output per operation yet";

    for (size_t i = 0; i < output_shapes.size(); ++i) {
      auto output_shape = output_shapes[i];
      const NNAPIOperand output_operand =
          builder.CreateOperand(output_shape.data(), output_shape.size(), output_dtypes[i]);
      outputs.push_back(output_operand);
    }

    converter.Convert(builder, node, inputs, outputs);

    // Record the final output shape.
    node_output_map_.emplace(nid, outputs[0]);
  }

 private:
  // Mapping from JSON node IDs to NNAPI operand numbers.
  std::unordered_map<uint32_t, NNAPIOperand> node_output_map_;

#else   // ifdef TVM_GRAPH_EXECUTOR_NNAPI
  void Init(const ffi::Array<ffi::Tensor>& consts) final {
    TVM_FFI_THROW(InternalError)
        << "NNAPI runtime is not enabled. Build with TVM_USE_NNAPI_RUNTIME to enable it.";
  }

  void Run() final {
    TVM_FFI_THROW(InternalError)
        << "NNAPI runtime is not enabled. Build with TVM_USE_NNAPI_RUNTIME to enable it.";
  }
#endif  // ifdef TVM_GRAPH_EXECUTOR_NNAPI
};

ffi::Module NNAPIRuntimeCreate(const ffi::String& symbol_name, const ffi::String& graph_json,
                               const ffi::Array<ffi::String>& const_names) {
  auto n = ffi::make_object<NNAPIRuntime>(symbol_name, graph_json, const_names);
  return ffi::Module(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.nnapi_runtime_create", NNAPIRuntimeCreate)
      .def("ffi.Module.load_from_bytes.nnapi", JSONRuntimeBase::LoadFromBytes<NNAPIRuntime>);
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
