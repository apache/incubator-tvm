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
 * \file src/runtime/extra/contrib/example_npu/example_npu_runtime.cc
 * \brief Example NPU runtime demonstrating architectural concepts
 *
 * This runtime demonstrates key NPU architectural patterns:
 * - Multi-level memory hierarchy management
 * - Tiling for on-chip memory optimization
 * - Quantization/dequantization handling
 * - Operator fusion for reduced memory traffic
 * - Power-aware execution modes
 */

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/tensor.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "../json/json_node.h"
#include "../json/json_runtime.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

/*!
 * \brief NPU Memory Tier representation
 *
 * Models the hierarchical memory structure common in NPUs
 */
enum class MemoryTier {
  L0_REGISTER,  // Register file (immediate access)
  L1_SRAM,      // On-chip SRAM/scratchpad (single cycle)
  L2_CMX,       // Compute memory/shared memory (few cycles)
  L3_DRAM       // External DRAM (high latency)
};

/*!
 * \brief NPU Power Mode configuration
 */
enum class PowerMode {
  HIGH_PERFORMANCE,  // Maximum frequency, all units active
  BALANCED,          // Moderate frequency, selective unit activation
  LOW_POWER          // Reduced frequency, minimal units
};

/*!
 * \brief NPU Execution Engine types
 */
enum class ExecutionEngine {
  MATRIX_ENGINE,     // Systolic array/tensor cores
  VECTOR_ENGINE,     // SIMD vector units
  CONV_ENGINE,       // Specialized convolution hardware
  POOLING_ENGINE,    // Dedicated pooling units
  ACTIVATION_ENGINE  // Hardware activation functions
};

/*!
 * \brief NPU Memory allocation tracker
 *
 * Manages memory across different tiers for optimal data placement
 */
class NPUMemoryManager {
 public:
  NPUMemoryManager() {
    // Initialize memory sizes (in KB) - typical NPU values
    memory_sizes_[MemoryTier::L0_REGISTER] = 4;
    memory_sizes_[MemoryTier::L1_SRAM] = 256;
    memory_sizes_[MemoryTier::L2_CMX] = 512;
    memory_sizes_[MemoryTier::L3_DRAM] = 1024 * 1024;  // 1GB

    // Initialize available memory
    for (const auto& tier : memory_sizes_) {
      available_memory_[tier.first] = tier.second * 1024;  // Convert to bytes
    }
  }

  /*!
   * \brief Allocate memory in the appropriate tier
   * \param size_bytes Size to allocate
   * \param preferred_tier Preferred memory tier
   * \return Allocated memory tier
   */
  MemoryTier AllocateMemory(size_t size_bytes, MemoryTier preferred_tier) {
    // Try to allocate in preferred tier first
    if (available_memory_[preferred_tier] >= size_bytes) {
      available_memory_[preferred_tier] -= size_bytes;
      allocated_blocks_.push_back({preferred_tier, size_bytes});
      return preferred_tier;
    }

    // Fall back to higher tiers if needed
    for (int tier = static_cast<int>(preferred_tier) + 1;
         tier <= static_cast<int>(MemoryTier::L3_DRAM); ++tier) {
      MemoryTier current_tier = static_cast<MemoryTier>(tier);
      if (available_memory_[current_tier] >= size_bytes) {
        available_memory_[current_tier] -= size_bytes;
        allocated_blocks_.push_back({current_tier, size_bytes});
        LOG(INFO) << "Memory spilled from tier " << static_cast<int>(preferred_tier) << " to tier "
                  << tier;
        return current_tier;
      }
    }

    LOG(FATAL) << "Out of NPU memory for allocation of " << size_bytes << " bytes";
    return MemoryTier::L3_DRAM;
  }

  /*!
   * \brief Get memory access cost for a tier
   */
  int GetMemoryAccessCost(MemoryTier tier) {
    static const std::unordered_map<MemoryTier, int> access_costs = {{MemoryTier::L0_REGISTER, 0},
                                                                     {MemoryTier::L1_SRAM, 1},
                                                                     {MemoryTier::L2_CMX, 4},
                                                                     {MemoryTier::L3_DRAM, 100}};
    return access_costs.at(tier);
  }

 private:
  std::unordered_map<MemoryTier, size_t> memory_sizes_;
  std::unordered_map<MemoryTier, size_t> available_memory_;
  std::vector<std::pair<MemoryTier, size_t>> allocated_blocks_;
};

/*!
 * \brief NPU Tiling engine for large tensors
 *
 * Demonstrates how NPUs tile large tensors to fit in on-chip memory
 */
class NPUTilingEngine {
 public:
  struct TileInfo {
    int tile_h;
    int tile_w;
    int num_tiles_h;
    int num_tiles_w;
    size_t tile_size_bytes;
  };

  /*!
   * \brief Calculate optimal tiling for a tensor
   */
  static TileInfo CalculateTiling(const ffi::Array<int64_t>& shape, size_t dtype_bytes,
                                  size_t available_sram_bytes) {
    TileInfo info;

    // Default tile size (typical NPU values)
    info.tile_h = 32;
    info.tile_w = 32;

    if (shape.size() < 2) {
      info.num_tiles_h = 1;
      info.num_tiles_w = 1;
      info.tile_size_bytes = dtype_bytes;
      for (auto dim : shape) {
        info.tile_size_bytes *= dim;
      }
      return info;
    }

    int64_t height = shape[shape.size() - 2];
    int64_t width = shape[shape.size() - 1];

    // Adjust tile size to fit in SRAM
    size_t tile_elements = info.tile_h * info.tile_w;
    size_t batch_channels = 1;
    for (size_t i = 0; i < shape.size() - 2; ++i) {
      batch_channels *= shape[i];
    }

    info.tile_size_bytes = tile_elements * batch_channels * dtype_bytes;

    // Reduce tile size if needed
    while (info.tile_size_bytes > available_sram_bytes && (info.tile_h > 8 || info.tile_w > 8)) {
      info.tile_h = std::max(8, info.tile_h / 2);
      info.tile_w = std::max(8, info.tile_w / 2);
      tile_elements = info.tile_h * info.tile_w;
      info.tile_size_bytes = tile_elements * batch_channels * dtype_bytes;
    }

    // Calculate number of tiles needed
    info.num_tiles_h = (height + info.tile_h - 1) / info.tile_h;
    info.num_tiles_w = (width + info.tile_w - 1) / info.tile_w;

    LOG(INFO) << "Tiling tensor to " << info.num_tiles_h << "x" << info.num_tiles_w
              << " tiles of size " << info.tile_h << "x" << info.tile_w;

    return info;
  }
};

/*!
 * \brief NPU Quantization handler
 *
 * Demonstrates quantization/dequantization for NPU acceleration
 */
class NPUQuantizationEngine {
 public:
  /*!
   * \brief Quantize float32 to int8
   */
  static void QuantizeToInt8(const float* input, int8_t* output, size_t num_elements, float scale,
                             int zero_point) {
    for (size_t i = 0; i < num_elements; ++i) {
      int quantized = static_cast<int>(std::round(input[i] / scale + zero_point));
      quantized = std::max(-128, std::min(127, quantized));
      output[i] = static_cast<int8_t>(quantized);
    }
  }

  /*!
   * \brief Dequantize int8 to float32
   */
  static void DequantizeFromInt8(const int8_t* input, float* output, size_t num_elements,
                                 float scale, int zero_point) {
    for (size_t i = 0; i < num_elements; ++i) {
      output[i] = scale * (static_cast<float>(input[i]) - zero_point);
    }
  }

  /*!
   * \brief Calculate quantization parameters
   */
  static std::pair<float, int> CalculateQuantizationParams(const float* data, size_t num_elements) {
    if (num_elements == 0) return {1.0f, 0};
    float min_val = *std::min_element(data, data + num_elements);
    float max_val = *std::max_element(data, data + num_elements);

    // Guard against zero range (e.g. constant tensor) to avoid division by zero.
    float scale = std::max(max_val - min_val, 1e-7f) / 255.0f;
    int zero_point = static_cast<int>(std::round(-min_val / scale));

    return {scale, zero_point};
  }
};

/*!
 * \brief Example NPU runtime implementation with architectural concepts
 */
class ExampleNPURuntime : public JSONRuntimeBase {
 public:
  ExampleNPURuntime(const std::string& symbol_name, const std::string& graph_json,
                    const ffi::Array<ffi::String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names), power_mode_(PowerMode::BALANCED) {}

  ~ExampleNPURuntime() = default;

  const char* kind() const override { return "example_npu_json"; }

  /*!
   * \brief Initialize the runtime with NPU-specific setup
   */
  void Init(const ffi::Array<ffi::Tensor>& consts) override {
    TVM_FFI_ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required constants.";

    SetupConstants(consts);

    // NPU-specific initialization
    LOG(INFO) << "Initializing Example NPU Runtime";
    LOG(INFO) << "  Memory hierarchy: L0(4KB) -> L1(256KB) -> L2(512KB) -> L3(DRAM)";
    LOG(INFO) << "  Execution engines: Matrix, Vector, Conv, Pooling, Activation";
    LOG(INFO) << "  Power mode: " << GetPowerModeString();
    LOG(INFO) << "  Graph nodes: " << nodes_.size();

    // Analyze graph for optimization opportunities
    AnalyzeGraphForOptimization();
  }

  /*!
   * \brief Run the computation graph with NPU execution model
   */
  void Run() override {
    LOG(INFO) << "Executing on Example NPU with " << nodes_.size() << " operations";

    // Process each node
    for (size_t i = 0; i < nodes_.size(); ++i) {
      const auto& node = nodes_[i];

      if (node.GetOpType() == "kernel") {
        const std::string& op_name = node.GetOpName();

        // Select execution engine based on operation
        ExecutionEngine engine = SelectExecutionEngine(op_name);
        LOG(INFO) << "Operation " << op_name << " -> Engine: " << GetEngineString(engine);

        // Check for fusion opportunities
        bool is_fused = op_name.find("fused") != std::string::npos;
        if (is_fused) {
          LOG(INFO) << "  Executing fused operation - reducing memory traffic";
        }

        // Dispatch to appropriate implementation.
        // More specific names must be checked before more general ones, since
        // op_name.find() is a substring match (e.g. "depthwise_conv2d" also
        // contains "conv2d", and "dequantize" also contains "quantize").
        if (op_name.find("matmul") != std::string::npos ||
            op_name.find("dense") != std::string::npos) {
          ExecuteMatMul(node, engine, is_fused);
        } else if (op_name.find("depthwise") != std::string::npos) {
          ExecuteDepthwiseConv2D(node, engine);
        } else if (op_name.find("conv2d") != std::string::npos) {
          ExecuteConv2D(node, engine, is_fused);
        } else if (op_name.find("conv1d") != std::string::npos) {
          ExecuteConv1D(node, engine);
        } else if (op_name.find("pool") != std::string::npos) {
          ExecutePooling(node, engine);
        } else if (op_name.find("relu") != std::string::npos ||
                   op_name.find("sigmoid") != std::string::npos ||
                   op_name.find("tanh") != std::string::npos) {
          ExecuteActivation(node, engine);
        } else if (op_name.find("batch_norm") != std::string::npos) {
          ExecuteBatchNorm(node, engine);
        } else if (op_name.find("add") != std::string::npos ||
                   op_name.find("multiply") != std::string::npos) {
          ExecuteElementwise(node, engine);
        } else if (op_name.find("dequantize") != std::string::npos) {
          ExecuteDequantization(node);
        } else if (op_name.find("quantize") != std::string::npos) {
          ExecuteQuantization(node);
        } else {
          LOG(WARNING) << "Unsupported operation: " << op_name;
        }
      }
    }

    LOG(INFO) << "NPU execution completed";
  }

 private:
  NPUMemoryManager memory_manager_;
  PowerMode power_mode_;
  std::unordered_map<std::string, int> op_fusion_groups_;

  /*!
   * \brief Select the appropriate NPU execution engine
   */
  ExecutionEngine SelectExecutionEngine(const std::string& op_name) {
    if (op_name.find("conv") != std::string::npos) {
      return ExecutionEngine::CONV_ENGINE;
    } else if (op_name.find("matmul") != std::string::npos ||
               op_name.find("dense") != std::string::npos) {
      return ExecutionEngine::MATRIX_ENGINE;
    } else if (op_name.find("pool") != std::string::npos) {
      return ExecutionEngine::POOLING_ENGINE;
    } else if (op_name.find("relu") != std::string::npos ||
               op_name.find("sigmoid") != std::string::npos) {
      return ExecutionEngine::ACTIVATION_ENGINE;
    } else {
      return ExecutionEngine::VECTOR_ENGINE;
    }
  }

  /*!
   * \brief Analyze graph for NPU optimization opportunities
   */
  void AnalyzeGraphForOptimization() {
    LOG(INFO) << "Analyzing graph for NPU optimizations:";

    int fusion_opportunities = 0;
    int quantization_candidates = 0;
    size_t total_memory_required = 0;

    for (const auto& node : nodes_) {
      if (node.GetOpType() == "kernel") {
        const std::string& op_name = node.GetOpName();

        // Check for fusion
        if (op_name.find("fused") != std::string::npos) {
          fusion_opportunities++;
        }

        // Check for quantization opportunities
        if (node.HasAttr("T")) {
          auto dtype_iter = node.GetAttr<ffi::Array<ffi::String>>("T");
          if (!dtype_iter.empty() && dtype_iter[0] == "int8") {
            quantization_candidates++;
          }
        }

        // Estimate memory requirements
        auto shape_iter = node.GetOpShape();
        if (!shape_iter.empty()) {
          size_t node_memory = 4;  // bytes per element
          for (const auto& output_shape : shape_iter) {
            for (auto dim : output_shape) {
              node_memory *= dim;
            }
          }
          total_memory_required += node_memory;
        }
      }
    }

    LOG(INFO) << "  Fusion opportunities: " << fusion_opportunities;
    LOG(INFO) << "  Quantization candidates: " << quantization_candidates;
    LOG(INFO) << "  Total memory required: " << total_memory_required / (1024.0 * 1024.0) << " MB";

    // Determine if tiling is needed
    if (total_memory_required > 256 * 1024) {  // > 256KB SRAM
      LOG(INFO) << "  Tiling will be required for large tensors";
    }
  }

  /*!
   * \brief Execute matrix multiplication on NPU matrix engine
   */
  void ExecuteMatMul(const JSONGraphNode& node, ExecutionEngine engine, bool is_fused) {
    LOG(INFO) << "  Executing MatMul on " << GetEngineString(engine);

    // Get input shapes
    const auto& inputs = node.GetInputs();
    if (inputs.size() >= 2) {
      // Demonstrate memory allocation
      MemoryTier input_tier = memory_manager_.AllocateMemory(1024 * 4, MemoryTier::L1_SRAM);
      MemoryTier weight_tier = memory_manager_.AllocateMemory(1024 * 4, MemoryTier::L1_SRAM);

      LOG(INFO) << "    Input allocated in tier " << static_cast<int>(input_tier);
      LOG(INFO) << "    Weights allocated in tier " << static_cast<int>(weight_tier);

      // Check if operation fits matrix engine dimensions (e.g., 16x16)
      LOG(INFO) << "    Using 16x16 systolic array for acceleration";
    }

    if (is_fused) {
      LOG(INFO) << "    Fused with activation - saving memory bandwidth";
    }

    // In a real implementation: dispatch to NPU matrix multiplication unit
  }

  /*!
   * \brief Execute 2D convolution with tiling if needed
   */
  void ExecuteConv2D(const JSONGraphNode& node, ExecutionEngine engine, bool is_fused) {
    LOG(INFO) << "  Executing Conv2D on " << GetEngineString(engine);

    // Get operation shape
    const auto& shapes = node.GetOpShape();
    if (!shapes.empty()) {
      const auto& output_shape = shapes[0];

      // Calculate if tiling is needed
      size_t output_size = 4;  // float32
      for (auto dim : output_shape) {
        output_size *= dim;
      }

      if (output_size > 256 * 1024) {  // Larger than L1 SRAM
        auto tile_info = NPUTilingEngine::CalculateTiling(output_shape, 4, 256 * 1024);

        LOG(INFO) << "    Tiling required: " << tile_info.num_tiles_h << "x"
                  << tile_info.num_tiles_w << " tiles";
        LOG(INFO) << "    Tile size: " << tile_info.tile_h << "x" << tile_info.tile_w;

        // Process tiles sequentially
        for (int th = 0; th < tile_info.num_tiles_h; ++th) {
          for (int tw = 0; tw < tile_info.num_tiles_w; ++tw) {
            LOG(INFO) << "      Processing tile [" << th << "," << tw << "]";
            // In a real implementation: process tile on NPU
          }
        }
      } else {
        LOG(INFO) << "    Single-pass execution (fits in L1 SRAM)";
      }

      if (is_fused) {
        LOG(INFO) << "    Fused with activation - saving memory bandwidth";
      }
    }

    // Check for quantized execution
    if (node.HasAttr("T")) {
      auto dtype_iter = node.GetAttr<ffi::Array<ffi::String>>("T");
      if (!dtype_iter.empty() && dtype_iter[0] == "int8") {
        LOG(INFO) << "    Using INT8 convolution for 4x speedup";
      }
    }
  }

  /*!
   * \brief Execute 1D convolution using vector engine
   */
  void ExecuteConv1D(const JSONGraphNode& node, ExecutionEngine engine) {
    LOG(INFO) << "  Executing Conv1D on " << GetEngineString(engine);
    LOG(INFO) << "    Vectorization width: 64 elements";

    // In a real implementation: dispatch to vector processing unit
  }

  /*!
   * \brief Execute depthwise convolution with channel parallelism
   */
  void ExecuteDepthwiseConv2D(const JSONGraphNode& node, ExecutionEngine engine) {
    LOG(INFO) << "  Executing DepthwiseConv2D on " << GetEngineString(engine);
    LOG(INFO) << "    Channel-parallel execution for efficiency";

    // In a real implementation: process each channel independently
  }

  /*!
   * \brief Execute pooling with streaming
   */
  void ExecutePooling(const JSONGraphNode& node, ExecutionEngine engine) {
    LOG(INFO) << "  Executing Pooling on " << GetEngineString(engine);
    LOG(INFO) << "    Streaming mode - no intermediate storage";

    // In a real implementation: stream through pooling unit
  }

  /*!
   * \brief Execute activation function
   */
  void ExecuteActivation(const JSONGraphNode& node, ExecutionEngine engine) {
    const std::string& op_name = node.GetOpName();
    LOG(INFO) << "  Executing Activation on " << GetEngineString(engine);

    if (op_name.find("sigmoid") != std::string::npos || op_name.find("tanh") != std::string::npos) {
      LOG(INFO) << "    Using lookup table for complex activation";
    } else if (op_name.find("relu") != std::string::npos) {
      LOG(INFO) << "    Using comparator unit for ReLU";
    }

    // In a real implementation: dispatch to activation unit
  }

  /*!
   * \brief Execute batch normalization
   */
  void ExecuteBatchNorm(const JSONGraphNode& node, ExecutionEngine engine) {
    LOG(INFO) << "  Executing BatchNorm on " << GetEngineString(engine);
    LOG(INFO) << "    Computing in float16 for efficiency";
    LOG(INFO) << "    Fusion candidate with previous convolution";

    // In a real implementation: fuse with conv if possible
  }

  /*!
   * \brief Execute element-wise operations
   */
  void ExecuteElementwise(const JSONGraphNode& node, ExecutionEngine engine) {
    LOG(INFO) << "  Executing Elementwise on " << GetEngineString(engine);
    LOG(INFO) << "    SIMD width: 64 elements";

    // In a real implementation: vectorized execution
  }

  /*!
   * \brief Execute quantization
   */
  void ExecuteQuantization(const JSONGraphNode& node) {
    LOG(INFO) << "  Executing Quantization";
    LOG(INFO) << "    Converting float32 -> int8";

    // Example quantization (in real NPU, this would be hardware-accelerated)
    float dummy_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto [scale, zero_point] = NPUQuantizationEngine::CalculateQuantizationParams(dummy_data, 4);

    LOG(INFO) << "    Scale: " << scale << ", Zero point: " << zero_point;
  }

  /*!
   * \brief Execute dequantization
   */
  void ExecuteDequantization(const JSONGraphNode& node) {
    LOG(INFO) << "  Executing Dequantization";
    LOG(INFO) << "    Converting int8 -> float32";

    // In a real implementation: hardware dequantization
  }

  /*!
   * \brief Get string representation of power mode
   */
  std::string GetPowerModeString() const {
    switch (power_mode_) {
      case PowerMode::HIGH_PERFORMANCE:
        return "HIGH_PERFORMANCE";
      case PowerMode::BALANCED:
        return "BALANCED";
      case PowerMode::LOW_POWER:
        return "LOW_POWER";
      default:
        return "UNKNOWN";
    }
  }

  /*!
   * \brief Get string representation of execution engine
   */
  std::string GetEngineString(ExecutionEngine engine) const {
    switch (engine) {
      case ExecutionEngine::MATRIX_ENGINE:
        return "MATRIX_ENGINE";
      case ExecutionEngine::VECTOR_ENGINE:
        return "VECTOR_ENGINE";
      case ExecutionEngine::CONV_ENGINE:
        return "CONV_ENGINE";
      case ExecutionEngine::POOLING_ENGINE:
        return "POOLING_ENGINE";
      case ExecutionEngine::ACTIVATION_ENGINE:
        return "ACTIVATION_ENGINE";
      default:
        return "UNKNOWN";
    }
  }
};

/*!
 * \brief Create the Example NPU runtime module
 */
ffi::Module ExampleNPURuntimeCreate(const ffi::String& symbol_name, const ffi::String& graph_json,
                                    const ffi::Array<ffi::String>& const_names) {
  auto n = tvm::ffi::make_object<ExampleNPURuntime>(symbol_name, graph_json, const_names);
  return ffi::Module(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.ExampleNPUJSONRuntimeCreate", ExampleNPURuntimeCreate)
      .def("ffi.Module.load_from_bytes.example_npu_json",
           JSONRuntimeBase::LoadFromBytes<ExampleNPURuntime>);
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
