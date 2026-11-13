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
 * \file src/runtime/contrib/clml/clml_runtime.h
 * \brief CLML header
 */
#ifndef TVM_RUNTIME_CONTRIB_CLML_CLML_RUNTIME_H_
#define TVM_RUNTIME_CONTRIB_CLML_CLML_RUNTIME_H_

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#if !defined(CL_TARGET_OPENCL_VERSION)
#define CL_TARGET_OPENCL_VERSION 300
#endif

#include <CL/cl.h>
#include <CL/opencl.h>
#include <stdlib.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/runtime/tensor.h>
#include <tvm/support/file_utils.h>

#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../json/json_node.h"
#include "../json/json_runtime.h"

#ifdef TVM_GRAPH_EXECUTOR_CLML
#include <CL/cl_qcom_ml_ops.h>

#include "backend/opencl/runtime/opencl_common.h"

#define CAT_I(a, b) a##b
#define CAT(a, b) CAT_I(a, b)

#define CLML_CHECK_ERROR(e, API)                                         \
  {                                                                      \
    TVM_FFI_ICHECK(e == CL_SUCCESS) << "CLML Error:" #API " code=" << e; \
  }

#if CL_QCOM_ML_OPS_H_MAJOR_VERSION > 4
#define V5_API(API, ...)                                                            \
  e = (reinterpret_cast<CLMLInterfaceV5QCOM*>(CLMLWorkspace::Global()->h_ClmlIntf)) \
          ->API(__VA_ARGS__);                                                       \
  CLML_CHECK_ERROR(e, API);
#else
#define V5_API(API, ...) \
  TVM_FFI_THROW(InternalError) << "CLML Error:" #API " - Incompatible V5 API call\n";
#endif
#if CL_QCOM_ML_OPS_H_MAJOR_VERSION > 3
#define V4_API(API, ...)                                                            \
  e = (reinterpret_cast<CLMLInterfaceV4QCOM*>(CLMLWorkspace::Global()->h_ClmlIntf)) \
          ->API(__VA_ARGS__);                                                       \
  CLML_CHECK_ERROR(e, API);
#else
#define V4_API(API, ...) \
  TVM_FFI_THROW(InternalError) << "CLML Error:" #API " - Incompatible V4 API call\n";
#endif

#if CL_QCOM_ML_OPS_H_MAJOR_VERSION > 2
#define V3_API(API, ...)                                                            \
  e = (reinterpret_cast<CLMLInterfaceV3QCOM*>(CLMLWorkspace::Global()->h_ClmlIntf)) \
          ->API(__VA_ARGS__);                                                       \
  CLML_CHECK_ERROR(e, API);
#else
#define V3_API(API, ...) \
  TVM_FFI_THROW(InternalError) << "CLML Error:" #API " - Incompatible V3 API call\n";
#endif

#if CL_QCOM_ML_OPS_H_MAJOR_VERSION > 1
#define V2_API(API, ...)                                                            \
  e = (reinterpret_cast<CLMLInterfaceV2QCOM*>(CLMLWorkspace::Global()->h_ClmlIntf)) \
          ->API(__VA_ARGS__);                                                       \
  CLML_CHECK_ERROR(e, API);
#else
#define V2_API(API, ...) \
  TVM_FFI_THROW(InternalError) << "CLML Error:" #API " - Incompatible V2 API call\n";
#endif

#define V1_API(API, ...)                                                            \
  e = (reinterpret_cast<CLMLInterfaceV1QCOM*>(CLMLWorkspace::Global()->h_ClmlIntf)) \
          ->API(__VA_ARGS__);                                                       \
  CLML_CHECK_ERROR(e, API);

#define CLML_CALL(API, ...)                                                                    \
  {                                                                                            \
    cl_int e;                                                                                  \
    switch (CLMLWorkspace::Global()->target_major) {                                           \
      case 1:                                                                                  \
        V1_API(API, __VA_ARGS__);                                                              \
        break;                                                                                 \
      case 2:                                                                                  \
        V2_API(API, __VA_ARGS__);                                                              \
        break;                                                                                 \
      case 3:                                                                                  \
        V3_API(API, __VA_ARGS__);                                                              \
        break;                                                                                 \
      case 4:                                                                                  \
        V4_API(API, __VA_ARGS__);                                                              \
        break;                                                                                 \
      default:                                                                                 \
        TVM_FFI_THROW(InternalError) << "CLML Error:" #API " - Unsupported target version \n"; \
    }                                                                                          \
  }

#define CLML_CALL_VERSIONED(APICALL, VERSION, ...) CAT(CAT(V, VERSION), _API)(APICALL, __VA_ARGS__)

#define CALL_CASE(VERSION, API, ...)                \
  case VERSION:                                     \
    CLML_CALL_VERSIONED(API, VERSION, __VA_ARGS__); \
    break;

// clCreateMLOpClipQCOM
#define CLML_CALL_clCreateMLOpClipQCOM(...)                                          \
  cl_int e;                                                                          \
  switch (CLMLWorkspace::Global()->target_major) {                                   \
    CALL_CASE(2, clCreateMLOpClipQCOM, __VA_ARGS__)                                  \
    CALL_CASE(3, clCreateMLOpClipQCOM, __VA_ARGS__)                                  \
    CALL_CASE(4, clCreateMLOpClipQCOM, __VA_ARGS__)                                  \
    CALL_CASE(5, clCreateMLOpClipQCOM, __VA_ARGS__)                                  \
    default:                                                                         \
      TVM_FFI_THROW(InternalError) << "CLML Error: - Unsupported target version \n"; \
  }

// clSetMLTensorDimensionsQCOM
#define CLML_CALL_clSetMLTensorDimensionsQCOM(...)                                   \
  cl_int e;                                                                          \
  switch (CLMLWorkspace::Global()->target_major) {                                   \
    CALL_CASE(5, clSetMLTensorDimensionsQCOM, __VA_ARGS__)                           \
    default:                                                                         \
      TVM_FFI_THROW(InternalError) << "CLML Error: - Unsupported target version \n"; \
  }

// clUpdateMLOpQCOM
#define CLML_CALL_clUpdateMLOpQCOM(...)                                              \
  cl_int e;                                                                          \
  switch (CLMLWorkspace::Global()->target_major) {                                   \
    CALL_CASE(5, clUpdateMLOpQCOM, __VA_ARGS__)                                      \
    default:                                                                         \
      TVM_FFI_THROW(InternalError) << "CLML Error: - Unsupported target version \n"; \
  }

// clGetMLOpDeviceMemoryRequirementsQCOM
#define CLML_CALL_clGetMLOpDeviceMemoryRequirementsQCOM(...)                         \
  cl_int e;                                                                          \
  switch (CLMLWorkspace::Global()->target_major) {                                   \
    CALL_CASE(5, clGetMLOpDeviceMemoryRequirementsQCOM, __VA_ARGS__)                 \
    default:                                                                         \
      TVM_FFI_THROW(InternalError) << "CLML Error: - Unsupported target version \n"; \
  }

// clUpdateMLOpDeviceMemoryQCOM
#define CLML_CALL_clUpdateMLOpDeviceMemoryQCOM(...)                                  \
  cl_int e;                                                                          \
  switch (CLMLWorkspace::Global()->target_major) {                                   \
    CALL_CASE(5, clUpdateMLOpDeviceMemoryQCOM, __VA_ARGS__)                          \
    default:                                                                         \
      TVM_FFI_THROW(InternalError) << "CLML Error: - Unsupported target version \n"; \
  }

// clCreateMLTensorQCOM and clCreateMLTensorWithUsageQCOM
#define CALL_clCreateMLTensorQCOM(VERSION, CONTEXT, TENSORPROPS, TENSORDESC, USAGE, TENSOR) \
  CALL_CASE(VERSION, clCreateMLTensorQCOM, CONTEXT, TENSORPROPS, TENSORDESC, TENSOR)

#define CALL_clCreateMLTensorWithUsageQCOM(VERSION, CONTEXT, TENSORPROPS, TENSORDESC, USAGE, \
                                           TENSOR)                                           \
  CALL_CASE(VERSION, clCreateMLTensorWithUsageQCOM, CONTEXT, TENSORPROPS, TENSORDESC, USAGE, TENSOR)

#define CLML_CALL_clCreateMLTensorQCOM(CONTEXT, TENSORPROPS, TENSORDESC, USAGE, TENSOR)     \
  cl_int e;                                                                                 \
  switch (CLMLWorkspace::Global()->target_major) {                                          \
    CALL_clCreateMLTensorQCOM(1, CONTEXT, TENSORPROPS, TENSORDESC, USAGE, TENSOR);          \
    CALL_clCreateMLTensorQCOM(2, CONTEXT, TENSORPROPS, TENSORDESC, USAGE, TENSOR);          \
    CALL_clCreateMLTensorQCOM(3, CONTEXT, TENSORPROPS, TENSORDESC, USAGE, TENSOR);          \
    CALL_clCreateMLTensorWithUsageQCOM(4, CONTEXT, TENSORPROPS, TENSORDESC, USAGE, TENSOR); \
    CALL_clCreateMLTensorWithUsageQCOM(5, CONTEXT, TENSORPROPS, TENSORDESC, USAGE, TENSOR); \
    default:                                                                                \
      TVM_FFI_THROW(InternalError) << "CLML Error: - Unsupported target version \n";        \
  }

/* Version compatibility for CLML Tensor creation */
#if CL_QCOM_ML_OPS_H_MAJOR_VERSION < 4
typedef enum _cl_ml_tensor_usage_qcom {
  CL_TENSOR_USAGE_INVALID_QCOM = 0,
  CL_TENSOR_USAGE_UNUSED_QCOM = 1,
  CL_TENSOR_USAGE_PARAMETER_QCOM = 2,
  CL_TENSOR_USAGE_CNN_QCOM = 3,
  CL_TENSOR_USAGE_TNN_QCOM = 4,
} cl_ml_tensor_usage_qcom;
#endif

/*! \brief Magic number for CLML Tuning cache entry */
static const uint64_t kTVMCLMLTuningCacheMagic = 0x434C4D4C54554E45;

#define DEBUG_MEMORY_ALLOC false
#define DEBUG_STATS false
#define DEBUG_CLML false
#define LOG_MEM LOG_IF(WARNING, DEBUG_MEMORY_ALLOC)
#define LOG_STATS LOG_IF(WARNING, DEBUG_STATS)
#define LOG_CLML LOG_IF(WARNING, DEBUG_CLML)

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime::json;
using JSONGraphNode = tvm::runtime::json::JSONGraphNode;

class CLMLThreadEntry;

/*!
 * \brief CLML workspace.
 */
class CLMLWorkspace {
 public:
  /* Constructor */
  CLMLWorkspace();
  /*!
   * \brief Get the thread local ThreadEntry
   */
  virtual CLMLThreadEntry* GetThreadEntry();

  /* CLML Context */
  void* h_ClmlIntf = nullptr;
  cl::OpenCLWorkspace* workspace = nullptr;
  cl::OpenCLThreadEntry* tentry = nullptr;
  cl_device_id device_id;
  cl_platform_id platform_id;

  /* Tuning Support */
  bool is_tuning_run;
  char* tuning_file;

  /* Recordable Queues */
  bool is_recordable_queue = false;

  /* On chip memory support */
  bool is_on_chip_memory = false;

  /* On chip memory size */
  size_t onchip_mem_size = 0;

  /* get the global workspace */
  static CLMLWorkspace* Global();

  bool ExtensionStringPresent(std::string extn);

  /* DDR memory management */
  std::map<cl_mem, std::pair<int, int>> ddr_global_pool;  // buf, size and ref count

  /* Device API version information */
  int target_major;
  int target_minor;
};

/*! \brief Thread local workspace */
class CLMLThreadEntry {
 public:
  /* get the global workspace */
  static CLMLThreadEntry* ThreadLocal();
};

/*!
 * \brief Node descriptor to hold various information related to a Node.
 */
struct NodeDescriptor {
  std::shared_ptr<cl_ml_tensor_memory_desc_qcom> tensor_desc = nullptr;
  JSONGraphNode node;
  // Check the flag and them pick the layout.
  bool custom_layout = false;
  cl_ml_tensor_layout_qcom layout;
  cl_channel_type dtype = CL_FLOAT;
  cl_ml_tensor_usage_qcom usage = CL_TENSOR_USAGE_INVALID_QCOM;
  std::vector<cl_ml_tensor_properties_qcom> tensorProps = {};
  bool is_dynamic_tensor = false;
};

/*!
 * \brief CLML Function descriptor to hold various information related to a Function.
 */
struct CLMLFunctionDesc {
  /* Created CLML operation handle for particular Op in graph */
  cl_ml_op_qcom op;
  /* layer names of operator in subgraph */
  std::string layer_name;

#if (CL_QCOM_ML_OPS_H_MAJOR_VERSION >= 5)
  std::vector<cl_update_ml_op_properties_qcom> op_props = {};
#else
  std::vector<intptr_t> op_props = {};
#endif
};

/*!
 * \brief CLML objects we cache in order to avoid needing to construct
 * a new layer each time.
 */
struct CachedLayer {
  /* List of all created CLML operation handles in graph */
  std::vector<CLMLFunctionDesc> function;
  /* Map of function and original JsonNode */
  std::map<cl_ml_op_qcom, std::pair<int, JSONGraphNode>> op_node_map;
  /* The input tensor map  */
  std::map<int, std::shared_ptr<cl_ml_tensor_memory_desc_qcom>> inputs;
  /* A place holder Tensor representing TVM Tensor as CLML Tensor */
  std::map<int, std::shared_ptr<cl_ml_tensor_memory_desc_qcom>> in_placeholder;
  /* The Output tensor map */
  std::vector<std::shared_ptr<cl_ml_tensor_memory_desc_qcom>> outputs;
  /* A place holder Tensor representing TVM Tensor as CLML Tensor */
  std::vector<std::shared_ptr<cl_ml_tensor_memory_desc_qcom>> out_placeholder;
  /* Tensor shape exception list while returning from CLML Subgraph */
  std::map<int, std::vector<size_t>> out_shapes;
  /* Map of nodeid and descriptors */
  std::map<int, struct NodeDescriptor> storage_map;
  /* Tensor memory descriptors list to set after backing memory allocation */
  std::vector<cl_ml_tensor_memory_desc_qcom> tensorMemDescs;
  /* Index map of tensor memory descriptors list */
  std::map<int, int> tensorMemDescs_indexmap;  // nid, index
  cl_ml_tensor_mem_desc_set_qcom descriptorSet = nullptr;
  /* A dummy CLML tensor used across various ops */
  cl_ml_tensor_qcom unusedTensor = nullptr;

  /* Graph level tuning cache */
  cl_ml_tuningcache_qcom tuning_cache = nullptr;

  /* Memory management */
  std::map<int, int> storage_ref_map;  // NodeId & ref. count
  /* Activation node id & life span (the layer after which we can free) */
  std::map<int, int> life_span;
  std::map<size_t, size_t> on_chip_pool_size;                   // Mem start & size
  std::map<size_t, int> on_chip_pool_alloc_info;                // Mem start & node_id
  std::map<int, std::pair<size_t, size_t>> on_chip_alloc_plan;  // Final Alloc Plan
  std::map<int, size_t> on_chip_reject;                         // On-Chip reject info
  bool alloc_ping_pong;                                         // Allocation stratagy
  int in_chip_total_free;                                       // Total available
  int in_chip_total_alloc;                                      // Free memory
  int on_chip_alert_fail;                                       // Faliure due to fragmentation

  /* DDR memory planner */
  std::map<cl_mem, std::pair<int, bool>> ddr_storage_ref_map;  // local pool reference count
  std::map<int, cl_mem> ddr_alloc_plan;                        // allocation map <nid, cl_mem>

  cl_command_queue recordable_queue = nullptr;
  cl_recording_qcom recording = nullptr;
};

struct tensor_dims_t {
  uint32_t n, c, h, w;
};

#define CLML_QUEUE \
  CLMLWorkspace::Global()->workspace->GetQueue(CLMLWorkspace::Global()->tentry->device)
#define CLML_CTX CLMLWorkspace::Global()->workspace->contexts[CLMLWorkspace::Global()->platform_id]

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_GRAPH_EXECUTOR_CLML
#endif  // TVM_RUNTIME_CONTRIB_CLML_CLML_RUNTIME_H_
