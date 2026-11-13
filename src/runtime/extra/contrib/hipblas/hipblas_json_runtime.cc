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
 * \file src/runtime/contrib/hipblas/hipblas_json_runtime.cc
 * \brief A simple JSON runtime for HIPBLAS.
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/tensor.h>

#include <cstddef>
#include <string>
#include <vector>

#include "../../rocm_utils.h"
#include "../json/json_node.h"
#include "../json/json_runtime.h"
#include "hipblas_utils.h"

namespace tvm {
namespace runtime {
namespace contrib {
using namespace tvm::runtime;
using namespace tvm::runtime::json;
class HipblasJSONRuntime : public JSONRuntimeBase {
 public:
  HipblasJSONRuntime(const std::string& symbol_name, const std::string& graph_json,
                     const ffi::Array<ffi::String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  void Init(const ffi::Array<ffi::Tensor>& consts) override {}

  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) override {
    // JSONRuntimeBase::SetInputOutputBuffers(...) is not thread safe. Since HipblasJSONRuntime
    // can be used by multiple GPUs running on different threads, we avoid using that function
    // and directly call hipBLAS on the inputs from ffi::PackedArgs.
    ffi::ObjectPtr<ffi::Object> sptr_to_self = ffi::GetObjectPtr<ffi::Object>(this);
    if (this->symbol_name_ == name) {
      return ffi::Function([sptr_to_self, this](ffi::PackedArgs args, ffi::Any* rv) {
        TVM_FFI_ICHECK(this->initialized_) << "The module has not been initialized";
        this->Run(args);
      });
    } else {
      return JSONRuntimeBase::GetFunction(name);
    }
  }

  const char* kind() const override { return "hipblas_json"; }  // May be overridden

  void Run(ffi::PackedArgs args) {
    int device_id = -1;
    std::vector<const DLTensor*> dl_tensors(NumEntries());

    for (size_t i = 0; i < static_cast<size_t>(args.size()); i++) {
      auto eid = i < input_var_eid_.size() ? input_var_eid_[i]
                                           : EntryID(outputs_[i - input_var_eid_.size()]);

      const DLTensor* arg;
      if (auto opt_nd = args[i].as<ffi::Tensor>(); opt_nd && opt_nd->defined()) {
        ffi::Tensor arr = opt_nd.value();
        arg = arr.GetDLTensorPtr();
      } else {
        arg = args[i].cast<DLTensor*>();
      }

      dl_tensors[eid] = arg;
      device_id = arg->device.device_id;
    }
    if (device_id == -1) {
      ROCM_CALL(hipGetDevice(&device_id));
    }
    auto* entry_ptr = tvm::contrib::HipBlasLtThreadEntry::ThreadLocal(DLDevice{kDLROCM, device_id});
    hipStream_t stream = static_cast<hipStream_t>(TVMFFIEnvGetStream(kDLROCM, device_id));

    auto get_input = [this, &dl_tensors](const JSONGraphNode& node, int idx) {
      TVM_FFI_ICHECK_LT(idx, node.GetInputs().size());
      auto eid = EntryID(node.GetInputs()[idx]);
      TVM_FFI_ICHECK(eid < dl_tensors.size());
      return dl_tensors[eid];
    };

    auto get_inputs = [=](const JSONGraphNode& node, bool has_bias) {
      const DLTensor* bias = nullptr;
      if (has_bias) {
        bias = get_input(node, 2);
      }
      return std::make_tuple(get_input(node, 0), get_input(node, 1), bias);
    };

    for (size_t i = 0; i < nodes_.size(); ++i) {
      const auto& node = nodes_[i];
      if (node.GetOpType() == "kernel") {
        auto op_name = node.GetOpName();
        uint32_t output_eid = EntryID(outputs_[0]);
        auto out_ptr = dl_tensors[output_eid];
        bool transa = false;
        bool transb = false;
        hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;

        if (op_name.find("transposed") != std::string::npos) {
          transb = true;
        }

        if (op_name.find("relu") != std::string::npos) {
          epilogue = HIPBLASLT_EPILOGUE_RELU_BIAS;
        } else if (op_name.find("gelu") != std::string::npos) {
          epilogue = HIPBLASLT_EPILOGUE_GELU_BIAS;
        } else if (op_name.find("bias") != std::string::npos) {
          epilogue = HIPBLASLT_EPILOGUE_BIAS;
        }

        auto [a_ptr, b_ptr, bias_ptr] = get_inputs(node, epilogue != HIPBLASLT_EPILOGUE_DEFAULT);

        tvm::contrib::CallHipblasLt(entry_ptr->handle, stream, entry_ptr->matmul_pref_desc, a_ptr,
                                    b_ptr, bias_ptr, out_ptr, transa, transb,
                                    entry_ptr->workspace_ptr, entry_ptr->workspace_size, epilogue);
      }
    }
  }

  void Run() override { TVM_FFI_THROW(InternalError) << "Unreachable"; }
};

ffi::Module HipblasJSONRuntimeCreate(ffi::String symbol_name, ffi::String graph_json,
                                     const ffi::Array<ffi::String>& const_names) {
  auto n = ffi::make_object<HipblasJSONRuntime>(symbol_name, graph_json, const_names);
  return ffi::Module(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.HipblasJSONRuntimeCreate", HipblasJSONRuntimeCreate)
      .def("ffi.Module.load_from_bytes.hipblas_json",
           JSONRuntimeBase::LoadFromBytes<HipblasJSONRuntime>);
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
