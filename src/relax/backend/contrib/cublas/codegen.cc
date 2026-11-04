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
 * \file src/relax/backend/contrib/cublas/codegen.cc
 * \brief Implementation of the CUBLAS JSON serializer.
 */
#include <builtin_fp16.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/relax/expr.h>

#include <cstring>
#include <string>

#include "../codegen_json/codegen_json.h"
#include "../utils.h"

namespace tvm {
namespace relax {
namespace contrib {

using JSONGraphNode = tvm::runtime::json::JSONGraphNode;
using JSONGraphNodeEntry = tvm::runtime::json::JSONGraphNodeEntry;
using JSONSerializer = backend::contrib::JSONSerializer;
using backend::contrib::NodeEntries;

class CublasJSONSerializer : public JSONSerializer {
 public:
  CublasJSONSerializer(ffi::Map<GenericConst, ffi::String> constant_names,
                       ffi::Map<Var, Expr> bindings)
      : JSONSerializer(constant_names), bindings_(bindings) {}

  using JSONSerializer::VisitExpr_;

  NodeEntries VisitExpr_(const CallObj* call_node) final {
    const auto* fn_var = call_node->op.as<VarObj>();
    TVM_FFI_ICHECK(fn_var);
    const auto fn = bindings_[ffi::GetRef<Var>(fn_var)].as_or_throw<Function>();
    TVM_FFI_ICHECK(fn.defined()) << "Expects the callee to be a function.";

    auto composite_opt = fn->GetAttr(tvm::relax::attr::kComposite);
    TVM_FFI_ICHECK(composite_opt.has_value()) << "Only composite functions are supported.";

    std::string composite_name = composite_opt.value();

    NodeEntries inputs_tmp;
    for (const auto& arg : call_node->args) {
      auto res = VisitExpr(arg);
      inputs_tmp.insert(inputs_tmp.end(), res.begin(), res.end());
    }

    TVM_FFI_ICHECK(inputs_tmp.size() <= 4);
    NodeEntries inputs(inputs_tmp.size());

    auto arg_idx = backend::ExtractArgIdx(composite_name, fn);
    inputs[0] = inputs_tmp[static_cast<int64_t>(arg_idx["lhs"]->value)];
    inputs[1] = inputs_tmp[static_cast<int64_t>(arg_idx["rhs"]->value)];
    if (inputs_tmp.size() == 3) {
      inputs[2] = inputs_tmp[static_cast<int64_t>(arg_idx["bias"]->value)];
    } else if (inputs_tmp.size() == 4) {
      inputs[2] = inputs_tmp[static_cast<int64_t>(arg_idx["scaleA"]->value)];
      inputs[3] = inputs_tmp[static_cast<int64_t>(arg_idx["scaleB"]->value)];
    }

    auto node = std::make_shared<JSONGraphNode>(composite_name, /* name_ */
                                                "kernel",       /* op_type_ */
                                                inputs, 1 /* num_outputs_ */);
    if (composite_name.find("dequantize") != std::string::npos) {
      const CallObj* dequantize_call = backend::GetOpInFunction(fn, "relax.dequantize");
      if (dequantize_call->args[1]->IsInstance<GenericConstObj>()) {
        const auto* const_expr = dequantize_call->args[1].as<GenericConstObj>();
        auto ty = const_expr->ty.as_or_throw<TensorType>();
        ffi::Tensor scale = const_expr->value.cast<ffi::Tensor>();
        TVM_FFI_ICHECK(scale.defined());
        TVM_FFI_ICHECK_EQ(scale.device().device_type, kDLCPU);
        TVM_FFI_ICHECK_NE(scale.data_ptr(), nullptr);
        const char* scale_data = static_cast<const char*>(scale.data_ptr()) + scale.byte_offset();
        float alpha = 1.0;
        if (ty->dtype == PrimType::Float(16)) {
          TVM_FFI_ICHECK_GE(ffi::GetDataSize(*scale.GetDLTensorPtr()), sizeof(uint16_t));
          uint16_t value;
          std::memcpy(&value, scale_data, sizeof(value));
          alpha = __extendXfYf2__<uint16_t, uint16_t, 10, float, uint32_t, 23>(value);
        } else {
          TVM_FFI_ICHECK(ty->dtype == PrimType::Float(32));
          TVM_FFI_ICHECK_GE(ffi::GetDataSize(*scale.GetDLTensorPtr()), sizeof(alpha));
          std::memcpy(&alpha, scale_data, sizeof(alpha));
        }

        node->SetAttr("dq_scale", static_cast<double>(alpha));
      }
    }

    const CallObj* root_call = backend::GetOpInFunction(fn, "relax.matmul");
    SetCallNodeAttribute(node, root_call);
    return AddObj(node, ffi::GetRef<Expr>(call_node));
  }

 private:
  /*! \brief The bindings to look up composite functions. */
  ffi::Map<Var, Expr> bindings_;
};

ffi::Array<ffi::Module> CublasCompiler(ffi::Array<Function> functions,
                                       ffi::Map<ffi::String, ffi::Any> /*unused*/,
                                       ffi::Map<GenericConst, ffi::String> constant_names) {
  ffi::Array<ffi::Module> compiled_functions;

  for (const auto& func : functions) {
    CublasJSONSerializer serializer(constant_names, AnalyzeVar2Value(func));
    serializer.serialize(func);
    auto graph_json = serializer.GetJSON();
    auto constant_names = serializer.GetConstantNames();
    const auto pf = tvm::ffi::Function::GetGlobalRequired("runtime.CublasJSONRuntimeCreate");
    auto func_name = GetExtSymbol(func);
    compiled_functions.push_back(pf(func_name, graph_json, constant_names).cast<ffi::Module>());
  }

  return compiled_functions;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.ext.cublas", CublasCompiler);
}

}  // namespace contrib
}  // namespace relax
}  // namespace tvm
