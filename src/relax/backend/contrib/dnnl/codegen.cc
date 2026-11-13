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
 * \file src/relax/backend/contrib/dnnl/codegen.cc
 * \brief Implementation of the DNNL JSON serializer.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/relax/expr.h>

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

class DNNLJSONSerializer : public JSONSerializer {
 public:
  DNNLJSONSerializer(ffi::Map<GenericConst, ffi::String> constant_names,
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

    NodeEntries inputs;
    for (const auto& arg : call_node->args) {
      auto res = VisitExpr(arg);
      inputs.insert(inputs.end(), res.begin(), res.end());
    }
    auto node = std::make_shared<JSONGraphNode>(composite_name, /* name_ */
                                                "kernel",       /* op_type_ */
                                                inputs, 1 /* num_outputs_ */);

    const CallObj* root_call = nullptr;
    if (composite_name.find("conv2d") != std::string::npos) {
      root_call = backend::GetOpInFunction(fn, "relax.nn.conv2d");
    } else {
      TVM_FFI_THROW(InternalError) << "Unimplemented pattern: " << composite_name;
    }

    SetCallNodeAttribute(node, root_call);
    return AddObj(node, ffi::GetRef<Expr>(call_node));
  }

 private:
  /*! \brief The bindings to look up composite functions. */
  ffi::Map<Var, Expr> bindings_;
};

ffi::Array<ffi::Module> DNNLCompiler(ffi::Array<Function> functions,
                                     ffi::Map<ffi::String, ffi::Any> /*unused*/,
                                     ffi::Map<GenericConst, ffi::String> constant_names) {
  ffi::Array<ffi::Module> compiled_functions;

  for (const auto& func : functions) {
    DNNLJSONSerializer serializer(constant_names, AnalyzeVar2Value(func));
    serializer.serialize(func);
    auto graph_json = serializer.GetJSON();
    auto constant_names = serializer.GetConstantNames();
    const auto pf = tvm::ffi::Function::GetGlobalRequired("runtime.DNNLJSONRuntimeCreate");
    auto func_name = GetExtSymbol(func);
    compiled_functions.push_back(pf(func_name, graph_json, constant_names).cast<ffi::Module>());
  }

  return compiled_functions;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.ext.dnnl", DNNLCompiler);
}

}  // namespace contrib
}  // namespace relax
}  // namespace tvm
