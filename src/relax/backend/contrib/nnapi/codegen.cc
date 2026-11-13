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

#include <tvm/ffi/cast.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>
#include <tvm/runtime/logging.h>

#include <memory>
#include <string>
#include <vector>

#include "../../../transform/utils.h"
#include "../codegen_json/codegen_json.h"
#include "tvm/relax/op/manipulate.h"

namespace tvm {
namespace relax {
namespace contrib {

using JSONSerializer = backend::contrib::JSONSerializer;
using JSONGraphNode = backend::contrib::JSONGraphNode;
using JSONGraphNodeEntry = backend::contrib::JSONGraphNodeEntry;
using JSONGraphObjectPtr = backend::contrib::JSONGraphObjectPtr;
using NodeEntries = backend::contrib::NodeEntries;

class NNAPIJSONSerializer;

class CollectFromCompositeFunctionBody : public ExprVisitor {
 public:
  explicit CollectFromCompositeFunctionBody(NNAPIJSONSerializer* serializer)
      : serializer_(serializer), node_(std::make_shared<JSONGraphNode>()) {}

  void VisitExpr_(const CallObj* call_node) override;

  void SetPermuteDimsAttribute(const CallObj* call_node) {
    const auto* permute_dims_attr = call_node->attrs.as<PermuteDimsAttrs>();
    TVM_FFI_ICHECK(permute_dims_attr);
    if (permute_dims_attr->axes) {
      ffi::Array<int64_t> axes;
      for (auto axis : permute_dims_attr->axes.value()) {
        axes.push_back(axis);
      }
      node_->SetAttr("axes", std::move(axes));
    }
  }

  void SetAstypeAttribute(const CallObj* call_node) {
    const auto* astype_attrs = call_node->attrs.as<AstypeAttrs>();
    TVM_FFI_ICHECK(astype_attrs);
    node_->SetAttr("astype_dtype", ffi::String(ffi::DLDataTypeToString(astype_attrs->dtype)));
  }

  void SetMeanAttribute(const CallObj* call_node) {
    const auto* mean_attrs = call_node->attrs.as<StatisticalAttrs>();
    TVM_FFI_ICHECK(mean_attrs);
    TVM_FFI_ICHECK(mean_attrs->axis.has_value());

    {
      ffi::Array<int64_t> axis;
      for (auto dim : mean_attrs->axis.value()) {
        axis.push_back(dim);
      }
      node_->SetAttr("axis", std::move(axis));
    }
    node_->SetAttr("keepdims", static_cast<int64_t>(mean_attrs->keepdims ? 1 : 0));
  }

  void SetConv2dAttribute(const CallObj* call_node) {
    const auto* conv2d_attr = call_node->attrs.as<Conv2DAttrs>();
    TVM_FFI_ICHECK(conv2d_attr) << "didn't catch attributes";

    ffi::Array<int64_t> strides;
    if (!conv2d_attr->strides.empty()) {
      for (auto stride : conv2d_attr->strides) {
        strides.push_back(static_cast<int64_t>(stride));
      }
    } else {
      strides.push_back(1);
      strides.push_back(1);
    }

    ffi::Array<int64_t> padding;
    for (auto pad : conv2d_attr->padding) {
      padding.push_back(static_cast<int64_t>(pad));
    }

    node_->SetAttr("strides", std::move(strides));
    node_->SetAttr("padding", std::move(padding));
    node_->SetAttr("group", static_cast<int64_t>(conv2d_attr->groups));
  }

  void SetMaxPool2dAttribute(const CallObj* call_node) {
    const auto* max_pool_2d_attr = call_node->attrs.as<Pool2DAttrs>();
    TVM_FFI_ICHECK(max_pool_2d_attr) << "didn't catch attributes";

    ffi::Array<int64_t> strides;
    if (!max_pool_2d_attr->strides.empty()) {
      for (auto stride : max_pool_2d_attr->strides) {
        strides.push_back(static_cast<int64_t>(stride));
      }
    } else {
      strides.push_back(1);
      strides.push_back(1);
    }

    ffi::Array<int64_t> padding;
    for (auto pad : max_pool_2d_attr->padding) {
      padding.push_back(static_cast<int64_t>(pad));
    }

    ffi::Array<int64_t> pool_size;
    for (auto size : max_pool_2d_attr->pool_size) {
      pool_size.push_back(static_cast<int64_t>(size));
    }

    node_->SetAttr("strides", std::move(strides));
    node_->SetAttr("padding", std::move(padding));
    node_->SetAttr("pool_size", std::move(pool_size));
  }

  NNAPIJSONSerializer* serializer_;
  JSONGraphObjectPtr node_;
};

class NNAPIJSONSerializer : public JSONSerializer {
 public:
  explicit NNAPIJSONSerializer(ffi::Map<GenericConst, ffi::String> constant_names,
                               ffi::Map<Var, Expr> bindings)
      : JSONSerializer(constant_names), bindings_(bindings) {}
  using JSONSerializer::VisitExpr_;

  std::vector<JSONGraphNodeEntry> VisitExpr_(const CallObj* call_node) final {
    const auto* fn_var = call_node->op.as<VarObj>();
    TVM_FFI_ICHECK(fn_var);
    const auto fn = bindings_[ffi::GetRef<Var>(fn_var)].as_or_throw<Function>();
    TVM_FFI_ICHECK(fn.defined()) << "Expects the callee to be a function.";

    auto composite_opt = fn->GetAttr(tvm::relax::attr::kComposite);
    TVM_FFI_ICHECK(composite_opt.has_value()) << "Only composite functions are supported.";

    std::string composite_name = composite_opt.value();

    CollectFromCompositeFunctionBody collector(this);
    collector.VisitExpr(fn->body);

    NodeEntries inputs;
    for (const auto& arg : call_node->args) {
      auto res = VisitExpr(arg);
      inputs.insert(inputs.end(), res.begin(), res.end());
    }

    auto node = std::make_shared<JSONGraphNode>(composite_name, /* name_ */
                                                "kernel",       /* op_type_ */
                                                inputs, 1 /* num_outputs_ */);
    node->CaptureAttrs(*collector.node_);

    VLOG(1) << "Adding node " << composite_name << " with " << node->GetInputs().size()
            << " inputs";
    return AddObj(node, ffi::GetRef<Expr>(call_node));
  }

 private:
  ffi::Map<Var, Expr> bindings_;
};

void CollectFromCompositeFunctionBody::VisitExpr_(const CallObj* call_node) {
  const auto* op_node = call_node->op.as<OpObj>();
  TVM_FFI_ICHECK(op_node != nullptr);
  std::string name = op_node->name;
  if (name == "relax.permute_dims") {
    SetPermuteDimsAttribute(call_node);
  } else if (name == "relax.astype") {
    SetAstypeAttribute(call_node);
  } else if (name == "relax.mean") {
    SetMeanAttribute(call_node);
  } else if (name == "relax.nn.conv2d") {
    SetConv2dAttribute(call_node);
  } else if (name == "relax.nn.max_pool2d") {
    SetMaxPool2dAttribute(call_node);
  } else {
  }
  ExprVisitor::VisitExpr_(call_node);
}

ffi::Array<ffi::Module> NNAPICompiler(ffi::Array<Function> functions,
                                      ffi::Map<ffi::String, ffi::Any> /*unused*/,
                                      ffi::Map<GenericConst, ffi::String> constant_names) {
  VLOG(1) << "NNAPI Compiler";

  ffi::Array<ffi::Module> compiled_functions;
  for (const auto& func : functions) {
    NNAPIJSONSerializer serializer(constant_names, AnalyzeVar2Value(func));
    serializer.serialize(func);
    auto graph_json = serializer.GetJSON();
    auto constant_names = serializer.GetConstantNames();
    const auto pf = tvm::ffi::Function::GetGlobalRequired("runtime.nnapi_runtime_create");
    auto func_name = GetExtSymbol(func);
    auto result = pf(func_name, graph_json, constant_names);
    tvm::ffi::Module mod = result.cast<tvm::ffi::Module>();
    compiled_functions.push_back(mod);
  }

  return compiled_functions;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.ext.nnapi", NNAPICompiler);
}

}  // namespace contrib
}  // namespace relax
}  // namespace tvm
