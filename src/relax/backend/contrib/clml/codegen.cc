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
 * \file src/relax/backend/contrib/clml/codegen.cc
 * \brief Implementation of the OpenCLML JSON serializer.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/ir/transform.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>
#include <tvm/runtime/logging.h>

#include <memory>
#include <string>
#include <vector>

#include "../../../transform/utils.h"
#include "../codegen_json/codegen_json.h"
#include "../utils.h"

namespace tvm {
namespace relax {
namespace contrib {

/*! \brief Attributes to store the compiler options for OpenCLML. */
struct OpenCLMLCompilerConfigObj : public ffi::Object {
  IntImm clml_version = IntImm::Int32(3);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<OpenCLMLCompilerConfigObj>().def_ro(
        "clml_version", &OpenCLMLCompilerConfigObj::clml_version,
        "OpenCLML version as (major, minor, patch).", refl::DefaultValue(IntImm::Int32(3)));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.ext.attrs.OpenCLMLCompilerConfig",
                                    OpenCLMLCompilerConfigObj, ffi::Object);
};

class OpenCLMLCompilerConfig : public ffi::ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(OpenCLMLCompilerConfig, ffi::ObjectRef,
                                                OpenCLMLCompilerConfigObj);
};

TVM_FFI_STATIC_INIT_BLOCK() { OpenCLMLCompilerConfigObj::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  ::tvm::transform::PassContext::RegisterConfigOption<OpenCLMLCompilerConfig>(
      "relax.ext.clml.options");
}

using JSONGraphNode = tvm::runtime::json::JSONGraphNode;
using JSONGraphNodeEntry = tvm::runtime::json::JSONGraphNodeEntry;
using JSONGraphObjectPtr = backend::contrib::JSONGraphObjectPtr;
using OpAttrExtractor = backend::contrib::OpAttrExtractor;
using JSONSerializer = backend::contrib::JSONSerializer;

class OpenCLMLJSONSerializer;

/*!
 * \brief Collect the constants and attributes from all operator calls in the body
 * of a "Composite" function.
 */
class CollectCLMLFromCompositeFunctionBody : public ExprVisitor {
 public:
  explicit CollectCLMLFromCompositeFunctionBody(OpenCLMLJSONSerializer* serializer)
      : serializer_(serializer), node_(std::make_shared<JSONGraphNode>()) {}

  void VisitExpr_(const GenericConstObj* constant_node) final;
  void VisitExpr_(const CallObj* call_node) final;

  void SetGenericAttributes(const CallObj* call_node) {
    if (backend::IsOp(call_node, "relax.nn.relu")) {
      node_->SetAttr("activation_type", ffi::String("relu"));
    }

    OpAttrExtractor extractor(node_);
    const ffi::Object* attr_obj = call_node->attrs.get();
    extractor.Extract(const_cast<ffi::Object*>(attr_obj));
  }

  OpenCLMLJSONSerializer* serializer_;
  /*! \brief Accumulated translated arguments. */
  std::vector<JSONGraphNodeEntry> args_;
  /*!
   * \brief Temporary node into which we'll accumulate attributes. Ideally this would be the
   * final JSONGraphNode however we don't yet know how many inputs that will have.
   */
  JSONGraphObjectPtr node_;
};

/*!
 * \brief Generates an OpenCLMLModule from a relax expression by serializing the expression to a
 * json representation. OpenCLML is not required here because use of OpenCLML APIs is deferred until
 * runtime.
 */
class OpenCLMLJSONSerializer : public JSONSerializer {
 public:
  explicit OpenCLMLJSONSerializer(ffi::Map<GenericConst, ffi::String> constant_names,
                                  ffi::Map<Var, Expr> bindings)
      : JSONSerializer(constant_names), bindings_(bindings) {}

  /*!
   * \brief A series of operators that form a composite
   * convolution. Supports nn.conv2d
   */
  struct CompositeConvNode {
    const CallObj* pad = nullptr;
    const CallObj* conv = nullptr;
    const CallObj* bn = nullptr;
    const CallObj* bias = nullptr;
    const CallObj* activation = nullptr;
    std::string act_type;
  };

  using JSONSerializer::VisitExpr_;

  std::vector<JSONGraphNodeEntry> VisitExpr_(const CallObj* call_node) final {
    // The call must be to an inline "Composite" function
    const auto* fn_var = call_node->op.as<VarObj>();
    TVM_FFI_ICHECK(fn_var);
    const auto fn = bindings_[ffi::GetRef<Var>(fn_var)].as_or_throw<Function>();

    auto opt_composite = fn->GetAttr(tvm::relax::attr::kComposite);
    TVM_FFI_ICHECK(opt_composite.has_value());
    std::string name = opt_composite.value();

    std::shared_ptr<JSONGraphNode> node;

    if (backend::EndsWithPattern(name, "nn.conv2d") ||
        backend::EndsWithPattern(name, "nn.pad_conv2d") ||
        backend::EndsWithPattern(name, "nn.pad_conv2d_transpose")) {
      node = CreateCompositeConvJSONNode(call_node);
    } else {
      // Collect the constants and attributes of all operator calls inside the composite body.
      CollectCLMLFromCompositeFunctionBody collector(this);
      collector.VisitExpr(fn->body);

      // Capture the args to the "Composite" function as inputs for this node.
      std::vector<JSONGraphNodeEntry> inputs;
      for (const auto& arg : call_node->args) {
        auto res = VisitExpr(arg);
        inputs.insert(inputs.end(), res.begin(), res.end());
      }
      // Capture constants from the composite function body as additional inputs for this node.
      for (const auto& node : collector.args_) {
        inputs.emplace_back(node);
      }

      // Create the final node.
      node = std::make_shared<JSONGraphNode>(name,
                                             /*op_type=*/"kernel", inputs,
                                             /*num_output=*/1);

      // Transfer attributes from the collector's node to the final node.
      node->CaptureAttrs(*collector.node_);

      VLOG(1) << name << " has " << node->GetInputs().size() << " inputs";
    }

    return AddObj(node, ffi::GetRef<Expr>(call_node));
  }

  /*!
   * \brief Extract convolution nodes from a composite function.
   *
   * \param cn The call node of the composite function.
   * \return Extracted composite convolution nodes.
   */
  CompositeConvNode UnpackCompositeConvolution(const CallObj* cn) {
    CompositeConvNode nodes{};

    const auto* fn_var = cn->op.as<VarObj>();
    TVM_FFI_ICHECK(fn_var);
    const auto fn = bindings_[ffi::GetRef<Var>(fn_var)].as_or_throw<Function>();
    auto opt_composite = fn->GetAttr(tvm::relax::attr::kComposite);
    TVM_FFI_ICHECK(opt_composite.has_value());

    nodes.pad = backend::TryGetOpInFunction(fn, "relax.nn.pad");
    nodes.conv = backend::TryGetOpInFunction(fn, "relax.nn.conv2d");

    if (!nodes.conv) {
      nodes.conv = backend::TryGetOpInFunction(fn, "relax.nn.conv2d_transpose");
    }
    TVM_FFI_ICHECK(nodes.conv) << "No Convolution op found in composite function";
    nodes.bn = backend::TryGetOpInFunction(fn, "relax.nn.batch_norm");
    nodes.bias = backend::TryGetOpInFunction(fn, "relax.add");
    nodes.activation = backend::TryGetOpInFunction(fn, "relax.nn.relu");
    nodes.act_type = "relu";
    return nodes;
  }

  /*!
   * \brief Create a JSON representation of a composite convolution.
   *
   * \param cn The call to be represented.
   * \return A JSON representation of a specific operator.
   */
  std::shared_ptr<JSONGraphNode> CreateCompositeConvJSONNode(const CallObj* cn) {
    CompositeConvNode nodes = UnpackCompositeConvolution(cn);

    const auto* fn_var = cn->op.as<VarObj>();
    TVM_FFI_ICHECK(fn_var);
    const auto fn = bindings_[ffi::GetRef<Var>(fn_var)].as_or_throw<Function>();
    auto opt_composite = fn->GetAttr(tvm::relax::attr::kComposite);
    TVM_FFI_ICHECK(opt_composite.has_value());
    std::string name = opt_composite.value();

    std::vector<JSONGraphNodeEntry> inputs;

    inputs.push_back(VisitExpr(cn->args[0])[0]);
    inputs.push_back(VisitExpr(nodes.conv->args[1])[0]);
    if (nodes.bias) {
      inputs.push_back(VisitExpr(nodes.bias->args[1])[0]);
    }
    // Deal with Batchnorm Fusing here
    if (nodes.bn) {
      inputs.push_back(VisitExpr(nodes.bn->args[1])[0]);
      inputs.push_back(VisitExpr(nodes.bn->args[2])[0]);
      inputs.push_back(VisitExpr(nodes.bn->args[3])[0]);
      inputs.push_back(VisitExpr(nodes.bn->args[4])[0]);
    }

    auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);
    SetCallNodeAttribute(json_node, nodes.conv);

    if (nodes.bn) {
      const auto* bn_attr = nodes.bn->attrs.as<BatchNormAttrs>();
      json_node->SetAttr(
          "batchnorm",
          ffi::Array<ffi::String>{std::to_string(bn_attr->axis), std::to_string(bn_attr->epsilon),
                                  std::to_string(bn_attr->center), std::to_string(bn_attr->scale)});
    }

    // Override attributes
    if (nodes.pad) {
      const auto* pad_attr = nodes.pad->attrs.as<PadAttrs>();
      TVM_FFI_ICHECK(pad_attr);
      auto p = pad_attr->pad_width;
      // Pad layout for TVM: dimension wise pre and post padding.
      // CLML takes dimension wise pre-padding followed by dimension wise post-padding for W, H.
      json_node->SetAttr("padding", ffi::Array<int64_t>{p[4], p[6], p[5], p[7]});
    }

    if (nodes.activation) {
      json_node->SetAttr("activation_type", ffi::String(nodes.act_type));
    }
    return json_node;
  }

  static void SaveGlobalAttributes(std::shared_ptr<JSONGraphNode> node) {
    auto ctx = transform::PassContext::Current();
    auto cfg = ctx->GetConfigOrDefault<OpenCLMLCompilerConfig>("relax.ext.clml.options");
    node->SetAttr("clml_version", static_cast<int64_t>(cfg->clml_version->value));
  }

 private:
  /*! \brief The bindings to look up composite functions. */
  ffi::Map<Var, Expr> bindings_;
};

void CollectCLMLFromCompositeFunctionBody::VisitExpr_(const GenericConstObj* constant_node) {
  for (const auto& entry : serializer_->VisitExpr(ffi::GetRef<GenericConst>(constant_node))) {
    args_.emplace_back(entry);
  }
}

void CollectCLMLFromCompositeFunctionBody::VisitExpr_(const CallObj* call_node) {
  SetGenericAttributes(call_node);
  ExprVisitor::VisitExpr_(call_node);
}

/*!
 * \brief Create runtime modules for OpenCLML.
 * \param functions The extern functions to be compiled via OpenCLML
 * \return Runtime modules.
 */
ffi::Array<ffi::Module> OpenCLMLCompiler(ffi::Array<Function> functions,
                                         ffi::Map<ffi::String, Any> /*unused*/,
                                         ffi::Map<GenericConst, ffi::String> constant_names) {
  ffi::Array<ffi::Module> compiled_functions;
  for (const auto& func : functions) {
    VLOG(1) << "OpenCLML partition:" << std::endl << func;
    OpenCLMLJSONSerializer serializer(constant_names, AnalyzeVar2Value(func));
    serializer.serialize(func);
    std::string graph_json = serializer.GetJSON();
    auto constant_names = serializer.GetConstantNames();
    const auto pf = tvm::ffi::Function::GetGlobalRequired("runtime.clml_runtime_create");
    std::string func_name = GetExtSymbol(func);
    VLOG(1) << "Creating clml ffi::Module for '" << func_name << "'";
    compiled_functions.push_back(pf(func_name, graph_json, constant_names).cast<ffi::Module>());
  }
  return compiled_functions;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.ext.openclml", OpenCLMLCompiler);
}

/*!
 * \brief Check whether OpenCLML graph executor is enabled.
 * \return True if enabled, False if not.
 */
inline constexpr bool IsOpenCLMLRuntimeEnabled() {
#if TVM_GRAPH_EXECUTOR_CLML
  return true;
#else
  return false;
#endif  // TVM_GRAPH_EXECUTOR_CLML
}

/*!
 * \brief Get OpenCLML version that TVM is built against.
 * \return The OpenCLML SDK version.
 */
IntImm GetOpenCLMLVersion() {
#if TVM_GRAPH_EXECUTOR_CLML
  return IntImm::Int32(TVM_CLML_VERSION);
#else
  return IntImm::Int32(3);
#endif  // TVM_GRAPH_EXECUTOR_CLML
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.is_openclml_runtime_enabled", IsOpenCLMLRuntimeEnabled)
      .def("relax.get_openclml_version", GetOpenCLMLVersion);
}

}  // namespace contrib
}  // namespace relax
}  // namespace tvm
