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
 * \file src/relax/backend/contrib/tensorrt/codegen.cc
 * \brief Implementation of the TensorRT JSON serializer.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/ir/op.h>
#include <tvm/ir/transform.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/op/manipulate.h>
#include <tvm/relax/op/nn.h>
#include <tvm/relax/op/statistical.h>
#include <tvm/relax/type.h>
#include <tvm/relax/utils.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/index_map.h>

#include <memory>
#include <string>
#include <vector>

#include "../../../transform/utils.h"
#include "../codegen_json/codegen_json.h"
#include "../utils.h"

#if TVM_GRAPH_EXECUTOR_TENSORRT
#include "NvInfer.h"
#endif

namespace tvm {
namespace relax {
using namespace tvm::prim;

namespace contrib {

/*! \brief Attributes to store the compiler options for TensorRT. */
struct TensorRTCompilerConfigObj : public ffi::Object {
  ffi::Array<int64_t> tensorrt_version;
  bool use_implicit_batch;
  size_t max_workspace_size;
  bool remove_no_mac_subgraphs;
  bool use_fp16;
  bool use_uint8;
  bool build_at_compile_time;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TensorRTCompilerConfigObj>()
        .def_ro("tensorrt_version", &TensorRTCompilerConfigObj::tensorrt_version,
                "TensorRT version as (major, minor, patch).",
                refl::DefaultValue(ffi::Array<int64_t>({6, 0, 1})))
        .def_ro("use_implicit_batch", &TensorRTCompilerConfigObj::use_implicit_batch,
                "Use implicit batch (removed in TensorRT 10; networks are always explicit-batch)",
                refl::DefaultValue(false))
        .def_ro("max_workspace_size", &TensorRTCompilerConfigObj::max_workspace_size,
                "Max workspace size", refl::DefaultValue(size_t(1) << 30))
        .def_ro("remove_no_mac_subgraphs", &TensorRTCompilerConfigObj::remove_no_mac_subgraphs,
                "Remove no-mac subgraphs", refl::DefaultValue(false))
        .def_ro("use_fp16", &TensorRTCompilerConfigObj::use_fp16, "Use FP16",
                refl::DefaultValue(false))
        .def_ro("use_uint8", &TensorRTCompilerConfigObj::use_uint8, "Use uint8",
                refl::DefaultValue(false))
        .def_ro("build_at_compile_time", &TensorRTCompilerConfigObj::build_at_compile_time,
                "Build and embed TensorRT engines during code generation",
                refl::DefaultValue(false));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.ext.attrs.TensorRTCompilerConfig",
                                    TensorRTCompilerConfigObj, ffi::Object);
};

class TensorRTCompilerConfig : public ffi::ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TensorRTCompilerConfig, ffi::ObjectRef,
                                                TensorRTCompilerConfigObj);
};

TVM_FFI_STATIC_INIT_BLOCK() { TensorRTCompilerConfigObj::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  ::tvm::transform::PassContext::RegisterConfigOption<TensorRTCompilerConfig>(
      "relax.ext.tensorrt.options");
}

using JSONGraphNode = tvm::runtime::json::JSONGraphNode;
using JSONGraphNodeEntry = tvm::runtime::json::JSONGraphNodeEntry;
using JSONGraphObjectPtr = backend::contrib::JSONGraphObjectPtr;
using OpAttrExtractor = backend::contrib::OpAttrExtractor;
using JSONSerializer = backend::contrib::JSONSerializer;

/*!
 * \brief Collect the primitive operator call and its attributes from a "Composite" function.
 */
class CollectFromCompositeFunctionBody : public ExprVisitor {
 public:
  CollectFromCompositeFunctionBody() : node_(std::make_shared<JSONGraphNode>()) {}

  void VisitExpr_(const CallObj* call_node) final;

  void SetGenericAttributes(const CallObj* call_node) {
    OpAttrExtractor extractor(node_);
    const ffi::Object* attr_obj = call_node->attrs.get();
    extractor.Extract(const_cast<ffi::Object*>(attr_obj));
  }

  // Serialize an op's non-tensor arguments (scalars/shapes) as "arg_<name>" attributes; the "arg_"
  // prefix avoids JSONGraphNode's reserved "shape"/"dtype".
  void SetArgumentAttributes(const CallObj* call_node) {
    const auto* op_node = call_node->op.as<OpObj>();
    if (op_node == nullptr) return;
    const ffi::Array<ArgumentInfo>& arg_infos = op_node->args_info;
    for (size_t i = 0; i < call_node->args.size() && i < arg_infos.size(); ++i) {
      const Expr& arg = call_node->args[i];
      const std::string key = "arg_" + std::string(arg_infos[i]->name);
      if (auto prim_value = arg.as<PrimExpr>()) {
        PrimExpr value = prim_value.value();
        if (const auto* imm = value.as<IntImmObj>()) {
          node_->SetAttr(key, static_cast<int64_t>(imm->value));
        } else if (const auto* fimm = value.as<FloatImmObj>()) {
          node_->SetAttr(key, static_cast<double>(fimm->value));
        }
      } else if (const auto* shape_expr = arg.as<ShapeExprObj>()) {
        SetIntArrayAttr(key, shape_expr->values);
      }
    }
  }

  // Relax reduce axis is optional; materialize the all-axes default (it otherwise serializes as
  // "").
  void MaybeFillReduceAxes(const CallObj* call_node) {
    const auto* attrs = call_node->attrs.as<StatisticalAttrs>();
    if (attrs == nullptr || attrs->axis.has_value()) return;
    const auto* tensor_ty = GetType(call_node->args[0]).as<TensorTypeObj>();
    if (tensor_ty == nullptr || !tensor_ty->shape.has_value()) return;
    const auto* shape = tensor_ty->shape.value().as<ShapeExprObj>();
    if (shape == nullptr) return;
    ffi::Array<int64_t> all_axes;
    for (size_t i = 0; i < shape->values.size(); ++i) all_axes.push_back(static_cast<int64_t>(i));
    node_->SetAttr("axis", std::move(all_axes));
  }

  // strided_slice's axes/begin/end/strides are tuple args the op does not name; serialize by
  // position.
  void SetStridedSliceArguments(const CallObj* call_node) {
    const auto* op_node = call_node->op.as<OpObj>();
    if (op_node == nullptr || op_node->name != "relax.strided_slice") return;
    static constexpr const char* kNames[] = {"arg_axes", "arg_begin", "arg_end", "arg_strides"};
    for (size_t i = 1; i < call_node->args.size() && i <= 4; ++i) {
      const auto* tuple = call_node->args[i].as<TupleObj>();
      if (tuple == nullptr) continue;
      ffi::Array<PrimExpr> values;
      for (const Expr& field : tuple->fields) {
        if (auto prim_value = field.as<PrimExpr>()) {
          values.push_back(prim_value.value());
        }
      }
      if (values.size() == tuple->fields.size()) SetIntArrayAttr(kNames[i - 1], values);
    }
  }

  // Serialize static integer PrimExprs as an int64 array attribute (skips non-constant entries).
  void SetIntArrayAttr(const std::string& key, const ffi::Array<PrimExpr>& exprs) {
    ffi::Array<int64_t> values;
    for (const PrimExpr& expr : exprs) {
      const auto* imm = expr.as<IntImmObj>();
      if (imm == nullptr) return;
      values.push_back(static_cast<int64_t>(imm->value));
    }
    node_->SetAttr(key, std::move(values));
  }

  // layout_transform's IndexMap is not generically serializable; emit a pure permutation as
  // "arg_axes". Returns true for layout_transform (so generic extraction is skipped for it).
  bool TrySetLayoutTransformAttributes(const CallObj* call_node) {
    const auto* op_node = call_node->op.as<OpObj>();
    if (op_node == nullptr || op_node->name != "relax.layout_transform") return false;
    const auto* attrs = call_node->attrs.as<LayoutTransformAttrs>();
    if (attrs == nullptr) return true;
    auto index_map = attrs->index_map;
    const auto& initial = index_map->initial_indices;
    const auto& final_indices = index_map->final_indices;
    if (initial.size() != final_indices.size()) return true;
    ffi::Array<int64_t> permutation;
    for (const PrimExpr& expr : final_indices) {
      auto var = expr.as<PrimVar>();
      if (!var.has_value()) return true;
      int64_t pos = -1;
      for (size_t j = 0; j < initial.size(); ++j) {
        if (initial[j].get() == var.value().get()) {
          pos = static_cast<int64_t>(j);
          break;
        }
      }
      if (pos < 0) return true;
      permutation.push_back(pos);
    }
    node_->SetAttr("arg_axes", std::move(permutation));
    return true;
  }

  /*! \brief The primitive operator call in the composite function. */
  const CallObj* operator_call_{nullptr};
  /*!
   * \brief Temporary node into which we'll accumulate attributes. Ideally this would be the
   * final JSONGraphNode however we don't yet know how many inputs that will have.
   */
  JSONGraphObjectPtr node_;
};

/*!
 * \brief Generates an TensorRTModule from a relax expression by serializing the expression to a
 * json representation. TensorRT APIs are deferred until runtime unless compile-time engine
 * building is explicitly enabled.
 */
class TensorRTJSONSerializer : public JSONSerializer {
 public:
  explicit TensorRTJSONSerializer(ffi::Map<GenericConst, ffi::String> constant_names,
                                  ffi::Map<Var, Expr> bindings)
      : JSONSerializer(constant_names), bindings_(bindings) {}

  using JSONSerializer::VisitExpr_;

  std::vector<JSONGraphNodeEntry> VisitExpr_(const CallObj* call_node) final {
    // The call must be to an inline "Composite" function
    const auto* fn_var = call_node->op.as<VarObj>();
    TVM_FFI_ICHECK(fn_var);
    const auto fn = bindings_[ffi::GetRef<Var>(fn_var)].as_or_throw<Function>();

    auto opt_composite = fn->GetAttr(tvm::relax::attr::kComposite);
    TVM_FFI_ICHECK(opt_composite.has_value());
    std::string name = opt_composite.value();

    // TensorRT patterns describe a single primitive operator and use the Composite function only
    // to bind that operator's arguments and attributes.
    CollectFromCompositeFunctionBody collector;
    collector.VisitExpr(fn->body);
    TVM_FFI_ICHECK(collector.operator_call_ != nullptr)
        << "TensorRT Composite function " << name
        << " must contain exactly one primitive Relax operator call";

    // Bind Composite parameters back to the caller. The primitive operator's tensor arguments
    // are serialized below in their original order, regardless of whether they are constants or
    // parameters. Non-tensor scalar and shape arguments have already been captured as attributes.
    TVM_FFI_ICHECK_EQ(fn->params.size(), call_node->args.size());
    ffi::Map<Var, Expr> param_bindings;
    for (size_t i = 0; i < call_node->args.size(); ++i) {
      param_bindings.Set(fn->params[i], call_node->args[i]);
    }

    std::vector<JSONGraphNodeEntry> inputs;
    auto append_tensor_inputs = [&](const auto& self, const Expr& expr) -> void {
      if (const auto* tuple = expr.as<TupleObj>()) {
        for (const Expr& field : tuple->fields) self(self, field);
        return;
      }
      Type type = GetType(expr);
      if (type->IsInstance<TensorTypeObj>() || type->IsInstance<TupleTypeObj>()) {
        auto entries = VisitExpr(expr);
        inputs.insert(inputs.end(), entries.begin(), entries.end());
      }
    };
    for (const Expr& arg : collector.operator_call_->args) {
      append_tensor_inputs(append_tensor_inputs, Bind(arg, param_bindings));
    }

    // Create the final node.
    auto node = std::make_shared<JSONGraphNode>(name,
                                                /*op_type=*/"kernel", inputs,
                                                /*num_output=*/1);

    // Transfer attributes from the collector's node to the final node.
    node->CaptureAttrs(*collector.node_);

    // Capture global settings on the JSON node.
    SaveGlobalAttributes(node);

    VLOG(1) << name << " has " << node->GetInputs().size() << " inputs";

    return AddObj(node, ffi::GetRef<Expr>(call_node));
  }

  static void SaveGlobalAttributes(std::shared_ptr<JSONGraphNode> node) {
    auto ctx = transform::PassContext::Current();
    auto cfg = ctx->GetConfigOrDefault<TensorRTCompilerConfig>("relax.ext.tensorrt.options");
    TVM_FFI_ICHECK_EQ(cfg->tensorrt_version.size(), 3);
    ffi::Array<int64_t> tensorrt_version = {cfg->tensorrt_version[0], cfg->tensorrt_version[1],
                                            cfg->tensorrt_version[2]};
    node->SetAttr("tensorrt_version", std::move(tensorrt_version));
    node->SetAttr("use_implicit_batch", static_cast<int64_t>(cfg->use_implicit_batch));
    node->SetAttr("max_workspace_size", static_cast<int64_t>(cfg->max_workspace_size));
    node->SetAttr("use_fp16", static_cast<int64_t>(cfg->use_fp16));
    node->SetAttr("use_uint8", static_cast<int64_t>(cfg->use_uint8));
  }

 private:
  /*! \brief The bindings to look up composite functions. */
  ffi::Map<Var, Expr> bindings_;
};

void CollectFromCompositeFunctionBody::VisitExpr_(const CallObj* call_node) {
  TVM_FFI_ICHECK(call_node->op->IsInstance<OpObj>())
      << "TensorRT Composite functions must contain exactly one primitive Relax operator call";
  TVM_FFI_ICHECK(operator_call_ == nullptr)
      << "TensorRT Composite functions must contain exactly one primitive Relax operator call";
  operator_call_ = call_node;
  if (!TrySetLayoutTransformAttributes(call_node)) {
    SetGenericAttributes(call_node);
    SetArgumentAttributes(call_node);
    SetStridedSliceArguments(call_node);
    MaybeFillReduceAxes(call_node);
  }
  ExprVisitor::VisitExpr_(call_node);
}

/*!
 * \brief Create runtime modules for TensorRT.
 * \param functions The extern functions to be compiled via TensorRT
 * \return Runtime modules.
 */
ffi::Array<ffi::Module> TensorRTCompiler(ffi::Array<Function> functions,
                                         ffi::Map<ffi::String, ffi::Any> /*unused*/,
                                         ffi::Map<GenericConst, ffi::String> constant_names) {
  auto cfg = transform::PassContext::Current()->GetConfig<TensorRTCompilerConfig>(
      "relax.ext.tensorrt.options");
  bool build_at_compile_time = cfg.has_value() && cfg.value()->build_at_compile_time;
  ffi::Map<ffi::String, ffi::Tensor> constant_tensors;
  if (build_at_compile_time) {
    for (const auto& entry : constant_names) {
      constant_tensors.Set(entry.second, entry.first->value.cast<ffi::Tensor>());
    }
  }

  ffi::Array<ffi::Module> compiled_functions;
  for (const auto& func : functions) {
    VLOG(1) << "TensorRT partition:" << std::endl << func;
    if (build_at_compile_time) {
      // Reject dynamic inputs before the JSON serializer requires integer shapes.
      auto check_input_type = [&](const auto& self, const Type& type) -> void {
        if (const auto* tuple = type.as<TupleTypeObj>()) {
          for (const auto& field : tuple->fields) self(self, field);
          return;
        }
        const auto* tensor = type.as<TensorTypeObj>();
        TVM_FFI_CHECK(tensor != nullptr && tensor->shape.has_value(), ValueError)
            << "TensorRT compile-time engine building requires static positive input dimensions";
        const auto* shape = tensor->shape.value().as<ShapeExprObj>();
        TVM_FFI_CHECK(shape != nullptr, ValueError)
            << "TensorRT compile-time engine building requires static positive input dimensions";
        for (const auto& dim : shape->values) {
          const auto* value = dim.as<IntImmObj>();
          TVM_FFI_CHECK(value != nullptr && value->value > 0, ValueError)
              << "TensorRT compile-time engine building requires static positive input dimensions";
        }
      };
      for (const auto& param : func->params) check_input_type(check_input_type, GetType(param));
    }
    TensorRTJSONSerializer serializer(constant_names, AnalyzeVar2Value(func));
    serializer.serialize(func);
    std::string graph_json = serializer.GetJSON();
    VLOG(1) << "TensorRT JSON:" << std::endl << graph_json;
    auto ordered_constant_names = serializer.GetConstantNames();
    const auto pf = tvm::ffi::Function::GetGlobalRequired("runtime.tensorrt_runtime_create");
    std::string func_name = GetExtSymbol(func);
    VLOG(1) << "Creating tensorrt ffi::Module for '" << func_name << "'";
    auto module = pf(func_name, graph_json, ordered_constant_names).cast<ffi::Module>();
    if (build_at_compile_time) {
      auto build_engine = module->GetFunction("build_engine");
      TVM_FFI_CHECK(build_engine.has_value(), RuntimeError)
          << "TensorRT compile-time engine building requires the TensorRT runtime";
      ffi::Array<ffi::Tensor> ordered_constants;
      for (const auto& name : ordered_constant_names) {
        ordered_constants.push_back(constant_tensors[name]);
      }
      build_engine.value()(ordered_constants);
    }
    compiled_functions.push_back(module);
  }
  return compiled_functions;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.ext.tensorrt", TensorRTCompiler);
}

/*!
 * \brief Check whether TensorRT graph executor is enabled.
 * \return True if enabled, False if not.
 */
inline constexpr bool IsTensorRTRuntimeEnabled() {
#if TVM_GRAPH_EXECUTOR_TENSORRT
  return true;
#else
  return false;
#endif  // TVM_GRAPH_EXECUTOR_TENSORRT
}

/*!
 * \brief Get TensorRT version that TVM is built against.
 * \return Array of three integers for major, minor, and patch, or empty array if TensorRT graph
 * runtime is not enabled.
 */
ffi::Array<int64_t> GetTensorRTVersion() {
#if TVM_GRAPH_EXECUTOR_TENSORRT
  return {NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH};
#else
  return {};
#endif  // TVM_GRAPH_EXECUTOR_TENSORRT
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.is_tensorrt_runtime_enabled", IsTensorRTRuntimeEnabled)
      .def("relax.get_tensorrt_version", GetTensorRTVersion);
}

}  // namespace contrib
}  // namespace relax
}  // namespace tvm
