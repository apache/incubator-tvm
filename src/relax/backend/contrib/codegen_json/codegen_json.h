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
 * \file relax/backend/contrib/codegen_json/codegen_json.h
 * \brief Utilities for json codegen and runtime
 */
#ifndef TVM_RELAX_BACKEND_CONTRIB_CODEGEN_JSON_CODEGEN_JSON_H_
#define TVM_RELAX_BACKEND_CONTRIB_CODEGEN_JSON_CODEGEN_JSON_H_

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../../../../runtime/extra/contrib/json/json_node.h"
#include "../../../../runtime/extra/contrib/json/json_runtime.h"
#include "../../../transform/utils.h"
#include "../utils.h"

namespace tvm {
namespace relax {
using namespace tvm::prim;

namespace backend {
namespace contrib {

using namespace tvm::runtime::json;

using ShapeVector = std::vector<std::vector<int64_t>>;
using TypeVector = std::vector<std::string>;
using JSONGraphObjectPtr = std::shared_ptr<JSONGraphNode>;

/*!
 * \brief Helper class to extract all attributes of a certain op and save them
 * into text format.
 */
class OpAttrExtractor {
 public:
  explicit OpAttrExtractor(JSONGraphObjectPtr node) : node_(node) {}

  void SetNodeAttr(const char* key, ffi::Any value) { node_->SetAttr(key, std::move(value)); }

  void Visit(const char* key, double* value) { SetNodeAttr(key, *value); }

  void Visit(const char* key, int64_t* value) { SetNodeAttr(key, *value); }

  void Visit(const char* key, uint64_t* value) { SetNodeAttr(key, static_cast<int64_t>(*value)); }

  void Visit(const char* key, int* value) { SetNodeAttr(key, static_cast<int64_t>(*value)); }

  void Visit(const char* key, bool* value) { SetNodeAttr(key, static_cast<int64_t>(*value)); }

  void Visit(const char* key, std::string* value) { SetNodeAttr(key, ffi::String(*value)); }

  void Visit(const char* key, ffi::Optional<double>* value) {
    if (value->has_value()) {
      SetNodeAttr(key, value->value());
    } else {
      SetNodeAttr(key, ffi::String(""));
    }
  }

  void Visit(const char* key, ffi::Optional<int64_t>* value) {
    if (value->has_value()) {
      SetNodeAttr(key, value->value());
    } else {
      SetNodeAttr(key, ffi::String(""));
    }
  }

  void Visit(const char* key, DLDataType* value) {
    if (!(value->code == kDLOpaqueHandle && value->bits == 0 && value->lanes == 0)) {
      SetNodeAttr(key, ffi::String(ffi::DLDataTypeToString(*value)));
    } else {
      SetNodeAttr(key, ffi::String(""));
    }
  }

  void Visit(const char* key, ffi::Any* value) {
    if (const auto* an = (*value).as<ffi::ArrayObj>()) {
      // Determine element type from first element and build typed array
      if (an->size() == 0) {
        SetNodeAttr(key, ffi::Array<int64_t>{});
        return;
      }
      // Classify elements: check if all are int, or all are numeric (int or float)
      bool can_be_int = true;
      bool can_be_double = true;
      for (size_t i = 0; i < an->size(); ++i) {
        const auto& elem = (*an)[i];
        bool is_int = elem.try_cast<int64_t>() || elem.as<IntImmObj>();
        bool is_float = elem.try_cast<double>() || elem.as<FloatImmObj>();
        if (!is_int) {
          can_be_int = false;
        }
        if (!is_int && !is_float) {
          can_be_double = false;
          break;
        }
      }
      if (can_be_int) {
        ffi::Array<int64_t> attr;
        for (size_t i = 0; i < an->size(); ++i) {
          if (auto opt_int = (*an)[i].try_cast<int64_t>()) {
            attr.push_back(opt_int.value());
          } else if (const auto* im = (*an)[i].as<IntImmObj>()) {
            attr.push_back(static_cast<int64_t>(im->value));
          }
        }
        SetNodeAttr(key, std::move(attr));
      } else if (can_be_double) {
        // Mixed int/float array: promote all to double
        ffi::Array<double> attr;
        for (size_t i = 0; i < an->size(); ++i) {
          const auto& elem = (*an)[i];
          if (auto opt_float = elem.try_cast<double>()) {
            attr.push_back(opt_float.value());
          } else if (const auto* fm = elem.as<FloatImmObj>()) {
            attr.push_back(fm->value);
          } else if (auto opt_int = elem.try_cast<int64_t>()) {
            attr.push_back(static_cast<double>(opt_int.value()));
          } else if (const auto* im = elem.as<IntImmObj>()) {
            attr.push_back(static_cast<double>(im->value));
          }
        }
        SetNodeAttr(key, std::move(attr));
      } else {
        // Fall back to string array
        ffi::Array<ffi::String> attr;
        for (size_t i = 0; i < an->size(); ++i) {
          if (auto opt_str = (*an)[i].as<ffi::String>()) {
            attr.push_back(*opt_str);
          } else {
            TVM_FFI_THROW(InternalError) << "Not supported type: " << (*an)[i].GetTypeKey();
          }
        }
        SetNodeAttr(key, std::move(attr));
      }
    } else if (*value == nullptr) {  // Skip NullValue
      SetNodeAttr(key, ffi::String(""));
    } else if (const auto* im = (*value).as<IntImmObj>()) {
      SetNodeAttr(key, static_cast<int64_t>(im->value));
    } else if (const auto* fm = (*value).as<FloatImmObj>()) {
      SetNodeAttr(key, static_cast<double>(fm->value));
    } else if (const auto opt_str = (*value).as<ffi::String>()) {
      SetNodeAttr(key, *opt_str);
    } else {
      TVM_FFI_THROW(InternalError) << "Not yet supported type: " << (*value).GetTypeKey();
    }
  }

  void Extract(ffi::Object* node) {
    if (node) {
      this->VisitObjectFields(node);
    }
  }

 private:
  void VisitObjectFields(ffi::Object* obj) {
    const TVMFFITypeInfo* tinfo = TVMFFIGetTypeInfo(obj->type_index());
    TVM_FFI_ICHECK(tinfo->metadata != nullptr)
        << "Object `" << obj->GetTypeKey()
        << "` misses reflection registration and do not support serialization";
    ffi::reflection::ForEachFieldInfo(tinfo, [&](const TVMFFIFieldInfo* field_info) {
      Any field_value = ffi::reflection::FieldGetter(field_info)(obj);
      switch (field_value.type_index()) {
        case ffi::TypeIndex::kTVMFFINone: {
          SetNodeAttr(field_info->name.data, ffi::String(""));
          break;
        }
        case ffi::TypeIndex::kTVMFFIBool:
        case ffi::TypeIndex::kTVMFFIInt: {
          int64_t value = field_value.cast<int64_t>();
          this->Visit(field_info->name.data, &value);
          break;
        }
        case ffi::TypeIndex::kTVMFFIFloat: {
          double value = field_value.cast<double>();
          this->Visit(field_info->name.data, &value);
          break;
        }
        case ffi::TypeIndex::kTVMFFIDataType: {
          DLDataType value = field_value.cast<DLDataType>();
          this->Visit(field_info->name.data, &value);
          break;
        }
        case ffi::TypeIndex::kTVMFFITensor: {
          this->Visit(field_info->name.data, &field_value);
          break;
        }
        default: {
          this->Visit(field_info->name.data, &field_value);
          break;
        }
      }
    });
  }

  JSONGraphObjectPtr node_;
};

using NodeEntries = std::vector<JSONGraphNodeEntry>;

/*! \brief Serialize a Relax expression to JSON. */
class JSONSerializer : public relax::MemoizedExprTranslator<NodeEntries> {
 public:
  using MemoizedExprTranslator<NodeEntries>::VisitExpr_;
  using MemoizedExprTranslator<NodeEntries>::VisitBinding_;

  /*!
   * \brief Constructor
   * \param constant_names The names of all constants in the original module.
   */
  explicit JSONSerializer(const ffi::Map<GenericConst, ffi::String>& constant_names)
      : constant_names_(std::move(constant_names)) {}

  void serialize(Function func) {
    // First we convert all the parameters into input nodes.
    for (const auto& param : func->params) {
      auto node_ptr = std::make_shared<JSONGraphNode>(param->name, "input" /* op_type_ */);
      memo_[param] = AddObj(node_ptr, param);
    }
    heads_ = VisitExpr(func->body);
  }

  /*!\brief Return the required constants. */
  ffi::Array<ffi::String> GetConstantNames() const { return constants_used_; }

  /*!\brief Return the generated json. */
  std::string GetJSON() {
    namespace json = ffi::json;
    return std::string(json::Stringify(SaveToJSON()));
  }

 protected:
  /*!
   * \brief Add a node to graph.
   *
   * \param node A graph node. It is a shared pointer. Some attributes of it
   *        will be added, i.e. shape and type. These attributes are attached to
   *        the JSON graph in the end.
   * \param expr The relax expression.
   * \return A list of graph entry nodes. It the relax expr is a tuple type, we
   *         will flatten it.
   */
  NodeEntries AddObj(JSONGraphObjectPtr node, const Expr& expr) {
    auto ty = GetType(expr);
    auto node_id = nodes_.size();
    nodes_.push_back(node);
    NodeEntries ret;
    ShapeVector shape;
    TypeVector dtype;

    // Flatten tuple node.
    if (const auto* tuple_ty = ty.as<TupleTypeObj>()) {
      for (size_t i = 0; i < tuple_ty->fields.size(); ++i) {
        const auto* tensor_ty = tuple_ty->fields[i].as<TensorTypeObj>();
        TVM_FFI_ICHECK(tensor_ty) << "Expect TensorType, but received: ."
                                  << tuple_ty->fields[i]->GetTypeKey();
        TVM_FFI_ICHECK(tensor_ty->shape.has_value()) << "Expect shape to be defined.";
        ShapeExpr output_shape = tensor_ty->shape.value().as_or_throw<ShapeExpr>();
        ret.push_back(JSONGraphNodeEntry(node_id, i));
        shape.emplace_back(GetIntShape(output_shape->values));
        dtype.emplace_back(DType2String(tensor_ty->dtype.value()->dtype));
      }
      node->SetNumOutput(tuple_ty->fields.size());
    } else {
      const auto* tensor_ty = ty.as<TensorTypeObj>();
      TVM_FFI_ICHECK(tensor_ty) << "Expect TensorType, but received: " << ty->GetTypeKey();
      TVM_FFI_ICHECK(tensor_ty->shape.has_value()) << "Expect shape to be defined.";
      ShapeExpr output_shape = tensor_ty->shape.value().as_or_throw<ShapeExpr>();

      shape.emplace_back(GetIntShape(output_shape->values));
      dtype.emplace_back(DType2String(tensor_ty->dtype.value()->dtype));
      ret.push_back(JSONGraphNodeEntry(node_id, 0));
    }
    node->SetShape(shape);
    {
      std::vector<DLDataType> dl_dtypes;
      for (const auto& dt : dtype) {
        dl_dtypes.push_back(ffi::StringToDLDataType(dt));
      }
      node->SetDType(dl_dtypes);
    }
    return ret;
  }

  void SetCallNodeAttribute(JSONGraphObjectPtr node, const CallObj* cn) {
    if (cn->op.as<OpObj>()) {
      OpAttrExtractor extractor(node);
      const ffi::Object* call_attr = cn->attrs.get();
      extractor.Extract(const_cast<ffi::Object*>(call_attr));
    } else if (const auto* fn = cn->op.as<FunctionObj>()) {
      TVM_FFI_ICHECK(false);
      auto pattern = fn->GetAttr(tvm::relax::attr::kPartitionedFromPattern);
      TVM_FFI_ICHECK(pattern.has_value());
      node->SetAttr(tvm::relax::attr::kPartitionedFromPattern.name, pattern.value());
    }
  }

  NodeEntries VisitBinding_(const MatchCastObj* binding) {
    TVM_FFI_THROW(InternalError) << "JSON runtime currently doesn't match cast\n";
    return {};
  }

  NodeEntries VisitBinding(const Binding& binding) {
    NodeEntries nodes;
    if (const auto* node = binding.as<VarBindingObj>()) {
      auto from_b = VisitBinding_(node);
      nodes.insert(nodes.end(), from_b.begin(), from_b.end());
    } else if (const auto* node = binding.as<MatchCastObj>()) {
      auto from_b = VisitBinding_(node);
      nodes.insert(nodes.end(), from_b.begin(), from_b.end());
    } else {
      TVM_FFI_THROW(TypeError) << "Invalid type: " << binding->GetTypeKey();
    }
    return nodes;
  }

  NodeEntries VisitBindingBlock(const BindingBlock& block) {
    NodeEntries nodes;
    if (const auto* node = block.as<DataflowBlockObj>()) {
      auto from_bb = VisitBindingBlock_(node);
      nodes.insert(nodes.end(), from_bb.begin(), from_bb.end());
    } else if (const auto* node = block.as<BindingBlockObj>()) {
      auto from_bb = VisitBindingBlock_(node);
      nodes.insert(nodes.end(), from_bb.begin(), from_bb.end());
    } else {
      TVM_FFI_THROW(TypeError) << "Invalid type: " << block->GetTypeKey();
    }
    return nodes;
  }

  NodeEntries VisitBindingBlock_(const BindingBlockObj* block) {
    NodeEntries nodes;
    for (Binding binding : block->bindings) {
      auto from_b = VisitBinding(binding);
      nodes.insert(nodes.end(), from_b.begin(), from_b.end());
    }
    return nodes;
  }

  NodeEntries VisitBindingBlock_(const DataflowBlockObj* block) {
    NodeEntries nodes;
    for (Binding binding : block->bindings) {
      auto from_b = VisitBinding(binding);
      nodes.insert(nodes.end(), from_b.begin(), from_b.end());
    }
    return nodes;
  }

  NodeEntries VisitExpr_(const SeqExprObj* op) {
    NodeEntries nodes;
    for (BindingBlock block : op->blocks) {
      VisitBindingBlock(block);
    }
    auto from_body = VisitExpr(op->body);
    nodes.insert(nodes.end(), from_body.begin(), from_body.end());
    return nodes;
  }

  NodeEntries VisitExprDefault_(const ffi::Object* op) {
    TVM_FFI_THROW(InternalError) << "JSON runtime currently doesn't support " << op->GetTypeKey();
    return {};
  }

  NodeEntries VisitExpr_(const GenericConstObj* cn) {
    auto name = constant_names_.find(ffi::GetRef<GenericConst>(cn));
    TVM_FFI_ICHECK(name != constant_names_.end())
        << "Cannot find the name of the constant: " << ffi::GetRef<GenericConst>(cn);
    constants_used_.push_back((*name).second);
    auto node = std::make_shared<JSONGraphNode>((*name).second, "const" /* op_type_ */);
    return AddObj(node, ffi::GetRef<Expr>(cn));
  }

  NodeEntries VisitExpr_(const TupleObj* tn) {
    NodeEntries fields;
    for (const auto& field : tn->fields) {
      auto ref = VisitExpr(field);
      fields.insert(fields.end(), ref.begin(), ref.end());
    }
    return fields;
  }

  NodeEntries VisitExpr_(const CallObj* cn) {
    Expr expr = ffi::GetRef<Expr>(cn);
    std::string name;
    if (const auto* op_node = cn->op.as<OpObj>()) {
      name = op_node->name;
    } else if (const auto* fn = cn->op.as<FunctionObj>()) {
      auto comp = fn->GetAttr(tvm::relax::attr::kComposite);
      TVM_FFI_ICHECK(comp.has_value()) << "JSON runtime only supports composite functions.";
      name = comp.value();
    } else {
      TVM_FFI_THROW(InternalError)
          << "JSON runtime does not support calls to " << cn->op->GetTypeKey();
    }

    // TODO(@sunggg): Revisit when we have op naming convention.
    // Currently, simply remove "relax." prefix to make it work.
    name = std::string("jsonruntime.") + name.substr(6);

    NodeEntries inputs;
    for (const auto& arg : cn->args) {
      auto res = VisitExpr(arg);
      inputs.insert(inputs.end(), res.begin(), res.end());
    }
    auto node = std::make_shared<JSONGraphNode>(name,     /* name_ */
                                                "kernel", /* op_type_ */
                                                inputs, 1 /* num_outputs_ */);
    SetCallNodeAttribute(node, cn);
    return AddObj(node, ffi::GetRef<Expr>(cn));
  }

  NodeEntries VisitExpr_(const TupleGetItemObj* gtn) {
    auto vtuple = VisitExpr(gtn->tuple);
    return {vtuple[gtn->index]};
  }

  NodeEntries VisitExpr_(const FunctionObj* fn) {
    TVM_FFI_ICHECK(fn->GetAttr(tvm::relax::attr::kComposite).has_value())
        << "JSON runtime only supports composite functions";

    // FunctionObj should be handled by the caller.
    return {};
  }

  ffi::json::Value SaveToJSON() {
    namespace json = ffi::json;
    std::vector<size_t> arg_nodes;
    for (size_t i = 0; i < nodes_.size(); ++i) {
      auto node = nodes_[i];
      if (node->IsLeaf()) {
        arg_nodes.push_back(i);
      }
    }
    size_t num_entry = 0;
    std::vector<size_t> node_row_ptr{0};
    for (auto node : nodes_) {
      num_entry += node->GetNumOutput();
      node_row_ptr.push_back(num_entry);
    }
    json::Object obj;
    // nodes
    json::Array nodes_arr;
    for (auto& n : nodes_) {
      nodes_arr.push_back(n->SaveToJSON());
    }
    obj.Set(ffi::String("nodes"), std::move(nodes_arr));
    // arg_nodes
    json::Array arg_arr;
    for (auto x : arg_nodes) arg_arr.push_back(static_cast<int64_t>(x));
    obj.Set(ffi::String("arg_nodes"), std::move(arg_arr));
    // heads
    json::Array heads_arr;
    for (const auto& h : heads_) {
      heads_arr.push_back(h.SaveToJSON());
    }
    obj.Set(ffi::String("heads"), std::move(heads_arr));
    // node_row_ptr
    json::Array nrp_arr;
    for (auto x : node_row_ptr) nrp_arr.push_back(static_cast<int64_t>(x));
    obj.Set(ffi::String("node_row_ptr"), std::move(nrp_arr));
    return obj;
  }

 private:
  /*! \brief JSON graph nodes. */
  std::vector<JSONGraphObjectPtr> nodes_;
  /*! \brief Output of the JSON graph. */
  NodeEntries heads_;
  /*! \brief The list of required constants, ordered. */
  ffi::Array<ffi::String> constants_used_;
  /*! \brief The names of all constants in the original module. */
  const ffi::Map<GenericConst, ffi::String> constant_names_;
};

}  // namespace contrib
}  // namespace backend
}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_BACKEND_CONTRIB_CODEGEN_JSON_CODEGEN_JSON_H_
