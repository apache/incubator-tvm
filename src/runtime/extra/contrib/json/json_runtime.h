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
 * \file src/runtime/contrib/json/json_runtime.h
 * \brief Utilities for json runtime.
 */

#ifndef TVM_RUNTIME_CONTRIB_JSON_JSON_RUNTIME_H_
#define TVM_RUNTIME_CONTRIB_JSON_JSON_RUNTIME_H_

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/json.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/runtime/tensor.h>
#include <tvm/support/bytes_io.h>
#include <tvm/support/io.h>

#include <cstddef>
#include <mutex>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "json_node.h"

namespace tvm {
namespace runtime {
namespace json {

/*!
 * \brief A json runtime that executes the serialized JSON format. This runtime
 * can be extended by user defined runtime for execution.
 */
class JSONRuntimeBase : public ffi::ModuleObj {
 public:
  JSONRuntimeBase(const std::string& symbol_name, const std::string& graph_json,
                  const ffi::Array<ffi::String> const_names)
      : symbol_name_(symbol_name), graph_json_(graph_json), const_names_(const_names) {
    LoadGraph(graph_json_);
  }

  const char* kind() const override { return "json"; }  // May be overridden

  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const override {
    return ffi::Module::kBinarySerializable | ffi::Module::kRunnable;
  }

  /*! \brief Initialize a specific json runtime. */
  virtual void Init(const ffi::Array<ffi::Tensor>& consts) = 0;

  /*! \brief Invoke the execution engine to inteprete a specific json runtime. */
  virtual void Run() = 0;

  /*!
   * \brief Invoke the debugger
   * \return External compiler specific debug blob
   */
  virtual std::string DebugDump(void) {
    TVM_FFI_THROW(InternalError) << "Not expected to be here : Debug dump w/o support ?";
    return "";
  }

  /*!
   * \brief Get a packed function.
   * \param name The name/symbol of the function.
   * \param sptr_to_self The pointer to the module node.
   * \return The packed function.
   */
  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) override {
    ffi::ObjectPtr<ffi::Object> sptr_to_self = ffi::GetObjectPtr<ffi::Object>(this);
    if (name == "get_symbol") {
      return ffi::Function(
          [sptr_to_self, this](ffi::PackedArgs args, ffi::Any* rv) { *rv = this->symbol_name_; });
    } else if (name == "get_const_vars") {
      return ffi::Function(
          [sptr_to_self, this](ffi::PackedArgs args, ffi::Any* rv) { *rv = this->const_names_; });
    } else if (this->symbol_name_ == name) {
      return ffi::Function([sptr_to_self, this](ffi::PackedArgs args, ffi::Any* rv) {
        TVM_FFI_ICHECK(this->initialized_) << "The module has not been initialized";

        // Bind argument tensors to data entries.
        this->SetInputOutputBuffers(args);

        // Execute the subgraph.
        this->Run();
      });
    } else if ("__init_" + this->symbol_name_ == name) {
      // The function to initialize constant tensors.
      return ffi::Function([sptr_to_self, this](ffi::PackedArgs args, ffi::Any* rv) {
        TVM_FFI_ICHECK_EQ(args.size(), 1U);
        std::lock_guard<std::mutex> guard(this->initialize_mutex_);
        if (!this->initialized_) {
          auto consts = args[0].cast<ffi::Array<ffi::Tensor>>();
          for (const auto& tensor : consts) {
            TVM_FFI_ICHECK(tensor.defined());
          }
          this->Init(consts);
          this->initialized_ = true;
        }
        *rv = 0;
      });
    } else {
      return std::nullopt;
    }
  }

  ffi::Bytes SaveToBytes() const override {
    std::string result;
    support::BytesOutStream stream(&result);
    // Save the symbol
    stream.Write(symbol_name_);
    // Save the graph
    stream.Write(graph_json_);
    // Save the required const names
    std::vector<std::string> consts;
    for (const auto& it : const_names_) {
      consts.push_back(it);
    }
    stream.Write(consts);
    return ffi::Bytes(std::move(result));
  }

  template <typename T,
            typename = typename std::enable_if<std::is_base_of<JSONRuntimeBase, T>::value>::type>
  static ffi::Module LoadFromBytes(const ffi::Bytes& bytes) {
    support::BytesInStream stream(bytes);
    std::string symbol;
    std::string graph_json;
    std::vector<std::string> consts;
    // Load the symbol
    TVM_FFI_ICHECK(stream.Read(&symbol)) << "Loading symbol name failed";
    TVM_FFI_ICHECK(stream.Read(&graph_json)) << "Loading graph json failed";
    TVM_FFI_ICHECK(stream.Read(&consts)) << "Loading the const name list failed";
    ffi::Array<ffi::String> const_names;
    for (const auto& it : consts) {
      const_names.push_back(it);
    }
    auto n = ffi::make_object<T>(symbol, graph_json, const_names);
    return ffi::Module(n);
  }

  /*!
   * \brief Get the JSON generated by codegen.
   *
   * \param format the format to return.
   * \return A string of JSON.
   */
  ffi::String InspectSource(const ffi::String& format) const override { return graph_json_; }

 protected:
  /*!
   * \brief Set up the input and output buffers by binding their DLTensor pointers to the
   * corresponding data entry.
   *
   * \param args The packed args.
   */
  void SetInputOutputBuffers(const ffi::PackedArgs& args) {
    TVM_FFI_ICHECK_EQ(args.size(), input_var_eid_.size() + outputs_.size())
        << "Found mismatch in the number of provided data entryies and required.";

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

      // Assign input/output the Tensor pointers to data entry so that we can directly
      // read/write host buffers.
      data_entry_[eid] = arg;
    }
  }

  /*!
   * \brief Load the graph and record the entries for inputs and constants.
   *
   * \param graph_json The graph in the json format.
   */
  void LoadGraph(const std::string& graph_json) {
    namespace json = ::tvm::ffi::json;
    auto root = json::Parse(graph_json).cast<json::Object>();
    this->Load(root);
    std::vector<std::string> consts;
    for (size_t i = 0; i < input_nodes_.size(); i++) {
      uint32_t nid = input_nodes_[i];
      std::string name = nodes_[nid].name_;
      if (nodes_[nid].op_type_ == "input") {
        TVM_FFI_ICHECK_EQ(nodes_[nid].GetOpShape().size(), nodes_[nid].GetOpDataType().size());
        for (size_t j = 0; j < nodes_[nid].GetOpShape().size(); ++j) {
          input_var_eid_.push_back(EntryID(nid, j));
        }
        nodes_[nid].SetNumOutput(nodes_[nid].GetOpShape().size());
      } else {
        TVM_FFI_ICHECK_EQ(nodes_[nid].op_type_, "const");
        auto pos = std::find(std::begin(const_names_), std::end(const_names_), name);
        TVM_FFI_ICHECK(pos != std::end(const_names_)) << "Found non-existent constant: " << name;
        const_idx_.push_back(nid);
        consts.push_back(name);
      }
    }
    TVM_FFI_ICHECK_EQ(consts.size(), const_names_.size())
        << "Found mismatch for the number of constants in the graph and required.";

    for (size_t i = 0; i < consts.size(); i++) {
      TVM_FFI_ICHECK_EQ(consts[i], const_names_[i])
          << "The position of constant in the graph must be the same as the required.";
    }

    // Reserve data entries.
    data_entry_.resize(NumEntries());
  }

  /*!
   * \brief Set up the constants/weights for inference by binding their DLTensor pointer to
   * the corresponding data entry.
   *
   * \param consts A list of constant Tensor to be used.
   */
  void SetupConstants(const ffi::Array<ffi::Tensor>& consts) {
    for (const auto& tensor : consts) {
      TVM_FFI_ICHECK(tensor.defined());
    }
    for (size_t i = 0; i < consts.size(); ++i) {
      data_entry_[EntryID(const_idx_[i], 0)] = consts[i].GetDLTensorPtr();
    }
  }

  // Load the graph.
  void Load(ffi::json::Object root) {
    namespace json = ::tvm::ffi::json;
    for (const auto& kv : root) {
      std::string key = std::string(kv.first.cast<ffi::String>());
      if (key == "nodes") {
        auto arr = kv.second.cast<json::Array>();
        nodes_.clear();
        for (const auto& n : arr) {
          JSONGraphNode node;
          node.Load(n.cast<json::Object>());
          nodes_.push_back(std::move(node));
        }
      } else if (key == "arg_nodes") {
        auto arr = kv.second.cast<json::Array>();
        input_nodes_.clear();
        for (const auto& v : arr) {
          input_nodes_.push_back(static_cast<uint32_t>(v.cast<int64_t>()));
        }
      } else if (key == "node_row_ptr") {
        auto arr = kv.second.cast<json::Array>();
        node_row_ptr_.clear();
        for (const auto& v : arr) {
          node_row_ptr_.push_back(static_cast<size_t>(v.cast<int64_t>()));
        }
      } else if (key == "heads") {
        auto arr = kv.second.cast<json::Array>();
        outputs_.clear();
        for (const auto& e : arr) {
          JSONGraphNodeEntry entry;
          entry.Load(e.cast<ffi::json::Array>());
          outputs_.push_back(entry);
        }
      } else if (key == "symbol") {
        // ignored
      } else {
        TVM_FFI_THROW(InternalError) << "Unknown key: " << key;
      }
    }
  }

  // Get the node entry index.
  uint32_t EntryID(uint32_t nid, uint32_t index) const { return node_row_ptr_[nid] + index; }

  // Get the node entry index.
  uint32_t EntryID(const JSONGraphNodeEntry& e) const { return EntryID(e.id_, e.index_); }

  // Number of node entries.
  uint32_t NumEntries() const { return node_row_ptr_.back(); }

 protected:
  /*! \brief The only subgraph name for this module. */
  std::string symbol_name_;
  /*! \brief The graph. */
  std::string graph_json_;
  /*! \brief The required constant names. */
  ffi::Array<ffi::String> const_names_;
  /*! \brief The json graph nodes. */
  std::vector<JSONGraphNode> nodes_;
  /*! \brief The input nodes, including variables and constants. */
  std::vector<uint32_t> input_nodes_;
  /*! \brief Used for quick entry indexing. */
  std::vector<uint32_t> node_row_ptr_;
  /*! \brief Output entries. */
  std::vector<JSONGraphNodeEntry> outputs_;
  /*! \brief Data of that entry. */
  std::vector<const DLTensor*> data_entry_;
  /*! \brief Map the input name to entry id. */
  std::vector<uint32_t> input_var_eid_;
  /*! \brief input const node index. */
  std::vector<uint32_t> const_idx_;
  /*! \brief Indicate if the engine has been initialized. */
  bool initialized_{false};
  /*! \brief Initializer mutex*/
  std::mutex initialize_mutex_;
};

}  // namespace json
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_CONTRIB_JSON_JSON_RUNTIME_H_
