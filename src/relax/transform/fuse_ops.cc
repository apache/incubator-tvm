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
 * \file src/relax/transform/fuse_ops.cc
 * \brief This file contains a pass which groups bindings in a dataflow block of Relax
 * functions and generate a new grouped Relax function for each group, according to the fusion
 * algorithm described below. By grouping bindings into new Relax functions, we substitute the
 * bindings in the function being manipulated into function calls to the new grouped function.
 *
 * A follow-up pass named "FuseTIR" will generate a TIR tirx::Function for each grouped function.
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/ir/expr_functor.h>
#include <tvm/ir/function.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/dataflow_matcher.h>
#include <tvm/relax/dataflow_pattern.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/relax/utils.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/function.h>

#include <optional>

#include "../../support/arena.h"
#include "../analysis/graph_partitioner.h"
#include "tvm/relax/expr.h"
#include "utils.h"

namespace tvm {
namespace relax {
using namespace tvm::prim;

struct ExprIdentityLess {
  bool operator()(const Expr& lhs, const Expr& rhs) const { return lhs.get() < rhs.get(); }
};

TVM_FFI_STATIC_INIT_BLOCK() {
  transform::FusionPatternObj::RegisterReflection();
  transform::PatternCheckContextObj::RegisterReflection();
}

/*
  Note on Fusing algorithm:

  The main challenge of general fusor is to handle possible diamond shape branches,
  in the following graph, conv2d can be fused to elemwise add.

            conv2d
            /  |  \
           /   |   \
         op    op   op
          \    |    /
           \   |   /
          elemwise add
               |

  However, at the point of conv2d we do not necessarily know that all the future paths
  will merge at the elemwise add. The fusion algorithm applies post-dominator analysis.

  The immediate post-dominator of a node defined by the closest node where all the future path goes
  into. In the above case, the elemwise add is the post-dominator of conv2d. The general algorithm
  is as follows:

  - Construct a DAG of dataflow graph for dominator analysis
  - Construct a post-dominator tree which gives immediate post dominator of each node.
  - Run fusion algorithm with the given post-dominator information.

  Note that, because we run analysis on a DAG, we use a single pass post-dominator
  tree construction algorithm via LCA, which is simpler than the full version that handles cycles.

  The fusion algorithm traverses from each node and checks if it can be fused to its
  immediate post dominator. It has to check the following things:

  - CheckPath: check all the path between a node and its immediate post-dominator
               satisfies the fuse condition.
  - Note that these intermediate node can already be fused with another nodes, the algorithm
      will still run correctly.
  - CommitFuse: mark all the nodes between source and post-dominator as the same group.
  - We use an Union-Find data structure to manage the groups.
*/

using support::LinkNode;

constexpr uint32_t kMaxFusedOps = 256;

TVM_FFI_STATIC_INIT_BLOCK() {
  ::tvm::transform::PassContext::RegisterConfigOption<int64_t>("relax.FuseOps.max_depth");
}

class GraphCreator : public ExprVisitor {
 public:
  /*!
   * \brief Create a IndexedForwardGraph according to the input module. The graph will be used for
   * graph partition and operator fusion.
   * \param mod The module which the creation accords to
   * \param arena The allocator of all the internal node objects
   * \return The created IndexedForwardGraph
   */
  static IndexedForwardGraph Create(IRModule mod, support::Arena* arena) {
    GraphCreator creator(mod, arena);
    for (const auto& it : mod->functions) {
      // Only visit Relax functions with neither attr::kPrimitive.name nor
      // attr::kCodegen.name.  Relax functions with `attr::kPrimitive.name` are
      // previously fused functions, potentially from a previous use
      // of `FuseOps` or `FuseOpsByPattern`.  Relax functions with
      // `attr::kCodegen.name` are previously fused functions from
      // `FuseOpsByPattern`, when the `annotate_codegen` option is
      // true.
      const auto* func = it.second.as<FunctionObj>();
      if (func == nullptr || func->HasNonzeroAttr(tvm::relax::attr::kPrimitive.name) ||
          func->GetAttr(tvm::relax::attr::kCodegen).has_value()) {
        continue;
      }
      creator(ffi::GetRef<Function>(func));
    }

    // The algorithm of the graph creator ensures that each created node will be added to the
    // post-dfs order and will be set its op pattern. Thus we check whether all these containers
    // have the same size.
    size_t n_nodes = creator.graph_.node_map.size();
    TVM_FFI_ICHECK_EQ(n_nodes, creator.graph_.post_dfs_order.size());
    TVM_FFI_ICHECK_EQ(n_nodes, creator.initialized_nodes_.size());

    return creator.graph_;
  }

 private:
  explicit GraphCreator(IRModule mod, support::Arena* arena)
      : mod_(std::move(mod)), arena_(arena) {}

  void VisitExpr_(const FunctionObj* func) final {
    for (const Var& param : func->params) {
      IndexedForwardGraph::Node* param_node = CreateNode(param.get());
      // The parameter is passed in from the outside, and thus it's marked as an external reference,
      // and it's pattern is `kOpaque`.
      MarkAsExternRef(param_node);
      SetNodePattern(param_node, OpPatternKind::kOpaque);
      AddToPostDFSOrder(param_node, param.get());
    }
    if (auto opt_num_input = func->GetAttr(tvm::relax::attr::kNumInput)) {
      for (int i = static_cast<int>(opt_num_input.value());
           i < static_cast<int>(func->params.size()); ++i) {
        input_params_.insert(func->params[i].get());
      }
    }
    ExprVisitor::VisitExpr_(func);
  }

  void VisitBinding_(const MatchCastObj* binding) final {
    IndexedForwardGraph::Node* node = CreateNode(binding->var.get());
    SetNodePattern(node, OpPatternKind::kOpaque);
    AddToPostDFSOrder(node, binding->var.get());
  }

  void VisitBinding_(const VarBindingObj* binding) final {
    IndexedForwardGraph::Node* node = CreateNode(binding->var.get());

    // If the variable is not a dataflow variable, it must be the output variable of this dataflow
    // block
    if (!binding->var->IsInstance<DataflowVarObj>()) {
      this->MarkAsExternRef(node);
    }
    if (const auto* call = binding->value.as<CallObj>()) {
      // Case 1. The expression is a CallObj
      VisitCall(call, node);
    } else if (const auto* tuple_get_item = binding->value.as<TupleGetItemObj>()) {
      // Case 2. The expression is a TupleGetItemObj
      VisitTupleGetItem(tuple_get_item, node);
    } else {
      VisitUnsupportedNode(binding->value, node);
      // Case 3. The type of the expression is not fusion-supported.
      // In this case, we skip adding edges, adding an empty node into graph.
    }
    AddToPostDFSOrder(node, binding->var.get());
  }

  /********** Non-Leaf Expression Nodes **********/

  void VisitCall(const CallObj* call, IndexedForwardGraph::Node* binding_var_node) {
    TVM_FFI_ICHECK_NOTNULL(binding_var_node);

    static const Op call_tir_op_ = Op::Get("relax.call_tir");
    static const Op call_tir_inplace_op_ = Op::Get("relax.call_tir_inplace");

    OpPatternKind pattern = OpPatternKind::kOpaque;
    ffi::Array<Expr> args = call->args;

    // - If the op being called is a TIR tirx::Function, we get the function op pattern directly
    // from the function attribute and visit the arguments one by one.
    // - Otherwise, the pattern of the current binding variable node is set to `kOpaque`, and we
    // recurse into the call expression.
    const auto* op = call->op.as<OpObj>();
    if (op == call_tir_op_.get() || op == call_tir_inplace_op_.get()) {
      const GlobalVar& global_var = call->args[0].as_or_throw<GlobalVar>();
      tirx::Function func = mod_->Lookup(global_var).as_or_throw<tirx::Function>();

      // Override args for call_tir
      args = call->args[1].as_or_throw<Tuple>()->fields;

      ffi::Optional<int64_t> opt_pattern = func->GetAttr(tvm::relax::attr::kOpPattern);
      if (opt_pattern.has_value()) {
        pattern = static_cast<OpPatternKind>(opt_pattern.value());
      } else {
        pattern = OpPatternKind::kOpaque;
      }
    }
    // The pattern of the current binding variable node is set to the pattern of this operator.
    SetNodePattern(binding_var_node, pattern);
    // Visit all call args
    for (const Expr& arg : args) {
      TVM_FFI_ICHECK(IsLeafOrTuple(arg))
          << "FuseOps expects all relax::Call nodes to have non-nested arguments, "
          << "but " << ffi::GetRef<Expr>(call) << " has argument " << arg
          << ", which is neither a leaf node nor a relax::Tuple";
      VisitLeaf(arg, binding_var_node, pattern);
    }
  }

  void VisitTupleGetItem(const TupleGetItemObj* tuple_item,
                         IndexedForwardGraph::Node* binding_var_node) {
    TVM_FFI_ICHECK_NOTNULL(binding_var_node);

    auto pattern = OpPatternKind::kInjective;
    if (input_params_.count(tuple_item->tuple.as<VarObj>())) {
      // TupleGetItem for fetching the parameter from the packed param tuple is treated as opaque
      // and won't be fused. This prevents the usage of packed param tuple changes the order of the
      // fusion result as the function usually begins with fetching the parameters.
      pattern = OpPatternKind::kOpaque;
    }
    SetNodePattern(binding_var_node, pattern);
    VisitLeaf(tuple_item->tuple, binding_var_node, pattern);
  }

  void VisitUnsupportedNode(const Expr& expr, IndexedForwardGraph::Node* binding_var_node) {
    TVM_FFI_ICHECK_NOTNULL(binding_var_node);
    SetNodePattern(binding_var_node, OpPatternKind::kOpaque);

    auto visit_leaves = [this, &binding_var_node](const Expr& e) {
      if (e->IsInstance<VarObj>() || e->IsInstance<GenericConstObj>()) {
        VisitLeaf(e, binding_var_node, OpPatternKind::kOpaque);
      }
    };
    PostOrderVisit(expr, visit_leaves);
  }

  /********** Leaf Expression Nodes **********/

  void VisitLeaf(const Expr& leaf_expr, IndexedForwardGraph::Node* binding_var_node,
                 const OpPatternKind& pattern) {
    TVM_FFI_ICHECK_NOTNULL(binding_var_node);

    // Recursive visit if it's Tuple
    if (const auto* tuple = leaf_expr.as<TupleObj>()) {
      for (const Expr& expr : tuple->fields) {
        VisitLeaf(expr, binding_var_node, pattern);
      }
      return;
    }

    if (!leaf_expr.as<ShapeExprObj>() && !leaf_expr.as<VarObj>() &&
        !leaf_expr.as<GenericConstObj>() && !leaf_expr.as<PrimExpr>() &&
        !leaf_expr.as<StringImmObj>() && !leaf_expr.as<DataTypeImmObj>()) {
      // Skip GlobalVar, ExternFunc, OpObj.
      return;
    }

    auto it = graph_.node_map.find(leaf_expr.get());
    IndexedForwardGraph::Node* leaf_node = nullptr;
    if (it != graph_.node_map.end()) {
      leaf_node = it->second;
    } else {
      leaf_node = CreateNode(leaf_expr.get());
      // Since we never fuse constants, the pattern of the constant is set to `kOpaque`.
      SetNodePattern(leaf_node, OpPatternKind::kOpaque);
      AddToPostDFSOrder(leaf_node, leaf_expr.get());
    }
    AddEdge(leaf_node, binding_var_node, pattern);
  }

  /********** Helper Functions **********/

  /*!
   * \brief Create a graph node corresponding to the input key
   * \param key The object which is used to create the graph node
   * \return The created graph node
   * \note The node corresponding to each key is supposed to be created for only once
   */
  IndexedForwardGraph::Node* CreateNode(const ffi::Object* key) {
    TVM_FFI_ICHECK(graph_.node_map.find(key) == graph_.node_map.end())
        << "The object " << ffi::GetRef<ffi::ObjectRef>(key)
        << " appears at multiple definition sites.";
    auto* node = arena_->make<IndexedForwardGraph::Node>();
    graph_.node_map[key] = node;
    return node;
  }

  /*!
   * \brief Append the input node to the post-dfs order of the graph
   * \param node The node to be appended
   * \param key The key corresponding to the node
   * \note Each node is supposed to be appended to the post-dfs order for only once
   */
  void AddToPostDFSOrder(IndexedForwardGraph::Node* node, const ffi::Object* key) {
    auto it = graph_.node_map.find(key);
    TVM_FFI_ICHECK(it != graph_.node_map.end() && it->second == node)
        << "Cannot add node " << ffi::GetRef<ffi::ObjectRef>(key) << " to the post-DFS order, "
        << "because the node for this object has not yet been created.";

    // We only set the reference of the node when adding it to the post-dfs order. Thus, if the
    // reference of a node is already set, it must have been appended to the post-dfs order.
    TVM_FFI_ICHECK(node->ref == nullptr)
        << "Cannot add node " << ffi::GetRef<ffi::ObjectRef>(key) << " to the post-DFS order, "
        << "because it has already been added.";

    node->ref = key;
    node->index = graph_.post_dfs_order.size();
    graph_.post_dfs_order.push_back(node);
  }

  /*!
   * \brief Add an edge from the input start to the input end in the graph, with specific pattern
   * \param start The start of the edge
   * \param end The end of the edge
   * \param pattern The pattern of this edge
   */
  void AddEdge(IndexedForwardGraph::Node* start, IndexedForwardGraph::Node* end,
               OpPatternKind pattern) {
    auto* link = arena_->make<LinkNode<IndexedForwardGraph::Edge>>();
    link->value.node = end;
    link->value.pattern = pattern;
    start->outputs.Push(link);
  }

  /*!
   * \brief Mark a given node as "external reference", which means the node cannot be fused as an
   * intermediate node
   * \param node The graph node to be marked
   */
  void MarkAsExternRef(IndexedForwardGraph::Node* node) { node->extern_ref = true; }

  /*!
   * \brief Set the pattern of the input node
   * \param node The graph node to be set
   * \param pattern The pattern of the node
   */
  void SetNodePattern(IndexedForwardGraph::Node* node, OpPatternKind pattern) {
    TVM_FFI_ICHECK(initialized_nodes_.find(node) == initialized_nodes_.end())
        << "The input node " << ffi::GetRef<ffi::ObjectRef>(node->ref)
        << " cannot have have its OpPatternKind set more than once.";
    initialized_nodes_.insert(node);
    node->pattern = pattern;
  }

 private:
  /*! \brief The IRModule from which the indexed forward graph is created */
  IRModule mod_;
  /*! \brief The allocator of all the internal node objects */
  support::Arena* arena_;
  /*! \brief The created indexed forward graph */
  IndexedForwardGraph graph_;
  /*! \brief The graph nodes whose patterns are set */
  std::unordered_set<IndexedForwardGraph::Node*> initialized_nodes_;
  /*! \brief The model params in the function input */
  std::unordered_set<const VarObj*> input_params_;
};

/*!
 * \brief The ExprMutator used to create a new grouped function
 * \details The workflow of this ExprMutator is:
 *  - The bindings in the function will be added by OperatorFusor via `AppendBinding(...)`.
 *  - When adding a new binding through `AppendBinding(...)`, we check whether the variables and
 *  constants used by the binding are defined by some previous added binding. And for the undefined
 *  variables and constants, we add them to the argument list and created new variables as the
 *  corresponding parameters.
 *  - When `CreateFunction()` is called, we go through each binding and update the binding with the
 *  new parameters. After that we wrap all bindings with a DataflowBlock and a Function.
 */
class FunctionCreator : public ExprMutator {
 public:
  explicit FunctionCreator(bool lift_constant, ffi::Map<Var, Expr> outer_bindings)
      : outer_bindings_(std::move(outer_bindings)), lift_constant_(lift_constant) {}
  /*!
   * \brief Append a new binding to this function and possibly create new parameters for the
   * function accordingly
   * \param binding The binding to be appended
   * \note Allowed bindings are:
   *  - VarBinding with value being a call node calling `relax.call_tir` or
   *    `relax.call_tir_inplace`.
   *  - VarBinding with value being a tuple-get-item node.
   * // TODO(tvm-team): handle match shape
   */
  void AppendBinding(const Binding& binding) {
    TVM_FFI_ICHECK(!function_.has_value())
        << "The `function_` is supposed to be uncreated when adding bindings";

    if (const auto* var_binding = binding.as<VarBindingObj>()) {
      if (const auto* call = var_binding->value.as<CallObj>()) {
        if (call->op.same_as(Op::Get("relax.call_tir")) ||
            call->op.same_as(Op::Get("relax.call_tir_inplace"))) {
          // Update the name of the function.
          name_hint_ = name_hint_ + "_" + call->args[0].as_or_throw<GlobalVar>()->name_hint;

          const Tuple& args = call->args[1].as_or_throw<Tuple>();
          for (const Expr& arg : args->fields) {
            CheckDefAndUpdateParam(arg);
            TVM_FFI_ICHECK(GetTypeAs<TupleTypeObj>(arg) == nullptr);
          }
          // TODO(tvm-team): handle shape expr
        } else {
          if (call->op->IsInstance<OpObj>()) {
            name_hint_ = name_hint_ + "_" + call->op.as_or_throw<Op>()->name;
          } else if (call->op->IsInstance<GlobalVarObj>()) {
            std::string gvar_name = call->op.as_or_throw<GlobalVar>()->name_hint;
            if (auto pos = gvar_name.find("fused_"); pos == 0) {
              name_hint_ = name_hint_ + "_" + gvar_name.substr(std::string("fused_").size());
            } else {
              name_hint_ = name_hint_ + "_" + gvar_name;
            }
          }

          for (const Expr& arg : call->args) {
            if (auto tuple = arg.as<TupleObj>()) {
              for (const Expr& tup_arg : tuple->fields) {
                CheckDefAndUpdateParam(tup_arg);
                TVM_FFI_ICHECK(GetTypeAs<TupleTypeObj>(tup_arg) == nullptr);
              }
            } else {
              CheckDefAndUpdateParam(arg);
            }
            if (GetTypeAs<TupleTypeObj>(arg) != nullptr) {
              // The argument is fully referenced. Thus we remove it from the mapping.
              partially_used_tuple_params_.erase(arg.get());
            }
          }
        }
      } else if (var_binding->value.as<TupleGetItemObj>()) {
        const auto* tuple_item = var_binding->value.as<TupleGetItemObj>();
        CheckDefAndUpdateParam(tuple_item->tuple);

        if (partially_used_tuple_params_.find(tuple_item->tuple.get()) !=
            partially_used_tuple_params_.end()) {
          // Appending get-item index to the mapping.
          partially_used_tuple_params_[tuple_item->tuple.get()].push_back(tuple_item->index);
        }
      }

      // Mark the binding variable as defined.
      defined_vars_.insert(var_binding->var.get());
      // Set var as output true if the binding is not a dataflow variable
      if (!var_binding->var->IsInstance<DataflowVarObj>()) {
        AppendOutput(var_binding->var);
      }
    } else {
      // TODO(tvm-team): handle match_cast
    }
    bindings_.push_back(binding);
  }

  /*! \brief Set a var defined in the group as output. */
  void AppendOutput(const Var& var) {
    TVM_FFI_ICHECK(defined_vars_.count(var.get()));
    if (GetOutputIndex(var)) return;
    output_vars_.push_back(var.get());
  }

  /*! \brief Variables returned from the grouped function, in return-value order. */
  const std::vector<const VarObj*>& output_vars() const { return output_vars_; }

  /*!
   * \brief Create the grouped function according to the collected bindings and parameters
   * \param composite_name The name to identify the pattern this function is created from, if any.
   * It will become the value of the kComposite attribute of the created function.
   * \note The created function won't be returned immediately. It's stored in the `function_` field.
   */
  void CreateFunction(ffi::Map<ffi::String, Any> group_attrs) {
    // Step 1. Start constructing a new dataflow block.
    builder_->BeginDataflowBlock();

    // Step 2. Handing partially used tuple parameters: replacing entire tuple
    // parameters with the parameters of its fields that are accessed in the
    // function.
    std::unordered_map<const ExprObj*, std::unordered_map<int, Var>> tuple_get_item_remap;
    for (auto& [tuple_arg, item_indices] : partially_used_tuple_params_) {
      TVM_FFI_ICHECK(!item_indices.empty());
      int param_idx = tuple_param_idx_[tuple_arg];
      Var param = params_[param_idx];
      ffi::String param_name = params_[param_idx]->name;
      TupleType param_ty = tuple_arg->ty.as_or_throw<TupleType>();

      ffi::Array<Expr> item_args;
      ffi::Array<Var> item_params;
      item_args.reserve(item_indices.size());
      item_params.reserve(item_indices.size());
      for (int item_idx : item_indices) {
        Var item_param(param_name + "_" + std::to_string(item_idx), param_ty->fields[item_idx]);
        item_args.push_back(TupleGetItem(ffi::GetRef<Expr>(tuple_arg), item_idx));
        item_params.push_back(item_param);
        tuple_get_item_remap[tuple_arg].insert_or_assign(item_idx, item_param);
      }
      arguments_.erase(arguments_.begin() + param_idx);
      arguments_.insert(arguments_.begin() + param_idx, item_args.begin(), item_args.end());
      params_.erase(params_.begin() + param_idx);
      params_.insert(params_.begin() + param_idx, item_params.begin(), item_params.end());
    }

    // Step 3. Now that the complete function boundary is known, inline primitive arguments whose
    // symbols participate in the boundary's tensor/shape types.
    InlineShapeDependentPrimArgs();

    // Step 4. Visit each binding and collect outputs one by one.
    std::vector<ffi::Optional<Expr>> output_slots(output_vars_.size());
    for (const Binding& binding : bindings_) {
      // Special handing for TupleGetItem.
      if (const auto* var_binding = binding.as<VarBindingObj>()) {
        if (const auto* tuple_get_item = var_binding->value.as<TupleGetItemObj>()) {
          auto it = tuple_get_item_remap.find(tuple_get_item->tuple.get());
          if (it != tuple_get_item_remap.end()) {
            TVM_FFI_ICHECK(it->second.find(tuple_get_item->index) != it->second.end());
            var_remap_.insert_or_assign(var_binding->var, it->second.at(tuple_get_item->index));
            if (auto output_idx = GetOutputIndex(binding->var)) {
              output_slots[*output_idx] = it->second.at(tuple_get_item->index);
            }
            continue;
          }
        }
      }

      if (auto output_idx = GetOutputIndex(binding->var)) {
        // Case 1. It is an output binding
        // We only allow VarBinding as output.
        const auto* var_binding = binding.as<VarBindingObj>();
        TVM_FFI_ICHECK_NOTNULL(var_binding);
        Var output_var = builder_->EmitOutput(VisitExpr(var_binding->value));
        var_remap_.insert_or_assign(var_binding->var, output_var);
        output_slots[*output_idx] = output_var;
      } else {
        // Case 2. It is an internal binding, add it to the binding list.
        VisitBinding(binding);
      }
    }

    ffi::Array<Expr> outputs;
    outputs.reserve(output_slots.size());
    for (const auto& output : output_slots) {
      TVM_FFI_ICHECK(output.has_value()) << "Missing fused function output";
      outputs.push_back(output.value());
    }

    // Step 5. Finish constructing the new block.
    BindingBlock new_block = builder_->EndBlock();
    if (outputs.empty()) {
      // If the result is not used outside
      LOG(WARNING) << "There are dead codes in the current IRModule, please run the "
                      "DeadCodeElimination Pass before FuseOps";
      function_ = std::nullopt;
    } else {
      Expr body = outputs.size() == 1 ? outputs[0] : Tuple(outputs);
      body = builder_->Normalize(body);
      body = builder_->Normalize(SeqExpr({new_block}, body));
      group_attrs.Set(tvm::relax::attr::kPrimitive.name, true);
      Function function = Function(/*params=*/params_,       //
                                   /*body=*/body,            //
                                   /*ret_ty=*/std::nullopt,  //
                                   /*is_pure=*/true,         //
                                   /*attrs=*/DictAttrs(group_attrs));
      ffi::Array<PrimExpr> free_vars = FreeSymbolicVars(function).Map(
          [](const tvm::Var& var) { return var.as_or_throw<PrimExpr>(); });
      if (!free_vars.empty()) {
        params_.push_back(Var("tir_vars", ShapeType(free_vars)));
        arguments_.push_back(ShapeExpr(free_vars));
        function = Function(/*params=*/params_,       //
                            /*body=*/body,            //
                            /*ret_ty=*/std::nullopt,  //
                            /*is_pure=*/true,         //
                            /*attrs=*/DictAttrs(group_attrs));
      }
      function_ = SymbolicVarRenewMutator::Renew(function);
    }
  }

  /*! \brief The original bindings of the function */
  ffi::Array<Binding> bindings_;
  /*! \brief The parameters of the function */
  ffi::Array<Var> params_;
  /*! \brief The arguments to call the function on the caller side */
  ffi::Array<Expr> arguments_;
  /*! \brief The name for the fused function */
  ffi::String name_hint_ = "fused";
  /*! \brief The constructed Relax function */
  ffi::Optional<Function> function_ = std::nullopt;

 private:
  std::optional<size_t> GetOutputIndex(Var v) {
    auto it = std::find(output_vars_.begin(), output_vars_.end(), v.get());
    if (it != output_vars_.end()) {
      return std::distance(output_vars_.begin(), it);
    }
    return std::nullopt;
  }

  Expr ResolveOuterBinding(const Expr& expr) {
    Expr bound_value = expr;
    std::unordered_set<const VarObj*> visited;
    while (const auto* current_var = bound_value.as<VarObj>()) {
      if (!visited.insert(current_var).second) break;
      auto it = outer_bindings_.find(ffi::GetRef<Var>(current_var));
      if (it == outer_bindings_.end()) break;
      bound_value = (*it).second;
    }
    return bound_value;
  }

  void InlineShapeDependentPrimArgs() {
    ffi::Array<Type> boundary_types = params_.Map([](const Var& param) { return GetType(param); });
    for (const VarObj* output_var : output_vars_) {
      boundary_types.push_back(GetType(ffi::GetRef<Var>(output_var)));
    }

    std::unordered_set<tvm::Var> boundary_shape_vars;
    for (const tvm::Var& var : TIRVarsInType(TupleType(boundary_types))) {
      boundary_shape_vars.insert(var);
    }

    for (const Expr& argument : deferred_prim_args_) {
      auto it = std::find_if(arguments_.begin(), arguments_.end(),
                             [&](const Expr& candidate) { return candidate.same_as(argument); });
      TVM_FFI_ICHECK(it != arguments_.end());

      Expr bound_value = ResolveOuterBinding(argument);
      bool inline_argument =
          bound_value.same_as(argument) && IsShapeDependentPrimExpr(argument, boundary_shape_vars);
      bool inline_bound_value = !bound_value.same_as(argument) &&
                                IsShapeDependentPrimExpr(bound_value, boundary_shape_vars);
      if (!inline_argument && !inline_bound_value) continue;

      if (inline_bound_value) {
        const auto* argument_var = argument.as<VarObj>();
        TVM_FFI_ICHECK(argument_var);
        inlined_bindings_.insert_or_assign(argument_var, bound_value);
      }

      size_t param_idx = it - arguments_.begin();
      arguments_.erase(arguments_.begin() + param_idx);
      params_.erase(params_.begin() + param_idx);
    }
  }

  /*!
   * \brief Check whether the input expression is defined within this function. If not, create a new
   * parameter for the expression.
   * \param expr The expression to be checked
   */
  void CheckDefAndUpdateParam(const Expr& expr) {
    // If the expression has already served as an argument, no need to create another one for it.
    if (std::find_if(arguments_.begin(), arguments_.end(), [&](const Expr& argument) {
          return argument.same_as(expr);
        }) != arguments_.end()) {
      return;
    }

    // If the expression is not a variable or is a undefined variable, it should be populated as a
    // parameter of the relax function.
    const auto* var = expr.as<VarObj>();
    if (var != nullptr && defined_vars_.count(var) == 0) {
      Var bound_var = ffi::GetRef<Var>(var);
      Expr bound_value = ResolveOuterBinding(bound_var);
      if (!bound_value.same_as(bound_var) && IsInlinableConstants(bound_value)) {
        inlined_bindings_.insert_or_assign(var, bound_value);
        return;
      }
    }
    if ((var == nullptr || defined_vars_.count(var) == 0) &&
        (lift_constant_ || !expr->IsInstance<GenericConstObj>())) {
      ffi::String name =
          var != nullptr ? var->name : ffi::String("param_" + std::to_string(n_param_for_const_++));
      Type param_ty = GetType(expr);
      if (!IsInlinableConstants(expr)) {
        Var param(std::move(name), GetType(expr));
        arguments_.push_back(expr);
        params_.push_back(param);
        if (IsSymbolicPrimExpr(expr)) {
          deferred_prim_args_.push_back(expr);
        }
      }

      // Mark the tuple parameter is partially referenced in the beginning.
      // We will remove it from the mapping once we find it is fully referenced.
      if (param_ty->IsInstance<TupleTypeObj>()) {
        partially_used_tuple_params_[expr.get()] = {};
        tuple_param_idx_[expr.get()] = static_cast<int>(arguments_.size()) - 1;
      }
    }
  }

  Expr VisitExpr(const Expr& expr) final {
    // If the expression serves as an argument, return its correspondng parameter.
    auto it = std::find_if(arguments_.begin(), arguments_.end(),
                           [&](const Expr& argument) { return argument.same_as(expr); });
    if (it != arguments_.end()) {
      return params_[it - arguments_.begin()];
    }
    if (const auto* var = expr.as<VarObj>()) {
      if (auto inlined = inlined_bindings_.find(var); inlined != inlined_bindings_.end()) {
        return inlined->second;
      }
    }
    // Otherwise, recurse into this expression.
    return ExprMutator::VisitExpr(expr);
  }

  // Check if the expression is constant PrimExpr or ShapeExpr or tuple of them that can be
  // inlined in the composite functions and excluded from args/params.
  bool IsSymbolicPrimExpr(const Expr& expr) {
    if (expr.as<CallObj>()) return false;
    if (auto prim_value = expr.as<PrimExpr>()) {
      return !tvm::UndefinedVars(prim_value.value()).empty();
    }
    return false;
  }

  bool IsShapeDependentPrimExpr(const Expr& expr,
                                const std::unordered_set<tvm::Var>& referenced_shape_vars) {
    if (!IsSymbolicPrimExpr(expr)) return false;
    ffi::Array<tvm::Var> undefined_vars = tvm::UndefinedVars(expr.as_or_throw<PrimExpr>());
    return std::all_of(undefined_vars.begin(), undefined_vars.end(),
                       [&](const tvm::Var& var) { return referenced_shape_vars.count(var); });
  }

  bool IsInlinableConstants(const Expr& expr) {
    if (const auto* tuple = expr.as<TupleObj>()) {
      return std::all_of(tuple->fields.begin(), tuple->fields.end(),
                         [this](const Expr& e) { return IsInlinableConstants(e); });
    } else if (expr.as<VarObj>() || expr.as<CallObj>()) {
      return false;
    } else if (auto prim_value = expr.as<PrimExpr>()) {
      return tvm::UndefinedVars(prim_value.value()).empty();
    } else if (const auto* shape_expr = expr.as<ShapeExprObj>()) {
      return std::all_of(shape_expr->values.begin(), shape_expr->values.end(),
                         [](const PrimExpr& e) { return tvm::UndefinedVars(e).empty(); });
    }
    return false;
  }

 private:
  /*! \brief The variables defined in this function */
  std::unordered_set<const VarObj*> defined_vars_;
  /*! \brief Caller variables replaced by statically inlinable bound values. */
  std::unordered_map<const VarObj*, Expr> inlined_bindings_;
  /*! \brief Symbolic primitive arguments classified after the complete boundary is known. */
  ffi::Array<Expr> deferred_prim_args_;
  /*! \brief The number of parameters reserved for constants */
  int n_param_for_const_ = 0;
  /*! \brief The output vars */
  std::vector<const VarObj*> output_vars_;
  /*! \brief Bindings in the caller function, used to inline static leaf expressions. */
  ffi::Map<Var, Expr> outer_bindings_;
  /*! \brief Whether or not to lift bound constants to parameters */
  bool lift_constant_;
  /*! \brief Mapping from tuple parameter of the function to its position index */
  std::unordered_map<const ExprObj*, int> tuple_param_idx_;
  /*!
   * \brief Mapping from partially referenced tuple parameter to the list of
   * indices that the parameter is referred by TupleGetItem
   */
  std::unordered_map<const ExprObj*, std::vector<int>> partially_used_tuple_params_;
};

/*!
 * \brief The ExprMutator used to fuse the operators in Relax functions
 * \details Given the partition results on the indexed-forward graph, for each group whose size is
 * larger than one, we create a new grouped function for it, containing all bindings in that group.
 * And we substitute the bindings in a group with a single function call to the newly created
 * grouped function. The workflow of this ExprMutator is: for each dataflow block,
 *   - we go through the bindings one by one. For each binding, if it is in a group whose size is
 *   larger than one, we add the binding to the function of the group it is in and update the
 *   parameters and arguments of that function;
 *   - then we finalize all the grouped functions by updating their bindings using BlockBuilder;
 *   - lastly, we go through the bindings again and substitute the bindings in a group with a single
 *   call to the corresponding grouped function.
 *
 * After transforming a Relax function, we update the function in the IRModule. Besides, we add all
 * newly created grouped function to the IRModule.
 */
class OperatorFusor : public ExprMutator {
 public:
  using Group = GraphPartitioner::Group;
  using GroupMap = std::unordered_map<const ffi::Object*, Group*>;

  OperatorFusor(IRModule mod, const GroupMap& obj2group, bool lift_constants = true)
      : ExprMutator(mod),
        mod_(std::move(mod)),
        obj2group_(obj2group),
        lift_constants_(lift_constants) {}

  /*!
   * \brief Construct a new operator fusor. Given the indexed-forward graph and the graph partition
   * result on that graph, the constructor creates a mapping from each leaf AST object
   * (e.g. parameters, variables, constants) to the group of the node corresponding to the object
   * in the graph.
   * \param mod The IRModule to be transformed
   * \param graph The indexed-forward graph of the input IRModule
   * \param groups The grouped result of the group partition on the input indexed-forward graph.
   * \param lift_constant Whether or not to lift bound constants to parameters of the grouped
   * function.
   */
  OperatorFusor(IRModule mod, const IndexedForwardGraph& graph, const std::vector<Group*>& groups,
                bool lift_constant = true)
      : OperatorFusor(mod, CreateGroupMap(graph, groups), lift_constant) {}

  /*!
   * \brief The main transformation on the IRModule
   * \return The new IRModule after transformation
   */
  IRModule Transform(const ffi::Array<ffi::String>& entry_function_names = {}) {
    ffi::Array<GlobalVar> entry_functions;
    if (entry_function_names.empty()) {
      entry_functions = mod_->GetGlobalVars();
    } else {
      for (const auto& name : entry_function_names) {
        entry_functions.push_back(mod_->GetGlobalVar(name));
      }
    }
    for (const auto& gv : entry_functions) {
      const auto& func = mod_->Lookup(gv);
      // Only visit Relax functions with neither attr::kPrimitive.name nor
      // attr::kCodegen.name.
      if (func->IsInstance<relax::FunctionObj>() &&
          !func->HasNonzeroAttr(tvm::relax::attr::kPrimitive.name) &&
          !func->GetAttr(tvm::relax::attr::kCodegen).has_value()) {
        outer_bindings_ = AnalyzeVar2Value(func);
        auto updated_func = VisitExpr(func).as_or_throw<Function>();
        builder_->UpdateFunction(gv, updated_func);
        outer_bindings_ = {};
      }
    }
    return builder_->GetContextIRModule();
  }

 private:
  static GroupMap CreateGroupMap(const IndexedForwardGraph& graph,
                                 const std::vector<Group*>& groups) {
    GroupMap obj2group;
    for (int nid = 0; nid < static_cast<int>(graph.post_dfs_order.size()); ++nid) {
      Group* group_root = groups[nid]->FindRoot();
      TVM_FFI_ICHECK(group_root != nullptr);
      TVM_FFI_ICHECK(graph.post_dfs_order[nid]->ref != nullptr);
      obj2group[graph.post_dfs_order[nid]->ref] = group_root;
    }
    return obj2group;
  }

  BindingBlock VisitBindingBlock_(const DataflowBlockObj* block) final {
    group2func_.clear();
    CollectGroupsWithCall(block->bindings);

    // Step 1. Collect the bindings for each grouped function.
    CollectFuncBindings(block->bindings);

    // Step 2. Collect all group's boundary (i.e. the output vars for each group)
    CollectFuncBoundary(block->bindings);

    // Step 3. Create the grouped function for each group.
    for (auto& [g, creator] : group2func_) {
      creator.CreateFunction(g->attrs);
    }

    // Step 4. Start generating the new binding block.
    //  - For groups with single binding, we directly recurse into the binding and emit the new one.
    //  - For groups with multiple bindings, we emit the call to the grouped function only when
    //  visiting the last binding of the group, because only by doing this we don't break the
    //  dependencies among the bindings of different groups. And therefore, we will skip all but the
    //  last binding of the group.
    builder_->BeginDataflowBlock();

    // Preserve the original binding order when emitting TupleGetItem bindings for groups with
    // multiple boundary outputs.  Missing entries are filled when the grouped call is emitted.
    std::unordered_map<Group*, std::vector<Var>> pending_output_remap;

    // A grouped function which returns a tuple requires attaching TupleGetItem to each element and
    // remapping variables in earlier bindings appropriately. Thus, a binding whose value depends on
    // some elements of a tuple from other group's function must be emitted after a call to the
    // tuple-producing function is emitted and remapping is done.
    // To guarantee this, we process bindings in the order of the topological sort of the group
    // dependency relations.
    for (const auto& binding : TopoSortByGroupDep(block->bindings)) {
      // Case 1. If the binding is the only binding in its group, recurse into it and emit the
      // transformed binding as usual.
      Group* group = GetGroupFromBinding(binding);
      if (!NeedsGroupedFunction(group)) {
        VisitBinding(binding);
        continue;
      }

      const auto& it_creator = group2func_.find(group);
      TVM_FFI_ICHECK(it_creator != group2func_.end());
      const FunctionCreator& func_info = it_creator->second;

      if (!func_info.function_.has_value()) {
        // The function is not created yet, so we skip the binding.
        continue;
      }
      const Function& func = func_info.function_.value();

      const auto& output_vars = func_info.output_vars();
      if (output_vars.size() > 1 && std::find(output_vars.begin(), output_vars.end(),
                                              binding->var.get()) != output_vars.end()) {
        pending_output_remap[group].push_back(binding->var);
      }

      // Case 2. If the binding is not the last binding of the group, we skip it.
      if (!func_info.bindings_.back().same_as(binding)) {
        continue;
      }

      // Case 3. The binding is the last binding of the group.
      const auto* var_binding = binding.as<VarBindingObj>();
      TVM_FFI_ICHECK(var_binding != nullptr)
          << "The last binding of a group whose size is larger than 1 "
             "is supposed to be a variable binding";

      // Step a. Add the grouped function to the IRModule
      GlobalVar gv = builder_->AddFunction(func, func_info.name_hint_);

      // Step b. Create the call to the deduplicated function, and then emit the call.
      // A multi-output call is internal to this dataflow block, while a single-output call has the
      // same dataflow/output status as its sole boundary variable.  The last binding is only the
      // insertion point and may itself be a dead internal binding.
      TVM_FFI_ICHECK(!output_vars.empty());
      Call call_to_emit = Call(Type::Missing(), gv, UpdateArgs(func_info.arguments_));

      Var new_var = output_vars.size() == 1 && !output_vars[0]->IsInstance<DataflowVarObj>()
                        ? builder_->EmitOutput(call_to_emit)
                        : builder_->Emit(call_to_emit);

      // Step c. Remap every boundary output to the corresponding result of the grouped call.
      // FunctionCreator uses output_vars() order when it constructs a multi-output tuple.  A
      // single boundary output is returned directly, including when that output is itself a tuple.
      if (output_vars.size() == 1) {
        var_remap_.insert_or_assign(ffi::GetRef<Var>(output_vars[0]), new_var);
        continue;
      }

      std::unordered_set<const VarObj*> remapped_outputs;
      auto remap_output = [&](const Var& output_var) {
        auto it = std::find(output_vars.begin(), output_vars.end(), output_var.get());
        TVM_FFI_ICHECK(it != output_vars.end());
        int index = static_cast<int>(std::distance(output_vars.begin(), it));
        TupleGetItem tuple_get(new_var, index);
        if (output_var->IsInstance<DataflowVarObj>()) {
          var_remap_.insert_or_assign(output_var, builder_->Emit(tuple_get));
        } else {
          var_remap_.insert_or_assign(output_var, builder_->EmitOutput(tuple_get));
        }
        remapped_outputs.insert(output_var.get());
      };

      if (auto it = pending_output_remap.find(group); it != pending_output_remap.end()) {
        for (const Var& output_var : it->second) {
          remap_output(output_var);
        }
      }
      for (const VarObj* output_var : output_vars) {
        if (!remapped_outputs.count(output_var)) {
          remap_output(ffi::GetRef<Var>(output_var));
        }
      }
    }
    // Step 5. Finish the binding block generation.
    return builder_->EndBlock();
  }

  /*! \brief Record the groups that contain at least one call, e.g. a call to `relax.call_tir`. */
  void CollectGroupsWithCall(const ffi::Array<Binding>& bindings) {
    groups_with_call_.clear();
    for (const Binding& binding : bindings) {
      const auto* var_binding = binding.as<VarBindingNode>();
      if (var_binding && var_binding->value->IsInstance<CallNode>()) {
        groups_with_call_.insert(GetGroupFromBinding(binding));
      }
    }
  }

  /*!
   * \brief Check whether a grouped function should be created for the group.
   * \note A group with a single binding needs no function. Neither does a group without any call,
   * e.g. a chain of TupleGetItem: there is nothing to fuse, and the resulting function would
   * contain no PrimFunc call for FuseTIR to lower.
   */
  bool NeedsGroupedFunction(Group* group) const {
    if (!group->attrs.empty()) {
      return true;
    }
    return group->num_nodes > 1 && groups_with_call_.count(group);
  }

  /*!
   * \brief Collect the bindings for each grouped function and update the information of the grouped
   * function
   * \param bindings The bindings to be collected
   * \note The function update is done by `AppendBinding(...)`
   */
  void CollectFuncBindings(const ffi::Array<Binding>& bindings) {
    for (const Binding& binding : bindings) {
      // If the binding is the only binding in its group, there is no need to create a new function.
      Group* group = GetGroupFromBinding(binding);
      if (!NeedsGroupedFunction(group)) {
        continue;
      }
      // Add the binding to the grouped function it's in, and update the function information
      // accordingly.
      auto it = group2func_.try_emplace(group, lift_constants_, outer_bindings_).first;
      it->second.AppendBinding(binding);
    }
  }

  void CollectFuncBoundary(const ffi::Array<Binding>& bindings) {
    for (const Binding& binding : bindings) {
      // Step 1. Get current binding's group
      Group* cur_group = GetGroupFromBinding(binding);

      // Step 2. Collect all used vars in the binding value and update bondary.
      // - If the var's group is same as the binding's, the var is defined in the same group
      // - If the var's group is different with the binding's, the var must be the output from
      //   another group. Mark it to be the group output.
      auto update_boundary = [this, binding, &cur_group](const Expr& e) {
        if (e->IsInstance<VarObj>() && obj2group_.count(e.get())) {
          const Var& used_var = e.as_or_throw<Var>();
          Group* producer_group = GetGroupFromVar(used_var);
          // Only check those group defined before.
          // Skip the vars from input or groups with single binding.
          if (producer_group != cur_group) {
            for (Group* depgroup : group_deps_[producer_group]) {
              TVM_FFI_ICHECK(depgroup != cur_group)
                  << "A cyclic dependency detected between the groups " << binding->var->name
                  << " and " << used_var->name << " are in.";
            }
            group_deps_[cur_group].push_back(producer_group);
          }

          if (auto producer = group2func_.find(producer_group);
              producer_group != cur_group && producer != group2func_.end()) {
            producer->second.AppendOutput(used_var);
          }
        }
      };

      if (const auto* var_binding = binding.as<VarBindingObj>()) {
        PostOrderVisit(var_binding->value, update_boundary);
      } else {
        const auto* match_cast = binding.as<MatchCastObj>();
        TVM_FFI_ICHECK_NOTNULL(match_cast);
        PostOrderVisit(match_cast->value, update_boundary);
      }
    }
  }

  /*!
   * \brief Get the group which the input binding is in
   * \param binding The binding to be queried
   * \return The pointer to the group which the input binding is in
   */
  Group* GetGroupFromBinding(const Binding& binding) {
    Var var = binding->var;
    return GetGroupFromVar(var);
  }

  /*!
   * \brief Get the group which the input var is in
   * \param Var The var to be queried
   * \return The pointer to the group which the input var is in
   */
  Group* GetGroupFromVar(const Var& var) {
    const auto& it_group = obj2group_.find(var.get());
    TVM_FFI_ICHECK(it_group != obj2group_.end())
        << "Variable " << var << " could not be found in any group";
    Group* group = it_group->second;
    return group->FindRoot();
  }

  /*!
   * \brief Update the pre-stored arguments according to the variable remapping of the fusor, by
   * recursing into each argument
   * \param args The arguments to be updated
   * \return The updated arguments
   */
  ffi::Array<Expr> UpdateArgs(const ffi::Array<Expr>& args) {
    ffi::Array<Expr> new_args;
    new_args.reserve(args.size());
    for (const Expr& arg : args) {
      new_args.push_back(VisitExpr(arg));
    }
    return new_args;
  }

 private:
  // Topologically sort bindings according to the group dependency relations.
  ffi::Array<Binding> TopoSortByGroupDep(const ffi::Array<Binding>& bindings) {
    std::unordered_map<Group*, std::vector<Binding>> bindings_per_group;
    // The order to visit groups should respect the original order of bindings as much as possible.
    std::vector<Group*> group_order;
    for (const auto& binding : bindings) {
      auto g = GetGroupFromBinding(binding);
      group_order.push_back(g);  // Duplication does not matter since each group is visited once.
      bindings_per_group[g].push_back(binding);
    }

    std::unordered_set<Group*> visited;

    std::function<void(Group*, std::function<void(Group*)>)> dfs_visit;
    dfs_visit = [this, &visited, &dfs_visit](Group* g, auto leaf_fun) {
      if (!visited.count(g)) {
        visited.insert(g);
        for (auto dep : group_deps_[g]) {
          dfs_visit(dep, leaf_fun);
        }
        leaf_fun(g);
      }
    };

    ffi::Array<Binding> sorted;

    for (auto g : group_order) {
      dfs_visit(g, [&sorted, &bindings_per_group](Group* leaf) {
        for (const auto& binding : bindings_per_group[leaf]) {
          sorted.push_back(binding);
        }
      });
    }

    return sorted;
  }

  /*! \brief The IRModule. */
  IRModule mod_;
  /*! \brief Internal arena. */
  support::Arena arena_;
  /*! \brief The group assignment map. */
  GroupMap obj2group_;
  /*! \brief The groups in the current binding block that contain at least one call. */
  std::unordered_set<Group*> groups_with_call_;
  /*! \brief Internal function information map. */
  std::unordered_map<Group*, FunctionCreator> group2func_;
  /*! \brief Bindings visible while rewriting the current Relax function. */
  ffi::Map<Var, Expr> outer_bindings_;
  /*!
   * \brief A map from a group to its dependent groups, used to detect cyclic dependencies.
   * \note Use vector so we can be deterministic, there won't be a lot of dep groups so
   *       linear search is OK.
   */
  std::unordered_map<Group*, std::vector<Group*>> group_deps_;
  /*! \brief Whether or not to lift bound constants to parameters of the grouped function. */
  bool lift_constants_{true};
};

IRModule FuseOps(IRModule mod, int opt_level, size_t max_fuse_depth) {
  support::Arena arena;

  // Step 1. Create the indexed-forward graph according to the input IRModule.
  IndexedForwardGraph graph = GraphCreator::Create(mod, &arena);

  // Step 2. Partition the graph by applying the fusion algorithm.
  std::vector<GraphPartitioner::Group*> groups =
      GraphPartitioner(&arena, opt_level, max_fuse_depth, /*max_function_args=*/0).Partition(graph);

  // Step 3. Transform the IRModule by fusing the operators in accordance with the graph partition
  // results.
  return OperatorFusor(mod, graph, groups, /*lift_constants*/ true).Transform();
}

IRModule MakeGroupedFunctions(
    IRModule mod, const std::unordered_map<const ffi::Object*, GraphPartitioner::Group*>& partition,
    bool lift_constants, const ffi::Array<ffi::String>& entry_function_names) {
  return OperatorFusor(mod, partition, lift_constants).Transform(entry_function_names);
}

/*! \brief Create a "partitioning", a map from interior / leaf expr to its representative group,
 * based on the provided pattern. The result can be passed to OperatorFusor above to fuse operations
 * in a group and create a grouped function.
 */
class PatternBasedPartitioner : ExprVisitor {
 public:
  using Group = GraphPartitioner::Group;
  using GroupMap = OperatorFusor::GroupMap;
  using PatternCheckContext = transform::PatternCheckContext;
  using ExprVisitor::VisitExpr_;
  using FCheckMatch = ffi::TypedFunction<bool(const transform::PatternCheckContext&)>;
  using FAttrsGetter =
      ffi::TypedFunction<ffi::Map<ffi::String, ffi::Any>(const ffi::Map<ffi::String, Expr>&)>;

  static GroupMap Run(ffi::String pattern_name, DFPattern pattern,
                      ffi::Map<ffi::String, DFPattern> annotation_patterns, FCheckMatch check,
                      Expr expr, support::Arena* arena, FAttrsGetter attrs_getter) {
    PatternBasedPartitioner part(pattern_name, pattern, annotation_patterns, check, arena,
                                 attrs_getter);
    part.VisitExpr(expr);
    return part.group_map_;
  }

  PatternBasedPartitioner(ffi::String pattern_name, DFPattern pattern,
                          ffi::Map<ffi::String, DFPattern> annotation_patterns, FCheckMatch check,
                          support::Arena* arena, FAttrsGetter attrs_getter)
      : pat_name_(pattern_name),
        pat_(pattern),
        annotation_pat_(annotation_patterns),
        check_(check),
        arena_(arena),
        attrs_getter_(attrs_getter) {}

  void VisitBindingBlock_(const DataflowBlockObj* block) final {
    current_block_use_def_ = DataflowBlockUseDef(ffi::GetRef<DataflowBlock>(block));
    ExprVisitor::VisitBindingBlock_(block);
    current_block_use_def_ = {};
  }

  void VisitVarDef(const Var& var) final {
    Group* g = arena_->make<Group>();
    group_map_[var.get()] = g;
    vars_in_group_[g].push_back(var);
  }

  void VisitBinding_(const VarBindingObj* binding) final {
    bindings_.Set(binding->var, binding->value);
    value_to_bound_var_.Set(binding->value, binding->var);
    ExprVisitor::VisitBinding_(binding);
  }

  void VisitExpr_(const GenericConstObj* op) final { group_map_[op] = arena_->make<Group>(); }
  void VisitExpr_(const DataTypeImmObj* op) final { group_map_[op] = arena_->make<Group>(); }

  void VisitBinding_(const VarBindingObj* binding, const CallObj* call) final {
    VisitVarDef(binding->var);
    if (auto matches_opt = ExtractMatchedExpr(pat_, ffi::GetRef<Call>(call), bindings_)) {
      const auto& context = CreatePatternCheckContext(call, matches_opt.value());
      if (check_ != nullptr && !check_(context)) {
        return;
      }

      for (const auto& [pat, match] : matches_opt.value()) {
        if ((pat->IsInstance<CallPatternObj>() && !match.same_as(ffi::GetRef<Call>(call))) ||
            pat->IsInstance<TupleGetItemPatternObj>()) {
          auto g = GetGroup(match);
          if (g && g->FindRoot()->num_nodes > 1) {
            // This expression has already been matched to a previous pattern.
            // If the prior matched subgraph is subsumed by the new matched one,
            // we can safely merge them, obtaining a maximized matched subgraph enventually.
            // Otherwise, merging them will result in an incorrect subgraph,
            // so we keep the prior subgraph and discard the current one by directly return.
            auto vars_in_prior_matched_graph = vars_in_group_[g];
            if (!GraphSubsumedInMatchedValues(vars_in_prior_matched_graph, matches_opt.value()))
              return;
          }
        }
      }

      // If a match is found, put all matching expressions into the same group.
      // OperatorFusor also requires that the bound variable be in the same group as the RHS value.
      // Since is_op(...) based pattern only matches against call nodes on the right hand side,
      // we need to take care of groups corresponding to the LHS bound variables carefully.

      // In the example below, conv2d + relu pattern would match if the "call" variable in this
      // function points to the relu op. We identify the group corresponding to "conv1", and make
      // it the representative group for relu and conv2d on the RHS and also "lv" on the LHS.

      // with R.dataflow():
      //   lv: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.conv2d(...)
      //   conv1: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.relu(lv)

      // parent_group corresponds to the group of "conv1" above.
      auto parent_group = GetGroupForBoundVar(binding->var);
      TVM_FFI_ICHECK(parent_group);
      parent_group->attrs.Set(tvm::relax::attr::kComposite.name, pat_name_);
      if (attrs_getter_ != nullptr) {
        const auto& custom_attrs = attrs_getter_(context->annotated_expr);
        for (const auto& pair : custom_attrs) {
          parent_group->attrs.Set(pair.first, pair.second);
        }
      }
      for (const auto& [pat, match] : matches_opt.value()) {
        // Put all matching expressions into the parent group. But we need to be careful not to
        // merge expressions matched by a wildcard pattern, since a wildcard can match an output of
        // the previous group. For example, when there are two back-to-back conv2d ops, the output
        // of the first conv2d is matched to the input of the second conv2d via a wildcard pattern.
        // But we must avoid merging the first conv2d into the group of the second conv2d.
        if ((pat->IsInstance<CallPatternObj>() && !match.same_as(ffi::GetRef<Call>(call))) ||
            pat->IsInstance<TupleGetItemPatternObj>()) {
          // Put the bound variable on the LHS into the same parent group.
          AddToGroup(value_to_bound_var_[match], parent_group);
        }
      }
    }
  }

 private:
  void AddToGroup(Expr e, Group* to) {
    if (group_map_[e.get()] != to) {
      --group_map_[e.get()]->num_nodes;
      group_map_[e.get()]->parent = to;
      vars_in_group_[to].push_back(e);
      ++to->num_nodes;
    }
  }

  Group* GetGroupForBoundVar(const Var& bound_var) {
    TVM_FFI_ICHECK(group_map_.count(bound_var.get()));
    return group_map_[bound_var.get()]->FindRoot();
  }

  Group* GetGroup(const Expr& exp) {
    if (value_to_bound_var_.count(exp) && group_map_.count(value_to_bound_var_[exp].get())) {
      return group_map_[value_to_bound_var_[exp].get()];
    }
    return nullptr;
  }

  PatternCheckContext CreatePatternCheckContext(const CallObj* call,
                                                const ffi::Map<DFPattern, Expr>& matched_result) {
    ffi::Map<ffi::String, Expr> annotated_expr;
    for (const auto& it : annotation_pat_) {
      if (matched_result.count(it.second)) {
        annotated_expr.Set(it.first, matched_result[it.second]);
      }
    }

    ffi::Map<Var, Expr> matched_bindings;
    for (const auto& [pat, match] : matched_result) {
      if (pat->IsInstance<CallPatternObj>() || pat->IsInstance<TupleGetItemPatternObj>()) {
        matched_bindings.Set(value_to_bound_var_[match], match);
      }
    }

    return PatternCheckContext(ffi::GetRef<Call>(call), annotated_expr, matched_bindings,
                               current_block_use_def_, value_to_bound_var_);
  }

  // check if a previous matched subgraph is subsumed by the current matched result
  bool GraphSubsumedInMatchedValues(const ffi::Array<Expr>& vars_in_graph,
                                    const ffi::Map<DFPattern, Expr>& matched_result) {
    std::set<Expr, ExprIdentityLess> matched_vars;
    for (const auto& [pat, match] : matched_result) {
      if ((pat->IsInstance<CallPatternObj>() || pat->IsInstance<TupleGetItemPatternObj>()))
        matched_vars.insert(value_to_bound_var_[match]);
    }

    for (const auto var : vars_in_graph) {
      if (matched_vars.find(var) == matched_vars.end()) return false;
    }
    return true;
  }

  ffi::String pat_name_;
  DFPattern pat_;
  ffi::Map<ffi::String, DFPattern> annotation_pat_;
  FCheckMatch check_;
  support::Arena* arena_;
  FAttrsGetter attrs_getter_;
  ffi::Map<Var, Expr> bindings_;
  ffi::Map<Expr, Var> value_to_bound_var_;
  ffi::Map<Var, ffi::Array<Var>> current_block_use_def_;
  GroupMap group_map_;
  std::map<Group*, ffi::Array<Expr>> vars_in_group_;
};

/*!
 * \brief Wrap each created composite function with another function, whose body consists
 * only of a call to the composite function, and annotate the outer function  with kCodegen
 * and kGlobalSymbol attributes.
 */
class CompositeFunctionAnnotator : public ExprMutator {
 public:
  explicit CompositeFunctionAnnotator(IRModule mod) : ExprMutator(mod) {}
  using ExprMutator::VisitExpr_;

  IRModule Run() {
    auto mod = builder_->GetContextIRModule();
    for (const auto& gv : mod->GetGlobalVars()) {
      auto it = mod->functions.find(gv);
      // Note that the fusion pass may have already removed the function.
      if (it == mod->functions.end()) {
        continue;
      }
      const auto& base_func = (*it).second;
      if (const auto* func = base_func.as<FunctionObj>()) {
        if (func->GetAttr(tvm::relax::attr::kComposite).has_value() ||
            func->GetAttr(tvm::relax::attr::kCodegen).has_value()) {
          continue;
        }

        auto new_body = VisitWithNewScope(func->body, func->params);
        if (!new_body.same_as(func->body)) {
          auto new_func =
              Function(func->params, new_body, func->ret_ty, func->is_pure, func->attrs, func->loc);
          builder_->UpdateFunction(gv, new_func);
        }
      }
    }
    return builder_->GetContextIRModule();
  }

  Expr VisitExpr_(const CallObj* call_node) final {
    if (auto const* gvar = call_node->op.as<GlobalVarObj>()) {
      if (auto it = gvar_map_.find(gvar); it != gvar_map_.end()) {
        return Call(Type::Missing(), it->second, call_node->args);
      }
      auto func = builder_->GetContextIRModule()->Lookup(ffi::GetRef<GlobalVar>(gvar));
      if (auto composite_name = func->GetAttr(tvm::relax::attr::kComposite)) {
        auto new_func = VisitExpr(func).as_or_throw<Function>();
        auto codegen_name = GetCodegenName(composite_name.value());
        auto gsymbol = gvar->name_hint + "_" + codegen_name;
        new_func = WithAttrs(new_func, {{tvm::relax::attr::kCodegen.name, codegen_name},
                                        {tvm::attr::kGlobalSymbol.name, gsymbol}});
        new_func = WithoutAttr(std::move(new_func), tvm::relax::attr::kPrimitive.name);
        builder_->GetContextIRModule()->Remove(ffi::GetRef<GlobalVar>(gvar));
        auto new_gvar = builder_->AddFunction(new_func, gsymbol);
        gvar_map_.insert_or_assign(gvar, new_gvar);
        return Call(Type::Missing(), new_gvar, call_node->args);
      }
    }
    return ExprMutator::VisitExpr_(call_node);
  }

  Expr VisitExpr_(const FunctionObj* func_node) final {
    Function f_inner = ExprMutator::VisitExpr_(func_node).as_or_throw<Function>();

    if (!func_node->GetAttr(tvm::relax::attr::kComposite)) {
      // This lambda function doesn't have `attr::kComposite.name`, so it
      // was not produced by FuseOps.
      return f_inner;
    }

    f_inner = WithoutAttr(std::move(f_inner), tvm::relax::attr::kPrimitive.name);

    ffi::Array<Var> param_vars;
    ffi::Array<Expr> params;

    for (auto v : func_node->params) {
      Var new_v(v->name, GetType(v));
      param_vars.push_back(new_v);
      params.push_back(new_v);
    }

    // We cannot delegate to `ExprMutator::VisitExpr_(const FunctionObj*)` at this point, as it
    // would recursively visit the Call node.  However, we are still required to generate
    // well-formed Relax IR.  As a result, we need to build the SeqExpr ourselves.
    Var local_func_var("local_func", GetType(f_inner));
    Var output_var("output", f_inner->ret_ty);
    SeqExpr new_body({BindingBlock({
                         VarBinding(local_func_var, f_inner),
                         VarBinding(output_var, Call(Type::Missing(), local_func_var, params)),
                     })},
                     output_var);

    // pure if the inner func is pure (no need to force purity if it's forced for the inner func)
    return Function(param_vars, new_body, func_node->ret_ty, f_inner->is_pure);
  }

 private:
  /*! \brief A map from old global vars to their replacements. */
  std::unordered_map<const GlobalVarObj*, GlobalVar> gvar_map_;
};

IRModule FuseOpsByPattern(const tvm::ffi::Array<transform::FusionPattern>& patterns, IRModule mod,
                          bool bind_constants, bool annotate_codegen,
                          ffi::Array<ffi::String> entry_function_names) {
  support::Arena arena;

  for (const auto& pattern : patterns) {
    ffi::Array<Function> entry_functions;
    if (entry_function_names.size()) {
      for (const auto& name : entry_function_names) {
        auto gv = mod->GetGlobalVar(name);
        auto func = mod->Lookup(gv);
        TVM_FFI_ICHECK(func->IsInstance<FunctionObj>())
            << "Entry function must be a relax function";
        entry_functions.push_back(func.as_or_throw<Function>());
      }
    } else {
      for (const auto& gv : mod->GetGlobalVars()) {
        const auto& base_func = mod->Lookup(gv);
        if (base_func->IsInstance<tirx::FunctionObj>()) {
          continue;
        }
        const FunctionObj* function = base_func.as<FunctionObj>();
        if (function->GetAttr(tvm::relax::attr::kPrimitive).value_or(false) ||
            function->GetAttr(tvm::relax::attr::kComposite).has_value() ||
            function->GetAttr(tvm::relax::attr::kCodegen).has_value()) {
          continue;
        }
        entry_functions.push_back(base_func.as_or_throw<Function>());
      }
    }
    OperatorFusor::GroupMap group_map;
    for (const auto& func : entry_functions) {
      auto map = PatternBasedPartitioner::Run(
          pattern->name, pattern->pattern, pattern->annotation_patterns,
          pattern->check.value_or(nullptr), func, &arena, pattern->attrs_getter.value_or(nullptr));
      for (const auto& [key, value] : map) {
        TVM_FFI_CHECK(!group_map.count(key), ValueError)
            << "IRModule is invalid.  "
            << "The object " << ffi::GetRef<ffi::ObjectRef>(key)
            << " appears in multiple partitions, "
            << "which can occur when the IRModule was not single-site assignment";
        group_map.insert({key, value});
      }
    }
    mod = MakeGroupedFunctions(mod, group_map, /*lift_constants*/ !bind_constants,
                               entry_function_names);
  }
  if (annotate_codegen) {
    return CompositeFunctionAnnotator(mod).Run();
  }
  return mod;
}

namespace transform {

FusionPattern::FusionPattern(ffi::String name, DFPattern pattern,
                             ffi::Map<ffi::String, DFPattern> annotation_patterns,
                             ffi::Optional<ffi::Function> check,
                             ffi::Optional<ffi::Function> attrs_getter) {
  ffi::ObjectPtr<FusionPatternObj> n = ffi::make_object<FusionPatternObj>(pattern);
  n->name = std::move(name);
  n->annotation_patterns = std::move(annotation_patterns);
  n->check = check;
  n->attrs_getter = attrs_getter;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "relax.transform.FusionPattern",
      [](ffi::String name, DFPattern pattern, ffi::Map<ffi::String, DFPattern> annotation_patterns,
         ffi::Optional<ffi::Function> check, ffi::Optional<ffi::Function> attrs_getter) {
        return FusionPattern(name, pattern, annotation_patterns, check, attrs_getter);
      });
}

PatternCheckContext::PatternCheckContext(Expr matched_expr,
                                         ffi::Map<ffi::String, Expr> annotated_expr,
                                         ffi::Map<Var, Expr> matched_bindings,
                                         ffi::Map<Var, ffi::Array<Var>> var_usages,
                                         ffi::Map<Expr, Var> value_to_bound_var) {
  ffi::ObjectPtr<PatternCheckContextObj> n =
      ffi::make_object<PatternCheckContextObj>(std::move(matched_expr));
  n->annotated_expr = std::move(annotated_expr);
  n->matched_bindings = std::move(matched_bindings);
  n->var_usages = std::move(var_usages);
  n->value_to_bound_var = std::move(value_to_bound_var);
  data_ = std::move(n);
}

Pass FuseOps(int fuse_opt_level) {
  auto pass_func =  //
      [=](IRModule m, PassContext pc) {
        int opt_level = fuse_opt_level == -1 ? pc->opt_level : fuse_opt_level;
        auto max_fuse_depth =
            pc->GetConfig<int64_t>("relax.FuseOps.max_depth").value_or(kMaxFusedOps);
        return relax::FuseOps(m, opt_level, static_cast<size_t>(max_fuse_depth));
      };
  return CreateModulePass(/*pass_function=*/pass_func,  //
                          /*opt_level=*/0,              //
                          /*name=*/"FuseOps");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.FuseOps", FuseOps);
}

Pass FuseOpsByPattern(const tvm::ffi::Array<FusionPattern>& patterns, bool bind_constants,
                      bool annotate_codegen, const ffi::Array<ffi::String>& entry_function_names) {
  auto pass_func =  //
      [=](IRModule m, PassContext pc) {
        return relax::FuseOpsByPattern(patterns, m, bind_constants, annotate_codegen,
                                       entry_function_names);
      };
  return CreateModulePass(/*pass_function=*/pass_func,  //
                          /*opt_level=*/0,              //
                          /*name=*/"FuseOpsByPattern");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.FuseOpsByPattern", FuseOpsByPattern);
}

}  // namespace transform

}  // namespace relax
}  // namespace tvm
