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
 * \file src/relax/transform/topological_sort.cc
 * \brief Perform a topological sort of Dataflow blocks
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>

#include <algorithm>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace {
struct InputNode {};
struct OutputNode {};

using DataflowNode = std::variant<InputNode, OutputNode, tvm::Var>;

bool operator==(const DataflowNode& a, const DataflowNode& b) {
  if (const tvm::Var* var_a = std::get_if<tvm::Var>(&a)) {
    if (const tvm::Var* var_b = std::get_if<tvm::Var>(&b)) {
      const tvm::VarObj* ptr_a = var_a->get();
      const tvm::VarObj* ptr_b = var_b->get();
      return ptr_a == ptr_b;
    }
  }

  return a.index() == b.index();
}

}  // namespace

template <>
struct std::hash<DataflowNode> {
  std::size_t operator()(const DataflowNode& node) const noexcept {
    if (const tvm::Var* var = std::get_if<tvm::Var>(&node)) {
      const tvm::VarObj* ptr = var->get();
      std::hash<decltype(ptr)> hasher;
      return hasher(ptr);
    } else {
      auto index = node.index();
      std::hash<decltype(index)> hasher;
      return hasher(index);
    }
  }
};

namespace tvm {
namespace relax {

namespace {

enum class TraversalOrder {
  DepthFirst,
  BreadthFirst,
};

enum class StartingLocation {
  FromInputs,
  FromOutputs,
};

struct Dependencies {
  std::vector<DataflowNode> binding_order;
  std::unordered_map<DataflowNode, std::deque<DataflowNode>> downstream_users;
  std::unordered_map<DataflowNode, std::deque<DataflowNode>> upstream_requirements;
};

class BindingOrderCollector : ExprVisitor {
 public:
  static Dependencies Collect(const Expr& expr) {
    BindingOrderCollector visitor;
    visitor.dependencies_.binding_order.push_back(InputNode());
    visitor(expr);

    // If there is a variable without any inputs (e.g. `R.const(1)`)
    // or an unused variable, these must be handled somewhere, to
    // ensure they are visited corrected.  It's easiest to perform the
    // depth/breadth-first search if handled here, with `std::nullopt`
    // acting as a special value, so that the later traversal doesn't
    // need to check for this special case.
    std::vector<DataflowNode> zero_input_bindings;
    std::vector<DataflowNode> unused_bindings;
    for (const auto& var : visitor.dependencies_.binding_order) {
      if (std::holds_alternative<Var>(var)) {
        if (!visitor.dependencies_.upstream_requirements.count(var)) {
          zero_input_bindings.push_back(var);
        }
        if (!visitor.dependencies_.downstream_users.count(var)) {
          unused_bindings.push_back(var);
        }
      }
    }

    for (const auto& var : zero_input_bindings) {
      visitor.dependencies_.upstream_requirements[var].push_back(InputNode());
      visitor.dependencies_.downstream_users[InputNode()].push_back(var);
    }
    for (auto it = unused_bindings.rbegin(); it != unused_bindings.rend(); it++) {
      const auto& var = *it;
      visitor.dependencies_.upstream_requirements[OutputNode()].push_front(var);
      visitor.dependencies_.downstream_users[var].push_front(OutputNode());
    }

    visitor.dependencies_.binding_order.push_back(OutputNode());

    return visitor.dependencies_;
  }

 private:
  void VisitVarDef(const Var&) override {}

  void VisitExpr_(const FunctionObj* op) override {
    for (const auto& var : op->params) {
      graph_vars_.insert(var);
      dependencies_.downstream_users[InputNode()].push_back(var);
      dependencies_.upstream_requirements[var].push_back(InputNode());
    }
    VisitExpr(op->body);
  }

  void VisitBinding(const Binding& binding) override {
    auto cache = current_binding_;
    current_binding_ = binding->var;
    graph_vars_.insert(binding->var);
    dependencies_.binding_order.push_back(binding->var);
    ExprVisitor::VisitBinding(binding);
    current_binding_ = cache;
  }

  void VisitExpr_(const VarObj* op) override {
    Var upstream_requirement = ffi::GetRef<Var>(op);
    if (!graph_vars_.count(upstream_requirement)) {
      return;
    }
    auto downstream_user = current_binding_;

    dependencies_.downstream_users[upstream_requirement].push_back(downstream_user);
    dependencies_.upstream_requirements[downstream_user].push_back(upstream_requirement);
  }

  DataflowNode current_binding_ = OutputNode();
  std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> graph_vars_;
  Dependencies dependencies_;
};

class TopologicalSorter : public ExprMutator {
 public:
  TopologicalSorter(TraversalOrder order, StartingLocation starting_location)
      : order_(order), starting_location_(starting_location) {}

  Expr VisitExpr_(const FunctionObj* op) override {
    auto cached = dependencies_;
    dependencies_ = BindingOrderCollector::Collect(ffi::GetRef<Expr>(op));

    if (starting_location_ == StartingLocation::FromOutputs) {
      std::reverse(dependencies_.binding_order.begin(), dependencies_.binding_order.end());
    }
    if (order_ == TraversalOrder::DepthFirst) {
      for (auto& [upstream_var, downstream_vars] : dependencies_.downstream_users) {
        std::reverse(downstream_vars.begin(), downstream_vars.end());
      }
    }

    auto output = ExprMutator::VisitExpr_(op);
    dependencies_ = cached;
    return output;
  }

  BindingBlock VisitBindingBlock_(const DataflowBlockObj* op) override {
    auto block = ffi::GetRef<DataflowBlock>(op);

    // A map from not-yet-defined variables to the binding that will
    // define the variable.  Items are removed from this map as they
    // are collected into `new_bindings`.
    std::unordered_map<Var, Binding> to_emit;
    for (const auto& binding : block->bindings) {
      to_emit.insert({binding->var, binding});
    }

    // A lookup map of `Var -> Var` edges, used to find the bindings
    // that may be emitted next.  When starting at the function
    // inputs, this is the map from variables to the downstream
    // variables that depend on them.  When starting at the function
    // outputs, this is the map from variables to the upstream
    // variables that they require.
    const auto& forward_edge_lookup = [&]() {
      switch (starting_location_) {
        case StartingLocation::FromInputs:
          return dependencies_.downstream_users;
        case StartingLocation::FromOutputs:
          return dependencies_.upstream_requirements;
        default:
          TVM_FFI_THROW(InternalError) << "Invalid enum value for StartingLocation";
      }
    }();

    // A lookup map of `Var -> Var` edges, used to determine if a
    // binding can legally be emitted.  When starting at the function
    // inputs, this is the map from variables to the upstream
    // variables that they require.  (i.e. A variable may not be
    // defined earlier than its last input.)  When starting at the
    // function outputs, this is the map from variables to the
    // downstream variables that depend on them.  (i.e. A variable may
    // not be defined later than its first usage.)
    const auto& backward_edge_lookup = [&]() {
      switch (starting_location_) {
        case StartingLocation::FromInputs:
          return dependencies_.upstream_requirements;
        case StartingLocation::FromOutputs:
          return dependencies_.downstream_users;
        default:
          TVM_FFI_THROW(InternalError) << "Invalid enum value for StartingLocation";
      }
    }();

    // The search state for nodes that must still be visited.  When
    // doing a depth-first search, this is used as a stack, with
    // `push_back` and `pop_back`.  When doing a breadth-first search,
    // this is used as a queue, with `push_back` and `pop_front`.  A
    // `std::deque` is used to support these two use cases.
    auto deque = [&]() -> std::deque<DataflowNode> {
      switch (starting_location_) {
        case StartingLocation::FromInputs:
          return {InputNode()};
        case StartingLocation::FromOutputs:
          return {OutputNode()};
        default:
          TVM_FFI_THROW(InternalError) << "Invalid enum value for StartingLocation";
      }
    }();

    std::unordered_set<DataflowNode> visited;

    // Given a variable that has just been defined (or std::nullopt for the
    // function's output), mark nodes as ready to visit.
    auto push_descendents_to_stack = [&](const DataflowNode& var) {
      auto it = forward_edge_lookup.find(var);
      if (it == forward_edge_lookup.end()) {
        return;
      }
      const auto& adjacent_vars = it->second;

      for (const auto& adjacent_var : adjacent_vars) {
        bool legal_to_output = [&]() -> bool {
          if (visited.count(adjacent_var)) {
            return false;
          }

          auto it = backward_edge_lookup.find(adjacent_var);
          TVM_FFI_ICHECK(it != backward_edge_lookup.end());
          const auto& prerequisites = it->second;
          return std::all_of(prerequisites.begin(), prerequisites.end(),
                             [&visited](const auto& var) { return visited.count(var); });
        }();

        if (legal_to_output) {
          deque.push_back(adjacent_var);
        }
      }
    };

    std::vector<Binding> new_bindings;
    while (deque.size()) {
      DataflowNode visiting;
      switch (order_) {
        case TraversalOrder::DepthFirst: {
          visiting = deque.back();
          deque.pop_back();
          break;
        }
        case TraversalOrder::BreadthFirst: {
          visiting = deque.front();
          deque.pop_front();
          break;
        }
        default: {
          TVM_FFI_THROW(InternalError)
              << "Invalid value for TraversalOrder: " << static_cast<int>(order_);
        }
      }

      if (auto var = std::get_if<Var>(&visiting)) {
        if (auto iter_emit = to_emit.find(*var); iter_emit != to_emit.end()) {
          new_bindings.push_back(iter_emit->second);
          to_emit.erase(iter_emit);
        }
      }
      visited.insert(visiting);
      push_descendents_to_stack(visiting);
    }

    TVM_FFI_ICHECK_EQ(to_emit.size(), 0)
        << "After visiting all bindings, "
        << "no bindings should remain to emit.  "
        << "However, bindings " <<
        [&]() {
          ffi::Array<Var> arr;
          for (const auto& [var, binding] : to_emit) {
            arr.push_back(var);
          }
          return arr;
        }()
        << " still remain after emitting "
        << ffi::Array<Binding>(new_bindings.begin(), new_bindings.end())
               .Map([](const Binding& binding) { return binding->var; });

    if (starting_location_ == StartingLocation::FromOutputs) {
      std::reverse(new_bindings.begin(), new_bindings.end());
    }

    block.CopyOnWrite()->bindings = new_bindings;
    return ExprMutator::VisitBindingBlock_(block.get());
  }

 private:
  TraversalOrder order_;
  StartingLocation starting_location_;
  Dependencies dependencies_;
};
}  // namespace

namespace transform {

Pass TopologicalSort(TraversalOrder order, StartingLocation starting_location) {
  auto pass_func = [=](Function func, IRModule, PassContext) {
    TopologicalSorter mutator(order, starting_location);
    return mutator(func).as_or_throw<Function>();
  };
  return relax::transform::CreateFunctionPass(pass_func, 0, "TopologicalSort");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.TopologicalSort",
                        [](ffi::String order_str, ffi::String direction_str) -> Pass {
                          TraversalOrder order = [&]() {
                            if (order_str == "depth-first") {
                              return TraversalOrder::DepthFirst;
                            } else if (order_str == "breadth-first") {
                              return TraversalOrder::BreadthFirst;
                            } else {
                              TVM_FFI_THROW(ValueError)
                                  << "Invalid value for traversal order: \"" << order_str << "\".  "
                                  << "Allowed values are \"depth-first\" or \"breadth-first\"";
                            }
                          }();

                          StartingLocation starting_location = [&]() {
                            if (direction_str == "from-inputs") {
                              return StartingLocation::FromInputs;
                            } else if (direction_str == "from-outputs") {
                              return StartingLocation::FromOutputs;
                            } else {
                              TVM_FFI_THROW(ValueError)
                                  << "Invalid value for starting location: \"" << direction_str
                                  << "\".  "
                                  << "Allowed values are \"from-inputs\" or \"from-outputs\"";
                            }
                          }();

                          return TopologicalSort(order, starting_location);
                        });
}

}  // namespace transform

}  // namespace relax
}  // namespace tvm
