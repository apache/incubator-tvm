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
 *
 * \file detect_recursion.cc
 *
 * \brief Analysis to detect global recursive or mutually recursive functions.
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr_functor.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>

namespace tvm {
namespace relax {

/*
 * General approach to detecting recursion:
 *   Suppose we have a dependency graph of global functions,
 *   where function A depends on function B if A contains a reference to B
 *   (i.e., an edge A->B means A references B). If function A is recursive,
 *   then it has a self-edge A->A.
 *
 *   Note that the call can happen _anywhere_ in the function's body:
 *   All that is important for mutual recursion is that one function
 *   needs the other to be in scope (it needs to know about it) to define
 *   the body. This includes calls that happen inside local function definitions,
 *   branches that may not execute, etc.
 *
 *   Then detecting simple recursion and mutual recursion is a problem of cycle
 *   detection: Functions F1, F2, ..., Fn are mutually recursive if there exists
 *   a single directed cycle that contains all of them.
 *
 *   We aim to find the _largest_ directed cycles in the graph, as there can
 *   be smaller cycles within the larger ones, as in the following example:
 *
 *   A <-> B <-> C
 *   ^     |     ^
 *   |     v     |
 *   |     D     |
 *   |     |     |
 *   v     v     v
 *   E <-> F <-> G
 *
 *   Handling a case like this in a directed graph is very difficult
 *   because most simple algorithms (variations of DFS) aim to find the smallest
 *   cycle, but in this case, we have multiple cycles that go through nodes multiple times:
 *   E.g., A->B->D->F->E->A, B->C->G->F->D->B, and A->B->C->G->F->E->A.
 *   However, we would consider _all_ of these nodes to be mutually recursive,
 *   and there is a single cycle: A->B->C->G->F->E->A->B->D->F->E->A (must go through A twice)
 *
 *   We can use Johnson's elementary circuit-finding algorithm (1975):
 *   https://epubs.siam.org/doi/10.1137/0204007
 *   and find all elementary circuits in the graph, which are cycles that go
 *   through nodes at most once.
 *
 *   With all the elementary cycles, we can coalesce different cycles that involve the
 *   same node, which would all form a group of mutually recursive functions
 */

class DependencyGatherer : public ExprVisitor {
 public:
  explicit DependencyGatherer(const IRModule& m) : m_(m) {}

  std::unordered_set<std::string> Track(const Function& func) {
    this->VisitExpr(func);
    return deps_;
  }

  void VisitExpr_(const GlobalVarObj* gv) override {
    // disregard Functions
    if (!m_->Lookup(ffi::GetRef<GlobalVar>(gv)).as<relax::FunctionObj>()) {
      return;
    }
    deps_.insert(gv->name_hint);
  }

 private:
  std::unordered_set<std::string> deps_;
  const IRModule& m_;
};

using adjacency_map = std::unordered_map<std::string, std::unordered_set<std::string>>;
using node_set = std::unordered_set<size_t>;
using adjacency_index = std::vector<node_set>;

adjacency_map GatherDependencyGraph(const IRModule& m) {
  adjacency_map ret;
  for (auto gv_func : m->functions) {
    const relax::FunctionObj* func = gv_func.second.as<relax::FunctionObj>();
    // disregard Functions and the like
    if (!func) {
      continue;
    }
    std::string name = gv_func.first->name_hint;
    auto deps = DependencyGatherer(m).Track(ffi::GetRef<relax::Function>(func));
    ret.insert({name, deps});
  }
  return ret;
}

// the graph algorithm pseudocode assumes vertices are indices and makes use of the fact you can
// increment them, so for ease, we convert the strings to indices by some ordering
adjacency_index ConvertToIndices(const adjacency_map& graph,
                                 const std::vector<std::string>& ordering) {
  adjacency_index ret;
  for (size_t i = 0; i < ordering.size(); i++) {
    std::string current = ordering[i];
    node_set neighbors;
    for (size_t j = 0; j < ordering.size(); j++) {
      if (graph.at(current).count(ordering[j])) {
        neighbors.insert(j);
      }
    }
    ret.push_back(neighbors);
  }
  return ret;
}

/********* Strongly connected component (SCC) detection, needed for Johnson's algorithm *********/
// Based on the pseudocode for Tarjan's SCC detection algorithm
// See: https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm

// Modification: We take a min_vert parameter to ignore all vertices below that.
// This is because Johnson's algorithm searches for SCCs on a subgraph of
// all vertices after a certain one (per some arbitrary ordering)

void StronglyConnect(size_t node, const adjacency_index& graph, size_t min_vert,
                     // use signed ints so that -1 can indicate undefined/unvisited
                     std::vector<int>* indices, std::vector<int>* low_links,
                     std::vector<size_t>* stack, std::vector<bool>* on_stack,
                     std::vector<node_set>* sccs, int* running_index) {
  indices->operator[](node) = *running_index;
  low_links->operator[](node) = *running_index;
  (*running_index)++;
  stack->push_back(node);
  on_stack->operator[](node) = true;

  auto children = graph.at(node);
  for (auto child : children) {
    // ignore children outside the verts we are checking
    if (child < min_vert) {
      continue;
    }
    if (indices->at(child) == -1) {
      StronglyConnect(child, graph, min_vert, indices, low_links, stack, on_stack, sccs,
                      running_index);
      low_links->operator[](node) = std::min(low_links->at(node), low_links->at(child));
    } else if (on_stack->at(child)) {
      low_links->operator[](node) = std::min(low_links->at(node), indices->at(child));
    }
  }

  // root node -> have found an SCC
  if (low_links->at(node) == indices->at(node)) {
    node_set scc;
    size_t m;
    do {
      m = stack->back();
      stack->pop_back();
      on_stack->operator[](m) = false;
      scc.insert(m);
    } while (m != node);
    sccs->push_back(scc);
  }
}

std::vector<node_set> FindStronglyConnectedComponents(const adjacency_index& graph,
                                                      size_t min_vert) {
  std::vector<size_t> stack;
  std::vector<node_set> sccs;
  int running_index = 0;

  std::vector<int> indices;
  std::vector<int> low_links;
  std::vector<bool> on_stack;
  for (size_t i = 0; i < graph.size(); i++) {
    indices.push_back(-1);
    low_links.push_back(-1);
    on_stack.push_back(false);
  }

  for (size_t i = min_vert; i < graph.size(); i++) {
    StronglyConnect(i, graph, min_vert, &indices, &low_links, &stack, &on_stack, &sccs,
                    &running_index);
  }
  return sccs;
}

/********* Helper functions needed for Johnson's algorithm *********/

// return strongly connected componenet containing the least vertex
node_set GetLeastSCC(const std::vector<node_set>& sccs) {
  int min_idx = 0;
  bool min_found = false;
  size_t min = 0;
  for (size_t i = 0; i < sccs.size(); i++) {
    if (!min_found) {
      min = *(sccs[i].begin());
      min_found = true;
      min_idx = i;
    }

    for (size_t v : sccs[i]) {
      if (v < min) {
        min = v;
        min_idx = i;
      }
    }
  }
  return sccs[min_idx];
}

size_t LeastVertex(const node_set& scc) {
  bool min_found = false;
  size_t min = 0;
  for (size_t v : scc) {
    if (!min_found) {
      min = v;
      min_found = true;
    }
    if (v < min) {
      min = v;
    }
  }
  return min;
}

/********* Johnson's algorithm implementation *********/
// implementation is based directly on the pseudocode from
// "Finding All the Elementary Circuits of a Directed Graph" (Johnson, 1975)

void Unblock(std::vector<bool>* blocked, std::vector<node_set>* blocked_nodes, size_t node) {
  blocked->operator[](node) = false;
  // copy so we don't modify the set we're iterating on
  auto blocked_on_node = node_set(blocked_nodes->at(node));
  for (auto blocked_node : blocked_on_node) {
    blocked_nodes->at(node).erase(blocked_node);
    if (blocked->at(blocked_node)) {
      Unblock(blocked, blocked_nodes, blocked_node);
    }
  }
}

bool CheckCircuit(const adjacency_index& graph, const node_set& scc,
                  std::vector<node_set>* blocked_nodes, std::vector<bool>* blocked,
                  std::vector<size_t>* current_stack, std::vector<node_set>* found_circuits,
                  size_t s, size_t v) {
  bool f = false;
  blocked->operator[](v) = true;
  current_stack->push_back(v);
  for (size_t child : graph[v]) {
    // ignore any node that's not in the SCC:
    // the algorithm considers only the subgraph pertaining to the SCC
    if (!scc.count(child)) {
      continue;
    }
    if (child == s) {
      // we found a circuit, so report it
      auto new_circuit = node_set(current_stack->begin(), current_stack->end());
      new_circuit.insert(s);
      found_circuits->push_back(new_circuit);
      f = true;
    } else if (!blocked->at(child)) {
      if (CheckCircuit(graph, scc, blocked_nodes, blocked, current_stack, found_circuits, s,
                       child)) {
        f = true;
      }
    }
  }
  if (f) {
    Unblock(blocked, blocked_nodes, v);
  } else {
    for (size_t child : graph[v]) {
      if (!scc.count(child)) {
        continue;
      }
      if (!blocked_nodes->at(child).count(v)) {
        blocked_nodes->at(child).insert(v);
      }
    }
  }
  current_stack->pop_back();
  return f;
}

std::vector<node_set> DetectElementaryCircuits(const adjacency_index& graph) {
  std::vector<node_set> blocked_nodes;
  for (size_t i = 0; i < graph.size(); i++) {
    blocked_nodes.push_back(node_set());
  }

  std::vector<bool> blocked;
  for (size_t i = 0; i < graph.size(); i++) {
    blocked.push_back(false);
  }
  std::vector<size_t> current_stack;
  std::vector<node_set> found_circuits;

  size_t s = 0;
  while (s < graph.size()) {
    auto sccs = FindStronglyConnectedComponents(graph, s);
    auto scc = GetLeastSCC(sccs);
    s = LeastVertex(scc);
    // Note: the pseudocode calls for an early exit if the subgraph is empty.
    // However, that will never happen (there will always be at least one SCC
    // with at least one node)
    for (size_t i = s; i < graph.size(); i++) {
      if (!scc.count(i)) {
        continue;
      }
      blocked[i] = false;
      blocked_nodes[i].clear();
    }
    CheckCircuit(graph, scc, &blocked_nodes, &blocked, &current_stack, &found_circuits, s, s);
    s++;
  }
  return found_circuits;
}

/********* Coalescing the circuits and returning the results *********/

// given all elementary circuits, we want to coalesce any circuits that share nodes
std::vector<node_set> CoalesceCircuits(const std::vector<node_set>& circuits) {
  std::vector<node_set> ret;
  std::unordered_set<size_t> merged;
  bool changed = false;
  for (size_t i = 0; i < circuits.size(); i++) {
    if (merged.count(i)) {
      continue;
    }
    node_set current(circuits[i].begin(), circuits[i].end());
    for (size_t j = i + 1; j < circuits.size(); j++) {
      if (merged.count(j)) {
        continue;
      }
      for (size_t member : current) {
        if (circuits[j].count(member)) {
          changed = true;
          merged.insert(j);
          current.insert(circuits[j].begin(), circuits[j].end());
        }
      }
    }
    ret.push_back(current);
  }
  // try again if something changed, as there may be more chances to coalesce
  if (changed) {
    return CoalesceCircuits(ret);
  }
  return ret;
}

tvm::ffi::Array<tvm::ffi::Array<GlobalVar>> DetectRecursion(const IRModule& m) {
  auto graph = GatherDependencyGraph(m);

  // have to decide on some ordering for names
  std::vector<std::string> name_ordering;
  for (auto kv : graph) {
    name_ordering.push_back(kv.first);
  }

  auto indices = ConvertToIndices(graph, name_ordering);
  auto groups = CoalesceCircuits(DetectElementaryCircuits(indices));

  // convert to expected representation
  tvm::ffi::Array<tvm::ffi::Array<GlobalVar>> ret;
  for (auto group : groups) {
    tvm::ffi::Array<GlobalVar> found;
    for (size_t node : group) {
      found.push_back(m->GetGlobalVar(name_ordering[node]));
    }
    ret.push_back(found);
  }
  return ret;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.analysis.detect_recursion", DetectRecursion);
}

}  // namespace relax
}  // namespace tvm
