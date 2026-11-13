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
 * \file graph.cc
 * \brief Utilities to get information about schedule graph.
 */
#include "graph.h"

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/te/operation.h>
#include <tvm/tirx/stmt_functor.h>

#include <unordered_set>
#include <vector>

namespace tvm {
namespace te {

// construct a read graph that gives readers of each operation
// that the root depend on
ReadGraph CreateReadGraph(const ffi::Array<Operation>& roots) {
  ReadGraph rmap;
  std::vector<Operation> stack;
  std::unordered_set<const ffi::Object*> visited;
  // initialize the roots
  for (Operation op : roots) {
    stack.push_back(op);
    visited.insert(op.get());
  }

  while (!stack.empty()) {
    Operation op = stack.back();
    stack.pop_back();
    ffi::Array<Tensor> deps = op->InputTensors();
    rmap.Set(op, deps);
    for (Tensor t : deps) {
      if (t->op.has_value() && visited.count(t->op.value().get()) == 0) {
        visited.insert(t->op.value().get());
        stack.push_back(t->op.value());
      }
    }
  }
  return rmap;
}

void PostDFSOrder(const Operation& op, const ReadGraph& g, std::unordered_set<Operation>* visited,
                  ffi::Array<Operation>* post_order) {
  if (visited->count(op)) return;
  visited->insert(op);
  for (const auto& t : g.at(op)) {
    PostDFSOrder(t->op.value(), g, visited, post_order);
  }
  post_order->push_back(op);
}

ffi::Array<Operation> PostDFSOrder(const ffi::Array<Operation>& roots, const ReadGraph& g) {
  std::unordered_set<Operation> visited;
  ffi::Array<Operation> post_order;
  for (Operation op : roots) {
    PostDFSOrder(op, g, &visited, &post_order);
  }
  return post_order;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("schedule.CreateReadGraph", CreateReadGraph)
      .def("schedule.PostDFSOrder", [](const ffi::Array<Operation>& roots, const ReadGraph& g) {
        return PostDFSOrder(roots, g);
      });
}

}  // namespace te
}  // namespace tvm
