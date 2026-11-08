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

#ifndef TVM_RELAX_TRANSFORM_GLOBAL_VAR_REFERENCE_H_
#define TVM_RELAX_TRANSFORM_GLOBAL_VAR_REFERENCE_H_

#include <tvm/ir/module.h>
#include <tvm/relax/expr_functor.h>

#include <unordered_set>
#include <utility>

namespace tvm {
namespace relax {

// References used as values require the original function definition and signature.
class GlobalVarValueCollector : public ExprVisitor {
 public:
  static std::unordered_set<GlobalVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> Collect(
      const IRModule& mod) {
    GlobalVarValueCollector collector;
    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto func = base_func.as<Function>()) {
        collector.VisitExpr(func.value());
      }
    }
    return std::move(collector.references_);
  }

 private:
  void VisitExpr_(const GlobalVarNode* op) final { references_.insert(ffi::GetRef<GlobalVar>(op)); }

  void VisitExpr_(const CallNode* op) final {
    if (!op->op.as<GlobalVarNode>()) {
      VisitExpr(op->op);
    }
    for (const auto& arg : op->args) {
      VisitExpr(arg);
    }
  }

  std::unordered_set<GlobalVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> references_;
};

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_TRANSFORM_GLOBAL_VAR_REFERENCE_H_
