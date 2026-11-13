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
 * Lower block init stmt into branch stmt
 * \file lower_reduction.cc
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/transform.h>

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

class InitBlockLower : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) override {
    if (value.as<ExprObj>()) return ffi::Unchanged();
    return StmtExprMutator::Mutate(value, inplace_mode);
  }

 private:
  UnchangedOr<Stmt> Mutate_(const SBlockObj* block, InplaceMode inplace_mode) final {
    if (!block->init.has_value()) {
      return StmtExprMutator::Mutate_(block, inplace_mode);
    }
    Stmt init = DoLowering(block->init.value(), block->iter_vars);
    Stmt body = Mutate(block->body, inplace_mode).ValueOrUnchanged(block->body);
    body = SeqStmt({init, body});
    if (inplace_mode == InplaceMode::kAllow) {
      auto* writable = const_cast<SBlockObj*>(block);
      writable->init = std::nullopt;
      writable->body = std::move(body);
      return ffi::Unchanged();
    } else {
      auto copy = ffi::make_object<SBlockObj>(*block);
      copy->init = std::nullopt;
      copy->body = std::move(body);
      return SBlock(std::move(copy));
    }
  }

  static Stmt DoLowering(const Stmt& init, const ffi::Array<IterVar>& iter_vars) {
    std::vector<PrimExpr> conditions;
    for (const IterVar& var : iter_vars) {
      if (var->iter_type == IterVarType::kCommReduce) {
        conditions.push_back(equal(var->var, var->dom.value()->begin));
      }
    }
    // Handle the case where there is no condition
    if (conditions.empty()) {
      return init;
    }
    // Concat the conditions with logical and (&&)
    PrimExpr cond = conditions[0];
    for (size_t i = 1; i < conditions.size(); ++i) {
      cond = logical_and(cond, conditions[i]);
    }
    return If(cond, init);
  }
};

namespace transform {

Pass LowerInitBlock() {
  auto pass_func = [](Function f, IRModule m, PassContext ctx) {
    auto fptr = f.CopyOnWrite();
    fptr->body = ffi::make_object<InitBlockLower>()
                     ->Mutate(fptr->body, InplaceMode::kAllow)
                     .ValueOrUnchanged(std::move(fptr->body));
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "s_tir.LowerInitBlock");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.LowerInitBlock", LowerInitBlock);
}

}  // namespace transform

}  // namespace s_tir
}  // namespace tvm
