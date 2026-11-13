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
 * \file src/relax/transform/remove_purity_checking.cc
 * \brief Apply kForcePure in all pure functions and unwrap all calls to pure overrides
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/relax/utils.h>

namespace tvm {
namespace relax {

class PurityRemover : public ExprMutator {
 public:
  using ExprMutator::VisitExpr_;

  Function RemovePurity(Function func) {
    bool purity = func->is_pure;
    auto ret = func;
    if (purity) {
      ret = WithAttr<Function>(func, tvm::relax::attr::kForcePure.name, true);
    }
    auto new_body = VisitExpr(ret->body);
    if (!new_body.same_as(ret->body)) {
      return Function(ret->params, new_body, ret->ret_ty, ret->is_pure, ret->attrs, ret->loc);
    }
    return ret;
  }

  Expr VisitExpr_(const CallObj* call) override {
    if (call->op.same_as(call_pure_packed_op_)) {
      auto ret = Call(Type::Missing(), call->args[0],
                      ffi::Array<Expr>(call->args.begin() + 1, call->args.end()), call->attrs,
                      call->ty_args);
      return VisitExpr(ret);
    }
    if (call->op.same_as(call_inplace_packed_op_)) {
      // call_inplace_packed has its own attrs so we don't pass those down
      auto ret = Call(Type::Missing(), call->args[0],
                      ffi::Array<Expr>(call->args.begin() + 1, call->args.end()), std::nullopt,
                      call->ty_args);
      return VisitExpr(ret);
    }
    if (call->op.same_as(invoke_pure_closure_op_)) {
      auto ret = Call(Type::Missing(), invoke_closure_op_, call->args, call->attrs, call->ty_args);
      return VisitExpr(ret);
    }
    return ExprMutator::VisitExpr_(call);
  }

  Expr VisitExpr_(const FunctionObj* func) override {
    // handling inner functions: we will remove purity annotations from them too
    return RemovePurity(ffi::GetRef<Function>(func));
  }

 private:
  const Op call_pure_packed_op_ = Op::Get("relax.call_pure_packed");
  const Op call_inplace_packed_op_ = Op::Get("relax.call_inplace_packed");
  const Op invoke_pure_closure_op_ = Op::Get("relax.invoke_pure_closure");
  const Op invoke_closure_op_ = Op::Get("relax.invoke_closure");
};

Function RemovePurityChecking(const Function& f) { return PurityRemover().RemovePurity(f); }

namespace transform {

Pass RemovePurityChecking() {
  auto pass_func = [=](const Function& f, IRModule mod, PassContext pc) {
    return relax::RemovePurityChecking(f);
  };
  return CreateFunctionPass(pass_func, 0, "RemovePurityChecking");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.RemovePurityChecking", RemovePurityChecking);
}

}  // namespace transform

}  // namespace relax
}  // namespace tvm
