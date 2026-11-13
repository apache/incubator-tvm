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
 * \file src/relax/transform/attach_attr_layout_free_tensors.cc
 * \brief Attach layout_free_tensors for layout-free tensors.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/s_tir/function.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm {
namespace relax {

class AttrAttacher : public ExprMutator {
 public:
  static IRModule Transform(const IRModule& mod) {
    AttrAttacher mutator(mod);
    for (auto [gvar, func] : mod->functions) {
      if (func->IsInstance<relax::FunctionObj>()) {
        // clear the layout_free_exprs_ for each function
        mutator.layout_free_exprs_.clear();
        mutator.builder_->UpdateFunction(gvar, mutator.VisitExpr(func).as_or_throw<BaseFunc>());
      }
    }
    return mutator.builder_->GetContextIRModule();
  }

 private:
  explicit AttrAttacher(IRModule mod) : ExprMutator(mod), mod_(mod) {}

  using ExprMutator::VisitExpr_;
  Expr VisitExpr_(const FunctionObj* op) final {
    if (auto opt_num_input = op->attrs.GetAttr(tvm::relax::attr::kNumInput)) {
      TVM_FFI_ICHECK(layout_free_exprs_.empty())
          << "meet a non-global function with num_input attr";
      size_t num_input = opt_num_input.value();
      for (size_t i = num_input; i < op->params.size(); i++) {
        layout_free_exprs_.insert(op->params[i].get());
      }
    }
    return ExprMutator::VisitExpr_(op);
  }

  Expr VisitExpr_(const GenericConstObj* op) final {
    if (op->value.as<ffi::TensorObj>()) layout_free_exprs_.insert(op);
    return ExprMutator::VisitExpr_(op);
  }

  Expr VisitExpr_(const CallObj* op) final {
    static const Op call_tir_op_ = Op::Get("relax.call_tir");
    Call call = ExprMutator::VisitExpr_(op).as_or_throw<Call>();
    if (!call->op.same_as(call_tir_op_)) {
      return call;
    }
    GlobalVar gv = call->args[0].as_or_throw<GlobalVar>();
    ffi::Array<Expr> call_tir_args = call->args[1].as_or_throw<Tuple>()->fields;
    // Compute the layout free tensors
    ffi::Array<int64_t> layout_free_tensors;
    for (size_t i = 0; i < call_tir_args.size(); i++) {
      if (layout_free_exprs_.count(call_tir_args[i].get())) {
        layout_free_tensors.push_back(i);
      }
    }
    // Attach the layout free tensors to the tirx::Function
    tirx::Function func = WithAttr(mod_->Lookup(gv).as_or_throw<tirx::Function>(),
                                   tvm::s_tir::attr::kLayoutFreeTensors.name, layout_free_tensors);
    // Renew defs
    func = tirx::RenewDef(func);
    // Add the updated tirx::Function in the IRModule
    // Note the blockbuilder would automatically combine the same tirx function
    // So we don't need to worry about the duplicate insertion
    GlobalVar new_gv = builder_->AddFunction(func, gv->name_hint);
    // Create a new call node with the updated tirx::Function
    auto n = ffi::make_object<CallObj>(*op);
    n->args = {new_gv, Tuple(call_tir_args)};
    return Call(n);
  }

 private:
  IRModule mod_;
  std::unordered_set<const ExprObj*> layout_free_exprs_;
};
namespace transform {

Pass AttachAttrLayoutFreeTensors() {
  auto pass_func = [=](IRModule mod, PassContext pc) { return AttrAttacher::Transform(mod); };
  auto pass = CreateModulePass(pass_func, 0, "_AttachAttrLayoutFreeTensors");
  // Apply DeadCodeElimination to remove unused tirx::Function
  return tvm::transform::Sequential({pass, DeadCodeElimination()}, "AttachAttrLayoutFreeTensors");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.AttachAttrLayoutFreeTensors", AttachAttrLayoutFreeTensors);
}
}  // namespace transform
}  // namespace relax
}  // namespace tvm
