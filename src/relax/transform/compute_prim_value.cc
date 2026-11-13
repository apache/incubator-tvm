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

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/ir/function.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/op.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op_attr.h>
#include <tvm/relax/transform.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm {
namespace relax {

namespace {

bool HasRelaxCallCapabilities(const CallObj* call) {
  auto op = call->op.as<Op>();
  if (!op) return true;
  static auto infer_type_map = Op::GetAttrMap(tvm::op_attr::kInferType);
  static auto infer_type_with_builder_map =
      Op::GetAttrMap(tvm::relax::op_attr::kInferTypeWithBuilder);
  static auto legalize_map = Op::GetAttrMap(tvm::relax::op_attr::kLegalize);
  return infer_type_map.count(op.value()) || infer_type_with_builder_map.count(op.value()) ||
         legalize_map.count(op.value());
}

class PrimExprComputeInjector : public ExprMutator {
 public:
  IRModule Finalize() const { return builder_->Finalize(); }

 private:
  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const CallObj* op) final {
    Call call = ffi::GetRef<Call>(op);
    if (auto prim_expr = call.as<PrimExpr>()) {
      if (call->op.as<Op>() && !HasRelaxCallCapabilities(op)) {
        return LiftPrimValue(prim_expr.value());
      }
    }
    return ExprMutator::VisitExpr_(op);
  }

#define RELAX_LIFT_PRIM_EXPR(OP)                                         \
  Expr VisitExpr_(const OP* op) final {                                  \
    return LiftPrimValue(ffi::GetRef<Expr>(op).as_or_throw<PrimExpr>()); \
  }

  RELAX_LIFT_PRIM_EXPR(TensorLoadObj);
  RELAX_LIFT_PRIM_EXPR(prim::AddObj);
  RELAX_LIFT_PRIM_EXPR(prim::LShiftObj);
  RELAX_LIFT_PRIM_EXPR(prim::RShiftObj);
  RELAX_LIFT_PRIM_EXPR(prim::BitwiseAndObj);
  RELAX_LIFT_PRIM_EXPR(prim::BitwiseOrObj);
  RELAX_LIFT_PRIM_EXPR(prim::BitwiseXorObj);
  RELAX_LIFT_PRIM_EXPR(prim::BitwiseNotObj);
  RELAX_LIFT_PRIM_EXPR(prim::SubObj);
  RELAX_LIFT_PRIM_EXPR(prim::MulObj);
  RELAX_LIFT_PRIM_EXPR(prim::DivObj);
  RELAX_LIFT_PRIM_EXPR(prim::ModObj);
  RELAX_LIFT_PRIM_EXPR(prim::FloorDivObj);
  RELAX_LIFT_PRIM_EXPR(prim::FloorModObj);
  RELAX_LIFT_PRIM_EXPR(prim::MinObj);
  RELAX_LIFT_PRIM_EXPR(prim::MaxObj);
  RELAX_LIFT_PRIM_EXPR(prim::EQObj);
  RELAX_LIFT_PRIM_EXPR(prim::NEObj);
  RELAX_LIFT_PRIM_EXPR(prim::LTObj);
  RELAX_LIFT_PRIM_EXPR(prim::LEObj);
  RELAX_LIFT_PRIM_EXPR(prim::GTObj);
  RELAX_LIFT_PRIM_EXPR(prim::GEObj);
  RELAX_LIFT_PRIM_EXPR(prim::AndObj);
  RELAX_LIFT_PRIM_EXPR(prim::OrObj);
  RELAX_LIFT_PRIM_EXPR(prim::CastObj);
  RELAX_LIFT_PRIM_EXPR(prim::NotObj);
  RELAX_LIFT_PRIM_EXPR(prim::SelectObj);
  RELAX_LIFT_PRIM_EXPR(prim::RampObj);
  RELAX_LIFT_PRIM_EXPR(prim::BroadcastObj);
  RELAX_LIFT_PRIM_EXPR(prim::ShuffleObj);
  RELAX_LIFT_PRIM_EXPR(tvm::IntImmObj);
  RELAX_LIFT_PRIM_EXPR(tvm::FloatImmObj);

#undef RELAX_LIFT_PRIM_EXPR

  Expr VisitExpr_(const ShapeExprObj* op) final { return ffi::GetRef<Expr>(op); }

  PrimExpr VisitTypePrimExprField(const PrimExpr& expr) final { return expr; }

  Expr LiftPrimValue(const PrimExpr& node) {
    if (node->IsInstance<IntImmObj>() || node->IsInstance<VarObj>()) {
      return node;
    }

    tvm::PrimType ret_ty = node.ty();
    auto param_vars = tvm::UndefinedVars(node);
    tvm::Stmt body = tvm::Return(node);

    tirx::Function func(param_vars, tvm::SeqStmt(body), ret_ty,
                        DictAttrs({{tvm::tirx::attr::kIsHostFunc.name, true},
                                   {tvm::attr::kScriptNamespace.name, ffi::String("s_tir")}}));
    func = tirx::RenewDef(func);

    auto callee = builder_->AddFunction(func, "compute_symbolic_expr");

    Tuple args(param_vars.Map(
        [](const tvm::Var& tir_var) -> relax::Expr { return tir_var.as_or_throw<PrimExpr>(); }));
    const Op& bridge = Op::Get("relax.call_tir_packed");
    if (s_tir::IsPureFunction(func)) {
      return Call(Type::Missing(), Op::Get("relax.call_pure_packed"), {bridge, callee, args});
    }
    return Call(Type::Missing(), bridge, {callee, args});
  }
};

}  // namespace

namespace transform {

Pass ComputePrimValue() {
  auto pass_func = [=](IRModule mod, PassContext pc) -> IRModule {
    PrimExprComputeInjector mutator;

    IRModule updates;
    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto func = base_func.as<Function>()) {
        auto updated = mutator(func.value()).as_or_throw<Function>();
        if (!updates.same_as(base_func)) {
          updates->Add(gvar, updated);
        }
      }
    }

    if (updates->functions.size()) {
      auto write_ptr = mod.CopyOnWrite();
      write_ptr->Update(updates);
      write_ptr->Update(mutator.Finalize());
    }

    return mod;
  };
  return CreateModulePass(pass_func, 0, "ComputePrimValue");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.ComputePrimValue", ComputePrimValue);
}

}  // namespace transform

}  // namespace relax
}  // namespace tvm
