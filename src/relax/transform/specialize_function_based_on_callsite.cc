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
 * \file src/relax/transform/specialize_tir_params.cc
 * \brief Update tirx::Function tensors based on updated scope (or structure) info.
 */

#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/nested_msg.h>
#include <tvm/relax/op/op.h>
#include <tvm/relax/op_attr.h>
#include <tvm/relax/transform.h>
#include <tvm/tirx/index_map.h>
#include <tvm/tirx/op/memory.h>

#include <tuple>

#include "../op/tensor/manipulate.h"
#include "infer_layout_utils.h"
#include "utils.h"

namespace tvm {
namespace relax {
using namespace tvm::prim;

using tvm::tirx::TensorVar;

static ffi::Array<PrimExpr> GetShapeFromTensorType(const TensorType& tensor_ty) {
  auto shape = tensor_ty->GetShape();
  TVM_FFI_ICHECK(shape.has_value());
  return shape.value();
}

class SpecializeTIRCallArgs : ExprMutator {
 public:
  IRModule Run(IRModule mod) {
    mod_ = mod;
    for (const auto& [gv, func] : mod->functions) {
      if (func->IsInstance<relax::FunctionObj>()) {
        const auto& base_func = mod->Lookup(gv);
        // Only non primitive relax functions
        if (base_func->HasNonzeroAttr(tvm::relax::attr::kPrimitive.name)) {
          continue;
        }
        relax::Function update_func = VisitExpr(func).as_or_throw<Function>();
        updates_->Add(gv, update_func);
      }
    }
    mod_.CopyOnWrite()->Update(updates_);
    return mod_;
  }

  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const CallObj* call_node) override {
    auto call = ExprMutator::VisitExpr_(call_node).as_or_throw<Call>();
    static const Op call_tir_op = Op::Get("relax.call_tir");
    if (call->op.same_as(call_tir_op)) {
      return SpecializeTirFunction(call);
    }
    return call;
  }

 private:
  Expr SpecializeTirFunction(Call call) {
    auto gv = call->args[0].as_or_throw<GlobalVar>();
    auto pfunc = mod_->Lookup(gv).as_or_throw<tirx::Function>();
    auto args = call->args[1].as_or_throw<Tuple>()->fields;
    ffi::Map<tvm::Var, TensorVar> param_map;

    for (size_t i = 0; i < args.size(); ++i) {
      auto ty = GetType(args[i]);
      TVM_FFI_ICHECK(ty->IsInstance<TensorTypeObj>())
          << "Expected Tensor struct Info for call :" << call->op;
      auto tensor_ty = ty.as_or_throw<TensorType>();
      TVM_FFI_ICHECK(tensor_ty->shape.has_value()) << "Shape undefined for call:" << call->args[0];
      runtime::MemoryScope scope;
      if (tensor_ty->vdevice.has_value()) {
        scope = tensor_ty->vdevice.value()->memory_scope;
      }
      ffi::String name;
      if (args[i]->IsInstance<tvm::VarObj>()) {
        name = args[i].as_or_throw<Var>()->name;
      } else {
        name = std::string({static_cast<char>('A' + i)});
      }

      const TensorVar& tensor = tirx::TensorVar(
          name,
          tirx::TensorType(GetShapeFromTensorType(tensor_ty), tensor_ty->dtype.value(), scope));
      param_map.Set(pfunc->params[i], tensor);
    }
    runtime::MemoryScope scope;
    auto out_ty = call->ty_args[0];
    if (out_ty->IsInstance<TensorTypeObj>()) {
      auto ty = out_ty.as_or_throw<TensorType>();
      if (ty->vdevice.has_value()) {
        scope = ty->vdevice.value()->memory_scope;
      }
      const TensorVar& tensor = tirx::TensorVar(
          "ret_val", tirx::TensorType(GetShapeFromTensorType(ty), ty->dtype.value(), scope));
      param_map.Set(pfunc->params[pfunc->params.size() - 1], tensor);
    } else {
      TVM_FFI_ICHECK(out_ty->IsInstance<TupleTypeObj>())
          << "Expect output type of call_tir to be either TupleType or "
             "TensorType, but got "
          << out_ty;

      const auto& tuple_ty = out_ty.as_or_throw<TupleType>();
      ffi::Array<Type> ty_fields;
      int index = 0;
      for (const auto& si : tuple_ty->fields) {
        TVM_FFI_ICHECK(si->IsInstance<TensorTypeObj>())
            << "Fields of TupleType must be TensorType for call_tir "
               "output structinfo, but got "
            << si;
        auto ty = si.as_or_throw<TensorType>();
        scope = runtime::MemoryScope();
        if (ty->vdevice.has_value()) {
          scope = ty->vdevice.value()->memory_scope;
        }

        const TensorVar& tensor =
            tirx::TensorVar("ret_val_" + std::to_string(index),
                            tirx::TensorType(GetShapeFromTensorType(ty), ty->dtype.value(), scope));
        param_map.Set(pfunc->params[args.size() + index], tensor);
        index++;
      }
    }

    // call_tir supplies dense tensor descriptors. Bind their symbolic dimensions
    // explicitly, retaining each parameter's access contract and storage scope.
    sym::Analyzer analyzer;
    ffi::Map<tvm::Var, Expr> replacements;
    auto bind = [&](const PrimExpr& symbolic, const PrimExpr& concrete) {
      if (analyzer->CanProveEqual(symbolic, concrete)) return;
      auto var = symbolic.as<PrimVar>();
      TVM_FFI_CHECK(var.has_value() && !replacements.count(var.value()), ValueError)
          << "Inconsistent call_tir tensor dimension: " << symbolic << " vs. " << concrete;
      replacements.Set(var.value(), concrete);
      auto context = analyzer->GetConstraintContext();
      context->Append(sym::BindingConstraint(var.value(), concrete));
      analyzer->SyncConstraintContext(context);
    };
    for (const auto& param : pfunc->params) {
      auto descriptor = param_map.Get(param);
      if (!descriptor) continue;
      auto tensor = param.as_or_throw<TensorVar>();
      auto actual = descriptor.value();
      TVM_FFI_CHECK_EQ(tensor.shape().size(), actual.shape().size(), ValueError);
      TVM_FFI_CHECK(tensor.ty()->dtype == actual.ty()->dtype, ValueError);
      for (size_t i = 0; i < tensor.shape().size(); ++i) {
        bind(tensor.shape()[i], actual.shape()[i]);
      }
      auto source = tirx::MatchShapeStridePattern(tensor.ty());
      auto target = tirx::MatchShapeStridePattern(actual.ty());
      TVM_FFI_CHECK(source.has_value() && target.has_value(), ValueError)
          << "call_tir requires shape-aligned strided tensor layouts";
      for (size_t i = 0; i < source.value().get<1>().size(); ++i) {
        bind(source.value().get<1>()[i], target.value().get<1>()[i]);
      }
      bind(source.value().get<2>(), target.value().get<2>());
      // Preserve layout and alignment; the Relax descriptor only chooses scope.
      auto ty = tensor.ty();
      replacements.Set(param, TensorVar(param->name,
                                        tirx::TensorType(ty->shape, ty->dtype, actual.scope(),
                                                         ty->data_alignment, ty->layout, ty->loc),
                                        param->loc));
    }
    ffi::Array<tvm::Var> params;
    for (const auto& param : pfunc->params) {
      if (!param.as<PrimVar>() || !replacements.count(param)) params.push_back(param);
    }
    auto signature = tirx::Function(params, pfunc->body, pfunc->ret_type, pfunc->attrs, pfunc->loc);
    auto mapped = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(
        signature,
        [&](const tvm::Var& var) -> Expr { return replacements.Get(var).value_or(var); });
    auto new_pfunc = ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(
                         mapped,
                         [](const tvm::Call& call) {
                           if (!call->op.same_as(tirx::tensor_data_ptr_op())) return call;
                           // Pointer projections inherit the newly bound tensor storage scope.
                           auto copy = ffi::make_object<tvm::CallObj>(*call.get());
                           copy->ty = tvm::Call::ReinferType(copy.get());
                           return tvm::Call(std::move(copy));
                         },
                         [&](const PrimExpr& expr) { return analyzer->Simplify(expr); })
                         .as_or_throw<tirx::Function>();
    auto new_function =
        WithAttr(new_pfunc, tvm::relax::attr::kScoped.name, static_cast<int64_t>(1));
    updates_->Add(gv, new_function);
    return call;
  }
  IRModule mod_;
  IRModule updates_;
};

namespace transform {

Pass SpecializeFunctionBasedOnCallSite() {
  auto pass_func = [=](IRModule mod, PassContext pc) {
    return relax::SpecializeTIRCallArgs().Run(mod);
  };
  return CreateModulePass(/*pass_function=*/pass_func,
                          /*opt_level=*/0,
                          /*pass_name=*/"SpecializeFunctionBasedOnCallSite");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.SpecializeFunctionBasedOnCallSite",
                        SpecializeFunctionBasedOnCallSite);
}
}  // namespace transform
}  // namespace relax
}  // namespace tvm
