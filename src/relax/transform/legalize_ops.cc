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
 * \file tvm/relax/transform/legalize_ops.cc
 * \brief Legalize high-level operator calls in Relax functions to call_tir
 * with corresponding low-level TIR Functions.
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op_attr.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/transform.h>
#include <tvm/tirx/transform.h>

#include <set>

#include "utils.h"

namespace tvm {
namespace relax {

struct OpIdentityLess {
  bool operator()(const Op& lhs, const Op& rhs) const { return lhs.get() < rhs.get(); }
};

TVM_FFI_STATIC_INIT_BLOCK() {
  ::tvm::transform::PassContext::RegisterConfigOption<bool>("relax.transform.apply_legalize_ops");
}

class LegalizeMutator : public ExprMutator {
 public:
  explicit LegalizeMutator(const IRModule& mod,
                           const ffi::Optional<ffi::Map<ffi::String, ffi::Function>>& cmap,
                           const ffi::Optional<ffi::Array<ffi::String>> skip_ops,
                           bool enable_warning)
      : ExprMutator(mod), mod_(std::move(mod)), enable_warning_(enable_warning) {
    if (cmap) {
      cmap_ = cmap.value();
    }
    if (skip_ops.has_value()) {
      for (const auto name : skip_ops.value()) {
        skip_ops_.insert(Op::Get(name));
      }
    }
  }

  IRModule Transform() {
    for (const auto& gv : mod_->GetGlobalVars()) {
      const auto& func = mod_->Lookup(gv);
      if (func->IsInstance<FunctionObj>()) {
        auto updated_func = this->VisitExpr(func).as_or_throw<Function>();
        builder_->UpdateFunction(gv, updated_func.as_or_throw<BaseFunc>());
      }
    }

    IRModule output = builder_->GetContextIRModule();
    if (generated_tir_with_target_attr_) {
      // It is possible that every call to a legalized tirx::Function
      // contains VDevice annotations.  In that case, the tirx::Function
      // without a target annotation no longer has any callers, and
      // should be removed.
      output = relax::transform::DeadCodeElimination()(output);

      // Avoid accidental sharing of TIR variables in the legalized
      // Functions, when kernels for multiple devices are generated
      // from the same tirx::Function.
      output = s_tir::transform::ConvertSSA()(output);
    }

    return output;
  }

 private:
  using ExprMutator::VisitExpr_;

  bool WrapPureCondition(const Op& op, const Expr& legalized) {
    static const auto& purity_map = Op::GetAttrMap(tvm::relax::op_attr::kPurity);

    const CallObj* call = legalized.as<CallObj>();

    if (!call) {
      // Unlikely for this condition to be met, but it is possible.
      // For example, an operation could produce a Tuple output, and
      // be legalized into separate calls for each item in the Tuple.
      return false;
    }

    bool pure_original_op = purity_map.get(op, false);
    bool pure_legalized_op = [&]() -> bool {
      if (auto legalized_op = call->op.as<Op>()) {
        return purity_map.get(legalized_op.value(), false);
      } else if (auto func_ty = call->op->ty.as<FuncTypeObj>()) {
        return func_ty->purity;
      } else {
        return false;
      }
    }();

    // If the original op was pure, but the legalized op was not,
    // the legalized op may occur in a context that requires pure
    // functions, such as a `relax::DataflowBlock`.  In this case,
    // we should wrap the legalized operation to indicate that it is
    // still pure.
    return pure_original_op && !pure_legalized_op;
  }

  Call WrapPureCall(const Call& ret) {
    static const Op call_pure_packed_op = Op::Get("relax.call_pure_packed");
    ffi::Array<Expr> ret_args = {ret->op};
    for (auto arg : ret->args) {
      ret_args.push_back(arg);
    }
    return Call(Type::Missing(), call_pure_packed_op, ret_args, ret->attrs, ret->ty_args);
  }

  ffi::Optional<Target> GetTarget(const ffi::Array<Type>& types) {
    for (auto ty : types) {
      if (const auto* tinfo = ty.as<TensorTypeObj>()) {
        if (tinfo->vdevice.has_value()) {
          auto vdevice = tinfo->vdevice.value();
          if (vdevice->target.has_value()) {
            return vdevice->target;
          }
        }
      } else if (const auto* tup_ty = ty.as<TupleTypeObj>()) {
        return GetTarget(tup_ty->fields);
      }
    }
    return std::nullopt;
  }

  Expr BindTarget(Expr expr) {
    if (!expr->IsInstance<CallObj>()) {
      // FLegalize returned something other than a relax::Call.  This
      // post-processing only handles cases where legalization
      // produces a lowered call node.  In principle, this
      // post-processing isn't necessary, and FLegalize should already
      // have generated vdevice-aware kernels, so hopefully the
      // FLegalize implementation did so.
      return expr;
    }

    auto call = expr.as_or_throw<Call>();

    auto vdevice_target = GetTarget(call->ty_args);
    if (!vdevice_target.has_value()) {
      // No vdevice annotation is present, so we don't need to apply
      // any updates.
      return expr;
    }

    if (call->args.empty()) {
      return expr;
    }

    auto gvar = call->args[0].as<GlobalVar>();
    if (!gvar.has_value()) {
      // This is not a call into a legalized function within the
      // current IRModule, so no post-processing is required.
      return expr;
    }

    auto base_func = builder_->GetContextIRModule()->Lookup(gvar.value());
    auto opt_function = base_func.as<tirx::Function>();
    if (!opt_function) {
      // The call is to something other than a tirx::Function.  It may be
      // another Relax function, in which case the legalization of its
      // body will handle any additional target annotations.
      return expr;
    }
    auto function = opt_function.value();

    auto func_target = function->GetAttr(tvm::attr::kTarget);
    if (func_target && func_target.value()->kind == vdevice_target.value()->kind) {
      // The function already has compatible annotations for the
      // target, so no modifications are required.
      return expr;
    }

    // The FLegalize function generated a tirx::Function, but that tirx::Function
    // doesn't have annotations compatible with the vdevice required
    // by the Relax Type.  Update the call to instead call a
    // `tirx::Function` with the appropriate target annotation.  In the
    // future, this may be treated as a bug in the FLegalize
    // implementation, rather than expected output from it.
    auto new_function = WithAttr(function, tvm::attr::kTarget.name, vdevice_target.value());
    auto new_gvar_name = [&]() -> std::string {
      std::stringstream ss;
      ss << gvar.value()->name_hint;
      ss << "_";
      ss << vdevice_target.value()->kind->name;
      return ss.str();
    }();
    auto new_gvar = builder_->AddFunction(new_function, new_gvar_name);
    generated_tir_with_target_attr_ = true;

    call.CopyOnWrite()->args.Set(0, new_gvar);
    return call;
  }

  Expr VisitExpr_(const CallObj* call) final {
    Call visited_call = this->VisitExprPostOrder_(call).as_or_throw<Call>();
    static const auto& legalize_map = Op::GetAttrMap(tvm::relax::op_attr::kLegalize);
    static const auto& call_packed_map = Op::GetAttrMap(tvm::relax::op_attr::kCallPacked);
    static const auto& requires_arg_shapes_map =
        Op::GetAttrMap(tvm::relax::op_attr::kRequiresArgumentShapes);
    static const Op call_pure_packed_op = Op::Get("relax.call_pure_packed");
    static const Op call_tir_op = Op::Get("relax.call_tir");
    static const Op call_dps_packed_op = Op::Get("relax.call_dps_packed");
    auto* op_node = visited_call->op.as<OpObj>();

    // Not an OpObj
    if (op_node == nullptr) {
      return visited_call;
    }
    auto op = ffi::GetRef<Op>(op_node);

    if (skip_ops_.find(op) != skip_ops_.end()) {
      return visited_call;
    }

    // TODO(Lunderberg):
    //
    //     Improve the "argument shape not known" fallback case in
    //     `CanLegalizeCall`, as failure to legalize can produce
    //     unexpected errors during CodeGenVM.  This could be done by
    //     having `R.Tensor(ndim=2)` be syntactic sugar for
    //     `R.Tensor(shape=[m, n])`, where `m` and `n` are new shape
    //     variables.  This would allow legalization into dynamic TIR
    //     PrimFuncs.
    //
    //     This fallback would only be applicable for cases where both
    //     the dtype and the dimensionality are known.  While Relax can
    //     express a tensor with unknown dtype and dimensionality as
    //     `TensorType(DLDataType{kDLOpaqueHandle, 0, 0}, kUnknownNDim)`,
    //     TIR cannot express unknown dtype or unknown dimensionality.
    //
    // For data-dependent ops (e.g. `R.dynamic_strided_slice`), the
    // shape of the output depends on the runtime values stored in a
    // tensor, and `CanLegalizeCall` allows the return shape to remain
    // unknown. The Legalizer will insert their shape functions, which
    // are manually registered for each data-dependent op, and match
    // cast to define symbolic output shapes. These symbolic output
    // shapes at compile time can be used by later operations to refer
    // to the runtime shape.
    bool shapes_are_known_if_required = CanLegalizeCall(op, visited_call);

    FLegalize legalization_func;

    if (auto opt_custom_legalize = cmap_.Get(op->name);
        opt_custom_legalize && shapes_are_known_if_required) {
      // First choice, use a custom legalization function
      legalization_func = opt_custom_legalize.value();
    } else if (legalize_map.count(op) && shapes_are_known_if_required) {
      // Second choice, use a default legalization
      legalization_func = legalize_map[op];
    } else if (call_packed_map.count(op)) {
      // Third choice, use an explicit ffi::String replacement.  This does not require the shape
      ffi::String packed_func_name = call_packed_map[op];
      legalization_func = [packed_func_name](const BlockBuilder& bb, const Call& call) -> Expr {
        return Call(Type::Missing(), ExternFunc(packed_func_name), call->args, std::nullopt,
                    {GetType(call)});
      };
    } else {
      // No legalization.
      if (enable_warning_ && !op.same_as(call_tir_op) && !op.same_as(call_dps_packed_op) &&
          !op.same_as(call_pure_packed_op) && !op.same_as(Op::Get("relax.call_tir_packed"))) {
        if (shapes_are_known_if_required) {
          LOG(WARNING) << "No legalization func for " << op->name << " is found.";
        } else {
          LOG(WARNING) << "Cannot legalize " << visited_call
                       << ", missing known shapes for arguments and return value";
        }
      }
      return visited_call;
    }

    // The legalization function may call `builder_->Emit()` as part
    // of its implementation.  In that case, any operations it emits
    // must be caught such that they be checked for recursive
    // legalization.  This is done by wrapping the legalized value in
    // a SeqExpr, which can first be visited, then unwrapped by the
    // normalization.
    if (builder_->CurrentBlockIsDataFlow()) {
      builder_->BeginDataflowBlock();
    } else {
      builder_->BeginBindingBlock();
    }
    Expr legalized = legalization_func(builder_, visited_call);

    // Append the target attribute to any tirx::Function generated in
    // legalization.
    legalized = BindTarget(legalized);

    legalized = builder_->Normalize(legalized);

    BindingBlock prologue = builder_->EndBlock();
    for (const auto& binding : prologue->bindings) {
      VisitBinding(binding);
    }

    if (WrapPureCondition(op, legalized)) {
      legalized = WrapPureCall(legalized.as_or_throw<Call>());
    }

    // Legalization may have introduced additional operations that
    // must be legalized as well.  For example, a user-custom
    // intrinsic whose legalization is implemented in terms of relax
    // intrinsics.  The base case of the recursion occurs when no
    // additional legalization steps are found.
    //
    // Only perform recursive legalization when the legalization
    // function returned a modified expression, as some legalizations
    // return the original expression if they are unable to produce a
    // legalized version.
    if (!legalized.same_as(visited_call)) {
      legalized = VisitExpr(legalized);
    }

    return legalized;
  }

  /*! \brief The context IRModule. */
  IRModule mod_;
  /*! \brief The customized legalization function map. */
  ffi::Map<ffi::String, ffi::Function> cmap_;
  /*! \brief If VDevice annotations produced at least one tirx::Function with a Target attr*/
  bool generated_tir_with_target_attr_{false};
  /*!
   * \brief A boolean value indicating if to print warnings for CallObj whose op's
   * legalization function is not registered.
   */
  bool enable_warning_;
  /*!
   * \brief List of ops to be skipped from legalization
   */
  std::set<Op, OpIdentityLess> skip_ops_;
};

namespace transform {

Pass LegalizeOps(ffi::Optional<ffi::Map<ffi::String, ffi::Function>> cmap,
                 ffi::Optional<ffi::Array<ffi::String>> skip_ops, bool enable_warning) {
  auto pass_func = [=](IRModule mod, PassContext pc) {
    bool apply_legalize_ops =
        pc->GetConfig<bool>("relax.transform.apply_legalize_ops").value_or(true);
    if (apply_legalize_ops) {
      mod = LegalizeMutator(mod, cmap, skip_ops, enable_warning).Transform();
    }
    return mod;
  };
  return CreateModulePass(/*pass_function=*/pass_func,
                          /*opt_level=*/0,
                          /*pass_name=*/"LegalizeOps");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.LegalizeOps", LegalizeOps);
}

}  // namespace transform

}  // namespace relax
}  // namespace tvm
