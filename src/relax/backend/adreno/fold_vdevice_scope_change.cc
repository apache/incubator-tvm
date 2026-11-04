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
 * \file src/relax/backend/adreno/fold_vdevice_scope_change.cc
 * \brief This is a texture specific pass that can optimize unnecessary to_device copies.
 * Like texture_scope -> ToVDevice -> global scope. In this case the producer can directly
 * store into global scope avoiding unnecessary device copy.
 */

#include <tvm/ffi/cast.h>
#include <tvm/relax/backend/adreno/transform.h>
#include <tvm/relax/dataflow_matcher.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/nested_msg.h>
#include <tvm/relax/op/op.h>
#include <tvm/relax/op_attr.h>
#include <tvm/tirx/index_map.h>

#include <tuple>

#include "../../../s_tir/backend/adreno/texture_utils.h"
#include "../../op/tensor/manipulate.h"
#include "../../transform/infer_layout_utils.h"
#include "../../transform/utils.h"

namespace tvm {
namespace relax {
namespace backend {
namespace adreno {

namespace {
std::tuple<DFPattern, ffi::TypedFunction<Expr(Expr, ffi::Map<DFPattern, Expr>)>> CreatePatterns(
    ffi::Map<Expr, ffi::Array<Expr>> consumers) {
  auto pat_gv = WildcardPattern();

  auto pat_inp = WildcardPattern();
  auto pat_call_tir = IsOp("relax.call_tir")(pat_gv, pat_inp);
  auto pattern_out = IsOp("relax.to_vdevice")(pat_call_tir);

  auto rewriter = [=](Expr expr, ffi::Map<DFPattern, Expr> matches) -> Expr {
    const auto* call_tir = matches[pat_call_tir].as<CallObj>();
    TVM_FFI_CHECK(call_tir, InternalError)
        << "Match of relax.call_tir operator should produce Call, "
        << "but instead produces " << matches[pat_call_tir] << " with type "
        << matches[pat_call_tir]->GetTypeKey();

    const auto* out = matches[pattern_out].as<CallObj>();
    TVM_FFI_CHECK(out, InternalError) << "Match of relax.to_vdevice operator should produce Call, "
                                      << "but instead produces " << matches[pattern_out]
                                      << " with type " << matches[pattern_out]->GetTypeKey();

    const auto* vdev_attrs = out->attrs.as<ToVDeviceAttrs>();
    TVM_FFI_CHECK(vdev_attrs, InternalError)
        << "Attributes for relax.to_vdevice operator should be ToVDeviceAttrs, "
        << "but were instead " << out->attrs << " with type " << out->GetTypeKey();

    const auto* tir_out_ty = call_tir->ty_args[0].as<TensorTypeObj>();
    if (!tir_out_ty) return expr;

    if (!tir_out_ty->vdevice.has_value()) return expr;

    const VarObj* arg_var = out->args[0].as<VarObj>();
    if (consumers.find(ffi::GetRef<Expr>(arg_var)) != consumers.end()) {
      if (consumers[ffi::GetRef<Expr>(arg_var)].size() > 1) {
        /* Don't do to_device optimization as we are not the only consumer */
        return expr;
      }
    }

    if (texture::IsTextureStorage(tir_out_ty->vdevice.value()->memory_scope) &&
        (vdev_attrs->dst_vdevice->memory_scope == runtime::MemoryScope::Global())) {
      auto shape_arr = tir_out_ty->GetShape().value();
      auto new_ty = TensorType(ShapeExpr(shape_arr), tir_out_ty->dtype, vdev_attrs->dst_vdevice);

      return Call(Type::Missing(), call_tir->op, call_tir->args, call_tir->attrs, {new_ty});
    }
    return expr;
  };

  return {pattern_out, rewriter};
}

}  // namespace

class CollectConsumerDetails : public ExprVisitor {
 public:
  explicit CollectConsumerDetails(Target target) : target_(std::move(target)) {}
  using ExprVisitor::VisitExpr_;

  ffi::Map<Expr, ffi::Array<Expr>> Collect(const IRModule& mod, Function func,
                                           const Target& target) {
    mod_ = mod;
    target_ = target;
    VisitExpr(func->body);
    // Extend the consumer details for tuple items
    for (const auto& val : arg_to_binding) {
      if (consumers.find(val.first) != consumers.end()) {
        if (consumers.find(val.second) == consumers.end()) {
          consumers.Set(val.second, consumers[val.first]);
        } else {
          auto ent = consumers[val.second];
          for (auto ent_val : consumers[val.first]) {
            ent.push_back(ent_val);
          }
          consumers.Set(val.second, ent);
        }
      }
    }
    return consumers;
  }

  void VisitBinding_(const VarBindingObj* binding,
                     const TupleGetItemObj* tuple_get_item_node) final {
    if (arg_to_binding.find(ffi::GetRef<Expr>(binding->var.get())) == arg_to_binding.end()) {
      arg_to_binding.Set(ffi::GetRef<Expr>(binding->var.get()),
                         ffi::GetRef<Expr>(tuple_get_item_node->tuple.get()));
    }
  }

  void VisitExpr_(const CallObj* call) final {
    static const Op call_tir_op = Op::Get("relax.call_tir");
    Tuple func_args =
        call->op.same_as(call_tir_op) ? call->args[1].as_or_throw<Tuple>() : Tuple(call->args);

    for (auto arg : func_args->fields) {
      auto ty = GetType(arg);
      if (auto tensor_ty = ty.as<TensorType>()) {
        ffi::Array<Expr> call_list;

        const VarObj* arg_var = arg.as<VarObj>();

        if (consumers.find(ffi::GetRef<Expr>(arg_var)) != consumers.end()) {
          call_list = consumers[ffi::GetRef<Expr>(arg_var)];
        }
        call_list.push_back(ffi::GetRef<Expr>(call));
        consumers.Set(ffi::GetRef<Expr>(arg_var), call_list);
      }
    }
  }

 private:
  /* Map of each Var consumption by a call node */
  ffi::Map<Expr, ffi::Array<Expr>> consumers;
  ffi::Map<Expr, Expr> arg_to_binding;
  IRModule mod_;
  Target target_;
};

namespace transform {

Pass FoldVDeviceScopeChange() {
  auto pass_func = [=](Function func, IRModule mod, PassContext pc) {
    /* here Target doesn't matter as the consumers we use only to find multiple consumers */
    auto consumers = CollectConsumerDetails(Target("opencl"))
                         .Collect(mod, func.as_or_throw<Function>(), Target("opencl"));
    auto [pattern, rewriter] = CreatePatterns(consumers);
    return RewriteCall(pattern, rewriter, func);
  };
  return CreateFunctionPass(pass_func, 1, "FoldVDeviceScopeChange");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.backend.adreno.transform.FoldVDeviceScopeChange",
                        FoldVDeviceScopeChange);
}
}  // namespace transform
}  // namespace adreno
}  // namespace backend
}  // namespace relax
}  // namespace tvm
