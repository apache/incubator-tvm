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
 * \file tvm/relax/distributed/transform/lower_distir.cc
 * \brief Pass for lowering DistIR into Relax
 *  This pass assumes all the TensorIR functions are in local view,
 *  so the pass only handles sharding relax tensor shape and
 *  inserting necessary broadcast and scatter for inputs.
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/distributed/axis_group_graph.h>
#include <tvm/relax/distributed/transform.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op/ccl.h>
#include <tvm/tirx/stmt_functor.h>

#include "../../../s_tir/schedule/transform.h"
#include "../../op/ccl/ccl.h"
#include "../../op/tensor/manipulate.h"
#include "utils.h"

namespace tvm {
namespace relax {
using namespace tvm::prim;

namespace distributed {

class DistIRSharder : public ExprMutator {
 public:
  static IRModule LowerDistIR(IRModule mod) { return DistIRSharder(mod).Lower(); }

 private:
  explicit DistIRSharder(IRModule mod) : ExprMutator(mod) {}

  IRModule Lower() {
    auto mod = builder_->GetContextIRModule();
    for (const auto& [gv, base_func] : mod->functions) {
      const auto* func_ = base_func.as<FunctionObj>();
      if (func_ == nullptr || !IsDistIRFunc(ffi::GetRef<Function>(func_))) {
        continue;
      }
      Function func = RewriteFunction(ffi::GetRef<Function>(func_));
      builder_->UpdateFunction(gv, func);
    }
    return builder_->GetContextIRModule();
  }

  ShapeExpr ShardShape(ShapeExpr orig_shape, DeviceMesh device_mesh, Placement placement) {
    ffi::Shape device_mesh_shape = device_mesh->shape;
    ffi::Array<PrimExpr> new_tensor_shape_value = orig_shape->values;
    for (int i = 0; i < static_cast<int>(device_mesh_shape.size()); i++) {
      if (placement->dim_specs[i]->kind == PlacementSpecKind::kSharding) {
        int shard_size = device_mesh_shape[i];
        int axis = placement->dim_specs[i]->axis;
        new_tensor_shape_value.Set(axis, floordiv(orig_shape->values[axis], shard_size));
      }
    }
    return ShapeExpr(new_tensor_shape_value);
  }

  TensorType ShardDTensorType(DTensorType orig_ty) {
    TensorType tensor_ty = orig_ty->tensor_ty;
    TVM_FFI_ICHECK(tensor_ty->shape);
    const auto* orig_shape = tensor_ty->shape.as<ShapeExprObj>();
    auto new_tensor_ty = ffi::make_object<TensorTypeObj>(*tensor_ty.get());
    new_tensor_ty->shape =
        ShardShape(ffi::GetRef<ShapeExpr>(orig_shape), orig_ty->device_mesh, orig_ty->placement);
    return TensorType(new_tensor_ty);
  }

  Type ConvertType(Type orig_ty, bool shard_shape) {
    if (const auto* dtensor_ty = orig_ty.as<DTensorTypeObj>()) {
      if (shard_shape) {
        return ShardDTensorType(ffi::GetRef<DTensorType>(dtensor_ty));
      } else {
        return dtensor_ty->tensor_ty;
      }
    } else if (const auto* tuple_ty = orig_ty.as<TupleTypeObj>()) {
      ffi::Array<Type> new_fields;
      for (const auto& field_ty : tuple_ty->fields) {
        if (const auto* dtensor_ty = field_ty.as<DTensorTypeObj>()) {
          if (shard_shape) {
            new_fields.push_back(ShardDTensorType(ffi::GetRef<DTensorType>(dtensor_ty)));
          } else {
            new_fields.push_back(dtensor_ty->tensor_ty);
          }
        } else {
          new_fields.push_back(field_ty);
        }
      }
      return TupleType(new_fields);
    } else {
      return orig_ty;
    }
  }

  Expr ShardInputParamTensorAndConstant(Expr input) {
    TVM_FFI_ICHECK(!input->ty.as<MissingType>().has_value());
    Type old_ty = GetType(input);
    Type new_ty = ConvertType(old_ty, false);
    if (const auto* var = input.as<VarObj>()) {
      Var new_param(var->name, new_ty);
      return new_param;
    } else if (const auto* constant = input.as<GenericConstObj>()) {
      for (const auto& spec : old_ty.as_or_throw<DTensorType>()->placement->dim_specs) {
        TVM_FFI_ICHECK(spec->kind == PlacementSpecKind::kReplica);
      }
      GenericConst new_constant(new_ty, constant->value.cast<ffi::Tensor>());
      return new_constant;
    } else {
      TVM_FFI_THROW(InternalError)
          << "Cannot shard tensor which is not Var or GenericConst: " << input;
      throw;
    }
  }

  void EmitBroadcastOrScatter(Expr old_expr, Expr new_expr, DTensorType dtensor_ty) {
    // FIXME: this is a hack that only works for 1d device mesh
    TVM_FFI_ICHECK(dtensor_ty->device_mesh->shape.size() == 1);
    PlacementSpec sharding_spec = dtensor_ty->placement->dim_specs[0];
    if (sharding_spec->kind == PlacementSpecKind::kReplica) {
      Var new_var = builder_->Emit(broadcast_from_worker0(new_expr));
      if (const auto* var = old_expr.as<VarObj>()) {
        var_remap_.insert_or_assign(ffi::GetRef<Var>(var), new_var);
      } else {
        tuple_getitem_remap_.insert_or_assign(old_expr.as_or_throw<TupleGetItem>(), new_var);
      }
    } else if (sharding_spec->kind == PlacementSpecKind::kSharding) {
      Var scatter_var = builder_->Emit(
          scatter_from_worker0(new_expr, dtensor_ty->device_mesh->shape[0], sharding_spec->axis));
      if (const auto* var = old_expr.as<VarObj>()) {
        var_remap_.insert_or_assign(ffi::GetRef<Var>(var), scatter_var);
      } else {
        tuple_getitem_remap_.insert_or_assign(old_expr.as_or_throw<TupleGetItem>(), scatter_var);
      }
    } else {
      TVM_FFI_THROW(InternalError) << "Unsupported placement spec";
    }
  }

  void InputPreprocessing() {
    for (int i = 0; i < static_cast<int>(func_->params.size()); i++) {
      Var param = func_->params[i];
      if (const auto* dtensor_ty = GetTypeAs<DTensorTypeObj>(param)) {
        EmitBroadcastOrScatter(param, new_params_[i], ffi::GetRef<DTensorType>(dtensor_ty));
      } else if (const auto* tuple_ty = GetTypeAs<TupleTypeObj>(param)) {
        for (int j = 0; j < static_cast<int>(tuple_ty->fields.size()); j++) {
          if (const auto* dtensor_ty = tuple_ty->fields[j].as<DTensorTypeObj>()) {
            EmitBroadcastOrScatter(TupleGetItem(param, j), TupleGetItem(new_params_[i], j),
                                   ffi::GetRef<DTensorType>(dtensor_ty));
          }
        }
      }
    }
  }

  Function RewriteFunction(Function func) {
    ffi::Array<Var> new_params;
    for (const Var& var : func->params) {
      Var new_param = ShardInputParamTensorAndConstant(var).as_or_throw<Var>();
      var_remap_.insert_or_assign(var, new_param);
      new_params.push_back(new_param);
    }
    func_ = func;
    new_params_ = new_params;
    auto new_body = VisitWithNewScope(func->body, new_params);
    Function new_func(new_params, new_body, std::nullopt, func->is_pure, func->attrs);
    return new_func;
  }

  void VisitBinding_(const VarBindingObj* binding, const TupleGetItemObj* val) {
    if (tuple_getitem_remap_.count(ffi::GetRef<TupleGetItem>(val))) {
      var_remap_.insert_or_assign(binding->var,
                                  tuple_getitem_remap_.at(ffi::GetRef<TupleGetItem>(val)));
    } else {
      ExprMutator::VisitBinding_(binding, val);
    }
  }

  BindingBlock VisitBindingBlock_(const BindingBlockObj* block) {
    builder_->BeginBindingBlock();
    InputPreprocessing();
    for (Binding binding : block->bindings) {
      this->VisitBinding(binding);
    }
    return builder_->EndBlock();
  }

  BindingBlock VisitBindingBlock_(const DataflowBlockObj* block) {
    builder_->BeginDataflowBlock();
    InputPreprocessing();
    for (auto binding : block->bindings) {
      this->VisitBinding(binding);
    }
    return builder_->EndBlock();
  }

  Call HandleSpecialCaseinDTensorLowering(const CallObj* call, Var binding_var) {
    static Op reshape_op = Op::Get("relax.reshape");
    static Op call_tir_op = Op::Get("relax.call_tir");
    static Op call_tir_local_view_op = Op::Get("relax.dist.call_tir_local_view");
    if (call->op.same_as(reshape_op)) {
      TVM_FFI_ICHECK(call->args[1].as<ShapeExprObj>());
      const auto* out_ty = GetTypeAs<DTensorTypeObj>(binding_var);
      TVM_FFI_ICHECK(out_ty);
      auto new_call_node = ffi::make_object<CallObj>(*call);
      new_call_node->args.Set(1, ShardShape(call->args[1].as_or_throw<ShapeExpr>(),
                                            out_ty->device_mesh, out_ty->placement));
      return Call(new_call_node);
    } else if (call->op.same_as(call_tir_local_view_op)) {
      auto new_call_node = ffi::make_object<CallObj>(*call);
      new_call_node->op = call_tir_op;
      new_call_node->ty_args = {ConvertType(GetType(binding_var), true)};
      return Call(new_call_node);
    } else if (call->op.same_as(call_tir_op)) {
      TVM_FFI_THROW(InternalError)
          << "call_tir should be lowered to call_tir_local_view before lowering to relax";
    } else if (const auto* extern_func = call->op.as<ExternFuncObj>()) {
      auto new_call_node = ffi::make_object<CallObj>(*call);
      if (extern_func->global_symbol == "vm.builtin.distributed.attention_kv_cache_append") {
        new_call_node->op = ExternFunc("vm.builtin.attention_kv_cache_append");
      } else if (extern_func->global_symbol == "vm.builtin.distributed.attention_kv_cache_view") {
        new_call_node->op = ExternFunc("vm.builtin.attention_kv_cache_view");
        auto orig_shape = call->args[1].as_or_throw<ShapeExpr>();
        const auto* out_ty = GetTypeAs<DTensorTypeObj>(binding_var);
        TVM_FFI_ICHECK(out_ty);
        ShapeExpr new_shape = ShardShape(orig_shape, out_ty->device_mesh, out_ty->placement);
        new_call_node->args.Set(1, new_shape);
        new_call_node->ty_args = {TensorType(new_shape, out_ty->tensor_ty->dtype)};
      }
      return Call(new_call_node);
    }
    return ffi::GetRef<Call>(call);
  }

  void VisitBinding_(const VarBindingObj* binding, const CallObj* val) {
    Call new_call =
        this->VisitExpr(HandleSpecialCaseinDTensorLowering(val, binding->var)).as_or_throw<Call>();
    ReEmitBinding(binding, builder_->Normalize(new_call));
  }

  Function func_{ffi::UnsafeInit{}};
  ffi::Array<Var> new_params_;
  std::unordered_map<TupleGetItem, Var, ffi::StructuralHash, ffi::StructuralEqual>
      tuple_getitem_remap_;
};

namespace transform {

Pass LowerDistIR() {
  auto pass_func = [=](IRModule m, PassContext pc) { return DistIRSharder::LowerDistIR(m); };
  return CreateModulePass(pass_func, 1, "LowerDistIR");
}
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.distributed.transform.LowerDistIR", LowerDistIR);
}
}  // namespace transform

}  // namespace distributed
}  // namespace relax
}  // namespace tvm
