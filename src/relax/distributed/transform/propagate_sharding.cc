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
 * \file tvm/relax/distributed/transform/propagate_sharding.cc
 * \brief Pass for propagating sharding information.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/distributed/axis_group_graph.h>
#include <tvm/relax/distributed/transform.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op/distributed.h>
#include <tvm/relax/op/linear_algebra.h>
#include <tvm/relax/op/manipulate.h>
#include <tvm/relax/op/statistical.h>

#include <numeric>

#include "../../op/distributed/distributed.h"
#include "../../op/distributed/utils.h"
#include "utils.h"
namespace tvm {
namespace relax {
namespace distributed {

void CollectAxisGraphBinary(const VarBindingObj* binding, const CallObj* call,
                            AxisGroupGraph* axis_group_graph) {
  const std::vector<std::string> binary_op_names = {
      "add",     "subtract",      "multiply", "divide",     "power",     "floor_divide", "equal",
      "greater", "greater_equal", "less",     "less_equal", "not_equal", "minimum",      "maximum"};
  for (const auto& op_name : binary_op_names) {
    const Op binary_op = Op::Get("relax." + op_name);
    if (call->op.same_as(binary_op)) {
      BuildAxisGraphBinary(binding->var, ffi::GetRef<Call>(call), axis_group_graph);
      break;
    }
  }
}

void CollectAxisGraphUnary(const VarBindingObj* binding, const CallObj* call,
                           AxisGroupGraph* axis_group_graph) {
  const std::vector<std::string> unary_op_names = {
      "abs",    "acos",     "acosh",
      "asin",   "asinh",    "atan",
      "atanh",  "ceil",     "cos",
      "cosh",   "exp",      "floor",
      "log",    "negative", "nn.relu",
      "round",  "rsqrt",    "sigmoid",
      "sign",   "sin",      "sinh",
      "square", "sqrt",     "tan",
      "tanh",   "clip",     "isfinite",
      "isinf",  "isnan",    "dist.annotate_sharding",
      "erf",    "nn.gelu",  "builtin.stop_lift_params"};
  for (const auto& op_name : unary_op_names) {
    const Op unary_op = Op::Get("relax." + op_name);
    if (call->op.same_as(unary_op)) {
      BuildAxisGraphUnary(binding->var, ffi::GetRef<Call>(call), axis_group_graph);
    }
  }
}

void CollectAxisGraphReduce(const VarBindingObj* binding, const CallObj* call,
                            AxisGroupGraph* axis_group_graph) {
  const std::vector<std::string> reduction_op_names = {"sum",  "max", "min",      "prod",
                                                       "mean", "std", "variance", "nn.softmax"};
  for (const auto& op_name : reduction_op_names) {
    const Op reduction_op = Op::Get("relax." + op_name);
    if (call->op.same_as(reduction_op)) {
      BuildAxisGraphReduce(binding->var, ffi::GetRef<Call>(call), axis_group_graph);
      break;
    }
  }
}

void CollectAxisGraphMatmul(const VarBindingObj* binding, const CallObj* call,
                            AxisGroupGraph* axis_group_graph) {
  static const Op matmul_op = Op::Get("relax.matmul");
  if (call->op.same_as(matmul_op)) {
    BuildAxisGraphMatmul(binding->var, ffi::GetRef<Call>(call), axis_group_graph);
  }
}

void CollectAxisGraphPermuteDims(const VarBindingObj* binding, const CallObj* call,
                                 AxisGroupGraph* axis_group_graph) {
  static const Op permute_dims_op = Op::Get("relax.permute_dims");
  if (call->op.same_as(permute_dims_op)) {
    BuildAxisGraphPermuteDims(binding->var, ffi::GetRef<Call>(call), axis_group_graph);
  }
}

void CollectAxisGraphReshape(const VarBindingObj* binding, const CallObj* call,
                             AxisGroupGraph* axis_group_graph) {
  static const Op reshape_op = Op::Get("relax.reshape");
  if (call->op.same_as(reshape_op)) {
    BuildAxisGraphReshape(binding->var, ffi::GetRef<Call>(call), axis_group_graph);
  }
}

void CollectAxisGraphForDeviceMesh(const VarBindingObj* binding, const CallObj* call,
                                   AxisGroupGraph* axis_group_graph) {
  ffi::Array<Expr> tensor_list;
  static const Op call_tir_op = Op::Get("relax.call_tir");
  ffi::Array<Expr> args;
  if (call->op.same_as(call_tir_op)) {
    args = call->args[1].as_or_throw<Tuple>()->fields;
  } else {
    args = call->args;
  }
  for (const auto& arg : args) {
    if (arg->ty.as<TensorTypeObj>()) {
      tensor_list.push_back(arg);
    }
  }
  for (int i = 0; i < static_cast<int>(tensor_list.size()); i++) {
    axis_group_graph->JoinAxis(Axis(tensor_list[i].get(), -1), {binding->var.get(), -1},
                               distributed::AxisGroupGraph::EdgeType::kDescend);
  }
}

/*!
 * \brief Build an axis group graph for propagation.
 */
class AxisGroupGraphBuilder : public ExprVisitor {
 public:
  static void BuildAxisGroupGraph(AxisGroupGraph* axis_group_graph, const Function& func,
                                  const IRModule& mod) {
    AxisGroupGraphBuilder builder(axis_group_graph, mod);
    builder.VisitExpr(func);
  }

 private:
  explicit AxisGroupGraphBuilder(AxisGroupGraph* axis_group_graph, IRModule mod)
      : axis_group_graph_(axis_group_graph), mod_(mod) {}

  void VisitBinding_(const VarBindingObj* binding, const CallObj* val) {
    CollectAxisGraphBinary(binding, val, axis_group_graph_);
    CollectAxisGraphUnary(binding, val, axis_group_graph_);
    CollectAxisGraphReduce(binding, val, axis_group_graph_);
    CollectAxisGraphMatmul(binding, val, axis_group_graph_);
    CollectAxisGraphPermuteDims(binding, val, axis_group_graph_);
    CollectAxisGraphReshape(binding, val, axis_group_graph_);
    static const Op call_tir_op = Op::Get("relax.call_tir");
    if (val->op.same_as(call_tir_op)) {
      if (ffi::Optional<tirx::Function> func = MatchFunction(mod_, val->args[0])) {
        BuildAxisGraphCallTIR(binding->var, ffi::GetRef<Call>(val), func.value(),
                              axis_group_graph_);
      }
    }
    CollectAxisGraphForDeviceMesh(binding, val, axis_group_graph_);
    ExprVisitor::VisitBinding_(binding, val);
  }

  void VisitBinding_(const VarBindingObj* binding, const TupleGetItemObj* val) {
    axis_group_graph_->JoinAxis(Axis(val->tuple.get(), -1, val->index), {binding->var.get(), -1},
                                distributed::AxisGroupGraph::EdgeType::kDescend);
    const auto* tensor_ty = GetTypeAs<TensorTypeObj>(binding->var);
    if (!tensor_ty) {
      ExprVisitor::VisitBinding_(binding, val);
      return;
    }
    int ndim = tensor_ty->ndim;
    for (int i = 0; i < ndim; i++) {
      axis_group_graph_->JoinAxis(Axis(val->tuple.get(), i, val->index), {binding->var.get(), i},
                                  distributed::AxisGroupGraph::EdgeType::kDescend);
    }
    ExprVisitor::VisitBinding_(binding, val);
  }

  void VisitBinding_(const VarBindingObj* binding, const VarObj* val) {
    ffi::Array<TensorType> tensor_tys;
    if (const auto* tensor_ty = binding->var->ty.as<TensorTypeObj>()) {
      tensor_tys.push_back(ffi::GetRef<TensorType>(tensor_ty));
    } else if (const auto* tuple_ty = binding->var->ty.as<TupleTypeObj>()) {
      TVM_FFI_ICHECK(tuple_ty);
      for (const auto& field_ty : tuple_ty->fields) {
        tensor_tys.push_back(field_ty.as_or_throw<TensorType>());
      }
    } else {
      ExprVisitor::VisitBinding_(binding, val);
      return;
    }
    for (int idx = 0; idx < static_cast<int>(tensor_tys.size()); idx++) {
      int ndim = tensor_tys[idx]->ndim;
      for (int i = -1; i < ndim; i++) {
        axis_group_graph_->JoinAxis({val, i, idx}, {binding->var.get(), i, idx},
                                    distributed::AxisGroupGraph::EdgeType::kDescend);
      }
    }
    ExprVisitor::VisitBinding_(binding, val);
  }

  AxisGroupGraph* axis_group_graph_;
  IRModule mod_;
};

/*!
 * \brief Collect the sharding annotations and add source sharding spec in axis group graph.
 */
class ShardingAnnotationCollector : public ExprVisitor {
 public:
  static void CollectShardingAnnotation(AxisGroupGraph* axis_group_graph, const Function& func) {
    ShardingAnnotationCollector collector(axis_group_graph);
    collector.VisitExpr(func);
  }

 private:
  explicit ShardingAnnotationCollector(AxisGroupGraph* axis_group_graph)
      : axis_group_graph_(axis_group_graph) {}
  void VisitBinding_(const VarBindingObj* binding, const CallObj* val) {
    static const Op annotate_sharding_op = Op::Get("relax.dist.annotate_sharding");
    if (val->op.same_as(annotate_sharding_op)) {
      const auto* attrs = val->attrs.as<DistributionAttrs>();
      TVM_FFI_ICHECK(attrs);

      for (int i = 0; i < static_cast<int>(attrs->placement->dim_specs.size()); i++) {
        const PlacementSpec& placement_spec = attrs->placement->dim_specs[i];
        if (placement_spec->kind == PlacementSpecKind::kSharding) {
          axis_group_graph_->AddSrcShardingPoint({binding->var.get(), placement_spec->axis},
                                                 {attrs->device_mesh, i});
        }
      }
      axis_group_graph_->AddSrcShardingPoint({binding->var.get(), -1}, {attrs->device_mesh, -1});
    }
    ExprVisitor::VisitBinding_(binding, val);
  }

  AxisGroupGraph* axis_group_graph_;
};

/*!
 * \brief Check if the sharding of each tensor is legal.
 */
class ShardingConflictHandler : public ExprVisitor {
 public:
  static void HandleShardingConflict(AxisGroupGraph* axis_group_graph, Function function) {
    axis_group_graph->PropagateShardingSpec();
    ShardingConflictHandler handler(axis_group_graph);
    handler.VisitExpr(function);
    for (const Var& var : function->params) {
      if (GetTypeAs<TensorTypeObj>(var)) {
        handler.CheckTensorShardingCompatible(var);
      }
    }
    axis_group_graph->PropagateShardingSpec();
  }

 private:
  explicit ShardingConflictHandler(AxisGroupGraph* axis_group_graph)
      : axis_group_graph_(axis_group_graph) {}

  void CheckTensorShardingCompatible(Var var) {
    const auto* tensor_ty = GetTypeAs<TensorTypeObj>(var);
    TVM_FFI_ICHECK(tensor_ty);
    const auto* shape = tensor_ty->shape.as<ShapeExprObj>();
    TVM_FFI_ICHECK(shape);
    int ndim = tensor_ty->ndim;
    std::unordered_set<int> sharded_mesh_dim;
    ffi::Optional<DeviceMesh> device_mesh;
    for (int i = -1; i < ndim; i++) {
      std::optional<AxisShardingSpec> sharding_spec;
      int has_sharding_spec;
      std::tie(sharding_spec, has_sharding_spec) =
          axis_group_graph_->GetAxisShardingSpec({var.get(), i});
      if (!has_sharding_spec) {
        continue;
      }

      if (device_mesh.has_value()) {
        TVM_FFI_ICHECK(ffi::StructuralEqual()(device_mesh.value(), sharding_spec->first))
            << "Sharding conflict detected for tensor " << var->name << ": Device Mesh mismatch"
            << ". Conflict Handling logic will be added in the future.";
      } else {
        device_mesh = sharding_spec->first;
      }
      if (i >= 0) {
        int sharding_dim = sharding_spec->second;
        TVM_FFI_ICHECK(sharded_mesh_dim.count(sharding_dim) == 0)
            << "Sharding conflict detected for tensor " << var->name
            << ": Replicate sharding device mesh axis " << sharding_dim
            << ". Conflict Handling logic will be added in the future.";
        sharded_mesh_dim.insert(sharding_dim);
        if (const auto* val = shape->values[i].as<IntImmObj>()) {
          if (val->value < device_mesh.value()->shape[sharding_spec->second]) {
            axis_group_graph_->AddPropagationCutPoint({var.get(), i}, sharding_spec.value());
          }
        }
      }
    }
  }

  void CheckConstantNoSharding(GenericConst constant) {
    const auto* tensor_ty = GetTypeAs<TensorTypeObj>(constant);
    for (int i = 0; i < tensor_ty->ndim; i++) {
      std::optional<AxisShardingSpec> sharding_spec;
      int has_sharding_spec;
      std::tie(sharding_spec, has_sharding_spec) =
          axis_group_graph_->GetAxisShardingSpec({constant.get(), i});
      TVM_FFI_ICHECK(!has_sharding_spec)
          << "GenericConst is not allowed to be sharded. Please convert it into an input param.";
    }
  }

  void VisitExpr_(const CallObj* op) final {
    ffi::Array<Expr> args = GetCallArgs(ffi::GetRef<Call>(op));
    for (const auto& arg : args) {
      if (arg.as<GenericConstObj>() && arg->ty.as<TensorTypeObj>()) {
        CheckConstantNoSharding(arg.as_or_throw<GenericConst>());
      }
    }
    ExprVisitor::VisitExpr_(op);
  }

  void VisitBinding_(const VarBindingObj* binding) final {
    if (GetTypeAs<TensorTypeObj>(binding->var)) {
      CheckTensorShardingCompatible(binding->var);
    }
    ExprVisitor::VisitBinding_(binding);
  }

  AxisGroupGraph* axis_group_graph_;
};

/*!
 * \brief Build distributed IR from given sharding annotation
 */
class DistributedIRBuilder : public ExprMutator {
 public:
  explicit DistributedIRBuilder(const IRModule& module) : ExprMutator(module) {}

  IRModule BuildDistributedIR() {
    auto mod = builder_->GetContextIRModule();
    for (const auto& [gv, base_func] : mod->functions) {
      const auto* func_ = base_func.as<FunctionObj>();
      if (func_ == nullptr || !IsShardingAnnotatedFunc(ffi::GetRef<Function>(func_))) {
        continue;
      }
      Function func = RewriteFunction(ffi::GetRef<Function>(func_), mod);
      builder_->UpdateFunction(gv, func);
    }
    return builder_->GetContextIRModule();
  }

 private:
  using ExprMutator::VisitExpr_;

  DTensorType ConvertToDTensorType(TensorType tensor_ty, Expr expr, int tuple_idx = 0) {
    int ndim = tensor_ty->ndim;
    auto mesh_spec =
        std::get<0>(axis_group_graph_.GetAxisShardingSpec({expr.get(), -1, tuple_idx}));
    TVM_FFI_ICHECK(mesh_spec.has_value())
        << expr << "[" << tuple_idx << "] is not assigned device mesh";
    DeviceMesh device_mesh = mesh_spec->first;
    ffi::Array<PlacementSpec> placement_specs(
        std::vector<PlacementSpec>(device_mesh->shape.size(), PlacementSpec::Replica()));
    for (int i = 0; i < ndim; i++) {
      std::optional<AxisShardingSpec> sharding_spec;
      bool has_sharding_spec;
      std::tie(sharding_spec, has_sharding_spec) =
          axis_group_graph_.GetAxisShardingSpec({expr.get(), i, tuple_idx});
      if (has_sharding_spec) {
        int sharding_dim = sharding_spec->second;
        placement_specs.Set(sharding_dim, PlacementSpec::Sharding(i));
      }
    }
    return DTensorType(tensor_ty, device_mesh, Placement(placement_specs));
  }

  Expr RewriteInputTensorAndConstant(Expr tensor) {
    Type new_ty = Type::Missing();
    if (tensor->ty.as<TensorTypeObj>()) {
      new_ty = ConvertToDTensorType(tensor->ty.as_or_throw<TensorType>(), tensor);
    } else if (const auto* tuple = tensor->ty.as<TupleTypeObj>()) {
      ffi::Array<Type> tuple_ty_fields;
      for (int i = 0; i < static_cast<int>(tuple->fields.size()); i++) {
        if (tuple->fields[i].as<TensorTypeObj>()) {
          tuple_ty_fields.push_back(
              ConvertToDTensorType(tuple->fields[i].as_or_throw<TensorType>(), tensor, i));
        } else {
          tuple_ty_fields.push_back(tuple->fields[i]);
        }
      }
      new_ty = TupleType(tuple_ty_fields);
    }

    if (const auto* var = tensor.as<VarObj>()) {
      Var new_param(var->name, new_ty);
      return new_param;
    } else if (const auto* constant = tensor.as<GenericConstObj>()) {
      GenericConst new_constant(new_ty, constant->value.cast<ffi::Tensor>());
      return new_constant;
    } else {
      TVM_FFI_THROW(InternalError) << "Cannot rewrite tensor which is not a Var or GenericConst";
      throw;
    }
  }

  Function RewriteFunction(Function func, IRModule mod) {
    // Step 1. Construct AxisGroupGraph
    AxisGroupGraphBuilder::BuildAxisGroupGraph(&axis_group_graph_, func, mod);
    // Step 2. Collect Sharding Annotation
    ShardingAnnotationCollector::CollectShardingAnnotation(&axis_group_graph_, func);
    // Step 3. Handle Sharding Conflict
    ShardingConflictHandler::HandleShardingConflict(&axis_group_graph_, func);
    // Step 4. Rewrite Function
    ffi::Array<Var> new_params;
    for (const Var& var : func->params) {
      if (GetTypeAs<TensorTypeObj>(var) || GetTypeAs<TupleTypeObj>(var)) {
        Var new_param = RewriteInputTensorAndConstant(var).as_or_throw<Var>();
        input_tensor_remap_.Set(var, new_param);
        new_params.push_back(new_param);
      } else {
        new_params.push_back(var);
      }
    }
    auto new_body = VisitWithNewScope(func->body, new_params);
    Function new_func(new_params, new_body, std::nullopt, func->is_pure, func->attrs);
    return new_func;
  }

  Expr VisitExpr_(const CallObj* call) final {
    static const Op call_tir_op = Op::Get("relax.call_tir");
    FBuildAxisGraph f = [&](const Var& var, const Call& call, AxisGroupGraph* axis_group_graph) {
      ffi::Optional<tirx::Function> function =
          MatchFunction(this->builder_->GetContextIRModule(), call->args[0]);
      TVM_FFI_ICHECK(function);
      return BuildAxisGraphCallTIR(var, call, function.value(), axis_group_graph);
    };
    Call new_call = ExprMutator::VisitExpr_(call).as_or_throw<Call>();
    ffi::Array<Expr> args = GetCallArgs(new_call);
    for (int i = 0; i < static_cast<int>(args.size()); i++) {
      if (args[i].as<GenericConstObj>() && args[i]->ty.as<TensorTypeObj>()) {
        args.Set(i, RewriteInputTensorAndConstant(args[i]));
      }
    }

    ffi::ObjectPtr<CallObj> n = ffi::make_object<CallObj>(*new_call.get());
    if (new_call->op.same_as(call_tir_op)) {
      // do not infer output type when arg size is 0
      if (!args.empty()) {
        n->args.Set(1, Tuple(args));
        n->ty_args = {InferShardingSpec(Call(n), this->builder_, new_call->ty_args[0], f)};
      }
    } else {
      n->args = args;
    }

    if (const auto* extern_func = new_call->op.as<ExternFuncObj>()) {
      if (extern_func->global_symbol == "vm.builtin.attention_kv_cache_append") {
        n->op = ExternFunc("vm.builtin.distributed.attention_kv_cache_append");
      } else if (extern_func->global_symbol == "vm.builtin.attention_kv_cache_view") {
        n->op = ExternFunc("vm.builtin.distributed.attention_kv_cache_view");
      }
    }
    return Call(n);
  }

  Expr RemoveAnnotateSharding(Call call) {
    static const Op annotate_sharding_op = Op::Get("relax.dist.annotate_sharding");
    if (call->op.same_as(annotate_sharding_op)) {
      return call->args[0];
    } else {
      return call;
    }
  }

  Expr InsertRedistribute(Expr expr, DeviceMesh device_mesh, Placement placement) {
    return redistribute(expr, device_mesh, placement);
  }

  Call RewriteOutType(Call call, DeviceMesh device_mesh, ffi::Array<Placement> placements) {
    // In cases when inference fails (for example, arg size is 0), use the propagated output type.
    Call new_call = call;
    static Op call_tir_op = Op::Get("relax.call_tir");
    if (const auto* extern_func = call->op.as<ExternFuncObj>()) {
      if (extern_func->global_symbol == "vm.builtin.distributed.attention_kv_cache_view") {
        ffi::ObjectPtr<CallObj> new_call_node = ffi::make_object<CallObj>(*call.get());
        Type new_dtensor_ty =
            DTensorType(call->ty_args[0].as_or_throw<TensorType>(), device_mesh, placements[0]);
        new_call_node->ty_args = {new_dtensor_ty};
        new_call = Call(new_call_node);
        new_call->ty = new_dtensor_ty;
      }
    } else if (call->op.same_as(call_tir_op)) {
      TVM_FFI_ICHECK(call->ty_args.size() == 1);
      if (!TypeCompatibleWithDistIR(call->ty_args)) {
        ffi::ObjectPtr<CallObj> new_call_node = ffi::make_object<CallObj>(*call.get());
        if (placements.size() == 1) {
          new_call_node->ty_args = {
              DTensorType(call->ty_args[0].as_or_throw<TensorType>(), device_mesh, placements[0])};
        } else {
          const auto* tuple_ty = call->ty_args[0].as<TupleTypeObj>();
          TVM_FFI_ICHECK(placements.size() == tuple_ty->fields.size());
          ffi::Array<Type> new_tuple_ty_fields;
          for (int i = 0; i < static_cast<int>(placements.size()); i++) {
            new_tuple_ty_fields.push_back(DTensorType(tuple_ty->fields[i].as_or_throw<TensorType>(),
                                                      device_mesh, placements[i]));
          }
          new_call_node->ty_args = {TupleType(new_tuple_ty_fields)};
        }
        new_call = Call(new_call_node);
        new_call->ty = new_call_node->ty_args[0];
      }
    }
    return new_call;
  }

  void VisitBinding_(const VarBindingObj* binding, const CallObj* val) {
    ffi::Array<TensorType> orig_output_tys;
    if (const auto* tensor_ty = GetTypeAs<TensorTypeObj>(binding->var)) {
      orig_output_tys.push_back(ffi::GetRef<TensorType>(tensor_ty));
    } else if (const auto* tuple_ty = GetTypeAs<TupleTypeObj>(binding->var)) {
      for (const auto& field_ty : tuple_ty->fields) {
        orig_output_tys.push_back(field_ty.as_or_throw<TensorType>());
      }
    } else {
      ExprMutator::VisitBinding_(binding, val);
      return;
    }
    // Get the annotated output type from the axis group graph.
    auto mesh_spec = std::get<0>(axis_group_graph_.GetAxisShardingSpec({binding->var.get(), -1}));
    TVM_FFI_ICHECK(mesh_spec.has_value());
    DeviceMesh device_mesh = mesh_spec->first;
    ffi::Array<Placement> placements;  // every tuple element has a placement
    for (int idx = 0; idx < static_cast<int>(orig_output_tys.size()); idx++) {
      ffi::Array<PlacementSpec> placement_specs(
          std::vector<PlacementSpec>(device_mesh->shape.size(), PlacementSpec::Replica()));
      for (int i = 0; i < orig_output_tys[idx]->ndim; i++) {
        std::optional<AxisShardingSpec> sharding_spec;
        bool has_sharding_spec;
        std::tie(sharding_spec, has_sharding_spec) =
            axis_group_graph_.GetAxisShardingSpec({binding->var.get(), i, idx});
        if (has_sharding_spec) {
          placement_specs.Set(sharding_spec->second, PlacementSpec::Sharding(i));
        }
      }
      placements.push_back(Placement(placement_specs));
    }
    // get inferred output type from type deduction
    Call new_call = this->VisitExpr(binding->value).as_or_throw<Call>();
    new_call =
        builder_->Normalize(RewriteOutType(new_call, device_mesh, placements)).as_or_throw<Call>();

    if (const auto* inferred_dtensor_ty = new_call->ty.as<DTensorTypeObj>()) {
      Expr new_value = RemoveAnnotateSharding(new_call);
      if (!ffi::StructuralEqual()(
              DTensorType(inferred_dtensor_ty->tensor_ty, device_mesh, placements[0]),
              new_call->ty)) {
        new_value = InsertRedistribute(new_value, device_mesh, placements[0]);
      }
      if (const auto* var = new_value.as<VarObj>()) {
        var_remap_.insert_or_assign(binding->var, ffi::GetRef<Var>(var));
      } else {
        ReEmitBinding(binding, builder_->Normalize(new_value));
      }
    } else {
      const auto* inferred_tuple_ty = new_call->ty.as<TupleTypeObj>();
      TVM_FFI_ICHECK(inferred_tuple_ty) << new_call;
      Var new_var = builder_->Emit(new_call);
      var_remap_.insert_or_assign(binding->var, new_var);
      for (int i = 0; i < static_cast<int>(inferred_tuple_ty->fields.size()); i++) {
        if (!ffi::StructuralEqual()(
                DTensorType(inferred_tuple_ty->fields[i].as_or_throw<DTensorType>()->tensor_ty,
                            device_mesh, placements[i]),
                inferred_tuple_ty->fields[i])) {
          Var redistribute_var = builder_->Emit(
              InsertRedistribute(TupleGetItem(new_var, i), device_mesh, placements[i]));
          tuple_getitem_remap_.insert_or_assign(TupleGetItem(binding->var, i), redistribute_var);
        }
      }
    }
  }

  void VisitBinding_(const VarBindingObj* binding, const TupleGetItemObj* val) {
    if (tuple_getitem_remap_.count(ffi::GetRef<TupleGetItem>(val))) {
      var_remap_.insert_or_assign(binding->var,
                                  tuple_getitem_remap_.at(ffi::GetRef<TupleGetItem>(val)));
    } else {
      ExprMutator::VisitBinding_(binding, val);
    }
  }

  Expr VisitExpr_(const VarObj* var) final {
    Var var_ref = ffi::GetRef<Var>(var);
    auto it = input_tensor_remap_.find(var_ref);
    if (it != input_tensor_remap_.end()) {
      var_remap_.insert_or_assign(var_ref, (*it).second);
    }
    return ExprMutator::VisitExpr_(var);
  }

  ffi::Map<Var, Var> input_tensor_remap_;
  std::unordered_map<TupleGetItem, Var, ffi::StructuralHash, ffi::StructuralEqual>
      tuple_getitem_remap_;
  AxisGroupGraph axis_group_graph_;
};
namespace transform {

Pass PropagateSharding() {
  auto pass_func = [=](IRModule m, PassContext pc) {
    return DistributedIRBuilder(m).BuildDistributedIR();
  };
  return CreateModulePass(pass_func, 1, "PropagateSharding");
}
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.distributed.transform.PropagateSharding", PropagateSharding);
}
}  // namespace transform

}  // namespace distributed
}  // namespace relax
}  // namespace tvm
