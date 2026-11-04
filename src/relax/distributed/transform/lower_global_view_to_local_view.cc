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
 * \file tvm/relax/distributed/transform/lower_global_view_to_local_view.cc
 * \brief Pass for lowering global view TensorIR into local view
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/distributed/axis_group_graph.h>
#include <tvm/relax/distributed/transform.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op/ccl.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/tirx/function.h>

#include "../../../s_tir/schedule/transform.h"
#include "utils.h"
namespace tvm {
namespace tirx {
using namespace tvm::prim;

using namespace tvm::relax::distributed;

class DistTensorReplacer : public s_tir::StmtExprMutator {
 public:
  static Stmt TensorReplace(Stmt stmt, ffi::Map<TensorVar, TensorVar> tensor_map) {
    auto replacer = ffi::make_object<DistTensorReplacer>(tensor_map);
    return replacer->Mutate(stmt, InplaceMode::kDisallow).ValueOrUnchanged(stmt);
  }

  explicit DistTensorReplacer(const ffi::Map<TensorVar, TensorVar>& tensor_map) {
    for (const auto& [source, target] : tensor_map) {
      VarRemapSet(source, target);
    }
  }
};

class DistSBlockInfoCollector : public s_tir::StmtExprVisitor {
 private:
  ffi::Optional<VisitInterrupt> Visit_(const TensorStoreObj* op) final {
    tensor_access_indices[op->dest.as_or_throw<tvm::tirx::TensorVar>()].push_back(op->indices);
    return s_tir::StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadObj* op) final {
    tensor_access_indices[op->source.as_or_throw<tvm::tirx::TensorVar>()].push_back(op->indices);
    return s_tir::StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const s_tir::SBlockObj* op) final {
    for (const auto& iter_var : op->iter_vars) {
      if (iter_var->iter_type == s_tir::kCommReduce) {
        TVM_FFI_ICHECK(op->writes.size() == 1);
        reduce_tensor_ = op->writes[0]->source.as_or_throw<tvm::tirx::TensorVar>();
      }
    }
    return s_tir::StmtExprVisitor::Visit_(op);
  }

  bool IsReduceTensorAccess(const PrimExpr& expr) {
    if (const auto* tensor_load = expr.as<TensorLoadObj>()) {
      return reduce_tensor_.has_value() &&
             tensor_load->source.as_or_throw<tvm::tirx::TensorVar>().same_as(
                 reduce_tensor_.value());
    }
    return false;
  }

  ffi::Optional<VisitInterrupt> Visit_(const prim::AddObj* op) final {
    if (IsReduceTensorAccess(op->a) || IsReduceTensorAccess(op->b)) {
      reduce_kind = "sum";
    }
    return s_tir::StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const prim::MulObj* op) final {
    if (IsReduceTensorAccess(op->a) || IsReduceTensorAccess(op->b)) {
      reduce_kind = "prod";
    }
    return s_tir::StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const prim::MinObj* op) final {
    if (IsReduceTensorAccess(op->a) || IsReduceTensorAccess(op->b)) {
      reduce_kind = "min";
    }
    return s_tir::StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const prim::MaxObj* op) final {
    if (IsReduceTensorAccess(op->a) || IsReduceTensorAccess(op->b)) {
      reduce_kind = "max";
    }
    return s_tir::StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<TensorVar> reduce_tensor_;

 public:
  std::unordered_map<TensorVar, ffi::Array<ffi::Array<PrimExpr>>, ffi::ObjectPtrHash,
                     ffi::ObjectPtrEqual>
      tensor_access_indices;
  std::string reduce_kind;
};

class DistributedTensorCompactor : public s_tir::StmtExprMutator {
  // FIXME: change to use unordered_map<int, AxisShardingSpec> (represent dim and sharding spec)
  // Currently we assume device mesh is only 1d, but when we support 2d, we need to change this
  using DimShard = std::unordered_map<int, int>;

 public:
  static std::tuple<tirx::Function, std::string> DistTensorCompact(
      const std::vector<ShardingSpec>& sharding_specs, tirx::Function function) {
    function = tirx::RenewDef(function);
    auto compactor = ffi::make_object<DistributedTensorCompactor>(sharding_specs, function);
    ffi::Array<Var> new_params;
    ffi::Map<TensorVar, TensorVar> replace_tensor_map;
    for (const Var& param : function->params) {
      if (!param->ty.as<TensorTypeObj>()) {
        new_params.push_back(param);
        continue;
      }
      TensorVar tensor = param.as_or_throw<TensorVar>();
      TensorVar shard_tensor = compactor->ShardTensor(tensor);
      new_params.push_back(shard_tensor);
      if (!shard_tensor.same_as(tensor)) {
        replace_tensor_map.Set(tensor, shard_tensor);
      }
    }
    auto new_body =
        compactor->Mutate(function->body, InplaceMode::kDisallow).ValueOrUnchanged(function->body);
    if (new_body.has_value()) {
      new_body = DistTensorReplacer::TensorReplace(new_body.value(), replace_tensor_map);
    }
    tirx::Function new_func(new_params, new_body, function->ret_type, function->attrs,
                            function->loc);
    return std::make_tuple(new_func, compactor->add_allreduce_kind_);
  }

  DistributedTensorCompactor(const std::vector<ShardingSpec>& sharding_specs,
                             tirx::Function function)
      : sharding_specs_(sharding_specs) {
    PropagateShardingSpecOnBlock(function);
  }

 private:
  // todo: if cannot propagate, insert allgather
  // todo: if reduce, insert allreduce
  void PropagateShardingSpecOnBlock(tirx::Function function) {
    extractor_->Visit(function->body);
    std::unordered_set<TensorAxis, TensorAxisHash> visited;
    for (int i = 0, j = 0; i < static_cast<int>(function->params.size()); i++) {
      Var param_var = function->params[i];
      if (!param_var->ty.as<TensorTypeObj>()) {
        continue;
      }
      TensorVar param_tensor = param_var.as_or_throw<TensorVar>();
      ShardingSpec spec = sharding_specs_[j++];

      for (int mesh_dim = 0; mesh_dim < static_cast<int>(spec.first->shape.size()); mesh_dim++) {
        PlacementSpec dim_placement = spec.second->dim_specs[mesh_dim];
        if (dim_placement->kind == PlacementSpecKind::kReplica) {
          continue;
        }
        std::vector<TensorAxis> tensor_axis_group;
        extractor_->DFSGraph({param_tensor, dim_placement->axis}, &visited, &tensor_axis_group);
        for (const auto& tensor_axis : tensor_axis_group) {
          tensor_shards_[tensor_axis.first][tensor_axis.second] = spec.first->shape[mesh_dim];
        }
      }
    }
  }

  ffi::Array<s_tir::IterVar> ShardIterVar(
      s_tir::SBlock block,
      const std::unordered_map<TensorVar, ffi::Array<ffi::Array<PrimExpr>>, ffi::ObjectPtrHash,
                               ffi::ObjectPtrEqual>& tensor_access_indices) {
    std::vector<TensorVar> tensors;
    for (const auto& read : block->reads) {
      tensors.push_back(read->source.as_or_throw<tvm::tirx::TensorVar>());
    }
    for (const auto& write : block->writes) {
      tensors.push_back(write->source.as_or_throw<tvm::tirx::TensorVar>());
    }
    ffi::Map<Var, Range> iter_var_range;
    for (const auto& iter_var : block->iter_vars) {
      iter_var_range.Set(iter_var->var, iter_var->dom.value());
    }
    sym::Analyzer analyzer;
    for (const auto& tensor : tensors) {
      if (tensor_access_indices.count(tensor) == 0 || tensor_shards_.count(tensor) == 0) {
        continue;
      }
      ffi::Array<ffi::Array<PrimExpr>> access_indices = tensor_access_indices.at(tensor);
      DimShard dim_shards = tensor_shards_[tensor];
      for (const auto& access_index : access_indices) {
        for (const auto& pr : dim_shards) {
          int dim = pr.first;
          int shard = pr.second;
          auto sharding_var = GetShardingVarFromIndex(access_index[dim], iter_var_range, analyzer);
          if (!sharding_var.has_value()) {
            continue;
          }
          Var var = sharding_var.value();
          TVM_FFI_ICHECK(!iter_var_shards_.count(var) || iter_var_shards_[var] == shard)
              << "A loop cannot have different sharding";
          iter_var_shards_[var] = shard;
        }
      }
    }

    ffi::Array<s_tir::IterVar> new_iter_vars;
    for (const auto& iter_var : block->iter_vars) {
      if (iter_var_shards_.count(iter_var->var)) {
        int shard = iter_var_shards_[iter_var->var];
        if (shard > 1) {
          Range dom = iter_var->dom.value();
          TVM_FFI_ICHECK(IsZero(dom->begin));
          sym::Analyzer analyzer;
          TVM_FFI_ICHECK(analyzer->CanProve(floormod(dom->extent, shard) == 0));
          new_iter_vars.push_back(
              s_tir::IterVar(Range::FromMinExtent(dom->begin, floordiv(dom->extent, shard)),
                             iter_var->var, iter_var->iter_type, iter_var->thread_tag));
          continue;
        }
      }
      new_iter_vars.push_back(iter_var);
    }
    return new_iter_vars;
  }

  TensorVar ShardTensor(TensorVar tensor) {
    if (tensor_shards_.count(tensor) == 0) {
      return tensor;
    }
    DimShard dim_shards = tensor_shards_[tensor];
    ffi::Array<PrimExpr> shape;
    for (int i = 0; i < static_cast<int>(tensor.shape().size()); i++) {
      if (dim_shards.count(i)) {
        shape.push_back(floordiv(tensor.shape()[i], dim_shards[i]));
      } else {
        shape.push_back(tensor.shape()[i]);
      }
    }
    auto new_type_node = ffi::make_object<TensorTypeObj>(*tensor.ty().get());
    new_type_node->shape = std::move(shape);
    TensorType new_type(std::move(new_type_node));
    return TensorVar(tensor->name, std::move(new_type), tensor->loc);
  }

  UnchangedOr<Stmt> Mutate_(const s_tir::SBlockObj* op, InplaceMode inplace_mode) final {
    s_tir::SBlock block = s_tir::StmtExprMutator::Mutate_(op, inplace_mode)
                              .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                              .as_or_throw<s_tir::SBlock>();
    auto collector = ffi::make_object<DistSBlockInfoCollector>();
    collector->Visit(block);
    ffi::Array<s_tir::IterVar> new_iter_vars =
        ShardIterVar(block, collector->tensor_access_indices);
    ffi::Array<TensorVar> new_alloc_tensors;
    ffi::Map<TensorVar, TensorVar> tensor_map;
    for (const TensorVar& tensor : block->alloc_tensors) {
      TensorVar sharded_tensor = ShardTensor(tensor);
      if (!sharded_tensor.same_as(tensor)) {
        tensor_map.Set(tensor, sharded_tensor);
      }
      new_alloc_tensors.push_back(sharded_tensor);
    }
    // condition for adding allreduce:
    // sharding on reduction axis
    for (const s_tir::IterVar& iter_var : new_iter_vars) {
      if (iter_var->iter_type == s_tir::kCommReduce && iter_var_shards_.count(iter_var->var)) {
        TVM_FFI_ICHECK(add_allreduce_kind_ == "");
        AddAllReduceBlock(collector->reduce_kind);
        break;
      }
    }
    ffi::ObjectPtr<s_tir::SBlockObj> new_block =
        ffi::make_object<s_tir::SBlockObj>(*block.operator->());
    new_block->iter_vars = new_iter_vars;
    new_block->alloc_tensors = new_alloc_tensors;
    if (new_block->name_hint == "root") {
      new_block->alloc_tensors.insert(new_block->alloc_tensors.end(),
                                      allocated_tensor_under_root.begin(),
                                      allocated_tensor_under_root.end());
    }
    new_block->body = DistTensorReplacer::TensorReplace(block->body, tensor_map);
    return s_tir::SBlock(new_block);
  }

  void AddAllReduceBlock(std::string reduce_kind) { add_allreduce_kind_ = reduce_kind; }

  UnchangedOr<Stmt> Mutate_(const s_tir::SBlockRealizeObj* op, InplaceMode inplace_mode) final {
    s_tir::SBlockRealize realize = s_tir::StmtExprMutator::Mutate_(op, inplace_mode)
                                       .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                                       .as_or_throw<s_tir::SBlockRealize>();

    for (int i = 0; i < static_cast<int>(realize->iter_values.size()); i++) {
      PrimExpr iter_value = realize->iter_values[i];
      s_tir::IterVar iter_var = realize->block->iter_vars[i];
      if (!iter_var_shards_.count(iter_var->var)) {
        continue;
      }
      auto loop_var = iter_value.as<PrimVar>();
      TVM_FFI_ICHECK(loop_var);
      loop_var_shards_[loop_var.value()] = iter_var_shards_[iter_var->var];
    }
    return realize;
  }

  UnchangedOr<Stmt> Mutate_(const ForObj* op, InplaceMode inplace_mode) final {
    For new_loop = s_tir::StmtExprMutator::Mutate_(op, inplace_mode)
                       .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                       .as_or_throw<For>();
    if (loop_var_shards_.count(op->var)) {
      int shard = loop_var_shards_[op->var];
      if (shard > 1) {
        sym::Analyzer analyzer;
        TVM_FFI_ICHECK(analyzer->CanProve(
            floormod(Range::GetExtentFromBeginEnd(new_loop->begin, new_loop->end), shard) == 0));
        new_loop.CopyOnWrite()->end =
            new_loop->begin +
            floordiv(Range::GetExtentFromBeginEnd(new_loop->begin, new_loop->end), shard);
        return new_loop;
      }
    }
    return new_loop;
  }

  std::unordered_map<Var, int> iter_var_shards_;
  std::unordered_map<Var, int> loop_var_shards_;
  ffi::Array<TensorVar> allocated_tensor_under_root;
  ffi::ObjectPtr<TensorAxisGraphExtractor> extractor_ =
      ffi::make_object<TensorAxisGraphExtractor>();
  std::vector<ShardingSpec> sharding_specs_;
  std::unordered_map<TensorVar, DimShard, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> tensor_shards_;
  std::string add_allreduce_kind_;
};

}  // namespace tirx
}  // namespace tvm

namespace tvm {
namespace relax {
namespace distributed {

class LowerTIRToLocalView : public ExprMutator {
 public:
  explicit LowerTIRToLocalView(IRModule mod) : ExprMutator(mod) {}

  IRModule Lower() {
    auto mod = builder_->GetContextIRModule();
    for (const auto& [gv, base_func] : mod->functions) {
      const auto* func_ = base_func.as<FunctionObj>();
      if (func_ == nullptr || !IsDistIRFunc(ffi::GetRef<Function>(func_))) {
        continue;
      }
      Expr new_func_body = this->VisitExpr(func_->body);
      ffi::ObjectPtr<FunctionObj> new_func = ffi::make_object<FunctionObj>(*func_);
      new_func->body = new_func_body;
      builder_->UpdateFunction(gv, Function(new_func));
    }
    return builder_->GetContextIRModule();
  }

 private:
  inline ffi::Array<DTensorType> ExtractDTensorType(Var var) {
    if (const auto* dtensor_ty = GetTypeAs<DTensorTypeObj>(var)) {
      return {ffi::GetRef<DTensorType>(dtensor_ty)};
    } else if (const auto* tuple_ty = GetTypeAs<TupleTypeObj>(var)) {
      ffi::Array<DTensorType> ret;
      for (const auto& field : tuple_ty->fields) {
        ret.push_back(field.as_or_throw<DTensorType>());
      }
      return ret;
    } else {
      TVM_FFI_THROW(InternalError)
          << "The output of a call_tir should be a DTensorType or TupleType";
    }
  }

  void VisitBinding_(const VarBindingObj* binding, const CallObj* val) final {
    static const Op call_tir_op = Op::Get("relax.call_tir");
    if (!val->op.same_as(call_tir_op)) {
      ExprMutator::VisitBinding_(binding, val);
      return;
    }
    std::vector<ShardingSpec> sharding_specs;
    ffi::Array<Expr> args = val->args[1].as_or_throw<Tuple>()->fields;
    GlobalVar gvar = val->args[0].as_or_throw<GlobalVar>();
    tirx::Function function = MatchFunction(builder_->GetContextIRModule(), gvar).value();
    TVM_FFI_ICHECK_LE(args.size(), function->params.size());
    for (size_t i = 0; i < args.size(); ++i) {
      const Expr& arg = args[i];
      const tvm::Var& param = function->params[i];
      if (param->ty.as<tirx::TensorTypeObj>()) {
        const auto* ty = GetTypeAs<DTensorTypeObj>(arg);
        TVM_FFI_CHECK(ty, TypeError)
            << "Expected tensor parameter " << param << " to receive a distributed tensor, but "
            << arg << " has type " << GetType(arg);
        sharding_specs.push_back(ShardingSpec(ty->device_mesh, ty->placement));
      } else {
        TVM_FFI_CHECK(arg.as<PrimExpr>(), TypeError)
            << "Expected scalar parameter " << param
            << " to receive an individual primitive expression, but " << arg << " has type "
            << GetType(arg);
      }
    }
    Var output_var = binding->var;
    ffi::Array<DTensorType> output_tys = ExtractDTensorType(output_var);
    for (const auto& ty : output_tys) {
      sharding_specs.push_back(ShardingSpec(ty->device_mesh, ty->placement));
    }
    auto [new_function, allreduce_kind] =
        tirx::DistributedTensorCompactor::DistTensorCompact(sharding_specs, function);
    auto new_gvar = builder_->AddFunction(new_function, gvar->name_hint);
    Call call = this->VisitExpr(binding->value).as_or_throw<Call>();
    ffi::ObjectPtr<CallObj> new_call_node = ffi::make_object<CallObj>(*call.get());
    new_call_node->op = Op::Get("relax.dist.call_tir_local_view");
    new_call_node->args.Set(0, new_gvar);
    Call new_call(new_call_node);
    if (allreduce_kind != "") {
      ffi::ObjectPtr<AllReduceAttrs> attrs = ffi::make_object<AllReduceAttrs>();
      attrs->op_type = allreduce_kind;
      new_call =
          Call(Type::Missing(), Op::Get("relax.ccl.allreduce"), {new_call}, Attrs(attrs), {});
    }
    ReEmitBinding(binding, this->builder_->Normalize(new_call));
  }
};

namespace transform {

Pass LowerGlobalViewToLocalView() {
  auto pass_func = [=](IRModule m, PassContext pc) { return LowerTIRToLocalView(m).Lower(); };
  return CreateModulePass(pass_func, 1, "LowerGlobalViewToLocalView");
}
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.distributed.transform.LowerGlobalViewToLocalView",
                        LowerGlobalViewToLocalView);
}
}  // namespace transform

}  // namespace distributed
}  // namespace relax
}  // namespace tvm
