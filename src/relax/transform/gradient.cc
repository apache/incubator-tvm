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
 * \file src/relax/transform/gradient.cc
 * \brief Reverse-mode automatic differentiation.
 *
 * Now only supports differentiating one function in the IRModule with one dataflow block
 * with respect to the only return value of the function, which needs to be scalar.
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/ir/op.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/nested_msg.h>
#include <tvm/relax/op/op.h>
#include <tvm/relax/op_attr.h>
#include <tvm/relax/transform.h>

#include <unordered_set>

#include "../op/tensor/binary.h"
#include "../op/tensor/create.h"
#include "gradient_simplifier.h"
#include "utils.h"

namespace tvm {
namespace relax {

// We will use NestedMsg<Expr> to handle adjoint updates involving tuple handling
using AdjointMsg = NestedMsg<Expr>;
using VarSet = std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

// Used in CallTIRWithGradCollector. call_tir -> call_tir_with_grad
using CallTIRWithGradInfo = std::unordered_map<Call, Call, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

/*!
 * \brief Collect all call_tir_with_grad nodes, transform them into call_tir nodes, and collect the
 * te_grad_name and te_grad_kwargs information.
 */
class CallTIRWithGradEliminator : private ExprMutator {
 public:
  /*!
   * \brief Collect all variables that needs to be checkpointed, and remove the start_checkpoint
   * and the end_checkpoint bindings.
   *
   * \param func The original function
   * \return The function with all start_checkpoint and end_checkpoint bindings removed, and a
   * VarSet containing all checkpointed vars.
   */
  static Function Transform(const Function& func) {
    return CallTIRWithGradEliminator().VisitExpr(func).as_or_throw<Function>();
  }

 private:
  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const CallObj* call_node) final {
    if (!call_node->op.same_as(Op::Get("relax.call_tir_with_grad"))) {
      return ExprMutator::VisitExpr_(call_node);
    }
    return Call(Type::Missing(), Op::Get("relax.call_tir"), call_node->args, {}, call_node->ty_args,
                call_node->loc);
  }
};

/*!
 * \brief Collect all variables that needs to be checkpointed, and remove the start_checkpoint
 * and the end_checkpoint bindings.
 *
 * Here we have some principles to determine which var should be checkpointed:
 * 1. Input of the function is checkpointed
 * 2. For var x marked with start_checkpoint() (wrapped by start_checkpoint), it means x is an input
 *    to some checkpoint function. So var x is checkpointed
 * 3. For other var x , find its predecessor path.
 *   a. If every predecessor path is marked with end_checkpoint(), x is checkpointed
 *   b. Else, there must exists a predecessor path marked with start_checkpoint(). So x is not
 *      checkpointed
 */
class CheckpointCollector : private ExprMutator {
 public:
  /*!
   * \brief Collect all variables that needs to be checkpointed, and remove the start_checkpoint
   * and the end_checkpoint bindings.
   *
   * \param func The original function
   * \return The function with all start_checkpoint and end_checkpoint bindings removed.
   */
  Function Transform(const Function& func) {
    auto collector = CheckpointCollector();
    return this->VisitExpr(func).as_or_throw<Function>();
  }

  // checkpointed vars
  VarSet checkpoints;
  // mapping from vars that are wrapped in start_checkpoint or end_checkpoint to the original vars
  std::unordered_map<Var, Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> var_mapping;

 private:
  Expr VisitExpr_(const FunctionObj* func) final {
    for (auto var : func->params) {
      checkpoints.insert(var);
    }

    return ExprMutator::VisitExpr_(func);
  }

  void VisitBinding(const Binding& binding) {
    static const auto s_cp = Op::Get("relax.grad.start_checkpoint");
    static const auto e_cp = Op::Get("relax.grad.end_checkpoint");

    // If every variable that the variable of binding relies on is either
    // 1) the output of end_checkpoint; 2) checkpointed
    // then the variable of binding will be checkpointed
    auto var_binding = binding.as<VarBindingObj>();
    TVM_FFI_ICHECK(var_binding);

    auto value_call = var_binding->value.as<CallObj>();
    if (!value_call || (!value_call->op.same_as(s_cp) && !value_call->op.same_as(e_cp))) {
      bool all_inner_var_checkpointed = true;
      PostOrderVisit(var_binding->value, [this, &all_inner_var_checkpointed](const Expr& expr) {
        if (auto var = expr.as<VarObj>()) {
          Var var_ref = ffi::GetRef<Var>(var);
          all_inner_var_checkpointed &=
              (checkpoints.count(var_ref) != 0 || e_vars_.count(var_ref) != 0);
        }
      });

      if (all_inner_var_checkpointed) {
        checkpoints.insert(var_binding->var);
      }
    }

    ExprMutator::VisitBinding(binding);
  }

  // mark vars to be checkpointed, and eliminate bindings with checkpoint calls
  void VisitBinding_(const VarBindingObj* binding, const CallObj* value) final {
    static const auto s_cp = Op::Get("relax.grad.start_checkpoint");
    static const auto e_cp = Op::Get("relax.grad.end_checkpoint");

    if (value->op.same_as(s_cp) || value->op.same_as(e_cp)) {
      // Eliminate the binding
      auto var = value->args[0].as<VarObj>();
      TVM_FFI_ICHECK(var) << "The first argument of relax.grad.start_checkpoint and "
                             "relax.grad.end_checkpoint should be a Var";
      // var might already be remapped. Find the original var
      auto orig_var = ExprMutator::VisitExpr(ffi::GetRef<Var>(var)).as_or_throw<Var>();
      // Add remapping from binding->var to new_var
      if (!binding->var.as<DataflowVarObj>() && var->IsInstance<DataflowVarObj>()) {
        // For output binding, emit a dummy binding
        this->var_remap_.insert_or_assign(binding->var,
                                          builder_->EmitOutput(orig_var, orig_var->name));
      } else {
        this->var_remap_.insert_or_assign(binding->var, orig_var);
      }
      var_mapping.insert_or_assign(binding->var, orig_var);

      if (value->op.same_as(s_cp)) {
        // mark the original var to be checkpointed
        checkpoints.insert(orig_var);
      } else if (value->op.same_as(e_cp)) {
        e_vars_.insert(binding->var);
      }
    } else {
      ExprMutator::VisitBinding_(binding, value);
    }
  }

  // vars that are the output of end_checkpoint
  VarSet e_vars_;
};

/*!
 * \brief A tool class for BackwardBindingGenerator
 * Generate the checkpoint bindings. To be specific, in the backward process, we need to use vars
 * computed in the forward process. Those vars contained in the given checkpoints array, and the
 * inputs of the function, will be used as is; other vars will be computed again (this will
 * generate bindings) using the checkpoint vars.
 */
class CheckpointGenerator : private ExprMutator {
 public:
  /*!
   * \brief Generate the checkpoint bindings for BackwardBindingGenerator
   *
   * \param builder The BlockBuilder of BackwardBindingGenerator, used to generate bindings
   * \param orig_params The parameters of the forward function
   * \param forward_block The forward DataflowBlock
   * \param checkpoints The checkpointed vars. checkpoints being empty means all Vars are
   * checkpointed
   */
  CheckpointGenerator(const BlockBuilder& builder, const ffi::Array<Var>& orig_params,
                      const DataflowBlock& forward_block, const VarSet& checkpoints)
      : builder_(builder) {
    // func params will always be checkpointed
    for (auto var : orig_params) {
      checkpoint_map_.Set(var, var);
    }

    for (auto binding : forward_block->bindings) {
      auto* var_binding = binding.as<VarBindingObj>();
      TVM_FFI_ICHECK(var_binding) << "Now only support VarBindingObj";
      auto var = var_binding->var;
      binding_map_.Set(var, var_binding->value);
      if (checkpoints.count(var)) {
        checkpoint_map_.Set(var, var);
      }
    }
  }

  // Receives the forward binding var and value, returns the checkpointed binding var and value.
  std::pair<Var, Expr> UpdateBinding(const Var& var, const Expr& value) {
    Expr new_value = VisitExpr(value);
    auto it = checkpoint_map_.find(var);
    if (it != checkpoint_map_.end()) {
      return std::make_pair((*it).second, new_value);
    }
    auto new_var = builder_->Emit(new_value, var->name + "_cp");
    checkpoint_map_.Set(var, new_var);
    return std::make_pair(new_var, new_value);
  }

 private:
  using ExprMutator::VisitExpr_;

  // Visit the use-site of a defined Var
  Expr VisitExpr_(const VarObj* op) final { return VisitVar(ffi::GetRef<Var>(op)); }

  // Visit the use-site of a defined DataflowVar
  Expr VisitExpr_(const DataflowVarObj* op) final { return VisitVar(ffi::GetRef<Var>(op)); }

  Expr VisitVar(const Var& var) {
    auto it = checkpoint_map_.find(var);
    if (it != checkpoint_map_.end()) {
      return (*it).second;
    }
    Var new_var = builder_->Emit(VisitExpr(binding_map_[var]), var->name + "_cp");
    checkpoint_map_.Set(var, new_var);
    return new_var;
  }

  // The only purpose of this function is create a new expr for Call node
  // to pass the structual equal check
  Expr VisitExpr_(const CallObj* call_node) final {
    Expr new_op = this->VisitExpr(call_node->op);

    tvm::ffi::Array<Expr> call_args;
    for (Expr arg : call_node->args) {
      Expr new_arg = this->VisitExpr(arg);
      call_args.push_back(new_arg);
    }
    Type ret_ty = Type::Missing();
    if (call_node->ty.as<PrimTypeObj>()) {
      if (auto op = call_node->op.as<Op>()) {
        static auto infer_type_map = Op::GetAttrMap(tvm::op_attr::kInferType);
        static auto infer_type_with_builder_map =
            Op::GetAttrMap(tvm::relax::op_attr::kInferTypeWithBuilder);
        if (!infer_type_map.count(op.value()) && !infer_type_with_builder_map.count(op.value())) {
          ret_ty = call_node->ty.as_or_throw<Type>();
        }
      }
    }
    return Call(ret_ty, new_op, call_args, call_node->attrs, call_node->ty_args);
  }

  BlockBuilder builder_;
  // The mapping from the forward vars to the checkpoint vars.
  ffi::Map<Var, Var> checkpoint_map_;
  // The mapping from the forward vars to their bindings, used to generate checkpoint bindings
  ffi::Map<Var, Expr> binding_map_;
};

/*!
 * \brief A tool class for GradientMutator
 * Visit the forward bindings and generate the backward bindings
 */
class BackwardBindingGenerator : private ExprVisitor {
 public:
  /*!
   * \brief Generate the backward bindings for the corresponding GradientMutator
   *
   * \param builder The BlockBuilder of GradientMutator, used to generate bindings
   * \param forward_block The forward DataflowBlock
   * \param require_grads The Var list to differentiate w.r.t.
   * \param orig_params The params of the forward function. Used for checkpointing
   * \param target_var The target Var to differentiate
   * \param orig_return_value The original return value of the function. The new return value is a
   * 2-tuple, containing the original return value, and a tuple of the adjoints of parameters
   * \param checkpoints The checkpointed vars. checkpoints being empty means all Vars are
   * checkpointed
   * \return The return expr of new adjoint function.
   */
  static Expr Generate(const BlockBuilder& builder, const DataflowBlock& forward_block,
                       const ffi::Array<Var>& require_grads, const Var& target_var,
                       const ffi::Array<Var>& orig_params, const Expr& orig_return_value,
                       const CheckpointCollector& cp_collector) {
    CheckpointGenerator checkpoint_generator(builder, orig_params, forward_block,
                                             cp_collector.checkpoints);
    BackwardBindingGenerator generator(builder, cp_collector, checkpoint_generator);

    // Initialize the adjoint of target_var as ones op. We have already checked the target.
    auto* target_ty = GetTypeAs<TensorTypeObj>(target_var);
    generator.UpdateAdjoint(target_var,
                            ones(target_ty->shape.value(), target_ty->dtype.value()->dtype));

    // Do reverse-mode ad, so visit bindings backwards
    for (auto it = forward_block->bindings.rbegin(); it != forward_block->bindings.rend(); ++it) {
      generator.VisitBinding(*it);
    }

    return generator.Epilogue(require_grads, orig_return_value);
  }

 private:
  explicit BackwardBindingGenerator(const BlockBuilder& builder,
                                    const CheckpointCollector& cp_collector,
                                    const CheckpointGenerator& checkpoint_generator)
      : builder_(builder),
        cp_collector_(cp_collector),
        checkpoint_generator_(checkpoint_generator) {}

  void VisitBinding(const Binding& binding) final {
    // TODO(chaofan, yixin): support other types of bindings
    TVM_FFI_ICHECK(binding->IsInstance<VarBindingObj>()) << "Now only support VarBindingObj";
    auto* var_binding = binding.as<VarBindingObj>();

    if (adjoint_var_map_.count(var_binding->var) == 0) {
      // Optimization: this var is not used in the following bindings
      return;
    }

    Expr value = var_binding->value;
    // TODO(chaofan, yixin): support other types of binding values
    TVM_FFI_ICHECK(value->IsInstance<CallObj>() || value->IsInstance<TupleObj>() ||
                   value->IsInstance<TupleGetItemObj>() || value->IsInstance<VarObj>() ||
                   value->IsInstance<GenericConstObj>())
        << "Now does not support the type of binding value: " << value;

    ExprVisitor::VisitBinding_(var_binding);
  }

  // The following functions will handle the adjoint expr of the inputs of binding
  // For call node, we would call the registered gradient functions
  void VisitBinding_(const VarBindingObj* binding, const CallObj* call) final {
    // Skip if it is not an Op
    if (!call->op->IsInstance<OpObj>()) {
      return;
    }

    static const auto& gradient_op_map = Op::GetAttrMap(tvm::relax::op_attr::kPrimalGradient);
    static const constexpr char* te_grad_func_prefix = "tvm.relax.te_grad._register.";

    Var adjoint_var = adjoint_var_map_[binding->var];
    const Op& call_op = call->op.as_or_throw<Op>();

    // Support for checkpointing
    auto [checkpoint_var, checkpoint_call] =
        checkpoint_generator_.UpdateBinding(binding->var, ffi::GetRef<Call>(call));

    if (call_op.same_as(Op::Get("relax.call_tir"))) {
      TVM_FFI_THROW(InternalError)
          << "Differentiation of call_tir op without registering corresponding gradient "
             "function is not supported yet.";
    } else if (call_op.same_as(Op::Get("relax.call_tir_with_grad"))) {
      // tirx gradient registering
      auto te_grad_name = call->attrs.as<CallTIRWithGradAttrs>()->te_grad_name;
      const auto grad_func =
          tvm::ffi::Function::GetGlobalRequired(te_grad_func_prefix + te_grad_name);
      Var partials =
          grad_func(checkpoint_var, checkpoint_call.as_or_throw<Call>(), adjoint_var, builder_)
              .cast<Var>();
      Tuple args = call->args[1].as_or_throw<Tuple>();
      auto* tuple_ty = GetTypeAs<TupleTypeObj>(partials);
      if (!tuple_ty) {
        // result_var is a tensor
        TVM_FFI_ICHECK(args->fields.size() == 1);
        UpdateAdjoint(args->fields[0], partials);
      } else {
        TVM_FFI_ICHECK(args->fields.size() == tuple_ty->fields.size());
        for (int i = 0; i < static_cast<int>(args->fields.size()); ++i) {
          UpdateAdjoint(args->fields[i], TupleGetItem(partials, i));
        }
      }
    } else {
      const ffi::Array<Expr>& partials = gradient_op_map[call_op](
          checkpoint_var, checkpoint_call.as_or_throw<Call>(), adjoint_var, builder_);
      TVM_FFI_ICHECK(partials.size() == call->args.size()) << "partials number != inputs number";
      for (size_t i = 0; i < partials.size(); ++i) {
        Expr partial = partials[i];
        if (IsCallNoGrad(partial)) {  // no grad: don't update
          continue;
        }
        UpdateAdjoint(call->args[i], partial);
      }
    }
  }

  // For Tuple nodes, we would iterate over the input tuple and update adjoint exprs for each input
  // e.g.
  // a = (b, c) -->
  // b_adjoint += a_adjoint_var[0], c_adjoint += a_adjoint_var[1]
  //
  // a = ((b, c), d) -->
  // b_adjoint += a_adjoint_var[0][0], c_adjoint += a_adjoint_var[0][1],
  // d_adjoint += a_adjoint_var[1]
  void VisitBinding_(const VarBindingObj* binding, const TupleObj* tuple) final {
    UpdateAdjoint(ffi::GetRef<Tuple>(tuple), adjoint_var_map_[binding->var]);
  }

  // For TupleGetItem nodes, we do a partial update
  // e.g.
  // b = a[0] -->
  // a_adjoint[0] += b_adjoint_var
  // If a_adjoint does not exist, we would create a zeros tuple as a_adjoint first, and then add
  void VisitBinding_(const VarBindingObj* binding, const TupleGetItemObj* tuple_get_item) final {
    TVM_FFI_ICHECK(tuple_get_item->tuple->IsInstance<VarObj>())
        << "The tuple field of a TupleGetItem is not bound to a Var";
    auto* tuple_ty = GetTypeAs<TupleTypeObj>(tuple_get_item->tuple);
    TVM_FFI_ICHECK(tuple_ty) << "The tuple field of a TupleGetItem must has a TupleType";

    const Var& tuple_var = tuple_get_item->tuple.as_or_throw<Var>();
    if (adjoint_var_map_.count(tuple_var) == 0) {
      auto nested_zeros = NestedZeros(ffi::GetRef<Type>(tuple_ty)).as_or_throw<Tuple>();
      auto tuple_fields = nested_zeros->fields;
      tuple_fields.Set(tuple_get_item->index, adjoint_var_map_[binding->var]);
      EmitAdjoint(tuple_var, Tuple(tuple_fields), false);
    } else {
      Expr updated_adjoint = AddInTuple(adjoint_var_map_[tuple_var], tuple_get_item->index,
                                        adjoint_var_map_[binding->var]);
      EmitAdjoint(tuple_var, updated_adjoint, false);
    }
  }

  // For assign nodes, we add the adjoint of output to the adjoint of input
  void VisitBinding_(const VarBindingObj* binding, const DataflowVarObj* var) final {
    UpdateAdjoint(ffi::GetRef<Var>(var), adjoint_var_map_[binding->var]);
  }

  void VisitBinding_(const VarBindingObj* binding, const VarObj* var) final {
    UpdateAdjoint(ffi::GetRef<Var>(var), adjoint_var_map_[binding->var]);
  }

  // For constant nodes, we do not have to handle it because it does not contribute to the adjoint
  void VisitBinding_(const VarBindingObj* binding, const GenericConstObj* var) final { return; }

  // Add partial to the adjoint of expr
  // expr may be a argument of a func call / tuple definition. Its type can be
  // 1) var 2) constant (in this case, the adjoint will not be updated)
  // 3) (maybe nested) tuple of vars / constant
  //
  // We use NestedMsg to simplify handling (nested) tuples. That requires converting partial from
  // expr to NestedMsg or backwards.
  void UpdateAdjoint(const Expr& expr, const Expr& partial) {
    AdjointMsg partial_msg = ExprToAdjointMsg(builder_->Normalize(partial));
    DecomposeNestedMsg(expr, partial_msg, [&](Expr leaf, AdjointMsg msg) {
      if (leaf->IsInstance<VarObj>()) {
        const Var& v = leaf.as_or_throw<Var>();
        Expr updated_adjoint_expr = builder_->Normalize(AdjointMsgToExpr(msg));
        auto it = adjoint_var_map_.find(v);
        if (it != adjoint_var_map_.end()) {
          updated_adjoint_expr = TupleAwareAdd((*it).second, updated_adjoint_expr);
        }
        EmitAdjoint(v, updated_adjoint_expr, false);
      } else if (leaf->IsInstance<GenericConstObj>()) {
        // nothing to do
      } else if (leaf->IsInstance<ShapeExprObj>()) {
        // must be no grad
        TVM_FFI_ICHECK(IsCallNoGrad(partial));
      } else {
        TVM_FFI_THROW(InternalError)
            << "UpdateAdjoint: leaf type not supported. Currently Var and GenericConst leaves "
               "are supported.";
      }
    });
  }

  // Handle the return value of the AD function.
  // Returns the new return value, which would be like:
  // Tuple(original_return_value,
  //       Tuple(adjoint_of_require_grads_1, adjoint_of_require_grads_2, ...))
  Expr Epilogue(const ffi::Array<Var>& require_grads, const Expr& orig_return_value) {
    // create adjoint variables for inputs, and then bind adjoints
    ffi::Array<Expr> out_adjoints;

    for (Var var : require_grads) {
      // var might be wrapped in start_checkpoint or end_checkpoint, so we should find the original
      // var first
      if (cp_collector_.var_mapping.count(var)) {
        var = cp_collector_.var_mapping.at(var);
      }
      // If the var don't have adjoint var, it do not contribute to the target. So its adjoint is
      // zeros
      auto it = adjoint_var_map_.find(var);
      if (it == adjoint_var_map_.end()) {
        UpdateAdjoint(var, NestedZeros(GetType(var)));
      }
      Var adjoint_output_var = EmitAdjoint(var, adjoint_var_map_[var], true);
      out_adjoints.push_back(adjoint_output_var);
    }

    return Tuple({orig_return_value, Tuple(out_adjoints)});
  }

  // Emit the adjoint expr as the name `original_var_name` + "_adjoint"
  Var EmitAdjoint(const Var& source_var, const Expr& adjoint, bool is_output) {
    Var adjoint_var = is_output ? builder_->EmitOutput(adjoint, source_var->name + "_adjoint_out")
                                : builder_->Emit(adjoint, source_var->name + "_adjoint");
    if (!is_output) {
      adjoint_var_map_.Set(source_var, adjoint_var);
    }
    return adjoint_var;
  }

  static bool IsCallNoGrad(const Expr& expr) {
    return expr->IsInstance<CallObj>() &&
           expr.as_or_throw<Call>()->op.same_as(Op::Get("relax.grad.no_grad"));
  }

  static Expr AdjointMsgToExpr(AdjointMsg msg) {
    return NestedMsgToExpr<Expr>(msg, [](ffi::Optional<Expr> leaf_expr) {
      if (!leaf_expr.has_value()) {
        TVM_FFI_THROW(InternalError) << "Null should not exist in AdjointMsg.";
      }
      return leaf_expr.value();
    });
  }

  static AdjointMsg ExprToAdjointMsg(Expr expr) {
    return MapToNestedMsgByType<Expr>(expr, [](Expr leaf) {
      TVM_FFI_ICHECK(GetTypeAs<TensorTypeObj>(leaf))
          << "The leaf of adjoint: " << leaf << " should have Type and be a Tensor.";
      return AdjointMsg(leaf);
    });
  }

  // Create a zeros Expr with specified type
  // When ty is TupleType, we would create a (nested) Tuple containing zeros
  static Expr NestedZeros(const Type& ty) {
    AdjointMsg msg = MapToNestedMsg<Expr>(ty, [](Type ty) {
      auto* tensor_ty = ty.as<TensorTypeObj>();
      TVM_FFI_ICHECK(tensor_ty) << "The leaf of adjoint should be a Tensor.";
      TVM_FFI_ICHECK(tensor_ty->shape.has_value()) << "Missing shape when building zeros tuple.";
      const Expr& init = zeros(tensor_ty->shape.value(), tensor_ty->dtype.value()->dtype);
      return init;
    });
    return AdjointMsgToExpr(msg);
  }

  // Return lhs + rhs. Requires lhs and rhs has the same Type.
  // Use NestedMsg to handle cases when lhs and rhs are tuples.
  static Expr TupleAwareAdd(const Expr& lhs, const Expr& rhs) {
    AdjointMsg res = CombineNestedMsg(
        ExprToAdjointMsg(lhs), ExprToAdjointMsg(rhs), [](Expr l_leaf, Expr r_leaf) {
          auto* ty = GetTypeAs<TensorTypeObj>(l_leaf);
          TVM_FFI_ICHECK(ty) << "The leaf of adjoint should have Type and be a Tensor.";
          TVM_FFI_ICHECK(GetTypeAs<TensorTypeObj>(r_leaf))
              << "The leaf of adjoint should have Type and be a Tensor.";
          Expr res = add(l_leaf, r_leaf);
          UpdateType(res, ffi::GetRef<Type>(ty));
          return res;
        });
    return AdjointMsgToExpr(res);
  }

  // Perform an addition in a specified position of tuple.
  // tuple[index] += increment
  // Impl:
  // Step 1) t1 = tuple[0], t2 = tuple[1], t3 = tuple[2]
  // Step 2）t2_new = t2 + increment (TupleAwareAdd)
  // Step 3) tuple_new = (t1, t2_new, t3)
  static Expr AddInTuple(const Expr& tuple, int index, const Expr& increment) {
    auto* ty = GetTypeAs<TupleTypeObj>(tuple);
    TVM_FFI_ICHECK(ty) << "The first argument of AddInTuple should have tuple type.";
    TVM_FFI_ICHECK(index >= 0 && index < static_cast<int>(ty->fields.size()));
    ffi::Array<Expr> res;
    for (size_t i = 0; i < ty->fields.size(); ++i) {
      Expr field = [&]() -> Expr {
        if (const auto* expr_tuple = tuple.as<TupleObj>()) {
          return expr_tuple->fields[i];
        }
        return TupleGetItem(tuple, i);
      }();
      if (static_cast<int>(i) == index) {
        field = TupleAwareAdd(field, increment);
      }
      res.push_back(field);
    }
    return Tuple(res);
  }

  // The block builder of the corresponding GradientMutator, to emit bindings
  BlockBuilder builder_;
  // Forward Var to its adjoint Var
  ffi::Map<Var, Var> adjoint_var_map_;
  // information collected by CheckpointCollector
  CheckpointCollector cp_collector_;
  // The generator for checkpoint bindings
  CheckpointGenerator checkpoint_generator_;
};

class GradientMutator : private ExprMutator {
 public:
  static IRModule Transform(IRModule mod, ffi::String func_name,
                            ffi::Optional<ffi::Array<Var>> require_grads, int target_index) {
    // Step 1. Copy function
    auto* old_func = mod->Lookup(func_name).as<FunctionObj>();
    TVM_FFI_ICHECK(old_func) << func_name << "is not a Relax Function";
    auto copier = FunctionCopier();
    auto new_func = copier.Copy(ffi::GetRef<Function>(old_func));

    // Step 2. Handle the checkpoints and eliminate start_checkpoint and end_checkpoint ops
    auto cp_collector = CheckpointCollector();
    new_func = cp_collector.Transform(new_func);

    // Step 3. Handle require_grads
    // When require_grads is not specified, it would be set to all params of the function
    if (!require_grads) {
      require_grads = new_func->params;
    } else {
      require_grads = CheckAndMapRequireGrads(require_grads.value(), copier.GetVarMap(), func_name);
    }

    // Step 4. Generate the adjoint function, use RemoveAllUnused to simplify it, and then return
    // the IRModule with the adjoint function
    return GradientMutator(mod, require_grads.value(), target_index, cp_collector)
        .AddAdjointFunction(new_func, func_name, true);
  }

 private:
  GradientMutator(const IRModule& module, const ffi::Array<Var>& require_grads, int target_index,
                  const CheckpointCollector& cp_collector)
      : ExprMutator(module),
        require_grads_(require_grads),
        cp_collector_(cp_collector),
        target_index_(target_index) {}

  // Add the adjoint function of func to the IRModule using BlockBuilder
  IRModule AddAdjointFunction(const Function& func, const ffi::String& func_name,
                              bool remove_all_unused = true) {
    // Step 4.1 forward -> forward + backward
    auto new_func = VisitExpr(func).as_or_throw<Function>();

    // Step 4.2 Convert call_tir_with_grad nodes into call_tir nodes
    // because call_tir_with_grad nodes is not actually implemented
    new_func = CallTIRWithGradEliminator::Transform(new_func);

    if (remove_all_unused) {
      new_func = RemoveAllUnused(new_func).as_or_throw<Function>();
    }

    // Step 4.3 Simplify specific patterns generated by the gradient pass. Especially, simplify
    // transpose + matmul patterns. For details see the document of SimplifyGradient
    new_func = SimplifyGradient(new_func);

    // Step 4.4 mark the transformed function as public
    // because the original function may be public, and have gsymbol attribute as func_name
    auto new_func_name = func_name + "_adjoint";
    auto new_func_with_gsymbol = WithAttr(new_func, tvm::attr::kGlobalSymbol.name, new_func_name);

    // Step 4.5 Add the transformed function to IRModule
    builder_->AddFunction(new_func_with_gsymbol, new_func_name);
    return builder_->GetContextIRModule();
  }

  Expr VisitExpr_(const FunctionObj* func) final {
    orig_params_ = func->params;
    Expr new_body = this->VisitExpr(func->body);

    return Function(func->params, new_body, std::nullopt, func->is_pure, func->attrs);
  }

  Expr VisitExpr_(const SeqExprObj* seq_expr) final {
    // TODO(chaofan, yixin): multiple blocks AD
    TVM_FFI_ICHECK(seq_expr->blocks.size() == 1) << "now only support one dataflow block";
    // TODO(chaofan, yixin): AD in non-dataflow block.
    TVM_FFI_ICHECK(seq_expr->blocks[0]->IsInstance<DataflowBlockObj>())
        << "now only support one dataflow block";

    // the return value should be a VarObj, and a scalar
    orig_return_expr_ = seq_expr->body;
    CheckAndSetTarget(seq_expr->body, target_index_);

    BindingBlock new_block = this->VisitBindingBlock(seq_expr->blocks[0]);
    return SeqExpr({new_block}, return_expr_);
  }

  BindingBlock VisitBindingBlock_(const DataflowBlockObj* block) final {
    builder_->BeginDataflowBlock();
    // accept bindings in the original block
    for (const auto& binding : block->bindings) {
      this->VisitBinding(binding);
    }

    // generate backward bindings and the return value
    return_expr_ = BackwardBindingGenerator::Generate(builder_, ffi::GetRef<DataflowBlock>(block),
                                                      require_grads_, target_var_, orig_params_,
                                                      orig_return_expr_, cp_collector_);

    return builder_->EndBlock();
  }

  static bool IsFloatTensorType(const Type& ty) {
    auto* tensor_ty = ty.as<TensorTypeObj>();
    // Gradient eligibility preserves the old float-kind check; lanes do not affect this policy.
    return tensor_ty && !tensor_ty->IsUnknownDtype() &&
           tensor_ty->dtype.value().MatchesCode(DLDataTypeCode::kDLFloat);
  }

  // When the return value is a Var, it is the target;
  // when the return value is a Tuple, the target is the target_index-th field of the return value
  // Check that the target should be a Var of scalar tensor ty
  void CheckAndSetTarget(const Expr& e, int target_index) {
    if (auto* var = e.as<VarObj>()) {
      TVM_FFI_ICHECK_EQ(target_index, 0)
          << "When the function has only one return value, target_index can "
             "only be 0. But the target_index specified is "
          << target_index;
      target_var_ = ffi::GetRef<Var>(var);
    } else if (auto* tuple = e.as<TupleObj>()) {
      TVM_FFI_ICHECK(target_index >= 0 && target_index < static_cast<int>(tuple->fields.size()))
          << "target_index should be in the range of the number of return values of the "
             "function. "
             "But the specified target_index is "
          << target_index << ", while the number of return values is " << tuple->fields.size();
      auto* var = tuple->fields[target_index].as<VarObj>();
      TVM_FFI_ICHECK(var) << "Target must be a Var, but the specified target is "
                          << tuple->fields[target_index];
      target_var_ = ffi::GetRef<Var>(var);
    } else {
      TVM_FFI_THROW(InternalError)
          << "The return value of the function must be Var or Tuple. However, the return "
             "value of the given function is "
          << e;
    }
    auto target_ty = GetType(target_var_);
    TVM_FFI_ICHECK(IsScalarTensor(target_ty) && IsFloatTensorType(target_ty))
        << "The differentiation target must be a float scalar (0-dim Tensor), but the Type "
           "of the given target "
        << target_var_ << " is " << GetType(target_var_);
  }

  // Check every Var in require_grads:
  // 1. there should be no duplicate var
  // 2. every var should be a parameter or a intermediate var in the function
  // 3. the type of the input var should be Tensor of floating point dtype, or Tuple of that
  static ffi::Array<Var> CheckAndMapRequireGrads(const ffi::Array<Var>& require_grads,
                                                 const ffi::Map<Var, Var>& var_map,
                                                 const ffi::String& func_name) {
    VarSet var_set;
    ffi::Array<Var> mapped_vars;
    for (const auto& var : require_grads) {
      auto it = var_map.find(var);
      TVM_FFI_ICHECK(it != var_map.end())
          << "There is no Var named " << var->name << " in the function " << func_name;
      TVM_FFI_ICHECK_EQ(var_set.count(var), 0) << "Var " << var->name << " appears more than once";
      var_set.emplace(var);
      mapped_vars.push_back((*it).second);

      TVM_FFI_ICHECK(IsNestedTensorConditioned(GetType(var), IsFloatTensorType))
          << "Only Tensors of floating point dtype or Tuples of float "
             "Tensors can require gradients, but the Type of Var "
          << var->name << " is " << GetType(var);
    }
    return mapped_vars;
  }

  // differentiation sources
  ffi::Array<Var> require_grads_;
  // information collected by CheckpointCollector
  CheckpointCollector cp_collector_;
  // the differentiation target
  int target_index_;
  Var target_var_{ffi::UnsafeInit{}};
  // the return value of the original function and the differentiated function
  ffi::Array<Var> orig_params_;
  Expr orig_return_expr_{ffi::UnsafeInit{}};
  Expr return_expr_{ffi::UnsafeInit{}};
};

namespace transform {

Pass Gradient(ffi::String func_name, ffi::Optional<ffi::Array<Var>> require_grads,
              int target_index) {
  auto pass_func = [=](IRModule mod, PassContext pc) {
    return relax::GradientMutator::Transform(mod, func_name, require_grads, target_index);
  };
  return CreateModulePass(/*pass_function=*/pass_func,
                          /*opt_level=*/0,
                          /*pass_name=*/"Gradient");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.Gradient", Gradient);
}

}  // namespace transform

}  // namespace relax
}  // namespace tvm
