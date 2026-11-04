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
 * \file relax/transform/rewrite_cuda_graph.cc
 * \brief Pass for transforming Relax module to execute with CUDA graph.
 *
 * CUDA graph provides a way to capture a sequence of CUDA kernel launches in the runtime and
 * save them as a graph. The graph can be executed multiple times with less overhead than launching
 * kernels individually. This pass rewrites the Relax module to execute with CUDA graph.
 *
 * The transformation is done in two steps:
 *
 * 1. Identify the regions that can be captured by CUDA graph and create the rewriting plan with
 * `CUDAGraphRewritePlanner`.
 *
 * A region is a subgraph of the Relax function that are executed statically. A region is executed
 * statically if 1) it only depends on the memory allocated internally in the Relax function with
 * constant shapes, 2) it only contains kernel launches (there are no control flow).
 *
 * This transformation is expected to run after `StaticPlanBlockMemory`. After
 * `StaticPlanBlockMemory`, all the tensors that can be statically allocated are allocated with
 * `R.memory.alloc_storage` and `R.memory.alloc_tensor`, while other tensors will be allocated via
 * `R.builtin.alloc_tensor`.
 *
 * `CUDAGraphRewritePlanner` is executed at the level of BindingBlock. It first identify all the
 * storage objects allocated with `R.memory.alloc_storage` within the BindingBlock, and then
 * identify the static regions by propagating starting from the storage objects.
 *
 * All the calls to `R.memory.alloc_storage` within the same BindingBlock are grouped into a single
 * new function. Each of the static regions are lifted to a new function.
 *
 * 2. Lift the regions identified in step 1 to a separate function and rewrite the original function
 * with `CUDAGraphRewriter`.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr_functor.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/backend.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/utils.h>
#include <tvm/tirx/stmt_functor.h>

#include <unordered_map>
#include <vector>

#include "../../support/arena.h"
#include "../../support/ordered_set.h"
namespace tvm {
namespace relax {

TVM_FFI_STATIC_INIT_BLOCK() {
  ::tvm::transform::PassContext::RegisterConfigOption<bool>("relax.backend.use_cuda_graph");
}

/*! \brief The rewriting plan of lifting a region for either allocation or capturing for cuda graph
 * execution
 */
struct LiftedFunctionRewritePlan {
  // The lifted function for allocation or capturing
  Function func{ffi::UnsafeInit{}};
  // Whether the lifted function is for allocation or capturing
  bool is_alloc;
  // The binding var before which the lifted function should be invoked
  const VarObj* launch_point;

  // Variable remappings between the original function and the lifted function

  // The bindings in the original function that are lifted
  std::vector<const VarBindingObj*> lifted_bindings;
  // The corresponding binding vars in the original function of the outputs of the lifted function
  // to the index of the element in the output tuple of the lifted function.
  std::unordered_map<const VarObj*, int> outputs;
  // The corresponding binding vars in the original function of the inputs of the lifted function
  std::vector<const VarObj*> inputs;
  // The tirx vars in the original function that are propagated to the lifted function
  ffi::Optional<ShapeExpr> propogated_tir_vars = std::nullopt;
};

/*! \brief Builder of the lifted function for cuda graph capturing or allocations */
class FuncBuilder : public ExprMutator {
 public:
  /*!
   * \brief Add a binding to the new function
   * \param binding The binding to add
   */
  void AddBinding(const VarBindingObj* binding) { bindings_.push_back(binding); }

  /*!
   * \brief Mark a variable as the input of the new function.
   * \param var The variable to mark as input
   */
  void MarkInput(const VarObj* var) { inputs_.push_back(var); }
  /*!
   * \brief Mark a TIR variable as the ShapeExpr input of the new function.
   * \param var The variable to mark as input
   */
  void MarkShapeExprInput(const PrimVar& var) { shape_expr_inputs_.push_back(var); }
  /*!
   * \brief Mark a variable as the output of the new function. The variable must be the LHS of an
   * existing binding in the new function.
   * \param var The variable to mark as output
   */
  void MarkOutput(const VarObj* var) { outputs_.push_back(var); }

  /*! \brief Get the number of bindings in the new function */
  auto size() const { return bindings_.size(); }

  /*! \brief Build the new function */
  Function Build() {
    ffi::Array<Var> params;
    ffi::Optional<Var> shape_expr = std::nullopt;
    if (shape_expr_inputs_.size()) {
      ffi::Array<PrimExpr> tir_vars;
      for (const PrimVar& var : shape_expr_inputs_) {
        PrimVar new_var = var.CopyWithSuffix("");
        var_remap_.insert_or_assign(var, new_var);
        tir_vars.push_back(new_var);
      }
      shape_expr = Var("shape_expr", ShapeType(tir_vars));
    }
    // Set up the parameters
    for (const auto* input : inputs_) {
      auto new_var = Var(input->name, VisitExprDepTypeField(input->ty.as_or_throw<Type>()));
      var_remap_.insert_or_assign(ffi::GetRef<Var>(input), new_var);
      params.push_back(new_var);
    }
    if (shape_expr) {
      params.push_back(shape_expr.value());
    }
    // Emit the function body
    builder_->BeginBindingBlock();
    for (const auto* binding : bindings_) {
      VisitBinding_(binding);
    }
    // Set up the outputs
    ffi::Array<Expr> outputs;
    for (const auto* var : outputs_) {
      outputs.push_back(VisitExpr_(var));
    }
    auto output = builder_->Emit(Tuple(outputs));
    auto block = builder_->EndBlock();
    auto body = builder_->Normalize(SeqExpr({block}, output));
    ffi::Map<ffi::String, Any> attrs;
    attrs.Set(tvm::relax::attr::kForcePure.name, true);
    auto func = Function(params, body, output->ty.as_or_throw<Type>(),
                         /*is_pure=*/true, /*attrs=*/DictAttrs(attrs));
    return func;
  }

  support::OrderedSet<const VarObj*> inputs_;
  support::OrderedSet<const VarObj*> outputs_;
  support::OrderedSet<PrimVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> shape_expr_inputs_;
  std::vector<const VarBindingObj*> bindings_;
};

// Collect the storage objects that are used as the function output
class OutputStorageCollector : public ExprVisitor {
 public:
  static std::unordered_set<const VarObj*> Collect(const Function& func) {
    OutputStorageCollector collector;
    collector.VisitExpr(func);
    return std::move(collector.output_storages_);
  }

 private:
  void VisitExpr_(const SeqExprObj* seq_expr) final {
    auto output_vars = FreeVars(seq_expr->body);
    for (const auto& var : output_vars) {
      output_vars_.insert(var.get());
    }
    // Visit the blocks in reverse order for backward propagation
    for (auto it = seq_expr->blocks.rbegin(); it != seq_expr->blocks.rend(); ++it) {
      VisitBindingBlock(*it);
    }
  }

  void VisitBinding_(const VarBindingObj* binding, const CallObj* call) final {
    static const auto mem_alloc_tensor_op = Op::Get("relax.memory.alloc_tensor");
    if (output_vars_.count(binding->var.get()) && call->op.same_as(mem_alloc_tensor_op)) {
      output_storages_.insert(call->args[0].as<VarObj>());
    }
  }

  void VisitBindingBlock_(const BindingBlockObj* binding_block) override {
    // Visit the bindings in reverse order
    for (auto it = binding_block->bindings.rbegin(); it != binding_block->bindings.rend(); ++it) {
      VisitBinding(*it);
    }
  }

  void VisitBinding_(const VarBindingObj* binding, const VarObj* var) final {
    if (output_vars_.count(binding->var.get())) {
      output_vars_.insert(var);
    }
  }

  void VisitBinding_(const VarBindingObj* binding, const TupleObj* tuple) final {
    if (output_vars_.count(binding->var.get())) {
      for (const auto& field : tuple->fields) {
        output_vars_.insert(field.as<VarObj>());
      }
    }
  }

  std::unordered_set<const VarObj*> output_storages_;
  std::unordered_set<const VarObj*> output_vars_;
};

/*!
 * \brief The planner for rewriting the function to enable cuda graph capturing.
 */
class CUDAGraphRewritePlanner : public ExprVisitor {
 public:
  explicit CUDAGraphRewritePlanner(const IRModule& mod, support::Arena* arena)
      : mod_(mod), arena_(arena) {}
  std::pair<std::vector<LiftedFunctionRewritePlan*>, std::vector<LiftedFunctionRewritePlan*>>
  Plan() {
    for (const auto& pair : mod_->functions) {
      if (pair.second->IsInstance<FunctionObj>()) {
        // If a function has the num_input attribute, the last func->params.size() - num_inputs
        // inputs are assumed to be fixed and thus they can be captured into a cuda graph.
        // The symbolic variables in the type of the fixed inputs (weights) are also allowed
        // to be captured.
        // If the hints for capturing symbolic variables via
        // 'relax.rewrite_cuda_graph.capture_symbolic_vars' annotation, the actual variables with
        // these names are extracted from the type for the capturing.
        const auto& func = pair.second.as_or_throw<Function>();
        int64_t num_inputs =
            func->attrs.GetAttr(tvm::relax::attr::kNumInput).value_or(func->params.size());
        auto capture_symbolic_var_name_hints = ExtractSymbolicVarHints(func);
        for (int i = 0; i < static_cast<int>(func->params.size()); ++i) {
          ffi::Array<tvm::Var> symbolic_vars =
              DefinableTIRVarsInType(func->params[i]->ty.as_or_throw<Type>());
          if (i < num_inputs) {
            for (const auto& symbolic_var : symbolic_vars) {
              auto prim_var = symbolic_var.as_or_throw<PrimVar>();
              if (capture_symbolic_var_name_hints.count(symbolic_var->name)) {
                capture_symbolic_vars_.insert(prim_var);
              }
            }
          } else {
            static_vars_.insert(func->params[i].get());
            for (const auto& symbolic_var : symbolic_vars) {
              capture_symbolic_vars_.insert(symbolic_var.as_or_throw<PrimVar>());
            }
          }
        }
        disabled_storage_vars_ = OutputStorageCollector::Collect(func);
        VisitExpr(func);
      }
    }
    auto region_to_plan = [&](FuncBuilder* region, bool is_alloc) -> LiftedFunctionRewritePlan* {
      auto* plan = arena_->make<LiftedFunctionRewritePlan>();
      plan->is_alloc = true;
      plan->func = region->Build();
      TVM_FFI_ICHECK(region->size());
      plan->launch_point = region->bindings_.front()->var.get();
      plan->is_alloc = is_alloc;
      plan->lifted_bindings = std::move(region->bindings_);
      if (region->shape_expr_inputs_.size()) {
        ffi::Array<PrimExpr> tir_vars;
        for (const PrimVar& var : region->shape_expr_inputs_) {
          tir_vars.push_back(var);
        }
        plan->propogated_tir_vars = ShapeExpr(tir_vars);
      }
      plan->inputs.assign(region->inputs_.begin(), region->inputs_.end());
      for (const auto* var : region->outputs_) {
        plan->outputs[var] = plan->outputs.size();
      }
      return plan;
    };

    std::vector<LiftedFunctionRewritePlan*> alloc_plans, capture_plans;
    alloc_plans.reserve(alloc_storages_.size());
    capture_plans.reserve(captured_regions_.size());
    std::transform(alloc_storages_.begin(), alloc_storages_.end(), std::back_inserter(alloc_plans),
                   [&](FuncBuilder* region) { return region_to_plan(region, /*is_alloc=*/true); });
    std::transform(captured_regions_.begin(), captured_regions_.end(),
                   std::back_inserter(capture_plans),
                   [&](FuncBuilder* region) { return region_to_plan(region, /*is_alloc=*/false); });
    return {std::move(alloc_plans), std::move(capture_plans)};
  }

  /*!
   * \brief Extract the name hints of the symbolic variables that are allowed to be captured
   * from the function attributes.
   */
  std::unordered_set<ffi::String> ExtractSymbolicVarHints(const Function& func) {
    auto symbolic_var_names = func->attrs.GetAttr(tvm::relax::attr::kCaptureSymbolicVars)
                                  .value_or(ffi::Array<ffi::String>());
    return {symbolic_var_names.begin(), symbolic_var_names.end()};
  }

  /*!
   *\brief Start a new static region. This method should be called when encountering a
   * CUDA kernel launch (calls to tirx::Function or ExternFunc) that only depends on static
   *parameters.
   */
  void StartRegion() { current_block_scope_.capture_builder = arena_->make<FuncBuilder>(); }

  /*!
   * \brief Finish a static region. This method should be called when non-static bindings or
   * unsupported operations are encountered.
   */
  void EndRegion() {
    if (current_block_scope_.capture_builder && current_block_scope_.capture_builder->size()) {
      captured_regions_.emplace_back(current_block_scope_.capture_builder);
    }
    current_block_scope_.capture_builder = nullptr;
  }

  void VisitExpr_(const FunctionObj* func) final {
    current_function_scope_.alloc_storage_builder = arena_->make<FuncBuilder>();
    ExprVisitor::VisitExpr_(func);
    if (current_function_scope_.alloc_storage_builder->outputs_.size()) {
      alloc_storages_.emplace_back(current_function_scope_.alloc_storage_builder);
    }
    current_function_scope_.alloc_storage_builder = nullptr;
  }

  void VisitBindingBlock_(const BindingBlockObj* binding_block) final {
    BindingSBlockScope new_scope;
    std::swap(new_scope, current_block_scope_);
    for (const auto& binding : binding_block->bindings) {
      VisitBinding(binding);
    }
    EndRegion();
    std::swap(new_scope, current_block_scope_);
  }

  void VisitBinding_(const VarBindingObj* binding, const CallObj* call) final {
    static const auto mem_alloc_storage_op = Op::Get("relax.memory.alloc_storage");
    static const auto builtin_alloc_tensor_op = Op::Get("relax.builtin.alloc_tensor");
    static const auto call_builtin_with_ctx_op = Op::Get("relax.call_builtin_with_ctx");

    if (call->op.same_as(mem_alloc_storage_op)) {
      if (IsStaticAllocStorage(binding)) {
        AddStaticBinding(binding, /*is_alloc_storage=*/true);
      }
      return;
    } else if (call->op.same_as(builtin_alloc_tensor_op)) {
      return;
    }

    const auto* call_gv = call->op.as<GlobalVarObj>();
    bool call_function =
        call_gv ? mod_->Lookup(ffi::GetRef<GlobalVar>(call_gv))->IsInstance<tirx::FunctionObj>()
                : false;

    // Check whether the call can be lifted to the capture function. It requires all the arguments
    // to be static and the call to be a kernel launch or a pure operation (e.g. memory view).
    std::vector<const VarObj*> args;
    std::vector<PrimVar> tir_vars;
    bool is_all_static = [&]() {
      if (!IsStatic(call->args, &args, &tir_vars)) {
        return false;
      }
      if (call_gv != nullptr && !call_function) {
        // calls to other Relax functions are not allowed
        return false;
      }
      if (const auto* extern_func = call->op.as<ExternFuncObj>();
          extern_func != nullptr && extern_func->global_symbol.starts_with("vm.builtin")) {
        return false;
      }
      return true;
    }();

    if (is_all_static) {
      bool is_kernel_launch = [&]() {
        static const auto null_value_op = Op::Get("relax.null_value");

        if (call_function) {
          return true;
        }
        if (call->op.as<ExternFuncObj>()) {
          return true;
        }
        if (const auto* op = call->op.as<OpObj>()) {
          return !op->name.starts_with("relax.memory") && !op->name.starts_with("relax.builtin") &&
                 op->name != "relax.reshape" && !ffi::GetRef<Op>(op).same_as(null_value_op) &&
                 !ffi::GetRef<Op>(op).same_as(call_builtin_with_ctx_op);
        }
        return false;
      }();
      if (current_block_scope_.capture_builder == nullptr && is_kernel_launch) {
        StartRegion();
      }
      AddStaticBinding(binding, /*is_alloc_storage=*/false);
      MarkAsFuncInput(args, tir_vars);
    } else {
      EndRegion();
    }

    MarkAsFuncOutput(args);
  }

  void MarkAsFuncInput(const std::vector<const VarObj*>& vars,
                       const std::vector<PrimVar>& tir_vars = {}) {
    if (current_block_scope_.capture_builder == nullptr) {
      return;
    }
    for (const VarObj* var : vars) {
      auto it = binding_to_region_.find(var);
      if (it == binding_to_region_.end() || it->second != current_block_scope_.capture_builder) {
        current_block_scope_.capture_builder->MarkInput(var);
      }
    }
    for (const PrimVar& tir_var : tir_vars) {
      current_block_scope_.capture_builder->MarkShapeExprInput(tir_var);
    }
  }

  void MarkAsFuncOutput(const std::vector<const VarObj*>& vars) {
    for (const VarObj* var : vars) {
      if (auto it = binding_to_region_.find(var);
          it != binding_to_region_.end() && it->second != current_block_scope_.capture_builder) {
        it->second->MarkOutput(var);
      }
    }
  }

  void VisitBinding_(const VarBindingObj* binding, const VarObj* var) final {
    if (IsStatic(ffi::GetRef<Var>(var))) {
      AddStaticBinding(binding, false);
      MarkAsFuncInput({var});
    } else {
      EndRegion();
    }
    MarkAsFuncOutput({var});
  }

  void VisitBinding_(const VarBindingObj* binding, const GenericConstObj* constant) final {
    AddStaticBinding(binding, false);
  }

  void VisitBinding_(const MatchCastObj* binding) final {
    // A match_cast stays in the original function even when its value was lifted into the
    // current capture region, so the region has to return that value.
    if (const auto* var = binding->value.as<VarObj>()) {
      if (auto it = binding_to_region_.find(var); it != binding_to_region_.end()) {
        it->second->MarkOutput(var);
      }
    }
    ExprVisitor::VisitBinding_(binding);
  }

  void VisitBinding_(const VarBindingObj* binding, const TupleObj* tuple) final {
    std::vector<const VarObj*> args;
    std::vector<PrimVar> tir_vars;
    if (IsStatic(tuple->fields, &args, &tir_vars)) {
      AddStaticBinding(binding, false);
      MarkAsFuncInput(args, tir_vars);
    } else {
      EndRegion();
    }
    MarkAsFuncOutput(args);
  }

  void VisitBinding_(const VarBindingObj* binding, const TupleGetItemObj* tuple_get_item) final {
    const VarObj* tuple = tuple_get_item->tuple.as<VarObj>();
    TVM_FFI_ICHECK(tuple);
    if (IsStatic(tuple_get_item->tuple)) {
      AddStaticBinding(binding, false);
      MarkAsFuncInput({tuple});
    } else {
      EndRegion();
    }
    MarkAsFuncOutput({tuple});
  }

  bool IsStatic(const PrimExpr& expr,
                [[maybe_unused]] std::vector<const VarObj*>* vars_collector = nullptr,
                std::vector<PrimVar>* tir_vars_collector = nullptr) {
    bool is_static = true;
    std::unordered_set<const ffi::Object*> visited;
    auto walk_fn = [&](const tvm::Var& var) -> ffi::Expected<ffi::WalkResult> {
      auto prim_var = var.as<PrimVar>();
      if (!prim_var || !visited.insert(prim_var.value().get()).second) {
        return ffi::WalkResult::Advance();
      }
      if (!capture_symbolic_vars_.count(prim_var.value())) {
        is_static = false;
        return ffi::WalkResult::Advance();
      }
      if (tir_vars_collector != nullptr) {
        tir_vars_collector->push_back(prim_var.value());
      }
      return ffi::WalkResult::Advance();
    };
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(expr, walk_fn);
    return is_static;
  }

  bool IsStatic(const Expr& expr, std::vector<const VarObj*>* vars_collector = nullptr,
                std::vector<PrimVar>* tir_vars_collector = nullptr) {
    if (expr->IsInstance<GenericConstObj>() || expr->IsInstance<DataTypeImmObj>() ||
        expr->IsInstance<StringImmObj>() || expr->IsInstance<GlobalVarObj>()) {
      return true;
    }
    if (const auto* var = expr.as<VarObj>();
        var && (expr.as<DataflowVarObj>() || static_vars_.count(var))) {
      return IsStaticRuntimeVar(var, vars_collector, tir_vars_collector);
    }
    if (auto prim_value = expr.as<PrimExpr>()) {
      return IsStatic(prim_value.value(), vars_collector, tir_vars_collector);
    }
    if (const auto* var = expr.as<VarObj>()) {
      return IsStaticRuntimeVar(var, vars_collector, tir_vars_collector);
    }

    if (const auto* shape = expr.as<ShapeExprObj>()) {
      return IsStatic(shape->values, vars_collector, tir_vars_collector);
    }
    if (const auto* tuple = expr.as<TupleObj>()) {
      return IsStatic(tuple->fields, vars_collector, tir_vars_collector);
    }
    return false;
  }

  bool IsStaticRuntimeVar(const VarObj* var, std::vector<const VarObj*>* vars_collector,
                          std::vector<PrimVar>* tir_vars_collector) {
    if (vars_collector != nullptr) {
      vars_collector->push_back(var);
    }
    return static_vars_.count(var) &&
           IsStatic(var->ty.as_or_throw<Type>(), vars_collector, tir_vars_collector);
  }

  template <typename T>
  bool IsStatic(const ffi::Array<T>& exprs, std::vector<const VarObj*>* vars_collector = nullptr,
                std::vector<PrimVar>* tir_vars_collector = nullptr) {
    bool result = true;
    for (const auto& expr : exprs) {
      // If vars_collector is provided, we will collect all the vars in the exprs and we should
      // not perform short-circuiting.
      result &= IsStatic(expr, vars_collector, tir_vars_collector);
      if (vars_collector == nullptr && tir_vars_collector == nullptr && !result) {
        return false;
      }
    }
    return result;
  }

  bool IsStatic(const Type& ty, std::vector<const VarObj*>* vars_collector = nullptr,
                std::vector<PrimVar>* tir_vars_collector = nullptr) {
    if (const auto* tensor_ty = ty.as<TensorTypeObj>()) {
      if (auto shape = tensor_ty->GetShape()) {
        return IsStatic(shape.value(), vars_collector, tir_vars_collector);
      }
    } else if (const auto* shape_ty = ty.as<ShapeTypeObj>()) {
      if (shape_ty->values) {
        return IsStatic(shape_ty->values.value(), vars_collector, tir_vars_collector);
      }
    } else if (const auto* tuple_ty = ty.as<TupleTypeObj>()) {
      return IsStatic(tuple_ty->fields, vars_collector, tir_vars_collector);
    } else if (ty.as<AnyTypeObj>() || ty.as<PrimTypeObj>()) {
      return true;
    }
    return false;
  }

 private:
  bool IsStaticAllocStorage(const VarBindingObj* binding) {
    if (disabled_storage_vars_.count(binding->var.get())) {
      return false;
    }
    // Check if the allocation has constant shape
    const auto* alloc_storage_call = binding->value.as<CallObj>();
    auto shape = alloc_storage_call->args[0].as_or_throw<ShapeExpr>();
    return std::all_of(shape->values.begin(), shape->values.end(),
                       [](const PrimExpr& expr) { return expr.as<IntImmObj>() != nullptr; });
  }

  /*!
   * \brief Add a static bindings. This is used to mark the bindings that are known to be static
   * and further propagate to other bindings.
   * \param binding the binding to add
   * \param is_alloc_storage whether the binding is call to R.memory.alloc_storage
   */
  void AddStaticBinding(const VarBindingObj* binding, bool is_alloc_storage) {
    if (is_alloc_storage) {
      current_function_scope_.alloc_storage_builder->AddBinding(binding);
      binding_to_region_[binding->var.get()] = current_function_scope_.alloc_storage_builder;
    } else if (current_block_scope_.capture_builder != nullptr) {
      // Add the binding if the capture builder exists. It is possible that capture builder is
      // null when it is not capturing. This is the case that there are not yet any kernel launches
      // encountered, in this case static bindings (e.g. binding of other non-kernel-launch
      // operations) are marked but are not lifted.
      current_block_scope_.capture_builder->AddBinding(binding);
      binding_to_region_[binding->var.get()] = current_block_scope_.capture_builder;
    }
    static_vars_.emplace(binding->var.get());
  }

  /*! \brief The states of the current scope (the BindingBlock) which is a FuncBuilder.
   * The FuncBuilder are initialized with nullptr, meaning the planner is currently not doing any
   * lifting. They are initialized lazily when a binding that can be lifted is encountered.
   * They are reset to nullptr when an unsupported operation is encountered.
   */
  struct BindingSBlockScope {
    FuncBuilder* capture_builder = nullptr;  // The builder for the capture function
  };

  /*! \brief The states of the current function scope which is a FuncBuilder to build the storage
   * allocation function.
   */
  struct FunctionScope {
    FuncBuilder* alloc_storage_builder = nullptr;  // The builder for the allocation function
  };

  // The IRModule
  IRModule mod_;
  // States of the current block scope
  BindingSBlockScope current_block_scope_;
  // States of the current function scope
  FunctionScope current_function_scope_;
  // Variables whose buffer address is fixed
  std::unordered_set<const VarObj*> static_vars_;
  // Symbolic variables that are allowed to be captured. This can come from symbolic shapes of
  // weights or hints in the function annotations.
  std::unordered_set<PrimVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> capture_symbolic_vars_;
  // Binding to the FuncBuilder if the binding is lifted. This is used to update the inputs/outputs
  // of the lifted function when its binding is used outside.
  std::unordered_map<const VarObj*, FuncBuilder*> binding_to_region_;
  // The regions for capturing.
  std::vector<FuncBuilder*> captured_regions_;
  // The regions for allocation.
  std::vector<FuncBuilder*> alloc_storages_;
  // The binding variables that are not allowed to be captured.
  std::unordered_set<const VarObj*> disabled_storage_vars_;
  // The arena.
  support::Arena* arena_;
};

/*!
 * \brief Merge storage allocations from different functions by reusing the largest allocation that
 * can be shared among all the functions. The original rewriting plans are updated in-place to use
 * the merged storage allocations.
 *
 * When multiple functions are rewritten to be executed with CUDA graph, the storage allocations
 * from different functions can be reused. This functions merge multiple storage allocations
 * functions to a single function that allocates the sufficiently large storage to be shared among
 * all the functions.
 *
 * \param alloc_plans The allocation plans of the functions to be merged.
 * \return The new allocation function that merges the storage allocations.
 */
Function MergeAllocationPlans(const std::vector<LiftedFunctionRewritePlan*>& alloc_plans) {
  // The storage record that contains the size of the storage allocation and the binding of the
  // storage allocation.
  struct StorageRecord {
    // The size of the storage object in bytes
    int64_t size;
    // The binding of the storage allocation
    const VarBindingObj* binding;
    // The source rewriting plan that the storage record is from
    LiftedFunctionRewritePlan* src;

    bool operator<(const StorageRecord& other) const { return size < other.size; }
  };
  // Using an (ordered) map to make sure the result is deterministic
  std::map<ffi::String, std::vector<std::vector<StorageRecord>>> storage_records;
  static const auto mem_alloc_storage_op = Op::Get("relax.memory.alloc_storage");

  // Collect the storage records for each storage scope. Storage records are stored separately
  // for each original function.
  for (int plan_id = 0; plan_id < static_cast<int>(alloc_plans.size()); ++plan_id) {
    LiftedFunctionRewritePlan* plan = alloc_plans[plan_id];
    TVM_FFI_ICHECK(plan->is_alloc);
    for (const VarBindingObj* binding : plan->lifted_bindings) {
      // Extract the stroage record from the Call expr.
      Call alloc_storage = binding->value.as_or_throw<Call>();
      TVM_FFI_ICHECK(alloc_storage->op.same_as(mem_alloc_storage_op));
      auto storage_shape = alloc_storage->args[0].as_or_throw<ShapeExpr>();
      TVM_FFI_ICHECK_EQ(storage_shape->values.size(), 1);
      int64_t size = static_cast<int64_t>(storage_shape->values[0].as_or_throw<IntImm>()->value);
      int64_t virtual_device_id = static_cast<int64_t>(
          alloc_storage->args[1].as_or_throw<PrimExpr>().as_or_throw<IntImm>()->value);
      TVM_FFI_ICHECK_EQ(virtual_device_id, 0);
      ffi::String storage_scope = alloc_storage->args[2].as_or_throw<StringImm>()->value;
      auto [it, _] = storage_records.try_emplace(storage_scope, alloc_plans.size());
      it->second[plan_id].emplace_back(StorageRecord{size, binding, plan});
    }
  }

  // Merge the storage records within each storage scope.
  // This is achieved by sorting the storage records in descending order of size and then merging
  // storage allocations from different functions to the largest allocation that can be shared
  // among all the functions.
  // This assumes that multiple functions will not run concurrently.
  std::vector<const VarBindingObj*> merged_allocs;
  // Merge the storage records within each storage scope.
  for (auto& [storage_scope, curr_scope_records] : storage_records) {
    // The number of storages needed for the current storage scope, which is the maximum number of
    // storage records among all the functions.
    int num_storages = 0;
    for (auto& records_of_plan : curr_scope_records) {
      // Sort descending by size, preserve the original order if the sizes are equal.
      std::stable_sort(records_of_plan.rbegin(), records_of_plan.rend());
      num_storages = std::max(num_storages, static_cast<int>(records_of_plan.size()));
    }
    // The iterators to scan the storage records of all functions from the left to the right
    // at the same time.
    std::vector<int> iters(alloc_plans.size(), 0);
    for (int i = 0; i < num_storages; i++) {
      // The storage records from different functions that can be merged to the same storage.
      std::vector<StorageRecord> to_merge;
      for (int plan_index = 0; plan_index < static_cast<int>(curr_scope_records.size());
           plan_index++) {
        if (iters[plan_index] < static_cast<int>(curr_scope_records[plan_index].size())) {
          to_merge.push_back(curr_scope_records[plan_index][iters[plan_index]++]);
        }
      }
      const StorageRecord& largest_storage =
          *std::max_element(to_merge.begin(), to_merge.end(),
                            [](const auto& lhs, const auto& rhs) { return lhs < rhs; });
      // Merge the records to the largest allocation by updating the index of the output element
      // to that of the new allocation function.
      int storage_index = static_cast<int>(merged_allocs.size());
      for (const StorageRecord& rec : to_merge) {
        auto* plan = rec.src;
        plan->outputs.at(rec.binding->var.get()) = storage_index;
      }
      merged_allocs.push_back(largest_storage.binding);
    }
  }
  // Create the new allocation function for the merged allocations.
  FuncBuilder builder;
  for (const auto* binding : merged_allocs) {
    builder.AddBinding(binding);
    builder.MarkOutput(binding->var.get());
  }
  return builder.Build();
}

/*! \brief The rewriter for CUDA graph */
class CUDAGraphRewriter : public ExprMutator {
 public:
  explicit CUDAGraphRewriter(const IRModule& mod) : ExprMutator(mod) {}

  IRModule Rewrite() {
    CUDAGraphRewritePlanner planner(builder_->GetContextIRModule(), &arena_);

    // Collect the target functions for rewriting before any mutation.
    std::vector<std::pair<GlobalVar, Function>> target_functions;
    for (const auto& [gv, func] : builder_->GetContextIRModule()->functions) {
      if (func->IsInstance<FunctionObj>()) {
        target_functions.emplace_back(gv, func.as_or_throw<Function>());
      }
    }

    auto [alloc_plans, capture_plans] = planner.Plan();
    if (alloc_plans.size()) {
      auto global_alloc_func = MergeAllocationPlans(alloc_plans);
      gv_global_alloc_ = builder_->AddFunction(global_alloc_func, "cuda_graph_alloc");
    }
    for (const auto* plan : alloc_plans) {
      subgraph_launches_[plan->launch_point] = plan;
    }
    for (const auto* plan : capture_plans) {
      subgraph_launches_[plan->launch_point] = plan;
    }

    for (const auto& [gv, func] : target_functions) {
      current_func_ = gv;
      auto new_func = VisitExpr(func).as_or_throw<Function>();
      if (!new_func.same_as(func)) {
        builder_->UpdateFunction(gv, new_func);
      }
    }
    return builder_->GetContextIRModule();
  }

  void LaunchSubgraph(const VarBindingObj* op, const LiftedFunctionRewritePlan* plan) {
    static const auto call_builtin_with_ctx_op = Op::Get("relax.call_builtin_with_ctx");
    static const auto& builtin_run_or_capture = ExternFunc("vm.builtin.cuda_graph.run_or_capture");
    static const auto& builtin_get_cached_alloc =
        ExternFunc("vm.builtin.cuda_graph.get_cached_alloc");

    Expr launch_subgraph = [&]() -> Expr {
      if (plan->is_alloc) {
        // Storage allocation should be fully static and shouldn't depend on any symbolic variables.
        TVM_FFI_ICHECK(!plan->propogated_tir_vars.has_value());
        TVM_FFI_ICHECK(plan->inputs.empty());
        auto gv_alloc = gv_global_alloc_.value();
        auto ret_ty = gv_alloc->ty.as_or_throw<FuncType>()->ret;
        return Call(Type::Missing(), call_builtin_with_ctx_op,
                    {builtin_get_cached_alloc, Tuple({gv_alloc, PrimExpr(IntImm::Int64(0))})},
                    std::nullopt, {ret_ty});
      } else {
        auto gv_func = builder_->AddFunction(
            plan->func, current_func_.value()->name_hint + "_cuda_graph_capture");
        Type call_ty = plan->func->ret_ty;
        // Arguments of the lifted function
        ffi::Array<Expr> args;
        for (const auto& arg : plan->inputs) {
          args.push_back(VisitExpr_(arg));
        }
        if (plan->propogated_tir_vars.has_value()) {
          ShapeExpr propogated_tir_vars = plan->propogated_tir_vars.value();
          args.push_back(propogated_tir_vars);
          // The ret_ty of the lifted function can contain symbolic variables. We need to
          // bind the symbolic parameters to the actual values.
          const auto& shape_expr = plan->func->params.back();
          auto symbolic_params = shape_expr->ty.as_or_throw<ShapeType>()->values.value();
          ffi::Map<Var, Expr> var_remap;
          TVM_FFI_ICHECK_EQ(symbolic_params.size(), propogated_tir_vars->values.size());
          for (int i = 0; i < static_cast<int>(symbolic_params.size()); ++i) {
            var_remap.Set(symbolic_params[i].as_or_throw<PrimVar>(),
                          propogated_tir_vars->values[i]);
          }
          call_ty = Bind(call_ty, var_remap);
        }
        // Arguments of builtin_run_or_capture
        ffi::Array<Expr> tuple_arg_fields{gv_func, Tuple(args),
                                          PrimExpr(IntImm::Int64(index_capture_++))};
        if (plan->propogated_tir_vars.has_value()) {
          // The shape expr is explicitly passed twice, one as the last argument of the lifted
          // function, one as the last argument of builtin_run_or_capture as the cache key.
          // Explicitly passing it twice simplifies the handling during the capture phase.
          tuple_arg_fields.push_back(plan->propogated_tir_vars.value());
        }
        return Call(Type::Missing(), call_builtin_with_ctx_op,
                    {builtin_run_or_capture, Tuple(tuple_arg_fields)}, std::nullopt, {call_ty});
      }
    }();
    Expr ret_value = builder_->Emit(launch_subgraph);
    for (const auto& [var, tuple_index] : plan->outputs) {
      var_redef_.insert_or_assign(var, TupleGetItem(ret_value, tuple_index));
    }
    std::transform(plan->lifted_bindings.begin(), plan->lifted_bindings.end(),
                   std::inserter(lifted_binding_vars_, lifted_binding_vars_.end()),
                   [](const BindingObj* binding) { return binding->var.get(); });
  }

  void VisitBinding_(const VarBindingObj* op) final {
    if (subgraph_launches_.count(op->var.get())) {
      LaunchSubgraph(op, subgraph_launches_[op->var.get()]);
    }
    if (auto it = var_redef_.find(op->var.get());
        it != var_redef_.end() && !var_remap_.count(op->var)) {
      EmitRedef(op->var.get(), it->second);
      return;
    }
    if (lifted_binding_vars_.count(op->var.get())) {
      // The binding is lifted to the subgraph and will be removed from the original function.
      return;
    }
    ExprMutator::VisitBinding_(op);
  }

  Expr VisitExpr_(const VarObj* op) final {
    Var var = ffi::GetRef<Var>(op);
    if (auto it = var_remap_.find(var); it != var_remap_.end()) {
      return it->second;
    }
    if (auto it = var_redef_.find(op); it != var_redef_.end()) {
      // This is the case that the var is used as an input to another lifted when
      // the original var definition is not visited yet.
      return EmitRedef(op, it->second);
    }
    return ffi::GetRef<Expr>(op);
  }

  Var EmitRedef(const VarObj* var, const Expr& redef) {
    auto new_var = builder_->Emit(redef, var->name);
    var_remap_.insert_or_assign(ffi::GetRef<Var>(var), new_var);
    return new_var;
  }

  std::unordered_map<const VarObj*, const LiftedFunctionRewritePlan*> subgraph_launches_;
  std::unordered_map<const VarObj*, Expr> var_redef_;
  std::unordered_set<const VarObj*> lifted_binding_vars_;
  int index_alloc_ = 0;
  int index_capture_ = 0;
  support::Arena arena_;
  ffi::Optional<GlobalVar> gv_global_alloc_ = std::nullopt;
  ffi::Optional<GlobalVar> current_func_ = std::nullopt;
};

IRModule RewriteCUDAGraph(IRModule mod) {
  CUDAGraphRewriter rewriter(mod);
  mod = rewriter.Rewrite();
  return mod;
}

namespace transform {

Pass RewriteCUDAGraph() {
  auto pass_func =  //
      [=](IRModule mod, PassContext pc) {
        bool use_cuda_graph = pc->GetConfig<bool>("relax.backend.use_cuda_graph").value_or(false);
        if (use_cuda_graph) {
          mod = relax::RewriteCUDAGraph(std::move(mod));
        }

        return mod;
      };
  return CreateModulePass(pass_func, 0, "RewriteCUDAGraph");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.RewriteCUDAGraph", RewriteCUDAGraph);
}

}  // namespace transform

}  // namespace relax
}  // namespace tvm
