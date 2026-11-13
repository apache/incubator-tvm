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

/*! \file src/relax/transform/lazy_transform_params.cc */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include <optional>
#include <unordered_map>

#include "utils.h"

namespace tvm {
namespace relax {
using namespace tvm::prim;

namespace {
std::optional<int64_t> GetNumInputParams(const FunctionObj* func) {
  if (auto opt_int_imm = func->GetAttr<IntImm>(tvm::relax::attr::kNumInput.name)) {
    int64_t num_input_params = static_cast<int64_t>(opt_int_imm.value()->value);
    TVM_FFI_CHECK_GE(num_input_params, 0, ValueError)
        << "Annotation for tvm::relax::attr::kNumInput.name (\"" << tvm::relax::attr::kNumInput.name
        << "\") must be non-negative, but was " << num_input_params;
    TVM_FFI_CHECK_LE(static_cast<size_t>(num_input_params), func->params.size(), ValueError)
        << "Annotation for tvm::relax::attr::kNumInput.name (\"" << tvm::relax::attr::kNumInput.name
        << "\") specifies " << num_input_params << " parameters to be provided at runtime, "
        << "but the function only accepts " << func->params.size() << " parameters in total";
    return num_input_params;
  } else {
    return std::nullopt;
  }
}

class LazyInputMutator : public ExprMutator {
 public:
  Expr VisitExpr_(const FunctionObj* func) override {
    if (plan_.has_value()) {
      return ExprMutator::VisitExpr_(func);
    }

    int64_t num_input_params = GetNumInputParams(func).value_or(0);

    std::unordered_map<Var, size_t> param_lookup;
    for (size_t i = num_input_params; i < func->params.size(); i++) {
      param_lookup.insert({func->params[i], i - num_input_params});
    }

    Var fget_param("fget_param", FuncType({PrimType::Int(64), AnyType()}, AnyType()));

    ffi::Array<Var> new_params(func->params.begin(), func->params.begin() + num_input_params);
    new_params.push_back(fget_param);

    auto array_externally_visible_vars = DefinableTIRVarsInType(TupleType(new_params.Map(GetType)));
    std::unordered_set<tvm::Var> externally_visible_vars(array_externally_visible_vars.begin(),
                                                         array_externally_visible_vars.end());
    Type new_ret_ty = EraseToWellDefined(func->ret_ty, [&](const Var& var) -> ffi::Optional<Expr> {
      if (auto prim_var = var.as<PrimVar>();
          prim_var && externally_visible_vars.count(prim_var.value())) {
        return prim_var.value().as_or_throw<PrimExpr>();
      }
      return std::nullopt;
    });

    auto node = ffi::GetRef<Function>(func);
    node.CopyOnWrite()->params = new_params;
    node.CopyOnWrite()->ret_ty = new_ret_ty;
    node = WithAttr(node, tvm::relax::attr::kNumInput.name, num_input_params + 1);

    plan_ = FunctionPlan{std::move(param_lookup), fget_param};
    auto output = ExprMutator::VisitExpr_(node.get()).as_or_throw<Function>();
    plan_.reset();
    return output;
  }

  Expr VisitExpr_(const VarObj* op) override {
    if (plan_) {
      Var var = ffi::GetRef<Var>(op);
      if (auto it = plan_->param_lookup.find(var); it != plan_->param_lookup.end()) {
        auto untyped = builder_->Emit(Call(Type::Missing(), plan_->fget_param,
                                           {
                                               PrimExpr(IntImm::Int64(it->second)),
                                               StringImm(var->name),
                                           }),
                                      var->name + "_untyped");
        return builder_->EmitMatchCast(untyped, GetType(var), var->name);
      }
    }

    return ExprMutator::VisitExpr_(op);
  }

 private:
  struct FunctionPlan {
    std::unordered_map<Var, size_t> param_lookup;
    Expr fget_param;
  };
  std::optional<FunctionPlan> plan_;
};

class LazyOutputMutator : public ExprMutator {
 public:
  Expr VisitExpr_(const FunctionObj* func) override {
    if (plan_.has_value()) {
      return ExprMutator::VisitExpr_(func);
    }

    std::unordered_map<Var, std::vector<size_t>> output_lookup;
    std::vector<std::tuple<size_t, Expr>> inline_outputs;
    auto define_lookup = [&](size_t output_index, Expr output_value) {
      if (auto var = output_value.as<Var>()) {
        output_lookup[var.value()].push_back(output_index);
      } else {
        inline_outputs.push_back({output_index, output_value});
      }
    };

    auto func_body = func->body.as_or_throw<SeqExpr>();
    if (auto tuple_output = func_body->body.as<TupleObj>()) {
      for (size_t i = 0; i < tuple_output->fields.size(); i++) {
        define_lookup(i, tuple_output->fields[i]);
      }
    } else {
      define_lookup(0, func_body->body);
    }

    Var fset_output("fset_output", FuncType({PrimType::Int(64), AnyType()},
                                            TupleType(ffi::Array<Type>{}), /* purity = */ false));
    plan_ = FunctionPlan{std::move(output_lookup), fset_output};

    std::optional<int64_t> num_input_params = GetNumInputParams(func);

    auto new_params = func->params;
    new_params.insert(new_params.begin() + num_input_params.value_or(func->params.size()),
                      fset_output);

    BindingBlock start_of_func = [&]() {
      ffi::Array<Binding> propagated_params;
      for (auto param : func->params) {
        GenerateSetOutputCalls(param, [&](const auto& fset_output_call) {
          Var void_output("_void", TupleType(ffi::Array<Type>{}));
          propagated_params.push_back(VarBinding(void_output, fset_output_call));
        });
      }
      return BindingBlock(propagated_params);
    }();
    BindingBlock end_of_func = [&]() {
      ffi::Array<Binding> propagated_params;
      for (const auto& [output_index, expr] : inline_outputs) {
        Call fset_output_call(Type::Missing(), fset_output,
                              {PrimExpr(IntImm::Int64(output_index)), expr});
        Var void_output("_void", TupleType(ffi::Array<Type>{}));
        propagated_params.push_back(VarBinding(void_output, fset_output_call));
      }
      return BindingBlock(propagated_params);
    }();

    ffi::Array<BindingBlock> new_blocks = func_body->blocks;
    new_blocks.insert(new_blocks.begin(), start_of_func);
    new_blocks.push_back(end_of_func);
    Expr new_body = SeqExpr(new_blocks, Tuple(ffi::Array<Expr>{}));

    auto node = ffi::GetRef<Function>(func);
    {
      auto write_ptr = node.CopyOnWrite();
      write_ptr->params = new_params;
      write_ptr->body = new_body;
      write_ptr->is_pure = false;
    }
    if (num_input_params.has_value()) {
      node = WithAttr(node, tvm::relax::attr::kNumInput.name, num_input_params.value() + 1);
    }

    auto output = ExprMutator::VisitExpr_(node.get()).as_or_throw<Function>();
    plan_.reset();
    return output;
  }

  void VisitBinding(const Binding& binding) override {
    ExprMutator::VisitBinding(binding);
    GenerateSetOutputCalls(binding->var, [this](const auto& fset_output_call) {
      builder_->Emit(fset_output_call, "_void");
    });
  }

 private:
  template <typename Callback>
  void GenerateSetOutputCalls(const Var& var, Callback callback) {
    if (plan_.has_value()) {
      if (auto it = plan_->output_lookup.find(var); it != plan_->output_lookup.end()) {
        for (auto output_index : it->second) {
          callback(Call(Type::Missing(), plan_->fset_output,
                        {PrimExpr(IntImm::Int64(output_index)), var}));
        }
      }
    }
  }

  struct FunctionPlan {
    std::unordered_map<Var, std::vector<size_t>> output_lookup;
    Expr fset_output;
  };
  std::optional<FunctionPlan> plan_;
};
}  // namespace

Function WithLazyInputs(Function func) {
  LazyInputMutator mutator;

  func = mutator.VisitExpr(func).as_or_throw<Function>();
  func = EliminateCommonSubexpr(func).as_or_throw<Function>();
  func = RemoveAllUnused(func).as_or_throw<Function>();
  return func;
}

Function WithLazyOutputs(Function func) {
  LazyOutputMutator mutator;

  func = mutator.VisitExpr(func).as_or_throw<Function>();
  return func;
}

namespace transform {

Pass LazyGetInput() {
  auto pass_func = [](Function func, IRModule, PassContext) -> Function {
    if (!func->GetAttr(tvm::attr::kGlobalSymbol).has_value()) {
      return func;
    }
    return WithLazyInputs(func);
  };
  return CreateFunctionPass(/*pass_function=*/pass_func,
                            /*opt_level=*/0,
                            /*pass_name=*/"LazyGetInput");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.LazyGetInput", LazyGetInput);
}

Pass LazySetOutput() {
  auto pass_func = [](Function func, IRModule, PassContext) -> Function {
    if (!func->GetAttr(tvm::attr::kGlobalSymbol).has_value()) {
      return func;
    }
    return WithLazyOutputs(func);
  };
  return CreateFunctionPass(/*pass_function=*/pass_func,
                            /*opt_level=*/0,
                            /*pass_name=*/"LazySetOutput");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.LazySetOutput", LazySetOutput);
}

}  // namespace transform
}  // namespace relax
}  // namespace tvm
