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

/*! \file src/relax/transform/decompose_ops.cc */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/op/nn.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/tirx/function.h>

#include <unordered_set>

#include "utils.h"

namespace tvm {
namespace relax {
using namespace tvm::prim;

TensorType MatchTensorType(Expr data) {
  auto _ty = MatchType<TensorType>(data);
  TVM_FFI_ICHECK(_ty.has_value()) << "Expect data to be a tensor, but get " << GetType(data);
  return _ty.value();
}

Expr ExpandToMatchInput(Expr data, int ndim, ffi::Array<int64_t> axes) {
  axes = GetOrderedPositiveAxes(axes, ndim);
  ffi::Array<int64_t> expand_axes;
  for (int i = 0, j = 0; i < ndim; ++i) {
    if (j < static_cast<int>(axes.size()) && i == axes[j]) {
      ++j;
    } else {
      expand_axes.push_back(i);
    }
  }
  return expand_dims(data, expand_axes);
}

Tuple DecomposeBatchNorm(const Call& call) {
  auto attrs = call->attrs.as<BatchNormAttrs>();
  TVM_FFI_ICHECK_NOTNULL(attrs);

  Expr data = call->args[0];
  TensorType ty = MatchTensorType(data);
  Expr gamma = call->args[1];
  Expr beta = call->args[2];

  Expr moving_mean = ExpandToMatchInput(call->args[3], ty->ndim, {attrs->axis});
  Expr moving_var = ExpandToMatchInput(call->args[4], ty->ndim, {attrs->axis});

  // output = (x - mean) / sqrt(var + epsilon) * gamma + beta
  Expr epsilon = MakeConstantScalar(attrs->epsilon, ty->dtype.value()->dtype);
  Expr sqrt_var = sqrt(add(moving_var, epsilon));
  Expr out = divide(subtract(data, moving_mean), sqrt_var);

  if (attrs->scale) {
    out = multiply(out, ExpandToMatchInput(gamma, ty->ndim, {attrs->axis}));
  }
  if (attrs->center) {
    out = add(out, ExpandToMatchInput(beta, ty->ndim, {attrs->axis}));
  }

  return Tuple({out, call->args[3], call->args[4]});
}

Expr MutateBatchNormForTraining(Call call) {
  auto attrs = call->attrs.as<BatchNormAttrs>();
  TVM_FFI_ICHECK_NOTNULL(attrs);

  TVM_FFI_ICHECK_EQ(call->args.size(), 5);
  Expr data = call->args[0];
  Expr gamma = call->args[1];
  Expr beta = call->args[2];
  Expr moving_mean = call->args[3];
  Expr moving_var = call->args[4];

  TensorType ty = MatchTensorType(data);

  ffi::Array<int64_t> reduce_axes;
  for (int i = 0; i < ty->ndim; ++i) {
    if (i != attrs->axis) {
      reduce_axes.push_back(i);
    }
  }

  Expr data_mean = mean(data, reduce_axes, false);
  Expr data_var = variance(data, reduce_axes, false);

  Expr momentum = MakeConstantScalar(attrs->momentum, ty->dtype.value()->dtype);
  Expr one_minus_mom = MakeConstantScalar(1 - attrs->momentum, ty->dtype.value()->dtype);

  Expr new_moving_mean = add(multiply(one_minus_mom, moving_mean), multiply(momentum, data_mean));
  Expr new_moving_var = add(multiply(one_minus_mom, moving_var), multiply(momentum, data_var));

  call.CopyOnWrite()->args = {data, gamma, beta, data_mean, data_var};
  // return call;

  return relax::Tuple({TupleGetItem(call, 0), new_moving_mean, new_moving_var});
}

Expr DecomposeLayerNorm(const Call& call) {
  auto attrs = call->attrs.as<LayerNormAttrs>();
  TVM_FFI_ICHECK_NOTNULL(attrs);

  Expr data = call->args[0];
  TensorType ty = MatchTensorType(data);
  Expr gamma = call->args[1];
  Expr beta = call->args[2];

  Expr data_mean = mean(data, attrs->axes, true);
  Expr data_var = variance(data, attrs->axes, true);

  // output = (x - mean) / sqrt(var + epsilon) * gamma + beta
  Expr epsilon = MakeConstantScalar(attrs->epsilon, ty->dtype.value()->dtype);
  Expr sqrt_var = sqrt(add(data_var, epsilon));
  Expr out = divide(subtract(data, data_mean), sqrt_var);

  if (attrs->scale) {
    out = multiply(out, gamma);
  }
  if (attrs->center) {
    out = add(out, beta);
  }

  return out;
}

Expr TensorToShape(const Call& call_node, const BlockBuilder& builder) {
  TVM_FFI_ICHECK(!call_node->ty.as<MissingType>().has_value());
  Expr expr = call_node->args[0];
  const ShapeTypeObj* ty = GetTypeAs<ShapeTypeObj>(call_node);
  TVM_FFI_ICHECK(ty);
  // call builtin function that converts tensor to shape tuple
  // TODO(@sunggg): Register operator for "vm.builtin.tensor_to_shape"
  static const Op call_pure_packed_op = Op::Get("relax.call_pure_packed");
  Var call = builder->Emit(Call(Type::Missing(), call_pure_packed_op,
                                {ExternFunc("vm.builtin.tensor_to_shape"), expr}, {},
                                {ffi::GetRef<ShapeType>(ty)}));

  // Operators like reshape take the output of `TensorToShape` as their output shape.
  // Because TOPI expects to have such output shape in symbolic shape at least (i.e.,
  // ffi::Array<PrimExpr>), we define symbolic variables and returns them as a ShapeExpr.
  ffi::Array<PrimExpr> shape_var;
  for (int i = 0; i < ty->ndim; i++) {
    shape_var.push_back(tvm::Var("x", PrimType::Int(64)).as_or_throw<PrimExpr>());
  }
  // bind symbolic variables to the shape tuple
  tvm::Var var("y", ShapeType(shape_var));
  builder->EmitNormalized(MatchCast(var, call, ShapeType(shape_var)));
  return ShapeExpr(shape_var);
}

/*! \brief Update operators that have a training-specific form
 *
 * Some operators, such as relax.op.batch_norm, need additional
 * processing when being run for training.  This mutator applies any mutations required
 */
class TrainingOperatorMutator : public ExprMutator {
 private:
  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const CallObj* call_node) final {
    Call call = VisitExprPostOrder_(call_node).as_or_throw<Call>();
    if (call->op.same_as(batch_norm_op_)) {
      return MutateBatchNormForTraining(call);
    } else if (call->op.same_as(layer_norm_op_)) {
      // Here we only decompose LayerNorm in training because it is more efficient as a single op.
      // In the future maybe we can also remove this decomposition during training.
      return DecomposeLayerNorm(call);
    } else {
      return call;
    }
  }

  /* composite opeartor list */
  const Op batch_norm_op_ = Op::Get("relax.nn.batch_norm");
  const Op layer_norm_op_ = Op::Get("relax.nn.layer_norm");
};

class OpDecomposer : public ExprMutator {
 private:
  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const CallObj* call_node) final {
    Call call = VisitExprPostOrder_(call_node).as_or_throw<Call>();
    if (call->op.same_as(batch_norm_op_)) {
      return DecomposeBatchNorm(call);
    } else if (call->op.same_as(tensor_to_shape_op_)) {
      return TensorToShape(call, builder_);
    }
    return call;
  }

  /* composite opeartor list */
  const Op batch_norm_op_ = Op::Get("relax.nn.batch_norm");
  const Op tensor_to_shape_op_ = Op::Get("relax.tensor_to_shape");
};

namespace transform {

namespace {

/*!
 * \brief Apply a pass to a single named function within an IRModule.
 *
 * Replaces all other functions with dummy ExternFunc stubs so that the
 * pass does not see them, then restores the original module.  Uses
 * exact name match (not a regex) because all in-tree callers supply a
 * literal function name.
 */
Pass ApplyDecomposeToFunction(Pass pass, ffi::String func_name) {
  auto pass_func = [pass, func_name](IRModule mod, PassContext) -> IRModule {
    std::unordered_set<ffi::String> keep_original_version;
    std::unordered_set<ffi::String> internal_functions;
    IRModule subset;

    for (auto [gvar, func] : mod->functions) {
      if (gvar->name_hint == func_name) {
        if (!func->GetAttr(tvm::attr::kGlobalSymbol).has_value()) {
          // Mark internal functions as externally-exposed so that
          // call-tracing transforms inside the pass do not remove them.
          internal_functions.insert(gvar->name_hint);
          func = WithAttr(std::move(func), tvm::attr::kGlobalSymbol.name, gvar->name_hint);
        }
      } else {
        // Replace non-target functions with stubs to keep references intact.
        keep_original_version.insert(gvar->name_hint);
        func = relax::ExternFunc("dummy_" + std::string(gvar->name_hint));
        func->ty = gvar->ty;
      }
      subset->Add(gvar, func);
    }

    IRModule new_subset = pass(subset);
    if (new_subset.same_as(subset)) {
      return mod;
    }

    auto write_ptr = mod.CopyOnWrite();
    for (auto [gvar, func] : new_subset->functions) {
      if (!keep_original_version.count(gvar->name_hint)) {
        if (auto it = write_ptr->global_var_map_.find(gvar->name_hint);
            it != write_ptr->global_var_map_.end()) {
          write_ptr->Remove((*it).second);
        }
        if (internal_functions.count(gvar->name_hint)) {
          func = WithoutAttr(std::move(func), tvm::attr::kGlobalSymbol.name);
        }
        write_ptr->Add(gvar, func);
      }
    }
    return mod;
  };

  std::string pass_name = "ApplyDecomposeTo" + std::string(func_name);
  return CreateModulePass(pass_func, 0, pass_name);
}

}  // namespace

Pass MutateOpsForTraining() {
  auto pass_func = [](Function func, IRModule, PassContext) -> Function {
    TrainingOperatorMutator mutator;
    return mutator(func).as_or_throw<Function>();
  };
  return CreateFunctionPass(/*pass_function=*/pass_func,
                            /*opt_level=*/0,
                            /*pass_name=*/"MutateOpsForTraining");
}

Pass DecomposeOps() {
  auto pass_func = [](Function func, IRModule, PassContext) -> Function {
    OpDecomposer mutator;
    return mutator(func).as_or_throw<Function>();
  };
  return CreateFunctionPass(/*pass_function=*/pass_func,
                            /*opt_level=*/0,
                            /*pass_name=*/"DecomposeOps");
}

Pass DecomposeOpsForInference(ffi::Optional<ffi::String> func_name) {
  if (func_name) {
    return ApplyDecomposeToFunction(DecomposeOps(), func_name.value());
  } else {
    return DecomposeOps();
  }
}

Pass DecomposeOpsForTraining(ffi::Optional<ffi::String> func_name) {
  auto module_pass = tvm::transform::Sequential({MutateOpsForTraining(), DecomposeOps()},
                                                "DecomposeOpsForTraining");
  if (func_name) {
    return ApplyDecomposeToFunction(module_pass, func_name.value());
  } else {
    return module_pass;
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.transform.DecomposeOpsForInference", DecomposeOpsForInference)
      .def("relax.transform.DecomposeOpsForTraining", DecomposeOpsForTraining);
}

}  // namespace transform
}  // namespace relax
}  // namespace tvm
