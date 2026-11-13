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
 * \file src/relax/transform/utils.h
 * \brief Additional utility classes and functions for working with the Relax IR.
 */
#ifndef TVM_RELAX_TRANSFORM_UTILS_H_
#define TVM_RELAX_TRANSFORM_UTILS_H_

#include <builtin_fp16.h>
#include <tvm/base.h>
#include <tvm/ffi/cast.h>
#include <tvm/ir/expr_functor.h>
#include <tvm/ir/function.h>
#include <tvm/ir/module.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../analysis/graph_partitioner.h"
#include "../op/nn/convolution.h"
#include "../op/nn/nn.h"
#include "../op/nn/pooling.h"
#include "../op/tensor/binary.h"
#include "../op/tensor/create.h"
#include "../op/tensor/datatype.h"
#include "../op/tensor/index.h"
#include "../op/tensor/linear_algebra.h"
#include "../op/tensor/manipulate.h"
#include "../op/tensor/search.h"
#include "../op/tensor/set.h"
#include "../op/tensor/statistical.h"
#include "../op/tensor/ternary.h"
#include "../op/tensor/unary.h"

namespace tvm {
namespace relax {

/*!
 * \brief A simple wrapper around ExprFunctor for a single argument case.
 *  The result of visit is memoized.
 */
template <typename OutputType>
class MemoizedExprTranslator : public ExprFunctor<OutputType(const Expr&)> {
  using BaseFunctor = ExprFunctor<OutputType(const Expr&)>;

 public:
  /*! \brief virtual destructor */
  virtual ~MemoizedExprTranslator() {}

  /*!
   * \brief The memoized call.
   * \param n The expression node.
   * \return The result of the call
   */
  virtual OutputType VisitExpr(const Expr& n) {
    TVM_FFI_ICHECK(n.defined());
    auto it = memo_.find(n);
    if (it != memo_.end()) {
      return it->second;
    }
    auto res = BaseFunctor::VisitExpr(n);
    memo_.insert_or_assign(n, res);
    return res;
  }

  virtual OutputType VisitExpr_(const VarObj* vn) {
    TVM_FFI_ICHECK(memo_.count(ffi::GetRef<Expr>(vn)));
    return memo_.at(ffi::GetRef<Expr>(vn));
  }

  virtual OutputType VisitExpr_(const DataflowVarObj* vn) {
    return VisitExpr_(static_cast<const VarObj*>(vn));
  }

  virtual OutputType VisitBinding_(const VarBindingObj* binding) {
    TVM_FFI_ICHECK_EQ(memo_.count(binding->var), 0);
    auto v = VisitExpr(binding->value);
    memo_.insert_or_assign(binding->var, v);
    return v;
  }

 protected:
  /*! \brief Internal map used for memoization. */
  std::unordered_map<Expr, OutputType, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> memo_;
};

/*!
 * \brief Dead code elimination
 * Currently it removes:
 *   1. Unused local VarBindings in a DataflowBlock.
 *      The used var set is set to empty at the beginning of each DataflowBlock.
 *      We reverse scan the DataflowBlock, if a VarBinding
 *        - bindings to a dataflowvar, or
 *        - is used in the used var set
 *      We keep it and add its var to the used var set. Otherwise, we remove it.
 *   2. Unused Relax functions in the module.
 *      We detect the call chain from the entry function, and remove all unused functions.
 * \param mod The target module
 * \param entry_functions list of entry functions
 * \return The updated module.
 */
TVM_DLL IRModule DeadCodeElimination(const IRModule& mod, ffi::Array<ffi::String> entry_funcs);

/*!
 * \brief Get the external symbol of the Relax function name.
 *
 * \param func The provided function.
 * \return An external symbol.
 */
inline std::string GetExtSymbol(const Function& func) {
  const auto name_node = func->GetAttr(tvm::attr::kGlobalSymbol);
  TVM_FFI_ICHECK(name_node.has_value()) << "Fail to retrieve external symbol.";
  return std::string(name_node.value());
}

/*!
 * \brief Fuse ops or functions according to the given partition, and grouped them into a new
 * function.
 *
 * \param mod The input module.
 * \param partition A mapping from a subexpression to the containing group.
 * \param lift_constants Whether or not to lift bound constants to parameters of the
 * grouped function.
 * \param entry_function_names The names of the entry functions.
 * \return A new module containing grouped functions.
 */
IRModule MakeGroupedFunctions(
    IRModule mod, const std::unordered_map<const ffi::Object*, GraphPartitioner::Group*>& partition,
    bool lift_constants = true, const ffi::Array<ffi::String>& entry_function_names = {});

/*!
 * \brief Check if the given Type is a scalar tensor. The ty should be an instance of
 * TensorType; its shape must be ShapeExpr.
 * \param ty The Type to be checked.
 * \return true if the given Type is a scalar tensor.
 */
bool IsScalarTensor(const Type& ty);

/*!
 * \brief Check if the given expr is a scalar tensor. Now the shape of the tensor expr must be
 * ShapeExpr.
 * \param expr The expr to be checked.
 * \return true if the given expr is a scalar tensor.
 */
bool IsScalarTensor(const Expr& expr);

/*!
 * \brief Check if the given Type is a nested tensor Type satisfying the given
 * condition f_condition.
 * \param ty The Type to be checked.
 * \param f_condition The condition function for each leaf Type with signature
 * `bool f_condition(TensorType)`.
 * \tparam FType The condition function type.
 * \return true if the given Type is a nested tensor satisfying the given f_condition.
 */
template <typename FType>
bool IsNestedTensorConditioned(const Type& ty, FType f_condition) {
  if (const auto* tensor_ty = ty.as<TensorTypeObj>()) {
    return f_condition(ffi::GetRef<TensorType>(tensor_ty));
  } else if (const auto* tuple_ty = ty.as<TupleTypeObj>()) {
    return !std::any_of(tuple_ty->fields.begin(), tuple_ty->fields.end(), [&](const Type& field) {
      return !IsNestedTensorConditioned(field, f_condition);
    });
  }
  return false;
}

/*!
 * \brief Check if the given Type is a nested tensor.
 * \param ty The Type to be checked.
 * \return true if the given Type is a nested tensor.
 */
bool IsNestedTensor(const Type& ty);

/*!
 * \brief Check if the given expr is a nested tensor.
 * \param expr The expr to be checked.
 * \return true if the given expr is a nested tensor.
 */
bool IsNestedTensor(const Expr& expr);

// TODO(@bohan): implements some postorder function accepts a visitor closure
class VarReplacer : public ExprMutator {
 public:
  using VarMap = std::unordered_map<Var, Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

  explicit VarReplacer(const VarMap& var_remap) : var_remap_(var_remap) {}

  static Expr Replace(const Expr& expr, const VarMap& var_remap) {
    VarReplacer replacer(var_remap);
    return replacer(expr);
  }

 private:
  Expr VisitExpr_(const VarObj* op) final {
    Var var = ffi::GetRef<Var>(op);
    auto it = var_remap_.find(var);
    return it == var_remap_.end() ? var : it->second;
  }

  const VarMap& var_remap_;
};

/*!
 * \brief Renew the definition of dependent primitive vars in Relax.
 * \details This mutator is used to prevent the same primitive Var object from being defined in
 *          different functions, which is malformed.
 */
class SymbolicVarRenewMutator : public ExprMutator {
 public:
  static Function Renew(const Function& function) {
    SymbolicVarRenewMutator mutator;
    return mutator.VisitExpr(function).as_or_throw<Function>();
  }
  SymbolicVarRenewMutator() : SymbolicVarRenewMutator(false) {}

 protected:
  explicit SymbolicVarRenewMutator(bool renew_all_var_definitions)
      : renew_all_var_definitions_(renew_all_var_definitions) {}

  using relax::ExprMutator::VisitExpr;
  using relax::ExprMutator::VisitExpr_;

  static Var CopyVar(const VarObj* op, Type ty) {
    ffi::Optional<Type> ty_annotation =
        ty.as<MissingType>().has_value() ? std::nullopt : ffi::Optional<Type>(ty);
    if (op->IsInstance<DataflowVarObj>()) {
      return DataflowVar(op->name, std::move(ty_annotation), op->loc);
    }
    return Var(op->name, std::move(ty), op->loc);
  }

  Type RenewType(const VarObj* op) {
    return op->ty.as<MissingType>().has_value() ? op->ty : this->VisitExprDepTypeField(op->ty);
  }

  Var RenewVarDefinition(const VarObj* op) {
    Var old_var = ffi::GetRef<Var>(op);
    if (auto it = var_remap_.find(old_var); it != var_remap_.end()) {
      return it->second;
    }

    Type new_ty = RenewType(op);
    bool is_dataflow = op->IsInstance<DataflowVarObj>();
    bool renew = renew_all_var_definitions_ || (!is_dataflow && op->ty.as<PrimTypeObj>()) ||
                 !new_ty.same_as(op->ty);
    if (!renew) {
      return old_var;
    }

    Var renewed = CopyVar(op, std::move(new_ty));
    var_remap_.insert_or_assign(old_var, renewed);
    return renewed;
  }

  Expr VisitExpr_(const VarObj* op) final {
    Var old_var = ffi::GetRef<Var>(op);
    if (auto it = var_remap_.find(old_var); it != var_remap_.end()) {
      return it->second;
    }
    if (!op->ty.as<PrimTypeObj>()) {
      return ExprMutator::VisitExpr_(op);
    }

    Var renewed = CopyVar(op, op->ty);
    var_remap_.insert_or_assign(old_var, renewed);
    return renewed;
  }

  Expr VisitExpr_(const DataflowVarObj* op) final {
    Var old_var = ffi::GetRef<Var>(op);
    if (auto it = var_remap_.find(old_var); it != var_remap_.end()) {
      return it->second;
    }
    if (!renew_all_var_definitions_) {
      return old_var;
    }

    Var renewed = CopyVar(op, op->ty);
    var_remap_.insert_or_assign(old_var, renewed);
    return renewed;
  }

  Var VisitVarDef_(const VarObj* op) override { return RenewVarDefinition(op); }

  Var VisitVarDef_(const DataflowVarObj* op) override { return RenewVarDefinition(op); }

  Expr VisitExpr_(const FunctionObj* op) final {
    tvm::ffi::Array<Var> params;
    bool all_params_unchanged = true;
    for (Var param : op->params) {
      Var new_param = this->VisitVarDef(param);
      params.push_back(new_param);
      if (!param.same_as(new_param)) {
        var_remap_.insert_or_assign(param, new_param);
        all_params_unchanged = false;
      }
    }

    Expr body = this->VisitWithNewScope(op->body, params);

    if (all_params_unchanged && body.same_as(op->body)) {
      return ffi::GetRef<Expr>(op);
    } else {
      auto new_ret_ty = this->VisitExprDepTypeField(op->ret_ty);
      return Function(params, body, new_ret_ty, op->is_pure, op->attrs);
    }
  }

  bool renew_all_var_definitions_{false};
};

/*!
 * \brief Copy a function while renewing every Var object, including dependent primitive uses.
 * \details All variables that are bound inside the original function would be copied to satisfy
 * the restriction in the well-formed check: Variables in Relax must be bound exactly once.
 */
class FunctionCopier : public SymbolicVarRenewMutator {
 public:
  FunctionCopier() : SymbolicVarRenewMutator(true) {}
  Function Copy(Function func) { return VisitExpr(func).as_or_throw<Function>(); }
  ffi::Map<Var, Var> GetVarMap() {
    ffi::Map<Var, Var> result;
    for (const auto& [old_var, new_var] : var_remap_) {
      result.Set(old_var, new_var);
    }
    return result;
  }
};

/*!
 * \brief Create a GenericConst with a scalar
 *
 * \param dtype The data type.
 * \param value The value of the scalar.
 * \return A GenericConst.
 */
template <typename T>
inline GenericConst MakeConstantScalar(T value, DLDataType dtype) {
  ffi::Tensor arr = runtime::Empty({}, dtype, {kDLCPU, 0});
  if (dtype == DLDataType{kDLFloat, 32, 1}) {
    *static_cast<float*>(arr.data_ptr()) = static_cast<float>(value);
  } else if (dtype == DLDataType{kDLFloat, 64, 1}) {
    *static_cast<double*>(arr.data_ptr()) = static_cast<double>(value);
  } else if (dtype == DLDataType{kDLInt, 32, 1}) {
    *static_cast<int32_t*>(arr.data_ptr()) = static_cast<int32_t>(value);
  } else if (dtype == DLDataType{kDLInt, 64, 1}) {
    *static_cast<int64_t*>(arr.data_ptr()) = static_cast<int64_t>(value);
  } else if (dtype == DLDataType{kDLBool, 8, 1}) {
    *static_cast<bool*>(arr.data_ptr()) = static_cast<bool>(value);
  } else if (dtype == DLDataType{kDLUInt, 8, 1}) {
    *static_cast<uint8_t*>(arr.data_ptr()) = static_cast<uint8_t>(value);
  } else if (dtype == DLDataType{kDLUInt, 16, 1}) {
    *static_cast<uint16_t*>(arr.data_ptr()) = static_cast<uint16_t>(value);
  } else if (dtype == DLDataType{kDLUInt, 32, 1}) {
    *static_cast<uint32_t*>(arr.data_ptr()) = static_cast<uint32_t>(value);
  } else if (dtype == DLDataType{kDLUInt, 64, 1}) {
    *static_cast<uint64_t*>(arr.data_ptr()) = static_cast<uint64_t>(value);
  } else if (dtype == DLDataType{kDLInt, 8, 1}) {
    *static_cast<int8_t*>(arr.data_ptr()) = static_cast<int8_t>(value);
  } else if (dtype == DLDataType{kDLInt, 16, 1}) {
    *static_cast<int16_t*>(arr.data_ptr()) = static_cast<int16_t>(value);
  } else if (dtype == DLDataType{kDLInt, 32, 1}) {
    *static_cast<int32_t*>(arr.data_ptr()) = static_cast<int32_t>(value);
  } else if (dtype == DLDataType{kDLInt, 64, 1}) {
    *static_cast<int64_t*>(arr.data_ptr()) = static_cast<int64_t>(value);
  } else if (dtype == DLDataType{kDLFloat, 16, 1}) {
    // convert to float16 storage is uint16_t
    *static_cast<uint16_t*>(arr.data_ptr()) =
        __truncXfYf2__<float, uint32_t, 23, uint16_t, uint16_t, 10>(static_cast<float>(value));
  } else if (dtype == DLDataType{kDLBfloat, 16, 1}) {
    // convert to bfloat16 storage is uint16_t
    *static_cast<uint16_t*>(arr.data_ptr()) =
        __truncXfYf2__<float, uint32_t, 23, uint16_t, uint16_t, 7>(static_cast<float>(value));
  } else {
    TVM_FFI_THROW(InternalError) << "Unsupported dtype " << dtype;
  }
  return MakeTensorConst(arr);
}

inline ffi::Array<int64_t> GetOrderedPositiveAxes(const ffi::Array<int64_t>& axes, int ndim) {
  std::vector<int64_t> ret;
  ret.reserve(axes.size());
  for (int64_t axis_val : axes) {
    if (axis_val < 0) {
      axis_val += ndim;
    }
    TVM_FFI_ICHECK(axis_val >= 0 && axis_val < ndim)
        << "axis " << axis_val << " is out of bounds for array of "
        << "dimension " << ndim;
    ret.push_back(axis_val);
  }
  std::sort(ret.begin(), ret.end());
  ffi::Array<int64_t> result;
  result.reserve(ret.size());
  for (int64_t x : ret) result.push_back(x);
  return result;
}

inline ffi::String GetCodegenName(const std::string& composite_name) {
  auto delim_pos = composite_name.find(".");
  TVM_FFI_ICHECK(delim_pos != std::string::npos)
      << "The pattern name for a composite function should "
         "start with a compiler name followed by period.";
  return composite_name.substr(0, delim_pos);
}

inline int GetDeviceIndexByScope(const IRModule& mod, runtime::MemoryScope scope) {
  if (mod->global_infos.find("vdevice") == mod->global_infos.end()) {
    return 0;
  }
  ffi::Array<GlobalInfo> vdevices = mod->global_infos["vdevice"];
  for (int i = 0; i < static_cast<int>(vdevices.size()); ++i) {
    if (scope == vdevices[i].as<VDevice>().value()->memory_scope) {
      return i;
    }
  }
  return 0;
}

inline int GetDeviceIndex(const IRModule& mod, const VDevice& vdevice) {
  ffi::Array<GlobalInfo> vdevices = mod->global_infos["vdevice"];
  for (int i = 0; i < static_cast<int>(vdevices.size()); ++i) {
    if (vdevices[i].same_as(vdevice)) {
      return i;
    }
  }
  TVM_FFI_THROW(InternalError) << "The vdevice is not in the ir_module.";
  return -1;
}

inline ffi::Optional<VDevice> GetGlobalVDevice(const IRModule& mod, const int index) {
  ffi::Optional<VDevice> ret;
  if (mod->global_infos.find("vdevice") != mod->global_infos.end()) {
    ffi::Array<GlobalInfo> vdevices = mod->global_infos["vdevice"];
    if (index < static_cast<int>(vdevices.size())) {
      ret = vdevices[index].as<VDevice>();
    }
  }
  return ret;
}

/* \brief Eliminate common subexpressions
 *
 * Utility for simplifying relax expressions by removing common
 * subexpressions.
 *
 * \param expr The expression to be updated
 *
 * \param call_only If true, only eliminate relax::Call nodes.  If
 * false, eliminate any common subexpressions.
 *
 * \ret The updated expression
 */
Expr EliminateCommonSubexpr(const Expr& expr, bool call_only = false);

/* \brief Remove use of trivial bindings
 *
 * Utility for simplifying relax expressions by folding var bindings
 * and match shape nodes.  May include other forms of simplification
 * in the future.  Ideally should be used before constant folding and
 * eliminating unused bindings.
 *
 * \param expr The expression to be canonicalized
 *
 * \ret The canonicalized expression
 */
Expr CanonicalizeBindings(Expr expr);

/* \brief Remove use of trivial bindings
 *
 * Utility for converting from individual model parameters to a single
 * parameter with a tuple of parameters.  If the `kNumInput` attribute
 * is absent, no model parameters are present, so no updates are made.
 *
 * \param func The function to be updated.
 *
 * \param param_tuple_name The name of the tuple parameter.  If
 * unspecified, defaults to "model_params"
 *
 * \ret The updated function.
 */
Function BundleModelParams(const Function& func,
                           ffi::Optional<ffi::String> param_tuple_name = std::nullopt);

/*! \brief Compose two functions
 *
 * Given two functions `func_a` and `func_b`, produce `func_c` such
 * that `func_c(x)` is equivalent to `func_b(func_a(x))`.
 *
 * If the output if `func_a` is not usable as the input of `func_b`,
 * an error will be raised.
 *
 * \param func_a The first function to be composed.
 * \param func_b The second function to be composed.
 * \return The composed function
 */
TVM_DLL Function ComposeFunctions(Function func_a, Function func_b);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_TRANSFORM_UTILS_H_
