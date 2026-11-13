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

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/op.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op_attr.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/function.h>

namespace tvm {
namespace relax {
using namespace tvm::prim;

class ConstantFolder : public ExprMutator {
 public:
  static Function Fold(Function func, IRModule ctx_module) {
    ConstantFolder folder(std::move(ctx_module));
    func = RemoveAllUnused(folder(func)).as_or_throw<Function>();
    return func;
  }

 private:
  explicit ConstantFolder(IRModule ctx_module) : ExprMutator(ctx_module) {}

  /*!
   * \brief Pattern match the shape inside the given type to a
   * constant shape and get runtime shape tuple from it.
   * \param ty The given type whose shape inside is to be casted.
   * \return The runtime shape tuple, or nullopt if it is not a constant shape.
   * \note Only TensorType is supported. Returns std::nullopt
   * if the input type is not TensorType.
   */
  static ffi::Optional<ffi::Shape> MatchConstShape(const Type& ty) {
    const auto* tensor_ty = ty.as<TensorTypeObj>();
    if (tensor_ty == nullptr) {
      return std::nullopt;
    }

    const auto* shape = tensor_ty->shape.as<ShapeExprObj>();
    TVM_FFI_ICHECK(shape != nullptr) << "type given by call_tir should have ShapeExpr shape";

    std::vector<int64_t> shape_values;
    for (const auto v : shape->values) {
      auto* ptr = v.as<IntImmObj>();
      if (!ptr) return std::nullopt;
      auto value = ptr->value.as<int64_t>();
      if (!value.has_value()) return std::nullopt;
      shape_values.push_back(*value);
    }
    return ffi::Shape(shape_values.begin(), shape_values.end());
  }

  /*!
   * \brief Pattern match op to constant array arguments.
   * \return The constant array arguments, or nullopt if match fails.
   */
  static ffi::Optional<ffi::Array<ffi::Tensor>> MatchConstArrayArgs(const ffi::Array<Expr>& args) {
    ffi::Array<ffi::Tensor> res;
    for (auto arg : args) {
      auto* ptr = arg.as<::tvm::GenericConstObj>();
      if (!ptr || !ptr->value.as<ffi::TensorObj>()) return std::nullopt;
      res.push_back(ptr->value.cast<ffi::Tensor>());
    }
    return res;
  }

  /*!
   * \brief Pattern match op to a TIR function and look it up.
   * \return The TIR function, or nullopt if pattern match fails.
   */
  ffi::Optional<tirx::Function> MatchFunction(const Expr& op) {
    const GlobalVar& global_var = op.as_or_throw<GlobalVar>();
    // NOTE: as check works for nullptr(returns null)
    ffi::Optional<BaseFunc> base_func = builder_->GetContextIRModule()->functions.Get(global_var);
    if (auto* pfunc = base_func.as<tirx::FunctionObj>()) {
      return ffi::GetRef<tirx::Function>(pfunc);
    }
    return std::nullopt;
  }

  /*!
   * \brief Get a cached build version of func
   * \return The cached func, nullopt if func cannot be built.
   */
  ffi::Optional<ffi::Function> GetCachedBuild(tirx::Function func) {
    // TODO(tvm-team): consider another way of bulk extract and build tirx::Function once
    // would be helpful for future cases where tirx::Function recursively call into each other
    Target eval_cpu_target{"llvm"};

    auto it = func_build_cache_.find(func);
    if (it != func_build_cache_.end()) {
      return it->second;
    }
    ffi::Optional<ffi::Function> build_func = std::nullopt;

    try {
      // Not all the function can be directly built via llvm, for example, if a function is
      // already scheduled to only work on GPU, we will need to skip this in the const folder for
      // now
      // TODO(Hongyi): further check and narrow the scope of foldable function
      const auto pf = tvm::ffi::Function::GetGlobalRequired("tirx.compile");
      func = WithAttr(func, tvm::attr::kGlobalSymbol.name, ffi::String("tir_function"));
      ffi::Module rt_module = pf(func, eval_cpu_target).cast<ffi::Module>();
      build_func = rt_module->GetFunction("tir_function");
    } catch (const tvm::ffi::Error& err) {
      // build failure may happen in which case we skip
      DLOG(WARNING) << "Build failure for function " << func << ", Error message: " << err.what();
    }
    func_build_cache_[func] = build_func;
    return build_func;
  }

  /*!
   * \brief Checks if it is useful to fold \p expr.
   * \details Folding an expr is a trade-off - we are materializing a constant in the IRModule and
   * paying compile time cost to avoid the cost of executing this expr at runtime. For example,
   * folding iota ops could result in large constants being materialized, thus increasing the size
   * of the program.
   */
  static bool ExprContainsTensor(const Expr& expr) {
    if (GetType(expr).as<TensorTypeObj>()) {
      return true;
    }
    if (const auto* tuple = expr.as<TupleObj>()) {
      for (const auto& field : tuple->fields) {
        if (ExprContainsTensor(field)) {
          return true;
        }
      }
    }
    return false;
  }

  bool ShouldBeFolded(Expr expr) {
    // Skip folding for creation ops (no tensor inputs) that produce large outputs.
    // These ops (e.g., zeros, ones, full, arange) are cheap to compute at runtime,
    // and folding them would materialize large constants in the binary.
    static constexpr int64_t kMaxFoldElements = 1024;

    const auto* call = expr.as<CallObj>();
    if (!call) return true;

    const auto* tensor_ty = call->ty.as<TensorTypeObj>();
    if (!tensor_ty) return true;

    auto opt_shape = tensor_ty->GetShape();
    if (!opt_shape) return true;

    int64_t num_elements = 1;
    for (const auto& dim : opt_shape.value()) {
      const auto* int_dim = dim.as<IntImmObj>();
      if (!int_dim) return true;
      auto d = int_dim->value.as<int64_t>();
      if (int_dim->value <= 0) return true;
      if (!d.has_value() || num_elements > kMaxFoldElements / *d) {
        num_elements = kMaxFoldElements + 1;
        break;
      }
      num_elements *= *d;
    }

    if (num_elements <= kMaxFoldElements) return true;

    // Large output. Only skip if there are no tensor inputs,
    // i.e., this is a pure creation op.
    for (const auto& arg : call->args) {
      if (ExprContainsTensor(arg)) {
        return true;
      }
    }

    return false;
  }

  // Try constant evaluate a call_tir with a single tensor output.
  // Returns std::nullopt on failure.
  ffi::Optional<Expr> ConstEvaluateCallTIR(tirx::Function tir_func,
                                           ffi::Array<ffi::Tensor> arr_args, ffi::Shape shape,
                                           DLDataType ret_type) {
    // obtain function from the cache.
    ffi::Optional<ffi::Function> func = GetCachedBuild(tir_func);
    if (!func) return std::nullopt;

    // here the vector size has an additional + 1 because we need to put ret_tensor at the end
    std::vector<AnyView> packed_args(arr_args.size() + 1);

    DLDevice cpu_dev = {DLDeviceType::kDLCPU, 0};
    ffi::Tensor ret_tensor = runtime::Empty(shape, ret_type, cpu_dev);

    // avoid set rvalue ref which get de-allocated later, store args in a vector
    // where temp_args[i] are lvalue ref that is stable
    std::vector<ffi::Tensor> temp_args(arr_args.begin(), arr_args.end());

    size_t arg_offset = 0;
    for (; arg_offset < arr_args.size(); ++arg_offset) {
      packed_args[arg_offset] = temp_args[arg_offset];
    }
    // set return value
    packed_args[arg_offset++] = ret_tensor;

    ffi::Any ret;
    // invoke
    func.value().CallPacked(ffi::PackedArgs(packed_args.data(), packed_args.size()), &ret);
    return MakeTensorConst(ret_tensor);
  }

  // Try constant evaluate a call_tir with tuple outputs (multiple output tensors).
  // Returns std::nullopt on failure.
  ffi::Optional<Expr> ConstEvaluateCallTIRTuple(tirx::Function tir_func,
                                                ffi::Array<ffi::Tensor> arr_args,
                                                const TupleTypeObj* tuple_ty) {
    ffi::Optional<ffi::Function> func = GetCachedBuild(tir_func);
    if (!func) return std::nullopt;

    DLDevice cpu_dev = {DLDeviceType::kDLCPU, 0};
    size_t num_outputs = tuple_ty->fields.size();

    // Match shapes and dtypes for all output fields.
    std::vector<ffi::Tensor> ret_tensors;
    for (size_t i = 0; i < num_outputs; ++i) {
      ffi::Optional<ffi::Shape> shape = MatchConstShape(tuple_ty->fields[i]);
      if (!shape) return std::nullopt;
      auto tensor_ty = tuple_ty->fields[i].as_or_throw<TensorType>();
      if (tensor_ty->IsUnknownDtype()) return std::nullopt;
      ret_tensors.push_back(
          runtime::Empty(shape.value(), tensor_ty->dtype.value()->dtype, cpu_dev));
    }

    // Pack input args + all output tensors.
    std::vector<ffi::Tensor> temp_args(arr_args.begin(), arr_args.end());
    std::vector<AnyView> packed_args;
    packed_args.reserve(temp_args.size() + num_outputs);
    for (const auto& arg : temp_args) {
      packed_args.push_back(arg);
    }
    for (const auto& out_tensor : ret_tensors) {
      packed_args.push_back(out_tensor);
    }

    ffi::Any ret;
    func.value().CallPacked(ffi::PackedArgs(packed_args.data(), packed_args.size()), &ret);

    ffi::Array<Expr> fields;
    for (size_t i = 0; i < num_outputs; ++i) {
      fields.push_back(MakeTensorConst(ret_tensors[i]));
    }
    return Tuple(fields);
  }

  // Returns the folded expr if the call is successfully folded to constant, otherwise null.
  ffi::Optional<Expr> VisitCallTIR(Call call) {
    // call_tir needs to have at least two arguments
    TVM_FFI_ICHECK_GE(call->args.size(), 2);
    ffi::Optional<tirx::Function> func = MatchFunction(call->args[0]);
    TVM_FFI_ICHECK(call->args[1].as<TupleObj>()) << "call_tir.args[1] must be Tuple";
    ffi::Optional<ffi::Array<ffi::Tensor>> arr_args =
        MatchConstArrayArgs(call->args[1].as<TupleObj>()->fields);
    TVM_FFI_ICHECK_EQ(call->ty_args.size(), 1) << "call_tir should have exactly one ty arg";

    if (!func || !arr_args) return {};

    // Handle tuple output: ty_args[0] is a TupleType.
    if (const auto* tuple_ty = call->ty_args[0].as<TupleTypeObj>()) {
      return ConstEvaluateCallTIRTuple(func.value(), arr_args.value(), tuple_ty);
    }

    // Handle single tensor output.
    ffi::Optional<ffi::Shape> shape = MatchConstShape(call->ty_args[0]);
    if (shape) {
      TensorType ret_ty = call->ty.as_or_throw<TensorType>();
      return ConstEvaluateCallTIR(func.value(), arr_args.value(), shape.value(),
                                  ret_ty->dtype.value()->dtype);
    }
    return {};
  }

  using ExprMutator::VisitExpr_;

  // TODO(@sunggg):
  // Next PR will support fold with ffi::Function and MatchCast
  // Until then, DecomposeOps() should be applied after
  // this pass to fold `tensor_to_shape` op.
  Expr VisitExpr_(const CallObj* call) final {
    // post-order mutation
    Call post_call = VisitExprPostOrder_(call).as_or_throw<Call>();

    // Check if it is useful to fold this call
    if (!ShouldBeFolded(post_call)) return post_call;

    static const Op call_tir_op = Op::Get("relax.call_tir");
    static const auto& infer_type_map = Op::GetAttrMap(tvm::op_attr::kInferType);
    static const auto& infer_type_with_builder_map =
        Op::GetAttrMap(tvm::relax::op_attr::kInferTypeWithBuilder);
    static const auto& legalize_map = Op::GetAttrMap(tvm::relax::op_attr::kLegalize);
    auto* op_node = post_call->op.as<OpObj>();

    // Not an OpObj
    if (op_node == nullptr) {
      return post_call;
    }
    auto op = ffi::GetRef<Op>(op_node);

    if (op.same_as(call_tir_op)) {
      return VisitCallTIR(post_call).value_or(post_call);
    }

    // Special logic to fold ShapeExpr between operators
    // e.g.,
    //  <Before>
    //     lv: R.Shape([16, 16]) = R.shape([16, 16])
    //     gv: R.Tensor(lv2, dtype="float32") = R.reshape(data, lv)
    //  <After>
    //     gv: R.Tensor(lv2, dtype="float32") = R.reshape(data, R.shape([16, 16]))
    //
    ffi::Array<Expr> new_args;
    for (auto arg : post_call->args) {
      if (arg->IsInstance<VarObj>()) {
        ffi::Optional<Expr> val = LookupBinding(arg.as_or_throw<Var>());
        if (val.has_value() && val.value()->IsInstance<ShapeExprObj>()) {
          new_args.push_back(val.value());
          continue;
        }
      }
      new_args.push_back(arg);
    }
    Type ret_ty = Type::Missing();
    if (post_call->ty.as<PrimTypeObj>() && !infer_type_map.count(op) &&
        !infer_type_with_builder_map.count(op)) {
      ret_ty = post_call->ty.as_or_throw<Type>();
    }
    post_call =
        Call(ret_ty, post_call->op, new_args, post_call->attrs, post_call->ty_args, post_call->loc);

    // If we are in a dataflow block, we can fold ops.
    if (builder_->CurrentBlockIsDataFlow()) {
      // Check if we can them to call_tir
      if (legalize_map.count(op)) {
        // Get the legalized expression
        Call post_call_normalized = builder_->Normalize(post_call).as_or_throw<Call>();
        Expr legalized_expr = builder_->Normalize(legalize_map[op](builder_, post_call_normalized));
        // If the legalized expression is call_tir, try to fold it.
        const CallObj* call = legalized_expr.as<CallObj>();
        if (call && call->op.same_as(call_tir_op)) {
          return VisitCallTIR(ffi::GetRef<Call>(call)).value_or(post_call);
        }
      } else if (op->name == "relax.tensor_to_shape") {
        // Special handling for composite op "relax.tensor_to_shape"
        // If its input is constant, we can access its value and create ShapeExpr
        // TODO(@sunggg):
        //   currently, we do not have a info map about decomposition.
        //   Thus, this is a temporary solution until we have a consensus about
        //   how to deal with composite ops. One possibility is we register the
        //   decomposition map for each op in a similar way we do for legalization.
        TVM_FFI_ICHECK_EQ(post_call->args.size(), 1);
        Expr arg = post_call->args[0];
        if (arg->IsInstance<GenericConstObj>()) {
          GenericConst constant = arg.as_or_throw<GenericConst>();
          ffi::Tensor ndarray = constant->value.cast<ffi::Tensor>();
          TVM_FFI_ICHECK(ndarray.defined());
          TVM_FFI_ICHECK_EQ(ndarray.device().device_type, kDLCPU);
          TVM_FFI_ICHECK(ndarray.IsContiguous());
          TVM_FFI_ICHECK_EQ(ndarray.byte_offset(), 0);
          TVM_FFI_ICHECK_EQ(ndarray.ndim(), 1);
          const int64_t* data = static_cast<const int64_t*>(ndarray.data_ptr());
          int64_t num_elems = ndarray.shape()[0];
          ffi::Array<PrimExpr> shape_values;
          for (int64_t i = 0; i < num_elems; i++) {
            shape_values.push_back(IntImm::Int64(data[i]));
          }
          return ShapeExpr(shape_values);
        }
      } else if (op->name == "relax.shape_to_tensor") {
        // Special handling for "relax.shape_to_tensor" since it is implemented in ffi::Function.
        // TODO(sunggg): revisit this when we extend ConstantFolding to fold ffi::Function.
        Expr arg = post_call->args[0];
        ShapeExpr shape = arg.as_or_throw<ShapeExpr>();
        ffi::Array<PrimExpr> values = shape->values;
        ffi::Array<int64_t> arr;
        bool is_known = true;
        for (size_t i = 0; i < values.size(); i++) {
          PrimExpr val = values[i];
          if (!val.ty().MatchesElementType(DLDataTypeCode::kDLInt, 64)) {
            is_known = false;
            break;
          }
          arr.push_back(static_cast<int64_t>(val.as<IntImmObj>()->value));
        }
        if (is_known) {
          const auto func = tvm::ffi::Function::GetGlobalRequired("relax.run.shape_to_tensor");
          ffi::Tensor vals = func(arr).cast<ffi::Tensor>();
          TVM_FFI_ICHECK(vals.defined());
          return MakeTensorConst(vals);
        }
      }
    }

    return post_call;
  }

  Expr VisitExpr_(const VarObj* op) final {
    ffi::Optional<Expr> opt = LookupBinding(ffi::GetRef<Var>(op));
    // `as` check checks if opt is not null and is instance of constant
    if (opt.as<::tvm::GenericConstObj>()) {
      return opt.value();
    }
    return ExprMutator::VisitExpr_(op);
  }

  // cache for function build, via structural equality
  std::unordered_map<tirx::Function, ffi::Optional<ffi::Function>, ffi::StructuralHash,
                     ffi::StructuralEqual>
      func_build_cache_;
};

namespace transform {

Pass FoldConstant() {
  auto pass_func = [=](Function f, IRModule m, PassContext pc) {
    return ConstantFolder::Fold(f, m);
  };
  return CreateFunctionPass(pass_func, 0, "FoldConstant");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.FoldConstant", FoldConstant);
}

}  // namespace transform

}  // namespace relax
}  // namespace tvm
