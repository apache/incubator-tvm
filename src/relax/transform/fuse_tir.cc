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
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/ir/prim/op.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op/op.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/tirx/function.h>

#include <unordered_map>
#include <unordered_set>

namespace tvm {
namespace tirx {
using namespace tvm::prim;

/*!
 * \brief Match symbolic vars according to the given PrimExpr, and update the var_remap.
 * Will throw errors if there is a mismatch.
 */
class SymbolicMatcher : ExprFunctor<void(const Expr& n, const PrimExpr& other)> {
 public:
  explicit SymbolicMatcher(sym::AnalyzerObj* analyzer, ffi::Map<tvm::Var, PrimExpr>* var_remap)
      : analyzer_(analyzer), var_remap_(var_remap) {}

  void Match(const ffi::Array<PrimExpr>& params, const ffi::Array<PrimExpr>& args) {
    TVM_FFI_ICHECK_EQ(params.size(), args.size());
    for (size_t i = 0; i < params.size(); ++i) {
      Match(params[i], args[i]);
    }
  }
  void Match(const PrimExpr& param, const PrimExpr& arg) {
    Dispatch(param, arg);
    auto f_substitute = [this](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
      if (auto repl = var_remap_->Get(var)) return ffi::Any(*std::move(repl));
      return ffi::Unchanged();
    };
    must_prove_ =
        analyzer_->Simplify(ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(must_prove_, f_substitute)
                                .as_or_throw<PrimExpr>());
    TVM_FFI_ICHECK(!IsZero(must_prove_));
  }

 private:
  void Dispatch(const Expr& expr, const PrimExpr& other) final {
    PrimExpr node = expr.as_or_throw<PrimExpr>();
    if (node.same_as(other)) {
      return;
    } else if (node.ty().code() != other.ty().code()) {
      TVM_FFI_THROW(InternalError)
          << "Parameter expression " << node << " with dtype " << node.ty()->dtype
          << " cannot match to argument " << other << " with dtype " << other.ty()->dtype;
    } else {
      ExprFunctor::Dispatch(expr, other);
    }
  }

#define TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(OpName)                       \
  void Dispatch_(const OpName* op, const PrimExpr& other) {              \
    const auto* rhs = other.as<OpName>();                                \
    if (rhs) {                                                           \
      Dispatch(op->a, rhs->a);                                           \
      Dispatch(op->b, rhs->b);                                           \
    } else {                                                             \
      must_prove_ = must_prove_ && (ffi::GetRef<PrimExpr>(op) == other); \
    }                                                                    \
  }

  TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(prim::AddObj);
  TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(prim::SubObj);
  TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(prim::MulObj);
  TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(prim::DivObj);
  TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(prim::ModObj);
  TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(prim::EQObj);
  TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(prim::NEObj);
  TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(prim::LTObj);
  TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(prim::LEObj);
  TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(prim::GTObj);
  TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(prim::GEObj);
  TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(prim::AndObj);
  TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(prim::OrObj);
  TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(prim::MinObj);
  TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(prim::MaxObj);
  TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(prim::FloorDivObj);
  TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(prim::FloorModObj);

  void Dispatch_(const IntImmObj* op, const PrimExpr& other) {
    const auto* rhs = other.as<IntImmObj>();
    if (!rhs || (op->value != rhs->value)) {
      TVM_FFI_THROW(InternalError)
          << "Parameter expression " << ffi::GetRef<PrimExpr>(op)
          << " expected an integer argument with value " << op->value << ", "
          << "but was provided with the argument " << other;
    }
  }

  void Dispatch_(const FloatImmObj* op, const PrimExpr& other) {
    const auto* rhs = other.as<FloatImmObj>();
    if (!rhs || (op->value != rhs->value)) {
      TVM_FFI_THROW(InternalError) << "Parameter expression " << ffi::GetRef<PrimExpr>(op)
                                   << " expected an float argument with value " << op->value << ", "
                                   << "but was provided with the argument " << other;
    }
  }

  void Dispatch_(const prim::CastObj* op, const PrimExpr& other) {
    const auto* rhs = other.as<prim::CastObj>();
    if (!rhs) {
      TVM_FFI_THROW(InternalError)
          << "Parameter expression " << ffi::GetRef<PrimExpr>(op) << " expected an cast to "
          << op->ty.as_or_throw<PrimType>()->dtype << " as the argument, "
          << "but was provided with the argument " << other;
    }
    Dispatch(op->value, rhs->value);
  }

  void Dispatch_(const VarObj* op, const PrimExpr& rhs) {
    auto lhs = ffi::GetRef<Var>(op);
    PrimType lhs_ty = op->ty.as_or_throw<PrimType>();

    if (lhs.same_as(rhs)) {
      // Reference identity, no further checks needed.
    } else if (lhs_ty.code() != rhs.ty().code()) {
      TVM_FFI_THROW(InternalError)
          << "Parameter expression " << lhs << " with dtype " << lhs_ty->dtype
          << " cannot match to argument " << rhs << " with dtype " << rhs.ty()->dtype;
    } else if (auto it = var_remap_->find(lhs); it != var_remap_->end()) {
      Dispatch((*it).second, rhs);
    } else {
      var_remap_->Set(lhs, rhs);
    }
  }

  void Dispatch_(const prim::SelectObj* op, const PrimExpr& other) {
    const auto* rhs = other.as<prim::SelectObj>();
    if (rhs) {
      Dispatch(op->true_value, rhs->true_value);
      Dispatch(op->false_value, rhs->false_value);
    } else {
      must_prove_ = must_prove_ && (ffi::GetRef<PrimExpr>(op) == other);
    }
  }

  sym::AnalyzerObj* analyzer_;
  ffi::Map<tvm::Var, PrimExpr>* var_remap_;
  PrimExpr must_prove_ = IntImm::Bool(true);
};

/*!
 * \brief Substitute a given source tensor with a given target tensor in statements or expressions.
 */
class FuseTIRTensorSubstitutor : public s_tir::StmtExprMutator {
 public:
  explicit FuseTIRTensorSubstitutor(const ffi::Map<TensorVar, TensorVar>& tensor_map,
                                    const ffi::Map<Var, PrimExpr>& var_map) {
    for (const auto& [var, value] : var_map) {
      VarRemapSet(var, value);
    }
    for (const auto& [src, tgt] : tensor_map) {
      VarRemapSet(src, tgt);
    }
  }

  TensorVar SubstituteAllocatedTensor(TensorVar tensor) {
    TVM_FFI_ICHECK(VarRemapGet(tensor).type_index() == ffi::TypeIndex::kTVMFFINone);
    return WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&] {
      return Mutate(tensor).as_or_throw<UnchangedOr<TensorVar>>().ValueOrUnchanged(tensor);
    });
  }

 private:
  UnchangedOr<Stmt> Mutate_(const s_tir::SBlockObj* op, InplaceMode inplace_mode) final {
    s_tir::SBlock block = s_tir::StmtExprMutator::Mutate_(op, inplace_mode)
                              .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                              .as_or_throw<s_tir::SBlock>();
    ffi::Array<TensorRegion> reads = UnionAccessRegion(block->reads);
    ffi::Array<TensorRegion> writes = UnionAccessRegion(block->writes);
    if (!reads.same_as(block->reads) || !writes.same_as(block->writes)) {
      auto* n = block.CopyOnWrite();
      n->reads = std::move(reads);
      n->writes = std::move(writes);
    }
    return block;
  }

  ffi::Array<tvm::TensorRegion> UnionAccessRegion(const ffi::Array<TensorRegion>& regions) const {
    // For now we only allow tensors to access the same elements.
    // e.g. `[A[vi, vj], A[vi, vj]]` is a legal pattern but need to union to `A[vi, vj]`
    // However, `A[vi, vj], A[vi, vj + 1]` is not allow for now.
    // Note: the order of return region should remain the same as the first occurrence of the region
    ffi::Array<TensorRegion> ret;
    std::unordered_map<const VarObj*, ffi::Array<Range>> tensor_region_set;

    for (const TensorRegion& region : regions) {
      auto it = tensor_region_set.find(region->source.as_or_throw<tvm::tirx::TensorVar>().get());
      if (it == tensor_region_set.end()) {
        ret.push_back(region);
        tensor_region_set[region->source.as_or_throw<tvm::tirx::TensorVar>().get()] =
            region->region;
      }
    }

    if (ret.size() == regions.size()) {
      return regions;
    } else {
      return ret;
    }
  }
};

/*! \brief A mutator which detect block name duplication and deduplicate the names. */
class SBlockNameDeduplicator : public s_tir::StmtExprMutator {
 public:
  using s_tir::StmtExprMutator::Mutate;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) final {
    if (value.as<ExprObj>()) return ffi::Unchanged();
    return s_tir::StmtExprMutator::Mutate(value, inplace_mode);
  }

 private:
  UnchangedOr<Stmt> Mutate_(const s_tir::SBlockObj* op, InplaceMode inplace_mode) final {
    s_tir::SBlock block = s_tir::StmtExprMutator::Mutate_(op, inplace_mode)
                              .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                              .as_or_throw<s_tir::SBlock>();

    ffi::String name = GetUniqueName(block->name_hint);

    if (name == block->name_hint) {
      return block;

    } else {
      auto* n = block.CopyOnWrite();
      n->name_hint = std::move(name);
      return block;
    }
  }

  ffi::String GetUniqueName(const ffi::String& prefix) {
    std::string str_prefix = std::string(prefix);

    // Find where the trailing digits start
    size_t base_len = str_prefix.length();
    while (base_len > 0 && std::isdigit(str_prefix[base_len - 1])) {
      --base_len;
    }

    std::string base_name;
    int64_t start_num = 0;
    bool has_suffix = base_len < str_prefix.length();

    if (has_suffix) {
      base_name = str_prefix.substr(0, base_len);
      try {
        start_num = std::stoll(str_prefix.substr(base_len));
      } catch (const std::out_of_range&) {
        // Fallback: if the number is too large, treat the whole string as a base name.
        has_suffix = false;
        base_name = str_prefix;
      }
    } else {
      base_name = str_prefix;
    }

    // Check if the original name is available
    ffi::String candidate = prefix;
    if (!name_count_.count(candidate)) {
      name_count_[candidate] = 0;
      return candidate;
    }

    // Generate unique name by incrementing the numeric suffix
    int64_t counter = has_suffix ? start_num + 1 : 1;
    while (true) {
      candidate = ffi::String(base_name + std::to_string(counter));
      if (!name_count_.count(candidate)) {
        name_count_[candidate] = 0;
        return candidate;
      }
      ++counter;
      TVM_FFI_ICHECK_GT(counter, 0)
          << "Counter overflow when generating unique block name for prefix: " << prefix;
    }
  }

  /*! \brief The count map to make block name unique. */
  std::unordered_map<ffi::String, int> name_count_;
};

}  // namespace tirx

namespace relax {

static ffi::Array<int64_t> GetInplaceOutputIndices(const ffi::Array<int64_t>& inplace_indices,
                                                   int num_inputs) {
  ffi::Array<int64_t> ret;
  int last_idx = num_inputs;
  for (int64_t i : inplace_indices) {
    if (i >= 0) {
      ret.push_back(i);
    } else {
      TVM_FFI_ICHECK_EQ(i, -1)
          << "The only negative index expected in inplace_indices is -1, but got " << i;
      ret.push_back(last_idx);
      last_idx++;
    }
  }

  return ret;
}

class RelaxToTIRVarMapCollector : public ExprVisitor {
 public:
  explicit RelaxToTIRVarMapCollector(const IRModule& mod) : mod_(mod) {}
  static ffi::Map<Expr, tirx::TensorVar> Collect(const IRModule& mod, const Function& func) {
    RelaxToTIRVarMapCollector visitor(mod);
    visitor(func->body);
    return visitor.relax_to_tir_var_map_;
  }

 private:
  void VisitBinding_(const VarBindingObj* binding) final {
    current_var_ = binding->var;
    ExprVisitor::VisitBinding_(binding);
  }

  void VisitExpr_(const CallObj* call) {
    static const Op call_tir_op_ = Op::Get("relax.call_tir");
    static const Op call_tir_inplace_op_ = Op::Get("relax.call_tir_inplace");

    TVM_FFI_ICHECK(call->op.same_as(call_tir_op_) || call->op.same_as(call_tir_inplace_op_))
        << "Only call_tir and call_tir_inplace are supported in primitive function, but got: "
        << ffi::GetRef<Expr>(call);
    CollectVarMapping(call, current_var_, call->op.same_as(call_tir_inplace_op_));
  }

  void CollectVarMapping(const CallObj* call, const Expr& lhs_var, bool in_place) {
    GlobalVar gv = call->args[0].as_or_throw<GlobalVar>();
    tirx::Function function_ = mod_->Lookup(gv).as_or_throw<tirx::Function>();
    const auto& tir_args = function_->params;

    const auto& relax_args = call->args[1].as_or_throw<Tuple>()->fields;

    ffi::Array<Expr> relax_results;
    if (lhs_var->IsInstance<TupleObj>()) {
      relax_results = lhs_var.as_or_throw<Tuple>()->fields;
    } else {
      TVM_FFI_ICHECK(lhs_var->IsInstance<VarObj>())
          << "The lhs_var is expected to be either tuple or var";
      relax_results = {lhs_var.as_or_throw<Var>()};
    }

    size_t num_inputs = relax_args.size();
    size_t num_outputs = relax_results.size();

    ffi::Array<int64_t> output_idxs;
    if (in_place) {
      const auto* attrs = call->attrs.as<CallTIRInplaceAttrs>();
      TVM_FFI_ICHECK(attrs) << "Must have CallTIRInplaceAttrs for an in-place call";
      output_idxs = GetInplaceOutputIndices(attrs->inplace_indices, num_inputs);
    } else {
      for (size_t i = num_inputs; i < num_inputs + num_outputs; i++) {
        output_idxs.push_back(i);
      }
    }

    // If the `expr` is already seen (present in the map), validate whether the mapped tensor is
    // structurally equal to the `new_buf` passed
    auto ValidateTensorCompatibility = [this](tirx::TensorVar new_buf, Expr expr) {
      if (auto it = relax_to_tir_var_map_.find(expr); it != relax_to_tir_var_map_.end()) {
        TVM_FFI_ICHECK(ffi::StructuralEqual()((*it).second.ty(), new_buf.ty()))
            << "Inconsistent tensors " << (*it).second << " and " << new_buf
            << " mapped to the same relax var: " << expr;
      }
    };
    for (size_t i = 0; i < tir_args.size(); ++i) {
      const auto& tir_var = tir_args[i];
      if (auto tir_tensor = tir_var.as<tirx::TensorVar>()) {
        if (i < num_inputs) {
          const auto& relax_var = relax_args[i];
          ValidateTensorCompatibility(tir_tensor.value(), relax_var);
          relax_to_tir_var_map_.Set(relax_var, tir_tensor.value());
        }
        if (auto it = std::find(output_idxs.begin(), output_idxs.end(), i);
            it != output_idxs.end()) {
          int result_idx = it - output_idxs.begin();
          const auto& relax_var = relax_results[result_idx];
          ValidateTensorCompatibility(tir_tensor.value(), relax_var);
          relax_to_tir_var_map_.Set(relax_var, tir_tensor.value());
        }
      }
    }
  }

 private:
  /*! \brief The IRModule */
  const IRModule& mod_;
  ffi::Map<Expr, tirx::TensorVar> relax_to_tir_var_map_;
  Var current_var_{ffi::UnsafeInit{}};
};

class FusedTIRConstructor : public ExprVisitor {
 public:
  /*!
   * \brief Construct a fused TIR tirx::Function from a relax sub-function
   * \param mod The IRModule
   * \param gv The global var of relax subfunction to be fused into one tirx::Function
   * \return The fused TIR tirx::Function and the in-place indices (non-empty for an in-place call)
   */
  static std::pair<tirx::Function, ffi::Array<int64_t>> GetFusedTIR(const IRModule& mod,
                                                                    const GlobalVar& gv) {
    FusedTIRConstructor visitor(mod, gv->name_hint);
    BaseFunc f = mod->Lookup(gv);
    TVM_FFI_ICHECK(f->IsInstance<relax::FunctionObj>())
        << "Expected relax functions, but got: " << f->GetTypeKey();
    TVM_FFI_ICHECK(f->HasNonzeroAttr(tvm::relax::attr::kPrimitive.name))
        << "Expected a function with attr `kPrimitive`";
    visitor(f.as_or_throw<relax::Function>());
    ffi::Array<int64_t> inplace_indices;
    for (size_t idx : visitor.inplace_indices_) {
      inplace_indices.push_back(static_cast<int64_t>(idx));
    }
    return {visitor.fused_tir_, inplace_indices};
  }

 private:
  explicit FusedTIRConstructor(const IRModule& mod, const ffi::String& func_name)
      : mod_(mod), func_name_(func_name) {}

  void VisitExpr_(const FunctionObj* func) final {
    auto relax_to_tir_var_map =
        RelaxToTIRVarMapCollector::Collect(mod_, ffi::GetRef<Function>(func));
    std::vector<ffi::Variant<PrimVar, tirx::TensorVar>> function_params;
    for (const Var& relax_param : func->params) {
      size_t size_before = function_params.size();
      CollectFunctionParams(relax_param, &function_params, relax_to_tir_var_map.Get(relax_param));

      auto param_tensors = [&]() -> ffi::Array<tirx::TensorVar> {
        ffi::Array<tirx::TensorVar> out;
        for (size_t i = size_before; i < function_params.size(); i++) {
          if (auto buf = function_params[i].as<tirx::TensorVar>()) {
            out.push_back(buf.value());
          }
        }
        return out;
      }();

      func_info_.expr2tensors.Set(relax_param, param_tensors);
    }

    // Preserve the Relax function's parameter order.  Tensor and primitive
    // parameters are both explicit call_tir arguments, while output tensors
    // are appended after the complete explicit argument prefix.
    for (const auto& param : function_params) {
      if (auto opt = param.as<tirx::TensorVar>()) {
        auto tensor = opt.value();
        // Differentiate tensor name and param name by adding prefix
        // `p_` to the tensor name.  Every symbol should be unique in
        // TVMScript, and while they can be de-deplicated when
        // printed, it's more readable when done explicitly.  Since
        // TensorVar is used more than param it gets the name with better
        // readability.
        tvm::Var param = tvm::Var("p_" + tensor->name, PtrType::VoidPtrType());
        func_info_.params.push_back(param);
        func_info_.tensor_map.Set(param, tensor);
      } else if (auto var = param.as<PrimVar>()) {
        func_info_.params.push_back(var.value());
      }
    }

    // Step 2. Visit Function body and create intermediate tensors
    ExprVisitor::VisitExpr_(func);

    // Step 3. Create and remap tensors for function output
    Expr body = func->body->body;
    auto it = func_info_.expr2tensors.find(body);
    TVM_FFI_ICHECK(it != func_info_.expr2tensors.end())
        << "Fail to detect output tensors for function body";

    const ffi::Array<tirx::TensorVar>& tensors = (*it).second;

    // map of input tensors to indices (helpful for detecting in-place inputs)
    std::unordered_map<tirx::TensorVar, size_t, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
        tensor_to_idx;
    std::unordered_map<tvm::Var, size_t> input_to_idx;
    for (size_t i = 0; i < func_info_.params.size(); i++) {
      input_to_idx[func_info_.params[i]] = i;
    }
    for (auto [var, tensor] : func_info_.tensor_map) {
      if (auto it = input_to_idx.find(var); it != input_to_idx.end()) {
        tensor_to_idx[tensor] = (*it).second;
      }
    }

    // numbered separately because the number of output *vars* might differ from the
    // number of outputs if there are in-place inputs
    int out_idx = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
      // Do not add output vars for in-place inputs
      // (i.e., already listed among the tensor parameters, which would
      // otherwise result in duplicate parameters)
      if (auto it = tensor_to_idx.find(tensors[i]); it != tensor_to_idx.end()) {
        auto idx = (*it).second;
        TVM_FFI_ICHECK(!inplace_indices_.count(idx))
            << "In-place index " << idx << " used twice! An argument must be aliased.";
        inplace_indices_.insert(idx);
        continue;
      }

      tvm::Var param = tvm::Var("p_output" + std::to_string(out_idx), PtrType::VoidPtrType());
      out_idx++;
      func_info_.tensor_map.Set(param, tensors[i]);
      func_info_.params.push_back(param);
      func_info_.output_tensors.insert(tensors[i].get());
    }

    // Step 4. Create tirx::Function
    fused_tir_ = ConstructFunc();
  }

  void VisitBinding_(const VarBindingObj* binding) final {
    // Update expr2tensors by visiting values.
    this->VisitExpr(binding->value);
    auto it = func_info_.expr2tensors.find(binding->value);
    if (it != func_info_.expr2tensors.end()) {
      // assign binding var to the tensors of the value
      func_info_.expr2tensors.Set(binding->var, (*it).second);
    } else {
      TVM_FFI_THROW(InternalError) << "Unsupported binding value: " << binding->value;
    }
  }

  void VisitBinding_(const MatchCastObj* match_cast) final {
    TVM_FFI_THROW(InternalError) << "MatchCast is unsupported in primitive functions";
  }

  void VisitExpr_(const CallObj* call) final {
    ExprVisitor::VisitExpr_(call);
    static const Op call_tir_op_ = Op::Get("relax.call_tir");
    static const Op call_tir_inplace_op_ = Op::Get("relax.call_tir_inplace");

    TVM_FFI_ICHECK(call->op.same_as(call_tir_op_) || call->op.same_as(call_tir_inplace_op_))
        << "Only call_tir and call_tir_inplace are supported in primitive function, but got: "
        << ffi::GetRef<Expr>(call);

    // Step 1. Get Global var and tirx::Function
    GlobalVar gv = call->args[0].as_or_throw<GlobalVar>();
    tirx::Function function_ = mod_->Lookup(gv).as_or_throw<tirx::Function>();

    // Step 2. Renew all vars/tensor definitions and blocks to avoid duplication
    tirx::Function function = tirx::RenewDef(function_);

    // Step 3. Check functions are all schedulable funcs. i.e. the body of func is root block
    // TODO(Siyuan): support un-schedulable functions.
    TVM_FFI_ICHECK(function->body.has_value() && function->body.value()->size() == 1 &&
                   function->body.value()->seq[0].as<s_tir::SBlockRealizeObj>())
        << "Only schedulable functions (whose body is the root block) can be fused";
    s_tir::SBlockRealize root_realize =
        function->body.value()->seq[0].as_or_throw<s_tir::SBlockRealize>();
    const s_tir::SBlock& root_block = root_realize->block;

    // Step 4. Add all the original alloc_tensors and body to the fused function.
    func_info_.alloc_tensors.insert(func_info_.alloc_tensors.end(),
                                    root_block->alloc_tensors.begin(),
                                    root_block->alloc_tensors.end());
    func_info_.bodies.push_back(root_block->body);

    // Step 5. Map input arguments to tensor
    MapInputTensor(function, call->args[1]);
    const ffi::Array<ffi::Array<PrimExpr>>& output_tensor_shapes = GetCallTIROutputShapes(call);

    AllocateIntermediateTensor(call, function, output_tensor_shapes);

    // Update fused func name
    func_info_.global_name += "_" + gv->name_hint;
  }

  void VisitExpr_(const TupleGetItemObj* tuple_get_item) final {
    ExprVisitor::VisitExpr_(tuple_get_item);
    auto it = func_info_.expr2tensors.find(tuple_get_item->tuple);
    if (it != func_info_.expr2tensors.end()) {
      int begin_buf_idx = 0;
      int end_buf_idx = 0;
      const TupleType& tuple_ty = tuple_get_item->tuple->ty.as_or_throw<TupleType>();
      for (int i = 0; i < tuple_get_item->index; ++i) {
        begin_buf_idx += GetTotalTensorSize(tuple_ty->fields[i]);
      }
      end_buf_idx = begin_buf_idx + GetTotalTensorSize(tuple_ty->fields[tuple_get_item->index]);
      func_info_.expr2tensors.Set(
          ffi::GetRef<Expr>(tuple_get_item),
          {(*it).second.begin() + begin_buf_idx, (*it).second.begin() + end_buf_idx});
    }
  }

  void VisitExpr_(const TupleObj* tuple) final {
    ExprVisitor::VisitExpr_(tuple);
    ffi::Array<tirx::TensorVar> tensors;
    for (const Expr& expr : tuple->fields) {
      auto it = func_info_.expr2tensors.find(expr);
      if (it != func_info_.expr2tensors.end()) {
        tensors.insert(tensors.end(), (*it).second.begin(), (*it).second.end());
      }
    }
    if (!tensors.empty()) {
      func_info_.expr2tensors.Set(ffi::GetRef<Expr>(tuple), tensors);
    }
  }

  void VisitExpr_(const GenericConstObj* op) final {
    if (!op->value.as<ffi::TensorObj>()) return;
    TVM_FFI_THROW(InternalError) << "Tensor constants are not supported in primitive functions.";
  }

  /*!
   * \brief Get the number of outputs for a call_tir node.
   * \return The number of outputs.
   */
  static ffi::Array<ffi::Array<PrimExpr>> GetCallTIROutputShapes(const CallObj* call) {
    static const Op call_tir_op_ = Op::Get("relax.call_tir");
    static const Op call_tir_inplace_op_ = Op::Get("relax.call_tir_inplace");
    TVM_FFI_ICHECK(call->op.same_as(call_tir_op_) || call->op.same_as(call_tir_inplace_op_));
    TVM_FFI_ICHECK_EQ(call->ty_args.size(), 1);
    auto get_tensor_shape =
        [](const TensorTypeObj* ty) {
          const auto* shape_expr = ty->shape.as<ShapeExprObj>();
          TVM_FFI_ICHECK(shape_expr)
              << "FuseTIR expects all parameters are Tensors with symbolic shape.";
          return shape_expr->values;
        };
    if (const auto* tuple_ty = call->ty_args[0].as<TupleTypeObj>()) {
      ffi::Array<ffi::Array<PrimExpr>> shapes;
      for (const Type& field : tuple_ty->fields) {
        const auto* tensor_ty = field.as<TensorTypeObj>();
        TVM_FFI_ICHECK(tensor_ty) << "CallTIR ty_args are expected to be TensorType or Tuple of "
                                     "TensorType, but got "
                                  << call->ty_args[0];
        shapes.push_back(get_tensor_shape(tensor_ty));
      }
      return shapes;
    } else if (const auto* tensor_ty = call->ty_args[0].as<TensorTypeObj>()) {
      return {get_tensor_shape(tensor_ty)};
    } else {
      TVM_FFI_ICHECK(tensor_ty) << "CallTIR ty_args are expected to be TensorType or Tuple of "
                                   "TensorType, but got "
                                << call->ty_args[0];
      throw;
    }
  }

  /*! \brief Map old TIR func param tensor to new tensor, and then update `tensor_subst_map` */
  void MapArgsToTensor(const ffi::Array<Expr> args, const ffi::Array<tirx::TensorVar>& tensors) {
    size_t tensor_idx = 0;
    for (const Expr& arg : args) {
      if (const auto* v = arg.as<VarObj>()) {
        auto it = func_info_.expr2tensors.find(ffi::GetRef<Var>(v));
        // Substitute the tensor with the already allocated one if it is an intermediate var
        if (it != func_info_.expr2tensors.end()) {
          for (const tirx::TensorVar& target_tensor : (*it).second) {
            TVM_FFI_ICHECK_LT(tensor_idx, tensors.size());
            const tirx::TensorVar& tensor = tensors[tensor_idx];
            func_info_.symbolic_var_matcher.Match(tensor.shape(), target_tensor.shape());
            func_info_.tensor_subst_map.Set(tensor, target_tensor);
            tensor_idx++;
          }
        }
      }
    }
    // Make sure every tensor is mapped.
    TVM_FFI_ICHECK_EQ(tensor_idx, tensors.size());
  }

  /*!
   * \brief Update tensor mapping `func_info_.tensor_subst_map` for input args
   * \param func The old TIR tirx::Function
   * \param output_size The number of output params. All output params are at the end of param list.
   */
  void MapInputTensor(const tirx::Function& func, const relax::Expr& args) {
    ffi::Array<Expr> arg_list;
    ffi::Array<tirx::TensorVar> tensor_list;
    ffi::Array<Expr> call_args = args.as_or_throw<Tuple>()->fields;

    TVM_FFI_ICHECK_GE(func->params.size(), call_args.size());
    for (size_t i = 0; i < call_args.size(); ++i) {
      const Expr& arg = call_args[i];
      const tvm::Var& param = func->params[i];
      if (auto tensor = param.as<tirx::TensorVar>()) {
        arg_list.push_back(arg);
        tensor_list.push_back(tensor.value());
      } else {
        auto prim_arg = arg.as<PrimExpr>();
        TVM_FFI_CHECK(prim_arg.has_value(), TypeError)
            << "Expected scalar parameter " << param
            << " to receive an individual primitive expression, but " << arg << " has type "
            << GetType(arg);
        func_info_.symbolic_var_matcher.Match(param.as_or_throw<PrimExpr>(), prim_arg.value());
      }
    }

    MapArgsToTensor(arg_list, tensor_list);
  }

  static ffi::Array<tirx::TensorVar> GetFunctionOutputParams(
      const tirx::Function& func, const ffi::Array<int64_t>& output_indices) {
    size_t n = func->params.size();
    size_t output_size = output_indices.size();
    TVM_FFI_ICHECK_GE(n, output_size);

    ffi::Array<tirx::TensorVar> ret;
    for (int64_t idx : output_indices) {
      int i = static_cast<int>(idx);
      const tvm::Var& param = func->params[static_cast<size_t>(i)];
      auto tensor = param.as<tirx::TensorVar>();
      TVM_FFI_ICHECK(tensor.has_value())
          << "The output params of a tirx::Function must be tensors, but parameter " << i
          << " has type " << param->ty;
      ret.push_back(tensor.value());
    }
    return ret;
  }

  /*!
   * \brief Allocate tensor(s) and update `func_info.expr2tensors` if the tirx::Function output(s)
   * are intermediate results.
   * \param expr The relax Expr, which can be binding vars or binding values.
   * \param func The old TIR tirx::Function
   * \param output_shapes The shape of output params.
   */
  void AllocateIntermediateTensor(const CallObj* call, const tirx::Function& func,
                                  const ffi::Array<ffi::Array<PrimExpr>>& output_shapes) {
    bool is_inplace = call->op.same_as(Op::Get("relax.call_tir_inplace"));

    size_t n = func->params.size();
    int num_inputs = call->args[1].as_or_throw<Tuple>()->fields.size();
    size_t output_size = output_shapes.size();
    TVM_FFI_ICHECK_GE(n, output_size);
    ffi::Array<tirx::TensorVar> output_tensors;
    ffi::Array<int64_t> output_idxs;
    if (is_inplace) {
      const auto* attrs = call->attrs.as<CallTIRInplaceAttrs>();
      TVM_FFI_ICHECK(attrs) << "Must have CallTIRInplaceAttrs for an in-place call";
      output_idxs = GetInplaceOutputIndices(attrs->inplace_indices, num_inputs);
    } else {
      for (size_t i = 0; i < output_size; i++) {
        output_idxs.push_back(num_inputs + i);
      }
    }

    ffi::Array<tirx::TensorVar> output_params = GetFunctionOutputParams(func, output_idxs);
    for (size_t i = 0; i < output_size; ++i) {
      const tirx::TensorVar& tensor = output_params[i];

      // if this is an inplace output, do not do an intermediate allocation
      if (output_idxs[i] < num_inputs) {
        auto it = func_info_.tensor_subst_map.find(tensor);
        TVM_FFI_ICHECK(it != func_info_.tensor_subst_map.end())
            << "Inplace output tensor " << tensor << " must be mapped to a defined input";
        output_tensors.push_back((*it).second);
        continue;
      }

      auto unify_name_hints = [this, &tensor]() {
        ffi::String base_name = tensor->name;
        ffi::String unique_name = base_name + "_intermediate";
        size_t unique_id = 0;
        std::unordered_set<std::string> names;

        for (auto& _tensor : func_info_.alloc_tensors) {
          names.insert(_tensor->name);
        }

        while (names.find(unique_name) != names.end()) {
          unique_name = unique_name + "_" + std::to_string(++unique_id);
        }
        return unique_name;
      };
      // Update tensor with new symbolic shape according to the ty
      auto new_type_node = ffi::make_object<tirx::TensorTypeObj>(*tensor.ty().get());
      new_type_node->shape = output_shapes[i];
      tirx::TensorType new_type(std::move(new_type_node));
      tirx::TensorVar new_tensor(unify_name_hints(), std::move(new_type), tensor->loc);
      func_info_.alloc_tensors.push_back(new_tensor);
      output_tensors.push_back(new_tensor);

      // Match the shape of the output tensor with the shape
      func_info_.symbolic_var_matcher.Match(tensor.shape(), new_tensor.shape());
      func_info_.tensor_subst_map.Set(tensor, new_tensor);
    }
    // Update expr2tensors
    func_info_.expr2tensors.Set(ffi::GetRef<Expr>(call), output_tensors);
  }

  /*!
   * \brief Collect TIR func params and tensors with specified relax type and shape
   * \param ty The type
   * \param name_hint The name hint for params and tensors
   * \param out The vector into which to collect the params/tensors
   */
  static void CollectFunctionParams(const Var& relax_param,
                                    std::vector<ffi::Variant<PrimVar, tirx::TensorVar>>* out,
                                    const ffi::Optional<tirx::TensorVar>& tir_tensor_param) {
    auto ty = GetType(relax_param);

    TVM_FFI_CHECK(!ty.as<TupleTypeObj>(), InternalError)
        << "All tuple parameters should be expanded before this point in FuseTIR.  "
        << "However, parameter " << relax_param << " has type " << ty;

    auto name_hint = relax_param->name;

    if (const auto* tensor_type = ty.as<TensorTypeObj>()) {
      // Case 1. The relax param is a Tensor, we directly create a tirx var and tensor
      const auto* shape_expr = tensor_type->shape.as<ShapeExprObj>();
      TVM_FFI_ICHECK(shape_expr) << "FuseTIR expects all Tensor parameters have a known shape.";
      PrimType dtype = tensor_type->dtype.value();
      tirx::TensorVar tensor =
          tir_tensor_param.has_value()
              ? tirx::TensorVar(name_hint, tirx::TensorType(shape_expr->values, dtype,
                                                            tir_tensor_param.value().scope()))
              : tirx::TensorVar(name_hint, tirx::TensorType(shape_expr->values, dtype));
      out->push_back(std::move(tensor));

    } else if (ty.as<PrimTypeObj>()) {
      // Case 2. The relax param is a scalar, so its canonical Var is a TIR parameter.
      out->push_back(relax_param.as_or_throw<PrimVar>());

    } else if (const auto* shape_expr = ty.as<ShapeTypeObj>()) {
      // Case 3. The relax param is a tuple of scalars, each represented as a tirx var
      for (const auto& var : shape_expr->values.value()) {
        auto prim_var = var.as<PrimVar>();
        TVM_FFI_ICHECK(prim_var.has_value());
        out->push_back(prim_var.value());
      }
    } else {
      TVM_FFI_THROW(TypeError) << "The param type of tirx::Function is expected to be "
                               << "Tensor, PrimExpr, or ShapeExpr, "
                               << "but got " << ty->GetTypeKey();
    }
  }

  /*!
   * \brief Construct fused TIR func with collected FuseFuncInfo
   * \return The fused TIR
   */
  tirx::Function ConstructFunc() {
    ffi::Map<ffi::String, Any> attr_map;
    attr_map.Set(tvm::tirx::attr::kNoAlias.name, true);
    attr_map.Set(tvm::attr::kScriptNamespace.name, ffi::String("s_tir"));
    auto subst = ffi::make_object<tirx::FuseTIRTensorSubstitutor>(func_info_.tensor_subst_map,
                                                                  func_info_.symbolic_var_remap);
    TVM_FFI_ICHECK(func_info_.global_name != "fused");
    // Remove output tensors from func_info_.alloc_tensors
    ffi::Array<tirx::TensorVar> alloc_tensors;
    for (const tirx::TensorVar& buf : func_info_.alloc_tensors) {
      if (func_info_.output_tensors.count(buf.get()) == 0) {
        alloc_tensors.push_back(subst->SubstituteAllocatedTensor(buf));
      }
    }
    tvm::Stmt body = tvm::SeqStmt(func_info_.bodies);
    body = ffi::make_object<tirx::SBlockNameDeduplicator>()->Mutate(body).ValueOrUnchanged(body);

    body = subst->Mutate(body).ValueOrUnchanged(body);
    body = s_tir::SBlock({}, {}, {}, "root", std::move(body), std::nullopt, alloc_tensors);
    body = s_tir::SBlockRealize({}, IntImm::Bool(true), body.as_or_throw<s_tir::SBlock>());
    ffi::Array<tvm::Var> params = func_info_.params.Map([&](const tvm::Var& param) -> tvm::Var {
      if (auto tensor = func_info_.tensor_map.Get(param)) {
        return tensor.value();
      }
      return param;
    });
    tirx::Function func(params, tvm::SeqStmt(body), VoidType(), DictAttrs(attr_map));
    // Renew function defs to prevent using the same symbolic vars in different functions
    return tirx::RenewDef(func);
  }

  /*! \brief Get DynTensor numbers from recursive Tuples. */
  static size_t GetTotalTensorSize(const Type& ty) {
    if (ty.as<TensorTypeObj>()) {
      return 1;
    } else if (const auto* tuple_ty = ty.as<TupleTypeObj>()) {
      size_t num = 0;
      for (const Type& ty : tuple_ty->fields) {
        num += GetTotalTensorSize(ty);
      }
      return num;
    } else {
      TVM_FFI_THROW(InternalError) << "TensorType and TupleType are expect, but got: " << ty;
      return 0;
    }
  }

  /********** Function Info **********/

  /*! \brief auxiliary information for FuseTIR */
  struct FuseFuncInfo {
    /*! \brief The arguments for calling function */
    ffi::Array<Expr> arguments;
    /*!
     * \brief The map from each dataflow var (intermediate var) to the corresponding tensors
     * allocated in the fused func
     */
    ffi::Map<Expr, ffi::Array<tirx::TensorVar>> expr2tensors;
    /*! \brief The tensors to allocate in the fused func*/
    ffi::Array<tirx::TensorVar> alloc_tensors;
    /*! \brief The bodies of the original funcs, which is also the body of the fused func. */
    ffi::Array<tvm::Stmt> bodies;
    /*! \brief The params of the fused function*/
    ffi::Array<tvm::Var> params;
    /*!
     * \brief The map from tensor in original functions to corresponding tensor in the fused
     * function
     */
    ffi::Map<tirx::TensorVar, tirx::TensorVar> tensor_subst_map;
    /*! \brief Tensor annotations keyed by their placeholder parameters. */
    ffi::Map<tvm::Var, tirx::TensorVar> tensor_map;
    /*! \brief The output tensors among the function parameters. */
    std::unordered_set<const tvm::VarObj*> output_tensors;
    /*! \brief The name of the fused function */
    std::string global_name = "fused";

    /*! \brief The map from symbolic var to its value in the fused function
     *
     * This is used in the default initialization of
     * `symbolic_var_matcher`, and must be before it in the struct
     * order.
     */
    ffi::Map<tvm::Var, PrimExpr> symbolic_var_remap;

    /*! \brief The map from symbolic var to its value in the fused function
     *
     * This is used in the default initialization of
     * `symbolic_var_matcher`, and must be before it in the struct
     * order.
     */
    sym::Analyzer analyzer;

    /*! \brief The map from symbolic var to its corresponding var in the fused function */
    tirx::SymbolicMatcher symbolic_var_matcher =
        tirx::SymbolicMatcher(analyzer.get(), &symbolic_var_remap);
  };

  /*! \brief The IRModule */
  const IRModule& mod_;
  /*! \brief The name hint for the input func. */
  ffi::String func_name_;
  /*! \brief The helper info to fuse TIR function */
  FuseFuncInfo func_info_;
  /*! \brief The tirx function after fusion*/
  tirx::Function fused_tir_{ffi::UnsafeInit{}};
  /*! \brief Indices of inputs that are used for in-place computation */
  std::unordered_set<size_t> inplace_indices_;
};

std::vector<size_t> GetTupleAccessedIndices(const FunctionObj* func, const Var& tuple_var) {
  // Need to be ordered
  std::vector<size_t> indices;
  PostOrderVisit(func->body, [&indices, tuple_var](Expr e) {
    if (auto tup_get = e.as<TupleGetItemObj>(); tup_get && tup_get->tuple.same_as(tuple_var)) {
      if (std::find(indices.begin(), indices.end(), tup_get->index) == indices.end()) {
        indices.push_back(tup_get->index);
      }
    }
  });
  return indices;
}

/*!
 * \brief The helper class to fuse TIR functions and build a new module which calls the fused TIR.
 */
class TIRFuseMutator : public ExprMutator {
 public:
  static IRModule Transform(IRModule mod) {
    // Collect all primitive relax functions
    ffi::Map<GlobalVar, Function> primitive_relax;
    for (const auto& gvar : mod->GetGlobalVars()) {
      const auto& base_func = mod->Lookup(gvar);
      // Only fuse primitive relax functions
      if (base_func->HasNonzeroAttr(tvm::relax::attr::kPrimitive.name)) {
        if (auto func = base_func.as<relax::Function>()) {
          primitive_relax.Set(gvar, func.value());
        }
      }
    }

    if (primitive_relax.empty()) {
      return mod;
    }

    mod.CopyOnWrite();

    IRModule updates;
    std::unordered_map<GlobalVar, Replacement> replacements;

    // Since TIRFuseMutator will delete bunch of tirx::Function, we create an empty block builder.

    // Step 1. Fuse all primitive relax functions, store the result in `fused_tir_funcs_`
    for (const auto& [old_gvar, func] : primitive_relax) {
      const auto& [function, indices] = FusedTIRConstructor::GetFusedTIR(mod, old_gvar);

      GlobalVar new_gvar(old_gvar->name_hint);
      UpdateType(new_gvar, GetType(function));

      mod->Remove(old_gvar);
      updates->Add(new_gvar, function);
      replacements.insert_or_assign(old_gvar, Replacement{new_gvar, func, indices});
    }

    TIRFuseMutator mutator(replacements);

    // Step 2. Update all non-primitive relax functions and add it, with the dependent function,
    // into the new IRModule

    for (const auto& [gv, func] : mod->functions) {
      if (func->IsInstance<relax::FunctionObj>()) {
        TVM_FFI_ICHECK(!func->HasNonzeroAttr(tvm::relax::attr::kPrimitive.name))
            << "Module should not contain any primitive relax functions at this point";
        relax::Function update_func = mutator.VisitExpr(func).as_or_throw<Function>();
        if (!update_func.same_as(func)) {
          updates->Add(gv, update_func);
        }
      }
    }

    // Step 4. Copy over updated functions and return.
    mod->Update(updates);
    return mod;
  }

 private:
  struct Replacement {
    GlobalVar fused_tir_gvar;
    Function original_function;
    ffi::Array<int64_t> inplace_indices;
  };

  explicit TIRFuseMutator(std::unordered_map<GlobalVar, Replacement> replacements)
      : replacements_(replacements) {}

  using ExprMutator::VisitExpr_;

  // Get shape from call tirx
  static Expr GetCallTIRShape(Type ty) {
    if (auto* tuple = ty.as<TupleTypeObj>()) {
      ffi::Array<Expr> fields = tuple->fields.Map([&](Type x) { return GetCallTIRShape(x); });
      return Tuple(fields);
    } else {
      auto* tensor = ty.as<TensorTypeObj>();
      TVM_FFI_ICHECK(tensor) << "FuseTIR can only take tensor or tuple type";
      auto* shape_expr = tensor->shape.as<ShapeExprObj>();
      TVM_FFI_ICHECK(shape_expr) << "FuseTIR requires all intermediate values have shape";
      return ffi::GetRef<ShapeExpr>(shape_expr);
    }
  }

  Expr VisitExpr_(const CallObj* op) final {
    static const Op call_tir_op_ = Op::Get("relax.call_tir");
    static const Op call_tir_inplace_op_ = Op::Get("relax.call_tir_inplace");

    Call call = builder_->Normalize(ExprMutator::VisitExpr_(op)).as_or_throw<Call>();

    auto opt_gvar = call->op.as<GlobalVar>();
    if (!opt_gvar) {
      // Case 1. The Call isn't a relax-to-relax function call, no need to update.
      return call;
    }
    GlobalVar old_gvar = opt_gvar.value();

    auto it = replacements_.find(old_gvar);
    if (it == replacements_.end()) {
      // Case 2. The callee function is not a primitive relax
      // function, no need to update.
      return call;
    }
    const Replacement& replacement = it->second;
    const GlobalVar& fused_tir_gv = replacement.fused_tir_gvar;
    const Function& relax_func = replacement.original_function;

    // Case 3. It calls a primitive relax function, update the call
    // into a call_tir or call_tir_inplace.

    // Step a. Collect all relax/symbolic arguments.  Tuple arguments
    // are not supported by tirx::Function, so this step verifies that
    // ExpandTupleArguments has already removed them.
    ffi::Array<Expr> arg_list;
    for (size_t i = 0; i < call->args.size(); ++i) {
      auto arg = call->args[i];
      auto ty = GetType(arg);

      TVM_FFI_CHECK(
          !relax_func->params[i]->ty->IsInstance<TupleTypeObj>() && !ty.as<TupleTypeObj>(),
          InternalError)
          << "All tuple parameters should be expanded before this point in FuseTIR.  "
          << "However, argument " << arg << " with type " << arg->ty << " is passed as argument "
          << i << " to Primitive Relax function " << old_gvar << ", which expects parameter "
          << relax_func->params[i] << " to have type " << relax_func->params[i]->ty;

      if (const auto* shape = ty.as<ShapeTypeObj>()) {
        TVM_FFI_ICHECK(shape->values.has_value())
            << "FuseTIR requires all shape input has ty value.";
        for (const PrimExpr& prim_value : shape->values.value()) {
          TVM_FFI_ICHECK(prim_value.as<PrimVar>())
              << "All shape inputs are expected to be single tirx var.";
          arg_list.push_back(prim_value);
        }
      } else if (ty.as<PrimTypeObj>()) {
        if (auto literal = arg.as<PrimExpr>()) {
          arg_list.push_back(literal.value());
        } else {
          TVM_FFI_THROW(TypeError) << "FuseTIR expects scalar arguments to be PrimExpr, "
                                   << "but received " << arg;
        }

      } else {
        arg_list.push_back(arg);
      }
    }

    // Step b. Create call_tir or call_tir_inplace
    ffi::Array<Expr> call_args = {fused_tir_gv, Tuple(arg_list)};
    Op call_op = call_tir_op_;
    ffi::Optional<Attrs> call_attrs = call->attrs;
    if (replacement.inplace_indices.size()) {
      call_op = call_tir_inplace_op_;
      auto inplace_attrs = ffi::make_object<CallTIRInplaceAttrs>();
      inplace_attrs->inplace_indices = replacement.inplace_indices;
      call_attrs = Attrs(inplace_attrs);
    }
    return Call(Type::Missing(), call_op, call_args, call_attrs, {GetType(call)});
  }

 private:
  /*! \brief The map from global var to how it should be replaced
   *
   * Has one entry for each primitive relax function in the IRModule.
   */
  std::unordered_map<GlobalVar, Replacement> replacements_;
};

IRModule FuseTIR(IRModule mod) {
  mod = TIRFuseMutator::Transform(mod);
  return mod;
}

namespace transform {

Pass FuseTIR() {
  auto pass_func =  //
      [=](IRModule m, PassContext pc) { return relax::FuseTIR(m); };
  auto inner_pass = CreateModulePass(/*pass_function=*/pass_func,  //
                                     /*opt_level=*/0,              //
                                     /*pass_name=*/"FuseTIRInner");
  return tvm::transform::Sequential(
      {
          ExpandTupleArguments(),
          RemoveUnusedParameters(),
          inner_pass,
          DeadCodeElimination(),
      },
      "FuseTIR");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.FuseTIR", FuseTIR);
}

}  // namespace transform

}  // namespace relax
}  // namespace tvm
