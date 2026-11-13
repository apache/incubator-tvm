/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file lower_match_tensor.cc
 * \brief The pass for lowering match_tensor.
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/transform.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/type.h>

#include "../transform/ir_utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::prim;
using namespace tvm::tirx;
class MatchTensorLower : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  explicit MatchTensorLower(const Function& func) {
    for (const Var& param : func->params) {
      // Mark input var as const variable.
      auto prim_type = param->ty.as<PrimType>();
      if (prim_type) {
        VarRemapSet(param, param.as_or_throw<PrimExpr>());
      }
    }
  }

 private:
  UnchangedOr<Stmt> Mutate_(const SBlockObj* op, InplaceMode inplace_mode) final {
    for (const MatchTensorRegion& match_tensor : op->match_tensors) {
      CheckAndUpdateVarMap(match_tensor);
    }
    // Preserve match-tensor lookup keys when the inherited Var environment
    // remaps their tensor type annotations.
    std::vector<TensorVar> orig_tensors;
    for (const auto& kv : match_tensors_) {
      orig_tensors.push_back(kv.first);
    }
    SBlock stmt = StmtExprMutator::Mutate_(op, inplace_mode)
                      .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                      .as_or_throw<SBlock>();
    // Add remapped tensor keys to match_tensors_
    for (const TensorVar& orig_tensor : orig_tensors) {
      if (auto remapped = VarRemapGet(orig_tensor).as<TensorVar>()) {
        if (!match_tensors_.count(remapped.value())) {
          match_tensors_.Set(remapped.value(), match_tensors_[orig_tensor]);
        }
      }
    }
    op = stmt.as<SBlockObj>();
    TVM_FFI_ICHECK(op != nullptr);
    ffi::Array<TensorRegion> reads =
        op->reads.Map(std::bind(&MatchTensorLower::VisitTensorRegion, this, std::placeholders::_1));
    ffi::Array<TensorRegion> writes = op->writes.Map(
        std::bind(&MatchTensorLower::VisitTensorRegion, this, std::placeholders::_1));

    if (reads.same_as(op->reads) && writes.same_as(op->writes) && op->match_tensors.empty()) {
      return stmt;
    } else {
      auto* n = stmt.CopyOnWrite();
      // Match tensors are aliases of their source region. Their placement belongs
      // to that alias definition, which disappears along with the match tensor.
      if (auto value = n->annotations.Get(s_tir::attr::kTensorAllocatedAddr.name)) {
        TensorAllocatedAddresses addresses;
        for (const auto& entry : value.value().cast<TensorAllocatedAddresses>()) {
          bool is_alias = false;
          for (const auto& match : n->match_tensors) {
            is_alias |= entry.get<0>().same_as(match->tensor);
          }
          if (!is_alias) addresses.push_back(entry);
        }
        if (addresses.empty()) {
          n->annotations.erase(s_tir::attr::kTensorAllocatedAddr.name);
        } else {
          n->annotations.Set(s_tir::attr::kTensorAllocatedAddr.name, addresses);
        }
      }
      n->match_tensors = {};
      n->reads = std::move(reads);
      n->writes = std::move(writes);
      return stmt;
    }
  }

  UnchangedOr<Stmt> Mutate_(const ForObj* op, InplaceMode inplace_mode) final {
    {
      auto context = analyzer_->GetConstraintContext();
      context->Append(sym::RangeConstraint(op->var, Range(op->begin, op->end)));
      analyzer_->SyncConstraintContext(context);
    }
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Expr> Mutate_(const CallObj* op, InplaceMode inplace_mode) final {
    if ((op->op.same_as(tirx::masked_load_op()) || op->op.same_as(tirx::masked_store_op())) &&
        !op->args.empty()) {
      if (auto var = op->args[0].as<Var>(); var && var.value()->ty.as<TensorTypeObj>()) {
        TensorVar tensor = var.value().as_or_throw<TensorVar>();
        TVM_FFI_ICHECK(!match_tensors_.count(tensor))
            << "Predicated tensor access is not currently supported in lower match tensor pass.";
      }
    }
    if (op->op.same_as(tirx::tensor_data_ptr_op()) && op->args.size() == 1) {
      if (auto var = op->args[0].as<Var>();
          var.has_value() && var.value()->ty.as<TensorTypeObj>()) {
        auto it = match_tensors_.find(var.value().as_or_throw<TensorVar>());
        if (it != match_tensors_.end()) {
          return (*it).second->source.as_or_throw<tvm::tirx::TensorVar>().data();
        }
      }
    }
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreObj* op, InplaceMode inplace_mode) final {
    // Save the original tensor before base class mutation may remap it
    TensorVar orig_tensor = op->dest.as_or_throw<TensorVar>();
    TensorStore stmt = StmtExprMutator::Mutate_(op, inplace_mode)
                           .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                           .as_or_throw<TensorStore>();
    op = stmt.as<TensorStoreObj>();
    TVM_FFI_ICHECK(op != nullptr);

    // Look up using original tensor (before the inherited Var environment may have remapped it)
    auto it = match_tensors_.find(orig_tensor);
    if (it == match_tensors_.end()) {
      return stmt;
    } else {
      const TensorVar& tensor = (*it).first;
      const TensorRegion& source = (*it).second;

      auto* n = stmt.CopyOnWrite();
      n->indices = ConvertIndices(MatchTensorRegion(tensor, source), op->indices);
      n->dest = source->source.as_or_throw<tvm::tirx::TensorVar>();
      return stmt;
    }
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* op, InplaceMode inplace_mode) final {
    // Save the original tensor before base class mutation may remap it
    TensorVar orig_tensor = op->source.as_or_throw<tvm::tirx::TensorVar>();
    PrimExpr expr =
        StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<PrimExpr>(op));
    op = expr.as<TensorLoadObj>();
    TVM_FFI_ICHECK(op != nullptr);

    auto it = match_tensors_.find(orig_tensor);
    if (it == match_tensors_.end()) {
      return expr;
    } else {
      const TensorVar& tensor = (*it).first;
      const TensorRegion& source = (*it).second;
      ffi::Array<PrimExpr> indices = ConvertIndices(MatchTensorRegion(tensor, source), op->indices);
      return TensorLoad(source->source.as_or_throw<tvm::tirx::TensorVar>(), indices);
    }
  }

  TensorRegion VisitTensorRegion(const TensorRegion& tensor_region) {
    const TensorVar& tensor = tensor_region->source.as_or_throw<tvm::tirx::TensorVar>();
    auto it = match_tensors_.find(tensor);
    if (it == match_tensors_.end()) {
      return tensor_region;
    } else {
      const TensorRegion& source = (*it).second;
      ffi::Array<Range> region =
          ConvertRegion(MatchTensorRegion(tensor, source), tensor_region->region);
      return TensorRegion(source->source.as_or_throw<tvm::tirx::TensorVar>(), std::move(region),
                          TensorRegionType());
    }
  }

  void CheckAndUpdateVarMap(const MatchTensorRegion& match_tensor) {
    // Step.1. Check
    const TensorVar& tensor = match_tensor->tensor;
    const TensorRegion& source = VisitTensorRegion(match_tensor->source);
    const TensorVar& source_tensor = source->source.as_or_throw<tvm::tirx::TensorVar>();

    PrimExpr target_origin = tvm::tirx::MatchShapeStridePattern(tensor.ty()).value().get<2>();
    PrimExpr source_origin =
        tvm::tirx::MatchShapeStridePattern(source_tensor.ty()).value().get<2>();
    auto bind_origin = [&](PrimExpr value) {
      Bind(target_origin, value, tensor->name + ".elem_offset");
    };

    // Step.1.1. Check scope & dtype
    TVM_FFI_ICHECK_EQ(tensor.scope(), source_tensor.scope())
        << "MatchTensor " << tensor << " scope mismatch:" << tensor.scope() << "vs."
        << source_tensor.scope();
    TVM_FFI_ICHECK_EQ(tensor.dtype(), source_tensor.dtype())
        << "MatchTensor " << tensor << " data type mismatch:" << tensor.dtype() << "vs."
        << source_tensor.dtype();

    // Step.1.2. Check data alignment
    if (source_tensor.ty()->data_alignment % tensor.ty()->data_alignment != 0) {
      LOG(WARNING) << "Trying to bind tensor to another one with lower alignment requirement "
                   << " required alignment=" << tensor.ty()->data_alignment
                   << ", provided alignment=" << source_tensor.ty()->data_alignment;
    }
    if (IsZero(target_origin)) {
      TVM_FFI_ICHECK(IsZero(source_origin))
          << "Trying to bind a TensorVar with offset into one without offset "
          << " required elem_offset=" << target_origin
          << ", provided elem_offset=" << source_origin;
    }

    // Step.2. Update
    match_tensors_.Set(tensor, source);
    // Step.2.1. Update element offset
    // We use the ElemOffset method to avoid duplicating the index calculation.
    {
      ffi::Array<PrimExpr> indices;
      indices.reserve(source->region.size());
      for (const Range& range : source->region) {
        indices.push_back(range->begin);
      }

      ffi::Array<PrimExpr> tensor_start_indices = source_tensor.ty()->ElemOffset(indices);
      if (tensor_start_indices.size() == 1) {
        bind_origin(tensor_start_indices[0]);

      } else {
        // Non-zero elem_offset is ill-defined for non-flat memory.
        // If needed in the future, will require `ffi::Array<PrimExpr>
        // elem_offsets`, with one offset for each flattened index.
        bind_origin(IntImm(target_origin.ty(), 0));
      }
    }

    // Step 2.3. Check and update the alias's physical strides.
    TVM_FFI_ICHECK(source->region.size() >= tensor.shape().size());
    int offset = source->region.size() - tensor.shape().size();
    // An unspecified alias layout inherits the source region's physical strides.
    // Explicit layouts instead constrain (or bind) those strides.
    if (tensor.ty()->layout.has_value()) {
      auto target_strides = tvm::tirx::MatchShapeStridePattern(tensor.ty()).value().get<1>();
      auto source_strides = tvm::tirx::MatchShapeStridePattern(source_tensor.ty()).value().get<1>();
      for (size_t i = 0; i < tensor.shape().size(); ++i) {
        Bind(target_strides[i], source_strides[i + offset],
             tensor->name + ".strides_" + std::to_string(i));
      }
    }

    // Step 2.4. Check and update shape
    for (size_t i = 0; i < tensor.shape().size(); ++i) {
      const Range& range = source->region[i + offset];
      Bind(tensor.shape()[i], range->extent, tensor->name + ".shape_" + std::to_string(i));
    }
  }

  void Bind(const Expr& arg, Expr value, const std::string& arg_name = "argument") {
    auto arg_prim = arg.as<PrimExpr>();
    auto value_prim = value.as<PrimExpr>();
    if (arg_prim && value_prim && arg_prim.value().ty() != value_prim.value().ty()) {
      PrimType arg_ty = arg_prim.value().ty();
      PrimType value_ty = value_prim.value().ty();
      bool same_lanes = arg_ty.lanes() == value_ty.lanes();
      if (arg_ty.MatchesCode(DLDataTypeCode::kDLInt) &&
          value_ty.MatchesCode(DLDataTypeCode::kDLInt) && same_lanes) {
        value = cast(value_prim.value(), arg_ty);
      } else {
        TVM_FFI_ICHECK_EQ(arg_ty->dtype, value_ty->dtype)
            << "The data type mismatched: " << arg_ty->dtype << " vs. " << value_ty->dtype;
      }
    }
    // Handle recursive case
    auto f_substitute = [this](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
      if (auto repl = VarRemapGet(var); repl != nullptr) return repl;
      return ffi::Unchanged();
    };
    value = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(std::move(value), f_substitute)
                .as_or_throw<PrimExpr>();
    if (arg->IsInstance<VarObj>()) {
      Var v = arg.as_or_throw<Var>();
      auto replacement = VarRemapGet(v);
      if (replacement == nullptr) {
        VarRemapSet(v, value);
        if (auto prim_value = value.as<PrimExpr>()) {
          {
            auto context = analyzer_->GetConstraintContext();
            context->Append(sym::BindingConstraint(v, prim_value.value()));
            analyzer_->SyncConstraintContext(context);
          }
        }
      } else {
        AssertBinding(replacement.as_or_throw<Expr>(), value, arg_name);
      }
    } else {
      // Match a divisible linear offset through the same equation it declares.
      ffi::Array<PrimVar> unknown;
      for (const auto& var : UndefinedVars(arg)) {
        if (VarRemapGet(var) == nullptr) unknown.push_back(var.as_or_throw<PrimVar>());
      }
      if (unknown.size() == 1 && arg_prim) {
        auto coefficients = tirx::DetectLinearEquation(arg_prim.value(), unknown);
        if (coefficients.size() == 2 && coefficients[0].as<IntImmObj>() &&
            !IsZero(coefficients[0])) {
          PrimExpr delta = analyzer_->Simplify(value.as_or_throw<PrimExpr>() - coefficients[1]);
          TVM_FFI_ICHECK(analyzer_->CanProve(truncmod(delta, coefficients[0]) == 0))
              << "The tensor match offset must be divisible by " << coefficients[0];
          Bind(unknown[0], analyzer_->Simplify(truncdiv(delta, coefficients[0])), arg_name);
        }
      }
      AssertBinding(arg, value, arg_name);
    }
  }

  void AssertBinding(const Expr& lhs, const Expr& rhs, const std::string& arg_name = "argument") {
    if (auto lhs_prim = lhs.as<PrimExpr>()) {
      PrimExpr rhs_prim = rhs.as_or_throw<PrimExpr>();
      TVM_FFI_ICHECK(analyzer_->CanProve(lhs_prim.value() == rhs_prim))
          << "The tensor match constraint for " << arg_name << " unmet: " << lhs << "==" << rhs
          << ".";
    } else {
      TVM_FFI_ICHECK(ffi::StructuralEqual()(lhs, rhs))
          << "The tensor match constraint for " << arg_name << " unmet: " << lhs << "==" << rhs
          << ".";
    }
  }

  /*! \brief TensorVar region mapping. */
  ffi::Map<TensorVar, TensorRegion> match_tensors_;
  /*! \brief The analyzer */
  sym::Analyzer analyzer_;
};

namespace transform {

Pass LowerMatchTensor() {
  auto pass_func = [](Function f, IRModule m, PassContext ctx) {
    auto fptr = f.CopyOnWrite();
    fptr->body = ffi::make_object<MatchTensorLower>(f)
                     ->Mutate(fptr->body, InplaceMode::kAllow)
                     .ValueOrUnchanged(std::move(fptr->body));
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "s_tir.LowerMatchTensor");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.LowerMatchTensor", LowerMatchTensor);
}

}  // namespace transform

}  // namespace s_tir
}  // namespace tvm
