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

#include "ir_utils.h"

#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/scope_stack.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/transform.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op/memory.h>

#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace tvm {
namespace s_tir {
using namespace tirx;
using namespace tvm::prim;

namespace {
class SIRConvertSSA final : public StmtExprMutator {
 public:
  TVM_DEFINE_OBJECT_FUNCTOR_DEFAULT_CONSTRUCTOR(SIRConvertSSA, StmtExprMutator)
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  Function VisitFunction(Function func);
  IRModule VisitIRModule(IRModule mod);

  UnchangedOr<Stmt> Mutate_(const SBlockObj* op, InplaceMode mode) {
    SBlock block = ffi::GetRef<SBlock>(op);
    return scope_.WithNewScope([&]() -> Stmt {
      auto iter_vars = op->iter_vars.Map([&](IterVar iter) {
        Var var = DefineVar(iter->var);
        if (!var.same_as(iter->var)) iter.CopyOnWrite()->var = var.as_or_throw<PrimVar>();
        return iter;
      });
      if (!iter_vars.same_as(op->iter_vars)) block.CopyOnWrite()->iter_vars = iter_vars;
      return s_tir::StmtExprMutator::MutateBlock(this, block.get(),
                                                 block.unique() ? mode : InplaceMode::kDisallow)
          .ValueOrUnchanged(block);
    });
  }

  UnchangedOr<Stmt> Mutate_(const SBlockRealizeObj* op, InplaceMode mode) {
    return s_tir::StmtExprMutator::MutateBlockRealize(this, op, mode);
  }

 private:
  UnchangedOr<Expr> Mutate_(const VarObj* op, InplaceMode inplace_mode) final;
  Var DefineVar(Var var);
  ffi::Optional<Var> GetVarRemap(const Var& var);
  UnchangedOr<Stmt> Mutate_(const BindObj* op, InplaceMode inplace_mode) final;
  UnchangedOr<Stmt> Mutate_(const IfObj* op, InplaceMode inplace_mode) final;
  UnchangedOr<Stmt> Mutate_(const ForObj* op, InplaceMode inplace_mode) final;
  UnchangedOr<Stmt> Mutate_(const WhileObj* op, InplaceMode inplace_mode) final;
  UnchangedOr<Stmt> Mutate_(const RegionStmtObj* op, InplaceMode inplace_mode) final;
  UnchangedOr<PrimExpr> Mutate_(const prim::LetObj* op, InplaceMode inplace_mode) final;
  static Var MakeNewVar(const Var& old_var);
  void SetVarRemap(const Var& old_var, const Var& new_var);

  // Lexical visibility ends with each scope, but definitions stay unique across the module.
  ScopeStack<std::unordered_map<const VarObj*, Var>> scope_;
  std::unordered_set<const VarObj*> defined_;
};

Function SIRConvertSSA::VisitFunction(Function func) {
  return scope_.WithNewScope([&]() -> Function {
    std::unordered_set<const VarObj*> parameter_symbols;
    // Define explicit parameters before the symbolic values in their types.
    for (const Var& param : func->params) {
      DefineVar(param);
      parameter_symbols.insert(param.get());
    }
    for (const Var& param : func->params) {
      ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
          param->ty, [&](const Var& var) -> ffi::Expected<ffi::WalkResult> {
            if (!parameter_symbols.count(var.get())) {
              parameter_symbols.insert(var.get());
              DefineVar(var);
            }
            return ffi::WalkResult::Advance();
          });
    }
    auto params = func->params.Map([&](const Var& var) {
      Var mapped = GetVarRemap(var).value_or(var);
      Type type = Mutate(var->ty, InplaceMode::kDisallow)
                      .as_or_throw<UnchangedOr<Type>>()
                      .ValueOrUnchanged(var->ty);
      if (!type.same_as(mapped->ty)) {
        mapped = mapped.CopyWithType(type);
        SetVarRemap(var, mapped);
      }
      return mapped;
    });

    auto body_result = Mutate(func->body, InplaceMode::kDisallow);
    bool body_unchanged = body_result.UnchangedOrSameAs(func->body);
    auto body = std::move(body_result).ValueOrUnchanged(func->body);

    if (!params.same_as(func->params) || !body_unchanged) {
      auto* node = func.CopyOnWrite();
      node->params = std::move(params);
      node->body = std::move(body);
      node->ty = node->func_type_annotation();
    }
    return func;
  });
}

UnchangedOr<Expr> SIRConvertSSA::Mutate_(const VarObj* op, InplaceMode inplace_mode) {
  Var var = ffi::GetRef<Var>(op);
  if (def_region_kind() != kTVMFFIDefRegionKindNone) {
    if (def_region_kind() == kTVMFFIDefRegionKindPattern) {
      if (auto mapped = GetVarRemap(var)) return mapped.value();
    }
    return DefineVar(var);
  }
  Var mapped = GetVarRemap(var).value_or(var);
  if (!mapped.same_as(var)) return mapped;
  return ffi::Unchanged();
}

UnchangedOr<PrimExpr> SIRConvertSSA::Mutate_(const prim::LetObj* op, InplaceMode inplace_mode) {
  ffi::Any previous_remap = VarRemapGet(op->var);
  PrimExpr result = scope_.WithNewScope([&] {
    return StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<PrimExpr>(op));
  });
  VarRemapSet(op->var, previous_remap);
  return result;
}

Var SIRConvertSSA::DefineVar(Var var) {
  Type type = WithDefRegionKind(kTVMFFIDefRegionKindNone, [&] {
    return Mutate(var->ty, InplaceMode::kDisallow)
        .as_or_throw<UnchangedOr<Type>>()
        .ValueOrUnchanged(var->ty);
  });
  Var result = defined_.count(var.get()) ? MakeNewVar(var) : var;
  defined_.insert(var.get());
  if (!type.same_as(result->ty)) result = result.CopyWithType(type);
  SetVarRemap(var, result);
  return result;
}

ffi::Optional<Var> SIRConvertSSA::GetVarRemap(const Var& var) {
  for (auto it = scope_.rbegin(); it != scope_.rend(); ++it) {
    if (auto found = it->find(var.get()); found != it->end()) return found->second;
  }
  return std::nullopt;
}

UnchangedOr<Stmt> SIRConvertSSA::Mutate_(const BindObj* op, InplaceMode inplace_mode) {
  Var var = op->var;
  ffi::Any previous_remap = VarRemapGet(var);
  // The ordinary Bind path rewrites the RHS before defining the Var and propagates its type.
  Stmt stmt = StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
  Var bound_var = stmt.as<BindObj>()->var;
  if (!bound_var.same_as(GetVarRemap(var).value_or(var))) SetVarRemap(var, bound_var);
  VarRemapSet(var, previous_remap);
  return stmt;
}

UnchangedOr<Stmt> SIRConvertSSA::Mutate_(const IfObj* op, InplaceMode inplace_mode) {
  // Each branch gets its own scope so Bind remaps in one branch
  // do not leak into the other.
  auto condition_result = Mutate(op->condition, inplace_mode);
  bool condition_unchanged = condition_result.UnchangedOrSameAs(op->condition);
  PrimExpr condition = std::move(condition_result).ValueOrUnchanged(op->condition);
  Stmt then_case = scope_.WithNewScope([&]() -> Stmt {
    return Mutate(op->then_case, inplace_mode).ValueOrUnchanged(op->then_case);
  });
  ffi::Optional<SeqStmt> else_case;
  if (op->else_case) {
    else_case = scope_.WithNewScope([&]() -> SeqStmt {
      return Mutate(op->else_case.value(), inplace_mode).ValueOrUnchanged(op->else_case.value());
    });
  }
  if (condition_unchanged && then_case.same_as(op->then_case) && else_case.same_as(op->else_case)) {
    return ffi::Unchanged();
  }
  return If(condition, then_case, else_case);
}

UnchangedOr<Stmt> SIRConvertSSA::Mutate_(const ForObj* op, InplaceMode inplace_mode) {
  const Var& v = op->var;
  if (defined_.count(v.get())) {
    return scope_.WithNewScope([&]() -> Stmt {
      Var new_var = MakeNewVar(v);
      SetVarRemap(v, new_var);
      Stmt stmt =
          StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
      auto n = ffi::make_object<ForObj>(*stmt.as<ForObj>());
      n->var = new_var.as_or_throw<PrimVar>();
      return For(n);
    });
  } else {
    defined_.insert(v.get());
    return scope_.WithNewScope([&]() -> Stmt {
      SetVarRemap(v, v);
      return StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    });
  }
}

UnchangedOr<Stmt> SIRConvertSSA::Mutate_(const WhileObj* op, InplaceMode inplace_mode) {
  return scope_.WithNewScope([&]() -> Stmt {
    return StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
  });
}

UnchangedOr<Stmt> SIRConvertSSA::Mutate_(const RegionStmtObj* op, InplaceMode inplace_mode) {
  auto attrs = Mutate(op->attrs, InplaceMode::kDisallow)
                   .as_or_throw<UnchangedOr<DictAttrs>>()
                   .ValueOrUnchanged(op->attrs);
  auto args = Mutate(op->args, InplaceMode::kDisallow)
                  .as_or_throw<UnchangedOr<ffi::Array<Expr>>>()
                  .ValueOrUnchanged(op->args);
  ffi::Array<Var> params;
  Stmt body = scope_.WithNewScope([&]() -> Stmt {
    for (const Var& var : op->body_params) params.push_back(DefineVar(var));
    return Mutate(op->body, InplaceMode::kDisallow).ValueOrUnchanged(op->body);
  });
  ffi::Array<Var> results;
  for (const Var& var : op->result_vars) results.push_back(DefineVar(var));
  return RegionStmt(op->op, std::move(args), std::move(params), std::move(attrs), std::move(body),
                    std::move(results), op->loc);
}

Var SIRConvertSSA::MakeNewVar(const Var& old_var) {
  return Var(old_var->name, old_var->ty, old_var->loc);
}

void SIRConvertSSA::SetVarRemap(const Var& old_var, const Var& new_var) {
  scope_.Current().insert_or_assign(old_var.get(), new_var);
}

IRModule SIRConvertSSA::VisitIRModule(IRModule mod) {
  ffi::Map<GlobalVar, BaseFunc> functions;
  bool made_change = false;
  for (auto [gvar, base_func] : mod->functions) {
    if (auto* ptr = base_func.as<tirx::FunctionObj>()) {
      auto updated = VisitFunction(ffi::GetRef<tirx::Function>(ptr));
      if (!updated.same_as(base_func)) {
        made_change = true;
        base_func = updated;
      }
    }
    functions.Set(gvar, base_func);
  }
  if (made_change) {
    mod.CopyOnWrite()->functions = std::move(functions);
  }
  return mod;
}
}  // namespace

Stmt ConvertSSA(Stmt stmt) {
  return ffi::make_object<SIRConvertSSA>()
      ->Mutate(stmt, InplaceMode::kAllow)
      .ValueOrUnchanged(stmt);
}

IRModule ConvertSSA(IRModule mod) {
  return ffi::make_object<SIRConvertSSA>()->VisitIRModule(std::move(mod));
}

namespace transform {
Pass ConvertSSA() {
  auto pass_func = [](IRModule mod, PassContext ctx) { return s_tir::ConvertSSA(std::move(mod)); };
  return tvm::transform::CreateModulePass(pass_func, 0, "s_tir.ConvertSSA");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("s_tir.transform.ConvertSSA", ConvertSSA);
}
}  // namespace transform

ffi::Array<PrimExpr> ConvertIndices(const MatchTensorRegion& match_tensor,
                                    const ffi::Array<PrimExpr>& indices) {
  const TensorVar& target = match_tensor->tensor;
  const TensorRegion& source = match_tensor->source;
  TVM_FFI_ICHECK_EQ(indices.size(), target.shape().size());

  sym::Analyzer analyzer;
  ffi::Array<PrimExpr> result;
  result.reserve(source->region.size());
  size_t offset = source->region.size() - indices.size();
  for (size_t i = 0; i < offset; ++i) {
    const Range& range = source->region[i];
    TVM_FFI_ICHECK(analyzer->CanProve(range->extent == 1));
    result.push_back(range->begin);
  }
  for (size_t i = 0; i < indices.size(); ++i) {
    const Range& range = source->region[i + offset];
    const PrimExpr& index = indices[i];
    result.push_back(range->begin + index);
  }
  return result;
}

ffi::Array<Range> ConvertRegion(const MatchTensorRegion& match_tensor,
                                const ffi::Array<Range>& region) {
  const TensorVar& target = match_tensor->tensor;
  const TensorRegion& source = match_tensor->source;
  TVM_FFI_ICHECK_EQ(region.size(), target.shape().size());

  sym::Analyzer analyzer;
  ffi::Array<Range> result;
  result.reserve(source->region.size());
  size_t offset = source->region.size() - region.size();
  for (size_t i = 0; i < offset; ++i) {
    const Range& source_range = source->region[i];
    TVM_FFI_ICHECK(analyzer->CanProve(source_range->extent == 1));
    result.push_back(source_range);
  }
  for (size_t i = 0; i < region.size(); ++i) {
    const Range& source_range = source->region[i + offset];
    const Range& target_range = region[i];
    result.push_back(
        Range::FromMinExtent(source_range->begin + target_range->begin, target_range->extent));
  }
  return result;
}

/*! \brief Collect storage alignment information from annotations. */
class StorageAlignCollector : public StmtExprVisitor {
 public:
  ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
    if (value.as<ExprObj>()) return std::nullopt;
    return StmtExprVisitor::Visit(value);
  }

 private:
  friend std::unordered_map<Var, StorageAlignAnnotation> CollectStorageAlignAnnotation(
      const Stmt& body);

  /*! \brief SBlock: resolve each annotation's tensor index through the write regions. */
  ffi::Optional<VisitInterrupt> Visit_(const SBlockObj* op) final {
    auto it = op->annotations.find(attr::kTensorDimAlign.name);
    if (it != op->annotations.end()) {
      auto annotation = (*it).second.as_or_throw<StorageAlignAnnotation>();
      for (const auto& item : annotation) {
        storage_align_[op->writes[item.get<0>()]->source.as_or_throw<TensorVar>()].push_back(item);
      }
    }
    return StmtExprVisitor::Visit_(op);
  }

  /*! \brief AllocTensor: check for tensor_dim_align annotations. */
  ffi::Optional<VisitInterrupt> Visit_(const BindObj* op) final {
    if (const auto* call = op->value.as<CallObj>();
        call && call->op.same_as(tirx::alloc_tensor_op())) {
      return DispatchAllocTensor(op, call);
    }
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> DispatchAllocTensor(const BindObj* op, const CallObj* call) {
    DictAttrs annotations = call->attrs.as_or_throw<DictAttrs>();
    auto it = annotations->dict.find(attr::kTensorDimAlign.name);
    if (it != annotations->dict.end()) {
      auto storage_align_annotation = (*it).second.as_or_throw<StorageAlignAnnotation>();
      for (const auto& storage_align_tuple : storage_align_annotation) {
        int tensor_index = storage_align_tuple.get<0>();
        // the first tensor idx info is meaningless for alloc
        // stmt and should set as negative intentionally.
        TVM_FFI_ICHECK_EQ(tensor_index, -1);
        storage_align_[op->var.as_or_throw<TensorVar>()].push_back(storage_align_tuple);
      }
    }
    return StmtExprVisitor::Visit_(op);
  }

  /*! \brief The map from tensor var to its storage alignment information. */
  std::unordered_map<Var, StorageAlignAnnotation> storage_align_;
};

std::unordered_map<Var, StorageAlignAnnotation> CollectStorageAlignAnnotation(const Stmt& body) {
  auto collector = ffi::make_object<StorageAlignCollector>();
  collector->Visit(body);
  return std::move(collector->storage_align_);
}

}  // namespace s_tir
}  // namespace tvm
