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

#include "create_function.h"

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/ir/function.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/unique_name_supply.h>
#include <tvm/s_tir/function.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/sym/analyzer.h>
#include <tvm/te/operation.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/type.h>
#include <tvm/topi/tags.h>

#include <algorithm>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../s_tir/ir/data_type_rewriter.h"
#include "graph.h"

namespace tvm {
namespace tirx {
using namespace tvm::prim;

namespace {

// Only SBlock.iter_vars entries have the retained axis-metadata role. Walk their
// fields explicitly so opaque values in domains, types or annotations remain errors.
class OpaqueArtifactVerifier : public ObjectVisitor {
 public:
  ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) final {
    if (const auto* block = value.as<s_tir::SBlockObj>()) {
      for (const auto& axis : block->iter_vars) {
        Visit(axis->dom);
        Visit(axis->var);
        Visit(axis->ty);
        Visit(axis->loc);
      }
      Visit(block->reads);
      Visit(block->writes);
      Visit(block->body);
      Visit(block->init);
      Visit(block->alloc_tensors);
      Visit(block->match_tensors);
      Visit(block->annotations);
      Visit(block->loc);
      return std::nullopt;
    }
    if (const auto* expr = value.as<OpaqueExprObj>()) {
      TVM_FFI_THROW(InternalError)
          << "CreateFunction produced construction-only opaque artifact " << expr->GetTypeKey();
    }
    if (value.as<OpaqueTypeObj>()) {
      TVM_FFI_THROW(InternalError)
          << "CreateFunction produced construction-only opaque artifact ir.OpaqueType";
    }
    return ObjectVisitor::Visit(value);
  }
};

void VerifyNoOpaqueArtifacts(const Function& func) {
  ffi::make_object<OpaqueArtifactVerifier>()->Visit(func);
}

}  // namespace

/*! \brief The helper mutator that transforms Tensor-callee Calls to TensorLoad. */
class TensorLoadToTensorTransformer : public s_tir::StmtExprMutator {
 public:
  using s_tir::StmtExprMutator::Mutate_;

  explicit TensorLoadToTensorTransformer(
      const std::unordered_map<te::Tensor, TensorVar>& tensor2tensors)
      : s_tir::StmtExprMutator(VTableInstance()), tensor2tensors_(tensor2tensors) {}

  TensorVar NormalizeTensorType(const TensorVar& tensor) {
    auto type = Mutate(tensor->ty).ValueOrUnchanged(tensor->ty).cast<TensorType>();
    if (type.same_as(tensor->ty)) return tensor;
    return TensorVar(tensor->name, type, tensor->loc);
  }

  s_tir::IterVar NormalizeAxisDomain(const s_tir::IterVar& axis) {
    if (!axis->dom.has_value()) return axis;
    auto domain = Mutate(axis->dom).ValueOrUnchanged(axis->dom).cast<Range>();
    if (axis->dom.same_as(domain)) return axis;
    auto copy = ffi::make_object<s_tir::IterVarObj>(*axis.get());
    copy->dom = std::move(domain);
    return s_tir::IterVar(std::move(copy));
  }

  UnchangedOr<Expr> Mutate_(const s_tir::IterVarObj* op, InplaceMode) { return op->var; }

  UnchangedOr<Expr> Mutate_(const te::ReduceObj* op, InplaceMode inplace_mode) {
    return Mutate_(static_cast<const OpaqueExprObj*>(op), inplace_mode);
  }

  UnchangedOr<Expr> Mutate_(const OpaqueExprObj* op, InplaceMode inplace_mode) final {
    const auto* reduce =
        op->IsInstance<te::ReduceObj>() ? static_cast<const te::ReduceObj*>(op) : nullptr;
    if (reduce == nullptr) {
      return s_tir::StmtExprMutator::Mutate_(op, inplace_mode);
    }

    auto axis =
        reduce->axis.Map([this](const s_tir::IterVar& axis) { return NormalizeAxisDomain(axis); });
    bool axis_unchanged = axis.same_as(reduce->axis);

    auto source_update =
        Mutate(reduce->source, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    auto init_update =
        Mutate(reduce->init, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    bool source_unchanged = source_update.UnchangedOrSameAs(reduce->source);
    bool init_unchanged = init_update.UnchangedOrSameAs(reduce->init);
    ffi::Array<PrimExpr> source = std::move(source_update).ValueOrUnchanged(reduce->source);
    ffi::Array<PrimExpr> init = std::move(init_update).ValueOrUnchanged(reduce->init);
    auto condition_update = this->Mutate(reduce->condition, inplace_mode);
    bool condition_unchanged = condition_update.UnchangedOrSameAs(reduce->condition);
    PrimExpr condition = std::move(condition_update).ValueOrUnchanged(reduce->condition);

    if (axis_unchanged && source_unchanged && init_unchanged && condition_unchanged) {
      return ffi::Unchanged();
    }
    return te::Reduce(reduce->combiner, source, axis, condition, reduce->value_index, init,
                      reduce->loc);
  }

  UnchangedOr<Expr> Mutate_(const CallObj* op, InplaceMode inplace_mode) final {
    Call call = s_tir::StmtExprMutator::Mutate_(op, inplace_mode)
                    .ValueOrUnchanged(ffi::GetRef<Expr>(op))
                    .as_or_throw<Call>();
    if (!te::IsTensorLoad(call)) {
      return call;
    }
    te::Tensor tensor = te::GetTensorFromLoad(call);
    auto it = tensor2tensors_.find(tensor);
    TVM_FFI_ICHECK(it != tensor2tensors_.end()) << "IndexError: Cannot find the tensor " << tensor;
    const TensorVar& tir_tensor = it->second;
    return TensorLoad(tir_tensor, te::GetTensorLoadIndices(call), call->loc);
  }

 private:
  static const VTable* VTableInstance() {
    static const VTable table = [] {
      VTable table;
      s_tir::StmtExprMutator::InitVTable(&table);
      SetDispatch<TensorLoadToTensorTransformer, s_tir::IterVarObj>(&table);
      SetDispatch<TensorLoadToTensorTransformer, te::ReduceObj>(&table);
      table.Finalize();
      return table;
    }();
    return &table;
  }
  /*! \brief The Map from Operations to tensors */
  const std::unordered_map<te::Tensor, TensorVar>& tensor2tensors_;
};

/*! \brief The helper mutator to rewrite tensor and tensor var accessed by block body */
class TensorSubstituter : public s_tir::StmtExprMutator {
 public:
  explicit TensorSubstituter(const std::unordered_map<const VarObj*, Expr>& var_map,
                             const std::unordered_map<const VarObj*, TensorVar>& tensor_map) {
    for (const auto& [source, target] : tensor_map) {
      VarRemapSet(ffi::AnyView(source), target);
    }
    for (const auto& [source, target] : var_map) {
      VarRemapSet(ffi::AnyView(source), target);
    }
  }
};

/*! \brief Helper data structure to store information. */
struct CreateFuncInfo {
  /*! \brief The Tensor arg_list. */
  ffi::Array<te::Tensor> arg_list;
  /*! \brief The map from each Tensor to its corresponding tensor. */
  std::unordered_map<te::Tensor, TensorVar> tensor2tensors;
  /*! \brief The transformer from Tensor-callee Calls to TensorLoad. */
  ffi::ObjectPtr<TensorLoadToTensorTransformer> transformer;
  /*! \brief The tensors should be allocated at function root. */
  ffi::Array<TensorVar> root_alloc;
  /*! \brief The unique name supply to make block name unique. */
  UniqueNameSupply name_supply;

  ffi::String FreshName(ffi::String base_name) { return name_supply->FreshName(base_name); }

  explicit CreateFuncInfo(ffi::Array<te::Tensor> arg_list)
      : arg_list(std::move(arg_list)),
        transformer(ffi::make_object<TensorLoadToTensorTransformer>(tensor2tensors)) {}

  bool IsArg(const te::Tensor& tensor) const {
    return std::any_of(arg_list.begin(), arg_list.end(),
                       [&tensor](const te::Tensor& arg) { return tensor == arg; });
  }
};

class LayoutFreePlaceholdersNormalizer : public s_tir::StmtExprMutator {
 public:
  using s_tir::StmtExprMutator::Mutate;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) final {
    if (value.as<ExprObj>()) return ffi::Unchanged();
    return s_tir::StmtExprMutator::Mutate(value, inplace_mode);
  }

  Function Process(Function func) {
    for (int i = 0, n = func->params.size(); i < n; ++i) {
      if (auto tensor = func->params[i].as<TensorVar>()) {
        tensor2index_[tensor.value()] = i;
      }
    }
    FunctionObj* f = func.CopyOnWrite();
    f->body = Mutate(f->body, InplaceMode::kDisallow).ValueOrUnchanged(f->body);
    if (this->layout_free_tensor_indices_.empty()) {
      return func;
    }
    ffi::Array<int64_t> indices;
    indices.reserve(this->layout_free_tensor_indices_.size());
    for (int i : this->layout_free_tensor_indices_) {
      indices.push_back(i);
    }
    return WithAttr(std::move(func), tvm::s_tir::attr::kLayoutFreeTensors.name, indices);
  }

  UnchangedOr<Stmt> Mutate_(const s_tir::SBlockObj* _block, InplaceMode inplace_mode) final {
    s_tir::SBlock block = s_tir::StmtExprMutator::Mutate_(_block, inplace_mode)
                              .ValueOrUnchanged(ffi::GetRef<Stmt>(_block))
                              .as_or_throw<s_tir::SBlock>();
    s_tir::SBlockObj* n = block.CopyOnWrite();
    if (auto opt_ann = n->annotations.Get(topi_attr)) {
      ffi::Array<TensorVar> new_tensors;
      for (TensorVar tensor : opt_ann.value().as_or_throw<ffi::Array<TensorVar>>()) {
        auto it = tensor2index_.find(tensor);
        if (it != tensor2index_.end()) {
          layout_free_tensor_indices_.insert(it->second);
        } else {
          new_tensors.push_back(tensor);
        }
      }
      if (new_tensors.empty()) {
        n->annotations.erase(topi_attr);
      } else {
        n->annotations.Set(topi_attr, new_tensors);
      }
    }
    for (const ffi::String& attr : this->blocklist) {
      auto it = n->annotations.find(attr);
      if (it != n->annotations.end()) {
        n->annotations.erase(attr);
      }
    }
    return block;
  }

  std::unordered_map<tirx::TensorVar, int, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> tensor2index_;
  std::set<int> layout_free_tensor_indices_;
  ffi::String topi_attr = tvm::topi::attr::kLayoutFreePlaceholders;
  std::vector<ffi::String> blocklist = {tvm::topi::attr::kConstMatrix,
                                        tvm::topi::attr::kAutoSchedulerSimplifyConstTensorIndices,
                                        tvm::topi::attr::kWorkload};
};

/**!
 * \brief The iter levels specify nested structure wrt iteration domain dependencies.
 * (1) Each iter should reside in exactly one level.
 * (2) The domain of low level iter should be either free or ony depend on iters in high level.
 **/
using NestedIterLevels = std::vector<std::vector<s_tir::IterVar>>;

NestedIterLevels GenerateNestedIterLevels(const ffi::Array<s_tir::IterVar>& axes,
                                          sym::AnalyzerObj* analyzer) {
  int global_max_depth = 0;
  std::unordered_map<Var, int> depth;
  std::unordered_map<Var, s_tir::IterVar> var2iter;
  for (const auto& axis : axes) {
    var2iter.emplace(axis->var, axis);
  }

  std::function<int(const s_tir::IterVar&)> traverse = [&](const s_tir::IterVar& axis) -> int {
    auto depth_it = depth.find(axis->var);
    if (depth_it != depth.end()) {  // cache
      return depth_it->second;
    }
    std::vector<Var> dep_vars;
    for (const Var& v : UndefinedVars(analyzer->Simplify(axis->dom.value()->begin))) {
      dep_vars.push_back(v);
    }
    for (const Var& v : UndefinedVars(analyzer->Simplify(axis->dom.value()->extent))) {
      dep_vars.push_back(v);
    }
    int cur_depth = 0;
    for (const Var& v : dep_vars) {
      auto it = var2iter.find(v);
      if (it == var2iter.end()) {
        // not axis var dependency, maybe a symbolic shape var or others.
        continue;
      }
      int depth = traverse(it->second);
      cur_depth = std::max(cur_depth, depth + 1);
    }
    depth.emplace_hint(depth_it, axis->var, cur_depth);
    global_max_depth = std::max(global_max_depth, cur_depth);
    return cur_depth;
  };

  for (const auto& axis : axes) {
    traverse(axis);
  }
  NestedIterLevels levels;
  levels.resize(global_max_depth + 1);
  for (const auto& axis : axes) {
    const Var& var = axis->var;
    levels[depth[var]].push_back(axis);
  }
  return levels;
}

/*!
 * \brief Generate output tensors from compute op's output tensors, and bind to context func info.
 * \param compute_op The target compute op.
 * \param info Generation context info.
 * \returns The output tensor objects, ordered by compute op's outputs.
 **/
ffi::Array<TensorVar> GenerateOutputTensors(const te::ComputeOp& compute_op, CreateFuncInfo* info) {
  // Step 1. Collect output tensors in TE operation.
  ffi::Array<te::Tensor> tensors;
  if (compute_op->body[0]->IsInstance<te::ReduceObj>()) {
    auto f_reducer_equal = [](const te::ReduceObj* a, const te::ReduceObj* b) -> bool {
      ffi::StructuralEqual eq;
      return eq(a->combiner, b->combiner) &&    //
             eq(a->source, b->source) &&        //
             eq(a->axis, b->axis) &&            //
             eq(a->condition, b->condition) &&  //
             eq(a->init, b->init);
    };
    PrimExpr expr_body = compute_op->body[0];
    tensors.push_back(compute_op.output(0));
    const te::ReduceObj* reduce = expr_body.as<te::ReduceObj>();
    // specially handle reduction inline for multiplre reductions.
    for (size_t k = 1; k < compute_op->body.size(); ++k) {
      const te::ReduceObj* reduce_ = compute_op->body[k].as<te::ReduceObj>();
      TVM_FFI_ICHECK(reduce_);
      TVM_FFI_ICHECK(f_reducer_equal(reduce_, reduce))
          << "The Reduce inputs of ComputeOp should have the same attribute except value_index, "
          << "but the first argument has body " << ffi::GetRef<PrimExpr>(reduce_) << ", while the "
          << k << "-th argument has body " << ffi::GetRef<PrimExpr>(reduce);
      tensors.push_back(compute_op.output(k));
    }
  } else {
    for (size_t k = 0; k < compute_op->body.size(); ++k) {
      tensors.push_back(compute_op.output(k));
    }
  }
  // Step 2. Prepare tensors for compute outputs
  //  - Declare tensors
  //  - Update `op2tensors`
  //  - Add the non-argument tensors to `alloc_tensor` of the root block
  ffi::Array<TensorVar> tir_tensors;
  for (const te::Tensor& tensor : tensors) {
    TensorVar tir_tensor =
        TensorVar(tensor->GetNameHint(), TensorType(info->transformer->Mutate(tensor->shape)
                                                        .ValueOrUnchanged(tensor->shape)
                                                        .cast<ffi::Array<PrimExpr>>(),
                                                    tensor->dtype, runtime::MemoryScope::Global()));
    info->tensor2tensors.insert_or_assign(tensor, tir_tensor);
    tir_tensors.push_back(tir_tensor);
    if (!info->IsArg(tensor)) {
      info->root_alloc.push_back(info->tensor2tensors.at(tensor));
    }
  }
  return tir_tensors;
}

/*!
 * \brief Generate block annotation dict from compute op attrs.
 * \param compute_op The target compute op.
 * \param info Generation context info.
 * \returns The block annotation dict.
 **/
ffi::Map<ffi::String, ffi::Any> GenerateBlockAnnotations(const te::ComputeOp& compute_op,
                                                         CreateFuncInfo* info) {
  ffi::Map<ffi::String, ffi::Any> annotations;
  auto mutate_attr = [&info](const ffi::Any& value) -> ffi::Any {
    if (auto tensor_value = value.try_cast<te::Tensor>()) {
      return info->tensor2tensors.at(tensor_value.value());
    } else {
      return info->transformer->Mutate(value).ValueOrUnchanged(value);
    }
  };
  for (const auto& pair : compute_op->attrs) {
    const ffi::String& key = pair.first;
    const Any& value = pair.second;
    // TensorIR will not allow Tensor data structure
    if (value.as<ffi::ArrayObj>()) {
      const auto array_value = value.as_or_throw<ffi::Array<ffi::Any>>();
      annotations.Set(key, array_value.Map(mutate_attr));
    } else {
      annotations.Set(key, mutate_attr(value));
    }
  }
  // Set script_parsing_detect_access
  annotations.Set(tvm::s_tir::attr::kScriptParsingDetectAccess.name, 3);
  return annotations;
}

/*!
 * \brief Generate init stmt for reduction.
 * \param indices Target store indices for the block.
 * \param tensors Target store tensors for the block.
 * \param reduce Reduce description node.
 * \param var_map Var re-mapping for TE compute axes.
 * \param info Generation context info.
 * \returns Init stmt.
 **/
SeqStmt GenerateInitStmt(const ffi::Array<PrimExpr>& indices, const ffi::Array<TensorVar>& tensors,
                         const te::ReduceObj* reduce, const ffi::Map<Var, PrimExpr>& var_map,
                         CreateFuncInfo* info) {
  auto f_substitute = [&var_map](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (auto repl = var_map.Get(var)) return ffi::Any(*std::move(repl));
    return ffi::Unchanged();
  };
  // helper to transform the expr and remap iters to the block domain
  auto f_transform_and_remap = [&](const PrimExpr& e) {
    return ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(
               info->transformer->Mutate(e).ValueOrUnchanged(e), f_substitute)
        .as_or_throw<PrimExpr>();
  };
  int n_tensors = tensors.size();
  ffi::Array<Stmt> init_stmts;
  init_stmts.reserve(n_tensors);
  for (int i = 0; i < n_tensors; ++i) {
    const TensorVar& tensor = tensors[i];
    PrimExpr identity = f_transform_and_remap(reduce->combiner->identity_element[i]);
    init_stmts.push_back(TensorStore(tensor, indices, identity));
  }
  return SeqStmt(init_stmts);
}

/*!
 * \brief Generate body execution stmt.
 * \param indices Target store indices for the block.
 * \param tensors Target store tensors for the block.
 * \param var_map Var re-mapping for TE compute axes.
 * \param expr_body Target computation expression.
 * \param info Generation context info.
 * \param analyzer Arithmetic analyzer in context.
 * \returns Init stmt.
 **/
Stmt GenerateBodyStmt(const ffi::Array<PrimExpr>& indices, const ffi::Array<TensorVar>& tensors,
                      const ffi::Map<Var, PrimExpr>& var_map, PrimExpr expr_body,
                      CreateFuncInfo* info, sym::AnalyzerObj* analyzer) {
  auto f_substitute = [&var_map](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (auto repl = var_map.Get(var)) return ffi::Any(*std::move(repl));
    return ffi::Unchanged();
  };
  // helper to transform the expr and remap iters to the block domain
  auto f_transform_and_remap = [&](const PrimExpr& e) {
    return ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(
               info->transformer->Mutate(e).ValueOrUnchanged(e), f_substitute)
        .as_or_throw<PrimExpr>();
  };
  if (const auto* reduce = expr_body.as<te::ReduceObj>()) {
    // Case 1. Reduce compute
    int n_tensors = tensors.size();

    ffi::Array<PrimExpr> lhs;
    ffi::Array<PrimExpr> rhs;
    lhs.reserve(n_tensors);
    rhs.reserve(n_tensors);

    // Make the LHS operands and RHS operands:
    //  - A LHS operand is the tensor storing the reduction result, with corresponding indices.
    //  - A RHS operand is the value to be reduced.
    for (int i = 0; i < n_tensors; ++i) {
      const PrimExpr& left = TensorLoad(tensors[i], indices);
      const PrimExpr& right = analyzer->Simplify(f_transform_and_remap(reduce->source[i]));
      lhs.push_back(left);
      rhs.push_back(right);
      TVM_FFI_ICHECK_EQ(left.ty()->dtype, right.ty()->dtype);
    }

    ffi::Array<Var> temp_vars;
    ffi::Array<Stmt> body_stmts;
    temp_vars.reserve(n_tensors);
    body_stmts.reserve(n_tensors);

    // - When there is only one tensor, we directly create a TensorStore which stores "combiner(lhs,
    //   rhs)" into the target tensor position.
    // - In case there are multiple tensors, to avoid incorrect results, we create some intermediate
    //   variables and use Bind nodes to bind the variables with "combiner(lhs, rhs)". After that,
    //   we then store the value of the variables into the target tensor positions.
    for (int i = 0; i < n_tensors; ++i) {
      const TensorVar& tensor = tensors[i];
      PrimExpr value = [&]() -> PrimExpr {
        if (n_tensors > 1) {
          temp_vars.push_back(Var("v_" + tensor->name, lhs[i].ty()));
          return temp_vars.back().as_or_throw<PrimExpr>();
        }
        PrimExpr combined = reduce->combiner.get()->operator()(lhs, rhs)[i];
        return f_transform_and_remap(combined);
      }();
      body_stmts.push_back(TensorStore(tensor, indices, value));
    }
    Stmt body = SeqStmt(body_stmts);
    if (n_tensors > 1) {
      // When there are multiple tensors, we wrap the body with Bind stmts.
      ffi::Array<Stmt> bind_stmts;
      for (int i = 0; i < n_tensors; ++i) {
        PrimExpr value = f_transform_and_remap(reduce->combiner.get()->operator()(lhs, rhs)[i]);
        bind_stmts.push_back(Bind(temp_vars[i], std::move(value)));
      }
      bind_stmts.push_back(body);
      body = SeqStmt(bind_stmts);
    }
    return body;
  } else {
    // Case 2. Data parallel compute
    TVM_FFI_ICHECK_EQ(tensors.size(), 1);
    const PrimExpr& compute_body = f_transform_and_remap(expr_body);
    return TensorStore(tensors[0], indices, analyzer->Simplify(compute_body));
  }
}

/*! \brief Record loops, block vars and binding in the single level scope. */
struct NestedScopeInfo {
  // loop var and range in the scope.
  std::vector<std::pair<Var, Range>> loop_vars;
  // block iters for current level's block.
  ffi::Array<s_tir::IterVar> block_iters;
  // block bindings for current level's block.
  ffi::Array<PrimExpr> bindings;
  // store indices for current level's block.
  ffi::Array<PrimExpr> store_indices;
  // mapping from original TE compute axes to new block vars.
  ffi::Map<Var, PrimExpr> axes_remap;

  // helper to add new block var
  void AddBlockIter(const ffi::Optional<s_tir::IterVar>& origin_axis, const s_tir::IterVar& iter,
                    const PrimExpr& value) {
    block_iters.push_back(iter);
    bindings.push_back(value);
    if (origin_axis.has_value()) {
      if (iter->iter_type != s_tir::IterVarType::kCommReduce) {
        store_indices.push_back(iter->var);
      }
      axes_remap.Set(origin_axis.value()->var, iter->var);
    }
  }

  // helper to renew leaf block var defs to ensure SSA.
  void Renew(const ffi::Array<s_tir::IterVar>& origin_axes) {
    block_iters.MutateByApply([](const s_tir::IterVar& itervar) {
      auto n = ffi::make_object<s_tir::IterVarObj>(*itervar.get());
      n->var = n->var.CopyWithSuffix("");
      return s_tir::IterVar(n);
    });
    for (size_t i = 0; i < origin_axes.size(); ++i) {
      Var block_var = block_iters[i]->var;
      if (origin_axes[i]->iter_type != s_tir::IterVarType::kCommReduce) {
        store_indices.Set(i, block_var.as_or_throw<PrimExpr>());
      }
      axes_remap.Set(origin_axes[i]->var, block_var.as_or_throw<PrimExpr>());
    }
  }
};

Stmt GenerateStmtFromCompute(const te::ComputeOp& compute_op, CreateFuncInfo* info,
                             sym::AnalyzerObj* analyzer) {
  // Step 1. Collect all iter axes in original TE compute op
  ffi::Array<s_tir::IterVar> axes = compute_op->axis;
  axes.insert(axes.end(), compute_op->reduce_axis.begin(), compute_op->reduce_axis.end());

  // Step 2. Prepare nested iteration scopes.
  // For each axis, we generate loop and the first block binding at the level it belongs to.
  // In lower levels, we just create new block var and bind it to the previous level block var.
  axes = axes.Map(
      [&](const s_tir::IterVar& axis) { return info->transformer->NormalizeAxisDomain(axis); });
  auto axes_levels = GenerateNestedIterLevels(axes, analyzer);
  TVM_FFI_ICHECK(!axes_levels.empty());
  std::vector<NestedScopeInfo> scopes;
  scopes.reserve(axes_levels.size());
  // Initialize a nested reduction at its outermost reduction level.
  size_t reduction_init_scope = axes_levels.size() - 1;
  std::unordered_set<Var> defined_axes;
  for (size_t i = 0; i < axes_levels.size(); ++i) {
    NestedScopeInfo cur_scope;
    for (size_t j = 0; j < axes.size(); ++j) {
      const s_tir::IterVar& axis = axes[j];
      PrimType index_type = PrimType::Int(
          std::max(axis->dom.value()->begin.ty().bits(), axis->dom.value()->extent.ty().bits()));
      bool first_times_define =
          std::any_of(axes_levels[i].begin(), axes_levels[i].end(),
                      [&](const s_tir::IterVar& candidate) { return candidate.same_as(axis); });
      if (first_times_define) {
        if (axis->iter_type == s_tir::IterVarType::kCommReduce) {
          reduction_init_scope = std::min(reduction_init_scope, i);
        }
        Var loop_var = Var(axis->var->name, index_type);
        Var block_var("v_" + axis->var->name, index_type);
        PrimExpr min = axis->dom.value()->begin;
        PrimExpr extent = axis->dom.value()->extent;
        if (i > 0) {
          const auto& scope_repl = scopes[i - 1].axes_remap;
          auto f_substitute =
              [&scope_repl](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
            if (auto repl = scope_repl.Get(var)) return ffi::Any(*std::move(repl));
            return ffi::Unchanged();
          };
          min = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(min, f_substitute)
                    .as_or_throw<PrimExpr>();
          extent = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(extent, f_substitute)
                       .as_or_throw<PrimExpr>();
        }
        Range dom = Range::FromMinExtent(analyzer->Simplify(min), analyzer->Simplify(extent));
        s_tir::IterVar new_block_iter(dom, block_var.as_or_throw<PrimVar>(), axis->iter_type,
                                      axis->thread_tag, axis->loc);
        cur_scope.loop_vars.emplace_back(loop_var, dom);
        cur_scope.AddBlockIter(axis, new_block_iter, loop_var.as_or_throw<PrimExpr>());
        defined_axes.insert(axis->var);
      } else if (defined_axes.count(axis->var)) {
        TVM_FFI_ICHECK_GT(i, 0);
        TVM_FFI_ICHECK(scopes[i - 1].axes_remap.count(axis->var));
        PrimExpr prev_binding = scopes[i - 1].axes_remap.at(axis->var);
        Var block_var("v_" + axis->var->name, index_type);
        Range dom = Range::FromMinExtent(prev_binding, MakeConst(index_type, 1));
        s_tir::IterVar new_block_iter(dom, block_var.as_or_throw<PrimVar>(), axis->iter_type,
                                      axis->thread_tag, axis->loc);
        cur_scope.AddBlockIter(axis, new_block_iter, prev_binding);
      }
    }
    if (i == axes_levels.size() - 1 && cur_scope.block_iters.empty()) {
      // for the leaf scope, we ensure at least one block var exists
      s_tir::IterVar dummy(Range::FromMinExtent(0, 1), PrimVar("vi", PrimType::Int(32)),
                           s_tir::IterVarType::kDataPar);
      cur_scope.AddBlockIter(std::nullopt, dummy, 0);
    }
    scopes.push_back(cur_scope);
  }

  // Step 3. Generate output tensors for each output tensor
  ffi::Array<TensorVar> tensors = GenerateOutputTensors(compute_op, info);

  // Step 4. Generate leaf block stmts.
  ffi::Array<Stmt> seq_stmt;
  auto leaf = scopes.back();
  ffi::Map<ffi::String, ffi::Any> annotations = GenerateBlockAnnotations(compute_op, info);
  const te::ReduceObj* reduce = compute_op->body[0].as<te::ReduceObj>();

  if (reduce) {
    PrimExpr expr_body = compute_op->body[0];
    ffi::Optional<SeqStmt> init{std::nullopt};
    if (reduction_init_scope == scopes.size() - 1) {
      init = GenerateInitStmt(leaf.store_indices, tensors, reduce, leaf.axes_remap, info);
    }
    Stmt body =
        GenerateBodyStmt(leaf.store_indices, tensors, leaf.axes_remap, expr_body, info, analyzer);
    seq_stmt.push_back(
        s_tir::SBlockRealize(/*iter_values=*/leaf.bindings,
                             /*predicate=*/IntImm::Bool(true),
                             /*block=*/
                             s_tir::SBlock(/*iter_vars=*/leaf.block_iters,
                                           /*reads=*/{},
                                           /*writes=*/{},
                                           /*name_hint=*/info->FreshName(compute_op->name),
                                           /*body=*/body,
                                           /*init=*/init,
                                           /*alloc_tensors=*/{},
                                           /*match_tensors=*/{},
                                           /*annotations=*/annotations)));

  } else {
    for (int i = 0; i < compute_op->num_outputs(); ++i) {
      if (i > 0) {
        // Renew block var defs to ensure SSA
        leaf.Renew(axes);
      }
      PrimExpr expr_body = compute_op->body[i];
      Stmt body = GenerateBodyStmt(leaf.store_indices, {tensors[i]}, leaf.axes_remap, expr_body,
                                   info, analyzer);
      seq_stmt.push_back(
          s_tir::SBlockRealize(/*iter_values=*/leaf.bindings,
                               /*predicate=*/IntImm::Bool(true),
                               /*block=*/
                               s_tir::SBlock(/*iter_vars=*/leaf.block_iters,
                                             /*reads=*/{},
                                             /*writes=*/{},
                                             /*name_hint=*/info->FreshName(tensors[i]->name),
                                             /*body=*/body,
                                             /*init=*/std::nullopt,
                                             /*alloc_tensors=*/{},
                                             /*match_tensors=*/{},
                                             /*annotations=*/annotations)));
    }
  }
  Stmt body = SeqStmt(seq_stmt);

  // Step 4. Generate nested parent scopes.
  for (size_t i = scopes.size(); i > 0; --i) {
    const auto& cur = scopes[i - 1];
    if (i < scopes.size()) {
      auto block_name = info->FreshName(compute_op->name + "_l" + std::to_string(i));
      const auto& block_iters = cur.block_iters;

      ffi::Optional<SeqStmt> init{std::nullopt};
      if (reduce && i - 1 == reduction_init_scope) {
        init = GenerateInitStmt(cur.store_indices, tensors, reduce, cur.axes_remap, info);
      }

      // wrap nested block
      body = s_tir::SBlockRealize(/*iter_values=*/cur.bindings,
                                  /*predicate=*/IntImm::Bool(true),
                                  /*block=*/
                                  s_tir::SBlock(/*iter_vars=*/block_iters,
                                                /*reads=*/{},
                                                /*writes=*/{},
                                                /*name_hint=*/block_name,
                                                /*body=*/body,
                                                /*init=*/init,
                                                /*alloc_tensors=*/{},
                                                /*match_tensors=*/{},
                                                /*annotations=*/annotations));
    }
    for (size_t j = cur.loop_vars.size(); j > 0; --j) {
      const auto& [loop_var, dom] = cur.loop_vars[j - 1];
      body = For(loop_var.as_or_throw<PrimVar>(), dom->begin, dom->begin + dom->extent,
                 IntImm(loop_var.as_or_throw<PrimVar>().ty(), 1), ForKind::kDefault, body);
    }
  }
  return body;
}

Stmt GenerateStmtFromExternOp(const te::ExternOp& extern_op, CreateFuncInfo* info,
                              sym::AnalyzerObj* analyzer) {
  // Step 1. Check all inputs are visited before and update var_map.
  std::unordered_map<const VarObj*, Expr> var_map;
  std::unordered_map<const VarObj*, TensorVar> input_tensor_map;
  TVM_FFI_ICHECK_EQ(extern_op->inputs.size(), extern_op->input_placeholders.size());
  for (size_t i = 0; i < extern_op->inputs.size(); ++i) {
    const TensorVar& placeholder = extern_op->input_placeholders[i];
    const te::Tensor& input_tensor = extern_op->inputs[i];
    auto it = info->tensor2tensors.find(input_tensor);
    TVM_FFI_ICHECK(it != info->tensor2tensors.end());
    var_map.insert_or_assign(placeholder.get(), it->second);
    input_tensor_map.insert_or_assign(placeholder.get(), it->second);
  }

  // Step 2. Update info with its output tensor and placeholder tensor.
  TVM_FFI_ICHECK_EQ(extern_op->num_outputs(), extern_op->output_placeholders.size());
  for (int i = 0; i < extern_op->num_outputs(); ++i) {
    const TensorVar& placeholder = extern_op->output_placeholders[i];
    const te::Tensor& output_tensor = extern_op.output(i);
    TensorVar output_tir_tensor = info->transformer->NormalizeTensorType(placeholder);
    if (!info->IsArg(output_tensor)) {
      if (auto offset_var =
              tvm::tirx::MatchShapeStridePattern(placeholder.ty()).value().get<2>().as<PrimVar>()) {
        var_map.insert_or_assign(offset_var.value().get(), IntImm(offset_var.value().ty(), 0));
      }
      ffi::ObjectPtr<TensorTypeObj> type =
          ffi::make_object<TensorTypeObj>(*output_tir_tensor.ty().get());
      if (type->layout.has_value()) {
        auto tile = type->layout.value().as_or_throw<tirx::TileLayout>();
        type->layout = tirx::TileLayout(tile->shard, tile->replica, {});
        if (type->HasDefaultLayout(ffi::GetRef<sym::Analyzer>(analyzer)))
          type->layout = std::nullopt;
      }
      output_tir_tensor =
          TensorVar(output_tir_tensor->name, TensorType(std::move(type)), output_tir_tensor->loc);
      info->root_alloc.push_back(output_tir_tensor);
    }
    input_tensor_map.insert_or_assign(placeholder.get(), output_tir_tensor);
    var_map.insert_or_assign(placeholder.get(), output_tir_tensor);
    info->tensor2tensors.insert_or_assign(output_tensor, output_tir_tensor);
  }

  // The access region does not need to be collected here, as it will
  // be generated with the later application of "s_tir.script.Complete" in
  // GenerateAndCompleteFunction.  Waiting until later also handles
  // the case where there is only a single BlockNode, which then
  // becomes the root s_tir::SBlock of the function, and should not have
  // reads/writes filled in.

  auto substituter = ffi::make_object<TensorSubstituter>(var_map, input_tensor_map);
  Stmt lowered_body = info->transformer->Mutate(extern_op->body).ValueOrUnchanged(extern_op->body);
  Stmt substituted_body =
      substituter->Mutate(lowered_body, InplaceMode::kDisallow).ValueOrUnchanged(lowered_body);

  auto transformer = ffi::make_object<TensorLoadToTensorTransformer>(info->tensor2tensors);
  Stmt body = transformer->Mutate(substituted_body, InplaceMode::kDisallow)
                  .ValueOrUnchanged(substituted_body);

  // Step 4. Generate opaque block as body.
  return s_tir::SBlockRealize(/*iter_values=*/{},
                              /*predicate=*/IntImm::Bool(true),
                              /*block=*/
                              s_tir::SBlock(/*iter_vars=*/{},
                                            /*reads=*/{},
                                            /*writes=*/{},
                                            /*name_hint=*/info->FreshName(extern_op->name),
                                            /*body=*/std::move(body),
                                            /*init=*/std::nullopt,
                                            /*alloc_tensors=*/{},
                                            /*match_tensors=*/{},
                                            /*annotations=*/
                                            info->transformer->Mutate(extern_op->attrs)
                                                .ValueOrUnchanged(extern_op->attrs)
                                                .cast<ffi::Map<ffi::String, ffi::Any>>()));
}

ffi::Array<te::Operation> CollectOrderedOps(const ffi::Array<te::Tensor>& arg_list) {
  ffi::Array<te::Operation> arg_ops;
  for (const te::Tensor& arg : arg_list) {
    arg_ops.push_back(arg->op.value());
  }
  te::ReadGraph g = te::CreateReadGraph(arg_ops);
  ffi::Array<te::Operation> order = te::PostDFSOrder(arg_ops, g);

  for (const te::Operation& op : order) {
    if (!(op->IsInstance<te::PlaceholderOpObj>() || op->IsInstance<te::ComputeOpObj>() ||
          op->IsInstance<te::ExternOpObj>()))
      TVM_FFI_THROW(InternalError)
          << "TypeError: Unsupported Operation: " << op->GetTypeKey() << ". "
          << "Only te.placeholder and te.compute are allowed for now.";
  }
  return order;
}

void InitializeTensorBinds(const ffi::Array<te::Operation>& ordered_ops, CreateFuncInfo* info) {
  // Process any TE operations which contain user defined tensors
  for (const auto& op : ordered_ops) {
    // Initialize the tensor2tensor binds map with tensors defined by the te.extern
    if (const auto* extern_op = op.as<te::ExternOpObj>()) {
      TVM_FFI_ICHECK_EQ(extern_op->inputs.size(), extern_op->input_placeholders.size());
      for (size_t i = 0; i < extern_op->inputs.size(); ++i) {
        const te::Tensor& input = extern_op->inputs[i];
        const TensorVar& tensor = extern_op->input_placeholders[i];
        info->tensor2tensors.insert_or_assign(input,
                                              info->transformer->NormalizeTensorType(tensor));
      }
    }
  }
}

void RewriteStageToBlock(const te::Operation& op, CreateFuncInfo* info,
                         ffi::Array<Stmt>* root_stmts, sym::AnalyzerObj* analyzer) {
  if (const auto* placeholder = op.as<te::PlaceholderOpObj>()) {
    // Case 1. PlaceholderOp (te.placeholder)
    TVM_FFI_ICHECK_EQ(op->num_outputs(), 1);
    const te::Tensor& tensor = op.output(0);
    // Check op is in op list
    TVM_FFI_ICHECK(info->IsArg(tensor))
        << "The operation " << op << " produces tensor " << tensor
        << ", but this tensor does not appear as a function argument.  "
        << "The function accepts arguments " << info->arg_list;
    // Declare a tensor for any argument tensors without a pre-existing
    // tensor declaration recorded in the tensor2tensor binds map
    if (info->tensor2tensors.count(tensor) == 0) {
      const TensorVar& tir_tensor = TensorVar(
          placeholder->name, TensorType(info->transformer->Mutate(placeholder->shape)
                                            .ValueOrUnchanged(placeholder->shape)
                                            .cast<ffi::Array<PrimExpr>>(),
                                        placeholder->dtype, runtime::MemoryScope::Global()));
      info->tensor2tensors.insert_or_assign(tensor, tir_tensor);
    }
  } else if (auto compute_op = op.as<te::ComputeOp>()) {
    // Case 2. ComputeOp (te.compute)
    root_stmts->push_back(GenerateStmtFromCompute(compute_op.value(), info, analyzer));
  } else if (const auto extern_op = op.as<te::ExternOp>()) {
    // Case 3. ExternOp (te.extern)
    root_stmts->push_back(GenerateStmtFromExternOp(extern_op.value(), info, analyzer));
  } else {
    TVM_FFI_ICHECK(false) << "TypeError: Unsupported Operation: " << op->GetTypeKey() << ". "
                          << "Only te.placeholder and te.compute are allowed for now.";
  }
}

Function GenerateAndCompleteFunction(const ffi::Array<te::Tensor>& arg_list,
                                     const ffi::Array<Stmt>& root_stmts, CreateFuncInfo* info) {
  ffi::Array<Var> parameters;
  for (const te::Tensor& tensor : arg_list) {
    auto it = info->tensor2tensors.find(tensor);
    TVM_FFI_ICHECK(it != info->tensor2tensors.end());
    parameters.push_back(it->second);
  }
  SeqStmt body(root_stmts);
  body = info->transformer->Mutate(body, InplaceMode::kAllow).ValueOrUnchanged(body);
  Function func = WithAttrs(Function(/*params=*/std::move(parameters),
                                     /*body=*/std::move(body),
                                     /*ret_type=*/VoidType()),
                            {{tvm::attr::kGlobalSymbol.name, ffi::String("main")},
                             {tvm::tirx::attr::kNoAlias.name, true},
                             {tvm::attr::kScriptNamespace.name, ffi::String("s_tir")}});
  const auto fcomplete = tvm::ffi::Function::GetGlobal("s_tir.script.Complete");
  TVM_FFI_ICHECK(fcomplete.has_value());
  func = (*fcomplete)(std::move(func), info->root_alloc).cast<Function>();
  return func;
}

Function CreateFunction(const ffi::Array<te::Tensor>& arg_list,
                        std::optional<PrimType> index_dtype_override) {
  // Information used in CreateFunction and its sub-functions.
  CreateFuncInfo info(arg_list);
  // Root body stmts.
  ffi::Array<Stmt> root_stmts;
  // Analyzer
  sym::Analyzer analyzer;

  // Step 1. Create ordered array of operations and validate they are supported.
  ffi::Array<te::Operation> order = CollectOrderedOps(arg_list);

  // Step 2. Initialize tensor binds map
  InitializeTensorBinds(order, &info);

  // Step 3. Rewrite compute stages into blocks.
  for (const te::Operation& op : order) {
    RewriteStageToBlock(op, &info, &root_stmts, analyzer.get());
  }

  // Step 4. Create and complete the function.
  auto func = GenerateAndCompleteFunction(arg_list, root_stmts, &info);
  if (index_dtype_override.has_value()) {
    func = ffi::make_object<s_tir::IndexDataTypeNormalizer>(index_dtype_override.value())
               ->Rewrite(std::move(func));
  }
  auto result = ffi::make_object<LayoutFreePlaceholdersNormalizer>()->Process(std::move(func));
  VerifyNoOpaqueArtifacts(result);
  return result;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("te.CreateFunction", [](ffi::PackedArgs args, ffi::Any* ret) {
    ffi::Array<ffi::ObjectRef> arg_list = args[0].cast<ffi::Array<ffi::ObjectRef>>();
    std::optional<PrimType> index_dtype_override{std::nullopt};
    // Add conversion to make std::optional compatible with FFI.
    if (args[1] != nullptr) {
      index_dtype_override = args[1].cast<PrimType>();
    }
    *ret = CreateFunction(arg_list, index_dtype_override);
  });
}

// Relax version impl
Function GenerateAndCompleteFunction(const ffi::Array<ffi::ObjectRef>& arg_tir_var_list,
                                     const ffi::Array<Stmt>& root_stmts, CreateFuncInfo* info) {
  ffi::Array<Var> parameters;
  for (const ffi::ObjectRef& arg : arg_tir_var_list) {
    if (auto opt_tensor = arg.as<te::Tensor>()) {
      te::Tensor tensor = opt_tensor.value();
      auto it = info->tensor2tensors.find(tensor);
      TVM_FFI_ICHECK(it != info->tensor2tensors.end());
      parameters.push_back(it->second);
    } else if (auto var = arg.as<PrimVar>()) {
      parameters.push_back(var.value());
    }
  }
  SeqStmt body(root_stmts);
  body = info->transformer->Mutate(body, InplaceMode::kAllow).ValueOrUnchanged(body);
  Function func = WithAttrs(Function(/*params=*/std::move(parameters),
                                     /*body=*/std::move(body),
                                     /*ret_type=*/VoidType()),
                            {{tvm::attr::kGlobalSymbol.name, ffi::String("main")},
                             {tvm::tirx::attr::kNoAlias.name, true},
                             {tvm::attr::kScriptNamespace.name, ffi::String("s_tir")}});
  const auto fcomplete = tvm::ffi::Function::GetGlobal("s_tir.script.Complete");
  TVM_FFI_ICHECK(fcomplete.has_value());
  func = (*fcomplete)(std::move(func), info->root_alloc).cast<Function>();
  return func;
}

Function CreateFunction(const ffi::Array<ffi::ObjectRef>& arg_list,
                        std::optional<PrimType> index_dtype_override) {
  ffi::Array<te::Tensor> tensor_arg_list;
  for (const ffi::ObjectRef& x : arg_list) {
    if (auto tensor_node = x.as<te::TensorObj>()) {
      te::Tensor tensor = ffi::GetRef<te::Tensor>(tensor_node);
      tensor_arg_list.push_back(tensor);
    }
  }
  // Information used in CreateFunction and its sub-functions.
  CreateFuncInfo info(tensor_arg_list);
  // Root body stmts.
  ffi::Array<Stmt> root_stmts;
  // Analyzer
  sym::Analyzer analyzer;

  // Step 1. Create ordered array of operations and validate they are supported.
  ffi::Array<te::Operation> order = CollectOrderedOps(tensor_arg_list);

  // Step 2. Initialize tensor binds map
  InitializeTensorBinds(order, &info);

  // Step 3. Rewrite compute stages into blocks.
  for (const te::Operation& op : order) {
    RewriteStageToBlock(op, &info, &root_stmts, analyzer.get());
  }
  auto func = GenerateAndCompleteFunction(arg_list, root_stmts, &info);
  if (index_dtype_override.has_value()) {
    func = ffi::make_object<s_tir::IndexDataTypeNormalizer>(index_dtype_override.value())
               ->Rewrite(std::move(func));
  }
  auto result = ffi::make_object<LayoutFreePlaceholdersNormalizer>()->Process(std::move(func));
  VerifyNoOpaqueArtifacts(result);
  return result;
}

}  // namespace tirx
}  // namespace tvm
