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
 * \file src/relax/transform/split_tir_layout_rewrite.cc
 * \brief Use for rewriting the TIRs after meta_schedule layout rewrite post process.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/ir/transform.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/s_tir/function.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/tirx/function.h>

#include <algorithm>
#include <cstddef>

namespace tvm {
namespace tirx {
using namespace tvm::prim;

class SplitFunctionLayoutRewrite : public s_tir::StmtExprMutator {
 public:
  using s_tir::StmtExprMutator::Mutate;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) final {
    if (value.as<ExprObj>()) return ffi::Unchanged();
    return s_tir::StmtExprMutator::Mutate(value, inplace_mode);
  }

  explicit SplitFunctionLayoutRewrite(const tirx::Function& func) : original_func_(func) {}
  std::tuple<ffi::Optional<tirx::Function>, tirx::Function> Transform(const tirx::Function& func) {
    TVM_FFI_ICHECK(func->body.has_value() && func->body.value()->size() == 1 &&
                   func->body.value()->seq[0].as<s_tir::SBlockRealizeObj>())
        << "The body of the function should be a root block.";
    const auto& block = func->body.value()->seq[0].as<s_tir::SBlockRealizeObj>()->block;
    visit_root_block(block.get());
    if (layout_rewrite_preproc_stmts_.size() > 0) {
      return std::make_tuple(create_layout_rewrite_preproc_func(), create_compute_func());
    } else {
      return std::make_tuple(std::nullopt, func);
    }
  }

 private:
  void sort_rewrite_infos() {
    std::sort(
        rewrite_infos_.begin(), rewrite_infos_.end(),
        [](const RewriteInfo& a, const RewriteInfo& b) { return a.tensor_index < b.tensor_index; });
  }

  tirx::Function create_layout_rewrite_preproc_func() const {
    // Step 1: Check the number of pre_rewrite_tensors and post_rewrite_tensors
    TVM_FFI_ICHECK(rewrite_infos_.size() > 0) << "There should be at least one tensor rewrite.";

    // Step 2: Create the params for the new tirx::Function
    ffi::Array<Var> params;

    for (const auto& info : rewrite_infos_) {
      params.push_back(info.pre_rewrite_tensor);
    }
    for (const auto& info : rewrite_infos_) {
      params.push_back(info.post_rewrite_tensor);
    }

    // Step 3: Create the body for the new tirx::Function
    TVM_FFI_ICHECK(layout_rewrite_preproc_stmts_.size() > 0)
        << "There should be at least one layout rewrite preproc stmt.";
    Stmt body = SeqStmt(layout_rewrite_preproc_stmts_);
    body = s_tir::SBlockRealize(
        /*iter_values=*/ffi::Array<PrimExpr>(),
        /*predicate=*/IntImm::Bool(true),
        /*block=*/
        s_tir::SBlock(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{},
                      /*name_hint=*/"root", body));

    ffi::Map<ffi::String, ffi::Any> dict;
    for (const auto& [key, original_value] : original_func_->attrs->dict) {
      if (key == tvm::attr::kGlobalSymbol.name) {
        dict.Set(key, original_value.as_or_throw<ffi::String>() + "_weight_prepack");
      } else if (key != tvm::s_tir::attr::kLayoutFreeTensors.name) {
        dict.Set(key, original_value);
      }
    }
    DictAttrs attrs(dict);
    tirx::Function func = tirx::Function(params, SeqStmt(body), VoidType(), attrs);

    return tirx::RenewDef(func);
  }

  tirx::Function create_compute_func() const {
    // Step 1: Create the params for the new tirx::Function
    ffi::Array<Var> params = original_func_->params;
    for (const auto& info : rewrite_infos_) {
      const Var& param = params[info.tensor_index];
      TVM_FFI_ICHECK(param.as<tirx::TensorVar>().value() == info.pre_rewrite_tensor);
      params.Set(info.tensor_index, info.post_rewrite_tensor);
    }

    // Step 2: Create the body for the new tirx::Function
    Stmt body = SeqStmt(compute_stmts_);
    s_tir::SBlock original_block =
        original_func_->body.value()->seq[0].as<s_tir::SBlockRealizeObj>()->block;
    ffi::Array<TensorVar> alloc_tensors;
    for (const auto& tensor : original_block->alloc_tensors) {
      auto it =
          std::find_if(rewrite_infos_.begin(), rewrite_infos_.end(),
                       [&](const RewriteInfo& info) { return info.post_rewrite_tensor == tensor; });
      if (it == rewrite_infos_.end()) {
        alloc_tensors.push_back(tensor);
      }
    }

    body = s_tir::SBlockRealize(
        /*iter_values=*/ffi::Array<PrimExpr>(),
        /*predicate=*/IntImm::Bool(true),
        /*block=*/
        s_tir::SBlock(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{},
                      /*name_hint=*/"root", body,
                      /*init=*/std::nullopt,
                      /*alloc_tensors=*/alloc_tensors));

    ffi::Map<ffi::String, ffi::Any> dict;
    for (const auto& [key, original_value] : original_func_->attrs->dict) {
      if (key == tvm::attr::kGlobalSymbol.name) {
        dict.Set(key, original_value.as_or_throw<ffi::String>() + "_prepacked");
      } else if (key != tvm::s_tir::attr::kLayoutFreeTensors.name) {
        dict.Set(key, original_value);
      }
    }
    DictAttrs attrs(dict);
    tirx::Function func = tirx::Function(params, SeqStmt(body), VoidType(), attrs);

    return tirx::RenewDef(func);
  }

  void visit_root_block(const s_tir::SBlockObj* op) {
    for (const auto& stmt : op->body->seq) {
      current_subtree_ = 0;
      Stmt new_stmt = this->Mutate(stmt).ValueOrUnchanged(stmt);
      TVM_FFI_ICHECK(current_subtree_ != 0) << "There should be at least a block in the subtree.";
      TVM_FFI_ICHECK(op->body->size() != 1 || current_subtree_ == -1)
          << "There should be a compute block if there is only one subtree under the root.";
      if (current_subtree_ == 1) {
        layout_rewrite_preproc_stmts_.push_back(new_stmt);
      } else {
        compute_stmts_.push_back(new_stmt);
      }
    }
  }
  UnchangedOr<Stmt> Mutate_(const s_tir::SBlockObj* op, InplaceMode inplace_mode) final {
    s_tir::SBlock block = s_tir::StmtExprMutator::Mutate_(op, inplace_mode)
                              .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                              .as_or_throw<s_tir::SBlock>();
    auto it = op->annotations.find(tvm::s_tir::attr::kMetaScheduleLayoutRewritePreproc.name);
    bool is_layout_rewrite_preproc =
        it != op->annotations.end() && IsOne((*it).second.cast<PrimExpr>());

    if (current_subtree_ == 0) {
      current_subtree_ = is_layout_rewrite_preproc ? 1 : -1;
    } else if (current_subtree_ == 1) {
      TVM_FFI_ICHECK(is_layout_rewrite_preproc)
          << "There is a layout rewrite block in the subtree, but meet a non-layout rewrite block.";
    } else {
      TVM_FFI_ICHECK(!is_layout_rewrite_preproc)
          << "There is a non-layout rewrite block in the subtree, but meet a layout rewrite block.";
    }

    if (is_layout_rewrite_preproc) {
      TVM_FFI_ICHECK(op->reads.size() == 1)
          << "There should be only one read tensor in the layout rewrite";
      TVM_FFI_ICHECK(op->writes.size() == 1)
          << "There should be only one write tensor in the layout rewrite";
      TVM_FFI_ICHECK(op->alloc_tensors.empty())
          << "There should be no alloc tensor in the layout rewrite";
      TVM_FFI_ICHECK(op->match_tensors.empty())
          << "There should be no match tensor in the layout rewrite";
      const TensorVar& preproc_tensor = op->reads[0]->source.as_or_throw<tvm::tirx::TensorVar>();
      int tensor_index = -1;
      for (size_t i = 0; i < original_func_->params.size(); ++i) {
        TensorVar tensor = original_func_->params[i].as_or_throw<tvm::tirx::TensorVar>();
        if (tensor == preproc_tensor) {
          tensor_index = i;
          break;
        }
      }
      TVM_FFI_ICHECK(tensor_index != -1)
          << "The preproc tensor is not found in the original function.";
      rewrite_infos_.push_back(
          RewriteInfo{tensor_index, op->reads[0]->source.as_or_throw<tvm::tirx::TensorVar>(),
                      op->writes[0]->source.as_or_throw<tvm::tirx::TensorVar>()});

      auto new_annotations = op->annotations;
      new_annotations.erase(tvm::s_tir::attr::kMetaScheduleLayoutRewritePreproc.name);
      auto n = ffi::make_object<s_tir::SBlockObj>(*block.get());
      n->annotations = new_annotations;
      return s_tir::SBlock(n);
    }
    return block;
  }

 public:
  struct RewriteInfo {
    int tensor_index;
    TensorVar pre_rewrite_tensor;
    TensorVar post_rewrite_tensor;
  };
  std::vector<RewriteInfo> rewrite_infos_;

 private:
  /*! \brief The stmts that are used for layout rewrite preproc*/
  ffi::Array<Stmt> layout_rewrite_preproc_stmts_;
  /*! \brief The stmts that are other than layout rewrite preproc*/
  ffi::Array<Stmt> compute_stmts_;
  /*!
   \brief Whether the current subtree is a layout rewrite preproc subtree.
          -1: visited a non-layout rewrite preproc block
           0: unsure, not visited any block
           1: visited a layout rewrite preproc block
  */
  int current_subtree_;
  /*! \brief The original function*/
  tirx::Function original_func_;
};
}  // namespace tirx

namespace relax {
class SplitLayoutRewritePreproc : public ExprMutator {
 public:
  static IRModule Transform(const IRModule& mod) {
    SplitLayoutRewritePreproc mutator(mod);

    // Step 1: Split the function into preproc and compute
    for (auto [gv, func] : mod->functions) {
      if (func->IsInstance<tirx::FunctionObj>()) {
        auto tir_rewriter =
            ffi::make_object<tirx::SplitFunctionLayoutRewrite>(func.as_or_throw<tirx::Function>());
        auto [preproc_func, compute_func] =
            tir_rewriter->Transform(func.as_or_throw<tirx::Function>());
        if (preproc_func.has_value()) {
          mutator.split_funcs_.emplace(gv.get(),
                                       std::make_tuple(preproc_func.value(), compute_func));
          mutator.rewrite_infos_.emplace(gv.get(), tir_rewriter->rewrite_infos_);
        }
      }
    }

    for (auto [gv, func] : mod->functions) {
      if (func->IsInstance<relax::FunctionObj>()) {
        auto relax_func = func.as_or_throw<relax::Function>();
        mutator.builder_->UpdateFunction(gv, mutator(relax_func).as_or_throw<relax::Function>());
      }
    }
    return mutator.builder_->GetContextIRModule();
  }

 private:
  explicit SplitLayoutRewritePreproc(const IRModule& mod) : ExprMutator(mod) {}
  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const CallObj* op) final {
    static const Op call_tir_op = Op::Get("relax.call_tir");
    Call call = ExprMutator::VisitExpr_(op).as_or_throw<Call>();

    // Step 1: Skip call to other than `tirx.call_tir`
    if (!call->op.same_as(call_tir_op)) {
      return call;
    }

    // Step 2: Skip if there is no preproc stage
    const GlobalVar gv = call->args[0].as_or_throw<GlobalVar>();
    auto it = split_funcs_.find(gv.get());
    if (it == split_funcs_.end()) {
      return call;
    }

    // Step 3: Get the preproc and compute functions and update the module
    const auto& [preproc_func, compute_func] = it->second;
    GlobalVar preproc_gv = builder_->AddFunction(preproc_func, gv->name_hint + "_weight_prepack");
    GlobalVar compute_gv = builder_->AddFunction(compute_func, gv->name_hint + "_prepacked");
    // Step 4. Get rewrite infos
    auto rewrite_infos_it = rewrite_infos_.find(gv.get());
    TVM_FFI_ICHECK(rewrite_infos_it != rewrite_infos_.end())
        << "Rewrite infos are not found for " << gv->name_hint;
    const auto& rewrite_infos = rewrite_infos_it->second;

    // Step 5: Emit the preproc call
    ffi::Array<Expr> call_tir_args = call->args[1].as_or_throw<Tuple>()->fields;
    ffi::Array<Expr> preproc_args;
    ffi::Array<Type> preproc_ty_list;
    for (const auto& info : rewrite_infos) {
      preproc_args.push_back(call_tir_args[info.tensor_index]);
      tirx::TensorVar rewritten_tensor = info.post_rewrite_tensor;
      for (const auto& shape_expr : rewritten_tensor.shape()) {
        TVM_FFI_ICHECK(shape_expr.as<IntImmObj>())
            << "Currently does not support rewrite tensor with "
               "dynamic shape.";
      }
      preproc_ty_list.push_back(
          TensorType(ShapeExpr(rewritten_tensor.shape()), rewritten_tensor.dtype()));
    }
    Type preproc_ty = preproc_ty_list.size() > 1        //
                          ? TupleType(preproc_ty_list)  //
                          : preproc_ty_list[0];

    // Step 6: Call the preproc function
    Expr preproc_call = builder_->Emit(
        Call(Type::Missing(), call_tir_op, {preproc_gv, Tuple(preproc_args)}, {}, {preproc_ty}));
    if (rewrite_infos.size() == 1) {
      call_tir_args.Set(rewrite_infos[0].tensor_index, preproc_call);
    } else {
      for (size_t i = 0; i < rewrite_infos.size(); ++i) {
        call_tir_args.Set(rewrite_infos[i].tensor_index, TupleGetItem(preproc_call, i));
      }
    }
    Expr main_call = builder_->Emit(
        Call(Type::Missing(), call_tir_op, {compute_gv, Tuple(call_tir_args)}, {}, call->ty_args));

    return main_call;
  }

 private:
  std::unordered_map<const GlobalVarObj*, std::tuple<tirx::Function, tirx::Function>> split_funcs_;
  std::unordered_map<const GlobalVarObj*,
                     std::vector<tirx::SplitFunctionLayoutRewrite::RewriteInfo>>
      rewrite_infos_;
};

}  // namespace relax

namespace transform {
Pass SplitLayoutRewritePreproc() {
  auto pass_func = [](IRModule mod, PassContext pc) {
    return relax::SplitLayoutRewritePreproc::Transform(mod);
  };
  auto pass = CreateModulePass(pass_func, 0, "SplitLayoutRewritePreproc");
  return tvm::transform::Sequential({pass, relax::transform::DeadCodeElimination()},
                                    "SplitLayoutRewritePreproc");
}
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.SplitLayoutRewritePreproc", SplitLayoutRewritePreproc);
}
}  // namespace transform
}  // namespace tvm
