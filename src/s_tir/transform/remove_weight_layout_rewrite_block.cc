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
 * \file remove_weight_layout_rewrite_block.cc
 * \brief Remove weight layout rewrite block before benchmark
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/transform.h>
#include <tvm/tirx/index_map.h>

#include <unordered_set>

namespace tvm {
namespace s_tir {
using namespace tvm::prim;
using namespace tvm::tirx;

class RemoveLayoutRewriteBlock : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) override {
    if (value.as<ExprObj>()) return ffi::Unchanged();
    return StmtExprMutator::Mutate(value, inplace_mode);
  }

  static std::tuple<Function, ffi::Map<TensorVar, TensorVar>,
                    std::unordered_map<const VarObj*, IndexMap>,
                    std::unordered_map<const VarObj*, ffi::Array<PrimExpr>>>
  Rewrite(Function f) {
    auto rewriter = ffi::make_object<RemoveLayoutRewriteBlock>();

    FunctionObj* n = f.CopyOnWrite();
    n->body = rewriter->Mutate(n->body, InplaceMode::kAllow).ValueOrUnchanged(std::move(n->body));
    return std::make_tuple(f, rewriter->tensor_map_, rewriter->tensor_var_to_index_map_,
                           rewriter->tensor_var_to_rewritten_shape_);
  }

 private:
  UnchangedOr<Stmt> Mutate_(const SBlockObj* op, InplaceMode inplace_mode) final {
    SBlock block = StmtExprMutator::Mutate_(op, inplace_mode)
                       .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                       .as_or_throw<SBlock>();

    auto it = block->annotations.find(tvm::s_tir::attr::kMetaScheduleLayoutRewritePreproc.name);
    if (it == block->annotations.end() || !IsOne((*it).second.cast<PrimExpr>())) {
      // The block is not a weight layout block
      // Remove allocates if needed
      ffi::Array<TensorVar> alloc_tensors;
      for (const TensorVar& tensor : block->alloc_tensors) {
        if (!rewritten_tensors_.count(tensor)) {
          alloc_tensors.push_back(tensor);
        }
      }
      if (alloc_tensors.size() < block->alloc_tensors.size()) {
        SBlockObj* n = block.CopyOnWrite();
        n->alloc_tensors = std::move(alloc_tensors);
        return block;
      } else {
        return block;
      }
    }

    // Step 0. Checking block attrs
    TVM_FFI_ICHECK(block->alloc_tensors.empty());
    TVM_FFI_ICHECK(block->match_tensors.empty());

    // Step 1. Checking the body is a TensorStore
    const auto* store =
        block->body->size() == 1 ? block->body->seq[0].as<TensorStoreObj>() : nullptr;
    TVM_FFI_ICHECK(store);

    // Step 2. Checking the rhs of tensor store is a TensorLoad
    const auto* load = store->value.as<TensorLoadObj>();
    TVM_FFI_ICHECK(load);

    // Step 3. Update TensorVar
    tensor_map_.Set(load->source.as_or_throw<tvm::tirx::TensorVar>(),
                    store->dest.as_or_throw<TensorVar>());
    rewritten_tensors_.insert(store->dest.as_or_throw<TensorVar>());

    // Step 4. Set block body as no_op
    Stmt old_body = block->body;
    SBlockObj* n = block.CopyOnWrite();
    n->body = SeqStmt({});
    n->reads = {};
    n->writes = {};

    ffi::Array<PrimVar> load_indices;
    for (auto ind : load->indices) {
      TVM_FFI_ICHECK(ind.as<PrimVar>());
      load_indices.push_back(ind.as_or_throw<PrimVar>());
    }
    tensor_var_to_index_map_.insert_or_assign(
        load->source.as_or_throw<tvm::tirx::TensorVar>().get(),
        IndexMap(load_indices, store->indices));

    tensor_var_to_rewritten_shape_[load->source.as_or_throw<tvm::tirx::TensorVar>().get()] =
        store->dest.as_or_throw<TensorVar>().shape();

    return block;
  }

  /*! \brief The tensor map from original layout tensor to rewritten tensor */
  ffi::Map<TensorVar, TensorVar> tensor_map_;
  /*! \brief The tensor map from original layout tensor to rewritten tensor */
  std::unordered_set<TensorVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> rewritten_tensors_;
  /*! \brief Maps a tensor load to an index map associated with the load / store
    in a layout rewrite block. */
  std::unordered_map<const VarObj*, IndexMap> tensor_var_to_index_map_;
  /*! \brief Maps a tensor load to the shape of the corresponding rewritten tensor. */
  std::unordered_map<const VarObj*, ffi::Array<PrimExpr>> tensor_var_to_rewritten_shape_;
};

class WeightLayoutRewriteBlockRemover : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) override {
    if (value.as<ExprObj>()) return ffi::Unchanged();
    return StmtExprMutator::Mutate(value, inplace_mode);
  }

  static Function Remove(Function f, bool skip_tensor_rewrite) {
    auto [f_, tensor_map, tensor_var_to_index_map, tensor_var_to_rewritten_shape] =
        RemoveLayoutRewriteBlock::Rewrite(f);

    FunctionObj* n = f_.CopyOnWrite();

    ffi::Array<tvm::Var> params;
    for (const tvm::Var& param : f_->params) {
      auto opt_tensor = param.as<TensorVar>();
      if (!opt_tensor.has_value()) {
        params.push_back(param);
        continue;
      }
      TensorVar tensor = opt_tensor.value();
      auto it = tensor_map.find(tensor);
      if (it != tensor_map.end()) {
        params.push_back((*it).second);
      } else {
        params.push_back(param);
      }
    }
    n->params = std::move(params);
    return f_;
  }
};

namespace transform {

Pass RemoveWeightLayoutRewriteBlock(bool skip_tensor_rewrite) {
  auto pass_func = [skip_tensor_rewrite](Function f, IRModule m, PassContext ctx) {
    return WeightLayoutRewriteBlockRemover::Remove(std::move(f), skip_tensor_rewrite);
  };
  return CreateFunctionPass(pass_func, 0, "s_tir.RemoveWeightLayoutRewriteBlock");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.RemoveWeightLayoutRewriteBlock",
                        RemoveWeightLayoutRewriteBlock);
}

}  // namespace transform

}  // namespace s_tir
}  // namespace tvm
