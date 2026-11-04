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
#include <tvm/s_tir/function.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/topi/tags.h>

#include <optional>
#include <unordered_set>

#include "../utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

/*!
 * \brief Collect the block and index where the tensor is read.
 * \note The tensor is expected to be read by only one TensorLoad
 */
class TensorReadPosCollector : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;

  explicit TensorReadPosCollector(const TensorVar& tensor) : tensor_(tensor.get()) {}

  const std::pair<SBlock, int>& GetTensorLocation() const { return tensor_loc_.value(); }

  const ffi::Optional<IndexMap> GetTensorIndexMap() const { return tensor_index_map_; }

 private:
  ffi::Optional<VisitInterrupt> Visit_(const ForObj* op) final {
    loop_stack_.push_back(ffi::GetRef<For>(op));
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
    loop_stack_.pop_back();
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const SBlockRealizeObj* op) final {
    ffi::Optional<SBlockRealize> outer_block_realize = ffi::GetRef<SBlockRealize>(op);
    std::swap(outer_block_realize, cur_realize_);
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
    std::swap(cur_realize_, outer_block_realize);
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadObj* op) final {
    TVM_FFI_ICHECK(cur_realize_.has_value()) << "TensorLoad occurred outside of any block";

    const TensorVar& tensor = op->source.as_or_throw<tvm::tirx::TensorVar>();
    if (tensor_ == tensor.get()) {
      ffi::Map<Var, PrimExpr> subst_map;
      for (size_t i = 0; i < cur_realize_.value()->iter_values.size(); i++) {
        const Var& var = cur_realize_.value()->block->iter_vars[i]->var;
        const PrimExpr& value = cur_realize_.value()->iter_values[i];
        subst_map.Set(var, value);
      }
      auto f_substitute =
          [&subst_map](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
        if (auto repl = subst_map.Get(var)) return ffi::Any(*std::move(repl));
        return ffi::Unchanged();
      };
      ffi::Array<PrimExpr> subst_indices;
      for (const PrimExpr& e : op->indices) {
        subst_indices.push_back(
            ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(e, f_substitute).as_or_throw<PrimExpr>());
      }
      tensor_index_map_ = SuggestIndexMap(/*tensor=*/tensor,                              //
                                          /*indices=*/subst_indices,                      //
                                          /*loops=*/loop_stack_,                          //
                                          /*predicate=*/cur_realize_.value()->predicate,  //
                                          /*analyzer=*/analyzer_.get());
      int tensor_index = GetReadTensorIndex(cur_realize_.value()->block, tensor);
      TVM_FFI_ICHECK(tensor_index != -1);
      tensor_loc_ = std::make_pair(cur_realize_.value()->block, tensor_index);
    }
    return std::nullopt;
  }

  static int GetReadTensorIndex(const SBlock& block, const TensorVar& tensor) {
    for (size_t i = 0; i < block->reads.size(); i++) {
      if (block->reads[i]->source.as_or_throw<tvm::tirx::TensorVar>().same_as(tensor)) {
        return i;
      }
    }
    return -1;
  }

  /*! \brief The tensor of interest. */
  const VarObj* tensor_;
  /*! \brief The block that consumes the tensor and the corresponding read index. */
  std::optional<std::pair<SBlock, int>> tensor_loc_;
  /*! \brief The proposed IndexMap. */
  ffi::Optional<IndexMap> tensor_index_map_;

  /*! \brief Loop stack for calculating IndexMap. */
  ffi::Array<For> loop_stack_;
  /*! \brief Arithmetic analyzer. */
  sym::Analyzer analyzer_;
  /*! \brief Current BlockRealize scope, used in recursive visit */
  ffi::Optional<SBlockRealize> cur_realize_;
};

class LayoutFreeTensorCollector : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;

  ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
    if (value.as<ExprObj>()) return std::nullopt;
    return StmtExprVisitor::Visit(value);
  }

  ffi::Optional<VisitInterrupt> Visit_(const SBlockObj* block) final {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(block));
    if (auto ann = block->annotations.Get(tvm::topi::attr::kLayoutFreePlaceholders)) {
      for (TensorVar tensor : ann.value().as_or_throw<ffi::Array<TensorVar>>()) {
        tensors.insert(tensor);
      }
    }
    return std::nullopt;
  }

  std::unordered_set<TensorVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> tensors;
};

ffi::Array<TensorVar> CollectLayoutFreeTensors(const FunctionObj* func) {
  // Only rewrite Functions with attr "layout_free_tensors"
  ffi::Array<int64_t> layout_free_tensor_index =
      func->GetAttr(s_tir::attr::kLayoutFreeTensors.name, ffi::Array<int64_t>()).value();

  ffi::Array<TensorVar> layout_free_tensors;
  for (int64_t index : layout_free_tensor_index) {
    TVM_FFI_ICHECK(static_cast<size_t>(index) < func->params.size());
    const Var& param = func->params[index];
    layout_free_tensors.push_back(param.as_or_throw<tvm::tirx::TensorVar>());
  }

  auto collector = ffi::make_object<LayoutFreeTensorCollector>();
  collector->Visit(func->body);

  for (auto tensor : collector->tensors) {
    layout_free_tensors.push_back(tensor);
  }
  return layout_free_tensors;
}

std::optional<std::tuple<SBlock, int, IndexMap>> GetSuggestedIndexMap(TensorVar tensor,
                                                                      const FunctionObj* function) {
  auto collector = ffi::make_object<TensorReadPosCollector>(tensor);
  collector->Visit(function->body);

  const auto& index_map = collector->GetTensorIndexMap();

  if (!index_map.has_value()) {
    return std::nullopt;
  }

  const auto& [anchor_block, tensor_index] = collector->GetTensorLocation();

  return std::make_tuple(anchor_block, tensor_index, index_map.value());
}

/*! \brief Get a chain of cache-read blocks, starting from the one consuming tensor. */
std::vector<std::string> GetCacheReadChain(const TensorVar& tensor, const FunctionObj* function) {
  class TensorReadChainCollector : public StmtExprVisitor {
   public:
    using StmtExprVisitor::Visit_;

    ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
      if (value.as<ExprObj>()) return std::nullopt;
      return StmtExprVisitor::Visit(value);
    }

    explicit TensorReadChainCollector(const TensorVar& tensor) : cur_tensor_(tensor.get()) {}

    ffi::Optional<VisitInterrupt> Visit_(const SBlockObj* op) final {
      // Check if this block is doing cache_read or a similar operation that consumes cur_tensor_.
      if (!op->init && op->reads.size() == 1 && op->writes.size() == 1 &&
          op->reads[0]->source.as_or_throw<tvm::tirx::TensorVar>().get() == cur_tensor_) {
        cache_read_chain.push_back(op->name_hint);
        cur_tensor_ = op->writes[0]->source.as_or_throw<tvm::tirx::TensorVar>().get();
      }
      return StmtExprVisitor::Visit_(op);
    }

    std::vector<std::string> cache_read_chain;

   private:
    const VarObj* cur_tensor_;
  };

  auto collector = ffi::make_object<TensorReadChainCollector>(tensor);
  collector->Visit(function->body);
  return collector->cache_read_chain;
}

bool RewriteLayout(const Schedule& sch) {
  std::vector<std::pair<StmtSRef, ffi::String>> results;
  auto add_layout_rewrite_block = [&sch](SBlockRV consumer_block_rv, int tensor_index) {
    SBlockRV rewrite_block_rv = sch->CacheRead(consumer_block_rv, tensor_index, "global");
    sch->Annotate(rewrite_block_rv, s_tir::attr::kMetaScheduleLayoutRewritePreproc.name, true);
  };

  for (const auto& [g_var, base_func] : sch->mod()->functions) {
    const ffi::String& func_name = g_var->name_hint;
    const auto* function = base_func.as<FunctionObj>();
    // Only consider Function
    if (function == nullptr) {
      continue;
    }

    for (auto tensor : CollectLayoutFreeTensors(function)) {
      const auto cache_read_chain = GetCacheReadChain(tensor, function);
      if (cache_read_chain.empty()) {
        // The common case, where the layout-free tensor is directly consumed by an anchor op such
        // as conv2d or dense.
        auto tup_opt = GetSuggestedIndexMap(tensor, function);
        if (tup_opt == std::nullopt) continue;

        auto [anchor_block, tensor_index, index_map] = *tup_opt;
        auto anchor_block_rv = sch->GetSBlock(anchor_block->name_hint, func_name);
        add_layout_rewrite_block(anchor_block_rv, tensor_index);
        sch->TransformLayout(anchor_block_rv, tensor_index, TensorIndexType::kRead, index_map,
                             std::nullopt);
      } else {
        // When the layout-free tensor is consumed by cache_read, we need to find the index map
        // for a cache-read tensor that is directly consumed by an anchor op. The last tensor
        // in cache_read_chain corresponds to that tensor.
        SBlock cache_read_block = sch->Get(sch->GetSBlock(cache_read_chain.back(), func_name));
        TVM_FFI_ICHECK_EQ(cache_read_block->writes.size(), 1);
        auto tup_opt = GetSuggestedIndexMap(
            cache_read_block->writes[0]->source.as_or_throw<tvm::tirx::TensorVar>(), function);
        if (tup_opt == std::nullopt) continue;

        auto [anchor_block, tensor_index, index_map] = *tup_opt;
        // Transform the layout of the last cache-read tensor.
        sch->TransformLayout(sch->GetSBlock(anchor_block->name_hint, func_name), tensor_index,
                             TensorIndexType::kRead, index_map, std::nullopt);

        // Propagate the layout transformation over cache_read_chain, starting from
        // the next-to-last cache-read tensor.
        for (int i = static_cast<int>(cache_read_chain.size()) - 1; i >= 0; --i) {
          SBlockRV cache_read_block_rv = sch->GetSBlock(cache_read_chain[i], func_name);
          if (i == 0) {
            // Before the first cache_read that consumes the layout-free tensor, insert
            // a layout-rewrite block. Another cache-read tensor is added, and its layout is
            // transformed by TransformLayout below.
            add_layout_rewrite_block(cache_read_block_rv, 0);
          }
          sch->TransformLayout(cache_read_block_rv, 0, TensorIndexType::kRead, index_map,
                               std::nullopt);
        }
      }
    }
  }
  return true;
}

}  // namespace s_tir

namespace s_tir {
namespace meta_schedule {
/*! \brief Layout Rewrite. */
class RewriteLayoutObj : public PostprocObj {
 public:
  // Inherited from PostprocObj
  void InitializeWithTuneContext(const TuneContext& context) final {}

  // Inherited from PostprocObj
  bool Apply(const s_tir::Schedule& sch) final {
    try {
      return s_tir::RewriteLayout(sch);
    } catch (const std::runtime_error& e) {
      return false;
    }
  }

  Postproc Clone() const {
    ffi::ObjectPtr<RewriteLayoutObj> n = ffi::make_object<RewriteLayoutObj>(*this);
    return Postproc(ffi::UnsafeInit{}, n);
  }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<RewriteLayoutObj>();
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.RewriteLayout", RewriteLayoutObj,
                                    PostprocObj);
};

Postproc Postproc::RewriteLayout() {
  auto n = ffi::make_object<RewriteLayoutObj>();
  return Postproc(ffi::UnsafeInit{}, n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  RewriteLayoutObj::RegisterReflection();
  refl::GlobalDef().def("s_tir.meta_schedule.PostprocRewriteLayout", Postproc::RewriteLayout);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
