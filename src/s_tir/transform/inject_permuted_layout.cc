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
 * \file inject_permuted_layout.cc
 * \brief The pass injects permuted layout for shared memory tensors to avoid bank conflicts.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/stmt_functor_with_constraint_context.h>
#include <tvm/s_tir/transform.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/function.h>

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

using namespace sym;
using namespace runtime;

class PermutedLayoutInjector : public StmtExprMutatorWithConstraintContext {
 public:
  using StmtExprMutatorWithConstraintContext::Mutate;
  using StmtExprMutatorWithConstraintContext::Mutate_;

  static Function Transform(Function func) {
    Analyzer analyzer;

    auto new_body = ffi::make_object<PermutedLayoutInjector>(analyzer)
                        ->Mutate(func->body)
                        .as_or_throw<UnchangedOr<ffi::Optional<SeqStmt>>>()
                        .ValueOrUnchanged(func->body);
    auto func_node = func.CopyOnWrite();
    func_node->body = new_body;
    return func;
  }

  explicit PermutedLayoutInjector(const Analyzer& analyzer)
      : StmtExprMutatorWithConstraintContext(analyzer->GetConstraintContext()) {}

 private:
  ffi::Array<PrimExpr> PermuteIndices(PrimExpr row_idx, PrimExpr col_idx, int row_size) {
    TVM_FFI_ICHECK(permute_);
    // Index after vectorizing by 8
    PrimExpr col_idx_outer = floordiv(col_idx, VECTORIZE_FACTOR),
             col_idx_inner = floormod(col_idx, VECTORIZE_FACTOR);
    PrimExpr new_col_idx_outer{ffi::UnsafeInit{}};
    if (row_size % 64 == 0) {
      // Use 8 * 8 permuted layout
      // Every number below corresponds to 8 consecutive fp16 number in shared mem, i.e. one read
      // Every row below corresponds to 32 banks
      // 0  1  2  3  4  5  6  7    ==>    0  1  2  3  4  5  6  7
      // 0  1  2  3  4  5  6  7    ==>    1  0  3  2  5  4  7  6
      // 0  1  2  3  4  5  6  7    ==>    2  3  0  1  6  7  4  5
      // 0  1  2  3  4  5  6  7    ==>    3  2  1  0  7  6  5  4
      // 0  1  2  3  4  5  6  7    ==>    4  5  6  7  0  1  2  3
      // 0  1  2  3  4  5  6  7    ==>    5  4  7  6  1  0  3  2
      // 0  1  2  3  4  5  6  7    ==>    6  7  4  5  2  3  0  1
      // 0  1  2  3  4  5  6  7    ==>    7  6  5  4  3  2  1  0
      auto row_idx_sub = floormod(row_idx, 8);
      new_col_idx_outer = col_idx_outer ^ row_idx_sub;
    } else {
      TVM_FFI_ICHECK(row_size % 32 == 0);
      // Use 8 * 4 permuted layout
      // Every number below corresponds to 8 consecutive fp16 number in shared mem, i.e. one read
      // Every row below corresponds to 16 banks
      // 0  1  2  3    ==>    0  1  2  3
      // 0  1  2  3    ==>    0  1  2  3
      // 0  1  2  3    ==>    1  0  3  2
      // 0  1  2  3    ==>    1  0  3  2
      // 0  1  2  3    ==>    2  3  0  1
      // 0  1  2  3    ==>    2  3  0  1
      // 0  1  2  3    ==>    3  2  1  0
      // 0  1  2  3    ==>    3  2  1  0
      // View with 8 elements per row:
      // 0  1  2  3  4  0  1  2  3    ==>    0  1  2  3  0  1  2  3
      // 0  1  2  3  4  0  1  2  3    ==>    1  0  3  2  1  0  3  2
      // 0  1  2  3  4  0  1  2  3    ==>    2  3  0  1  2  3  0  1
      // 0  1  2  3  4  0  1  2  3    ==>    3  2  1  0  3  2  1  0
      auto row_idx_sub = floormod(row_idx, 8);
      new_col_idx_outer = col_idx_outer ^ floordiv(row_idx_sub, 2);
    }
    return {row_idx, GetSyncedAnalyzer()->Simplify(new_col_idx_outer * 8 + col_idx_inner)};
  }

  static bool CheckAnnotation(const Any& annotation) {
    if (auto opt_str = annotation.as<ffi::String>()) {
      // Support string annotation for backward compatibility
      return *opt_str != "";
    } else if (auto* node = annotation.as<IntImmObj>()) {
      return node->value != 0;
    } else if (auto opt_val = annotation.try_cast<int64_t>()) {
      return *opt_val != 0;
    } else {
      TVM_FFI_THROW(InternalError) << "Invalid permuted layout annotation: " << annotation;
    }
  }

  UnchangedOr<Stmt> Mutate_(const SBlockObj* op, InplaceMode inplace_mode) final {
    if (op->annotations.count(tvm::s_tir::attr::kPermutedLayout) == 0 ||
        !CheckAnnotation(op->annotations.at(tvm::s_tir::attr::kPermutedLayout))) {
      return StmtExprMutatorWithConstraintContext::Mutate_(op, inplace_mode);
    }

    auto prev_permute = permute_;
    permute_ = true;

    SBlock block = StmtExprMutatorWithConstraintContext::Mutate_(op, inplace_mode)
                       .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                       .as_or_throw<SBlock>();

    permute_ = prev_permute;

    // Erase the permuted_layout annotation after the pass
    auto block_node = block.CopyOnWrite();
    block_node->annotations.erase(tvm::s_tir::attr::kPermutedLayout);
    return block;
  }

  int CheckAndGetTensorRowSize(TensorVar tensor) {
    TVM_FFI_ICHECK(tensor.shape().size() >= 2)
        << "The dimension of TensorVar \"" << tensor->name << "\" with shape " << tensor.shape()
        << " should be at least 2";

    auto dim = tensor.shape().size();
    auto tensor_row_size = tensor.shape()[dim - 1].as<IntImmObj>()->value;
    auto tensor_col_size = tensor.shape()[dim - 2].as<IntImmObj>()->value;

    if (tensor_row_size % 64 != 0) {
      TVM_FFI_ICHECK(tensor_row_size % 32 == 0)
          << "Permuted SLayout for TensorVar \"" << tensor->name << "\" with shape "
          << tensor.shape()
          << " is not supported since its second dimension is not divisible by 32";
      TVM_FFI_ICHECK(tensor_col_size % 2 == 0)
          << "Permuted SLayout for TensorVar \"" << tensor->name << "\" with shape "
          << tensor.shape()
          << " is not supported since its first dimension is not divisible by 2 and second "
             "dimension is not divisible by 64";
    }

    return tensor_row_size.as<int>().value();
  }

  ffi::Array<PrimExpr> HandleTensorIndices(TensorVar tensor, ffi::Array<PrimExpr> indices) {
    auto tensor_row_size = CheckAndGetTensorRowSize(tensor);

    // Mutate the last two indices
    auto indices_size = indices.size();
    PrimExpr row_idx = indices[indices_size - 2];
    PrimExpr col_idx = indices[indices_size - 1];
    auto new_indices = PermuteIndices(row_idx, col_idx, tensor_row_size);
    indices.Set(indices_size - 2, new_indices[0]);
    indices.Set(indices_size - 1, new_indices[1]);
    return indices;
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreObj* op, InplaceMode inplace_mode) final {
    static const auto scope_shared = runtime::MemoryScope::Get("shared");
    static const auto scope_shared_dyn = runtime::MemoryScope::Get("shared.dyn");

    // Rewrite write from global to shared.dyn or shared
    // We assume the shape of the shared memory is [..., row_size, col_size],
    // where row_size is divisible by 64, or divisible by 32 and col_size is divisible by 2.
    auto store = StmtExprMutatorWithConstraintContext::Mutate_(op, inplace_mode)
                     .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                     .as_or_throw<TensorStore>();

    if (!permute_ || store->dest.as_or_throw<TensorVar>().shape().size() < 2) {
      return store;
    }

    auto scope = store->dest.as_or_throw<TensorVar>().scope();
    if ((scope != scope_shared && scope != scope_shared_dyn)) {
      return store;
    }

    auto store_node = store.CopyOnWrite();
    store_node->indices =
        HandleTensorIndices(store_node->dest.as_or_throw<TensorVar>(), store_node->indices);
    return store;
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* op, InplaceMode inplace_mode) final {
    static const auto scope_shared = runtime::MemoryScope::Get("shared");
    static const auto scope_shared_dyn = runtime::MemoryScope::Get("shared.dyn");

    // Rewrite load from shared or shared.dyn to global
    auto load = StmtExprMutatorWithConstraintContext::Mutate_(op, inplace_mode)
                    .ValueOrUnchanged(ffi::GetRef<PrimExpr>(op))
                    .as_or_throw<TensorLoad>();

    if (!permute_ || load->source.as_or_throw<tvm::tirx::TensorVar>().shape().size() < 2) {
      return load;
    }

    auto scope = load->source.as_or_throw<tvm::tirx::TensorVar>().scope();
    if ((scope != scope_shared && scope != scope_shared_dyn)) {
      return load;
    }

    return TensorLoad(
        load->source.as_or_throw<tvm::tirx::TensorVar>(),
        HandleTensorIndices(load->source.as_or_throw<tvm::tirx::TensorVar>(), load->indices),
        load->loc);
  }

  static constexpr size_t VECTORIZE_FACTOR = 8;

  bool permute_ = false;
};

namespace transform {

Pass InjectPermutedLayout() {
  auto pass_func = [=](Function f, IRModule m, PassContext ctx) {
    return PermutedLayoutInjector::Transform(std::move(f));
  };
  return CreateFunctionPass(pass_func, 0, "s_tir.InjectPermutedLayout");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.InjectPermutedLayout", InjectPermutedLayout);
}

}  // namespace transform

}  // namespace s_tir
}  // namespace tvm
