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
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/transform.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/analysis.h>

#include "../memory_scope.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

/*!
 * \brief Rewriter for all m16n8k8.matrix[A/B/C] tensor. This pass mainly do two things:
 *     1. Lower m16n8k8.matrix[A/B/C] tensor to local registers, where each thread holds their
 *        own part of the matrix;
 *     2. Rewrite access of mma.matrixC so it can access the correct part of the matrix.
 *   The reason why access of m16n8k8.matrix[A/B] tensor doesn't need this kind of rewrite is
 *   that their access is through opaque accesses to per-thread fragments.
 *   We cannot use this kind of opaque access in matrixC too since the ptx stmatrix is only
 *   supported for sm90 or higher. Therefore, writeback of matrixC is limited to the
 *   transparent way.
 */
class MmaTensorLayoutTransformer : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  UnchangedOr<Stmt> Mutate_(const SBlockObj* op, InplaceMode inplace_mode) {
    SBlock block = ffi::GetRef<SBlock>(op);
    auto* n = block.CopyOnWrite();
    auto fmutate = [this](const TensorVar& tensor) {
      static const auto scope_local = runtime::MemoryScope::Get("local");
      const auto scope_mma_matrixC = GetRegisteredMemoryScope("mma.matrixC");
      const auto scope_mma_matrixA = GetRegisteredMemoryScope("mma.matrixA");
      const auto scope_mma_matrixB = GetRegisteredMemoryScope("mma.matrixB");

      // m16n8k8.matrix[A/B/C] tensors are composed ofseveral small blocks. Assume the block's
      // shape is [bi, bj]. Inside each small block, we have 8 threads in stride dimension and 4
      // threads in contiguous dimension, so we change the tensor's shape from [i, j]
      // to [i // bi, j // bj, bi // 8, bj // 4].
      if ((scope_mma_matrixC && tensor.scope() == *scope_mma_matrixC)) {
        // mma.matrixC
        // bi = 16, bj = 8
        size_t size = tensor.shape().size();
        TVM_FFI_ICHECK_GE(size, 2);
        const IntImmObj* dim0 = tensor.shape()[size - 2].as<IntImmObj>();
        const IntImmObj* dim1 = tensor.shape()[size - 1].as<IntImmObj>();
        TVM_FFI_ICHECK(dim0 != nullptr && dim1 != nullptr);
        TVM_FFI_ICHECK(dim0->value % 16 == 0 && dim1->value % 8 == 0);

        std::vector<PrimExpr> new_shape;
        for (size_t i = 0; i < size - 2; ++i) {
          new_shape.push_back(tensor.shape()[i]);
        }
        new_shape.insert(new_shape.end(),
                         {IntImm::Int32(dim0->value / 16), IntImm::Int32(dim1->value / 8), 2, 2});

        TensorVar new_tensor =
            TensorVar(tensor->name, TensorType(std::move(new_shape), tensor.dtype(), scope_local));
        VarRemapSet(tensor, new_tensor);
        return new_tensor;

      } else if ((scope_mma_matrixA && tensor.scope() == *scope_mma_matrixA)) {
        // mma.matrixA
        // bi = 32, bj = 8
        size_t size = tensor.shape().size();
        TVM_FFI_ICHECK_GE(size, 2);
        const IntImmObj* dim0 = tensor.shape()[size - 2].as<IntImmObj>();
        const IntImmObj* dim1 = tensor.shape()[size - 1].as<IntImmObj>();
        TVM_FFI_ICHECK(dim0 != nullptr && dim1 != nullptr);
        TVM_FFI_ICHECK(dim0->value % 32 == 0 && dim1->value % 8 == 0);
        std::vector<PrimExpr> new_shape;
        for (size_t i = 0; i < size - 2; ++i) {
          new_shape.push_back(tensor.shape()[i]);
        }
        new_shape.insert(new_shape.end(),
                         {IntImm::Int32(dim0->value / 32), IntImm::Int32(dim1->value / 8), 4, 2});

        TensorVar new_tensor =
            TensorVar(tensor->name, TensorType(std::move(new_shape), tensor.dtype(), scope_local));
        VarRemapSet(tensor, new_tensor);
        return new_tensor;

      } else if ((scope_mma_matrixB && tensor.scope() == *scope_mma_matrixB)) {
        // mma.matrixB
        // bj = 8, bj = 32
        size_t size = tensor.shape().size();
        TVM_FFI_ICHECK_GE(size, 2);
        const IntImmObj* dim0 = tensor.shape()[size - 2].as<IntImmObj>();
        const IntImmObj* dim1 = tensor.shape()[size - 1].as<IntImmObj>();
        TVM_FFI_ICHECK(dim0 != nullptr && dim1 != nullptr);
        TVM_FFI_ICHECK(dim0->value % 8 == 0 && dim1->value % 32 == 0);
        std::vector<PrimExpr> new_shape;
        for (size_t i = 0; i < size - 2; ++i) {
          new_shape.push_back(tensor.shape()[i]);
        }
        new_shape.insert(new_shape.end(),
                         {IntImm::Int32(dim0->value / 8), IntImm::Int32(dim1->value / 32), 1, 8});

        TensorVar new_tensor =
            TensorVar(tensor->name, TensorType(std::move(new_shape), tensor.dtype(), scope_local));
        VarRemapSet(tensor, new_tensor);
        return new_tensor;
      }
      return tensor;
    };
    n->alloc_tensors.MutateByApply(fmutate);
    n->body = Mutate(n->body, inplace_mode).ValueOrUnchanged(n->body);
    return block;
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreObj* op, InplaceMode inplace_mode) {
    const auto scope_mma_matrixC = GetRegisteredMemoryScope("mma.matrixC");
    const auto scope_mma_matrixA = GetRegisteredMemoryScope("mma.matrixA");
    const auto scope_mma_matrixB = GetRegisteredMemoryScope("mma.matrixB");

    TensorVar original_tensor = op->dest.as_or_throw<TensorVar>();
    auto value = Mutate(op->value, inplace_mode);
    auto indices =
        Mutate(op->indices, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    TensorStore store = ffi::GetRef<TensorStore>(op);
    if (!value.UnchangedOrSameAs(op->value) || !indices.UnchangedOrSameAs(op->indices)) {
      auto* n = store.CopyOnWrite();
      n->value = std::move(value).ValueOrUnchanged(op->value);
      n->indices = std::move(indices).ValueOrUnchanged(op->indices);
    }
    if (auto replacement = VarRemapGet(original_tensor).as<TensorVar>()) {
      auto* n = store.CopyOnWrite();
      if ((scope_mma_matrixC && original_tensor.scope() == *scope_mma_matrixC)) {
        auto new_indices = MatrixCIndices(store->indices);
        n->dest = replacement.value();
        n->indices = std::move(new_indices);
      } else if ((scope_mma_matrixA && original_tensor.scope() == *scope_mma_matrixA) ||
                 (scope_mma_matrixB && original_tensor.scope() == *scope_mma_matrixB)) {
        TVM_FFI_ICHECK(false)
            << "TransformMmaTensorLayout requires " << original_tensor.scope()
            << " tensors to be accessed through opaque ldmatrix/mma_sync operations, but found "
               "an explicit TensorStore.";
      }
    }
    return store;
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* op, InplaceMode inplace_mode) {
    const auto scope_mma_matrixC = GetRegisteredMemoryScope("mma.matrixC");

    TensorVar tensor = op->source.as_or_throw<TensorVar>();
    // Remap the source together with its indices below, after the scope checks.
    auto indices_result =
        Mutate(op->indices, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    TensorLoad load = ffi::GetRef<TensorLoad>(op);
    if (!indices_result.UnchangedOrSameAs(op->indices)) {
      load.CopyOnWrite()->indices = std::move(indices_result).ValueUnchecked();
    }
    if (auto replacement = VarRemapGet(tensor).as<TensorVar>()) {
      ffi::Array<PrimExpr> indices = load->indices;
      if ((scope_mma_matrixC && tensor.scope() == *scope_mma_matrixC)) {
        indices = MatrixCIndices(load->indices);
      } else {
        TVM_FFI_ICHECK(false)
            << "TransformMmaTensorLayout requires " << tensor.scope()
            << " tensors to be accessed through opaque ldmatrix/mma_sync operations, but found "
               "an explicit TensorLoad.";
      }
      return TensorLoad(replacement.value(), indices, load->loc);
    }
    return load;
  }

 private:
  // Preserve the per-thread accumulator layout independently of tensor-intrinsic registration.
  ffi::Array<PrimExpr> MatrixCIndices(const ffi::Array<PrimExpr>& indices) {
    TVM_FFI_ICHECK_EQ(indices.size(), 2U);
    PrimExpr i = indices[0], j = indices[1];
    return {analyzer->Simplify(floordiv(i, 16)), analyzer->Simplify(floordiv(j, 8)),
            analyzer->Simplify(floormod(floordiv(i, 8), 2)), analyzer->Simplify(floormod(j, 2))};
  }

  sym::Analyzer analyzer;
};

namespace transform {

Pass TransformMmaTensorLayout() {
  auto pass_func = [=](Function f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = ffi::make_object<MmaTensorLayoutTransformer>()
                  ->Mutate(n->body, InplaceMode::kAllow)
                  .ValueOrUnchanged(std::move(n->body));
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "s_tir.TransformMmaTensorLayout");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.TransformMmaTensorLayout", TransformMmaTensorLayout);
}
}  // namespace transform

}  // namespace s_tir
}  // namespace tvm
