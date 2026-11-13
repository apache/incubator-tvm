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
#ifndef TVM_S_TIR_TRANSFORM_MEMHAMMER_REWRITE_RULE_H_
#define TVM_S_TIR_TRANSFORM_MEMHAMMER_REWRITE_RULE_H_

#include <tvm/ffi/function.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/transform.h>
#include <tvm/sym/iter_affine_map.h>
#include <tvm/target/target.h>

#include <utility>

#include "../schedule/utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::prim;
using namespace tvm::tirx;

/*! \brief The set containing all possible constraints of a data copy */
struct ConstraintSet {
  /*! \brief The extents of the thread binding loops */
  ffi::Map<ffi::String, int64_t> thread_extent;
  /*! \brief The outer loops surrounding the data copy */
  ffi::Array<For> outer_loops;
  /*! \brief The read region of the data copy */
  TensorRegion read_region;
  /*! \brief The write region of the data copy */
  TensorRegion write_region;
  /*! \brief The dtype size in bits */
  int data_bits;
  /*! \brief Whether to insert a local stage in the data copy */
  int add_local_stage = 0;
  /*! \brief The vectorization length in bytes */
  int vector_bytes = 1;

  explicit ConstraintSet(ffi::Map<ffi::String, int64_t> thread_extent,  //
                         ffi::Array<For> outer_loops,                   //
                         TensorRegion read_region,                      //
                         TensorRegion write_region,                     //
                         int data_bits,                                 //
                         const ffi::Map<ffi::String, ffi::Any>& ann)
      : thread_extent(thread_extent),
        outer_loops(outer_loops),
        read_region(read_region),
        write_region(write_region),
        data_bits(data_bits) {
    if (auto add_local_stage = ann.Get(tvm::s_tir::attr::kLocalStage.name)) {
      this->add_local_stage = static_cast<bool>(add_local_stage.value().cast<IntImm>()->value);
    }
    if (auto vector_bytes = ann.Get(tvm::s_tir::attr::kVectorBytes.name)) {
      this->vector_bytes = vector_bytes.value().cast<IntImm>()->value.as<int>().value();
    }
  }
};

/*! \brief The set containing all possible outputs of a rewrite rule */
struct OutputSet {
  /*! \brief New tensors allocated after rewrite */
  ffi::Array<TensorVar> alloc_tensor;
  /*! \brief The minimal padding size of a tensor in base 2 logarithm */
  ffi::Map<TensorVar, int64_t> padding_min;
};

/*!
 * \brief Rules to rewrite a data copy.
 */
class RewriteRule {
 protected:
  /* RewriteRule() = default; */
  /*!
   * \brief Rewrite the stmt under certain constraints
   * \param stmt The stmt
   * \param constraints The constraints of the rewrite
   * \param output Some additional information that the rewrite rule produces. (including the new
   *               tensor to be allocated, etc.)
   * \return the stmt after rewrite
   */
  virtual Stmt Rewrite(const Stmt& stmt, const ConstraintSet& constraints,
                       OutputSet* output) const = 0;
  /*!
   * \brief Whether the rewrite rule can be applied to the stmt under certain constraints
   * \param stmt The stmt
   * \param constraints The constraints of the rewrite
   * \return A boolean flag indicating whether the rule can be applied
   */
  virtual bool CanApply(const Stmt& stmt, const ConstraintSet& constraints) const { return true; }

 public:
  inline Stmt Apply(const Stmt& stmt, const ConstraintSet& constraints, OutputSet* output) const {
    if (CanApply(stmt, constraints)) {
      return Rewrite(stmt, constraints, output);
    } else {
      return stmt;
    }
  }
};

inline bool IsScope(const TensorVar& tensor, runtime::MemoryScope expected) {
  static const auto scope_shared = runtime::MemoryScope::Get("shared");
  static const auto scope_shared_dyn = runtime::MemoryScope::Get("shared.dyn");

  auto scope = tensor.scope();
  if (scope == expected) return true;
  if (expected == scope_shared) {
    return scope == scope_shared_dyn;
  }
  if (expected == runtime::MemoryScope::Global()) {
    return IsGlobalMemoryScope(scope);
  }
  return false;
}

inline bool IsCopyBetweenScope(const TensorVar& src_tensor, const TensorVar& tgt_tensor,
                               runtime::MemoryScope src_scope, runtime::MemoryScope tgt_scope) {
  return IsScope(src_tensor, src_scope) && IsScope(tgt_tensor, tgt_scope);
}

/*!
 * \brief Coalesce and vectorize memory access.
 */
class CoalescedAccess : public RewriteRule {
 public:
  CoalescedAccess() = default;
  Stmt Rewrite(const Stmt& stmt, const ConstraintSet& constraints, OutputSet* output) const final;
  bool CanApply(const Stmt& stmt, const ConstraintSet& constraints) const final {
    static const auto scope_shared = runtime::MemoryScope::Get("shared");

    TensorVar src_tensor = constraints.read_region->source.as_or_throw<tvm::tirx::TensorVar>();
    TensorVar tgt_tensor = constraints.write_region->source.as_or_throw<tvm::tirx::TensorVar>();
    return IsCopyBetweenScope(src_tensor, tgt_tensor, runtime::MemoryScope::Global(),
                              scope_shared) ||
           IsCopyBetweenScope(src_tensor, tgt_tensor, scope_shared, runtime::MemoryScope::Global());
  }
};

/*!
 * \brief Transform from A[f(i,j)] = B[i,j] to A[i,j] = B[f^{-1}(i,j)]
 */
class InverseMapping : public RewriteRule {
 public:
  InverseMapping() = default;
  Stmt Rewrite(const Stmt& stmt, const ConstraintSet& constraints, OutputSet* output) const final;
  bool CanApply(const Stmt& stmt, const ConstraintSet& constraints) const final {
    static const auto scope_shared = runtime::MemoryScope::Get("shared");

    TensorVar src_tensor = constraints.read_region->source.as_or_throw<tvm::tirx::TensorVar>();
    TensorVar tgt_tensor = constraints.write_region->source.as_or_throw<tvm::tirx::TensorVar>();
    return IsCopyBetweenScope(src_tensor, tgt_tensor, scope_shared, runtime::MemoryScope::Global());
  }
};

/*!
 * \brief Create a local stage when loading from global memory to shared memory.
 */
class CreateLocalStage : public RewriteRule {
 public:
  CreateLocalStage() = default;
  Stmt Rewrite(const Stmt& stmt, const ConstraintSet& constraints, OutputSet* output) const final;
  bool CanApply(const Stmt& stmt, const ConstraintSet& constraints) const final {
    static const auto scope_shared = runtime::MemoryScope::Get("shared");

    TensorVar src_tensor = constraints.read_region->source.as_or_throw<tvm::tirx::TensorVar>();
    TensorVar tgt_tensor = constraints.write_region->source.as_or_throw<tvm::tirx::TensorVar>();
    return IsCopyBetweenScope(src_tensor, tgt_tensor, runtime::MemoryScope::Global(),
                              scope_shared) &&
           IsOne(constraints.add_local_stage);
  }
};

/*!
 * \brief Add a cache stage in shared memory. Perform tensor core rewrite for wmma->shared, and
 *  perform coalescing and vectorizing for shared->global.
 */
class WmmaToGlobal : public RewriteRule {
 public:
  WmmaToGlobal() = default;
  Stmt Rewrite(const Stmt& stmt, const ConstraintSet& constraints, OutputSet* output) const final;
  bool CanApply(const Stmt& stmt, const ConstraintSet& constraints) const final {
    static const auto scope_wmma_accumulator = runtime::MemoryScope::Get("wmma.accumulator");

    TensorVar src_tensor = constraints.read_region->source.as_or_throw<tvm::tirx::TensorVar>();
    TensorVar tgt_tensor = constraints.write_region->source.as_or_throw<tvm::tirx::TensorVar>();
    return IsCopyBetweenScope(src_tensor, tgt_tensor, scope_wmma_accumulator,
                              runtime::MemoryScope::Global());
  }
};

/*!
 * \brief Add a cache stage in shared memory. Perform tensor core rewrite for mma->shared, and
 *  perform coalescing and vectorizing for shared->global.
 */
class MmaToGlobal : public RewriteRule {
 public:
  MmaToGlobal() = default;
  Stmt Rewrite(const Stmt& stmt, const ConstraintSet& constraints, OutputSet* output) const final;
  bool CanApply(const Stmt& stmt, const ConstraintSet& constraints) const final {
    static const auto scope_mma_matrixC = runtime::MemoryScope::Get("mma.matrixC");

    TensorVar src_tensor = constraints.read_region->source.as_or_throw<tvm::tirx::TensorVar>();
    TensorVar tgt_tensor = constraints.write_region->source.as_or_throw<tvm::tirx::TensorVar>();
    return IsCopyBetweenScope(src_tensor, tgt_tensor, scope_mma_matrixC,
                              runtime::MemoryScope::Global());
  }
};

/*!
 * \brief Rewrite shared->wmma data copy with load_matrix_sync
 */
class SharedToWmma : public RewriteRule {
 public:
  SharedToWmma() = default;
  Stmt Rewrite(const Stmt& stmt, const ConstraintSet& constraints, OutputSet* output) const final;
  bool CanApply(const Stmt& stmt, const ConstraintSet& constraints) const final {
    static const auto scope_shared = runtime::MemoryScope::Get("shared");
    static const auto scope_wmma_matrix_a = runtime::MemoryScope::Get("wmma.matrix_a");
    static const auto scope_wmma_matrix_b = runtime::MemoryScope::Get("wmma.matrix_b");

    TensorVar src_tensor = constraints.read_region->source.as_or_throw<tvm::tirx::TensorVar>();
    TensorVar tgt_tensor = constraints.write_region->source.as_or_throw<tvm::tirx::TensorVar>();
    return IsCopyBetweenScope(src_tensor, tgt_tensor, scope_shared, scope_wmma_matrix_a) ||
           IsCopyBetweenScope(src_tensor, tgt_tensor, scope_shared, scope_wmma_matrix_b);
  }
};

/*!
 * \brief Rewrite wmma->shared data copy with store_matrix_sync
 */
class WmmaToShared : public RewriteRule {
 public:
  WmmaToShared() = default;
  Stmt Rewrite(const Stmt& stmt, const ConstraintSet& constraints, OutputSet* output) const final;
  bool CanApply(const Stmt& stmt, const ConstraintSet& constraints) const final {
    static const auto scope_wmma_accumulator = runtime::MemoryScope::Get("wmma.accumulator");
    static const auto scope_shared = runtime::MemoryScope::Get("shared");

    TensorVar src_tensor = constraints.read_region->source.as_or_throw<tvm::tirx::TensorVar>();
    TensorVar tgt_tensor = constraints.write_region->source.as_or_throw<tvm::tirx::TensorVar>();
    return IsCopyBetweenScope(src_tensor, tgt_tensor, scope_wmma_accumulator, scope_shared);
  }
};

/*!
 * \brief Insert a cache stage to the compute location
 * \param stmt the stmt
 * \param is_write_cache whether to write a read cache or write cache
 * \param storage_scope the storage scope of the new cache
 * \param compute_location the compute location.
 * \param outer_loops the outer loops of this stmt
 * \param alloc_tensor the new cache block
 * \return a pair. The first is the stmt after transformation.
 *         The second is the SeqStmt that contains 2 stages (one original and another inserted).
 */
std::pair<Stmt, SeqStmt> InsertCacheStage(Stmt stmt, bool is_write_cache, ffi::String storage_scope,
                                          ffi::Optional<For> compute_location,
                                          const ffi::Array<For>& outer_loops,
                                          TensorVar* alloc_tensor);

}  // namespace s_tir
}  // namespace tvm

#endif  // TVM_S_TIR_TRANSFORM_MEMHAMMER_REWRITE_RULE_H_
