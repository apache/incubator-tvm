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
#ifndef TVM_S_TIR_SCHEDULE_TRANSFORM_H_
#define TVM_S_TIR_SCHEDULE_TRANSFORM_H_

#include <tvm/ir/prim/expr.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/s_tir/schedule/schedule.h>
#include <tvm/s_tir/schedule/state.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/stmt_functor_with_constraint_context.h>

#include <unordered_map>
#include <utility>

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

/******** Annotation ********/

/*!
 * \brief Create a new block with the given annotation added
 * \param block The block with original annotation
 * \param attr_key The annotation key to be added
 * \param attr_value The annotation value to be added
 * \return A new block with the given annotation as its last annotation
 */
SBlock WithAnnotation(const SBlockObj* block, const ffi::String& attr_key,
                      const ffi::ObjectRef& attr_value);

/******** Tensor Related ********/

/*!
 * \brief Create a new tensor by changing the storage scope.
 * \param tensor The given tensor.
 * \param scope The target storage scope.
 * \return The new tensor with target storage scope.
 */
TensorVar WithScope(const TensorVar& tensor, runtime::MemoryScope scope);

/*!
 * \brief Create a new tensor by changint the data type.
 * \param tensor The given tensor.
 * \param scope The target data type.
 * \return The new tensor with target data type.
 */
TensorVar WithDType(const TensorVar& tensor, PrimType dtype);

/*!
 * \brief Replaces the tensor within the specific sequence of regions
 * \param regions The regions whose tensors are to be replaced
 * \param source The tensor to be replaced
 * \param target The tensor to be replaced to
 * \return The new sequence of regions after replacement
 */
ffi::Array<TensorRegion> ReplaceTensor(ffi::Array<TensorRegion> regions, const TensorVar& source,
                                       const TensorVar& target);

/*!
 * \brief Replaces the tensor within the specific sequence of regions
 * \param regions The regions whose tensors are to be replaced
 * \param tensor_map The mapping from old tensors to new tensors
 * \return The new sequence of regions after replacement
 */
ffi::Array<TensorRegion> ReplaceTensor(ffi::Array<TensorRegion> regions,
                                       const ffi::Map<TensorVar, TensorVar>& tensor_map);

/*!
 * \brief Replaces the tensor within the specific sequence of match_tensors
 * \param match_tensors The match_tensors whose tensors are to be replaced
 * \param source The tensor to be replaced
 * \param target The tensor to be replaced to
 * \return The new sequence of match_tensors after replacement
 */
ffi::Array<MatchTensorRegion> ReplaceTensor(ffi::Array<MatchTensorRegion> match_tensors,
                                            const TensorVar& source, const TensorVar& target);

/*!
 * \brief Replaces the tensor region within the specific sequence of regions
 * \param regions The regions to be replaced
 * \param source_tensor The tensor to whose region is to be replaced
 * \param target The tensor region to be replaced to
 * \return The new sequence of regions after replacement
 */
ffi::Array<TensorRegion> ReplaceTensorRegion(ffi::Array<TensorRegion> regions,
                                             const TensorVar& source_tensor,
                                             const TensorRegion& target);

/*!
 * \brief Replaces the tensor region within the specific sequence of match_tensors
 * \param regions The match_tensors to be replaced
 * \param source_tensor The tensor to whose region is to be replaced
 * \param target The tensor region to be replaced to
 * \return The new sequence of match_tensors after replacement
 */
ffi::Array<MatchTensorRegion> ReplaceTensorRegion(ffi::Array<MatchTensorRegion> match_tensors,
                                                  const TensorVar& source_tensor,
                                                  const TensorRegion& target);

/*!
 * \brief A helper mutator which recursively replaces the old tensor with the new tensor and
 * collects the block sref reuse information for the following replacement.
 *
 * If the tensor to be replaced in used as the source in `match_tensors`, depending the specific
 * use cases, the target tensors in `match_tensors` may also need to be mutated. In this
 * case, this class should be subclassed to explicitly handle `match_tensors`.
 */
class ReplaceTensorMutator : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  /*!
   * \brief The constructor
   * \param old_tensor The old tensor
   * \param new_tensor The new tensor
   * \param block_sref_reuse Optional map to record mapping between old and new blocks that reuse
   *        sref.
   */
  ReplaceTensorMutator(const TensorVar& old_tensor, TensorVar new_tensor,
                       ffi::Map<SBlock, SBlock>* block_sref_reuse);

  ReplaceTensorMutator(const ffi::Map<TensorVar, TensorVar>& tensor_map,
                       ffi::Map<SBlock, SBlock>* block_sref_reuse);

 protected:
  UnchangedOr<Expr> Mutate_(const CallObj* op, InplaceMode inplace_mode) override;

  virtual MatchTensorRegion VisitMatchTensorRegion(const MatchTensorRegion& match_tensor);

  UnchangedOr<Stmt> Mutate_(const SBlockObj* block, InplaceMode inplace_mode) override;

  /*! \brief The block sref reuse map for the following replacement */
  ffi::Map<SBlock, SBlock>* block_sref_reuse_;
};

/******** SBlock Removal ********/

/*!
 * \brief Construct a new AST, with a specific sref tree leaf removed.
 * The leaf's ancestors who have only a single child will be removed too.
 * \param leaf_block_sref The block/loop sref to the sref tree leaf to be removed
 * \param src_stmt The root of the subtree where the replacement begins
 * \param tgt_stmt The root of the subtree after the replacement
 * \return A boolean indicating if the leaf can be removed successfully
 * \note Read before use:
 * 1) Removal is not conducted beyond scope-level.
 * 2) This method only works properly when the scope root is a stage pipeline.
 *
 * An example of the removal plan, say we are removing the leaf block "B" from the AST.
 *
 *  \code
 *    with block([], "scope_root"):
 *        ...
 *        with block([128, 128], "B") as [vi, vj]:
 *            B[vi, vj] = A[vi, vj] + 1.0
 *        with block([128, 128], "C") as [vi, vj]:
 *            C[vi, vj] = B[vi, vj] * 2.0
 *  \endcode
 *
 * Ths method does not mutate the AST, instead it returns the a `(src_stmt, tgt_stmt)` pair as a
 * plan to substitute certain pieces of the IR.
 *
 * In our example, it returns block "scope_root" as `src_stmt`, and the result `tgt_stmt` is:
 *
 *  \code
 *    with block([], "scope_root"):
 *        ...
 *        with block([128, 128], "C") as [vi, vj]:
 *            C[vi, vj] = B[vi, vj] * 2.0
 *  \endcode
 */
void LeafBlockRemovalPlan(const ScheduleState& self, const StmtSRef& leaf_block_sref,
                          Stmt* src_stmt, Stmt* tgt_stmt);

/*!
 * \brief Tile a subset of loops in the block according to the given tensor intrinsic.
 * \param self The schedule to which tiling is applied
 * \param block_rv The block whose subset of loops will be tiled
 * \param intrin_name The name of a tensor intrinsic, must be registerd via
 * TensorIntrin.register(...) beforehand
 * \param allow_padding Whether to allow padding when tiling
 * \return LoopRV corresponding to the outermost loop of a
 * block tiled according to the given intrin, std::nullopt if a valid loop mapping is not found
 */
ffi::Optional<s_tir::LoopRV> TileWithTensorIntrin(const s_tir::Schedule& sch,
                                                  const s_tir::SBlockRV& block_rv,
                                                  const ffi::String& intrin_name,
                                                  bool allow_padding = false);

/******** SBlock mutation ********/

/*!
 * \brief Simplifier for indices of tensor access and block tensor access regions.
 */
class BlockTensorAccessSimplifier : public s_tir::StmtExprMutatorWithConstraintContext {
 public:
  using s_tir::StmtExprMutatorWithConstraintContext::Mutate;
  using s_tir::StmtExprMutatorWithConstraintContext::Mutate_;

  /*!
   * \brief Simplify indices of tensor access and block tensor access regions in the statement
   * \param stmt The statement to be simplified
   * \param analyzer The arithmetic analyzer
   * \return The simplified statement
   */
  static Stmt Simplify(const Stmt& stmt, const sym::Analyzer& analyzer) {
    auto simplifier = ffi::make_object<BlockTensorAccessSimplifier>(analyzer);
    return simplifier->Mutate(stmt).ValueOrUnchanged(stmt);
  }

  explicit BlockTensorAccessSimplifier(const sym::Analyzer& analyzer)
      : StmtExprMutatorWithConstraintContext(analyzer->GetConstraintContext()) {}

 private:
  void SimplifyAccessRegion(ffi::Array<TensorRegion>* old_access_regions);
  void SimplifyTensorIndices(ffi::Array<PrimExpr>* indices);

  UnchangedOr<Stmt> Mutate_(const SBlockObj* op, InplaceMode inplace_mode) final;
  UnchangedOr<Stmt> Mutate_(const TensorStoreObj* op, InplaceMode inplace_mode) final;
  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* op, InplaceMode inplace_mode) final;
};

}  // namespace s_tir
}  // namespace tvm

#endif  // TVM_S_TIR_SCHEDULE_TRANSFORM_H_
