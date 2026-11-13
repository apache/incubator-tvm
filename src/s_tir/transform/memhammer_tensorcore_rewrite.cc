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
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/type.h>

#include "./memhammer_rewrite_rule.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

/*!
 * \brief Tile the 2 innermost loops to extent=16. This helps further tensor core rewrite.
 * \param stmt The stmt
 * \return A pair. The first is the stmt after transformation.
 *         The second is the compute location where we may add write cache.
 */
std::pair<Stmt, ffi::Optional<For>> TileWmmaBlock(Stmt stmt) {
  Stmt body = stmt;
  std::vector<const ForObj*> loops;
  while (const ForObj* loop = body.as<ForObj>()) {
    loops.push_back(loop);
    body = loop->body->size() == 1 ? loop->body->seq[0] : loop->body;
  }
  int n = loops.size();
  PrimExpr extent_last1 = Range::GetExtentFromBeginEnd(loops[n - 1]->begin, loops[n - 1]->end);
  PrimExpr extent_last2 = Range::GetExtentFromBeginEnd(loops[n - 2]->begin, loops[n - 2]->end);
  {
    sym::Analyzer analyzer;
    if (!analyzer->CanProveEqual(floormod(extent_last1, 16), 0) ||
        !analyzer->CanProveEqual(floormod(extent_last2, 16), 0)) {
      return std::make_pair(stmt, std::nullopt);
    }
  }
  Var new_loop_vars[4] = {
      /*0:*/ loops[n - 2]->var.CopyWithSuffix("_0"),
      /*1:*/ loops[n - 1]->var.CopyWithSuffix("_0"),
      /*2:*/ loops[n - 2]->var.CopyWithSuffix("_1"),
      /*3:*/ loops[n - 1]->var.CopyWithSuffix("_1"),
  };
  ffi::Map<Var, PrimExpr> loop_var_map{
      {loops[n - 2]->var,
       new_loop_vars[0].as_or_throw<PrimExpr>() * 16 + new_loop_vars[2].as_or_throw<PrimExpr>()},
      {loops[n - 1]->var,
       new_loop_vars[1].as_or_throw<PrimExpr>() * 16 + new_loop_vars[3].as_or_throw<PrimExpr>()},
  };
  auto f_substitute = [&loop_var_map](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (auto repl = loop_var_map.Get(var)) return ffi::Any(*std::move(repl));
    return ffi::Unchanged();
  };
  body = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(std::move(body), f_substitute)
             .as_or_throw<Stmt>();
  {
    PrimExpr factor[4] = {
        /*0:*/ floordiv(extent_last2, 16),  //
        /*1:*/ floordiv(extent_last1, 16),  //
        /*3:*/ 16,                          //
        /*4:*/ 16,                          //
    };
    body = For(new_loop_vars[3].as_or_throw<PrimVar>(), 0, factor[3],
               IntImm(new_loop_vars[3].as_or_throw<PrimVar>().ty(), 1), ForKind::kDefault,
               std::move(body));
    body = For(new_loop_vars[2].as_or_throw<PrimVar>(), 0, factor[2],
               IntImm(new_loop_vars[2].as_or_throw<PrimVar>().ty(), 1), ForKind::kDefault,
               std::move(body));
    body = For(new_loop_vars[1].as_or_throw<PrimVar>(), 0, factor[1],
               IntImm(new_loop_vars[1].as_or_throw<PrimVar>().ty(), 1), ForKind::kDefault,
               std::move(body));
    body = For(new_loop_vars[0].as_or_throw<PrimVar>(), 0, factor[0],
               IntImm(new_loop_vars[0].as_or_throw<PrimVar>().ty(), 1), ForKind::kDefault,
               std::move(body));
  }
  For compute_location = body.as_or_throw<For>();
  for (int i = n - 3; i >= 0; i--) {
    auto new_loop = ffi::GetRef<For>(loops[i]);
    new_loop.CopyOnWrite()->body = std::move(body);
    body = new_loop;
  }
  return {body, compute_location};
}

ffi::Array<Range> RelaxIndices(const ffi::Array<PrimExpr>& indices,
                               const ffi::Array<PrimExpr>& shape,
                               const ffi::Map<Var, sym::IntSet>& var_dom) {
  ffi::Array<sym::IntSet> int_set;
  int_set.reserve(indices.size());
  for (auto& indice : indices) {
    int_set.push_back(sym::EvalSet(indice, var_dom));
  }
  int ndim = int_set.size();
  ffi::Array<Range> region;
  region.reserve(ndim);
  for (int i = 0; i < ndim; ++i) {
    region.push_back(int_set[i].CoverRange(Range::FromMinExtent(0, shape[i])).value());
  }
  return region;
}

/*!
 * \brief Rewrite the data copy that stores to wmma fragment with wmma::load_matrix_sync
 * \param stmt The stmt to rewrite
 * \return The stmt after rewrite
 */
Stmt RewriteWmmaLoad(Stmt stmt) {
  static const auto scope_wmma_matrix_a = runtime::MemoryScope::Get("wmma.matrix_a");

  using sym::IntSet;
  const PrimType dtype_ty = PrimType::Float(16);
  const PrimType& dtype = dtype_ty;
  const PrimType int32_ty = PrimType::Int(32);

  Stmt body = stmt;
  std::vector<const ForObj*> loops;
  while (const ForObj* loop = body.as<ForObj>()) {
    loops.push_back(loop);
    body = loop->body->size() == 1 ? loop->body->seq[0] : loop->body;
  }
  int n = loops.size();

  ffi::Map<Var, IntSet> var_dom{
      {loops[n - 1]->var,
       IntSet::FromMinExtent(loops[n - 1]->begin,
                             Range::GetExtentFromBeginEnd(loops[n - 1]->begin, loops[n - 1]->end))},
      {loops[n - 2]->var,
       IntSet::FromMinExtent(loops[n - 2]->begin,
                             Range::GetExtentFromBeginEnd(loops[n - 2]->begin, loops[n - 2]->end))},
  };
  // TODO(tian): the assumption that the RHS of TensorStore is TensorLoad may not be accurate
  const TensorStoreObj* tensor_store = TVM_TYPE_AS(body, TensorStoreObj);
  const TensorLoadObj* tensor_load = TVM_TYPE_AS(tensor_store->value, TensorLoadObj);

  TensorVar src_tensor = tensor_load->source.as_or_throw<tvm::tirx::TensorVar>();
  TensorVar tgt_tensor = tensor_store->dest.as_or_throw<TensorVar>();
  std::string layout = tgt_tensor.scope() == scope_wmma_matrix_a ? "row_major" : "col_major";
  static const auto memory_axis = tvm::tirx::Axis::Get("m");
  TensorVar new_src_tensor(
      "src",
      TensorType({IntImm::Int32(16), IntImm::Int32(16)}, dtype, src_tensor.scope(), 64,
                 tvm::tirx::TileLayout(
                     {tvm::tirx::Iter(IntImm::Int32(16), PrimVar("s1", int32_ty), memory_axis),
                      tvm::tirx::Iter(IntImm::Int32(16), PrimVar("s0", int32_ty), memory_axis)},
                     {}, {{memory_axis, 16 * PrimVar("src_elem_offset", int32_ty)}})));
  TensorVar new_tgt_tensor(
      "tgt", TensorType({IntImm::Int32(16), IntImm::Int32(16)}, dtype, tgt_tensor.scope(), 64,
                        tvm::tirx::TileLayout(
                            {tvm::tirx::Iter(IntImm::Int32(16), IntImm::Int32(16), memory_axis),
                             tvm::tirx::Iter(IntImm::Int32(16), IntImm::Int32(1), memory_axis)},
                            {}, {{memory_axis, 16 * PrimVar("tgt_elem_offset", int32_ty)}})));

  ffi::Array<Range> read_region = RelaxIndices(tensor_load->indices, src_tensor.shape(), var_dom);
  ffi::Array<Range> write_region = RelaxIndices(tensor_store->indices, tgt_tensor.shape(), var_dom);
  static const Op gpu_load_matrix_sync_op = Op::Get("tirx.gpu_load_matrix_sync");
  Stmt wmma_body = SBlockRealize(
      /*iter_values=*/{},
      /*predicate=*/IntImm::Bool(true),
      SBlock(
          /*iter_vars=*/{},
          /*reads=*/{TensorRegion(src_tensor, read_region, TensorRegionType())},
          /*writes=*/{TensorRegion(tgt_tensor, write_region, TensorRegionType())},
          /*name_hint=*/"wmma_load",
          /*body=*/
          Evaluate(Call(
              /*data=*/PrimType::Void(),
              /*op=*/gpu_load_matrix_sync_op,
              ffi::Array<Expr>{
                  /*0:*/ new_tgt_tensor.data(),
                  /*1:*/ PrimExpr(16),
                  /*2:*/ PrimExpr(16),
                  /*3:*/ PrimExpr(16),
                  /*4:*/
                  floordiv(tvm::tirx::MatchShapeStridePattern(new_tgt_tensor.ty()).value().get<2>(),
                           256) +
                      floordiv(floormod(tvm::tirx::MatchShapeStridePattern(new_tgt_tensor.ty())
                                            .value()
                                            .get<2>(),
                                        256),
                               16),
                  /*5:*/
                  Call(new_src_tensor.data()->ty, tirx::address_of_op(),
                       {TensorLoad(new_src_tensor, {PrimExpr(0), PrimExpr(0)})}),
                  /*6:*/
                  tvm::tirx::MatchShapeStridePattern(new_src_tensor.ty())
                      .value()
                      .get<1>()[tvm::tirx::MatchShapeStridePattern(new_src_tensor.ty())
                                    .value()
                                    .get<1>()
                                    .size() -
                                2],
                  /*7:*/ StringImm(layout),
              })),
          /*init=*/std::nullopt,
          /*alloc_tensors=*/{},
          /*match_tensors=*/
          {
              /*0:*/ MatchTensorRegion(new_src_tensor,
                                       TensorRegion(src_tensor, read_region, TensorRegionType())),
              /*1:*/
              MatchTensorRegion(new_tgt_tensor,
                                TensorRegion(tgt_tensor, write_region, TensorRegionType())),
          },
          /*annotations=*/{}));
  for (int i = n - 3; i >= 0; i--) {
    auto new_loop = ffi::GetRef<For>(loops[i]);
    new_loop.CopyOnWrite()->body = std::move(wmma_body);
    wmma_body = new_loop;
  }
  return wmma_body;
}

/*!
 * \brief Rewrite the data copy that loads from wmma fragment with wmma::store_matrix_sync
 * \param stmt The stmt to rewrite
 * \return The stmt after rewrite
 */
Stmt RewriteWmmaStore(Stmt stmt) {
  static const auto scope_wmma_accumulator = runtime::MemoryScope::Get("wmma.accumulator");

  using sym::IntSet;
  const PrimType int32_ty = PrimType::Int(32);

  Stmt body = stmt;
  std::vector<const ForObj*> loops;
  while (const ForObj* loop = body.as<ForObj>()) {
    loops.push_back(loop);
    body = loop->body->size() == 1 ? loop->body->seq[0] : loop->body;
  }
  int n = loops.size();

  ffi::Map<Var, IntSet> var_dom{
      {loops[n - 1]->var,
       IntSet::FromMinExtent(loops[n - 1]->begin,
                             Range::GetExtentFromBeginEnd(loops[n - 1]->begin, loops[n - 1]->end))},
      {loops[n - 2]->var,
       IntSet::FromMinExtent(loops[n - 2]->begin,
                             Range::GetExtentFromBeginEnd(loops[n - 2]->begin, loops[n - 2]->end))},
  };
  // TODO(tian): the assumption that the RHS of TensorStore is TensorLoad may not be accurate
  const TensorStoreObj* tensor_store = TVM_TYPE_AS(body, TensorStoreObj);
  const TensorLoadObj* tensor_load = nullptr;
  auto walk_fn = [&](const TensorLoad& load) -> ffi::Expected<ffi::WalkResult> {
    if (load->source.as_or_throw<tvm::tirx::TensorVar>().scope() == scope_wmma_accumulator) {
      TVM_FFI_ICHECK(tensor_load == nullptr ||
                     tensor_load->source.as_or_throw<tvm::tirx::TensorVar>().same_as(
                         load->source.as_or_throw<tvm::tirx::TensorVar>()))
          << "More than one source tensor of wmma accumulator found";
      tensor_load = load.get();
    }
    return ffi::WalkResult::Advance();
  };
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(tensor_store->value, walk_fn);
  TensorVar src_tensor = tensor_load->source.as_or_throw<tvm::tirx::TensorVar>();
  TensorVar tgt_tensor = tensor_store->dest.as_or_throw<TensorVar>();

  PrimType dtype_ty = src_tensor.dtype();
  const PrimType& dtype = dtype_ty;

  static const auto memory_axis = tvm::tirx::Axis::Get("m");
  TensorVar new_src_tensor(
      "src", TensorType({IntImm::Int32(16), IntImm::Int32(16)}, dtype, src_tensor.scope(), 64,
                        tvm::tirx::TileLayout(
                            {tvm::tirx::Iter(IntImm::Int32(16), IntImm::Int32(16), memory_axis),
                             tvm::tirx::Iter(IntImm::Int32(16), IntImm::Int32(1), memory_axis)},
                            {}, {{memory_axis, 16 * PrimVar("src_elem_offset", int32_ty)}})));
  TensorVar new_tgt_tensor(
      "tgt",
      TensorType({IntImm::Int32(16), IntImm::Int32(16)}, dtype, tgt_tensor.scope(), 64,
                 tvm::tirx::TileLayout(
                     {tvm::tirx::Iter(IntImm::Int32(16), PrimVar("s1", int32_ty), memory_axis),
                      tvm::tirx::Iter(IntImm::Int32(16), PrimVar("s0", int32_ty), memory_axis)},
                     {}, {{memory_axis, 16 * PrimVar("tgt_elem_offset", int32_ty)}})));

  ffi::Array<Range> read_region = RelaxIndices(tensor_load->indices, src_tensor.shape(), var_dom);
  ffi::Array<Range> write_region = RelaxIndices(tensor_store->indices, tgt_tensor.shape(), var_dom);
  static const Op gpu_store_matrix_sync_op = Op::Get("tirx.gpu_store_matrix_sync");
  Stmt wmma_body = SBlockRealize(
      /*iter_values=*/{},  //
      /*predicate=*/IntImm::Bool(true),
      SBlock(
          /*iter_vars=*/{},
          /*reads=*/{TensorRegion(src_tensor, read_region, TensorRegionType())},
          /*writes=*/{TensorRegion(tgt_tensor, write_region, TensorRegionType())},
          /*name_hint=*/"wmma_store",
          Evaluate(Call(
              /*data=*/PrimType::Void(),
              /*op=*/gpu_store_matrix_sync_op,
              ffi::Array<Expr>{
                  /*0:*/ new_src_tensor.data(),
                  /*1:*/ PrimExpr(16),
                  /*2:*/ PrimExpr(16),
                  /*3:*/ PrimExpr(16),
                  /*4:*/
                  floordiv(tvm::tirx::MatchShapeStridePattern(new_src_tensor.ty()).value().get<2>(),
                           256) +
                      floordiv(floormod(tvm::tirx::MatchShapeStridePattern(new_src_tensor.ty())
                                            .value()
                                            .get<2>(),
                                        256),
                               16),
                  /*5:*/
                  Call(new_tgt_tensor.data()->ty, tirx::address_of_op(),
                       {TensorLoad(new_tgt_tensor, {PrimExpr(0), PrimExpr(0)})}),
                  /*6:*/
                  tvm::tirx::MatchShapeStridePattern(new_tgt_tensor.ty()).value().get<1>()[0],
                  /*7:*/ StringImm("row_major")})),
          /*init=*/std::nullopt,
          /*alloc_tensors=*/{},
          /*match_tensors=*/
          {
              MatchTensorRegion(new_src_tensor,
                                TensorRegion(src_tensor, read_region, TensorRegionType())),
              MatchTensorRegion(new_tgt_tensor,
                                TensorRegion(tgt_tensor, write_region, TensorRegionType())),
          },
          /*annotations=*/{}));
  for (int i = n - 3; i >= 0; i--) {
    auto new_loop = ffi::GetRef<For>(loops[i]);
    new_loop.CopyOnWrite()->body = std::move(wmma_body);
    wmma_body = new_loop;
  }
  return wmma_body;
}

Stmt SharedToWmma::Rewrite(const Stmt& stmt, const ConstraintSet& constraints,
                           OutputSet* output) const {
  Stmt after_tiling = TileWmmaBlock(stmt).first;
  output->padding_min.Set(constraints.read_region->source.as_or_throw<tvm::tirx::TensorVar>(), 8);
  return RewriteWmmaLoad(after_tiling);
}

Stmt WmmaToShared::Rewrite(const Stmt& stmt, const ConstraintSet& constraints,
                           OutputSet* output) const {
  Stmt after_tiling = TileWmmaBlock(stmt).first;
  output->padding_min.Set(constraints.write_region->source.as_or_throw<tvm::tirx::TensorVar>(), 8);
  return RewriteWmmaStore(after_tiling);
}

class WmmaToGlobalRewriter : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  WmmaToGlobalRewriter(const SeqStmtObj* tgt_stmt, const ConstraintSet& constraints)
      : tgt_stmt_(tgt_stmt), constraints_(constraints) {}

 private:
  UnchangedOr<Stmt> Mutate_(const SeqStmtObj* op, InplaceMode inplace_mode) final {
    if (op == tgt_stmt_) {
      TVM_FFI_ICHECK_EQ(op->seq.size(), 2);
      Stmt wmma_to_shared = RewriteWmmaStore(op->seq[0]);
      Stmt shared_to_global = CoalescedAccess().Rewrite(op->seq[1], constraints_, nullptr);
      return SeqStmt({wmma_to_shared, shared_to_global});
    } else {
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }
  }

  const SeqStmtObj* tgt_stmt_;
  const ConstraintSet& constraints_;
};

Stmt WmmaToGlobal::Rewrite(const Stmt& stmt, const ConstraintSet& constraints,
                           OutputSet* output) const {
  auto [body, compute_location] = TileWmmaBlock(stmt);
  SeqStmt seq{ffi::UnsafeInit{}};
  TensorVar cache_tensor{ffi::UnsafeInit{}};
  // Step 1. add a shared memory cache
  std::tie(body, seq) = InsertCacheStage(std::move(body), true, "shared.dyn", compute_location,
                                         constraints.outer_loops, &cache_tensor);
  output->alloc_tensor.push_back(cache_tensor);
  output->padding_min.Set(cache_tensor, 8);
  // Step 2. do coalesced rewrite and tensor core rewrite respectively for 2 parts
  auto rewriter = ffi::make_object<WmmaToGlobalRewriter>(seq.get(), constraints);
  return rewriter->Mutate(body).ValueOrUnchanged(body);
}

std::pair<Stmt, ffi::Optional<For>> TileMmaToGlobalBlock(Stmt stmt) {
  // i, j = sch.get_loops(block)[2:]
  // i_0, i_1 = sch.split(i, factors=[None, 8])
  // j_0, j_1 = sch.split(j, factors=[None, 8])
  // sch.reorder(i_0, j_0, i_1, j_1)
  Stmt body = stmt;
  std::vector<const ForObj*> loops;
  while (const ForObj* loop = body.as<ForObj>()) {
    loops.push_back(loop);
    body = loop->body->size() == 1 ? loop->body->seq[0] : loop->body;
  }
  int n = loops.size();
  PrimExpr extent_last1 = Range::GetExtentFromBeginEnd(loops[n - 1]->begin, loops[n - 1]->end);
  PrimExpr extent_last2 = Range::GetExtentFromBeginEnd(loops[n - 2]->begin, loops[n - 2]->end);
  {
    sym::Analyzer analyzer;
    // Only tile when both extent % 8 == 0
    if (!analyzer->CanProveEqual(floormod(extent_last1, 8), 0) ||
        !analyzer->CanProveEqual(floormod(extent_last2, 8), 0)) {
      return std::make_pair(stmt, std::nullopt);
    }
  }
  Var new_loop_vars[4] = {
      /*0:*/ loops[n - 2]->var.CopyWithSuffix("_0"),
      /*1:*/ loops[n - 1]->var.CopyWithSuffix("_0"),
      /*2:*/ loops[n - 2]->var.CopyWithSuffix("_1"),
      /*3:*/ loops[n - 1]->var.CopyWithSuffix("_1"),
  };
  ffi::Map<Var, PrimExpr> loop_var_map{
      {loops[n - 2]->var,
       new_loop_vars[0].as_or_throw<PrimExpr>() * 8 + new_loop_vars[2].as_or_throw<PrimExpr>()},
      {loops[n - 1]->var,
       new_loop_vars[1].as_or_throw<PrimExpr>() * 8 + new_loop_vars[3].as_or_throw<PrimExpr>()},
  };
  auto f_substitute = [&loop_var_map](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (auto repl = loop_var_map.Get(var)) return ffi::Any(*std::move(repl));
    return ffi::Unchanged();
  };
  body = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(std::move(body), f_substitute)
             .as_or_throw<Stmt>();
  {
    PrimExpr factor[4] = {
        /*0:*/ floordiv(extent_last2, 8),  //
        /*1:*/ floordiv(extent_last1, 8),  //
        /*3:*/ 8,                          //
        /*4:*/ 8,                          //
    };
    body = For(new_loop_vars[3].as_or_throw<PrimVar>(), 0, factor[3],
               IntImm(new_loop_vars[3].as_or_throw<PrimVar>().ty(), 1), ForKind::kDefault,
               std::move(body));
    body = For(new_loop_vars[2].as_or_throw<PrimVar>(), 0, factor[2],
               IntImm(new_loop_vars[2].as_or_throw<PrimVar>().ty(), 1), ForKind::kDefault,
               std::move(body));
    body = For(new_loop_vars[1].as_or_throw<PrimVar>(), 0, factor[1],
               IntImm(new_loop_vars[1].as_or_throw<PrimVar>().ty(), 1), ForKind::kDefault,
               std::move(body));
    body = For(new_loop_vars[0].as_or_throw<PrimVar>(), 0, factor[0],
               IntImm(new_loop_vars[0].as_or_throw<PrimVar>().ty(), 1), ForKind::kDefault,
               std::move(body));
  }
  For compute_location = body.as_or_throw<For>();
  for (int i = n - 3; i >= 0; i--) {
    auto new_loop = ffi::GetRef<For>(loops[i]);
    new_loop.CopyOnWrite()->body = std::move(body);
    body = new_loop;
  }
  return {body, compute_location};
}

/*!
 * \brief Rewrite the data copy that loads from mma fragment
 * \param stmt The stmt to rewrite
 * \return The stmt after rewrite
 */
Stmt RewriteMmaStore(Stmt stmt) {
  static const auto scope_mma_matrixC = runtime::MemoryScope::Get("mma.matrixC");

  using sym::IntSet;
  const PrimType int32_ty = PrimType::Int(32);

  // Step 1. Get inner loop body
  Stmt body = stmt;
  std::vector<const ForObj*> loops;
  while (const ForObj* loop = body.as<ForObj>()) {
    loops.push_back(loop);
    body = loop->body->size() == 1 ? loop->body->seq[0] : loop->body;
  }
  int n = loops.size();

  ffi::Map<Var, IntSet> var_dom{
      {loops[n - 1]->var,
       IntSet::FromMinExtent(loops[n - 1]->begin,
                             Range::GetExtentFromBeginEnd(loops[n - 1]->begin, loops[n - 1]->end))},
      {loops[n - 2]->var,
       IntSet::FromMinExtent(loops[n - 2]->begin,
                             Range::GetExtentFromBeginEnd(loops[n - 2]->begin, loops[n - 2]->end))},
  };

  // Step 2. Find matrixC tensor
  const TensorStoreObj* tensor_store = TVM_TYPE_AS(body, TensorStoreObj);
  const TensorLoadObj* tensor_load = nullptr;
  auto walk_fn = [&](const TensorLoad& load) -> ffi::Expected<ffi::WalkResult> {
    if (load->source.as_or_throw<tvm::tirx::TensorVar>().scope() == scope_mma_matrixC) {
      TVM_FFI_ICHECK(tensor_load == nullptr ||
                     tensor_load->source.as_or_throw<tvm::tirx::TensorVar>().same_as(
                         load->source.as_or_throw<tvm::tirx::TensorVar>()))
          << "More than one source tensor of mma accumulator found";
      tensor_load = load.get();
    }
    return ffi::WalkResult::Advance();
  };
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(tensor_store->value, walk_fn);

  // Step 3. Create new mma body
  // We have the assumption that two innermost loops are the 8 * 8 loop generated by
  // TileMmaToGlobalBlock. Here, we rewrite the 8 * 8 loop to have the threadIdx.x
  // binding so that each thread only read from their part of tensor, and write to
  // corresponding place in shared memory.
  // Please refer to `9.7.13.4.7. Matrix Fragments for mma.m16n8k8` in
  // https://docs.nvidia.com/cuda/archive/11.1.0/pdf/ptx_isa_7.1.pdf

  // Step 3.1. Generate new tensor
  TensorVar src_tensor = tensor_load->source.as_or_throw<tvm::tirx::TensorVar>();
  TensorVar tgt_tensor = tensor_store->dest.as_or_throw<TensorVar>();
  PrimType dtype_ty = src_tensor.dtype();
  const PrimType& dtype = dtype_ty;
  static const auto memory_axis = tvm::tirx::Axis::Get("m");
  TensorVar new_src_tensor(
      "src", TensorType({IntImm::Int32(8), IntImm::Int32(8)}, dtype, src_tensor.scope(), 64,
                        tvm::tirx::TileLayout(
                            {tvm::tirx::Iter(IntImm::Int32(8), IntImm::Int32(8), memory_axis),
                             tvm::tirx::Iter(IntImm::Int32(8), IntImm::Int32(1), memory_axis)},
                            {}, {{memory_axis, 8 * PrimVar("src_elem_offset", int32_ty)}})));
  TensorVar new_tgt_tensor(
      "tgt",
      TensorType({IntImm::Int32(8), IntImm::Int32(8)}, dtype, tgt_tensor.scope(), 64,
                 tvm::tirx::TileLayout(
                     {tvm::tirx::Iter(IntImm::Int32(8), PrimVar("s1", int32_ty), memory_axis),
                      tvm::tirx::Iter(IntImm::Int32(8), PrimVar("s0", int32_ty), memory_axis)},
                     {}, {{memory_axis, 8 * PrimVar("tgt_elem_offset", int32_ty)}})));

  // Step 3.2. Generate new r/w region
  ffi::Array<Range> read_region = RelaxIndices(tensor_load->indices, src_tensor.shape(), var_dom);
  ffi::Array<Range> write_region = RelaxIndices(tensor_store->indices, tgt_tensor.shape(), var_dom);

  // Step 3.3. Generate new inner loop body
  // for v in T.vectorized(2):
  //   tgt[tx // 4, (tx % 4) * 2 + vec] = src[tx // 4, (tx % 4) * 2 + vec]
  PrimVar tx("tx");
  PrimVar vec("vec");
  Stmt mma_body = SBlockRealize(
      /*iter_values=*/{},  //
      /*predicate=*/IntImm::Bool(true),
      SBlock(
          /*iter_vars=*/{},
          /*reads=*/{TensorRegion(src_tensor, read_region, TensorRegionType())},
          /*writes=*/{TensorRegion(tgt_tensor, write_region, TensorRegionType())},
          /*name_hint=*/"mma_store",
          RegionStmt(tirx::launch_thread_op(), {StringImm("threadIdx.x"), IntImm::Int32(32)}, {tx},
                     DictAttrs(), /*body=*/
                     For(vec.as_or_throw<PrimVar>(), 0, 2,
                         IntImm(vec.as_or_throw<PrimVar>().ty(), 1), ForKind::kVectorized,
                         TensorStore(new_tgt_tensor, {floordiv(tx, 4), floormod(tx, 4) * 2 + vec},
                                     TensorLoad(new_src_tensor,
                                                {floordiv(tx, 4), floormod(tx, 4) * 2 + vec})))),
          /*init=*/std::nullopt,
          /*alloc_tensors=*/{},
          /*match_tensors=*/
          {
              MatchTensorRegion(new_src_tensor,
                                TensorRegion(src_tensor, read_region, TensorRegionType())),
              MatchTensorRegion(new_tgt_tensor,
                                TensorRegion(tgt_tensor, write_region, TensorRegionType())),
          },
          /*annotations=*/{}));

  // Step 3.4. wrap outer loops
  for (int i = n - 3; i >= 0; i--) {
    auto new_loop = ffi::GetRef<For>(loops[i]);
    new_loop.CopyOnWrite()->body = std::move(mma_body);
    mma_body = new_loop;
  }
  return mma_body;
}

class MmaToGlobalRewriter : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  MmaToGlobalRewriter(const SeqStmtObj* tgt_stmt, const ConstraintSet& constraints)
      : tgt_stmt_(tgt_stmt), constraints_(constraints) {}

 private:
  UnchangedOr<Stmt> Mutate_(const SeqStmtObj* op, InplaceMode inplace_mode) final {
    if (op == tgt_stmt_) {
      TVM_FFI_ICHECK_EQ(op->seq.size(), 2);
      // Rewrite for local to shared.dyn
      // In this rewrite, we store local matrixC tensor to corresponding place in shared memory
      Stmt mma_to_shared = RewriteMmaStore(op->seq[0]);
      // Coalesce access for shared.dyn to global
      Stmt shared_to_global = CoalescedAccess().Rewrite(op->seq[1], constraints_, nullptr);
      return SeqStmt({mma_to_shared, shared_to_global});
    } else {
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }
  }

  const SeqStmtObj* tgt_stmt_;
  const ConstraintSet& constraints_;
};

Stmt MmaToGlobal::Rewrite(const Stmt& stmt, const ConstraintSet& constraints,
                          OutputSet* output) const {
  auto [body, compute_location] = TileMmaToGlobalBlock(stmt);
  SeqStmt seq{ffi::UnsafeInit{}};
  TensorVar cache_tensor{ffi::UnsafeInit{}};
  // Step 1. add a shared memory cache
  std::tie(body, seq) = InsertCacheStage(std::move(body), true, "shared.dyn", compute_location,
                                         constraints.outer_loops, &cache_tensor);
  output->alloc_tensor.push_back(cache_tensor);
  output->padding_min.Set(cache_tensor, 8);
  // Step 2. do coalesced rewrite and tensor core rewrite respectively for 2 parts
  auto rewriter = ffi::make_object<MmaToGlobalRewriter>(seq.get(), constraints);
  return rewriter->Mutate(body).ValueOrUnchanged(body);
}

}  // namespace s_tir
}  // namespace tvm
