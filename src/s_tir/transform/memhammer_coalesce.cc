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
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/tirx/stmt.h>

#include "./memhammer_rewrite_rule.h"

namespace tvm {
namespace s_tir {
using namespace tvm::prim;
using namespace tvm::tirx;

/*!
 * \brief Fuse consecutive loops
 * \param body the outer-most loop
 * \return the fused loop
 */
Stmt FuseNestLoops(Stmt body) {
  std::vector<const ForObj*> loops;
  while (const ForObj* loop = body.as<ForObj>()) {
    loops.push_back(loop);
    body = loop->body->size() == 1 ? loop->body->seq[0] : loop->body;
  }
  std::string suffix;
  int n = loops.size();
  for (int i = 1; i < n; i++) {
    suffix += "_" + loops[i]->var->name;
  }
  suffix += "_fused";
  PrimVar fused_var = loops[0]->var.CopyWithSuffix(suffix);
  ffi::Map<Var, PrimExpr> subst_map;
  PrimExpr tot = fused_var;
  for (int i = n - 1; i >= 0; i--) {
    subst_map.Set(loops[i]->var,
                  floormod(tot, Range::GetExtentFromBeginEnd(loops[i]->begin, loops[i]->end)));
    tot = floordiv(tot, Range::GetExtentFromBeginEnd(loops[i]->begin, loops[i]->end));
  }
  auto f_substitute = [&](const Var& v) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (auto repl = subst_map.Get(v)) return ffi::Any(*std::move(repl));
    return ffi::Unchanged();
  };
  PrimExpr fused_extent = 1;
  for (int i = 0; i < n; i++) {
    fused_extent *= Range::GetExtentFromBeginEnd(loops[i]->begin, loops[i]->end);
  }
  body = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(std::move(body), f_substitute)
             .as_or_throw<Stmt>();
  return For(fused_var, 0, fused_extent, IntImm(fused_var.ty(), 1), ForKind::kDefault,
             std::move(body));
}

/*!
 * \brief a combination of split, bind, vectorize,
 *        a helper function to perform coalesced load/store
 * \param stmt the stmt to do transformation
 * \param constraints The constraints, including thread extents, vector bytes, and data bits.
 * \return The stmt after transformation
 */
Stmt SplitBindVectorize(const Stmt& stmt, const ConstraintSet& constraints) {
  const ForObj* loop = TVM_TYPE_AS(stmt, ForObj);
  int loop_extent = Range::GetExtentFromBeginEnd(loop->begin, loop->end)
                        .as_or_throw<IntImm>()
                        ->value.as<int>()
                        .value();
  int vector_bytes = constraints.vector_bytes;
  int data_bits = constraints.data_bits;
  int vector_len = std::max(1, vector_bytes * 8 / data_bits);
  int tot_threads = 1;
  // generate thread binding loops
  std::vector<int> factors{-1};
  std::vector<std::string> thread_axis;
  if (ffi::Optional<int64_t> o_t = constraints.thread_extent.Get("threadIdx.z")) {
    int t = o_t.value();
    tot_threads *= t;
    factors.push_back(t);
    thread_axis.push_back("threadIdx.z");
  }
  if (ffi::Optional<int64_t> o_t = constraints.thread_extent.Get("threadIdx.y")) {
    int t = o_t.value();
    tot_threads *= t;
    factors.push_back(t);
    thread_axis.push_back("threadIdx.y");
  }
  if (ffi::Optional<int64_t> o_t = constraints.thread_extent.Get("threadIdx.x")) {
    int t = o_t.value();
    tot_threads *= t;
    factors.push_back(t);
    thread_axis.push_back("threadIdx.x");
  }
  // generate vectorized loop
  factors.push_back(vector_len);
  // generate outer loop
  factors[0] = (loop_extent + tot_threads * vector_len - 1) / (tot_threads * vector_len);
  // create new loop vars
  int n = factors.size();
  std::vector<PrimVar> new_loop_vars;
  new_loop_vars.reserve(n);
  sym::Analyzer analyzer;
  for (int i = 0; i < n; i++) {
    const PrimExpr& factor = factors[i];
    PrimVar var = loop->var.CopyWithSuffix("_" + std::to_string(i));
    {
      auto context = analyzer->GetConstraintContext();
      context->Append(sym::RangeConstraint(var, Range::FromMinExtent(0, factor)));
      analyzer->SyncConstraintContext(context);
    }
    new_loop_vars.push_back(var);
  }
  // substitute fused loop var with new loop vars
  PrimExpr substitute_value = 0;
  for (int i = 0; i < n; i++) {
    substitute_value *= factors[i];
    substitute_value += new_loop_vars[i].as_or_throw<PrimExpr>();
  }
  // Construct the new loop nest
  auto f_substitute =
      [&loop, &substitute_value](const Var& v) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (v.same_as(loop->var)) return ffi::Any(substitute_value);
    return ffi::Unchanged();
  };
  Stmt body =
      ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(loop->body, f_substitute).as_or_throw<Stmt>();
  PrimExpr predicate = substitute_value < Range::GetExtentFromBeginEnd(loop->begin, loop->end);
  if (!analyzer->CanProve(predicate)) {
    body = If(predicate, body);
  }
  body = For(new_loop_vars.back().as_or_throw<PrimVar>(), 0, vector_len,
             IntImm(new_loop_vars.back().as_or_throw<PrimVar>().ty(), 1), ForKind::kVectorized,
             std::move(body));
  for (int i = n - 2; i >= 1; i--) {
    body = For(new_loop_vars[i].as_or_throw<PrimVar>(), 0, factors[i],
               IntImm(new_loop_vars[i].as_or_throw<PrimVar>().ty(), 1), ForKind::kParallel,
               std::move(body),
               {{tvm::s_tir::attr::kThreadBinding.name, ffi::String(thread_axis[i - 1])}});
  }
  return For(new_loop_vars[0].as_or_throw<PrimVar>(), 0, factors[0],
             IntImm(new_loop_vars[0].as_or_throw<PrimVar>().ty(), 1), ForKind::kDefault,
             std::move(body));
}

Stmt CoalescedAccess::Rewrite(const Stmt& stmt, const ConstraintSet& constraints,
                              OutputSet* output) const {
  Stmt after_fuse = FuseNestLoops(stmt);
  Stmt after_split = SplitBindVectorize(std::move(after_fuse), constraints);
  return after_split;
}

/*!
 * \brief Get the index mapping of a specific stmt.
 *        The stmt is like:
 *        for i0:
 *          ...
 *          for in:
 *            A[f(i0, ..., in])] = B[i0, ..., in],
 *        where f is the index mapping we want to get.
 * \param constraints The constraints, including the write region that is required to calculate
 * the index mapping
 * \return The mapping in the form of j0, ..., jm, where j0, ... jm = f(i0, ..., in)
 */
ffi::Array<PrimExpr> GetMapping(const Stmt& stmt, const ConstraintSet& constraints) {
  Stmt body = stmt;
  while (const ForObj* loop = body.as<ForObj>()) {
    body = loop->body->size() == 1 ? loop->body->seq[0] : loop->body;
  }
  const TensorStoreObj* tensor_store = TVM_TYPE_AS(body, TensorStoreObj);
  TensorRegion write_region = constraints.write_region;
  const ffi::Array<PrimExpr>& write_index = tensor_store->indices;
  TVM_FFI_ICHECK(write_region->region.size() == write_index.size() &&
                 write_region->source.as_or_throw<tvm::tirx::TensorVar>().same_as(
                     tensor_store->dest.as_or_throw<TensorVar>()));
  ffi::Array<PrimExpr> result;
  sym::Analyzer analyzer;
  for (int i = 0; i < static_cast<int>(write_region->region.size()); i++) {
    PrimExpr pattern = analyzer->Simplify(write_index[i] - write_region->region[i]->begin);
    if (!IsZero(pattern)) {
      result.push_back(pattern);
    }
  }
  return result;
}

Stmt InverseMapping::Rewrite(const Stmt& stmt, const ConstraintSet& constraints,
                             OutputSet* output) const {
  Stmt body = stmt;
  ffi::Map<PrimVar, Range> var_range;
  ffi::Array<PrimExpr> loop_vars;
  // Step 1. Get index mapping
  ffi::Array<PrimExpr> mapping_pattern = GetMapping(stmt, constraints);
  while (const ForObj* loop = body.as<ForObj>()) {
    var_range.Set(loop->var, Range(loop->begin, loop->end));
    loop_vars.push_back(loop->var);
    body = loop->body->size() == 1 ? loop->body->seq[0] : loop->body;
  }
  // Step 2. Get Inverse mapping
  sym::Analyzer analyzer;
  auto iter_map =
      sym::DetectIterMap(mapping_pattern, var_range, IntImm::Bool(true), sym::Bijective, analyzer);
  TVM_FFI_ICHECK_EQ(iter_map->indices.size(), loop_vars.size());
  ffi::Map<Var, PrimExpr> inverse_mapping = sym::InverseAffineIterMap(iter_map->indices, loop_vars);
  // Step 3. Generate new body
  TensorRegion read_region = constraints.read_region;
  TensorRegion write_region = constraints.write_region;
  ffi::Array<PrimExpr> write_index;
  ffi::Array<PrimExpr> read_index;
  ffi::Array<PrimVar> new_loop_vars;
  ffi::Map<Var, PrimExpr> substitute_map;
  // Step 3.1 construct target tensor indices
  for (int i = 0, j = 0; i < static_cast<int>(write_region->region.size()); i++) {
    if (IsOne(write_region->region[i]->extent)) {
      write_index.push_back(write_region->region[i]->begin);
    } else {
      PrimVar var = loop_vars[j].as_or_throw<PrimVar>().CopyWithSuffix("_inverse");
      new_loop_vars.push_back(var);
      substitute_map.Set(loop_vars[j++].as_or_throw<Var>(), var);
      write_index.push_back(write_region->region[i]->begin + var);
    }
  }
  auto f_substitute =
      [&substitute_map](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (auto repl = substitute_map.Get(var)) return ffi::Any(*std::move(repl));
    return ffi::Unchanged();
  };
  // Step 3.2 construct source tensor indices
  for (int i = 0, j = 0; i < static_cast<int>(read_region->region.size()); i++) {
    if (IsOne(read_region->region[i]->extent)) {
      read_index.push_back(read_region->region[i]->begin);
    } else {
      PrimExpr inverse = inverse_mapping[loop_vars[j++].as_or_throw<Var>()];
      inverse = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(inverse, f_substitute)
                    .as_or_throw<PrimExpr>();
      read_index.push_back(read_region->region[i]->begin + inverse);
    }
  }
  TensorLoad new_tensor_load =
      TensorLoad(read_region->source.as_or_throw<tvm::tirx::TensorVar>(), read_index);
  TensorStore new_tensor_store = TensorStore(
      write_region->source.as_or_throw<tvm::tirx::TensorVar>(), write_index, new_tensor_load);
  Stmt ret = new_tensor_store;
  // Step 3.3 construct loop body
  for (int i = static_cast<int>(new_loop_vars.size()) - 1; i >= 0; i--) {
    PrimExpr extent = write_region->region[i]->extent;
    ret = For(new_loop_vars[i], 0, extent, IntImm(new_loop_vars[i].ty(), 1), ForKind::kDefault,
              std::move(ret));
  }
  return ret;
}
}  // namespace s_tir
}  // namespace tvm
