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
#include <tvm/tirx/type.h>

#include "../utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::prim;
using namespace tvm::tirx;

/*!
 * \brief Auxiliary class that collects the IterSplitExpr in the indexing pattern
 * to help decision making in layout transformation
 */
class SplitExprCollector {
 public:
  /*!
   * \brief The corresponding IterSplitExpr, simplified for our case
   * The pattern is `source // lower_factor % extent * scale`
   */
  struct SplitExpr {
    /*! \brief The source variable */
    PrimVar source;
    /*! \brief The lower factor of the split expression */
    int64_t lower_factor;
    /*! \brief The extent of the split expression */
    int64_t extent;
  };

  /*!
   * \brief Collect the split expressions in the indexing pattern
   * \param index The indexing pattern
   * \param input_iters The input iterators' domain
   * \param predicate The predicate of the affine map
   * \param check_level The iter mapping checking level
   * \param analyzer The analyzer
   * \return The collected split expressions
   */
  static std::vector<SplitExpr> Collect(const PrimExpr& index,
                                        const ffi::Map<PrimVar, Range>& input_iters,  //
                                        const PrimExpr& predicate,                    //
                                        sym::IterMapLevel check_level,                //
                                        sym::AnalyzerObj* analyzer) {
    sym::Analyzer analyzer_ref = ffi::GetRef<sym::Analyzer>(analyzer);
    sym::IterMapResult res = sym::DetectIterMap({analyzer->Simplify(index)}, input_iters, predicate,
                                                check_level, analyzer_ref);
    const auto& iter_sum_exprs = res->indices;
    if (iter_sum_exprs.empty()) {
      return {};
    }
    TVM_FFI_ICHECK_EQ(iter_sum_exprs.size(), 1);
    if (iter_sum_exprs[0]->args.size() == 0) {
      return {};
    }
    SplitExprCollector collector;
    collector.Visit(iter_sum_exprs[0]);
    if (collector.failed_) {
      return {};
    }
    return std::move(collector.exprs_);
  }

 private:
  void Visit(const sym::IterSplitExpr& expr) {
    if (auto var = expr->source->source.as<PrimVar>()) {
      const auto* lower_factor_imm = expr->lower_factor.as<IntImmObj>();
      auto lower_factor = lower_factor_imm ? lower_factor_imm->value.as<int64_t>() : std::nullopt;
      const auto* extent_imm = expr->extent.as<IntImmObj>();
      auto extent = extent_imm ? extent_imm->value.as<int64_t>() : std::nullopt;
      if (!lower_factor.has_value() || !extent.has_value()) {
        failed_ = true;
        return;
      }
      exprs_.push_back(SplitExpr{var.value(), *lower_factor, *extent});
    } else if (auto iter_sum_expr = expr->source->source.as<sym::IterSumExpr>()) {
      Visit(iter_sum_expr.value());
    } else {
      TVM_FFI_ICHECK(false) << "Unexpected type: " << expr->source->source->GetTypeKey();
    }
  }

  void Visit(const sym::IterSumExpr& expr) {
    for (const sym::IterSplitExpr& arg : expr->args) {
      Visit(arg);
    }
  }

  /*! \brief Whether the analysis failed */
  bool failed_ = false;
  /*! \brief The collected split expressions */
  std::vector<SplitExpr> exprs_;
};

ffi::Optional<IndexMap> SuggestIndexMap(const TensorVar& tensor,
                                        const ffi::Array<PrimExpr>& indices,
                                        const ffi::Array<For>& loops, const PrimExpr& predicate,
                                        sym::AnalyzerObj* analyzer) {
  int ndim = tensor.shape().size();
  int n_loops = loops.size();
  // Step 1. Collect the domains and indices of loop variables
  ffi::Map<PrimVar, Range> input_iters;
  std::unordered_map<const VarObj*, int> var2id;
  var2id.reserve(n_loops);
  for (int i = 0; i < n_loops; ++i) {
    const For& loop = loops[i];
    input_iters.Set(loop->var, Range(loop->begin, loop->end));
    var2id.emplace(loop->var.get(), i);
  }
  // Step 2. Calculate a functor that flattens a multi-dimensional index
  auto f_flatten_index =
      [ndim, strides = tvm::tirx::MatchShapeStridePattern(tensor.ty()).value().get<1>(),
       dtype = tensor.ty()->DefaultIndexType()](const ffi::Array<PrimExpr>& indices) -> PrimExpr {
    PrimExpr flatten_index = IntImm(dtype, 0);
    for (int i = 0; i < ndim; ++i) {
      flatten_index = flatten_index + strides[i] * indices[i];
    }
    return flatten_index;
  };
  // Step 3. Detect the IterSplitExpr of the indexing pattern
  std::vector<SplitExprCollector::SplitExpr> split_exprs = SplitExprCollector::Collect(
      /*index=*/f_flatten_index(indices), input_iters, predicate,
      /*check_level=*/sym::IterMapLevel::Surjective, analyzer);
  if (split_exprs.empty()) {
    return std::nullopt;
  }
  // Step 4. Sort the order of the split expressions
  std::vector<int> order(split_exprs.size(), 0);
  std::generate(order.begin(), order.end(), [n = 0]() mutable { return n++; });
  std::sort(order.begin(), order.end(), [&split_exprs, &var2id](int _a, int _b) -> bool {
    const SplitExprCollector::SplitExpr& a = split_exprs[_a];
    const SplitExprCollector::SplitExpr& b = split_exprs[_b];
    int a_var_id = var2id.at(a.source.get());
    int b_var_id = var2id.at(b.source.get());
    if (a_var_id != b_var_id) {
      return a_var_id < b_var_id;
    }
    return a.lower_factor > b.lower_factor;
  });
  // Compute the inverse permutation by argsort
  std::vector<int> inverse_order = order;
  std::sort(inverse_order.begin(), inverse_order.end(),
            [&order](int _a, int _b) -> bool { return order[_a] < order[_b]; });
  // Step 5. Create the indexing mapping
  auto f_alter_layout = [f_flatten_index = std::move(f_flatten_index),  //
                         &split_exprs,                                  //
                         &order,                                        //
                             & shape = tensor.shape(),                  //
                         analyzer                                       //
  ](ffi::Array<Var> indices) -> ffi::Array<PrimExpr> {
    TVM_FFI_ICHECK_EQ(indices.size(), shape.size());
    for (int i = 0, n = indices.size(); i < n; ++i) {
      {
        auto context = analyzer->GetConstraintContext();
        context->Append(sym::RangeConstraint(indices[i], Range::FromMinExtent(0, shape[i])));
        analyzer->SyncConstraintContext(context);
      }
    }
    // Step 5.1: Fuse all indices into a flattened one
    PrimExpr index =
        f_flatten_index(indices.Map([](const Var& var) { return var.as_or_throw<PrimExpr>(); }));
    int ndim = split_exprs.size();
    // Step 5.2. Split the flattened index according to `split_exprs`
    std::vector<PrimExpr> split;
    split.reserve(ndim);
    for (int i = ndim - 1; i >= 0; --i) {
      index = analyzer->Simplify(index);
      int64_t extent = split_exprs[i].extent;
      split.push_back(analyzer->Simplify(floormod(index, extent)));
      index = floordiv(index, extent);
    }
    std::reverse(split.begin(), split.end());
    // Step 5.3. Reorder the indexing pattern according to `order`
    ffi::Array<PrimExpr> results;
    results.reserve(ndim);
    for (int i = 0; i < ndim; ++i) {
      results.push_back(split[order[i]]);
    }
    return results;
  };
  // Step 6: Create the inverse index mapping.
  auto f_inverse = [&inverse_order, &split_exprs, &shape = tensor.shape(),
                    analyzer](ffi::Array<Var> indices) -> ffi::Array<PrimExpr> {
    TVM_FFI_ICHECK_EQ(indices.size(), split_exprs.size());
    // Step 6.1: Reorder the indices according to `inverse_order`. This is the inverse of Step 5.3.
    // After the inverse permutation, indices[i] corresponds to split_exprs[i]
    ffi::Array<Var> inv_permuted_indices;
    inv_permuted_indices.reserve(indices.size());
    for (int i = 0, n = indices.size(); i < n; ++i) {
      const Var& index = indices[inverse_order[i]];
      inv_permuted_indices.push_back(index);
      {
        auto context = analyzer->GetConstraintContext();
        context->Append(sym::RangeConstraint(
            index, Range::FromMinExtent(0, IntImm::Int32(split_exprs[i].extent))));
        analyzer->SyncConstraintContext(context);
      }
    }

    // Step 6.2: Fuse all the indices. This is the inverse of Step 5.2.
    PrimExpr flattened_index = IntImm(indices[0]->ty.as_or_throw<PrimType>(), 0);
    int64_t stride = 1;
    for (int i = static_cast<int>(split_exprs.size()) - 1; i >= 0; --i) {
      flattened_index =
          inv_permuted_indices[i].as_or_throw<PrimExpr>() * IntImm::Int32(stride) + flattened_index;
      stride *= split_exprs[i].extent;
    }
    // Step 6.3: Split the flattened index into multiple indices. This is the inverse of Step 5.1.
    ffi::Array<PrimExpr> result;
    result.reserve(shape.size());
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
      PrimExpr index = analyzer->Simplify(floormod(flattened_index, shape[i]));
      flattened_index = floordiv(flattened_index, shape[i]);
      result.push_back(index);
    }
    return ffi::Array<PrimExpr>(result.rbegin(), result.rend());
  };
  IndexMap inverse_index_map = IndexMap::FromFunc(split_exprs.size(), f_inverse);
  return IndexMap::FromFunc(ndim, f_alter_layout, inverse_index_map);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.schedule.SuggestIndexMap",
                        [](TensorVar tensor, ffi::Array<PrimExpr> indices, ffi::Array<For> loops,
                           PrimExpr predicate) {
                          sym::Analyzer analyzer;
                          return SuggestIndexMap(tensor, indices, loops, predicate, analyzer.get());
                        });
}

}  // namespace s_tir
}  // namespace tvm
