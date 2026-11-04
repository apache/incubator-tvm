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
 * \file tirx/analysis/identify_memcpy.cc
 * \brief Check if a loop nest is equivalent to memcpy
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/stmt_functor_with_constraint_context.h>
#include <tvm/sym/int_set.h>
#include <tvm/sym/iter_affine_map.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>

#include <optional>
#include <sstream>
#include <string>
#include <variant>

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

std::variant<MemCpyDetails, std::string> IdentifyMemCpyImpl(const For& loop,
                                                            sym::AnalyzerObj* analyzer) {
  ffi::Map<Var, sym::IntSet> loop_intervals;
  ffi::Map<PrimVar, Range> loop_ranges;
  PrimExpr total_loop_iterations = 1;

  // Walk through the loop nest, stopping at the first loop whose body
  // is not a loop.
  Stmt stmt = loop;
  while (auto* for_node = stmt.as<ForObj>()) {
    loop_ranges.Set(for_node->var, Range(for_node->begin, for_node->end));
    loop_intervals.Set(for_node->var, sym::IntSet::FromMinExtent(
                                          for_node->begin, Range::GetExtentFromBeginEnd(
                                                               for_node->begin, for_node->end)));
    total_loop_iterations =
        total_loop_iterations * Range::GetExtentFromBeginEnd(for_node->begin, for_node->end);

    stmt = for_node->body->size() == 1 ? for_node->body->seq[0] : for_node->body;
  }

  TensorStore store{ffi::UnsafeInit{}};
  if (auto opt = stmt.as<TensorStore>()) {
    store = opt.value();
  } else {
    return static_cast<const std::stringstream&>(
               std::stringstream()
               << "Expected innermost loop to have TensorStore body, but instead found " << stmt)
        .str();
  }

  TensorLoad load{ffi::UnsafeInit{}};
  if (auto opt = store->value.as<TensorLoad>()) {
    load = opt.value();
  } else {
    return static_cast<const std::stringstream&>(
               std::stringstream()
               << "Expected TensorStore's value to be TensorLoad, but instead found "
               << store->value)
        .str();
  }

  // Now, we have a TensorStore whose value is a TensorLoad.  Because
  // non-flat physical indices are target-dependent, only handle cases
  // where the tensor will be flattened to a 1-d physical tensor.
  ffi::Array<PrimExpr> flattened_dst =
      store->dest.as_or_throw<TensorVar>().ty()->ElemOffset(store->indices);
  ffi::Array<PrimExpr> flattened_src =
      load->source.as_or_throw<tvm::tirx::TensorVar>().ty()->ElemOffset(load->indices);

  if (flattened_dst.size() != 1 || flattened_src.size() != 1) {
    return static_cast<const std::stringstream&>(
               std::stringstream()
               << "Expected flattened dimension of src/dest to be 1, but found"
               << flattened_src.size() << "-d src and " << flattened_dst.size() << "-d dst")
        .str();
  }
  PrimExpr src_index = flattened_src[0];
  PrimExpr dst_index = flattened_dst[0];

  // First check, do the input/output form affine subsets of their
  // respective tensors?
  //
  // For example, should exclude the following, indices are not affine
  //
  // for i in T.serial(16):
  //     B[i] = A[T.abs(i-8)]

  sym::Analyzer analyzer_ref = ffi::GetRef<sym::Analyzer>(analyzer);
  auto src_iter_map = sym::DetectIterMap({src_index}, loop_ranges, IntImm::Bool(true),
                                         sym::IterMapLevel::Bijective, analyzer_ref);
  if (src_iter_map->errors.size()) {
    return static_cast<const std::stringstream&>(std::stringstream()
                                                 << "sym::DetectIterMap(src) returned "
                                                 << src_iter_map->errors.size() << " errors: ["
                                                 << src_iter_map->errors << "]"
                                                 << " for src_index = " << src_index)
        .str();
  }
  auto dst_iter_map = sym::DetectIterMap({dst_index}, loop_ranges, IntImm::Bool(true),
                                         sym::IterMapLevel::Bijective, analyzer_ref);
  if (dst_iter_map->errors.size()) {
    return static_cast<const std::stringstream&>(std::stringstream()
                                                 << "sym::DetectIterMap(dst) returned "
                                                 << dst_iter_map->errors.size() << " errors: ["
                                                 << dst_iter_map->errors << "]"
                                                 << " for dst_index = " << dst_index)
        .str();
  }

  // Second check, are those affine subsets contiguous?  If so, then
  // the index expressions will visit every location between the min
  // and the max.  This checks surjectivity over a linear region,
  // which may not be the same as DetectIterMap's check of
  // surjectivity over the affine subset.
  //
  // For example, should exclude the following, doesn't touch all
  // output locations within the output region touched.
  //
  // for i in T.serial(16):
  //     B[2*i] = A[i]
  //
  // Similarly, should exclude the following, doesn't touch all
  // input locations within the input region touched.
  //
  // for i in T.serial(16):
  //     B[i] = A[2*i]
  total_loop_iterations = analyzer->Simplify(total_loop_iterations);
  auto src_interval = analyzer->int_set(src_index, loop_intervals);
  auto dst_interval = analyzer->int_set(dst_index, loop_intervals);

  if (!src_interval.HasLowerBound() || !src_interval.HasUpperBound()) {
    return static_cast<const std::stringstream&>(std::stringstream()
                                                 << "Expected known bounds for src, but found "
                                                 << src_interval << " for expression " << src_index)
        .str();
  }
  if (!dst_interval.HasLowerBound() || !dst_interval.HasUpperBound()) {
    return static_cast<const std::stringstream&>(std::stringstream()
                                                 << "Expected known bounds for dst, but found "
                                                 << dst_interval << " for expression " << dst_index)
        .str();
  }

  {
    PrimExpr must_prove = total_loop_iterations == src_interval.max() - src_interval.min() + 1;
    PrimExpr simplified = analyzer->Simplify(must_prove);
    if (!analyzer->CanProve(simplified)) {
      return static_cast<const std::stringstream&>(
                 std::stringstream()
                 << "Mismatch between loop iterations (" << total_loop_iterations
                 << ") and number of src indices touched (" << src_interval
                 << ".  Equality to prove simplified to " << simplified)
          .str();
    }
  }
  {
    PrimExpr must_prove = total_loop_iterations == dst_interval.max() - dst_interval.min() + 1;
    PrimExpr simplified = analyzer->Simplify(must_prove);
    if (!analyzer->CanProve(simplified)) {
      return static_cast<const std::stringstream&>(
                 std::stringstream()
                 << "Mismatch between loop iterations (" << total_loop_iterations
                 << ") and number of dst indices touched (" << dst_interval
                 << ".  Equality to prove simplified to " << simplified)
          .str();
    }
  }

  // Third check, is there a transformation applied between the input
  // and output iterators?
  //
  // For example, the following would pass all checks so far, but
  // converts between row-major and column-major layouts, and could
  // not be specified as a memcpy.
  //
  // for i,j in T.grid(4,4):
  //     B[i,j] = A[j,i]

  auto src_iter_sum = src_iter_map->indices[0];
  auto dst_iter_sum = dst_iter_map->indices[0];

  if (src_iter_sum->args.size() != dst_iter_sum->args.size()) {
    return static_cast<const std::stringstream&>(
               std::stringstream()
               << "IterMap for src/dst unpacked to different number of IterSplitExpr: "
               << src_iter_sum->args.size() << " for src, " << dst_iter_sum->args.size()
               << " for dst.  "
               << "IterMaps were detected as src = " << src_iter_sum << ", dst = " << dst_iter_sum)
        .str();
  }
  std::vector<sym::IterSplitExpr> src_iter_terms(src_iter_sum->args.begin(),
                                                 src_iter_sum->args.end());
  std::vector<sym::IterSplitExpr> dst_iter_terms(dst_iter_sum->args.begin(),
                                                 dst_iter_sum->args.end());

  auto make_comparison_tuple = [](const sym::IterSplitExpr& expr) {
    auto as_int_or_zero = [](auto& val) -> ffi::BigInt {
      if (auto* as_int = val.template as<IntImmObj>()) {
        return as_int->value;
      } else {
        return 0;
      }
    };
    return std::tuple{
        static_cast<bool>(expr->scale.as<IntImmObj>()),        as_int_or_zero(expr->scale),
        static_cast<bool>(expr->extent.as<IntImmObj>()),       as_int_or_zero(expr->lower_factor),
        static_cast<bool>(expr->lower_factor.as<IntImmObj>()), as_int_or_zero(expr->lower_factor),
    };
  };
  auto sorting_function = [&make_comparison_tuple](const sym::IterSplitExpr& lhs,
                                                   const sym::IterSplitExpr& rhs) -> bool {
    return make_comparison_tuple(lhs) < make_comparison_tuple(rhs);
  };
  std::sort(src_iter_terms.begin(), src_iter_terms.end(), sorting_function);
  std::sort(dst_iter_terms.begin(), dst_iter_terms.end(), sorting_function);

  for (size_t i = 0; i < src_iter_terms.size(); i++) {
    const sym::IterSplitExpr& src_term = src_iter_terms[i];
    const sym::IterSplitExpr& dst_term = dst_iter_terms[i];

    if (!analyzer->CanProve(
            sym::NormalizeIterMapToExpr(src_term->source->source == dst_term->source->source))) {
      return static_cast<const std::stringstream&>(
                 std::stringstream()
                 << "Term " << i << " had different source, src_term->source = " << src_term->source
                 << ", dst_term->source = " << dst_term->source)
          .str();
    }
    if (!analyzer->CanProve(src_term->lower_factor == dst_term->lower_factor)) {
      return static_cast<const std::stringstream&>(
                 std::stringstream()
                 << "Term " << i << " had different lower_factor, src_term->lower_factor = "
                 << src_term->lower_factor
                 << ", dst_term->lower_factor = " << dst_term->lower_factor)
          .str();
    }
    if (!analyzer->CanProve(src_term->extent == dst_term->extent)) {
      return static_cast<const std::stringstream&>(
                 std::stringstream()
                 << "Term " << i << " had different extent, src_term->extent = " << src_term->extent
                 << ", dst_term->extent = " << dst_term->extent)
          .str();
    }
    if (!analyzer->CanProve(src_term->scale == dst_term->scale)) {
      return static_cast<const std::stringstream&>(
                 std::stringstream()
                 << "Term " << i << " had different scale, src_term->scale = " << src_term->scale
                 << ", dst_term->scale = " << dst_term->scale)
          .str();
    }
  }

  TensorRegion src_region =
      TensorRegion(load->source.as_or_throw<tvm::tirx::TensorVar>(),
                   DomainTouched(loop, load->source.as_or_throw<tvm::tirx::TensorVar>(), true, true)
                       .Map([](const ffi::Optional<Range>& range) { return range.value(); }),
                   TensorRegionType());
  TensorRegion dst_region =
      TensorRegion(store->dest.as_or_throw<TensorVar>(),
                   DomainTouched(loop, store->dest.as_or_throw<TensorVar>(), true, true)
                       .Map([](const ffi::Optional<Range>& range) { return range.value(); }),
                   TensorRegionType());

  return MemCpyDetails{src_region, dst_region};
}

std::optional<MemCpyDetails> IdentifyMemCpy(const For& loop, const sym::Analyzer& analyzer) {
  auto result = IdentifyMemCpyImpl(loop, analyzer.get());
  if (auto* ptr = std::get_if<MemCpyDetails>(&result)) {
    return *ptr;
  } else {
    return std::nullopt;
  }
}

// Expose the IdentifyMemCpy functionality to Python API for purpose of unit testing.
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.analysis._identify_memcpy", [](const Stmt& stmt) {
    ffi::Array<ffi::ObjectRef> output;

    struct Visitor : s_tir::StmtExprVisitorWithConstraintContext {
     public:
      using s_tir::StmtExprVisitorWithConstraintContext::Visit_;

      explicit Visitor(ffi::Array<ffi::ObjectRef>* output) : output(output) {}
      ffi::Array<ffi::ObjectRef>* output;

     private:
      ffi::Optional<VisitInterrupt> Visit_(const ForObj* op) override {
        For loop = ffi::GetRef<For>(op);
        auto result = IdentifyMemCpyImpl(loop, GetSyncedAnalyzer());
        if (auto* ptr = std::get_if<MemCpyDetails>(&result)) {
          output->push_back(ffi::Array{ptr->source, ptr->dest});
        } else if (auto* ptr = std::get_if<std::string>(&result)) {
          output->push_back(StringImm(*ptr));
        } else {
          TVM_FFI_THROW(InternalError) << "Internal error, unhandled std::variant type";
        }

        return StmtExprVisitorWithConstraintContext::Visit_(op);
      }
    };

    auto visitor = ffi::make_object<Visitor>(&output);
    visitor->Visit(stmt);

    return output;
  });
}

}  // namespace s_tir
}  // namespace tvm
