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
 *  Out of bounds array access static analyzer.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/stmt_functor_with_constraint_context.h>
#include <tvm/tirx/transform.h>

#include "../schedule/error.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;
namespace transform {
struct OOBLocation {
  TensorVar tensor;
  size_t dimension;
  ffi::ObjectRef index;
  sym::IntSet index_bounds;
  sym::IntSet shape_bounds;
};

class OOBError : public s_tir::ScheduleErrorContextObj {
 public:
  OOBError(IRModule mod, std::vector<OOBLocation> locations) : mod_(mod), locations_(locations) {}
  ffi::String FastErrorString() const final { return "Out of bound memory access"; }

  ffi::String DetailRenderTemplate() const final {
    std::stringstream s;
    for (const auto& oob : locations_) {
      s << "Out of bounds memory access on tensor " << oob.tensor->name << " dimension "
        << oob.dimension << ".";
      s << " index " << oob.index << " with bounds [" << oob.index_bounds.min() << ", "
        << oob.index_bounds.max() << "] is outside the range [0, " << oob.shape_bounds.min()
        << "].";
      s << "\n";
    }
    return s.str();
  }
  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final {
    std::vector<ffi::ObjectRef> locs;
    for (auto loc : locations_) {
      locs.push_back(loc.index);
    }
    return locs;
  }

 private:
  IRModule mod_;
  std::vector<OOBLocation> locations_;
};
class OOBCheckerVisitor final : public s_tir::StmtExprVisitorWithConstraintContext {
 public:
  using s_tir::StmtExprVisitorWithConstraintContext::Visit_;

  ffi::Optional<VisitInterrupt> Visit_(const TensorStoreObj* node) final {
    for (size_t i = 0; i < node->dest.as_or_throw<TensorVar>().shape().size(); i++) {
      CheckBounds(node, node->dest.as_or_throw<TensorVar>(), i);
    }
    return StmtExprVisitorWithConstraintContext::Visit_(node);
  }
  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadObj* node) final {
    TensorVar tensor = node->source.as_or_throw<tvm::tirx::TensorVar>();
    for (size_t i = 0; i < tensor.shape().size(); i++) {
      CheckBounds(node, tensor, i);
    }
    return StmtExprVisitorWithConstraintContext::Visit_(node);
  }

  template <class T>
  void CheckBounds(const T* node, const TensorVar& tensor, size_t i) {
    auto ind_bounds = GetSyncedAnalyzer()->int_set(node->indices[i]);
    auto shape_bounds = GetSyncedAnalyzer()->int_set(tensor.shape()[i]);
    // We would expect that
    // `analyzer_.CanProve(node->indices[i] < 0 || node->indices[i] >= tensor->shape[i])`
    // would be the way to check if any out of bounds access occurs here, but `CanProve` checks if
    // the statement is true for all possible values (universal quantification). For a mix of in
    // bounds and out of bounds access, no out of bounds access would be reported. We instead want
    // to check if there is any value for which the access is out of bounds (existential
    // quantification).
    // An solution would be to check that the index is in bounds for every possible value. This
    // has the problem that some valid access patterns maybe be valid but not provably valid. We
    // prefer that this analysis is conservative and only shows errors that are provable. This leads
    // us to the following check: are the bounds of the index outside the bounds of the shape.
    if (GetSyncedAnalyzer()->CanProve(ind_bounds.max() >= shape_bounds.min()) ||
        GetSyncedAnalyzer()->CanProve(ind_bounds.min() < 0)) {
      errors.push_back({tensor, i, node->indices[i], ind_bounds, shape_bounds});
    }
  }

  std::vector<OOBLocation> errors;
};

tvm::transform::Pass OOBChecker() {
  auto pass_func = [=](tirx::Function func, IRModule mod, tvm::transform::PassContext ctx) {
    auto checker = ffi::make_object<OOBCheckerVisitor>();
    checker->Visit(func->body);
    if (checker->errors.size() > 0) {
      // mod doesn't contain our function, so we construct a new mod with out function
      IRModule func_mod({{GlobalVar("main"), func}});
      TVM_FFI_THROW(ScheduleError)
          << OOBError(func_mod, checker->errors).RenderReport("Out of bounds checker");
    }
    return func;
  };
  return tirx::transform::CreateFunctionPass(pass_func, 0, "s_tir.analysis.OOBChecker");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.analysis.OOBChecker", OOBChecker);
}
}  // namespace transform
}  // namespace s_tir
}  // namespace tvm
