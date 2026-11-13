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
 * \file relax/analysis/layout_transformation.cc
 * \brief Analyze the tirx::Function and suggest layout transformation on it's blocks and tensors
 * based on the user provided layout transformations on it's outputs.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/relax/analysis.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/sym/analyzer.h>
#include <tvm/sym/iter_affine_map.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/index_map.h>

namespace tvm {
namespace relax {

using namespace tirx;

/********** Helper Functions **********/

/*! \brief Checks if a transformation is bijective affine over the given ranges */
static bool IsBijectiveAffine(const IndexMap& m, const ffi::Array<Range>& ranges) {
  ffi::Map<PrimVar, Range> input_iters;
  TVM_FFI_ICHECK_EQ(m->initial_indices.size(), ranges.size());
  for (size_t i = 0; i < ranges.size(); i++) {
    input_iters.Set(m->initial_indices[i], ranges[i]);
  }
  sym::Analyzer analyzer;
  auto iter_map_result = DetectIterMap(m->final_indices, input_iters, /* predicate = */ 1,
                                       /*check_level=*/sym::IterMapLevel::Bijective, analyzer,
                                       /*simplify_trivial_iterators=*/true);
  return !iter_map_result->indices.empty();
}

/*!
 * \brief Analyzer to collect iterators from IterSumExpr.
 * \details Analyzes the indices from DetectIterMap analysis to collect the spatial iterators that
 * are used in it. This is important to get which spatial iterators are accessed in each index
 * of tensor access.
 */
class IndexAnalyzer : public s_tir::StmtExprVisitor {
 public:
  ffi::Array<tvm::Var> Analyze(const sym::IterSumExpr& expr) {
    Visit(expr);
    return iterators_;
  }

 private:
  /*! \brief Override Visit for iter expr type processing */
  ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
    if (const auto* op = value.as<sym::IterSumExprObj>()) {
      for (const auto& arg : op->args) TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(arg));
      return Visit(op->base);
    }
    if (const auto* op = value.as<sym::IterSplitExprObj>()) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(VisitIterMark(op->source));
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->lower_factor));
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->extent));
      return Visit(op->scale);
    }
    return s_tir::StmtExprVisitor::Visit(value);
  }

  ffi::Optional<VisitInterrupt> VisitIterMark(const sym::IterMark& op) {
    if (auto var = op->source.as<PrimVar>())
      iterators_.push_back(var.value());
    else
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->source));
    return Visit(op->extent);
  }

 private:
  ffi::Array<tvm::Var> iterators_;
};

/*!
 * \brief Analyzes IterMapResult to get the Spatial Layout of tensor access.
 * \details We define Spatial Layout of a tensor access as an array of length equal to the
 * dimensions of the tensor. i-th element of Spatial Layout contains spatial iter var used from the
 * block iteration domain. For indices, where no spatial iter vars are used, the spatial layout
 * element is empty. If any of the tensor access indices use multiple spatial iter vars, the spatial
 * layout is undefined.
 *
 * Here are a few examples of inferred spatial layout from tensor access. si denotes i-th spatial
 * iter var, and ri denotes i-th reduction iter var.
 *
 * SpatialLayout(A[s0*constant, s1]) = {s0, s1}
 * SpatialLayout(A[s0, constant, r0, s1]) = {s0, null, null, s1}
 * SpatialLayout(A[s0 * c + s1]) = undefined
 */
using SpatialLayout = ffi::Array<ffi::Optional<tvm::Var>>;
static SpatialLayout GetSpatialLayout(const sym::IterMapResult& iter_map_result) {
  TVM_FFI_ICHECK(!iter_map_result->indices.empty());
  SpatialLayout result;
  for (const sym::IterSumExpr& index : iter_map_result->indices) {
    auto index_analyzer = ffi::make_object<IndexAnalyzer>();
    ffi::Array<tvm::Var> iter_vars = index_analyzer->Analyze(index);
    if (iter_vars.size() >= 2) {
      LOG(WARNING) << "[LayoutInference] Unable to get spatial layout of access: "
                   << sym::NormalizeIterMapToExpr(index);
      return {};
    }
    if (iter_vars.empty()) {
      result.push_back({});
      continue;
    }
    result.push_back(iter_vars[0]);
  }
  return result;
}

/*!
 * \brief Checks if the two spatial layouts are identical. Two empty spatial layouts are treated as
 * unequal.
 */
static bool AreIdenticalSpatialAccess(const SpatialLayout& s0, const SpatialLayout& s1) {
  if (s0.empty() || s1.empty()) return false;
  if (s0.size() != s1.size()) return false;
  for (size_t i = 0; i < s0.size(); ++i) {
    if ((!s0[i].has_value() && s1[i].has_value()) || (s0[i].has_value() && !s1[i].has_value()))
      return false;
    if (!s0[i].same_as(s1[i])) return false;
  }
  return true;
}

/*!
 * \brief Checks if the block accesses a tensor sequentially in terms of spatial dimensions
 * (ignoring reduction dimensions). It checks that the order of spatial iter vars in spatial layout
 * of a tensor access is same as the order of spatial iter vars in block domain.
 */
using VarToBlockIndexMap = std::unordered_map<tvm::Var, int>;
static bool IsSequentialAccess(const SpatialLayout& iterators,
                               const VarToBlockIndexMap& iter_to_block_index) {
  int last_value = -1;
  for (const auto& i : iterators) {
    if (!i.has_value()) continue;
    auto it = iter_to_block_index.find(i.value());
    TVM_FFI_ICHECK(it != iter_to_block_index.end());
    int blk_index = it->second;
    if (blk_index <= last_value) return false;
    last_value = blk_index;
  }
  return true;
}

/*! \brief Checks if two IndexMaps represent identical transforms */
static bool AreIdenticalTransforms(const IndexMap& t0, const IndexMap& t1) {
  if (t0->initial_indices.size() != t1->initial_indices.size()) return false;
  if (t0->final_indices.size() != t1->final_indices.size()) return false;

  // Create a new shape expression.
  ffi::Array<PrimExpr> t1_initial_indices =
      t1->initial_indices.Map([](tvm::Var i) { return i.as_or_throw<PrimExpr>(); });
  sym::Analyzer analyzer;
  auto t0_output = t0->MapIndices(t1_initial_indices, analyzer);
  for (size_t i = 0; i < t0_output.size(); ++i) {
    if (!analyzer->CanProveEqual(t0_output[i], t1->final_indices[i])) return false;
  }
  return true;
}

/*!
 * \brief Returns the layout transformation for a target spatial layout from the source spatial
 * layout and transformation.
 * \details Given the source tensor spatial layout \p src_spatial_layout and its transformation \p
 * src_transformation, this function constructs the transformation for the target tensor whose
 * spatial layout is given as \p tgt_spatial_layout.
 *
 * The algorithm is explained below using an example:
 *
 * Let's say the source transformation is lambda N, C, H, W -> (N, H, W, C // 4, C %
 * 4), source spatial layout is 'NCHW' and target spatial layout is 'KCHW'.
 *
 * Step 1: Copy over the source transformation initial & final indices to target transformation
 * initial and final indices.
 * target transformation = lambda N, C, H, W -> (N, H, W, C // 4, C %4)
 *
 * Step 2: Drop any vars from initial indices which do not occur in target tensor using source and
 * target spatial layouts.
 * target transformation = lambda C, H, W -> (N, H, W, C // 4, C %4)
 *
 * Step 3: Erase any expression from final indices which is dependent on a var not present in
 * initial indices.
 * target transformation = lambda C, H, W -> (H, W, C // 4, C %4)
 *
 * Step 4: Go over the target spatial layout and add any missing dims to both initial and final
 * indices. This is done by checking if any iterator in target spatial layout is not present in
 * source spatial layout.
 * target transformation = lambda dim, C, H, W -> (dim, H, W, C // 4, C %4)
 */
using VarSet = std::unordered_set<tvm::Var>;
static ffi::Optional<IndexMap> InferLayoutTransformation(const SpatialLayout& src_spatial_layout,
                                                         const IndexMap& src_transformation,
                                                         const SpatialLayout& tgt_spatial_layout) {
  // Copy over the src transformation intial and final indices
  std::vector<tvm::Var> initial_indices(src_transformation->initial_indices.begin(),
                                        src_transformation->initial_indices.end());
  std::vector<PrimExpr> final_indices(src_transformation->final_indices.begin(),
                                      src_transformation->final_indices.end());

  // Get the iterator var set used in target spatial layout.
  VarSet tgt_var_set;
  for (const auto& i : tgt_spatial_layout) {
    if (i.has_value()) tgt_var_set.insert(i.value());
  }

  // Erase initial indices corresponding to iter vars that do not occur in target spatial layout.
  // Also compute the var set of initial indices.
  auto initial_indices_it = initial_indices.begin();
  VarSet initial_indices_var_set;
  for (const auto& i : src_spatial_layout) {
    TVM_FFI_ICHECK(i.has_value());
    if (tgt_var_set.count(i.value())) {
      initial_indices_var_set.insert(*initial_indices_it);
      initial_indices_it++;
      continue;
    }
    initial_indices_it = initial_indices.erase(initial_indices_it);
  }

  // Erase any expressions in final indices that have undefined vars
  auto final_indices_it = final_indices.begin();
  while (final_indices_it != final_indices.end()) {
    // Collect all the vars used in this final index.
    ffi::Array<tvm::Var> used_vars = tvm::UndefinedVars(*final_indices_it);
    TVM_FFI_ICHECK(!used_vars.empty())
        << "IndexMap expression must always contain tvm::Var nodes but found none in: "
        << *final_indices_it;

    bool has_undefined_vars = std::any_of(used_vars.begin(), used_vars.end(),
                                          [&initial_indices_var_set](const tvm::Var& v) {
                                            return initial_indices_var_set.count(v) == 0;
                                          });

    // If all vars are from initial indices, nothing to do for this final index.
    if (!has_undefined_vars) {
      final_indices_it++;
      continue;
    }
    // We are about to drop this expr from final indices since it has undefined vars. Check if it is
    // dependent on any of the initial indices. If it is dependent, this cannot be dropped and we
    // bail by returning null.
    // This captures the scenario where the source transformation is unpacking a dimension (e.g,
    // "H4h" -> "H*4+h" ) and the tensor we are trying to infer the transformation of has 'h'
    // dimension, but not 'H'. So, it is dependent on undefined var 'H' and defined var 'h'.
    bool depends_on_initial_indices = std::any_of(used_vars.begin(), used_vars.end(),
                                                  [&initial_indices_var_set](const tvm::Var& v) {
                                                    return initial_indices_var_set.count(v) != 0;
                                                  });
    if (depends_on_initial_indices) {
      LOG(WARNING)
          << "[LayoutInference] Tensor access is dependent on both defined and undefined vars";
      return {};
    }
    // It is ok to erase this final index expression as it only depends on undefined vars.
    final_indices_it = final_indices.erase(final_indices_it);
  }

  // Go over the target spatial layout and add any missing dims to both initial and final indices.
  // This is done by checking if any iterator in target spatial layout is not present in source
  // spatial layout.
  VarSet src_var_set;
  for (const auto& i : src_spatial_layout) {
    TVM_FFI_ICHECK(i.has_value());
    src_var_set.insert(i.value());
  }

  initial_indices_it = initial_indices.begin();
  final_indices_it = final_indices.begin();
  for (const auto& i : tgt_spatial_layout) {
    if (i.has_value() && src_var_set.count(i.value())) {
      initial_indices_it++;
      if (final_indices_it != final_indices.end()) final_indices_it++;
      continue;
    }

    PrimVar new_dim("d");
    PrimExpr new_dim_expr = new_dim;
    initial_indices_it = initial_indices.insert(initial_indices_it, new_dim);
    final_indices_it = final_indices.insert(final_indices_it, new_dim_expr);
    // Advance past the newly inserted element so subsequent iterations start correctly.
    initial_indices_it++;
    final_indices_it++;
  }

  ffi::Array<tvm::Var> initial_array(initial_indices.begin(), initial_indices.end());
  ffi::Array<PrimExpr> final_array(final_indices.begin(), final_indices.end());
  return IndexMap(initial_array.Map([](tvm::Var var) { return var.as_or_throw<PrimVar>(); }),
                  final_array);
}

/*!
 * \brief Analyzes the Block and given output tensor transformations to propose
 * transformations of block and read tensors.
 * \details It does a best effort analysis to propose transformations which would preserve
 * sequential access to tensors (especially output tensors). Since this is best effort, it is
 * possible that the Block is too complex for analysis. In such a case, no transformations are
 * proposed. Limitations:
 * 1. Expects exactly one write tensor in the block whose transformation is given by
 * `write_transformation`.
 * 2. Expects write tensor access to be affine and only use spatial iterators of the block.
 * 3. Proposes transformations to a read tensor if all access to it are affine.
 */
class BlockAnalyzer : public s_tir::StmtExprVisitor {
 public:
  explicit BlockAnalyzer(const s_tir::SBlock& block,
                         const ffi::Map<TensorVar, IndexMap>& transformation_cache,
                         IndexMap write_transformation)
      : can_transform_block_(true),
        write_transformation_(write_transformation),
        block_(block),
        tensor_transformation_cache_(transformation_cache) {}

  void Analyze() {
    const auto& block = block_;
    TVM_FFI_ICHECK(block_->writes.size() == 1);
    auto write_tensor = block_->writes[0]->source.as_or_throw<tvm::tirx::TensorVar>();

    ComputeBlockSpatialDomain();

    // Visit the block body to collect load/store access patterns of different tensors.
    Visit(block_->body);

    // While visiting the load/store accesses it is possible we see an unexpected pattern, such as
    // nested block or write access to multiple tensors. In such a case, we can return early as we
    // would not be making any layout suggesstions.
    if (!can_transform_block_) {
      LOG(WARNING) << "[LayoutInference] Unable to transform block " << block->name_hint;
      return;
    }

    // Get iterator ordering and it's spatial layout.
    VarToBlockIndexMap iter_var_to_block_index;
    SpatialLayout block_spatial_layout;
    int index = 0;
    for (const auto& iter_var : block->iter_vars) {
      auto var = iter_var->var;
      iter_var_to_block_index[var] = index++;
      block_spatial_layout.push_back(static_cast<tvm::Var>(var));
    }

    // Helper to get the spatial layout of tensor from tensor access map.
    auto get_spatial_layout = [&](TensorVar b) -> SpatialLayout {
      auto it = tensor_access_info_.find(b);
      if (it == tensor_access_info_.end()) {
        return {};
      }
      auto access_info = it->second;
      return access_info.GetValidSpatialLayout();
    };

    // Check that write has sequential access within the block.
    SpatialLayout write_spatial_layout = get_spatial_layout(write_tensor);
    if (write_spatial_layout.empty()) {
      can_transform_block_ = false;
      return;
    }
    if (!IsSequentialAccess(write_spatial_layout, iter_var_to_block_index)) {
      can_transform_block_ = false;
      return;
    }

    // Infer Block transformation from write tensor transformation.
    auto maybe_block_transformation = InferLayoutTransformation(
        write_spatial_layout, write_transformation_, block_spatial_layout);
    if (!maybe_block_transformation.has_value()) {
      can_transform_block_ = false;
      return;
    }
    block_transformation_ = maybe_block_transformation.value();

    ffi::Array<Range> block_ranges =
        block_->iter_vars.Map([](const s_tir::IterVar& i) { return i->dom.value(); });
    if (!IsBijectiveAffine(block_transformation_.value(), block_ranges)) {
      can_transform_block_ = false;
      LOG(WARNING) << "[LayoutInference] Inferred block transformation is not bijective affine, "
                      "transformation: ("
                   << block_transformation_ << ") over range (" << block_ranges << ")";
      return;
    }

    // Infer read tensor transformations from write tensor transformation.
    for (const auto& r : block->reads) {
      SpatialLayout read_spatial_layout =
          get_spatial_layout(r->source.as_or_throw<tvm::tirx::TensorVar>());
      if (read_spatial_layout.empty()) continue;
      if (!IsSequentialAccess(read_spatial_layout, iter_var_to_block_index)) continue;

      auto maybe_read_transformation = InferLayoutTransformation(
          write_spatial_layout, write_transformation_, read_spatial_layout);
      if (!maybe_read_transformation.has_value()) continue;
      IndexMap read_transformation = maybe_read_transformation.value();
      if (tensor_transformation_cache_.count(r->source.as_or_throw<tvm::tirx::TensorVar>()) != 0) {
        if (!AreIdenticalTransforms(
                read_transformation,
                tensor_transformation_cache_[r->source.as_or_throw<tvm::tirx::TensorVar>()]))
          LOG(WARNING)
              << "[LayoutInference] Tensor: " << r->source.as_or_throw<tvm::tirx::TensorVar>()
              << " has conflicting transform proposals -- (preferred) "
              << tensor_transformation_cache_[r->source.as_or_throw<tvm::tirx::TensorVar>()]
              << " vs. " << read_transformation;
        continue;
      }
      read_tensor_transformations_.Set(r->source.as_or_throw<tvm::tirx::TensorVar>(),
                                       read_transformation);
    }
  }

 private:
  // Helper class to keep track of spatial layout of tensor as we visit multiple accesses to this
  // tensor within the block.
  class TensorAccessInfo {
   public:
    TensorAccessInfo() : is_valid_(true) {}
    void Update(SpatialLayout s) {
      if (!IsValid()) return;
      if (spatial_layout_.empty()) spatial_layout_ = s;
      if (!AreIdenticalSpatialAccess(s, spatial_layout_)) {
        Invalidate();
        return;
      }
    }
    bool IsValid() { return is_valid_; }
    void Invalidate() { is_valid_ = false; }
    SpatialLayout GetValidSpatialLayout() {
      if (!IsValid()) return {};
      return spatial_layout_;
    }

   private:
    bool is_valid_;
    SpatialLayout spatial_layout_;
  };

  // Helper to break down the indices of tensor access.
  SpatialLayout DetectTensorAccessIterMap(ffi::Array<PrimExpr> indices) {
    auto result = sym::DetectIterMap(
        /*indices=*/indices, /*input_iters*/ spatial_dom_,
        /*predicate*/ 1, /*check_level*/ sym::IterMapLevel::NoCheck, sym_analyzer_);
    if (result->indices.empty()) {
      DLOG(INFO) << "[LayoutInference] Failed to analyze indices " << indices
                 << ", error: " << result->errors;
      return {};
    }
    return GetSpatialLayout(result);
  }

  // Compute the spatial domain map of block
  void ComputeBlockSpatialDomain() {
    for (const s_tir::IterVar& v : block_->iter_vars) {
      if (v->iter_type == s_tir::kDataPar) {
        spatial_dom_.Set(v->var, v->dom.value());
        continue;
      }
      if (v->iter_type == s_tir::kCommReduce) continue;
      LOG(WARNING) << "[LayoutInference] Cannot compute block spatial domain in presence of "
                      "unknown block iter_type : "
                   << v->iter_type;
      can_transform_block_ = false;
      return;
    }
  }

  ffi::Optional<VisitInterrupt> Visit_(const s_tir::SBlockObj* op) final {
    // Blocks with nested blocks cannot be handled yet.
    LOG(WARNING) << "[LayoutInference] Nested blocks are not supported for layout inference yet";
    can_transform_block_ = false;

    return std::nullopt;
  }
  ffi::Optional<VisitInterrupt> Visit_(const TensorStoreObj* op) final {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(s_tir::StmtExprVisitor::Visit_(op));

    TensorAccessInfo& access_info =
        tensor_access_info_[op->dest.as_or_throw<tvm::tirx::TensorVar>()];

    // Fast path to ignore further analysis if we know that the tensor access is invalid.
    if (!access_info.IsValid()) return std::nullopt;

    // Only single write tensor is supported for each block.
    if (!op->dest.same_as(block_->writes[0]->source.as_or_throw<tvm::tirx::TensorVar>())) {
      access_info.Invalidate();
      LOG(WARNING) << "[LayoutInference] Exactly one write tensor is supported for layout "
                      "inference, found two: "
                   << op->dest << " and "
                   << block_->writes[0]->source.as_or_throw<tvm::tirx::TensorVar>();
      can_transform_block_ = false;
      return std::nullopt;
    }

    // If the write tensor access cannot be analyzed, no transformation to the block will be made.
    auto detected_spatial_layout = DetectTensorAccessIterMap(op->indices);
    if (detected_spatial_layout.empty()) {
      access_info.Invalidate();
      return std::nullopt;
    }

    // Check if we have access info for this tensor, if present, the two accesses must be
    // identical.
    access_info.Update(detected_spatial_layout);

    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadObj* op) final {
    TensorVar read_tensor = op->source.as_or_throw<tvm::tirx::TensorVar>();
    TensorAccessInfo& access_info =
        tensor_access_info_[op->source.as_or_throw<tvm::tirx::TensorVar>()];

    auto detected_spatial_layout = DetectTensorAccessIterMap(op->indices);

    if (detected_spatial_layout.empty()) {
      access_info.Invalidate();
      return std::nullopt;
    }
    access_info.Update(detected_spatial_layout);

    return std::nullopt;
  }

 public:
  bool CanBeTransformed() { return can_transform_block_; }
  IndexMap GetSBlockTransformation() { return block_transformation_.value(); }
  ffi::Map<TensorVar, IndexMap> GetReadTensorTransformations() {
    return read_tensor_transformations_;
  }

 private:
  bool can_transform_block_;
  IndexMap write_transformation_;
  ffi::Map<PrimVar, Range> spatial_dom_;
  sym::Analyzer sym_analyzer_;

  s_tir::SBlock block_;
  ffi::Optional<IndexMap> block_transformation_;

  ffi::Map<TensorVar, IndexMap> read_tensor_transformations_;
  const ffi::Map<TensorVar, IndexMap>& tensor_transformation_cache_;
  std::unordered_map<TensorVar, TensorAccessInfo, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      tensor_access_info_;
};

/*!
 * \brief Analyzes the tirx::Function and user provided output tensor transformations to propose
 * transformations of block and tensors within the tirx::Function.
 * \details It does a best effort analysis to propose transformations which would preserve
 * sequential access to tensors (especially output tensors). Since this is best effort, it is
 * possible that the tirx::Function is too complex for analysis. In such a case, no transformations
 * are proposed.
 */
class FunctionAnalyzer : public s_tir::StmtExprVisitor {
 public:
  explicit FunctionAnalyzer(const tirx::Function& func,
                            ffi::Array<IndexMap> write_transformations) {
    TVM_FFI_ICHECK_LE(write_transformations.size(), func->params.size())
        << "Incompatible Function and write_transformations";

    size_t first_write_index = func->params.size() - write_transformations.size();
    for (size_t i = 0; i < write_transformations.size(); ++i) {
      auto param = func->params[first_write_index + i];
      ffi::Optional<TensorVar> param_buf = param.as<TensorVar>();
      TVM_FFI_ICHECK(param_buf.has_value());
      TVM_FFI_ICHECK_EQ(param_buf.value().shape().size(),
                        write_transformations[i]->initial_indices.size())
          << "Mismatch between output tensor shape and index map";
      tensor_transformation_cache_.Set(param_buf.value(), write_transformations[i]);
    }
  }
  ffi::Map<s_tir::SBlock, ffi::Map<ffi::ObjectRef, IndexMap>> GetSuggestedTransforms() {
    ffi::Map<s_tir::SBlock, ffi::Map<ffi::ObjectRef, IndexMap>> result;
    for (const auto& [block, index_map] : block_transformations_) {
      ffi::Map<ffi::ObjectRef, IndexMap> block_transformations;
      block_transformations.Set(block, index_map);
      for (const auto& tensor : block_to_tensor_[block]) {
        block_transformations.Set(tensor, tensor_transformation_cache_[tensor]);
      }
      result.Set(block, block_transformations);
    }
    return result;
  }

 private:
  ffi::Optional<VisitInterrupt> Visit_(const s_tir::SBlockObj* op) final {
    if (op->name_hint == "root") {
      // Skip the root block
      return s_tir::StmtExprVisitor::Visit_(op);
    }

    s_tir::SBlock block = ffi::GetRef<s_tir::SBlock>(op);
    // Get block write tensor transformation.
    if (block->writes.size() != 1) return std::nullopt;
    auto write_tensor = block->writes[0]->source.as_or_throw<tvm::tirx::TensorVar>();
    block_to_tensor_[block].push_back(write_tensor);
    auto block_analyzer = ffi::make_object<BlockAnalyzer>(
        block, tensor_transformation_cache_, tensor_transformation_cache_[write_tensor]);
    block_analyzer->Analyze();

    if (!block_analyzer->CanBeTransformed()) return std::nullopt;
    // Collect the suggested transformations
    block_transformations_.Set(block, block_analyzer->GetSBlockTransformation());

    for (const auto& [tensor, index_map] : block_analyzer->GetReadTensorTransformations()) {
      // BlockAnalyzer makes sure that it does not propose transformation for a tensor for which a
      // transformation has already been proposed by other blocks or by write_transformations which
      // are input to this analysis.
      TVM_FFI_ICHECK_EQ(tensor_transformation_cache_.count(tensor), 0);
      tensor_transformation_cache_.Set(tensor, index_map);
      block_to_tensor_[block].push_back(tensor);
    }

    return std::nullopt;
  }

 private:
  ffi::Map<TensorVar, IndexMap> tensor_transformation_cache_;
  ffi::Map<s_tir::SBlock, IndexMap> block_transformations_;
  std::unordered_map<s_tir::SBlock, ffi::Array<TensorVar>, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      block_to_tensor_;
};

ffi::Map<s_tir::SBlock, ffi::Map<ffi::ObjectRef, tirx::IndexMap>> SuggestLayoutTransforms(
    const tirx::Function& function, ffi::Array<IndexMap> write_tensor_transformations) {
  // No changes to the tirx::Function are required if no transformations on output tensors.
  if (write_tensor_transformations.empty()) return {};

  auto analyzer = ffi::make_object<FunctionAnalyzer>(function, write_tensor_transformations);
  analyzer->Visit(function->body);
  return analyzer->GetSuggestedTransforms();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "relax.analysis.suggest_layout_transforms",
      [](tirx::Function fn, ffi::Array<tirx::IndexMap> write_tensor_transformations) {
        return SuggestLayoutTransforms(fn, write_tensor_transformations);
      });
}

}  // namespace relax
}  // namespace tvm
