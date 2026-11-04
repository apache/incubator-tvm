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
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/transform.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/transform.h>

#include <cmath>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::prim;
using namespace tvm::tirx;

using support::NDIntSet;

/*! \brief Type for multi-dimensional index */
using MultiIndex = std::vector<PrimExpr>;
/*! \brief Vector of int64_t */
using IntVec = std::vector<int64_t>;
/*! \brief Vector of for loops */
using ForVec = std::vector<const ForObj*>;

/*!
 * \brief An unordered_map for (for, tensor) => V
 * \tparam V The value type
 */
template <class V>
using ForTensorMap =
    std::unordered_map<const ForObj*,
                       std::unordered_map<TensorVar, V, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>>;

/*! \brief Given x, compute log2(|x| + 1) */
inline double slog(double x) { return x >= 0 ? std::log2(x + 1) : std::log2(-x + 1); }

namespace utils {

/*!
 * \brief Get the shape of the tensor
 * \param tensor The tensor
 * \param analyzer The analyzer
 * \return The shape of the tensor
 */
std::vector<int64_t> GetTensorShape(const TensorVar& tensor, sym::AnalyzerObj* analyzer) {
  int ndim = tensor.shape().size();
  std::vector<int64_t> result;
  result.reserve(ndim);
  for (const PrimExpr& i : tensor.shape()) {
    if (const IntImmObj* int_imm = i.as<IntImmObj>()) {
      result.push_back(static_cast<int64_t>(int_imm->value));
      continue;
    }
    sym::ConstIntBound bound = analyzer->const_int_bound(i);
    if (0 <= bound->max_value && bound->max_value < sym::ConstIntBound::kPosInf) {
      result.push_back(bound->max_value);
    } else {
      result.push_back(1);
    }
  }
  return result;
}

/*!
 * \brief Given a loop, return its `auto_unroll_max_step` annotation if it exists
 * \param loop The loop to be checked
 * \return The value of `auto_unroll_max_step` if it exists, or -1 if it does not exist
 */
int64_t GetAutoUnrollMaxStep(const ForObj* loop) {
  if (ffi::Optional<IntImm> auto_unroll =
          GetAnn<IntImm>(loop, tvm::tirx::attr::kAutoUnrollMaxStep.name)) {
    return static_cast<int64_t>(auto_unroll.value()->value);
  }
  return -1;
}

/*!
 * \brief Given a list of loops, return the extent of the first loop if the list is not empty,
 * and the first loop has constant extent. Otherwise returns the default value given
 * \param loops The list of loops to be checked
 * \param default_value The default value to be returned if the list is empty or the first loop
 * does not have constant extent
 * \return The extent of the first loop if the list is not empty, or the first loop has constant
 * extent. Otherwise returns the default value
 */
int64_t FirstLoopExtent(const ForVec& loops, int64_t default_value) {
  if (!loops.empty()) {
    PrimExpr extent_imm_expr = Range::GetExtentFromBeginEnd(loops[0]->begin, loops[0]->end);
    const auto* extent_imm = extent_imm_expr.as<IntImmObj>();
    if (auto extent = extent_imm ? extent_imm->value.as<int64_t>() : std::nullopt;
        extent.has_value()) {
      return *extent;
    }
  }
  return default_value;
}

/*!
 * \brief Relax each of the multi-indexing pattern according to the domains bound in the analyzer,
 * and then union them into a single region
 * \param multi_index_pattern A list of multi-index pattern to be relaxed
 * \param numel The size of the single region after union
 * \param analyzer The analyzer that contains the domain information
 * \return The relaxed and unioned region
 */
IntVec RelaxAndUnion(const std::vector<MultiIndex>& multi_indices, int64_t* numel,
                     sym::AnalyzerObj* analyzer) {
  *numel = 1;
  if (multi_indices.empty()) {
    return {};
  }
  int n_indices = multi_indices.size();
  int ndim = multi_indices[0].size();
  IntVec access_shape(ndim, 0);
  for (int i = 0; i < ndim; ++i) {
    int64_t minimum = sym::ConstIntBound::kPosInf;
    int64_t maximum = sym::ConstIntBound::kNegInf;
    for (int j = 0; j < n_indices; ++j) {
      sym::ConstIntBound bound = analyzer->const_int_bound(multi_indices[j][i]);
      minimum = std::min(minimum, bound->min_value);
      maximum = std::max(maximum, bound->max_value);
    }
    *numel *= maximum - minimum + 1;
    access_shape[i] = maximum - minimum + 1;
  }
  return access_shape;
}

/*!
 * \brief Given a list of multi-index pattern, return the minimal stride of a variable on it
 * \param multi_indices The list of multi-index pattern
 * \param tensor_stride The stride of the tensor
 * \param var The variable to be checked
 * \return The minimal stride of the variable on the multi-index pattern
 */
int64_t GetVarStride(const std::vector<MultiIndex>& multi_indices, const IntVec& tensor_stride,
                     const Var& var) {
  class CoefficientExtractor : public StmtExprVisitor {
   public:
    using StmtExprVisitor::Visit_;

    static int64_t Extract(const PrimExpr& expr, const Var& var) {
      auto extractor = ffi::make_object<CoefficientExtractor>(var);
      extractor->Visit(expr);
      return (extractor->visited_var && !extractor->visited_mul && !extractor->visited_add)
                 ? 1
                 : (extractor->visited_var ? extractor->stride : 0);
    }

    explicit CoefficientExtractor(const Var& var)
        : var(var), stride(0), visited_var(false), visited_add(false), visited_mul(false) {}

   private:
    ffi::Optional<VisitInterrupt> Visit_(const MulObj* node) override {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(node));
      if (visited_var && !visited_add) {
        if (const auto* a = node->a.as<IntImmObj>()) {
          visited_mul = true;
          stride = static_cast<int64_t>(a->value);
        } else if (const auto* b = node->b.as<IntImmObj>()) {
          visited_mul = true;
          stride = static_cast<int64_t>(b->value);
        }
      }
      return std::nullopt;
    }

    ffi::Optional<VisitInterrupt> Visit_(const AddObj* node) override {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(node));
      if (visited_var && !visited_mul) {
        visited_add = true;
        stride = 1;
      }
      return std::nullopt;
    }

    ffi::Optional<VisitInterrupt> Visit_(const VarObj* node) override {
      if (node == var.get()) {
        visited_var = true;
        stride = 2;
      }
      return std::nullopt;
    }

    const Var& var;
    int64_t stride;
    bool visited_var;
    bool visited_add;
    bool visited_mul;
  };

  constexpr int64_t kNotFound = std::numeric_limits<int64_t>::max();
  int ndim = tensor_stride.size();
  // Calculate the min stride possible
  int64_t result = kNotFound;
  for (const MultiIndex& multi_index : multi_indices) {
    TVM_FFI_ICHECK_EQ(multi_index.size(), tensor_stride.size());
    // Find the rightest dimension that contains the given variable
    for (int i = ndim - 1; i >= 0; --i) {
      int64_t coef = CoefficientExtractor::Extract(multi_index[i], var);
      if (coef != 0) {
        result = std::min(result, std::abs(coef) * tensor_stride[i]);
        break;
      }
    }
  }
  return (result == kNotFound) ? 0 : result;
}

/*!
 * \brief Converts a 2-dimensional STL vector to a TVM Tensor
 * \param src The source 2-dimensional STL vector
 * \param second_dim_size The length of the second dimension. When the first dim of src is 0,
 * second_dim_size must be specified, and in such case the shape of the result Tensor is
 * (0, second_dim_size).
 * \return The converted TVM Tensor
 */
ffi::Tensor AsTensor(const std::vector<std::vector<double>>& src, int second_dim_size = -1) {
  int n = src.size();
  TVM_FFI_ICHECK(!src.empty() || second_dim_size != -1);
  int m = src.empty() ? second_dim_size : src[0].size();
  ffi::Tensor tgt = runtime::Empty(
      /*shape=*/{n, m},
      /*dtype=*/DLDataType{kDLFloat, 64, 1},
      /*ctx=*/DLDevice{kDLCPU, 0});
  double* data = static_cast<double*>(tgt.data_ptr());
  for (const std::vector<double>& row : src) {
    for (double v : row) {
      *data++ = v;
    }
  }
  return tgt;
}

}  // namespace utils

namespace transform {

/*!
 * \brief Create a pass that simplifies the IR for feature extraction
 * \return The pass created
 */
Pass SimplifyForFeatureExtraction(bool normalize_thread_bindings = false) {
  class Simplifier : public StmtExprMutator {
   public:
    using StmtExprMutator::Mutate;
    using StmtExprMutator::Mutate_;

    explicit Simplifier(bool normalize_thread_bindings)
        : normalize_thread_bindings_(normalize_thread_bindings) {}

    static Stmt Run(Stmt stmt, bool normalize_thread_bindings) {
      return ffi::make_object<Simplifier>(normalize_thread_bindings)
          ->Mutate(stmt, InplaceMode::kAllow)
          .ValueOrUnchanged(std::move(stmt));
    }

   private:
    static bool HasTensorLoad(const PrimExpr& expr) {
      auto walk_fn = [](const TensorLoad&) -> ffi::Expected<ffi::WalkResult> {
        return ffi::WalkResult::Interrupt(ffi::VisitInterrupt(true));
      };
      auto result = ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(expr, walk_fn);
      return result.has_value() ? result.value()->value.cast<bool>() : false;
    }

    UnchangedOr<PrimExpr> Mutate_(const SelectObj* node, InplaceMode inplace_mode) final {
      if (normalize_thread_bindings_) return StmtExprMutator::Mutate_(node, inplace_mode);
      if (HasTensorLoad(node->true_value) || HasTensorLoad(node->false_value) ||
          HasTensorLoad(node->condition)) {
        return ffi::Unchanged();
      }
      return prim::MakeConst(node->ty.as_or_throw<PrimType>(), 1.0);
    }

    UnchangedOr<Stmt> Mutate_(const ForObj* loop, InplaceMode inplace_mode) final {
      auto mutate_retained_loop = [&]() -> UnchangedOr<Stmt> {
        if (!normalize_thread_bindings_) return StmtExprMutator::Mutate_(loop, inplace_mode);
        auto annotations = Mutate(loop->annotations, inplace_mode)
                               .as_or_throw<UnchangedOr<ffi::Map<ffi::String, ffi::Any>>>();
        auto result = StmtExprMutator::Mutate_(loop, inplace_mode);
        if (annotations.UnchangedOrSameAs(loop->annotations)) return result;
        For updated =
            std::move(result).ValueOrUnchanged(ffi::GetRef<Stmt>(loop)).as_or_throw<For>();
        updated.CopyOnWrite()->annotations = std::move(annotations).ValueUnchecked();
        return updated;
      };
      if (!normalize_thread_bindings_ &&
          IsZero(Range::GetExtentFromBeginEnd(loop->begin, loop->end))) {
        return Evaluate(0);
      }
      if (normalize_thread_bindings_ && tvm::s_tir::GetThreadBinding(loop).has_value()) {
        const auto tag = tvm::s_tir::GetThreadBinding(loop).value();
        if (tag.starts_with("vthread")) {
          // Virtual axes are independent iterations and isolate hardware aliases.
          thread_bindings_.push_back(nullptr);
          auto result = mutate_retained_loop();
          thread_bindings_.pop_back();
          return result;
        }
        for (auto it = thread_bindings_.rbegin(); it != thread_bindings_.rend() && *it; ++it) {
          const ForObj* outer = *it;
          if (tvm::s_tir::GetThreadBinding(outer).value() != tag) continue;
          // Feature counts describe hardware work, not repeated lexical names
          // for one axis. Normalize only nested aliases in this analysis input.
          sym::Analyzer analyzer;
          TVM_FFI_CHECK(
              analyzer->CanProveEqual(loop->begin, outer->begin) &&
                  analyzer->CanProveEqual(Range::GetExtentFromBeginEnd(loop->begin, loop->end),
                                          Range::GetExtentFromBeginEnd(outer->begin, outer->end)),
              ValueError)
              << "Incompatible nested thread extents for " << tag;
          auto previous_remap = VarRemapGet(loop->var);
          VarRemapSet(loop->var, prim::cast(outer->var, loop->var.ty()));
          auto annotations = Mutate(loop->annotations, inplace_mode)
                                 .as_or_throw<UnchangedOr<ffi::Map<ffi::String, ffi::Any>>>()
                                 .ValueOrUnchanged(loop->annotations);
          Stmt body = Mutate(loop->body, inplace_mode).ValueOrUnchanged(loop->body);
          VarRemapSet(loop->var, previous_remap);
          annotations.erase(tvm::s_tir::attr::kThreadBinding.name);
          if (!annotations.empty()) {
            PrimType ty = loop->var.ty();
            body = For(PrimVar("annotation", ty), IntImm(ty, 0), IntImm(ty, 1), IntImm(ty, 1),
                       ForKind::kDefault, std::move(body), std::move(annotations));
          }
          return body;
        }
        thread_bindings_.push_back(loop);
        auto result = mutate_retained_loop();
        thread_bindings_.pop_back();
        return result;
      }
      if (!normalize_thread_bindings_ && IsZero(loop->begin) &&
          IsOne(Range::GetExtentFromBeginEnd(loop->begin, loop->end)) &&
          loop->kind == ForKind::kDefault && loop->annotations.empty()) {
        VarRemapSet(loop->var, MakeConst(loop->var.ty(), 0.0));
        return Mutate(loop->body, inplace_mode).ValueOrUnchanged(loop->body);
      } else {
        return mutate_retained_loop();
      }
    }

    UnchangedOr<Stmt> Mutate_(const SBlockObj* op, InplaceMode inplace_mode) final {
      if (!normalize_thread_bindings_) return StmtExprMutator::Mutate_(op, inplace_mode);
      auto annotations = Mutate(op->annotations, inplace_mode)
                             .as_or_throw<UnchangedOr<ffi::Map<ffi::String, ffi::Any>>>();
      auto result = StmtExprMutator::Mutate_(op, inplace_mode);
      if (annotations.UnchangedOrSameAs(op->annotations)) return result;
      SBlock block =
          std::move(result).ValueOrUnchanged(ffi::GetRef<Stmt>(op)).as_or_throw<SBlock>();
      block.CopyOnWrite()->annotations = std::move(annotations).ValueUnchecked();
      return block;
    }

    UnchangedOr<Stmt> Mutate_(const RegionStmtObj* region, InplaceMode inplace_mode) final {
      thread_bindings_.push_back(nullptr);
      auto result = StmtExprMutator::Mutate_(region, inplace_mode);
      thread_bindings_.pop_back();
      return result;
    }

    bool normalize_thread_bindings_;
    std::vector<const ForObj*> thread_bindings_;
  };
  auto pass_func = [normalize_thread_bindings](Function f, IRModule m, PassContext ctx) {
    if (!f->body.has_value()) return f;
    FunctionObj* n = f.CopyOnWrite();
    n->body = Simplifier::Run(std::move(n->body).value(), normalize_thread_bindings);
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "tirx.SimplifyForFeatureExtraction");
}

/*!
 * \brief Create a list of passes that preprocesses the IR for feature extraction
 * \return The list of passes created
 */
tvm::transform::Sequential PassListForPerStoreFeature() {
  return tvm::transform::Sequential({
      s_tir::transform::RemoveWeightLayoutRewriteBlock(/*skip_tensor_rewrite*/ true),
      s_tir::transform::SimplifyForFeatureExtraction(),
      s_tir::transform::LowerCrossThreadReduction(),
      s_tir::transform::LowerInitBlock(),
      s_tir::transform::PlanAndUpdateTensorAllocationLocation(),
      s_tir::transform::ConvertBlocksToOpaque(),
      s_tir::transform::CompactTensorAllocation(),
      s_tir::transform::StmtSimplify(),
      s_tir::transform::LowerAutoCopy(),
      s_tir::transform::SimplifyForFeatureExtraction(/*normalize_thread_bindings=*/true),
      s_tir::transform::LowerMatchTensor(),
      s_tir::transform::StmtSimplify(),
  });
}

}  // namespace transform

/*! \brief A data structure managing loop nests */
struct LoopNest {
  int64_t prod = 1;    // The product of the extents of all the loops
  ForVec loops;        // All the loops
  IntVec auto_unroll;  // The loops with auto unroll policy
  ForVec parallel;     // The loops whose ForKind are kParallel
  ForVec vectorize;    // The loops whose ForKind are kVectorized
  ForVec unroll;       // The loops whose ForKind are kUnrolled
  ForVec blockIdx_x;   // The loops explicitly bound to blockIdx.x
  ForVec blockIdx_y;   // The loops explicitly bound to blockIdx.y
  ForVec blockIdx_z;   // The loops explicitly bound to blockIdx.z
  ForVec threadIdx_x;  // The loops explicitly bound to threadIdx.x
  ForVec threadIdx_y;  // The loops explicitly bound to threadIdx.y
  ForVec threadIdx_z;  // The loops explicitly bound to threadIdx.z
  ForVec vthread;      // The loops explicitly bound to vthread.*

  /*!
   * \brief Push a new loop into the loop nest
   * \param loop The loop to be pushed
   * \param auto_unroll_attr The auto unroll attribute of the loop
   * \return A list of for loops that the loop is bound to
   */
  ForVec* Push(const ForObj* loop, int64_t* auto_unroll_attr) {
    PrimExpr extent_imm_expr = Range::GetExtentFromBeginEnd(loop->begin, loop->end);
    const auto* extent_imm = extent_imm_expr.as<IntImmObj>();
    if (auto extent = extent_imm ? extent_imm->value.as<int64_t>() : std::nullopt;
        extent.has_value()) {
      this->prod *= *extent;
    }
    this->loops.push_back(loop);
    if ((*auto_unroll_attr = utils::GetAutoUnrollMaxStep(loop)) > 0) {
      this->auto_unroll.push_back(*auto_unroll_attr);
    }
    ForVec* ref_loops = nullptr;
    if (loop->kind == ForKind::kParallel && !tvm::s_tir::GetThreadBinding(loop).has_value()) {
      ref_loops = &parallel;
    } else if (loop->kind == ForKind::kVectorized) {
      ref_loops = &vectorize;
    } else if (loop->kind == ForKind::kUnrolled) {
      ref_loops = &unroll;
    } else if (tvm::s_tir::GetThreadBinding(loop).has_value()) {
      std::string thread_tag = tvm::s_tir::GetThreadBinding(loop).value();
      if (thread_tag == "blockIdx.x") {
        ref_loops = &blockIdx_x;
      } else if (thread_tag == "blockIdx.y") {
        ref_loops = &blockIdx_y;
      } else if (thread_tag == "blockIdx.z") {
        ref_loops = &blockIdx_z;
      } else if (thread_tag == "threadIdx.x") {
        ref_loops = &threadIdx_x;
      } else if (thread_tag == "threadIdx.y") {
        ref_loops = &threadIdx_y;
      } else if (thread_tag == "threadIdx.z") {
        ref_loops = &threadIdx_z;
      } else if (thread_tag.starts_with("vthread")) {
        ref_loops = &vthread;
      } else {
        TVM_FFI_THROW(ValueError) << "Unable to recognize thread tag: " << thread_tag;
      }
    }
    if (ref_loops != nullptr) {
      ref_loops->push_back(loop);
    }
    return ref_loops;
  }

  /*!
   * \brief Pop the last loop from the loop nest
   * \param loop The loop to be popped
   * \param ref_loops The list of for loops that the loop is bound to
   * \param auto_unroll_attr The auto unroll attribute of the loop
   */
  void Pop(const ForObj* loop, ForVec* ref_loops, int auto_unroll_attr) {
    if (ref_loops) {
      ref_loops->pop_back();
    }
    if (auto_unroll_attr > 0) {
      this->auto_unroll.pop_back();
    }
    PrimExpr extent_imm_expr = Range::GetExtentFromBeginEnd(loop->begin, loop->end);
    const auto* extent_imm = extent_imm_expr.as<IntImmObj>();
    if (auto extent = extent_imm ? extent_imm->value.as<int64_t>() : std::nullopt;
        extent.has_value()) {
      this->prod /= *extent;
    }
    this->loops.pop_back();
  }
};

/****** Group 1: Computation related features ******/

namespace group1 {

/*! \brief Group 1 features */
struct Feature {
  /*! \brief Arithmetic features */
  struct ArithOps {
    // Float-point arithmetic features
    int64_t float_mad = 0;         // The number of float MAD (Multiply–add) ops
    int64_t float_add_sub = 0;     // The number of float add and sub ops
    int64_t float_mul = 0;         // The number of float multiply ops
    int64_t float_div_mod = 0;     // The number of float div and mod ops
    int64_t float_cmp = 0;         // The number of float comparison ops
    int64_t float_math_func = 0;   // The number of float math func calls
    int64_t float_other_func = 0;  // The number of other float func calls
    // Integer arithmetic features
    int64_t int_mad = 0;         // The number of integer MAD (Multiply–add) ops
    int64_t int_add_sub = 0;     // The number of integer add and sub ops
    int64_t int_mul = 0;         // The number of integer multiply ops
    int64_t int_div_mod = 0;     // The number of integer div and mod ops
    int64_t int_cmp = 0;         // The number of integer comparison ops
    int64_t int_math_func = 0;   // The number of integer math func calls
    int64_t int_other_func = 0;  // The number of other integer func calls
    // Other arithmetic features
    int64_t bool_op = 0;    // The number of bool ops
    int64_t select_op = 0;  // The number of select ops

    static constexpr int64_t kCount = 16;

    ArithOps() = default;
    ArithOps(const TensorStoreObj* store, int64_t prod_loop_extent);

    void Export(std::vector<double>* v) const {
      double vs[] = {
          slog(float_mad), slog(float_add_sub),   slog(float_mul),        slog(float_div_mod),
          slog(float_cmp), slog(float_math_func), slog(float_other_func),  //
          slog(int_mad),   slog(int_add_sub),     slog(int_mul),          slog(int_div_mod),
          slog(int_cmp),   slog(int_math_func),   slog(int_other_func),  //
          slog(bool_op),   slog(select_op),
      };
      v->insert(v->end(), std::begin(vs), std::end(vs));
    }
  };

  /*! \brief Loop binding features */
  struct ForKindFeature {
    enum class Pos : int {
      kPosNone = 0,           // Does not have this kind of annotation
      kPosInnerSpatial = 1,   // The annotated iterator is the innermost spatial iterator
      kPosMiddleSpatial = 2,  // The annotated iterator is a middle spatial iterator
      kPosOuterSpatial = 3,   // The annotated iterator is the outermost spatial iterator
      kPosInnerReduce = 4,    // The annotated iterator is the innermost reduce iterator
      kPosMiddleReduce = 5,   // The annotated iterator is a middle reduce iterator
      kPosOuterReduce = 6,    // The annotated iterator is the outermost reduce iterator
      kPosMixed = 7,          // The annotated iterator is a mixed space and reduce iterator
    };
    int64_t num = 0;           // The number of iterators with the annotation
    int64_t prod = 0;          // The product of the lengths of iterators with the annotation
    int64_t len = 0;           // The length of the innermost iterator with the annotation
    Pos pos = Pos::kPosMixed;  // The position of the iterators with the annotation

    static constexpr int64_t kCount = 11;

    explicit ForKindFeature(const ForVec& loops);

    void Export(std::vector<double>* v) const {
      double vs[] = {
          slog(num),
          slog(prod),
          slog(len),
          static_cast<double>(static_cast<int>(pos) == 0),
          static_cast<double>(static_cast<int>(pos) == 1),
          static_cast<double>(static_cast<int>(pos) == 2),
          static_cast<double>(static_cast<int>(pos) == 3),
          static_cast<double>(static_cast<int>(pos) == 4),
          static_cast<double>(static_cast<int>(pos) == 5),
          static_cast<double>(static_cast<int>(pos) == 6),
          static_cast<double>(static_cast<int>(pos) == 7),
      };
      v->insert(v->end(), std::begin(vs), std::end(vs));
    }
  };

  ArithOps arith_ops;           // Arithmetic features
  ForKindFeature vectorize;     // Loop binding features: kVectorize
  ForKindFeature unroll;        // Loop binding features: kUnroll
  ForKindFeature parallel;      // Loop binding features: kParallel
  bool is_gpu = false;          // If the program is running on GPU
  int64_t blockIdx_x_len = 1;   // The length of blockIdx.x
  int64_t blockIdx_y_len = 1;   // The length of blockIdx.y
  int64_t blockIdx_z_len = 1;   // The length of blockIdx.z
  int64_t threadIdx_x_len = 1;  // The length of threadIdx.x
  int64_t threadIdx_y_len = 1;  // The length of threadIdx.y
  int64_t threadIdx_z_len = 1;  // The length of threadIdx.z
  int64_t vthread_len = 1;      // The length of virtual thread

  static constexpr int64_t kCount = ArithOps::kCount + ForKindFeature::kCount * 3 + 8;

  explicit Feature(const TensorStoreObj* store, const LoopNest& loop_nest, bool is_gpu)
      : arith_ops(store, loop_nest.prod),
        vectorize(loop_nest.vectorize),
        unroll(loop_nest.unroll),
        parallel(loop_nest.parallel) {
    if (is_gpu) {
      this->is_gpu = true;
      this->blockIdx_x_len = utils::FirstLoopExtent(loop_nest.blockIdx_x, 1);
      this->blockIdx_y_len = utils::FirstLoopExtent(loop_nest.blockIdx_y, 1);
      this->blockIdx_z_len = utils::FirstLoopExtent(loop_nest.blockIdx_z, 1);
      this->threadIdx_x_len = utils::FirstLoopExtent(loop_nest.threadIdx_x, 1);
      this->threadIdx_y_len = utils::FirstLoopExtent(loop_nest.threadIdx_y, 1);
      this->threadIdx_z_len = utils::FirstLoopExtent(loop_nest.threadIdx_z, 1);
      this->vthread_len = utils::FirstLoopExtent(loop_nest.vthread, 1);
    }
  }

  void Export(std::vector<double>* v) const {
    this->arith_ops.Export(v);
    this->vectorize.Export(v);
    this->unroll.Export(v);
    this->parallel.Export(v);
    double vs[] = {
        static_cast<double>(is_gpu),  //
        slog(blockIdx_x_len),        slog(blockIdx_y_len),  slog(blockIdx_z_len),
        slog(threadIdx_x_len),       slog(threadIdx_y_len), slog(threadIdx_z_len),
        slog(vthread_len),
    };
    v->insert(v->end(), std::begin(vs), std::end(vs));
  }
};

Feature::ArithOps::ArithOps(const TensorStoreObj* store, int64_t prod_loop_extent) {
  class ArithOpCounter : public StmtExprVisitor {
   public:
    using StmtExprVisitor::Visit_;

#define TVM_FEATURE_SIMPLE(Type, Counter)                      \
  ffi::Optional<VisitInterrupt> Visit_(const Type* op) final { \
    result_.Counter += this->prod_loop_extent_;                \
    return StmtExprVisitor::Visit_(op);                        \
  }
#define TVM_FEATURE_BINARY(Type, FloatCounter, IntCounter)                      \
  ffi::Optional<VisitInterrupt> Visit_(const Type* op) final {                  \
    if (op->ty.as_or_throw<PrimType>().MatchesCode(DLDataTypeCode::kDLFloat)) { \
      result_.FloatCounter += this->prod_loop_extent_;                          \
    } else {                                                                    \
      result_.IntCounter += this->prod_loop_extent_;                            \
    }                                                                           \
    return StmtExprVisitor::Visit_(op);                                         \
  }
    TVM_FEATURE_SIMPLE(AndObj, bool_op);
    TVM_FEATURE_SIMPLE(OrObj, bool_op);
    TVM_FEATURE_SIMPLE(NotObj, bool_op);
    TVM_FEATURE_SIMPLE(SelectObj, select_op);
    TVM_FEATURE_SIMPLE(prim::LShiftObj, int_math_func);
    TVM_FEATURE_SIMPLE(prim::RShiftObj, int_math_func);
    TVM_FEATURE_SIMPLE(prim::BitwiseAndObj, int_math_func);
    TVM_FEATURE_SIMPLE(prim::BitwiseOrObj, int_math_func);
    TVM_FEATURE_SIMPLE(prim::BitwiseXorObj, int_math_func);
    TVM_FEATURE_SIMPLE(prim::BitwiseNotObj, int_math_func);
    TVM_FEATURE_BINARY(AddObj, float_add_sub, int_add_sub);
    TVM_FEATURE_BINARY(SubObj, float_add_sub, int_add_sub);
    TVM_FEATURE_BINARY(MulObj, float_mul, int_mul);
    TVM_FEATURE_BINARY(DivObj, float_div_mod, int_div_mod);
    TVM_FEATURE_BINARY(ModObj, float_div_mod, int_div_mod);
    TVM_FEATURE_BINARY(FloorDivObj, float_div_mod, int_div_mod);
    TVM_FEATURE_BINARY(FloorModObj, float_div_mod, int_div_mod);
    TVM_FEATURE_BINARY(MaxObj, float_cmp, int_cmp);
    TVM_FEATURE_BINARY(MinObj, float_cmp, int_cmp);
    TVM_FEATURE_BINARY(EQObj, float_cmp, int_cmp);
    TVM_FEATURE_BINARY(NEObj, float_cmp, int_cmp);
    TVM_FEATURE_BINARY(LTObj, float_cmp, int_cmp);
    TVM_FEATURE_BINARY(LEObj, float_cmp, int_cmp);
    TVM_FEATURE_BINARY(GTObj, float_cmp, int_cmp);
    TVM_FEATURE_BINARY(GEObj, float_cmp, int_cmp);
#undef TVM_FEATURE_BINARY
#undef TVM_FEATURE_SIMPLE

    ffi::Optional<VisitInterrupt> Visit_(const CallObj* op) final {
      auto result_type = op->ty.as<PrimType>();
      if (!result_type) {
        return StmtExprVisitor::Visit_(op);
      }
      static auto op_call_effect_ = Op::GetAttrMap(tvm::op_attr::kCallEffectKind);
      CallEffectKind effect_kind = op_call_effect_[op->op.as_or_throw<Op>()];
      bool is_pure =
          effect_kind == CallEffectKind::kPure || effect_kind == CallEffectKind::kExprAnnotation;
      if (is_pure) {
        if (result_type.value().MatchesCode(DLDataTypeCode::kDLFloat)) {
          result_.float_math_func += prod_loop_extent_;
        } else {
          result_.int_math_func += prod_loop_extent_;
        }
      } else {
        if (result_type.value().MatchesCode(DLDataTypeCode::kDLFloat)) {
          result_.float_other_func += prod_loop_extent_;
        } else {
          result_.int_other_func += prod_loop_extent_;
        }
      }
      return StmtExprVisitor::Visit_(op);
    }

    int64_t prod_loop_extent_;
    ArithOps result_;
  };
  auto counter = ffi::make_object<ArithOpCounter>();
  counter->prod_loop_extent_ = prod_loop_extent;
  counter->Visit(store->value);
  *this = counter->result_;
}

Feature::ForKindFeature::ForKindFeature(const ForVec& loops) {
  if (loops.empty()) {
    this->num = 0;
    this->prod = 0;
    this->len = 0;
    this->pos = ForKindFeature::Pos::kPosNone;
  } else {
    PrimExpr last_loop_extent_imm_expr =
        Range::GetExtentFromBeginEnd(loops.back()->begin, loops.back()->end);
    const auto* last_loop_extent_imm = last_loop_extent_imm_expr.as<IntImmObj>();
    auto last_loop_extent =
        last_loop_extent_imm ? last_loop_extent_imm->value.as<int64_t>() : std::nullopt;
    this->num = loops.size();
    this->len = last_loop_extent.has_value() ? *last_loop_extent : 1;
    this->pos = ForKindFeature::Pos::kPosMixed;
    int64_t& prod = this->prod = 1;
    for (const ForObj* loop : loops) {
      PrimExpr extent_imm_expr = Range::GetExtentFromBeginEnd(loop->begin, loop->end);
      const auto* extent_imm = extent_imm_expr.as<IntImmObj>();
      if (auto extent = extent_imm ? extent_imm->value.as<int64_t>() : std::nullopt;
          extent.has_value()) {
        prod *= *extent;
      }
    }
  }
}

}  // namespace group1

namespace group2 {

/*! \brief Group 2 features */
struct Feature {
  enum class AccessType : int {
    /*! The tensor is read but not written */
    kRead = 0,
    /*! The tensor is written but not read */
    kWrite = 1,
    /*! The tensor is both read and written */
    kReadWrite = 2,
    /*! Unknown type */
    kUnknownRW = 3,
  };
  enum class ReuseType : int {
    /*! TensorVar reuse because accessed on each iteration of a loop */
    kLoopMultipleRead = 0,
    /*! TensorVar reuse because it is serially accessed */
    kSerialMultipleReadWrite = 1,
    /*! No tensor reuse */
    kNoReuse = 2,
  };

  struct SubFeature {
    /*! \brief The tensor this feature is for */
    TensorVar tensor;
    /*! \brief The access type of the tensor */
    AccessType access_type = AccessType::kUnknownRW;
    /*! \brief A list of multi-dimensonal indices used to access the tensor */
    std::vector<MultiIndex> multi_indices = {};
    // Access information
    /*! \brief loop_accessed_numel[i][...] means the number of elements accessed by loops[i] */
    std::vector<std::unordered_map<TensorVar, int64_t, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>>
        loop_accessed_numel = {};
    /*! \brief The shape of the data access */
    IntVec access_shape;
    /*! \brief The bytes that are continuously accessed */
    int64_t num_continuous_bytes = 1;
    // Stride information
    /*! \brief The min stride of the access */
    int64_t min_stride = 0;
    /*! \brief The innermost stride */
    int64_t innermost_stride = 0;
    /*! \brief The product of the non-strided loops */
    int64_t prod_non_strided_loop_extent = 0;
    // Reuse information
    /*! The type of data reuse */
    ReuseType reuse_type = ReuseType::kNoReuse;
    /*! The reuse distance in terms of number of iterations */
    double reuse_dis_iter = 0.0;
    /*! The reuse distance in terms of bytes */
    double reuse_dis_bytes = 0.0;
    /*! The reuse count */
    int64_t reuse_ct = 0;
    // Features
    /*! The touched memory in bytes */
    double bytes;
    /*! The touched unique memory in bytes */
    double unique_bytes;
    /*! The number of touched cache lines */
    double lines;
    /*! The number touched unique cache lines */
    double unique_lines;
    /*! bytes / reuse_ct */
    double bytes_d_reuse_ct;
    /*! unique_bytes / reuse_ct */
    double unique_bytes_d_reuse_ct;
    /*! lines / reuse_ct */
    double lines_d_reuse_ct;
    /*! unique_lines / reuse_ct */
    double unique_lines_d_reuse_ct;
    /*! The stride in access */
    double stride;

    static constexpr int64_t kCount = 18;

    void Export(std::vector<double>* v) const {
      double vs[] = {
          static_cast<double>(static_cast<int>(access_type) == 0),
          static_cast<double>(static_cast<int>(access_type) == 1),
          static_cast<double>(static_cast<int>(access_type) == 2),
          // FeatureSet::TensorAccess::AccessType::kUnknownRW is ignored
          slog(bytes),
          slog(unique_bytes),
          slog(lines),
          slog(unique_lines),
          static_cast<double>(static_cast<int>(reuse_type) == 0),
          static_cast<double>(static_cast<int>(reuse_type) == 1),
          static_cast<double>(static_cast<int>(reuse_type) == 2),
          slog(reuse_dis_iter),
          slog(reuse_dis_bytes),
          slog(reuse_ct),
          slog(bytes_d_reuse_ct),
          slog(unique_bytes_d_reuse_ct),
          slog(lines_d_reuse_ct),
          slog(unique_lines_d_reuse_ct),
          slog(stride),
      };
      v->insert(v->end(), std::begin(vs), std::end(vs));
    }

    static void Pad(std::vector<double>* v) { v->insert(v->end(), 18, 0.0); }

    void SetStride(const LoopNest& loop_nest, sym::AnalyzerObj* analyzer);

    void SetReuse(const LoopNest& loop_nest,     //
                  int64_t top_loop_touch_bytes,  //
                  const ForTensorMap<IntVec>& tensor_touched_under_loop);

    void SetFeature(const LoopNest& loop_nest, int64_t cache_line_bytes);

    explicit SubFeature(TensorVar tensor, AccessType access_type,
                        std::vector<MultiIndex> multi_indices, int n_loops)
        : tensor(std::move(tensor)),
          access_type(access_type),
          multi_indices(multi_indices),
          loop_accessed_numel(n_loops) {}
  };

  void Export(std::vector<double>* v, int tensors_per_store) const {
    int n = sub_features.size();
    for (int i = 0; i < tensors_per_store; ++i) {
      if (i < n) {
        sub_features[i].Export(v);
      } else {
        SubFeature::Pad(v);
      }
    }
  }

  explicit Feature(const TensorStoreObj* store, const LoopNest& loop_nest, int64_t cache_line_bytes,
                   IntVec* for_touched_bytes, ForTensorMap<IntVec>* tensor_touched_under_loop,
                   sym::AnalyzerObj* analyzer);

  void Init(const TensorStoreObj* store, int n_loops);

  void SetRegion(const LoopNest& loop_nest,                        //
                 IntVec* for_touched_bytes,                        //
                 ForTensorMap<IntVec>* tensor_touched_under_loop,  //
                 sym::AnalyzerObj* analyzer);

  std::vector<SubFeature> sub_features;
};

void Feature::Init(const TensorStoreObj* store, int n_loops) {
  struct Info {
    AccessType access_type = AccessType::kUnknownRW;
    std::vector<MultiIndex> multi_indices;
  };
  std::unordered_map<TensorVar, Info, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> tensor_info;
  {
    Info& info = tensor_info[store->dest.as_or_throw<TensorVar>()];
    info.access_type = AccessType::kWrite;
    info.multi_indices.push_back({store->indices.begin(), store->indices.end()});
  }
  auto walk_fn = [&tensor_info](const TensorLoad& load) -> ffi::Expected<ffi::WalkResult> {
    TensorVar tensor = load->source.as_or_throw<tvm::tirx::TensorVar>();
    Info& info = tensor_info[tensor];
    switch (info.access_type) {
      case AccessType::kRead:
        break;
      case AccessType::kWrite:
        info.access_type = AccessType::kReadWrite;
        break;
      case AccessType::kReadWrite:
        break;
      case AccessType::kUnknownRW:
      default:
        info.access_type = AccessType::kRead;
        break;
    }
    if (info.access_type != AccessType::kReadWrite) {
      info.multi_indices.push_back({load->indices.begin(), load->indices.end()});
    }
    return ffi::WalkResult::Advance();
  };
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(store->value, walk_fn);
  this->sub_features.reserve(tensor_info.size());
  for (const auto& kv : tensor_info) {
    this->sub_features.emplace_back(kv.first, kv.second.access_type,
                                    std::move(kv.second.multi_indices), n_loops);
  }
}

void Feature::SetRegion(const LoopNest& loop_nest, IntVec* for_touched_bytes,
                        ForTensorMap<IntVec>* tensor_touched_under_loop,
                        sym::AnalyzerObj* analyzer) {
  int n_loops = loop_nest.loops.size();
  const std::vector<const ForObj*>& loops = loop_nest.loops;
  // Step 1. Initialize and bind all the loop variables to a constant
  *for_touched_bytes = IntVec(n_loops, 0);
  for (int i = 0; i < n_loops; ++i) {
    const ForObj* loop = loops[i];
    {
      sym::ConstraintContext context;
      analyzer->GetConstraintContext()->ForEach([&](const sym::Constraint& premise) {
        if (auto binding = premise.as<sym::BindingConstraintObj>();
            binding && binding->var.same_as(loop->var))
          return;
        if (auto range = premise.as<sym::RangeConstraintObj>();
            range && range->var.same_as(loop->var))
          return;
        context->Append(premise);
      });
      context->Append(sym::BindingConstraint(loop->var, loop->begin));
      analyzer->SyncConstraintContext(context);
    }
  }
  // Step 2. Corner case: no loops
  if (n_loops == 0) {
    // In this case, the `access_shape` is not calculated
    for (SubFeature& feature : sub_features) {
      feature.access_shape = IntVec(feature.tensor.shape().size(), 1);
    }
    return;
  }
  // Step 3. Gradually bind the loops from inner to outer,
  // calculate the area the loops touch on each tensor
  for (int i = n_loops - 1; i >= 0; --i) {
    const ForObj* loop = loops[i];
    {
      sym::ConstraintContext context;
      analyzer->GetConstraintContext()->ForEach([&](const sym::Constraint& premise) {
        if (auto binding = premise.as<sym::BindingConstraintObj>();
            binding && binding->var.same_as(loop->var))
          return;
        if (auto range = premise.as<sym::RangeConstraintObj>();
            range && range->var.same_as(loop->var))
          return;
        context->Append(premise);
      });
      context->Append(sym::RangeConstraint(loop->var, Range(loop->begin, loop->end)));
      analyzer->SyncConstraintContext(context);
    }
    int64_t& touched_bytes = (*for_touched_bytes)[i] = 0;
    for (SubFeature& feature : sub_features) {
      TensorVar tensor = feature.tensor;
      // Note: `feature.access_shape` for `i == 0` is the only one preserved,
      // while others are discarded
      int64_t numel;
      feature.access_shape = utils::RelaxAndUnion(feature.multi_indices, &numel, analyzer);
      numel = std::max<int64_t>(0, numel);
      feature.loop_accessed_numel[i][tensor] = numel;
      touched_bytes += numel * static_cast<int64_t>(tensor.dtype().StorageBytes());
      (*tensor_touched_under_loop)[loop][tensor].push_back(numel);
    }
  }
}

void Feature::SubFeature::SetStride(const LoopNest& loop_nest, sym::AnalyzerObj* analyzer) {
  int n_loops = loop_nest.loops.size();
  const std::vector<const ForObj*>& loops = loop_nest.loops;
  // For each tensor, we find the loop stride on it
  TensorVar tensor = this->tensor;
  int ndim = tensor.shape().size();
  IntVec tensor_shape = utils::GetTensorShape(tensor, analyzer);
  // Calculate the tensor's stride from its shape
  IntVec tensor_stride(ndim);
  if (ndim >= 1) {
    tensor_stride[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
      tensor_stride[i] = tensor_stride[i + 1] * tensor_shape[i + 1];
    }
  }
  // Calculate `num_continuous_bytes`
  {
    int64_t& num_continuous_bytes = this->num_continuous_bytes = 1;
    const IntVec& access_shape = this->access_shape;
    TVM_FFI_ICHECK_EQ(access_shape.size(), tensor_shape.size());
    for (int i = ndim - 1; i >= 0; --i) {
      if (access_shape[i] == tensor_shape[i]) {
        num_continuous_bytes =
            tensor_shape[i] * static_cast<int64_t>(tensor.dtype().StorageBytes());
        break;
      }
    }
  }
  // Enumerate loops from inner to outer
  int i = 0;
  // Calculate this->min_stride
  int64_t& stride = this->min_stride = 0;
  for (i = n_loops - 1; i >= 0; --i) {
    stride = utils::GetVarStride(this->multi_indices, tensor_stride, loops[i]->var);
    if (stride != 0) {
      break;
    }
  }
  // Calculate this->innermost_stride
  this->innermost_stride = (i == n_loops - 1) ? stride : 0;
  // Calculate this->prod
  int64_t& prod = this->prod_non_strided_loop_extent = 1;
  for (int j = n_loops - 1; j > i; --j) {
    PrimExpr extent_imm_expr = Range::GetExtentFromBeginEnd(loops[j]->begin, loops[j]->end);
    const auto* extent_imm = extent_imm_expr.as<IntImmObj>();
    if (auto extent = extent_imm ? extent_imm->value.as<int64_t>() : std::nullopt;
        extent.has_value()) {
      prod *= *extent;
    }
  }
}

void Feature::SubFeature::SetReuse(const LoopNest& loop_nest, int64_t top_loop_touch_bytes,
                                   const ForTensorMap<IntVec>& tensor_touched_under_loop) {
  TensorVar tensor = this->tensor;
  // Step 3.1. Collect all `Var`s that appears in the tensor region
  std::unordered_set<const VarObj*> region_vars;
  auto walk_fn = [&region_vars](const Var& var) -> ffi::Expected<ffi::WalkResult> {
    if (auto prim_var = var.as<PrimVar>()) {
      region_vars.insert(prim_var.value().get());
    }
    return ffi::WalkResult::Advance();
  };
  for (const MultiIndex& multi_index : this->multi_indices) {
    for (const PrimExpr& index : multi_index) {
      ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(index, walk_fn);
    }
  }
  // Default case: no reuse
  ReuseType& reuse_type = this->reuse_type = ReuseType::kNoReuse;
  double& reuse_dis_iter = this->reuse_dis_iter = 0;
  double& reuse_dis_bytes = this->reuse_dis_bytes = 0;
  int64_t& reuse_ct = this->reuse_ct = 0;

  // Step 3.2. Enumerate loops from inner to outer, find the first loop with reuse
  int n_loops = loop_nest.loops.size();
  const std::vector<const ForObj*>& loops = loop_nest.loops;
  for (int i = n_loops - 1; i >= 0; --i) {
    const ForObj* loop = loops[i];
    // Case 1. Find an invariant loop, i.e. reuse with kLoopMultipleRead
    if (!region_vars.count(loop->var.get())) {
      reuse_type = ReuseType::kLoopMultipleRead;
      PrimExpr extent_imm_expr = Range::GetExtentFromBeginEnd(loop->begin, loop->end);
      const auto* extent_imm = extent_imm_expr.as<IntImmObj>();
      if (auto extent = extent_imm ? extent_imm->value.as<int64_t>() : std::nullopt;
          extent.has_value()) {
        reuse_ct = *extent;
      } else {
        reuse_ct = 1;
      }
      reuse_dis_iter = 1;
      for (int j = n_loops - 1; j > i; --j) {
        PrimExpr extent_imm_expr = Range::GetExtentFromBeginEnd(loops[j]->begin, loops[j]->end);
        const auto* extent_imm = extent_imm_expr.as<IntImmObj>();
        if (auto extent = extent_imm ? extent_imm->value.as<int64_t>() : std::nullopt;
            extent.has_value()) {
          reuse_dis_iter *= *extent;
        }
      }
      reuse_dis_bytes = 0.0;
      if (i == n_loops - 1) {
        reuse_dis_bytes = top_loop_touch_bytes;
      } else {
        for (const auto& iter : tensor_touched_under_loop.at(loops[i + 1])) {
          TensorVar tensor = iter.first;
          const IntVec& numels = iter.second;
          int64_t numel = std::accumulate(numels.begin(), numels.end(), int64_t(0));
          reuse_dis_bytes += numel * static_cast<int64_t>(tensor.dtype().StorageBytes());
        }
      }
      break;
    }
    // Case 2. Find serial reuse, i.e. reuse with kSerialMultipleReadWrite
    const IntVec& touched = tensor_touched_under_loop.at(loop).at(tensor);
    if (touched.size() >= 2) {
      int64_t extent = 1;
      PrimExpr ext_imm_expr = Range::GetExtentFromBeginEnd(loop->begin, loop->end);
      const auto* ext_imm = ext_imm_expr.as<IntImmObj>();
      if (auto ext = ext_imm ? ext_imm->value.as<int64_t>() : std::nullopt; ext.has_value()) {
        extent = *ext;
      }
      reuse_type = ReuseType::kSerialMultipleReadWrite;
      reuse_ct = touched.size() - 1;
      reuse_dis_iter = *std::min_element(touched.begin(), touched.end());
      reuse_dis_bytes = 0.0;
      for (const auto& iter : tensor_touched_under_loop.at(loop)) {
        TensorVar tensor = iter.first;
        const IntVec& numels = iter.second;
        int64_t numel = std::accumulate(numels.begin(), numels.end(), int64_t(0));
        reuse_dis_bytes += numel * static_cast<int64_t>(tensor.dtype().StorageBytes());
      }
      reuse_dis_iter /= extent;
      reuse_dis_bytes /= extent;
      break;
    }
  }
}

void Feature::SubFeature::SetFeature(const LoopNest& loop_nest, int64_t cache_line_bytes) {
  int64_t dtype_bytes = static_cast<int64_t>(this->tensor.dtype().StorageBytes());
  this->stride = this->innermost_stride;
  this->bytes = dtype_bytes * loop_nest.prod;
  if (loop_nest.loops.empty()) {
    this->unique_bytes = 1;
    this->lines = 1;
    this->unique_lines = 1;
  } else {
    this->unique_bytes =
        static_cast<double>(this->loop_accessed_numel.front().at(tensor)) * dtype_bytes;
    this->lines = static_cast<double>(loop_nest.prod) / this->prod_non_strided_loop_extent *
                  std::min(1.0, 1.0 * this->min_stride * dtype_bytes / cache_line_bytes);
    this->lines = std::max(1.0, this->lines);
    this->unique_lines = static_cast<double>(this->unique_bytes) /
                         std::min(cache_line_bytes, this->num_continuous_bytes);
    this->unique_lines = std::max(1.0, this->unique_lines);
  }
  double proxy_reuse_ct = this->reuse_ct > 0 ? this->reuse_ct : 0.5;
  this->bytes_d_reuse_ct = this->bytes / proxy_reuse_ct;
  this->unique_bytes_d_reuse_ct = this->unique_bytes / proxy_reuse_ct;
  this->lines_d_reuse_ct = this->lines / proxy_reuse_ct;
  this->unique_lines_d_reuse_ct = this->unique_lines / proxy_reuse_ct;
}

Feature::Feature(const TensorStoreObj* store, const LoopNest& loop_nest, int64_t cache_line_bytes,
                 IntVec* for_touched_bytes, ForTensorMap<IntVec>* tensor_touched_under_loop,
                 sym::AnalyzerObj* analyzer) {
  int n_loops = loop_nest.loops.size();
  // Step 0. Initialize data structures
  this->Init(store, n_loops);
  // Step 1. Calculate region-related feature
  this->SetRegion(loop_nest, for_touched_bytes, tensor_touched_under_loop, analyzer);
  // Step 2. Calculate stride-related feature
  for (auto& feature : sub_features) {
    feature.SetStride(loop_nest, analyzer);
  }
  // Step 3. Calculate reuse-related feature
  int64_t top_loop_touch_bytes = 0.0;
  if (n_loops > 0) {
    for (const SubFeature& feature : sub_features) {
      int64_t bytes = static_cast<int64_t>(feature.tensor.dtype().StorageBytes());
      int64_t n_tensor = feature.loop_accessed_numel[0].size();
      top_loop_touch_bytes += bytes * n_tensor;
    }
  }
  for (auto& feature : sub_features) {
    feature.SetReuse(loop_nest, top_loop_touch_bytes, *tensor_touched_under_loop);
  }
  // Step 4. Calculate rest of the features
  for (auto& feature : sub_features) {
    feature.SetFeature(loop_nest, cache_line_bytes);
  }
  // Step 5. Sort the features
  std::sort(sub_features.begin(), sub_features.end(), [](const SubFeature& a, const SubFeature& b) {
    if (a.lines != b.lines) {
      return a.lines > b.lines;
    }
    if (a.bytes != b.bytes) {
      return a.bytes > b.bytes;
    }
    return a.tensor->name < b.tensor->name;
  });
}

}  // namespace group2

namespace group3 {

/*! \brief Group 3 feature */
struct Feature {
  /*!
   * \brief See the wiki page [1] for details
   *
   * Arithmetic intensity is FLOPs/unique bytes of memory touched. A value is computed
   * for each set of loop nests starting with just the innermost loop and
   * reaching to include all loops. There are a variable number of loops, so
   * n_samples are taken from the curve of arithmetic intensity vs flops. This
   * biases the values towards larger loops.
   *
   * Note that the denominator is unique bytes of memory touched. Repeated
   * access to the same byte of memory counts as only a single byte touched.
   *
   * Values are scaled by log2(x + 1).
   *
   * [1] https://en.wikipedia.org/wiki/Roofline_model
   */
  std::vector<double> arith_intensity_curve;

  void Export(std::vector<double>* v) const {
    v->insert(v->end(), arith_intensity_curve.begin(), arith_intensity_curve.end());
  }

  explicit Feature(int n_samples, const LoopNest& loop_nest, const IntVec& for_touched_bytes,
                   const group1::Feature::ArithOps& arith_ops)
      : arith_intensity_curve(n_samples, 0.0) {
    const std::vector<const ForObj*>& loops = loop_nest.loops;
    TVM_FFI_ICHECK_EQ(loops.size(), for_touched_bytes.size());
    int n_loops = loops.size();
    // Calculate `memory_bytes`
    std::vector<double> memory_bytes;
    memory_bytes.resize(n_loops);
    for (int i = 0; i < n_loops; ++i) {
      memory_bytes[n_loops - 1 - i] = for_touched_bytes[i];
    }
    // Calculate `compute_ops` and `cur_compute_ops`
    std::vector<double> compute_ops;
    double total_compute_ops = arith_ops.float_mad + arith_ops.float_add_sub + arith_ops.float_mul +
                               arith_ops.float_div_mod + arith_ops.float_cmp +
                               arith_ops.float_math_func + arith_ops.float_other_func;
    total_compute_ops /= loop_nest.prod;
    for (int i = n_loops - 1; i >= 0; --i) {
      PrimExpr extent_imm_expr = Range::GetExtentFromBeginEnd(loops[i]->begin, loops[i]->end);
      const auto* extent_imm = extent_imm_expr.as<IntImmObj>();
      if (auto extent = extent_imm ? extent_imm->value.as<int64_t>() : std::nullopt;
          extent.has_value()) {
        total_compute_ops *= *extent;
      }
      compute_ops.push_back(total_compute_ops);
    }
    // Fill the feature set
    if (total_compute_ops <= 0 || compute_ops.empty()) {
      for (int i = 0; i < n_samples; ++i) {
        arith_intensity_curve[i] = 0.0;
      }
      return;
    }
    total_compute_ops = compute_ops.back();
    int p = 0;
    for (int i = 0; i < n_samples; ++i) {
      double& result = arith_intensity_curve[i];
      double cur_compute_ops = static_cast<double>(i + 1) / n_samples * total_compute_ops;
      // Find the first `p` that `compute[p] >= total * (i + 1) / N`
      for (; p < n_loops; ++p) {
        if (compute_ops[p] >= cur_compute_ops - 1e-4) {
          break;
        }
      }
      TVM_FFI_ICHECK_LT(p, n_loops);
      if (p == 0) {
        result = slog(compute_ops[p] / memory_bytes[p]);
      } else {
        double base = compute_ops[p - 1] / memory_bytes[p - 1];
        double slope =
            (compute_ops[p] / memory_bytes[p] - compute_ops[p - 1] / memory_bytes[p - 1]) /
            (compute_ops[p] - compute_ops[p - 1]);
        result = slog(base + slope * (cur_compute_ops - compute_ops[p - 1]));
      }
    }
  }
};

}  // namespace group3

namespace group4 {

/*! \brief Group 4 feature */
struct Feature {
  int64_t alloc_size = 0;        // The size of allocated tensor in bytes
  int64_t alloc_prod = 0;        // alloc_outer_prod * alloc_inner_prod
  int64_t alloc_outer_prod = 1;  // The product of lengths of loops outside the scope of the alloc

  static constexpr int64_t kCount = 4;

  void Export(std::vector<double>* v, int64_t outer_prod) const {
    double vs[] = {
        slog(alloc_size),
        slog(alloc_prod),
        slog(alloc_outer_prod),
        slog(static_cast<double>(outer_prod) / alloc_outer_prod),
    };
    v->insert(v->end(), std::begin(vs), std::end(vs));
  }

  Feature() = default;

  explicit Feature(const LoopNest& loop_nest, const TensorVar& tensor, sym::AnalyzerObj* analyzer) {
    std::vector<int64_t> shape = utils::GetTensorShape(tensor, analyzer);
    int64_t numel = 1;
    for (int64_t x : shape) {
      numel *= x;
    }
    alloc_size = numel * static_cast<int64_t>(tensor.dtype().StorageBytes());
    alloc_prod = numel * loop_nest.prod;
    alloc_outer_prod = loop_nest.prod;
  }
};

}  // namespace group4

namespace group5 {

/*! \brief Group 5 feature */
struct Feature {
  int64_t outer_prod;        // The product of lengths of outer loops
  int num_loops;             // The number of outer loops
  int auto_unroll_max_step;  // The value of annotation "tirx.auto_unroll_max_step"

  static constexpr int64_t kCount = 3;

  void Export(std::vector<double>* v) const {
    double vs[] = {
        slog(outer_prod),
        slog(num_loops),
        slog(auto_unroll_max_step),
    };
    v->insert(v->end(), std::begin(vs), std::end(vs));
  }

  explicit Feature(const LoopNest& loop_nest) {
    this->outer_prod = loop_nest.prod;
    this->num_loops = loop_nest.loops.size();
    this->auto_unroll_max_step = loop_nest.auto_unroll.empty() ? 0 : loop_nest.auto_unroll.back();
  }
};

}  // namespace group5

namespace group6 {

/*! \brief The auxiliary feature extractor for workloads */
class WorkloadEmbeddingExtractor : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;

  ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
    if (value.as<ExprObj>()) return std::nullopt;
    return StmtExprVisitor::Visit(value);
  }

  static std::vector<double> Extract(const IRModule& mod) {
    auto self = ffi::make_object<WorkloadEmbeddingExtractor>();
    for (const auto& kv : mod->functions) {
      if (const FunctionObj* func = kv.second.as<FunctionObj>()) {
        self->Visit(func->body);
      }
    }
    return self->embedding;
  }

 private:
  ffi::Optional<VisitInterrupt> Visit_(const SBlockObj* block) final {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(block));
    std::string name = block->name_hint;
    std::for_each(name.begin(), name.end(), [](char& c) { c = ::tolower(c); });
    if (name.find("softmax") != std::string::npos) {
      embedding[0] = 1.0;
    } else if ((name.find("max") != std::string::npos) || (name.find("min") != std::string::npos)) {
      embedding[1] = 1.0;
    } else if (name.find("add") != std::string::npos) {
      embedding[2] = 1.0;
    } else if (name.find("batch_matmul") != std::string::npos) {
      embedding[3] = 1.0;
    } else if (name.find("matmul") != std::string::npos) {
      embedding[4] = 1.0;
    } else if (name.find("depthwiseconv2d") != std::string::npos) {
      embedding[5] = 1.0;
    } else if (name.find("conv2d_winograd") != std::string::npos) {
      embedding[6] = 1.0;
    } else if (name.find("conv2d") != std::string::npos) {
      embedding[7] = 1.0;
    }
    return std::nullopt;
  }

  std::vector<double> embedding = std::vector<double>(8, 0.0);
};

/*! \brief Group 6 feature */
struct Feature {
  explicit Feature(const IRModule& mod) {
    this->feature = WorkloadEmbeddingExtractor::Extract(mod);
  }

  void Export(std::vector<double>* v) const {
    v->insert(v->end(), std::begin(feature), std::end(feature));
  }

  std::vector<double> feature;  // The workload embedding
  static constexpr int64_t kCount = 8;
};

}  // namespace group6

/*! \brief The feature extracted */
struct Feature {
  const VarObj* tensor = nullptr;
  int tensor_order = -1;
  std::unique_ptr<group1::Feature> group1 = nullptr;
  std::unique_ptr<group2::Feature> group2 = nullptr;
  std::unique_ptr<group3::Feature> group3 = nullptr;
  std::unique_ptr<group4::Feature> group4 = nullptr;
  std::unique_ptr<group5::Feature> group5 = nullptr;
  std::shared_ptr<group6::Feature> group6 = nullptr;

  bool operator<(const Feature& other) const { return tensor_order < other.tensor_order; }
};

/*! \brief The main feature extractor */
class PerStoreFeatureCollector : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;

  ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
    if (value.as<ExprObj>()) return std::nullopt;
    return StmtExprVisitor::Visit(value);
  }

  static std::vector<Feature> Collect(bool is_gpu, int64_t cache_line_bytes,
                                      int64_t arith_intensity_curve_num_samples,
                                      const IRModule& mod) {
    auto collector = ffi::make_object<PerStoreFeatureCollector>(is_gpu, cache_line_bytes,
                                                                arith_intensity_curve_num_samples);
    for (const auto& kv : mod->functions) {
      if (const FunctionObj* func = kv.second.as<FunctionObj>()) {
        collector->Visit(func->body);
        for (const Var& param : func->params) {
          if (auto tensor = param.as<TensorVar>()) {
            collector->HandleTensorAlloc(tensor.value());
          }
        }
      }
    }
    std::vector<Feature> result;
    result.reserve(collector->tensor_features_.size());
    for (auto& it : collector->tensor_features_) {
      Feature& feature = it.second;
      if (feature.tensor != nullptr) {
        TVM_FFI_ICHECK(feature.group1);
        TVM_FFI_ICHECK(feature.group2);
        TVM_FFI_ICHECK(feature.group3);
        TVM_FFI_ICHECK(feature.group5);
        if (feature.group4 == nullptr) {
          feature.group4 = std::make_unique<group4::Feature>();
        }
        result.push_back(std::move(feature));
      }
    }
    std::sort(result.begin(), result.end());
    return result;
  }

 private:
  ffi::Optional<VisitInterrupt> Visit_(const ForObj* loop) final {
    int64_t auto_unroll;
    ForVec* for_vec = loop_nest_.Push(loop, &auto_unroll);
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(loop));
    loop_nest_.Pop(loop, for_vec, auto_unroll);
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorStoreObj* store) final {
    if (store->value->IsInstance<IntImmObj>() || store->value->IsInstance<FloatImmObj>()) {
      return std::nullopt;
    }
    const VarObj* tensor = store->dest.as_or_throw<TensorVar>().get();
    Feature& feature = tensor_features_[tensor];
    if (feature.tensor == nullptr) {
      feature.tensor = tensor;
      feature.tensor_order = tensor_features_.size();
    }
    feature.group1 = std::make_unique<group1::Feature>(store, loop_nest_, is_gpu_);
    feature.group2 =
        std::make_unique<group2::Feature>(store, loop_nest_, cache_line_bytes_, &for_touched_bytes_,
                                          &tensor_touched_under_loop_, analyzer_.get());
    feature.group3 =
        std::make_unique<group3::Feature>(arith_intensity_curve_num_samples_, loop_nest_,
                                          for_touched_bytes_, feature.group1->arith_ops);
    feature.group5 = std::make_unique<group5::Feature>(loop_nest_);
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const SBlockObj* block) final {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(block));
    for (const TensorVar& tensor : block->alloc_tensors) {
      HandleTensorAlloc(tensor);
    }
    return std::nullopt;
  }

  void HandleTensorAlloc(const TensorVar& tensor) {
    Feature& feature = tensor_features_[tensor.get()];
    feature.group4 = std::make_unique<group4::Feature>(loop_nest_, tensor, analyzer_.get());
  }

 public:
  explicit PerStoreFeatureCollector(bool is_gpu, int64_t cache_line_bytes,
                                    int64_t arith_intensity_curve_num_samples)
      : is_gpu_(is_gpu),
        cache_line_bytes_(cache_line_bytes),
        arith_intensity_curve_num_samples_(arith_intensity_curve_num_samples) {}

 private:
  bool is_gpu_;
  int64_t cache_line_bytes_;
  int64_t arith_intensity_curve_num_samples_;
  sym::Analyzer analyzer_;
  LoopNest loop_nest_ = {};
  IntVec for_touched_bytes_ = {};
  ForTensorMap<IntVec> tensor_touched_under_loop_ = {};
  std::unordered_map<const VarObj*, Feature> tensor_features_ = {};
};

}  // namespace s_tir
}  // namespace tvm

namespace tvm {
namespace s_tir {
namespace meta_schedule {

class PerStoreFeatureObj : public FeatureExtractorObj {
 public:
  int tensors_per_store;
  int arith_intensity_curve_num_samples;
  int cache_line_bytes;
  bool extract_workload;
  int feature_vector_length;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PerStoreFeatureObj>()
        .def_ro("tensors_per_store", &PerStoreFeatureObj::tensors_per_store)
        .def_ro("arith_intensity_curve_num_samples",
                &PerStoreFeatureObj::arith_intensity_curve_num_samples)
        .def_ro("cache_line_bytes", &PerStoreFeatureObj::cache_line_bytes)
        .def_ro("extract_workload", &PerStoreFeatureObj::extract_workload)
        .def_ro("feature_vector_length", &PerStoreFeatureObj::feature_vector_length);
  }

  void ExtractSingle(IRModule mod, bool is_gpu, std::vector<std::vector<double>>* results) {
    static tvm::transform::Sequential passes = s_tir::transform::PassListForPerStoreFeature();
    mod = passes(std::move(mod));
    std::vector<s_tir::Feature> features = s_tir::PerStoreFeatureCollector::Collect(
        is_gpu, this->cache_line_bytes, this->arith_intensity_curve_num_samples, mod);
    int n_features = features.size();
    results->resize(n_features);
    for (int i = 0; i < n_features; ++i) {
      const s_tir::Feature& feature = features[i];
      std::vector<double>& result = (*results)[i];
      result.reserve(feature_vector_length);
      feature.group1->Export(&result);
      feature.group2->Export(&result, this->tensors_per_store);
      feature.group3->Export(&result);
      feature.group4->Export(&result, feature.group5->outer_prod);
      feature.group5->Export(&result);
    }
  }

  ffi::Array<ffi::Tensor> ExtractFrom(const TuneContext& tune_context,
                                      const ffi::Array<MeasureCandidate>& candidates) {
    auto& target_keys = tune_context->target.value()->keys;
    bool is_gpu = std::find(target_keys.begin(), target_keys.end(), "gpu") != target_keys.end();
    std::vector<ffi::Optional<ffi::Tensor>> results;
    results.resize(candidates.size());
    std::unique_ptr<s_tir::group6::Feature> feature_group6 = nullptr;
    if (extract_workload) {
      feature_group6 = std::make_unique<s_tir::group6::Feature>(tune_context->mod.value());
    }
    auto f = [this, is_gpu, &feature_group6, &candidates, &results](int, int task_id) -> void {
      const auto& candidate = candidates[task_id];
      std::vector<std::vector<double>> features;
      ExtractSingle(DeepCopyIRModule(candidate->sch->mod()), is_gpu, &features);
      if (extract_workload) {
        for (auto& feature : features) {
          feature_group6->Export(&feature);
        }
      }
      results[task_id] = s_tir::utils::AsTensor(features, this->feature_vector_length);
    };
    support::parallel_for_dynamic(0, candidates.size(), tune_context->num_threads, f);
    ffi::Array<ffi::Tensor> features;
    features.reserve(results.size());
    for (auto& result : results) features.push_back(std::move(result).value());
    return features;
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.PerStoreFeature", PerStoreFeatureObj,
                                    FeatureExtractorObj);
};

FeatureExtractor FeatureExtractor::PerStoreFeature(int tensors_per_store,
                                                   int arith_intensity_curve_num_samples,
                                                   int cache_line_bytes, bool extract_workload) {
  ffi::ObjectPtr<PerStoreFeatureObj> n = ffi::make_object<PerStoreFeatureObj>();
  n->tensors_per_store = tensors_per_store;
  n->arith_intensity_curve_num_samples = arith_intensity_curve_num_samples;
  n->cache_line_bytes = cache_line_bytes;
  n->extract_workload = extract_workload;
  n->feature_vector_length = s_tir::group1::Feature::kCount +                                  //
                             s_tir::group2::Feature::SubFeature::kCount * tensors_per_store +  //
                             arith_intensity_curve_num_samples +                               //
                             s_tir::group4::Feature::kCount +                                  //
                             s_tir::group5::Feature::kCount;
  if (extract_workload) {
    n->feature_vector_length += s_tir::group6::Feature::kCount;
  }
  return FeatureExtractor(ffi::UnsafeInit{}, n);
}

TVM_FFI_STATIC_INIT_BLOCK() { PerStoreFeatureObj::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.meta_schedule.FeatureExtractorPerStoreFeature",
                        FeatureExtractor::PerStoreFeature);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
