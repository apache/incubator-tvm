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
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/transform.h>
#include <tvm/sym/iter_affine_map.h>
#include <tvm/target/target.h>
#include <tvm/tirx/type.h>

#include <array>
#include <stack>

#include "../schedule/utils.h"
#include "./memhammer_rewrite_rule.h"
#include "tvm/tirx/stmt.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

using support::NDIntSet;

// rewrite rules
static InverseMapping inverse_mapping;
static CoalescedAccess coalesced_access;
static CreateLocalStage create_local_stage;
static SharedToWmma shared_to_wmma;
static WmmaToGlobal wmma_to_global;
static WmmaToShared wmma_to_shared;
static MmaToGlobal mma_to_global;

/*!
 * \brief A class to perform auto padding.
 *
 * One simple way to perform auto padding is to fix each padding size for each dimension at the
 * same time, calculate the precise access index and the bank conflict,
 * and choose the one with minimal conflict. However, this algorithm has exponential complexity.
 * Suppose we have d dimensions and the padding size is 0-31, we need to calculate bank
 * conflict for 32^{d-1} times.
 * We propose a fast incremental algorithm that works for affine inputs, and it only calculate
 * bank conflict for 32*{d-1} times. To be specific, we first decide the optimal padding size for
 * dimension d-2, then for dimension d-3, ..., finally for dimension 0. It involves 2 steps.
 *
 * First, we analyze how a typical warp accesses the shared memory banks.
 * A typical warp means setting all irrelevant loop vars to 0, and only keeps the threads in a warp.
 * For each dimension, the access index is represented by
 * x_1 * scale_1 + ... + x_n * scale_n (x_i is loop var)
 * Note: The affine property guarantees that {x_i} must be independent,
 * otherwise the algorithm is wrong.
 * We will use this information to keep a list for each dimension called "iteration space" that
 * records the resulting index as x_i takes each possible value.
 *
 * For example, the index is [outer*2+ty, tx*4+vec], where tx is threadIdx.x, and ty is threadIdx.y.
 * tx is in [0, 16), and ty is in [0, 2).
 * We will first get a warp access [ty, tx*4] because outer and vec are irrelevant loop vars.
 * It's obvious that ty, tx*4 are both in the form of x_1 * scale_1 + ... + x_n * scale_n.
 * In this case, we will keep lists {{0, 1}, {0, 4, ..., 60}}
 *
 * Next, we choose a padding size that has minimal conflict from the last dimension to first one.
 * To calculate the conflict, we calculate the Cartesian product of the iteration space of all
 * dimensions not higher than this. Each single point of product space represents access index
 * of a particular thread, by which we can calculate the accessed memory bank. The conflict is
 * the highest access frequency among the banks.
 *
 */
class AutoPadder {
 public:
  /**
   * \brief Do padding to the given tensors in shard memory
   * \param tensors the given tensors
   * \return the list of new padded tensors
   */
  ffi::Array<TensorVar> PadSharedMemory(const ffi::Array<TensorVar>& tensors) {
    static const auto scope_shared = runtime::MemoryScope::Get("shared");
    static const auto scope_shared_dyn = runtime::MemoryScope::Get("shared.dyn");

    ffi::Array<TensorVar> result;

    for (const TensorVar& tensor : tensors) {
      runtime::MemoryScope scope = tensor.scope();
      if ((scope == scope_shared || scope == scope_shared_dyn)) {
        auto iter_spaces = iter_spaces_[tensor.get()];
        if (iter_spaces.empty()) {
          result.push_back(tensor);
          continue;
        }
        // The access index represented by points in the cartesian product of lower dimension
        // iteration spaces
        std::vector<std::vector<int>> low_dim_iter_space(iter_spaces.size(), std::vector<int>());

        int n = tensor.shape().size();
        int data_bits = tensor.dtype().bits();
        // Step 1. initialize `low_dim_iter_space` with the iteration space of the last dim
        for (int i = 0; i < static_cast<int>(iter_spaces.size()); i++) {
          auto last_dim_iter_space = iter_spaces[i][n - 1];
          low_dim_iter_space[i] = last_dim_iter_space;
        }
        PrimExpr stride = 1;
        ffi::Array<PrimExpr> reverse_strides;
        int pad_min = static_cast<int>(padding_min_.Get(tensor).value_or(1));
        // Step 2. For each dimension, select a padding that has minimal bank conflict
        for (int k = n - 2; k >= 0; k--) {  // dims
          int max_pad_size = static_cast<int>(std::min(
              max_pad_factor_ *
                  static_cast<double>((stride * tensor.shape()[k + 1]).as<IntImmObj>()->value),
              static_cast<double>(32 * 32 / data_bits)));
          int min_conflict = INT32_MAX;
          int min_conflict_pad = -1;
          for (int pad = 0; pad <= max_pad_size; pad += pad_min) {  // select padding
            int padded_stride = (((stride * tensor.shape()[k + 1]).as<IntImmObj>()->value + pad) %
                                 (32 * 32 / data_bits))
                                    .as<int>()
                                    .value();
            int conflict = 0;
            for (int i = 0; i < static_cast<int>(iter_spaces.size()); i++) {  // accesses
              auto iter_space = iter_spaces[i][k];
              int bank[32]{0};
              for (int v1 : iter_space) {
                for (int v2 : low_dim_iter_space[i]) {
                  int comb = (v1 * padded_stride + v2) * data_bits / 32 % 32;
                  bank[comb]++;
                }
              }
              for (int j = 0; j < 32; j++) {
                conflict = std::max(conflict, bank[j]);
              }
            }
            if (conflict < min_conflict) {
              min_conflict = conflict;
              min_conflict_pad = pad;
            }
          }
          // update low_dim_iter_space with
          for (int i = 0; i < static_cast<int>(iter_spaces.size()); i++) {  // accesses
            auto iter_space = iter_spaces[i][k];
            if (!iter_space.empty()) {
              int padded_stride =
                  (((stride * tensor.shape()[k + 1]).as<IntImmObj>()->value + min_conflict_pad) %
                   (32 * 32 / data_bits))
                      .as<int>()
                      .value();
              std::vector<int> loc;
              for (int v1 : iter_space) {
                for (int v2 : low_dim_iter_space[i]) {
                  loc.push_back(((v1 * padded_stride + v2) * data_bits) % (32 * 32 / data_bits));
                }
              }
              low_dim_iter_space[i] = loc;
            }
          }
          stride = stride * tensor.shape()[k + 1] + min_conflict_pad;
          reverse_strides.push_back(stride);
        }
        // Step 3. create the new padded tensor
        ffi::ObjectPtr<TensorTypeObj> b = ffi::make_object<TensorTypeObj>(*tensor.ty().get());
        ffi::Array<PrimExpr> strides;
        for (int i = static_cast<int>(reverse_strides.size()) - 1; i >= 0; i--) {
          strides.push_back(reverse_strides[i]);
        }
        strides.push_back(1);
        static const auto memory_axis = tvm::tirx::Axis::Get("m");
        ffi::Array<tvm::tirx::Iter> shard;
        for (size_t i = 0; i < b->shape.size(); ++i) {
          shard.push_back(tvm::tirx::Iter(b->shape[i], strides[i], memory_axis));
        }
        ffi::Map<tvm::tirx::Axis, PrimExpr> offset;
        if (b->layout.has_value())
          offset = b->layout.value().as_or_throw<tvm::tirx::TileLayout>()->offset;
        b->layout = tvm::tirx::TileLayout(std::move(shard), {}, std::move(offset));
        TensorVar new_tensor = TensorVar(tensor->name, TensorType(std::move(b)), tensor->loc);
        result.push_back(new_tensor);
        padded_tensor_map_.Set(tensor, new_tensor);
      } else {
        result.push_back(tensor);
      }
    }
    return result;
  }

  /**
   * \brief Replace all occurrence of the old tensor with the new tensor in the stmt
   * \param stmt the stmt to do replacement
   * \return the stmt after replacement
   */
  Stmt RewriteTensorAccess(const Stmt& stmt) {
    class Rewriter : public StmtExprMutator {
     public:
      using StmtExprMutator::Mutate;
      using StmtExprMutator::Mutate_;

      explicit Rewriter(const ffi::Map<TensorVar, TensorVar>& tensor_map) {
        for (const auto& [tensor, replacement] : tensor_map) VarRemapSet(tensor, replacement);
      }

     private:
      UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* _op, InplaceMode inplace_mode) final {
        TensorLoad load = StmtExprMutator::Mutate_(_op, inplace_mode)
                              .ValueOrUnchanged(ffi::GetRef<PrimExpr>(_op))
                              .as_or_throw<TensorLoad>();
        TensorVar tensor = load->source.as_or_throw<tvm::tirx::TensorVar>();
        if (auto replacement = VarRemapGet(tensor).as<TensorVar>()) {
          return TensorLoad(replacement.value(), load->indices, load->loc);
        }
        return load;
      }

      UnchangedOr<Stmt> Mutate_(const TensorStoreObj* _op, InplaceMode inplace_mode) final {
        TensorStore store = StmtExprMutator::Mutate_(_op, inplace_mode)
                                .ValueOrUnchanged(ffi::GetRef<Stmt>(_op))
                                .as_or_throw<TensorStore>();
        TensorStoreObj* op = store.CopyOnWrite();
        if (auto replacement = VarRemapGet(op->dest.as_or_throw<TensorVar>()).as<TensorVar>()) {
          op->dest = replacement.value();
        }
        return store;
      }

      UnchangedOr<Stmt> Mutate_(const SBlockObj* op, InplaceMode inplace_mode) final {
        // To reduce the number of blocks in block sref reuse map, we check whether the block is
        // really mutated (i.e., the old tensor appears in the block). If so, we return the block
        // after mutation. Otherwise we just return the original block.
        bool changed = false;
        // Step 1. Mutate the read region.
        ffi::Array<TensorRegion> reads;
        for (const TensorRegion& read : op->reads) {
          if (auto replacement =
                  VarRemapGet(read->source.as_or_throw<tvm::tirx::TensorVar>()).as<TensorVar>()) {
            changed = true;
            reads.push_back(TensorRegion(replacement.value(), read->region, TensorRegionType()));
          } else {
            reads.push_back(read);
          }
        }
        // Step 2. Mutate the write region.
        ffi::Array<TensorRegion> writes;
        for (const TensorRegion& write : op->writes) {
          if (auto replacement =
                  VarRemapGet(write->source.as_or_throw<tvm::tirx::TensorVar>()).as<TensorVar>()) {
            changed = true;
            writes.push_back(TensorRegion(replacement.value(), write->region, TensorRegionType()));
          } else {
            writes.push_back(write);
          }
        }
        // Step 4. Mutate `match_tensors`. If an old tensor appears as a source of
        // MatchTensorRegion, the storage scope of the target tensor also needs to be set.
        ffi::Array<MatchTensorRegion> match_tensors;
        for (const MatchTensorRegion& match_tensor : op->match_tensors) {
          if (auto replacement =
                  VarRemapGet(match_tensor->source->source.as_or_throw<tvm::tirx::TensorVar>())
                      .as<TensorVar>()) {
            changed = true;
            TensorVar new_tensor = replacement.value();
            match_tensors.push_back(MatchTensorRegion(
                match_tensor->tensor,
                TensorRegion(new_tensor, match_tensor->source->region, TensorRegionType())));
          } else {
            match_tensors.push_back(match_tensor);
          }
        }
        // Step 5. Recursively mutate the block.
        Stmt res =
            StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
        if (res.get() != op) {
          changed = true;
        }

        if (changed) {
          SBlock block = std::move(res).as_or_throw<SBlock>();
          SBlockObj* n = block.CopyOnWrite();
          n->reads = std::move(reads);
          n->writes = std::move(writes);
          n->match_tensors = std::move(match_tensors);
          return block;
        } else {
          return ffi::Unchanged();
        }
      }
    };
    auto rewriter = ffi::make_object<Rewriter>(padded_tensor_map_);
    return rewriter->Mutate(stmt).ValueOrUnchanged(stmt);
  }

  /**
   * \brief an equivalent of scale * loop_var with loop_var: {min=0, extent=extent}
   */
  struct Pattern {
    int extent;
    int scale;
  };

  /**
   * \brief Collect pattern from indices
   */
  class PatternCollector : public StmtExprVisitor {
   public:
    using StmtExprVisitor::Visit_;

   private:
    ffi::Optional<VisitInterrupt> Visit_(const VarObj* op) final {
      if (!success_) {
        return std::nullopt;
      }
      int extent =
          var_range_[ffi::GetRef<Var>(op)]->extent.as<IntImmObj>()->value.as<int>().value();
      if (extent > 1) {
        stack_.push({{extent, 1}});
      } else {
        stack_.push({});
      }
      return std::nullopt;
    }

    ffi::Optional<VisitInterrupt> Visit_(const AddObj* op) final {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
      if (!success_) {
        return std::nullopt;
      }
      std::vector<Pattern> merged_patterns;
      std::vector<Pattern> r = stack_.top();
      stack_.pop();
      std::vector<Pattern> l = stack_.top();
      stack_.pop();
      for (const Pattern& pattern : l) {
        merged_patterns.push_back(pattern);
      }
      for (const Pattern& pattern : r) {
        merged_patterns.push_back(pattern);
      }
      if (merged_patterns.empty()) {
        stack_.push({});
        return std::nullopt;
      }
      std::vector<Pattern> ret;
      ret.push_back(merged_patterns[0]);
      for (int i = 0; i < static_cast<int>(merged_patterns.size()); i++) {
        Pattern prev_pattern = ret.back();
        if (merged_patterns[i].extent * merged_patterns[i].scale == prev_pattern.scale) {
          ret.pop_back();
          ret.push_back(
              {prev_pattern.extent * merged_patterns[i].extent, merged_patterns[i].scale});
        }
      }
      stack_.push(ret);
      return std::nullopt;
    }

    ffi::Optional<VisitInterrupt> Visit_(const FloorDivObj* op) final {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
      if (!success_) {
        return std::nullopt;
      }
      std::vector<Pattern> inner = stack_.top();
      stack_.pop();
      int lower_factor = op->b.as<IntImmObj>()->value.as<int>().value();
      std::vector<Pattern> ret;
      for (const Pattern& pattern : inner) {
        if (pattern.scale >= lower_factor) {
          if (pattern.scale % lower_factor == 0) {
            ret.push_back({pattern.extent, pattern.scale / lower_factor});
          } else {
            success_ = false;
          }
        } else if (pattern.scale * pattern.extent > lower_factor) {
          if ((pattern.scale * pattern.extent) % lower_factor == 0) {
            ret.push_back({pattern.extent * pattern.scale / lower_factor, 1});
          } else {
            success_ = false;
          }
        }
      }
      stack_.push(ret);
      return std::nullopt;
    }

    ffi::Optional<VisitInterrupt> Visit_(const FloorModObj* op) final {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
      if (!success_) {
        return std::nullopt;
      }
      std::vector<Pattern> inner = stack_.top();
      stack_.pop();
      int extent = op->b.as<IntImmObj>()->value.as<int>().value();
      std::vector<Pattern> ret;
      for (const Pattern& pattern : inner) {
        if (pattern.scale < extent) {
          if (extent % pattern.scale == 0) {
            if (extent / pattern.scale < pattern.extent) {
              ret.push_back({extent / pattern.scale, pattern.scale});
            } else {
              ret.push_back({pattern.extent, pattern.scale});
            }
          } else {
            success_ = false;
          }
        }
      }
      stack_.push(ret);
      return std::nullopt;
    }

    ffi::Optional<VisitInterrupt> Visit_(const MulObj* op) final {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
      if (!success_) {
        return std::nullopt;
      }
      std::vector<Pattern> inner = stack_.top();
      stack_.pop();
      int scale = op->b.as<IntImmObj>()->value.as<int>().value();
      std::vector<Pattern> ret;
      for (const Pattern& pattern : inner) {
        ret.push_back({pattern.extent, pattern.scale * scale});
      }
      stack_.push(ret);
      return std::nullopt;
    }

   public:
    explicit PatternCollector(const ffi::Map<Var, Range>& var_range) : var_range_(var_range) {}

    /*!
     * \brief Collect the iteration space for given indices. The iteration space is the possible
     * values that an index can take (do not remove duplicate).
     * For example, the input is [ty, tx*4], where tx is in [0, 16), and ty is in [0, 2).
     * The output would be {{0, 1}, {0, 4, ..., 60}}
     * \param indices The indices to analyze
     * \param var_range The range of loop variables
     * \param data_bits The size of dtype in bits
     * \return The iteration space. The first array represents dimensions, and the second array
     * represents the iteration space of one dimension
     */
    static std::vector<std::vector<int>> CollectIterationSpace(
        const ffi::Array<PrimExpr>& indices, const ffi::Map<Var, Range>& var_range, int data_bits) {
      auto collector = ffi::make_object<PatternCollector>(var_range);
      std::vector<std::vector<int>> ret;
      for (int i = 0; i < static_cast<int>(indices.size()); i++) {
        collector->Visit(indices[i]);
        if (collector->success_ && collector->stack_.size() == 1) {
          auto patterns = collector->stack_.top();
          int extent_prod = 1;
          for (const Pattern& p : patterns) {
            extent_prod *= p.extent;
          }
          std::vector<int> iter_space;
          for (int thread_id = 0; thread_id < extent_prod; thread_id++) {
            int index = 0;
            int n = thread_id;
            for (int j = static_cast<int>(patterns.size()) - 1; j >= 0; j--) {
              int val = n % patterns[j].extent;
              index += val * patterns[j].scale;
              n /= patterns[j].extent;
            }
            iter_space.push_back(index);
          }

          ret.push_back(iter_space);
          collector->stack_.pop();
        } else {
          ret.push_back({});
        }
      }
      return ret;
    }

    std::stack<std::vector<Pattern>> stack_;
    const ffi::Map<Var, Range>& var_range_;
    bool success_ = true;
  };

  /*! A utility class for calling CollectIterationSpace to each tensor access*/
  class IterSpaceAnalyzer : public StmtExprVisitor {
   public:
    using StmtExprVisitor::Visit_;
    IterSpaceAnalyzer(const ffi::Map<Var, PrimExpr>& substitute_map, AutoPadder* self,
                      int data_bits, const ffi::Map<ffi::String, int64_t> warp_thread_extent)
        : substitute_map_(substitute_map),
          self(self),
          data_bits_(data_bits),
          warp_thread_extent_(warp_thread_extent) {}

   private:
    bool CheckVarContiguous(PrimExpr e, Var var, const ffi::Map<Var, PrimExpr>& subst_map) {
      auto f_substitute_zero = [var](const Var& v) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
        if (v.same_as(var)) return ffi::Any(IntImm::Int32(0));
        return ffi::Unchanged();
      };
      auto f_substitute_one = [var](const Var& v) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
        if (v.same_as(var)) return ffi::Any(IntImm::Int32(1));
        return ffi::Unchanged();
      };
      auto f_substitute = [&subst_map](const Var& v) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
        if (auto repl = subst_map.Get(v)) return ffi::Any(*std::move(repl));
        return ffi::Unchanged();
      };
      PrimExpr e1 = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(e, f_substitute_zero)
                        .as_or_throw<PrimExpr>();
      PrimExpr e2 = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(e, f_substitute_one)
                        .as_or_throw<PrimExpr>();
      sym::Analyzer analyzer;
      PrimExpr delta = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(e2 - e1, f_substitute)
                           .as_or_throw<PrimExpr>();
      return !analyzer->CanProve(delta != 1);
    }

    ffi::Optional<VisitInterrupt> Visit_(const ForObj* op) final {
      if (!tvm::s_tir::GetThreadBinding(op).has_value()) {
        substitute_map_.Set(op->var, op->begin);
      } else {
        int64_t extent =
            warp_thread_extent_.Get(tvm::s_tir::GetThreadBinding(op).value()).value_or(1);
        var_range_.Set(op->var, Range::FromMinExtent(op->begin, IntImm::Int64(extent)));
      }
      if (op->kind == ForKind::kVectorized) {
        vector_var = op->var;
        vector_length_ = Range::GetExtentFromBeginEnd(op->begin, op->end)
                             .as<IntImmObj>()
                             ->value.as<int>()
                             .value();
      }
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
      if (op->kind == ForKind::kVectorized) {
        vector_length_ = -1;
      }
      if (!tvm::s_tir::GetThreadBinding(op).has_value()) {
        substitute_map_.erase(op->var);
      }
      return std::nullopt;
    }
    /*!
     * \brief Take a typical warp and collect the iteration space for tensor store
     * For example, the access is A[outer*2+ty, tx*4+vec] = xxx, where tx is threadIdx.x, and ty is
     * threadIdx.y. tx is in [0, 16), and ty is in [0, 2).
     * The iteration space would be {{0, 1}, {0, 4, ..., 60}}.
     * \param op the tensor store
     */
    ffi::Optional<VisitInterrupt> Visit_(const TensorStoreObj* op) final {
      static const auto scope_shared = runtime::MemoryScope::Get("shared");
      static const auto scope_shared_dyn = runtime::MemoryScope::Get("shared.dyn");

      runtime::MemoryScope scope = op->dest.as_or_throw<TensorVar>().scope();
      if ((scope == scope_shared || scope == scope_shared_dyn)) {
        ffi::Array<PrimExpr> substitued_indices;
        sym::Analyzer analyzer;
        auto f_substitute = [this](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
          if (auto repl = substitute_map_.Get(var)) return ffi::Any(*std::move(repl));
          return ffi::Unchanged();
        };
        for (const PrimExpr& e : op->indices) {
          substitued_indices.push_back(
              analyzer->Simplify(ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(e, f_substitute)
                                     .as_or_throw<PrimExpr>()));
        }
        std::vector<std::vector<int>> iter_space =
            PatternCollector::CollectIterationSpace(substitued_indices, var_range_, data_bits_);
        if (!iter_space.empty()) {
          self->iter_spaces_[op->dest.as_or_throw<TensorVar>().get()].push_back(iter_space);
        }
        if (vector_length_ != -1 &&
            CheckVarContiguous(op->indices.back(), vector_var.value(), substitute_map_)) {
          int64_t m = self->padding_min_.Get(op->dest.as_or_throw<TensorVar>()).value_or(1);
          self->padding_min_.Set(op->dest.as_or_throw<TensorVar>(),
                                 std::max(static_cast<int64_t>(vector_length_), m));
        }
      }
      return StmtExprVisitor::Visit_(op);
    }
    /*!
     * \brief Take a typical warp and collect the iteration space for tensor load
     * For example, the access is xxx = A[outer*2+ty, tx*4+vec], where tx is threadIdx.x, and ty is
     * threadIdx.y. tx is in [0, 16), and ty is in [0, 2).
     * The iteration space would be {{0, 1}, {0, 4, ..., 60}}.
     * \param op the tensor load
     */
    ffi::Optional<VisitInterrupt> Visit_(const TensorLoadObj* op) final {
      static const auto scope_shared = runtime::MemoryScope::Get("shared");
      static const auto scope_shared_dyn = runtime::MemoryScope::Get("shared.dyn");

      TensorVar tensor = op->source.as_or_throw<tvm::tirx::TensorVar>();
      runtime::MemoryScope scope = tensor.scope();
      if ((scope == scope_shared || scope == scope_shared_dyn)) {
        ffi::Array<PrimExpr> substitued_indices;
        sym::Analyzer analyzer;
        auto f_substitute = [this](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
          if (auto repl = substitute_map_.Get(var)) return ffi::Any(*std::move(repl));
          return ffi::Unchanged();
        };
        for (const PrimExpr& e : op->indices) {
          substitued_indices.push_back(
              analyzer->Simplify(ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(e, f_substitute)
                                     .as_or_throw<PrimExpr>()));
        }
        std::vector<std::vector<int>> iter_space =
            PatternCollector::CollectIterationSpace(substitued_indices, var_range_, data_bits_);
        if (!iter_space.empty()) {
          self->iter_spaces_[tensor.get()].push_back(iter_space);
        }
        if (vector_length_ != -1 &&
            CheckVarContiguous(substitued_indices.back(), vector_var.value(), substitute_map_)) {
          int64_t m = self->padding_min_.Get(tensor).value_or(1);
          self->padding_min_.Set(tensor, std::max(static_cast<int64_t>(vector_length_), m));
        }
      }
      return StmtExprVisitor::Visit_(op);
    }

    /*!
     * \brief Take a typical warp and collect the iteration space for load_matrix_sync and
     * store_matrix_sync
     * For example, the access region is A[y*16+16, x*16+16], where y and x are not bound to
     * threadIdx. The iteration space would be {{0, 1, ..., 15}, {0, 1, ..., 15}}.
     * \param op the call node
     */
    ffi::Optional<VisitInterrupt> Visit_(const SBlockObj* op) final {
      static const auto scope_shared = runtime::MemoryScope::Get("shared");
      static const auto scope_shared_dyn = runtime::MemoryScope::Get("shared.dyn");

      if (const auto* eval = op->body->size() == 1 ? op->body->seq[0].as<EvaluateObj>() : nullptr) {
        if (const auto* call = eval->value.as<CallObj>()) {
          static const Op gpu_load_matrix_sync_op = Op::Get("tirx.gpu_load_matrix_sync");
          static const Op gpu_store_matrix_sync_op = Op::Get("tirx.gpu_store_matrix_sync");
          if (call->op.same_as(gpu_load_matrix_sync_op) ||
              call->op.same_as(gpu_store_matrix_sync_op)) {
            for (const MatchTensorRegion& r : op->match_tensors) {
              TensorVar src_tensor = r->source->source.as_or_throw<tvm::tirx::TensorVar>();
              runtime::MemoryScope scope = src_tensor.scope();
              if ((scope == scope_shared || scope == scope_shared_dyn)) {
                ffi::Array<Range> region = r->source->region;
                ffi::Array<PrimExpr> indices;
                for (int i = 0; i < static_cast<int>(region.size()); i++) {
                  PrimVar var("region" + std::to_string(i));
                  indices.push_back(region[i]->begin + static_cast<PrimExpr>(var));
                  var_range_.Set(var, Range::FromMinExtent(0, region[i]->extent));
                }
                ffi::Array<PrimExpr> substitued_indices;
                sym::Analyzer analyzer;
                auto f_substitute =
                    [this](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
                  if (auto repl = substitute_map_.Get(var)) return ffi::Any(*std::move(repl));
                  return ffi::Unchanged();
                };
                for (const PrimExpr& e : indices) {
                  substitued_indices.push_back(analyzer->Simplify(
                      ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(e, f_substitute)
                          .as_or_throw<PrimExpr>()));
                }
                std::vector<std::vector<int>> iter_space = PatternCollector::CollectIterationSpace(
                    substitued_indices, var_range_, data_bits_);
                if (!iter_space.empty()) {
                  self->iter_spaces_[src_tensor.get()].push_back(iter_space);
                }
              }
            }
          }
        }
      }
      return std::nullopt;
    }

    ffi::Map<Var, PrimExpr> substitute_map_;
    AutoPadder* self;
    int data_bits_;
    ffi::Map<ffi::String, int64_t> warp_thread_extent_;
    ffi::Map<Var, Range> var_range_;
    int vector_length_ = -1;
    ffi::Optional<Var> vector_var;
  };

  /*!
   * \brief Analyze the shared memory access
   * \param stmt The data copy
   * \param outer_loops The outer loops of the stmt
   * \param data_bits The length of dtype in bits
   * \param thread_extent The extents of all thread binding loops
   */
  void AnalyzeSharedMemoryAccess(const Stmt& stmt, const ffi::Array<For>& outer_loops,
                                 int data_bits,
                                 const ffi::Map<ffi::String, int64_t>& thread_extent) {
    ffi::Map<ffi::String, int64_t> warp_thread_extent;
    int64_t prod = 1;
    ffi::Array<ffi::String> thread_tags{"threadIdx.x", "threadIdx.y", "threadIdx.z"};
    for (int i = 0; i < 3; i++) {
      int64_t extent = thread_extent.Get(thread_tags[i]).value_or(1);
      if (prod * extent >= 32) {
        int64_t warp_part = 32 / prod;
        warp_thread_extent.Set(thread_tags[i], warp_part);
        prod *= warp_part;
        break;
      } else {
        warp_thread_extent.Set(thread_tags[i], extent);
        prod *= extent;
      }
    }
    ffi::Map<Var, PrimExpr> substitute_map;
    for (const For& loop : outer_loops) {
      substitute_map.Set(loop->var, loop->begin);
    }
    auto iter_space_analyzer =
        ffi::make_object<IterSpaceAnalyzer>(substitute_map, this, data_bits, warp_thread_extent);
    iter_space_analyzer->Visit(stmt);
  }

 private:
  /*! \brief A map from the old tensors to the new padded tensors */
  ffi::Map<TensorVar, TensorVar> padded_tensor_map_;
  /*! \brief A map from each tensor to the iteration spaces of the accesses*/
  std::unordered_map<const VarObj*, std::vector<std::vector<std::vector<int>>>> iter_spaces_;
  /*! \brief A map from each tensor to their minimal padding size */
  ffi::Map<TensorVar, int64_t> padding_min_;
  /*! \brief max padding size in relative to the original shape*/
  const double max_pad_factor_ = 0.25;

  friend class AutoCopyMutator;
};

class AutoCopyMutator : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  explicit AutoCopyMutator(ffi::Map<ffi::String, int64_t> thread_extent)
      : thread_extent_(thread_extent) {}
  /**
   * \brief Replace old tensors with padded tensors in the stmt
   * \param stmt The stmt to rewrite
   * \return The stmt after rewrite
   */
  Stmt RewritePaddingBody(const Stmt& stmt) { return padder.RewriteTensorAccess(stmt); }

 private:
  UnchangedOr<Stmt> Mutate_(const SBlockObj* op, InplaceMode inplace_mode) final {
    static const auto scope_wmma_accumulator = runtime::MemoryScope::Get("wmma.accumulator");

    SBlock block = StmtExprMutator::Mutate_(op, inplace_mode)
                       .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                       .as_or_throw<SBlock>();
    // only rewrite the block annotated with "auto_copy"
    if (!GetAnn<bool>(op, tvm::s_tir::attr::kAutoCopy.name).value_or(false)) {
      SBlockObj* n = block.CopyOnWrite();
      n->alloc_tensors = padder.PadSharedMemory(std::move(n->alloc_tensors));
      return block;
    }
    TVM_FFI_ICHECK_EQ(block->writes.size(), 1);
    TVM_FFI_ICHECK_GE(block->reads.size(), 1);

    TensorRegion target_read = block->reads[0];
    if (block->reads.size() > 1) {
      bool found = false;
      for (size_t i = 0; i < block->reads.size(); i++) {
        if (block->reads[i]->source.as_or_throw<tvm::tirx::TensorVar>().scope() ==
            scope_wmma_accumulator) {
          found = true;
          target_read = block->reads[i];
        }
      }
      TVM_FFI_ICHECK(found) << "Multiple tensor read";
    }

    int data_bits = target_read->source.as_or_throw<tvm::tirx::TensorVar>().dtype().bits();
    ConstraintSet constraints(this->thread_extent_,  //
                              this->outer_loops_,    //
                              target_read,           //
                              block->writes[0],      //
                              data_bits,             //
                              block->annotations);
    SBlockObj* n = block.CopyOnWrite();
    OutputSet outputs;
    TVM_FFI_ICHECK_EQ(n->body->size(), 1);
    Stmt rewritten = n->body->seq[0];
    for (RewriteRule* rule : rules) {
      rewritten = rule->Apply(rewritten, constraints, &outputs);
    }
    n->body = SeqStmt(rewritten, n->body->loc);
    for (const TensorVar& tensor : outputs.alloc_tensor) {
      n->alloc_tensors.push_back(tensor);
    }
    for (const auto& p : outputs.padding_min) {
      int64_t m = padder.padding_min_.Get(p.first).value_or(1);
      padder.padding_min_.Set(p.first, std::max(p.second, m));
    }
    padder.AnalyzeSharedMemoryAccess(block->body, outer_loops_, data_bits, thread_extent_);
    n->alloc_tensors = padder.PadSharedMemory(std::move(n->alloc_tensors));
    return block;
  }

  UnchangedOr<Stmt> Mutate_(const ForObj* op, InplaceMode inplace_mode) final {
    outer_loops_.push_back(ffi::GetRef<For>(op));
    Stmt stmt = StmtExprMutator::Mutate_(op, InplaceMode::kDisallow)
                    .ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    outer_loops_.pop_back();
    return stmt;
  }

  /*! \brief Thread extents collected. */
  ffi::Map<ffi::String, int64_t> thread_extent_;
  /*! \brief The outer loops during recursive visit */
  ffi::Array<For> outer_loops_;
  /*! \brief Calculating optimal padding size */
  AutoPadder padder;

  /*! \brief All rewrite rules. */
  const std::array<RewriteRule*, 7> rules = {&inverse_mapping,     //
                                             &coalesced_access,    //
                                             &create_local_stage,  //
                                             &shared_to_wmma,      //
                                             &wmma_to_global,      //
                                             &wmma_to_shared,      //
                                             &mma_to_global};
};

/*!
 * \brief Collect the extent for all thread binding loops.
 */
class ThreadExtentCollector : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;
  ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
    if (value.as<ExprObj>()) return std::nullopt;
    return StmtExprVisitor::Visit(value);
  }
  static ffi::Map<ffi::String, int64_t> CollectThreadExtent(const Stmt& stmt) {
    auto collector = ffi::make_object<ThreadExtentCollector>();
    collector->Visit(stmt);
    return collector->thread_extent_;
  }

 private:
  ffi::Optional<VisitInterrupt> Visit_(const SBlockObj* op) final {
    if (ffi::Optional<int64_t> warp_execution =
            GetAnn<int64_t>(op, tvm::s_tir::attr::kWarpExecution.name)) {
      if (warp_execution.value() != 0) {
        thread_extent_.Set("threadIdx.x", 32);
      }
    }
    return StmtExprVisitor::Visit_(op);
  }
  ffi::Optional<VisitInterrupt> Visit_(const ForObj* op) final {
    if (tvm::s_tir::GetThreadBinding(op).has_value()) {
      PrimExpr extent_expr = Range::GetExtentFromBeginEnd(op->begin, op->end);
      if (const auto* extent = extent_expr.as<IntImmObj>()) {
        thread_extent_.Set(tvm::s_tir::GetThreadBinding(op).value(),
                           static_cast<int64_t>(extent->value));
      }
    }
    return StmtExprVisitor::Visit_(op);
  }

  /*! \brief the map from thread tag to its extent */
  ffi::Map<ffi::String, int64_t> thread_extent_;
};

namespace transform {

Pass LowerAutoCopy() {
  auto pass_func = [](Function f, IRModule m, PassContext ctx) {
    if (!f->body.has_value()) return f;
    auto* n = f.CopyOnWrite();
    auto mutator = ffi::make_object<AutoCopyMutator>(
        ThreadExtentCollector::CollectThreadExtent(n->body.value()));
    n->body = mutator->Mutate(n->body.value(), InplaceMode::kAllow)
                  .ValueOrUnchanged(std::move(n->body).value());
    n->body = mutator->RewritePaddingBody(n->body.value());
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "s_tir.LowerAutoCopy");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.LowerAutoCopy", LowerAutoCopy);
}

}  // namespace transform
}  // namespace s_tir
}  // namespace tvm
