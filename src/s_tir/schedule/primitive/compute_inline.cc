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
#include <tvm/ir/prim/expr.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/op/memory.h>

#include "../utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::prim;
using namespace tvm::tirx;

static const char kErrBodyInline[] = R"(The body of the inlined block should be in form of
    'A[f(i, j, k, ...)] = g(i, j, k, ...)',
where the store indices mapping f on the left are bijective affine.)";

static const char kErrBodyReverseInline[] = R"(The body of the inlined block should be in form of
    `B[...] = g(i, j, k, A[f(i, j, k, ...)] ...)`,
where A is the only tensor the block consumes, whose indices are distinct atomic variables,
and there should be no variables other than the index variables), and f is a bijective affine
mapping and there should not be predicates in the inlined block. The iter domains of the inlined
block should be covered by the producer block.)";

class HasInitBlock : public ScheduleErrorContextObj {
 public:
  explicit HasInitBlock(IRModule mod, SBlock block) : mod_(mod), block_(block) {}

  ffi::String FastErrorString() const final {
    return "ScheduleError: The block has init statement";
  }

  ffi::String DetailRenderTemplate() const final {
    return "ScheduleError: The block has init statement: {0}";
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_}; }

  static void Check(const IRModule& mod, const SBlock& block) {
    if (block->init.has_value()) {
      throw MakeScheduleError<HasInitBlock>(mod, block);
    }
  }

 private:
  IRModule mod_;
  SBlock block_;
};

class NotSingleReadWriteTensor : public ScheduleErrorContextObj {
 public:
  explicit NotSingleReadWriteTensor(IRModule mod, bool is_read, SBlock block)
      : mod_(mod), is_read_(is_read), block_(std::move(block)) {}

  ffi::String FastErrorString() const final {
    return is_read_ ? "ScheduleError: The block is allowed to read only a single tensor region"
                    : "ScheduleError: The block is allowed to write only a single tensor region";
  }

  ffi::String DetailRenderTemplate() const final {
    if (is_read_) {
      int k = block_->reads.size();
      return "The block is only allowed to read a single tensor region, but it reads " +
             std::to_string(k) + " region(s): {0}";
    } else {
      int k = block_->writes.size();
      return "The block is only allowed to write a single tensor region, but it writes " +
             std::to_string(k) + " region(s): {0}";
    }
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_}; }

  IRModule mod_;
  bool is_read_;
  SBlock block_;

  static TensorVar GetSingleRead(const ScheduleState& self, const SBlock& block,
                                 const StmtSRef& scope_root_sref) {
    const std::unordered_map<TensorVar, ffi::Array<StmtSRef>, ffi::ObjectPtrHash,
                             ffi::ObjectPtrEqual>& tensor_writers =
        self->block_info.at(scope_root_sref).scope->tensor_writers;
    const VarObj* read_tensor = nullptr;
    for (const TensorRegion& read_region : block->reads) {
      const VarObj* tensor = read_region->source.as_or_throw<tvm::tirx::TensorVar>().get();
      if (tensor == read_tensor) {
        continue;
      }
      if (tensor_writers.count(ffi::GetRef<Var>(tensor).as_or_throw<TensorVar>()) > 0) {
        if (read_tensor != nullptr) {
          throw MakeScheduleError<NotSingleReadWriteTensor>(self->mod, true, block);
        }
        read_tensor = tensor;
      }
    }
    if (read_tensor == nullptr) {
      throw MakeScheduleError<NotSingleReadWriteTensor>(self->mod, true, block);
    }
    return ffi::GetRef<Var>(read_tensor).as_or_throw<TensorVar>();
  }

  static TensorVar GetSingleWrite(const ScheduleState& self, const SBlock& block) {
    if (block->writes.size() != 1) {
      throw MakeScheduleError<NotSingleReadWriteTensor>(self->mod, false, block);
    }
    return block->writes[0]->source.as_or_throw<tvm::tirx::TensorVar>();
  }
};

class BodyAnalysisError : public ScheduleErrorContextObj {
 public:
  explicit BodyAnalysisError(bool is_reverse, IRModule mod, SBlock block)
      : is_reverse_(is_reverse), mod_(mod), block_(std::move(block)) {}

  ffi::String FastErrorString() const final {
    return "ScheduleError: The block cannot be inlined because its body pattern does not meet the "
           "condition for inlining";
  }

  ffi::String DetailRenderTemplate() const final {
    return is_reverse_ ? kErrBodyReverseInline : kErrBodyInline;
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_}; }

  bool is_reverse_;
  IRModule mod_;
  SBlock block_;
};

class NonSingleProducerError : public ScheduleErrorContextObj {
 public:
  explicit NonSingleProducerError(IRModule mod, SBlock block)
      : mod_(mod), block_(std::move(block)) {}

  ffi::String FastErrorString() const final {
    return "ScheduleError: The consumer block to be inlined is required to have only a single "
           "producer block, and the producer block should be a complete block who has only a "
           "single consumer";
  }

  ffi::String DetailRenderTemplate() const final {
    return "The consumer block {0} to be inlined is required to have only a single "
           "producer block, and the producer block should be a complete block who has only a "
           "single consumer";
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_}; }

  IRModule mod_;
  SBlock block_;

  /*!
   * \brief Check if the block has a single producer.
   * \param self The schedule state
   * \param block_sref The sref of the block to be checked
   * \param scope_root_sref The sref of the scope root
   * \return The sref of the producer block if the block has a single producer
   * \throw ScheduleError if the block does not have a single producer
   */
  static StmtSRef Check(const ScheduleState& self, const StmtSRef& consumer_block_sref,
                        const StmtSRef& scope_root_sref) {
    const SBlockObj* scope_block = TVM_SREF_TO_SBLOCK(scope_root_sref);
    const SBlockObj* consumer_block = TVM_SREF_TO_SBLOCK(consumer_block_sref);
    TensorVar consumer_tensor = NotSingleReadWriteTensor::GetSingleRead(
        self, ffi::GetRef<SBlock>(consumer_block), scope_root_sref);
    class ProducerFinder : public StmtExprVisitor {
     public:
      using StmtExprVisitor::Visit_;

      ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
        if (value.as<ExprObj>()) return std::nullopt;
        return StmtExprVisitor::Visit(value);
      }

      static std::vector<SBlock> GetProducer(const ScheduleState& self,
                                             const StmtSRef& scope_root_sref,
                                             const TensorVar& tensor, const SBlock& scope_block) {
        auto finder = ffi::make_object<ProducerFinder>(self, scope_root_sref, tensor);
        finder->Visit(scope_block);
        return finder->producer_across_scope_.back();
      }

      explicit ProducerFinder(const ScheduleState& self, const StmtSRef& scope_root_sref,
                              const TensorVar& tensor)
          : self_(self), scope_root_sref_(scope_root_sref), tensor_(tensor) {
        producer_across_scope_.push_back({});
      }

     private:
      ffi::Optional<VisitInterrupt> Visit_(const SBlockObj* node) final {
        producer_across_scope_.push_back({});
        TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(node));
        // not a leaf block
        if (!producer_across_scope_.back().empty()) {
          auto producer_under_block = producer_across_scope_.back();
          producer_across_scope_.pop_back();
          producer_across_scope_.back().insert(producer_across_scope_.back().end(),
                                               producer_under_block.begin(),
                                               producer_under_block.end());
          return std::nullopt;
        }
        // leaf block
        producer_across_scope_.pop_back();
        for (const auto& write : node->writes) {
          if (write->source.as_or_throw<tvm::tirx::TensorVar>().same_as(tensor_)) {
            // Check if the producer block is a complete block
            StmtSRef producer_block_sref = self_->stmt2ref.at(node);
            if (!IsCompleteBlock(self_, producer_block_sref, scope_root_sref_)) {
              throw MakeScheduleError<NonSingleProducerError>(self_->mod,
                                                              ffi::GetRef<SBlock>(node));
            }
            producer_across_scope_.back().push_back(ffi::GetRef<SBlock>(node));
            break;
          }
        }
        return std::nullopt;
      }
      ScheduleState self_;
      StmtSRef scope_root_sref_;
      TensorVar tensor_;
      std::vector<std::vector<SBlock>> producer_across_scope_;
    };
    std::vector<SBlock> producer_across_scope = ProducerFinder::GetProducer(
        self, scope_root_sref, consumer_tensor, ffi::GetRef<SBlock>(scope_block));
    if (producer_across_scope.size() != 1) {
      throw MakeScheduleError<NonSingleProducerError>(self->mod,
                                                      ffi::GetRef<SBlock>(consumer_block));
    }
    return self->stmt2ref.at(producer_across_scope[0].get());
  }
};

class OpaqueAccessError : public ScheduleErrorContextObj {
 public:
  explicit OpaqueAccessError(IRModule mod, StmtSRef scope_root_sref)
      : mod_(mod), scope_root_(ffi::GetRef<SBlock>(TVM_SREF_TO_SBLOCK(scope_root_sref))) {
    const SBlockObj* scope_root = TVM_SREF_TO_SBLOCK(scope_root_sref);
    this->scope_root_ = ffi::GetRef<SBlock>(scope_root);
  }

  ffi::String FastErrorString() const final {
    return "ScheduleError: The tensor to be inlined has opaque access (e.g. `B.data`), or its "
           "subregion is matched into other blocks";
  }

  ffi::String DetailRenderTemplate() const final {
    return "The tensor to be inlined has opaque access (e.g. `B.data`), or its "
           "subregion is matched into other blocks: {0}";
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {scope_root_}; }

  IRModule mod_;
  SBlock scope_root_;
};

class ProducerHasNonTrivialPredicateError : public ScheduleErrorContextObj {
 public:
  explicit ProducerHasNonTrivialPredicateError(IRModule mod, SBlockRealize producer,
                                               PrimExpr new_predicate)
      : mod_(mod), producer_(producer), new_predicate_(new_predicate) {}

  ffi::String FastErrorString() const final {
    return "ScheduleError: The producer block has a non-trivial predicate.";
  }

  ffi::String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "ScheduleError: The producer block {0} has a non-trivial predicate "
       << producer_->predicate << " that cannot be implied by the synthesized predicate "
       << new_predicate_ << " of the new inlined block.";
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {producer_}; }

  IRModule mod_;
  SBlockRealize producer_;
  PrimExpr new_predicate_;
};

/*!
 * \brief The base class of the inliner, which handles:
 * 1) Substitute a subtree with the specific block being inlined
 * 2) Update the block signature to reflect the changes of read/write/allocated tensors
 * 3) Maintain a list of index variables and their substitution of the tensor being inlined
 */
class BaseInliner : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  explicit BaseInliner(const TensorVar& inlined_tensor, const SBlock& inlined_block,
                       const StmtSRef& scope_root_sref)
      : inlined_tensor_(inlined_tensor),
        inlined_store_(inlined_block->body->size() == 1
                           ? inlined_block->body->seq[0].as<TensorStoreObj>()
                           : nullptr),
        scope_root_sref_(scope_root_sref) {
    AddTensorsInBlockSignature(inlined_block.get());
  }

 protected:
  UnchangedOr<Expr> Mutate_(const TensorRegionObj* op, InplaceMode inplace_mode) final {
    if (!op->source.as<TensorVar>()) {
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }
    auto region = Mutate(op->region).as_or_throw<UnchangedOr<ffi::Array<Range>>>();
    if (region.UnchangedOrSameAs(op->region)) return ffi::Unchanged();
    TensorRegion node = ffi::GetRef<TensorRegion>(op);
    node.CopyOnWrite()->region = std::move(region).ValueUnchecked();
    return node;
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* op, InplaceMode inplace_mode) override {
    auto indices = Mutate(op->indices).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    TensorLoad node = ffi::GetRef<TensorLoad>(op);
    if (!indices.UnchangedOrSameAs(op->indices)) {
      node.CopyOnWrite()->indices = std::move(indices).ValueUnchecked();
    }
    return node;
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreObj* op, InplaceMode inplace_mode) override {
    auto value = Mutate(op->value);
    auto indices = Mutate(op->indices).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    TensorStore node = ffi::GetRef<TensorStore>(op);
    if (!value.UnchangedOrSameAs(op->value) || !indices.UnchangedOrSameAs(op->indices)) {
      auto* n = node.CopyOnWrite();
      n->value = std::move(value).ValueOrUnchanged(op->value);
      n->indices = std::move(indices).ValueOrUnchanged(op->indices);
    }
    return node;
  }

  UnchangedOr<Expr> Mutate_(const VarObj* var, InplaceMode inplace_mode) final {
    if (def_region_kind() == kTVMFFIDefRegionKindNone) {
      CheckOpaqueAccess(var);
    }
    return StmtExprMutator::Mutate_(var, inplace_mode);
  }

  UnchangedOr<Expr> Mutate_(const CallObj* op, InplaceMode inplace_mode) final {
    if (op->op.same_as(tirx::address_of_op())) {
      if (const auto* load = op->args[0].as<TensorLoadObj>()) {
        CheckOpaqueAccess(load->source.as_or_throw<TensorVar>().get());
        // An address operand must remain a TensorLoad, not the inlined value.
        auto operand = BaseInliner::Mutate_(load, inplace_mode);
        if (operand.UnchangedOrSameAs(ffi::GetRef<TensorLoad>(load))) return ffi::Unchanged();
        Call call = ffi::GetRef<Call>(op);
        call.CopyOnWrite()->args.Set(0, std::move(operand).ValueUnchecked());
        return call;
      }
    }
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Stmt> Mutate_(const ForObj* loop, InplaceMode inplace_mode) final {
    if (src_stmt.get() == loop) {
      loop = tgt_stmt.as<ForObj>();
      TVM_FFI_ICHECK(loop != nullptr);
    }
    return StmtExprMutator::Mutate_(loop, loop->unique() ? inplace_mode : InplaceMode::kDisallow)
        .ValueOrUnchanged(ffi::GetRef<Stmt>(loop));
  }

  UnchangedOr<Stmt> Mutate_(const SBlockObj* block, InplaceMode inplace_mode) {
    CheckMatchTensorRegion(block);
    AddTensorsInBlockSignature(block);
    SBlock src_block = ffi::GetRef<SBlock>(block);
    if (src_block.same_as(src_stmt)) {
      block = tgt_stmt.as<SBlockObj>();
      TVM_FFI_ICHECK(block != nullptr);
    }
    SBlock tgt_block =
        StmtExprMutator::Mutate_(block, block->unique() ? inplace_mode : InplaceMode::kDisallow)
            .ValueOrUnchanged(ffi::GetRef<Stmt>(block))
            .as_or_throw<SBlock>();
    bool is_scope_root = src_block.get() == scope_root_sref_->stmt;
    tgt_block = UpdateTensorsInBlockSignature(std::move(tgt_block), is_scope_root);
    block_reuse.Set(src_block, tgt_block);
    return tgt_block;
  }

 private:
  /*!
   * \brief Add the tensors in the block signature to the `tensor_var_map_`,
   * which is used for auto-completion of a block's read/write region
   * \param block The block whose signature to be added
   */
  void AddTensorsInBlockSignature(const SBlockObj* block) {
    for (const TensorRegion& tensor_region : block->reads) {
      const TensorVar& tensor = tensor_region->source.as_or_throw<tvm::tirx::TensorVar>();
      tensor_var_map_.Set(tensor, tensor);
    }
    for (const TensorRegion& tensor_region : block->writes) {
      const TensorVar& tensor = tensor_region->source.as_or_throw<tvm::tirx::TensorVar>();
      tensor_var_map_.Set(tensor, tensor);
    }
    for (const TensorVar& tensor : block->alloc_tensors) {
      tensor_var_map_.Set(tensor, tensor);
    }
  }

  /*!
   * \brief Update the following block signature:
   * 1) T.alloc_tensor, if the block is scope root
   * 2) T.reads, if the block is not scope root
   * 3) T.writes, if the block is not scope root
   * \param block The block to be updated
   * \param is_scope_root A flag indicating if a block is the scope root of the block to be inlined
   * \return The updated block
   */
  SBlock UpdateTensorsInBlockSignature(SBlock block, bool is_scope_root) {
    // Step 1. Update `BlockNode::alloc_tensors`
    ffi::Array<TensorVar> alloc_tensors;
    if (is_scope_root) {
      alloc_tensors.reserve(block->alloc_tensors.size());
      for (const TensorVar& alloc_tensor : block->alloc_tensors) {
        if (!alloc_tensor.same_as(inlined_tensor_)) {
          alloc_tensors.push_back(alloc_tensor);
        }
      }
    } else {
      alloc_tensors = std::move(block->alloc_tensors);
    }
    // Step 2. Update `BlockNode::reads` and `BlockNode::writes`
    ffi::Array<TensorRegion> reads = std::move(block->reads);
    ffi::Array<TensorRegion> writes = std::move(block->writes);
    auto f_access_inline_tensor = [this](const TensorRegion& access) {
      return access->source.as_or_throw<tvm::tirx::TensorVar>().same_as(this->inlined_tensor_);
    };
    if (!is_scope_root && (std::any_of(reads.begin(), reads.end(), f_access_inline_tensor) ||
                           std::any_of(writes.begin(), writes.end(), f_access_inline_tensor))) {
      ffi::Array<ffi::Array<TensorRegion>> inspected =
          GetSBlockReadWriteRegion(block, tensor_var_map_);
      reads = inspected[0];
      writes = inspected[1];
    }
    // Step 3. Assemble the result
    SBlockObj* n = block.CopyOnWrite();
    n->reads = std::move(reads);
    n->writes = std::move(writes);
    n->alloc_tensors = std::move(alloc_tensors);
    return block;
  }

  /*!
   * \brief Opaque access to the tensor to be inlined is disallowed.
   * This method checks if a tensor var belongs to the tensor
   * \param tensor_var The tensor var to be checked
   */
  void CheckOpaqueAccess(const VarObj* tensor_var) {
    if (inlined_tensor_.get() == tensor_var) {
      this->has_opaque_access = true;
    }
  }

  /*!
   * \brief The tensor to be inlined is not allowed to be region matched.
   * This method checks if a block has the disallowed behavior of tensor region match.
   * \param block The block to be checked
   */
  void CheckMatchTensorRegion(const SBlockObj* block) {
    for (const MatchTensorRegion& match_tensor_region : block->match_tensors) {
      const TensorVar& matched =
          match_tensor_region->source->source.as_or_throw<tvm::tirx::TensorVar>();
      if (matched.same_as(inlined_tensor_)) {
        this->has_opaque_access = true;
      }
    }
  }

 protected:
  /*! \brief The tensor to be inlined */
  TensorVar inlined_tensor_;
  /*! \brief The body of the block to be inlined */
  const TensorStoreObj* inlined_store_{nullptr};
  /*! \brief The scope root */
  StmtSRef scope_root_sref_;
  /*! \brief Maps a tensor's data field to itself */
  ffi::Map<Var, TensorVar> tensor_var_map_;
  /*! \brief The indices used for indexing the tensor to be inlined */
  std::vector<Var> idx_vars_;

 public:
  /*!
   * \brief The Stmt to be replaced when removing the leaf block
   * \note The pair (src_stmt, tgt_stmt) are produced by LeafBlockRemovalPlan to indicate a
   * transformation on top of the input AST. We take this approach to avoid changing the AST twice
   */
  Stmt src_stmt{ffi::UnsafeInit{}};
  /*! \brief The Stmt to be replaced to when removing the leaf block */
  Stmt tgt_stmt{ffi::UnsafeInit{}};
  /*! \brief The reuse mapping of block srefs */
  ffi::Map<SBlock, SBlock> block_reuse;
  /*! \brief Indicates if there is any opaque access of the inlined tensor */
  bool has_opaque_access{false};
};

/*!
 * \brief Helper to inline the producer block into its consumer(s)
 * The derived class implements the following functionalities:
 * 1) Substitute `TensorLoad` on the tensor to be inlined
 * to its value calculation in the producer block
 * 2) Analyze the producer block to determine the remapping of index variables
 */
class ComputeInliner : public BaseInliner {
 public:
  using BaseInliner::Mutate;
  using BaseInliner::Mutate_;

  explicit ComputeInliner(const TensorVar& inlined_tensor, const SBlock& producer_block,
                          const StmtSRef& scope_root_sref)
      : BaseInliner(inlined_tensor, producer_block, scope_root_sref) {}

  bool BodyPatternAllowInline(const SBlock& producer_block) {
    if (inlined_store_ == nullptr) {
      return false;
    }

    // Fast path on trivial case:
    // Check the store indices are same with the block iters;
    store_value_ = inlined_store_->value;
    size_t num_iters = producer_block->iter_vars.size();
    size_t tensor_ndim = inlined_store_->indices.size();
    if (num_iters == tensor_ndim) {
      std::vector<Var> idx_vars;
      idx_vars.reserve(num_iters);
      for (size_t i = 0; i < num_iters; ++i) {
        const IterVar& iter = producer_block->iter_vars[i];
        const PrimExpr& e = inlined_store_->indices[i];
        if (e.same_as(iter->var) || (analyzer_->CanProveEqual(e, 0) &&
                                     analyzer_->CanProveEqual(iter->dom.value()->begin, 0) &&
                                     analyzer_->CanProveEqual(iter->dom.value()->extent, 1))) {
          idx_vars.push_back(iter->var);
        } else {
          break;
        }
      }
      if (idx_vars.size() == num_iters) {
        // match success
        idx_vars_ = std::move(idx_vars);
        return true;
      }
    }

    // If the mapping for store indices is non-trivial
    // check bijective mapping from producer iter var to store indices
    ffi::Map<PrimVar, Range> producer_iter_doms;
    for (const auto& iter : producer_block->iter_vars) {
      producer_iter_doms.Set(iter->var, iter->dom.value());
    }
    sym::IterMapResult res = sym::DetectIterMap(
        /*indices=*/inlined_store_->indices,
        /*input_iters=*/producer_iter_doms,
        /*predicate=*/true,
        /*check_level=*/sym::IterMapLevel::Bijective,
        /*analyzer=*/analyzer_,
        /*simplify_trivial_iterators=*/false);
    if (!res->errors.empty()) {
      // Failure: indices of TensorStore are not bijective affine
      return false;
    }
    idx_vars_.clear();
    idx_vars_.reserve(tensor_ndim);
    for (size_t i = 0; i < tensor_ndim; ++i) {
      idx_vars_.emplace_back("ph_" + std::to_string(i), inlined_store_->indices[i].ty());
    }
    ffi::Array<PrimExpr> prim_idx_vars;
    prim_idx_vars.reserve(idx_vars_.size());
    for (const Var& var : idx_vars_) prim_idx_vars.push_back(var.as_or_throw<PrimExpr>());
    auto inverse_iter_map = sym::InverseAffineIterMap(res->indices, prim_idx_vars);
    for (const auto& iter : producer_block->iter_vars) {
      if (IsConstInt(iter->dom.value()->begin) &&
          analyzer_->CanProveEqual(iter->dom.value()->extent, 1)) {
        // fallback mapping for constant iters
        inverse_iter_map.Set(iter->var, iter->dom.value()->begin);
      }
    }
    auto f_substitute =
        [&inverse_iter_map](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
      if (auto repl = inverse_iter_map.Get(var)) return ffi::Any(*std::move(repl));
      return ffi::Unchanged();
    };
    store_value_ = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(store_value_, f_substitute)
                       .as_or_throw<PrimExpr>();
    return true;
  }

 private:
  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* _load, InplaceMode inplace_mode) final {
    TensorLoad load = BaseInliner::Mutate_(_load, inplace_mode)
                          .ValueOrUnchanged(ffi::GetRef<PrimExpr>(_load))
                          .as_or_throw<TensorLoad>();
    if (!load->source.as_or_throw<tvm::tirx::TensorVar>().same_as(inlined_tensor_)) {
      return load;
    }
    return ReplaceInlinedTensor(std::move(load));
  }

  PrimExpr ReplaceInlinedTensor(TensorLoad load) {
    TVM_FFI_ICHECK_EQ(load->indices.size(), idx_vars_.size());
    auto substituter = ffi::make_object<StmtExprMutator>();
    for (size_t i = 0; i < idx_vars_.size(); ++i) {
      substituter->VarRemapSet(idx_vars_[i], load->indices[i]);
    }
    return substituter->Mutate(store_value_).ValueOrUnchanged(store_value_);
  }

  /*! \brief The arithmetic analyzer */
  sym::Analyzer analyzer_;
  /*! \brief The store value for inlinement. If the producer
   store indices are trivial, it is wrt the producer block iter var,
   otherwise it is wrt to the placeholder vars of store indices. */
  PrimExpr store_value_{ffi::UnsafeInit{}};
};

/*!
 * \brief Helper to inline the consumer block into its producer
 * The derived class implements the following functionalities:
 * 1) Analyze the consumer block to determine the remapping of index variables
 * 2) Substitute `TensorStore` of the tensor to be inlined,
 * replacing it with direct writing to the tensor that consumer writes
 */
class ReverseComputeInliner : public BaseInliner {
 public:
  using BaseInliner::Mutate;
  using BaseInliner::Mutate_;

  class Substituter : public StmtExprMutator {
   public:
    using StmtExprMutator::Mutate;
    using StmtExprMutator::Mutate_;

    explicit Substituter(ReverseComputeInliner* self) : self_(self) {
      for (const IterVar& iter : self->consumer_block_->iter_vars) {
        auto replacement = self->VarRemapGet(iter->var);
        if (replacement != nullptr) VarRemapSet(iter->var, replacement);
      }
    }

   private:
    UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* _load, InplaceMode inplace_mode) final {
      TensorLoad load = StmtExprMutator::Mutate_(_load, inplace_mode)
                            .ValueOrUnchanged(ffi::GetRef<PrimExpr>(_load))
                            .as_or_throw<TensorLoad>();
      return load->source.as_or_throw<tvm::tirx::TensorVar>().same_as(self_->inlined_tensor_)
                 ? self_->producer_rhs_
                 : load;
    }

    ReverseComputeInliner* self_;
  };

  class RecursionResolver : public StmtExprMutator {
   public:
    using StmtExprMutator::Mutate;
    using StmtExprMutator::Mutate_;

    explicit RecursionResolver(ReverseComputeInliner* self) : self_(self) {
      for (const IterVar& iter : self->consumer_block_->iter_vars) {
        auto replacement = self->VarRemapGet(iter->var);
        if (replacement != nullptr) VarRemapSet(iter->var, replacement);
      }
    }

   private:
    UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* _load, InplaceMode inplace_mode) final {
      TensorLoad load = StmtExprMutator::Mutate_(_load, inplace_mode)
                            .ValueOrUnchanged(ffi::GetRef<PrimExpr>(_load))
                            .as_or_throw<TensorLoad>();
      if (!load->source.as_or_throw<TensorVar>().same_as(self_->inlined_tensor_)) return load;
      PrimExpr replacement = TensorLoad(self_->inlined_store_->dest.as_or_throw<TensorVar>(),
                                        self_->inlined_store_->indices);
      return StmtExprMutator::Mutate(ffi::AnyView(replacement), InplaceMode::kDisallow)
          .ValueOrUnchanged(std::move(replacement))
          .as_or_throw<PrimExpr>();
    }

    ReverseComputeInliner* self_;
  };

  explicit ReverseComputeInliner(const TensorVar& inlined_tensor, const SBlockObj* producer_block,
                                 const SBlockRealize& consumer_block_realize,
                                 const StmtSRef& scope_root_sref, const IRModule& mod)
      : BaseInliner(inlined_tensor, consumer_block_realize->block, scope_root_sref),
        producer_block_(producer_block),
        consumer_block_(consumer_block_realize->block.get()) {
    // Initialize the predicates to ensure consumer block iters are in-bound
    consumer_iter_in_bound_ = IntImm::Bool(true);
    for (const IterVar& iter : consumer_block_realize->block->iter_vars) {
      consumer_iter_in_bound_ =
          consumer_iter_in_bound_ &&
          (iter->var >= iter->dom.value()->begin &&
           static_cast<PrimExpr>(iter->var) < iter->dom.value()->begin + iter->dom.value()->extent);
    }
  }

  bool BodyPatternAllowInline(const SBlockRealize& consumer_block_realize) {
    const SBlock& consumer_block = consumer_block_realize->block;

    if (!IsOne(consumer_block_realize->predicate)) {
      // Failure: Predicate is the consumer block is not supported
      return false;
    }
    if (inlined_store_ == nullptr) {
      // Failure: block body is not TensorStore
      return false;
    }
    std::vector<const TensorLoadObj*> loads = ExtractTensorLoad(inlined_tensor_, inlined_store_);
    if (loads.size() == 0) {
      // Failure: no TensorLoad from the `inlined_tensor_`
      return false;
    }

    // Collect block iter domains and update the substition map
    ffi::Map<PrimVar, Range> consumer_iter_doms;
    for (const auto& iter_var : consumer_block->iter_vars) {
      consumer_iter_doms.Set(iter_var->var, iter_var->dom.value());
      // Set default mapping for unit iters
      if (IsConstInt(iter_var->dom.value()->extent, 1) &&
          IsConstInt(iter_var->dom.value()->begin)) {
        VarRemapSet(iter_var->var, iter_var->dom.value()->begin);
      }
    }

    for (const TensorLoadObj* load : loads) {
      if (!UpdateAndCheckIndexExprs(load->indices)) {
        return false;
      }
    }

    sym::IterMapResult res = sym::DetectIterMap(
        /*indices=*/tensor_load_indices_,
        /*input_iters=*/consumer_iter_doms,
        /*predicate=*/true,
        /*check_level=*/sym::IterMapLevel::NoCheck,
        /*analyzer=*/analyzer_,
        /*simplify_trivial_iterators=*/false);
    tensor_load_iter_map_ = res->indices;
    if (tensor_load_iter_map_.empty()) {
      // Failure: indices of TensorLoad are not bijective affine
      return false;
    }

    if (producer_block_->body->size() != 1) {
      return false;
    }
    const TensorStoreObj* producer_store = nullptr;
    const Stmt& producer_body = producer_block_->body->seq[0];
    if (const auto* producer_if = producer_body.as<IfObj>()) {
      if (producer_if->else_case.has_value()) {
        return false;
      }
      producer_store = producer_if->then_case->size() == 1
                           ? producer_if->then_case->seq[0].as<TensorStoreObj>()
                           : nullptr;
    } else {
      producer_store = producer_body.as<TensorStoreObj>();
      if (producer_block_->annotations.count(tvm::s_tir::attr::kAutoCopy.name) != 0) {
        const ForObj* producer_inner_loop = producer_body.as<ForObj>();
        if (producer_inner_loop == nullptr) return false;
        for (;;) {
          if (producer_inner_loop->body->size() != 1) return false;
          const Stmt& child = producer_inner_loop->body->seq[0];
          if (const auto* inner_loop = child.as<ForObj>()) {
            producer_inner_loop = inner_loop;
          } else {
            producer_store = child.as<TensorStoreObj>();
            break;
          }
        }
      }
    }
    if (producer_store == nullptr) {
      // Failure: producer block body is not TensorStore
      return false;
    }
    CreateInverseMapping(producer_store->indices);
    if (!CheckConsumerCovered()) {
      // Failure: consumer block iter domains are not covered by the producer block
      return false;
    }

    return true;
  }

 private:
  /*! \brief Generate the predicate after inlining based on the consumer predicate */
  SBlockRealize BuildInlinedConsumerPredicate(SBlockRealize producer_block_realize) {
    // Bind the producer block iter domains for simplification
    ffi::Map<Var, PrimExpr> subst_map;
    SBlock producer_block = producer_block_realize->block;
    for (int i = 0, n = producer_block->iter_vars.size(); i < n; ++i) {
      const IterVar& iter = producer_block->iter_vars[i];
      const PrimExpr& binding = producer_block_realize->iter_values[i];
      subst_map.Set(iter->var, binding);
      {
        auto context = analyzer_->GetConstraintContext();
        context->Append(sym::RangeConstraint(iter->var, iter->dom.value()));
        analyzer_->SyncConstraintContext(context);
      }
    }
    if (producer_block->annotations.count(tvm::s_tir::attr::kAutoCopy.name) != 0) {
      auto bind = [&](const ForObj* loop) {
        {
          auto context = analyzer_->GetConstraintContext();
          context->Append(sym::RangeConstraint(
              loop->var, Range::FromMinExtent(
                             IntImm(Range::GetExtentFromBeginEnd(loop->begin, loop->end).ty(), 0),
                             Range::GetExtentFromBeginEnd(loop->begin, loop->end))));
          analyzer_->SyncConstraintContext(context);
        }
      };
      TVM_FFI_ICHECK_EQ(producer_block->body->size(), 1);
      const ForObj* producer_inner_loop = producer_block->body->seq[0].as<ForObj>();
      while (producer_inner_loop != nullptr) {
        bind(producer_inner_loop);
        TVM_FFI_ICHECK_EQ(producer_inner_loop->body->size(), 1);
        producer_inner_loop = producer_inner_loop->body->seq[0].as<ForObj>();
      }
    }
    // Substitute the consumer block iters with the corresponding iters in the producer blocks
    PrimExpr predicate = ffi::make_object<Substituter>(this)
                             ->Mutate(consumer_iter_in_bound_)
                             .ValueOrUnchanged(consumer_iter_in_bound_);
    // Simplify the predicate using the producer block iter domains
    predicate = analyzer_->Simplify(predicate);
    if (IsOne(predicate)) {
      return producer_block_realize;
    }
    if (const auto* if_ = producer_block->body->size() == 1
                              ? producer_block->body->seq[0].as<IfObj>()
                              : nullptr) {
      if (!if_->else_case.has_value()) {
        PrimExpr if_predicate = analyzer_->Simplify(if_->condition);
        if (!ffi::StructuralEqual()(predicate, if_predicate)) {
          predicate = analyzer_->Simplify(predicate && if_->condition);
          producer_block.CopyOnWrite()->body = if_->then_case;
        }
      }
    }
    auto f_substitute = [&subst_map](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
      if (auto repl = subst_map.Get(var)) return ffi::Any(*std::move(repl));
      return ffi::Unchanged();
    };
    PrimExpr outer_predicate =
        ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(predicate, f_substitute)
            .as_or_throw<PrimExpr>();
    auto n = producer_block_realize.CopyOnWrite();
    n->block = producer_block;
    n->predicate = analyzer_->Simplify(outer_predicate);
    return ffi::GetRef<SBlockRealize>(n);
  }

  UnchangedOr<Stmt> Mutate_(const SBlockRealizeObj* op, InplaceMode inplace_mode) final {
    SBlock src_block = op->block;
    SBlockRealize tgt_block_realize = StmtExprMutator::Mutate_(op, inplace_mode)
                                          .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                                          .as_or_throw<SBlockRealize>();
    if (src_block.get() == producer_block_) {
      tgt_block_realize = BuildInlinedConsumerPredicate(tgt_block_realize);
      block_reuse.Set(src_block, tgt_block_realize->block);
    }
    return tgt_block_realize;
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreObj* _store, InplaceMode inplace_mode) final {
    TensorStore store = BaseInliner::Mutate_(_store, inplace_mode)
                            .ValueOrUnchanged(ffi::GetRef<Stmt>(_store))
                            .as_or_throw<TensorStore>();
    if (!store->dest.as_or_throw<TensorVar>().same_as(inlined_tensor_)) {
      return store;
    }
    return ReplaceInlinedTensor(std::move(store));
  }

  /*!
   * \brief Check the consumer block iter domains are covered by the producer block iter domains
   * \return Whether the consumer block iter domains are covered
   */
  bool CheckConsumerCovered() {
    ffi::Map<Var, sym::IntSet> producer_iter_doms;
    for (const IterVar& iter_var : producer_block_->iter_vars) {
      producer_iter_doms.Set(iter_var->var, sym::IntSet::FromRange(iter_var->dom.value()));
    }
    // For each block iter in the consumer block, find the corresponding expression in the producer
    for (const IterVar& iter : consumer_block_->iter_vars) {
      if (auto producer_iter = VarRemapGet(iter->var).as<PrimExpr>()) {
        sym::IntSet producer_iter_range = sym::EvalSet(producer_iter.value(), producer_iter_doms);
        if (analyzer_->CanProve(producer_iter_range.min() > iter->dom.value()->begin) ||
            analyzer_->CanProve(producer_iter_range.max() <
                                iter->dom.value()->begin + iter->dom.value()->extent - 1)) {
          return false;
        }
      } else {
        return false;
      }
    }
    return true;
  }

  /*!
   * \brief Apply the inverse of `tensor_load_iter_map_` to producer indices. Seed the inherited
   * remapping environment with the result. It will be later used to transform the TensorStore
   * indices of the producer.
   * \param producer_indices The TensorStore indices of the producer.
   */
  void CreateInverseMapping(const ffi::Array<PrimExpr> producer_indices) {
    auto inverse_iter_map = sym::InverseAffineIterMap(tensor_load_iter_map_, producer_indices);
    for (const auto& pair : inverse_iter_map) {
      VarRemapSet(pair.first, pair.second);
    }
  }

  Stmt ReplaceInlinedTensor(TensorStore producer) {
    // "producer->value" may contain the tensor that is inlined in cases of reduction,
    // so we need to resolve the recursion first
    producer_rhs_ = ffi::make_object<RecursionResolver>(this)
                        ->Mutate(producer->value)
                        .ValueOrUnchanged(producer->value);
    return ffi::make_object<Substituter>(this)
        ->Mutate(ffi::GetRef<TensorStore>(inlined_store_))
        .ValueOrUnchanged(ffi::GetRef<TensorStore>(inlined_store_));
  }

  /*!
   * \brief Extracts expressions that loads a specific tensor
   * \param tensor The tensor to be loaded from
   * \param from The TensorStore statement to be extracted from
   * \return A list of `TensorLoad` expressions
   */
  static std::vector<const TensorLoadObj*> ExtractTensorLoad(const TensorVar& tensor,
                                                             const TensorStoreObj* from) {
    struct Extractor : public StmtExprVisitor {
      using StmtExprVisitor::Visit_;

      ffi::Optional<VisitInterrupt> Visit_(const TensorLoadObj* load) final {
        if (load->source.as_or_throw<tvm::tirx::TensorVar>().get() == tensor) {
          result.push_back(load);
        }
        return StmtExprVisitor::Visit_(load);
      }
      const VarObj* tensor;
      std::vector<const TensorLoadObj*> result;
    };
    auto extractor = ffi::make_object<Extractor>();
    extractor->tensor = tensor.get();
    for (const PrimExpr& expr : from->indices) {
      extractor->Visit(expr);
    }
    extractor->Visit(from->value);
    return std::move(extractor->result);
  }

  /*!
   * \brief Update `tensor_load_indices_` with the given indices. If `tensor_load_indices_` is
   *        already non-empty, check it is consistent with the given indices.
   * \param indices The indices
   * \param expected_ndim The expected ndim of the access
   * \return A boolean flag indicating if the check is successful
   */
  bool UpdateAndCheckIndexExprs(const ffi::Array<PrimExpr>& indices) {
    if (tensor_load_indices_.empty()) {
      tensor_load_indices_ = indices;
    } else if (!std::equal(tensor_load_indices_.begin(), tensor_load_indices_.end(),
                           indices.begin(), indices.end(), prim::ExprDeepEqual())) {
      // Failure: indices are not consistent in different TensorLoads
      return false;
    }
    return true;
  }

  /*! \brief The RHS value of the producer's TensorStore statement */
  PrimExpr producer_rhs_{ffi::UnsafeInit{}};
  /*! \brief The indices of the consumer's TensorLoad */
  ffi::Array<PrimExpr> tensor_load_indices_;
  /*! \brief The IterMap representing the indices of the consumer's TensorLoad */
  ffi::Array<sym::IterSumExpr> tensor_load_iter_map_;
  /*! \brief The producer block */
  const SBlockObj* producer_block_{nullptr};
  /* \brief The consumer block */
  const SBlockObj* consumer_block_{nullptr};
  /*! \brief The predicate to ensure the consumer block iters are in-bound. It will be inserted
   * as the predicate of the producer block after inlining.
   */
  PrimExpr consumer_iter_in_bound_{ffi::UnsafeInit{}};
  /*! \brief The arithmetic analyzer */
  sym::Analyzer analyzer_;
};

void ComputeInlineImpl(ScheduleState self, const StmtSRef& producer_block_sref,
                       bool check_only = false) {
  const SBlockObj* _producer_block = TVM_SREF_TO_SBLOCK(producer_block_sref);
  SBlock producer_block = ffi::GetRef<SBlock>(_producer_block);
  HasInitBlock::Check(self->mod, producer_block);
  TensorVar inlined_tensor = NotSingleReadWriteTensor::GetSingleWrite(self, producer_block);
  // Step 1. Get the scope block
  StmtSRef scope_root_sref = GetScopeRoot(self, producer_block_sref,
                                          /*require_stage_pipeline=*/true);
  // Step 2. Check completeness
  CheckNotOutputBlock(self, producer_block_sref, scope_root_sref);
  CheckCompleteBlock(self, producer_block_sref, scope_root_sref);
  // Step 3. Analyze the block body
  auto inliner = ffi::make_object<ComputeInliner>(inlined_tensor, producer_block, scope_root_sref);
  if (!inliner->BodyPatternAllowInline(producer_block)) {
    throw MakeScheduleError<BodyAnalysisError>(false, self->mod, producer_block);
  }
  // Step 4. Create a plan that removes the leaf block to be inlined
  LeafBlockRemovalPlan(self, producer_block_sref, &inliner->src_stmt, &inliner->tgt_stmt);
  // Step 5. Create an AST where the leaf `producer_block_sref` points to is removed,
  // and update other blocks who read from the removed block
  Stmt tgt_stmt = inliner->Mutate(ffi::GetRef<Stmt>(scope_root_sref->stmt))
                      .ValueOrUnchanged(ffi::GetRef<Stmt>(scope_root_sref->stmt));
  if (inliner->has_opaque_access) {
    throw MakeScheduleError<OpaqueAccessError>(self->mod, scope_root_sref);
  }
  // Step 6. Do the real mutation on the AST and the sref tree in the schedule state
  if (check_only) {
    return;
  }
  self->Replace(scope_root_sref, tgt_stmt, inliner->block_reuse);
}

void ComputeInline(ScheduleState self, const StmtSRef& producer_block_sref) {
  ComputeInlineImpl(self, producer_block_sref);
}

bool CanComputeInline(const ScheduleState& self, const StmtSRef& producer_block_sref) {
  try {
    ComputeInlineImpl(self, producer_block_sref, true);
  } catch (const tvm::ffi::Error& e) {
    return false;
  }
  return true;
}

void ReverseComputeInlineImpl(ScheduleState self, const StmtSRef& consumer_block_sref,
                              bool check_only = false) {
  const SBlockObj* _consumer_block = TVM_SREF_TO_SBLOCK(consumer_block_sref);
  SBlock consumer_block = ffi::GetRef<SBlock>(_consumer_block);
  SBlockRealize consumer_block_realize = GetSBlockRealize(self, consumer_block_sref);
  HasInitBlock::Check(self->mod, consumer_block);
  // Step 1. Get the scope block
  StmtSRef scope_root_sref = GetScopeRoot(self, consumer_block_sref,  //
                                          /*require_stage_pipeline=*/true);
  TensorVar inlined_tensor =
      NotSingleReadWriteTensor::GetSingleRead(self, consumer_block, scope_root_sref);
  // Step 2. Check completeness
  CheckCompleteBlock(self, consumer_block_sref, scope_root_sref);
  // Step 3. Check if the consumer has a single complete producer, and the producer is not an output
  // block
  StmtSRef producer_block_sref =
      NonSingleProducerError::Check(self, consumer_block_sref, scope_root_sref);
  CheckNotOutputBlock(self, producer_block_sref, scope_root_sref);
  // Step 4. Analyze the block body
  auto inliner = ffi::make_object<ReverseComputeInliner>(
      inlined_tensor, producer_block_sref->StmtAs<SBlockObj>(), consumer_block_realize,
      scope_root_sref, self->mod);
  if (!inliner->BodyPatternAllowInline(consumer_block_realize)) {
    throw MakeScheduleError<BodyAnalysisError>(true, self->mod, consumer_block);
  }
  // Step 5. Create a plan that removes the leaf block to be inlined
  LeafBlockRemovalPlan(self, consumer_block_sref, &inliner->src_stmt, &inliner->tgt_stmt);
  // Step 6. Create an AST where the leaf `consumer_block_sref` points to is removed,
  // and update other blocks who read from the removed block
  Stmt tgt_stmt = inliner->Mutate(ffi::GetRef<Stmt>(scope_root_sref->stmt))
                      .ValueOrUnchanged(ffi::GetRef<Stmt>(scope_root_sref->stmt));
  if (inliner->has_opaque_access) {
    throw MakeScheduleError<OpaqueAccessError>(self->mod, scope_root_sref);
  }
  // Step 7. Do the real mutation on the AST and the sref tree in the schedule state
  if (check_only) {
    return;
  }
  self->Replace(scope_root_sref, tgt_stmt, inliner->block_reuse);
  // Step 8. Update the cached flags
  sym::Analyzer analyzer;
  SBlockInfo& block_info = self->block_info[producer_block_sref];
  block_info.affine_binding = IsAffineBinding(
      /*realize=*/GetSBlockRealize(self, producer_block_sref),
      /*loop_var_ranges=*/
      LoopDomainOfSRefTreePath(ffi::GetRef<StmtSRef>(producer_block_sref->parent)),
      /*analyzer=*/analyzer.get());
}

bool CanReverseComputeInline(const ScheduleState& self, const StmtSRef& block_sref) {
  try {
    ReverseComputeInlineImpl(self, block_sref, true);
  } catch (const tvm::ffi::Error& e) {
    return false;
  }
  return true;
}

void ReverseComputeInline(ScheduleState self, const StmtSRef& consumer_block_sref) {
  ReverseComputeInlineImpl(self, consumer_block_sref);
}

/*!
 * \brief Helper to fuse epilogue block into reduction block
 * Uses generalized approach to handle any epilogue expression without pattern matching
 */
class ReductionEpilogueFuser : public BaseInliner {
 public:
  using BaseInliner::Mutate;
  using BaseInliner::Mutate_;

  explicit ReductionEpilogueFuser(const TensorVar& reduction_tensor,
                                  const SBlockObj* reduction_block,
                                  const SBlockRealize& epilogue_block_realize,
                                  const StmtSRef& scope_root_sref)
      : BaseInliner(reduction_tensor, epilogue_block_realize->block, scope_root_sref),
        reduction_block_(reduction_block),
        epilogue_block_(epilogue_block_realize->block.get()) {
    // Disable opaque access check for epilogue fusion
    // Epilogue blocks can read multiple tensors (temp + bias), which is allowed
    has_opaque_access = false;
  }

  // Override CheckOpaqueAccess to allow multiple tensor reads
  void CheckOpaqueAccess(const VarObj* tensor_var) {
    // For epilogue fusion, we allow multiple tensor reads (temp + bias)
    // So we don't check for opaque access
    // BaseInliner::CheckOpaqueAccess(tensor_var);  // Don't call base class
  }

  bool BodyPatternAllowFusion(const SBlockRealize& epilogue_block_realize);

  // Step 2: Create single fused reduction block
  SBlock CreateFusedReductionBlock(const SBlockObj* reduction_block,
                                   const SBlockRealizeObj* reduction_realize);

 private:
  bool IsReductionBlock(const SBlockObj* block);
  void ExtractEpilogueInfo();
  // Helper function to extract TensorLoad nodes from TensorStore
  static std::vector<const TensorLoadObj*> ExtractTensorLoad(const TensorVar& tensor,
                                                             const TensorStoreObj* from) {
    struct Extractor : public StmtExprVisitor {
      using StmtExprVisitor::Visit_;

      ffi::Optional<VisitInterrupt> Visit_(const TensorLoadObj* load) final {
        if (load->source.as_or_throw<tvm::tirx::TensorVar>().same_as(tensor)) {
          result.push_back(load);
        }
        // Continue visiting child nodes (indices)
        return StmtExprVisitor::Visit_(load);
      }
      TensorVar tensor{ffi::UnsafeInit{}};
      std::vector<const TensorLoadObj*> result;
    };
    auto extractor = ffi::make_object<Extractor>();
    extractor->tensor = tensor;
    // Visit indices first (though they typically don't contain TensorLoad)
    for (const PrimExpr& expr : from->indices) {
      extractor->Visit(expr);
    }
    // Visit the value expression (e.g., max(temp + C, 0) for ReLU)
    extractor->Visit(from->value);
    return std::move(extractor->result);
  }

  const SBlockObj* reduction_block_;
  const SBlockObj* epilogue_block_;
  // Generalized approach: store the entire epilogue expression
  PrimExpr epilogue_expression_{
      ffi::UnsafeInit{}};  // The entire epilogue expression (e.g., temp + C, max(temp + C, 0))
  const TensorLoadObj* reduction_tensor_load_{
      nullptr};  // The reduction tensor load in epilogue expression
  TensorVar epilogue_output_tensor_{ffi::UnsafeInit{}};  // Output tensor D
  ffi::Array<PrimExpr> epilogue_output_indices_;         // Indices of D[vi, vj]
  ffi::Optional<TensorRegion> epilogue_output_region_;   // Write region of D
  ffi::Optional<TensorVar> epilogue_addend_tensor_;      // Additional tensor (e.g., bias tensor C)
  ffi::Optional<TensorRegion> epilogue_addend_region_;   // Read region of additional tensor
};

bool ReductionEpilogueFuser::BodyPatternAllowFusion(const SBlockRealize& epilogue_block_realize) {
  // 1. Validate predicate
  if (!IsOne(epilogue_block_realize->predicate)) {
    // Failure: Predicate in epilogue block is not supported
    return false;
  }

  // 2. Check if epilogue body is TensorStore
  if (inlined_store_ == nullptr) {
    // Failure: epilogue block body is not TensorStore
    return false;
  }

  // 3. Check if epilogue reads from reduction tensor
  std::vector<const TensorLoadObj*> loads = ExtractTensorLoad(inlined_tensor_, inlined_store_);
  if (loads.size() == 0) {
    // Failure: no TensorLoad from the reduction tensor
    return false;
  }

  // 4. Generalized approach: store the entire epilogue expression
  // Verify reduction tensor appears exactly once (required for fusion correctness)
  if (loads.size() != 1) {
    // Failure: The reduction result (temp) must be used exactly once in the
    // epilogue expression for fusion.
    return false;
  }

  // Store the epilogue expression and reduction tensor load
  epilogue_expression_ = inlined_store_->value;
  reduction_tensor_load_ = loads[0];

  // 5. Reject epilogues that scale the reduction result with non-additive ops
  // For example, (reduce_out * 2.0) + C[i] is not a valid bias-style epilogue.
  // We only allow the reduction result to be combined via Add/Min/Max shells.
  class ScalingDetector : public StmtExprVisitor {
   public:
    using StmtExprVisitor::Visit_;

    explicit ScalingDetector(const TensorVar& tensor)
        : finder_(ffi::make_object<TargetFinder>(tensor)) {}

    bool HasScaling(const PrimExpr& expr) {
      has_scaling_ = false;
      Visit(expr);
      return has_scaling_;
    }

   private:
    class TargetFinder : public StmtExprVisitor {
     public:
      using StmtExprVisitor::Visit_;

      explicit TargetFinder(const TensorVar& tensor) : tensor_(tensor) {}

      bool Find(const PrimExpr& e) {
        found_ = false;
        Visit(e);
        return found_;
      }

     private:
      ffi::Optional<VisitInterrupt> Visit_(const TensorLoadObj* op) final {
        if (op->source.as_or_throw<tvm::tirx::TensorVar>().same_as(tensor_)) {
          found_ = true;
          return std::nullopt;
        }
        return StmtExprVisitor::Visit_(op);
      }

      TensorVar tensor_;
      bool found_{false};
    };

    // Helper to check if a subtree contains a load from the reduction tensor
    bool ContainsTarget(const PrimExpr& expr) { return finder_->Find(expr); }

    ffi::Optional<VisitInterrupt> Visit_(const MulObj* op) final {
      if (has_scaling_) return std::nullopt;
      // If either operand subtree contains the reduction tensor load,
      // we treat this as invalid scaling of the reduction result.
      if (ContainsTarget(op->a) || ContainsTarget(op->b)) {
        has_scaling_ = true;
        return std::nullopt;
      }
      return StmtExprVisitor::Visit_(op);
    }

    ffi::Optional<VisitInterrupt> Visit_(const DivObj* op) final {
      if (has_scaling_) return std::nullopt;
      if (ContainsTarget(op->a) || ContainsTarget(op->b)) {
        has_scaling_ = true;
        return std::nullopt;
      }
      return StmtExprVisitor::Visit_(op);
    }

    ffi::Optional<VisitInterrupt> Visit_(const ModObj* op) final {
      if (has_scaling_) return std::nullopt;
      if (ContainsTarget(op->a) || ContainsTarget(op->b)) {
        has_scaling_ = true;
        return std::nullopt;
      }
      return StmtExprVisitor::Visit_(op);
    }

    ffi::ObjectPtr<TargetFinder> finder_;
    bool has_scaling_{false};
  };

  {
    auto detector = ffi::make_object<ScalingDetector>(inlined_tensor_);
    if (detector->HasScaling(inlined_store_->value)) {
      // Failure: Non-additive scaling of the reduction result is not supported
      return false;
    }
  }

  // 6. Check if producer is a reduction block
  if (!IsReductionBlock(reduction_block_)) {
    // Failure: producer is not a reduction block
    return false;
  }

  // 7. Extract epilogue information (output tensor, indices, regions, etc.)
  ExtractEpilogueInfo();

  return true;
}

bool ReductionEpilogueFuser::IsReductionBlock(const SBlockObj* block) {
  // Check if block has reduction iter vars
  for (const IterVar& iter : block->iter_vars) {
    if (iter->iter_type == kCommReduce) {
      return true;
    }
  }
  return false;
}

void ReductionEpilogueFuser::ExtractEpilogueInfo() {
  // Extract epilogue output tensor and indices
  epilogue_output_tensor_ = inlined_store_->dest.as_or_throw<TensorVar>();
  epilogue_output_indices_ = inlined_store_->indices;

  // Extract epilogue output region from epilogue block writes
  for (const TensorRegion& write : epilogue_block_->writes) {
    if (write->source.as_or_throw<tvm::tirx::TensorVar>().same_as(epilogue_output_tensor_)) {
      epilogue_output_region_ = write;
      break;
    }
  }

  // Generalized approach: extract all non-reduction tensors from epilogue expression
  // Find all tensors in epilogue expression (except the reduction tensor)
  struct TensorExtractor : public StmtExprVisitor {
    using StmtExprVisitor::Visit_;

    ffi::Optional<VisitInterrupt> Visit_(const TensorLoadObj* load) final {
      if (!load->source.as_or_throw<tvm::tirx::TensorVar>().same_as(reduction_tensor)) {
        other_tensors.insert(load->source.as_or_throw<tvm::tirx::TensorVar>().get());
      }
      return StmtExprVisitor::Visit_(load);
    }
    TensorVar reduction_tensor{ffi::UnsafeInit{}};
    std::unordered_set<const VarObj*> other_tensors;
  };
  auto extractor = ffi::make_object<TensorExtractor>();
  extractor->reduction_tensor = inlined_tensor_;
  extractor->Visit(epilogue_expression_);

  // Extract the first non-reduction tensor and its region
  // In most cases, there's one additional tensor (e.g., bias tensor)
  if (!extractor->other_tensors.empty()) {
    const VarObj* first_tensor = *extractor->other_tensors.begin();
    epilogue_addend_tensor_ = ffi::GetRef<Var>(first_tensor).as_or_throw<TensorVar>();
    // Find the read region from epilogue block reads
    for (const TensorRegion& read : epilogue_block_->reads) {
      if (read->source.as_or_throw<tvm::tirx::TensorVar>().get() == first_tensor) {
        epilogue_addend_region_ = read;
        break;
      }
    }
  }
}

SBlock ReductionEpilogueFuser::CreateFusedReductionBlock(
    const SBlockObj* reduction_block, const SBlockRealizeObj* reduction_realize) {
  ffi::ObjectPtr<SBlockObj> new_block = ffi::make_object<SBlockObj>(*reduction_block);

  // 1. Map epilogue block vars to reduction block vars
  std::vector<Var> reduction_data_vars;
  for (const IterVar& iter_var : reduction_block->iter_vars) {
    if (iter_var->iter_type == IterVarType::kDataPar) {
      reduction_data_vars.push_back(iter_var->var);
    }
  }
  std::vector<Var> epilogue_data_vars;
  for (const IterVar& iter_var : epilogue_block_->iter_vars) {
    if (iter_var->iter_type == IterVarType::kDataPar) {
      epilogue_data_vars.push_back(iter_var->var);
    }
  }

  TVM_FFI_CHECK_EQ(reduction_data_vars.size(), epilogue_data_vars.size(), ValueError)
      << "The number of data parallel iter vars must be the same in the reduction "
         "and epilogue blocks.";

  std::unordered_map<Var, Var> var_map;
  for (size_t i = 0; i < reduction_data_vars.size(); ++i) {
    var_map.insert_or_assign(epilogue_data_vars[i], reduction_data_vars[i]);
  }
  auto f_substitute = [&var_map](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (auto it = var_map.find(var); it != var_map.end()) {
      return ffi::Any(it->second);
    }
    return ffi::Unchanged();
  };

  // 2. Generalized init transformation: substitute reduction tensor load with identity element (0)
  // Create a substituter to replace reduction_tensor_load_ with identity element
  class InitSubstituter : public StmtExprMutator {
   public:
    using StmtExprMutator::Mutate;
    using StmtExprMutator::Mutate_;

    InitSubstituter(const TensorVar& target_tensor, PrimExpr identity_elem)
        : target_tensor_(target_tensor), identity_elem_(identity_elem) {}

    UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* op, InplaceMode inplace_mode) final {
      TensorLoad load = StmtExprMutator::Mutate_(op, inplace_mode)
                            .ValueOrUnchanged(ffi::GetRef<PrimExpr>(op))
                            .as_or_throw<TensorLoad>();
      if (load->source.as_or_throw<tvm::tirx::TensorVar>().same_as(target_tensor_)) {
        return identity_elem_;
      }
      return load;
    }

   private:
    TensorVar target_tensor_;
    PrimExpr identity_elem_;
  };

  // Identity element for reduction (assumed to be 0 for addition-based reductions)
  PrimExpr identity_elem = prim::MakeConst(epilogue_output_tensor_.dtype(), 0);

  // Substitute reduction tensor load with identity element
  auto init_subst = ffi::make_object<InitSubstituter>(inlined_tensor_, identity_elem);
  PrimExpr init_epilogue =
      init_subst->Mutate(epilogue_expression_).ValueOrUnchanged(epilogue_expression_);

  // Apply index mapping
  init_epilogue = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(init_epilogue, f_substitute)
                      .as_or_throw<PrimExpr>();

  // Simplify the expression (e.g., 0 + C[vi, vj] -> C[vi, vj])
  sym::Analyzer analyzer;
  init_epilogue = analyzer->Simplify(init_epilogue);

  ffi::Array<PrimExpr> init_indices =
      epilogue_output_indices_.Map([&f_substitute](const PrimExpr& index) {
        return ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(index, f_substitute)
            .as_or_throw<PrimExpr>();
      });
  TensorStore new_init_store = TensorStore(epilogue_output_tensor_, init_indices, init_epilogue);
  new_block->init = new_init_store;

  // 3. Generalized update transformation: apply epilogue expression with reduction tensor replaced
  // If reduction tensor load's parent is Add and other operand is not a reduction tensor,
  // remove that operand (bias addend) from update expression
  class UpdateSubstituter : public StmtExprMutator {
   public:
    using StmtExprMutator::Mutate;
    using StmtExprMutator::Mutate_;

    UpdateSubstituter(const TensorVar& old_tensor, const TensorVar& new_tensor,
                      const TensorVar& reduction_tensor, const PrimExpr& epilogue_expr,
                      const std::unordered_map<Var, Var>& var_map)
        : old_tensor_(old_tensor),
          new_tensor_(new_tensor),
          reduction_tensor_(reduction_tensor),
          epilogue_expression_(epilogue_expr),
          var_map_(var_map) {}

    UnchangedOr<Stmt> Mutate_(const TensorStoreObj* op, InplaceMode inplace_mode) final {
      TensorStore store = StmtExprMutator::Mutate_(op, inplace_mode)
                              .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                              .as_or_throw<TensorStore>();
      if (store->dest.as_or_throw<TensorVar>().same_as(old_tensor_)) {
        // Replace old_tensor_ in store->value with new_tensor_ to get the reduction update
        // expression This ensures store->value references new_tensor_ instead of old_tensor_
        class ReductionUpdateReplacer : public StmtExprMutator {
         public:
          using StmtExprMutator::Mutate;
          using StmtExprMutator::Mutate_;

          ReductionUpdateReplacer(const TensorVar& old_tensor, const TensorVar& new_tensor)
              : old_tensor_(old_tensor), new_tensor_(new_tensor) {}

          UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* op, InplaceMode inplace_mode) final {
            TensorLoad load = StmtExprMutator::Mutate_(op, inplace_mode)
                                  .ValueOrUnchanged(ffi::GetRef<PrimExpr>(op))
                                  .as_or_throw<TensorLoad>();
            if (load->source.as_or_throw<tvm::tirx::TensorVar>().same_as(old_tensor_)) {
              load.CopyOnWrite()->source = new_tensor_;
              return load;
            }
            return load;
          }

         private:
          TensorVar old_tensor_;
          TensorVar new_tensor_;
        };

        auto reduction_replacer =
            ffi::make_object<ReductionUpdateReplacer>(old_tensor_, new_tensor_);
        PrimExpr reduction_update =
            reduction_replacer->Mutate(store->value).ValueOrUnchanged(store->value);

        // Generalized approach: apply epilogue expression with reduction tensor load replaced
        // If reduction tensor load's direct parent is Add and the other operand is not a reduction
        // tensor, remove that operand (bias addend) from the update expression
        class GeneralizedEpilogueApplier : public StmtExprMutator {
         public:
          using StmtExprMutator::Mutate;
          using StmtExprMutator::Mutate_;

          GeneralizedEpilogueApplier(const TensorVar& target_tensor,
                                     const TensorVar& reduction_tensor, const PrimExpr& replacement)
              : target_tensor_(target_tensor),
                reduction_tensor_(reduction_tensor),
                replacement_(replacement),
                found_target_load_(false) {}

          UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* op, InplaceMode inplace_mode) final {
            TensorLoad load = StmtExprMutator::Mutate_(op, inplace_mode)
                                  .ValueOrUnchanged(ffi::GetRef<PrimExpr>(op))
                                  .as_or_throw<TensorLoad>();
            if (load->source.as_or_throw<tvm::tirx::TensorVar>().same_as(target_tensor_)) {
              found_target_load_ = true;
              // Check if parent is Add (will be checked in Dispatch_(const AddObj*))
              return replacement_;
            }
            return load;
          }

          UnchangedOr<PrimExpr> Mutate_(const AddObj* op, InplaceMode inplace_mode) final {
            // Visit children first to see if we find the target tensor load
            bool found_before = found_target_load_;
            found_target_load_ = false;

            PrimExpr a = Mutate(op->a, inplace_mode).ValueOrUnchanged(op->a);
            bool found_in_a = found_target_load_;
            found_target_load_ = false;

            PrimExpr b = Mutate(op->b, inplace_mode).ValueOrUnchanged(op->b);
            bool found_in_b = found_target_load_;

            // If target tensor load was found in this Add node
            if (found_in_a || found_in_b) {
              // Check if the other operand is NOT from the reduction tensor
              // If so, it's likely a bias addend that should be removed in update
              bool other_is_reduction = false;
              if (found_in_a) {
                // Check if b is from reduction tensor
                if (const auto* load_b = b.as<TensorLoadObj>()) {
                  other_is_reduction =
                      load_b->source.as_or_throw<tvm::tirx::TensorVar>().same_as(reduction_tensor_);
                }
                if (!other_is_reduction) {
                  // b is the bias addend, remove it
                  return a;
                }
              } else {  // found_in_b
                // Check if a is from reduction tensor
                if (const auto* load_a = a.as<TensorLoadObj>()) {
                  other_is_reduction =
                      load_a->source.as_or_throw<tvm::tirx::TensorVar>().same_as(reduction_tensor_);
                }
                if (!other_is_reduction) {
                  // a is the bias addend, remove it
                  return b;
                }
              }
              // If other operand is also from reduction tensor, keep the Add
              return Add(a, b);
            }

            // Target tensor load not found in this Add, return as is
            found_target_load_ = found_before;
            return Add(a, b);
          }

         private:
          const TensorVar& target_tensor_;
          const TensorVar& reduction_tensor_;
          const PrimExpr& replacement_;
          bool found_target_load_;
        };

        auto applier = ffi::make_object<GeneralizedEpilogueApplier>(old_tensor_, reduction_tensor_,
                                                                    reduction_update);
        PrimExpr new_value =
            applier->Mutate(epilogue_expression_).ValueOrUnchanged(epilogue_expression_);

        // Apply index mapping
        auto f_substitute = [this](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
          if (auto it = var_map_.find(var); it != var_map_.end()) {
            return ffi::Any(it->second);
          }
          return ffi::Unchanged();
        };
        new_value = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(new_value, f_substitute)
                        .as_or_throw<PrimExpr>();

        return TensorStore(new_tensor_, store->indices, new_value);
      }
      return store;
    }

    UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* op, InplaceMode inplace_mode) final {
      TensorLoad load = StmtExprMutator::Mutate_(op, inplace_mode)
                            .ValueOrUnchanged(ffi::GetRef<PrimExpr>(op))
                            .as_or_throw<TensorLoad>();
      if (load->source.as_or_throw<tvm::tirx::TensorVar>().same_as(old_tensor_)) {
        load.CopyOnWrite()->source = new_tensor_;
        return load;
      }
      return load;
    }

   private:
    TensorVar old_tensor_;
    TensorVar new_tensor_;
    TensorVar reduction_tensor_;
    PrimExpr epilogue_expression_;
    std::unordered_map<Var, Var> var_map_;
  };

  // Apply index mapping to epilogue expression first
  PrimExpr epilogue_expr_mapped =
      ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(epilogue_expression_, f_substitute)
          .as_or_throw<PrimExpr>();

  auto replacer = ffi::make_object<UpdateSubstituter>(
      inlined_tensor_, epilogue_output_tensor_, inlined_tensor_, epilogue_expr_mapped, var_map);
  new_block->body = replacer->Mutate(reduction_block->body).ValueOrUnchanged(reduction_block->body);

  // 4. Update write regions
  ffi::Array<TensorRegion> new_writes;
  for (const TensorRegion& write : reduction_block->writes) {
    if (write->source.as_or_throw<tvm::tirx::TensorVar>().same_as(inlined_tensor_)) {
      ffi::Array<Range> mapped_region = write->region.Map([&f_substitute](const Range& range) {
        PrimExpr min = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(range->begin, f_substitute)
                           .as_or_throw<PrimExpr>();
        PrimExpr extent = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(range->extent, f_substitute)
                              .as_or_throw<PrimExpr>();
        return Range::FromMinExtent(min, extent);
      });
      new_writes.push_back(
          TensorRegion(epilogue_output_tensor_, mapped_region, TensorRegionType()));
    } else {
      new_writes.push_back(write);
    }
  }
  new_block->writes = new_writes;

  // 5. Update read regions: add all tensors from epilogue expression (except reduction tensor)
  ffi::Array<TensorRegion> new_reads;
  std::unordered_set<const VarObj*> read_tensors;

  // Add all non-reduction tensors from epilogue expression
  for (const TensorRegion& read : epilogue_block_->reads) {
    if (!read->source.as_or_throw<tvm::tirx::TensorVar>().same_as(inlined_tensor_)) {
      ffi::Array<Range> mapped_region = read->region.Map([&f_substitute](const Range& range) {
        PrimExpr min = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(range->begin, f_substitute)
                           .as_or_throw<PrimExpr>();
        PrimExpr extent = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(range->extent, f_substitute)
                              .as_or_throw<PrimExpr>();
        return Range::FromMinExtent(min, extent);
      });
      new_reads.push_back(TensorRegion(read->source.as_or_throw<tvm::tirx::TensorVar>(),
                                       mapped_region, TensorRegionType()));
      read_tensors.insert(read->source.as_or_throw<tvm::tirx::TensorVar>().get());
    }
  }

  // Add existing read regions from reduction block (A, B, etc.)
  for (const TensorRegion& read : reduction_block->reads) {
    if (!read->source.as_or_throw<tvm::tirx::TensorVar>().same_as(inlined_tensor_)) {
      // Only add non-temp tensors that haven't been added yet
      if (read_tensors.find(read->source.as_or_throw<tvm::tirx::TensorVar>().get()) ==
          read_tensors.end()) {
        new_reads.push_back(read);
        read_tensors.insert(read->source.as_or_throw<tvm::tirx::TensorVar>().get());
      }
    }
  }

  new_block->reads = new_reads;

  return SBlock(new_block);
}

/*!
 * \brief Check if a tensor is still referenced by other blocks in the scope
 */
static bool CheckTensorStillUsed(const SBlock& scope_root, const TensorVar& tensor) {
  class TensorUsageChecker : public StmtExprVisitor {
   public:
    using StmtExprVisitor::Visit_;

    ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
      if (value.as<ExprObj>()) return std::nullopt;
      return StmtExprVisitor::Visit(value);
    }

    explicit TensorUsageChecker(const TensorVar& tensor) : tensor_(tensor) {}

    bool CheckStmt(const Stmt& stmt) {
      found_usage_ = false;
      Visit(stmt);
      return found_usage_;
    }

   private:
    ffi::Optional<VisitInterrupt> Visit_(const SBlockRealizeObj* op) final {
      if (found_usage_) return std::nullopt;

      if (!op || !op->block.defined()) {
        return StmtExprVisitor::Visit_(op);
      }

      const SBlockObj* block = op->block.get();
      if (!block) {
        return StmtExprVisitor::Visit_(op);
      }

      // Check reads
      for (const TensorRegion& read : block->reads) {
        if (read->source.as_or_throw<tvm::tirx::TensorVar>().same_as(tensor_)) {
          found_usage_ = true;
          return std::nullopt;
        }
      }

      // Check writes
      for (const TensorRegion& write : block->writes) {
        if (write->source.as_or_throw<tvm::tirx::TensorVar>().same_as(tensor_)) {
          found_usage_ = true;
          return std::nullopt;
        }
      }

      // Continue visiting nested blocks
      return StmtExprVisitor::Visit_(op);
    }

    ffi::Optional<VisitInterrupt> Visit_(const SBlockObj* op) final {
      if (found_usage_) return std::nullopt;
      if (!op) return std::nullopt;

      // Check alloc_tensors
      for (const TensorVar& tensor : op->alloc_tensors) {
        if (tensor.same_as(tensor_)) {
          found_usage_ = true;
          return std::nullopt;
        }
      }

      return StmtExprVisitor::Visit_(op);
    }

    const TensorVar& tensor_;
    bool found_usage_{false};
  };

  if (!scope_root->body.defined()) {
    return false;
  }

  auto checker = ffi::make_object<TensorUsageChecker>(tensor);
  return checker->CheckStmt(scope_root->body);
}

/*!
 * \brief Helper class to replace reduction and epilogue blocks with a single fused block
 */
class SingleBlockFusionReplacer : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) override {
    if (value.as<ExprObj>()) return ffi::Unchanged();
    return StmtExprMutator::Mutate(value, inplace_mode);
  }

  static SBlock Replace(SBlock old_scope_root, SBlock new_fused_block, SBlock old_reduction_block,
                        SBlock old_epilogue_block, TensorVar reduction_tensor) {
    auto replacer = ffi::make_object<SingleBlockFusionReplacer>(
        std::move(new_fused_block), std::move(old_reduction_block), std::move(old_epilogue_block),
        std::move(reduction_tensor));
    SBlock result = replacer->Mutate(old_scope_root, InplaceMode::kAllow)
                        .ValueOrUnchanged(std::move(old_scope_root))
                        .as_or_throw<SBlock>();

    // Check if reduction_tensor is still referenced by other blocks
    bool tensor_still_used = CheckTensorStillUsed(result, reduction_tensor);

    // Remove intermediate temp tensor only if it's not used by other blocks
    if (!tensor_still_used) {
      SBlockObj* p = result.CopyOnWrite();
      ffi::Array<TensorVar> new_alloc_tensors;
      for (const TensorVar& tensor : p->alloc_tensors) {
        if (!tensor.same_as(reduction_tensor)) {
          new_alloc_tensors.push_back(tensor);
        }
      }
      p->alloc_tensors = new_alloc_tensors;
    }

    return result;
  }

  explicit SingleBlockFusionReplacer(SBlock new_fused_block, SBlock old_reduction_block,
                                     SBlock old_epilogue_block, TensorVar reduction_tensor)
      : new_fused_block_(std::move(new_fused_block)),
        old_reduction_block_(std::move(old_reduction_block)),
        old_epilogue_block_(std::move(old_epilogue_block)),
        reduction_tensor_(std::move(reduction_tensor)) {}

 private:
  UnchangedOr<Stmt> Mutate_(const ForObj* loop, InplaceMode inplace_mode) final {
    SeqStmt mutated_body = Mutate(loop->body, inplace_mode).ValueOrUnchanged(loop->body);
    if (mutated_body->seq.empty()) {
      return mutated_body;
    }

    return For(loop->var, loop->begin, loop->end, loop->step, loop->kind, mutated_body,
               loop->annotations, loop->loc);
  }

  UnchangedOr<Stmt> Mutate_(const SBlockRealizeObj* realize, InplaceMode inplace_mode) final {
    if (realize->block.same_as(old_reduction_block_)) {
      // Replace reduction block with new fused block
      ffi::ObjectPtr<SBlockRealizeObj> new_realize = ffi::make_object<SBlockRealizeObj>(*realize);
      new_realize->block = new_fused_block_;
      return SBlockRealize(new_realize);
    } else if (realize->block.same_as(old_epilogue_block_)) {
      // Remove epilogue block completely.
      return SeqStmt({});
    }
    return StmtExprMutator::Mutate_(realize, inplace_mode);
  }

  UnchangedOr<Stmt> Mutate_(const SeqStmtObj* seq, InplaceMode inplace_mode) final {
    ffi::Array<Stmt> new_stmts;
    for (size_t i = 0; i < seq->seq.size(); ++i) {
      Stmt stmt = seq->seq[i];
      Stmt new_stmt = Mutate(stmt).ValueOrUnchanged(stmt).as_or_throw<Stmt>();
      new_stmts.push_back(new_stmt);
    }
    return SeqStmt(new_stmts, seq->loc);
  }

  SBlock new_fused_block_;
  SBlock old_reduction_block_;
  SBlock old_epilogue_block_;
  TensorVar reduction_tensor_;
};

void FuseReductionEpilogueImpl(ScheduleState self, const StmtSRef& reduction_block_sref,
                               const StmtSRef& epilogue_block_sref, bool check_only = false) {
  const SBlockObj* _reduction_block = TVM_SREF_TO_SBLOCK(reduction_block_sref);
  const SBlockObj* _epilogue_block = TVM_SREF_TO_SBLOCK(epilogue_block_sref);

  SBlock reduction_block = ffi::GetRef<SBlock>(_reduction_block);
  SBlock epilogue_block = ffi::GetRef<SBlock>(_epilogue_block);
  SBlockRealize epilogue_block_realize = GetSBlockRealize(self, epilogue_block_sref);

  // Step 1. Get the scope block
  StmtSRef scope_root_sref =
      GetScopeRoot(self, epilogue_block_sref, /*require_stage_pipeline=*/true);

  // Step 2. Get the reduction tensor (intermediate tensor)
  TensorVar reduction_tensor = NotSingleReadWriteTensor::GetSingleWrite(self, reduction_block);

  // Step 3. Check completeness and reduction block properties
  CheckReductionBlock(self, reduction_block_sref, scope_root_sref);
  CheckCompleteBlock(self, epilogue_block_sref, scope_root_sref);
  CheckNotOutputBlock(self, reduction_block_sref, scope_root_sref);

  // Step 4. Analyze the epilogue pattern
  auto fuser = ffi::make_object<ReductionEpilogueFuser>(reduction_tensor, _reduction_block,
                                                        epilogue_block_realize, scope_root_sref);
  if (!fuser->BodyPatternAllowFusion(epilogue_block_realize)) {
    throw MakeScheduleError<BodyAnalysisError>(true, self->mod, epilogue_block);
  }

  if (check_only) {
    return;
  }

  // Step 5. Create single fused reduction block
  SBlockRealize reduction_realize = GetSBlockRealize(self, reduction_block_sref);
  SBlock fused_block = fuser->CreateFusedReductionBlock(_reduction_block, reduction_realize.get());

  // Step 6. Transform and replace IR
  const SBlockObj* old_scope_root = TVM_SREF_TO_SBLOCK(scope_root_sref);

  SBlock new_scope_root =
      SingleBlockFusionReplacer::Replace(ffi::GetRef<SBlock>(old_scope_root), fused_block,
                                         reduction_block, epilogue_block, reduction_tensor);

  // Step 7. Update schedule state
  ffi::Map<SBlock, SBlock> block_reuse;
  block_reuse.Set(ffi::GetRef<SBlock>(old_scope_root), new_scope_root);
  block_reuse.Set(reduction_block, fused_block);
  self->Replace(scope_root_sref, new_scope_root, block_reuse);

  // Step 8. Update SBlockInfo
  self->UpdateScopeSBlockInfo(GetSBlockRealize(self, scope_root_sref));
}

void FuseReductionEpilogue(ScheduleState self, const StmtSRef& reduction_block_sref,
                           const StmtSRef& epilogue_block_sref) {
  FuseReductionEpilogueImpl(self, reduction_block_sref, epilogue_block_sref);
}

/******** InstructionKind Registration ********/

struct ComputeInlineTraits : public UnpackedInstTraits<ComputeInlineTraits> {
  static constexpr const char* kName = "ComputeInline";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, SBlockRV block_rv) {
    return sch->ComputeInline(block_rv);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block_rv) {
    PythonAPICall py("compute_inline");
    py.Input("block", block_rv);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

struct ReverseComputeInlineTraits : public UnpackedInstTraits<ReverseComputeInlineTraits> {
  static constexpr const char* kName = "ReverseComputeInline";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, SBlockRV block_rv) {
    return sch->ReverseComputeInline(block_rv);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block_rv) {
    PythonAPICall py("reverse_compute_inline");
    py.Input("block", block_rv);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

TVM_FFI_STATIC_INIT_BLOCK() {
  RegisterInstructionKind<ComputeInlineTraits>();
  RegisterInstructionKind<ReverseComputeInlineTraits>();
}

struct FuseReductionEpilogueTraits : public UnpackedInstTraits<FuseReductionEpilogueTraits> {
  static constexpr const char* kName = "FuseReductionEpilogue";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, SBlockRV reduction_block_rv,
                                      SBlockRV epilogue_block_rv) {
    return sch->FuseReductionEpilogue(reduction_block_rv, epilogue_block_rv);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs,
                                      ffi::String reduction_block_rv,
                                      ffi::String epilogue_block_rv) {
    PythonAPICall py("fuse_reduction_epilogue");
    py.Input("reduction_block", reduction_block_rv);
    py.Input("epilogue_block", epilogue_block_rv);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

TVM_FFI_STATIC_INIT_BLOCK() { RegisterInstructionKind<FuseReductionEpilogueTraits>(); }

}  // namespace s_tir
}  // namespace tvm
