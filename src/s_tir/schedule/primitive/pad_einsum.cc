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
#include <tvm/ir/prim/op.h>
#include <tvm/s_tir/stmt.h>

#include "../utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

/*!
 * \brief Check if tensor indices are all Vars and expr
 * \param tensor_access The TensorLoad or TensorStore
 * \return The indices if the indices are all Vars, otherwise std::nullopt
 */
ffi::Optional<ffi::Array<Var>> CheckTrivialTensorIndices(
    const ffi::Array<PrimExpr>& tensor_access) {
  ffi::Array<Var> indices;
  for (const PrimExpr& index : tensor_access) {
    if (index->IsInstance<IntImmObj>()) {
      continue;
    }
    auto var = index.as<PrimVar>();
    if (!var.has_value()) {
      return std::nullopt;
    }
    indices.push_back(var.value());
  }
  return indices;
}

ffi::Optional<ffi::Array<Var>> CheckTrivialTensorAccess(const TensorRegion& tensor_region) {
  ffi::Array<Var> indices;
  indices.reserve(tensor_region->region.size());
  for (const Range& range : tensor_region->region) {
    if (!tvm::prim::IsOne(range->extent)) {
      return std::nullopt;
    }
    if (range->begin->IsInstance<IntImmObj>()) {
      continue;
    }
    if (auto var = range->begin.as<PrimVar>()) {
      indices.push_back(var.value());
    } else {
      return std::nullopt;
    }
  }
  return indices;
}

/*! \brief The schedule error class when the padding size is invalid. */
class InvalidPaddingError : public ScheduleErrorContextObj {
 public:
  InvalidPaddingError(IRModule mod, SBlock block, ffi::Array<int64_t> padding)
      : mod_(std::move(mod)), block_(std::move(block)), padding_(std::move(padding)) {}
  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_}; }
  ffi::String FastErrorString() const final {
    return "ScheduleError: The padding size for the block is invalid.";
  }
  ffi::String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The padding for the block {0} are invalid. It should be a list of "
       << block_->iter_vars.size() << " positive integers. Got " << padding_;
    return os.str();
  }

  static void Check(const ScheduleState& self, const SBlock& block, ffi::Array<int64_t> padding) {
    if (padding.size() != block->iter_vars.size()) {
      throw MakeScheduleError<InvalidPaddingError>(self->mod, block, padding);
    }
    for (int64_t pad : padding) {
      if (pad <= 0) {
        throw MakeScheduleError<InvalidPaddingError>(self->mod, block, padding);
      }
    }
  }

 private:
  IRModule mod_;
  SBlock block_;
  ffi::Array<int64_t> padding_;
};

/*! \brief The schedule error class when the block body is not an Einsum pattern. */
class NonEinsumError : public ScheduleErrorContextObj {
 public:
  explicit NonEinsumError(IRModule mod, SBlock block)
      : mod_(std::move(mod)), block_(std::move(block)) {}

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_}; }
  ffi::String FastErrorString() const final {
    return "ScheduleError: The block is not a computation of Einsum pattern.";
  }
  ffi::String DetailRenderTemplate() const final {
    return "The block {0} not a computation of Einsum pattern.";
  }

 private:
  IRModule mod_;
  SBlock block_;
};

/*! \brief Data structure that represents a Einsum computation. */
struct Einsum {
  // The output tensor
  ffi::Array<TensorVar> output_tensors;
  // The indices of the output tensor
  ffi::Map<TensorVar, ffi::Array<Var>> output_indices;
  // The input tensors
  ffi::Array<TensorVar> input_tensors;
  // The indices of the input tensors
  ffi::Map<TensorVar, ffi::Array<Var>> input_indices;
};

struct TensorPadding {
  TensorVar tensor;
  TensorVar padded_tensor;

  static TensorPadding FromTensorRegion(const TensorRegion& tensor_region,
                                        const ffi::Map<Var, PrimExpr>& iter_extents) {
    TensorVar tensor = tensor_region->source.as_or_throw<tvm::tirx::TensorVar>();
    ffi::Array<PrimExpr> shape;
    shape.reserve(tensor_region->region.size());
    int ndim = tensor_region->region.size();
    for (int i = 0; i < ndim; ++i) {
      PrimExpr pos = tensor_region->region[i]->begin;
      TVM_FFI_ICHECK(pos->IsInstance<IntImmObj>() || pos.as<PrimVar>());
      if (pos->IsInstance<IntImmObj>()) {
        shape.push_back(IntImm(pos.ty(), 1));
      } else if (ffi::Optional<PrimExpr> extent = iter_extents.Get(pos.as_or_throw<Var>())) {
        shape.push_back(extent.value());
      } else {
        shape.push_back(tensor_region->source.as_or_throw<tvm::tirx::TensorVar>().shape()[i]);
      }
    }
    return {tensor,
            TensorVar(tensor->name + "_pad", TensorType(shape, tensor.dtype(), tensor.scope()))};
  }

  Stmt MakeCopyBlock(bool is_read, ffi::Array<SBlock>* blocks, sym::AnalyzerObj* analyzer) {
    ffi::Array<Var> loop_vars;
    ffi::Array<Range> loop_doms;
    ffi::Array<IterVar> iter_vars;
    ffi::Array<Range> instance_dom;
    ffi::Array<PrimExpr> indices;
    int ndim = tensor.shape().size();
    for (int i = 0; i < ndim; ++i) {
      PrimExpr dim = is_read ? padded_tensor.shape()[i] : tensor.shape()[i];
      Range dom = Range::FromMinExtent(IntImm(dim.ty(), 0), dim);
      loop_vars.push_back(Var("i" + std::to_string(i), dim.ty()));
      loop_doms.push_back(dom);
      IterVar iter_var(dom, PrimVar("v" + std::to_string(i), dim.ty()), kDataPar);
      instance_dom.push_back(Range::FromMinExtent(iter_var->var, IntImm(dim.ty(), 1)));
      iter_vars.push_back(iter_var);
      indices.push_back(iter_var->var);
    }
    Stmt body = [&]() -> Stmt {
      if (is_read) {
        PrimExpr predicate = IntImm::Bool(true);
        for (int i = 0; i < ndim; ++i) {
          if (!analyzer->CanProveEqual(tensor.shape()[i], padded_tensor.shape()[i])) {
            predicate = predicate && (indices[i] < tensor.shape()[i]);
          }
        }
        PrimExpr rhs = TensorLoad(tensor, indices);
        return TensorStore(padded_tensor, indices,
                           if_then_else(predicate, rhs, prim::MakeConst(rhs.ty(), 0)));
      } else {
        return TensorStore(tensor, indices, TensorLoad(padded_tensor, indices));
      }
    }();
    TensorRegion read_region = TensorRegion(tensor, instance_dom, TensorRegionType());
    TensorRegion write_region = TensorRegion(padded_tensor, instance_dom, TensorRegionType());
    if (!is_read) {
      std::swap(read_region, write_region);
    }
    SBlock new_block(iter_vars, {read_region}, {write_region}, padded_tensor->name,
                     std::move(body));
    blocks->push_back(new_block);
    ffi::Array<PrimExpr> prim_loop_vars;
    prim_loop_vars.reserve(loop_vars.size());
    for (const Var& var : loop_vars) prim_loop_vars.push_back(var.as_or_throw<PrimExpr>());
    body = SBlockRealize(prim_loop_vars, IntImm::Bool(true), new_block);
    for (int i = ndim - 1; i >= 0; --i) {
      body = For(loop_vars[i].as_or_throw<PrimVar>(), loop_doms[i]->begin,
                 loop_doms[i]->begin + loop_doms[i]->extent,
                 IntImm(loop_vars[i].as_or_throw<PrimVar>().ty(), 1), ForKind::kDefault,
                 std::move(body));
    }
    return body;
  }
};

Einsum ExtractEinsum(const ScheduleState& self, const SBlock& block) {
  Einsum result;
  std::unordered_set<const VarObj*> tensor_used;
  int n_reads = block->reads.size();
  for (int i = 0; i < n_reads; ++i) {
    const TensorVar& tensor = block->reads[i]->source.as_or_throw<tvm::tirx::TensorVar>();
    if (tensor_used.count(tensor.get()) != 0) {
      throw MakeScheduleError<NonEinsumError>(self->mod, block);
    }
    tensor_used.insert(tensor.get());
    if (ffi::Optional<ffi::Array<Var>> opt_indices = CheckTrivialTensorAccess(block->reads[i])) {
      result.input_tensors.push_back(tensor);
      result.input_indices.Set(tensor, opt_indices.value());
    } else {
      throw MakeScheduleError<NonEinsumError>(self->mod, block);
    }
  }
  int n_writes = block->writes.size();
  for (int i = 0; i < n_writes; ++i) {
    const TensorVar& tensor = block->writes[i]->source.as_or_throw<tvm::tirx::TensorVar>();
    if (tensor_used.count(tensor.get()) != 0) {
      throw MakeScheduleError<NonEinsumError>(self->mod, block);
    }
    tensor_used.insert(tensor.get());
    if (ffi::Optional<ffi::Array<Var>> opt_indices = CheckTrivialTensorAccess(block->writes[i])) {
      result.output_tensors.push_back(tensor);
      result.output_indices.Set(tensor, opt_indices.value());
    } else {
      throw MakeScheduleError<NonEinsumError>(self->mod, block);
    }
  }
  return result;
}

class TensorNotAllocatedInScopeError : public ScheduleErrorContextObj {
 public:
  explicit TensorNotAllocatedInScopeError(IRModule mod, TensorVar tensor)
      : mod_(std::move(mod)), tensor_(std::move(tensor)) {}

  ffi::String FastErrorString() const final {
    return "ScheduleError: The tensor is not allocated as an intermediate tensor in current "
           "Function.";
  }

  ffi::String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The tensor " << tensor_->name
       << " is not allocated as an intermediate tensor in current Function.";
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {}; }

 private:
  IRModule mod_;
  TensorVar tensor_;
};

/*! \brief The schedule error class when the producer block cannot be padded. */
class InvalidProducerError : public ScheduleErrorContextObj {
 public:
  explicit InvalidProducerError(IRModule mod, SBlock producer)
      : mod_(std::move(mod)), producer_(std::move(producer)) {}

  ffi::String FastErrorString() const final {
    return "ScheduleError: The producer block cannot be padded.";
  }

  ffi::String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The producer block {0} cannot be padded. It should write to a single tensor and the "
          "body should be a TensorStore.";
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {producer_}; }

 private:
  IRModule mod_;
  SBlock producer_;
};

class PadEinsumTensorReplacer : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  UnchangedOr<Stmt> Mutate_(const SBlockObj* old_block_ptr, InplaceMode inplace_mode) final {
    SBlock old_block = ffi::GetRef<SBlock>(old_block_ptr);
    SBlock block = StmtExprMutator::Mutate_(old_block_ptr, inplace_mode)
                       .ValueOrUnchanged(ffi::GetRef<Stmt>(old_block_ptr))
                       .as_or_throw<SBlock>();
    ffi::Array<IterVar> iter_vars;
    iter_vars.reserve(block->iter_vars.size());
    for (const IterVar& iter_var : block->iter_vars) {
      if (ffi::Optional<PrimExpr> new_dom = iter2padded_extents.Get(iter_var->var)) {
        ffi::ObjectPtr<IterVarObj> new_iter_var = ffi::make_object<IterVarObj>(*iter_var.get());
        new_iter_var->dom = Range::FromMinExtent(iter_var->dom.value()->begin, new_dom.value());
        iter_vars.push_back(IterVar(new_iter_var));
      } else {
        iter_vars.push_back(iter_var);
      }
    }
    ffi::Array<TensorRegion> reads;
    reads.reserve(block->reads.size());
    for (const TensorRegion& read : block->reads) {
      if (ffi::Optional<TensorVar> tensor =
              VarRemapGet(read->source.as_or_throw<tvm::tirx::TensorVar>()).as<TensorVar>()) {
        reads.push_back(TensorRegion(tensor.value(), read->region, TensorRegionType()));
      } else {
        reads.push_back(read);
      }
    }
    ffi::Array<TensorRegion> writes;
    writes.reserve(block->writes.size());
    for (const TensorRegion& write : block->writes) {
      if (ffi::Optional<TensorVar> tensor =
              VarRemapGet(write->source.as_or_throw<tvm::tirx::TensorVar>()).as<TensorVar>()) {
        writes.push_back(TensorRegion(tensor.value(), write->region, TensorRegionType()));
      } else {
        writes.push_back(write);
      }
    }
    SBlock new_block =
        SBlock(iter_vars, reads, writes, block->name_hint, block->body, block->init,
               /*alloc_tensors=*/{}, /*match_tensors=*/{}, /*annotations=*/block->annotations);
    block_sref_reuse_.Set(old_block, new_block);
    return new_block;
  }

  UnchangedOr<Stmt> Mutate_(const ForObj* old_for_ptr, InplaceMode inplace_mode) final {
    For old_for = ffi::GetRef<For>(old_for_ptr);
    For new_for = StmtExprMutator::Mutate_(old_for_ptr, inplace_mode)
                      .ValueOrUnchanged(ffi::GetRef<Stmt>(old_for_ptr))
                      .as_or_throw<For>();
    if (ffi::Optional<PrimExpr> new_extent = loop_var2padded_extent.Get(new_for->var)) {
      ffi::ObjectPtr<ForObj> new_for_ptr = ffi::make_object<ForObj>(*new_for.get());
      new_for_ptr->end = new_for_ptr->begin + new_extent.value();
      new_for = For(new_for_ptr);
    }
    return new_for;
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreObj* old_store_ptr, InplaceMode inplace_mode) final {
    TensorStore store = StmtExprMutator::Mutate_(old_store_ptr, inplace_mode)
                            .ValueOrUnchanged(ffi::GetRef<Stmt>(old_store_ptr))
                            .as_or_throw<TensorStore>();
    if (ffi::Optional<TensorVar> tensor =
            VarRemapGet(store->dest.as_or_throw<TensorVar>()).as<TensorVar>()) {
      return TensorStore(tensor.value(), store->indices, store->value);
    } else {
      return store;
    }
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* old_load_ptr, InplaceMode inplace_mode) final {
    TensorLoad load = StmtExprMutator::Mutate_(old_load_ptr, inplace_mode)
                          .ValueOrUnchanged(ffi::GetRef<PrimExpr>(old_load_ptr))
                          .as_or_throw<TensorLoad>();
    if (ffi::Optional<TensorVar> tensor = VarRemapGet(load->source).as<TensorVar>()) {
      return TensorLoad(tensor.value(), load->indices);
    } else {
      return load;
    }
  }

  ffi::Map<Var, PrimExpr> iter2padded_extents;
  ffi::Map<Var, PrimExpr> loop_var2padded_extent;
  ffi::Map<SBlock, SBlock> block_sref_reuse_;
};

void PadEinsum(ScheduleState self, const StmtSRef& block_sref, const ffi::Array<int64_t>& padding) {
  sym::Analyzer analyzer;
  // Step 1: Input checking and error handling
  const SBlockObj* block = TVM_SREF_TO_SBLOCK(block_sref);
  SBlockRealize realize = GetSBlockRealize(self, block_sref);
  StmtSRef scope_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/true);
  const SBlockObj* scope_block = TVM_SREF_TO_SBLOCK(scope_sref);
  InvalidPaddingError::Check(self, ffi::GetRef<SBlock>(block), padding);
  // Step 2. Extract the Einsum pattern
  ExtractEinsum(self, ffi::GetRef<SBlock>(block));
  // Step 3. Figure out the padding needed
  auto replacer = ffi::make_object<PadEinsumTensorReplacer>();
  for (int i = 0, n = padding.size(); i < n; ++i) {
    const IterVar& iter = block->iter_vars[i];
    PrimExpr dom = iter->dom.value()->extent;
    PrimExpr pad_imm = IntImm(dom.ty(), padding[i]);
    PrimExpr new_dom = analyzer->Simplify(ceildiv(dom, pad_imm) * pad_imm);
    if (!analyzer->CanProveEqual(new_dom, dom)) {
      replacer->iter2padded_extents.Set(iter->var, new_dom);
      if (auto loop_var = realize->iter_values[i].as<PrimVar>()) {
        replacer->iter2padded_extents.Set(loop_var.value(), new_dom);
        replacer->loop_var2padded_extent.Set(loop_var.value(), new_dom);
      }
    }
  }
  auto f_needs_padding = [&replacer](const ffi::Array<Range>& region) {
    for (const Range& range : region) {
      if (auto var = range->begin.as<PrimVar>()) {
        if (replacer->iter2padded_extents.count(var.value())) {
          return true;
        }
      }
    }
    return false;
  };
  // Step 3. Convert the subtree under the scope root
  ffi::Array<Stmt> scope_body = scope_block->body->seq;
  // Step 4. Find out the block of our interest
  int pos = -1;
  for (int i = 0; i < static_cast<int>(scope_body.size()); ++i) {
    auto walk_fn = [&block](const SBlock& node) -> ffi::Expected<ffi::WalkResult> {
      if (node.get() == block) {
        return ffi::WalkResult::Interrupt(ffi::VisitInterrupt(true));
      }
      return ffi::WalkResult::Advance();
    };
    auto result = ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(scope_body[i], walk_fn);
    if (result.has_value() && result.value()->value.cast<bool>()) {
      pos = i;
      break;
    }
  }
  TVM_FFI_ICHECK_NE(pos, -1);
  // Step 5. For each tensor, if it needs padding, create a new tensor and a new block
  ffi::Array<Stmt> read_blocks;
  ffi::Array<Stmt> write_blocks;
  ffi::Array<SBlock> new_copy_blocks;
  ffi::Array<TensorVar> alloc_tensors;
  for (const TensorRegion& tensor_region : block->reads) {
    if (f_needs_padding(tensor_region->region)) {
      TensorPadding bp =
          TensorPadding::FromTensorRegion(tensor_region, replacer->iter2padded_extents);
      replacer->VarRemapSet(bp.tensor, bp.padded_tensor);
      read_blocks.push_back(bp.MakeCopyBlock(true, &new_copy_blocks, analyzer.get()));
      alloc_tensors.push_back(bp.padded_tensor);
    }
  }
  for (const TensorRegion& tensor_region : block->writes) {
    if (f_needs_padding(tensor_region->region)) {
      TensorPadding bp =
          TensorPadding::FromTensorRegion(tensor_region, replacer->iter2padded_extents);
      replacer->VarRemapSet(bp.tensor, bp.padded_tensor);
      write_blocks.push_back(bp.MakeCopyBlock(false, &new_copy_blocks, analyzer.get()));
      alloc_tensors.push_back(bp.padded_tensor);
    }
  }
  // Step 6. Create new scope body
  ffi::Array<Stmt> new_scope_body;
  for (int i = 0; i < static_cast<int>(scope_body.size()); ++i) {
    if (i != pos) {
      new_scope_body.push_back(scope_body[i]);
      continue;
    }
    new_scope_body.insert(new_scope_body.end(), read_blocks.begin(), read_blocks.end());
    new_scope_body.push_back(replacer->Mutate(scope_body[i]).ValueOrUnchanged(scope_body[i]));
    new_scope_body.insert(new_scope_body.end(), write_blocks.begin(), write_blocks.end());
  }
  // Step 7. Create new scope
  SBlock new_scope_block = [&]() {
    ffi::ObjectPtr<SBlockObj> n = ffi::make_object<SBlockObj>(*scope_block);
    n->body = SeqStmt(new_scope_body, scope_block->body->loc);
    n->alloc_tensors.insert(n->alloc_tensors.end(), alloc_tensors.begin(), alloc_tensors.end());
    return SBlock(n);
  }();
  replacer->block_sref_reuse_.Set(ffi::GetRef<SBlock>(scope_block), new_scope_block);
  // Step 8. Do replacement and update flags
  self->Replace(scope_sref, new_scope_block, replacer->block_sref_reuse_);
  for (const SBlock& block : new_copy_blocks) {
    StmtSRef block_sref = self->stmt2ref.at(block.get());
    SBlockInfo& block_info = self->block_info[block_sref];
    block_info.affine_binding = true;
    block_info.region_cover = true;
    block_info.stage_pipeline = true;
  }
}

/******** Instruction Registration ********/

struct PadEinsumTraits : public UnpackedInstTraits<PadEinsumTraits> {
  static constexpr const char* kName = "PadEinsum";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, SBlockRV block, ffi::Array<int64_t> padding) {
    sch->PadEinsum(block, padding);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block,
                                      ffi::Array<int64_t> padding) {
    PythonAPICall py("pad_einsum");
    py.Input("block", block);
    py.Input("padding", padding);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

TVM_FFI_STATIC_INIT_BLOCK() { RegisterInstructionKind<PadEinsumTraits>(); }

}  // namespace s_tir
}  // namespace tvm
