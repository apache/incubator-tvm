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
#include <tvm/runtime/memory_scope.h>
#include <tvm/s_tir/stmt.h>

#include <string>

#include "../utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

using support::NDIntSet;

bool HasTensor(const ffi::Array<TensorRegion>& tensor_regions, const TensorVar& tensor) {
  for (const TensorRegion& tensor_region : tensor_regions) {
    if (tensor_region->source.as_or_throw<tvm::tirx::TensorVar>().same_as(tensor)) {
      return true;
    }
  }
  return false;
}

void RelaxTensorRegions(const ffi::Array<TensorRegion>& tensor_regions,
                        const TensorVar& tensor,                    //
                        const ffi::Map<Var, sym::IntSet>& var_dom,  //
                        const ffi::Map<Var, PrimExpr>& bindings,    //
                        std::vector<NDIntSet>* relaxed_regions) {
  auto f_substitute = [&bindings](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (auto repl = bindings.Get(var)) return ffi::Any(*std::move(repl));
    return ffi::Unchanged();
  };
  for (const TensorRegion& tensor_region : tensor_regions) {
    if (tensor_region->source.as_or_throw<tvm::tirx::TensorVar>().same_as(tensor)) {
      ffi::Array<Range> mapped_region =
          tensor_region->region.Map([&f_substitute](const Range& range) {
            PrimExpr min = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(range->begin, f_substitute)
                               .as_or_throw<PrimExpr>();
            PrimExpr extent =
                ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(range->extent, f_substitute)
                    .as_or_throw<PrimExpr>();
            return Range::FromMinExtent(min, extent);
          });
      ffi::Array<sym::IntSet> relaxed_region = sym::EvalSet(mapped_region, var_dom);
      relaxed_regions->push_back({relaxed_region.begin(), relaxed_region.end()});
    }
  }
}

class ScopeReplacer : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  static SBlock Replace(const SBlockObj* scope_block, const TensorVar& dst, const ForObj* old_loop,
                        const ForObj* new_loop) {
    ffi::ObjectPtr<SBlockObj> new_scope_block = ffi::make_object<SBlockObj>(*scope_block);
    new_scope_block->body = ffi::make_object<ScopeReplacer>(old_loop, new_loop)
                                ->Mutate(new_scope_block->body, InplaceMode::kAllow)
                                .ValueOrUnchanged(std::move(new_scope_block->body));
    new_scope_block->alloc_tensors.push_back(dst);
    return SBlock(new_scope_block);
  }

  explicit ScopeReplacer(const ForObj* old_loop, const ForObj* new_loop)
      : old_loop_(old_loop), new_loop_(new_loop), found_(false) {}

 private:
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) final {
    if (value.as<ExprObj>() || found_) return ffi::Unchanged();
    return StmtExprMutator::Mutate(value, inplace_mode);
  }
  UnchangedOr<Stmt> Mutate_(const SBlockObj* block, InplaceMode inplace_mode) final {
    return ffi::Unchanged();
  }
  UnchangedOr<Stmt> Mutate_(const ForObj* loop, InplaceMode inplace_mode) final {
    if (loop == old_loop_) {
      found_ = true;
      return ffi::GetRef<For>(new_loop_);
    }
    return StmtExprMutator::Mutate_(loop, inplace_mode);
  }

  const ForObj* old_loop_;
  const ForObj* new_loop_;
  bool found_;
};

class ReadWriteAtTensorReplacer : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  explicit ReadWriteAtTensorReplacer(const TensorVar& src, const TensorVar& dst,
                                     ffi::Map<SBlock, SBlock>* block_sref_reuse)
      : src_(src), dst_(dst), block_sref_reuse_(block_sref_reuse) {}

 private:
  UnchangedOr<Stmt> Mutate_(const TensorStoreObj* _store, InplaceMode inplace_mode) final {
    TensorStore store = StmtExprMutator::Mutate_(_store, inplace_mode)
                            .ValueOrUnchanged(ffi::GetRef<Stmt>(_store))
                            .as_or_throw<TensorStore>();
    if (store->dest.as_or_throw<TensorVar>().same_as(src_)) {
      ffi::ObjectPtr<TensorStoreObj> new_store = ffi::make_object<TensorStoreObj>(*store.get());
      new_store->dest = dst_;
      return TensorStore(new_store);
    }
    return store;
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* _load, InplaceMode inplace_mode) final {
    TensorLoad load = StmtExprMutator::Mutate_(_load, inplace_mode)
                          .ValueOrUnchanged(ffi::GetRef<PrimExpr>(_load))
                          .as_or_throw<TensorLoad>();
    if (load->source.as_or_throw<tvm::tirx::TensorVar>().same_as(src_)) {
      return TensorLoad(dst_, load->indices, load->loc);
    }
    return load;
  }

  UnchangedOr<Stmt> Mutate_(const SBlockObj* _block, InplaceMode inplace_mode) final {
    SBlock old_block = ffi::GetRef<SBlock>(_block);
    SBlock block = StmtExprMutator::Mutate_(_block, InplaceMode::kDisallow)
                       .ValueOrUnchanged(ffi::GetRef<Stmt>(_block))
                       .as_or_throw<SBlock>();
    ffi::ObjectPtr<SBlockObj> new_block = ffi::make_object<SBlockObj>(*block.get());
    new_block->reads = ReplaceTensor(new_block->reads, src_, dst_);
    new_block->writes = ReplaceTensor(new_block->writes, src_, dst_);
    block_sref_reuse_->Set(old_block, SBlock(new_block));
    return SBlock(new_block);
  }

  const TensorVar& src_;
  const TensorVar& dst_;
  ffi::Map<SBlock, SBlock>* block_sref_reuse_;
};

struct ReadWriteAtImpl {
  template <bool is_read>
  static StmtSRef Main(ScheduleState self, const StmtSRef& loop_sref, const StmtSRef& block_sref,
                       int tensor_index, const ffi::String& storage_scope,
                       ffi::Map<ffi::String, Any> annotations) {
    const SBlockObj* block = TVM_SREF_TO_SBLOCK(block_sref);
    TensorVar src = GetNthAccessTensor(self, ffi::GetRef<SBlock>(block), tensor_index,
                                       is_read ? TensorIndexType::kRead : TensorIndexType::kWrite);
    TensorVar dst = WithScope(src, runtime::MemoryScope::Get(storage_scope));
    ReadWriteAtImpl impl(self, loop_sref, src, dst, annotations);
    std::pair<For, SBlockRealize> new_loop_block =
        impl.MakeLoopAndBlock<is_read>(src->name + "_" + storage_scope);
    StmtSRef result_block_sref =
        impl.ReplaceScopeBlock(new_loop_block.first.get(), new_loop_block.second->block.get());
    impl.UpdateSBlockInfo(result_block_sref, !new_loop_block.second->iter_values.empty());
    return result_block_sref;
  }

 private:
  static ffi::Map<Var, Range> GetLoopDomain(const StmtSRefObj* loop_sref) {
    ffi::Map<Var, Range> result;
    for (const ForObj* loop; (loop = loop_sref->StmtAs<ForObj>()) != nullptr;
         loop_sref = loop_sref->parent) {
      result.Set(loop->var, Range(loop->begin, loop->end));
    }
    return result;
  }

  StmtSRef ReplaceScopeBlock(const ForObj* new_loop, const SBlockObj* new_block) {
    StmtSRef scope_root_sref = GetScopeRoot(self_, loop_sref_,
                                            /*require_stage_pipeline=*/true);
    const SBlockObj* scope_block = TVM_SREF_TO_SBLOCK(scope_root_sref);
    SBlock new_scope_block = ScopeReplacer::Replace(scope_block, dst_, loop_, new_loop);
    block_sref_reuse_.Set(ffi::GetRef<SBlock>(scope_block), new_scope_block);
    self_->Replace(scope_root_sref, new_scope_block, block_sref_reuse_);
    return self_->stmt2ref.at(new_block);
  }

  void UpdateSBlockInfo(const StmtSRef& new_block_sref, bool affine_binding) {
    SBlockInfo& block_info = self_->block_info[new_block_sref];
    block_info.affine_binding = affine_binding;
    block_info.region_cover = true;
    block_info.stage_pipeline = true;
  }

  template <bool is_read>
  std::pair<For, SBlockRealize> MakeLoopAndBlock(const ffi::String& new_block_name_hint) {
    ffi::Array<Stmt> subtrees = AsArray(loop_->body);
    int n_subtrees = subtrees.size();
    runtime::MemoryScope scope = dst_.scope();
    std::vector<NDIntSet> relaxed_regions;
    std::vector<int> r_pos;
    std::vector<int> w_pos;
    relaxed_regions.reserve(n_subtrees);
    r_pos.reserve(n_subtrees);
    w_pos.reserve(n_subtrees);
    // Step 1. Iterate over all subtrees
    for (int i = 0; i < n_subtrees; ++i) {
      bool r_visited = false;
      bool w_visited = false;
      auto f_visit = [this, &relaxed_regions, &r_visited, &w_visited,
                      &scope](const SBlockRealize& realize) -> ffi::Expected<ffi::WalkResult> {
        const SBlockObj* block = realize->block.get();
        bool has_r = HasTensor(block->reads, src_);
        bool has_w = HasTensor(block->writes, src_);
        r_visited = r_visited || has_r;
        w_visited = w_visited || has_w;
        if (is_read ? has_r : has_w) {
          RelaxTensorRegions(
              /*tensor_regions=*/is_read ? block->reads : block->writes,
              /*tensor=*/src_,
              /*var_dom=*/
              sym::AsIntSet(LoopDomainOfSRefTreePath(
                  /*low_inclusive=*/ffi::GetRef<StmtSRef>(self_->stmt2ref.at(block)->parent),
                  /*high_exclusive=*/loop_sref_,
                  /*extra_relax_scope=*/scope)),
              /*bindings=*/GetBindings(realize),
              /*relaxed_regions=*/&relaxed_regions);
        }
        return ffi::WalkResult::Skip();
      };
      ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(subtrees[i], f_visit);
      if (r_visited) {
        r_pos.push_back(i);
      }
      if (w_visited) {
        w_pos.push_back(i);
      }
    }
    // Step 2. Calculate `insert_pos` and [st, ed) for tensor replacement
    int insert_pos = -1, st = -1, ed = -1;
    if (is_read) {
      TVM_FFI_ICHECK(!r_pos.empty());
      // No write after the first read
      TVM_FFI_ICHECK(w_pos.empty() || w_pos.back() < r_pos.front());
      // Can be inserted at [0, r_pos.front()], i.e. before the first read
      insert_pos = r_pos.front();
      // TensorVar reads in [insert_pos, +oo) is rewritten
      st = insert_pos;
      ed = n_subtrees;
    } else {
      TVM_FFI_ICHECK(!w_pos.empty());
      // No read after the last write
      TVM_FFI_ICHECK(r_pos.empty() || r_pos.back() <= w_pos.back());
      // Can be inserted into (w_pos.back(), +oo), i.e. after the last write
      insert_pos = w_pos.back() + 1;
      st = 0;
      ed = insert_pos;
    }
    // Step 3. Calculate `domain`, the domain of tensor access
    NDIntSet relaxed = support::NDIntSetUnion(relaxed_regions);
    int ndim = relaxed.size();
    ffi::Array<Range> domain;
    domain.reserve(ndim);
    for (int i = 0; i < ndim; ++i) {
      const sym::IntSet& int_set = relaxed[i];
      PrimExpr min = analyzer_->Simplify(int_set.min());
      PrimExpr extent = analyzer_->Simplify(int_set.max() + 1 - min);
      domain.push_back(Range::FromMinExtent(min, extent));
    }
    // Step 4. Insert the auto copy block and replace tensors
    auto replacer = ffi::make_object<ReadWriteAtTensorReplacer>(src_, dst_, &block_sref_reuse_);
    for (int i = st; i < ed; ++i) {
      Stmt stmt = subtrees[i];
      subtrees.Set(
          i, replacer->Mutate(stmt, InplaceMode::kDisallow).ValueOrUnchanged(std::move(stmt)));
    }
    SBlockRealize realize =
        is_read
            ? MakeSBlock(src_, dst_, new_block_name_hint, GetLoopDomain(loop_sref_.get()), domain)
            : MakeSBlock(dst_, src_, new_block_name_hint, GetLoopDomain(loop_sref_.get()), domain);
    subtrees.insert(subtrees.begin() + insert_pos, realize);
    ffi::ObjectPtr<ForObj> new_loop = ffi::make_object<ForObj>(*loop_);
    new_loop->body = SeqStmt(std::move(subtrees));
    return {For(new_loop), realize};
  }

  SBlockRealize MakeSBlock(const TensorVar& copy_from, const TensorVar& copy_to,
                           const ffi::String& name_hint, const ffi::Map<Var, Range>& loop_domain,
                           ffi::Array<Range> domain) const {
    int n = domain.size();
    std::vector<Var> loop_vars;
    loop_vars.reserve(n);
    for (int i = 0; i < n; ++i) {
      loop_vars.push_back(PrimVar("ax" + std::to_string(i)));
    }
    ffi::Map<Var, PrimExpr> bindings;
    ffi::Array<IterVar> iter_vars;
    ffi::Array<PrimExpr> iter_values;
    ffi::Array<PrimExpr> indices;
    iter_vars.reserve(n);
    iter_values.reserve(n);
    indices.reserve(n);
    for (int i = 0; i < n; ++i) {
      auto f_substitute = [&loop_domain, &bindings, &iter_vars, &iter_values](
                              const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
        auto it = bindings.find(var);
        if (it != bindings.end()) {
          return ffi::Any((*it).second);
        }
        Range range = loop_domain.at(var);
        Var v("v" + std::to_string(iter_vars.size()), var->ty, var->loc);
        bindings.Set(var, v.as_or_throw<PrimExpr>());
        iter_values.push_back(var.as_or_throw<PrimExpr>());
        iter_vars.push_back(IterVar(range, v.as_or_throw<PrimVar>(), IterVarType::kDataPar));
        return ffi::Any(v.as_or_throw<PrimExpr>());
      };
      ffi::ObjectPtr<RangeObj> dom = ffi::make_object<RangeObj>(*domain[i].get());
      dom->begin =
          ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(std::move(dom->begin), f_substitute)
              .as_or_throw<PrimExpr>();
      dom->extent =
          ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(std::move(dom->extent), f_substitute)
              .as_or_throw<PrimExpr>();
      domain.Set(i, Range(ffi::UnsafeInit{}, dom));
    }
    for (int i = 0; i < n; ++i) {
      indices.push_back(domain[i]->begin + loop_vars[i].as_or_throw<PrimExpr>());
    }
    Stmt stmt = TensorStore(copy_to, /*indices=*/indices, /*value=*/TensorLoad(copy_from, indices));
    for (int i = n - 1; i >= 0; --i) {
      stmt = For(loop_vars[i].as_or_throw<PrimVar>(), IntImm::Int32(0), domain[i]->extent,
                 IntImm(loop_vars[i].as_or_throw<PrimVar>().ty(), 1), ForKind::kDefault, stmt);
    }
    return SBlockRealize(
        /*values=*/iter_values,
        /*predicate=*/IntImm::Bool(true),
        SBlock(/*iter_vars=*/iter_vars,
               /*reads=*/{TensorRegion(copy_from, domain, TensorRegionType())},
               /*writes=*/{TensorRegion(copy_to, domain, TensorRegionType())},
               /*name_hint=*/name_hint,  //
               /*body=*/std::move(stmt),
               /*init=*/std::nullopt,
               /*alloc_tensors=*/{},
               /*match_tensors=*/{},
               /*annotations=*/annotations_));
  }

  explicit ReadWriteAtImpl(ScheduleState self, const StmtSRef& loop_sref, const TensorVar& src,
                           const TensorVar& dst, ffi::Map<ffi::String, Any> annotations)
      : self_(self),
        loop_sref_(loop_sref),
        loop_(nullptr),
        src_(src),
        dst_(dst),
        annotations_(annotations),
        block_sref_reuse_(),
        analyzer_(sym::Analyzer()) {
    loop_ = TVM_SREF_TO_FOR(loop_sref);
  }

  ScheduleState self_;
  const StmtSRef& loop_sref_;
  const ForObj* loop_;
  const TensorVar& src_;
  const TensorVar& dst_;
  ffi::Map<ffi::String, Any> annotations_;
  ffi::Map<SBlock, SBlock> block_sref_reuse_;
  sym::Analyzer analyzer_;
};

StmtSRef ReadAt(ScheduleState self, const StmtSRef& loop_sref, const StmtSRef& block_sref,
                int read_tensor_index, const ffi::String& storage_scope) {
  return ReadWriteAtImpl::Main<true>(self, loop_sref, block_sref, read_tensor_index, storage_scope,
                                     {{s_tir::attr::kAutoCopy.name, true}});
}

StmtSRef WriteAt(ScheduleState self, const StmtSRef& loop_sref, const StmtSRef& block_sref,
                 int write_tensor_index, const ffi::String& storage_scope) {
  return ReadWriteAtImpl::Main<false>(self, loop_sref, block_sref, write_tensor_index,
                                      storage_scope, {{s_tir::attr::kAutoCopy.name, true}});
}

/******** Instruction Registration ********/

struct ReadAtTraits : public UnpackedInstTraits<ReadAtTraits> {
  static constexpr const char* kName = "ReadAt";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  StmtSRef ReadAt(ScheduleState self, const StmtSRef& loop_sref, const StmtSRef& block_sref,
                  int tensor_index, const ffi::String& storage_scope);
  static SBlockRV UnpackedApplyToSchedule(Schedule sch, LoopRV loop, SBlockRV block,
                                          int read_tensor_index, ffi::String storage_scope) {
    return sch->ReadAt(loop, block, read_tensor_index, storage_scope);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String loop,
                                      ffi::String block, int read_tensor_index,
                                      ffi::String storage_scope) {
    PythonAPICall py("read_at");
    py.Input("loop", loop);
    py.Input("block", block);
    py.Input("read_tensor_index", read_tensor_index);
    py.Input("storage_scope", storage_scope);
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

struct WriteAtTraits : public UnpackedInstTraits<WriteAtTraits> {
  static constexpr const char* kName = "WriteAt";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static SBlockRV UnpackedApplyToSchedule(Schedule sch, LoopRV loop, SBlockRV block,
                                          int write_tensor_index, ffi::String storage_scope) {
    return sch->WriteAt(loop, block, write_tensor_index, storage_scope);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String loop,
                                      ffi::String block, int write_tensor_index,
                                      ffi::String storage_scope) {
    PythonAPICall py("write_at");
    py.Input("loop", loop);
    py.Input("block", block);
    py.Input("write_tensor_index", write_tensor_index);
    py.Input("storage_scope", storage_scope);
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

TVM_FFI_STATIC_INIT_BLOCK() {
  RegisterInstructionKind<ReadAtTraits>();
  RegisterInstructionKind<WriteAtTraits>();
}

}  // namespace s_tir
}  // namespace tvm
