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
#include <tvm/s_tir/stmt.h>

#include "../utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

class SRefTreeVerifier : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;

  ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
    if (value.as<ExprObj>()) return std::nullopt;
    return StmtExprVisitor::Visit(value);
  }

  static void Verify(const ScheduleStateObj* self) {
    ffi::make_object<SRefTreeVerifier>(self)->Verify();
  }

  /*! \brief Constructor */
  explicit SRefTreeVerifier(const ScheduleStateObj* self) : self_(self) {}

 private:
  void Verify() {
    for (const auto& [_, base_func] : self_->mod->functions) {
      if (const auto* func = base_func.as<FunctionObj>()) {
        this->Visit(func->body);
      }
    }
    TVM_FFI_ICHECK_EQ(n_sref_visited_, static_cast<int>(self_->stmt2ref.size()));
    for (const auto& kv : self_->block_info) {
      const StmtSRef& sref = kv.first;
      TVM_FFI_CHECK(sref->stmt != nullptr, InternalError)
          << "An expired sref is found in the block_scope mapping";
      auto it = self_->stmt2ref.find(sref->stmt);
      TVM_FFI_CHECK(it != self_->stmt2ref.end(), InternalError)
          << "The sref points to a statement that does not exist in stmt2ref";
      const StmtSRef& sref2 = it->second;
      TVM_FFI_CHECK(sref.same_as(sref2), InternalError)
          << "The sref points to a statement whose corresponding sref in stmt2ref "
             "is not the same object as itself";
    }
    TVM_FFI_ICHECK_EQ(n_block_sref_visited_, static_cast<int>(self_->block_info.size()));
  }

  ffi::Optional<VisitInterrupt> Visit_(const SBlockObj* block) final {
    if (init_block_depth_) {
      TVM_FFI_CHECK(!self_->stmt2ref.count(block), InternalError)
          << "A block inside init block has its "
             "corresponding sref, which is not allowed";
      return StmtExprVisitor::Visit_(block);
    }
    TVM_FFI_CHECK(self_->stmt2ref.count(block), InternalError)
        << "A BlockNode should appear in sref map, but it didn't\n"
        << ffi::GetRef<Stmt>(block);
    ++n_sref_visited_;
    ++n_block_sref_visited_;
    const StmtSRef& sref = self_->stmt2ref.at(block);
    TVM_FFI_CHECK(self_->block_info.count(sref), InternalError)
        << "Cannot find scope information of the BlockNode:\n"
        << ffi::GetRef<Stmt>(block);
    TVM_FFI_CHECK(sref->parent == ancestors_.back(), InternalError)
        << "Parent information mismatch for BlockNode:\n"
        << ffi::GetRef<Stmt>(block) << "\nIts parent is supposed to be:\n"
        << ffi::GetRef<Stmt>(ancestors_.back()->stmt)
        << "\nHowever, its parent is incorrect and is:\n"
        << (sref->parent ? ffi::Optional<Stmt>(ffi::GetRef<Stmt>(sref->parent->stmt))
                         : ffi::Optional<Stmt>(std::nullopt));
    ancestors_.push_back(sref.operator->());
    if (block->init.has_value()) {
      ++init_block_depth_;
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(block->init.value()));
      --init_block_depth_;
    }
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(block->body));
    ancestors_.pop_back();
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const ForObj* loop) final {
    if (init_block_depth_) {
      TVM_FFI_CHECK(!self_->stmt2ref.count(loop), InternalError)
          << "A loop inside init block has its "
             "corresponding sref, which is not allowed";
      return StmtExprVisitor::Visit_(loop);
    }
    TVM_FFI_CHECK(self_->stmt2ref.count(loop), InternalError)
        << "A ForObj should appear in sref map, but it didn't\n"
        << ffi::GetRef<Stmt>(loop);
    ++n_sref_visited_;
    const StmtSRef& sref = self_->stmt2ref.at(loop);
    ffi::Optional<Stmt> stmt = std::nullopt;
    TVM_FFI_CHECK(sref->parent == ancestors_.back(), InternalError)
        << "Parent information mismatch for ForObj:\n"
        << ffi::GetRef<Stmt>(loop) << "\nIts parent is supposed to be:\n"
        << ffi::GetRef<Stmt>(ancestors_.back()->stmt)
        << "\nHowever, its parent is incorrect and is:\n"
        << (sref->parent ? ffi::Optional<Stmt>(ffi::GetRef<Stmt>(sref->parent->stmt))
                         : ffi::Optional<Stmt>(std::nullopt));
    ancestors_.push_back(sref.operator->());
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(loop));
    ancestors_.pop_back();
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const SeqStmtObj* seq_stmt) final {
    // Verify seq_index
    if (init_block_depth_) {
      return StmtExprVisitor::Visit_(seq_stmt);
    }
    int n = static_cast<int>(seq_stmt->seq.size());
    for (int i = 0; i < n; ++i) {
      const Stmt& child = seq_stmt->seq[i];
      StmtSRef sref{ffi::UnsafeInit{}};
      if (const auto* realize = child.as<SBlockRealizeObj>()) {
        const auto* block = realize->block.get();
        TVM_FFI_ICHECK(self_->stmt2ref.count(block));
        sref = self_->stmt2ref.at(block);
      } else if (child->IsInstance<ForObj>()) {
        TVM_FFI_ICHECK(self_->stmt2ref.count(child.get()));
        sref = self_->stmt2ref.at(child.get());
      } else {
        continue;
      }
      TVM_FFI_CHECK_EQ(sref->seq_index, i, InternalError) << "A StmtSRef has incorrect seq_index";
    }
    return StmtExprVisitor::Visit_(seq_stmt);
  }

  /*! \brief The schedule it belongs to */
  const ScheduleStateObj* self_;
  /*! \brief Parent information during the visit */
  std::vector<const StmtSRefObj*> ancestors_ = {nullptr};
  /*! \brief If the visitor is currently in the init block */
  int init_block_depth_ = 0;
  /*! \brief Number of srefs that are visited */
  int n_sref_visited_ = 0;
  /*! \brief Number of block srefs that are visited */
  int n_block_sref_visited_ = 0;
};

void VerifySRefTree(const ScheduleState& self) { SRefTreeVerifier::Verify(self.get()); }

void VerifyCachedFlags(const ScheduleState& self) {
  std::vector<StmtSRef> block_info_not_found;
  std::vector<std::tuple<StmtSRef, bool, bool>> block_info_wrong_affine_binding;
  std::vector<std::tuple<StmtSRef, bool, bool>> block_info_wrong_region_cover;
  std::vector<std::tuple<StmtSRef, bool, bool>> block_info_wrong_stage_pipeline;

  ScheduleState new_state(self->mod);
  for (const auto& kv : new_state->stmt2ref) {
    const StmtObj* stmt = kv.first;
    const StmtSRef& new_sref = kv.second;
    if (stmt->IsInstance<ForObj>() || !self->stmt2ref.count(stmt)) {
      continue;
    }
    const SBlockInfo& new_block_info = new_state->block_info.at(new_sref);
    const StmtSRef& old_sref = self->stmt2ref.at(stmt);
    if (!self->block_info.count(old_sref)) {
      block_info_not_found.push_back(new_sref);
      continue;
    }
    const SBlockInfo& old_block_info = self->block_info.at(old_sref);
    if (new_block_info.affine_binding != old_block_info.affine_binding) {
      block_info_wrong_affine_binding.emplace_back(new_sref,  //
                                                   new_block_info.affine_binding,
                                                   old_block_info.affine_binding);
    }
    if (new_block_info.region_cover != old_block_info.region_cover) {
      block_info_wrong_region_cover.emplace_back(new_sref,  //
                                                 new_block_info.region_cover,
                                                 old_block_info.region_cover);
    }
    if (new_block_info.stage_pipeline != old_block_info.stage_pipeline) {
      block_info_wrong_stage_pipeline.emplace_back(new_sref,  //
                                                   new_block_info.stage_pipeline,
                                                   old_block_info.stage_pipeline);
    }
  }

  bool has_not_found = !block_info_not_found.empty();
  bool has_wrong_affine_binding = !block_info_wrong_affine_binding.empty();
  bool has_wrong_region_cover = !block_info_wrong_region_cover.empty();
  bool has_wrong_stage_pipeline = !block_info_wrong_stage_pipeline.empty();
  if (!(has_not_found || has_wrong_affine_binding || has_wrong_region_cover ||
        has_wrong_stage_pipeline)) {
    return;
  }
  std::ostringstream os;
  if (has_not_found) {
    os << "- SBlockInfo not found:";
    for (const StmtSRef& block_sref : block_info_not_found) {
      const auto* block = block_sref->StmtAs<SBlockObj>();
      TVM_FFI_ICHECK(block);
      os << " " << block->name_hint;
    }
    os << std::endl;
  }
  if (has_wrong_affine_binding) {
    os << "- Wrong affine_binding: ";
    for (const std::tuple<StmtSRef, bool, bool>& record : block_info_wrong_affine_binding) {
      const StmtSRef& block_sref = std::get<0>(record);
      bool expected = std::get<1>(record);
      bool actual = std::get<2>(record);
      const auto* block = block_sref->StmtAs<SBlockObj>();
      TVM_FFI_ICHECK(block);
      os << " (" << block->name_hint << ", expected=" << expected << ", actual=" << actual << ")";
    }
    os << std::endl;
  }
  if (has_wrong_region_cover) {
    os << "- Wrong region_cover: ";
    for (const std::tuple<StmtSRef, bool, bool>& record : block_info_wrong_region_cover) {
      const StmtSRef& block_sref = std::get<0>(record);
      bool expected = std::get<1>(record);
      bool actual = std::get<2>(record);
      const auto* block = block_sref->StmtAs<SBlockObj>();
      TVM_FFI_ICHECK(block);
      os << " (" << block->name_hint << ", expected=" << expected << ", actual=" << actual << ")";
    }
    os << std::endl;
  }
  if (has_wrong_stage_pipeline) {
    os << "- Wrong stage_pipeline: ";
    for (const std::tuple<StmtSRef, bool, bool>& record : block_info_wrong_stage_pipeline) {
      const StmtSRef& block_sref = std::get<0>(record);
      bool expected = std::get<1>(record);
      bool actual = std::get<2>(record);
      const auto* block = block_sref->StmtAs<SBlockObj>();
      TVM_FFI_ICHECK(block);
      os << " (" << block->name_hint << ", expected=" << expected << ", actual=" << actual << ")";
    }
    os << std::endl;
  }
  TVM_FFI_THROW(InternalError) << "Schedule verification failed. The IR is:\n"
                               << self->mod << "\nThe errors are:\n"
                               << os.str();
  throw;
}

}  // namespace s_tir
}  // namespace tvm
