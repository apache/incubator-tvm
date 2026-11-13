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
 * \file tvm/s_tir/schedule/state.h
 * \brief This file defines ScheduleState, the core data structure of TensorIR scheduling.
 */
#ifndef TVM_S_TIR_SCHEDULE_STATE_H_
#define TVM_S_TIR_SCHEDULE_STATE_H_

#include <tvm/base.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/s_tir/sblock_scope.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/function.h>

#include <unordered_map>
#include <utility>

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

/*!
 * \brief The information about a TensorIR block, it contains two categories of information
 * 1) Info on the block scope rooted at a specific block, including dependency tracking,
 * flags indicating if the scope is a stage pipeline, etc.
 * 2) Info on the block itself, including if the block has a quasi-affine binding, if the regions it
 * reads are completely covered by their producers, etc.
 */
struct SBlockInfo {
  /*! \brief Property of a block scope rooted at the block, storing dependencies in the scope */
  SBlockScope scope{ffi::UnsafeInit()};
  // The properties below are information about the current block realization under its parent scope
  /*! \brief Property of a block, indicating the block realization binding is quasi-affine */
  bool affine_binding{false};
  /*!
   * \brief Property of a block, indicating each of the block's read regions is fully
   * produced by its producers
   */
  bool region_cover{false};
  /*!
   * \brief This property indicates that the block scope (rooted at its corresponding block) is
   * equivalent to of a stage pipeline. Under the following conditions:
   *
   * 1) The region cover property holds for every of its child blocks
   * 2) No write-after-read dependency or opaque dependency, only read-after-write and
   * write-after-write are allowed
   * 3) All the statements in the scope are schedulable statements, i.e. SBlock and For
   */
  bool stage_pipeline{false};

  SBlockInfo() = default;

  explicit SBlockInfo(SBlockScope scope, bool affine_binding = false, bool region_cover = false,
                      bool stage_pipeline = false)
      : scope(std::move(scope)),         //
        affine_binding(affine_binding),  //
        region_cover(region_cover),
        stage_pipeline(stage_pipeline) {}
};

/*!
 * \brief The bitmask of the debug flag in the ScheduleStateObj.
 * \sa ScheduleStateObj
 */
enum ScheduleDebugMask : uint32_t {
  /*! \brief Verify the correctness of the sref tree */
  kVerifySRefTree = 1,
  /*! \brief Verify the correctness of affine_binding, region_cover and stage_pipeline */
  kVerifyCachedFlags = 2,
};

/*!
 * \brief The state of scheduling, which exposes a `Replace` method as
 * the primary interface for all the scheduling primitives to manipulate the TensorIR.
 *
 * The data structure contains the following information
 * 1) The AST being scheduled (mod)
 * 2) The sref tree of schedulable statements (indicated by the srefs)
 * 3) The dependency information of each block scope (block_info)
 * 4) A reverse mapping from the AST nodes to that in the sref tree (stmt2ref)
 * 5) A debug flag, if set, extra checking is enabled (debug_mask)
 * 6) A check flag, if set, enable prequisite check for schedule primitives (enable_check)
 */
class ScheduleStateObj : public ffi::Object {
 public:
  /*! \brief The AST of the module being scheduled */
  IRModule mod;
  /*!
   * \brief Mapping from a block sref to its correpsonding SBlockInfo,
   * tracking the dependency inside the block scope,
   * and storing necessary information flags for scheduling
   */
  std::unordered_map<StmtSRef, SBlockInfo, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> block_info;
  /*! \brief The reverse mapping from block/for-loop to their corresponding srefs */
  std::unordered_map<const StmtObj*, StmtSRef> stmt2ref;
  /*!
   * \brief Do extra correctness checking after the class creation
   * and each time after calling the Replace method.
   * \sa ScheduleDebugMask
   */
  int debug_mask;
  /*!
   * \brief Whether to enable prequisite checks for schedule primitives.
   */
  bool enable_check;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ScheduleStateObj>()
        .def_ro("mod", &ScheduleStateObj::mod)
        .def_ro("debug_mask", &ScheduleStateObj::debug_mask)
        .def_ro("enable_check", &ScheduleStateObj::enable_check);
  }

  /*!
   * \brief Replace the part of the AST, as being pointed to by `src_sref`,
   * with a specific statement `tgt_stmt`, and maintain the sref tree accordingly.
   * Replace will try to perform copy on write as much as possible when the ScheduleState holds
   * the only copy to the IRModule and IR nodes.
   *
   * Only 3 types of replacements are allowed: from `src_sref->stmt` to `tgt_stmt`.
   * 1) SBlock -> SBlock
   * 2) Loop -> Loop
   * 3) Loop -> SBlockRealize
   *
   * \param src_sref The sref to the statement to be replaced
   * \param tgt_stmt The statement to be replaced in
   * \param block_sref_reuse Maps an old block (to be replaced in the subtree under
   * `src_sref->stmt`) to a new block (replaced to, in the subtree under `tgt_stmt`), and enforces
   * reuse of srefs between them (rather than create new srefs) i.e. after being replaced, the sref
   * that points to the old block will point to the new one
   * \note The reuse of loop srefs are detected automatically according to the reuse of loop vars.
   */
  TVM_DLL void Replace(const tirx::StmtSRef& src_sref, const Stmt& tgt_stmt,
                       const ffi::Map<SBlock, SBlock>& block_sref_reuse);
  /*!
   * \brief Trigger the verification according to the `debug_mask` bitmask.
   * 1) If the bitmask `kVerifySRefTree` is on, verify the correctness of the sref tree.
   * 2) If the bitmask `kVerifyCachedFlags` is on, verify the correctness of `affine_binding`,
   * `region_cover` and `stage_pipeline`
   */
  TVM_DLL void DebugVerify() const;

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.ScheduleState", ScheduleStateObj, ffi::Object);

  /******** Property of blocks ********/
  /*! \brief Returns the SBlockInfo correpsonding to the block sref */
  TVM_DLL SBlockInfo GetSBlockInfo(const StmtSRef& block_sref) const;
  /*!
   * \brief Recalculate the SBlockInfo recursively under stmt.
   * If stmt is a SBlock itself, we will not reset its affine binding flag unless it doesn't
   * have block vars, since the affine flag depends on the outer scope of stmt.
   */
  TVM_DLL void UpdateScopeSBlockInfo(const Stmt& stmt);
  /*!
   * \brief Get the SBlockScope correpsonding to the sref of scope root block
   * \param scope_root The block sref to be retrieved
   * \return The corresponding SBlockScope
   */
  SBlockScope GetSBlockScope(const StmtSRef& scope_root) const {
    return GetSBlockInfo(scope_root).scope;
  }
  /*!
   * \brief Check a cached flag indicating if the specific block has quasi-affine bindings
   * \param block_sref The block sref to be checked
   * \return A boolean flag indicating if the block has quasi-affine bindings
   */
  bool IsAffineBlockBinding(const StmtSRef& block_sref) const {
    return GetSBlockInfo(block_sref).affine_binding;
  }
  /*!
   * \brief Check a cached flag indicating if each of the specific consumer block's read region
   * is fully produced by its producers
   * \param consumer_block_sref The specific consumer block
   * \return A boolean flag indicating if the block has quasi-affine bindings
   */
  bool IsRegionCoveredConsumer(const StmtSRef& consumer_block_sref) const {
    return GetSBlockInfo(consumer_block_sref).region_cover;
  }
  /*!
   * \brief Check a cached flag indicating if a block scope is an equivalence of a stage pipeline
   * \param scope_root The block sref to be retrieved
   * \return The corresponding SBlockScope
   */
  bool IsStagePipeline(const StmtSRef& scope_root) const {
    return GetSBlockInfo(scope_root).stage_pipeline;
  }
};

/*!
 * \brief Managed reference to ScheduleStateObj
 * \sa ScheduleStateObj
 */
class ScheduleState : public ffi::ObjectRef {
 public:
  explicit ScheduleState(ffi::UnsafeInit tag, ffi::ObjectPtr<ScheduleStateObj> data)
      : ffi::ObjectRef(tag) {
    data_ = std::move(data);
  }

  /*!
   * \brief Construct a schedule state from an IRModule
   * \param mod The IRModule to be scheduled
   * \param debug_mask Do extra correctness checking after the class creation
   * and each time after calling the Replace method.
   * \param enable_check Whether enables prerequisite checks for schedule primitives.
   */
  TVM_DLL explicit ScheduleState(IRModule mod, int debug_mask = 0, bool enable_check = true);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ScheduleState, ffi::ObjectRef, ScheduleStateObj);
};

}  // namespace s_tir
}  // namespace tvm

#endif  // TVM_S_TIR_SCHEDULE_STATE_H_
