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
 * \file tvm/s_tir/sblock_scope.h
 * \brief Definition of two pillar data structure for TensorIR scheduling: StmtSRef, SBlockScope.
 * \sa StmtSRefObj
 * \sa SBlockScopeObj
 */
#ifndef TVM_S_TIR_SBLOCK_SCOPE_H_
#define TVM_S_TIR_SBLOCK_SCOPE_H_

#include <tvm/base.h>
#include <tvm/ir/module.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>

#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace tirx {

/*!
 * \brief An object that refers to schedulable elements (block/for-loop) in TensorIR, aka "sref".
 *
 * Glossary
 * - s_tir::SBlock sref: A StmtSRef that points to a TensorIR s_tir::SBlock.
 * - Loop sref: A StmtSRef that points to a TensorIR for loop.
 * - Parent sref: The parent reference of an sref is the block or loop reference to the closest
 schedulable statement. We define closest to be the nearest schedulable statement of an ancestor in
 the AST.
 * schedulable statement of its ancestors on the TensorIR AST.
 * - Root sref: Sref to the root block. Every sref has exactly one parent sref except for root sref.
 * - Sref tree: The parent-children-relationship of srefs that forms a tree, uniquely determined by
 * the TensorIR AST.
 */
class StmtSRefObj : public ffi::Object {
 public:
  /*!
   * \brief The block or `for` stmt the object refers to
   * \note Non-owned reference (raw pointer) is used here, so that we can perform copy-on-write
   * optimization on statements when possible. The strong reference is held in the ScheduleState.
   */
  const StmtObj* stmt;
  /*! \brief The parent sref. */
  StmtSRefObj* parent;
  /*!
   * \brief If the statement the sref points to is an element of a SeqStmt in the AST,
   * then `seq_index` is set to its index; otherwise `seq_index` is -1
   */
  int64_t seq_index;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<StmtSRefObj>().def_ro("seq_index", &StmtSRefObj::seq_index);
  }

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.StmtSRef", StmtSRefObj, ffi::Object);

  /*! \brief Reset the object inplace to the invalid state */
  void Reset() {
    this->stmt = nullptr;
    this->parent = nullptr;
    this->seq_index = -1;
  }

  /*!
   * \brief Get the referenced statement with proper type checking.
   * It serves the same purpose as `ffi::ObjectRef::as`, but does not acquire strong reference to
   * `stmt`
   * \tparam StmtType The type that `this->stmt` to be downcasted to. Presumably
   * tvm::s_tir::SBlockObj or tvm::ForObj
   * \return nullptr if type check fails, otherwise the casted result for `this->stmt`
   */
  template <typename StmtType>
  const StmtType* StmtAs() const {
    if (stmt != nullptr && stmt->IsInstance<StmtType>()) {
      return static_cast<const StmtType*>(stmt);
    } else {
      return nullptr;
    }
  }
};

/*!
 * \brief Managed reference to StmtSRefObj
 * \sa StmtSRefObj
 */
class StmtSRef : public ffi::ObjectRef {
 public:
  /*!
   * \brief The constructor
   * \param stmt The corresponding stmt node, can be either block or for loop.
   * \param parent The parent sref.
   * \param seq_index The location in an array if the parent of the stmt contains multiple children.
   * -1 if the parent does not contain multiple children.
   */
  TVM_DLL explicit StmtSRef(const StmtObj* stmt, StmtSRefObj* parent, int64_t seq_index);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StmtSRef, ffi::ObjectRef, StmtSRefObj);

 public:
  /*!
   * \return A special StmtSRef, which doesn't point to any stmt in the AST,
   * only serving as a "mark" to hint compute-at to do the work of compute-inline
   * \note This is only as a faked loop sref for compute-at and reverse-compute-at,
   * i.e.
   *
   * compute-at(block, loop_sref):
   *   compute-inline(block)                if loop_sref.same_as(InlineMark())
   *   no-op                                if loop_sref.same_as(RootMark())
   *   compute-at-impl(block, loop_sref)    otherwise
   */
  TVM_DLL static StmtSRef InlineMark();
  /*!
   * \return A special StmtSRef, which doesn't point to any stmt in the AST,
   * only serving as a "mark" to hint compute-at to do nothing
   * \note This is only as a faked loop sref for compute-at and reverse-compute-at,
   * i.e.
   *
   * compute-at(block, loop_sref):
   *   compute-inline(block)                if loop_sref.same_as(InlineMark())
   *   no-op                                if loop_sref.same_as(RootMark())
   *   compute-at-impl(block, loop_sref)    otherwise
   */
  TVM_DLL static StmtSRef RootMark();
};

class SRefTreeCreator : public s_tir::StmtExprVisitor {
 public:
  using s_tir::StmtExprVisitor::Visit_;

  ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
    if (value.as<ExprObj>()) return std::nullopt;
    return s_tir::StmtExprVisitor::Visit(value);
  }

  /*!
   * \brief StmtSRef Tree Creator
   * \param mod The module being scheduled.
   * \param include_loops Ignore ForNodes if this value is false
   */
  static std::unordered_map<const StmtObj*, StmtSRef> Create(IRModule mod,
                                                             bool include_loops = true) {
    auto creator = ffi::make_object<SRefTreeCreator>(include_loops);
    for (const auto& kv : mod->functions) {
      const BaseFunc& base_func = kv.second;
      if (auto opt = base_func.as<Function>()) {
        auto func = opt.value();
        creator->Visit(func->body);
      }
    }
    return std::move(creator->stmt2ref_);
  }

  explicit SRefTreeCreator(bool include_loops) : include_loops_(include_loops) {}

 private:
  /*!
   * \brief Add a new statement to the stack, which becomes the current scope
   * \param stmt A for-loop statement or a block statement
   */
  void PushSRef(const StmtObj* stmt);

  /*! \brief Pop the top of the scope and record it in stmt2ref map */
  void PopAndRecordSRef();

  ffi::Optional<VisitInterrupt> Visit_(const ForObj* loop) final;

  ffi::Optional<VisitInterrupt> Visit_(const s_tir::SBlockRealizeObj* realize) final;

  ffi::Optional<VisitInterrupt> Visit_(const SeqStmtObj* seq_stmt) final;

  bool include_loops_;
  /*! \brief The result ScheduleStateObj */
  std::unordered_map<const StmtObj*, StmtSRef> stmt2ref_;
  /*! \brief The stack frame used to indicate the current scope */
  std::vector<StmtSRef> srefs_;
};

/*!
 * \brief Type of dependency. Right now we have 4 types of dependencies
 * 1) Read-after-write (kRAW)
 * 2) Write-after-write (kWAW)
 * 3) Write-after-read (kWAR)
 * 4) Opaque dependency (kOpaque)
 */
enum class DepKind : int32_t {
  kRAW = 0,
  kWAW = 1,
  kWAR = 2,
  kOpaque = 3,
};

/*!
 * \brief A tuple (src, dst, kind) representing certain types of dependency.
 * For example, (A, B, kRAW) means block B depends on block A, and the dependency kind is
 * read-after-write, which means block B reads the result written by block A.
 */
class DependencyObj : public ffi::Object {
 public:
  explicit DependencyObj(ffi::UnsafeInit tag) : src(tag), dst(tag) {}

  explicit DependencyObj(StmtSRef src, StmtSRef dst, DepKind kind)
      : src(std::move(src)), dst(std::move(dst)), kind(kind) {}
  /*! \brief The source of the dependency relation */
  StmtSRef src;
  /*! \brief The destination of the dependency relation */
  StmtSRef dst;
  /*! \brief The dependency kind */
  DepKind kind;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DependencyObj>()
        .def_ro("src", &DependencyObj::src)
        .def_ro("dst", &DependencyObj::dst)
        .def_ro("kind", &DependencyObj::kind);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.Dependency", DependencyObj, ffi::Object);
};

/*!
 * \brief Managed reference to DependencyObj
 * \sa DependencyObj
 */
class Dependency : public ffi::ObjectRef {
 public:
  /*! \brief Constructor */
  TVM_DLL explicit Dependency(StmtSRef src, StmtSRef dst, DepKind kind);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Dependency, ffi::ObjectRef, DependencyObj);
};

/*!
 * \brief An object with 1-to-1 correspondence with each block reference in the sref tree.
 * This data structure is used to track the producer-consumer dependencies between blocks.
 * For example even leaf nodes have a scope node, even though they have no dependencies.
 *
 * Glossary:
 * - s_tir::SBlock scope: A contiguous subtree of the sref tree, rooted at each s_tir::SBlock sref,
 * whose components are:
 *   - scope root: a block sref
 *   - internal srefs: loop srefs
 *   - scope leaves: block srefs
 * - Child block: The scope leaf blocks under the scope root or a specific internal sref
 */
class SBlockScopeObj : public ffi::Object {
 public:
  /*!
   * \brief Lookup table for the `src` of dependencies
   * \note We intentionally didn't use tvm::Map as the data structure, because we need the values
   * inside to be mutable so that they could be further maintained properly during transformations.
   */
  std::unordered_map<StmtSRef, ffi::Array<Dependency>, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      src2deps;
  /*! \brief Lookup table for the `dst` of dependencies */
  std::unordered_map<StmtSRef, ffi::Array<Dependency>, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      dst2deps;
  /*! \brief The mapping from the tensor to the blocks who write it */
  std::unordered_map<TensorVar, ffi::Array<StmtSRef>, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      tensor_writers;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SBlockScopeObj>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.SBlockScope", SBlockScopeObj, ffi::Object);

 public:
  /******** Dependency ********/
  /*!
   * \brief Get all dependencies whose `src` equals `src`
   * \param src The queried block
   * \return The dependencies
   */
  TVM_DLL ffi::Array<Dependency> GetDepsBySrc(const StmtSRef& src) const;
  /*!
   * \brief Get all dependencies whose `dst` equals `dst`
   * \param dst The queried block
   * \return The dependencies
   */
  TVM_DLL ffi::Array<Dependency> GetDepsByDst(const StmtSRef& dst) const;
};

/*!
 * \brief Managed reference to SBlockScopeObj
 * \sa SBlockScopeObj
 */
class SBlockScope : public ffi::ObjectRef {
 public:
  /*!
   * \brief Constructor from ffi::ObjectPtr<SBlockScopeObj>.
   * \param data The object pointer.
   */
  explicit SBlockScope(ffi::ObjectPtr<SBlockScopeObj> data) : ffi::ObjectRef(data) {
    TVM_FFI_ICHECK(data != nullptr);
  }
  /*! \brief The constructor creating an empty block scope with on dependency information */
  TVM_DLL SBlockScope();
  /*!
   * \brief Create the object with the specific leaf blocks, and compute the dependency information
   * between the leaf blocks.
   * \param child_block_srefs The srefs to the leaf blocks
   * \note We assume the leaf blocks are given in pre-DFS order
   */
  TVM_DLL explicit SBlockScope(const ffi::Array<StmtSRef>& child_block_srefs);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SBlockScope, ffi::ObjectRef, SBlockScopeObj);
};

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_S_TIR_SBLOCK_SCOPE_H_
