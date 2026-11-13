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
 * \file tvm/s_tir/stmt_functor.h
 * \brief Native traversal and statement dispatch for schedulable TIR.
 */
#ifndef TVM_S_TIR_STMT_FUNCTOR_H_
#define TVM_S_TIR_STMT_FUNCTOR_H_

#include <tvm/base.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm {
namespace s_tir {

/*!
 * \brief Extend TIRX statement dispatch with schedulable blocks.
 * \tparam FType The statement signature, using the native Dispatch API.
 */
template <typename FType>
class StmtFunctor;

template <typename R, typename... Args>
class StmtFunctor<R(const Stmt&, Args...)> : public tirx::StmtFunctor<R(const Stmt&, Args...)> {
  using Parent = tirx::StmtFunctor<R(const Stmt&, Args...)>;

 public:
  TVM_DEFINE_OBJECT_FUNCTOR_DEFAULT_CONSTRUCTOR(StmtFunctor, Parent)
  using Parent::Dispatch_;

  virtual R Dispatch_(const SBlockObj* op, Args... args) {
    return this->DispatchDefault_(op, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const SBlockRealizeObj* op, Args... args) {
    return this->DispatchDefault_(op, std::forward<Args>(args)...);
  }

 protected:
  using VTable = typename Parent::VTable;
  explicit StmtFunctor(const VTable* vtable) : Parent(vtable) {}
  static void InitVTable(VTable* vtable) {
    Parent::InitVTable(vtable);
    Parent::template SetDispatch<StmtFunctor, SBlockObj>(vtable);
    Parent::template SetDispatch<StmtFunctor, SBlockRealizeObj>(vtable);
  }
};

/*!
 * \brief Extend native TIRX traversal with schedulable block semantics.
 *
 * Block iterator binders and policy annotations are not expression uses. Tensor
 * definitions precede their regions; ordinary statements reuse TIRX hooks.
 * Structural traversal remains available independently with its full field walk.
 * Generic TIRX visitors traverse these nodes structurally, including whole
 * iterators and annotations. Both paths visit allocation and match-tensor
 * definitions before region uses. Producer placement annotations are expression uses.
 * Only this S-TIR subclass supplies native block hooks; the core TIRX table does not
 * register dialect nodes.
 */
class TVM_DLL StmtExprVisitor : public tirx::StmtExprVisitor {
 public:
  TVM_DEFINE_OBJECT_FUNCTOR_DEFAULT_CONSTRUCTOR(StmtExprVisitor, tirx::StmtExprVisitor)
  using tirx::StmtExprVisitor::Visit;
  using tirx::StmtExprVisitor::Visit_;

  virtual ffi::Optional<VisitInterrupt> Visit_(const SBlockObj* op);
  virtual ffi::Optional<VisitInterrupt> Visit_(const SBlockRealizeObj* op);

  // Shared native traversal for specialized TIRX helpers extended by S-TIR.
  static ffi::Optional<VisitInterrupt> VisitBlock(ffi::StructuralVisitorObj* visitor,
                                                  const SBlockObj* op);
  static ffi::Optional<VisitInterrupt> VisitBlockRealize(ffi::StructuralVisitorObj* visitor,
                                                         const SBlockRealizeObj* op);

 protected:
  explicit StmtExprVisitor(const VTable* vtable) : tirx::StmtExprVisitor(vtable) {}
  static void InitVTable(VTable* vtable);
};

/*!
 * \brief Extend native TIRX mutation while preserving block iterator binders.
 *
 * Reuses inherited remapping and ownership checks. Block policy annotations are left
 * intact; producer placement annotations follow tensor remapping. Structural mutation
 * separately provides the full field rewrite.
 * Generic TIRX mutators instead use the full structural rewrite, including
 * iterator definitions and annotations. Both paths establish allocation and
 * match-tensor remaps before visiting region uses and preserve copy-on-write.
 */
class TVM_DLL StmtExprMutator : public tirx::StmtExprMutator {
 public:
  TVM_DEFINE_OBJECT_FUNCTOR_DEFAULT_CONSTRUCTOR(StmtExprMutator, tirx::StmtExprMutator)
  using tirx::StmtExprMutator::Mutate;
  using tirx::StmtExprMutator::Mutate_;

  virtual UnchangedOr<Stmt> Mutate_(const SBlockObj* op, InplaceMode inplace_mode);
  virtual UnchangedOr<Stmt> Mutate_(const SBlockRealizeObj* op, InplaceMode inplace_mode);

  // Share block ownership and binder rules with specialized TIRX helpers.
  static UnchangedOr<Stmt> MutateBlock(ffi::StructuralMutatorObj* mutator, const SBlockObj* op,
                                       InplaceMode inplace_mode);
  static UnchangedOr<Stmt> MutateBlockRealize(ffi::StructuralMutatorObj* mutator,
                                              const SBlockRealizeObj* op, InplaceMode inplace_mode);

 protected:
  explicit StmtExprMutator(const VTable* vtable) : tirx::StmtExprMutator(vtable) {}
  static void InitVTable(VTable* vtable);
};

}  // namespace s_tir
}  // namespace tvm
#endif  // TVM_S_TIR_STMT_FUNCTOR_H_
