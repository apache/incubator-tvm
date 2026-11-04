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
 * \file tvm/s_tir/stmt_functor_with_constraint_context.h
 * \brief Statement and expression functors with a constraint context.
 */

#ifndef TVM_S_TIR_STMT_FUNCTOR_WITH_CONSTRAINT_CONTEXT_H_
#define TVM_S_TIR_STMT_FUNCTOR_WITH_CONSTRAINT_CONTEXT_H_

#include <tvm/base.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/tirx/stmt_functor_with_constraint_context.h>

namespace tvm {
namespace s_tir {

class TVM_DLL StmtExprVisitorWithConstraintContext
    : public tirx::StmtExprVisitorWithConstraintContext {
 public:
  using Parent = tirx::StmtExprVisitorWithConstraintContext;
  using Parent::Visit;
  using Parent::Visit_;
  PrimExpr Simplify(const PrimExpr& expr) { return GetSyncedAnalyzer()->Simplify(expr); }
  TVM_DEFINE_OBJECT_FUNCTOR_DEFAULT_CONSTRUCTOR(StmtExprVisitorWithConstraintContext, Parent)
  virtual ffi::Optional<VisitInterrupt> Visit_(const SBlockObj* op);
  virtual ffi::Optional<VisitInterrupt> Visit_(const SBlockRealizeObj* op) {
    return s_tir::StmtExprVisitor::VisitBlockRealize(this, op);
  }

 protected:
  sym::AnalyzerObj* GetSyncedAnalyzer() {
    analyzer_->SyncConstraintContext(constraints);
    return analyzer_.get();
  }
  sym::Analyzer analyzer_;

  static void InitVTable(VTable* vtable) {
    Parent::InitVTable(vtable);
    SetDispatch<StmtExprVisitorWithConstraintContext, SBlockObj>(vtable);
    SetDispatch<StmtExprVisitorWithConstraintContext, SBlockRealizeObj>(vtable);
  }
};

class TVM_DLL StmtExprMutatorWithConstraintContext
    : public tirx::StmtExprMutatorWithConstraintContext {
 public:
  using Parent = tirx::StmtExprMutatorWithConstraintContext;
  using Parent::Mutate;
  using Parent::Mutate_;
  explicit StmtExprMutatorWithConstraintContext(const sym::ConstraintContext& context)
      : Parent(GlobalVTable()) {
    constraints = context->Snapshot();
  }
  virtual UnchangedOr<Stmt> Mutate_(const SBlockObj* op, InplaceMode inplace_mode);
  virtual UnchangedOr<Stmt> Mutate_(const SBlockRealizeObj* op, InplaceMode inplace_mode) {
    return s_tir::StmtExprMutator::MutateBlockRealize(this, op, inplace_mode);
  }

 protected:
  static void InitVTable(VTable* vtable) {
    Parent::InitVTable(vtable);
    SetDispatch<StmtExprMutatorWithConstraintContext, SBlockObj>(vtable);
    SetDispatch<StmtExprMutatorWithConstraintContext, SBlockRealizeObj>(vtable);
  }
  static const VTable* GlobalVTable() {
    static const VTable table = [] {
      VTable table;
      InitVTable(&table);
      table.Finalize();
      return table;
    }();
    return &table;
  }
};

}  // namespace s_tir
}  // namespace tvm
#endif  // TVM_S_TIR_STMT_FUNCTOR_WITH_CONSTRAINT_CONTEXT_H_
