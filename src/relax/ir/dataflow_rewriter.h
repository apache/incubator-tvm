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
 * \file src/relax/dataflow_rewriter.h
 * \brief Pattern match/rewriters for Relax
 */
#ifndef TVM_RELAX_IR_DATAFLOW_REWRITER_H_
#define TVM_RELAX_IR_DATAFLOW_REWRITER_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/transform.h>
#include <tvm/relax/expr.h>

#include <optional>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "dataflow_matcher.h"

namespace tvm {
namespace relax {

struct RewriteSpec {
  ffi::Map<Var, Expr> variable_rewrites;
  ffi::Map<GlobalVar, BaseFunc> new_subroutines;

  explicit operator bool() const { return variable_rewrites.size(); }

  void Append(RewriteSpec other);
};

class PatternMatchingRewriterObj : public tvm::transform::PassObj {
 public:
  virtual RewriteSpec RewriteBindings(const ffi::Array<Binding>& bindings) const {
    return RewriteSpec();
  }

  static void RegisterReflection() {
    // PatternMatchingRewriterObj has no fields to register
  }

  IRModule operator()(IRModule mod, const tvm::transform::PassContext& pass_ctx) const override;
  tvm::transform::PassInfo Info() const override;
  TVM_FFI_DECLARE_OBJECT_INFO("relax.dpl.PatternMatchingRewriter", PatternMatchingRewriterObj,
                              PassObj);
};

class PatternMatchingRewriter : public tvm::transform::Pass {
 public:
  static PatternMatchingRewriter FromPattern(
      DFPattern pattern,
      ffi::TypedFunction<ffi::Optional<Expr>(Expr, ffi::Map<DFPattern, Expr>)> func,
      ffi::Optional<ffi::Array<DFPattern>> additional_bindings = std::nullopt,
      ffi::Map<GlobalVar, BaseFunc> new_subroutines = {});

  static PatternMatchingRewriter FromModule(IRModule mod);

  Expr operator()(Expr expr);
  using Pass::operator();

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PatternMatchingRewriter, Pass,
                                                PatternMatchingRewriterObj);
};

class ExprPatternRewriterObj : public PatternMatchingRewriterObj {
 public:
  explicit ExprPatternRewriterObj(DFPattern value) : pattern(std::move(value)) {}
  explicit ExprPatternRewriterObj(ffi::UnsafeInit) : pattern(ffi::UnsafeInit{}) {}

  DFPattern pattern;
  ffi::TypedFunction<ffi::Optional<Expr>(Expr, ffi::Map<DFPattern, Expr>)> func;
  ffi::Optional<ffi::Array<DFPattern>> additional_bindings;
  ffi::Map<GlobalVar, BaseFunc> new_subroutines;

  RewriteSpec RewriteBindings(const ffi::Array<Binding>& bindings) const final;

  ffi::Optional<Expr> RewriteExpr(const Expr& expr, const ffi::Map<Var, Expr>& bindings) const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ExprPatternRewriterObj>()
        .def_ro("pattern", &ExprPatternRewriterObj::pattern)
        .def_ro("func", &ExprPatternRewriterObj::func);
  }
  TVM_FFI_DECLARE_OBJECT_INFO("relax.dpl.ExprPatternRewriter", ExprPatternRewriterObj,
                              PatternMatchingRewriterObj);
};

class ExprPatternRewriter : public PatternMatchingRewriter {
 public:
  ExprPatternRewriter(DFPattern pattern,
                      ffi::TypedFunction<ffi::Optional<Expr>(Expr, ffi::Map<DFPattern, Expr>)> func,
                      ffi::Optional<ffi::Array<DFPattern>> additional_bindings = std::nullopt,
                      ffi::Map<GlobalVar, BaseFunc> new_subroutines = {});

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ExprPatternRewriter, PatternMatchingRewriter,
                                                ExprPatternRewriterObj);
};

class OrRewriterObj : public PatternMatchingRewriterObj {
 public:
  OrRewriterObj(PatternMatchingRewriter lhs, PatternMatchingRewriter rhs)
      : lhs(std::move(lhs)), rhs(std::move(rhs)) {}
  explicit OrRewriterObj(ffi::UnsafeInit) : lhs(ffi::UnsafeInit{}), rhs(ffi::UnsafeInit{}) {}
  PatternMatchingRewriter lhs;
  PatternMatchingRewriter rhs;

  RewriteSpec RewriteBindings(const ffi::Array<Binding>& bindings) const override;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<OrRewriterObj>()
        .def_ro("lhs", &OrRewriterObj::lhs)
        .def_ro("rhs", &OrRewriterObj::rhs);
  }
  TVM_FFI_DECLARE_OBJECT_INFO("relax.dpl.OrRewriter", OrRewriterObj, PatternMatchingRewriterObj);
};

class OrRewriter : public PatternMatchingRewriter {
 public:
  OrRewriter(PatternMatchingRewriter lhs, PatternMatchingRewriter rhs);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(OrRewriter, PatternMatchingRewriter, OrRewriterObj);
};

class TupleRewriterObj : public PatternMatchingRewriterObj {
 public:
  ffi::Array<DFPattern> patterns;
  ffi::TypedFunction<ffi::Optional<Expr>(Expr, ffi::Map<DFPattern, Expr>)> func;
  ffi::Optional<ffi::Array<DFPattern>> additional_bindings;
  ffi::Map<GlobalVar, BaseFunc> new_subroutines;

  RewriteSpec RewriteBindings(const ffi::Array<Binding>& bindings) const override;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TupleRewriterObj>()
        .def_ro("patterns", &TupleRewriterObj::patterns)
        .def_ro("func", &TupleRewriterObj::func);
  }
  TVM_FFI_DECLARE_OBJECT_INFO("relax.dpl.TupleRewriter", TupleRewriterObj,
                              PatternMatchingRewriterObj);

 private:
  struct VarInfo {
    Var var;
    Expr expr;
    ffi::Array<ffi::Optional<ffi::Map<DFPattern, Expr>>> matches;
    std::unordered_set<Var> downstream_usage;
    bool used = false;
  };

  ffi::Map<Var, Expr> GenerateVariableRewrites(const ffi::Array<Binding>& bindings) const;

  std::optional<std::vector<Expr>> TryMatchByBindingIndex(const std::vector<VarInfo>& info_vec,
                                                          const std::vector<size_t>& indices) const;
};

class TupleRewriter : public PatternMatchingRewriter {
 public:
  TupleRewriter(ffi::Array<DFPattern> patterns,
                ffi::TypedFunction<ffi::Optional<Expr>(Expr, ffi::Map<DFPattern, Expr>)> func,
                ffi::Optional<ffi::Array<DFPattern>> additional_bindings = std::nullopt,
                ffi::Map<GlobalVar, BaseFunc> new_subroutines = {});

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TupleRewriter, PatternMatchingRewriter,
                                                TupleRewriterObj);
};

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_IR_DATAFLOW_REWRITER_H_
