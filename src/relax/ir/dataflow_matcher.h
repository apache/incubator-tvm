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
 * \file src/tvm/relax/dataflow_matcher.h
 * \brief The auxiliary data structure for dataflow matcher.
 */
#ifndef TVM_RELAX_IR_DATAFLOW_MATCHER_H_
#define TVM_RELAX_IR_DATAFLOW_MATCHER_H_

#include <tvm/ir/prim/op.h>
#include <tvm/relax/dataflow_matcher.h>
#include <tvm/relax/dataflow_pattern.h>
#include <tvm/relax/dataflow_pattern_functor.h>
#include <tvm/sym/analyzer.h>

#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace relax {

class DFPatternMatcher : public DFPatternFunctor<bool(const DFPattern&, const Expr&)> {
 public:
  using var2val_t = ffi::Map<Var, Expr>;

  explicit DFPatternMatcher() : DFPatternMatcher(var2val_t{}) {}
  explicit DFPatternMatcher(var2val_t var2val) : var2val_(std::move(var2val)) {}
  bool Match(const DFPattern& pattern, const Expr& expr);
  ffi::Map<DFPattern, Expr> GetMemo() { return memo_; }

  /* \brief Unwrap trivial expressions/bindings */
  static Expr UnwrapBindings(Expr expr, const ffi::Map<Var, Expr>& bindings);

 protected:
  bool VisitDFPattern(const DFPattern& pattern, const Expr& expr) override;
  bool VisitDFPattern_(const OrPatternObj* op, const Expr& expr) override;
  bool VisitDFPattern_(const AndPatternObj* op, const Expr& expr) override;
  bool VisitDFPattern_(const NotPatternObj* op, const Expr& expr) override;
  bool VisitDFPattern_(const AttrPatternObj* op, const Expr& expr) override;
  bool VisitDFPattern_(const CallPatternObj* op, const Expr& expr) override;
  bool VisitDFPattern_(const ConstantPatternObj* op, const Expr& expr) override;
  bool VisitDFPattern_(const DataTypePatternObj* op, const Expr& expr) override;
  bool VisitDFPattern_(const ExprPatternObj* op, const Expr& expr) override;
  bool VisitDFPattern_(const FunctionPatternObj* op, const Expr& expr) override;
  bool VisitDFPattern_(const ShapePatternObj* op, const Expr& expr) override;
  bool VisitDFPattern_(const TupleGetItemPatternObj* op, const Expr& expr) override;
  bool VisitDFPattern_(const TuplePatternObj* op, const Expr& expr) override;
  bool VisitDFPattern_(const TypePatternObj* op, const Expr& expr) override;
  bool VisitDFPattern_(const WildcardPatternObj* op, const Expr& expr) override;
  bool VisitDFPattern_(const VarPatternObj* op, const Expr& expr) override;

  bool VisitDFPattern_(const DataflowVarPatternObj* op, const Expr& expr) override;
  bool VisitDFPattern_(const GlobalVarPatternObj* op, const Expr& expr) override;
  bool VisitDFPattern_(const ExternFuncPatternObj* op, const Expr& expr) override;
  bool VisitDFPattern_(const PrimArrPatternObj* op, const Expr& expr) override;
  bool VisitDFPattern_(const UnorderedTuplePatternObj* op, const Expr& expr) override;

  void ClearMap(size_t watermark);
  bool TryUnorderedMatch(size_t idx, const tvm::ffi::Array<DFPattern> patterns,
                         const tvm::ffi::Array<Expr> fields, std::vector<int8_t>& match_cache,
                         std::vector<bool>& matched);

  std::unordered_map<DFPattern, Expr, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> memo_;
  var2val_t var2val_;
  std::vector<DFPattern> matched_nodes_;
  PrimExpr symbolic_expr_condition_{IntImm::Bool(true)};
  sym::Analyzer analyzer_;
  bool memoize_ = true;
};

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_IR_DATAFLOW_MATCHER_H_
