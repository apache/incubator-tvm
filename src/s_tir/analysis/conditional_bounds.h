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
 * \file s_tir/analysis/conditional_bounds.h
 * \brief Scoped conditional bounds for S-TIR tensor analysis.
 */
#ifndef TVM_S_TIR_ANALYSIS_CONDITIONAL_BOUNDS_H_
#define TVM_S_TIR_ANALYSIS_CONDITIONAL_BOUNDS_H_

#include <tvm/ir/prim/expr.h>
#include <tvm/ir/with_context.h>
#include <tvm/sym/int_set.h>

#include <vector>

namespace tvm {
namespace s_tir {

/*!
 * \brief Context helper to update domain map within conditional scope.
 * Assume the condition is `0 <= i && i < 9` and domain of i is [0, 20], Then
 * `With<ConditionalBoundsContext> ctx(condition, &relax_map, &hint_map, &constraints)`
 * step into scope where dom_map[i] is [0, 8]; and
 * `With<ConditionalBoundsContext> ctx(!condition, &relax_map, &hint_map, &constraints)`
 * step into scope where dom_map[i] is [9, 20]
 */
class ConditionalBoundsContext {
 private:
  friend class With<ConditionalBoundsContext>;
  /*!
   * \brief Construct a condition bounds context.
   * \param condition The condition holds on true branch.
   * \param relax_map The domain map for relaxed vars to update.
   * \param hint_map The domain map for free vars to update.
   * \param pending_conditions The stack of unresolved constraints.
   */
  ConditionalBoundsContext(const PrimExpr& condition, ffi::Map<Var, sym::IntSet>* relax_map,
                           ffi::Map<Var, sym::IntSet>* hint_map,
                           std::vector<PrimExpr>* pending_constraints);
  void EnterWithScope();
  void ExitWithScope();

  /*! \brief Helper to solve related variable's bound within conditional scope.*/
  ffi::Optional<ffi::Map<Var, Range>> TrySolveCondition();

  /*! \brief the condition holds on true branch. */
  const PrimExpr& condition_;
  /*! \brief domain map for relaxed vars to update */
  ffi::Map<Var, sym::IntSet>* relax_map_;
  /*! \brief domain map for free vars to update */
  ffi::Map<Var, sym::IntSet>* hint_map_;
  /*! \brief unresolved condition stack */
  std::vector<PrimExpr>* pending_conditions_;
  /*! \brief used to record and restore original var bounds */
  ffi::Map<Var, sym::IntSet> origin_map_;
  /*! \brief used to record unresolved conditions num. */
  size_t origin_pending_conditions_num_;
};

}  // namespace s_tir
}  // namespace tvm

#endif  // TVM_S_TIR_ANALYSIS_CONDITIONAL_BOUNDS_H_
