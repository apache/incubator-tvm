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
 * \file shape_analysis.cc
 *
 * \brief Utilities for shape analysis.
 */

#include <tvm/relax/analysis.h>
#include <tvm/sym/analyzer.h>

namespace tvm {
namespace relax {

bool CanProveShapeEqual(const ffi::Array<PrimExpr>& lhs, const ffi::Array<PrimExpr>& rhs,
                        const sym::Analyzer& ana) {
  if (lhs.same_as(rhs)) return true;
  if (lhs.size() != rhs.size()) return false;
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!ana->CanProveEqual(lhs[i], rhs[i])) return false;
  }
  return true;
}

bool CanProveShapeEqual(const Expr& lhs, const Expr& rhs, const sym::Analyzer& ana) {
  if (lhs.same_as(rhs)) return true;
  auto* lhs_shape = lhs.as<ShapeExprObj>();
  auto* rhs_shape = rhs.as<ShapeExprObj>();

  if (lhs_shape && rhs_shape) {
    return CanProveShapeEqual(lhs_shape->values, rhs_shape->values, ana);
  } else {
    return false;
  }
}

}  // namespace relax
}  // namespace tvm
