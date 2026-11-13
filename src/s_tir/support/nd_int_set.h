/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
#ifndef TVM_S_TIR_SUPPORT_ND_INT_SET_H_
#define TVM_S_TIR_SUPPORT_ND_INT_SET_H_

#include <tvm/ir/expr.h>
#include <tvm/sym/int_set.h>

#include <vector>

namespace tvm {
namespace support {

/*! \brief An N-dimensional integer set representing a rectangle region */
using NDIntSet = std::vector<sym::IntSet>;

/*!
 * \brief Construct an N-dimensional integer set representing a region.
 * \param region The region.
 * \return The constructed set.
 */
inline NDIntSet NDIntSetFromRegion(const ffi::Array<Range>& region) {
  NDIntSet result;
  result.reserve(region.size());
  for (const Range& range : region) {
    result.push_back(sym::IntSet::FromRange(range));
  }
  return result;
}

/*!
 * \brief Construct an N-dimensional integer set representing a shape.
 * \param shape The shape which is an array of the length of each dimension.
 * \return The constructed set.
 */
inline NDIntSet NDIntSetFromShape(const ffi::Array<PrimExpr>& shape) {
  PrimExpr zero = IntImm::Int32(0);
  NDIntSet result;
  result.reserve(shape.size());
  for (const PrimExpr& extent : shape) {
    result.push_back(sym::IntSet::FromMinExtent(zero, extent));
  }
  return result;
}

/*!
 * \brief Construct an N-dimensional integer set representing a point.
 * \param indices The N-dimensional indices representing the point.
 * \return The constructed set.
 */
inline NDIntSet NDIntSetFromPoint(const ffi::Array<PrimExpr>& indices) {
  NDIntSet result;
  result.reserve(indices.size());
  for (const PrimExpr& index : indices) {
    result.push_back(sym::IntSet::SinglePoint(index));
  }
  return result;
}

/*!
 * \brief Create a union set of two sets, possibly relaxed. The RHS set will be combined into the
 *        LHS set.
 * \param lhs The first N-dimensional integer set
 * \param rhs The second N-dimensional integer set
 */
inline void NDIntSetUnionWith(NDIntSet* lhs, const NDIntSet& rhs) {
  TVM_FFI_ICHECK_EQ(lhs->size(), rhs.size());
  int ndim = rhs.size();
  for (int i = 0; i < ndim; ++i) {
    sym::IntSet& int_set = lhs->at(i);
    int_set = sym::Union({int_set, rhs.at(i)});
  }
}

/*!
 * \brief Union a list of N-dimensional integer sets
 * \param nd_int_sets The N-dimensional integer sets to be merged.
 * \return The result of the union
 */
inline NDIntSet NDIntSetUnion(const std::vector<NDIntSet>& nd_int_sets) {
  TVM_FFI_ICHECK(!nd_int_sets.empty());
  int n = nd_int_sets.size();
  if (n == 1) {
    return nd_int_sets[0];
  }
  int ndim = nd_int_sets[0].size();
  for (int i = 1; i < n; ++i) {
    TVM_FFI_ICHECK_EQ(nd_int_sets[i].size(), ndim);
  }
  NDIntSet result;
  result.reserve(ndim);
  ffi::Array<sym::IntSet> int_sets;
  for (int dim = 0; dim < ndim; ++dim) {
    int_sets.clear();
    for (int i = 0; i < n; ++i) {
      int_sets.push_back(nd_int_sets[i][dim]);
    }
    result.push_back(sym::Union(int_sets));
  }
  return result;
}

/*!
 * \brief Create an empty N-dimensional integer set.
 * \param ndim The number of dimensions.
 * \return The constructed set.
 */
inline NDIntSet NDIntSetEmpty(int ndim) {
  return std::vector<sym::IntSet>(ndim, sym::IntSet::Nothing());
}

/*!
 * \brief The N-dimensional version of EvalSet.
 * \param nd_int_set The N-dimensional integer set to be evaluated.
 * \param dom_map The domain of each variable.
 * \return An N-dimensional integer set that can cover all the possible values of the N-dimensional
 *         integer set.
 * \sa EvalSet
 */
inline NDIntSet NDIntSetEval(const NDIntSet& nd_int_set,
                             const ffi::Map<Var, sym::IntSet>& dom_map) {
  NDIntSet ret;
  ret.reserve(nd_int_set.size());
  for (const sym::IntSet& s : nd_int_set) {
    ret.push_back(EvalSet(s, dom_map));
  }
  return ret;
}

}  // namespace support
}  // namespace tvm

#endif  // TVM_S_TIR_SUPPORT_ND_INT_SET_H_
