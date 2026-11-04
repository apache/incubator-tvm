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

#ifndef TVM_S_TIR_TRANSFORM_IR_UTILS_H_
#define TVM_S_TIR_TRANSFORM_IR_UTILS_H_

#include <tvm/ffi/container/tuple.h>
#include <tvm/ir/module.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/s_tir/stmt.h>

#include <string>
#include <unordered_map>

namespace tvm {
namespace s_tir {

/*! \brief Convert a schedulable statement or module to SSA form. */
Stmt ConvertSSA(Stmt stmt);
IRModule ConvertSSA(IRModule mod);

/*!
 * \brief Convert match tensor target tensor access indices to original one.
 * \param indices The indices of the target tensor
 * \return The indices of source tensor.
 */
ffi::Array<PrimExpr> ConvertIndices(const MatchTensorRegion& match_tensor,
                                    const ffi::Array<PrimExpr>& indices);

/*!
 * \brief Convert match tensor target tensor region to original one.
 * \param region The sub-region of the target tensor
 * \return The region of source tensor.
 */
ffi::Array<Range> ConvertRegion(const MatchTensorRegion& match_tensor,
                                const ffi::Array<Range>& region);

// Information of tensor core fragment.
struct FragmentInfo {
  // fragment shape
  int m, n, k;
  // fragment layout (row-major or column-major)
  std::string layout;
  // scope of the fragment (wmma.matrix_a, wmma.matrix_b, or wmma.accumulator)
  runtime::MemoryScope scope;
  FragmentInfo() = default;
  FragmentInfo(int _m, int _n, int _k, const std::string& _layout, runtime::MemoryScope _scope)
      : m(_m), n(_n), k(_k), layout(_layout), scope(_scope) {}

  int GetSize() const {
    static const auto scope_wmma_matrix_a = runtime::MemoryScope::Get("wmma.matrix_a");
    static const auto scope_wmma_matrix_b = runtime::MemoryScope::Get("wmma.matrix_b");
    static const auto scope_wmma_accumulator = runtime::MemoryScope::Get("wmma.accumulator");

    if (scope == scope_wmma_matrix_a) {
      return m * k;
    } else if (scope == scope_wmma_matrix_b) {
      return n * k;
    } else if (scope == scope_wmma_accumulator) {
      return m * n;
    } else {
      TVM_FFI_ICHECK(0);
      throw;
    }
  }
};

/*!
 * \brief Extract information of tensor core fragment from the IR.
 * \param stmt The stmt to visit.
 * \return Map from tensor variables to the fragment info.
 */
std::unordered_map<const VarObj*, FragmentInfo> GetTensorCoreFragmentInfo(const Stmt& stmt);

/*! \brief The quad used by StorageAlign for (tensor_idx, axis, factor, offset) */
using StorageAlignTuple = ffi::Tuple<int32_t, int32_t, int32_t, int32_t>;
/*! \brief A list of StorageAlignTuple, used by StorageAlign */
using StorageAlignAnnotation = ffi::Array<StorageAlignTuple>;
/*!
 * \brief Collect storage alignment annotations for all tensor vars within body.
 * \param body The stmt to collect.
 * \return The result dict from tensor var to storage align annotations.
 */
std::unordered_map<tvm::Var, StorageAlignAnnotation> CollectStorageAlignAnnotation(
    const Stmt& body);

}  // namespace s_tir
}  // namespace tvm
#endif  // TVM_S_TIR_TRANSFORM_IR_UTILS_H_
