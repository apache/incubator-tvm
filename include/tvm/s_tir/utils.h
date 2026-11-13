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
#ifndef TVM_S_TIR_UTILS_H_
#define TVM_S_TIR_UTILS_H_

#include <tvm/s_tir/sblock_scope.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/stmt.h>

#include <unordered_map>

namespace tvm {
namespace tirx {

/*!
 * \brief A helper macro to convert an sref to the statement it points to,
 * then check if the downcasting succeeded.
 * \param Result The result variable, used for checking
 * \param SRef The SRef to be cast
 * \param Type The type to be cast to, can be s_tir::SBlock or For
 */
#define TVM_SREF_AS_OR_ERR(Result, SRef, Type) \
  SRef->StmtAs<Type>();                        \
  TVM_FFI_CHECK(Result, TypeError)

/*!
 * \brief A helper macro to convert an sref to the block it points to,
 *
 * Throws an internal error if downcasting fails.  The variable name
 * in the parent scope is used for the error message.
 *
 * \param SRef The SRef to be cast
 */
#define TVM_SREF_TO_SBLOCK(SRef)                                                        \
  [&]() {                                                                               \
    auto result = TVM_SREF_AS_OR_ERR(result, (SRef), ::tvm::s_tir::SBlockObj)           \
                  << "Expects StmtSRef `" << #SRef << "` points to `Block`, but gets: " \
                  << ((SRef)->stmt ? (SRef)->stmt->GetTypeKey() : "None");              \
    return result;                                                                      \
  }()

/*!
 * \brief A helper macro to convert an sref to the for-loop it points to
 *
 * Throws an internal error if downcasting fails.  The variable name
 * in the parent scope is used for the error message.
 *
 * \param SRef The SRef to be cast
 */
#define TVM_SREF_TO_FOR(SRef)                                                          \
  [&]() {                                                                              \
    auto result = TVM_SREF_AS_OR_ERR(result, (SRef), ::tvm::ForObj)                    \
                  << "Expects StmtSRef `" << #SRef << "` points to `Loop`, but gets: " \
                  << ((SRef)->stmt ? (SRef)->stmt->GetTypeKey() : "None");             \
    return result;                                                                     \
  }()

/*!
 * \brief Cast a TVM ffi::ObjectRef to its corresponding container using
 * `ffi::ObjectRef::as<Type>`, then check if the downcasting succeeded.
 * \param Result The result variable, used for checking
 * \param From The ffi::ObjectRef to be downcast
 * \param Type The type to be downcast to
 */
#define TVM_TYPE_AS_OR_ERR(Result, From, Type) \
  From.as<Type>();                             \
  TVM_FFI_CHECK(Result, TypeError)

/*!
 * \brief Cast a TVM ffi::ObjectRef to its corresponding container using
 * `ffi::ObjectRef::as<Type>`, throwing an internal error if downcast fails.
 * \param From The ffi::ObjectRef to be downcast
 * \param Type The type to be downcast to
 */
#define TVM_TYPE_AS(From, Type)                                                    \
  [&]() {                                                                          \
    auto result = TVM_TYPE_AS_OR_ERR(result, (From), Type)                         \
                  << "Expects `" << #From << "` to have type `" << Type::_type_key \
                  << "`, but gets: " << ((From).GetTypeKey());                     \
    return result;                                                                 \
  }()

/*!
 * \brief Set the `StmtSRefObj::seq_index` field for stmt
 * \param stmt2ref The stmt2ref map to be updated with seq_index
 * \param stmt The statement, or the realize node of the statement whose sref to be set
 * \param seq_index The seq_index to be set
 * \param include_loops Ignore ForNodes if this value is false
 * \note The method is NOP for statements that are not schedulable, i.e. not For or s_tir::SBlock
 */
inline void SetSeqIndex(std::unordered_map<const StmtObj*, StmtSRef>& stmt2ref,  // NOLINT(*)
                        const Stmt& stmt, int seq_index, bool include_loops = true) {
  if (const auto* realize = stmt.as<s_tir::SBlockRealizeObj>()) {
    const s_tir::SBlockObj* block = realize->block.get();
    TVM_FFI_ICHECK(stmt2ref.count(block));
    stmt2ref.at(block)->seq_index = seq_index;
  } else if (const auto* block = stmt.as<s_tir::SBlockObj>()) {
    TVM_FFI_ICHECK(stmt2ref.count(block));
    stmt2ref.at(block)->seq_index = seq_index;
  } else if (const auto* loop = stmt.as<ForObj>()) {
    if (!include_loops) return;
    TVM_FFI_ICHECK(stmt2ref.count(loop));
    stmt2ref.at(loop)->seq_index = seq_index;
  }
}

/*!
 * \brief Update seq_index of the children of a SeqStmt
 * \param stmt2ref The stmt2ref map to be updated with indices
 * \param seq_stmt The SeqStmt whose children need updating
 * \param include_loops Ignore ForNodes if this value is false
 */
inline void SetSeqIndexInChildren(
    std::unordered_map<const StmtObj*, StmtSRef>& stmt2ref,  // NOLINT(*)
    const SeqStmtObj* seq_stmt, bool include_loops = true) {
  int i = 0;
  for (const Stmt& stmt : seq_stmt->seq) {
    SetSeqIndex(stmt2ref, stmt, i, include_loops);
    ++i;
  }
}

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_S_TIR_UTILS_H_
