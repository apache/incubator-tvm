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
 * \file storage_access.h
 * \brief Common data structure for storage access analysis.
 */
#ifndef TVM_S_TIR_TRANSFORM_STORAGE_ACCESS_H_
#define TVM_S_TIR_TRANSFORM_STORAGE_ACCESS_H_

#include <tvm/ir/attrs.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/sym/int_set.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace s_tir {
using namespace tirx;

using runtime::MemoryScope;
/*!
 * \brief Base class of storage access analysis
 */
class StorageAccessVisitor : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;

  /*! \brief Storage access type */
  enum AccessType {
    kRead,
    kWrite,
    kSync,
    kAlloc,
    // acquired version of read, only need to handle WAR dep.
    kReadAcquire
  };
  /*! \brief An access entry */
  struct AccessEntry {
    /*! \brief The thread index that access this entry */
    ffi::Array<IterVar> threads;
    /*! \brief The buffer variable, if any */
    Var buffer{ffi::UnsafeInit{}};
    /*! \brief The access data type */
    PrimType dtype = PrimType::Void();
    /*! \brief The touched access range
     *
     * Has one IntSet for each index in the buffer being accessed.
     */
    ffi::Array<sym::IntSet> touched;
    /*! \brief The type of access */
    AccessType type;
    /*! \brief The storage scope */
    MemoryScope scope;
  };
  /*! \brief Access pattern about a single statement */
  struct StmtEntry {
    /*! \brief The statement */
    const ffi::Object* stmt;
    /*! \brief access patterns in the statement */
    std::vector<AccessEntry> access;
  };
  // override visitor pattern
  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadObj* op) final;
  ffi::Optional<VisitInterrupt> Visit_(const TensorStoreObj* op) final;
  ffi::Optional<VisitInterrupt> Visit_(const EvaluateObj* op) final;
  ffi::Optional<VisitInterrupt> Visit_(const BindObj* op) final;
  ffi::Optional<VisitInterrupt> Visit_(const RegionStmtObj* op) final;
  ffi::Optional<VisitInterrupt> Visit_(const ForObj* op) final;
  ffi::Optional<VisitInterrupt> Visit_(const IfObj* op) final;
  ffi::Optional<VisitInterrupt> Visit_(const WhileObj* op) final;
  ffi::Optional<VisitInterrupt> Visit_(const CallObj* op) final;
  ffi::Optional<VisitInterrupt> Visit_(const VarObj* op) final;

 protected:
  StorageAccessVisitor() { scope_.push_back(std::vector<StmtEntry>()); }
  /*! \return number of conditions in the current scope. */
  int condition_counter() const { return condition_counter_; }
  /*! \return whether we are in device environment. */
  bool in_device_env() const { return in_device_env_; }
  /*! \return environment threads */
  const ffi::Array<IterVar>& env_threads() const { return env_threads_; }
  /*!
   * \brief Whether we need analyze the buffer in current scope.
   * \param buffer The buffer to be checked
   * \param scope The scope of the buffer.
   * \return Whether the analysis of buffer is enabled.
   */
  virtual bool Enabled(const VarObj* buffer, const MemoryScope& scope) const { return true; }
  /*!
   * \brief Summarize the sequence of operations into parent.
   *
   *  Insert synchronization if necessary and remove un-necessary
   *  memory access which are already synced.
   *
   * \param seq The sequence of the access operations.
   * \param loop Pass loop node if it is a loop, otherwise nullptr.
   * \return The summarized sequence that represent access that
   *  the parent should taken care of to synchronize.
   */
  virtual std::vector<AccessEntry> Summarize(std::vector<StmtEntry> seq, const ForObj* loop) = 0;
  /*!
   * \brief Get the scope of the buffer array.
   * \return The scope of the final buffer array.
   */
  MemoryScope GetScope(Var buffer_var) const;
  /*! \brief Resolve a logical buffer view to its physical storage root. */
  Var ResolveBuffer(Var buffer_var) const;
  // access scope
  std::vector<std::vector<StmtEntry>> scope_;

 private:
  void RecordOpaqueAccess(Var source);
  bool HasOffsetPointer(const Expr& value) const;
  // whether access appending is enabled.
  bool allow_append_{false};
  // Pointer operands of an opaque call may be read or written by that call.
  bool in_opaque_call_{false};
  // Whether we are in device environment
  bool in_device_env_{false};
  // Whether we are inside condition.
  int condition_counter_{0};
  // the current free stmt entry.
  StmtEntry curr_stmt_;
  // The involving threads
  ffi::Array<IterVar> env_threads_;
  // Physical storage root for each declared logical buffer view.
  std::unordered_map<const VarObj*, Var> buffer_aliases_;
  // Resolve pointer bindings at their use, without treating address construction as access.
  std::unordered_map<const VarObj*, Expr> pointer_values_;
  // Shifted or retyped aliases cannot compare coordinates with their storage root.
  std::unordered_set<const VarObj*> offset_aliases_;
};
}  // namespace s_tir
}  // namespace tvm
#endif  // TVM_S_TIR_TRANSFORM_STORAGE_ACCESS_H_
