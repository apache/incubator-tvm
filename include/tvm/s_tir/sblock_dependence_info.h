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
 * \file tvm/s_tir/sblock_dependence_info.h
 * \brief Define SBlockDependenceInfoObj that uses the SBlockScope and StmtSRef objects to
 * store the block level dependences
 * \sa SBlockDependenceInfoObj
 */

/**
 * @brief An object that builds and maintains block scope and StmtSref mapping for Dependence
 * analysis
 */

#ifndef TVM_S_TIR_SBLOCK_DEPENDENCE_INFO_H_
#define TVM_S_TIR_SBLOCK_DEPENDENCE_INFO_H_

#include <tvm/base.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/sblock_scope.h>

#include <unordered_map>

namespace tvm {
namespace tirx {

/**
 * @brief An object that helps build and query block level dependences using the 2 core objects
 * SBlockScope and StmtSRef
 *
 * The data structures exposed are:
 * 1) sref2scope: Mapping from the srefs to its corresponding SBlockScope
 * 2) stmt2ref: Mapping from blocks to corresponding StmtSRefs
 *
 * Note that this object does not store SRefs to loops as the purpose is only to expose block level
 * dependences. This provides the advantage that the scope block (parent block) for a given block
 * sref can be directly accessed using the sref->parent member
 */
class SBlockDependenceInfoObj : public ffi::Object {
 public:
  /*!
   * \brief Mapping from a block sref to its corresponding SBlockScope,
   * tracking the dependency inside the block scope,
   */
  std::unordered_map<StmtSRef, SBlockScope, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> sref2scope;
  /*! \brief The reverse mapping from block/for-loop to their corresponding srefs */
  std::unordered_map<const StmtObj*, StmtSRef> stmt2ref;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SBlockDependenceInfoObj>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.SBlockDependenceInfo", SBlockDependenceInfoObj,
                                    ffi::Object);

  /*!
   * \brief Get the SBlockScope corresponding to the sref of scope root block
   * \param scope_root The block sref to be retrieved
   * \return The corresponding SBlockScope
   */
  SBlockScope GetSBlockScope(const StmtSRef& scope_root) const {
    auto it = sref2scope.find(scope_root);
    TVM_FFI_CHECK(it != sref2scope.end(), IndexError)
        << "Cannot find the corresponding SBlockScope to the block sref:\n"
        << ffi::GetRef<Stmt>(scope_root->stmt);
    return it->second;
  }
};

/*!
 * \brief Managed reference to SBlockDependenceInfoObj
 * \sa SBlockDependenceInfo
 */
class SBlockDependenceInfo : public ffi::ObjectRef {
  /*! \brief Construct an empty SBlockDependenceInfo
   */
  TVM_DLL SBlockDependenceInfo();

 public:
  /*! \brief Construct a SBlockDependenceInfo from IRModule
   */
  TVM_DLL SBlockDependenceInfo(IRModule mod);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SBlockDependenceInfo, ffi::ObjectRef,
                                                SBlockDependenceInfoObj);
};

}  // namespace tirx
}  // namespace tvm
#endif  // TVM_S_TIR_SBLOCK_DEPENDENCE_INFO_H_
