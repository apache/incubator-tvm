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
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/sblock_scope.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/utils.h>

namespace tvm {
namespace tirx {

TVM_FFI_STATIC_INIT_BLOCK() {
  StmtSRefObj::RegisterReflection();
  DependencyObj::RegisterReflection();
  SBlockScopeObj::RegisterReflection();
}

/******** Utility functions ********/

template <class K, class V>
using SMap = std::unordered_map<K, V, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

/*!
 * \brief Add a dependency relation.
 * \param src The source of the dependency
 * \param dst The destination of the dependecy
 * \param kind Type of the dependency
 * \note This method is effectively NOP on self-loops
 */
void AddDependency(SBlockScopeObj* self, const StmtSRef& src, const StmtSRef& dst, DepKind kind) {
  if (!src.same_as(dst)) {
    Dependency dep(src, dst, kind);
    self->src2deps[src].push_back(dep);
    self->dst2deps[dst].push_back(dep);
  }
}

/******** Constructors ********/

StmtSRef::StmtSRef(const StmtObj* stmt, StmtSRefObj* parent, int64_t seq_index) {
  ffi::ObjectPtr<StmtSRefObj> n = ffi::make_object<StmtSRefObj>();
  n->stmt = stmt;
  n->parent = parent;
  n->seq_index = seq_index;
  data_ = std::move(n);
}

StmtSRef StmtSRef::InlineMark() {
  static StmtSRef result(nullptr, nullptr, -1);
  return result;
}

StmtSRef StmtSRef::RootMark() {
  static StmtSRef result(nullptr, nullptr, -1);
  return result;
}

Dependency::Dependency(StmtSRef src, StmtSRef dst, DepKind kind) {
  ffi::ObjectPtr<DependencyObj> node =
      ffi::make_object<DependencyObj>(std::move(src), std::move(dst), kind);
  data_ = std::move(node);
}

SBlockScope::SBlockScope() { data_ = ffi::make_object<SBlockScopeObj>(); }

SBlockScope::SBlockScope(const ffi::Array<StmtSRef>& child_block_srefs) {
  ffi::ObjectPtr<SBlockScopeObj> n = ffi::make_object<SBlockScopeObj>();
  SMap<TensorVar, ffi::Array<StmtSRef>> tensor_readers;
  SMap<TensorVar, ffi::Array<StmtSRef>>& tensor_writers = n->tensor_writers;
  for (const StmtSRef& child_block_sref : child_block_srefs) {
    const s_tir::SBlockObj* child_block = TVM_SREF_TO_SBLOCK(child_block_sref);
    // Step 1. Update `tensor_readers` and `tensor_writers` for each tensor
    for (const TensorRegion& region : child_block->reads) {
      tensor_readers[region->source.as_or_throw<tvm::tirx::TensorVar>()].push_back(
          child_block_sref);
    }
    for (const TensorRegion& region : child_block->writes) {
      tensor_writers[region->source.as_or_throw<tvm::tirx::TensorVar>()].push_back(
          child_block_sref);
    }
    // Step 2. Update RAW dependency
    for (const TensorRegion& region : child_block->reads) {
      auto it = tensor_writers.find(region->source.as_or_throw<tvm::tirx::TensorVar>());
      if (it != tensor_writers.end()) {
        for (const StmtSRef& from : it->second) {
          AddDependency(n.get(), from, child_block_sref, DepKind::kRAW);
        }
      }
    }
    // Step 3. Update WAW dependency
    for (const TensorRegion& region : child_block->writes) {
      auto it = tensor_writers.find(region->source.as_or_throw<tvm::tirx::TensorVar>());
      if (it != tensor_writers.end()) {
        for (const StmtSRef& from : it->second) {
          AddDependency(n.get(), from, child_block_sref, DepKind::kWAW);
        }
      }
    }
    // Step 4. Update WAR dependency
    for (const TensorRegion& region : child_block->writes) {
      auto it = tensor_readers.find(region->source.as_or_throw<tvm::tirx::TensorVar>());
      if (it != tensor_readers.end()) {
        for (const StmtSRef& from : it->second) {
          AddDependency(n.get(), from, child_block_sref, DepKind::kWAR);
        }
      }
    }
  }
  data_ = std::move(n);
}

/******** Dependency ********/

ffi::Array<Dependency> SBlockScopeObj::GetDepsBySrc(const StmtSRef& block_sref) const {
  auto iter = this->src2deps.find(block_sref);
  if (iter != this->src2deps.end()) {
    return iter->second;
  } else {
    return {};
  }
}

ffi::Array<Dependency> SBlockScopeObj::GetDepsByDst(const StmtSRef& block_sref) const {
  auto iter = this->dst2deps.find(block_sref);
  if (iter != this->dst2deps.end()) {
    return iter->second;
  } else {
    return {};
  }
}

/*!
 * \brief Add a new statement to the stack, which becomes the current scope
 * \param stmt A for-loop statement or a block statement
 */
void SRefTreeCreator::PushSRef(const StmtObj* stmt) {
  if (srefs_.empty()) {
    srefs_.push_back(
        StmtSRef(stmt,
                 /*parent=*/nullptr,
                 /*seq_index=*/-1));  // `seq_index` will be set properly in SetSeqIndex
  } else {
    StmtSRefObj* parent = srefs_.back().get();
    srefs_.push_back(
        StmtSRef(stmt, parent,
                 /*seq_index=*/-1));  // `seq_index` will be set properly in SetSeqIndex
  }
}

/*! \brief Pop the top of the scope and record it in stmt2ref map */
void SRefTreeCreator::PopAndRecordSRef() {
  StmtSRef sref = std::move(srefs_.back());
  stmt2ref_.insert_or_assign(sref->stmt, sref);
  srefs_.pop_back();
}

ffi::Optional<VisitInterrupt> SRefTreeCreator::Visit_(const ForObj* loop) {
  if (!include_loops_) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(loop->body));
  } else {
    PushSRef(loop);
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(loop->body));
    PopAndRecordSRef();
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> SRefTreeCreator::Visit_(const s_tir::SBlockRealizeObj* realize) {
  const s_tir::SBlockObj* block = realize->block.get();
  PushSRef(block);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(block->body));  // `block->init` is not visited
  PopAndRecordSRef();
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> SRefTreeCreator::Visit_(const SeqStmtObj* seq_stmt) {
  // Set `seq_index` information for SeqStmtObj
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(s_tir::StmtExprVisitor::Visit_(seq_stmt));
  SetSeqIndexInChildren(stmt2ref_, seq_stmt, include_loops_);
  return std::nullopt;
}

/******** FFI ********/

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("s_tir.StmtSRefStmt",
           [](StmtSRef sref) -> ffi::Optional<Stmt> {
             return sref->stmt ? ffi::Optional<Stmt>(ffi::GetRef<Stmt>(sref->stmt)) : std::nullopt;
           })
      .def("s_tir.StmtSRefParent",
           [](StmtSRef sref) -> ffi::Optional<StmtSRef> {
             return sref->parent ? ffi::Optional<StmtSRef>(ffi::GetRef<StmtSRef>(sref->parent))
                                 : std::nullopt;
           })
      .def("s_tir.StmtSRefRootMark", StmtSRef::RootMark)
      .def("s_tir.StmtSRefInlineMark", StmtSRef::InlineMark)
      .def_method("s_tir.SBlockScopeGetDepsBySrc", &SBlockScopeObj::GetDepsBySrc)
      .def_method("s_tir.SBlockScopeGetDepsByDst", &SBlockScopeObj::GetDepsByDst);
}

}  // namespace tirx
}  // namespace tvm
