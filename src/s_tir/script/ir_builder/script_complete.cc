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
 * \file s_tir/script/ir_builder/script_complete.cc
 * \brief Used by TVM Script parser to expand incomplete TIR input
 */

#include "./script_complete.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/sym/int_set.h>
#include <tvm/tirx/analysis.h>

#include <utility>

namespace tvm {
namespace s_tir {

using namespace tvm::tirx;

/*! \brief Generate surrounding loops automatically */
class ScriptCompleter : public s_tir::StmtExprMutator {
 public:
  using s_tir::StmtExprMutator::Mutate;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) final {
    if (value.as<ExprObj>()) return ffi::Unchanged();
    return s_tir::StmtExprMutator::Mutate(value, inplace_mode);
  }

  explicit ScriptCompleter(ffi::Map<Var, TensorVar>* tensor_var_map)
      : tensor_var_map_(tensor_var_map) {}

 private:
  ffi::Map<Var, TensorVar>* tensor_var_map_;

  UnchangedOr<Stmt> Mutate_(const ForObj* op, InplaceMode inplace_mode) final {
    GetThreadBinding(op);
    return s_tir::StmtExprMutator::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Stmt> Mutate_(const s_tir::SBlockRealizeObj* op, InplaceMode inplace_mode) final {
    for (const PrimExpr& value : op->iter_values) {
      PrimType value_ty = value.ty();
      TVM_FFI_ICHECK(value_ty.code() == DLDataTypeCode::kDLInt)
          << "BlockRealize iter_value expected a IntImm, but got " << value_ty->dtype;
    }
    return s_tir::StmtExprMutator::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Stmt> Mutate_(const s_tir::SBlockObj* op, InplaceMode inplace_mode) final {
    // Tensors allocated in the block can be accessed by its body.
    for (const auto& alloc_tensor : op->alloc_tensors) {
      tensor_var_map_->Set(alloc_tensor, alloc_tensor);
    }
    for (const auto& match_tensor : op->match_tensors) {
      const TensorVar& target_tensor = match_tensor->tensor;
      tensor_var_map_->Set(target_tensor, target_tensor);
    }

    bool is_root_block = this->is_root_block_;
    this->is_root_block_ = false;
    s_tir::SBlock block = s_tir::StmtExprMutator::Mutate_(op, inplace_mode)
                              .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                              .as_or_throw<s_tir::SBlock>();
    this->is_root_block_ = is_root_block;

    // Remove tensors allocated inside block to detect its access region
    for (const auto& alloc_tensor : op->alloc_tensors) {
      tensor_var_map_->erase(alloc_tensor);
    }
    for (const auto& match_tensor : op->match_tensors) {
      const TensorVar& target_tensor = match_tensor->tensor;
      tensor_var_map_->erase(target_tensor);
    }
    // Get access detection mask
    // 0 for provided region, 1 and 3 for need detect read, 2 and 3 for need detect write
    int mask = 0;
    auto it = op->annotations.find(tvm::s_tir::attr::kScriptParsingDetectAccess.name);
    if (it != op->annotations.end()) {
      mask = (*it).second.cast<int>();
    }
    // ignore root block or blocks which already has reads/writes regions
    if (mask != 0) {
      auto access_region = GetSBlockAccessRegion(block, *tensor_var_map_);
      const ffi::Array<TensorRegion>& reads = access_region[0];
      const ffi::Array<TensorRegion>& writes = access_region[1];
      const ffi::Array<TensorRegion>& opaque = access_region[2];
      TVM_FFI_CHECK(opaque.empty(), ValueError)
          << "Can not auto detect tensor access region from tirx.Load, tirx.Store or "
             "direct access by tensor data. Please annotation the access region manually";
      auto* n = block.CopyOnWrite();
      if (!is_root_block) {
        if (mask & 1) n->reads = reads;
        if (mask & 2) n->writes = writes;
      }
      n->annotations = op->annotations;
      n->annotations.erase(tvm::s_tir::attr::kScriptParsingDetectAccess.name);
      return block;
    } else {
      return block;
    }
  }

  UnchangedOr<Stmt> Mutate_(const BindObj* op, InplaceMode inplace_mode) final {
    Stmt stmt =
        s_tir::StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    const Var& var = stmt.as<BindObj>()->var;
    // A flat definition registers its tensor for access detection in subsequent siblings.
    if (auto tensor = var.as<TensorVar>(); tensor && !tensor_var_map_->count(var)) {
      tensor_var_map_->Set(var, tensor.value());
    }
    return stmt;
  }

  bool is_root_block_ = true;
};

Function ScriptComplete(Function func, const ffi::Array<TensorVar>& root_allocates,
                        const TensorAllocatedAddresses& root_addresses) {
  func = WithAttr(std::move(func), tvm::attr::kScriptNamespace.name, ffi::String("s_tir"));
  if (!func->body.has_value()) return func;
  ffi::Map<Var, TensorVar> tensor_var_map;
  for (const Var& param : func->params) {
    if (auto tensor = param.as<TensorVar>()) {
      tensor_var_map.Set(tensor.value(), tensor.value());
    }
  }
  for (const auto& alloc : root_allocates) {
    tensor_var_map.Set(alloc, alloc);
  }

  Stmt res = func->body.value();

  // Generate root block automatically.  This is done before
  // ScriptCompleter, in order to fill the root block's Ts.reads() and
  // Ts.writes() annotations, as if it had been explicitly written.
  bool should_insert_root = [&]() -> bool {
    if (root_allocates.size()) {
      return true;
    }
    auto* block_realize = func->body.value()->seq.size() == 1
                              ? func->body.value()->seq[0].as<s_tir::SBlockRealizeObj>()
                              : nullptr;
    if (block_realize && block_realize->block->iter_vars.size()) {
      return true;
    }
    if (!block_realize && ContainsNode<s_tir::SBlockRealizeObj>(func->body.value())) {
      return true;
    }
    return false;
  }();

  if (should_insert_root) {
    ffi::Map<ffi::String, ffi::Any> annotations;
    if (!root_addresses.empty()) {
      annotations.Set(s_tir::attr::kTensorAllocatedAddr.name, root_addresses);
    }
    s_tir::SBlock root_block({}, {}, {}, "root", std::move(res), std::nullopt, root_allocates, {},
                             annotations);
    res = s_tir::SBlockRealize({}, IntImm::Bool(true), std::move(root_block));
  }

  // generate surrounding loops automatically
  auto script_completer = ffi::make_object<ScriptCompleter>(&tensor_var_map);
  res = script_completer->Mutate(res, InplaceMode::kAllow).ValueOrUnchanged(std::move(res));

  if (func->body.same_as(res)) {
    return func;
  } else {
    auto fptr = func.CopyOnWrite();
    fptr->body = tvm::SeqStmt(res);
    return func;
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.script.Complete",
                        [](Function func, const ffi::Array<TensorVar>& root_allocates) {
                          return ScriptComplete(std::move(func), root_allocates);
                        });
}

}  // namespace s_tir
}  // namespace tvm
