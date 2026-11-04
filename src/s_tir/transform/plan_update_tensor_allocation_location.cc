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
 * \brief Planning where tensors to be allocated and update the AST.
 * \file plan_update_tensor_allocation_location.cc
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/transform.h>
#include <tvm/tirx/analysis.h>

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

class CollectManagedAllocations : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;
  ffi::Optional<VisitInterrupt> Visit_(const SBlockObj* op) final {
    for (const auto& tensor : op->alloc_tensors) {
      managed_allocations.insert(tensor.get());
    }
    for (const auto& tensor : op->match_tensors) {
      managed_allocations.insert(tensor->tensor.get());
    }
    return StmtExprVisitor::Visit_(op);
  }

  /*! \brief Tensors that are allocated outside of the BlockNode, and should not be moved by
   * TensorAllocationLocator. */
  std::unordered_set<const VarObj*> managed_allocations;
};

/*! \brief Collect the allocate tensor order. */
class TensorAllocateOrderCollector : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;
  static ffi::Array<TensorVar> Collect(const Function& func) {
    auto collector = ffi::make_object<TensorAllocateOrderCollector>();
    for (const Var& param : func->params) {
      if (auto tensor = param.as<TensorVar>()) {
        collector->tensor_alloc_recorder_.push_back(tensor.value());
      }
    }
    collector->Visit(func->body);
    return std::move(collector->tensor_alloc_recorder_);
  }

 private:
  bool find(const TensorVar& tensor) {
    return std::find(tensor_alloc_recorder_.begin(), tensor_alloc_recorder_.end(), tensor) !=
           tensor_alloc_recorder_.end();
  }

  ffi::Optional<VisitInterrupt> Visit_(const SBlockObj* op) final {
    for (const TensorVar& tensor : op->alloc_tensors) {
      tensor_alloc_recorder_.push_back(tensor);
    }
    // Also visit match_tensors to collect tensors that only appear in read and match_tensor
    // regions.
    for (const auto& region : op->match_tensors) {
      if (!find(region->source->source.as_or_throw<tvm::tirx::TensorVar>())) {
        tensor_alloc_recorder_.push_back(
            region->source->source.as_or_throw<tvm::tirx::TensorVar>());
      }
    }

    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadObj* op) final {
    if (!find(op->source.as_or_throw<tvm::tirx::TensorVar>())) {
      tensor_alloc_recorder_.push_back(op->source.as_or_throw<tvm::tirx::TensorVar>());
    }
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorStoreObj* op) final {
    if (!find(op->dest.as_or_throw<TensorVar>())) {
      tensor_alloc_recorder_.push_back(op->dest.as_or_throw<TensorVar>());
    }
    return StmtExprVisitor::Visit_(op);
  }

  /*! \brief The tensor allocated order recorder. */
  ffi::Array<TensorVar> tensor_alloc_recorder_;
};

class TensorAllocationLocator : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  explicit TensorAllocationLocator(const Function& func) {
    ffi::Map<TensorVar, ffi::Optional<Stmt>> tensor_lca = DetectTensorAccessLCA(func);
    // The tensor_alloc_recorder Array is used to keep the tensor allocation order
    // since the tensor_lca Map is unordered.
    ffi::Array<TensorVar> tensor_alloc_recorder = TensorAllocateOrderCollector::Collect(func);
    std::unordered_set<const VarObj*> arg_tensor_vars;
    auto collector = ffi::make_object<CollectManagedAllocations>();
    collector->Visit(func->body);
    managed_allocations_ = collector->managed_allocations;

    for (const Var& param : func->params) {
      if (auto tensor = param.as<TensorVar>()) {
        arg_tensor_vars.emplace(tensor.value().get());
        tensor_data_to_tensor_.Set(tensor.value(), tensor.value());
      }
    }
    // create tensors to be allocated at each stmts
    for (const auto& tensor : tensor_alloc_recorder) {
      auto it = tensor_lca.find(tensor);
      if (it != tensor_lca.end()) {
        const StmtObj* stmt = (*it).second.has_value() ? (*it).second.value().get() : nullptr;
        if (arg_tensor_vars.count(tensor.get())) {
          continue;
        }
        if (managed_allocations_.count(tensor.get())) {
          alloc_tensors_[stmt].push_back(tensor);
        }
        tensor_data_to_tensor_.Set(tensor, tensor);
      }
    }
  }

 private:
  UnchangedOr<Stmt> Mutate_(const ForObj* op, InplaceMode inplace_mode) final {
    auto it = alloc_tensors_.find(op);
    if (it == alloc_tensors_.end()) {
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }
    for (const TensorVar& tensor : it->second) {
      tensor_data_to_tensor_.Set(tensor, tensor);
    }
    auto node = StmtExprMutator::Mutate_(op, inplace_mode)
                    .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                    .as_or_throw<For>();

    ffi::Array<TensorVar> new_block_alloc_tensors;
    for (const TensorVar& tensor : it->second) {
      if (managed_allocations_.count(tensor.get())) {
        tensor_data_to_tensor_.erase(tensor);
        new_block_alloc_tensors.push_back(tensor);
      }
    }

    if (new_block_alloc_tensors.size()) {
      node.CopyOnWrite()->body = InjectOpaqueBlock(node->body, new_block_alloc_tensors);
    }

    return node;
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj* op, InplaceMode inplace_mode) final {
    auto it = alloc_tensors_.find(op);
    if (it == alloc_tensors_.end()) {
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }
    for (const TensorVar& tensor : it->second) {
      tensor_data_to_tensor_.Set(tensor, tensor);
    }
    auto node = StmtExprMutator::Mutate_(op, inplace_mode)
                    .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                    .as_or_throw<RegionStmt>();

    ffi::Array<TensorVar> new_block_alloc_tensors;
    for (const TensorVar& tensor : it->second) {
      if (managed_allocations_.count(tensor.get())) {
        tensor_data_to_tensor_.erase(tensor);
        new_block_alloc_tensors.push_back(tensor);
      }
    }

    if (new_block_alloc_tensors.size()) {
      node.CopyOnWrite()->body = InjectOpaqueBlock(node->body, new_block_alloc_tensors);
    }

    return node;
  }

  UnchangedOr<Stmt> Mutate_(const SBlockObj* op, InplaceMode inplace_mode) final {
    TVM_FFI_ICHECK(!op->init.has_value());
    ffi::Array<TensorVar> alloc_tensors;
    auto it = alloc_tensors_.find(op);
    if (it != alloc_tensors_.end()) {
      alloc_tensors = it->second;
      for (const TensorVar& tensor : it->second) {
        tensor_data_to_tensor_.Set(tensor, tensor);
      }
    }
    for (const MatchTensorRegion match_tensor : op->match_tensors) {
      const Var target_var = match_tensor->tensor;
      const Var source_var = match_tensor->source->source.as_or_throw<tvm::tirx::TensorVar>();
      TVM_FFI_ICHECK(tensor_data_to_tensor_.count(source_var));
      tensor_data_to_tensor_.Set(target_var, match_tensor->tensor);
    }
    SBlock stmt = StmtExprMutator::Mutate_(op, inplace_mode)
                      .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                      .as_or_throw<SBlock>();
    op = stmt.as<SBlockObj>();
    TVM_FFI_ICHECK(op != nullptr);

    // No longer consider tensors created by match_tensor inside the block when updating access
    // region.
    for (const MatchTensorRegion match_tensor : op->match_tensors) {
      const Var target_var = match_tensor->tensor;
      tensor_data_to_tensor_.erase(target_var);
    }
    // No longer consider tensors allocated inside the block when updating access region.
    if (it != alloc_tensors_.end()) {
      for (const TensorVar& tensor : it->second) {
        tensor_data_to_tensor_.erase(tensor);
      }
    }

    SBlockObj* n = stmt.CopyOnWrite();
    n->alloc_tensors = std::move(alloc_tensors);
    // Erase tensor allocated inside the block from access region.
    n->reads = RemoveRedundantTensorRegion(n->reads);
    n->writes = RemoveRedundantTensorRegion(n->writes);
    return stmt;
  }

  Stmt InjectOpaqueBlock(Stmt body, const ffi::Array<TensorVar>& alloc_tensors) {
    TVM_FFI_ICHECK(!alloc_tensors.empty());
    SBlock opaque_block(/*iter_vars=*/{},
                        /*reads=*/{},
                        /*writes=*/{},
                        /*name_hint=*/"",
                        /*body=*/std::move(body),
                        /*init=*/std::nullopt,
                        /*alloc_tensors=*/alloc_tensors);
    SBlockObj* n = opaque_block.CopyOnWrite();
    ffi::Array<ffi::Array<TensorRegion>> access =
        GetSBlockReadWriteRegion(opaque_block, tensor_data_to_tensor_);
    n->reads = access[0];
    n->writes = access[1];
    SBlockRealize realize({}, IntImm::Bool(true), std::move(opaque_block));
    return realize;
  }

  ffi::Array<TensorRegion> RemoveRedundantTensorRegion(
      const ffi::Array<TensorRegion>& region) const {
    ffi::Array<TensorRegion> result;
    for (const TensorRegion& tensor_region : region) {
      if (tensor_data_to_tensor_.count(tensor_region->source.as_or_throw<tvm::tirx::TensorVar>())) {
        result.push_back(tensor_region);
      }
    }
    return result;
  }

  /*! \brief The map from stmt to the tensors to be allocated under it. */
  std::unordered_map<const StmtObj*, ffi::Array<TensorVar>> alloc_tensors_;
  /*! \brief The tensor already allocated during recursive visiting. */
  ffi::Map<Var, TensorVar> tensor_data_to_tensor_;
  /*! \brief Tensors that are allocated within a BlockNode, and may be moved. */
  std::unordered_set<const VarObj*> managed_allocations_;
};

namespace transform {

Pass PlanAndUpdateTensorAllocationLocation() {
  auto pass_func = [=](Function f, IRModule m, PassContext ctx) {
    auto fptr = f.CopyOnWrite();
    auto locator = ffi::make_object<TensorAllocationLocator>(f);
    fptr->body = locator->Mutate(fptr->body).ValueOrUnchanged(fptr->body);
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "s_tir.PlanAndUpdateTensorAllocationLocation");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.PlanAndUpdateTensorAllocationLocation",
                        PlanAndUpdateTensorAllocationLocation);
}

}  // namespace transform

}  // namespace s_tir
}  // namespace tvm
