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
 * \file thread_storage_sync.cc
 */
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/transform.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op/gpu.h>

#include <unordered_set>

#include "storage_access.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

class ThreadSyncPlanner : public StorageAccessVisitor {
 public:
  using StorageAccessVisitor::Visit_;
  explicit ThreadSyncPlanner(MemoryScope sync_scope) : sync_scope_(sync_scope) {}

  // The syncs inserted before each statement
  std::unordered_set<const ffi::Object*> syncs_inserted_;

 protected:
  bool Enabled(const VarObj* buf, const MemoryScope& scope) const final {
    return in_device_env() && scope == sync_scope_;
  }
  // Plan the sync
  std::vector<AccessEntry> Summarize(std::vector<StmtEntry> seq, const ForObj* loop) final {
    static const auto scope_shared_dyn = runtime::MemoryScope::Get("shared.dyn");

    // Redirect all "shared.dyn" buffer access to the same buffer var
    // so that the accesses can be planned together.
    ffi::Optional<Var> shared_dyn_buf;
    for (StmtEntry& entry : seq) {
      for (AccessEntry& access : entry.access) {
        if (access.scope == scope_shared_dyn && access.buffer.defined()) {
          if (!shared_dyn_buf.has_value()) {
            shared_dyn_buf = access.buffer;
          } else {
            access.buffer = shared_dyn_buf.value();
          }
        }
      }
    }

    // Unsynced reads and writes
    std::vector<AccessEntry> reads;
    std::vector<AccessEntry> writes;
    // if it is a loop, rotate two times to consider effect of loop.
    // simulation based approach to find dependencies
    for (size_t i = 0; i < seq.size(); ++i) {
      const StmtEntry& s = seq[i];
      // check if sync before statement is needed.
      bool sync_before_stmt = (syncs_inserted_.count(s.stmt) != 0);
      // Apply the syncs added already.
      if (sync_before_stmt) {
        reads.clear();
        writes.clear();
      }
      for (const AccessEntry& acc : s.access) {
        if (acc.type == kRead) {
          if (FindConflict(writes, acc)) {
            sync_before_stmt = true;
            break;
          }
        } else if (acc.type == kWrite) {
          if (FindConflict(reads, acc)) {
            sync_before_stmt = true;
            break;
          }
        } else if (acc.type == kSync) {
          reads.clear();
          writes.clear();
        }
      }
      // If sync is inserted. remove the irrelevant things.
      if (sync_before_stmt) {
        reads.clear();
        writes.clear();
      }
      // Add the read/write of current statement
      for (const AccessEntry& acc : s.access) {
        if (acc.type == kRead) {
          reads.push_back(acc);
        } else if (acc.type == kWrite) {
          writes.push_back(acc);
        } else if (acc.type == kSync) {
          reads.clear();
          writes.clear();
        }
      }
      if (sync_before_stmt) {
        TVM_FFI_ICHECK_EQ(condition_counter(), 0) << "Cannot insert syncs inside condition";
        syncs_inserted_.insert(s.stmt);
      }
    }
    if (loop != nullptr) {
      for (size_t i = 0; i < seq.size(); ++i) {
        const StmtEntry& s = seq[i];
        if (syncs_inserted_.count(s.stmt) != 0) break;
        if (reads.empty() && writes.empty()) break;
        bool sync_before_stmt = false;
        for (const AccessEntry& acc : s.access) {
          if (acc.type == kRead) {
            if (FindConflict(writes, acc)) {
              sync_before_stmt = true;
              break;
            }
          } else if (acc.type == kWrite) {
            if (FindConflict(reads, acc)) {
              sync_before_stmt = true;
              break;
            }
          } else if (acc.type == kSync) {
            reads.clear();
            writes.clear();
          }
        }
        if (sync_before_stmt) {
          TVM_FFI_ICHECK_EQ(condition_counter(), 0) << "Cannot insert syncs inside condition";
          syncs_inserted_.insert(s.stmt);
          break;
        }
      }
    }
    // return the exposed entries, remove unecessary ones.
    int sync_count = 0;
    // head are before first sync, tail are after last sync
    std::vector<AccessEntry> head, tail;
    AccessEntry esync;
    esync.threads = this->env_threads();
    esync.type = kSync;
    esync.scope = sync_scope_;

    for (const StmtEntry& s : seq) {
      if (syncs_inserted_.count(s.stmt)) {
        if (sync_count != 0) {
          tail.clear();
        } else {
          head.push_back(esync);
        }
        ++sync_count;
      }
      for (const AccessEntry& acc : s.access) {
        if (acc.type == kSync) {
          if (sync_count != 0) {
            tail.clear();
          } else {
            head.push_back(esync);
          }
          ++sync_count;
        } else {
          if (sync_count != 0) {
            tail.push_back(acc);
          } else {
            head.push_back(acc);
          }
        }
      }
    }
    head.insert(head.end(), tail.begin(), tail.end());
    return head;
  }

 private:
  // find conflicting entry in vec.
  bool FindConflict(const std::vector<AccessEntry>& prev, const AccessEntry& curr) {
    for (const AccessEntry& x : prev) {
      if (FindConflict(x, curr)) {
        return true;
      }
    }
    return false;
  }

  bool FindConflict(const AccessEntry& prev, const AccessEntry& curr) {
    // Access to different buffers does not conflict.
    if (!prev.buffer.same_as(curr.buffer)) {
      return false;
    }

    // Raw pointers and differently shaped views may refer to the same storage.
    // Their element-wise coordinates cannot prove the accesses disjoint.
    if (prev.touched.empty() || curr.touched.empty() ||
        prev.touched.size() != curr.touched.size()) {
      return true;
    }

    // Assumes no race between threads
    // Same index value means no conflicts
    // TODO(tqchen) more standard set based testing.
    bool has_same_index = true;
    // Even if access has the same index, those indices need to
    // depend on the innermost thread id to avoid race condition
    bool depends_on_thread_index = true;
    const VarObj* thread_index_var = nullptr;
    if (!curr.threads.empty()) {
      thread_index_var = curr.threads.back()->var.get();
    }

    for (size_t i = 0; i < prev.touched.size(); i++) {
      const auto& prev_intset = prev.touched[i];
      const auto& curr_intset = curr.touched[i];

      if (prev_intset.IsSinglePoint() && curr_intset.IsSinglePoint()) {
        PrimExpr prev_index = prev_intset.PointValue();
        PrimExpr curr_index = curr_intset.PointValue();
        has_same_index = prim::ExprDeepEqual()(prev_index, curr_index);
        if (thread_index_var != nullptr) {
          auto f_uses_thread_index = [=](const tvm::VarObj* parameter) {
            return parameter == thread_index_var;
          };
          auto walkfn = [&](const Var& var) -> ffi::Expected<ffi::WalkResult> {
            return f_uses_thread_index(var.get())
                       ? ffi::WalkResult::Interrupt(ffi::VisitInterrupt(var))
                       : ffi::WalkResult::Advance();
          };
          depends_on_thread_index =
              depends_on_thread_index &&
              ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(curr_index, walkfn).has_value() &&
              ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(prev_index, walkfn).has_value();
        }
      } else {
        has_same_index = false;
      }

      if (!(has_same_index && depends_on_thread_index)) {
        break;
      }
    }
    if (has_same_index && depends_on_thread_index) {
      return false;
    }

    // If nothing else allows sharing the same buffer, then they are
    // in conflict.
    return true;
  }

 private:
  // synchronization scope
  MemoryScope sync_scope_;
};

// There are cases where necessary syncthreads is not inserted by ThreadSyncInserter.
// For example, syncthreads is needed after async_wait in the second loop below,
// but since ThreadSyncInserter is not aware of the asynchronous semantics, it cannot tell
// that the syncthreads is needed there.
//
// // Pipeline prologue
// for i in range(125):
//    with async_copy_scope():
//       shared[(i + 3) % 4] = ...
//    async_commit(0)
// ...
//
// // Pipeline Epilogue
// for i in range(3):
//    async_wait(0, 2 - i)
//    local[...] = shared[(i + 125) % 4]

// This class adds syncthreads after all async_wait operations. That includes syncthreads that
// can be inserted by ThreadSyncInserter as well, but ThreadSyncInserter will not insert
// duplicate syncthreads if it finds an existing one at the synchronization point.
class ThreadSyncAfterWaitQueueInserter : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  explicit ThreadSyncAfterWaitQueueInserter(MemoryScope sync_scope) : sync_scope_(sync_scope) {}

  UnchangedOr<Stmt> Mutate_(const EvaluateObj* op, InplaceMode inplace_mode) final {
    if (const auto* call = op->value.as<CallObj>(); call && call->op.same_as(s_tir::async_wait())) {
      auto sync = Evaluate(
          Call(PrimType::Int(32), tirx::gpu_storage_sync_op(), {StringImm(sync_scope_.name())}));
      return SeqStmt({ffi::GetRef<Stmt>(op), sync});
    }
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }

 private:
  MemoryScope sync_scope_;
};

class ThreadSyncInserter : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  ThreadSyncInserter(MemoryScope sync_scope, const std::unordered_set<const ffi::Object*>& syncs)
      : sync_scope_(sync_scope), syncs_(syncs) {}

  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) final {
    const auto* stmt = value.as<StmtObj>();
    if (!stmt) return StmtExprMutator::Mutate(value, inplace_mode);
    if (syncs_.empty()) return ffi::Unchanged();
    if (!syncs_.count(stmt)) return StmtExprMutator::Mutate(value, inplace_mode);
    Stmt barrier = Evaluate(
        Call(PrimType::Int(32), tirx::gpu_storage_sync_op(), {StringImm(sync_scope_.name())})
            .as_or_throw<PrimExpr>());
    // Mutate after query, to avoid stmt change.
    auto result = StmtExprMutator::Mutate(value, inplace_mode);
    Stmt body = std::move(result).ValueOrUnchanged(value).as_or_throw<Stmt>();
    return ffi::Any(SeqStmt({barrier, body}));
  }

 private:
  // data structure.
  MemoryScope sync_scope_;
  const std::unordered_set<const ffi::Object*>& syncs_;
};

Stmt ThreadSync(Stmt stmt, MemoryScope sync_scope) {
  static const auto scope_shared = runtime::MemoryScope::Get("shared");

  if (sync_scope == scope_shared) {
    stmt = ffi::make_object<ThreadSyncAfterWaitQueueInserter>(sync_scope)
               ->Mutate(stmt)
               .ValueOrUnchanged(stmt);
  }
  auto planner = ffi::make_object<ThreadSyncPlanner>(sync_scope);
  planner->Visit(stmt);
  return ffi::make_object<ThreadSyncInserter>(sync_scope, planner->syncs_inserted_)
      ->Mutate(stmt, InplaceMode::kAllow)
      .ValueOrUnchanged(std::move(stmt));
}

// The synchronization markers have served their purpose after all ThreadSync passes.
// Only CUDA has backend queue operations; other targets retain synchronous copies.
class SynchronizationLowerer : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  explicit SynchronizationLowerer(bool is_cuda) : is_cuda_(is_cuda) {}

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj* op, InplaceMode inplace_mode) final {
    if (op->op.same_as(s_tir::manual_sync()) || op->op.same_as(s_tir::async_copy_scope())) {
      return Mutate(op->body, inplace_mode).ValueOrUnchanged(op->body);
    }
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Stmt> Mutate_(const EvaluateObj* op, InplaceMode inplace_mode) final {
    const auto* call = op->value.as<CallObj>();
    if (!call ||
        (!call->op.same_as(s_tir::async_commit()) && !call->op.same_as(s_tir::async_wait()))) {
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }
    if (!is_cuda_) return Evaluate(0);
    TVM_FFI_ICHECK_EQ(call->args[0].as_or_throw<IntImm>()->value, 0)
        << "For CUDA, the index of an async queue must be 0.";
    if (call->op.same_as(s_tir::async_commit())) {
      static const Op commit = Op::Get("tirx.ptx.cp_async_commit_group");
      return Evaluate(Call(PrimType::Void(), commit,
                           {StringImm("async"), StringImm("commit_group"), StringImm("")}));
    }
    static const Op wait = Op::Get("tirx.ptx.cp_async_wait_group");
    // PTX's immediate operand retains the compile-time wait-count requirement.
    return Evaluate(
        Call(PrimType::Void(), wait,
             {call->args[1], StringImm("async"), StringImm("wait_group"), StringImm("")}));
  }

 private:
  bool is_cuda_;
};

namespace transform {

Pass ThreadSync(runtime::MemoryScope storage_scope) {
  auto pass_func = [storage_scope](Function f, IRModule m, PassContext ctx) {
    if (!f->body.has_value()) return f;
    auto* n = f.CopyOnWrite();
    n->body = s_tir::ThreadSync(std::move(n->body).value(), storage_scope);
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "s_tir.ThreadSync");
}

Pass LowerSynchronization() {
  auto pass_func = [](Function f, IRModule, PassContext) {
    if (!f->body.has_value()) return f;
    auto target = f->GetAttr(tvm::attr::kTarget);
    bool is_cuda = target && target.value()->kind->name == "cuda";
    auto* n = f.CopyOnWrite();
    n->body = ffi::make_object<SynchronizationLowerer>(is_cuda)
                  ->Mutate(n->body.value(), InplaceMode::kAllow)
                  .ValueOrUnchanged(n->body.value());
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "s_tir.LowerSynchronization");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("s_tir.transform.ThreadSync", static_cast<Pass (*)(runtime::MemoryScope)>(ThreadSync))
      .def("s_tir.transform.LowerSynchronization", LowerSynchronization);
}

}  // namespace transform
}  // namespace s_tir
}  // namespace tvm
