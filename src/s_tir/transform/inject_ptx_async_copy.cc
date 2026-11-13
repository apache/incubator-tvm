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
 * \brief Replace copy from global to shared with async copy
 * \file inject_ptx_async_copy.cc
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/transform.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op/memory.h>

#include <unordered_map>
#include <utility>
#include <vector>

#include "storage_access.h"
#include "tvm/s_tir/stmt.h"
#include "tvm/tirx/stmt.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

class PTXAsyncCopyInjector : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) override {
    if (value.as<ExprObj>()) return ffi::Unchanged();
    return StmtExprMutator::Mutate(value, inplace_mode);
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj* region, InplaceMode inplace_mode) {
    if (region->op.same_as(s_tir::async_copy_scope())) {
      TVM_FFI_ICHECK(in_async == false) << "Nested async scopes not supported";
      in_async = true;
      auto body = this->Mutate(region->body, inplace_mode).ValueOrUnchanged(region->body);
      in_async = false;
      return body;
    }
    return StmtExprMutator::Mutate_(region, inplace_mode);
  }

  UnchangedOr<Stmt> Mutate_(const BindObj* op, InplaceMode inplace_mode) final {
    const auto* call = op->value.as<CallObj>();
    if (call) {
      const auto* attrs = call->attrs.as<DictAttrsObj>();
      // Track only ordinary fresh allocations and exact, unannotated views.
      if (!call->attrs.has_value() || (attrs && attrs->dict.empty())) {
        if (call->op.same_as(alloc_tensor_op()) && call->args.empty()) {
          allocations_.emplace(op->var.get(), op->var.as_or_throw<TensorVar>());
        } else if (call->op.same_as(decl_tensor_op()) && call->args.size() == 1) {
          const auto* data = call->args[0].as<CallObj>();
          if (data && data->op.same_as(tensor_data_ptr_op()) && data->args.size() == 1) {
            const auto* source = data->args[0].as<VarObj>();
            auto it = allocations_.find(source);
            if (it != allocations_.end()) {
              TensorVar view = op->var.as_or_throw<TensorVar>();
              if (view.dtype() == it->second.dtype() && view.scope() == it->second.scope() &&
                  !view.ty()->layout.has_value()) {
                allocations_.emplace(op->var.get(), it->second);
              }
            }
          }
        }
      }
    }
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Stmt> Mutate_(const ForObj* op, InplaceMode inplace_mode) final {
    if (in_async) {
      if (auto copy = CoalesceHalfCopy(op)) return *copy;
    }
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }

  Stmt InjectPTX(const TensorLoadObj* load, const TensorStoreObj* store, bool predicated = false,
                 ffi::Optional<PrimExpr> predicate_value = std::nullopt) {
    static const auto scope_shared_dyn = runtime::MemoryScope::Get("shared.dyn");

    TVM_FFI_ICHECK(!predicated || predicate_value.has_value());
    if (load->source.as_or_throw<tvm::tirx::TensorVar>().scope() ==
        runtime::MemoryScope::Global()) {
      TVM_FFI_ICHECK(load->indices.size() == 1 && store->indices.size() == 1);
      TVM_FFI_ICHECK(load->indices[0].ty().lanes() == store->indices[0].ty().lanes());

      const int indices_lanes = load->indices[0].ty().lanes();
      const int bytes =
          indices_lanes *
          static_cast<int>(load->source.as_or_throw<tvm::tirx::TensorVar>().dtype().StorageBytes());

      if (bytes == 4 || bytes == 8 || bytes == 16) {
        PrimType dst_elem_type = store->dest.as_or_throw<TensorVar>().dtype();
        PrimType src_elem_type = load->source.as_or_throw<TensorVar>().dtype();

        int index_factor = 1;
        if (dst_elem_type != src_elem_type) {
          // The only case where src and dst have different dtypes is when the dst shared memory
          // is a byte tensor generated by merging dynamic shared memory.
          TVM_FFI_ICHECK(store->dest.as_or_throw<TensorVar>().scope() == scope_shared_dyn);
          TVM_FFI_ICHECK((dst_elem_type == PrimType::UInt(8)));
          // TensorStore/Load have the "pointer reinterpret" semantics according to their
          // "value" dtype. Their "indices" are supposed to be applied after such pointer cast,
          // for example: ((*float16)(byte_buffer))[tensor->indices] = fp16_value;
          // To replace TensorStore/Load with cp.async, we need to multiply the store index by
          // the byte size of the "value" dtype, to get the correct offset into the byte tensor.
          index_factor = static_cast<int>(src_elem_type.StorageBytes());
        }

        if (indices_lanes == 1) {
          auto src_offset = load->indices[0];
          auto dst_offset = store->indices[0];
          ffi::Array<Expr> args = {
              store->dest.as_or_throw<TensorVar>().data(), mul(dst_offset, PrimExpr(index_factor)),
              load->source.as_or_throw<tvm::tirx::TensorVar>().data(), src_offset, PrimExpr(bytes)};
          // use arguments size to indicate whether or not to use predicated cp.async
          if (predicated) {
            args.push_back(predicate_value.value());
          }
          static const Op ptx_cp_async_op = Op::Get("tirx.s_tir.cp_async_raw");
          return Evaluate(Call(store->dest.as_or_throw<TensorVar>().dtype(), ptx_cp_async_op, args)
                              .as_or_throw<PrimExpr>());
        }

        // Predicated load don't support vectorized indexing.
        if (!predicated) {
          // Only some vectorized indexing patterns are supported for now.
          auto src_offset = [=]() -> ffi::Optional<PrimExpr> {
            if (load->indices[0]->IsInstance<prim::RampObj>()) {
              return load->indices[0].as<prim::RampObj>()->base;
            }
            return std::nullopt;
          }();

          auto dst_offset = [=]() -> ffi::Optional<PrimExpr> {
            if (store->indices[0].as<prim::RampObj>()) {
              return store->indices[0].as<prim::RampObj>()->base;
            } else if (store->indices[0].as<prim::AddObj>()) {
              // The case where the dst tensor is a byte tensor generated by merging dynamic
              // shared memory.
              // A_shared.dyn[(ramp(...), 1, 8) + x8(17408))] = A_global[ramp(...),1, 8)]
              auto* add = store->indices[0].as<prim::AddObj>();
              if (!add->a->IsInstance<prim::RampObj>()) return std::nullopt;
              if (!add->b->IsInstance<prim::BroadcastObj>()) return std::nullopt;
              return prim::Add(add->a.as<prim::RampObj>()->base,
                               add->b.as<prim::BroadcastObj>()->value);
            }
            return std::nullopt;
          }();
          if (src_offset.has_value() && dst_offset.has_value()) {
            static const Op ptx_cp_async_op = Op::Get("tirx.s_tir.cp_async_raw");
            return Evaluate(Call(store->dest.as_or_throw<TensorVar>().dtype(), ptx_cp_async_op,
                                 {store->dest.as_or_throw<TensorVar>().data(),
                                  mul(dst_offset.value(), PrimExpr(index_factor)),
                                  load->source.as_or_throw<tvm::tirx::TensorVar>().data(),
                                  src_offset.value(), PrimExpr(bytes)})
                                .as_or_throw<PrimExpr>());
          }
        } else {
          // Only some vectorized indexing patterns are supported for now.
          auto src_offset = [=]() -> ffi::Optional<PrimExpr> {
            if (load->indices[0]->IsInstance<prim::RampObj>()) {
              return load->indices[0].as<prim::RampObj>()->base;
            }
            return std::nullopt;
          }();

          auto dst_offset = [=]() -> ffi::Optional<PrimExpr> {
            if (store->indices[0].as<prim::RampObj>()) {
              return store->indices[0].as<prim::RampObj>()->base;
            } else if (store->indices[0].as<prim::AddObj>()) {
              // The case where the dst tensor is a byte tensor generated by merging dynamic
              // shared memory.
              // A_shared.dyn[(ramp(...), 1, 8) + x8(17408))] = A_global[ramp(...),1, 8)]
              auto* add = store->indices[0].as<prim::AddObj>();
              if (!add->a->IsInstance<prim::RampObj>()) return std::nullopt;
              if (!add->b->IsInstance<prim::BroadcastObj>()) return std::nullopt;
              return prim::Add(add->a.as<prim::RampObj>()->base,
                               add->b.as<prim::BroadcastObj>()->value);
            }
            return std::nullopt;
          }();

          if (src_offset.has_value() && dst_offset.has_value()) {
            static const Op ptx_cp_async_op = Op::Get("tirx.s_tir.cp_async_raw");
            return Evaluate(Call(store->dest.as_or_throw<TensorVar>().dtype(), ptx_cp_async_op,
                                 {store->dest.as_or_throw<TensorVar>().data(),
                                  mul(dst_offset.value(), PrimExpr(index_factor)),
                                  load->source.as_or_throw<tvm::tirx::TensorVar>().data(),
                                  src_offset.value(), PrimExpr(bytes), predicate_value.value()})
                                .as_or_throw<PrimExpr>());
          }
        }
      }
    }
    return StmtExprMutator::Mutate_(store, InplaceMode::kDisallow)
        .ValueOrUnchanged(ffi::GetRef<Stmt>(store));
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreObj* store, InplaceMode inplace_mode) {
    static const auto scope_shared = runtime::MemoryScope::Get("shared");
    static const auto scope_shared_dyn = runtime::MemoryScope::Get("shared.dyn");

    if (in_async && (store->dest.as_or_throw<TensorVar>().scope() == scope_shared ||
                     store->dest.as_or_throw<TensorVar>().scope() == scope_shared_dyn)) {
      if (auto* load = store->value.as<TensorLoadObj>()) {
        return InjectPTX(load, store);
      } else if (auto* call = store->value.as<CallObj>()) {
        // tirx.if_then_else is a call to prim::if_then_else_op()
        if (call->op.same_as(prim::if_then_else_op()) && call->args.size() == 3) {
          if (auto* load = call->args[1].as<TensorLoadObj>()) {
            // Only default value of 0 is supported since 0 is the default value used by cp.async
            // ptx. @see section 9.7.8.22.3. of
            // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-memory-operations
            bool else_value_is_zero = false;
            if (auto* b = call->args[2].as<prim::BroadcastObj>()) {
              if (auto* f = b->value.as<FloatImmObj>()) {
                else_value_is_zero = f->value == 0.0f;
              }
            }
            if (auto* f = call->args[2].as<FloatImmObj>()) {
              else_value_is_zero = f->value == 0.0f;
            }
            if (else_value_is_zero) {
              return InjectPTX(load, store, true, call->args[0].as_or_throw<PrimExpr>());
            }
          }
        }
      }
    }
    return StmtExprMutator::Mutate_(store, inplace_mode);
  }

 private:
  ffi::Optional<Stmt> CoalesceHalfCopy(const ForObj* loop) {
    // A short, contiguous half copy can be one hardware transaction even when
    // the general loop vectorizer leaves it scalar. Do not widen arbitrary
    // computation, predicates, or loops with other effects.
    const auto* extent = loop->end.as<IntImmObj>();
    if (loop->kind != ForKind::kDefault || !prim::IsZero(loop->begin) || !loop->HasTrivialStep() ||
        !loop->annotations.empty() || !extent ||
        (extent->value != 2 && extent->value != 4 && extent->value != 8) ||
        loop->body->seq.empty()) {
      return std::nullopt;
    }
    const int bytes = extent->value.as<int>().value() * 2;
    static const auto shared = runtime::MemoryScope::Get("shared");
    static const auto shared_dyn = runtime::MemoryScope::Get("shared.dyn");
    static const Op copy_op = Op::Get("tirx.s_tir.cp_async_raw");
    std::vector<std::pair<const VarObj*, PrimExpr>> destinations;
    ffi::Array<Stmt> copies;
    for (const Stmt& stmt : loop->body->seq) {
      const auto* store = stmt.as<TensorStoreObj>();
      if (!store) return std::nullopt;
      const auto* load = store->value.as<TensorLoadObj>();
      if (!load || load->indices.size() != 1 || store->indices.size() != 1) {
        return std::nullopt;
      }
      TensorVar src = load->source.as_or_throw<TensorVar>();
      TensorVar dst = store->dest.as_or_throw<TensorVar>();
      auto allocation = allocations_.find(dst.get());
      if (src.scope() != runtime::MemoryScope::Global() ||
          (dst.scope() != shared && dst.scope() != shared_dyn) ||
          src.dtype() != PrimType::Float(16) || store->value.ty() != src.dtype() ||
          dst.dtype() != src.dtype() || src.ty()->layout.has_value() ||
          dst.ty()->layout.has_value() || !load->indices[0].ty().IsScalar() ||
          !store->indices[0].ty().IsScalar() || allocation == allocations_.end() ||
          allocation->second.ty()->layout.has_value() ||
          allocation->second.ty()->data_alignment < bytes ||
          allocation->second.ty()->data_alignment % bytes != 0 ||
          src.ty()->data_alignment < bytes || src.ty()->data_alignment % bytes != 0 ||
          dst.ty()->data_alignment < bytes || dst.ty()->data_alignment % bytes != 0) {
        return std::nullopt;
      }
      auto source = DetectLinearEquation(load->indices[0], {loop->var});
      auto destination = DetectLinearEquation(store->indices[0], {loop->var});
      if (source.size() != 2 || destination.size() != 2 || !prim::IsOne(source[0]) ||
          !prim::IsOne(destination[0]) || SideEffect(source[1]) != CallEffectKind::kPure ||
          SideEffect(destination[1]) != CallEffectKind::kPure ||
          !analyzer_->CanProveEqual(floormod(source[1], PrimExpr(bytes / 2)), 0) ||
          !analyzer_->CanProveEqual(floormod(destination[1], PrimExpr(bytes / 2)), 0)) {
        return std::nullopt;
      }
      // Storage rewriting can merge destinations into views of one allocation.
      // Prove their copied intervals disjoint before reordering their stores.
      for (const auto& [root, offset] : destinations) {
        if (root == allocation->second.get() &&
            !analyzer_->CanProve(destination[1] + bytes / 2 <= offset) &&
            !analyzer_->CanProve(offset + bytes / 2 <= destination[1])) {
          return std::nullopt;
        }
      }
      destinations.emplace_back(allocation->second.get(), destination[1]);
      // Reads are global and destinations are disjoint fresh shared storage.
      copies.push_back(
          Evaluate(Call(dst.dtype(), copy_op,
                        {dst.data(), destination[1], src.data(), source[1], PrimExpr(bytes)})
                       .as_or_throw<PrimExpr>()));
    }
    return SeqStmt(copies);
  }

  sym::Analyzer analyzer_;
  std::unordered_map<const VarObj*, TensorVar> allocations_;
  bool in_async{false};
};

namespace transform {

Pass InjectPTXAsyncCopy() {
  auto pass_func = [=](Function f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = ffi::make_object<PTXAsyncCopyInjector>()->Mutate(n->body).ValueOrUnchanged(n->body);
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "s_tir.InjectPTXAsyncCopy");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.InjectPTXAsyncCopy", InjectPTXAsyncCopy);
}

}  // namespace transform

}  // namespace s_tir
}  // namespace tvm
