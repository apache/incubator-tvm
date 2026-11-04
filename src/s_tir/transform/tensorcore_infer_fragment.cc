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
 * \brief Infer TensorCore metadata from tensor intrinsic.
 * \file tensorcore_fragment.cc
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/transform.h>
#include <tvm/tirx/op/memory.h>

#include <unordered_map>
#include <unordered_set>

#include "ir_utils.h"
#include "storage_access.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

const VarObj* GetBufferVarFromData(const Expr& data) {
  if (const auto* var = data.as<VarObj>()) {
    return var;
  }
  if (const auto* call = data.as<CallObj>();
      call && call->op.same_as(tirx::tensor_data_ptr_op()) && call->args.size() == 1) {
    return call->args[0].as<VarObj>();
  }
  return nullptr;
}

// Get fragment information from tensor intrinsics
class FragmentGetter : public s_tir::StmtExprVisitor {
 public:
  using s_tir::StmtExprVisitor::Visit_;
  ffi::Optional<VisitInterrupt> Visit_(const CallObj* op) final {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(s_tir::StmtExprVisitor::Visit_(op));

    static const Op gpu_load_matrix_sync_op = Op::Get("tirx.gpu_load_matrix_sync");
    static const Op gpu_store_matrix_sync_op = Op::Get("tirx.gpu_store_matrix_sync");
    static const Op gpu_fill_fragment_op = Op::Get("tirx.gpu_fill_fragment");
    if (op->op.same_as(gpu_load_matrix_sync_op) || op->op.same_as(gpu_store_matrix_sync_op)) {
      static const auto scope_wmma_matrix_a = runtime::MemoryScope::Get("wmma.matrix_a");
      static const auto scope_wmma_matrix_b = runtime::MemoryScope::Get("wmma.matrix_b");
      static const auto scope_wmma_accumulator = runtime::MemoryScope::Get("wmma.accumulator");

      // Get shape and layout information from load and store intrinsic
      TVM_FFI_ICHECK_EQ(op->args.size(), 8U);
      const VarObj* buffer_var = GetBufferVarFromData(op->args[0]);
      TVM_FFI_ICHECK(buffer_var);
      // Get shape
      const IntImmObj* m = op->args[1].as<IntImmObj>();
      const IntImmObj* n = op->args[2].as<IntImmObj>();
      const IntImmObj* k = op->args[3].as<IntImmObj>();
      const StringImmObj* layout = op->args[7].as<StringImmObj>();
      TVM_FFI_ICHECK(m);
      TVM_FFI_ICHECK(n);
      TVM_FFI_ICHECK(k);
      TVM_FFI_ICHECK(layout);

      runtime::MemoryScope scope =
          (buffer_var->ty.as<TensorTypeObj>() ? buffer_var->ty.as<TensorTypeObj>()->scope
                                              : buffer_var->ty.as_or_throw<PtrType>()->scope);
      if (fragments.count(buffer_var)) {
        // check if the fragment has met before
        FragmentInfo info = fragments[buffer_var];
        TVM_FFI_ICHECK_EQ(m->value, info.m);
        TVM_FFI_ICHECK_EQ(n->value, info.n);
        TVM_FFI_ICHECK_EQ(k->value, info.k);
        if (scope == scope_wmma_matrix_a || scope == scope_wmma_matrix_b) {
          TVM_FFI_ICHECK_EQ(layout->value, info.layout);
        }
      } else {
        // store metadata
        FragmentInfo info;
        if (scope == scope_wmma_matrix_a || scope == scope_wmma_matrix_b) {
          info = FragmentInfo(m->value.as<int>().value(), n->value.as<int>().value(),
                              k->value.as<int>().value(), layout->value, scope);
        } else if (scope == scope_wmma_accumulator) {
          info = FragmentInfo(m->value.as<int>().value(), n->value.as<int>().value(),
                              k->value.as<int>().value(), "", scope);
        }
        fragments[buffer_var] = info;
      }
    } else if (op->op.same_as(gpu_fill_fragment_op)) {
      // Get shape information from fill intrinsic
      TVM_FFI_ICHECK_EQ(op->args.size(), 6U);
      const VarObj* buffer_var = GetBufferVarFromData(op->args[0]);
      TVM_FFI_ICHECK(buffer_var);
      // Get shape
      const IntImmObj* m = op->args[1].as<IntImmObj>();
      const IntImmObj* n = op->args[2].as<IntImmObj>();
      const IntImmObj* k = op->args[3].as<IntImmObj>();
      TVM_FFI_ICHECK(m);
      TVM_FFI_ICHECK(n);
      TVM_FFI_ICHECK(k);

      runtime::MemoryScope scope =
          (buffer_var->ty.as<TensorTypeObj>() ? buffer_var->ty.as<TensorTypeObj>()->scope
                                              : buffer_var->ty.as_or_throw<PtrType>()->scope);
      if (fragments.count(buffer_var)) {
        FragmentInfo info = fragments[buffer_var];
        TVM_FFI_ICHECK_EQ(m->value, info.m);
        TVM_FFI_ICHECK_EQ(n->value, info.n);
        TVM_FFI_ICHECK_EQ(k->value, info.k);
      } else {
        // default to row major ordering
        FragmentInfo info(m->value.as<int>().value(), n->value.as<int>().value(),
                          k->value.as<int>().value(), "row_major", scope);
        fragments[buffer_var] = info;
      }
    }
    return std::nullopt;
  }

  // Fragment metadata for all fragments
  std::unordered_map<const VarObj*, FragmentInfo> fragments;
};

std::unordered_map<const VarObj*, FragmentInfo> GetTensorCoreFragmentInfo(const Stmt& stmt) {
  auto getter = ffi::make_object<s_tir::FragmentGetter>();
  getter->Visit(stmt);
  return std::move(getter->fragments);
}

// Check shape of fragment making sure it is a valid shape for gpu_mma_sync
class FragmentChecker : public s_tir::StmtExprVisitor {
 public:
  using s_tir::StmtExprVisitor::Visit_;
  explicit FragmentChecker(const FragmentGetter& getter) : fragment_getter(getter) {}

  ffi::Optional<VisitInterrupt> Visit_(const CallObj* op) final {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(s_tir::StmtExprVisitor::Visit_(op));
    // Check shape when calling gpu_mma_sync
    static const Op gpu_mma_sync_op = Op::Get("tirx.gpu_mma_sync");
    static const Op bmma_sync_op = Op::Get("tirx.cuda.bmma_sync");
    if (op->op.same_as(gpu_mma_sync_op) || op->op.same_as(bmma_sync_op)) {
      TVM_FFI_ICHECK_EQ(op->args.size(), 8U);
      const VarObj* buffer_var_d = GetBufferVarFromData(op->args[0]);
      const VarObj* buffer_var_a = GetBufferVarFromData(op->args[2]);
      const VarObj* buffer_var_b = GetBufferVarFromData(op->args[4]);
      const VarObj* buffer_var_c = GetBufferVarFromData(op->args[6]);
      TVM_FFI_ICHECK(buffer_var_d);
      TVM_FFI_ICHECK(buffer_var_a);
      TVM_FFI_ICHECK(buffer_var_b);
      TVM_FFI_ICHECK(buffer_var_c);

      // Check all fragment A, B, C and D have the same shape
      TVM_FFI_ICHECK(CheckShape(buffer_var_d, buffer_var_a));
      TVM_FFI_ICHECK(CheckShape(buffer_var_d, buffer_var_b));
      TVM_FFI_ICHECK(CheckShape(buffer_var_d, buffer_var_c));
    }
    return std::nullopt;
  }

 private:
  // A tool for checking shapes of two fragments
  bool CheckShape(const VarObj* buffer1, const VarObj* buffer2) {
    TVM_FFI_ICHECK(fragment_getter.fragments.count(buffer1))
        << "Tensorecore fragment " << buffer1->name
        << " must be filled (with gpu_fill_fragment) or loaded (with gpu_load_matrix_sync) before "
           "use.";
    TVM_FFI_ICHECK(fragment_getter.fragments.count(buffer2))
        << "Tensorecore fragment " << buffer2->name
        << " must be filled (with gpu_fill_fragment) or loaded (with gpu_load_matrix_sync) before "
           "use.";
    FragmentInfo info1 = fragment_getter.fragments.at(buffer1);
    FragmentInfo info2 = fragment_getter.fragments.at(buffer2);
    return info1.m == info2.m && info1.n == info2.n && info1.k == info2.k;
  }
  // Fragment infomation
  const FragmentGetter& fragment_getter;
};

// Store the metadata into attributes
class InferFragmenter : public s_tir::StmtExprMutator {
 public:
  using s_tir::StmtExprMutator::Mutate;
  using s_tir::StmtExprMutator::Mutate_;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) override {
    if (value.as<ExprObj>()) return ffi::Unchanged();
    return s_tir::StmtExprMutator::Mutate(value, inplace_mode);
  }

  explicit InferFragmenter(const FragmentGetter& getter) : fragment_getter(getter) {}

  UnchangedOr<Stmt> Mutate_(const BindObj* op, InplaceMode inplace_mode) final {
    if (const auto* call = op->value.as<CallObj>();
        call && call->op.same_as(tirx::alloc_tensor_op())) {
      auto it = fragment_getter.fragments.find(op->var.get());
      if (it == fragment_getter.fragments.end()) return ffi::Unchanged();
      const FragmentInfo& info = it->second;
      DictAttrs attrs = call->attrs.as_or_throw<DictAttrs>();
      auto annotations = attrs->dict;
      auto set_annotation = [&](const char* key, const std::string& value) {
        if (auto existing = annotations.Get(key)) {
          auto str = existing.value().as<ffi::String>();
          TVM_FFI_CHECK(str && str.value() == value, ValueError)
              << "Conflicting " << key << " on fragment " << op->var->name;
        } else {
          annotations.Set(key, ffi::String(value));
        }
      };
      set_annotation(
          tvm::s_tir::attr::kFragmentShape.name,
          std::to_string(info.m) + ", " + std::to_string(info.n) + ", " + std::to_string(info.k));
      if (!info.layout.empty()) set_annotation(tvm::s_tir::attr::kFragmentLayout.name, info.layout);
      if (annotations.same_as(attrs->dict)) return ffi::Unchanged();
      auto value = ffi::GetRef<Call>(call);
      value.CopyOnWrite()->attrs = DictAttrs(std::move(annotations));
      return Bind(op->var, std::move(value), op->loc);
    }
    return s_tir::StmtExprMutator::Mutate_(op, inplace_mode);
  }

 private:
  // Fragment infomation
  const FragmentGetter& fragment_getter;
};

Stmt InferFragment(Stmt stmt) {
  auto getter = ffi::make_object<FragmentGetter>();
  getter->Visit(stmt);
  auto checker = ffi::make_object<FragmentChecker>(*getter);
  checker->Visit(stmt);
  stmt = ffi::make_object<InferFragmenter>(*getter)
             ->Mutate(stmt, InplaceMode::kAllow)
             .ValueOrUnchanged(std::move(stmt));
  return stmt;
}

namespace transform {

Pass InferFragment() {
  auto pass_func = [](Function f, IRModule m, PassContext ctx) {
    if (!f->body.has_value()) return f;
    auto* n = f.CopyOnWrite();
    n->body = s_tir::InferFragment(std::move(n->body).value());
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "s_tir.InferFragment");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.InferFragment", static_cast<Pass (*)()>(InferFragment));
}

}  // namespace transform
}  // namespace s_tir
}  // namespace tvm
