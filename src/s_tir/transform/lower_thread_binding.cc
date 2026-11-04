/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/transform.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt.h>

namespace tvm {
namespace s_tir {
using namespace tvm::prim;
using namespace tvm::tirx;

class ThreadBindingLowerer : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

 private:
  UnchangedOr<Stmt> Mutate_(const SBlockObj* op, InplaceMode inplace_mode) final {
    auto annotations = Mutate(op->annotations, inplace_mode)
                           .as_or_throw<UnchangedOr<ffi::Map<ffi::String, ffi::Any>>>();
    auto result = StmtExprMutator::Mutate_(op, inplace_mode);
    if (annotations.UnchangedOrSameAs(op->annotations)) return result;
    SBlock block = std::move(result).ValueOrUnchanged(ffi::GetRef<Stmt>(op)).as_or_throw<SBlock>();
    block.CopyOnWrite()->annotations = std::move(annotations).ValueUnchecked();
    return block;
  }

  UnchangedOr<Stmt> Mutate_(const ForObj* op, InplaceMode inplace_mode) final {
    if (!tvm::s_tir::GetThreadBinding(op).has_value()) {
      auto annotations = Mutate(op->annotations, inplace_mode)
                             .as_or_throw<UnchangedOr<ffi::Map<ffi::String, ffi::Any>>>();
      auto result = StmtExprMutator::Mutate_(op, inplace_mode);
      if (annotations.UnchangedOrSameAs(op->annotations)) return result;
      For loop = std::move(result).ValueOrUnchanged(ffi::GetRef<Stmt>(op)).as_or_throw<For>();
      loop.CopyOnWrite()->annotations = std::move(annotations).ValueUnchecked();
      return loop;
    }
    PrimExpr min = Mutate(op->begin, inplace_mode).ValueOrUnchanged(op->begin);
    PrimExpr end = Mutate(op->end, inplace_mode).ValueOrUnchanged(op->end);
    PrimExpr extent = Range::GetExtentFromBeginEnd(min, end);
    TVM_FFI_ICHECK(IsZero(min)) << "Thread binding loops must start at zero";
    TVM_FFI_ICHECK(tvm::s_tir::GetThreadBinding(op).has_value());
    TVM_FFI_ICHECK(!op->annotations.count(tvm::s_tir::attr::kLoopPartitionHint.name) ||
                   op->annotations.at(tvm::s_tir::attr::kLoopPartitionHint.name) == nullptr)
        << "Run LoopPartition before LowerThreadBinding";
    PrimVar launch_var(op->var->name, extent.ty());
    auto previous_remap = VarRemapGet(op->var);
    VarRemapSet(op->var, prim::cast(launch_var, op->var.ty()));
    auto annotations = Mutate(op->annotations, inplace_mode)
                           .as_or_throw<UnchangedOr<ffi::Map<ffi::String, ffi::Any>>>()
                           .ValueOrUnchanged(op->annotations);
    Stmt body = Mutate(op->body, inplace_mode).ValueOrUnchanged(op->body);
    VarRemapSet(op->var, previous_remap);
    annotations.erase(tvm::s_tir::attr::kThreadBinding.name);
    if (!annotations.empty()) {
      PrimType ty = op->var.ty();
      body = For(PrimVar("annotation", ty), IntImm(ty, 0), IntImm(ty, 1), IntImm(ty, 1),
                 ForKind::kDefault, std::move(body), std::move(annotations), op->loc);
    }
    return RegionStmt(tirx::launch_thread_op(),
                      {StringImm(tvm::s_tir::GetThreadBinding(op).value()), extent}, {launch_var},
                      DictAttrs(), std::move(body), {}, op->loc);
  }
};

namespace transform {

Pass LowerThreadBinding() {
  auto pass_func = [](Function f, IRModule m, PassContext ctx) {
    if (!f->body.has_value()) return f;
    auto lower = ffi::make_object<ThreadBindingLowerer>();
    auto fptr = f.CopyOnWrite();
    fptr->body = lower->Mutate(fptr->body.value(), InplaceMode::kAllow)
                     .ValueOrUnchanged(std::move(fptr->body).value());
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "s_tir.LowerThreadBinding");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.LowerThreadBinding", LowerThreadBinding);
}

}  // namespace transform
}  // namespace s_tir
}  // namespace tvm
