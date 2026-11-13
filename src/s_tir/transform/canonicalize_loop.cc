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
 * \file s_tir/transform/canonicalize_loop.cc
 * \brief Canonicalize all loops to start from zero and step one.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/device_api.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/transform.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/function.h>

#include <utility>

namespace tvm {
namespace s_tir {
using namespace tvm::prim;

using namespace tvm::tirx;

class LoopCanonicalizer : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  LoopCanonicalizer() = default;

 private:
  UnchangedOr<Stmt> Mutate_(const ForObj* op, InplaceMode inplace_mode) final {
    if (IsZero(op->begin) && op->HasTrivialStep()) {
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }
    const auto* loop_var = op->var.get();
    PrimType loop_var_ty = loop_var->ty.as_or_throw<PrimType>();
    PrimExpr step = op->step;

    // report warning for negative step, since it would be a forever loop
    if (!analyzer_->CanProveGreaterEqual(step, 1)) {
      // TODO(tvm): prove dynamic shaped step
      TVM_FFI_THROW(InternalError)
          << "Loop step for " << op->var << " may not be positive: " << step;
    }

    VarRemapSet(op->var, op->var.as_or_throw<PrimExpr>() * step + op->begin);
    Stmt body = Mutate(op->body, inplace_mode).ValueOrUnchanged(op->body);
    PrimExpr min = IntImm(loop_var_ty, 0);
    PrimExpr extent =
        analyzer_->Simplify(ceildiv(Range::GetExtentFromBeginEnd(op->begin, op->end), step));
    if (inplace_mode == InplaceMode::kAllow) {
      auto* writable = const_cast<ForObj*>(op);
      writable->body = std::move(body);
      writable->begin = std::move(min);
      writable->end = std::move(extent);
      writable->step = IntImm(loop_var_ty, 1);
      return ffi::Unchanged();
    } else {
      auto copy = ffi::make_object<ForObj>(*op);
      copy->body = std::move(body);
      copy->begin = std::move(min);
      copy->end = std::move(extent);
      copy->step = IntImm(loop_var_ty, 1);
      return For(std::move(copy));
    }
  }

  sym::Analyzer analyzer_;
};

namespace transform {

Pass CanonicalizeLoop() {
  auto pass_func = [=](Function func, IRModule m, PassContext ctx) {
    FunctionObj* fptr = func.CopyOnWrite();
    fptr->body = ffi::make_object<LoopCanonicalizer>()
                     ->Mutate(fptr->body, InplaceMode::kAllow)
                     .ValueOrUnchanged(std::move(fptr->body));
    return func;
  };
  return CreateFunctionPass(pass_func, 0, "s_tir.CanonicalizeLoop");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.CanonicalizeLoop", CanonicalizeLoop);
}

}  // namespace transform

}  // namespace s_tir
}  // namespace tvm
