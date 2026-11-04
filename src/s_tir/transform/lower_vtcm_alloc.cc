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

#include <tvm/backend/opencl/op/memory.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/transform.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt.h>

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

inline bool IsVtcmStorage(runtime::MemoryScope scope) {
  static const auto vtcm = runtime::MemoryScope::Get("global.vtcm");
  return scope == vtcm;
}

class VtcmAllocator : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  VtcmAllocator() {}

  UnchangedOr<Stmt> Mutate_(const BindObj* op, InplaceMode inplace_mode) final {
    if (const auto* call = op->value.as<CallObj>();
        call && call->op.same_as(tirx::alloc_tensor_op())) {
      return Mutate_AllocTensor(op, call, inplace_mode);
    }
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Stmt> Mutate_AllocTensor(const BindObj* op, const CallObj* call,
                                       InplaceMode inplace_mode) {
    runtime::MemoryScope scope = call->ty_args[0].as_or_throw<TensorType>()->scope;
    if (IsVtcmStorage(scope)) {
      tvm::Tuple shape = tvm::Tuple(call->ty_args[0].as_or_throw<TensorType>()->shape);
      ffi::Array<Expr> args;
      args.push_back(StringImm(scope.name()));
      args.push_back(IntImm::Int64(shape->fields.size()));
      args.push_back(Call(PtrType(PrimType::Int(64)), tirx::stack_make_shape_op(), shape->fields));
      TensorVar tensor = op->var.as_or_throw<TensorVar>();
      return Bind(tensor,
                  Call(tensor.ty(), tirx::decl_tensor_op(),
                       {Call(tensor.ty()->DataPtrType(),
                             tvm::backend::opencl::nd_mem_alloc_with_scope_op(), args)},
                       {}, {tensor.ty()}, call->loc),
                  op->loc);
    }
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }

 protected:
  runtime::MemoryScope GetStorageScope(const Var& var) {
    auto* ptr = var->ty.as<PtrTypeObj>();
    TVM_FFI_ICHECK(ptr) << "Buffer Var's type annotation must be of PtrType";
    return ptr->scope;
  }
};

Function LowerVtcmAlloc(Function func) {
  auto fptr = func.CopyOnWrite();
  fptr->body = ffi::make_object<VtcmAllocator>()
                   ->Mutate(fptr->body, InplaceMode::kAllow)
                   .ValueOrUnchanged(std::move(fptr->body));
  return func;
}

namespace transform {

Pass LowerVtcmAlloc() {
  auto pass_func = [=](Function f, IRModule m, PassContext ctx) {
    return s_tir::LowerVtcmAlloc(std::move(f));
  };
  return CreateFunctionPass(pass_func, 0, "s_tir.LowerVtcmAlloc");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.LowerVtcmAlloc", static_cast<Pass (*)()>(LowerVtcmAlloc));
}

}  // namespace transform

}  // namespace s_tir
}  // namespace tvm
