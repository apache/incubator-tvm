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

/*!
 * \file inject_texture_alloc.cc
 */

#include <tvm/backend/opencl/op/memory.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/backend/adreno/transform.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/stmt_functor_with_constraint_context.h>
#include <tvm/sym/iter_affine_map.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/memory.h>

#include "texture_utils.h"

namespace tvm {
namespace s_tir {
namespace backend {
namespace adreno {
using namespace tvm::tirx;
using texture::ApplyTexture2DFlattening;
using texture::DefaultTextureLayoutSeparator;
using texture::IsTextureStorage;

/*!
 * \brief Inject Texture Alloc Intrinsic right after tensor allocations are realized.
 */
class TextureAllocInjector : public s_tir::StmtExprMutatorWithConstraintContext {
 public:
  using s_tir::StmtExprMutatorWithConstraintContext::Mutate;
  using s_tir::StmtExprMutatorWithConstraintContext::Mutate_;

  static Function Inject(Function func) {
    sym::Analyzer ana;
    auto pass = ffi::make_object<TextureAllocInjector>(ana);
    auto writer = func.CopyOnWrite();
    pass->AddParamConstraints(func);
    writer->body = pass->Mutate(func->body)
                       .as_or_throw<UnchangedOr<ffi::Optional<SeqStmt>>>()
                       .ValueOrUnchanged(func->body);
    return func;
  }

  explicit TextureAllocInjector(const sym::Analyzer& ana)
      : StmtExprMutatorWithConstraintContext(ana->GetConstraintContext()) {}

 private:
  UnchangedOr<Stmt> Mutate_(const BindObj* op, InplaceMode inplace_mode) final {
    if (const auto* call = op->value.as<CallObj>();
        call && call->op.same_as(tirx::alloc_tensor_op())) {
      return Mutate_AllocTensor(op, call, inplace_mode);
    }
    return StmtExprMutatorWithConstraintContext::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Stmt> Mutate_AllocTensor(const BindObj* op, const CallObj* call,
                                       InplaceMode inplace_mode) {
    Stmt stmt = StmtExprMutatorWithConstraintContext::Mutate_(op, inplace_mode)
                    .ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    runtime::MemoryScope scope = call->ty_args[0].as_or_throw<TensorType>()->scope;
    if (IsTextureStorage(scope)) {
      op = stmt.as<BindObj>();
      if (const auto* call = op ? op->value.as<CallObj>() : nullptr;
          !call || !call->op.same_as(tirx::alloc_tensor_op())) {
        TVM_FFI_THROW(InternalError) << "Expected an allocation binding after tensor mutation";
      }
      const auto* allocation = op->value.as<CallObj>();
      tvm::Tuple shape = tvm::Tuple(allocation->ty_args[0].as_or_throw<tirx::TensorType>()->shape);
      DLDataType dtype = allocation->ty_args[0].as_or_throw<tirx::TensorType>()->dtype->dtype;
      ffi::Array<PrimExpr> extents =
          shape->fields.Map([](const Expr& e) { return e.as_or_throw<PrimExpr>(); });
      TVM_FFI_ICHECK(extents.size() >= 3) << "Only 2D Array RGBA texture is currently supported";
      const int data_bits = dtype.bits,
                vec_length = extents.back().as<IntImmObj>()->value.as<int>().value();
      const int channel_size = data_bits * vec_length;
      TVM_FFI_ICHECK(channel_size == 128 || channel_size == 64)
          << "Invalid Channel Size: " << channel_size << " bits";

      size_t axis = DefaultTextureLayoutSeparator(extents.size(), scope);
      auto texture = ApplyTexture2DFlattening<PrimExpr>(extents, extents.size(), axis);
      ffi::Array<Expr> args;
      args.push_back(StringImm(scope.name()));
      args.push_back(IntImm::Int64(3));
      args.push_back(Call(PtrType(PrimType::Int(64)), tirx::stack_make_shape_op(),
                          {texture.width, texture.height, texture.depth}));
      args.push_back(IntImm::Int64(channel_size));
      stmt = Bind(op->var.as_or_throw<TensorVar>(),
                  Call(op->var.as_or_throw<TensorVar>().ty(), tirx::decl_tensor_op(),
                       {Call(op->var.as_or_throw<TensorVar>().ty()->DataPtrType(),
                             tvm::backend::opencl::nd_mem_alloc_with_scope_op(), args)},
                       {}, {op->var.as_or_throw<TensorVar>().ty()}, allocation->loc),
                  op->loc);
    }
    return stmt;
  }

 protected:
  runtime::MemoryScope GetStorageScope(const Var& buffer_var) {
    auto* ptr = buffer_var->ty.as<PtrTypeObj>();
    TVM_FFI_ICHECK(ptr) << "Buffer Var's type annotation must be of PtrType";
    return ptr->scope;
  }
};

namespace transform {

Pass InjectTextureAlloc() {
  auto pass_func = [=](Function f, IRModule m, PassContext ctx) {
    return TextureAllocInjector::Inject(std::move(f));
  };
  return tirx::transform::CreateFunctionPass(pass_func, 0,
                                             "s_tir.backend.adreno.InjectTextureAlloc");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.backend.adreno.transform.InjectTextureAlloc", InjectTextureAlloc);
}

}  // namespace transform

}  // namespace adreno
}  // namespace backend
}  // namespace s_tir
}  // namespace tvm
