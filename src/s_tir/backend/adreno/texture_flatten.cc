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
 * \file texture_flatten.cc
 * \brief Flattens texture storage from multi-dimensional array
 * to 2D (width, height, depth) array access
 */

#include <tvm/backend/opencl/op/memory.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/s_tir/backend/adreno/transform.h>
#include <tvm/s_tir/stmt_functor_with_constraint_context.h>
#include <tvm/te/operation.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt.h>

#include <unordered_map>

#include "texture_utils.h"

namespace tvm {
namespace s_tir {
namespace backend {
namespace adreno {
using namespace tvm::tirx;
using texture::DefaultTextureLayoutSeparator;
using texture::IsTextureStorage;

class TextureLoweringBase : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  explicit TextureLoweringBase(const ffi::Array<Var>& params,
                               StmtExprVisitorWithConstraintContext* bound_analyzer)
      : bound_analyzer_{bound_analyzer} {
    for (const Var& param : params) {
      if (auto tensor = param.as<TensorVar>()) {
        extern_tensor_.insert(tensor.value());
      }
    }
  }

  inline PrimExpr SimplifyOffset(const ffi::Array<PrimExpr>& shape,
                                 const ffi::Array<PrimExpr>& index) const {
    PrimExpr base = IntImm::Int32(0);
    TVM_FFI_ICHECK_EQ(shape.size(), index.size());
    if (index.size() > 0) {
      PrimExpr offset = index[0];
      for (size_t i = 1; i < index.size(); ++i) {
        offset = bound_analyzer_->Simplify(offset * shape[i] + index[i]);
      }
      base = base + offset;
    }
    return base;
  }

 protected:
  runtime::MemoryScope GetStorageScope(const TensorVar& tensor) { return tensor.scope(); }

  // Set of all external input and output tensors
  std::unordered_set<TensorVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> extern_tensor_;
  // Bound analzer
  StmtExprVisitorWithConstraintContext* bound_analyzer_;
};

// Lower Nd storage access to 2d texture access using lowering convention
// specified by the tensors storage scope.
class TextureFlattener : public TextureLoweringBase {
 public:
  using TextureLoweringBase::Mutate;
  using TextureLoweringBase::Mutate_;

  explicit TextureFlattener(const ffi::Array<Var>& params,
                            StmtExprVisitorWithConstraintContext* bound_analyzer)
      : TextureLoweringBase(params, bound_analyzer) {}

  UnchangedOr<Stmt> Mutate_(const TensorStoreObj* op, InplaceMode inplace_mode) final {
    Stmt stmt = StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    op = stmt.as<TensorStoreObj>();
    runtime::MemoryScope storage_scope = GetStorageScope(op->dest.as_or_throw<TensorVar>());
    // Lower to two dimensional access
    if (IsTextureStorage(storage_scope)) {
      ffi::Array<Expr> args = GetTextureAccessArgs(op, op->dest.as_or_throw<TensorVar>());
      args.push_back(op->value);
      stmt = Evaluate(Call(args[0]->ty, tvm::backend::opencl::texture2d_store_op(), args));
    }

    return stmt;
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* op, InplaceMode inplace_mode) final {
    PrimExpr expr =
        StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<PrimExpr>(op));
    op = expr.as<TensorLoadObj>();
    // Lower to two dimensional access
    runtime::MemoryScope storage_scope =
        GetStorageScope(op->source.as_or_throw<tvm::tirx::TensorVar>());
    if (IsTextureStorage(storage_scope)) {
      ffi::Array<Expr> args =
          GetTextureAccessArgs(op, op->source.as_or_throw<tvm::tirx::TensorVar>());
      args.push_back(op->indices.back());
      expr = Call(op->source.as_or_throw<tvm::tirx::TensorVar>().dtype(),
                  tvm::backend::opencl::texture2d_load_op(), args)
                 .as_or_throw<PrimExpr>();
    }

    return expr;
  }

 protected:
  template <typename T>
  ffi::Array<Expr> GetTextureAccessArgs(const T* op, const TensorVar& tensor) {
    ffi::Array<Expr> args;
    if (let_binding_.count(tensor)) {
      args.push_back(let_binding_.at(tensor));
    } else {
      args.push_back(tensor.data());
    }
    ffi::Array<PrimExpr> row_dims, row_indices, col_dims, col_indices, depth_dims, depth_indices;
    size_t axis = DefaultTextureLayoutSeparator(tensor.shape().size(), GetStorageScope(tensor));
    for (size_t i = 0; i < tensor.shape().size() - 1; i++) {
      if (i < (axis - 1)) {
        depth_dims.push_back(tensor.shape()[i]);
        depth_indices.push_back(op->indices[i]);
      } else if (i < axis) {
        col_dims.push_back(tensor.shape()[i]);
        col_indices.push_back(op->indices[i]);
      } else {
        row_dims.push_back(tensor.shape()[i]);
        row_indices.push_back(op->indices[i]);
      }
    }
    PrimExpr row_offset = SimplifyOffset(row_dims, row_indices);
    PrimExpr col_offset = SimplifyOffset(col_dims, col_indices);
    PrimExpr depth_offset = SimplifyOffset(depth_dims, depth_indices);
    PrimExpr channel_size =
        IntImm(PrimType::Int(32, 1),
               tensor.shape().back().as_or_throw<IntImm>()->value * tensor.dtype().bits());
    args.push_back(row_offset);
    args.push_back(col_offset);
    args.push_back(depth_offset);
    args.push_back(channel_size);
    return args;
  }

  // Bindings to new texture vars with texture pointer scope
  std::unordered_map<Var, PrimExpr> let_binding_;
};

Function TextureFlattenHandler(Function func) {
  auto fptr = func.CopyOnWrite();
  auto bound_analyzer = ffi::make_object<StmtExprVisitorWithConstraintContext>();
  bound_analyzer->Visit(fptr->body);
  fptr->body = ffi::make_object<TextureFlattener>(fptr->params, bound_analyzer.get())
                   ->Mutate(fptr->body, InplaceMode::kAllow)
                   .ValueOrUnchanged(std::move(fptr->body));
  return func;
}

namespace transform {

Pass TextureFlatten() {
  auto pass_func = [=](Function f, IRModule m, PassContext ctx) {
    return TextureFlattenHandler(std::move(f));
  };
  return tirx::transform::CreateFunctionPass(pass_func, 0, "s_tir.backend.adreno.TextureFlatten");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.backend.adreno.transform.TextureFlatten", TextureFlatten);
}

}  // namespace transform

}  // namespace adreno
}  // namespace backend
}  // namespace s_tir
}  // namespace tvm
