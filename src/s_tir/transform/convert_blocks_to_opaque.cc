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
 * \file convert_block_to_opaque.cc
 * \brief Convert the blocks to opaque blocks which do not have block vars.
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/transform.h>

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

/*!
 * \brief Substitute expr via BlockRealize value bindings and convert each block into opaque
 *        blocks.
 */
class OpaqueBlockConverter : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  static ffi::Optional<SeqStmt> Convert(const Function& f) {
    auto substituter = ffi::make_object<OpaqueBlockConverter>();
    return substituter->Mutate(f->body).ValueOrUnchanged(f->body);
  }

  OpaqueBlockConverter() = default;

 private:
  UnchangedOr<Expr> Mutate_(const VarObj* var, InplaceMode inplace_mode) final {
    if (def_region_kind() == kTVMFFIDefRegionKindNone) {
      TVM_FFI_ICHECK(!forbidden_iter_vars_.count(var))
          << "Variable " << var->name << " occurs in the predicate or iter_values of a block, "
          << "but isn't defined until the body of the block";
    }
    return StmtExprMutator::Mutate_(var, inplace_mode);
  }

  UnchangedOr<Stmt> Mutate_(const SBlockObj* block, InplaceMode inplace_mode) final {
    TVM_FFI_ICHECK(!block->init.has_value())
        << "Block Init part is not allowed in pass ConvertBlocksToOpaque";
    auto node = ffi::make_object<SBlockObj>(*block);
    node->iter_vars.clear();
    return StmtExprMutator::Mutate_(node.get(), inplace_mode)
        .ValueOrUnchanged(SBlock(std::move(node)));
  }

  UnchangedOr<Stmt> Mutate_(const SBlockRealizeObj* realize, InplaceMode inplace_mode) final {
    const auto* block_op = realize->block.get();
    TVM_FFI_ICHECK(!block_op->init.has_value());

    // Step 1. Visit the predicate and iter_values, without any variable bindings
    for (const auto& iter : block_op->iter_vars) forbidden_iter_vars_.insert(iter->var.get());
    auto predicate_result = Mutate(realize->predicate, inplace_mode);
    bool predicate_unchanged = predicate_result.UnchangedOrSameAs(realize->predicate);
    PrimExpr predicate = std::move(predicate_result).ValueOrUnchanged(realize->predicate);
    ffi::Array<PrimExpr> iter_values = Mutate(ffi::AnyView(realize->iter_values), inplace_mode)
                                           .ValueOrUnchanged(realize->iter_values)
                                           .as_or_throw<ffi::Array<PrimExpr>>();
    for (const auto& iter : block_op->iter_vars) forbidden_iter_vars_.erase(iter->var.get());

    // Step 2. Update "block vars => binding values" for substitution.
    TVM_FFI_ICHECK_EQ(block_op->iter_vars.size(), iter_values.size());
    for (int i = 0, n = block_op->iter_vars.size(); i < n; ++i) {
      IterVar block_var = block_op->iter_vars[i];
      PrimExpr value = iter_values[i];
      PrimExpr v = Mutate(value).ValueOrUnchanged(value);
      VarRemapSet(block_var->var, v);
    }
    // Step 3. Visit recursively.
    auto new_block_result = Mutate(realize->block, inplace_mode);
    bool new_block_unchanged = new_block_result.UnchangedOrSameAs(realize->block);
    SBlock new_block =
        std::move(new_block_result).ValueOrUnchanged(realize->block).as_or_throw<SBlock>();

    // Step 4. Return
    if (predicate_unchanged && iter_values.same_as(realize->iter_values) && new_block_unchanged &&
        realize->iter_values.size() == 0) {
      return ffi::Unchanged();
    } else {
      return SBlockRealize({}, predicate, new_block);
    }
  }

  /* \brief Variables that may not occur in the current context */
  std::unordered_set<const VarObj*> forbidden_iter_vars_;
};

namespace transform {

Pass ConvertBlocksToOpaque() {
  auto pass_func = [=](Function f, IRModule m, PassContext ctx) {
    FunctionObj* fptr = f.CopyOnWrite();
    fptr->body = OpaqueBlockConverter::Convert(f);
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "s_tir.ConvertBlocksToOpaque");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.ConvertBlocksToOpaque", ConvertBlocksToOpaque);
}
}  // namespace transform

}  // namespace s_tir
}  // namespace tvm
