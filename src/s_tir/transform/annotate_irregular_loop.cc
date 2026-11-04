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

#include <tvm/ffi/cast.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/transform.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/transform.h>
#include <tvm/tirx/stmt.h>

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

class IrregularLoopAnnotator : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) override {
    if (value.as<ExprObj>()) return ffi::Unchanged();
    return StmtExprMutator::Mutate(value, inplace_mode);
  }

  static Stmt Annotate(const Stmt& body) {
    return ffi::make_object<IrregularLoopAnnotator>()->Mutate(body).ValueOrUnchanged(body);
  }

  IrregularLoopAnnotator() = default;

 private:
  UnchangedOr<Stmt> Mutate_(const ForObj* op, InplaceMode inplace_mode) final {
    bool cur_has_jump = has_jump_;
    has_jump_ = false;
    For res = StmtExprMutator::Mutate_(op, inplace_mode)
                  .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                  .as_or_throw<For>();
    if (has_jump_) {
      TVM_FFI_ICHECK(op->kind == ForKind::kDefault)
          << "Loop kind " << op->kind << " is invalid for irregular loop " << op->var;
      for (const char* key :
           {tvm::tirx::attr::kAutoUnrollMaxStep.name, tvm::tirx::attr::kUnrollExplicit.name,
            tvm::s_tir::attr::kLoopPartitionHint.name,
            tvm::s_tir::attr::kSoftwarePipelineStage.name}) {
        TVM_FFI_ICHECK(!res->annotations.count(key))
            << "Annotation `" << key << "` is invalid for irregular loop " << op->var;
      }
      res.CopyOnWrite()->annotations.Set(tvm::s_tir::attr::kIrregularLoopMark.name, 1);
    }
    std::swap(cur_has_jump, has_jump_);
    return res;
  }

  UnchangedOr<Stmt> Mutate_(const WhileObj* op, InplaceMode inplace_mode) final {
    bool cur_has_jump = has_jump_;
    has_jump_ = false;
    Stmt res = StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    std::swap(cur_has_jump, has_jump_);
    return res;
  }

  UnchangedOr<Stmt> Mutate_(const BreakObj* op, InplaceMode inplace_mode) final {
    has_jump_ = true;
    return ffi::Unchanged();
  }

  UnchangedOr<Stmt> Mutate_(const ContinueObj* op, InplaceMode inplace_mode) final {
    has_jump_ = true;
    return ffi::Unchanged();
  }

  bool has_jump_{false};
};

namespace transform {

Pass AnnotateIrregularLoop() {
  auto pass_func = [](Function func, IRModule mod, PassContext ctx) -> Function {
    if (!func->body.has_value()) return func;
    func.CopyOnWrite()->body = IrregularLoopAnnotator::Annotate(func->body.value());
    return func;
  };

  return CreateFunctionPass(pass_func, 0, "s_tir.AnnotateIrregularLoop");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.AnnotateIrregularLoop", AnnotateIrregularLoop);
}

}  // namespace transform
}  // namespace s_tir
}  // namespace tvm
