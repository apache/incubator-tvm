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
 * \file src/relax/transform/to_non_dataflow.cc
 * \brief Transform all dataflow structure to non-dataflow version.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/relax/utils.h>

namespace tvm {
namespace relax {
using namespace tvm::prim;

class ToNonDFMutator : public ExprMutator {
 public:
  Var VisitVarDef(const Var& var) final {
    if (var.as<DataflowVarObj>()) {
      Var new_var = Var(var->name, GetType(var), var->loc);
      this->var_remap_.insert_or_assign(var, new_var);
      return new_var;
    }
    return var;
  }

  BindingBlock VisitBindingBlock_(const DataflowBlockObj* block) final {
    builder_->BeginBindingBlock();
    for (Binding binding : block->bindings) {
      this->VisitBinding(binding);
    }
    return builder_->EndBlock();
  }
};

Expr ToNonDataflow(const Expr& e) { return ToNonDFMutator().VisitExpr(e); }

namespace transform {

Pass ToNonDataflow() {
  auto pass_func = [=](Function f, IRModule m, PassContext pc) {
    return ToNonDataflow(f).as_or_throw<Function>();
  };
  return CreateFunctionPass(pass_func, 0, "ToNonDataflow");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.ToNonDataflow", ToNonDataflow);
}

}  // namespace transform

}  // namespace relax
}  // namespace tvm
