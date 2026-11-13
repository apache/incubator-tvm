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

#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

namespace tvm {
namespace relax {
namespace testing {

class EmptyCppMutator : public relax::ExprMutator {};

tvm::transform::Pass ApplyEmptyCppMutator() {
  auto pass_func = [](Function func, IRModule, tvm::transform::PassContext) -> Function {
    EmptyCppMutator mutator;
    return mutator.VisitExpr(std::move(func)).as_or_throw<Function>();
  };
  return tvm::relax::transform::CreateFunctionPass(pass_func, 0,
                                                   "relax.testing.ApplyEmptyCppMutator");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.testing.transform.ApplyEmptyCppMutator", ApplyEmptyCppMutator);
}

}  // namespace testing
}  // namespace relax
}  // namespace tvm
