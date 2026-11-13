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
 * \file src/relax/transform/rewrite_dataflow_reshape.cc
 * \brief Transform all reshape within dataflow block to a relax.reshape operator
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/function.h>

#include <vector>

#include "../op/tensor/manipulate.h"

namespace tvm {
namespace relax {

std::vector<size_t> GetUsedTensorArgIndices(const tirx::Function& fn, size_t num_args) {
  std::vector<size_t> indices;
  for (size_t i = 0; i < num_args; ++i) {
    if (auto tensor = fn->params[i].as<tirx::TensorVar>()) {
      tvm::Var tensor_var = tensor.value();
      auto walkfn = [=](const tvm::Var& var) -> ffi::Expected<ffi::WalkResult> {
        return var.get() == tensor_var.get() ? ffi::WalkResult::Interrupt(ffi::VisitInterrupt(var))
                                             : ffi::WalkResult::Advance();
      };
      if (ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(fn->body, walkfn).has_value()) {
        indices.push_back(i);
      }
    }
  }
  return indices;
}

class DataflowReshapeRewriter : public ExprMutator {
 public:
  explicit DataflowReshapeRewriter(const IRModule& mod) : mod_(mod) {}

 private:
  using ExprMutator::VisitExpr_;

  BindingBlock VisitBindingBlock(const BindingBlock& block) final {
    // We only rewrite the bindings inside dataflow blocks.
    if (const auto* dataflow_block = block.as<DataflowBlockObj>()) {
      return VisitBindingBlock_(dataflow_block);
    } else {
      return block;
    }
  }

  void VisitBinding_(const VarBindingObj* binding) final {
    // We only rewrite the bindings that are not dataflow output (which means they are not
    // externally referenced)
    if (!binding->var->IsInstance<DataflowVarObj>()) {
      this->builder_->EmitNormalized(ffi::GetRef<VarBinding>(binding));
    } else {
      ExprMutator::VisitBinding_(binding);
    }
  }

  Expr VisitExpr_(const CallObj* call) final {
    static const Op call_tir_op = Op::Get("relax.call_tir");
    if (!call->op.same_as(call_tir_op)) {
      return ffi::GetRef<Call>(call);
    }

    // We bring the calls of reshape tirx::Function back to calls of high-level
    // relax.reshape op, which will be lowered to calls of the ExternFunc
    // vm.builtin.reshape in the LowerRuntimeBuiltin pass.

    auto prim_fn =
        mod_->Lookup(call->args[0].as_or_throw<GlobalVar>()).as_or_throw<tirx::Function>();
    auto arg_tuple = call->args[1].as_or_throw<Tuple>()->fields;
    auto used_tensor_arg_indices = GetUsedTensorArgIndices(prim_fn, arg_tuple.size());

    // The number of inputs to call_tir(reshape, (...)) might not be one, since FuseOps
    // can generate a fused TupleGetItem + reshape function whose input is a tuple. FuseTIR
    // then flattens the tuple input so that the fused TIR reshape function ends up having
    // multiple input tensors. But only one of them should be accessed and reshaped.
    if (used_tensor_arg_indices.size() != 1) {
      return ffi::GetRef<Call>(call);
    }

    auto arg = arg_tuple[used_tensor_arg_indices[0]];

    if (!IsCallingTIRReshape(call, arg)) {
      return ffi::GetRef<Call>(call);
    }

    TensorType res_ty = call->ty.as_or_throw<TensorType>();
    return reshape(arg, res_ty->shape.value());
  }

  bool IsCallingTIRReshape(const CallObj* call, Expr inp) {
    const GlobalVar& global_var = call->args[0].as_or_throw<GlobalVar>();
    const auto* func = mod_->functions.Get(global_var).value().as<tirx::FunctionObj>();
    TVM_FFI_ICHECK_NOTNULL(func);
    if (!HasReshapePattern(ffi::GetRef<tirx::Function>(func))) {
      return false;
    }

    // The reshape operator expects that the number of elements in the source is the same
    // as the number of elements in the result. There are operators that could have a reshape
    // pattern that don't meet this requirement (e.g. strided_slice), and they should not be
    // converted to reshape.
    TVM_FFI_ICHECK(!inp->ty.as<MissingType>().has_value() &&
                   !call->ty.as<MissingType>().has_value());
    TensorType inp_ty = inp->ty.as_or_throw<TensorType>();
    TensorType res_ty = call->ty.as_or_throw<TensorType>();

    if (inp_ty->IsUnknownDtype() || inp_ty->dtype != res_ty->dtype) {
      return false;
    }
    TVM_FFI_ICHECK(inp_ty->shape.has_value() && res_ty->shape.has_value());
    if (inp_ty->IsUnknownNdim() || res_ty->IsUnknownNdim()) {
      return false;
    }
    auto product = [](ffi::Array<PrimExpr> args) -> PrimExpr {
      // Scalar tensors may be empty indicating a single element.
      PrimExpr p = args.empty() ? PrimExpr(1) : args[0];
      for (int i = 1, e = args.size(); i < e; ++i) p *= args[i];
      return p;
    };
    auto inp_count = product(inp_ty->GetShape().value());
    auto res_count = product(res_ty->GetShape().value());
    if (!sym::Analyzer()->CanProveEqual(inp_count, res_count)) {
      return false;
    }

    return true;
  }

  const IRModule& mod_;
};

Expr RewriteDataflowReshape(const Function& f, const IRModule& mod) {
  return DataflowReshapeRewriter(mod)(f);
}

namespace transform {

Pass RewriteDataflowReshape() {
  auto pass_func = [=](Function f, IRModule m, PassContext pc) {
    return RewriteDataflowReshape(f, m).as_or_throw<Function>();
  };
  return CreateFunctionPass(pass_func, 0, "RewriteDataflowReshape");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.RewriteDataflowReshape", RewriteDataflowReshape);
}

}  // namespace transform

}  // namespace relax
}  // namespace tvm
