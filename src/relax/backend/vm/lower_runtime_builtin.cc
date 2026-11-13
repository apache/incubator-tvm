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
 * \file src/relax/backend/vm/lower_runtime_builtin.cc
 * \brief Lowers most builtin functions and packed calls.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/backend.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op/op.h>
#include <tvm/relax/op_attr.h>
#include <tvm/relax/type.h>

namespace tvm {
namespace relax {
using namespace tvm::prim;

// This pass lowers most ops to VM specific builtins.
// TODO(relax-team): revisit after PrimExpr.
class LowerRuntimeBuiltinMutator : public ExprMutator {
 public:
  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const CallObj* call_node) final {
    static const auto& lower_builtin_fmap = Op::GetAttrMap(tvm::relax::op_attr::kLowerBuiltin);
    // post-order mutation
    Call call = VisitExprPostOrder_(call_node).as_or_throw<Call>();

    if (call->op.same_as(call_tir_dyn_op_)) {
      return CallTIRDyn(call);
    } else if (call->op.same_as(reshape_op_)) {
      return Reshape(call);
    } else if (call->op.same_as(shape_of_op_)) {
      return ShapeOf(call);
    } else if (call->op.same_as(tensor_to_shape_op_)) {
      return TensorToShape(call);
    } else if (call->op.same_as(call_py_func_op_)) {
      return CallPyFunc(call);
    } else if (call->op.same_as(to_vdevice_op_)) {
      return ToDevice(call);
    } else if (call->op.same_as(make_closure_op_)) {
      return MakeClosure(call);
    } else if (call->op.same_as(invoke_closure_op_)) {
      return InvokeClosure(call);
    } else if (call->op.same_as(alloc_tensor_op_)) {
      TVM_FFI_THROW(InternalError) << "LowerRuntimeBuiltin encountered " << call->op
                                   << " in expression " << ffi::GetRef<Call>(call_node) << ".  "
                                   << "This operation should have been lowered earlier "
                                   << "using the 'relax.transform.LowerAllocTensor' pass.";
    } else if (call->op.same_as(mem_alloc_storage_op_)) {
      return MakeMemAllocStorage(call);
    } else if (call->op.same_as(mem_alloc_tensor_op_)) {
      return MakeMemAllocTensor(call);
    } else if (call->op.same_as(mem_kill_storage_op_) || call->op.same_as(mem_kill_tensor_op_)) {
      return MakeMemKillObject(call);
    } else if (const auto* op_node = call->op.as<OpObj>()) {
      Op op = ffi::GetRef<Op>(op_node);
      if (lower_builtin_fmap.count(op)) {
        return lower_builtin_fmap[op](builder_, call);
      }
    }
    return call;
  }

  Expr MakeMemAllocStorage(const Call& call) {
    PrimExpr runtime_device_index = call->args[1].as_or_throw<PrimExpr>();
    StringImm storage_scope = call->args[2].as_or_throw<StringImm>();
    DataTypeImm output_dtype = DataTypeImm((DLDataType{kDLUInt, 8, 1}));
    return Call(Type::Missing(), vm_alloc_storage_op_,
                {call->args[0], runtime_device_index, output_dtype, storage_scope}, std::nullopt);
  }

  Expr MakeMemAllocTensor(const Call& call) {
    PrimExpr offset = call->args[1].as_or_throw<PrimExpr>();
    DataTypeImm dtype = call->args[3].as_or_throw<DataTypeImm>();

    ffi::Array<Expr> call_args = {call->args[0], offset, call->args[2], dtype};
    if (5 == call->args.size()) {
      call_args.push_back(call->args[4]);
    }

    return Call(Type::Missing(), vm_alloc_tensor_op_, call_args, std::nullopt);
  }

  Expr MakeMemKillObject(const Call& call) {
    TVM_FFI_ICHECK_EQ(call->args.size(), 1);
    return Call(Type::Missing(), vm_kill_object_op_, {call->args[0]}, std::nullopt);
  }

  Expr CallTIRDyn(const Call& call_node) {
    TVM_FFI_ICHECK(call_node->args.size() == 2);
    TVM_FFI_ICHECK(call_node->args[0]->IsInstance<GlobalVarObj>());
    TVM_FFI_ICHECK(call_node->args[1]->IsInstance<TupleObj>());
    ffi::Array<Expr> args;

    auto tir_args = call_node->args[1].as_or_throw<Tuple>();
    args.push_back(call_node->args[0]);
    for (Expr arg : tir_args->fields) {
      args.push_back(arg);
    }
    return Call(Type::Missing(), builtin_call_tir_dyn_, args, std::nullopt, {void_ty_});
  }

  Expr Reshape(const Call& call_node) {
    TVM_FFI_ICHECK(call_node->args.size() == 2);
    TVM_FFI_ICHECK(!call_node->ty.as<MissingType>().has_value());
    auto arg = call_node->args[1];

    TVM_FFI_CHECK(arg->ty->IsInstance<ShapeTypeObj>(), TypeError)
        << "LowerRuntimeBuiltin expects the shape arg of R.reshape "
        << "to be a ShapeExpr or VarObj bound to a ShapeExpr.  "
        << "However, in expression " << call_node << ", the shape argument " << arg << " has type "
        << arg->ty;

    return Call(Type::Missing(), builtin_reshape_, call_node->args, std::nullopt,
                {GetType(call_node)});
  }

  Expr ShapeOf(const Call& call_node) {
    TVM_FFI_ICHECK(call_node->args.size() == 1);
    TVM_FFI_ICHECK(!call_node->ty.as<MissingType>().has_value());
    return Call(Type::Missing(), builtin_shape_of_, call_node->args, std::nullopt,
                {GetType(call_node)});
  }

  Expr TensorToShape(const Call& call_node) {
    TVM_FFI_ICHECK(call_node->args.size() == 1);
    TVM_FFI_ICHECK(!call_node->ty.as<MissingType>().has_value());

    return Call(Type::Missing(), builtin_tensor_to_shape_, call_node->args, std::nullopt,
                {GetType(call_node)});
  }

  Expr CallPyFunc(const Call& call_node) {
    TVM_FFI_ICHECK(call_node->args.size() == 2);
    TVM_FFI_ICHECK(!call_node->ty.as<MissingType>().has_value());

    // Create tuple with function name and arguments tuple
    ffi::Array<Expr> tuple_fields;
    tuple_fields.push_back(call_node->args[0]);  // function name
    tuple_fields.push_back(call_node->args[1]);  // arguments tuple
    auto combined_tuple = Tuple(tuple_fields);

    // Direct call to vm.builtin.call_py_func
    return Call(Type::Missing(), builtin_call_py_func_, {combined_tuple}, call_node->attrs,
                call_node->ty_args, call_node->loc);
  }

  Expr ToDevice(const Call& call_node) {
    // TODO(yongwww): replace ToVDeviceAttrs with related Expr
    TVM_FFI_ICHECK(call_node->args.size() == 1);
    TVM_FFI_ICHECK(!call_node->ty.as<MissingType>().has_value());
    auto attrs = call_node->attrs.as<ToVDeviceAttrs>();
    ffi::Array<Expr> args;
    args.push_back(call_node->args[0]);
    // Get the DLDeviceType and device_id from VDevice
    VDevice vdev = attrs->dst_vdevice;
    int dev_type = vdev->target.value()->GetTargetDeviceType();
    int dev_id = vdev->vdevice_id;
    StringImm storage_scope = StringImm(vdev->memory_scope.name());
    args.push_back(IntImm::Int64(dev_type));
    args.push_back(IntImm::Int64(dev_id));
    args.push_back(storage_scope);
    return Call(Type::Missing(), builtin_to_device_, args, call_node->attrs, {GetType(call_node)});
  }

  Expr MakeClosure(const Call& call_node) {
    TVM_FFI_ICHECK(call_node->args.size() == 2);
    TVM_FFI_ICHECK(call_node->args[0]->IsInstance<GlobalVarObj>());
    TVM_FFI_ICHECK(call_node->args[1]->IsInstance<TupleObj>());

    ffi::Array<Expr> args;
    auto func = call_node->args[0];
    auto closure_args = call_node->args[1].as_or_throw<Tuple>();

    args.push_back(func);
    for (Expr arg : closure_args->fields) {
      args.push_back(arg);
    }

    return Call(Type::Missing(), builtin_make_closure_, args, std::nullopt, {object_ty_});
  }

  Expr InvokeClosure(const Call& call_node) {
    TVM_FFI_ICHECK(call_node->args.size() == 2);
    TVM_FFI_ICHECK(call_node->args[0]->IsInstance<VarObj>());
    TVM_FFI_ICHECK(call_node->args[1]->IsInstance<TupleObj>());

    ffi::Array<Expr> args;

    args.push_back(call_node->args[0]);

    // args for the invoke_closure
    auto invoke_closure_args = call_node->args[1].as_or_throw<Tuple>();
    for (Expr arg : invoke_closure_args->fields) {
      args.push_back(arg);
    }
    return Call(Type::Missing(), call_builtin_with_ctx_op_, {builtin_invoke_closure_, Tuple(args)},
                std::nullopt, {object_ty_});
  }

  const Op call_builtin_with_ctx_op_ = Op::Get("relax.call_builtin_with_ctx");
  const Type object_ty_ = AnyType();
  const Type void_ty_ = TupleType(ffi::Array<Type>({}));
  // object to pattern match.
  const Op call_tir_dyn_op_ = Op::Get("relax.vm.call_tir_dyn");
  const Op reshape_op_ = Op::Get("relax.reshape");
  const Op shape_of_op_ = Op::Get("relax.shape_of");
  const Op tensor_to_shape_op_ = Op::Get("relax.tensor_to_shape");
  const Op call_py_func_op_ = Op::Get("relax.call_py_func");
  const Op to_vdevice_op_ = Op::Get("relax.to_vdevice");
  const Op make_closure_op_ = Op::Get("relax.make_closure");
  const Op invoke_closure_op_ = Op::Get("relax.invoke_closure");
  const Op alloc_tensor_op_ = Op::Get("relax.builtin.alloc_tensor");
  const Op mem_alloc_storage_op_ = Op::Get("relax.memory.alloc_storage");
  const Op mem_alloc_tensor_op_ = Op::Get("relax.memory.alloc_tensor");
  const Op mem_kill_storage_op_ = Op::Get("relax.memory.kill_storage");
  const Op mem_kill_tensor_op_ = Op::Get("relax.memory.kill_tensor");
  // functions to lower to
  const Op vm_alloc_storage_op_ = Op::Get("relax.vm.alloc_storage");
  const Op vm_alloc_tensor_op_ = Op::Get("relax.vm.alloc_tensor");
  const Op vm_kill_object_op_ = Op::Get("relax.vm.kill_object");
  // Function to compute allocated shape.
  const ExternFunc builtin_compute_alloc_shape_{"vm.builtin.compute_alloc_shape"};
  const ExternFunc builtin_call_tir_dyn_{"vm.builtin.call_tir_dyn"};
  const ExternFunc builtin_reshape_{"vm.builtin.reshape"};
  const ExternFunc builtin_shape_of_{"vm.builtin.shape_of"};
  const ExternFunc builtin_tensor_to_shape_{"vm.builtin.tensor_to_shape"};
  const ExternFunc builtin_call_py_func_{"vm.builtin.call_py_func"};
  const ExternFunc builtin_to_device_{"vm.builtin.to_device"};
  const ExternFunc builtin_make_closure_{"vm.builtin.make_closure"};
  const ExternFunc builtin_invoke_closure_{"vm.builtin.invoke_closure"};
};

Expr LowerRuntimeBuiltin(const Expr& e) { return LowerRuntimeBuiltinMutator().VisitExpr(e); }

namespace transform {

Pass LowerRuntimeBuiltin() {
  auto pass_func = [=](Function f, IRModule m, PassContext pc) {
    return LowerRuntimeBuiltin(f).as_or_throw<Function>();
  };
  return CreateFunctionPass(pass_func, 0, "LowerRuntimeBuiltin");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.LowerRuntimeBuiltin", LowerRuntimeBuiltin);
}

}  // namespace transform
}  // namespace relax
}  // namespace tvm
