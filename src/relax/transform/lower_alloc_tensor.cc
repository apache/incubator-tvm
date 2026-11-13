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
 * \file src/relax/transform/lower_alloc_tensor.cc
 * \brief Lower any relax.builtin.alloc_tensor remaining after static planning
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/tirx/op/memory.h>

#include "utils.h"

namespace tvm {
namespace relax {
using namespace tvm::prim;

namespace {
class Mutator : public ExprMutator {
 public:
  explicit Mutator(IRModule mod) : ctx_mod_(mod) {}

  using ExprMutator::VisitExpr_;
  Expr VisitExpr_(const CallObj* op) override {
    static const Op alloc_tensor_op = Op::Get("relax.builtin.alloc_tensor");
    static const Op mem_alloc_storage_op = Op::Get("relax.memory.alloc_storage");
    static const Op mem_alloc_tensor_op = Op::Get("relax.memory.alloc_tensor");

    if (op->op.same_as(alloc_tensor_op)) {
      TVM_FFI_ICHECK_EQ(op->args.size(), 4)
          << "Op " << op->op << " should have three arguments, "
          << "[shape, dtype, runtime_device_index, storage_scope].  "
          << "However, received " << ffi::GetRef<Call>(op);

      auto shape_arg = op->args[0];
      auto dtype = op->args[1].as_or_throw<DataTypeImm>();
      PrimExpr runtime_device_index = op->args[2].as_or_throw<PrimExpr>();
      StringImm storage_scope = op->args[3].as_or_throw<StringImm>();

      auto shape = [&]() -> ffi::Array<PrimExpr> {
        if (auto ptr = shape_arg.as<ShapeExprObj>()) {
          return ptr->values;
        }

        auto ty = GetType(shape_arg);
        if (auto ptr = ty.as<ShapeTypeObj>()) {
          if (ptr->values) {
            return ptr->values.value();
          }
        }

        TVM_FFI_THROW(InternalError)
            << "Shape argument for " << alloc_tensor_op << " should be a ShapeExpr, "
            << "or a variable that holds a ShapeExpr.  "
            << "However, received argument " << shape_arg << " with type " << ty;
        TVM_FFI_UNREACHABLE();
      }();

      PrimExpr nbytes = [&]() -> PrimExpr {
        PrimType dtype_ty(dtype->value);
        TVM_FFI_ICHECK(!dtype_ty.IsScalableVector())
            << "Cannot statically compute allocation size for scalable vector dtype " << dtype_ty;
        PrimExpr nbytes = IntImm::Int64(static_cast<int64_t>(dtype_ty.StorageBytes()));
        for (const auto& dim : shape) {
          nbytes *= dim;
        }
        return nbytes;
      }();

      ShapeExpr size({nbytes});

      int64_t vdevice_index = -1;
      if (const auto* int_imm = op->args[2].as<IntImmObj>()) {
        vdevice_index = int_imm->value.as<int>().value();
      }
      ffi::Optional<VDevice> vdevice = GetGlobalVDevice(ctx_mod_, vdevice_index);

      if (vdevice.has_value()) {
        std::string dev_kind = vdevice.value()->target.value()->kind->name;
        PrimExpr dev_size = IntImm::Int64(1);
        if (vdevice.value()->memory_scope != runtime::MemoryScope::Global()) {
          auto device_size_handler =
              tvm::ffi::Function::GetGlobal(std::string("DeviceGetMemSize.") + dev_kind);
          if (device_size_handler.has_value()) {
            dev_size *=
                (*device_size_handler)(shape, dtype->value, vdevice.value()).cast<PrimExpr>();
            size = ShapeExpr({dev_size});
          }
          auto device_scope_handler =
              tvm::ffi::Function::GetGlobal(std::string("DeviceScopeCompatibility.") + dev_kind);
          if (device_scope_handler.has_value()) {
            runtime::MemoryScope dev_scope =
                (*device_scope_handler)(vdevice.value()->target.value(),
                                        vdevice.value()->memory_scope)
                    .cast<runtime::MemoryScope>();
            storage_scope = StringImm(dev_scope.name());
          }
        }
      }

      auto offset = IntImm::Int64(0);

      Expr storage = Call(
          Type::Missing(), mem_alloc_storage_op,
          {size, runtime_device_index, storage_scope, DataTypeImm((DLDataType{kDLUInt, 8, 1}))});
      storage = builder_->Emit(storage, "storage");
      Expr tensor = Call(Type::Missing(), mem_alloc_tensor_op,
                         {storage, offset, shape_arg, dtype, op->args[2]});
      return tensor;
    } else {
      return ExprMutator::VisitExpr_(op);
    }
  }

 private:
  IRModule ctx_mod_;
};
}  // namespace

Expr LowerAllocTensor(IRModule m, Expr expr) {
  Mutator mutator(m);
  return mutator(expr);
}

namespace transform {

Pass LowerAllocTensor() {
  auto pass_func = [=](Function func, IRModule m, PassContext pc) {
    return relax::LowerAllocTensor(m, std::move(func)).as_or_throw<Function>();
  };
  return CreateFunctionPass(pass_func, /*opt_level=*/0, "LowerAllocTensor");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.LowerAllocTensor", LowerAllocTensor);
}

}  // namespace transform
}  // namespace relax
}  // namespace tvm
