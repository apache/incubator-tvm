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
 *
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tvm/relax/transform/update_vdevice.cc
 * \brief Update Virtual Device pass.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

namespace tvm {
namespace relax {

class VDeviceMutator : public ExprMutator {
 public:
  VDeviceMutator(const IRModule& mod, VDevice new_vdevice, int64_t index)
      : ExprMutator(mod),
        mod_(mod),
        new_vdevice_(new_vdevice),
        old_vdevice_(mod->global_infos["vdevice"][index].as_or_throw<VDevice>()) {}

  using ExprMutator::VisitExpr_;

  Expr VisitExpr(const Expr& expr) final {
    auto visited_expr = ExprMutator::VisitExpr(expr);
    if (!visited_expr->ty.as<MissingType>().has_value()) {
      auto* tinfo = GetTypeAs<TensorTypeObj>(visited_expr);
      bool unchanged = true;
      if (tinfo != nullptr) {
        if (tinfo->vdevice.has_value()) {
          VDevice cur_vdevice = tinfo->vdevice.value();
          if (cur_vdevice == old_vdevice_) {
            unchanged = false;
          }
        }
      }
      if (!unchanged) {
        if (tinfo->shape.has_value()) {
          visited_expr->ty =
              TensorType(tinfo->shape.value(), tinfo->dtype, new_vdevice_, tinfo->loc);
        } else {
          visited_expr->ty = TensorType(tinfo->dtype, tinfo->ndim, new_vdevice_, tinfo->loc);
        }
      }
    }
    return visited_expr;
  }

  IRModule Run() {
    for (const auto& [gv, func] : mod_->functions) {
      if (func->IsInstance<relax::FunctionObj>()) {
        relax::Function update_func = VisitExpr(func).as_or_throw<Function>();
        builder_->UpdateFunction(gv, update_func);
      }
    }
    ffi::Array<GlobalInfo> new_vdevices;
    for (auto vdev : mod_->global_infos["vdevice"]) {
      if (vdev == old_vdevice_) {
        new_vdevices.push_back(new_vdevice_);
      } else {
        new_vdevices.push_back(vdev);
      }
    }
    IRModule new_mod = builder_->GetContextIRModule();
    new_mod->UpdateGlobalInfo("vdevice", new_vdevices);
    return new_mod;
  }

 private:
  /*! \brief Input IRModule */
  IRModule mod_;
  /*! \brief The new virtual device */
  VDevice new_vdevice_;
  /*! \brief The virtual device to be updated */
  VDevice old_vdevice_;
};

namespace transform {

Pass UpdateVDevice(VDevice new_vdevice, int64_t index) {
  auto pass_func = [=](IRModule m, PassContext pc) {
    return relax::VDeviceMutator(m, new_vdevice, index).Run();
  };
  return CreateModulePass(/*pass_function=*/pass_func,
                          /*opt_level=*/0,
                          /*pass_name=*/"UpdateVDevice");
}
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.UpdateVDevice", UpdateVDevice);
}

}  // namespace transform
}  // namespace relax
}  // namespace tvm
