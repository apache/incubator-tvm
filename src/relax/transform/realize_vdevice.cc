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
 * \file tvm/relax/transform/realize_vdevice.cc
 * \brief Propagate virtual device information.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op/op.h>
#include <tvm/relax/transform.h>

namespace tvm {
namespace relax {

namespace {

class VDeviceLookup {
 public:
  explicit VDeviceLookup(IRModule mod) {
    auto opt_global_info = mod->global_infos.Get("vdevice");
    if (!opt_global_info) return;

    auto downcast_vdevice = [](GlobalInfo info) -> VDevice {
      if (auto vdevice = info.as<VDevice>()) {
        return vdevice.value();
      } else {
        TVM_FFI_THROW(TypeError)
            << "Each item in an IRModule's \"vdevice\" annotation must be a VDevice, "
            << "but instead found item of type " << info->GetTypeKey();
      }
    };

    opt_vdevices_ = opt_global_info.value().Map(downcast_vdevice);
  }

  VDevice operator()(ffi::Optional<Attrs> hint_on_device_attrs) {
    auto attrs = hint_on_device_attrs.as<HintOnDeviceAttrs>();
    TVM_FFI_ICHECK(attrs);
    int32_t device_type = attrs->device_type;
    int32_t device_id = attrs->index;
    auto memory_scope = attrs->memory_scope;

    TVM_FFI_CHECK(opt_vdevices_.has_value(), ValueError)
        << "The target VDevice in the GlobalInfos was not found.";

    auto vdevices = opt_vdevices_.value();
    TVM_FFI_CHECK_GE(device_id, 0, ValueError)
        << "The device id in R.hint_on_device must not be negative";

    for (auto vdevice : vdevices) {
      int dev_type = vdevice->target.value()->GetTargetDeviceType();
      if (dev_type == device_type && vdevice->vdevice_id == device_id &&
          memory_scope == vdevice->memory_scope) {
        return vdevice;
      }
    }
    TVM_FFI_THROW(ValueError)
        << "Expected to find device with type " << device_id << " and id " << device_id
        << ", but no such device was found in the IRModule's \"vdevice\" annotation";
    TVM_FFI_UNREACHABLE();
  }

 private:
  ffi::Optional<ffi::Array<VDevice>> opt_vdevices_ = std::nullopt;
};

class DeviceHintCollector : ExprVisitor {
 public:
  static std::tuple<ffi::Map<Var, VDevice>, ffi::Map<Var, VDevice>> Collect(IRModule mod) {
    DeviceHintCollector visitor{VDeviceLookup(mod)};

    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto func = base_func.as<Function>()) {
        visitor(func.value());
      }
    }

    return {visitor.known_vdevice_, visitor.hint_on_device_inputs_};
  }

 private:
  explicit DeviceHintCollector(VDeviceLookup vdevice_lookup) : vdevice_lookup_(vdevice_lookup) {}

  void VisitExpr_(const FunctionObj* func) override {
    ExprVisitor::VisitExpr_(func);

    std::function<void(Expr, Type)> check_ret_ty = [this, &check_ret_ty](Expr expr, Type ty) {
      // If the function is annotated as returning a tensor on a
      // specific device, then that annotation may be propagated into
      // the returned variable.
      if (auto tensor_info = ty.as<TensorTypeObj>();
          tensor_info && tensor_info->vdevice.has_value()) {
        if (auto opt_var = expr.as<Var>()) {
          auto var = opt_var.value();
          if (!known_vdevice_.count(var)) {
            known_vdevice_.Set(var, tensor_info->vdevice.value());
          }
        }
      }

      // If the function is annotated as returning a tuple of tensors,
      // where some elements of the tuple are tensors that exist on a
      // specific device, then those annotations may be propagated
      // into the corresponding tensor annotations.
      if (auto tuple_info = ty.as<TupleTypeObj>()) {
        // The returned tuple is not necessarily an in-line tuple.  In
        // order to find the variables that are bound to the
        // individual tuple elements, we may need to unwrap the
        // variable bindings in order to find the tuple itself.  This
        // unwrapping is not required for the tensor case, as it would
        // already be handled when propagating VDevice across variable
        // definitions.
        while (auto bound_value = LookupBinding(expr)) {
          expr = bound_value.value();
        }

        // Even after unwrapping variable bindings, the resulting
        // expression is not required to be a tuple literal.  For
        // example, the function may return one of its arguments as an
        // output, or may return the result of a `relax::Call` that
        // produces a tuple of outputs.
        if (auto tuple = expr.as<TupleObj>()) {
          TVM_FFI_CHECK_EQ(tuple_info->fields.size(), tuple->fields.size(), ValueError)
              << "Function returns a tuple with " << tuple->fields.size() << " elements, "
              << "but is annotated as returning a tuple with " << tuple_info->fields.size()
              << " elements";
          for (size_t i = 0; i < tuple->fields.size(); i++) {
            check_ret_ty(tuple->fields[i], tuple_info->fields[i]);
          }
        }
      }
    };

    check_ret_ty(func->body->body, func->ret_ty);
  }

  void VisitVarDef(const Var& var) override {
    if (auto tinfo = var->ty.as<TensorTypeObj>(); tinfo && tinfo->vdevice.has_value()) {
      known_vdevice_.Set(var, tinfo->vdevice.value());
    }
    ExprVisitor::VisitVarDef(var);
  }

  void VisitBinding(const Binding& binding) override {
    ExprVisitor::VisitBinding(binding);
    binding_lookup_.Set(binding->var, GetBoundValue(binding));
  }

  void VisitBinding_(const VarBindingObj* binding, const CallObj* call) override {
    ExprVisitor::VisitBinding_(binding, call);
    if (call->op.same_as(hint_on_device_op_)) {
      auto vdevice = vdevice_lookup_(call->attrs);
      known_vdevice_.Set(binding->var, vdevice);

      TVM_FFI_ICHECK_EQ(call->args.size(), 1);
      if (auto arg_var = call->args[0].as<Var>()) {
        hint_on_device_inputs_.Set(arg_var.value(), vdevice);
      }
    }
  }

  ffi::Optional<Expr> LookupBinding(const Expr& expr) const {
    if (auto var = expr.as<Var>()) {
      if (auto bound = binding_lookup_.Get(var.value())) {
        return bound.value();
      }
    }
    return std::nullopt;
  }

  // A lookup to identify the VDevice from the IRModule attributes,
  // given the device type and device id from the R.hint_on_device
  // attributes.
  VDeviceLookup vdevice_lookup_;

  // A lookup of variable bindings, used to unwrap the variable
  // bindings in functions that return a tuple.
  ffi::Map<Var, Expr> binding_lookup_;

  // A map from Var to the VDevice they are known to occur on.  This
  // only contains variables whose location is explicitly known
  // (e.g. output of `R.hint_on_device`, variables with explicit
  // `VDevice` in their type), and does not include variables
  // whose location is (e.g. input of `R.hint_on_device`).
  ffi::Map<Var, VDevice> known_vdevice_;

  // A map from Var to the VDevice they are expected to occur on.  If
  // a variable appears in both `known_vdevice_` and
  // `hint_on_device_inputs_`, then `known_vdevice_` takes priority.
  //
  // For example, `B = R.hint_on_device(A, tvm.cuda(0))` implies that
  // `B` must be located on "cuda:0".  However, `A` may already have a
  // `VDevice` annotation, or may be the output of `R.to_device`.
  // Therefore, we only determine that `A` is located on "cuda:0" if
  // no other annotation has already provided a known location for
  // `A`.
  ffi::Map<Var, VDevice> hint_on_device_inputs_;

  // The `R.hint_on_device` operator.
  const Op hint_on_device_op_ = Op::Get("relax.hint_on_device");
};

// Utility to determine which Var instances must be located on the
// same VDevice.
class VDeviceSetCollector : ExprVisitor {
 public:
  static ffi::Map<Var, ffi::Array<Var>> Collect(IRModule mod) {
    VDeviceSetCollector visitor;
    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto func = base_func.as<Function>()) {
        visitor(func.value());
      }
    }
    return visitor.var_to_co_located_vars_;
  }

 private:
  void VisitBinding(const Binding& binding) override {
    auto cached = current_binding_;
    current_binding_ = binding->var;
    ExprVisitor::VisitBinding(binding);
    current_binding_ = cached;
  }

  void VisitExpr_(const CallObj* call) override {
    if (!call->op.same_as(to_vdevice_op_) && !call->op.same_as(hint_on_device_op_)) {
      ExprVisitor::VisitExpr_(call);
    }
  }

  void VisitExprDepTypeField(const Type&) final {}

  void VisitExpr_(const VarObj* op) override {
    if (current_binding_ && !GetType(current_binding_.value()).as<PrimTypeObj>()) {
      auto var = ffi::GetRef<Var>(op);
      if (GetType(var).as<PrimTypeObj>()) {
        return;
      }
      var_to_co_located_vars_[current_binding_.value()].push_back(var);
      var_to_co_located_vars_[var].push_back(current_binding_.value());
    }
  }

  ffi::Optional<Var> current_binding_ = std::nullopt;

  // Lookup from relax variable to the set of relax variables which
  // must be located on the same device.  For example, a trivial
  // binding `B = A` implies that both `B` and `A` are on the same
  // device.  Similarly, `C = R.add(A,B)` implies that `A`, `B`, and
  // `C` are all on the same device.
  //
  // In general, variables that are used as part of the same
  // `relax::Call` operation must be located on the same device, with
  // the exception of `R.hint_on_device` and `R.to_vdevice`, which may
  // introduce a transfer across devices.
  std::unordered_map<Var, ffi::Array<Var>> var_to_co_located_vars_;

  const Op hint_on_device_op_ = Op::Get("relax.hint_on_device");
  const Op to_vdevice_op_ = Op::Get("relax.to_vdevice");
};

ffi::Map<Var, VDevice> InferVDevice(IRModule mod) {
  auto [explicit_annotations, hint_on_device_args] = DeviceHintCollector::Collect(mod);

  auto co_located_var_lookup = VDeviceSetCollector::Collect(mod);

  ffi::Map<Var, VDevice> known_vdevice;
  std::vector<Var> to_visit;

  // A helper function to propagate all `known_vdevice` entries based
  // on the connections in `co_located_var_lookup`.
  auto propagate = [&]() {
    while (to_visit.size()) {
      Var visiting = to_visit.back();
      to_visit.pop_back();

      if (auto upstream_vars = co_located_var_lookup.Get(visiting)) {
        auto vdevice = known_vdevice.at(visiting);
        for (Var upstream_var : upstream_vars.value()) {
          if (!known_vdevice.count(upstream_var)) {
            known_vdevice.Set(upstream_var, vdevice);
            to_visit.push_back(upstream_var);
          }
        }
      }
    }
  };

  // First round, mark variables whose vdevice is explicitly known
  // (e.g. the output of R.hint_on_device), and propagate.
  for (const auto& [var, vdevice] : explicit_annotations) {
    to_visit.push_back(var);
    known_vdevice.Set(var, vdevice);
  }
  propagate();

  // Second round, mark variables whose vdevice is hinted at (e.g. the
  // input of R.hint_on_device), and propagate.
  for (const auto& [var, vdevice] : hint_on_device_args) {
    if (!known_vdevice.count(var)) {
      to_visit.push_back(var);
      known_vdevice.Set(var, vdevice);
    }
  }
  propagate();

  return known_vdevice;
}

// Update the module to include the inferred VDevice annotations.
class VDeviceTypeUpdater : ExprMutator {
 public:
  static IRModule Apply(IRModule mod, ffi::Map<Var, VDevice> vdevice_map) {
    VDeviceTypeUpdater mutator(VDeviceLookup(mod), vdevice_map);

    IRModule updates;

    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto func = base_func.as<Function>()) {
        auto updated = mutator(func.value()).as_or_throw<Function>();
        if (!updated.same_as(base_func)) {
          updates->Add(gvar, updated);
        }
      }
    }

    if (updates->functions.size()) {
      mod.CopyOnWrite()->Update(updates);
    }

    return mod;
  }

 private:
  VDeviceTypeUpdater(VDeviceLookup vdevice_lookup, ffi::Map<Var, VDevice> vdevice_map)
      : vdevice_lookup_(vdevice_lookup), vdevice_map_(vdevice_map) {}

  Var VisitVarDef(const Var& old_var) override {
    auto var = ExprMutator::VisitVarDef(old_var);
    if (auto tinfo = var->ty.as<TensorTypeObj>()) {
      if (auto opt = vdevice_map_.Get(old_var)) {
        auto vdevice = opt.value();
        TensorType new_ty = [&]() {
          if (tinfo->shape.has_value()) {
            return TensorType(tinfo->shape.value(), tinfo->dtype, vdevice, tinfo->loc);
          } else {
            return TensorType(tinfo->dtype, tinfo->ndim, vdevice, tinfo->loc);
          }
        }();

        if (var->IsInstance<DataflowVarObj>()) {
          var = DataflowVar(var->name, new_ty, var->loc);
        } else {
          var = Var(var->name, new_ty, var->loc);
        }
      }
    }

    return var;
  }

  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const CallObj* op) override {
    auto call = ExprMutator::VisitExpr_(op).as_or_throw<Call>();

    if (!call->op.same_as(hint_on_device_op_)) {
      return call;
    }

    TVM_FFI_ICHECK_EQ(call->args.size(), 1);
    auto arg = call->args[0];
    auto input_vdevice = arg->ty.as_or_throw<TensorType>()->vdevice;
    auto output_vdevice = vdevice_lookup_(call->attrs);

    if (input_vdevice.has_value() && input_vdevice.value() == output_vdevice) {
      return arg;
    } else {
      ffi::ObjectPtr<ToVDeviceAttrs> attrs = ffi::make_object<ToVDeviceAttrs>(output_vdevice);
      return Call(Type::Missing(), to_vdevice_op_, {arg}, Attrs(attrs), {});
    }
  }

  VDeviceLookup vdevice_lookup_;
  ffi::Map<Var, VDevice> vdevice_map_;
  const Op hint_on_device_op_ = Op::Get("relax.hint_on_device");
  const Op to_vdevice_op_ = Op::Get("relax.to_vdevice");
};
}  // namespace

namespace transform {

Pass RealizeVDevice() {
  auto pass_func = [=](IRModule mod, PassContext pc) {
    auto known_vdevices = InferVDevice(mod);
    return VDeviceTypeUpdater::Apply(mod, known_vdevices);
  };
  return CreateModulePass(/*pass_function=*/pass_func,
                          /*opt_level=*/0,
                          /*pass_name=*/"RealizeVDevice");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.RealizeVDevice", RealizeVDevice);
}

}  // namespace transform
}  // namespace relax
}  // namespace tvm
