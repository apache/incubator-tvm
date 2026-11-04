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
 * \file tirx/analysis/calculate_allocated_memory.cc
 * \brief Calculate allocated memory per memory scope required by Functions.
 */
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/transform.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <optional>
#include <unordered_map>

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

runtime::MemoryScope GetStorageScope(const Var& var) {
  auto* ptr = var->ty.as<PtrTypeObj>();
  TVM_FFI_ICHECK(ptr) << "Buffer Var's type annotation must be of PtrType";
  return ptr->scope;
}

/*!
 * \brief Allocation calculator for tensor allocation bindings.
 */
class AllocTensorCalculator : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;

  tvm::ffi::Map<ffi::String, int64_t> operator()(const Function& func) {
    this->Visit(func->body);
    tvm::ffi::Map<ffi::String, int64_t> res;
    for (auto [k, v] : _max_size) {
      res.Set(k.name(), v);
    }
    return res;
  }

 private:
  ffi::Optional<VisitInterrupt> Visit_(const BindObj* op) final {
    if (const auto* call = op->value.as<CallObj>();
        call && call->op.same_as(tirx::alloc_tensor_op())) {
      return DispatchAllocTensor(op, call);
    }
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> DispatchAllocTensor(const BindObj* op, const CallObj* call) {
    tvm::Tuple shape = tvm::Tuple(call->ty_args[0].as_or_throw<TensorType>()->shape);
    DLDataType dtype = call->ty_args[0].as_or_throw<TensorType>()->dtype->dtype;
    runtime::MemoryScope scope = call->ty_args[0].as_or_throw<TensorType>()->scope;
    auto search = _current_size.find(scope);
    if (search == _current_size.end()) {
      _current_size[scope] = 0;
      _max_size[scope] = 0;
    }
    int64_t size = 1;
    for (const Expr& e : shape->fields) {
      if (auto* imm = e.as<IntImmObj>()) {
        size = static_cast<int64_t>(size * imm->value);
      } else {
        size = 0;
        break;
      }
    }
    size *= static_cast<int64_t>(PrimType(dtype).StorageBytes());
    _current_size[scope] += size;
    _max_size[scope] = std::max(_current_size[scope], _max_size[scope]);
    return StmtExprVisitor::Visit_(op);
  }
  ffi::Optional<VisitInterrupt> Visit_(const ForObj* op) override {
    auto snapshot = _current_size;
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
    _current_size = snapshot;
    return std::nullopt;
  }
  ffi::Optional<VisitInterrupt> Visit_(const IfObj* op) override {
    auto snapshot = _current_size;
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
    _current_size = snapshot;
    return std::nullopt;
  }
  ffi::Optional<VisitInterrupt> Visit_(const RegionStmtObj* op) override {
    auto snapshot = _current_size;
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
    _current_size = snapshot;
    return std::nullopt;
  }
  std::unordered_map<runtime::MemoryScope, int64_t, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      _max_size;
  std::unordered_map<runtime::MemoryScope, int64_t, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      _current_size;
};

tvm::ffi::Map<ffi::String, tvm::ffi::Map<ffi::String, int64_t> > CalculateAllocatedBytes(
    const Function& func) {
  tvm::ffi::Map<ffi::String, tvm::ffi::Map<ffi::String, int64_t> > results;
  auto alloc_tensor_result = ffi::make_object<AllocTensorCalculator>()->operator()(func);
  results.Set("main", alloc_tensor_result);
  return results;
}

tvm::ffi::Map<ffi::String, tvm::ffi::Map<ffi::String, int64_t> > CalculateAllocatedBytes(
    const IRModule& mod) {
  tvm::ffi::Map<ffi::String, tvm::ffi::Map<ffi::String, int64_t> > results;
  for (const auto& kv : mod->functions) {
    if (auto function = kv.second.as<tirx::Function>()) {
      ffi::String func_name = kv.first->name_hint;
      auto alloc_tensor_result =
          ffi::make_object<AllocTensorCalculator>()->operator()(function.value());
      results.Set(func_name, alloc_tensor_result);
    }
  }
  return results;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "s_tir.analysis.calculate_allocated_bytes",
      [](ffi::ObjectRef obj) -> tvm::ffi::Map<ffi::String, tvm::ffi::Map<ffi::String, int64_t> > {
        if (auto func = obj.as<Function>()) {
          return CalculateAllocatedBytes(func.value());
        } else if (auto mod = obj.as<IRModule>()) {
          return CalculateAllocatedBytes(mod.value());
        } else {
          TVM_FFI_THROW(TypeError)
              << "Expect the input to be either Function or IRModule, but gets: "
              << obj->GetTypeKey();
          throw;
        }
      });
}

bool VerifyVTCMLimit(const IRModule& mod, int64_t limit) {
  auto all_sizes = CalculateAllocatedBytes(mod);
  for (const auto& kv : all_sizes) {
    auto sizes = kv.second;
    const auto vtcm_allocated = sizes.Get("global.vtcm").value_or(0);
    if (limit > 0 && vtcm_allocated > limit) {
      return false;
    }
  }
  return true;
}

bool VerifyVTCMLimit(const Function& func, int64_t limit) {
  auto sizes = CalculateAllocatedBytes(func)["main"];
  const auto vtcm_allocated = sizes.Get("global.vtcm").value_or(0);
  if (limit > 0 && vtcm_allocated > limit) {
    return false;
  }
  return true;
}

int64_t GetVTCMCapacity(ffi::Optional<Target> target, const tvm::transform::PassContext& pass_ctx) {
  if (!target.has_value()) target = Target::Current(/*allow_not_defined=*/true);
  if (target.has_value() && target.value()->kind->name == "hexagon") {
    auto value = target.value()->GetAttr<int64_t>("vtcm-capacity").value();
    if (value > 0) return value;
  }
  return pass_ctx->GetConfig<int64_t>("tirx.vtcm_capacity").value_or(0);
}

ffi::Array<tvm::transform::Pass> GetVTCMCompactionPasses() {
  auto pass_list = ffi::Array<tvm::transform::Pass>();
  pass_list.push_back(s_tir::transform::LowerInitBlock());
  pass_list.push_back(s_tir::transform::PlanAndUpdateTensorAllocationLocation());
  pass_list.push_back(s_tir::transform::ConvertBlocksToOpaque());
  pass_list.push_back(s_tir::transform::CompactTensorAllocation());
  pass_list.push_back(s_tir::transform::LowerMatchTensor());
  pass_list.push_back(s_tir::transform::InjectSoftwarePipeline());
  pass_list.push_back(s_tir::transform::LowerOpaqueBlock());
  pass_list.push_back(s_tir::transform::LowerThreadBinding());
  pass_list.push_back(tirx::transform::FlattenTensor());
  pass_list.push_back(tirx::transform::StmtSimplify());
  pass_list.push_back(tirx::transform::VectorizeLoop(true));
  pass_list.push_back(tirx::transform::StorageRewrite());
  return pass_list;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.analysis.get_vtcm_compaction_passes",
                        []() { return GetVTCMCompactionPasses(); });
}

namespace transform {

Pass VerifyVTCMLimit(ffi::Optional<Target> default_target) {
  auto pass_func = [=](IRModule mod, PassContext ctx) {
    for (auto kv : mod->functions) {
      if (auto opt = kv.second.as<Function>()) {
        auto func = opt.value();

        std::optional<int64_t> limit = std::nullopt;
        if (auto func_target = func->GetAttr(tvm::attr::kTarget)) {
          limit = GetVTCMCapacity(func_target.value(), ctx);
        } else if (default_target) {
          limit = GetVTCMCapacity(default_target.value(), ctx);
        }

        if (limit.has_value() && limit.value() > 0) {
          auto sizes = CalculateAllocatedBytes(func)["main"];
          const auto vtcm_allocated = sizes.Get("global.vtcm").value_or(0);
          if (vtcm_allocated > limit.value()) {
            TVM_FFI_THROW(RuntimeError)
                << "The global.vtcm memory allocation limit has been exceeded "
                << "(allocated: " << vtcm_allocated << ", limit: " << limit.value() << ").\n"
                << "In function\n"
                << func;
          }
        }
      }
    }
    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "s_tir.VerifyVTCMLimit");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.VerifyVTCMLimit", VerifyVTCMLimit);
}

}  // namespace transform
}  // namespace s_tir
}  // namespace tvm
