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
 * \file domain_touched.cc
 * \brief Analyze tensor domains touched by a statement
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/stmt_functor_with_constraint_context.h>
#include <tvm/sym/int_set.h>
#include <tvm/te/tensor.h>

#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace tvm {
namespace s_tir {

using namespace tirx;
using sym::IntSet;

namespace {

using TensorTouches = std::vector<std::vector<IntSet>>;

struct LoadAccess {
  TensorTouches set;
};

struct StoreAccess {
  TensorTouches set;
};

struct CombinedAccess {
  TensorTouches set;
};

using TensorDomainAccess = std::tuple<LoadAccess, StoreAccess, CombinedAccess>;

}  // namespace

// Find Read region of the tensor in the stmt.
class TensorTouchedDomain final : public s_tir::StmtExprVisitorWithConstraintContext {
 public:
  using s_tir::StmtExprVisitorWithConstraintContext::Visit_;

  std::unordered_map<const VarObj*, TensorDomainAccess>& GetAccessedTensorRegions() {
    return tensor_access_map_;
  }

  ffi::Array<ffi::Optional<Range>> FindUnion(const TensorVar& tensor, bool consider_loads,
                                             bool consider_stores) {
    ffi::Array<ffi::Optional<Range>> ret;
    auto kv = tensor_access_map_.find(tensor.get());
    if (kv == tensor_access_map_.end()) {
      LOG(WARNING) << "[s_tir::TensorDomainTouched] "
                   << "The requested tensor is not contained in the provided stmt body: " << tensor;
      return ret;
    }

    TensorTouches bounds;
    if (consider_loads && consider_stores) {
      bounds = std::get<CombinedAccess>(kv->second).set;
    } else if (consider_loads) {
      bounds = std::get<LoadAccess>(kv->second).set;
    } else if (consider_stores) {
      bounds = std::get<StoreAccess>(kv->second).set;
    } else {
      TVM_FFI_ICHECK(false)
          << "Must consider at least on of either loads and stores, but both are false";
    }
    for (size_t i = 0; i < bounds.size(); ++i) {
      ret.push_back(sym::Union(bounds[i]).CoverRange(std::nullopt));
    }
    return ret;
  }

 private:
  using Parent = s_tir::StmtExprVisitorWithConstraintContext;

  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadObj* op) final {
    TensorVar tensor = op->source.as_or_throw<tvm::tirx::TensorVar>();
    // Record load-exclusive tensor access
    Touch(&std::get<LoadAccess>(tensor_access_map_[tensor.get()]).set, op->indices);
    // Record load-store inclusive tensor access
    Touch(&std::get<CombinedAccess>(tensor_access_map_[tensor.get()]).set, op->indices);
    return Parent::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorStoreObj* op) final {
    // Record store-exclusive tensor access
    Touch(&std::get<StoreAccess>(tensor_access_map_[op->dest.as_or_throw<TensorVar>().get()]).set,
          op->indices);
    // Record load-store inclusive tensor access
    Touch(
        &std::get<CombinedAccess>(tensor_access_map_[op->dest.as_or_throw<TensorVar>().get()]).set,
        op->indices);
    return Parent::Visit_(op);
  }

  void Touch(TensorTouches* bounds, const ffi::Array<PrimExpr>& args) {
    if (args.size() > bounds->size()) {
      bounds->resize(args.size());
    }
    for (size_t i = 0; i < args.size(); ++i) {
      if (args[i].as<prim::RampObj>()) {
        (*bounds)[i].emplace_back(IntSet::Vector(args[i]));
      } else {
        (*bounds)[i].emplace_back(GetSyncedAnalyzer()->int_set(args[i]));
      }
    }
  }

  std::unordered_map<const VarObj*, TensorDomainAccess> tensor_access_map_;
};

ffi::Array<ffi::Optional<Range>> DomainTouched(const Stmt& stmt, const TensorVar& tensor,
                                               bool consider_loads, bool consider_stores) {
  auto visitor = ffi::make_object<TensorTouchedDomain>();
  visitor->Visit(stmt);
  return visitor->FindUnion(tensor, consider_loads, consider_stores);
}

ffi::Map<TensorVar, ffi::Array<ffi::ObjectRef>> DomainTouchedAccessMap(const Function& func) {
  auto visitor = ffi::make_object<TensorTouchedDomain>();
  visitor->Visit(func->body);
  auto tensor_access_map = visitor->GetAccessedTensorRegions();
  ffi::Map<TensorVar, ffi::Array<ffi::ObjectRef>> ret;
  for (auto& var : func->params) {
    if (!var->ty.as<TensorTypeObj>()) {
      continue;
    }
    TensorVar tensor = var.as_or_throw<TensorVar>();
    auto& access = tensor_access_map[tensor.get()];
    ffi::Array<ffi::Array<IntSet>> loads, stores, combined;
    for (std::vector<IntSet>& touch : std::get<LoadAccess>(access).set) {
      loads.push_back(ffi::Array<IntSet>(touch));
    }
    for (std::vector<IntSet>& touch : std::get<StoreAccess>(access).set) {
      stores.push_back(ffi::Array<IntSet>(touch));
    }
    for (std::vector<IntSet>& touch : std::get<CombinedAccess>(access).set) {
      combined.push_back(ffi::Array<IntSet>(touch));
    }

    ffi::Array<ffi::ObjectRef> fields;
    fields.push_back(loads);
    fields.push_back(stores);
    fields.push_back(combined);
    ret.Set(tensor, fields);
  }
  return ret;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("s_tir.DomainTouched", DomainTouched)
      .def("s_tir.DomainTouchedAccessMap", DomainTouchedAccessMap);
}

}  // namespace s_tir
}  // namespace tvm
