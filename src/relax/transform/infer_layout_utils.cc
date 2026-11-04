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
#include "infer_layout_utils.h"

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>

#include "utils.h"

namespace tvm {
namespace relax {

using s_tir::IterVar;
using tirx::SLayout;

std::string TransposeSubLayoutStrLike(const std::string ref_str, const std::string& src_str,
                                      const std::string& desired_str) {
  std::string out;
  for (const char& c : desired_str) {
    if (std::isupper(c)) {
      auto res = src_str.find(c, 0);
      TVM_FFI_ICHECK(res != std::string::npos)
          << "Invalid SLayout:"
          << "can't find " << c << " in source layout" << src_str;
      out.push_back(ref_str[res]);
    } else if (isdigit(c)) {
      out.push_back(c);
    } else if (std::islower(c)) {
      auto res = src_str.find(std::toupper(c), 0);
      TVM_FFI_ICHECK(res != std::string::npos)
          << "Invalid SLayout:"
          << "can't find " << c << " in source layout" << src_str;
      out.push_back(std::tolower(ref_str[res]));
    }
  }
  return out;
}

ffi::Optional<SLayout> TransposeSubLayoutLike(const SLayout& ref, const SLayout& src,
                                              const SLayout& desired) {
  std::string ref_str = ref.name();
  std::string src_str = src.name();
  std::string desired_str = desired.name();
  std::string out = TransposeSubLayoutStrLike(ref_str, src_str, desired_str);
  return SLayout::Create(out);
}

SLayout TransposeLike(const ffi::Optional<SLayout>& input, const ffi::Optional<SLayout>& src,
                      const ffi::Optional<SLayout>& dst) {
  size_t src_ndim = src.has_value() ? src.value().ndim() : 0;
  size_t dst_ndim = dst.has_value() ? dst.value().ndim() : 0;
  size_t input_ndim = input.has_value() ? input.value().ndim() : 0;
  TVM_FFI_ICHECK(src_ndim == dst_ndim && input_ndim == src_ndim)
      << "Layouts must have the same size";
  std::vector<s_tir::IterVar> axes;
  for (size_t i = 0; i < src_ndim; ++i) {
    axes.push_back(input.value()->axes[src.value().IndexOf(dst.value()[i])]);
  }
  return SLayout(axes);
}

ffi::String TransposeStrLike(const ffi::String& input, const ffi::Optional<SLayout>& src,
                             const ffi::Optional<SLayout>& dst) {
  size_t src_ndim = src.has_value() ? src.value().ndim() : 0;
  size_t dst_ndim = dst.has_value() ? dst.value().ndim() : 0;
  TVM_FFI_ICHECK(src_ndim == dst_ndim && input.size() == src_ndim)
      << "Layouts must have the same size";
  std::string axes;
  for (size_t i = 0; i < src_ndim; ++i) {
    axes.push_back(input.at(src.value().IndexOf(dst.value()[i])));
  }
  return axes;
}

int FindAxis(const SLayout& dst, int axis) {
  axis = (axis + dst.ndim()) % dst.ndim();
  std::string layout_name = dst.name();
  layout_name.erase(std::remove_if(layout_name.begin(), layout_name.end(),
                                   [](unsigned char c) { return std::isdigit(c); }),
                    layout_name.end());
  return layout_name.find('A' + axis);
}

SLayout InitialLayout(int ndim) {
  TVM_FFI_ICHECK(ndim >= 0 && ndim <= 26) << "Only support up to 26 dimensions, but got " << ndim;
  return SLayout("ABCDEFGHIJKLMNOPQRSTUVWXYZ").SubLayout(0, ndim).value();
}

LayoutDecision InitialLayoutDecision(int ndim) {
  if (ndim == kUnknownNDim) {
    return LayoutDecision::InitUnknownDim();
  }
  TVM_FFI_ICHECK(ndim >= 0 && ndim <= 26) << "Only support up to 26 dimensions, but got " << ndim;
  return SLayout("ABCDEFGHIJKLMNOPQRSTUVWXYZ").SubLayout(0, ndim).value();
}

NLayout InitialNLayout(const Type& ty) {
  auto fmapleaf = [&](const Type& ty) -> NLayout {
    if (const auto* tensor_ty = ty.as<TensorTypeObj>()) {
      return NLayout(InitialLayoutDecision(tensor_ty->ndim));
    }
    return LayoutDecision::InitUnknownDim();
  };
  return MapToNestedMsg<LayoutDecision>(ty, fmapleaf);
}

NLayout InitialNLayout(const Expr& expr) { return InitialNLayout(GetType(expr)); }

LayoutDecision GetLayoutDecision(const VarLayoutMap& var_layout_map, const Expr& arg) {
  NLayout nlayout = GetNLayout(var_layout_map, arg);
  TVM_FFI_ICHECK(nlayout.IsLeaf()) << "Cannot get layout for " << arg;
  return nlayout.LeafValue();
}

NLayout GetNLayout(const VarLayoutMap& var_layout_map, const Expr& arg) {
  auto fmapleaf = [&](const Expr& expr) -> NLayout {
    if (const auto* var = expr.as<VarObj>()) {
      auto it = var_layout_map.find(ffi::GetRef<Var>(var));
      if (it != var_layout_map.end()) {
        return (*it).second;
      } else {
        return InitialNLayout(expr);
      }
    } else if (const auto* constant = expr.as<GenericConstObj>();
               constant && constant->value.as<ffi::TensorObj>()) {
      return InitialLayoutDecision(constant->value.cast<ffi::Tensor>().shape().size());
    }
    return LayoutDecision::InitUnknownDim();
  };
  return MapToNestedMsg<LayoutDecision>(arg, fmapleaf);
}

bool NoDesiredLayout(const Call& call,
                     const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts) {
  const OpObj* op_node = call->op.as<OpObj>();
  if (op_node == nullptr) return false;
  const auto& it = desired_layouts.find(op_node->name);
  return it == desired_layouts.end();
}

LayoutDecision FollowDecision(const LayoutDecision& src, int dst_ndim) {
  int src_ndim = (src->layout.has_value() ? src->layout.value().ndim() : 0);
  // broadcast case
  if (src_ndim == dst_ndim) {
    return src;
  } else {
    TVM_FFI_ICHECK_LT(src_ndim, dst_ndim)
        << "Cannot broadcast from " << src_ndim << " to " << dst_ndim;
    std::string layout = InitialLayout(dst_ndim - src_ndim).name();
    for (int i = 0; i < src_ndim; ++i) {
      layout.push_back((src->layout.has_value() ? src->layout.value().name() : "__undef__")[i] +
                       dst_ndim - src_ndim);
    }
    return LayoutDecision(SLayout::Create(layout));
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  LayoutDecisionObj::RegisterReflection();
  InferLayoutOutputObj::RegisterReflection();
}

}  // namespace relax
}  // namespace tvm
