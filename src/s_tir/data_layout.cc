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
 * \file src/lang/data_layout.cc
 * \brief Data SLayout expression.
 */
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/extra/dataclass.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/data_layout.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/analysis.h>

#include <algorithm>
#include <cctype>

namespace tvm {
namespace tirx {

using s_tir::IterVar;
using s_tir::IterVarObj;
using tvm::Var;

const SLayoutAxis SLayoutAxis::UPPER_CASE[] = {
    SLayoutAxis('A'), SLayoutAxis('B'), SLayoutAxis('C'), SLayoutAxis('D'), SLayoutAxis('E'),
    SLayoutAxis('F'), SLayoutAxis('G'), SLayoutAxis('H'), SLayoutAxis('I'), SLayoutAxis('J'),
    SLayoutAxis('K'), SLayoutAxis('L'), SLayoutAxis('M'), SLayoutAxis('N'), SLayoutAxis('O'),
    SLayoutAxis('P'), SLayoutAxis('Q'), SLayoutAxis('R'), SLayoutAxis('S'), SLayoutAxis('T'),
    SLayoutAxis('U'), SLayoutAxis('V'), SLayoutAxis('W'), SLayoutAxis('X'), SLayoutAxis('Y'),
    SLayoutAxis('Z')};

const SLayoutAxis SLayoutAxis::LOWER_CASE[] = {
    SLayoutAxis('a'), SLayoutAxis('b'), SLayoutAxis('c'), SLayoutAxis('d'), SLayoutAxis('e'),
    SLayoutAxis('f'), SLayoutAxis('g'), SLayoutAxis('h'), SLayoutAxis('i'), SLayoutAxis('j'),
    SLayoutAxis('k'), SLayoutAxis('l'), SLayoutAxis('m'), SLayoutAxis('n'), SLayoutAxis('o'),
    SLayoutAxis('p'), SLayoutAxis('q'), SLayoutAxis('r'), SLayoutAxis('s'), SLayoutAxis('t'),
    SLayoutAxis('u'), SLayoutAxis('v'), SLayoutAxis('w'), SLayoutAxis('x'), SLayoutAxis('y'),
    SLayoutAxis('z')};

const SLayoutAxis& SLayoutAxis::Get(const char name) {
  TVM_FFI_ICHECK((name >= 'A' && name <= 'Z') || (name >= 'a' && name <= 'z'))
      << "Invalid layout axis name: " << name << ". Has to be A-Z or a-z.";
  return (name >= 'A' && name <= 'Z') ? SLayoutAxis::UPPER_CASE[name - 'A']
                                      : SLayoutAxis::LOWER_CASE[name - 'a'];
}

const SLayoutAxis& SLayoutAxis::Get(const IterVar& itvar) {
  const std::string axis = itvar->var.get()->name;
  TVM_FFI_ICHECK_EQ(axis.size(), 1) << "Invalid layout axis " << axis;
  return SLayoutAxis::Get(axis[0]);
}

const SLayoutAxis& SLayoutAxis::Get(const std::string& name) {
  TVM_FFI_ICHECK_EQ(name.length(), 1) << "Invalid axis " << name;
  return SLayoutAxis::Get(name[0]);
}

SLayout::SLayout(const ffi::Array<IterVar>& axes) {
  auto node = ffi::make_object<SLayoutObj>();
  node->axes = axes;
  std::ostringstream repr;

  for (const IterVar& packed_axis : axes) {
    auto unpacked_axes = UnpackIterVar(packed_axis);
    bool is_grouped = unpacked_axes.size() > 1;

    if (is_grouped) repr << "[";
    for (const IterVar& axis : unpacked_axes) {
      if (const auto* factor = axis->dom.value()->extent.as<IntImmObj>()) {
        TVM_FFI_ICHECK_GT(factor->value, 0);
        repr << factor->value;
      } else {
        TVM_FFI_ICHECK(!is_grouped)
            << "Only Subordinate Axes with extent is allowed within a packed dim";
      }
      TVM_FFI_ICHECK_EQ(axis->var.get()->name.size(), 1)
          << "Invalid layout axis " << axis->var.get()->name;
      char c = axis->var.get()->name.operator std::string()[0];
      TVM_FFI_ICHECK((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))
          << "Invalid layout axis " << c;
      repr << axis->var.get()->name;
    }
    if (is_grouped) repr << "]";
  }

  node->name = repr.str();
  data_ = std::move(node);
}

SLayout::SLayout(const std::string& name, PrimType index_ty) {  // NOLINT(*)
  TVM_FFI_CHECK(index_ty.code() == DLDataTypeCode::kDLInt, TypeError)
      << "The input dtype should be integer type";
  TVM_FFI_CHECK(!name.empty() && name != "__undef__", ValueError)
      << "Undefined layout requires Optional<SLayout>";

  auto node = ffi::make_object<SLayoutObj>();
  node->name = name;

  // parse layout string
  int32_t factor = 0;
  bool in_packing = false;
  std::vector<IterVar> unpacked_axes;

  for (char c : name) {
    if (c >= 'A' && c <= 'Z') {
      TVM_FFI_ICHECK_EQ(factor, 0) << "Invalid layout " << name << ": invalid factor size "
                                   << factor << " before dimension " << c;
      PrimVar axis_var(std::string(1, c), index_ty);
      IterVar axis(Range(IntImm(index_ty, 0), axis_var), axis_var, s_tir::kDataPar);
      if (!in_packing) {
        node->axes.push_back(axis);
      } else {
        unpacked_axes.push_back(axis);
      }
    } else if (c >= 'a' && c <= 'z') {
      TVM_FFI_ICHECK_GT(factor, 0) << "Invalid layout " << name << ": invalid factor size "
                                   << factor << " for dimension " << c;
      std::stringstream name;
      name << factor << c;
      IterVar axis(Range(IntImm(index_ty, 0), IntImm(index_ty, factor)),
                   PrimVar(name.str(), index_ty), s_tir::kDataPar);
      if (!in_packing) {
        node->axes.push_back(axis);
      } else {
        unpacked_axes.push_back(axis);
      }
      factor = 0;
    } else if (c >= '0' && c <= '9') {
      TVM_FFI_ICHECK(factor >= 0) << "Invalid layout " << name << ": _ is adjacent to a number.";
      factor = factor * 10 + c - '0';
    } else if (c == '[') {
      TVM_FFI_ICHECK(!in_packing) << "Invalid layout " << name << ": can't do nested packing";
      in_packing = true;
    } else if (c == ']') {
      TVM_FFI_ICHECK(in_packing) << "Invalid layout " << name
                                 << ": encountered ] without matching bracket";
      TVM_FFI_ICHECK(unpacked_axes.size() > 1)
          << "Invalid layout " << name << ": found empty/single packed axis";
      std::stringstream ss;
      ffi::BigInt extent = 1;
      for (auto& axis : unpacked_axes) {
        TVM_FFI_ICHECK(axis->dom.value()->extent.as<IntImmObj>())
            << "Invalid SLayout " << name << ": can't have variable sized node(" << axis->var->name
            << ") within a packed axis";
        auto axis_name = axis->var->name.operator std::string();
        auto factor = axis->dom.value()->extent.as<IntImm>().value();
        ss << axis_name;
        extent *= factor->value;
      }
      std::string grouped_name = ss.str();
      IterVar grouped_axis(Range(IntImm(index_ty, 0), IntImm(index_ty, extent)),
                           PrimVar(grouped_name, index_ty), s_tir::kDataPar);
      node->axes.push_back(grouped_axis);

      in_packing = false;
      unpacked_axes.clear();
    } else {
      TVM_FFI_THROW(InternalError) << "Invalid layout " << name;
    }
  }
  TVM_FFI_ICHECK(in_packing == false)
      << "Invalid SLayout " << name << ": haven't terminated the packing sequence";

  // validate layout
  std::vector<int> axis_cnt(256, 0);
  for (const IterVar& pv : node->axes) {
    for (const IterVar& v : UnpackIterVar(pv)) {
      auto axis_str = v->var.get()->name.operator std::string();
      TVM_FFI_ICHECK_EQ(axis_str.size(), 1);
      char axis = axis_str[0];
      TVM_FFI_ICHECK((axis >= 'a' && axis <= 'z') || (axis >= 'A' && axis <= 'Z'));
      axis_cnt[axis] += 1;
    }
  }
  for (const IterVar& pv : node->axes) {
    for (const IterVar& v : UnpackIterVar(pv)) {
      char axis = v->var.get()->name.operator std::string()[0];
      if (axis >= 'a' && axis <= 'z') {
        TVM_FFI_ICHECK(axis_cnt[axis - 'a' + 'A'])
            << "Invalid layout " << name << ": missing axis " << std::toupper(axis);
        TVM_FFI_ICHECK(axis_cnt[axis] == 1)
            << "Invalid layout " << name << ": found more than one subordinate "
            << std::toupper(axis);
      }
    }
  }

  data_ = std::move(node);
}

ffi::Optional<SLayout> SLayout::Create(const std::string& name, PrimType index_ty) {
  TVM_FFI_CHECK(index_ty.code() == DLDataTypeCode::kDLInt, TypeError)
      << "The input dtype should be integer type";
  if (name.empty() || name == "__undef__") return std::nullopt;
  return SLayout(name, index_ty);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  SLayoutObj::RegisterReflection();
  refl::TypeAttrDef<SLayoutObj>().def(ffi::type_attr::kRepr.name,
                                      [](SLayout l, ffi::Function) -> ffi::String {
                                        return "SLayout(" + std::string(l->name) + ")";
                                      });

  refl::GlobalDef().def("s_tir.SLayout",
                        [](std::string name, PrimType dtype) -> ffi::Optional<SLayout> {
                          return SLayout::Create(name, dtype);
                        });
}

ffi::Optional<SLayout> SLayout::SubLayout(size_t pos, size_t len) const {
  if (!defined() || pos > ndim()) return SLayout::Undef();
  if (len == 0) return SLayout(ffi::Array<IterVar>());
  if (pos + len > ndim()) len = ndim() - pos;
  ffi::Array<IterVar> new_layout;
  const auto axes = operator->()->axes;
  for (size_t i = pos; i < pos + len; ++i) {
    new_layout.push_back(axes[i]);
  }
  return SLayout(new_layout);
}

ffi::Array<IterVar> SLayout::UnpackIterVar(IterVar packed_iter) {
  ffi::Array<IterVar> result;
  int64_t factor = 0, final_factor = 1;

  std::string name(packed_iter->var->name.c_str());
  PrimType index_ty = packed_iter->var.ty();

  for (auto ch : name) {
    if (ch >= '0' && ch <= '9') {
      factor = factor * 10 + (ch - '0');
    } else if (ch >= 'a' && ch <= 'z') {
      TVM_FFI_ICHECK(factor != 0) << "Invalid Factor Size";
      result.push_back(IterVar(Range(IntImm(index_ty, 0), IntImm(index_ty, factor)),
                               PrimVar(std::string(1, ch), index_ty), s_tir::kDataPar));
      final_factor *= factor;
      factor = 0;
    } else if (ch >= 'A' && ch <= 'Z') {
      TVM_FFI_ICHECK(factor == 0) << "Can't have non-zero factors for primal axis";
      PrimVar axis_var(std::string(1, ch), index_ty);
      result.push_back(IterVar(Range(IntImm(index_ty, 0), axis_var), axis_var, s_tir::kDataPar));
    }
  }

  return result;
}

IterVar SLayout::PackIterVar(ffi::Array<IterVar> iter_vars) {
  std::stringstream name;
  size_t extent = 1;

  PrimType index_ty = iter_vars[0]->dom.value()->extent.as<PrimExpr>().value().ty();
  for (auto itvar : iter_vars) {
    TVM_FFI_ICHECK(itvar->dom.value()->extent.as<IntImm>())
        << "Packed Axis can contain only Subordinate Axes";
    name << itvar->dom.value()->extent.as<IntImm>().value() << itvar->var->name;
    extent = (ffi::BigInt(extent) * itvar->dom.value()->extent.as<IntImm>().value()->value)
                 .as<size_t>()
                 .value();
  }

  return IterVar(Range(IntImm(index_ty, 0), IntImm(index_ty, extent)),
                 PrimVar(name.str(), index_ty), s_tir::kDataPar);
}

int32_t SLayout::FactorOf(const SLayoutAxis& axis) const {
  if (!defined()) return -1;
  const SLayoutAxis& sub = axis.ToSubordinate();

  int32_t factor = 1;
  bool has_sub = false;
  for (const IterVar& packed_itvar : operator->()->axes) {
    for (auto itvar : UnpackIterVar(packed_itvar)) {
      if (sub == SLayoutAxis::Get(itvar)) {
        has_sub = true;
        int32_t val = itvar->dom.value()->extent.as<IntImmObj>()->value.as<int32_t>().value();
        factor *= val;
      }
    }
  }
  factor = has_sub ? factor : -1;

  return factor;
}

inline bool GetStoreRule(ffi::Array<PrimExpr>* index_rule, ffi::Array<PrimExpr>* shape_rule,
                         const SLayout& src_layout, const SLayout& dst_layout) {
  if (!src_layout.defined() || src_layout.name().empty()) {
    LOG(WARNING) << "src layout '" << src_layout.name() << "' is invalid.";
    return false;
  }
  if (!dst_layout.defined() || dst_layout.name().empty()) {
    LOG(WARNING) << "dst layout '" << dst_layout.name() << "' is invalid.";
    return false;
  }

  std::vector<bool> exists(128, false);
  std::vector<PrimExpr> norm_indexes(128, PrimExpr(0));

  for (size_t i = 0; i < src_layout.ndim(); i++) {
    auto factor = src_layout.PackedAxisAt(i)->dom.value()->extent;
    auto src_unpacked_axes = SLayout::UnpackIterVar(src_layout.PackedAxisAt(i));

    if (src_unpacked_axes.size() == 1 && SLayoutAxis::Get(src_unpacked_axes[0]).IsPrimal()) {
      const auto& prim_axis = SLayoutAxis::Get(src_unpacked_axes[0]);
      int64_t offset = src_layout.FactorOf(prim_axis);
      if (offset == -1)
        norm_indexes[prim_axis.name()[0] - 'A'] =
            norm_indexes[prim_axis.name()[0] - 'A'] + src_layout.PackedAxisAt(i)->var;
      else
        norm_indexes[prim_axis.name()[0] - 'A'] =
            norm_indexes[prim_axis.name()[0] - 'A'] +
            src_layout.PackedAxisAt(i)->var * src_layout.FactorOf(prim_axis);
      exists[prim_axis.name()[0]] = true;
    } else {
      ffi::BigInt value = 1;
      std::vector<ffi::BigInt> index_divs(src_unpacked_axes.size());
      for (size_t j = 0; j < src_unpacked_axes.size(); j++) {
        index_divs[j] = value;
        const auto* extent = src_unpacked_axes[j]->dom.value()->extent.as<IntImmObj>();
        TVM_FFI_ICHECK(extent) << "Expected Integer Extents for Offset Calculation";
        index_divs.push_back(value);
        value *= extent->value;
      }
      std::reverse(index_divs.begin(), index_divs.end());

      for (size_t j = 0; j < src_unpacked_axes.size(); j++) {
        PrimExpr extent = src_unpacked_axes[j]->dom.value()->extent;
        const SLayoutAxis& store_axis_impl = SLayoutAxis::Get(src_unpacked_axes[j]);
        const SLayoutAxis& sub_axis = store_axis_impl.ToSubordinate(); /* Not Needed */
        const SLayoutAxis& prim_axis = store_axis_impl.ToPrimal();

        PrimExpr factor_ij = indexdiv(src_layout.PackedAxisAt(i)->var,
                                      IntImm(src_layout.PackedAxisAt(i)->var.ty(), index_divs[j]));
        if (j != 0) factor_ij = indexmod(factor_ij, extent);

        for (size_t k = i; k < src_layout.ndim(); k++) {
          size_t l = 0;
          if (k == i) l = j + 1;

          auto inter_unpacked_axes = SLayout::UnpackIterVar(src_layout.PackedAxisAt(k));
          for (; l < inter_unpacked_axes.size(); l++) {
            const SLayoutAxis& axis = SLayoutAxis::Get(inter_unpacked_axes[l]);
            if (axis == sub_axis) {
              IntImm sub_extent = inter_unpacked_axes[l]->dom.value()->extent.as_or_throw<IntImm>();
              factor_ij = factor_ij * IntImm(sub_extent.ty(), sub_extent->value);
            }
          }
        }

        norm_indexes[prim_axis.name()[0] - 'A'] =
            norm_indexes[prim_axis.name()[0] - 'A'] + factor_ij;
      }
    }
  }

  sym::Analyzer ana;

  for (size_t i = 0; i < dst_layout.ndim(); i++) {
    const auto dst_unpacked_axes = SLayout::UnpackIterVar(dst_layout.PackedAxisAt(i));

    if (dst_unpacked_axes.size() == 1 && SLayoutAxis::Get(dst_unpacked_axes[0]).IsPrimal()) {
      const auto& prim_axis = SLayoutAxis::Get(dst_unpacked_axes[0]);
      if (!exists[prim_axis.name()[0]]) return false;
      int64_t offset = dst_layout.FactorOf(prim_axis);
      if (offset != -1) {
        index_rule->push_back(
            indexdiv(norm_indexes[prim_axis.name()[0] - 'A'], dst_layout.FactorOf(prim_axis)));
        shape_rule->push_back(
            indexdiv(norm_indexes[prim_axis.name()[0] - 'A'] + (dst_layout.FactorOf(prim_axis) - 1),
                     dst_layout.FactorOf(prim_axis)));
      } else {
        index_rule->push_back(norm_indexes[prim_axis.name()[0] - 'A']);
        shape_rule->push_back(norm_indexes[prim_axis.name()[0] - 'A']);
      }
    } else {
      PrimExpr factor(0);
      for (size_t j = 0; j < dst_unpacked_axes.size(); j++) {
        const auto& prim_axis = SLayoutAxis::Get(dst_unpacked_axes[j]).ToPrimal();
        const auto& sub_axis = SLayoutAxis::Get(dst_unpacked_axes[j]).ToSubordinate();
        const auto* extent = dst_unpacked_axes[j]->dom.value()->extent.as<IntImmObj>();
        TVM_FFI_ICHECK(extent) << "Expected extent to be IntImmObj";

        size_t divfactor = 1;
        for (size_t k = i; k < dst_layout.ndim(); k++) {
          size_t l = 0;
          if (k == i) l = j + 1;

          const auto inter_unpacked_axes = SLayout::UnpackIterVar(dst_layout.PackedAxisAt(k));
          for (; l < inter_unpacked_axes.size(); l++) {
            const auto& axis = SLayoutAxis::Get(inter_unpacked_axes[l]);
            if (sub_axis == axis) {
              const auto* sub_extent = inter_unpacked_axes[l]->dom.value()->extent.as<IntImmObj>();
              TVM_FFI_ICHECK(sub_extent) << "Expected Integer Extents for Offset Calculation";
              divfactor = (ffi::BigInt(divfactor) * sub_extent->value).as<size_t>().value();
            }
          }
        }

        factor = factor + indexmod(indexdiv(norm_indexes[prim_axis.name()[0] - 'A'], divfactor),
                                   IntImm(extent->ty.as_or_throw<PrimType>(), extent->value));
        for (size_t k = j + 1; k < dst_unpacked_axes.size(); k++) {
          factor = factor * dst_unpacked_axes[k]->dom.value()->extent.as<IntImm>().value();
        }
      }
      ana->Simplify(factor);
      index_rule->push_back(factor);
      shape_rule->push_back(factor);
    }
  }

  std::stringstream ss;
  ss << "index rule for " << src_layout.name() << "-->" << dst_layout.name() << ": [ ";
  for (const auto& r : *index_rule) {
    ss << r << ", ";
  }
  ss << "]" << std::endl;

  ss << "shape rule for " << src_layout.name() << "-->" << dst_layout.name() << ": [ ";
  for (const auto& r : *shape_rule) {
    ss << r << ", ";
  }
  ss << "]" << std::endl;
  VLOG(1) << ss.str() << std::endl;

  return true;
}

inline ffi::Array<PrimExpr> TransformIndex(const ffi::Array<PrimExpr>& src_index,
                                           const ffi::Array<IterVar>& src_axis,
                                           const ffi::Array<PrimExpr>& transform_rule) {
  sym::Analyzer ana;
  ffi::Array<PrimExpr> result;
  std::unordered_map<const tvm::VarObj*, PrimExpr> bind_map;
  for (size_t i = 0; i < src_index.size(); ++i) {
    bind_map.insert_or_assign(src_axis[i]->var.get(), src_index[i]);
  }
  auto f_substitute = [&bind_map](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (auto it = bind_map.find(var.get()); it != bind_map.end()) {
      return ffi::Any(it->second);
    }
    return ffi::Unchanged();
  };
  for (PrimExpr rule : transform_rule) {
    result.push_back(ana->Simplify(
        ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(rule, f_substitute).as_or_throw<PrimExpr>()));
  }
  return result;
}

ffi::Array<PrimExpr> SBijectiveLayout::ForwardIndex(const ffi::Array<PrimExpr>& src_index) const {
  TVM_FFI_ICHECK(defined()) << "Cannot operate on an undefined bijective layout.";
  const SBijectiveLayoutObj* self = operator->();
  TVM_FFI_ICHECK_EQ(src_index.size(), self->src_layout->axes.size())
      << "Input mismatch with layout " << self->src_layout;
  return TransformIndex(src_index, self->src_layout->axes, self->index_forward_rule);
}

ffi::Array<PrimExpr> SBijectiveLayout::BackwardIndex(const ffi::Array<PrimExpr>& dst_index) const {
  TVM_FFI_ICHECK(defined()) << "Cannot operate on an undefined bijective layout.";
  const SBijectiveLayoutObj* self = operator->();
  TVM_FFI_ICHECK_EQ(dst_index.size(), self->dst_layout->axes.size())
      << "Output mismatch with layout " << self->dst_layout;
  return TransformIndex(dst_index, self->dst_layout->axes, self->index_backward_rule);
}

inline ffi::Array<PrimExpr> TransformShape(const ffi::Array<PrimExpr>& src_shape,
                                           const ffi::Array<IterVar>& src_axis,
                                           const ffi::Array<IterVar>& target_axis,
                                           const ffi::Array<PrimExpr>& transform_rule) {
  sym::Analyzer ana;
  TVM_FFI_ICHECK_EQ(src_shape.size(), src_axis.size())
      << "Input shape size " << src_shape.size() << " mismatch with the expected shape size "
      << src_axis.size();
  // bind variables for original axes
  // for major-axis, bind the corresponding size
  // for minor-axis, simply bind it as 0, so that we can reuse forward/backward_rule,
  // e.g., (C * 16 + c) / 32
  std::unordered_map<const tvm::VarObj*, PrimExpr> bind_map;
  for (size_t i = 0; i < src_shape.size(); ++i) {
    PrimExpr orig_shape = src_shape[i];
    IterVar orig_axis = src_axis[i];
    auto layout = SLayout::UnpackIterVar(orig_axis);
    if (layout.size() != 1 || !SLayoutAxis::Get(layout[0]).IsPrimal()) {
      if (orig_shape.defined()) {
        const auto* orig_shape_const = orig_shape.as<IntImmObj>();
        const auto* orig_axis_extent = orig_axis->dom.value()->extent.as<IntImmObj>();
        if (orig_shape_const) {
          TVM_FFI_ICHECK_EQ(orig_shape_const->value, orig_axis_extent->value)
              << "Input shape mismatch at index " << i << ". Expected "
              << orig_axis->dom.value()->extent << ", get " << orig_shape;
        }
      }
      bind_map.insert_or_assign(orig_axis->var.get(), IntImm(orig_axis->var.ty(), 0));
    } else {
      bind_map.insert_or_assign(orig_axis->var.get(),
                                orig_axis->var.ty() == orig_shape.ty()
                                    ? orig_shape
                                    : prim::cast(orig_shape, orig_axis->var.ty()));
    }
  }
  auto f_substitute = [&bind_map](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (auto it = bind_map.find(var.get()); it != bind_map.end()) {
      return ffi::Any(it->second);
    }
    return ffi::Unchanged();
  };
  // infer the target shape,
  // for major-axis, use the forward/backward_rule directly,
  // for minor-axis, simply use the extent.
  ffi::Array<PrimExpr> result;
  TVM_FFI_ICHECK_EQ(transform_rule.size(), target_axis.size());
  for (size_t i = 0; i < transform_rule.size(); ++i) {
    PrimExpr rule = transform_rule[i];
    IterVar axis = target_axis[i];
    auto layout = SLayout::UnpackIterVar(axis);
    if (layout.size() != 1 || !SLayoutAxis::Get(layout[0]).IsPrimal()) {
      result.push_back(axis->dom.value()->extent);
    } else {
      result.push_back(
          ana->Simplify(ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(rule, f_substitute)
                            .as_or_throw<PrimExpr>()));
    }
  }

  std::stringstream ss;
  ss << "shape rule for " << SLayout(src_axis).name() << "-->" << SLayout(target_axis).name()
     << ": [ ";
  for (const auto& r : transform_rule) {
    ss << r << ", ";
  }
  ss << "]" << std::endl;

  ss << "shape transform: [ ";
  for (const auto& s : src_shape) {
    ss << s << ", ";
  }
  ss << "] --> [ ";
  for (const auto& r : result) {
    ss << r << ", ";
  }
  ss << "]" << std::endl;
  VLOG(1) << std::endl << ss.str();

  return result;
}

ffi::Array<PrimExpr> SBijectiveLayout::ForwardShape(const ffi::Array<PrimExpr>& shape) const {
  TVM_FFI_ICHECK(defined()) << "Cannot operate on an undefined bijective layout.";
  const SBijectiveLayoutObj* self = operator->();
  return TransformShape(shape, self->src_layout->axes, self->dst_layout->axes,
                        self->shape_forward_rule);
}

ffi::Array<PrimExpr> SBijectiveLayout::BackwardShape(const ffi::Array<PrimExpr>& shape) const {
  TVM_FFI_ICHECK(defined()) << "Cannot operate on an undefined bijective layout.";
  const SBijectiveLayoutObj* self = operator->();
  return TransformShape(shape, self->dst_layout->axes, self->src_layout->axes,
                        self->shape_backward_rule);
}

SBijectiveLayout::SBijectiveLayout(SLayout src_layout, SLayout dst_layout) {
  auto result = Create(std::move(src_layout), std::move(dst_layout));
  TVM_FFI_CHECK(result.has_value(), ValueError) << "Layouts are not bijectively convertible";
  data_ = result.value().data_;
}

ffi::Optional<SBijectiveLayout> SBijectiveLayout::Create(SLayout src_layout, SLayout dst_layout) {
  auto n = ffi::make_object<SBijectiveLayoutObj>(std::move(src_layout), std::move(dst_layout));
  if (!GetStoreRule(&n->index_forward_rule, &n->shape_forward_rule, n->src_layout, n->dst_layout)) {
    return std::nullopt;
  }
  TVM_FFI_ICHECK(
      GetStoreRule(&n->index_backward_rule, &n->shape_backward_rule, n->dst_layout, n->src_layout));
  return SBijectiveLayout(ffi::UnsafeInit{}, std::move(n));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  SBijectiveLayoutObj::RegisterReflection();
  refl::TypeAttrDef<SBijectiveLayoutObj>().def(
      ffi::type_attr::kRepr.name, [](SBijectiveLayout bl, ffi::Function) -> ffi::String {
        return "SBijectiveLayout(" + std::string(bl->src_layout.name()) + "->" +
               std::string(bl->dst_layout.name()) + ")";
      });

  refl::GlobalDef().def("s_tir.SBijectiveLayout",
                        [](ffi::Optional<SLayout> src_layout,
                           ffi::Optional<SLayout> dst_layout) -> ffi::Optional<SBijectiveLayout> {
                          if (!src_layout || !dst_layout) return std::nullopt;
                          return SBijectiveLayout::Create(src_layout.value(), dst_layout.value());
                        });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("s_tir.SLayoutIndexOf",
           [](SLayout layout, std::string axis) -> int { return layout.IndexOf(axis); })
      .def("s_tir.SLayoutFactorOf",
           [](SLayout layout, std::string axis) -> int {
             return layout.FactorOf(SLayoutAxis::Get(axis));
           })
      .def("s_tir.SLayoutNdim", [](SLayout layout) -> int { return layout.ndim(); })
      .def("s_tir.SLayoutGetItem",
           [](SLayout layout, int idx) -> std::string {
             const auto& axis = layout.PackedAxisAt(idx);
             return axis->var->name;
           })
      .def_method("s_tir.SBijectiveLayoutForwardIndex", &SBijectiveLayout::ForwardIndex)
      .def_method("s_tir.SBijectiveLayoutBackwardIndex", &SBijectiveLayout::BackwardIndex)
      .def_method("s_tir.SBijectiveLayoutForwardShape", &SBijectiveLayout::ForwardShape)
      .def_method("s_tir.SBijectiveLayoutBackwardShape", &SBijectiveLayout::BackwardShape);
}
}  // namespace tirx
}  // namespace tvm
