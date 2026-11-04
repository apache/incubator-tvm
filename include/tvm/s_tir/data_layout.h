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
 * \file tvm/s_tir/data_layout.h
 * \brief SLayout expression to describe the data organization of a tensor.
 *  And SBijectiveLayout to mapping two data layouts between each other.
 */
#ifndef TVM_S_TIR_DATA_LAYOUT_H_
#define TVM_S_TIR_DATA_LAYOUT_H_

#include <tvm/base.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/s_tir/iter_var.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "tvm/ir/expr.h"

namespace tvm {
namespace tirx {

class SLayout;

class SLayoutAxis {
 public:
  static const SLayoutAxis& Get(const char name);

  // Get the singleton SLayoutAxis using itvar->var->name
  static const SLayoutAxis& Get(const s_tir::IterVar& itvar);

  // Get the singleton SLayoutAxis using name[0] (size of name must be 1).
  static const SLayoutAxis& Get(const std::string& name);

  inline bool IsPrimal() const { return name_ >= 'A' && name_ <= 'Z'; }
  inline std::string name() const { return std::string(1, name_); }

  // if current axis is primal, switch the axis to its subordinate one,
  // else switch to the primal.
  inline const SLayoutAxis& ToDual() const {
    if (name_ >= 'A' && name_ <= 'Z') {
      return SLayoutAxis::Get(name_ - 'A' + 'a');
    } else {
      return SLayoutAxis::Get(name_ - 'a' + 'A');
    }
  }

  // return the primal axis. If it is already primal, return itself.
  const SLayoutAxis& ToPrimal() const { return IsPrimal() ? *this : ToDual(); }

  // return the subordinate axis. If it is already subordinate, return itself.
  const SLayoutAxis& ToSubordinate() const { return IsPrimal() ? ToDual() : *this; }

  inline bool operator==(const SLayoutAxis& rhs) const { return name_ == rhs.name_; }

  friend std::ostream& operator<<(std::ostream& os, const SLayoutAxis& l) {
    os << l.name();
    return os;
  }

 private:
  static const SLayoutAxis UPPER_CASE[];
  static const SLayoutAxis LOWER_CASE[];
  SLayoutAxis(const SLayoutAxis&);
  SLayoutAxis& operator=(const SLayoutAxis&);
  explicit SLayoutAxis(const char name) : name_(name) {}

  const char name_;
};

/*!
 * \brief SLayout is to describe how data is organized within an N-dimention tensor.
 *  It is composed of upper cases, lower cases and numbers,
 *  where upper case indicates a primal axis and
 *  the corresponding lower case with factor size indicates the subordinate axis.
 *  For example, NCHW16c can describe a 5-D tensor of
 *  [batch_size, channel, height, width, channel_block].
 *  Here subordinate axis channel_block=16 is the factor size of the primal axis C (channel).
 *  SLayout for scalar is defined, while both its name and axes have size 0.
 */
class SLayoutObj : public ffi::Object {
 public:
  /*! \brief string representation of layout, "" for scalar. */
  ffi::String name;
  /*! \brief specify each axis of the layout,
   *   in which the variable name is the name of the axis.
   *   The s_tir::IterVar's extent indicates the size of the axis,
   *   it is a variable for a primal axis, but a constant for a subordinate axis.
   *   Empty for scalar's layout.
   */
  ffi::Array<s_tir::IterVar> axes;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SLayoutObj>()
        .def_ro("name", &SLayoutObj::name)
        .def_ro("axes", &SLayoutObj::axes);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.SLayout", SLayoutObj, ffi::Object);
};

/*!
 * \brief Managed reference to SLayoutObj
 * \sa SLayoutObj
 */
class SLayout : public ffi::ObjectRef {
 public:
  explicit SLayout(const ffi::Array<s_tir::IterVar>& axes);

  /*! \brief construct from a string */
  SLayout(const tvm::ffi::String& name) : SLayout(name.operator std::string()) {}  // NOLINT(*)

  /*! \brief construct from a string */
  SLayout(const char* name) : SLayout(std::string(name)) {}  // NOLINT(*)

  /*!
   * \brief construct from a string.
   * \param name input in layout convention:
   *        upper case indicates a dimension and
   *        the corresponding lower case with factor size
   *        indicates the split dimension.
   *        "__undef__" is reserved for an absent Optional<SLayout>.
   * \param index_ty The type of generated axes vars in the returned layout.
   *        It is required to be integer type.
   */
  TVM_DLL SLayout(const std::string& name, PrimType index_ty = PrimType::Int(32));  // NOLINT(*)
  TVM_DLL static ffi::Optional<SLayout> Create(const std::string& name,
                                               PrimType index_ty = PrimType::Int(32));

  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  SLayoutObj* operator->() { return static_cast<SLayoutObj*>(get_mutable()); }

  /*!
   * \brief Return an undefined layout.
   * \return an absent layout.
   */
  static ffi::Optional<SLayout> Undef() { return std::nullopt; }

  /*!
   * \brief Packs the Given Array of IterVars into a Single s_tir::IterVar. Each s_tir::IterVar in
   * the Array should represent either a single primal axis or one or more subordinate axis
   * \param iters Array of iter vars to be packed
   * \return A packed iter var
   */
  static s_tir::IterVar PackIterVar(ffi::Array<s_tir::IterVar> iters);

  /*!
   * \brief Unpacks a Packed s_tir::IterVar into its constituents
   * \param packed_iter A Packed s_tir::IterVar containing a single primal axis or one or more
   * subordinate axis
   * \return Constituent IterVars
   */
  static ffi::Array<s_tir::IterVar> UnpackIterVar(s_tir::IterVar packed_iter);

  /*!
   * \brief Returns a sub-layout which is the portion of the object
   *        that starts at dimension \p pos and spans \p len dimensions
   *        (or until the end of the layout, whichever comes first).
   * \param pos The start position.
   * \param len The length of the sub-layout. if 0, return layout of scalar
   * \return A newly constructed SLayout object, or std::nullopt if pos is out of range.
   */
  ffi::Optional<SLayout> SubLayout(size_t pos, size_t len) const;

  /*!
   * \brief Split \p axis by \p size and put the sub-axis to position \p target_pos.
   * \param axis The source axis to be split. It must be a primal-axis;
   * \param target_pos The target position of the newly split subordinate-axis.
   * \param factor size of the sub-dimension.
   * \return A newly constructed SLayout object.
   */
  SLayout Split(const SLayoutAxis& axis, size_t target_pos, int32_t factor) const;

  /*! \return number of dimensions */
  inline size_t ndim() const {
    if (!defined()) return 0;
    return operator->()->axes.size();
  }

  /*! \return number of super dimensions */
  inline size_t ndim_primal() const {
    if (!defined()) return 0;
    size_t ct = 0;
    for (auto px : operator->()->axes) {
      auto iter_vars = UnpackIterVar(px);
      for (auto x : iter_vars) {
        if (SLayoutAxis::Get(x).IsPrimal()) {
          ct++;
        }
      }
    }
    return ct;
  }

  /*!
   * \brief Returns a new layout where the dims have been expanded to match the primal dimensions.
   * \param dst_layout The dst layout to which current layout has to be expanded.
   * \return The expanded SLayout.
   */
  inline ffi::Optional<SLayout> ExpandPrimal(const SLayout& dst_layout) {
    // 1) Find the axis which are missing in the current layout. Make them the prefix.
    std::string new_src_layout_str = "";
    for (auto packed_axis : dst_layout->axes) {
      auto iter_vars = UnpackIterVar(packed_axis);
      for (auto dst_axis : iter_vars) {
        if (SLayoutAxis::Get(dst_axis).IsPrimal()) {
          if (!this->Contains(SLayoutAxis::Get(dst_axis))) {
            new_src_layout_str += dst_axis->var->name;
          }
        }
      }
    }
    // 2) Now, add the primal axis of the current layout.
    new_src_layout_str += this->name();
    return SLayout::Create(new_src_layout_str);
  }

  /*!
   * \brief return the index of the input axis.
   *        If it is not found in the layout or the layout is undefined,
   *        return -1.
   * \param axis The input axis either a layout axis, or a packed axis
   * \return the index or -1 if not found.
   */
  inline int32_t IndexOf(const std::string& axis) const {
    if (!this->defined()) return -1;
    const auto axes = operator->()->axes;
    for (size_t i = 0; i < axes.size(); ++i) {
      if (axes[i]->var->name == axis) return static_cast<int32_t>(i);
    }
    return -1;
  }

  /*!
   * \brief return the index of the input axis.
   *        If it is not found in the layout or the layout is undefined,
   *        return -1.
   * \param axis the input layout axis.
   * \return the index or -1 if not found.
   */
  inline int32_t IndexOf(const SLayoutAxis& axis) const { return IndexOf(axis.name()); }

  /*!
   * \brief return the index of the input axis.
   *        If it is not found in the layout or the layout is undefined,
   *        return -1.
   * \param iter the input iter var.
   * \return the index or -1 if not found.
   */
  inline int32_t IndexOf(const s_tir::IterVar& iter) const { return IndexOf(iter->var->name); }

  /*!
   * \brief Get the factor size of the subordinate axis.
   * \param axis the input primal-axis or subordinate-axis.
   * \return the size of the subordinate-axis of \p axis (if \p axis is a primal-axis),
   *         or the size of \p axis itself (if \p axis is a subordinate-axis).
   *         Return -1 if \p axis is not in the layout the layout is undefined.
   */
  int32_t FactorOf(const SLayoutAxis& axis) const;

  /*!
   * \brief Whether the layout contains an axis.
   * \param axis axis to be checked.
   * \return Whether the layout contains the axis.
   */
  bool Contains(const SLayoutAxis& axis) const {
    if (!defined()) return false;
    for (const s_tir::IterVar packed_var : operator->()->axes) {
      auto iter_vars = UnpackIterVar(packed_var);
      for (auto var : iter_vars) {
        if (var->var->name == axis.name()) {
          return true;
        }
      }
    }
    return false;
  }

  const SLayoutAxis& operator[](int32_t i) const {
    TVM_FFI_ICHECK(defined()) << "Try to access axis from an undefined layout.";
    int32_t index = i < 0 ? static_cast<int32_t>(ndim() + i) : i;
    TVM_FFI_ICHECK(index >= 0 && static_cast<size_t>(index) < ndim()) << "Invalid index " << i;
    const s_tir::IterVar axis = operator->()->axes[index];
    return SLayoutAxis::Get(axis);
  }

  s_tir::IterVar PackedAxisAt(int32_t i) const {
    TVM_FFI_ICHECK(defined()) << "Try to access axis from an undefined layout.";
    int32_t index = i < 0 ? static_cast<int32_t>(ndim() + i) : i;
    TVM_FFI_ICHECK(index >= 0 && static_cast<size_t>(index) < ndim()) << "Invalid index " << i;
    const s_tir::IterVar axis = operator->()->axes[index];
    return axis;
  }

  /*! \return the string description of the layout */
  inline std::string name() const {
    if (!defined()) return "__undef__";
    return operator->()->name;
  }

  /*!
   * \brief Whether the two layouts are equal.
   * \param rhs Another layout.
   * \return whether the two layouts are equal.
   */
  inline bool Equals(const SLayout& rhs) const { return name() == rhs.name(); }

  /*!
   * \brief allow output string of layout to ostream
   * \param os the output stream
   * \param l the layout
   * \return the ostream
   */
  friend std::ostream& operator<<(std::ostream& os, const SLayout& l) {
    os << l.name();
    return os;
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SLayout, ffi::ObjectRef, SLayoutObj);
};

// Internal node container SBijectiveLayout
class SBijectiveLayoutObj : public ffi::Object {
 public:
  explicit SBijectiveLayoutObj(ffi::UnsafeInit tag) : src_layout(tag), dst_layout(tag) {}

  SBijectiveLayoutObj(SLayout src, SLayout dst)
      : src_layout(std::move(src)), dst_layout(std::move(dst)) {}
  /*! \brief Describes how source axes can be mapped to the destination axes,
   *   e.g., [i0 / 16, i1, i0 % 16] can describe NC -> NC16n
   */
  ffi::Array<PrimExpr> index_forward_rule;
  /*! \brief Describes how destination axes can be mapped to the source axes */
  ffi::Array<PrimExpr> index_backward_rule;
  /*! \brief Describes how source shapes can be mapped to the destination shapes */
  ffi::Array<PrimExpr> shape_forward_rule;
  /*! \brief Describes how destination shapes can be mapped to the source shapes */
  ffi::Array<PrimExpr> shape_backward_rule;

  /*! \brief The source layout */
  SLayout src_layout;
  /*! \brief The destination layout */
  SLayout dst_layout;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SBijectiveLayoutObj>()
        .def_ro("src_layout", &SBijectiveLayoutObj::src_layout)
        .def_ro("dst_layout", &SBijectiveLayoutObj::dst_layout)
        .def_ro("index_forward_rule", &SBijectiveLayoutObj::index_forward_rule)
        .def_ro("index_backward_rule", &SBijectiveLayoutObj::index_backward_rule)
        .def_ro("shape_forward_rule", &SBijectiveLayoutObj::shape_forward_rule)
        .def_ro("shape_backward_rule", &SBijectiveLayoutObj::shape_backward_rule);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.SBijectiveLayout", SBijectiveLayoutObj, ffi::Object);
};

/*!
 * \brief Bijective function mapping for data layout transformation.
 *   Given two SLayout, SBijectiveLayout build and store the mapping rules,
 *   provides API to transform N-dimention tensor from the source indices (i0, i1, .., im)
 *   to the destination indices (j0, j1, .., jm).
 */
class SBijectiveLayout : public ffi::ObjectRef {
 public:
  explicit SBijectiveLayout(ffi::UnsafeInit tag, ffi::ObjectPtr<SBijectiveLayoutObj> data)
      : ffi::ObjectRef(tag) {
    data_ = std::move(data);
  }

  /*!
   * \brief The constructor
   * \param src_layout The source layout
   * \param dst_layout The destination layout
   */
  TVM_DLL SBijectiveLayout(SLayout src_layout, SLayout dst_layout);
  TVM_DLL static ffi::Optional<SBijectiveLayout> Create(SLayout src_layout, SLayout dst_layout);

  // Given the source shape, infer the destination shape.
  TVM_DLL ffi::Array<PrimExpr> ForwardShape(const ffi::Array<PrimExpr>& shape) const;
  // Given the destination shape, recover the source shape.
  TVM_DLL ffi::Array<PrimExpr> BackwardShape(const ffi::Array<PrimExpr>& dst_shape) const;
  // Given the destination indices, infer the destination indices.
  TVM_DLL ffi::Array<PrimExpr> ForwardIndex(const ffi::Array<PrimExpr>& index) const;
  // Given the destination indices, recover the source indices.
  TVM_DLL ffi::Array<PrimExpr> BackwardIndex(const ffi::Array<PrimExpr>& dst_index) const;

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SBijectiveLayout, ffi::ObjectRef,
                                                SBijectiveLayoutObj);
};

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_S_TIR_DATA_LAYOUT_H_
