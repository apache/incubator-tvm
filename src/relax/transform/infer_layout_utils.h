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
 * \file infer_layout_utils.h
 * \brief Utility functions to alter the layouts of operators or replace primitive operators with
          other expressions. This pass can be used for computing convolution in
          custom layouts or other general weight pre-transformation.
 */

#ifndef TVM_RELAX_TRANSFORM_INFER_LAYOUT_UTILS_H_
#define TVM_RELAX_TRANSFORM_INFER_LAYOUT_UTILS_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/nested_msg.h>
#include <tvm/relax/op/create.h>
#include <tvm/relax/op/datatype.h>
#include <tvm/relax/op/image.h>
#include <tvm/relax/op/linear_algebra.h>
#include <tvm/relax/op/manipulate.h>
#include <tvm/relax/op/nn.h>
#include <tvm/relax/op/statistical.h>
#include <tvm/relax/op_attr.h>
#include <tvm/s_tir/data_layout.h>

#include <array>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace relax {

using tirx::SLayout;

/*!
 * \brief A layout decision node that holds the layout decision of the tensor.
 * \param layout The layout of the tensor.
 */
class LayoutDecisionObj : public ffi::Object {
 public:
  /*! \brief The layout decision of the tensor. */
  ffi::Optional<SLayout> layout;
  /*! \brief Whether the dim of tensor is unknown. */
  bool is_unknown_dim = false;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<LayoutDecisionObj>()
        .def_ro("layout", &LayoutDecisionObj::layout)
        .def_ro("is_unknown_dim", &LayoutDecisionObj::is_unknown_dim);
  }

  TVM_FFI_DECLARE_OBJECT_INFO("relax.transform.LayoutDecision", LayoutDecisionObj, ffi::Object);
};

class LayoutDecision : public ffi::ObjectRef {
 public:
  LayoutDecision(SLayout layout, bool is_unknown_dim = false)  // NOLINT(*)
      : LayoutDecision(ffi::Optional<SLayout>(std::move(layout)), is_unknown_dim) {}

  LayoutDecision(ffi::Optional<SLayout> layout, bool is_unknown_dim = false) {  // NOLINT(*)
    auto n = ffi::make_object<LayoutDecisionObj>();
    n->layout = std::move(layout);
    n->is_unknown_dim = is_unknown_dim;
    data_ = n;
  }

  static LayoutDecision InitUnknownDim() { return LayoutDecision(SLayout::Undef(), true); }

  inline std::string name() const {
    if (operator->()->is_unknown_dim) {
      return "unknown_dim";
    }
    return operator->()->layout.has_value() ? operator->()->layout.value().name() : "__undef__";
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(LayoutDecision, ffi::ObjectRef, LayoutDecisionObj);
};

using NLayout = NestedMsg<LayoutDecision>;

/*!
 * \brief An output structure to hold results from FInferCorrectLayout calls.
 * \param input_layouts Inferred input layouts.
 * \param output_layouts Inferred output layouts.
 * \param new_attrs Updated attributes consistent with inferred layouts.
 */
class InferLayoutOutputObj : public ffi::Object {
 public:
  ffi::Array<NLayout> input_layouts;
  ffi::Array<NLayout> output_layouts;
  ffi::Optional<Attrs> new_attrs;
  ffi::Map<IntImm, Expr> new_args;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<InferLayoutOutputObj>()
        .def_ro("input_layouts", &InferLayoutOutputObj::input_layouts)
        .def_ro("output_layouts", &InferLayoutOutputObj::output_layouts)
        .def_ro("new_attrs", &InferLayoutOutputObj::new_attrs)
        .def_ro("new_args", &InferLayoutOutputObj::new_args);
  }

  TVM_FFI_DECLARE_OBJECT_INFO("relax.transform.InferLayoutOutput", InferLayoutOutputObj,
                              ffi::Object);
};

class InferLayoutOutput : public ffi::ObjectRef {
 public:
  explicit InferLayoutOutput(ffi::Array<NLayout> input_layouts, ffi::Array<NLayout> output_layouts,
                             ffi::Optional<Attrs> new_attrs, ffi::Map<IntImm, Expr> new_args = {}) {
    auto n = ffi::make_object<InferLayoutOutputObj>();
    n->input_layouts = std::move(input_layouts);
    n->output_layouts = std::move(output_layouts);
    n->new_attrs = std::move(new_attrs);
    n->new_args = std::move(new_args);
    data_ = n;
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(InferLayoutOutput, ffi::ObjectRef,
                                                InferLayoutOutputObj);
};

struct NLayoutEqual {
  bool operator()(const NLayout& a, const NLayout& b) const {
    auto layout_equal = [](const LayoutDecision& a, const LayoutDecision& b) {
      if (a.defined() && b.defined()) {
        return a.name() == b.name();
      }
      return a.defined() == b.defined();
    };
    return Equal(a, b, layout_equal);
  }
};

using VarLayoutMap = ffi::Map<Var, NLayout>;

// Compatibility names; new callers can deduce the value type from the key.
using FRelaxInferLayout = decltype(op_attr::kInferLayout)::ValueType;

/*!
 * \brief Initialize a layout given the number of dimensions.
 * \param ndim The number of dimensions.
 * \return The initialized layout.
 */
SLayout InitialLayout(int ndim);

/*!
 * \brief Initialize a layout decision given the number of dimensions.
 * \param ndim The number of dimensions.
 * \return The initialized layout decision.
 */
LayoutDecision InitialLayoutDecision(int ndim);

/*!
 * \brief Initialize a nested layout decision given the type.
 * \param ty The ty.
 * \return The initialized nested layout decision.
 */
NLayout InitialNLayout(const Type& ty);

/*!
 * \brief Initialize a nested layout decision given expression
 * \param ty The expr
 * \return The initialized nested layout decision.
 */
NLayout InitialNLayout(const Expr& expr);

/*!
 * \brief Transposing given layout with subindexing
 * \param ref The layout to be transformed.
 * \param src The source layout.
 * \param dst The destination layout.
 * \return The transposed dst layout.
 */
ffi::Optional<SLayout> TransposeSubLayoutLike(const SLayout& ref, const SLayout& src,
                                              const SLayout& desired);

/*!
 * \brief Transposing given layout in string format with subindexing
 * \param ref The layout to be transformed.
 * \param src The source layout.
 * \param dst The destination layout.
 * \return The transposed dst layout.
 */
std::string TransposeSubLayoutStrLike(const std::string ref_str, const std::string& src_str,
                                      const std::string& desired_str);

/*!
 * \brief Transpose the input layout  like the src layout to the dst layout.
 * \param input The input layout.
 * \param src The source layout.
 * \param dst The destination layout.
 * \return The transposed input layout.
 */
SLayout TransposeLike(const ffi::Optional<SLayout>& input, const ffi::Optional<SLayout>& src,
                      const ffi::Optional<SLayout>& dst);

/*!
 * \brief Transpose the input string like the src layout to the dst layout.
 * \param input The input str.
 * \param src The source layout.
 * \param dst The destination layout.
 * \return The transposed input str.
 */
ffi::String TransposeStrLike(const ffi::String& input, const ffi::Optional<SLayout>& src,
                             const ffi::Optional<SLayout>& dst);

/*!
 * \brief Find axis in the dst layout. 0 represents the first axis, 1 represents the second axis,
 * etc.
 * \param dst The destination layout.
 * \param axis The axis to be found
 * \return The axis in the dst layout.
 */
int FindAxis(const SLayout& dst, int axis);

/*!
 * \brief Get the layout decision of the expr. The expr must be a Tensor.
 * \param var_layout_map The layout of the variables.
 * \param arg The expr.
 * \return The layout decision of the expr.
 */
LayoutDecision GetLayoutDecision(const VarLayoutMap& var_layout_map, const Expr& arg);

/*!
 * \brief Get the nested layout decision of the expr. The expr must be a nested Tensor.
 * \param var_layout_map The layout of the variables.
 * \param arg The expr.
 * \return The nested layout decision of the expr.
 */
NLayout GetNLayout(const VarLayoutMap& var_layout_map, const Expr& arg);

/*!
 * \brief Check if the op is not in the desired layout
 * \param call The call node contains the op
 * \param desired_layouts The desired layouts of the operator.
 * \return True if the op is not in the desired layout.
 */
bool NoDesiredLayout(const Call& call,
                     const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts);

/*!
 * \brief Let a tensor with ndim to follow the src layout decision.
 * \param src The source layout decision.
 * \param dst_ndim The number of dimensions of the tensor.
 * \return The layout decision of the tensor.
 */
LayoutDecision FollowDecision(const LayoutDecision& src, int dst_ndim);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_TRANSFORM_INFER_LAYOUT_UTILS_H_
