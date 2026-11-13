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
 * \file tvm/relax/op/manipulate.h
 * \brief Attributes for tensor manipulation operators.
 */
#ifndef TVM_RELAX_OP_MANIPULATE_H_
#define TVM_RELAX_OP_MANIPULATE_H_

#include <tvm/relax/expr.h>
#include <tvm/tirx/index_map.h>

namespace tvm {
namespace relax {

/*! \brief Attributes used in concat operators */
struct ConcatAttrs : public AttrsObj {
  ffi::Optional<int64_t> axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ConcatAttrs", ConcatAttrs, AttrsObj);
};  // struct ConcatAttrs

/*! \brief Attributes used in expand_dims operators */
struct ExpandDimsAttrs : public AttrsObj {
  ffi::Array<int64_t> axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ExpandDimsAttrs", ExpandDimsAttrs, AttrsObj);
};  // struct ExpandDimsAttrs

/*! \brief Attributes used in layout_transform operator */
struct LayoutTransformAttrs : public AttrsObj {
  explicit LayoutTransformAttrs(tirx::IndexMap value) : index_map(std::move(value)) {}
  explicit LayoutTransformAttrs(ffi::UnsafeInit) : index_map(ffi::UnsafeInit{}) {}
  tirx::IndexMap index_map;
  // pad_value is chosen to be of PrimExpr type, as it represents constant TIR POD expression. This
  // needs to be revisited in case PrimExpr is evolved to represent symbolic expression in future.
  ffi::Optional<PrimExpr> pad_value;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.LayoutTransformAttrs", LayoutTransformAttrs,
                                    AttrsObj);
};  // struct LayoutTransformAttrs

/*! \brief Attributes used in permute_dims operator */
struct PermuteDimsAttrs : public AttrsObj {
  ffi::Optional<ffi::Array<int64_t>> axes;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.PermuteDimsAttrs", PermuteDimsAttrs, AttrsObj);
};  // struct PermuteDimsAttrs

/*! \brief Attributes used in split operator */
struct SplitAttrs : public AttrsObj {
  ffi::ObjectRef indices_or_sections;
  int axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.SplitAttrs", SplitAttrs, AttrsObj);
};  // struct SplitAttrs

/*! \brief Attributes used in squeeze operators */
struct SqueezeAttrs : public AttrsObj {
  ffi::Optional<ffi::Array<int64_t>> axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.SqueezeAttrs", SqueezeAttrs, AttrsObj);
};  // struct SqueezeAttrs

/*! \brief Attributes used in stack operators */
struct StackAttrs : public AttrsObj {
  ffi::Optional<int64_t> axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.StackAttrs", StackAttrs, AttrsObj);
};  // struct StackAttrs

/*! \brief Attributes used in repeat operators */
struct RepeatAttrs : public AttrsObj {
  int repeats;
  ffi::Optional<int64_t> axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.RepeatAttrs", RepeatAttrs, AttrsObj);
};  // struct RepeatAttrs

/*! \brief Attributes used in tile operators */
struct TileAttrs : public AttrsObj {
  ffi::Array<int64_t> repeats;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.TileAttrs", TileAttrs, AttrsObj);
};  // struct TileAttrs

/*! \brief Attributes used in flip operators */
struct FlipAttrs : public AttrsObj {
  int64_t axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.FlipAttrs", FlipAttrs, AttrsObj);
};  // struct FlipAttrs

/*! \brief Attributes used in reverse_sequence operators */
struct ReverseSequenceAttrs : public AttrsObj {
  int64_t seq_axis;
  int64_t batch_axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ReverseSequenceAttrs", ReverseSequenceAttrs,
                                    AttrsObj);
};  // struct ReverseSequenceAttrs

/*! \brief Attributes used in gather_elements operators */
struct GatherElementsAttrs : public AttrsObj {
  int64_t axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.GatherElementsAttrs", GatherElementsAttrs,
                                    AttrsObj);
};  // struct GatherElementsAttrs

/*! \brief Attributes used in gather_nd operators */
struct GatherNDAttrs : public AttrsObj {
  int64_t batch_dims;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.GatherNDAttrs", GatherNDAttrs, AttrsObj);
};  // struct GatherNDAttrs

/*! \brief Attributes used in index_put operator */
struct IndexPutAttrs : public AttrsObj {
  bool accumulate;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.IndexPutAttrs", IndexPutAttrs, AttrsObj);
};  // struct IndexPutAttrs

/*! \brief Attribute used in meshgrid operator */
struct MeshgridAttrs : public AttrsObj {
  ffi::Optional<ffi::String> indexing;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.MeshgridAttrs", MeshgridAttrs, AttrsObj);
};

/*! \brief Attributes used in scatter_elements operators */
struct ScatterElementsAttrs : public AttrsObj {
  int64_t axis;
  ffi::String reduction;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ScatterElementsAttrs", ScatterElementsAttrs,
                                    AttrsObj);
};  // struct ScatterElementsAttrs

/*! \brief Attributes used in scatter_nd operators */
struct ScatterNDAttrs : public AttrsObj {
  ffi::String reduction;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ScatterNDAttrs", ScatterNDAttrs, AttrsObj);
};  // struct ScatterNDAttrs

/*! \brief Attributes used in slice_scatter operator */
struct SliceScatterAttrs : public AttrsObj {
  int axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.SliceScatterAttrs", SliceScatterAttrs, AttrsObj);
};  // struct SliceScatterAttrs

/*! \brief Attributes used in one_hot operator */
struct OneHotAttrs : public AttrsObj {
  int depth;
  int axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.OneHotAttrs", OneHotAttrs, AttrsObj);
};  // struct OneHotAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_MANIPULATE_H_
