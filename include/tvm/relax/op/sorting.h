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
 * \file tvm/relax/op/sorting.h
 * \brief Attributes for sorting operators.
 */
#ifndef TVM_RELAX_OP_SORTING_H_
#define TVM_RELAX_OP_SORTING_H_

#include <tvm/relax/expr.h>
#include <tvm/tirx/index_map.h>

namespace tvm {
namespace relax {

/*! \brief Attributes used in sort operator */
struct SortAttrs : public AttrsObj {
  int axis;
  bool descending;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.SortAttrs", SortAttrs, AttrsObj);
};  // struct SortAttrs

/*! \brief Attributes used in argsort operator */
struct ArgsortAttrs : public AttrsObj {
  int axis;
  bool descending;
  ffi::Optional<DLDataType> dtype;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ArgsortAttrs", ArgsortAttrs, AttrsObj);
};  // struct ArgsortAttrs

/*! \brief Attributes used in topk operator */
struct TopKAttrs : public AttrsObj {
  int k;
  int axis;
  bool largest;
  ffi::String ret_type;
  ffi::Optional<DLDataType> dtype;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.TopKAttrs", TopKAttrs, AttrsObj);
};  // struct TopKAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_SORTING_H_
