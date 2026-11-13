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
 * \file tvm/relax/op/ccl.h
 * \brief Attributes for ccl operators.
 */
#ifndef TVM_RELAX_OP_CCL_H_
#define TVM_RELAX_OP_CCL_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {

/*! \brief Attributes used in allreduce operators */
struct AllReduceAttrs : public tvm::AttrsObj {
  ffi::String op_type;
  bool in_group;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.AllReduceAttrs", AllReduceAttrs, AttrsObj);
};  // struct AllReduceAttrs

/*! \brief Attributes used in allgather operators */
struct AllGatherAttrs : public tvm::AttrsObj {
  int num_workers;
  bool in_group;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.AllGatherAttrs", AllGatherAttrs, AttrsObj);
};  // struct AllGatherAttrs

/*! \brief Attributes used in scatter operators */
struct ScatterCollectiveAttrs : public tvm::AttrsObj {
  int num_workers;
  int axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ScatterCollectiveAttrs", ScatterCollectiveAttrs,
                                    AttrsObj);
};  // struct ScatterCollectiveAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_CCL_H_
