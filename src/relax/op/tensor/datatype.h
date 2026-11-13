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
 * \file datatype.h
 * \brief The functions to make Relax datatype operator calls.
 */
#ifndef TVM_RELAX_OP_TENSOR_DATATYPE_H_
#define TVM_RELAX_OP_TENSOR_DATATYPE_H_

#include <tvm/relax/op/datatype.h>

#include "../op_common.h"

namespace tvm {
namespace relax {

/*!
 * \brief Cast input tensor to the given data type.
 * \param x The input data to the operator.
 * \param dtype The target data type
 * \return The casted result.
 */
Expr astype(Expr x, DLDataType dtype);

/*!
 * \brief A wrapper to wrap the input const tensor to the given data type.
 * \param x The input const tensor to the operator.
 * \param dtype The target data type
 * \return The wrapped result.
 */
Expr wrap_param(Expr x, DLDataType dtype);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TENSOR_DATATYPE_H_
