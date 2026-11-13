/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  Sex The NOTICE file
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
 * KIND, either express or implied.  Sex The License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relax/op/tensor/qdq.h
 * \brief The functions to make Relax quantize/dequantize operator calls.
 */
#ifndef TVM_RELAX_OP_TENSOR_QDQ_H_
#define TVM_RELAX_OP_TENSOR_QDQ_H_

#include <tvm/relax/op/qdq.h>

namespace tvm {
namespace relax {

/*!
 * \brief Quantize op.
 * This operator takes input and produces quantized output. The input tensor can be of any shape.
 * The output shape is the same as input shape.
 * \param data  The input tensor to be quantized.
 * \param scale The output scale.
 * \param zero_point The output zero_point.
 * \param axis The channel axis for quantization.
 * \param out_dtype The data type of the output tensor.
 * \return The computed result.
 */
Expr quantize(Expr data, Expr scale, Expr zero_point, int axis, DLDataType out_dtype);

/*!
 * \brief Dequantize op.
 * This operator takes input and produces dequantized output. The input tensor can be of any shape.
 * The output shape is the same as input shape.
 * \param data  The input tensor to be dequantized.
 * \param scale The input scale.
 * \param zero_point The input zero_point.
 * \param axis The channel axis for dequantization.
 * \param out_dtype The data type of the output tensor.
 * \return The computed result.
 */
Expr dequantize(Expr data, Expr scale, Expr zero_point, int axis, DLDataType out_dtype);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TENSOR_QDQ_H_
