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
 * \file create.h
 * \brief The functions to make Relax tensor-creation operator calls.
 */
#ifndef TVM_RELAX_OP_TENSOR_CREATE_H_
#define TVM_RELAX_OP_TENSOR_CREATE_H_

#include <tvm/ffi/container/variant.h>
#include <tvm/relax/op/create.h>

#include "../op_common.h"
#include "tvm/relax/expr.h"

namespace tvm {
namespace relax {

/*!
 * \brief Fill array with scalar value.
 * \param shape The shape of the created tensor.
 * \param fill_value The value to fill. Must be a scalar tensor.
 * \param dtype The data type of the created tensor.
 * If dtype is not given, it will by default use the dtype of fill_value.
 * \return The result tensor.
 */
Expr full(ffi::Variant<ffi::Array<PrimExpr>, Expr> shape, Expr fill_value,
          ffi::Optional<DLDataType> dtype);

/*!
 * \brief Construct a tensor such that
 * - its shape is the same as the input data tensor's shape,
 * - its value is filled with the input scalar fill value.
 * \param x The input tensor, which provides the shape, and dtype
 * when the input dtype is void.
 * \param fill_value The value to fill. Must be a scalar tensor.
 * \param dtype The data type of the created tensor. If it is
 * void, the input tensor's dtype will be used.
 * \return The result tensor.
 */
Expr full_like(Expr x, Expr fill_value, ffi::Optional<DLDataType> dtype);

/*!
 * \brief Construct a tensor of all ones, with the input shape and dtype.
 * \param shape The shape of the created tensor.
 * \param dtype The data type of the created tensor.
 * \return The result tensor.
 */
Expr ones(Expr shape, DLDataType dtype);

/*!
 * \brief Construct a tensor with all ones, with shape of the input tensor shape.
 * \param x The input tensor, which provides the shape, and dtype
 * when the input dtype is void.
 * \param dtype The data type of the created tensor. If it is
 * void, the input tensor's dtype will be used.
 * \return The result tensor.
 */
Expr ones_like(Expr x, ffi::Optional<DLDataType> dtype);

/*!
 * \brief Construct a tensor of all zeros, with the input shape and dtype.
 * \param shape The shape of the created tensor.
 * \param dtype The data type of the created tensor.
 * \return The result tensor.
 */
Expr zeros(Expr shape, DLDataType dtype);

/*!
 * \brief Construct a tensor with all zeros, with shape of the input tensor shape.
 * \param x The input tensor, which provides the shape, and dtype
 * when the input dtype is void.
 * \param dtype The data type of the created tensor. If it is
 * void, the input tensor's dtype will be used.
 * \return The result tensor.
 */
Expr zeros_like(Expr x, ffi::Optional<DLDataType> dtype);

/*!
 * \brief Construct a 2-D tensor with ones on the diagonal and zeros elsewhere.
 * \param n The number of rows and columns in the output.
 * \param m The number of columns in the output. If None, defaults to n.
 * \param k The index of the diagonal. A positive value refers to an upper diagonal,
 *          a negative value to a lower diagonal, and 0 to the main diagonal.
 * \param dtype The data type of the created tensor.
 * \return The result tensor.
 */
Expr eye(PrimExpr n, PrimExpr m, PrimExpr k, DLDataType dtype);

/*!
 * \brief Construct a tensor with ones on the diagonal and zeros elsewhere,
 *        with shape and dtype similar to the input tensor.
 * \param x The input tensor, which provides the shape, and dtype
 * when the input dtype is void.
 * \param k The index of the diagonal. A positive value refers to an upper diagonal,
 *          a negative value to a lower diagonal, and 0 to the main diagonal.
 * \param dtype The data type of the created tensor. If it is
 * void, the input tensor's dtype will be used.
 * \return The result tensor.
 */
Expr eye_like(Expr x, PrimExpr k, ffi::Optional<DLDataType> dtype);

/*! \brief Construct a tensor with evenly spaced elements. */
Expr arange(PrimExpr start, PrimExpr stop, PrimExpr step, DLDataType dtype);

/*!
 * \brief Hamming window function.
 * \param window_size The size of the returned window.
 * \param periodic If True, returns a window to be used as periodic function.
 * If False, return a symmetric window.
 * \param alpha The co-efficient alpha.
 * \param beta The co-efficient beta.
 * \param dtype The data type of the created tensor.
 * \return The result tensor.
 */
Expr hamming_window(PrimExpr window_size, PrimExpr periodic, PrimExpr alpha, PrimExpr beta,
                    DLDataType dtype);

/*! \brief Return the lower triangular part of a matrix or a batch of matrices. */
Expr tril(Expr x, Expr k);

/*! \brief Return the upper triangular part of a matrix or a batch of matrices. */
Expr triu(Expr x, Expr k);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TENSOR_CREATE_H_
