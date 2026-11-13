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

/*! \file tvm/te/reduction.h
 * \brief Tensor-expression reduction constructors.
 */
#ifndef TVM_TE_REDUCTION_H_
#define TVM_TE_REDUCTION_H_

#include <tvm/base.h>
#include <tvm/s_tir/iter_var.h>

namespace tvm::prim {
/*!
 * \brief sum of source expression over axis
 * \param source The source expression.
 * \param axis List of iteration variables that will be used for reduction.
 * \param init The value with which to initialize the output.
 * \param loc The location of this operation in the source.
 * \return The result.
 */
TVM_DLL PrimExpr sum(PrimExpr source, ffi::Array<s_tir::IterVar> axis,
                     ffi::Array<PrimExpr> init = {}, Location loc = UnknownLoc());

/*!
 * \brief logical And of source expression over axis
 * \param source The source expression.
 * \param axis List of iteration variables that will be used for reduction.
 * \param init The value with which to initialize the output.
 * \param loc The location of this operation in the source.
 */
TVM_DLL PrimExpr all(PrimExpr source, ffi::Array<s_tir::IterVar> axis,
                     ffi::Array<PrimExpr> init = {}, Location loc = UnknownLoc());

/*!
 * \brief logical Or of source expression over axis
 * \param source The source expression.
 * \param axis List of iteration variables that will be used for reduction.
 * \param init The value with which to initialize the output.
 * \param loc The location of this operation in the source.
 * \return The result.
 */
TVM_DLL PrimExpr any(PrimExpr source, ffi::Array<s_tir::IterVar> axis,
                     ffi::Array<PrimExpr> init = {}, Location loc = UnknownLoc());

/*!
 * \brief product of source expression over axis
 * \param source The source expression.
 * \param axis List of iteration variables that will be used for reduction.
 * \param init The value with which to initialize the output.
 * \param loc The location of this operation in the source.
 * \return The result.
 */
TVM_DLL PrimExpr prod(PrimExpr source, ffi::Array<s_tir::IterVar> axis,
                      ffi::Array<PrimExpr> init = {}, Location loc = UnknownLoc());

}  // namespace tvm::prim

namespace tvm {
/*!
 * \brief max of source expression over axis
 * \param source The source expression.
 * \param axis List of iteration variables that will be used for reduction.
 * \param init The value with which to initialize the output.
 * \param loc The location of this operation in the source.
 * \return The result.
 */
TVM_DLL PrimExpr max(PrimExpr source, ffi::Array<s_tir::IterVar> axis,
                     ffi::Array<PrimExpr> init = {}, Location loc = UnknownLoc());

/*!
 * \brief max of source expression over axis
 * \param source The source expression.
 * \param axis List of iteration variables that will be used for reduction.
 * \param init The value with which to initialize the output.
 * \param loc The location of this operation in the source.
 * \return The result.
 */
TVM_DLL PrimExpr min(PrimExpr source, ffi::Array<s_tir::IterVar> axis,
                     ffi::Array<PrimExpr> init = {}, Location loc = UnknownLoc());

}  // namespace tvm
#endif  // TVM_TE_REDUCTION_H_
