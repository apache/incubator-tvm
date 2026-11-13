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
 * \file fuse.h
 * \brief Fuse operation
 */
#ifndef TVM_TOPI_DETAIL_FUSE_H_
#define TVM_TOPI_DETAIL_FUSE_H_

#include <tvm/te/operation.h>

namespace tvm {
namespace topi {
namespace detail {

using namespace tvm::te;

/*!
 * \brief Fuse all of the given args
 *
 * \param stage The stage in which to apply the fuse
 * \param args The iteration variables to be fused
 *
 * \return The fused iteration variable
 */
inline IterVar Fuse(Stage stage, const ffi::Array<IterVar>& args) {
  IterVar res(ffi::UnsafeInit{});
  stage.fuse(args, &res);
  return res;
}

}  // namespace detail
}  // namespace topi
}  // namespace tvm
#endif  // TVM_TOPI_DETAIL_FUSE_H_
