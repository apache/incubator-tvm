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
 * \file multibox_transform_loc.h
 * \brief The functions to make Relax multibox_transform_loc operator calls.
 */

#ifndef TVM_RELAX_OP_VISION_MULTIBOX_TRANSFORM_LOC_H_
#define TVM_RELAX_OP_VISION_MULTIBOX_TRANSFORM_LOC_H_

#include <tvm/relax/op/vision.h>

#include "../op_common.h"

namespace tvm {
namespace relax {

/*! \brief Decode SSD box encodings and prepare class scores (TFLite-compatible). */
Expr multibox_transform_loc(Expr cls_pred, Expr loc_pred, Expr anchor, bool clip, double threshold,
                            ffi::Array<double> variances, bool keep_background, bool apply_softmax);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_VISION_MULTIBOX_TRANSFORM_LOC_H_
