/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*!
 * \file nms.h
 * \brief The functions to make Relax Non-maximum suppression operator calls.
 */

#ifndef TVM_RELAX_OP_VISION_NMS_H_
#define TVM_RELAX_OP_VISION_NMS_H_

#include <tvm/ffi/string.h>
#include <tvm/relax/op/vision.h>

#include "../op_common.h"

namespace tvm {
namespace relax {

/*! \brief Compute All Class NonMaximumSuppression. */
Expr all_class_non_max_suppression(Expr boxes, Expr scores, Expr max_output_boxes_per_class,
                                   Expr iou_threshold, Expr score_threshold,
                                   ffi::String output_format);

/*! \brief Get valid count of bounding boxes given a score threshold. */
Expr get_valid_counts(Expr data, double score_threshold, int id_index, int score_index);

/*! \brief Non-maximum suppression for object detection. */
Expr non_max_suppression(Expr data, Expr valid_count, Expr indices, int max_output_size,
                         double iou_threshold, bool force_suppress, int top_k, int coord_start,
                         int score_index, int id_index, bool return_indices, bool invalid_to_bottom,
                         double soft_nms_sigma = 0.0, double score_threshold = 0.0);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_VISION_NMS_H_
