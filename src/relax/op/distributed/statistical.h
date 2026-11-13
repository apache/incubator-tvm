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
 * \file statistical.h
 * \brief The functions to infer type for distributed statistical operator
 */

#ifndef TVM_RELAX_OP_DISTRIBUTED_STATISTICAL_H_
#define TVM_RELAX_OP_DISTRIBUTED_STATISTICAL_H_

#include <tvm/relax/op_attr.h>

#include "utils.h"

namespace tvm {
namespace relax {
namespace distributed {

Type InferDistTypeStatistical(const Call& call, const BlockBuilder& ctx);

// clang-format off
#define RELAX_REGISTER_STATISTICAL_DIST_INFER_TYPE(OpName) \
  OpDef("relax." #OpName)                              \
      .set_attr(tvm::relax::distributed::op_attr::kInferTypeWithBuilder, InferDistTypeStatistical)
// clang-format on

}  // namespace distributed
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_DISTRIBUTED_STATISTICAL_H_
