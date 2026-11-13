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
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/meta_schedule/extracted_task.h>
#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>
#include <tvm/tirx/function.h>

#include "../../te/operation/create_function.h"
#include "./utils.h"

namespace tvm {
namespace s_tir {
namespace meta_schedule {

ExtractedTask::ExtractedTask(ffi::String task_name, IRModule mod, Target target,
                             ffi::Array<IRModule> dispatched, int weight) {
  ffi::ObjectPtr<ExtractedTaskObj> n = ffi::make_object<ExtractedTaskObj>(target);
  n->task_name = task_name;
  n->mod = mod;
  n->dispatched = dispatched;
  n->weight = weight;
  data_ = n;
}

TVM_FFI_STATIC_INIT_BLOCK() { ExtractedTaskObj::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.meta_schedule.ExtractedTask",
                        [](ffi::String task_name, IRModule mod, Target target,
                           ffi::Array<IRModule> dispatched, int weight) -> ExtractedTask {
                          return ExtractedTask(task_name, mod, target, dispatched, weight);
                        });
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
