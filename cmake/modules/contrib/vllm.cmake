# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

if(TVM_USE_VLLM)
  message(STATUS "Build with vllm paged attention kernel.")
  enable_language(CUDA)

  file(GLOB VLLM_CONTRIB_SRC CONFIGURE_DEPENDS src/runtime/extra/contrib/vllm/*.cu src/runtime/extra/contrib/vllm/*.cc)
  add_library(tvm_vllm_objs OBJECT ${VLLM_CONTRIB_SRC})
  tvm_configure_target(tvm_vllm_objs)
  target_include_directories(tvm_vllm_objs PRIVATE src/runtime/extra/contrib/vllm)
  target_link_libraries(tvm_vllm_objs PRIVATE tvm_runtime_extra_defs)
  target_link_libraries(tvm_runtime_extra PRIVATE tvm_vllm_objs)
endif(TVM_USE_VLLM)
