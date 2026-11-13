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

# Example NPU Codegen
if(TVM_USE_EXAMPLE_NPU_CODEGEN)
    message(STATUS "Build with Example NPU codegen")

    file(GLOB COMPILER_EXAMPLE_NPU_SRCS CONFIGURE_DEPENDS src/relax/backend/contrib/example_npu/*.cc)
    list(APPEND COMPILER_SRCS ${COMPILER_EXAMPLE_NPU_SRCS})

    file(GLOB RUNTIME_EXAMPLE_NPU_SRCS CONFIGURE_DEPENDS src/runtime/extra/contrib/example_npu/*.cc)
    if(NOT TVM_USE_EXAMPLE_NPU_RUNTIME)
        add_library(tvm_example_npu_objs OBJECT ${RUNTIME_EXAMPLE_NPU_SRCS})
        tvm_configure_target(tvm_example_npu_objs)
        target_link_libraries(tvm_example_npu_objs PRIVATE tvm_runtime_extra_defs)
        target_link_libraries(tvm_runtime_extra PRIVATE tvm_example_npu_objs)
    endif()
endif()

# Example NPU Runtime — goes into libtvm_runtime_extra.
if(TVM_USE_EXAMPLE_NPU_RUNTIME)
    message(STATUS "Build with Example NPU runtime")

    file(GLOB RUNTIME_EXAMPLE_NPU_SRCS CONFIGURE_DEPENDS src/runtime/extra/contrib/example_npu/*.cc)
    add_library(tvm_example_npu_objs OBJECT ${RUNTIME_EXAMPLE_NPU_SRCS})
    tvm_configure_target(tvm_example_npu_objs)
    target_link_libraries(tvm_example_npu_objs PRIVATE tvm_runtime_extra_defs)
    target_link_libraries(tvm_runtime_extra PRIVATE tvm_example_npu_objs)

    target_compile_definitions(tvm_example_npu_objs PRIVATE TVM_GRAPH_EXECUTOR_EXAMPLE_NPU)
endif()
