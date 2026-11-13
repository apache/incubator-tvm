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

# TensorRT Codegen only. This can be enabled independently of TVM_USE_TENSORRT_RUNTIME to enable
# compilation of TensorRT modules without requiring TensorRT to be installed. The compiled modules
# will only be able to be executed using a TVM built with TVM_USE_TENSORRT_RUNTIME=ON.

include(FindPackageHandleStandardArgs)

if(TVM_USE_TENSORRT_CODEGEN)
    message(STATUS "Build with TensorRT codegen")
    file(GLOB COMPILER_TENSORRT_SRCS CONFIGURE_DEPENDS src/relax/backend/contrib/tensorrt/*.cc)
    set_source_files_properties(${COMPILER_TENSORRT_SRCS} PROPERTIES COMPILE_FLAGS "-Wno-deprecated-declarations")
    file(GLOB RUNTIME_TENSORRT_SRCS CONFIGURE_DEPENDS src/runtime/extra/contrib/tensorrt/tensorrt_runtime.cc)
    set_source_files_properties(${RUNTIME_TENSORRT_SRCS} PROPERTIES COMPILE_FLAGS "-Wno-deprecated-declarations")
    list(APPEND COMPILER_SRCS ${COMPILER_TENSORRT_SRCS})
    if(NOT TVM_USE_TENSORRT_RUNTIME)
        add_library(tvm_tensorrt_objs OBJECT ${RUNTIME_TENSORRT_SRCS})
        tvm_configure_target(tvm_tensorrt_objs)
        target_link_libraries(tvm_tensorrt_objs PRIVATE tvm_runtime_extra_defs)
        target_link_libraries(tvm_runtime_extra PRIVATE tvm_tensorrt_objs)
    endif()
endif()

# TensorRT Runtime — goes into libtvm_runtime_extra.
if(TVM_USE_TENSORRT_RUNTIME)
    if(IS_DIRECTORY ${TVM_USE_TENSORRT_RUNTIME})
        set(TENSORRT_ROOT_DIR ${TVM_USE_TENSORRT_RUNTIME})
        message(STATUS "Custom TensorRT path: " ${TENSORRT_ROOT_DIR})
    endif()
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS ${TENSORRT_ROOT_DIR} PATH_SUFFIXES include)
    find_library(TENSORRT_LIB_DIR nvinfer HINTS ${TENSORRT_ROOT_DIR} PATH_SUFFIXES lib)
    find_package_handle_standard_args(TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIB_DIR)
    if(NOT TENSORRT_FOUND)
        message(ERROR "Could not find TensorRT.")
    endif()
    message(STATUS "TENSORRT_LIB_DIR: " ${TENSORRT_LIB_DIR})
    # TRT runtime sources
    file(GLOB RUNTIME_TENSORRT_SRCS CONFIGURE_DEPENDS src/runtime/extra/contrib/tensorrt/*.cc)
    set_source_files_properties(${RUNTIME_TENSORRT_SRCS} PROPERTIES COMPILE_FLAGS "-Wno-deprecated-declarations")
    add_library(tvm_tensorrt_objs OBJECT ${RUNTIME_TENSORRT_SRCS})
    tvm_configure_target(tvm_tensorrt_objs)
    target_link_libraries(tvm_tensorrt_objs PRIVATE tvm_runtime_extra_defs)
    target_link_libraries(tvm_runtime_extra PRIVATE tvm_tensorrt_objs ${TENSORRT_LIB_DIR})

    target_include_directories(tvm_tensorrt_objs PRIVATE ${TENSORRT_INCLUDE_DIR})
    target_include_directories(tvm_compiler PRIVATE ${TENSORRT_INCLUDE_DIR})
    target_compile_definitions(tvm_tensorrt_objs PRIVATE TVM_GRAPH_EXECUTOR_TENSORRT)
    target_compile_definitions(tvm_compiler PRIVATE TVM_GRAPH_EXECUTOR_TENSORRT)
endif()
