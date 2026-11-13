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

if(TVM_USE_CLML)
    file(GLOB CLML_RELAX_CONTRIB_SRC CONFIGURE_DEPENDS src/relax/backend/contrib/clml/*.cc)
    file(GLOB CLML_RUNTIME_MODULE CONFIGURE_DEPENDS src/runtime/extra/contrib/clml/clml_runtime.cc)
    if(TARGET tvm_runtime_opencl)
        get_target_property(_clml_opencl_headers tvm_runtime_opencl INTERFACE_INCLUDE_DIRECTORIES)
    endif()
    if(NOT _clml_opencl_headers)
        find_path(_clml_opencl_headers CL/cl.h
            HINTS "${TVM_OPENCL_HEADERS_ROOT}" ENV OPENCL_SDK
            PATH_SUFFIXES include REQUIRED)
    endif()
    list(APPEND COMPILER_SRCS ${CLML_RELAX_CONTRIB_SRC})
    if(NOT TVM_USE_CLML_GRAPH_EXECUTOR)
        add_library(tvm_clml_objs OBJECT ${CLML_RUNTIME_MODULE})
        tvm_configure_target(tvm_clml_objs)
        target_include_directories(tvm_clml_objs SYSTEM PRIVATE ${_clml_opencl_headers})
        target_link_libraries(tvm_clml_objs PRIVATE tvm_runtime_extra_defs)
        target_link_libraries(tvm_runtime_extra PRIVATE tvm_clml_objs)
    endif()
    message(STATUS "Build with CLML support : " ${TVM_USE_CLML})
    if(NOT TVM_USE_CLML STREQUAL "ON")
        set(CLML_VERSION_HEADER "${TVM_USE_CLML}/CL/cl_qcom_ml_ops.h")
        if(EXISTS ${CLML_VERSION_HEADER})
            file(READ ${CLML_VERSION_HEADER} ver)
            string(REGEX MATCH "CL_QCOM_ML_OPS_H_MAJOR_VERSION ([0-9]*)" _ ${ver})
            set(CLML_VERSION_MAJOR ${CMAKE_MATCH_1})
        else()
            set(CLML_VERSION_MAJOR "4")
        endif()
    else()
        set(CLML_VERSION_MAJOR "4")
    endif()
    target_compile_definitions(tvm_compiler PRIVATE TVM_CLML_VERSION=${CLML_VERSION_MAJOR})
    message(STATUS "CLML SDK Version :" ${CLML_VERSION_MAJOR})
else()
    set(CLML_VERSION_MAJOR "4")
endif()

if(TVM_USE_CLML_GRAPH_EXECUTOR)
    set(CLML_PATH ${CMAKE_CURRENT_SOURCE_DIR}/clml)
    # Detect custom CLML path.
    if(NOT TVM_USE_CLML_GRAPH_EXECUTOR STREQUAL "ON")
        set(CLML_PATH ${TVM_USE_CLML_GRAPH_EXECUTOR})
    endif()

    file(GLOB CLML_CONTRIB_SRC CONFIGURE_DEPENDS src/runtime/extra/contrib/clml/*)

    # CMake needs to find clml library, include and support directories
    # in the path specified by CLML_PATH.
    set(CLML_INCLUDE_DIRS ${CLML_PATH}/include ${CLML_PATH})
    find_library(EXTERN_CLML_COMPUTE_LIB
          NAMES OpenCL libOpenCL
          HINTS "${CLML_PATH}" "${CLML_PATH}/lib64" "${CLML_PATH}/lib"
          )
    if(NOT EXTERN_CLML_COMPUTE_LIB)
        string(FIND ${ANDROID_ABI} "64" ARCH_64)
        set(EXTERN_CLML_COMPUTE_LIB "")
        if(ARCH_64 GREATER -1)
            list(APPEND EXTERN_CLML_COMPUTE_LIB ${CLML_PATH}/lib64/libOpenCL.so ${CLML_PATH}/lib64/libOpenCL_system.so)
        else()
            list(APPEND EXTERN_CLML_COMPUTE_LIB ${CLML_PATH}/lib/libOpenCL.so ${CLML_PATH}/lib/libOpenCL_system.so)
        endif()
    endif()
    add_library(tvm_clml_objs OBJECT ${CLML_CONTRIB_SRC})
    tvm_configure_target(tvm_clml_objs)
    target_link_libraries(tvm_clml_objs PRIVATE tvm_runtime_extra_defs)
    target_link_libraries(tvm_runtime_extra PRIVATE tvm_clml_objs ${EXTERN_CLML_COMPUTE_LIB})
    message(STATUS "Build with CLML graph runtime support: "
            ${EXTERN_CLML_COMPUTE_LIB})

    target_include_directories(tvm_clml_objs PRIVATE ${CLML_INCLUDE_DIRS})
    target_compile_definitions(tvm_clml_objs PRIVATE TVM_GRAPH_EXECUTOR_CLML)
    target_compile_definitions(tvm_compiler PRIVATE TVM_GRAPH_EXECUTOR_CLML)

    message(STATUS "Enable OpenCL as fallback to CLML")
    set(TVM_TIRX_USE_OPENCL ${CLML_PATH})
endif()
