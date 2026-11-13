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

if(IS_DIRECTORY ${TVM_USE_DNNL})
  find_library(EXTERN_LIBRARY_DNNL NAMES dnnl HINTS ${TVM_USE_DNNL}/lib/)
  if(EXTERN_LIBRARY_DNNL STREQUAL "EXTERN_LIBRARY_DNNL-NOTFOUND")
    message(WARNING "Cannot find DNNL library at ${TVM_USE_DNNL}.")
  else()
    file(GLOB DNNL_CONTRIB_SRC CONFIGURE_DEPENDS src/relax/backend/contrib/dnnl/*.cc)
    list(APPEND COMPILER_SRCS ${DNNL_CONTRIB_SRC})

    file(GLOB DNNL_CONTRIB_SRC CONFIGURE_DEPENDS src/runtime/extra/contrib/dnnl/dnnl_json_runtime.cc
                                        src/runtime/extra/contrib/dnnl/dnnl_utils.cc
                                        src/runtime/extra/contrib/dnnl/dnnl.cc
                                        src/runtime/extra/contrib/cblas/dnnl_blas.cc)
    add_library(tvm_dnnl_objs OBJECT ${DNNL_CONTRIB_SRC})
    tvm_configure_target(tvm_dnnl_objs)
    target_compile_definitions(tvm_dnnl_objs PRIVATE USE_JSON_RUNTIME=1)
    target_include_directories(tvm_dnnl_objs SYSTEM PRIVATE ${TVM_USE_DNNL}/include)
    target_link_libraries(tvm_dnnl_objs PRIVATE tvm_runtime_extra_defs)
    target_link_libraries(tvm_runtime_extra PRIVATE tvm_dnnl_objs ${EXTERN_LIBRARY_DNNL})
    message(STATUS "Build with DNNL JSON runtime: " ${EXTERN_LIBRARY_DNNL})
  endif()
elseif((TVM_USE_DNNL STREQUAL "ON") OR (TVM_USE_DNNL STREQUAL "JSON"))
  file(GLOB DNNL_CONTRIB_SRC CONFIGURE_DEPENDS src/relax/backend/contrib/dnnl/*.cc)
  list(APPEND COMPILER_SRCS ${DNNL_CONTRIB_SRC})

  find_library(EXTERN_LIBRARY_DNNL dnnl)
  file(GLOB DNNL_CONTRIB_SRC CONFIGURE_DEPENDS src/runtime/extra/contrib/dnnl/dnnl_json_runtime.cc
                                      src/runtime/extra/contrib/dnnl/dnnl_utils.cc
                                      src/runtime/extra/contrib/dnnl/dnnl.cc
                                      src/runtime/extra/contrib/cblas/dnnl_blas.cc)
  add_library(tvm_dnnl_objs OBJECT ${DNNL_CONTRIB_SRC})
  tvm_configure_target(tvm_dnnl_objs)
  target_compile_definitions(tvm_dnnl_objs PRIVATE USE_JSON_RUNTIME=1)
  target_link_libraries(tvm_dnnl_objs PRIVATE tvm_runtime_extra_defs)
  target_link_libraries(tvm_runtime_extra PRIVATE tvm_dnnl_objs ${EXTERN_LIBRARY_DNNL})
  message(STATUS "Build with DNNL JSON runtime: " ${EXTERN_LIBRARY_DNNL})
elseif(TVM_USE_DNNL STREQUAL "C_SRC")
  find_library(EXTERN_LIBRARY_DNNL dnnl)
  file(GLOB DNNL_CONTRIB_SRC CONFIGURE_DEPENDS src/runtime/extra/contrib/dnnl/dnnl.cc
                                      src/runtime/extra/contrib/dnnl/dnnl_utils.cc
                                      src/runtime/extra/contrib/cblas/dnnl_blas.cc)
  add_library(tvm_dnnl_objs OBJECT ${DNNL_CONTRIB_SRC})
  tvm_configure_target(tvm_dnnl_objs)
  target_link_libraries(tvm_dnnl_objs PRIVATE tvm_runtime_extra_defs)
  target_link_libraries(tvm_runtime_extra PRIVATE tvm_dnnl_objs ${EXTERN_LIBRARY_DNNL})
  message(STATUS "Build with DNNL C source module: " ${EXTERN_LIBRARY_DNNL})
elseif(TVM_USE_DNNL STREQUAL "OFF")
  # pass
else()
  message(FATAL_ERROR "Invalid option: TVM_USE_DNNL=" ${TVM_USE_DNNL})
endif()
