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

# HIPBLAS contrib goes into libtvm_runtime_extra.
if(TVM_TIRX_USE_ROCM AND TVM_USE_HIPBLAS)
  message(STATUS "Build with HIPBLAS support")
  file(GLOB HIPBLAS_CONTRIB_SRC CONFIGURE_DEPENDS src/relax/backend/contrib/hipblas/*.cc)
  list(APPEND COMPILER_SRCS ${HIPBLAS_CONTRIB_SRC})
  file(GLOB HIPBLAS_CONTRIB_SRCS CONFIGURE_DEPENDS src/runtime/extra/contrib/hipblas/*.cc)
  add_library(tvm_hipblas_objs OBJECT ${HIPBLAS_CONTRIB_SRCS})
  tvm_configure_target(tvm_hipblas_objs)
  target_link_libraries(tvm_hipblas_objs PRIVATE tvm_runtime_extra_defs)
  target_link_libraries(tvm_runtime_extra PRIVATE tvm_hipblas_objs ${ROCM_HIPBLAS_LIBRARY})
  if(NOT ROCM_HIPBLASLT_LIBRARY STREQUAL "ROCM_HIPBLASLT_LIBRARY-NOTFOUND")
    target_link_libraries(tvm_runtime_extra PRIVATE ${ROCM_HIPBLASLT_LIBRARY})
  endif()
endif(TVM_TIRX_USE_ROCM AND TVM_USE_HIPBLAS)

if(TVM_TIRX_USE_ROCM AND TVM_USE_THRUST)
  message(STATUS "Build with rocThrust support")
  # We need to override CXX to hipcc. This is required by rocthrust
  if(${CMAKE_CXX_COMPILER} MATCHES "hipcc$")
    message(STATUS "Using hipcc compiler to compile rocthrust code.")
  else()
    message(FATAL_ERROR "Set CXX=hipcc to compile rocthrust code.")
  endif()

  find_package(rocprim REQUIRED)
  find_package(rocthrust REQUIRED)
  set_source_files_properties(src/runtime/extra/contrib/thrust/thrust.cu PROPERTIES LANGUAGE CXX)
  add_library(tvm_rocthrust_objs OBJECT src/runtime/extra/contrib/thrust/thrust.cu)
  tvm_configure_target(tvm_rocthrust_objs)
  target_link_libraries(tvm_rocthrust_objs PRIVATE tvm_runtime_extra_defs roc::rocthrust)
  target_link_libraries(tvm_runtime_extra PRIVATE tvm_rocthrust_objs roc::rocthrust)
endif(TVM_TIRX_USE_ROCM AND TVM_USE_THRUST)
