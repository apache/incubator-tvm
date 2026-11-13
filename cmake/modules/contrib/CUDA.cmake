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

# Contrib sources gated by TVM_TIRX_USE_CUDA go into libtvm_runtime_extra.
# See the RuntimeExtra assembly block in CMakeLists.txt.

if(TVM_TIRX_USE_CUDA AND TVM_USE_CUDNN)
  message(STATUS "Build with cuDNN support")
  file(GLOB CUDNN_RELAX_CONTRIB_SRC CONFIGURE_DEPENDS src/relax/backend/contrib/cudnn/*.cc)
  list(APPEND COMPILER_SRCS ${CUDNN_RELAX_CONTRIB_SRC})
  file(GLOB CONTRIB_CUDNN_SRCS CONFIGURE_DEPENDS src/runtime/extra/contrib/cudnn/*.cc)
  add_library(tvm_cudnn_objs OBJECT ${CONTRIB_CUDNN_SRCS})
  tvm_configure_target(tvm_cudnn_objs)
  target_include_directories(tvm_cudnn_objs SYSTEM PRIVATE ${CUDA_CUDNN_INCLUDE_DIRS})
  target_link_libraries(tvm_cudnn_objs PRIVATE tvm_runtime_extra_defs)
  target_link_libraries(tvm_runtime_extra PRIVATE tvm_cudnn_objs ${CUDA_CUDNN_LIBRARY})
endif(TVM_TIRX_USE_CUDA AND TVM_USE_CUDNN)

if(TVM_TIRX_USE_CUDA AND TVM_USE_CUDNN_FRONTEND)
  message(STATUS "Build with cuDNN Frontend support")
  if(IS_DIRECTORY ${TVM_USE_CUDNN_FRONTEND})
    find_file(CUDNN_FRONTEND_HEADER cudnn_frontend.h HINTS ${TVM_USE_CUDNN_FRONTEND}/include)
  else()
    find_file(CUDNN_FRONTEND_HEADER cudnn_frontend.h)
  endif()
  if(NOT CUDNN_FRONTEND_HEADER)
    message(FATAL_ERROR "Cannot find cudnn_frontend.h, please set TVM_USE_CUDNN_FRONTEND to the path of the cuDNN frontend header")
  endif()
  file(GLOB CONTRIB_CUDNN_FRONTEND_SRCS CONFIGURE_DEPENDS src/runtime/extra/contrib/cudnn/cudnn_frontend/*.cc)
  get_filename_component(CUDNN_FRONTEND_INCLUDE_DIR ${CUDNN_FRONTEND_HEADER} DIRECTORY)
  if(TARGET tvm_cudnn_objs)
    target_include_directories(tvm_cudnn_objs SYSTEM PRIVATE ${CUDNN_FRONTEND_INCLUDE_DIR})
    target_compile_definitions(tvm_cudnn_objs PRIVATE TVM_USE_CUDNN_FRONTEND=1)
  endif()
  add_library(tvm_cudnn_frontend_objs OBJECT ${CONTRIB_CUDNN_FRONTEND_SRCS})
  tvm_configure_target(tvm_cudnn_frontend_objs)
  target_include_directories(tvm_cudnn_frontend_objs SYSTEM PRIVATE
    ${CUDNN_FRONTEND_INCLUDE_DIR} ${CUDA_CUDNN_INCLUDE_DIRS})
  target_link_libraries(tvm_cudnn_frontend_objs PRIVATE tvm_runtime_extra_defs)
  target_link_libraries(tvm_runtime_extra PRIVATE tvm_cudnn_frontend_objs ${CUDA_CUDNN_LIBRARY})
endif(TVM_TIRX_USE_CUDA AND TVM_USE_CUDNN_FRONTEND)

if(TVM_TIRX_USE_CUDA AND TVM_USE_CUBLAS)
  message(STATUS "Build with cuBLAS support")
  file(GLOB CUBLAS_CONTRIB_SRC CONFIGURE_DEPENDS src/relax/backend/contrib/cublas/*.cc)
  list(APPEND COMPILER_SRCS ${CUBLAS_CONTRIB_SRC})
  file(GLOB CONTRIB_CUBLAS_SRCS CONFIGURE_DEPENDS src/runtime/extra/contrib/cublas/*.cc)
  add_library(tvm_cublas_objs OBJECT ${CONTRIB_CUBLAS_SRCS})
  tvm_configure_target(tvm_cublas_objs)
  target_link_libraries(tvm_cublas_objs PRIVATE tvm_runtime_extra_defs)
  target_link_libraries(tvm_runtime_extra PRIVATE tvm_cublas_objs ${CUDA_CUBLAS_LIBRARY})
  if(NOT CUDA_CUBLASLT_LIBRARY STREQUAL "CUDA_CUBLASLT_LIBRARY-NOTFOUND")
    target_link_libraries(tvm_runtime_extra PRIVATE ${CUDA_CUBLASLT_LIBRARY})
  endif()
endif(TVM_TIRX_USE_CUDA AND TVM_USE_CUBLAS)

if(TVM_TIRX_USE_CUDA AND TVM_USE_THRUST)
  message(STATUS "Build with Thrust support")
  file(GLOB CONTRIB_THRUST_SRC CONFIGURE_DEPENDS src/runtime/extra/contrib/thrust/*.cu)
  add_library(tvm_thrust_objs OBJECT ${CONTRIB_THRUST_SRC})
  tvm_configure_target(tvm_thrust_objs)
  target_link_libraries(tvm_thrust_objs PRIVATE tvm_runtime_extra_defs)
  target_compile_options(tvm_thrust_objs PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
  if(NOT TVM_USE_THRUST MATCHES ${IS_TRUE_PATTERN})
    set(CCCL_DIR "${TVM_USE_THRUST}")
    find_package(CCCL REQUIRED COMPONENTS Thrust
      HINTS "${TVM_USE_THRUST}"
      PATH_SUFFIXES lib/cmake/cccl lib64/cmake/cccl
      NO_DEFAULT_PATH)
    target_link_libraries(tvm_thrust_objs PRIVATE CCCL::Thrust)
  endif()
  target_link_libraries(tvm_runtime_extra PRIVATE tvm_thrust_objs)
endif(TVM_TIRX_USE_CUDA AND TVM_USE_THRUST)

if(TVM_TIRX_USE_CUDA AND TVM_USE_CURAND)
  message(STATUS "Build with cuRAND support")
  message(STATUS "${CUDA_CURAND_LIBRARY}")
  file(GLOB CONTRIB_CURAND_SRC_CC CONFIGURE_DEPENDS src/runtime/extra/contrib/curand/*.cc)
  file(GLOB CONTRIB_CURAND_SRC_CU CONFIGURE_DEPENDS src/runtime/extra/contrib/curand/*.cu)
  add_library(tvm_curand_objs OBJECT ${CONTRIB_CURAND_SRC_CC} ${CONTRIB_CURAND_SRC_CU})
  tvm_configure_target(tvm_curand_objs)
  target_link_libraries(tvm_curand_objs PRIVATE tvm_runtime_extra_defs)
  target_link_libraries(tvm_runtime_extra PRIVATE tvm_curand_objs ${CUDA_CURAND_LIBRARY})
endif(TVM_TIRX_USE_CUDA AND TVM_USE_CURAND)
